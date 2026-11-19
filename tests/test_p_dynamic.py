# -*- coding: utf-8 -*-
"""Multi-species dynamic FBA (mscommunity.commdynamic).

The toy community is built the way tests/test_l_members.py builds its own (ModelSEED-style member models merged by
build_from_species_models, MSCommunity with drains closed and kappa 1e6). Every flux is forced by stoichiometry, so
the trajectories can be written down by hand:

    memA (c1)   glucose importer; biomass  10 glucose (+ 5 O2) -> 1 gDW + 20 acetate; acetate exporter (export
                only); with fermentation=True also glucose -> 2 acetate (no O2), which max growth never needs
    memB (c2)   acetate importer; biomass  20 acetate (+ 10 O2) -> 1 gDW

    yields      Y_A = 0.1 gDW / mmol glucose, Y_B = 0.05 gDW / mmol acetate, so with saturating uptake
                (Km = 0) mu_A = 0.1 vmax_A,glc and mu_B = 0.05 vmax_B,ac

The real-model anchors (CommPhitting maltose co-culture) run in a fresh interpreter and only with
MSCOMMUNITY_SLOW_TESTS=1, since each takes minutes.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
from cobra import Metabolite, Model, Reaction

from mscommunity.commdynamic import (
    DEFAULT_KINETICS, Trajectory, calibrate_monoculture, dynamic, member_model, simulate_models)
from mscommunity.commhelper import build_from_species_models
from mscommunity.commmembers import member_reactions
from mscommunity.mscommsim import MSCommunity

GLC, AC, O2, BIO = "cpd00027", "cpd00029", "cpd00007", "cpd11416"
Y_A, Y_B = 0.1, 0.05


def _member(mid, kind, o2=False, fermentation=False):
    m = Model(mid)
    mets = {}

    def rxn(rid, stoich, lb=0.0, ub=1000.0):
        r = Reaction(rid, lower_bound=lb, upper_bound=ub)
        for (cid, comp) in stoich:
            key = f"{cid}_{comp}"
            mets.setdefault(key, Metabolite(key, name=key, compartment=comp))
        r.add_metabolites({mets[f"{cid}_{comp}"]: v for (cid, comp), v in stoich.items()})
        return r

    rs = [rxn(f"EX_{GLC}_e0", {(GLC, "e0"): -1}, -10.0), rxn(f"EX_{AC}_e0", {(AC, "e0"): -1})]
    if o2:
        rs += [rxn(f"EX_{O2}_e0", {(O2, "e0"): -1}, -1000.0), rxn("rxn05468_c0", {(O2, "e0"): -1, (O2, "c0"): 1})]
    if kind == "A":
        bio = {(GLC, "c0"): -10, (BIO, "c0"): 1, (AC, "c0"): 20}
        if o2:
            bio[(O2, "c0")] = -5
        rs += [rxn("rxn10000_c0", {(GLC, "e0"): -1, (GLC, "c0"): 1}),
               rxn("rxn10002_c0", {(AC, "c0"): -1, (AC, "e0"): 1}),
               rxn("bio1", bio)]
        if fermentation:
            rs.append(rxn("rxn20000_c0", {(GLC, "c0"): -1, (AC, "c0"): 2}))
    else:
        bio = {(AC, "c0"): -20, (BIO, "c0"): 1}
        if o2:
            bio[(O2, "c0")] = -10
        rs += [rxn("rxn10003_c0", {(AC, "e0"): -1, (AC, "c0"): 1}), rxn("bio1", bio)]
    rs.append(rxn("SK_cpd11416_c0", {(BIO, "c0"): -1}))
    # the ATP hydrolysis MSCommunity looks for (idle: nothing here makes ATP)
    rs.append(rxn("rxn00062_c0", {("cpd00002", "c0"): -1, ("cpd00001", "c0"): -1, ("cpd00008", "c0"): 1,
                                  ("cpd00009", "c0"): 1, ("cpd00067", "c0"): 1}))
    m.add_reactions(rs)
    m.objective = "bio1"
    return m


def _community(o2=False, fermentation=False):
    model = build_from_species_models([_member("memA", "A", o2, fermentation), _member("memB", "B", o2)],
                                      abundances={"memA": 0.5, "memB": 0.5})
    return MSCommunity(model=model, ids=["memA", "memB"], kinetic_coeff=1e6, close_member_drains=True)


def _model_state(model):
    return ({r.id: tuple(r.bounds) for r in model.reactions}, sorted(c.name for c in model.constraints),
            sorted(v.name for v in model.variables), str(model.objective.expression), model.objective.direction)


# --------------------------------------------------------------------------- member_model
def test_member_model_is_the_members_reactions_plus_its_exchanges():
    comm = _community()
    cmodel = comm.util.model
    mm = member_model(comm, "memA")
    own = {r.id for r in member_reactions(comm, "memA")}
    ids = {r.id for r in mm.reactions}
    assert own <= ids
    assert ids - own == {f"EX_{GLC}_e0", f"EX_{AC}_e0"}
    for rid in ids - own:                                     # the community's medium bounds
        assert mm.reactions.get_by_id(rid).bounds == cmodel.reactions.get_by_id(rid).bounds
    note = mm.notes["member_model"]
    assert (note["biomass"], note["drain"]) == ("bio2", "SK_cpd11416_c1")
    assert mm.reactions.get_by_id("SK_cpd11416_c1").bounds == (0.0, 1000.0)       # opened (closed in comm)
    assert cmodel.reactions.get_by_id("SK_cpd11416_c1").bounds == (0, 0)
    assert mm.solver.interface is cmodel.solver.interface
    mm.reactions.get_by_id(f"EX_{GLC}_e0").lower_bound = -1.0
    assert mm.slim_optimize() == pytest.approx(Y_A)            # 1 mmol glucose -> 0.1 gDW
    mB = member_model(comm, comm.members.get_by_id("memB"))
    assert f"EX_{GLC}_e0" not in mB.reactions and f"EX_{AC}_e0" in mB.reactions

    # a repair written in community ids applies unchanged; a callable's return value is kept
    fixed = member_model(comm, 1, repair={"rxn10002_c1": (0.0, 5.0), "rxn10003_c2": (0.0, 0.0)})
    assert fixed.reactions.get_by_id("rxn10002_c1").bounds == (0.0, 5.0)
    assert fixed.notes["member_model"]["repair"] == {"applied": {"rxn10002_c1": (0.0, 5.0)},
                                                     "absent": ["rxn10003_c2"]}
    tagged = member_model(comm, "memA", repair=lambda m: m.reactions.get_by_id("rxn10000_c1").id)
    assert tagged.notes["member_model"]["repair"] == "rxn10000_c1"
    lean = member_model(comm, "memA", subset=["rxn10000_c1"])
    assert {r.id for r in lean.reactions} == {"rxn10000_c1", "bio2", "SK_cpd11416_c1", f"EX_{GLC}_e0"}
    with pytest.raises(KeyError):
        member_model(comm, "memA", subset=["rxn10003_c2"])


# --------------------------------------------------------------------------- members form
def test_saturating_phase_follows_the_analytic_exponentials():
    """Km = 0: caps are vmax until a pool runs out, so mu_A = 0.1 * 3, mu_B = 0.05 * 4 (times 1 - 1e-7: the pFBA
    stage holds growth at that fraction of the optimum) and the integrator's biomass integrals make every step
    exact."""
    comm = _community()
    X0, S0 = {"memA": 0.01, "memB": 0.02}, {GLC: 10.0, AC: 5.0}
    kin = {"memA": {GLC: (3.0, 0.0)}, "memB": {f"EX_{AC}_e0": {"vmax": 4.0, "Km": 0.0}}}
    traj = dynamic(comm, 5.0, 0.5, X0, S0, kinetics=kin)
    assert isinstance(traj, Trajectory) and traj.form == "members" and traj.tiebreak == {"stage": "pfba"}
    t = traj.times
    assert t[0] == 0.0 and t[-1] == 5.0 and traj.stats["rejected_steps"] == 0
    muA, muB = 0.3 * (1 - 1e-7), 0.2 * (1 - 1e-7)
    XA, XB = 0.01 * np.exp(muA * t), 0.02 * np.exp(muB * t)
    np.testing.assert_allclose(traj.X["memA"], XA, rtol=1e-9)
    np.testing.assert_allclose(traj.X["memB"], XB, rtol=1e-9)
    np.testing.assert_allclose(traj.mu["memA"], muA, rtol=1e-9)
    np.testing.assert_allclose(traj.mu["memB"], muB, rtol=1e-9)
    np.testing.assert_allclose(traj.concentrations[GLC], 10.0 - (XA - 0.01) / Y_A, rtol=1e-9)
    np.testing.assert_allclose(traj.concentrations[AC], 5.0 + 20 * (XA - 0.01) - (XB - 0.02) / Y_B, rtol=1e-9)
    np.testing.assert_allclose(traj.composition["memA"], XA / (XA + XB), rtol=1e-9)
    # the ledger: who took and released what
    led = traj.ledger()
    assert led["memA"][GLC]["uptake"] == pytest.approx((XA[-1] - 0.01) / Y_A, rel=1e-9)
    assert led["memA"][AC]["secretion"] == pytest.approx(20 * (XA[-1] - 0.01), rel=1e-9)
    assert led["memB"][AC]["uptake"] == pytest.approx((XB[-1] - 0.02) / Y_B, rel=1e-9)
    assert GLC not in led["memB"] and led["memA"][AC]["uptake"] == 0.0
    assert traj.growth_rate("memA", (1.0, 4.0)) == pytest.approx(muA, rel=1e-9)
    assert traj.at(2.25)["X"]["memA"] == pytest.approx(np.interp(2.25, t, XA))
    assert traj.settings["kinetics"]["memB"] == {AC: (4.0, 0.0)}
    assert set(traj.settings["tracked"]) == {GLC, AC} and traj.otr is None
    json.dumps(traj.as_dict())
    assert list(traj.to_frame().columns[:3]) == ["X_memA", "a_memA", "mu_memA"]


def test_cross_feeding_to_exhaustion_balances_and_never_empties_a_pool():
    """B starts without acetate and lives on what A releases. Everything A takes ends up as its biomass and
    acetate, B grows on that acetate, and no pool is ever set to 0."""
    comm = _community()
    X0, S0 = {"memA": 0.01, "memB": 0.01}, {GLC: 2.0}
    kin = {"memA": {GLC: (3.0, 0.01)}, "memB": {AC: (4.0, 0.05)}}
    traj = dynamic(comm, 40.0, 0.5, X0, S0, kinetics=kin)
    glc, ac = traj.concentrations[GLC], traj.concentrations[AC]
    assert traj.mu["memB"][0] == 0.0 and traj.status[0]["memB"] == "idle"
    assert np.all(glc > 0) and np.all(ac[1:] > 0) and ac[0] == 0.0
    assert glc[-1] < 1e-6 and ac[-1] < 1e-6                      # both exhausted, neither reset to 0
    led = traj.ledger()
    up_glc, se_ac, up_ac = led["memA"][GLC]["uptake"], led["memA"][AC]["secretion"], led["memB"][AC]["uptake"]
    assert glc[-1] == pytest.approx(2.0 - up_glc, abs=1e-12)     # ledger and pools balance exactly
    assert ac[-1] == pytest.approx(se_ac - up_ac, abs=1e-12)
    lost = traj.stats["patankar_mismatch_mM"]                    # LP uptake not realised in the pool
    assert se_ac == pytest.approx(2 * (up_glc + lost.get(GLC, 0.0)), rel=1e-9)   # 20 acetate per 10 glucose
    assert traj.X["memA"][-1] - 0.01 == pytest.approx(Y_A * (up_glc + lost.get(GLC, 0.0)), rel=1e-9)
    assert traj.X["memB"][-1] - 0.01 == pytest.approx(Y_B * (up_ac + lost.get(AC, 0.0)), rel=1e-9)
    assert abs(sum(lost.values())) < 1e-5
    assert traj.X["memA"][-1] == pytest.approx(0.01 + Y_A * 2.0, rel=1e-5)
    assert traj.X["memB"][-1] == pytest.approx(0.01 + Y_B * 4.0, rel=1e-4)
    assert traj.stats["rejected_steps"] > 0                      # step control near each depletion
    # B's rate follows the acetate A releases: zero at first, saturating later
    assert traj.mu["memB"].max() == pytest.approx(0.05 * 4.0 * traj.concentrations[AC].max()
                                                  / (0.05 + traj.concentrations[AC].max()), rel=1e-6)
    # record_every keeps the ledger complete
    sparse = dynamic(comm, 40.0, 0.5, X0, S0, kinetics=kin, record_every=7)
    assert len(sparse.times) < len(traj.times) and sparse.times[-1] == 40.0
    assert sparse.ledger()["memB"][AC]["uptake"] == pytest.approx(up_ac, rel=1e-12)


def test_community_strict_matches_the_members_form_on_disjoint_members():
    """With every e0 compound capped per member at the current pool, a community of members that share no
    metabolism is DMMM in one LP: same trajectory. The community model is left as it was."""
    comm = _community()
    before = _model_state(comm.util.model)
    X0, S0 = {"memA": 0.01, "memB": 0.01}, {GLC: 2.0}
    kin = {"memA": {GLC: (3.0, 0.01)}, "memB": {AC: (4.0, 0.05)}}
    a = dynamic(comm, 30.0, 0.5, X0, S0, kinetics=kin)
    for basis in ("gross", "net"):
        b = dynamic(comm, 30.0, 0.5, X0, S0, kinetics=kin, form="community_strict", cap_basis=basis)
        assert b.form == "community_strict" and b.settings["cap_basis"] == basis
        assert b.ledger_rows == ("memA", "memB", "shared")
        grid = np.linspace(0.0, 30.0, 121)
        sa, sb = a.at(grid), b.at(grid)
        for m in ("memA", "memB"):
            np.testing.assert_allclose(sb["X"][m], sa["X"][m], rtol=1e-5)
        np.testing.assert_allclose(sb["S"][AC], sa["S"][AC], rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(b.mu["memA"][:5], a.mu["memA"][:5], rtol=1e-6)
        assert b.ledger()["memB"][AC]["uptake"] == pytest.approx(a.ledger()["memB"][AC]["uptake"], rel=1e-5)
        assert not b.ledger().get("shared")
        assert _model_state(comm.util.model) == before
        assert comm._member_caps == {} if hasattr(comm, "_member_caps") else True


def test_o2_sharing_rules_give_the_hand_computed_rates():
    """X_A = X_B = 0.1 gDW/L, OTR 0.07 mmol/L/h. Unlimited demand: A 5 * 0.3 = 1.5, B 10 * 0.2 = 2 mmol/gDW/h,
    0.35 mmol/L/h in all.
        demand   each member gets 0.07 / 0.35 = 0.2 of its demand: mu_A 0.06, mu_B 0.04
        equal    0.07 / 0.2 = 0.35 mmol/gDW/h each: mu_A 0.35 / 5 = 0.07, mu_B 0.35 / 10 = 0.035
        joint    one LP maximises a_A mu_A + a_B mu_B with 5 a_A mu_A + 10 a_B mu_B <= 0.35: A's growth costs
                 half the O2, so mu_A = 0.35 / 5 / 0.5 = 0.14 and mu_B = 0"""
    comm = _community(o2=True)
    X0, S0 = {"memA": 0.1, "memB": 0.1}, {GLC: 10.0, AC: 10.0}
    kin = {"memA": {GLC: (3.0, 0.0)}, "memB": {AC: (4.0, 0.0)}}
    run = lambda **kw: dynamic(comm, 0.01, 0.01, X0, S0, kinetics=kin, otr=0.07, **kw)  # noqa: E731
    free = dynamic(comm, 0.01, 0.01, X0, S0, kinetics=kin)
    assert (free.mu["memA"][0], free.mu["memB"][0]) == (pytest.approx(0.3), pytest.approx(0.2))
    assert free.o2_rule is None and O2 not in free.ledger_compounds
    demand = run()
    assert demand.o2_rule == "demand" and demand.o2_limited[0]
    assert (demand.mu["memA"][0], demand.mu["memB"][0]) == (pytest.approx(0.06), pytest.approx(0.04))
    o2_used = demand.ledger()["memA"][O2]["uptake"] + demand.ledger()["memB"][O2]["uptake"]
    assert o2_used == pytest.approx(0.07 * 0.01, rel=1e-3)
    equal = run(o2_rule="equal")
    assert (equal.mu["memA"][0], equal.mu["memB"][0]) == (pytest.approx(0.07), pytest.approx(0.035))
    joint = run(form="community_strict")
    assert joint.o2_rule == "joint" and joint.o2_limited[0]
    assert (joint.mu["memA"][0], joint.mu["memB"][0]) == (pytest.approx(0.14), pytest.approx(0.0, abs=1e-6))
    cequal = run(form="community_strict", o2_rule="equal")
    assert (cequal.mu["memA"][0], cequal.mu["memB"][0]) == (pytest.approx(0.07), pytest.approx(0.035))
    with pytest.raises(ValueError, match="not defined for form"):
        run(form="community_strict", o2_rule="demand")
    with pytest.raises(ValueError, match="not defined for form"):
        run(o2_rule="joint")
    # dissolved O2 as a pool: kLa (C* - C) in, per-member kinetics out; it settles where transfer meets demand
    kla = dynamic(comm, 2.0, 0.05, X0, S0, kinetics={"memA": {GLC: (3.0, 0.0), O2: (10.0, 0.001)},
                                                     "memB": {AC: (4.0, 0.0), O2: (10.0, 0.001)}},
                  otr=(5.0, 0.2))
    assert kla.o2_rule == "kinetics" and O2 in kla.concentrations
    c = kla.concentrations[O2]
    assert c[0] == 0.2 and np.all(c > 0) and np.all(np.diff(c[-10:]) < 0)      # tracking demand, no oscillation
    demand = 5 * kla.mu["memA"][-1] * kla.X["memA"][-1] + 10 * kla.mu["memB"][-1] * kla.X["memB"][-1]
    assert c[-1] == pytest.approx(0.2 - demand / 5.0, abs=0.01)                  # kLa (C* - C) = demand


def test_tiebreak_on_the_optimal_face_changes_the_by_product_split_and_is_recorded():
    """O2-limited, A's growth needs 10 glucose per gDW but it may take 3 mmol/gDW/h: the 3 - 10 mu_A left over
    can be fermented to acetate at no cost to growth. pFBA ferments none; max acetate ferments all of it."""
    comm = _community(o2=True, fermentation=True)
    X0, S0 = {"memA": 0.1, "memB": 0.1}, {GLC: 10.0, AC: 10.0}
    kin = {"memA": {GLC: (3.0, 0.0)}, "memB": {AC: (4.0, 0.0)}}
    base = dynamic(comm, 0.01, 0.01, X0, S0, kinetics=kin, otr=0.07, o2_rule="equal")
    tb = dynamic(comm, 0.01, 0.01, X0, S0, kinetics=kin, otr=0.07, o2_rule="equal",
                 tiebreak={"memA": (f"EX_{AC}_e0", "max")})
    assert tb.tiebreak == {"stage": "secondary+pfba", "secondary": {"memA": (f"EX_{AC}_e0", "max")}}
    assert tb.mu["memA"][0] == pytest.approx(base.mu["memA"][0]) == pytest.approx(0.07)
    G = base.ledger()["memA"][GLC]["uptake"] / (10 * 0.07)        # A's biomass integral over the step
    assert base.ledger()["memA"][AC]["secretion"] == pytest.approx(20 * 0.07 * G, rel=1e-6)
    assert tb.ledger()["memA"][AC]["secretion"] == pytest.approx((20 * 0.07 + 2 * (3.0 - 10 * 0.07)) * G, rel=1e-6)
    vertex = dynamic(comm, 0.01, 0.01, X0, S0, kinetics=kin, otr=0.07, o2_rule="equal", tiebreak="vertex")
    assert vertex.tiebreak == {"stage": "vertex"} and any("vertex" in w for w in vertex.warnings)
    with pytest.raises(ValueError, match="not valid for form"):
        dynamic(comm, 0.01, 0.01, X0, S0, kinetics=kin, form="community_strict", tiebreak={"memA": ("bio2", "max")})


# --------------------------------------------------------------------------- chemostat
def test_chemostat_settles_on_the_monod_steady_state():
    """A alone, D = 0.1 1/h, feed 5 mM glucose, vmax 3, Km 0.5: mu = 0.3 S / (0.5 + S) = D at S* = 0.25 mM, and
    X* = Y_A (5 - S*) = 0.475 gDW/L. Positivity-preserving steps approach it without a period-2 orbit."""
    comm = _community()
    mA = member_model(comm, "memA")
    traj = simulate_models({"memA": mA}, 150.0, 0.5, {"memA": 0.01}, {GLC: 5.0},
                           kinetics={"memA": {GLC: (3.0, 0.5)}}, feed={GLC: 5.0}, D=0.1)
    S, X = traj.concentrations[GLC], traj.X["memA"]
    assert S[-1] == pytest.approx(0.25, rel=1e-3) and X[-1] == pytest.approx(0.475, rel=1e-3)
    assert np.all(S > 0) and np.all(np.abs(np.diff(S[-20:])) < 1e-4)
    led = traj.ledger()
    balance = 5.0 + traj.inflow.sum(axis=0) - traj.outflow.sum(axis=0)
    j = traj.ledger_compounds.index(GLC)
    assert S[-1] == pytest.approx(balance[j] - led["memA"][GLC]["uptake"], abs=1e-9)
    assert mA.reactions.get_by_id(f"EX_{GLC}_e0").bounds == (-10.0, 1000.0)   # copy=True: model untouched
    with pytest.raises(ValueError, match="feed needs a dilution rate"):
        simulate_models({"memA": mA}, 1.0, 0.5, {"memA": 0.01}, {GLC: 5.0}, kinetics={}, feed={GLC: 5.0})


# --------------------------------------------------------------------------- calibration
def test_calibrate_monoculture_recovers_vmax_from_a_synthetic_time_course():
    comm = _community()
    mA = member_model(comm, "memA")
    truth = simulate_models({"memA": mA}, 20.0, 0.5, {"memA": 0.01}, {GLC: 3.0}, kinetics={"memA": {GLC: (2.0, 0.05)}})
    t = np.linspace(0.0, 20.0, 21)
    data = {"t": t, "X": np.interp(t, truth.times, truth.X["memA"]),
            "S": np.interp(t, truth.times, truth.concentrations[GLC])}
    fit = calibrate_monoculture(mA, data, params=("vmax",), substrate=GLC, S0={GLC: 3.0},
                                kinetics={GLC: (0.7, 0.05)}, dt=0.5)
    assert fit["yield_gDW_per_mmol"] == pytest.approx(Y_A)
    assert fit["initial"]["vmax"] == pytest.approx(fit["mu_obs"] / Y_A)   # the Louca-type starting value
    assert fit["params"]["vmax"] == pytest.approx(2.0, rel=1e-3) and fit["success"]
    assert fit["kinetics"] == {GLC: (pytest.approx(2.0, rel=1e-3), 0.05)} and fit["rmse_lnX"] < 1e-3


# --------------------------------------------------------------------------- validation
def test_arguments_are_checked():
    comm = _community()
    X0, S0, kin = {"memA": 0.01, "memB": 0.01}, {GLC: 1.0}, {}
    with pytest.raises(ValueError, match="form must be"):
        dynamic(comm, 1.0, 0.5, X0, S0, kinetics=kin, form="joint")
    with pytest.raises(KeyError, match="X0 has no biomass"):
        dynamic(comm, 1.0, 0.5, {"memA": 0.01}, S0, kinetics=kin)
    with pytest.raises(KeyError, match="not a member"):
        dynamic(comm, 1.0, 0.5, X0, S0, kinetics={"memC": {GLC: (1, 1)}})
    with pytest.raises(ValueError, match="cap_basis"):
        dynamic(comm, 1.0, 0.5, X0, S0, kinetics=kin, form="community_strict", cap_basis="both")
    with pytest.raises(ValueError, match="vmax and Km"):
        dynamic(comm, 1.0, 0.5, X0, S0, kinetics={"memA": {GLC: (-1.0, 0.1)}})
    assert DEFAULT_KINETICS == (10.0, 0.01)


# --------------------------------------------------------------------------- the CommPhitting maltose anchors
CP = "/home/freiburger/Documents/CommPhitting"
MM = "/home/freiburger/Documents/3H11_R12_SynCom/Metabolic_modeling"
DATA = [f"{CP}/ecoli.json", f"{CP}/pf.json", f"{MM}/data/multisystem/commphitting_specs.json",
        f"{MM}/data/multisystem3/commphitting_data/measurements.json",
        f"{MM}/data/multisystem3/commphitting_data/growth_maltose_timeseries.csv",
        f"{MM}/data/multisystem3/systems/commphitting_maltose/gate.json"]
SLOW = os.environ.get("MSCOMMUNITY_SLOW_TESTS") == "1"


def _fresh(check, timeout=3600):
    tests = os.path.dirname(os.path.abspath(__file__))
    code = f"import sys; sys.path[:0] = [{tests!r}, {os.path.dirname(tests)!r}]; import test_p_dynamic as t; t.{check}()"
    try:
        run = subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(tests), capture_output=True, text=True,
                             timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{check} took over {timeout} s in a fresh interpreter")
    assert run.returncode == 0, run.stderr[-6000:]
    assert "ANCHOR_OK" in run.stdout, run.stdout[-3000:]
    return run.stdout


@pytest.mark.skipif(not (SLOW and all(os.path.exists(p) for p in DATA)),
                    reason="set MSCOMMUNITY_SLOW_TESTS=1 (minutes per run) with the CommPhitting data on this machine")
def test_commphitting_maltose_members_form_otr2():
    _fresh("_anchor_members")


@pytest.mark.skipif(not (SLOW and all(os.path.exists(p) for p in DATA)),
                    reason="set MSCOMMUNITY_SLOW_TESTS=1 (minutes per run) with the CommPhitting data on this machine")
def test_commphitting_maltose_community_strict_otr1_orient_both():
    _fresh("_anchor_community_strict")


# the inputs of abund_dynamic_lib.system_spec('maltose', overrides={'OTR': ..., 'Km_maltose': 0.001}), as recorded in
# data/multisystem3/abundance/dynamic/runs/maltose__I__OTR-2__Kmmal-0.001.json -> spec
MALT, NH4 = "cpd00179", "cpd00013"
SPEC = {"k": {"ecoli": 0.931367081598368, "pf": 1.2425276240437015},
        "X0": {"ecoli": 0.01682874135730523, "pf": 0.019870441602830267},
        "S0": {MALT: 5.0, NH4: 18.695083193120208},
        "kinetics": {"ecoli": {AC: (2.055855884765465, 0.01), MALT: (1.3938509390541232, 0.001)},
                     "pf": {AC: (8.156673338605204, 0.01)}},
        "t_end": 27.157289444444444, "dt": 0.05}


def _maltose_community(orient_both=False):
    """build_from_species_models + MSCommunity as multisystem3_sys_commphitting_maltose_lib.build does (a_ec 0.8704,
    kappa 1e6 rows removed, drains closed), the commphitting_specs.json maltose medium (every EX_ export-only, then
    the medium uptakes; carbon closed, set by the kinetics), P. fluorescens' two maltose transports closed, and
    optionally the 65 orient_both bounds of systems/commphitting_maltose/gate.json."""
    import contextlib
    import io

    import cobra
    models = []
    for name in ("ecoli", "pf"):
        m = cobra.io.load_json_model(f"{CP}/{name}.json")
        m.id = name
        models.append(m)
    a_ec = 0.8704
    with contextlib.redirect_stdout(io.StringIO()):
        merged = build_from_species_models(models, model_id="commphitting_maltose", name="commphitting_maltose",
                                           abundances={"ecoli": a_ec, "pf": 1 - a_ec}, commkinetics=False)
        abund = {"ecoli": {"abundance": a_ec, "biomass_compound": merged.metabolites.get_by_id("cpd11416_c1")},
                 "pf": {"abundance": 1 - a_ec, "biomass_compound": merged.metabolites.get_by_id("cpd11416_c2")}}
        comm = MSCommunity(model=merged, abundances=abund, ID="commphitting_maltose", kinetic_coeff=1e6,
                           close_member_drains=True)
    model = comm.util.model
    model.remove_cons_vars([c for c in model.constraints if c.name.endswith("_commKin")])
    medium = json.load(open(DATA[2]))["conditions"]["maltose"]["medium"]
    for r in model.reactions:
        if r.id.startswith("EX_"):
            r.bounds = (0.0, 1000.0)
    for ex, entry in medium.items():
        model.reactions.get_by_id(ex).lower_bound = float(entry["uptake_bound_mmol_gDW_h"])
    for c in (MALT, "cpd00136", AC):
        if f"EX_{c}_e0" in model.reactions:
            model.reactions.get_by_id(f"EX_{c}_e0").lower_bound = 0.0
    for rid in ("rxn05170_c2", "rxn05607_c2"):
        model.reactions.get_by_id(rid).bounds = (0.0, 0.0)
    if orient_both:
        for rid, (_, new) in json.load(open(DATA[5]))["forms"]["orient_both"]["changed"].items():
            model.reactions.get_by_id(rid).bounds = tuple(new)
    return comm


def _score(traj):
    """abund_dynamic_lib.score's composition snapshots and window rates: X interpolated at the measured read
    times, OD_m = X_m / k_m, a = OD_ec / OD; slopes of ln OD over the reads in a window."""
    import pandas as pd
    meas = json.load(open(DATA[3]))["conditions"]["maltose"]
    t = pd.read_csv(DATA[4])["time_h"].to_numpy()
    od = {m: np.interp(t, traj.times, traj.X[m]) / SPEC["k"][m] for m in ("ecoli", "pf")}
    a = od["ecoli"] / (od["ecoli"] + od["pf"])
    snaps = {k: float(np.interp(s["time_h"], t, a)) for k, s in meas["composition"]["snapshots"].items() if k != "t0"}

    def slope(y, w):
        msk = (t >= w[0]) & (t <= w[1]) & (y > 0)
        return float(np.polyfit(t[msk], np.log(y[msk]), 1)[0])
    tw, pfw = meas["total_growth_rate_from_OD"]["window_h"], meas["member_growth_rates"]["pf"]["window_h"]
    return {"snapshots": snaps, "R_pf": slope(od["ecoli"], pfw) - slope(od["pf"], pfw),
            "R_tot": slope(od["ecoli"], tw) - slope(od["pf"], tw), "mu_pf_pf_window": slope(od["pf"], pfw),
            "final_OD": float((od["ecoli"] + od["pf"])[-1])}


def _report(tag, traj, score):
    print(json.dumps({"tag": tag, "score": score, "stats": traj.stats, "warnings": traj.warnings,
                      "settings": {k: traj.settings[k] for k in ("tracked", "ample", "cap_basis") if k in traj.settings}},
                     default=str), flush=True)


def _anchor_members():
    """Form I (DMMM on the members, pFBA, demand-proportional O2) at OTR 2 mmol/L/h, Km maltose 0.001 mM:
    abund_dynamic_lib gave a_ec 0.9517 / 0.9813 / 0.9794 (mid-exponential / end of window / final read) and
    R_pf -0.0359 1/h (runs/maltose__I__OTR-2__Kmmal-0.001.json; dt 0.02 gives -0.0355, the verifier's own
    integrator -0.0364). commdynamic: 0.9517 / 0.9813 / 0.9794, R_pf -0.0351 (803 steps, about 2.5 min); the
    member models extracted from the community are the prototype's isolates exactly (reactions and bounds)."""
    comm = _maltose_community()
    traj = dynamic(comm, SPEC["t_end"], SPEC["dt"], SPEC["X0"], SPEC["S0"], kinetics=SPEC["kinetics"], otr=2.0)
    score = _score(traj)
    _report("members_otr2", traj, score)
    assert traj.o2_rule == "demand" and traj.tiebreak == {"stage": "pfba"}
    snaps = score["snapshots"]
    assert snaps["mid_exponential_of_total_OD"] == pytest.approx(0.9517, abs=1e-3)
    assert snaps["end_of_total_OD_exponential_window"] == pytest.approx(0.9813, abs=1e-3)
    assert snaps["final_read"] == pytest.approx(0.9794, abs=1e-3)
    assert score["R_pf"] == pytest.approx(-0.0359, abs=5e-3)
    print("ANCHOR_OK", flush=True)


def _anchor_community_strict():
    """C-strict on the orient_both community at OTR 1 (joint O2): the author's CommunityStepper gave a_ec 0.9516 /
    0.9793 / 0.9663 and R_pf -0.2113, the verifier's independent stepper 0.952 / 0.980 / 0.968 and -0.2032
    (_verify/m-dynamic/r3/x5_authorstepper_Cstrict_OTR1_orient_both.json, x2b_ob_otr1.json). Both use net
    per-member rows; this runs cap_basis="net". commdynamic: 0.9516 / 0.9793 / 0.9655, R_pf -0.218 (gross caps
    give the same to 4 decimals; about 7 min). The "R_pf about -0.14" sometimes quoted for this cell is the
    nah_closed network at OTR 1.5 (-0.138 / -0.144), not orient_both at OTR 1."""
    comm = _maltose_community(orient_both=True)
    traj = dynamic(comm, SPEC["t_end"], SPEC["dt"], SPEC["X0"], SPEC["S0"], kinetics=SPEC["kinetics"], otr=1.0,
                   form="community_strict", cap_basis="net")
    score = _score(traj)
    _report("community_strict_otr1_orient_both", traj, score)
    assert traj.o2_rule == "joint"
    snaps = score["snapshots"]
    assert snaps["mid_exponential_of_total_OD"] == pytest.approx(0.952, abs=2e-3)
    assert snaps["end_of_total_OD_exponential_window"] == pytest.approx(0.980, abs=3e-3)
    assert snaps["final_read"] == pytest.approx(0.967, abs=5e-3)
    assert -0.23 < score["R_pf"] < -0.19
    print("ANCHOR_OK", flush=True)
