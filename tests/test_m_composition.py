# -*- coding: utf-8 -*-
"""Composition infrastructure (mscommunity.commcomposition): steadycom, calibrate_kappa,
kappa_sat_coupled, composition_profile.

The toy is the two-member community of 3H11_R12_SynCom/Metabolic_modeling/
abund_fixed_composition_toylib.py, rebuilt in the MSCommunity layout (ModelSEED-style member models
merged by build_from_species_models, wrapped in MSCommunity with drains closed and kappa 1e6):

    shared pool       S = glucose cpd00027, P = acetate cpd00029 (by-product), T = maltose cpd00179
    member m          tS_m S_e -> S_m; resp_m S_m -> e_m ATP (A: e = 3, B: e = 2.985); ferm_m S_m -> ATP + 2 P_m;
                      tP_m P_m <-> P_e; Presp_m P_m -> (e-1)/2 ATP; Pcarb_m 2 P_m + ATP -> S_m;
                      NGAM rxn00062 ATP ->; biomass 10 S_m + 20 ATP -> 1 gDW; drain SK_cpd11416
    niche             A takes S only, B takes T only (T_e -> 2 S_B), no by-product exchange
    hydrolase         A takes S; B takes S and opens T into the pool through its periplasm
                      (rxn30000_p2 T_e -> 2 S_p2, rxn30001_p2 S_p2 -> S_e), cf. B. thetaiotaomicron's dextran

Yields: Y_A = 1 / (10 + 20/3) = 0.06, Y_B = 1 / (10 + 20/2.985) gDW per mmol S. The real-model
anchors (SYNTHESIS.md 4.1-4.2) run in a fresh interpreter, one per system, and skip when the
models or the campaign libraries are absent; they also need MSCOMMUNITY_SLOW_TESTS=1 (each takes
0.5-3 min)."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import pytest
from cobra import Metabolite, Model, Reaction

from mscommunity.commcomposition import (
    calibrate_kappa, composition_profile, kappa_sat_coupled, steadycom)
from mscommunity.commmembers import cap_member_uptake, member_caps, set_member_maintenance
from mscommunity.commhelper import build_from_species_models
from mscommunity.mscommsim import MSCommunity

S, P, T, BIO = "cpd00027", "cpd00029", "cpd00179", "cpd11416"
ATP, H2O, ADP, PI, H = "cpd00002", "cpd00001", "cpd00008", "cpd00009", "cpd00067"
E_RESP = {"A": 3.0, "B": 2.985}
BIO_S, BIO_E = 10.0, 20.0


def yield_on_S(mid):
    return 1.0 / (BIO_S + BIO_E / E_RESP[mid])


def _toy_member(mid, *, takes_S=True, p_open=True, hydrolase=False, niche_T=False):
    m = Model(mid)
    mets = {}

    def rxn(rid, stoich, lb=0.0, ub=1000.0):
        r = Reaction(rid, lower_bound=lb, upper_bound=ub)
        for (cid, comp), v in stoich.items():
            key = f"{cid}_{comp}"
            mets.setdefault(key, Metabolite(key, name=key, compartment=comp))
        r.add_metabolites({mets[f"{cid}_{comp}"]: v for (cid, comp), v in stoich.items()})
        return r

    atp = lambda n: {(ATP, "c0"): n, (H2O, "c0"): n, (ADP, "c0"): -n, (PI, "c0"): -n, (H, "c0"): -n}  # noqa: E731
    e = E_RESP[mid]
    rs = [rxn(f"EX_{S}_e0", {(S, "e0"): -1}, -1000.0), rxn(f"EX_{P}_e0", {(P, "e0"): -1}, -1000.0),
          rxn(f"EX_{T}_e0", {(T, "e0"): -1}, -1000.0),
          rxn("rxn10000_c0", {(S, "e0"): -1, (S, "c0"): 1}, 0.0, 1000.0 if takes_S else 0.0),
          rxn("rxn20001_c0", {(S, "c0"): -1, **atp(e)}),
          rxn("rxn20002_c0", {(S, "c0"): -1, (P, "c0"): 2, **atp(1)}),
          rxn("rxn10002_c0", {(P, "c0"): -1, (P, "e0"): 1}, -1000.0 if p_open else 0.0, 1000.0 if p_open else 0.0),
          rxn("rxn20003_c0", {(P, "c0"): -1, **atp((e - 1.0) / 2.0)}),
          rxn("rxn20004_c0", {(P, "c0"): -2, (S, "c0"): 1, **atp(-1)}),
          rxn("rxn00062_c0", atp(-1)),
          rxn("bio1", {(S, "c0"): -BIO_S, (BIO, "c0"): 1, **atp(-BIO_E)}),
          rxn("SK_cpd11416_c0", {(BIO, "c0"): -1})]
    if hydrolase:
        rs += [rxn("rxn30000_p0", {(T, "e0"): -1, (S, "p0"): 2}), rxn("rxn30001_p0", {(S, "p0"): -1, (S, "e0"): 1})]
    if niche_T:
        rs += [rxn("rxn30002_c0", {(T, "e0"): -1, (S, "c0"): 2})]
    m.add_reactions(rs)
    m.objective = "bio1"
    return m


def toy(variant, *, q_S=5.0, q_T=0.0, a=(0.5, 0.5), kinetic_coeff=None):
    """The toy community in the MSCommunity layout (members A = index 1, B = index 2). Kinetic rows
    removed unless kinetic_coeff is given (as the campaign builds): even at kappa 1e6 they hold a
    member of zero abundance at zero flux, which would hide the relabelling the toy is made to show."""
    if variant == "niche":
        members = [_toy_member("A", p_open=False), _toy_member("B", takes_S=False, p_open=False, niche_T=True)]
    elif variant == "hydrolase":
        members = [_toy_member("A"), _toy_member("B", hydrolase=True)]
    elif variant == "shared":
        members = [_toy_member("A"), _toy_member("B")]
    else:
        raise ValueError(variant)
    model = build_from_species_models(members, abundances={"A": a[0], "B": a[1]})
    comm = MSCommunity(model=model, ids=["A", "B"], kinetic_coeff=kinetic_coeff or 1e6, close_member_drains=True)
    m = comm.util.model
    if kinetic_coeff is None:
        m.remove_cons_vars([c for c in m.constraints if c.name.endswith("_commKin")])
    m.solver = "gurobi" if "gurobi" in _solvers() else "glpk"
    if m.solver.interface.__name__.endswith("gurobi_interface"):
        m.solver.problem.Params.Threads = 1
        m.solver.problem.Params.OutputFlag = 0
    m.reactions.get_by_id(f"EX_{S}_e0").bounds = (-q_S, 1000.0)
    m.reactions.get_by_id(f"EX_{T}_e0").bounds = (-q_T, 0.0)
    m.reactions.get_by_id(f"EX_{P}_e0").bounds = (0.0, 1000.0)
    m.objective = comm.primary_biomass.id
    return comm


def _solvers():
    import optlang
    return {k.lower() for k, v in optlang.available_solvers.items() if v}


def _snapshot(comm):
    """Everything the composition functions may touch and must put back."""
    m = comm.util.model
    cfg = m.solver.configuration.tolerances
    snap = {"bounds": {r.id: tuple(r.bounds) for r in m.reactions},
            "rows": {c.name: (c.lb, c.ub) for c in m.constraints},
            "variables": {v.name: (v.lb, v.ub) for v in m.variables},
            "objective": (str(m.objective.expression), m.objective.direction),
            "bio1": {met.id: c for met, c in comm.primary_biomass.metabolites.items()},
            "abundance": {mm.id: mm.abundance for mm in comm.members},
            "abundances": dict(comm.abundances),
            "registry": sorted(getattr(comm, "_member_caps", {})),
            "tolerances": (cfg.feasibility, cfg.optimality)}
    params = getattr(m.solver.problem, "Params", None)
    if params is not None:
        snap["threads"] = params.Threads
    return snap


GLC_A = yield_on_S("A") * 5.0           # A alone on the 5 mmol S supply: 0.3 1/h
T_B = 2 * yield_on_S("B") * 1.0         # B alone on the 1 mmol T supply: 0.1198 1/h


# --------------------------------------------------------------------------- steadycom on the toys
def test_niche_steadycom_with_caps_returns_the_flat_face():
    """SYNTHESIS 4.2: niche with caps V_S,A 10 and V_T,B 3 (mmol per gDW of that member per h):
    mu* = 6 Y_B (B's cap binds for X_B <= 1/3) and every X_A in [2/3, 0.835] reaches it."""
    comm = toy("niche", q_S=5.0, q_T=1.0)
    res = steadycom(comm, caps={"A": {S: 10.0}, "B": {T: 3.0}})
    assert res.status == "optimal"
    assert res.mu_max == pytest.approx(3.0 * T_B, rel=1e-7)          # 0.3592778
    lo, hi = res.a_ranges[1.0]["A"]
    assert lo == pytest.approx(2 / 3, abs=1e-4) and hi == pytest.approx(GLC_A / (3.0 * T_B), abs=1e-4)
    assert (round(lo, 3), round(hi, 3)) == (0.667, 0.835)
    assert res.a_ranges[0.99]["A"][0] < lo and res.a_ranges[0.99]["A"][1] > hi        # nested
    assert res.solo == pytest.approx({"A": GLC_A, "B": T_B}, rel=1e-7)
    assert res.best_solo[0] == "A" and res.delta_over_best_solo == pytest.approx(3.0 * T_B - GLC_A, rel=1e-6)
    assert res.warning_codes == set()
    assert res.cap_activity["mcap_sc_B_cpd00179"]["ratio"] == pytest.approx(1.0, abs=1e-5)   # B's cap binds
    assert res.mu_at({"A": 0.7, "B": 0.3}) == pytest.approx(3.0 * T_B, rel=1e-6)
    assert res.mu_at([{"A": 1, "B": 0}, {"A": 0.9, "B": 0.1}]) == pytest.approx([GLC_A, GLC_A / 0.9], rel=1e-6)


def test_niche_coupled_profile_optimum():
    """SYNTHESIS 4.2: the coupled-form optimum of niche is a* = 0.715 (analytic 0.7147):
    mu(a) = min(Y_A q_S / a, 2 Y_B q_T / (1 - a))."""
    comm = toy("niche", q_S=5.0, q_T=1.0)
    a_star = GLC_A / (GLC_A + T_B)
    prof = composition_profile(comm, objective="coupled")
    assert prof.status == "optimal"
    assert prof.composition["A"] == pytest.approx(a_star, abs=1e-6) and round(prof.composition["A"], 4) == 0.7147
    assert prof.optimum == pytest.approx(GLC_A + T_B, rel=1e-7)
    (lo, hi), = prof.sets[0.99]
    assert lo < a_star < hi
    (lo1, hi1), = prof.sets[1.0]                                         # refined edges around a*
    assert lo1 == pytest.approx(a_star, abs=1e-5) and hi1 == pytest.approx(a_star, abs=1e-5)
    assert prof.zero_member_flux and all(z.total == pytest.approx(0.0, abs=1e-9) for z in prof.zero_member_flux)
    # on a 1001-point grid without refinement the argmax is the grid point 0.715
    grid = [i / 1000 for i in range(1001)]
    prof = composition_profile(comm, objective="coupled", grid=grid, refine=False)
    assert prof.composition["A"] == pytest.approx(0.715)
    # the composition-free face has the same optimum (max sum v_bio = max_a mu_coupled(a))
    free = composition_profile(comm, objective="free")
    assert free.optimum == pytest.approx(GLC_A + T_B, rel=1e-7)
    assert free.sets[1.0]["A"][0] == pytest.approx(a_star, abs=1e-5)


def test_hydrolase_zero_member_carries_flux_until_caps_or_kappa_tie_it_to_biomass():
    """SYNTHESIS 4.2: in hydrolase, B at a_B = 0 still opens T for A (1.0 mmol T per gDW per h), so
    the coupled form grows at 0.42 with B absent; a cap q a_B or a kappa row stops it."""
    comm = toy("hydrolase", q_S=5.0, q_T=1.0)
    grid = [0.0, 0.5, 1.0]
    prof = composition_profile(comm, objective="coupled", grid=grid, refine=False)
    mu = {c["A"]: v for c, v in prof.curve}
    assert mu[1.0] == pytest.approx(7.0 * yield_on_S("A"), rel=1e-7)     # 0.42: A on 5 S + 2 x 1 T
    zero_b = next(z for z in prof.zero_member_flux if z.member == "B")
    assert zero_b.fluxes["rxn30000_p2"] == pytest.approx(1.0, rel=1e-5)   # B hydrolyses all T at a_B = 0
    assert zero_b.total == pytest.approx(3.0, rel=1e-5)                  # T in, 2 S out (at (1 - 1e-6) mu)

    with comm.util.model:                                                # a cap q a_B is 0 at a_B = 0
        cap_member_uptake(comm, "B", T, q=2.0)
        prof = composition_profile(comm, objective="coupled", grid=grid, refine=False)
        zero_b = next(z for z in prof.zero_member_flux if z.member == "B")
        assert zero_b.total == pytest.approx(0.0, abs=1e-9)
        assert {c["A"]: v for c, v in prof.curve}[1.0] == pytest.approx(GLC_A, rel=1e-7)
    assert member_caps(comm) == []

    kin = toy("hydrolase", q_S=5.0, q_T=1.0, kinetic_coeff=1000.0)      # sum |v_B| <= kappa v_bio,B = 0
    prof = composition_profile(kin, objective="coupled", grid=grid, refine=False)
    zero_b = next(z for z in prof.zero_member_flux if z.member == "B")
    assert zero_b.total == pytest.approx(0.0, abs=1e-9)


def test_hydrolase_eq9_removes_zero_biomass_flux_in_steadycom():
    """With eq. 9 a member at X_B = 0 carries no flux: A alone grows at 0.3, not the relabelled 0.42."""
    comm = toy("hydrolase", q_S=5.0, q_T=1.0)
    res = steadycom(comm)
    mu, fluxes = res.mu_at({"A": 1.0, "B": 0.0}, fluxes=True)
    assert mu == pytest.approx(GLC_A, rel=1e-7)
    assert fluxes["rxn30000_p2"] == pytest.approx(0.0, abs=1e-9) and fluxes["rxn10000_c2"] == pytest.approx(0.0, abs=1e-9)
    free = steadycom(comm, scale_bounds=None)
    mu, fluxes = free.mu_at({"A": 1.0, "B": 0.0}, fluxes=True)
    assert mu == pytest.approx(7.0 * yield_on_S("A"), rel=1e-7)
    assert fluxes["rxn30000_p2"] == pytest.approx(1.0, rel=1e-7)          # relabelling without eq. 9
    assert res.mu_max == pytest.approx(0.42, rel=1e-7) and "not_identifiable" in res.warning_codes
    # the sliver eq. 9 alone still allows: B at the eq.-9 bound of its export (2 S <= 1000 X_B)
    assert res.a_ranges[1.0]["B"][0] == pytest.approx(0.002, rel=1e-3)
    sliver = steadycom(comm, x_min=0.01)
    hits = [w for w in sliver.warnings if w.code == "relabelling"]
    assert [w.member for w in hits] == ["B"]
    assert "rxn30001_p2" in {rid for rid, v, bound in hits[0].data["reactions"]}


def test_steadycom_reduces_to_predict_abundances_without_eq9_and_caps():
    """The docstring's statement: drains open, no caps, no eq. 9 -> max sum v_bio, the LP of
    predict_abundances(regularization=False); its composition is inside the reported range."""
    from mscommunity.commmembers import open_member_drains, restore_member_drains
    comm = toy("hydrolase", q_S=5.0, q_T=1.0)
    res = steadycom(comm, scale_bounds=None)
    previous = open_member_drains(comm)
    try:
        a = comm.predict_abundances(regularization=False, pfba=False)
        total = sum(comm.solution.fluxes[m.primary_biomass.id] for m in comm.members)
    finally:
        restore_member_drains(comm, previous)
    assert res.mu_max == pytest.approx(total, rel=1e-7)
    lo, hi = res.a_ranges[1.0]["A"]
    assert lo - 1e-6 <= a["A"] <= hi + 1e-6


def test_corner_and_monoculture_warnings():
    """shared, S limiting, caps 5 / 5.5: A's yield is 0.2 % higher, so A alone is optimal (corner);
    a monoculture rate outside the solo rate is flagged."""
    comm = toy("shared", q_S=5.0)
    res = steadycom(comm, caps={"A": {S: 5.0}, "B": {S: 5.5}}, monoculture_rates={"A": 0.3, "B": (0.4, (0.35, 0.45))})
    assert res.mu_max == pytest.approx(GLC_A, rel=1e-6)
    assert res.best_solo[0] == "A" and abs(res.delta_over_best_solo) < 1e-6 * GLC_A
    assert "corner" in res.warning_codes
    assert {w.member for w in res.warnings if w.code == "solo_outside_ci"} == {"B"}
    assert res.a_ranges[1.0]["A"][1] == pytest.approx(1.0, abs=1e-6)


def test_kappa_calibration_and_kappa_below_saturation():
    """kappa_sat,m = min sum |v| / v_bio at the solo optimum = (10 + 20/e) + 20/e for a member on S
    (uptake plus respiration per gDW). A kappa below it forbids the member's solo optimum."""
    comm = toy("shared", q_S=5.0)
    sat = {mid: BIO_S + 2 * BIO_E / E_RESP[mid] for mid in ("A", "B")}
    cal = calibrate_kappa(comm, per_member=True)
    assert dict(cal) == pytest.approx(sat, rel=1e-6) and cal.kappa_sat == pytest.approx(sat, rel=1e-6)
    shared = calibrate_kappa(comm)
    assert shared.shared == pytest.approx(sat["B"], rel=1e-6) and set(shared.values()) == {shared.shared}
    assert not shared.warnings                                            # the default 750 is far above 23.4
    ok = steadycom(comm, kappa=cal)
    assert ok.solo == pytest.approx(ok.solo_kappa_off, rel=1e-6) and "kappa_below_sat" not in ok.warning_codes
    low = steadycom(comm, kappa=0.95 * sat["A"])
    assert {w.member for w in low.warnings if w.code == "kappa_below_sat"} == {"A", "B"}
    assert low.solo["A"] < low.solo_kappa_off["A"] * 0.99
    with comm.util.model:                                                 # installed rows below kappa_sat
        comm.add_commkinetics(20.0)
        cal = calibrate_kappa(comm)
        assert any(w.code == "kappa_default" and "installed" in w.message for w in cal.warnings)


def test_kappa_sat_coupled_is_the_saturation_threshold():
    """kappa_sat_coupled(a): at kappa_sat (1 + 1e-6) the coupled form reaches (1 - 1e-7) mu_off;
    1 % below it, it does not."""
    comm = toy("hydrolase", q_S=5.0, q_T=1.0)
    a = {"A": 0.8, "B": 0.2}
    k = kappa_sat_coupled(comm, a)
    model = comm.util.model
    with model:
        comm.set_abundance({mid: {"abundance": v} for mid, v in a.items()})
        mu_off = model.slim_optimize()
        comm.add_commkinetics(k * (1 + 1e-6))
        assert model.slim_optimize() >= mu_off * (1 - 1e-7) - 1e-12
        comm.add_commkinetics(k * 0.99)
        assert model.slim_optimize(error_value=0.0) < mu_off * (1 - 1e-7)
        model.remove_cons_vars([c for c in model.constraints if c.name.endswith("_commKin")])


def test_non_monotone_feasibility_and_no_false_zero():
    """kappa rows plus maintenance A_m X_m make feasibility an interval in mu (the kinetic budget
    kappa mu X_m vanishes at low mu, the maintenance demand does not): the search starts from the
    best solo and still finds the top. With a demand nothing can pay, the status is infeasible and
    mu_max None, never 0."""
    comm = toy("shared", q_S=5.0)
    res = steadycom(comm, caps={"A": {S: 5.0}, "B": {S: 5.5}}, maintenance={"A": 1.0, "B": 1.0}, kappa=30.0)
    assert res.status == "optimal" and res.mu_max >= res.best_solo[1] * (1 - 1e-9)
    floor = res.mu_at({"A": 1.0, "B": 0.0})
    assert floor == pytest.approx(res.solo["A"], rel=1e-6)
    low = steadycom(comm, caps={"A": {S: 5.0}, "B": {S: 5.5}}, maintenance={"A": 1.0, "B": 1.0}, kappa=30.0)
    assert low.mu_max == pytest.approx(res.mu_max, rel=1e-9)
    dead = steadycom(comm, maintenance={"A": 1e4, "B": 1e4})
    assert dead.status == "infeasible" and dead.mu_max is None and "infeasible" in dead.warning_codes


def test_chemostat_mode_maximises_total_biomass_at_D():
    """mu = D, supplies per litre: X_m = Y_m supply_m / D for the niche (each member on its own
    substrate), so X* = (0.3 + 0.1198) / D and a_A = 0.7147."""
    comm = toy("niche", q_S=5.0, q_T=1.0)
    D = 0.2
    res = steadycom(comm, mode="chemostat", D=D, caps={"A": {S: 10.0}, "B": {T: 3.0}})
    assert res.status == "optimal" and res.X_star == pytest.approx((GLC_A + T_B) / D, rel=1e-7)
    lo, hi = res.a_ranges[1.0]["A"]
    assert lo == pytest.approx(GLC_A / (GLC_A + T_B), abs=1e-5) and hi == pytest.approx(lo, abs=1e-5)
    assert res.solo == pytest.approx({"A": GLC_A / D, "B": T_B / D}, rel=1e-7)
    assert res.X_at({"A": 0.5, "B": 0.5}) == pytest.approx(2 * T_B / D, rel=1e-6)
    supplied = steadycom(comm, mode="chemostat", D=D, supplies={S: 2.5})
    assert supplied.solo["A"] == pytest.approx(GLC_A / 2 / D, rel=1e-7)
    with pytest.raises(ValueError):
        res.mu_at({"A": 0.5, "B": 0.5})
    with pytest.raises(ValueError):
        steadycom(comm, mode="chemostat")


# --------------------------------------------------------------------------- restoration
def test_every_function_restores_the_community():
    """Forced member bounds, abundance-basis caps and maintenance rows, kinetic rows, drains, bio1,
    the objective, tolerances, threads and the abundances come back exactly; the run itself saw them
    released or lifted."""
    comm = toy("hydrolase", q_S=5.0, q_T=1.0, kinetic_coeff=500.0)
    model = comm.util.model
    model.reactions.get_by_id("rxn00062_c1").lower_bound = 0.05                # a floor at the declared a
    cap_member_uptake(comm, "B", T, q=2.0)                                        # basis "abundance"
    set_member_maintenance(comm, "B", 0.5)
    comm.set_abundance({"A": {"abundance": 0.7}, "B": {"abundance": 0.3}})
    before = _snapshot(comm)
    res = steadycom(comm, caps={"A": {S: 8.0}, "B": {("rxn30000_p2",): 2.0}}, maintenance={"A": 0.2},
                    kappa={"A": 400.0, "B": 400.0}, supplies={S: 6.0}, solver_threads=1)
    assert res.status == "optimal"
    assert res.released == {"rxn00062_c1": (0.05, 1000.0)}
    assert {"A_commKin", "B_commKin", "mcap_B_cpd00179", "mmaint_B"} <= set(res.lifted)
    assert "lifted_rows" in res.warning_codes
    assert _snapshot(comm) == before
    res.mu_at({"A": 0.5, "B": 0.5})
    assert _snapshot(comm) == before
    steadycom(comm, mode="chemostat", D=0.1)
    assert _snapshot(comm) == before
    calibrate_kappa(comm, per_member=True, maintenance={"B": 0.1})
    assert _snapshot(comm) == before
    kappa_sat_coupled(comm, {"A": 1.0, "B": 0.0})
    assert _snapshot(comm) == before
    composition_profile(comm, objective="coupled", grid=[0.0, 0.3, 0.6, 1.0])
    assert _snapshot(comm) == before
    composition_profile(comm, objective="free")
    assert _snapshot(comm) == before
    with model:                                                                   # inside a caller's context
        steadycom(comm, caps={"A": {S: 8.0}})
        assert _snapshot(comm) == before
    with pytest.raises(KeyError):                                                 # a failing call
        steadycom(comm, caps={"A": {"cpd99999": 1.0}})
    assert _snapshot(comm) == before
    with model:                                                                   # abundances in a caller's context
        kappa_sat_coupled(comm, {"A": 1.0, "B": 0.0})
        composition_profile(comm, objective="coupled", grid=[0.0, 1.0], refine=False)
        assert _snapshot(comm) == before
        # A's forced ATPM floor cannot be paid at a_A = 0: no kappa exists, and the error says so
        with pytest.raises(ValueError, match="no kappa"):
            kappa_sat_coupled(comm, {"A": 0.0, "B": 1.0})
        assert _snapshot(comm) == before
    assert _snapshot(comm) == before


# --------------------------------------------------------------------------- real communities (slow)
# The campaign systems of 3H11_R12_SynCom/Metabolic_modeling, built by abund_common exactly as the
# abundance study built them (SYNTHESIS.md 4.1-4.2 anchors). Each system runs in a fresh interpreter
# (see tests/test_l_members.py for why), and only with MSCOMMUNITY_SLOW_TESTS=1.
CAMPAIGN = "/home/freiburger/Documents/3H11_R12_SynCom/Metabolic_modeling"
_HAVE_CAMPAIGN = all(os.path.exists(os.path.join(CAMPAIGN, p)) for p in (
    "abund_common.py", "multisystem_sweep.py", "data/multisystem3/systems",
    "data/multisystem3/commphitting_data/measurements.json"))
slow_real = pytest.mark.skipif(
    not (_HAVE_CAMPAIGN and os.environ.get("MSCOMMUNITY_SLOW_TESTS") == "1" and "gurobi" in _solvers()),
    reason="real-community anchors need the 3H11_R12_SynCom campaign files, Gurobi and MSCOMMUNITY_SLOW_TESTS=1")


def _run_real(name, timeout=1500):
    tests = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(tests)
    code = f"import sys; sys.path[:0] = [{tests!r}, {root!r}]; import test_m_composition as t; t._real_{name}()"
    try:
        run = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail(f"the {name} anchors took over {timeout} s in a fresh interpreter")
    assert run.returncode == 0, run.stderr[-6000:]
    marker = f"{name.upper()}_OK"
    assert marker in run.stdout, run.stdout[-3000:]
    return json.loads(run.stdout.split(marker, 1)[1].strip().splitlines()[0])


def _campaign():
    import logging
    import mscommunity
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.abspath(mscommunity.__file__).startswith(root), mscommunity.__file__
    if CAMPAIGN not in sys.path:
        sys.path.append(CAMPAIGN)
    import abund_common
    logging.disable(logging.NOTSET)          # build('kpbt') silences every logger at import
    return abund_common


def _monocultures(condition):
    with open(os.path.join(CAMPAIGN, "data/multisystem3/commphitting_data/measurements.json")) as fh:
        rates = json.load(fh)["conditions"][condition]["monoculture_rates_same_medium"]
    return {member: (rates[key]["mu_per_h"], tuple(rates[key]["ci95_regression_per_h"]))
            for member, key in (("ecoli", "ec"), ("pf", "pf"))}


@slow_real
def test_real_maltose_4hb_anchors():
    out = _run_real("mal4hb")
    assert out["mu_r1"] == pytest.approx(0.3447752, abs=5e-7)


@slow_real
def test_real_syncom_anchors():
    out = _run_real("syncom")
    assert out["mu_maint"] == pytest.approx(0.0170679, abs=5e-8)


@slow_real
def test_real_acetate_kappa_sat():
    out = _run_real("acetate")
    assert out["kappa_sat"]["pf"] == pytest.approx(751.0, abs=0.05)


@slow_real
def test_real_kpbt_chemostat_anchors():
    out = _run_real("kpbt")
    assert out["X_star"] == pytest.approx(0.5418, abs=5e-5)


def _real_mal4hb():
    """maltose + 4HB with model-yield caps (abund_common.build_mal4hb('model_yield')): the per-member
    caps q = mu_mono / Y_model on E. coli's maltose and P. fluorescens' 4HB uptake, the community
    exchanges for both opened to +-1000 (the per-member caps are then the limit). R1 = caps only."""
    A = _campaign()
    from mscommunity.commmembers import member_net_exchange, member_transporters
    b = A.build_mal4hb("model_yield", solver="gurobi")
    comm = b.msc
    q = b.info["per_member_specific_caps"]
    caps = {"ecoli": {"cpd00179": q["ecoli"]["maltose"]}, "pf": {"cpd00136": q["pf"]["4HB"]}}
    supplies = {"EX_cpd00179_e0": (-1000.0, 1000.0), "EX_cpd00136_e0": (-1000.0, 1000.0)}
    mono = _monocultures("maltose_4HB")
    before = _snapshot(comm)

    r1 = steadycom(comm, caps=caps, supplies=supplies, monoculture_rates=mono, solver_threads=4)
    # SYNTHESIS anchor 0.3447752 is the prototype's (abund_steadycom_lib) value; the adversarial
    # verifier's independent fixed-composition implementation got 0.34477496308
    # (_verify/m-steadycom/v_mal4hb.json arms.R1.mu_max), which this reproduces to 1e-9. The 2.2e-7
    # gap is not the solver tolerance, eq. 9 on member sinks or gross vs net caps (all checked).
    assert r1.mu_max == pytest.approx(0.34477496307887506, abs=1e-9)
    assert r1.mu_max == pytest.approx(0.3447752, abs=5e-7)
    assert {"corner", "relabelling"} <= r1.warning_codes
    lo, hi = r1.a_ranges[1.0]["ecoli"]
    assert 0.00078 < lo <= hi < 0.0008                                   # the 0.00079 E. coli sliver
    relabel = next(w for w in r1.warnings if w.code == "relabelling")
    assert relabel.member == "ecoli"
    glu = {r.id for r in member_transporters(comm, "ecoli", "cpd00023")}
    hits = [(rid, v) for rid, v, bound in relabel.data["reactions"] if rid in glu]
    assert hits                                                           # glutamate moves at the eq.-9 bound,
    for rid, v in hits:                                                   # 1000 mmol per gDW of E. coli per h
        assert abs(v) / relabel.data["a"] == pytest.approx(1000.0, rel=1e-6)
    assert member_net_exchange(comm, r1.fluxes, "ecoli")["cpd00023"] < 0  # E. coli releases glutamate

    r2 = steadycom(comm, caps=caps, supplies=supplies, kappa=472.5, monoculture_rates=mono, solver_threads=4)
    assert r2.mu_max == pytest.approx(0.2978314, abs=1e-7)
    lo, hi = r2.a_ranges[1.0]["ecoli"]
    assert lo == pytest.approx(0.567055, abs=2e-6) and hi == pytest.approx(0.567083, abs=2e-6)
    assert r2.solo["pf"] is None or r2.solo["pf"] < 1e-9
    assert {w.member for w in r2.warnings if w.code == "solo_outside_ci"} == {"pf"}
    assert r2.solo_kappa_off["pf"] == pytest.approx(mono["pf"][0], abs=1e-4)          # 0.3443 alone without kappa

    cal = calibrate_kappa(comm, per_member=True, caps=caps, supplies=supplies, solver_threads=4)
    assert cal.kappa_sat["ecoli"] == pytest.approx(420.1, abs=0.05)
    assert cal.kappa_sat["pf"] == pytest.approx(556.7, abs=0.05)
    assert _snapshot(comm) == before
    print("MAL4HB_OK", json.dumps({"mu_r1": r1.mu_max, "mu_k472": r2.mu_max, "a_ec_k472": r2.a_ranges[1.0]["ecoli"],
                                   "kappa_sat": cal.kappa_sat}), flush=True)


def _real_syncom():
    """The published 3H11 + R12 SynCom (abund_common.build_syncom = multisystem_sweep.syncom_spec,
    assemble_community, constrain; kinetic rows removed; ATPM floors a_m A_m / 58.33)."""
    A = _campaign()
    b = A.build_syncom(solver="gurobi")
    comm = b.msc
    before = _snapshot(comm)
    cal = calibrate_kappa(comm, per_member=True, solver_threads=4)
    assert cal.kappa_sat["3H11"] == pytest.approx(1103.6, abs=0.05)
    assert cal.kappa_sat["R12"] == pytest.approx(1421.6, abs=0.05)
    assert any(w.code == "kappa_default" for w in cal.warnings)          # 750 < 1421.6
    maintenance = {k: (rxn, A_m) for k, (rxn, A_m) in b.info["maintenance_per_gDW_member"].items()}
    res = steadycom(comm, maintenance=maintenance, feasibility_tol=1e-9, solver_threads=4)
    assert res.mu_max == pytest.approx(0.0170679, abs=5e-8)
    assert set(res.released) == {"ATPM_c1", "ATPM_c2"}                   # the floors at the declared a
    assert _snapshot(comm) == before

    # the round trip: kappa_sat of the coupled form at a, then predict_abundances at that kappa returns a
    probe12 = {0.3: 3142.2925519819273, 0.5: 1795.548807317192, 0.658: 1362.0136447295095, 0.8: 1234.127162869595}
    returned = {}
    for a, k_probe in probe12.items():
        fresh = A.build_syncom(solver="gurobi")
        k = kappa_sat_coupled(fresh.msc, {"3H11": a, "R12": 1 - a}, solver_threads=4)
        assert k == pytest.approx(k_probe, rel=1e-5)
        fresh.model.solver.problem.Params.Threads = 4
        A.kin_on(fresh.msc, k)
        A.open_drains(fresh.msc, (0.0, 1000.0))
        with A.quiet():
            out = fresh.msc.predict_abundances(regularization=False, pfba=False)
        assert out["3H11"] == pytest.approx(a, abs=1e-6)
        returned[a] = out["3H11"]
    print("SYNCOM_OK", json.dumps({"mu_maint": res.mu_max, "kappa_sat": cal.kappa_sat,
                                   "round_trip": {str(a): v for a, v in returned.items()}}), flush=True)


def _real_acetate():
    """CommPhitting E. coli + P. fluorescens on acetate (abund_common.build_acetate('q_min')): each
    member's kappa_sat at its solo optimum; P. fluorescens' is above the constructor default 750."""
    A = _campaign()
    b = A.build_acetate("q_min", solver="gurobi")
    cal = calibrate_kappa(b.msc, per_member=True, solver_threads=4)
    assert cal.kappa_sat["pf"] == pytest.approx(751.0, abs=0.05)
    assert cal.kappa_sat["ecoli"] == pytest.approx(680.7, abs=0.05)
    assert any(w.code == "kappa_default" and "750" in w.message for w in cal.warnings)
    print("ACETATE_OK", json.dumps({"kappa_sat": cal.kappa_sat}), flush=True)


def _real_kpbt():
    """K. pneumoniae + B. thetaiotaomicron at S6 (abund_common.build_kpbt('S6', X=1.0, maint='free')):
    supplies per litre, mu = D = 0.2 1/h, max sum X_m."""
    A = _campaign()
    b = A.build_kpbt("S6", X=1.0, maint="free", solver="gurobi")
    comm = b.msc
    D = b.info["D_per_h"]
    before = _snapshot(comm)
    res = steadycom(comm, mode="chemostat", D=D, solver_threads=4)
    assert res.X_star == pytest.approx(0.5418, abs=5e-5)
    lo, hi = res.a_ranges[1.0]["Kp"]
    assert lo <= 0.9044 + 1e-4 and hi >= 0.9044 - 1e-4 and hi - lo < 2e-4
    assert res.solo["Kp"] == pytest.approx(0.104, abs=5e-4)              # Kp alone, eq. 9
    free = steadycom(comm, mode="chemostat", D=D, scale_bounds=None, solver_threads=4)
    assert free.X_star == pytest.approx(res.X_star, rel=1e-7)            # composition-free: the same optimum
    assert free.solo["Kp"] == pytest.approx(0.538, abs=5e-4)             # Kp alone uses Bt's zero-biomass hydrolysis
    assert _snapshot(comm) == before
    print("KPBT_OK", json.dumps({"X_star": res.X_star, "a_kp": [lo, hi], "solo_kp_eq9": res.solo["Kp"],
                                 "solo_kp_free": free.solo["Kp"]}), flush=True)
