# -*- coding: utf-8 -*-
"""Chemostat steady states (mscommunity.commchemostat).

Toys (ModelSEED-style members merged by build_from_species_models, as in tests/test_l_members.py),
all energy-limited so every number has a closed form:

    glucose eater   glucose_e0 -> glucose_c (rxn10000), glucose -> 30 ATP + 2 acetate (rxn20000),
                    acetate exported (rxn10002); biomass 100 ATP per gDW; ATPM rxn00062
                    yield 0.3 gDW / mmol glucose; Michaelis-Menten ceiling (vmax 10, Km 0.001 mM)
    acetate eater   acetate_e0 -> acetate_c (rxn10002), acetate -> 10 ATP (rxn20001); no glucose
                    transporter; biomass 100 ATP; ceiling on acetate (vmax 10, Km 0.01 mM)

Pirt (one member, D 0.1 1/h, 1 mM glucose): q = (100 D + m) / 30, S* = Km q / (vmax - q),
X* = D (S_in - S*) / q = 0.300 gDW/L without maintenance and 0.2609 with m = 1.5 mmol ATP/gDW/h.

The Kp + Bt anchors (Khazaei 2020, route-A member models, the paper's kinetics) run in a fresh
interpreter when the 3H11_R12_SynCom data are on this machine; see test_kpbt_anchors.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import subprocess
import sys

import pytest
from cobra import Metabolite, Model, Reaction

from mscommunity.commchemostat import ConcentrationSwitch, chemostat, member_block
from mscommunity.commhelper import build_from_species_models
from mscommunity.commmembers import member_biomass_variables, member_caps, set_member_maintenance
from mscommunity.mscommsim import MSCommunity

GLC, AC, BIO = "cpd00027", "cpd00029", "cpd11416"
ATP, H2O, ADP, PI, H = "cpd00002", "cpd00001", "cpd00008", "cpd00009", "cpd00067"
D, S_IN = 0.1, 1.0
GLC_KIN, AC_KIN = (10.0, 0.001, "toy"), (10.0, 0.01, "toy")


def _member(mid, eats="glucose"):
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
    rs = [rxn(f"EX_{GLC}_e0", {(GLC, "e0"): -1}, -1000.0), rxn(f"EX_{AC}_e0", {(AC, "e0"): -1}, -1000.0)]
    if eats == "glucose":
        rs += [rxn("rxn10000_c0", {(GLC, "e0"): -1, (GLC, "c0"): 1}),
               rxn("rxn20000_c0", {(GLC, "c0"): -1, (AC, "c0"): 2, **atp(30)}),
               rxn("rxn10002_c0", {(AC, "c0"): -1, (AC, "e0"): 1}, -1000.0)]
    else:
        rs += [rxn("rxn10002_c0", {(AC, "e0"): -1, (AC, "c0"): 1}),
               rxn("rxn20001_c0", {(AC, "c0"): -1, **atp(10)})]
    rs += [rxn("rxn00062_c0", atp(-1)),
           rxn("bio1", {**atp(-100), (BIO, "c0"): 1}),
           rxn("SK_cpd11416_c0", {(BIO, "c0"): -1})]
    m.add_reactions(rs)
    m.objective = "bio1"
    return m


def _community(*members):
    ids = [m.id for m in members]
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_from_species_models(list(members), abundances={i: 1.0 / len(ids) for i in ids})
        comm = MSCommunity(model=model, ids=ids, kinetic_coeff=1e6, close_member_drains=True)
    return comm


def _pirt():
    return _community(_member("pirt"))


def _pair():
    return _community(_member("A"), _member("B", eats="acetate"))


def _pirt_X(m_atp, vmax=10.0, Km=0.001):
    q = (100 * D + m_atp) / 30
    s = Km * q / (vmax - q)
    return D * (S_IN - s) / q, s, q


def _one(states, *members):
    found = states.of(*members)
    assert len(found) == 1, (members, states, states.unsolved)
    return found[0]


# --------------------------------------------------------------------------- Pirt
def test_pirt_without_and_with_maintenance():
    comm = _pirt()
    n_rows = len(comm.util.model.constraints)
    kin = {"pirt": {f"EX_{GLC}_e0": GLC_KIN}}
    for m_atp, maintenance, expect in ((0.0, "per_member", 0.300), (1.5, {"pirt": 1.5}, 0.2609)):
        res = chemostat(comm, D, {GLC: S_IN}, kinetics=kin, maintenance=maintenance)
        X, s, q = _pirt_X(m_atp)
        st = _one(res, "pirt")
        assert st.X["pirt"] == pytest.approx(X, rel=1e-6)
        assert round(st.X["pirt"], 4 if m_atp else 3) == expect
        assert st.residuals[GLC] == pytest.approx(s, rel=1e-5)
        assert set(st.residuals) == {GLC}             # acetate: neither fed nor taken, so not tracked
        assert st.mu["pirt"] == pytest.approx(D, rel=1e-7) and st.a == {"pirt": 1.0}
        assert st.exists and st.stable and st.max_real_eigenvalue < 0
        assert st.binding == {"pirt": [f"EX_{GLC}_e0"]}
        (sy,) = st.supply_yield["pirt"]
        assert sy["yield_gDW_per_mmol"] == pytest.approx(D / q)
        assert sy["X_from_yield_x_share_x_supply"] == pytest.approx(st.X["pirt"], rel=1e-9)
        washout = _one(res)
        assert washout.residuals[GLC] == S_IN and washout.invasion["pirt"] > 0 and not washout.stable
    assert len(comm.util.model.constraints) == n_rows            # the community model is untouched


def test_pirt_maintenance_from_a_registered_row():
    comm = _pirt()
    X = member_biomass_variables(comm)
    set_member_maintenance(comm, "pirt", 1.5, basis="X", X=X)
    res = chemostat(comm, D, {GLC: S_IN}, kinetics={"pirt": {GLC: GLC_KIN}}, subsets=[("pirt",)])
    st = _one(res, "pirt")
    assert st.X["pirt"] == pytest.approx(_pirt_X(1.5)[0], rel=1e-6)
    assert res.settings["maintenance"]["pirt"][2] == "registered mmaint_pirt"
    # maintenance=None releases it again; the registered row itself stays on the community model
    res = chemostat(comm, D, {GLC: S_IN}, kinetics={"pirt": {GLC: GLC_KIN}}, subsets=[("pirt",)], maintenance=None)
    assert _one(res, "pirt").X["pirt"] == pytest.approx(_pirt_X(0.0)[0], rel=1e-6)
    assert [c.name for c in member_caps(comm)] == ["mmaint_pirt"]


def test_member_block_is_the_member_model():
    comm = _pirt()
    b = member_block(comm, "pirt", tracked=(GLC,))
    ids = {r.id for r in b.model.reactions}
    assert {"rxn10000_c1", "rxn20000_c1", "rxn10002_c1", "rxn00062_c1", "bio2", "SK_cpd11416_c1"} <= ids
    assert "bio1" not in ids and b.model.reactions.get_by_id("SK_cpd11416_c1").bounds == (0.0, 1000.0)
    assert b.exchange == {GLC: f"EX_{GLC}_e0", AC: f"EX_{AC}_e0"}
    assert b.closed == [GLC]                                    # tracked without kinetics: no uptake
    assert b.respond({GLC: 1.0}).mu == 0.0
    b = member_block(comm, "pirt", tracked=(GLC,), kinetics=[], tie_break=None)
    assert b.pfba_model is None


# --------------------------------------------------------------------------- cross-feeding pair
def _pair_states(**kw):
    comm = _pair()
    kin = {"A": {f"EX_{GLC}_e0": GLC_KIN}, "B": {f"EX_{AC}_e0": AC_KIN}}
    return comm, chemostat(comm, D, {GLC: S_IN}, kinetics=kin, **kw)


def test_cross_feeding_pair_states():
    comm, res = _pair_states()
    XA, sg, _ = _pirt_X(0.0)
    q_B = 100 * D / 10
    s_ac = AC_KIN[1] * q_B / (AC_KIN[0] - q_B)
    XB = (2 * D * (S_IN - sg) - D * s_ac) / q_B
    both = _one(res, "A", "B")
    assert both.X["A"] == pytest.approx(XA, rel=1e-6) and both.X["B"] == pytest.approx(XB, rel=1e-6)
    assert both.residuals[AC] == pytest.approx(s_ac, rel=1e-5)
    assert both.a["A"] == pytest.approx(XA / (XA + XB), rel=1e-6)
    assert both.stable and len(both.eigenvalues) == 4 and both.max_real_eigenvalue < 0
    assert both.uptake["A"][AC] == pytest.approx(-2 * 100 * D / 30, rel=1e-6)     # A releases acetate
    (sy_b,) = both.supply_yield["B"]
    assert sy_b["cpd"] == AC and set(sy_b["supply_mmol_per_L_h"]["released_by"]) == {"A"}
    assert sy_b["X_from_yield_x_share_x_supply"] == pytest.approx(both.X["B"], rel=1e-9)
    assert sy_b["share_of_supply"] == pytest.approx(1 - D * s_ac / (2 * D * (S_IN - sg)), rel=1e-6)
    # A alone accumulates acetate, which B invades; B alone has nothing to eat; washout is invaded by A
    alone = _one(res, "A")
    assert alone.residuals[AC] == pytest.approx(2 * (S_IN - sg), rel=1e-6)
    mu_B = AC_KIN[0] * alone.residuals[AC] / (AC_KIN[1] + alone.residuals[AC]) * 10 / 100
    assert alone.invasion["B"] == pytest.approx(mu_B - D, rel=1e-6) and not alone.stable
    assert res.of("B") == [] and ("B",) in res.unsolved
    assert _one(res).invasion["A"] > 0
    assert [s.members for s in res if s.stable] == [("A", "B")]
    json.dumps(both.as_dict())


def test_switch_existence_criterion_and_invasion():
    thr = 1e-5
    comm, res = _pair_states(switches={"B": ConcentrationSwitch(GLC, thr)})
    both = _one(res, "A", "B")
    assert not both.exists and not both.stable
    assert both.switch_on == {"B": False} and both.switch_margins["B"] == pytest.approx(thr - both.residuals[GLC])
    (crit,) = both.existence
    q_thr = GLC_KIN[0] * thr / (GLC_KIN[1] + thr)
    assert crit["consumers"]["A"]["X_needed"] == pytest.approx(D * (S_IN - thr) / q_thr, rel=1e-9)
    assert crit["consumers"]["A"]["binding_at_state"] and not crit["consumers"]["A"]["satisfied"]
    alone = _one(res, "A")
    assert alone.invasion["B"] == pytest.approx(-D) and alone.invasion_without_switches["B"] > 0
    # a plain callable gates growth too, but reports no margin and no criterion
    comm, res = _pair_states(switches={"B": lambda conc: conc[GLC] <= thr}, subsets=[("A", "B")])
    both = _one(res, "A", "B")
    assert not both.exists and both.switch_margins == {"B": None} and both.existence == []


def test_members_and_argument_checks():
    comm = _pair()
    kin = {"A": {f"EX_{GLC}_e0": GLC_KIN}, "B": {f"EX_{AC}_e0": AC_KIN}}
    res = chemostat(comm, D, {GLC: S_IN}, kinetics=kin, members=["A"])
    assert [s.members for s in res] == [(), ("A",)] and set(res[1].X) == {"A"}
    with pytest.raises(ValueError, match="gas"):
        chemostat(comm, D, {"cpd00007": 0.2}, kinetics=kin, gas={"cpd00007": (10.0, 0.2)})
    with pytest.raises(KeyError):
        chemostat(comm, D, {"cpd99999": 1.0}, kinetics=kin)
    with pytest.raises(ValueError, match="D must be"):
        chemostat(comm, 0.0, {GLC: S_IN}, kinetics=kin)
    with pytest.raises(ValueError, match="no transporter"):
        chemostat(comm, D, {GLC: S_IN}, kinetics={"B": {f"EX_{GLC}_e0": GLC_KIN}})
    # a reaction-set key: the compound is the one the reactions move
    res = chemostat(comm, D, {GLC: S_IN}, kinetics={"A": {("rxn10000_c1",): GLC_KIN}, "B": kin["B"]},
                    subsets=[("A", "B")])
    assert _one(res, "A", "B").ceilings["A"][0].cpd == GLC


# --------------------------------------------------------------------------- Kp + Bt (Khazaei 2020)
KPBT = "/home/freiburger/Documents/3H11_R12_SynCom/Metabolic_modeling/data/multisystem3/kpbt"
KPBT_FILES = ("Kp_routeA.json", "Bt_routeA.json", "medium_routeA.json", "inputs.json")
PROTOTYPE_A_KP = {"S4": 0.8476, "S5": 0.8149, "S6": 0.7982, "S7": 0.7835, "S8": 0.7781}


@pytest.mark.skipif(not all(os.path.exists(os.path.join(KPBT, f)) for f in KPBT_FILES),
                    reason="the 3H11_R12_SynCom Kp/Bt inputs are not on this machine")
def test_kpbt_anchors():
    """The joint kinetic chemostat state of K. pneumoniae + B. thetaiotaomicron (route-A member
    models, the paper's code kinetics, shipped maintenance) against the prototype
    3H11_R12_SynCom/Metabolic_modeling/abund_chemostat_steady.py (steady_shipped_campaign.json):
    S6 a_Kp 0.798 with every Jacobian eigenvalue negative, S4-S8 a_Kp 0.848 ... 0.778, and under
    the paper's 0.35 uM Bt O2 rule no Kp+Bt state at any of them, X_Kp >= 0.483 gDW/L needed.
    Runs in a fresh interpreter (see test_l_members' CommPhitting test for why)."""
    tests = os.path.dirname(os.path.abspath(__file__))
    code = f"import sys; sys.path[:0] = [{tests!r}, {os.path.dirname(tests)!r}]; " \
           "import test_m_chemostat as t; t._kpbt_checks()"
    try:
        run = subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(tests), capture_output=True,
                             text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        pytest.fail("the Kp/Bt chemostat anchors took over 1800 s in a fresh interpreter")
    assert run.returncode == 0, run.stderr[-6000:]
    assert "KPBT_CHECKS_OK" in run.stdout, run.stdout[-3000:]


def _kpbt_build():
    import cobra
    kp = cobra.io.load_json_model(os.path.join(KPBT, "Kp_routeA.json"))
    bt = cobra.io.load_json_model(os.path.join(KPBT, "Bt_routeA.json"))
    kp.id, bt.id = "Kp", "Bt"
    # extracellular (e0-only) reactions keep an _e0 suffix in the merge: attribute them from the member models
    e0only = {m.id: [r.id for r in m.reactions if not r.boundary and all(x.compartment == "e0" for x in r.metabolites)]
              for m in (kp, bt)}
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_from_species_models([kp, bt], model_id="kpbt", name="kpbt",
                                          abundances={"Kp": 0.8, "Bt": 0.2}, commkinetics=False)
        comm = MSCommunity(model=model, abundances={
            "Kp": {"abundance": 0.8, "biomass_compound": model.metabolites.cpd11416_c1},
            "Bt": {"abundance": 0.2, "biomass_compound": model.metabolites.cpd11416_c2}},
            ID="kpbt", kinetic_coeff=1e6, close_member_drains=True)
    # the prototype's static medium (abund_chemostat_lib.MemberLP): every exchange closed, then the
    # feed salts and H2O / H+ / CO2 open; B. thetaiotaomicron's trace additions (0.01) are its own
    model = comm.util.model
    for r in model.exchanges:
        r.lower_bound = 0.0
    for cpd in ("cpd00009", "cpd00205", "cpd00013", "cpd00048", "cpd00971", "cpd00099", "cpd00063",
                "cpd00254", "cpd00030", "cpd00149", "cpd10515", "cpd00001", "cpd00067", "cpd00011"):
        if f"EX_{cpd}_e0" in model.reactions:
            model.reactions.get_by_id(f"EX_{cpd}_e0").lower_bound = -1000.0
    medium = json.load(open(os.path.join(KPBT, "medium_routeA.json")))
    trace = {ex: (-0.01, 1000.0) for ex in medium["Bt"] if ex in medium["trace_additions_provenance"]}
    inputs = json.load(open(os.path.join(KPBT, "inputs.json")))
    mm = inputs["matlab"]["michaelis_menten_code"]
    src = "update_uptakes_BTKP_analytical.m (kpbt/inputs.json matlab.michaelis_menten_code)"
    v = lambda d: (float(d["vmax_mmol_per_gDW_h"]), float(d["Km_mM"]), src)  # noqa: E731
    cys = (10.0, 0.01, "paper Methods default for every other metabolite")
    kinetics = {"Kp": {"EX_cpd00027_e0": v(mm["Kp"]["glucose"]), "EX_cpd00007_e0": v(mm["Kp"]["oxygen"]),
                       "EX_cpd00029_e0": v(mm["Kp"]["acetate"]), "EX_cpd00130_e0": v(mm["Kp"]["malate"]),
                       "EX_cpd00084_e0": cys},
                "Bt": {("GLCt4_c2", "GLCt2r_c2"): v(mm["Bt"]["glucose"]), "EX_dextran40_e0": v(mm["Bt"]["dextran"]),
                       "EX_cpd00084_e0": cys}}
    cv = inputs["uptake_convention"]
    reactor = {"D": cv["D_per_h"], "gas": {"cpd00007": (cv["kLa_per_h"], cv["C_star_mM"])},
               "dextran": cv["dextran_feed_mM"], "threshold": inputs["matlab"]["Bt_O2_threshold_mM"],
               "glucose": inputs["reactor"]["feed"]["glucose_mM_by_sample"]}
    return comm, kinetics, reactor, {"extracellular": e0only, "member_bounds": {"Bt": trace}}


def _kpbt_checks():
    comm, kinetics, R, extra = _kpbt_build()
    rule = {"Bt": ConcentrationSwitch("cpd00007", R["threshold"])}
    x_needed = R["gas"]["cpd00007"][0] * (R["gas"]["cpd00007"][1] - R["threshold"]) / (
        20.0 * R["threshold"] / (0.002 + R["threshold"]))

    def run(sample, **kw):
        feed = {"cpd00027": R["glucose"][sample], "dextran40": R["dextran"], "cpd00084": 0.25}
        return chemostat(comm, R["D"], feed, kinetics=kinetics, gas=R["gas"], **extra, **kw)

    # S6, every subset, the O2 rule on
    res = run("S6", switches=rule)
    co = res.of("Kp", "Bt")
    assert len(co) == 1, (res, res.unsolved)
    co = co[0]
    assert round(co.a["Kp"], 3) == 0.798 and abs(co.a["Kp"] - PROTOTYPE_A_KP["S6"]) < 1e-4
    assert co.X["Kp"] == pytest.approx(0.297699, abs=2e-6) and co.X["Bt"] == pytest.approx(0.075287, abs=2e-6)
    assert all(e.real < 0 for e in co.eigenvalues) and -0.21 < co.max_real_eigenvalue < -0.19
    assert co.stable_smooth and not co.stability_ambiguous
    assert not co.exists and not co.stable and co.residuals["cpd00007"] * 1000 == pytest.approx(0.6298, abs=1e-3)
    (crit,) = co.existence
    assert crit["consumers"]["Kp"]["X_needed"] == pytest.approx(x_needed, rel=1e-9)
    assert round(crit["consumers"]["Kp"]["X_needed"], 3) == 0.483 and crit["consumers"]["Kp"]["binding_at_state"]
    assert set(co.binding["Kp"]) == {"EX_cpd00027_e0", "EX_cpd00007_e0"}
    assert "EX_dextran40_e0" in co.binding["Bt"]
    kp = res.of("Kp")[0]
    assert kp.X["Kp"] == pytest.approx(0.091068, abs=2e-6) and kp.exists and kp.stable
    assert kp.invasion["Bt"] == pytest.approx(-R["D"]) and kp.invasion_without_switches["Bt"] == pytest.approx(0.9143, abs=1e-3)
    assert not res.of()[0].stable and not res.of("Bt")[0].exists
    # without the rule the switch is not evaluated: exists (and stability) would follow the smooth
    # field alone, which the toy tests check; the fixed points do not depend on it
    assert co.stable_smooth and co.invasion == {}
    # S4-S8: continued from S6 along the feed in both directions
    got = {"S6": co.a["Kp"]}
    for path in (("S5", "S4"), ("S7", "S8")):
        warm = res
        for sample in path:
            st = run(sample, switches=rule, subsets=[("Kp", "Bt")], warm=warm, stop_after=1)
            (s,) = st.of("Kp", "Bt")
            got[sample] = s.a["Kp"]
            assert abs(s.a["Kp"] - PROTOTYPE_A_KP[sample]) < 1e-4, (sample, s.a["Kp"])
            assert not s.exists and all(e.real < 0 for e in s.eigenvalues)
            assert s.existence[0]["consumers"]["Kp"]["X_needed"] == pytest.approx(x_needed, rel=1e-9)
            assert s.X["Kp"] < x_needed
            warm = st
    assert [round(got[s], 3) for s in PROTOTYPE_A_KP] == [0.848, 0.815, 0.798, 0.783, 0.778]
    print("KPBT_CHECKS_OK", json.dumps({k: round(v, 6) for k, v in got.items()}), flush=True)
