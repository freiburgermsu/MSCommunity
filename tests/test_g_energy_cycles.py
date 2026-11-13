# -*- coding: utf-8 -*-
"""Energy-generating cycles across community members (mscommunity.commcycles).

Two members, each clean on its own: A makes ATP turning nitrate into nitrite, B makes ATP turning
nitrite back into nitrate. Neither can start without the other's product, so single-member tests
pass, but the community makes ATP with every uptake closed. B's direction is uphill by 35 kcal/mol,
so the thermodynamic gate closes it, and both members still grow on glucose.
"""
from __future__ import annotations

import pytest
from cobra import Metabolite, Model, Reaction

from mscommunity.commhelper import build_from_species_models
from mscommunity.commcycles import (cut_energy_cycles, direction_thermo, energy_from_nothing,
                                    find_energy_cycle)

GLC, NO3, NO2 = "cpd00027", "cpd00209", "cpd00075"
UPHILL = {"rxn90001": {"dg": -10.0, "err": 1.0, "source": "test", "reversibility": "="},
          "rxn90002": {"dg": 35.0, "err": 1.0, "source": "test", "reversibility": "="}}
BOTH_DOWNHILL = {"rxn90001": dict(UPHILL["rxn90001"]),
                 "rxn90002": dict(UPHILL["rxn90002"], dg=-10.0)}


def _member(mid, takes, gives, rxn_id):
    m = Model(mid)
    mets = {}
    for cid in (GLC, NO3, NO2, "cpd00002", "cpd00001", "cpd00008", "cpd00009", "cpd00067", "cpd11416"):
        mets[f"{cid}_c0"] = Metabolite(f"{cid}_c0", name=cid, compartment="c0")
    for cid in (GLC, NO3, NO2):
        mets[f"{cid}_e0"] = Metabolite(f"{cid}_e0", name=cid, compartment="e0")
    m.add_metabolites(list(mets.values()))
    c = lambda cid: mets[f"{cid}_c0"]
    rxns = []
    for cid, bound in ((GLC, -10), (NO3, 0), (NO2, 0)):
        ex = Reaction(f"EX_{cid}_e0", lower_bound=bound, upper_bound=1000)
        ex.add_metabolites({mets[f"{cid}_e0"]: -1})
        rxns.append(ex)
    for i, cid in enumerate((GLC, takes, gives)):
        tr = Reaction(f"rxn1000{i}_c0", lower_bound=-1000, upper_bound=1000)
        tr.add_metabolites({mets[f"{cid}_e0"]: -1, c(cid): 1})
        rxns.append(tr)
    # the cycle step: one nitrogen oxide into the other, making ATP
    step = Reaction(f"{rxn_id}_c0", lower_bound=-1000, upper_bound=1000)
    step.add_metabolites({c(takes): -1, c("cpd00008"): -1, c("cpd00009"): -1, c("cpd00067"): -1,
                          c(gives): 1, c("cpd00002"): 1, c("cpd00001"): 1})
    # glycolysis, lumped: glucose pays for its own ATP
    glyc = Reaction("rxn20000_c0", lower_bound=0, upper_bound=1000)
    glyc.add_metabolites({c(GLC): -1, c("cpd00008"): -2, c("cpd00009"): -2, c("cpd00067"): -2,
                          c("cpd00002"): 2, c("cpd00001"): 2})
    bio = Reaction("bio1", lower_bound=0, upper_bound=1000)
    bio.add_metabolites({c(GLC): -1, c("cpd00002"): -5, c("cpd00001"): -5, c("cpd00008"): 5,
                         c("cpd00009"): 5, c("cpd00067"): 5, c("cpd11416"): 1})
    sink = Reaction("SK_cpd11416_c0", lower_bound=0, upper_bound=1000)
    sink.add_metabolites({c("cpd11416"): -1})
    m.add_reactions(rxns + [step, glyc, bio, sink])
    m.objective = "bio1"
    return m


def _members():
    return [_member("reducer", NO3, NO2, "rxn90001"), _member("oxidizer", NO2, NO3, "rxn90002")]


def _community():
    return build_from_species_models(_members(), abundances={"reducer": 0.5, "oxidizer": 0.5})


def _names(comm):
    return {i + 1: mid for i, mid in enumerate(comm.notes["member_biomass_cpds"])}


def _grows(model):
    return model.slim_optimize(error_value=0.0) > 1e-6


def test_each_member_alone_makes_nothing_from_nothing():
    for member in _members():
        res = energy_from_nothing(member, biomass="bio1")
        assert res["atp_from_nothing"] == pytest.approx(0.0, abs=1e-9)
        assert res["growth_from_nothing"] == pytest.approx(0.0, abs=1e-9)


def test_the_community_makes_atp_from_nothing_but_not_growth():
    comm = _community()
    res = energy_from_nothing(comm)
    assert res["atp_from_nothing"] > 1.0
    assert set(res["atp_by_member"]) <= {1, 2}
    # a growth test cannot see it: biomass still needs glucose
    assert res["growth_from_nothing"] == pytest.approx(0.0, abs=1e-9)


def test_the_cycle_is_named_with_its_cross_fed_metabolites():
    comm = _community()
    cyc = find_energy_cycle(comm, thermo=UPHILL, names=_names(comm))
    steps = {r["msid"]: r for r in cyc["reactions"] if r["msid"] in UPHILL}
    assert set(steps) == {"rxn90001", "rxn90002"}
    assert steps["rxn90002"]["infeasible"] and not steps["rxn90001"]["infeasible"]
    assert set(cyc["cross_fed"]) == {f"{NO3}_e0", f"{NO2}_e0"}
    for flows in cyc["cross_fed"].values():
        assert set(flows) == {"reducer", "oxidizer"}


def test_the_gate_uses_the_concentration_range():
    comm = _community()
    rxn = next(r for r in comm.reactions if r.id.startswith("rxn90002"))
    fwd = direction_thermo(rxn, 1, UPHILL["rxn90002"])
    rev = direction_thermo(rxn, -1, UPHILL["rxn90002"])
    # 2 products at 1 uM, 3 substrates at 20 mM (water and H+ left out)
    assert fwd["dg_min"] == pytest.approx(35.0 + 0.59248 * (2 * -13.8155 - 3 * -3.91202), abs=0.01)
    assert fwd["infeasible"] and not rev["infeasible"]


def test_cutting_closes_the_uphill_direction_and_keeps_growth():
    comm = _community()
    assert _grows(comm)
    report = cut_energy_cycles(comm, thermo=UPHILL, protect=lambda m, cuts: _grows(m))
    assert report["status"] == "clean"
    assert {c["msid"] for c in report["cuts"]} == {"rxn90002"}
    assert all(c["direction"] == ">" for c in report["cuts"])
    assert energy_from_nothing(comm, electrons=False)["atp_from_nothing"] == pytest.approx(0.0, abs=1e-9)
    assert _grows(comm)
    step = next(r for r in comm.reactions if r.id.startswith("rxn90002"))
    assert step.bounds == (-1000, 0)


def test_a_cycle_the_gate_cannot_rule_out_is_reported_not_cut():
    comm = _community()
    before = {r.id: r.bounds for r in comm.reactions}
    report = cut_energy_cycles(comm, thermo=BOTH_DOWNHILL)
    assert report["status"] == "feasible-direction cycle"
    assert report["cuts"] == []
    assert {r.id: r.bounds for r in comm.reactions} == before


def test_protect_keeps_a_cut_it_rejects_open():
    comm = _community()
    report = cut_energy_cycles(comm, thermo=UPHILL, protect=lambda m, cuts: False)
    assert report["status"] == "protected"
    assert [k["msid"] for k in report["kept_open"]] == ["rxn90002"]
    step = next(r for r in comm.reactions if r.id.startswith("rxn90002"))
    assert step.bounds == (-1000, 1000)


def test_auto_false_only_proposes():
    comm = _community()
    report = cut_energy_cycles(comm, thermo=UPHILL, auto=False)
    assert report["status"] == "proposed"
    assert [c["msid"] for c in report["proposed"]] == ["rxn90002"]
    assert energy_from_nothing(comm, electrons=False)["atp_from_nothing"] > 1.0


def test_the_local_loop_law_removes_a_cycle_the_gate_cannot():
    from mscommunity.commcycles import lazy_loop_law
    comm = _community()
    assert cut_energy_cycles(comm, thermo=BOTH_DOWNHILL)["status"] == "feasible-direction cycle"
    law, report = lazy_loop_law(comm)
    assert report["status"] == "clean"
    assert report["rounds"][0]["atp_from_nothing"] > 1.0
    assert {"rxn90001_c1", "rxn90002_c2"} <= set(law.binaries)
    assert len(law.pinned) == 2
    # growth on glucose does not need the cycle and survives the law
    assert _grows(comm)
    law.remove()
    assert energy_from_nothing(comm, electrons=False)["atp_from_nothing"] > 1.0


def test_the_atp_pin_stops_a_covered_cycle_feeding_biomass():
    """With glucose open, the cycle's ATP goes to biomass instead of a test reaction. Loop law
    over every reaction does not stop that by itself: the potentials can make ATP synthesis
    downhill. Pinning ATP hydrolysis downhill does, and growth falls to what glycolysis pays
    for: 2 ATP per glucose, 1 glucose + 5 ATP per biomass, 10 glucose -> 10 / 3.5."""
    from mscommunity.commcycles import LocalLoopLaw
    internal = lambda m: [r for r in m.reactions if not r.boundary]
    comm = _community()
    assert comm.slim_optimize() == pytest.approx(10.0)
    law = LocalLoopLaw(comm, pin_atp=False)
    law.add(internal(comm))
    assert comm.slim_optimize() == pytest.approx(10.0)
    law.remove()
    law = LocalLoopLaw(comm, pin_atp=True)
    law.add(internal(comm))
    assert comm.slim_optimize() == pytest.approx(10.0 / 3.5, rel=1e-6)
    with comm:
        law.remove()
        for r in comm.reactions:
            if r.id.startswith(("rxn90001", "rxn90002")):
                r.bounds = (0, 0)
        assert comm.slim_optimize() == pytest.approx(10.0 / 3.5, rel=1e-6)


def test_loop_capable_reactions_are_the_cycle_and_its_transports():
    from mscommunity.commcycles import LocalLoopLaw, loop_capable_reactions
    comm = _community()
    loops = loop_capable_reactions(comm)
    assert {"rxn90001_c1", "rxn90002_c2"} <= loops
    # glucose uptake and glycolysis need the boundary, so no cycle can use them
    assert not {r.id for r in comm.reactions if r.id.startswith(("rxn20000", "rxn10000"))} & loops
    law = LocalLoopLaw(comm)
    law.add([comm.reactions.get_by_id(r) for r in loops])
    assert energy_from_nothing(comm, electrons=False)["atp_from_nothing"] == pytest.approx(0.0, abs=1e-6)
    assert comm.slim_optimize() == pytest.approx(10.0 / 3.5, rel=1e-6)


def test_the_fallback_breaks_a_feasible_direction_cycle_at_its_most_uphill_step():
    thermo = {"rxn90001": dict(UPHILL["rxn90001"]), "rxn90002": dict(UPHILL["rxn90002"], dg=4.0)}
    comm = _community()
    assert cut_energy_cycles(comm, thermo=thermo)["status"] == "feasible-direction cycle"
    report = cut_energy_cycles(comm, thermo=thermo, fallback="most_uphill",
                               protect=lambda m, cuts: _grows(m))
    assert report["status"] == "clean"
    assert {(c["msid"], c["direction"], c["tier"]) for c in report["cuts"]} == {("rxn90002", ">", 2)}
    assert _grows(comm)


def test_one_potential_profile_removes_the_cycle_and_keeps_each_members_growth():
    from mscommunity.commcycles import member_reference, orient_by_potentials, potential_profile
    comm = _community()
    reference, grown = member_reference(comm)
    assert set(grown) == {1, 2} and all(g > 0 for g in grown.values())
    dg, report = potential_profile(comm, reference)
    assert not report["against_reference"] and not report["irreversible_blocked"]
    changed = orient_by_potentials(comm, dg)
    assert changed
    assert energy_from_nothing(comm, electrons=False)["atp_from_nothing"] == pytest.approx(0.0, abs=1e-6)
    # what glycolysis pays for, as under the loop law with the ATP pin
    assert comm.slim_optimize() == pytest.approx(10.0 / 3.5, rel=1e-6)


def test_the_reference_from_member_models_matches_the_one_from_the_community():
    from mscommunity.commcycles import member_reference
    comm = _community()
    from_comm, grown = member_reference(comm)
    from_models, grown2 = member_reference(comm, member_models=dict(enumerate(_members(), 1)))
    assert grown2 == pytest.approx(grown)
    assert set(from_models) == set(from_comm)
    for rid, v in from_comm.items():
        assert from_models[rid] == pytest.approx(v, abs=1e-6)


def test_the_build_warns_and_the_community_class_orients(caplog):
    import logging
    from mscommunity.mscommsim import MSCommunity
    with caplog.at_level(logging.WARNING, logger="mscommunity.commhelper"):
        model = build_from_species_models(_members(), abundances={"reducer": 0.5, "oxidizer": 0.5},
                                          check_energy_cycles=True)
    assert any("ATP with every uptake closed" in r.message for r in caplog.records)
    comm = MSCommunity(model=model, ids=["reducer", "oxidizer"], kinetic_coeff=1e6)
    assert comm.energy_cycles(electrons=False)["atp_from_nothing"] > 1.0
    assert comm.util.model.slim_optimize() == pytest.approx(10.0)
    report = comm.orient_by_thermodynamics(reference="pfba")
    assert comm.energy_cycles(electrons=False)["atp_from_nothing"] == pytest.approx(0.0, abs=1e-6)
    # this optimum ran on the cycle's ATP: the profile orients against that part of the
    # reference and growth falls to what glycolysis pays for
    assert report["profile"]["against_reference"]
    assert comm.util.model.slim_optimize() == pytest.approx(10.0 / 3.5, rel=1e-6)


def test_both_references_keep_every_member_growing():
    from mscommunity.commcycles import orient_community
    comm = _community()
    report = orient_community(comm, reference="both")
    against_members, against_optimum = report["history"][0]["against_reference"]
    # the members' own fluxes hold no cycle and are kept; the optimum's cycle is not
    assert against_members == 0 and against_optimum > 0
    assert energy_from_nothing(comm, electrons=False)["atp_from_nothing"] == pytest.approx(0.0, abs=1e-6)
    assert comm.slim_optimize() == pytest.approx(10.0 / 3.5, rel=1e-6)


def test_the_cutting_plane_loop_law_finds_the_optimum_without_the_cycle():
    from cobra.flux_analysis import pfba
    from mscommunity.commcycles import conformal_cycle, loop_law_optimum, orient_community
    comm = _community()
    lp = {r.id: v for r, v in zip(comm.reactions, pfba(comm).fluxes)}
    assert {"rxn90001_c1", "rxn90002_c2"} <= set(conformal_cycle(comm, lp))
    safe = comm.copy()
    orient_community(safe, reference="both")
    start = pfba(safe).fluxes.to_dict()
    assert conformal_cycle(comm, start) == []
    fluxes, report = loop_law_optimum(comm, start=start)
    assert report["status"] == "no cycle"
    assert report["rounds"][0]["objective"] == pytest.approx(10.0)
    assert report["rounds"][-1]["objective"] == pytest.approx(10.0 / 3.5, rel=1e-6)
    assert conformal_cycle(comm, fluxes) == []


# --------------------------------------------------------------------------- package fixes (Nov 2026)
def _kinetic_community():
    """The nitrate/nitrite community wrapped in MSCommunity, so its *_commKin rows are on the model."""
    from mscommunity.mscommsim import MSCommunity
    model = build_from_species_models(_members(), abundances={"reducer": 0.5, "oxidizer": 0.5})
    return MSCommunity(model=model, ids=["reducer", "oxidizer"], kinetic_coeff=1e6)


def test_the_sealed_tests_lift_the_kinetic_rows_themselves():
    """With the community-kinetics rows on the model every member flux is capped at kappa x its
    growth, which is 0 with nothing coming in: the sealed tests read 0 / "no cycle" / "clean"
    unless they lift the rows themselves (the CommPhitting merges read 0 against a real 250)."""
    from mscommunity.commcycles import (electrons_from_nothing, lazy_loop_law, loop_capable_reactions)
    comm = _kinetic_community()
    model = comm.util.model
    rows = {c.name: (c.lb, c.ub) for c in model.constraints if c.name.endswith("_commKin")}
    assert rows
    res = energy_from_nothing(model, electrons=False)
    assert res["atp_from_nothing"] > 1.0
    assert sorted(res["kinetic_rows_lifted"]) == sorted(rows)
    assert electrons_from_nothing(model)["kinetic_rows_lifted"] == res["kinetic_rows_lifted"]
    cyc = find_energy_cycle(model, thermo=UPHILL)
    assert cyc["status"] == "cycle" and cyc["kinetic_rows_lifted"]
    cut = cut_energy_cycles(model, thermo=UPHILL, auto=False)
    assert cut["status"] == "proposed" and cut["kinetic_rows_lifted"]
    found = {}
    loops = loop_capable_reactions(model, report=found)
    assert {"rxn90001_c1", "rxn90002_c2"} <= loops and found["kinetic_rows_lifted"]
    law, rep = lazy_loop_law(model)
    assert rep["rounds"][0]["atp_from_nothing"] > 1.0 and rep["kinetic_rows_lifted"]
    law.remove()
    # the rows are back as they were, and lift_kinetics=False gives the old reading
    assert {c.name: (c.lb, c.ub) for c in model.constraints if c.name.endswith("_commKin")} == rows
    assert energy_from_nothing(model, electrons=False, lift_kinetics=False)["atp_from_nothing"] == \
        pytest.approx(0.0, abs=1e-9)
    assert comm.energy_cycles(electrons=False)["kinetic_rows_lifted"] == res["kinetic_rows_lifted"]


def _leaky_member():
    """A single model whose ATP comes only from phosphate and protons taken in and pyrophosphate
    sent out (2 Pi + H+ -> PPi + H2O, uphill, run as if it paid for an ATP): the route the sealed
    gate cannot see because it closes every uptake (the 88-member 'both' orientation: 811)."""
    m = _member("leaky", NO3, NO2, "rxn90001")
    PI, PPI, H, H2O = "cpd00009", "cpd00012", "cpd00067", "cpd00001"
    mets = {x.id: x for x in m.metabolites}
    new = []
    for cid in (PI, PPI, H, H2O):
        e = mets.get(f"{cid}_e0") or Metabolite(f"{cid}_e0", name=cid, compartment="e0")
        mets[f"{cid}_e0"] = e
        ex = Reaction(f"EX_{cid}_e0", lower_bound=0, upper_bound=1000)
        ex.add_metabolites({e: -1})
        tr = Reaction(f"rxn9100{len(new)}_c0", lower_bound=-1000, upper_bound=1000)
        tr.add_metabolites({e: -1, mets[f"{cid}_c0"] if f"{cid}_c0" in mets else
                            Metabolite(f"{cid}_c0", name=cid, compartment="c0"): 1})
        new += [ex, tr]
    m.add_reactions(new)
    mets = {x.id: x for x in m.metabolites}
    c = lambda cid: mets[f"{cid}_c0"]
    leak = Reaction("rxn91999_c0", lower_bound=0, upper_bound=100)
    leak.add_metabolites({c("cpd00008"): -1, c(PI): -3, c(H): -2,
                          c("cpd00002"): 1, c(PPI): 1, c(H2O): 2})
    m.add_reactions([leak])
    return m


def test_the_boundary_ion_arms_see_the_h_pi_route_the_sealed_gate_misses():
    m = _leaky_member()
    sealed = energy_from_nothing(m, electrons=False)
    assert sealed["atp_from_nothing"] == pytest.approx(0.0, abs=1e-9)
    res = energy_from_nothing(m, electrons=False, boundary_ions=True)
    assert res["atp_from_nothing"] == pytest.approx(0.0, abs=1e-9)
    arms = res["boundary_ions"]
    assert arms["H+"]["atp"] == pytest.approx(0.0, abs=1e-9)
    assert arms["H+,Pi"]["atp"] == pytest.approx(100.0, rel=1e-6)
    assert arms["H+,Pi"]["boundary"]["EX_cpd00012_e0"] == pytest.approx(100.0, rel=1e-6)
    assert res["atp_from_boundary_ions"] == pytest.approx(100.0, rel=1e-6)
    # an extra ion set opens those uptakes too, on top of H+ and Pi
    more = energy_from_nothing(m, electrons=False, boundary_ions=["cpd00209"])
    assert set(more["boundary_ions"]) == {"H+", "H+,Pi", "ions"}
    assert "cpd00209" in more["boundary_ions"]["ions"]["open"]


def test_net_atp_yield_subtracts_what_the_model_makes_without_the_substrate():
    from mscommunity.commcycles import net_atp_yield
    clean = net_atp_yield(_member("glycolytic", NO3, NO2, "rxn90001"), GLC, acceptor=None)
    # lumped glycolysis: 2 ATP per glucose, nothing without it
    assert clean["gross"] == pytest.approx(2.0, rel=1e-6)
    assert clean["substrate_closed"] == pytest.approx(0.0, abs=1e-9)
    assert clean["net"] == pytest.approx(2.0, rel=1e-6)
    leaky = net_atp_yield(_leaky_member(), GLC, acceptor=None)
    # H+ and Pi are open in the yield test, so the gross yield counts the 100 the leak makes
    assert leaky["gross"] == pytest.approx(102.0, rel=1e-6)
    assert leaky["substrate_closed"] == pytest.approx(100.0, rel=1e-6)
    assert leaky["net"] == pytest.approx(2.0, rel=1e-6)


def test_loop_law_optimum_reports_the_gap_and_a_flux_without_the_integrality_leak():
    from cobra.flux_analysis import pfba
    from mscommunity.commcycles import conformal_cycle, loop_law_optimum, orient_community
    comm = _community()
    safe = comm.copy()
    orient_community(safe, reference="both")
    fluxes, report = loop_law_optimum(comm, start=pfba(safe).fluxes.to_dict())
    assert report["status"] == "no cycle"
    assert "mip_gap" in report and "objective_bound" in report
    assert report["mip_gap"] is None or report["mip_gap"] <= 1e-3
    assert report["clean_objective"] == pytest.approx(10.0 / 3.5, rel=1e-6)
    clean = report["clean_fluxes"]
    assert conformal_cycle(comm, clean) == []
    law = report["law"]
    # every covered reaction the found orientation shuts carries exactly nothing
    for rid, (f, b) in report["clean_orientation"].items():
        if not f and not b:
            assert clean[rid] == 0.0
    # the binaries are free again afterwards
    assert all(v.lb == 0 and v.ub == 1 for fb in law.binaries.values() for v in fb)


def _two_routes():
    """One member, glucose to ATP by two parallel routes of equal cost (via X or via Y): pFBA
    has two optimal vertices, and which one a solver returns depends on where it starts."""
    m = _member("parallel", NO3, NO2, "rxn90001")
    mets = {x.id: x for x in m.metabolites}
    new = []
    for cid in ("cpd00100", "cpd00101"):
        mets[f"{cid}_c0"] = Metabolite(f"{cid}_c0", name=cid, compartment="c0")
    c = lambda cid: mets[f"{cid}_c0"]
    for k, cid in enumerate(("cpd00100", "cpd00101")):
        a = Reaction(f"rxn9300{k}_c0", lower_bound=0, upper_bound=1000)
        a.add_metabolites({c(GLC): -1, c(cid): 1})
        b = Reaction(f"rxn9301{k}_c0", lower_bound=0, upper_bound=1000)
        b.add_metabolites({c(cid): -1, c("cpd00008"): -2, c("cpd00009"): -2, c("cpd00067"): -2,
                           c("cpd00002"): 2, c("cpd00001"): 2})
        new += [a, b]
    m.add_reactions(new)
    m.reactions.get_by_id("rxn20000_c0").bounds = (0, 0)
    m.objective = "bio1"
    return m


def _with_history(model, route):
    """Leave the solver's warm start on one route: maximise it once, then restore the objective."""
    model.objective = f"rxn9301{route}_c0"
    model.slim_optimize()
    model.objective = "bio1"
    return model


@pytest.mark.parametrize("solver", ["glpk", "gurobi"])
def test_robust_pfba_on_a_canonical_copy_does_not_depend_on_the_solver_history(solver):
    from mscommunity.commcycles import robust_pfba
    try:
        a, b = _two_routes(), _two_routes()
        a.solver, b.solver = solver, solver
    except Exception:
        pytest.skip(f"{solver} not available")
    _with_history(a, 0)
    _with_history(b, 1)
    used = lambda sol: tuple(rid for rid in ("rxn93010_c0", "rxn93011_c0") if sol.fluxes[rid] > 1e-9)
    fa, fb = robust_pfba(a, canonical=True, threads=1), robust_pfba(b, canonical=True, threads=1)
    assert used(fa) == used(fb)
    assert fa.objective_value == pytest.approx(fb.objective_value)
    # on the models themselves each solver returns the route its warm start sat on
    # (GLPK and Gurobi 13, Nov 2026): same optimum, different vertex
    pa, pb = robust_pfba(a), robust_pfba(b)
    assert pa.objective_value == pytest.approx(pb.objective_value)
    assert used(pa) == ("rxn93010_c0",) and used(pb) == ("rxn93011_c0",)
