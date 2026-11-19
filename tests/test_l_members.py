# -*- coding: utf-8 -*-
"""Member bookkeeping (mscommunity.commmembers).

The toy community is built the way tests/test_f_biomass_drains.py builds its own: ModelSEED-style
member models merged by build_from_species_models and wrapped in MSCommunity (drains closed,
kinetic rows at kappa 1e6, which never bind here). Each member turns glucose into ATP (2 per
glucose, lumped glycolysis leaving acetate) and needs 1 glucose + 5 ATP per unit of biomass, so
3.5 glucose per unit biomass when maintenance is idle: a member with specific glucose uptake q
grows at q / 3.5.

    memA (c1)       glucose importer rxn10000 and exporter rxn10004 (c1 <-> e0)
    memB (c2, p2)   glucose through its periplasm: rxn10000_p2 written outward (glc_p2 <=> glc_e0,
                    so uptake runs it in REVERSE) and rxn10001_c2 (p2 -> c2); rxn10003_c2, an
                    ATP-driven hydrolysis maltose_e0 -> 2 glucose_e0 coupled to its cytosol (a member
                    reaction that RELEASES glucose into e0, the dextran case); rxn10005_c2, the same
                    hydrolysis with every metabolite in e0 (no member owns it)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from cobra import Metabolite, Model, Reaction

from mscommunity.commcycles import member_index
from mscommunity.commhelper import build_from_species_models
from mscommunity.commkineticpkg import member_kinetic_reactions
from mscommunity.commmembers import (
    cap_member_uptake, close_member_drains, lifted_kinetic_rows, member_biomass_variables, member_caps,
    member_net_exchange, member_rates, member_reactions, member_table, member_transporters,
    member_uptake_expression, open_member_drains, reactions_by_member, refresh_caps,
    remove_member_caps, restore_member_drains, set_member_abundances, set_member_maintenance,
    atp_maintenance_reaction)
from mscommunity.mscommsim import MSCommunity

GLC, AC, MAL, BIO = "cpd00027", "cpd00029", "cpd00179", "cpd11416"
ATP, H2O, ADP, PI, H = "cpd00002", "cpd00001", "cpd00008", "cpd00009", "cpd00067"
GLC_PER_BIOMASS = 3.5


def _member(mid, periplasm=False):
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
    rs = [rxn(f"EX_{GLC}_e0", {(GLC, "e0"): -1}, -10.0), rxn(f"EX_{AC}_e0", {(AC, "e0"): -1}),
          rxn(f"EX_{MAL}_e0", {(MAL, "e0"): -1})]
    if periplasm:
        rs += [rxn("rxn10000_p0", {(GLC, "p0"): -1, (GLC, "e0"): 1}, -1000.0),
               rxn("rxn10001_c0", {(GLC, "p0"): -1, (GLC, "c0"): 1}),
               rxn("rxn10003_c0", {(MAL, "e0"): -1, (GLC, "e0"): 2, **atp(-1)}),
               rxn("rxn10005_c0", {(MAL, "e0"): -1, (GLC, "e0"): 2})]
    else:
        rs += [rxn("rxn10000_c0", {(GLC, "e0"): -1, (GLC, "c0"): 1}),
               rxn("rxn10004_c0", {(GLC, "c0"): -1, (GLC, "e0"): 1})]
    rs += [rxn("rxn10002_c0", {(AC, "c0"): -1, (AC, "e0"): 1}, -1000.0),
           rxn("rxn20000_c0", {(GLC, "c0"): -1, (AC, "c0"): 2, **atp(2)}),
           rxn("rxn00062_c0", atp(-1)),
           rxn("bio1", {(GLC, "c0"): -1, (BIO, "c0"): 1, **atp(-5)}),
           rxn("SK_cpd11416_c0", {(BIO, "c0"): -1})]
    m.add_reactions(rs)
    m.objective = "bio1"
    return m


def _community():
    model = build_from_species_models([_member("memA"), _member("memB", periplasm=True)],
                                      abundances={"memA": 0.5, "memB": 0.5})
    return MSCommunity(model=model, ids=["memA", "memB"], kinetic_coeff=1e6, close_member_drains=True)


def _value(expr):
    return float(expr.subs({v: v.primal for v in expr.free_symbols}))


def _grow(comm):
    model = comm.util.model
    model.objective = comm.primary_biomass.id
    sol = model.optimize()
    assert sol.status == "optimal"
    return sol


# --------------------------------------------------------------------------- membership
def test_member_reactions_partition_the_member_reactions():
    comm = _community()
    model = comm.util.model
    by_member = {m.id: {r.id for r in member_reactions(comm, m)} for m in comm.members}
    a, b = by_member["memA"], by_member["memB"]
    assert not a & b
    owned = {r.id for r in model.reactions if member_index(r) not in (0, None)} - {"bio1"}
    assert a | b == owned
    for ids in (a, b):
        assert "bio1" not in ids and not any(rid.startswith("EX_") for rid in ids)
    assert "rxn10005_c2" not in a | b                    # e0-only: the shared pool's
    assert {"rxn10000_p2", "rxn10001_c2", "rxn10003_c2", "bio3", "SK_cpd11416_c2"} <= b
    assert {"rxn10000_c1", "rxn10004_c1", "rxn00062_c1", "bio2", "SK_cpd11416_c1"} <= a
    # the same sets in one pass, from the bare model by index, and the kinetic row's membership
    assert {k: {r.id for r in v} for k, v in reactions_by_member(comm).items()} == by_member
    assert {r.id for r in member_reactions(model, 2)} == b
    for m in comm.members:
        assert ({r.id for r in member_reactions(comm, m, include_biomass=False)}
                == {r.id for r in member_kinetic_reactions(comm, m)})


def test_member_transporters_by_compound():
    comm = _community()
    ids = lambda rs: {r.id for r in rs}  # noqa: E731
    assert ids(member_transporters(comm, "memA", GLC)) == {"rxn10000_c1", "rxn10004_c1"}
    assert ids(member_transporters(comm, "memB", f"{GLC}_e0")) == {"rxn10000_p2", "rxn10003_c2"}
    assert ids(member_transporters(comm, 2, MAL)) == {"rxn10003_c2"}
    assert member_transporters(comm, "memA", MAL) == []
    assert "rxn10002_c1" in ids(member_transporters(comm, "memA"))
    with pytest.raises(KeyError):
        member_transporters(comm, "memA", "cpd99999")


# --------------------------------------------------------------------------- uptake
def test_uptake_expressions_evaluate_to_the_pinned_fluxes():
    comm = _community()
    model = comm.util.model
    pins = {"rxn10000_c1": 6.0, "rxn10004_c1": 2.0, "rxn10000_p2": -5.0, "rxn10003_c2": 1.0}
    for rid, v in pins.items():
        model.reactions.get_by_id(rid).bounds = (v, v)
    model.reactions.get_by_id(f"EX_{MAL}_e0").lower_bound = -1.0
    sol = _grow(comm)

    expr = lambda *a, **k: _value(member_uptake_expression(comm, *a, **k))  # noqa: E731
    assert expr("memA", GLC) == pytest.approx(4.0)                     # 6 in, 2 out
    assert expr("memA", GLC, gross=True) == pytest.approx(6.0)
    assert expr("memB", GLC) == pytest.approx(3.0)                     # 5 in (reverse), 2 released
    assert expr("memB", GLC, gross=True) == pytest.approx(5.0)
    assert expr("memB", GLC, reactions=["rxn10000_p2"]) == pytest.approx(5.0)
    assert expr("memB", MAL) == pytest.approx(1.0)

    net_a, net_b = member_net_exchange(comm, sol, "memA"), member_net_exchange(comm, sol.fluxes, "memB")
    assert net_a[GLC] == pytest.approx(4.0) and net_b[GLC] == pytest.approx(3.0)
    assert net_b[MAL] == pytest.approx(1.0) and MAL not in net_a
    assert net_a[AC] == pytest.approx(-2.0 * sol.fluxes["rxn20000_c1"]) and net_a[AC] < 0
    # the members' net uptakes add up to what the community takes in (rxn10005 idle)
    assert sol.fluxes["rxn10005_c2"] == pytest.approx(0.0, abs=1e-9)
    assert net_a[GLC] + net_b[GLC] == pytest.approx(-sol.fluxes[f"EX_{GLC}_e0"])


def test_uptake_expression_refuses_what_is_not_a_member_uptake():
    comm = _community()
    with pytest.raises(ValueError, match="no transporter"):
        member_uptake_expression(comm, "memA", MAL)
    with pytest.raises(ValueError, match="boundary"):
        member_uptake_expression(comm, "memB", GLC, reactions=[f"EX_{GLC}_e0"])
    with pytest.raises(ValueError, match="not to memB"):
        member_uptake_expression(comm, "memB", GLC, reactions=["rxn10000_c1"])
    with pytest.raises(ValueError, match="does not move"):
        member_uptake_expression(comm, "memB", GLC, reactions=["rxn10001_c2"])
    with pytest.raises(ValueError):
        member_uptake_expression(comm, "memB", None)
    # an e0-only reaction may be counted for a member when the caller says so
    expr = member_uptake_expression(comm, "memB", None, reactions=["rxn10005_c2"], gross=True)
    assert {v.name for v in expr.free_symbols} == {comm.util.model.reactions.rxn10005_c2.forward_variable.name}


# --------------------------------------------------------------------------- caps
def test_abundance_cap_binds_and_follows_refresh_caps():
    comm = _community()
    model = comm.util.model
    q = 2.0
    row = cap_member_uptake(comm, "memA", GLC, q=q)
    assert row.name == f"mcap_memA_{GLC}" and row.ub == pytest.approx(q * 0.5)
    assert not {v.name for v in row.variables} & {model.reactions.get_by_id(f"EX_{GLC}_e0").forward_variable.name,
                                                  model.reactions.get_by_id(f"EX_{GLC}_e0").reverse_variable.name}
    sol = _grow(comm)
    assert sol.fluxes["rxn10000_c1"] == pytest.approx(q * 0.5)          # the cap binds
    assert sol.objective_value == pytest.approx(q / GLC_PER_BIOMASS)  # specific uptake q -> mu = q / 3.5

    # set_abundance alone leaves the row at the old a_m: memA keeps 1.0 mmol for a quarter of the biomass
    comm.set_abundance({"memA": {"abundance": 0.25}, "memB": {"abundance": 0.75}})
    assert _grow(comm).objective_value == pytest.approx(q * 0.5 / GLC_PER_BIOMASS / 0.25)
    assert refresh_caps(comm) == [row.name]
    sol = _grow(comm)
    assert model.constraints[row.name].ub == pytest.approx(q * 0.25)
    assert sol.fluxes["rxn10000_c1"] == pytest.approx(q * 0.25)
    assert sol.objective_value == pytest.approx(q / GLC_PER_BIOMASS)

    # set_member_abundances does both; caps are listed and removed through the registry
    assert set_member_abundances(comm, {"memA": 3, "memB": 1}) == {"memA": 0.75, "memB": 0.25}
    assert model.constraints[row.name].ub == pytest.approx(q * 0.75)
    assert [c.name for c in member_caps(comm, "memA")] == [row.name]
    assert remove_member_caps(comm, cpd=GLC) == [row.name]
    assert row.name not in model.constraints and member_caps(comm) == []


def test_X_cap_follows_the_supplied_variable():
    comm = _community()
    model = comm.util.model
    X = member_biomass_variables(comm)
    assert set(X) == {"memA", "memB"} and member_biomass_variables(comm)["memA"] is X["memA"]
    X["memA"].lb, X["memA"].ub = 0.3, 0.3
    row = cap_member_uptake(comm, "memA", GLC, q=2.0, basis="X", X=X)
    assert row.ub == 0.0 and row.get_linear_coefficients([X["memA"]])[X["memA"]] == pytest.approx(-2.0)
    model.objective = model.problem.Objective(member_uptake_expression(comm, "memA", GLC, gross=True),
                                              direction="max")
    with lifted_kinetic_rows(comm) as lifted:
        assert sorted(lifted) == ["memA_commKin", "memB_commKin"]
        assert model.slim_optimize() == pytest.approx(0.6)
        X["memA"].lb = X["memA"].ub = 0.1
        assert model.slim_optimize() == pytest.approx(0.2)
        set_member_abundances(comm, {"memA": 0.9, "memB": 0.1})          # nothing to refresh on X
        assert model.slim_optimize() == pytest.approx(0.2)
    assert model.constraints["memA_commKin"].ub == 0
    with pytest.raises(ValueError, match="needs X"):
        cap_member_uptake(comm, "memB", GLC, q=1.0, basis="X")
    stray = model.problem.Variable("not_in_model")
    with pytest.raises(ValueError, match="not a variable of the model"):
        cap_member_uptake(comm, "memB", GLC, q=1.0, basis="X", X=stray)


def test_caps_added_inside_a_model_context_go_with_it():
    comm = _community()
    model = comm.util.model
    kept = cap_member_uptake(comm, "memB", GLC, q=1.0)
    with model:
        cap_member_uptake(comm, "memA", GLC, q=2.0)
        remove_member_caps(comm, "memB")
        assert [c.member for c in member_caps(comm)] == ["memA"]
    assert [c.name for c in member_caps(comm)] == [kept.name]
    assert f"mcap_memA_{GLC}" not in model.constraints and kept.name in model.constraints


# --------------------------------------------------------------------------- maintenance
def test_set_member_maintenance_raises_without_an_atpm():
    comm = _community()
    comm.util.model.remove_reactions(["rxn00062_c1"])
    assert atp_maintenance_reaction(comm, "memA", required=False) is None
    with pytest.raises(LookupError, match="no ATP maintenance"):
        set_member_maintenance(comm, "memA", 1.0)
    assert member_caps(comm) == []
    assert atp_maintenance_reaction(comm, "memB").id == "rxn00062_c2"


def test_maintenance_row_replaces_the_constant_floor():
    comm = _community()
    model = comm.util.model
    atpm = model.reactions.get_by_id("rxn00062_c1")
    atpm.lower_bound = 0.3
    A = 2.0
    row = set_member_maintenance(comm, "memA", A)
    assert row.name == "mmaint_memA" and row.lb == pytest.approx(A * 0.5) and atpm.lower_bound == 0.0
    sol = _grow(comm)
    # 10 glucose: 3.5 per unit biomass, plus memA's maintenance at 2 ATP per glucose
    assert sol.fluxes["rxn00062_c1"] == pytest.approx(A * 0.5)
    assert sol.objective_value == pytest.approx((10.0 - A * 0.5 / 2) / GLC_PER_BIOMASS)
    set_member_abundances(comm, {"memA": 0.8, "memB": 0.2})
    assert model.constraints["mmaint_memA"].lb == pytest.approx(A * 0.8)
    assert remove_member_caps(comm, kind="maintenance") == ["mmaint_memA"]
    assert atpm.lower_bound == pytest.approx(0.3)
    X = member_biomass_variables(comm)
    row = set_member_maintenance(comm, "memA", A, basis="X", X=X["memA"])
    assert row.lb == 0.0 and row.get_linear_coefficients([X["memA"]])[X["memA"]] == pytest.approx(-A)


# --------------------------------------------------------------------------- rates and drains
def test_member_rates_equal_mu_com_with_drains_closed_and_differ_when_open():
    comm = _community()
    model = comm.util.model
    sol = _grow(comm)
    assert all(mu == pytest.approx(sol.objective_value) for mu in member_rates(comm, sol).values())

    q_b = 1.75                                         # memB limits the community at mu = 0.5
    cap_member_uptake(comm, "memB", GLC, q=q_b)
    sol = _grow(comm)
    mu_com = sol.objective_value
    assert mu_com == pytest.approx(q_b / GLC_PER_BIOMASS)
    assert all(mu == pytest.approx(mu_com) for mu in member_rates(comm, sol).values())

    previous = open_member_drains(comm)
    assert previous == {"SK_cpd11416_c1": (0, 0), "SK_cpd11416_c2": (0, 0)}
    model.reactions.get_by_id(comm.primary_biomass.id).bounds = (mu_com, mu_com)
    model.objective = "bio2"                           # memA grows on the glucose memB cannot take
    sol = model.optimize()
    rates = member_rates(comm, sol)
    assert rates["memB"] == pytest.approx(mu_com)
    assert rates["memA"] > mu_com + 1.0
    assert rates["memA"] == pytest.approx(mu_com + sol.fluxes["SK_cpd11416_c1"] / 0.5)
    assert member_rates(comm, sol, abundances={"memA": 1.0, "memB": 1.0})["memA"] == pytest.approx(
        sol.fluxes["bio2"])
    restore_member_drains(comm, previous)
    assert all(model.reactions.get_by_id(r).bounds == (0, 0) for r in previous)
    open_member_drains(comm, bounds=(0.0, 5.0))
    assert close_member_drains(comm) == {"SK_cpd11416_c1": (0.0, 5.0), "SK_cpd11416_c2": (0.0, 5.0)}


def test_member_rates_refuse_a_failed_solve():
    comm = _community()
    model = comm.util.model
    model.reactions.get_by_id(f"EX_{GLC}_e0").bounds = (0.0, 0.0)
    model.reactions.get_by_id(comm.primary_biomass.id).lower_bound = 1.0
    with pytest.raises(ValueError, match="not optimal"):
        member_rates(comm, model.optimize())


# --------------------------------------------------------------------------- reporting
def test_member_table_reports_compartments_transporters_and_the_shared_pool():
    comm = _community()
    cap_member_uptake(comm, "memA", GLC, q=2.0)
    table = member_table(comm)
    a, b = table["memA"], table["memB"]
    assert (a.index, a.biomass, a.drain, a.drain_bounds, a.atp_maintenance) == (
        1, "bio2", "SK_cpd11416_c1", (0, 0), "rxn00062_c1")
    assert a.compartments == ("c1",) and b.compartments == ("c2", "p2")
    assert b.transporters[GLC] == ("rxn10000_p2", "rxn10003_c2") and b.transporters[MAL] == ("rxn10003_c2",)
    assert b.extracellular == ("rxn10005_c2",) and a.extracellular == ()
    assert table.shared == ("rxn10005_c2",) and table.unassigned == ()
    assert table.community_biomass == "bio1"
    assert [c.name for c in table.caps] == [f"mcap_memA_{GLC}"]
    assert a.n_reactions == len(member_reactions(comm, "memA"))
    json.dumps(table.as_dict())
    assert list(table.to_frame().index) == ["memA", "memB"]


# --------------------------------------------------------------------------- a real community
ECOLI = "/home/freiburger/Documents/CommPhitting/ecoli.json"
PF = "/home/freiburger/Documents/CommPhitting/pf.json"


@pytest.mark.skipif(not (os.path.exists(ECOLI) and os.path.exists(PF)),
                    reason="the CommPhitting member models are not on this machine")
def test_commphitting_maltose_community_partitions_cleanly():
    """E. coli + P. fluorescens merged as 3H11_R12_SynCom/Metabolic_modeling/
    multisystem3_sys_commphitting_maltose_lib.build does (build_raw at the mid-exponential a_ec,
    MSCommunity with drains closed and kappa 1e6). The campaign's medium, knockouts and replayed
    bounds change bounds only, not membership, so they are left out.

    The checks run in a fresh interpreter. Once a process has called icecream's ic() (mscommsim.py
    does, e.g. in _compute_relative_abundance_from_solution and _set_solution, which earlier tests
    reach), icecream's `executing` has tagged every node of the calling module's AST with .parent,
    including the ast.Load() instance CPython shares between all parsed trees; from then on cobra
    deep-copies that whole module AST with every gene-reaction rule it parses (GPR.__init__). In
    the full session, loading and merging these two models sat in that deepcopy for over 30 min
    and filled a 16 GiB memory cap, against about 15 s in a fresh process. The subprocess keeps
    that state out, and its timeout turns a stall into a failure instead of a hung suite."""
    tests = os.path.dirname(os.path.abspath(__file__))
    code = f"import sys; sys.path[:0] = [{tests!r}, {os.path.dirname(tests)!r}]; " \
           "import test_l_members as t; t._maltose_checks()"
    try:
        run = subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(tests), capture_output=True,
                             text=True, timeout=900)
    except subprocess.TimeoutExpired:
        pytest.fail("building the CommPhitting maltose community took over 900 s in a fresh interpreter")
    assert run.returncode == 0, run.stderr[-6000:]
    assert "MALTOSE_CHECKS_OK" in run.stdout, run.stdout[-2000:]


def _maltose_checks():
    import cobra
    models = []
    for name, path in (("ecoli", ECOLI), ("pf", PF)):
        m = cobra.io.load_json_model(path)
        m.id = name
        models.append(m)
    a_ec = 0.8704
    merged = build_from_species_models(models, model_id="maltose", name="maltose",
                                       abundances={"ecoli": a_ec, "pf": 1 - a_ec}, commkinetics=False)
    abund = {"ecoli": {"abundance": a_ec, "biomass_compound": merged.metabolites.get_by_id("cpd11416_c1")},
             "pf": {"abundance": 1 - a_ec, "biomass_compound": merged.metabolites.get_by_id("cpd11416_c2")}}
    comm = MSCommunity(model=merged, abundances=abund, ID="maltose", kinetic_coeff=1e6, close_member_drains=True)
    model = comm.util.model

    owned = reactions_by_member(comm)
    seen = [r.id for rxns in owned.values() for r in rxns]
    assert len(seen) == len(set(seen))                                   # no reaction in two members
    table = member_table(comm)
    rest = {r.id for r in model.reactions} - set(seen)
    exchanges = {r.id for r in model.reactions if r.boundary and member_index(r) == 0}
    assert table.unassigned == ()
    assert rest == exchanges | set(table.shared) | {table.community_biomass}   # each reaction in one class
    assert all(member_index(model.reactions.get_by_id(rid)) == m.index
               for m in comm.members for rid in (r.id for r in owned[m.id]))

    maltose = [r.id for r in member_transporters(comm, "ecoli", MAL)]
    assert maltose and "rxn05170_c1" in maltose
    assert all(rid.endswith("_c1") for rid in maltose)
    assert table["ecoli"].transporters[MAL] == tuple(maltose)

    # at an optimum, the members' net uptakes and the boundary fluxes balance every e0 compound
    sol = model.optimize()
    assert sol.status == "optimal"
    nets = {m.id: member_net_exchange(comm, sol, m, tol=0.0) for m in comm.members}
    shared_mets = {met.id for rid in table.shared for met in model.reactions.get_by_id(rid).metabolites}
    checked = 0
    for met in model.metabolites:
        if met.compartment != "e0" or met.id in shared_mets:
            continue
        boundary = sum(float(c) * sol.fluxes[r.id] for r in met.reactions if r.boundary
                       for m, c in r.metabolites.items() if m.id == met.id)
        members = sum(n.get(met.id[:-3], 0.0) for n in nets.values())
        assert members == pytest.approx(boundary, abs=1e-6)
        checked += 1
    assert checked > 50
    print(f"MALTOSE_CHECKS_OK {len(seen)} member reactions, {checked} e0 balances", flush=True)


def test_set_member_abundances_inside_a_context_restores_the_cap_rows():
    """cobra does not track optlang row bounds, so the cap / maintenance rows refresh_caps rewrote kept the
    context's composition after `with model:` while the abundances and bio1 reverted."""
    comm = _community()
    model = comm.util.model
    cap = cap_member_uptake(comm, "memA", GLC, q=7.0)
    maint = set_member_maintenance(comm, "memB", 1.0)
    before = {name: (model.constraints[name].lb, model.constraints[name].ub) for name in (cap.name, maint.name)}
    assert before[cap.name][1] == pytest.approx(3.5) and before[maint.name][0] == pytest.approx(0.5)
    with model:
        set_member_abundances(comm, {"memA": 0.9, "memB": 0.1})
        assert model.constraints[cap.name].ub == pytest.approx(6.3)
        assert model.constraints[maint.name].lb == pytest.approx(0.1)
    assert {name: (model.constraints[name].lb, model.constraints[name].ub) for name in before} == before
    assert comm.members.memA.abundance == pytest.approx(0.5)
    # a bare refresh_caps inside a context is undone too
    comm.members.memA.abundance = 0.2
    with model:
        refresh_caps(comm)
        assert model.constraints[cap.name].ub == pytest.approx(1.4)
    assert model.constraints[cap.name].ub == pytest.approx(3.5)
