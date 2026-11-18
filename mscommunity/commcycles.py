# -*- coding: utf-8 -*-
"""Energy-generating cycles that run across the members of a community model.

A member can be free of energy-generating cycles on its own and still close one with another
member: A turns X into Y and makes ATP, B turns Y back into X and makes ATP, and neither can
start without the other's product. A test on a single member cannot see such a cycle, because
the member never receives X or Y; it shows only in the assembled community, as ATP (or reduced
products) made with every uptake closed.

Finding them
    energy_from_nothing   the test: ATP, growth and electrons with nothing (or only CO2, H2O and
                          H+) coming in
    find_energy_cycle     the cycle a solver finds: its reactions by member, each one's direction
                          against the ModelSEED database and thermodynamics, and the metabolites
                          the members pass around it
    conformal_cycle       the cycle inside a given flux distribution, if any (one LP)

Removing them -- tried on the nine members of the codiffusion day-174 cycle and on the whole
88-member community (pilots/energy_cycles/README.md has the numbers):
    loop_law_optimum      the exact loop law (LocalLoopLaw, a MILP) on just the cycles the
                          optimum tries to use: solve, find a cycle in the solution, cover it,
                          solve again, warm-started from a cycle-free flux. The pilot's loop-law
                          optimum in 110 s, where the loop law on every loop-capable reaction
                          (loop_capable_reactions) found nothing above 0 in 30 min unstarted.
    orient_community      one metabolite-potential profile for the whole community, with ATP
                          hydrolysis pinned downhill; every reaction keeps only the directions the
                          profile allows. Provably leaves no ATP-generating cycle, including
                          one that feeds biomass (not one charging another nucleotide: see
                          "Not covered"); LPs only. It keeps a reference flux wherever that
                          holds no cycle: on a loop_law_optimum solution it keeps the loop-law
                          optimum and turns every later simulation into an ordinary cycle-free LP.
                          Without one: "pfba" (the community's own optimum) keeps more when that
                          optimum holds few cycles (9 members: 19.61 of 19.91), "both" (each
                          member alone, then the optimum) when it leans on them (88 members:
                          11.83 vs 6.17 of 18.21), and "both" always keeps every member growing.
    cut_energy_cycles     closes, cycle by cycle, the directions thermodynamics rules out (the gate
                          below). It finds template curations (directions no concentrations could
                          drive), but cannot finish: what is left runs on steps feasible one by
                          one, and its "most_uphill" fallback over-cuts without converging.

Thermodynamic gate. A direction is ruled out only when no metabolite concentrations in
[c_min, c_max] could make it run:

    dG'min = dG'0 + RT * (sum_products n ln c_min - sum_substrates n ln c_max) > n_sigma * err

with water and protons left out of the sums. dG'0 and its error come from the ModelSEED
database's "thermodynamics" field (eQuilibrator, else dGPredictor, else group contribution;
kcal/mol). Reactions that cross a membrane are not gated: their dG depends on the membrane
potential, which the database values leave out. The defaults (1 uM - 20 mM, 2 sigma) gate only
directions that are uphill by more than any plausible concentration ratio can pay for.

Not covered by default: the pin is on ATP hydrolysis only, so "energy-generating" above means
ATP-generating. A cycle that charges another nucleotide -- GTP, say -- and hands it straight to
biomass is not excluded, not even where nucleoside-diphosphate kinase links GTP to ATP: the
potentials can put GTP's phosphorylation potential below ATP's, which orients the kinase
ATP -> GTP and leaves the GTP loop downhill. The link only lets energy_from_nothing see such a
loop (as ATP made through the kinase) and lets the pin stop that ATP-making use. In the
planted-cycle validation (tests/test_i_cycle_validation.py, 50 seeds per type) the ATP-pinned loop
law exceeded the ATP+GTP-pinned optimum on every GTP-loop instance, linked or not (growth up to
1.28x and 1.56x), and orient_community ("pfba", "both") kept all of that excess on the unlinked
loops and part of it on the linked ones (mean 1.18x; 1.04-1.06x). pins="ntp" (LocalLoopLaw,
loop_law_optimum, potential_profile, orient_community, conformal_cycle, energy_from_nothing) pins
GTP, CTP, UTP and pyrophosphate hydrolysis downhill as well and adds their hydrolysis tests; it is
opt-in because it changes the default orientations and costs a few more rows.

The sealed tests close every uptake, so they cannot see a route that draws free energy through a
boundary ion -- phosphate and protons taken in, pyrophosphate sent out, say (811 mmol ATP/gDW/h on
one 88-member orientation that passes the sealed gate). energy_from_nothing(boundary_ions=True)
adds arms that reopen H+ and Pi (and optionally the medium's inorganic uptakes), and net_atp_yield
gives the ATP yield of a substrate net of what the same test makes without it. Electrons made by
reactions that do not conserve charge are a curation problem, not a cycle -- electrons_from_nothing
reports them, nothing here removes them.

Reproducibility. orient_community, cut_energy_cycles, robust_pfba(canonical=True) and
member_reference(canonical=True) solve on canonical_copy(model): a copy whose rows are rebuilt in
reaction order, started from a defined state (GLPK's standard basis; a fresh Gurobi model with
Threads, Presolve and Method set explicitly). Which optimal vertex a simplex returns follows its
start and the stored order of the rows' terms, and the profile honours whichever pFBA vertex it is
given; on the Kp/Bt merges three fresh builds gave three orientations per reference before this.
It makes the result reproducible for one solver configuration and thread count, not better: the
vertex it reproduces is no more right than the others.

Where the code lives. The single-model audit -- the hydrolysis tests and their pins, sealing
(lax / strict), the floor release, energy_from_nothing, electrons_from_nothing, net_atp_yield,
the cycle LP of find_energy_cycle, the dG gate and conformal_cycle -- is
modelseedpy.core.msenergyaudit. This module calls it with what a community adds: a test in
each member cytosol rather than the community compartment c0, results by member index, the
*_commKin rows lifted (lift_kinetics), and a cycle's members and the metabolites they pass
around it. A ModelSEEDpy without that module gets one warning and MSCommunity's own copy
(_fallback_energy_audit, at the end of this file), which is to be removed once the
ModelSEEDpy MSCommunity needs has the module.
"""
import importlib
import logging
import re
import warnings
from math import isnan
from types import SimpleNamespace

from cobra import Reaction
from optlang.symbolics import Zero

logger = logging.getLogger(__name__)

ORIENT_TOL = 1e-5  # |dG| of a potential profile below this (x its margin) is 0: solver tolerance
KINETIC_ROW_SUFFIX = "_commKin"
_COMP = re.compile(r"^([a-z]+)(\d+)$")
# The generic constants and functions (ATP, NTP_PINS, close_uptakes, conformal_cycle, ...) are
# bound at the end of this file, from modelseedpy.core.msenergyaudit or the fallback copy.


# --------------------------------------------------------------------------- model helpers
def member_index(rxn):
    """The member a community reaction belongs to.

    build_from_species_models gives member i's compartments the index i (c3, p3, ...) and the shared
    pool and the community biomass the index 0. Returns i for a member's reaction, 0 for a reaction
    in the shared pool or the community compartment only, None for one spanning several members
    (the community biomass reaction)."""
    idx = set()
    for met in rxn.metabolites:
        m = _COMP.match(met.compartment or "")
        if m and m.group(1) != "e":
            idx.add(int(m.group(2)))
    idx.discard(0)
    if not idx:
        return 0
    return idx.pop() if len(idx) == 1 else None


def _cpd(met):
    return met.id.split("_")[0]


def _value(model):
    v = model.slim_optimize(error_value=float("nan"))
    return None if isnan(v) else v


# --------------------------------------------------------------------------- the audit, by member
def _member_cytosols(model, pins):
    """compartments= for the generic tests: every member cytosol c<i> that holds a pin's
    metabolites, and the community compartment c0 only where it is the one compartment holding
    them -- c0 is the community compartment in a community and the cytosol in a single model."""
    found = {}
    for tri, comp, _ in hydrolysis_sites(model, pins):
        m = _COMP.match(comp)
        if m:
            found.setdefault(tri, []).append((comp, int(m.group(2))))
    keep = set()
    for tri, comps in found.items():
        if len(comps) > 1:
            comps = [(comp, idx) for comp, idx in comps if idx != 0]
        keep.update((tri, comp) for comp, _ in comps)
    return lambda tri, comp: (tri, comp) in keep


def _member_key(key):
    """A generic test key ("c3", or (compound, "c3")) as a member key (3, or (compound, 3))."""
    if isinstance(key, tuple):
        return key[0], int(_COMP.match(key[1]).group(2))
    return int(_COMP.match(key).group(2))


def _by_member(report):
    """A generic audit report in this module's terms, its key order kept: rows_lifted becomes
    kinetic_rows_lifted, atp_by_compartment {"c3": flux} becomes atp_by_member {3: flux}."""
    out = {}
    for k, v in report.items():
        if k == "rows_lifted":
            out["kinetic_rows_lifted"] = v
        elif k == "atp_by_compartment":
            out["atp_by_member"] = {_member_key(c): f for c, f in v.items()}
        else:
            out[k] = v
    return out


def lifted_kinetic_rows(model, lift=True, suffix=KINETIC_ROW_SUFFIX):
    """Lift the community-kinetics rows (names ending in `suffix`, as MSCommunity.add_commkinetics
    and CommKineticPkg write them) for the duration of the block: they cap each member's flux at
    kappa x its growth, so with nothing coming in they hold every flux at 0 and hide every cycle
    -- which still runs, up to that cap, whenever the community grows. Bounds go to +-1e30, not
    None (optlang's Gurobi interface refuses a free row), and come back on exit. `with model:`
    does not track row bounds, so this restores them itself. Yields the lifted rows' names.
    (msenergyaudit.lifted_rows with the suffix.)"""
    return _ea.lifted_rows(model, suffix if lift else None)


def add_atp_tests(model, pins=None):
    """One ATP hydrolysis reaction per member compartment that has ATP, ADP, phosphate, water and
    H+ (ATP + H2O -> ADP + Pi + H+, id ATPtest_c<i>). Call inside `with model:` so they go again.
    Returns {member index: reaction}.

    pins="ntp" (or a list of (triphosphate, diphosphate) pairs, see NTP_PINS) adds a hydrolysis
    test for each other pinned compound too (GTP, CTP, UTP; PPi + H2O -> 2 Pi + H+), id
    NTPtest_<compound>_c<i>, keyed (compound, member index) in the returned dict. A GTP loop
    that feeds biomass is invisible to the ATP tests alone. (msenergyaudit.add_hydrolysis_tests
    on the member cytosols.)"""
    pins = _resolve_pins(pins)
    tests = _ea.add_hydrolysis_tests(model, pins, _member_cytosols(model, pins))
    return {_member_key(k): r for k, r in tests.items()}


def energy_from_nothing(model, biomass="bio1", electrons=True, tol=1e-6, release_floors=True,
                        closure="lax", biomass_outlet=True, lift_kinetics=True, boundary_ions=False,
                        pins=None, formulas=None):
    """What the community makes from nothing (msenergyaudit.energy_from_nothing, by member).

    atp_from_nothing        max total ATP hydrolysis over the members with every uptake closed
                            (with pins="ntp": ATP, GTP, CTP, UTP and PPi hydrolysis summed)
    atp_by_member           {member index: flux} at that optimum (members above `tol`)
    ntp_by_test             {"<compound>_c<i>": flux}, the other hydrolysis tests (pins="ntp")
    growth_from_nothing     max `biomass` flux with every uptake closed (skipped if it is None)
    electrons_from_nothing  max electrons exported in REDUCED_PRODUCTS with only CO2, H2O and H+
                            coming in; a clean community makes none, since reducing CO2 with
                            electrons from water needs an energy source
    released                {reaction id: old bounds} of the forced fluxes released for the test
    kinetic_rows_lifted     names of the community-kinetics rows lifted for the test
    All three are 0 for a community without energy-generating cycles. None means the LP failed.
    Fluxes in mmol/gDW/h.

    release_floors  first let every forced non-boundary flux stop (release_forced_fluxes):
                    a positive lower bound (a maintenance-ATP floor, a measured rate fixed as an
                    equality) goes to 0, and so does a negative upper bound. A floor cannot run
                    on nothing, so with it in place the test is infeasible, not 0. Only bounds
                    change; constraint rows are lifted by lift_kinetics. False keeps the bounds
                    as they are (the earlier behaviour).
    closure         "lax": every boundary reaction may still remove matter (the default and the
                    earlier behaviour); "strict": every boundary reaction closed both ways. They
                    differ only where reactions make matter from nothing, which "lax" lets out.
    biomass_outlet  with "strict", keep the biomass outlets able to remove biomass; False makes
                    growth_from_nothing 0 by construction (the electron test keeps its own
                    closure: CO2, H2O and H+ in, anything out).
    lift_kinetics   lift the *_commKin rows for the tests (lifted_kinetic_rows). With them in
                    place every member flux is capped at kappa x its growth, 0 on nothing, and the
                    test reads 0 on a community that makes ATP at the flux bounds (the CommPhitting
                    merges: 0 against 250). False keeps them (the earlier behaviour).
    boundary_ions   False (default); True adds arms that reopen H+ ("H+") and H+ with phosphate
                    ("H+,Pi") and report the ATP made through them, the boundary fluxes at that
                    optimum and atp_from_boundary_ions (the largest). A clean community makes none:
                    taking in Pi and H+ and sending out, say, pyrophosphate is uphill. The sealed
                    test cannot see such a route, because it closes every uptake. "medium" adds an
                    arm "ions" with the medium's inorganic uptakes too (medium_inorganic_uptakes,
                    with `formulas` for metabolites that carry none); a list of compounds adds that
                    arm with those compounds.
    pins            None (ATP tests only) or "ntp": hydrolysis tests for GTP, CTP, UTP and PPi too
                    (add_atp_tests)."""
    pins = _resolve_pins(pins)
    out = _ea.energy_from_nothing(
        model, biomass=biomass, electrons=electrons, tol=tol, release_floors=release_floors,
        closure=closure, biomass_outlet=biomass_outlet,
        lift_rows=KINETIC_ROW_SUFFIX if lift_kinetics else None, boundary_ions=boundary_ions,
        pins=pins, formulas=formulas, compartments=_member_cytosols(model, pins))
    if "boundary_ions" in out:
        out["boundary_ions"] = {label: _by_member(arm) for label, arm in out["boundary_ions"].items()}
    return _by_member(out)


def electrons_from_nothing(model, release_floors=True, lift_kinetics=True):
    """Max electrons exported in REDUCED_PRODUCTS with only CO2, H2O and H+ coming in, and the
    reduced products carrying them ({compound: export flux}). release_floors and lift_kinetics as
    in energy_from_nothing. (msenergyaudit.electrons_from_nothing.)"""
    return _by_member(_ea.electrons_from_nothing(
        model, release_floors=release_floors, lift_rows=KINETIC_ROW_SUFFIX if lift_kinetics else None))


def net_atp_yield(model, substrate, acceptor="cpd00007", open_cpds=None, rate=1.0,
                  release_floors=True, lift_kinetics=True, pins=None):
    """The ATP yield of `substrate` net of what the same test makes without it
    (msenergyaudit.net_atp_yield, with the members' ATP tests).

    The test: every uptake closed except `open_cpds` (default YIELD_OPEN: H2O, H+, Pi, CO2), the
    `acceptor` (a compound, a list, or None) and the substrate at up to `rate` (mmol/gDW/h);
    forced fluxes released, kinetic rows lifted, the members' ATP tests summed and maximised.
    gross is that maximum per mmol of the allowed substrate; substrate_closed the same test with
    the substrate shut, per the same mmol; net = gross - substrate_closed. A model that makes ATP
    from H+ and Pi alone (a boundary route the sealed gate cannot see) counts it in the gross
    yield; the net yield does not. A cycle-free model can still be too efficient, which only the
    net yield shows against the biochemistry (24 per glucose aerobically, say).

    Returns {substrate, rate, open, atp_with_substrate, atp_without (mmol ATP/gDW/h),
    substrate_uptake (mmol/gDW/h), gross, substrate_closed, net (mmol ATP per mmol substrate)}."""
    pins = _resolve_pins(pins)
    return _by_member(_ea.net_atp_yield(
        model, substrate, acceptor=acceptor, open_cpds=YIELD_OPEN if open_cpds is None else open_cpds,
        rate=rate, release_floors=release_floors,
        lift_rows=KINETIC_ROW_SUFFIX if lift_kinetics else None, pins=pins,
        compartments=_member_cytosols(model, pins)))


def find_energy_cycle(model, flux=1.0, member=None, thermo=None, names=None, gapfilled=None,
                      tol=1e-7, release_floors=True, lift_kinetics=True, pins=None, **gate):
    """The smallest-flux cycle that makes `flux` ATP with every uptake closed, or the most ATP
    the community makes from nothing when that is less (msenergyaudit.find_energy_cycle on the
    members' ATP tests), by member.

    member     make the ATP in this member (index); default any member
    thermo     load_msdb_thermo() output, for each reaction's database direction and dG
    names      {member index: member id}
    gapfilled  {member index or id: set of ModelSEED reaction ids gapfilled into that member}
    release_floors  release forced fluxes first (release_forced_fluxes), as energy_from_nothing
    lift_kinetics   lift the *_commKin rows first (lifted_kinetic_rows), as energy_from_nothing
    pins       None (ATP tests) or "ntp" (the other hydrolysis tests count too; add_atp_tests)
    gate       c_min, c_max, n_sigma for direction_thermo

    Returns {status, atp_flux, reactions: [...], cross_fed: {e0 metabolite: {member: net
    production}}, members, kinetic_rows_lifted}. status is "cycle" when the community makes ATP
    from nothing, "no cycle" when it makes none, and "infeasible" when the sealed community
    cannot run at all (a forced flux kept with release_floors=False, say) -- which says nothing
    about cycles. atp_flux is the ATP the cycle makes (mmol/gDW/h): `flux`, or the community's
    maximum when that is less (the copy before the move read that case as "no cycle"); 0.0
    with no cycle, None when infeasible."""
    names, gapfilled = names or {}, gapfilled or {}
    pins = _resolve_pins(pins)
    compartment = None if member is None else f"c{member}"
    try:
        found = _ea.find_energy_cycle(
            model, flux=flux, compartment=compartment, thermo=thermo, tol=tol,
            release_floors=release_floors, lift_rows=KINETIC_ROW_SUFFIX if lift_kinetics else None,
            pins=pins, compartments=_member_cytosols(model, pins), **gate)
    except KeyError as exc:  # no ATP test in that member: name the member, as before
        if compartment is not None and exc.args == (compartment,):
            raise KeyError(member) from None
        raise
    if found["status"] != "cycle":
        return {"status": found["status"], "atp_flux": found["atp_flux"], "reactions": [],
                "cross_fed": {}, "kinetic_rows_lifted": found["rows_lifted"]}
    boundary = {r.id for r in model.boundary}
    cross_fed = {}
    for rid, v in found["fluxes"].items():
        if rid.startswith(TEST_PREFIXES):
            continue  # the tests hold no shared-pool metabolite (and are gone from the model)
        rxn = model.reactions.get_by_id(rid)
        idx = member_index(rxn)
        for met, coef in rxn.metabolites.items():
            if met.compartment == "e0" and rid not in boundary and idx:
                mem = names.get(idx, idx)
                cross_fed.setdefault(met.id, {}).setdefault(mem, 0.0)
                cross_fed[met.id][mem] += coef * v
    rows = []
    for r in found["reactions"]:
        idx = member_index(model.reactions.get_by_id(r["reaction"]))
        mem = names.get(idx, idx)
        gf = gapfilled.get(idx, gapfilled.get(mem, ()))
        rows.append({"reaction": r["reaction"], "msid": r["msid"], "member_index": idx, "member": mem,
                     **{k: r[k] for k in ("flux", "bounds", "db_reversibility", "name", "against_db",
                                          "dg0", "dg_min", "dg_err", "dg_source", "infeasible")},
                     "gapfilled": bool(r["msid"] and r["msid"] in gf)})
    cross_fed = {m: {k: round(x, 9) for k, x in d.items() if abs(x) > tol}
                 for m, d in cross_fed.items()}
    cross_fed = {m: d for m, d in cross_fed.items() if any(x > 0 for x in d.values())
                 and any(x < 0 for x in d.values())}
    return {"status": "cycle", "atp_flux": found["atp_flux"], "reactions": rows, "cross_fed": cross_fed,
            "members": sorted({str(r["member"]) for r in rows if r["member_index"]}),
            "kinetic_rows_lifted": found["rows_lifted"]}


def _interface(model):
    return model.solver.interface.__name__.rsplit(".", 1)[-1]


def reset_start(model, threads=None):
    """Put the model's solver on a defined starting state for its next solve: GLPK on its standard
    basis (glp_std_basis), Gurobi with every earlier solution and warm start discarded
    (Model.reset()) and Threads set if `threads` is given. Other solvers are left as they are.
    Returns the model."""
    name = _interface(model)
    prob = model.solver.problem
    if name.startswith("glpk"):
        import swiglpk
        swiglpk.glp_std_basis(prob)
    elif name == "gurobi_interface":
        if threads:
            prob.Params.Threads = int(threads)
        prob.update()
        prob.reset()
    return model


def canonical_copy(model, threads=None):
    """model.copy() on a defined starting state, for solves whose optimal vertex has to be
    reproducible: the copy's rows are rebuilt from the reactions in model order (two builds of
    one community whose rows store their terms in different orders give identical copies), and
    its solver starts from reset_start. On Gurobi the copy is a fresh model read back from LP
    text, which drops raw parameters set outside optlang; Threads, Presolve and Method are set on
    it explicitly from the original (Threads from `threads` if given). The same configuration
    and thread count then give the same vertex -- reproducible, not more right."""
    prob = model.solver.problem
    params = {}
    if _interface(model) == "gurobi_interface":
        for p in ("Threads", "Presolve", "Method", "Seed"):
            try:
                params[p] = getattr(prob.Params, p)
            except Exception:
                pass
        if threads:
            params["Threads"] = int(threads)
    work = model.copy()
    if params:
        for p, v in params.items():
            setattr(work.solver.problem.Params, p, v)
    return reset_start(work, threads)


def robust_pfba(model, canonical=False, threads=None):
    """cobra's pfba at the full optimum, retried at 99.99% and 99.9% of it when the solver calls
    the full-optimum problem infeasible -- which it can, numerically, on a community of 100k+
    reactions (v7d day 168: the oriented copy's pFBA raised Infeasible). Returns the Solution.

    canonical=True solves on canonical_copy(model, threads), each attempt from reset_start, so
    the vertex returned does not depend on what was solved on `model` before (pFBA's optimum is
    seldom unique, and a simplex returns the vertex nearest its start). The default solves on
    `model` from whatever state it is in, as before."""
    from cobra.flux_analysis import pfba
    from cobra.exceptions import OptimizationError
    work = canonical_copy(model, threads) if canonical else model
    last = None
    for frac in (1.0, 0.9999, 0.999):
        try:
            if canonical:
                reset_start(work, threads)
            return pfba(work, fraction_of_optimum=frac)
        except (OptimizationError, Exception) as exc:  # Infeasible is an OptimizationError
            last = exc
    raise last



# --------------------------------------------------------------------------- cutting
def _close(rxn, direction):
    """Close the direction `rxn` ran in (direction > 0: forward)."""
    if direction > 0:
        rxn.bounds = (min(rxn.lower_bound, 0), 0)
    else:
        rxn.bounds = (0, max(rxn.upper_bound, 0))


def apply_direction_rules(model, rules):
    """Set the allowed direction of every member's copy of a ModelSEED reaction:
    rules = {msid: ">" | "<" | "0"}. Returns [(reaction id, old bounds, new bounds)]."""
    changed = []
    for rxn in model.reactions:
        rule = rules.get(msid(rxn) or "")
        if rule is None:
            continue
        old = tuple(rxn.bounds)
        if rule == ">":
            rxn.bounds = (max(rxn.lower_bound, 0), max(rxn.upper_bound, 0))
        elif rule == "<":
            rxn.bounds = (min(rxn.lower_bound, 0), min(rxn.upper_bound, 0))
        elif rule == "0":
            rxn.bounds = (0, 0)
        else:
            raise ValueError(f"rule for {msid(rxn)} must be '>', '<' or '0', not {rule!r}")
        if tuple(rxn.bounds) != old:
            changed.append((rxn.id, old, tuple(rxn.bounds)))
    return changed


def cut_energy_cycles(model, thermo=None, msdb_path=None, rules=None, auto=True, scope="all_members",
                      protect=None, max_rounds=25, tol=1e-6, names=None, gapfilled=None,
                      c_min=1e-6, c_max=0.02, n_sigma=2.0, fallback=None, fallback_min_dg0=0.0,
                      canonical=True, threads=None, lift_kinetics=True):
    """Remove the community's energy-generating cycles, changing `model` in place.

    canonical  run the sealed tests (energy_from_nothing, find_energy_cycle) on canonical_copy(
               model, threads), its bounds kept equal to `model`'s after every round, so the cycle
               found -- one optimal vertex of a minimum-flux LP -- does not depend on the
               solver's history or on the stored order of the rows' terms. Cuts, `protect` and
               the report act on `model` itself. False tests `model` directly (the earlier
               behaviour).
    lift_kinetics  lift the *_commKin rows for the sealed tests (energy_from_nothing); with them
               in place the community makes nothing on nothing and the report says "clean"
               whatever cycles it holds.

    Each round tests ATP from nothing; if any is made, finds the cycle (find_energy_cycle) and
    closes the directions in it that thermodynamics rules out (direction_thermo), most uphill
    first. The cut is applied to every member's copy of the reaction (scope "all_members": a
    thermodynamic verdict holds for all of them, and a cycle cut in one member would otherwise
    reappear through another's copy) or to the cycle's copy only (scope "member").

    rules     {msid: ">"|"<"|"0"} applied first (apply_direction_rules), e.g. curations already
              decided on
    auto      False: find the first cycle and return its gated candidates without cutting
    protect   callable(model, cuts) -> bool, run after each round's cuts; False reverts them and
              the round retries them one at a time, keeping those protect accepts (cuts: the
              list of {reaction, msid, direction, ...} dicts just applied)
    fallback  None, or "most_uphill": when the gate rules out nothing in a cycle, cut the one step
              with the highest dG'0 in its running direction (above fallback_min_dg0; then steps
              against the database direction that have no dG) that protect accepts, one per round.
              Every cycle has to be broken somewhere; this breaks it at its most doubtful step. It
              is a heuristic (loop law would choose among the steps jointly), so cuts made this
              way carry tier 2 in the report and are curation proposals to check, not verdicts.

    Returns a report with status:
      "clean"                     no ATP from nothing
      "proposed"                  auto=False; the candidates are in report["proposed"]
      "feasible-direction cycle"  a cycle remains whose directions the gate cannot rule out (and
                                  fallback is off or finds no step); it is in
                                  report["remaining_cycle"]; loop law is needed
      "protected"                 every remaining candidate would fail `protect`
      "max rounds"
    """
    if thermo is None and msdb_path is not None:
        thermo = load_msdb_thermo(msdb_path)
    if not thermo:
        raise ValueError("cut_energy_cycles needs thermodynamic data: pass thermo= or msdb_path=")
    gate = {"c_min": c_min, "c_max": c_max, "n_sigma": n_sigma}
    report = {"rules_applied": [], "rounds": [], "cuts": [], "kept_open": [], "status": "max rounds",
              "canonical": bool(canonical)}
    if rules:
        report["rules_applied"] = apply_direction_rules(model, rules)
    work = canonical_copy(model, threads) if canonical else model

    def sync():
        if work is not model:
            for r in model.reactions:
                w = work.reactions.get_by_id(r.id)
                if w.bounds != r.bounds:
                    w.bounds = r.bounds

    refused = set()
    for rnd in range(max_rounds):
        sync()  # the last round's cuts (and undos) onto the copy the tests run on
        test = energy_from_nothing(work, electrons=False, biomass=None, tol=tol, lift_kinetics=lift_kinetics)
        report.setdefault("kinetic_rows_lifted", test["kinetic_rows_lifted"])
        atp = test["atp_from_nothing"]
        entry = {"round": rnd, "atp_from_nothing": atp, "atp_members": len(test["atp_by_member"])}
        report["rounds"].append(entry)
        if atp is None:
            report["status"] = "test failed"
            break
        if atp <= tol:
            report["status"] = "clean"
            break
        cycle = find_energy_cycle(work, thermo=thermo, names=names, gapfilled=gapfilled,
                                  lift_kinetics=lift_kinetics, **gate)
        entry.update(cycle_reactions=len(cycle["reactions"]), cycle_members=cycle.get("members"))
        cands = [r for r in cycle["reactions"] if r["infeasible"] and (r["msid"], r["flux"] > 0) not in refused]
        # most uphill first; among equals, gapfilled reactions first
        cands.sort(key=lambda r: (-r["dg_min"], not r["gapfilled"]))
        if not auto:
            report["proposed"], report["status"] = cands, "proposed"
            report["cycle"] = cycle
            break
        if not cands and fallback == "most_uphill":
            cuts = []
            for c in _uphill_steps(cycle, refused, fallback_min_dg0):
                one = _cut(model, [c], scope, tier=2)
                if protect is None or protect(model, one):
                    cuts = one
                    break
                _undo(model, one)
                refused.add((c["msid"], c["flux"] > 0))
                report["kept_open"].append({k: c[k] for k in ("reaction", "msid", "member", "flux", "dg0", "dg_min")})
            if not cuts:
                report["status"] = "protected" if refused else "feasible-direction cycle"
                report["remaining_cycle"] = cycle
                break
            entry["cuts"] = [c["reaction"] for c in cuts]
            report["cuts"] += cuts
            continue
        if not cands:
            report["status"] = "feasible-direction cycle" if not refused else "protected"
            report["remaining_cycle"] = cycle
            break
        cuts = _cut(model, cands, scope)
        if protect is not None and not protect(model, cuts):
            _undo(model, cuts)
            cuts = []
            for c in cands:
                one = _cut(model, [c], scope)
                if protect(model, one):
                    cuts += one
                else:
                    _undo(model, one)
                    refused.add((c["msid"], c["flux"] > 0))
                    report["kept_open"].append({k: c[k] for k in ("reaction", "msid", "member", "flux", "dg0", "dg_min")})
        entry["cuts"] = [c["reaction"] for c in cuts]
        report["cuts"] += cuts
    return report


def _uphill_steps(cycle, refused, min_dg0):
    """The fallback's candidates, best first: steps with dG'0 above min_dg0 in their running
    direction, most uphill first (against the database direction, then gapfilled, break ties);
    then steps against the database direction with no dG."""
    rows = [r for r in cycle["reactions"] if r["msid"] and (r["msid"], r["flux"] > 0) not in refused]
    with_dg = sorted((r for r in rows if r["dg0"] is not None and r["dg0"] > min_dg0),
                     key=lambda r: (-r["dg0"], not r["against_db"], not r["gapfilled"]))
    without = sorted((r for r in rows if r["dg0"] is None and r["against_db"]),
                     key=lambda r: (not r["gapfilled"], -abs(r["flux"])))
    return with_dg + without


def _cut(model, cands, scope, tier=1):
    """Close each candidate's running direction (in every member's copy if scope is
    "all_members"); returns one record per reaction changed."""
    done, cuts = set(), []
    for c in cands:
        key = (c["msid"], c["flux"] > 0)
        if key in done:
            continue
        done.add(key)
        if scope == "all_members" and c["msid"]:
            targets = [r for r in model.reactions if msid(r) == c["msid"]]
        else:
            targets = [model.reactions.get_by_id(c["reaction"])]
        for rxn in targets:
            old = tuple(rxn.bounds)
            _close(rxn, c["flux"])
            if tuple(rxn.bounds) != old:
                cuts.append({"reaction": rxn.id, "msid": c["msid"], "direction": ">" if c["flux"] > 0 else "<",
                             "old_bounds": old, "member_index": member_index(rxn),
                             "found_in": c["reaction"], "dg0": c["dg0"], "dg_min": c["dg_min"],
                             "gapfilled": c["gapfilled"], "against_db": c["against_db"], "tier": tier})
    return cuts


def _undo(model, cuts):
    for c in reversed(cuts):
        model.reactions.get_by_id(c["reaction"]).bounds = c["old_bounds"]


# --------------------------------------------------------------------------- localized loop law
class LocalLoopLaw:
    """The loop law, with a strict margin, on a chosen set of reactions only.

    For each covered reaction r, two binaries and the potentials of its metabolites:

        f_r = 1:  forward flux allowed,  dG_r = sum_m S_mr mu_m <= -eps
        b_r = 1:  reverse flux allowed,  dG_r >= eps
        f_r + b_r <= 1;  with both 0 the reaction carries no flux and dG_r is free

    A covered set that contains a whole cycle cannot carry flux around it, whatever the cycle's
    biochemistry, so this removes cycles that no single direction cut can (each direction feasible
    alone, the cycle infeasible jointly). With eps > 0 the degenerate dG = 0 everywhere does not
    satisfy it (the reason for the strict margin; see ModelSEEDpy SimpleThermoPkg's dg_margin).
    Potentials are free (no dG data), so a covered reaction may run either way: the law forbids
    cycles, not uphill steps, which only `cut_energy_cycles`' gate can see.

    pin_atp adds, for every member compartment with ATP, ADP, phosphate, water and H+,
    dG(ATP hydrolysis) <= -eps unconditionally (no binary). ATP hydrolysis is exergonic in any
    living cell, and the pin makes any covered cycle whose net is ATP synthesis infeasible in every
    medium -- including one that would hand its ATP to biomass instead of to a test reaction.

    pins="ntp" pins GTP, CTP, UTP and pyrophosphate hydrolysis the same way (NTP_PINS; or a
    list of (triphosphate, diphosphate) pairs): without it a covered cycle that charges GTP and
    hands it to biomass stays allowed, NDK-linked to ATP or not (the T3/T3L validation).

    The rows go on `model` and stay until remove(); solves become MILPs with two binaries per
    covered reaction. `lazy_loop_law` or `loop_capable_reactions` chooses the reactions."""

    def __init__(self, model, dg_margin=1.0, max_potential=1000.0, pin_atp=True, pins=None):
        if dg_margin <= 0:
            raise ValueError("dg_margin must be > 0: with 0 the potentials can all be equal and every cycle passes")
        self.model, self.eps, self.M = model, float(dg_margin), float(max_potential)
        self.potentials, self.binaries, self.pinned, self.pinned_all = {}, {}, [], []
        self.pins = _resolve_pins(pins, pin_atp)
        self._objs = []
        if self.pins:
            self._pin()

    def _mu(self, met, new):
        var = self.potentials.get(met.id)
        if var is None:
            var = self.model.problem.Variable(f"llmu_{met.id}", lb=-self.M, ub=self.M)
            self.potentials[met.id] = var
            new.append(var)
        return var

    def _pin(self):
        """dG(hydrolysis) <= -eps for every pin in every member cytosol that has it. `pinned` lists
        the compartments with the ATP pin (as before), `pinned_all` every (compound, compartment)."""
        mets = {met.id: met for met in self.model.metabolites}
        new, rows = [], []
        for tri, comp, st in hydrolysis_sites(self.model, self.pins):
            expr = sum(n * self._mu(mets[mid], new) for mid, n in st.items())
            name = f"llpin_atp_{comp}" if tri == ATP else f"llpin_ntp_{tri}_{comp}"
            rows.append(self.model.problem.Constraint(expr, ub=-self.eps, name=name))
            if tri == ATP:
                self.pinned.append(comp)
            self.pinned_all.append((tri, comp))
        self.model.add_cons_vars(new + rows)
        self._objs += new + rows

    _pin_atp = _pin  # the earlier name

    def add(self, reactions):
        """Cover `reactions` (boundary reactions and those already covered are skipped).
        Returns the number newly covered."""
        new, rows = [], []
        for rxn in reactions:
            if rxn.id in self.binaries or rxn.boundary or not rxn.metabolites:
                continue
            f = self.model.problem.Variable(f"llf_{rxn.id}", type="binary")
            b = self.model.problem.Variable(f"llb_{rxn.id}", type="binary")
            new += [f, b]
            self.binaries[rxn.id] = (f, b)
            dg = sum(coef * self._mu(met, new) for met, coef in rxn.metabolites.items())
            big = self.M * sum(abs(c) for c in rxn.metabolites.values()) + self.eps
            ub, lb = max(rxn.upper_bound, 0), max(-rxn.lower_bound, 0)
            rows += [
                self.model.problem.Constraint(rxn.forward_variable - ub * f, ub=0, name=f"llcf_{rxn.id}"),
                self.model.problem.Constraint(rxn.reverse_variable - lb * b, ub=0, name=f"llcb_{rxn.id}"),
                self.model.problem.Constraint(f + b, ub=1, name=f"llx_{rxn.id}"),
                self.model.problem.Constraint(dg + big * f, ub=big - self.eps, name=f"lldf_{rxn.id}"),
                self.model.problem.Constraint(dg - big * b, lb=self.eps - big, name=f"lldb_{rxn.id}")]
        self.model.add_cons_vars(new + rows)
        self._objs += new + rows
        return sum(1 for v in new if v.type == "binary") // 2

    def start(self, fluxes, tol=1e-9):
        """A complete MIP start (Gurobi only) from a flux distribution the law allows, e.g. an
        orient_community optimum: the fluxes, the binaries their signs imply, and potentials from
        a feasibility LP (each covered reaction that carries flux downhill by eps, ATP pinned).
        Returns False, setting nothing, if no such potentials exist (the fluxes contain a
        cycle)."""
        import numpy as np
        import scipy.sparse as sp
        from scipy.optimize import linprog
        names = sorted(self.potentials)
        col = {m: i for i, m in enumerate(names)}
        rows, cols, vals, rhs = [], [], [], []
        for rid in self.binaries:
            v = fluxes.get(rid, 0.0)
            if abs(v) <= tol:
                continue
            sign = 1.0 if v > 0 else -1.0  # forward: dG <= -eps;  reverse: -dG <= -eps
            for met, c in self.model.reactions.get_by_id(rid).metabolites.items():
                rows.append(len(rhs)); cols.append(col[met.id]); vals.append(sign * c)
            rhs.append(-self.eps)
        for row in self.model.constraints:
            if row.name.startswith("llpin_"):  # every hydrolysis pin, whoever added it
                for var, c in row.get_linear_coefficients(row.variables).items():
                    rows.append(len(rhs)); cols.append(col[var.name[len("llmu_"):]]); vals.append(c)
                rhs.append(row.ub)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(len(rhs), len(names)))
        res = linprog(np.zeros(len(names)), A_ub=A, b_ub=np.array(rhs),
                      bounds=[(-self.M, self.M)] * len(names), method="highs")
        if res.status != 0:
            return False
        g = self.model.solver.problem
        if not hasattr(g, "getVarByName"):  # MIP starts are set through gurobipy only
            return False
        self.model.solver.update()
        def put(name, value):
            var = g.getVarByName(name)
            if var is not None:
                var.Start = value
        for r in self.model.reactions:
            v = fluxes.get(r.id, 0.0)
            put(r.forward_variable.name, max(v, 0.0)); put(r.reverse_variable.name, max(-v, 0.0))
        for rid, (f, b) in self.binaries.items():
            v = fluxes.get(rid, 0.0)
            put(f.name, 1.0 if v > tol else 0.0); put(b.name, 1.0 if v < -tol else 0.0)
        for m, x in zip(names, res.x):
            put(self.potentials[m].name, float(x))
        return True

    def remove(self):
        self.model.remove_cons_vars(self._objs)
        self._objs, self.potentials, self.binaries, self.pinned, self.pinned_all = [], {}, {}, [], []


def loop_capable_reactions(model, tol=1e-6, seeds=8, processes=None, seed=0, lift_kinetics=True,
                           pins=None, report=None):
    """The internal reactions that can carry flux with every boundary reaction shut and an ATP
    hydrolysis reaction in each member (add_atp_tests): the only ones a cycle can use, counting
    the cycles that would make ATP if something consumed it -- which LocalLoopLaw's ATP pin
    stands in for. Covering exactly these with LocalLoopLaw(pin_atp=True) is the full loop law
    (a reaction that cannot carry flux in that closed network is in no such cycle), with one
    binary for each of them instead of each of the community's reactions.

    `seeds` LPs with random objectives mark most of them; flux variability analysis settles the
    rest (`processes` workers). The *_commKin rows are lifted meanwhile (lift_kinetics; they
    would hold every flux at 0); pins="ntp" adds the other hydrolysis tests (use the same pins
    as the LocalLoopLaw). `report`, a dict, receives kinetic_rows_lifted."""
    with lifted_kinetic_rows(model, lift_kinetics) as lifted:
        if report is not None:
            report["kinetic_rows_lifted"] = lifted
        return _loop_capable(model, tol, seeds, processes, seed, _resolve_pins(pins))


def _loop_capable(model, tol, seeds, processes, seed, pins):
    import random
    from cobra.flux_analysis import flux_variability_analysis
    rng = random.Random(seed)
    internal = [r for r in model.reactions if not r.boundary]
    found = set()
    with model:
        release_forced_fluxes(model)
        for r in model.boundary:
            r.bounds = (0, 0)
        add_atp_tests(model, pins)
        for _ in range(seeds):
            coefs = {}
            for r in internal:
                if r.id not in found:
                    w = rng.choice((-1.0, 1.0)) * rng.random()
                    coefs[r.forward_variable], coefs[r.reverse_variable] = w, -w
            model.objective = model.problem.Objective(Zero, direction="max")
            model.objective.set_linear_coefficients(coefs)
            if _value(model) is None:
                break
            new = {r.id for r in internal if abs(r.flux) > tol} - found
            if not new:
                break
            found |= new
        rest = [r for r in internal if r.id not in found]
        if rest:
            fva = flux_variability_analysis(model, rest, fraction_of_optimum=0, processes=processes)
            found |= set(fva.index[(fva["minimum"].abs() > tol) | (fva["maximum"].abs() > tol)])
    return found


def lazy_loop_law(model, max_rounds=50, tol=1e-6, time_limit=None, law=None, **law_args):
    """Cover just enough reactions with LocalLoopLaw that the community makes no ATP from nothing.

    Each round maximises ATP from nothing under the current law (a MILP once anything is covered);
    if any is made, every internal reaction carrying flux is covered and the round repeats. A new
    cycle must use a reaction outside the covered set, so the set grows each round and the loop
    ends; in practice it stops at a small fraction of the community's reactions, which is what keeps
    the MILP tractable where the full loop law (one binary per reaction) is not.

    The *_commKin rows are lifted for the tests (lift_kinetics; they would hold every flux at 0,
    and the law would cover nothing) and are back on `model` afterwards. The tests are the law's
    pins: ATP, or every pinned nucleotide and PPi with pins="ntp".

    Returns (law, report); the law's rows stay on `model` for the simulations that follow."""
    import time
    lift = law_args.pop("lift_kinetics", True)
    law = law or LocalLoopLaw(model, **law_args)
    report = {"rounds": [], "status": "max rounds", "pinned": list(law.pinned)}
    saved = model.solver.configuration.timeout
    if time_limit is not None:
        model.solver.configuration.timeout = time_limit
    try:
        with lifted_kinetic_rows(model, lift) as lifted:
            report["kinetic_rows_lifted"] = lifted
            _lazy_rounds(model, law, report, max_rounds, tol)
    finally:
        model.solver.configuration.timeout = saved
    report["covered"] = len(law.binaries)
    return law, report


def _lazy_rounds(model, law, report, max_rounds, tol):
    import time
    for rnd in range(max_rounds):
        with model:
            release_forced_fluxes(model)
            tests = add_atp_tests(model, law.pins or ATP_PINS)
            close_uptakes(model)
            model.objective = model.problem.Objective(sum(r.flux_expression for r in tests.values()),
                                                      direction="max")
            t = time.time()
            v = _value(model)
            status = model.solver.status
            active = ([r for r in model.reactions if abs(r.flux) > tol and not r.boundary
                       and not r.id.startswith(TEST_PREFIXES)] if v is not None and v > tol else [])
        row = {"round": rnd, "atp_from_nothing": v, "status": status, "covered": len(law.binaries),
               "seconds": round(time.time() - t, 2)}
        report["rounds"].append(row)
        if v is None:
            report["status"] = f"solve failed ({status})"
            break
        if v <= tol:
            report["status"] = "clean"
            break
        row["added"] = law.add(active)


# --------------------------------------------------------------------------- one potential profile
def member_reference(model, biomass_prefix="bio", tol=1e-9, member_models=None, canonical=False, threads=None):
    """Each member growing alone on the model's current medium, by pFBA, the other members shut:
    {reaction id: flux} over all members (their reactions are disjoint), and {member index:
    growth}. A member that is free of energy-generating cycles alone gives a pFBA flux with no
    cycle in it, so this is a reference the potentials of `potential_profile` can always honour;
    it keeps every member's own growth feasible after `orient_by_potentials`. Members that cannot
    grow alone contribute nothing.

    member_models  {member index: the member's own cobra model} -- much faster on a large
                   community: each member is solved on its own model, with the community's
                   shared-pool uptake bounds, and its reaction ids are mapped the way
                   build_from_species_models renames them (rxn..._c0 -> rxn..._c<i>).
    canonical      solve on canonical_copy(model, threads) (each member's pFBA vertex then does not
                   depend on the solver's history); orient_community passes its own canonical
                   copy instead."""
    from cobra.flux_analysis import pfba
    if member_models:
        return _member_reference_from_models(model, member_models, biomass_prefix, tol)
    if canonical:
        model = canonical_copy(model, threads)
    biomass = {}
    for rxn in model.reactions:
        if re.match(rf"^{biomass_prefix}\d+$", rxn.id) and rxn.id != f"{biomass_prefix}1":
            idx = member_index(rxn)
            if idx:
                biomass[idx] = rxn
    drains = {}
    for rxn in model.boundary:
        met = next(iter(rxn.metabolites))
        m = _COMP.match(met.compartment or "")
        if _cpd(met) == "cpd11416" and m and m.group(2) != "0" and rxn.metabolites[met] < 0:
            drains[int(m.group(2))] = rxn
    owner = {r.id: member_index(r) for r in model.reactions}
    reference, grown = {}, {}
    for idx, bio in sorted(biomass.items()):
        with model:
            for r in model.reactions:
                o = owner[r.id]
                if (o is None and r.id != bio.id) or (o and o != idx):
                    r.bounds = (0, 0)
            if idx in drains:
                drains[idx].bounds = (0, 1000)
            else:
                met = next(m for m in bio.metabolites if _cpd(m) == "cpd11416" and bio.metabolites[m] > 0)
                sink = Reaction(f"DMref_{met.id}", lower_bound=0, upper_bound=1000)
                sink.add_metabolites({met: -1})
                model.add_reactions([sink])
            model.objective = bio
            g = model.slim_optimize(error_value=0.0)
            grown[idx] = g
            if g <= tol:
                continue
            sol = robust_pfba(model)
            reference.update({rid: v for rid, v in sol.fluxes.items()
                              if abs(v) > tol and owner.get(rid) == idx})
    return reference, grown


def _member_reference_from_models(model, member_models, biomass_prefix, tol):
    from cobra.flux_analysis import pfba
    uptake = {}
    for rxn in model.exchanges:
        met = next(iter(rxn.metabolites))
        if met.compartment == "e0":
            uptake[met.id] = rxn.lower_bound
    community_bio = {}
    for rxn in model.reactions:
        if re.match(rf"^{biomass_prefix}\d+$", rxn.id) and rxn.id != f"{biomass_prefix}1":
            idx = member_index(rxn)
            if idx:
                community_bio[idx] = rxn.id
    ids = {r.id for r in model.reactions}
    reference, grown = {}, {}
    for idx, mm in member_models.items():
        with mm:
            for rxn in mm.exchanges:
                met = next(iter(rxn.metabolites))
                if met.compartment.startswith("e"):
                    rxn.lower_bound = uptake.get(met.id, 0.0)
            bio = next((r for r in mm.reactions if re.match(rf"^{biomass_prefix}\d+$", r.id)), None)
            if bio is None:
                continue
            mm.objective = bio
            g = mm.slim_optimize(error_value=0.0)
            grown[idx] = g
            if g <= tol:
                continue
            for rid, v in robust_pfba(mm).fluxes.items():
                if abs(v) <= tol or rid.startswith("EX_"):
                    continue
                if rid == bio.id:
                    cid = community_bio.get(idx)
                else:
                    cid = re.sub(r"_([a-z]+)\d+$", rf"_\g<1>{idx}", rid)
                if cid in ids:
                    reference[cid] = v
    return reference, grown


def potential_profile(model, reference=None, dg_margin=1.0, max_potential=1e4, pin_atp=True,
                      w_reference=1.0, w_irreversible=1e2, lp="auto", threads=None, flux_floor=None,
                      pins=None):
    """One metabolite potential vector for the whole community, from which `orient_by_potentials`
    fixes reaction directions. With ATP hydrolysis pinned downhill (dG <= -dg_margin in every
    member), no orientation it gives can run an energy-generating cycle: around any cycle the dG
    of its steps sum to zero, every allowed step has dG <= 0, so all are 0 -- and no step of an
    ATP-making cycle can be, because ATP hydrolysis is not. The same argument rules out a cycle
    that hands its ATP to biomass. This is an LP and needs no binaries; the price is that one
    profile serves every simulation, where the loop law would find a profile per solution.

    LPs, lexicographically:
      1. honour the reference: minimise the dG against the `reference` flux directions
         ({reaction id: flux}, e.g. member_reference or a loop-law solution). `reference` can be
         a list of (flux dict, weight) pairs: they are honoured one after another, each holding
         what the earlier ones achieved (one LP each), so an earlier reference is never traded
         for a later one -- the members' own growth before the community optimum, say. (Summed
         into one LP, many small violations of the optimum outweighed one member reaction, and
         a community could no longer grow.) A reference with no cycle in it (conformal_cycle) is
         honoured exactly; one with a cycle is broken where it costs least.
      2. holding that, orient as little as possible: minimise |dG| over the internal reactions,
         w_irreversible on a dG that would block an irreversible reaction (positive for one that
         runs forward only, negative for reverse only), 1 on any other (a reversible reaction
         losing one direction). As many reactions as possible keep dG = 0 and both directions.
    (A single weighted LP let many small orientations outweigh a large reference flux, and
    broke a loop-law optimum that held no cycle.)

    lp: "gurobi" (gurobipy's barrier with crossover), "highs" (scipy's HiGHS, dual simplex),
    "highs-ipm" (scipy's HiGHS interior point), or "auto": gurobi when gurobipy imports, else
    highs-ipm. The barrier takes the 90k-reaction day-174 community in about a minute per LP;
    HiGHS's simplex took about 20 minutes. The community's cobra solver is not touched.
    threads goes to Gurobi (None: its default, i.e. gurobi.env or all cores); give it for a
    reproducible profile, since the barrier's vertex can change with the thread count.
    pins="ntp" pins GTP, CTP, UTP and PPi hydrolysis downhill too (NTP_PINS), which extends the
    guarantee to cycles that charge those nucleotides; report["pinned"] lists the compartments
    with the ATP pin, report["pins"] every (compound, compartment).
    Returns (dG {reaction id: dG}, report)."""
    import numpy as np
    import scipy.sparse as sp
    refs = reference if isinstance(reference, list) else [(reference or {}, w_reference)]
    rxns = [r for r in model.reactions if not r.boundary and r.metabolites]
    mets = {m.id: i for i, m in enumerate(model.metabolites)}
    nm, nr = len(mets), len(rxns)
    rows, cols, vals = [], [], []
    for j, r in enumerate(rxns):
        for met, c in r.metabolites.items():
            rows.append(j); cols.append(mets[met.id]); vals.append(c)
    St = sp.csr_matrix((vals, (rows, cols)), shape=(nr, nm))
    eye = sp.identity(nr, format="csr")
    # x = [mu (nm), p (nr), q (nr)],  dG = St mu = p - q,  p, q >= 0
    A_eq = sp.hstack([St, -eye, eye], format="csr")
    # stage 1, one LP per reference in order: against that reference. Every reference flux weighs
    # the same by default. flux_floor (off by default) shrinks the weight below it in proportion
    # to the flux -- meant for a MILP's big-M "trickle" flows, but it also cheapens real trace
    # fluxes (a cofactor at 1e-6 that some member's biomass needs) until breaking one costs less
    # than the next stage's tolerance: v7d/v7e days 147 and 149 reoriented to growth 0 that way.
    # Clean a MILP reference instead (drop |v| below its integrality leak) before passing it.
    stage1 = []
    for ref, w in refs:
        rp, rq = np.zeros(nr), np.zeros(nr)
        for j, r in enumerate(rxns):
            v = ref.get(r.id, 0.0)
            scale = 1.0 if not flux_floor else min(1.0, abs(v) / flux_floor)
            if v > 0:
                rp[j] = w * scale
            elif v < 0:
                rq[j] = w * scale
        stage1.append((rp, rq))
    wp, wq = np.ones(nr), np.ones(nr)      # stage 2: orienting at all
    for j, r in enumerate(rxns):
        if r.lower_bound >= 0 and r.upper_bound > 0:
            wp[j] = w_irreversible
        elif r.upper_bound <= 0 and r.lower_bound < 0:
            wq[j] = w_irreversible
    zeros = np.zeros(nm)
    A_ub, b_ub, pinned = sp.csr_matrix((0, nm + 2 * nr)), np.zeros(0), []
    sites = hydrolysis_sites(model, _resolve_pins(pins, pin_atp))
    if sites:
        prow, pcol, pval = [], [], []
        for k, (tri, comp, st) in enumerate(sites):
            for mid, n in st.items():
                prow.append(k); pcol.append(mets[mid]); pval.append(n)
            if tri == ATP:
                pinned.append(comp)
        A_ub = sp.csr_matrix((pval, (prow, pcol)), shape=(len(sites), nm + 2 * nr))
        b_ub = -dg_margin * np.ones(len(sites))
    if lp == "auto":
        try:
            import gurobipy  # noqa: F401
            lp = "gurobi"
        except ImportError:
            lp = "highs-ipm"
    solve = (lambda c, A, b: _profile_gurobi(c, A_eq, A, b, nm, nr, max_potential, threads)) if lp == "gurobi" \
        else (lambda c, A, b: _profile_scipy(c, A_eq, A, b, nm, nr, max_potential, lp))
    against = []
    for rp, rq in stage1:
        c1 = np.concatenate([zeros, rp, rq])
        if not c1.any():
            against.append(0.0)
            continue
        _, cost = solve(c1, A_ub, b_ub)
        against.append(cost)
        A_ub = sp.vstack([A_ub, sp.csr_matrix(c1)], format="csr")
        b_ub = np.concatenate([b_ub, [cost * (1 + 1e-9) + 1e-7]])
    against = against if isinstance(reference, list) else (against[0] if against else 0.0)
    x, objective = solve(np.concatenate([zeros, wp, wq]), A_ub, b_ub)
    # dG from the LP's own split, not recomputed as St mu: with potentials up to max_potential
    # the equality rows hold only to the solver's feasibility tolerance, and St mu then puts
    # 1e-6-sized noise on a dG the vertex has at exactly 0 -- enough for orient_by_potentials to
    # close a direction at random
    dg = x[nm:nm + nr] - x[nm + nr:]
    tol = ORIENT_TOL * max(1.0, dg_margin)
    dg[np.abs(dg) <= tol] = 0.0
    report = {"reactions": nr, "metabolites": nm, "pinned": pinned,
              "pins": [(tri, comp) for tri, comp, _ in sites], "against_cost": against,
              "objective": objective, "lp": lp,
              "oriented": int(np.sum(np.abs(dg) > tol)),
              "against_reference": [[rxns[j].id for j in range(nr)
                                     if (ref.get(rxns[j].id, 0) > 0 and dg[j] > tol)
                                     or (ref.get(rxns[j].id, 0) < 0 and dg[j] < -tol)] for ref, _ in refs],
              "irreversible_blocked": [rxns[j].id for j in range(nr)
                                       if (rxns[j].lower_bound >= 0 and rxns[j].upper_bound > 0 and dg[j] > tol)
                                       or (rxns[j].upper_bound <= 0 and rxns[j].lower_bound < 0 and dg[j] < -tol)]}
    if not isinstance(reference, list):
        report["against_reference"] = report["against_reference"][0]
    return {r.id: float(dg[j]) for j, r in enumerate(rxns)}, report


def _profile_scipy(c, A_eq, A_ub, b_ub, nm, nr, max_potential, method):
    import numpy as np
    from scipy.optimize import linprog
    bounds = [(-max_potential, max_potential)] * nm + [(0, None)] * (2 * nr)
    res = linprog(c, A_ub=A_ub if A_ub.shape[0] else None, b_ub=b_ub if A_ub.shape[0] else None,
                  A_eq=A_eq, b_eq=np.zeros(nr), bounds=bounds, method=method)
    if res.status != 0:
        raise RuntimeError(f"potential profile LP failed: {res.message}")
    return res.x, float(res.fun)


def _profile_gurobi(c, A_eq, A_ub, b_ub, nm, nr, max_potential, threads):
    """potential_profile's LP through gurobipy's matrix interface: barrier, then crossover to a
    vertex. The vertex matters: the barrier alone stops at the centre of the optimal face, where a
    dG that some optimal profiles leave at 0 comes out small but not 0 and would fix a direction
    for nothing."""
    import numpy as np
    import gurobipy as gp
    from gurobipy import GRB
    env = gp.Env(params={"OutputFlag": 0, **({"Threads": int(threads)} if threads else {})})
    g = gp.Model(env=env)
    lb = np.concatenate([np.full(nm, -max_potential), np.zeros(2 * nr)])
    ub = np.concatenate([np.full(nm, max_potential), np.full(2 * nr, GRB.INFINITY)])
    x = g.addMVar(nm + 2 * nr, lb=lb, ub=ub, obj=c)
    g.addMConstr(A_eq, x, "=", np.zeros(nr))
    if A_ub is not None and A_ub.shape[0]:
        g.addMConstr(A_ub, x, "<", b_ub)
    g.Params.Method = 2
    if threads:
        g.Params.Threads = threads
    g.optimize()
    if g.Status != GRB.OPTIMAL:
        raise RuntimeError(f"potential profile LP failed: Gurobi status {g.Status}")
    out = x.X.copy(), float(g.ObjVal)
    g.dispose(); env.dispose()
    return out


def orient_by_potentials(model, dg, tol=None):
    """Let each reaction run only downhill in the profile: forward only where dG < -tol, reverse
    only where dG > tol, both ways where |dG| <= tol (default ORIENT_TOL: a profile's real dGs
    are of the order of its margin; below that is solver tolerance, which once shut a member's
    glucose uptake). Returns [(reaction id, old, new bounds)]."""
    tol = ORIENT_TOL if tol is None else tol
    changed = []
    for rid, g in dg.items():
        if abs(g) <= tol:
            continue
        rxn = model.reactions.get_by_id(rid)
        old = tuple(rxn.bounds)
        if g < 0:
            rxn.bounds = (max(rxn.lower_bound, 0), max(rxn.upper_bound, 0))
        else:
            rxn.bounds = (min(rxn.lower_bound, 0), min(rxn.upper_bound, 0))
        if tuple(rxn.bounds) != old:
            changed.append((rid, old, tuple(rxn.bounds)))
    return changed


def orient_community(model, reference=None, rounds=1, tol=1e-9, member_models=None, canonical=True,
                     no_growth_fallback="both", **profile_args):
    """member_reference -> potential_profile -> orient_by_potentials on `model` (in place), on the
    model's current medium and objective.

    reference  "pfba" or "both" (recommended), "members" / None, a flux dict, or a list of (flux,
               weight) pairs:
               "pfba"     the model's own pFBA optimum. Kept whole where it holds no cycle; never
                          lost growth on the cycle-free controls of the planted-cycle validation
                          (100/100) and keeps more where the optimum holds few cycles (the nine
                          day-174 cycle members: 19.61 vs 11.02 for "both", of 19.91).
               "both"     each member growing alone first (weighted 1e4), then the optimum as far
                          as it agrees with them: every member that grows alone keeps growing;
                          never lost growth on the cycle-free controls (100/100); keeps more where
                          the optimum leans on cycles (day 174, 88 members: 11.83 vs 6.17 for
                          "pfba", of 18.21).
               "members"  each member alone only (member_reference). Not recommended: it shuts
                          most cross-feeding, and on the cycle-free controls it lost growth in
                          19/100 instances, down to 0 where a member lives on another's product
                          (CommPhitting maltose: P. fluorescens grows 0 alone, and the community
                          went to 0). When a member cannot grow alone, the reference now
                          falls back (no_growth_fallback, below, with a warning).
               a dict     {reaction id: flux}, e.g. a loop-law MILP solution, honoured where it
                          holds no cycle, so the objective it allows is at least the reference's.
    no_growth_fallback  for reference "members" (or None) when some member grows <= tol alone:
               "both" (default) adds the whole community pFBA optimum after the members' own
               fluxes -- the "both" reference; "optimum" adds the optimum restricted to the
               reactions of the members that cannot grow alone; None keeps the earlier behaviour
               (they get nothing, and the profile may shut their uptake). On the planted-cycle
               controls (N1, 50 seeds: 6 with such a member, each with an energy-neutral ATP
               shuttle between two other members that no member uses alone) "both" kept the LP
               optimum on all 6, "optimum" on none (it raised two from 0, 1.383 and 5.057 of
               1.768 and 5.915, and lowered two that had kept it, 4.897 and 3.919 of 6.443 and
               5.440), the earlier behaviour on 2. The report's "fallback" names the members and
               the mode.
    canonical  True (default): every solve (member references, pFBA, the objective of each
               round) runs on canonical_copy(model, threads) -- rows rebuilt in reaction order,
               GLPK's standard basis or a fresh Gurobi model with Threads/Presolve/Method set --
               and the bounds found are written back to `model`. Before this, the orientation
               followed the solver's warm start and the rows' stored term order (three fresh
               Kp/Bt builds gave three orientations per reference). It makes the result
               reproducible for one solver configuration and thread count, not better. False
               solves on `model` as it is (the earlier behaviour).
    profile_args go to potential_profile: threads (also used for the canonical copy's Gurobi),
               lp, pins ("ntp" pins GTP, CTP, UTP and PPi hydrolysis too), w_reference, ...

    With rounds > 1, each further round adds the oriented model's pFBA optimum to the reference
    and derives a new profile from the original bounds. The optimum stays feasible (the new profile
    honours it), so the objective cannot fall, and directions the first profile fixed for no
    reason can reopen. member_models goes to member_reference. Returns a report with the
    objective of each round."""
    threads = profile_args.get("threads")
    work = canonical_copy(model, threads) if canonical else model
    original = {r.id: tuple(r.bounds) for r in work.reactions}
    boundary = {r.id for r in work.boundary}
    grown = None
    optimum = None
    fallback = None

    def pfba_optimum():
        return {rid: f for rid, f in robust_pfba(work).fluxes.items() if abs(f) > tol and rid not in boundary}

    if reference in ("pfba", "both"):
        # the unoriented optimum itself: kept whole when it holds no cycle (pFBA rarely pays
        # for one); whatever cycle it does hold, the soft weights orient against
        optimum = pfba_optimum()
    if reference is None or reference in ("members", "both"):
        members, grown = member_reference(work, member_models=member_models)
    if reference == "both":
        # each member alone first (cycle-free, so the community can always grow), then as much
        # of the optimum as agrees with them
        w = profile_args.pop("w_reference", 1e4)
        reference = [(members, w), (optimum, w)]   # in this order: members first
    elif reference == "pfba":
        reference = optimum
    elif reference is None or reference == "members":
        stuck = sorted(i for i, g in (grown or {}).items() if g <= tol)
        if stuck and no_growth_fallback:
            if no_growth_fallback not in ("optimum", "both"):
                raise ValueError(f"no_growth_fallback must be 'optimum', 'both' or None, not {no_growth_fallback!r}")
            optimum = pfba_optimum()
            if no_growth_fallback == "both":
                second = optimum
            else:
                owner = {r.id: member_index(r) for r in work.reactions}
                second = {rid: f for rid, f in optimum.items() if owner.get(rid) in stuck}
            logger.warning("orient_community(reference='members'): member(s) %s cannot grow alone on this "
                           "medium; their reactions follow the community optimum instead (%s, %d reactions). "
                           "reference='pfba' or 'both' is recommended.", stuck, no_growth_fallback, len(second))
            fallback = {"members": stuck, "mode": no_growth_fallback, "reactions": len(second)}
            w = profile_args.pop("w_reference", 1e4)
            reference = [(members, w), (second, w)]
        else:
            reference = members
    reference = list(reference) if isinstance(reference, list) else dict(reference)
    history, changed, profile = [], [], {}
    for k in range(rounds):
        for r in work.reactions:
            r.bounds = original[r.id]
        dg, profile = potential_profile(work, reference, **profile_args)
        changed = orient_by_potentials(work, dg)
        v = _value(work)
        against = profile["against_reference"]
        history.append({"round": k, "objective": v, "oriented": profile["oriented"],
                        "bounds_changed": len(changed),
                        "reference": [len(r) for r, _ in reference] if isinstance(reference, list) else len(reference),
                        "against_reference": [len(x) for x in against] if isinstance(reference, list) else len(against),
                        "irreversible_blocked": len(profile["irreversible_blocked"])})
        if k + 1 < rounds and v is not None and v > tol:
            new = pfba_optimum()
            if isinstance(reference, list):
                reference = reference + [(new, reference[-1][1])]
            else:
                reference.update(new)
    if work is not model:
        for r in model.reactions:
            b = work.reactions.get_by_id(r.id).bounds
            if tuple(r.bounds) != tuple(b):
                r.bounds = b
    return {"history": history, "changed": changed, "profile": profile, "member_growth_alone": grown,
            "fallback": fallback, "canonical": bool(canonical)}


def _incumbent(model):
    """The objective of the last solve, also when it stopped at a limit with a solution in hand
    (cobra's slim_optimize reports only optimal solves)."""
    v = _value(model)
    if v is None and model.solver.status in ("time_limit", "feasible", "suboptimal", "iteration_limit"):
        try:
            v = float(model.solver.objective.value)
        except Exception:
            v = None
        if v is not None and (isnan(v) or abs(v) == float("inf")):
            v = None  # stopped without any solution
    return v


def _mip_info(model):
    """(relative MIP gap, best bound) of the last solve when it was a Gurobi MIP, else (None, None)."""
    prob = model.solver.problem
    try:
        if not prob.IsMIP:
            return None, None
        return float(prob.MIPGap), float(prob.ObjBound)
    except Exception:
        return None, None


def loop_law_optimum(model, start=None, law=None, max_rounds=50, time_limit=None, tol=1e-7, time_budget=None,
                     threads=None, clean=True, **law_args):
    """The model's optimum under the loop law (ATP pinned), found by covering only the cycles the
    optimum tries to use: solve; if the solution holds a cycle (conformal_cycle), cover its
    reactions with LocalLoopLaw and solve again. The first solve is the plain LP; each later one
    is a MILP whose binaries are the covered reactions only, warm-started from `start` -- a flux
    with no cycle in it, e.g. an orient_community optimum, which every cover allows. At the end
    the solution holds no cycle, so it is the loop-law optimum (to the MILP's gap).

    time_limit caps each solve, time_budget the whole loop (seconds). A solve that stops at its time
    limit with a cycle-free solution gets one more solve with all the budget left; when the budget
    runs out the status says so and `fluxes` is the last solution, which may still hold a cycle (check it with
    conformal_cycle before using it as a cycle-free reference).

    law_args go to LocalLoopLaw (pins="ntp" pins GTP, CTP, UTP and PPi hydrolysis as well; the
    cycle test, conformal_cycle, then uses the same pins, and `start` must come from an
    orientation made with them too, else it is not a valid MIP start). threads goes to
    conformal_cycle's Gurobi environment.

    The report gives, per MILP round, "mip_gap" and "objective_bound" (Gurobi; None on other
    solvers), and for the best cycle-free solution "mip_gap", "objective_bound" and, with clean,
    a cleaned flux: a MILP lets a binary sit at its integrality tolerance, so a reaction whose
    binary is "0" can still carry up to IntFeasTol x its bound (1e-5 x 1000 = 0.01 on Gurobi's
    defaults) -- a trickle that is not in the orientation the solution claims. The clean solve
    fixes every binary at the best solution's rounded value (reactions covered after that round:
    by the sign of their flux) and solves once more, then takes the minimum-total-flux solution
    at that optimum (pFBA; a plain re-solve can return a vertex with a free loop through
    uncovered reactions): "clean_orientation" {reaction: (forward, reverse)}, "clean_objective"
    (the optimum in that orientation), "clean_fluxes" (exactly 0 on every reaction the
    orientation shuts), "clean_fraction" (of the optimum the pFBA held: 1.0, else 0.9999),
    "clean_cycle" (conformal_cycle of it, [] when none) and "clean_status". The binaries are
    freed again afterwards.

    Returns (fluxes, report); the law's rows stay on `model` (report["law"]) until removed."""
    import time
    law = law or LocalLoopLaw(model, **law_args)
    report = {"rounds": [], "status": "max rounds", "law": law,
              "best_objective": float("-inf"), "best_fluxes": None, "mip_gap": None, "objective_bound": None}
    saved = model.solver.configuration.timeout
    if time_limit is not None:
        model.solver.configuration.timeout = time_limit
    fluxes = {}
    resume = False
    best_binaries = {}
    began = time.time()
    try:
        for rnd in range(max_rounds):
            if time_budget is not None and time.time() - began > time_budget:
                report["status"] = "time budget"
                break
            # a resumed solve keeps its search tree: setting a start would change the model
            started = (not resume) and bool(start) and bool(law.binaries) and law.start(start)
            t = time.time()
            v = _incumbent(model)
            row = {"round": rnd, "objective": v, "status": model.solver.status, "covered": len(law.binaries),
                   "started": started, "seconds": round(time.time() - t, 1)}
            row["mip_gap"], row["objective_bound"] = _mip_info(model) if law.binaries else (None, None)
            report["rounds"].append(row)
            if v is None:
                report["status"] = f"solve failed ({model.solver.status})"
                break
            fluxes = {r.id: r.flux for r in model.reactions}
            cycle = conformal_cycle(model, fluxes, tol=tol, pins=law.pins, threads=threads)
            row["cycle"] = len(cycle)
            if not cycle and v > report["best_objective"]:
                report["best_objective"], report["best_fluxes"] = v, fluxes
                report["mip_gap"], report["objective_bound"] = row["mip_gap"], row["objective_bound"]
                best_binaries = {rid: (int(round(f.primal or 0.0)), int(round(b.primal or 0.0)))
                                 for rid, (f, b) in law.binaries.items()}
            if not cycle:
                left = None if time_budget is None else time_budget - (time.time() - began)
                if model.solver.status != "optimal" and not resume and (left is None or left > 60):
                    # stopped at the round's time limit with a cycle-free incumbent: unproven, so
                    # give the same MILP one solve with all the budget left (or no limit at all),
                    # then cover any cycle that solution holds as usual
                    resume = True
                    row["extended_to"] = None if left is None else round(left)
                    model.solver.configuration.timeout = left
                    continue
                report["status"] = "no cycle" if model.solver.status == "optimal" else "no cycle (time budget, unproven)"
                break
            if resume:
                model.solver.configuration.timeout = time_limit if time_limit is not None else saved
                resume = False
            row["added"] = law.add([model.reactions.get_by_id(rid) for rid in cycle])
            if not row["added"]:
                report["status"] = "cycle already covered (solver tolerance)"
                break
        if clean and report["best_fluxes"] is not None and law.binaries:
            model.solver.configuration.timeout = time_limit if time_limit is not None else saved
            _clean_loop_law(model, law, report, best_binaries, tol, threads)
    finally:
        model.solver.configuration.timeout = saved
    return fluxes, report


def _clean_loop_law(model, law, report, best_binaries, tol, threads):
    """loop_law_optimum's final solve with every binary fixed at the best solution's value."""
    best = report["best_fluxes"]
    orient = {}
    for rid in law.binaries:
        if rid in best_binaries:
            orient[rid] = best_binaries[rid]
        else:  # covered after the best round: that solution had no binary here, so its sign
            v = best.get(rid, 0.0)
            orient[rid] = (int(v > tol), int(v < -tol))
    saved = [(var, var.lb, var.ub) for fb in law.binaries.values() for var in fb]
    try:
        for rid, (f, b) in law.binaries.items():
            fv, bv = orient[rid]
            f.set_bounds(fv, fv)
            b.set_bounds(bv, bv)
        v = _incumbent(model)
        report["clean_status"] = model.solver.status
        report["clean_objective"] = v
        report["clean_orientation"] = orient
        report["clean_fluxes"], report["clean_cycle"], report["clean_fraction"] = None, None, None
        if v is not None:
            # the optimum in this orientation, at minimum total flux: a plain re-solve can return a
            # vertex in which reactions the law does not cover run a free loop at no cost (49 of
            # 200 cleaned fluxes in the planted-cycle sweep held such a cycle); pFBA drops them
            from cobra.flux_analysis import pfba
            clean = None
            for frac in (1.0, 0.9999):
                try:
                    sol = pfba(model, fraction_of_optimum=frac)
                    clean, report["clean_fraction"] = sol.fluxes.to_dict(), frac
                    break
                except Exception:  # infeasible at the full optimum, numerically
                    continue
            if clean is None:
                clean = {r.id: r.flux for r in model.reactions}
            for rid, (fv, bv) in orient.items():
                if not fv and not bv:
                    clean[rid] = 0.0  # held at 0 by its binaries; drop the solver's +-1e-12 noise
            report["clean_fluxes"] = clean
            report["clean_cycle"] = conformal_cycle(model, clean, tol=tol, pins=law.pins, threads=threads)
    finally:
        for var, lb, ub in saved:
            var.set_bounds(lb, ub)


# --------------------------------------------------------------------------- the generic audit
_AUDIT_MODULE = "modelseedpy.core.msenergyaudit"


def _fallback_energy_audit():
    """FALLBACK, TO BE REMOVED once the ModelSEEDpy that MSCommunity needs has
    modelseedpy.core.msenergyaudit (ModelSEEDpy branch energy-audit, commit db96d56): delete this
    function and the fallback branch of _load_energy_audit.

    A copy of modelseedpy/core/msenergyaudit.py as of that commit, its module docstring left out
    and its body indented into this function so that it stays in this file; returns its names as
    a namespace. Change the module, not this copy."""
    import glob
    import json
    import logging
    import os
    import re
    from contextlib import contextmanager
    from math import isnan
    from math import log as ln

    from cobra import Reaction
    from optlang.symbolics import Zero

    logger = logging.getLogger(__name__)

    RT = 0.0019872 * 298.15  # kcal/mol at 25 C, the units of the ModelSEED dG values

    # phosphoanhydride hydrolysis pins: (triphosphate, diphosphate); diphosphate None is
    # pyrophosphate (PPi + H2O -> 2 Pi + H+); every other pin is NTP + H2O -> NDP + Pi + H+
    ATP, ADP, PI, PPI = "cpd00002", "cpd00008", "cpd00009", "cpd00012"
    H2O, PROTON = "cpd00001", "cpd00067"
    ATP_PINS = ((ATP, ADP),)
    NTP_PINS = (
        (ATP, ADP),
        ("cpd00038", "cpd00031"),  # GTP/GDP
        ("cpd00052", "cpd00096"),  # CTP/CDP
        ("cpd00062", "cpd00014"),  # UTP/UDP
        (PPI, None),  # pyrophosphate
    )
    TEST_PREFIXES = ("ATPtest_", "NTPtest_")
    BIOMASS_CPD = "cpd11416"

    # electrons each reduced product carries, relative to CO2, H2O and H+:
    # H2, methane, formate, ethanol, L-lactate, D-lactate, succinate
    REDUCED_PRODUCTS = {
        "cpd11640": 2,
        "cpd01024": 8,
        "cpd00047": 2,
        "cpd00363": 12,
        "cpd00159": 12,
        "cpd00221": 12,
        "cpd00036": 14,
    }
    ELECTRON_TEST_INPUTS = ("cpd00011", "cpd00001", "cpd00067")  # CO2, H2O, H+

    # the boundary-ion arms of energy_from_nothing; the yield test's background uptakes
    BOUNDARY_ION_ARMS = (("H+", (PROTON,)), ("H+,Pi", (PROTON, PI)))
    YIELD_OPEN = (H2O, PROTON, PI, "cpd00011")  # H2O, H+, Pi, CO2
    # left out of the "medium" ion arm: electron acceptors (a donor-free test cannot use
    # them, and they are not ions in the sense meant here) and inorganic electron donors
    # (H2 + CO2 is a real energy source for methanogens and acetogens)
    ELECTRON_ACCEPTORS = {
        "cpd00007",
        "cpd00209",
        "cpd00075",
        "cpd00048",
        "cpd00106",
        "cpd00659",
        "cpd10516",
        "cpd00268",
    }
    INORGANIC_DONORS = {"cpd11640", "cpd00239", "cpd10515"}  # H2, H2S, Fe2+
    INORGANIC_CARBON = {"cpd00011", "cpd00242"}  # CO2, HCO3-

    # the thermodynamic gate of find_energy_cycle
    NOT_IN_CONCENTRATION_TERM = {H2O, PROTON}  # fixed by the pH of dG'0
    THERMO_SOURCES = ("eQuilibrator", "dGPredictor", "Group contribution")

    _CYTOSOL = re.compile(r"^c\d*$")
    _MSID = re.compile(r"^(rxn\d{5})")


    # ---------------------------------------------------------------- pins and tests
    def resolve_pins(pins=None, pin_atp=True):
        """The pins as a tuple of (triphosphate, diphosphate-or-None) compound ids.

        pins: None (ATP_PINS, or no pins with pin_atp=False), "atp", "ntp" (NTP_PINS), or an
        iterable of (triphosphate, diphosphate-or-None) pairs."""
        if pins is None:
            return ATP_PINS if pin_atp else ()
        if isinstance(pins, str):
            if pins.lower() == "ntp":
                return NTP_PINS
            if pins.lower() == "atp":
                return ATP_PINS
            raise ValueError(
                f"pins must be None, 'atp', 'ntp' or a list of pairs, not {pins!r}"
            )
        return tuple((tri, di) for tri, di in pins)


    def hydrolysis_stoichiometry(tri, di):
        """{compound: coefficient} of tri + H2O -> di + Pi + H+, or with di None of
        PPi + H2O -> 2 Pi + H+."""
        if di is None:
            return {tri: -1, H2O: -1, PI: 2, PROTON: 1}
        return {tri: -1, H2O: -1, di: 1, PI: 1, PROTON: 1}


    def _cpd(met):
        return met.id.split("_")[0]


    def _selected(compartments, tri, comp):
        if compartments is None:
            return True
        if callable(compartments):
            return bool(compartments(tri, comp))
        if isinstance(compartments, str):  # one id: "c10" must not select "c1"
            compartments = (compartments,)
        return comp in compartments


    def hydrolysis_sites(model, pins=ATP_PINS, compartments=None):
        """[(triphosphate, compartment, {metabolite id: coefficient})]: every cytosol (c,
        c0, c1, ...) that holds all the metabolites of a pin's hydrolysis, in pin order,
        then model order.

        pins as in resolve_pins. compartments: None (every such cytosol), one compartment
        id, an iterable of ids, or a callable (triphosphate, compartment) -> bool."""
        mets = {m.id for m in model.metabolites}
        out = []
        for tri, di in resolve_pins(pins):
            st = hydrolysis_stoichiometry(tri, di)
            for met in model.metabolites:
                comp = met.compartment or ""
                if _cpd(met) != tri or not _CYTOSOL.match(comp):
                    continue
                if not _selected(compartments, tri, comp):
                    continue
                if all(f"{c}_{comp}" in mets for c in st):
                    out.append((tri, comp, {f"{c}_{comp}": n for c, n in st.items()}))
        return out


    def add_hydrolysis_tests(model, pins=None, compartments=None):
        """Add one hydrolysis test reaction per pin and site (hydrolysis_sites), each
        irreversible with an upper bound of 1000 mmol/gDW/h: ATP + H2O -> ADP + Pi + H+ with
        id ATPtest_<comp>, and, for the other pins, NTPtest_<compound>_<comp>. Call inside
        `with model:` so they go again. Returns {compartment: reaction} for ATP and
        {(compound, compartment): reaction} for the other pins.

        Maximising the sum of their fluxes with every uptake closed gives the ATP (and
        nucleotide) a model makes from nothing; a GTP loop that feeds biomass is invisible
        to the ATP tests alone, hence pins="ntp". With no site at all (metabolite ids not in
        the ModelSEED namespace, say) nothing is added and a warning is logged: every test
        built on these would then read 0 whatever the model makes."""
        mets = {met.id: met for met in model.metabolites}
        tests = {}
        for tri, comp, st in hydrolysis_sites(model, resolve_pins(pins), compartments):
            if tri == ATP:
                rid, label, key = f"ATPtest_{comp}", "ATP", comp
            else:
                rid, label, key = f"NTPtest_{tri}_{comp}", tri, (tri, comp)
            rxn = Reaction(
                rid,
                name=f"{label} hydrolysis test {comp}",
                lower_bound=0,
                upper_bound=1000,
            )
            rxn.add_metabolites({mets[mid]: n for mid, n in st.items()})
            tests[key] = rxn
        if not tests:
            logger.warning(
                "%s: no cytosol holds all the metabolites of a pinned hydrolysis "
                "(ModelSEED ids such as cpd00002_c0); the energy tests measure nothing",
                model.id,
            )
        model.add_reactions(list(tests.values()))
        return tests


    def _summary_by_test(tests, tol):
        """{compartment: flux} of the ATP tests and {"<compound>_<comp>": flux} of the
        others, each above tol."""
        atp, ntp = {}, {}
        for key, rxn in tests.items():
            if rxn.flux <= tol:
                continue
            if isinstance(key, tuple):
                ntp[f"{key[0]}_{key[1]}"] = rxn.flux
            else:
                atp[key] = rxn.flux
        return atp, ntp


    # ---------------------------------------------------------------- sealing
    def _rows_to_lift(model, rows):
        if rows is None or rows is False:
            return []
        if isinstance(rows, str):
            return [c for c in model.constraints if c.name.endswith(rows)]
        if callable(rows):
            return [c for c in model.constraints if rows(c)]
        names = set(rows)
        missing = names - {c.name for c in model.constraints}
        if missing:
            raise ValueError(f"no constraint rows named {sorted(missing)}")
        return [c for c in model.constraints if c.name in names]


    @contextmanager
    def lifted_rows(model, rows=None):
        """Lift constraint rows for the duration of the block and restore them on exit.

        rows: None (the default: lift nothing), a string (every row whose name ends with
        it), an iterable of row names, or a callable taking an optlang constraint and
        returning a bool. A finite bound goes to +-1e30, not None (optlang's Gurobi
        interface refuses a free row). `with model:` does not track row bounds, so this
        restores them itself. Yields the lifted rows' names."""
        selected = _rows_to_lift(model, rows)
        saved = [(c, c.lb, c.ub) for c in selected]
        try:
            for c in selected:
                if c.ub is not None:
                    c.ub = 1e30
                if c.lb is not None:
                    c.lb = -1e30
            yield [c.name for c in selected]
        finally:
            for c, lb, ub in saved:  # only the bounds changed above (not None) go back
                if ub is not None:
                    c.ub = ub
                if lb is not None:
                    c.lb = lb


    def release_forced_fluxes(model):
        """Let every non-boundary reaction stop: a positive lower bound (a maintenance-ATP
        floor, a measured rate fixed as an equality) goes to 0, and so does a negative upper
        bound.

        A test of what a model makes from nothing has to do this first: a forced flux cannot
        run with every uptake closed, so with it in place the sealed LP is infeasible rather
        than 0. Call inside `with model:`. Returns {reaction id: (old lower bound, old upper
        bound)} of the reactions released (mmol/gDW/h)."""
        released = {}
        for rxn in model.reactions:
            if rxn.boundary:
                continue
            lb, ub = rxn.bounds
            if lb > 0 or ub < 0:
                released[rxn.id] = (lb, ub)
                rxn.bounds = (min(lb, 0), max(ub, 0))
        return released


    def close_uptakes(
        model,
        open_cpds=(),
        strict=False,
        biomass_outlet=True,
        extracellular="e0",
        biomass_cpd=BIOMASS_CPD,
    ):
        """Let every boundary reaction (exchange, sink, demand) only remove matter, except
        the exchanges of `open_cpds` in the `extracellular` compartment, which may also
        supply up to 1000 mmol/gDW/h. Call inside `with model:`.

        strict=True closes removal as well: every boundary reaction gets (0, 0), except
        `open_cpds` and, with biomass_outlet, the outlets of the biomass pseudo-metabolite
        (`biomass_cpd`), so that a growth test can still measure something. With
        biomass_outlet=False strict is literal and any growth from nothing is 0 by
        construction.

        "Remove" follows each reaction's own direction: forward for one written as
        consumption (met <=>, the cobra convention), reverse for one written as production
        (<=> met)."""
        for rxn in model.boundary:
            met = next(iter(rxn.metabolites))
            consumes = rxn.metabolites[met] < 0
            lb, ub = rxn.lower_bound, rxn.upper_bound
            if open_cpds and _cpd(met) in open_cpds and met.compartment == extracellular:
                rxn.bounds = (-1000, max(ub, 0)) if consumes else (min(lb, 0), 1000)
            elif strict and not (biomass_outlet and _cpd(met) == biomass_cpd):
                rxn.bounds = (0, 0)
            else:
                rxn.bounds = (0, max(ub, 0)) if consumes else (min(lb, 0), 0)


    def seal(
        model,
        closure="lax",
        release_floors=True,
        open_cpds=(),
        biomass_outlet=True,
        extracellular="e0",
        biomass_cpd=BIOMASS_CPD,
    ):
        """release_forced_fluxes (if release_floors) and close_uptakes with the given
        closure: "lax", boundary reactions may only remove matter; "strict", every boundary
        reaction closed (but the biomass outlets, with biomass_outlet). Call inside `with
        model:`. Returns the released bounds {reaction id: (old lower bound, old upper
        bound)}."""
        if closure not in ("lax", "strict"):
            raise ValueError(f"closure must be 'lax' or 'strict', not {closure!r}")
        released = release_forced_fluxes(model) if release_floors else {}
        close_uptakes(
            model,
            open_cpds=open_cpds,
            strict=closure == "strict",
            biomass_outlet=biomass_outlet,
            extracellular=extracellular,
            biomass_cpd=biomass_cpd,
        )
        return released


    def _value(model):
        v = model.slim_optimize(error_value=float("nan"))
        return None if isnan(v) else v


    def _maximise_tests(model, tests):
        made = sum(r.flux_expression for r in tests.values())
        model.objective = model.problem.Objective(made, direction="max")
        return _value(model)


    def medium_inorganic_uptakes(model, formulas=None, exclude=None, extracellular="e0"):
        """The `extracellular` compounds the model's current medium lets in that are
        inorganic -- no carbon in the formula, or CO2 / HCO3- -- other than electron
        acceptors and inorganic electron donors (ELECTRON_ACCEPTORS, INORGANIC_DONORS;
        `exclude` replaces both).

        formulas: {compound: formula}, e.g. from the ModelSEED database, for metabolites
        that carry none. Returns (compounds, compounds with no formula), each sorted."""
        if exclude is None:
            exclude = ELECTRON_ACCEPTORS | INORGANIC_DONORS
        exclude = set(exclude)
        formulas = formulas or {}
        ions, unknown = set(), set()
        for rxn in model.boundary:
            met = next(iter(rxn.metabolites))
            if met.compartment != extracellular:
                continue
            if rxn.metabolites[met] < 0:
                uptake = rxn.lower_bound < 0
            else:
                uptake = rxn.upper_bound > 0
            cpd = _cpd(met)
            if not uptake or cpd in exclude:
                continue
            formula = formulas.get(cpd) or met.formula or ""
            if not formula:
                unknown.add(cpd)
                continue
            if re.search(r"C(?![a-z])", formula) and cpd not in INORGANIC_CARBON:
                continue
            ions.add(cpd)
        return sorted(ions), sorted(unknown)


    # ---------------------------------------------------------------- the tests
    def _boundary_ion_arms(model, arms, release_floors, pins, tol, compartments, extra):
        """ATP made (summed tests) with every uptake closed except each arm's compounds
        (lax)."""
        out = {}
        for label, cpds in arms:
            with model:
                seal(model, "lax", release_floors, open_cpds=tuple(cpds), **extra)
                tests = add_hydrolysis_tests(model, pins, compartments)
                v = _maximise_tests(model, tests)
                rec = {"open": list(cpds), "atp": v}
                if v is not None and v > tol:
                    rec["atp_by_compartment"], ntp = _summary_by_test(tests, tol)
                    if ntp:
                        rec["ntp_by_test"] = ntp
                    fluxes = {r.id: r.flux for r in model.boundary}
                    rec["boundary"] = {k: v for k, v in fluxes.items() if abs(v) > tol}
            out[label] = rec
        return out


    def energy_from_nothing(
        model,
        biomass="bio1",
        electrons=True,
        tol=1e-6,
        release_floors=True,
        closure="lax",
        biomass_outlet=True,
        lift_rows=None,
        boundary_ions=False,
        pins=None,
        formulas=None,
        compartments=None,
        extracellular="e0",
        biomass_cpd=BIOMASS_CPD,
    ):
        """What the model makes from nothing. All values in mmol/gDW/h (growth: the biomass
        reaction's flux); each is 0 for a model without energy-generating cycles, and None
        means the LP failed (infeasible).

        atp_from_nothing        max total ATP hydrolysis over the tests
                                (add_hydrolysis_tests) with every uptake closed; with
                                pins="ntp", ATP, GTP, CTP, UTP and PPi hydrolysis summed
        atp_by_compartment      {compartment: ATP test flux} at that optimum, above `tol`
        ntp_by_test             {"<compound>_<comp>": flux}, the other hydrolysis tests
                                (only when pins are not ATP alone)
        n_atp_tests             the number of ATP test reactions
        growth_from_nothing     max `biomass` flux with every uptake closed (skipped if
                                biomass is None or not in the model)
        electrons_from_nothing  max electrons exported in REDUCED_PRODUCTS with only CO2,
                                H2O and H+ coming in (electrons=True;
                                electrons_from_nothing)
        released                {reaction id: old bounds} of the forced fluxes released
        rows_lifted             names of the constraint rows lifted for the tests

        closure         "lax" (default) or "strict"; see the module docstring
        release_floors  True (default): release forced fluxes first (release_forced_fluxes),
                        else a floor such as ATP maintenance makes the test infeasible
                        (None), not 0
        biomass_outlet  with "strict", keep the biomass outlets able to remove biomass;
                        False makes growth_from_nothing 0 by construction (the electron test
                        keeps its own closure: CO2, H2O and H+ in, anything out)
        lift_rows       constraint rows to lift for the tests (lifted_rows); None lifts
                        nothing
        boundary_ions   False (default); True adds arms that reopen H+ ("H+") and H+ with
                        phosphate ("H+,Pi") and report, per arm, the ATP made, the boundary
                        fluxes at that optimum and, as atp_from_boundary_ions, the largest.
                        A clean model makes none: taking in Pi and H+ and sending out, say,
                        pyrophosphate is uphill. "medium" adds an arm "ions" with the
                        medium's inorganic uptakes too (medium_inorganic_uptakes, with
                        `formulas` for metabolites that carry none); a list of compounds
                        adds that arm with those compounds.
        pins            None (ATP tests only), "ntp", or a list of pairs (resolve_pins)
        compartments    where to test (hydrolysis_sites); None: every cytosol"""
        out = {"closure": closure}
        pins = resolve_pins(pins)
        extra = {"extracellular": extracellular, "biomass_cpd": biomass_cpd}
        arms = []
        if boundary_ions is not False and boundary_ions is not None:
            arms = list(BOUNDARY_ION_ARMS)
            ions = []
            if isinstance(boundary_ions, str):
                if boundary_ions != "medium":
                    raise ValueError(
                        "boundary_ions must be True, 'medium' or a list of compounds, "
                        f"not {boundary_ions!r}"
                    )
                ions, unknown = medium_inorganic_uptakes(
                    model, formulas, extracellular=extracellular
                )
                out["ions_without_formula"] = unknown
            elif boundary_ions is not True:
                ions = list(boundary_ions)
            if ions:
                arms.append(("ions", tuple(dict.fromkeys((PROTON, PI) + tuple(ions)))))
        with lifted_rows(model, lift_rows) as lifted:
            out["rows_lifted"] = lifted
            with model:
                released = seal(
                    model, closure, release_floors, biomass_outlet=biomass_outlet, **extra
                )
                out["released"] = released
                tests = add_hydrolysis_tests(model, pins, compartments)
                out["atp_from_nothing"] = _maximise_tests(model, tests)
                if out["atp_from_nothing"] is not None:
                    atp, ntp = _summary_by_test(tests, tol)
                else:
                    atp, ntp = {}, {}
                out["atp_by_compartment"] = atp
                if pins != ATP_PINS:
                    out["ntp_by_test"] = ntp
                out["n_atp_tests"] = sum(1 for k in tests if not isinstance(k, tuple))
                if biomass is not None and biomass in model.reactions:
                    model.objective = biomass
                    out["growth_from_nothing"] = _value(model)
            if arms:
                by_arm = _boundary_ion_arms(
                    model, arms, release_floors, pins, tol, compartments, extra
                )
                out["boundary_ions"] = by_arm
                vals = [a["atp"] for a in by_arm.values()]
                vals = [v for v in vals if v is not None]
                out["atp_from_boundary_ions"] = max(vals) if vals else None
            if electrons:
                e = electrons_from_nothing(
                    model, release_floors=release_floors, extracellular=extracellular
                )
                e.pop("rows_lifted", None)
                out.update(e)
        return out


    def electrons_from_nothing(
        model, release_floors=True, lift_rows=None, extracellular="e0"
    ):
        """Max electrons exported in REDUCED_PRODUCTS (mmol e-/gDW/h) with only CO2, H2O and
        H+ coming in (lax closure otherwise), and the reduced products carrying them
        ({compound: export flux, mmol/gDW/h}). A clean model makes none, since reducing CO2
        with electrons from water needs an energy source. Electrons made by reactions that
        do not conserve charge are a curation problem, not a cycle; this reports them,
        nothing here removes them. release_floors and lift_rows as in
        energy_from_nothing."""
        with lifted_rows(model, lift_rows) as lifted, model:
            seal(
                model,
                "lax",
                release_floors,
                open_cpds=ELECTRON_TEST_INPUTS,
                extracellular=extracellular,
            )
            coefs, exports = {}, {}
            for rxn in model.exchanges:
                met = next(iter(rxn.metabolites))
                if met.compartment == extracellular and _cpd(met) in REDUCED_PRODUCTS:
                    # an exchange written as consumption exports with forward flux
                    w = REDUCED_PRODUCTS[_cpd(met)] * -rxn.metabolites[met]
                    coefs[rxn.forward_variable], coefs[rxn.reverse_variable] = w, -w
                    exports[_cpd(met)] = rxn
            if not coefs:
                return {
                    "electrons_from_nothing": 0.0,
                    "electron_carriers": {},
                    "rows_lifted": lifted,
                }
            model.objective = model.problem.Objective(Zero, direction="max")
            model.objective.set_linear_coefficients(coefs)
            v = _value(model)
            carriers = {}
            if v is not None:
                carriers = {c: r.flux for c, r in exports.items() if abs(r.flux) > 1e-9}
        return {
            "electrons_from_nothing": v,
            "electron_carriers": carriers,
            "rows_lifted": lifted,
        }


    def net_atp_yield(
        model,
        substrate,
        acceptor="cpd00007",
        open_cpds=YIELD_OPEN,
        rate=1.0,
        release_floors=True,
        lift_rows=None,
        pins=None,
        compartments=None,
        extracellular="e0",
    ):
        """The ATP yield of `substrate` (a compound id) net of what the same test makes
        without it.

        The test: every uptake closed except `open_cpds` (default H2O, H+, Pi, CO2), the
        `acceptor` (a compound, a list, or None) and the substrate at up to `rate`
        (mmol/gDW/h); forced fluxes released (release_floors), `lift_rows` lifted, the ATP
        tests summed and maximised. gross is that maximum per mmol of the allowed substrate;
        substrate_closed the same test with the substrate shut, per the same mmol; net =
        gross - substrate_closed. A model that makes ATP from H+ and Pi alone (a boundary
        route the sealed test cannot see) counts it in the gross yield; the net yield does
        not. A cycle-free model can still be too efficient, which only the net yield shows
        against the biochemistry (about 24 per glucose aerobically, say).

        Returns {substrate, rate, open, rows_lifted, atp_with_substrate, atp_without (mmol
        ATP/gDW/h), substrate_uptake (mmol/gDW/h), gross, substrate_closed, net (mmol ATP
        per mmol substrate)}."""
        acc = (acceptor,) if isinstance(acceptor, str) else tuple(acceptor or ())
        opened = tuple(dict.fromkeys(tuple(open_cpds) + acc))
        sid = f"{substrate}_{extracellular}"
        exs = [r for r in model.boundary if any(m.id == sid for m in r.metabolites)]
        if not exs:
            raise ValueError(f"no boundary reaction for {sid}")

        def run(with_substrate):
            with model:
                seal(
                    model,
                    "lax",
                    release_floors,
                    open_cpds=opened,
                    extracellular=extracellular,
                )
                met = model.metabolites.get_by_id(sid)
                for ex in exs:
                    coef = ex.metabolites[met]
                    lb, ub = ex.lower_bound, ex.upper_bound
                    # uptake of up to `rate`, whichever way the reaction is written
                    if with_substrate:
                        ex.bounds = (-rate, max(ub, 0)) if coef < 0 else (min(lb, 0), rate)
                    else:
                        ex.bounds = (0, max(ub, 0)) if coef < 0 else (min(lb, 0), 0)
                tests = add_hydrolysis_tests(model, pins, compartments)
                v = _maximise_tests(model, tests)
                up = None
                if v is not None:
                    up = sum(ex.flux * ex.metabolites[met] for ex in exs)
            return v, up

        with lifted_rows(model, lift_rows) as lifted:
            g, up = run(True)
            c, _ = run(False)
        out = {
            "substrate": substrate,
            "rate": rate,
            "open": list(opened),
            "rows_lifted": lifted,
            "atp_with_substrate": g,
            "atp_without": c,
            "substrate_uptake": up,
            "gross": None if g is None else g / rate,
            "substrate_closed": None if c is None else c / rate,
        }
        out["net"] = None if g is None or c is None else (g - c) / rate
        return out


    # ---------------------------------------------------------------- the cycle
    def msid(rxn):
        """The ModelSEED reaction id behind a model reaction id (rxn00571_c0 -> rxn00571),
        or None."""
        m = _MSID.match(rxn.id)
        return m.group(1) if m else None


    def load_msdb_thermo(msdb_path):
        """{reaction id: {reversibility, dg, err, source, name}} from a ModelSEEDDatabase
        checkout (Biochemistry/reaction_*.json). dg/err are dG'0 and its standard error in
        kcal/mol from the first of THERMO_SOURCES with a value (None if none has one)."""
        pattern = os.path.join(msdb_path, "Biochemistry", "reaction_*.json")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no Biochemistry/reaction_*.json under {msdb_path}")
        db = {}
        for f in files:
            with open(f) as fh:
                reactions = json.load(fh)
            for r in reactions:
                entry = {
                    "reversibility": r.get("reversibility"),
                    "name": r.get("name"),
                    "dg": None,
                    "err": None,
                    "source": None,
                }
                thermo = r.get("thermodynamics") or {}
                for src in THERMO_SOURCES:
                    val = thermo.get(src) or []
                    if len(val) < 2:
                        continue
                    if not all(isinstance(x, (int, float)) for x in val[:2]):
                        continue
                    if abs(val[0]) < 1e6 and abs(val[1]) < 1e6:
                        entry.update(dg=float(val[0]), err=float(val[1]), source=src)
                        break
                db[r["id"]] = entry
        return db


    def direction_thermo(rxn, direction, entry, c_min=1e-6, c_max=0.02, n_sigma=2.0):
        """dG'0 and dG'min (kcal/mol) of `rxn` run forward (direction > 0) or in reverse,
        and whether thermodynamics rules that direction out. None where there is no dG'0 or
        the reaction crosses a membrane (its dG then depends on the membrane potential).

        A direction is ruled out only when no concentrations in [c_min, c_max] (mol/L)
        could make it run, that is when dG'min - n_sigma * err > 0, where

            dG'min = dG'0 + RT * (sum_products n ln c_min - sum_substrates n ln c_max)

        with water and protons left out of the sums. entry is a load_msdb_thermo value.
        """
        if not entry or entry.get("dg") is None or len(rxn.compartments) > 1:
            return None
        sign = 1 if direction > 0 else -1
        dg0 = sign * entry["dg"]
        conc = 0.0
        for met, coef in rxn.metabolites.items():
            if _cpd(met) in NOT_IN_CONCENTRATION_TERM:
                continue
            n = sign * coef  # > 0: a product in the running direction
            conc += n * ln(c_min) if n > 0 else n * ln(c_max)
        dg_min = dg0 + RT * conc
        return {
            "dg0": dg0,
            "dg_min": dg_min,
            "err": entry["err"],
            "source": entry["source"],
            "infeasible": dg_min - n_sigma * entry["err"] > 0,
        }


    def _cycle_row(rxn, v, thermo, gapfilled, gate):
        base = msid(rxn)
        entry = thermo.get(base) if base else None
        rev = entry.get("reversibility") if entry else None
        row = {
            "reaction": rxn.id,
            "msid": base,
            "flux": v,
            "bounds": list(rxn.bounds),
            "db_reversibility": rev,
            "name": (entry or {}).get("name") or rxn.name,
            "against_db": bool((rev == ">" and v < 0) or (rev == "<" and v > 0)),
        }
        td = direction_thermo(rxn, v, entry, **gate)
        if td:
            row.update(
                dg0=td["dg0"],
                dg_min=td["dg_min"],
                dg_err=td["err"],
                dg_source=td["source"],
                infeasible=td["infeasible"],
            )
        else:
            row.update(dg0=None, dg_min=None, dg_err=None, dg_source=None, infeasible=False)
        row["gapfilled"] = bool(base and base in gapfilled)
        return row


    def find_energy_cycle(
        model,
        flux=1.0,
        compartment=None,
        thermo=None,
        gapfilled=None,
        tol=1e-7,
        release_floors=True,
        lift_rows=None,
        pins=None,
        compartments=None,
        extracellular="e0",
        **gate,
    ):
        """The smallest-total-flux cycle that makes `flux` mmol ATP/gDW/h with every uptake
        closed (lax closure, forced fluxes released with release_floors, `lift_rows`
        lifted), or, when the model makes some ATP from nothing but less than `flux`, the
        smallest-total-flux cycle at the most ATP it makes.

        compartment  make the ATP in this compartment's test; default the sum over all tests
        thermo       load_msdb_thermo() output: each reaction's database direction and dG
        gapfilled    a set of ModelSEED reaction ids gapfilled into the model
        pins         None (ATP tests) or "ntp" (the other hydrolysis tests count too)
        gate         c_min, c_max, n_sigma for direction_thermo

        Returns {status, atp_flux, reactions, fluxes, rows_lifted}. status is "cycle" when
        the model makes ATP (above tol) from nothing; "no cycle" when it makes none; or
        "infeasible" when the sealed model cannot run at all (a forced flux kept with
        release_floors=False, say), which says nothing about cycles. atp_flux is the ATP the
        cycle makes (mmol/gDW/h): `flux`, or less when that is all the model makes (0.0 with
        no cycle, None when infeasible). reactions has one row per reaction in the cycle
        (the test reactions left out): reaction, msid, flux, bounds (during the test),
        db_reversibility, name, against_db (runs against the database direction), dg0,
        dg_min, dg_err, dg_source, infeasible (direction_thermo), gapfilled. fluxes is
        {reaction id: flux} of every reaction above tol, tests included."""
        gapfilled, thermo = set(gapfilled or ()), thermo or {}
        with lifted_rows(model, lift_rows) as lifted:
            out = _find_energy_cycle(
                model,
                flux,
                compartment,
                thermo,
                gapfilled,
                tol,
                release_floors,
                resolve_pins(pins),
                compartments,
                extracellular,
                gate,
            )
        out["rows_lifted"] = lifted
        return out


    def _find_energy_cycle(
        model,
        flux,
        compartment,
        thermo,
        gapfilled,
        tol,
        release_floors,
        pins,
        compartments,
        extracellular,
        gate,
    ):
        with model:
            seal(model, "lax", release_floors, extracellular=extracellular)
            tests = add_hydrolysis_tests(model, pins, compartments)
            if compartment is not None:
                made = tests[compartment].flux_expression
            else:
                made = sum(r.flux_expression for r in tests.values())
            con = model.problem.Constraint(made, lb=flux, ub=flux, name="energy_cycle_atp")
            model.add_cons_vars(con)
            _minimise_total_flux(model)
            atp_flux = flux
            if _value(model) is None:  # it cannot make `flux`: what can it make?
                con.lb, con.ub = 0, None
                model.objective = model.problem.Objective(made, direction="max")
                most = _value(model)
                if most is None or most <= tol:
                    con.lb, con.ub = 0, 0
                    _minimise_total_flux(model)
                    status = "no cycle" if _value(model) is not None else "infeasible"
                    return _no_cycle(status)
                con.ub, con.lb = most, most
                _minimise_total_flux(model)
                if _value(model) is None:  # the maximum, to within the solver's tolerance
                    con.lb = most * (1 - 1e-6)
                    if _value(model) is None:
                        return _no_cycle("infeasible")
                atp_flux = sum(r.flux for r in tests.values())
                if compartment is not None:
                    atp_flux = tests[compartment].flux
            fluxes = {r.id: r.flux for r in model.reactions if abs(r.flux) > tol}
            rows = []
            for rid, v in fluxes.items():
                if rid.startswith(TEST_PREFIXES):
                    continue
                rxn = model.reactions.get_by_id(rid)
                rows.append(_cycle_row(rxn, v, thermo, gapfilled, gate))
        return {
            "status": "cycle",
            "atp_flux": atp_flux,
            "reactions": rows,
            "fluxes": fluxes,
        }


    def _no_cycle(status):
        atp = 0.0 if status == "no cycle" else None
        return {"status": status, "atp_flux": atp, "reactions": [], "fluxes": {}}


    def _minimise_total_flux(model):
        model.objective = model.problem.Objective(Zero, direction="min")
        model.objective.set_linear_coefficients(
            {
                v: 1
                for r in model.reactions
                for v in (r.forward_variable, r.reverse_variable)
            }
        )


    def conformal_cycle(
        model, fluxes, tol=1e-7, pin_atp=True, lp="auto", threads=None, pins=None
    ):
        """The internal reactions of a cycle inside a flux distribution, or [] if it holds
        none.

        A cycle here is a steady-state flux u through internal reactions only, each run the
        way `fluxes` ({reaction id: flux}) runs it (support and signs within the flux's
        own), plus, with pin_atp, any amount of ATP hydrolysis in each cytosol: exactly what
        makes `fluxes` violate the loop law with ATP hydrolysis pinned downhill (no
        potentials can make every step of u downhill). One LP over the active reactions; the
        support of its solution is returned. pins="ntp" allows hydrolysis of every pinned
        nucleotide and PPi as well, so a GTP loop that feeds biomass is a cycle here too.

        lp: "gurobi" (gurobipy), "highs" (scipy's HiGHS, scipy >= 1.6) or "auto" (gurobi
        when gurobipy imports, else highs; and highs, with a warning, when Gurobi raises
        GurobiError, as a size-limited licence does on a large model); threads goes to the
        Gurobi environment (None: gurobi.env or Gurobi's default). Neither solver is a
        dependency of this module until this function is called."""
        import numpy as np
        import scipy.sparse as sp
        from scipy.optimize import linprog

        boundary = {r.id for r in model.boundary}
        active = [
            model.reactions.get_by_id(rid)
            for rid, v in fluxes.items()
            if abs(v) > tol and rid not in boundary and rid in model.reactions
        ]
        cols, signs = [], []
        for r in active:
            cols.append({m.id: c for m, c in r.metabolites.items()})
            signs.append(1.0 if fluxes[r.id] > 0 else -1.0)
        n_active = len(cols)
        for _, _, st in hydrolysis_sites(model, resolve_pins(pins, pin_atp)):
            cols.append(dict(st))
            signs.append(1.0)
        if not n_active:
            return []
        index = {}
        rows, cidx, vals = [], [], []
        for j, col in enumerate(cols):
            for mid, c in col.items():
                i = index.setdefault(mid, len(index))
                rows.append(i)
                cidx.append(j)
                vals.append(c * signs[j])
        A = sp.csr_matrix((vals, (rows, cidx)), shape=(len(index), len(cols)))
        # w_j = sign_j * u_j in [0, 1]; maximise the flux through the active reactions
        # only, so a pin alone (ATP hydrolysis with nothing making the ATP) is not a cycle
        c = np.concatenate([-np.ones(n_active), np.zeros(len(cols) - n_active)])
        auto = lp == "auto"
        if auto:
            try:
                import gurobipy  # noqa: F401

                lp = "gurobi"
            except ImportError:
                lp = "highs"
        solved = None
        if lp == "gurobi":
            import gurobipy as gp

            try:
                solved = _conformal_lp_gurobi(gp, c, A, threads)
            except gp.GurobiError as exc:
                if not auto:
                    raise
                logger.warning("conformal_cycle: Gurobi failed (%s); using HiGHS", exc)
        if solved is None:
            res = linprog(
                c,
                A_eq=A,
                b_eq=np.zeros(A.shape[0]),
                bounds=[(0, 1)] * len(cols),
                method="highs",
            )
            ok = res.status == 0
            solved = ok, (-res.fun if ok else 0.0), res.x
        ok, value, x = solved
        if not ok or value <= tol:
            return []
        return [active[j].id for j in range(n_active) if x[j] > tol]


    def _conformal_lp_gurobi(gp, c, A, threads):
        """conformal_cycle's LP in Gurobi: (optimal, objective value, solution)."""
        import numpy as np

        params = {"OutputFlag": 0}
        if threads:
            params["Threads"] = int(threads)
        env = gp.Env(params=params)
        try:
            g = gp.Model(env=env)
            try:
                w = g.addMVar(len(c), lb=0, ub=1, obj=c)
                g.addMConstr(A, w, "=", np.zeros(A.shape[0]))
                g.optimize()
                ok = g.Status == gp.GRB.OPTIMAL
                value = -g.ObjVal if g.SolCount else 0.0
                x = w.X.copy() if g.SolCount else None
            finally:
                g.dispose()
        finally:
            env.dispose()
        return ok, value, x
    return SimpleNamespace(**locals())


def _load_energy_audit():
    """modelseedpy.core.msenergyaudit, or, when the installed ModelSEEDpy lacks it, one warning
    and _fallback_energy_audit()."""
    try:
        return importlib.import_module(_AUDIT_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != _AUDIT_MODULE:
            raise  # no ModelSEEDpy at all, or the module is there but something it imports is not
    warnings.warn(f"{_AUDIT_MODULE} is not in the installed ModelSEEDpy, so mscommunity.commcycles "
                  "uses its own copy of the energy audit, which will be removed; install a ModelSEEDpy "
                  f"that has {_AUDIT_MODULE}.", UserWarning, stacklevel=2)
    return _fallback_energy_audit()


_ea = _load_energy_audit()

# the generic names this module has always offered, from the audit
RT, REDUCED_PRODUCTS, ELECTRON_TEST_INPUTS = _ea.RT, _ea.REDUCED_PRODUCTS, _ea.ELECTRON_TEST_INPUTS
NOT_IN_CONCENTRATION_TERM, THERMO_SOURCES = _ea.NOT_IN_CONCENTRATION_TERM, _ea.THERMO_SOURCES
ATP, ADP, PI, PPI, H2O, PROTON = _ea.ATP, _ea.ADP, _ea.PI, _ea.PPI, _ea.H2O, _ea.PROTON
ATP_PINS, NTP_PINS, TEST_PREFIXES = _ea.ATP_PINS, _ea.NTP_PINS, _ea.TEST_PREFIXES
BOUNDARY_ION_ARMS, YIELD_OPEN = _ea.BOUNDARY_ION_ARMS, _ea.YIELD_OPEN
ELECTRON_ACCEPTORS, INORGANIC_DONORS = _ea.ELECTRON_ACCEPTORS, _ea.INORGANIC_DONORS
INORGANIC_CARBON = _ea.INORGANIC_CARBON
_resolve_pins = _ea.resolve_pins
hydrolysis_sites = _ea.hydrolysis_sites
release_forced_fluxes = _ea.release_forced_fluxes
close_uptakes = _ea.close_uptakes
medium_inorganic_uptakes = _ea.medium_inorganic_uptakes
msid = _ea.msid
load_msdb_thermo = _ea.load_msdb_thermo
direction_thermo = _ea.direction_thermo
conformal_cycle = _ea.conformal_cycle

# Private names this module had before the audit moved, kept because analysis scripts outside
# the package import them (_seal and _maximise_tests: 3H11_R12_SynCom's kpbt and cyclebench
# scripts, codiffusion_bioreactor's energy_leak_check_v7eo2). Each takes the arguments the old
# one took. _summary_by_test, _boundary_ion_arms and _find_energy_cycle are not kept: their
# arguments changed and no caller was found.
_seal = _ea.seal  # (model, closure="lax", release_floors=True, open_cpds=(), biomass_outlet=True)
_maximise_tests = _ea._maximise_tests  # (model, tests): reads tests.values() only
_hydrolysis = _ea.hydrolysis_stoichiometry
_MSID = _ea._MSID
