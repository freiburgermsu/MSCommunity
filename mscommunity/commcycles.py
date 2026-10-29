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
                          profile allows. Provably leaves no energy-generating cycle, including
                          one that feeds biomass; LPs only. It keeps a reference flux wherever that
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

Not covered: the ATP pin is on ATP only (a cycle that makes GTP, say, and hands it to biomass is
excluded only where GTP is linked to ATP), and electrons made by reactions that do not conserve
charge are a curation problem, not a cycle -- electrons_from_nothing reports them, nothing here
removes them.
"""
import glob
import json
import logging
import os
import re
from math import log as ln, isnan

from cobra import Reaction
from optlang.symbolics import Zero

logger = logging.getLogger(__name__)

RT = 0.0019872 * 298.15  # kcal/mol at 25 C, the units of the ModelSEED dG values
ORIENT_TOL = 1e-5  # |dG| of a potential profile below this (x its margin) is 0: solver tolerance

# electrons each reduced product carries, relative to CO2, H2O and H+:
# H2, methane, formate, ethanol, L-lactate, D-lactate, succinate
REDUCED_PRODUCTS = {"cpd11640": 2, "cpd01024": 8, "cpd00047": 2, "cpd00363": 12,
                    "cpd00159": 12, "cpd00221": 12, "cpd00036": 14}
ELECTRON_TEST_INPUTS = ("cpd00011", "cpd00001", "cpd00067")  # CO2, H2O, H+
NOT_IN_CONCENTRATION_TERM = {"cpd00001", "cpd00067"}  # water, H+ (fixed by the pH of dG'0)
THERMO_SOURCES = ("eQuilibrator", "dGPredictor", "Group contribution")
_COMP = re.compile(r"^([a-z]+)(\d+)$")
_MSID = re.compile(r"^(rxn\d{5})")


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


def msid(rxn):
    """The ModelSEED reaction id behind a community reaction id (rxn00571_c16 -> rxn00571)."""
    m = _MSID.match(rxn.id)
    return m.group(1) if m else None


def _cpd(met):
    return met.id.split("_")[0]


def add_atp_tests(model):
    """One ATP hydrolysis reaction per member compartment that has ATP, ADP, phosphate, water and
    H+ (ATP + H2O -> ADP + Pi + H+, id ATPtest_c<i>). Call inside `with model:` so they go again.
    Returns {member index: reaction}."""
    mets = {met.id: met for met in model.metabolites}
    tests = {}
    for met in model.metabolites:
        m = _COMP.match(met.compartment or "")
        if _cpd(met) != "cpd00002" or not m or m.group(1) != "c":
            continue
        comp = met.compartment
        need = {c: mets.get(f"{c}_{comp}") for c in ("cpd00002", "cpd00001", "cpd00008", "cpd00009", "cpd00067")}
        if any(v is None for v in need.values()):
            continue
        rxn = Reaction(f"ATPtest_{comp}", name=f"ATP hydrolysis test {comp}", lower_bound=0, upper_bound=1000)
        rxn.add_metabolites({need["cpd00002"]: -1, need["cpd00001"]: -1,
                             need["cpd00008"]: 1, need["cpd00009"]: 1, need["cpd00067"]: 1})
        tests[int(m.group(2))] = rxn
    # c0 is the community compartment in a community and the cytosol in a single model
    if len(tests) > 1:
        tests.pop(0, None)
    model.add_reactions(list(tests.values()))
    return tests


def close_uptakes(model, open_cpds=()):
    """Every boundary reaction (exchange, sink, demand) may only remove matter, except the
    shared-pool exchanges of `open_cpds`, which may also supply up to 1000. Call inside `with model:`."""
    for rxn in model.boundary:
        met = next(iter(rxn.metabolites))
        if open_cpds and _cpd(met) in open_cpds and met.compartment == "e0":
            rxn.bounds = (-1000, max(rxn.upper_bound, 0))
        else:
            rxn.bounds = (0, max(rxn.upper_bound, 0))


def _value(model):
    v = model.slim_optimize(error_value=float("nan"))
    return None if isnan(v) else v


def energy_from_nothing(model, biomass="bio1", electrons=True, tol=1e-6):
    """What the community makes from nothing.

    atp_from_nothing        max total ATP hydrolysis over the members with every uptake closed
    atp_by_member           {member index: flux} at that optimum (members above `tol`)
    growth_from_nothing     max `biomass` flux with every uptake closed (skipped if it is None)
    electrons_from_nothing  max electrons exported in REDUCED_PRODUCTS with only CO2, H2O and H+
                            coming in; a clean community makes none, since reducing CO2 with
                            electrons from water needs an energy source
    All three are 0 for a community without energy-generating cycles. None means the LP failed
    (e.g. a reaction forced above 0 cannot run on nothing)."""
    out = {}
    with model:
        tests = add_atp_tests(model)
        close_uptakes(model)
        model.objective = model.problem.Objective(sum(r.flux_expression for r in tests.values()), direction="max")
        out["atp_from_nothing"] = _value(model)
        out["atp_by_member"] = ({i: r.flux for i, r in tests.items() if r.flux > tol}
                                if out["atp_from_nothing"] is not None else {})
        out["n_atp_tests"] = len(tests)
        if biomass is not None and biomass in model.reactions:
            model.objective = biomass
            out["growth_from_nothing"] = _value(model)
    if electrons:
        out.update(electrons_from_nothing(model))
    return out


def electrons_from_nothing(model):
    """Max electrons exported in REDUCED_PRODUCTS with only CO2, H2O and H+ coming in, and the
    reduced products carrying them ({compound: export flux})."""
    with model:
        close_uptakes(model, open_cpds=ELECTRON_TEST_INPUTS)
        coefs, exports = {}, {}
        for rxn in model.exchanges:
            met = next(iter(rxn.metabolites))
            if met.compartment == "e0" and _cpd(met) in REDUCED_PRODUCTS:
                # an exchange consumes its metabolite: export is forward flux
                w = REDUCED_PRODUCTS[_cpd(met)] * -rxn.metabolites[met]
                coefs[rxn.forward_variable], coefs[rxn.reverse_variable] = w, -w
                exports[_cpd(met)] = rxn
        if not coefs:
            return {"electrons_from_nothing": 0.0, "electron_carriers": {}}
        model.objective = model.problem.Objective(Zero, direction="max")
        model.objective.set_linear_coefficients(coefs)
        v = _value(model)
        carriers = {c: r.flux for c, r in exports.items() if abs(r.flux) > 1e-9} if v is not None else {}
    return {"electrons_from_nothing": v, "electron_carriers": carriers}


# --------------------------------------------------------------------------- thermodynamics
def load_msdb_thermo(msdb_path):
    """{reaction id: {reversibility, dg, err, source, name}} from a ModelSEEDDatabase checkout
    (Biochemistry/reaction_*.json). dg/err are dG'0 and its standard error in kcal/mol from the
    first of THERMO_SOURCES with a value (None if none has one)."""
    files = sorted(glob.glob(os.path.join(msdb_path, "Biochemistry", "reaction_*.json")))
    if not files:
        raise FileNotFoundError(f"no Biochemistry/reaction_*.json under {msdb_path}")
    db = {}
    for f in files:
        for r in json.load(open(f)):
            entry = {"reversibility": r.get("reversibility"), "name": r.get("name"),
                     "dg": None, "err": None, "source": None}
            thermo = r.get("thermodynamics") or {}
            for src in THERMO_SOURCES:
                val = thermo.get(src) or []
                if len(val) >= 2 and isinstance(val[0], (int, float)) and isinstance(val[1], (int, float)) \
                        and abs(val[0]) < 1e6 and abs(val[1]) < 1e6:
                    entry.update(dg=float(val[0]), err=float(val[1]), source=src)
                    break
            db[r["id"]] = entry
    return db


def direction_thermo(rxn, direction, entry, c_min=1e-6, c_max=0.02, n_sigma=2.0):
    """dG'0 and dG'min (kcal/mol) of `rxn` run forward (direction > 0) or in reverse, and whether
    thermodynamics rules that direction out (dG'min - n_sigma * err > 0). None where there is no
    dG'0 or the reaction crosses a membrane."""
    if not entry or entry.get("dg") is None or len(rxn.compartments) > 1:
        return None
    sign = 1 if direction > 0 else -1
    dg0 = sign * entry["dg"]
    conc = 0.0
    for met, coef in rxn.metabolites.items():
        if _cpd(met) in NOT_IN_CONCENTRATION_TERM:
            continue
        n = sign * coef  # > 0: a product in the running direction
        conc += n * ln(c_min) if n > 0 else n * ln(c_max)  # n < 0 for substrates: -|n| ln c_max
    dg_min = dg0 + RT * conc
    return {"dg0": dg0, "dg_min": dg_min, "err": entry["err"], "source": entry["source"],
            "infeasible": dg_min - n_sigma * entry["err"] > 0}


# --------------------------------------------------------------------------- the cycle
def find_energy_cycle(model, flux=1.0, member=None, thermo=None, names=None, gapfilled=None,
                      tol=1e-7, **gate):
    """The smallest-flux cycle that makes `flux` ATP with every uptake closed.

    member     make the ATP in this member (index); default any member
    thermo     load_msdb_thermo() output, for each reaction's database direction and dG
    names      {member index: member id}
    gapfilled  {member index or id: set of ModelSEED reaction ids gapfilled into that member}
    gate       c_min, c_max, n_sigma for direction_thermo

    Returns {status, reactions: [...], cross_fed: {e0 metabolite: {member: net production}}};
    status is "no cycle" when the community cannot make the ATP."""
    names, gapfilled, thermo = names or {}, gapfilled or {}, thermo or {}
    with model:
        tests = add_atp_tests(model)
        close_uptakes(model)
        made = tests[member].flux_expression if member is not None else sum(r.flux_expression for r in tests.values())
        con = model.problem.Constraint(made, lb=flux, ub=flux, name="energy_cycle_atp")
        model.add_cons_vars(con)
        model.objective = model.problem.Objective(Zero, direction="min")
        model.objective.set_linear_coefficients(
            {v: 1 for r in model.reactions for v in (r.forward_variable, r.reverse_variable)})
        if _value(model) is None:
            return {"status": "no cycle", "reactions": [], "cross_fed": {}}
        fluxes = {r.id: r.flux for r in model.reactions if abs(r.flux) > tol}
        boundary = {r.id for r in model.boundary}
        rows, cross_fed = [], {}
        for rid, v in fluxes.items():
            rxn = model.reactions.get_by_id(rid)
            idx = member_index(rxn)
            for met, coef in rxn.metabolites.items():
                if met.compartment == "e0" and rid not in boundary and idx:
                    mem = names.get(idx, idx)
                    cross_fed.setdefault(met.id, {}).setdefault(mem, 0.0)
                    cross_fed[met.id][mem] += coef * v
            if rid.startswith("ATPtest_"):
                continue
            base = msid(rxn)
            entry = thermo.get(base) if base else None
            row = {"reaction": rid, "msid": base, "member_index": idx, "member": names.get(idx, idx),
                   "flux": v, "bounds": list(rxn.bounds),
                   "db_reversibility": entry.get("reversibility") if entry else None,
                   "name": (entry or {}).get("name") or rxn.name}
            rev = row["db_reversibility"]
            row["against_db"] = bool((rev == ">" and v < 0) or (rev == "<" and v > 0))
            td = direction_thermo(rxn, v, entry, **gate)
            row.update({"dg0": td["dg0"], "dg_min": td["dg_min"], "dg_err": td["err"],
                        "dg_source": td["source"], "infeasible": td["infeasible"]} if td else
                       {"dg0": None, "dg_min": None, "dg_err": None, "dg_source": None, "infeasible": False})
            gf = gapfilled.get(idx, gapfilled.get(row["member"], ()))
            row["gapfilled"] = bool(base and base in gf)
            rows.append(row)
    cross_fed = {m: {k: round(x, 9) for k, x in d.items() if abs(x) > tol}
                 for m, d in cross_fed.items()}
    cross_fed = {m: d for m, d in cross_fed.items() if any(x > 0 for x in d.values())
                 and any(x < 0 for x in d.values())}
    return {"status": "cycle", "reactions": rows, "cross_fed": cross_fed,
            "members": sorted({str(r["member"]) for r in rows if r["member_index"]})}


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
                      c_min=1e-6, c_max=0.02, n_sigma=2.0, fallback=None, fallback_min_dg0=0.0):
    """Remove the community's energy-generating cycles, changing `model` in place.

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
    report = {"rules_applied": [], "rounds": [], "cuts": [], "kept_open": [], "status": "max rounds"}
    if rules:
        report["rules_applied"] = apply_direction_rules(model, rules)
    refused = set()
    for rnd in range(max_rounds):
        test = energy_from_nothing(model, electrons=False, biomass=None, tol=tol)
        atp = test["atp_from_nothing"]
        entry = {"round": rnd, "atp_from_nothing": atp, "atp_members": len(test["atp_by_member"])}
        report["rounds"].append(entry)
        if atp is None:
            report["status"] = "test failed"
            break
        if atp <= tol:
            report["status"] = "clean"
            break
        cycle = find_energy_cycle(model, thermo=thermo, names=names, gapfilled=gapfilled, **gate)
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

    The rows go on `model` and stay until remove(); solves become MILPs with two binaries per
    covered reaction. `lazy_loop_law` or `loop_capable_reactions` chooses the reactions."""

    def __init__(self, model, dg_margin=1.0, max_potential=1000.0, pin_atp=True):
        if dg_margin <= 0:
            raise ValueError("dg_margin must be > 0: with 0 the potentials can all be equal and every cycle passes")
        self.model, self.eps, self.M = model, float(dg_margin), float(max_potential)
        self.potentials, self.binaries, self.pinned = {}, {}, []
        self._objs = []
        if pin_atp:
            self._pin_atp()

    def _mu(self, met, new):
        var = self.potentials.get(met.id)
        if var is None:
            var = self.model.problem.Variable(f"llmu_{met.id}", lb=-self.M, ub=self.M)
            self.potentials[met.id] = var
            new.append(var)
        return var

    def _pin_atp(self):
        mets = {met.id: met for met in self.model.metabolites}
        new, rows = [], []
        for met in self.model.metabolites:
            m = _COMP.match(met.compartment or "")
            if _cpd(met) != "cpd00002" or not m or m.group(1) != "c":
                continue
            comp = met.compartment
            stoich = {"cpd00002": -1, "cpd00001": -1, "cpd00008": 1, "cpd00009": 1, "cpd00067": 1}
            if any(f"{c}_{comp}" not in mets for c in stoich):
                continue
            expr = sum(n * self._mu(mets[f"{c}_{comp}"], new) for c, n in stoich.items())
            rows.append(self.model.problem.Constraint(expr, ub=-self.eps, name=f"llpin_atp_{comp}"))
            self.pinned.append(comp)
        self.model.add_cons_vars(new + rows)
        self._objs += new + rows

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
            if row.name.startswith("llpin_atp_"):
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
        self._objs, self.potentials, self.binaries, self.pinned = [], {}, {}, []


def loop_capable_reactions(model, tol=1e-6, seeds=8, processes=None, seed=0):
    """The internal reactions that can carry flux with every boundary reaction shut and an ATP
    hydrolysis reaction in each member (add_atp_tests): the only ones a cycle can use, counting
    the cycles that would make ATP if something consumed it -- which LocalLoopLaw's ATP pin
    stands in for. Covering exactly these with LocalLoopLaw(pin_atp=True) is the full loop law
    (a reaction that cannot carry flux in that closed network is in no such cycle), with one
    binary for each of them instead of each of the community's reactions.

    `seeds` LPs with random objectives mark most of them; flux variability analysis settles the
    rest (`processes` workers)."""
    import random
    from cobra.flux_analysis import flux_variability_analysis
    rng = random.Random(seed)
    internal = [r for r in model.reactions if not r.boundary]
    found = set()
    with model:
        for r in model.boundary:
            r.bounds = (0, 0)
        add_atp_tests(model)
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

    Returns (law, report); the law's rows stay on `model` for the simulations that follow."""
    import time
    law = law or LocalLoopLaw(model, **law_args)
    report = {"rounds": [], "status": "max rounds", "pinned": list(law.pinned)}
    saved = model.solver.configuration.timeout
    if time_limit is not None:
        model.solver.configuration.timeout = time_limit
    try:
        for rnd in range(max_rounds):
            with model:
                tests = add_atp_tests(model)
                close_uptakes(model)
                model.objective = model.problem.Objective(sum(r.flux_expression for r in tests.values()),
                                                          direction="max")
                t = time.time()
                v = _value(model)
                status = model.solver.status
                active = ([r for r in model.reactions if abs(r.flux) > tol and not r.boundary
                           and not r.id.startswith("ATPtest_")] if v is not None and v > tol else [])
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
    finally:
        model.solver.configuration.timeout = saved
    report["covered"] = len(law.binaries)
    return law, report


# --------------------------------------------------------------------------- one potential profile
def member_reference(model, biomass_prefix="bio", tol=1e-9, member_models=None):
    """Each member growing alone on the model's current medium, by pFBA, the other members shut:
    {reaction id: flux} over all members (their reactions are disjoint), and {member index:
    growth}. A member that is free of energy-generating cycles alone gives a pFBA flux with no
    cycle in it, so this is a reference the potentials of `potential_profile` can always honour;
    it keeps every member's own growth feasible after `orient_by_potentials`. Members that cannot
    grow alone contribute nothing.

    member_models  {member index: the member's own cobra model} -- much faster on a large
                   community: each member is solved on its own model, with the community's
                   shared-pool uptake bounds, and its reaction ids are mapped the way
                   build_from_species_models renames them (rxn..._c0 -> rxn..._c<i>)."""
    from cobra.flux_analysis import pfba
    if member_models:
        return _member_reference_from_models(model, member_models, biomass_prefix, tol)
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
            sol = pfba(model)
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
            for rid, v in pfba(mm).fluxes.items():
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
                      w_reference=1.0, w_irreversible=1e2, lp="auto", threads=None, flux_floor=1e-3):
    """One metabolite potential vector for the whole community, from which `orient_by_potentials`
    fixes reaction directions. With ATP hydrolysis pinned downhill (dG <= -dg_margin in every
    member), no orientation it gives can run an energy-generating cycle: around any cycle the dG
    of its steps sum to zero, every allowed step has dG <= 0, so all are 0 -- and no step of an
    ATP-making cycle can be, because ATP hydrolysis is not. The same argument rules out a cycle
    that hands its ATP to biomass. This is an LP and needs no binaries; the price is that one
    profile serves every simulation, where the loop law would find a profile per solution.

    Two LPs, lexicographically:
      1. honour the reference: minimise the weighted dG against each `reference` flux direction
         ({reaction id: flux}, e.g. member_reference or a loop-law solution; or a list of
         (flux dict, weight) pairs, whose weights add up per reaction). A reference with no
         cycle in it (conformal_cycle) is honoured exactly; one with a cycle is broken where it
         costs least. A reference flux below flux_floor weighs in proportion to its size, so
         numerical trickle flows lose to the fluxes that carry the solution.
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
    rp, rq = np.zeros(nr), np.zeros(nr)    # stage 1: against the reference
    wp, wq = np.ones(nr), np.ones(nr)      # stage 2: orienting at all
    for j, r in enumerate(rxns):
        for ref, w in refs:
            v = ref.get(r.id, 0.0)
            if v > 0:
                rp[j] += w * min(1.0, v / flux_floor)
            elif v < 0:
                rq[j] += w * min(1.0, -v / flux_floor)
        if r.lower_bound >= 0 and r.upper_bound > 0:
            wp[j] = w_irreversible
        elif r.upper_bound <= 0 and r.lower_bound < 0:
            wq[j] = w_irreversible
    zeros = np.zeros(nm)
    A_ub, b_ub, pinned = sp.csr_matrix((0, nm + 2 * nr)), np.zeros(0), []
    if pin_atp:
        prow, pcol, pval = [], [], []
        stoich = {"cpd00002": -1, "cpd00001": -1, "cpd00008": 1, "cpd00009": 1, "cpd00067": 1}
        for met in model.metabolites:
            m = _COMP.match(met.compartment or "")
            if _cpd(met) != "cpd00002" or not m or m.group(1) != "c":
                continue
            comp = met.compartment
            if any(f"{k}_{comp}" not in mets for k in stoich):
                continue
            for k, n in stoich.items():
                prow.append(len(pinned)); pcol.append(mets[f"{k}_{comp}"]); pval.append(n)
            pinned.append(comp)
        if pinned:
            A_ub = sp.csr_matrix((pval, (prow, pcol)), shape=(len(pinned), nm + 2 * nr))
            b_ub = -dg_margin * np.ones(len(pinned))
    if lp == "auto":
        try:
            import gurobipy  # noqa: F401
            lp = "gurobi"
        except ImportError:
            lp = "highs-ipm"
    solve = (lambda c, A, b: _profile_gurobi(c, A_eq, A, b, nm, nr, max_potential, threads)) if lp == "gurobi" \
        else (lambda c, A, b: _profile_scipy(c, A_eq, A, b, nm, nr, max_potential, lp))
    c1 = np.concatenate([zeros, rp, rq])
    against = 0.0
    if c1.any():
        _, against = solve(c1, A_ub, b_ub)
        A_ub = sp.vstack([A_ub, sp.csr_matrix(c1)], format="csr")
        b_ub = np.concatenate([b_ub, [against * (1 + 1e-9) + 1e-7]])
    x, objective = solve(np.concatenate([zeros, wp, wq]), A_ub, b_ub)
    # dG from the LP's own split, not recomputed as St mu: with potentials up to max_potential
    # the equality rows hold only to the solver's feasibility tolerance, and St mu then puts
    # 1e-6-sized noise on a dG the vertex has at exactly 0 -- enough for orient_by_potentials to
    # close a direction at random
    dg = x[nm:nm + nr] - x[nm + nr:]
    tol = ORIENT_TOL * max(1.0, dg_margin)
    dg[np.abs(dg) <= tol] = 0.0
    report = {"reactions": nr, "metabolites": nm, "pinned": pinned, "against_cost": against,
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
    env = gp.Env(params={"OutputFlag": 0})
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


def orient_community(model, reference=None, rounds=1, tol=1e-9, member_models=None, **profile_args):
    """member_reference -> potential_profile -> orient_by_potentials on `model` (in place), on the
    model's current medium and objective.

    reference  None or "members": each member growing alone (member_reference); "pfba": the
               model's own pFBA optimum; "both": the members (weighted 10x) and the optimum; or
               {reaction id: flux}, e.g. a loop-law MILP solution, or a list of (flux, weight).
               The orientation keeps a reference feasible where it holds no cycle, so the
               objective it allows is at least the reference's, and "both" always leaves
               every member able to grow. Where the optimum leans on cycles, "both" keeps more
               (day 174, 88 members: 11.83 vs 6.17 for "pfba", of 18.21); where it holds few,
               "pfba" does (the nine cycle members: 19.61 vs 11.02, of 19.91).

    With rounds > 1, each further round adds the oriented model's pFBA optimum to the reference
    and derives a new profile from the original bounds. The optimum stays feasible (the new profile
    honours it), so the objective cannot fall, and directions the first profile fixed for no
    reason can reopen. member_models goes to member_reference. Returns a report with the
    objective of each round."""
    from cobra.flux_analysis import pfba
    original = {r.id: tuple(r.bounds) for r in model.reactions}
    boundary = {r.id for r in model.boundary}
    grown = None
    optimum = None
    if reference in ("pfba", "both"):
        # the unoriented optimum itself: kept whole when it holds no cycle (pFBA rarely pays
        # for one); whatever cycle it does hold, the soft weights orient against
        optimum = {rid: f for rid, f in pfba(model).fluxes.items() if abs(f) > tol and rid not in boundary}
    if reference is None or reference in ("members", "both"):
        members, grown = member_reference(model, member_models=member_models)
    if reference == "both":
        # each member alone first (cycle-free, so the community can always grow), then as much
        # of the optimum as agrees with them
        w = profile_args.pop("w_reference", 1e4)
        reference = [(members, 10 * w), (optimum, w)]
    elif reference == "pfba":
        reference = optimum
    elif reference is None or reference == "members":
        reference = members
    reference = list(reference) if isinstance(reference, list) else dict(reference)
    history, changed, profile = [], [], {}
    for k in range(rounds):
        for r in model.reactions:
            r.bounds = original[r.id]
        dg, profile = potential_profile(model, reference, **profile_args)
        changed = orient_by_potentials(model, dg)
        v = _value(model)
        against = profile["against_reference"]
        history.append({"round": k, "objective": v, "oriented": profile["oriented"],
                        "bounds_changed": len(changed),
                        "reference": [len(r) for r, _ in reference] if isinstance(reference, list) else len(reference),
                        "against_reference": [len(x) for x in against] if isinstance(reference, list) else len(against),
                        "irreversible_blocked": len(profile["irreversible_blocked"])})
        if k + 1 < rounds and v is not None and v > tol:
            new = {rid: f for rid, f in pfba(model).fluxes.items() if abs(f) > tol and rid not in boundary}
            if isinstance(reference, list):
                reference = reference + [(new, reference[-1][1])]
            else:
                reference.update(new)
    return {"history": history, "changed": changed, "profile": profile, "member_growth_alone": grown}


def conformal_cycle(model, fluxes, tol=1e-7, pin_atp=True, lp="auto"):
    """The internal reactions of a cycle inside a flux distribution, or [] if it holds none.

    A cycle here is a steady-state flux u through internal reactions only, each run the way
    `fluxes` runs it (support and signs within the flux's own), plus, with pin_atp, any amount of
    ATP hydrolysis in each member: exactly what makes `fluxes` violate the loop law with ATP
    hydrolysis pinned downhill (no potentials can make every step of u downhill). One LP over
    the active reactions; the support of its solution is returned. lp: "gurobi", "highs" or
    "auto" (gurobi when gurobipy imports)."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.optimize import linprog
    boundary = {r.id for r in model.boundary}
    active = [model.reactions.get_by_id(rid) for rid, v in fluxes.items()
              if abs(v) > tol and rid not in boundary and rid in model.reactions]
    cols, signs = [], []
    for r in active:
        cols.append({m.id: c for m, c in r.metabolites.items()})
        signs.append(1.0 if fluxes[r.id] > 0 else -1.0)
    n_active = len(cols)
    if pin_atp:
        mets = {m.id for m in model.metabolites}
        stoich = {"cpd00002": -1, "cpd00001": -1, "cpd00008": 1, "cpd00009": 1, "cpd00067": 1}
        for m in model.metabolites:
            mm = _COMP.match(m.compartment or "")
            if _cpd(m) == "cpd00002" and mm and mm.group(1) == "c":
                comp = m.compartment
                if all(f"{k}_{comp}" in mets for k in stoich):
                    cols.append({f"{k}_{comp}": n for k, n in stoich.items()})
                    signs.append(1.0)
    if not n_active:
        return []
    index = {}
    rows, cidx, vals = [], [], []
    for j, col in enumerate(cols):
        for mid, c in col.items():
            i = index.setdefault(mid, len(index))
            rows.append(i); cidx.append(j); vals.append(c * signs[j])
    A = sp.csr_matrix((vals, (rows, cidx)), shape=(len(index), len(cols)))
    # w_j = sign_j * u_j in [0, 1]; maximise the flux through the active reactions only, so a pin
    # alone (ATP hydrolysis with nothing making the ATP) is not a cycle
    c = np.concatenate([-np.ones(n_active), np.zeros(len(cols) - n_active)])
    if lp == "auto":
        try:
            import gurobipy  # noqa: F401
            lp = "gurobi"
        except ImportError:
            lp = "highs"
    if lp == "gurobi":
        import gurobipy as gp
        env = gp.Env(params={"OutputFlag": 0})
        g = gp.Model(env=env)
        w = g.addMVar(len(cols), lb=0, ub=1, obj=c)
        g.addMConstr(A, w, "=", np.zeros(len(index)))
        g.optimize()
        ok, value, x = g.Status == gp.GRB.OPTIMAL, -g.ObjVal if g.SolCount else 0.0, w.X.copy() if g.SolCount else None
        g.dispose(); env.dispose()
    else:
        res = linprog(c, A_eq=A, b_eq=np.zeros(len(index)), bounds=[(0, 1)] * len(cols), method="highs")
        ok, value, x = res.status == 0, -res.fun if res.status == 0 else 0.0, res.x
    if not ok or value <= tol:
        return []
    return [active[j].id for j in range(n_active) if x[j] > tol]


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


def loop_law_optimum(model, start=None, law=None, max_rounds=50, time_limit=None, tol=1e-7, time_budget=None,
                     **law_args):
    """The model's optimum under the loop law (ATP pinned), found by covering only the cycles the
    optimum tries to use: solve; if the solution holds a cycle (conformal_cycle), cover its
    reactions with LocalLoopLaw and solve again. The first solve is the plain LP; each later one
    is a MILP whose binaries are the covered reactions only, warm-started from `start` -- a flux
    with no cycle in it, e.g. an orient_community optimum, which every cover allows. At the end
    the solution holds no cycle, so it is the loop-law optimum (to the MILP's gap).

    time_limit caps each solve, time_budget the whole loop (seconds); when the budget runs out the
    status says so and `fluxes` is the last solution, which may still hold a cycle (check it with
    conformal_cycle before using it as a cycle-free reference).

    Returns (fluxes, report); the law's rows stay on `model` (report["law"]) until removed."""
    import time
    law = law or LocalLoopLaw(model, **law_args)
    report = {"rounds": [], "status": "max rounds", "law": law}
    saved = model.solver.configuration.timeout
    if time_limit is not None:
        model.solver.configuration.timeout = time_limit
    fluxes = {}
    began = time.time()
    try:
        for rnd in range(max_rounds):
            if time_budget is not None and time.time() - began > time_budget:
                report["status"] = "time budget"
                break
            started = bool(start) and bool(law.binaries) and law.start(start)
            t = time.time()
            v = _incumbent(model)
            row = {"round": rnd, "objective": v, "status": model.solver.status, "covered": len(law.binaries),
                   "started": started, "seconds": round(time.time() - t, 1)}
            report["rounds"].append(row)
            if v is None:
                report["status"] = f"solve failed ({model.solver.status})"
                break
            fluxes = {r.id: r.flux for r in model.reactions}
            cycle = conformal_cycle(model, fluxes, tol=tol)
            row["cycle"] = len(cycle)
            if not cycle:
                report["status"] = "no cycle"
                break
            row["added"] = law.add([model.reactions.get_by_id(rid) for rid in cycle])
            if not row["added"]:
                report["status"] = "cycle already covered (solver tolerance)"
                break
    finally:
        model.solver.configuration.timeout = saved
    return fluxes, report
