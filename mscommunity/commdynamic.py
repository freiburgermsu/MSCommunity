# -*- coding: utf-8 -*-
"""Multi-species dynamic FBA for MSCommunity communities: member biomasses X_m (gDW/L) and a shared
medium S_c (mM) integrated in time, with each member's growth and exchanges from FBA at every step.

Two forms (`form=`), which give different answers and must be stated with every result:

    "members"            DMMM (Zhuang et al. 2011, doi:10.1038/ismej.2010.117). Each member's own model
                         (member_model) is solved alone at every step, max mu_m, with its exchange of every
                         tracked compound c bounded by Michaelis-Menten kinetics on the shared pool,
                             release_m,c >= -vmax_m,c * S_c / (Km_m,c + S_c)     mmol per gDW of THAT member per h.
                         By-products enter the pool and are offered back to every member through the same
                         kinetics; members interact only through the pool (and O2, see below).
    "community_strict"   One LP of the community model per step (comm.util.model): bio1 shut, member drains
                         open, X_m variables (commmembers.member_biomass_variables) fixed at a_m = X_m / X,
                         every non-boundary member reaction bound multiplied by a_m (SteadyCom eq. 9, Chan et
                         al. 2017, so a member without biomass carries no flux), and a strict per-member cap on
                         EVERY tracked e0 compound,
                             uptake_m,c - q_m,c * X_m <= 0,   q_m,c = vmax_m,c S_c / (Km_m,c + S_c)
                         (commmembers.cap_member_uptake, basis "X"; gross uptake by default, cap_basis="net"
                         for the prototype's net rows), objective max sum_m v_bio,m, mu_m = v_bio,m / a_m.
                         Because every e0 compound is capped at its CURRENT pool concentration, a by-product
                         must accumulate in the medium before a partner can take it up (DMMM's assumption,
                         inside one LP); what differs from "members" is that O2 and the members' fluxes are
                         allocated jointly by one optimum.

Both forms solve lexicographically: max growth, then (tiebreak={...}) a chosen flux held at its min or max on
the optimal face, then pFBA (min sum |v| at growth >= (1 - 1e-7) of the optimum). The optimal face of a
member's FBA is usually wide in the by-products (on the CommPhitting maltose co-culture the E. coli acetate
split alone moves the P. fluorescens-window rate difference from -0.0004 to -0.039 1/h at OTR 2), so the
form and the tie-break are explicit parameters and are recorded on the Trajectory.

A member whose max growth is <= 1e-12 is idle for the step: mu = 0, but it still exchanges what its forced fluxes
need (the least total flux with no growth required, e.g. the substrate an ATPM floor burns). "community_strict"
solves a member below a_min of the culture with its own member LP, clamps every rate to the member's solo maximum on
the current pool, and releases the forced bounds of a member that cannot pay them when the community LP is
infeasible (see dynamic()).

Units
    X gDW/L; S mM (= mmol/L); specific rates mmol/gDW_member/h and 1/h; OTR mmol O2/L/h; kLa 1/h; D 1/h.
    In "community_strict" the LP itself is per gDW of community (the MSCommunity convention), with X_m = a_m.

What is tracked
    Every compound with an exchange in the models is either TRACKED (a pool S_c, starting at S0.get(c, 0); its
    uptake set by the kinetics alone, whatever the medium bound) or AMPLE (left at the model's exchange bounds,
    not tracked: salts, water, protons, CO2 ...). Default: ample = the compounds whose exchange lower bound is
    negative in the current medium and that are not in S0 or feed (and not O2); everything else is tracked, so
    by-products appear when a member secretes them. `ample=` or `tracked=` override it. A compound without a
    member-specific (vmax, Km) gets `default_kinetics` (10 mmol/gDW/h, 0.01 mM: the default uptake kinetics
    stated by Khazaei et al. 2020, doi:10.1126/sciadv.aba0353); give (0, Km) to forbid an uptake.

O2 (otr=)
    None          O2 left at the models' bounds (ample).
    float         a quasi-steady transfer limit OTR (mmol O2/L/h) on the whole culture; O2 is not a pool.
                  o2_rule decides how a limiting OTR is shared:
                    "demand" (members default)  each member is solved with O2 at its model bound; if the summed
                                     demand sum_m o2_m X_m exceeds OTR, each member is capped at o2_m * OTR / demand
                                     (per gDW of member) and re-solved once (demand-proportional sharing)
                    "equal"          every member capped at OTR / X per gDW (one solve)
                    "joint" (community_strict default)  the community LP allocates O2: EX_O2 >= -OTR / X
    (kLa, C*)     dissolved O2 is a pool with transfer kLa (C* - C) and per-member uptake kinetics (from
                  `kinetics`, else default_kinetics); o2_rule does not apply.

Integration (the update is positivity-preserving, and no concentration is ever reset to 0)
    Rates are held over a step; X_m(t + h) = X_m exp((mu_m - D) h), and member m exchanges r_m,c * G_m with
    G_m = X_m (exp((mu_m - D) h) - 1) / (mu_m - D) its biomass integral. Per compound, with production P (member
    secretion, feed D S_f, O2 transfer) and uptake U over the step and linear loss L (dilution D h, O2 kLa h):
        explicit  S' = S + P - U - L S          used when S' >= (1 - max_depletion) S: exactly conservative
        Patankar  S' = (S + P) / (1 + L + U / S) (modified Patankar-Euler; positive for any h), used when the
                  members then realise their uptake to within rtol (|1 - S'/S| U <= atol + rtol U): stiff
                  quasi-steady pools (dissolved O2) and the tail of a depleted pool
    Otherwise the step is shortened (step control; no LP is re-solved, the rates depend only on the state) down
    to dt_min, where Patankar is forced and a warning recorded. The ledger records what each member realised
    (Patankar-weighted uptakes), so ledger and pools balance exactly. Pools below s_min are not offered to the
    members (caps 0), so a depleted pool keeps its last positive value instead of decaying to underflow.

Chemostat / fed-batch: D (1/h, a number or a callable of t) dilutes biomass and pools and brings in `feed`
({cpd: mM}, or a callable of t). A fed-batch at volume V(t) with inflow F(t) is the same system with D = F / V.

Functions take the MSCommunity `comm` (MSCommunity wraps them); member_model returns a stand-alone cobra.Model.
Reference prototypes: 3H11_R12_SynCom/Metabolic_modeling/abund_dynamic_lib.py (forms I and C), with its
verifier's round-3 files (data/multisystem3/abundance/_verify/m-dynamic/r3/).

Validation (tests/test_p_dynamic.py; the CommPhitting E. coli + P. fluorescens maltose co-culture, OD-based E. coli
fraction at the mid-exponential / end-of-window / final reads, R_pf = mu_ec - mu_pf in the P. fluorescens window)
    members, OTR 2, pFBA         0.9517 / 0.9813 / 0.9794, R_pf -0.0351   (prototype, fixed dt 0.05: 0.9517 / 0.9813 /
                                 0.9794, -0.0359; dt 0.02: -0.0355; the verifier's integrator -0.0364)
    ... min / max E. coli acetate  R_pf -0.0005 / -0.0371                 (prototype -0.0004 / -0.0389)
    community_strict, OTR 1,     0.9516 / 0.9793 / 0.9655, R_pf -0.218    (prototype stepper 0.9516 / 0.9793 / 0.9663,
    orient_both repair           (gross caps: identical to 4 decimals)      -0.211; verifier 0.952 / 0.980 / 0.968, -0.203)
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from cobra import Model, Reaction
from optlang.symbolics import add

from mscommunity.commmembers import (
    cap_member_uptake, lifted_kinetic_rows, member_biomass_variables, member_reactions, member_table,
    member_uptake_expression, reactions_by_member)

logger = logging.getLogger(__name__)

SHARED = "e0"
O2 = "cpd00007"
DEFAULT_KINETICS = (10.0, 0.01)          # vmax mmol/gDW/h, Km mM (Khazaei et al. 2020 default uptake kinetics)
FORMS = ("members", "community_strict")
O2_RULES = {"members": ("demand", "equal"), "community_strict": ("joint", "equal")}
CAP_BASES = ("gross", "net")
FACE = 1e-7          # growth held at (1 - FACE) of the optimum for the tie-break and pFBA stages
FACE_RETRY = 1e-5    # ... and at (1 - FACE_RETRY) when the pFBA stage fails at FACE
IDLE = 1e-12         # growth at or below this: the member is idle for the step (mu = 0; its exchanges are those
                     # of the least total flux with no growth required, e.g. the substrate an ATPM floor burns)
RELAXED = 1e30
A_MIN = 1e-4         # community_strict: a member below this share of the culture is solved by its own member LP
CLAMP_ATOL = 1e-9    # community_strict: a rate above the member's solo maximum by more than this is clamped to it,
CLAMP_WARN_RTOL, CLAMP_WARN_ATOL = 1e-4, 1e-4   # ... with a warning when the excess is beyond this band (1/h): smaller
                                                # excesses are solver tolerance and immaterial (doubling time > 0.8 y)

__all__ = ["DEFAULT_KINETICS", "Trajectory", "member_model", "dynamic", "simulate_models",
           "calibrate_monoculture"]


# --------------------------------------------------------------------------- small helpers
def _cpd(key) -> str:
    """'EX_cpd00027_e0', 'cpd00027_e0' or 'cpd00027' -> 'cpd00027'."""
    key = str(getattr(key, "id", key))
    if key.startswith("EX_"):
        key = key[3:]
    suffix = f"_{SHARED}"
    return key[:-len(suffix)] if key.endswith(suffix) else key


def _resolve_member(comm, member):
    if isinstance(member, bool):
        raise TypeError(f"not a member: {member!r}")
    if isinstance(member, int):
        for m in comm.members:
            if m.index == member:
                return m
        raise KeyError(f"no member with index {member}")
    key = getattr(member, "id", member)
    if key not in comm.members:
        raise KeyError(f"no member {key!r}; members {[m.id for m in comm.members]}")
    return comm.members.get_by_id(key)


def _exchanges(model) -> Dict[str, Tuple[Reaction, float]]:
    """{compound: (boundary reaction, its coefficient on <cpd>_e0)} for the model's e0 exchanges; an EX_
    reaction is preferred where a compound has several boundary reactions."""
    out = {}
    for r in model.reactions:
        if not r.boundary or len(r.metabolites) != 1:
            continue
        met, coef = next(iter(r.metabolites.items()))
        if met.compartment != SHARED and not met.id.endswith(f"_{SHARED}"):
            continue
        key = _cpd(met.id)
        if key in out and out[key][0].id.startswith("EX_"):
            continue
        out[key] = (r, float(coef))
    return out


def _cap_bounds(coef, cap, base, keep):
    """Exchange bounds allowing at most `cap` uptake (release = -coef * v >= -cap), the secretion side as in
    `base`. keep=True: never looser than base (an O2 share); keep=False: the cap replaces the base uptake bound
    (a tracked compound's uptake is set by its kinetics alone)."""
    lb, ub = base
    if coef < 0:
        new = -cap / -coef
        return (max(lb, new) if keep else new, ub)
    new = cap / coef
    return (lb, min(ub, new) if keep else new)


def _pair(value, where):
    if isinstance(value, Mapping):
        vmax, km = value["vmax"], value["Km"]
    else:
        vmax, km = value[0], value[1]
    vmax, km = float(vmax), float(km)
    if not (math.isfinite(vmax) and math.isfinite(km)) or vmax < 0 or km < 0:
        raise ValueError(f"{where}: vmax and Km must be finite numbers >= 0, not ({vmax}, {km})")
    return vmax, km


def _normalize_kinetics(kinetics, member_ids):
    out = {mid: {} for mid in member_ids}
    for mid, entries in (kinetics or {}).items():
        if mid not in out:
            raise KeyError(f"kinetics for {mid!r}, which is not a member ({list(member_ids)})")
        for key, value in entries.items():
            out[mid][_cpd(key)] = _pair(value, f"kinetics[{mid!r}][{key!r}]")
    return out


def _as_function(value, what):
    """None, a number / mapping (constant) or a callable of t."""
    if value is None or callable(value):
        return value
    if isinstance(value, Mapping):
        const = {_cpd(k): float(v) for k, v in value.items()}
        return lambda t: const
    const = float(value)
    if not math.isfinite(const) or const < 0:
        raise ValueError(f"{what} must be a finite number >= 0, not {value}")
    return lambda t: const


def _set_threads(model, threads):
    """Set Gurobi's thread count on the model's solver; returns the previous value (None if not Gurobi)."""
    if threads is None or "gurobi" not in model.solver.interface.__name__:
        return None
    try:
        previous = model.solver.problem.Params.Threads
        model.solver.problem.Params.Threads = int(threads)
        return previous
    except Exception:                                               # noqa: BLE001
        logger.debug("could not set Gurobi Threads", exc_info=True)
        return None


# --------------------------------------------------------------------------- member models
def member_model(comm, member, repair=None, subset=None, *, shared=True, drain_bounds=None) -> Model:
    """The member's own model extracted from the community: copies of its reactions (commmembers.
    member_reactions: its compartments' reactions, its transporters to e0, its biomass reaction, drain and
    sinks), the e0-only reactions no member owns that touch its e0 metabolites (shared=True; on a
    build_from_species_models community these are reactions the member models brought, e.g. an extracellular
    hydrolysis), and every boundary reaction of the community on an e0 metabolite it touches, with the
    community's CURRENT bounds (the medium). Reaction and metabolite ids stay the community's (c<i>, e0), so a
    repair written for the community applies unchanged.

    The member's biomass drain is opened to drain_bounds (default: the bounds it was built with,
    member.biomass_drain_bounds, or (0, 1000) if those are closed) and its biomass reaction is the objective.
    subset: an iterable of reaction ids, or a callable(reaction) -> bool, restricting which member reactions
    are copied (the biomass reaction and drain are always kept). repair: a callable(model) applied to the new
    model (its return value is kept in model.notes["member_model"]["repair"]), or a mapping
    {reaction id: (lb, ub)} replayed onto the reactions present (the ones absent are listed). The new model
    uses the community model's solver interface."""
    m = _resolve_member(comm, member)
    cmodel = comm.util.model
    rxns = member_reactions(comm, m)
    keep_always = {r.id for r in (m.primary_biomass, m.biomass_drain) if r is not None}
    if subset is not None:
        if callable(subset):
            rxns = [r for r in rxns if r.id in keep_always or subset(r)]
        else:
            wanted = {getattr(x, "id", x) for x in subset}
            unknown = wanted - {r.id for r in rxns}
            if unknown:
                raise KeyError(f"subset names reactions that are not {m.id}'s: {sorted(unknown)[:10]}")
            rxns = [r for r in rxns if r.id in keep_always or r.id in wanted]
    e0 = {met.id for r in rxns for met in r.metabolites if met.compartment == SHARED}
    extra = []
    if shared:
        for rid in member_table(comm).shared:
            r = cmodel.reactions.get_by_id(rid)
            if {met.id for met in r.metabolites} & e0:
                extra.append(r)
        e0 |= {met.id for r in extra for met in r.metabolites}
    boundary, missing = [], []
    for met_id in sorted(e0):
        met = cmodel.metabolites.get_by_id(met_id)
        found = [r for r in met.reactions if r.boundary and len(r.metabolites) == 1]
        if found:
            boundary += found
        else:
            missing.append(met_id)
    new = Model(f"{cmodel.id}_{m.id}", name=f"{m.id} from {cmodel.id}")
    new.solver = cmodel.problem
    new.add_reactions([r.copy() for r in rxns + extra + boundary])
    for met_id in missing:   # an e0 metabolite with no exchange in the community: closed for uptake
        new.add_boundary(new.metabolites.get_by_id(met_id), type="exchange", reaction_id=f"EX_{met_id}",
                         lb=0.0, ub=1000.0)
    drain = new.reactions.get_by_id(m.biomass_drain.id) if m.biomass_drain is not None else None
    if drain is not None:
        built = tuple(getattr(m, "biomass_drain_bounds", (0.0, 1000.0)))
        drain.bounds = tuple(drain_bounds) if drain_bounds is not None else (built if built[1] > 0 else (0.0, 1000.0))
    bio = new.reactions.get_by_id(m.primary_biomass.id)
    new.objective = bio
    record = None
    if repair is not None:
        if callable(repair):
            record = repair(new)
        elif isinstance(repair, Mapping):
            applied, absent = {}, []
            for rid, bounds in repair.items():
                if rid in new.reactions:
                    new.reactions.get_by_id(rid).bounds = tuple(float(b) for b in bounds)
                    applied[rid] = tuple(float(b) for b in bounds)
                else:
                    absent.append(rid)
            record = {"applied": applied, "absent": absent}
        else:
            raise TypeError("repair must be a callable(model) or a mapping {reaction id: (lb, ub)}")
    new.notes["member_model"] = {"community": cmodel.id, "member": m.id, "index": m.index, "biomass": bio.id,
                                 "drain": drain.id if drain is not None else None,
                                 "shared_reactions": [r.id for r in extra], "exchanges_created": missing,
                                 "repair": record}
    return new


def _biomass_of(model, biomass=None):
    if biomass is not None:
        return getattr(biomass, "id", biomass)
    note = model.notes.get("member_model") if isinstance(model.notes, Mapping) else None
    if note and note.get("biomass") in model.reactions:
        return note["biomass"]
    terms = [v for v in model.objective.expression.free_symbols]
    rxns = {r.id for r in model.reactions if r.forward_variable in terms or r.reverse_variable in terms}
    if len(rxns) != 1:
        raise ValueError(f"{model.id}: give the biomass reaction (biomass=); the objective has {sorted(rxns)}")
    return rxns.pop()


# --------------------------------------------------------------------------- lexicographic LP
class _LexLP:
    """max c.v; optionally one flux held at its min or max on that face; then min sum |v| (pFBA), on a cobra
    model through optlang. The rows are added to the model (inside the caller's context where there is one).
    The objective is switched on the solver directly, so cobra's context does not record every switch: the
    caller restores the model objective."""

    def __init__(self, model, objective_rxns, tiebreak=None, pfba=True, tag="dyn"):
        self.model = model
        self.pfba = pfba
        self.expr = add([r.flux_expression for r in objective_rxns])
        self.obj_max = model.problem.Objective(self.expr, direction="max")
        self.fix = model.problem.Constraint(self.expr, lb=-RELAXED, name=f"_{tag}_fix_growth")
        rows = [self.fix]
        self.sec = None
        if tiebreak is not None:
            rid, sense = tiebreak
            if sense not in ("min", "max"):
                raise ValueError(f"tie-break sense must be 'min' or 'max', not {sense!r}")
            rxn = model.reactions.get_by_id(rid)
            self.sec_sense = sense
            self.obj_sec = model.problem.Objective(rxn.flux_expression, direction=sense)
            if sense == "min":
                self.sec = model.problem.Constraint(rxn.flux_expression, ub=RELAXED, name=f"_{tag}_tiebreak")
            else:
                self.sec = model.problem.Constraint(rxn.flux_expression, lb=-RELAXED, name=f"_{tag}_tiebreak")
            rows.append(self.sec)
        model.add_cons_vars(rows)
        self.rows = rows
        self.obj_pfba = model.problem.Objective(
            add([v for r in model.reactions for v in (r.forward_variable, r.reverse_variable)]), direction="min")
        self.n_solves = 0
        self.seconds = 0.0

    def remove(self):
        """Take the rows out again (outside a model context)."""
        self.model.remove_cons_vars([row for row in self.rows if row.name in self.model.constraints])

    def _relax_sec(self):
        if self.sec is not None:
            if self.sec_sense == "min":
                self.sec.ub = RELAXED
            else:
                self.sec.lb = -RELAXED

    def _optimize(self, objective):
        solver = self.model.solver
        solver.objective = objective
        status = solver.optimize()
        self.n_solves += 1
        return status

    def solve(self):
        """(status, z, primal values) with status 'optimal', 'idle' (z <= IDLE), 'vertex' (the pFBA stage failed
        twice; the max-growth vertex is returned), or the solver status of a failed first stage (values None).
        An idle LP still returns values: with pFBA, the least total flux with no growth required (what the
        model's forced fluxes, e.g. an ATPM floor, need; the caller records those exchanges with mu = 0)."""
        t0 = time.perf_counter()
        try:
            self.fix.lb = -RELAXED
            self._relax_sec()
            status = self._optimize(self.obj_max)
            if status != "optimal":
                return status, None, None
            z = float(self.model.solver.objective.value)
            if z <= IDLE:
                if self.pfba and self._optimize(self.obj_pfba) != "optimal":
                    self._optimize(self.obj_max)
                return "idle", z, self.model.solver.primal_values
            if self.sec is not None:
                self.fix.lb = z * (1 - FACE)
                if self._optimize(self.obj_sec) == "optimal":
                    w = float(self.model.solver.objective.value)
                    tol = FACE * max(1.0, abs(w)) + 1e-9
                    if self.sec_sense == "min":
                        self.sec.ub = w + tol
                    else:
                        self.sec.lb = w - tol
            out = "optimal"
            if self.pfba:
                self.fix.lb = z * (1 - FACE)
                if self._optimize(self.obj_pfba) != "optimal":
                    self._relax_sec()
                    self.fix.lb = z * (1 - FACE_RETRY)
                    if self._optimize(self.obj_pfba) != "optimal":
                        self.fix.lb = -RELAXED
                        self._optimize(self.obj_max)
                        out = "vertex"
            elif self.sec is not None:   # the tie-break stage's solution is the answer
                pass
            values = self.model.solver.primal_values
            return out, z, values
        finally:
            self._relax_sec()
            self.seconds += time.perf_counter() - t0

    def maximum(self):
        """(status, z) of the max-growth stage alone (z None unless optimal)."""
        t0 = time.perf_counter()
        try:
            self.fix.lb = -RELAXED
            self._relax_sec()
            status = self._optimize(self.obj_max)
            return status, (float(self.model.solver.objective.value) if status == "optimal" else None)
        finally:
            self.seconds += time.perf_counter() - t0


def _flux(values, rxn):
    return values[rxn.forward_variable.name] - values[rxn.reverse_variable.name]


# --------------------------------------------------------------------------- steppers
@dataclass
class _Rates:
    mu: np.ndarray                 # 1/h per member
    release: np.ndarray            # rows: members (per gDW member) [+ shared per gDW community]; cols: ledger cpds
    status: list                   # per member
    o2_limited: bool = False
    notes: list = field(default_factory=list)   # (kind, member, value): what the stepper had to do this step


class _MembersStepper:
    form = "members"

    def __init__(self, models, biomass, tracked, ledger, o2_mode, o2_rule, tiebreak, pfba, threads):
        self.ids = list(models)
        self.models, self.lp, self.bio = models, {}, {}
        self.ledger = ledger
        self.col = {c: j for j, c in enumerate(ledger)}
        self.o2_mode, self.o2_rule = o2_mode, o2_rule
        self.ex, self.base, self.current, self.objectives = {}, {}, {}, {}
        self.threads_saved = {}
        for mid, model in models.items():
            previous = _set_threads(model, threads)
            if previous is not None:
                self.threads_saved[mid] = previous
            self.objectives[mid] = (model.objective.expression, model.objective.direction)
            self.bio[mid] = model.reactions.get_by_id(biomass[mid])
            self.lp[mid] = _LexLP(model, [self.bio[mid]], (tiebreak or {}).get(mid), pfba, tag="dyn")
            ex = _exchanges(model)
            self.ex[mid] = {c: ex[c] for c in ledger if c in ex}
            self.base[mid] = {c: tuple(r.bounds) for c, (r, _) in self.ex[mid].items()}
            self.current[mid] = dict(self.base[mid])
        self.tracked = [c for c in tracked]

    def _bounds(self, mid, caps, o2cap):
        """Push the exchange bounds of one member: tracked compounds by their caps, O2 by o2cap (None: base)."""
        for c, (rxn, coef) in self.ex[mid].items():
            if c == O2 and self.o2_mode != "kla":
                cap = o2cap
                bounds = self.base[mid][c] if cap is None else _cap_bounds(coef, cap, self.base[mid][c], keep=True)
            elif c in caps:
                bounds = _cap_bounds(coef, caps[c], self.base[mid][c], keep=False)
            else:
                bounds = self.base[mid][c]
            if bounds != self.current[mid][c]:
                rxn.bounds = bounds
                self.current[mid][c] = bounds

    def _solve_member(self, mid, caps, o2cap):
        self._bounds(mid, caps, o2cap)
        status, z, values = self.lp[mid].solve()
        rel = np.zeros(len(self.ledger))
        if values is None:
            return 0.0, rel, status
        for c, (rxn, coef) in self.ex[mid].items():
            rel[self.col[c]] = -coef * _flux(values, rxn)
        mu = 0.0 if status == "idle" else max(_flux(values, self.bio[mid]), 0.0)
        return mu, rel, status

    def rates(self, X, caps, otr):
        n = len(self.ids)
        mu, rel, status = np.zeros(n), np.zeros((n, len(self.ledger))), [None] * n
        o2cap = [None] * n
        limited = False
        if otr is not None and self.o2_mode == "otr" and self.o2_rule == "equal":
            share = otr / max(float(np.sum(X)), 1e-300)
            o2cap = [share] * n
        for i, mid in enumerate(self.ids):
            mu[i], rel[i], status[i] = self._solve_member(mid, caps[i], o2cap[i])
        if otr is not None and self.o2_mode == "otr" and O2 in self.col:
            o2 = np.maximum(-rel[:, self.col[O2]], 0.0)
            if self.o2_rule == "equal":
                limited = bool(np.any(o2 >= np.asarray(o2cap) * (1 - 1e-6)))
            else:
                demand = float(np.dot(o2, X))
                if demand > otr * (1 + 1e-9):
                    limited = True
                    for i, mid in enumerate(self.ids):
                        mu[i], rel[i], status[i] = self._solve_member(mid, caps[i], o2[i] * otr / demand)
        return _Rates(mu, rel, status, limited)

    @property
    def n_solves(self):
        return sum(lp.n_solves for lp in self.lp.values())

    @property
    def lp_seconds(self):
        return sum(lp.seconds for lp in self.lp.values())

    def close(self):
        """Put the models back as they were: exchange bounds, objective, no added rows, solver threads."""
        for mid, model in self.models.items():
            for c, (rxn, _) in self.ex[mid].items():
                rxn.bounds = self.base[mid][c]
            if mid in self.lp:
                self.lp[mid].remove()
            expr, direction = self.objectives[mid]
            model.objective = model.problem.Objective(expr, direction=direction)
            if mid in self.threads_saved:
                _set_threads(model, self.threads_saved[mid])


def _exchange_like(rxn):
    return rxn.boundary and all(met.compartment == SHARED or met.id.endswith(f"_{SHARED}") for met in rxn.metabolites)


class _SoloMembers:
    """The members' own models (member_model) beside a community_strict run, solved the way the "members" form solves
    them: the rate of a member too rare for the community LP (a_m < a_min), the ceiling a community rate is clamped
    to, and whether a member alone can pay its forced fluxes."""

    def __init__(self, comm, tracked, ledger, o2_mode, pfba, threads):
        models = {m.id: member_model(comm, m) for m in comm.members}
        biomass = {m.id: m.primary_biomass.id for m in comm.members}
        self.stepper = _MembersStepper(models, biomass, tracked, ledger, o2_mode, None, None, pfba, threads)
        self.forced = {mid: [(r, r.lower_bound, r.upper_bound) for r in model.reactions
                             if not _exchange_like(r) and (r.lower_bound > 0 or r.upper_bound < 0)]
                       for mid, model in models.items()}

    def solve(self, mid, caps, o2cap):
        """(mu, release per gDW member, status) of the members-form solve with these caps (O2 per gDW: o2cap)."""
        return self.stepper._solve_member(mid, caps, o2cap)

    def _maximum(self, mid, caps, release):
        self.stepper._bounds(mid, caps, None)
        forced = self.forced[mid] if release else []
        for r, lb, ub in forced:
            r.bounds = (min(lb, 0.0), max(ub, 0.0))
        try:
            return self.stepper.lp[mid].maximum()
        finally:
            for r, lb, ub in forced:
                r.bounds = (lb, ub)

    def ceiling(self, mid, caps):
        """max mu of the member alone on the current pool: tracked compounds at its kinetic caps, O2 not limited by a
        transfer rate, forced bounds released. None when that LP is not optimal."""
        status, z = self._maximum(mid, caps, release=True)
        return max(z, 0.0) if status == "optimal" else None

    def feasible(self, mid, caps):
        """Whether the member alone can meet its forced fluxes on the current pool (O2 not limited)."""
        return self._maximum(mid, caps, release=False)[0] == "optimal"

    @property
    def n_solves(self):
        return self.stepper.n_solves

    @property
    def lp_seconds(self):
        return self.stepper.lp_seconds

    def close(self):
        self.stepper.close()


class _CommunityStrictStepper:
    """One community LP per step on comm.util.model. Structural changes (bio1 shut, drains open, X_m variables,
    cap rows, the growth-fix rows, lifted *_commKin rows) are made inside a model context that the caller's
    ExitStack closes; per-step bounds (eq. 9 scaling, exchanges, X_m = a_m) are written to the solver directly
    and cobra's bounds are pushed back at the end. The cap rows' q is rewritten in place every step (the X_m
    coefficient), so their commmembers registry entries carry the placeholder rate 1.0 while the run lasts.

    Beside the community LP each member's own model (member_model, _SoloMembers) solves what the community LP cannot
    be trusted with: a member below a_min of the culture (its per-community fluxes sit at the solver's tolerance, so
    v_bio / a_m is noise), the ceiling every community rate is clamped to (the member alone on the current pool), and
    which member cannot pay its forced fluxes when the community LP is infeasible (those are released for the step)."""
    form = "community_strict"

    def __init__(self, comm, tracked, ledger, o2_mode, o2_rule, tiebreak, pfba, cap_basis, kinetic_rows,
                 scale_bounds, threads, stack: ExitStack, a_min=A_MIN):
        self.comm = comm
        model = self.model = comm.util.model
        self.ids = [m.id for m in comm.members]
        self.ledger = ledger
        self.col = {c: j for j, c in enumerate(ledger)}
        self.o2_mode, self.o2_rule = o2_mode, o2_rule
        self.tracked = list(tracked)
        self.a_min = float(a_min)
        self.scale_bounds = bool(scale_bounds)
        # the members' own models, cut from the community as it is (before the run's structural changes)
        self.solo = _SoloMembers(comm, tracked, ledger, o2_mode, pfba, threads)
        stack.callback(self.solo.close)
        previous_threads = _set_threads(model, threads)
        if previous_threads is not None:
            stack.callback(_set_threads, model, previous_threads)
        # everything structural happens inside a model context, undone on exit
        saved_objective = (model.objective.expression, model.objective.direction)
        stack.callback(self._restore_objective, saved_objective)
        stack.enter_context(model)
        if kinetic_rows == "lift":
            stack.enter_context(lifted_kinetic_rows(comm))
        elif kinetic_rows != "keep":
            raise ValueError(f"kinetic_rows must be 'lift' or 'keep', not {kinetic_rows!r}")
        members = list(comm.members)
        self.bio = {m.id: model.reactions.get_by_id(m.primary_biomass.id) for m in members}
        model.reactions.get_by_id(comm.primary_biomass.id).bounds = (0.0, 0.0)
        for m in members:
            if m.biomass_drain is not None:
                built = tuple(getattr(m, "biomass_drain_bounds", (0.0, 1000.0)))
                model.reactions.get_by_id(m.biomass_drain.id).bounds = built if built[1] > 0 else (0.0, 1000.0)
        # X_m: the caps' basis, fixed at a_m every step
        existing = {m.id: f"X_{m.id}" in model.variables for m in members}
        self.X = member_biomass_variables(comm)
        self._x_saved = {mid: (v.lb, v.ub) for mid, v in self.X.items() if existing[mid]}
        stack.callback(self._restore_x)
        # membership, once
        by_member = reactions_by_member(comm)
        self.transporters = {}
        self.own_nb = {}       # non-boundary member reactions and their bounds: scaled by a_m (eq. 9), shut below a_min
        self.forced = {}       # forced member bounds (lb > 0 or ub < 0): (reaction, lb, ub, scaled with a_m)
        for m in members:
            own = by_member[m.id]
            self.transporters[m.id] = [r for r in own if any(met.compartment == SHARED for met in r.metabolites)]
            rs = [r for r in own if not r.boundary]
            self.own_nb[m.id] = (rs, np.array([r.lower_bound for r in rs]), np.array([r.upper_bound for r in rs]))
            drain = m.biomass_drain.id if m.biomass_drain is not None else None
            self.forced[m.id] = [(r, r.lower_bound, r.upper_bound, not r.boundary) for r in own
                                 if r.id != drain and (r.lower_bound > 0 or r.upper_bound < 0)]
        stack.callback(self._resync_bounds)
        self.shared = [model.reactions.get_by_id(rid) for rid in member_table(comm).shared]
        # strict caps: one row per member and tracked compound the member can move
        capped = list(self.tracked) + ([O2] if o2_mode == "otr" and o2_rule == "equal" else [])
        self.caps = {}
        for m in members:
            by_cpd = {}
            for r in self.transporters[m.id]:
                for met in r.metabolites:
                    if met.compartment == SHARED:
                        by_cpd.setdefault(_cpd(met.id), []).append(r)
            for c in capped:
                rxns = by_cpd.get(c)
                if not rxns:
                    continue
                # written with q = 1 so that X_m is in the row; q is rewritten in place every step
                name = f"dyncap_{m.id}_{c}"
                if cap_basis == "gross":
                    row = cap_member_uptake(comm, m, c, q=1.0, reactions=rxns, basis="X", X=self.X, name=name)
                else:
                    expr = member_uptake_expression(comm, m, c, reactions=rxns, gross=False)
                    row = model.problem.Constraint(expr - 1.0 * self.X[m.id], ub=0.0, name=name)
                    model.add_cons_vars([row])
                    row = model.constraints[name]
                self.caps[(m.id, c)] = [row, 1.0]
        # community exchanges of tracked compounds and O2
        ex = _exchanges(model)
        self.ex = {c: ex[c] for c in self.tracked if c in ex}
        self.o2_ex = ex.get(O2)
        self.ex_base = {c: tuple(r.bounds) for c, (r, _) in self.ex.items()}
        self.o2_base = tuple(self.o2_ex[0].bounds) if self.o2_ex is not None else None
        self.index = {mid: i for i, mid in enumerate(self.ids)}
        self.lp =_LexLP(model, list(self.bio.values()), (tiebreak or {}).get("community"), pfba, tag="dyncom")

    # ---- restoration
    def _restore_objective(self, saved):
        expr, direction = saved
        self.model.objective = self.model.problem.Objective(expr, direction=direction)

    def _restore_x(self):
        for mid, (lb, ub) in self._x_saved.items():
            if f"X_{mid}" in self.model.variables:
                self.model.variables[f"X_{mid}"].set_bounds(lb, ub)

    def _resync_bounds(self):
        """Per-step bounds were written to the solver directly; push cobra's (restored) bounds back."""
        for rs, _, _ in self.own_nb.values():
            for r in rs:
                r.update_variable_bounds()
        for forced in self.forced.values():
            for r, _, _, _ in forced:
                r.update_variable_bounds()
        for r, _ in self.ex.values():
            r.update_variable_bounds()
        if self.o2_ex is not None:
            self.o2_ex[0].update_variable_bounds()

    @staticmethod
    def _set(rxn, lb, ub):
        rxn.forward_variable.set_bounds(max(lb, 0.0), max(ub, 0.0))
        rxn.reverse_variable.set_bounds(max(-ub, 0.0), max(-lb, 0.0))

    # ---- one step
    def rates(self, X, caps, otr):
        n = len(self.ids)
        Xt = float(np.sum(X))
        a = X / Xt if Xt > 0 else np.zeros(n)
        rare = [i for i in range(n) if 0.0 < a[i] < self.a_min]
        a_lp = a.copy()            # the community LP's shares: members below a_min are taken out of it
        a_lp[rare] = 0.0
        touched = []               # (reaction, lb, ub) written for this step only; put back after it
        for i, mid in enumerate(self.ids):
            self.X[mid].set_bounds(float(a_lp[i]), float(a_lp[i]))
            rs, lb0, ub0 = self.own_nb[mid]
            if self.scale_bounds:
                for r, lo, hi in zip(rs, lb0 * a_lp[i], ub0 * a_lp[i]):
                    self._set(r, lo, hi)
            elif i in rare:
                for r, lo, hi in zip(rs, lb0, ub0):
                    self._set(r, 0.0, 0.0)
                    touched.append((r, lo, hi))
        cidx = {c: j for j, c in enumerate(self.tracked)}
        for (mid, c), entry in self.caps.items():
            i = self.index[mid]
            if c == O2 and c not in cidx:
                q = otr / Xt if (otr is not None and Xt > 0) else 0.0
            else:
                q = float(caps[i].get(c, 0.0))
            if q != entry[1]:
                entry[0].set_linear_coefficients({self.X[mid]: -q})
                entry[1] = q
        for c, (r, coef) in self.ex.items():
            total = float(sum(a_lp[i] * caps[i].get(c, 0.0) for i in range(n)))
            lb, ub = _cap_bounds(coef, total, self.ex_base[c], keep=False)
            self._set(r, lb, ub)
        if self.o2_ex is not None and self.o2_mode == "otr":
            r, coef = self.o2_ex
            if otr is None:
                self._set(r, *self.o2_base)
            else:
                self._set(r, *_cap_bounds(coef, otr / Xt if Xt > 0 else 0.0, self.o2_base, keep=True))
        status, z, values = self.lp.solve()
        notes = []
        if values is None:
            status, z, values, released = self._release_unpayable(status, caps, a_lp, touched)
            notes += [("released", self.ids[i], tuple(names)) for i, names in released.items()]
        mu = np.zeros(n)
        rel = np.zeros((n + 1, len(self.ledger)))
        stat = [status] * n
        if values is not None:
            for i, mid in enumerate(self.ids):
                if a_lp[i] <= 0:
                    continue
                mu[i] = 0.0 if status == "idle" else max(_flux(values, self.bio[mid]), 0.0) / a_lp[i]
                for r in self.transporters[mid]:
                    v = _flux(values, r)
                    if v == 0.0:
                        continue
                    for met, coef in r.metabolites.items():
                        if met.compartment == SHARED:
                            j = self.col.get(_cpd(met.id))
                            if j is not None:
                                rel[i, j] += float(coef) * v / a_lp[i]
            for r in self.shared:
                v = _flux(values, r)
                if v == 0.0:
                    continue
                for met, coef in r.metabolites.items():
                    j = self.col.get(_cpd(met.id))
                    if j is not None:
                        rel[n, j] += float(coef) * v
        for r, lo, hi in touched:
            self._set(r, lo, hi)
        # members below a_min: their own LP, with their own caps on the pool and the per-gDW O2 share
        o2share = otr / Xt if (otr is not None and self.o2_mode == "otr" and Xt > 0) else None
        for i in rare:
            mid = self.ids[i]
            mu[i], rel[i], stat[i] = self.solo.solve(mid, caps[i], o2share)
            notes.append(("a_min", mid, float(a[i])))
        # no member grows faster than it would alone on the current pool
        for i, mid in enumerate(self.ids):
            if i in rare or mu[i] <= 0.0:
                continue
            ceiling = self.solo.ceiling(mid, caps[i])
            if ceiling is not None and mu[i] > ceiling + CLAMP_ATOL:
                beyond = mu[i] > ceiling * (1.0 + CLAMP_WARN_RTOL) + CLAMP_WARN_ATOL
                notes.append(("clamped" if beyond else "clamped_within_tolerance", mid, float(mu[i] - ceiling)))
                rel[i] *= ceiling / mu[i]
                mu[i] = ceiling
        limited = False
        if otr is not None and self.o2_mode == "otr" and O2 in self.col and Xt > 0:
            used = float(-np.dot(np.minimum(rel[:n, self.col[O2]], 0.0), a))
            limited = used >= (otr / Xt) * (1 - 1e-6)
        return _Rates(mu, rel, stat, limited, notes)

    def _release(self, i, share, touched):
        """Release member i's forced bounds on the solver for this step; returns the reaction ids."""
        names = []
        for r, lb0, ub0, scaled in self.forced[self.ids[i]]:
            f = share if (scaled and self.scale_bounds) else 1.0
            lo, hi = lb0 * f, ub0 * f
            self._set(r, min(lo, 0.0), max(hi, 0.0))
            touched.append((r, lo, hi))
            names.append(r.id)
        return names

    def _unrelease(self, i, share):
        for r, lb0, ub0, scaled in self.forced[self.ids[i]]:
            f = share if (scaled and self.scale_bounds) else 1.0
            self._set(r, lb0 * f, ub0 * f)

    def _release_unpayable(self, status, caps, a_lp, touched):
        """The community LP failed: release the forced bounds of the member(s) that cannot meet them and re-solve.
        First the members that are infeasible alone on the current pool; if that is not enough, each other member with
        forced bounds in turn, then all of them. Returns (status, z, values, {member index: released reaction ids});
        the incoming status and no values when nothing helps."""
        cands = [i for i in range(len(self.ids)) if a_lp[i] > 0 and self.forced[self.ids[i]]]
        released = {}

        def resolve(idx):
            for i in idx:
                released[i] = self._release(i, a_lp[i], touched)
            return self.lp.solve()

        alone = [i for i in cands if not self.solo.feasible(self.ids[i], caps[i])]
        if alone:
            st, z, values = resolve(alone)
            if values is not None:
                return st, z, values, released
        rest = [i for i in cands if i not in released]
        if len(rest) > 1:
            for i in rest:
                st, z, values = resolve([i])
                if values is not None:
                    return st, z, values, released
                self._unrelease(i, a_lp[i])
                released.pop(i)
        if rest:
            st, z, values = resolve(rest)
            if values is not None:
                return st, z, values, released
        return status, None, None, {}

    def summarize(self, events):
        """The run's warnings for what the stepper had to do: {(kind, member): [(t, value)]}. Clamps within the
        solver's tolerance (kind "clamped_within_tolerance") are only counted (Trajectory.stats["notes"])."""
        out = []
        for (kind, mid), entries in events.items():
            k, first = len(entries), entries[0][0]
            if kind == "a_min":
                out.append(f"{mid}: below a_min = {self.a_min:g} of the culture at {k} step(s) (first at t = {first:.4g} h, "
                           f"a_m = {entries[0][1]:.3g}); solved there by its own member LP (the members form, its own "
                           "kinetic caps on the pool), since the community LP's fluxes for it sit at the solver's tolerance")
            elif kind == "clamped":
                worst = max(v for _, v in entries)
                out.append(f"{mid}: the community LP gave a rate above the member's solo maximum on the current pool at "
                           f"{k} step(s) (first at t = {first:.4g} h, largest excess {worst:.3g} 1/h); clamped to it, "
                           "with its exchanges scaled to match")
            elif kind == "released":
                rxns = sorted({r for _, names in entries for r in names})
                out.append(f"{mid}: the community LP was infeasible with {mid}'s forced fluxes ({', '.join(rxns[:6])}"
                           f"{', ...' if len(rxns) > 6 else ''}) at {k} step(s) (first at t = {first:.4g} h); they were "
                           "released at those steps so that the community could be solved")
        return out

    @property
    def n_solves(self):
        return self.lp.n_solves + self.solo.n_solves

    @property
    def lp_seconds(self):
        return self.lp.seconds + self.solo.lp_seconds

    def close(self):
        pass


# --------------------------------------------------------------------------- trajectory
@dataclass
class Trajectory:
    """A dynamic run (dynamic / simulate_models).

    times            recorded times, h (every record_every-th accepted step, and the last)
    X                {member: gDW/L at each recorded time}
    concentrations   {compound: mM at each recorded time} (tracked pools, O2 too when it is a pool)
    mu               {member: specific growth rate (1/h) of the LP solved at each recorded time, i.e. the rate
                     used over the following step}
    uptake, secretion  arrays [interval, ledger row, compound], mmol/L REALISED over the interval from times[k]
                     to times[k+1] (Patankar-weighted where that update was used): the per-member exchange
                     ledger. Rows are ledger_rows (the members, plus "shared" for e0-only reactions in
                     "community_strict", per L of culture); columns ledger_compounds (tracked compounds, and O2)
    inflow, outflow  arrays [interval, compound], mmol/L brought in by the feed / O2 transfer and washed out
    o2_limited       whether the O2 transfer limit bound at each recorded time
    status           [{member: LP status}] at each recorded time
    form, tiebreak, o2_rule, otr, cap_basis ...  the settings that change results (see settings)
    warnings         what the run had to do that the caller should know"""
    times: np.ndarray
    members: Tuple[str, ...]
    X: Dict[str, np.ndarray]
    concentrations: Dict[str, np.ndarray]
    mu: Dict[str, np.ndarray]
    ledger_rows: Tuple[str, ...]
    ledger_compounds: Tuple[str, ...]
    uptake: np.ndarray
    secretion: np.ndarray
    inflow: np.ndarray
    outflow: np.ndarray
    o2_limited: np.ndarray
    status: list
    form: str
    tiebreak: object
    o2_rule: Optional[str]
    otr: object
    settings: dict
    stats: dict
    warnings: list = field(default_factory=list)

    @property
    def composition(self) -> Dict[str, np.ndarray]:
        """{member: X_m / sum X} at each recorded time."""
        tot = np.sum([self.X[m] for m in self.members], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return {m: np.where(tot > 0, self.X[m] / tot, np.nan) for m in self.members}

    def at(self, t) -> dict:
        """Linear interpolation of X, composition and concentrations at time(s) t."""
        t = np.asarray(t, dtype=float)
        out = {"X": {m: np.interp(t, self.times, self.X[m]) for m in self.members},
               "S": {c: np.interp(t, self.times, v) for c, v in self.concentrations.items()}}
        tot = np.sum(list(out["X"].values()), axis=0)
        out["composition"] = {m: out["X"][m] / tot for m in self.members}
        return out

    def growth_rate(self, member, window, times=None) -> Optional[float]:
        """Slope of ln X_m over the window (h): least squares over the recorded points in it, or over X_m
        interpolated at `times` (e.g. measured read times) inside it. None with fewer than 3 points."""
        lo, hi = window
        if times is None:
            t, y = self.times, self.X[member]
        else:
            t = np.asarray(times, dtype=float)
            y = np.interp(t, self.times, self.X[member])
        msk = (t >= lo) & (t <= hi) & (y > 0)
        if msk.sum() < 3:
            return None
        return float(np.polyfit(t[msk], np.log(y[msk]), 1)[0])

    def ledger(self, window=None) -> dict:
        """{ledger row: {compound: {"uptake": mmol/L, "secretion": mmol/L}}} summed over the intervals that
        start inside window (h; default the whole run). Entries that are 0 both ways are left out."""
        if len(self.times) < 2:
            return {}
        start = self.times[:-1]
        msk = np.ones(len(start), bool) if window is None else (start >= window[0]) & (start < window[1])
        up, se = self.uptake[msk].sum(axis=0), self.secretion[msk].sum(axis=0)
        out = {}
        for i, row in enumerate(self.ledger_rows):
            entries = {c: {"uptake": float(up[i, j]), "secretion": float(se[i, j])}
                       for j, c in enumerate(self.ledger_compounds) if up[i, j] != 0 or se[i, j] != 0}
            if entries:
                out[row] = entries
        return out

    def to_frame(self):
        """pandas DataFrame indexed by time: X_<m>, a_<m>, mu_<m>, S_<cpd>."""
        from pandas import DataFrame
        comp = self.composition
        data = {}
        for m in self.members:
            data[f"X_{m}"], data[f"a_{m}"], data[f"mu_{m}"] = self.X[m], comp[m], self.mu[m]
        for c, v in self.concentrations.items():
            data[f"S_{c}"] = v
        return DataFrame(data, index=self.times).rename_axis("time_h")

    def as_dict(self) -> dict:
        """Plain lists and dicts (json.dumps-able); the ledger as totals over the run."""
        def plain(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, Mapping):
                return {str(k): plain(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [plain(v) for v in o]
            if isinstance(o, (np.floating, np.integer, np.bool_)):
                return o.item()
            return o
        return plain({"times": self.times, "members": self.members, "X": self.X,
                      "concentrations": self.concentrations, "mu": self.mu, "composition": self.composition,
                      "o2_limited": self.o2_limited, "form": self.form, "tiebreak": self.tiebreak,
                      "o2_rule": self.o2_rule, "otr": self.otr, "settings": self.settings, "stats": self.stats,
                      "ledger_totals": self.ledger(), "warnings": self.warnings})


# --------------------------------------------------------------------------- integrator
def _phi(x):
    """(exp(x) - 1) / x, 1 at x = 0."""
    return np.where(np.abs(x) < 1e-10, 1.0 + x / 2.0, np.expm1(x) / np.where(x == 0, 1.0, x))


def _integrate(stepper, members, X0, S0, tracked, ledger, kin, default, t_end, dt, otr, o2_mode, feed, D,
               record_every, max_depletion, rtol, atol, dt_min, s_min, max_steps, warnings, callback=None,
               t_start=0.0):
    """The time loop (see the module docstring, Integration). Returns the records, the ledger intervals and
    the step statistics."""
    nm, nc = len(members), len(tracked)
    col = {c: j for j, c in enumerate(ledger)}
    tcol = np.array([col[c] for c in tracked], dtype=int)
    X = np.array([float(X0[m]) for m in members])
    S = np.array([float(S0.get(c, 0.0)) for c in tracked])
    vmax = np.array([[kin[m].get(c, default)[0] for c in tracked] for m in members], dtype=float).reshape(nm, nc)
    Km = np.array([[kin[m].get(c, default)[1] for c in tracked] for m in members], dtype=float).reshape(nm, nc)
    kla = (float(otr[0]), float(otr[1])) if o2_mode == "kla" else None
    j_o2 = tracked.index(O2) if (kla is not None and O2 in tracked) else None
    t, h = float(t_start), float(dt)
    rec = {"t": [], "X": [], "S": [], "mu": [], "o2": [], "status": []}
    intervals = {"up": [], "se": [], "in": [], "out": []}
    acc = None
    n_acc = n_rej = n_forced = n_patankar = 0
    failed, vertex, events = {}, {}, {}
    mismatch_total = np.zeros(nc)
    h_min_seen = float(dt)
    since_record = 0

    while True:
        live = S > s_min
        caps_arr = np.where(live[None, :], vmax * S / (Km + np.where(live, S, 1.0)), 0.0)
        caps = [dict(zip(tracked, caps_arr[i])) for i in range(nm)]
        rates = stepper.rates(X, caps, otr if o2_mode == "otr" else None)
        for kind, mid, value in rates.notes:
            events.setdefault((kind, mid), []).append((t, value))
        for i, st in enumerate(rates.status):
            if st == "vertex":
                vertex.setdefault(members[i], []).append(t)
            elif st not in ("optimal", "idle"):
                failed.setdefault(members[i], []).append(t)
        final = t >= t_end or n_acc >= max_steps
        if not rec["t"] or since_record >= record_every or (final and since_record > 0):
            if acc is not None:
                for k in intervals:
                    intervals[k].append(acc[k])
            rec["t"].append(t)
            rec["X"].append(X.copy())
            rec["S"].append(S.copy())
            rec["mu"].append(rates.mu.copy())
            rec["o2"].append(bool(rates.o2_limited))
            rec["status"].append(dict(zip(members, rates.status)))
            if callback is not None:
                callback(t, dict(zip(members, X)), dict(zip(tracked, S)))
            nrow = rates.release.shape[0]
            acc = {"up": np.zeros((nrow, len(ledger))), "se": np.zeros((nrow, len(ledger))),
                   "in": np.zeros(len(ledger)), "out": np.zeros(len(ledger))}
            since_record = 0
        if final:
            if t < t_end:
                warnings.append(f"stopped at t = {t:.6g} h after max_steps = {max_steps} accepted steps")
            break
        Dn = float(D(t)) if D is not None else 0.0
        fd = feed(t) if feed is not None else {}
        f_vec = np.array([float(fd.get(c, 0.0)) for c in tracked])
        rel = rates.release
        rel_up, rel_se = np.maximum(-rel, 0.0), np.maximum(rel, 0.0)
        g = rates.mu - Dn
        pos = S > 0
        S_safe = np.where(pos, S, 1.0)
        while True:
            h = min(h, t_end - t)
            G = X * h * _phi(g * h)                       # biomass integrals, gDW h / L
            Grow = np.append(G, G.sum()) if rel.shape[0] == nm + 1 else G
            P = rel_se[:, tcol].T @ Grow + Dn * h * f_vec  # production over the step, mM
            U = rel_up[:, tcol].T @ Grow                   # the members' uptake over the step, mM
            L = np.full(nc, Dn * h)                        # linear loss coefficient
            if j_o2 is not None:
                P[j_o2] += kla[0] * h * kla[1]
                L[j_o2] += kla[0] * h
            S_exp = S + P - U - L * S
            S_pat = np.where(pos, (S + P) / (1.0 + L + U / S_safe), np.maximum(P - U, 0.0) / (1.0 + L))
            rho = np.where(pos, S_pat / S_safe, 1.0)
            mismatch = U * np.abs(1.0 - rho)
            explicit_ok = pos & (S_exp >= (1.0 - max_depletion) * S)
            patankar_ok = pos & (mismatch <= atol + rtol * U)
            empty_ok = ~pos & (P - U >= -atol)
            ok = explicit_ok | patankar_ok | empty_ok
            if ok.all() or h <= dt_min:
                n_forced += int(not ok.all())
                break
            n_rej += 1
            bad = ~ok & pos
            drain = (U - P + L * S)[bad]
            with np.errstate(divide="ignore", invalid="ignore"):
                est = (np.where(drain > 0, 0.9 * h * max_depletion * S[bad] / drain, 0.5 * h).min()
                       if bad.any() else 0.5 * h)
            h = max(dt_min, min(0.5 * h, max(0.1 * h, float(est))))
        use_pat = ~explicit_ok                             # Patankar wherever the explicit update is not safe
        S_new = np.where(use_pat, S_pat, S_exp)
        realised = np.where(use_pat, rho, 1.0)
        n_patankar += int(np.any(use_pat & (U > 0)))
        mismatch_total += np.where(use_pat, U * (1.0 - rho), 0.0)
        weight = np.ones(len(ledger))
        weight[tcol] = realised
        acc["up"] += rel_up * Grow[:, None] * weight[None, :]
        acc["se"] += rel_se * Grow[:, None]
        S_loss = np.where(use_pat, S_new, S)               # the pool the linear losses act on
        acc["in"][tcol] += Dn * h * f_vec
        acc["out"][tcol] += Dn * h * S_loss
        if j_o2 is not None:
            acc["in"][col[O2]] += kla[0] * h * (kla[1] - S_loss[j_o2])
        X = X * np.exp(g * h)
        S = S_new
        t = t_end if t_end - (t + h) <= 1e-9 * max(1.0, t_end) else t + h
        n_acc += 1
        since_record += 1
        h_min_seen = min(h_min_seen, h)
        h = min(float(dt), 2.0 * h)

    for mid, ts in failed.items():
        warnings.append(f"{mid}: the growth LP failed at {len(ts)} step(s) (first at t = {ts[0]:.4g} h); the "
                        "member was held idle there")
    for mid, ts in vertex.items():
        warnings.append(f"{mid}: the pFBA stage failed at {len(ts)} step(s) (first at t = {ts[0]:.4g} h); the "
                        "max-growth vertex was used there")
    summarize = getattr(stepper, "summarize", None)
    if events and summarize is not None:
        warnings.extend(summarize(events))
    notes = {}
    for (kind, mid), entries in events.items():
        notes.setdefault(kind, {})[mid] = len(entries)
    if n_forced:
        warnings.append(f"{n_forced} step(s) reached dt_min = {dt_min} h and were taken with the Patankar update "
                        "although the members' uptake was not realised to within rtol")
    stats = {"accepted_steps": n_acc, "rejected_steps": n_rej, "forced_steps": n_forced,
             "patankar_steps": n_patankar, "smallest_step_h": h_min_seen,
             "patankar_mismatch_mM": {c: float(v) for c, v in zip(tracked, mismatch_total) if v != 0.0},
             "notes": notes}
    return rec, (intervals["up"], intervals["se"], intervals["in"], intervals["out"]), stats


def _ample_default(models, S0, feed_keys, o2_handled):
    ample = set()
    for model in models:
        for c, (r, coef) in _exchanges(model).items():
            uptake_open = (coef < 0 and r.lower_bound < 0) or (coef > 0 and r.upper_bound > 0)
            if uptake_open:
                ample.add(c)
    ample -= set(S0) | set(feed_keys)
    if o2_handled:
        ample.discard(O2)
    return ample


def _setup(models, S0, feed, kinetics, default, otr, o2_rule, form, ample, tracked, member_ids):
    if otr is None:
        o2_mode = None
    elif isinstance(otr, (tuple, list)):
        if len(otr) != 2:
            raise ValueError("otr as a pair is (kLa 1/h, C* mM)")
        o2_mode = "kla"
    else:
        if not math.isfinite(float(otr)) or float(otr) < 0:
            raise ValueError(f"otr must be a number >= 0 (mmol O2/L/h) or a (kLa, C*) pair, not {otr!r}")
        o2_mode = "otr"
    rules = O2_RULES[form]
    if o2_mode == "otr":
        o2_rule = rules[0] if o2_rule is None else o2_rule
        if o2_rule not in rules:
            raise ValueError(f"o2_rule {o2_rule!r} is not defined for form {form!r}; use one of {rules}")
    elif o2_rule is not None and o2_mode is None:
        logger.info("o2_rule is ignored without an O2 transfer rate (otr=None)")
        o2_rule = None
    elif o2_mode == "kla":
        o2_rule = "kinetics"
    S0 = {_cpd(k): float(v) for k, v in S0.items()}
    for c, v in S0.items():
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"S0[{c!r}] must be a finite concentration >= 0, not {v}")
    feed_keys = set()
    if feed is not None and not callable(feed):
        feed_keys = {_cpd(k) for k in feed}
    exchanged = set()
    for model in models:
        exchanged |= set(_exchanges(model))
    if tracked is not None:
        tracked_set = {_cpd(c) for c in tracked}
        ample_set = exchanged - tracked_set
    else:
        ample_set = {_cpd(c) for c in ample} if ample is not None else _ample_default(models, S0, feed_keys,
                                                                                    o2_mode is not None)
        tracked_set = exchanged - ample_set
    if o2_mode == "otr":   # a transfer-rate limit: O2 is neither a pool nor left at the model bounds
        tracked_set.discard(O2)
        ample_set.discard(O2)
    if o2_mode == "kla":
        if O2 not in exchanged:
            raise ValueError("otr=(kLa, C*) but no model exchanges O2 (cpd00007)")
        tracked_set.add(O2)
        ample_set.discard(O2)
        S0.setdefault(O2, float(otr[1]))
    tracked_list = sorted(tracked_set)
    ledger = list(tracked_list)
    if O2 not in tracked_set and O2 in exchanged and o2_mode == "otr":
        ledger.append(O2)
    kin = _normalize_kinetics(kinetics, member_ids)
    default = _pair(default, "default_kinetics")
    warns = []
    unused = {(m, c) for m in kin for c in kin[m] if c not in tracked_set}
    if unused:
        warns.append(f"kinetics given for compounds that are not tracked (ignored): {sorted(unused)[:10]}")
    absent = set(S0) - exchanged
    if absent:
        warns.append(f"S0 names compounds no model exchanges (they stay in the pool untouched): {sorted(absent)}")
        tracked_list = sorted(tracked_set | absent)
        ledger = tracked_list + [c for c in ledger if c not in tracked_list]
    return o2_mode, o2_rule, S0, sorted(ample_set), tracked_list, ledger, kin, default, warns


def _tiebreak_record(tiebreak, pfba):
    if tiebreak is None:
        return {"stage": "pfba" if pfba else "vertex"}
    return {"stage": "secondary+pfba" if pfba else "secondary", "secondary": {k: tuple(v) for k, v in tiebreak.items()}}


def _check_tiebreak(tiebreak, form, member_ids):
    if tiebreak in (None, "pfba"):
        return None, True
    if tiebreak == "vertex":
        return None, False
    if not isinstance(tiebreak, Mapping):
        raise TypeError("tiebreak is None / 'pfba', 'vertex', or {member or 'community': (reaction id, 'min'|'max')}")
    allowed = {"community"} if form == "community_strict" else set(member_ids)
    bad = set(tiebreak) - allowed
    if bad:
        raise ValueError(f"tiebreak keys {sorted(bad)} are not valid for form {form!r} (use {sorted(allowed)})")
    return {k: (str(v[0]), str(v[1])) for k, v in tiebreak.items()}, True


def _run(stepper, members, X0, *, form, otr, feed, D, t_end, dt, record_every, tiebreak, pfba, extra_settings,
         setup, max_depletion, rtol, atol, dt_min, s_min, max_steps, callback, t_start=0.0):
    t0 = time.perf_counter()
    o2_mode, o2_rule, S0n, ample_list, tracked_list, ledger, kin, default, warns = setup
    feed_f = _as_function(feed, "feed")
    D_f = _as_function(D, "D")
    if feed_f is not None and D_f is None:
        raise ValueError("feed needs a dilution rate D (1/h; a callable D(t) = F(t) / V(t) for a fed-batch)")
    missing = set(members) - set(X0)
    if missing:
        raise KeyError(f"X0 has no biomass for {sorted(missing)}")
    if any(float(X0[m]) < 0 for m in members) or sum(float(X0[m]) for m in members) <= 0:
        raise ValueError("X0 must be >= 0 for every member and positive for at least one")
    if not (math.isfinite(float(t_start)) and float(t_start) <= float(t_end)):
        raise ValueError(f"t_start ({t_start}) must be a finite time <= t_end ({t_end})")
    warnings = list(warns)
    if form == "members" and o2_mode == "otr" and O2 not in ledger:
        warnings.append("otr is given but no member exchanges O2 (cpd00007): it has no effect")
    if not pfba:
        warnings.append("tiebreak='vertex': no pFBA stage; the by-product split is the solver's vertex and is "
                        "not reproducible across solvers")
    rec, led, stats = _integrate(stepper(tracked_list, ledger, o2_mode, o2_rule), members, X0, S0n, tracked_list,
                                 ledger, kin, default, float(t_end), float(dt), otr, o2_mode, feed_f, D_f,
                                 int(record_every), max_depletion, rtol, atol, dt_min, s_min, max_steps, warnings,
                                 callback, float(t_start))
    st = stepper.instance
    times = np.array(rec["t"])
    Xa = np.array(rec["X"]).reshape(len(times), len(members))
    Sa = np.array(rec["S"]).reshape(len(times), len(tracked_list))
    mua = np.array(rec["mu"]).reshape(len(times), len(members))
    rows = tuple(members) + (("shared",) if form == "community_strict" else ())
    nrow, nled = len(rows), len(ledger)
    up = np.array(led[0]).reshape(-1, nrow, nled) if led[0] else np.zeros((0, nrow, nled))
    se = np.array(led[1]).reshape(-1, nrow, nled) if led[1] else np.zeros((0, nrow, nled))
    inf = np.array(led[2]).reshape(-1, nled) if led[2] else np.zeros((0, nled))
    outf = np.array(led[3]).reshape(-1, nled) if led[3] else np.zeros((0, nled))
    stats.update({"lp_solves": st.n_solves, "lp_seconds": round(st.lp_seconds, 3),
                  "seconds": round(time.perf_counter() - t0, 3), "recorded": len(times)})
    settings = {"t_end": float(t_end), "dt": float(dt), "record_every": int(record_every), "X0": dict(X0),
                "S0": dict(S0n), "kinetics": {m: dict(v) for m, v in kin.items()}, "default_kinetics": default,
                "tracked": list(tracked_list), "ample": list(ample_list), "o2_mode": o2_mode,
                "feed": None if feed is None else ("callable" if callable(feed) else dict(feed)),
                "D": None if D is None else ("callable" if callable(D) else float(D)),
                "max_depletion": max_depletion, "rtol": rtol, "atol": atol, "dt_min": dt_min, "s_min": s_min,
                "t_start": float(t_start)}
    settings.update(extra_settings)
    return Trajectory(times=times, members=tuple(members),
                      X={m: Xa[:, i] for i, m in enumerate(members)},
                      concentrations={c: Sa[:, j] for j, c in enumerate(tracked_list)},
                      mu={m: mua[:, i] for i, m in enumerate(members)},
                      ledger_rows=rows, ledger_compounds=tuple(ledger), uptake=up, secretion=se, inflow=inf,
                      outflow=outf, o2_limited=np.array(rec["o2"], bool), status=rec["status"], form=form,
                      tiebreak=_tiebreak_record(tiebreak, pfba), o2_rule=o2_rule,
                      otr=None if otr is None else (tuple(otr) if isinstance(otr, (tuple, list)) else float(otr)),
                      settings=settings, stats=stats, warnings=warnings)


class _Factory:
    """Builds the stepper once _setup has fixed the tracked set, and keeps it for the statistics."""

    def __init__(self, build):
        self.build, self.instance = build, None

    def __call__(self, tracked, ledger, o2_mode, o2_rule):
        self.instance = self.build(tracked, ledger, o2_mode, o2_rule)
        return self.instance


_INTEGRATION_DEFAULTS = dict(max_depletion=0.5, rtol=1e-2, atol=1e-7, dt_min=1e-6, s_min=1e-12, max_steps=200000,
                             callback=None, t_start=0.0)


def simulate_models(models, t_end, dt, X0, S0, *, kinetics, otr=None, o2_rule=None, tiebreak=None, feed=None,
                    D=None, record_every=1, default_kinetics=DEFAULT_KINETICS, ample=None, tracked=None,
                    biomass=None, threads=1, copy=True, **integration) -> Trajectory:
    """form="members" on stand-alone models: {member id: cobra.Model} (e.g. member_model(comm, m), or isolates
    whose e0 exchanges use the same compound ids). Each model's biomass is biomass[member] or its objective's
    single reaction. copy=True works on copies, so the models given are not changed. See dynamic()."""
    ids = list(models)
    work = {mid: (m.copy() if copy else m) for mid, m in models.items()}
    bio = {mid: _biomass_of(work[mid], (biomass or {}).get(mid)) for mid in ids}
    tb, pfba = _check_tiebreak(tiebreak, "members", ids)
    opts = dict(_INTEGRATION_DEFAULTS, **integration)
    setup = _setup(list(work.values()), S0, feed, kinetics, default_kinetics, otr, o2_rule, "members", ample,
                   tracked, ids)
    factory = _Factory(lambda tr, led, mode, rule: _MembersStepper(work, bio, tr, led, mode, rule, tb, pfba, threads))
    try:
        return _run(factory, ids, X0, form="members", otr=otr, feed=feed, D=D, t_end=t_end, dt=dt,
                    record_every=record_every, tiebreak=tb, pfba=pfba, extra_settings={"biomass": bio},
                    setup=setup, **opts)
    finally:
        if factory.instance is not None:
            factory.instance.close()


def dynamic(comm, t_end, dt, X0, S0, *, kinetics, form="members", otr=None, o2_rule=None, tiebreak=None,
            feed=None, D=None, record_every=1, default_kinetics=DEFAULT_KINETICS, ample=None, tracked=None,
            member_models=None, repair=None, cap_basis="gross", kinetic_rows="lift", scale_bounds=True, threads=1,
            a_min=None, **integration) -> Trajectory:
    """Multi-species dynamic FBA of the community from X0 ({member: gDW/L}) and S0 ({compound: mM}) to t_end (h),
    steps of at most dt (h), recording every record_every-th accepted step. See the module docstring for the
    forms, units, tracked set, O2 and the integrator.

    kinetics          {member: {compound: (vmax mmol/gDW_member/h, Km mM)}} (compound as 'cpd00179',
                      'cpd00179_e0' or 'EX_cpd00179_e0'; a mapping with 'vmax'/'Km' also works); any tracked
                      compound a member has no entry for gets default_kinetics
    form              "members" (DMMM) or "community_strict"
    otr, o2_rule      O2 transfer: None, OTR (mmol/L/h) or (kLa, C*); o2_rule None means "demand" for members and
                      "joint" for community_strict (the resolved rule is recorded)
    tiebreak          None / "pfba": max growth then pFBA; "vertex": no pFBA (solver-dependent split);
                      {member: (reaction id, "min"|"max")} (members) or {"community": (...)} (community_strict):
                      that flux held at its min or max on the max-growth face, then pFBA
    feed, D           chemostat / fed-batch (see the module docstring)
    member_models     members form: {member: cobra.Model} to use instead of member_model(comm, m, repair=repair)
    cap_basis         community_strict: "gross" (cap_member_uptake rows on the import direction) or "net"
                      (member_uptake_expression net uptake, the prototype's rows)
    kinetic_rows      community_strict: "lift" (default; the *_commKin rows would forbid any flux of a member that
                      is not growing) or "keep". The member LPs below (a_min route, rate ceiling, feasibility
                      probe) are member_model copies and carry no *_commKin rows either way
    scale_bounds      community_strict: multiply every non-boundary member reaction bound by a_m (eq. 9)
    a_min             community_strict (default 1e-4): a member whose share a_m = X_m / X of the culture is below
                      a_min is taken out of the community LP for the step and solved by its own member LP
                      (member_model; the "members" form solve, with its own kinetic caps on the pool and, under an
                      OTR, the per-gDW O2 share OTR / X), with a warning. The community LP is per gDW of community,
                      so a rare member's fluxes sit at the solver's feasibility tolerance and mu_m = v_bio,m / a_m
                      turns that noise into a rate: a member growing at 2.857 1/h alone came out at 285.7 1/h at
                      a_m = 1e-10, 1000 at 1e-12 and 10 % low at 1e-6. The relative error is about tol / (q a_m)
                      (q its uptake rate), about 1e-3 at a_m = 1e-4. a_min=0 switches the route off.
                      Every rate the community LP gives is also checked against the member's own solo maximum on
                      the current pool (its member LP: tracked compounds at its kinetic caps, O2 not limited by the
                      transfer rate, forced bounds released) and clamped to it, its exchanges scaled to match: a rate
                      above what the member reaches alone is solver noise, or a medium bound that, being per gDW of
                      community, gives a small member more per gDW than it gets alone. A clamp by more than 1e-4
                      relative + 1e-4 1/h is a warning; smaller ones are solver tolerance (on the CommPhitting
                      maltose run, up to 7.5e-6 1/h at hundreds of steps) and are only counted, in
                      Trajectory.stats["notes"]["clamped_within_tolerance"] (stats["notes"] counts every such
                      event per member: "a_min", "clamped", "released" too).
                      When the community LP is infeasible, the forced bounds (lb > 0 or ub < 0, e.g. a maintenance
                      drain) of the member(s) that cannot meet them are released for that step and the LP is solved
                      again: first the members whose own LP is infeasible alone on the current pool, then, if that
                      is not enough, each other member with forced bounds in turn, then all of them; a warning names
                      each member, its reactions and the steps. Without this, one member's unpayable forced flux (a
                      drain on a substrate not yet in the pool) idled every member for the whole run.
    integration       max_depletion (0.5), rtol (1e-2), atol (1e-7 mM), dt_min (1e-6 h), s_min (1e-12 mM),
                      max_steps (200000), callback (None: a callable(t, {member: X}, {compound: S}) called at
                      every recorded state, e.g. to report progress), t_start (0.0 h: the time of X0 and S0;
                      the run covers t_start to t_end, and D(t) / feed(t) are called with that clock)
    The community model is left as it was (bounds, rows, objective). Returns a Trajectory."""
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}, not {form!r}")
    ids = [m.id for m in comm.members]
    if form == "members":
        if cap_basis != "gross" or kinetic_rows != "lift" or not scale_bounds or a_min is not None:
            logger.info("cap_basis, kinetic_rows, scale_bounds and a_min apply to form='community_strict' only")
        if member_models is None:
            models = {mid: member_model(comm, mid, repair=repair) for mid in ids}
            copy = False
        else:
            if repair is not None:
                raise ValueError("give either member_models or repair, not both")
            models = {mid: member_models[mid] for mid in ids}
            copy = True
        traj = simulate_models(models, t_end, dt, X0, S0, kinetics=kinetics, otr=otr, o2_rule=o2_rule,
                               tiebreak=tiebreak, feed=feed, D=D, record_every=record_every,
                               default_kinetics=default_kinetics, ample=ample, tracked=tracked, threads=threads,
                               copy=copy, **integration)
        traj.settings["member_models"] = "given" if member_models is not None else "member_model(comm, m)"
        traj.settings["repair"] = None if repair is None else ("callable" if callable(repair) else "mapping")
        return traj
    if member_models is not None:
        raise ValueError("member_models applies to form='members' only")
    if repair is not None:
        raise ValueError("form='community_strict' runs the community model as it is: apply a repair to "
                         "comm.util.model first")
    if cap_basis not in CAP_BASES:
        raise ValueError(f"cap_basis must be one of {CAP_BASES}, not {cap_basis!r}")
    a_min = A_MIN if a_min is None else float(a_min)
    if not (math.isfinite(a_min) and 0.0 <= a_min < 1.0):
        raise ValueError(f"a_min must be a share in [0, 1), not {a_min}")
    tb, pfba = _check_tiebreak(tiebreak, form, ids)
    opts = dict(_INTEGRATION_DEFAULTS, **integration)
    setup = _setup([comm.util.model], S0, feed, kinetics, default_kinetics, otr, o2_rule, form, ample, tracked, ids)
    with ExitStack() as stack:
        factory = _Factory(lambda tr, led, mode, rule: _CommunityStrictStepper(
            comm, tr, led, mode, rule, tb, pfba, cap_basis, kinetic_rows, scale_bounds, threads, stack, a_min))
        return _run(factory, ids, X0, form=form, otr=otr, feed=feed, D=D, t_end=t_end, dt=dt,
                    record_every=record_every, tiebreak=tb, pfba=pfba,
                    extra_settings={"cap_basis": cap_basis, "kinetic_rows": kinetic_rows,
                                    "scale_bounds": bool(scale_bounds), "a_min": a_min}, setup=setup, **opts)


# --------------------------------------------------------------------------- calibration
def _model_yield(model, biomass, cpd):
    """gDW per mmol of cpd for the model alone: uptake of cpd fixed at 1 mmol/gDW/h, max biomass."""
    ex = _exchanges(model).get(cpd)
    if ex is None:
        raise KeyError(f"{model.id} has no exchange for {cpd}")
    rxn, coef = ex
    with model:
        model.objective = model.reactions.get_by_id(biomass)
        if coef < 0:
            rxn.bounds = (-1.0 / -coef, -1.0 / -coef)
        else:
            rxn.bounds = (1.0 / coef, 1.0 / coef)
        value = model.slim_optimize(error_value=float("nan"))
    return None if not math.isfinite(value) else float(value)


def calibrate_monoculture(member_model, timeseries, params=("vmax",), *, substrate, S0, X0=None, kinetics=None,
                          default_kinetics=DEFAULT_KINETICS, otr=None, o2_rule=None, dt=0.05, t_end=None,
                          x0=None, bounds=None, substrate_weight=1.0, biomass=None, member_id="member",
                          max_nfev=50, **dynamic_kw) -> dict:
    """Fit the uptake kinetics (vmax and/or Km) of `substrate` to a monoculture time course by least squares
    (scipy.optimize.least_squares on log parameters) of dynamic FBA (simulate_models, one member) against
    ln X at the measured times and, when timeseries has "S", the substrate concentration (scaled by its
    initial value, weighted by substrate_weight).

    member_model  the member's cobra.Model (member_model(comm, m), or an isolate)
    timeseries    mapping or DataFrame with "t" (h), "X" (gDW/L) and optionally "S" (mM of the substrate)
    params        names among ("vmax", "Km")
    S0            {compound: mM} of the monoculture medium at the first measured time (must include the substrate);
                  X0 defaults to X at that time. The simulation starts at the first measured time t[0], not at
                  t = 0 (a series sampled from t = 2 h used to be simulated from 0 with X(2 h) as X0: vmax 1.687
                  for a truth of 2.0), and runs to t_end (default the last measured time)
    kinetics      the member's other kinetics {compound: (vmax, Km)}; the substrate's entry, if present, is the
                  starting point
    x0            starting values {param: value}; vmax defaults to the Louca-type design mu_obs / Y (mu_obs the
                  ln-X slope over the measured points below half the final biomass, Y the model's yield on the
                  substrate), Km to the substrate's entry or default_kinetics
    Returns {"params", "kinetics", "initial", "yield_gDW_per_mmol", "mu_obs", "cost", "rmse_lnX", "success",
    "message", "nfev", "trajectory"}."""
    from scipy.optimize import least_squares

    params = tuple(params)
    if not params or set(params) - {"vmax", "Km"}:
        raise ValueError(f"params must be a non-empty subset of ('vmax', 'Km'), not {params}")
    sub = _cpd(substrate)
    get = (lambda k: timeseries[k]) if not hasattr(timeseries, "columns") else (lambda k: timeseries[k].to_numpy())
    t_obs = np.asarray(get("t"), dtype=float)
    X_obs = np.asarray(get("X"), dtype=float)
    has_S = ("S" in timeseries) if isinstance(timeseries, Mapping) else ("S" in getattr(timeseries, "columns", ()))
    S_obs = np.asarray(get("S"), dtype=float) if has_S else None
    S0 = {_cpd(k): float(v) for k, v in S0.items()}
    if sub not in S0:
        raise ValueError(f"S0 must include the substrate {sub}")
    first = int(np.argmin(t_obs))
    t_start = float(dynamic_kw.pop("t_start", t_obs[first]))
    X0v = float(X0) if X0 is not None else float(X_obs[first])
    t_end = float(t_end) if t_end is not None else float(t_obs.max())
    bio = _biomass_of(member_model, biomass)
    own = {_cpd(k): _pair(v, f"kinetics[{k!r}]") for k, v in (kinetics or {}).items()}
    start = dict(zip(("vmax", "Km"), own.get(sub, _pair(default_kinetics, "default_kinetics"))))
    Y = _model_yield(member_model, bio, sub)
    ok = (X_obs > 0) & (X_obs <= 0.5 * np.nanmax(X_obs))
    mu_obs = float(np.polyfit(t_obs[ok], np.log(X_obs[ok]), 1)[0]) if ok.sum() >= 3 else None
    if mu_obs is not None and Y and mu_obs > 0:
        start["vmax"] = mu_obs / Y
    if x0:
        start.update({k: float(v) for k, v in x0.items()})
    initial = {p: start[p] for p in params}
    lo = np.log([(bounds or {}).get(p, (1e-6, 1e4))[0] for p in params])
    hi = np.log([(bounds or {}).get(p, (1e-6, 1e4))[1] for p in params])
    z0 = np.clip(np.log([initial[p] for p in params]), lo + 1e-9, hi - 1e-9)
    s_scale = max(S0[sub], float(np.nanmax(S_obs)) if S_obs is not None else 0.0, 1e-12)
    keep = X_obs > 0
    last = {}

    def simulate(z):
        vals = dict(start)
        vals.update({p: float(np.exp(v)) for p, v in zip(params, z)})
        kin = {member_id: dict(own, **{sub: (vals["vmax"], vals["Km"])})}
        traj = simulate_models({member_id: member_model}, t_end, dt, {member_id: X0v}, S0, kinetics=kin, otr=otr,
                               o2_rule=o2_rule, default_kinetics=default_kinetics, biomass={member_id: bio},
                               t_start=t_start, **dynamic_kw)
        last["traj"], last["vals"] = traj, vals
        return traj

    def residuals(z):
        traj = simulate(z)
        Xm = np.interp(t_obs, traj.times, traj.X[member_id])
        r = [np.log(np.maximum(Xm[keep], 1e-300)) - np.log(X_obs[keep])]
        if S_obs is not None and sub in traj.concentrations:
            Sm = np.interp(t_obs, traj.times, traj.concentrations[sub])
            msk = np.isfinite(S_obs)
            r.append(substrate_weight * (Sm[msk] - S_obs[msk]) / s_scale)
        return np.concatenate(r)

    fit = least_squares(residuals, z0, bounds=(lo, hi), diff_step=1e-3, max_nfev=max_nfev, x_scale=1.0)
    traj = simulate(fit.x)
    Xm = np.interp(t_obs, traj.times, traj.X[member_id])
    rmse = float(np.sqrt(np.mean((np.log(Xm[keep]) - np.log(X_obs[keep])) ** 2)))
    vals = last["vals"]
    return {"params": {p: vals[p] for p in params}, "kinetics": {sub: (vals["vmax"], vals["Km"])},
            "initial": initial, "yield_gDW_per_mmol": Y, "mu_obs": mu_obs, "cost": float(fit.cost),
            "rmse_lnX": rmse, "success": bool(fit.success), "message": fit.message, "nfev": int(fit.nfev),
            "trajectory": traj}
