# -*- coding: utf-8 -*-
"""Community composition: balanced growth (SteadyCom), kappa calibration and composition profiles.

Every function takes the MSCommunity `comm`, changes comm.util.model only inside `with
comm.util.model:` (plus explicit restores for what cobra does not track: solver tolerances and
threads, row bounds, member abundances) and leaves the community as it found it.

    steadycom(comm, ...)              balanced growth in the sense of Chan, Simons & Maranas 2017
                                      (PLoS Comput Biol 13:e1005539, doi:10.1371/journal.pcbi.1005539)
    calibrate_kappa(comm, ...)        kappa_sat,m: the smallest kinetic coefficient at which member m
                                      alone reaches its solo optimum (composition-free)
    kappa_sat_coupled(comm, a)        the same for the coupled form at a declared composition a
    composition_profile(comm, ...)    mu_com(a) of the coupled form on a grid, or the composition
                                      ranges on the composition-free face, with the flux carried by
                                      members of zero abundance

Units
    Fluxes are mmol per gDW of community per h (batch, X0 = 1) or mmol per L per h (chemostat,
    X_m in gDW/L). q (uptake caps) and A_m (maintenance) are mmol per gDW of THAT member per h;
    mu and D are 1/h; kappa is mmol per gDW (sum |v| per unit of biomass flux).

Why composition needs member-scaled bounds (Chan et al. 2017; abundance SYNTHESIS sec. 3.2)
    In an MSCommunity model every member flux is per gDW of community and no default bound
    scales with a member's own biomass, so one member's work can be credited to another member's
    biomass (relabelling). SteadyCom's eq. 9 (LB_j X_m <= v_j <= UB_j X_m) and per-member caps on
    X_m tie capacity to biomass. Even then, a max-mu balanced-growth model that reproduces the
    monocultures cannot predict coexistence below the fastest solo rate: every member alone is a
    feasible point of the LP, so mu_max >= max_m mu_solo,m, and a mixture is returned only where it
    is strictly faster. Every composition answer here is therefore a RANGE with diagnostics.
"""
from __future__ import annotations

import inspect
import logging
import math
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from optlang.symbolics import Zero, add

from mscommunity.commkineticpkg import member_kinetic_reactions
from mscommunity.commmembers import (
    CAP_PREFIX, cap_member_uptake, close_member_drains, lifted_kinetic_rows, member_biomass_variables,
    member_caps, member_reactions, open_member_drains, reactions_by_member, refresh_caps,
    set_member_maintenance)

logger = logging.getLogger(__name__)

__all__ = [
    "CompositionWarning", "SteadyComResult", "KappaCalibration", "ProfileResult", "ZeroMemberFlux",
    "steadycom", "calibrate_kappa", "kappa_sat_coupled", "composition_profile",
]

LIFT = 1e30                      # a lifted row bound (optlang's Gurobi interface refuses a free row)
OPT_SLACK = 1e-6                 # the "at the optimum" level is (1 - OPT_SLACK) x the optimum
DRAIN_UB = 1000.0                # member drains open to (0, DRAIN_UB) while member biomass leaves by them
CHEMOSTAT_BOUND_SCALE = 1e3      # eq. 9 in chemostat mode: own bounds x this, so LB X_m <= v <= UB X_m binds
_COMPOSITION_ROW_NAMES = ("min_comm_growth",)
_COMPOSITION_ROW_SUFFIXES = ("_regularization",)


# =========================================================================== records
@dataclass(frozen=True)
class CompositionWarning:
    """One diagnostic. code is one of
        corner             the optimum is a single-member corner (no mixture beats the best member
                           alone by more than corner_tol, or every other member is a sliver < x_min)
        solo_outside_ci    a member's solo rate (or solo X) lies outside the supplied monoculture CI
        relabelling        a member at a_m < x_min carries flux at its eq.-9 bound (or, without eq. 9,
                           carries flux at all): its biomass is a label, not a capacity
        not_identifiable   caps=None: no per-member capacity ties uptake to X_m, so the composition
                           is not identified by metabolism
        kappa_below_sat    a member's solo optimum is lower with the kappa rows than without them
                           (kappa < its kappa_sat): the kappa forbids that member's monoculture
        second_interval    feasibility in mu is not an interval; a higher feasible interval was found
        lifted_rows        rows that carry a declared composition were lifted for the run
        solver_status      an LP ended neither optimal nor infeasible (treated as infeasible)
        infeasible / unbounded   no balanced-growth solution / no finite optimum
        no_solo_growth     calibrate_kappa: a member does not grow alone, so it has no kappa_sat
        kappa_default      calibrate_kappa: a kappa_sat exceeds the installed or default kinetic_coeff
        caps_on_declared_a composition_profile("free"): abundance-basis caps read the declared a
    data holds the numbers behind the message (relabelling: {"a": a_m, "reactions": [(reaction,
    flux, bound x X_m), ...]} for every reaction at its bound)."""
    code: str
    message: str
    member: Optional[str] = None
    data: Optional[dict] = field(default=None, compare=False, repr=False)

    def __str__(self):
        return f"[{self.code}] {self.message}"


@dataclass
class SteadyComResult:
    """steadycom's answer. Ranges, never a bare point.

    mode            "batch" or "chemostat"
    status          "optimal", "infeasible" or "unbounded"
    mu_max          batch: the largest feasible balanced-growth rate, 1/h (None unless optimal)
    X_star          chemostat: the largest total biomass sum_m X_m at mu = D, gDW/L
    D               chemostat dilution rate (1/h)
    a_ranges        {fraction: {member: (lo, hi)}}: the range of a_m = X_m / sum X over the
                    feasible set at mu = fraction * mu_max (batch) or sum X = fraction * X_star
                    (chemostat); fraction 1.0 is evaluated at (1 - 1e-6) x the optimum. Each set is
                    the hull of its own range and those of every larger fraction (see steadycom).
                    None for a member whose range LP failed at every level down to that fraction.
    X_opt           {member: X_m} of one optimal solution (max X of the first member at fraction 1.0)
    fluxes          {reaction id: net flux} of that solution
    solo            {member: mu_m alone (batch, 1/h) or X_m alone (chemostat, gDW/L)}; None when
                    the member cannot grow alone
    solo_kappa_off  the same with the kappa rows lifted (only when kappa was given)
    best_solo       (member, value) of the best solo
    delta_over_best_solo  mu_max - best solo mu (batch) or X_star - best solo X (chemostat)
    cap_activity    {cap row: {"member", "cpd", "q", "uptake", "qX", "ratio"}} at X_opt
                    (uptake and qX in the model's flux units; ratio = uptake / (q X_m))
    warnings        [CompositionWarning]; see warning_codes
    mu_upper_bound  batch: max sum v_bio / X0 with the growth rows relaxed (no mu can exceed it)
    search_log      [(mu, feasible)] of the mu search (batch)
    released        {reaction: bounds} of the forced member bounds released for the run
    lifted          rows lifted for the run (community-kinetics rows, abundance-basis caps, ...)
    n_eq9_rows      number of eq.-9 rows written
    settings        the arguments the run used
    mu_at / X_at    re-solve at a given composition (see the methods)"""
    mode: str
    status: str
    mu_max: Optional[float] = None
    X_star: Optional[float] = None
    D: Optional[float] = None
    a_ranges: Dict[float, Dict[str, Optional[Tuple[float, float]]]] = field(default_factory=dict)
    X_opt: Dict[str, float] = field(default_factory=dict)
    fluxes: Dict[str, float] = field(default_factory=dict, repr=False)
    solo: Dict[str, Optional[float]] = field(default_factory=dict)
    solo_kappa_off: Optional[Dict[str, Optional[float]]] = None
    best_solo: Optional[Tuple[str, float]] = None
    delta_over_best_solo: Optional[float] = None
    cap_activity: Dict[str, dict] = field(default_factory=dict)
    warnings: List[CompositionWarning] = field(default_factory=list)
    mu_upper_bound: Optional[float] = None
    search_log: List[Tuple[float, bool]] = field(default_factory=list, repr=False)
    released: Dict[str, Tuple[float, float]] = field(default_factory=dict, repr=False)
    lifted: List[str] = field(default_factory=list, repr=False)
    n_eq9_rows: int = 0
    seconds: float = 0.0
    settings: dict = field(default_factory=dict, repr=False)
    _resolve: Optional[Callable] = field(default=None, repr=False, compare=False)

    @property
    def warning_codes(self):
        return {w.code for w in self.warnings}

    @property
    def optimum(self):
        return self.mu_max if self.mode == "batch" else self.X_star

    def mu_at(self, composition, *, fluxes=False):
        """batch: the largest mu at which the community grows in balanced growth with its
        composition FIXED (X_m = a_m X0), under the same caps, maintenance, kappa and eq. 9 as this
        run: one LP with mu as a variable (v_bio_m = mu a_m X0). None when no mu >= 0 is feasible.
        composition: {member: a} (normalised to sum 1), or a list of them (returns a list).
        fluxes=True returns (mu, {reaction: net flux}) instead. Rebuilds the run's LP on the
        community as it is now, so call it before changing the community."""
        if self.mode != "batch":
            raise ValueError("mu_at is for batch mode; in chemostat mode mu = D and X_at gives the biomass")
        return self._resolve("mu_at", composition, fluxes)

    def X_at(self, composition, *, fluxes=False):
        """chemostat: the largest total biomass sum X (gDW/L) at mu = D with the composition fixed
        (X_m = a_m sum X); one LP. Otherwise as mu_at."""
        if self.mode != "chemostat":
            raise ValueError("X_at is for chemostat mode; use mu_at in batch mode")
        return self._resolve("X_at", composition, fluxes)

    def summary(self) -> dict:
        """Plain numbers for logging / json."""
        return {"mode": self.mode, "status": self.status, "mu_max": self.mu_max, "X_star": self.X_star,
                "a_ranges": {str(f): r for f, r in self.a_ranges.items()}, "solo": self.solo,
                "solo_kappa_off": self.solo_kappa_off, "best_solo": self.best_solo,
                "delta_over_best_solo": self.delta_over_best_solo, "X_opt": self.X_opt,
                "cap_activity": self.cap_activity, "warnings": [str(w) for w in self.warnings],
                "n_eq9_rows": self.n_eq9_rows, "seconds": self.seconds}


class KappaCalibration(dict):
    """calibrate_kappa's answer: {member id: kappa} to hand to steadycom(kappa=...) or, for a single
    value, comm.add_commkinetics(max(calibration.values())).

    kappa_sat   {member: kappa_sat,m} (None for a member that does not grow alone)
    shared      max_m kappa_sat,m: the smallest single kappa at which every member reaches its solo
                optimum (the composition-free "kappa_mono")
    solo        {member: the kappa-off solo optimum the calibration used}
    per_member  whether the dict holds each member's own kappa_sat (True) or `shared` for all
    warnings    [CompositionWarning]"""

    def __init__(self, values, *, kappa_sat, shared, solo, per_member, warnings):
        super().__init__(values)
        self.kappa_sat = kappa_sat
        self.shared = shared
        self.solo = solo
        self.per_member = per_member
        self.warnings = warnings


@dataclass(frozen=True)
class ZeroMemberFlux:
    """The least flux a member of zero abundance must carry at the optimum (min sum |v| over its
    reactions, biomass excluded, with the objective held at (1 - 1e-6) x its optimum).
    total in the model's flux units; fluxes {reaction: net flux} of that minimal solution."""
    member: str
    composition: Dict[str, float]
    total: float
    fluxes: Dict[str, float]


@dataclass
class ProfileResult:
    """composition_profile's answer.

    objective     "coupled" or "free"
    status        "optimal" or "infeasible"
    optimum       coupled: max_a mu_com(a) (1/h); free: max sum_m v_bio,m (per gDW community per h)
    composition   {member: a} at the optimum (coupled: the refined argmax; free: one optimal vertex)
    sets          {fraction: ...}: coupled with two members, [(lo, hi), ...] intervals of the FIRST
                  member's a on the grid where mu(a) >= fraction * optimum; coupled with more
                  members, the grid compositions that pass; free, {member: (lo, hi)} ranges of
                  a_m = v_bio,m / sum v_bio (hull over that fraction and every larger one)
    curve         coupled: [(composition, mu or None)] on the grid
    zero_member_flux  [ZeroMemberFlux] for members at a_m = 0 (coupled: grid points with a zero
                  member; free: members whose range at the optimum reaches 0)
    warnings      [CompositionWarning]"""
    objective: str
    status: str
    optimum: Optional[float] = None
    composition: Dict[str, float] = field(default_factory=dict)
    sets: Dict[float, object] = field(default_factory=dict)
    curve: List[Tuple[Dict[str, float], Optional[float]]] = field(default_factory=list, repr=False)
    zero_member_flux: List[ZeroMemberFlux] = field(default_factory=list)
    warnings: List[CompositionWarning] = field(default_factory=list)
    seconds: float = 0.0


# =========================================================================== helpers
def _member(comm, key):
    """The CommunityMember given as the object, its id or its compartment index."""
    if isinstance(key, bool):
        raise TypeError(f"not a member: {key!r}")
    if isinstance(key, int):
        for m in comm.members:
            if m.index == key:
                return m
        raise KeyError(f"no member with index {key}")
    key = getattr(key, "id", key)
    if key not in comm.members:
        raise KeyError(f"no member {key!r}; members {[m.id for m in comm.members]}")
    return comm.members.get_by_id(key)


def _solve(model):
    """(objective value or None, solver status)."""
    v = model.slim_optimize(error_value=float("nan"))
    status = model.solver.status
    if status != "optimal" or v is None or math.isnan(v):
        return None, status
    return float(v), status


@contextmanager
def _solver_settings(model, feasibility_tol=None, threads=None):
    """Feasibility and optimality tolerances and (Gurobi) threads for the block; restored after.
    Neither is tracked by `with model:`."""
    cfg = model.solver.configuration
    saved_tol, saved_threads = [], None
    if feasibility_tol is not None:
        for attr in ("feasibility", "optimality"):
            try:
                old = getattr(cfg.tolerances, attr)
                setattr(cfg.tolerances, attr, float(feasibility_tol))
                saved_tol.append((attr, old))
            except Exception as exc:                               # noqa: BLE001
                logger.info("tolerance %s=%s not set on %s: %s", attr, feasibility_tol,
                            type(model.solver).__module__, exc)
    if threads is not None:
        params = getattr(model.solver.problem, "Params", None)
        if params is not None and hasattr(params, "Threads"):
            saved_threads = params.Threads
            params.Threads = int(threads)
        else:
            logger.info("solver_threads=%s ignored: %s has no thread parameter", threads,
                        type(model.solver).__module__)
    try:
        yield
    finally:
        for attr, old in saved_tol:
            try:
                setattr(cfg.tolerances, attr, old)
            except Exception:                                      # noqa: BLE001
                logger.warning("could not restore the %s tolerance to %s", attr, old)
        if saved_threads is not None:
            model.solver.problem.Params.Threads = saved_threads


def _lift_composition_rows(comm, stack):
    """Lift, for the life of `stack`, the rows that carry a declared composition: member caps and
    maintenance rows on basis "abundance" (q a_m, A_m a_m at the declared a_m) and the
    regularization rows of MSCommunity.regularization. Returns their names."""
    model, lifted = comm.util.model, []
    rows = [rec.name for rec in member_caps(comm) if rec.basis == "abundance"]
    rows += [c.name for c in model.constraints
             if c.name in _COMPOSITION_ROW_NAMES or c.name.endswith(_COMPOSITION_ROW_SUFFIXES)]
    for name in rows:
        row = model.constraints[name]
        saved = (row.lb, row.ub)
        # widen ub first: a lb above the current ub (or below it) is refused by optlang
        row.ub = LIFT if row.ub is not None else None
        row.lb = -LIFT if row.lb is not None else None
        stack.callback(_set_row_bounds, row, *saved)
        lifted.append(name)
    return lifted


def _set_row_bounds(row, lb, ub):
    """Set both bounds of an optlang row in an order the interface accepts."""
    if lb is not None and row.ub is not None and lb > row.ub:
        row.ub = ub
        row.lb = lb
    else:
        row.lb = lb
        row.ub = ub


def _kinetic_terms(comm, member):
    """{optlang variable: weight p_r} of the member's community-kinetics row without its biomass:
    the live <member>_commKin row's terms when the model has one, else the terms add_commkinetics
    would write (commkineticpkg.member_kinetic_reactions, weights comm.rxnProbs or 1)."""
    model = comm.util.model
    bio = member.primary_biomass
    skip = {bio.forward_variable.name, bio.reverse_variable.name}
    name = f"{member.id}_commKin"
    if name in model.constraints:
        row = model.constraints[name]
        coefs = row.get_linear_coefficients(row.variables)
        terms = {v: float(c) for v, c in coefs.items() if v.name not in skip and c != 0}
        if terms:
            return terms
    probs = getattr(comm, "rxnProbs", None) or {}
    terms = {}
    for r in member_kinetic_reactions(comm, member):
        w = float(probs.get(r.id, 1))
        terms[model.variables[r.forward_variable.name]] = w
        terms[model.variables[r.reverse_variable.name]] = w
    return terms


def _exchange(model, key):
    """The boundary reaction for an exchange id or a compound (cpd or cpd_e0)."""
    key = getattr(key, "id", key)
    if key in model.reactions:
        return model.reactions.get_by_id(key)
    cpd = key[:-3] if key.endswith("_e0") else key
    rid = f"EX_{cpd}_e0"
    if rid in model.reactions:
        return model.reactions.get_by_id(rid)
    raise KeyError(f"no exchange {key!r} (looked for {key} and {rid})")


def _normalize_caps(comm, caps):
    """[(member, cpd or None, reactions or None, q, label)] from
    {member: {cpd | (reaction ids,): q | {"q": q, "reactions": [...]}}}."""
    out = []
    for mkey, entries in (caps or {}).items():
        m = _member(comm, mkey)
        for key, value in entries.items():
            reactions = None
            if isinstance(value, Mapping):
                q, reactions = value["q"], value.get("reactions")
            else:
                q = value
            if isinstance(key, (tuple, list)):
                cpd, reactions = None, [getattr(r, "id", r) for r in key]
                label = reactions[0]
            else:
                cpd = str(getattr(key, "id", key))
                cpd = cpd[:-3] if cpd.endswith("_e0") else cpd
                label = cpd
            out.append((m, cpd, reactions, float(q), label))
    return out


def _normalize_maintenance(comm, maintenance):
    """[(member, reaction or None, A_m)] from {member: A_m | (reaction, A_m)}."""
    out = []
    for mkey, value in (maintenance or {}).items():
        m = _member(comm, mkey)
        if isinstance(value, (tuple, list)):
            rxn, A = value
        else:
            rxn, A = None, value
        out.append((m, rxn, float(A)))
    return out


def _normalize_kappa(comm, kappa):
    """{member id: kappa or None} or None."""
    if kappa is None:
        return None
    if isinstance(kappa, Mapping):
        out = {}
        for mkey, value in kappa.items():
            out[_member(comm, mkey).id] = None if value is None else float(value)
        missing = [m.id for m in comm.members if m.id not in out]
        if missing:
            raise ValueError(f"kappa has no value for {missing}; give one per member (None: no kinetic row)")
        return out
    return {m.id: float(kappa) for m in comm.members}


def _normalize_composition(comm, composition):
    """{member id: a} normalised to sum 1, for every member."""
    values = {}
    for mkey, v in composition.items():
        values[_member(comm, mkey).id] = float(v["abundance"] if isinstance(v, Mapping) else v)
    ids = [m.id for m in comm.members]
    missing = [k for k in ids if k not in values]
    if missing:
        raise ValueError(f"the composition has no value for {missing}")
    if any(v < 0 for v in values.values()):
        raise ValueError(f"negative abundance in {values}")
    total = sum(values.values())
    if not total > 0:
        raise ValueError("the composition sums to zero")
    return {k: values[k] / total for k in ids}


def _monoculture_ci(spec, rtol):
    """(lo, hi, reference) from a value, a (lo, hi) pair, (value, (lo, hi)) or
    {"mu" | "value": v, "ci": (lo, hi)}."""
    if isinstance(spec, Mapping):
        ref = spec.get("mu", spec.get("value"))
        ci = spec.get("ci", spec.get("ci95"))
        if ci is not None:
            return float(ci[0]), float(ci[1]), ref
        return float(ref) * (1 - rtol), float(ref) * (1 + rtol), ref
    if isinstance(spec, (tuple, list)):
        if len(spec) == 2 and isinstance(spec[1], (tuple, list)):
            return float(spec[1][0]), float(spec[1][1]), float(spec[0])
        if len(spec) == 2:
            return float(spec[0]), float(spec[1]), None
        raise ValueError(f"monoculture rate {spec!r}: give a value, (lo, hi) or (value, (lo, hi))")
    return float(spec) * (1 - rtol), float(spec) * (1 + rtol), float(spec)


def _hull(ranges_by_level):
    """Nest the ranges: each fraction's range becomes the hull of its own and every larger
    fraction's (fractions in descending order)."""
    out, acc = {}, {}
    for f in sorted(ranges_by_level, reverse=True):
        cur = {}
        for k, r in ranges_by_level[f].items():
            prev = acc.get(k)
            if r is None:
                cur[k] = prev
            elif prev is None:
                cur[k] = r
            else:
                cur[k] = (min(r[0], prev[0]), max(r[1], prev[1]))
        out[f] = cur
        acc = {k: v for k, v in cur.items() if v is not None}
    return out


def _net_fluxes(model, primal):
    return {r.id: primal[r.forward_variable.name] - primal[r.reverse_variable.name] for r in model.reactions}


def _default_kinetic_coeff():
    try:
        from mscommunity.mscommsim import MSCommunity
        return inspect.signature(MSCommunity.__init__).parameters["kinetic_coeff"].default
    except Exception:                                              # noqa: BLE001
        return None


# =========================================================================== the balanced-growth LP
class _BalancedGrowth:
    """The SteadyCom LP on comm.util.model for the life of a `with` block.

    On entry (all inside `with comm.util.model:`): solver tolerances / threads set; the package's
    *_commKin rows and the rows that carry a declared composition lifted; bio1 shut; every member
    drain opened to (0, 1000); forced member bounds (lb > 0 or ub < 0) released to include 0;
    supplies written on the exchanges; X_m variables (member_biomass_variables) bounded; the rows
        sum_m X_m = X0                              (batch; chemostat: >= 0, fixed only for ranges)
        v_bio,m - mu X_m = 0                        growth (mu a coefficient, searched; chemostat: D)
        v_j - UB_j X_m <= 0, v_j - LB_j X_m >= 0    eq. 9, every scaled member reaction, nonzero bound
        sum_r p_r |v_r| - kappa_m v_bio,m <= 0      when kappa is given (= kappa mu X_m by the growth row)
        gross uptake - q X_m <= 0                   cap_member_uptake(basis="X")
        ATPM_m - A_m X_m >= 0                       set_member_maintenance(basis="X")
    added. On exit all of it is undone."""

    def __init__(self, comm, *, caps=None, maintenance=None, kappa=None, scale_bounds="all", mode="batch",
                 D=None, supplies=None, X0=1.0, feasibility_tol=1e-9, solver_threads=None):
        if mode not in ("batch", "chemostat"):
            raise ValueError(f"mode must be 'batch' or 'chemostat', not {mode!r}")
        if mode == "chemostat" and not (D is not None and D > 0):
            raise ValueError("chemostat mode needs the dilution rate D > 0 (1/h)")
        if mode == "batch" and not X0 > 0:
            raise ValueError("X0 must be > 0")
        if scale_bounds not in ("all", None) and isinstance(scale_bounds, str):
            raise ValueError(f"scale_bounds must be 'all', None or reaction ids, not {scale_bounds!r}")
        self.comm, self.model = comm, comm.util.model
        self.members = list(comm.members)
        self.ids = [m.id for m in self.members]
        self.mode, self.D, self.X0 = mode, (float(D) if D is not None else None), float(X0)
        self.supplies = dict(supplies or {})
        self.tol, self.threads = feasibility_tol, solver_threads
        self.caps_spec = _normalize_caps(comm, caps)
        self.maint_spec = _normalize_maintenance(comm, maintenance)
        self.kappa = _normalize_kappa(comm, kappa)
        self.scale = scale_bounds if scale_bounds in ("all", None) else {getattr(r, "id", r) for r in scale_bounds}
        self.statuses = []

    # ------------------------------------------------------------------ set-up / tear-down
    def __enter__(self):
        stack = ExitStack()
        try:
            self._setup(stack)
        except BaseException:
            stack.close()
            raise
        self._stack = stack
        return self

    def __exit__(self, *exc):
        self._stack.close()
        return False

    def _setup(self, stack):
        comm, model = self.comm, self.model
        stack.enter_context(model)       # bounds, rows, variables, objective: undone by cobra on exit
        stack.enter_context(_solver_settings(model, self.tol, self.threads))
        self.lifted = list(stack.enter_context(lifted_kinetic_rows(comm)))
        self.lifted += _lift_composition_rows(comm, stack)
        model.reactions.get_by_id(comm.primary_biomass.id).bounds = (0.0, 0.0)
        open_member_drains(comm, (0.0, DRAIN_UB))
        owned = reactions_by_member(comm, include_biomass=False)
        self.released = {}
        for m in self.members:
            for r in owned[m.id]:
                lb, ub = r.lower_bound, r.upper_bound
                if lb > 0 or ub < 0:
                    self.released[r.id] = (lb, ub)
                    r.bounds = (min(lb, 0.0), max(ub, 0.0))
        self.supplied = {}
        for key, value in self.supplies.items():
            rxn = _exchange(model, key)
            self.supplied[rxn.id] = tuple(rxn.bounds)
            if isinstance(value, (tuple, list)):
                rxn.bounds = (float(value[0]), float(value[1]))
            else:
                rxn.lower_bound = -float(value)

        # member biomass variables (shared with the other composition methods; bounds restored)
        names = {m.id: f"X_{m.id}" for m in self.members}
        for mid, name in names.items():
            if name in model.variables:
                var = model.variables[name]
                stack.callback(var.set_bounds, var.lb, var.ub)
        self.X = member_biomass_variables(comm)
        for var in self.X.values():
            var.set_bounds(0.0, self.X0 if self.mode == "batch" else None)

        P, rows = model.problem, []
        Xsum = add(list(self.X.values()))
        if self.mode == "batch":
            self.sum_row = P.Constraint(Xsum, lb=self.X0, ub=self.X0, name="sc_sum_X")
        else:
            self.sum_row = P.Constraint(Xsum, lb=0.0, name="sc_sum_X")
        rows.append(self.sum_row)
        self.mu_now = self.D if self.mode == "chemostat" else 1.0
        self.growth = {}
        for m in self.members:
            row = P.Constraint(m.primary_biomass.flux_expression - self.mu_now * self.X[m.id], lb=0.0, ub=0.0,
                               name=f"sc_growth_{m.id}")
            self.growth[m.id] = row
            rows.append(row)

        # SteadyCom eq. 9: LB_j X_m <= v_j <= UB_j X_m
        self.eq9 = []         # (reaction id, member id, lb, ub, forward name, reverse name)
        self.n_eq9 = 0
        if self.scale is not None:
            factor = (max(1.0, self.X0) if self.mode == "batch" else CHEMOSTAT_BOUND_SCALE)
            for m in self.members:
                for r in owned[m.id]:
                    if self.scale != "all" and r.id not in self.scale:
                        continue
                    lb, ub = r.lower_bound, r.upper_bound
                    wrote = False
                    if ub > 0 and math.isfinite(ub):
                        rows.append(P.Constraint(r.flux_expression - ub * self.X[m.id], ub=0.0, name=f"sc_ub_{r.id}"))
                        self.n_eq9 += 1
                        wrote = True
                    if lb < 0 and math.isfinite(lb):
                        rows.append(P.Constraint(r.flux_expression - lb * self.X[m.id], lb=0.0, name=f"sc_lb_{r.id}"))
                        self.n_eq9 += 1
                        wrote = True
                    if wrote:
                        self.eq9.append((r.id, m.id, lb, ub, r.forward_variable.name, r.reverse_variable.name))
                        if factor != 1.0:
                            r.bounds = (lb * factor if math.isfinite(lb) else lb, ub * factor if math.isfinite(ub) else ub)

        # kinetic rows, X form through the growth row: sum p |v| <= kappa v_bio,m = kappa mu X_m
        self.kin = {}
        kin_terms = {}
        if self.kappa:
            for m in self.members:
                if self.kappa.get(m.id) is None:
                    continue
                row = P.Constraint(Zero, ub=0.0, name=f"sc_kin_{m.id}")
                self.kin[m.id] = row
                kin_terms[m.id] = _kinetic_terms(comm, m)
                rows.append(row)
        model.add_cons_vars(rows)
        model.solver.update()
        for mid, row in self.kin.items():
            bio = comm.members.get_by_id(mid).primary_biomass
            k = self.kappa[mid]
            coefs = dict(kin_terms[mid])
            coefs[bio.forward_variable] = -k
            coefs[bio.reverse_variable] = k
            row.set_linear_coefficients(coefs)

        # per-member caps and maintenance on X_m (commmembers; registered and undone with the context)
        self.caps = {}
        for m, cpd, reactions, q, label in self.caps_spec:
            name = f"{CAP_PREFIX}sc_{m.id}_{label}"
            cap_member_uptake(comm, m, cpd, q=q, reactions=reactions, basis="X", X=self.X, name=name)
            self.caps[name] = (m.id, cpd, q)
        self.maint = {}
        for m, rxn, A in self.maint_spec:
            row = set_member_maintenance(comm, m, A, reaction=rxn, basis="X", X=self.X)
            self.maint[row.name] = (m.id, A)
        model.solver.update()
        self.mu_var = None
        self._objective(self.X[self.ids[0]], "max")

    # ------------------------------------------------------------------ primitives
    def _objective(self, expr, direction):
        self.model.objective = self.model.problem.Objective(expr, direction=direction)

    def solve(self):
        v, status = _solve(self.model)
        if status not in ("optimal", "infeasible"):
            self.statuses.append(status)
        return v

    def set_mu(self, mu):
        self.mu_now = float(mu)
        for mid, row in self.growth.items():
            row.set_linear_coefficients({self.X[mid]: -self.mu_now})

    def feasible(self, mu):
        self.set_mu(mu)
        self._objective(self.X[self.ids[0]], "max")
        return self.solve() is not None

    @contextmanager
    def fixed_X(self, values):
        saved = {k: (v.lb, v.ub) for k, v in self.X.items()}
        try:
            for k, val in values.items():
                self.X[k].set_bounds(float(val), float(val))
            yield
        finally:
            for k, (lb, ub) in saved.items():
                self.X[k].set_bounds(lb, ub)

    @contextmanager
    def growth_free(self, members):
        """The growth rows of `members` relaxed to v_bio,m >= 0 (no mu coupling) for the block."""
        try:
            for mid in members:
                row = self.growth[mid]
                row.set_linear_coefficients({self.X[mid]: 0.0})
                row.ub = None
            yield
        finally:
            for mid in members:
                row = self.growth[mid]
                row.ub = 0.0
                row.set_linear_coefficients({self.X[mid]: -self.mu_now})

    @contextmanager
    def kappa_lifted(self):
        try:
            for row in self.kin.values():
                row.ub = LIFT
            yield
        finally:
            for row in self.kin.values():
                row.ub = 0.0

    @contextmanager
    def total_fixed(self, total):
        """chemostat: sum X fixed at `total` for the block."""
        try:
            _set_row_bounds(self.sum_row, float(total), float(total))
            yield
        finally:
            _set_row_bounds(self.sum_row, 0.0, None)

    def primal(self):
        return dict(self.model.solver.primal_values)

    def biomass_sum(self):
        return add([m.primary_biomass.flux_expression for m in self.members])

    # ------------------------------------------------------------------ solves
    def solo(self, k):
        """batch: member k alone (X_k = X0, the others 0), max v_bio,k / X0 (1/h); chemostat: the
        others at X = 0, max X_k at mu = D (gDW/L). None when infeasible."""
        if self.mode == "batch":
            with self.fixed_X({j: (self.X0 if j == k else 0.0) for j in self.ids}), self.growth_free([k]):
                self._objective(self.comm.members.get_by_id(k).primary_biomass.flux_expression, "max")
                v = self.solve()
            return None if v is None else v / self.X0
        with self.fixed_X({j: 0.0 for j in self.ids if j != k}):
            self._objective(self.X[k], "max")
            return self.solve()

    def mu_upper_bound(self):
        """max sum_m v_bio,m / X0 with every growth row relaxed: no balanced-growth mu exceeds it
        (sum_m v_bio,m = mu X0 at any balanced point). None if infeasible, inf if unbounded."""
        with self.growth_free(self.ids):
            self._objective(self.biomass_sum(), "max")
            v, status = _solve(self.model)
        if v is None:
            return math.inf if status == "unbounded" else None
        return v / self.X0

    def search_mu(self, seeds, up=1.02, rtol=1e-9, down=1.1, probes=(1.01, 1.05, 1.2, 1.5, 2.0)):
        """(mu_max or None, status, log, notes). Scan up from the best feasible seed by `up` until
        infeasible, bisect to relative `rtol`, then probe above for a second feasible interval and
        continue from there if one is found. Without a feasible seed, scan down from the
        relaxation bound by `down`. Never assumes feasibility is monotone in mu."""
        log, notes = [], []

        def test(mu):
            ok = self.feasible(mu)
            log.append((float(mu), bool(ok)))
            return ok

        mu_ub = self.mu_upper_bound()
        if mu_ub is None:
            return None, "infeasible", log, notes
        if math.isinf(mu_ub):
            notes.append("the relaxation is unbounded; mu searched up to 1e3 1/h")
            mu_ub = 1e3
        self.mu_ub = mu_ub
        start = None
        for s in sorted({float(s) for s in seeds if s is not None and s > 0}, reverse=True):
            p = min(s * (1 - 1e-9), mu_ub)
            if test(p):
                start = p
                break
        if start is None:
            p, floor = mu_ub, mu_ub * 1e-7
            while p > floor:
                if test(p):
                    start = p
                    break
                p /= down
        if start is None:
            return None, "infeasible", log, notes
        a = start
        for _ in range(10):
            b, p = None, a
            while True:
                q = p * up
                if q >= mu_ub:
                    if test(mu_ub):
                        a, b = mu_ub, None
                    else:
                        a, b = p, mu_ub
                    break
                if test(q):
                    p = q
                else:
                    a, b = p, q
                    break
            if b is not None:
                while b - a > rtol * b:
                    mid = 0.5 * (a + b)
                    if test(mid):
                        a = mid
                    else:
                        b = mid
            nxt = None
            for f in probes:
                q = min(a * f, mu_ub)
                if q <= a * (1 + 1e3 * rtol):
                    continue
                if test(q):
                    nxt = q
                    break
                if q >= mu_ub:
                    break
            if nxt is None:
                break
            notes.append(f"feasible again at mu = {nxt:.9g} above an infeasible gap after {a:.9g}")
            a = nxt
        self.set_mu(a)
        return a, "optimal", log, notes

    def ranges(self, level):
        """{member: (lo, hi) of a_m} and the primal values of every range LP, at mu = level
        (batch) or sum X = level (chemostat)."""
        out, sols = {}, []

        def run():
            denom = self.X0 if self.mode == "batch" else level
            for mid in self.ids:
                vals = []
                for sense in ("min", "max"):
                    self._objective(self.X[mid], sense)
                    v = self.solve()
                    vals.append(v)
                    if v is not None:
                        sols.append((mid, sense, self.primal()))
                out[mid] = None if None in vals else (vals[0] / denom, vals[1] / denom)

        if self.mode == "batch":
            self.set_mu(level)
            run()
        else:
            with self.total_fixed(level):
                run()
        return out, sols

    def mu_at(self, composition, want_fluxes=False):
        comp = _normalize_composition(self.comm, composition)
        if self.mu_var is None:
            self.mu_var = self.model.problem.Variable("sc_mu", lb=0.0)
            self.model.add_cons_vars([self.mu_var])
            self.model.solver.update()
        vals = {k: comp[k] * self.X0 for k in self.ids}
        with self.fixed_X(vals):
            try:
                for mid, row in self.growth.items():
                    row.set_linear_coefficients({self.X[mid]: 0.0, self.mu_var: -vals[mid]})
                self._objective(self.mu_var, "max")
                v = self.solve()
                fl = _net_fluxes(self.model, self.primal()) if (want_fluxes and v is not None) else None
            finally:
                for mid, row in self.growth.items():
                    row.set_linear_coefficients({self.X[mid]: -self.mu_now, self.mu_var: 0.0})
        return (v, fl) if want_fluxes else v

    def X_at(self, composition, want_fluxes=False):
        comp = _normalize_composition(self.comm, composition)
        P, total = self.model.problem, add(list(self.X.values()))
        rows = [P.Constraint(self.X[k] - comp[k] * total, lb=0.0, ub=0.0, name=f"sc_comp_{k}") for k in self.ids[:-1]]
        self.model.add_cons_vars(rows)
        try:
            self._objective(total, "max")
            v = self.solve()
            fl = _net_fluxes(self.model, self.primal()) if (want_fluxes and v is not None) else None
        finally:
            self.model.remove_cons_vars(rows)
        return (v, fl) if want_fluxes else v

    def kinetic_ratio(self, k, solo_value, slack=1e-7):
        """min sum_r p_r |v_r| / v_bio,k with member k alone at (1 - slack) x its solo optimum."""
        member = self.comm.members.get_by_id(k)
        terms = _kinetic_terms(self.comm, member)
        if self.mode == "batch":
            mu = solo_value * (1 - slack)
            fix, vbio = {j: (self.X0 if j == k else 0.0) for j in self.ids}, mu * self.X0
            self.set_mu(mu)
        else:
            xk = solo_value * (1 - slack)
            fix, vbio = {j: (xk if j == k else 0.0) for j in self.ids}, self.D * xk
        with self.fixed_X(fix):
            self.model.objective = self.model.problem.Objective(Zero, direction="min")
            self.model.objective.set_linear_coefficients(terms)
            v = self.solve()
        return None if v is None else v / vbio

    # ------------------------------------------------------------------ diagnostics
    def relabelling(self, primal, x_min, flux_tol=1e-9):
        """[(member, a_m, [(reaction, flux, bound x X_m)])] for members with 0 <= a_m < x_min that
        carry flux at an eq.-9 bound (with eq. 9) or any flux (without) in this solution."""
        total = sum(primal[v.name] for v in self.X.values())
        if not total > 0:
            return []
        out = []
        by_member = {}
        for rid, mid, lb, ub, fwd, rev in self.eq9:
            by_member.setdefault(mid, []).append((rid, lb, ub, fwd, rev))
        for mid in self.ids:
            x = primal[self.X[mid].name]
            a = x / total
            if a >= x_min:
                continue
            hits = []
            if self.eq9:
                for rid, lb, ub, fwd, rev in by_member.get(mid, []):
                    v = primal[fwd] - primal[rev]
                    if ub > 0 and v > flux_tol and v >= ub * x * (1 - 1e-6) - flux_tol:
                        hits.append((rid, v, ub * x))
                    elif lb < 0 and v < -flux_tol and v <= lb * x * (1 - 1e-6) + flux_tol:
                        hits.append((rid, v, lb * x))
            else:
                member = self.comm.members.get_by_id(mid)
                for r in member_reactions(self.comm, member, include_biomass=False):
                    v = primal[r.forward_variable.name] - primal[r.reverse_variable.name]
                    if abs(v) > 1e-6:
                        hits.append((r.id, v, None))
            if hits:
                hits.sort(key=lambda h: -abs(h[1]))
                out.append((mid, a, hits))
        return out

    def cap_activity(self, primal):
        out = {}
        for name, (mid, cpd, q) in self.caps.items():
            row = self.model.constraints[name]
            coefs = row.get_linear_coefficients(row.variables)
            value = sum(float(c) * primal[v.name] for v, c in coefs.items())
            x = primal[self.X[mid].name]
            uptake = value + q * x
            out[name] = {"member": mid, "cpd": cpd, "q": q, "uptake": uptake, "qX": q * x,
                         "ratio": uptake / (q * x) if q * x > 1e-12 else None}
        return out


# =========================================================================== steadycom
def steadycom(comm, *, caps=None, maintenance=None, kappa=None, scale_bounds="all", mode="batch", D=None,
              supplies=None, X0=1.0, fractions=(0.999, 0.99, 0.95), monoculture_rates=None, feasibility_tol=1e-9,
              solver_threads=None, x_min=1e-3, corner_tol=1e-6, monoculture_rtol=0.05, mu_rtol=1e-9
              ) -> SteadyComResult:
    """Balanced growth of the community (SteadyCom; Chan, Simons & Maranas 2017, PLoS Comput Biol
    13:e1005539): every member grows at one rate mu, its biomass X_m is an LP variable, and every
    member flux is bounded by its specific bounds times X_m. Returns a SteadyComResult with
    composition RANGES, solo viability and diagnostics; never a bare point.

    READ THIS FIRST. A max-mu balanced-growth model that reproduces the monocultures cannot predict
    coexistence below the fastest solo rate: each member alone is a feasible point of the LP, so
    mu_max >= max_m mu_solo,m, and a mixture is chosen only where it is strictly faster than the
    best member alone. With the drains open and no caps (scale_bounds=None, caps=None,
    maintenance=None) this reduces to predict_abundances(regularization=False): max sum_m v_bio,m,
    whose composition is a vertex of a typically flat face. Composition is identified only by
    capacities that scale with X_m (caps, eq. 9 with meaningful bounds, kappa) and only where they
    bind.

    LP (batch; the kappa row is written as kappa v_bio,m, which equals kappa mu X_m by the growth row):
        X_m >= 0, sum_m X_m = X0                             gDW (X0 = 1: fractions of community biomass)
        v_bio,m - mu X_m = 0                                 mu fixed per LP, searched
        LB_j X_m <= v_j <= UB_j X_m                          eq. 9, scale_bounds="all": every member
                                                             reaction (biomass and drain excluded) with a
                                                             nonzero bound; None: none; or reaction ids
        gross uptake_m(c) - q X_m <= 0                       caps (commmembers.cap_member_uptake, basis "X")
        ATPM_m - A_m X_m >= 0                                maintenance (set_member_maintenance, basis "X")
        sum_r p_r |v_r| - kappa_m v_bio,m <= 0               kappa given: the *_commKin term set and weights
    The package's own *_commKin rows are lifted for the run (kappa=None: no kinetic rows at all).

    Set-up and restore. bio1 is shut and every member drain opened to (0, 1000), so member biomass
    leaves through its drain at mu X_m. Forced bounds on member reactions (lb > 0 or ub < 0, e.g.
    ATPM floors written as a_m x A_m for a declared composition) are released to include 0: they
    carry the declared composition, and maintenance re-enters only as A_m X_m via `maintenance`.
    Member caps / maintenance rows on basis "abundance" and regularization rows are lifted for the
    same reason (warning lifted_rows). Exchange bounds are kept unless `supplies` overrides them.
    Everything is undone on return (`with comm.util.model:` plus explicit restores).

    Arguments
        caps         {member: {cpd: q, (reaction ids,): q, cpd: {"q": q, "reactions": [...]}}}, q in
                     mmol per gDW of that member per h on the member's GROSS uptake from e0
                     (cap_member_uptake). A tuple key caps the forward flux of those reactions.
        maintenance  {member: A_m or (ATPM reaction, A_m)}, mmol ATP per gDW of that member per h
        kappa        None, one float, or {member: kappa_m} (e.g. calibrate_kappa(comm, per_member=True))
        mode         "batch": max mu at sum X = X0. "chemostat": mu = D fixed, X_m in gDW/L, maximise
                     sum_m X_m in one LP; exchange bounds are then per litre (mmol/L/h)
        supplies     {exchange id or cpd: uptake rate} (lower bound -rate) or {...: (lb, ub)}; batch in
                     mmol per X0 gDW per h, chemostat in mmol per L per h (D x the feed concentration)
        fractions    composition ranges are reported at each fraction of the optimum, and at 1.0
        monoculture_rates  {member: mu | (lo, hi) | (mu, (lo, hi)) | {"mu": .., "ci": (lo, hi)}}: the
                     measured monoculture rate (X in chemostat mode); a bare value gets +-monoculture_rtol
        feasibility_tol    solver feasibility and optimality tolerance for the run (at SynCom rates of
                     ~0.017 1/h the solver default 1e-6 is ~6e-5 relative)
        solver_threads     Gurobi Threads for the run
        x_min        a member below this fraction counts as a sliver (relabelling / corner checks)
        corner_tol   relative gain over the best solo below which the optimum is called a corner
        mu_rtol      relative width of the final mu bisection bracket

    mu search (batch). Solo optima first (member k alone: X_k = X0, the others 0, max v_bio,k).
    From the best feasible seed (best solo, and mu at the declared composition) mu is scanned up by
    2 % until infeasible, bisected to mu_rtol, then probed at 1.01-2 x for a second feasible interval
    (kappa rows plus a forced flux make feasibility an interval in mu, not a downward-closed set);
    no seed feasible: scanned down from the relaxation bound max sum v_bio / X0. If nothing is
    feasible the status is "infeasible" and mu_max None: a false 0 is never returned.

    Ranges. a_ranges[f] is min / max of X_m / sum X at mu = f mu_max (sum X = f X_star in chemostat
    mode); fraction 1.0 at (1 - 1e-6) x the optimum. Each set is then widened to the hull of the
    sets at every larger fraction. Without forced fluxes the feasible set at f mu_max already
    contains every composition that reaches a mu in [f mu_max, mu_max] (scale the fluxes down);
    with maintenance or other forced flux that need not hold, and the hull is the safe report.

    Warnings (result.warnings, also logged): corner (delta_over_best_solo <= corner_tol x best solo,
    or every member but one below x_min at the optimum), solo_outside_ci, relabelling (a member below
    x_min carrying flux at an eq.-9 bound in an optimal solution), not_identifiable (caps None),
    kappa_below_sat, second_interval, lifted_rows, solver_status."""
    t0 = time.time()
    settings = dict(caps=caps, maintenance=maintenance, kappa=kappa, scale_bounds=scale_bounds, mode=mode, D=D,
                    supplies=supplies, X0=X0, feasibility_tol=feasibility_tol, solver_threads=solver_threads)
    fractions = tuple(sorted({float(f) for f in fractions if 0 < float(f) < 1}, reverse=True))
    warns = []

    def warn(code, message, member=None, data=None):
        w = CompositionWarning(code, message, member, data)
        warns.append(w)
        logger.warning("steadycom: %s", w)

    if not caps:
        warn("not_identifiable", "caps=None: no per-member capacity ties uptake to X_m, so the composition is "
             "not identified by metabolism (relabelling); read a_ranges as the set of compositions the "
             "stoichiometry allows, not as a prediction")

    result = SteadyComResult(mode=mode, status="infeasible", D=D if mode == "chemostat" else None,
                             settings=settings)
    with _BalancedGrowth(comm, **settings) as bg:
        result.released, result.lifted, result.n_eq9_rows = dict(bg.released), list(bg.lifted), bg.n_eq9
        abundance_rows = [n for n in bg.lifted if not n.endswith("_commKin")]
        if abundance_rows:
            warn("lifted_rows", f"rows that carry the declared composition were lifted for this run: {abundance_rows}; "
                 "pass them as caps= / maintenance= to apply them on X_m")
        solo = {k: bg.solo(k) for k in bg.ids}
        result.solo = solo
        if bg.kin:
            with bg.kappa_lifted():
                result.solo_kappa_off = {k: bg.solo(k) for k in bg.ids}
            for k in bg.ids:
                on, off = solo[k], result.solo_kappa_off[k]
                if off is not None and off > 1e-9 and (on is None or on < off * (1 - 1e-6)):
                    warn("kappa_below_sat", f"{k}: solo {'infeasible' if on is None else f'{on:.6g}'} with kappa "
                         f"{bg.kappa[k]:g} against {off:.6g} without it: the kappa is below this member's kappa_sat "
                         "and forbids its solo optimum", k)
        growers = {k: v for k, v in solo.items() if v is not None}
        if growers:
            best = max(growers, key=growers.get)
            result.best_solo = (best, growers[best])

        if mode == "batch":
            seeds = [result.best_solo[1]] if result.best_solo else []
            declared = {m.id: m.abundance for m in comm.members}
            if sum(declared.values()) > 0:
                seeds.append(bg.mu_at(declared))
            mu_max, status, log, notes = bg.search_mu(seeds, rtol=mu_rtol)
            result.search_log = log
            result.mu_upper_bound = getattr(bg, "mu_ub", None)
            for note in notes:
                warn("second_interval" if "feasible again" in note else "unbounded", note)
            result.status = status
            if status != "optimal":
                warn("infeasible", "no feasible balanced-growth rate: the community cannot grow in balanced "
                     "growth under these constraints (mu_max is None, not 0)")
            else:
                result.mu_max = mu_max
                levels = {1.0: mu_max * (1 - OPT_SLACK), **{f: f * mu_max for f in fractions}}
        else:
            bg._objective(add(list(bg.X.values())), "max")
            v, status = _solve(bg.model)
            if v is None:
                result.status = "unbounded" if status == "unbounded" else "infeasible"
                warn(result.status, f"chemostat at D = {D}: {status}; no steady state")
            else:
                result.status = "optimal"
                result.X_star = v
                levels = {1.0: v * (1 - OPT_SLACK), **{f: f * v for f in fractions}}

        if result.status == "optimal":
            opt = result.optimum
            raw, top_solutions = {}, []
            for f, level in levels.items():
                raw[f], sols = bg.ranges(level)
                if f == 1.0:
                    top_solutions = sols
            result.a_ranges = _hull(raw)
            rep = next((p for mid, sense, p in top_solutions if mid == bg.ids[0] and sense == "max"), None)
            if rep is None and top_solutions:
                rep = top_solutions[0][2]
            if rep is not None:
                result.X_opt = {k: rep[bg.X[k].name] for k in bg.ids}
                result.fluxes = _net_fluxes(bg.model, rep)
                result.cap_activity = bg.cap_activity(rep)
            seen = set()
            for _, _, p in top_solutions:
                for mid, a, hits in bg.relabelling(p, x_min):
                    if mid in seen:
                        continue
                    seen.add(mid)
                    shown = ", ".join(f"{rid} {v:.4g}" + (f" (bound x X_m {b:.4g})" if b is not None else "")
                                      for rid, v, b in hits[:5])
                    where = "at its eq.-9 bound" if bg.eq9 else "(no eq.-9 rows)"
                    warn("relabelling", f"{mid} at a = {a:.3g} < {x_min:g} carries flux {where} in an optimal "
                         f"solution ({len(hits)} reactions; {shown}): its biomass is a label, not a capacity", mid,
                         {"a": a, "reactions": hits})
            if result.best_solo is not None:
                best, bval = result.best_solo
                result.delta_over_best_solo = opt - bval
                top = result.a_ranges.get(1.0, {})
                big = [k for k, r in top.items() if r is None or r[1] >= x_min]
                if result.delta_over_best_solo <= corner_tol * max(abs(bval), 1e-12):
                    warn("corner", f"the optimum {opt:.9g} beats {best} alone ({bval:.9g}) by "
                         f"{result.delta_over_best_solo:.3g} (<= {corner_tol:g} relative): the optimum is the "
                         f"single-member corner {best} alone")
                elif len(big) == 1:
                    others = {k: top[k] for k in bg.ids if k != big[0]}
                    warn("corner", f"at the optimum every member but {big[0]} is below a = {x_min:g} ({others}): "
                         f"a single-member corner; its gain of {result.delta_over_best_solo:.3g} over {best} "
                         "alone is carried by the sliver")
        if bg.statuses:
            warn("solver_status", f"{len(bg.statuses)} LPs ended {sorted(set(bg.statuses))} and were treated as infeasible")

    for mkey, spec in (monoculture_rates or {}).items():
        k = _member(comm, mkey).id
        lo, hi, ref = _monoculture_ci(spec, monoculture_rtol)
        s = result.solo.get(k)
        if s is None or not (lo <= s <= hi):
            warn("solo_outside_ci", f"{k}: solo {'infeasible' if s is None else f'{s:.6g}'} outside its monoculture "
                 f"{'' if ref is None else f'{float(ref):.6g} '}[{lo:.6g}, {hi:.6g}]: the model does not reproduce this "
                 "monoculture under these constraints", k)

    result.warnings = warns
    result.seconds = round(time.time() - t0, 2)

    def resolve(kind, composition, want_fluxes):
        many = isinstance(composition, (list, tuple))
        comps = list(composition) if many else [composition]
        with _BalancedGrowth(comm, **settings) as bg2:
            if kind == "mu_at":
                out = [bg2.mu_at(c, want_fluxes) for c in comps]
            else:
                out = [bg2.X_at(c, want_fluxes) for c in comps]
        return out if many else out[0]

    result._resolve = resolve
    return result


# =========================================================================== kappa calibration
def calibrate_kappa(comm, per_member=False, *, caps=None, maintenance=None, scale_bounds="all", mode="batch",
                    D=None, supplies=None, X0=1.0, feasibility_tol=1e-9, solver_threads=None, slack=1e-7
                    ) -> KappaCalibration:
    """kappa_sat,m = min sum_r p_r |v_r| / v_bio,m at member m's solo optimum: the smallest
    kinetic coefficient at which member m ALONE still reaches the growth it reaches without kinetic
    rows. One solo LP and one LP per member (min of its *_commKin terms, weights p_r, at
    (1 - slack) x the solo optimum). Composition-free: no declared abundance enters.

    The solo optimum is steadycom's (same arguments, same set-up): member m alone (X_m = X0, the
    others 0) in balanced growth with eq. 9, caps and maintenance on X_m, forced member bounds
    released, the package's kinetic rows lifted. The term set and weights are those of the live
    <member>_commKin row, else what add_commkinetics would write (member_kinetic_reactions,
    comm.rxnProbs).

    Returns a KappaCalibration, a dict {member: kappa}: each member's own kappa_sat,m with
    per_member=True (for steadycom(kappa=...)); with per_member=False the shared value
    max_m kappa_sat,m for every member (the smallest single kappa that forbids no member's solo
    optimum; for comm.add_commkinetics). .kappa_sat holds the per-member values either way. A member
    that does not grow alone has no kappa_sat (None) and gets the shared value.

    Warns (CompositionWarning "kappa_default") when a kappa_sat exceeds the kappa of the installed
    *_commKin rows (comm.kinCoef) or the constructor default kinetic_coeff: such a kappa forbids
    that member's measured monoculture (P. fluorescens on acetate: 751.0 > 750). A kappa below
    max kappa_sat makes the member slower alone than its monoculture, which is a fitted input, not
    a prediction."""
    warns = []

    def warn(code, message, member=None):
        w = CompositionWarning(code, message, member)
        warns.append(w)
        logger.warning("calibrate_kappa: %s", w)

    sat, solo = {}, {}
    with _BalancedGrowth(comm, caps=caps, maintenance=maintenance, kappa=None, scale_bounds=scale_bounds, mode=mode,
                         D=D, supplies=supplies, X0=X0, feasibility_tol=feasibility_tol,
                         solver_threads=solver_threads) as bg:
        for k in bg.ids:
            s = bg.solo(k)
            solo[k] = s
            if s is None or s <= 1e-12:
                sat[k] = None
                warn("no_solo_growth", f"{k} does not grow alone (solo {s}): it has no kappa_sat", k)
                continue
            sat[k] = bg.kinetic_ratio(k, s, slack)
            if sat[k] is None:
                warn("solver_status", f"{k}: the min-flux LP at its solo optimum failed", k)
    finite = [v for v in sat.values() if v is not None]
    shared = max(finite) if finite else None
    if per_member:
        values = {k: (v if v is not None else shared) for k, v in sat.items()}
    else:
        values = {k: shared for k in sat}
    model = comm.util.model
    installed = [c for c in model.constraints if c.name.endswith("_commKin") and c.ub is not None and c.ub < LIFT]
    kin_coef = getattr(comm, "kinCoef", None)
    default = _default_kinetic_coeff()
    if shared is not None:
        worst = max((k for k in sat if sat[k] is not None), key=lambda k: sat[k])
        if installed and kin_coef is not None and kin_coef < shared:
            warn("kappa_default", f"the installed kinetic rows use kappa {kin_coef:g}, below {worst}'s kappa_sat "
                 f"{shared:.6g}: {worst} cannot reach its solo optimum", worst)
        if default is not None and default < shared:
            warn("kappa_default", f"the constructor default kinetic_coeff={default:g} is below {worst}'s kappa_sat "
                 f"{shared:.6g}: at that default {worst} cannot reach its solo optimum", worst)
    return KappaCalibration(values, kappa_sat=sat, shared=shared, solo=solo, per_member=per_member, warnings=warns)


def kappa_sat_coupled(comm, abundances, *, slack=1e-7, feasibility_tol=1e-9, solver_threads=None) -> float:
    """kappa_sat(a) of the COUPLED form at the declared composition a: the smallest shared kappa at
    which the community with abundances a in bio1 and the member drains closed still reaches
    (1 - slack) x its kinetic-row-free growth mu_off(a). One LP: bio1 fixed at (1 - slack) mu_off,
    so v_bio,m = a_m bio1 is a constant and sum_r p_r |v_r| <= kappa a_m bio1 is linear in kappa;
    minimise kappa. abundances: {member: a} (normalised). Forced bounds stay as configured (this is
    the coupled form as built); the package's kinetic rows are lifted while it is computed.

    The round trip, which is why this number is not a prediction: with the kinetic rows installed at
    kappa_sat(a) (comm.add_commkinetics(kappa)) and the drains open, predict_abundances(
    regularization=False) returns a again (SynCom: a = 0.3, 0.5, 0.658, 0.8 come back as 0.3000001,
    0.5000002, 0.6580002, 0.8000001). A kappa calibrated on the coupled form at a declared
    composition is that composition fed back through kappa; agreement with it is not evidence.

    Raises ValueError when the coupled form does not grow at a. Restores the abundances, drains,
    rows and objective."""
    model = comm.util.model
    comp = _normalize_composition(comm, abundances)
    original = _abundance_state(comm)
    with ExitStack() as stack:
        stack.callback(_restore_abundances, comm, original)     # bio1 is written untracked (_write_bio1)
        stack.enter_context(model)
        stack.enter_context(_solver_settings(model, feasibility_tol, solver_threads))
        stack.enter_context(lifted_kinetic_rows(comm))
        _write_abundances(comm, comp)
        close_member_drains(comm)
        bio1 = model.reactions.get_by_id(comm.primary_biomass.id)
        model.objective = model.problem.Objective(bio1.flux_expression, direction="max")
        mu_off, status = _solve(model)
        if mu_off is None or mu_off <= 0:
            raise ValueError(f"the coupled form does not grow at {comp} ({status}, {mu_off})")
        target = mu_off * (1 - slack)
        bio1.bounds = (target, target)
        t = model.problem.Variable("kappa_sat_t", lb=0.0)
        rows = {m.id: model.problem.Constraint(Zero, ub=0.0, name=f"ksat_{m.id}") for m in comm.members}
        model.add_cons_vars([t, *rows.values()])
        model.solver.update()
        for m in comm.members:
            coefs = dict(_kinetic_terms(comm, m))
            coefs[t] = -comp[m.id] * target
            rows[m.id].set_linear_coefficients(coefs)
        model.objective = model.problem.Objective(t, direction="min")
        kappa, status = _solve(model)
        if kappa is None:
            zero = [k for k, v in comp.items() if v == 0]
            raise ValueError(f"no kappa lets the coupled form at {comp} reach (1 - {slack:g}) x its growth "
                             f"without kinetic rows (the kappa LP ended {status})"
                             + (f"; members at a_m = 0 ({zero}) get sum |v| <= kappa x 0, so any flux forced on "
                                "them (a constant floor) makes every kappa infeasible" if zero else ""))
        for m in comm.members:
            v = m.primary_biomass.flux
            if abs(v - comp[m.id] * target) > 1e-6 * max(target, 1e-9):
                logger.warning("kappa_sat_coupled: %s grows %.9g, not a_m x bio1 = %.9g (is its drain closed?)",
                               m.id, v, comp[m.id] * target)
    return kappa


# =========================================================================== composition profile
def _grid_compositions(comm, grid):
    ids = [m.id for m in comm.members]
    if grid is None:
        if len(ids) != 2:
            raise ValueError("composition_profile(objective='coupled') with more than two members needs grid=: "
                             "a list of {member: a} compositions")
        grid = [i / 100 for i in range(101)]
    out = []
    for g in grid:
        if isinstance(g, Mapping):
            out.append(_normalize_composition(comm, g))
        else:
            if len(ids) != 2:
                raise ValueError("a numeric grid gives the first member's a and needs exactly two members")
            a = float(g)
            if not 0 <= a <= 1:
                raise ValueError(f"grid value {a} outside [0, 1]")
            out.append({ids[0]: a, ids[1]: 1.0 - a})
    return out


def _neighbour(points, a, step):
    """The grid point next to `a` in direction step (-1, +1), or None at the end of the grid."""
    i = points.index(a) + step
    return points[i] if 0 <= i < len(points) else None


def _edge(mu_of, ids, a_in, a_out, level, tol):
    """Bisect between a_in (mu >= level) and a_out (mu < level, or None: no neighbour) on the first
    member's a; returns the last a that passes."""
    if a_out is None:
        return a_in
    while abs(a_out - a_in) > tol:
        mid = 0.5 * (a_in + a_out)
        v = mu_of({ids[0]: mid, ids[1]: 1.0 - mid})
        if v is not None and v >= level:
            a_in = mid
        else:
            a_out = mid
    return a_in


def _intervals(points, values, threshold):
    """[(lo, hi)] runs of consecutive grid points with value >= threshold."""
    runs, start, last = [], None, None
    for p, v in zip(points, values):
        if v is not None and v >= threshold:
            if start is None:
                start = p
            last = p
        elif start is not None:
            runs.append((start, last))
            start = None
    if start is not None:
        runs.append((start, last))
    return runs


def _member_biomass_ids(comm):
    return [getattr(m.biomass_cpd, "id", m.biomass_cpd) for m in comm.members]


def _abundance_state(comm):
    """Everything a composition change touches: the member-biomass coefficients of bio1, the
    members' abundance attributes, comm.abundances and the bounds of every abundance-basis cap /
    maintenance row (kept as they are, stale or not)."""
    model = comm.util.model
    bio1 = model.reactions.get_by_id(comm.primary_biomass.id)
    ids = set(_member_biomass_ids(comm))
    coefs = {met.id: float(c) for met, c in bio1.metabolites.items() if met.id in ids}
    rows = {rec.name: (model.constraints[rec.name].lb, model.constraints[rec.name].ub)
            for rec in member_caps(comm) if rec.basis == "abundance"}
    return (coefs, {m.id: m.abundance for m in comm.members}, dict(getattr(comm, "abundances", None) or {}), rows)


def _write_bio1(comm, coefs):
    """Set bio1's member-biomass coefficients ({metabolite id: coefficient}, 0 removes) WITHOUT
    cobra's context bookkeeping (reversibly=False). Inside `with model:` cobra's undo of
    add_metabolites(combine=False) looks every metabolite up among the old coefficients by object:
    it fails for a member whose biomass_cpd is not the very object in bio1 (the SynCom build) and
    for one that an earlier zero abundance removed. Callers restore bio1 explicitly instead."""
    model = comm.util.model
    bio1 = model.reactions.get_by_id(comm.primary_biomass.id)
    bio1.add_metabolites({model.metabolites.get_by_id(mid): c for mid, c in coefs.items()},
                         combine=False, reversibly=False)


def _write_abundances(comm, composition):
    """MSCommunity.set_abundance + commmembers.refresh_caps for a composition, with bio1 written by
    _write_bio1 (no context bookkeeping; restore with _restore_abundances)."""
    comp = _normalize_composition(comm, composition)
    _write_bio1(comm, {mid: -comp[m.id] for m, mid in zip(comm.members, _member_biomass_ids(comm))})
    for m in comm.members:
        m.abundance = comp[m.id]
    comm.abundances = dict(comp)
    refresh_caps(comm)


def _restore_abundances(comm, state):
    """Put back _abundance_state."""
    coefs, members, abundances, rows = state
    _write_bio1(comm, {mid: coefs.get(mid, 0.0) for mid in _member_biomass_ids(comm)})
    for m in comm.members:
        m.abundance = members[m.id]
    comm.abundances = abundances
    model = comm.util.model
    for name, (lb, ub) in rows.items():
        if name in model.constraints:
            _set_row_bounds(model.constraints[name], lb, ub)


def _min_member_flux(comm, member, composition):
    """ZeroMemberFlux of `member` under the current model (objective level already held)."""
    model = comm.util.model
    rxns = member_reactions(comm, member, include_biomass=False)
    model.objective = model.problem.Objective(Zero, direction="min")
    model.objective.set_linear_coefficients({v: 1.0 for r in rxns for v in (r.forward_variable, r.reverse_variable)})
    total, _ = _solve(model)
    if total is None:
        return None
    primal = model.solver.primal_values
    fluxes = {}
    for r in rxns:
        v = primal[r.forward_variable.name] - primal[r.reverse_variable.name]
        if abs(v) > 1e-9:
            fluxes[r.id] = v
    return ZeroMemberFlux(member=member.id, composition=dict(composition), total=total, fluxes=fluxes)


def composition_profile(comm, *, objective="free", fractions=(0.999, 0.99), grid=None, refine=True,
                        zero_tol=1e-9, feasibility_tol=1e-9, solver_threads=None, edge_tol=1e-6) -> ProfileResult:
    """How sharply the community's optimum fixes its composition: the report every composition
    answer should carry (predict_abundances' included). The model is used as configured (kinetic
    rows, caps, forced bounds and exchanges stay); everything changed for the profile is restored.

    objective="coupled": mu_com(a), the optimum of bio1 with abundances a written into bio1
        (set_member_abundances, which also refreshes abundance-basis caps) and every member drain
        closed, on a grid: two members, the first member's a on `grid` (default 0, 0.01, ..., 1;
        floats) and, with refine=True, a golden-section refinement of the argmax between its grid
        neighbours and a bisection of each set edge to edge_tol; more members, `grid` = a list of
        {member: a} (sets are then the grid points that pass). In this form every member grows at
        mu_com by construction (v_bio,m = a_m bio1), so the curve says which declared compositions
        the stoichiometry can feed, not which one the community picks.
    objective="free": with the drains open (0, 1000) and bio1 shut, the objective sum_m v_bio,m
        (bio1 replaced by the sum, as predict_abundances(regularization=False)); the range of
        a_m = v_bio,m / sum v_bio at each fraction of the optimum (sum v_bio fixed at that level;
        each range widened to the hull of the larger fractions').

    Both report the optimum, the composition sets at 1.0 (= (1 - 1e-6) x the optimum) and each
    fraction, and the flux carried by members with a_m = 0: the LEAST total flux (sum |v|) such a
    member must carry at the optimum, with its reactions. A member of zero abundance that must
    carry flux is relabelling: its work is credited to another member's biomass (toy hydrolase:
    1.0 mmol T per gDW per h without caps; 0 once a cap q a_m or a kappa row ties it to its biomass)."""
    if objective not in ("coupled", "free"):
        raise ValueError(f"objective must be 'coupled' or 'free', not {objective!r}")
    t0 = time.time()
    fractions = tuple(sorted({float(f) for f in fractions if 0 < float(f) < 1}, reverse=True))
    model = comm.util.model
    ids = [m.id for m in comm.members]
    warns = []
    result = ProfileResult(objective=objective, status="infeasible")
    original = _abundance_state(comm)
    with ExitStack() as stack:
        stack.callback(_restore_abundances, comm, original)     # bio1 is written untracked (_write_bio1)
        stack.enter_context(model)
        stack.enter_context(_solver_settings(model, feasibility_tol, solver_threads))
        bio1 = model.reactions.get_by_id(comm.primary_biomass.id)
        if objective == "coupled":
            close_member_drains(comm)
            comps = _grid_compositions(comm, grid)
            curve = []

            def mu_of(comp):
                _write_abundances(comm, comp)
                model.objective = model.problem.Objective(bio1.flux_expression, direction="max")
                return _solve(model)[0]

            for comp in comps:
                curve.append((comp, mu_of(comp)))
            result.curve = curve
            values = [v for _, v in curve if v is not None]
            if values:
                result.status = "optimal"
                i_best = max(range(len(curve)), key=lambda i: -math.inf if curve[i][1] is None else curve[i][1])
                best_comp, best_mu = curve[i_best]
                if refine and len(ids) == 2 and len(comps) > 2:
                    pts = [c[ids[0]] for c in comps]
                    lo = pts[i_best - 1] if i_best > 0 else pts[i_best]
                    hi = pts[i_best + 1] if i_best + 1 < len(pts) else pts[i_best]
                    lo, hi = min(lo, hi), max(lo, hi)
                    f = lambda a: (lambda v: -math.inf if v is None else v)(mu_of({ids[0]: a, ids[1]: 1 - a}))  # noqa: E731
                    g = (math.sqrt(5) - 1) / 2
                    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
                    fc, fd = f(c), f(d)
                    for _ in range(60):
                        if hi - lo < 1e-9:
                            break
                        if fc >= fd:
                            hi, d, fd = d, c, fc
                            c = hi - g * (hi - lo)
                            fc = f(c)
                        else:
                            lo, c, fc = c, d, fd
                            d = lo + g * (hi - lo)
                            fd = f(d)
                    a_ref = 0.5 * (lo + hi)
                    mu_ref = f(a_ref)
                    if mu_ref > best_mu:
                        best_comp, best_mu = {ids[0]: a_ref, ids[1]: 1 - a_ref}, mu_ref
                        curve.append((dict(best_comp), best_mu))
                if len(ids) == 2:
                    curve.sort(key=lambda cv: cv[0][ids[0]])
                result.curve = curve
                result.optimum, result.composition = best_mu, dict(best_comp)
                levels = {1.0: best_mu * (1 - OPT_SLACK), **{fr: fr * best_mu for fr in fractions}}
                for fr, level in levels.items():
                    if len(ids) != 2:
                        result.sets[fr] = [c for c, v in curve if v is not None and v >= level]
                        continue
                    pts = [c[ids[0]] for c, _ in curve]
                    runs = _intervals(pts, [v for _, v in curve], level)
                    if refine:
                        runs = [(_edge(mu_of, ids, a_lo, _neighbour(pts, a_lo, -1), level, edge_tol),
                                 _edge(mu_of, ids, a_hi, _neighbour(pts, a_hi, +1), level, edge_tol))
                                for a_lo, a_hi in runs]
                    result.sets[fr] = runs
                for comp, v in curve:
                    zero = [k for k in ids if comp[k] <= zero_tol]
                    if v is None or not zero:
                        continue
                    _write_abundances(comm, comp)
                    for k in zero:
                        with model:
                            bio1.lower_bound = v * (1 - OPT_SLACK)
                            z = _min_member_flux(comm, comm.members.get_by_id(k), comp)
                        if z is not None:
                            result.zero_member_flux.append(z)
        else:
            if any(rec.basis == "abundance" and rec.kind == "uptake" for rec in member_caps(comm)):
                warns.append(CompositionWarning(
                    "caps_on_declared_a", "abundance-basis caps (q a_m) use the DECLARED a_m, not the free "
                    "composition v_bio,m / sum v_bio; use steadycom(caps=...) to tie caps to member biomass"))
            bio1.bounds = (0.0, 0.0)
            open_member_drains(comm, (0.0, DRAIN_UB))
            total = add([m.primary_biomass.flux_expression for m in comm.members])
            model.objective = model.problem.Objective(total, direction="max")
            T, status = _solve(model)
            if T is not None and T > 0:
                result.status, result.optimum = "optimal", T
                row = model.problem.Constraint(total, lb=0.0, name="cp_total")
                model.add_cons_vars([row])
                levels = {1.0: T * (1 - OPT_SLACK), **{fr: fr * T for fr in fractions}}
                raw, rep = {}, None
                for fr, level in levels.items():
                    _set_row_bounds(row, level, level)
                    raw[fr] = {}
                    for m in comm.members:
                        vals = []
                        for sense in ("min", "max"):
                            model.objective = model.problem.Objective(m.primary_biomass.flux_expression, direction=sense)
                            v, _ = _solve(model)
                            vals.append(v)
                            if fr == 1.0 and sense == "max" and m.id == ids[0] and v is not None:
                                rep = dict(model.solver.primal_values)
                        raw[fr][m.id] = None if None in vals else (vals[0] / level, vals[1] / level)
                result.sets = _hull(raw)
                if rep is not None:
                    vb = {m.id: rep[m.primary_biomass.forward_variable.name] - rep[m.primary_biomass.reverse_variable.name]
                          for m in comm.members}
                    s = sum(vb.values())
                    result.composition = {k: v / s for k, v in vb.items()}
                _set_row_bounds(row, levels[1.0], levels[1.0])
                for m in comm.members:
                    r = result.sets[1.0].get(m.id)
                    if r is None or r[0] > zero_tol:
                        continue
                    with model:
                        m.primary_biomass.bounds = (0.0, 0.0)
                        z = _min_member_flux(comm, m, {m.id: 0.0})
                    if z is not None:
                        result.zero_member_flux.append(z)
            elif T is not None:
                result.status = "optimal"
                result.optimum = T
                warns.append(CompositionWarning("infeasible", "the free optimum is 0: no member grows"))
            else:
                warns.append(CompositionWarning("infeasible", f"the composition-free LP ended {status}"))
    for w in warns:
        logger.warning("composition_profile: %s", w)
    result.warnings = warns
    result.seconds = round(time.time() - t0, 2)
    return result
