# -*- coding: utf-8 -*-
"""MICOM's cooperative tradeoff on an MSCommunity model (MSCommunity.micom delegates here).

MICOM (Diener, Gibbons & Resendis-Antonio 2020, mSystems 5:e00606-19; micom 0.37.1
community.py, problems.cooperative_tradeoff / regularize_l2_norm, solution.crossover) builds
the community from taxon models whose reactions keep their own bounds, in mmol per gDW of
THAT taxon per h, with the abundance a_i as the coefficient that carries each taxon's
exchanges into the shared medium. Its community objective is sum_i a_i mu_i. The cooperative
tradeoff then solves

    stage 1   F = max sum_i a_i mu_i                     (with mu_i >= min_growth_i)
    stage 2   min sum_i mu_i^2   s.t.   f F <= sum_i a_i mu_i <= F
    crossover (only when the stage-2 QP is not optimal) fix mu_i <= its QP value, max sum_i a_i mu_i
    pFBA      (optional) mu_i >= (1 - tol) mu_i* - tol, min sum |v| over every reaction

Translation to MSCommunity's frame, where every flux is per gDW of COMMUNITY (v_r = a_m q_r)
and the members share one e0 pool (commmembers):

    taxon units      every member reaction's bounds (commmembers.reactions_by_member: its own
                     compartments, its transporters to e0, its biomass reaction and other
                     boundary reactions on its own metabolites) are multiplied by a_m. The bounds
                     are read as the member model built them, per gDW of member; a bound a caller
                     already wrote in community units would be scaled twice (write it as a
                     commmembers row instead: rows are never scaled).
    member biomass   bio1 shut (0, 0), every member biomass drain opened to (0, 1000) after the
                     scaling, so v_bio_m = a_m mu_m leaves through the member's own drain and
                     sum_i a_i mu_i = sum_m v_bio_m.
    stage 2          sum_m (v_bio_m / a_m)^2 on the member biomass forward variables, with the row
                     f F <= sum_m v_bio_m <= F (exact, no relaxation).
    min_growth       v_bio_m >= a_m min_growth_m in both stages (a lower bound on bio_m).

Not reproduced: MICOM's per-taxon internal exchange reactions, whose import bound it replaces by
max_exchange (default 100 mmol/gDW_taxon/h). MSCommunity has no such reactions; a member's uptake
is bounded by its scaled transporters and by any commmembers.cap_member_uptake rows. On the five
CommPhitting/SynCom builds those rows changed no rate by more than 3.1e-6 1/h
(3H11_R12_SynCom/.../abundance/micom/README.md).

Rows are left as they are: the community-kinetics rows *_commKin (sum |v| <= kappa v_bio_m,
which MICOM does not have; homogeneous in the member's fluxes, so in taxon units they read
sum |q| <= kappa mu_m) and the commmembers caps / maintenance rows (already per gDW of
community on basis "abundance"). Lift the kinetic rows for MICOM proper with
`with commmembers.lifted_kinetic_rows(comm): comm.micom(...)`. A row that forces bio1 (a
min_comm_growth floor left by regularization, a comm_tradeoff row left by an old micom) makes
stage 1 infeasible once bio1 is shut; the result then carries that status.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import isfinite, isnan
from typing import Dict, List, Optional

from optlang.symbolics import Zero, add

from mscommunity import commmembers as _cm
from mscommunity.commcycles import reset_start

logger = logging.getLogger(__name__)

TRADEOFF_ROW = "micom_community_growth"
DRAIN_OPEN = (0.0, 1000.0)
OPTIMAL = "optimal"

__all__ = ["MicomResult", "cooperative_tradeoff", "qp_backend"]


# --------------------------------------------------------------------------- result
@dataclass
class MicomResult:
    """One medium's cooperative tradeoff.

    medium               the medium applied (None: the community's medium as it was)
    tradeoff             f, the fraction of the stage-1 optimum kept in stage 2
    abundances           {member: a_m} used (normalised to sum 1). An INPUT, never changed here
    status               "optimal" when stage 2 (or its crossover) solved; otherwise the failing
                         stage's solver status, and the fields below are None
    community_growth_max F = max sum_m a_m mu_m (= sum_m v_bio_m), stage 1, 1/h
    community_growth     sum_m a_m mu_m at the stage-2 point, f F <= it <= F
    growth_rates         {member: mu_m = v_bio_m / a_m} at the stage-2 point, 1/h; None where a_m = 0
    objective            the stage-2 objective sum_m mu_m^2
    solution             cobra Solution: the stage-2 (or crossover) point, or with pfba=True the
                         pFBA solution, whose member biomass fluxes may sit up to the solver's
                         feasibility tolerance (relative) below a_m mu_m
    crossover            True when the stage-2 QP was not optimal and MICOM's crossover LP ran
    stage_status         {"stage 1": status, "stage 2": status, "crossover": ..., "pfba": ...}"""
    medium: Optional[dict]
    tradeoff: float
    abundances: Dict[str, float]
    status: str
    community_growth_max: Optional[float] = None
    community_growth: Optional[float] = None
    growth_rates: Optional[Dict[str, Optional[float]]] = None
    objective: Optional[float] = None
    solution: Optional[object] = None
    crossover: bool = False
    stage_status: Dict[str, str] = field(default_factory=dict)

    @property
    def fluxes(self):
        """solution.fluxes (None without a solution), for callers of the old micom()."""
        return None if self.solution is None else self.solution.fluxes

    @property
    def l2_rates(self) -> Optional[Dict[str, float]]:
        """{member: f F a_m / sum a^2}: the stage-2 rates when no member constraint binds."""
        if self.community_growth_max is None:
            return None
        s2 = sum(a * a for a in self.abundances.values())
        return {k: self.tradeoff * self.community_growth_max * a / s2 for k, a in self.abundances.items()}

    @property
    def l2_deviation(self) -> Optional[float]:
        """max_m |mu_m - f F a_m / sum a^2| over members with a_m > 0: ~0 unless a member
        constraint (a bound, a cap, a kinetic row, the medium) binds at the stage-2 point."""
        if self.growth_rates is None or self.community_growth_max is None:
            return None
        l2 = self.l2_rates
        return max((abs(mu - l2[k]) for k, mu in self.growth_rates.items() if mu is not None), default=0.0)


# --------------------------------------------------------------------------- QP backend
@contextmanager
def qp_backend(comm):
    """Run the block on a QP-capable solver: the incumbent when it already is one, else the best
    available backend (mscommsim._pick_qp_backend), swapped back on exit. RuntimeError, before
    anything is changed, when no QP backend is installed. On exit the incoming linear objective
    (captured as reaction coefficients, interface-independent) is put back while the QP backend
    is still live -- a quadratic objective cannot be cloned into an LP-only interface, so
    restoring the solver first would raise -- and then the solver is swapped back."""
    from mscommunity.mscommsim import _QP_CAPABLE, _pick_qp_backend
    model = comm.util.model
    ogSolver = model.solver.interface.__name__.rsplit(".", 1)[-1].replace("_interface", "")
    qpSolver = _pick_qp_backend(current=ogSolver)
    if qpSolver is None:
        raise RuntimeError("micom() minimizes a quadratic objective, but no QP-capable solver "
                           f"is available (any of {sorted(_QP_CAPABLE)}); install gurobi, cplex, "
                           "or osqp (the 'hybrid' HiGHS+OSQP path).")
    swapped = qpSolver != ogSolver
    ogObjCoefs = {rxn.id: rxn.objective_coefficient for rxn in model.reactions if rxn.objective_coefficient}
    ogObjDir = model.objective.direction
    if swapped:
        model.solver = qpSolver
    try:
        yield qpSolver
    finally:
        ogObjExpr = add([coef * model.reactions.get_by_id(rxnID).flux_expression
                         for rxnID, coef in ogObjCoefs.items()]) if ogObjCoefs else Zero
        model.objective = model.problem.Objective(ogObjExpr, direction=ogObjDir)
        if swapped:
            model.solver = ogSolver


# --------------------------------------------------------------------------- inputs
def _composition(comm, abundances):
    """{member id: a_m}, normalised to sum 1: the community's current abundances, or `abundances`
    ({id: a} or {id: {"abundance": a}}) for every member."""
    ids = [m.id for m in comm.members]
    if abundances is None:
        raw = {m.id: float(m.abundance) for m in comm.members}
    else:
        raw = {k: float(v["abundance"] if isinstance(v, Mapping) else v) for k, v in abundances.items()}
        if set(raw) != set(ids):
            raise ValueError(f"give an abundance for every member: missing {sorted(set(ids) - set(raw))}, "
                             f"unknown {sorted(set(raw) - set(ids))}")
    bad = {k: v for k, v in raw.items() if not isfinite(v) or v < 0}
    if bad:
        raise ValueError(f"abundances must be finite and >= 0: {bad}")
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("the abundances sum to 0")
    return {k: raw[k] / total for k in ids}


def _floors(min_growth, comm):
    """{member id: minimum specific growth rate, 1/h} from a number or {member id: rate}."""
    if isinstance(min_growth, Mapping):
        unknown = set(min_growth) - {m.id for m in comm.members}
        if unknown:
            raise ValueError(f"min_growth names unknown members {sorted(unknown)}")
        out = {m.id: float(min_growth.get(m.id, 0.0)) for m in comm.members}
    else:
        out = {m.id: float(min_growth) for m in comm.members}
    bad = {k: v for k, v in out.items() if not isfinite(v) or v < 0}
    if bad:
        raise ValueError(f"min_growth must be finite and >= 0: {bad}")
    return out


def _media(media):
    if media is None:
        return [None]
    if isinstance(media, Mapping) or not isinstance(media, (list, tuple)):
        return [media]
    return list(media)


# --------------------------------------------------------------------------- state
def _restore_bounds(model, bounds):
    for rid, bnds in bounds.items():
        rxn = model.reactions.get_by_id(rid)
        if tuple(rxn.bounds) != tuple(bnds):
            rxn.bounds = bnds


def _apply_abundances(comm, a):
    """Write `a` into the community (bio1 coefficients, member abundances and the caps on basis
    "abundance") when it differs from the current composition. Returns the restore callable."""
    if all(abs(float(m.abundance) - a[m.id]) <= 1e-15 for m in comm.members):
        return lambda: None
    old_a = {m.id: m.abundance for m in comm.members}
    old_attr = getattr(comm, "abundances", None)
    old_bio1 = dict(comm.primary_biomass.metabolites)
    _cm.set_member_abundances(comm, a)

    def restore():
        for m in comm.members:
            m.abundance = old_a[m.id]
        comm.abundances = old_attr
        comm.primary_biomass.add_metabolites(old_bio1, combine=False)
        _cm.refresh_caps(comm)
    return restore


def _l2_term(rxn, a):
    """(v_bio / a)^2 on the forward variable alone when the reverse one is fixed at 0 (MICOM keeps
    only the free variables of a taxon objective), else on the net flux."""
    rev = rxn.reverse_variable
    if rev.ub is not None and rev.ub - (rev.lb or 0.0) <= 1e-6:
        return (1.0 / (a * a)) * rxn.forward_variable ** 2
    return ((1.0 / (a * a)) * rxn.flux_expression ** 2).expand()


def _flux(rxn):
    return float(rxn.forward_variable.primal) - float(rxn.reverse_variable.primal)


def _solve(model):
    """slim_optimize; one cold-basis retry (commcycles.reset_start) when it is not optimal,
    as MICOM's optimize_with_retry resets the solver once."""
    value = model.slim_optimize(error_value=float("nan"))
    if model.solver.status != OPTIMAL:
        reset_start(model)
        value = model.slim_optimize(error_value=float("nan"))
    return value, model.solver.status


def _solution(model):
    from cobra.core.solution import get_solution
    try:
        return get_solution(model, reactions=model.reactions, metabolites=model.metabolites)
    except Exception as error:   # duals are not available from every QP backend
        logger.debug("get_solution with duals failed (%s); reading the primal values only", error)
        from cobra import Solution
        from pandas import Series
        fluxes = Series({r.id: _flux(r) for r in model.reactions}, name="fluxes")
        return Solution(model.solver.objective.value, model.solver.status, fluxes)


# --------------------------------------------------------------------------- the tradeoff
def cooperative_tradeoff(comm, media=None, tradeoff=0.5, *, abundances=None, pfba=False,
                         min_growth=0.0) -> List[MicomResult]:
    """MICOM's cooperative tradeoff on `comm` (see the module docstring), one MicomResult per
    medium. Run it on a QP-capable solver (MSCommunity.micom wraps it in qp_backend). Every
    reaction bound, the composition, the drains and the rows it adds are put back on exit; the
    objective is the caller's (qp_backend restores it)."""
    tradeoff = float(tradeoff)
    if not (0.0 <= tradeoff <= 1.0):
        raise ValueError(f"tradeoff must be in [0, 1], not {tradeoff}")
    a = _composition(comm, abundances)
    floors = _floors(min_growth, comm)
    model = comm.util.model
    bounds0 = {r.id: tuple(r.bounds) for r in model.reactions}
    restore_abundances = _apply_abundances(comm, a)
    try:
        return [_one_medium(comm, medium, tradeoff, a, floors, pfba, bounds0) for medium in _media(media)]
    finally:
        _restore_bounds(model, bounds0)
        restore_abundances()


def _taxon_units(comm, a, floors):
    """Scale every member reaction by a_m, shut bio1, open the member drains, set the floors."""
    owned = _cm.reactions_by_member(comm)
    drains = {m.biomass_drain.id for m in comm.members if m.biomass_drain is not None}
    for m in comm.members:
        am = a[m.id]
        if am == 1.0:
            continue
        for rxn in owned[m.id]:
            if rxn.id in drains:
                continue
            lb, ub = rxn.bounds
            rxn.bounds = (am * lb, am * ub)
    comm.primary_biomass.bounds = (0.0, 0.0)
    _cm.open_member_drains(comm, DRAIN_OPEN)
    for m in comm.members:
        floor = floors[m.id] * a[m.id]
        if floor > 0:
            bio = m.primary_biomass
            if floor > bio.upper_bound:
                raise ValueError(f"min_growth {floors[m.id]} of {m.id} exceeds its biomass bound "
                                 f"{bio.upper_bound / a[m.id]} per gDW of member")
            bio.lower_bound = max(bio.lower_bound, floor)


def _one_medium(comm, medium, f, a, floors, pfba, bounds0):
    model = comm.util.model
    _restore_bounds(model, bounds0)
    if medium is not None:
        comm.util.add_medium(medium)
    _taxon_units(comm, a, floors)
    members = list(comm.members)
    live = [m for m in members if a[m.id] > 0]
    total = add([m.primary_biomass.flux_expression for m in members])
    result = MicomResult(medium=medium, tradeoff=f, abundances=dict(a), status="")

    # stage 1: F = max sum_m v_bio_m
    model.objective = model.problem.Objective(total, direction="max")
    F, status = _solve(model)
    result.stage_status["stage 1"] = status
    if status != OPTIMAL or isnan(F):
        result.status = f"stage 1 {status}"
        logger.warning("micom: the stage-1 LP (max sum of member growth) of %s is %s; no rates for "
                       "this medium. A row forcing bio1 (min_comm_growth, comm_tradeoff) makes it "
                       "infeasible once bio1 is shut.", model.id, status)
        return result
    F = max(float(F), 0.0)
    result.community_growth_max = F

    # stage 2: min sum_m (v_bio_m / a_m)^2 with f F <= sum_m v_bio_m <= F
    row = model.problem.Constraint(total, lb=f * F, ub=F, name=TRADEOFF_ROW)
    model.add_cons_vars([row])
    try:
        model.objective = model.problem.Objective(
            add([_l2_term(m.primary_biomass, a[m.id]) for m in live]), direction="min")
        objective = model.slim_optimize(error_value=float("nan"))
        status = model.solver.status
        result.stage_status["stage 2"] = status
        if status != OPTIMAL:
            ok = _crossover(comm, model, row, live, result)
            if not ok:
                return result
        v = {m.id: _flux(m.primary_biomass) for m in members}
        result.growth_rates = {k: (v[k] / a[k] if a[k] > 0 else None) for k in v}
        result.community_growth = sum(v.values())
        result.objective = (sum(mu * mu for mu in result.growth_rates.values() if mu is not None)
                            if result.crossover else float(objective))
        result.solution = _solution(model)
        result.status = OPTIMAL
        if pfba:
            _pfba(model, live, v, result)
        return result
    finally:
        model.remove_cons_vars([model.constraints[TRADEOFF_ROW]])


def _crossover(comm, model, row, live, result):
    """MICOM's crossover (solution.crossover): each member's growth at most its value at the
    non-optimal QP point, community growth in [0, that total + 1e-6], max the community growth."""
    try:
        v = {m.id: _flux(m.primary_biomass) for m in live}
    except Exception:
        v = None
    if v is None or not all(isfinite(x) for x in v.values()):
        result.status = f"stage 2 {result.stage_status['stage 2']}"
        logger.warning("micom: the stage-2 QP of %s is %s and gives no point to cross over from",
                       model.id, result.stage_status["stage 2"])
        return False
    logger.warning("micom: the stage-2 QP of %s is %s; running MICOM's crossover", model.id,
                   result.stage_status["stage 2"])
    for m in live:
        bio = m.primary_biomass
        bio.upper_bound = max(bio.lower_bound, v[m.id])
    # lb first: the non-optimal point may sit below f F, and optlang refuses ub < lb
    row.lb = 0.0
    row.ub = max(sum(v.values()), 0.0) + 1e-6
    model.objective = model.problem.Objective(row.expression, direction="max")
    _, status = _solve(model)
    result.stage_status["crossover"] = status
    result.crossover = True
    if status != OPTIMAL:
        result.status = f"crossover {status}"
        logger.warning("micom: the crossover LP of %s is %s; no rates for this medium", model.id, status)
        return False
    return True


def _pfba(model, live, v, result):
    """MICOM's pFBA (solution.add_pfba_objective): member growth >= (1 - tol) of its stage-2
    value - tol (per gDW of member), then min the summed flux of every reaction."""
    tol = float(model.solver.configuration.tolerances.feasibility or 1e-6)
    for m in live:
        bio = m.primary_biomass
        a = result.abundances[m.id]
        bio.lower_bound = min(bio.upper_bound, max(0.0, (1.0 - tol) * v[m.id] - tol * a))
    model.objective = model.problem.Objective(Zero, direction="min", sloppy=True)
    model.objective.set_linear_coefficients(
        {var: 1.0 for rxn in model.reactions for var in (rxn.forward_variable, rxn.reverse_variable)})
    _, status = _solve(model)
    result.stage_status["pfba"] = status
    if status == OPTIMAL:
        result.solution = _solution(model)
    else:
        logger.warning("micom: the pFBA LP of %s is %s; returning the stage-2 solution", model.id, status)
