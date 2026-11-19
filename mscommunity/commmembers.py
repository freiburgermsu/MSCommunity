# -*- coding: utf-8 -*-
"""Member bookkeeping for MSCommunity community models: which reactions are whose, what each
member takes from the shared pool, per-member uptake caps and maintenance rows, and member rates.

This module is the one place that answers "which member does this belong to" and "how much of
compound X does member m take up". The balanced-growth (SteadyCom), MICOM, chemostat and
multi-species dynamic FBA additions build on it; they should not re-derive membership.

Layout (build_from_species_models)
    member i's compartments    c<i> (cytosol), p<i> (periplasm, where the member model has one)
    shared pool                e0, one metabolite per compound for every member (<cpd>_e0)
    community compartment      c0: bio1 (consumes a_m of each member's cpd11416_c<i>) and its sink
    member biomass             bio<k> producing cpd11416_c<i>, drained by SK_/DM_cpd11416_c<i>

Membership (one definition, commcycles.member_index)
    A reaction belongs to member i when every non-e0 metabolite it touches sits in a compartment
    of index i. So a member owns its cytosolic and periplasmic reactions, its transporters between
    those and e0, its own biomass reaction, its biomass drain and any other boundary reaction on
    its own metabolites (DM_/SK_ sinks). Not owned by any member: the exchanges and every other
    reaction whose metabolites are all in e0 (member_index 0: the shared pool -- an extracellular
    hydrolysis carrying a member's id suffix lands here, see MemberInfo.extracellular), the
    community biomass bio1 (excluded by identity, so a one-member community does not hand it to its
    member) and any reaction spanning two members (member_index None; MemberTable.unassigned lists
    them, and a well-formed build has none). The member sets are therefore disjoint by construction.

Units
    Fluxes are mmol per gDW of COMMUNITY per h, as everywhere in an MSCommunity model; member
    uptake expressions and member_net_exchange are in those units. Specific rates are per gDW of
    THAT member: q (uptake caps) and A_m (maintenance) in mmol/gDW_member/h, mu_m in 1/h. With
    the member's abundance a_m (gDW member per gDW community, comm.set_abundance normalises them to
    sum to 1) a specific rate r becomes the community-basis flux r * a_m.

Caps and maintenance rows (cap_member_uptake, set_member_maintenance)
    Each adds ONE optlang row to comm.util.model and records it in comm._member_caps
    ({row name: MemberCap}), so member_caps lists them, refresh_caps rewrites them and
    remove_member_caps removes them (and puts back what they changed).
        basis="abundance"  gross uptake <= q * a_m,  ATPM_m >= A_m * a_m;  a_m read when the row is
                           written. comm.set_abundance does NOT update these rows: call
                           refresh_caps(comm) after it, or use set_member_abundances, which does both.
        basis="X"          gross uptake - q * X_m <= 0,  ATPM_m - A_m * X_m >= 0;  X_m an optlang
                           variable the caller owns and has already added to the model (e.g.
                           member_biomass_variables). Nothing to refresh: the row follows X_m.
    Caps sit on the member's transporters (forward / reverse variables), never on the shared EX_
    reaction: a cap on EX_ limits the whole community, not the member. Rows are named
    mcap_<member>_<cpd> and mmaint_<member>; neither ends in _commKin, so the kinetic rows
    (MSCommunity.add_commkinetics / CommKineticPkg, which rebuild only *_commKin) and
    lifted_kinetic_rows leave them alone. Inside `with comm.util.model:` an added row, its registry
    entry and a released maintenance floor are all undone on exit, and so are the row bounds that
    refresh_caps (and therefore set_member_abundances) rewrites: cobra does not track optlang row
    bounds, so refresh_caps registers its own undo step that puts the previous bounds back. (Without
    it the rows kept the context's composition after exit while the abundances and bio1 reverted.)

What downstream code must know
    * X variables: member_biomass_variables(comm) creates (or returns the existing) X_<member id>
      per member, lb 0, no ub, in the model. The caller adds sum_m X_m = 1 (SteadyCom), its
      dilution balance (chemostat) or fixes X_m (dynamic FBA), and removes the variables when done;
      remove_member_caps does not touch them. The prototype rows of a SteadyCom LP are in
      3H11_R12_SynCom/Metabolic_modeling/abund_common.py BalancedGrowth: bio1 shut, member drains
      opened (open_member_drains), v_bio_m - mu * X_m = 0, caps and maintenance on basis="X".
    * set_abundance: rows on basis="abundance" are stale after comm.set_abundance until
      refresh_caps(comm). MSCommunity.set_abundance rewrites bio1 only; bounds the caller tied to the
      composition elsewhere (a fixed EX_ rate) are the caller's to update.
    * Kinetic rows: the *_commKin row of member m is sum_r |v_r| <= kappa * v_bio_m over the
      member's metabolism (commkineticpkg.member_kinetic_reactions = member_reactions(...,
      include_biomass=False) on a well-formed build). It and the caps coexist; with basis="X" the
      kinetic budget is kappa * mu * X_m.
    * Energy audits: MSCommunity.energy_cycles / commcycles.energy_from_nothing release forced
      reaction BOUNDS and lift *_commKin rows, but not a maintenance ROW, which forces ATPM flux
      that a sealed model cannot pay for: the test LP goes infeasible and reports None. Remove the
      maintenance rows (remove_member_caps(comm, kind="maintenance")) or lift them
      (commcycles._ea.lifted_rows(model, [cap.name for cap in member_caps(comm, kind="maintenance")]))
      around an audit.
    * member_rates: with every drain closed, member m's biomass can leave only through bio1, so
      v_bio_m = a_m * v_bio1 and mu_m = mu_com for every member by construction; a member rate that
      differs from mu_com means its drain is open and carrying flux (mu_m = mu_com + v_drain_m / a_m).

Most functions take the MSCommunity. member_reactions, reactions_by_member, member_transporters,
member_uptake_expression, member_net_exchange and lifted_kinetic_rows also take its cobra model
(comm.util.model, or a copy of it) with the member given by its index. A member is given as the
CommunityMember, its id or its index.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import partial
from math import isfinite
from typing import Dict, Optional, Tuple

from cobra.util.context import get_context
from optlang.symbolics import add

from mscommunity import commcycles as _cycles
from mscommunity.commcycles import member_index

logger = logging.getLogger(__name__)

SHARED = "e0"                      # the shared extracellular pool
CAP_PREFIX = "mcap_"               # uptake cap rows: mcap_<member>_<cpd>
MAINT_PREFIX = "mmaint_"           # maintenance rows: mmaint_<member>
ATPM_MSID = "rxn00062"             # ATP + H2O -> ADP + Pi + H+ (ModelSEED ATP maintenance)
BASES = ("abundance", "X")
_REGISTRY = "_member_caps"
_ID_SUFFIX = re.compile(r"_([a-z]+)(\d+)$")

__all__ = [
    "MemberCap", "MemberInfo", "MemberTable",
    "member_reactions", "reactions_by_member", "member_transporters", "member_uptake_expression",
    "member_net_exchange", "member_rates", "open_member_drains", "close_member_drains",
    "restore_member_drains", "lifted_kinetic_rows", "member_biomass_variables", "cap_member_uptake",
    "set_member_maintenance", "member_caps", "refresh_caps", "remove_member_caps",
    "set_member_abundances", "atp_maintenance_reaction", "member_table",
]


# --------------------------------------------------------------------------- records
@dataclass(frozen=True)
class MemberCap:
    """One row added by cap_member_uptake (kind "uptake") or set_member_maintenance (kind
    "maintenance"), as kept in comm._member_caps.

    rate       q (uptake) or A_m (maintenance), mmol per gDW of the member per h
    basis      "abundance" (rate * a_m, a constant refresh_caps rewrites) or "X" (rate * X_m)
    X          the X_m variable's name when basis == "X"
    cpd        the e0 compound capped (without _e0), None for a maintenance row or a cap over
               reactions= with no compound
    reactions  the reactions in the row: the transporters (uptake) or the ATPM reaction
    released   maintenance only: the ATPM reaction's positive lower bound the row replaced (put
               back by remove_member_caps), else None"""
    name: str
    kind: str
    member: str
    cpd: Optional[str]
    rate: float
    basis: str
    X: Optional[str]
    reactions: Tuple[str, ...]
    released: Optional[float] = None


@dataclass(frozen=True)
class MemberInfo:
    """One member, for reporting (member_table).

    biomass / drain     its biomass reaction and biomass drain ids; drain_bounds as they are now
    atp_maintenance     the reaction set_member_maintenance would use, or None
    compartments        its own compartments (c<i>, p<i>, ...)
    n_reactions         len(member_reactions(comm, member))
    transporters        {e0 compound: (reaction ids moving it between the member and e0)}
    extracellular       e0-only reactions whose id carries this member's suffix (_c<i>): an
                        extracellular enzyme of the member's model. NOT member reactions (all their
                        metabolites are in the shared pool); pass them as reactions= to count them"""
    id: str
    index: int
    abundance: float
    biomass: Optional[str]
    drain: Optional[str]
    drain_bounds: Optional[Tuple[float, float]]
    atp_maintenance: Optional[str]
    compartments: Tuple[str, ...]
    n_reactions: int
    transporters: Dict[str, Tuple[str, ...]]
    extracellular: Tuple[str, ...]


@dataclass(frozen=True)
class MemberTable:
    """member_table(comm): every member, plus what no member owns.

    community_biomass   bio1's id
    shared              non-boundary reactions with every metabolite in e0 (no member owns them)
    unassigned          reactions spanning two or more members, other than bio1 (should be empty)
    caps                the rows in comm._member_caps"""
    members: Tuple[MemberInfo, ...]
    community_biomass: Optional[str]
    shared: Tuple[str, ...]
    unassigned: Tuple[str, ...]
    caps: Tuple[MemberCap, ...]

    def __getitem__(self, member_id):
        for info in self.members:
            if info.id == member_id:
                return info
        raise KeyError(member_id)

    def as_dict(self):
        """Plain dicts / lists / tuples (json.dumps-able)."""
        return asdict(self)

    def to_frame(self):
        """pandas DataFrame, one row per member (transporters as a compound count)."""
        from pandas import DataFrame
        rows = []
        for info in self.members:
            row = asdict(info)
            row["compartments"] = ",".join(info.compartments)
            row["transporters"] = len(info.transporters)
            row["extracellular"] = ",".join(info.extracellular)
            rows.append(row)
        return DataFrame(rows).set_index("id")


# --------------------------------------------------------------------------- resolution
def _is_community(comm):
    return hasattr(comm, "members") and hasattr(comm, "util")


def _model(comm):
    """The cobra model of an MSCommunity, or the model itself."""
    return comm.util.model if _is_community(comm) else comm


def _require_community(comm, what):
    if not _is_community(comm):
        raise TypeError(f"{what} needs the MSCommunity (it reads member abundances and keeps its "
                        f"registry there), not a {type(comm).__name__}")


def _member(comm, member):
    """The CommunityMember given as the object, its id or its index."""
    _require_community(comm, "a member given by object or id")
    if isinstance(member, bool):
        raise TypeError(f"not a member: {member!r}")
    if isinstance(member, int):
        for m in comm.members:
            if m.index == member:
                return m
        raise KeyError(f"no member with index {member}; indices {[m.index for m in comm.members]}")
    key = getattr(member, "id", member)
    if key not in comm.members:
        raise KeyError(f"no member {key!r}; members {[m.id for m in comm.members]}")
    return comm.members.get_by_id(key)


def _index(comm, member):
    """The member's compartment index; a bare model takes the index itself."""
    if _is_community(comm):
        return _member(comm, member).index
    if isinstance(member, int) and not isinstance(member, bool) and member > 0:
        return member
    raise TypeError(f"on a bare model give the member as its compartment index (1, 2, ...), not {member!r}")


def _label(comm, idx):
    if _is_community(comm):
        return _member(comm, idx).id
    return f"member {idx}"


def _live(model, rxn):
    """The model's own reaction for a reaction object or id (objects can be stale after a copy)."""
    return model.reactions.get_by_id(getattr(rxn, "id", rxn))


def _e0_id(cpd):
    cpd = str(getattr(cpd, "id", cpd))
    return cpd if cpd.endswith(f"_{SHARED}") else f"{cpd}_{SHARED}"


def _cpd_key(met_id):
    suffix = f"_{SHARED}"
    return met_id[:-len(suffix)] if met_id.endswith(suffix) else met_id


def _coefficient(rxn, met_id):
    """The reaction's coefficient on a metabolite, looked up by id (objects compare by identity)."""
    for met, coef in rxn.metabolites.items():
        if met.id == met_id:
            return float(coef)
    return 0.0


def _fluxes(fluxes):
    """A flux mapping from a cobra Solution, a pandas Series or a dict; refuses a failed solve."""
    status = getattr(fluxes, "status", None)
    if status is not None and hasattr(fluxes, "fluxes"):
        if status != "optimal":
            raise ValueError(f"the solution is {status!r}, not optimal: its fluxes mean nothing")
        return fluxes.fluxes
    return fluxes


# --------------------------------------------------------------------------- membership
def _community_biomass_ids(comm):
    if _is_community(comm) and comm.primary_biomass is not None:
        return {comm.primary_biomass.id}
    # bare model: the reaction(s) making the community biomass cpd11416_c0 from member biomass
    return {r.id for r in _model(comm).reactions if len(r.metabolites) > 1
            and any(m.id.startswith("cpd11416") and m.compartment == "c0" and c > 0
                    for m, c in r.metabolites.items())}


def _member_biomass_ids(comm, idx):
    """The member's own biomass reaction and biomass drain."""
    if _is_community(comm):
        m = _member(comm, idx)
        return {r.id for r in (m.primary_biomass, m.biomass_drain) if r is not None}
    comp, out = f"c{idx}", set()
    for r in _model(comm).reactions:
        for met, c in r.metabolites.items():
            if met.id.startswith("cpd11416") and met.compartment == comp:
                if (c > 0 and len(r.metabolites) > 1) or (c < 0 and len(r.metabolites) == 1):
                    out.add(r.id)
    return out


def _assignments(comm):
    """[(reaction, owner)] for every reaction of the model: owner = member index, 0 for the
    shared pool and the community compartment, None for the community biomass and for a reaction
    spanning several members. One pass of member_index over the model."""
    community = _community_biomass_ids(comm)
    return [(r, None if r.id in community else member_index(r)) for r in _model(comm).reactions]


def member_reactions(comm, member, *, include_biomass=True) -> list:
    """The cobra reactions belonging to `member` (see Membership in the module docstring): its
    compartments' reactions, its transporters to e0, its biomass reaction, drain and other
    boundary reactions on its own metabolites. Never bio1, an exchange or an e0-only reaction.
    include_biomass=False leaves out the member's biomass reaction and biomass drain (what the
    kinetic row counts). Reactions in model order."""
    idx = _index(comm, member)
    out = [r for r, owner in _assignments(comm) if owner == idx]
    if not include_biomass:
        drop = _member_biomass_ids(comm, idx)
        out = [r for r in out if r.id not in drop]
    return out


def reactions_by_member(comm, *, include_biomass=True) -> dict:
    """member_reactions for every member in one pass over the model: {member id: [reactions]}
    (on a bare model {member index: [reactions]}, for every index that owns a reaction)."""
    grouped = {}
    for r, owner in _assignments(comm):
        if owner:
            grouped.setdefault(owner, []).append(r)
    if not _is_community(comm):
        if not include_biomass:
            for idx in grouped:
                drop = _member_biomass_ids(comm, idx)
                grouped[idx] = [r for r in grouped[idx] if r.id not in drop]
        return grouped
    out = {}
    for m in comm.members:
        rxns = grouped.get(m.index, [])
        if not include_biomass:
            drop = _member_biomass_ids(comm, m)
            rxns = [r for r in rxns if r.id not in drop]
        out[m.id] = rxns
    return out


def _transporters(rxns, met_id=None):
    out = []
    for r in rxns:
        e0 = {m.id for m in r.metabolites if m.compartment == SHARED}
        if e0 and (met_id is None or met_id in e0):
            out.append(r)
    return out


def member_transporters(comm, member, cpd=None) -> list:
    """The member's reactions that move a compound between its compartments and e0 (every
    member reaction touching an e0 metabolite), or only those touching <cpd>_e0 when `cpd` is
    given ("cpd00027" or "cpd00027_e0"). For a member with a periplasm these are the outer-membrane
    steps e0 <-> p<i>: uptake is counted where the compound leaves the shared pool.
    KeyError if <cpd>_e0 is not in the model; [] if the member has no transporter for it."""
    met_id = None
    if cpd is not None:
        met_id = _e0_id(cpd)
        if met_id not in _model(comm).metabolites:
            raise KeyError(f"no shared-pool metabolite {met_id} in {_model(comm).id}")
    return _transporters(member_reactions(comm, member), met_id)


# --------------------------------------------------------------------------- uptake
def _override(comm, model, idx, reactions, met_id):
    """Validate reactions= : live, not boundary, the member's own or e0-only, touching the compound."""
    out, community = [], _community_biomass_ids(comm)
    for item in reactions:
        r = _live(model, item)
        if r.boundary:
            raise ValueError(f"{r.id} is a boundary reaction: a member's uptake is counted on its "
                             "transporters, never on the shared exchange (that limits the whole community)")
        owner = None if r.id in community else member_index(r)
        e0_only = all(m.compartment == SHARED for m in r.metabolites)
        if owner != idx and not (owner == 0 and e0_only):
            raise ValueError(f"{r.id} belongs to {'no single member' if owner is None else f'index {owner}'}, "
                             f"not to {_label(comm, idx)} (index {idx}) or the shared pool")
        if met_id is not None and _coefficient(r, met_id) == 0.0:
            raise ValueError(f"{r.id} does not move {met_id}")
        out.append(r)
    if not out:
        raise ValueError("reactions= is empty")
    return out


def _uptake_terms(comm, member, cpd, reactions, gross):
    """([(optlang variable, coefficient)], [reactions]) of the member's uptake of <cpd>_e0.

    Per reaction, s = -(its coefficient on <cpd>_e0) is the uptake per unit of forward flux.
    Net: s * (forward - reverse). Gross (import direction only): s * forward when s > 0,
    |s| * reverse when s < 0. With cpd=None (reactions= only) s = 1: each reaction as written is
    the uptake direction."""
    model = _model(comm)
    idx = _index(comm, member)
    met_id = None
    if cpd is not None:
        met_id = _e0_id(cpd)
        if met_id not in model.metabolites:
            raise KeyError(f"no shared-pool metabolite {met_id} in {model.id}")
    elif reactions is None:
        raise ValueError("give the compound, or the reactions whose forward flux is the uptake (reactions=)")
    if reactions is None:
        rxns = member_transporters(comm, idx, cpd)
        if not rxns:
            raise ValueError(f"{_label(comm, idx)} has no transporter for {met_id}")
    else:
        rxns = _override(comm, model, idx, reactions, met_id)
    terms = []
    for r in rxns:
        s = 1.0 if met_id is None else -_coefficient(r, met_id)
        if gross:
            if s > 0:
                terms.append((r.forward_variable, s))
            elif s < 0:
                terms.append((r.reverse_variable, -s))
        else:
            terms += [(r.forward_variable, s), (r.reverse_variable, -s)]
    return terms, rxns


def member_uptake_expression(comm, member, cpd, *, reactions=None, gross=False):
    """optlang expression of the member's uptake of `cpd` from e0, in mmol per gDW of community
    per h; positive = the member takes it from the pool.

    gross=False: net over the member's transporters for cpd (a transporter running outward counts
    negative). gross=True: import direction only (each transporter's forward or reverse variable,
    whichever moves cpd inward), so uptake through one transporter cannot be netted against
    release through another; this is what a capacity cap needs.
    reactions= replaces the transporter set (ids or reactions): the member's own reactions or
    e0-only ones, each touching <cpd>_e0, never an exchange. Use it where the net over all
    transporters mixes in a release, e.g. B. thetaiotaomicron's dextran hydrolysis, which frees
    glucose into e0 that the member then takes back up. With cpd=None and reactions= given, each
    reaction's forward flux is the uptake (one unit = 1 mmol).
    Raises KeyError for an unknown compound and ValueError when the member has no transporter
    for it."""
    terms, _ = _uptake_terms(comm, member, cpd, reactions, gross)
    return add([coef * var for var, coef in terms])


def member_net_exchange(comm, solution_or_fluxes, member, *, tol=1e-9) -> dict:
    """{compound: net flux into the member from e0} in a solution, mmol per gDW of community per h:
    positive = uptake, negative = release. Compound keys are e0 metabolite ids without "_e0".
    Entries with |flux| <= tol are left out. Divide by a_m for per-gDW-member rates; summed over
    members (plus any e0-only reactions) it balances the exchanges: sum_m net_m[c] = -v(EX_c_e0)
    when no e0-only reaction touches c. Accepts a cobra Solution (must be optimal), a pandas
    Series or a dict of fluxes."""
    fluxes = _fluxes(solution_or_fluxes)
    net = {}
    for r in member_transporters(comm, member):
        v = float(fluxes[r.id])
        if v == 0.0:
            continue
        for met, coef in r.metabolites.items():
            if met.compartment == SHARED:
                key = _cpd_key(met.id)
                net[key] = net.get(key, 0.0) - float(coef) * v
    return {k: v for k, v in net.items() if abs(v) > tol}


# --------------------------------------------------------------------------- rates and drains
def member_rates(comm, fluxes, *, abundances=None) -> dict:
    """{member id: specific growth mu_m = v_bio_m / a_m} in 1/h, v_bio_m the member's biomass
    reaction flux (per gDW community) and a_m its current abundance (comm.members[...].abundance),
    or the value in `abundances` ({member id: a_m}; e.g. the X_m of a balanced-growth solution).
    None for a member with a_m = 0.

    With every member drain closed (close_member_drains), member biomass can leave only through
    bio1, which takes a_m of it per unit: v_bio_m = a_m * v_bio1, so mu_m = mu_com for every member
    BY CONSTRUCTION and the rates carry no information. With a drain open,
    mu_m = mu_com + v_drain_m / a_m."""
    _require_community(comm, "member_rates")
    fluxes = _fluxes(fluxes)
    out = {}
    for m in comm.members:
        a = float(abundances[m.id]) if abundances is not None else float(m.abundance)
        v = float(fluxes[m.primary_biomass.id])
        out[m.id] = v / a if a > 0 else None
    return out


def _set_drains(comm, bounds_of):
    _require_community(comm, "the member drain helpers")
    model, previous = _model(comm), {}
    for m in comm.members:
        if m.biomass_drain is None:
            continue
        rxn = _live(model, m.biomass_drain)
        previous[rxn.id] = tuple(rxn.bounds)
        rxn.bounds = tuple(float(b) for b in bounds_of(m))
    return previous


def open_member_drains(comm, bounds=(0.0, 1000.0)) -> dict:
    """Set every member's biomass drain to `bounds` (None: the bounds it was built with,
    member.biomass_drain_bounds). Returns {drain id: previous bounds} for restore_member_drains.
    Inside `with comm.util.model:` cobra also undoes it on exit."""
    return _set_drains(comm, lambda m: m.biomass_drain_bounds if bounds is None else bounds)


def close_member_drains(comm) -> dict:
    """Shut every member's biomass drain, (0, 0) -- what MSCommunity(close_member_drains=True)
    does at construction -- so member growth is pinned to a_m * bio1. Returns {drain id: previous
    bounds} for restore_member_drains. (comm.close_member_drains, the constructor flag, is not
    changed: the drains' bounds are the state.)"""
    return _set_drains(comm, lambda m: (0.0, 0.0))


def restore_member_drains(comm, previous) -> None:
    """Put back the drain bounds returned by open_member_drains / close_member_drains."""
    model = _model(comm)
    for rid, bounds in previous.items():
        model.reactions.get_by_id(rid).bounds = tuple(bounds)


def lifted_kinetic_rows(comm, lift=True):
    """Context manager lifting the *_commKin rows for the duration of the block (bounds to +-1e30,
    restored on exit); yields the lifted rows' names. commcycles.lifted_kinetic_rows, which takes
    the model, unwrapped from the MSCommunity here (a bare model is passed through). Caps and
    maintenance rows are not lifted."""
    return _cycles.lifted_kinetic_rows(_model(comm), lift)


def member_biomass_variables(comm, *, prefix="X", lb=0.0, ub=None) -> dict:
    """{member id: optlang variable <prefix>_<member id>} in the model, created with bounds
    (lb, ub) where missing and returned as they are where present (so repeated calls and the four
    downstream solvers share one X_m per member). The caller owns them: their meaning (gDW member
    per gDW community in SteadyCom, gDW/L in a chemostat), the rows tying them together and their
    removal (model.remove_cons_vars)."""
    _require_community(comm, "member_biomass_variables")
    model, new, names = _model(comm), [], {}
    for m in comm.members:
        name = f"{prefix}_{m.id}"
        names[m.id] = name
        if name not in model.variables:
            new.append(model.problem.Variable(name, lb=lb, ub=ub))
    if new:
        model.add_cons_vars(new)
    return {mid: model.variables[name] for mid, name in names.items()}


# --------------------------------------------------------------------------- caps
def _registry(comm):
    reg = getattr(comm, _REGISTRY, None)
    if reg is None:
        reg = {}
        setattr(comm, _REGISTRY, reg)
    return reg


def _basis_variable(model, basis, X, member):
    if basis == "abundance":
        if X is not None:
            raise ValueError("X is used only with basis='X'")
        return None
    if basis != "X":
        raise ValueError(f"basis must be one of {BASES}, not {basis!r}")
    if X is None:
        raise ValueError("basis='X' needs X: the member's optlang variable, or {member id: variable}")
    if isinstance(X, Mapping):
        X = X[member.id]
    name = getattr(X, "name", None)
    if name is None or name not in model.variables:
        raise ValueError(f"X ({name or X!r}) is not a variable of the model: add it first "
                         "(member_biomass_variables, or model.add_cons_vars)")
    return model.variables[name]   # the live one, should the solver interface have been swapped


def _rate(value, what):
    value = float(value)
    if not isfinite(value) or value < 0:
        raise ValueError(f"{what} must be a finite number >= 0, not {value}")
    return value


def _install(comm, record, constraint):
    model, reg = _model(comm), _registry(comm)
    if record.name in reg:
        _uninstall(comm, record.name)   # same name: the new row replaces the old one
    elif record.name in model.constraints:
        raise ValueError(f"the model already has a row {record.name} that this module did not add")
    model.add_cons_vars([constraint])   # cobra undoes this on leaving `with model:`
    reg[record.name] = record
    context = get_context(model)
    if context is not None:
        context(partial(reg.pop, record.name, None))
    return model.constraints[record.name]


def _uninstall(comm, name):
    model, reg = _model(comm), _registry(comm)
    record = reg.pop(name)
    context = get_context(model)
    if context is not None:
        context(partial(reg.__setitem__, name, record))
    if name in model.constraints:
        model.remove_cons_vars([model.constraints[name]])
    if record.released is not None and record.reactions[0] in model.reactions:
        rxn = model.reactions.get_by_id(record.reactions[0])
        if rxn.lower_bound == 0.0:
            rxn.lower_bound = record.released
        else:
            logger.warning("%s: lower bound changed to %s since %s released its floor %s; left as is",
                           rxn.id, rxn.lower_bound, name, record.released)
    return record


def cap_member_uptake(comm, member, cpd=None, *, q, reactions=None, basis="abundance", X=None, name=None):
    """Cap the member's GROSS uptake of `cpd` from e0 at q mmol per gDW of THAT member per h and
    return the optlang row (added to comm.util.model and recorded in comm._member_caps):

        basis="abundance"   gross_uptake <= q * a_m      (a_m now; refresh_caps after set_abundance)
        basis="X"           gross_uptake - q * X_m <= 0  (X: the member's variable, already in the
                            model, or {member id: variable})

    gross_uptake is member_uptake_expression(..., gross=True) over the member's transporters for
    cpd, or over reactions= (see there; cpd=None with reactions= caps their forward flux). The
    cap is never written on the shared EX_ reaction. name defaults to mcap_<member>_<cpd> (with
    cpd=None, mcap_<member>_<first reaction>); a second cap with the same name replaces the
    first. Raises KeyError / ValueError as member_uptake_expression does, and ValueError for a
    negative q or a bad basis."""
    _require_community(comm, "cap_member_uptake")
    model, m = _model(comm), _member(comm, member)
    q = _rate(q, "q")
    X_var = _basis_variable(model, basis, X, m)
    terms, rxns = _uptake_terms(comm, m, cpd, reactions, gross=True)
    key = _cpd_key(_e0_id(cpd)) if cpd is not None else None
    name = name or f"{CAP_PREFIX}{m.id}_{key or rxns[0].id}"
    if X_var is None:
        row = model.problem.Constraint(add([c * v for v, c in terms]), ub=q * m.abundance, name=name)
    else:
        row = model.problem.Constraint(add([c * v for v, c in terms] + [-q * X_var]), ub=0.0, name=name)
    record = MemberCap(name=name, kind="uptake", member=m.id, cpd=key, rate=q, basis=basis,
                       X=None if X_var is None else X_var.name, reactions=tuple(r.id for r in rxns))
    return _install(comm, record, row)


def atp_maintenance_reaction(comm, member, reaction=None, *, required=True):
    """The member's ATP maintenance reaction in the live model: `reaction` (id or object) when
    given, else rxn00062_c<i>, else a member reaction named ATPM (ATPM_c<i>), else the ATP
    hydrolysis CommunityMember found or made at construction (member.atp_hydrolysis) if it is
    still in the model. LookupError when there is none or several ATPM reactions compete (None
    with required=False)."""
    _require_community(comm, "atp_maintenance_reaction")
    model, m = _model(comm), _member(comm, member)
    if reaction is not None:
        rxn = _live(model, reaction)
        if member_index(rxn) != m.index:
            raise ValueError(f"{rxn.id} is not a reaction of {m.id} (index {m.index})")
        return rxn
    rid = f"{ATPM_MSID}_c{m.index}"
    if rid in model.reactions:
        return model.reactions.get_by_id(rid)
    named = [r for r in member_reactions(comm, m) if _ID_SUFFIX.sub("", r.id).upper() == "ATPM"]
    if len(named) > 1:
        if not required:
            return None
        raise LookupError(f"{m.id} has several ATPM reactions {[r.id for r in named]}: pass reaction=")
    if named:
        return named[0]
    own = getattr(m, "atp_hydrolysis", None)
    if own is not None and own.id in model.reactions and member_index(model.reactions.get_by_id(own.id)) == m.index:
        return model.reactions.get_by_id(own.id)
    if required:
        raise LookupError(f"{m.id} has no ATP maintenance reaction: no {rid}, no ATPM_c{m.index}, and "
                          "member.atp_hydrolysis is not in the model; pass reaction=")
    return None


def set_member_maintenance(comm, member, A_m, *, reaction=None, basis="abundance", X=None):
    """Require the member's ATP maintenance flux: ATPM_m >= A_m * a_m (basis="abundance") or
    ATPM_m - A_m * X_m >= 0 (basis="X"), A_m in mmol ATP per gDW of the member per h. Returns the
    optlang row (named mmaint_<member>, recorded in comm._member_caps; a second call replaces it).

    The reaction is atp_maintenance_reaction(comm, member, reaction): LookupError when the member
    has none -- a maintenance demand with nowhere to go is never silently dropped. A positive
    constant lower bound on that reaction (a floor written for one composition) is set to 0, so
    the row is the maintenance demand, and is put back by remove_member_caps. See the module
    docstring for energy audits, which this row makes infeasible until it is removed or lifted."""
    _require_community(comm, "set_member_maintenance")
    model, m = _model(comm), _member(comm, member)
    A = _rate(A_m, "A_m")
    rxn = atp_maintenance_reaction(comm, m, reaction)
    X_var = _basis_variable(model, basis, X, m)
    name = f"{MAINT_PREFIX}{m.id}"
    if name in _registry(comm):
        _uninstall(comm, name)   # restores the floor the old row released before we look at it
    elif name in model.constraints:   # checked before the floor is touched
        raise ValueError(f"the model already has a row {name} that this module did not add")
    released = None
    if rxn.lower_bound > 0:
        released = float(rxn.lower_bound)
        rxn.lower_bound = 0.0   # cobra undoes this on leaving `with model:`
    if X_var is None:
        row = model.problem.Constraint(rxn.flux_expression, lb=A * m.abundance, name=name)
    else:
        row = model.problem.Constraint(rxn.flux_expression - A * X_var, lb=0.0, name=name)
    record = MemberCap(name=name, kind="maintenance", member=m.id, cpd=None, rate=A, basis=basis,
                       X=None if X_var is None else X_var.name, reactions=(rxn.id,), released=released)
    return _install(comm, record, row)


def _drop_stale(comm):
    """Forget registry entries whose row is gone from the model (removed by other code)."""
    model, reg = _model(comm), _registry(comm)
    for name in [n for n in reg if n not in model.constraints]:
        logger.warning("member cap %s is no longer in the model; forgetting it", name)
        reg.pop(name)


def member_caps(comm, member=None, *, kind=None) -> list:
    """The MemberCap records in comm._member_caps, optionally one member's and/or one kind's
    ("uptake" or "maintenance")."""
    _require_community(comm, "member_caps")
    _drop_stale(comm)
    mid = _member(comm, member).id if member is not None else None
    return [rec for rec in _registry(comm).values()
            if (mid is None or rec.member == mid) and (kind is None or rec.kind == kind)]


def refresh_caps(comm) -> list:
    """Rewrite every basis="abundance" row from the members' current abundances (uptake:
    ub = q * a_m; maintenance: lb = A_m * a_m). Call after comm.set_abundance, which does not.
    basis="X" rows need nothing. Returns the names rewritten. Inside `with comm.util.model:` the
    rows' previous bounds are put back when the context exits (one undo step registered here,
    since cobra does not track optlang row bounds)."""
    model, done, previous = _model(comm), [], {}
    for rec in member_caps(comm):
        if rec.basis != "abundance":
            continue
        row, a = model.constraints[rec.name], float(comm.members.get_by_id(rec.member).abundance)
        previous[rec.name] = (row.lb, row.ub)
        if rec.kind == "uptake":
            row.ub = rec.rate * a
        else:
            row.lb = rec.rate * a
        done.append(rec.name)
    context = get_context(model)
    if context is not None and previous:
        context(partial(_restore_row_bounds, model, previous))
    return done


def _restore_row_bounds(model, bounds):
    """Put back {row name: (lb, ub)} on the rows still in the model, in an order optlang accepts."""
    for name, (lb, ub) in bounds.items():
        if name not in model.constraints:
            continue
        row = model.constraints[name]
        if lb is not None and row.ub is not None and lb > row.ub:
            row.ub, row.lb = ub, lb
        else:
            row.lb, row.ub = lb, ub


def remove_member_caps(comm, member=None, cpd=None, *, kind=None) -> list:
    """Remove the rows in comm._member_caps matching member / compound / kind (all of them by
    default) from the model and the registry, putting back any ATPM floor a maintenance row
    released. Returns the names removed."""
    key = _cpd_key(_e0_id(cpd)) if cpd is not None else None
    names = [rec.name for rec in member_caps(comm, member, kind=kind) if key is None or rec.cpd == key]
    for name in names:
        _uninstall(comm, name)
    return names


def set_member_abundances(comm, abundances) -> dict:
    """comm.set_abundance, then refresh_caps. abundances: {member id: a} or the
    {member id: {"abundance": a}} form set_abundance takes, for every member; normalised to sum to
    1 as set_abundance does. Returns the normalised {member id: a_m}. Inside `with comm.util.model:`
    everything it changes reverts on exit: bio1, the abundances (set_abundance) and the
    abundance-basis cap / maintenance row bounds (refresh_caps)."""
    _require_community(comm, "set_member_abundances")
    values = {k: (v["abundance"] if isinstance(v, Mapping) else v) for k, v in abundances.items()}
    ids = {m.id for m in comm.members}
    if set(values) != ids:
        raise ValueError(f"give an abundance for every member: missing {sorted(ids - set(values))}, "
                         f"unknown {sorted(set(values) - ids)}")
    comm.set_abundance({k: {"abundance": float(v)} for k, v in values.items()})
    refresh_caps(comm)
    return {m.id: m.abundance for m in comm.members}


# --------------------------------------------------------------------------- reporting
def member_table(comm) -> MemberTable:
    """MemberTable of the community as it is now (one pass over the model)."""
    _require_community(comm, "member_table")
    model = _model(comm)
    assigned = _assignments(comm)
    owned = reactions_by_member(comm)
    shared = [r for r, owner in assigned if owner == 0 and not r.boundary
              and all(met.compartment == SHARED for met in r.metabolites)]
    community = _community_biomass_ids(comm)
    infos = []
    for m in comm.members:
        rxns = owned[m.id]
        transporters = {}
        for r in _transporters(rxns):
            for met in r.metabolites:
                if met.compartment == SHARED:
                    transporters.setdefault(_cpd_key(met.id), []).append(r.id)
        comps = sorted({met.compartment for r in rxns for met in r.metabolites if met.compartment != SHARED})
        suffix = [r.id for r in shared if (s := _ID_SUFFIX.search(r.id)) and int(s.group(2)) == m.index]
        drain = _live(model, m.biomass_drain) if m.biomass_drain is not None else None
        atpm = atp_maintenance_reaction(comm, m, required=False)
        infos.append(MemberInfo(
            id=m.id, index=m.index, abundance=float(m.abundance),
            biomass=m.primary_biomass.id if m.primary_biomass is not None else None,
            drain=drain.id if drain is not None else None,
            drain_bounds=tuple(drain.bounds) if drain is not None else None,
            atp_maintenance=atpm.id if atpm is not None else None,
            compartments=tuple(comps), n_reactions=len(rxns),
            transporters={k: tuple(v) for k, v in sorted(transporters.items())},
            extracellular=tuple(suffix)))
    return MemberTable(
        members=tuple(infos),
        community_biomass=next(iter(sorted(community)), None),
        shared=tuple(r.id for r in shared),
        unassigned=tuple(r.id for r, owner in assigned if owner is None and r.id not in community),
        caps=tuple(member_caps(comm)))
