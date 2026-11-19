# -*- coding: utf-8 -*-
"""Chemostat steady states of an MSCommunity: every subset of surviving members grows at mu = D.

chemostat(comm, D, feed, kinetics=...) enumerates the steady states of a continuous stirred-tank
reactor (CSTR) that holds the community's members, one subset of surviving members at a time (the
others washed out), and reports for each state its member biomasses, residual concentrations,
composition, local stability, the growth rates of the absent members (invasion), which uptake
ceilings bind, the margins of growth switches, the supply x yield decomposition of each member's
biomass and the existence criteria of switched members.

Model (the dynamic multi-species FBA structure of Khazaei et al. 2020, solved for its fixed points)
    state       X_m (gDW member / L) for every member, S_c (mM) for every TRACKED compound c
    dX_m/dt  =  (mu_m(S) - D) X_m
    dS_c/dt  =  D (S_in,c - S_c) - sum_m X_m u_m,c(S)          liquid compounds
    dC_g/dt  =  kLa (C*_g - C_g) - sum_m X_m u_m,g(S)          gas compounds (gas=); the liquid
                                                               outflow D*C_g is neglected (kLa >> D)
    mu_m(S), u_m,c(S): member m's own FBA (its growth rate, 1/h, and its net uptake of c from the
    pool, mmol per gDW of m per h; negative = release) with uptake ceilings vmax S/(Km + S) per gDW
    of m at the residual concentrations S.
    A steady state with surviving set P: mu_m(S) = D for m in P, X_m = 0 otherwise, and every
    tracked balance zero. The washout state (P empty: S = S_in, C = C*) is always reported.

Member blocks, not one merged LP
    Each member is its own LP, built from its block of the merged community model
    (commmembers.member_reactions: its compartments, its transporters to e0 and its biomass
    reaction and drain) plus one exchange per e0 compound it touches. The block is the member's own
    model: its fluxes are per gDW of THAT member, its biomass leaves through its drain (opened in
    the block) and mu_m is its biomass flux. The members interact only through the residual
    concentrations of the pool. This is what makes each X_m a consequence of the member's own
    yield on its own limiting supply; a single community LP with fluxes per gDW of community can
    credit one member's uptake to another member's biomass (relabelling) and imposes
    mu_m = mu_com instead of letting each member maximise its own growth.

    Block exchanges:
      tracked compound with kinetics for the member   lower bound -1000; the uptake is limited by a
                                                      ceiling row (below)
      tracked compound without kinetics for it        lower bound 0: the member does not take it
                                                      from the pool (it may release it)
      any other e0 compound                           the bounds of the community model's exchange
                                                      for it (the static, non-limiting medium: set
                                                      comm.util.model's exchanges as for any FBA),
                                                      overridden per member by member_bounds=;
                                                      (0, 1000) when the community has no exchange
    Ceiling rows (one per kinetics entry), per gDW of the member:
      "EX_<cpd>_e0" or "<cpd>"   gross import of <cpd> through the member's transporters
                                 (commmembers.member_uptake_expression(..., gross=True))
      (reaction ids,)            gross import through exactly those reactions (each reaction's
                                 forward or reverse variable, whichever imports the compound); use
                                 it when the member also RELEASES the compound into the pool (B.
                                 thetaiotaomicron's dextran hydrolysis frees glucose into e0)
      ceiling = vmax S/(Km + S) at the compound's residual S (0 at S <= 0). Uptake caps registered
      with commmembers.cap_member_uptake are honoured as constant ceilings q (S-independent).
    Extracellular reactions (all metabolites in e0) belong to no member in the merged model; a
    block includes those attributed to it by `extracellular=` (default: the reactions
    MemberInfo.extracellular lists, i.e. e0-only reactions whose id carries the member's _c<i>
    suffix; reactions that kept an _e0 suffix are NOT attributed by that rule -- pass them).

Maintenance, per gDW of the member (never a per-community floor)
    "per_member"  the member's ATP maintenance reaction (commmembers.atp_maintenance_reaction) keeps
                  its bounds as they stand in the merged model, read in the block as mmol ATP per
                  gDW of the member per h (right for the floor a member model brings, e.g. a fixed
                  ATPM of 6.8; WRONG for a floor someone rescaled to the community basis, A_m * a_m:
                  pass a dict then). A maintenance row registered with
                  commmembers.set_member_maintenance (basis "abundance" or "X") gives A_m instead:
                  the block's ATPM lower bound is its rate, the floor it released is ignored.
    {member: A_m} ATPM lower bound A_m in that member's block (others: "per_member")
    None / "free" every member's ATPM released to (0, max(ub, 1000))

Tie-break
    tie_break="pfba": each member's fluxes are taken at minimal total flux among its growth-optimal
    solutions (bio >= mu_max (1 - 1e-9)), so the by-products it releases -- and with them every
    cross-fed residual -- are reproducible. tie_break=None takes the solver's vertex.

Switches
    switches={member: callable(conc) -> bool}: True = the member can grow; False = zero growth and
    zero fluxes (a strict anaerobe above an O2 threshold). conc is {compound: mM} over the tracked
    compounds. The steady states are solved with every switch ignored (the smooth field), then a
    state EXISTS only if the switch of every surviving member is on there; states that do not exist
    are still returned (exists=False) because a threshold is often a guess. Invasion rates of the
    absent members are reported with their switches honoured (and without). A ConcentrationSwitch
    (compound, threshold) also reports its margin and the existence criterion: the consumer biomass
    needed to pull (or keep) the compound across the threshold, when the consumers' ceilings bind,
    e.g. X_Kp >= kLa (C* - thr) / q_Kp,O2(thr) for O2. A plain callable reports on/off only.

What the result means (read before using a composition)
    * mu = D removes the max-growth selection that makes balanced-growth (SteadyCom-type) LPs return
      a single-member corner: every surviving member grows at D by construction, so the
      composition is not chosen by an objective.
    * The composition then follows SUPPLY x YIELD: each member's X_m is its yield on its limiting
      supply times the share of that supply it receives (SteadyState.supply_yield). Michaelis-Menten
      kinetics set the residuals (often micromolar) and decide which states EXIST (a residual
      against a switch threshold) and which are stable, but they barely move the composition.
    * Results are therefore only as good as the member yields and maintenance (they set X_m) and
      the partition of shared substrates (who takes which released compound, e.g. the glucose a
      hydrolase frees into the pool). A wrong yield gives a wrong composition with a converged,
      stable, well-conditioned fixed point. On K. pneumoniae + B. thetaiotaomicron (Khazaei 2020)
      B. thetaiotaomicron is dextran-limited at an X fixed by its dextran yield and K. pneumoniae
      takes the feed glucose, so a_Kp falls with the glucose feed (0.848 at 5 mM to 0.778 at 0); the
      prototype's x0.5 - x2 changes of every Km and vmax moved a_Kp by <= 0.003, while whether the
      state exists under the paper's O2 rule is decided by X_Kp >= 0.483 gDW/L, i.e. by the yield.

Numerics
    For each subset the unknowns (X_m of the survivors and every tracked S) are solved in log space
    by scipy least_squares (trf) from several starts (generic supply x yield seeds, perturbed, plus
    seeds= / warm= starts); a root is accepted at max |scaled residual| <= tol (balances scaled by
    D * S_in,c, or D * 1 mM when unfed, gas by kLa * C*; growth by D). Roots closer than 1e-4 in log
    distance are merged. A residual that ends on the 1e-12 mM floor of the log variables (a
    compound nobody supplies) is reported as 0. Enumeration is by multistart: several roots per
    subset are kept when found, and a subset without a root is listed in .unsolved with the best
    residual reached -- evidence, not proof, that it has none. Member LP responses are cached on the
    concentrations of the member's own kinetic compounds, so perturbing X or a compound the member
    does not take costs no LP. Local
    stability: eigenvalues of the central-difference Jacobian (relative step 1e-3) of the reduced
    ODE (survivors' X and all tracked S, switches ignored); forward and backward differences are
    computed too and `stability_ambiguous` flags a sign disagreement (the LP responses are
    piecewise linear). The absent members' eigenvalues are their invasion rates mu_a(S*) - D. A
    subset that contains a member which cannot reach mu = D even with every ceiling at vmax is
    skipped (no state can have it surviving).

Units: X gDW/L; concentrations mM; D and kLa 1/h; vmax and fluxes mmol per gDW of the member per h;
Km mM; yields gDW per mmol.

Reference prototype: 3H11_R12_SynCom/Metabolic_modeling/abund_chemostat_lib.py and
abund_chemostat_steady.py (Kp + Bt, Khazaei 2020); results in
data/multisystem3/abundance/chemostat/README.md.
"""
from __future__ import annotations

import itertools
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from cobra import Metabolite, Model, Reaction
from optlang.symbolics import add

from mscommunity.commmembers import (
    SHARED, atp_maintenance_reaction, member_caps, member_reactions, member_table, member_transporters,
    member_uptake_expression)

logger = logging.getLogger(__name__)

__all__ = ["ConcentrationSwitch", "Ceiling", "SteadyState", "SteadyStates", "chemostat",
           "member_block", "MemberBlock"]

PFBA_RELAX = 1e-9           # pFBA keeps biomass >= mu_max (1 - PFBA_RELAX)
CAP_ZERO = 1e-300           # only underflow is written as 0: tiny ceilings carry the MM slope vmax/Km
MU_ZERO = 1e-12
LOG_FLOOR = 1e-12           # lower bound of every unknown (gDW/L or mM)
LOG_CEIL = 1e4
X_SURVIVOR_MIN = 1e-9       # a root with a survivor below this (gDW/L) is a washout, not a state
BINDING_MIN = 1e-9          # a ceiling below this (mmol/gDW/h) limits nothing worth reporting


# --------------------------------------------------------------------------- records
@dataclass(frozen=True)
class ConcentrationSwitch:
    """A growth switch on one compound's residual: the member grows only while
    conc[cpd] <= threshold (off="above": switched off above the threshold, e.g. a strict anaerobe's
    O2 limit) or conc[cpd] >= threshold (off="below"). threshold in mM. Callable as a switch;
    margin(conc) is the distance to the threshold on the ON side (positive = on, mM)."""
    cpd: str
    threshold: float
    off: str = "above"

    def __post_init__(self):
        if self.off not in ("above", "below"):
            raise ValueError(f"off must be 'above' or 'below', not {self.off!r}")
        object.__setattr__(self, "cpd", _cpd_of(self.cpd))
        object.__setattr__(self, "threshold", float(self.threshold))

    def margin(self, conc):
        c = float(conc.get(self.cpd, 0.0))
        return self.threshold - c if self.off == "above" else c - self.threshold

    def __call__(self, conc):
        return self.margin(conc) >= 0.0


@dataclass(frozen=True)
class Ceiling:
    """One Michaelis-Menten uptake ceiling of one member at a steady state.

    key        the kinetics key as given; cpd the compound whose residual drives it
    reactions  the reactions whose gross import is capped
    cap        vmax S/(Km + S) at the residual, mmol per gDW of the member per h
    uptake     the gross import through those reactions at the state (same units)
    binding    uptake >= cap (1 - 1e-6) - 1e-9 and cap > 1e-12"""
    key: str
    cpd: str
    reactions: Tuple[str, ...]
    vmax: float
    Km: float
    source: Optional[str]
    residual_mM: float
    cap: float
    uptake: float
    binding: bool


@dataclass
class SteadyState:
    """One steady state of chemostat(): the surviving `members` grow at mu = D.

    X                {member: gDW/L} for every member considered (absent: 0)
    residuals        {compound: mM} of every tracked compound
    a                {member: X_m / sum X} over the survivors (gDW fraction); {} at washout
    mu               {member: growth rate at the residuals, 1/h, switches ignored}
    uptake           {member: {compound: net uptake from the pool, mmol/gDW_member/h; < 0 release}}
    eigenvalues      eigenvalues of the reduced ODE's Jacobian (survivors' X and every tracked S,
                     switches ignored), sorted by real part; [] at washout without tracked S
    jacobian         names, central-difference matrix, forward / backward eigenvalues and the
                     relative one-sided disagreement
    stable_smooth    every central, forward and backward eigenvalue has a negative real part
    stability_ambiguous  central and one-sided differences disagree on the sign
    invasion         {absent member: mu_a(S*) - D}, switches honoured (the absent eigenvalues)
    invasion_without_switches  the same with every switch ignored
    ceilings         {member: [Ceiling]} for the survivors
    binding          {member: [ceiling keys that bind]}
    switch_on        {member: bool} for every member with a switch, at the residuals
    switch_margins   {member: margin (mM, positive = on side)} for ConcentrationSwitch members,
                     None for plain callables
    near_switch      a survivor's margin is within the finite-difference reach: the Jacobian there
                     straddles a discontinuity
    exists           every survivor's switch is on (the state is a steady state of the switched
                     model, not only of the smooth one)
    stable           exists, stable_smooth and every invasion rate < 0
    supply_yield     {member: [{cpd, uptake_per_gDW, yield_gDW_per_mmol, uptake_mmol_per_L_h,
                     supply_mmol_per_L_h {feed|gas|released_by}, share_of_supply, ...}]} over the
                     member's binding ceilings: X_m = yield x share x supply
    existence        [criterion per ConcentrationSwitch of a survivor]: the consumer biomass needed
                     (or allowed) to hold the compound on the ON side of the threshold
    residual_norm    max |scaled residual| at the root
    """
    members: Tuple[str, ...]
    X: Dict[str, float]
    residuals: Dict[str, float]
    a: Dict[str, float]
    mu: Dict[str, float]
    uptake: Dict[str, Dict[str, float]]
    eigenvalues: List[complex]
    jacobian: dict
    stable_smooth: bool
    stability_ambiguous: bool
    invasion: Dict[str, float]
    invasion_without_switches: Dict[str, float]
    ceilings: Dict[str, List[Ceiling]]
    binding: Dict[str, List[str]]
    switch_on: Dict[str, bool]
    switch_margins: Dict[str, Optional[float]]
    near_switch: bool
    exists: bool
    stable: bool
    supply_yield: Dict[str, list]
    existence: List[dict]
    residual_norm: float
    D: float
    starts: int = 0
    seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def X_total(self):
        return float(sum(self.X.values()))

    @property
    def max_real_eigenvalue(self):
        return max((e.real for e in self.eigenvalues), default=None)

    def as_dict(self):
        """Plain types (json.dumps-able): complex numbers as [real, imag], inf as None."""
        out = _plain(asdict(self))
        out["X_total"] = self.X_total
        return out

    def __repr__(self):
        a = ", ".join(f"{m} {v:.4f}" for m, v in self.a.items())
        return (f"SteadyState({'+'.join(self.members) or 'washout'}: X {self.X_total:.4g} gDW/L"
                f"{', a ' + a if a else ''}; exists {self.exists}, stable {self.stable}, "
                f"max Re(eig) {self.max_real_eigenvalue})")


class SteadyStates(list):
    """The list chemostat() returns, plus `unsolved` ({subset: why no state was found}) and
    `settings` (D, feed, gas, tracked compounds, LP counts)."""

    def __init__(self, states=(), unsolved=None, settings=None):
        super().__init__(states)
        self.unsolved = dict(unsolved or {})
        self.settings = dict(settings or {})

    def of(self, *members):
        """The states whose survivors are exactly `members` (order free)."""
        want = frozenset(members)
        return [s for s in self if frozenset(s.members) == want]


# --------------------------------------------------------------------------- helpers
def _plain(o):
    if isinstance(o, Mapping):
        return {(k if isinstance(k, (str, int, float, bool)) or k is None else str(k)): _plain(v)
                for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, (complex, np.complexfloating)):
        return [float(o.real), float(o.imag)]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def _cpd_of(key):
    """'EX_cpd00027_e0', 'cpd00027_e0' or 'cpd00027' -> 'cpd00027'."""
    key = str(getattr(key, "id", key))
    if key.startswith("EX_"):
        key = key[3:]
    suffix = f"_{SHARED}"
    return key[:-len(suffix)] if key.endswith(suffix) else key


def _mm(vmax, Km, s):
    if s <= 0.0:
        return 0.0
    cap = vmax * s / (Km + s)
    return cap if cap > CAP_ZERO else 0.0


def _resolve_member(comm, member):
    key = getattr(member, "id", member)
    if isinstance(key, int) and not isinstance(key, bool):
        for m in comm.members:
            if m.index == key:
                return m
    if key not in comm.members:
        raise KeyError(f"no member {key!r}; members {[m.id for m in comm.members]}")
    return comm.members.get_by_id(key)


@dataclass(frozen=True)
class _Kin:
    label: str
    cpd: str
    reactions: Optional[Tuple[str, ...]]     # None: the member's transporters for cpd
    vmax: float
    Km: float
    source: Optional[str]


def _parse_kinetic_value(value, where):
    if isinstance(value, Mapping):
        vmax, Km = value["vmax"], value["Km"]
        source, cpd = value.get("source"), value.get("cpd")
    else:
        value = tuple(value)
        if len(value) not in (2, 3):
            raise ValueError(f"{where}: give (vmax, Km) or (vmax, Km, source), not {value!r}")
        vmax, Km = value[0], value[1]
        source, cpd = (value[2] if len(value) == 3 else None), None
    vmax, Km = float(vmax), float(Km)
    if not (math.isfinite(vmax) and vmax >= 0 and math.isfinite(Km) and Km >= 0):
        raise ValueError(f"{where}: vmax and Km must be finite and >= 0, not ({vmax}, {Km})")
    return vmax, Km, (None if source is None else str(source)), cpd


def _parse_kinetics(comm, kinetics, base_tracked):
    """{member id: [_Kin]} from the kinetics argument; reaction-tuple keys get their compound from
    the e0 compounds every listed reaction moves, among the tracked ones."""
    model = comm.util.model
    parsed, pending = {}, []
    for mkey, entries in (kinetics or {}).items():
        m = _resolve_member(comm, mkey)
        parsed.setdefault(m.id, [])
        for key, value in entries.items():
            where = f"kinetics[{m.id!r}][{key!r}]"
            vmax, Km, source, cpd = _parse_kinetic_value(value, where)
            if isinstance(key, (tuple, list)):
                rids = tuple(str(getattr(r, "id", r)) for r in key)
                missing = [r for r in rids if r not in model.reactions]
                if missing:
                    raise KeyError(f"{where}: no reactions {missing} in {model.id} (give the merged "
                                   f"model's ids; {m.id} is compartment index {m.index})")
                pending.append((m.id, rids, vmax, Km, source, cpd, where))
            else:
                c = _cpd_of(cpd if cpd is not None else key)
                if f"{c}_{SHARED}" not in model.metabolites:
                    raise KeyError(f"{where}: no shared-pool metabolite {c}_{SHARED}")
                parsed[m.id].append(_Kin(str(key), c, None, vmax, Km, source))
    tracked = set(base_tracked) | {k.cpd for ks in parsed.values() for k in ks}
    for mid, rids, vmax, Km, source, cpd, where in pending:
        if cpd is None:
            moved = None
            for rid in rids:
                e0 = {_cpd_of(met.id) for met in model.reactions.get_by_id(rid).metabolites
                      if met.compartment == SHARED}
                moved = e0 if moved is None else moved & e0
            candidates = sorted((moved or set()) & tracked) or sorted(moved or set())
            if len(candidates) != 1:
                raise ValueError(f"{where}: cannot tell which compound drives this ceiling (the "
                                 f"reactions all move {sorted(moved or [])}); give the value as "
                                 "{'vmax': ..., 'Km': ..., 'cpd': ...}")
            cpd = candidates[0]
        label = "(" + ", ".join(rids) + ")"
        parsed[mid].append(_Kin(label, _cpd_of(cpd), rids, vmax, Km, source))
        tracked.add(_cpd_of(cpd))
    return parsed, tracked


# --------------------------------------------------------------------------- member blocks
@dataclass
class _Response:
    mu: float
    uptake: Dict[str, float]           # tracked compound -> net uptake per gDW member
    caps: Dict[str, float]             # kinetic label -> ceiling
    gross: Dict[str, float]            # kinetic label -> gross import through the capped reactions
    status: str


class MemberBlock:
    """One member's own LP, built from its block of the merged community model (see the module
    docstring). respond(conc) sets the Michaelis-Menten ceilings at `conc` ({compound: mM}),
    maximises the member's biomass flux and, with tie_break="pfba", takes the minimal-total-flux
    solution at that growth rate. All rates are per gDW of the member.

    model          the block (cobra Model; solver = the community model's), objective max biomass
    pfba_model     a copy whose objective is the total flux (tie_break="pfba"), else None
    biomass        the member's biomass reaction id; mu = its flux times its biomass coefficient
    exchange       {e0 compound: block exchange id}
    tracked        the tracked compounds; closed: those this member may not take up (no kinetics)
    kinetics       [_Kin] with one ceiling row each: rows {key label: row name}, reactions_of
                   {key label: the capped reactions}
    maintenance    (ATPM reaction id, (lb, ub), where it came from) as set in the block, or None
    calls, solves  responses computed and LPs solved

    Built with kinetics=() and tracked=(...) it is the member's LP with every tracked uptake shut,
    which is how mu_at_vmax / respond can be checked against the member model it came from."""

    def __init__(self, comm, member, *, tracked=(), kinetics=(), maintenance="per_member",
                 tie_break="pfba", extracellular=None, member_bounds=None, threads=1):
        self.comm = comm
        self.member = m = _resolve_member(comm, member)
        self.id, self.index = m.id, m.index
        self.tracked = tuple(sorted(set(tracked)))
        self.tie_break = tie_break
        if tie_break not in ("pfba", None, "vertex"):
            raise ValueError(f"tie_break must be 'pfba' or None, not {tie_break!r}")
        self.calls = self.solves = 0
        # everything is written on GLPK (cheap to populate and to copy, and its copy does not go
        # through an LP file), then both LPs move to the community model's solver in one rebuild each
        self._build(extracellular or (), member_bounds or {})
        self._set_maintenance(maintenance)
        self._add_ceilings(kinetics)
        self._close_untaken()
        self.pfba_model = None
        if tie_break == "pfba":
            self.pfba_model = pf = self.model.copy()
            pf.objective = pf.problem.Objective(
                add([v for r in pf.reactions for v in (r.forward_variable, r.reverse_variable)]), direction="min")
        self._finalise(threads)

    def _finalise(self, threads):
        """Move the LP(s) to the community model's solver, copy its tolerances, set the threads and
        take the live optlang objects (they change with the solver)."""
        cmodel = self.comm.util.model
        for lp in (self.model, self.pfba_model):
            if lp is None:
                continue
            try:
                if lp.problem.__name__ != cmodel.problem.__name__:
                    lp.solver = cmodel.problem
            except Exception as exc:         # pragma: no cover
                logger.warning("%s: keeping the block on %s (%s)", self.id, lp.problem.__name__, exc)
            cfg_src, cfg = cmodel.solver.configuration, lp.solver.configuration
            for tol in ("feasibility", "optimality"):
                try:
                    setattr(cfg.tolerances, tol, getattr(cfg_src.tolerances, tol))
                except Exception:            # pragma: no cover
                    pass
            if "gurobi" in lp.problem.__name__:
                try:
                    lp.solver.problem.Params.Threads = int(threads)
                except Exception:            # pragma: no cover
                    pass
        self._bio = self.model.reactions.get_by_id(self.biomass)
        self._row = {label: self.model.constraints[name] for label, name in self.rows.items()}
        if self.pfba_model is not None:
            self._pf_bio = self.pfba_model.reactions.get_by_id(self.biomass)
            self._pf_row = {label: self.pfba_model.constraints[name] for label, name in self.rows.items()}

    # ---- construction
    def _build(self, extracellular, member_bounds):
        comm, m = self.comm, self.member
        cmodel = comm.util.model
        rxns = list(member_reactions(comm, m))
        own = {r.id for r in rxns}
        for item in extracellular:
            r = cmodel.reactions.get_by_id(getattr(item, "id", item))
            if any(met.compartment != SHARED for met in r.metabolites) or r.boundary:
                raise ValueError(f"{r.id} is not an extracellular (e0-only, non-boundary) reaction")
            if r.id not in own:
                rxns.append(r)
                own.add(r.id)
        if m.primary_biomass is None:
            raise ValueError(f"{m.id} has no biomass reaction")
        bio = cmodel.reactions.get_by_id(m.primary_biomass.id)
        bio_cpd = getattr(m.biomass_cpd, "id", m.biomass_cpd)
        coef = next((float(c) for met, c in bio.metabolites.items() if met.id == bio_cpd), 0.0)
        if coef <= 0:
            raise ValueError(f"{bio.id} does not produce {m.id}'s biomass compound {bio_cpd}")
        self.biomass, self.biomass_coef = bio.id, coef
        block = Model(f"{cmodel.id}__{m.id}")
        try:
            block.solver = "glpk"            # populate cheaply, then one rebuild (commhelper's reason)
        except Exception:                    # pragma: no cover
            pass
        mets = {}
        new = []
        for r in rxns:
            nr = Reaction(r.id, name=r.name, lower_bound=r.lower_bound, upper_bound=r.upper_bound)
            stoich = {}
            for met, c in r.metabolites.items():
                if met.id not in mets:
                    mets[met.id] = Metabolite(met.id, name=met.name, compartment=met.compartment)
                stoich[mets[met.id]] = float(c)
            nr.add_metabolites(stoich)
            new.append(nr)
        # one exchange per e0 compound the block touches; bounds from the community medium
        self.exchange = {}
        for mid in sorted(k for k, met in mets.items() if met.compartment == SHARED):
            cpd = _cpd_of(mid)
            ex_id = f"EX_{cpd}_{SHARED}"
            bounds = (0.0, 1000.0)
            if ex_id in cmodel.reactions and cmodel.reactions.get_by_id(ex_id).boundary:
                bounds = tuple(cmodel.reactions.get_by_id(ex_id).bounds)
            else:
                for r in cmodel.metabolites.get_by_id(mid).reactions:
                    if r.boundary:
                        bounds = tuple(r.bounds)
                        ex_id = r.id
                        break
            if ex_id in own:                 # a member boundary on e0 (unusual): keep the name free
                ex_id = f"EX_{cpd}_{SHARED}_block"
            ex = Reaction(ex_id, lower_bound=bounds[0], upper_bound=bounds[1])
            ex.add_metabolites({mets[mid]: -1.0})
            new.append(ex)
            self.exchange[cpd] = ex_id
        for key, b in member_bounds.items():
            cpd = _cpd_of(key)
            if cpd in self.tracked:
                raise ValueError(f"member_bounds[{m.id!r}][{key!r}]: {cpd} is tracked; its uptake is set "
                                 "by kinetics (or closed without them)")
            if cpd not in self.exchange:
                raise KeyError(f"member_bounds[{m.id!r}][{key!r}]: {m.id} touches no {cpd}_{SHARED}")
            lb, ub = (float(b), None) if not isinstance(b, (tuple, list)) else (float(b[0]), float(b[1]))
            for nr in new:
                if nr.id == self.exchange[cpd]:
                    nr.lower_bound = lb
                    if ub is not None:
                        nr.upper_bound = ub
        block.add_metabolites(list(mets.values()))
        block.add_reactions(new)
        # the member's biomass leaves through its drain (shut in the community when drains are closed)
        drain = getattr(m, "biomass_drain", None)
        if drain is not None and drain.id in block.reactions:
            block.reactions.get_by_id(drain.id).bounds = (0.0, 1000.0)
        else:
            sink = Reaction(f"SK_{bio_cpd}", lower_bound=0.0, upper_bound=1000.0)
            sink.add_metabolites({block.metabolites.get_by_id(bio_cpd): -1.0})
            block.add_reactions([sink])
        self.model = block
        bio_rxn = block.reactions.get_by_id(self.biomass)
        block.objective = block.problem.Objective(bio_rxn.flux_expression, direction="max")

    def _set_maintenance(self, maintenance):
        comm, m, block = self.comm, self.member, self.model
        registered = {c.member: c for c in member_caps(comm, kind="maintenance")}
        atpm = atp_maintenance_reaction(comm, m, required=False)
        if atpm is not None and atpm.id not in block.reactions:
            atpm = None
        self.maintenance = None
        if isinstance(maintenance, str) and maintenance.lower() in ("free", "none"):
            maintenance = None
        if maintenance is None:
            if atpm is not None:
                r = block.reactions.get_by_id(atpm.id)
                r.bounds = (0.0, max(r.upper_bound, 1000.0))
                self.maintenance = (r.id, tuple(r.bounds), "released")
            return
        if isinstance(maintenance, Mapping):
            given = {(_resolve_member(comm, k).id): v for k, v in maintenance.items()}
            if m.id in given:
                if atpm is None:
                    raise LookupError(f"maintenance for {m.id}: it has no ATP maintenance reaction")
                A = float(given[m.id])
                if not (math.isfinite(A) and A >= 0):
                    raise ValueError(f"maintenance[{m.id!r}] must be a finite number >= 0, not {A}")
                r = block.reactions.get_by_id(atpm.id)
                r.bounds = (A, max(r.upper_bound, A, 1000.0) if r.upper_bound < A else r.upper_bound)
                self.maintenance = (r.id, tuple(r.bounds), "given")
                return
        elif maintenance != "per_member":
            raise ValueError(f"maintenance must be 'per_member', a {{member: A_m}} dict or None, not {maintenance!r}")
        if m.id in registered:
            rec = registered[m.id]
            r = block.reactions.get_by_id(rec.reactions[0])
            r.lower_bound = 0.0
            if r.upper_bound < rec.rate:
                r.upper_bound = max(rec.rate, 1000.0)
            r.lower_bound = rec.rate
            self.maintenance = (r.id, tuple(r.bounds), f"registered {rec.name}")
        elif atpm is not None:
            r = block.reactions.get_by_id(atpm.id)
            self.maintenance = (r.id, tuple(r.bounds), "as built")

    def _add_ceilings(self, kinetics):
        block = self.model
        self.kinetics = list(kinetics)
        self.rows, self.reactions_of = {}, {}
        rows = []
        for k in self.kinetics:
            reactions = list(k.reactions) if k.reactions is not None else None
            if k.cpd not in self.exchange:
                raise ValueError(f"kinetics[{self.id!r}][{k.label}]: {self.id} has no transporter for "
                                 f"{k.cpd}_{SHARED} (no reaction of its block touches it)")
            if reactions is not None:
                missing = [r for r in reactions if r not in block.reactions]
                if missing:
                    raise ValueError(f"kinetics[{self.id!r}][{k.label}]: {missing} are not reactions of "
                                     f"{self.id}'s block (its own reactions or extracellular ones attributed to it)")
            expr = member_uptake_expression(block, self.index, k.cpd, reactions=reactions, gross=True)
            name = f"chemo_cap_{self.id}_{len(rows)}"
            rows.append(block.problem.Constraint(expr, ub=k.vmax, name=name))
            self.rows[k.label] = name
            if reactions is None:
                reactions = [r.id for r in member_transporters(block, self.index, k.cpd)]
            self.reactions_of[k.label] = tuple(reactions)
        # registered static uptake caps (commmembers.cap_member_uptake): q per gDW of the member
        for rec in member_caps(self.comm, self.member, kind="uptake"):
            present = [r for r in rec.reactions if r in block.reactions]
            if not present:
                continue
            expr = member_uptake_expression(block, self.index, rec.cpd, reactions=present, gross=True)
            rows.append(block.problem.Constraint(expr, ub=rec.rate, name=f"chemo_q_{rec.name}"))
        if rows:
            block.add_cons_vars(rows)

    def _close_untaken(self):
        """Tracked compounds without kinetics for this member: no net uptake from the pool."""
        taken = {k.cpd for k in self.kinetics}
        self.closed = []
        for cpd in self.tracked:
            if cpd in taken or cpd not in self.exchange:
                if cpd in taken and cpd in self.exchange:
                    ex = self.model.reactions.get_by_id(self.exchange[cpd])
                    ex.lower_bound = -1000.0
                continue
            ex = self.model.reactions.get_by_id(self.exchange[cpd])
            ex.lower_bound = 0.0
            if ex.upper_bound < 0:
                ex.upper_bound = 0.0
            self.closed.append(cpd)

    # ---- solving
    def mu_at_vmax(self):
        """Growth with every kinetic ceiling at vmax (S -> infinity): no state can have this member
        surviving if this is below D."""
        caps = {k.label: k.vmax for k in self.kinetics}
        return self._solve(caps).mu

    def caps_at(self, conc):
        return {k.label: _mm(k.vmax, k.Km, float(conc.get(k.cpd, 0.0))) for k in self.kinetics}

    def respond(self, conc):
        return self._solve(self.caps_at(conc))

    def _solve(self, caps):
        self.calls += 1
        for label, cap in caps.items():
            self._row[label].ub = cap
        block = self.model
        status = block.solver.optimize()
        self.solves += 1
        if status != "optimal":
            return _Response(0.0, {c: 0.0 for c in self.tracked}, caps, {k: 0.0 for k in caps}, f"infeasible ({status})")
        v_bio = float(self._bio.forward_variable.primal - self._bio.reverse_variable.primal)
        mu = v_bio * self.biomass_coef
        if self.pfba_model is not None and v_bio > MU_ZERO:
            pf = self.pfba_model
            for label, cap in caps.items():
                self._pf_row[label].ub = cap
            fv = self._pf_bio.forward_variable
            fv.lb = v_bio * (1.0 - PFBA_RELAX)
            status2 = pf.solver.optimize()
            self.solves += 1
            if status2 == "optimal":
                out = self._read(pf, self._pf_row, mu, caps)
                fv.lb = 0.0
                return out
            fv.lb = 0.0                      # numerical trouble: keep the growth vertex
            logger.debug("%s: pFBA LP %s; using the growth vertex", self.id, status2)
        return self._read(block, self._row, mu, caps)

    def _read(self, model, rows, mu, caps):
        """The response from the solution in the solver now (read before any bound changes)."""
        uptake = {}
        for cpd in self.tracked:
            ex = self.exchange.get(cpd)
            if ex is None:
                uptake[cpd] = 0.0
                continue
            r = model.reactions.get_by_id(ex)
            uptake[cpd] = -float(r.forward_variable.primal - r.reverse_variable.primal)
        gross = {label: float(rows[label].primal) for label in rows}
        return _Response(mu if mu > MU_ZERO else 0.0, uptake, caps, gross, "optimal")


def member_block(comm, member, **kwargs) -> MemberBlock:
    """The member's own LP as chemostat() builds it (for inspection or reuse): see MemberBlock."""
    return MemberBlock(comm, member, **kwargs)


# --------------------------------------------------------------------------- the reactor
class _Reactor:
    def __init__(self, blocks, tracked, feed, gas, D, switches):
        self.blocks = blocks                         # {member id: MemberBlock}
        self.tracked = tuple(tracked)                # ordered compound ids
        self.feed = {c: float(feed.get(c, 0.0)) for c in self.tracked}
        self.gas = dict(gas)                         # {cpd: (kLa, C*)}
        self.D = float(D)
        self.switches = dict(switches)
        self.inputs = {mid: tuple(sorted({k.cpd for k in b.kinetics})) for mid, b in blocks.items()}
        self._cache = {mid: {} for mid in blocks}

    def response(self, mid, conc, honour=False):
        """Member mid's response at conc (switch honoured or not); cached on its kinetic inputs."""
        if honour and mid in self.switches and not self.switches[mid](conc):
            b = self.blocks[mid]
            return _Response(0.0, {c: 0.0 for c in b.tracked}, b.caps_at(conc),
                             {k.label: 0.0 for k in b.kinetics}, "switched off")
        key = tuple(float(f"{max(conc.get(c, 0.0), 0.0):.15g}") for c in self.inputs[mid])
        cache = self._cache[mid]
        hit = cache.get(key)
        if hit is None:
            hit = self.blocks[mid].respond({c: max(conc.get(c, 0.0), 0.0) for c in self.tracked})
            if len(cache) > 50000:
                cache.clear()
            cache[key] = hit
        return hit

    def rhs(self, X, S, members, honour=False):
        """(dX {member: gDW/L/h}, dS {cpd: mM/h}, responses) for biomasses X (members listed) and
        residuals S."""
        conc = {c: max(S.get(c, 0.0), 0.0) for c in self.tracked}
        resp = {m: self.response(m, conc, honour) for m in members}
        dX = {m: (resp[m].mu - self.D) * X.get(m, 0.0) for m in members}
        dS = {}
        for c in self.tracked:
            use = sum(X.get(m, 0.0) * resp[m].uptake.get(c, 0.0) for m in members)
            if c in self.gas:
                kla, cstar = self.gas[c]
                dS[c] = kla * (cstar - conc[c]) - use
            else:
                dS[c] = self.D * (self.feed[c] - conc[c]) - use
        return dX, dS, resp

    def scale(self, c):
        if c in self.gas:
            kla, cstar = self.gas[c]
            return kla * cstar if kla * cstar > 0 else 1.0
        return self.D * (self.feed[c] if self.feed[c] > 0 else 1.0)

    def inflow(self, c):
        """Supply of c that does not come from a member, mmol/L/h at S = 0 (feed, or gas transfer)."""
        if c in self.gas:
            kla, cstar = self.gas[c]
            return kla * cstar
        return self.D * self.feed[c]

    def washout_conc(self):
        return {c: (self.gas[c][1] if c in self.gas else self.feed[c]) for c in self.tracked}

    def lp_counts(self):
        return {mid: {"responses": b.calls, "lp_solves": b.solves} for mid, b in self.blocks.items()}


# --------------------------------------------------------------------------- root finding
def _names(subset, tracked):
    return [("X", m) for m in subset] + [("S", c) for c in tracked]


def _residual_fn(R, subset, names):
    D = R.D

    def f(z):
        vals = np.exp(z)
        X = {n: v for (kind, n), v in zip(names, vals) if kind == "X"}
        S = {n: v for (kind, n), v in zip(names, vals) if kind == "S"}
        dX, dS, resp = R.rhs(X, S, subset)
        out = []
        for kind, n in names:
            out.append((resp[n].mu - D) / D if kind == "X" else dS[n] / R.scale(n))
        return np.array(out)
    return f


def _seeds(R, subset, n_starts):
    """Generic supply x yield starting points for `subset`."""
    D, tracked = R.D, R.tracked
    est_u, est_r, mu_ref = {}, {}, {}
    for m in subset:
        b = R.blocks[m]
        resp = b._solve({k.label: k.vmax for k in b.kinetics})
        scale = D / resp.mu if resp.mu > MU_ZERO else 1.0
        mu_ref[m] = resp.mu
        est_u[m] = {k.label: max(resp.gross.get(k.label, 0.0), 0.0) * scale for k in b.kinetics}
        est_r[m] = {c: max(-resp.uptake.get(c, 0.0), 0.0) * scale for c in tracked}
    # residual of each consumed compound: the most competitive consumer's R*-type estimate
    S = {}
    for c in tracked:
        guesses = []
        for m in subset:
            for k in R.blocks[m].kinetics:
                if k.cpd != c:
                    continue
                u = est_u[m].get(k.label, 0.0)
                if 0 < u < k.vmax:
                    guesses.append(max(k.Km * u / (k.vmax - u), 1e-9))
                elif u <= 0:
                    guesses.append(max(k.Km, 1e-6))
                else:
                    guesses.append(max(10 * k.Km, 1e-6))
        S[c] = min(guesses) if guesses else None
    X = {m: 0.1 for m in subset}
    for _ in range(3):
        prod = {c: sum(X[m] * est_r[m].get(c, 0.0) for m in subset) for c in tracked}
        newX = {}
        for m in subset:
            cands = []
            for k in R.blocks[m].kinetics:
                u = est_u[m].get(k.label, 0.0)
                if u <= 0:
                    continue
                consumers = sum(1 for j in subset for kk in R.blocks[j].kinetics
                                if kk.cpd == k.cpd and est_u[j].get(kk.label, 0.0) > 0)
                supply = R.inflow(k.cpd) + prod.get(k.cpd, 0.0) - X[m] * est_r[m].get(k.cpd, 0.0)
                if supply > 0:
                    cands.append((D / u) * supply / max(consumers, 1) / D)
            newX[m] = min(cands) if cands else 0.1
            newX[m] = min(max(newX[m], 1e-6), 1e3)
        X = newX
    prod = {c: sum(X[m] * est_r[m].get(c, 0.0) for m in subset) for c in tracked}
    for c in tracked:
        if S[c] is None:
            base = R.gas[c][1] if c in R.gas else R.feed[c] + prod[c] / D
            S[c] = max(base, 1e-9)
        cap = (R.gas[c][1] if c in R.gas else R.feed[c] + prod[c] / D)
        if cap > 0:
            S[c] = min(S[c], cap)
        S[c] = max(S[c], 1e-10)
    base = {**{("X", m): X[m] for m in subset}, **{("S", c): S[c] for c in tracked}}
    variants = [(1.0, 1.0), (10.0, 1.0), (0.1, 0.5), (1.0, 2.0), (30.0, 0.7), (0.03, 1.5)]
    out = []
    for sf, xf in variants[:max(n_starts, 1)]:
        out.append({k: (v * xf if k[0] == "X" else v * sf) for k, v in base.items()})
    return out


def _dedupe(roots, tol=1e-4):
    out = []
    for r in roots:
        dup = False
        for o in out:
            keys = set(r["values"]) | set(o["values"])
            if all(abs(math.log(max(r["values"].get(k, 0.0), 1e-15)) - math.log(max(o["values"].get(k, 0.0), 1e-15))) < tol
                   for k in keys if r["values"].get(k, 0.0) > 1e-10 or o["values"].get(k, 0.0) > 1e-10):
                dup = True
                break
        if not dup:
            out.append(r)
    return out


def _solve_subset(R, subset, seeds, *, tol, stop_after, ls_options):
    from scipy.optimize import least_squares
    names = _names(subset, R.tracked)
    f = _residual_fn(R, subset, names)
    lo = np.full(len(names), math.log(LOG_FLOOR))
    hi = np.full(len(names), math.log(LOG_CEIL))
    roots, best, tried = [], None, 0
    opts = dict(method="trf", diff_step=1e-6, xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=400)
    opts.update(ls_options or {})
    for seed in seeds:
        tried += 1
        z0 = np.array([math.log(min(max(seed.get(k, 1e-6), LOG_FLOOR * 10), LOG_CEIL / 10)) for k in names])
        try:
            sol = least_squares(f, z0, bounds=(lo, hi), **opts)
        except Exception as exc:                                   # noqa: BLE001
            logger.debug("subset %s: least_squares failed: %r", subset, exc)
            continue
        res = f(sol.x)
        norm = float(np.max(np.abs(res))) if len(res) else 0.0
        values = {k: float(v) for k, v in zip(names, np.exp(sol.x))}
        # a residual pinned at the log floor is a compound nobody supplies: its root is S = 0
        values = {k: (0.0 if k[0] == "S" and v <= LOG_FLOOR * (1 + 1e-6) else v) for k, v in values.items()}
        ok = norm <= tol and all(values[("X", m)] >= X_SURVIVOR_MIN for m in subset)
        if best is None or norm < best:
            best = norm
        if ok:
            roots.append({"values": values, "norm": norm, "nfev": int(sol.nfev)})
            if len(roots) >= stop_after:
                break
    return _dedupe(roots), best, tried


# --------------------------------------------------------------------------- diagnostics
def _jacobian(R, subset, X, S, rel=1e-3):
    """Central / forward / backward differences of the reduced ODE (survivors' X and all S),
    switches ignored."""
    names = _names(subset, R.tracked)
    x0 = np.array([X[n] if kind == "X" else S[n] for kind, n in names])

    def F(x):
        Xd = {n: v for (kind, n), v in zip(names, x) if kind == "X"}
        Sd = {n: v for (kind, n), v in zip(names, x) if kind == "S"}
        dX, dS, _ = R.rhs(Xd, Sd, subset)
        return np.array([dX[n] if kind == "X" else dS[n] for kind, n in names])

    n = len(names)
    if n == 0:
        return {"names": [], "J_central": [], "eig": [], "eig_forward": [], "eig_backward": []}
    f0 = F(x0)
    Jc, Jf, Jb = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    for j in range(n):
        h = rel * max(abs(x0[j]), 1e-9)
        xp, xm = x0.copy(), x0.copy()
        xp[j] += h
        xm[j] = max(x0[j] - h, 0.0)
        hm = x0[j] - xm[j]
        fp, fm = F(xp), F(xm)
        Jc[:, j] = (fp - fm) / (h + hm)
        Jf[:, j] = (fp - f0) / h
        Jb[:, j] = (f0 - fm) / hm if hm > 0 else Jf[:, j]
    ev, evf, evb = (np.linalg.eigvals(J) for J in (Jc, Jf, Jb))
    srt = lambda e: sorted((complex(x) for x in e), key=lambda z: (z.real, z.imag))  # noqa: E731
    return {"names": [f"{k}:{v}" for k, v in names], "J_central": Jc.tolist(),
            "eig": srt(ev), "eig_forward": srt(evf), "eig_backward": srt(evb),
            "one_sided_disagreement_rel": float(np.max(np.abs(Jf - Jb)) / max(np.max(np.abs(Jc)), 1e-12)),
            "step_rel": rel}


def _ceilings(R, mid, conc, resp):
    b = R.blocks[mid]
    out = []
    for k in b.kinetics:
        cap = resp.caps.get(k.label, 0.0)
        u = resp.gross.get(k.label, 0.0)
        out.append(Ceiling(key=k.label, cpd=k.cpd, reactions=b.reactions_of.get(k.label, ()),
                           vmax=k.vmax, Km=k.Km, source=k.source, residual_mM=float(conc.get(k.cpd, 0.0)),
                           cap=cap, uptake=u, binding=bool(cap > BINDING_MIN and u >= cap * (1 - 1e-6) - 1e-12)))
    return out


def _gross_release(R, j, c, resp):
    """Member j's release of c into the pool, mmol per gDW of j per h: net release, or, when j has
    a ceiling on c, its capped import minus its net uptake (the release a reaction-set ceiling
    exists for, e.g. glucose freed by an extracellular-facing hydrolase, which j may take back)."""
    net = resp[j].uptake.get(c, 0.0)
    capped = [resp[j].gross.get(k.label, 0.0) for k in R.blocks[j].kinetics if k.cpd == c]
    if capped:
        return max(max(capped) - net, 0.0)
    return max(-net, 0.0)


def _supply_yield(R, subset, X, conc, resp, ceilings):
    """Each survivor's binding ceilings as supply x yield: X_m = (D / q) * share * supply."""
    D = R.D
    out = {}
    for m in subset:
        rows = []
        for cl in ceilings[m]:
            if not cl.binding or cl.uptake <= 0:
                continue
            c = cl.cpd
            released = {j: float(X[j] * _gross_release(R, j, c, resp)) for j in subset
                        if X[j] * _gross_release(R, j, c, resp) > 0}
            if c in R.gas:
                kla, cstar = R.gas[c]
                outside = {"gas": float(kla * (cstar - conc[c]))}
            else:
                outside = {"feed": float(D * R.feed[c])}
            supply = sum(outside.values()) + sum(released.values())
            taken = float(X[m] * cl.uptake)
            rows.append({
                "cpd": c, "ceiling": cl.key, "residual_mM": float(conc[c]),
                "uptake_per_gDW": float(cl.uptake), "net_uptake_per_gDW": float(resp[m].uptake.get(c, 0.0)),
                "yield_gDW_per_mmol": float(D / cl.uptake),
                "uptake_mmol_per_L_h": taken,
                "supply_mmol_per_L_h": {**outside, "released_by": released, "total": float(supply)},
                "supply_mM_equivalent": float(supply / D),
                "share_of_supply": float(taken / supply) if supply > 0 else None,
                "washed_out_mmol_per_L_h": 0.0 if c in R.gas else float(D * conc[c]),
                # X_m = Y * share * supply / D  (= Y (S_in - S) for a single consumer of a fed compound)
                "X_from_yield_x_share_x_supply": float((D / cl.uptake) * (taken / supply) * supply / D)
                if supply > 0 else None})
        out[m] = rows
    return out


def _existence(R, subset, X, conc, resp, ceilings):
    """For each survivor with a ConcentrationSwitch: the consumer biomass needed to hold the
    compound on the ON side of its threshold, if the consumers' ceilings bind at the threshold."""
    out = []
    for m in subset:
        sw = R.switches.get(m)
        if not isinstance(sw, ConcentrationSwitch):
            continue
        c, thr = sw.cpd, sw.threshold
        rec = {"member": m, "cpd": c, "threshold_mM": thr, "off": sw.off, "residual_mM": conc.get(c),
               "margin_mM": sw.margin(conc), "on": sw(conc)}
        if c not in R.tracked:
            rec["criterion"] = None
            rec["note"] = f"{c} is not a tracked compound: its residual is not modelled"
            out.append(rec)
            continue
        if c in R.gas:
            kla, cstar = R.gas[c]
            supply = float(kla * (cstar - thr))
            supply_terms = {"gas_transfer_at_threshold": supply}
        else:
            released = float(sum(X[j] * _gross_release(R, j, c, resp) for j in subset))
            supply = float(R.D * (R.feed[c] - thr) + released)
            supply_terms = {"feed_at_threshold": float(R.D * (R.feed[c] - thr)), "released_at_state": released}
        consumers = {}
        for j in subset:
            for k in R.blocks[j].kinetics:
                if k.cpd == c:
                    q = _mm(k.vmax, k.Km, thr)
                    bind = next((cl.binding for cl in ceilings[j] if cl.key == k.label), None)
                    if j not in consumers:
                        consumers[j] = {"q_at_threshold": q, "binding_at_state": bind, "X": float(X[j])}
                    elif q > consumers[j]["q_at_threshold"]:     # several ceilings on c: the loosest
                        consumers[j].update(q_at_threshold=q, binding_at_state=bind)
        capacity = sum(v["X"] * v["q_at_threshold"] for v in consumers.values())
        bound = "X_needed" if sw.off == "above" else "X_allowed"
        for j, v in consumers.items():
            others = capacity - v["X"] * v["q_at_threshold"]
            v[bound] = (supply - others) / v["q_at_threshold"] if v["q_at_threshold"] > 0 else math.inf
            v["satisfied"] = (v["X"] >= v[bound]) if sw.off == "above" else (v["X"] <= v[bound])
        rec.update({
            "supply_at_threshold_mmol_per_L_h": supply, "supply_terms": supply_terms,
            "uptake_capacity_at_threshold_mmol_per_L_h": capacity,
            "consumers": consumers,
            "criterion": (f"sum_k X_k q_k({c} = {thr:g} mM) {'>=' if sw.off == 'above' else '<='} "
                          f"{supply:.6g} mmol/L/h"),
            "assumes": "every consumer is limited by its ceiling on this compound at the threshold "
                       "(binding_at_state tells whether it is at the state found)"})
        if not consumers:
            rec["note"] = f"no surviving member has kinetics on {c}: nothing can move it across the threshold"
        out.append(rec)
    return out


def _analyse(R, subset, everyone, X, S, norm, D, *, rel_step):
    t = time.time()
    conc = {c: max(S.get(c, 0.0), 0.0) for c in R.tracked}
    resp = {m: R.response(m, conc) for m in everyone}
    resp_sw = {m: R.response(m, conc, honour=True) for m in everyone}
    jac = _jacobian(R, subset, X, S, rel=rel_step)
    eig = jac["eig"]
    re = [e.real for e in eig]
    ref, rbf = [e.real for e in jac["eig_forward"]], [e.real for e in jac["eig_backward"]]
    neg = lambda xs: all(x < 0 for x in xs)  # noqa: E731
    stable_smooth = neg(re) and neg(ref) and neg(rbf)
    ambiguous = not (neg(re) == neg(ref) == neg(rbf))
    absent = [m for m in everyone if m not in subset]
    invasion = {m: resp_sw[m].mu - D for m in absent}
    invasion_free = {m: resp[m].mu - D for m in absent}
    ceilings = {m: _ceilings(R, m, conc, resp[m]) for m in subset}
    binding = {m: [cl.key for cl in ceilings[m] if cl.binding] for m in subset}
    switch_on = {m: bool(R.switches[m](conc)) for m in everyone if m in R.switches}
    margins = {m: (R.switches[m].margin(conc) if isinstance(R.switches[m], ConcentrationSwitch) else None)
               for m in everyone if m in R.switches}
    near = False
    for m in subset:
        sw = R.switches.get(m)
        if isinstance(sw, ConcentrationSwitch) and sw.cpd in conc:
            if abs(margins[m]) <= 3 * rel_step * max(conc[sw.cpd], 1e-12) + 1e-12:
                near = True
    exists = all(switch_on.get(m, True) for m in subset)
    stable = bool(exists and stable_smooth and all(v < 0 for v in invasion.values()) and not near)
    Xall = {m: (X.get(m, 0.0) if m in subset else 0.0) for m in everyone}
    tot = sum(X[m] for m in subset)
    a = {m: X[m] / tot for m in subset} if tot > 0 else {}
    warnings = []
    if ambiguous:
        warnings.append("central and one-sided Jacobians disagree on stability")
    if near:
        warnings.append("a survivor's switch threshold is within the finite-difference reach")
    for m in subset:
        if resp[m].status != "optimal":
            warnings.append(f"{m}: LP {resp[m].status} at the state")
    return SteadyState(
        members=tuple(subset), X=Xall, residuals=dict(conc), a=a,
        mu={m: resp[m].mu for m in everyone},
        uptake={m: dict(resp[m].uptake) for m in everyone},
        eigenvalues=eig, jacobian=jac, stable_smooth=bool(stable_smooth), stability_ambiguous=bool(ambiguous),
        invasion=invasion, invasion_without_switches=invasion_free, ceilings=ceilings, binding=binding,
        switch_on=switch_on, switch_margins=margins, near_switch=near, exists=bool(exists), stable=stable,
        supply_yield=_supply_yield(R, subset, X, conc, resp, ceilings),
        existence=_existence(R, subset, X, conc, resp, ceilings),
        residual_norm=float(norm), D=D, seconds=round(time.time() - t, 2), warnings=warnings)


# --------------------------------------------------------------------------- entry point
def chemostat(comm, D, feed, *, kinetics, gas=None, switches=None, maintenance="per_member",
              tie_break="pfba", members=None, subsets=None, extracellular=None, member_bounds=None,
              seeds=None, warm=None, n_starts=4, stop_after=2, tol=1e-7, jacobian_step=1e-3,
              threads=1, least_squares_options=None) -> SteadyStates:
    """Steady states of the community in a chemostat at dilution rate D (1/h): every subset of
    surviving members grows at mu = D, the others are washed out. Each member is its own LP (its
    block of the merged model, fluxes per gDW of that member) maximising its own growth under
    Michaelis-Menten ceilings at the residual concentrations; the members are coupled only through
    the reactor balances D (S_in - S) = sum_m X_m q_m(S) (and kLa (C* - C) for a gas). See the
    module docstring for the model and the member blocks.

    Read the composition as SUPPLY x YIELD: mu = D removes the max-growth selection of
    balanced-growth LPs, so each X_m is the member's yield on its limiting supply times its share of
    that supply (SteadyState.supply_yield); kinetics set residuals, existence and stability, not the
    composition. A result is only as good as the member yields and maintenance and the partition of
    shared substrates (who takes the compounds a member releases into the pool).

    D            dilution rate, 1/h
    feed         {compound: mM} in the inflow ("cpd00027", "cpd00027_e0" or "EX_cpd00027_e0")
    kinetics     {member: {exchange id or compound, or (reaction ids,): (vmax, Km[, source])}}:
                 Michaelis-Menten ceilings per gDW of the member (vmax mmol/gDW_member/h, Km mM,
                 source a free-text provenance); a value may be {"vmax", "Km", "source", "cpd"}
                 (cpd names the compound of a reaction-tuple key when it is ambiguous)
    gas          {compound: (kLa 1/h, C* mM)}: gas-liquid transfer kLa (C* - C) instead of dilution
    switches     {member: callable(conc) -> bool}; ConcentrationSwitch for margins and existence
                 criteria
    maintenance  "per_member" (default), {member: A_m mmol ATP/gDW_member/h} or None
    tie_break    "pfba" (default) or None
    members      the members in the reactor (default: all); the others are left out entirely
    subsets      only these surviving sets (iterables of member ids); default every subset of
                 `members`, the empty one (washout) included
    extracellular {member: [e0-only reaction ids]} attributed to the member's block (default: the
                 id-suffix rule, MemberInfo.extracellular)
    member_bounds {member: {untracked exchange or compound: lb or (lb, ub)}}: per-member static
                 bounds (mmol/gDW_member/h) on compounds the feed does not track
    seeds        extra starts: [{member or compound: value}] (X in gDW/L, S in mM), tried first
    warm         SteadyStates (or a list of SteadyState) from an earlier call: each state's X and
                 residuals seed the same subset (continuation along a parameter)
    n_starts / stop_after / tol   generic starts per subset (at most 6: the supply x yield seed and
                 five rescalings of it), successes that end a subset's search, acceptance on
                 max |scaled residual|
    jacobian_step  relative finite-difference step of the stability Jacobian
    threads      solver threads per member LP (Gurobi)

    Returns SteadyStates (a list of SteadyState, in subset order: washout, singles, pairs, ...),
    with .unsolved {subset: reason} and .settings."""
    t0 = time.time()
    D = float(D)
    if not (math.isfinite(D) and D > 0):
        raise ValueError(f"D must be a finite number > 0, not {D}")
    gas = {_cpd_of(k): (float(v[0]), float(v[1])) for k, v in (gas or {}).items()}
    for c, (kla, cstar) in gas.items():
        if not (kla > 0 and cstar >= 0):
            raise ValueError(f"gas[{c!r}]: kLa must be > 0 and C* >= 0, not ({kla}, {cstar})")
    feed = {_cpd_of(k): float(v) for k, v in (feed or {}).items()}
    for c, v in feed.items():
        if not (math.isfinite(v) and v >= 0):
            raise ValueError(f"feed[{c!r}] must be a finite concentration >= 0, not {v}")
        if c in gas:
            raise ValueError(f"{c} is a gas (gas=): its supply is kLa (C* - C), leave it out of feed")
    model = comm.util.model
    for c in list(feed) + list(gas):
        if f"{c}_{SHARED}" not in model.metabolites:
            raise KeyError(f"no shared-pool metabolite {c}_{SHARED} in {model.id}")
    everyone = [m.id for m in comm.members] if members is None else [_resolve_member(comm, m).id for m in members]
    if len(set(everyone)) != len(everyone):
        raise ValueError(f"members lists a member twice: {everyone}")
    kin, tracked = _parse_kinetics(comm, kinetics, set(feed) | set(gas))
    kin = {k: v for k, v in kin.items() if k in everyone}          # members= leaves the others out
    tracked = tuple(sorted(tracked))
    switches = {(_resolve_member(comm, k).id): v for k, v in (switches or {}).items()}
    for k, v in switches.items():
        if not callable(v):
            raise TypeError(f"switches[{k!r}] is not callable")
    extracellular = extracellular if extracellular is not None else \
        {info.id: list(info.extracellular) for info in member_table(comm).members}
    extracellular = {(_resolve_member(comm, k).id): v for k, v in extracellular.items()}
    member_bounds = {(_resolve_member(comm, k).id): v for k, v in (member_bounds or {}).items()}

    blocks = {}
    for mid in everyone:
        blocks[mid] = MemberBlock(comm, mid, tracked=tracked, kinetics=kin.get(mid, ()),
                                  maintenance=maintenance, tie_break=tie_break,
                                  extracellular=extracellular.get(mid, ()),
                                  member_bounds=member_bounds.get(mid, {}), threads=threads)
    R = _Reactor(blocks, tracked, feed, gas, D, switches)
    reach = {mid: blocks[mid].mu_at_vmax() for mid in everyone}

    if subsets is None:
        subsets = [c for r in range(len(everyone) + 1) for c in itertools.combinations(everyone, r)]
    else:
        subsets = [tuple(m for m in everyone if m in {_resolve_member(comm, x).id for x in s}) for s in subsets]
    warm_by = {}
    for st in (warm or []):
        warm_by.setdefault(frozenset(st.members), []).append(st)
    states, unsolved = [], {}
    for subset in subsets:
        t = time.time()
        if not subset:
            conc = R.washout_conc()
            st = _analyse(R, (), everyone, {}, conc, 0.0, D, rel_step=jacobian_step)
            st.seconds = round(time.time() - t, 2)
            states.append(st)
            continue
        slow = [m for m in subset if reach[m] < D * (1 - 1e-9)]
        if slow:
            unsolved[subset] = (f"{slow} cannot grow at D = {D} even with every ceiling at vmax "
                                f"(mu {', '.join(f'{m} {reach[m]:.4g}' for m in slow)})")
            continue
        starts = []
        for st in warm_by.get(frozenset(subset), []):
            start = {("X", m): st.X[m] for m in subset if st.X.get(m, 0.0) > 0}
            start.update({("S", c): st.residuals[c] for c in tracked if st.residuals.get(c, 0.0) > 0})
            starts.append(start)                     # missing / zero entries: from the generic seed
        for sd in (seeds or []):
            if set(k for k in sd if k in everyone) == set(subset) or not any(k in everyone for k in sd):
                s = {("X", m): float(sd[m]) for m in subset if m in sd}
                s.update({("S", _cpd_of(k)): float(v) for k, v in sd.items() if k not in everyone})
                starts.append(s)
        generic = _seeds(R, subset, n_starts)
        for g in generic:
            for k, v in g.items():
                for s in starts:
                    s.setdefault(k, v)
        starts += generic
        roots, best, tried = _solve_subset(R, subset, starts, tol=tol, stop_after=stop_after,
                                           ls_options=least_squares_options)
        if not roots:
            unsolved[subset] = f"no root from {tried} starts (best max |scaled residual| {best})"
            continue
        for r in roots:
            X = {m: r["values"][("X", m)] for m in subset}
            S = {c: r["values"][("S", c)] for c in tracked}
            st = _analyse(R, subset, everyone, X, S, r["norm"], D, rel_step=jacobian_step)
            st.starts = tried
            st.seconds = round(time.time() - t, 2)
            states.append(st)
    settings = {"D": D, "feed": feed, "gas": gas, "tracked": tracked, "members": everyone,
                "kinetics": {m: [asdict(k) for k in ks] for m, ks in kin.items()},
                "maintenance": {m: b.maintenance for m, b in blocks.items()},
                "closed_uptakes": {m: b.closed for m, b in blocks.items()},
                "tie_break": tie_break, "mu_at_vmax": reach, "lp": R.lp_counts(),
                "seconds": round(time.time() - t0, 1)}
    return SteadyStates(states, unsolved, settings)
