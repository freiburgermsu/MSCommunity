# -*- coding: utf-8 -*-
"""Formal validation of mscommunity.commcycles on synthetic communities with planted cycles.

A seeded generator builds small communities (2-4 members, shared pool e0, member compartments
c1..cn, ModelSEED-style ids) directly in cobra, in the layout build_from_species_models produces
(member biomass bio2..bioN, drains SK_cpd11416_c<i>, community biomass bio1 and its sink), with
one PLANTED cycle of known type, or none:

  T1   proton-motive transport cycle across members (CommPhitting type): an M/H+ antiporter in
       member A and an M/H+ symporter in member B run M through e0 in opposite directions; an
       M <=> N interconversion and N uniports close the loop; each member's ATP synthase cashes
       the protons it pumped
  T2   cross-member redox cycle (nitrate/nitrite type): A turns M into N making ATP, B turns N
       back (making ATP or not); with 3+ members sometimes a 3-step chain M -> N -> Z -> M
  T3   GTP loop that pays off only by feeding biomass: A charges GTP turning M into N, B turns N
       back, A's biomass consumes GTP; GTP is NOT linked to ATP, so no ATP test can see it (the
       limitation named in commcycles' docstring)
  T3L  the same with nucleoside-diphosphate kinase (ATP + GDP <=> ADP + GTP) in A
  T4   cycle within one member: a symporter and a uniport for the same metabolite plus ATP
       synthase, or a metabolic pair X <=> Y making ATP and Y <=> X
  N1   negative control: legitimate cross-feeding (A secretes acetate, B grows on it)
  N2   negative control: no cycle, many alternative optima (parallel routes of equal yield,
       an optional cross-fed intermediate)

Perturbations: member count and abundances, glucose supply, ATP/proton/biomass stoichiometry,
cycle flux bounds, an optional ATP-driven CO2 -> formate module (so the electron test has
something to see), optional ATP synthases in the controls, and 0-4 energy-neutral decoys (dead
ends, a non-energy reaction loop, identically proton-coupled transporters in several members, an
ATP-wasting side reaction, a 1:1 ATP shuttle between members). Every instance's ground truth is
checked before it is scored (verify_truth); a violation is a generator bug and raises.

Properties (evaluate):
  P1  detection: energy_from_nothing > 0 iff a cycle is planted; equal to an independent LP;
      lax and strict closures agree
  P2  guarantee: after orient_community (members, pfba, both), ATP, electrons and growth from
      nothing are 0
  P3  retention: oriented growth / exact loop-law optimum (loop_law_optimum, cross-checked by
      LocalLoopLaw over every internal reaction and by the LP with the planted step cut); never
      above 1
  P4  no harm (N1, N2): oriented growth equals the LP optimum; bounds changed anyway
  P5  idempotence: orienting again keeps the optimum (P5a) and changes no bound (P5b)
  P6  find_energy_cycle returns a flux that holds a cycle (conformal_cycle), containing the
      planted steps and naming the planted cross-fed metabolites
  P7  the ATP pin is necessary: the loop law without it lets the cycle feed biomass
The full randomized sweep is Metabolic_modeling/multisystem3_cyclevalid_props.py; this module
runs one fixed seed per type.
"""
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field

import pytest
from cobra import Metabolite, Model, Reaction

TOL = 1e-6
KINDS = ("T1", "T2", "T3", "T3L", "T4", "N1", "N2")
PLANTED = {"T1", "T2", "T3", "T3L", "T4"}
ATP_VISIBLE = {"T1", "T2", "T3L", "T4"}  # T3 charges GTP only, which no ATP test sees
REFERENCES = ("members", "pfba", "both")

H, H2O, ATP, ADP, PI = "cpd00067", "cpd00001", "cpd00002", "cpd00008", "cpd00009"
GTP, GDP = "cpd00038", "cpd00031"
GLC, BIO, AC = "cpd00027", "cpd11416", "cpd00029"
CO2, FORMATE, O2 = "cpd00011", "cpd00047", "cpd00007"
# cycle metabolites (none of them a REDUCED_PRODUCT, so the electron test sees only the module)
PAIRS = [("cpd00209", "cpd00075"), ("cpd00023", "cpd00024"), ("cpd00130", "cpd00032"),
         ("cpd00106", "cpd00041"), ("cpd00020", "cpd00035"), ("cpd00033", "cpd00054"),
         ("cpd00268", "cpd00081")]
THIRD = ["cpd00137", "cpd00072", "cpd00080"]
DECOY_CPDS = ["cpd00066", "cpd00069", "cpd00060", "cpd00065", "cpd00119", "cpd00107", "cpd00156",
              "cpd00322", "cpd00039", "cpd00051", "cpd00084", "cpd00129", "cpd00161", "cpd00132",
              "cpd00053", "cpd00064", "cpd00117", "cpd00186", "cpd00100", "cpd00079"]
Q_CPD = "cpd00095"  # N2's cross-feedable intermediate


class GeneratorError(AssertionError):
    """An instance whose ground truth does not hold: a bug in the generator, not in the code."""


def _atp(n, comp, tri=ATP, di=ADP):
    """n > 0: n (ADP + Pi + H+ -> ATP + H2O); n < 0: hydrolysis."""
    return {(di, comp): -n, (PI, comp): -n, (H, comp): -n, (tri, comp): n, (H2O, comp): n}


def _add(*parts):
    out = {}
    for p in parts:
        for k, v in p.items():
            out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v != 0}


@dataclass
class Instance:
    kind: str
    seed: int
    model: Model
    n: int
    roles: dict
    params: dict
    cut: list = field(default_factory=list)       # reactions whose closure removes the cycle
    core: set = field(default_factory=set)        # reactions every found cycle must use
    shuttle: set = field(default_factory=set)     # e0 metabolites the cycle passes between members
    decoys: list = field(default_factory=list)
    gtp_comps: list = field(default_factory=list)

    @property
    def planted(self):
        return self.kind in PLANTED

    @property
    def visible(self):
        return self.kind in ATP_VISIBLE


class _Builder:
    def __init__(self, name):
        self.model = Model(name)
        self.mets, self.rxns, self.medium = {}, [], {}

    def met(self, cpd, comp):
        mid = f"{cpd}_{comp}"
        if mid not in self.mets:
            self.mets[mid] = Metabolite(mid, name=cpd, compartment=comp)
        return self.mets[mid]

    def rxn(self, rid, stoich, lb=-1000.0, ub=1000.0):
        r = Reaction(rid, lower_bound=lb, upper_bound=ub)
        r.add_metabolites({self.met(c, comp): v for (c, comp), v in stoich.items()})
        self.rxns.append(r)
        return rid

    def finish(self, abundances):
        n = len(abundances)
        self.rxn("bio1", _add({(BIO, f"c{i}"): -a for i, a in abundances.items()}, {(BIO, "c0"): 1}), 0, 1000)
        self.rxn(f"SK_{BIO}_c0", {(BIO, "c0"): -1}, -1000, 1000)
        for mid, met in sorted(self.mets.items()):
            if met.compartment == "e0":
                cpd = mid.split("_")[0]
                self.rxn(f"EX_{cpd}_e0", {(cpd, "e0"): -1}, -self.medium.get(cpd, 0.0), 1000)
        self.model.add_reactions(sorted(self.rxns, key=lambda r: r.id))
        self.model.objective = "bio1"
        threads1(self.model)
        return self.model


def threads1(model):
    """One solver thread (the shared host); a no-op off Gurobi."""
    prob = model.solver.problem
    if hasattr(prob, "Params"):
        prob.Params.Threads = 1
        prob.Params.OutputFlag = 0
    return model


def copy1(model):
    return threads1(model.copy())


# --------------------------------------------------------------------------- generator
def _base_member(b, i, y, k, carbon="glc", acetate_out=False, gtp_need=0, gtp_route=None):
    """A glucose (or acetate) user: carbon uptake, a lumped catabolism making y ATP, biomass
    needing 1 carbon + k ATP (+ gtp_need GTP), and its drain."""
    c = f"c{i}"
    if carbon == "glc":
        b.rxn(f"rxn10000_{c}", {(GLC, "e0"): -1, (GLC, c): 1})
        b.rxn(f"rxn20000_{c}", _add({(GLC, c): -1}, _atp(y, c), {(AC, c): 1} if acetate_out else {}), 0, 1000)
        if acetate_out:
            b.rxn(f"rxn10001_{c}", {(AC, "e0"): -1, (AC, c): 1})
        cpd = GLC
    else:  # acetate user
        b.rxn(f"rxn10001_{c}", {(AC, "e0"): -1, (AC, c): 1})
        b.rxn(f"rxn20002_{c}", _add({(AC, c): -1}, _atp(y, c)), 0, 1000)
        cpd = AC
    bio = _add({(cpd, c): -1}, _atp(-k, c), {(BIO, c): 1})
    if gtp_need:
        bio = _add(bio, _atp(-gtp_need, c, GTP, GDP))
    b.rxn(f"bio{i + 1}", bio, 0, 1000)
    b.rxn(f"SK_{BIO}_{c}", {(BIO, c): -1}, 0, 1000)
    if gtp_route == "glucose":   # substrate-level GTP from glucose, unlinked to ATP
        b.rxn(f"rxn20003_{c}", _add({(GLC, c): -1}, _atp(1, c, GTP, GDP)), 0, 1000)
    elif gtp_route == "ndk":     # nucleoside-diphosphate kinase: ATP + GDP <=> ADP + GTP
        b.rxn(f"rxn20004_{c}", {(ATP, c): -1, (GDP, c): -1, (ADP, c): 1, (GTP, c): 1})


def _synthase(b, i, s):
    c = f"c{i}"
    b.rxn(f"rxn91004_{c}", _add({(ADP, c): -1, (PI, c): -1, (H, "e0"): -s},
                                {(ATP, c): 1, (H2O, c): 1, (H, c): s - 1}))


def _e_module(b, i):
    """ATP-driven CO2 -> formate (+ O2): electrons out of CO2 and water need energy, so a clean
    community makes no formate from nothing, and one with free ATP does."""
    c = f"c{i}"
    b.rxn(f"rxn95000_{c}", {(CO2, "e0"): -1, (CO2, c): 1})
    b.rxn(f"rxn95001_{c}", _add({(CO2, c): -1}, _atp(-1, c), {(FORMATE, c): 1, (O2, c): 0.5}), 0, 1000)
    b.rxn(f"rxn95002_{c}", {(FORMATE, "e0"): -1, (FORMATE, c): 1})
    b.rxn(f"rxn95003_{c}", {(O2, "e0"): -1, (O2, c): 1})


def _decoy(b, rng, kind, members, pool, j):
    """Energy-neutral decoys; returns a description."""
    i = rng.choice(members)
    c = f"c{i}"
    num = 80000 + 10 * j
    if kind == "dead_end":
        x = pool.pop()
        b.rxn(f"rxn{num:05d}_{c}", {(GLC, c): -1, (x, c): 1}, 0, 1000)
        return {"decoy": kind, "member": i}
    if kind == "reaction_loop":   # three reversible steps P -> R -> S -> P, no cofactor
        p, r, s = pool.pop(), pool.pop(), pool.pop()
        b.rxn(f"rxn{num:05d}_{c}", {(p, c): -1, (r, c): 1})
        b.rxn(f"rxn{num + 1:05d}_{c}", {(r, c): -1, (s, c): 1})
        b.rxn(f"rxn{num + 2:05d}_{c}", {(s, c): -1, (p, c): 1})
        return {"decoy": kind, "member": i}
    if kind == "same_coupling":   # Z/H+ and W/H+ symporters (1:1) and Z <=> W in several members
        z, w = pool.pop(), pool.pop()
        who = sorted(rng.sample(members, k=min(len(members), rng.choice([2, 3]))))
        for m in who:
            cm = f"c{m}"
            b.rxn(f"rxn{num:05d}_{cm}", {(z, "e0"): -1, (H, "e0"): -1, (z, cm): 1, (H, cm): 1})
            b.rxn(f"rxn{num + 1:05d}_{cm}", {(w, "e0"): -1, (H, "e0"): -1, (w, cm): 1, (H, cm): 1})
            b.rxn(f"rxn{num + 2:05d}_{cm}", {(z, cm): -1, (w, cm): 1})
        return {"decoy": kind, "members": who}
    if kind == "atp_waste":       # glucose + ATP -> a secreted product
        x = pool.pop()
        b.rxn(f"rxn{num:05d}_{c}", _add({(GLC, c): -1, (x, c): 1}, _atp(-1, c)), 0, 1000)
        b.rxn(f"rxn{num + 1:05d}_{c}", {(x, "e0"): -1, (x, c): 1})
        return {"decoy": kind, "member": i}
    if kind == "atp_shuttle":     # R + ATP <=> S + ADP + Pi in two members, R and S cross-fed: 1:1
        r, s = pool.pop(), pool.pop()
        who = sorted(rng.sample(members, k=2)) if len(members) > 1 else members
        for m in who:
            cm = f"c{m}"
            b.rxn(f"rxn{num:05d}_{cm}", _add({(r, cm): -1, (s, cm): 1}, _atp(-1, cm)))
            b.rxn(f"rxn{num + 1:05d}_{cm}", {(r, "e0"): -1, (r, cm): 1})
            b.rxn(f"rxn{num + 2:05d}_{cm}", {(s, "e0"): -1, (s, cm): 1})
        return {"decoy": kind, "members": who}
    raise ValueError(kind)


DECOYS = ("dead_end", "reaction_loop", "same_coupling", "atp_waste", "atp_shuttle")


def make_instance(kind, seed, verify=True):
    """One synthetic community of `kind` (KINDS) from `seed`; deterministic."""
    if kind not in KINDS:
        raise ValueError(kind)
    rng = random.Random(1_000_003 * KINDS.index(kind) + seed)
    chain = kind == "T2" and rng.random() < 0.4
    n = rng.choice([2, 3, 4])
    if chain:
        n = max(n, 3)
    members = list(range(1, n + 1))
    raw = [rng.uniform(0.2, 1.0) for _ in members]
    abund = {i: r / sum(raw) for i, r in zip(members, raw)}
    y = {i: rng.choice([1, 2, 3]) for i in members}
    k = {i: rng.choice([3, 4, 5, 6, 8]) for i in members}
    U = rng.choice([200.0, 500.0, 1000.0])
    glc = round(rng.uniform(5, 20), 3)
    pool = list(DECOY_CPDS)
    rng.shuffle(pool)
    m_cpd, n_cpd = rng.choice(PAIRS)
    if rng.random() < 0.5:
        m_cpd, n_cpd = n_cpd, m_cpd
    b = _Builder(f"{kind}_{seed}")
    b.medium[GLC] = glc
    inst = Instance(kind, seed, None, n, {}, {"n": n, "abundance": abund, "y": y, "k": k, "U": U,
                                              "glucose": glc, "pair": (m_cpd, n_cpd)})
    roles = rng.sample(members, k=3 if chain else 2)
    a_, b_ = roles[0], roles[1]
    inst.roles = {"A": a_, "B": b_} if kind != "T4" else {"A": a_}
    ca, cb = f"c{a_}", f"c{b_}"
    special = set()

    if kind == "T1":
        h, g, s = rng.choice([1, 2]), rng.choice([1, 2]), rng.choice([3, 4])
        inst.params.update(h=h, g=g, s=s)
        b.rxn(f"rxn91000_{ca}", {(m_cpd, "e0"): -1, (H, ca): -h, (m_cpd, ca): 1, (H, "e0"): h}, -U, U)
        b.rxn(f"rxn91003_{cb}", {(m_cpd, "e0"): -1, (H, "e0"): -g, (m_cpd, cb): 1, (H, cb): g}, -U, U)
        for c in (ca, cb):
            b.rxn(f"rxn91001_{c}", {(m_cpd, c): -1, (n_cpd, c): 1}, -U, U)
            b.rxn(f"rxn91002_{c}", {(n_cpd, "e0"): -1, (n_cpd, c): 1}, -U, U)
        _synthase(b, a_, s)
        _synthase(b, b_, s)
        inst.cut = [f"rxn91000_{ca}"]
        inst.core = {f"rxn91000_{ca}", f"rxn91003_{cb}", f"rxn91001_{ca}", f"rxn91001_{cb}",
                     f"rxn91002_{ca}", f"rxn91002_{cb}", f"rxn91004_{ca}", f"rxn91004_{cb}"}
        inst.shuttle = {f"{m_cpd}_e0", f"{n_cpd}_e0"}
    elif kind == "T2":
        p = rng.choice([0.5, 1, 1.5, 2, 3])
        q = rng.choice([0, 0.5, 1, 2])
        inst.params.update(p=p, q=q, chain=chain)
        if not chain:
            b.rxn(f"rxn92000_{ca}", _add({(m_cpd, ca): -1, (n_cpd, ca): 1}, _atp(p, ca)), -U, U)
            b.rxn(f"rxn92001_{cb}", _add({(n_cpd, cb): -1, (m_cpd, cb): 1}, _atp(q, cb)), -U, U)
            cpds = [m_cpd, n_cpd]
            uni = {ca: (m_cpd, n_cpd), cb: (n_cpd, m_cpd)}
        else:
            z = rng.choice(THIRD)
            cc = f"c{roles[2]}"
            r = rng.choice([0, 1])
            inst.params.update(r=r, z=z)
            inst.roles["C"] = roles[2]
            b.rxn(f"rxn92000_{ca}", _add({(m_cpd, ca): -1, (n_cpd, ca): 1}, _atp(p, ca)), -U, U)
            b.rxn(f"rxn92001_{cb}", _add({(n_cpd, cb): -1, (z, cb): 1}, _atp(q, cb)), -U, U)
            b.rxn(f"rxn92002_{cc}", _add({(z, cc): -1, (m_cpd, cc): 1}, _atp(r, cc)), -U, U)
            inst.core.add(f"rxn92002_{cc}")
            cpds = [m_cpd, n_cpd, z]
            uni = {ca: (m_cpd, n_cpd), cb: (n_cpd, z), cc: (z, m_cpd)}
        for c, pair in uni.items():
            for cpd in pair:
                rid = b.rxn(f"rxn9201{cpds.index(cpd)}_{c}", {(cpd, "e0"): -1, (cpd, c): 1}, -U, U)
                inst.core.add(rid)
        inst.shuttle = {f"{cpd}_e0" for cpd in cpds}
        inst.core |= {f"rxn92000_{ca}", f"rxn92001_{cb}"}
        inst.cut = [f"rxn92000_{ca}"]
    elif kind in ("T3", "T3L"):
        p = rng.choice([1, 2])
        gneed = rng.choice([1, 2, 3])
        inst.params.update(p=p, gtp_need=gneed)
        b.rxn(f"rxn93000_{ca}", _add({(m_cpd, ca): -1, (n_cpd, ca): 1}, _atp(p, ca, GTP, GDP)), -U, U)
        b.rxn(f"rxn93001_{cb}", {(n_cpd, cb): -1, (m_cpd, cb): 1}, -U, U)
        for c in (ca, cb):
            b.rxn(f"rxn93010_{c}", {(m_cpd, "e0"): -1, (m_cpd, c): 1}, -U, U)
            b.rxn(f"rxn93011_{c}", {(n_cpd, "e0"): -1, (n_cpd, c): 1}, -U, U)
        _base_member(b, a_, y[a_], k[a_], gtp_need=gneed, gtp_route="ndk" if kind == "T3L" else "glucose")
        special.add(a_)
        inst.cut = [f"rxn93000_{ca}"]
        inst.core = {f"rxn93000_{ca}", f"rxn93001_{cb}", f"rxn93010_{ca}", f"rxn93010_{cb}",
                     f"rxn93011_{ca}", f"rxn93011_{cb}"} | ({f"rxn20004_{ca}"} if kind == "T3L" else set())
        inst.shuttle = {f"{m_cpd}_e0", f"{n_cpd}_e0"}
        inst.gtp_comps = [ca]
    elif kind == "T4":
        variant = rng.choice(["transport", "metabolic"])
        inst.params["variant"] = variant
        if variant == "transport":
            h, s = rng.choice([1, 2]), rng.choice([3, 4])
            inst.params.update(h=h, s=s)
            b.rxn(f"rxn94000_{ca}", {(m_cpd, "e0"): -1, (H, "e0"): -h, (m_cpd, ca): 1, (H, ca): h}, -U, U)
            b.rxn(f"rxn94001_{ca}", {(m_cpd, "e0"): -1, (m_cpd, ca): 1}, -U, U)
            _synthase(b, a_, s)
            inst.cut = [f"rxn94000_{ca}"]
            inst.core = {f"rxn94000_{ca}", f"rxn94001_{ca}", f"rxn91004_{ca}"}
        else:
            p = rng.choice([0.5, 1, 2])
            inst.params["p"] = p
            b.rxn(f"rxn94002_{ca}", _add({(m_cpd, ca): -1, (n_cpd, ca): 1}, _atp(p, ca)), -U, U)
            b.rxn(f"rxn94003_{ca}", {(n_cpd, ca): -1, (m_cpd, ca): 1}, -U, U)
            inst.cut = [f"rxn94002_{ca}"]
            inst.core = {f"rxn94002_{ca}", f"rxn94003_{ca}"}
    elif kind == "N1":
        _base_member(b, a_, y[a_], k[a_], acetate_out=True)
        _base_member(b, b_, y[b_], k[b_], carbon="ac")
        special |= {a_, b_}
    elif kind == "N2":
        cross = rng.random() < 0.6
        inst.params["q_cross_fed"] = cross
        for i in members:
            c = f"c{i}"
            b.rxn(f"rxn20005_{c}", {(GLC, c): -1, (Q_CPD, c): 1})
            b.rxn(f"rxn20006_{c}", _add({(Q_CPD, c): -1}, _atp(y[i], c)), 0, 1000)
            if cross:
                b.rxn(f"rxn10002_{c}", {(Q_CPD, "e0"): -1, (Q_CPD, c): 1})

    for i in members:
        if i not in special:
            _base_member(b, i, y[i], k[i])
    if kind in ("N1", "N2"):
        for i in members:
            if rng.random() < 0.5:
                _synthase(b, i, rng.choice([3, 4]))
    if rng.random() < 0.3:
        e_member = a_ if kind in ("T1", "T2", "T4", "T3L") else rng.choice(members)
        _e_module(b, e_member)
        inst.params["e_module"] = e_member
    for j in range(rng.randint(0, 4)):
        inst.decoys.append(_decoy(b, rng, rng.choice(DECOYS), members, pool, j))
    inst.model = b.finish(abund)
    if verify:
        verify_truth(inst)
    return inst


# --------------------------------------------------------------------------- independent checks
def _comp_members(rxn):
    return {int(m.compartment[1:]) for m in rxn.metabolites
            if m.compartment.startswith("c") and m.compartment[1:].isdigit() and m.compartment != "c0"}


def independent_atp(model, only=None):
    """Max total ATP hydrolysis with every uptake closed, written from scratch (scipy HiGHS on the
    stoichiometric matrix): forced fluxes released, boundary reactions may only remove matter,
    one hydrolysis column per member cytosol. only: a member index; every other member's
    reactions are shut."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.optimize import linprog
    mets = {m.id: j for j, m in enumerate(model.metabolites)}
    rows, cols, vals, bounds = [], [], [], []
    for j, r in enumerate(model.reactions):
        for m, c in r.metabolites.items():
            rows.append(mets[m.id]); cols.append(j); vals.append(c)
        lb, ub = r.bounds
        if len(r.metabolites) == 1:
            lb, ub = 0.0, max(ub, 0.0)
        else:
            lb, ub = min(lb, 0.0), max(ub, 0.0)
            mem = _comp_members(r)
            if only is not None and (r.id == "bio1" or (mem and mem != {only})):
                lb, ub = 0.0, 0.0
        bounds.append((lb, ub))
    nr = len(bounds)
    tests = []
    comps = sorted({m.compartment for m in model.metabolites if m.compartment.startswith("c")
                    and m.compartment != "c0"})
    for comp in comps:
        if only is not None and comp != f"c{only}":
            continue
        need = {ATP: -1, H2O: -1, ADP: 1, PI: 1, H: 1}
        if all(f"{c}_{comp}" in mets for c in need):
            j = nr + len(tests)
            for c, v in need.items():
                rows.append(mets[f"{c}_{comp}"]); cols.append(j); vals.append(v)
            tests.append(j)
            bounds.append((0.0, 1000.0))
    S = sp.csr_matrix((vals, (rows, cols)), shape=(len(mets), len(bounds)))
    c = np.zeros(len(bounds))
    c[tests] = -1.0
    res = linprog(c, A_eq=S, b_eq=np.zeros(len(mets)), bounds=bounds, method="highs")
    return -res.fun if res.status == 0 else None


def lp_growth(model, closed=()):
    with model:
        for rid in closed:
            model.reactions.get_by_id(rid).bounds = (0, 0)
        return model.slim_optimize(error_value=float("nan"))


def verify_truth(inst):
    """The instance's ground truth, checked before any property is scored."""
    m = inst.model
    tag = f"{inst.kind} seed {inst.seed}"
    g = lp_growth(m)
    if not g > TOL:
        raise GeneratorError(f"{tag}: the community does not grow on its medium ({g})")
    atp = independent_atp(m)
    if inst.visible and not atp > TOL:
        raise GeneratorError(f"{tag}: planted cycle makes no ATP from nothing ({atp})")
    if not inst.visible and not abs(atp) <= TOL:
        raise GeneratorError(f"{tag}: ATP from nothing {atp} where none is planted")
    for i in range(1, inst.n + 1):
        alone = independent_atp(m, only=i)
        planted_alone = inst.kind == "T4" and i == inst.roles["A"]
        if planted_alone and not alone > TOL:
            raise GeneratorError(f"{tag}: the within-member cycle of member {i} makes nothing alone")
        if not planted_alone and not abs(alone) <= TOL:
            raise GeneratorError(f"{tag}: member {i} alone makes ATP from nothing ({alone})")
    if inst.planted:
        with m:
            for rid in inst.cut:
                m.reactions.get_by_id(rid).bounds = (0, 0)
            left = independent_atp(m)
            gcut = m.slim_optimize(error_value=float("nan"))
        if not abs(left) <= TOL:
            raise GeneratorError(f"{tag}: a cycle is left after the planted cut ({left})")
        if inst.kind == "T3" and not g > gcut + TOL:
            raise GeneratorError(f"{tag}: the GTP loop does not feed biomass ({g} vs {gcut})")
    return True


# --------------------------------------------------------------------------- loop-law oracles
def _pin(law, model, tri, di, comps):
    """Add `tri` hydrolysis pins (dG <= -eps) to a LocalLoopLaw, in the given compartments."""
    mets = {m.id: m for m in model.metabolites}
    new, rows = [], []
    for comp in comps:
        st = {tri: -1, H2O: -1, di: 1, PI: 1, H: 1}
        if any(f"{c}_{comp}" not in mets for c in st):
            continue
        expr = sum(v * law._mu(mets[f"{c}_{comp}"], new) for c, v in st.items())
        rows.append(model.problem.Constraint(expr, ub=-law.eps, name=f"llpin_{tri}_{comp}"))
    model.add_cons_vars(new + rows)
    law._objs += new + rows


def full_loop_law(model, pin_atp=True, extra_pins=(), pins=None):
    """Growth under LocalLoopLaw on every internal reaction (the MILP, no cutting planes).
    pins goes to LocalLoopLaw (None: its default, ATP only)."""
    from mscommunity.commcycles import LocalLoopLaw
    m = copy1(model)
    law = LocalLoopLaw(m, pin_atp=pin_atp, **({"pins": pins} if pins is not None else {}))
    for tri, di, comps in extra_pins:
        _pin(law, m, tri, di, comps)
    law.add([r for r in m.reactions if not r.boundary])
    return m.slim_optimize(error_value=float("nan"))


# --------------------------------------------------------------------------- properties
def _close(a, b, tol=TOL):
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _zero(x):
    return x is not None and abs(x) <= TOL


def evaluate(inst, references=REFERENCES, exact=True, pins=None, orient_args=None):
    """Every property on one instance; returns a JSON-friendly record. pins goes to every
    commcycles call that takes it (None: the default, ATP only); orient_args to orient_community."""
    from mscommunity.commcycles import (conformal_cycle, energy_from_nothing, find_energy_cycle,
                                        loop_law_optimum, orient_community, robust_pfba)
    t0 = time.time()
    m = inst.model
    rec = {"kind": inst.kind, "seed": inst.seed, "n": inst.n, "params": _jsonable(inst.params),
           "decoys": inst.decoys, "n_reactions": len(m.reactions)}
    g_lp = lp_growth(m)
    rec["lp_growth"] = g_lp
    # P1 detection
    pk = {} if pins is None else {"pins": pins}
    ok = dict(orient_args or {}, **pk)
    efn = energy_from_nothing(m, **pk)
    efs = energy_from_nothing(m, closure="strict", **pk)
    ind = independent_atp(m)
    detected = any((efn.get(k) or 0) > TOL for k in ("atp_from_nothing", "electrons_from_nothing",
                                                     "growth_from_nothing"))
    rec["P1"] = {"atp": efn["atp_from_nothing"], "electrons": efn["electrons_from_nothing"],
                 "growth": efn.get("growth_from_nothing"), "independent_atp": ind,
                 "strict_atp": efs["atp_from_nothing"], "detected": detected,
                 "planted": inst.planted,
                 "pass": detected == inst.planted,
                 "matches_independent": _close(efn["atp_from_nothing"], ind),
                 "lax_equals_strict": _close(efn["atp_from_nothing"], efs["atp_from_nothing"])}
    # exact loop-law optimum and its cross-checks
    rec["oracle"] = {}
    both = copy1(m)
    orient_community(both, reference="both", threads=1, **pk)
    start = robust_pfba(both).fluxes.to_dict()
    ll = copy1(m)
    _, rep = loop_law_optimum(ll, start=start, **pk)
    g_ll = rep["best_objective"] if rep["status"] == "no cycle" else None
    rec["oracle"]["loop_law_optimum"] = g_ll
    rec["oracle"]["status"] = rep["status"]
    rec["oracle"]["rounds"] = len(rep["rounds"])
    rec["oracle"]["mip_gap"] = rep.get("mip_gap")
    rec["oracle"]["clean_objective"] = rep.get("clean_objective")
    rec["oracle"]["clean_cycle"] = (len(rep["clean_cycle"]) if rep.get("clean_cycle") is not None else None)
    rec["oracle"]["clean_status"] = rep.get("clean_status")
    if exact:
        g_full = full_loop_law(m, pins=pins)
        rec["oracle"]["full_milp"] = g_full
        rec["oracle"]["agree_full"] = _close(g_ll, g_full)
    if inst.planted:
        g_cut = lp_growth(m, inst.cut)
        rec["oracle"]["cut_lp"] = g_cut
        if inst.kind != "T3" and inst.kind != "T3L":
            rec["oracle"]["agree_cut"] = _close(g_ll, g_cut)
    if inst.gtp_comps:   # the thermodynamic truth with GTP hydrolysis pinned as well
        g_ntp = full_loop_law(m, extra_pins=[(GTP, GDP, inst.gtp_comps)])
        rec["oracle"]["ntp_pinned"] = g_ntp
        rec["oracle"]["agree_cut_ntp"] = _close(g_ntp, rec["oracle"]["cut_lp"])
        rec["oracle"]["atp_pin_leak"] = (g_ll - g_ntp) if g_ll is not None else None
    truth = g_ll if not inst.gtp_comps else rec["oracle"]["ntp_pinned"]
    # P2-P5 per reference
    rec["refs"] = {}
    for ref in references:
        o = copy1(m)
        r1 = orient_community(o, reference=ref, threads=1, **ok)
        g1 = o.slim_optimize(error_value=float("nan"))
        e1 = energy_from_nothing(o, **pk)
        row = {"growth": g1, "bounds_changed": len(r1["changed"]),
               "atp": e1["atp_from_nothing"], "electrons": e1["electrons_from_nothing"],
               "growth_from_nothing": e1.get("growth_from_nothing")}
        row["P2"] = all(_zero(e1.get(k)) for k in ("atp_from_nothing", "electrons_from_nothing",
                                                    "growth_from_nothing"))
        row["P3_ratio"] = (g1 / g_ll) if g_ll and g_ll > TOL else None
        row["P3_sound"] = g_ll is not None and g1 <= g_ll + TOL * max(1.0, g_ll)
        if inst.gtp_comps:
            row["ratio_vs_ntp_truth"] = g1 / truth if truth and truth > TOL else None
            row["leak_over_truth"] = g1 - truth
        if not inst.planted:
            row["P4"] = _close(g1, g_lp)
        r2 = orient_community(o, reference=ref, threads=1, **ok)
        g2 = o.slim_optimize(error_value=float("nan"))
        row["P5a"] = _close(g1, g2)
        row["P5b"] = len(r2["changed"]) == 0
        row["P5_bounds_changed_again"] = len(r2["changed"])
        rec["refs"][ref] = row
    # P6 the cycle found
    if inst.visible:
        cyc = find_energy_cycle(m)
        rx = {r["reaction"] for r in cyc["reactions"]}
        flux = {r["reaction"]: r["flux"] for r in cyc["reactions"]}
        conf = set(conformal_cycle(m, flux)) if cyc["status"] == "cycle" else set()
        named = set(cyc["cross_fed"])
        rec["P6"] = {"status": cyc["status"], "n_reactions": len(rx), "conformal": len(conf),
                     "conformal_in_found": conf <= rx and bool(conf),
                     "core_found": inst.core <= rx, "missing_core": sorted(inst.core - rx),
                     "shuttle_named": inst.shuttle <= named,
                     "named": sorted(named),
                     "no_extra_named": named <= inst.shuttle | {f"{H}_e0"}}
        rec["P6"]["pass"] = (cyc["status"] == "cycle" and rec["P6"]["conformal_in_found"]
                             and rec["P6"]["core_found"] and rec["P6"]["shuttle_named"])
    # P7 the pin is necessary
    if inst.visible and exact and g_ll is not None:
        feeds = g_lp > g_ll + TOL * max(1.0, g_ll)
        g_unpinned = full_loop_law(m, pin_atp=False)
        rec["P7"] = {"cycle_feeds_biomass": feeds, "unpinned": g_unpinned, "pinned": g_ll,
                     "pin_necessary": feeds and g_unpinned > g_ll + TOL * max(1.0, g_ll),
                     "unpinned_equals_lp": _close(g_unpinned, g_lp)}
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


# --------------------------------------------------------------------------- the suite subset
@pytest.fixture(scope="module", autouse=True)
def _one_thread(tmp_path_factory):
    """Gurobi environments made by commcycles (potential profiles, conformal_cycle) read their
    defaults from gurobi.env in the working directory: one thread on the shared host."""
    d = tmp_path_factory.mktemp("grb")
    (d / "gurobi.env").write_text("Threads 1\n")
    old = os.getcwd()
    os.chdir(d)
    yield
    os.chdir(old)


SEEDS = {"T1": 0, "T2": 0, "T3": 0, "T3L": 0, "T4": 0, "N1": 0, "N2": 0}


@pytest.mark.parametrize("kind", KINDS)
def test_generated_ground_truth_holds(kind):
    inst = make_instance(kind, SEEDS[kind])
    assert verify_truth(inst)


@pytest.mark.parametrize("kind", KINDS)
def test_properties_on_one_instance_per_type(kind):
    rec = evaluate(make_instance(kind, SEEDS[kind]))
    p1 = rec["P1"]
    assert p1["matches_independent"] and p1["lax_equals_strict"]
    if kind == "T3":
        # the documented blind spot: a GTP loop is invisible to an ATP test
        assert not p1["detected"]
    else:
        assert p1["pass"]
    assert rec["oracle"]["status"] == "no cycle" and rec["oracle"]["agree_full"]
    if "agree_cut" in rec["oracle"]:
        assert rec["oracle"]["agree_cut"]
    for ref, row in rec["refs"].items():
        assert row["P2"], (ref, row)
        assert row["P3_sound"], (ref, row)
        assert row["P5a"], (ref, row)
        if kind in ("N1", "N2") and ref != "members":
            assert row["P4"], (ref, row)
    if kind in ATP_VISIBLE:
        assert rec["P6"]["pass"], rec["P6"]
    if "P7" in rec and rec["P7"]["cycle_feeds_biomass"]:
        assert rec["P7"]["pin_necessary"]


def test_t1_makes_atp_at_the_flux_bound_through_the_protons():
    """The CommPhitting arithmetic: per turn, h + g protons leave the cytosols and s re-enter
    per ATP, so ATP from nothing is (h + g) U / s at the turn bound U."""
    inst = make_instance("T1", SEEDS["T1"])
    from mscommunity.commcycles import energy_from_nothing
    p = inst.params
    atp = energy_from_nothing(inst.model, electrons=False)["atp_from_nothing"]
    assert atp == pytest.approx((p["h"] + p["g"]) * p["U"] / p["s"], rel=1e-6)


def test_the_direct_build_matches_build_from_species_models():
    """The generator writes the merged layout directly; assembling the same members through
    build_from_species_models gives the same tests."""
    from mscommunity.commhelper import build_from_species_models
    from mscommunity.commcycles import energy_from_nothing
    inst = make_instance("T2", SEEDS["T2"])
    members = split_members(inst)
    merged = build_from_species_models(members, abundances={mm.id: a for mm, a in
                                                            zip(members, inst.params["abundance"].values())})
    threads1(merged)
    a = energy_from_nothing(inst.model)
    b = energy_from_nothing(merged)
    for key in ("atp_from_nothing", "growth_from_nothing", "electrons_from_nothing"):
        assert a[key] == pytest.approx(b[key], abs=1e-6)
    assert inst.model.slim_optimize() == pytest.approx(merged.slim_optimize(), rel=1e-6)


def split_members(inst):
    """Each member of a generated community as its own model, in c0/e0, for build_from_species_models."""
    out = []
    comm = inst.model
    for i in range(1, inst.n + 1):
        mm = Model(f"m{i}")
        mets = {}

        def met(old):
            comp = "e0" if old.compartment == "e0" else "c0"
            mid = f"{old.id.split('_')[0]}_{comp}"
            if mid not in mets:
                mets[mid] = Metabolite(mid, name=old.name, compartment=comp)
            return mets[mid]
        rxns = []
        for r in comm.reactions:
            if r.id == "bio1" or r.id.startswith(("EX_", f"SK_{BIO}_c0")) or _comp_members(r) != {i}:
                continue
            rid = "bio1" if r.id == f"bio{i + 1}" else r.id.rsplit("_", 1)[0] + "_c0"
            nr = Reaction(rid, lower_bound=r.lower_bound, upper_bound=r.upper_bound)
            nr.add_metabolites({met(x): v for x, v in r.metabolites.items()})
            rxns.append(nr)
        for mid, x in sorted(mets.items()):
            if x.compartment == "e0":
                ex = comm.reactions.get_by_id(f"EX_{mid}")
                nr = Reaction(ex.id, lower_bound=ex.lower_bound, upper_bound=ex.upper_bound)
                nr.add_metabolites({x: -1})
                rxns.append(nr)
        mm.add_reactions(rxns)
        mm.objective = "bio1"
        out.append(mm)
    return out


# --------------------------------------------------------------------------- the gate recipe
def _floored(kind, seed, floor=0.5, members=None):
    """A generated community with a maintenance-ATP floor (rxn00062, ATP + H2O -> ADP + Pi + H+,
    lower bound `floor`) in every member (or in `members`), as constrain() installs ATPM floors."""
    inst = make_instance(kind, seed)
    rxns = []
    mets = {x.id: x for x in inst.model.metabolites}
    for i in (members or range(1, inst.n + 1)):
        c = f"c{i}"
        r = Reaction(f"rxn00062_{c}", lower_bound=floor, upper_bound=1000)
        r.add_metabolites({mets[f"{k}_{c}"]: v for k, v in
                           {ATP: -1, H2O: -1, ADP: 1, PI: 1, H: 1}.items()})
        rxns.append(r)
    inst.model.add_reactions(rxns)
    return inst


def test_floors_make_the_sealed_test_infeasible_and_releasing_them_gives_zero():
    from mscommunity.commcycles import energy_from_nothing, find_energy_cycle
    inst = _floored("N1", SEEDS["N1"])
    m = inst.model
    before = energy_from_nothing(m, release_floors=False)
    assert before["atp_from_nothing"] is None and before["released"] == {}
    for closure in ("lax", "strict"):
        after = energy_from_nothing(m, closure=closure)
        assert after["atp_from_nothing"] == pytest.approx(0.0, abs=1e-9)
        assert after["growth_from_nothing"] == pytest.approx(0.0, abs=1e-9)
        assert after["electrons_from_nothing"] == pytest.approx(0.0, abs=1e-9)
        assert set(after["released"]) == {f"rxn00062_c{i}" for i in range(1, inst.n + 1)}
        assert all(v == (0.5, 1000) for v in after["released"].values())
    # the bounds are back
    assert all(m.reactions.get_by_id(f"rxn00062_c{i}").lower_bound == 0.5 for i in range(1, inst.n + 1))
    assert find_energy_cycle(m, release_floors=False)["status"] == "infeasible"
    assert find_energy_cycle(m)["status"] == "no cycle"


def test_with_floors_in_place_none_says_nothing_about_cycles():
    """Floors left in: a clean community reads None, and so does one with a cycle as soon as one
    member is out of the cycle's reach and cannot pay its floor. Released: the cycle's own value."""
    from mscommunity.commcycles import energy_from_nothing
    plain = make_instance("T2", SEEDS["T2"])
    ref = energy_from_nothing(plain.model, electrons=False)
    makers = set(ref["atp_by_member"])
    outside = set(range(1, plain.n + 1)) - makers
    assert outside, "seed chosen so that one member is outside the cycle"
    inst = _floored("T2", SEEDS["T2"])
    released = energy_from_nothing(inst.model, electrons=False)
    assert released["atp_from_nothing"] == pytest.approx(ref["atp_from_nothing"], rel=1e-6)
    assert energy_from_nothing(inst.model, electrons=False, release_floors=False)["atp_from_nothing"] is None
    # floors only where the cycle makes ATP: the cycle pays for them, and the test reads a number
    only = _floored("T2", SEEDS["T2"], members=makers)
    kept = energy_from_nothing(only.model, electrons=False, release_floors=False)["atp_from_nothing"]
    assert kept is not None and kept > 1.0


def test_strict_without_the_biomass_outlet_is_vacuous_for_growth():
    from mscommunity.commcycles import energy_from_nothing
    inst = make_instance("N2", SEEDS["N2"])
    r = energy_from_nothing(inst.model, closure="strict", biomass_outlet=False)
    assert r["growth_from_nothing"] == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        energy_from_nothing(inst.model, closure="sealed")


def test_the_community_class_releases_floors_and_lifts_the_kinetic_rows():
    from mscommunity.commhelper import build_from_species_models
    from mscommunity.mscommsim import MSCommunity
    inst = make_instance("N1", SEEDS["N1"])
    members = split_members(inst)
    ab = {mm.id: a for mm, a in zip(members, inst.params["abundance"].values())}
    model = build_from_species_models(members, abundances=ab)
    threads1(model)
    for i in range(1, inst.n + 1):   # maintenance floors, as constrain() installs them
        mets = {x.id: x for x in model.metabolites}
        r = Reaction(f"rxn00062_c{i}", lower_bound=0.5, upper_bound=1000)
        r.add_metabolites({mets[f"{k}_c{i}"]: v for k, v in {ATP: -1, H2O: -1, ADP: 1, PI: 1, H: 1}.items()})
        model.add_reactions([r])
    comm = MSCommunity(model=model, ids=list(ab), kinetic_coeff=1e6)
    threads1(comm.util.model)
    old = comm.energy_cycles(release_floors=False)
    assert old["atp_from_nothing"] is None
    new = comm.energy_cycles()
    assert new["atp_from_nothing"] == pytest.approx(0.0, abs=1e-9)
    assert new["growth_from_nothing"] == pytest.approx(0.0, abs=1e-9)
    assert set(new["released"]) == {f"rxn00062_c{i}" for i in range(1, inst.n + 1)}
    assert new["kinetic_rows_lifted"]
    assert comm.energy_cycles(closure="strict")["atp_from_nothing"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- package fixes (Nov 2026)
def test_members_reference_keeps_a_member_that_cannot_grow_alone(caplog):
    """N1 seed 11: member 4 lives on member 3's acetate and grows 0 alone. The members reference
    used to honour nothing of it, and the profile shut its acetate uptake: community growth 0
    against an LP optimum of 1.768. Now the reference falls back to "both" (members, then the
    community optimum), with a warning, and the community keeps its optimum. The restricted
    fallback ("optimum": the optimum on member 4's reactions only) keeps member 4 growing but
    not an energy-neutral ATP shuttle between members 1 and 3 that no member uses alone
    (rxn80010-12): 1.383."""
    import logging
    from mscommunity.commcycles import energy_from_nothing, orient_community
    inst = make_instance("N1", 11)
    g_lp = lp_growth(inst.model)
    assert g_lp == pytest.approx(1.768043567, rel=1e-6)
    old = copy1(inst.model)
    rep = orient_community(old, reference="members", threads=1, no_growth_fallback=None)
    assert rep["fallback"] is None and old.slim_optimize(error_value=0.0) == pytest.approx(0.0, abs=1e-9)
    o = copy1(inst.model)
    with caplog.at_level(logging.WARNING, logger="mscommunity.commcycles"):
        rep = orient_community(o, reference="members", threads=1)
    assert any("cannot grow alone" in r.message for r in caplog.records)
    stuck = [i for i, g in rep["member_growth_alone"].items() if g <= TOL]
    assert stuck == [4] and rep["fallback"]["members"] == stuck and rep["fallback"]["mode"] == "both"
    assert o.slim_optimize() == pytest.approx(g_lp, rel=1e-6)
    e = energy_from_nothing(o)
    assert all(_zero(e[k]) for k in ("atp_from_nothing", "electrons_from_nothing", "growth_from_nothing"))
    part = copy1(inst.model)
    orient_community(part, reference="members", threads=1, no_growth_fallback="optimum")
    assert part.reactions.get_by_id("rxn10001_c4").upper_bound > 0   # acetate uptake still open
    with part:
        part.objective = "bio5"                                        # member 4's biomass
        assert part.slim_optimize() > TOL
    assert part.slim_optimize() == pytest.approx(1.383136683, rel=1e-6)


@pytest.mark.parametrize("kind", ["T3", "T3L"])
def test_ntp_pins_close_the_gtp_leak(kind):
    """With GTP hydrolysis pinned as well (pins="ntp"), the loop-law optimum equals the
    ATP+GTP-pinned truth, the GTP loop is visible to the sealed test, and no orientation keeps
    growth above the truth. With the default (ATP only) the leak is there, as before."""
    from mscommunity.commcycles import (NTP_PINS, energy_from_nothing, loop_law_optimum,
                                        orient_community, robust_pfba)
    inst = make_instance(kind, SEEDS[kind])
    m = inst.model
    truth = full_loop_law(m, extra_pins=[(GTP, GDP, inst.gtp_comps)])
    assert full_loop_law(m, pins=NTP_PINS) == pytest.approx(truth, rel=1e-6)
    # ATP only: the loop law leaks over the truth
    safe = copy1(m)
    orient_community(safe, reference="both", threads=1)
    _, rep = loop_law_optimum(copy1(m), start=robust_pfba(safe).fluxes.to_dict())
    assert rep["status"] == "no cycle" and rep["best_objective"] > truth * (1 + 1e-6)
    # NTP pins: no leak
    safe = copy1(m)
    orient_community(safe, reference="both", threads=1, pins="ntp")
    _, rep = loop_law_optimum(copy1(m), start=robust_pfba(safe).fluxes.to_dict(), pins="ntp")
    assert rep["status"] == "no cycle"
    assert rep["best_objective"] == pytest.approx(truth, rel=1e-6)
    for ref in REFERENCES:
        o = copy1(m)
        orient_community(o, reference=ref, threads=1, pins="ntp")
        assert o.slim_optimize() <= truth * (1 + 1e-6), ref
    ntp = energy_from_nothing(m, pins="ntp")
    assert ntp["atp_from_nothing"] > TOL and "ntp_by_test" in ntp
    if kind == "T3":   # unlinked: only the GTP test sees it (linked, NDK hands it to the ATP test)
        assert ntp["ntp_by_test"]
        assert energy_from_nothing(m)["atp_from_nothing"] == pytest.approx(0.0, abs=TOL)
