# -*- coding: utf-8 -*-
"""MSCommunity.micom (MICOM's cooperative tradeoff, mscommunity.commmicom), micom_legacy, and the
regularization / predict_abundances fixes that came with it.

The toy is tests/test_a_mscommsim.py's pair: memA (Species0) eats cpd00027 (community uptake cap
10), memB (Species1) eats cpd00082 (cap 4), each turns one carbon into one unit of its biomass,
and bio1 takes 0.5 of each member's biomass. With drains open, sum_m v_bio_m <= 10 + 4 = 14.

    stage 1   F = max sum_m v_bio_m
    stage 2   min sum_m (v_bio_m / a_m)^2   s.t.  f F <= sum_m v_bio_m <= F
    no member constraint binding  =>  mu_m = f F a_m / sum_k a_k^2
"""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import warnings

import pytest

from mscommunity import commmicom
from mscommunity.commmembers import reactions_by_member
from mscommunity.mscommsim import MSCommunity, _pick_qp_backend
from modelseedpy.core.exceptions import FeasibilityError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_a_mscommsim import _community, _competing, _members  # noqa: E402

needs_qp = pytest.mark.skipif(_pick_qp_backend() is None, reason="no QP-capable optlang backend installed")

A_64 = {"Species0": 0.6, "Species1": 0.4}
MEDIUM = {"EX_cpd00027_e0": 10, "EX_cpd00082_e0": 4}
HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.dirname(HERE)


def _state(comm):
    """Everything micom() promises to put back."""
    model = comm.util.model
    return {
        "bounds": {r.id: tuple(r.bounds) for r in model.reactions},
        "objective": (str(model.objective.expression), model.objective.direction),
        "rows": {c.name: (c.lb, c.ub) for c in model.constraints},
        "variables": sorted(v.name for v in model.variables),
        "abundances": {m.id: m.abundance for m in comm.members},
        "abundances_attr": dict(comm.abundances),
        "bio1": {met.id: c for met, c in comm.primary_biomass.metabolites.items()},
        "solver": model.solver.interface.__name__,
    }


# --------------------------------------------------------------------------- the L2 rule
@needs_qp
@pytest.mark.parametrize("f", [0.3, 0.5, 0.8, 0.9])
def test_stage2_rates_are_the_l2_formula_when_no_member_constraint_binds(f):
    comm = _community()
    (res,) = comm.micom(tradeoff=f, abundances=A_64)
    assert res.status == "optimal"
    assert res.community_growth_max == pytest.approx(14.0, abs=1e-9)
    s2 = sum(a * a for a in A_64.values())
    for mid, a in A_64.items():
        assert abs(res.growth_rates[mid] - f * 14.0 * a / s2) <= 1e-9
    assert res.l2_deviation <= 1e-9
    # the assumption behind the formula: the tradeoff row sits at its lower bound
    assert res.community_growth == pytest.approx(f * 14.0, abs=1e-9)
    # rate ratio = abundance ratio, whatever f
    assert res.growth_rates["Species0"] / res.growth_rates["Species1"] == pytest.approx(1.5, abs=1e-9)
    assert res.objective == pytest.approx(sum(mu * mu for mu in res.growth_rates.values()), rel=1e-9)
    # member growth in the solution is a_m mu_m and leaves through the member's drain
    for m in comm.members:
        v = res.fluxes[m.primary_biomass.id]
        assert v == pytest.approx(A_64[m.id] * res.growth_rates[m.id], abs=1e-9)
        assert res.fluxes[m.biomass_drain.id] == pytest.approx(v, abs=1e-9)
    assert res.fluxes[comm.primary_biomass.id] == pytest.approx(0.0, abs=1e-12)


@needs_qp
def test_a_binding_member_constraint_moves_the_rates_off_the_formula():
    # a = (0.5, 0.5), f = 0.9: the L2 point would give memB 6.3 > its carbon 4, so memB sits on
    # its cap (mu 8) and memA takes the rest of 12.6 (mu 17.2)
    comm = _community()
    (res,) = comm.micom(tradeoff=0.9)
    assert res.status == "optimal"
    assert res.growth_rates["Species1"] == pytest.approx(8.0, abs=1e-7)
    assert res.growth_rates["Species0"] == pytest.approx(17.2, abs=1e-7)
    assert res.l2_deviation == pytest.approx(4.6, abs=1e-7)


@needs_qp
def test_stage1_is_the_drains_open_maximum_with_taxon_unit_bounds():
    comm = _community()
    model = comm.util.model
    # memB's transporter limited to 3 per gDW of memB: at a_B = 0.4 that is 1.2 per gDW community
    model.reactions.get_by_id("rxn10000_c2").upper_bound = 3.0
    (res,) = comm.micom(tradeoff=0.5, abundances=A_64)

    # the same LP built independently on a copy
    ref = model.copy()
    for mid, rxns in reactions_by_member(comm).items():
        for r in rxns:
            rr = ref.reactions.get_by_id(r.id)
            rr.bounds = (A_64[mid] * r.lower_bound, A_64[mid] * r.upper_bound)
    ref.reactions.get_by_id("bio1").bounds = (0.0, 0.0)
    for m in comm.members:
        ref.reactions.get_by_id(m.biomass_drain.id).bounds = (0.0, 1000.0)
    ref.objective = {ref.reactions.get_by_id(m.primary_biomass.id): 1.0 for m in comm.members}
    ref.objective_direction = "max"
    F_ref = ref.slim_optimize()

    assert F_ref == pytest.approx(10.0 + 0.4 * 3.0, abs=1e-9)
    assert res.community_growth_max == pytest.approx(F_ref, abs=1e-9)
    # untouched bounds would allow 10 + 3
    assert model.reactions.get_by_id("rxn10000_c2").upper_bound == 3.0


@needs_qp
def test_abundances_are_an_input_and_the_composition_is_restored():
    comm = _community()
    before = _state(comm)
    (res,) = comm.micom(tradeoff=0.5, abundances={"Species0": 3, "Species1": 2})
    assert res.abundances == pytest.approx(A_64)
    assert _state(comm) == before
    assert comm.abundances == pytest.approx({"Species0": 0.5, "Species1": 0.5})
    with pytest.raises(ValueError, match="every member"):
        comm.micom(abundances={"Species0": 1.0})


@needs_qp
def test_min_growth_holds_each_member_in_both_stages():
    comm = _community()
    (res,) = comm.micom(tradeoff=0.3, abundances=A_64, min_growth={"Species1": 6.0})
    assert res.growth_rates["Species1"] == pytest.approx(6.0, abs=1e-7)   # the formula gives 3.23
    assert res.community_growth == pytest.approx(0.3 * 14.0, abs=1e-7)


@needs_qp
def test_pfba_holds_the_stage2_rates_and_minimises_flux():
    comm = _community()
    (plain,) = comm.micom(tradeoff=0.5, abundances=A_64)
    (res,) = comm.micom(tradeoff=0.5, abundances=A_64, pfba=True)
    assert res.stage_status["pfba"] == "optimal"
    assert res.growth_rates == pytest.approx(plain.growth_rates, abs=1e-9)
    for m in comm.members:
        assert res.fluxes[m.primary_biomass.id] == pytest.approx(
            A_64[m.id] * res.growth_rates[m.id], rel=1e-5)
    assert res.fluxes.abs().sum() <= plain.fluxes.abs().sum() + 1e-6


# --------------------------------------------------------------------------- restoration
@needs_qp
def test_micom_restores_bounds_rows_objective_composition_and_solver():
    comm = _community()
    comm.util.model.medium = {"EX_cpd00027_e0": 8.0, "EX_cpd00082_e0": 3.0}
    before = _state(comm)
    results = comm.micom([MEDIUM, {"EX_cpd00027_e0": 5}], tradeoff=0.7, abundances=A_64, pfba=True)
    assert [r.status for r in results] == ["optimal", "optimal"]
    assert results[1].community_growth_max == pytest.approx(5.0, abs=1e-9)   # cpd00082 closed
    assert _state(comm) == before
    assert comm.solution is results[-1].solution


@needs_qp
def test_micom_restores_everything_when_a_solve_raises(monkeypatch):
    comm = _community()
    before = _state(comm)

    def boom(model):
        raise RuntimeError("injected")
    monkeypatch.setattr(commmicom, "_solution", boom)
    with pytest.raises(RuntimeError, match="injected"):
        comm.micom(MEDIUM, tradeoff=0.5, abundances=A_64)
    assert _state(comm) == before


@needs_qp
def test_a_failed_stage1_is_a_status_not_an_exception():
    comm = _community()
    # a floor on bio1, as regularization leaves it, is infeasible once micom shuts bio1
    comm.util.create_constraint(comm.util.model.problem.Constraint(
        comm.primary_biomass.flux_expression, name="min_comm_growth", lb=1.0))
    (res,) = comm.micom(tradeoff=0.5)
    assert res.status.startswith("stage 1") and res.growth_rates is None and res.solution is None


def test_micom_raises_without_a_qp_backend_and_leaves_the_solver(monkeypatch):
    import optlang
    monkeypatch.setattr(optlang, "available_solvers", {}, raising=False)
    comm = _community()
    before = _state(comm)
    with pytest.raises(RuntimeError, match="QP-capable"):
        comm.micom(MEDIUM)
    assert _state(comm) == before


# --------------------------------------------------------------------------- legacy
@needs_qp
def test_micom_legacy_warns_and_reproduces_the_old_output():
    comm = _community()
    before = comm.util.model.solver.interface.__name__
    with pytest.warns(DeprecationWarning, match="tradeoff \\* mu_max"):
        sols = comm.micom_legacy(MEDIUM)            # the old default tradeoff 0.6
    # the old test (test_a_mscommsim.test_micom_keeps_a_qp_capable_incumbent_and_restores_the_solver)
    assert len(sols) == 1
    assert comm.util.model.solver.interface.__name__ == before
    # the old output: bio1 max is min(10, 4) / 0.5 = 8; bio1 held at 0.6 x 8 = 4.8 and every
    # member at the declared share, v_bio_m = 0.5 x 4.8, i.e. mu_m = 0.6 x mu_max for both
    fluxes = sols[0].fluxes
    assert fluxes["bio1"] == pytest.approx(4.8, abs=1e-6)
    assert fluxes["bio2"] == pytest.approx(2.4, abs=1e-6)
    assert fluxes["bio3"] == pytest.approx(2.4, abs=1e-6)
    # the comm_tradeoff row no longer outlives the call
    assert "comm_tradeoff" not in comm.util.model.constraints


# --------------------------------------------------------------------------- regularization (MS5)
def _glpk_community(members=None):
    comm = _community(members)
    comm.util.model.solver = "glpk"
    comm.set_objective(targets=[m.primary_biomass.forward_variable for m in comm.members])
    return comm


def _fail_bio1_max(monkeypatch, times):
    """Make the GLPK solve of max bio1 report 'infeasible' `times` times (math.inf: always),
    the way GLPK's warm basis did on CommPhitting maltose; every copy is affected too."""
    import optlang.glpk_interface as glpk
    real = glpk.Model.optimize
    left = {"n": times}

    def optimize(self):
        status = real(self)
        names = {v.name for v in self.objective.variables}
        if self.objective.direction == "max" and "bio1" in names and all(n.startswith("bio1") for n in names) \
                and left["n"] > 0:
            left["n"] -= 1
            self._status = "infeasible"
        return self._status
    monkeypatch.setattr(glpk.Model, "optimize", optimize)
    return left


def test_a_spurious_infeasible_floor_solve_is_retried_and_the_floor_installed(monkeypatch, caplog):
    comm = _glpk_community()
    left = _fail_bio1_max(monkeypatch, 1)
    with caplog.at_level(logging.WARNING, logger="mscommunity.mscommsim"):
        comm.regularization(linear=True)
    assert left["n"] == 0                                    # the injected failure did fire
    cons = {c.name: c for c in comm.util.model.constraints}
    assert "min_comm_growth" in cons
    # bio1 max = min(10, 4) / 0.5 = 8, times the growth fraction for two members
    assert cons["min_comm_growth"].lb == pytest.approx(8.0 * 0.999 * math.exp(-0.06), rel=1e-9)
    assert any("cold basis" in r.message for r in caplog.records)


def test_a_floor_solve_that_never_succeeds_raises(monkeypatch):
    comm = _glpk_community()
    before_obj = str(comm.util.model.objective.expression)
    _fail_bio1_max(monkeypatch, math.inf)
    with pytest.raises(FeasibilityError, match="min_comm_growth.*warm: infeasible.*cold basis: "
                                               "infeasible.*canonical copy: infeasible"):
        comm.regularization(linear=True)
    assert "min_comm_growth" not in comm.util.model.constraints
    assert str(comm.util.model.objective.expression) == before_obj


def _unequal_yields():
    """Both members on one carbon capped at 10; memB makes 2 biomass per carbon, memA 1. Solo
    maxima 10 and 20; max bio1 at (0.5, 0.5) is 40/3. With the floor bio1 >= 0.9408 * 40/3 both
    members need v_bio >= 6.27, the ladder settles on 0.3 and the answer is A 6.27 / B 7.47
    (a_A 0.456). Without the floor the 0.5 rung is feasible: A 5 / B 10 (a_A 1/3) -- the
    maltose symptom (the solo-floor answer) on a toy."""
    a, b = _competing()
    bio = b.reactions.get_by_id("bio1")
    bio.add_metabolites({b.metabolites.get_by_id("cpd00027_c0"): 0.5})   # -1 + 0.5 carbon per biomass
    return [a, b]


def test_predict_abundances_survives_a_spurious_floor_infeasible(monkeypatch):
    clean = _glpk_community(_unequal_yields()).predict_abundances(regularization=True, pfba=False)
    floor = 0.999 * math.exp(-0.06) * 40 / 3 * 0.5          # v_bio of each member at the floor
    assert clean["Species0"] == pytest.approx(floor / (floor + 2 * (10 - floor)), rel=1e-6)
    comm = _glpk_community(_unequal_yields())
    left = _fail_bio1_max(monkeypatch, 1)
    got = comm.predict_abundances(regularization=True, pfba=False)
    assert left["n"] == 0
    assert got == pytest.approx(clean, abs=1e-9)
    assert got["Species0"] != pytest.approx(1 / 3, abs=1e-3)   # what skipping the floor returned


def test_predict_abundances_cleans_up_when_regularization_raises(monkeypatch):
    comm = _glpk_community()
    model = comm.util.model
    timeout = model.solver.configuration.timeout
    obj = str(model.objective.expression)
    _fail_bio1_max(monkeypatch, math.inf)
    with pytest.raises(FeasibilityError):
        comm.predict_abundances(regularization=True, pfba=False, timeout=17)
    names = {c.name for c in model.constraints}
    assert not any("regularization" in n or n == "min_comm_growth" for n in names)
    assert model.solver.configuration.timeout == timeout
    assert str(model.objective.expression) == obj


# --------------------------------------------------------------------------- predict_abundances
def test_predict_abundances_sets_the_timeout_before_its_first_solve(monkeypatch):
    comm = _community()
    model = comm.util.model
    original = model.solver.configuration.timeout
    seen, real = [], model.slim_optimize

    def spy(*args, **kwargs):
        seen.append(model.solver.configuration.timeout)
        return real(*args, **kwargs)
    monkeypatch.setattr(model, "slim_optimize", spy)
    comm.predict_abundances(regularization=False, pfba=False, timeout=17)
    assert seen and seen[0] == 17
    assert model.solver.configuration.timeout == original


def test_dropped_kinetic_rows_are_logged_and_flagged(caplog, capsys):
    from cobra import Model  # noqa: F401
    from mscommunity.commhelper import build_from_species_models
    # kappa 0.5 < 1: a member's transporter alone carries v_bio, so the rows allow no growth
    comm = MSCommunity(model=build_from_species_models(_members()), kinetic_coeff=0.5)
    capsys.readouterr()
    assert comm.kinetic_rows_dropped is False
    with caplog.at_level(logging.WARNING, logger="mscommunity.mscommsim"):
        abund = comm.predict_abundances(regularization=False, pfba=False)
    assert abund == pytest.approx({"Species0": 10 / 14, "Species1": 4 / 14}, rel=1e-6)
    assert comm.kinetic_rows_dropped is True
    assert any("kinetic rows" in r.message and "dropped" in r.message for r in caplog.records)
    assert "Kinetic constraints disabled" not in capsys.readouterr().out
    # the rows are back, and the next call that does not drop them resets the flag
    assert {"Species0_commKin", "Species1_commKin"} <= {c.name for c in comm.util.model.constraints}
    comm.add_commkinetics(750)
    comm.predict_abundances(regularization=False, pfba=False)
    assert comm.kinetic_rows_dropped is False


# --------------------------------------------------------------------------- real model
MM = "/home/freiburger/Documents/3H11_R12_SynCom/Metabolic_modeling"
RESULTS = os.path.join(MM, "data/multisystem3/abundance/micom/results.json")

_ANCHOR = r"""
import json, sys
sys.path.insert(0, {mm!r})
import mscommunity
assert mscommunity.__file__.startswith({wt!r}), mscommunity.__file__
import abund_common as AC
b = AC.build('maltose', solver='gurobi')
b.model.solver.problem.Params.Threads = 4
a = {{m.id: m.abundance for m in b.msc.members}}
(res,) = b.msc.micom(tradeoff=0.99)
print('ANCHOR' + json.dumps({{'a': a, 'status': res.status, 'mu': res.growth_rates,
                             'F': res.community_growth_max, 'dev': res.l2_deviation}}))
"""


@needs_qp
@pytest.mark.skipif(not os.path.exists(RESULTS), reason="the 3H11_R12_SynCom MICOM results are not on this machine")
def test_faithful_micom_reproduces_the_prototype_on_commphitting_maltose():
    """micom at f 0.99 on the CommPhitting maltose community (abund_common.build('maltose'),
    measured composition) against abund_micom_lib's faithful arm: E. coli 0.3155 f and
    P. fluorescens 0.0470 f (results.json -> A.maltose.arms.faithful.rows[f = 0.99].mu).
    Built in a fresh interpreter, as the campaign does."""
    ref = json.load(open(RESULTS))["A"]["maltose"]["arms"]["faithful"]
    row = next(r for r in ref["rows"] if r["f"] == 0.99)
    env = dict(os.environ, PYTHONPATH=WORKTREE, OMP_NUM_THREADS="4", JAX_PLATFORMS="cpu")
    out = subprocess.run([sys.executable, "-c", _ANCHOR.format(mm=MM, wt=WORKTREE)], cwd=MM, env=env,
                         capture_output=True, text=True, timeout=1800)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("ANCHOR")]
    assert lines, f"the maltose build failed:\n{out.stdout[-3000:]}\n{out.stderr[-3000:]}"
    got = json.loads(lines[-1][len("ANCHOR"):])
    assert got["status"] == "optimal"
    assert got["a"] == pytest.approx(ref["a"], abs=1e-12)
    assert got["F"] == pytest.approx(ref["F_max_sum_vbio"], abs=1e-6)
    for member in ("ecoli", "pf"):
        assert abs(got["mu"][member] - row["mu"][member]) <= 1e-6
    assert got["mu"]["ecoli"] == pytest.approx(0.3155 * 0.99, abs=1e-4)
    assert got["mu"]["pf"] == pytest.approx(0.0470 * 0.99, abs=1e-4)
    assert got["dev"] <= 1e-6
