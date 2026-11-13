# -*- coding: utf-8 -*-
"""ANN surrogates of community FBA (mscommunity.commsurrogate; Song et al. 2025, Sci Rep 15:6042).

Two synthetic models whose FBA optimum is known in closed form:

    community   memA eats cpd00027, memB eats cpd00082, bio1 = 0.5 memA + 0.5 memB, so the
                community grows at 2 min(uA, uB) and each member makes min(uA, uB) biomass
    organism    glucose (cpd00027) goes to biomass or to secreted acetate (cpd00029); an
                optional maintenance drain forces `maintenance` of it through, so the LP is
                infeasible below that uptake
"""
from __future__ import annotations

import numpy as np
import pytest
from cobra import Metabolite, Model, Reaction
from pandas import DataFrame

from mscommunity.commhelper import build_from_species_models
from mscommunity.commsurrogate import (FBASurrogate, community_outputs, community_to_nn, sample_fba,
                                       train_surrogate)
from mscommunity.mscommsim import MSCommunity

GLC, ACE, XYL = "cpd00027", "cpd00029", "cpd00082"


def _member(mid, carbon, uptake):
    m = Model(mid)
    mets = {f"{cid}_{comp}": Metabolite(f"{cid}_{comp}", name=f"{cid}_{comp}", compartment=comp)
            for cid, comp in [(carbon, "e0"), (carbon, "c0"), ("cpd11416", "c0")]
                             # ATP, H2O, ADP, Pi, H+: MSCommunity builds each member's ATP hydrolysis
                             + [(cid, "c0") for cid in ("cpd00002", "cpd00001", "cpd00008", "cpd00009", "cpd00067")]}
    m.add_metabolites(list(mets.values()))
    ex = Reaction(f"EX_{carbon}_e0", lower_bound=-float(uptake), upper_bound=1000)
    ex.add_metabolites({mets[f"{carbon}_e0"]: -1})
    tr = Reaction("rxn10000_c0", lower_bound=0, upper_bound=1000)
    tr.add_metabolites({mets[f"{carbon}_e0"]: -1, mets[f"{carbon}_c0"]: 1})
    bio = Reaction("bio1", lower_bound=0, upper_bound=1000)
    bio.add_metabolites({mets[f"{carbon}_c0"]: -1, mets["cpd11416_c0"]: 1})
    m.add_reactions([ex, tr, bio])
    m.objective = "bio1"
    return m


def _community():
    return MSCommunity(model=build_from_species_models([_member("memA", GLC, 10), _member("memB", XYL, 4)]))


def _organism(maintenance=0.0):
    m = Model("organism")
    met = {key: Metabolite(key, compartment=key[-2:]) for key in
           (f"{GLC}_e0", f"{GLC}_c0", f"{ACE}_c0", f"{ACE}_e0", "cpd11416_c0")}
    def rxn(rid, stoich, lb=0.0, ub=1000.0):
        r = Reaction(rid, lower_bound=lb, upper_bound=ub)
        r.add_metabolites({met[k]: v for k, v in stoich.items()})
        return r
    m.add_reactions([
        rxn(f"EX_{GLC}_e0", {f"{GLC}_e0": -1}, lb=-10),
        rxn(f"EX_{ACE}_e0", {f"{ACE}_e0": -1}),
        rxn("rxn00001_c0", {f"{GLC}_e0": -1, f"{GLC}_c0": 1}),
        rxn("rxn00002_c0", {f"{GLC}_c0": -1, f"{ACE}_c0": 1}),
        rxn("rxn00003_c0", {f"{ACE}_c0": -1, f"{ACE}_e0": 1}),
        rxn("rxn00004_c0", {f"{GLC}_c0": -1}, lb=maintenance),
        rxn("bio1", {f"{GLC}_c0": -1, "cpd11416_c0": 1}),
        rxn("SK_cpd11416_c0", {"cpd11416_c0": -1}),
    ])
    m.objective = "bio1"
    return m


# --- sampling -------------------------------------------------------------------

def test_community_samples_are_the_fba_optimum_at_each_draw():
    comm = _community()
    inputs = [f"EX_{GLC}_e0", f"EX_{XYL}_e0"]
    outputs = community_outputs(comm, inputs, member_exchanges=True)
    X, Y = sample_fba(comm.util.model, inputs, outputs, n_samples=200, seed=3)
    # each input is drawn on [0, its current uptake limit]
    assert X[inputs[0]].between(0, 10).all() and X[inputs[1]].between(0, 4).all()
    low = np.minimum(X[inputs[0]], X[inputs[1]])
    assert np.allclose(Y["community_biomass"], 2 * low, atol=1e-6)
    for mem in comm.members:
        assert np.allclose(Y[f"{mem.id}_biomass"], low, atol=1e-6)
    # actual exchange fluxes come back signed (uptake < 0), and so does each member's own share
    assert np.allclose(Y[inputs[0]], -low, atol=1e-6) and np.allclose(Y[inputs[1]], -low, atol=1e-6)
    assert np.allclose(Y[f"Species0_{GLC}_e0"], -low, atol=1e-6)
    assert np.allclose(Y[f"Species1_{XYL}_e0"], -low, atol=1e-6)
    # reproducible draw for draw
    X2, Y2 = sample_fba(comm.util.model, inputs, outputs, n_samples=200, seed=3)
    assert X.equals(X2) and np.allclose(Y, Y2)


def test_sampling_leaves_the_model_as_it_was():
    comm = _community()
    before = {r.id: r.bounds for r in comm.util.model.reactions}
    objective = str(comm.util.model.objective.expression)
    sample_fba(comm.util.model, [f"EX_{GLC}_e0"], ["bio1"], n_samples=20)
    assert {r.id: r.bounds for r in comm.util.model.reactions} == before
    assert str(comm.util.model.objective.expression) == objective


def test_lexicographic_steps_hold_each_earlier_optimum_at_its_fraction():
    # max biomass (= u), then max acetate with biomass >= 0.6 u  ->  acetate = 0.4 u
    org = _organism()
    X, Y = sample_fba(org, {f"EX_{GLC}_e0": 10}, ["bio1", f"EX_{ACE}_e0"], n_samples=50,
                      steps=[("bio1", 0.6), f"EX_{ACE}_e0"])
    u = X[f"EX_{GLC}_e0"]
    assert np.allclose(Y["bio1"], 0.6 * u, atol=1e-5)
    assert np.allclose(Y[f"EX_{ACE}_e0"], 0.4 * u, atol=1e-5)
    with pytest.raises(ValueError, match="fraction"):
        sample_fba(org, {f"EX_{GLC}_e0": 10}, ["bio1"], steps=["bio1", f"EX_{ACE}_e0"])
    with pytest.raises(KeyError, match="EX_nothing"):
        sample_fba(org, {f"EX_{GLC}_e0": 10}, ["bio1"], steps=[("bio1", 0.5), "EX_nothing"])
    with pytest.raises(ValueError, match="not an exchange"):
        sample_fba(org, {"rxn00002_c0": 10}, ["bio1"])


def test_infeasible_draws_are_dropped_or_recorded_as_zero():
    org = _organism(maintenance=1.0)          # infeasible below 1 mmol/gDW/h of glucose
    box = {f"EX_{GLC}_e0": 10}
    X, Y = sample_fba(org, box, ["bio1"], n_samples=300, seed=1)
    assert len(X) < 300 and (X[f"EX_{GLC}_e0"] >= 1 - 1e-9).all()
    assert np.allclose(Y["bio1"], X[f"EX_{GLC}_e0"] - 1, atol=1e-6)
    Xz, Yz = sample_fba(org, box, ["bio1"], n_samples=300, seed=1, infeasible="zero")
    assert len(Xz) == 300
    below = Xz[f"EX_{GLC}_e0"] < 1
    assert below.sum() == 300 - len(X) and (Yz.loc[below, "bio1"] == 0).all()


def test_sampling_statistics_travel_with_the_samples_and_the_surrogate(tmp_path):
    org = _organism(maintenance=1.0)          # ~10% of draws from [0, 10] are infeasible
    box = {f"EX_{GLC}_e0": 10}
    X, Y = sample_fba(org, box, ["bio1"], n_samples=300, seed=1)
    stats = X.attrs["sampling"]
    assert stats == Y.attrs["sampling"]
    assert stats["n_draws"] == 300 and stats["n_kept"] == stats["n_optimal"] == len(X) < 300
    assert sum(stats["statuses"].values()) == 300 and stats["statuses"]["optimal"] == len(X)
    # pfba raises on an infeasible LP, which the cpu backend records as "error:Infeasible"
    assert sum(n for s, n in stats["statuses"].items() if s != "optimal") == 300 - len(X)
    assert set(stats["statuses"]) == {"optimal", "error:Infeasible"}
    assert stats["box"] == {f"EX_{GLC}_e0": [0.0, 10.0]}
    # the network never sees below the maintenance demand, whatever the box says
    lo, hi = stats["kept_range"][f"EX_{GLC}_e0"]
    assert lo >= 1 - 1e-9 and lo == X[f"EX_{GLC}_e0"].min() and hi == X[f"EX_{GLC}_e0"].max()
    Xz, _ = sample_fba(org, box, ["bio1"], n_samples=300, seed=1, infeasible="zero")
    zs = Xz.attrs["sampling"]
    assert zs["n_kept"] == 300 and zs["n_optimal"] == len(X) and zs["statuses"] == stats["statuses"]
    assert zs["infeasible"] == "zero"
    # train_surrogate and community_to_nn carry it, and so does the JSON
    surr = train_surrogate(X, Y, hidden_layers=1, nodes=3)
    assert surr.sampling == stats
    surr.save(tmp_path / "s.json")
    assert FBASurrogate.load(tmp_path / "s.json").sampling == stats
    surr2 = community_to_nn(org, inputs=box, n_samples=100, seed=1, hidden_layers=1, nodes=3)
    assert surr2.sampling["n_draws"] == 100 and surr2.sampling["n_kept"] == surr2.sampling["n_optimal"] < 100
    # frames that did not come from sample_fba (e.g. read back from a CSV) have none
    bare = train_surrogate(DataFrame(X.to_numpy(), columns=X.columns), DataFrame(Y.to_numpy(), columns=Y.columns),
                           hidden_layers=1, nodes=3)
    assert bare.sampling is None


def _pinned(bounds, flipped=False):
    """The organism with its glucose exchange bounds set to `bounds`; `flipped` writes the
    exchange as `<=> glc_e0`, so that its uptake runs on the upper bound."""
    org = _organism()
    ex = org.reactions.get_by_id(f"EX_{GLC}_e0")
    if flipped:
        ex.add_metabolites({org.metabolites.get_by_id(f"{GLC}_e0"): 2})
    ex.bounds = bounds
    return org


def test_a_pinned_input_is_refused_instead_of_silently_truncating_the_box():
    ex = f"EX_{GLC}_e0"
    # the validated-uptake equality, a forced minimum, and the same on a `<=> met` exchange
    for org, forced in ((_pinned((-5.0, -5.0)), 5), (_pinned((-10.0, -2.0)), 2), (_pinned((2.0, 10.0), True), 2)):
        with pytest.raises(ValueError, match=rf"force an uptake of at least {forced}.*lower bound > upper bound"):
            sample_fba(org, {ex: 10}, ["bio1"], n_samples=20)
        with pytest.raises(ValueError, match="Lift the pin"):
            community_to_nn(org, inputs={ex: 10}, n_samples=20)
    # a bare id samples [0, current uptake limit], which a pin makes [0, pin]: refused too
    with pytest.raises(ValueError, match="force an uptake"):
        sample_fba(_pinned((-5.0, -5.0)), [ex], ["bio1"], n_samples=20)
    # starting the range at the pin is fine, and so is an exchange that merely forbids secretion
    X, Y = sample_fba(_pinned((-5.0, -5.0)), {ex: (5.0, 10.0)}, ["bio1"], n_samples=50)
    assert len(X) == 50 and X.attrs["sampling"]["statuses"] == {"optimal": 50}
    assert np.allclose(Y["bio1"], X[ex], atol=1e-6)
    X, _ = sample_fba(_pinned((-10.0, 0.0)), {ex: 10}, ["bio1"], n_samples=50)
    assert len(X) == 50 and X[ex].min() < 1


# --- the network ------------------------------------------------------------------

def test_the_surrogate_is_the_closed_form_of_the_fitted_network(tmp_path):
    org = _organism()
    X, Y = sample_fba(org, {f"EX_{GLC}_e0": 10}, ["bio1", f"EX_{ACE}_e0"], n_samples=200,
                      steps=[("bio1", 0.6), f"EX_{ACE}_e0"])
    surr = train_surrogate(X, Y, hidden_layers=2, nodes=6)
    assert surr.hidden_layers == (6, 6)
    # numpy forward pass == sklearn's, once mapminmax is applied and reversed
    Xa = X.to_numpy()
    scaled = surr.estimator.predict((Xa - surr.x_offset) * surr.x_gain - 1)
    assert np.allclose(surr.predict(Xa), (scaled + 1) / surr.y_gain + surr.y_offset)
    # every input form gives the same numbers
    point = {f"EX_{GLC}_e0": 5.0}
    as_dict = surr.predict(point)
    assert np.allclose([as_dict[o] for o in surr.outputs], surr.predict(np.array([5.0])))
    assert np.allclose(surr.predict(X).to_numpy(), surr.predict(Xa))
    # and survive a round trip through JSON
    surr.save(tmp_path / "surrogate.json")
    loaded = FBASurrogate.load(tmp_path / "surrogate.json")
    assert np.allclose(loaded.predict(Xa), surr.predict(Xa))
    assert loaded.input_bounds == surr.input_bounds == {f"EX_{GLC}_e0": (float(X.min().iloc[0]), float(X.max().iloc[0]))}
    # linear in u on this model, so near-perfect everywhere
    assert (surr.scores["test"] > 0.9999).all()


def test_clip_and_the_exhaustion_guard_bound_what_the_network_answers(tmp_path):
    comm = _community()
    inputs = [f"EX_{GLC}_e0", f"EX_{XYL}_e0"]
    X, Y = sample_fba(comm.util.model, inputs, community_outputs(comm, inputs), n_samples=300, seed=2)
    surr = train_surrogate(X, Y, hidden_layers=1, nodes=4)
    lo, hi = surr.output_limits()
    assert surr.output_bounds and np.all(lo <= hi)
    assert np.all(lo >= Y.min().to_numpy() - 1e-12) and np.all(hi <= Y.max().to_numpy() + 1e-12)
    # well outside the box the network extrapolates; clipped, it stays inside what the LP produced
    far = np.array([[40.0, 20.0], [-5.0, -5.0], [0.0, 0.0]])
    raw, clipped = surr.predict(far), surr.predict(far, clip="training")
    assert np.all((clipped >= lo - 1e-12) & (clipped <= hi + 1e-12))
    assert not np.allclose(raw, clipped)
    # the guard zeroes every output where every named substrate is exhausted, and only there
    guard = {f"EX_{GLC}_e0": 0.0, f"EX_{XYL}_e0": 0.0}
    pts = np.array([[0.0, 0.0], [0.0, 3.0], [5.0, 0.0], [5.0, 3.0]])
    raw, guarded = surr.predict(pts), surr.predict(pts, exhausted=guard)
    assert np.all(guarded[0] == 0.0) and np.allclose(guarded[1:], raw[1:])
    # a threshold above 0 cuts in below it; a list of inputs means threshold 0
    assert np.all(surr.predict(np.array([0.05, 0.05]), exhausted={f"EX_{GLC}_e0": 0.1, f"EX_{XYL}_e0": 0.1}) == 0)
    assert np.all(surr.predict(np.array([0.0, 0.0]), exhausted=inputs) == 0)
    # one rule for every input form, and the guard wins over the clip
    as_dict = surr.predict({inputs[0]: pts[:, 0], inputs[1]: pts[:, 1]}, clip="training", exhausted=guard)
    as_frame = surr.predict(DataFrame(pts, columns=inputs), clip="training", exhausted=guard)
    as_array = surr.predict(pts, clip="training", exhausted=guard)
    assert np.allclose(np.column_stack([as_dict[o] for o in surr.outputs]), as_array)
    assert np.allclose(as_frame.to_numpy(), as_array) and np.all(as_array[0] == 0)
    with pytest.raises(ValueError, match="clip"):
        surr.predict(pts, clip="box")
    with pytest.raises(KeyError, match="exhausted"):
        surr.predict(pts, exhausted={"EX_cpd00007_e0": 0.0})
    # output_bounds survive JSON; a surrogate saved before it existed recovers it from mapminmax
    surr.save(tmp_path / "s.json")
    loaded = FBASurrogate.load(tmp_path / "s.json")
    assert loaded.output_bounds == surr.output_bounds
    legacy = surr.to_dict()
    del legacy["output_bounds"], legacy["sampling"]
    old = FBASurrogate.from_dict(legacy)
    assert old.output_bounds == {} and old.sampling is None
    assert np.allclose(np.vstack(old.output_limits()), np.vstack(surr.output_limits()), rtol=1e-12, atol=1e-12)


def test_grid_search_keeps_the_smallest_network_within_tolerance():
    comm = _community()
    inputs = [f"EX_{GLC}_e0", f"EX_{XYL}_e0"]
    X, Y = sample_fba(comm.util.model, inputs, ["bio1"], n_samples=300)
    surr = train_surrogate(X, Y, hidden_layers=[1, 2], nodes=[2, 8], repeats=2, tolerance=0.1)
    grid = surr.grid
    assert len(grid) == 4 and grid["selected"].sum() == 1
    chosen = grid.index[grid["selected"]][0]
    best = grid["validation_mse"].min()
    assert grid.loc[chosen, "validation_mse"] <= best * 1.1 + 1e-15
    # nothing tried before it (the smaller networks) was good enough
    assert (grid.loc[:chosen - 1, "validation_mse"] > best * 1.1 + 1e-15).all()
    row = grid.loc[chosen]
    assert surr.hidden_layers == (int(row["nodes"]),) * int(row["hidden_layers"])


def test_community_converts_into_a_trustworthy_surrogate():
    comm = _community()
    surr = comm.to_neural_network(n_samples=600, hidden_layers=2, nodes=8, member_exchanges=True)
    # default inputs: the medium's exchanges, on [0, their uptake limits]
    assert surr.input_bounds == {f"EX_{GLC}_e0": (0.0, 10.0), f"EX_{XYL}_e0": (0.0, 4.0)}
    assert {"Species0_biomass", "Species1_biomass", "community_biomass", f"EX_{GLC}_e0",
            f"EX_{XYL}_e0", f"Species0_{GLC}_e0", f"Species1_{XYL}_e0"} <= set(surr.outputs)
    assert (surr.scores["test"] > 0.999).all()
    # it reproduces the optimum away from the samples: 2 min(uA, uB)
    pred = surr.predict({f"EX_{GLC}_e0": [7.0, 2.0], f"EX_{XYL}_e0": [3.0, 3.5]})
    assert np.allclose(pred["community_biomass"], [6.0, 4.0], atol=0.1)


def test_a_single_organism_model_converts_too():
    surr = community_to_nn(_organism(), inputs={f"EX_{GLC}_e0": 10}, extra_outputs=[f"EX_{ACE}_e0"],
                           steps=[("bio1", 0.6), f"EX_{ACE}_e0"], n_samples=200, hidden_layers=1, nodes=4)
    assert surr.outputs == ("bio1", f"EX_{GLC}_e0", f"EX_{ACE}_e0")
    assert np.allclose(list(surr.predict({f"EX_{GLC}_e0": 5.0}).values()), [3.0, -5.0, 2.0], atol=0.02)
    with pytest.raises(ValueError, match="MSCommunity"):
        community_to_nn(_organism(), member_exchanges=True)
