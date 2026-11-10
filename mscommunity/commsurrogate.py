# -*- coding: utf-8 -*-
"""Neural-network surrogates of community FBA (Song et al. 2025, Sci Rep 15:6042).

Song et al. replace the LP that a dynamic-FBA or reactive-transport model solves at every
time step and grid cell by a feed-forward ANN trained on randomly sampled FBA solutions, so
the metabolism enters the ODE/PDE right-hand side as a closed-form algebraic expression:

  1. draw the uptake UPPER LIMITS of the limiting nutrients uniformly on [0, max];
  2. solve a (multi-step, lexicographic) FBA at each draw and record the outputs -- growth,
     byproduct secretion, and the ACTUAL flux of every input, which need not equal its limit
     (oxygen, in their Fig. S1);
  3. fit one multi-input multi-output network: inputs mapped onto [-1, 1] (mapminmax), tanh
     hidden layers (tansig), a linear output layer (purelin), outputs mapped back. Their MIMO
     networks had 5 layers of 10 nodes, chosen by grid search as the smallest that no larger
     one improved on, from a 70/15/15 train/validation/test split of 5000 solutions;
  4. drop the trained network into the ODE as its source/sink term.

`community_to_nn` does this for a compartmentalized MSCommunity: the inputs are uptake limits
on the shared extracellular (e0) exchanges and the outputs are each member's biomass flux,
the community biomass, the community's actual exchange fluxes and, optionally, each member's
own net exchange of a shared metabolite (read off its compartment's transporters).
`sample_fba` and `train_surrogate` are the two halves, for when the samples should be kept
and refit (e.g. a wider hyperparameter grid) without solving the LPs again.

Two departures from the paper, both deliberate:
  * training is L2-regularised L-BFGS (scikit-learn's MLPRegressor) rather than MATLAB's
    Bayesian-regularised Levenberg-Marquardt (trainbr); the network itself is the same;
  * a draw whose LP is not optimal is dropped by default (`infeasible="drop"`). The authors'
    code recorded zeros instead, which `infeasible="zero"` reproduces -- that teaches the
    surrogate "no activity" where the LP has no solution, e.g. below a maintenance demand.

The cybernetic combination of substrate-specific surrogates (their u_Lac, u_Pyr, u_Ace) is
downstream of this module: train one surrogate per growth mode by narrowing `inputs`.
"""
from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
from cobra import Model
from pandas import DataFrame

from mscommunity.batched_lp import CommunityProblem, LPInstance, get_batched_solver

logger = logging.getLogger(__name__)

Box = Tuple[float, float]
Output = Union[str, Mapping[str, float]]


# --- inputs and outputs ------------------------------------------------------

def _exchange(model: Model, ident: str):
    """The exchange reaction `ident` names: its id, or its metabolite with or without `_e0`."""
    for rid in (ident, f"EX_{ident}", f"EX_{ident}_e0"):
        if rid in model.reactions:
            rxn = model.reactions.get_by_id(rid)
            # an internal reaction would have its lower bound "set to -uptake" without complaint
            if not rxn.boundary:
                raise ValueError(f"{rid} is not an exchange of {model.id}, so it has no uptake limit")
            return rxn
    raise KeyError(f"{ident!r} names no exchange reaction of {model.id}")


def _max_uptake(rxn) -> float:
    # cobra's medium convention: uptake runs on the lower bound of `met <=>`, and on the
    # upper bound of the rarer `<=> met`
    return max(0.0, -rxn.lower_bound if rxn.reactants else rxn.upper_bound)


def _input_box(model: Model, inputs) -> Dict[str, Box]:
    """{exchange id: (min, max) uptake} from a medium-style dict, a list of ids, or a mix.

    A number is the maximum uptake (sampled from 0, as the paper does), a pair is the range,
    and a bare id takes the range [0, its current uptake limit]."""
    if isinstance(inputs, Mapping):
        items = list(inputs.items())
    else:
        items = [(ident, None) for ident in inputs]
    box = {}
    for ident, spec in items:
        rxn = _exchange(model, ident)
        if spec is None:
            spec = _max_uptake(rxn)
        lo, hi = (0.0, float(spec)) if np.isscalar(spec) else map(float, spec)
        if not 0 <= lo <= hi:
            raise ValueError(f"the uptake range of {rxn.id} must satisfy 0 <= min <= max, not {(lo, hi)}")
        if hi == lo:
            logger.warning("Input %s has the degenerate uptake range [%s, %s]; pass {%r: max} "
                           "to sample it.", rxn.id, lo, hi, rxn.id)
        box[rxn.id] = (lo, hi)
    if not box:
        raise ValueError("No inputs: pass the exchanges whose uptake limits the surrogate should take.")
    return box


def _output_map(outputs) -> Dict[str, Dict[str, float]]:
    """{label: {reaction id: coefficient}}: every output is a linear functional of the fluxes."""
    if not isinstance(outputs, Mapping):
        outputs = {rid: rid for rid in outputs}
    return {label: ({spec: 1.0} if isinstance(spec, str) else {r: float(c) for r, c in spec.items()})
            for label, spec in outputs.items()}


def community_outputs(community, inputs: Sequence[str], extra_outputs=None,
                      member_exchanges=False) -> Dict[str, Dict[str, float]]:
    """The default outputs of a community surrogate.

    `<member>_biomass` (each member's primary biomass flux), `community_biomass` (bio1), the
    actual flux of every input exchange, then `extra_outputs` (reaction ids, or
    {label: reaction id | {reaction id: coefficient}}). With `member_exchanges`, also
    `<member>_<met>_e0`: the member's net secretion of that shared metabolite into e0
    (negative = uptake), summed over the transporters of its own compartment. True takes the
    metabolites of every exchange among the inputs and extra outputs; a list names them.
    A member's share can be non-unique even under pFBA -- the community optimum often leaves
    open which member takes up a nutrient both can use -- and then no network can learn it;
    its validation score in `FBASurrogate.scores` says so."""
    model = community.util.model
    outputs = {f"{mem.id}_biomass": {mem.primary_biomass.id: 1.0} for mem in community.members}
    outputs["community_biomass"] = {community.primary_biomass.id: 1.0}
    outputs.update({rid: {rid: 1.0} for rid in inputs})
    if extra_outputs is not None:
        outputs.update(_output_map(extra_outputs))
    if member_exchanges:
        if member_exchanges is True:
            rxns = [model.reactions.get_by_id(rid) for rid in sorted({rid for spec in outputs.values() for rid in spec})]
            mets = [met for rxn in rxns if rxn in model.exchanges for met in rxn.metabolites]
        else:
            mets = [next(iter(_exchange(model, ident).metabolites)) for ident in member_exchanges]
        for mem in community.members:
            for met in mets:
                coefs = {rxn.id: float(rxn.metabolites[met]) for rxn in mem.reactions if met in rxn.metabolites}
                if coefs:
                    outputs[f"{mem.id}_{met.id}"] = coefs
    return outputs


# --- sampling -----------------------------------------------------------------

def _floor(optimum: float, alpha: float, tol: float = 1e-7) -> float:
    """The lower bound `target >= alpha * optimum` of a lexicographic step, written so that it
    still relaxes the optimum when the optimum is negative, and kept a hair below the optimum
    so that alpha = 1 does not make the next LP numerically infeasible."""
    return optimum - (1.0 - alpha) * abs(optimum) - tol * max(1.0, abs(optimum))


def _steps(steps) -> List[Tuple[Optional[str], Optional[float]]]:
    if steps is None:
        return [(None, None)]
    out = []
    for k, step in enumerate(steps):
        target, alpha = (step, None) if isinstance(step, str) else step
        if k < len(steps) - 1 and (alpha is None or not 0 <= alpha <= 1):
            raise ValueError(f"step {k} ({target}) needs a fraction 0 <= alpha <= 1 of its optimum to "
                             f"hold while the later steps are optimised, not {alpha!r}")
        out.append((target, alpha))
    return out


def sample_fba(model: Model, inputs, outputs, n_samples: int = 5000, steps=None, pfba: bool = True,
               seed: int = 0, infeasible: str = "drop", backend: str = "cpu", workers: int = 1,
               batch_size: Optional[int] = None) -> Tuple[DataFrame, DataFrame]:
    """FBA solutions at uptake limits drawn uniformly from a box (Song et al.'s training data).

    model       the cobra model to solve, as configured: its bounds, objective and every
                constraint a package added (an MSCommunity's CommKinetics, closed member
                drains, element-uptake limits) shape the samples. It is not changed.
    inputs      the exchanges to vary: {id: max uptake}, {id: (min, max)}, or ids, whose range
                is then [0, current uptake limit]. Uptakes are positive magnitudes.
    outputs     what to record: reaction ids, or {label: reaction id | {reaction id: coef}}.
                Fluxes are net and signed (uptake < 0), as the LP reports them.
    steps       None maximises the model's own objective. Otherwise the multi-step LP of the
                paper: [(reaction, alpha), ..., reaction] maximises each reaction in turn,
                holding every earlier one at >= alpha times its optimum. Their lactate case
                was [("bio1", 0.6721), ("EX_cpd00020_e0", 0.6848), "EX_cpd00029_e0"]: max
                biomass, then pyruvate at 67% of max biomass, then acetate at 68% of max
                pyruvate. Those fractions were fitted to S. oneidensis yields; they are not
                defaults for another model.
    pfba        finish the last step with pFBA (min total flux at its optimum), which makes the
                recorded fluxes unique, and so learnable, where the LP has alternate optima.
    infeasible  "drop" a draw whose LP is not optimal, or record it as all-"zero" outputs.
    backend     the batched-LP backend. "cpu" solves the cobra model itself; the matrix
                backends drop non-antisymmetric rows -- which is what a CommKinetics
                constraint is -- so for an MSCommunity they solve a relaxation.

    Returns (X, Y): DataFrames of the drawn uptake limits (columns = input exchanges) and the
    recorded outputs (columns = output labels), one row per kept draw.
    """
    if infeasible not in ("drop", "zero"):
        raise ValueError("infeasible must be 'drop' or 'zero'")
    box = _input_box(model, inputs)
    outmap = _output_map(outputs)
    plan = _steps(steps)
    named = {rid for spec in outmap.values() for rid in spec} | {t for t, _ in plan if t is not None}
    missing = sorted(named - set(model.reactions.list_attr("id")))
    if missing:
        raise KeyError(f"outputs or steps name reactions {model.id} does not have: {missing}")

    names = list(box)
    lo, hi = (np.array([box[n][i] for n in names]) for i in (0, 1))
    X = np.random.default_rng(seed).uniform(lo, hi, size=(int(n_samples), len(names)))
    rxns = [model.reactions.get_by_id(n) for n in names]
    # each input exchange's uptake bound moves with the draw; its other bound stays put
    on_lb = [bool(r.reactants) for r in rxns]
    static = [r.upper_bound if lb else r.lower_bound for r, lb in zip(rxns, on_lb)]

    def patch(row):
        return {r.id: ((-u, s) if lb else (s, u)) for r, u, lb, s in zip(rxns, row, on_lb, static)}

    if backend == "cpu":
        # The cobra backend solves `problem.model` directly and never reads the matrices, so do
        # not extract them: extraction drops (and warns about) the very constraints this
        # backend keeps.
        problem = CommunityProblem(model=model, var_names=(), var_index={}, lb=np.empty(0),
                                   ub=np.empty(0), c=np.empty(0), sense=model.objective.direction,
                                   S=sp.csr_matrix((0, 0)), met_names=())
    else:
        problem = CommunityProblem.from_model(model)
    solver = get_batched_solver(backend, workers=workers)
    # A BatchedSolution carries every flux of the model, so solve in chunks and keep only the
    # outputs; ~2M stored fluxes per chunk whatever the model's size.
    batch_size = batch_size or max(1, min(500, 2_000_000 // max(1, len(model.reactions))))

    Y = np.zeros((len(X), len(outmap)))
    ok = np.ones(len(X), dtype=bool)
    statuses: Dict[str, int] = {}
    for start in range(0, len(X), batch_size):
        idx = range(start, min(start + batch_size, len(X)))
        bounds = {i: patch(X[i]) for i in idx}
        alive = list(idx)
        for k, (target, alpha) in enumerate(plan):
            last = k == len(plan) - 1
            instances = [LPInstance(id=str(i), bounds=bounds[i], pfba=pfba and last,
                                    objective=None if target is None else {target: 1.0},
                                    sense=None if target is None else "max") for i in alive]
            survivors = []
            for i, sol in zip(alive, solver.solve(problem, instances)):
                if sol.status != "optimal":
                    ok[i] = False
                    statuses[sol.status] = statuses.get(sol.status, 0) + 1
                elif last:
                    Y[i] = [sum(c * sol.fluxes[r] for r, c in spec.items()) for spec in outmap.values()]
                else:
                    tlo, thi = bounds[i].get(target, model.reactions.get_by_id(target).bounds)
                    bounds[i] = {**bounds[i], target: (max(tlo, _floor(sol.fluxes[target], alpha)), thi)}
                    survivors.append(i)
            alive = survivors if not last else alive
        logger.info("sample_fba(%s): %d/%d draws solved", model.id, idx.stop, len(X))

    if not ok.all():
        logger.warning("sample_fba(%s): %d of %d draws had no optimal LP (%s); %s.", model.id,
                       int((~ok).sum()), len(X), statuses,
                       "dropped" if infeasible == "drop" else "recorded as zero outputs")
    if infeasible == "drop":
        X, Y = X[ok], Y[ok]
    else:
        Y[~ok] = 0.0
    return DataFrame(X, columns=names), DataFrame(Y, columns=list(outmap))


# --- the network --------------------------------------------------------------

def _mapminmax(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """MATLAB's mapminmax settings onto [-1, 1]: (offset, gain), with gain 1 on a constant column."""
    lo, span = A.min(axis=0), np.ptp(A, axis=0)
    return lo, np.where(span > 0, 2.0 / np.where(span > 0, span, 1.0), 1.0)


@dataclass
class FBASurrogate:
    """A trained FBA surrogate as the closed-form network of Song et al.

        x' = (x - x_offset) * x_gain - 1                    mapminmax onto [-1, 1]
        a  = tanh(a_prev @ W_l + b_l)   for each hidden l    tansig
        y' = a @ W_out + b_out                               purelin
        y  = (y' + 1) / y_gain + y_offset                    mapminmax, reversed

    `predict` (or calling the object) evaluates it with numpy alone, which is what lets it sit
    inside an ODE right-hand side; `to_dict` exposes the constants for porting it elsewhere.
    It takes uptake limits in the units and sign of `sample_fba` (positive magnitudes) and is
    only trained inside `input_bounds`; outside, it extrapolates.
    """

    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    x_offset: np.ndarray
    x_gain: np.ndarray
    weights: List[np.ndarray]
    biases: List[np.ndarray]
    y_offset: np.ndarray
    y_gain: np.ndarray
    input_bounds: Dict[str, Box] = field(default_factory=dict)
    scores: Optional[DataFrame] = None      # R^2 per output on the train / validation / test rows
    grid: Optional[DataFrame] = None        # validation error of every hyperparameter tried
    samples: Optional[Tuple[DataFrame, DataFrame]] = field(default=None, repr=False)
    estimator: Optional[object] = field(default=None, repr=False)

    @property
    def hidden_layers(self) -> Tuple[int, ...]:
        return tuple(W.shape[1] for W in self.weights[:-1])

    def __repr__(self) -> str:
        fit = "" if self.scores is None else f", worst test R^2 {self.scores['test'].min():.6g}"
        return (f"FBASurrogate({len(self.inputs)} inputs -> {len(self.outputs)} outputs, "
                f"tanh hidden layers {self.hidden_layers}{fit})")

    def _forward(self, X: np.ndarray) -> np.ndarray:
        a = (X - self.x_offset) * self.x_gain - 1.0
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            a = np.tanh(a @ W + b)
        return (a @ self.weights[-1] + self.biases[-1] + 1.0) / self.y_gain + self.y_offset

    def predict(self, X):
        """Outputs at uptake limits `X`: an array (one row per point, or a single point), a
        DataFrame or a {input: value(s)} dict -- answered in the same kind."""
        if isinstance(X, DataFrame):
            return DataFrame(self._forward(X[list(self.inputs)].to_numpy(float)),
                             columns=list(self.outputs), index=X.index)
        if isinstance(X, Mapping):
            cols = np.broadcast_arrays(*(np.asarray(X[name], float) for name in self.inputs))
            Y = self._forward(np.stack([c.ravel() for c in cols], axis=1))
            shape = cols[0].shape
            return {name: (float(y[0]) if not shape else y.reshape(shape)) for name, y in zip(self.outputs, Y.T)}
        X = np.asarray(X, float)
        return self._forward(X[None, :])[0] if X.ndim == 1 else self._forward(X)

    __call__ = predict

    def to_dict(self) -> dict:
        return {"inputs": list(self.inputs), "outputs": list(self.outputs),
                "x_offset": self.x_offset.tolist(), "x_gain": self.x_gain.tolist(),
                "weights": [W.tolist() for W in self.weights], "biases": [b.tolist() for b in self.biases],
                "y_offset": self.y_offset.tolist(), "y_gain": self.y_gain.tolist(),
                "input_bounds": {k: list(v) for k, v in self.input_bounds.items()},
                "scores": None if self.scores is None else self.scores.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "FBASurrogate":
        arr = lambda v: np.asarray(v, float)
        return cls(inputs=tuple(d["inputs"]), outputs=tuple(d["outputs"]), x_offset=arr(d["x_offset"]),
                   x_gain=arr(d["x_gain"]), weights=[arr(W) for W in d["weights"]],
                   biases=[arr(b) for b in d["biases"]], y_offset=arr(d["y_offset"]), y_gain=arr(d["y_gain"]),
                   input_bounds={k: tuple(v) for k, v in d.get("input_bounds", {}).items()},
                   scores=None if d.get("scores") is None else DataFrame(d["scores"]))

    def save(self, path: str) -> None:
        with open(path, "w") as out:
            json.dump(self.to_dict(), out)

    @classmethod
    def load(cls, path: str) -> "FBASurrogate":
        with open(path) as inp:
            return cls.from_dict(json.load(inp))


def _r2(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    ss_res = ((Y - P) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)


def train_surrogate(X: DataFrame, Y: DataFrame, hidden_layers: Union[int, Sequence[int]] = 5,
                    nodes: Union[int, Sequence[int]] = 10, alpha: Union[float, Sequence[float]] = 1e-6,
                    split: Tuple[float, float, float] = (0.70, 0.15, 0.15), repeats: int = 3,
                    tolerance: float = 0.05, seed: int = 0, max_iter: int = 5000,
                    input_bounds: Optional[Mapping[str, Box]] = None) -> FBASurrogate:
    """Fit the paper's MIMO network to sampled FBA solutions (`sample_fba`'s X and Y).

    hidden_layers, nodes   the depth and width of the tanh hidden layers (default 5 x 10, the
                           paper's MIMO size). Give lists to grid-search them, as the paper did:
                           every combination is trained on `repeats` random train/validation
                           splits, and the SMALLEST whose mean validation error is within
                           `tolerance` (relative) of the best is kept.
    alpha                  the L2 penalty of the L-BFGS fit; a list joins the grid search.
    split                  train / validation / test fractions. The test rows are set aside
                           once, before any search, and only scored.

    The result's `scores` holds R^2 per output on each part of the split, and `grid` the search.
    """
    if not np.isclose(sum(split), 1.0):
        raise ValueError(f"split fractions must sum to 1, not {split}")
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPRegressor

    Xa, Ya = X.to_numpy(float), Y.to_numpy(float)
    n = len(Xa)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = int(round(split[2] * n))
    test, rest = order[:n_test], order[n_test:]
    n_val = int(round(split[1] * n))
    if n_val < 1 or len(rest) - n_val < 1:
        raise ValueError(f"{n} samples are too few for a {split} split")

    listed = lambda v: list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v]
    grid = sorted(product(listed(hidden_layers), listed(nodes), listed(alpha)),
                  # smallest network first; for one size, the stronger penalty first
                  key=lambda g: (Xa.shape[1] * g[1] + (g[0] - 1) * g[1] ** 2 + g[1] * Ya.shape[1], -g[2]))
    searching = len(grid) > 1
    splits = [rest] + [rng.permutation(rest) for _ in range(repeats - 1 if searching else 0)]

    def fit(layers, width, penalty, rows):
        tr, va = rows[n_val:], rows[:n_val]
        # mapminmax settings come from the training rows alone, as in MATLAB
        xo, xg = _mapminmax(Xa[tr])
        yo, yg = _mapminmax(Ya[tr])
        net = MLPRegressor(hidden_layer_sizes=(width,) * layers, activation="tanh", solver="lbfgs",
                           alpha=penalty, max_iter=max_iter, tol=1e-10, random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            net.fit((Xa[tr] - xo) * xg - 1.0, (Ya[tr] - yo) * yg - 1.0)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            logger.warning("train_surrogate: the %dx%d network (alpha=%g) stopped at max_iter=%d "
                           "before converging.", layers, width, penalty, max_iter)
        surrogate = FBASurrogate(inputs=tuple(X.columns), outputs=tuple(Y.columns), x_offset=xo, x_gain=xg,
                                 weights=[np.array(W) for W in net.coefs_],
                                 biases=[np.array(b) for b in net.intercepts_], y_offset=yo, y_gain=yg,
                                 estimator=net)
        err = (((surrogate.predict(Xa[va]) - Ya[va]) * yg) ** 2).mean()  # in the scaled outputs
        return surrogate, err, tr, va

    records, fits = [], {}
    for g in grid:
        errs = []
        for r, rows in enumerate(splits):
            surrogate, err, tr, va = fit(*g, rows)
            errs.append(err)
            if r == 0:
                fits[g] = (surrogate, tr, va)
        records.append({"hidden_layers": g[0], "nodes": g[1], "alpha": g[2],
                        "validation_mse": float(np.mean(errs)), "validation_mse_sd": float(np.std(errs))})
    table = DataFrame(records)
    best = table["validation_mse"].min()
    pick = next(i for i, rec in enumerate(records) if rec["validation_mse"] <= best * (1 + tolerance) + 1e-15)
    table["selected"] = [i == pick for i in range(len(records))]
    surrogate, tr, va = fits[grid[pick]]
    if searching:
        logger.info("train_surrogate: kept %d x %d nodes (alpha=%g) of %d configurations.",
                    grid[pick][0], grid[pick][1], grid[pick][2], len(grid))

    surrogate.scores = DataFrame({part: _r2(Ya[rows], surrogate.predict(Xa[rows])) if len(rows) else np.nan
                                  for part, rows in (("train", tr), ("validation", va), ("test", test))},
                                 index=list(Y.columns))
    surrogate.grid = table
    surrogate.samples = (X, Y)
    surrogate.input_bounds = dict(input_bounds) if input_bounds is not None else \
        {c: (float(X[c].min()), float(X[c].max())) for c in X.columns}
    return surrogate


# --- one call -----------------------------------------------------------------

def community_to_nn(community, inputs=None, extra_outputs=None, member_exchanges=False, steps=None,
                    n_samples: int = 5000, pfba: bool = True, seed: int = 0, infeasible: str = "drop",
                    backend: str = "cpu", workers: int = 1, **train_args) -> FBASurrogate:
    """Convert a community (or a single-organism) FBA model into Song et al.'s ANN surrogate.

    community   an MSCommunity, sampled as configured (abundances, CommKinetics, closed member
                drains all shape what "member growth" means), or a plain cobra Model, whose
                objective reactions then stand in for the members' biomass.
    inputs      the uptake limits the network takes; see `sample_fba`. By default every
                exchange of the current medium, each on [0, its current uptake] -- for a rich
                medium that is dozens of dimensions that 5000 samples cannot cover, so name the
                limiting nutrients (the paper took the carbon source and O2) where you can.
    extra_outputs, member_exchanges   see `community_outputs`. The outputs always include
                every member's biomass, the community biomass and the actual input fluxes.
    steps, n_samples, pfba, seed, infeasible, backend, workers   see `sample_fba`.
    train_args  hidden_layers, nodes, alpha, split, repeats, tolerance, max_iter; see
                `train_surrogate`.

    Returns the trained `FBASurrogate`; its `scores` say how far to trust it and its `samples`
    let `train_surrogate` refit it without solving the LPs again.
    """
    if isinstance(community, Model):
        model = community
        box = _input_box(model, inputs if inputs is not None else model.medium)
        objective = model.objective.variables
        outputs = {rxn.id: {rxn.id: 1.0} for rxn in model.reactions if rxn.forward_variable in objective}
        outputs.update({rid: {rid: 1.0} for rid in box})
        if extra_outputs is not None:
            outputs.update(_output_map(extra_outputs))
        if member_exchanges:
            raise ValueError("member_exchanges needs an MSCommunity, whose members say which reactions are whose")
    else:
        model = community.util.model
        box = _input_box(model, inputs if inputs is not None else model.medium)
        outputs = community_outputs(community, list(box), extra_outputs, member_exchanges)
    logger.info("community_to_nn(%s): %d inputs %s -> %d outputs", model.id, len(box), list(box), len(outputs))
    X, Y = sample_fba(model, box, outputs, n_samples=n_samples, steps=steps, pfba=pfba, seed=seed,
                      infeasible=infeasible, backend=backend, workers=workers)
    return train_surrogate(X, Y, seed=seed, input_bounds=box, **train_args)
