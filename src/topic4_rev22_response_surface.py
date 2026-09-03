"""rev22-DCI Task 5: per-component response surfaces, feasibility and minimax proposals.

Pure model-fitting helpers on top of scikit-learn. No filesystem I/O, no SNN, no
validation metrics. One Gaussian-process surface is fitted per training component
(spec section 7); the proposal scalar ``J = max_k E_k`` is never fitted, only
evaluated on the component surfaces (spec section 6). All models work on the unit
cube of the frozen 4-D domain so that ARD length scales are comparable.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

PARAMETER_ORDER = ("g_LEE", "g_LEI", "theta_FT_deg", "AR_FT")
REFERENCE = (0.5, 1.0, 45.0, 2.0)
COMPONENTS = ("D_support", "D_order", "D_lag", "D_cover")
SHRINKAGE_PRIOR_WEIGHT = 8.0
LENGTH_SCALE_BOUNDS = (0.05, 5.0)


# --------------------------------------------------------------------------- #
# noise and scaling
# --------------------------------------------------------------------------- #
def pooled_shrinkage_noise(z_sd_by_candidate: Mapping[str, float | None],
                           n_units: int | Mapping[str, int]) -> dict:
    """Spec section 7 pooled-shrinkage observation variance of each candidate mean.

    ``s_c^2 = n_c * jackknife_sd^2`` is the unit-level variance equivalent, ``s_pool^2``
    the median over finite candidates, ``s_tilde_c^2 = (8 s_pool^2 + (n_c-1) s_c^2) /
    (8 + n_c - 1)`` and ``Var(mean) = s_tilde_c^2 / n_c``.
    """
    def units_of(name):
        return int(n_units[name]) if isinstance(n_units, Mapping) else int(n_units)

    unit_var = {}
    for name, sd in z_sd_by_candidate.items():
        if sd is None or not np.isfinite(sd):
            continue
        unit_var[name] = units_of(name) * float(sd) ** 2
    if not unit_var:
        return {"s_pool_sq": None, "variance": {}, "unit_variance": {}, "n_finite": 0}
    s_pool_sq = float(np.median(list(unit_var.values())))
    variance = {}
    for name, s_c_sq in unit_var.items():
        n_c = units_of(name)
        s_tilde_sq = (SHRINKAGE_PRIOR_WEIGHT * s_pool_sq + (n_c - 1) * s_c_sq) / (
            SHRINKAGE_PRIOR_WEIGHT + n_c - 1)
        variance[name] = float(s_tilde_sq / n_c)
    return {"s_pool_sq": s_pool_sq, "variance": variance, "unit_variance": unit_var,
            "n_finite": len(unit_var)}


def _bounds(domain: Mapping) -> np.ndarray:
    bounds = np.asarray([[float(domain[k][0]), float(domain[k][1])] for k in PARAMETER_ORDER], float)
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("every domain interval must have hi > lo")
    return bounds


def unit_cube(x, domain: Mapping) -> np.ndarray:
    bounds = _bounds(domain)
    values = np.asarray(x, float)
    return (values - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def from_unit_cube(u, domain: Mapping) -> np.ndarray:
    bounds = _bounds(domain)
    values = np.asarray(u, float)
    return bounds[:, 0] + values * (bounds[:, 1] - bounds[:, 0])


def _as_matrix(X) -> np.ndarray:
    values = np.asarray(X, float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != len(PARAMETER_ORDER):
        raise ValueError("X must have shape (n, 4) in the order " + ", ".join(PARAMETER_ORDER))
    return values


def _matern_kernel():
    return ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.full(len(PARAMETER_ORDER), 0.5), length_scale_bounds=LENGTH_SCALE_BOUNDS, nu=2.5)


# --------------------------------------------------------------------------- #
# surfaces
# --------------------------------------------------------------------------- #
@dataclass
class ComponentSurface:
    """Matern-5/2 GP on the unit cube with per-point observation variance.

    Targets are centred and scaled inside the wrapper so that the per-point ``alpha``
    (observation variance) is expressed in the same normalized units as the kernel.
    """
    gp: GaussianProcessRegressor
    domain: Mapping
    y_mean: float
    y_scale: float
    kernel_str: str
    n_train: int

    def predict(self, X_physical) -> tuple[np.ndarray, np.ndarray]:
        u = unit_cube(_as_matrix(X_physical), self.domain)
        mean, sd = self.gp.predict(u, return_std=True)
        return mean * self.y_scale + self.y_mean, sd * self.y_scale


def fit_component_gp(X, z, noise_var, domain: Mapping, seed: int, *, n_restarts: int = 8,
                     kernel=None, optimizer="fmin_l_bfgs_b") -> ComponentSurface:
    X = _as_matrix(X)
    z = np.asarray(z, float)
    noise_var = np.asarray(noise_var, float)
    if z.shape != (len(X),) or noise_var.shape != (len(X),):
        raise ValueError("z and noise_var must align with X rows")
    keep = np.isfinite(z) & np.isfinite(noise_var)
    if keep.sum() < 3:
        raise ValueError("a component surface needs at least three finite observations")
    u = unit_cube(X[keep], domain)
    y = z[keep]
    y_mean = float(y.mean())
    y_scale = float(y.std()) if y.std() > 1e-12 else 1.0
    alpha = np.maximum(noise_var[keep] / y_scale ** 2, 1e-10)
    gp = GaussianProcessRegressor(
        kernel=_matern_kernel() if kernel is None else kernel, alpha=alpha,
        normalize_y=False, n_restarts_optimizer=int(n_restarts), random_state=int(seed),
        optimizer=optimizer,
    )
    gp.fit(u, (y - y_mean) / y_scale)
    return ComponentSurface(gp=gp, domain=domain, y_mean=y_mean, y_scale=y_scale,
                            kernel_str=str(gp.kernel_), n_train=int(keep.sum()))


@dataclass
class FeasibilityModel:
    domain: Mapping
    constant: float | None = None
    classifier: GaussianProcessClassifier | None = None

    @property
    def is_constant(self) -> bool:
        return self.constant is not None

    def predict_proba(self, X_physical) -> np.ndarray:
        X = _as_matrix(X_physical)
        if self.constant is not None:
            return np.full(len(X), float(self.constant))
        u = unit_cube(X, self.domain)
        return self.classifier.predict_proba(u)[:, 1]


def fit_feasibility(X, feasible_bool, domain: Mapping, seed: int, *, n_restarts: int = 2) -> FeasibilityModel:
    X = _as_matrix(X)
    feasible = np.asarray(feasible_bool, bool)
    if feasible.shape != (len(X),):
        raise ValueError("feasibility flags must align with X rows")
    if feasible.all():
        return FeasibilityModel(domain=domain, constant=1.0)
    if not feasible.any():
        return FeasibilityModel(domain=domain, constant=0.0)
    classifier = GaussianProcessClassifier(kernel=_matern_kernel(), n_restarts_optimizer=int(n_restarts),
                                           random_state=int(seed))
    classifier.fit(unit_cube(X, domain), feasible.astype(int))
    return FeasibilityModel(domain=domain, classifier=classifier)


@dataclass
class TreeSurface:
    model: ExtraTreesRegressor
    domain: Mapping
    n_train: int

    def predict(self, X_physical) -> tuple[np.ndarray, np.ndarray]:
        u = unit_cube(_as_matrix(X_physical), self.domain)
        per_tree = np.stack([tree.predict(u) for tree in self.model.estimators_], axis=0)
        return per_tree.mean(axis=0), per_tree.std(axis=0)


def fit_tree_ensemble(X, z, domain: Mapping, seed: int, *, n_estimators: int = 600) -> TreeSurface:
    X = _as_matrix(X)
    z = np.asarray(z, float)
    keep = np.isfinite(z)
    if keep.sum() < 3:
        raise ValueError("a tree surface needs at least three finite observations")
    model = ExtraTreesRegressor(n_estimators=int(n_estimators), min_samples_leaf=2, random_state=int(seed))
    model.fit(unit_cube(X[keep], domain), z[keep])
    return TreeSurface(model=model, domain=domain, n_train=int(keep.sum()))


def loo_diagnostics(X, z, noise_var, domain: Mapping, seed: int, *, n_restarts: int = 8,
                    refit_hyperparameters: bool = True) -> dict:
    """Leave-one-candidate-out GP prediction error, rank correlation and 90% coverage.

    Formal diagnostics refit kernel hyperparameters inside every fold. The faster
    fixed-kernel mode is retained only for explicitly labelled engineering canaries.
    """
    X = _as_matrix(X)
    z = np.asarray(z, float)
    noise_var = np.asarray(noise_var, float)
    keep = np.isfinite(z) & np.isfinite(noise_var)
    X, z, noise_var = X[keep], z[keep], noise_var[keep]
    n = len(z)
    if n < 4:
        return {"n": int(n), "rmse": None, "spearman": None, "coverage_90": None,
                "status": "TOO_FEW_POINTS"}
    full = fit_component_gp(X, z, noise_var, domain, seed, n_restarts=n_restarts)
    predictions, sds = np.empty(n), np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        if refit_hyperparameters:
            fold = fit_component_gp(X[mask], z[mask], noise_var[mask], domain, seed, n_restarts=n_restarts)
        else:
            fold = fit_component_gp(X[mask], z[mask], noise_var[mask], domain, seed, n_restarts=0,
                                    kernel=full.gp.kernel_, optimizer=None)
        mean, sd = fold.predict(X[i:i + 1])
        predictions[i], sds[i] = mean[0], sd[0]
    residual = predictions - z
    rho = spearmanr(predictions, z).statistic if n >= 3 else float("nan")
    covered = np.abs(residual) <= 1.645 * np.sqrt(sds ** 2 + noise_var)
    return {
        "n": int(n),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "spearman": None if not np.isfinite(rho) else float(rho),
        "coverage_90": float(np.mean(covered)),
        "observed_range": float(z.max() - z.min()),
        "kernel": full.kernel_str,
        "refit_hyperparameters": bool(refit_hyperparameters),
        "status": "OK",
    }


# --------------------------------------------------------------------------- #
# proposals
# --------------------------------------------------------------------------- #
def _parse_mask(mask: str) -> np.ndarray:
    if len(mask) != len(PARAMETER_ORDER) or any(c not in "01" for c in mask):
        raise ValueError("mask must be four characters of 0/1 in the order " + ", ".join(PARAMETER_ORDER))
    return np.asarray([c == "1" for c in mask], bool)


def _sobol_unit(n_points: int, n_dim: int, seed: int) -> np.ndarray:
    sampler = qmc.Sobol(d=int(n_dim), scramble=True, seed=int(seed))
    m = int(np.ceil(np.log2(max(int(n_points), 2))))
    return sampler.random_base2(m)[: int(n_points)]


def _predict_excess(models: Mapping, identifiable: Sequence[str], X_physical) -> tuple[np.ndarray, np.ndarray]:
    """Return (E, sd) with shape (n, len(identifiable)); E = max(0, mean)."""
    means, sds = [], []
    for k in identifiable:
        mean, sd = models[k].predict(X_physical)
        means.append(mean)
        sds.append(sd)
    return np.maximum(0.0, np.column_stack(means)), np.column_stack(sds)


def _pareto_mask(values: np.ndarray) -> np.ndarray:
    """Non-dominated rows for minimization (a row dominates if <= everywhere and < somewhere)."""
    n = len(values)
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        others = values[keep]
        dominated = np.all(others <= values[i], axis=1) & np.any(others < values[i], axis=1)
        if dominated.any():
            keep[i] = False
    return keep


def _proposal(models: Mapping, identifiable: Sequence[str], mask: str, reference, domain: Mapping,
              feasibility: FeasibilityModel | None, *, min_feasibility: float, n_candidates: int,
              seed: int, pareto_cap: int, refine: bool) -> dict:
    identifiable = list(identifiable)
    if not identifiable:
        raise ValueError("the identifiable set is empty; no proposal can be formed")
    free = _parse_mask(mask)
    reference = np.asarray(reference, float)
    u_ref = unit_cube(reference, domain)

    def feasibility_of(X_physical) -> np.ndarray:
        if feasibility is None:
            return np.ones(len(_as_matrix(X_physical)))
        return feasibility.predict_proba(X_physical)

    def describe(u_point: np.ndarray, status: str = "OK", extra: dict | None = None) -> dict:
        x = from_unit_cube(u_point, domain)
        E, sd = _predict_excess(models, identifiable, x)
        j = int(np.argmax(E[0]))
        out = {
            "status": status, "mask": mask, "x": x.tolist(),
            "unit": u_point.tolist(),
            "predicted_excess": {k: float(E[0, i]) for i, k in enumerate(identifiable)},
            "predicted_sd": {k: float(sd[0, i]) for i, k in enumerate(identifiable)},
            "J": float(E[0].max()), "argmax_component": identifiable[j],
            "feasibility": float(feasibility_of(x)[0]),
            "identifiable": identifiable,
        }
        if extra:
            out.update(extra)
        return out

    if not free.any():
        return describe(u_ref, extra={"pareto_set": [], "n_feasible_candidates": 1,
                                       "n_grid_candidates": 1})

    grid = np.tile(u_ref, (int(n_candidates), 1))
    grid[:, free] = _sobol_unit(int(n_candidates), int(free.sum()), seed)
    X_grid = from_unit_cube(grid, domain)
    p_grid = feasibility_of(X_grid)
    feasible = p_grid >= float(min_feasibility)
    if not feasible.any():
        return {"status": "NO_FEASIBLE_REGION", "mask": mask, "x": None, "J": None,
                "n_feasible_candidates": 0, "n_grid_candidates": int(n_candidates),
                "identifiable": identifiable, "pareto_set": []}
    E_grid, _ = _predict_excess(models, identifiable, X_grid[feasible])
    J_grid = E_grid.max(axis=1)
    feasible_idx = np.flatnonzero(feasible)
    order = np.argsort(J_grid, kind="stable")
    best_u = grid[feasible_idx[order[0]]].copy()
    best_J = float(J_grid[order[0]])

    if refine:
        free_idx = np.flatnonzero(free)

        def objective(v):
            u = u_ref.copy()
            u[free_idx] = np.clip(v, 0.0, 1.0)
            E, _ = _predict_excess(models, identifiable, from_unit_cube(u, domain))
            return float(E[0].max())

        for start in order[:5]:
            u0 = grid[feasible_idx[start]]
            result = minimize(objective, u0[free_idx], method="L-BFGS-B",
                              bounds=[(0.0, 1.0)] * len(free_idx))
            u_new = u_ref.copy()
            u_new[free_idx] = np.clip(result.x, 0.0, 1.0)
            if not np.isfinite(result.fun):
                continue
            if feasibility_of(from_unit_cube(u_new, domain))[0] < float(min_feasibility):
                continue
            if float(result.fun) < best_J - 1e-12:
                best_u, best_J = u_new, float(result.fun)

    pareto_keep = _pareto_mask(E_grid)
    pareto_rows = np.flatnonzero(pareto_keep)
    pareto_rows = pareto_rows[np.argsort(J_grid[pareto_rows], kind="stable")][: int(pareto_cap)]
    pareto_set = [
        {"x": X_grid[feasible_idx[r]].tolist(),
         "predicted_excess": {k: float(E_grid[r, i]) for i, k in enumerate(identifiable)},
         "J": float(J_grid[r])}
        for r in pareto_rows
    ]
    return describe(best_u, extra={
        "pareto_set": pareto_set,
        "n_feasible_candidates": int(feasible.sum()),
        "n_grid_candidates": int(n_candidates),
        "grid_best_J": float(J_grid[order[0]]),
    })


def conditional_minimax_proposal(gps: Mapping, identifiable: Sequence[str], mask: str, reference,
                                 domain: Mapping, feasibility: FeasibilityModel | None, *,
                                 min_feasibility: float = 0.80, n_candidates: int = 4096,
                                 seed: int = 0, pareto_cap: int = 64) -> dict:
    """Minimax point of predicted normalized excess over the identifiable components.

    Locked coordinates (mask ``0``) stay exactly at the reference; the search covers
    the free coordinates on a scrambled Sobol grid restricted to predicted feasibility
    at least ``min_feasibility``, then is refined by L-BFGS-B from the five best grid
    points. The predicted Pareto set of the feasible grid is returned alongside.
    """
    return _proposal(gps, identifiable, mask, reference, domain, feasibility,
                     min_feasibility=min_feasibility, n_candidates=n_candidates, seed=seed,
                     pareto_cap=pareto_cap, refine=True)


def tree_proposal(trees: Mapping, identifiable: Sequence[str], mask: str, reference, domain: Mapping,
                  feasibility: FeasibilityModel | None, *, min_feasibility: float = 0.80,
                  n_candidates: int = 4096, seed: int = 0, pareto_cap: int = 64) -> dict:
    """Same rule on the tree-ensemble surfaces (grid search only; trees are piecewise constant)."""
    return _proposal(trees, identifiable, mask, reference, domain, feasibility,
                     min_feasibility=min_feasibility, n_candidates=n_candidates, seed=seed,
                     pareto_cap=pareto_cap, refine=False)


def proposals_disagree(gp_x, tree_x, domain: Mapping, *, threshold: float = 0.25) -> dict:
    if gp_x is None or tree_x is None:
        return {"disagree": None, "unit_distance": None, "threshold": float(threshold)}
    distance = float(np.linalg.norm(unit_cube(gp_x, domain) - unit_cube(tree_x, domain)))
    return {"disagree": bool(distance > float(threshold)), "unit_distance": distance,
            "threshold": float(threshold)}


# --------------------------------------------------------------------------- #
# convenience
# --------------------------------------------------------------------------- #
def fit_all(design_rows: Sequence[Mapping], identifiable: Sequence[str], domain: Mapping, seed: int, *,
            n_restarts: int = 8, n_estimators: int = 600, components: Sequence[str] = COMPONENTS) -> dict:
    """Fit component GPs, feasibility, tree ensembles and LOO diagnostics from design rows.

    Each row: ``{"x": [4], "Z": {k: value|None}, "jackknife_sd": {k: value|None},
    "n_units": int, "feasible": bool}`` and optionally ``"candidate_id"``.
    """
    rows = list(design_rows)
    X = _as_matrix([r["x"] for r in rows])
    names = [str(r.get("candidate_id", i)) for i, r in enumerate(rows)]
    feasible = np.asarray([bool(r["feasible"]) for r in rows], bool)
    output = {"identifiable": list(identifiable), "gps": {}, "trees": {}, "loo": {}, "noise": {},
              "feasibility": fit_feasibility(X, feasible, domain, seed),
              "n_rows": len(rows), "n_feasible": int(feasible.sum())}
    for k in components:
        z = np.asarray([np.nan if r["Z"].get(k) is None else float(r["Z"][k]) for r in rows], float)
        sd_map = {n: (None if r["jackknife_sd"].get(k) is None else float(r["jackknife_sd"][k]))
                  for n, r in zip(names, rows)}
        n_units = {n: int(r["n_units"]) for n, r in zip(names, rows)}
        noise = pooled_shrinkage_noise(sd_map, n_units)
        noise_var = np.asarray([noise["variance"].get(n, np.nan) for n in names], float)
        # only feasible, finite rows enter a continuous surface
        usable = feasible & np.isfinite(z) & np.isfinite(noise_var)
        output["noise"][k] = {"s_pool_sq": noise["s_pool_sq"], "n_finite": noise["n_finite"],
                              "n_usable": int(usable.sum())}
        if usable.sum() < 3:
            output["gps"][k] = None
            output["trees"][k] = None
            output["loo"][k] = {"status": "TOO_FEW_POINTS", "n": int(usable.sum())}
            continue
        output["gps"][k] = fit_component_gp(X[usable], z[usable], noise_var[usable], domain, seed,
                                            n_restarts=n_restarts)
        output["trees"][k] = fit_tree_ensemble(X[usable], z[usable], domain, seed, n_estimators=n_estimators)
        output["loo"][k] = loo_diagnostics(X[usable], z[usable], noise_var[usable], domain, seed,
                                           n_restarts=n_restarts, refit_hyperparameters=True)
    missing = [k for k in identifiable if output["gps"].get(k) is None]
    if missing:
        raise ValueError(f"identifiable components without a fitted surface: {missing}")
    return output
