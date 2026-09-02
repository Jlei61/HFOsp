"""Negative-binomial (NB2) ridge GLM used identically by every evaluation arm.

The count target is heavily overdispersed (variance/mean 7-385 in v0.3.1), so a
Poisson score would reward whichever arm happens to be least miscalibrated.
Every arm therefore shares: this family, this fitting procedure, the same
standardisation rows, the same ridge grid and the same selection rows.  Nothing
is re-estimated on scoring rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

ETA_CLIP = 30.0


def nb_nll(y: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    """Per-observation NB2 negative log-likelihood, Var = mu + alpha mu^2."""

    y = np.asarray(y, dtype=np.float64)
    mu = np.clip(np.asarray(mu, dtype=np.float64), 1e-12, None)
    r = 1.0 / float(alpha)
    ll = (
        gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
        + r * (np.log(r) - np.log(r + mu))
        + y * (np.log(mu) - np.log(r + mu))
    )
    return -ll


def _alpha_ml(y: np.ndarray, mu: np.ndarray, log_bounds: tuple[float, float]) -> float:
    def objective(log_alpha: float) -> float:
        return float(nb_nll(y, mu, float(np.exp(log_alpha))).sum())

    result = minimize_scalar(objective, bounds=log_bounds, method="bounded",
                             options={"xatol": 1e-4})
    return float(np.exp(result.x))


@dataclass
class NegativeBinomialRidge:
    """Ridge-penalised NB2 regression with ML dispersion, frozen after ``fit``."""

    ridge: float = 1.0
    max_iter: int = 50
    alpha_log_bounds: tuple[float, float] = (-9.0, 6.0)
    tol: float = 1e-7
    fixed_alpha: float | None = None
    x_mean: np.ndarray | None = None
    x_scale: np.ndarray | None = None
    coef_: np.ndarray = field(default=None, repr=False)
    intercept_: float = field(default=None, repr=False)
    alpha_: float = field(default=None, repr=False)
    x_mean_: np.ndarray = field(default=None, repr=False)
    x_scale_: np.ndarray = field(default=None, repr=False)
    converged_: bool = field(default=False, repr=False)
    n_iter_: int = field(default=0, repr=False)
    n_fit_rows_: int = field(default=0, repr=False)
    fit_history_: list = field(default_factory=list, repr=False)

    # -- standardisation -------------------------------------------------------
    def _standardise(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.x_mean_) / self.x_scale_
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def _penalised_loglik(self, z: np.ndarray, y: np.ndarray, beta: np.ndarray,
                          alpha: float) -> float:
        eta = np.clip(beta[0] + z @ beta[1:], -ETA_CLIP, ETA_CLIP)
        mu = np.exp(eta)
        return float(-nb_nll(y, mu, alpha).sum() - 0.5 * self.ridge * float(beta[1:] @ beta[1:]))

    def _newton_beta(self, z: np.ndarray, y: np.ndarray, beta: np.ndarray,
                     alpha: float) -> tuple[np.ndarray, bool]:
        n, p = z.shape
        design = np.column_stack([np.ones(n), z])
        penalty = np.full(p + 1, float(self.ridge))
        penalty[0] = 0.0
        current = self._penalised_loglik(z, y, beta, alpha)
        converged = False
        for _ in range(self.max_iter):
            eta = np.clip(design @ beta, -ETA_CLIP, ETA_CLIP)
            mu = np.exp(eta)
            denom = 1.0 + alpha * mu
            score = design.T @ ((y - mu) / denom) - penalty * beta
            weight = mu / denom
            hessian = design.T @ (design * weight[:, None]) + np.diag(penalty)
            try:
                step = np.linalg.solve(hessian, score)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, score, rcond=None)[0]
            scale = 1.0
            improved = False
            for _halving in range(20):
                candidate = beta + scale * step
                value = self._penalised_loglik(z, y, candidate, alpha)
                if value >= current - 1e-12:
                    improved = True
                    break
                scale *= 0.5
            if not improved:
                break
            change = float(np.max(np.abs(candidate - beta)))
            beta = candidate
            if value - current < self.tol * max(1.0, abs(current)) and change < 1e-6:
                current = value
                converged = True
                break
            current = value
        return beta, converged

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NegativeBinomialRidge":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError("x must be (n, p) and y must be (n,)")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("non-finite design or target")
        if (y < 0).any():
            raise ValueError("counts must be non-negative")
        self.x_mean_ = np.asarray(self.x_mean, dtype=np.float64) if self.x_mean is not None else x.mean(axis=0)
        if self.x_scale is not None:
            self.x_scale_ = np.asarray(self.x_scale, dtype=np.float64)
        else:
            scale = x.std(axis=0)
            self.x_scale_ = np.where(scale > 1e-9, scale, 1.0)
        z = self._standardise(x)
        mean_y = max(float(y.mean()), 1e-8)
        var_y = float(y.var()) if y.size > 1 else mean_y
        if self.fixed_alpha is not None:
            alpha = float(self.fixed_alpha)
        else:
            moment = (var_y - mean_y) / (mean_y ** 2)
            alpha = float(np.clip(moment, np.exp(self.alpha_log_bounds[0]) * 10, np.exp(self.alpha_log_bounds[1]) / 10))
        beta = np.zeros(z.shape[1] + 1)
        beta[0] = np.log(mean_y)
        self.fit_history_ = []
        converged = False
        for outer in range(20):
            beta, beta_ok = self._newton_beta(z, y, beta, alpha)
            eta = np.clip(beta[0] + z @ beta[1:], -ETA_CLIP, ETA_CLIP)
            mu = np.exp(eta)
            if self.fixed_alpha is None:
                new_alpha = _alpha_ml(y, mu, self.alpha_log_bounds)
            else:
                new_alpha = alpha
            objective = self._penalised_loglik(z, y, beta, new_alpha)
            self.fit_history_.append({"outer": outer, "alpha": new_alpha, "objective": objective,
                                      "beta_converged": bool(beta_ok)})
            rel = abs(np.log(new_alpha) - np.log(alpha))
            alpha = new_alpha
            if beta_ok and rel < 1e-4:
                converged = True
                break
        self.coef_ = beta[1:].copy()
        self.intercept_ = float(beta[0])
        self.alpha_ = float(alpha)
        self.converged_ = bool(converged)
        self.n_iter_ = len(self.fit_history_)
        self.n_fit_rows_ = int(y.size)
        return self

    def linear_predictor(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        return np.clip(self.intercept_ + self._standardise(x) @ self.coef_, -ETA_CLIP, ETA_CLIP)

    def predict_mu(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.linear_predictor(x))

    def nll(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Per-row NB NLL with every fitted quantity frozen (no test-time fit)."""

        return nb_nll(np.asarray(y, dtype=np.float64), self.predict_mu(x), self.alpha_)

    def state_dict(self) -> dict[str, Any]:
        return {
            "ridge": float(self.ridge),
            "coef": self.coef_.tolist(),
            "intercept": float(self.intercept_),
            "alpha": float(self.alpha_),
            "x_mean": self.x_mean_.tolist(),
            "x_scale": self.x_scale_.tolist(),
            "converged": bool(self.converged_),
            "n_iter": int(self.n_iter_),
            "n_fit_rows": int(self.n_fit_rows_),
            "fixed_alpha": None if self.fixed_alpha is None else float(self.fixed_alpha),
        }


def select_and_refit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fit_rows: np.ndarray,
    select_rows: np.ndarray,
    refit_rows: np.ndarray,
    ridge_grid: Sequence[float],
    alpha_log_bounds: tuple[float, float] = (-9.0, 6.0),
    fixed_alpha: float | None = None,
    max_iter: int = 50,
) -> dict[str, Any]:
    """Fit on ``fit_rows``, choose ridge on ``select_rows``, refit on ``refit_rows``.

    Standardisation constants are estimated on ``fit_rows`` once and reused for
    the refit, so the refit only re-estimates regression weights and dispersion.
    Grid-edge selections are flagged, never dropped; failed ridges are recorded.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    select_rows = np.asarray(select_rows, dtype=np.int64)
    refit_rows = np.asarray(refit_rows, dtype=np.int64)
    if fit_rows.size == 0 or select_rows.size == 0 or refit_rows.size == 0:
        raise ValueError("every stage needs at least one row")
    x_mean = x[fit_rows].mean(axis=0)
    scale = x[fit_rows].std(axis=0)
    x_scale = np.where(scale > 1e-9, scale, 1.0)
    grid = [float(v) for v in ridge_grid]
    path: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    best: tuple[float, float] | None = None
    for ridge in grid:
        try:
            model = NegativeBinomialRidge(
                ridge=ridge, max_iter=max_iter, alpha_log_bounds=alpha_log_bounds,
                fixed_alpha=fixed_alpha, x_mean=x_mean, x_scale=x_scale,
            ).fit(x[fit_rows], y[fit_rows])
            score = float(model.nll(x[select_rows], y[select_rows]).mean())
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
            failures.append({"ridge": ridge, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not np.isfinite(score):
            failures.append({"ridge": ridge, "error": "non-finite selection score"})
            continue
        path.append({"ridge": ridge, "selection_nll": score})
        if best is None or score < best[0]:
            best = (score, ridge)
    if best is None:
        raise RuntimeError(f"every ridge failed: {failures}")
    selected = best[1]
    model = NegativeBinomialRidge(
        ridge=selected, max_iter=max_iter, alpha_log_bounds=alpha_log_bounds,
        fixed_alpha=fixed_alpha, x_mean=x_mean, x_scale=x_scale,
    ).fit(x[refit_rows], y[refit_rows])
    return {
        "model": model,
        "selected_ridge": selected,
        "ridge_at_edge": bool(selected in (grid[0], grid[-1])),
        "selection_nll": float(best[0]),
        "path": path,
        "solver_failures": failures,
        "n_fit_rows": int(fit_rows.size),
        "n_select_rows": int(select_rows.size),
        "n_refit_rows": int(refit_rows.size),
        "n_features": int(x.shape[1]),
    }
