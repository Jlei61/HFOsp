"""Fit the shared explicit-history baseline ``H`` and publish ``log mu_H`` per anchor.

The model agent fixes ``H`` as its base model and learns only a residual on top
of ``log mu_H``.  This module therefore publishes, for every anchor on the grid
and every horizon, the NB ridge prediction of ``H_strong`` (primary) and
``H_rate`` (nested control), fitted under the frozen v0.3.2 recipe:
fit on base_fit -> select ridge on inner_val -> refit on base_refit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contract import atomic_npz, finite_or_none, now_iso
from .h1_eval import H1Design
from .nb_glm import NegativeBinomialRidge, select_and_refit
from .partition import EVAL_PHASES, REFIT_PHASE
from .timeline import EvalTimeline

REGISTRY_FORMAT = "group_event_state_v0_3_2_history_baseline_registry"


def fit_history_for_patient(tl: EvalTimeline, cfg: Mapping[str, Any], design: H1Design,
                            out_dir: Path) -> dict[str, Any]:
    """Return the registry entry for one patient and write its arrays."""

    out_dir = Path(out_dir)
    nb = cfg["nb_glm"]
    horizons = tl.horizons_seconds
    primary_variant = cfg["primary_history"]
    entry: dict[str, Any] = {
        "subject": tl.subject, "dataset": tl.dataset, "generated": now_iso(),
        "primary_variant": primary_variant, "horizons": {}, "variants": {},
        "partition": tl.partition.as_dict(),
        "n_anchors": int(tl.grid.n_anchors),
        "fit_recipe": "NB2 ridge GLM: fit base_fit (0-60%), ridge selected on inner_val (60-70%), "
                      "refit on base_refit (0-70%) with base_fit standardisation; dev_val/dev_test never fitted",
    }
    arrays: dict[str, np.ndarray] = {"anchor_time": tl.grid.t_anchor,
                                     "anchor_segment": tl.grid.segment_index,
                                     "anchor_phase": tl.anchor_phase_labels()}
    y_all = np.zeros(tl.grid.n_anchors, dtype=np.int64)
    selected_by_horizon: dict[str, dict[str, float]] = {}
    for h_i, horizon in enumerate(horizons):
        key = str(int(horizon))
        eligible = np.flatnonzero(tl.grid.eligible[:, h_i])
        y = y_all.copy()
        y[eligible] = tl.window_counts(eligible, h_i)
        arrays[f"count_{key}"] = y
        arrays[f"eligible_{key}"] = tl.grid.eligible[:, h_i]
        rows = {p: tl.anchor_indices(p, h_i) for p in EVAL_PHASES + (REFIT_PHASE,)}
        for variant, (x, names) in design.history.items():
            fit_rows, select_rows, refit_rows = rows["base_fit"], rows["inner_val"], rows[REFIT_PHASE]
            spec: dict[str, Any] = {
                "variant": variant, "horizon_seconds": float(horizon), "n_features": int(x.shape[1]),
                "n_fit_rows": int(fit_rows.size), "n_select_rows": int(select_rows.size),
                "n_refit_rows": int(refit_rows.size),
            }
            if fit_rows.size < 3 or refit_rows.size < 3:
                spec["status"] = "not_estimable_insufficient_fit_rows"
                entry["variants"].setdefault(variant, {})[key] = spec
                continue
            try:
                if select_rows.size >= 1:
                    fit = select_and_refit(
                        x, y, fit_rows=fit_rows, select_rows=select_rows, refit_rows=refit_rows,
                        ridge_grid=nb["ridge_grid"], alpha_log_bounds=tuple(nb["alpha_log_bounds"]),
                        max_iter=int(nb["max_irls_iter"]),
                    )
                    spec["selection"] = "inner_val"
                else:
                    # No complete inner_val window at this horizon: the horizon is a-priori
                    # ineligible (see endpoint_eligibility.json).  Publish anyway so the
                    # model agent's reader keeps every horizon, with the ridge inherited from
                    # the 1800 s selection of the same variant.  Never used for a claim.
                    inherited = selected_by_horizon.get("1800", {}).get(variant)
                    grid = [float(v) for v in nb["ridge_grid"]]
                    ridge = inherited if inherited is not None else grid[len(grid) // 2]
                    fit = select_and_refit(
                        x, y, fit_rows=fit_rows, select_rows=fit_rows[:1], refit_rows=refit_rows,
                        ridge_grid=(ridge,), alpha_log_bounds=tuple(nb["alpha_log_bounds"]),
                        max_iter=int(nb["max_irls_iter"]),
                    )
                    spec["selection"] = "inherited_from_1800s_no_inner_val_window"
            except (RuntimeError, np.linalg.LinAlgError, ValueError) as exc:
                spec["status"] = "solver_failure"
                spec["reason"] = f"{type(exc).__name__}: {exc}"
                entry["variants"].setdefault(variant, {})[key] = spec
                continue
            model = fit["model"]
            log_mu = model.linear_predictor(x)
            intercept_ref = NegativeBinomialRidge(ridge=1.0, alpha_log_bounds=tuple(nb["alpha_log_bounds"])).fit(
                np.zeros((refit_rows.size, 1)), y[refit_rows])
            arrays[f"log_mu_{variant}_{key}"] = log_mu
            selected_by_horizon.setdefault(key, {})[variant] = float(fit["selected_ridge"])
            score = {}
            for phase in ("dev_val", "dev_test"):
                idx = rows[phase]
                score[phase] = {
                    "n_anchors": int(idx.size),
                    "mean_nb_nll": finite_or_none(model.nll(x[idx], y[idx]).mean()) if idx.size else None,
                    "mean_observed": finite_or_none(y[idx].mean()) if idx.size else None,
                    "mean_predicted": finite_or_none(model.predict_mu(x[idx]).mean()) if idx.size else None,
                    "intercept_only_mean_nb_nll": finite_or_none(
                        intercept_ref.nll(np.zeros((idx.size, 1)), y[idx]).mean()) if idx.size else None,
                }
            spec.update({
                "status": "ok",
                "selected_ridge": float(fit["selected_ridge"]),
                "ridge_at_edge": bool(fit["ridge_at_edge"]),
                "selection_nll": fit["selection_nll"],
                "ridge_path": fit["path"],
                "solver_failures": fit["solver_failures"],
                "nb_alpha": float(model.alpha_),
                "nb_log_dispersion": float(-np.log(model.alpha_)),   # log r, Var = mu + mu^2 / r
                "intercept": float(model.intercept_),
                "converged": bool(model.converged_),
                "log_mu_key": f"log_mu_{variant}_{key}",
                "scores": score,
                "feature_names": list(names),
                "model": model.state_dict(),
            })
            entry["variants"].setdefault(variant, {})[key] = spec
        primary = entry["variants"].get(primary_variant, {}).get(key)
        if primary is not None and primary.get("status") == "ok":
            entry["horizons"][key] = {k: v for k, v in primary.items() if k not in ("model", "feature_names", "ridge_path")}
    array_path = atomic_npz(out_dir / f"{tl.subject}_history_baseline.npz", arrays)
    entry["arrays"] = str(array_path)
    # The model agent's reader looks up `arrays` -> keys `anchor_time` + `log_mu_h` per horizon.
    for key in list(entry["horizons"]):
        h_arrays = {
            "anchor_time": tl.grid.t_anchor,
            "log_mu_h": arrays[f"log_mu_{primary_variant}_{key}"],
            "count": arrays[f"count_{key}"],
            "eligible": arrays[f"eligible_{key}"],
            "anchor_phase": arrays["anchor_phase"],
        }
        path = atomic_npz(out_dir / f"{tl.subject}_history_{primary_variant}_{key}s.npz", h_arrays)
        entry["horizons"][key]["arrays"] = str(path)
        entry["horizons"][key]["anchor_key"] = "anchor_time"
        entry["horizons"][key]["log_mu_key"] = "log_mu_h"
    return entry
