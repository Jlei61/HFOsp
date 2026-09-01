"""Run every arm of the A1/A2 comparison on one patient's fixed anchor grid.

The arms differ only in which columns enter ``X``:

    intercept          nothing (TRAIN marginals) -- the floor, always reported
    B_multiscale       the interpretable multiscale baseline
    B + S(producer)    the load-bearing nested increment (CC 6)
    S(producer)        state alone, secondary
    B + shift_k(S)     the time-specificity null (CC 6), k*grid > horizon

Everything else -- anchors, windows, masks, standardisation, likelihoods,
optimiser, ridge grid -- is identical across arms by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .readout import (
    ReadoutConfig,
    block_circular_shift,
    estimability,
    fit_readout,
    gain_table,
    reference_scores,
    score_readout,
    shiftable_sessions,
    validate_shift_exceeds_horizon,
)
from .subject import SubjectTimeline
from .timeline import SPLIT_NAMES, effective_independent_windows

# Reported (not acted on): below this many independent windows the 10% inner
# split is too thin to say anything on its own.  Ridge selection does not depend
# on it -- lambda comes from chronological CV inside TRAIN -- but a reader needs
# to know which (patient, horizon) cells are thin.
MIN_VAL_INDEPENDENT_WINDOWS = 3

# Block-circular-shift offsets, in grid steps beyond the horizon.  Several are
# run and the null level is their median, so one unlucky alignment cannot decide
# the time-specificity question.
SHIFT_EXTRA_STEPS: tuple[int, ...] = (1, 2, 4, 8)


@dataclass(frozen=True)
class EvaluationConfig:
    readout: ReadoutConfig = ReadoutConfig()
    mlp_hidden: int = 32
    run_mlp_baseline: bool = True
    shift_extra_steps: tuple[int, ...] = SHIFT_EXTRA_STEPS
    # Secondary arms are run for a named subset of producers.  The load-bearing
    # increment ``B`` vs ``B+S`` is always run for every producer; the state-only
    # arm and the extra shift offsets are diagnostics, and running them for every
    # (producer, seed) triples the cost of the whole evaluation for no additional
    # load-bearing content.
    state_only_for: tuple[str, ...] | None = None
    shift_for: tuple[str, ...] | None = None
    extra_shift_for: tuple[str, ...] | None = None


def _stack(*blocks: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(b, dtype=np.float64) for b in blocks], axis=1)


def _fit_and_score(
    tl: SubjectTimeline,
    x: np.ndarray,
    h_i: int,
    *,
    config: ReadoutConfig,
    forced_lambdas: Mapping[str, float] | None = None,
    extra_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
    tr = tl.anchor_rows("train", h_i, extra_mask)
    va = tl.anchor_rows("val", h_i, extra_mask)
    te = tl.anchor_rows("test", h_i, extra_mask)
    stats_tr = tl.window_stats("train", h_i, extra_mask)
    stats_va = tl.window_stats("val", h_i, extra_mask)
    stats_te = tl.window_stats("test", h_i, extra_mask)
    pinned = (
        None if not forced_lambdas
        else {k: (float(v),) for k, v in forced_lambdas.items()}
    )
    fit = fit_readout(
        x[tr], stats_tr, x[va], stats_va,
        n_contacts=tl.n_contacts, n_dims=tl.n_dims, config=config,
        family_lambdas=pinned,
    )
    scores = score_readout(
        fit, x[te], stats_te, block_slices=tl.marks.block_slices,
        n_contacts=tl.n_contacts, n_dims=tl.n_dims,
    )
    payload = {
        "scores": {k: v.as_dict() for k, v in scores.items()},
        "lambda_by_family": dict(fit.lam),
        "lambda_at_grid_edge": dict(fit.lam_at_grid_edge),
        "n_features": fit.n_features,
        "val_nll_per_unit": dict(fit.val_objective),
    }
    return payload, dict(fit.lam), scores


def evaluate_subject(
    tl: SubjectTimeline,
    state_by_producer: Mapping[str, np.ndarray] | None = None,
    *,
    config: EvaluationConfig = EvaluationConfig(),
) -> dict[str, Any]:
    """All arms x all horizons for one patient, on one shared anchor grid."""

    states = dict(state_by_producer or {})
    for name, values in states.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape[0] != tl.grid.n_anchors:
            raise ValueError(
                f"{name}: state has {arr.shape[0]} rows for {tl.grid.n_anchors} anchors"
            )
        states[name] = arr

    ones = np.ones((tl.grid.n_anchors, 1), dtype=np.float64)
    x_base = _stack(ones, tl.baseline.x)
    x_clock = _stack(ones, tl.baseline.x[:, tl.baseline.clock_only_columns()])

    out: dict[str, Any] = {
        "subject": tl.subject,
        "dataset": tl.dataset,
        "config": {
            "readout_lambdas": list(config.readout.lambdas),
            "mlp_hidden": config.mlp_hidden,
            "shift_extra_steps": list(config.shift_extra_steps),
            "min_val_independent_windows": MIN_VAL_INDEPENDENT_WINDOWS,
        },
        "producers_present": sorted(states),
        "horizons": {},
    }

    for h_i, horizon in enumerate(tl.config.horizons_seconds):
        key = f"{int(horizon)}s"
        counts = {
            name: {
                "n_anchors": int(tl.anchor_mask(name, h_i).sum()),
                "n_independent_windows": effective_independent_windows(
                    tl.segments, tl.split, name, horizon
                ),
            }
            for name in SPLIT_NAMES
        }
        entry: dict[str, Any] = {"denominators": counts, "arms": {}}
        if any(counts[n]["n_anchors"] == 0 for n in SPLIT_NAMES):
            entry["status"] = "insufficient_coverage"
            out["horizons"][key] = entry
            continue
        entry["status"] = "ok"

        entry["val_is_thin"] = bool(
            counts["val"]["n_independent_windows"] < MIN_VAL_INDEPENDENT_WINDOWS
        )

        stats_tr = tl.window_stats("train", h_i)
        stats_te = tl.window_stats("test", h_i)
        ref = reference_scores(
            stats_tr, stats_te, n_contacts=tl.n_contacts, n_dims=tl.n_dims
        )
        entry["arms"]["intercept"] = {
            "scores": {k: v.as_dict() for k, v in ref.items()},
            "kind": "reference",
        }

        def _run(name: str, x: np.ndarray, kind: str, readout: ReadoutConfig,
                 extra_mask: np.ndarray | None = None) -> Any:
            payload, _lam, scores = _fit_and_score(
                tl, x, h_i, config=readout, extra_mask=extra_mask
            )
            payload["kind"] = kind
            payload["estimability"] = estimability(scores, ref)
            if extra_mask is not None:
                payload["anchor_subset"] = {
                    "n_test_anchors": int(tl.anchor_rows("test", h_i, extra_mask).sum()),
                    "fraction_of_full": float(
                        tl.anchor_rows("test", h_i, extra_mask).sum()
                        / max(int(tl.anchor_mask("test", h_i).sum()), 1)
                    ),
                }
            entry["arms"][name] = payload
            return scores

        _run("B_clock_only", x_clock, "clock_and_current_event_baseline", config.readout)
        base_scores = _run("B_multiscale", x_base, "baseline", config.readout)
        if config.run_mlp_baseline:
            _run(
                "B_multiscale_mlp",
                x_base,
                "baseline_capacity_check",
                ReadoutConfig(
                    lambdas=config.readout.lambdas, max_iter=config.readout.max_iter,
                    hidden=config.mlp_hidden, seed=config.readout.seed,
                ),
            )

        for producer, values in sorted(states.items()):
            finite = np.isfinite(values).all(axis=1)
            subset = None if bool(finite.all()) else finite
            local_base = base_scores
            if subset is not None:
                # A control that could not be built for every anchor is scored on
                # its own subset, against a baseline re-fitted on that same
                # subset -- never against the full-anchor baseline.
                local_base = _run(f"B_multiscale|subset({producer})", x_base,
                                  "baseline_on_subset", config.readout, subset)
            s = _run(f"B+S({producer})", _stack(x_base, values),
                     "baseline_plus_state", config.readout, subset)
            entry["arms"][f"B+S({producer})"]["gain_vs_baseline"] = gain_table(s, local_base)
            if config.state_only_for is None or producer in config.state_only_for:
                s_alone = _run(f"S({producer})", _stack(ones, values),
                               "state_only", config.readout, subset)
                entry["arms"][f"S({producer})"]["gain_vs_baseline"] = gain_table(
                    s_alone, local_base
                )

            shift_gains: list[dict[str, float]] = []
            min_steps = int(math.ceil(horizon / tl.config.grid_seconds))
            # A state that is already a wrong-time control (the matched donor
            # writes NaN where no donor existed) is not shifted again: rolling it
            # would mix donor rows with the zeros that stand in for missing ones.
            shift_steps_list = () if subset is not None else config.shift_extra_steps
            if config.shift_for is not None and producer not in config.shift_for:
                shift_steps_list = ()
            elif (config.extra_shift_for is not None
                  and producer not in config.extra_shift_for):
                shift_steps_list = shift_steps_list[:1]
            for extra in shift_steps_list:
                steps = min_steps + int(extra)
                validate_shift_exceeds_horizon(steps, tl.config.grid_seconds, horizon)
                usable, total = shiftable_sessions(tl.grid.session_id, steps)
                if usable == 0:
                    continue
                shifted = block_circular_shift(
                    np.nan_to_num(values), tl.grid.session_id, tl.grid.t_anchor, steps
                )
                name = f"B+shift{steps}(S({producer}))"
                s_shift = _run(name, _stack(x_base, shifted), "block_shift_null",
                               config.readout, subset)
                g = gain_table(s_shift, local_base)
                entry["arms"][name]["gain_vs_baseline"] = g
                entry["arms"][name]["shift_steps"] = steps
                entry["arms"][name]["shift_seconds"] = steps * tl.config.grid_seconds
                entry["arms"][name]["n_anchors_in_shiftable_sessions"] = usable
                entry["arms"][name]["n_anchors_total"] = total
                shift_gains.append(g)
            if shift_gains:
                keys = sorted(set().union(*[set(g) for g in shift_gains]))
                entry.setdefault("shift_null_median_gain", {})[producer] = {
                    k: float(np.median([g[k] for g in shift_gains if k in g]))
                    for k in keys
                }
        out["horizons"][key] = entry
    return out
