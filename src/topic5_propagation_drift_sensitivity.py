"""Confound controls for the frozen propagation-drift readouts.

The frozen primary asked whether elapsed seconds still cost ordering agreement
once intervening event count is held fixed.  Two alternative explanations
survive that control and are tested here instead of being asserted away:

* **block coarseness** — every block holds exactly 20 events but spans a
  variable amount of time.  Where events are sparse a block's 20 events are
  spread over longer, so its averaged ordering is intrinsically noisier and will
  look less like anything.  A pair far apart in seconds tends to be built from
  such long blocks, which reproduces the frozen negative correlation without any
  drift.
* **shared-contact attrition** — pairs that straddle a recording boundary may
  share fewer supported contacts, and a rank correlation over fewer contacts is
  both noisier and differently scaled.

Both are additive controls, so the honest test is a partial rank correlation
against several controls at once.  This module is deliberately separate from the
frozen `topic5_propagation_drift` module so that adding it cannot invalidate the
hash lock on the already-executed primary.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.topic5_propagation_drift import (
    MINIMUM_RESIDUAL_FRACTION,
    template_similarity,
)


def _rank_matrix(columns: Sequence[Sequence[float]], keep: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return np.column_stack(
        [rankdata(np.asarray(column, dtype=float)[keep]) for column in columns]
    )


def partial_spearman_multi(
    outcome: Sequence[float],
    driver: Sequence[float],
    controls: Sequence[Sequence[float]],
    *,
    minimum_n: int = 200,
) -> dict[str, Any]:
    """Rank correlation of outcome with driver after removing several controls.

    Returns the correlation together with the share of the driver's rank spread
    that survived the controls, because a partial correlation computed on a
    driver the controls have already absorbed is numerical noise dressed as a
    result.
    """

    y = np.asarray(outcome, dtype=float)
    x = np.asarray(driver, dtype=float)
    stack = [np.asarray(column, dtype=float) for column in controls]
    if not stack:
        raise ValueError("at least one control is required")
    keep = np.isfinite(y) & np.isfinite(x)
    for column in stack:
        keep &= np.isfinite(column)
    if int(keep.sum()) < int(minimum_n):
        return {"rho": None, "residual_fraction": None, "n": int(keep.sum()),
                "status": "UNRESOLVED_TOO_FEW_PAIRS"}

    from scipy.stats import rankdata

    ry = rankdata(y[keep])
    rx = rankdata(x[keep])
    design = np.column_stack([np.ones(int(keep.sum())), _rank_matrix(stack, keep)])
    coefficients, *_ = np.linalg.lstsq(design, np.column_stack([ry, rx]), rcond=None)
    residuals = np.column_stack([ry, rx]) - design @ coefficients
    resid_y, resid_x = residuals[:, 0], residuals[:, 1]
    fraction = float(np.std(resid_x) / np.std(rx)) if np.std(rx) > 0 else None
    if (
        fraction is None
        or np.std(resid_x) < MINIMUM_RESIDUAL_FRACTION * np.std(rx)
        or np.std(resid_y) < MINIMUM_RESIDUAL_FRACTION * np.std(ry)
    ):
        return {"rho": None, "residual_fraction": fraction, "n": int(keep.sum()),
                "status": "UNRESOLVED_COLLINEAR"}
    value = float(np.corrcoef(resid_y, resid_x)[0, 1])
    return {
        "rho": None if not np.isfinite(value) else value,
        "residual_fraction": fraction,
        "n": int(keep.sum()),
        "status": "RESOLVED",
    }


def annotated_pairs(
    blocks: Sequence[Mapping[str, Any]],
    *,
    max_pairs: int,
    seed: int,
    min_support: float,
    min_shared: int,
) -> list[dict[str, Any]]:
    """Same pairs as the frozen primary, plus the two confound covariates."""

    n = len(blocks)
    index_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(index_pairs) > int(max_pairs):
        rng = np.random.default_rng(int(seed))
        chosen = rng.choice(len(index_pairs), size=int(max_pairs), replace=False)
        index_pairs = [index_pairs[int(k)] for k in np.sort(chosen)]

    rows = []
    for i, j in index_pairs:
        left, right = blocks[i], blocks[j]
        similarity = template_similarity(
            left["mean_rank"],
            left["support"],
            right["mean_rank"],
            right["support"],
            min_support,
            min_shared,
        )
        if similarity is None:
            continue
        shared = int(
            np.sum(
                (np.asarray(left["support"]) >= min_support)
                & (np.asarray(right["support"]) >= min_support)
                & np.isfinite(np.asarray(left["mean_rank"], dtype=float))
                & np.isfinite(np.asarray(right["mean_rank"], dtype=float))
            )
        )
        left_span = float(left["t_end"]) - float(left["t_start"])
        right_span = float(right["t_end"]) - float(right["t_start"])
        rows.append(
            {
                # Block indices travel with the pair so downstream stratifiers
                # (day/night phase, for one) can join per-block attributes
                # without rebuilding the pair set and risking a different subset.
                "left_index": i,
                "right_index": j,
                "d_events": abs(
                    float(right["event_mid_index"]) - float(left["event_mid_index"])
                ),
                "d_seconds": abs(float(right["t_mid"]) - float(left["t_mid"])),
                "same_source": str(left["source_id"]) == str(right["source_id"]),
                "similarity": similarity,
                "n_shared_contacts": shared,
                "mean_block_span_seconds": 0.5 * (left_span + right_span),
                "max_block_span_seconds": max(left_span, right_span),
            }
        )
    return rows
