"""How far the per-patient propagation ordering drifts, and against which clock.

The frozen V3.0 line answered "does one event's unpredictable part forecast the
future ordering" and found no route-consistent signal.  It never measured the
plainer quantity: **how different is the ordering after N more events, or after
T more hours, and does crossing into another recording cost extra.**

Two questions live here, and they are separable only because event rate is not
constant:

* *event clock vs wall clock* — at a matched number of intervening events, do
  block pairs further apart in seconds look less alike?
* *cross-recording cost* — at a matched number of intervening events, are pairs
  that straddle a continuity-unit boundary less alike than pairs inside one unit?

Everything here is a distance measurement on frozen event streams.  No observer,
response model or state filter is fitted, and nothing can change the frozen V3.0
evidence level.

Masking is mandatory, not optional: a contact that never participated in a block
still carries a finite number in the stored rank field, and averaging it in would
manufacture a phantom ordering.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


def block_templates(
    rank_field: np.ndarray,
    participation: np.ndarray,
    source_label: Sequence[Any],
    abs_time: Sequence[float],
    *,
    block_events: int,
    min_participation: float,
) -> list[dict[str, Any]]:
    """Average the ordering over consecutive events, never across a source break.

    A block is `block_events` consecutive events of one continuity unit; a unit's
    trailing remainder is dropped rather than padded, so every block carries the
    same event budget.  Per contact the mean is taken over participating events
    only; a contact with no participation in the block gets `nan` and support 0.
    """

    rank = np.asarray(rank_field, dtype=float)
    part = np.asarray(participation).astype(bool)
    # Labels arrive as recording names, so they are strings, not row indices.
    sources = np.asarray(source_label).astype(str)
    times = np.asarray(abs_time, dtype=float)
    if int(block_events) < 2:
        raise ValueError("block_events must be at least 2")
    if not (rank.shape == part.shape):
        raise ValueError("rank_field and participation must have the same shape")
    if not (rank.shape[0] == sources.shape[0] == times.shape[0]):
        raise ValueError("event-aligned inputs must share the first dimension")

    size = int(block_events)
    blocks: list[dict[str, Any]] = []
    order = np.argsort(times, kind="stable")
    for label in np.unique(sources):
        unit = order[sources[order] == label]
        for start in range(0, len(unit) - size + 1, size):
            index = unit[start : start + size]
            unit_part = part[index]
            counts = unit_part.sum(axis=0).astype(float)
            support = counts / float(size)
            summed = np.where(unit_part, rank[index], 0.0).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_rank = np.where(counts > 0, summed / np.maximum(counts, 1.0), np.nan)
            mean_rank = np.where(support >= float(min_participation), mean_rank, np.nan)
            blocks.append(
                {
                    "source_id": str(label),
                    "event_start_index": int(index.min()),
                    "event_stop_index": int(index.max()) + 1,
                    "event_mid_index": float(np.mean(index)),
                    "n_events": size,
                    "t_start": float(times[index].min()),
                    "t_end": float(times[index].max()),
                    "t_mid": float(np.mean(times[index])),
                    "mean_rank": mean_rank,
                    "support": support,
                }
            )
    blocks.sort(key=lambda row: (row["t_mid"], row["source_id"]))
    return blocks


def template_similarity(
    left_rank: np.ndarray,
    left_support: np.ndarray,
    right_rank: np.ndarray,
    right_support: np.ndarray,
    min_support: float,
    min_shared: int,
) -> float | None:
    """Rank-order agreement over contacts supported on both sides."""

    a = np.asarray(left_rank, dtype=float)
    b = np.asarray(right_rank, dtype=float)
    shared = (
        (np.asarray(left_support, dtype=float) >= float(min_support))
        & (np.asarray(right_support, dtype=float) >= float(min_support))
        & np.isfinite(a)
        & np.isfinite(b)
    )
    if int(shared.sum()) < int(min_shared):
        return None
    left, right = a[shared], b[shared]
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    value = spearmanr(left, right).statistic
    return None if not np.isfinite(value) else float(value)


def drift_pairs(
    blocks: Sequence[Mapping[str, Any]],
    *,
    max_pairs: int,
    seed: int,
    min_support: float,
    min_shared: int,
) -> list[dict[str, Any]]:
    """Pairwise ordering agreement, tagged by event, time and source separation.

    When the full pair set exceeds `max_pairs` a deterministic random subset is
    drawn, so the returned pair count is never silently capped by truncation of
    an ordered list (which would bias toward short separations).
    """

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
        rows.append(
            {
                "d_events": abs(
                    float(right["event_mid_index"]) - float(left["event_mid_index"])
                ),
                "d_seconds": abs(float(right["t_mid"]) - float(left["t_mid"])),
                "same_source": str(left["source_id"]) == str(right["source_id"]),
                "similarity": similarity,
            }
        )
    return rows


#: A residual whose spread has collapsed to this fraction of the original spread
#: is numerical noise, not signal.  Correlating two such residuals returns a
#: spurious +/-1, which is the exact hazard here: inside one recording elapsed
#: seconds and intervening event count are near-collinear.
MINIMUM_RESIDUAL_FRACTION = 1e-6


def _rank_residual(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def rank_residual_fraction(
    driver: Sequence[float],
    control: Sequence[float],
) -> float | None:
    """Share of the driver's rank spread that survives removing the control.

    Recorded alongside every partial correlation so a near-zero-power cell is
    visible instead of silently reported as a confident zero.
    """

    x = np.asarray(driver, dtype=float)
    z = np.asarray(control, dtype=float)
    keep = np.isfinite(x) & np.isfinite(z)
    if int(keep.sum()) < 3:
        return None
    from scipy.stats import rankdata

    rx, rz = rankdata(x[keep]), rankdata(z[keep])
    spread = float(np.std(rx))
    if spread == 0.0:
        return None
    design = np.column_stack([np.ones_like(rz), rz])
    return float(np.std(_rank_residual(rx, design)) / spread)


def partial_spearman(
    outcome: Sequence[float],
    driver: Sequence[float],
    control: Sequence[float],
    *,
    minimum_n: int = 20,
) -> float | None:
    """Rank correlation of `outcome` with `driver` after removing `control`.

    Used to ask whether elapsed seconds still cost ordering agreement once the
    number of intervening events is accounted for.  Inside one recording the two
    clocks are strongly coupled, so this only has power where event rate varies;
    `None` means the control absorbed the driver and no independent variation
    was left, which is a different statement from "the effect is zero".
    """

    y = np.asarray(outcome, dtype=float)
    x = np.asarray(driver, dtype=float)
    z = np.asarray(control, dtype=float)
    keep = np.isfinite(y) & np.isfinite(x) & np.isfinite(z)
    if int(keep.sum()) < int(minimum_n):
        return None
    from scipy.stats import rankdata

    ry, rx, rz = (rankdata(v[keep]) for v in (y, x, z))
    if np.ptp(rz) == 0.0 or np.ptp(rx) == 0.0 or np.ptp(ry) == 0.0:
        return None
    design = np.column_stack([np.ones_like(rz), rz])
    resid_y = _rank_residual(ry, design)
    resid_x = _rank_residual(rx, design)
    if (
        np.std(resid_x) < MINIMUM_RESIDUAL_FRACTION * np.std(rx)
        or np.std(resid_y) < MINIMUM_RESIDUAL_FRACTION * np.std(ry)
    ):
        return None
    value = np.corrcoef(resid_y, resid_x)[0, 1]
    return None if not np.isfinite(value) else float(value)


def matched_event_distance_contrast(
    pairs: Sequence[Mapping[str, Any]],
    *,
    bin_edges: Sequence[float],
    min_pairs_per_cell: int,
) -> list[dict[str, Any]]:
    """Within-unit versus across-unit agreement at a matched event separation.

    Cells missing either arm are dropped rather than reported one-sided, because
    a one-armed cell cannot support the contrast the bin exists to make.
    """

    edges = list(map(float, bin_edges))
    cells = []
    for low, high in zip(edges, edges[1:]):
        inside = [
            row
            for row in pairs
            if low <= float(row["d_events"]) < high
        ]
        same = [row["similarity"] for row in inside if row["same_source"]]
        cross = [row["similarity"] for row in inside if not row["same_source"]]
        if len(same) < int(min_pairs_per_cell) or len(cross) < int(min_pairs_per_cell):
            continue
        same_seconds = [row["d_seconds"] for row in inside if row["same_source"]]
        cross_seconds = [row["d_seconds"] for row in inside if not row["same_source"]]
        same_events = [row["d_events"] for row in inside if row["same_source"]]
        cross_events = [row["d_events"] for row in inside if not row["same_source"]]
        cells.append(
            {
                "d_events_low": low,
                "d_events_high": high,
                "n_same_source": len(same),
                "n_cross_source": len(cross),
                "median_same_source": float(np.median(same)),
                "median_cross_source": float(np.median(cross)),
                "cross_minus_same": float(np.median(cross)) - float(np.median(same)),
                "median_seconds_same_source": float(np.median(same_seconds)),
                "median_seconds_cross_source": float(np.median(cross_seconds)),
                # Matching audit: the bin controls event separation only coarsely,
                # so the residual imbalance inside the bin must stay visible.
                "median_events_same_source": float(np.median(same_events)),
                "median_events_cross_source": float(np.median(cross_events)),
                "event_imbalance": float(np.median(cross_events))
                - float(np.median(same_events)),
            }
        )
    return cells
