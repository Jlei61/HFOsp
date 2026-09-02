"""Common-drive covariates on a fixed physical grid, not on the event clock.

``M0`` is the arm that says IEDs are a readout of a shared slow process.  For that
arm to be a fair opponent it has to see the shared process: background SEEG, clock
time and where in the recording it is.  The v0.1 consolidated stream attaches a
background vector to every *event*, which is exactly the wrong shape here -- an
arm whose drive is applied once per event has the event count leaking into its
state through the number of times the drive was applied.

So the background is re-read from the block shards on its own 30 s grid, pooled
into fixed 5-minute cells, and used as a *relaxation target* that is constant
inside a cell.  Because ``b + (S - b) * exp(-dt/tau)`` composes exactly over
sub-intervals of a constant ``b``, inserting extra event steps inside a cell
leaves the state bit-for-bit unchanged.  That invariance is what makes ``M0``
genuinely event-free, and it is asserted in the regression tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .io import write_npz_atomic

# Background cells match the anchor grid; a cell is the finest resolution at
# which the common drive is allowed to change.
CELL_SECONDS = 300.0

# Clock covariates are evaluated once per cell, never per event, for the same
# count-invariance reason.
CLOCK_FEATURE_NAMES = (
    "sin_time_of_day",
    "cos_time_of_day",
    "sin_half_day",
    "cos_half_day",
    "log_hours_into_recording",
    "fraction_through_recording",
)


def _pool_contacts(features: np.ndarray) -> np.ndarray:
    """(anchors, contacts, f) -> (anchors, 2f): mean and spread across contacts.

    The state is a whole-network quantity, so the background enters as a network
    summary.  Keeping the spread as well as the mean means a drive that is focal
    (one shaft lighting up) is not confused with one that is global.
    """

    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"expected (anchors, contacts, features), got {x.shape}")
    finite = np.isfinite(x)
    x = np.where(finite, x, 0.0)
    n = finite.sum(axis=1).clip(min=1)
    mean = x.sum(axis=1) / n
    var = (np.where(finite, (x - mean[:, None, :]) ** 2, 0.0)).sum(axis=1) / n
    return np.concatenate([mean, np.sqrt(var)], axis=1)


def build_background_table(cache_dir: Path) -> dict[str, Any]:
    """Absolute-time background anchors for one patient, pooled over contacts.

    Times are float64 seconds since epoch throughout: a float32 timestamp near
    1.2e9 has a resolution of about 128 s, which would silently merge anchors.
    """

    cache_dir = Path(cache_dir)
    manifests = sorted(cache_dir.glob("*.manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"no shards under {cache_dir}")

    times: list[np.ndarray] = []
    feats: list[np.ndarray] = []
    names: list[str] | None = None
    window_s: float | None = None
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        shard = manifest_path.with_name(f"{manifest['record_name']}.npz")
        if not shard.exists():
            continue
        if names is None:
            names = list(manifest["background"]["feature_names"])
            window_s = float(manifest["background"]["window_seconds"])
        with np.load(shard) as z:
            t = np.asarray(z["background_time_s"], dtype=np.float64)
            if t.size == 0:
                continue
            f = _pool_contacts(z["background_features"])
        times.append(t + float(manifest["block_start_epoch"]))
        feats.append(f)

    if not times:
        raise ValueError(f"{cache_dir.name}: shards carry no background anchors")
    t_abs = np.concatenate(times)
    features = np.concatenate(feats, axis=0)
    order = np.argsort(t_abs, kind="stable")
    t_abs, features = t_abs[order], features[order]

    pooled_names = [f"{n}_mean" for n in (names or [])] + [f"{n}_spread" for n in (names or [])]
    return {
        "anchor_time": t_abs,
        "anchor_features": features.astype(np.float32),
        "feature_names": pooled_names,
        "window_seconds": float(window_s or 2.0),
    }


def cell_index(times: np.ndarray, origin: float, *, cell_seconds: float = CELL_SECONDS) -> np.ndarray:
    """Which fixed physical cell each absolute time falls in."""

    return np.floor((np.asarray(times, dtype=np.float64) - float(origin)) / float(cell_seconds)).astype(np.int64)


def cell_background(
    cell_starts: np.ndarray,
    anchor_time: np.ndarray,
    anchor_features: np.ndarray,
    *,
    cell_seconds: float = CELL_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal background for each cell: anchors that finished before it started.

    Strictly before, and only from the *previous* cell onward -- an anchor inside
    the cell would let the state read the interval it is being asked to predict
    over.  Cells with no preceding anchor get a validity flag of 0 rather than an
    imputed value, because a missing observation is not a quiet one.
    """

    starts = np.asarray(cell_starts, dtype=np.float64)
    t = np.asarray(anchor_time, dtype=np.float64)
    f = np.asarray(anchor_features, dtype=np.float32)
    out = np.zeros((starts.size, f.shape[1]), dtype=np.float32)
    valid = np.zeros(starts.size, dtype=bool)
    if t.size == 0:
        return out, valid

    lo = np.searchsorted(t, starts - float(cell_seconds), side="left")
    hi = np.searchsorted(t, starts, side="left")
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            out[i] = f[a:b].mean(axis=0)
            valid[i] = True
        elif b > 0:
            # Fall back to the most recent anchor before the cell, with the flag
            # still set: the value is real, just older than one cell.
            out[i] = f[b - 1]
            valid[i] = True
    return out, valid


def clock_features(times: np.ndarray, t_start: float, t_stop: float) -> np.ndarray:
    """Time-of-day and position-in-admission covariates, evaluated per cell."""

    t = np.asarray(times, dtype=np.float64)
    seconds_of_day = np.mod(t, 86400.0)
    phase = 2.0 * np.pi * seconds_of_day / 86400.0
    hours_in = np.clip((t - float(t_start)) / 3600.0, 0.0, None)
    span = max(float(t_stop) - float(t_start), 1.0)
    return np.stack(
        [
            np.sin(phase),
            np.cos(phase),
            np.sin(2.0 * phase),
            np.cos(2.0 * phase),
            np.log1p(hours_in),
            np.clip((t - float(t_start)) / span, 0.0, 1.0),
        ],
        axis=1,
    ).astype(np.float32)


def write_background_table(cache_dir: Path, out_path: Path) -> dict[str, Any]:
    table = build_background_table(cache_dir)
    write_npz_atomic(
        out_path,
        anchor_time=table["anchor_time"],
        anchor_features=table["anchor_features"],
    )
    meta = {
        "subject": Path(cache_dir).name,
        "n_anchors": int(table["anchor_time"].size),
        "feature_names": table["feature_names"],
        "window_seconds": table["window_seconds"],
        "cell_seconds": CELL_SECONDS,
        "clock_feature_names": list(CLOCK_FEATURE_NAMES),
        "first_anchor_epoch": float(table["anchor_time"][0]),
        "last_anchor_epoch": float(table["anchor_time"][-1]),
    }
    return meta
