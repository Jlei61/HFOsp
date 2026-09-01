#!/usr/bin/env python3
"""Derive A4 latent-decomposition and matched-donor state variants (CPU only).

* ``fast_only`` / ``slow_only`` -- frozen read-outs of one latent block.  These
  are column slices of an already-frozen state, so they cost nothing and add no
  new fit; SP 4.2 keeps them as diagnostics, never as the slow-state claim.
* ``matched_donor`` -- the coarse wrong-time control.  An anchor's state is
  replaced by the state of a *different* anchor that matches only on session,
  time-of-day bin, coverage bin and recent-rate bin (CC 6 forbids matching on
  size or participation, which may themselves be the slow signal).  Anchors with
  no donor are written as NaN and the matchable fraction is recorded, so the
  consumer scores this arm on its own subset instead of quietly averaging over a
  half-built control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v02.registry import atomic_write_json  # noqa: E402
from src.topic5_group_event_state.v02.subject import (  # noqa: E402
    SubjectTimelineConfig,
    load_subject_timeline,
)

MIN_DONOR_SEPARATION_SECONDS = 3600.0
N_DONORS = 10


def _write(out_dir: Path, subject: str, state: np.ndarray, template: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{subject}.npz"
    tmp = dst.with_suffix(".npz.tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, state=state.astype(np.float32), **template)
    tmp.replace(dst)


def matched_donor_state(
    values: np.ndarray, tl, *, seed: int
) -> tuple[np.ndarray, float]:
    """Replace each anchor's state with a coarsely matched wrong-time anchor's."""

    rng = np.random.default_rng(seed)
    t = tl.grid.t_anchor
    session = tl.grid.session_id
    tod = ((t % 86400.0) // (86400.0 / 4)).astype(np.int64)

    names = list(tl.baseline.names)
    cover = np.digitize(
        tl.baseline.x[:, names.index("log_seconds_into_segment")],
        np.quantile(tl.baseline.x[:, names.index("log_seconds_into_segment")], [0.33, 0.67]),
    )
    rate_col = names.index("rate_tau1800")
    rate = np.digitize(
        tl.baseline.x[:, rate_col],
        np.quantile(tl.baseline.x[:, rate_col], [0.33, 0.67]),
    )

    keys = np.stack([session, tod, cover, rate], axis=1)
    out = np.full_like(values, np.nan)
    matched = 0
    lookup: dict[tuple, np.ndarray] = {}
    for k in {tuple(row) for row in keys}:
        lookup[k] = np.flatnonzero((keys == np.array(k)).all(axis=1))
    for i in range(values.shape[0]):
        pool = lookup[tuple(keys[i])]
        pool = pool[np.abs(t[pool] - t[i]) > MIN_DONOR_SEPARATION_SECONDS]
        if pool.size == 0:
            continue
        pick = pool if pool.size <= N_DONORS else rng.choice(pool, N_DONORS, replace=False)
        out[i] = values[int(rng.choice(pick))]
        matched += 1
    return out, matched / max(values.shape[0], 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--d-fast", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variants", nargs="+",
                        default=["fast_only", "slow_only", "matched_donor"])
    args = parser.parse_args()

    src = Path(args.state_dir)
    report: dict[str, dict] = {}
    for path in sorted(src.glob("*.npz")):
        subject = path.stem
        with np.load(path) as z:
            state = np.asarray(z["state"], dtype=np.float32)
            template = {k: np.asarray(z[k]) for k in z.files if k != "state"}
        entry: dict[str, float] = {}
        if "fast_only" in args.variants:
            _write(src.parent / f"{src.name}_fast_only", subject,
                   state[:, : args.d_fast], template)
        if "slow_only" in args.variants:
            _write(src.parent / f"{src.name}_slow_only", subject,
                   state[:, args.d_fast:], template)
        if "matched_donor" in args.variants:
            tl = load_subject_timeline(subject, config=SubjectTimelineConfig())
            donor, frac = matched_donor_state(state.astype(np.float64), tl, seed=args.seed)
            _write(src.parent / f"{src.name}_matched_donor", subject, donor, template)
            entry["matched_fraction"] = float(frac)
        report[subject] = entry
        print(f"{subject}: {entry}", flush=True)
    atomic_write_json(src.parent / f"{src.name}_variants.json", {
        "source_state_dir": str(src), "d_fast": args.d_fast,
        "variants": list(args.variants), "per_subject": report,
        "min_donor_separation_seconds": MIN_DONOR_SEPARATION_SECONDS,
        "n_donors": N_DONORS,
    })


if __name__ == "__main__":
    main()
