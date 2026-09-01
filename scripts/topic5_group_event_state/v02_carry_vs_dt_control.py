#!/usr/bin/env python3
"""Does carrying history beat a control that carries only the time since the last event?

This is the contrast that decides H1, and it is not the pre-registered
block-shift null.  Shifting a state in time destroys the correspondence between
the state and the block that follows it -- but it *also* destroys the anchor's own
``time since the last event``, so a shifted arm loses both at once.  The
memoryless producer separates them: its state is, by construction, a
96-dimensional exponential basis in that one number and nothing else
(``test_the_memoryless_arm_carries_only_the_time_since_the_last_event``).

Two tables are written:

* the paired per-patient contrast ``B + S(carrying)`` vs ``B + S(dt-only)``;
* how much of each producer's own state variance is a smooth function of the time
  since the last event, which is what makes the contrast interpretable.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v02.aggregate import load_results, sign_test_p  # noqa: E402
from src.topic5_group_event_state.v02.registry import atomic_write_json  # noqa: E402
from src.topic5_group_event_state.v02.subject import (  # noqa: E402
    SubjectTimelineConfig,
    load_subject_timeline,
)

REPO_OUT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/group_event_state/v0_2/h1_h2a"
)
STATE_ROOT = Path("/data/hfosp_group_event_state_v0_2/agent_a/producers/main/states")
CONTROL = "P_memoryless_seed"
ENDPOINTS = ("count", "participation", "continuous")


def _median_score(entry, prefix: str, endpoint: str) -> float | None:
    values = [
        payload["scores"][endpoint]["nll_per_unit"]
        for name, payload in entry["arms"].items()
        if name.startswith(prefix)
        and payload.get("estimability", {}).get(endpoint, "ok") == "ok"
        and endpoint in payload["scores"]
    ]
    return float(np.median(values)) if values else None


def dt_explained_variance(subject: str, state_dir: Path, degree: int = 5) -> float:
    """Share of a producer's state variance that a smooth function of dt explains."""

    path = state_dir / f"{subject}.npz"
    if not path.exists():
        return float("nan")
    tl = load_subject_timeline(subject, config=SubjectTimelineConfig())
    with np.load(path) as z:
        state = np.asarray(z["state"], dtype=np.float64)
    dt = tl.grid.seconds_since_last_event
    ok = np.isfinite(dt)
    x = np.log1p(np.clip(dt[ok], 0.0, None))
    basis = np.stack([x ** k for k in range(degree + 1)], axis=1)
    beta, *_ = np.linalg.lstsq(basis, state[ok], rcond=None)
    resid = state[ok] - basis @ beta
    denom = float(state[ok].var(axis=0).sum())
    return float(1.0 - resid.var(axis=0).sum() / denom) if denom > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--future-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, default=STATE_ROOT)
    parser.add_argument("--carrying", nargs="+", default=["P_slow", "P_local"])
    parser.add_argument("--out-root", type=Path, default=REPO_OUT)
    parser.add_argument("--tag", default="main")
    args = parser.parse_args()

    results = load_results(args.future_root)
    rows: list[dict] = []
    for endpoint in ENDPOINTS:
        for horizon in sorted({h for r in results for h in r["horizons"]},
                              key=lambda s: float(s[:-1])):
            for carrying in args.carrying:
                diffs: dict[str, float] = {}
                for r in results:
                    entry = r["horizons"].get(horizon)
                    if not entry or entry.get("status") != "ok":
                        continue
                    a = _median_score(entry, f"B+S({carrying}_seed", endpoint)
                    b = _median_score(entry, f"B+S({CONTROL}", endpoint)
                    if a is None or b is None:
                        continue
                    diffs[r["subject"]] = b - a          # positive = carrying wins
                v = np.array(list(diffs.values()), dtype=float)
                if v.size == 0:
                    continue
                rows.append({
                    "endpoint": endpoint,
                    "horizon_seconds": float(horizon[:-1]),
                    "carrying_producer": carrying,
                    "control": "P_memoryless (state = a function of dt alone)",
                    "n_subjects": int(v.size),
                    "n_carrying_better": int((v > 0).sum()),
                    "median_advantage_of_carrying": float(np.median(v)),
                    "p_sign": sign_test_p(int((v > 0).sum()), int(v.size)),
                    "per_subject": json.dumps({k: round(x, 6) for k, x in diffs.items()}),
                })

    dt_rows: list[dict] = []
    subjects = sorted(r["subject"] for r in results)
    for producer in list(args.carrying) + ["P_memoryless"]:
        state_dir = args.state_root / f"{producer}_seed1"
        if not state_dir.exists():
            continue
        for subject in subjects:
            dt_rows.append({
                "producer": producer,
                "subject": subject,
                "dt_explained_variance": dt_explained_variance(subject, state_dir),
            })

    out = Path(args.out_root) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    for name, table in (("carrying_vs_dt_only.csv", rows),
                        ("state_variance_explained_by_dt.csv", dt_rows)):
        if not table:
            continue
        tmp = (out / name).with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        tmp.replace(out / name)

    summary = {
        "question": (
            "Does a producer that carries history beat one whose state is a "
            "function of the time since the last event and nothing else?"
        ),
        "why_the_block_shift_null_cannot_answer_it": (
            "Shifting a state within its session destroys both the link to the "
            "following block and the anchor's own time-since-last-event, so a "
            "shifted arm loses two things at once and cannot separate them."
        ),
        "cells": [{k: v for k, v in r.items() if k != "per_subject"} for r in rows],
        "state_variance_explained_by_dt_median": {
            p: float(np.nanmedian([r["dt_explained_variance"] for r in dt_rows
                                   if r["producer"] == p]))
            for p in sorted({r["producer"] for r in dt_rows})
        },
    }
    atomic_write_json(out / "carrying_vs_dt_only.json", summary)
    print(json.dumps(summary["state_variance_explained_by_dt_median"], indent=2))
    for r in summary["cells"]:
        print(f"{r['endpoint']:14s} {int(r['horizon_seconds']):5d}s {r['carrying_producer']:8s} "
              f"median {r['median_advantage_of_carrying']:+.5f} "
              f"{r['n_carrying_better']}/{r['n_subjects']} p={r['p_sign']:.3f}")


if __name__ == "__main__":
    main()
