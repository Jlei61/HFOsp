#!/usr/bin/env python3
"""Training-health evidence table: the engineering layer of the acceptance.

Answers, with numbers rather than assertion: did every run actually converge and
move its parameters, did any run hit a non-finite step or an OOM ladder, how long
and how much memory did they take -- and, per EI 2, **are the three seeds
genuinely different fits** rather than one fit counted three times (a
byte-identical checkpoint across seeds counts once).
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

from src.topic5_group_event_state.v02.registry import atomic_write_json, file_hash  # noqa: E402

DEFAULT_ROOT = Path("/data/hfosp_group_event_state_v0_2/agent_a/producers/main")
REPO_OUT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/group_event_state/v0_2/h1_h2a"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-root", type=Path, default=REPO_OUT)
    parser.add_argument("--tag", default="main")
    args = parser.parse_args()

    rows: list[dict] = []
    hashes: dict[tuple[str, str], dict[str, str]] = {}
    for result_path in sorted((args.producer_root / "runs").glob("*/*/seed*/result.json")):
        payload = json.loads(result_path.read_text())
        run_dir = result_path.parent
        ckpt = run_dir / "checkpoint.pt"
        h = file_hash(ckpt) if ckpt.exists() else ""
        key = (payload["subject"], payload["producer"])
        hashes.setdefault(key, {})[str(payload["seed"])] = h
        update = payload.get("param_update_magnitude", {})
        rows.append({
            "subject": payload["subject"],
            "producer": payload["producer"],
            "seed": payload["seed"],
            "selected_epoch": payload["selected_epoch"],
            "n_epochs_run": payload["n_epochs_run"],
            "stop_reason": payload["stop_reason"],
            "train_seconds": payload["train_seconds"],
            "peak_memory_reserved_gb": payload.get("peak_memory_reserved_gb"),
            "oom_retries": payload.get("oom_retries", 0),
            "n_train_segments": payload["n_train_segments"],
            "n_train_events": payload["n_train_events"],
            "n_train_anchors": payload["n_train_anchors"],
            "grad_norm_mean_last_epoch": payload["history"][-1]["grad_norm_mean"],
            "n_nonfinite_steps": sum(h_.get("n_nonfinite_steps", 0)
                                     for h_ in payload["history"]),
            "update_encoder": update.get("encoder"),
            "update_state": update.get("state"),
            "update_heads": update.get("heads"),
            "update_future": update.get("future"),
            "future_weight_5m": payload.get("future_loss_weights", {}).get("future_0"),
            "future_weight_30m": payload.get("future_loss_weights", {}).get("future_1"),
            "future_weight_120m": payload.get("future_loss_weights", {}).get("future_2"),
            "tau_slow_median_seconds": payload.get("tau_slow_seconds", [None, None, None])[1],
            "checkpoint_sha256": h,
            "selected_last_epoch": payload["selected_epoch"] == payload["n_epochs_run"] - 1,
        })

    duplicate_seeds = [
        {"subject": s, "producer": p, "identical_seeds": sorted(seeds)}
        for (s, p), by_seed in hashes.items()
        for h, seeds in
        {h: [k for k, v in by_seed.items() if v == h] for h in set(by_seed.values())}.items()
        if h and len(seeds) > 1
    ]

    def _stat(key: str) -> dict:
        values = np.array([r[key] for r in rows if r[key] is not None], dtype=float)
        if values.size == 0:
            return {}
        return {"median": float(np.median(values)), "min": float(values.min()),
                "max": float(values.max())}

    summary = {
        "n_runs": len(rows),
        "n_subjects": len({r["subject"] for r in rows}),
        "producers": sorted({r["producer"] for r in rows}),
        "stop_reasons": {
            reason: sum(1 for r in rows if r["stop_reason"] == reason)
            for reason in sorted({r["stop_reason"] for r in rows})
        },
        "n_runs_that_selected_their_last_epoch": sum(
            1 for r in rows if r["selected_last_epoch"]
        ),
        "n_runs_with_a_nonfinite_step": sum(1 for r in rows if r["n_nonfinite_steps"]),
        "n_runs_with_an_oom_retry": sum(1 for r in rows if r["oom_retries"]),
        "train_seconds": _stat("train_seconds"),
        "peak_memory_reserved_gb": _stat("peak_memory_reserved_gb"),
        "param_update_magnitude": {
            group: _stat(f"update_{group}")
            for group in ("encoder", "state", "heads", "future")
        },
        "future_loss_weights": {
            "5m": _stat("future_weight_5m"),
            "30m": _stat("future_weight_30m"),
            "120m": _stat("future_weight_120m"),
        },
        "tau_slow_median_seconds": _stat("tau_slow_median_seconds"),
        "n_subject_producer_groups_with_byte_identical_seeds": len(duplicate_seeds),
        "byte_identical_seed_groups": duplicate_seeds,
        "note": (
            "A run that selected its own last epoch never saw the validation "
            "objective turn, so its stopping point is a budget, not a "
            "convergence; those are counted separately.  Byte-identical seeds "
            "would mean one fit counted several times (EI 2)."
        ),
    }

    out = Path(args.out_root) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "training_health.json", summary)
    if rows:
        fields = list(rows[0])
        tmp = (out / "training_health.csv").with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(out / "training_health.csv")
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "byte_identical_seed_groups"}, indent=2))


if __name__ == "__main__":
    main()
