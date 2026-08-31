#!/usr/bin/env python3
"""Train and evaluate one (patient, arm, seed) of Group-Event State v0.1.

A CUDA out-of-memory is a *resource* failure, not a scientific result: the run
halves its event chunk and retries, and if it still cannot fit it is recorded as
``resource_failed`` so it can never be read as "this arm did not learn".
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.topic5_group_event_state.dataset import SubjectSequence  # noqa: E402
from src.topic5_group_event_state.train import (  # noqa: E402
    TrainConfig,
    build_arms,
    train_one,
)

MAIN_TREE = Path("/home/honglab/leijiaxin/HFOsp")
V0_1 = MAIN_TREE / "results/epi_prssm/group_event_state/v0_1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_1/dataset"))
    parser.add_argument("--out-root", type=Path, default=V0_1 / "runs")
    parser.add_argument("--chunk-events", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--max-train-seconds", type=float, default=2400.0)
    parser.add_argument("--lr-encoder", type=float, default=3e-4)
    parser.add_argument("--lr-state", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--oom-retries", type=int, default=3)
    parser.add_argument("--tag", default="main")
    args = parser.parse_args()

    run_id = f"{args.subject}__{args.arm}__seed{args.seed}"
    out_dir = Path(args.out_root) / args.tag / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if result_path.exists():
        print(f"SKIP {run_id} (already complete)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq = SubjectSequence(Path(args.dataset_root) / args.subject)
    arm = build_arms()[args.arm]
    cfg = TrainConfig(
        chunk_events=args.chunk_events,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_epochs=args.min_epochs,
        max_train_seconds=args.max_train_seconds,
        lr_encoder=args.lr_encoder,
        lr_state=args.lr_state,
        weight_decay=args.weight_decay,
        amp=not args.no_amp,
    )

    attempts: list[dict] = []
    started = time.time()
    for attempt in range(args.oom_retries + 1):
        try:
            result = train_one(seq, arm, args.seed, cfg, device, out_dir)
            result["status"] = "ok"
            result["oom_attempts"] = attempts
            result["gpu"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            result["peak_gpu_bytes"] = (
                int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
            )
            result["wall_seconds"] = round(time.time() - started, 1)
            tmp = result_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(result, indent=2, sort_keys=True, default=float))
            os.replace(tmp, result_path)
            print(
                f"OK {run_id} epoch={result['selected_epoch']} "
                f"test_delay={result['test']['delay']:.4f} "
                f"test_timing={result['test']['timing']:.4f} "
                f"{result['wall_seconds']}s"
            )
            return
        except torch.cuda.OutOfMemoryError as exc:
            attempts.append(
                {"attempt": attempt, "chunk_events": cfg.chunk_events, "error": str(exc)[:200]}
            )
            torch.cuda.empty_cache()
            cfg = replace(cfg, chunk_events=max(8, cfg.chunk_events // 2))
            print(f"OOM {run_id}: retrying with chunk_events={cfg.chunk_events}", flush=True)
        except Exception as exc:
            payload = {
                "status": "error",
                "subject": args.subject,
                "arm": args.arm,
                "seed": args.seed,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=12),
            }
            (out_dir / "error.json").write_text(json.dumps(payload, indent=2))
            print(f"ERROR {run_id}: {exc}", flush=True)
            raise SystemExit(1)

    payload = {
        "status": "resource_failed",
        "subject": args.subject,
        "arm": args.arm,
        "seed": args.seed,
        "oom_attempts": attempts,
        "note": "CUDA OOM after all retries; this is NOT a negative scientific result",
    }
    (out_dir / "resource_failed.json").write_text(json.dumps(payload, indent=2))
    print(f"RESOURCE_FAILED {run_id}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
