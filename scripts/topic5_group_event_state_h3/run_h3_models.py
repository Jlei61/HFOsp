#!/usr/bin/env python3
"""C2/C3: fit one (patient, arm, seed) and score its unseen future blocks.

Arms differ in exactly one thing -- whether events reach the state transition, and
through which channel.  Everything else, including the checkpoint criterion, is
shared, so the reported gain is attributable to the edge rather than to the fit.

Scores are written per block, not pre-aggregated: the independent denominator is
the disjoint physical block, and only a file that keeps the blocks apart can be
re-analysed under a different aggregation later without re-running the model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state_h3.io import (  # noqa: E402
    payload_hash,
    write_json_atomic,
    write_npz_atomic,
)
from src.topic5_group_event_state_h3.models import ARM_NAMES, H3Config, build_model  # noqa: E402
from src.topic5_group_event_state_h3.runtime import (  # noqa: E402
    AGENT_C_ROOT,
    context_summary,
    disjoint_mask,
    load_subject,
)
from src.topic5_group_event_state_h3.support import MAIN_HORIZONS_MINUTES  # noqa: E402
from src.topic5_group_event_state_h3.train import (  # noqa: E402
    TrainConfig,
    run_epoch,
    train_arm,
)

OUT_ROOT = ROOT / "results/epi_prssm/group_event_state/v0_2/h3"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def evaluate(ctx, model, cfg, horizons, device, splits=("development_test", "inner_validation")):
    """Full causal pass, then keep only the pre-registered disjoint blocks."""

    model.eval()
    with torch.no_grad():
        _loss, collected = run_epoch(
            model, ctx.tensors, horizons, device, cfg,
            train_split="train", collect_splits=splits,
        )

    per_block: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}
    for split in splits:
        blocks = collected.get(split, {})
        summary[split] = {}
        for horizon in horizons:
            ids = blocks.get(f"anchor_ids_{horizon}")
            if ids is None or ids.size == 0:
                summary[split][str(horizon)] = {"status": "no_blocks"}
                continue
            segs = blocks[f"segment_{horizon}"]
            keep = disjoint_mask(ctx, horizon, segs, ids)
            if not keep.any():
                # Sliding anchors exist but none of them is independent.  Saying
                # so is the finding; averaging the overlapping ones would report a
                # denominator this recording never had.
                summary[split][str(horizon)] = {
                    "status": "no_independent_blocks",
                    "n_sliding_anchors_not_independent": int(ids.size),
                    "n_disjoint_blocks": 0,
                }
                continue
            count_ll = blocks[f"count_{horizon}"][keep]
            mark_ll = blocks[f"mark_{horizon}"][keep]
            groups = blocks[f"mark_groups_{horizon}"][keep]
            has = blocks[f"has_{horizon}"][keep]
            truth = blocks[f"count_true_{horizon}"][keep]
            prefix = f"{split}__{horizon}"
            per_block[f"{prefix}__segment"] = segs[keep]
            per_block[f"{prefix}__anchor_id"] = ids[keep]
            per_block[f"{prefix}__anchor_time"] = np.asarray(
                [ctx.anchor_time(int(s), int(a)) for s, a in zip(segs[keep], ids[keep])],
                dtype=np.float64,
            )
            per_block[f"{prefix}__count_logscore"] = count_ll
            per_block[f"{prefix}__mark_logscore"] = mark_ll
            per_block[f"{prefix}__mark_group_logscore"] = groups
            per_block[f"{prefix}__has_events"] = has
            per_block[f"{prefix}__count_true"] = truth
            summary[split][str(horizon)] = {
                "status": "ok",
                "n_disjoint_blocks": int(keep.sum()),
                "n_sliding_anchors_not_independent": int(ids.size),
                "n_blocks_with_events": int(has.sum()),
                "mean_count_logscore": float(np.mean(count_ll)),
                "mean_mark_logscore": float(np.mean(mark_ll[has])) if bool(has.any()) else float("nan"),
                "mean_mark_group_logscore": {
                    name: float(np.mean(groups[has, i])) if bool(has.any()) else float("nan")
                    for i, (name, _sl) in enumerate(ctx.tensors.mark_groups)
                },
                "median_count_true": float(np.median(truth)),
            }
    return summary, per_block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--arm", required=True, choices=list(ARM_NAMES))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizons", nargs="*", type=int, default=list(MAIN_HORIZONS_MINUTES))
    parser.add_argument("--max-epochs", type=int, default=TrainConfig.max_epochs)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--max-train-seconds", type=float, default=TrainConfig.max_train_seconds)
    parser.add_argument("--window-steps", type=int, default=TrainConfig.window_steps)
    parser.add_argument("--d-state", type=int, default=H3Config.d_state)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=AGENT_C_ROOT / "checkpoints")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_id = f"{args.subject}__{args.arm}__seed{args.seed}"
    result_path = Path(args.out_root) / "machine" / args.tag / f"{run_id}.json"
    block_path = Path(args.checkpoint_root) / args.tag / f"{run_id}__blocks.npz"
    ckpt_path = Path(args.checkpoint_root) / args.tag / f"{run_id}__checkpoint.pt"

    model_cfg = H3Config(d_state=args.d_state, horizons_minutes=tuple(args.horizons))
    train_cfg = TrainConfig(
        max_epochs=args.max_epochs,
        lr=args.lr,
        max_train_seconds=args.max_train_seconds,
        window_steps=args.window_steps,
    )
    config_hash = payload_hash(
        {
            "arm": args.arm,
            "seed": args.seed,
            "model": model_cfg.as_dict(),
            "train": train_cfg.as_dict(),
            "horizons": list(args.horizons),
        }
    )
    if result_path.exists() and not args.overwrite:
        existing = json.loads(result_path.read_text())
        if existing.get("status") == "ok" and existing.get("config_hash") == config_hash:
            print(f"{run_id}: cached")
            return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    started = time.time()
    ctx = load_subject(args.subject, device, horizons=args.horizons)
    model = build_model(
        args.arm,
        model_cfg,
        ctx.tensors.n_drive_features,
        int(ctx.stream.features.count_features.shape[1]),
        int(ctx.stream.features.mark_features.shape[1]),
        args.seed,
        mean_event_rate_hz=ctx.tensors.train_event_rate_hz,
    ).to(device)

    fit = train_arm(model, ctx.tensors, args.horizons, device, train_cfg, seed=args.seed)
    summary, per_block = evaluate(ctx, model, train_cfg, args.horizons, device)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_ckpt = ckpt_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arm": args.arm,
            "seed": args.seed,
            "subject": args.subject,
            "model_config": model_cfg.as_dict(),
            "train_config": train_cfg.as_dict(),
            "config_hash": config_hash,
            "support_hash": ctx.support_hash,
            "mark_loc": ctx.tensors.mark_loc,
            "mark_scale": ctx.tensors.mark_scale,
        },
        tmp_ckpt,
    )
    os.replace(tmp_ckpt, ckpt_path)
    write_npz_atomic(block_path, **per_block)

    payload = {
        "status": "ok",
        "run_id": run_id,
        "subject": args.subject,
        "arm": args.arm,
        "seed": args.seed,
        "tag": args.tag,
        "config_hash": config_hash,
        "support_hash": ctx.support_hash,
        "source_commit": _git_commit(),
        "device": str(device),
        "horizons_minutes": list(args.horizons),
        "uses_background": True,
        "uses_waveform": True,
        "uses_multiband": True,
        "event_update": args.arm != "M0_no_feedback",
        "feedback_model": {
            "M0_no_feedback": "observer_only",
            "M1_count_rate_feedback": "count_rate_edge",
            "M2_mark_specific_feedback": "count_rate_plus_mark_edge",
        }[args.arm],
        "physical_dt": True,
        "train_event_rate_hz": float(ctx.tensors.train_event_rate_hz),
        "context": context_summary(ctx),
        "fit": fit,
        "evaluation": summary,
        "block_file": str(block_path),
        "checkpoint_file": str(ckpt_path),
        "total_seconds": round(time.time() - started, 1),
        "peak_gpu_gib": (
            float(torch.cuda.max_memory_allocated(device)) / (1024**3)
            if device.type == "cuda"
            else 0.0
        ),
    }
    write_json_atomic(payload, result_path)
    dev = summary.get("development_test", {})
    line = " ".join(
        f"{h}m:count={dev.get(str(h), {}).get('mean_count_logscore', float('nan')):.4f}"
        f"/n={dev.get(str(h), {}).get('n_disjoint_blocks', 0)}"
        for h in args.horizons
    )
    print(f"{run_id}: {line} peak={payload['peak_gpu_gib']:.2f}GiB {payload['total_seconds']}s")


if __name__ == "__main__":
    main()
