#!/usr/bin/env python3
"""Audit CPU/GPU equivalence for cached LBSS attenuation evaluation.

CPU sidecars only accelerate target-free attenuation units.  This audit loads
the same frozen checkpoints on CPU and GPU, applies the same active-edge
attenuation, and verifies that held-out metrics and deterministic rollouts are
unchanged to a prespecified numerical tolerance.  It never reads ictal targets
and never writes model fields.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_topic5_lbss_attenuation_v0_2 import evaluate_variant  # noqa: E402
from src.topic5_lbss_analysis_v0_2 import attenuate_mask, instantiate_lbss  # noqa: E402


def evaluate_on_device(
    out: Path, metrics_path: Path, device: torch.device, alpha: float
) -> tuple[dict, list[dict]]:
    model, decoder, metrics, plane, events, provenance = instantiate_lbss(
        out, metrics_path, device
    )
    provenance = dict(provenance)
    thresholds = metrics["distance_thresholds_mm"]
    provenance["distance_thresholds_mm"] = (
        float(thresholds["q50"]), float(thresholds["q80"])
    )
    mask = model.added_mask.detach().cpu().numpy().astype(bool)
    attenuate_mask(model, mask, alpha)
    return evaluate_variant(model, decoder, events, provenance, plane, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root", type=Path,
        default=Path("results/topic5_lbss_full_tissue_rnn_v0_3"),
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--n-units", type=int, default=3)
    parser.add_argument(
        "--fit-ids", nargs="*", default=None,
        help="Optional explicit frozen fits for a bounded audit.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CPU/GPU equivalence audit")
    out = args.out_root.resolve()
    paths = sorted(
        (out / "per_fit").glob("*/L3_LOCAL_PLUS_LEARNED_LR/seed0/metrics.json")
    )
    if args.fit_ids:
        requested = set(args.fit_ids)
        paths = [path for path in paths if path.parents[2].name in requested]
        if {path.parents[2].name for path in paths} != requested:
            raise RuntimeError("one or more requested fit ids were not found")
        selected = paths
    elif len(paths) < args.n_units:
        raise RuntimeError(f"requested {args.n_units} units, found {len(paths)}")
    else:
        # Deterministic spread across the sorted cohort rather than a favourable case.
        take = np.linspace(0, len(paths) - 1, args.n_units, dtype=int)
        selected = [paths[index] for index in take]
    rows = []
    all_pass = True
    for metrics_path in selected:
        cpu_metrics, cpu_records = evaluate_on_device(
            out, metrics_path, torch.device("cpu"), args.alpha
        )
        gpu_metrics, gpu_records = evaluate_on_device(
            out, metrics_path, torch.device("cuda:0"), args.alpha
        )
        finite_differences = {
            key: abs(float(cpu_metrics[key]) - float(gpu_metrics[key]))
            for key in cpu_metrics
            if np.isfinite(cpu_metrics[key]) and np.isfinite(gpu_metrics[key])
        }
        nonfinite_match = all(
            bool(np.isfinite(cpu_metrics[key])) == bool(np.isfinite(gpu_metrics[key]))
            for key in cpu_metrics
        )
        cpu_sequences = [record["generated_rank_sets"] for record in cpu_records]
        gpu_sequences = [record["generated_rank_sets"] for record in gpu_records]
        rollout_exact = cpu_sequences == gpu_sequences
        max_abs = max(finite_differences.values(), default=0.0)
        passed = bool(nonfinite_match and rollout_exact and max_abs <= args.atol)
        all_pass &= passed
        metadata = json.loads(metrics_path.read_text())
        rows.append({
            "fit_id": metadata["fit_id"],
            "subject": metadata["subject"],
            "seed": int(metadata["seed"]),
            "alpha": args.alpha,
            "max_abs_metric_difference": max_abs,
            "metric_differences": finite_differences,
            "nonfinite_pattern_match": nonfinite_match,
            "rollout_sequences_exact": rollout_exact,
            "n_test_rollouts": len(cpu_records),
            "pass": passed,
        })
    payload = {
        "contract": "topic5_lbss_attenuation_device_equivalence_v0_3",
        "purpose": "validate that CPU sidecars and the formal GPU runner share one numerical evaluation contract",
        "n_units": len(rows),
        "alpha": args.alpha,
        "absolute_tolerance": args.atol,
        "all_units_pass": all_pass,
        "rows": rows,
        "target_values_read": False,
    }
    destination = out / "ATTENUATION_DEVICE_EQUIVALENCE_AUDIT.json"
    destination.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
