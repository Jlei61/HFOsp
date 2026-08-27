#!/usr/bin/env python3
"""Re-score frozen R1.7A T2 cells on independent within-segment event blocks.

The frozen T2-R2.0 report contains only a next-event average over the whole
D_mechanism layer.  Because the exposure kernel is exponential with rho =
exp(-1/100), consecutive rows share about 100 events of history, so that average
is not an independent-sample summary: `epilepsiae_1125` averages 2904 rows whose
independent 100-event block count is 33.  The previous round (R1.6 minimal H3)
used independent-block medians as a criterion; the R1.7 implementation does not
compute them, so no R1.7 T2 effect can be believed on the average alone.

This script adds that missing view **without refitting anything**: it reloads the
frozen T1 checkpoint, rebuilds the identical four-arm design, loads the already
fitted edges from the cell's `edges.pt`, and evaluates each arm per complete
non-overlapping within-segment block.  Nothing about the frozen fit changes, and
results are written to a separate root so the committed T2 artifacts stay intact.

This is exploratory instrument development, not a new pre-registered analysis.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.r1_7_t2 import (
    R1_7_T2_REVISION, build_r1_7a_r2_designs, load_fitted_r1_7a_t1,
)
from src.topic5_continuous_marked_state_r1.t2_r2 import (
    ExposureEdge, evaluate_r2_edge_by_block,
)


RESCORE_REVISION = "r1_7_t2_independent_block_rescore_v1"
ARMS = ("no_edge", "real_cumulative", "state_matched_placebo", "current_event_only")
CONTROLS = ("no_edge", "state_matched_placebo", "current_event_only")


def block_bootstrap(values: np.ndarray, *, draws: int = 2000, seed: int = 1701) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {"estimate": float(values.mean()) if len(values) else None,
                "ci95": [None, None], "n_blocks": int(len(values))}
    rng = np.random.default_rng(seed)
    sampled = [float(np.mean(values[rng.integers(0, len(values), len(values))]))
               for _ in range(draws)]
    return {"estimate": float(values.mean()),
            "ci95": np.quantile(sampled, [.025, .975]).tolist(),
            "n_blocks": int(len(values)), "draws": draws, "seed": seed,
            "median": float(np.median(values)),
            "fraction_favourable": float(np.mean(values < 0))}


def rescore_cell(root: Path, output: Path, subject: str, seed: int, source: str,
                 *, device: str, batch_size: int, block_events: int) -> dict:
    frozen_path = root / "t2_r2" / subject / f"{source}_seed_{seed}_n_100/result.json"
    frozen = json.loads(frozen_path.read_text())
    if frozen.get("revision") != R1_7_T2_REVISION:
        raise ValueError(f"unexpected frozen T2 revision: {frozen_path}")
    context = load_fitted_r1_7a_t1(subject, seed, device=device, root=root)
    one_step, _, design_audit = build_r1_7a_r2_designs(context, source=source)
    payload = torch.load(Path(frozen["checkpoint"]), map_location=device,
                         weights_only=False)
    if (payload.get("subject") != subject or payload.get("seed") != seed
            or payload.get("source") != source):
        raise ValueError("edge checkpoint payload mismatch")
    reference = one_step["real_cumulative"]
    exposure_dim = 1 if reference.exposure.ndim == 1 else reference.exposure.shape[1]
    per_arm = {}
    for arm in ARMS:
        edge = ExposureEdge(reference.current_state.shape[1], exposure_dim).to(device)
        edge.load_state_dict(payload["edges"][arm]); edge.eval()
        per_arm[arm] = evaluate_r2_edge_by_block(
            context.model, edge, one_step[arm], split="validation",
            event_segment=context.event_segment, device=device,
            batch_size=batch_size, block_events=block_events,
        )
    counts = {arm: len(value) for arm, value in per_arm.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"arms produced different block counts: {counts}")
    keys = [(b["segment"], b["block"]) for b in per_arm["real_cumulative"]]
    for arm in ARMS:
        if [(b["segment"], b["block"]) for b in per_arm[arm]] != keys:
            raise ValueError("arms disagree on block identity")
    contrasts = {}
    for control in CONTROLS:
        delta = np.asarray([
            r["joint_nll_per_event"] - c["joint_nll_per_event"]
            for r, c in zip(per_arm["real_cumulative"], per_arm[control])
        ], dtype=np.float64)
        contrasts[f"real_minus_{control}"] = {
            "per_block": delta.tolist(), "bootstrap": block_bootstrap(delta),
        }
    result = {
        "status": "COMPLETE", "revision": RESCORE_REVISION,
        "frozen_t2_result": str(frozen_path),
        "frozen_t2_result_sha256": contract.sha256_file(frozen_path),
        "frozen_t2_revision": frozen["revision"],
        "subject": subject, "seed": seed, "source": source,
        "scale_events": frozen["scale_events"],
        "block_events": block_events,
        "n_blocks": len(keys),
        "refitted": False,
        "next_event_average_from_frozen_cell": {
            f"real_minus_{c}": frozen["comparisons"]["next_event"][f"real_minus_{c}"]["joint_nll_per_event"]
            for c in CONTROLS
        },
        "independent_block_contrasts": contrasts,
        "real_edge_estimable": frozen["real_edge_estimable"],
        "primary_next_event_increment": frozen["primary_next_event_increment"],
        "exploratory_instrument_development_not_preregistered": True,
        "formal_test_partition_opened": False, "sealed_opened": False,
    }
    contract.atomic_json(
        output / subject / f"{source}_seed_{seed}_n_100/result.json", result
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=contract.RESULT_ROOT / "r1_7a")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--block-events", type=int, default=100)
    args = parser.parse_args()
    output = args.output_root or (args.root / "t2_r2_block_rescore")
    summary = json.loads((args.root / "reports/r1_7a_summary.json").read_text())
    rows = []
    for subject in summary["t2_run_subjects"]:
        for seed in range(5):
            fit = json.loads(
                (args.root / "fits" / subject / f"seed_{seed}/result.json").read_text()
            )
            if not fit["stable_checkpoint"]:
                continue
            for source in ("load", "participation"):
                rows.append(rescore_cell(
                    args.root, output, subject, seed, source,
                    device=args.device, batch_size=args.batch_size,
                    block_events=args.block_events,
                ))
                print(f"rescored {subject}/{source}/seed_{seed}: "
                      f"{rows[-1]['n_blocks']} blocks", flush=True)
    status = {
        "status": "COMPLETE", "revision": RESCORE_REVISION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_cells": len(rows), "refitted": False,
        "exploratory_instrument_development_not_preregistered": True,
        "formal_test_partition_opened": False, "sealed_opened": False,
    }
    contract.atomic_json(output / "RESCORE_STATUS.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
