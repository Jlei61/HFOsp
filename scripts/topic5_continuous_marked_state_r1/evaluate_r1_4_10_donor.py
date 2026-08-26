#!/usr/bin/env python3
"""Same-checkpoint 10-donor sensitivity for the R1.4 explicit primary arm."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.r1_2 import evaluate_full_t1
from src.topic5_continuous_marked_state_r1.r1_2b_diagnostics import (
    evaluate_mark_endpoints,
    median_metric_dict,
    metric_contrast,
    strict_matched_wrong_time_permutations,
)
from src.topic5_continuous_marked_state_r1.t2_r2_human import (
    R1_4_REVISION,
    load_fitted_r1_4_explicit_t1,
)


SUBJECTS = (
    "epilepsiae_620", "epilepsiae_958", "yuquan_huanghanwen",
    "epilepsiae_922", "yuquan_pengzihang", "yuquan_hanyuxuan",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, choices=SUBJECTS)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--root", type=Path, default=contract.RESULT_ROOT / "r1_4"
    )
    args = parser.parse_args()
    context = load_fitted_r1_4_explicit_t1(
        args.subject, args.seed, device=args.device, r1_4_root=args.root
    )
    if context.anchor_embedding is None:
        raise RuntimeError("R1.4 context lacks anchor embeddings")
    result_path = (
        args.root / "fits" / args.subject
        / f"explicit_seed_{args.seed}" / "result.json"
    )
    source_result = json.loads(result_path.read_text())
    cache = json.loads(Path(source_result["observation_cache_manifest"]).read_text())
    coverage = np.load(Path(cache["contact_mask"]), mmap_mode="r").mean(1)
    design = context.design
    table = context.coverage
    segment = np.searchsorted(
        table.stop, np.asarray(design.anchor_time, dtype=np.float64), side="right"
    )
    if np.any(segment >= len(table.start)):
        raise ValueError("anchor after final recorded segment")
    inside = (
        (design.anchor_time >= table.start[segment])
        & (design.anchor_time < table.stop[segment])
    )
    if not bool(inside.all()):
        raise ValueError("anchor outside recorded segment")
    permutations, matched, audit = strict_matched_wrong_time_permutations(
        design, coverage, anchor_segment=segment, n_donors=10,
        min_separation_seconds=1800.0,
    )
    correct = asdict(evaluate_full_t1(
        context.model, design, context.anchor_embedding, "validation",
        device=args.device, matched_anchor_mask=matched,
    ))
    endpoint_correct = asdict(evaluate_mark_endpoints(
        context.model, design, context.anchor_embedding, device=args.device,
        matched_anchor_mask=matched,
    ))
    wrong = [asdict(evaluate_full_t1(
        context.model, design, context.anchor_embedding, "validation",
        device=args.device, state_permutation=permutation,
        matched_anchor_mask=matched,
    )) for permutation in permutations]
    endpoint_wrong = [asdict(evaluate_mark_endpoints(
        context.model, design, context.anchor_embedding, device=args.device,
        state_permutation=permutation, matched_anchor_mask=matched,
    )) for permutation in permutations]
    wrong_median = median_metric_dict(wrong)
    endpoint_wrong_median = median_metric_dict(endpoint_wrong)
    output = (
        args.root / "sensitivity_10_donor" / args.subject
        / f"explicit_seed_{args.seed}.json"
    )
    value = {
        "status": "COMPLETE", "revision": R1_4_REVISION,
        "subject": args.subject, "seed": int(args.seed), "arm": "explicit",
        "source_checkpoint": source_result["checkpoint"],
        "source_checkpoint_sha256": source_result["checkpoint_sha256"],
        "same_checkpoint_as_primary_5_donor": True,
        "audit": audit, "correct": correct, "wrong_median": wrong_median,
        "correct_minus_wrong_median": metric_contrast(correct, wrong_median),
        "endpoint_correct": endpoint_correct,
        "endpoint_wrong_median": endpoint_wrong_median,
        "endpoint_correct_minus_wrong_median": metric_contrast(
            endpoint_correct, endpoint_wrong_median
        ),
        "formal_test_partition_opened": False, "sealed_opened": False,
    }
    contract.atomic_json(output, value)
    print(json.dumps({
        "status": value["status"], "subject": args.subject,
        "seed": args.seed,
        "correct_minus_wrong_joint": value["correct_minus_wrong_median"][
            "joint_nll_per_event"
        ], "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
