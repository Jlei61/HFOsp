#!/usr/bin/env python3
"""Re-audit the frozen long total-effect artefacts after the 2026-08-26 review.

The six Zhangjiaqi arms were produced by revision v1, which reported a
hardcoded gap count, no exposure-kernel diagnostic and no independent-window
budget, and which scored the exposure arms against a raw no-edge reference that
they beat through a free intercept.  Re-running them needs the GPU observer, and
the outcome is a structural zero either way, so this script recomputes the
missing diagnostics directly from the frozen design, the saved window indices
and the frozen T1 checkpoint instead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.state import StableGenerator
from src.topic5_continuous_marked_state_r1.t2_long_total import (
    LONG_TOTAL_REVISION,
    count_windows_crossing_segment,
    effective_memory_audit,
    endpoint_support_audit,
)


WINDOWS = ("event_count_10000", "physical_6h")
SEEDS = (0, 1, 2)


def _generator(checkpoint: dict, state_dim: int) -> tuple[np.ndarray, dict]:
    """Frozen K plus how far the state model moved from its initialisation."""
    generator = StableGenerator(state_dim)
    reference = {
        name: value.detach().clone()
        for name, value in generator.state_dict().items()
    }
    for name in reference:
        generator.state_dict()[name].copy_(
            checkpoint[f"state.generator.{name}"].float()
        )
    matrix = generator.matrix().detach().numpy().astype(np.float64)
    moved = {
        f"state.generator.{name}": float(
            (checkpoint[f"state.generator.{name}"].float() - value).abs().max()
        )
        for name, value in reference.items()
    }
    for name in ("state_timing.weight", "state_size.weight",
                 "state_contact.weight"):
        moved[name] = float(checkpoint[name].abs().max())
    return matrix, moved


def _occurrence_block_norms(time: np.ndarray, start: np.ndarray,
                            end: np.ndarray, matrix: np.ndarray,
                            *, max_windows: int = 256) -> np.ndarray:
    """||sum_j exp(K (t_e - t_j))||_F per window, via one eigendecomposition."""
    values, vectors = np.linalg.eig(matrix)
    inverse = np.linalg.inv(vectors)
    take = np.unique(
        np.linspace(0, len(start) - 1, min(int(max_windows), len(start))).astype(int)
    )
    norms = []
    for row in take:
        lo, hi = int(start[row]), int(end[row])
        delta = (time[hi] - time[lo:hi]) / 60.0
        summed = np.exp(np.outer(values, delta)).sum(axis=1)
        block = vectors @ np.diag(summed) @ inverse
        norms.append(float(np.linalg.norm(block.real)))
    return np.asarray(norms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="yuquan_zhangjiaqi")
    parser.add_argument(
        "--root", type=Path,
        default=contract.RESULT_ROOT / "t2_long_total_effect",
    )
    args = parser.parse_args()
    coverage = CoverageTable.load(
        contract.RESULT_ROOT / "r1_2/coverage" / f"{args.subject}.npz"
    )
    design = np.load(
        contract.RESULT_ROOT / "r1_2/cache" / args.subject / "full_design.npz",
        allow_pickle=True,
    )
    time = design["event_time"].astype(np.float64)
    segment = np.searchsorted(coverage.stop, time, side="right").astype(np.int64)
    checkpoint = torch.load(
        args.root / "t1_r1_3/fits" / args.subject / "explicit_seed_0/model.pt",
        map_location="cpu", weights_only=False,
    )["model"]
    matrix, moved = _generator(checkpoint, int(checkpoint["state_timing.weight"].shape[1]))
    payload = {
        "status": "COMPLETE",
        "purpose": (
            "post-review diagnostics for frozen long total-effect artefacts; "
            "recomputed from the design, the saved window indices and the "
            "frozen T1, not from a rerun"
        ),
        "audited_revision": "t2_long_total_effect_decoder_space_v1",
        "current_module_revision": LONG_TOTAL_REVISION,
        "subject": args.subject,
        "t1_state_model": {
            "max_abs_parameter_change_from_initialisation": moved,
            "state_model_entirely_at_initialisation": bool(
                max(moved.values()) == 0.0
            ),
            "note": (
                "the generator, the observation correction and all three "
                "state-to-event readouts are at their constructor defaults, so "
                "'frozen T1 generator' here means an untrained default flow"
            ),
        },
        "recorded_segments": [
            {
                "segment": index,
                "hours": float((stop - start) / 3600.0),
                "events": int(np.sum(segment == index)),
                "train_events": int(np.sum(
                    (segment == index) & (design["event_split"] == 0)
                )),
                "validation_events": int(np.sum(
                    (segment == index) & (design["event_split"] == 1)
                )),
            }
            for index, (start, stop) in enumerate(zip(coverage.start, coverage.stop))
        ],
        "windows": {},
    }
    for window in WINDOWS:
        support = np.load(
            args.root / "human" / args.subject / window
            / f"seed_{SEEDS[0]}/parameters_and_support.npz"
        )
        start = support["start_index"].astype(np.int64)
        end = support["end_index"].astype(np.int64)
        split = support["split"].astype(np.int8)
        validation = np.flatnonzero(split == 1)
        norms = _occurrence_block_norms(time, start[validation], end[validation], matrix)
        payload["windows"][window] = {
            "windows_cross_unrecorded_gap_computed": count_windows_crossing_segment(
                start, end, segment
            ),
            # Restricted to the validation rows so the kernel statistics sit on
            # the same denominator the reports quote.
            "effective_exposure_time_scale": effective_memory_audit(
                time, start[validation], end[validation], matrix
            ),
            "endpoint_support": endpoint_support_audit(time, end, split, matrix),
            "occurrence_block_variation": {
                "sampled_validation_windows": int(len(norms)),
                "norm_mean": float(norms.mean()),
                "norm_sd": float(norms.std()),
                "coefficient_of_variation": float(norms.std() / norms.mean()),
            },
        }
    payload["reporting_rule"] = (
        "the nominal window (10000 events / 6 h) is not the tested time scale; "
        "report the generator time constant and the effective weighted event "
        "count, and report the endpoint span rather than the window count as "
        "the validation denominator"
    )
    payload["sealed_opened"] = False
    payload["formal_test_partition_opened"] = False
    contract.atomic_json(args.root / "reports/post_review_audit.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
