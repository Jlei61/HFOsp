#!/usr/bin/env python3
"""Dedicated positive/null audit for persistent H3 feedback readouts.

The one-step H3 synthetic task is deliberately not reused as a proxy for this
different estimand. Here a known six-hour count or mark impulse bank is
inserted directly into a held-out future background target. A frozen M0 source
is then held fixed while persistent M1/M2 readouts are fitted. This tests the
long-edge instrument without claiming that human data are trainable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v037.contracts import atomic_json
from src.topic5_group_event_state.v037.h3_generative import (
    H3Config, H3Data, PhysiologicalGenerativeState,
)
from src.topic5_group_event_state.v037.h3_persistent import train_persistent_h3_subject


TRUTHS = ("common_drive", "count_feedback", "mark_feedback")
SEEDS = (31, 47, 71)


def _lagged_bank(value: np.ndarray, phi: float) -> np.ndarray:
    result = np.zeros_like(value, dtype=np.float32)
    for row in range(1, value.shape[0]):
        result[row] = phi * result[row - 1] + phi * value[row - 1]
    return result


def _synthetic_data(truth: str, seed: int, n: int = 1100) -> H3Data:
    rng = np.random.default_rng(seed)
    time_axis = np.arange(n, dtype=np.float64) * 300.0
    phase = np.full(n, "SELECTION", dtype="<U10")
    phase[:600] = "FIT"; phase[600:820] = "INNER"
    count_input = np.zeros((n, 1), dtype=np.float32)
    grammar_input = np.zeros((n, 2), dtype=np.float32)
    if truth == "count_feedback":
        count_input[:, 0] = rng.normal(size=n)
    elif truth == "mark_feedback":
        grammar_input[:, 0] = rng.normal(size=n)
        grammar_input[:, 1] = rng.normal(size=n)
    phi = float(np.exp(-300.0 / 21600.0))
    count_memory = _lagged_bank(count_input[:, 0], phi)
    mark_memory = _lagged_bank(grammar_input[:, 0], phi)
    background = np.zeros((n, 2), dtype=np.float32)
    if truth == "count_feedback":
        background[:, 0] = 0.40 * count_memory
        background[:, 1] = -0.25 * count_memory
    elif truth == "mark_feedback":
        background[:, 0] = 0.35 * mark_memory
        background[:, 1] = 0.20 * mark_memory
    return H3Data(
        subject=f"synthetic_persistent_{truth}", time=time_axis,
        segment=np.zeros(n, dtype=np.int64), phase=phase,
        exposure_seconds=np.full(n, 300.0, dtype=np.float32),
        count=np.ones(n, dtype=np.float32), count_input=count_input,
        grammar_input=grammar_input, grammar_target=np.zeros((n, 2), dtype=np.float32),
        grammar_valid=np.ones(n, dtype=bool), background_target=background,
        background_valid=np.ones(n, dtype=bool), context=np.zeros((n, 2), dtype=np.float32),
        context_names=("common_drive_0", "common_drive_1"),
        transforms={"truth": truth, "synthetic_only": True,
                    "implanted_tau_seconds": 21600.0,
                    "implanted_output": "future_background"},
    )


def _write_frozen_source(data: H3Data, seed: int, source_dir: Path) -> None:
    config = H3Config(seed=seed)
    torch.manual_seed(seed)
    model = PhysiologicalGenerativeState(
        data.context.shape[1], data.grammar_input.shape[1], data.grammar_target.shape[1],
        data.background_target.shape[1], config,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    source_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(config), "models": {"M0_common_drive": model.state_dict()}},
               source_dir / "checkpoint.pt")
    atomic_json(source_dir / "card.json", {
        "format": "group_event_state_v0_3_7_h3_persistent_synthetic_source_v1",
        "subject": data.subject, "seed": seed, "synthetic_only": True,
        "biological_evidence": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out-root", type=Path,
        default=Path("/data/hfosp_group_event_state_v0_3_7/h3_persistent_instrument"),
    )
    args = parser.parse_args(); args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for truth in TRUTHS:
        for seed in SEEDS:
            data = _synthetic_data(truth, seed)
            source_dir = args.out_root / "synthetic_source" / truth / f"seed{seed}"
            _write_frozen_source(data, seed, source_dir)
            out = args.out_root / truth / f"seed{seed}"
            card = train_persistent_h3_subject(
                data.subject, seed, source_root=args.out_root, out_dir=out,
                device=torch.device(args.device), data_override=data,
                source_dir_override=source_dir,
            )
            rows.append({"truth": truth, "seed": seed, **card["primary_contrasts"]})
    keys = (
        "persistent_count_over_one_step_count",
        "persistent_count_over_one_step_grammar",
        "persistent_count_over_one_step_background",
        "persistent_mark_over_one_step_count",
        "persistent_mark_over_one_step_grammar",
        "persistent_mark_over_one_step_background",
        "persistent_count_real_over_fitted_wrong_time_count",
        "persistent_count_real_over_fitted_wrong_time_background",
        "persistent_mark_real_over_fitted_wrong_time_grammar",
        "persistent_mark_real_over_fitted_wrong_time_background",
    )
    medians = {
        truth: {key: float(np.median([float(row[key]) for row in rows if row["truth"] == truth]))
                for key in keys}
        for truth in TRUTHS
    }
    checks = {
        "common_drive_count_false_positive_control": abs(medians["common_drive"]["persistent_count_over_one_step_background"]) < 1e-8,
        "common_drive_mark_false_positive_control": abs(medians["common_drive"]["persistent_mark_over_one_step_background"]) < 1e-8,
        "persistent_count_feedback_recovered": medians["count_feedback"]["persistent_count_over_one_step_background"] > 0.05,
        "persistent_mark_feedback_recovered": medians["mark_feedback"]["persistent_mark_over_one_step_background"] > 0.05,
        "count_feedback_beats_fitted_wrong_time_capacity_control": medians["count_feedback"]["persistent_count_real_over_fitted_wrong_time_background"] > 0.05,
        "mark_feedback_beats_fitted_wrong_time_capacity_control": medians["mark_feedback"]["persistent_mark_real_over_fitted_wrong_time_background"] > 0.05,
    }
    payload = {
        "format": "group_event_state_v0_3_7_h3_persistent_instrument_v2",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "by_truth_median": medians, "rows": rows,
        "human_trainability_inferred": False, "biological_feedback_inferred": False,
        "purpose": "dedicated persistent feedback fitter positive and exact-null recovery",
    }
    atomic_json(args.out_root / "audit.json", payload)
    marker = args.out_root / "HUMAN_RELEASED"
    if payload["status"] == "PASS":
        marker.write_text("persistent H3 fitting instrument passed; no human claim implied\n", encoding="utf-8")
    elif marker.exists():
        marker.unlink()
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
