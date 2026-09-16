#!/usr/bin/env python3
"""Positive and null recovery checks for the independent H3 model family.

Synthetic recovery validates the instrument and false-positive controls only;
it is never reported as evidence that human data are trainable or that H3 is
true in people.
"""

from __future__ import annotations

import argparse
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
    H3Config, H3Data, _fit_pca, _fit_scale, _residualise, train_h3_subject,
)


def synthetic_data(truth: str, seed: int, n: int = 720) -> H3Data:
    if truth not in {"common_drive", "count_feedback", "mark_feedback"}:
        raise ValueError(truth)
    rng = np.random.default_rng(seed)
    time_axis = np.arange(n, dtype=np.float64) * 300.0
    angle = 2.0 * np.pi * time_axis / 86400.0
    base_context = np.column_stack((np.sin(angle), np.cos(angle), rng.normal(size=n)))
    phase = np.full(n, "SELECTION", dtype="<U10")
    phase[:400] = "FIT"; phase[400:560] = "INNER"
    fit = np.flatnonzero(phase == "FIT")
    state = np.zeros(n, dtype=np.float64)
    count = np.zeros(n, dtype=np.float32)
    grammar_raw = np.zeros((n, 4), dtype=np.float64)
    load = np.asarray([1.0, -0.7, 0.5, 0.25])
    for t in range(n):
        rate = np.exp(0.35 + 0.35 * base_context[t, 0] + 0.35 * state[t])
        count[t] = rng.poisson(np.clip(rate, 0.05, 12.0))
        innovation = rng.normal(scale=0.7, size=4)
        grammar_raw[t] = load * state[t] + innovation
        if t + 1 < n:
            next_state = 0.985 * state[t] + 0.10 * base_context[t, 0] + rng.normal(scale=0.05)
            if truth in {"count_feedback", "mark_feedback"}:
                # A deliberately detectable positive control.  The synthetic
                # gate validates recovery, not the minimum human effect size.
                next_state += 0.60 * (count[t] - rate)
            if truth == "mark_feedback":
                next_state += 0.55 * innovation[0]
            state[t + 1] = np.clip(next_state, -4.0, 4.0)
    exposure = np.full(n, 300.0, dtype=np.float32)
    missing = rng.random(n) < 0.08
    exposure[missing] = 0.0
    grammar_valid = (~missing) & (count > 0)
    background_raw = np.column_stack((
        state + rng.normal(scale=0.12, size=n),
        0.6 * state + rng.normal(scale=0.12, size=n),
        base_context[:, 0] + rng.normal(scale=0.12, size=n),
    ))
    background, background_centre, background_scale = _fit_scale(background_raw, fit[~missing[fit]])
    context_raw = np.concatenate((base_context, background), axis=1)
    context, context_centre, context_scale = _fit_scale(context_raw, fit)
    count_rate = np.log1p(count)[:, None]
    count_scaled, count_centre, count_scale = _fit_scale(count_rate, fit[~missing[fit]])
    count_input, count_beta = _residualise(count_scaled, context, fit[~missing[fit]])
    count_input[missing] = 0.0
    grammar, grammar_centre, grammar_scale = _fit_scale(grammar_raw, fit[grammar_valid[fit]])
    grammar_residual, grammar_beta = _residualise(
        grammar, np.concatenate((context, count_scaled), axis=1), fit[grammar_valid[fit]],
    )
    grammar_input, grammar_components = _fit_pca(grammar_residual, fit[grammar_valid[fit]], rank=4)
    grammar[~grammar_valid] = 0.0
    grammar_input[~grammar_valid] = 0.0
    return H3Data(
        subject=f"synthetic_{truth}", time=time_axis, segment=np.zeros(n, dtype=np.int64),
        phase=phase, exposure_seconds=exposure, count=count, count_input=count_input,
        grammar_input=grammar_input, grammar_target=grammar.copy(), grammar_valid=grammar_valid,
        background_target=background, background_valid=~missing,
        context=context, context_names=("circadian_sin", "circadian_cos", "common_drive_noise",
                                        "background_0", "background_1", "background_2"),
        transforms={
            "truth": truth, "synthetic_only": True,
            "context_centre": context_centre.tolist(), "context_scale": context_scale.tolist(),
            "count_centre": count_centre.tolist(), "count_scale": count_scale.tolist(),
            "grammar_centre": grammar_centre.tolist(), "grammar_scale": grammar_scale.tolist(),
            "background_centre": background_centre.tolist(), "background_scale": background_scale.tolist(),
            "count_beta": count_beta.tolist(), "grammar_beta": grammar_beta.tolist(),
            "grammar_components": grammar_components.tolist(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h3_instrument"))
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args(); args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for truth in ("common_drive", "count_feedback", "mark_feedback"):
        for seed in (31, 47, 71):
            config = H3Config(
                seed=seed, max_steps=args.steps, validate_every=20,
                patience_checks=12, learning_rate=2e-3,
            )
            out = args.out_root / truth / f"seed{seed}"
            card = train_h3_subject(synthetic_data(truth, seed), config,
                                    device=torch.device(args.device), out_dir=out)
            rows.append({"truth": truth, "seed": seed, **card["primary_contrasts"]})
    by_truth = {}
    for truth in ("common_drive", "count_feedback", "mark_feedback"):
        subset = [row for row in rows if row["truth"] == truth]
        by_truth[truth] = {
            key: float(np.median([float(row[key]) for row in subset]))
            for key in (
                "count_feedback_gain_M1_over_M0", "mark_feedback_gain_M2_over_M1",
                "count_feedback_gain_on_future_background", "mark_feedback_gain_on_future_background",
                "joint_supportive_gain_M2_over_M0",
            )
        }
    checks = {
        "common_drive_count_false_positive_control": by_truth["common_drive"]["count_feedback_gain_on_future_background"] < 0.02,
        "common_drive_mark_false_positive_control": by_truth["common_drive"]["mark_feedback_gain_on_future_background"] < 0.02,
        "count_feedback_recovered": by_truth["count_feedback"]["count_feedback_gain_on_future_background"] > 0.01,
        "mark_feedback_recovered": by_truth["mark_feedback"]["mark_feedback_gain_on_future_background"] > 0.01,
    }
    payload = {
        "format": "group_event_state_v0_3_7_h3_instrument_audit_v3",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "by_truth_median": by_truth, "rows": rows,
        "human_trainability_inferred": False,
        "purpose": "implementation recovery and false-positive control only",
    }
    atomic_json(args.out_root / "audit.json", payload)
    if payload["status"] == "PASS":
        (args.out_root / "HUMAN_RELEASED").write_text(
            "H3 implementation recovered count and mark feedback and passed common-drive null controls.\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
