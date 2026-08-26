#!/usr/bin/env python3
"""Calibrate physical-time versus event-count clock contrast on real timelines."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.exposure import exposure_pair


ROOT = contract.RESULT_ROOT
TAU_MINUTES = 5.0
BETAS = (0.0, 0.05, 0.1, 0.2)
NOISE_SEEDS = tuple(range(5))


def _take(index: np.ndarray, maximum: int) -> np.ndarray:
    if len(index) <= maximum:
        return index
    return index[np.linspace(0, len(index) - 1, maximum, dtype=int)]


def _standardize(train: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(train, axis=0)
    scale = np.maximum(np.std(train, axis=0), 1e-6)
    return (train - center) / scale, (valid - center) / scale


def _mse(train_x: np.ndarray, train_y: np.ndarray,
         valid_x: np.ndarray, valid_y: np.ndarray) -> float:
    design_train = np.c_[np.ones(len(train_x)), train_x]
    design_valid = np.c_[np.ones(len(valid_x)), valid_x]
    ridge = np.eye(design_train.shape[1]) * 1e-4
    ridge[0, 0] = 0.0
    coef = np.linalg.solve(design_train.T @ design_train + ridge,
                           design_train.T @ train_y)
    return float(np.mean((valid_y - design_valid @ coef) ** 2))


def _subject(subject: str, payload: dict) -> dict:
    times = payload["event_time"].numpy().astype(np.float64)
    session = payload["session_index"].numpy().astype(np.int64)
    bound = contract.load_split(subject)
    split = np.full(len(times), 2, dtype=np.int8)
    split[times < bound.dev_end_epoch] = 1
    split[times < bound.train_end_epoch] = 0
    adjacent = (
        (split[1:] == split[:-1]) & (split[:-1] < 2)
        & (session[1:] == session[:-1]) & (np.diff(times) > 0)
    )
    train_intervals = np.diff(times)[adjacent & (split[:-1] == 0)] / 60.0
    step = float(np.median(train_intervals))
    innovation_rng = np.random.default_rng(
        int.from_bytes(subject.encode("utf-8"), "little") % (2**32)
    )
    innovation = innovation_rng.normal(size=len(times)).astype(np.float32)
    innovation -= float(np.mean(innovation[split == 0]))
    physical, _, _ = exposure_pair(
        times, innovation, session, split, TAU_MINUTES,
        decay_clock="physical_time", event_count_step_minutes=step,
    )
    count, _, _ = exposure_pair(
        times, innovation, session, split, TAU_MINUTES,
        decay_clock="event_count", event_count_step_minutes=step,
    )
    pair_index = np.flatnonzero(adjacent)
    train_index = _take(pair_index[split[pair_index] == 0], 8000)
    valid_index = _take(pair_index[split[pair_index] == 1], 4000)
    if len(train_index) < 50 or len(valid_index) < 50:
        raise RuntimeError(f"{subject}: insufficient clock-calibration pairs")

    previous_interval = np.r_[step * 60.0, np.maximum(np.diff(times), 1e-3)]
    baseline = np.c_[innovation, np.log(previous_interval),
                     np.log1p(np.arange(len(times)) % 20)]
    base_train, base_valid = _standardize(
        baseline[train_index], baseline[valid_index]
    )
    physical_train, physical_valid = _standardize(
        physical[train_index, None], physical[valid_index, None]
    )
    count_train, count_valid = _standardize(
        count[train_index, None], count[valid_index, None]
    )
    exposure_corr = float(np.corrcoef(
        physical[valid_index], count[valid_index]
    )[0, 1])

    rows = []
    for truth, true_train, true_valid in (
        ("physical_time", physical_train[:, 0], physical_valid[:, 0]),
        ("event_count", count_train[:, 0], count_valid[:, 0]),
    ):
        for beta in BETAS:
            deltas = []
            for noise_seed in NOISE_SEEDS:
                rng = np.random.default_rng(
                    noise_seed + 1009 * (int.from_bytes(
                        subject.encode("utf-8"), "little"
                    ) % 100000)
                )
                noise_train = rng.normal(size=len(train_index))
                noise_valid = rng.normal(size=len(valid_index))
                target_train = beta * true_train + noise_train
                target_valid = beta * true_valid + noise_valid
                physical_mse = _mse(
                    np.c_[base_train, physical_train], target_train,
                    np.c_[base_valid, physical_valid], target_valid,
                )
                count_mse = _mse(
                    np.c_[base_train, count_train], target_train,
                    np.c_[base_valid, count_valid], target_valid,
                )
                deltas.append(physical_mse - count_mse)
            rows.append({
                "truth": truth,
                "beta": beta,
                "median_physical_minus_event_count_mse": float(np.median(deltas)),
                "noise_seed_values": deltas,
            })
    return {
        "subject": subject,
        "dataset": str(payload["dataset"]),
        "n_train": int(len(train_index)),
        "n_validation": int(len(valid_index)),
        "train_median_iei_minutes": step,
        "nominal_memory_events": TAU_MINUTES / step,
        "validation_exposure_correlation": exposure_corr,
        "rows": rows,
    }


def _aggregate(subjects: list[dict]) -> list[dict]:
    output = []
    for truth in ("physical_time", "event_count"):
        for beta in BETAS:
            values = np.asarray([
                next(row["median_physical_minus_event_count_mse"]
                     for row in subject["rows"]
                     if row["truth"] == truth and row["beta"] == beta)
                for subject in subjects
            ])
            correct = values < 0 if truth == "physical_time" else values > 0
            output.append({
                "truth": truth,
                "beta": beta,
                "median_physical_minus_event_count_mse": float(np.median(values)),
                "n_correct_clock": int(np.sum(correct)),
                "n_patients": int(len(values)),
                "leave_one_patient_median_range": [
                    float(min(np.median(np.delete(values, i)) for i in range(len(values)))),
                    float(max(np.median(np.delete(values, i)) for i in range(len(values)))),
                ],
            })
    return output


def main() -> None:
    payloads = torch.load(
        contract.COHORT_CACHE, map_location="cpu", weights_only=False
    )
    subjects_manifest = json.loads(contract.SPLIT_MANIFEST.read_text())["subjects"]
    subjects = [_subject(subject, payloads[subject])
                for subject in subjects_manifest]
    correlations = np.asarray([
        subject["validation_exposure_correlation"] for subject in subjects
    ])
    output = {
        "contract": contract.REVISION,
        "analysis_revision": "real_timeline_clock_truth_recovery_v1",
        "tau_minutes": TAU_MINUTES,
        "betas": list(BETAS),
        "noise_seeds": list(NOISE_SEEDS),
        "n_patients": len(subjects),
        "median_physical_event_count_exposure_correlation": float(
            np.median(correlations)
        ),
        "exposure_correlation_iqr": [
            float(np.percentile(correlations, 25)),
            float(np.percentile(correlations, 75)),
        ],
        "aggregate": _aggregate(subjects),
        "per_subject": subjects,
        "sealed_opened": False,
        "interpretation": (
            "The calibration preserves each patient's actual irregular event times. "
            "Negative physical-minus-count MSE favours the physical clock. Recovery "
            "must reverse sign when the synthetic truth changes from physical time to "
            "event count; beta=0 measures the null floor."
        ),
    }
    path = ROOT / "exposure_clock_control/CLOCK_IDENTIFIABILITY_SYNTHETIC.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True))
    os.replace(temporary, path)
    print(json.dumps({"path": str(path), "n_patients": len(subjects)}))


if __name__ == "__main__":
    main()
