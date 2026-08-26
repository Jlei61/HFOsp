#!/usr/bin/env python3
"""Isolated state-dimension sensitivity for the six-patient T1 pilot.

This is deliberately not part of the registered state-8 aggregation.  It fits
only the T1 arm at state_dim=16 and reuses the matching exact-zero T0 control
from the active state-8 package.  T0 has no learned or propagated state, so its
prediction is independent of state_dim by construction.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.regular_t1 import (
    REGULAR_T1_REVISION,
    fit_regular_t1,
)


SENSITIVITY_REVISION = "state16_capacity_sensitivity_v1_on_regular_t1_v10"
SUMMARY_REVISION = "state16_capacity_patient_unit_three_seed_v2"
VARIANTS = ("spectral", "raw")
ENDPOINTS = ("joint_nll", "timing_nll", "mark_nll")


def sensitivity_root() -> Path:
    return contract.RESULT_ROOT / "regular_t1/sensitivities/state16"


def active_root(variant: str) -> Path:
    if variant == "spectral":
        return contract.RESULT_ROOT / "regular_t1"
    return contract.RESULT_ROOT / "regular_t1/raw_e0"


def result_path(subject: str, variant: str, seed: int) -> Path:
    return sensitivity_root() / variant / "runs" / f"{subject}__t1__s{seed}.json"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def run_worker(subject: str, variant: str, epochs: int, seed: int) -> dict:
    output = result_path(subject, variant, seed)
    if output.exists():
        old = json.loads(output.read_text())
        if (
            old.get("sensitivity_revision") == SENSITIVITY_REVISION
            and old.get("regular_t1_revision") == REGULAR_T1_REVISION
            and old.get("subject") == subject
            and old.get("observation_variant") == variant
            and int(old.get("seed", -1)) == seed
            and int(old.get("state_dim", -1)) == 16
            and int(old.get("epochs", -1)) == epochs
            and old.get("sealed_opened") is False
        ):
            return {"status": "SKIPPED", "path": str(output)}
        raise ValueError(f"configuration collision at {output}")
    result = fit_regular_t1(
        subject,
        "t1_regular_observation",
        seed=seed,
        epochs=epochs,
        observation_variant=variant,
        state_dim=16,
    )
    result.update({
        "analysis_tier": "post-hoc capacity sensitivity; not a primary test",
        "sensitivity_revision": SENSITIVITY_REVISION,
        "control_contract": (
            "Compared with the active same-seed T0 exact-zero-state control. "
            "T0 predictions are independent of state_dim by construction."
        ),
    })
    atomic_json(output, result)
    return {"status": "DONE", "path": str(output)}


def metric_contrasts(t1: dict, t0: dict) -> dict:
    output: dict[str, dict] = {}
    for layer in (
        "validation_filtered",
        "validation_correction_off_from_split_start",
    ):
        output[layer] = {
            endpoint: float(t1[layer][endpoint] - t0[layer][endpoint])
            for endpoint in ENDPOINTS
        }
    output["post_anchor_correction_off"] = {}
    for horizon in ("5", "10", "20"):
        output["post_anchor_correction_off"][horizon] = {
            endpoint: float(
                t1["post_anchor_correction_off"][horizon][endpoint]
                - t0["post_anchor_correction_off"][horizon][endpoint]
            )
            for endpoint in ENDPOINTS
        }
    output["state_swap"] = {
        endpoint: float(
            t1["matched_wrong_time_state_swap"]["endpoints"][endpoint]
            ["wrong_minus_correct"]
        )
        for endpoint in ENDPOINTS
    }
    return output


def aggregate(epochs: int, seeds: tuple[int, ...]) -> dict:
    rows = []
    for variant in VARIANTS:
        for subject in contract.PILOT_SUBJECTS:
            for seed in seeds:
                t1_16 = json.loads(result_path(subject, variant, seed).read_text())
                root = active_root(variant) / "runs"
                t0 = json.loads(
                    (root / f"{subject}__t0_no_observation_state__s{seed}.json").read_text()
                )
                t1_8 = json.loads(
                    (root / f"{subject}__t1_regular_observation__s{seed}.json").read_text()
                )
                if not (
                    t0.get("sealed_opened") is False
                    and t1_8.get("sealed_opened") is False
                    and t1_16.get("sealed_opened") is False
                    and t0.get("regular_t1_revision") == REGULAR_T1_REVISION
                    and t1_8.get("regular_t1_revision") == REGULAR_T1_REVISION
                    and t1_16.get("sensitivity_revision") == SENSITIVITY_REVISION
                    and int(t1_16.get("seed", -1)) == seed
                ):
                    raise ValueError(
                        f"stale or unsealed sensitivity input: "
                        f"{subject}/{variant}/seed{seed}"
                    )
                c16 = metric_contrasts(t1_16, t0)
                c8 = metric_contrasts(t1_8, t0)
                rows.append({
                    "subject": subject,
                    "observation_variant": variant,
                    "seed": seed,
                    "state8_t1_minus_t0": c8,
                    "state16_t1_minus_t0": c16,
                    "state16_minus_state8_contrast": {
                        "validation_filtered": {
                            endpoint: c16["validation_filtered"][endpoint]
                            - c8["validation_filtered"][endpoint]
                            for endpoint in ENDPOINTS
                        },
                        "validation_correction_off_from_split_start": {
                            endpoint: c16["validation_correction_off_from_split_start"][endpoint]
                            - c8["validation_correction_off_from_split_start"][endpoint]
                            for endpoint in ENDPOINTS
                        },
                        "post_anchor_correction_off": {
                            horizon: {
                                endpoint: c16["post_anchor_correction_off"][horizon][endpoint]
                                - c8["post_anchor_correction_off"][horizon][endpoint]
                                for endpoint in ENDPOINTS
                            }
                            for horizon in ("5", "10", "20")
                        },
                    },
                })

    per_subject = []
    for variant in VARIANTS:
        for subject in contract.PILOT_SUBJECTS:
            found = [
                row for row in rows
                if row["subject"] == subject
                and row["observation_variant"] == variant
            ]
            if sorted(row["seed"] for row in found) != list(seeds):
                raise ValueError(f"incomplete seed set: {subject}/{variant}")
            summary = {
                "subject": subject,
                "observation_variant": variant,
                "n_seeds": len(found),
                "state16_t1_minus_t0": {},
                "state16_minus_state8_contrast": {},
            }
            for field in (
                "state16_t1_minus_t0", "state16_minus_state8_contrast"
            ):
                for layer in (
                    "validation_filtered",
                    "validation_correction_off_from_split_start",
                ):
                    summary[field][layer] = {
                        endpoint: float(np.median([
                            row[field][layer][endpoint] for row in found
                        ]))
                        for endpoint in ENDPOINTS
                    }
                summary[field]["post_anchor_correction_off"] = {
                    horizon: {
                        endpoint: float(np.median([
                            row[field]["post_anchor_correction_off"][horizon][endpoint]
                            for row in found
                        ]))
                        for endpoint in ENDPOINTS
                    }
                    for horizon in ("5", "10", "20")
                }
            per_subject.append(summary)

    cohort = {}
    for variant in VARIANTS:
        found = [
            row for row in per_subject if row["observation_variant"] == variant
        ]
        cohort[variant] = {"n_patients": len(found), "layers": {}}
        for layer in (
            "validation_filtered",
            "validation_correction_off_from_split_start",
        ):
            cohort[variant]["layers"][layer] = {}
            for endpoint in ENDPOINTS:
                values16 = np.asarray([
                    row["state16_t1_minus_t0"][layer][endpoint] for row in found
                ])
                capacity = np.asarray([
                    row["state16_minus_state8_contrast"][layer][endpoint]
                    for row in found
                ])
                cohort[variant]["layers"][layer][endpoint] = {
                    "median_patient_state16_t1_minus_t0": float(np.median(values16)),
                    "n_patients_state16_t1_better": int(np.sum(values16 < 0)),
                    "median_patient_state16_minus_state8_contrast": float(
                        np.median(capacity)
                    ),
                    "n_patients_state16_improves_contrast": int(np.sum(capacity < 0)),
                }
        cohort[variant]["layers"]["post_anchor_correction_off"] = {}
        for horizon in ("5", "10", "20"):
            cohort[variant]["layers"]["post_anchor_correction_off"][horizon] = {}
            for endpoint in ENDPOINTS:
                values16 = np.asarray([
                    row["state16_t1_minus_t0"]["post_anchor_correction_off"]
                    [horizon][endpoint] for row in found
                ])
                capacity = np.asarray([
                    row["state16_minus_state8_contrast"]
                    ["post_anchor_correction_off"][horizon][endpoint]
                    for row in found
                ])
                cohort[variant]["layers"]["post_anchor_correction_off"][horizon][endpoint] = {
                    "median_patient_state16_t1_minus_t0": float(np.median(values16)),
                    "n_patients_state16_t1_better": int(np.sum(values16 < 0)),
                    "median_patient_state16_minus_state8_contrast": float(
                        np.median(capacity)
                    ),
                    "n_patients_state16_improves_contrast": int(np.sum(capacity < 0)),
                }
    payload = {
        "contract": contract.REVISION,
        "regular_t1_revision": REGULAR_T1_REVISION,
        "sensitivity_revision": SENSITIVITY_REVISION,
        "summary_revision": SUMMARY_REVISION,
        "state_dim": 16,
        "seeds": list(seeds),
        "epochs": epochs,
        "n_runs": len(rows),
        "rows": rows,
        "per_subject": per_subject,
        "cohort_patient_unit": cohort,
        "sealed_opened": False,
        "claim_boundary": (
            "Post-hoc three-seed capacity sensitivity. It can identify a gross "
            "state-dimension bottleneck, but it cannot replace the registered "
            "state-8 three-seed analysis or support a new H1 claim."
        ),
    }
    atomic_json(sensitivity_root() / "STATE16_SENSITIVITY_SUMMARY.json", payload)
    return payload


def orchestrate(epochs: int, workers: int, seeds: tuple[int, ...]) -> None:
    jobs = [
        (subject, variant, seed)
        for variant in VARIANTS
        for subject in contract.PILOT_SUBJECTS
        for seed in seeds
    ]
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for subject, variant, seed in jobs:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-subject", subject,
                "--worker-variant", variant,
                "--worker-seed", str(seed),
                "--epochs", str(epochs),
            ]
            future = pool.submit(
                subprocess.run,
                command,
                cwd=contract.REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            futures[future] = (subject, variant, seed)
        for future in as_completed(futures):
            subject, variant, seed = futures[future]
            completed = future.result()
            if completed.returncode:
                failures.append({
                    "subject": subject,
                    "variant": variant,
                    "seed": seed,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                })
    if failures:
        atomic_json(sensitivity_root() / "FAILURES.json", {"failures": failures})
        raise RuntimeError(f"{len(failures)} state16 workers failed")
    summary = aggregate(epochs, seeds)
    print(json.dumps({
        "status": "DONE",
        "n_runs": summary["n_runs"],
        "path": str(sensitivity_root() / "STATE16_SENSITIVITY_SUMMARY.json"),
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-subject", choices=contract.PILOT_SUBJECTS)
    parser.add_argument("--worker-variant", choices=VARIANTS)
    parser.add_argument("--worker-seed", type=int)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()
    worker_values = (
        args.worker_subject, args.worker_variant, args.worker_seed
    )
    if any(value is None for value in worker_values) != all(
        value is None for value in worker_values
    ):
        raise ValueError("worker subject, variant, and seed must be supplied together")
    if args.worker_subject is not None:
        result = run_worker(
            args.worker_subject, args.worker_variant, args.epochs, args.worker_seed
        )
        print(json.dumps(result, sort_keys=True))
        return
    seeds = tuple(sorted({int(value) for value in args.seeds.split(",")}))
    if not seeds:
        raise ValueError("at least one seed is required")
    orchestrate(args.epochs, args.workers, seeds)


if __name__ == "__main__":
    main()
