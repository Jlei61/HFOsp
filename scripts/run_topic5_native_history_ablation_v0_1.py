#!/usr/bin/env python3
"""Matched-history ablation on the model-native readout, without retraining.

The model-native rollout re-introduces three things at once relative to the
composite generator: ordered within-event history, the model's own contact
bias, and the model's own STOP head.  Its whole-event improvement therefore
cannot be attributed to ordered history on its own.

This script holds the checkpoint, the revealed first rank set, the random
numbers, the decoder, the STOP head and the candidate masking fixed, and varies
only what the recurrent state is fed:

* ``ordered``  — the contact the model just emitted (reproduces Phase D);
* ``frozen``   — nothing after the revealed source, so no within-event history;
* ``shuffled`` — a uniformly drawn eligible contact, preserving progress and
  token type while destroying identity correspondence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from scripts.train_topic5_interictal_rank_distribution import load_records  # noqa: E402
from scripts.run_topic5_training_sufficiency_loso_v0_1 import (  # noqa: E402
    _paired_uniform_seed,
    _stop_curve_mae,
)
from src.topic5_constructive_event_generator import (  # noqa: E402
    event_length_wasserstein,
    remove_revealed_source,
)
from src.topic5_constructive_readback import transition_errors  # noqa: E402
from src.topic5_rank_distribution import (  # noqa: E402
    LinearStateSequenceRNN,
    distribution_errors,
)
from src.topic5_training_sufficiency import (  # noqa: E402
    paired_native_rollout,
    run_environment,
)

HISTORY_MODES = ("ordered", "frozen", "shuffled")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=ROOT / "results/topic5_rnn_training_sufficiency_v0_1/formal",
    )
    parser.add_argument("--condition", default="converged_teacher_forced")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "results/topic5_rnn_training_sufficiency_v0_1/analysis",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()

    started = time.time()
    formal_root = (
        args.formal_root if args.formal_root.is_absolute() else ROOT / args.formal_root
    )
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(args.cpu_threads))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    records = load_records(
        args.dataset_root if args.dataset_root.is_absolute() else ROOT / args.dataset_root
    )
    cells = sorted((formal_root / args.condition).glob("seed_*/*"))
    if not cells:
        raise RuntimeError(f"no cells under {formal_root / args.condition}")

    rows = []
    parity = []
    for index, cell in enumerate(cells, start=1):
        if not (cell / "DONE.json").is_file():
            raise RuntimeError(f"incomplete cell: {cell}")
        subject = cell.name
        seed = int(cell.parent.name.split("_")[1])
        record = records[subject]
        checkpoint = torch.load(
            cell / "checkpoint.pt", map_location="cpu", weights_only=False
        )
        model = LinearStateSequenceRNN(**checkpoint["model_kwargs"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        offsets = checkpoint["heldout_local_offsets"]
        offset = offsets[max(offsets)].to(device)

        eval_indices = np.asarray(record.eval_indices, dtype=int)
        observed = np.asarray(record.group_ids[eval_indices], dtype=np.int16)
        observed_count = np.asarray(record.group_count[eval_indices], dtype=np.int16)
        source = observed == 0
        n_events, n_contacts = observed.shape
        uniforms = np.random.default_rng(
            _paired_uniform_seed(subject, seed)
        ).random((n_events, n_contacts), dtype=np.float64)
        suffix_observed = remove_revealed_source(observed, source)
        suffix_observed_count = np.maximum(observed_count - 1, 0)

        features = torch.as_tensor(
            record.contact_features, dtype=torch.float32, device=device
        ).unsqueeze(0)
        contact_mask = torch.ones((1, n_contacts), dtype=torch.bool, device=device)
        history_seed = int(
            hashlib.sha256(f"{subject}:{seed}:history".encode()).hexdigest()[:8], 16
        )

        for mode in HISTORY_MODES:
            generated, generated_count = paired_native_rollout(
                model,
                features,
                contact_mask,
                offset,
                source,
                uniforms,
                history_mode=mode,
                history_seed=history_seed,
            )
            suffix = distribution_errors(
                remove_revealed_source(generated, source),
                np.maximum(generated_count - 1, 0),
                suffix_observed,
                suffix_observed_count,
                bins=10,
            )
            rows.append(
                {
                    "subject": subject,
                    "dataset": record.dataset,
                    "seed": seed,
                    "condition": args.condition,
                    "history_mode": mode,
                    "n_events": int(n_events),
                    "n_contacts": int(n_contacts),
                    **{f"suffix_{key}": value for key, value in suffix.items()},
                    **transition_errors(observed, generated),
                    "event_length_wasserstein": event_length_wasserstein(
                        generated_count, observed_count
                    ),
                    "stop_hazard_mae": _stop_curve_mae(
                        generated_count, observed_count, n_contacts=n_contacts
                    ),
                }
            )
            if mode == "ordered":
                # the ordered arm must reproduce the Phase D native rollout
                with np.load(cell / "rollouts.npz", allow_pickle=False) as archive:
                    reference = archive["native_model__event_group_ids"]
                    reference_count = archive["native_model__event_group_count"]
                parity.append(
                    {
                        "subject": subject,
                        "seed": seed,
                        "group_ids_identical": bool(
                            np.array_equal(generated, reference)
                        ),
                        "group_count_identical": bool(
                            np.array_equal(generated_count, reference_count)
                        ),
                    }
                )
        if index % 20 == 0:
            print(
                json.dumps(
                    {"cells_done": index, "of": len(cells),
                     "elapsed_seconds": round(time.time() - started, 1)}
                ),
                flush=True,
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "d_native_history_ablation_cells.csv", index=False)
    parity_frame = pd.DataFrame(parity)

    from scipy import stats

    endpoints = {
        "transition_correlation": True,
        "suffix_rank_wasserstein": False,
        "suffix_precedence_correlation": True,
        "suffix_participation_mae": False,
        "event_length_wasserstein": False,
        "stop_hazard_mae": False,
    }
    patient = (
        frame.groupby(["history_mode", "subject", "dataset"])[list(endpoints)]
        .mean()
        .reset_index()
    )
    patient.to_csv(out / "d_native_history_ablation_patients.csv", index=False)

    tests = {}
    rng = np.random.default_rng(20260801)
    for baseline in ("frozen", "shuffled"):
        block = {}
        for endpoint, higher in endpoints.items():
            wide = patient.pivot_table(
                index="subject", columns="history_mode", values=endpoint
            )[["ordered", baseline]].dropna()
            delta = wide["ordered"] - wide[baseline]
            if not higher:
                delta = -delta
            values = delta.to_numpy(float)
            draws = np.median(
                rng.choice(values, size=(5000, values.size), replace=True), axis=1
            )
            nonzero = values[values != 0]
            block[endpoint] = {
                "n_patients": int(values.size),
                "median_gain": float(np.median(values)),
                "bootstrap_ci_median_gain": [
                    float(np.quantile(draws, 0.025)),
                    float(np.quantile(draws, 0.975)),
                ],
                "n_improved": int(np.sum(values > 0)),
                "wilcoxon_p": (
                    float(stats.wilcoxon(nonzero).pvalue) if nonzero.size else 1.0
                ),
            }
        tests[f"ordered__vs__{baseline}"] = block

    payload = {
        "contract": "topic5_rnn_training_sufficiency_v0_1_native_history_ablation",
        "question": (
            "does the model-native whole-event improvement come from ordered "
            "within-event history, or from re-introducing the model's own "
            "contact bias and STOP head?"
        ),
        "condition": args.condition,
        "history_modes": list(HISTORY_MODES),
        "held_fixed": [
            "checkpoint", "revealed first rank set", "uniform random numbers",
            "decoder", "STOP head", "candidate masking",
        ],
        "n_cells": int(len(cells)),
        "n_patients": int(patient.subject.nunique()),
        "ordered_arm_reproduces_phase_d": {
            "n_cells": int(len(parity_frame)),
            "group_ids_identical": int(parity_frame.group_ids_identical.sum()),
            "group_count_identical": int(parity_frame.group_count_identical.sum()),
            "all_identical": bool(
                parity_frame.group_ids_identical.all()
                and parity_frame.group_count_identical.all()
            ),
        },
        "descriptive_patient_median": {
            mode: {
                endpoint: float(
                    patient.loc[patient.history_mode == mode, endpoint].median()
                )
                for endpoint in endpoints
            }
            for mode in HISTORY_MODES
        },
        "paired_tests": tests,
        "runtime_seconds": float(time.time() - started),
        "environment": run_environment(),
        "ictal_target_read": False,
    }
    (out / "d_native_history_ablation.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps({
        "ordered_reproduces_phase_d": payload["ordered_arm_reproduces_phase_d"],
        "median": payload["descriptive_patient_median"],
    }, indent=2))


if __name__ == "__main__":
    main()
