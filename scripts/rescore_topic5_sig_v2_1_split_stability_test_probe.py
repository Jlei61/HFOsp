#!/usr/bin/env python3
"""Rescore D3 operators on inner-test probes not used for checkpoint selection."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_sig_matched_baseline_ladder import _build as _build_baseline  # noqa: E402
from src.topic5_shared_propagation_field import (  # noqa: E402
    load_subject_rank_events,
    sha256_file,
)
from src.topic5_stable_interaction_graph import StableInteractionGraph  # noqa: E402


CONFIG = ROOT / "config/topic5_stable_interaction_identifiability_v2_1.yaml"
LADDER_ROOT = (
    ROOT
    / "results/topic5_stable_interaction_graph/development"
    / "human_matched_baseline_ladder_v0_2_training_adequacy"
)
D3_ROOT = (
    ROOT
    / "results/topic5_stable_interaction_graph/development"
    / "v2_1_split_stability"
)
OUTPUT = (
    ROOT
    / "results/topic5_stable_interaction_graph/development"
    / "v2_1_split_stability_test_probe_rescore"
)
CONDITIONS = ("real", "m1_phase_surrogate", "m3_template_surrogate")


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _subsample(indices: np.ndarray, limit: int) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if len(values) <= int(limit):
        return values
    return values[np.linspace(0, len(values) - 1, int(limit)).astype(int)]


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float).ravel()
    b = np.asarray(right, dtype=float).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < 10 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return float("nan")
    return float(spearmanr(a[valid], b[valid]).statistic)


def _load_null(subject: str, name: str, contacts: int, seed: int):
    path = LADDER_ROOT / "per_run" / subject / f"seed_{seed}/checkpoint.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["models"][name]["state_dict"]
    model = _build_baseline(name, contacts, state["static_bias"].numpy())
    model.load_state_dict(state)
    model.eval()
    return model, path


def _generate(model, groups: np.ndarray, counts: np.ndarray, seed: int) -> np.ndarray:
    with torch.no_grad():
        return model.generate_conditioned(
            torch.as_tensor(groups, dtype=torch.long),
            torch.as_tensor(counts, dtype=torch.long),
            seed=int(seed),
        ).numpy().astype(np.int16)


def _load_sig_models(subject: str, contacts: int) -> tuple[dict[str, dict[str, dict[int, StableInteractionGraph]]], Path]:
    path = D3_ROOT / "per_subject" / subject / "checkpoints.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    output = {condition: {"early": {}, "late": {}} for condition in CONDITIONS}
    for key, value in checkpoint["models"].items():
        condition, half, seed_text = key.split("/")
        state = value["state_dict"]
        model = StableInteractionGraph(
            contacts,
            static_bias=state["static_bias"].numpy(),
            learn_graph=True,
        )
        model.load_state_dict(state)
        model.eval()
        output[condition][half][int(seed_text)] = model
    return output, path


def _rescore_subject(subject: str, config: dict[str, Any]) -> dict[str, Any]:
    record = load_subject_rank_events(
        ROOT / config["data"]["dataset_dir"], subject
    )
    _, _, test = record.development_split(
        float(config["data"]["validation_fraction"]),
        float(config["data"]["test_fraction"]),
    )
    test = _subsample(
        test, int(config["split_stability"]["max_common_probe_events"])
    )
    if np.intersect1d(test, record.old_heldout20_indices).size:
        raise RuntimeError(f"{subject}: outer heldout20 leakage")
    contacts = len(record.contact_names)
    null_seed = int(config["split_stability"]["null_generator_fit_seed"])
    m1, m1_path = _load_null(
        subject, "m1_markov_matched_phase", contacts, null_seed
    )
    m3, m3_path = _load_null(
        subject, "m3_latent_template", contacts, null_seed
    )
    groups = {
        "real": record.group_ids[test],
        "m1_phase_surrogate": _generate(
            m1, record.group_ids[test], record.group_count[test], 20260911
        ),
        "m3_template_surrogate": _generate(
            m3, record.group_ids[test], record.group_count[test], 20260912
        ),
    }
    models, d3_checkpoint = _load_sig_models(subject, contacts)
    operators = {condition: {"early": {}, "late": {}} for condition in CONDITIONS}
    counts = torch.as_tensor(record.group_count[test], dtype=torch.long)
    for condition in CONDITIONS:
        probe = torch.as_tensor(groups[condition], dtype=torch.long)
        for half in ("early", "late"):
            for seed, model in models[condition][half].items():
                operators[condition][half][seed] = (
                    model.empirical_marginal_intervention_matrix(probe, counts)
                    .numpy()
                )
    conditions = {}
    for condition in CONDITIONS:
        values = [
            _spearman(
                operators[condition]["early"][left],
                operators[condition]["late"][right],
            )
            for left, right in itertools.product(
                sorted(operators[condition]["early"]),
                sorted(operators[condition]["late"]),
            )
        ]
        conditions[condition] = {
            "cross_seed_half_stability_median": float(np.nanmedian(values)),
            "cross_seed_half_stability_min": float(np.nanmin(values)),
            "n_comparisons": len(values),
        }
    strongest_null = max(
        conditions["m1_phase_surrogate"]["cross_seed_half_stability_median"],
        conditions["m3_template_surrogate"]["cross_seed_half_stability_median"],
    )
    return {
        "subject": subject,
        "n_test_probe_events": len(test),
        "probe_contract": (
            "inner test was not used to fit or select the saved early/late SIG "
            "or M1/M3 checkpoints; it has been viewed in prior development "
            "analyses, so this remains exploratory sensitivity, not confirmation"
        ),
        "conditions": conditions,
        "real_minus_strongest_matched_null_stability": (
            conditions["real"]["cross_seed_half_stability_median"]
            - strongest_null
        ),
        "dependencies": {
            "d3_checkpoint": {
                "path": str(d3_checkpoint),
                "sha256": sha256_file(d3_checkpoint),
            },
            "m1_m3_checkpoint": {
                "path": str(m1_path),
                "sha256": sha256_file(m1_path),
                "same_path_for_m3": str(m3_path) == str(m1_path),
            },
        },
        "input_sha256": record.input_sha256,
        "old_heldout20_scored": False,
        "snn_inputs_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text())
    patients = []
    for subject in config["pilot"]["subjects"]:
        row = _rescore_subject(str(subject), config)
        patients.append(row)
        print(json.dumps({
            "subject": subject,
            "real_minus_null": row[
                "real_minus_strongest_matched_null_stability"
            ],
        }))
    values = np.asarray(
        [row["real_minus_strongest_matched_null_stability"] for row in patients],
        dtype=float,
    )
    payload = {
        "contract": "topic5_stable_interaction_identifiability_v2_1_d3_test_probe_rescore",
        "status": "COMPLETE_TEST_PROBE_SENSITIVITY",
        "n_subjects": len(patients),
        "real_minus_strongest_null": {
            "median": float(np.nanmedian(values)),
            "n_positive": int(np.sum(values > 0)),
            "values": values.tolist(),
        },
        "patients": patients,
        "old_heldout20_scored": False,
        "snn_inputs_read": False,
        "source_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(CONFIG),
    }
    _write(args.output_dir / "D3_TEST_PROBE_RESCORE.json", payload)
    print(json.dumps(payload["real_minus_strongest_null"], indent=2))


if __name__ == "__main__":
    main()
