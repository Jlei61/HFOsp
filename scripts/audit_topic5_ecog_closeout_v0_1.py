#!/usr/bin/env python3
"""Final engineering-contract audit for the ECoG physical-neighbour experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1"
    ))
    parser.add_argument("--allow-pending-figure", action="store_true")
    args = parser.parse_args()
    root = args.root
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "pass": bool(passed), "evidence": evidence})

    input_audit_path = root / "INPUT_AUDIT.json"
    sparse_path = root / "sparse_validation/SPARSE_READ_EQUIVALENCE_AUDIT.json"
    input_audit = json.loads(input_audit_path.read_text())
    sparse = json.loads(sparse_path.read_text())
    check("input_audit_pass", bool(input_audit.get("pass")), str(input_audit_path))
    check("sparse_vs_full_rank_exact", bool(
        sparse.get("all_participation_exact") and sparse.get("all_rank_matrices_exact")
    ), str(sparse_path))

    graph_counts = {}
    for neighbourhood in ("four_neighbour", "eight_neighbour"):
        for subject in ("958", "1084"):
            manifest = json.loads((root / f"graphs/{subject}/{neighbourhood}/GRAPH_MANIFEST.json").read_text())
            key = f"{subject}_{neighbourhood}"
            graph_counts[key] = {"wrong": manifest["n_wrong"], "random": manifest["n_degree_random"]}
    check("all_graph_null_families_frozen", all(
        value == {"wrong": 31, "random": 31} for value in graph_counts.values()
    ), graph_counts)

    summaries = sorted((root / "training").glob("*/*/summary.json"))
    formal = [json.loads(path.read_text()) for path in summaries if not json.loads(path.read_text()).get("smoke", False)]
    check("four_neighbour_training_units_384", len(formal) == 384, len(formal))
    check("training_contract_current", all(
        item.get("training_device_type") == "cpu"
        and item.get("batch_size") == 512
        and item.get("microsteps") == 2
        and item.get("state_dim") == 1
        and item.get("top1_contract") == "top_prediction_is_any_member_of_tied_next_rank_set_v0.1"
        for item in formal
    ), "CPU, batch=512, microsteps=2, state_dim=1, tied-set top1")
    checkpoint_ok = []
    for item in formal:
        path = Path(item["checkpoint_path"])
        checkpoint_ok.append(path.exists() and sha256_file(path) == item["checkpoint_sha256"])
    check("checkpoint_hashes_exact", len(checkpoint_ok) == 384 and all(checkpoint_ok), f"{sum(checkpoint_ok)}/384")
    check("training_outputs_finite", all(finite_tree({
        "train": item["train"], "validation": item["validation"], "test": item["test"]
    }) for item in formal), "train/validation/test")
    initial_by_subject_seed: dict[str, set[str]] = {}
    for item in formal:
        initial_by_subject_seed.setdefault(
            f"{item['subject']}_seed{item['seed_index']}", set()
        ).add(item["initial_parameter_sha256"])
    check("paired_initial_parameters_identical", all(len(values) == 1 for values in initial_by_subject_seed.values()), {
        key: len(value) for key, value in initial_by_subject_seed.items()
    })
    worker_failures = []
    worker_logs = list((root / "training/worker_logs").glob("*.json"))
    worker_logs += list((root / "training_eight_neighbour/worker_logs").glob("*.json"))
    worker_logs += list((root / "training_one_microstep/worker_logs").glob("*.json"))
    for path in worker_logs:
        payload = json.loads(path.read_text())
        failed = payload.get("failed", [])
        if isinstance(failed, list):
            worker_failures.extend({"path": str(path), "failure": item} for item in failed)
    check("all_training_worker_failures_zero", not worker_failures, {
        "n_logs": len(worker_logs), "n_failures": len(worker_failures),
    })

    graph_summary = json.loads((root / "summary/GRAPH_TRAINING_SUMMARY.json").read_text())
    extended_summary = json.loads((root / "summary/HELDOUT_EXTENDED_SUMMARY.json").read_text())
    field_summary = json.loads((root / "summary/FREE_FIELD_SUMMARY.json").read_text())
    patch_summary = json.loads((root / "summary/PATCH_NECESSITY_SUMMARY.json").read_text())
    inbound_summary = json.loads((root / "summary_inbound/INBOUND_ENTRY_SUMMARY.json").read_text())
    eight_summary = json.loads((root / "summary/EIGHT_NEIGHBOUR_SUMMARY.json").read_text())
    one_step_summary = json.loads((root / "summary/ONE_MICROSTEP_SUMMARY.json").read_text())
    tie_summary = json.loads((root / "summary/TIE_TOLERANCE_SENSITIVITY.json").read_text())
    check("four_neighbour_summary_complete", graph_summary.get("complete") and graph_summary.get("observed_formal_units") == 384, graph_summary.get("observed_formal_units"))
    check("heldout_extended_metrics_complete", extended_summary.get("complete") and extended_summary.get("n_units") == 384, extended_summary.get("n_units"))
    check("free_fields_complete", field_summary.get("complete") and field_summary.get("n_units") == 24, field_summary.get("n_units"))
    check("patch_units_complete", patch_summary.get("complete") and patch_summary.get("n_units") == 12, patch_summary.get("n_units"))
    check("inbound_entry_repair_units_complete", inbound_summary.get("complete") and inbound_summary.get("n_units") == 6, inbound_summary.get("n_units"))
    check("eight_neighbour_sensitivity_complete", eight_summary.get("complete") and eight_summary.get("n_units") == 192, eight_summary.get("n_units"))
    check("one_microstep_sensitivity_complete", one_step_summary.get("complete") and one_step_summary.get("n_units") == 192, one_step_summary.get("n_units"))
    check("tie_tolerance_data_sensitivity_complete", tie_summary.get("complete") and len(tie_summary.get("results", [])) == 6, len(tie_summary.get("results", [])))

    patch_audits = list((root / "patch_necessity").glob("*/*/*/SUMMARY.json"))
    patch_payloads = [json.loads(path.read_text()) for path in patch_audits]
    check("patch_parameter_hashes_unchanged", len(patch_payloads) == 12 and all(
        item.get("parameter_hash_unchanged") for item in patch_payloads
    ), f"{sum(bool(item.get('parameter_hash_unchanged')) for item in patch_payloads)}/12")
    check("patch_controls_32", len(patch_payloads) == 12 and all(
        item.get("n_controls_per_patch") == 32 for item in patch_payloads
    ), f"{len(patch_payloads)} units")

    inbound_audits = list((root / "patch_necessity_inbound").glob("*/*/patch_2x2/SUMMARY.json"))
    inbound_payloads = [json.loads(path.read_text()) for path in inbound_audits]
    check("inbound_entry_parameter_hashes_unchanged", len(inbound_payloads) == 6 and all(
        item.get("parameter_hash_unchanged") for item in inbound_payloads
    ), f"{sum(bool(item.get('parameter_hash_unchanged')) for item in inbound_payloads)}/6")
    check("inbound_entry_controls_32", len(inbound_payloads) == 6 and all(
        item.get("n_controls_per_patch") == 32
        and item.get("lesion_mode") == "inbound_first_entry"
        and item.get("first_entry_contract") == "no_patch_contact_recruited_before_next_rank_v0.1"
        for item in inbound_payloads
    ), f"{len(inbound_payloads)} units")

    figure_dir = root / "figures"
    figure_paths = [figure_dir / f"topic5_ecog_physical_neighbourhood_v0_1.{suffix}" for suffix in ("png", "pdf", "svg")]
    figure_ready = all(path.exists() and path.stat().st_size > 0 for path in figure_paths) and (figure_dir / "README.md").exists()
    check("figure_triplet_and_readme", figure_ready or args.allow_pending_figure, [str(path) for path in figure_paths])
    all_pass = all(item["pass"] for item in checks)
    payload = {
        "schema": "topic5_ecog_physical_neighbourhood_closeout_audit_v0.1",
        "status": "PASS" if all_pass else "FAIL",
        "n_checks": len(checks),
        "n_pass": sum(item["pass"] for item in checks),
        "checks": checks,
    }
    output = root / "CLOSEOUT_AUDIT.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
