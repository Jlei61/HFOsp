#!/usr/bin/env python3
"""Aggregate rev22 Task 10b structural controls after validation is opened."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
STAGE = ARTIFACT_ROOT / "results/topic4_sef_hfo/data_driven_dual_core_interictal_identifiability"


def _module():
    spec = importlib.util.spec_from_file_location(
        "rev22_validation", ROOT / "scripts/aggregate_topic4_rev22_validation.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _index(manifest: dict) -> dict[str, dict]:
    return {str(row["candidate_id"]): row for row in manifest["candidates"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-config", type=Path,
                        default=ROOT / "config/topic4_rev22_dci_dual_core_interictal_identifiability.json")
    parser.add_argument("--structural-config", type=Path,
                        default=ROOT / "config/topic4_rev22_dci_structural_null_execution.json")
    parser.add_argument("--structural-manifest", type=Path,
                        default=STAGE / "structural_nulls/candidate_manifest.json")
    parser.add_argument("--main-manifest", type=Path,
                        default=STAGE / "response_fit/final_execution_candidate_manifest.json")
    parser.add_argument("--seed-manifest", type=Path,
                        default=STAGE / "response_design/seed_manifest.json")
    parser.add_argument("--structural-workers", type=Path,
                        default=STAGE / "structural_nulls/workers")
    parser.add_argument("--confirmation-workers", type=Path,
                        default=STAGE / "confirmation/workers")
    parser.add_argument("--training-contract", type=Path,
                        default=STAGE / "objective_qualification/patient_training_contract_v1.npz")
    parser.add_argument("--output", type=Path,
                        default=STAGE / "structural_nulls/structural_null_aggregate.json")
    parser.add_argument("--bootstrap-draws", type=int, default=4096)
    args = parser.parse_args()
    module = _module()
    analysis = _read(args.analysis_config)
    config = _read(args.structural_config)
    manifest = _read(args.structural_manifest)
    main_manifest = _read(args.main_manifest)
    seeds = _read(args.seed_manifest)
    response_fit = _read(STAGE / "response_fit/response_fit.json")
    budget = response_fit["recall_support_budget"]
    if budget.get("status") != "OK":
        raise RuntimeError("recall support budget is not estimable")

    def configured(name: str) -> Path:
        record = analysis["inputs"][name]
        path = ARTIFACT_ROOT / record["path"]
        if module._sha256(path) != record["sha256"]:
            raise RuntimeError(f"configured input changed: {name}")
        return path

    support = _read(configured("patient_support_config"))
    classifier_record = support["inputs"]["old_ab_train_only_classifier"]
    classifier_path = ARTIFACT_ROOT / classifier_record["path"]
    if module._sha256(classifier_path) != classifier_record["sha256"]:
        raise RuntimeError("direction classifier changed")
    context, metadata = module.load_validation_context(
        training_contract_path=args.training_contract,
        patient_training_target_path=configured("patient_training_target"),
        heldout_path=configured("patient_heldout_npz"),
        contact_contract_path=configured("contact_contract"),
        classifier_manifest_path=classifier_path,
        n_cov=int(budget["n_cov"]), r_cov=float(budget["r_cov"]),
        floor_draws=64, lag_cap_ms=float(analysis["objective"]["lag_cap_ms"]),
        n_pair_min=int(analysis["objective"]["n_pair_min"]),
        cover_quantile=float(analysis["objective"]["cover_quantile"]),
    )
    seed_hash = module._sha256(args.seed_manifest)
    response_hash = manifest["response_design_manifest_sha256"]
    units = module._phase_units(seeds, "confirmation")[:6]
    structural = _index(manifest)
    main = _index(main_manifest)

    raw, scored = {}, {}
    for index, (candidate_id, candidate) in enumerate(structural.items()):
        candidate_units = [module._validate_worker(
            args.structural_workers, candidate, unit["topology_seed"], unit["dynamics_seed"],
            manifest_hash=response_hash, seed_hash=seed_hash,
            contact_names=metadata["contact_names"],
        ) for unit in units]
        raw[candidate_id] = candidate_units
        scored[candidate_id] = module._score_candidate(
            candidate_units, context, phase="structural_nulls", candidate_id=candidate_id,
            seed=20260920 + index * 1000, recall_subsamples=200, c2st_resamples=20)

    marker = [row.get("structural_null_primary_selection") for row in structural.values()
              if row.get("structural_null_primary_selection")]
    full_id = str(marker[0]["candidate_id"])
    intact_ids = {"M0000": "dci_p000", "full": full_id}
    intact_raw, intact_scored = {}, {}
    for index, (condition, candidate_id) in enumerate(intact_ids.items()):
        candidate = main[candidate_id]
        candidate_units = [module._validate_worker(
            args.confirmation_workers, candidate, unit["topology_seed"], unit["dynamics_seed"],
            manifest_hash=response_hash, seed_hash=seed_hash,
            contact_names=metadata["contact_names"],
        ) for unit in units]
        intact_raw[condition] = candidate_units
        intact_scored[condition] = module._score_candidate(
            candidate_units, context, phase="intact_confirmation_prefix", candidate_id=candidate_id,
            seed=20269920 + index * 1000, recall_subsamples=200, c2st_resamples=20)

    contrasts = []
    for index, (candidate_id, candidate) in enumerate(structural.items()):
        record = candidate["structural_null"]
        family, condition = record["family"], record["condition"]
        if family == "isotropic_graph":
            contrasts.append({
                "candidate_id": candidate_id, "family": family, "condition": condition,
                "status": "UNPAIRED_TOPOLOGY_CONTROL",
                "null_endpoints": scored[candidate_id]["primary_endpoints"],
                "intact_endpoints": intact_scored[condition]["primary_endpoints"],
            })
            continue
        if family == "node_blocking_factor":
            if condition != "full":
                continue
            reference_id = candidate_id.rsplit("_", 1)[0] + "_M0000"
            contrasts.append({
                "candidate_id": candidate_id, "reference_id": reference_id,
                "family": family, "condition": "full_minus_M0000_within_node",
                "contrast": module.paired_nonlinear_bootstrap(
                    raw[candidate_id], raw[reference_id], context, draws=args.bootstrap_draws,
                    seed=20270920 + index, recall_subsamples=200),
            })
            continue
        contrasts.append({
            "candidate_id": candidate_id, "intact_candidate_id": intact_ids[condition],
            "family": family, "condition": condition,
            "contrast": module.paired_nonlinear_bootstrap(
                intact_raw[condition], raw[candidate_id], context, draws=args.bootstrap_draws,
                seed=20270920 + index, recall_subsamples=200),
        })
    payload = {
        "schema_id": "topic4_rev22_dci_structural_null_aggregate_v1",
        "status": "STRUCTURAL_NULL_AGGREGATE_COMPLETE",
        "candidate_count": len(structural), "unit_count": len(structural) * len(units),
        "full_model_candidate_id": full_id,
        "scores": scored, "intact_prefix_scores": intact_scored, "contrasts": contrasts,
        "input_hashes": {
            **metadata["input_hashes"],
            "structural_manifest": module._sha256(args.structural_manifest),
            "main_manifest": module._sha256(args.main_manifest),
            "seed_manifest": seed_hash,
        },
        "claim_boundary": ("Fixed-topology comparisons are paired over six topology seeds; "
                           "AR=1 rebuilt topology is descriptive and unpaired; Node-factor "
                           "contrasts test direction preservation, not Node selection."),
    }
    module._atomic_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "candidates": len(structural),
                      "contrasts": len(contrasts), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
