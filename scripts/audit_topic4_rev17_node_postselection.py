#!/usr/bin/env python3
"""Audit same-network natural KMeans for one frozen rev17 dual Node field."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev17_node_postselection.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_topic4_rev15_node_postselection as base  # noqa: E402
from scripts import rescore_topic4_rev13_exact_off_static_node as exact  # noqa: E402
from scripts import rescore_topic4_rev14_static_node_historical_libraries as historical  # noqa: E402
from scripts.paper_figures import plot_fig4_spatial_edge_flow_validation as fig4  # noqa: E402


STATUS_ACCEPTED = "REV17_NODE_POSTSELECTION_ACCEPTED"
STATUS_REJECTED = "REV17_NODE_POSTSELECTION_REJECTED"
WORKER_STATUS = "REV12ND_NODE_WORKER_COMPLETE"


def _worker_arrays(
    *, confirmation: dict[str, Any], manifest: dict[str, Any],
    candidate_id: str, mapping_sha256: str, seed: int, root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = root / confirmation["output_root"] / "workers" / (
        f"{candidate_id}_seed_{seed}.json"
    )
    if not path.is_file():
        raise RuntimeError(f"rev17 confirmation worker is missing: {path}")
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != WORKER_STATUS
        or payload.get("candidate_id") != candidate_id
        or int(payload.get("seed")) != int(seed)
    ):
        raise RuntimeError("rev17 confirmation worker identity changed")
    if payload.get("node_mapping", {}).get("mapping_sha256") != mapping_sha256:
        raise RuntimeError("rev17 confirmation worker dual mapping changed")
    mechanism = payload.get("mechanism_freeze", {})
    if any(mechanism.get(key) != "off" for key in ("EE", "E_to_I", "Z_M")):
        raise RuntimeError("rev17 postselection worker activated another mechanism")
    provenance = payload.get("provenance", {})
    if (
        provenance.get("runtime_modules_dirty") not in (False, 0)
        or provenance.get("runtime_modules_match_expected_commit") not in (True, 1)
        or provenance.get("config_sha256") != manifest["config_sha256"]
    ):
        raise RuntimeError("rev17 confirmation worker provenance changed")
    arrays_path = root / payload["arrays"]["path"]
    if not arrays_path.is_file() or base._sha256(arrays_path) != payload["arrays"]["sha256"]:
        raise RuntimeError("rev17 confirmation worker arrays changed")
    arrays = exact._load_npz_keys(arrays_path, exact.WORKER_ARRAY_KEYS)
    return arrays, {
        "json": {"path": str(path), "sha256": base._sha256(path)},
        "npz": {"path": str(arrays_path), "sha256": base._sha256(arrays_path)},
    }


def audit(config_path: Path = DEFAULT_CONFIG,
          root: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    config_path = config_path.resolve(); root = root.resolve()
    config = json.loads(config_path.read_text())
    if config.get("schema_id") != "topic4_rev17_node_postselection_v1":
        raise RuntimeError("rev17 postselection schema changed")
    if config["boundaries"].get("patient_heldout_used") is not False or config[
        "boundaries"
    ].get("EE_EtoI_ZM") != "off":
        raise RuntimeError("rev17 postselection crossed a boundary")
    paths = {
        key: base._resolve_record(record, artifact_root=root)
        for key, record in config["inputs"].items()
    }
    confirmation = json.loads(paths["confirmation_config"].read_text())
    manifest = json.loads(paths["confirmation_manifest"].read_text())
    confirmation_audit = json.loads(paths["confirmation_audit"].read_text())
    candidate_id = config["selected_candidate"]["candidate_id"]
    mapping_hash = config["selected_candidate"]["mapping_sha256"]
    if (
        confirmation.get("schema_id") != "topic4_rev17_node_confirmation_v1"
        or manifest.get("status") != "REV17_NODE_CONFIRMATION_CANDIDATES_FROZEN"
        or manifest.get("config_sha256") != base._sha256(paths["confirmation_config"])
        or confirmation_audit.get("scientific_confirmation", {}).get("accepted") is not True
        or confirmation_audit.get("candidate_id") != candidate_id
    ):
        raise RuntimeError("rev17 postselection source confirmation changed")
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    if candidates[candidate_id]["node_mapping"]["mapping_sha256"] != mapping_hash:
        raise RuntimeError("rev17 postselection candidate mapping changed")
    j14 = json.loads(paths["j14_config"].read_text())
    context = historical._patient_context(j14, root)
    semantic = base.patient_semantic_audit(
        patient=context["patient"],
        figure2_field=json.loads(paths["figure2_template_field"].read_text()),
    )
    network_rows, reusable, worker_inputs = [], [], []
    for seed in config["network_seeds"]:
        arrays, inputs = _worker_arrays(
            confirmation=confirmation, manifest=manifest,
            candidate_id=candidate_id, mapping_sha256=mapping_hash,
            seed=int(seed), root=root,
        )
        row, data = base._score_network(
            arrays=arrays, seed=int(seed), context=context,
            minimum_contacts=int(config["event_contract"]["minimum_readable_contacts"]),
        )
        network_rows.append(row); reusable.append(data)
        worker_inputs.append({"network_seed": int(seed), **inputs})
    pooled_ranks = np.concatenate([row["ranks"] for row in reusable], axis=0)
    pooled_supervised = np.concatenate([row["supervised"] for row in reusable])
    bundle = {
        "clean": np.ones(len(pooled_ranks), dtype=bool),
        "ranks": pooled_ranks, "labels": pooled_supervised,
        "static": {"contact_names": np.asarray(context["patient"]["contact_names"])},
    }
    natural = None
    if len(pooled_ranks) >= 6:
        try:
            natural = fig4._canonical_rank_kmeans(
                bundle, min_shared_contacts=int(
                    config["event_contract"]["minimum_readable_contacts"]
                ),
            )
        except (RuntimeError, ValueError):
            natural = None
    if natural is None:
        selected = np.arange(len(pooled_ranks), dtype=int)
        direction = pooled_supervised.copy()
        mapped = np.full(len(pooled_ranks), -1, dtype=int)
        mapping = {"direction_purity": 0.0, "ami_with_supervised_direction": -1.0}
        kmeans_status = "NOT_EVALUABLE_INSUFFICIENT_OR_DEGENERATE_EVENTS"
    else:
        selected = np.asarray(natural["clean_global_index"], dtype=int)
        direction = pooled_supervised[selected]
        mapped, mapping = base._map_clusters(
            np.asarray(natural["labels"], dtype=int), direction,
        )
        kmeans_status = "OK"
    patient_profiles = base._mean_profiles(
        context["patient"]["all_ranks"], context["patient"]["all_labels"],
    )
    supervised_profiles = base._mean_profiles(pooled_ranks[selected], direction)
    kmeans_profiles = base._mean_profiles(pooled_ranks[selected], mapped)
    pooled_matrix = base._similarity(kmeans_profiles, patient_profiles)
    decision = base.acceptance_decision(
        network_rows=network_rows, pooled_matrix=pooled_matrix,
        acceptance=config["acceptance"],
    )
    provenance = base._provenance()
    status = (
        "INVALID_PROVENANCE" if not provenance["formal_ready"]
        else STATUS_ACCEPTED if decision["accepted"] else STATUS_REJECTED
    )
    output_root = root / config["output_root"] / "analysis"
    output_json = output_root / "node_postselection_audit.json"
    output_csv = output_root / "node_postselection_network_summary.csv"
    payload = {
        "schema_id": "topic4_rev17_node_postselection_audit_v1",
        "status": status, "candidate_id": candidate_id,
        "provenance": provenance, "semantic_mode_audit": semantic,
        "network_results": network_rows,
        "pooled": {
            "kmeans_status": kmeans_status,
            "formal_clean_events": int(len(selected)),
            "supervised_counts_MTA_MTB": [
                int(np.sum(direction == mode)) for mode in base.SEMANTIC_MODE_ORDER
            ],
            "kmeans_counts_MTA_MTB": [
                int(np.sum(mapped == mode)) for mode in base.SEMANTIC_MODE_ORDER
            ],
            "kmeans_ami_with_supervised_direction": mapping[
                "ami_with_supervised_direction"
            ],
            "kmeans_direction_purity": mapping["direction_purity"],
            "kmeans_stability_ami_median": (
                None if natural is None else natural["stability_ami_median"]
            ),
            "supervised_patient_matrix_MTA_MTB_by_TA_TB": base._similarity(
                supervised_profiles, patient_profiles,
            ),
            "kmeans_patient_matrix_MTA_MTB_by_TA_TB": pooled_matrix,
            "equal_network_supervised_patient_matrix_MTA_MTB_by_TA_TB": base._similarity(
                np.nanmean(np.asarray([
                    row["supervised_profiles"] for row in reusable
                ]), axis=0), patient_profiles,
            ),
            "equal_network_kmeans_patient_matrix_MTA_MTB_by_TA_TB": base._similarity(
                np.nanmean(np.asarray([
                    row["kmeans_profiles"] for row in reusable
                ]), axis=0), patient_profiles,
            ),
        },
        "acceptance": decision,
        "inputs": {
            "config": {"path": str(config_path), "sha256": base._sha256(config_path)},
            "hashed_inputs": config["inputs"], "workers": worker_inputs,
        },
        "boundaries": {
            **config["boundaries"], "patient_heldout_loaded": False,
            "field_reranking_performed": False, "SNN_simulation_run": False,
        },
        "claim_boundary": config["claim_boundary"],
        "outputs": {"json": str(output_json), "network_csv": str(output_csv)},
    }
    base._atomic_json(output_json, payload)
    base._atomic_csv(output_csv, [{
        "network_seed": row["network_seed"],
        "formal_clean_events": row["event_selection"]["n_formal_clean"],
        "supervised_MTA": row["supervised_counts_MTA_MTB"][0],
        "supervised_MTB": row["supervised_counts_MTA_MTB"][1],
        "kmeans_MTA": row["kmeans_counts_MTA_MTB"][0],
        "kmeans_MTB": row["kmeans_counts_MTA_MTB"][1],
        "kmeans_ami": row["kmeans_ami_with_supervised_direction"],
        "kmeans_purity": row["kmeans_direction_purity"],
    } for row in network_rows])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    payload = audit(args.config, args.artifact_root)
    print(json.dumps({
        "status": payload["status"], "candidate_id": payload["candidate_id"],
        "accepted": payload["acceptance"]["accepted"],
        "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
