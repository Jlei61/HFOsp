#!/usr/bin/env python3
"""Final engineering/scientific-contract audit for full-tissue LBSS v0.3."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUT = Path("results/topic5_lbss_full_tissue_rnn_v0_3")
ARMS = {
    "L0_LOCAL_ONLY",
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2_LOCAL_PLUS_RANDOM_LR",
    "L3_LOCAL_PLUS_LEARNED_LR",
    "C_L3_ORDER_SHUFFLED",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def nonfinite_json_paths(value, prefix: str = "") -> list[str]:
    """Return JSON paths containing NaN/Inf, which stdlib json may accept."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(nonfinite_json_paths(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(nonfinite_json_paths(item, f"{prefix}[{index}]"))
    elif isinstance(value, float) and not np.isfinite(value):
        found.append(prefix)
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure-dir", type=Path, default=None)
    args = parser.parse_args()
    out = args.out_root.resolve()
    selected_marker_path = out / "SELECTED_PRIMARY_ROOT.json"
    selected_marker = read_json(selected_marker_path) if selected_marker_path.exists() else {}
    controller_out = (
        Path(selected_marker["source_out_root"]).resolve()
        if selected_marker else out
    )
    figure_dir = (
        args.figure_dir.resolve() if args.figure_dir is not None
        else out.parents[0] / "paper-ready-figure/fig6_lbss_full_tissue_rnn/figures"
    )

    errors: list[str] = []
    spatial_decision_path = controller_out / "SPATIAL_DECISION_COMPLETE.json"
    if not spatial_decision_path.exists():
        errors.append(f"missing target-free spatial decision: {spatial_decision_path}")
        spatial_decision = {}
    else:
        spatial_decision = read_json(spatial_decision_path)
    trigger_path = controller_out / "SPATIAL_SEARCH_TRIGGER_DECISION.json"
    if not trigger_path.exists() or read_json(trigger_path).get("target_values_read") is not False:
        errors.append("spatial-search trigger is missing or not target-free")
    search = controller_out / "development_spatial_search_v0_4"
    for name in ("SCREEN_DECISION.json", "SPATIAL_MODEL_DECISION.json"):
        path = search / name
        if path.exists() and read_json(path).get("target_values_read") is not False:
            errors.append(f"spatial selection read early-ictal target: {path}")
    if selected_marker:
        pointer_path = controller_out / "PRIMARY_ARTIFACT_POINTER.json"
        pointer = read_json(pointer_path) if pointer_path.exists() else {}
        if Path(pointer.get("artifact_root", "/missing")).resolve() != out:
            errors.append("selected primary artifact pointer does not resolve to audited root")
        if spatial_decision.get("selected_contract") != "FULL_COHORT_SELECTED_SPATIAL_CONFIG":
            errors.append("selected root lacks full-cohort spatial confirmation")
        formal_decision = search / "FORMAL_SELECTED_DECISION.json"
        if (
            not formal_decision.exists()
            or read_json(formal_decision).get("target_values_read") is not False
        ):
            errors.append("selected full-cohort decision is missing or target-contaminated")
        selected_config_path = out / "SELECTED_SPATIAL_CONFIG.json"
        selected_config = (
            read_json(selected_config_path) if selected_config_path.exists() else {}
        )
        if not selected_config:
            errors.append("selected root lacks explicit spatial-configuration provenance")
        else:
            if selected_config.get("target_values_read") is not False:
                errors.append("selected spatial configuration is target-contaminated")
            if selected_config.get("config_id") != spatial_decision.get("selected_config_id"):
                errors.append("selected spatial configuration does not match controller decision")
            if selected_config.get("all_465_units_match_overrides") is not True:
                errors.append("selected spatial configuration was not verified across all 465 units")
            expected_hash = selected_marker.get("selected_config_sha256")
            if expected_hash and sha256(selected_config_path) != expected_hash:
                errors.append("selected spatial-configuration hash does not match root marker")
    repair_path = search / "MULTISTATE_REPAIR_PROVENANCE.json"
    repair = read_json(repair_path) if repair_path.exists() else {}
    if repair and repair.get("target_values_read") is not False:
        errors.append("multistate repair provenance is not target-free")
    comparator_audit_path = out.parent / "topic5_lbss_rnn_v0_2" / "NO_REC_EQUIVALENCE_AUDIT.json"
    comparator_audit = read_json(comparator_audit_path) if comparator_audit_path.exists() else {}
    if comparator_audit.get("verdict") != "EQUIVALENT_ENOUGH_FOR_MATCHED_CONTRAST":
        errors.append(
            "Claim A no-recurrence comparator lacks its required equivalence audit: "
            f"{comparator_audit_path}"
        )
    done = list((out / "per_fit").glob("*/*/seed*/DONE.json"))
    failed = list((out / "per_fit").glob("*/*/seed*/FAILED.json"))
    oom = list((out / "per_fit").glob("*/*/seed*/OOM.json"))
    metrics_paths = list((out / "per_fit").glob("*/*/seed*/metrics.json"))
    if len(done) != 465 or len(metrics_paths) != 465:
        errors.append(f"formal matrix incomplete: DONE={len(done)}, metrics={len(metrics_paths)}")
    if failed or oom:
        errors.append(f"unresolved failures: failed={len(failed)}, oom={len(oom)}")

    metric_rows = []
    producer_hashes = set()
    for path in metrics_paths:
        value = read_json(path)
        metric_rows.append({
            "fit_id": value["fit_id"], "subject": value["subject"],
            "arm": value["arm"], "seed": int(value["seed"]),
            "converged": bool(value["converged"]),
            "checkpoint": bool(value["best_checkpoint_eligible"]),
            "target_values_read": bool(value["target_values_read"]),
            "contact_nll": float(value["test"]["contact_nll"]),
            "distal_nll": float(value["distance_bins"]["distal"]["contact_nll"]),
        })
        producer_hashes.add(tuple(sorted(value["producer_hashes"].items())))
    metrics = pd.DataFrame(metric_rows)
    if len(metrics):
        if metrics.fit_id.nunique() != 31 or metrics.subject.nunique() != 21:
            errors.append(
                f"spatial denominator changed: fits={metrics.fit_id.nunique()}, patients={metrics.subject.nunique()}"
            )
        if set(metrics.arm) != ARMS or set(metrics.seed) != {0, 1, 2}:
            errors.append("arm/seed matrix changed")
        if not metrics.converged.all() or not metrics.checkpoint.all():
            errors.append("nonconverged or checkpoint-ineligible formal units")
        if metrics.target_values_read.any():
            errors.append("formal interictal unit read early-ictal target")
        if not np.isfinite(metrics[["contact_nll", "distal_nll"]].to_numpy(float)).all():
            errors.append("nonfinite formal metric")
    if len(producer_hashes) != 1:
        errors.append(f"mixed producer hashes across formal units: {len(producer_hashes)}")

    geometry = pd.read_csv(out / "LATENT_DOMAIN_AUDIT.csv")
    geometry = geometry[geometry.version.eq("v0.3")]
    if len(geometry) != 31:
        errors.append(f"geometry audit has {len(geometry)} fits, expected 31")
    if len(geometry) and (
        int(geometry.n_zero_h_nodes.min()) < 16
        or float(geometry.zero_h_fraction.min()) < 0.10
        or not geometry.all_nodes_one_strong_component.astype(bool).all()
    ):
        errors.append("full-tissue zero-H/strong-connectivity contract failed")

    required_markers = (
        "FORMAL_TRAINING_COMPLETE.json",
        "INTERICTAL_ANALYSIS_COMPLETE.json",
        "MODEL_FIELDS_FROZEN.json",
        "PATHWAY_ANALYSIS_COMPLETE.json",
        "ATTENUATED_FIELD_MANIFEST.json",
        "ATTENUATION_COMPLETE.json",
        "TARGET_UNSEAL_AUTHORIZATION.json",
        "EARLY_ICTAL_SCORING_COMPLETE.json",
        "TARGET_ACCESS_AUDIT.json",
        "PIPELINE_COMPLETE.json",
        "LBSS_CLAIM_ADJUDICATION_V0_3_COMPLETE.json",
    )
    for name in required_markers:
        if not (out / name).exists():
            errors.append(f"missing marker: {name}")

    attenuation_marker = (
        read_json(out / "ATTENUATION_COMPLETE.json")
        if (out / "ATTENUATION_COMPLETE.json").exists() else {}
    )
    unit_caches = list((out / "attenuation" / "unit_cache").glob("*/*/seed*.json.gz"))
    expected_unit_caches = 31 * 3 * 4
    if attenuation_marker.get("n_restart_safe_unit_caches") != expected_unit_caches:
        errors.append(
            "restart-safe attenuation unit denominator changed: "
            f"{attenuation_marker.get('n_restart_safe_unit_caches')} != {expected_unit_caches}"
        )
    if len(unit_caches) != expected_unit_caches:
        errors.append(
            f"attenuation unit caches incomplete: {len(unit_caches)} != {expected_unit_caches}"
        )

    device_audit_path = out / "ATTENUATION_DEVICE_EQUIVALENCE_AUDIT.json"
    device_audit = read_json(device_audit_path) if device_audit_path.exists() else {}
    if not device_audit_path.exists():
        errors.append("missing CPU/GPU attenuation equivalence audit")
    elif (
        device_audit.get("all_units_pass") is not True
        or int(device_audit.get("n_units", 0)) < 3
        or device_audit.get("target_values_read") is not False
    ):
        errors.append(
            "CPU/GPU attenuation equivalence requires >=3 target-free passing units"
        )

    target = read_json(out / "TARGET_ACCESS_AUDIT.json") if (out / "TARGET_ACCESS_AUDIT.json").exists() else {}
    expected_target = {
        "figure3_parent_patients": 17,
        "figure3_parent_seizures": 167,
        "spatial_model_patients": 12,
        "spatial_model_seizures": 141,
        "strict_broadband_patients": 11,
        "strict_broadband_seizures": 92,
    }
    for key, expected in expected_target.items():
        if target.get(key) != expected:
            errors.append(f"target denominator changed: {key}={target.get(key)} != {expected}")

    authorization_path = out / "TARGET_UNSEAL_AUTHORIZATION.json"
    if authorization_path.exists():
        authorization = read_json(authorization_path)
        for name, digest in authorization.get("frozen_hashes", {}).items():
            path = out / name
            if not path.exists() or sha256(path) != digest:
                errors.append(f"post-authorization frozen hash mismatch: {name}")
        for name, key in (
            ("MODEL_FIELD_MANIFEST.csv", "intact_field_manifest_sha256"),
            ("ATTENUATED_FIELD_MANIFEST.csv", "attenuated_field_manifest_sha256"),
        ):
            if not (out / name).exists() or sha256(out / name) != authorization.get(key):
                errors.append(f"post-authorization manifest hash mismatch: {name}")

    figure_complete = figure_dir / "FIGURE6_COMPLETE.json"
    if not figure_complete.exists():
        errors.append(f"missing final Figure 6 completion marker: {figure_complete}")
    else:
        assets = read_json(figure_complete).get("assets_sha256", {})
        for name, digest in assets.items():
            path = figure_dir / name
            if not path.exists() or sha256(path) != digest:
                errors.append(f"Figure 6 asset hash mismatch: {name}")
    figure_metadata_path = figure_dir / "FIGURE6_METADATA.json"
    if not figure_metadata_path.exists():
        errors.append(f"missing final Figure 6 metadata: {figure_metadata_path}")
    else:
        bad_paths = nonfinite_json_paths(read_json(figure_metadata_path))
        if bad_paths:
            errors.append(
                "Figure 6 metadata contains nonfinite values: " + ", ".join(bad_paths)
            )

    payload = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "formal_units": len(metrics_paths),
        "formal_patients": int(metrics.subject.nunique()) if len(metrics) else 0,
        "formal_fits": int(metrics.fit_id.nunique()) if len(metrics) else 0,
        "unresolved_failed": len(failed),
        "unresolved_oom": len(oom),
        "geometry_zero_h_fraction": {
            "min": float(geometry.zero_h_fraction.min()) if len(geometry) else None,
            "median": float(geometry.zero_h_fraction.median()) if len(geometry) else None,
            "max": float(geometry.zero_h_fraction.max()) if len(geometry) else None,
        },
        "contact_space_reference": {"interictal_patients": 34, "early_ictal_patients": 17, "seizures": 167},
        "spatial_exact_join": {"patients": 12, "seizures": 141},
        "strict_broadband": {"patients": 11, "seizures": 92},
        "no_recurrence_comparator": {
            "audit": str(comparator_audit_path),
            "verdict": comparator_audit.get("verdict"),
            "n_units_compared": comparator_audit.get("n_units_compared"),
        },
        "spatial_selection": {
            "controller_root": str(controller_out),
            "decision": str(spatial_decision_path),
            "selected_contract": spatial_decision.get("selected_contract"),
            "audited_root_is_selected": bool(selected_marker),
            "multistate_repair_recorded": bool(repair),
        },
        "attenuation_restart_safe_unit_caches": len(unit_caches),
        "attenuation_device_equivalence": {
            "path": str(device_audit_path),
            "n_units": device_audit.get("n_units"),
            "all_units_pass": device_audit.get("all_units_pass"),
            "maximum_absolute_metric_difference": max(
                (
                    float(row.get("max_abs_metric_difference", float("nan")))
                    for row in device_audit.get("rows", [])
                ),
                default=None,
            ),
        },
        "figure_dir": str(figure_dir),
    }
    (out / "CLOSEOUT_AUDIT.json").write_text(json.dumps(payload, indent=2) + "\n")
    if errors:
        raise RuntimeError("closeout audit failed:\n" + "\n".join(errors))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
