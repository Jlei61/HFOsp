#!/usr/bin/env python3
"""Fail-closed package audit for the fixed R1.2b experiment."""
from __future__ import annotations

import json
from pathlib import Path

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.r1_2b import (
    R1_2B_REVISION, R1_2B_SUBJECTS,
)


def main() -> None:
    root = contract.RESULT_ROOT / "r1_2b"
    checks = []

    def check(name: str, passed: bool, detail) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    result_paths = []
    for subject in R1_2B_SUBJECTS:
        cache_path = root / "cache" / subject / "manifest.json"
        cache = json.loads(cache_path.read_text())
        upstream = json.loads(Path(cache["upstream_cache_manifest"]).read_text())
        check(
            f"{subject}:cache_revision_sealed",
            cache.get("status") == "COMPLETE"
            and cache.get("r1_2b_revision") == R1_2B_REVISION
            and cache.get("sealed_opened") is False,
            {"status": cache.get("status"), "revision": cache.get("r1_2b_revision")},
        )
        check(
            f"{subject}:cache_denominator",
            cache["n_train_anchors"] == upstream["n_train_anchors"]
            and cache["n_validation_anchors"] == upstream["n_validation_anchors"]
            and cache["n_anchors"] == (
                upstream["n_train_anchors"] + upstream["n_validation_anchors"]
            ),
            {
                "cache": [cache["n_train_anchors"], cache["n_validation_anchors"]],
                "upstream": [upstream["n_train_anchors"], upstream["n_validation_anchors"]],
            },
        )
        check(
            f"{subject}:epoch_zero_bridge_boundary",
            cache.get("bridge_selected_epochs") == {
                "explicit": 0, "explicit_raw": 0
            } and cache.get("initial_raw_gain") == 0.0,
            cache.get("bridge_selected_epochs"),
        )
        for field, hash_field in (
            ("base_contact_node", "base_contact_node_sha256"),
            ("raw_contact_node", "raw_contact_node_sha256"),
            ("contact_mask", "contact_mask_sha256"),
        ):
            check(
                f"{subject}:{field}_hash",
                contract.sha256_file(cache[field]) == cache[hash_field],
                cache[field],
            )
        for arm in ("joint_explicit", "joint_explicit_raw"):
            for seed in (0, 1, 2):
                path = root / "joint" / subject / f"{arm}_seed_{seed}" / "result.json"
                result_paths.append(path)
                value = json.loads(path.read_text())
                prefix = f"{subject}:{arm}:seed{seed}"
                check(
                    f"{prefix}:identity",
                    value.get("status") == "COMPLETE"
                    and value.get("subject") == subject
                    and value.get("arm") == arm
                    and value.get("seed") == seed
                    and value.get("r1_2b_revision") == R1_2B_REVISION,
                    {key: value.get(key) for key in (
                        "status", "subject", "arm", "seed", "r1_2b_revision"
                    )},
                )
                check(
                    f"{prefix}:scientific_contract",
                    value.get("sealed_opened") is False
                    and value.get("full_recorded_support") is True
                    and value.get("state_dim") == 8
                    and value.get("observer_to_state_lr_ratio") == 0.1
                    and value.get("frozen_upstream_raw_temporal_encoder") is True,
                    {
                        "sealed": value.get("sealed_opened"),
                        "state_dim": value.get("state_dim"),
                        "lr_ratio": value.get("observer_to_state_lr_ratio"),
                    },
                )
                trainable = value.get("trainable_parameter_names", [])
                observer_names = [name for name in trainable if name.startswith("last_observer")]
                check(
                    f"{prefix}:trainable_observer_scope",
                    bool(observer_names)
                    and all(name.startswith((
                        "last_observer.pool_token", "last_observer.spatial",
                        "last_observer.output_norm", "last_observer.raw_gain",
                    )) for name in observer_names),
                    observer_names,
                )
                check(
                    f"{prefix}:zero_effect_parity",
                    max(value.get("initial_parity_abs", {"bad": 1.0}).values()) <= 1e-6,
                    value.get("initial_parity_abs"),
                )
                horizon = value.get("horizon_correction_off", {})
                check(
                    f"{prefix}:horizon_contract",
                    horizon.get("status") == "COMPLETE"
                    and set(horizon.get("horizons", {})) == {"5", "10", "20"}
                    and horizon.get("future_event_history_teacher_forced") is True
                    and horizon.get("future_observation_correction_off") is True
                    and horizon.get("recorded_gaps_excluded") is True,
                    {key: horizon.get(key) for key in (
                        "status", "future_event_history_teacher_forced",
                        "future_observation_correction_off", "recorded_gaps_excluded"
                    )},
                )
                check(
                    f"{prefix}:hashes",
                    contract.sha256_file(value["checkpoint"]) == value["checkpoint_sha256"]
                    and contract.sha256_file(value["cache_manifest"]) == value["cache_manifest_sha256"]
                    and contract.sha256_file(value["baseline_checkpoint"]) == value["baseline_checkpoint_sha256"]
                    and contract.sha256_file(value["frozen_r1_2_reference"]["path"])
                    == value["frozen_r1_2_reference"]["sha256"],
                    value["checkpoint"],
                )

    report_root = root / "reports"
    for name in (
        "r1_2b_summary.json", "r1_2b_patient_first.csv",
        "combined_route_audit_plain_2026-08-25.md",
        "combined_route_audit_technical_2026-08-25.md",
        "combined_route_audit_manifest.json",
    ):
        check(f"report:{name}", (report_root / name).exists(), str(report_root / name))
    route = json.loads((report_root / "combined_route_audit_manifest.json").read_text())
    check(
        "route_boundary",
        route.get("scientific_route_deviation") is False
        and route.get("sealed_opened") is False
        and bool(route.get("implementation_coverage_gap")),
        {
            "deviation": route.get("scientific_route_deviation"),
            "gap": route.get("implementation_coverage_gap"),
        },
    )
    check("fit_count", len(result_paths) == 18, len(result_paths))
    final = {
        "status": "PASS" if all(value["pass"] for value in checks) else "FAIL",
        "all_pass": all(value["pass"] for value in checks),
        "contract": contract.REVISION,
        "r1_2b_revision": R1_2B_REVISION,
        "n_checks": len(checks), "n_fits": len(result_paths),
        "checks": checks, "sealed_opened": False,
    }
    output = root / "manifests/FINAL_PACKAGE_AUDIT.json"
    contract.atomic_json(output, final)
    print(json.dumps(final, indent=2, sort_keys=True))
    if not final["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
