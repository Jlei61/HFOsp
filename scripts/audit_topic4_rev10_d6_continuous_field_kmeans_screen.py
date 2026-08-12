"""Rank the D6 smooth-field sensitivity screen with the Fig.4 contract."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_topic4_rev10_d5_3_spatial_ou_kmeans_grid import (  # noqa: E402
    audit_candidate,
)
from scripts.paper_figures.plot_fig4_spatial_edge_flow_validation import (  # noqa: E402
    _load_bundle,
)
from scripts.run_topic4_rev9l_forced_source_worker import _sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d6_continuous_field_kmeans_screen.json"


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False,
        ))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def recruitment_consistency(bundle, eligible_network_seeds):
    patient_ranks = np.asarray(bundle["patient"]["patient_train_ranks"], float)
    patient_labels = np.asarray(
        bundle["patient"]["patient_train_old_labels"], int,
    )
    patient = [
        float(np.median(np.isfinite(patient_ranks[patient_labels == mode]).sum(axis=1)))
        for mode in (0, 1)
    ]
    records = np.asarray([row["seed"] for row in bundle["records"]], int)
    model, by_seed = [], {}
    for mode in (0, 1):
        values = []
        for seed in eligible_network_seeds:
            selected = (
                bundle["clean"] & (bundle["labels"] == mode)
                & (records == int(seed))
            )
            if np.any(selected):
                value = float(np.median(
                    np.isfinite(bundle["ranks"][selected]).sum(axis=1)
                ))
                values.append(value)
                by_seed.setdefault(str(seed), {})["AB"[mode]] = value
        model.append(float(np.mean(values)) if values else 0.0)
    errors = [abs(model[i] - patient[i]) / patient_ranks.shape[1] for i in (0, 1)]
    return {
        "patient_train_median_recruited_contacts": {"A": patient[0], "B": patient[1]},
        "model_equal_network_median_recruited_contacts": {"A": model[0], "B": model[1]},
        "absolute_error_fraction_of_15": {"A": errors[0], "B": errors[1]},
        "worst_mode_error": float(max(errors)),
        "by_seed": by_seed,
    }


def d6_score(row, recruitment):
    if not row["evaluable"]:
        return None
    return float(
        (1.0 - row["balanced_kmeans"]["purity_median"])
        + 0.125 * (1.0 - row["signed_patient_margin"])
        + 0.25 * recruitment["worst_mode_error"]
        + 0.10 * row["activity"]["mean_network_ood_fraction"]
        + 0.05 * row["activity"]["mean_network_fraction_time_above_detector"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"]
    manifest_path = root / "candidate_manifest.json"
    summary_path = root / "canary_summary_returned_only.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    if manifest.get("status") != (
            "REV10D6_CONTINUOUS_FIELD_SENSITIVITY_LIBRARY_FROZEN"):
        raise RuntimeError("D6 manifest is not frozen")
    if summary.get("status") != (
            "REV10D6_RETURNED_ONLY_CONTINUOUS_FIELD_SCREEN_COMPLETE"):
        raise RuntimeError("D6 aggregate is incomplete")
    aggregate = {row["candidate_id"]: row for row in summary["candidate_rows"]}
    selection = config["search"]["kmeans_selection"]
    rows = []
    for candidate in manifest["candidate_set"]["candidates"]:
        row = audit_candidate(
            config_path, root, candidate,
            aggregate[candidate["candidate_id"]], selection,
        )
        bundle = _load_bundle(
            config_path, root, candidate["candidate_id"],
            allow_exploratory_candidate=True,
        )
        eligible = row["balanced_kmeans"].get("eligible_network_seeds", [])
        recruitment = recruitment_consistency(bundle, eligible)
        row["recruitment_consistency"] = recruitment
        row["selection_score"] = d6_score(row, recruitment)
        row["node_field"] = {
            key: candidate["node_field"].get(key)
            for key in (
                "candidate_id", "field_sha256", "role", "roughness",
                "residual_coordinates",
            )
        }
        rows.append(row)
    ranked = sorted(rows, key=lambda row: (
        row["selection_score"] is None,
        np.inf if row["selection_score"] is None else row["selection_score"],
        row["candidate_id"],
    ))
    evaluable = [row for row in ranked if row["selection_score"] is not None]
    best = evaluable[0] if evaluable else None
    baseline = next(row for row in rows if row["candidate_id"] == "edge_noop")
    payload = {
        "status": (
            "REV10D6_CONTINUOUS_FIELD_SENSITIVITY_COMPLETE"
            if best is not None
            else "REV10D6_NO_EVALUABLE_CONTINUOUS_FIELD_DIRECTION"
        ),
        "selected_candidate_id": None if best is None else best["candidate_id"],
        "selected_score": None if best is None else best["selection_score"],
        "baseline_score": baseline["selection_score"],
        "selected_minus_baseline_score": (
            None if best is None or baseline["selection_score"] is None
            else float(best["selection_score"] - baseline["selection_score"])
        ),
        "top_candidate_ids_for_fresh_refinement": [
            row["candidate_id"] for row in evaluable[:6]
        ],
        "candidate_rows": ranked,
        "selection_contract": selection,
        "selection_is_exploratory_not_a_gate": True,
        "patient_matched_q05_is_reported_not_enforced": True,
        "claim_boundary": (
            "development fit networks; one-direction smooth-field sensitivity "
            "under frozen OU; no optimizer-convergence, patient-blind, or "
            "Fig4 acceptance claim"
        ),
        "inputs": {
            "config": {"path": str(config_path.relative_to(ROOT)),
                       "sha256": _sha256(config_path)},
            "manifest": {"path": str(manifest_path.relative_to(ROOT)),
                         "sha256": _sha256(manifest_path)},
            "summary": {"path": str(summary_path.relative_to(ROOT)),
                        "sha256": _sha256(summary_path)},
        },
    }
    output = root / "canary_verdict.json"
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "selected_candidate_id": payload["selected_candidate_id"],
        "selected_score": payload["selected_score"],
        "baseline_score": payload["baseline_score"],
        "selected_minus_baseline_score": payload[
            "selected_minus_baseline_score"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
