"""Repair D6 selection with natural-proportion and cross-fitted KMeans."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_topic4_rev10_d5_3_spatial_ou_kmeans_grid import (  # noqa: E402
    balanced_network_mode_bootstrap,
)
from scripts.paper_figures.plot_fig4_spatial_edge_flow_validation import (  # noqa: E402
    _load_bundle,
)
from scripts.run_topic4_rev9l_forced_source_worker import _sha256  # noqa: E402
from src.topic4_d6_natural_kmeans import (  # noqa: E402
    contact_split_folds,
    crossfit_patient_readout,
    natural_kmeans,
    network_bootstrap,
)


DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d6_continuous_field_kmeans_screen.json"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(
            _jsonable(payload), indent=2, sort_keys=True, allow_nan=False,
        ))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _finite(value):
    return None if value is None or not np.isfinite(value) else float(value)


def _candidate_metrics(config_path, root, candidate, contract, selection):
    bundle = _load_bundle(
        config_path, root, candidate["candidate_id"],
        allow_exploratory_candidate=True,
    )
    folds = contact_split_folds(contract)
    record_seed = np.asarray([row["seed"] for row in bundle["records"]], int)
    patient_ranks = bundle["patient"]["patient_train_ranks"]
    patient_labels = bundle["patient"]["patient_train_old_labels"]
    patient_recruitment = {
        "A": float(np.median(np.sum(np.isfinite(
            patient_ranks[patient_labels == 0]), axis=1))),
        "B": float(np.median(np.sum(np.isfinite(
            patient_ranks[patient_labels == 1]), axis=1))),
    }
    by_seed = {}
    pooled_index, pooled_crossfit_labels = [], []
    for seed in bundle["config"]["search"][bundle["network_seed_key"]]:
        index = np.flatnonzero(bundle["clean"] & (record_seed == int(seed)))
        crossfit = crossfit_patient_readout(
            bundle["ranks"][index], patient_ranks, patient_labels, folds,
        )
        natural = natural_kmeans(
            bundle["ranks"][index], crossfit["consensus_labels"],
            random_state=int(seed),
        )
        consensus = np.asarray(crossfit.pop("consensus_labels"), int)
        recruitment = {}
        for mode, name in ((0, "A"), (1, "B")):
            selected = consensus == mode
            model = (
                float(np.median(np.sum(np.isfinite(
                    bundle["ranks"][index][selected]), axis=1)))
                if np.any(selected) else None
            )
            recruitment[name] = {
                "n_events": int(np.sum(selected)),
                "model_median_contacts": model,
                "patient_median_contacts": patient_recruitment[name],
                "absolute_error_fraction_of_15": (
                    None if model is None else abs(
                        model - patient_recruitment[name]
                    ) / bundle["ranks"].shape[1]
                ),
            }
        by_seed[str(seed)] = {
            "n_formal_clean_events": int(len(index)),
            "natural_kmeans": {
                key: value for key, value in natural.items()
                if key not in {"valid_event_mask", "cluster_labels"}
            },
            "crossfit_patient_readout": crossfit,
            "recruitment": recruitment,
        }
        valid = natural.get("valid_event_mask")
        if valid is not None:
            pooled_index.extend(index[np.asarray(valid, bool)].tolist())
            pooled_crossfit_labels.extend(consensus[np.asarray(valid, bool)].tolist())
    pooled = natural_kmeans(
        bundle["ranks"][np.asarray(pooled_index, int)],
        np.asarray(pooled_crossfit_labels, int), random_state=0,
    ) if pooled_index else {"status": "INSUFFICIENT_EVENTS", "n_events": 0}
    pooled = {
        key: value for key, value in pooled.items()
        if key not in {"valid_event_mask", "cluster_labels"}
    }
    evaluable = [
        row for row in by_seed.values()
        if row["natural_kmeans"]["status"] == "OK"
    ]
    natural_values = [
        row["natural_kmeans"]["direction_balanced_alignment"]
        for row in evaluable
    ]
    margins = [
        row["crossfit_patient_readout"]["signed_margin"]
        for row in by_seed.values()
        if row["crossfit_patient_readout"]["signed_margin"] is not None
    ]
    recruitment_errors = {}
    for name in ("A", "B"):
        values = [
            row["recruitment"][name]["absolute_error_fraction_of_15"]
            for row in by_seed.values()
            if row["recruitment"][name]["absolute_error_fraction_of_15"] is not None
        ]
        recruitment_errors[name] = float(np.mean(values)) if values else None
    finite_rec = [value for value in recruitment_errors.values() if value is not None]
    balanced_secondary = balanced_network_mode_bootstrap(
        bundle,
        events_per_mode=selection["balanced_events_per_mode_per_network"],
        draws=selection["bootstrap_draws"], seed=selection["bootstrap_seed"],
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "node_field_role": candidate["node_field"].get("role"),
        "n_networks": len(by_seed),
        "n_natural_kmeans_evaluable_networks": len(evaluable),
        "natural_kmeans_by_network": by_seed,
        "natural_kmeans_pooled_event_weighted_descriptive": pooled,
        "natural_balanced_alignment_equal_network": network_bootstrap(
            natural_values,
        ),
        "crossfit_margin_equal_network": network_bootstrap(margins),
        "recruitment_error_equal_network": recruitment_errors,
        "recruitment_worst_mode_error": (
            float(max(finite_rec)) if finite_rec else None
        ),
        "balanced_supervised_mode_bootstrap_secondary_only": balanced_secondary,
    }


def _normalized(values, minimize):
    values = np.asarray(values, float)
    low, high = float(np.min(values)), float(np.max(values))
    if high <= low:
        return np.zeros_like(values)
    scaled = (values - low) / (high - low)
    return scaled if minimize else 1.0 - scaled


def _freeze_candidates(rows):
    baseline = next(row for row in rows if row["candidate_id"] == "edge_noop")
    eligible = [
        row for row in rows if row["candidate_id"] != "edge_noop"
        and row["n_natural_kmeans_evaluable_networks"] >= 2
        and row["natural_kmeans_pooled_event_weighted_descriptive"].get(
            "direction_purity"
        ) is not None
        and row["crossfit_margin_equal_network"] is not None
        and row["recruitment_worst_mode_error"] is not None
    ]
    if len(eligible) < 4:
        raise RuntimeError("fewer than four nonbaseline D6 candidates are evaluable")
    natural = sorted(eligible, key=lambda row: (
        -row["natural_kmeans_pooled_event_weighted_descriptive"]["direction_purity"],
        row["candidate_id"],
    ))
    recruitment = sorted(eligible, key=lambda row: (
        row["recruitment_worst_mode_error"], row["candidate_id"],
    ))
    crossfit = sorted(eligible, key=lambda row: (
        -row["crossfit_margin_equal_network"]["equal_network_mean"],
        row["candidate_id"],
    ))
    natural_error = _normalized([
        row["natural_kmeans_pooled_event_weighted_descriptive"]["direction_purity"]
        for row in eligible
    ], minimize=False)
    recruitment_error = _normalized([
        row["recruitment_worst_mode_error"] for row in eligible
    ], minimize=True)
    crossfit_error = _normalized([
        row["crossfit_margin_equal_network"]["equal_network_mean"]
        for row in eligible
    ], minimize=False)
    compromise = sorted(zip(
        np.sqrt(natural_error ** 2 + recruitment_error ** 2 + crossfit_error ** 2),
        eligible,
    ), key=lambda item: (item[0], item[1]["candidate_id"]))
    category_winners = {
        "warm_baseline": baseline["candidate_id"],
        "best_natural_pooled_purity": natural[0]["candidate_id"],
        "best_recruitment": recruitment[0]["candidate_id"],
        "best_crossfit_patient_margin": crossfit[0]["candidate_id"],
        "pareto_knee": compromise[0][1]["candidate_id"],
    }
    selected = []
    for candidate_id in category_winners.values():
        if candidate_id not in selected:
            selected.append(candidate_id)
    for _, row in compromise:
        if len(selected) >= 5:
            break
        if row["candidate_id"] not in selected:
            selected.append(row["candidate_id"])
    if len(selected) != 5:
        raise RuntimeError("D6 closeout requires five distinct candidates")
    roles = {candidate_id: [] for candidate_id in selected}
    for role, candidate_id in category_winners.items():
        roles.setdefault(candidate_id, []).append(role)
    for candidate_id in selected:
        if not roles[candidate_id]:
            roles[candidate_id].append("pareto_reserve")
    return {
        "candidate_ids": selected,
        "selection_roles": roles,
        "category_winners": category_winners,
        "primary_candidate_id": category_winners["best_natural_pooled_purity"],
    }


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
    if manifest["status"] != "REV10D6_CONTINUOUS_FIELD_SENSITIVITY_LIBRARY_FROZEN":
        raise RuntimeError("D6 manifest is not frozen")
    if summary["status"] != "REV10D6_RETURNED_ONLY_CONTINUOUS_FIELD_SCREEN_COMPLETE":
        raise RuntimeError("D6 aggregate is incomplete")
    contract_path = ROOT / config["inputs"]["contact_contract"]["path"]
    contract = json.loads(contract_path.read_text())
    rows = [
        _candidate_metrics(
            config_path, root, candidate, contract,
            config["search"]["kmeans_selection"],
        )
        for candidate in manifest["candidate_set"]["candidates"]
    ]
    freeze = _freeze_candidates(rows)
    payload = {
        "status": "REV10D6_NATURAL_KMEANS_RESCORING_COMPLETE",
        "contact_split_contract": {
            "method": "alternating within-shaft order; assign on one fold and evaluate on the other, then swap",
            "folds": [fold.tolist() for fold in contact_split_folds(contract)],
            "assignment_and_evaluation_contacts_are_disjoint": True,
        },
        "primary_metric": (
            "per-network KMeans on all formal-clean returned events at natural "
            "mode proportions; networks are equal-weight independent units"
        ),
        "balanced_mode_bootstrap_is_secondary_only": True,
        "fresh_closeout_freeze": freeze,
        "candidate_rows": rows,
        "historical_canary_verdict_is_not_a_final_selection_contract": True,
        "claim_boundary": (
            "fit-network zero-simulation candidate freezing only; no fresh-network, "
            "patient-blind, complete-distribution, or Fig.4 acceptance claim"
        ),
        "inputs": {
            "config": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha256(config_path)},
            "manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": _sha256(manifest_path)},
            "summary": {"path": str(summary_path.relative_to(ROOT)), "sha256": _sha256(summary_path)},
            "contact_contract": {"path": str(contract_path.relative_to(ROOT)), "sha256": _sha256(contract_path)},
        },
    }
    output = root / "repaired_natural_kmeans_selection.json"
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "fresh_closeout_freeze": freeze,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
