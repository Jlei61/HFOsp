"""Aggregate rev10-R with network seeds, not pooled events, as units."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.rescore_topic4_rev10_sa_historical_artifacts import (  # noqa: E402
    load_scoring_contract,
    score_mode_conditioned_events,
)
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _load_json_input,
    _runtime_provenance,
    _sha256,
)
from src.topic4_shaft_aware import (  # noqa: E402
    centered_smooth_max,
    contract_groups,
    contract_pairs,
)
from src.topic4_shaft_aware_direction import (  # noqa: E402
    all_event_shaft_participation,
    assign_direction_modes,
    mode_conditioned_joint_support,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_r_graph_edge_flow.json"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(
            _jsonable(payload), indent=2, sort_keys=True,
        ))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".csv.tmp")
    os.close(fd)
    try:
        with open(temporary, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _classifier_from_manifest(manifest):
    classifier = dict(manifest["direction_classifier"])
    for key in (
        "coef", "class_centers", "class_precisions", "ood_distance_thresholds",
    ):
        classifier[key] = np.asarray(classifier[key], float)
    return classifier


def _worker_complete(payload, npz_path, config_sha, manifest_sha, commit):
    provenance = payload.get("provenance", {})
    return bool(
        payload.get("status") == "REV10R_EDGE_FLOW_WORKER_COMPLETE"
        and payload.get("config", {}).get("sha256") == config_sha
        and payload.get("manifest", {}).get("sha256") == manifest_sha
        and provenance.get("expected_git_commit") == commit
        and provenance.get("runtime_modules_match_expected_commit") is True
        and not provenance.get("runtime_modules_dirty")
        and payload.get("arrays", {}).get("sha256") == _sha256(npz_path)
    )


def _mode_shape_scores(score6, score3):
    output, source = {}, {}
    for mode, name in ((0, "A"), (1, "B")):
        row6 = score6["modes"].get(str(mode), {})
        row3 = score3["modes"].get(str(mode), {})
        if row6.get("status") == "OK":
            output[name] = float(row6["objective"]["mode_score"])
            source[name] = "n6"
        elif row3.get("status") == "OK":
            output[name] = float(row3["objective"]["mode_score"])
            source[name] = "n3_fallback"
        else:
            output[name] = 8.0
            source[name] = "unsupported_penalty"
    return output, source


def _score_seed(onsets, *, classifier, groups, pairs, embedding, targets,
                floors6, floors3, scoring_config, objective):
    onsets = np.asarray(onsets, float)
    if len(onsets):
        assigned = assign_direction_modes(
            onsets, groups=groups, embedding=embedding, classifier=classifier,
        )
        labels = np.asarray(assigned["labels"], int)
        ood = np.asarray(assigned["ood"], bool)
    else:
        labels, ood = np.empty(0, int), np.empty(0, bool)
    support = mode_conditioned_joint_support(onsets, labels, ood, groups)
    icl = np.isfinite(onsets[:, groups["ICL"]]).any(axis=1)
    scl = np.isfinite(onsets[:, groups["SCL"]]).any(axis=1)
    clean = icl & scl & ~ood
    clean_onsets, clean_labels = onsets[clean], labels[clean]
    score6 = score_mode_conditioned_events(
        clean_onsets, clean_labels, groups=groups, pairs=pairs,
        embedding=embedding, targets=targets, floors=floors6,
        config=scoring_config, fixed_events_per_mode=6,
    )
    score3 = score_mode_conditioned_events(
        clean_onsets, clean_labels, groups=groups, pairs=pairs,
        embedding=embedding, targets=targets, floors=floors3,
        config=scoring_config, fixed_events_per_mode=3,
    )
    shape_by_mode, source = _mode_shape_scores(score6, score3)
    required = int(objective[
        "minimum_joint_in_distribution_events_per_mode_per_network"
    ])
    support_deficit = {
        name: max(0.0, required - support[name]["n_joint_in_distribution"])
        / max(1, required)
        for name in ("A", "B")
    }
    tau = float(objective["lse_temperature"])
    shape_lse = centered_smooth_max(list(shape_by_mode.values()), tau)
    support_lse = centered_smooth_max(list(support_deficit.values()), tau)
    ood_fraction = float(np.mean(ood)) if len(ood) else 1.0
    selection_score = (
        shape_lse + float(objective["support_weight"]) * support_lse
        + float(objective["ood_weight"]) * ood_fraction
    )
    return {
        "n_events": int(len(onsets)),
        "n_clean_joint_in_distribution": int(np.sum(clean)),
        "mode_conditioned_joint_support": support,
        "all_event_shaft_participation": all_event_shaft_participation(
            onsets, groups,
        ),
        "ood_fraction": ood_fraction,
        "shape_score_source": source,
        "shape_by_mode": shape_by_mode,
        "shape_lse": float(shape_lse),
        "support_deficit_by_mode": support_deficit,
        "support_lse": float(support_lse),
        "selection_score": float(selection_score),
        "score_n6": score6,
        "score_n3": score3,
    }


def _pareto(rows):
    values = np.asarray([
        [row["mean_network_shape_A"], row["mean_network_shape_B"],
         row["mean_network_support_A"], row["mean_network_support_B"],
         row["mean_network_ood_fraction"]]
        for row in rows
    ])
    output = []
    for index, value in enumerate(values):
        output.append(not any(
            np.all(other <= value) and np.any(other < value)
            for other_index, other in enumerate(values) if other_index != index
        ))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    config_sha = _sha256(config_path)
    commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    output_root = ROOT / config["output_root"]
    manifest_path = output_root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = _sha256(manifest_path)
    if manifest["config"]["sha256"] != config_sha:
        raise RuntimeError("rev10-R manifest uses another config")

    contract = _load_json_input(config["inputs"]["contact_contract"])
    groups, pairs = contract_groups(contract), contract_pairs(contract)
    scoring_config = _load_json_input(config["inputs"]["shaft_aware_scoring_config"])
    target_path = config["inputs"]["shaft_aware_target_npz"]["path"]
    floor_path = config["inputs"]["shaft_aware_floors"]["path"]
    names, embedding, targets, floors6 = load_scoring_contract(
        target_path, floor_path, "FULL_TIMING", fixed_events_per_mode=6,
    )
    _, _, _, floors3 = load_scoring_contract(
        target_path, floor_path, "FULL_TIMING", fixed_events_per_mode=3,
    )
    expected_names = np.asarray([
        row["contact_name"] for row in contract["contacts"]
    ]).astype(str)
    if not np.array_equal(names.astype(str), expected_names):
        raise RuntimeError("scoring and contact contracts differ")
    classifier = _classifier_from_manifest(manifest)
    objective = config["search"]["objective"]
    seeds = list(map(int, config["search"]["fit_network_seeds"]))
    rows, details, worker_inputs = [], {}, []
    for candidate in manifest["candidate_set"]["candidates"]:
        by_seed, metadata = {}, []
        for seed in seeds:
            stem = output_root / "workers" / f"{candidate['candidate_id']}_seed_{seed}"
            json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
            payload = json.loads(json_path.read_text())
            if not _worker_complete(
                    payload, npz_path, config_sha, manifest_sha, commit):
                raise RuntimeError(f"stale rev10-R worker: {stem}")
            with np.load(npz_path, allow_pickle=False) as loaded:
                worker_names = np.asarray(loaded["contact_names"]).astype(str)
                onsets = np.asarray(loaded["onsets"], float)
            if not np.array_equal(worker_names, names.astype(str)):
                raise RuntimeError(f"contact order changed: {stem}")
            by_seed[str(seed)] = _score_seed(
                onsets, classifier=classifier, groups=groups, pairs=pairs,
                embedding=embedding, targets=targets, floors6=floors6,
                floors3=floors3, scoring_config=scoring_config,
                objective=objective,
            )
            metadata.append(payload)
            worker_inputs.append({
                "candidate_id": candidate["candidate_id"], "seed": seed,
                "json_sha256": _sha256(json_path), "npz_sha256": _sha256(npz_path),
            })
        values = list(by_seed.values())
        runaway = int(sum(
            payload["run"]["runaway_early_stop_ms"] is not None
            for payload in metadata
        ))
        mean_score = float(np.mean([value["selection_score"] for value in values]))
        if runaway:
            mean_score = 1000.0 + runaway
        row = {
            "candidate_id": candidate["candidate_id"],
            "selection_score_equal_network": mean_score,
            "n_runaway_networks": runaway,
            "total_events_descriptive": int(sum(value["n_events"] for value in values)),
            "mean_network_events": float(np.mean([value["n_events"] for value in values])),
            "mean_network_shape_A": float(np.mean([
                value["shape_by_mode"]["A"] for value in values
            ])),
            "mean_network_shape_B": float(np.mean([
                value["shape_by_mode"]["B"] for value in values
            ])),
            "mean_network_support_A": float(np.mean([
                value["support_deficit_by_mode"]["A"] for value in values
            ])),
            "mean_network_support_B": float(np.mean([
                value["support_deficit_by_mode"]["B"] for value in values
            ])),
            "mean_network_ood_fraction": float(np.mean([
                value["ood_fraction"] for value in values
            ])),
            "networks_with_clean_A": int(sum(
                value["mode_conditioned_joint_support"]["A"][
                    "n_joint_in_distribution"
                ] > 0 for value in values
            )),
            "networks_with_clean_B": int(sum(
                value["mode_conditioned_joint_support"]["B"][
                    "n_joint_in_distribution"
                ] > 0 for value in values
            )),
            "networks_with_both_clean_modes": int(sum(
                all(value["mode_conditioned_joint_support"][name][
                    "n_joint_in_distribution"
                ] > 0 for name in ("A", "B"))
                for value in values
            )),
            "edge_ratio_min": float(min(
                value["edge_audit"]["edge_ratio"]["min"]
                for value in metadata
            )),
            "edge_ratio_max": float(max(
                value["edge_audit"]["edge_ratio"]["max"]
                for value in metadata
            )),
            "max_incoming_E_error": float(max(
                value["edge_audit"]["max_abs_incoming_E_error"]
                for value in metadata
            )),
        }
        rows.append(row)
        details[candidate["candidate_id"]] = {
            "by_seed": by_seed,
            "edge_audit_by_seed": {
                str(payload["seed"]): payload["edge_audit"]
                for payload in metadata
            },
        }
    for row, flag in zip(rows, _pareto(rows)):
        row["pareto_nondominated"] = bool(flag)
    rows.sort(key=lambda row: (
        row["n_runaway_networks"] > 0,
        row["selection_score_equal_network"], row["candidate_id"],
    ))
    baseline = next(row for row in rows if row["candidate_id"] == "edge_noop")
    summary = {
        "status": "REV10R_FIT_SCREEN_COMPLETE",
        "scientific_role": config["scientific_role"],
        "safe_claim": (
            "all candidate scores use equal network weights; event-pooled counts "
            "are descriptive only; mode shape is scored only on joint and "
            "patient-supported events, while absent support remains a penalty"
        ),
        "baseline_candidate_id": "edge_noop",
        "baseline": baseline,
        "diagnostic_best_candidate_id": rows[0]["candidate_id"],
        "candidate_rows": rows,
        "candidate_details": details,
        "network_seeds": seeds,
        "worker_inputs": worker_inputs,
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": config_sha},
        "provenance": _runtime_provenance(args.expected_commit),
    }
    _atomic_csv(output_root / "fit_screen_candidate_summary.csv", rows)
    _atomic_json(output_root / "fit_screen_summary.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "diagnostic_best_candidate_id": summary["diagnostic_best_candidate_id"],
        "baseline_score": baseline["selection_score_equal_network"],
        "best_score": rows[0]["selection_score_equal_network"],
    }, indent=2))


if __name__ == "__main__":
    main()
