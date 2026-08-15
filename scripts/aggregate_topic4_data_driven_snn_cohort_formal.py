#!/usr/bin/env python3
"""Select, confirm and adjudicate the formal data-driven SNN cohort.

Stage A and B only ever see the patient training split; the held-out blocks
enter at stage C and are never used to choose a candidate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_cohort_formal_scoring import (  # noqa: E402
    adjudicate,
    cohort_summary,
    confirm_subject,
    endpoint_statistic,
    score_readout,
)

DEFAULT_CONFIG = ROOT / "config/topic4_data_driven_snn_cohort_formal_v1.json"
TRACKED_MODULES = (
    "scripts/aggregate_topic4_data_driven_snn_cohort_formal.py",
    "src/topic4_cohort_formal_scoring.py",
    "src/topic4_data_driven_cohort_formal.py",
    "src/topic4_canonical_shaft_layout.py",
    "src/topic4_data_driven_cohort.py",
)
MODES = ("ta", "tb")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        with open(temporary, "w") as stream:
            json.dump(_json_ready(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".csv.tmp")
    os.close(handle)
    try:
        with open(temporary, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _provenance(config_path: Path, expected_commit: str) -> dict:
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *TRACKED_MODULES,
         str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "expected_git_commit": commit,
        "runtime_modules_dirty": bool(dirty),
        "runtime_file_sha256": {
            name: _sha256(ROOT / name) for name in TRACKED_MODULES
        },
        "config_sha256": _sha256(config_path),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


class Cohort:
    """Frozen patient targets, layouts and the scoring contract they imply."""

    def __init__(self, config: dict):
        self.config = config
        self.output_root = ROOT / config["output_root"]
        self.layout_audit = json.loads(
            (self.output_root / "cohort_layout_audit.json").read_text()
        )
        self.target_root = ROOT / config["inputs"]["source_target_root"]["path"]
        self.subjects = self.layout_audit["subjects"]
        self.search = config["search"]

    def scorer_kwargs(self, subject: dict, split: str) -> dict:
        subject_id = subject["subject_id"]
        target_json = json.loads(
            (self.target_root / f"{subject_id}.json").read_text()
        )
        with np.load(
            self.target_root / f"{subject_id}_target.npz", allow_pickle=False,
        ) as loaded:
            names = [str(value) for value in loaded["contact_order"]]
            payload = {
                "contact_order": names,
                "profiles": np.asarray(loaded[f"{split}_profiles"], float),
                "recruitment": np.asarray([
                    loaded[f"{split}_{mode}_recruitment"] for mode in MODES
                ], float),
                "precedence": np.asarray([
                    loaded[f"{split}_{mode}_precedence"] for mode in MODES
                ], float),
            }
            centers = np.asarray(loaded["kmeans_centers"], float)
        return {
            "target": payload,
            "contact_names": names,
            "patient_centers": centers,
            "ood_threshold": float(target_json["target"]["train_distance_q95"]),
            "minimum_contacts": int(self.search["minimum_contacts_per_event"]),
            "minimum_events_per_mode": int(self.search["minimum_events_per_mode"]),
            "kmeans_seed": int(self.search["kmeans_seed"]),
            "kmeans_n_init": int(self.search["kmeans_n_init"]),
        }

    def permutations(self, subject: dict) -> np.ndarray:
        with np.load(
            ROOT / subject["layout_npz"], allow_pickle=False,
        ) as loaded:
            return np.asarray(loaded["within_shaft_null_permutations"], int)

    def worker_ranks(self, worker_npz: Path, layout: str, index: int) -> np.ndarray:
        with np.load(worker_npz, allow_pickle=False) as loaded:
            key = f"{layout}_{index:02d}_ranks"
            if key not in loaded:
                raise KeyError(f"worker output is missing {key}: {worker_npz}")
            return np.asarray(loaded[key], float)


def _worker_paths(output_root: Path, candidate: str, seed: int) -> tuple[Path, Path]:
    stem = f"{candidate}_seed_{seed}"
    return (output_root / "workers" / f"{stem}.json",
            output_root / "workers" / f"{stem}.npz")


def _load_valid_worker(output_root: Path, candidate: str, seed: int,
                       expected_commit: str) -> dict | None:
    json_path, npz_path = _worker_paths(output_root, candidate, seed)
    if not json_path.exists() or not npz_path.exists():
        return None
    payload = json.loads(json_path.read_text())
    provenance = payload.get("provenance", {})
    if (payload.get("candidate_id") != candidate or payload.get("seed") != seed
            or provenance.get("expected_git_commit") != expected_commit
            or not provenance.get("runtime_modules_match_expected_commit")
            or provenance.get("runtime_modules_dirty")):
        raise RuntimeError(f"worker provenance is not frozen: {json_path}")
    if payload.get("output_npz_sha256") != _sha256(npz_path):
        raise RuntimeError(f"worker npz hash does not match its json: {npz_path}")
    if payload.get("status") == "INVALID_RUNAWAY":
        return {"status": "INVALID_RUNAWAY", "payload": payload, "npz": npz_path}
    if payload.get("status") != "COMPLETE":
        raise RuntimeError(f"unexpected worker status in {json_path}")
    return {"status": "COMPLETE", "payload": payload, "npz": npz_path}


def _select(cohort: Cohort, candidates: list[str], seeds: list[int],
            expected_commit: str, *, restrict: dict[str, list[str]] | None = None
            ) -> dict:
    """Score every allowed candidate for every subject on the training split."""
    scores: dict[str, dict[str, list[float]]] = {}
    runaway = set()
    missing = []
    for candidate in candidates:
        for seed in seeds:
            worker = _load_valid_worker(
                cohort.output_root, candidate, seed, expected_commit,
            )
            if worker is None:
                missing.append({"candidate": candidate, "seed": seed})
                continue
            for index, subject in enumerate(cohort.subjects):
                subject_id = subject["subject_id"]
                if restrict is not None and candidate not in restrict.get(subject_id, []):
                    continue
                if worker["status"] == "INVALID_RUNAWAY":
                    runaway.add(candidate)
                    scores.setdefault(subject_id, {}).setdefault(
                        candidate, []
                    ).append(2.0)
                    continue
                ranks = cohort.worker_ranks(worker["npz"], "canonical", index)
                score = score_readout(
                    ranks, include_natural_kmeans=False,
                    **cohort.scorer_kwargs(subject, "train"),
                )
                scores.setdefault(subject_id, {}).setdefault(candidate, []).append(
                    float(score["selection_score"])
                )
    if missing:
        raise RuntimeError(f"{len(missing)} selection workers are missing: {missing[:5]}")
    ranked = {}
    for subject_id, per_candidate in scores.items():
        order = sorted(
            per_candidate.items(), key=lambda item: (float(np.mean(item[1])), item[0]),
        )
        ranked[subject_id] = [
            {
                "candidate_id": candidate,
                "mean_selection_score": float(np.mean(values)),
                "n_seeds": len(values),
            }
            for candidate, values in order
        ]
    return {"ranked": ranked, "runaway_candidates": sorted(runaway)}


def run_stage_a(cohort: Cohort, expected_commit: str) -> dict:
    manifest = json.loads((cohort.output_root / "candidate_manifest.json").read_text())
    candidates = [row["candidate_id"] for row in manifest["candidate_set"]["candidates"]]
    seeds = [int(seed) for seed in cohort.config["search"]["fit_network_seeds"]]
    selection = _select(cohort, candidates, seeds, expected_commit)
    shortlist = {
        subject_id: [row["candidate_id"] for row in rows[:2]]
        for subject_id, rows in selection["ranked"].items()
    }
    union = sorted({candidate for rows in shortlist.values() for candidate in rows})
    return {
        "schema_version": "topic4_formal_cohort_stage_a_v1",
        "status": "STAGE_A_SHORTLIST_READY",
        "split_used": "patient train blocks only",
        "seeds": seeds,
        "n_candidates_scored": len(candidates),
        "runaway_candidates": selection["runaway_candidates"],
        "per_subject_ranking": selection["ranked"],
        "per_subject_shortlist": shortlist,
        "stage_b_candidates": union,
    }


def run_stage_b(cohort: Cohort, expected_commit: str) -> dict:
    stage_a = json.loads((cohort.output_root / "stage_a_selection.json").read_text())
    shortlist = stage_a["per_subject_shortlist"]
    seeds = [
        int(seed) for key in ("fit_network_seeds", "selection_network_seeds")
        for seed in cohort.config["search"][key]
    ]
    selection = _select(
        cohort, stage_a["stage_b_candidates"], seeds, expected_commit,
        restrict=shortlist,
    )
    selected = {
        subject_id: rows[0]["candidate_id"]
        for subject_id, rows in selection["ranked"].items()
    }
    return {
        "schema_version": "topic4_formal_cohort_stage_b_v1",
        "status": "STAGE_B_SELECTION_READY",
        "split_used": "patient train blocks only",
        "seeds": seeds,
        "per_subject_ranking": selection["ranked"],
        "per_subject_selected_candidate": selected,
        "stage_c_candidates": sorted(set(selected.values())),
        "selected_candidate_counts": {
            candidate: int(sum(1 for value in selected.values() if value == candidate))
            for candidate in sorted(set(selected.values()))
        },
    }


def _confirm_layout(cohort: Cohort, subject: dict, index: int, candidate: str,
                    seeds: list[int], layout: str, expected_commit: str) -> dict | None:
    kwargs = cohort.scorer_kwargs(subject, "heldout")
    permutations = cohort.permutations(subject)
    per_seed = []
    for seed in seeds:
        worker = _load_valid_worker(
            cohort.output_root, candidate, seed, expected_commit,
        )
        if worker is None:
            raise RuntimeError(f"confirmation worker missing: {candidate} seed {seed}")
        if worker["status"] == "INVALID_RUNAWAY":
            per_seed.append({
                "seed": seed, "status": "INVALID_RUNAWAY",
                "observed_weakest_mode_loss": 1.0, "null_median": 1.0,
                "delta_null_median_minus_observed": 0.0,
                "subject_endpoint_pass": False,
                "natural_kmeans": {"same_network_k2": False},
                "n_in_distribution_events": 0,
            })
            continue
        try:
            ranks = cohort.worker_ranks(worker["npz"], layout, index)
        except KeyError:
            return None
        row = confirm_subject(
            ranks, permutations=permutations,
            minimum_events=int(cohort.search["minimum_events_per_mode"]),
            minimum_seed_ami=float(
                cohort.config["endpoint"]["same_network_k2"]["requires"][
                    "natural_seed_ami_median_min"
                ]
            ),
            **kwargs,
        )
        per_seed.append({"seed": seed, **row})
    deltas = np.asarray(
        [row["delta_null_median_minus_observed"] for row in per_seed], float,
    )
    evaluable = int(sum(row.get("status") == "EVALUABLE" for row in per_seed))
    return {
        "subject_id": subject["subject_id"],
        "layout": layout,
        "candidate_id": candidate,
        "n_contacts": int(subject["n_contacts"]),
        "n_seeds": len(per_seed),
        "n_evaluable_seeds": evaluable,
        "delta_null_median_minus_observed": float(np.median(deltas)),
        "observed_weakest_mode_loss": float(np.median([
            row["observed_weakest_mode_loss"] for row in per_seed
        ])),
        "null_median": float(np.median([row["null_median"] for row in per_seed])),
        "permutation_p_median": float(np.median([
            row.get("permutation_p", 1.0) for row in per_seed
        ])),
        "minimum_reachable_p": float(per_seed[0].get("minimum_reachable_p", 1.0)),
        "n_in_distribution_events": float(np.median([
            row.get("n_in_distribution_events") or 0 for row in per_seed
        ])),
        "subject_endpoint_pass": bool(
            evaluable >= max(3, len(per_seed) - 1) and float(np.median(deltas)) > 0.0
        ),
        "natural_kmeans": {
            "same_network_k2": bool(
                sum(bool(row["natural_kmeans"].get("same_network_k2"))
                    for row in per_seed) >= max(3, len(per_seed) - 1)
            ),
            "n_seeds_with_same_network_k2": int(sum(
                bool(row["natural_kmeans"].get("same_network_k2")) for row in per_seed
            )),
        },
        "per_seed": per_seed,
    }


def run_stage_c(cohort: Cohort, expected_commit: str) -> dict:
    stage_b = json.loads((cohort.output_root / "stage_b_selection.json").read_text())
    selected = stage_b["per_subject_selected_candidate"]
    seeds = [int(seed) for seed in cohort.config["search"]["confirmation_network_seeds"]]
    canonical, real = [], []
    for index, subject in enumerate(cohort.subjects):
        subject_id = subject["subject_id"]
        candidate = selected[subject_id]
        canonical.append(_confirm_layout(
            cohort, subject, index, candidate, seeds, "canonical", expected_commit,
        ))
        if subject["in_real_geometry_sensitivity_cohort"]:
            row = _confirm_layout(
                cohort, subject, index, candidate, seeds, "real", expected_commit,
            )
            if row is not None:
                real.append(row)
    per_seed_pass = {
        str(seed): [
            bool(row["per_seed"][position]["subject_endpoint_pass"])
            for row in canonical
        ]
        for position, seed in enumerate(seeds)
    }
    endpoint = cohort.config["endpoint"]
    summary = cohort_summary(
        canonical, real,
        pass_fraction_min=float(endpoint["cohort_primary"]["pass_fraction_min"]),
        alpha=float(endpoint["cohort_primary"]["alpha"]),
        per_seed_pass=per_seed_pass,
    )
    verdict = adjudicate(
        summary,
        same_network_k2_min=float(
            endpoint["same_network_k2"]["cohort_fraction_min"]
        ),
    )
    deltas = np.asarray(
        [row["delta_null_median_minus_observed"] for row in canonical], float,
    )
    median = float(np.median(deltas))
    representative = canonical[int(np.argmin(np.abs(deltas - median)))]
    return {
        "schema_version": "topic4_formal_cohort_stage_c_v1",
        "status": verdict["status"],
        "verdict": verdict,
        "denominators": cohort.layout_audit["denominators"],
        "split_used": "patient held-out recording blocks",
        "confirmation_seeds": seeds,
        "selected_candidate_counts": stage_b["selected_candidate_counts"],
        "cohort": summary,
        "representative_subject": {
            "subject_id": representative["subject_id"],
            "candidate_id": representative["candidate_id"],
            "delta_null_median_minus_observed": representative[
                "delta_null_median_minus_observed"
            ],
            "chosen_by": "closest to the cohort median null-relative delta",
        },
        "canonical_subjects": canonical,
        "real_geometry_subjects": real,
    }


STAGES = {"a": run_stage_a, "b": run_stage_b, "c": run_stage_c}
OUTPUTS = {
    "a": "stage_a_selection.json",
    "b": "stage_b_selection.json",
    "c": "cohort_result.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    cohort = Cohort(config)
    commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    payload = STAGES[args.stage](cohort, commit)
    payload["config"] = {
        "path": str(config_path.relative_to(ROOT)),
        "sha256": _sha256(config_path),
    }
    payload["provenance"] = _provenance(config_path, args.expected_commit)
    _atomic_json(cohort.output_root / OUTPUTS[args.stage], payload)

    if args.stage == "c":
        rows = [
            {
                "subject_id": row["subject_id"],
                "candidate_id": row["candidate_id"],
                "n_evaluable_seeds": row["n_evaluable_seeds"],
                "observed_weakest_mode_loss": f"{row['observed_weakest_mode_loss']:.6f}",
                "null_median": f"{row['null_median']:.6f}",
                "delta": f"{row['delta_null_median_minus_observed']:.6f}",
                "permutation_p_median": f"{row['permutation_p_median']:.6f}",
                "minimum_reachable_p": f"{row['minimum_reachable_p']:.6f}",
                "subject_endpoint_pass": str(row["subject_endpoint_pass"]),
                "same_network_k2": str(row["natural_kmeans"]["same_network_k2"]),
                "median_in_distribution_events": f"{row['n_in_distribution_events']:.1f}",
            }
            for row in payload["canonical_subjects"]
        ]
        _atomic_csv(cohort.output_root / "cohort_subjects_canonical.csv", rows)
        print(json.dumps({
            "status": payload["status"],
            "pass_fraction": payload["cohort"]["pass_fraction"],
            "median_delta": payload["cohort"]["primary_test"]["median_delta"],
            "wilcoxon_p": payload["cohort"]["primary_test"]["wilcoxon_p"],
            "same_network_k2_fraction": payload["cohort"]["same_network_k2_fraction"],
            "representative": payload["representative_subject"]["subject_id"],
        }, indent=2))
    else:
        print(json.dumps({
            "status": payload["status"],
            "next_candidates": payload.get("stage_b_candidates")
            or payload.get("stage_c_candidates"),
        }, indent=2))


if __name__ == "__main__":
    main()
