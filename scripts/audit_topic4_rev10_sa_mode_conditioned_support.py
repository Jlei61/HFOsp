"""Re-adjudicate frozen continuous-field runs without new simulation."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.aggregate_topic4_rev10_sa_spline_field_search import (  # noqa: E402
    _classifier_from_manifest,
    _worker_complete,
)
from scripts.rescore_topic4_rev10_sa_historical_artifacts import (  # noqa: E402
    load_scoring_contract,
)
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _runtime_provenance,
    _sha256,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_shaft_aware import contract_groups  # noqa: E402
from src.topic4_shaft_aware_direction import (  # noqa: E402
    assign_direction_modes,
    mode_conditioned_joint_support,
)


ROOT = Path(__file__).resolve().parents[1]


def _mode_seed_count(seed_support, mode):
    return int(sum(
        value[mode]["n_joint_in_distribution"] > 0
        for value in seed_support.values()
    ))


def audit(config_path, *, worker_commit, expected_commit,
          minimum_events_per_mode, minimum_seeds_per_mode):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    config_sha = _sha256(config_path)
    output_root = ROOT / config["output_root"]
    manifest_path = output_root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["config"]["sha256"] != config_sha:
        raise RuntimeError("candidate manifest and config differ")
    worker_commit = subprocess.check_output(
        ["git", "rev-parse", worker_commit], cwd=ROOT, text=True,
    ).strip()
    contract = json.loads((ROOT / config["inputs"]["contact_contract"]["path"]).read_text())
    groups = contract_groups(contract)
    names, embedding, _, _ = load_scoring_contract(
        config["inputs"]["shaft_aware_target_npz"]["path"],
        config["inputs"]["shaft_aware_floors"]["path"],
        "FULL_TIMING",
        fixed_events_per_mode=int(
            config["search"]["objective"]["fixed_events_per_mode"]
        ),
    )
    classifier = _classifier_from_manifest(manifest)
    seeds = [int(value) for value in config["search"]["network_seeds"]]
    rows, details, inputs = [], {}, []
    for candidate in manifest["candidate_set"]["candidates"]:
        onset_blocks, seed_support = [], {}
        for seed in seeds:
            stem = output_root / "workers" / f"{candidate['candidate_id']}_seed_{seed}"
            json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
            payload = json.loads(json_path.read_text())
            if not _worker_complete(payload, npz_path, config_sha, worker_commit):
                raise RuntimeError(f"stale worker: {stem}")
            with np.load(npz_path, allow_pickle=False) as loaded:
                worker_names = np.asarray(loaded["contact_names"]).astype(str)
                onsets = np.asarray(loaded["onsets"], float)
            if not np.array_equal(worker_names, names):
                raise RuntimeError(f"contact order changed: {stem}")
            assigned = assign_direction_modes(
                onsets, groups=groups, embedding=embedding,
                classifier=classifier,
            )
            seed_support[str(seed)] = mode_conditioned_joint_support(
                onsets, assigned["labels"], assigned["ood"], groups,
            )
            onset_blocks.append(onsets)
            inputs.append({
                "candidate_id": candidate["candidate_id"], "seed": seed,
                "json_sha256": _sha256(json_path), "npz_sha256": _sha256(npz_path),
            })
        onsets = np.concatenate(onset_blocks) if onset_blocks else np.empty((0, len(names)))
        assigned = assign_direction_modes(
            onsets, groups=groups, embedding=embedding, classifier=classifier,
        )
        support = mode_conditioned_joint_support(
            onsets, assigned["labels"], assigned["ood"], groups,
        )
        seed_counts = {
            mode: _mode_seed_count(seed_support, mode) for mode in ("A", "B")
        }
        eligible = all(
            support[mode]["n_joint_in_distribution"] >= minimum_events_per_mode
            and seed_counts[mode] >= minimum_seeds_per_mode
            for mode in ("A", "B")
        )
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "n_events": int(len(onsets)),
            "mode_A_count": support["A"]["n_events"],
            "mode_B_count": support["B"]["n_events"],
            "mode_A_joint_count": support["A"]["n_joint"],
            "mode_B_joint_count": support["B"]["n_joint"],
            "mode_A_in_distribution_count": support["A"]["n_in_distribution"],
            "mode_B_in_distribution_count": support["B"]["n_in_distribution"],
            "mode_A_joint_in_distribution_count": support["A"]["n_joint_in_distribution"],
            "mode_B_joint_in_distribution_count": support["B"]["n_joint_in_distribution"],
            "mode_A_seed_count": seed_counts["A"],
            "mode_B_seed_count": seed_counts["B"],
            "eligible": bool(eligible),
        })
        details[candidate["candidate_id"]] = {
            "pooled": support,
            "by_seed": seed_support,
        }
    eligible_ids = [row["candidate_id"] for row in rows if row["eligible"]]
    return {
        "status": (
            "MODE_CONDITIONED_JOINT_SUPPORT_FOUND" if eligible_ids
            else "MODE_CONDITIONED_JOINT_SUPPORT_NOT_FOUND"
        ),
        "safe_claim": (
            "direction labels, joint-shaft participation, and patient-support "
            "membership were coupled event by event; pooled joint support alone "
            "does not qualify a two-mode repertoire"
        ),
        "eligible_candidate_ids": eligible_ids,
        "minimum_joint_in_distribution_events_per_mode": minimum_events_per_mode,
        "minimum_seeds_per_mode": minimum_seeds_per_mode,
        "candidate_rows": rows,
        "candidate_details": details,
        "network_seeds": seeds,
        "inputs": {
            "config": {"path": str(config_path.relative_to(ROOT)), "sha256": config_sha},
            "manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": _sha256(manifest_path)},
            "workers": inputs,
        },
        "provenance": _runtime_provenance(expected_commit),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker-commit", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--minimum-events-per-mode", type=int, default=1)
    parser.add_argument("--minimum-seeds-per-mode", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = audit(
        args.config, worker_commit=args.worker_commit,
        expected_commit=args.expected_commit,
        minimum_events_per_mode=args.minimum_events_per_mode,
        minimum_seeds_per_mode=args.minimum_seeds_per_mode,
    )
    atomic_write_json(payload, Path(args.out))
    print(json.dumps({
        "status": payload["status"],
        "eligible_candidate_ids": payload["eligible_candidate_ids"],
        "output": str(Path(args.out).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
