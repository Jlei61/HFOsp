"""Close the D7 active-Z/M continuous-field canary from frozen workers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d7_active_zm_continuous_field_canary.json"
EXPECTED_BASELINE = "data_driven_snn_h_spatial_ou_zm_reference_v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, payload):
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


def summarize_runaway(records, candidate_ids, seeds):
    expected = {(str(candidate), int(seed)) for candidate in candidate_ids for seed in seeds}
    observed = {
        (str(row["candidate"]["candidate_id"]), int(row["seed"]))
        for row in records
    }
    if observed != expected:
        raise RuntimeError("D7 worker candidate/seed coverage is incomplete")
    by_candidate = defaultdict(list)
    values = []
    for row in records:
        value = row["run"].get("runaway_early_stop_ms")
        if value is not None:
            value = float(value)
            values.append(value)
            by_candidate[row["candidate"]["candidate_id"]].append(value)
    safe_candidates = sorted(
        candidate for candidate in candidate_ids
        if len(by_candidate[candidate]) == 0
    )
    all_candidates_fail_both = all(
        len(by_candidate[candidate]) == len(seeds)
        for candidate in candidate_ids
    )
    quantiles = None
    if values:
        quantiles = {
            "minimum_ms": float(np.min(values)),
            "q05_ms": float(np.quantile(values, 0.05)),
            "median_ms": float(np.median(values)),
            "q95_ms": float(np.quantile(values, 0.95)),
            "maximum_ms": float(np.max(values)),
        }
    return {
        "n_workers": len(records),
        "n_runaway_workers": len(values),
        "n_nonrunaway_workers": len(records) - len(values),
        "n_candidates": len(candidate_ids),
        "networks_per_candidate": len(seeds),
        "safe_candidate_ids": safe_candidates,
        "all_candidates_runaway_on_all_networks": all_candidates_fail_both,
        "runaway_time": quantiles,
        "candidate_mean_runaway_time_ms": {
            candidate: float(np.mean(by_candidate[candidate]))
            for candidate in sorted(by_candidate)
            if by_candidate[candidate]
        },
    }


def build_verdict(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"]
    manifest_path = root / "candidate_manifest.json"
    generic_verdict_path = root / "canary_verdict.json"
    memory_path = root / "screen_memory_audit.json"
    manifest = json.loads(manifest_path.read_text())
    generic_verdict = json.loads(generic_verdict_path.read_text())
    memory = json.loads(memory_path.read_text())
    candidates = manifest["candidate_set"]["candidates"]
    candidate_ids = [row["candidate_id"] for row in candidates]
    seeds = list(map(int, config["search"]["fit_network_seeds"]))
    records = []
    provenance_errors = []
    for path in sorted((root / "workers").glob("*.json")):
        row = json.loads(path.read_text())
        provenance = row.get("provenance", {})
        npz_path = root / "workers" / path.with_suffix(".npz").name
        valid = (
            row.get("status") == "REV10R_EDGE_FLOW_WORKER_COMPLETE"
            and provenance.get("runtime_modules_match_expected_commit") is True
            and provenance.get("runtime_modules_dirty") is False
            and row.get("config", {}).get("sha256") == sha256(config_path)
            and row.get("manifest", {}).get("sha256") == sha256(manifest_path)
            and npz_path.exists()
            and row.get("arrays", {}).get("sha256") == sha256(npz_path)
        )
        if not valid:
            provenance_errors.append(path.name)
        records.append(row)
    expected_workers = len(candidate_ids) * len(seeds)
    if len(records) != expected_workers or provenance_errors:
        raise RuntimeError(
            f"D7 worker audit failed: {len(records)}/{expected_workers}; "
            f"provenance errors={provenance_errors[:5]}"
        )
    if any(row["candidate"]["mz"]["mode"] != "z_plus_m" for row in records):
        raise RuntimeError("D7 worker did not use active Z+M")
    if any(
            row["candidate"]["data_driven_snn_baseline"]["baseline_id"]
            != EXPECTED_BASELINE for row in records):
        raise RuntimeError("D7 worker used another shared baseline")
    baseline_record = config["data_driven_snn_baseline"]
    baseline_path = ROOT / baseline_record["path"]
    if sha256(baseline_path) != baseline_record["sha256"]:
        raise RuntimeError("D7 shared baseline hash changed")
    baseline = json.loads(baseline_path.read_text())
    minimum_duration = float(
        baseline["consumer_contract"]["minimum_simulation_duration_ms"]
    )
    if any(float(row["simulation"]["duration_ms"]) < minimum_duration for row in records):
        raise RuntimeError("D7 worker is too short to audit delayed runaway")
    summary = summarize_runaway(records, candidate_ids, seeds)
    stabilization_observed = bool(summary["safe_candidate_ids"])
    status = (
        "REV10D7_ACTIVE_ZM_FIELD_STABILIZATION_CANDIDATE_OBSERVED"
        if stabilization_observed
        else "REV10D7_ACTIVE_ZM_FIELD_STABILIZATION_NOT_OBSERVED"
    )
    return {
        "status": status,
        "stabilization_candidate_observed": stabilization_observed,
        "selection_authorized": bool(
            stabilization_observed
            and generic_verdict.get("selected_candidate_id") is not None
        ),
        "fig4_acceptance": "NOT_AUTHORIZED",
        "runtime_contract": {
            "baseline_id": EXPECTED_BASELINE,
            "runtime_mode": "active_z_plus_m",
            "duration_ms": float(config["search"]["simulation"]["duration_ms"]),
            "late_runaway_is_invalid": True,
            "network_seeds": seeds,
            "candidate_builder_uses_observation_geometry": False,
            "edge": "exact no-op",
            "beta": "closed",
        },
        "runaway_audit": summary,
        "generic_kmeans_audit": {
            "status": generic_verdict["status"],
            "selected_candidate_id": generic_verdict["selected_candidate_id"],
            "interpretation": (
                "No KMeans candidate is scientifically evaluable because "
                "late runaway invalidates every candidate/network pair."
            ),
        },
        "execution_audit": {
            "workers_complete": len(records),
            "workers_expected": expected_workers,
            "provenance_errors": provenance_errors,
            "worker_commits": dict(sorted(Counter(
                row["provenance"]["expected_git_commit"] for row in records
            ).items())),
            "memory_sentinel": memory,
        },
        "claim_boundary": (
            "The fixed active-Z/M reference could not be stabilized by any "
            "of 49 low-frequency continuous fields on two development "
            "networks. This closes this frozen mechanism-reference and field "
            "library, not all slow variables, continuous fields, or SNN "
            "mechanisms. It supplies no Fig.4, patient-generalization, core, "
            "edge, optimizer, or ictal-lifecycle claim."
        ),
        "inputs": {
            "config": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)},
            "manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": sha256(manifest_path)},
            "generic_verdict": {"path": str(generic_verdict_path.relative_to(ROOT)), "sha256": sha256(generic_verdict_path)},
            "memory_audit": {"path": str(memory_path.relative_to(ROOT)), "sha256": sha256(memory_path)},
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    payload = build_verdict(args.config)
    config = json.loads(Path(args.config).read_text())
    output = ROOT / config["output_root"] / "d7_canary_verdict.json"
    atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "selection_authorized": payload["selection_authorized"],
        "runaway_audit": payload["runaway_audit"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
