"""Aggregate per-network detailed rev9 edge-structure audits."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_rev9_factorial.json"
DEFAULT_DIR = ROOT / (
    "results/topic4_sef_hfo/data_driven_core_field_rev9/edge_structure_detail")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True,
        stderr=subprocess.DEVNULL).strip()


def _paired_median_interval(values, *, seed, repeats=4000):
    values = np.asarray(values, float)
    estimate = np.nanmedian(values, axis=0)
    rng = np.random.default_rng(int(seed))
    draws = np.stack([
        np.nanmedian(values[rng.integers(0, len(values), len(values))], axis=0)
        for _ in range(int(repeats))])
    return dict(
        estimate=estimate.tolist(),
        interval_95=np.nanquantile(draws, [0.025, 0.975], axis=0).tolist(),
        n_networks=int(len(values)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input-dir", default=str(DEFAULT_DIR / "workers"))
    parser.add_argument("--out", default=str(DEFAULT_DIR / "edge_structure_detail_summary.json"))
    args = parser.parse_args()
    commit_at_start = _git("rev-parse", "HEAD")
    producer_sha_at_start = _sha256(__file__)
    config = json.loads(Path(args.config).read_text())
    paths = [Path(args.input_dir) / f"seed{int(seed)}.json" for seed in config["seeds"]]
    if not all(path.exists() for path in paths):
        missing = [str(path) for path in paths if not path.exists()]
        raise FileNotFoundError(f"missing edge structure workers: {missing}")
    workers = [json.loads(path.read_text()) for path in paths]
    if any(row["status"] != "REV9_EDGE_STRUCTURE_DETAIL_WORKER_COMPLETE"
           for row in workers):
        raise RuntimeError("one or more structure workers are incomplete")
    labels = workers[0]["labels"]
    if any(row["labels"] != labels for row in workers):
        raise RuntimeError("worker component labels differ")
    fields = (
        "old_flow_share", "new_flow_share", "flow_ratio",
        "old_pair_delay_ms", "new_pair_delay_ms", "pair_delay_delta_ms",
        "group_outgoing_ratio", "old_group_target_delay_ms",
        "new_group_target_delay_ms", "group_target_delay_delta_ms",
    )
    summaries = {}
    for index, field in enumerate(fields):
        summaries[field] = _paired_median_interval(
            [row["summary"][field] for row in workers],
            seed=20262000 + index)
    summaries["outgoing_log_ratio_vs_h_spearman"] = _paired_median_interval(
        [row["summary"]["outgoing_log_ratio_vs_h_spearman"] for row in workers],
        seed=20262100)

    ratio = np.asarray(summaries["flow_ratio"]["estimate"], float)
    outgoing = np.asarray(summaries["group_outgoing_ratio"]["estimate"], float)
    delay = np.asarray(summaries["group_target_delay_delta_ms"]["estimate"], float)
    background = len(labels) - 1
    component_indices = list(range(background))
    findings = dict(
        component_self_flow_percent=[
            float(100.0 * (ratio[index, index] - 1.0)) for index in component_indices],
        background_source_to_component_target_percent=[
            float(100.0 * (ratio[index, background] - 1.0))
            for index in component_indices],
        component_source_to_background_target_percent=[
            float(100.0 * (ratio[background, index] - 1.0))
            for index in component_indices],
        source_group_outgoing_percent=(100.0 * (outgoing - 1.0)).tolist(),
        target_group_weighted_delay_delta_ms=delay.tolist(),
        interpretation=(
            "the frozen edge map is field-assortative recurrent redistribution: it "
            "preserves each target's incoming-E budget while changing source influence; "
            "it is not a new topology and not a strictly local radial core"),
    )
    payload = dict(
        status="REV9_EDGE_STRUCTURE_DETAIL_COMPLETE",
        scientific_role="paired 12-network zero-integration structural sidecar",
        alpha=float(workers[0]["alpha"]), beta=0.0, labels=labels,
        direction_contract=workers[0]["direction_contract"],
        membership_contract=workers[0]["membership_contract"],
        summaries=summaries, findings=findings,
        conservation=dict(
            max_incoming_error=float(max(
                row["summary"]["incoming_max_abs_error"] for row in workers)),
            max_total_weight_relative_error=float(max(
                row["summary"]["total_weight_relative_error"] for row in workers))),
        workers=[{
            "seed": row["seed"], "path": str(path), "sha256": _sha256(path),
            "cache_path": row["network"]["cache_path"],
            "cache_sha256": row["network"]["cache_sha256"],
        } for row, path in zip(workers, paths)],
        provenance=dict(
            git_commit_at_start=commit_at_start,
            producer_sha256_at_start=producer_sha_at_start,
            relevant_modules_dirty_at_start=bool(_git(
                "status", "--porcelain", "--",
                "scripts/aggregate_topic4_rev9_edge_structure_detail.py",
                "src/topic4_rev9_edge_structure.py"))),
    )
    atomic_write_json(payload, args.out)
    print(json.dumps({
        "status": payload["status"], "n_networks": len(workers),
        "findings": findings, "out": args.out,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
