#!/usr/bin/env python3
"""Target-free, exact-equivalent attenuation cache acceleration.

Free rollout is a deterministic function of a model and its observed first
rank.  Many held-out events share the same first-rank contact set.  This
sidecar evaluates each distinct start once and expands the generated sequence
back to the original event order.  It never changes decisions, perturbation
draws, doses, models, or field aggregation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Callable, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_topic5_lbss_attenuation_v0_2 as attenuation_base  # noqa: E402
import run_topic5_multiscale_attenuation_v0_5 as attenuation  # noqa: E402
from build_topic5_multiscale_fields_v0_5 import sha256_file  # noqa: E402
from src.topic5_rnn_motif_v0_4 import rollout_with_size_head as original_rollout  # noqa: E402


DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
DEFAULT_OLD = ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3"


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def deduplicated_rollout_with_size_head(
    model,
    size_head,
    starts: Sequence[np.ndarray],
    device: torch.device,
    rollout_fn: Callable = original_rollout,
) -> list[list[list[int]]]:
    """Return the exact per-event schema after evaluating each unique start once."""
    keys = [tuple(np.asarray(start, dtype=int).tolist()) for start in starts]
    unique: dict[tuple[int, ...], np.ndarray] = {}
    for key, start in zip(keys, starts):
        unique.setdefault(key, np.asarray(start, dtype=int))
    generated = rollout_fn(model, size_head, list(unique.values()), device)
    lookup = dict(zip(unique, generated))
    return [
        [list(map(int, rank_set)) for rank_set in lookup[key]]
        for key in keys
    ]


def annotate_cache(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    payload["rollout_dedup_contract"] = (
        "DETERMINISTIC_SAME_MODEL_SAME_FIRST_RANK_EXACT_EXPANSION"
    )
    payload["rollout_dedup_producer_sha256"] = sha256_file(Path(__file__).resolve())
    payload["target_values_read"] = False
    temporary = path.with_name(path.name + f".tmp.hotfill.{os.getpid()}")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"), allow_nan=True)
    os.replace(temporary, path)
    return {
        "path": str(path), "sha256": sha256_file(path),
        "fit_id": path.parents[1].name, "target": path.parent.name,
        "seed": int(path.stem.replace(".json", "").replace("seed", "")),
    }


def worker(job: tuple[str, str, str, str]) -> dict:
    out_string, metrics_string, target, device_string = job
    out, metrics_path = Path(out_string), Path(metrics_string)
    cache = attenuation.unit_cache(out, metrics_path, target)
    if cache.exists():
        return {"status": "ALREADY_PRESENT", "path": str(cache)}
    torch.set_num_threads(2)
    attenuation_base.rollout_with_size_head = deduplicated_rollout_with_size_head
    attenuation.evaluate_unit(out, metrics_path, target, torch.device(device_string))
    evidence = annotate_cache(cache)
    return {"status": "HOTFILLED", **evidence}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-first-fits", type=int, default=8)
    parser.add_argument(
        "--include-existing-matches", action="store_true",
        help=(
            "allow exact-parity cache recovery for a matched-local unit whose "
            "deterministic match manifest already exists"
        ),
    )
    args = parser.parse_args()
    out, old = args.out_root.resolve(), args.old_root.resolve()
    if os.environ.get("TOPIC5_V0_5_TARGET_SEALED") != "1":
        raise RuntimeError("attenuation hotfill must run inside the physical target embargo")
    if (out / "TARGET_UNSEAL_AUTHORIZATION.json").exists():
        raise RuntimeError("attenuation hotfill is forbidden after target authorization")
    parity_path = out / "ATTENUATION_HOTFILL_EXACT_PARITY.json"
    parity = json.loads(parity_path.read_text())
    if not (
        parity.get("status") == "PASS_TARGET_FREE"
        and parity.get("target_values_read") is False
        and parity.get("events") == 1492
        and parity.get("mismatches") == 0
    ):
        raise RuntimeError("deduplicated rollout exact-parity evidence is missing or invalid")
    paths = attenuation.metrics_paths(out, old)
    fit_order = []
    for metrics_path, _target in paths:
        fit_id = json.loads(metrics_path.read_text())["fit_id"]
        if fit_id not in fit_order:
            fit_order.append(fit_id)
    allowed = set(fit_order[int(args.skip_first_fits):])
    jobs = []
    for metrics_path, target in paths:
        metrics = json.loads(metrics_path.read_text())
        cache = attenuation.unit_cache(out, metrics_path, target)
        # Stay well ahead of the original ordered executor.  Any unit that has
        # begun a matched-local search is left entirely to the original worker.
        match = out / "attenuation/matched_local" / metrics["fit_id"] / f"seed{metrics['seed']}" / "match.json"
        if (
            metrics["fit_id"] in allowed
            and not cache.exists()
            and (args.include_existing_matches or not match.exists())
        ):
            jobs.append((str(out), str(metrics_path), target, args.device))
    active = {
        "contract": "topic5_attenuation_rollout_dedup_hotfill_v0_5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "producer_script": str(Path(__file__).resolve()),
        "producer_script_sha256": sha256_file(Path(__file__).resolve()),
        "jobs": len(jobs), "workers": min(max(1, args.workers), 4),
        "skip_first_fits": int(args.skip_first_fits),
        "include_existing_matches": bool(args.include_existing_matches),
        "exact_parity_sha256": sha256_file(parity_path),
        "target_values_read": False,
    }
    write_json(out / "ATTENUATION_HOTFILL_ACTIVE.json", active)
    rows = []
    with ProcessPoolExecutor(max_workers=active["workers"]) as executor:
        futures = [executor.submit(worker, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if index % 20 == 0:
                print(json.dumps({"completed": index, "total": len(jobs)}), flush=True)
    # This runner may be invoked in sequential target-free passes over
    # disjoint fit ranges.  Normalize provenance over every cache that was
    # produced by this exact deduplicated-rollout contract so the final marker
    # is cumulative and the pre-unseal guard can verify one producer hash.
    cumulative_evidence = []
    for cache in sorted((out / "attenuation/unit_cache").glob("**/*.json.gz")):
        with gzip.open(cache, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("rollout_dedup_contract") != (
            "DETERMINISTIC_SAME_MODEL_SAME_FIRST_RANK_EXACT_EXPANSION"
        ):
            continue
        cumulative_evidence.append(annotate_cache(cache))
    complete = {
        **active,
        "status": "PASS_TARGET_FREE",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "hotfilled": len(cumulative_evidence),
        "current_run_hotfilled": sum(
            row["status"] == "HOTFILLED" for row in rows
        ),
        "already_present": sum(row["status"] == "ALREADY_PRESENT" for row in rows),
        "cumulative_provenance_normalization": True,
        "cache_evidence": cumulative_evidence,
    }
    write_json(out / "ATTENUATION_HOTFILL_COMPLETE.json", complete)
    (out / "ATTENUATION_HOTFILL_ACTIVE.json").unlink(missing_ok=True)
    print(json.dumps({key: value for key, value in complete.items() if key != "cache_evidence"}, indent=2))


if __name__ == "__main__":
    main()
