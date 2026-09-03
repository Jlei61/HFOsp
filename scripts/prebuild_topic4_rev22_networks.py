#!/usr/bin/env python3
"""Pre-build the frozen network caches for the rev22 qualification/confirmation topologies.

Structure only: it reconstructs the substrate once per topology seed so the frozen network
pickle exists before the controller launches many workers on the same new seed. Without
this, the first worker of every new topology pays the build cost inside its own wall clock
and several workers can race to build the same missing cache, each holding the build's
peak memory at the same time.

No simulation, no patient data, no candidate parameter other than the frozen reference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src" / "snn_engine"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_REV20 = ROOT / "config/topic4_rev20_dc_dual_core_mechanism_atlas.json"
DEFAULT_TRANSITION = ROOT / "config/topic4_rev22_dci_transition_execution.json"
DEFAULT_SEEDS = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_dual_core_interictal_identifiability/"
    "response_design/seed_manifest.json")


def build_one(seed: int, *, artifact_root: Path, rev20: dict, transition_path: Path,
              node_field: dict) -> dict:
    from src.topic4_zm_ictal_transition import build_substrate, load_round_config

    started = time.time()
    transition = load_round_config(transition_path)
    reference = rev20["reference"]
    substrate = build_substrate(
        transition, str(reference["base_substrate_candidate_id"]), int(seed),
        cache_dir=str(artifact_root / rev20["network_cache"]),
        ee_dose=float(reference["g_EE"]), etoi_dose=float(reference["g_EtoI"]),
        node_candidate_override=dict(node_field),
        node_depth_shrinkage=float(reference["signed_depth_shrinkage"]),
        node_gain=float(reference["node_gain"]),
        # the ellipse mechanism is left at its exact no-op default: the network cache key
        # depends on the engine parameters, the registered axis and the engine AR, never on
        # the reweighting, so a warm-up must not pin a reweighting reference at all.
        artifact_root=artifact_root, topology_seed=int(seed), dynamics_seed=int(seed),
    )
    cache = substrate.network_cache or {}
    return {"seed": int(seed), "seconds": round(time.time() - started, 1),
            "cache_hit": bool(cache.get("hit")), "cache_path": cache.get("frozen_cache_path"),
            "cache_sha256": cache.get("cache_sha256"),
            "n_e": int(substrate.n_e), "n_i": int(substrate.n_i)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--stage", choices=("qualification", "confirmation", "both"), default="both")
    parser.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--rev20-config", type=Path, default=DEFAULT_REV20)
    parser.add_argument("--transition-config", type=Path, default=DEFAULT_TRANSITION)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rev20 = json.loads(args.rev20_config.read_text())
    manifest_path = args.artifact_root / rev20["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    node_field = [c for c in manifest["candidates"] if c.get("is_reference")][0]["node_field"]
    if args.seeds:
        seeds = list(dict.fromkeys(int(s) for s in args.seeds))
    else:
        seed_manifest = json.loads(args.seed_manifest.read_text())
        stages = ("qualification", "confirmation") if args.stage == "both" else (args.stage,)
        seeds = list(dict.fromkeys(
            int(unit["topology_seed"]) for stage in stages
            for unit in seed_manifest[stage]["units"]))
    print(f"pre-building {len(seeds)} topologies with {args.workers} workers: {seeds}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    started = time.time()
    records = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {pool.submit(build_one, seed, artifact_root=args.artifact_root, rev20=rev20,
                               transition_path=args.transition_config, node_field=node_field): seed
                   for seed in seeds}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(f"[{time.time()-started:6.0f}s] seed {record['seed']}: "
                  f"{'cache hit' if record['cache_hit'] else 'built'} in {record['seconds']}s",
                  flush=True)
    payload = {"schema_id": "topic4_rev22_dci_network_prebuild_v1",
               "seeds": sorted(r["seed"] for r in records),
               "records": sorted(records, key=lambda r: r["seed"]),
               "elapsed_seconds": round(time.time() - started, 1),
               "claim_boundary": "network cache warm-up only; no simulation and no scoring"}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"seeds": len(records),
                      "built": sum(0 if r["cache_hit"] else 1 for r in records),
                      "elapsed_s": payload["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
