#!/usr/bin/env python3
"""Build one independent LC6A E->I graph condition from the frozen C0 graph.

The five graph conditions are independent graph-only interventions.  This entry point lets Q1/Q2/Q3
use separate resource-gated workers while C1 is still mixing.  It uses the same generator, seeds,
legality contract, and output format as the serial family builder.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src/snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_topic4_fcxr_lc6a_graph_family as FAMILY  # noqa: E402
import run_m4_phaseplane as PP  # noqa: E402
from src.topic4_fcxr_lc6_surround import (  # noqa: E402
    audit_basic_legality,
    compare_outdegree_to_c0,
    construction_q,
    extract_e_to_e,
    extract_e_to_i,
    extract_i_to_e,
    make_rewired_graph,
    rewire_e_to_i_targetwise,
    source_outdegree_audit,
    validate_q_target,
)


ALLOWED = ("C1", "Q1", "Q2", "Q3")


def build_condition(manifest_path, condition):
    if condition not in ALLOWED:
        raise ValueError(f"condition must be one of {ALLOWED}")
    manifest_path, manifest = FAMILY._validate_manifest(manifest_path)
    generation = manifest["graph_generation"]
    graph_contract = manifest["graph_contract"]
    graph_rows = {row["id"]: row for row in manifest["graph_family"]}
    resources = FAMILY._meminfo()
    if resources["mem_available_gib"] < 48.0:
        raise RuntimeError("one graph worker requires at least 48 GiB MemAvailable")
    started = time.time()
    S = PP.build_substrate(manifest["model"]["connection_seed"])
    c0 = extract_e_to_i(S["net"], S["NE"], S["NI"])
    ee = extract_e_to_e(S["net"], S["NE"])
    i2e = extract_i_to_e(S["net"], S["NE"], S["NI"])
    ee_width = FAMILY._widths(ee, S["posE"], S["posE"], S["axis_unit"])
    i2e_width = FAMILY._widths(i2e, S["posI"], S["posE"], S["axis_unit"])
    c0_width = FAMILY._widths(c0, S["posE"], S["posI"], S["axis_unit"])
    c0_q = construction_q(c0_width, i2e_width, ee_width)
    c0_outdegree = source_outdegree_audit(
        c0.sources, S["posE"], S["axis_unit"], sheet_size_mm=S["L"],
        edge_margin_mm=generation["edge_margin_mm"],
    )
    row = graph_rows[condition]
    q_target = c0_q if row["q_marginal"] == "legacy" else float(row["q_marginal"])
    tolerance = float(row["tolerance"])
    if condition == "C1":
        l_parallel = float(S["p"].l_IE)
        desired_sigma = float(c0_width["sigma_parallel_mm"])
    else:
        l_parallel, desired_sigma = FAMILY._target_parallel_width(
            q_target, c0_width, i2e_width, ee_width, S["p"].l_IE,
        )
    seed = int(generation["condition_graph_seeds"][condition])
    sources, chain = rewire_e_to_i_targetwise(
        c0.sources, S["posE"], S["posI"], S["axis_unit"],
        l_parallel=l_parallel,
        l_perpendicular=float(generation["perpendicular_sampler_width_mm"]),
        graph_seed=seed,
        n_sweeps=int(generation["initial_sweeps"]),
        proposal_block_size=int(generation["proposal_block_size"]),
        proposal_perpendicular_bin_mm=float(generation["proposal_perpendicular_bin_mm"]),
    )
    candidate = make_rewired_graph(
        c0, sources, S["posE"], S["posI"], graph_seed=seed,
        tau0_ms=S["p"].tau0, v_axon_mm_per_ms=S["p"].v_axon,
        delay_dt_ms=S["p"].delay_dt, engine_dt_ms=S["p"].dt,
    )
    width = FAMILY._widths(candidate, S["posE"], S["posI"], S["axis_unit"])
    q_observed = construction_q(width, i2e_width, ee_width)
    legality = audit_basic_legality(c0, candidate, ne=S["NE"])
    outdegree = source_outdegree_audit(
        candidate.sources, S["posE"], S["axis_unit"], sheet_size_mm=S["L"],
        edge_margin_mm=generation["edge_margin_mm"],
    )
    outdegree_comparison = compare_outdegree_to_c0(
        outdegree, c0_outdegree,
        relative_tolerance=graph_contract["source_out_degree_relative_tolerance"],
    )
    errors = []
    try:
        validate_q_target(q_observed, q_target, tolerance)
    except RuntimeError as exc:
        errors.append(str(exc))
    if not outdegree_comparison["within_contract"]:
        errors.append("source out-degree distribution exceeds the frozen C0 tolerance")
    audit = {
        **legality,
        "status": "COMPLETE",
        "condition": condition,
        "role": row["role"],
        "manifest": str(manifest_path),
        "manifest_sha256": FAMILY._sha(manifest_path),
        "trajectory_outcome_read": False,
        "q_target": q_target,
        "q_tolerance": tolerance,
        "construction_q": q_observed,
        "proposal_l_parallel_mm": l_parallel,
        "proposal_l_perpendicular_mm": float(generation["perpendicular_sampler_width_mm"]),
        "desired_e_to_i_sigma_parallel_mm": desired_sigma,
        "marginal_e_to_i": width,
        "source_outdegree": outdegree,
        "source_outdegree_vs_c0": outdegree_comparison,
        "chain": chain,
        "common_maximum_sweeps": int(generation["maximum_sweeps"]),
        "graph_legality": "PASS" if not errors else "FAIL",
        "graph_legality_errors": errors,
        "frozen_reference_widths": {
            "e_to_e": ee_width, "i_to_e": i2e_width,
            "c0_e_to_i": c0_width, "c0_construction_q": c0_q,
        },
        "resource_start": resources,
        "resource_end": FAMILY._meminfo(),
        "wall_s": time.time() - started,
    }
    FAMILY._save_graph(FAMILY.OUT / f"graphs/{condition}.npz", candidate, audit)
    FAMILY._write_json(FAMILY.OUT / f"graph_condition_{condition}.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=ALLOWED, required=True)
    parser.add_argument(
        "--execution-manifest", type=Path,
        default=ROOT / "config/topic4_fcxr_lc6a_patient_axis_surround.json",
    )
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("single-condition graph build requires --confirm-run")
    FAMILY.OUT.mkdir(parents=True, exist_ok=True)
    lock_path = FAMILY.OUT / f".graph_condition_{args.condition}.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"graph condition {args.condition} is already running") from exc
        running = FAMILY.OUT / f"RUNNING_LC6A_GRAPH_{args.condition}.json"
        failed = FAMILY.OUT / f"FAILED_LC6A_GRAPH_{args.condition}.json"
        done = FAMILY.OUT / f"DONE_LC6A_GRAPH_{args.condition}.json"
        FAMILY._write_json(running, {"status": "RUNNING", "pid": os.getpid(), "condition": args.condition})
        try:
            result = build_condition(args.execution_manifest, args.condition)
            FAMILY._write_json(done, {
                "status": "DONE", "condition": args.condition,
                "graph_legality": result["graph_legality"],
                "audit": str(FAMILY.OUT / f"graph_condition_{args.condition}.json"),
            })
            failed.unlink(missing_ok=True)
            print(json.dumps(FAMILY._jsonable(result), indent=2, sort_keys=True))
        except BaseException as exc:
            FAMILY._write_json(failed, {
                "status": "FAILED", "condition": args.condition,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        finally:
            running.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
