#!/usr/bin/env python3
"""Build and freeze the FCXR-LC6A C0/C1/Q1/Q2/Q3 E->I graph family.

This is a graph-only stage.  It never runs the SNN and never reads a trajectory
outcome.  Target widths may therefore be calibrated from frozen connection
geometry without leaking natural-onset results into graph selection.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src/snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_m4_phaseplane as PP  # noqa: E402
from src.topic4_fcxr_lc6_surround import (  # noqa: E402
    EToIGraph,
    audit_basic_legality,
    compare_outdegree_to_c0,
    construction_q,
    empirical_edge_widths,
    extract_e_to_e,
    extract_e_to_i,
    extract_i_to_e,
    graph_sha256,
    make_rewired_graph,
    rewire_e_to_i_targetwise,
    source_outdegree_audit,
    validate_q_target,
)


OUT = ROOT / "results/topic4_sef_hfo/fcxr_lc6a_patient_axis_surround"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _save_graph(path, graph: EToIGraph, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        sources=graph.sources,
        weights=graph.weights,
        delay_steps=graph.delay_steps,
        graph_sha256=np.asarray([graph_sha256(graph)]),
        metadata_json=np.asarray([json.dumps(_jsonable(metadata), sort_keys=True)]),
    )
    os.replace(tmp, path)


def _meminfo():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        values[key] = float(raw.split()[0]) / 1024.0 / 1024.0
    return {
        "mem_available_gib": values["MemAvailable"],
        "swap_used_mib": (values["SwapTotal"] - values["SwapFree"]) * 1024.0,
    }


def _validate_manifest(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    if payload.get("experiment_id") != "fcxr_lc6a_patient_axis_surround":
        raise RuntimeError("wrong LC6A execution manifest")
    for relative, expected in payload["blessed_engine_sha256"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"blessed engine hash mismatch: {relative}")
    graph_ids = [row["id"] for row in payload["graph_family"]]
    if graph_ids != ["C0", "C1", "Q1", "Q2", "Q3"]:
        raise RuntimeError("graph family is not the locked five-condition order")
    return path, payload


def _widths(graph, source_positions, target_positions, axis):
    return empirical_edge_widths(
        graph.sources, source_positions, target_positions, axis, chunk_targets=128,
    )


def _target_parallel_width(q_target, c0_e2i, i2e, ee, legacy_l_parallel):
    desired_variance = (
        (float(q_target) * float(ee["sigma_parallel_mm"])) ** 2
        - float(i2e["sigma_parallel_mm"]) ** 2
    )
    if desired_variance <= 0.0:
        raise RuntimeError(f"q={q_target} cannot yield a positive E-to-I width")
    desired_sigma = float(np.sqrt(desired_variance))
    observed_sigma = float(c0_e2i["sigma_parallel_mm"])
    return float(legacy_l_parallel) * desired_sigma / observed_sigma, desired_sigma


def build_family(manifest_path):
    manifest_path, manifest = _validate_manifest(manifest_path)
    generation = manifest["graph_generation"]
    graph_contract = manifest["graph_contract"]
    graph_rows = {row["id"]: row for row in manifest["graph_family"]}
    resources = _meminfo()
    if resources["mem_available_gib"] < 96.0:
        raise RuntimeError("graph family build requires at least 96 GiB MemAvailable")
    started = time.time()
    S = PP.build_substrate(manifest["model"]["connection_seed"])
    c0 = extract_e_to_i(S["net"], S["NE"], S["NI"])
    ee = extract_e_to_e(S["net"], S["NE"])
    i2e = extract_i_to_e(S["net"], S["NE"], S["NI"])
    ee_width = _widths(ee, S["posE"], S["posE"], S["axis_unit"])
    i2e_width = _widths(i2e, S["posI"], S["posE"], S["axis_unit"])
    c0_width = _widths(c0, S["posE"], S["posI"], S["axis_unit"])
    c0_q = construction_q(c0_width, i2e_width, ee_width)
    c0_outdegree = source_outdegree_audit(
        c0.sources, S["posE"], S["axis_unit"], sheet_size_mm=S["L"],
        edge_margin_mm=generation["edge_margin_mm"],
    )
    family = {"C0": c0}
    audits = {
        "C0": {
            **audit_basic_legality(c0, c0, ne=S["NE"]),
            "marginal_e_to_i": c0_width,
            "construction_q": c0_q,
            "source_outdegree": c0_outdegree,
            "off_path_exact": True,
        }
    }
    _save_graph(OUT / "graphs/C0.npz", c0, audits["C0"])

    p = S["p"]
    l_perpendicular = float(generation["perpendicular_sampler_width_mm"])
    initial_sweeps = int(generation["initial_sweeps"])
    maximum_sweeps = int(generation["maximum_sweeps"])
    if initial_sweeps > maximum_sweeps:
        raise RuntimeError("initial graph sweeps exceed common maximum budget")
    for condition in ("C1", "Q1", "Q2", "Q3"):
        row = graph_rows[condition]
        q_target = c0_q if row["q_marginal"] == "legacy" else float(row["q_marginal"])
        tolerance = float(row["tolerance"])
        if condition == "C1":
            l_parallel = float(p.l_IE)
            desired_sigma = float(c0_width["sigma_parallel_mm"])
        else:
            l_parallel, desired_sigma = _target_parallel_width(
                q_target, c0_width, i2e_width, ee_width, p.l_IE,
            )
        seed = int(generation["condition_graph_seeds"][condition])
        sources, chain = rewire_e_to_i_targetwise(
            c0.sources, S["posE"], S["posI"], S["axis_unit"],
            l_parallel=l_parallel, l_perpendicular=l_perpendicular,
            graph_seed=seed, n_sweeps=initial_sweeps,
            proposal_block_size=int(generation["proposal_block_size"]),
            proposal_perpendicular_bin_mm=float(generation["proposal_perpendicular_bin_mm"]),
        )
        candidate = make_rewired_graph(
            c0, sources, S["posE"], S["posI"], graph_seed=seed,
            tau0_ms=p.tau0, v_axon_mm_per_ms=p.v_axon,
            delay_dt_ms=p.delay_dt, engine_dt_ms=p.dt,
        )
        width = _widths(candidate, S["posE"], S["posI"], S["axis_unit"])
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
            "role": row["role"],
            "q_target": q_target,
            "q_tolerance": tolerance,
            "construction_q": q_observed,
            "proposal_l_parallel_mm": l_parallel,
            "proposal_l_perpendicular_mm": l_perpendicular,
            "desired_e_to_i_sigma_parallel_mm": desired_sigma,
            "marginal_e_to_i": width,
            "source_outdegree": outdegree,
            "source_outdegree_vs_c0": outdegree_comparison,
            "chain": chain,
            "common_maximum_sweeps": maximum_sweeps,
            "graph_legality": "PASS" if not errors else "FAIL",
            "graph_legality_errors": errors,
        }
        family[condition] = candidate
        audits[condition] = audit
        _save_graph(OUT / f"graphs/{condition}.npz", candidate, audit)
        _write_json(OUT / "graph_build_progress.json", {
            "status": "RUNNING", "completed": list(family), "audits": audits,
            "wall_s": time.time() - started,
        })

    payload = {
        "status": "COMPLETE",
        "stage": "LC6A_GRAPH_FAMILY",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha(manifest_path),
        "trajectory_outcome_read": False,
        "graph_ids": list(family),
        "frozen_reference_widths": {
            "e_to_e": ee_width, "i_to_e": i2e_width,
            "c0_e_to_i": c0_width, "c0_construction_q": c0_q,
        },
        "audits": audits,
        "all_graphs_legal": all(
            row.get("graph_legality", "PASS") == "PASS" for row in audits.values()
        ),
        "resource_start": resources,
        "resource_end": _meminfo(),
        "wall_s": time.time() - started,
    }
    _write_json(OUT / "graph_audit.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execution-manifest", type=Path,
        default=ROOT / "config/topic4_fcxr_lc6a_patient_axis_surround.json",
    )
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("graph family build requires --confirm-run")
    OUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUT / ".graph_family.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("LC6A graph family stage is already running") from exc
        running = OUT / "RUNNING_LC6A_GRAPH_FAMILY.json"
        failed = OUT / "FAILED_LC6A_GRAPH_FAMILY.json"
        done = OUT / "DONE_LC6A_GRAPH_FAMILY.json"
        _write_json(running, {"status": "RUNNING", "pid": os.getpid()})
        try:
            result = build_family(args.execution_manifest)
            _write_json(done, {
                "status": "DONE", "all_graphs_legal": result["all_graphs_legal"],
                "graph_audit": str(OUT / "graph_audit.json"),
            })
            failed.unlink(missing_ok=True)
            print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
        except BaseException as exc:
            _write_json(failed, {
                "status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        finally:
            running.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
