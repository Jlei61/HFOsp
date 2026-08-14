#!/usr/bin/env python3
"""Conditionally build and adjudicate one LC6A graph-realization confirmation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src/snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import aggregate_topic4_fcxr_lc6a_phenotypes as AGG  # noqa: E402
import audit_topic4_fcxr_lc6a_two_hop as TWOHOP  # noqa: E402
import build_topic4_fcxr_lc6a_graph_condition as CONDITION  # noqa: E402
import build_topic4_fcxr_lc6a_graph_family as FAMILY  # noqa: E402
import recalibrate_topic4_fcxr_lc6a_graph_conditions as RECAL  # noqa: E402
import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_fcxr_lc6a_natural_trajectory as NAT  # noqa: E402
from src.topic4_fcxr_lc6_surround import extract_i_to_e, graph_sha256  # noqa: E402
from src.topic4_fcxr_lc6_twohop import (  # noqa: E402
    coarse_two_hop_operator, sample_two_hop_latencies, spatial_bins,
    summarize_two_hop_operator,
)


OUT = NAT.OUT
PRELOCK = ROOT / "config/topic4_fcxr_lc6a_confirmation_prelock.json"
PHENOTYPE = OUT / "phenotype_map.json"
GAINS = OUT / "gain_forks.json"
LOCAL_LOCK = OUT / "local_classifier_manifest_addendum.json"
GRAPH_AUDIT = OUT / "graph_audit.json"
LOCK = OUT / "confirmation_lock.json"
FINAL = OUT / "confirmation_summary.json"
DONE = OUT / "DONE_LC6A_CONFIRMATION.json"
MECHANISM_FILES = (
    Path(__file__).resolve(), PRELOCK,
    ROOT / "scripts/build_topic4_fcxr_lc6a_graph_condition.py",
    ROOT / "scripts/run_topic4_fcxr_lc6a_natural_trajectory.py",
    ROOT / "src/topic4_fcxr_lc6_surround.py",
    ROOT / "src/topic4_fcxr_lc6_twohop.py",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes():
    return {str(path.relative_to(ROOT)): _sha(path) for path in MECHANISM_FILES}


def _write_json(path, payload):
    NAT._write_json(path, payload)


def _prelock():
    payload = json.loads(PRELOCK.read_text())
    if payload.get("experiment_id") != "fcxr_lc6a_graph_realization_confirmation":
        raise RuntimeError("wrong LC6A confirmation prelock")
    if payload["graph"]["trajectory_outcome_may_adjust_q_or_width"] is not False:
        raise RuntimeError("confirmation graph geometry cannot use trajectory outcomes")
    return payload


def select_candidate(phenotype, gains):
    gain_by = {row["condition"]: row for row in gains.get("rows", [])}
    eligible = []
    for row in phenotype.get("rows", []):
        if not row.get("boundedness", {}).get("bounded_candidate", False):
            continue
        gain = gain_by.get(row["condition"])
        if gain is None:
            continue
        usable = []
        for checkpoint in gain.get("checkpoints", []):
            probe = checkpoint["probe"]
            local = probe["local_saturation"]
            if (
                checkpoint.get("response_detected_nonzero")
                and not probe.get("registered_global_saturation", False)
                and float(local["max_near_refractory_fraction"]) < .05
                and not probe.get("diverged", False)
            ):
                usable.append(checkpoint["checkpoint"])
        if usable:
            eligible.append((
                float(row["boundedness"].get("boundedness_margin", -np.inf)),
                row["condition"], usable,
            ))
    if not eligible:
        return None
    margin, condition, checkpoints = sorted(eligible, key=lambda item: (-item[0], item[1]))[0]
    return {
        "condition": condition, "boundedness_margin": margin,
        "responsive_checkpoints": checkpoints,
    }


def _only_q_failed(audit):
    errors = list(audit.get("graph_legality_errors", []))
    return (
        audit.get("graph_legality") == "FAIL" and len(errors) == 1
        and str(errors[0]).startswith("construction q target unreachable:")
    )


def _condition_audit(condition):
    sidecar = OUT / f"graph_condition_{condition}.json"
    if sidecar.is_file():
        return json.loads(sidecar.read_text())
    _graph, metadata = NAT._load_graph(OUT / f"graphs/{condition}.npz")
    return metadata


def _archive_attempt(output_condition, index):
    destination = OUT / f"confirmation/superseded/{output_condition}/attempt_{index}"
    destination.mkdir(parents=True, exist_ok=False)
    for path in (
        OUT / f"graphs/{output_condition}.npz",
        OUT / f"graph_condition_{output_condition}.json",
    ):
        if path.exists():
            shutil.move(str(path), destination / path.name)
    return destination


def _two_hop(graph, *, audit_seed):
    graph_audit = json.loads(GRAPH_AUDIT.read_text())
    S = PP.build_substrate(1)
    i2e = extract_i_to_e(S["net"], S["NE"], S["NI"])
    bins = spatial_bins(S["posE"], sheet_size_mm=S["L"], n_bins_axis=24)
    ee_width = graph_audit["frozen_reference_widths"]["e_to_e"]
    operator = coarse_two_hop_operator(graph, i2e, bins, n_e=S["NE"], n_i=S["NI"])
    return {
        "operator": summarize_two_hop_operator(
            operator, bins, S["axis_unit"],
            ee_sigma_parallel_mm=ee_width["sigma_parallel_mm"],
            ee_sigma_perpendicular_mm=ee_width["sigma_perpendicular_mm"],
            edge_margin_mm=1.0,
        ),
        "latency": sample_two_hop_latencies(
            graph, i2e, n_e=S["NE"], n_i=S["NI"], engine_dt_ms=S["p"].dt,
            n_paths=20000, audit_seed=int(audit_seed),
        ),
    }


def build(manifest_path):
    prelock = _prelock()
    if LOCK.is_file():
        return json.loads(LOCK.read_text())
    for path in (PHENOTYPE, GAINS, LOCAL_LOCK, GRAPH_AUDIT):
        if not path.is_file():
            raise RuntimeError(f"confirmation prerequisite missing: {path}")
    phenotype = json.loads(PHENOTYPE.read_text())
    gains = json.loads(GAINS.read_text())
    candidate = select_candidate(phenotype, gains)
    common = {
        "status": "LOCKED", "stage": "LC6A_GRAPH_REALIZATION_CONFIRMATION",
        "prelock": str(PRELOCK), "prelock_sha256": _sha(PRELOCK),
        "phenotype_map": str(PHENOTYPE), "phenotype_map_sha256": _sha(PHENOTYPE),
        "gain_forks": str(GAINS), "gain_forks_sha256": _sha(GAINS),
        "local_classifier_lock": str(LOCAL_LOCK), "local_classifier_lock_sha256": _sha(LOCAL_LOCK),
        "source_sha256": _source_hashes(),
    }
    if candidate is None:
        payload = {
            **common, "authorized": False,
            "decision": prelock["decision"]["not_triggered"],
            "reason": "no canonical phenotype is both bounded and non-saturating responsive",
        }
        _write_json(LOCK, payload)
        return payload
    parent = candidate["condition"]
    builder_parent = "C1" if parent == "C0" else parent
    output_condition = f"CONF_{parent}_{prelock['graph']['output_suffix']}"
    original_graph, original_meta = NAT._load_graph(OUT / f"graphs/{parent}.npz")
    builder_meta = _condition_audit("C1") if parent == "C0" else original_meta
    l_parallel = float(builder_meta["proposal_l_parallel_mm"])
    graph_seed = int(prelock["graph"]["seed_by_parent_condition"][parent])
    history = []
    audit = CONDITION.build_condition(
        manifest_path, builder_parent, proposal_l_parallel_override=l_parallel,
        graph_seed_override=graph_seed, output_condition=output_condition,
        calibration_provenance={
            "role": "graph_realization_B_same_locked_q",
            "canonical_parent_condition": parent,
            "canonical_graph_sha256": graph_sha256(original_graph),
            "trajectory_outcome_used_only_to_trigger_confirmation": True,
            "trajectory_outcome_used_to_choose_q_or_width": False,
        },
    )
    history.append({
        "round": 0, "proposal_l_parallel_mm": audit["proposal_l_parallel_mm"],
        "sigma_parallel_mm": audit["marginal_e_to_i"]["sigma_parallel_mm"],
        "construction_q": audit["construction_q"], "graph_sha256": audit["graph_sha256"],
    })
    if audit["graph_legality"] != "PASS" and not _only_q_failed(audit):
        raise RuntimeError("confirmation graph failed a non-q legality contract")
    if audit["graph_legality"] != "PASS":
        c1 = _condition_audit("C1")
        low = {
            "l": float(c1["proposal_l_parallel_mm"]),
            "sigma": float(c1["marginal_e_to_i"]["sigma_parallel_mm"]),
        }
        previous = {
            "l": float(audit["proposal_l_parallel_mm"]),
            "sigma": float(audit["marginal_e_to_i"]["sigma_parallel_mm"]),
        }
        target_sigma = float(audit["desired_e_to_i_sigma_parallel_mm"])
        for correction in range(1, int(prelock["graph"]["maximum_graph_only_q_corrections"]) + 1):
            if np.isclose(low["l"], previous["l"]):
                # A legacy-q replicate can share C1's proposal width while differing in
                # empirical width solely because of its new graph seed.  The first correction
                # therefore uses a positive scale step; subsequent rounds have two secant anchors.
                new_l = float(np.clip(
                    previous["l"] * target_sigma / previous["sigma"],
                    prelock["graph"]["minimum_l_parallel_mm"],
                    prelock["graph"]["maximum_l_parallel_mm"],
                ))
            else:
                new_l = RECAL.secant_width(
                    l_low=low["l"], sigma_low=low["sigma"],
                    l_high=previous["l"], sigma_high=previous["sigma"],
                    sigma_target=target_sigma,
                    lower=prelock["graph"]["minimum_l_parallel_mm"],
                    upper=prelock["graph"]["maximum_l_parallel_mm"],
                )
            archive = _archive_attempt(output_condition, correction - 1)
            audit = CONDITION.build_condition(
                manifest_path, builder_parent, proposal_l_parallel_override=new_l,
                graph_seed_override=graph_seed, output_condition=output_condition,
                calibration_provenance={
                    "role": "confirmation_graph_only_q_correction",
                    "round": correction, "archived_previous_attempt": str(archive),
                    "canonical_parent_condition": parent,
                    "trajectory_outcome_used_to_choose_q_or_width": False,
                },
            )
            history.append({
                "round": correction, "proposal_l_parallel_mm": audit["proposal_l_parallel_mm"],
                "sigma_parallel_mm": audit["marginal_e_to_i"]["sigma_parallel_mm"],
                "construction_q": audit["construction_q"], "graph_sha256": audit["graph_sha256"],
            })
            if audit["graph_legality"] == "PASS":
                break
            if not _only_q_failed(audit):
                raise RuntimeError("confirmation correction failed a non-q graph contract")
            low, previous = previous, {
                "l": float(audit["proposal_l_parallel_mm"]),
                "sigma": float(audit["marginal_e_to_i"]["sigma_parallel_mm"]),
            }
    if audit["graph_legality"] != "PASS":
        raise RuntimeError("CONFIRMATION_GRAPH_TARGET_UNREACHABLE")
    graph_path = OUT / f"graphs/{output_condition}.npz"
    graph, metadata = NAT._load_graph(graph_path)
    two_hop = _two_hop(graph, audit_seed=772000 + list(NAT.GRAPH_IDS).index(parent))
    _write_json(OUT / f"confirmation/{output_condition}_two_hop.json", two_hop)
    payload = {
        **common, "authorized": True, "decision": "RUN_GRAPH_REALIZATION_B",
        "parent_condition": parent, "builder_parent_condition": builder_parent,
        "output_condition": output_condition, "candidate_evidence": candidate,
        "graph_seed": graph_seed, "graph_artifact": str(graph_path),
        "graph_artifact_sha256": _sha(graph_path), "graph_sha256": graph_sha256(graph),
        "construction_q": metadata["construction_q"], "graph_build_history": history,
        "two_hop": two_hop,
        "noise_seed": int(prelock["runtime"]["noise_seed"]),
        "same_q_weight_delay_rule": True,
    }
    _write_json(LOCK, payload)
    return payload


def finalize():
    lock = json.loads(LOCK.read_text())
    if lock.get("status") != "LOCKED":
        raise RuntimeError("confirmation lock missing")
    for key, path in (
        ("prelock_sha256", PRELOCK), ("phenotype_map_sha256", PHENOTYPE),
        ("gain_forks_sha256", GAINS), ("local_classifier_lock_sha256", LOCAL_LOCK),
    ):
        if lock[key] != _sha(path):
            raise RuntimeError(f"confirmation evidence drift: {path}")
    if lock["source_sha256"] != _source_hashes():
        raise RuntimeError("confirmation source drift")
    if not lock["authorized"]:
        payload = {
            "status": "COMPLETE_NOT_TRIGGERED", "decision": lock["decision"],
            "termination_tested": False, "lifecycle_tested": False,
        }
        _write_json(FINAL, payload)
        _write_json(DONE, {"status": "DONE", "decision": payload["decision"]})
        return payload
    summary_path = OUT / f"trajectories/{lock['output_condition']}/summary.json"
    if not summary_path.is_file():
        raise RuntimeError("authorized confirmation trajectory is incomplete")
    local_lock = json.loads(LOCAL_LOCK.read_text())
    confirmation = AGG._load_condition(lock["output_condition"], local_lock)
    phenotype = json.loads(PHENOTYPE.read_text())
    canonical = next(row for row in phenotype["rows"] if row["condition"] == lock["parent_condition"])
    c0 = next(row for row in phenotype["rows"] if row["condition"] == "C0")
    confirmation["baseline_tradeoff"] = AGG.baseline_tradeoff(
        confirmation["baseline_metrics"], c0["baseline_metrics"],
    )
    replicated = bool(confirmation["boundedness"].get("bounded_candidate", False))
    tradeoff = bool(canonical["baseline_tradeoff"]["tradeoff"] or confirmation["baseline_tradeoff"]["tradeoff"])
    if replicated and not tradeoff:
        decision = _prelock()["decision"]["replicated_bounded_carrier"]
    elif replicated:
        decision = "CARRIER_REPLICATED_WITH_BASELINE_TRADEOFF"
    else:
        decision = _prelock()["decision"]["not_replicated"]
    payload = {
        "status": "COMPLETE", "decision": decision,
        "parent_condition": lock["parent_condition"],
        "confirmation_condition": lock["output_condition"],
        "canonical": canonical, "confirmation": confirmation,
        "bounded_carrier_replicated": replicated,
        "termination_tested": False, "lifecycle_tested": False,
    }
    _write_json(FINAL, payload)
    _write_json(DONE, {
        "status": "DONE", "decision": decision, "summary": str(FINAL),
        "summary_sha256": _sha(FINAL),
    })
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("build", "finalize"))
    parser.add_argument(
        "--execution-manifest", type=Path,
        default=ROOT / "config/topic4_fcxr_lc6a_patient_axis_surround.json",
    )
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("LC6A confirmation requires --confirm-run")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f".confirmation_{args.stage}.lock").open("w") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("requested LC6A confirmation stage is already running") from exc
        running = OUT / f"RUNNING_LC6A_CONFIRMATION_{args.stage.upper()}.json"
        failed = OUT / f"FAILED_LC6A_CONFIRMATION_{args.stage.upper()}.json"
        _write_json(running, {"status": "RUNNING", "pid": os.getpid(), "stage": args.stage})
        try:
            result = build(args.execution_manifest) if args.stage == "build" else finalize()
            failed.unlink(missing_ok=True)
            print(json.dumps(NAT._jsonable(result), indent=2, sort_keys=True))
        except BaseException as exc:
            _write_json(failed, {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            running.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
