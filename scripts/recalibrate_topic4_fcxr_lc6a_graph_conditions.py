#!/usr/bin/env python3
"""Graph-only secant recalibration for LC6A Q1/Q2/Q3 construction coordinates."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src/snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_topic4_fcxr_lc6a_graph_condition as CONDITION  # noqa: E402
import build_topic4_fcxr_lc6a_graph_family as FAMILY  # noqa: E402


Q_IDS = ("Q1", "Q2", "Q3")
DEFAULT_ADDENDUM = ROOT / "config/topic4_fcxr_lc6a_graph_recalibration_addendum.json"


def _only_q_target_failed(audit: dict) -> bool:
    errors = list(audit.get("graph_legality_errors", []))
    return (
        audit.get("graph_legality") == "FAIL"
        and len(errors) == 1
        and str(errors[0]).startswith("construction q target unreachable:")
    )


def secant_width(*, l_low, sigma_low, l_high, sigma_high, sigma_target, lower, upper):
    denominator = float(sigma_high) - float(sigma_low)
    if not np.isfinite(denominator) or abs(denominator) < 1e-9:
        raise RuntimeError("graph-only secant anchors do not separate empirical width")
    value = float(l_high) + (
        (float(sigma_target) - float(sigma_high))
        * (float(l_high) - float(l_low)) / denominator
    )
    if not np.isfinite(value):
        raise RuntimeError("graph-only secant produced a non-finite width")
    return float(np.clip(value, float(lower), float(upper)))


def _archive_attempt(condition: str, round_index: int) -> Path:
    destination = FAMILY.OUT / f"superseded/graph_recalibration/{condition}/attempt_{round_index}"
    destination.mkdir(parents=True, exist_ok=False)
    candidates = (
        FAMILY.OUT / f"graphs/{condition}.npz",
        FAMILY.OUT / f"graph_condition_{condition}.json",
        FAMILY.OUT / f"DONE_LC6A_GRAPH_{condition}.json",
        FAMILY.OUT / f"FAILED_LC6A_GRAPH_{condition}.json",
    )
    for source in candidates:
        if source.exists():
            shutil.move(str(source), destination / source.name)
    return destination


def _read_condition(condition: str) -> dict:
    path = FAMILY.OUT / f"graph_condition_{condition}.json"
    if not path.is_file():
        raise RuntimeError(f"missing condition audit: {condition}")
    return json.loads(path.read_text())


def recalibrate(manifest_path: Path, addendum_path: Path) -> dict:
    manifest_path, _manifest = FAMILY._validate_manifest(manifest_path)
    addendum_path = Path(addendum_path).resolve()
    addendum = json.loads(addendum_path.read_text())
    if addendum.get("scope") != "graph_only_before_any_lc6a_snn_dynamics":
        raise RuntimeError("wrong graph recalibration addendum")
    if any((FAMILY.OUT / "trajectories" / key / "summary.json").is_file() for key in ("C0", *Q_IDS)):
        raise RuntimeError("graph recalibration is forbidden after LC6A trajectory outcomes exist")
    rule = addendum["rule"]
    c1 = _read_condition("C1")
    if c1.get("graph_legality") != "PASS":
        raise RuntimeError("same-sampler C1 anchor must be graph-legal")
    c1_anchor = {
        "l": float(c1["proposal_l_parallel_mm"]),
        "sigma": float(c1["marginal_e_to_i"]["sigma_parallel_mm"]),
        "graph_sha256": c1["graph_sha256"],
    }
    results = {}
    for condition in Q_IDS:
        audit = _read_condition(condition)
        if audit.get("graph_legality") == "PASS":
            results[condition] = {"status": "INITIAL_PASS", "audit": audit}
            continue
        if not _only_q_target_failed(audit):
            raise RuntimeError(f"{condition} failed a non-calibratable graph contract")
        previous = {
            "l": float(audit["proposal_l_parallel_mm"]),
            "sigma": float(audit["marginal_e_to_i"]["sigma_parallel_mm"]),
            "graph_sha256": audit["graph_sha256"],
        }
        low = dict(c1_anchor)
        history = [{"round": 0, **previous, "q": float(audit["construction_q"])}]
        target_sigma = float(audit["desired_e_to_i_sigma_parallel_mm"])
        for correction in range(1, int(rule["max_correction_rounds"]) + 1):
            new_l = secant_width(
                l_low=low["l"], sigma_low=low["sigma"],
                l_high=previous["l"], sigma_high=previous["sigma"],
                sigma_target=target_sigma,
                lower=rule["minimum_l_parallel_mm"],
                upper=rule["maximum_l_parallel_mm"],
            )
            archive = _archive_attempt(condition, correction - 1)
            provenance = {
                "addendum": str(addendum_path),
                "addendum_sha256": FAMILY._sha(addendum_path),
                "correction_round": correction,
                "anchor_low": low,
                "anchor_high": previous,
                "target_sigma_parallel_mm": target_sigma,
                "computed_l_parallel_mm": new_l,
                "archived_previous_attempt": str(archive),
                "trajectory_outcome_read": False,
            }
            audit = CONDITION.build_condition(
                manifest_path,
                condition,
                proposal_l_parallel_override=new_l,
                calibration_provenance=provenance,
            )
            FAMILY._write_json(FAMILY.OUT / f"DONE_LC6A_GRAPH_{condition}.json", {
                "status": "DONE", "condition": condition,
                "graph_legality": audit["graph_legality"],
                "audit": str(FAMILY.OUT / f"graph_condition_{condition}.json"),
                "recalibrated": True,
            })
            current = {
                "l": float(audit["proposal_l_parallel_mm"]),
                "sigma": float(audit["marginal_e_to_i"]["sigma_parallel_mm"]),
                "graph_sha256": audit["graph_sha256"],
            }
            history.append({"round": correction, **current, "q": float(audit["construction_q"])})
            if audit["graph_legality"] == "PASS":
                break
            if not _only_q_target_failed(audit):
                raise RuntimeError(f"{condition} correction failed a non-q graph contract")
            low, previous = previous, current
        if audit["graph_legality"] != "PASS":
            raise RuntimeError(f"GRAPH_TARGET_UNREACHABLE_ENGINEERING_STOP: {condition}")
        results[condition] = {"status": "RECALIBRATED_PASS", "history": history, "audit": audit}
    payload = {
        "status": "COMPLETE",
        "stage": "LC6A_GRAPH_ONLY_RECALIBRATION",
        "manifest": str(manifest_path),
        "manifest_sha256": FAMILY._sha(manifest_path),
        "addendum": str(addendum_path),
        "addendum_sha256": FAMILY._sha(addendum_path),
        "trajectory_outcome_read": False,
        "c1_anchor": c1_anchor,
        "results": results,
        "resource_end": FAMILY._meminfo(),
    }
    FAMILY._write_json(FAMILY.OUT / "graph_recalibration_audit.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execution-manifest", type=Path,
        default=ROOT / "config/topic4_fcxr_lc6a_patient_axis_surround.json",
    )
    parser.add_argument("--addendum", type=Path, default=DEFAULT_ADDENDUM)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("graph-only recalibration requires --confirm-run")
    with (FAMILY.OUT / ".graph_recalibration.lock").open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("LC6A graph recalibration is already running") from exc
        running = FAMILY.OUT / "RUNNING_LC6A_GRAPH_RECALIBRATION.json"
        failed = FAMILY.OUT / "FAILED_LC6A_GRAPH_RECALIBRATION.json"
        done = FAMILY.OUT / "DONE_LC6A_GRAPH_RECALIBRATION.json"
        FAMILY._write_json(running, {"status": "RUNNING", "pid": os.getpid()})
        try:
            result = recalibrate(args.execution_manifest, args.addendum)
            FAMILY._write_json(done, {
                "status": "DONE", "audit": str(FAMILY.OUT / "graph_recalibration_audit.json"),
            })
            failed.unlink(missing_ok=True)
            print(json.dumps(FAMILY._jsonable(result), indent=2, sort_keys=True))
        except BaseException as exc:
            FAMILY._write_json(failed, {
                "status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        finally:
            running.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
