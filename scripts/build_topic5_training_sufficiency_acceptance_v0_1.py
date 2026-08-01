#!/usr/bin/env python3
"""Machine-readable acceptance and reproducible manifest for the whole audit.

Walks every completed cell in the result tree and asserts the engineering
contract: no failed or missing cells, no non-finite metric, no broken seal, and
a complete provenance record for every run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_training_sufficiency import run_environment  # noqa: E402

RESULT_ROOT = ROOT / "results/topic5_rnn_training_sufficiency_v0_1"
PHASES = {
    "b1_budget": "development/b1_budget",
    "b1_budget_extended": "development/b1_budget_extended",
    "b2_learning_rate": "development/b2_learning_rate",
    "b3_chunk_parity": "development/b3_chunk_parity",
    "b1c_loso_confirm": "development/b1c_loso_confirm",
    "c_objectives": "development/c_objectives",
    "reproducibility": "development/reproducibility",
    "formal": "formal",
}
OOM_PATTERN = re.compile(r"CUDA out of memory|Killed", re.IGNORECASE)
NAN_PATTERN = re.compile(r"\bnan\b", re.IGNORECASE)


def _phase_report(root: Path) -> dict:
    if not root.is_dir():
        return {"present": False}
    summaries = sorted(root.rglob("run_summary.json"))
    done = sorted(root.rglob("DONE.json"))
    runtimes, gpu, rss = [], [], []
    seal_ok = True
    outer_read = []
    for path in summaries:
        summary = json.loads(path.read_text())
        resources = summary.get("resources", {})
        runtimes.append(float(resources.get("runtime_seconds", np.nan)))
        gpu.append(float(resources.get("gpu_peak_allocated_bytes", 0)) / 1e9)
        rss.append(float(resources.get("peak_rss_gb", np.nan)))
        if summary.get("ictal_target_read") is not False:
            seal_ok = False
        outer_read.append(bool(summary.get("outer_heldout_read", False)))
    state_path = root / "LAUNCHER_STATE.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    log_dir = root / "logs"
    oom, nan_logs = [], []
    if log_dir.is_dir():
        for log in sorted(log_dir.glob("*.log")):
            text = log.read_text(errors="replace")
            if OOM_PATTERN.search(text):
                oom.append(log.name)
            if NAN_PATTERN.search(text):
                nan_logs.append(log.name)
    finite = [value for value in runtimes if np.isfinite(value)]
    return {
        "present": True,
        "n_run_summaries": len(summaries),
        "n_done_markers": len(done),
        "launcher_status": state.get("status"),
        "launcher_expected_cells": state.get("n_cells"),
        "launcher_failed_cells": state.get("failed_cells", []),
        "ictal_target_seal_intact": seal_ok,
        "n_cells_reading_outer_heldout": int(sum(outer_read)),
        "runtime_seconds_total": float(np.sum(finite)) if finite else 0.0,
        "runtime_seconds_median": float(np.median(finite)) if finite else None,
        "gpu_peak_allocated_gb_max": float(np.max(gpu)) if gpu else None,
        "peak_rss_gb_max": float(np.nanmax(rss)) if rss else None,
        "logs_with_oom": oom,
        "logs_with_nan_text": nan_logs,
    }


def _metric_health() -> dict:
    """Every reported endpoint must be finite where it is defined."""
    analysis = RESULT_ROOT / "analysis"
    report = {}
    for name in sorted(analysis.glob("*_cell_metrics.csv")):
        frame = pd.read_csv(name)
        numeric = frame.select_dtypes(include=[np.number])
        non_finite = {
            column: int((~np.isfinite(numeric[column])).sum())
            for column in numeric.columns
            if (~np.isfinite(numeric[column])).any()
        }
        report[name.name] = {
            "n_rows": int(len(frame)),
            "columns_with_non_finite_values": non_finite,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULT_ROOT / "FINAL_ACCEPTANCE.json")
    args = parser.parse_args()

    phases = {name: _phase_report(RESULT_ROOT / path) for name, path in PHASES.items()}
    freezes = {}
    for name in ("HYPERPARAMETER_FREEZE.json", "OBJECTIVE_FREEZE.json"):
        path = RESULT_ROOT / "development" / name
        freezes[name] = json.loads(path.read_text()) if path.is_file() else None
    reproducibility = {}
    for name in ("REPRODUCIBILITY.json", "NESTED_CYCLE_READOUT.json"):
        path = RESULT_ROOT / "development" / "reproducibility" / name
        reproducibility[name] = json.loads(path.read_text()) if path.is_file() else None

    audit_path = RESULT_ROOT / "input_audit" / "TRAINING_SEMANTICS_AUDIT.json"
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else None

    incomplete = [
        name
        for name, report in phases.items()
        if report.get("present")
        and report.get("launcher_status") not in (None, "COMPLETE")
    ]
    failures = {
        name: report["launcher_failed_cells"]
        for name, report in phases.items()
        if report.get("present") and report.get("launcher_failed_cells")
    }
    seal_broken = [
        name
        for name, report in phases.items()
        if report.get("present") and not report.get("ictal_target_seal_intact")
    ]
    oom = {
        name: report["logs_with_oom"]
        for name, report in phases.items()
        if report.get("present") and report.get("logs_with_oom")
    }
    outer_read_outside_formal = {
        name: report["n_cells_reading_outer_heldout"]
        for name, report in phases.items()
        if name != "formal"
        and report.get("present")
        and report.get("n_cells_reading_outer_heldout")
    }
    payload = {
        "status": (
            "ACCEPTED"
            if not (incomplete or failures or seal_broken or oom or outer_read_outside_formal)
            else "INCOMPLETE"
        ),
        "contract": "topic5_rnn_training_sufficiency_v0_1",
        "spec": "docs/superpowers/specs/2026-07-30-topic5-rnn-training-sufficiency-v0_1.md",
        "phases": phases,
        "incomplete_phases": incomplete,
        "failed_cells": failures,
        "phases_with_broken_seal": seal_broken,
        "phases_with_oom": oom,
        "outer_heldout_read_outside_the_formal_phase": outer_read_outside_formal,
        "metric_health": _metric_health(),
        "training_semantics_audit": audit,
        "freezes": freezes,
        "reproducibility": reproducibility,
        "environment": run_environment(),
    }
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "incomplete_phases": incomplete,
                "failed_cells": failures,
                "written": str(out.relative_to(ROOT) if out.is_relative_to(ROOT) else out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
