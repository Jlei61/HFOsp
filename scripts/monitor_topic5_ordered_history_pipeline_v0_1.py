#!/usr/bin/env python3
"""Persist a compact stage-by-stage status for the ordered-history audit."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_ordered_history_architecture_audit"
STAGES = {
    "fixed_hidden_architectures": (
        BASE / "formal/architecture_controls_formal_20260729",
        204,
    ),
    "selected_matched_rank_shuffle": (
        BASE / "rank_shuffle/selected_architecture_rank_shuffle_20260729",
        102,
    ),
    "history_interventions": (
        BASE / "interventions/selected_history_interventions_20260729",
        102,
    ),
    "parameter_matched_sensitivity": (
        BASE / "parameter_matched/parameter_matched_formal_20260729",
        204,
    ),
}
ERROR_TOKENS = ("Traceback", "CUDA out of memory", "OutOfMemoryError", "nan loss")


def memory_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024**2
    return float("nan")


def gpu() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.check_output(command, text=True, timeout=5).strip()
        memory, utilization, temperature = [
            float(value.strip()) for value in line.split(",")
        ]
        return {
            "memory_used_mb": memory,
            "utilization_percent": utilization,
            "temperature_c": temperature,
        }
    except Exception:
        return {}


def error_hits(root: Path) -> list[str]:
    hits = []
    if not root.exists():
        return hits
    for path in root.rglob("*.log"):
        try:
            with path.open("rb") as handle:
                handle.seek(max(path.stat().st_size - 65536, 0))
                text = handle.read().decode(errors="replace")
        except OSError:
            continue
        if any(token in text for token in ERROR_TOKENS):
            hits.append(str(path.relative_to(ROOT)))
    return sorted(hits)[:20]


def snapshot() -> dict:
    stages = {}
    all_errors = []
    for label, (path, expected) in STAGES.items():
        completed = len(list(path.rglob("DONE.json"))) if path.exists() else 0
        launcher = path / "LAUNCHER_DONE.json"
        launcher_payload = json.loads(launcher.read_text()) if launcher.exists() else None
        errors = error_hits(path)
        all_errors.extend(errors)
        stages[label] = {
            "expected_cells": expected,
            "completed_cells": completed,
            "launcher": launcher_payload,
            "started": path.exists(),
        }
    analysis = BASE / "analysis"
    products = {
        "architecture_summary": (analysis / "ARCHITECTURE_SUMMARY.json").exists(),
        "history_intervention_summary": (
            analysis / "HISTORY_INTERVENTION_SUMMARY.json"
        ).exists(),
        "early_ictal_conditional_summary": (
            analysis / "EARLY_ICTAL_CONDITIONAL_SUMMARY.json"
        ).exists(),
        "parameter_matched_summary": (
            analysis / "PARAMETER_MATCHED_SENSITIVITY.json"
        ).exists(),
        "paper_figure": (
            ROOT
            / "results/paper-ready-figure/"
            "fig6_ordered_history_architecture_audit/figures/"
            "fig6_ordered_history_architecture_audit.png"
        ).exists(),
        "final_acceptance": (BASE / "FINAL_ACCEPTANCE.json").exists(),
    }
    complete = (
        (BASE / "watcher/WATCHER_DONE.json").exists()
        and all(products.values())
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE" if complete else "RUNNING",
        "stages": stages,
        "products": products,
        "recent_error_hits": sorted(set(all_errors)),
        "memory_available_gb": memory_available_gb(),
        "gpu": gpu(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=float, default=20)
    args = parser.parse_args()
    output = BASE / "PIPELINE_STATUS.json"
    while True:
        value = snapshot()
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(output)
        if value["status"] == "COMPLETE":
            break
        time.sleep(float(args.interval_seconds))


if __name__ == "__main__":
    main()
