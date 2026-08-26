#!/usr/bin/env python3
"""Generate the plain-language and technical reports from the artefacts on disk.

Both reports and FINAL_RUN_SUMMARY.json are produced from the same reader
functions, so the three cannot drift apart.  Nothing here is transcribed from a
log.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.topic5_epi_prssm.contracts import (  # noqa: E402
    OUTPUT_ROOT, atomic_write_text, code_revision, package_hash,
)

ARCHIVE = ROOT / "docs/archive/topic5"
DATE = "2026-08-18"
PLAIN = ARCHIVE / f"epi_prssm_v0_1_plain_chinese_report_{DATE}.md"
TECH = ARCHIVE / f"epi_prssm_v0_1_technical_report_{DATE}.md"

def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        pass
    return pd.DataFrame()


def gather() -> dict:
    seizure_paths = sorted(glob.glob(str(OUTPUT_ROOT / "seizure_link/runs/*.json")))
    return {
        "gate_a": read_json(OUTPUT_ROOT / "manifests/HARD_GATE_A.json"),
        "data": read_json(OUTPUT_ROOT / "manifests/DATA_MANIFEST.json"),
        "split": read_json(OUTPUT_ROOT / "manifests/SPLIT_MANIFEST.json"),
        "forbidden": read_json(OUTPUT_ROOT / "manifests/FORBIDDEN_INPUT_AUDIT.json"),
        "freeze": read_json(OUTPUT_ROOT / "manifests/INTERICTAL_MODEL_FREEZE.json"),
        "tau_freeze": read_json(OUTPUT_ROOT / "manifests/RESOURCE_TAU_FREEZE.json"),
        "inventory": read_csv(OUTPUT_ROOT / "data_audit/support_inventory.csv"),
        "prefix_inventory": read_csv(
            OUTPUT_ROOT / "data_audit/ambiguous_prefix_inventory_train_only.csv"),
        "variance": read_csv(OUTPUT_ROOT / "baseline/patient_repertoire_variance.csv"),
        "synthetic": read_json(OUTPUT_ROOT / "synthetic/SYNTHETIC_RECOVERY_SUMMARY.json"),
        "h1": read_json(OUTPUT_ROOT / "generator_ladder/GENERATOR_EVIDENCE_CARD.json"),
        "h1_runs": read_csv(OUTPUT_ROOT / "generator_ladder/model_runs.csv"),
        "h1_effects": read_csv(OUTPUT_ROOT / "generator_ladder/patient_effects.csv"),
        "h1_open_loop": read_csv(OUTPUT_ROOT / "generator_ladder/open_loop_horizon.csv"),
        "h1_reset": read_csv(OUTPUT_ROOT / "generator_ladder/state_reset.csv"),
        "h1_shuffle": read_csv(OUTPUT_ROOT / "generator_ladder/delta_t_shuffle.csv"),
        "h2a": read_json(OUTPUT_ROOT / "event_distribution/H2A_EVIDENCE_CARD.json"),
        "h2a_effects": read_csv(OUTPUT_ROOT / "event_distribution/full_event_effects.csv"),
        "h2a_swaps": read_csv(OUTPUT_ROOT / "event_distribution/state_swap_effects.csv"),
        "h2a_prefix": read_csv(OUTPUT_ROOT / "event_distribution/ambiguous_prefix_effects.csv"),
        "h2b_strict": [read_json(Path(p)) for p in seizure_paths],
        "h2b": [read_json(Path(p)) for p in sorted(glob.glob(
            str(OUTPUT_ROOT / "seizure_link_preictal/H2B_PRIMARY_EVIDENCE_CARD__*.json")))],
        "h2b_preictal_frames": {Path(p).stem: read_csv(Path(p)) for p in sorted(glob.glob(
            str(OUTPUT_ROOT / "seizure_link_preictal/preictal_effects__*.csv")))},
        "full_stream": read_json(OUTPUT_ROOT / "full_event_stream/FULL_STREAM_MANIFEST.json"),
        "goal3b_addendum": read_json(
            OUTPUT_ROOT / "manifests/INTERICTAL_MODEL_FREEZE_ADDENDUM_GOAL3B.json"),
        "h3a": read_json(OUTPUT_ROOT / "exposure_mechanism/H3A_EVIDENCE_CARD.json"),
        "h3b": read_json(OUTPUT_ROOT / "exposure_mechanism/H3B_EVIDENCE_CARD.json"),
        "h3_ladder": read_csv(OUTPUT_ROOT / "exposure_mechanism/resource_ladder.csv"),
        "h3_effects": read_csv(OUTPUT_ROOT / "exposure_mechanism/t1_t2_patient_effects.csv"),
        "h3_curve": read_csv(OUTPUT_ROOT / "exposure_mechanism/exposure_timescale_curve.csv"),
        "innovation": read_json(
            OUTPUT_ROOT / "exposure_mechanism/innovation_controls_summary.json"),
        "summary": read_json(OUTPUT_ROOT / "FINAL_RUN_SUMMARY.json"),
    }


from report_sections import plain_report, technical_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all34")
    args = parser.parse_args()
    data = gather()
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    atomic_write_text(PLAIN, plain_report(data))
    atomic_write_text(TECH, technical_report(data, args.cohort))
    print(f"wrote {PLAIN}")
    print(f"wrote {TECH}")


if __name__ == "__main__":
    main()
