#!/usr/bin/env python3
"""Regenerate the Axis-A A0 baseline fingerprint from the existing oneend read-outs.

Reproducible generator for the Stage A0 baseline (plan §1 + Stage A0). NO simulation:
it extracts the FROZEN fingerprint (src.sef_hfo_fingerprint) over the two existing
single-focus read-out artifacts and writes:
  - fingerprint/baseline_oneend.json          (schema dump + full per-run RunFingerprint)
  - fingerprint/baseline_oneend_summary.json   (provenance + aggregated primary features)

Run from repo root:  python scripts/run_sef_hfo_fingerprint_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sef_hfo_fingerprint import (  # noqa: E402
    extract_fingerprint,
    run_fingerprint_to_dict,
    propose_n_min_events,
    FINGERPRINT_SCHEMA,
    REQUIRES_EXTENDED_SAVE,
    SCHEMA_VERSION,
    PRIMARY_FEATURES,
    SECONDARY_FEATURES,
    DEFERRED_FEATURES,
)

BASE = ROOT / "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
OUT = BASE / "fingerprint"
ENGINE_VERSIONS = ROOT / "results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json"
PLAN = "docs/superpowers/plans/2026-06-14-sef-hfo-axisA-lesion-propagation-fingerprint.md (§1 + Stage A0)"
STATUS = "A0 schema FROZEN (user sign-off 2026-06-14, plan §0.1)"

RUNS = {  # source_label -> (readout json, rep npz)
    "neg": (BASE / "readout_oneend_neg_s1.json", BASE / "per_event/rep_oneend_neg_s1.npz"),
    "pos": (BASE / "readout_oneend_pos_s1.json", BASE / "per_event/rep_oneend_pos_s1.npz"),
}
SINGLE_FOCUS_COLLISION_NOTE = (
    "single-focus baseline: collisions N/A (one focus, no two-core co-ignition) -> 0")


def _summary_per_run(rf) -> dict:
    agg = rf.aggregate
    return {
        "source_label": rf.source_label,
        "seed": rf.seed,
        "config": rf.config,
        "n_events_total": rf.n_events_total,
        "n_clean_events": rf.n_clean_events,
        "clean_event_rate": rf.clean_event_rate,
        "collision_count": rf.collision_count,
        "collision_note": SINGLE_FOCUS_COLLISION_NOTE,
        "ambiguous_count": rf.ambiguous_count,
        "excluded_event_counts": rf.excluded_counts,
        "insufficient": rf.insufficient,
        "primary_aggregate": {
            "axis_dir": agg["axis_dir"],
            "pathway_width": agg["pathway_width"],
            "onset_jitter": agg["onset_jitter"],
        },
        "secondary_aggregate": {
            "n_part_median": agg["recruit_extent_secondary"]["n_part_median"],
            "along_span_median_mm": agg["recruit_extent_secondary"]["along_span_median_mm"],
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fingerprints = {lbl: extract_fingerprint(jp, npz) for lbl, (jp, npz) in RUNS.items()}

    # --- baseline_oneend.json: schema + full per-run RunFingerprint ---
    baseline = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "plan": PLAN,
        "schema": FINGERPRINT_SCHEMA,
        "requires_extended_readout_save": list(REQUIRES_EXTENDED_SAVE),
        "runs": {lbl: run_fingerprint_to_dict(rf) for lbl, rf in fingerprints.items()},
    }
    (OUT / "baseline_oneend.json").write_text(json.dumps(baseline, indent=2, ensure_ascii=False))

    # --- baseline_oneend_summary.json: provenance + aggregated view ---
    engine_full_sha = json.loads(ENGINE_VERSIONS.read_text()) if ENGINE_VERSIONS.exists() else {}
    n_min = propose_n_min_events({rf.tag: rf.n_clean_events for rf in fingerprints.values()})
    n_min["rationale"] = n_min["rationale"].replace(
        "Pending user sign-off; revisable at schema-freeze review.",
        "Approved at user sign-off 2026-06-14 as the A1/A3/A4 floor (plan §0.1).")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "plan": PLAN,
        "tier_lock": {
            "primary": list(PRIMARY_FEATURES),
            "secondary": list(SECONDARY_FEATURES),
            "deferred_not_primary": list(DEFERRED_FEATURES),
            "amplitude_proxy": "provenance/diagnostic only — never a primary comparison",
        },
        "engine_signature": {
            "readout_provenance_engine_sha": {
                lbl: (rf.engine_signature or {}) for lbl, rf in fingerprints.items()},
            "engine_versions_baseline_full_sha": engine_full_sha,
        },
        "per_run": {rf.tag: _summary_per_run(rf) for rf in fingerprints.values()},
        "n_min_events": n_min,
        "requires_extended_readout_save": {
            k: {"tier": FINGERPRINT_SCHEMA[k]["tier"],
                "what_to_persist": FINGERPRINT_SCHEMA[k]["what_to_persist"]}
            for k in REQUIRES_EXTENDED_SAVE},
    }
    (OUT / "baseline_oneend_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    for lbl, rf in fingerprints.items():
        print(f"{rf.tag}: n_clean={rf.n_clean_events}/{rf.n_events_total} "
              f"rate={rf.clean_event_rate} pathway_width_med="
              f"{rf.aggregate['pathway_width']['median_mm']} "
              f"sign_majority={rf.aggregate['axis_dir']['sign_majority']}")
    print(f"wrote {OUT/'baseline_oneend.json'} + baseline_oneend_summary.json "
          f"(schema_version={SCHEMA_VERSION})")


if __name__ == "__main__":
    main()
