#!/usr/bin/env python3
"""Freeze addendum for Goal 3b -- the pre-ictal observation arm.

This is an addendum, not a replacement: the original
``INTERICTAL_MODEL_FREEZE.json`` is read, referenced by hash, and left untouched.
The model family, checkpoints and per-contact parameters are exactly the ones that
were frozen before any label was read; the only thing this addendum declares is a
different *observation stream* and the lead times at which the observer is closed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_epi_prssm.contracts import (  # noqa: E402
    OUTPUT_ROOT, atomic_write_json, code_revision, package_hash, sha256_file,
)

BASE = OUTPUT_ROOT / "manifests/INTERICTAL_MODEL_FREEZE.json"
TARGET = OUTPUT_ROOT / "manifests/INTERICTAL_MODEL_FREEZE_ADDENDUM_GOAL3B.json"
STREAM_MANIFEST = OUTPUT_ROOT / "full_event_stream/FULL_STREAM_MANIFEST.json"

PRIMARY_LEAD_MINUTES = 30.0
AUXILIARY_LEAD_MINUTES = (60.0, 15.0, 5.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not BASE.exists():
        raise SystemExit(f"{BASE} missing: the base interictal freeze must exist first")
    if TARGET.exists() and not args.force:
        raise SystemExit(f"{TARGET} already exists; refusing to re-freeze without --force")
    base = json.loads(BASE.read_text())

    atomic_write_json(TARGET, {
        "contract": "topic5_epi_prssm_v0_1_interictal_model_freeze_addendum_goal3b",
        "hard_gate": "B (addendum)",
        "base_freeze": str(BASE),
        "base_freeze_sha256": sha256_file(BASE),
        "base_freeze_untouched": True,
        "what_changes": "only the observation stream and the observer cut-off; the model family, "
                        "the checkpoints, the patient baselines, the graphs and the state "
                        "dimension are exactly those in the base freeze",
        "why": (
            "the base Goal 3 arm runs on the definite-interictal stream, whose block policy "
            "deletes every block overlapping a seizure or its 120 min post-ictal guard, every "
            "block crossing a local day/night boundary, and both neighbours of any recording "
            "discontinuity longer than 5400 s.  The pre-ictal observations an online system "
            "would have had are therefore absent, and that arm ends up asking how long a state "
            "survives without observation rather than whether the state moves once the "
            "pre-ictal IEDs are observed."),
        "observation_stream": {
            "source": str(STREAM_MANIFEST),
            "sha256": sha256_file(STREAM_MANIFEST) if STREAM_MANIFEST.exists() else None,
            "rule": "every rebuilt event except those inside a seizure or inside the frozen "
                    "120 min post-ictal guard; pre-ictal events are kept",
            "encoding": "identical producer functions and constants as dataset_v0_4; parity "
                        "with the frozen stream is asserted per subject on participation, "
                        "group identity and rank",
            "seizure_labels_used_to_build_the_stream": False,
        },
        "observer_cutoff": {
            "primary_lead_minutes": PRIMARY_LEAD_MINUTES,
            "auxiliary_lead_minutes": list(AUXILIARY_LEAD_MINUTES),
            "rule": "events at or before onset minus the lead may update the observer; from the "
                    "cut-off to onset only the generator integrates, on real elapsed time",
            "onset_time_use": "alignment and scoring only; the onset time never enters the model "
                              "as an input, a target or a selection signal",
        },
        "reported_separately": [
            "filtered_at_cutoff: the observer-updated state at the cut-off, which is what an "
            "online system would hold at that moment",
            "open_loop_at_onset: the same state integrated autonomously from the cut-off to "
            "onset with no further observation",
        ],
        "primary_endpoints": base.get("primary_endpoints"),
        "nuisance_set": [
            "local event rate over 30 min, 2 h, 4 h and 8 h look-back windows",
            "median inter-event interval over the 2 h look-back window",
            "observation coverage of the look-back window",
            "day or night by local time",
            "position inside the recording session",
            "gap from the last admissible event to the cut-off",
        ],
        "pseudo_onset_matching": [
            "same patient", "same recording session",
            "same day/night bin",
            "outside every peri-ictal exclusion window",
            "matched observation coverage decile",
            "matched multi-scale rate (30 min, 2 h, 4 h, 8 h)",
            "matched median inter-event interval",
            "matched last-event gap decile",
        ],
        "planned_contrasts": [
            "per-patient median z at onset versus matched pseudo-onsets, per endpoint",
            "the same z after residualising on the nuisance set",
            "the nuisance set alone, reported as its own row so a state claim must beat it",
            "leave-seizure-out stability",
            "filtered-at-cutoff versus open-loop-at-onset, never pooled",
        ],
        "claim_rule": (
            "a state claim requires the state endpoint to survive residualisation on the "
            "multi-scale rate and interval nuisances; Topic 2 already establishes that the "
            "event rate itself drifts slowly and rises around seizures, so an unresidualised "
            "state effect is not evidence for a spatial-repertoire state"),
        "not_observable_rule": (
            "if a patient has no seizure with an admissible event inside the look-back window, "
            "that patient is recorded as NOT_OBSERVABLE_FROM_CURRENT_STREAM rather than as a "
            "negative"),
        "code_revision": code_revision(), "package_hash": package_hash(),
    })
    print(json.dumps({"wrote": str(TARGET),
                      "primary_lead_minutes": PRIMARY_LEAD_MINUTES,
                      "auxiliary_lead_minutes": list(AUXILIARY_LEAD_MINUTES)}, indent=2))


if __name__ == "__main__":
    main()
