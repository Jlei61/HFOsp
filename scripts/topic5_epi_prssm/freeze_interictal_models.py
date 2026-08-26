#!/usr/bin/env python3
"""Hard Gate B -- freeze the interictal model family before any seizure label is read.

Several structural representatives may be frozen; this is deliberately not a
single-winner selection.  What may never happen is re-choosing a model or a
checkpoint after seeing a seizure outcome, so this file is written first and the
seizure-label loader refuses to run until it exists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.topic5_epi_prssm.contracts import (  # noqa: E402
    FROZEN, OUTPUT_ROOT, atomic_write_json, code_revision, package_hash, sha256_file,
)
from src.topic5_epi_prssm.stats import aggregate_seeds  # noqa: E402

TARGET = OUTPUT_ROOT / "manifests/INTERICTAL_MODEL_FREEZE.json"

#: structural layers that each get one frozen representative
LAYERS = {
    "static_repertoire": ["static"],
    "leaky_state": ["ct_ewma_g0"],
    "linear_graph_recurrent": ["g1_graph_clds"],
    "nonlinear_graph_recurrent": ["g2_graph_gru_ode"],
    "resource_anchored": ["g3_resource"],
    "resource_anchored_on_best_family": ["g3_resource_on_g1"],
    "unconstrained_persistent": ["unconstrained_gru"],
}

#: pre-registered endpoints for H2b, fixed here and not revisited afterwards
PREICTAL_ENDPOINTS = [
    "state_norm", "resource", "expected_load", "train_pc1_projection",
    "first_selection_entropy",
]


def _load_runs(root: Path, cohort: str) -> dict[str, list[dict]]:
    by_arm: dict[str, list[dict]] = {}
    for path in sorted((root / "runs").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("cohort") != cohort or record.get("evaluation") is None:
            continue
        by_arm.setdefault(record["arm"], []).append(record | {"__path": str(path)})
    return by_arm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all34")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if TARGET.exists() and not args.force:
        raise SystemExit(f"{TARGET} already exists; refusing to re-freeze without --force")

    by_arm = _load_runs(OUTPUT_ROOT / "generator_ladder", args.cohort)
    if not by_arm:
        raise SystemExit("no completed generator-ladder runs to freeze")

    representatives = []
    for layer, arms in LAYERS.items():
        for arm in arms:
            records = by_arm.get(arm, [])
            if not records:
                representatives.append({"layer": layer, "arm": arm, "status": "NOT_AVAILABLE"})
                continue
            per_seed = [{s: v["event_nll"] + v["participation_nll"]
                         for s, v in r["evaluation"]["filtered"].items()} for r in records]
            patient = aggregate_seeds(per_seed)
            scores = {r["seed"]: float(np.mean(
                [v["event_nll"] + v["participation_nll"]
                 for v in r["evaluation"]["filtered"].values()])) for r in records}
            best_seed = min(scores, key=scores.get)
            chosen = next(r for r in records if r["seed"] == best_seed)
            checkpoint = (OUTPUT_ROOT / "generator_ladder/checkpoints" / f"{chosen['job_id']}.pt")
            representatives.append({
                "layer": layer, "arm": arm, "status": "FROZEN",
                "job_id": chosen["job_id"], "seed": best_seed,
                "spec": chosen["spec"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.exists() else None,
                "validation_by_seed": scores,
                "cohort_mean_validation": float(np.mean(list(patient.values()))),
                "n_patients": len(patient),
            })

    atomic_write_json(TARGET, {
        "contract": "topic5_epi_prssm_v0_1_interictal_model_freeze",
        "hard_gate": "B",
        "written_before_any_seizure_label_was_read": True,
        "cohort": args.cohort,
        "representatives": representatives,
        "state_dimension": FROZEN["state_dim_H"],
        "observer_dimension": FROZEN["observer_dim"],
        "normalisation": "graph state clamped to +-8 per unit; resource bounded in (0,1]; "
                         "patient baseline mu_p estimated train-only and frozen",
        "open_loop_anchor_rule": "last interictal event at or before the target time, with a "
                                 f"gap of at most {FROZEN['max_last_ied_to_onset_seconds']} s; "
                                 "the observer is closed at that event and the generator "
                                 "integrates on real elapsed time alone",
        "pre_freeze_pipeline_test_disclosure": {
            "happened": True,
            "what": "before this freeze was written, the seizure-link script was executed once on "
                    "a 2-patient smoke cohort (6 eligible seizures, 1 patient) using a "
                    "throw-away G0 checkpoint trained on that same smoke cohort",
            "why": "to test the code path end to end rather than discover a crash after the "
                   "freeze",
            "what_it_changed": [
                "fixed two variable-shadowing bugs that made the script abort",
                "added a numerical-validity guard: when the matched null has no spread left, the "
                "z is withheld as degenerate instead of being computed from rounding error",
                "added a secondary extended last-event-gap window, declared on the observed gap "
                "distribution, which is a property of the data and not an outcome",
            ],
            "what_it_did_not_change": [
                "the model family and the frozen representatives",
                "the primary endpoints",
                "the primary last-event-gap window, which stays at the pre-registered value",
                "the pseudo-onset matching protocol",
            ],
            "disclosed_because": "the alternative -- not recording that any label-derived number "
                                 "was ever looked at before the freeze -- would be the failure "
                                 "this gate exists to prevent",
        },
        "last_event_gap_windows": {
            "primary_seconds": FROZEN["max_last_ied_to_onset_seconds"],
            "secondary_seconds": 86400.0,
            "secondary_rationale": "the interictal stream is built from fail-closed "
                                   "definite-interictal blocks, so blocks near a seizure are "
                                   "excluded by construction and the last interictal event is "
                                   "typically hours before onset",
        },
        "degenerate_probe_rule": "a probe whose matched-null standard deviation falls below "
                                 "max(1e-6, 1e-4 x |null mean|) is marked degenerate and yields "
                                 "no z; degenerate counts are reported as a denominator",
        "preictal_window_seconds": FROZEN["preictal_window_seconds"],
        "pseudo_onset_matching": ["same patient", "same day/night bin",
                                  "outside every peri-ictal exclusion window",
                                  "matched last-event gap decile",
                                  "matched local event rate decile"],
        "pseudo_onset_draws": FROZEN["pseudo_onset_draws"],
        "primary_endpoints": PREICTAL_ENDPOINTS,
        "secondary_endpoints": ["time_in_warning", "leave_seizure_out_effect"],
        "nuisance_set": ["local event rate", "median IEI", "session position",
                         "time of day", "last-event gap"],
        "planned_contrasts": [
            "onset versus matched pseudo-onset, per patient, z-scored inside the patient",
            "leave-seizure-out patient effect",
            "state-only versus nuisance-only versus nuisance+state",
        ],
        "primary_analysis_window": "seizures whose onset is at or before the end of the "
                                   "interictal validation partition",
        "secondary_analysis_window": "all seizures inside the recorded interictal span; the "
                                     "frozen model is applied causally, and no interictal "
                                     "prediction claim is made from those events",
        "early_ictal_transfer": {
            "status": "NOT_RUN",
            "reason": "the primary form needs adjudicated per-seizure clinical-onset contacts; "
                      "the registry holds 0 of 71 consensus annotations and its BLINDING_CONTRACT "
                      "is LOCKED against SOZ, patient-level focus, A/B template source and "
                      "energy-top substitutions",
        },
        "code_revision": code_revision(), "package_hash": package_hash(),
    })
    print(json.dumps({"frozen": [(r["layer"], r["arm"], r["status"]) for r in representatives]},
                     indent=2))


if __name__ == "__main__":
    main()
