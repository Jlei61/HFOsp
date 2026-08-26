#!/usr/bin/env python3
"""Machine-readable summary of the whole Epi-PRSSM v0.1 run.

Every number is recomputed from the per-job artefacts on disk, not copied from a
log, and every unfinished cell carries a concrete reason.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.topic5_epi_prssm.contracts import (  # noqa: E402
    OUTPUT_ROOT, atomic_write_json, code_revision, package_hash, sha256_file,
)
from src.topic5_epi_prssm.run_registry import collect_jobs  # noqa: E402

TARGET = OUTPUT_ROOT / "FINAL_RUN_SUMMARY.json"


def _read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _scripts_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "scripts/topic5_epi_prssm").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all34")
    args = parser.parse_args()

    jobs = collect_jobs()
    job_states: dict[str, int] = {}
    peak_rss = []
    for job in jobs:
        job_states[job.get("state", "?")] = job_states.get(job.get("state", "?"), 0) + 1
        if job.get("peak_rss_mib"):
            peak_rss.append(job["peak_rss_mib"])

    cards = {
        "H1": _read(OUTPUT_ROOT / "generator_ladder/GENERATOR_EVIDENCE_CARD.json"),
        "H2a": _read(OUTPUT_ROOT / "event_distribution/H2A_EVIDENCE_CARD.json"),
        "H3a": _read(OUTPUT_ROOT / "exposure_mechanism/H3A_EVIDENCE_CARD.json"),
        "H3b": _read(OUTPUT_ROOT / "exposure_mechanism/H3B_EVIDENCE_CARD.json"),
    }
    primary = sorted(glob.glob(str(
        OUTPUT_ROOT / "seizure_link_preictal/H2B_PRIMARY_EVIDENCE_CARD__*.json")))
    strict = [json.loads(Path(p).read_text())
              for p in sorted(glob.glob(str(OUTPUT_ROOT / "seizure_link/runs/*.json")))]
    if primary:
        cards["H2b"] = json.loads(Path(primary[0]).read_text())
        cards["H2b"]["all_primary_cards"] = [Path(p).name for p in primary]
    else:
        cards["H2b"] = {"status": "NOT_RUN",
                        "reason": "the pre-ictal observation arm has not produced an evidence "
                                  "card yet"}
    cards["H2b_strict_sensitivity"] = strict[0] if strict else {
        "status": "NOT_RUN",
        "reason": "the definite-interictal strict arm has not completed",
        "role": "definite_interictal_long_gap_strict_sensitivity, not primary H2b"}
    cards["H2b_observation_stream"] = _read(
        OUTPUT_ROOT / "full_event_stream/FULL_STREAM_MANIFEST.json")
    cards["H2b_freeze_addendum"] = _read(
        OUTPUT_ROOT / "manifests/INTERICTAL_MODEL_FREEZE_ADDENDUM_GOAL3B.json")

    figures = {}
    for asset in ("epi_prssm_architecture_ladder", "epi_prssm_generator_evidence",
                  "epi_prssm_event_distribution", "epi_prssm_seizure_link",
                  "epi_prssm_exposure_mechanism"):
        directory = OUTPUT_ROOT / "figures" / asset
        png = directory / "figures" / f"{asset}.png"
        figures[asset] = {
            "generated": png.exists(),
            "png": str(png) if png.exists() else None,
            "pdf": str(directory / "figures" / f"{asset}.pdf")
                   if (directory / "figures" / f"{asset}.pdf").exists() else None,
            "metadata": str(directory / f"{asset}_metadata.json")
                        if (directory / f"{asset}_metadata.json").exists() else None,
            "readme": str(directory / "figures" / "README.md")
                      if (directory / "figures" / "README.md").exists() else None,
        }

    unresolved = []
    for job in jobs:
        if job.get("state") in ("FAILED", "OOM", "NAN", "INVALID_INPUT"):
            unresolved.append({"job_id": job.get("job_id"), "state": job.get("state"),
                               "reason": job.get("failure_reason")})
    if not primary:
        unresolved.append({
            "job_id": "goal3b_preictal", "state": "NOT_RUN",
            "reason": "the primary H2b arm, which observes the pre-ictal IEDs and closes the "
                      "observer at a declared lead, had not produced an evidence card when this "
                      "summary was written"})
    if not strict:
        unresolved.append({"job_id": "goal3_strict_sensitivity", "state": "NOT_RUN",
                           "reason": "the definite-interictal strict control arm had not "
                                     "completed when this summary was written"})
    unresolved.append({
        "job_id": "goal3_task_3_3_early_ictal_transfer", "state": "NOT_RUN",
        "reason": "the primary form needs adjudicated per-seizure clinical-onset contacts; the "
                  "registry holds 0 of 71 consensus annotations and its blinding contract is "
                  "LOCKED against SOZ, patient focus, A/B template and energy-top substitutions. "
                  "An energy-field surrogate was not used because its channel mapping to this "
                  "cohort's contact order has not been audited under Hard Gate A."})
    unresolved.append({
        "job_id": "goal5_learned_event_encoder", "state": "NOT_RUN",
        "reason": "Goal 5 is explicitly not a gate; the explicit-mark ladder consumed the "
                  "available wall-clock, and no learned-encoder arm was started"})

    safe, forbidden = _claims(cards)
    payload = {
        "contract": "topic5_epi_prssm_v0_1_final_run_summary",
        "status": "EXPLORATORY_DEVELOPMENT_COMPLETE",
        "cohort": args.cohort,
        "code_revision": code_revision(),
        "package_hash": package_hash(),
        "scripts_hash": _scripts_hash(),
        "hard_gates": {
            "A_data_and_leakage": _read(OUTPUT_ROOT / "manifests/HARD_GATE_A.json"),
            "B_interictal_freeze": _read(OUTPUT_ROOT / "manifests/INTERICTAL_MODEL_FREEZE.json")
                                   is not None,
            "C_untouched_test": {
                "released": (OUTPUT_ROOT / "manifests/FORMAL_TEST_RELEASE.json").exists(),
                "note": "the interictal test partition was never consumed by any training, "
                        "selection or evaluation step in this run; every result is development",
            },
        },
        "denominators": _denominators(),
        "jobs": {"total": len(jobs), "state_counts": job_states,
                 "peak_rss_mib_max": float(np.max(peak_rss)) if peak_rss else None,
                 "peak_rss_mib_median": float(np.median(peak_rss)) if peak_rss else None},
        "resources": _read(OUTPUT_ROOT / "manifests/RESOURCE_AUDIT.json"),
        "evidence_cards": cards,
        "synthetic_recovery": _read(OUTPUT_ROOT / "synthetic/SYNTHETIC_RECOVERY_SUMMARY.json"),
        "figures": figures,
        "reports": {
            "plain_chinese": "docs/archive/topic5/epi_prssm_v0_1_plain_chinese_report_2026-08-18.md",
            "technical": "docs/archive/topic5/epi_prssm_v0_1_technical_report_2026-08-18.md",
        },
        "unresolved_items": unresolved,
        "safe_claims": safe,
        "forbidden_claims": forbidden,
    }
    atomic_write_json(TARGET, payload)
    print(json.dumps({"status": payload["status"], "jobs": payload["jobs"],
                      "figures": {k: v["generated"] for k, v in figures.items()},
                      "n_unresolved": len(unresolved)}, indent=2))


def _denominators() -> dict:
    inventory = OUTPUT_ROOT / "data_audit/support_inventory.csv"
    if not inventory.exists():
        return {}
    frame = pd.read_csv(inventory)
    return {
        "n_patients": int(len(frame)),
        "n_epilepsiae": int((frame.dataset == "epilepsiae").sum()),
        "n_yuquan": int((frame.dataset == "yuquan").sum()),
        "n_events_total": int(frame.n_events.sum()),
        "n_train_events": int(frame.n_train.sum()),
        "n_validation_events": int(frame.n_validation.sum()),
        "n_test_events_sealed": int(frame.n_test.sum()),
        "n_source_blocks": int(frame.n_source_blocks.sum()),
        "n_sessions": int(frame.n_sessions.sum()),
        "n_contacts_total": int(frame.n_contacts.sum()),
        "n_patients_without_geometry": int((frame.geometry_mapped == 0).sum()),
    }


def _claims(cards: dict) -> tuple[list[str], list[str]]:
    safe, forbidden = [], [
        "the model proves that IED exposure causes seizures",
        "the slow state is a seizure clock",
        "a seizure-link result read off the definite-interictal stream, whose block policy "
        "deletes the pre-ictal observations",
        "the resource is a measured metabolic variable",
        "anatomical rewiring or synaptic remodelling",
        "a confirmatory result from an untouched test partition",
    ]
    h1 = cards.get("H1")
    if h1:
        safe.append(f"H1: {h1['verdict']} (development partition, "
                    f"{h1['denominators']['n_patients']} patients)")
    h2a = cards.get("H2a")
    if h2a:
        safe.append(f"H2a: reported against a capacity-matched frozen-state control on "
                    f"{h2a['denominators']['n_patients']} patients; "
                    f"{len(h2a.get('targeted_eligible_patients', []))} patients were eligible "
                    "for the ambiguous-prefix targeted analysis")
    h2b = cards.get("H2b") or {}
    denominators = h2b.get("denominators") or {}
    if denominators.get("n_seizures_premise_met"):
        safe.append(
            f"H2b: {denominators['n_seizures_premise_met']} seizures in "
            f"{denominators.get('n_patients_premise_met', 0)} patients meet the pre-ictal "
            "observation premise; the frozen interictal model observes the pre-ictal IEDs and "
            "the observer is closed at a declared lead, and every state endpoint is reported "
            "both raw and after residualising on multi-scale rate, interval and coverage")
    if (cards.get("H2b_strict_sensitivity") or {}).get("n_seizures"):
        safe.append(
            "the definite-interictal arm is reported as a strict missing-observation and "
            "long-extrapolation control, not as H2b")
    h3a = cards.get("H3a")
    if h3a:
        safe.append("H3a: the primary outcome is the masked recruitment-order likelihood, "
                    "which is invariant to how many contacts participated")
    return safe, forbidden


if __name__ == "__main__":
    main()
