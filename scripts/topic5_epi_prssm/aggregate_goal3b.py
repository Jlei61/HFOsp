#!/usr/bin/env python3
"""Aggregate Goal 3b into the primary H2b evidence card.

Three readings are kept apart throughout.  Every state endpoint is reported twice:
raw, and after residualising on the same nuisances the pseudo cut-offs were matched
on.  The nuisance set's own discriminability is reported as its own row, so a state
claim has to beat it rather than ride on it.

The primary analysis set is the one where the pre-ictal observation premise actually
holds -- enough admissible events in the look-back window, and an anchor close enough
to the cut-off.  A seizure that fails it is back in the strict arm's situation and is
reported in its own stratum, never pooled and never silently dropped.
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
import pandas as pd  # noqa: E402

from src.topic5_epi_prssm.contracts import (  # noqa: E402
    OUTPUT_ROOT, atomic_write_csv, atomic_write_json, code_revision, package_hash,
)
from src.topic5_epi_prssm.evaluate import PROBE_ENDPOINTS  # noqa: E402
from src.topic5_epi_prssm.stats import holm, paired_effect, stratify  # noqa: E402

OUT = OUTPUT_ROOT / "seizure_link_preictal"
READINGS = ("filtered_at_onset", "filtered_at_cutoff", "open_loop_at_onset")
NUISANCE_Z = ("rate_1800s", "rate_7200s", "rate_14400s", "rate_28800s",
              "median_iei", "coverage")
NUISANCE_COLUMNS = ["nuisance_rate_1800s", "nuisance_rate_7200s", "nuisance_rate_14400s",
                    "nuisance_rate_28800s", "nuisance_median_iei", "nuisance_coverage",
                    "nuisance_log_anchor_gap"]


def residualise(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    keep = np.isfinite(values) & np.isfinite(design).all(axis=1)
    out = np.full(len(values), np.nan)
    if keep.sum() < design.shape[1] + 3:
        return out
    A = np.column_stack([np.ones(int(keep.sum())), design[keep]])
    coefficients, *_ = np.linalg.lstsq(A, values[keep], rcond=None)
    out[keep] = values[keep] - A @ coefficients + coefficients[0]
    return out


def leave_one_out(frame: pd.DataFrame, column: str) -> dict:
    medians = []
    for drop in range(len(frame)):
        subset = frame.drop(frame.index[drop])
        if subset.empty:
            continue
        medians.append(float(subset.groupby("subject")[column].median().median()))
    if not medians:
        return {"status": "insufficient"}
    return {"n_folds": len(medians), "min_median": float(np.min(medians)),
            "max_median": float(np.max(medians)),
            "sign_stable": bool(np.sign(np.min(medians)) == np.sign(np.max(medians)))}


def analyse(frame: pd.DataFrame, dataset: dict, family: dict | None = None) -> dict:
    design_raw = frame[NUISANCE_COLUMNS].to_numpy(dtype=float)
    design = np.column_stack([np.log1p(np.clip(design_raw[:, :4], 0, None)), design_raw[:, 4:]])
    blocks: dict[str, dict] = {}
    for reading in READINGS:
        block: dict[str, dict] = {}
        for endpoint in PROBE_ENDPOINTS:
            column = f"{reading}__{endpoint}_z"
            if column not in frame:
                continue
            degenerate = frame.get(f"{reading}__{endpoint}_degenerate", pd.Series(False,
                                                                                 index=frame.index))
            # an endpoint that is missing for this layer arrives as an all-None object
            # column, and isfinite refuses object dtype; coercing first keeps the
            # "nothing usable here" case a result rather than a crash
            values = pd.to_numeric(frame[column], errors="coerce")
            usable = frame[np.isfinite(values)]
            if usable.empty:
                block[endpoint] = {"status": "all_degenerate",
                                   "n_degenerate": int(degenerate.sum())}
                continue
            per_patient = usable.groupby("subject")[column].median().to_dict()
            raw = paired_effect(per_patient, {s: 0.0 for s in per_patient},
                                label=f"{reading}::{endpoint}", lower_is_better=False)
            residual = residualise(frame[column].to_numpy(dtype=float), design)
            adjusted = None
            residual_frame = frame.assign(_r=residual)
            residual_frame = residual_frame[np.isfinite(residual_frame._r)]
            if not residual_frame.empty:
                per_patient_r = residual_frame.groupby("subject")._r.median().to_dict()
                adjusted = paired_effect(per_patient_r, {s: 0.0 for s in per_patient_r},
                                         label=f"{reading}::{endpoint}::residualised",
                                         lower_is_better=False)
            block[endpoint] = {
                "raw": raw.as_dict(),
                "residualised_on_nuisances": adjusted.as_dict() if adjusted else None,
                "n_degenerate": int(degenerate.sum()), "n_usable": int(len(usable)),
                "dataset_strata": stratify(raw, dataset),
                "leave_seizure_out": leave_one_out(usable, column),
            }
            if family is not None and reading == "open_loop_at_onset":
                family[f"{reading}::{endpoint}"] = raw.sign_test_p
        blocks[reading] = block
    return blocks


def nuisance_rows(frame: pd.DataFrame) -> dict:
    out = {}
    for key in NUISANCE_Z:
        column = f"nuisanceonly__{key}_z"
        if column not in frame:
            continue
        usable = frame[np.isfinite(frame[column])]
        if usable.empty:
            continue
        per_patient = usable.groupby("subject")[column].median().to_dict()
        out[key] = paired_effect(per_patient, {s: 0.0 for s in per_patient},
                                 label=f"nuisance_only::{key}",
                                 lower_is_better=False).as_dict()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", required=True)
    parser.add_argument("--lead-minutes", type=float, default=30.0)
    args = parser.parse_args()
    tag = f"{args.layer}__lead{int(args.lead_minutes)}m"

    records, rows, denominators = [], [], []
    for path in sorted((OUT / "per_subject").glob(f"*__{tag}.json")):
        record = json.loads(path.read_text())
        records.append(record)
        denominators.append({
            "subject": record["subject"], "dataset": record.get("dataset"),
            "status": record["status"], "reason": record.get("reason"),
            "n_seizures_total": record.get("n_seizures_total"),
            "n_seizures_in_span": record.get("n_seizures_in_span"),
            "n_seizures_eligible": record.get("n_seizures_eligible", 0),
            "n_seizures_premise_met": record.get("n_seizures_premise_met", 0),
            "n_events_admissible": record["stream"]["n_events_admissible"],
            "n_events_beyond_definite_interictal":
                record["stream"]["n_events_beyond_definite_interictal"],
        })
        rows.extend(record.get("per_seizure", []))
    if not records:
        raise SystemExit(f"no Goal 3b per-subject results for {tag}")
    frame = pd.DataFrame(rows)
    denominator_frame = pd.DataFrame(denominators)
    atomic_write_csv(OUT / f"preictal_effects__{tag}.csv", frame)
    atomic_write_csv(OUT / f"preictal_denominators__{tag}.csv", denominator_frame)

    card = {
        "contract": "topic5_epi_prssm_v0_1_h2b_primary_evidence_card",
        "role": "primary H2b: the observer consumes the pre-ictal IEDs and is closed at a "
                "declared lead time",
        "layer": args.layer, "lead_minutes": args.lead_minutes,
        "status": "EXPLORATORY_DEVELOPMENT",
        "premise_rule": records[0].get("premise_rule"),
        "denominators": {
            "n_patients_attempted": int(len(denominator_frame)),
            "n_patients_ok": int((denominator_frame.status == "ok").sum()),
            "n_patients_not_observable": int(
                (denominator_frame.status == "NOT_OBSERVABLE_FROM_CURRENT_STREAM").sum()),
            "not_observable_patients": denominator_frame.loc[
                denominator_frame.status != "ok", "subject"].tolist(),
            "n_seizures_eligible": int(len(frame)),
            "n_patients_with_eligible_seizures":
                int(frame.subject.nunique()) if len(frame) else 0,
            "n_events_admissible_total": int(denominator_frame.n_events_admissible.sum()),
            "n_events_recovered_beyond_definite_interictal": int(
                denominator_frame.n_events_beyond_definite_interictal.sum()),
        },
        "reading_definitions": {
            "filtered_at_onset": "the observer consumed every admissible event up to onset",
            "filtered_at_cutoff": "the observer stopped at onset minus the lead",
            "open_loop_at_onset": "the observer stopped at the cut-off and the generator then "
                                  "integrated autonomously to onset",
        },
        "claim_boundary": [
            "a state claim requires the endpoint to survive residualisation on the multi-scale "
            "rate and interval nuisances; Topic 2 already shows the rate drifts slowly and rises "
            "around seizures",
            "the onset time is used for alignment and scoring only and never enters the model",
            "a patient with no admissible pre-ictal anchor is NOT_OBSERVABLE_FROM_CURRENT_STREAM, "
            "not a negative",
            "the primary analysis set is the one where the pre-ictal observation premise holds; "
            "a seizure with an empty look-back window is back in the strict arm's situation and "
            "is reported in its own stratum",
            "development-partition result; no untouched-test claim is made here",
        ],
        "code_revision": code_revision(), "package_hash": package_hash(),
    }
    if frame.empty:
        card["status"] = "NOT_OBSERVABLE_FROM_CURRENT_STREAM"
        atomic_write_json(OUT / f"H2B_PRIMARY_EVIDENCE_CARD__{tag}.json", card)
        print(json.dumps(card["denominators"], indent=2))
        return

    dataset = dict(zip(frame.subject, frame.dataset))
    card["lookback_strata"] = (frame.lookback_stratum.value_counts().to_dict()
                               if "lookback_stratum" in frame else {})
    # how often the tempo caliper actually bound, so a reader can split the effect by
    # whether the balance was enforced for that seizure rather than assume it was
    if "median_iei_caliper_applied" in frame:
        applied = frame["median_iei_caliper_applied"].fillna(False).astype(bool)
        card["median_iei_caliper"] = {
            "n_seizures_caliper_applied": int(applied.sum()),
            "n_seizures_on_soft_fallback": int((~applied).sum()),
            "median_n_pseudo_after_caliper": (
                float(frame.loc[applied, "n_pseudo_after_caliper"].median())
                if applied.any() and "n_pseudo_after_caliper" in frame else None),
        }
    card["pseudo_exclusion_relaxed_seizures"] = int(
        frame.get("pseudo_exclusion_relaxed", pd.Series(False, index=frame.index)).sum())
    premise_mask = (frame["preictal_observation_premise_met"].astype(bool)
                    if "preictal_observation_premise_met" in frame
                    else pd.Series(True, index=frame.index))
    premise = frame[premise_mask]
    card["denominators"]["n_seizures_premise_met"] = int(len(premise))
    card["denominators"]["n_patients_premise_met"] = (
        int(premise.subject.nunique()) if len(premise) else 0)

    family: dict[str, float] = {}
    card["analysis_sets"] = {}
    for name, working, fam in (("primary_premise_met", premise, family),
                               ("secondary_all_eligible", frame, None)):
        if working.empty:
            card["analysis_sets"][name] = {"status": "empty",
                                           "reason": "no seizure meets this set's condition"}
            continue
        card["analysis_sets"][name] = {
            "n_seizures": int(len(working)), "n_patients": int(working.subject.nunique()),
            "readings": analyse(working, dataset, fam),
            "nuisance_only": nuisance_rows(working),
        }
    primary = card["analysis_sets"].get("primary_premise_met", {})
    fallback = card["analysis_sets"].get("secondary_all_eligible", {})
    source = primary if "readings" in primary else fallback
    card["readings"] = source.get("readings", {})
    card["nuisance_only"] = source.get("nuisance_only", {})
    card["headline_analysis_set"] = ("primary_premise_met" if "readings" in primary
                                     else "secondary_all_eligible")
    card["holm_corrected_open_loop_family"] = holm(family)
    atomic_write_json(OUT / f"H2B_PRIMARY_EVIDENCE_CARD__{tag}.json", card)
    print(json.dumps({"denominators": card["denominators"],
                      "lookback_strata": card.get("lookback_strata"),
                      "headline_set": card["headline_analysis_set"],
                      "open_loop_state_norm": (card["readings"].get("open_loop_at_onset", {})
                                               .get("state_norm", {}) or {}).get("raw", {})},
                     indent=2)[:1500])


if __name__ == "__main__":
    main()
