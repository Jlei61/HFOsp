#!/usr/bin/env python3
"""Post-hoc signed versus sign-free early-ictal field sensitivity.

This diagnostic reuses the frozen model fields, target vectors, and synchronized
permutation maps from v0.5.  It does not alter the registered signed endpoint.
The sign-free score is max(abs(r_pattern1), abs(r_pattern2)); every null draw
repeats the same absolute-value and best-pattern selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.score_topic5_multiscale_early_ictal_v0_5 import (  # noqa: E402
    load_broadband,
    load_candidates,
    signed_spearman_permutations,
)


DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
CONDITION = "INTACT|L3_LOCAL_PLUS_LEARNED_LR"
ENDPOINT = "canonical_full"


def summarize(values: np.ndarray) -> dict:
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    nonzero = finite[np.abs(finite) > 1e-9]
    p = 1.0 if not len(nonzero) else float(wilcoxon(nonzero, alternative="greater").pvalue)
    return {
        "n": int(len(finite)),
        "median": float(np.median(finite)) if len(finite) else float("nan"),
        "positive": int(np.sum(finite > 1e-9)),
        "negative": int(np.sum(finite < -1e-9)),
        "ties": int(np.sum(np.abs(finite) <= 1e-9)),
        "wilcoxon_p_greater": p,
    }


def score_two_patterns(candidate: dict, target: np.ndarray, permutations: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    r1, null1 = signed_spearman_permutations(candidate["a"], target, permutations)
    r2, null2 = signed_spearman_permutations(candidate["b"], target, permutations)
    signed = float(np.nanmax([r1, r2]))
    sign_free = float(np.nanmax(np.abs([r1, r2])))
    signed_null = np.fmax(null1, null2)
    sign_free_null = np.fmax(np.abs(null1), np.abs(null2))
    return signed, sign_free, signed_null, sign_free_null


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_root.resolve()
    routing = pd.read_csv(out / "EARLY_ICTAL_ROUTING_METADATA.csv")
    rows: list[dict] = []
    folded: dict[tuple[str, str, str], list[np.ndarray]] = {}

    for event in routing.itertuples(index=False):
        field_path = out / "model_fields/intact/per_patient" / str(event.subject) / "L3_LOCAL_PLUS_LEARNED_LR.npz"
        with np.load(field_path, allow_pickle=False) as data:
            contacts = data["contacts"].astype(str).tolist()
        target = load_broadband(str(event.subject), int(event.seizure_idx), contacts)
        candidate = load_candidates(out, str(event.subject), ENDPOINT, contacts)[CONDITION]
        with np.load(
            out / "null_maps" / f"{event.subject}__seizure{int(event.seizure_idx)}.npz",
            allow_pickle=False,
        ) as null_map:
            families = {
                "all_contacts": null_map["all_contact"].copy(),
                "within_electrode": null_map["within_shaft"].copy(),
                "matched_distance": null_map["distance_bin"].copy(),
            }

        for family, permutations in families.items():
            if not len(permutations):
                continue
            signed, sign_free, signed_null, sign_free_null = score_two_patterns(
                candidate, target, permutations
            )
            rows.append({
                "subject": str(event.subject),
                "seizure_idx": int(event.seizure_idx),
                "control": family,
                "signed_score": signed,
                "sign_free_score": sign_free,
                "signed_null_median": float(np.nanmedian(signed_null)),
                "sign_free_null_median": float(np.nanmedian(sign_free_null)),
            })
            folded.setdefault((str(event.subject), family, "signed"), []).append(signed_null)
            folded.setdefault((str(event.subject), family, "sign_free"), []).append(sign_free_null)

    per_event = pd.DataFrame(rows)
    patient_rows: list[dict] = []
    for (subject, family), group in per_event.groupby(["subject", "control"], sort=True):
        for orientation, score_column in (("signed", "signed_score"), ("sign_free", "sign_free_score")):
            score = float(np.nanmedian(group[score_column]))
            null = np.nanmedian(np.stack(folded[(subject, family, orientation)]), axis=0)
            patient_rows.append({
                "subject": subject,
                "control": family,
                "orientation": orientation,
                "n_seizures": int(len(group)),
                "observed": score,
                "null_median": float(np.nanmedian(null)),
                "margin": score - float(np.nanmedian(null)),
            })
    patient = pd.DataFrame(patient_rows)

    summary = {
        "contract": "topic5_multiscale_early_ictal_posthoc_sign_sensitivity_v0_5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "POSTHOC_DIAGNOSTIC_DOES_NOT_REPLACE_REGISTERED_SIGNED_ENDPOINT",
        "target": "clinical onset 0-10 s, 1-150 Hz broadband energy",
        "signed_definition": "max(r_pattern1, r_pattern2)",
        "sign_free_definition": "max(abs(r_pattern1), abs(r_pattern2))",
        "null_rule": "repeat sign handling and best-pattern selection inside every frozen permutation draw",
        "project_history_target_previously_viewed": True,
        "results": {},
    }
    for family in ("all_contacts", "within_electrode", "matched_distance"):
        summary["results"][family] = {}
        for orientation in ("signed", "sign_free"):
            values = patient[
                patient.control.eq(family) & patient.orientation.eq(orientation)
            ].margin.to_numpy(float)
            summary["results"][family][orientation] = summarize(values)

    early = out / "early_ictal"
    per_event.to_csv(early / "POSTHOC_SIGN_SENSITIVITY_PER_SEIZURE.csv", index=False)
    patient.to_csv(early / "POSTHOC_SIGN_SENSITIVITY_PER_PATIENT.csv", index=False)
    (early / "POSTHOC_SIGN_SENSITIVITY_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
