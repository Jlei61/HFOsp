#!/usr/bin/env python3
"""Task D: H3B as a transition-coupling analysis, not an AND gate.

The old H3B fired only when H2B and H3A were both supported and agreed in direction.
That made it a conjunction of two other verdicts rather than a question of its own,
and it meant a negative anywhere upstream silently deleted it.

Here it asks its own question: within a patient, do the seizures preceded by more
discharge exposure show a different pre-ictal state displacement, and does that
relationship depend on which subtype the seizure belongs to?

  * **Patient-internal case-crossover.**  Each seizure is compared against that
    patient's own matched pseudo-onsets, so between-patient differences in exposure,
    implantation and rate cannot produce the effect.
  * **Interactions, not a battery.**  exposure x subtype and exposure x state are two
    declared interactions; the per-subtype effects are descriptive output, never a
    set of tests to pick from.
  * **Mediation stays exploratory.**  With exposure, state and outcome measured on the
    same seizures and no intervention, a mediation coefficient is a description of
    covariance, not evidence of a pathway.

One leg is not runnable and is reported as such rather than substituted: the
"subtype-specific early recruitment" endpoint needs blind adjudicated onset contacts,
and the registry has 0 of 71 seizures adjudicated.  A locked blinding contract forbids
standing in the clinically-declared focus, the patient-level focus, template endpoints
or the highest-energy contact.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from _common import (  # noqa: E402
    FROZEN, OUTPUT_ROOT, atomic_write_csv, atomic_write_json, code_revision, package_hash,
)

H2B = OUTPUT_ROOT / "seizure_link_preictal"
CROSSWALK = OUTPUT_ROOT / "seizure_crosswalk"
OUT = OUTPUT_ROOT / "h3b_transition"

STATE_ENDPOINT = "open_loop_at_onset__first_selection_entropy_z"
EXPOSURE = "n_events_in_lookback_2h"
MIN_SEIZURES_PER_PATIENT = 5
MIN_SUBTYPE_SIZE = 3
N_SHUFFLES = 2000


def within_patient_slope(block: pd.DataFrame, x: str, y: str) -> float | None:
    rows = block[np.isfinite(block[x]) & np.isfinite(block[y])]
    if len(rows) < MIN_SEIZURES_PER_PATIENT or rows[x].nunique() < 3:
        return None
    a = rows[x].rank().to_numpy()
    b = rows[y].to_numpy()
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.polyfit(a / a.max(), b, 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", default="linear_graph_recurrent")
    parser.add_argument("--lead", default="lead30m")
    parser.add_argument("--band", default="broad_ER")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(FROZEN["bootstrap_seed"])

    frame = pd.read_csv(H2B / f"preictal_effects__{args.layer}__{args.lead}.csv")
    cw = pd.read_csv(CROSSWALK / f"crosswalk__{args.layer}__{args.lead}.csv")
    label = f"{args.band}__subtype_label"
    frame = frame.merge(cw[["subject", "seizure_id", "canonical_seizure_id",
                            label, f"{args.band}__outlier"]],
                        on=["subject", "seizure_id"], how="left")

    # --- leg 1: does exposure track the pre-ictal state displacement, within patient?
    slopes = []
    for subject, block in frame.groupby("subject"):
        slope = within_patient_slope(block, EXPOSURE, STATE_ENDPOINT)
        if slope is None:
            continue
        draws = []
        rows = block[np.isfinite(block[EXPOSURE]) & np.isfinite(block[STATE_ENDPOINT])]
        for _ in range(N_SHUFFLES):
            shuffled = rows.assign(**{STATE_ENDPOINT: rng.permutation(
                rows[STATE_ENDPOINT].to_numpy())})
            drawn = within_patient_slope(shuffled, EXPOSURE, STATE_ENDPOINT)
            if drawn is not None:
                draws.append(drawn)
        slopes.append({"subject": subject, "slope": slope, "n_seizures": int(len(rows)),
                       "null_median": float(np.median(draws)) if draws else np.nan,
                       "p_two_sided": float(np.mean(np.abs(draws) >= abs(slope)))
                                      if draws else np.nan})
    slope_frame = pd.DataFrame(slopes)

    from scipy.stats import binomtest
    exposure_state = {"status": "NO_USABLE_PATIENT"}
    if not slope_frame.empty:
        values = slope_frame.slope.to_numpy()
        exposure_state = {
            "status": "OK",
            "n_patients": len(values),
            "n_seizures": int(slope_frame.n_seizures.sum()),
            "median_within_patient_slope": float(np.median(values)),
            "n_positive": int((values > 0).sum()),
            "sign_test_p": float(binomtest(int((values > 0).sum()), len(values), 0.5).pvalue),
            "median_shuffle_p": float(np.nanmedian(slope_frame.p_two_sided)),
            "reading": "slope of the pre-ictal state displacement on within-patient "
                       "ranked exposure; each patient also carries its own "
                       "outcome-shuffle null",
        }

    # --- leg 2: does that relationship depend on subtype?  one interaction, not a battery
    interaction = {"status": "NO_PATIENT_WITH_TWO_USABLE_SUBTYPES"}
    per_patient = []
    for subject, block in frame.groupby("subject"):
        sub = block[block[label].notna() & (block[label] >= 0)]
        sub = sub[~sub[f"{args.band}__outlier"].fillna(False).astype(bool)]
        sizes = sub[label].value_counts()
        big = sizes[sizes >= MIN_SUBTYPE_SIZE].index.tolist()
        if len(big) < 2:
            continue
        by_subtype = {}
        for value in big:
            slope = within_patient_slope(sub[sub[label] == value], EXPOSURE, STATE_ENDPOINT)
            if slope is not None:
                by_subtype[int(value)] = slope
        if len(by_subtype) >= 2:
            spread = max(by_subtype.values()) - min(by_subtype.values())
            per_patient.append({"subject": subject, "spread": spread,
                                "by_subtype": by_subtype,
                                "n_seizures": int(len(sub))})
    if per_patient:
        spreads = np.array([r["spread"] for r in per_patient])
        interaction = {
            "status": "OK", "band": args.band,
            "n_patients": len(spreads),
            "median_spread_between_subtypes": float(np.median(spreads)),
            "per_patient": per_patient,
            "caution": "with this few patients the interaction is descriptive; a null "
                       "here means it could not be seen, not that it is absent",
        }

    card = {
        "contract": "topic5_epi_prssm_h3b_transition_coupling",
        "definition": "exposure history -> pre-ictal state, and whether that coupling "
                      "is subtype specific; explicitly NOT an AND gate over H2B and H3A",
        "layer": args.layer, "lead": args.lead, "band": args.band,
        "state_endpoint": STATE_ENDPOINT,
        "exposure_measure": EXPOSURE,
        "design": "patient-internal case-crossover against that patient's own matched "
                  "pseudo-onsets; patient is the unit; seizures nested within patient",
        "exposure_to_state": exposure_state,
        "exposure_x_subtype_interaction": interaction,
        "early_recruitment_leg": {
            "status": "NOT_RUNNABLE",
            "reason": "needs blind adjudicated onset contacts; 0 of 71 seizures are "
                      "adjudicated and a locked blinding contract forbids substituting "
                      "the clinical focus, the patient-level focus, template endpoints "
                      "or the highest-energy contact",
        },
        "mediation": {"status": "NOT_RUN",
                      "reason": "exposure, state and outcome are measured on the same "
                                "seizures with no intervention, so a mediation "
                                "coefficient would describe covariance only"},
        "independence": "a null here does not withdraw H1, H2a, H2b or H3a; those are "
                        "reported on their own evidence",
        "code_revision": code_revision(), "package_hash": package_hash(),
    }
    atomic_write_json(OUT / f"H3B_TRANSITION_CARD__{args.band}.json", card)
    atomic_write_csv(OUT / f"exposure_state_slopes__{args.band}.csv", slope_frame)
    print(json.dumps({"exposure_to_state": exposure_state,
                      "interaction_status": interaction.get("status"),
                      "interaction_n": interaction.get("n_patients")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
