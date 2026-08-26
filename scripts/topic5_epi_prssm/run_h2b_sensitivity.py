#!/usr/bin/env python3
"""Task B: H2B as a population layer plus an observability sensitivity layer, with
the subtype question asked as one interaction rather than a stack of p-values.

Design commitments, each of which exists to block a specific way of being wrong:

* **361 is the population layer, 203 is a sensitivity layer.**  The 203 are the
  seizures whose pre-ictal window was actually observed; treating them as the study
  size converts an observability filter into a cohort.
* **Observability also enters continuously** -- IED count in the look-back, anchor
  gap and coverage -- because a binary premise throws away the gradient.
* **Subtype labels are never pooled across patients.**  A "subtype 0" in one patient
  has nothing to do with "subtype 0" in another; they are separate clusterings.  The
  question is asked within patient first, and the cohort statement is about whether
  the within-patient difference exceeds a label-shuffle null.
* **One interaction, not many tests.**  Running the pre-ictal effect separately per
  subtype and reading off whichever is significant is a min-of-N artefact.
* **Patient is the unit.**  Seizures are nested inside patients and never counted as
  independent observations.
* The primary endpoint is a frozen decoder readout; the raw state magnitude is a
  sensitivity endpoint only.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from _common import (  # noqa: E402
    FROZEN, OUTPUT_ROOT, atomic_write_csv, atomic_write_json, code_revision, package_hash,
)
from src.topic5_epi_prssm.stats import paired_effect  # noqa: E402

H2B = OUTPUT_ROOT / "seizure_link_preictal"
CROSSWALK = OUTPUT_ROOT / "seizure_crosswalk"
OUT = OUTPUT_ROOT / "h2b_sensitivity"

#: frozen decoder readouts.  ``state_norm`` is the raw state magnitude and is a
#: sensitivity endpoint only -- it is not a readout of anything the model emits.
PRIMARY_ENDPOINT = "first_selection_entropy"
SECONDARY_ENDPOINTS = ("expected_load", "resource")
SENSITIVITY_ENDPOINT = "state_norm"
READINGS = ("open_loop_at_onset", "filtered_at_cutoff", "filtered_at_onset")

#: Topic 5 requires each consumer to declare how it treats tiny subtypes.
#: Rule here: a subtype with fewer than this many seizures cannot support a
#: within-patient contrast and is excluded from the interaction, but is counted and
#: reported rather than silently dropped.
MIN_SUBTYPE_SIZE = 3
N_LABEL_SHUFFLES = 2000


def load(layer: str, lead: str) -> pd.DataFrame:
    frame = pd.read_csv(H2B / f"preictal_effects__{layer}__{lead}.csv")
    cw = CROSSWALK / f"crosswalk__{layer}__{lead}.csv"
    if cw.exists():
        keys = pd.read_csv(cw)[["subject", "seizure_id", "canonical_seizure_id",
                                "timestamp_verified", "broad_ER__subtype_label",
                                "broad_ER__outlier", "gamma_ER__subtype_label",
                                "gamma_ER__outlier"]]
        frame = frame.merge(keys, on=["subject", "seizure_id"], how="left")
    frame["lead"] = lead
    frame["layer"] = layer
    return frame


def matched_set_rank(frame: pd.DataFrame, column: str) -> pd.Series:
    """NOT IMPLEMENTABLE FROM THE CURRENT PER-SEIZURE TABLE.

    The statistic that is wanted is where the real onset sits inside *its own* matched
    set of pseudo-onsets: bounded, needing no variance estimate, and therefore immune
    to the degeneracy that makes the stored z blow up when a pseudo set is nearly
    constant.  Residualising those z values produced a cohort effect of -121.8 SD with
    27 of 27 patients aligned, which is arithmetic, not biology.

    A first attempt here ranked the z values *across seizures* instead.  That is a
    monotone re-expression of the same numbers pooled over patients, so a per-patient
    median tested against zero asks only whether a patient sits above the cohort
    median -- roughly half do, by construction.  It looked like a clean null and was
    an artefact of the statistic.

    Computing it properly needs the per-seizure pseudo-onset distribution, which the
    producer summarises away.  It is therefore blocked on the same producer re-run that
    the median-interval caliper requires.
    """
    raise NotImplementedError(
        "needs per-seizure pseudo-onset values; re-run run_goal3b_preictal.py with the "
        "median-interval caliper and persist the matched-set distribution first")


def cohort_effect(frame: pd.DataFrame, column: str, label: str) -> dict | None:
    """Patient is the unit: seizures are collapsed within patient first."""
    usable = frame[np.isfinite(frame[column])]
    if usable.empty:
        return None
    per_patient = usable.groupby("subject")[column].median().to_dict()
    if len(per_patient) < 5:
        return None
    effect = paired_effect(per_patient, {s: 0.0 for s in per_patient},
                           label=label, lower_is_better=False).as_dict()
    effect["n_seizures_behind_it"] = int(len(usable))
    return effect


def continuous_observability(frame: pd.DataFrame, column: str) -> dict:
    """Observability as a gradient, not only as the binary premise."""
    out = {}
    for name, covariate in (("n_ied_lookback", "n_events_in_lookback_2h"),
                            ("anchor_gap_seconds", "anchor_gap_to_cutoff_seconds"),
                            ("coverage", "nuisance_coverage")):
        if covariate not in frame:
            continue
        rows = frame[np.isfinite(frame[column]) & np.isfinite(frame[covariate])]
        if len(rows) < 20:
            continue
        # within-patient Spearman, then patient is the unit
        per_patient = []
        for subject, block in rows.groupby("subject"):
            if len(block) < 4 or block[covariate].nunique() < 3:
                continue
            a = block[column].rank().to_numpy()
            b = block[covariate].rank().to_numpy()
            if a.std() == 0 or b.std() == 0:
                continue
            per_patient.append(float(np.corrcoef(a, b)[0, 1]))
        if len(per_patient) < 5:
            continue
        values = np.array(per_patient)
        out[name] = {
            "n_patients": len(values),
            "median_within_patient_spearman": float(np.median(values)),
            "n_positive": int((values > 0).sum()),
            "note": "within-patient rank correlation between the endpoint and the "
                    "observability covariate, then patient as the unit",
        }
    return out


def subtype_interaction(frame: pd.DataFrame, column: str, band: str,
                        rng: np.random.Generator) -> dict:
    """Within-patient subtype heterogeneity, with a within-patient label shuffle null.

    Subtype identity is patient-local, so the cohort statement is not "subtype 0
    differs from subtype 1" but "the spread between a patient's own subtypes is
    larger than relabelling that patient's seizures at random would give".
    """
    label_column = f"{band}__subtype_label"
    outlier_column = f"{band}__outlier"
    if label_column not in frame:
        return {"status": "NO_SUBTYPE_LABELS"}
    rows = frame[np.isfinite(frame[column]) & frame[label_column].notna()].copy()
    rows = rows[~rows[outlier_column].fillna(False).astype(bool)]
    rows = rows[rows[label_column] >= 0]
    if rows.empty:
        return {"status": "NO_LABELLED_SEIZURES"}

    observed, null_spreads, usable, excluded = [], [], [], []
    for subject, block in rows.groupby("subject"):
        sizes = block[label_column].value_counts()
        big = sizes[sizes >= MIN_SUBTYPE_SIZE].index.tolist()
        small = sizes[sizes < MIN_SUBTYPE_SIZE]
        if len(small):
            excluded.append({"subject": subject,
                             "subtypes_below_min_size": {int(k): int(v)
                                                         for k, v in small.items()}})
        if len(big) < 2:
            continue
        sub = block[block[label_column].isin(big)]
        means = sub.groupby(label_column)[column].mean()
        spread = float(means.max() - means.min())
        observed.append({"subject": subject, "spread": spread,
                         "n_subtypes_used": len(big),
                         "n_seizures": int(len(sub))})
        values = sub[column].to_numpy()
        labels = sub[label_column].to_numpy()
        draws = []
        for _ in range(N_LABEL_SHUFFLES):
            permuted = rng.permutation(labels)
            m = pd.Series(values).groupby(permuted).mean()
            draws.append(float(m.max() - m.min()))
        null_spreads.append({"subject": subject,
                             "null_median": float(np.median(draws)),
                             "null_p95": float(np.percentile(draws, 95)),
                             "p_value": float((np.array(draws) >= spread).mean())})
        usable.append(subject)

    if not observed:
        return {"status": "NO_PATIENT_WITH_TWO_SUBTYPES_OF_MIN_SIZE",
                "min_subtype_size": MIN_SUBTYPE_SIZE,
                "excluded_small_subtypes": excluded}

    obs = pd.DataFrame(observed).set_index("subject")
    nul = pd.DataFrame(null_spreads).set_index("subject")
    joined = obs.join(nul)
    excess = (joined.spread - joined.null_median).to_numpy()
    from scipy.stats import binomtest, combine_pvalues
    return {
        "status": "OK",
        "band": band,
        "min_subtype_size": MIN_SUBTYPE_SIZE,
        "small_subtype_rule": "excluded from the interaction, counted and reported",
        "n_patients_with_two_usable_subtypes": len(joined),
        "n_seizures_behind_it": int(joined.n_seizures.sum()),
        "median_observed_spread": float(joined.spread.median()),
        "median_null_spread": float(joined.null_median.median()),
        "median_excess_over_null": float(np.median(excess)),
        "n_patients_above_their_own_null": int((excess > 0).sum()),
        "sign_test_p": float(binomtest(int((excess > 0).sum()), len(excess), 0.5).pvalue),
        "combined_p_fisher": float(combine_pvalues(joined.p_value.clip(1e-6, 1))[1]),
        "per_patient": joined.reset_index().to_dict("records"),
        "excluded_small_subtypes": excluded,
        "interpretation_rule":
            "this is one interaction statistic over patients, not a per-subtype test "
            "battery; subtype identity is patient-local and is never pooled",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+",
                        default=["linear_graph_recurrent", "leaky_state",
                                 "resource_anchored_on_best_family"])
    parser.add_argument("--leads", nargs="+",
                        default=["lead5m", "lead15m", "lead30m", "lead60m"])
    parser.add_argument("--primary-lead", default="lead30m")
    parser.add_argument("--primary-layer", default="linear_graph_recurrent")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(FROZEN["bootstrap_seed"])

    grid_rows, cards = [], {}
    for layer in args.layers:
        for lead in args.leads:
            path = H2B / f"preictal_effects__{layer}__{lead}.csv"
            if not path.exists():
                continue
            frame = load(layer, lead)
            premise = frame.preictal_observation_premise_met.astype(bool)
            for population, subset in (("all_eligible", frame),
                                       ("high_observability", frame[premise])):
                for reading in READINGS:
                    for endpoint, tier in ([(PRIMARY_ENDPOINT, "primary")]
                                           + [(e, "secondary") for e in SECONDARY_ENDPOINTS]
                                           + [(SENSITIVITY_ENDPOINT, "sensitivity")]):
                        column = f"{reading}__{endpoint}_z"
                        if column not in subset:
                            continue
                        effect = cohort_effect(subset, column,
                                               f"{layer}::{lead}::{population}::{reading}::{endpoint}")
                        if effect is None:
                            continue
                        rank_effect = None      # see matched_set_rank's docstring
                        grid_rows.append({
                            "layer": layer, "lead": lead, "population": population,
                            "reading": reading, "endpoint": endpoint, "tier": tier,
                            "n_patients": effect["n_patients"],
                            "n_seizures": effect["n_seizures_behind_it"],
                            "median_delta": effect["median_delta"],
                            "ci_low": effect["ci_low"], "ci_high": effect["ci_high"],
                            "n_favourable": effect["n_favourable"],
                            "sign_test_p": effect["sign_test_p"],
                            "n_usable_seizures": effect["n_seizures_behind_it"],
                            "rank_median_delta": (rank_effect or {}).get("median_delta"),
                            "rank_n_favourable": (rank_effect or {}).get("n_favourable"),
                            "rank_sign_test_p": (rank_effect or {}).get("sign_test_p"),
                        })

    grid = pd.DataFrame(grid_rows)
    atomic_write_csv(OUT / "h2b_sensitivity_grid.csv", grid)

    primary = load(args.primary_layer, args.primary_lead)
    column = f"open_loop_at_onset__{PRIMARY_ENDPOINT}_z"
    cards["continuous_observability"] = continuous_observability(primary, column)
    for band in ("broad_ER", "gamma_ER"):
        cards[f"subtype_interaction__{band}"] = subtype_interaction(
            primary, column, band, rng)

    # leave-one-out, both kinds, on the primary cell
    def leave_out(kind: str) -> dict:
        usable = primary[np.isfinite(primary[column])]
        per_patient = usable.groupby("subject")[column].median()
        out = []
        if kind == "patient":
            for subject in per_patient.index:
                rest = per_patient.drop(subject)
                out.append({"left_out": subject, "median": float(rest.median()),
                            "n_favourable": int((rest > 0).sum()), "n": len(rest)})
        else:
            for sid in usable.seizure_id.unique():
                rest = usable[usable.seizure_id != sid].groupby("subject")[column].median()
                out.append({"left_out": str(sid), "median": float(rest.median()),
                            "n_favourable": int((rest > 0).sum()), "n": len(rest)})
        medians = np.array([r["median"] for r in out])
        return {"n_refits": len(out), "median_of_medians": float(np.median(medians)),
                "min": float(medians.min()), "max": float(medians.max()),
                "sign_stable": bool((medians > 0).all() or (medians < 0).all())}

    cards["leave_one_patient_out"] = leave_out("patient")
    cards["leave_one_seizure_out"] = leave_out("seizure")

    card = {
        "contract": "topic5_epi_prssm_h2b_sensitivity",
        "primary_layer": args.primary_layer, "primary_lead": args.primary_lead,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "endpoint_tiers": {"primary": PRIMARY_ENDPOINT,
                           "secondary": list(SECONDARY_ENDPOINTS),
                           "sensitivity_only": SENSITIVITY_ENDPOINT},
        "population_layers": {
            "all_eligible": "every eligible seizure at that lead; this is the "
                            "population layer and carries the main statement",
            "high_observability": "the subset whose pre-ictal window was actually "
                                  "observed; a sensitivity layer, never the study size"},
        "band_rule": "broad_ER is primary and gamma_ER is sensitivity; the two are not "
                     "reported as independent findings",
        **cards,
        "code_revision": code_revision(), "package_hash": package_hash(),
    }
    atomic_write_json(OUT / "H2B_SENSITIVITY_CARD.json", card)
    pd.set_option("display.width", 220)
    show = grid[(grid.layer == args.primary_layer) & (grid.tier != "secondary")]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
