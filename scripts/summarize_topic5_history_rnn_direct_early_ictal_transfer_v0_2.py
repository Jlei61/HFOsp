#!/usr/bin/env python3
"""Patient-first summary for direct history-state to early-ictal transfer v0.2."""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon


DEVELOPMENT_TARGET = "epilepsiae_1146"
MODELS = (
    "M0", "M1", "E0p5", "E2", "E6", "EM", "R2",
    "R2_ORDER_SHUFFLE", "R2_ZERO_STATE", "E2_TIME_SHUFFLE", "EM_TIME_SHUFFLE",
)
N_CHANNEL_PERMUTATIONS = 5_000
# Patient contrasts are differences of Spearman coefficients on 6-16 contacts,
# so genuinely tied patients appear either as exact zeros or as float residue
# around 1e-17.  Anything inside this band is a tie, not a signed observation.
TIE_TOLERANCE = 1e-9


def _p(values: np.ndarray) -> float:
    value = np.asarray(values, dtype=float)
    value = value[np.isfinite(value)]
    # Drop ties before the test.  SciPy also drops them internally, but its
    # ``auto`` method then falls back to the normal approximation, which is
    # anticonservative at the small effective n this cohort produces (n=4 all
    # positive: 0.034 approximate vs 0.0625 exact).  Stripping them here keeps
    # the exact signed-rank null.
    value = value[np.abs(value) > TIE_TOLERANCE]
    if not len(value):
        return 1.0
    return float(wilcoxon(value, alternative="greater", method="auto").pvalue)


def _ci(values: np.ndarray, seed: int) -> list[float]:
    value = np.asarray(values, dtype=float)
    value = value[np.isfinite(value)]
    if not len(value):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    sample = value[rng.integers(0, len(value), size=(20_000, len(value)))]
    return [float(v) for v in np.quantile(np.median(sample, axis=1), [0.025, 0.975])]


def _contrast(values: pd.Series, seed: int) -> dict:
    value = values.to_numpy(float)
    finite = value[np.isfinite(value)]
    return {
        "median": float(np.nanmedian(value)),
        "bootstrap_median_ci95": _ci(value, seed),
        "n_positive": int(np.sum(finite > TIE_TOLERANCE)),
        "n_negative": int(np.sum(finite < -TIE_TOLERANCE)),
        "n_tied": int(np.sum(np.abs(finite) <= TIE_TOLERANCE)),
        "n_patients": int(len(finite)),
        "one_sided_wilcoxon_p": _p(value),
    }


def _positive(statistic: dict) -> bool:
    """A median inside the tie band is not a positive effect."""

    return bool(statistic["median"] > TIE_TOLERANCE)


def _bh(p_values: list[float]) -> list[float]:
    value = np.asarray(p_values, dtype=float)
    order = np.argsort(value)
    ranked = value[order] * len(value) / np.arange(1, len(value) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1.0)
    return [float(item) for item in out]


def _rho(left: np.ndarray, right: np.ndarray) -> float:
    if np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return float("nan")
    return float(spearmanr(left, right).statistic)


def _rank_unit(value: np.ndarray) -> np.ndarray | None:
    ranked = rankdata(np.asarray(value, dtype=float), method="average")
    ranked = ranked - ranked.mean()
    norm = float(np.linalg.norm(ranked))
    if not np.isfinite(norm) or norm == 0:
        return None
    return ranked / norm


def _stable_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _channel_null_rows(
    predictions: pd.DataFrame,
    *,
    subject: str,
    n_perm: int = N_CHANNEL_PERMUTATIONS,
) -> list[dict]:
    """All-contact target-label shuffle with subject-first seizure folding.

    The same contact permutations are reused for every model, so model
    comparisons cannot be driven by different Monte Carlo draws.  Each draw
    independently permutes the target labels within each seizure, computes the
    seizure Spearman correlation, then takes the patient median across seizures.
    """

    rng = np.random.default_rng(_stable_seed(f"channel-null:{subject}"))
    seizure_payload: dict[str, dict] = {}
    reference = predictions.loc[predictions.model == "M0"].copy()
    for seizure, group in reference.groupby("seizure_id", sort=True):
        ordered = group.sort_values("contact_index")
        target = _rank_unit(ordered.target_z.to_numpy(float))
        if target is None:
            continue
        n_contacts = len(ordered)
        permutations = np.row_stack(
            [rng.permutation(n_contacts) for _ in range(n_perm)]
        )
        seizure_payload[str(seizure)] = {
            "target": target,
            "permutations": permutations,
            "contact_index": ordered.contact_index.to_numpy(int),
        }

    rows: list[dict] = []
    for model in MODELS:
        observed: list[float] = []
        null_by_seizure: list[np.ndarray] = []
        model_frame = predictions.loc[predictions.model == model]
        for seizure, payload in seizure_payload.items():
            ordered = model_frame.loc[
                model_frame.seizure_id.astype(str) == seizure
            ].sort_values("contact_index")
            if not np.array_equal(
                ordered.contact_index.to_numpy(int), payload["contact_index"]
            ):
                raise RuntimeError(
                    f"{subject}/{seizure}/{model}: contact denominator drift"
                )
            prediction = _rank_unit(ordered.prediction.to_numpy(float))
            if prediction is None:
                continue
            target = payload["target"]
            observed.append(float(np.dot(prediction, target)))
            null_by_seizure.append(
                target[payload["permutations"]] @ prediction
            )
        if not observed:
            continue
        patient_null = np.median(np.row_stack(null_by_seizure), axis=0)
        patient_observed = float(np.median(observed))
        null_median = float(np.median(patient_null))
        rows.append({
            "subject": subject,
            "model": model,
            "n_seizures": int(len(observed)),
            "n_channel_shuffle_draws": int(n_perm),
            "observed_patient_median_rho": patient_observed,
            "channel_null_median": null_median,
            "channel_null_p95": float(np.quantile(patient_null, 0.95)),
            "margin_vs_channel_null_median": patient_observed - null_median,
            "patient_permutation_p": float(
                (1 + np.sum(patient_null >= patient_observed)) / (n_perm + 1)
            ),
        })
    return rows


def _target_headroom(predictions: pd.DataFrame) -> dict:
    unique = predictions.loc[predictions.model == "M0", [
        "seizure_id", "contact_index", "target_z"
    ]].copy()
    seizures = sorted(unique.seizure_id.astype(str).unique())
    if len(seizures) < 2:
        return {
            "n_seizures": len(seizures),
            "pairwise_target_rho": float("nan"),
            "leave_one_seizure_out_mean_oracle_rho": float("nan"),
        }
    fields = {
        seizure: unique.loc[
            unique.seizure_id.astype(str) == seizure
        ].sort_values("contact_index").target_z.to_numpy(float)
        for seizure in seizures
    }
    pairwise = [_rho(fields[a], fields[b]) for a, b in combinations(seizures, 2)]
    oracle = []
    for seizure in seizures:
        others = [fields[value] for value in seizures if value != seizure]
        oracle.append(_rho(np.mean(others, axis=0), fields[seizure]))
    return {
        "n_seizures": len(seizures),
        "pairwise_target_rho": float(np.nanmean(pairwise)),
        "leave_one_seizure_out_mean_oracle_rho": float(np.nanmean(oracle)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--g0-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    done_paths = sorted(root.glob("*/DONE.json"))
    if len(done_paths) != 16:
        raise RuntimeError(f"direct transfer incomplete: {len(done_paths)}/16")

    patient_rows = []
    pairing_rows = []
    residual_rows = []
    headroom_rows = []
    channel_null_rows = []
    checkpoint_cycles = set()
    for done_path in done_paths:
        done = json.loads(done_path.read_text())
        if not bool(done.get("target_values_read", False)):
            raise RuntimeError(f"fold lacks explicit target access: {done_path}")
        if not str(done.get("target_unlock", "")).startswith(("DIRECT_V0_2:", "G1_")):
            raise RuntimeError(f"fold lacks valid direct-transfer authorization: {done_path}")
        subject = str(done["heldout_subject"])
        provenance = done.get("history_checkpoint_provenance") or {}
        if bool(provenance.get("target_values_read", True)):
            raise RuntimeError(f"{subject}: source HistoryRNN was not target-blind")
        checkpoint_cycles.add(int(provenance["history_cycles"]))
        metrics = pd.read_csv(done_path.parent / "heldout_seizure_metrics.csv")
        pivot = metrics.pivot(index="seizure_id", columns="model", values="spearman_rho")
        missing = set(MODELS) - set(pivot.columns)
        if missing:
            raise RuntimeError(f"{subject}: missing models {sorted(missing)}")
        row = {
            "subject": subject,
            "n_seizures": int(len(pivot)),
            # Spatial denominator of every Spearman below: the exact-name
            # intersection of the frozen scaffold and the rank dataset, not the
            # patient's full SEEG montage.
            "n_contacts": int(done["n_test_contacts"]),
        }
        for model in MODELS:
            row[f"rho_{model}"] = float(pivot[model].median())
        row.update({
            "rho_increment_R2_minus_M1": float((pivot.R2 - pivot.M1).median()),
            "rho_increment_E2_minus_M1": float((pivot.E2 - pivot.M1).median()),
            "rho_increment_E0p5_minus_M1": float((pivot.E0p5 - pivot.M1).median()),
            "rho_increment_E6_minus_M1": float((pivot.E6 - pivot.M1).median()),
            "rho_increment_EM_minus_M1": float((pivot.EM - pivot.M1).median()),
            "rho_true_R2_minus_order_shuffle": float(
                (pivot.R2 - pivot.R2_ORDER_SHUFFLE).median()
            ),
            "rho_true_R2_minus_zero_state": float(
                (pivot.R2 - pivot.R2_ZERO_STATE).median()
            ),
            "rho_increment_M1_minus_M0": float((pivot.M1 - pivot.M0).median()),
            "rho_true_E2_minus_time_shuffle": float(
                (pivot.E2 - pivot.E2_TIME_SHUFFLE).median()
            ),
            "rho_true_EM_minus_time_shuffle": float(
                (pivot.EM - pivot.EM_TIME_SHUFFLE).median()
            ),
        })
        patient_rows.append(row)

        predictions = pd.read_csv(done_path.parent / "heldout_contact_predictions.csv")
        channel_null_rows.extend(
            _channel_null_rows(predictions, subject=subject)
        )
        headroom_rows.append({"subject": subject, **_target_headroom(predictions)})

        wrong = pd.read_csv(done_path.parent / "heldout_wrong_state_pairing.csv")
        if not wrong.empty:
            for model_name, model_wrong in wrong.groupby("model", sort=True):
                correct = pivot[model_name].rename("correct_rho").reset_index()
                wrong_median = model_wrong.groupby(
                    "seizure_id", as_index=False
                ).wrong_pair_rho.median()
                paired = correct.merge(
                    wrong_median, on="seizure_id", validate="one_to_one"
                )
                pairing_rows.append({
                    "subject": subject,
                    "model": model_name,
                    "n_states": int(len(paired)),
                    "correct_rho": float(paired.correct_rho.median()),
                    "wrong_rho": float(paired.wrong_pair_rho.median()),
                    "correct_minus_wrong": float(
                        (paired.correct_rho - paired.wrong_pair_rho).median()
                    ),
                })

        residual = pd.read_csv(
            done_path.parent / "heldout_seizure_specific_residual.csv"
        )
        if not residual.empty:
            for model_name, model_residual in residual.groupby("model", sort=True):
                residual_rows.append({
                    "subject": subject,
                    "model": model_name,
                    "n_states": int(len(model_residual)),
                    "median_residual_rho": float(
                        model_residual.residual_rho.median()
                    ),
                    "mean_residual_rho": float(model_residual.residual_rho.mean()),
                })

    patient = pd.DataFrame(patient_rows).sort_values("subject")
    primary = patient.loc[patient.subject != DEVELOPMENT_TARGET].copy()
    if len(primary) != 15:
        raise RuntimeError(f"primary denominator drift: {len(primary)}/15")
    if len(checkpoint_cycles) != 1:
        raise RuntimeError(f"mixed HistoryRNN training budgets: {checkpoint_cycles}")
    pairing = pd.DataFrame(pairing_rows).sort_values("subject")
    residual = pd.DataFrame(residual_rows).sort_values("subject")
    headroom = pd.DataFrame(headroom_rows).sort_values("subject")
    channel_null = pd.DataFrame(channel_null_rows).sort_values(["subject", "model"])
    primary_pairing = pairing.loc[pairing.subject != DEVELOPMENT_TARGET].copy()
    primary_residual = residual.loc[residual.subject != DEVELOPMENT_TARGET].copy()
    primary_headroom = headroom.loc[headroom.subject != DEVELOPMENT_TARGET].copy()

    expected = pd.read_csv(args.g0_root.resolve() / "subject_causal_history_inventory.csv")
    expected_pairing = set(
        expected.loc[
            expected.g3_pairing_eligible
            & (expected.subject.astype(str) != DEVELOPMENT_TARGET),
            "subject",
        ].astype(str)
    )
    for model_name in ("R2", "E2", "EM"):
        observed = set(
            primary_pairing.loc[
                primary_pairing.model == model_name, "subject"
            ].astype(str)
        )
        if observed != expected_pairing:
            raise RuntimeError(
                f"state-seizure pairing denominator drifted for {model_name}"
            )

    patient.to_csv(root / "direct_transfer_patient_metrics.csv", index=False)
    pairing.to_csv(root / "state_seizure_pairing_metrics.csv", index=False)
    residual.to_csv(root / "seizure_specific_residual_metrics.csv", index=False)
    headroom.to_csv(root / "target_headroom_metrics.csv", index=False)
    channel_null.to_csv(
        root / "direct_transfer_channel_null_patient_metrics.csv", index=False
    )

    r2 = _contrast(primary.rho_increment_R2_minus_M1, 20260821)
    e0p5 = _contrast(primary.rho_increment_E0p5_minus_M1, 20260820)
    e2 = _contrast(primary.rho_increment_E2_minus_M1, 20260822)
    e6 = _contrast(primary.rho_increment_E6_minus_M1, 20260826)
    em = _contrast(primary.rho_increment_EM_minus_M1, 20260823)
    order = _contrast(primary.rho_true_R2_minus_order_shuffle, 20260824)
    zero_state = _contrast(primary.rho_true_R2_minus_zero_state, 20260825)
    e2_order = _contrast(primary.rho_true_E2_minus_time_shuffle, 20260827)
    em_order = _contrast(primary.rho_true_EM_minus_time_shuffle, 20260828)
    absolute = {
        model: _contrast(primary[f"rho_{model}"], 20260840 + index)
        for index, model in enumerate(("M0", "M1", "E0p5", "E2", "E6", "EM", "R2"))
    }
    primary_channel_null = channel_null.loc[
        channel_null.subject.astype(str) != DEVELOPMENT_TARGET
    ].copy()
    channel_null_statistics = {
        model: _contrast(
            primary_channel_null.loc[
                primary_channel_null.model == model,
                "margin_vs_channel_null_median",
            ],
            20260900 + index,
        )
        for index, model in enumerate(("M0", "M1", "E0p5", "E2", "E6", "EM", "R2"))
    }
    for model, statistics in channel_null_statistics.items():
        model_null = primary_channel_null.loc[
            primary_channel_null.model == model
        ]
        statistics["n_observed_above_patient_null_p95"] = int(
            np.sum(
                model_null.observed_patient_median_rho
                > model_null.channel_null_p95
            )
        )
        statistics["n_patient_permutation_p_lt_0p05"] = int(
            np.sum(model_null.patient_permutation_p < 0.05)
        )
    pairing_results = {
        model: _contrast(
            primary_pairing.loc[
                primary_pairing.model == model, "correct_minus_wrong"
            ],
            20260830 + index,
        )
        for index, model in enumerate(("R2", "E2", "EM"))
    }
    ewma_q = _bh([
        e0p5["one_sided_wilcoxon_p"], e2["one_sided_wilcoxon_p"],
        e6["one_sided_wilcoxon_p"], em["one_sided_wilcoxon_p"],
    ])
    for result_item, q_value in zip((e0p5, e2, e6, em), ewma_q):
        result_item["bh_fdr_q_across_ewma_family"] = q_value
    r2_increment_supported = (
        _positive(r2) and r2["one_sided_wilcoxon_p"] < 0.05
    )
    r2_channel_null_supported = (
        _positive(channel_null_statistics["R2"])
        and channel_null_statistics["R2"]["one_sided_wilcoxon_p"] < 0.05
    )
    r2_absolute_supported = (
        _positive(absolute["R2"]) and r2_channel_null_supported
    )
    r2_supported = r2_increment_supported and r2_absolute_supported
    order_supported = _positive(order) and order["one_sided_wilcoxon_p"] < 0.05
    zero_state_supported = (
        _positive(zero_state)
        and zero_state["one_sided_wilcoxon_p"] < 0.05
    )
    pairing_supported = (
        _positive(pairing_results["R2"])
        and pairing_results["R2"]["one_sided_wilcoxon_p"] < 0.05
    )
    em_supported = (
        _positive(em)
        and em["bh_fdr_q_across_ewma_family"] < 0.05
        and _positive(absolute["EM"])
        and _positive(channel_null_statistics["EM"])
        and channel_null_statistics["EM"]["one_sided_wilcoxon_p"] < 0.05
    )
    result = {
        "status": (
            "SEIZURE_CONDITIONED_HISTORY_SIGNAL_SUPPORTED"
            if r2_supported and order_supported and zero_state_supported and pairing_supported
            else (
                "DIRECT_R2_SIGNAL_SUPPORTED_BUT_NOT_SEIZURE_SPECIFIC"
                if r2_supported
                else (
                    "R2_RELATIVE_INCREMENT_ONLY_ABSOLUTE_NOT_SUPPORTED"
                    if r2_increment_supported
                    else "DIRECT_R2_INCREMENT_NOT_SUPPORTED"
                )
            )
        ),
        "contract": "topic5_history_rnn_direct_early_ictal_transfer_v0_2",
        "target_values_read": True,
        "primary_cohort": "15 development-excluded strict clinical-onset patients",
        "supportive_cohort": "all 16 strict clinical-onset patients",
        "n_completed_folds": len(done_paths),
        "history_checkpoint_cycles": int(next(iter(checkpoint_cycles))),
        "contact_denominator": {
            "definition": (
                "exact-name intersection of the frozen interictal scaffold and "
                "the rank dataset; not the full SEEG montage"
            ),
            "median_contacts_per_patient": float(primary.n_contacts.median()),
            "min_contacts_per_patient": int(primary.n_contacts.min()),
            "max_contacts_per_patient": int(primary.n_contacts.max()),
            "per_patient": {
                str(subject): int(count)
                for subject, count in zip(primary.subject, primary.n_contacts)
            },
            "power_note": (
                "Every Spearman, channel-shuffle null and patient contrast is "
                "computed on this denominator, so both positive and negative "
                "verdicts are coarse; a 6-contact patient has only 720 distinct "
                "channel permutations."
            ),
        },
        "primary_R2_minus_M1": r2,
        "activity_integrator_E0p5_minus_M1": e0p5,
        "activity_integrator_E2_minus_M1": e2,
        "activity_integrator_E6_minus_M1": e6,
        "multi_horizon_EM_minus_M1": em,
        "true_R2_minus_strict_order_shuffle": order,
        "true_R2_minus_zero_state": zero_state,
        "true_E2_minus_time_slot_shuffle": e2_order,
        "true_EM_minus_time_slot_shuffle": em_order,
        "state_seizure_correct_minus_wrong": pairing_results,
        "activity_integrator_status": (
            "SUPPORTED_ABOVE_STATIC_AND_UNORDERED_BASELINES"
            if em_supported
            else "NOT_SUPPORTED_AFTER_MULTIPLICITY_AND_ABSOLUTE_PERFORMANCE_CHECKS"
        ),
        "n_pairing_patients_primary": int(primary_pairing.subject.nunique()),
        "n_pairing_patients_supportive": int(pairing.subject.nunique()),
        "n_residual_patients_primary": int(primary_residual.subject.nunique()),
        "n_residual_patients_supportive": int(residual.subject.nunique()),
        "residual_patient_median_rho": {
            model: float(
                primary_residual.loc[
                    primary_residual.model == model, "median_residual_rho"
                ].median()
            )
            for model in ("R2", "E2", "EM")
        },
        "absolute_patient_median_rho": {
            model: float(primary[f"rho_{model}"].median())
            for model in ("M0", "M1", "E0p5", "E2", "E6", "EM", "R2")
        },
        "absolute_patient_rho_statistics": absolute,
        "all_contact_channel_shuffle": {
            "n_draws_per_patient": N_CHANNEL_PERMUTATIONS,
            "patient_fold": "median_across_seizures",
            "statistics": channel_null_statistics,
        },
        "target_headroom": {
            "n_repeat_seizure_patients": int(
                np.sum(primary_headroom.n_seizures >= 2)
            ),
            "median_pairwise_target_rho": float(
                primary_headroom.pairwise_target_rho.median(skipna=True)
            ),
            "median_leave_one_seizure_out_mean_oracle_rho": float(
                primary_headroom.leave_one_seizure_out_mean_oracle_rho.median(
                    skipna=True
                )
            ),
        },
        "claim_boundary": (
            "Predictive association only. Causal network shaping is not established by this analysis."
        ),
    }
    (root / "DIRECT_TRANSFER_SUMMARY.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
