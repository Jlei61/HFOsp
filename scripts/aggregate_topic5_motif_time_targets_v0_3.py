#!/usr/bin/env python3
"""Which motif explains how long the step took — read as a ladder, not a single score.

Every verdict is computed from the table at the moment of writing.  An earlier round
shipped a paragraph asserting a conclusion its own numbers contradicted, so nothing
here states an outcome that is not read back out of the data.

The ladder, each rung answering one question and none of them licensing another:

1. ``STEP_DISTANCE`` vs ``STEP_ONLY``  — does the distance actually travelled carry
   information about the duration, beyond how far into the event we are?
2. ``M0`` vs the strongest baseline    — does a recurrent local field add anything
   beyond that raw distance?
3. ``M1-M0``, ``M2-M1``, ``M3-M2``     — do the corridor, the directed transport and
   the axial feed-forward each add more?

Two things are deliberately *not* pooled into the statistics: the spread between
optimisation starts (a choice, not measurement noise) and the contact score (outside
the objective entirely this round).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULT_ROOT = ROOT / "results/topic5_motif_time_targets_v0_3"
BOOTSTRAP = 10000
RNG = np.random.default_rng(20260819)

CHAIN = ("M0_ISOTROPIC_DIFFUSION", "M1_AXIAL_CORRIDOR",
         "M2_DIRECTED_TRANSPORT", "M3_AXIAL_FEEDFORWARD_TRANSIENT")
FREE_ARM = "MFREE_LOW_RANK_UPPER_BOUND"
BASELINES = ("baseline_STEP_ONLY", "baseline_STEP_DISTANCE", "baseline_STATIC_TARGET")
# Which parameter each layer introduces, and whether it is bounded at zero.  Only two
# of the three are: ``beta`` is a SIGNED direction bias and its fitted median is
# negative, so calling all three "non-negative" was wrong.  Also, a parameter leaving
# zero means the optimiser moved it — not that a mechanism was "engaged"; that word
# claims more than the number supports.
LAYER_PARAMETER = {
    "M1_AXIAL_CORRIDOR": ("fitted_eta_raw", "non-negative strength, bounded at zero"),
    "M2_DIRECTED_TRANSPORT": ("fitted_beta", "SIGNED direction bias, not bounded"),
    "M3_AXIAL_FEEDFORWARD_TRANSIENT": ("fitted_gamma_raw",
                                       "non-negative strength, bounded at zero"),
}


def cohort(values: np.ndarray, label: str) -> dict:
    """Patient-level statistics.  The patient is the unit; starts are not resampled."""
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if values.size < 6:
        return {"label": label, "n": int(values.size)}
    draws = np.median(RNG.choice(values, size=(BOOTSTRAP, values.size), replace=True), axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    positive = int((values > 1e-12).sum())
    negative = int((values < -1e-12).sum())
    tied = int(values.size - positive - negative)
    return {
        "label": label, "n": int(values.size),
        "median": float(np.median(values)), "mean": float(values.mean()),
        "median_ci95": [float(low), float(high)],
        "crosses_zero": bool(low < 0.0 < high),
        # A warm-started child often keeps exactly the parent's checkpoint, so these
        # differences carry a spike at zero.  The bootstrap median then lands its lower
        # bound exactly on zero, "low < 0 < high" is False, and a purely interval-based
        # verdict reads that boundary artefact as evidence.  The strict version is what
        # the verdict uses.
        "interval_excludes_zero_strictly": bool(low > 0.0 or high < 0.0),
        "n_positive": positive, "n_negative": negative, "n_tied": tied,
        "wilcoxon_p": float(stats.wilcoxon(values).pvalue),
        # both sign tests, because they answer different questions and disagree here:
        # counting ties asks "did it help across the cohort", dropping them asks
        # "among patients where it changed anything, did it help more often than hurt"
        "sign_test_p": float(stats.binomtest(positive, values.size, 0.5).pvalue),
        "sign_test_p_ties_dropped": (
            float(stats.binomtest(positive, positive + negative, 0.5).pvalue)
            if positive + negative else float("nan")),
    }


def rung(table: pd.DataFrame, better: str, worse: str, label: str) -> dict:
    """``worse - better`` per patient on the held-out time error; positive favours ``better``."""
    left = table[table["arm"] == better].set_index("patient")["time_mse"]
    right = table[table["arm"] == worse].set_index("patient")["time_mse"]
    shared = sorted(set(left.index) & set(right.index))
    return cohort((right.loc[shared] - left.loc[shared]).to_numpy(), label)


def main() -> int:
    table = pd.read_csv(RESULT_ROOT / "PER_ARM_SCORES.csv")
    arms = [arm for arm in list(CHAIN) + [FREE_ARM] if arm in set(table["arm"])]
    per_patient = table.drop_duplicates(subset=["patient"]).set_index("patient")

    # --- rung 1: does the distance travelled explain the duration at all? ----
    ladder = {}
    step_only = per_patient["baseline_STEP_ONLY"]
    ladder["rung1_distance_beyond_step_index"] = cohort(
        (step_only - per_patient["baseline_STEP_DISTANCE"]).to_numpy(),
        "STEP_ONLY minus STEP_DISTANCE; positive = the distance travelled carries "
        "duration information the step index does not")
    ladder["rung1b_static_target_beyond_distance"] = cohort(
        (per_patient["baseline_STEP_DISTANCE"]
         - per_patient["baseline_STATIC_TARGET"]).to_numpy(),
        "STEP_DISTANCE minus STATIC_TARGET; positive = some contacts are habitually "
        "early or late beyond what the distance explains")

    # --- rung 2: does the recurrent field beat the strongest baseline? -------
    # The comparator is fixed before the test score is read.  The first pass took the
    # per-patient minimum over the three test-split baselines, which is choosing the
    # reference after seeing the number it is compared against.
    # STATIC_TARGET strictly nests the other two — it knows the step index, the distance
    # travelled AND the contact's habitual earliness — so it is the reference by
    # construction, chosen for what it knows rather than for what it scored.  Picking by
    # test performance is what the first pass did wrong; picking the richest baseline
    # needs no look at any score.
    #
    # A validation-selected ridge was tried and is reported as a sensitivity, not as the
    # primary: it left the reference unchanged at the median but weakened it badly on
    # three patients, and the only nominally significant rung-2 result rides on exactly
    # that weakening.  A regularisation that makes the control worse is not adopted.
    best_baseline = per_patient["baseline_STATIC_TARGET"]
    comparator_source = {
        "rule": "fixed STATIC_TARGET, unregularised; chosen because it strictly nests "
                "the other two baselines, not because of its score",
        "rejected_alternatives": {
            "per_patient_minimum_over_test": "selects the reference after seeing the "
                                             "split it is scored against",
            "validation_selected_ridge": "reported as a sensitivity below; it weakens "
                                         "the reference on three patients and the one "
                                         "significant rung-2 result does not survive "
                                         "either that removal or the unregularised "
                                         "reference",
        },
    }
    sensitivity_path = RESULT_ROOT / "TIME_BASELINES_VALIDATION_SELECTED.csv"
    comparator_sensitivity = {}
    if sensitivity_path.exists():
        alternative = pd.read_csv(sensitivity_path).set_index("patient")
        for arm in [a for a in list(CHAIN) + [FREE_ARM] if a in set(table["arm"])]:
            scores = table[table["arm"] == arm].set_index("patient")["time_mse"]
            shared = sorted(set(scores.index) & set(alternative.index))
            comparator_sensitivity[arm] = cohort(
                (alternative.loc[shared, "comparator_test_mse"]
                 - scores.loc[shared]).to_numpy(),
                f"{arm} against the validation-selected-ridge reference (sensitivity only)")
    for arm in arms:
        scores = table[table["arm"] == arm].set_index("patient")["time_mse"]
        shared = sorted(set(scores.index) & set(best_baseline.index))
        ladder[f"rung2_{arm}_beyond_best_baseline"] = cohort(
            (best_baseline.loc[shared] - scores.loc[shared]).to_numpy(),
            f"strongest time baseline minus {arm}; positive = the recurrent field adds "
            "something no baseline had")

    # --- rung 3: does each mechanism add on top of the previous one? ---------
    for index in range(1, len(CHAIN)):
        rich, simple = CHAIN[index], CHAIN[index - 1]
        if rich in arms and simple in arms:
            ladder[f"rung3_{rich}_over_{simple}"] = rung(
                table, rich, simple, f"{simple} minus {rich}; positive = the mechanism "
                f"{rich} adds helps beyond {simple}")
    if FREE_ARM in arms:
        for arm in CHAIN:
            if arm in arms:
                ladder[f"free_low_rank_over_{arm}"] = rung(
                    table, FREE_ARM, arm,
                    f"{arm} minus the free low-rank operator; positive = the free "
                    "operator does better.  It is rank-constrained and does not contain "
                    "the structured kernels, so this compares two families and does not "
                    "bound either")

    # --- did each mechanism ever switch on? ---------------------------------
    engagement = {}
    for arm, (column, semantics) in LAYER_PARAMETER.items():
        if arm not in arms or column not in table.columns:
            continue
        values = table[table["arm"] == arm].set_index("patient")[column].dropna()
        moved = values.abs() > 1e-6
        engagement[arm] = {
            "parameter": column, "parameter_semantics": semantics,
            "n_patients": int(values.size),
            "n_moved_off_zero": int(moved.sum()),
            "n_negative": int((values < -1e-6).sum()),
            "median_when_moved": float(values[moved].median()) if moved.any() else None,
            "note": "a tie with the parent means either that this term does not help here "
                    "or that the optimiser never moved it; the counts separate those two. "
                    "Moving off zero says the optimiser used the parameter, NOT that a "
                    "mechanism was engaged in any biological sense",
        }

    # --- optimisation basin sensitivity, reported apart from the statistics --
    basin = {}
    for arm in arms:
        frame = table[table["arm"] == arm]
        spread = frame["validation_spread_across_starts"].dropna()
        if not spread.size:
            continue
        modes = (frame["chosen_head_mode"].value_counts().to_dict()
                 if "chosen_head_mode" in frame.columns else {})
        basin[arm] = {
            "n_patients": int(spread.size),
            "median_spread": float(spread.median()),
            "p90_spread": float(spread.quantile(0.9)),
            "n_starts_median": float(frame["n_starts"].median()),
            # both inheritance modes are offered because neither dominates; if one of
            # them never wins, the extra starts were wasted and that should be visible
            "chosen_head_mode_counts": {str(k): int(v) for k, v in modes.items()},
        }

    # --- internal validity: does the gain track the clue, on the same estimand -
    validity = {}
    for column, name in (("adjacent_partial_spearman", "adjacent_steps"),
                         ("all_pairs_partial_spearman", "all_pairs")):
        if column not in table.columns:
            continue
        clue = per_patient[column]
        for arm in arms:
            scores = table[table["arm"] == arm].set_index("patient")["time_mse"]
            shared = sorted(set(scores.index) & set(clue.index))
            gain = (best_baseline.loc[shared] - scores.loc[shared]).to_numpy()
            reference = clue.loc[shared].to_numpy()
            keep = np.isfinite(gain) & np.isfinite(reference)
            if keep.sum() < 8:
                continue
            rho, pvalue = stats.spearmanr(reference[keep], gain[keep])
            validity[f"{arm}|{name}"] = {
                "spearman_rho": float(rho), "p_value": float(pvalue),
                "n_patients": int(keep.sum())}

    # --- inheritance and split hygiene --------------------------------------
    gaps = table["inheritance_max_state_gap"].dropna()
    audit = {
        "warm_start_max_state_gap": float(gaps.max()) if gaps.size else None,
        "warm_start_all_exact": bool(gaps.size and float(gaps.max()) < 1e-5),
        "n_patients": int(table["patient"].nunique()),
        "n_rows": int(len(table)),
        "m0_theta_recorded_as_null": bool(
            table[table["arm"] == CHAIN[0]]["chosen_theta_init"].isna().all()),
        "contact_score_is_outside_the_objective": True,
    }

    def verdict(key: str, noise_floor: float = float("nan")) -> str:
        """Three things must agree before a rung counts as supported.

        The interval alone is not enough: these differences carry a spike at zero, so a
        lower bound resting exactly on zero is a boundary artefact.  The sign test alone
        is not enough either: an effect can be reliably signed and still be far smaller
        than the spread between optimisation starts, in which case it cannot be told
        apart from where the optimiser happened to land.
        """
        entry = ladder.get(key, {})
        if "median" not in entry:
            return "INSUFFICIENT"
        median = entry["median"]
        cohort_p = entry.get("sign_test_p", float("nan"))
        dropped_p = entry.get("sign_test_p_ties_dropped", float("nan"))

        # The cohort question is "did this help across the 28 patients".  Dropping the
        # ties asks the narrower "among patients where anything changed, which way did
        # it go" — and the ties here are real outcomes (the nested model kept its
        # parent), not missing data.  So the cohort test carries the verdict and the
        # ties-dropped test can only earn a weaker label.
        strong = (entry["interval_excludes_zero_strictly"]
                  or (np.isfinite(cohort_p) and cohort_p < 0.05))
        weak = np.isfinite(dropped_p) and dropped_p < 0.05
        # The start spread is not a null distribution — it is a validation-side
        # sensitivity, a different random variable from the held-out patient effect.
        # It therefore annotates the label rather than gating it.
        sensitive = (np.isfinite(noise_floor) and noise_floor > 0
                     and abs(median) < noise_floor)
        if strong:
            label = "SUPPORTED" if median > 0 else "REVERSED"
            return f"{label} (OPTIMISATION-SENSITIVE)" if sensitive else label
        if weak:
            return "WEAK EXPLORATORY / OPTIMISATION-SENSITIVE" if sensitive else \
                   "WEAK EXPLORATORY"
        return "NOT ESTABLISHED"

    summary = {
        "contract": "topic5_motif_time_targets_v0_3_aggregate",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "question": "does the adjacent time interval distinguish uniform diffusion, an "
                    "axial corridor, directed transport and an axial feed-forward?",
        "unit_of_analysis": "patient; optimisation starts are choices, not seeds",
        "time_proxy_note": "within-event spectral-centroid position; not recruitment "
                           "time, not conduction delay, never a velocity",
        "ladder": ladder,
        "mechanism_engagement": engagement,
        "optimisation_basin_sensitivity": basin,
        "internal_validity_gain_vs_clue": validity,
        "engineering_audit": audit,
        "comparator_selection": comparator_source,
        "comparator_sensitivity_validation_selected_ridge": comparator_sensitivity,
        "naming_correction": {
            "arm_value_in_csv": FREE_ARM,
            "correct_description": "unconstrained low-rank alternative operator, rank 4-7 "
                                   "on the node graph.  It is NOT an upper bound: the "
                                   "structured kernels are full rank on their local "
                                   "support, so this arm does not contain them.  Beating "
                                   "it says nothing about whether the candidate family "
                                   "was wide enough.",
        },
        "verdicts": {
            "distance_explains_duration": verdict("rung1_distance_beyond_step_index"),
            **{f"recurrent_field_beyond_baselines|{arm}":
               verdict(f"rung2_{arm}_beyond_best_baseline",
                       basin.get(arm, {}).get("median_spread", float("nan")))
               for arm in CHAIN},
            **{f"{rich}_adds_over_{simple}":
               verdict(f"rung3_{rich}_over_{simple}",
                       basin.get(rich, {}).get("median_spread", float("nan")))
               for simple, rich in zip(CHAIN, CHAIN[1:])},
            # The free arm is rank-constrained while the structured kernels are full
            # rank on their local support, so it does not contain the motifs and is not
            # a bound on them.  Beating it therefore cannot show the candidate family
            # was wide enough, and this design does not answer that question at all.
            "candidate_motifs_too_narrow": "NOT ANSWERED BY THIS DESIGN",
            "free_low_rank_alternative_vs_motifs": {
                arm: verdict(f"free_low_rank_over_{arm}") for arm in CHAIN},
        },
    }
    (RESULT_ROOT / "MOTIF_TIME_EVIDENCE.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"patients {audit['n_patients']}   rows {audit['n_rows']}")
    print(f"warm-start 继承最大状态差 {audit['warm_start_max_state_gap']}  "
          f"全部逐位一致={audit['warm_start_all_exact']}")

    def show(title: str, keys: list[str]) -> None:
        print(f"\n{title}")
        for key in keys:
            entry = ladder.get(key)
            if not entry or "median" not in entry:
                print(f"  {key:52s} n={entry.get('n', 0) if entry else 0} 不足以聚合")
                continue
            low, high = entry["median_ci95"]
            span = ("跨零" if entry["crosses_zero"]
                    else ("不跨零" if entry["interval_excludes_zero_strictly"] else "触零"))
            print(f"  {key:52s} n={entry['n']:2d} 中位={entry['median']:+.5f} "
                  f"[{low:+.5f},{high:+.5f}] {span} "
                  f"+/-={entry['n_positive']}/{entry['n_negative']}/{entry['n_tied']} "
                  f"符号p={entry['sign_test_p']:.3f}(剔并列 {entry['sign_test_p_ties_dropped']:.3f})")

    show("第一层：距离是否解释时间间隔", [k for k in ladder if k.startswith("rung1")])
    show("第二层：递归场是否超过最强基线", [k for k in ladder if k.startswith("rung2")])
    show("第三层：每个机制是否再增加解释", [k for k in ladder if k.startswith("rung3")])
    show("自由低秩替代算子 vs 各 motif（它不含 motif，故只说明族间优劣，不判候选是否够宽）",
         [k for k in ladder if k.startswith("free_low_rank")])

    print("\n优化器是否把该参数移离零（不等于「机制被启用」）")
    for arm, entry in engagement.items():
        print(f"  {arm:34s} 移离零 {entry['n_moved_off_zero']}/{entry['n_patients']} "
              f"(其中为负 {entry['n_negative']})  中位={entry['median_when_moved']}  "
              f"[{entry['parameter_semantics']}]")
    print("\n优化盆地敏感性（起点间 validation 散布，不进统计）")
    for arm, entry in basin.items():
        print(f"  {arm:34s} 起点数={entry['n_starts_median']:.0f} "
              f"中位散布={entry['median_spread']:.5f} 九成={entry['p90_spread']:.5f} "
              f"读出模式胜出={entry['chosen_head_mode_counts']}")
    print("\n判决")
    for key, value in summary["verdicts"].items():
        if isinstance(value, dict):
            print(f"  {key}")
            for inner, state in value.items():
                print(f"      {str(state):28s} {inner}")
        else:
            print(f"  {str(value):28s} {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
