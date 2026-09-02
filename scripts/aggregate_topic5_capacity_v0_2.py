#!/usr/bin/env python3
"""Phase H: patient-first aggregation and the cohort evidence matrix.

The aggregation order is fixed and never shortcut:

    horizon-specific decision -> event -> seed / null basis -> patient -> cohort

Millions of decisions are not independent samples; the patient is the unit of
cohort inference.  A patient's aligned effect is always compared with that same
patient's own null median, never with a pooled null.

Every effect is reported as ``null - aligned`` on a loss, so a positive number
means the patient-aligned structure did better.  p-values are descriptive and
never decide whether a later analysis runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_strict_history_motif_v0_2 import (  # noqa: E402
    CHECKPOINT_HORIZONS,
    CHECKPOINT_HORIZON_WEIGHT,
    CHECKPOINT_SUFFIX_WEIGHT,
    primary_field_kind,
)

RESULT_ROOT = ROOT / "results/topic5_capacity_constrained_history_motif_v0_2"
NEAR_ZERO = 1e-3
BOOTSTRAP = 10000
RNG = np.random.default_rng(20260817)

GROUP_KEYS = ["patient", "block", "baseline_level", "structure", "family", "rank",
              "data_fraction", "basis_fraction", "f_form", "prefix_len", "time_head"]
METRICS = ["primary_objective", "common_target_objective",
           "total_nll_h1", "total_nll_h2", "total_nll_h3",
           "total_nll_h4", "total_nll_h5", "suffix_balanced_bce", "suffix_balanced_brier",
           "endpoint_distance_mm", "ordered_path_ablation_cost"]


def objective_from(scalars: dict, per_horizon: dict, family: str) -> float:
    kind = primary_field_kind(family)
    total = 0.0
    for horizon in CHECKPOINT_HORIZONS:
        value = per_horizon["total_nll"][horizon - 1]
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            total += CHECKPOINT_HORIZON_WEIGHT * value
    return total + CHECKPOINT_SUFFIX_WEIGHT * scalars[f"{kind}_balanced_bce"]


def unit_rows(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, states = [], {"complete": 0, "unresolved": 0, "missing": 0}
    for unit in manifest.to_dict("records"):
        directory = RESULT_ROOT / unit["output_dir"]
        status_path, metrics_path = directory / "status.json", directory / "metrics.json"
        if not status_path.exists():
            states["missing"] += 1
            continue
        state = json.loads(status_path.read_text()).get("state")
        if state != "complete" or not metrics_path.exists():
            states["unresolved"] += 1
            continue
        states["complete"] += 1
        payload = json.loads(metrics_path.read_text())
        test = payload["metrics"]["development_test"]
        ablated = test["ordered_path_ablated"]
        kind = primary_field_kind(unit["family"])
        row = {key: unit[key] for key in
               ["patient", "block", "baseline_level", "structure", "null_id", "family", "rank",
                "data_fraction", "basis_fraction", "f_form", "prefix_len", "time_head", "seed",
                "unit_id"]}
        row["primary_objective"] = objective_from(test["scalars"], test["per_horizon"], unit["family"])
        for horizon in range(1, 6):
            row[f"total_nll_h{horizon}"] = test["per_horizon"]["total_nll"][horizon - 1]
            row[f"denominator_h{horizon}"] = test["per_horizon"]["denominator"][horizon - 1]
        row["suffix_balanced_bce"] = test["scalars"][f"{kind}_balanced_bce"]
        row["suffix_balanced_brier"] = test["scalars"][f"{kind}_balanced_brier"]
        row["endpoint_distance_mm"] = test["scalars"][f"{kind}_endpoint_distance_mm"]
        row["ordered_path_ablation_cost"] = objective_from(
            ablated["scalars"], ablated["per_horizon"], unit["family"]) - row["primary_objective"]
        # Each family's primary objective closes on its own field (the autonomous
        # family on the five-step field it can generate, the direct family on its
        # independent full-suffix read-out), which is the frozen design and must
        # stay separate.  A difference BETWEEN the families therefore has a
        # different second term on each side, so the common-target objective below
        # is recorded as well: every unit scores suffix5, so this one is on a
        # single scale across families.
        row["common_target_objective"] = objective_from(
            test["scalars"], test["per_horizon"], "AUTONOMOUS_SHARED_OPERATOR")
        row["time_proxy_loss"] = test.get("time_proxy_loss", np.nan)
        row["ordered_parameter_count"] = payload["diagnostics"]["ordered_parameter_count"]
        row["total_parameter_count"] = payload["diagnostics"]["total_parameter_count"]
        row["transition_spectral_radius"] = payload["diagnostics"]["transition_spectral_radius"]
        row["best_epoch"] = payload["training"]["best_epoch"]
        row["wall_seconds"] = payload["training"]["wall_seconds"]
        row["nonfinite_batches"] = payload["training"]["nonfinite_batches"]
        rows.append(row)
    return pd.DataFrame(rows), states


def baseline_rows() -> pd.DataFrame:
    audit = json.loads(
        (RESULT_ROOT / "baseline" / "UNORDERED_INVARIANCE_AUDIT.json").read_text())
    rows = []
    for unit in audit["units"]:
        if "error" in unit:
            continue
        test = unit["held_out_scores"].get("development_test")
        if test is None:
            continue
        for family in ("DIRECT_HORIZON_UPPER_BOUND", "AUTONOMOUS_SHARED_OPERATOR", "ORDERLESS_BAG"):
            rows.append({
                "patient": unit["patient"], "baseline_level": unit["level"],
                "prefix_len": unit["prefix_len"], "family": family,
                "h0_primary_objective": objective_from(test["scalars"], test["per_horizon"], family),
                "h0_suffix_balanced_bce": test["scalars"][
                    f"{primary_field_kind(family)}_balanced_bce"],
                "h0_total_nll_h1": test["per_horizon"]["total_nll"][0],
                "h0_selected_rank": unit["selected_rank"],
                "h0_parameters": unit["trainable_parameters"],
            })
    return pd.DataFrame(rows)


def seed_spread(table: pd.DataFrame) -> pd.DataFrame:
    """Within-arm seed variability — the noise floor every effect must be read against.

    Multi-seed arms exist only where the frozen matrix asked for them, so this is
    reported as the measured floor rather than used to adjust any effect.
    """
    grouped = table.groupby(GROUP_KEYS + ["null_id"], dropna=False)["primary_objective"]
    spread = grouped.agg(["count", "min", "max", "std"]).reset_index()
    spread["seed_spread"] = spread["max"] - spread["min"]
    return spread[spread["count"] > 1]


def paired_effect_seed_spread(table: pd.DataFrame) -> pd.DataFrame:
    """The noise floor that actually applies to the primary claim.

    The marginal spread of one arm across seeds overstates the uncertainty of a
    paired contrast, because the two arms share the seed, the data and the frozen
    baseline.  This recomputes the central effect once per aligned seed against
    the same null median and reports how much that effect moves.
    """
    core = table[(table["block"] == "CORE1") & (table["rank"] == 4)
                 & (table["baseline_level"] == "U_FULL_SET")
                 & (table["family"] == "AUTONOMOUS_SHARED_OPERATOR")]
    rows = []
    for patient, group in core.groupby("patient"):
        nulls = group[group["structure"] == "H1_ANGLE_ROTATED_AXIS"]["primary_objective"]
        aligned = group[group["structure"] == "H1_PATIENT_ALIGNED"]
        if nulls.empty or len(aligned) < 2:
            continue
        reference = float(nulls.median())
        effects = reference - aligned["primary_objective"].to_numpy()
        rows.append({"patient": patient, "n_aligned_seeds": int(len(effects)),
                     "n_angle_nulls": int(len(nulls)),
                     "effect_min": float(effects.min()), "effect_max": float(effects.max()),
                     "effect_seed_spread": float(effects.max() - effects.min()),
                     "effect_median": float(np.median(effects))})
    return pd.DataFrame(rows)


def collapse(table: pd.DataFrame) -> pd.DataFrame:
    """seed -> null basis -> one number per (patient, arm)."""
    by_seed = table.groupby(GROUP_KEYS + ["null_id"], dropna=False)[
        METRICS + ["time_proxy_loss", "ordered_parameter_count", "transition_spectral_radius"]
    ].median().reset_index()
    by_null = by_seed.groupby(GROUP_KEYS, dropna=False).agg(
        **{metric: (metric, "median") for metric in METRICS},
        time_proxy_loss=("time_proxy_loss", "median"),
        ordered_parameter_count=("ordered_parameter_count", "median"),
        transition_spectral_radius=("transition_spectral_radius", "median"),
        n_nulls=("null_id", "nunique"),
    ).reset_index()
    return by_null


def cohort_statistics(values: np.ndarray, label: str) -> dict:
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return {"label": label, "n": 0}
    boot = RNG.choice(values, size=(BOOTSTRAP, values.size), replace=True).mean(axis=1)
    median_boot = np.median(
        RNG.choice(values, size=(BOOTSTRAP, values.size), replace=True), axis=1)
    signs = np.sign(values)
    flips = RNG.choice([-1.0, 1.0], size=(BOOTSTRAP, values.size))
    sign_null = (flips * np.abs(values)).mean(axis=1)
    out = {
        "label": label,
        "n": int(values.size),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "median_ci95": [float(np.percentile(median_boot, 2.5)), float(np.percentile(median_boot, 97.5))],
        "mean_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "n_positive": int((values > NEAR_ZERO).sum()),
        "n_negative": int((values < -NEAR_ZERO).sum()),
        "n_near_zero": int((np.abs(values) <= NEAR_ZERO).sum()),
        "sign_flip_p": float((np.abs(sign_null) >= abs(values.mean())).mean()),
    }
    if values.size >= 6 and np.any(values != 0):
        out["wilcoxon_p"] = float(stats.wilcoxon(values).pvalue)
    del signs
    return out


def runs_contrast(table: pd.DataFrame, core: dict,
                  reference: tuple[str, str], comparison: tuple[str, str],
                  metric: str = "primary_objective") -> dict:
    """Per-patient lists of ``reference - comparison`` over every run pair.

    ``contrast`` works on the collapsed table and so has one number per patient;
    this keeps every seed/null run so the interval can resample them.
    """
    def runs(family: str, structure: str) -> dict:
        frame = table
        for key, value in {**core, "family": family, "structure": structure}.items():
            if key in frame.columns:
                frame = frame[frame[key] == value]
        return frame.groupby("patient")[metric].apply(list).to_dict()

    left, right = runs(*reference), runs(*comparison)
    return {patient: [a - b for a in left[patient] for b in right[patient]]
            for patient in sorted(set(left) & set(right))}


def seed_aware_interval(per_patient_runs: dict, label: str) -> dict:
    """Cohort interval that resamples patients AND the training seed.

    ``cohort_statistics`` bootstraps the collapsed per-patient numbers, so its
    interval carries patient sampling only: ``collapse`` has already taken the
    median over seeds and nulls, and re-training variance is gone by then.  Here
    each patient keeps its list of individual runs, and every bootstrap draw
    picks a patient and then one run from that patient, so the interval carries
    both sources.  Arms that were fit once per patient (the frozen unordered
    baselines) contribute a single run and therefore no seed term.
    """
    patients = sorted(key for key, runs in per_patient_runs.items() if len(runs))
    if len(patients) < 3:
        return {"label": label, "n": len(patients)}
    runs = [np.asarray([v for v in per_patient_runs[key] if np.isfinite(v)], dtype=float)
            for key in patients]
    runs = [row for row in runs if row.size]
    n = len(runs)
    draws = np.empty(BOOTSTRAP)
    for index in range(BOOTSTRAP):
        chosen = RNG.integers(0, n, n)
        draws[index] = np.median([runs[j][RNG.integers(runs[j].size)] for j in chosen])
    point = float(np.median([np.median(row) for row in runs]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "label": label,
        "n": n,
        "median": point,
        "median_ci95_seed_aware": [float(low), float(high)],
        "crosses_zero": bool(low < 0.0 < high),
        "n_runs_median": float(np.median([row.size for row in runs])),
        "note": "resamples patients and, within each drawn patient, one training run; "
                "compare against median_ci95 in the matching patient-only entry",
    }


def contrast(collapsed: pd.DataFrame, reference: dict, comparison: dict,
             metric: str = "primary_objective") -> pd.DataFrame:
    """``reference - comparison`` per patient (positive favours ``comparison``)."""
    def select(filters: dict) -> pd.DataFrame:
        frame = collapsed
        for key, value in filters.items():
            frame = frame[frame[key] == value]
        return frame.set_index("patient")[metric]

    left, right = select(reference), select(comparison)
    shared = sorted(set(left.index) & set(right.index))
    return pd.DataFrame({
        "patient": shared,
        "reference": left.loc[shared].to_numpy(),
        "comparison": right.loc[shared].to_numpy(),
        "effect": left.loc[shared].to_numpy() - right.loc[shared].to_numpy(),
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RESULT_ROOT))
    arguments = parser.parse_args()
    out = Path(arguments.out)

    manifest = pd.read_csv(RESULT_ROOT / "MASTER_UNIT_MANIFEST.csv")
    eligible = manifest[manifest["eligible"]]
    table, states = unit_rows(eligible)
    table.to_csv(out / "PER_UNIT_DEVELOPMENT_TEST_SCORES.csv", index=False)
    collapsed = collapse(table)
    collapsed.to_csv(out / "PER_PATIENT_ARM_SCORES.csv", index=False)
    spread = seed_spread(table)
    spread.to_csv(out / "PER_PATIENT_SEED_SPREAD.csv", index=False)
    paired = paired_effect_seed_spread(table)
    paired.to_csv(out / "PER_PATIENT_PAIRED_EFFECT_SEED_SPREAD.csv", index=False)
    baselines = baseline_rows()

    core = dict(block="CORE1", baseline_level="U_FULL_SET", rank=4, data_fraction=100,
                basis_fraction=100, f_form="FULL", prefix_len=3, time_head=False)
    evidence: dict[str, dict] = {}
    tables: dict[str, pd.DataFrame] = {}

    # ---- E2/E3 direct vs autonomous, aligned vs every null -----------------
    direct_vs_auto = []
    for family in ("DIRECT_HORIZON_UPPER_BOUND", "AUTONOMOUS_SHARED_OPERATOR"):
        for null_structure in ("H1_ANGLE_ROTATED_AXIS", "H1_IDENTITY_PERMUTED",
                               "H1_LOCALITY_REWIRED", "H1_GEOMETRY_LAYOUT", "H1_SHAFT_GRADIENT"):
            frame = contrast(collapsed,
                             {**core, "family": family, "structure": null_structure},
                             {**core, "family": family, "structure": "H1_PATIENT_ALIGNED"})
            frame["family"], frame["null_structure"] = family, null_structure
            direct_vs_auto.append(frame)
            evidence[f"E3_aligned_vs_{null_structure}_{family}"] = cohort_statistics(
                frame["effect"].to_numpy(), f"{null_structure} minus aligned ({family})")
            if null_structure == "H1_ANGLE_ROTATED_AXIS":
                evidence[f"E3_aligned_vs_{null_structure}_{family}_seed_aware"] = (
                    seed_aware_interval(
                        runs_contrast(table, {**core, "family": family},
                                      (family, null_structure), (family, "H1_PATIENT_ALIGNED")),
                        f"{null_structure} minus aligned ({family}), patients and seeds "
                        f"resampled together"))
        gain = contrast(collapsed, {**core, "family": family, "structure": "H1_FREE_LOW_RANK"},
                        {**core, "family": family, "structure": "H1_PATIENT_ALIGNED"})
        gain["family"], gain["null_structure"] = family, "H1_FREE_LOW_RANK"
        direct_vs_auto.append(gain)
    tables["PER_PATIENT_DIRECT_VS_AUTONOMOUS"] = pd.concat(direct_vs_auto, ignore_index=True)

    # ---- E2: how much of the direct advantage survives the shared operator --
    combined = pd.concat(direct_vs_auto, ignore_index=True)
    angle = combined[combined["null_structure"] == "H1_ANGLE_ROTATED_AXIS"]
    wide = angle.pivot_table(index="patient", columns="family", values="effect").dropna()
    if {"DIRECT_HORIZON_UPPER_BOUND", "AUTONOMOUS_SHARED_OPERATOR"} <= set(wide.columns):
        evidence["E2_direct_minus_autonomous_structure_effect"] = cohort_statistics(
            (wide["DIRECT_HORIZON_UPPER_BOUND"] - wide["AUTONOMOUS_SHARED_OPERATOR"]).to_numpy(),
            "difference between the two families' AXIS-STRUCTURE effects (each family's own "
            "rotated minus aligned), not a difference in predictive accuracy.  Each side is "
            "scored on that family's own field, so the scales are not identical; see the "
            "common-target entry below")
        common = contrast(collapsed,
                          {**core, "family": "DIRECT_HORIZON_UPPER_BOUND",
                           "structure": "H1_ANGLE_ROTATED_AXIS"},
                          {**core, "family": "DIRECT_HORIZON_UPPER_BOUND",
                           "structure": "H1_PATIENT_ALIGNED"},
                          metric="common_target_objective")
        common_auto = contrast(collapsed,
                               {**core, "family": "AUTONOMOUS_SHARED_OPERATOR",
                                "structure": "H1_ANGLE_ROTATED_AXIS"},
                               {**core, "family": "AUTONOMOUS_SHARED_OPERATOR",
                                "structure": "H1_PATIENT_ALIGNED"},
                               metric="common_target_objective")
        merged = common.set_index("patient")["effect"].to_frame("direct").join(
            common_auto.set_index("patient")["effect"].to_frame("autonomous"), how="inner")
        evidence["E2_direct_minus_autonomous_structure_effect_common_suffix5"] = cohort_statistics(
            (merged["direct"] - merged["autonomous"]).to_numpy(),
            "same difference of axis-structure effects with BOTH families scored on the "
            "common suffix5 target, so the two sides are on one scale")

    # ---- ordered vs the orderless bag on the same aligned dictionary -------
    bag = contrast(collapsed, {**core, "family": "ORDERLESS_BAG",
                               "structure": "H1_ALIGNED_ORDERLESS_BAG"},
                   {**core, "family": "DIRECT_HORIZON_UPPER_BOUND",
                    "structure": "H1_PATIENT_ALIGNED"})
    evidence["E1_aligned_ordered_minus_aligned_bag"] = cohort_statistics(
        bag["effect"].to_numpy(),
        "aligned orderless bag minus aligned ordered, both on the same frozen dictionary and "
        "within a factor of ~1.1 on parameter count: this is the capacity-matched test of "
        "whether reading the order adds anything (positive = order helps)")
    evidence["E1_aligned_ordered_minus_aligned_bag_seed_aware"] = seed_aware_interval(
        runs_contrast(table, core, ("ORDERLESS_BAG", "H1_ALIGNED_ORDERLESS_BAG"),
                      ("DIRECT_HORIZON_UPPER_BOUND", "H1_PATIENT_ALIGNED")),
        "same contrast, resampling patients and both arms' training seeds")

    # ---- E1 free ordered state vs the unordered baseline -------------------
    reference = baselines[(baselines["baseline_level"] == "U_FULL_SET")
                          & (baselines["prefix_len"] == 3)
                          & (baselines["family"] == "AUTONOMOUS_SHARED_OPERATOR")]
    free_runs = (table[(table["structure"] == "H1_FREE_LOW_RANK")
                       & (table["block"] == "CORE1")
                       & (table["family"] == "AUTONOMOUS_SHARED_OPERATOR")
                       & (table["rank"] == 4)
                       & (table["baseline_level"] == "U_FULL_SET")]
                 .groupby("patient")["primary_objective"].apply(list).to_dict())
    free = collapsed[(collapsed["structure"] == "H1_FREE_LOW_RANK")
                     & (collapsed["block"] == "CORE1")
                     & (collapsed["family"] == "AUTONOMOUS_SHARED_OPERATOR")
                     & (collapsed["rank"] == 4)].set_index("patient")["primary_objective"]
    shared = sorted(set(free.index) & set(reference["patient"]))
    reference = reference.set_index("patient").loc[shared]
    ordered_gain = pd.DataFrame({
        "patient": shared,
        "h0_primary_objective": reference["h0_primary_objective"].to_numpy(),
        "free_primary_objective": free.loc[shared].to_numpy(),
        "effect": reference["h0_primary_objective"].to_numpy() - free.loc[shared].to_numpy(),
    })
    evidence["E1_free_low_rank_minus_unordered_baseline"] = cohort_statistics(
        ordered_gain["effect"].to_numpy(),
        "gain of the free low-dimensional ordered prefix branch over the selected frozen "
        "U_FULL_SET unordered baseline (autonomous family).  This is NOT an ordered-history "
        "gain: no permutation-invariant model was fit on the free basis, so the ordered and "
        "the learned-basis contributions are not separated here.  The capacity-matched "
        "ordered-vs-unordered test that does exist is "
        "E1_aligned_ordered_minus_aligned_bag, on the aligned dictionary.")
    evidence["E1_free_low_rank_minus_unordered_baseline_seed_aware"] = seed_aware_interval(
        {row.patient: [row.h0_primary_objective - value
                       for value in free_runs.get(row.patient, [])]
         for row in ordered_gain.itertuples()},
        "same contrast, resampling patients and the free arm's training seed")

    # ---- E4 bypass interaction --------------------------------------------
    bypass_rows = []
    for level in ("U_MINIMAL", "U_FULL_SET"):
        block = "CORE2" if level == "U_MINIMAL" else "CORE1"
        frame = contrast(collapsed,
                         {**core, "block": block, "baseline_level": level,
                          "family": "AUTONOMOUS_SHARED_OPERATOR",
                          "structure": "H1_ANGLE_ROTATED_AXIS"},
                         {**core, "block": block, "baseline_level": level,
                          "family": "AUTONOMOUS_SHARED_OPERATOR",
                          "structure": "H1_PATIENT_ALIGNED"})
        frame["baseline_level"] = level
        bypass_rows.append(frame)
        evidence[f"E4_delta_structure_{level}"] = cohort_statistics(
            frame["effect"].to_numpy(), f"angle null minus aligned under {level}")
    bypass = bypass_rows[0].merge(bypass_rows[1], on="patient", suffixes=("_minimal", "_full"))
    bypass["bypass_interaction"] = bypass["effect_minimal"] - bypass["effect_full"]
    tables["PER_PATIENT_BYPASS_INTERACTION"] = bypass
    evidence["E4_bypass_interaction"] = cohort_statistics(
        bypass["bypass_interaction"].to_numpy(),
        "structure advantage under the weak bypass minus under the strong bypass")

    # ---- E5 capacity and the two learning curves ---------------------------
    capacity = []
    for rank in (1, 2, 4, 8):
        block = "CORE1" if rank == 4 else "CAPACITY"
        frame = contrast(collapsed,
                         {**core, "block": block, "rank": rank,
                          "family": "AUTONOMOUS_SHARED_OPERATOR",
                          "structure": "H1_ANGLE_ROTATED_AXIS"},
                         {**core, "block": block, "rank": rank,
                          "family": "AUTONOMOUS_SHARED_OPERATOR",
                          "structure": "H1_PATIENT_ALIGNED"})
        frame["rank"] = rank
        capacity.append(frame)
        evidence[f"E5_capacity_rank{rank}"] = cohort_statistics(
            frame["effect"].to_numpy(), f"angle null minus aligned at rank {rank}")
    tables["PER_PATIENT_CAPACITY_CURVE"] = pd.concat(capacity, ignore_index=True)

    for name, basis_rule in (("END_TO_END", "fraction"), ("FIXED_BASIS", 100)):
        curve = []
        for fraction in (25, 50, 100):
            block = "CORE1" if fraction == 100 else "LEARNING"
            basis_fraction = 100 if basis_rule == 100 or fraction == 100 else fraction
            frame = contrast(collapsed,
                             {**core, "block": block, "data_fraction": fraction,
                              "basis_fraction": basis_fraction,
                              "family": "AUTONOMOUS_SHARED_OPERATOR",
                              "structure": "H1_ANGLE_ROTATED_AXIS"},
                             {**core, "block": block, "data_fraction": fraction,
                              "basis_fraction": basis_fraction,
                              "family": "AUTONOMOUS_SHARED_OPERATOR",
                              "structure": "H1_PATIENT_ALIGNED"})
            frame["data_fraction"], frame["basis_fraction"] = fraction, basis_fraction
            curve.append(frame)
            evidence[f"E5_{name.lower()}_fraction{fraction}"] = cohort_statistics(
                frame["effect"].to_numpy(), f"angle null minus aligned, {name} at {fraction}% data")
        tables[f"PER_PATIENT_{name}_DATA_CURVE"] = pd.concat(curve, ignore_index=True)

    # ---- E6 use phase -------------------------------------------------------
    use_path = RESULT_ROOT / "PER_PATIENT_ORDER_AND_PATH_ABLATION.csv"
    if use_path.exists():
        use = pd.read_csv(use_path)
        core_use = use[(use["block"] == "CORE1") & (use["baseline_level"] == "U_FULL_SET")
                       & (use["rank"] == 4)]
        for structure in ("H1_PATIENT_ALIGNED", "H1_ANGLE_ROTATED_AXIS", "H1_FREE_LOW_RANK"):
            frame = core_use[(core_use["structure"] == structure)
                             & (core_use["family"] == "AUTONOMOUS_SHARED_OPERATOR")]
            grouped = frame.groupby("patient")[
                ["prefix_order_cost_suffix_balanced_bce",
                 "ordered_path_ablation_cost_suffix_balanced_bce"]].median()
            evidence[f"E6_prefix_order_cost_{structure}"] = cohort_statistics(
                grouped["prefix_order_cost_suffix_balanced_bce"].to_numpy(),
                f"cost of reordering the observed prefix ({structure})")
            for column, key in (("prefix_order_cost_suffix_balanced_bce",
                                 f"E6_prefix_order_cost_{structure}_seed_aware"),
                                ("ordered_path_ablation_cost_suffix_balanced_bce",
                                 f"E6_ordered_path_ablation_cost_{structure}_seed_aware")):
                evidence[key] = seed_aware_interval(
                    frame.dropna(subset=[column]).groupby("patient")[column]
                    .apply(list).to_dict(),
                    f"{column} ({structure}), patients and seeds resampled together")
            evidence[f"E6_ordered_path_ablation_cost_{structure}"] = cohort_statistics(
                grouped["ordered_path_ablation_cost_suffix_balanced_bce"].to_numpy(),
                f"cost of zeroing the low-dimensional state ({structure})")

    transplant_path = RESULT_ROOT / "PER_PATIENT_BASIS_TRANSPLANT.csv"
    if transplant_path.exists():
        transplant = pd.read_csv(transplant_path)
        for column in ("delta_test_given_A", "delta_test_given_N", "transplant_interaction"):
            evidence[f"E6_basis_transplant_{column}"] = cohort_statistics(
                transplant[column].to_numpy(), f"basis transplant {column}")

    # ---- E0 representation ceiling ----------------------------------------
    ceiling_path = RESULT_ROOT / "PER_PATIENT_BASIS_CEILING.csv"
    if ceiling_path.exists():
        ceiling = pd.read_csv(ceiling_path)
        base = ceiling[(ceiling["field_kind"] == "suffix5")
                       & (ceiling["baseline_level"] == "U_FULL_SET")]
        # A rank-4 basis spans almost everything when only a handful of candidate
        # contacts remain, so the informative subset is reported as the primary
        # ceiling and the full set alongside it.
        for tag, focus in (("informative", base[base["ceiling_informative"]]), ("all", base)):
            wide = focus.groupby(["patient", "basis"])["relative_projection_error"].median().unstack()
            if "PATIENT_ALIGNED" not in wide:
                continue
            for other in [c for c in wide.columns if c != "PATIENT_ALIGNED"]:
                both = wide[["PATIENT_ALIGNED", other]].dropna()
                matched = other in ("ANGLE_ROTATED_AXIS", "IDENTITY_PERMUTED", "LOCALITY_REWIRED")
                entry = cohort_statistics(
                    (both[other] - both["PATIENT_ALIGNED"]).to_numpy(),
                    f"{other} minus aligned projection error, {tag} patients "
                    f"(positive = aligned spans more)")
                entry["matched_to_aligned"] = matched
                entry["note"] = (
                    "exactly matched: same kernel, same anisotropy strength, same "
                    "constant+shaft projection" if matched else
                    "not matched: geometry / shaft / free-PCA keep the constant field and the "
                    "shaft indicators that the aligned family has projected out, so this "
                    "comparison is conservative against the aligned basis")
                evidence[f"E0_ceiling_{tag}_{other}_minus_aligned"] = entry

    # ---- E7: which behavioural quantity actually improved -------------------
    for metric, label in (("total_nll_h1", "next rank set"), ("total_nll_h2", "two steps ahead"),
                          ("total_nll_h3", "three steps ahead"), ("total_nll_h4", "four steps ahead"),
                          ("total_nll_h5", "five steps ahead"),
                          ("suffix_balanced_bce", "accumulated five-step field"),
                          ("endpoint_distance_mm", "late-field endpoint distance")):
        frame = contrast(collapsed,
                         {**core, "family": "AUTONOMOUS_SHARED_OPERATOR",
                          "structure": "H1_ANGLE_ROTATED_AXIS"},
                         {**core, "family": "AUTONOMOUS_SHARED_OPERATOR",
                          "structure": "H1_PATIENT_ALIGNED"},
                         metric=metric)
        evidence[f"E7_endpoint_{metric}"] = cohort_statistics(
            frame["effect"].to_numpy(), f"angle null minus aligned on the {label}")
    time_frame = contrast(collapsed,
                          {**core, "block": "TIME_PROXY", "time_head": True,
                           "family": "AUTONOMOUS_SHARED_OPERATOR",
                           "structure": "H1_ANGLE_ROTATED_AXIS"},
                          {**core, "block": "TIME_PROXY", "time_head": True,
                           "family": "AUTONOMOUS_SHARED_OPERATOR",
                           "structure": "H1_PATIENT_ALIGNED"},
                          metric="time_proxy_loss")
    evidence["E7_spectral_centroid_latency_proxy"] = cohort_statistics(
        time_frame["effect"].to_numpy(),
        "angle null minus aligned on the spectral-centroid latency proxy — a timing "
        "variable, never a conduction delay or a speed")

    # ---- E8: coverage heterogeneity, described and never causal -------------
    coverage_path = RESULT_ROOT / "PER_PATIENT_COVERAGE_DESCRIPTORS.csv"
    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path).set_index("patient")
        primary = contrast(collapsed,
                           {**core, "family": "AUTONOMOUS_SHARED_OPERATOR",
                            "structure": "H1_ANGLE_ROTATED_AXIS"},
                           {**core, "family": "AUTONOMOUS_SHARED_OPERATOR",
                            "structure": "H1_PATIENT_ALIGNED"}).set_index("patient")
        shared = [p for p in primary.index if p in coverage.index]
        descriptors = {}
        for column in ("n_contacts", "n_shafts", "geometry_effective_dimension",
                       "ratio_second_to_first", "recorded_SOZ_annotation_fraction",
                       "n_recording_blocks"):
            if column not in coverage:
                continue
            pair = pd.DataFrame({"effect": primary.loc[shared, "effect"],
                                 "descriptor": coverage.loc[shared, column]}).dropna()
            if len(pair) >= 6:
                rho, pvalue = stats.spearmanr(pair["descriptor"], pair["effect"])
                descriptors[column] = {"n": int(len(pair)), "spearman_rho": float(rho),
                                       "spearman_p": float(pvalue)}
        evidence["E8_coverage_descriptor_associations"] = {
            "label": "exploratory association between coverage descriptors and the structure "
                     "effect; descriptive only, never causal and never used to exclude a patient",
            "n": len(shared), "associations": descriptors,
        }

    # ---- E7 endpoint matrix ------------------------------------------------
    endpoint = collapsed[(collapsed["block"].isin(["CORE1", "CORE2"]))]
    tables["PER_PATIENT_ENDPOINT_MATRIX"] = endpoint[
        GROUP_KEYS + ["total_nll_h1", "total_nll_h2", "total_nll_h3", "total_nll_h4",
                      "total_nll_h5", "suffix_balanced_bce", "suffix_balanced_brier",
                      "endpoint_distance_mm", "time_proxy_loss", "ordered_path_ablation_cost",
                      "ordered_parameter_count", "transition_spectral_radius", "n_nulls"]
    ]

    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    ordered_gain.to_csv(out / "PER_PATIENT_FREE_VS_UNORDERED_BASELINE.csv", index=False)
    bag.to_csv(out / "PER_PATIENT_ORDERED_VS_ORDERLESS_BAG.csv", index=False)

    status = {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_run_status",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "units_planned": int(len(manifest)),
        "units_eligible": int(len(eligible)),
        "units_complete": states["complete"],
        "units_unresolved": states["unresolved"],
        "units_missing": states["missing"],
        "units_with_nonfinite_batches": int((table["nonfinite_batches"] > 0).sum()),
        "total_wall_seconds": float(table["wall_seconds"].sum()),
        "per_block_complete": table.groupby("block").size().to_dict(),
    }
    (out / "RUN_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    (out / "COHORT_EVIDENCE_MATRIX.json").write_text(json.dumps({
        "contract": "topic5_capacity_constrained_history_motif_v0_2_cohort_evidence",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "aggregation_order": "horizon -> event -> seed/null -> patient -> cohort",
        "effect_sign_convention": "positive = patient-aligned structure has the lower loss",
        "near_zero_threshold": NEAR_ZERO,
        "primary_objective": "L_space on split 2: mean total NLL over h=1,2,3 plus the "
                             "event-balanced suffix BCE of the family's own spatial field",
        "seed_noise_floor": {
            "note": "within-arm spread of the split-2 objective across seeds; every "
                    "structure effect below must be read against this",
            "n_multi_seed_arms": int(len(spread)),
            "median_seed_spread": float(spread["seed_spread"].median()) if len(spread) else None,
            "p90_seed_spread": float(spread["seed_spread"].quantile(0.9)) if len(spread) else None,
            "paired_effect_note": "the marginal spread above overstates the uncertainty of a "
                                  "paired contrast; the paired figures below recompute the "
                                  "central effect once per aligned seed against the same null "
                                  "median and are the floor the primary claim must clear",
            "n_patients_with_multiple_aligned_seeds": int(len(paired)),
            "median_paired_effect_seed_spread": float(paired["effect_seed_spread"].median())
            if len(paired) else None,
            "p90_paired_effect_seed_spread": float(paired["effect_seed_spread"].quantile(0.9))
            if len(paired) else None,
            "median_paired_effect": float(paired["effect_median"].median()) if len(paired) else None,
        },
        "layers": evidence,
    }, indent=2) + "\n")

    print(f"units complete {states['complete']} / eligible {len(eligible)} "
          f"(unresolved {states['unresolved']}, missing {states['missing']})")
    if len(spread):
        print(f"seed noise floor (marginal): median {spread['seed_spread'].median():.5f}, "
              f"p90 {spread['seed_spread'].quantile(0.9):.5f} over {len(spread)} multi-seed arms")
    if len(paired):
        print(f"seed noise floor (paired effect): median {paired['effect_seed_spread'].median():.5f}, "
              f"p90 {paired['effect_seed_spread'].quantile(0.9):.5f} over {len(paired)} patients; "
              f"median effect {paired['effect_median'].median():+.5f}")
    for key in sorted(evidence):
        entry = evidence[key]
        # the coverage layer is a nested descriptor dict rather than a cohort
        # statistic, so it has no median to print
        if entry.get("n", 0) == 0 or "median" not in entry:
            continue
        # the seed-aware entries carry their own interval key and no sign counts
        interval = entry.get("median_ci95") or entry.get("median_ci95_seed_aware")
        counts = (f"+/-/0={entry['n_positive']}/{entry['n_negative']}/{entry['n_near_zero']}"
                  if "n_positive" in entry else "seed-aware")
        print(f"  {key:58s} n={entry['n']:2d} median={entry['median']:+.4f} "
              f"CI=[{interval[0]:+.4f},{interval[1]:+.4f}] {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
