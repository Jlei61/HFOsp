#!/usr/bin/env python3
"""Analysis for initial-state v1: primary paired effect, quality, figures, report.

--stage screen | replication : per-stage tables, statistics, quality, figures
--stage final                : compose the round-level deliverables and status
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import topic4_initial_state_runtime as rt  # noqa: E402
from src.topic4_initial_state import (  # noqa: E402
    ARMS, block_matched_reference, compare_input_digests, holm_adjust,
    missing_pair_bounds, off_diagonal_terms, paired_effect, per_block_reference,
    screen_verdict,
)
from src.topic4_observation_repaired import order_error  # noqa: E402
from scripts.review_topic4_same_network_events import GROUPS, group_times  # noqa: E402

ARM_COLOR = {"B0": "#7f7f7f", "B1": "#d95f02", "B2": "#1b9e77"}
ARM_LABEL = {"B0": "B0 common reset", "B1": "B1 core A +1 mV", "B2": "B2 core B +1 mV"}
MODE_COLOR = {0: "#1f77b4", 1: "#d62728"}
WINDOW_NAMES = ["0.5-6 s", "6-12 s", "12-18 s", "18-24 s", "0.5-24 s"]


# ------------------------------------------------------------------ loading
def load_runs(design, stage, source="formal"):
    out = rt.output_root(design)
    if source == "qualification":
        # exercise the analysis path on the short I0 units (never a scientific sample)
        jobs = [{"stage": stage, "stem": p.stem, "output_json": str(p), "output_npz": str(p.with_suffix(".npz"))}
                for p in sorted((out / "qualification").glob("qual_*.json"))]
    else:
        jobs = [j for j in rt.read(out / "jobs.json")["jobs"] if j["stage"] == stage]
    runs = []
    for job in jobs:
        path = Path(job["output_json"])
        if not path.exists():
            continue
        record = rt.read(path)
        if record.get("status") != "INITIAL_STATE_WORKER_COMPLETE":
            continue
        if bool(record.get("qualification_tag")) != (source == "qualification"):
            continue
        if rt.sha(record["arrays"]["path"]) != record["arrays"]["sha256"]:
            raise RuntimeError(f"arrays changed for {path.name}")
        arrays = np.load(record["arrays"]["path"])
        runs.append({"job": job, "record": record, "arrays": arrays})
    return runs


def primary_table(run):
    z = run["arrays"]
    centroid = np.asarray(z["centroid_ms"], float)
    primary = np.asarray(z["primary_event_indices"], int)
    return centroid, primary, np.asarray(z["event_mode"], int), np.asarray(z["event_support_state"], int)


# ------------------------------------------------------------------ run summary
def run_summary_rows(runs, design):
    rows = []
    late = design["statistics"]["primary_window_ms"]
    for run in runs:
        rec, z = run["record"], run["arrays"]
        centroid, primary, mode, state = primary_table(run)
        names = z["contact_names"].astype(str).tolist()
        scl = [i for i, n in enumerate(names) if n.startswith("SCL")]
        times = np.asarray(z["event_time_ms"], float)
        actual = float(rec["simulation"]["actual_duration_ms"])
        burnin = float(rec["observation"]["burnin_ms"])
        exposure_s = max(0.0, (actual - burnin) / 1000.0)
        primary_times = np.sort(times[primary]) if len(primary) else np.zeros(0)
        durations = [rec["events"][i]["qualifying_interval_ms"][1] - rec["events"][i]["qualifying_interval_ms"][0]
                     for i in primary]
        classified = [i for i in primary if mode[i] >= 0]
        row = {
            "stage": rec["stage"], "topology_seed": rec["topology_seed"], "dynamics_seed": rec["dynamics_seed"],
            "arm": rec["arm"], "initial_state_id": rec["initial_state_id"],
            "physical_status": rec["physical_status"], "actual_duration_ms": actual,
            "runaway_early_stop_ms": rec["simulation"]["runaway_early_stop_ms"],
            "n_detected_windows": rec["observation"]["n_detected_windows"],
            "n_primary_events": len(primary), "n_primary_classified": len(classified),
            "n_primary_unclassifiable": int(sum(mode[i] < 0 for i in primary)),
            "n_excluded_windows": int(rec["observation"]["n_detected_windows"] - len(primary)),
            "n_excluded_overlap": int(sum("overlapping_window" in e["primary_exclusion_reasons"] for e in rec["events"])),
            "n_excluded_prolonged": int(sum("prolonged_activity" in e["primary_exclusion_reasons"] for e in rec["events"])),
            "n_excluded_insufficient_centroids": int(sum("insufficient_estimable_centroids" in e["primary_exclusion_reasons"] for e in rec["events"])),
            "n_censored_boundary": len(rec["observation"]["boundary_or_low_window_support"]),
            "primary_event_rate_per_s": (len(primary) / exposure_s) if exposure_s > 0 else None,
            "median_inter_event_interval_ms": float(np.median(np.diff(primary_times))) if len(primary_times) > 1 else None,
            "median_qualifying_duration_ms": float(np.median(durations)) if durations else None,
            "scl_participation_fraction": (float(np.mean([np.isfinite(centroid[i, scl]).any() for i in primary]))
                                           if len(primary) else None),
            "supported_fraction": float(np.mean(state[classified] == 1)) if classified else None,
            "unsupported_fraction": float(np.mean(state[classified] == -1)) if classified else None,
            "median_distance_assigned_mode": (float(np.median([z["event_distance_modes"][i, mode[i]] for i in classified]))
                                              if classified else None),
            "input_digest_n_segments": len(rec["external_input_digest"]["segments"]),
            "simulation_wall_seconds": rec["simulation"]["simulation_wall_seconds"],
            "wall_seconds": rec["simulation"]["wall_seconds"], "peak_rss_gib": rec.get("peak_rss_gib"),
        }
        for name, counts in rec["window_counts"].items():
            row[f"{name}_M0"] = counts["mode_counts"][0]
            row[f"{name}_M1"] = counts["mode_counts"][1]
            row[f"{name}_unclassifiable"] = counts["n_unclassifiable"]
            row[f"{name}_mode0_proportion"] = counts["mode0_proportion"]
        row["late_window_ms"] = late
        rows.append(row)
    return rows


def all_event_rows(runs):
    rows = []
    for run in runs:
        rec, z = run["record"], run["arrays"]
        names = z["contact_names"].astype(str).tolist()
        centroid = np.asarray(z["centroid_ms"], float)
        for event in rec["events"]:
            i = event["detected_index"]
            row = {"stage": rec["stage"], "topology_seed": rec["topology_seed"],
                   "dynamics_seed": rec["dynamics_seed"], "arm": rec["arm"], **event}
            row["primary_exclusion_reasons"] = "|".join(event["primary_exclusion_reasons"])
            row["window_start_ms"], row["window_end_ms"] = event["window_ms"]
            row["qualifying_start_ms"], row["qualifying_end_ms"] = event["qualifying_interval_ms"]
            del row["window_ms"], row["qualifying_interval_ms"]
            for k, name in enumerate(names):
                row[f"centroid_{name}_ms"] = centroid[i, k] if np.isfinite(centroid[i, k]) else ""
            row.update({f"route_{k}": v for k, v in group_times(centroid[i], names).items()})
            rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


# ------------------------------------------------------------------ statistics
def by_seed_arm(runs):
    table = {}
    for run in runs:
        rec = run["record"]
        table.setdefault(int(rec["dynamics_seed"]), {})[rec["arm"]] = run
    return table


def contrast(table, seeds, window_key, left, right, stats, *, seed_offset):
    diffs, estimable, rows = [], [], []
    for index, seed in enumerate(seeds):
        arms = table.get(seed, {})
        pl = arms[left]["record"]["window_counts"][window_key]["mode0_proportion"] if left in arms else None
        pr = arms[right]["record"]["window_counts"][window_key]["mode0_proportion"] if right in arms else None
        ok = pl is not None and pr is not None
        rows.append({"dynamics_seed": seed, f"pi_{left}": pl, f"pi_{right}": pr,
                     "difference": (pl - pr) if ok else None, "estimable": ok,
                     "missing_arms": [a for a in (left, right) if a not in arms],
                     "counts_left": arms[left]["record"]["window_counts"][window_key]["mode_counts"] if left in arms else None,
                     "counts_right": arms[right]["record"]["window_counts"][window_key]["mode_counts"] if right in arms else None})
        if ok:
            diffs.append(pl - pr)
            estimable.append(seed)
    result = {"contrast": f"{left}-{right}", "window_key": window_key, "pairs": rows,
              "n_estimable": len(diffs), "n_design": len(seeds)}
    if len(diffs) >= 2:
        result["effect"] = paired_effect(diffs, n_resamples=int(stats["bootstrap_resamples"]),
                                         seed=int(stats["analysis_seed"]) + seed_offset, alpha=float(stats["alpha"]))
    else:
        result["effect"] = None
    if len(diffs) < len(seeds):
        result["status"] = "NOT_ESTIMABLE"
        result["observed_mean_of_estimable_pairs"] = float(np.mean(diffs)) if diffs else None
        result["full_design_mean_bounds_missing_at_plus_minus_one"] = (
            missing_pair_bounds(diffs, len(seeds) - len(diffs), n_total=len(seeds),
                                bounds=tuple(stats["missing_pair_effect_bounds"])))
    else:
        result["status"] = "ESTIMABLE"
    return result


def primary_effect(runs, design, stage):
    stats = design["statistics"]
    seeds = list(design["stages"][stage]["dynamics_seeds"])
    table = by_seed_arm(runs)
    primary = contrast(table, seeds, "primary", "B1", "B2", stats, seed_offset=0)
    windows = {f"secondary_{i}": name for i, name in enumerate(WINDOW_NAMES)}
    secondary = {}
    for key, name in windows.items():
        secondary[key] = {"window_name": name, "window_ms": stats["secondary_windows_ms"][int(key[-1])],
                          "B1-B2": contrast(table, seeds, key, "B1", "B2", stats, seed_offset=10 + int(key[-1])),
                          "B1-B0": contrast(table, seeds, key, "B1", "B0", stats, seed_offset=20 + int(key[-1])),
                          "B2-B0": contrast(table, seeds, key, "B2", "B0", stats, seed_offset=30 + int(key[-1]))}
    baseline_late = {"B1-B0": contrast(table, seeds, "primary", "B1", "B0", stats, seed_offset=40),
                     "B2-B0": contrast(table, seeds, "primary", "B2", "B0", stats, seed_offset=41)}
    raw = [c["effect"]["exchange"]["p_two_sided"] if c["effect"] else None for c in baseline_late.values()]
    if all(p is not None for p in raw):
        adjusted = holm_adjust(raw)
        for (key, c), p in zip(baseline_late.items(), adjusted):
            c["holm_adjusted_p"] = p
    early = secondary["secondary_0"]["B1-B2"]["effect"] if secondary["secondary_0"]["B1-B2"]["status"] == "ESTIMABLE" else None
    verdict = screen_verdict(primary["effect"] if primary["status"] == "ESTIMABLE" else None,
                             n_total=len(seeds), threshold=float(stats["priority_effect_threshold_fraction"]),
                             alpha=float(stats["alpha"]), early_effect=early)
    late = primary["effect"]
    return {
        "stage": stage, "topology_seed": design["stages"][stage]["topology_seed"],
        "n_runs_loaded": len(runs), "n_runs_design": 3 * len(seeds),
        "primary_window_ms": stats["primary_window_ms"], "primary_mode": stats["primary_mode"],
        "unit": stats["unit"], "event_weighting": stats["event_weighting"],
        "primary_contrast": primary,
        "delta_percentage_points": (100.0 * late["mean"]) if late else None,
        "ci95_percentage_points": ([100.0 * v for v in late["bootstrap"]["ci"]] if late else None),
        "exchange_p": late["exchange"]["p_two_sided"] if late else None,
        "unstable": late["instability"]["unstable"] if late else None,
        "verdict": verdict,
        "gate_rules": stats["screen_gate"],
        "secondary_windows": secondary,
        "baseline_contrasts_late_window_holm": baseline_late,
        "interpretation_bounds": {
            "PERSISTENT_INITIAL_STATE_EFFECT_SINGLE_GRAPH": "opens I2 on a second graph; not a patient bistability claim",
            "SMALL_EFFECT_BOUNDED_FOR_TESTED_PROBE": "effects above 10 points unsupported for this operating point, 1 mV probe and window",
            "INCONCLUSIVE": "interval and event counts delivered; no seed or window changes",
            "STARTUP_SENSITIVITY_ONLY": "difference confined to the start-up window; not persistent state selection",
            "NOT_ESTIMABLE": "at least one pair lacks a late-window proportion; bounds reported",
        },
        "created_unix": time.time(),
    }


# ------------------------------------------------------------------ crosscheck
def crosscheck(runs, design, stage):
    table = by_seed_arm(runs)
    per_seed = {}
    for seed, arms in table.items():
        row = {"arms_present": sorted(arms)}
        if "B0" in arms:
            base = arms["B0"]["record"]["external_input_digest"]["segments"]
            for arm in ("B1", "B2"):
                if arm in arms:
                    cmp = compare_input_digests(base, arms[arm]["record"]["external_input_digest"]["segments"])
                    row[f"digest_B0_vs_{arm}"] = {k: v for k, v in cmp.items() if k != "rows"}
                    sums_equal = np.array_equal(arms["B0"]["arrays"]["ext_segment_sums"][:cmp["n_complete_segments_compared"]],
                                                arms[arm]["arrays"]["ext_segment_sums"][:cmp["n_complete_segments_compared"]])
                    row[f"digest_B0_vs_{arm}"]["exact_poisson_sums_equal_on_common_segments"] = bool(sums_equal)
        identities = [arms[a]["record"]["static_array_identity"] for a in arms]
        row["static_identity_equal_across_arms"] = all(i == identities[0] for i in identities)
        row["initial_voltage_sha256"] = {a: arms[a]["record"]["initial_voltage"]["sha256"] for a in arms}
        row["physical_status"] = {a: arms[a]["record"]["physical_status"] for a in arms}
        per_seed[int(seed)] = row
    identities = [r["record"]["static_array_identity"] for r in runs]
    historical = None
    if stage == "screen":
        hist = rt.read(Path(design["sources"]["baseline_worker_arrays"]["path"]).with_suffix(".json"))["static_array_identity"]
        keys = ("positions_E_sha256", "h_sha256", "delta_vtheta_sha256", "vtheta_sha256",
                "ampa_topology_sha256", "ampa_values_sha256", "gaba_topology_sha256", "gaba_values_sha256")
        historical = bool(identities) and all(identities[0][k] == hist[k] for k in keys)
    all_equal = bool(identities) and all(i == identities[0] for i in identities)
    comparisons = [row.get(f"digest_B0_vs_{arm}", {}).get("all_complete_segments_equal", False)
                   for row in per_seed.values() for arm in ("B1", "B2") if arm in row["arms_present"]]
    all_digests = (bool(all(comparisons)) if comparisons else None)
    return {"stage": stage, "per_seed": per_seed,
            "static_identity_equal_across_all_runs": all_equal,
            "static_identity_matches_historical_mainline_worker": historical,
            "n_digest_comparisons": len(comparisons),
            "all_external_input_digests_equal_across_arms": all_digests,
            "prefix_only_seeds": [s for s, r in per_seed.items() if any(
                r.get(f"digest_B0_vs_{a}", {}).get("prefix_only") for a in ("B1", "B2"))]}


# ------------------------------------------------------------------ conditional quality
def pair_medians(table):
    """Median signed centroid difference t_j - t_i per contact pair over events with both finite."""
    n = table.shape[1]
    med = np.full((n, n), np.nan)
    count = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            d = table[:, j] - table[:, i]
            d = d[np.isfinite(d)]
            count[i, j] = count[j, i] = len(d)
            if len(d):
                med[i, j] = np.median(d)
                med[j, i] = -med[i, j]
    return med, count


def route_summary(records):
    summary = {"n_events": len(records), "no_SCL": int(sum(r["upper_SCL_n"] == 0 for r in records))}
    for key in ("middle_before_both_ICL_ends", "right_before_upper_and_left", "upper_and_left_before_right"):
        values = [r[key] for r in records if r[key] is not None]
        summary[key] = {"n_matching": int(sum(values)), "n_estimable": len(values),
                        "fraction": float(np.mean(values)) if values else None}
    return summary


def conditional_quality(runs, design, stage, evaluator, objective):
    stats = design["conditional_quality"]
    late = design["statistics"]["primary_window_ms"]
    names = runs[0]["arrays"]["contact_names"].astype(str).tolist() if runs else list(evaluator.groups)
    fit = evaluator.fit
    fit_labels = np.asarray(evaluator.fit_labels)
    fit_blocks = np.asarray(evaluator.blocks[evaluator.index["FIT"]])
    phi_fit = objective.embedding(fit)
    reference = {}
    for mode in range(evaluator.k):
        take = fit_labels == mode
        target = phi_fit[take].mean(axis=0)
        med, count = pair_medians(fit[take])
        reference[mode] = {
            "n_events": int(take.sum()), "target": target,
            "participation": np.isfinite(fit[take]).mean(axis=0),
            "pair_median": med, "pair_count": count,
            "routes": route_summary([group_times(row, names) for row in fit[take]]),
            "per_block": per_block_reference(phi_fit[take], fit_blocks[take], target,
                                             minimum_n=int(stats["minimum_N_for_conditional_D_off"])),
            "blocks": fit_blocks[take], "phi": phi_fit[take],
        }
    per_run = []
    pooled = {arm: {mode: [] for mode in range(evaluator.k)} for arm in ARMS}
    pooled_late = {arm: {mode: [] for mode in range(evaluator.k)} for arm in ARMS}
    for run in runs:
        rec, z = run["record"], run["arrays"]
        centroid, primary, mode_all, state = primary_table(run)
        phi = np.asarray(z["event_phi"], float)
        times = np.asarray(z["event_time_ms"], float)
        distance = np.asarray(z["event_distance_modes"], float)
        for mode in range(evaluator.k):
            idx = np.array([i for i in primary if mode_all[i] == mode], int)
            idx_late = np.array([i for i in idx if late[0] <= times[i] < late[1]], int)
            row = {"stage": stage, "dynamics_seed": rec["dynamics_seed"], "arm": rec["arm"], "mode": mode,
                   "n_events": int(len(idx)), "n_events_late": int(len(idx_late)),
                   "n_supported": int(np.sum(state[idx] == 1)) if len(idx) else 0,
                   "n_indeterminate": int(np.sum(state[idx] == 0)) if len(idx) else 0,
                   "n_unsupported_OOD": int(np.sum(state[idx] == -1)) if len(idx) else 0,
                   "median_distance_to_this_mode": float(np.median(distance[idx, mode])) if len(idx) else None,
                   "median_distance_to_other_mode": float(np.median(distance[idx, 1 - mode])) if len(idx) else None,
                   "median_relative_distance": (float(np.median(distance[idx, mode] / distance[idx, 1 - mode]))
                                                if len(idx) else None)}
            terms = off_diagonal_terms(phi[idx], reference[mode]["target"]) if len(idx) >= 2 else off_diagonal_terms(np.zeros((0, phi.shape[1])), reference[mode]["target"])
            row.update({f"D_off_{k}": v for k, v in terms.items()})
            terms_late = off_diagonal_terms(phi[idx_late], reference[mode]["target"]) if len(idx_late) >= 2 else {"status": "NOT_ESTIMABLE", "D_off": None}
            row["D_off_late_status"], row["D_off_late"] = terms_late["status"], terms_late["D_off"]
            if len(idx):
                sub = centroid[idx]
                row["participation"] = np.isfinite(sub).mean(axis=0).tolist()
                first = np.nanmin(sub, axis=1)[:, None]
                rel = sub - first
                row["relative_centroid_sd_per_contact_ms"] = np.nanstd(rel, axis=0).tolist()
                row["routes"] = route_summary([group_times(r, names) for r in sub])
                pooled[rec["arm"]][mode].append(sub)
                if len(idx_late):
                    pooled_late[rec["arm"]][mode].append(centroid[idx_late])
            per_run.append(row)
    # arm x mode aggregates (equal run weight; run-level bootstrap only, Q6)
    rng = np.random.default_rng(int(stats["seed"]))
    aggregate = {}
    for arm in ARMS:
        for mode in range(evaluator.k):
            rows = [r for r in per_run if r["arm"] == arm and r["mode"] == mode]
            d_values = np.array([r["D_off_D_off"] for r in rows if r["D_off_status"] == "ESTIMABLE"], float)
            n_values = np.array([r["n_events"] for r in rows], int)
            entry = {"n_runs": len(rows), "n_runs_estimable_D_off": int(len(d_values)),
                     "events_per_run": n_values.tolist(),
                     "median_events_per_run": float(np.median(n_values)) if len(n_values) else None,
                     "supported_fraction_pooled": (float(sum(r["n_supported"] for r in rows) / max(1, sum(r["n_events"] for r in rows)))
                                                   if rows else None),
                     "unsupported_fraction_pooled": (float(sum(r["n_unsupported_OOD"] for r in rows) / max(1, sum(r["n_events"] for r in rows)))
                                                     if rows else None),
                     "D_off_run_values": d_values.tolist(),
                     "D_off_run_mean": float(d_values.mean()) if len(d_values) else None,
                     "D_off_run_median": float(np.median(d_values)) if len(d_values) else None,
                     "A_run_mean": float(np.mean([r["D_off_A_mean_target_squared_distance"] for r in rows if r["D_off_status"] == "ESTIMABLE"])) if len(d_values) else None,
                     "B_run_mean": float(np.mean([r["D_off_B_finite_event_subtraction"] for r in rows if r["D_off_status"] == "ESTIMABLE"])) if len(d_values) else None}
            if len(d_values) >= 2:
                boot = d_values[rng.integers(0, len(d_values), size=(int(stats["bootstrap_resamples"]), len(d_values)))].mean(axis=1)
                entry["D_off_run_bootstrap_ci95"] = np.quantile(boot, [0.025, 0.975]).tolist()
            else:
                entry["D_off_run_bootstrap_ci95"] = None
            if pooled[arm][mode]:
                table = np.vstack(pooled[arm][mode])
                med, count = pair_medians(table)
                entry["pooled_n_events"] = int(len(table))
                entry["pooled_participation"] = np.isfinite(table).mean(axis=0).tolist()
                entry["participation_residual_vs_FIT"] = (np.isfinite(table).mean(axis=0) - reference[mode]["participation"]).tolist()
                entry["participation_mae_vs_FIT"] = float(np.abs(np.isfinite(table).mean(axis=0) - reference[mode]["participation"]).mean())
                ref_med = reference[mode]["pair_median"]
                valid = np.isfinite(med) & np.isfinite(ref_med) & (count >= 5) & (reference[mode]["pair_count"] >= 5)
                iu = np.triu_indices(len(names), 1)
                mask = valid[iu]
                entry["pair_signed_median_model_ms"] = med.tolist()
                entry["pair_signed_median_residual_mae_ms"] = float(np.abs(med[iu][mask] - ref_med[iu][mask]).mean()) if mask.any() else None
                entry["pair_signed_median_residual_n_pairs"] = int(mask.sum())
                order_tv = []
                sub_ref = fit[fit_labels == mode]
                for i, j in zip(*iu):
                    d1 = table[:, j] - table[:, i]; d1 = d1[np.isfinite(d1)]
                    d2 = sub_ref[:, j] - sub_ref[:, i]; d2 = d2[np.isfinite(d2)]
                    if len(d1) >= 5 and len(d2) >= 5:
                        order_tv.append(order_error(d1, d2, 2.0))
                entry["order_TV_at_2ms_mean"] = float(np.mean(order_tv)) if order_tv else None
                first = np.nanmin(table, axis=1)[:, None]
                entry["relative_centroid_sd_per_contact_ms"] = np.nanstd(table - first, axis=0).tolist()
                entry["routes"] = route_summary([group_times(r, names) for r in table])
                n_ref = int(np.median(n_values[n_values > 0])) if (n_values > 0).any() else 0
                entry["patient_block_matched_reference"] = block_matched_reference(
                    reference[mode]["phi"], reference[mode]["blocks"], reference[mode]["target"],
                    n_model=max(2, n_ref), n_resamples=int(stats["bootstrap_resamples"]),
                    seed=int(stats["seed"]) + 100 * ARMS.index(arm) + mode)
            else:
                entry["pooled_n_events"] = 0
            aggregate[f"{arm}_M{mode}"] = entry
    # mainline L_off bypass (Q4): saved, never used for any decision
    bypass = []
    for run in runs:
        centroid, primary, mode_all, state = primary_table(run)
        classified = [i for i in primary if mode_all[i] >= 0]
        score = objective.score_network(centroid[classified]) if len(classified) else {"status": "INSUFFICIENT_EVENTS"}
        bypass.append({"dynamics_seed": run["record"]["dynamics_seed"], "arm": run["record"]["arm"],
                       "status": score.get("status"), "loss_off": score.get("loss_off"),
                       "mode_counts": score.get("mode_counts")})
    fit_reference = {}
    for mode in range(evaluator.k):
        estimable = [r for r in reference[mode]["per_block"] if r["status"] == "ESTIMABLE"]
        fit_reference[f"M{mode}"] = {
            "n_events": reference[mode]["n_events"],
            "participation": reference[mode]["participation"].tolist(),
            "pair_signed_median_ms": reference[mode]["pair_median"].tolist(),
            "routes": reference[mode]["routes"],
            "per_block_D_off": [{k: v for k, v in r.items()} for r in reference[mode]["per_block"]],
            "per_block_D_off_median": float(np.median([r["D_off"] for r in estimable])) if estimable else None,
            "n_blocks_estimable": len(estimable), "n_blocks": len(reference[mode]["per_block"]),
        }
    return {"stage": stage, "contact_names": names, "classifier": stats["classifier"],
            "use_all_classified_events_including_unsupported": True,
            "no_reclustering_or_relabeling": True, "evaluator_metrics_called": False,
            "patient_time_packet_opened": False,
            "definition": {"D_off": "||mean phi - FIT mode mean phi||^2 - mean||phi_i - mean||^2/(N-1) on the frozen v2.1 joint kernel map; negatives retained; N<2 NOT_ESTIMABLE",
                           "participation": "fraction of events with a finite centroid per contact",
                           "pair_signed_median": "median over events of centroid(j)-centroid(i) where both finite",
                           "routes": "posthoc group-centroid descriptions from the same-network review (development diagnostics only)"},
            "per_run_mode": per_run, "aggregate_arm_mode": aggregate,
            "patient_FIT_reference": fit_reference,
            "mainline_L_off_bypass_not_used_for_any_decision": bypass}


# ------------------------------------------------------------------ figures
def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def fig_late_pairs(effect, path, stage_label):
    pairs = effect["primary_contrast"]["pairs"]
    seeds = [p["dynamics_seed"] for p in pairs]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax = axes[0]
    for k, p in enumerate(pairs):
        if p["pi_B1"] is not None and p["pi_B2"] is not None:
            ax.plot([0, 1], [p["pi_B1"], p["pi_B2"]], color="0.6", lw=0.8, zorder=1)
        if p["pi_B1"] is not None:
            ax.scatter(0, p["pi_B1"], color=ARM_COLOR["B1"], s=28, zorder=2)
        if p["pi_B2"] is not None:
            ax.scatter(1, p["pi_B2"], color=ARM_COLOR["B2"], s=28, zorder=2)
    b0 = [c["B1-B0"] for c in [effect["baseline_contrasts_late_window_holm"]]][0]["pairs"]
    for p in b0:
        if p.get("pi_B0") is not None:
            ax.scatter(0.5, p["pi_B0"], color=ARM_COLOR["B0"], marker="_", s=120, zorder=1)
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["B1\ncore A +1 mV", "B0\nreset", "B2\ncore B +1 mV"])
    ax.set_ylabel("Mode-0 proportion, 12-24 s")
    ax.set_ylim(-0.02, 1.02); ax.set_xlim(-0.25, 1.25)
    ax.set_title("Paired late-window proportions per noise seed", fontsize=9)
    _style(ax)
    ax = axes[1]
    d = [p["difference"] for p in pairs]
    x = np.arange(len(pairs))
    ax.axhspan(-0.10, 0.10, color="0.92", zorder=0)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.scatter(x, [np.nan if v is None else v for v in d], color="0.2", s=26, zorder=3)
    missing = [k for k, v in enumerate(d) if v is None]
    if missing:
        ax.scatter(missing, [0] * len(missing), marker="x", color="#b2182b", s=40, zorder=4, label="not estimable")
    if effect["primary_contrast"]["effect"]:
        e = effect["primary_contrast"]["effect"]
        ax.errorbar([len(pairs) + 0.8], [e["mean"]], yerr=[[e["mean"] - e["bootstrap"]["ci"][0]], [e["bootstrap"]["ci"][1] - e["mean"]]],
                    fmt="D", color="#762a83", capsize=3, ms=6, zorder=5, label="mean, 95% bootstrap CI")
    ax.set_xticks(list(x) + [len(pairs) + 0.8]); ax.set_xticklabels([str(s)[-2:] for s in seeds] + ["mean"], fontsize=7)
    ax.set_xlabel("Noise seed (last two digits)")
    ax.set_ylabel("Mode-0 proportion difference, B1 - B2")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.set_title("Per-pair difference; grey band = ±10 points", fontsize=9)
    _style(ax)
    fig.suptitle(f"Initial-state probe, {stage_label}: late-window mode proportion", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200); fig.savefig(Path(path).with_suffix(".pdf")); plt.close(fig)


def fig_windows(rows, path, stage_label):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), gridspec_kw={"width_ratios": [1.3, 1.3, 0.9]})
    keys = [f"secondary_{i}" for i in range(5)]
    for ai, arm in enumerate(ARMS):
        sub = [r for r in rows if r["arm"] == arm]
        for wi, key in enumerate(keys):
            props = [r[f"{key}_mode0_proportion"] for r in sub if r[f"{key}_mode0_proportion"] is not None]
            counts = [r[f"{key}_M0"] + r[f"{key}_M1"] for r in sub]
            xj = wi + (ai - 1) * 0.22
            axes[0].scatter(np.full(len(props), xj) + np.random.default_rng(ai + wi).uniform(-0.05, 0.05, len(props)),
                            props, color=ARM_COLOR[arm], s=14, alpha=0.7)
            if props:
                axes[0].plot([xj - 0.09, xj + 0.09], [np.median(props)] * 2, color=ARM_COLOR[arm], lw=2)
            axes[1].scatter(np.full(len(counts), xj) + np.random.default_rng(ai + wi + 7).uniform(-0.05, 0.05, len(counts)),
                            counts, color=ARM_COLOR[arm], s=14, alpha=0.7)
            if counts:
                axes[1].plot([xj - 0.09, xj + 0.09], [np.median(counts)] * 2, color=ARM_COLOR[arm], lw=2)
    for ax, ylabel, title in ((axes[0], "Mode-0 proportion (run)", "Mode proportion by window"),
                              (axes[1], "Classified primary events (run)", "Event yield by window")):
        ax.set_xticks(range(5)); ax.set_xticklabels(WINDOW_NAMES, fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9); _style(ax)
    axes[0].set_ylim(-0.02, 1.02)
    ax = axes[2]
    cats = [("n_excluded_overlap", "overlap"), ("n_excluded_prolonged", "prolonged"),
            ("n_excluded_insufficient_centroids", "few centroids"), ("n_primary_unclassifiable", "unclassifiable")]
    bottoms = np.zeros(3)
    hatches = ["", "//", "..", "xx"]
    for (key, label), hatch in zip(cats, hatches):
        vals = np.array([sum(r[key] for r in rows if r["arm"] == arm) for arm in ARMS], float)
        ax.bar(range(3), vals, bottom=bottoms, color=[ARM_COLOR[a] for a in ARMS], hatch=hatch, edgecolor="white", label=label)
        bottoms += vals
    runaway = [sum(r["physical_status"] != "COMPLETE_NO_RUNAWAY_BY_EXISTING_GATE" for r in rows if r["arm"] == arm) for arm in ARMS]
    for k, n in enumerate(runaway):
        ax.text(k, bottoms[k] + 0.5, f"early stop: {n}", ha="center", fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels(ARMS)
    ax.set_ylabel("Excluded windows, all runs")
    ax.set_title("Exclusions and early termination", fontsize=9)
    ax.set_ylim(0, max(1.0, float(bottoms.max())) * 1.35)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="0.7", hatch=h, edgecolor="white") for h in hatches]
    ax.legend(handles, [c[1] for c in cats], fontsize=7, frameon=False, loc="upper right", ncol=2)
    _style(ax)
    handles = [plt.Line2D([], [], color=ARM_COLOR[a], marker="o", ls="", label=ARM_LABEL[a]) for a in ARMS]
    axes[0].legend(handles=handles, fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(f"Initial-state probe, {stage_label}: mode proportions, yield and exclusions by time window", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200); fig.savefig(Path(path).with_suffix(".pdf")); plt.close(fig)


def fig_quality(quality, path, stage_label):
    names = quality["contact_names"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8), gridspec_kw={"width_ratios": [1.3, 1.0, 1.0]})
    for mode in range(2):
        ref = quality["patient_FIT_reference"][f"M{mode}"]
        ax = axes[mode, 0]
        ax.plot(range(len(names)), ref["participation"], color="k", lw=1.5, marker="s", ms=3, label="patient FIT")
        for arm in ARMS:
            entry = quality["aggregate_arm_mode"][f"{arm}_M{mode}"]
            if entry.get("pooled_n_events", 0):
                ax.plot(range(len(names)), entry["pooled_participation"], color=ARM_COLOR[arm], lw=1.2,
                        marker="o", ms=3, label=f"{ARM_LABEL[arm]} (n={entry['pooled_n_events']})")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_ylim(0, 1.02); ax.set_ylabel("Participation fraction")
        ax.set_title(f"Mode {mode}: per-contact participation", fontsize=9)
        ax.legend(fontsize=6.5, frameon=False, loc="lower left"); _style(ax)
        ax = axes[mode, 1]
        ref_med = np.asarray(ref["pair_signed_median_ms"], float)
        iu = np.triu_indices(len(names), 1)
        lim = 0.0
        for arm in ARMS:
            entry = quality["aggregate_arm_mode"][f"{arm}_M{mode}"]
            if entry.get("pooled_n_events", 0):
                med = np.asarray(entry["pair_signed_median_model_ms"], float)
                x, y = ref_med[iu], med[iu]
                ok = np.isfinite(x) & np.isfinite(y)
                ax.scatter(x[ok], y[ok], s=9, alpha=0.6, color=ARM_COLOR[arm],
                           label=f"{arm} MAE {entry['pair_signed_median_residual_mae_ms']:.1f} ms" if entry.get("pair_signed_median_residual_mae_ms") is not None else arm)
                lim = max(lim, np.nanmax(np.abs(np.r_[x[ok], y[ok]])) if ok.any() else 0)
        lim = max(lim, 5.0) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color="0.5", lw=0.8)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("Patient FIT pair median lag (ms)"); ax.set_ylabel("Model pair median lag (ms)")
        ax.set_title(f"Mode {mode}: signed pairwise timing", fontsize=9)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6.5, frameon=False, loc="upper left")
        _style(ax)
        ax = axes[mode, 2]
        blocks = [r["D_off"] for r in ref["per_block_D_off"] if r["status"] == "ESTIMABLE"]
        ax.scatter(np.full(len(blocks), -1) + np.random.default_rng(mode).uniform(-0.12, 0.12, len(blocks)), blocks,
                   color="k", s=10, alpha=0.6, label="patient FIT blocks")
        for k, arm in enumerate(ARMS):
            entry = quality["aggregate_arm_mode"][f"{arm}_M{mode}"]
            vals = entry["D_off_run_values"]
            ax.scatter(np.full(len(vals), k) + np.random.default_rng(k + 3).uniform(-0.12, 0.12, len(vals)), vals,
                       color=ARM_COLOR[arm], s=14, alpha=0.8)
            ci = entry.get("D_off_run_bootstrap_ci95")
            if ci:
                ax.plot([k, k], ci, color=ARM_COLOR[arm], lw=2.2)
                ax.plot([k - 0.15, k + 0.15], [entry["D_off_run_mean"]] * 2, color=ARM_COLOR[arm], lw=2.2)
            pr = entry.get("patient_block_matched_reference")
            if pr and pr.get("status") == "ESTIMABLE":
                q = pr["D_off_quantiles"]
                ax.fill_between([k - 0.35, k + 0.35], q[0], q[2], color="0.85", zorder=0)
                ax.plot([k - 0.35, k + 0.35], [q[1]] * 2, color="0.5", lw=1, zorder=0)
        ax.axhline(0, color="0.3", lw=0.6)
        ax.set_xticks([-1, 0, 1, 2]); ax.set_xticklabels(["FIT\nblocks"] + list(ARMS))
        ax.set_ylabel("D_off to the mode mean")
        ax.set_title(f"Mode {mode}: run-level D_off; grey = matched-N patient band", fontsize=9)
        _style(ax)
    fig.suptitle(f"Initial-state probe, {stage_label}: propagation quality per mode (frozen classifier, all classified events)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200); fig.savefig(Path(path).with_suffix(".pdf")); plt.close(fig)


def fig_timeline(runs, seed, path, stage_label):
    table = by_seed_arm(runs).get(seed, {})
    fig, axes = plt.subplots(3, 1, figsize=(13, 7.2), sharex=True)
    for ax, arm in zip(axes, ARMS):
        if arm not in table:
            ax.text(0.5, 0.5, f"{arm}: not available", transform=ax.transAxes, ha="center"); continue
        rec, z = table[arm]["record"], table[arm]["arrays"]
        t = np.asarray(z["trace_time_ms"], float) / 1000.0
        kernel = np.ones(20) / 20.0
        smoothed = []
        for key, color, label in (("trace_coreA_mean_V", ARM_COLOR["B1"], "core A mean V"),
                                  ("trace_coreB_mean_V", ARM_COLOR["B2"], "core B mean V")):
            v = np.convolve(np.asarray(z[key], float), kernel, mode="same")
            smoothed.append(v)
            ax.plot(t, v, color=color, lw=0.7, label=label)
        # inhibitory dips during events reach far below reset in this current-based
        # model; clip the display so the between-event core state stays readable
        allv = np.concatenate(smoothed)
        ax.set_ylim(float(np.percentile(allv, 2)), float(np.max(allv)) + 1.0)
        ax2 = ax.twinx()
        ax2.plot(t, np.convolve(np.asarray(z["trace_E_spikes"], float), kernel, mode="same"), color="0.55", lw=0.5, alpha=0.7)
        ax2.set_ylabel("E spikes / ms", fontsize=8, color="0.4"); ax2.tick_params(labelsize=7)
        ax2.spines["top"].set_visible(False)
        times = np.asarray(z["event_time_ms"], float) / 1000.0
        modes = np.asarray(z["event_mode"], int)
        primary = set(int(i) for i in z["primary_event_indices"])
        for i, (tt, m) in enumerate(zip(times, modes)):
            color = MODE_COLOR.get(int(m), "0.5")
            if i in primary:
                ax.axvline(tt, color=color, lw=1.2, alpha=0.9)
            else:
                ax.axvline(tt, color=color, lw=0.6, alpha=0.35, ls=":")
        ax.axvspan(0, 0.5, color="0.9", zorder=0)
        ax.axvspan(12, 24, color="#f7f0e8", zorder=0)
        ax.set_ylabel("mean V (mV)", fontsize=8)
        ax.set_title(f"{ARM_LABEL[arm]}  |  seed {seed}  |  {rec['physical_status'].replace('_', ' ').lower()}", fontsize=8.5, loc="left")
        _style(ax)
    handles = [plt.Line2D([], [], color=ARM_COLOR["B1"], label="core A mean V"),
               plt.Line2D([], [], color=ARM_COLOR["B2"], label="core B mean V"),
               plt.Line2D([], [], color="0.55", label="E spikes / ms (right axis)"),
               plt.Line2D([], [], color=MODE_COLOR[0], lw=1.2, label="primary event, mode 0"),
               plt.Line2D([], [], color=MODE_COLOR[1], lw=1.2, label="primary event, mode 1"),
               plt.Line2D([], [], color="0.5", lw=0.6, ls=":", label="excluded window")]
    axes[0].legend(handles=handles, fontsize=7, frameon=False, ncol=3, loc="upper right")
    axes[-1].set_xlabel("Time (s); shaded 0-0.5 s burn-in, tinted 12-24 s primary window")
    axes[-1].set_xlim(0, 24)
    fig.suptitle(f"Initial-state probe, {stage_label}: core state and event labels over the whole run (1 ms trace, 20 ms smoothing)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200); fig.savefig(Path(path).with_suffix(".pdf")); plt.close(fig)


# ------------------------------------------------------------------ stage driver
def analyze_stage(design, stage, evaluator, objective, source="formal"):
    out = rt.output_root(design) / (stage if source == "formal" else "qualification/analysis_dry_run")
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    runs = load_runs(design, stage, source=source)
    if not runs:
        raise RuntimeError(f"no completed runs for stage {stage}")
    rows = run_summary_rows(runs, design)
    write_csv(out / "run_summary.csv", rows)
    write_csv(out / "all_events.csv", all_event_rows(runs))
    effect = primary_effect(runs, design, stage)
    rt.write(out / "primary_effect.json", effect)
    check = crosscheck(runs, design, stage)
    rt.write(out / "qualification_crosscheck.json", check)
    quality = conditional_quality(runs, design, stage, evaluator, objective)
    rt.write(out / "conditional_propagation.json", quality)
    label = f"graph {design['stages'][stage]['topology_seed']} ({stage})"
    fig_late_pairs(effect, figures / f"fig1_late_window_pairs_{stage}.png", label)
    fig_windows(rows, figures / f"fig2_window_proportions_{stage}.png", label)
    fig_quality(quality, figures / f"fig3_conditional_propagation_{stage}.png", label)
    seed = min(design["stages"][stage]["dynamics_seeds"])
    fig_timeline(runs, seed, figures / f"fig4_core_state_timeline_{stage}_seed{seed}.png", label)
    return effect, check, quality, rows


def compose_final(design):
    out = rt.output_root(design)
    stages = [s for s in ("screen", "replication") if (out / s / "primary_effect.json").exists()]
    effects = {s: rt.read(out / s / "primary_effect.json") for s in stages}
    checks = {s: rt.read(out / s / "qualification_crosscheck.json") for s in stages}
    qualities = {s: rt.read(out / s / "conditional_propagation.json") for s in stages}
    screen = effects["screen"]
    replication = effects.get("replication")
    verdicts = {s: e["verdict"] for s, e in effects.items()}
    if replication is None:
        if screen["verdict"] == "PERSISTENT_INITIAL_STATE_EFFECT_SINGLE_GRAPH":
            initial_state_effect = "PERSISTENT_ON_ONE_GRAPH_REPLICATION_PENDING"
        else:
            initial_state_effect = screen["verdict"]
    else:
        same_direction = (screen["delta_percentage_points"] is not None and replication["delta_percentage_points"] is not None
                          and np.sign(screen["delta_percentage_points"]) == np.sign(replication["delta_percentage_points"]))
        if replication["verdict"] == "PERSISTENT_INITIAL_STATE_EFFECT_SINGLE_GRAPH" and same_direction:
            initial_state_effect = "PERSISTENT_STATE_CONDITIONED_PREFERENCE_REPLICATED_ON_TWO_GRAPHS_LIMITED_WINDOW"
        elif replication["verdict"] == "PERSISTENT_INITIAL_STATE_EFFECT_SINGLE_GRAPH":
            initial_state_effect = "TOPOLOGY_DEPENDENT_DIRECTION_REVERSED_ON_SECOND_GRAPH"
        else:
            initial_state_effect = f"SINGLE_GRAPH_ONLY_REPLICATION_{replication['verdict']}"
    # conditional propagation conclusion: per stage, does the preferred/any mode
    # come closer to the patient FIT structure than the common reset does?
    quality_conclusion = {}
    for s, q in qualities.items():
        agg = q["aggregate_arm_mode"]
        summary = {}
        for mode in range(2):
            ref = q["patient_FIT_reference"][f"M{mode}"]
            for arm in ARMS:
                e = agg[f"{arm}_M{mode}"]
                summary[f"{arm}_M{mode}"] = {
                    "runs_estimable": e["n_runs_estimable_D_off"], "D_off_run_mean": e["D_off_run_mean"],
                    "D_off_run_ci95": e["D_off_run_bootstrap_ci95"],
                    "patient_matched_band": (e.get("patient_block_matched_reference") or {}).get("D_off_quantiles"),
                    "participation_mae_vs_FIT": e.get("participation_mae_vs_FIT"),
                    "pair_signed_median_residual_mae_ms": e.get("pair_signed_median_residual_mae_ms"),
                    "order_TV_at_2ms_mean": e.get("order_TV_at_2ms_mean"),
                    "supported_fraction_pooled": e.get("supported_fraction_pooled"),
                    "unsupported_fraction_pooled": e.get("unsupported_fraction_pooled"),
                    "patient_block_D_off_median": ref["per_block_D_off_median"],
                }
        inside = {}
        for key, v in summary.items():
            band = v["patient_matched_band"]
            inside[key] = (None if (band is None or v["D_off_run_mean"] is None)
                           else bool(band[0] <= v["D_off_run_mean"] <= band[2]))
        quality_conclusion[s] = {"arm_mode": summary, "run_mean_inside_patient_matched_band": inside}
    # patient-conditional propagation verdict (descriptive tiers only)
    def improved(s):
        q = quality_conclusion[s]["arm_mode"]
        rows = []
        for mode in range(2):
            b0 = q[f"B0_M{mode}"]
            for arm in ("B1", "B2"):
                a = q[f"{arm}_M{mode}"]
                if a["D_off_run_mean"] is None or b0["D_off_run_mean"] is None:
                    continue
                ci_a, ci_b = a["D_off_run_ci95"], b0["D_off_run_ci95"]
                separated = bool(ci_a and ci_b and (ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0]))
                rows.append({"arm": arm, "mode": mode,
                             "D_off_run_mean": a["D_off_run_mean"], "B0_D_off_run_mean": b0["D_off_run_mean"],
                             "D_off_lower_than_B0_descriptive": a["D_off_run_mean"] < b0["D_off_run_mean"],
                             "D_off_run_level_intervals_separated_from_B0": separated,
                             "participation_closer_than_B0_descriptive": (a["participation_mae_vs_FIT"] is not None and b0["participation_mae_vs_FIT"] is not None
                                                                          and a["participation_mae_vs_FIT"] < b0["participation_mae_vs_FIT"]),
                             "timing_closer_than_B0_descriptive": (a["pair_signed_median_residual_mae_ms"] is not None and b0["pair_signed_median_residual_mae_ms"] is not None
                                                                   and a["pair_signed_median_residual_mae_ms"] < b0["pair_signed_median_residual_mae_ms"]),
                             "inside_patient_band": quality_conclusion[s]["run_mean_inside_patient_matched_band"][f"{arm}_M{mode}"]})
        return rows
    propagation_rows = {s: improved(s) for s in stages}
    any_inside = any(r["inside_patient_band"] for rows in propagation_rows.values() for r in rows)
    any_separated_better = any(r["D_off_lower_than_B0_descriptive"] and r["D_off_run_level_intervals_separated_from_B0"]
                               and r["participation_closer_than_B0_descriptive"] and r["timing_closer_than_B0_descriptive"]
                               for rows in propagation_rows.values() for r in rows)
    # Tiers are descriptive: "closer" requires the run-level D_off intervals of the arm
    # and of B0 not to overlap AND both other readouts to move the same way; otherwise
    # small descriptive differences are reported but not named an improvement.
    if any_inside:
        patient_conditional = "SOME_ARM_MODE_RUN_MEAN_INSIDE_MATCHED_PATIENT_BAND_DESCRIPTIVE_ONLY"
    elif any_separated_better:
        patient_conditional = "CLOSER_THAN_COMMON_RESET_ON_ALL_THREE_READOUTS_BUT_OUTSIDE_PATIENT_BAND"
    else:
        patient_conditional = "STRUCTURE_GAP_UNRESOLVED_ALL_ARM_MODE_OUTSIDE_PATIENT_BAND_NO_SEPARATED_IMPROVEMENT_VS_B0"
    final = {
        "status": "INITIAL_STATE_ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW",
        "created_unix": time.time(),
        "initial_state_effect": initial_state_effect,
        "patient_conditional_propagation": patient_conditional,
        "stage_verdicts": verdicts,
        "screen": {k: screen[k] for k in ("delta_percentage_points", "ci95_percentage_points", "exchange_p", "unstable",
                                           "verdict", "n_runs_loaded", "n_runs_design")},
        "replication": (None if replication is None else {k: replication[k] for k in (
            "delta_percentage_points", "ci95_percentage_points", "exchange_p", "unstable", "verdict", "n_runs_loaded", "n_runs_design")}),
        "crosscheck": {s: {k: v for k, v in c.items() if k != "per_seed"} for s, c in checks.items()},
        "propagation_comparison_vs_B0": propagation_rows,
        "quality_conclusion": quality_conclusion,
        "boundaries": [
            "a significant late-window difference on one or two graphs is not evidence of patient bistability",
            "two hand-picked initial states mixed at patient proportions are not a state-access mechanism",
            "the 1 mV cold-start voltage probe covers a small part of state space; negatives do not exclude synaptic, inhibitory, adaptive or slow-variable states",
            "route diagnostics are development checks on FIT only and feed nothing back into the probe",
        ],
    }
    rt.write(out / "primary_effect.json", {"final": final, "screen": screen, "replication": replication})
    rt.write(out / "conditional_propagation.json", {"final": quality_conclusion, **{s: q for s, q in qualities.items()}})
    rows = []
    events = []
    for s in stages:
        with open(out / s / "run_summary.csv") as stream:
            rows.extend(csv.DictReader(stream))
        with open(out / s / "all_events.csv") as stream:
            events.extend(csv.DictReader(stream))
    write_csv(out / "run_summary.csv", rows)
    write_csv(out / "all_events.csv", events)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    for s in stages:
        for src in (out / s / "figures").glob("fig*"):
            shutil.copy2(src, figures / src.name)
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=rt.DESIGN_PATH)
    parser.add_argument("--stage", choices=("screen", "replication", "final"), required=True)
    parser.add_argument("--source", choices=("formal", "qualification"), default="formal",
                        help="qualification = dry-run on the short I0 units; never a scientific sample")
    args = parser.parse_args()
    design = rt.load_design(args.design)
    if args.stage in ("screen", "replication"):
        evaluator = rt.load_evaluator(design)
        objective = rt.load_objective(design)
        effect, check, quality, rows = analyze_stage(design, args.stage, evaluator, objective, source=args.source)
        print(json.dumps({"stage": args.stage, "verdict": effect["verdict"],
                          "delta_pp": effect["delta_percentage_points"], "ci_pp": effect["ci95_percentage_points"],
                          "p": effect["exchange_p"], "n_runs": len(rows),
                          "digests_equal": check["all_external_input_digests_equal_across_arms"]}, indent=1))
    else:
        final = compose_final(design)
        print(json.dumps({k: final[k] for k in ("status", "initial_state_effect", "patient_conditional_propagation", "stage_verdicts")}, indent=1))


if __name__ == "__main__":
    main()
