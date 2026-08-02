#!/usr/bin/env python3
"""Adjudicate the four-by-nine seed-1 M response surface.

This is a causal development surface, not an acceptance classifier.  A durable
offset at g_M>0 only counts as M-associated when the paired g_M=0 trajectory
does not offset, or offsets at least one second later.  Returning interictal
events remain a separate, stronger endpoint.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import analyze_topic4_zm_lifecycle_sprint as A  # noqa: E402
from scripts import analyze_topic4_zm_fast_lifecycle_development as FAST  # noqa: E402


OUT = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint"
IN_ROOT = OUT / "seed1"


def _close(left, right, atol=1e-7):
    return left is not None and right is not None and abs(float(left) - float(right)) <= atol


def row_matches_manifest(analysis, manifest_row):
    mechanism = analysis["mechanism"]
    dep = mechanism.get("i2e_depression", {})
    iadapt = mechanism.get("i_adaptation", {})
    slow = mechanism.get("dynamic_slow_flow", {})
    if mechanism.get("arm") != manifest_row["arm"]:
        return False
    checks = (
        (dep.get("tau_D_ms"), manifest_row.get("tau_D_ms")),
        (dep.get("d_star_nominal"), manifest_row.get("d_star")),
        (mechanism.get("strength_scale"), manifest_row.get("strength_scale")),
        (slow.get("g_M"), manifest_row.get("g_M")),
        (slow.get("tau_M_ms"), manifest_row.get("tau_M_ms")),
        (slow.get("g_Z"), manifest_row.get("g_Z")),
    )
    if manifest_row["arm"] == "combined":
        checks += (
            (iadapt.get("tau_aI_ms"), manifest_row.get("tau_aI_ms")),
            (iadapt.get("f_aI"), manifest_row.get("f_aI")),
        )
    return all(_close(left, right) for left, right in checks)


def episode_duration_ms(row, observed_ms):
    episode = row["episode"]
    onset = episode.get("onset_ms")
    if onset is None:
        return None, False
    offset = episode.get("offset_ms")
    if offset is None:
        return max(0.0, float(observed_ms) - float(onset)), True
    return max(0.0, float(offset) - float(onset)), False


def paired_m_effect(row, baseline, *, minimum_advance_ms=1000.0):
    """Describe whether M produces an earlier durable exit than g_M=0."""
    if row is None or baseline is None:
        return {"status": "pair_missing", "causal_exit_candidate": False}
    if row["episode"].get("onset_ms") is None:
        return {"status": "prevention_or_no_onset", "causal_exit_candidate": False}
    if row["episode"].get("offset_ms") is None:
        return {"status": "no_durable_offset", "causal_exit_candidate": False}
    row_duration = row["episode"]["offset_ms"] - row["episode"]["onset_ms"]
    base_offset = baseline["episode"].get("offset_ms")
    base_onset = baseline["episode"].get("onset_ms")
    if base_offset is None:
        return {
            "status": "offset_vs_censored_gM0",
            "causal_exit_candidate": True,
            "duration_advance_ms": None,
        }
    if base_onset is None:
        return {"status": "gM0_no_onset_invalid_pair", "causal_exit_candidate": False}
    base_duration = base_offset - base_onset
    advance = float(base_duration - row_duration)
    return {
        "status": "offset_advanced" if advance >= minimum_advance_ms else "offset_not_advanced",
        "causal_exit_candidate": bool(advance >= minimum_advance_ms),
        "duration_advance_ms": advance,
    }


def paired_m_continuous_response(row, baseline):
    """Quantify M-dependent state changes even when both episodes are censored."""
    if row is None or baseline is None:
        return {"status": "pair_missing"}
    keys = (
        "core_mean_hz", "all_E_mean_hz", "median_energy_gain_db",
        "energy_occupancy_6db", "post_entry_core_cv",
        "spatial_effective_rank", "common_mode_pc1_fraction",
    )
    out = {"status": "paired"}
    for key in keys:
        value = row.get(key)
        reference = baseline.get(key)
        out[f"delta_{key}"] = (
            None if value is None or reference is None else float(value) - float(reference)
        )
        if key in {"core_mean_hz", "all_E_mean_hz"}:
            out[f"ratio_{key}"] = (
                None if value is None or reference in (None, 0) else float(value) / float(reference)
            )
    row_trace = row.get("slow_trace", {})
    base_trace = baseline.get("slow_trace", {})
    for key in ("z_core_final", "z_core_minimum", "m_peak"):
        value = row_trace.get(key)
        reference = base_trace.get(key)
        out[f"delta_{key}"] = (
            None if value is None or reference is None else float(value) - float(reference)
        )
    row_tail = row.get("tail_state", {})
    base_tail = baseline.get("tail_state", {})
    for key in (
        "core_mean_hz", "all_E_mean_hz", "core_cv",
        "spatial_effective_rank", "common_mode_pc1_fraction",
    ):
        value = row_tail.get(key)
        reference = base_tail.get(key)
        out[f"delta_tail_{key}"] = (
            None if value is None or reference is None else float(value) - float(reference)
        )
        if key in {"core_mean_hz", "all_E_mean_hz"}:
            out[f"ratio_tail_{key}"] = (
                None if value is None or reference in (None, 0) else float(value) / float(reference)
            )
    return out


def _tail_state_metrics(root, *, tail_ms=3000.0):
    """Describe the terminal state separately from earlier branch transitions."""
    with np.load(root / "traces.npz", allow_pickle=False) as data:
        core = np.asarray(data["coarse_core_rate_hz"], float)
        all_e = np.asarray(data["coarse_all_e_rate_hz"], float)
        kymo = np.asarray(data["coarse_kymo_axial"], float)
    bins = min(core.size, max(2, int(round(float(tail_ms) / 25.0))))
    core_tail = core[-bins:]
    all_tail = all_e[-bins:]
    kymo_tail = kymo[:, -bins:]
    core_mean = float(np.mean(core_tail))
    core_cv = float(np.std(core_tail) / max(abs(core_mean), 1e-12))
    spatial = FAST._post_entry_spatial_metrics(kymo_tail, skip_ms=0.0)
    pc1 = spatial.get("common_mode_pc1_fraction")
    if core_mean < 2.0:
        label = "silent_tail"
    elif core_cv < 0.15:
        label = "tonic_tail"
    elif pc1 is not None and pc1 >= 0.90:
        label = "common_mode_bursty_tail"
    else:
        label = "modulated_or_mixed_tail"
    return {
        "tail_ms": float(bins * 25.0),
        "label": label,
        "core_mean_hz": core_mean,
        "all_E_mean_hz": float(np.mean(all_tail)),
        "core_cv": core_cv,
        **{key: spatial.get(key) for key in (
            "spatial_effective_rank", "common_mode_pc1_fraction",
            "centroid_excursion_bins", "centroid_median_speed_bins_s", "status",
        )},
    }


def _trace_metrics(root, episode):
    with np.load(root / "traces.npz", allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key], float) for key in (
            "trace_m_core_mean", "trace_z_core_mean", "trace_S_G",
            "trace_phi_core_mean", "trace_i2e_resource_mean",
            "trace_i_adaptation_mean",
        ) if key in data.files}
        fine_time = np.asarray(data["fine_time_ms"], float) if "fine_time_ms" in data.files else None
        fine_core = np.asarray(data["fine_core_rate_hz"], float) if "fine_core_rate_hz" in data.files else None
        fine_all = np.asarray(data["fine_all_e_rate_hz"], float) if "fine_all_e_rate_hz" in data.files else None
    trace = arrays.get("trace_m_core_mean", np.zeros(0))
    if trace.size == 0:
        return {"m_initial": None, "m_peak": None, "m_at_offset": None}
    offset_ms = episode.get("offset_ms")
    offset_idx = None if offset_ms is None else min(trace.size - 1, int(round(offset_ms)))
    result = {
        "m_initial": float(trace[0]),
        "m_peak": float(np.max(trace)),
        "m_at_offset": None if offset_idx is None else float(trace[offset_idx]),
        "m_final": float(trace[-1]),
    }
    onset_ms = episode.get("onset_ms")
    if onset_ms is not None and fine_time is not None:
        mask = fine_time >= float(onset_ms)
        result["post_onset_core_mean_hz"] = (
            None if fine_core is None or not np.any(mask) else float(np.mean(fine_core[mask]))
        )
        result["post_onset_all_E_mean_hz"] = (
            None if fine_all is None or not np.any(mask) else float(np.mean(fine_all[mask]))
        )
    for key, values in arrays.items():
        if key == "trace_m_core_mean" or values.size == 0:
            continue
        label = {
            "trace_z_core_mean": "z_core",
            "trace_S_G": "S_G",
            "trace_phi_core_mean": "phi_core",
            "trace_i2e_resource_mean": "i2e_resource",
            "trace_i_adaptation_mean": "i_adaptation",
        }[key]
        index = None if offset_idx is None else min(values.size - 1, offset_idx)
        result.update({
            f"{label}_initial": float(values[0]),
            f"{label}_minimum": float(np.min(values)),
            f"{label}_maximum": float(np.max(values)),
            f"{label}_at_offset": None if index is None else float(values[index]),
            f"{label}_final": float(values[-1]),
        })
    if offset_idx is not None and "trace_z_core_mean" in arrays:
        z = arrays["trace_z_core_mean"]
        index = min(z.size - 1, offset_idx)
        result["z_core_post_offset_recovery"] = float(z[-1] - z[index])
    else:
        result["z_core_post_offset_recovery"] = None
    return result


def build_surface(manifest, analyses, summaries):
    rows = []
    matched = {}
    for config in manifest["rows"]:
        candidates = [row for row in analyses if row_matches_manifest(row, config)]
        # The 12-s discovery run has the same parameters as native M.  The M
        # surface is explicitly 20-s, so join on the registered run horizon.
        candidates = [
            row for row in candidates
            if _close(summaries[row["stem"]].get("T_ms"), config["T_ms"])
        ]
        if len(candidates) > 1:
            raise RuntimeError(f"ambiguous M-panel artifact for {config['config_id']}")
        matched[config["config_id"]] = candidates[0] if candidates else None

    for rank in range(manifest["n_selected_fast_phenotypes"]):
        configs = [row for row in manifest["rows"] if row["selection_rank"] == rank]
        baseline_cfg = next(row for row in configs if float(row["g_M"]) == 0.0)
        baseline = matched[baseline_cfg["config_id"]]
        for config in configs:
            analysis = matched[config["config_id"]]
            if analysis is None:
                rows.append({
                    **config, "status": "missing", "stem": None,
                    "causal_exit_candidate": False,
                })
                continue
            summary = summaries[analysis["stem"]]
            duration, censored = episode_duration_ms(analysis, summary["observed_ms"])
            effect = paired_m_effect(analysis, baseline)
            recovery = analysis["recovery"]
            trace_root = IN_ROOT / analysis["stem"]
            slow_trace = _trace_metrics(trace_root, analysis["episode"])
            tail_state = _tail_state_metrics(trace_root)
            rows.append({
                **config,
                "status": "complete",
                "stem": analysis["stem"],
                "phenotype": analysis["phenotype"],
                "onset_ms": analysis["episode"].get("onset_ms"),
                "offset_ms": analysis["episode"].get("offset_ms"),
                "episode_duration_ms": duration,
                "duration_right_censored": censored,
                "n_transient_offsets": len(analysis["episode"].get("transient_offset_bins", [])),
                "n_rapid_reentries": len(analysis["episode"].get("rapid_reentry_bins", [])),
                "causal_M_effect": effect,
                "causal_exit_candidate": effect["causal_exit_candidate"],
                "returning_event_candidate": recovery.get("single_event_candidate", False),
                "returning_distribution_recovered": recovery.get("distribution_recovered", False),
                "median_energy_gain_db": analysis["intensity"].get("median_gain_db_across_contacts"),
                "energy_occupancy_6db": analysis["intensity"].get("occupancy_above_6db"),
                "post_entry_core_cv": analysis.get("post_entry_core_cv"),
                "core_mean_hz": slow_trace.get("post_onset_core_mean_hz"),
                "all_E_mean_hz": slow_trace.get("post_onset_all_E_mean_hz"),
                "spatial_effective_rank": analysis["within_episode_spatial"].get("spatial_effective_rank"),
                "common_mode_pc1_fraction": analysis["within_episode_spatial"].get("common_mode_pc1_fraction"),
                "slow_trace": slow_trace,
                "tail_state": tail_state,
                "summary_path": analysis["summary_path"],
            })
    for rank in range(manifest["n_selected_fast_phenotypes"]):
        rank_rows = [row for row in rows if row["selection_rank"] == rank]
        baseline = next(
            (row for row in rank_rows if row.get("status") == "complete" and float(row["g_M"]) == 0.0),
            None,
        )
        for row in rank_rows:
            row["paired_M_response"] = paired_m_continuous_response(
                row if row.get("status") == "complete" else None,
                baseline,
            )
    return rows


def _flat(row):
    effect = row.get("causal_M_effect", {})
    response = row.get("paired_M_response", {})
    trace = row.get("slow_trace", {})
    keys = (
        "config_id", "selection_rank", "source_fast_id", "arm", "g_M", "tau_M_ms",
        "status", "stem", "phenotype", "onset_ms", "offset_ms", "episode_duration_ms",
        "duration_right_censored", "causal_exit_candidate", "returning_event_candidate",
        "returning_distribution_recovered", "median_energy_gain_db", "energy_occupancy_6db",
        "post_entry_core_cv", "spatial_effective_rank", "common_mode_pc1_fraction",
        "core_mean_hz", "all_E_mean_hz",
    )
    out = {key: row.get(key) for key in keys}
    out.update(
        m_effect_status=effect.get("status"),
        duration_advance_ms=effect.get("duration_advance_ms"),
        m_peak=trace.get("m_peak"), m_at_offset=trace.get("m_at_offset"),
        z_core_minimum=trace.get("z_core_minimum"),
        z_core_final=trace.get("z_core_final"),
        S_G_maximum=trace.get("S_G_maximum"),
        delta_core_mean_hz=response.get("delta_core_mean_hz"),
        ratio_core_mean_hz=response.get("ratio_core_mean_hz"),
        delta_all_E_mean_hz=response.get("delta_all_E_mean_hz"),
        ratio_all_E_mean_hz=response.get("ratio_all_E_mean_hz"),
        delta_median_energy_gain_db=response.get("delta_median_energy_gain_db"),
        delta_energy_occupancy_6db=response.get("delta_energy_occupancy_6db"),
        delta_z_core_final=response.get("delta_z_core_final"),
        tail_label=row.get("tail_state", {}).get("label"),
        tail_core_mean_hz=row.get("tail_state", {}).get("core_mean_hz"),
        tail_core_cv=row.get("tail_state", {}).get("core_cv"),
        tail_spatial_effective_rank=row.get("tail_state", {}).get("spatial_effective_rank"),
        tail_common_mode_pc1_fraction=row.get("tail_state", {}).get("common_mode_pc1_fraction"),
        ratio_tail_core_mean_hz=response.get("ratio_tail_core_mean_hz"),
    )
    return out


def main():
    manifest_path = OUT / "m_panel_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    roots = [path for path in sorted(IN_ROOT.glob("*")) if (path / "summary.json").is_file()]
    analyses = [A.analyze_one(path) for path in roots]
    summaries = {path.name: json.loads((path / "summary.json").read_text()) for path in roots}
    rows = build_surface(manifest, analyses, summaries)
    complete = [row for row in rows if row["status"] == "complete"]
    payload = {
        "schema": "topic4_zm_lifecycle_m_response_surface_v1_2026-08-02",
        "semantic_scope": "seed1_checkpoint_fork_development_not_same_parameter_autonomous_lifecycle",
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "n_expected": len(rows), "n_complete": len(complete),
        "n_causal_exit_candidates": sum(row["causal_exit_candidate"] for row in complete),
        "n_returning_event_candidates": sum(row["returning_event_candidate"] for row in complete),
        "n_returning_distributions": sum(row["returning_distribution_recovered"] for row in complete),
        "rows": rows,
    }
    path = OUT / "m_response_surface.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    flat = [_flat(row) for row in rows]
    with (OUT / "m_response_surface.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    print(json.dumps({key: payload[key] for key in (
        "n_expected", "n_complete", "n_causal_exit_candidates",
        "n_returning_event_candidates", "n_returning_distributions",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
