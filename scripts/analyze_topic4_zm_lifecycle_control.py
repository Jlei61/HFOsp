#!/usr/bin/env python3
"""Calibrate u_ref and adjudicate the finite-control dose response."""
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
from scripts import analyze_topic4_zm_lifecycle_m_panel as M  # noqa: E402


OUT = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint"
IN_ROOT = OUT / "seed1"


def control_response(
    fine_core,
    fine_all,
    uncontrolled_core,
    *,
    t0_ms,
    duration_ms,
    bin_ms=2.0,
):
    """Measure the pulse against its paired no-control continuation.

    A bursty source may fall by itself immediately after the registered pulse
    time.  Comparing the controlled response only with its own pre-pulse mean
    would therefore reward a natural burst offset.  The calibration quantity
    is the paired spike-count reduction over the actual pulse window, matching
    the fixed-perturbation contract.  The pre-pulse trace is retained as a
    deterministic future-noise/parity check and as a readable diagnostic.
    """
    core = np.asarray(fine_core, float)
    all_e = np.asarray(fine_all, float)
    uncontrolled = np.asarray(uncontrolled_core, float)
    if uncontrolled.shape != core.shape:
        raise ValueError("controlled and uncontrolled core traces must align")
    t0 = int(round(float(t0_ms) / bin_ms))
    t1 = int(round((float(t0_ms) + float(duration_ms)) / bin_ms))
    if not 0 < t0 < t1 <= core.size:
        raise ValueError("control window lies outside the paired traces")
    pre0 = max(0, t0 - int(round(500.0 / bin_ms)))
    post1 = min(core.size, t1 + int(round(200.0 / bin_ms)))
    baseline = float(np.mean(core[pre0:t0])) if t0 > pre0 else None
    controlled_count_proxy = float(np.sum(core[t0:t1]))
    uncontrolled_count_proxy = float(np.sum(uncontrolled[t0:t1]))
    drop = (
        None if uncontrolled_count_proxy <= 0.0 else
        float(1.0 - controlled_count_proxy / uncontrolled_count_proxy)
    )
    prefix_delta = np.abs(core[:t0] - uncontrolled[:t0])
    prefix_max_abs = float(np.max(prefix_delta)) if prefix_delta.size else 0.0
    prefix_identical = bool(prefix_max_abs <= 1e-12)
    silent = all_e[t0:post1] <= 0.0
    longest = 0
    current = 0
    for value in silent:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return {
        "pre_control_core_mean_hz": baseline,
        "controlled_pulse_window_core_mean_hz": float(np.mean(core[t0:t1])),
        "uncontrolled_pulse_window_core_mean_hz": float(
            np.mean(uncontrolled[t0:t1])
        ),
        "fractional_core_drop": drop,
        "fractional_core_drop_definition": (
            "1-controlled/uncontrolled paired core spike-count proxy in pulse window"
        ),
        "precontrol_pair_max_abs_hz": prefix_max_abs,
        "precontrol_pair_identical": prefix_identical,
        "longest_global_zero_rate_ms": float(longest * bin_ms),
        "calibration_target_met": bool(
            prefix_identical and drop is not None and 0.50 <= drop <= 0.70
            and longest * bin_ms <= 100.0
        ),
    }


def choose_u_ref(rows):
    valid = [
        row for row in rows
        if row["control_response"]["calibration_target_met"]
    ]
    if valid:
        chosen = min(valid, key=lambda row: row["control_uplift_mV"])
        return {"status": "calibrated", "u_ref_mV": chosen["control_uplift_mV"]}
    nonsilencing = [
        row for row in rows
        if row["control_response"]["fractional_core_drop"] is not None
        and row["control_response"]["longest_global_zero_rate_ms"] <= 100.0
    ]
    if not nonsilencing:
        return {"status": "uncalibrated_all_globally_silencing", "u_ref_mV": None}
    chosen = min(
        nonsilencing,
        key=lambda row: abs(row["control_response"]["fractional_core_drop"] - 0.60),
    )
    return {
        "status": "nearest_nonsilencing_outside_target",
        "u_ref_mV": chosen["control_uplift_mV"],
        "observed_fractional_drop": chosen["control_response"]["fractional_core_drop"],
    }


def paired_control_effect(row, *, minimum_advance_ms=1000.0):
    """Require a durable post-pulse exit that precedes the paired no-control exit.

    The selected source trajectory has the same checkpoint, future noise, fast
    mechanism, and M coordinate.  A late natural offset therefore cannot be
    relabelled as a control effect merely because it occurs after the pulse.
    """
    onset = row.get("onset_ms")
    offset = row.get("offset_ms")
    control_t0 = row.get("control_t0_ms")
    if onset is None:
        return {"status": "prevention_or_no_onset", "causal_control_exit_candidate": False}
    if offset is None:
        return {"status": "no_durable_offset", "causal_control_exit_candidate": False}
    if control_t0 is None or float(offset) < float(control_t0):
        return {"status": "offset_precedes_control", "causal_control_exit_candidate": False}
    base_onset = row.get("uncontrolled_onset_ms")
    base_offset = row.get("uncontrolled_offset_ms")
    if base_onset is None:
        return {"status": "uncontrolled_pair_missing", "causal_control_exit_candidate": False}
    if base_offset is None:
        return {
            "status": "offset_vs_censored_uncontrolled",
            "causal_control_exit_candidate": True,
            "duration_advance_ms": None,
        }
    controlled_duration = float(offset) - float(onset)
    uncontrolled_duration = float(base_offset) - float(base_onset)
    advance = uncontrolled_duration - controlled_duration
    return {
        "status": "offset_advanced" if advance >= minimum_advance_ms else "offset_not_advanced",
        "causal_control_exit_candidate": bool(advance >= minimum_advance_ms),
        "duration_advance_ms": float(advance),
    }


def _match(analysis, config, summary):
    if not M.row_matches_manifest(analysis, config):
        return False
    control = summary.get("finite_control") or {}
    return (
        M._close(summary.get("T_ms"), config["T_ms"])
        and control.get("target") == config.get("control_target")
        and M._close(control.get("t0_ms"), config.get("control_t0_ms"))
        and M._close(control.get("duration_ms"), config.get("control_duration_ms"))
        and M._close(control.get("uplift_mV"), config.get("control_uplift_mV"))
    )


def analyze_manifest(manifest):
    roots = [path for path in sorted(IN_ROOT.glob("*")) if (path / "summary.json").is_file()]
    analyses = {path.name: A.analyze_one(path) for path in roots}
    summaries = {path.name: json.loads((path / "summary.json").read_text()) for path in roots}
    rows = []
    for config in manifest["rows"]:
        hits = [stem for stem, analysis in analyses.items() if _match(analysis, config, summaries[stem])]
        if len(hits) > 1:
            raise RuntimeError(f"ambiguous control artifact for {config['config_id']}")
        if not hits:
            rows.append({**config, "status": "missing"}); continue
        stem = hits[0]
        analysis, summary = analyses[stem], summaries[stem]
        uncontrolled_summary = config.get("uncontrolled_summary_path")
        if uncontrolled_summary is None:
            rows.append({**config, "status": "uncontrolled_pair_missing"})
            continue
        uncontrolled_root = (ROOT / uncontrolled_summary).parent
        uncontrolled_npz = uncontrolled_root / "traces.npz"
        if not uncontrolled_npz.is_file():
            rows.append({**config, "status": "uncontrolled_pair_missing"})
            continue
        with np.load(IN_ROOT / stem / "traces.npz", allow_pickle=False) as data:
            with np.load(uncontrolled_npz, allow_pickle=False) as uncontrolled:
                response = control_response(
                    data["fine_core_rate_hz"], data["fine_all_e_rate_hz"],
                    uncontrolled["fine_core_rate_hz"],
                    t0_ms=config["control_t0_ms"],
                    duration_ms=config["control_duration_ms"],
                )
        episode = analysis["episode"]
        row = {
            **config, "status": "complete", "stem": stem,
            "phenotype": analysis["phenotype"],
            "onset_ms": episode.get("onset_ms"), "offset_ms": episode.get("offset_ms"),
            "durable_control_exit": bool(
                episode.get("offset_ms") is not None
                and episode["offset_ms"] >= config["control_t0_ms"]
            ),
            "exit_latency_from_control_ms": (
                None if episode.get("offset_ms") is None else
                float(episode["offset_ms"] - config["control_t0_ms"])
            ),
            "rapid_reentry_count": len(episode.get("rapid_reentry_bins", [])),
            "returning_event_candidate": analysis["recovery"].get("single_event_candidate", False),
            "returning_distribution_recovered": analysis["recovery"].get("distribution_recovered", False),
            "control_response": response,
            "summary_path": analysis["summary_path"],
        }
        effect = paired_control_effect(row)
        row["causal_control_effect"] = effect
        row["causal_control_exit_candidate"] = effect["causal_control_exit_candidate"]
        rows.append(row)
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("calibration", "dose"))
    args = ap.parse_args()
    manifest_path = OUT / f"control_{args.mode}_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = analyze_manifest(manifest)
    payload = {
        "schema": f"topic4_zm_lifecycle_control_{args.mode}_analysis_v1_2026-08-02",
        "semantic_scope": "seed1_finite_threshold_uplift_development_not_clinical_control",
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "n_expected": len(rows),
        "n_complete": sum(row["status"] == "complete" for row in rows),
        "rows": rows,
    }
    if args.mode == "calibration":
        decisions = []
        ranks = sorted({row["selection_rank"] for row in rows})
        for rank in ranks:
            subset = [row for row in rows if row["selection_rank"] == rank and row["status"] == "complete"]
            decisions.append({"selection_rank": rank, **choose_u_ref(subset)})
        payload["calibration_decisions"] = decisions
        # This self-pointer lets the dose manifest retain provenance.
        payload["source_path"] = str((OUT / "control_calibration_analysis.json").relative_to(ROOT))
    else:
        complete = [row for row in rows if row["status"] == "complete"]
        payload.update(
            n_durable_control_exits=sum(row["durable_control_exit"] for row in complete),
            n_causal_control_exit_candidates=sum(
                row["causal_control_exit_candidate"] for row in complete
            ),
            n_returning_event_candidates=sum(row["returning_event_candidate"] for row in complete),
            n_returning_distributions=sum(row["returning_distribution_recovered"] for row in complete),
        )
    path = OUT / f"control_{args.mode}_analysis.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    flat = []
    for row in rows:
        response = row.get("control_response", {})
        flat.append({
            key: row.get(key) for key in (
                "config_id", "selection_rank", "source_candidate_id", "status", "stem",
                "control_uplift_mV", "control_duration_ms", "phenotype", "onset_ms",
                "offset_ms", "durable_control_exit", "exit_latency_from_control_ms",
                "causal_control_exit_candidate",
                "rapid_reentry_count", "returning_event_candidate",
                "returning_distribution_recovered",
            )
        } | {
            "fractional_core_drop": response.get("fractional_core_drop"),
            "longest_global_zero_rate_ms": response.get("longest_global_zero_rate_ms"),
            "calibration_target_met": response.get("calibration_target_met"),
        })
    with (OUT / f"control_{args.mode}_analysis.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    print(json.dumps({key: value for key, value in payload.items() if key.startswith("n_")}, sort_keys=True))


if __name__ == "__main__":
    main()
