#!/usr/bin/env python3
"""Aggregate the fixed LC6A five-arm block into one continuous phenotype map."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src/snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_topic4_fcxr_lc6a_natural_trajectory as NAT  # noqa: E402
from src.topic4_fcxr_lc5 import SparseSpikeStream  # noqa: E402
from src.topic4_fcxr_lc6_phenotype import (  # noqa: E402
    baseline_tradeoff, classify_high_state, event_metrics, spatial_slow_flow_readout,
)
from src.topic4_fcxr_lc6_trajectory import apply_local_classifier, spatial_rate_maps  # noqa: E402


OUT = NAT.OUT
LOCK = OUT / "local_classifier_manifest_addendum.json"
MAP = OUT / "phenotype_map.json"
SUMMARY = OUT / "trajectory_summary.json"
CSV = OUT / "trajectory_summary.csv"
DONE = OUT / "DONE_LC6A_PHENOTYPE_MAP.json"
FIGURES = OUT / "figures"
CONDITIONS = NAT.GRAPH_IDS


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, payload):
    NAT._write_json(path, payload)


def _load_stream(path):
    with np.load(path, allow_pickle=False) as z:
        stream = SparseSpikeStream(
            np.asarray(z["steps"], np.int64), np.asarray(z["cells"], np.int32),
            int(z["n_steps"][0]), int(z["n_cells"][0]),
        )
        expected = str(z["sha256"][0])
    if stream.sha256 != expected:
        raise RuntimeError(f"spike stream hash mismatch: {path}")
    return stream


def _global_rate(stream, *, window_ms):
    width = int(round(float(window_ms) / NAT.U2.DT_MS))
    if stream.n_steps % width:
        raise RuntimeError("trajectory length is not aligned to registered rate windows")
    bins = stream.steps // width
    counts = np.bincount(bins, minlength=stream.n_steps // width)
    return counts / stream.n_cells / (float(window_ms) / 1000.0)


def _load_condition(condition, lock):
    arm = OUT / f"trajectories/{condition}"
    summary_path = arm / "summary.json"
    spike_path = arm / "spikes.npz"
    trace_path = arm / "traces.npz"
    spatial_path = arm / "spatial_readouts.npz"
    for path in (summary_path, spike_path, trace_path, spatial_path):
        if not path.is_file():
            raise RuntimeError(f"LC6A five-arm block incomplete: {path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "COMPLETE" or summary.get("condition") != condition:
        raise RuntimeError(f"invalid natural-trajectory summary: {summary_path}")
    stream = _load_stream(spike_path)
    if stream.sha256 != summary["spike_sha256"]:
        raise RuntimeError(f"summary/spike hash mismatch: {condition}")
    with np.load(trace_path, allow_pickle=False) as z:
        traces = {key: np.asarray(z[key]) for key in z.files}
    with np.load(spatial_path, allow_pickle=False) as z:
        spatial = {key: np.asarray(z[key]) for key in z.files}
    maps100 = spatial_rate_maps(
        stream.steps, stream.cells, spatial["cell_bins"], spatial["occupancy"],
        n_steps=stream.n_steps, dt_ms=NAT.U2.DT_MS, window_ms=100.0,
    )
    local = apply_local_classifier(maps100, spatial["occupancy"], lock["thresholds"])
    rate100 = _global_rate(stream, window_ms=100.0)
    adjudication = NAT.PREFIX._adjudicate(stream, NAT._rate_from_stream(stream))
    onsets = [
        value for value in (summary.get("onset_ms"), local.get("local_onset_ms"))
        if value is not None
    ]
    effective_onset = min(onsets) if onsets else None
    baseline_end = min(
        float(summary["T_ms"]), float(lock["C0_global_onset_ms"]),
        float(effective_onset) if effective_onset is not None else np.inf,
    )
    metrics = event_metrics(adjudication["returned"], end_ms=baseline_end)
    spatial_flow = spatial_slow_flow_readout(
        spatial["rate_maps_1s"], spatial["D_maps_1s"], spatial["positions_E"],
        spatial["cell_bins"], spatial["occupancy"],
        axis_unit=spatial["patient_axis_unit"], source_xy=spatial["source_xy"],
        sheet_size_mm=float(spatial["sheet_size_mm"][0]),
        local_rate_threshold_hz=lock["thresholds"]["rate_threshold_hz"],
        onset_ms=effective_onset,
    )
    high = classify_high_state(
        global_onset_ms=summary.get("onset_ms"), local_onset_ms=local.get("local_onset_ms"),
        offset_ms=summary.get("offset_ms"), total_ms=summary["T_ms"],
        global_rate_100ms=rate100, d_trace=traces["D_mean"], h_trace=traces["H_mean"],
        trace_dt_ms=float(traces["rate_dt_ms"][0]),
        max_near_refractory_fraction=summary["local_saturation"][
            "max_near_refractory_fraction"
        ],
        right_censored=bool(summary["observation_terminal"].get("right_censored", False)),
    )
    if high["headline"] == "NO_ONSET":
        required = int(np.ceil(1.5 * lock["C0_n_returning_pre_onset"]))
        high["headline"] = (
            "ENTRY_BLOCKED_WITH_IED"
            if int(summary["n_returning_pre_onset"]) >= required
            else "ENTRY_UNRESOLVED_LOW_EXPOSURE"
        )
        high["required_IED_exposure"] = required
    post = np.array([], float)
    if effective_onset is not None:
        post = rate100[int(np.floor(float(effective_onset) / 100.0)):]
    envelope_cv = (
        float(np.std(post) / np.mean(post))
        if post.size and np.mean(post) > 0 else None
    )
    persistence = summary["spatial_map_persistence"].get("median_consecutive_correlation")
    centroid_rms = spatial_flow["centroid_rms_mm"]
    spatial_phenotype = "NOT_APPLICABLE_NO_ONSET"
    if effective_onset is not None:
        spatial_phenotype = (
            "STATIONARY" if persistence is not None and np.isfinite(persistence)
            and float(persistence) >= .95 and centroid_rms is not None
            and np.isfinite(centroid_rms) and float(centroid_rms) <= .5
            else "DYNAMIC"
        )
    return {
        "condition": condition, "graph_sha256": summary["graph_sha256"],
        "construction_q": summary["graph_construction_q"],
        "T_ms": summary["T_ms"], "global_onset_ms": summary.get("onset_ms"),
        "local_onset_ms": local.get("local_onset_ms"), "effective_onset_ms": effective_onset,
        "offset_ms": summary.get("offset_ms"), "entry_status": high["headline"] if effective_onset is None else "NATURAL_ENTRY",
        "headline": high["headline"], "boundedness": high,
        "local_classifier": local, "baseline_metrics": metrics,
        "baseline_tradeoff": None, "spatial_slow_flow": spatial_flow,
        "spatial_phenotype": spatial_phenotype, "envelope_cv_100ms": envelope_cv,
        "global_rate_100ms_peak_hz": float(np.max(rate100, initial=0.0)),
        "global_rate_100ms_late_hz": float(np.mean(rate100[-20:])),
        "local_rate_q95_peak_hz": float(np.nanmax(np.nanquantile(maps100[:, spatial["occupancy"] > 0], .95, axis=1))),
        "local_rate_q99_peak_hz": float(np.nanmax(np.nanquantile(maps100[:, spatial["occupancy"] > 0], .99, axis=1))),
        "max_near_refractory_fraction": summary["local_saturation"]["max_near_refractory_fraction"],
        "time_fraction_near_refractory_above_5pct": summary["local_saturation"]["time_fraction_above_fraction_gate"],
        "current_decomposition": summary["current_decomposition"],
        "pinned_checkpoints": summary["pinned_checkpoints"],
        "source_summary": str(summary_path), "source_summary_sha256": _sha(summary_path),
        "source_spikes_sha256": _sha(spike_path), "source_traces_sha256": _sha(trace_path),
        "source_spatial_sha256": _sha(spatial_path),
    }


def select_fork_candidates(rows, *, maximum=2):
    eligible = [
        row for row in rows
        if row["effective_onset_ms"] is not None
        and "onset_plus_2s" in row.get("pinned_checkpoints", {})
    ]
    if not eligible or maximum <= 0:
        return []
    ranked = sorted(
        eligible,
        key=lambda row: (
            -float(row["boundedness"].get("boundedness_margin", -np.inf)),
            row["condition"],
        ),
    )
    selected = [ranked[0]]
    if maximum > 1:
        alternatives = [
            row for row in ranked[1:]
            if row["headline"] != selected[0]["headline"]
            or row["spatial_phenotype"] != selected[0]["spatial_phenotype"]
        ]
        if not alternatives:
            alternatives = ranked[1:]
        if alternatives:
            selected.append(alternatives[0])
    return [{
        "condition": row["condition"], "headline": row["headline"],
        "spatial_phenotype": row["spatial_phenotype"],
        "boundedness_margin": row["boundedness"].get("boundedness_margin"),
        "available_checkpoints": sorted(row["pinned_checkpoints"]),
    } for row in selected]


def _write_csv(rows):
    fields = (
        "condition", "construction_q", "T_ms", "global_onset_ms", "local_onset_ms",
        "effective_onset_ms", "entry_status", "headline", "spatial_phenotype",
        "global_rate_100ms_peak_hz", "global_rate_100ms_late_hz",
        "local_rate_q95_peak_hz", "local_rate_q99_peak_hz",
        "max_near_refractory_fraction", "time_fraction_near_refractory_above_5pct",
        "envelope_cv_100ms",
    )
    tmp = CSV.with_name(CSV.name + f".{os.getpid()}.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    os.replace(tmp, CSV)


def _plot(rows):
    FIGURES.mkdir(parents=True, exist_ok=True)
    names = [row["condition"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    onset = [np.nan if row["effective_onset_ms"] is None else row["effective_onset_ms"] / 1000 for row in rows]
    ax.bar(x, onset, color="#3B7EA1")
    ax.set_xticks(x, names); ax.set_ylabel("Onset time (s)"); ax.set_title("a  Natural entry")
    ax = axes[0, 1]
    ax.plot(x, [row["global_rate_100ms_peak_hz"] for row in rows], "o-", label="global peak")
    ax.plot(x, [row["local_rate_q99_peak_hz"] for row in rows], "s-", label="local q99 peak")
    ax.axhline(250, color="0.4", linestyle="--", label="registered saturation")
    ax.set_xticks(x, names); ax.set_yscale("log"); ax.set_ylabel("Rate (Hz)")
    ax.set_title("b  Global and local activity"); ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 0]
    ax.plot(x, [row["spatial_slow_flow"]["max_D_halo_lead_mm"] for row in rows], "o-", label="D halo lead")
    ax.plot(x, [row["spatial_slow_flow"]["max_active_area_mm2"] for row in rows], "s-", label="active area")
    ax.set_xticks(x, names); ax.set_title("c  Spatial slow-flow readouts"); ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    margins = [row["boundedness"].get("boundedness_margin", np.nan) for row in rows]
    colors = ["#2CA25F" if row["boundedness"].get("bounded_candidate") else "#B2182B" for row in rows]
    ax.bar(x, margins, color=colors); ax.axhline(0, color="0.3", linewidth=1)
    ax.set_xticks(x, names); ax.set_ylabel("Minimum normalized margin")
    ax.set_title("d  Bounded-carrier margin")
    fig.suptitle("FCXR-LC6A patient-axis E→I reach: fixed five-arm phenotype map")
    png = FIGURES / "lc6a_trajectory_phenotypes.png"
    pdf = FIGURES / "lc6a_trajectory_phenotypes.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def run():
    if not LOCK.is_file():
        raise RuntimeError("C0 local-classifier lock must exist before five-arm aggregation")
    lock = json.loads(LOCK.read_text())
    if lock.get("status") != "LOCKED" or lock.get("selection_used_Q_trajectory_outcomes") is not False:
        raise RuntimeError("invalid local-classifier lock")
    rows = [_load_condition(condition, lock) for condition in CONDITIONS]
    c0 = rows[0]["baseline_metrics"]
    for row in rows:
        row["baseline_tradeoff"] = baseline_tradeoff(row["baseline_metrics"], c0)
    candidates = select_fork_candidates(rows, maximum=2)
    counts = {label: sum(row["headline"] == label for row in rows) for label in sorted({row["headline"] for row in rows})}
    payload = {
        "status": "COMPLETE", "conditions": list(CONDITIONS), "rows": rows,
        "headline_counts": counts, "fork_candidates": candidates,
        "carrier_candidates": [row["condition"] for row in rows if row["boundedness"].get("bounded_candidate")],
        "termination_tested": False, "lifecycle_tested": False,
        "claim_boundary": (
            "LC6A tests whether patient-axis E-to-I reach opens a bounded high-state carrier; "
            "it does not test termination or a complete lifecycle."
        ),
    }
    _write_json(SUMMARY, {"status": "COMPLETE", "rows": rows})
    _write_json(MAP, payload); _write_csv(rows)
    png, pdf = _plot(rows)
    readme = FIGURES / "README.md"
    existing = readme.read_text() if readme.is_file() else ""
    section = f"""### {png.name}

这张图把 C0/C1/Q1/Q2/Q3 五条固定自然轨迹放在同一口径下比较：自然进入时刻、全局与局部放电、D halo/活动面积，以及 bounded-carrier 的最小余量。绿色余量只表示满足 LC6A 的 carrier 条件；它不表示已经自主终止，也不表示完整 lifecycle。

**关注点**：先看扩大患者轴 E→I reach 是否把 saturation 变成非饱和且 late drift 受控的高态，再看代价是否落在 baseline 或 D-halo 加速上。

### {pdf.name}

与 PNG 内容相同的矢量版本，用于核对各臂数值与标签。

**关注点**：termination 与 lifecycle 在 LC6A 固定为未测试。
"""
    tmp = readme.with_name(readme.name + f".{os.getpid()}.tmp")
    tmp.write_text((existing.rstrip() + "\n\n" + section).lstrip())
    os.replace(tmp, readme)
    _write_json(DONE, {
        "status": "DONE", "phenotype_map": str(MAP), "phenotype_map_sha256": _sha(MAP),
        "fork_candidates": candidates,
    })
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("LC6A phenotype aggregation requires --confirm-run")
    print(json.dumps(NAT._jsonable(run()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
