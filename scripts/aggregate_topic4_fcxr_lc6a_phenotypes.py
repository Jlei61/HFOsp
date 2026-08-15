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


COLORS = {
    "C0": "#222222", "C1": "#8A8A8A", "Q1": "#3B6FB6", "Q2": "#D8842F", "Q3": "#B33B3B",
}
SATURATION_HZ = 250.0
REFRACTORY_GATE = 0.05
DRIFT_GATE = 0.05
MICROSTATE_Q_TOLERANCE = 0.05
BASELINE_TOLERANCE = 0.25


def _per_second_rates(rows, natural_summaries=None):
    if natural_summaries is None:
        natural_summaries = {
            row["condition"]: json.loads(
                (OUT / f"trajectories/{row['condition']}/summary.json").read_text()
            )
            for row in rows
        }
    return {
        row["condition"]: np.asarray(
            natural_summaries[row["condition"]]["per_second_mean_rate_hz"], float,
        )
        for row in rows
    }


def _plot(rows, natural_summaries=None):
    """Four independent questions: entry, escalation shape, which bound broke, baseline cost."""

    FIGURES.mkdir(parents=True, exist_ok=True)
    names = [row["condition"] for row in rows]
    per_second = _per_second_rates(rows, natural_summaries)
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.6), constrained_layout=True)

    # a. Did entry timing move more than the same-q graph-microstate control spread?
    ax = axes[0, 0]
    q = np.asarray([row["construction_q"] for row in rows], float)
    onset = np.asarray(
        [np.nan if row["effective_onset_ms"] is None else row["effective_onset_ms"] / 1000
         for row in rows], float,
    )
    anchor = float(q[names.index("C0")])
    inside = np.abs(q - anchor) <= MICROSTATE_Q_TOLERANCE
    ax.axvspan(
        anchor - MICROSTATE_Q_TOLERANCE, anchor + MICROSTATE_Q_TOLERANCE,
        color="#BBBBBB", alpha=.28, lw=0,
    )
    ax.axhspan(
        float(np.min(onset[inside])), float(np.max(onset[inside])),
        color="#BBBBBB", alpha=.28, lw=0,
    )
    for index, name in enumerate(names):
        ax.scatter(q[index], onset[index], s=70, color=COLORS[name], zorder=3)
        ax.annotate(
            name, (q[index], onset[index]), textcoords="offset points",
            xytext=(7, 5), fontsize=9, color=COLORS[name],
        )
    ax.set_xlabel(r"Realized E$\to$I reach  $q_\parallel^{marginal}$")
    ax.set_ylabel("Natural entry time (s)")
    ax.set_title(
        "a  Entry timing against the same-$q$ microstate band", loc="left", fontsize=11,
    )
    ax.text(
        .02, .04,
        f"grey band: three realizations inside the registered\n"
        f"±{MICROSTATE_Q_TOLERANCE:g} same-$q$ tolerance "
        f"({', '.join(np.asarray(names)[inside])}) span "
        f"{np.min(onset[inside]):.0f}–{np.max(onset[inside]):.0f} s",
        transform=ax.transAxes, fontsize=8, color="#444444", va="bottom",
    )
    ax.set_ylim(0, max(15.0, float(np.nanmax(onset)) + 3.0))
    ax.spines[["top", "right"]].set_visible(False)

    # b. Once entered, does reach change the escalation itself?
    ax = axes[0, 1]
    for row in rows:
        name = row["condition"]
        rate = per_second[name]
        start = int(np.floor(float(row["effective_onset_ms"]) / 1000.0))
        segment = rate[start:]
        ax.plot(
            np.arange(segment.size), segment, "o-", ms=4, lw=1.6,
            color=COLORS[name], label=name,
        )
    ax.axhline(
        SATURATION_HZ, color="0.35", ls="--", lw=1.0,
        label=f"registered saturation ({SATURATION_HZ:.0f} Hz)",
    )
    ax.set_xlabel("Seconds after natural entry")
    ax.set_ylabel("Global mean rate, 1 s (Hz)")
    ax.set_title("b  Escalation after entry is reach-invariant", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # c. Which of the three registered bounded-carrier limits actually broke?
    ax = axes[1, 0]
    criteria = ("peak 1 s rate\n/ 250 Hz", "max refractory\nfraction / 5%", "late drift CI\n/ 0.05 s⁻¹")
    offsets = np.linspace(-.3, .3, len(rows))
    base = np.arange(len(criteria), dtype=float)
    for index, row in enumerate(rows):
        name = row["condition"]
        drift = max(
            float(row["boundedness"][key]["normalized_ci_high_per_s"])
            for key in ("rate_drift", "D_drift", "H_drift")
        )
        values = [
            float(np.max(per_second[name])) / SATURATION_HZ,
            float(row["max_near_refractory_fraction"]) / REFRACTORY_GATE,
            drift / DRIFT_GATE,
        ]
        ax.bar(base + offsets[index], values, width=.13, color=COLORS[name], label=name)
        for slot, value in enumerate(values):
            if value == 0.0:  # a log axis cannot draw an exact zero
                ax.annotate(
                    "0", (base[slot] + offsets[index], 0), xycoords=("data", "axes fraction"),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.5, color=COLORS[name],
                )
    ax.axhline(1.0, color="#B2182B", lw=1.2, ls="--")
    ax.text(
        2.42, 1.25, "registered limit", color="#B2182B", fontsize=8, ha="right",
    )
    ax.set_yscale("log")
    ax.set_xticks(base, criteria, fontsize=9)
    ax.set_ylabel("Measured / registered limit")
    ax.set_title(
        "c  Rate ceiling and late drift break; per-cell refractory does not",
        loc="left", fontsize=11,
    )
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # d. What did the interictal baseline pay for the wider reach?
    ax = axes[1, 1]
    metrics = ("event_rate_hz", "iei_median_ms", "duration_median_ms", "participation_median")
    labels = ("event rate", "median IEI", "median duration", "median participation")
    base = np.arange(len(metrics), dtype=float)
    for index, row in enumerate(rows):
        name = row["condition"]
        values = [
            0.0 if row["baseline_tradeoff"]["relative_differences"][key] is None
            else float(row["baseline_tradeoff"]["relative_differences"][key])
            for key in metrics
        ]
        ax.bar(base + offsets[index], values, width=.13, color=COLORS[name], label=name)
    ax.axhspan(-BASELINE_TOLERANCE, BASELINE_TOLERANCE, color="#BBBBBB", alpha=.3, lw=0)
    ax.axhline(0.0, color="0.3", lw=1.0)
    ax.set_xticks(base, labels, fontsize=9)
    ax.set_ylabel("Relative difference vs C0 pre-onset window")
    ax.set_title(
        "d  Q2/Q3 leave the ±25% baseline band; C1/Q1 do not", loc="left", fontsize=11,
    )
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "FCXR-LC6A patient-axis E→I reach: fixed five-arm phenotype map", fontsize=13,
    )
    fig.text(
        .5, -.012,
        "Axial D-halo / front-speed readouts are omitted: they carry no post-onset dynamic "
        "range on this substrate (see run_manifest.json post_hoc_corrections).",
        ha="center", va="top", fontsize=8, color="#555555",
    )
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

四格各回答一个独立问题：(a) 进入时刻相对"同一 reach、只换连线微状态"的对照带有没有真的移动；(b) 一旦进入，逐秒放电的升级曲线是否随 reach 改变；(c) 三条注册的 bounded-carrier 上限里哪一条被突破；(d) 间期基线事件统计付出了什么代价。轴向 D-halo / front-speed 读数已移除——它们在这块 substrate 上没有进入后的动态范围。

**关注点**：只有 Q3 的进入时刻落在同 q 对照带之外；进入之后五臂的升级曲线几乎重合。

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
