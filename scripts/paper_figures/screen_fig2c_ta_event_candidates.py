#!/usr/bin/env python3
"""Screen stronger E1146 TA single events under the locked Fig. 2c display contract.

The accepted TB event, frozen field geometry, participant support, frame window and one global
TA/TB envelope scale stay fixed.  Candidate ranking uses raw envelope/centroid measurements only;
rendered pixels never enter selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.topic5_interictal_event_field as ief  # noqa: E402
from scripts.paper_figures.screen_fig2c_tb_event_candidates import (  # noqa: E402
    _expected_middle_contacts,
    _percentile,
    _select_distinct_blocks,
)
from scripts.plot_topic5_interictal_event_envelope_field import (  # noqa: E402
    CMAP_NAME,
    OUT as EVENT_FIELD_OUT,
    SNR_MIN_Z,
    _display_vmax_events,
    _inventory,
    build_event,
    build_exemplar_pool,
    event_direction_clarity,
    event_stats,
    load_events,
    load_frozen,
    render,
)


DEFAULT_OUT = (
    ROOT / "results/interictal_propagation_masked/event_envelope_fields/ta_candidate_screen"
)
FRAME_WINDOW_MS = (-8.0, 50.0)
SCHEMA_ID = "fig2c_ta_event_candidate_screen_v1"


def _amplitude_metrics(event, frozen):
    time_ms = (np.asarray(event["t"], float) - float(event["t0"])) * ief.MS
    keep_t = (time_ms >= FRAME_WINDOW_MS[0]) & (time_ms <= FRAME_WINDOW_MS[1])
    part = np.asarray(event["part"], bool)
    axial = np.asarray(frozen["shafts"] == frozen["axial_shaft"], bool) & part
    other = ~np.asarray(frozen["shafts"] == frozen["axial_shaft"], bool) & part
    values = np.clip(
        np.asarray(event["env_z"], float)[part][:, keep_t],
        0.0, None,
    )
    contact_peaks = np.max(values, axis=1)
    axial_values = np.clip(np.asarray(event["env_z"], float)[axial][:, keep_t], 0.0, None)
    other_values = np.clip(np.asarray(event["env_z"], float)[other][:, keep_t], 0.0, None)

    frame_times = np.linspace(FRAME_WINDOW_MS[0], FRAME_WINDOW_MS[1], 5)
    frame_values = np.asarray([
        ief.values_at(
            event["env_z"], event["t"], event["t0"], frame_ms, avg_ms=3.0,
        )
        for frame_ms in frame_times
    ])[:, axial]
    frame_values = np.clip(frame_values, 0.0, None)
    frame_axial_max = np.max(frame_values, axis=1)
    active_threshold = max(5.0, 0.20 * float(np.max(axial_values)))
    active = frame_axial_max >= active_threshold
    xa = np.asarray(frozen["ax_mm"], float)[axial]
    frame_centroids = np.full(len(frame_times), np.nan)
    for i, weights in enumerate(frame_values):
        if active[i] and float(np.sum(weights)) > 0:
            frame_centroids[i] = float(np.sum(xa * weights) / np.sum(weights))
    use = np.isfinite(frame_centroids)
    if int(np.sum(use)) >= 2:
        frame_rho = float(np.corrcoef(
            rankdata(frame_times[use], method="average"),
            rankdata(frame_centroids[use], method="average"),
        )[0, 1])
        frame_span = float(frame_centroids[use][-1] - frame_centroids[use][0])
    else:
        frame_rho, frame_span = float("nan"), float("nan")
    axial_centroids = np.asarray(event["centroid_ms"], float)[axial]
    completion = float(np.mean(
        (axial_centroids >= FRAME_WINDOW_MS[0]) & (axial_centroids <= FRAME_WINDOW_MS[1])
    ))
    return {
        "envelope_q99": float(np.quantile(values, 0.99)),
        "envelope_max": float(np.max(values)),
        "contact_peak_median": float(np.median(contact_peaks)),
        "contact_peak_min": float(np.min(contact_peaks)),
        "axial_envelope_q99": float(np.quantile(axial_values, 0.99)),
        "axial_contact_peak_median": float(np.median(np.max(axial_values, axis=1))),
        "axial_to_other_q99_ratio": float(
            np.quantile(axial_values, 0.99) / max(np.quantile(other_values, 0.99), 1e-9)
        ),
        "axial_completion_by_50_fraction": completion,
        "n_active_axial_static_frames": int(np.sum(active)),
        "static_frame_axis_centroids_mm": [
            None if not np.isfinite(value) else float(value) for value in frame_centroids
        ],
        "static_frame_axis_rho": frame_rho,
        "static_frame_axis_span_mm": frame_span,
    }


def _ta_metrics(event, frozen):
    metrics = event_direction_clarity(event, frozen)
    metrics["left_to_right_monotonic_fraction"] = float(
        1.0 - metrics["right_to_left_monotonic_fraction"]
    )
    metrics["right_minus_left_centroid_ms"] = float(
        -metrics["left_minus_right_centroid_ms"]
    )
    metrics.update(_amplitude_metrics(event, frozen))
    return metrics


def _score_rows(rows):
    direction = _percentile([r["centroid_vs_axis_rho"] for r in rows])
    amplitude = _percentile([r["axial_envelope_q99"] for r in rows])
    contact_amp = _percentile([r["axial_contact_peak_median"] for r in rows])
    middle = _percentile([r["middle_peak_z_min"] for r in rows])
    monotonic = _percentile([r["left_to_right_monotonic_fraction"] for r in rows])
    frame_rho = _percentile([r["static_frame_axis_rho"] for r in rows])
    frame_span = _percentile([r["static_frame_axis_span_mm"] for r in rows])
    active_frames = _percentile([r["n_active_axial_static_frames"] for r in rows])
    completion = _percentile([r["axial_completion_by_50_fraction"] for r in rows])
    template = _percentile([r["rho_vs_template"] for r in rows])
    scores = (
        0.18 * direction + 0.18 * amplitude + 0.14 * contact_amp
        + 0.12 * frame_rho + 0.10 * frame_span + 0.08 * active_frames
        + 0.08 * completion + 0.06 * monotonic + 0.03 * middle + 0.03 * template
    )
    for row, score in zip(rows, scores):
        row["screen_score"] = float(score)
    return rows


def _gate_tier(row, n_middle_expected):
    other_shafts = [v for k, v in row["shaft_counts"].items() if k != row["axial_shaft"]]
    both_shafts = bool(other_shafts) and all(
        v["n_participating"] >= 2 and v["n_usable"] >= 2 and v["n_peak_z_ge_5"] >= 1
        for v in other_shafts
    )
    strict = (
        both_shafts
        and row["centroid_vs_axis_rho"] >= 0.75
        and row["n_middle_usable"] == n_middle_expected
        and row["middle_peak_z_min"] >= SNR_MIN_Z
        and row["right_minus_left_centroid_ms"] >= 8.0
        and row["left_to_right_monotonic_fraction"] >= 0.70
        and row["n_active_axial_static_frames"] >= 2
        and row["static_frame_axis_rho"] >= 0.50
        and row["static_frame_axis_span_mm"] >= 8.0
    )
    relaxed = (
        both_shafts
        and row["centroid_vs_axis_rho"] >= 0.65
        and row["n_middle_usable"] >= max(n_middle_expected - 1, 1)
        and row["middle_peak_z_min"] >= 3.0
        and row["right_minus_left_centroid_ms"] >= 5.0
        and row["left_to_right_monotonic_fraction"] >= 0.60
        and row["n_active_axial_static_frames"] >= 2
        and row["static_frame_axis_rho"] >= 0.30
        and row["static_frame_axis_span_mm"] >= 5.0
    )
    return "strict" if strict else ("relaxed" if relaxed else "outside")


def _write_readme(figures_dir, current, selected, raw_global_q99):
    blocks = []
    for label, row in [("current", current)] + [
        (f"candidate_{i:02d}", value) for i, value in enumerate(selected, 1)
    ]:
        status = "当前 TA exemplar" if label == "current" else "TA 增强候选"
        shaft_text = "、".join(
            f"{shaft} {counts['n_usable']}/{counts['n_participating']} usable"
            for shaft, counts in row["shaft_counts"].items()
        )
        blocks.append(
            f"### {row['figure_name']}\n\n"
            f"{status}；固定当前 TB exemplar、frozen geometry、participant-only support、6 mm "
            f"display kernel 和 −8…+50 ms 窗；每个静态 frame 按参与触点 top-3 mean 归一化到 0–1。"
            f"TA 沿 {row['axial_shaft']} 的"
            f"质心-轴 Spearman={row['centroid_vs_axis_rho']:+.3f}，左→右单调比例="
            f"{row['left_to_right_monotonic_fraction']:.3f}，右端减左端时差="
            f"{row['right_minus_left_centroid_ms']:.1f} ms，envelope q99="
            f"{row['envelope_q99']:.1f}，contact peak 中位数="
            f"{row['contact_peak_median']:.1f}；两杆参与为 {shaft_text}。\n\n"
            "**关注点**：在不改变 TB/几何/帧时刻的前提下比较 TA 左→右连续移动；绝对幅度"
            "只读上面的 raw 指标，不能从归一化颜色比较。这是候选筛查，不自动替换正式 Fig2-C。"
        )
    text = (
        "# Fig2-C E1146 TA 单事件增强候选筛查\n\n"
        f"中间场使用低视觉权重 `{CMAP_NAME}`；每对 TA/TB 的 4 个等间距静态帧由同一 "
        "joint-visible contact-level selector 确定，不按渲染像素手挑。"
        f"全候选联合 raw robust-z q99={raw_global_q99:.3f} 仅作为幅度审计，不作为显示上限。"
        "候选排序只读取原始 envelope/centroid 数值，不读取渲染像素；归一化候选图不能用于"
        "比较候选间绝对幅度。\n\n" + "\n".join(blocks)
    )
    path = Path(figures_dir) / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def run(ds_sid="epilepsiae_1146", *, output_dir=DEFAULT_OUT, n_candidates=4, top_k=500):
    dataset, subject = ds_sid.split("_", 1)
    if dataset != "epilepsiae":
        raise NotImplementedError("the raw TA candidate screen is wired for Epilepsiae only")
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    frozen = load_frozen(ds_sid)
    events = load_events(frozen, subject)
    inventory = _inventory(subject)
    canonical_path = EVENT_FIELD_OUT / f"{ds_sid}_event_envelope_field.json"
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    current_ta_pos = int(canonical["exemplar"]["TA"]["event_pos"])
    fixed_tb_pos = int(canonical["exemplar"]["TB"]["event_pos"])
    fixed_tb = build_event(events, fixed_tb_pos, inventory, subject, frozen)

    axial = np.asarray(frozen["shafts"] == frozen["axial_shaft"], bool)
    other = ~axial
    x = np.asarray(frozen["ax_mm"], float)
    q1, q2 = np.quantile(x[axial], [1.0 / 3.0, 2.0 / 3.0])
    middle = axial & (x >= q1) & (x <= q2)

    def candidate_filter(pos):
        part = np.asarray(events["bools"][:, pos], bool)
        if int(np.sum(part & axial)) < 6 or int(np.sum(part & other)) < 2:
            return False
        if not np.all(part[middle]):
            return False
        stored = ief.stored_lag_rel_ms(events["lag_raw"][:, pos], part)
        use = part & axial & np.isfinite(stored) & np.isfinite(x)
        if int(use.sum()) < 6:
            return False
        rho = float(np.corrcoef(
            rankdata(stored[use], method="average"), rankdata(x[use], method="average")
        )[0, 1])
        return np.isfinite(rho) and rho >= 0.50

    built, _, context = build_exemplar_pool(
        frozen, events, inventory, subject, 0, "TA", top_k=int(top_k),
        candidate_filter=candidate_filter, max_raw_candidates=40,
    )
    if current_ta_pos not in built:
        built[current_ta_pos] = build_event(
            events, current_ta_pos, inventory, subject, frozen,
        )
    expected_middle = _expected_middle_contacts(frozen)
    rows = []
    for pos, event in built.items():
        rho = context["rhos"].get(pos)
        if rho is None:
            rho = ief.template_likeness(
                events["masked"][:, pos], frozen["rank_a"], events["bools"][:, pos],
            )
        row = dict(
            event_pos=int(pos), block=str(event["stem"]),
            t_in_block_sec=float(event["t_in_block"]), rho_vs_template=float(rho),
            n_participating=int(event["n_part"]), n_snr=int(event["n_snr"]),
            stored_spread_ms=float(event["spread_ms"]),
            is_current_reference=bool(pos == current_ta_pos),
            **_ta_metrics(event, frozen),
        )
        row["n_middle_expected"] = len(expected_middle)
        row["gate_tier"] = _gate_tier(row, len(expected_middle))
        rows.append(row)
    _score_rows(rows)
    current = next(row for row in rows if row["is_current_reference"])
    alternatives = [row for row in rows if not row["is_current_reference"]]
    selected = _select_distinct_blocks(alternatives, int(n_candidates))
    if len(selected) < int(n_candidates):
        raise ValueError(f"only {len(selected)} TA alternatives pass strict/relaxed gates")

    display_events = [fixed_tb, built[current_ta_pos]] + [built[r["event_pos"]] for r in selected]
    global_vmax = _display_vmax_events(display_events, *FRAME_WINDOW_MS)
    render_rows = [("current_reference", current)] + [
        (f"candidate_{i:02d}", row) for i, row in enumerate(selected, 1)
    ]
    for label, row in render_rows:
        ta = built[row["event_pos"]]
        name = f"{label}_ta_pos_{row['event_pos']:05d}_{row['block']}.png"
        render(
            ds_sid, frozen, ta, fixed_tb, event_stats(ta, frozen), event_stats(fixed_tb, frozen),
            figures_dir / name, support_mode="participant", dpi=150,
            frame_window=FRAME_WINDOW_MS,
        )
        row["figure_name"] = name

    ranking = sorted(rows, key=lambda row: (row["gate_tier"] != "strict", -row["screen_score"]))
    payload = {
        "schema_id": SCHEMA_ID,
        "status": "candidate screen only; canonical Fig2-C not overwritten",
        "ds_sid": ds_sid,
        "frozen_fingerprint": frozen["fingerprint"],
        "fixed_tb_event_pos": fixed_tb_pos,
        "current_ta_event_pos": current_ta_pos,
        "display_contract": {
            "frame_window_ms": list(FRAME_WINDOW_MS),
            "raw_global_q99_robust_z_for_audit": float(global_vmax),
            "normalization": (
                "static: per-frame participant top3 mean to 0..1; "
                "GIF: per-event participant complete-window q99 to 0..1"
            ),
            "display_sigma_mm": float(frozen["display_sigma_mm"]),
            "support": "participant-only", "cmap": CMAP_NAME,
            "geometry": "frozen shared plane", "static_frame_count": 5,
        },
        "screening_contract": {
            "rendered_pixels_used": False,
            "score": (
                "0.18 direction-rho + 0.18 axial-envelope-q99 + 0.14 axial-contact-peak-"
                "median + 0.12 static-frame-axis-rho + 0.10 static-frame-axis-span + "
                "0.08 active-static-frames + 0.08 completion-by-50ms + 0.06 contact-"
                "monotonicity + 0.03 middle-min-peak-z + 0.03 template-rho; percentile scaled"
            ),
            "strict_gate": (
                "both shafts usable; rho>=0.75; all axial middle-third contacts usable; "
                "middle min peak-z>=5; right-minus-left centroid>=8 ms; left-to-right "
                "monotonic fraction>=0.70; >=2 active axial static frames; static-frame "
                "axis rho>=0.50 and span>=8 mm"
            ),
        },
        "selected_event_positions": [int(row["event_pos"]) for row in selected],
        "current_reference": current,
        "selected": selected,
        "all_ranked": ranking,
    }
    json_path = output_dir / "ta_candidate_screen.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = output_dir / "ta_candidate_screen.csv"
    csv_fields = [key for key, value in ranking[0].items() if not isinstance(value, (dict, list))]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranking)
    _write_readme(figures_dir, current, selected, global_vmax)
    print(f"[TA screen] selected {[row['event_pos'] for row in selected]}", flush=True)
    print(f"[TA screen] metadata {json_path}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="epilepsiae_1146")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-candidates", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=500)
    args = parser.parse_args()
    run(
        args.subject, output_dir=args.output_dir,
        n_candidates=args.n_candidates, top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
