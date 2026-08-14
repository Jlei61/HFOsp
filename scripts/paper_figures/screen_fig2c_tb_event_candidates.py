#!/usr/bin/env python3
"""Screen alternative E1146 TB single events under one locked Fig. 2c display contract.

This is an audit/screening producer, not the accepted Fig. 2c producer.  It keeps the
canonical TA exemplar fixed, excludes the current TB medoid from the alternatives, and
ranks raw-EEG-derived TB candidates without inspecting rendered pixels.  All comparison
figures share one frame window, field geometry, display sigma, and static frame-relative rule.
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
from scripts.plot_topic5_interictal_event_envelope_field import (  # noqa: E402
    CMAP_NAME,
    GIF_FPS,
    GIF_STEP_MS,
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
    pick_exemplar,
    render,
    render_gif,
)


DEFAULT_OUT = (
    ROOT
    / "results"
    / "interictal_propagation_masked"
    / "event_envelope_fields"
    / "tb_candidate_screen"
)
FRAME_WINDOW_MS = (-8.0, 50.0)
SCHEMA_ID = "fig2c_tb_event_candidate_screen_v1"


def _current_medoid(ds_sid, built, feats, ctx):
    """Use the TB event frozen in the current canonical metadata, with a top-40 fallback."""
    meta_path = EVENT_FIELD_OUT / f"{ds_sid}_event_envelope_field.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))["exemplar"]["TB"]
        pos = int(meta["event_pos"])
        if pos in built:
            return pos, float(meta.get("medoid_distance", np.nan)), "canonical_metadata"
    top40 = [int(p) for p in ctx["candidate_positions"][:40] if int(p) in built]
    pos, dist = ief.select_medoid_event(
        top40,
        feats,
        [
            ctx["target_spread_ms"],
            ctx["target_n_participating"],
            float(np.median([f[2] for f in feats.values()])),
        ],
    )
    return pos, dist, "recomputed_top40_medoid"


def _expected_middle_contacts(fz):
    axial = np.asarray(fz["shafts"] == fz["axial_shaft"], bool)
    x = np.asarray(fz["ax_mm"], float)
    q1, q2 = np.quantile(x[axial], [1.0 / 3.0, 2.0 / 3.0])
    return [str(n) for n, keep in zip(fz["names"], axial & (x >= q1) & (x <= q2)) if keep]


def _percentile(values):
    x = np.asarray(values, float)
    finite = np.isfinite(x)
    out = np.zeros_like(x)
    if finite.any():
        out[finite] = rankdata(x[finite], method="average") / float(finite.sum())
    return out


def _score_rows(rows):
    """Score numeric readout properties only; no field/image pixels enter selection."""
    reverse = _percentile([-r["centroid_vs_axis_rho"] for r in rows])
    middle = _percentile([r["middle_peak_z_min"] for r in rows])
    monotonic = _percentile([r["right_to_left_monotonic_fraction"] for r in rows])
    delay = _percentile([r["left_minus_right_centroid_ms"] for r in rows])
    template = _percentile([r["rho_vs_template"] for r in rows])
    scores = 0.35 * reverse + 0.25 * middle + 0.20 * monotonic + 0.10 * delay + 0.10 * template
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
        and
        row["centroid_vs_axis_rho"] <= -0.75
        and row["n_middle_usable"] == n_middle_expected
        and row["middle_peak_z_min"] >= SNR_MIN_Z
        and row["left_minus_right_centroid_ms"] >= 8.0
        and row["right_to_left_monotonic_fraction"] >= 0.70
    )
    relaxed = (
        both_shafts
        and
        row["centroid_vs_axis_rho"] <= -0.65
        and row["n_middle_usable"] >= max(n_middle_expected - 1, 1)
        and row["middle_peak_z_min"] >= 3.0
        and row["left_minus_right_centroid_ms"] >= 5.0
        and row["right_to_left_monotonic_fraction"] >= 0.60
    )
    return "strict" if strict else ("relaxed" if relaxed else "outside")


def _select_distinct_blocks(rows, n):
    """Prefer strict completeness before block diversity, then use relaxed events if needed."""
    picked, seen = [], set()
    for tier in ("strict", "relaxed"):
        pool = sorted(
            (r for r in rows if r["gate_tier"] == tier),
            key=lambda r: (-r["screen_score"], r["event_pos"]),
        )
        for row in pool:
            if row["block"] in seen:
                continue
            picked.append(row)
            seen.add(row["block"])
            if len(picked) == n:
                return picked
        for row in pool:
            if row in picked:
                continue
            picked.append(row)
            seen.add(row["block"])
            if len(picked) == n:
                return picked
    return picked


def _write_readme(
    figures_dir, selected, current, raw_global_q99, *, selected_for_fig2c_event_pos=None,
):
    blocks = []
    all_rows = [("current_reference", current)] + [
        (f"candidate_{i:02d}", row) for i, row in enumerate(selected, 1)
    ]
    for label, row in all_rows:
        filename = row["figure_name"]
        is_selected = (
            selected_for_fig2c_event_pos is not None
            and int(row["event_pos"]) == int(selected_for_fig2c_event_pos)
        )
        if is_selected:
            status = "当前 Fig2-C 选定 TB 单事件"
        elif label == "current_reference":
            status = "旧 medoid TB 参照事件"
        else:
            status = "筛查保留 TB 候选"
        shaft_text = "、".join(
            f"{shaft} {count['n_usable']}/{count['n_participating']} usable"
            for shaft, count in row["shaft_counts"].items()
        )
        blocks.append(
            f"### {filename}\n\n"
            f"{status}；TA exemplar、shared plane、6 mm display kernel 和 −8…+50 ms frame "
            f"window 固定；每个静态 frame 按本帧参与触点 top-3 mean 归一化到 0–1。TB 沿 "
            f"{row['axial_shaft']} 的质心-轴 Spearman="
            f"{row['centroid_vs_axis_rho']:+.3f}，中段可用 {row['n_middle_usable']}/"
            f"{row['n_middle_expected']}，中段最低 peak-z={row['middle_peak_z_min']:.1f}，"
            f"左端减右端质心时差={row['left_minus_right_centroid_ms']:.1f} ms；"
            f"两杆参与为 {shaft_text}。\n\n"
            f"**关注点**：只比较 TB 从 shared-axis 右端向左端的连续移动及中段是否断黑；"
            + (
                "该事件已锁为 Fig2-C representative single-event sample，不升级为群体证据。"
                if is_selected else
                "本图保留为选择 provenance，不自动替换正式 Fig2-C。"
            )
        )
        if row.get("gif"):
            gif = row["gif"]
            blocks.append(
                f"### {Path(gif['figure']).name}\n\n"
                f"同一 TA 与 TB event {row['event_pos']} 的动态版本；生物学步长 "
                f"{gif['biological_step_ms']:.0f} ms，播放 {gif['playback_fps']:.0f} fps，"
                f"并复用静态候选的 −8…+50 ms 窗口、participant-only support、6 mm "
                f"display kernel；GIF 另用同一事件的冻结完整窗 q99 分母。\n\n"
                f"**关注点**：观察 TB 两根杆参与时热区从 shared-axis 右端向中部/左端转移；"
                f"播放速度不代表真实生物学时间倍率。"
            )
    selection_text = (
        f"当前锁定 TB event={int(selected_for_fig2c_event_pos)}；"
        if selected_for_fig2c_event_pos is not None else
        "当前尚未在 screen metadata 中锁定正式 candidate；"
    )
    text = (
        "# Fig2-C E1146 TB 单事件候选筛查\n\n"
        "所有图固定同一个 TA exemplar，并锁定 frozen geometry、participant-only support、"
        f"6 mm display kernel 与 −8…+50 ms frame window。静态 frame 分别按参与触点 top-3 mean "
        f"归一化到 0–1；全候选联合 raw robust-z q99={raw_global_q99:.3f} 仅作幅度审计，不作显示上限。"
        "每行依次显示单事件 readout、4 个等间距 joint-visible normalized HFO envelope frames、以及冻结的 "
        "viridis template propagation-rank field；readout 取两次真实 STFT 窗交集，rank colorbar "
        "显示 artifact 实际数值。候选按原始 readout 指标筛选，不读取渲染"
        f"像素；{selection_text}\n\n"
        + "\n".join(blocks)
    )
    path = Path(figures_dir) / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def run(ds_sid="epilepsiae_1146", *, output_dir=DEFAULT_OUT, n_candidates=4, top_k=500):
    dataset, subject = ds_sid.split("_", 1)
    if dataset != "epilepsiae":
        raise NotImplementedError("the raw candidate screen is wired for Epilepsiae only")
    output_dir = Path(output_dir)
    previous_selection = None
    previous_meta = output_dir / "tb_candidate_screen.json"
    if previous_meta.exists():
        try:
            previous_selection = json.loads(
                previous_meta.read_text(encoding="utf-8")
            ).get("selected_for_fig2c_event_pos")
        except (OSError, ValueError, TypeError):
            previous_selection = None
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for stale in figures_dir.glob("*.png"):
        stale.unlink()

    fz = load_frozen(ds_sid)
    ev = load_events(fz, subject)
    inv = _inventory(subject)
    ta, ta_meta = pick_exemplar(fz, ev, inv, subject, 0, "TA")
    ta_stats = event_stats(ta, fz)
    axial = np.asarray(fz["shafts"] == fz["axial_shaft"], bool)
    other = ~axial
    x = np.asarray(fz["ax_mm"], float)
    q1, q2 = np.quantile(x[axial], [1.0 / 3.0, 2.0 / 3.0])
    middle_mask = axial & (x >= q1) & (x <= q2)

    def candidate_filter(pos):
        part = np.asarray(ev["bools"][:, pos], bool)
        if int(np.sum(part & axial)) < 6 or int(np.sum(part & other)) < 2:
            return False
        if not np.all(part[middle_mask]):
            return False
        stored = ief.stored_lag_rel_ms(ev["lag_raw"][:, pos], part)
        m = part & axial & np.isfinite(stored) & np.isfinite(x)
        if int(m.sum()) < 6:
            return False
        rho = float(np.corrcoef(
            rankdata(stored[m], method="average"), rankdata(x[m], method="average")
        )[0, 1])
        return np.isfinite(rho) and rho <= -0.50

    built, feats, ctx = build_exemplar_pool(
        fz, ev, inv, subject, 1, "TB", top_k=int(top_k),
        candidate_filter=candidate_filter, max_raw_candidates=40,
    )
    current_pos, current_dist, current_source = _current_medoid(ds_sid, built, feats, ctx)
    expected_middle = _expected_middle_contacts(fz)

    rows = []
    for pos, event in built.items():
        metrics = event_direction_clarity(event, fz)
        row = dict(
            event_pos=int(pos), block=str(event["stem"]),
            t_in_block_sec=float(event["t_in_block"]),
            rho_vs_template=float(ctx["rhos"][pos]),
            n_participating=int(event["n_part"]), n_snr=int(event["n_snr"]),
            stored_spread_ms=float(event["spread_ms"]),
            is_current_reference=bool(pos == current_pos),
            **metrics,
        )
        row["n_middle_expected"] = len(expected_middle)
        row["expected_middle_contacts"] = list(expected_middle)
        rows.append(row)
    _score_rows(rows)
    for row in rows:
        row["gate_tier"] = _gate_tier(row, len(expected_middle))

    current = next(r for r in rows if r["is_current_reference"])
    alternatives = [r for r in rows if not r["is_current_reference"]]
    selected = _select_distinct_blocks(alternatives, int(n_candidates))
    if len(selected) < int(n_candidates):
        raise ValueError(
            f"only {len(selected)} alternative TB events pass strict/relaxed clarity gates; "
            "increase --top-k before relaxing the declared gates"
        )

    events_for_scale = [ta, built[current_pos]] + [built[r["event_pos"]] for r in selected]
    global_vmax = _display_vmax_events(events_for_scale, *FRAME_WINDOW_MS)
    render_rows = [("current_reference", current)] + [
        (f"candidate_{i:02d}", row) for i, row in enumerate(selected, 1)
    ]
    for label, row in render_rows:
        suffix = f"{label}_tb_pos_{row['event_pos']:05d}_{row['block']}"
        png = figures_dir / f"{suffix}.png"
        tb = built[row["event_pos"]]
        render(
            ds_sid, fz, ta, tb, ta_stats, event_stats(tb, fz), png,
            support_mode="participant", dpi=150, frame_window=FRAME_WINDOW_MS,
        )
        row["figure_name"] = png.name

    ranking = sorted(rows, key=lambda r: (r["gate_tier"] != "strict", -r["screen_score"]))
    payload = dict(
        schema_id=SCHEMA_ID, ds_sid=ds_sid,
        status="candidate screen only; canonical Fig2-C not overwritten",
        frozen_fingerprint=fz["fingerprint"], axial_shaft=fz["axial_shaft"],
        fixed_ta=dict(event_pos=int(ta["pos"]), block=str(ta["stem"]), selection=ta_meta),
        current_tb=dict(
            event_pos=int(current_pos), medoid_distance=float(current_dist),
            source=current_source,
        ),
        display_contract=dict(
            frame_window_ms=list(FRAME_WINDOW_MS),
            raw_global_q99_robust_z_for_audit=float(global_vmax),
            normalization=("static: per-frame participant top3 mean to 0..1; "
                           "GIF: per-event participant complete-window q99 to 0..1"),
            display_sigma_mm=float(fz["display_sigma_mm"]), support="participant-only",
            cmap=CMAP_NAME, geometry="frozen shared plane",
        ),
        screening_contract=dict(
            expected_middle_contacts=expected_middle,
            strict_gate=(
                "each shaft has >=2 participating and usable contacts plus >=1 peak-z>=5; "
                "rho<=-0.75; all axial middle-third contacts usable; middle min peak-z>=5; "
                "left-minus-right centroid>=8 ms; right-to-left monotonic fraction>=0.70"
            ),
            relaxed_gate=(
                "same two-shaft gate; rho<=-0.65; at most one axial middle-third contact "
                "missing; middle min "
                "peak-z>=3; left-minus-right centroid>=5 ms; monotonic fraction>=0.60"
            ),
            score=(
                "percentile composite: 0.35 reverse-rho + 0.25 middle-min-peak-z + "
                "0.20 monotonicity + 0.10 endpoint-delay + 0.10 template-rho"
            ),
            selection=(
                "strict before relaxed; within each tier descending score with distinct blocks "
                "preferred before same-block events"
            ),
            rendered_pixels_used=False,
            cheap_prefilter=(
                f"top-{int(top_k)} template-concordant events; axial participants>=6; "
                "other-shaft participants>=2; all axial middle-third contacts participating; "
                "stored-lag rho<=-0.50; only first 40 survivors read from raw EEG"
            ),
        ),
        selected_event_positions=[int(r["event_pos"]) for r in selected],
        current_reference=current, selected=selected, all_ranked=ranking,
    )
    selected_positions = {int(r["event_pos"]) for r in selected}
    if previous_selection is not None and int(previous_selection) in selected_positions:
        payload["selected_for_fig2c_event_pos"] = int(previous_selection)
        payload["status"] = (
            "candidate screen retained as selection provenance; selected TB event is locked "
            "for the Fig2-C representative single-event panel"
        )
    json_path = output_dir / "tb_candidate_screen.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = output_dir / "tb_candidate_screen.csv"
    csv_fields = [k for k, v in ranking[0].items() if not isinstance(v, (list, dict))]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranking)
    readme = _write_readme(
        figures_dir, selected, current, global_vmax,
        selected_for_fig2c_event_pos=payload.get("selected_for_fig2c_event_pos"),
    )
    print(f"[screen] selected {[r['event_pos'] for r in selected]}", flush=True)
    print(f"[screen] metadata {json_path}", flush=True)
    print(f"[screen] README {readme}", flush=True)
    return payload


def render_candidate_gif(
    ds_sid, event_pos, *, output_dir=DEFAULT_OUT,
    step_ms=GIF_STEP_MS, fps=GIF_FPS, mark_selected=False,
):
    """Re-render one screened candidate as a matched static panel and GIF."""
    dataset, subject = ds_sid.split("_", 1)
    if dataset != "epilepsiae":
        raise NotImplementedError("the raw candidate GIF is wired for Epilepsiae only")
    output_dir = Path(output_dir)
    meta_path = output_dir / "tb_candidate_screen.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if payload["ds_sid"] != ds_sid:
        raise ValueError(f"screen metadata is for {payload['ds_sid']}, not {ds_sid}")
    event_pos = int(event_pos)
    selected = next((r for r in payload["selected"] if int(r["event_pos"]) == event_pos), None)
    if selected is None:
        selected = next(
            (
                r for r in payload.get("all_ranked", [])
                if int(r["event_pos"]) == event_pos
                and r.get("gate_tier") in {"strict", "relaxed"}
            ),
            None,
        )
    if selected is None:
        raise ValueError(
            f"TB event {event_pos} is not a direction-qualified event in the frozen screen"
        )

    fz = load_frozen(ds_sid)
    if fz["fingerprint"] != payload["frozen_fingerprint"]:
        raise ValueError("frozen field fingerprint drifted since candidate screening")
    ev = load_events(fz, subject)
    if int(ev["labels"][event_pos]) != 1:
        raise ValueError(f"event {event_pos} is no longer assigned to TB")
    inv = _inventory(subject)
    ta, _ = pick_exemplar(fz, ev, inv, subject, 0, "TA")
    tb = build_event(ev, event_pos, inv, subject, fz)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_name = selected.get("figure_name")
    if png_name is None:
        safe_block = Path(str(selected.get("block", "event"))).name
        png_name = f"candidate_locked_tb_pos_{event_pos:05d}_{safe_block}.png"
        selected["figure_name"] = png_name
    gif_path = figures_dir / f"{Path(png_name).stem}.gif"
    display = payload["display_contract"]
    raw_global_q99 = float(display.get(
        "raw_global_q99_robust_z_for_audit", display.get("global_vmax", np.nan),
    ))
    static_meta = render(
        ds_sid, fz, ta, tb, event_stats(ta, fz), event_stats(tb, fz),
        figures_dir / png_name, support_mode="participant", dpi=150,
        frame_window=display["frame_window_ms"],
    )
    gif_meta = render_gif(
        ds_sid, fz, ta, tb, event_stats(ta, fz), event_stats(tb, fz), gif_path,
        support_mode="participant", step_ms=float(step_ms), fps=float(fps),
        frame_window=display["frame_window_ms"],
    )
    selected["static"] = static_meta
    selected["gif"] = gif_meta
    if mark_selected:
        if not any(int(r["event_pos"]) == event_pos for r in payload["selected"]):
            payload["selected"].append(selected)
        payload["selected_for_fig2c_event_pos"] = int(event_pos)
        payload["status"] = (
            "candidate screen retained as selection provenance; selected TB event is locked "
            "for the Fig2-C representative single-event panel"
        )
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_readme(
        figures_dir, payload["selected"], payload["current_reference"],
        raw_global_q99,
        selected_for_fig2c_event_pos=payload.get("selected_for_fig2c_event_pos"),
    )
    print(f"[candidate-gif] {gif_path}", flush=True)
    return gif_meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="epilepsiae_1146")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-candidates", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=500)
    ap.add_argument("--gif-event-pos", type=int)
    ap.add_argument("--gif-step-ms", type=float, default=GIF_STEP_MS)
    ap.add_argument("--gif-fps", type=float, default=GIF_FPS)
    ap.add_argument(
        "--mark-selected-for-fig2c", action="store_true",
        help="lock --gif-event-pos as the representative Fig2-C TB single event in metadata",
    )
    args = ap.parse_args()
    if args.gif_event_pos is not None:
        render_candidate_gif(
            args.subject, args.gif_event_pos, output_dir=args.output_dir,
            step_ms=args.gif_step_ms, fps=args.gif_fps,
            mark_selected=args.mark_selected_for_fig2c,
        )
        return
    run(
        args.subject, output_dir=args.output_dir,
        n_candidates=args.n_candidates, top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
