#!/usr/bin/env python3
"""Build the author-selected Y1 revision using the reviewed Figure 1 helpers."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures import build_fig1_readability_review as shared
from matplotlib.ticker import FuncFormatter

plt = shared.plt
OUT = shared.CANON / "candidates/aligned_y1_20261009"
FIG = OUT / "figures"
PATIENT = "Y1"
A_ROOT = shared.CANON / "candidates/recording_chain_kshaft_20261009"


def rect(fig, x, y, w, h):
    width, height = fig.get_size_inches()
    return [x / width, y / height, w / width, h / height]


def draw_ce_aligned(fig, y, arr, clustered=False):
    start = len(fig.axes)
    summary = shared.draw_ce(fig, rect(fig, 0, y, 11.55, 4.05), arr, "Yuquan Y1", clustered)
    heat, colorbar, profile, *strip = fig.axes[start:]
    heat.set_position(rect(fig, 1.05, y + .68, 7.95, 3.10))
    colorbar.set_position(rect(fig, 9.22, y + .68, .14, 3.10))
    profile.set_position(rect(fig, 10.03, y + .68, 1.22, 3.10))
    heat.tick_params(axis="y", labelsize=8)
    if strip:
        strip[0].set_position(rect(fig, 1.05, y + .50, 7.95, .11))
    return summary, heat


def draw_b_compact(fig, x, y, spectrum):
    shared.image_in_rect(fig, rect(fig, x, y, 2.00, 4.30),
                         shared.CANON / "figures/fig1-panelb1.png")
    start = len(fig.axes)
    text_start = len(fig.texts)
    shared.draw_spectrum(fig, rect(fig, x + 2.0, y, 4.60, 4.30), spectrum)
    axes = fig.axes[start:]
    labels = [channel.split("-")[0] for channel in spectrum["meta"]["selection"]["selected_channels"]]
    for event, ax in enumerate(axes[:3]):
        ax.set_position(rect(fig, x + 2.54 + 1.27 * event, y + .50, 1.20, 3.70))
        ax.set_title("")
        ax.set_ylabel("")
        ax.set_yticklabels(labels if event == 0 else [])
        # Preserve the physical coordinates of the concatenated A recordings.
        center = .16 + .32 * event
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos, c=center: f"{c + value / 1000:.2f}"))
        ax.set_xlabel("Time (s)" if event == 1 else "", fontsize=12, labelpad=6)
        if event < 2:
            for xx in (.995, 1.035):
                ax.plot([xx-.012, xx+.012], [-.012, .012], color="black", lw=.65,
                        transform=ax.transAxes, clip_on=False)
    axes[3].set_position(rect(fig, x + 6.37, y + .50, .10, 3.70))
    title = fig.texts[text_start]
    title.set_text("Yuquan Y3")
    title.set_position(((x + 4.41) / fig.get_figwidth(), (y + 4.36) / fig.get_figheight()))
    title.set(ha="center", va="bottom", fontsize=12, fontweight="bold")
    return dict(title="Yuquan Y3", ylabel="", display_labels=labels,
                label_rule="left contact alias of each adjacent bipolar pair",
                xlabel="Time (s)", time_reference="original concatenated excerpt time",
                displayed_time_windows_sec=[[.12,.20],[.44,.52],[.76,.84]],
                spectrum_and_centroids_unchanged=True)


def draw_d_aligned(fig, x, y, records, mechanism):
    start = len(fig.axes)
    summary = shared.draw_d(fig, rect(fig, x-.65, y-.65, 4.30, 4.40), records, mechanism)
    diagram, stats = fig.axes[start:]
    diagram.set_position(rect(fig, x-.50, y+3.21, 3.90, .82))
    stats.set_position(rect(fig, x, y, 3.30, 3.10))
    for text in stats.texts:
        if text.get_text() in ("Yuquan", "Epilepsiae"):
            text.set_y(-.14)
    return summary, stats


def draw_f_aligned(fig, x, y, records):
    start = len(fig.axes)
    summary = shared.draw_f(fig, rect(fig, x-.65, y-.65, 4.30, 4.05), records)
    stats = fig.axes[start]
    stats.set_aspect("auto")
    stats.set_position(rect(fig, x, y, 3.30, 3.10))
    return summary, stats


def check_row_alignment(fig, left, right):
    fig.canvas.draw()
    a, b = left.get_position(), right.get_position()
    assert abs(a.y0-b.y0) < 1e-9
    assert abs(a.height-b.height) < 1e-9
    return dict(baseline_difference_mm=float((a.y0-b.y0)*fig.get_figheight()*25.4),
                height_difference_mm=float((a.height-b.height)*fig.get_figheight()*25.4),
                data_axis_height_mm=float(a.height*fig.get_figheight()*25.4))


def check_channel_labels(fig, arr):
    """Check real rendered text bounds, not just nominal font sizes."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    checked = 0
    min_gap = float("inf")
    for ax in fig.axes:
        labels = [t for t in ax.get_yticklabels() if t.get_visible()]
        if [t.get_text() for t in labels] != arr["ordered_names"]:
            continue
        bounds = sorted([t.get_window_extent(renderer) for t in labels], key=lambda b: b.y0)
        for first, second in zip(bounds, bounds[1:]):
            gap = second.y0 - first.y1
            min_gap = min(min_gap, gap)
            assert gap >= 0, f"Overlapping channel labels: {gap} pixels"
        checked += 1
    assert checked > 0
    return dict(heatmaps_checked=checked, minimum_label_gap_px=min_gap)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    previous = json.loads((shared.OUT / "metadata.json").read_text())
    canonical_before = {str(p): shared.sha(p) for p in (shared.CANON / "figures").glob("*") if p.is_file()}
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.unicode_minus": False})
    shared.old.propagation_plot._apply_masked_paths()
    records = shared.old._load_temporal_records()
    shared.old._assert_masked_mi_records(records)
    record = next(r for r in records if shared.public_patient_label(r["dataset"], r["subject"]) == PATIENT)
    arr = shared.old._load_exemplar_arrays(record, max_events=10**9)
    assert arr["channel_names"] == record["channel_names"]
    assert len(arr["channel_names"]) == 26
    assert record["adaptive_cluster"]["chosen_k"] == 2
    counts = [int(sum(arr["labels"] == i)) for i in range(2)]
    assert counts == [13160, 5030]
    assert sum(counts) == len(arr["valid_events"]) == record["adaptive_cluster"]["n_valid_events"] == 18190
    assert sorted(arr["clustered_events_all"].tolist()) == sorted(arr["valid_events"].tolist())
    label = "Yuquan Y1"
    print("Y1 loaded: 26 channels, 18190 events", flush=True)

    spectrum = shared.load_spectrum()
    a_path = A_ROOT / "figures/fig1-panela"
    assert json.loads((A_ROOT / "metadata.json").read_text())["schema"] == "fig1a_data_recording_chain_kshaft_v3"
    panel_meta, label_checks = {}, {}
    for clustered, panel in [(False, "c"), (True, "e")]:
        fig = plt.figure(figsize=(11.55, 4.15))
        panel_meta[panel], _ = draw_ce_aligned(fig, 0, arr, clustered)
        label_checks[panel] = check_channel_labels(fig, arr)
        shared.save(fig, FIG / f"fig1-panel{panel}", dpi=300)

    for suffix in (".png", ".pdf", ".svg"):
        shutil.copy2(a_path.with_suffix(suffix), FIG / f"fig1-panela{suffix}")
    fig = plt.figure(figsize=(6.65, 4.75))
    bmeta = draw_b_compact(fig, 0, .05, spectrum)
    shared.save(fig, FIG / "fig1-panelb", dpi=300)
    fig = plt.figure(figsize=(4.30, 4.85))
    dmeta, _ = draw_d_aligned(fig, .65, .65, records, previous["mechanism"])
    shared.save(fig, FIG / "fig1-paneld", dpi=300)
    fig = plt.figure(figsize=(4.30, 4.15))
    fmeta, _ = draw_f_aligned(fig, .65, .65, records)
    shared.save(fig, FIG / "fig1-panelf", dpi=300)

    width, height = 16, 13.80
    fig = plt.figure(figsize=(width, height))
    shared.image_in_rect(fig, rect(fig, .22, 9.14, 8.65, 4.28), a_path.with_suffix(".png"))
    draw_b_compact(fig, 9.10, 9.02, spectrum)
    _, c_axis = draw_ce_aligned(fig, 4.28, arr)
    _, e_axis = draw_ce_aligned(fig, .12, arr, True)
    _, d_axis = draw_d_aligned(fig, 12.30, 4.96, records, previous["mechanism"])
    _, f_axis = draw_f_aligned(fig, 12.30, .80, records)
    row_alignment = dict(CD=check_row_alignment(fig, c_axis, d_axis),
                         EF=check_row_alignment(fig, e_axis, f_axis))
    for letter, x, y in [("A", .192, 13.57), ("B", 8.96, 13.57),
                          ("C", .192, 8.97), ("D", 11.58, 8.97),
                          ("E", .192, 4.46), ("F", 11.58, 4.46)]:
        fig.text(x / width, y / height, letter, fontsize=23, fontweight="bold", va="top")
    label_checks["complete"] = check_channel_labels(fig, arr)
    assert dmeta == previous["panel_d"]
    assert fmeta == previous["panel_f"]
    shared.save(fig, FIG / "fig1-complete-layout", dpi=300)
    print("Figure 1 rendered", flush=True)

    canonical_after = {str(p): shared.sha(p) for p in (shared.CANON / "figures").glob("*") if p.is_file()}
    assert canonical_before == canonical_after
    source = shared.old.MASKED_ROOT / "per_subject" / f"{record['dataset']}_{record['subject']}.json"
    metadata = dict(
        status="AUTHOR_SELECTED_Y1_ALIGNED_LAYOUT_PENDING_VISUAL_REVIEW", selection_date="2026-10-09",
        patient_selection="Y1 explicitly selected by author for both C and E",
        producer=str(Path(__file__).resolve()), source_record=str(source),
        input_hashes={str(p): shared.sha(p) for p in [Path(__file__), Path(shared.__file__), source,
                     shared.OUT / "metadata.json", a_path.with_suffix(".png"), A_ROOT / "metadata.json"]},
        patient=PATIENT, n_channels=26, n_events=18190, cluster_counts=counts,
        panels=panel_meta, panel_d=dmeta, panel_f=fmeta,
        all_events_preserved=True, frozen_labels_preserved=True,
        template_uncertainty="mean +/- population SD; unchanged",
        channel_label_checks=label_checks, figure_size_inches=[width, height], row_alignment=row_alignment,
        panel_a=dict(source=str(a_path), version="fig1a_data_recording_chain_kshaft_v3", copied_without_changes=True),
        panel_b={**previous["spectrum"], "display": bmeta,
                 "A_source": str(a_path.with_suffix(".png")),
                 "A_B_link": "Same Y3 and event windows; A shows K1-K12, B preserves its original 10 E/K channels"},
        mechanism=previous["mechanism"],
        old_canonical_assets_preserved=True, human_visual_acceptance="PENDING",
        outputs={str(p.relative_to(OUT)): shared.sha(p) for p in FIG.iterdir() if p.suffix in {".png", ".pdf"}})
    shared.write_json(OUT / "metadata.json", metadata)
    shared.write_json(OUT / "validation.json", dict(status="PASS", patient="Y1", n_channels=26,
        n_events=18190, cluster_counts=counts, same_events_and_channel_order=True,
        channel_labels_do_not_overlap=True, label_checks=label_checks,
        row_alignment=row_alignment, latest_panel_a_exact_copy=True,
        panel_b_labels_restored=True,
        cohort_statistics_unchanged=True, old_canonical_assets_preserved=True,
        human_visual_acceptance="PENDING"))
    descriptions = {
        "a": "逐字节采用最新版recording_chain_kshaft_20261009的A（K杆放大和K1–K12波形）。保留该版脑图、连线、颜色、字体与Time (s)。",
        "b": "保留178段HFO showcase及Y3原三个事件的归一化谱，事件中心前后40 ms的谱值与质心不变。标题恢复为Yuquan Y3，去掉Channel和Event小标题；通道仅写双极对左端电极号，横轴恢复Time (s)，显示原拼接时轴中的三个放大窗并标注断轴。",
        "c": "作者已选择Yuquan Y1；展示全部18,190个有效事件的时间顺序热图及参与事件rank分布。26个通道按既有全事件平均rank排序，保留昼夜条。",
        "d": "沿用已审阅调整版的原始TIFF机制图与40人MI统计。Null点为深灰且不裁切，数据和显著性不变。",
        "e": "同一Y1事件全集按冻结标签分为TA 13,160个、TB 5,030个。通道顺序与C一致，红蓝曲线保留参与事件rank的均值±总体标准差。",
        "f": "沿用40人overall/within-template MI散点及single/multi配对inset。按正常字体比例在整图中绘制，统计数值不变。"}
    lines = ["# Figure 1：作者选定 Y1 的当前修订", "",
             "2026-10-09作者明确选择Y1用于C/E；病例选择已确认，完整拼版待目视检查。旧正式图与其他候选保留。", ""]
    for panel, description in descriptions.items():
        lines += [f"### fig1-panel{panel}.png / .pdf", "", description, "",
                  "**关注点**：C/E事件数和通道顺序一致；26个通道标签清楚，灰色表示未参与。", ""]
    lines += ["### fig1-complete-layout.png / .pdf", "",
              "恢复原先三行版式，A采用最新版K杆图；B收紧间距并恢复短标签。C/D、E/F各行数据轴严格等高且横轴底边对齐，26个通道与全部事件仍保留。", "",
              "**关注点**：检查A版本、B短标签与秒单位，以及两排数据轴等高；不把Y1描述为两模板完全反转。", ""]
    (FIG / "README.md").write_text("\n".join(lines))
    shared.write_json(shared.CANON / "current_revision.json", dict(
        status=metadata["status"], revision_root=str(OUT), selected_patient="Y1",
        panels=["C", "E"], complete_figure=str(FIG / "fig1-complete-layout.pdf"),
        producer=str(Path(__file__).resolve()), metadata=str(OUT / "metadata.json"),
        human_visual_acceptance="PENDING", previous_canonical_retained=True))
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
