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

plt = shared.plt
OUT = shared.CANON / "candidates/selected_y1_20261009"
FIG = OUT / "figures"
PATIENT = "Y1"


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

    # More height for 26 channels; the original font shapes and data stay intact.
    panel_meta, label_checks = {}, {}
    for clustered, panel in [(False, "c"), (True, "e")]:
        fig = plt.figure(figsize=(11.6, 5.6))
        panel_meta[panel] = shared.draw_ce(fig, [0, 0, 1, 1], arr, label, clustered)
        label_checks[panel] = check_channel_labels(fig, arr)
        shared.save(fig, FIG / f"fig1-panel{panel}", dpi=300)

    assets = {"a": shared.CANON / "candidates/recording_chain_20261009/figures/fig1-panela",
              "b": shared.OUT / "figures/fig1-panelb",
              "d": shared.OUT / "figures/fig1-paneld",
              "f": shared.OUT / "figures/fig1-panelf"}
    for panel, stem in assets.items():
        for suffix in (".png", ".pdf"):
            shutil.copy2(stem.with_suffix(suffix), FIG / f"fig1-panel{panel}{suffix}")

    # Physical-inch coordinates keep A/B/D/F at their reviewed sizes while
    # allocating extra vertical room to the two 26-channel heatmaps.
    width, height = 16, 17.325
    def rect(x, y, w, h):
        return [x / width, y / height, w / width, h / height]
    fig = plt.figure(figsize=(width, height))
    shared.image_in_rect(fig, rect(.4, 12.45, 8.56, 4.375), assets["a"].with_suffix(".png"))
    shared.image_in_rect(fig, rect(9.2, 12.45, 1.792, 4.375), shared.CANON / "figures/fig1-panelb1.png")
    spectrum = shared.load_spectrum()
    shared.draw_spectrum(fig, rect(11.04, 12.45, 4.8, 4.375), spectrum)
    shared.draw_ce(fig, rect(.24, 6.1625, 11.632, 5.6), arr, label)
    shared.draw_ce(fig, rect(.24, .3125, 11.632, 5.6), arr, label, True)
    dmeta = shared.draw_d(fig, rect(12, 8.45, 3.808, 3.4375), records, previous["mechanism"])
    fmeta = shared.draw_f(fig, rect(12, 2.6, 3.808, 3.5), records)
    for letter, x, y in [("A", .192, 16.95), ("B", 9.008, 16.95),
                          ("C", .192, 11.8875), ("D", 12, 11.8875),
                          ("E", .192, 6.0375), ("F", 12, 6.0375)]:
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
        status="AUTHOR_SELECTED_Y1_LAYOUT_PENDING_VISUAL_REVIEW", selection_date="2026-10-09",
        patient_selection="Y1 explicitly selected by author for both C and E",
        producer=str(Path(__file__).resolve()), source_record=str(source),
        input_hashes={str(p): shared.sha(p) for p in [Path(__file__), Path(shared.__file__), source,
                     shared.OUT / "metadata.json", assets["a"].with_suffix(".png")]},
        patient=PATIENT, n_channels=26, n_events=18190, cluster_counts=counts,
        panels=panel_meta, panel_d=dmeta, panel_f=fmeta,
        all_events_preserved=True, frozen_labels_preserved=True,
        template_uncertainty="mean +/- population SD; unchanged",
        channel_label_checks=label_checks, figure_size_inches=[width, height],
        panel_a="Reused reviewed recording_chain_20261009 asset; author is producing A separately",
        panel_b=previous["spectrum"], mechanism=previous["mechanism"],
        old_canonical_assets_preserved=True, human_visual_acceptance="PENDING",
        outputs={str(p.relative_to(OUT)): shared.sha(p) for p in FIG.iterdir() if p.suffix in {".png", ".pdf"}})
    shared.write_json(OUT / "metadata.json", metadata)
    shared.write_json(OUT / "validation.json", dict(status="PASS", patient="Y1", n_channels=26,
        n_events=18190, cluster_counts=counts, same_events_and_channel_order=True,
        channel_labels_do_not_overlap=True, label_checks=label_checks,
        cohort_statistics_unchanged=True, old_canonical_assets_preserved=True,
        human_visual_acceptance="PENDING"))
    descriptions = {
        "a": "复用同日 recording_chain_20261009 的 Y3 记录链素材。A由作者另行制作，本次仅将已有素材放入整图。",
        "b": "保留178段HFO showcase及Y3三个事件的归一化谱，分别放大到事件中心前后40 ms。谱图带独立通道标签，沿用原谱值和质心。",
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
              "将Y1的C/E接入此前A–F调整版，并增加C/E纵向空间容纳26个通道。A/B仍为此前Y3素材，C/E的Y1身份在图中独立标明。", "",
              "**关注点**：检查完整拼版的通道名称和rank曲线；本版不把Y1描述为两模板完全反转。", ""]
    (FIG / "README.md").write_text("\n".join(lines))
    shared.write_json(shared.CANON / "current_revision.json", dict(
        status=metadata["status"], revision_root=str(OUT), selected_patient="Y1",
        panels=["C", "E"], complete_figure=str(FIG / "fig1-complete-layout.pdf"),
        producer=str(Path(__file__).resolve()), metadata=str(OUT / "metadata.json"),
        human_visual_acceptance="PENDING", previous_canonical_retained=True))
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
