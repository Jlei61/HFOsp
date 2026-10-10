#!/usr/bin/env python3
"""Figure 1 readability candidate and complete, existing 40-patient review atlas.

No canonical assets are overwritten. Frozen masked labels/statistics and the
accepted spectrogram kernel are reused; patient choice remains an author choice.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fig1-review-mpl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PathCollection
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

from scripts.paper_figures import plot_fig1_interictal_hfo_temporal_scaffold as old
from scripts.paper_figures.fig1_spectrogram_utils import (
    compute_group_event_spectrogram_stack, centroid_alignment_audit, full_extent_edges,
)
from scripts.paper_figures.patient_public_labels import public_patient_label

CANON = ROOT / "results/paper-ready-figure/fig1"
OUT = CANON / "candidates/readability_review_20261009"
SHORTLIST = ["Y12", "E9", "E18", "E4", "E7"]
COLORS = ["#B2182B", "#2166AC"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def save(fig, stem, dpi=350):
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, facecolor="white")
    plt.close(fig)


def rect_axis(fig, rect, relative):
    x, y, w, h = rect
    a, b, c, d = relative
    return fig.add_axes([x + a*w, y + b*h, c*w, d*h])


def basic_style(ax, size=10):
    ax.tick_params(labelsize=size, length=3, pad=2)
    ax.spines[["top", "right"]].set_visible(False)


def load_spectrum():
    meta = json.loads((CANON / "figures/fig1-panelb2_metadata.json").read_text())
    source = CANON / "candidates/recording_chain_20261009/source/recording_and_geometry.npz"
    z = np.load(source)
    channels = z["channels"].tolist()
    assert channels == meta["selection"]["selected_channels"]
    signals = z["signals_V"]
    fs = meta["plot"]["fs_out"]
    duration = meta["plot"]["window_sec"]
    borders = np.arange(1, 4) * duration
    specs, times, freqs, centers = compute_group_event_spectrogram_stack(signals, fs, borders)
    audit = centroid_alignment_audit(specs, times, freqs, centers, duration, 0.7)
    assert audit["all_centroids_pass"]
    # Verify exact numerical correspondence to the old panel before changing xlim.
    previous = meta["plot"]["centroid_alignment_audit"]["centroids"]
    for row in previous:
        ci, ei = row["channel_index"], row["event_index_within_panel"]
        assert abs(centers[ci, ei, 0] - ei*duration - row["time_within_event_sec"]) < 1e-10
    return dict(meta=meta, source=str(source), source_sha256=sha(source), specs=specs,
                times=times, freqs=freqs, centers=centers, audit=audit)


def draw_spectrum(fig, rect, payload):
    specs, times, freqs, centers = [payload[k] for k in ("specs", "times", "freqs", "centers")]
    nfreq = len(freqs)
    labels = payload["meta"]["selection"]["selected_channels"]
    edges = full_extent_edges(times, 0, 0.96)
    axes = []
    for ei in range(3):
        ax = rect_axis(fig, rect, [0.225+0.239*ei, 0.13, 0.224, 0.75])
        center = 0.16+ei*0.32
        im = ax.pcolormesh((edges-center)*1000, np.arange(specs.shape[0]+1), specs,
                           cmap="coolwarm", vmin=0, vmax=1, rasterized=True)
        for ci in range(1, len(labels)):
            ax.axhline(ci*nfreq, color="0.75", lw=0.4, ls="--")
        ax.axvline(0, color="white", lw=0.6, alpha=0.75)
        x = (centers[:, ei, 0]-center)*1000
        y = np.arange(len(labels))*nfreq + centers[:, ei, 1]+0.5
        ax.plot(x, y, color="#d7191c", lw=1.05, zorder=4)
        ax.scatter(x, y, s=15, facecolor="#ffd166", edgecolor="#d7191c", lw=0.6, zorder=5)
        ax.set(xlim=(-40, 40), ylim=(len(labels)*nfreq, 0), xticks=[-40, 0, 40])
        ax.set_yticks((np.arange(len(labels))+0.5)*nfreq)
        ax.set_yticklabels(labels if ei == 0 else [])
        ax.tick_params(axis="y", length=0, labelsize=8.5)
        ax.tick_params(axis="x", labelsize=9, pad=3)
        ax.get_xticklabels()[0].set_ha("left")
        ax.get_xticklabels()[-1].set_ha("right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"Event {ei+1}", fontsize=10, pad=5)
        if ei == 0:
            ax.set_ylabel("Channel", fontsize=12, labelpad=8)
        axes.append(ax)
    axes[1].set_xlabel("Time from event center (ms)", fontsize=12, labelpad=6)
    cax = rect_axis(fig, rect, [0.961, 0.13, 0.017, 0.75])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 1])
    cb.ax.tick_params(labelsize=9, length=0)
    cb.outline.set_visible(False)
    x, y, w, h = rect
    fig.text(x+w*0.18, y+h*0.985, "Yuquan Y3 · normalized spectrogram", fontsize=11, va="top")


def crop_mi_source():
    source = ROOT / "ReplayIED/tiffs/fig2_prop_hist_9p_画板 1_画板 1.tif"
    # Original Illustrator-exported TIFF; only the illustrative upper part of B.
    box = (317, 1370, 1810, 1705)
    with Image.open(source) as im:
        assert im.size == (3759, 2920)
        crop = im.convert("RGB").crop(box)
    dest = OUT / "source/mi_mechanism_original_tiff.png"
    dest.parent.mkdir(exist_ok=True)
    crop.save(dest)
    return dict(source=str(source), sha256=sha(source), original_size=[3759, 2920],
                crop_pixels=list(box), cropped_size=list(crop.size),
                asset=str(dest), vector_ai_found=False, resampling="none")


def draw_d(fig, rect, records, mechanism):
    # Image shape and axes shape agree: no nonuniform stretch of the old drawing.
    ax_img = rect_axis(fig, rect, [0.00, 0.77, 0.99, 0.23])
    ax_img.imshow(Image.open(mechanism["asset"]), aspect="equal")
    ax_img.axis("off")
    ax = rect_axis(fig, rect, [0.17, 0.17, 0.81, 0.57])
    summary = old._plot_mi(ax, records)
    # The shared helper exposes four subject-scatter collections, in Data/Null order.
    scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(scatters) == 4
    for collection in scatters[1::2]:
        collection.set_facecolor("#343434")
        collection.set_edgecolor("#343434")
        collection.set_alpha(1)
        collection.set_linewidth(0.25)
        collection.set_sizes([14])
        collection.set_zorder(8)
        collection.set_clip_on(False)
    for t in ax.texts:
        if t.get_text() in ("Yuquan", "Epilepsiae"):
            t.set_y(-0.24)
            t.set_fontsize(10)
    ax.set_yticks([0, 0.2, 0.4])
    ax.set_ylabel("MI", fontsize=12)
    basic_style(ax, 10)
    # Signed permutation estimates really can be just below zero. Retain them;
    # the unclipped glyphs straddle the zero line at their exact data coordinate.
    null = np.array([r["legacy_mi"]["permuted_mean_median"] for r in records])
    return dict(statistics=summary, null_min=float(null.min()), null_max=float(null.max()),
                null_points=40, null_points_shifted=False, null_glyphs_clipped=False,
                null_color="#343434", zero_origin_preserved=True)


def draw_f(fig, rect, records):
    ax = rect_axis(fig, rect, [0.18, 0.19, 0.80, 0.77])
    summary = old._plot_uplift(ax, records)
    ax.set_xlabel("Overall MI", fontsize=12)
    ax.set_ylabel("Within-template MI", fontsize=12)
    ax.set_xticks([0, 0.4, 0.8])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    basic_style(ax, 10)
    # Physical axes, labels, and inset are rendered in the final canvas.
    ax.set_aspect("equal", adjustable="box")
    summary["dataset_legend_frame"]["rendered_fontsize_points"] = 7.5
    return summary


def draw_ce(fig, rect, arr, label, clustered=False):
    heat = rect_axis(fig, rect, [0.080, 0.19, 0.702, 0.68])
    cax = rect_axis(fig, rect, [0.804, 0.19, 0.013, 0.68])
    profile = rect_axis(fig, rect, [0.878, 0.19, 0.112, 0.68])
    order = arr["channel_order"]
    events = arr["clustered_events_all"] if clustered else arr["valid_events"]
    im = old.propagation_plot._plot_rank_heatmap(
        heat, arr["ranks"][order][:, events], arr["ordered_names"], "",
        display_bools=arr["bools"][order][:, events], ytick_fontsize=9.5, xtick_fontsize=9)
    heat.set_xlim(0, len(events))
    heat.set_ylim(0, len(order))
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0, len(order)-1])
    cb.ax.tick_params(labelsize=9, length=2)
    cb.ax.set_title("Rank\nFirst → Last", fontsize=9, pad=7)
    if clustered:
        boundary = int(sum(arr["labels"] == 0))
        # Draw a boundary line only: retain every event column, including those
        # next to the boundary; no artificial missing-data gap.
        heat.axvline(boundary, color="white", lw=1.5)
        for i, (lo, hi) in enumerate(((0, boundary), (boundary, len(events)))):
            heat.text((lo+hi)/2, len(order)*1.035, f"T{'AB'[i]} (n={hi-lo:,})",
                      color=COLORS[i], fontsize=11, fontweight="bold", ha="center", va="bottom")
        old.propagation_plot._plot_cluster_rank_fig4(
            profile, arr["ranks"], arr["bools"], arr["valid_events"], arr["labels"],
            order, arr["channel_names"], "", show_ylabels=False, show_legend=False,
            invert_yaxis=False, line_colors=COLORS, label_names=["TA", "TB"],
            marker_size=3.2, xtick_fontsize=9, label_fontsize=12)
        heat.set_xlabel("Population events (clustered)", fontsize=12)
    else:
        strip = rect_axis(fig, rect, [0.080, 0.135, 0.702, 0.029])
        old.propagation_plot._plot_daynight_strip(strip, arr["day_mask"])
        strip.set_xlim(-0.5, len(events)-0.5)
        strip.set_xlabel("Population events (time-ordered)", fontsize=12)
        heat.tick_params(axis="x", bottom=False, labelbottom=False)
        heat.text(0, 1.045, f"{label}  |  n={len(events):,}", transform=heat.transAxes,
                  fontsize=11, fontweight="bold", va="bottom")
        heat.legend(handles=[Patch(facecolor="white", edgecolor="black", label="Day"),
                             Patch(facecolor="black", label="Night")],
                    loc="lower right", bbox_to_anchor=(0.92, 1.025), ncol=2,
                    frameon=False, fontsize=9, borderaxespad=0, handlelength=1)
        old.propagation_plot._plot_rank_histogram(
            profile, arr["ranks"], arr["bools"], arr["valid_events"], order,
            arr["channel_names"], "", show_ylabels=False, label_fontsize=12, xtick_fontsize=9)
    basic_style(profile, 9)
    profile.set_xticks([0, (len(order)-1)//2, len(order)-1])
    profile.set_xlabel("Rank", fontsize=12)
    return dict(patient=label, n_events=int(len(events)), all_valid_events_displayed=True,
                channel_order=arr["ordered_names"], masked=True,
                cluster_counts=[int(sum(arr["labels"]==i)) for i in range(2)],
                profile_band="mean +/- population SD of participating-event ranks")


def prepare_atlas(records):
    rows, pages = [], []
    atlas = OUT / "all_patients"
    atlas.mkdir(exist_ok=True)
    for record in records:
        label = public_patient_label(record["dataset"], record["subject"])
        adaptive = record["adaptive_cluster"]
        clusters = adaptive["clusters"]
        corr = adaptive["inter_cluster_corr_matrix"]
        source = old.MASKED_ROOT / f"figures/per_subject/{record['dataset']}_{record['subject']}_propagation.png"
        row = dict(patient=label, dataset=record["dataset"], channels=record["n_channels"],
                   n_events=adaptive["n_valid_events"], k=adaptive["chosen_k"],
                   tau_A=clusters[0]["raw_tau"], tau_B=clusters[1]["raw_tau"],
                   weakest_cluster_tau=min(c["raw_tau"] for c in clusters),
                   template_spearman=(corr[0][1] if adaptive["chosen_k"]==2 else None),
                   reproducibility=record.get("time_split_reproducibility",{}).get("reproducibility_grade"),
                   existing_figure=str(source), status="AVAILABLE" if source.exists() else "MISSING",
                   selected_for_review=label in SHORTLIST)
        rows.append(row)
    rows.sort(key=lambda r:(r["dataset"], int(r["patient"][1:])))
    with (OUT / "patient_inventory.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with PdfPages(OUT / "figures/all_40_patients_existing.pdf") as pdf:
        for row in rows:
            if row["status"] != "AVAILABLE": continue
            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_axes([0.015, 0.015, 0.97, 0.90])
            with Image.open(row["existing_figure"]) as img:
                ax.imshow(img)
            ax.axis("off")
            fig.text(0.025, 0.96, f"{row['patient']}  ·  existing masked result  ·  K={row['k']}  ·  n={row['n_events']:,}", fontsize=16)
            fig.text(0.025, 0.925, "Original diagnostic figure; sampled heatmap counts and legacy colors are retained. No re-clustering.", fontsize=10)
            pdf.savefig(fig, dpi=200)
            fig.savefig(atlas / f"{row['patient']}.png", dpi=140)
            plt.close(fig)
            pages.append(f'<article id="{row["patient"]}"><h2>{row["patient"]}</h2><img loading="lazy" src="all_patients/{row["patient"]}.png"></article>')
    nav = " · ".join(f'<a href="#{r["patient"]}">{r["patient"]}</a>' for r in rows)
    (OUT / "all_patients.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Figure 1 患者图册</title><style>body{font-family:sans-serif;max-width:1500px;margin:30px auto;padding:20px}img{width:100%}nav{position:sticky;top:0;background:white;padding:16px;line-height:1.8}article{scroll-margin-top:100px}</style><h1>40 位患者的既有 masked 图</h1><p>保留全部患者，包括 K>2；旧诊断图保留原有抽样、计数、颜色。候选重绘另用完整事件和当前 TA/TB 配色。</p><nav>'+nav+'</nav>'+''.join(pages)+'</html>')
    return rows


def image_in_rect(fig, rect, path):
    ax = fig.add_axes(rect)
    ax.imshow(Image.open(path), aspect="equal")
    ax.axis("off")


def review_outputs(inventory):
    figures = OUT / "figures"
    with PdfPages(figures / "shortlist_CE.pdf") as pdf:
        for label in SHORTLIST:
            fig = plt.figure(figsize=(11.6, 7.6))
            row = next(r for r in inventory if r["patient"] == label)
            image_in_rect(fig, [0, 0, 1, 0.92], figures / f"CE-{label}.png")
            fig.text(0.05, 0.96,
                     f"{label}   within-template concordance: {float(row['tau_A']):.3f} / {float(row['tau_B']):.3f}"
                     f"   |   template correlation: {float(row['template_spearman']):.3f}", fontsize=11)
            pdf.savefig(fig, dpi=250)
            plt.close(fig)
    fig = plt.figure(figsize=(20, 13))
    for i, label in enumerate(SHORTLIST[:4]):
        row = next(r for r in inventory if r["patient"] == label)
        x, y = (i % 2)*0.5, 0.5-(i//2)*0.5
        image_in_rect(fig, [x, y, 0.5, 0.45], figures / f"CE-{label}.png")
        fig.text(x+0.03, y+0.47,
                 f"{label}  |  concordance {float(row['tau_A']):.2f} / {float(row['tau_B']):.2f}"
                 f"  |  correlation {float(row['template_spearman']):.2f}", fontsize=13, weight="bold")
    save(fig, figures / "shortlist_comparison", dpi=180)
    descriptions = {
        "all_40_patients_existing": "40 位患者已有的 masked 诊断图，每人一页；包括 K>2 的病例，未删去弱结果。旧图保留当时的事件抽样、颜色和标签，不能与候选重绘的全量事件数直接混用。",
        "shortlist_CE": "Y12、E9、E18、E4 及原 E7 的同版式 C/E 比较。每位患者的 C/E 使用同一全量有效事件集、同一通道顺序及冻结 adaptive labels；右侧保留参与事件 rank 的均值±总体标准差。",
        "shortlist_comparison": "四位候选的 C/E 总览，供作者选择，不构成新的群体统计检验。数字为既有两类内部 rank 一致性和模板相关性；选例用于说明现象，不改变 D/F 的40人结果。",
        "fig1-panelb2-zoom": "保留 Y3 原来的三个事件22、237、1458及10个双极通道，分别放大到各事件中心前后40 ms。使用原50 ms Hamming窗、10 ms步长、Gaussian sigma=1.5幅度谱、同一归一化和同一红色质心；放大不增加原始时间分辨率。",
        "fig1-panelb": "左侧保留178段HFO showcase，右侧为同一Y3患者的独立通道标记和事件中心放大谱。三个小谱窗是不同事件，不应读成连续的80 ms片段。",
        "fig1-paneld": "上方从旧Figure 2的原始Illustrator导出TIFF按原像素裁取MI机制图，尚未找到AI矢量源。下方沿用40位患者原统计与显著性，Null点改为深灰且关闭边界裁切；包括略负的估计，未平移数值。",
        "fig1-panelf": "原40位患者overall/within-template MI散点及配对inset直接在最终尺寸坐标轴重绘。保持数据、配对线、均值柱和显著性，字体按正常比例渲染。",
    }
    lines = ["# Figure 1 版式与患者选择候选", "", "状态：候选，等待作者选择C/E病例及目视检查；正式图未覆盖。A仅复用同日另一任务已经生成的Y3候选，不修改A的制作脚本。", ""]
    stems = sorted({p.stem for p in figures.iterdir() if p.suffix in {".png", ".pdf"}})
    for stem in stems:
        text = descriptions.get(stem)
        if stem.startswith("CE-"):
            text = f"{stem[3:]} 的时间顺序热图及双模板重排图。全部有效事件均入图，空白/浅灰为未参与通道，未缩窄标准差。"
        if stem.startswith("fig1-complete-layout-"):
            text = f"完整A–F候选拼版，C/E临时使用{stem.split('-')[-1]}。A为独立制作中的已有候选，B为放大谱，D为原始TIFF机制图与深色Null点，F按自然比例重绘。"
        extensions = " / ".join(p.suffix for p in figures.glob(stem+".*") if p.suffix in {".png", ".pdf"})
        lines += [f"### {stem}{extensions}", "", text or "本次候选图输出。", "", "**关注点**：检查通道与事件对应、时间刻度、模板间差异及最终画布的文字可读性；尚未通过人工验收。", ""]
    (figures / "README.md").write_text("\n".join(lines))
    for label in SHORTLIST:
        (OUT / "per_subject" / label / "figures" / "README.md").write_text(
            f"### fig1-panelc.png / .pdf\n\n{label}全部有效事件按时间排列，右侧为参与事件的rank分布。\n\n**关注点**：通道顺序与E相同，昼夜条沿用原时间定义。\n\n"
            f"### fig1-panele.png / .pdf\n\n同一{label}事件全集按冻结TA/TB标签分组，右侧为均值±总体标准差。\n\n**关注点**：两类计数之和等于C的总事件数，保留未参与通道空白。\n")


def compose(arr, label, payload, records, mechanism):
    fig = plt.figure(figsize=(16, 12.5))
    # A is supplied by another active task. Only reuse its existing candidate.
    a_path = CANON / "candidates/recording_chain_20261009/figures/fig1-panela.png"
    image_in_rect(fig, [0.025, 0.61, 0.535, 0.35], a_path)
    image_in_rect(fig, [0.575, 0.61, 0.112, 0.35], CANON / "figures/fig1-panelb1.png")
    draw_spectrum(fig, [0.69, 0.61, 0.30, 0.35], payload)
    draw_ce(fig, [0.015, 0.30, 0.727, 0.255], arr, label)
    draw_d(fig, [0.75, 0.29, 0.238, 0.275], records, mechanism)
    draw_ce(fig, [0.015, 0.025, 0.727, 0.255], arr, label, True)
    draw_f(fig, [0.75, 0.015, 0.238, 0.28], records)
    for letter, x, y in [("A",.012,.97),("B",.563,.97),("C",.012,.565),
                          ("D",.75,.575),("E",.012,.29),("F",.75,.30)]:
        fig.text(x,y,letter,fontsize=23,fontweight="bold",va="top")
    save(fig, OUT / f"figures/fig1-complete-layout-{label.split()[-1]}", dpi=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-atlas", action="store_true")
    args = parser.parse_args()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "per_subject").mkdir(exist_ok=True)
    canonical_hashes = {str(p):sha(p) for p in (CANON / "figures").glob("*") if p.is_file()}
    plt.rcParams.update({"font.family":"DejaVu Sans", "pdf.fonttype":42, "svg.fonttype":"none", "axes.unicode_minus":False})
    old.propagation_plot._apply_masked_paths()
    records = old._load_temporal_records()
    old._assert_masked_mi_records(records)
    assert len(records) == 40
    inventory = prepare_atlas(records) if not args.skip_atlas else list(csv.DictReader((OUT / "patient_inventory.csv").open()))
    print("atlas ready", flush=True)
    spectrum = load_spectrum()
    fig=plt.figure(figsize=(6.3,4.8));draw_spectrum(fig,[0,0,1,1],spectrum)
    save(fig,OUT / "figures/fig1-panelb2-zoom")
    fig=plt.figure(figsize=(8.2,4.8))
    image_in_rect(fig,[0.0,0.0,0.24,1],CANON / "figures/fig1-panelb1.png")
    draw_spectrum(fig,[0.25,0,0.75,1],spectrum)
    save(fig,OUT / "figures/fig1-panelb")
    mechanism = crop_mi_source()
    fig=plt.figure(figsize=(4.5,4.2));dmeta=draw_d(fig,[0,0,1,1],records,mechanism)
    save(fig,OUT / "figures/fig1-paneld")
    fig=plt.figure(figsize=(4.5,4.2));fmeta=draw_f(fig,[0,0,1,1],records)
    save(fig,OUT / "figures/fig1-panelf")
    by_label = {public_patient_label(r["dataset"],r["subject"]):r for r in records}
    patient_meta={}
    for label in SHORTLIST:
        print("render",label,flush=True)
        record=by_label[label]
        arr=old._load_exemplar_arrays(record,max_events=10**9)
        assert arr["channel_names"] == record["channel_names"], "Channel union mismatch"
        assert len(arr["valid_events"]) == record["adaptive_cluster"]["n_valid_events"]
        assert sum(c["n_events"] for c in record["adaptive_cluster"]["clusters"]) == len(arr["valid_events"])
        display=f"{record['dataset'].capitalize()} {label}"
        output=OUT / "per_subject" / label / "figures"
        output.mkdir(parents=True, exist_ok=True)
        entries={}
        for clustered,panel in [(False,"c"),(True,"e")]:
            fig=plt.figure(figsize=(11.6,3.5))
            entries[panel]=draw_ce(fig,[0,0,1,1],arr,display,clustered)
            save(fig, output / f"fig1-panel{panel}")
        fig=plt.figure(figsize=(11.6,7.0))
        draw_ce(fig,[0,.5,1,.5],arr,display)
        draw_ce(fig,[0,0,1,.5],arr,display,True)
        save(fig,OUT / f"figures/CE-{label}",dpi=250)
        if label in ["Y12","E7"]:compose(arr,display,spectrum,records,mechanism)
        patient_meta[label]=entries
    review_outputs(inventory)
    after={str(p):sha(p) for p in (CANON / "figures").glob("*") if p.is_file()}
    assert canonical_hashes==after
    write_json(OUT / "metadata.json",dict(
        status="CANDIDATE_PENDING_AUTHOR_SELECTION_AND_VISUAL_REVIEW",
        producer=str(Path(__file__).resolve()), formal_current_figure_replaced=False,
        canonical_assets_unchanged=True, n_patients=40, shortlist=SHORTLIST,
        input_hashes={str(p):sha(p) for p in [Path(__file__),
            Path(old.__file__), CANON / "figure1_panel_metadata.json",
            CANON / "candidates/recording_chain_20261009/figures/fig1-panela.png"]},
        selection_purpose="Illustration clarity only; cohort estimates unchanged; no inferential selection claim",
        spectrum=dict(source=spectrum["source"],source_sha256=spectrum["source_sha256"],
            events=spectrum["meta"]["selection"]["event_indices"],channels=spectrum["meta"]["selection"]["selected_channels"],
            xlim_ms=[-40,40], original_window_ms=320, magnification=4,
            same_spectrogram_and_centroids_verified=True, centroid_audit=spectrum["audit"],
            centroid_span_ms=(1000*np.ptp(spectrum["centers"][:,:,0],axis=0)).tolist(),
            time_reference="center of each original 320 ms window; three separate events, not continuous time",
            A_source="existing recording_chain_20261009 candidate; independently produced, not modified"),
        mechanism=mechanism,panel_d=dmeta,panel_f=fmeta,patients=patient_meta))
    print("DONE",OUT,flush=True)


if __name__ == "__main__":
    main()
