#!/usr/bin/env python3
"""Author review of every Yuquan chronological heatmap and frozen rank modes.

Reuses Figure 1's masked data loader and original rank plotting helpers. This
is descriptive exemplar selection, not a new cohort test or a re-clustering.
"""
from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures import build_fig1_readability_review as shared
from scripts.paper_figures.patient_public_labels import public_patient_label
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = shared.CANON / "candidates/yuquan_patient_review_20261009"
FIG = OUT / "figures"
PALETTE = ["#B2182B", "#2166AC", "#1B9E77", "#D95F02", "#7570B3", "#6B593E"]
PRIORITY = ["Y16", "Y2", "Y1", "Y18", "Y12"]


def draw_clusters(fig, rect, arr):
    if arr["chosen_k"] == 2:
        return shared.draw_ce(fig, rect, arr, "", clustered=True)
    # Keep all frozen modes for K>2 rather than forcing a TA/TB illustration.
    plot = shared.old.propagation_plot
    heat = shared.rect_axis(fig, rect, [.080, .19, .702, .68])
    cax = shared.rect_axis(fig, rect, [.804, .19, .013, .68])
    profile = shared.rect_axis(fig, rect, [.878, .19, .112, .68])
    order, events = arr["channel_order"], arr["clustered_events_all"]
    im = plot._plot_rank_heatmap(heat, arr["ranks"][order][:, events],
        arr["ordered_names"], "", display_bools=arr["bools"][order][:, events],
        ytick_fontsize=9.5, xtick_fontsize=9)
    heat.set(xlim=(0, len(events)), ylim=(0, len(order)))
    cb = fig.colorbar(im, cax=cax, ticks=[0, len(order)-1])
    cb.ax.tick_params(labelsize=9, length=2)
    cb.ax.set_title("Rank\nFirst → Last", fontsize=9, pad=7)
    ids, counts = np.unique(arr["labels"], return_counts=True)
    names = [f"T{i+1}" for i in range(len(ids))]
    boundary = 0
    for name, count, color in zip(names, counts, PALETTE):
        heat.text(boundary+count/2, len(order)*1.035, f"{name} (n={count:,})",
                  color=color, fontsize=9, weight="bold", ha="center", va="bottom")
        boundary += int(count)
        if boundary < len(events): heat.axvline(boundary, color="white", lw=1.5)
    heat.set_xlabel("Population events (clustered)", fontsize=12)
    plot._plot_cluster_rank_fig4(profile, arr["ranks"], arr["bools"],
        arr["valid_events"], arr["labels"], order, arr["channel_names"], "",
        show_ylabels=False, show_legend=False, invert_yaxis=False,
        line_colors=PALETTE[:len(ids)], label_names=names, marker_size=3,
        xtick_fontsize=9, label_fontsize=12)
    profile.set_xticks([0, (len(order)-1)//2, len(order)-1])
    shared.basic_style(profile, 9)
    return dict(n_events=int(len(events)), cluster_counts=counts.tolist(),
                channel_order=arr["ordered_names"], all_valid_events_displayed=True,
                masked=True, k=int(len(ids)))


def collect_row(record, arr, label):
    a = record["adaptive_cluster"]
    idx = arr["valid_events"]
    ranks = arr["ranks"][:, idx]
    mask = arr["bools"][:, idx] & np.isfinite(ranks)
    sds = [float(np.std(r[m])) for r, m in zip(ranks, mask) if m.any()]
    n_channels = len(arr["channel_names"])
    return dict(patient=label, n_events=len(idx), n_channels=n_channels,
        k=a["chosen_k"], overall_pairwise_tau=a["overall_tau"],
        within_mode_tau=[c["raw_tau"] for c in a["clusters"]],
        cluster_counts=[c["n_events"] for c in a["clusters"]],
        template_spearman=a["inter_cluster_corr_matrix"][0][1] if a["chosen_k"]==2 else None,
        median_channel_rank_sd_fraction=float(np.median(sds)/max(1,n_channels-1)),
        observed_channel_event_fraction=float(mask.mean()),
        reproducibility=record["time_split_reproducibility"]["reproducibility_grade"],
        source_record=str(shared.old.MASKED_ROOT / f"per_subject/yuquan_{record['subject']}.json"),
        channel_names=arr["channel_names"], display_channel_order=arr["ordered_names"])


def contact_sheet(labels, dest, full_ce=False):
    # Contain every source image uniformly; never distort font or axis aspect.
    width, row_height = 2600, (1450 if full_ce else 790)
    canvas = Image.new("RGB", (width, row_height*len(labels)), "white")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 37)
    draw = ImageDraw.Draw(canvas)
    for i,label in enumerate(labels):
        name = "CE" if full_ce else "chronological"
        source = FIG / "per_patient" / f"{label}-{name}.png"
        with Image.open(source) as im:
            im = im.convert("RGB")
            im.thumbnail((width-30, row_height-55), Image.Resampling.LANCZOS)
            canvas.paste(im, ((width-im.width)//2, i*row_height+50))
        draw.text((24,i*row_height+8), label, fill="#222222", font=font)
    canvas.save(dest, dpi=(200,200))


def finish_outputs(rows):
    labels = [r["patient"] for r in rows]
    with PdfPages(FIG / "yuquan_all20_chronological.pdf") as pdf:
        for label in labels:
            with Image.open(FIG / "per_patient" / f"{label}-chronological.png") as im:
                fig=plt.figure(figsize=(14, 14*im.height/im.width))
                shared.image_in_rect(fig,[0,0,1,1],im.filename)
                pdf.savefig(fig,dpi=250)
                plt.close(fig)
    sources = [str(FIG / "per_patient" / f"{label}-CE.pdf") for label in labels]
    subprocess.run(["pdfunite",*sources,str(FIG / "yuquan_all20_CE.pdf")],check=True)
    subprocess.run(["pdfunite",*[str(FIG / "per_patient" / f"{label}-CE.pdf") for label in PRIORITY],
                    str(FIG / "yuquan_priority_CE.pdf")],check=True)
    for start in range(0,20,5):
        subset=labels[start:start+5]
        contact_sheet(subset,FIG / f"chronological_{subset[0]}_{subset[-1]}.png")
    contact_sheet(PRIORITY,FIG / "priority_chronological.png")
    links=" · ".join(f'<a href="#{l}">{l}</a>' for l in labels)
    articles=[]
    for row in rows:
        label=row["patient"]
        articles.append(f'<article id="{label}"><h2>{label} · {row["n_channels"]} channels · '
                        f'n={row["n_events"]:,} · K={row["k"]}</h2><a href="figures/per_patient/{label}-CE.pdf">'
                        f'PDF 放大</a><img loading="lazy" src="figures/per_patient/{label}-CE.png"></article>')
    (OUT / "index.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Yuquan 全20人间期热图</title>'
        '<style>body{font-family:sans-serif;margin:30px auto;max-width:1600px;padding:20px}img{width:100%}nav{position:sticky;top:0;background:white;padding:15px;line-height:1.8}article{scroll-margin-top:100px}</style>'
        '<h1>Yuquan 全20人间期热图</h1><p>上排：全部有效事件按时间排列；下排：同一事件全集按冻结模板标签排列。'
        '灰色为未参与通道；右侧保留原rank分布和均值±标准差。K>2病例保留全部模板，不改成两类。</p><nav>'
        +links+'</nav>'+''.join(articles)+'</html>')
    descriptions={
        "yuquan_all20_chronological.pdf":"Y1至Y20各一页，专门比较未分组热图中的横向条带与右侧总体rank分布。所有有效事件均入图，通道按全数据的参与事件平均rank排序。",
        "yuquan_all20_CE.pdf":"Y1至Y20各一页，上排时间顺序热图，下排冻结模式重排热图。保留所有患者，包括K>2的Y4、Y10、Y11、Y17；未强制归成TA/TB。",
        "yuquan_priority_CE.pdf":"Y16、Y2、Y1、Y18与Y12的逐页比较。用于在整体条带连续性与模板区分度之间选例，不改变群体统计或重新聚类。",
        "priority_chronological.png":"Y16、Y2、Y1、Y18与Y12的未分组热图对照。保持真实缺失和总体rank分布，不筛选整齐事件。"}
    for start in range(1,21,5):
        descriptions[f"chronological_Y{start}_Y{start+4}.png"]=f"Y{start}至Y{start+4}的未分组热图总览。图像均按比例缩放；通道多的病例可在逐页PDF中放大查看。"
    text=["# Yuquan 患者间期热图审阅", "", "全20人；候选审阅，未替换Figure 1。", ""]
    for filename,desc in descriptions.items():
        text += [f"### {filename}","",desc,"","**关注点**：先看全程横向色带和rank分布宽度，再看双模板差异；保留参与缺失与标准差。",""]
    (FIG / "README.md").write_text("\n".join(text))
    text=[]
    for row in rows:
        l=row["patient"]
        text += [f"### {l}-chronological.png / .pdf","",f"{l}全部{row['n_events']:,}个有效事件的时间顺序热图及通道rank分布。浅灰为未参与通道。","","**关注点**：颜色代表事件内早晚顺序，列为按时间排序的事件，并非等间隔时间采样。","",
                 f"### {l}-CE.png / .pdf","",f"上排同一{l}时间顺序热图，下排按冻结K={row['k']}标签分组。右侧保留参与事件rank的均值±总体标准差。","","**关注点**：两排同通道顺序、同事件全集；各类计数之和必须等于总数。",""]
    (FIG / "per_patient/README.md").write_text("\n".join(text))


def main():
    (FIG / "per_patient").mkdir(parents=True,exist_ok=True)
    shared.old.propagation_plot._apply_masked_paths()
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"axes.unicode_minus":False})
    records=[r for r in shared.old._load_temporal_records() if r["dataset"]=="yuquan"]
    records.sort(key=lambda r:int(public_patient_label("yuquan",r["subject"])[1:]))
    assert len(records)==20
    before={str(p):shared.sha(p) for p in (shared.CANON / "figures").iterdir() if p.is_file()}
    rows=[]
    for record in records:
        label=public_patient_label("yuquan",record["subject"])
        print("render",label,flush=True)
        arr=shared.old._load_exemplar_arrays(record,max_events=10**9)
        assert arr["channel_names"]==record["channel_names"]
        assert len(arr["valid_events"])==record["adaptive_cluster"]["n_valid_events"]
        assert sum(c["n_events"] for c in record["adaptive_cluster"]["clusters"])==len(arr["valid_events"])
        assert len(arr["day_mask"])==len(arr["valid_events"])
        height=max(3.9,.20*len(arr["channel_names"]))
        fig=plt.figure(figsize=(14,height))
        shared.draw_ce(fig,[0,0,1,1],arr,f"Yuquan {label}")
        shared.save(fig,FIG / "per_patient" / f"{label}-chronological",dpi=250)
        fig=plt.figure(figsize=(14,2*height))
        shared.draw_ce(fig,[0,.5,1,.5],arr,f"Yuquan {label}")
        drawn=draw_clusters(fig,[0,0,1,.5],arr)
        assert sum(drawn["cluster_counts"])==len(arr["valid_events"])
        shared.save(fig,FIG / "per_patient" / f"{label}-CE",dpi=250)
        rows.append(collect_row(record,arr,label))
        del arr
        gc.collect()
    shared.write_json(OUT / "patient_inventory.json",rows)
    with (OUT / "patient_inventory.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    finish_outputs(rows)
    after={str(p):shared.sha(p) for p in (shared.CANON / "figures").iterdir() if p.is_file()}
    assert before==after
    shared.write_json(OUT / "metadata.json",dict(status="CANDIDATE_FOR_AUTHOR_SELECTION",
        human_visual_acceptance="PENDING",formal_current_figure_replaced=False,n_patients=20,
        all_events=True,all_channels=True,refit_clusters=False,mean_rank_bands="population SD, not SEM",
        original_outputs_unchanged=True,producer=str(Path(__file__).resolve()),
        producer_sha256=shared.sha(__file__),shared_renderer_sha256=shared.sha(shared.__file__),
        review_priority=PRIORITY,statistical_contract=dict(
            overall_tau="existing mean pairwise Kendall rank concordance across valid events within a patient",
            within_tau="same readout within each frozen cluster; event pairs, not independent patients",
            channel_sd="per-channel population SD over participating events, divided by n_channels-1; median across channels",
            comparison="descriptive patient-exemplar comparison only; no cohort inference",
            caution="rank ranges, participation and mode mixtures differ among patients")))
    print("DONE",OUT,flush=True)


if __name__=="__main__":
    main()
