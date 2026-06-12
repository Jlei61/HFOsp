"""Plot the MODEL's pooled spontaneous events with the EXACT real-data per-subject figure
(user 2026-06-11): the model goes through the same lagPat + same KMeans + same plot as real
patients, so model and data are directly comparable in one figure format.

This is the visual-parity capstone of "model through the real pipeline". It reuses
scripts/plot_interictal_propagation.py::plot_pr3_subject_figure unchanged — only the subject
resolver, the figure output dir, and the README hook are redirected to the model's pooled record.

HONEST FRAMING (locked): this is INSTRUMENT parity, NOT mechanism discovery — a single connectivity
axis makes the template space ~1-D so stable_k=2 is partly forced. The figure is labelled "model:".
The pipeline's inter-cluster corr shown on the figure is the masked-IMPUTED value (~+0.44, and
candidate_forward_reverse=none) — the same metric real subjects get; the model's TRUE opposition is
−0.945 on shared contacts / rank-displacement swap=strict (see caption + masked_pipeline_summary).
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import plot_interictal_propagation as P                                       # noqa: E402
from src.interictal_propagation import (load_subject_propagation_events,      # noqa: E402
                                        compute_adaptive_cluster_stereotypy)

MODEL_OUT = Path("results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous")
POOLED = MODEL_OUT / "pooled_bidir"
FIGDIR = MODEL_OUT / "figures" / "model_propagation"


def _make_record(tag):
    sub_dir = POOLED / tag
    ev = load_subject_propagation_events(sub_dir)
    R = np.asarray(ev["ranks"], float); B = np.asarray(ev["bools"], bool)
    names = list(ev["channel_names"])
    pr2 = compute_adaptive_cluster_stereotypy(R, B, names, use_masked_features=True)
    # reproducibility grade from the masked-pipeline summary if present (else unknown)
    grade = "unknown"
    import json
    mp = sub_dir.parent.parent / "pooled_bidir" / tag / "masked_pipeline_summary.json"
    if mp.exists():
        s = json.load(open(mp))
        repro = s.get("pr25", {}).get("forward_reverse_reproduced")
        grade = "strong" if (isinstance(repro, dict) and any(repro.values())) else "moderate"
    return dict(dataset="model", subject=tag, adaptive_cluster=pr2,
                propagation_stereotypy={"all": {"mean_tau": pr2.get("overall_tau", float("nan"))}},
                time_split_reproducibility={"reproducibility_grade": grade})


def main():
    tags = sys.argv[1:] or ["stage2_main_bidir", "stage2_low_abnormality_bidir"]
    FIGDIR.mkdir(parents=True, exist_ok=True)
    # redirect the real plotter at the model's pooled record + a dedicated fig dir
    P._resolve_subject_dir = lambda dataset, subject: POOLED / subject
    P.PR3_FIG_DIR = FIGDIR
    P._ensure_pr3_readme = lambda: None      # don't write the real-data README into the model dir
    for tag in tags:
        if not (POOLED / tag).exists():
            print(f"  (skip {tag}: no pooled dir)"); continue
        P.plot_pr3_subject_figure(_make_record(tag))
    # MODEL caption / caveat
    (FIGDIR / "README.md").write_text(
        "# model_propagation — MODEL plotted with the real per-subject pipeline\n\n"
        "模型自发事件（cm-SNN 虚拟病灶、两端池化）走**和真实病人完全相同的** lagPat + KMeans + "
        "`plot_pr3_subject_figure`，产出 `model_<config>_propagation.png`，与真实 per-subject 图直接可比。\n\n"
        "**诚实口径（锁）**：这是**仪器对齐**、非机制重现——单一连接轴 → 模板空间近 1 维 → stable_k=2 半被迫；"
        "图标题标 `model:`、不混进病人队列。图上 cluster 面板显示的 `inter-corr` 是管线 masked-插补后的值"
        "（~+0.44、forward/reverse:none），与真实被试**同一指标可比**；但模型**通道级真正的反相**是共享触点 "
        "**−0.945** / rank-displacement **swap=strict**（见 `pooled_bidir/<config>/masked_pipeline_summary.json`）。\n\n"
        "**关注点**：左下聚类 heatmap 两簇（正/反向事件）+ 右下两簇 rank 分布的峰值位置相反；"
        "把它和真实病人的 `<dataset>_<subject>_propagation.png` 并排，就是模型↔数据同款图对比。\n")
    print(f"done -> {FIGDIR}")


if __name__ == "__main__":
    main()
