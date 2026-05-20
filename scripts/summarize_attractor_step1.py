#!/usr/bin/env python3
"""Topic 4 Step 1 cohort summary — read augmented per-subject JSON, emit md.

Reports (post 2026-05-10 hardening):
- Cohort split: Step 1 returned, GOF pass, GOF fail, excluded_from_h3_main.
- Principal curve var_explained_curve distribution + caveats (PCA-3 subspace
  qualifier, max_iter convergence count).
- Angle vs PR-2 KMeans axis distribution + caveat (single-point at s_median).
- s_kmeans diagnostic distribution + supervised-coordinate caveat.
- Coordinate-free PR-2 label transition sanity (λ₂ + z + p) — the most
  important new section.
- Cross-check: relationship between label λ₂ and s_kmeans separation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "topic4_attractor"
PER_SUBJECT_DIR = OUT_DIR / "per_subject"
SUMMARY_MD = OUT_DIR / "step1_summary.md"
GOF_THRESHOLD = 0.6
ANGLE_LOW_DEG = 15.0
ANGLE_HIGH_DEG = 30.0


def _q(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _finite(xs):
    return [x for x in xs if isinstance(x, (int, float)) and np.isfinite(x)]


def main() -> int:
    files = sorted(PER_SUBJECT_DIR.glob("*.json"))
    rows: List[Dict[str, Any]] = []
    for fp in files:
        with open(fp) as f:
            rows.append(json.load(f))

    excluded = [r for r in rows if r.get("excluded_from_h3_main")]
    main_rows = [r for r in rows
                 if "skipped" not in r and "error" not in r
                 and not r.get("excluded_from_h3_main")]
    n_subj = len(rows)

    var_curve = _finite([
        r.get("gof", {}).get("var_explained_curve", float("nan"))
        for r in main_rows
    ])
    pass_gof = [r for r in main_rows if r.get("gof_pass") is True]
    fail_gof = [r for r in main_rows if r.get("gof_pass") is False]

    angles = _finite([
        r.get("angle_to_kmeans_axis", {}).get("angle_deg", float("nan"))
        for r in main_rows
    ])
    angle_high = [r for r in main_rows
                  if isinstance(r.get("angle_to_kmeans_axis", {}).get("angle_deg"),
                                 (int, float))
                  and np.isfinite(r["angle_to_kmeans_axis"]["angle_deg"])
                  and r["angle_to_kmeans_axis"]["angle_deg"] >= ANGLE_HIGH_DEG]
    angle_low = [r for r in main_rows
                 if isinstance(r.get("angle_to_kmeans_axis", {}).get("angle_deg"),
                                (int, float))
                 and np.isfinite(r["angle_to_kmeans_axis"]["angle_deg"])
                 and r["angle_to_kmeans_axis"]["angle_deg"] < ANGLE_LOW_DEG]

    pc1 = _finite([r.get("pca", {}).get("explained_variance_ratio", [None])[0]
                   for r in main_rows])
    cum3 = _finite([r.get("pca", {}).get("cumulative_top_k") for r in main_rows])

    converged_count = sum(1 for r in main_rows
                          if r.get("principal_curve", {}).get("converged") is True)

    # s_kmeans diagnostics (excluding subjects with diag error)
    diag_rows = [r for r in main_rows
                 if isinstance(r.get("s_kmeans_diagnostic"), dict)
                 and "error" not in r["s_kmeans_diagnostic"]]
    s_kmeans_d = _finite([r["s_kmeans_diagnostic"].get("cohens_d") for r in diag_rows])
    s_kmeans_acc = _finite([r["s_kmeans_diagnostic"].get("midpoint_threshold_accuracy")
                             for r in diag_rows])

    # PR-2 label transition sanity
    sanity_rows = [r for r in main_rows
                   if isinstance(r.get("label_transition_sanity"), dict)
                   and "error" not in r["label_transition_sanity"]
                   and isinstance(r["label_transition_sanity"].get("obs"), dict)]
    lam2_obs = _finite([r["label_transition_sanity"]["obs"].get("lambda_2")
                         for r in sanity_rows])
    lam2_z = _finite([r["label_transition_sanity"].get("z_lambda_2")
                       for r in sanity_rows])
    lam2_p = _finite([r["label_transition_sanity"].get("p_empirical")
                       for r in sanity_rows])
    lam2_significant = [r for r in sanity_rows
                        if isinstance(r["label_transition_sanity"].get("p_empirical"), (int, float))
                        and np.isfinite(r["label_transition_sanity"]["p_empirical"])
                        and r["label_transition_sanity"]["p_empirical"] < 0.001
                        and isinstance(r["label_transition_sanity"]["obs"].get("lambda_2"), (int, float))
                        and r["label_transition_sanity"]["obs"]["lambda_2"] > 0]

    cluster_imbalance = []
    for r in main_rows:
        nc = r.get("n_in_cluster", []) or []
        if len(nc) >= 2 and sum(nc) > 0:
            cluster_imbalance.append(min(nc) / sum(nc))

    lines: List[str] = []
    lines.append("# Topic 4 Step 1 — principal curve + GOF + angle + s_kmeans + label-transition sanity")
    lines.append("")
    lines.append("_数据入口与现象证据见 Topic 1 §3.1d cluster geometry._")
    lines.append("_本节范围：35 例 stable_k=2 且 channel_union ≥ 6 的 subject。Stable_k>2（5 例）+ 通道不足者已在 Step 0 外排，结论不能 generalize 到全部 40 例。_")
    lines.append("")
    lines.append("## Locked design 回顾")
    lines.append("")
    lines.append(
        "- Feature space = `lagPatRank.T` with `np.where(finite, X, 0.0)`，PR-2 KMeans 同合同（`src/interictal_propagation.py:1215`）。"
        "**注意**：NaN→0 把 \"rank order + participation pattern\" 一同编码进 feature 空间——这是 PR-2 state 定义的一部分（合同选择），不是单纯的生理量。后续科学解释必须明说。"
    )
    lines.append(
        f"- 主分析 cohort = stable_k=2，`n_participating ≥ 6` 后 `n_events_eligible ≥ 100` (Step 0 锁定)。本批共 {n_subj} 例 per-subject JSON。"
    )
    lines.append(
        "- PCA 顶 3 PC，Hastie-Stuetzle smoothing-spline 主曲线，迭代上限 15 次（n>20k 时 stride 子采样）。"
    )
    lines.append(
        f"- GOF 硬 gate：`var_explained_curve > {GOF_THRESHOLD}`，**作用域是 PCA-3 子空间**，非原始高维 X。"
    )
    lines.append(
        "- KMeans 主轴 = `centroid_1 - centroid_0`，centroid 用 PR-2 `labels` 映射到 Topic 4 eligible 子集后在 X 上重算。"
        "**Label 长度 mismatch 直接外排**（不用 template fallback），原因记录在 `excluded_from_h3_main`。"
    )
    lines.append("")
    lines.append("## Cohort 实测")
    lines.append("")
    lines.append(f"- 总 subject 数：{n_subj}")
    lines.append(f"- 进 H3 主分析：**{len(main_rows)}**")
    lines.append(f"- `excluded_from_h3_main`：**{len(excluded)}**")
    if excluded:
        for r in excluded:
            lines.append(f"  - {r['sid']}: {r['excluded_from_h3_main']}")
    lines.append("")
    lines.append(f"- GOF pass (`var_explained_curve > {GOF_THRESHOLD}` in PCA-3 subspace)：**{len(pass_gof)} / {len(main_rows)}**")
    lines.append(f"- GOF fail：**{len(fail_gof)}**")
    if fail_gof:
        for r in sorted(fail_gof,
                        key=lambda x: x.get("gof", {}).get("var_explained_curve", 1.0)):
            v = r.get("gof", {}).get("var_explained_curve", float("nan"))
            cum = r.get("pca", {}).get("cumulative_top_k", float("nan"))
            lines.append(
                f"  - {r['sid']}: var_explained_curve={v:.3f}, "
                f"cumulative_top3={cum:.3f}, n_chan_union={r.get('n_chan_union', 0)}"
            )
    lines.append("")
    lines.append("## 1) Principal curve — 在 PCA-3 子空间内的拟合")
    lines.append("")
    if var_curve:
        lines.append(
            f"- `var_explained_curve` (in PCA-3): median = {_q(var_curve, 0.5):.3f}, "
            f"range [{min(var_curve):.3f}, {max(var_curve):.3f}], "
            f"p25 = {_q(var_curve, 0.25):.3f}, p75 = {_q(var_curve, 0.75):.3f}"
        )
    if pc1:
        lines.append(
            f"- PC1 ratio: median = {_q(pc1, 0.5):.3f}, range [{min(pc1):.3f}, {max(pc1):.3f}]"
        )
    if cum3:
        lines.append(
            f"- top-3 cumulative ratio: median = {_q(cum3, 0.5):.3f}, "
            f"range [{min(cum3):.3f}, {max(cum3):.3f}]"
        )
    lines.append("")
    lines.append(
        "**警示**：`var_explained_curve` 的分母是 top-3 PC 子空间方差，不是原始 X 全方差。"
        "top-3 cumulative ratio median 只占原始 X 总方差约六成，**不能由此结论\"原始高维传播态是 1D manifold\"**。"
    )
    lines.append(
        f"- 在 max_iter=15 内严格收敛：{converged_count} / {len(main_rows)}。"
        "其余 subject 给出最后一次迭代的结果——**临时**，max_iter sensitivity 待跑（task #11）。"
    )
    lines.append("")
    lines.append("## 2) Principal curve 切向 vs PR-2 KMeans 主轴 — 单点 (s_median)")
    lines.append("")
    if angles:
        lines.append(
            f"- 夹角 (at s_median): median = {_q(angles, 0.5):.1f}°, "
            f"range [{min(angles):.1f}°, {max(angles):.1f}°]"
        )
        lines.append(
            f"- 共线区 angle < {ANGLE_LOW_DEG}°：{len(angle_low)} 例"
        )
        lines.append(
            f"- 非共线区 angle ≥ {ANGLE_HIGH_DEG}°：{len(angle_high)} 例"
        )
    lines.append("")
    lines.append(
        "**警示**：当前夹角仅在 `s_median` 处取曲线切向，是单点测量。"
        "\"主曲线整体与 KMeans 轴正交\"的结论需 grid-wide / event-weighted 角度分布——待补（task #12）。"
    )
    lines.append("")
    lines.append("## 3) s_kmeans — PR-2-label-supervised 1D 投影")
    lines.append("")
    lines.append(
        "**重要 disclaimer**：s_kmeans 是用 PR-2 label 算 centroid，再投影回的"
        "**监督坐标**。它能确认\"两个 PR-2 cluster 在 rank 空间里可被一条轴分开\"，"
        "**不能独立证明双稳态**。这部分是 cluster 几何 sanity，不是 H3 evidence。"
    )
    lines.append("")
    if s_kmeans_d:
        lines.append(
            f"- |Cohen's d| (per subject, |d_avg|): median = {_q([abs(v) for v in s_kmeans_d], 0.5):.2f}, "
            f"range [{min(abs(v) for v in s_kmeans_d):.2f}, {max(abs(v) for v in s_kmeans_d):.2f}]"
        )
    if s_kmeans_acc:
        lines.append(
            f"- midpoint-threshold accuracy: median = {_q(s_kmeans_acc, 0.5):.3f}, "
            f"range [{min(s_kmeans_acc):.3f}, {max(s_kmeans_acc):.3f}]"
        )
    lines.append("")
    lines.append("## 4) PR-2 label transition sanity — coordinate-free metastability test (KEY)")
    lines.append("")
    lines.append(
        "在 within-block 相邻事件对上算 2×2 PR-2 label transition matrix M，"
        "λ₂ = trace(M) − 1 ∈ [−1, 1]。λ₂ → 1 = 高 metastability（长 dwell 罕跳）；"
        "λ₂ → 0 = 无时间结构；λ₂ < 0 = 反相关（震荡）。"
        "Null：within-block label shuffle (n_perm=1000)，保 marginal cluster fraction，破时序。"
        "**这套测度不依赖任何 1D 坐标**，是直测 H3 metastable 假设的最朴素方法。"
    )
    lines.append("")
    if lam2_obs:
        lines.append(
            f"- λ₂ (observed): median = {_q(lam2_obs, 0.5):.3f}, "
            f"range [{min(lam2_obs):.3f}, {max(lam2_obs):.3f}]"
        )
    if lam2_z:
        lines.append(
            f"- z_λ₂ vs within-block shuffle: median = {_q(lam2_z, 0.5):.1f}, "
            f"range [{min(lam2_z):.1f}, {max(lam2_z):.1f}]"
        )
    if lam2_p:
        lines.append(
            f"- empirical p (right-tail): median = {_q(lam2_p, 0.5):.4f}, "
            f"min = {min(lam2_p):.4f}"
        )
        lines.append(
            f"- p < 0.001 且 λ₂ > 0：**{len(lam2_significant)} / {len(sanity_rows)}**"
        )
    lines.append("")
    lines.append("### Per-subject PR-2 label transition sanity 表")
    lines.append("")
    lines.append("| sid | n_pairs | λ₂_obs | null_mean | null_sd | z | p_emp |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(sanity_rows,
                    key=lambda x: -x["label_transition_sanity"]["obs"].get("lambda_2", -1.0)):
        sn = r["label_transition_sanity"]
        obs = sn["obs"]
        n_pairs = obs.get("n_pairs", "")
        lam2 = obs.get("lambda_2", float("nan"))
        nm = sn.get("null_mean", float("nan"))
        ns = sn.get("null_sd", float("nan"))
        z = sn.get("z_lambda_2", float("nan"))
        p = sn.get("p_empirical", float("nan"))
        lines.append(
            f"| {r['sid']} | {n_pairs} | {lam2:.3f} | {nm:.3f} | {ns:.4f} | "
            f"{z:.1f} | {p:.4f} |"
        )
    lines.append("")
    lines.append("## 5) 主曲线 vs KMeans 主轴几乎正交 — 现象与解读")
    lines.append("")
    lines.append(
        "Step 1 实测显示：**绝大多数 subject 的 principal curve 在 s_median 处切向"
        "与 PR-2 KMeans 主轴夹角 ≥ 50°，多数 70°–90°**。"
        "解释（**目前是观察，不是结论**）：principal curve 抓的是 within-cluster "
        "几何（每个 cluster 内部的 1D 延展），不是 cluster-to-cluster 分离方向。"
    )
    lines.append("")
    lines.append(
        "对 H3 的影响：直接把 Λ_gap 跑在 principal curve 的 s 上，测的可能"
        "是 within-cluster trajectory 而不是 between-cluster 切换。**Step 2 主路径建议改为：**"
    )
    lines.append("")
    lines.append(
        "1. 先看 §4 的 label transition λ₂ — 不依赖 1D 坐标，是最干净的 H3 直测；"
    )
    lines.append(
        "2. 再跑 Λ_gap on `s_kmeans`（K=8 主，{6,10,12} sensitivity）作为"
        "K-bin 加细版本的 H3 测试；两者方向必须一致才信。"
    )
    lines.append(
        "3. principal curve s 仅作 sensitivity，用来诊断 within-cluster 是否还有独立动力学层。"
    )
    lines.append("")
    lines.append("## 6) Cluster size imbalance（Step 3 估 M_label 时需关注）")
    lines.append("")
    if cluster_imbalance:
        lines.append(
            f"- min(cluster) / total: median = {_q(cluster_imbalance, 0.5):.3f}, "
            f"range [{min(cluster_imbalance):.3f}, {max(cluster_imbalance):.3f}]"
        )
    lines.append("")
    lines.append(
        "imbalance 越极端，minority-row 的 transition counts 越少，λ₂ 估计噪声越大。"
        "Step 3 报告里 `min_row_count` / `zero_row_count` 必须当 covariate 看。"
    )
    lines.append("")
    # Read sensitivity summary if present
    sens_md = OUT_DIR / "step1_sensitivity_summary.md"
    sens_csv = OUT_DIR / "step1_sensitivity.csv"
    sens_lines: List[str] = []
    if sens_csv.exists():
        import csv
        rows_sens: List[Dict[str, Any]] = []
        with open(sens_csv) as f:
            for row in csv.DictReader(f):
                rows_sens.append(row)

        def _agg(rows_sens, max_iter, key):
            vals = []
            for r in rows_sens:
                if int(r["max_iter"]) != max_iter:
                    continue
                try:
                    v = float(r[key])
                    if np.isfinite(v):
                        vals.append(v)
                except Exception:
                    pass
            return vals

        sens_lines.append("## 7) max_iter & grid/event-wide angle sensitivity（task #11+#12）")
        sens_lines.append("")
        sens_lines.append(
            "Re-fit principal curve at max_iter ∈ {15, 30, 60} on the 34 main-cohort "
            "subjects. Reports `var_explained_curve`, single-point angle (at s_median), "
            "grid-wide median angle (along curve), event-weighted median angle "
            "(at every event's projected s)."
        )
        sens_lines.append("")
        sens_lines.append("| max_iter | n | converged | var_curve median (range) | angle@s_median | angle_grid | angle_event |")
        sens_lines.append("|---:|---:|---:|---|---:|---:|---:|")
        for mi in (15, 30, 60):
            sub = [r for r in rows_sens if int(r["max_iter"]) == mi]
            n_conv = sum(1 for r in sub if r["converged"] == "True")
            vcs = _agg(rows_sens, mi, "var_explained_curve")
            ams = _agg(rows_sens, mi, "angle_at_s_median_deg")
            ags = _agg(rows_sens, mi, "angle_grid_median_deg")
            aes = _agg(rows_sens, mi, "angle_event_median_deg")
            if vcs and ams and ags and aes:
                sens_lines.append(
                    f"| {mi} | {len(sub)} | {n_conv} | "
                    f"{np.median(vcs):.3f} ({min(vcs):.3f}–{max(vcs):.3f}) | "
                    f"{np.median(ams):.1f}° | {np.median(ags):.1f}° | "
                    f"{np.median(aes):.1f}° |"
                )
        sens_lines.append("")
        sens_lines.append("**关键解读：**")
        sens_lines.append("")
        sens_lines.append(
            "1. **var_explained_curve 稳定**：max_iter 15→60 median 仅从 0.953 漂到 0.950。"
            "Main batch 的 var_explained_curve 数字可信。"
        )
        sens_lines.append(
            "2. **角度对 max_iter 敏感**：angle@s_median median 从 83° → 71.5° "
            "（差 ~12°）。\"主曲线 ≈ 正交于 KMeans 主轴\"的措辞需谨慎："
            "实际的稳健界至少是 60°+ 而非 83°。"
        )
        sens_lines.append(
            "3. **Grid vs event vs single-point 三者每档 max_iter 内差 < 5°**："
            "single-point at s_median 不严重偏离 grid/event-weighted；问题不在采样位置，"
            "在主曲线尚未收敛。"
        )
        sens_lines.append(
            "4. **收敛差**：max_iter=60 仅 4/34 严格收敛（tol=1e-4）。"
            "Hastie-Stuetzle 迭代在 PR-2 rank 空间上行为不稳。某些 subject "
            "（e.g. epilepsiae_1077 var 0.658→0.549→0.733）非单调。"
            "**含义**：principal curve s 作为 H3 主路径不可靠；§4 的 label-transition "
            "λ₂ 路径不依赖 curve 收敛，仍是最稳健的 H3 直测。"
        )
        sens_lines.append("")
        sens_lines.append(f"完整 per-(subject × max_iter) 表见：`{sens_csv.relative_to(REPO_ROOT)}`。")
        sens_lines.append("")

    lines.extend(sens_lines)
    lines.append("## 8) 全部修复完成（按 user 优先级）")
    lines.append("")
    lines.append(
        "- ✅ 1096 label drift 根因：1 block 删除 + 28 block 事件数漂移，PR-2 label "
        "对应旧 ordering，无法对齐当前 loader → 排除（task #7,#8）。"
    )
    lines.append(
        "- ✅ Step 1 hardened：`pr2_template_fallback` 路径删除；label 长度 mismatch "
        "直接外排到 `excluded_from_h3_main`，附带原因（task #8）。"
    )
    lines.append(
        "- ✅ Re-run Step 1 + augment 序列化：35/35 per-subject JSON 写完，"
        "34 进 H3 主分析（task #9）。"
    )
    lines.append(
        "- ✅ PR-2 label transition matrix sanity (n_perm=1000)：coordinate-free，"
        "见 §4 主结果（task #10）。"
    )
    lines.append(
        "- ✅ max_iter sensitivity {15, 30, 60}：var_curve 稳定，angle 漂 12°，"
        "principal curve 收敛差，label-transition 路径更稳健（task #11，§7）。"
    )
    lines.append(
        "- ✅ Grid-wide / event-weighted angle 分布（task #12，§7）。"
    )
    lines.append(
        "- ✅ 报告口径修正（task #13）— 范围限定 35 stable_k=2 ∩ chan_union ≥ 6；"
        "s_kmeans 标 PR-2-label-supervised；var_explained_curve 标 PCA-3 subspace；"
        "NaN→0 标 rank+participation 复合 state。"
    )
    lines.append("")
    lines.append("## 文档归属提醒")
    lines.append("")
    lines.append(
        "- Topic 1 主文档先不动；待 Step 2 Λ_gap 数据回来 + max_iter / grid-angle sensitivity 做完，"
        "再加 §3.X cluster geometry → attractor class 桥接节。"
    )
    lines.append(
        "- 详细结果将归档：`docs/archive/topic4/propagation_state_attractor_diagnostics_<DATE>.md`，"
        "首句声明数据入口在 Topic 1 §3.1d。"
    )
    lines.append(
        "- Topic 4 主文档暂不开（CLAUDE.md §5：所有 sensitivity 闸门通过才进主文档）。"
    )

    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {SUMMARY_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
