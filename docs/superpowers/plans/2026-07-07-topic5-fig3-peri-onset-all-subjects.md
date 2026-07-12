# Topic 5 Fig3-B peri-onset field similarity — all-subject run handoff

## 目标

把 E1146 已验收的 Fig3-B peri-onset field similarity 图扩展到所有可评估 subject。每个 subject 输出同一张 paper-ready 双面板：

1. `max(|r_A|, |r_B|)` sign-free scaffold similarity；
2. signed template A/B similarity polarity sidecar。

这条线是 Fig3 的 single-subject dynamic material / per-subject supplement pool，不是新的 cohort formal gate。不要把它写成 timing-order replay、direction replay 或机制证明。

## 当前已锁定合同

- 频段：`1-150 Hz` summed spectrogram log power；notch 滤波输入（50/100/150/200 Hz），**无额外 FFT-bin line mask**（与 Fig3-A / v2 的 bin-mask 版本谐波处理不同）。
- 归一化：per-channel baseline robust-z。
- 时间范围：`[-120,+20]s`，onset-aligned。
- 滑窗：10 s window，2 s step。
- x 轴：window center；`xlim` 贴第一个/最后一个 window center。
- 图形：单行双面板；浅线=单次 seizure，粗线=跨 seizure median，阴影=IQR；0 s 灰色虚线。
- 不画方差/n 诊断下排；variance、n、drops 写 JSON/CSV。
- Panel A 的 maxAB 是 sign-free scaffold readout；Panel B 的 signed A/B 只解释 polarity 稳定性。

## 已有 E1146 参考输出

- Figure: `results/paper-ready-figure/fig3_peri_onset_field_similarity/figures/epilepsiae_1146_peri_onset_field_similarity_paper_ready.png`
- PDF: `results/paper-ready-figure/fig3_peri_onset_field_similarity/figures/epilepsiae_1146_peri_onset_field_similarity_paper_ready.pdf`
- Summary: `results/paper-ready-figure/fig3_peri_onset_field_similarity/figures/epilepsiae_1146_peri_onset_field_similarity_paper_ready_summary.json`
- Source CSV: `results/topic5_ictal_recruitment/field_dynamics_signed/epilepsiae_1146_signed_broadband_1_150Hz_similarity_timecourse_m120_p20_10s_step2s_per_seizure.csv`

## 相关脚本

- 上游计算：`scripts/plot_topic5_signed_broadband_similarity_timecourse.py`
- Paper-ready 绘图：`scripts/paper_figures/plot_fig3_peri_onset_field_similarity.py`
- 单 seizure 诊断相似性：`scripts/compute_topic5_signed_broadband_similarity.py`

当前 `src.preprocessing.load_epilepsiae_block` 和 `src.ictal_onset_extraction.extract_seizure_window` 已支持 Epilepsiae raw block 局部 crop；不要回退到整段 block 读取。

## 推荐执行流程

先生成 subject 列表：

```bash
python - <<'PY'
from pathlib import Path
root = Path("results/spatial_modulation/propagation_geometry/observation_readout/real_subjects")
subs = sorted({p.name[:-len("_t_a.json")] for p in root.glob("*_t_a.json")})
for s in subs:
    print(s)
PY
```

逐 subject 跑上游 1-150 Hz timecourse：

```bash
python scripts/plot_topic5_signed_broadband_similarity_timecourse.py \
  --subject epilepsiae_1146 \
  --start-sec -120 \
  --stop-sec 20 \
  --band-lo 1 \
  --band-hi 150 \
  --window-sec 10 \
  --step-sec 2
```

再生成 paper-ready 图：

```bash
python scripts/paper_figures/plot_fig3_peri_onset_field_similarity.py \
  --subject epilepsiae_1146
```

全 subject 批处理时用 fail-closed wrapper：每个 subject 独立 try/except，记录 `ok / drop_reason`，不要因为一个 subject 失败中断整批。

## 需要新增的全队列索引

请补一个主索引 CSV / JSON，建议放：

- `results/paper-ready-figure/fig3_peri_onset_field_similarity/fig3_peri_onset_subject_index.csv`
- `results/paper-ready-figure/fig3_peri_onset_field_similarity/fig3_peri_onset_subject_index.json`

每行至少包含：

- `subject`
- `status`
- `drop_reason`
- `n_seizures`
- `n_windows`
- `maxAB_median_of_window_medians`
- `maxAB_median_of_window_variances`
- `signed_A_median_of_window_medians`
- `signed_B_median_of_window_medians`
- `source_csv`
- `figure_png`
- `figure_pdf`

## 验收标准

1. E1146 复现值应与当前 summary 一致：
   - `n_seizures=25`
   - `n_windows=66`
   - `maxAB_abs.median_of_window_medians≈0.6214`
   - `signed_A.median_of_window_medians≈-0.2212`
   - `signed_B.median_of_window_medians≈0.2100`
2. 每个成功 subject 都有 PNG、PDF、summary JSON、source per-seizure CSV、aggregate CSV。
3. 每个失败 subject 都有明确 `drop_reason`，例如 raw window missing、Nyquist too low、insufficient matched contacts、no eligible seizures、missing template A/B。
4. `figures/README.md` 不要被每个 subject 改成只描述最后一个 subject；保持泛化说明。
5. 不要把全 subject 输出写成 formal cohort statistic。它是 Fig3-B 的 per-subject material pool；正式 cohort shift 仍是 Fig3-A Data-vs-Null panel。

## 可复制给 agent 的任务提示

请在 `/home/honglab/leijiaxin/HFOsp` 继续 Topic 5 Fig3-B peri-onset field similarity 的全 subject 批处理。先阅读：

1. `docs/main_figure_plan.md` 的 Fig3-A / Fig3-B 段；
2. `docs/figure_style_guide.md` 的 Topic 5 · Fig3-B 图型规范；
3. 本文件 `docs/superpowers/plans/2026-07-07-topic5-fig3-peri-onset-all-subjects.md`。

目标：对所有有 `results/spatial_modulation/propagation_geometry/observation_readout/real_subjects/*_t_a.json` 的 subject，运行 `scripts/plot_topic5_signed_broadband_similarity_timecourse.py --start-sec -120 --stop-sec 20 --band-lo 1 --band-hi 150 --window-sec 10 --step-sec 2`，再运行 `scripts/paper_figures/plot_fig3_peri_onset_field_similarity.py --subject <subject>`，生成 per-subject PNG/PDF/summary，并写一个 subject index CSV/JSON 汇总成功和 drop。必须 fail-closed 记录每个失败 subject 的原因；不要用 1-45 Hz cache 顶替 1-150 Hz；不要改变 Fig3-B 画图风格；不要把结果写成 formal cohort gate。完成后给出：成功 subject 数、drop 表、E1146 sanity 数值、输出路径、是否需要进一步人工目视筛图。
