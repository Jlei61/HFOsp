# Fig3-A raw spectral context 正式验收（2026-07-18）

## 1. 验收结论

**通过，锁定为 paper-ready Fig3-A 正式版本。** Canonical producer 为
`scripts/paper_figures/plot_fig3_raw_spectral_context.py`，canonical 输出为
`results/paper-ready-figure/fig3a_raw_spectral_context/figures/`。原
`results/paper-ready-figure/archive/2026-08-09_stale_aliases/fig3_sup2_raw_spectral_context/`
仅保留历史溯源，不再作为正式 panel 引用。

这张图承担 reader-facing signal context：把同一 seizure 的 raw SEEG、单通道 TFR 和四档
band-power trajectory 放在一个连续 clinical-onset 时间轴上。它不是 cohort statistic，也不支持
timing-order replay、direction replay、onset-emergent alignment 或机制证明。

## 2. 冻结数据合同

- subject / seizure：`epilepsiae_1146`，seizure index `7`（公开标签 `E1146`）。
- reference：CAR；采样率与 seizure id 写入 summary JSON。
- raw channel pool：masked rank-displacement artifact 的 lagPat joint-valid contacts，共 15 条：
  `SCL6–SCL9`、`ICL1–ICL11`。
- spectral channel：`SCL9`；自动选择审计仍保留 alpha / beta 等计算，但正式画布不展示这两档。
- 连续显示窗：`[-120,+20] s`；baseline：`[-120,-90) s`；clinical-onset early-field shading：
  `[0,10) s`。
- TFR / band power：同一 SCL9 spectrogram，均为相对 baseline 的 dB 量。

## 3. 锁定画图合同

### 左侧 raw SEEG / TFR

- 左上 15 条 stacked raw SEEG；左下 SCL9 TFR。两图必须共享完全相同的 x limits 和 data-axis
  左右边界。
- TFR colorbar 使用独立窄列，顶部短标题 `TFR (dB)`；不得让 colorbar 自动缩窄 TFR，也不得把
  竖排 colorbar label 放在左右图之间造成 ylabel 串义。
- raw SEEG bottom/top padding 固定为 data span 的 `1% / 6%`；最低通道靠近 x-axis，但负向波形
  不能被裁切。
- 无内部 a/b 编号。标题只保留左对齐粗体 `E1146` 和 `TFR on SCL9`。

### 右侧 2×2 band-power panels

- 上排：`low bands (1–30 Hz)`、`gamma (30–80 Hz)`。
- 下排：`high-gamma (80–150 Hz)`、`broadband (1–150 Hz)`。
- 每行共享 y limits；数值 y ticks 与 `dB vs baseline` 只画在左图，右列不画独立 y ticks。
- x label 固定 `Time (s)`；标题只写 band name 与括号内频率范围。

### 时间标注

- baseline 用浅蓝窗，clinical-onset `[0,10) s` 用红色 alpha=`0.15` 阴影；标签只写
  `BASELINE` / `CLINICAL ONSET`。
- 不标 EEG onset；不画 EEG/clinical onset 竖线；不写 `CLINICAL 0–10 s`。

## 4. 验收证据

- producer 从真实 Epilepsiae seizure window 和 accepted rank-displacement channel source 重生成，
  没有重画静态替代图。
- PNG 与 PDF 由同一 producer、同一参数和同一数据状态连续生成；summary JSON 同步记录 subject、
  seizure、channel source、spectral channel、displayed bands、sidecar-only bands、窗口和输出路径。
- 已目检：raw/TFR 时间轴对齐；TFR colorbar 不挤压数据轴；最低 raw trace 未裁切；四个 band panel
  无遮挡；同行 y scale 一致；右列无重复 y ticks；clinical shading 清楚但不改变 `[0,10) s` 合同。
- 已通过 Python 编译、`git diff --check`、PNG/PDF 可读性和 metadata 字段检查。

## 5. 可报告与不可报告

**可以报告**：E1146 的代表性 seizure 在 clinical onset 附近呈现可由 raw SEEG、SCL9 TFR 和
low-to-high frequency band-power trajectories 共同核对的宽频增强。

**不可报告**：不能从这一单 seizure panel 推出 cohort prevalence、特定频带机制、HFO-specific
recruitment、传播方向重放、发作机制或临床因果关系。后续 field-concordance / trajectory / null
分析仍各自承担独立统计问题。
