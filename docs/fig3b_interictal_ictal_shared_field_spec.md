# Fig3-B：间期时序场与发作早期能量场规范

> 状态：paper-ready locked，视觉合同更新 2026-09-03
> 当前实例：E1146 seizure 2（`seizure_id=114600200102`）
> 唯一入口：`scripts/paper_figures/plot_fig3b_interictal_ictal_shared_field.py`

## 1. 这张图回答什么

在同一个患者特异、已冻结的 shared plane 上，把间期 TA timing field 与一例和 TA 最一致的真实发作早期 broadband power 并排展示。它是 representative-subject 的 signal-to-field bridge，不是 template-free 重建、replay 证明、cohort 统计或机制检验。

## 2. 冻结输入

- 轴、平面、触点顺序、support、方向和 fingerprint 只读 `results/interictal_propagation_masked/template_gradient_fields/per_subject/epilepsiae_1146.json`。
- consumer 必须通过 `scorers_from_interictal_record()` 的 fingerprint gate，并按 `interictal_field.contact_order` 做 exact channel-name join。
- 不得根据 seizure 值重拟合 axis、plane、transverse sign、support 或 kernel；缺输入时 fail closed。
- 左图固定展示 TA。右图先从 E1146 当前 25 次 complete / exact `1–150 Hz` 发作中要求 15/15 clinical `[0,10] s` robust-z power `>0`、`shared_a_signed>0` 且 TA 为 maxAB winner；再要求 power 与 `−rank_A` 的直接相关 `>=0.35`、rank 0–3 源区高于 rank 11–14 晚期区，且 SCL9/ICL11 两个最早端点的 normalized power 均 `>=0.30`。该审计得到 seizure 2 / 10 / 23 / 1 四个候选，最终经目视锁定 seizure 2。完整筛选表保存在 candidate summary；该 intentionally selected representative 不能称独立验证。

## 3. 发作早期能量合同

- 数据：E1146 seizure 2，CAR。Fig3-A 仍使用 seizure 7；两个 panel 不再声称来自同一次发作。
- 频带：精确 `1–150 Hz`；1 s PSD window、0.5 s hop，频带内 PSD 求和后取 log；与正式 concordance producer 常量一致。
- 时间：clinical onset 为 0，右图只平均完整的 `[0,10] s` spectral cells。
- baseline：真实 EEG onset 的 `[-120,-90] s`，逐触点 median/MAD robust-z；再减去同一 baseline 内 robust-z 中位数。
- 显示：对 15 个冻结触点的 `[0,10] s` 平均 robust-z 做连续 min–max 映射。**不得 rank、不得 sign flip**；原始 robust-z 值必须写入 metadata。

## 4. 投影与画布

- 左右严格使用相同的 shared TA axis、transverse sign、坐标范围、15 个触点和 TA support。
- 左图：冻结 TA rank min–max 到 `0=early, 1=late`，`viridis`；标题只写红色语义色 `TA fields`（`#B2182B`）。
- 右图：发作早期 broadband power 只为插值做连续 min–max，使用 `Blues`；高 power 为深色、低 power 为浅色，不用旧 `Reds` / `magma` painter。
- 两图只共用**显示** kernel `6 mm`；不得把 6 mm 写成冻结评分 kernel。
- 单行两列、等大方形 field；空间 y label 统一写 `Y (mm)` 且只在左图出现；两个 panel 各自写 `shared TA axis (mm)` xlabel；每图一条等高 colorbar。
- 左色条与 Fig2 统一为 normalized-rank `viridis` 显示语法：标题 `ranks`，ticks 为 `0 early / 0.5 / 1 late`；原始 propagation rank（当前 `0–14`）只保留在 metadata。右色条直接显示 `[0,10] s` baseline-normalized log-band-power robust-z。
- 深浅语义必须一致：左图最早传播为深色，右图最高 broadband power 为深色。
- 不写整体标题。左图只写红色 `TA fields`；右图按 `E10 | SZ3` / `Early ictal field` 两行写子图标题；不写内部 a/b 编号或长统计说明。右 colorbar 可见标题为 `power` / `z`。
- 锁定的 seizure 2 在 `[0,10] s` 的 15/15 触点 robust-z 均为正，范围 `+1.03–+3.68`；`shared_a_signed=0.719127`，direct early-rank correlation=`0.570884`。`shared_a_signed` 仍只量化触点间空间模式一致性，不能单独当作全局能量检验。

## 5. 产物与验收

- canonical candidate 目录：`results/paper-ready-figure/fig3b_interictal_ictal_shared_field/figures/`。
- 必须同时生成 PNG、vector PDF、metadata JSON 和中文 README。
- metadata 必须保存：raw seizure id、reference、band、clinical/baseline windows、frequency parameters、contact order、raw/display values、frozen fingerprint、morphology gate、候选审计路径、A/B scores、checkpoint parity 和 claim boundary；完整 25-seizure 表保存在 candidate summary。
- 正式重画至少检查：默认输出 seizure 2、15/15 exact-name match、15/15 early-power robust-z 为正、精确 1–150 Hz、完整 `[0,10] s`、`shared_a_signed=0.719127`、shared winner=TA、direct early-rank correlation=`0.570884`、checkpoint score 最大误差 `<=1e-12`、左右 display sigma 都为 6 mm、PNG/PDF 视觉一致、Python compile 和 `git diff --check`。
