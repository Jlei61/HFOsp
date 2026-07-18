# Fig3-B：间期时序场与发作早期能量场规范

> 状态：paper-ready candidate，2026-07-18
> 当前实例：E1146，seizure 15（`seizure_id=114601500102`）
> 唯一入口：`scripts/paper_figures/plot_fig3b_interictal_ictal_shared_field.py`

## 1. 这张图回答什么

在同一个患者特异、已冻结的 shared plane 上，把间期 TA timing field 与一例和 TA 最一致的真实发作早期 broadband power 并排展示。它是 representative-subject 的 signal-to-field bridge，不是 template-free 重建、replay 证明、cohort 统计或机制检验。

## 2. 冻结输入

- 轴、平面、触点顺序、support、方向和 fingerprint 只读 `results/interictal_propagation_masked/template_gradient_fields/per_subject/epilepsiae_1146.json`。
- consumer 必须通过 `scorers_from_interictal_record()` 的 fingerprint gate，并按 `interictal_field.contact_order` 做 exact channel-name join。
- 不得根据 seizure 值重拟合 axis、plane、transverse sign、support 或 kernel；缺输入时 fail closed。
- 左图固定展示 TA。右图按预先声明的可复现规则，从 E1146 当前 25 次 complete / exact `1–150 Hz` 发作中选择 `shared_a_signed` 最大者：seizure 15，`shared_a_signed=0.869905`、`shared_b_signed=-0.645552`。完整筛选表必须写入 metadata；该 best-case representative 不能称独立验证。

## 3. 发作早期能量合同

- 数据：E1146 seizure 15，CAR。Fig3-A 仍使用 seizure 7；两个 panel 不再声称来自同一次发作。
- 频带：精确 `1–150 Hz`；1 s PSD window、0.5 s hop，频带内 PSD 求和后取 log；与正式 concordance producer 常量一致。
- 时间：clinical onset 为 0，右图只平均完整的 `[0,10] s` spectral cells。
- baseline：真实 EEG onset 的 `[-120,-90] s`，逐触点 median/MAD robust-z；再减去同一 baseline 内 robust-z 中位数。
- 显示：对 15 个冻结触点的 `[0,10] s` 平均 robust-z 做连续 min–max 映射。**不得 rank、不得 sign flip**；原始 robust-z 值必须写入 metadata。

## 4. 投影与画布

- 左右严格使用相同的 shared TA axis、transverse sign、坐标范围、15 个触点和 TA support。
- 左图：冻结 TA rank min–max 到 `0=early, 1=late`，`viridis`；标题只写红色语义色 `TA`（`#B2182B`）。
- 右图：发作早期 broadband power 只为插值做连续 min–max，使用 `magma_r`；高 power 为深色、低 power 为浅色，不用旧 `Reds` painter。
- 两图只共用**显示** kernel `6 mm`；不得把 6 mm 写成冻结评分 kernel。
- 单行两列、等大方形 field；y 轴只在左图出现；两个 panel 各自写 `shared TA axis (mm)` xlabel；每图一条等高 colorbar。
- 左色条直接显示真实 propagation rank（当前 `0–14`，端点附 `early/late`）；右色条直接显示 `[0,10] s` baseline-normalized log-band-power robust-z。不得只写无量纲的 `0/1` 或只有 `low/high`。
- 深浅语义必须一致：左图最早传播为深色，右图最高 broadband power 为深色。
- 标题只写 `E1146`、红色 `TA fields` 与 `Broadband power`；不写内部 a/b 编号或长统计说明。

## 5. 产物与验收

- canonical candidate 目录：`results/paper-ready-figure/fig3b_interictal_ictal_shared_field/figures/`。
- 必须同时生成 PNG、vector PDF、metadata JSON 和中文 README。
- metadata 必须保存：raw seizure id、reference、band、clinical/baseline windows、frequency parameters、contact order、raw/display values、frozen fingerprint、完整 25-seizure TA 筛选表、A/B scores、checkpoint parity 和 claim boundary。
- 每次重画至少检查：自动选择 seizure 15、15/15 exact-name match、精确 1–150 Hz、完整 `[0,10] s`、`shared_a_signed=0.869905`、shared winner=TA、checkpoint score 最大误差 `<=1e-12`、左右 display sigma 都为 6 mm、PNG/PDF 视觉一致、Python compile 和 `git diff --check`。
