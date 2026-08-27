# R1.5 / H3-long 阶段报告：技术版

## 1. 冻结版本与分母

- R1.5 revision：`r1_5_long_support_explicit_extension_v1`；30 个 fits。
- H3 revision：`h3_long_exact_boxcar_affine_edge_v2`；130/130 个预定 cells。
- support revision：`r1_5_h3_long_exact_recorded_segment_support_v2`；34 人 corrected recorded-segment audit。
- synthetic：54 cells，all-pass=True。
- 所有 event time 已逐 subject 重算满足 TRAIN/validation < dev_end；formal/sealed=false。

## 2. R1.5 patient-first 结果

| 患者 | 身份 | 已更新 seed | persistent 有利 | correct-time 有利 | first subset 有利 | continuation 有利 | 联合稳定 |
|---|---|---:|---:|---:|---:|---:|---:|
| epilepsiae_1096 | independent_extension | 0/5 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 distinct |
| epilepsiae_384 | independent_extension | 5/5 | 2/5 | 5/5 | 0/5 | 2/5 | 2/2 distinct |
| yuquan_zhangkexuan | independent_extension | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 5/5 distinct |
| yuquan_chengshuai | previously_seen_long_record_calibration | 1/5 | 1/1 | 1/1 | 0/1 | 1/1 | 1/1 distinct |
| yuquan_chenziyang | previously_seen_long_record_calibration | 0/5 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 distinct |
| yuquan_zhangjiaqi | previously_seen_long_record_calibration | 0/5 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 distinct |

R1.5 的正式描述层要求每个 seed 同时满足 selected epoch>0、persistent−memoryless<0、correct−matched-wrong<0；epoch-0 不进入方向分母。患者层要求至少 3 个 stable seeds 且至少 3 个 distinct checkpoint hashes。

独立扩展层只有张克轩通过（1/3）：persistent、correct-time 与 first subset 均为 5/5，continuation 为 0/5。E384 的 correct-time 为 5/5，但 persistent 仅 2/5；E1096 为 5/5 epoch-0 no-update。校准层没有患者达到 3 个 stable checkpoints。

## 3. H3-long 设计

Exposure 是 TRAIN-only cross-fitted load 或 participation innovation 的 exact last-N boxcar sum，按 recorded coverage segment 重置。所有 trainable arms 有拟合 intercept；主对照为 state-matched non-overlap、current-event-only、chronological-trend、intercept-only，以及 full-control cell 中严格不重叠的 causal previous-N block。每个 cell 绑定 support、split、代码、R1.5 result/checkpoint 与身份 fingerprint。

130 个预定 seed-cells 的最终分类为：edge-estimable=29，ZERO_GRADIENT=60，ZERO_SELECTED=11，RANK_DEGENERATE=0，NONFINITE_GRADIENT=0，SUPPORT_NOT_ESTIMABLE=30。0/16 个 full-control 患者-source-尺度组合达到患者级阳性，0/10 个 boundary 组合达到支持标准。

| 患者 | source | N | 支持层 | 边可估 seed | 独立 validation 单元 | full 阳性 | boundary 支持 | H5 | H10 | real-state | real-time-trend |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| epilepsiae_1096 | load | 1000 | full_control | 0/5 | 10 | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| epilepsiae_1096 | participation | 1000 | full_control | 0/5 | 10 | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| epilepsiae_1096 | load | 3000 | boundary_incomplete_control | 0/5 | 7 | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| epilepsiae_1096 | participation | 3000 | boundary_incomplete_control | 0/5 | 7 | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| epilepsiae_384 | load | 1000 | full_control | 5/5 | 2 | 0/5 | 0/5 | 0/5 | 0/5 | +0.0453 | +0.1251 |
| epilepsiae_384 | participation | 1000 | full_control | 5/5 | 2 | 0/5 | 0/5 | 0/5 | 0/5 | +0.0527 | +0.1251 |
| yuquan_chengshuai | load | 1000 | full_control | 0/5 | 3 | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_chengshuai | participation | 1000 | full_control | 1/5 | 3 | 0/5 | 0/5 | 0/5 | 0/5 | -0.0022 | -0.0022 |
| yuquan_chengshuai | load | 3000 | full_control | 0/5 | 1 | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_chengshuai | participation | 3000 | full_control | 0/5 | 1 | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_chengshuai | load | 10000 | boundary_incomplete_control | 0/5 | NA | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_chengshuai | participation | 10000 | boundary_incomplete_control | 0/5 | NA | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_chenziyang | load | 1000 | full_control | 2/5 | 1 | 0/5 | 0/5 | 0/5 | 0/5 | +0.0005 | +0.0036 |
| yuquan_chenziyang | participation | 1000 | full_control | 5/5 | 1 | 0/5 | 0/5 | 0/5 | 0/5 | +0.0001 | +0.0006 |
| yuquan_chenziyang | load | 3000 | boundary_incomplete_control | 0/5 | NA | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_chenziyang | participation | 3000 | boundary_incomplete_control | 0/5 | NA | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_zhangjiaqi | load | 1000 | full_control | 1/5 | 5 | 0/5 | 0/5 | 0/5 | 0/5 | -0.0007 | -0.0007 |
| yuquan_zhangjiaqi | participation | 1000 | full_control | 0/5 | 5 | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_zhangjiaqi | load | 3000 | full_control | 0/5 | 2 | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_zhangjiaqi | participation | 3000 | full_control | 0/5 | 2 | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_zhangjiaqi | load | 10000 | boundary_incomplete_control | 1/5 | 1 | 0/5 | 0/5 | 0/5 | 0/5 | -0.0128 | +0.0028 |
| yuquan_zhangjiaqi | participation | 10000 | boundary_incomplete_control | 1/5 | 1 | 0/5 | 0/5 | 0/5 | 0/5 | +0.0031 | +0.0114 |
| yuquan_zhangkexuan | load | 1000 | full_control | 3/5 | 1 | 0/5 | 0/5 | 0/5 | 0/5 | -0.0006 | -0.0069 |
| yuquan_zhangkexuan | participation | 1000 | full_control | 5/5 | 1 | 0/5 | 0/5 | 0/5 | 0/5 | +0.0361 | +0.0243 |
| yuquan_zhangkexuan | load | 3000 | boundary_incomplete_control | 0/5 | NA | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |
| yuquan_zhangkexuan | participation | 3000 | boundary_incomplete_control | 0/5 | NA | 0/5 | 0/5 | 0/5 | 0/5 | NA | NA |

full-control 独立单元宽度为 2N，boundary 为 N；上表独立单元数在 state matching 后的最终共同支持上计算。`primary_full_control_increment` 还要求独立单元中位 contrast 同向，且至少 3 个 validation 单元。重复 seed payload 不重复计稳定性。

张克轩是唯一 stable-T1 患者，但 N=1,000 最终只有 1 个独立 validation 单元；load 的非零边 3/5，且 0/3 胜 causal previous block，participation 虽 5/5 可估但 0/5 胜 state-matched/current-event/causal block。N=3,000 在最终共同支持上不可估。故本轮 H3 无支持，但不能解释成充分检出力下的生物学阴性。

## 4. H5/H10 解释

H5/H10 仅在当前 seed 的 T1 合格时运行；真实累积必须同时胜过 state-matched、current-event、chronological-trend、intercept 和可用 causal block，并有非零传播位移。它关闭新的 raw correction 和后续 H3 jumps，但使用真实 future event history，因此是 teacher-forced one-shot persistence，不是 autonomous rollout。本轮 H5 和 H10 均为 0 个患者通过。

## 5. 验收边界

- ordinary negative 不触发停跑。
- `ZERO_SELECTED`、`RANK_DEGENERATE`、`ZERO_GRADIENT`、`NONFINITE_GRADIENT`、`SUPPORT_NOT_ESTIMABLE` 分开统计。
- 少于 3 个最终 validation 独立单元只作描述。
- 本轮最多支持 development predictive association；不能升级为 IED→state 因果机制。

机器审计（结果树）：`results/epi_prssm/continuous_marked_state/r1/r1_5_h3_long/final_reports/machine_audit.json`。

机器审计（随报告提交的快照）：`docs/archive/topic5/continuous_marked_state_r1_5_h3_long_machine_audit_2026-08-27.json`。
