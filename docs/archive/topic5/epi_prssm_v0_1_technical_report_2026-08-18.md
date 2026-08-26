# Epi-PRSSM v0.1 技术报告

**日期：** 2026-08-18 · **合同：** `topic5_epi_prssm_v0_1` · **状态：** `EXPLORATORY_DEVELOPMENT`（正式未触碰检验分区未释放）

- code revision: `7393745c6777adaf88fbf0c5bc087e4c2f1c0a9e`
- package hash (`src/topic5_epi_prssm/*.py`): `8fd11957dceec1c2a81b4b87ca9687fa5d8ab93557f5bc20715e4b4f38048087`
- scripts hash: `fb5f08c77bd2bd6b25101021114763715320fedd324f28d9ee8fc8719e4d40d8`
- cohort: `all34`

本报告的每一个数字都由 `scripts/topic5_epi_prssm/write_reports.py` 从`results/epi_prssm/v0_1/` 下的 per-job artefact 重新计算，未从日志抄写。阴性结果、失败运行与资源问题与阳性项同等可见。

---

## 1. 分母流

- 患者 34（Epilepsiae 18 / Yuquan 16）
- 事件 864,163；train 518,483、validation 172,831、test（封存）172,849
- 记录块 2097；session（300 s join）481
- 触点合计 492；无任何几何映射的患者 5 位（这些患者的图只含数据推出的有向传播支持）
- 记录时长合计 2203 小时；跨度中位数 3.55 天

**每位患者：**

| 患者 | 数据集 | 事件 | 触点 | train | validation | session | 记录小时 | IEI 中位数(s) | 几何映射 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epilepsiae_1073 | epilepsiae | 138275 | 6 | 82965 | 27655 | 38 | 156 | 1.44 | 3/6 |
| epilepsiae_1077 | epilepsiae | 32376 | 6 | 19425 | 6475 | 31 | 133 | 3.29 | 6/6 |
| epilepsiae_1084 | epilepsiae | 7318 | 11 | 4390 | 1464 | 29 | 116 | 3.45 | 11/11 |
| epilepsiae_1096 | epilepsiae | 140337 | 7 | 84202 | 28067 | 24 | 111 | 0.80 | 7/7 |
| epilepsiae_1125 | epilepsiae | 35971 | 8 | 21582 | 7194 | 28 | 98 | 2.15 | 7/8 |
| epilepsiae_1146 | epilepsiae | 12430 | 15 | 7458 | 2486 | 13 | 23 | 1.60 | 15/15 |
| epilepsiae_1150 | epilepsiae | 8561 | 9 | 5136 | 1712 | 22 | 113 | 10.04 | 9/9 |
| epilepsiae_139 | epilepsiae | 9184 | 7 | 5510 | 1837 | 19 | 96 | 6.41 | 7/7 |
| epilepsiae_253 | epilepsiae | 50747 | 8 | 30448 | 10149 | 38 | 207 | 3.23 | 8/8 |
| epilepsiae_384 | epilepsiae | 21495 | 9 | 12897 | 4299 | 9 | 35 | 2.43 | 9/9 |
| epilepsiae_442 | epilepsiae | 2667 | 15 | 1600 | 533 | 27 | 71 | 7.01 | 15/15 |
| epilepsiae_548 | epilepsiae | 6982 | 12 | 4189 | 1396 | 22 | 53 | 2.34 | 11/12 |
| epilepsiae_583 | epilepsiae | 7828 | 7 | 4696 | 1566 | 12 | 47 | 7.48 | 7/7 |
| epilepsiae_590 | epilepsiae | 4982 | 16 | 2989 | 996 | 42 | 166 | 34.43 | 16/16 |
| epilepsiae_620 | epilepsiae | 22408 | 9 | 13444 | 4482 | 38 | 201 | 6.66 | 9/9 |
| epilepsiae_635 | epilepsiae | 5111 | 10 | 3066 | 1022 | 11 | 78 | 6.45 | 7/10 |
| epilepsiae_922 | epilepsiae | 83638 | 8 | 50182 | 16728 | 12 | 40 | 0.91 | 8/8 |
| epilepsiae_958 | epilepsiae | 123419 | 16 | 74051 | 24684 | 37 | 160 | 1.56 | 16/16 |
| yuquan_chengshuai | yuquan | 27577 | 8 | 16546 | 5515 | 1 | 24 | 1.67 | 8/8 |
| yuquan_chenziyang | yuquan | 9609 | 10 | 5765 | 1922 | 1 | 24 | 3.02 | 0/10 |
| yuquan_gaolan | yuquan | 2993 | 12 | 1795 | 599 | 2 | 8 | 1.78 | 0/12 |
| yuquan_hanyuxuan | yuquan | 5468 | 22 | 3280 | 1094 | 1 | 26 | 1.98 | 0/22 |
| yuquan_huanghanwen | yuquan | 456 | 10 | 273 | 91 | 2 | 20 | 62.46 | 10/10 |
| yuquan_litengsheng | yuquan | 642 | 24 | 385 | 128 | 2 | 14 | 19.13 | 24/24 |
| yuquan_liyouran | yuquan | 2346 | 17 | 1407 | 469 | 2 | 22 | 4.86 | 17/17 |
| yuquan_pengzihang | yuquan | 22622 | 12 | 13573 | 4524 | 2 | 12 | 1.15 | 12/12 |
| yuquan_songzishuo | yuquan | 447 | 38 | 268 | 89 | 1 | 24 | 30.63 | 38/38 |
| yuquan_sunyuanxin | yuquan | 1282 | 12 | 769 | 256 | 3 | 8 | 11.96 | 0/12 |
| yuquan_wangyiyang | yuquan | 1919 | 22 | 1151 | 384 | 3 | 18 | 1.70 | 0/22 |
| yuquan_xuxinyi | yuquan | 6663 | 15 | 3997 | 1333 | 2 | 16 | 3.76 | 15/15 |
| yuquan_zhangbichen | yuquan | 8371 | 52 | 5022 | 1674 | 1 | 22 | 4.31 | 52/52 |
| yuquan_zhangjiaqi | yuquan | 48277 | 7 | 28966 | 9655 | 2 | 25 | 1.01 | 7/7 |
| yuquan_zhangkexuan | yuquan | 7948 | 26 | 4768 | 1590 | 3 | 12 | 2.65 | 26/26 |
| yuquan_zhaochenxi | yuquan | 3814 | 26 | 2288 | 763 | 1 | 24 | 7.05 | 26/26 |


## 2. 数据 / 划分 / 禁止输入审计（Hard Gate A）

**Hard Gate A 判定：`PASS`**，检查了 34 位患者，失败 0 项。

检查项：事件时间顺序、非参与触点不得携带 rank 或组标识（phantom rank）、并列关系来自显式组标识、划分严格按时间顺序。

**划分：**dataset_v0_4's own last-20% partition is the untouched test; its first-80% calibration partition is cut 75/25 in chronological order, realising the frozen 0.60/0.20/0.20 fractions without moving the sealed boundary；全部患者时间顺序正确 = `True`；test 状态 = `SEALED_UNTIL_FORMAL_TEST_RELEASE`。

**禁止输入：**

- 拒绝的触点特征：`['prefix_participation_support', 'prefix_participation_support_centered']`——their estimation partition is not recoverable from the artefact, so re-using them would put an unverifiable whole-record repertoire estimate inside a train-only baseline
- 几何：spec section 4 authorises the symmetric contact-geometry Laplacian as graph support; this is a documented divergence from the v4.0 contract, which forbade geometry；SOZ 仍禁止 = `True`

**与上一代管线的映射一致性（Task 2.1）：** 34/34 位患者在事件条数、绝对时间、参与矩阵、NaN 感知的 rank、触点数、记录块划分、封存分区边界七项上逐元素精确一致。详见 `results/epi_prssm/v0_1/baseline/CONTACT_RNN_PARITY.md`。

**v4.0 组件 reuse/adapt/reject 逐项判定：** `results/epi_prssm/v0_1/data_audit/V4_RECONCILIATION.md`。


## 3. just-in-time synthetic 标定

| truth | goal | 种子 | 可辨识 | 实际赢家 | 预注册期望 |
| --- | --- | --- | --- | --- | --- |
| `event_count_only` | goal4 | 3 | 3/3 | r3_events | event-count kernel beats the clock kernel |
| `event_rate_only_drift` | goal3 | 8 | 8/8 | g2 | rate moves, state does not: the nuisance control must catch it |
| `graph_recurrent_state` | goal1 | 8 | 8/8 | g1 | G1/G2 beat G0, and beat it open-loop |
| `hidden_common_cause` | goal4 | 3 | 3/3 | r1 | raw-load gain appears but the innovation challenge kills it |
| `latent_preictal_drift` | goal3 | 3 | 3/3 | g0 | state moves before onset beyond matched pseudo-onsets |
| `leaky_state` | goal1 | 8 | 8/8 | g2_frozen_node | G0 recovers it; graph recurrence adds nothing |
| `no_state` | goal1 | 11 | 11/11 | g2 | static wins or ties; a state model must not invent one |
| `no_state_false_adapter` | goal2 | 6 | 6/6 | node_film | adapter capacity alone must not create a state gain |
| `observer_overpowering` | goal1 | 3 | 3/3 | g0 | filtered ties; open-loop separates |
| `observer_resource_substitution` | goal4 | 3 | 1/3 | r1_flexible | flexible observer-resource correction imitates a resource |
| `r2_impulse` | goal4 | 3 | 3/3 | r1 | R2 beats matched R1 |
| `r3_integrated_exposure` | goal4 | 3 | 3/3 | r3_clock | R3 beats R2 and R1 at the generating timescale |
| `resource_direct_excitability` | goal4 | 3 | 3/3 | r3_clock | a resource acting straight on contact excitability is OUTSIDE the model family by contract; the arms should fail to recover it, which marks the boundary rather than refuting a resource |
| `state_conditioned_suffix` | goal2 | 6 | 6/6 | node_film | state adapters beat no_state; swap destroys the gain |
| `switching_state` | goal4 | 3 | 3/3 | g2 | a smooth resource cannot imitate switching |
| `t1_autonomous_resource` | goal4 | 3 | 2/3 | r0 | R1 beats R0; tau_r recoverable within an interval |


**这一轮 synthetic 直接改变了实验设计的两处：**

1. `no_state_false_adapter` 显示：把状态臂与「只有固定 repertoire」的 `static` 臂相比，大部分增益来自适配器自身的逐触点参数而不是状态。因此 H1 的第一级台阶改为与**容量配平的冻结状态臂**相比（`frozen_state_node`：适配器参数全在、状态逐触点但不随时间变）。
2. 资源类真值原本让资源直接改触点兴奋性，而 spec §5.1 明确禁止模型使用这条通路。已改写为资源调制「潜在状态到读出的增益」，并新增 `resource_direct_excitability` 真值把「模型族之外」这条边界显式画出来。旧版本的运行留在 `results/epi_prssm/v0_1/_invalidated_tau_parametrisation/synthetic/`。



## 4. H1：generator ladder

### 4.1 完整实验矩阵与每个运行的终态

| 臂 | seed | 状态 | epochs | 最优验证 | 用时(min) | 校正能量 | 时间常数中位数(s) | 稳定裕度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ct_ewma_g0 | 11 | COMPLETE | 10 | 2.09852 | 45 | 0.2832 | 882.0 | 0.00005 |
| ct_ewma_g0 | 12 | COMPLETE | 12 | 2.09709 | 59 | 0.3471 | 694.7 | 0.00011 |
| ct_ewma_g0 | 13 | COMPLETE | 12 | 2.09342 | 52 | 0.1246 | 2089.7 | 0.00005 |
| ct_ewma_g0_long_window | 11 | COMPLETE | 8 | 2.11391 | 80 | 0.1128 | 335.5 | 0.00004 |
| ct_ewma_g0_long_window | 12 | COMPLETE | 5 | 2.12557 | 55 | 0.3113 | 542.3 | 0.00005 |
| ct_ewma_g0_long_window | 13 | COMPLETE | 6 | 2.11566 | 64 | 0.0888 | 561.8 | 0.00004 |
| event_index_ewma | 11 | COMPLETE | 12 | 2.09822 | 64 | 0.2407 | 379.2 | 0.00005 |
| event_index_ewma | 12 | COMPLETE | 12 | 2.10613 | 54 | 0.1473 | 174.9 | 0.00011 |
| event_index_ewma | 13 | COMPLETE | 12 | 2.09253 | 56 | 0.2129 | 288.1 | 0.00014 |
| frozen_state | 11 | COMPLETE | 12 | 2.18145 | 25 | 0.0000 | 370.4 | 0.00009 |
| frozen_state | 12 | COMPLETE | 12 | 2.18145 | 26 | 0.0000 | 370.4 | 0.00009 |
| frozen_state | 13 | COMPLETE | 12 | 2.18145 | 25 | 0.0000 | 370.4 | 0.00009 |
| frozen_state_node | 11 | COMPLETE | 12 | 2.17431 | 26 | 0.0000 | 370.4 | 0.00009 |
| frozen_state_node | 12 | COMPLETE | 12 | 2.17742 | 28 | 0.0000 | 370.4 | 0.00009 |
| frozen_state_node | 13 | COMPLETE | 12 | 2.17461 | 28 | 0.0000 | 370.4 | 0.00009 |
| g1_graph_clds | 11 | COMPLETE | 12 | 2.08050 | 100 | 0.1241 | 309.4 | 0.00029 |
| g1_graph_clds | 12 | COMPLETE | 12 | 2.07702 | 98 | 0.4347 | 569.1 | 0.00014 |
| g1_graph_clds | 13 | COMPLETE | 12 | 2.07263 | 97 | 0.1468 | 308.2 | 0.00014 |
| g1_graph_clds_order_weighted | 11 | COMPLETE | 12 | 2.08554 | 276 | 0.1661 | 363.8 | 0.00016 |
| g1_graph_clds_order_weighted | 12 | COMPLETE | 12 | 2.08580 | 254 | 0.1912 | 333.2 | 0.00012 |
| g1_graph_clds_order_weighted | 13 | COMPLETE | 10 | 2.08551 | 237 | 0.1965 | 310.3 | 0.00011 |
| g2_compressed_state | 11 | COMPLETE | 12 | 2.10470 | 98 | 0.0956 | 5683.1 | 0.00009 |
| g2_compressed_state | 12 | COMPLETE | 11 | 2.10037 | 92 | 0.1156 | 5213.7 | 0.00010 |
| g2_compressed_state | 13 | COMPLETE | 12 | 2.10124 | 98 | 0.1736 | 6179.0 | 0.00008 |
| g2_graph_gru_ode | 11 | COMPLETE | 12 | 2.06653 | 107 | 0.1414 | 372.9 | 0.00010 |
| g2_graph_gru_ode | 12 | COMPLETE | 8 | 2.08276 | 80 | 0.2318 | 362.5 | 0.00009 |
| g2_graph_gru_ode | 13 | COMPLETE | 10 | 2.08461 | 94 | 0.3270 | 322.6 | 0.00006 |
| g2_graph_gru_ode_long_window | 11 | COMPLETE | 8 | 2.08320 | 222 | 0.1278 | 329.6 | 0.00014 |
| g2_graph_gru_ode_long_window | 12 | COMPLETE | 8 | 2.11471 | 238 | 0.4920 | 871.5 | 0.00023 |
| g2_graph_gru_ode_long_window | 13 | COMPLETE | 7 | 2.10295 | 178 | 0.1545 | 302.5 | 0.00013 |
| g3_flexible_resource_control | 11 | COMPLETE | 12 | 2.07075 | 120 | 0.2916 | 244.3 | 0.00014 |
| g3_flexible_resource_control | 12 | COMPLETE | 12 | 2.08639 | 119 | 0.2175 | 301.2 | 0.00031 |
| g3_flexible_resource_control | 13 | COMPLETE | 12 | 2.08116 | 119 | 0.2982 | 360.8 | 0.00012 |
| g3_resource | 11 | COMPLETE | 12 | 2.07116 | 117 | 0.1915 | 358.5 | 0.00009 |
| g3_resource | 12 | COMPLETE | 12 | 2.07035 | 117 | 0.1792 | 331.1 | 0.00008 |
| g3_resource | 13 | COMPLETE | 9 | 2.07588 | 97 | 0.2531 | 267.6 | 0.00010 |
| g3_resource_on_g1 | 11 | COMPLETE | 12 | 2.07786 | 280 | 0.2651 | 433.3 | 0.00008 |
| g3_resource_on_g1 | 12 | COMPLETE | 7 | 2.08036 | 152 | 0.1879 | 356.4 | 0.00023 |
| g3_resource_on_g1 | 13 | COMPLETE | 7 | 2.08590 | 162 | 0.1799 | 368.0 | 0.00010 |
| nuisance_timing_baseline | 11 | COMPLETE | 12 | 2.19228 | 23 | 0.0000 | 370.4 | 0.00009 |
| nuisance_timing_baseline | 12 | COMPLETE | 8 | 2.20228 | 16 | 0.0000 | 370.4 | 0.00009 |
| nuisance_timing_baseline | 13 | COMPLETE | 11 | 2.19134 | 22 | 0.0000 | 370.4 | 0.00009 |
| nuisance_timing_baseline_order_weighted | 11 | COMPLETE | 10 | 2.19337 | 70 | 0.0000 | 370.4 | 0.00009 |
| nuisance_timing_baseline_order_weighted | 12 | COMPLETE | 12 | 2.19207 | 84 | 0.0000 | 370.4 | 0.00009 |
| nuisance_timing_baseline_order_weighted | 13 | COMPLETE | 12 | 2.19360 | 93 | 0.0000 | 370.4 | 0.00009 |
| static | 11 | COMPLETE | 12 | 2.18128 | 47 | 0.0040 | 357.9 | 0.00008 |
| static | 12 | COMPLETE | 12 | 2.18128 | 48 | 0.0090 | 371.2 | 0.00010 |
| static | 13 | COMPLETE | 12 | 2.18128 | 47 | 0.0029 | 338.4 | 0.00010 |
| unconstrained_gru | 11 | COMPLETE | 11 | 2.17601 | 45 | 0.1853 | 370.4 | 0.00009 |
| unconstrained_gru | 12 | COMPLETE | 12 | 2.18029 | 50 | 0.2208 | 370.4 | 0.00009 |
| unconstrained_gru | 13 | COMPLETE | 12 | 2.18173 | 48 | 0.0958 | 370.4 | 0.00009 |


### 4.2 逐级台阶的患者级成对效应（主端点 event NLL）

| 对比 | 中位差 | 95% CI | 方向有利 | 符号检验 p | Wilcoxon p |
| --- | --- | --- | --- | --- | --- |
| ct_ewma_g0 - frozen_state_node | -0.0306 | [-0.0503, -0.0085] | 27/34 | 0.000821 | 4.66e-05 |
| g1_graph_clds - ct_ewma_g0 | -0.0104 | [-0.0162, -0.0056] | 29/34 | 3.86e-05 | 2.83e-06 |
| g2_graph_gru_ode - g1_graph_clds | +0.0047 | [+0.0003, +0.0126] | 11/34 | 0.0576 | 0.0065 |
| g3_resource - g2_graph_gru_ode | -0.0066 | [-0.0074, -0.0037] | 28/34 | 0.000195 | 3.16e-05 |
| g1_graph_clds_order_weighted - nuisance_timing_baseline_order_weighted | -0.0736 | [-0.0878, -0.0440] | 34/34 | 1.16e-10 | 1.16e-10 |
| g1_graph_clds_order_weighted - g1_graph_clds | +0.0052 | [+0.0010, +0.0102] | 8/34 | 0.00294 | 0.00282 |
| g3_resource_on_g1 - g1_graph_clds | +0.0087 | [+0.0026, +0.0175] | 3/34 | 7.66e-07 | 3.22e-07 |
| g3_resource - g1_graph_clds | -0.0002 | [-0.0017, +0.0026] | 18/34 | 0.864 | 0.612 |
| g3_resource_on_g1 - nuisance_timing_baseline | -0.0595 | [-0.0887, -0.0490] | 29/34 | 3.86e-05 | 1.73e-07 |
| ct_ewma_g0 - nuisance_timing_baseline | -0.0621 | [-0.0897, -0.0444] | 29/34 | 3.86e-05 | 7.45e-08 |
| g1_graph_clds - nuisance_timing_baseline | -0.0745 | [-0.0923, -0.0570] | 32/34 | 6.94e-08 | 2.21e-09 |
| g2_graph_gru_ode - nuisance_timing_baseline | -0.0694 | [-0.0964, -0.0477] | 32/34 | 6.94e-08 | 2.91e-09 |
| g3_resource - nuisance_timing_baseline | -0.0768 | [-0.0974, -0.0548] | 32/34 | 6.94e-08 | 1.16e-09 |
| nuisance_timing_baseline - frozen_state_node | +0.0179 | [+0.0072, +0.0355] | 6/34 | 0.000195 | 8.1e-06 |
| g2_graph_gru_ode_long_window - g2_graph_gru_ode | +0.0243 | [+0.0078, +0.0336] | 8/34 | 0.00294 | 1.13e-05 |
| ct_ewma_g0_long_window - ct_ewma_g0 | +0.0094 | [+0.0065, +0.0152] | 5/34 | 3.86e-05 | 0.000164 |
| frozen_state - static | -0.0002 | [-0.0027, +0.0003] | 19/34 | 0.608 | 0.0976 |
| frozen_state_node - frozen_state | -0.0004 | [-0.0076, +0.0033] | 18/34 | 0.864 | 0.326 |
| ct_ewma_g0 - frozen_state | -0.0327 | [-0.0569, -0.0203] | 25/34 | 0.00904 | 4.24e-05 |
| ct_ewma_g0 - static | -0.0366 | [-0.0640, -0.0197] | 25/34 | 0.00904 | 2.87e-05 |
| unconstrained_gru - ct_ewma_g0 | +0.0282 | [+0.0066, +0.0560] | 11/34 | 0.0576 | 0.000126 |
| event_index_ewma - ct_ewma_g0 | -0.0035 | [-0.0074, -0.0017] | 29/34 | 3.86e-05 | 5.12e-05 |
| g3_flexible_resource_control - g3_resource | +0.0045 | [+0.0007, +0.0061] | 8/34 | 0.00294 | 0.0842 |
| g2_compressed_state - g2_graph_gru_ode | +0.0063 | [+0.0029, +0.0099] | 7/34 | 0.000821 | 0.00128 |


### 4.3 掩蔽顺序端点（与参与人数无关）

| 对比 | 中位差 | 95% CI | 方向有利 | 符号检验 p | Wilcoxon p |
| --- | --- | --- | --- | --- | --- |
| ct_ewma_g0 - frozen_state_node | +0.0285 | [+0.0131, +0.0417] | 5/34 | 3.86e-05 | 2.83e-06 |
| g1_graph_clds - ct_ewma_g0 | -0.0130 | [-0.0180, -0.0098] | 26/34 | 0.00294 | 0.000465 |
| g2_graph_gru_ode - g1_graph_clds | +0.0004 | [-0.0010, +0.0066] | 14/34 | 0.392 | 0.334 |
| g3_resource - g2_graph_gru_ode | -0.0014 | [-0.0038, +0.0001] | 22/34 | 0.121 | 0.153 |
| g1_graph_clds_order_weighted - nuisance_timing_baseline_order_weighted | +0.0045 | [-0.0036, +0.0085] | 13/34 | 0.229 | 0.242 |
| g1_graph_clds_order_weighted - g1_graph_clds | -0.0241 | [-0.0342, -0.0195] | 34/34 | 1.16e-10 | 1.16e-10 |
| g3_resource_on_g1 - g1_graph_clds | +0.0103 | [+0.0061, +0.0178] | 2/34 | 6.94e-08 | 5.82e-10 |
| g3_resource - g1_graph_clds | +0.0015 | [-0.0026, +0.0085] | 16/34 | 0.864 | 0.317 |
| g3_resource_on_g1 - nuisance_timing_baseline | +0.0280 | [+0.0196, +0.0377] | 3/34 | 7.66e-07 | 7.45e-08 |
| ct_ewma_g0 - nuisance_timing_baseline | +0.0303 | [+0.0196, +0.0427] | 5/34 | 3.86e-05 | 2.21e-06 |
| g1_graph_clds - nuisance_timing_baseline | +0.0174 | [+0.0027, +0.0280] | 9/34 | 0.00904 | 0.00021 |
| g2_graph_gru_ode - nuisance_timing_baseline | +0.0164 | [+0.0050, +0.0314] | 9/34 | 0.00904 | 0.000126 |
| g3_resource - nuisance_timing_baseline | +0.0194 | [+0.0061, +0.0324] | 9/34 | 0.00904 | 0.000116 |
| nuisance_timing_baseline - frozen_state_node | -0.0000 | [-0.0056, +0.0034] | 17/34 | 1 | 0.826 |
| g2_graph_gru_ode_long_window - g2_graph_gru_ode | -0.0052 | [-0.0111, -0.0014] | 25/34 | 0.00904 | 0.0118 |
| ct_ewma_g0_long_window - ct_ewma_g0 | -0.0111 | [-0.0161, -0.0073] | 23/34 | 0.0576 | 0.0101 |
| frozen_state - static | +0.0012 | [-0.0008, +0.0032] | 13/34 | 0.229 | 0.0811 |
| frozen_state_node - frozen_state | +0.0002 | [-0.0031, +0.0029] | 17/34 | 1 | 0.478 |
| ct_ewma_g0 - frozen_state | +0.0327 | [+0.0147, +0.0403] | 4/34 | 6.16e-06 | 2.37e-07 |
| ct_ewma_g0 - static | +0.0355 | [+0.0176, +0.0389] | 4/34 | 6.16e-06 | 1.47e-07 |
| unconstrained_gru - ct_ewma_g0 | -0.0341 | [-0.0458, -0.0224] | 31/34 | 7.66e-07 | 3.57e-08 |
| event_index_ewma - ct_ewma_g0 | -0.0072 | [-0.0136, -0.0045] | 30/34 | 6.16e-06 | 5.02e-07 |
| g3_flexible_resource_control - g3_resource | +0.0043 | [+0.0013, +0.0075] | 11/34 | 0.0576 | 0.00319 |
| g2_compressed_state - g2_graph_gru_ode | +0.0082 | [+0.0036, +0.0114] | 10/34 | 0.0243 | 0.0146 |


### 4.4 开环（观测关闭）逐 horizon

| 臂 | H5 | H10 | H20 | H40 |
| --- | --- | --- | --- | --- |
| ct_ewma_g0 | -0.0388 | -0.0347 | -0.0353 | -0.0238 |
| ct_ewma_g0_long_window | -0.0298 | -0.0180 | -0.0060 | +0.0401 |
| event_index_ewma | -0.0334 | +0.0024 | +0.1279 | +0.4651 |
| frozen_state | -0.0015 | -0.0012 | -0.0008 | -0.0004 |
| frozen_state_node | -0.0013 | -0.0022 | -0.0037 | -0.0032 |
| g1_graph_clds | -0.0521 | -0.0455 | -0.0424 | -0.0393 |
| g1_graph_clds_order_weighted | -0.0459 | -0.0398 | -0.0387 | -0.0337 |
| g2_compressed_state | -0.0375 | -0.0322 | -0.0283 | -0.0275 |
| g2_graph_gru_ode | -0.0432 | -0.0382 | -0.0406 | -0.0390 |
| g2_graph_gru_ode_long_window | -0.0250 | -0.0184 | -0.0183 | -0.0125 |
| g3_flexible_resource_control | -0.0498 | -0.0438 | -0.0461 | -0.0423 |
| g3_resource | -0.0487 | -0.0457 | -0.0493 | -0.0482 |
| g3_resource_on_g1 | -0.0384 | -0.0344 | -0.0370 | -0.0334 |
| nuisance_timing_baseline | +0.0151 | +0.0116 | +0.0126 | +0.0121 |
| nuisance_timing_baseline_order_weighted | +0.0219 | +0.0218 | +0.0215 | +0.0208 |
| static | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| unconstrained_gru | +0.0004 | +0.0039 | +0.0066 | +0.0188 |


### 4.5 状态清零恢复曲线

| horizon (events) | 中位 NLL 惩罚 | n |
| --- | --- | --- |
| 1 | +8.58534 | 578 |
| 2 | +6.87097 | 578 |
| 5 | +2.39566 | 578 |
| 10 | +0.08750 | 578 |
| 20 | +0.00924 | 578 |
| 40 | +0.00398 | 578 |
| 80 | +0.00024 | 578 |

### 4.6 真实间隔打乱

| 臂 | 中位 NLL 惩罚 | n 患者 |
| --- | --- | --- |
| ct_ewma_g0 | +0.00054 | 34 |
| ct_ewma_g0_long_window | +0.00089 | 34 |
| event_index_ewma | +0.00012 | 34 |
| frozen_state | +0.00000 | 34 |
| frozen_state_node | +0.00000 | 34 |
| g1_graph_clds | +0.00118 | 34 |
| g1_graph_clds_order_weighted | +0.00031 | 34 |
| g2_compressed_state | +0.00000 | 34 |
| g2_graph_gru_ode | +0.00044 | 34 |
| g2_graph_gru_ode_long_window | -0.00015 | 34 |
| g3_flexible_resource_control | +0.00064 | 34 |
| g3_resource | +0.00028 | 34 |
| g3_resource_on_g1 | +0.00067 | 34 |
| nuisance_timing_baseline | +0.00000 | 34 |
| nuisance_timing_baseline_order_weighted | +0.00000 | 34 |
| static | +0.00000 | 34 |
| unconstrained_gru | -0.00031 | 34 |

### 4.7 Holm 校正（主家族）

```json
{
  "event_nll::g1_graph_clds-vs-ct_ewma_g0": 0.00015423260629177094,
  "event_nll::g3_resource-vs-g2_graph_gru_ode": 0.000585376750677824,
  "event_nll::ct_ewma_g0-vs-frozen_state_node": 0.0016427906230092049,
  "event_nll::g2_graph_gru_ode-vs-g1_graph_clds": 0.05761267291381955
}
```


## 5. H2a：state-conditioned readout

### 5.1 适配器容量 vs 状态（capacity-matched）

| 端点 | 适配器 | 状态源 | 对比 | 中位差 | 95% CI | 方向有利 | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| event_nll | initial_state | g0 | state vs frozen-state (capacity-matched) | +0.0001 | [+0.0000, +0.0003] | 10/34 | 0.0243 |
| event_nll | initial_state | g2 | state vs frozen-state (capacity-matched) | +0.0002 | [+0.0001, +0.0007] | 8/34 | 0.00294 |
| event_nll | initial_state | g3 | state vs frozen-state (capacity-matched) | +0.0001 | [+0.0000, +0.0006] | 8/34 | 0.00294 |
| event_nll | initial_state | frozen | adapter capacity alone | +0.0000 | [-0.0001, +0.0001] | 14/34 | 0.392 |
| event_nll | node_film | g0 | state vs frozen-state (capacity-matched) | -0.0306 | [-0.0503, -0.0085] | 27/34 | 0.000821 |
| event_nll | node_film | g2 | state vs frozen-state (capacity-matched) | -0.0349 | [-0.0552, -0.0161] | 28/34 | 0.000195 |
| event_nll | node_film | g3 | state vs frozen-state (capacity-matched) | -0.0443 | [-0.0642, -0.0189] | 29/34 | 3.86e-05 |
| event_nll | node_film | frozen | adapter capacity alone | -0.0031 | [-0.0069, +0.0033] | 19/34 | 0.608 |
| event_nll | edge_gate | g0 | state vs frozen-state (capacity-matched) | -0.0451 | [-0.0575, -0.0247] | 30/34 | 6.16e-06 |
| event_nll | edge_gate | g2 | state vs frozen-state (capacity-matched) | -0.0460 | [-0.0615, -0.0255] | 30/34 | 6.16e-06 |
| event_nll | edge_gate | g3 | state vs frozen-state (capacity-matched) | -0.0455 | [-0.0570, -0.0223] | 28/34 | 0.000195 |
| event_nll | edge_gate | frozen | adapter capacity alone | -0.0021 | [-0.0061, +0.0043] | 18/34 | 0.864 |
| order_nll | initial_state | g0 | state vs frozen-state (capacity-matched) | -0.0000 | [-0.0000, -0.0000] | 29/34 | 3.86e-05 |
| order_nll | initial_state | g2 | state vs frozen-state (capacity-matched) | +0.0000 | [+0.0000, +0.0001] | 8/34 | 0.00294 |
| order_nll | initial_state | g3 | state vs frozen-state (capacity-matched) | -0.0000 | [-0.0000, -0.0000] | 29/34 | 1.09e-05 |
| order_nll | initial_state | frozen | adapter capacity alone | +0.0000 | [+0.0000, +0.0000] | 0/34 | 0.125 |
| order_nll | node_film | g0 | state vs frozen-state (capacity-matched) | +0.0285 | [+0.0131, +0.0417] | 5/34 | 3.86e-05 |
| order_nll | node_film | g2 | state vs frozen-state (capacity-matched) | +0.0168 | [+0.0064, +0.0281] | 8/34 | 0.00294 |
| order_nll | node_film | g3 | state vs frozen-state (capacity-matched) | +0.0217 | [+0.0106, +0.0247] | 9/34 | 0.00904 |
| order_nll | node_film | frozen | adapter capacity alone | +0.0008 | [-0.0028, +0.0055] | 16/34 | 0.864 |
| order_nll | edge_gate | g0 | state vs frozen-state (capacity-matched) | +0.0259 | [+0.0124, +0.0325] | 5/34 | 3.86e-05 |
| order_nll | edge_gate | g2 | state vs frozen-state (capacity-matched) | +0.0170 | [+0.0041, +0.0215] | 10/34 | 0.0243 |
| order_nll | edge_gate | g3 | state vs frozen-state (capacity-matched) | +0.0150 | [+0.0069, +0.0217] | 5/34 | 3.86e-05 |
| order_nll | edge_gate | frozen | adapter capacity alone | +0.0008 | [-0.0032, +0.0062] | 15/34 | 0.608 |
| selection_nll | initial_state | g0 | state vs frozen-state (capacity-matched) | +0.0000 | [+0.0000, +0.0000] | 10/34 | 0.0243 |
| selection_nll | initial_state | g2 | state vs frozen-state (capacity-matched) | -0.0001 | [-0.0001, -0.0000] | 24/34 | 0.0243 |
| selection_nll | initial_state | g3 | state vs frozen-state (capacity-matched) | +0.0000 | [+0.0000, +0.0000] | 9/34 | 0.00904 |
| selection_nll | initial_state | frozen | adapter capacity alone | +0.0000 | [+0.0000, +0.0000] | 4/34 | 0.688 |
| selection_nll | node_film | g0 | state vs frozen-state (capacity-matched) | -0.0434 | [-0.0570, -0.0183] | 28/34 | 0.000195 |
| selection_nll | node_film | g2 | state vs frozen-state (capacity-matched) | -0.0447 | [-0.0647, -0.0265] | 28/34 | 0.000195 |
| selection_nll | node_film | g3 | state vs frozen-state (capacity-matched) | -0.0490 | [-0.0701, -0.0285] | 30/34 | 6.16e-06 |
| selection_nll | node_film | frozen | adapter capacity alone | -0.0032 | [-0.0063, +0.0021] | 20/34 | 0.392 |
| selection_nll | edge_gate | g0 | state vs frozen-state (capacity-matched) | -0.0476 | [-0.0585, -0.0269] | 29/34 | 3.86e-05 |
| selection_nll | edge_gate | g2 | state vs frozen-state (capacity-matched) | -0.0510 | [-0.0665, -0.0339] | 31/34 | 7.66e-07 |
| selection_nll | edge_gate | g3 | state vs frozen-state (capacity-matched) | -0.0470 | [-0.0638, -0.0285] | 28/34 | 0.000195 |
| selection_nll | edge_gate | frozen | adapter capacity alone | -0.0003 | [-0.0047, +0.0029] | 17/34 | 1 |
| stop_nll | initial_state | g0 | state vs frozen-state (capacity-matched) | +0.0001 | [+0.0000, +0.0004] | 7/34 | 0.000821 |
| stop_nll | initial_state | g2 | state vs frozen-state (capacity-matched) | +0.0003 | [+0.0001, +0.0009] | 6/34 | 0.000195 |
| stop_nll | initial_state | g3 | state vs frozen-state (capacity-matched) | +0.0001 | [+0.0000, +0.0005] | 7/34 | 0.000821 |
| stop_nll | initial_state | frozen | adapter capacity alone | +0.0000 | [-0.0001, +0.0001] | 14/34 | 0.392 |
| stop_nll | node_film | g0 | state vs frozen-state (capacity-matched) | +0.0001 | [+0.0001, +0.0009] | 5/34 | 3.86e-05 |
| stop_nll | node_film | g2 | state vs frozen-state (capacity-matched) | +0.0006 | [+0.0003, +0.0023] | 3/34 | 7.66e-07 |
| stop_nll | node_film | g3 | state vs frozen-state (capacity-matched) | +0.0002 | [+0.0001, +0.0014] | 6/34 | 0.000195 |
| stop_nll | node_film | frozen | adapter capacity alone | +0.0000 | [-0.0001, +0.0001] | 15/34 | 0.608 |
| stop_nll | edge_gate | g0 | state vs frozen-state (capacity-matched) | +0.0002 | [+0.0001, +0.0008] | 7/34 | 0.000821 |
| stop_nll | edge_gate | g2 | state vs frozen-state (capacity-matched) | +0.0002 | [+0.0001, +0.0009] | 6/34 | 0.000195 |
| stop_nll | edge_gate | g3 | state vs frozen-state (capacity-matched) | +0.0001 | [+0.0000, +0.0021] | 7/34 | 0.000821 |
| stop_nll | edge_gate | frozen | adapter capacity alone | +0.0000 | [-0.0001, +0.0001] | 15/34 | 0.608 |
| participation_nll | initial_state | g0 | state vs frozen-state (capacity-matched) | -0.0057 | [-0.0135, +0.0003] | 22/34 | 0.121 |
| participation_nll | initial_state | g2 | state vs frozen-state (capacity-matched) | -0.0227 | [-0.0294, -0.0159] | 33/34 | 4.07e-09 |
| participation_nll | initial_state | g3 | state vs frozen-state (capacity-matched) | -0.0203 | [-0.0293, -0.0124] | 32/34 | 6.94e-08 |
| participation_nll | initial_state | frozen | adapter capacity alone | +0.0021 | [+0.0009, +0.0038] | 8/34 | 0.00294 |
| participation_nll | node_film | g0 | state vs frozen-state (capacity-matched) | -0.0042 | [-0.0142, +0.0001] | 22/34 | 0.121 |
| participation_nll | node_film | g2 | state vs frozen-state (capacity-matched) | -0.0201 | [-0.0298, -0.0147] | 33/34 | 4.07e-09 |
| participation_nll | node_film | g3 | state vs frozen-state (capacity-matched) | -0.0226 | [-0.0315, -0.0148] | 33/34 | 4.07e-09 |
| participation_nll | node_film | frozen | adapter capacity alone | +0.0021 | [+0.0009, +0.0038] | 7/34 | 0.000821 |
| participation_nll | edge_gate | g0 | state vs frozen-state (capacity-matched) | -0.0015 | [-0.0105, +0.0024] | 19/34 | 0.608 |
| participation_nll | edge_gate | g2 | state vs frozen-state (capacity-matched) | -0.0216 | [-0.0302, -0.0136] | 34/34 | 1.16e-10 |
| participation_nll | edge_gate | g3 | state vs frozen-state (capacity-matched) | -0.0197 | [-0.0318, -0.0137] | 33/34 | 4.07e-09 |
| participation_nll | edge_gate | frozen | adapter capacity alone | +0.0021 | [+0.0009, +0.0038] | 8/34 | 0.00294 |

### 5.2 状态互换反事实

| 端点 | 臂 | 互换方式 | 中位差 | 方向有利 |
| --- | --- | --- | --- | --- |
| event_nll | edge_gate_frozen | swap_matched | +0.00000 | 0/34 |
| event_nll | edge_gate_frozen | swap_random | +0.00000 | 0/34 |
| event_nll | edge_gate_g0 | swap_matched | -0.02266 | 32/34 |
| event_nll | edge_gate_g0 | swap_random | -0.03296 | 33/34 |
| event_nll | edge_gate_g2 | swap_matched | -0.02663 | 31/34 |
| event_nll | edge_gate_g2 | swap_random | -0.03308 | 32/34 |
| event_nll | edge_gate_g3 | swap_matched | -0.02174 | 32/34 |
| event_nll | edge_gate_g3 | swap_random | -0.03056 | 32/34 |
| event_nll | edge_gate_only_g0 | swap_matched | -0.00000 | 18/34 |
| event_nll | edge_gate_only_g0 | swap_random | -0.00000 | 23/34 |
| event_nll | edge_gate_only_g2 | swap_matched | -0.00000 | 22/34 |
| event_nll | edge_gate_only_g2 | swap_random | -0.00000 | 25/34 |
| event_nll | edge_gate_only_g3 | swap_matched | -0.00000 | 22/34 |
| event_nll | edge_gate_only_g3 | swap_random | -0.00000 | 20/34 |
| event_nll | initial_state_frozen | swap_matched | +0.00000 | 0/34 |
| event_nll | initial_state_frozen | swap_random | +0.00000 | 0/34 |
| event_nll | initial_state_g0 | swap_matched | -0.00000 | 20/34 |
| event_nll | initial_state_g0 | swap_random | -0.00000 | 22/34 |
| event_nll | initial_state_g2 | swap_matched | -0.00000 | 17/34 |
| event_nll | initial_state_g2 | swap_random | -0.00000 | 19/34 |
| event_nll | initial_state_g3 | swap_matched | +0.00000 | 16/34 |
| event_nll | initial_state_g3 | swap_random | -0.00000 | 21/34 |
| event_nll | no_state | swap_matched | +0.00000 | 0/34 |
| event_nll | no_state | swap_random | +0.00000 | 0/34 |
| event_nll | node_film_frozen | swap_matched | +0.00000 | 0/34 |
| event_nll | node_film_frozen | swap_random | +0.00000 | 0/34 |
| event_nll | node_film_g0 | swap_matched | -0.02827 | 32/34 |
| event_nll | node_film_g0 | swap_random | -0.03527 | 33/34 |
| event_nll | node_film_g2 | swap_matched | -0.02084 | 32/34 |
| event_nll | node_film_g2 | swap_random | -0.03147 | 33/34 |
| event_nll | node_film_g3 | swap_matched | -0.02475 | 32/34 |
| event_nll | node_film_g3 | swap_random | -0.03227 | 33/34 |
| order_nll | edge_gate_frozen | swap_matched | +0.00000 | 0/34 |
| order_nll | edge_gate_frozen | swap_random | +0.00000 | 0/34 |
| order_nll | edge_gate_g0 | swap_matched | -0.00868 | 30/34 |
| order_nll | edge_gate_g0 | swap_random | -0.01206 | 32/34 |
| order_nll | edge_gate_g2 | swap_matched | -0.00787 | 27/34 |
| order_nll | edge_gate_g2 | swap_random | -0.01082 | 31/34 |
| order_nll | edge_gate_g3 | swap_matched | -0.00557 | 29/34 |
| order_nll | edge_gate_g3 | swap_random | -0.00977 | 31/34 |
| order_nll | edge_gate_only_g0 | swap_matched | +0.00000 | 1/34 |
| order_nll | edge_gate_only_g0 | swap_random | +0.00000 | 2/34 |
| order_nll | edge_gate_only_g2 | swap_matched | +0.00000 | 0/34 |
| order_nll | edge_gate_only_g2 | swap_random | +0.00000 | 2/34 |
| order_nll | edge_gate_only_g3 | swap_matched | +0.00000 | 4/34 |
| order_nll | edge_gate_only_g3 | swap_random | +0.00000 | 5/34 |
| order_nll | initial_state_frozen | swap_matched | +0.00000 | 0/34 |
| order_nll | initial_state_frozen | swap_random | +0.00000 | 0/34 |
| order_nll | initial_state_g0 | swap_matched | +0.00000 | 4/34 |
| order_nll | initial_state_g0 | swap_random | +0.00000 | 2/34 |
| order_nll | initial_state_g2 | swap_matched | +0.00000 | 2/34 |
| order_nll | initial_state_g2 | swap_random | +0.00000 | 4/34 |
| order_nll | initial_state_g3 | swap_matched | +0.00000 | 1/34 |
| order_nll | initial_state_g3 | swap_random | +0.00000 | 6/34 |
| order_nll | no_state | swap_matched | +0.00000 | 0/34 |
| order_nll | no_state | swap_random | +0.00000 | 0/34 |
| order_nll | node_film_frozen | swap_matched | +0.00000 | 0/34 |
| order_nll | node_film_frozen | swap_random | +0.00000 | 0/34 |
| order_nll | node_film_g0 | swap_matched | -0.01046 | 32/34 |
| order_nll | node_film_g0 | swap_random | -0.01357 | 31/34 |
| order_nll | node_film_g2 | swap_matched | -0.00827 | 29/34 |
| order_nll | node_film_g2 | swap_random | -0.01113 | 31/34 |
| order_nll | node_film_g3 | swap_matched | -0.00862 | 29/34 |
| order_nll | node_film_g3 | swap_random | -0.01239 | 31/34 |
| selection_nll | edge_gate_frozen | swap_matched | +0.00000 | 0/34 |
| selection_nll | edge_gate_frozen | swap_random | +0.00000 | 0/34 |
| selection_nll | edge_gate_g0 | swap_matched | -0.02125 | 32/34 |
| selection_nll | edge_gate_g0 | swap_random | -0.03266 | 33/34 |
| selection_nll | edge_gate_g2 | swap_matched | -0.02344 | 32/34 |
| selection_nll | edge_gate_g2 | swap_random | -0.03314 | 33/34 |
| selection_nll | edge_gate_g3 | swap_matched | -0.02109 | 33/34 |
| selection_nll | edge_gate_g3 | swap_random | -0.02978 | 32/34 |
| selection_nll | edge_gate_only_g0 | swap_matched | +0.00000 | 3/34 |
| selection_nll | edge_gate_only_g0 | swap_random | +0.00000 | 3/34 |
| selection_nll | edge_gate_only_g2 | swap_matched | +0.00000 | 0/34 |
| selection_nll | edge_gate_only_g2 | swap_random | +0.00000 | 0/34 |
| selection_nll | edge_gate_only_g3 | swap_matched | +0.00000 | 4/34 |
| selection_nll | edge_gate_only_g3 | swap_random | +0.00000 | 2/34 |
| selection_nll | initial_state_frozen | swap_matched | +0.00000 | 0/34 |
| selection_nll | initial_state_frozen | swap_random | +0.00000 | 0/34 |
| selection_nll | initial_state_g0 | swap_matched | +0.00000 | 0/34 |
| selection_nll | initial_state_g0 | swap_random | +0.00000 | 1/34 |
| selection_nll | initial_state_g2 | swap_matched | +0.00000 | 2/34 |
| selection_nll | initial_state_g2 | swap_random | +0.00000 | 2/34 |
| selection_nll | initial_state_g3 | swap_matched | +0.00000 | 1/34 |
| selection_nll | initial_state_g3 | swap_random | +0.00000 | 0/34 |
| selection_nll | no_state | swap_matched | +0.00000 | 0/34 |
| selection_nll | no_state | swap_random | +0.00000 | 0/34 |
| selection_nll | node_film_frozen | swap_matched | +0.00000 | 0/34 |
| selection_nll | node_film_frozen | swap_random | +0.00000 | 0/34 |
| selection_nll | node_film_g0 | swap_matched | -0.02890 | 33/34 |
| selection_nll | node_film_g0 | swap_random | -0.03486 | 33/34 |
| selection_nll | node_film_g2 | swap_matched | -0.02088 | 30/34 |
| selection_nll | node_film_g2 | swap_random | -0.03162 | 33/34 |
| selection_nll | node_film_g3 | swap_matched | -0.02576 | 30/34 |
| selection_nll | node_film_g3 | swap_random | -0.03329 | 32/34 |

### 5.3 歧义前缀定向分析

- targeted eligible：31 位
- not eligible（记为不适用，不是阴性）：3 位

| 前缀深度 | 中位增益 | 95% CI | 方向有利 | p |
| --- | --- | --- | --- | --- |
| by_arm | — | [—, —] | —/— | — |
| negative_control_arms | — | [—, —] | —/— | — |
| reading | 无可用配对 | — | — | — |

### 5.4 冻结 TA/TB 投影

`NOT_RUN`：TA/TB 模板标签是本模型族的禁止输入，且本队列没有释放冻结的下游投影。


## 6. H2b：frozen interictal → seizure link

### 6.1 观测流重建

- 被试 34；重建流 1,445,115 事件 vs 冻结的确定间期流 864,163 （1.67 倍，多出 580,952）
- 冻结流事件在重建流中的最低复原率：1.0000
- 编码逐元素一致（参与 / 组标识 / rank）：True（全部被试）
- 通道顺序与冻结队列一致；块选择只筛事件，不改通道顺序
- 构建重建流时**未读取任何发作标签**

| 患者 | 重建流 | 冻结流 | 倍数 | 复原率 |
| --- | --- | --- | --- | --- |
| epilepsiae_1073 | 193171 | 138275 | 1.40 | 1.0000 |
| epilepsiae_1077 | 54432 | 32376 | 1.68 | 1.0000 |
| epilepsiae_1084 | 11169 | 7318 | 1.53 | 1.0000 |
| epilepsiae_1096 | 223212 | 140337 | 1.59 | 1.0000 |
| epilepsiae_1125 | 70681 | 35971 | 1.96 | 1.0000 |
| epilepsiae_1146 | 46683 | 12430 | 3.76 | 1.0000 |
| epilepsiae_1150 | 12362 | 8561 | 1.44 | 1.0000 |
| epilepsiae_139 | 14438 | 9184 | 1.57 | 1.0000 |
| epilepsiae_253 | 75053 | 50747 | 1.48 | 1.0000 |
| epilepsiae_384 | 42533 | 21495 | 1.98 | 1.0000 |
| epilepsiae_442 | 6556 | 2667 | 2.46 | 1.0000 |
| epilepsiae_548 | 25282 | 6982 | 3.62 | 1.0000 |
| epilepsiae_583 | 10967 | 7828 | 1.40 | 1.0000 |
| epilepsiae_590 | 7579 | 4982 | 1.52 | 1.0000 |
| epilepsiae_620 | 30648 | 22408 | 1.37 | 1.0000 |
| epilepsiae_635 | 13973 | 5111 | 2.73 | 1.0000 |
| epilepsiae_922 | 243990 | 83638 | 2.92 | 1.0000 |
| epilepsiae_958 | 165577 | 123419 | 1.34 | 1.0000 |
| yuquan_chengshuai | 27577 | 27577 | 1.00 | 1.0000 |
| yuquan_chenziyang | 9609 | 9609 | 1.00 | 1.0000 |
| yuquan_gaolan | 7451 | 2993 | 2.49 | 1.0000 |
| yuquan_hanyuxuan | 5468 | 5468 | 1.00 | 1.0000 |
| yuquan_huanghanwen | 484 | 456 | 1.06 | 1.0000 |
| yuquan_litengsheng | 2070 | 642 | 3.22 | 1.0000 |
| yuquan_liyouran | 2346 | 2346 | 1.00 | 1.0000 |
| yuquan_pengzihang | 46055 | 22622 | 2.04 | 1.0000 |
| yuquan_songzishuo | 447 | 447 | 1.00 | 1.0000 |
| yuquan_sunyuanxin | 5085 | 1282 | 3.97 | 1.0000 |
| yuquan_wangyiyang | 1919 | 1919 | 1.00 | 1.0000 |
| yuquan_xuxinyi | 9646 | 6663 | 1.45 | 1.0000 |
| yuquan_zhangbichen | 8371 | 8371 | 1.00 | 1.0000 |
| yuquan_zhangjiaqi | 48277 | 48277 | 1.00 | 1.0000 |
| yuquan_zhangkexuan | 18190 | 7948 | 2.29 | 1.0000 |
| yuquan_zhaochenxi | 3814 | 3814 | 1.00 | 1.0000 |

### 6.2 冻结增补（Hard Gate B addendum）

- 基础冻结文件：`/home/honglab/leijiaxin/HFOsp/results/epi_prssm/v0_1/manifests/INTERICTAL_MODEL_FREEZE.json`，未被覆盖 = `True`
- 只改变了：only the observation stream and the observer cut-off; the model family, the checkpoints, the patient baselines, the graphs and the state dimension are exactly those in the base freeze
- 主提前量 30.0 min；辅助 [60.0, 15.0, 5.0] min
- onset 时间用途：alignment and scoring only; the onset time never enters the model as an input, a target or a selection signal
- 匹配集合：['same patient', 'same recording session', 'same day/night bin', 'outside every peri-ictal exclusion window', 'matched observation coverage decile', 'matched multi-scale rate (30 min, 2 h, 4 h, 8 h)', 'matched median inter-event interval', 'matched last-event gap decile']
- 主张规则：a state claim requires the state endpoint to survive residualisation on the multi-scale rate and interval nuisances; Topic 2 already establishes that the event rate itself drifts slowly and rises around seizures, so an unresidualised state effect is not evidence for a spatial-repertoire state

**冻结前流水线自检的完整披露：**

- 发生了：`True`；before this freeze was written, the seizure-link script was executed once on a 2-patient smoke cohort (6 eligible seizures, 1 patient) using a throw-away G0 checkpoint trained on that same smoke cohort
- 目的：to test the code path end to end rather than discover a crash after the freeze
- 它改变了：fixed two variable-shadowing bugs that made the script abort
- 它改变了：added a numerical-validity guard: when the matched null has no spread left, the z is withheld as degenerate instead of being computed from rounding error
- 它改变了：added a secondary extended last-event-gap window, declared on the observed gap distribution, which is a property of the data and not an outcome
- 它没有改变：the model family and the frozen representatives
- 它没有改变：the primary endpoints
- 它没有改变：the primary last-event-gap window, which stays at the pre-registered value
- 它没有改变：the pseudo-onset matching protocol

### 6.3 主分析 `leaky_state`，提前量 15 min

- 可分析患者 27/34；合格发作 363；不可观测患者 7 （['yuquan_chengshuai', 'yuquan_hanyuxuan', 'yuquan_liyouran', 'yuquan_songzishuo', 'yuquan_wangyiyang', 'yuquan_zhangjiaqi', 'yuquan_zhaochenxi']）
- 可入观测器的事件 1,111,565，其中 260,891 是删减流里没有的

**filtered_at_onset** — the observer consumed every admissible event up to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | -0.1050 | [-0.5201, +0.2617] | 10/27 | 0.248 | +10.2452 | [+9.4363, +10.6654] | 27/27 | 0 | True |
| resource | — | — | — | — | — | — | — | 182 | — |
| expected_load | -0.2148 | [-0.4345, +0.2588] | 10/27 | 0.248 | +4.7124 | [+2.9940, +5.4402] | 25/27 | 0 | True |
| first_selection_entropy | +0.1875 | [-0.0183, +0.7418] | 18/27 | 0.122 | -1.1093 | [-1.4874, -0.5993] | 4/27 | 0 | True |

**filtered_at_cutoff** — the observer stopped at onset minus the lead

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | +0.1600 | [-0.2324, +0.4950] | 18/27 | 0.122 | +8.8413 | [+8.4038, +9.2659] | 27/27 | 0 | True |
| resource | — | — | — | — | — | — | — | 182 | — |
| expected_load | +0.2172 | [-0.2194, +0.4462] | 15/27 | 0.701 | -3.3644 | [-4.5480, -1.4403] | 4/27 | 0 | True |
| first_selection_entropy | +0.0360 | [-0.2187, +0.6070] | 14/27 | 1 | +3.4710 | [+2.5543, +4.1325] | 27/27 | 0 | False |

**open_loop_at_onset** — the observer stopped at the cut-off and the generator then integrated autonomously to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | +0.0710 | [-0.2101, +0.4225] | 15/27 | 0.701 | -0.8064 | [-1.2856, -0.0351] | 8/27 | 0 | True |
| resource | — | — | — | — | — | — | — | 182 | — |
| expected_load | +0.0806 | [-0.2388, +0.6629] | 15/27 | 0.701 | -0.1515 | [-1.2621, +0.1109] | 9/27 | 0 | True |
| first_selection_entropy | +0.0439 | [-0.0844, +0.4228] | 15/27 | 0.701 | +2.8321 | [+2.4381, +3.5239] | 27/27 | 0 | True |

**干扰量自身的可分辨性（状态主张必须胜过这一行）：**

| 干扰量 | 中位 z | 95% CI | 方向有利 | p |
| --- | --- | --- | --- | --- |
| rate_1800s | +0.0768 | [-0.7962, +0.4003] | 14/27 | 1 |
| rate_7200s | -0.3816 | [-0.6337, -0.0835] | 8/27 | 0.0522 |
| rate_14400s | +0.0198 | [-0.3340, +0.4180] | 15/27 | 0.701 |
| rate_28800s | +0.0457 | [-0.5332, +0.3741] | 15/27 | 0.701 |
| median_iei | -0.8163 | [-1.5052, -0.4724] | 7/27 | 0.0192 |
| coverage | +0.1579 | [-0.4626, +0.3134] | 11/20 | 0.824 |

- Holm 校正（open-loop 家族）：`{'open_loop_at_onset::state_norm': 1.0, 'open_loop_at_onset::expected_load': 1.0, 'open_loop_at_onset::first_selection_entropy': 1.0}`

### 6.3 主分析 `leaky_state`，提前量 30 min

- 可分析患者 27/34；合格发作 361；不可观测患者 7 （['yuquan_chengshuai', 'yuquan_hanyuxuan', 'yuquan_liyouran', 'yuquan_songzishuo', 'yuquan_wangyiyang', 'yuquan_zhangjiaqi', 'yuquan_zhaochenxi']）
- 可入观测器的事件 1,111,565，其中 260,891 是删减流里没有的

**filtered_at_onset** — the observer consumed every admissible event up to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | -0.0830 | [-0.4048, +0.1641] | 12/27 | 0.701 | +4.0700 | [+3.2422, +4.4757] | 27/27 | 0 | True |
| resource | — | — | — | — | — | — | — | 203 | — |
| expected_load | -0.2502 | [-0.5295, -0.0584] | 7/27 | 0.0192 | +4.4173 | [+3.7313, +5.5050] | 26/27 | 0 | True |
| first_selection_entropy | +0.2989 | [+0.0189, +0.7410] | 19/27 | 0.0522 | -1.2244 | [-1.8971, -1.0507] | 3/27 | 0 | True |

**filtered_at_cutoff** — the observer stopped at onset minus the lead

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | +0.3975 | [-0.2421, +0.5691] | 16/27 | 0.442 | -1.4332 | [-2.0004, -0.5842] | 6/27 | 0 | True |
| resource | — | — | — | — | — | — | — | 203 | — |
| expected_load | -0.0144 | [-0.5789, +0.2490] | 12/27 | 0.701 | -13.1881 | [-16.5815, -11.7741] | 0/27 | 0 | True |
| first_selection_entropy | -0.0695 | [-0.6652, +0.2775] | 12/27 | 0.701 | +8.2246 | [+7.2628, +9.3344] | 27/27 | 0 | True |

**open_loop_at_onset** — the observer stopped at the cut-off and the generator then integrated autonomously to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | -0.0419 | [-0.4943, +0.1516] | 12/27 | 0.701 | -0.1247 | [-0.9839, +0.6051] | 13/27 | 0 | True |
| resource | — | — | — | — | — | — | — | 203 | — |
| expected_load | +0.0055 | [-0.4339, +0.4989] | 14/27 | 1 | -5.3816 | [-6.3408, -4.6100] | 0/27 | 0 | False |
| first_selection_entropy | -0.2322 | [-0.4658, +0.1293] | 12/27 | 0.701 | +6.4719 | [+5.7391, +7.2189] | 27/27 | 0 | True |

**干扰量自身的可分辨性（状态主张必须胜过这一行）：**

| 干扰量 | 中位 z | 95% CI | 方向有利 | p |
| --- | --- | --- | --- | --- |
| rate_1800s | -0.0552 | [-0.6102, +0.4854] | 13/27 | 1 |
| rate_7200s | -0.2483 | [-0.6453, +0.0203] | 9/27 | 0.122 |
| rate_14400s | -0.0541 | [-0.4991, +0.4465] | 11/27 | 0.442 |
| rate_28800s | -0.1158 | [-0.5505, +0.2843] | 13/27 | 1 |
| median_iei | -0.5287 | [-1.1807, +0.0941] | 9/27 | 0.122 |
| coverage | +0.0491 | [-0.6610, +0.2265] | 12/21 | 0.664 |

- Holm 校正（open-loop 家族）：`{'open_loop_at_onset::state_norm': 1.0, 'open_loop_at_onset::first_selection_entropy': 1.0, 'open_loop_at_onset::expected_load': 1.0}`

### 6.3 主分析 `leaky_state`，提前量 5 min

- 可分析患者 27/34；合格发作 363；不可观测患者 7 （['yuquan_chengshuai', 'yuquan_hanyuxuan', 'yuquan_liyouran', 'yuquan_songzishuo', 'yuquan_wangyiyang', 'yuquan_zhangjiaqi', 'yuquan_zhaochenxi']）
- 可入观测器的事件 1,111,565，其中 260,891 是删减流里没有的

**filtered_at_onset** — the observer consumed every admissible event up to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | +0.0418 | [-0.4704, +0.3752] | 13/25 | 1 | +0.3641 | [-0.6913, +1.0466] | 15/25 | 0 | False |
| resource | — | — | — | — | — | — | — | 154 | — |
| expected_load | -0.2280 | [-0.3395, +0.4870] | 10/25 | 0.424 | -24.1816 | [-25.6843, -23.3571] | 1/25 | 0 | True |
| first_selection_entropy | +0.2539 | [-0.1985, +0.5791] | 15/25 | 0.424 | +1.1315 | [+0.5692, +1.7139] | 20/25 | 0 | True |

**filtered_at_cutoff** — the observer stopped at onset minus the lead

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | +0.3929 | [-0.3639, +0.6139] | 16/25 | 0.23 | -0.8769 | [-1.3347, +0.5313] | 9/25 | 0 | True |
| resource | — | — | — | — | — | — | — | 154 | — |
| expected_load | +0.3114 | [-0.2535, +0.7084] | 16/25 | 0.23 | -10.1219 | [-13.9816, -7.8183] | 1/25 | 0 | True |
| first_selection_entropy | +0.0356 | [-0.5165, +0.3785] | 13/25 | 1 | +8.7203 | [+7.5497, +9.4749] | 24/25 | 0 | False |

**open_loop_at_onset** — the observer stopped at the cut-off and the generator then integrated autonomously to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | +0.3673 | [-0.4058, +0.6914] | 16/25 | 0.23 | -0.2552 | [-1.3006, +0.2934] | 9/25 | 0 | True |
| resource | — | — | — | — | — | — | — | 154 | — |
| expected_load | +0.5356 | [+0.1493, +0.7948] | 19/25 | 0.0146 | -13.0433 | [-14.1967, -11.1768] | 1/25 | 0 | True |
| first_selection_entropy | -0.0516 | [-0.3703, +0.4136] | 12/25 | 1 | +7.0763 | [+6.5185, +7.4748] | 24/25 | 0 | False |

**干扰量自身的可分辨性（状态主张必须胜过这一行）：**

| 干扰量 | 中位 z | 95% CI | 方向有利 | p |
| --- | --- | --- | --- | --- |
| rate_1800s | +0.0933 | [-0.5438, +0.8230] | 14/25 | 0.69 |
| rate_7200s | -0.4636 | [-1.1589, +0.1922] | 9/25 | 0.23 |
| rate_14400s | +0.0520 | [-0.5962, +0.3520] | 15/25 | 0.424 |
| rate_28800s | +0.0898 | [-0.5603, +1.4502] | 14/25 | 0.69 |
| median_iei | -0.7465 | [-1.2988, -0.1850] | 7/25 | 0.0433 |
| coverage | -0.4173 | [-1.4877, +0.1711] | 5/17 | 0.143 |

- Holm 校正（open-loop 家族）：`{'open_loop_at_onset::expected_load': 0.04389989376068115, 'open_loop_at_onset::state_norm': 0.4590458869934082, 'open_loop_at_onset::first_selection_entropy': 1.0}`

### 6.3 主分析 `leaky_state`，提前量 60 min

- 可分析患者 27/34；合格发作 360；不可观测患者 7 （['yuquan_chengshuai', 'yuquan_hanyuxuan', 'yuquan_liyouran', 'yuquan_songzishuo', 'yuquan_wangyiyang', 'yuquan_zhangjiaqi', 'yuquan_zhaochenxi']）
- 可入观测器的事件 1,111,565，其中 260,891 是删减流里没有的

**filtered_at_onset** — the observer consumed every admissible event up to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | -0.1935 | [-1.0497, +0.0392] | 9/27 | 0.122 | +4.7526 | [+4.4452, +5.3158] | 27/27 | 0 | True |
| resource | — | — | — | — | — | — | — | 220 | — |
| expected_load | -0.4296 | [-0.8299, -0.1286] | 5/27 | 0.00151 | -1.2841 | [-2.2488, +0.2872] | 12/27 | 0 | True |
| first_selection_entropy | +0.4626 | [-0.0917, +0.8391] | 17/27 | 0.248 | -1.8834 | [-2.4954, -1.2264] | 3/27 | 0 | True |

**filtered_at_cutoff** — the observer stopped at onset minus the lead

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | -0.3258 | [-0.5939, +0.4059] | 11/27 | 0.442 | -1.2615 | [-2.4985, -0.6951] | 7/27 | 0 | True |
| resource | — | — | — | — | — | — | — | 220 | — |
| expected_load | -0.0846 | [-0.5657, +0.4790] | 13/27 | 1 | -10.4583 | [-13.5341, -7.7155] | 0/27 | 0 | False |
| first_selection_entropy | -0.2888 | [-0.4181, +0.1588] | 12/27 | 0.701 | +4.8291 | [+3.0412, +5.3399] | 26/27 | 0 | True |

**open_loop_at_onset** — the observer stopped at the cut-off and the generator then integrated autonomously to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | -0.0227 | [-0.4582, +0.4616] | 13/27 | 1 | -2.2192 | [-2.3661, -1.6750] | 5/27 | 0 | False |
| resource | — | — | — | — | — | — | — | 220 | — |
| expected_load | +0.0464 | [-0.4721, +0.5402] | 14/27 | 1 | -0.5068 | [-0.7428, +0.1491] | 10/27 | 0 | False |
| first_selection_entropy | -0.3420 | [-0.6559, +0.4188] | 12/27 | 0.701 | +2.3510 | [+1.8689, +2.8889] | 26/27 | 0 | True |

**干扰量自身的可分辨性（状态主张必须胜过这一行）：**

| 干扰量 | 中位 z | 95% CI | 方向有利 | p |
| --- | --- | --- | --- | --- |
| rate_1800s | -0.2151 | [-0.3486, +0.5033] | 12/27 | 0.701 |
| rate_7200s | -0.4006 | [-0.6904, +0.1008] | 11/27 | 0.442 |
| rate_14400s | -0.2826 | [-0.6019, +0.1766] | 10/27 | 0.248 |
| rate_28800s | -0.3767 | [-1.1100, +0.5010] | 12/27 | 0.701 |
| median_iei | -0.4195 | [-0.7473, -0.1117] | 5/27 | 0.00151 |
| coverage | -0.5621 | [-1.1882, +0.3209] | 10/21 | 1 |

- Holm 校正（open-loop 家族）：`{'open_loop_at_onset::first_selection_entropy': 1.0, 'open_loop_at_onset::state_norm': 1.0, 'open_loop_at_onset::expected_load': 1.0}`

### 6.3 主分析 `linear_graph_recurrent`，提前量 15 min

- 可分析患者 27/34；合格发作 363；不可观测患者 7 （['yuquan_chengshuai', 'yuquan_hanyuxuan', 'yuquan_liyouran', 'yuquan_songzishuo', 'yuquan_wangyiyang', 'yuquan_zhangjiaqi', 'yuquan_zhaochenxi']）
- 可入观测器的事件 1,111,565，其中 260,891 是删减流里没有的

**filtered_at_onset** — the observer consumed every admissible event up to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | -0.1566 | [-0.4676, +0.2304] | 12/27 | 0.701 | -5.3737 | [-5.8493, -4.9903] | 1/27 | 1 | True |
| resource | — | — | — | — | — | — | — | 182 | — |
| expected_load | +0.1247 | [-0.4204, +0.2776] | 15/27 | 0.701 | +2.0446 | [+1.7650, +2.8515] | 25/27 | 1 | True |
| first_selection_entropy | +0.0747 | [-0.3699, +0.5474] | 16/27 | 0.442 | -0.3929 | [-0.7498, +0.2330] | 11/27 | 1 | True |

**filtered_at_cutoff** — the observer stopped at onset minus the lead

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | -0.2053 | [-0.5289, +0.3894] | 12/27 | 0.701 | -4.9156 | [-5.2455, -4.6909] | 0/27 | 3 | True |
| resource | — | — | — | — | — | — | — | 182 | — |
| expected_load | +0.2649 | [-0.2960, +0.5869] | 16/27 | 0.442 | +2.6804 | [+2.3908, +3.1514] | 27/27 | 3 | True |
| first_selection_entropy | -0.0167 | [-0.2488, +0.2417] | 13/27 | 1 | -3.5273 | [-4.8348, -2.0230] | 4/27 | 2 | False |

**open_loop_at_onset** — the observer stopped at the cut-off and the generator then integrated autonomously to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | +0.1019 | [-0.1859, +0.4106] | 16/27 | 0.442 | -2.1750 | [-2.5750, -1.8364] | 0/27 | 3 | True |
| resource | — | — | — | — | — | — | — | 182 | — |
| expected_load | -0.0553 | [-0.3485, +0.3900] | 13/27 | 1 | +2.2209 | [+2.0011, +2.4563] | 27/27 | 3 | False |
| first_selection_entropy | -0.0456 | [-0.3088, +0.3085] | 13/27 | 1 | -27.4284 | [-33.0994, -21.8616] | 1/27 | 2 | False |

**干扰量自身的可分辨性（状态主张必须胜过这一行）：**

| 干扰量 | 中位 z | 95% CI | 方向有利 | p |
| --- | --- | --- | --- | --- |
| rate_1800s | +0.0768 | [-0.7962, +0.4003] | 14/27 | 1 |
| rate_7200s | -0.3816 | [-0.6337, -0.0835] | 8/27 | 0.0522 |
| rate_14400s | +0.0198 | [-0.3340, +0.4180] | 15/27 | 0.701 |
| rate_28800s | +0.0457 | [-0.5332, +0.3741] | 15/27 | 0.701 |
| median_iei | -0.8163 | [-1.5052, -0.4724] | 7/27 | 0.0192 |
| coverage | +0.1579 | [-0.4626, +0.3134] | 11/20 | 0.824 |

- Holm 校正（open-loop 家族）：`{'open_loop_at_onset::state_norm': 1.0, 'open_loop_at_onset::expected_load': 1.0, 'open_loop_at_onset::first_selection_entropy': 1.0}`

### 6.3 主分析 `linear_graph_recurrent`，提前量 30 min

- 可分析患者 27/34；合格发作 361；不可观测患者 7 （['yuquan_chengshuai', 'yuquan_hanyuxuan', 'yuquan_liyouran', 'yuquan_songzishuo', 'yuquan_wangyiyang', 'yuquan_zhangjiaqi', 'yuquan_zhaochenxi']）
- 可入观测器的事件 1,111,565，其中 260,891 是删减流里没有的

**filtered_at_onset** — the observer consumed every admissible event up to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | -0.0753 | [-0.4661, +0.3112] | 13/27 | 1 | -9.1368 | [-9.4958, -8.3365] | 0/27 | 1 | False |
| resource | — | — | — | — | — | — | — | 203 | — |
| expected_load | -0.1306 | [-0.4407, +0.2322] | 10/27 | 0.248 | +2.2889 | [+1.9396, +2.5741] | 25/27 | 1 | True |
| first_selection_entropy | +0.3186 | [+0.0151, +0.8364] | 19/27 | 0.0522 | -0.5834 | [-1.1622, -0.1825] | 6/27 | 1 | True |

**filtered_at_cutoff** — the observer stopped at onset minus the lead

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | -0.0764 | [-0.3083, +0.3239] | 13/27 | 1 | -6.4774 | [-6.7964, -6.2626] | 0/27 | 4 | False |
| resource | — | — | — | — | — | — | — | 203 | — |
| expected_load | +0.0302 | [-0.4899, +0.4387] | 14/27 | 1 | +3.3260 | [+3.0106, +3.6346] | 27/27 | 4 | False |
| first_selection_entropy | +0.0015 | [-0.2315, +0.5904] | 14/27 | 1 | -78.5199 | [-85.8947, -73.3150] | 0/27 | 2 | False |

**open_loop_at_onset** — the observer stopped at the cut-off and the generator then integrated autonomously to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | +0.2198 | [-0.2274, +0.6069] | 15/27 | 0.701 | -0.8596 | [-1.1273, -0.7146] | 0/27 | 4 | True |
| resource | — | — | — | — | — | — | — | 203 | — |
| expected_load | -0.3578 | [-0.5678, +0.4820] | 11/27 | 0.442 | +0.9045 | [+0.7468, +1.1185] | 26/27 | 4 | True |
| first_selection_entropy | +0.2667 | [-0.1434, +0.6537] | 17/27 | 0.248 | -149.3067 | [-158.5276, -123.3265] | 0/27 | 2 | True |

**干扰量自身的可分辨性（状态主张必须胜过这一行）：**

| 干扰量 | 中位 z | 95% CI | 方向有利 | p |
| --- | --- | --- | --- | --- |
| rate_1800s | -0.0552 | [-0.6102, +0.4854] | 13/27 | 1 |
| rate_7200s | -0.2483 | [-0.6453, +0.0203] | 9/27 | 0.122 |
| rate_14400s | -0.0541 | [-0.4991, +0.4465] | 11/27 | 0.442 |
| rate_28800s | -0.1158 | [-0.5505, +0.2843] | 13/27 | 1 |
| median_iei | -0.5287 | [-1.1807, +0.0941] | 9/27 | 0.122 |
| coverage | +0.0491 | [-0.6610, +0.2265] | 12/21 | 0.664 |

- Holm 校正（open-loop 家族）：`{'open_loop_at_onset::first_selection_entropy': 0.743365690112114, 'open_loop_at_onset::expected_load': 0.8841366767883301, 'open_loop_at_onset::state_norm': 0.8841366767883301}`

### 6.3 主分析 `linear_graph_recurrent`，提前量 5 min

- 可分析患者 27/34；合格发作 363；不可观测患者 7 （['yuquan_chengshuai', 'yuquan_hanyuxuan', 'yuquan_liyouran', 'yuquan_songzishuo', 'yuquan_wangyiyang', 'yuquan_zhangjiaqi', 'yuquan_zhaochenxi']）
- 可入观测器的事件 1,111,565，其中 260,891 是删减流里没有的

**filtered_at_onset** — the observer consumed every admissible event up to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | -0.4349 | [-0.7501, +0.2460] | 11/25 | 0.69 | -1.9849 | [-2.4133, -1.7024] | 1/25 | 1 | True |
| resource | — | — | — | — | — | — | — | 154 | — |
| expected_load | +0.1507 | [-0.2474, +0.4899] | 13/25 | 1 | +1.5526 | [+1.4092, +2.0025] | 25/25 | 1 | False |
| first_selection_entropy | +0.0236 | [-0.3578, +0.6183] | 13/25 | 1 | -2.5049 | [-2.9579, -2.1210] | 0/25 | 1 | False |

**filtered_at_cutoff** — the observer stopped at onset minus the lead

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | -0.5669 | [-0.8346, -0.1516] | 7/25 | 0.0433 | -3.1192 | [-3.3071, -2.8312] | 1/25 | 3 | True |
| resource | — | — | — | — | — | — | — | 154 | — |
| expected_load | +0.3187 | [-0.1603, +0.8678] | 15/25 | 0.424 | +2.4532 | [+2.0136, +2.6969] | 25/25 | 3 | True |
| first_selection_entropy | +0.1366 | [-0.4139, +0.4451] | 13/25 | 1 | -42.1380 | [-48.3365, -39.8561] | 0/25 | 2 | False |

**open_loop_at_onset** — the observer stopped at the cut-off and the generator then integrated autonomously to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | -0.3770 | [-0.5171, -0.0565] | 5/25 | 0.00408 | -2.2420 | [-2.5081, -2.0888] | 1/25 | 3 | True |
| resource | — | — | — | — | — | — | — | 154 | — |
| expected_load | +0.2782 | [+0.0615, +0.7281] | 18/25 | 0.0433 | +1.3824 | [+1.1750, +1.8958] | 24/25 | 3 | True |
| first_selection_entropy | +0.1598 | [-0.4034, +0.4300] | 14/25 | 0.69 | -73.4138 | [-83.3591, -70.4459] | 0/25 | 2 | True |

**干扰量自身的可分辨性（状态主张必须胜过这一行）：**

| 干扰量 | 中位 z | 95% CI | 方向有利 | p |
| --- | --- | --- | --- | --- |
| rate_1800s | +0.0933 | [-0.5438, +0.8230] | 14/25 | 0.69 |
| rate_7200s | -0.4636 | [-1.1589, +0.1922] | 9/25 | 0.23 |
| rate_14400s | +0.0520 | [-0.5962, +0.3520] | 15/25 | 0.424 |
| rate_28800s | +0.0898 | [-0.5603, +1.4502] | 14/25 | 0.69 |
| median_iei | -0.7465 | [-1.2988, -0.1850] | 7/25 | 0.0433 |
| coverage | -0.4173 | [-1.4877, +0.1711] | 5/17 | 0.143 |

- Holm 校正（open-loop 家族）：`{'open_loop_at_onset::state_norm': 0.012231945991516113, 'open_loop_at_onset::expected_load': 0.08657050132751465, 'open_loop_at_onset::first_selection_entropy': 0.6900379657745361}`

### 6.3 主分析 `linear_graph_recurrent`，提前量 60 min

- 可分析患者 27/34；合格发作 360；不可观测患者 7 （['yuquan_chengshuai', 'yuquan_hanyuxuan', 'yuquan_liyouran', 'yuquan_songzishuo', 'yuquan_wangyiyang', 'yuquan_zhangjiaqi', 'yuquan_zhaochenxi']）
- 可入观测器的事件 1,111,565，其中 260,891 是删减流里没有的

**filtered_at_onset** — the observer consumed every admissible event up to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | +0.4114 | [-0.3015, +0.9718] | 17/27 | 0.248 | -6.8564 | [-7.9289, -6.2111] | 1/27 | 1 | True |
| resource | — | — | — | — | — | — | — | 220 | — |
| expected_load | -0.3717 | [-0.8800, +0.0343] | 9/27 | 0.122 | +2.6059 | [+2.3919, +2.8125] | 24/27 | 1 | True |
| first_selection_entropy | +0.3968 | [-0.1029, +0.8152] | 17/27 | 0.248 | -1.8899 | [-2.2697, -1.3924] | 2/27 | 1 | True |

**filtered_at_cutoff** — the observer stopped at onset minus the lead

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | -0.0991 | [-0.3586, +0.1968] | 13/27 | 1 | -4.5090 | [-5.0206, -4.1301] | 1/27 | 4 | False |
| resource | — | — | — | — | — | — | — | 220 | — |
| expected_load | -0.1773 | [-0.8340, +0.1120] | 9/27 | 0.122 | +2.7886 | [+2.5382, +3.5058] | 27/27 | 4 | True |
| first_selection_entropy | +0.0692 | [-0.3333, +0.2373] | 14/27 | 1 | -26.8761 | [-31.8286, -19.6800] | 1/27 | 2 | False |

**open_loop_at_onset** — the observer stopped at the cut-off and the generator then integrated autonomously to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | +0.2544 | [-0.0342, +0.4963] | 17/27 | 0.248 | +1.7037 | [+1.3493, +2.0870] | 26/27 | 4 | True |
| resource | — | — | — | — | — | — | — | 220 | — |
| expected_load | -0.1526 | [-0.4733, +0.0037] | 9/27 | 0.122 | -0.1363 | [-0.3649, +0.1950] | 11/27 | 4 | True |
| first_selection_entropy | +0.0996 | [-0.3052, +0.4288] | 15/27 | 0.701 | +0.5795 | [+0.0311, +0.7660] | 19/27 | 4 | True |

**干扰量自身的可分辨性（状态主张必须胜过这一行）：**

| 干扰量 | 中位 z | 95% CI | 方向有利 | p |
| --- | --- | --- | --- | --- |
| rate_1800s | -0.2151 | [-0.3486, +0.5033] | 12/27 | 0.701 |
| rate_7200s | -0.4006 | [-0.6904, +0.1008] | 11/27 | 0.442 |
| rate_14400s | -0.2826 | [-0.6019, +0.1766] | 10/27 | 0.248 |
| rate_28800s | -0.3767 | [-1.1100, +0.5010] | 12/27 | 0.701 |
| median_iei | -0.4195 | [-0.7473, -0.1117] | 5/27 | 0.00151 |
| coverage | -0.5621 | [-1.1882, +0.3209] | 10/21 | 1 |

- Holm 校正（open-loop 家族）：`{'open_loop_at_onset::expected_load': 0.36623436212539673, 'open_loop_at_onset::state_norm': 0.4955771267414093, 'open_loop_at_onset::first_selection_entropy': 0.7011080384254456}`

### 6.3 主分析 `resource_anchored_on_best_family`，提前量 15 min

- 可分析患者 27/34；合格发作 363；不可观测患者 7 （['yuquan_chengshuai', 'yuquan_hanyuxuan', 'yuquan_liyouran', 'yuquan_songzishuo', 'yuquan_wangyiyang', 'yuquan_zhangjiaqi', 'yuquan_zhaochenxi']）
- 可入观测器的事件 1,111,565，其中 260,891 是删减流里没有的

**filtered_at_onset** — the observer consumed every admissible event up to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | -0.0024 | [-0.4893, +0.4841] | 13/27 | 1 | -3.8593 | [-4.2449, -3.4072] | 1/27 | 1 | False |
| resource | -0.3807 | [-0.7361, -0.2705] | 0/10 | 0.00195 | -0.6468 | [-0.8767, -0.6107] | 0/10 | 156 | True |
| expected_load | +0.0101 | [-0.6674, +0.3130] | 14/27 | 1 | +3.7480 | [+3.1783, +4.2958] | 27/27 | 1 | False |
| first_selection_entropy | +0.2518 | [-0.2549, +0.7629] | 15/27 | 0.701 | -1.3983 | [-1.8382, -1.1328] | 2/27 | 1 | True |

**filtered_at_cutoff** — the observer stopped at onset minus the lead

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | +0.0028 | [-0.2125, +0.4813] | 14/27 | 1 | +1.0071 | [+0.6281, +1.4079] | 24/27 | 3 | False |
| resource | -0.3409 | [-0.7352, -0.2448] | 1/11 | 0.0117 | -0.2357 | [-0.3783, -0.1486] | 2/11 | 149 | True |
| expected_load | +0.0576 | [-0.4871, +0.3525] | 15/27 | 0.701 | +2.2493 | [+2.0835, +2.8618] | 25/27 | 3 | True |
| first_selection_entropy | +0.1630 | [-0.1937, +0.4968] | 15/27 | 0.701 | +0.0686 | [-0.2989, +0.3449] | 15/27 | 3 | True |

**open_loop_at_onset** — the observer stopped at the cut-off and the generator then integrated autonomously to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | +0.1437 | [-0.1236, +0.5944] | 16/27 | 0.442 | -2.1097 | [-2.3567, -1.5915] | 1/27 | 3 | True |
| resource | -0.3808 | [-0.7393, -0.2704] | 0/10 | 0.00195 | -0.6879 | [-0.9214, -0.6536] | 0/10 | 156 | True |
| expected_load | -0.2331 | [-0.5778, -0.0179] | 8/27 | 0.0522 | +0.2008 | [-0.4628, +0.5878] | 17/27 | 3 | True |
| first_selection_entropy | +0.3326 | [-0.1629, +0.6746] | 16/27 | 0.442 | -1.7850 | [-2.0706, -1.2363] | 1/27 | 3 | True |

**干扰量自身的可分辨性（状态主张必须胜过这一行）：**

| 干扰量 | 中位 z | 95% CI | 方向有利 | p |
| --- | --- | --- | --- | --- |
| rate_1800s | +0.0768 | [-0.7962, +0.4003] | 14/27 | 1 |
| rate_7200s | -0.3816 | [-0.6337, -0.0835] | 8/27 | 0.0522 |
| rate_14400s | +0.0198 | [-0.3340, +0.4180] | 15/27 | 0.701 |
| rate_28800s | +0.0457 | [-0.5332, +0.3741] | 15/27 | 0.701 |
| median_iei | -0.8163 | [-1.5052, -0.4724] | 7/27 | 0.0192 |
| coverage | +0.1579 | [-0.4626, +0.3134] | 11/20 | 0.824 |

- Holm 校正（open-loop 家族）：`{'open_loop_at_onset::resource': 0.0078125, 'open_loop_at_onset::expected_load': 0.1567169576883316, 'open_loop_at_onset::state_norm': 0.8841366767883301, 'open_loop_at_onset::first_selection_entropy': 0.8841366767883301}`

### 6.3 主分析 `resource_anchored_on_best_family`，提前量 30 min

- 可分析患者 27/34；合格发作 361；不可观测患者 7 （['yuquan_chengshuai', 'yuquan_hanyuxuan', 'yuquan_liyouran', 'yuquan_songzishuo', 'yuquan_wangyiyang', 'yuquan_zhangjiaqi', 'yuquan_zhaochenxi']）
- 可入观测器的事件 1,111,565，其中 260,891 是删减流里没有的

**filtered_at_onset** — the observer consumed every admissible event up to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | +0.0255 | [-0.4237, +0.6967] | 14/27 | 1 | -4.0568 | [-4.5720, -3.6667] | 1/27 | 1 | False |
| resource | -0.4090 | [-0.8083, -0.2516] | 0/10 | 0.00195 | -0.1972 | [-0.4114, -0.0308] | 2/10 | 180 | True |
| expected_load | -0.1082 | [-0.6592, +0.4539] | 12/27 | 0.701 | +3.6297 | [+3.2241, +3.9356] | 27/27 | 1 | True |
| first_selection_entropy | +0.4425 | [-0.0896, +0.8308] | 18/27 | 0.122 | -1.1106 | [-1.5924, -0.7900] | 3/27 | 1 | True |

**filtered_at_cutoff** — the observer stopped at onset minus the lead

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | -0.2501 | [-0.5081, +0.0328] | 9/27 | 0.122 | -1.6141 | [-1.7074, -1.1372] | 2/27 | 4 | True |
| resource | -0.3964 | [-0.7345, -0.1152] | 1/11 | 0.0117 | +0.1745 | [+0.0588, +0.3459] | 9/11 | 172 | True |
| expected_load | -0.1872 | [-0.7774, +0.5022] | 12/27 | 0.701 | +2.9732 | [+2.7496, +3.5680] | 26/27 | 4 | True |
| first_selection_entropy | +0.2060 | [-0.2336, +0.5191] | 16/27 | 0.442 | -1.4060 | [-1.7731, -0.9461] | 2/27 | 4 | True |

**open_loop_at_onset** — the observer stopped at the cut-off and the generator then integrated autonomously to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | -0.1970 | [-0.7618, +0.5527] | 13/27 | 1 | -1.7306 | [-1.8925, -1.2132] | 1/27 | 4 | True |
| resource | -0.4093 | [-0.8083, -0.2515] | 0/10 | 0.00195 | -0.1976 | [-0.4063, -0.0286] | 2/10 | 180 | True |
| expected_load | +0.0098 | [-0.5565, +0.7524] | 14/27 | 1 | +1.2878 | [+0.9128, +1.6019] | 26/27 | 4 | False |
| first_selection_entropy | +0.1398 | [-0.3914, +0.3929] | 14/27 | 1 | -1.2114 | [-1.4743, -0.9452] | 4/27 | 4 | False |

**干扰量自身的可分辨性（状态主张必须胜过这一行）：**

| 干扰量 | 中位 z | 95% CI | 方向有利 | p |
| --- | --- | --- | --- | --- |
| rate_1800s | -0.0552 | [-0.6102, +0.4854] | 13/27 | 1 |
| rate_7200s | -0.2483 | [-0.6453, +0.0203] | 9/27 | 0.122 |
| rate_14400s | -0.0541 | [-0.4991, +0.4465] | 11/27 | 0.442 |
| rate_28800s | -0.1158 | [-0.5505, +0.2843] | 13/27 | 1 |
| median_iei | -0.5287 | [-1.1807, +0.0941] | 9/27 | 0.122 |
| coverage | +0.0491 | [-0.6610, +0.2265] | 12/21 | 0.664 |

- Holm 校正（open-loop 家族）：`{'open_loop_at_onset::resource': 0.0078125, 'open_loop_at_onset::state_norm': 1.0, 'open_loop_at_onset::expected_load': 1.0, 'open_loop_at_onset::first_selection_entropy': 1.0}`

### 6.3 主分析 `resource_anchored_on_best_family`，提前量 5 min

- 可分析患者 27/34；合格发作 363；不可观测患者 7 （['yuquan_chengshuai', 'yuquan_hanyuxuan', 'yuquan_liyouran', 'yuquan_songzishuo', 'yuquan_wangyiyang', 'yuquan_zhangjiaqi', 'yuquan_zhaochenxi']）
- 可入观测器的事件 1,111,565，其中 260,891 是删减流里没有的

**filtered_at_onset** — the observer consumed every admissible event up to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | -0.4122 | [-0.8041, +0.0470] | 8/25 | 0.108 | -2.6376 | [-2.8344, -1.7116] | 0/25 | 1 | True |
| resource | -0.3018 | [-0.7360, -0.1989] | 0/11 | 0.000977 | +0.2211 | [+0.0045, +0.3217] | 9/11 | 134 | True |
| expected_load | +0.2521 | [-0.3637, +0.7851] | 15/25 | 0.424 | +2.4668 | [+2.1361, +3.0453] | 24/25 | 1 | True |
| first_selection_entropy | +0.1002 | [-0.3226, +0.8274] | 13/25 | 1 | -2.0995 | [-2.3481, -1.6609] | 2/25 | 1 | False |

**filtered_at_cutoff** — the observer stopped at onset minus the lead

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | -0.3992 | [-0.7788, -0.0883] | 7/25 | 0.0433 | -1.0728 | [-1.5855, -1.0071] | 3/25 | 3 | True |
| resource | -0.3055 | [-0.5385, -0.1668] | 1/12 | 0.00635 | +0.1032 | [-0.1215, +0.2138] | 8/12 | 133 | True |
| expected_load | +0.2017 | [-0.3336, +0.8668] | 14/25 | 0.69 | +3.1758 | [+2.3198, +3.6075] | 24/25 | 3 | True |
| first_selection_entropy | +0.1744 | [-0.0722, +0.4778] | 17/25 | 0.108 | -3.5133 | [-3.8545, -3.2802] | 1/25 | 3 | True |

**open_loop_at_onset** — the observer stopped at the cut-off and the generator then integrated autonomously to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | -0.4302 | [-0.4830, -0.0949] | 6/25 | 0.0146 | -3.0514 | [-3.4467, -2.6186] | 3/25 | 3 | True |
| resource | -0.2961 | [-0.7387, -0.1988] | 0/11 | 0.000977 | +0.2409 | [+0.0249, +0.3423] | 9/11 | 134 | True |
| expected_load | +0.3261 | [+0.0680, +0.4155] | 18/25 | 0.0433 | +1.7100 | [+1.1946, +1.9027] | 23/25 | 3 | True |
| first_selection_entropy | +0.2502 | [-0.1775, +0.5317] | 16/25 | 0.23 | -3.5556 | [-3.7488, -3.1007] | 0/25 | 3 | True |

**干扰量自身的可分辨性（状态主张必须胜过这一行）：**

| 干扰量 | 中位 z | 95% CI | 方向有利 | p |
| --- | --- | --- | --- | --- |
| rate_1800s | +0.0933 | [-0.5438, +0.8230] | 14/25 | 0.69 |
| rate_7200s | -0.4636 | [-1.1589, +0.1922] | 9/25 | 0.23 |
| rate_14400s | +0.0520 | [-0.5962, +0.3520] | 15/25 | 0.424 |
| rate_28800s | +0.0898 | [-0.5603, +1.4502] | 14/25 | 0.69 |
| median_iei | -0.7465 | [-1.2988, -0.1850] | 7/25 | 0.0433 |
| coverage | -0.4173 | [-1.4877, +0.1711] | 5/17 | 0.143 |

- Holm 校正（open-loop 家族）：`{'open_loop_at_onset::resource': 0.00390625, 'open_loop_at_onset::state_norm': 0.04389989376068115, 'open_loop_at_onset::expected_load': 0.08657050132751465, 'open_loop_at_onset::first_selection_entropy': 0.2295229434967041}`

### 6.3 主分析 `resource_anchored_on_best_family`，提前量 60 min

- 可分析患者 27/34；合格发作 360；不可观测患者 7 （['yuquan_chengshuai', 'yuquan_hanyuxuan', 'yuquan_liyouran', 'yuquan_songzishuo', 'yuquan_wangyiyang', 'yuquan_zhangjiaqi', 'yuquan_zhaochenxi']）
- 可入观测器的事件 1,111,565，其中 260,891 是删减流里没有的

**filtered_at_onset** — the observer consumed every admissible event up to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | +0.2322 | [-0.2544, +0.9142] | 15/27 | 0.701 | -6.1691 | [-6.5960, -5.1715] | 1/27 | 1 | True |
| resource | -0.3285 | [-0.6456, -0.1814] | 0/11 | 0.000977 | -0.3411 | [-0.3906, -0.2501] | 1/11 | 188 | True |
| expected_load | -0.3685 | [-1.2448, +0.2635] | 10/27 | 0.248 | +4.8690 | [+3.9313, +5.5898] | 27/27 | 1 | True |
| first_selection_entropy | +0.5842 | [-0.1654, +0.8065] | 18/27 | 0.122 | -1.8591 | [-2.1860, -1.5947] | 2/27 | 1 | True |

**filtered_at_cutoff** — the observer stopped at onset minus the lead

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | +0.2513 | [-0.1122, +0.6885] | 18/27 | 0.122 | -1.7019 | [-2.2011, -1.3452] | 1/27 | 4 | True |
| resource | -0.2434 | [-0.4530, -0.2004] | 0/13 | 0.000244 | -0.2115 | [-0.2991, +0.0002] | 4/13 | 179 | True |
| expected_load | -0.2853 | [-0.7470, +0.2039] | 10/27 | 0.248 | +2.8252 | [+2.1299, +3.1520] | 26/27 | 4 | True |
| first_selection_entropy | +0.1401 | [-0.1684, +0.4792] | 16/27 | 0.442 | -1.4113 | [-1.7965, -1.0808] | 1/27 | 4 | True |

**open_loop_at_onset** — the observer stopped at the cut-off and the generator then integrated autonomously to onset

| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| state_norm | +0.0362 | [-0.3080, +0.5884] | 15/27 | 0.701 | -1.0237 | [-1.4297, -0.2438] | 7/27 | 4 | True |
| resource | -0.3276 | [-0.6989, -0.1923] | 0/11 | 0.000977 | -0.3548 | [-0.4385, -0.2535] | 0/11 | 188 | True |
| expected_load | -0.1948 | [-0.8258, +0.4571] | 11/27 | 0.442 | +0.3745 | [-0.1620, +0.8182] | 17/27 | 4 | True |
| first_selection_entropy | +0.1612 | [-0.2173, +0.5873] | 17/27 | 0.248 | +1.6288 | [+1.2385, +2.1995] | 27/27 | 4 | True |

**干扰量自身的可分辨性（状态主张必须胜过这一行）：**

| 干扰量 | 中位 z | 95% CI | 方向有利 | p |
| --- | --- | --- | --- | --- |
| rate_1800s | -0.2151 | [-0.3486, +0.5033] | 12/27 | 0.701 |
| rate_7200s | -0.4006 | [-0.6904, +0.1008] | 11/27 | 0.442 |
| rate_14400s | -0.2826 | [-0.6019, +0.1766] | 10/27 | 0.248 |
| rate_28800s | -0.3767 | [-1.1100, +0.5010] | 12/27 | 0.701 |
| median_iei | -0.4195 | [-0.7473, -0.1117] | 5/27 | 0.00151 |
| coverage | -0.5621 | [-1.1882, +0.3209] | 10/21 | 1 |

- Holm 校正（open-loop 家族）：`{'open_loop_at_onset::resource': 0.00390625, 'open_loop_at_onset::first_selection_entropy': 0.743365690112114, 'open_loop_at_onset::expected_load': 0.8841366767883301, 'open_loop_at_onset::state_norm': 0.8841366767883301}`

### 6.4 被降级的严格对照（确定间期流 / 长间隔）

- 角色：`definite_interictal_long_gap_strict_sensitivity`；the definite-interictal block policy deletes the pre-ictal observations, so this arm measures how long a state survives without observation, not whether the state moves once the pre-ictal IEDs are observed
- 患者 25、发作 117
- 退化读数：state_norm 0/117、resource 117/117、expected_load 0/117、first_selection_entropy 0/117

| 距上一事件 | 发作数 | 患者数 |
| --- | --- | --- |
| le_300s | 8 | 8 |
| le_3600s | 90 | 23 |
| le_60s | 6 | 5 |
| le_900s | 13 | 9 |

**early-ictal transfer：`NOT_RUN`** — adjudicated per-seizure clinical-onset contacts are 0 of 71 and substitutions are forbidden by a LOCKED blinding contract

- 角色：`definite_interictal_long_gap_strict_sensitivity`；the definite-interictal block policy deletes the pre-ictal observations, so this arm measures how long a state survives without observation, not whether the state moves once the pre-ictal IEDs are observed
- 患者 25、发作 117
- 退化读数：state_norm 0/117、resource 117/117、expected_load 0/117、first_selection_entropy 0/117

| 距上一事件 | 发作数 | 患者数 |
| --- | --- | --- |
| le_300s | 8 | 8 |
| le_3600s | 90 | 23 |
| le_60s | 6 | 5 |
| le_900s | 13 | 9 |

**early-ictal transfer：`NOT_RUN`** — adjudicated per-seizure clinical-onset contacts are 0 of 71 and substitutions are forbidden by a LOCKED blinding contract

- 角色：`definite_interictal_long_gap_strict_sensitivity`；the definite-interictal block policy deletes the pre-ictal observations, so this arm measures how long a state survives without observation, not whether the state moves once the pre-ictal IEDs are observed
- 患者 25、发作 117
- 退化读数：state_norm 0/117、resource 117/117、expected_load 0/117、first_selection_entropy 0/117

| 距上一事件 | 发作数 | 患者数 |
| --- | --- | --- |
| le_300s | 8 | 8 |
| le_3600s | 90 | 23 |
| le_60s | 6 | 5 |
| le_900s | 13 | 9 |

**early-ictal transfer：`NOT_RUN`** — adjudicated per-seizure clinical-onset contacts are 0 of 71 and substitutions are forbidden by a LOCKED blinding contract

- 角色：`definite_interictal_long_gap_strict_sensitivity`；the definite-interictal block policy deletes the pre-ictal observations, so this arm measures how long a state survives without observation, not whether the state moves once the pre-ictal IEDs are observed
- 患者 25、发作 117
- 退化读数：state_norm 0/117、resource 0/117、expected_load 0/117、first_selection_entropy 0/117

| 距上一事件 | 发作数 | 患者数 |
| --- | --- | --- |
| le_300s | 8 | 8 |
| le_3600s | 90 | 23 |
| le_60s | 6 | 5 |
| le_900s | 13 | 9 |

**early-ictal transfer：`NOT_RUN`** — adjudicated per-seizure clinical-onset contacts are 0 of 71 and substitutions are forbidden by a LOCKED blinding contract

- 角色：`definite_interictal_long_gap_strict_sensitivity`；the definite-interictal block policy deletes the pre-ictal observations, so this arm measures how long a state survives without observation, not whether the state moves once the pre-ictal IEDs are observed
- 患者 25、发作 117
- 退化读数：state_norm 0/117、resource 71/117、expected_load 0/117、first_selection_entropy 0/117

| 距上一事件 | 发作数 | 患者数 |
| --- | --- | --- |
| le_300s | 8 | 8 |
| le_3600s | 90 | 23 |
| le_60s | 6 | 5 |
| le_900s | 13 | 9 |

**early-ictal transfer：`NOT_RUN`** — adjudicated per-seizure clinical-onset contacts are 0 of 71 and substitutions are forbidden by a LOCKED blinding contract

- 角色：`definite_interictal_long_gap_strict_sensitivity`；the definite-interictal block policy deletes the pre-ictal observations, so this arm measures how long a state survives without observation, not whether the state moves once the pre-ictal IEDs are observed
- 患者 25、发作 117
- 退化读数：state_norm 117/117、resource 117/117、expected_load 117/117、first_selection_entropy 117/117

| 距上一事件 | 发作数 | 患者数 |
| --- | --- | --- |
| le_300s | 8 | 8 |
| le_3600s | 90 | 23 |
| le_60s | 6 | 5 |
| le_900s | 13 | 9 |

**early-ictal transfer：`NOT_RUN`** — adjudicated per-seizure clinical-onset contacts are 0 of 71 and substitutions are forbidden by a LOCKED blinding contract

- 角色：`definite_interictal_long_gap_strict_sensitivity`；the definite-interictal block policy deletes the pre-ictal observations, so this arm measures how long a state survives without observation, not whether the state moves once the pre-ictal IEDs are observed
- 患者 25、发作 117
- 退化读数：state_norm 0/117、resource 117/117、expected_load 38/117、first_selection_entropy 45/117

| 距上一事件 | 发作数 | 患者数 |
| --- | --- | --- |
| le_300s | 8 | 8 |
| le_3600s | 90 | 23 |
| le_60s | 6 | 5 |
| le_900s | 13 | 9 |

**early-ictal transfer：`NOT_RUN`** — adjudicated per-seizure clinical-onset contacts are 0 of 71 and substitutions are forbidden by a LOCKED blinding contract



## 7. H3a / H3b：resource 与 IED exposure

### 7.1 τ_r 冻结（在任何 exposure 臂之前）

- 选中 τ_r = 7200 s；规则：one-standard-error over the declared grid, slowest tau inside the band
- 可辨识 = `False`；一个标准误带内的区间 [60.0, 7200.0] s
- exposure 结果参与了选择 = `False`

| τ_r (s) | seeds | 患者 | 平均验证 | SEM |
| --- | --- | --- | --- | --- |
| 60 | 3 | 34 | 2.44119 | 0.10563 |
| 300 | 3 | 34 | 2.43758 | 0.10525 |
| 1800 | 3 | 34 | 2.43106 | 0.10505 |
| 7200 | 3 | 34 | 2.43054 | 0.10472 |

### 7.2 resource ladder 运行状态

| 臂 | resource | seed | 状态 | 最优验证 | τ_r | τ_x | 核 | γ_q | γ_L | γ_x | 边界占用 | 塌缩 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t1_r0 | R0 | 11 | COMPLETE | 2.067 | — | — | clock | — | — | — | 0 | False |
| t1_r0 | R0 | 12 | COMPLETE | 2.083 | — | — | clock | — | — | — | 0 | False |
| t1_r0 | R0 | 13 | COMPLETE | 2.085 | — | — | clock | — | — | — | 0 | False |
| t1_r1_free_tau | R1 | 11 | COMPLETE | 2.071 | 2606 | — | clock | 0.0001387 | — | — | 0 | False |
| t1_r1_free_tau | R1 | 12 | COMPLETE | 2.07 | 7884 | — | clock | 0.000345 | — | — | 0 | False |
| t1_r1_free_tau | R1 | 13 | COMPLETE | 2.076 | 1259 | — | clock | 8.554e-05 | — | — | 0 | False |
| t1_r1_tau1800 | R1 | 11 | COMPLETE | 2.069 | 1800 | — | clock | 7.099e-05 | — | — | 0 | False |
| t1_r1_tau1800 | R1 | 12 | COMPLETE | 2.099 | 1800 | — | clock | 0.0008774 | — | — | 0 | False |
| t1_r1_tau1800 | R1 | 13 | COMPLETE | 2.074 | 1800 | — | clock | 0.001084 | — | — | 0 | False |
| t1_r1_tau300 | R1 | 11 | COMPLETE | 2.072 | 300 | — | clock | 7.606e-05 | — | — | 0 | False |
| t1_r1_tau300 | R1 | 12 | COMPLETE | 2.089 | 300 | — | clock | 0.0002923 | — | — | 0 | False |
| t1_r1_tau300 | R1 | 13 | COMPLETE | 2.084 | 300 | — | clock | 0.000106 | — | — | 0 | False |
| t1_r1_tau60 | R1 | 11 | COMPLETE | 2.065 | 60 | — | clock | 6.104e-05 | — | — | 0 | False |
| t1_r1_tau60 | R1 | 12 | COMPLETE | 2.084 | 60 | — | clock | 0.0001211 | — | — | 0 | False |
| t1_r1_tau60 | R1 | 13 | COMPLETE | 2.101 | 60 | — | clock | 0.0002791 | — | — | 0 | False |
| t1_r1_tau7200 | R1 | 11 | COMPLETE | 2.072 | 7200 | — | clock | 6.263e-05 | — | — | 0 | False |
| t1_r1_tau7200 | R1 | 12 | COMPLETE | 2.079 | 7200 | — | clock | 9.91e-05 | — | — | 0 | False |
| t1_r1_tau7200 | R1 | 13 | COMPLETE | 2.085 | 7200 | — | clock | 0.0002243 | — | — | 0 | False |
| t2_r2 | R2 | 11 | COMPLETE | 2.081 | 7200 | — | clock | 0.0001979 | 0.02137 | — | 0 | False |
| t2_r2 | R2 | 12 | COMPLETE | 2.08 | 7200 | — | clock | 0.0001768 | 0.008866 | — | 0 | False |
| t2_r2 | R2 | 13 | COMPLETE | 2.099 | 7200 | — | clock | 0.0001087 | 0.009135 | — | 0 | False |
| t2_r3_clock14400 | R3 | 11 | COMPLETE | 2.083 | 7200 | 1.44e+04 | clock | 7.944e-05 | — | 0.0001468 | 0.8592 | True |
| t2_r3_clock14400 | R3 | 12 | COMPLETE | 2.078 | 7200 | 1.44e+04 | clock | 7.024e-05 | — | 0.0001037 | 0.7505 | True |
| t2_r3_clock14400 | R3 | 13 | COMPLETE | 2.091 | 7200 | 1.44e+04 | clock | 8.153e-05 | — | 0.0002531 | 0.9298 | True |
| t2_r3_clock1800 | R3 | 11 | COMPLETE | 2.077 | 7200 | 1800 | clock | 0.0001044 | — | 0.0001164 | 0.0962 | False |
| t2_r3_clock1800 | R3 | 12 | COMPLETE | 2.085 | 7200 | 1800 | clock | 0.0001101 | — | 0.0001724 | 0.2248 | False |
| t2_r3_clock1800 | R3 | 13 | COMPLETE | 2.188 | 7200 | 1800 | clock | 7.395e-05 | — | 7.026e-07 | 0 | False |
| t2_r3_clock28800 | R3 | 11 | COMPLETE | 2.073 | 7200 | 2.88e+04 | clock | 8.575e-05 | — | 0.0001772 | 0.9413 | True |
| t2_r3_clock28800 | R3 | 12 | COMPLETE | 2.08 | 7200 | 2.88e+04 | clock | 8.366e-05 | — | 0.0001674 | 0.9388 | True |
| t2_r3_clock28800 | R3 | 13 | COMPLETE | 2.103 | 7200 | 2.88e+04 | clock | 0.0001197 | — | 0.0001644 | 0.938 | True |
| t2_r3_clock300 | R3 | 11 | COMPLETE | 2.077 | 7200 | 300 | clock | 8.423e-05 | — | 0.0001203 | 0 | False |
| t2_r3_clock300 | R3 | 12 | COMPLETE | 2.075 | 7200 | 300 | clock | 0.0001224 | — | 0.0002003 | 0 | False |
| t2_r3_clock300 | R3 | 13 | COMPLETE | 2.086 | 7200 | 300 | clock | 0.0001964 | — | 0.0001184 | 0 | False |
| t2_r3_clock3600 | R3 | 11 | COMPLETE | 2.075 | 7200 | 3600 | clock | 0.0001092 | — | 0.000182 | 0.4712 | True |
| t2_r3_clock3600 | R3 | 12 | COMPLETE | 2.098 | 7200 | 3600 | clock | 0.0001291 | — | 0.0001683 | 0.4537 | True |
| t2_r3_clock3600 | R3 | 13 | COMPLETE | 2.082 | 7200 | 3600 | clock | 7.158e-05 | — | 0.0001334 | 0.3759 | True |
| t2_r3_clock7200 | R3 | 11 | COMPLETE | 2.072 | 7200 | 7200 | clock | 9.224e-05 | — | 0.000215 | 0.7818 | True |
| t2_r3_clock7200 | R3 | 12 | COMPLETE | 2.095 | 7200 | 7200 | clock | 8.27e-05 | — | 0.0001685 | 0.6727 | True |
| t2_r3_clock7200 | R3 | 13 | COMPLETE | 2.097 | 7200 | 7200 | clock | 7.036e-05 | — | 0.0001086 | 0.5126 | True |
| t2_r3_clock900 | R3 | 11 | COMPLETE | 2.081 | 7200 | 900 | clock | 0.000114 | — | 8.391e-05 | 0 | False |
| t2_r3_clock900 | R3 | 12 | COMPLETE | 2.069 | 7200 | 900 | clock | 0.0001327 | — | 0.0001397 | 0 | False |
| t2_r3_clock900 | R3 | 13 | COMPLETE | 2.102 | 7200 | 900 | clock | 0.0001377 | — | 5.775e-05 | 0 | False |
| t2_r3_events10 | R3 | 11 | COMPLETE | 2.074 | 7200 | 10 | event_count | 8.716e-05 | — | 0.0001332 | 0 | False |
| t2_r3_events10 | R3 | 12 | COMPLETE | 2.078 | 7200 | 10 | event_count | 5.923e-05 | — | 0.0007612 | 0 | True |
| t2_r3_events10 | R3 | 13 | COMPLETE | 2.083 | 7200 | 10 | event_count | 0.0002003 | — | 0.0003122 | 0 | False |
| t2_r3_events20 | R3 | 11 | COMPLETE | 2.079 | 7200 | 20 | event_count | 7.934e-05 | — | 0.0001618 | 0 | False |
| t2_r3_events20 | R3 | 12 | COMPLETE | 2.071 | 7200 | 20 | event_count | 0.0001036 | — | 0.0002001 | 0 | False |
| t2_r3_events20 | R3 | 13 | COMPLETE | 2.1 | 7200 | 20 | event_count | 8.852e-05 | — | 0.0002445 | 0 | True |
| t2_r3_events40 | R3 | 11 | COMPLETE | 2.079 | 7200 | 40 | event_count | 0.0001023 | — | 0.0001515 | 0 | True |
| t2_r3_events40 | R3 | 12 | COMPLETE | 2.069 | 7200 | 40 | event_count | 7.274e-05 | — | 0.0001033 | 0 | False |
| t2_r3_events40 | R3 | 13 | COMPLETE | 2.076 | 7200 | 40 | event_count | 6.59e-05 | — | 0.0001474 | 0 | True |
| t2_r3_events5 | R3 | 11 | COMPLETE | 2.081 | 7200 | 5 | event_count | 8.559e-05 | — | 0.0001057 | 0 | False |
| t2_r3_events5 | R3 | 12 | COMPLETE | 2.085 | 7200 | 5 | event_count | 7.709e-05 | — | 7.476e-05 | 0 | False |
| t2_r3_events5 | R3 | 13 | COMPLETE | 2.085 | 7200 | 5 | event_count | 0.0003655 | — | 0.0003977 | 0 | False |
| t2_r3_events80 | R3 | 11 | COMPLETE | 2.077 | 7200 | 80 | event_count | 8.156e-05 | — | 0.0001099 | 0 | True |
| t2_r3_events80 | R3 | 12 | COMPLETE | 2.085 | 7200 | 80 | event_count | 8.969e-05 | — | 0.0001993 | 0 | True |
| t2_r3_events80 | R3 | 13 | COMPLETE | 2.073 | 7200 | 80 | event_count | 6.016e-05 | — | 6.61e-05 | 0 | True |

### 7.3 H3a predictive leg（相对匹配基臂）

| 端点 | 臂 | 中位差 | 95% CI | 方向有利 | 符号检验 p |
| --- | --- | --- | --- | --- | --- |
| event_nll | t1_r0 | +0.0066 | [+0.0037, +0.0074] | 6/34 | 0.000195 |
| event_nll | t1_r1_tau1800 | +0.0015 | [-0.0003, +0.0032] | 13/34 | 0.229 |
| event_nll | t1_r1_tau300 | +0.0072 | [+0.0051, +0.0088] | 3/34 | 7.66e-07 |
| event_nll | t1_r1_tau60 | +0.0047 | [+0.0025, +0.0100] | 5/34 | 3.86e-05 |
| event_nll | t1_r1_tau7200 | +0.0031 | [+0.0021, +0.0048] | 9/34 | 0.00904 |
| event_nll | t2_r2 | +0.0070 | [+0.0027, +0.0090] | 8/34 | 0.00294 |
| event_nll | t2_r3_clock14400 | +0.0029 | [-0.0004, +0.0057] | 13/34 | 0.229 |
| event_nll | t2_r3_clock1800 | +0.0088 | [+0.0030, +0.0143] | 5/34 | 3.86e-05 |
| event_nll | t2_r3_clock28800 | +0.0061 | [+0.0028, +0.0089] | 5/34 | 3.86e-05 |
| event_nll | t2_r3_clock300 | +0.0019 | [+0.0010, +0.0039] | 10/34 | 0.0243 |
| event_nll | t2_r3_clock3600 | +0.0055 | [+0.0041, +0.0102] | 4/34 | 6.16e-06 |
| event_nll | t2_r3_clock7200 | +0.0051 | [+0.0018, +0.0086] | 9/34 | 0.00904 |
| event_nll | t2_r3_clock900 | +0.0063 | [+0.0034, +0.0104] | 6/34 | 0.000195 |
| event_nll | t2_r3_events10 | +0.0036 | [+0.0025, +0.0051] | 8/34 | 0.00294 |
| event_nll | t2_r3_events20 | +0.0022 | [+0.0007, +0.0051] | 10/34 | 0.0243 |
| event_nll | t2_r3_events40 | +0.0030 | [+0.0020, +0.0045] | 3/34 | 7.66e-07 |
| event_nll | t2_r3_events5 | +0.0041 | [-0.0001, +0.0090] | 12/34 | 0.121 |
| event_nll | t2_r3_events80 | +0.0055 | [+0.0029, +0.0082] | 1/34 | 4.07e-09 |
| order_nll | t1_r0 | +0.0014 | [-0.0001, +0.0038] | 12/34 | 0.121 |
| order_nll | t1_r1_tau1800 | +0.0036 | [+0.0019, +0.0066] | 9/34 | 0.00904 |
| order_nll | t1_r1_tau300 | +0.0070 | [+0.0048, +0.0113] | 4/34 | 6.16e-06 |
| order_nll | t1_r1_tau60 | +0.0048 | [+0.0035, +0.0068] | 6/34 | 0.000195 |
| order_nll | t1_r1_tau7200 | +0.0060 | [+0.0021, +0.0094] | 8/34 | 0.00294 |
| order_nll | t2_r2 | +0.0080 | [+0.0034, +0.0156] | 8/34 | 0.00294 |
| order_nll | t2_r3_clock14400 | -0.0036 | [-0.0068, +0.0035] | 20/34 | 0.392 |
| order_nll | t2_r3_clock1800 | -0.0061 | [-0.0124, -0.0033] | 27/34 | 0.000821 |
| order_nll | t2_r3_clock28800 | +0.0012 | [-0.0016, +0.0052] | 15/34 | 0.608 |
| order_nll | t2_r3_clock300 | +0.0063 | [+0.0025, +0.0112] | 9/34 | 0.00904 |
| order_nll | t2_r3_clock3600 | -0.0011 | [-0.0049, +0.0012] | 18/34 | 0.864 |
| order_nll | t2_r3_clock7200 | +0.0009 | [-0.0025, +0.0029] | 15/34 | 0.608 |
| order_nll | t2_r3_clock900 | +0.0044 | [+0.0017, +0.0125] | 9/34 | 0.00904 |
| order_nll | t2_r3_events10 | +0.0031 | [+0.0016, +0.0067] | 9/34 | 0.00904 |
| order_nll | t2_r3_events20 | +0.0016 | [-0.0034, +0.0046] | 15/34 | 0.608 |
| order_nll | t2_r3_events40 | +0.0018 | [-0.0020, +0.0043] | 16/34 | 0.864 |
| order_nll | t2_r3_events5 | +0.0067 | [+0.0027, +0.0132] | 7/34 | 0.000821 |
| order_nll | t2_r3_events80 | +0.0065 | [+0.0038, +0.0091] | 3/34 | 7.66e-07 |
| selection_nll | t1_r0 | +0.0049 | [+0.0021, +0.0064] | 5/34 | 3.86e-05 |
| selection_nll | t1_r1_tau1800 | +0.0011 | [+0.0003, +0.0023] | 11/34 | 0.0576 |
| selection_nll | t1_r1_tau300 | +0.0063 | [+0.0052, +0.0085] | 2/34 | 6.94e-08 |
| selection_nll | t1_r1_tau60 | +0.0033 | [+0.0017, +0.0049] | 6/34 | 0.000195 |
| selection_nll | t1_r1_tau7200 | +0.0038 | [+0.0023, +0.0054] | 7/34 | 0.000821 |
| selection_nll | t2_r2 | +0.0085 | [+0.0047, +0.0097] | 2/34 | 6.94e-08 |
| selection_nll | t2_r3_clock14400 | +0.0035 | [+0.0020, +0.0066] | 8/34 | 0.00294 |
| selection_nll | t2_r3_clock1800 | +0.0055 | [+0.0021, +0.0085] | 5/34 | 3.86e-05 |
| selection_nll | t2_r3_clock28800 | +0.0066 | [+0.0029, +0.0109] | 3/34 | 7.66e-07 |
| selection_nll | t2_r3_clock300 | +0.0038 | [+0.0028, +0.0065] | 5/34 | 3.86e-05 |
| selection_nll | t2_r3_clock3600 | +0.0040 | [+0.0029, +0.0076] | 7/34 | 0.000821 |
| selection_nll | t2_r3_clock7200 | +0.0055 | [+0.0045, +0.0111] | 4/34 | 6.16e-06 |
| selection_nll | t2_r3_clock900 | +0.0061 | [+0.0040, +0.0092] | 5/34 | 3.86e-05 |
| selection_nll | t2_r3_events10 | +0.0041 | [+0.0021, +0.0056] | 7/34 | 0.000821 |
| selection_nll | t2_r3_events20 | +0.0040 | [+0.0020, +0.0075] | 7/34 | 0.000821 |
| selection_nll | t2_r3_events40 | +0.0011 | [+0.0005, +0.0022] | 8/34 | 0.00294 |
| selection_nll | t2_r3_events5 | +0.0081 | [+0.0052, +0.0095] | 5/34 | 3.86e-05 |
| selection_nll | t2_r3_events80 | +0.0056 | [+0.0034, +0.0080] | 1/34 | 4.07e-09 |
| stop_nll | t1_r0 | +0.0003 | [+0.0002, +0.0005] | 5/34 | 3.86e-05 |
| stop_nll | t1_r1_tau1800 | +0.0001 | [+0.0000, +0.0001] | 10/34 | 0.0243 |
| stop_nll | t1_r1_tau300 | +0.0000 | [+0.0000, +0.0001] | 11/34 | 0.0576 |
| stop_nll | t1_r1_tau60 | +0.0004 | [+0.0002, +0.0007] | 7/34 | 0.000821 |
| stop_nll | t1_r1_tau7200 | -0.0001 | [-0.0004, -0.0000] | 26/34 | 0.00294 |
| stop_nll | t2_r2 | +0.0000 | [-0.0004, +0.0001] | 13/34 | 0.229 |
| stop_nll | t2_r3_clock14400 | -0.0001 | [-0.0018, -0.0000] | 30/34 | 6.16e-06 |
| stop_nll | t2_r3_clock1800 | +0.0000 | [-0.0000, +0.0002] | 13/34 | 0.229 |
| stop_nll | t2_r3_clock28800 | +0.0002 | [+0.0001, +0.0003] | 8/34 | 0.00294 |
| stop_nll | t2_r3_clock300 | -0.0001 | [-0.0010, -0.0000] | 26/34 | 0.00294 |
| stop_nll | t2_r3_clock3600 | +0.0003 | [+0.0001, +0.0010] | 5/34 | 3.86e-05 |
| stop_nll | t2_r3_clock7200 | -0.0000 | [-0.0012, -0.0000] | 25/34 | 0.00904 |
| stop_nll | t2_r3_clock900 | +0.0001 | [+0.0000, +0.0003] | 7/34 | 0.000821 |
| stop_nll | t2_r3_events10 | -0.0001 | [-0.0001, -0.0000] | 27/34 | 0.000821 |
| stop_nll | t2_r3_events20 | -0.0000 | [-0.0010, +0.0000] | 17/34 | 1 |
| stop_nll | t2_r3_events40 | +0.0003 | [+0.0000, +0.0006] | 8/34 | 0.00294 |
| stop_nll | t2_r3_events5 | -0.0000 | [-0.0025, +0.0000] | 19/34 | 0.608 |
| stop_nll | t2_r3_events80 | -0.0001 | [-0.0001, -0.0000] | 26/34 | 0.00294 |
| participation_nll | t1_r0 | +0.0017 | [+0.0007, +0.0040] | 9/34 | 0.00904 |
| participation_nll | t1_r1_tau1800 | +0.0023 | [+0.0011, +0.0036] | 9/34 | 0.00904 |
| participation_nll | t1_r1_tau300 | +0.0028 | [+0.0018, +0.0036] | 6/34 | 0.000195 |
| participation_nll | t1_r1_tau60 | +0.0042 | [+0.0023, +0.0056] | 5/34 | 3.86e-05 |
| participation_nll | t1_r1_tau7200 | +0.0011 | [+0.0005, +0.0017] | 10/34 | 0.0243 |
| participation_nll | t2_r2 | +0.0054 | [+0.0029, +0.0075] | 1/34 | 4.07e-09 |
| participation_nll | t2_r3_clock14400 | +0.0059 | [+0.0038, +0.0091] | 1/34 | 4.07e-09 |
| participation_nll | t2_r3_clock1800 | +0.0074 | [+0.0053, +0.0094] | 0/34 | 1.16e-10 |
| participation_nll | t2_r3_clock28800 | +0.0021 | [+0.0009, +0.0050] | 6/34 | 0.000195 |
| participation_nll | t2_r3_clock300 | +0.0018 | [+0.0004, +0.0027] | 9/34 | 0.00904 |
| participation_nll | t2_r3_clock3600 | +0.0037 | [+0.0023, +0.0059] | 1/34 | 4.07e-09 |
| participation_nll | t2_r3_clock7200 | +0.0092 | [+0.0074, +0.0154] | 1/34 | 4.07e-09 |
| participation_nll | t2_r3_clock900 | +0.0070 | [+0.0059, +0.0098] | 2/34 | 6.94e-08 |
| participation_nll | t2_r3_events10 | +0.0035 | [+0.0015, +0.0047] | 4/34 | 6.16e-06 |
| participation_nll | t2_r3_events20 | +0.0053 | [+0.0032, +0.0067] | 4/34 | 6.16e-06 |
| participation_nll | t2_r3_events40 | +0.0017 | [+0.0006, +0.0028] | 9/34 | 0.00904 |
| participation_nll | t2_r3_events5 | +0.0023 | [+0.0013, +0.0035] | 5/34 | 3.86e-05 |
| participation_nll | t2_r3_events80 | +0.0007 | [-0.0004, +0.0026] | 13/34 | 0.229 |

### 7.4 exposure timescale curve（端点 order_nll）

| 核 | 尺度 | 中位差 | 95% CI | 方向有利 |
| --- | --- | --- | --- | --- |
| clock | 300 | +0.0063 | [+0.0025, +0.0112] | 9/34 |
| clock | 900 | +0.0044 | [+0.0017, +0.0125] | 9/34 |
| clock | 1800 | -0.0061 | [-0.0124, -0.0033] | 27/34 |
| clock | 3600 | -0.0011 | [-0.0049, +0.0012] | 18/34 |
| clock | 7200 | +0.0009 | [-0.0025, +0.0029] | 15/34 |
| clock | 14400 | -0.0036 | [-0.0068, +0.0035] | 20/34 |
| clock | 28800 | +0.0012 | [-0.0016, +0.0052] | 15/34 |
| event_count | 5 | +0.0067 | [+0.0027, +0.0132] | 7/34 |
| event_count | 10 | +0.0031 | [+0.0016, +0.0067] | 9/34 |
| event_count | 20 | +0.0016 | [-0.0034, +0.0046] | 15/34 |
| event_count | 40 | +0.0018 | [-0.0020, +0.0043] | 16/34 |
| event_count | 80 | +0.0065 | [+0.0038, +0.0091] | 3/34 |

### 7.5 H3a innovation / directionality leg

- 冻结 T1 来源：`t1_r1_tau60` (`goal4_exposure__G3-R1-node_film__t1_r1_tau60__s11__development__all34__f3182510a216abde`)
- expected load 模型：frozen T1 state summary + IEI + local rate + time of day, blocked expanding cross-fit with a 200-event embargo
- outcome：mean masked-order NLL over the next 20 events, residualised on IEI, local rate and time of day

| τ_x (s) | 患者 | 真实 vs 零 | 真实−状态匹配打乱 | 真实−时间反转 | 真实−事件计数核 | 真实−段打乱 | 真实−原始负荷核 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 300 | 32 | +0.1028 (24/32) | +0.0790 (23/32) | -0.1481 (0/32) | -0.0037 (15/32) | +0.0000 (15/32) | +0.0470 (19/32) |
| 1800 | 32 | +0.0437 (21/32) | +0.0248 (19/32) | -0.1010 (5/32) | -0.0393 (12/32) | +0.0000 (11/32) | +0.0424 (19/32) |
| 7200 | 32 | +0.0147 (17/32) | +0.0068 (18/32) | -0.0441 (9/32) | -0.0546 (11/32) | +0.0000 (11/32) | +0.0260 (19/32) |

### 7.6 H3a evidence card 摘要

```json
{
  "reference_arm": "t1_r1_free_tau",
  "resource_health": {
    "n_collapsed_runs": 19,
    "n_static_runs": 13,
    "n_never_consumed_runs": null,
    "median_floor_occupancy": 0.0,
    "note": "a resource that collapsed to its floor, sat at its ceiling, or never moved carries no resource information; such an arm's comparison must be read as 'this pathway was not used', not as evidence about a resource. The floor and the ceiling are opposite failures and are counted separately."
  },
  "holm_corrected_primary_family": {
    "order_nll::t2_r3_events80": 9.958166629076004e-06,
    "order_nll::t2_r3_clock1800": 0.00985674373805523,
    "order_nll::t2_r3_events5": 0.00985674373805523,
    "order_nll::t2_r2": 0.029350556433200836,
    "order_nll::t2_r3_clock300": 0.08137066941708326,
    "order_nll::t2_r3_clock900": 0.08137066941708326,
    "order_nll::t2_r3_events10": 0.08137066941708326,
    "order_nll::t2_r3_clock14400": 1.0,
    "order_nll::t2_r3_clock28800": 1.0,
    "order_nll::t2_r3_clock7200": 1.0,
    "order_nll::t2_r3_events20": 1.0,
    "order_nll::t2_r3_clock3600": 1.0,
    "order_nll::t2_r3_events40": 1.0
  },
  "denominators": {
    "n_runs": 57,
    "arms": [
      "t1_r0",
      "t1_r1_free_tau",
      "t1_r1_tau1800",
      "t1_r1_tau300",
      "t1_r1_tau60",
      "t1_r1_tau7200",
      "t2_r2",
      "t2_r3_clock14400",
      "t2_r3_clock1800",
      "t2_r3_clock28800",
      "t2_r3_clock300",
      "t2_r3_clock3600",
      "t2_r3_clock7200",
      "t2_r3_clock900",
      "t2_r3_events10",
      "t2_r3_events20",
      "t2_r3_events40",
      "t2_r3_events5",
      "t2_r3_events80"
    ],
    "n_patients": 34,
    "n_epilepsiae": 18,
    "n_yuquan": 16
  }
}
```

### 7.7 H3b

- 状态：`READ_ONLY_COMBINATION`
- H3a supported AND H2b supported AND the two point the same way
- H3b is only asserted when both legs are supported and agree in direction; otherwise it is reported as not asserted, which is not a negative for H1, H2a, H2b or H3a



## 8. 数值稳定性、资源边界与 observer 预算

- 非有限损失导致的失败运行：0
- 稳定裕度（最小阻尼率）中位数：0.000093（正值表示线性部分收缩）
- 拟合状态时间常数中位数：370.4 s（范围 174.9 – 6179.0 s）
- observer 校正能量中位数：0.14728
- 资源触底比例中位数：0.00000

**积分器：**指数积分（exponential Euler）。在消息项于一步内冻结的前提下，线性部分的解 `target + (H − target)·exp(−rate·Δt)` 是精确的，对任意 Δt 有界。本队列最大真实事件间隔为 5.2e5 秒；显式 Euler 在那里需要数千个子步，或者直接发散。单元测试在该间隔上验证了四级生成器与四条资源臂全部保持有限且有界。

**时间常数参数化：**`τ = exp(clamp(log τ, log 0.5, log 1e6))`，八个状态维度按 10 秒到 3 小时对数等间隔初始化。先前的 `softplus` 参数化在训练预算内最多只能达到约 20 秒，使模型无法表示分钟到小时尺度的慢状态；受影响的运行已归档到 `results/epi_prssm/v0_1/_invalidated_tau_parametrisation/` 并全部重跑。


## 9. 工程记录：worker 规模与资源

- 作业总数 690；终态分布 `{'FAILED': 3, 'COMPLETE': 687}`
- 单作业峰值常驻内存：中位 648 MiB，最大 1446 MiB
- worker 上限计算：假定峰值 1.6 GiB × 安全系数 1.25，系统内存保留 20.0 GiB 或总量的 0.2，CPU 保留 2 核，磁盘低水位 6.0 GiB
- 计算出的 worker 上限：36

每个 worker 强制 `OMP_NUM_THREADS=1` 等四项线程环境变量并 `torch.set_num_threads(1)`；实测每进程恰好占用 1 个逻辑核（30 个 worker 合计约 2977% CPU），未发生 oversubscription。

**为什么全部在 CPU 上跑：**状态维度 8、触点数 6–52，单步张量极小，GPU 上的 kernel launch 开销主导，单卡串行的吞吐远低于数十个单核进程并行。队列内批处理把每患者-事件的扫描成本从 861 μs 降到 41 μs（21 倍），读出仍按患者在未填充张量上计算，避免 6 触点患者为 52 触点患者买单。


## 10. 图形产出

| asset_id | 已生成 | PNG | PDF | metadata | README |
| --- | --- | --- | --- | --- | --- |
| `epi_prssm_architecture_ladder` | True | ✓ | ✓ | ✓ | ✓ |
| `epi_prssm_generator_evidence` | True | ✓ | ✓ | ✓ | ✓ |
| `epi_prssm_event_distribution` | True | ✓ | ✓ | ✓ | ✓ |
| `epi_prssm_seizure_link` | True | ✓ | ✓ | ✓ | ✓ |
| `epi_prssm_exposure_mechanism` | True | ✓ | ✓ | ✓ | ✓ |

每个 asset 的 PNG（600 dpi）、矢量 PDF、metadata JSON 与中文 README 由同一次运行产出。README 在图实际生成之后写入，不放空模板。


## 11. 精确复现命令

```bash
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
cd /home/honglab/leijiaxin/HFOsp

# 0. 构建队列缓存与 Goal 0（数据/划分/禁止输入/基线/inventory/Hard Gate A）
$PY scripts/topic5_epi_prssm/prepare_cohort.py
$PY scripts/topic5_epi_prssm/run_goal0.py
$PY -m pytest tests/topic5_epi_prssm -q

# 1. just-in-time synthetic 标定
$PY scripts/topic5_epi_prssm/build_plan.py --stage synthetic --seeds 0 1 2
$PY scripts/topic5_epi_prssm/launch_autonomous.py \
   --plan results/epi_prssm/v0_1/manifests/plans/synthetic_all34.json --tag synthetic --cap 14
$PY scripts/topic5_epi_prssm/aggregate_synthetic.py

# 2. Goal 1 → Goal 2 → Goal 4 → Goal 3 → 图 → 汇总（一条命令串完）
$PY scripts/topic5_epi_prssm/build_plan.py --stage goal1 --cohort all34 --epochs 12
$PY scripts/topic5_epi_prssm/launch_autonomous.py \
   --plan results/epi_prssm/v0_1/manifests/plans/goal1_all34.json --cap 36
$PY scripts/topic5_epi_prssm/run_full_matrix.py --cohort all34 --epochs 12 --cap 50 \
   --wait-goal1-plan results/epi_prssm/v0_1/manifests/plans/goal1_all34.json

# 3. 报告
$PY scripts/topic5_epi_prssm/write_final_summary.py --cohort all34
$PY scripts/topic5_epi_prssm/write_reports.py --cohort all34
```


## 12. 未完成单元与具体原因

| 单元 | 状态 | 原因 |
| --- | --- | --- |
| `goal1_generator__G0-R0-no_state__static__s11__development__all34__2b00e1e2bd5f5252` | FAILED | stopped on purpose: this run used the code in which the no_state adapter's STOP and participation heads still received the graph state, so the static reference was not state-free; re-run under plan goal1_static_rerun.json |
| `goal1_generator__G0-R0-no_state__static__s12__development__all34__a418ffe5b1ef89d6` | FAILED | stopped on purpose: this run used the code in which the no_state adapter's STOP and participation heads still received the graph state, so the static reference was not state-free; re-run under plan goal1_static_rerun.json |
| `goal1_generator__G0-R0-no_state__static__s13__development__all34__683b07b6c7a58fdb` | FAILED | stopped on purpose: this run used the code in which the no_state adapter's STOP and participation heads still received the graph state, so the static reference was not state-free; re-run under plan goal1_static_rerun.json |
| `goal3_task_3_3_early_ictal_transfer` | NOT_RUN | the primary form needs adjudicated per-seizure clinical-onset contacts; the registry holds 0 of 71 consensus annotations and its blinding contract is LOCKED against SOZ, patient focus, A/B template and energy-top substitutions. An energy-field surrogate was not used because its channel mapping to this cohort's contact order has not been audited under Hard Gate A. |
| `goal5_learned_event_encoder` | NOT_RUN | Goal 5 is explicitly not a gate; the explicit-mark ladder consumed the available wall-clock, and no learned-encoder arm was started |


## 13. claim boundary 与建议论文措辞

**允许的措辞：**

- H1: structured graph recurrent slow state (development partition, 34 patients)
- H2a: reported against a capacity-matched frozen-state control on 34 patients; 31 patients were eligible for the ambiguous-prefix targeted analysis
- H2b: 182 seizures in 27 patients meet the pre-ictal observation premise; the frozen interictal model observes the pre-ictal IEDs and the observer is closed at a declared lead, and every state endpoint is reported both raw and after residualising on multi-scale rate, interval and coverage
- the definite-interictal arm is reported as a strict missing-observation and long-extrapolation control, not as H2b
- H3a: the primary outcome is the masked recruitment-order likelihood, which is invariant to how many contacts participated

**禁止的措辞：**

- the model proves that IED exposure causes seizures
- the slow state is a seizure clock
- a seizure-link result read off the definite-interictal stream, whose block policy deletes the pre-ictal observations
- the resource is a measured metabolic variable
- anatomical rewiring or synaptic remodelling
- a confirmatory result from an untouched test partition

**证据卡各自独立：**H3 阴性不降低 H1、H2a 或 H2b；H2b 阴性只关闭 transition 解释，不影响 H3a；歧义前缀支持不足记为不适用，不记为 H2a 失败。
