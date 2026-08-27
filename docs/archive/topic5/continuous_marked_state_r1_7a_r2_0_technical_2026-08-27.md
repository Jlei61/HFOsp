# Continuous Marked State R1.7A / T2-R2.0 技术报告

## 0. 本轮回答了什么

R1.7A 在 10 位事前固定、从未参与任何架构 / 优化器 / 预算 / 阈值选择的 development 患者上，
用完全冻结的 R1.6 配置检验两件事：跨观察窗口保留的状态是否优于逐窗口重新编码
（persistent vs memoryless），以及正确时刻的状态是否优于同条件匹配的错误时刻状态
（correct vs matched wrong-time）。只有同时通过两层的患者，才用独立的后段数据运行
T2-R2.0，检验最近 100 次事件的 load innovation 或 participation-composition innovation
是否在已知 pre-event state 之外仍有增量。

**验收结论三条：**

1. 按事前判据（5 seeds 中 ≥3 同时通过两层），**4/10 患者复现**；上一轮为 1/6。
   但按 TRAIN 决定长度的连续时间块 bootstrap 区间，只有 `epilepsiae_1125`
   在充足独立块（37/35）上两层都不跨零。
2. H2a 的增益集中在 selecting group size 与 later continuation；
   `first subset` 方向不一致，在最强患者上区间显著不利。
   合同列的三个主端点中，`same_prefix_continuation` 在本队列**与 continuation 完全等值**，
   实际只有两个独立端点。
3. T2-R2.0 **未获得可采信支持**。冻结聚合器把 8 个 patient×source 单元中的 2 个判为
   `patient_source_support`（`epilepsiae_1125`/participation 与 `yuquan_zhangbichen`/load），
   但逐项复核后：`yuquan_zhangbichen`/load 是**安慰剂退化造成的假阳性**（见 §7.5），
   `epilepsiae_1125`/participation 内部一致但**缺少独立时间块确认**，
   而独立块确认是上一轮 R1.6 最小 H3 的判据之一。
   因此本轮 **8/8 单元均未达到上一轮自身的证据标准**；一个候选待逐块跟进。

正式检验分区、sealed 分区、seizure probe 与 paper-ready figures 全程未打开或未修改。

## 1. 冻结队列与排除 provenance

选择脚本 `build_r1_7a_inventory.py` 不读取任何模型结果，规则为：
先排除 13 位参与过旧决策的患者（provenance 见 `EXCLUSION_PROVENANCE`，含 R1.6 confirmation
面板全部 6 人），再要求 `n_contacts ≥ 6`、`train_events ≥ 1000`、`validation_events ≥ 300`、
`train_recorded_seconds ≥ 21600`、`validation_recorded_seconds ≥ 5400`，
最后按 (validation_events desc, validation_recorded_seconds desc, subject asc)
每个数据集取前 5 位。脚本内含一致性断言：选出的集合必须逐元素等于
`contract.R1_7A_SUBJECTS`，否则抛错。

## 2. 记录时长 60/40 分层

`split_validation_by_recorded_time` 对 validation 区间内的每个 coverage interval 求交，
按累计**记录时长**（非墙钟）取 60% 分界。因此跨记录缺口的时间不计入分层，
且 `state_recorded_seconds` 恒等于 `0.60 × total`。逐 cell 复核：
50/50 满足 `state_stop == mechanism_start`、时长守恒（<1e-6）、
比例误差 <1e-9，且 `total_recorded_seconds` 与 inventory 的
`validation_recorded_seconds` 逐位一致。

R1.7A 只在 D_state 计分（`time_lower/upper` 取自分层边界，
`d_mechanism_scored_here=False` 50/50）。

## 3. 优化器配置（逐字节取自 R1.6）

prefix/core：AdamW、lr 1e-3、weight_decay 0、warmup 0.1、clip 5、12 passes、chunk 128、
无短 patience。target alignment：AdamW、state/readout lr 3e-4、observer lr 3e-5、
weight_decay 0、warmup 0.1、clip 5、observer 8 + joint 8 passes、chunk 32。
每个 cell 记录 `frozen_r1_6_config_sha256 = f66f1758…9fabc`，与归档配置一致。
`epoch_zero_seen_inner_validation=False`（R1.5 不公平比较的修复），
`refit_mode=full_train`。

## 4. matched wrong-time 构造

`strict_matched_wrong_time_permutations` 以 **recorded coverage segment** 为供体组
（`donor_group_kind="recorded_coverage_segment"`、`same_recorded_coverage_segment=True`；
`same_session=False` 是互斥标志，表示未退回较松的 session 分组），
每个 anchor 取 5 个供体，最小时间间隔 1800 s，匹配特征为 10 维确定性历史坐标
加 raw-window contact coverage（均以 TRAIN 统计量标准化）。
候选不足 5 个的 anchor 记为未匹配并**排除出计分**。

## 5. 不确定度

患者优先：先取 5 seeds 中位数，再对 D_state 内的连续时间块做 bootstrap（2000 draws，
seed 1701）。块长由 `block_bootstrap_length_seconds` **仅从 TRAIN** 事件间隔导出
（100 × 中位间隔，截断到 [1800, 21600] s）。
每个 contrast 使用自己的有效事件权重：persistence 类用 `n_events`，
correct-vs-wrong 类用 `n_matched_events`；**无匹配事件的块被排除，不计为零效应**。
first_subset 与 continuation 各自独立 bootstrap。

## 6. T2-R2.0 设计

仅对 `patient_stable_state ≥3/5` 且该 seed 自身 `stable_checkpoint=True`、
且 D_mechanism 内至少 1 个同 coverage segment 完整 100-event block 的单元运行。
exposure 为 `x_e = exp(-1/100)·x_{e-1} + η_e`，**在每个 recorded segment 处重置**
（`resets_at_recorded_segment=True`），burn-in 100 events；
innovation 只用 TRAIN cross-fit。四臂 `no_edge / real_cumulative /
state_matched_placebo / current_event_only` 共享逐行相同的 support
（`current_index` 逐元素相等，否则抛 RuntimeError）；
`include_fitted_intercept_diagnostic=False` 且 `free_exposure_intercept_present`
必须为 False（双重锁）。observer / generator / history / decoder 全部冻结，
只拟合 signed low-rank B，并要求 B=0 处梯度非零。

## 6b. 数值结果

### 表 A 冻结队列与数据支持

| 患者 | 触点 | TRAIN 事件 | TRAIN 时长(h) | validation 事件 | validation 时长(h) | 覆盖段 |
|---|---:|---:|---:|---:|---:|---:|
| epilepsiae_1125 | 8 | 37702 | 57.5 | 8974 | 29.6 | 101 |
| yuquan_liyouran | 17 | 1406 | 9.4 | 469 | 12.3 | 1 |
| yuquan_zhangbichen | 52 | 4227 | 12.5 | 1472 | 4.0 | 2 |
| epilepsiae_253 | 8 | 36259 | 212.1 | 13704 | 18.2 | 149 |
| epilepsiae_1077 | 6 | 26197 | 130.4 | 8936 | 17.6 | 156 |
| epilepsiae_1073 | 6 | 103716 | 85.9 | 33800 | 35.7 | 127 |
| epilepsiae_1146 | 15 | 9906 | 34.4 | 5829 | 40.4 | 79 |
| yuquan_wangyiyang | 22 | 1150 | 7.1 | 384 | 2.0 | 1 |
| yuquan_xuxinyi | 15 | 5466 | 15.3 | 1333 | 2.6 | 3 |
| yuquan_zhaochenxi | 26 | 2287 | 16.7 | 763 | 3.3 | 1 |

### 表 B 五种子稳定性与 H1 区间（负值有利）

| 患者 | 稳定种子 | 非有限梯度 | 可评分种子 | 跨窗口留存 | 时刻专属 |
|---|---:|---:|---:|---|---|
| epilepsiae_1125 | 5/5 | 0 | 5 | -0.04214 [-0.06621, -0.02718] n=37 有利 | -0.08192 [-0.11698, -0.03869] n=35 有利 |
| yuquan_liyouran | 5/5 | 0 | 5 | -0.34374 [-0.60827, -0.21240] n=10 有利 | -0.05463 [-0.08540, +0.00179] n=10 跨零 |
| yuquan_zhangbichen | 5/5 | 0 | 5 | -0.10098 [-0.18348, -0.02292] n=5 有利 | -0.10173 [-0.20639, -0.01073] n=5 有利 |
| epilepsiae_253 | 4/5 | 0 | 5 | -0.00723 [-0.01500, +0.00169] n=24 跨零 | -0.00943 [-0.02068, -0.00045] n=21 有利 |
| epilepsiae_1077 | 1/5 | 4 | 1 | -0.06448 [-0.08616, -0.04433] n=24 有利 | -0.02569 [-0.04106, -0.01137] n=19 有利 |
| epilepsiae_1073 | 0/5 | 0 | 5 | 0 (退化区间) | 0 (退化区间) |
| epilepsiae_1146 | 0/5 | 0 | 5 | -0.00389 [-0.01241, +0.00376] n=25 跨零 | -0.00529 [-0.01792, +0.00679] n=20 跨零 |
| yuquan_wangyiyang | 0/5 | 0 | 5 | -0.11044 [-0.17736, -0.04914] n=3 有利 | +0.00146 [-0.00741, +0.00533] n=3 跨零 |
| yuquan_xuxinyi | 0/5 | 0 | 5 | -0.00456 [-0.01524, +0.00564] n=4 跨零 | +0.00190 [+0.00014, +0.00232] n=3 不利 |
| yuquan_zhaochenxi | 0/5 | 1 | 4 | +0.03672 [+0.00443, +0.07061] n=4 不利 | -0.00543 [-0.04512, +0.03235] n=4 跨零 |

### 表 C H2a 端点分解（患者优先中位数，括号内为有利种子数）

| 患者 | 留存·最先那组 | 留存·后续扩散 | 时刻·最先那组 | 时刻·后续扩散 |
|---|---|---|---|---|
| epilepsiae_1125 | +0.00542 (0/5) | -0.01452 (5/5) | +0.00287 (0/5) | -0.02009 (5/5) |
| yuquan_liyouran | -0.04824 (5/5) | -0.27800 (5/5) | -0.00630 (5/5) | -0.05062 (5/5) |
| yuquan_zhangbichen | -0.00370 (5/5) | -0.09531 (5/5) | -0.00312 (5/5) | -0.08413 (5/5) |
| epilepsiae_253 | -0.00401 (3/5) | +0.00585 (1/5) | -0.00076 (4/5) | -0.00369 (5/5) |
| epilepsiae_1077 | -0.01332 (1/5) | -0.00996 (1/5) | -0.00253 (1/5) | -0.00236 (1/5) |
| epilepsiae_1073 | +0.00000 (0/5) | +0.00000 (0/5) | +0.00000 (0/5) | +0.00000 (0/5) |
| epilepsiae_1146 | +0.00088 (0/5) | -0.00851 (5/5) | +0.00198 (0/5) | +0.00012 (2/5) |
| yuquan_wangyiyang | +0.01021 (0/5) | -0.11314 (5/5) | +0.00087 (0/5) | +0.00062 (0/5) |
| yuquan_xuxinyi | -0.00601 (5/5) | -0.00076 (4/5) | -0.00039 (5/5) | +0.00244 (0/5) |
| yuquan_zhaochenxi | +0.00133 (1/5) | +0.03914 (0/5) | +0.00319 (0/5) | -0.00692 (4/5) |

### 表 D H2a 端点区间

| 患者 | 留存·最先那组 | 留存·后续扩散 | 时刻·最先那组 | 时刻·后续扩散 |
|---|---|---|---|---|
| epilepsiae_1125 | +0.00592 [+0.00352, +0.00816] n=37 不利 | -0.01448 [-0.01963, -0.00946] n=37 有利 | +0.00299 [-0.00005, +0.00576] n=35 跨零 | -0.01921 [-0.02636, -0.00963] n=35 有利 |
| yuquan_liyouran | -0.04759 [-0.05901, -0.03373] n=10 有利 | -0.28179 [-0.52038, -0.16043] n=10 有利 | -0.00537 [-0.01531, +0.00668] n=10 跨零 | -0.04864 [-0.09054, +0.00621] n=10 跨零 |
| yuquan_zhangbichen | -0.00372 [-0.00728, -0.00081] n=5 有利 | -0.09575 [-0.17740, -0.02093] n=5 有利 | -0.00321 [-0.00796, +0.00039] n=5 跨零 | -0.09705 [-0.19838, -0.00679] n=5 有利 |
| epilepsiae_253 | -0.00158 [-0.00381, +0.00088] n=24 跨零 | +0.00449 [-0.00252, +0.01176] n=24 跨零 | -0.00067 [-0.00370, +0.00233] n=21 跨零 | -0.00435 [-0.01384, +0.00467] n=21 跨零 |
| epilepsiae_1077 | -0.01332 [-0.01724, -0.00892] n=24 有利 | -0.00996 [-0.01468, -0.00484] n=24 有利 | -0.00267 [-0.00546, +0.00004] n=19 跨零 | -0.00244 [-0.01005, +0.00443] n=19 跨零 |
| epilepsiae_1073 | 0 (退化区间) | 0 (退化区间) | 0 (退化区间) | 0 (退化区间) |
| epilepsiae_1146 | +0.00086 [-0.00162, +0.00377] n=25 跨零 | -0.00761 [-0.01371, -0.00203] n=25 有利 | +0.00244 [-0.00031, +0.00512] n=20 跨零 | -0.00061 [-0.00836, +0.00606] n=20 跨零 |
| yuquan_wangyiyang | +0.01026 [-0.00040, +0.01883] n=3 跨零 | -0.11311 [-0.18836, -0.05235] n=3 有利 | +0.00088 [+0.00017, +0.00116] n=3 不利 | +0.00044 [-0.00845, +0.00472] n=3 跨零 |
| yuquan_xuxinyi | -0.00576 [-0.00773, -0.00325] n=4 有利 | -0.00070 [-0.00789, +0.01238] n=4 跨零 | -0.00030 [-0.00168, +0.00051] n=3 跨零 | +0.00222 [-0.00059, +0.00470] n=3 跨零 |
| yuquan_zhaochenxi | +0.00152 [-0.00337, +0.00743] n=4 跨零 | +0.03914 [+0.01989, +0.06101] n=4 不利 | +0.00226 [-0.00174, +0.00629] n=4 跨零 | -0.00574 [-0.04790, +0.03123] n=4 跨零 |

### 表 E T2 资格与支持度分层

| 患者 | 患者层稳定 | D_mechanism 事件 | 完整 100 事件块 | 支持度 | 进入 T2 |
|---|---|---:|---:|---|---|
| epilepsiae_1125 | True | 3829 | 33 | COHORT_ELIGIBLE | True |
| yuquan_liyouran | True | 252 | 2 | CASE_ONLY | True |
| yuquan_zhangbichen | True | 836 | 8 | COHORT_ELIGIBLE | True |
| epilepsiae_253 | True | 3660 | 34 | COHORT_ELIGIBLE | True |
| epilepsiae_1077 | False | 2957 | 25 | COHORT_ELIGIBLE | False |
| epilepsiae_1073 | False | 13141 | 125 | COHORT_ELIGIBLE | False |
| epilepsiae_1146 | False | 680 | 4 | CASE_ONLY | False |
| yuquan_wangyiyang | False | 136 | 1 | CASE_ONLY | False |
| yuquan_xuxinyi | False | 461 | 4 | CASE_ONLY | False |
| yuquan_zhaochenxi | False | 338 | 3 | CASE_ONLY | False |

### 表 F T2 四臂结果（next event，负值有利）

| 患者 | 暴露来源 | 种子 | 仪器有效 | 选中了边 | 胜三对照 | real−无边 | real−安慰剂 | real−仅当前 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| epilepsiae_1125 | load | 5 | 5 | 5 | 0 | +0.00292 | +0.00118 | +0.00424 |
| epilepsiae_1125 | participation | 5 | 5 | 5 | 4 | -0.00361 | -0.00674 | -0.00347 |
| epilepsiae_253 | load | 4 | 4 | 4 | 1 | +0.04109 | +0.03754 | +0.04109 |
| epilepsiae_253 | participation | 4 | 4 | 0 | 0 | +0.00000 | -0.00526 | -0.00951 |
| yuquan_liyouran | load | 5 | 5 | 5 | 1 | +0.00284 | -0.00365 | +0.00284 |
| yuquan_liyouran | participation | 5 | 5 | 0 | 0 | +0.00000 | +0.00000 | +0.00000 |
| yuquan_zhangbichen | load | 5 | 5 | 5 | 3 | -0.00088 | -0.04158 | -0.00088 |
| yuquan_zhangbichen | participation | 5 | 5 | 5 | 2 | +0.00330 | -0.00252 | +0.00529 |

### 表 G T2 逐种子明细

| 患者 | 来源 | 种子 | 选中轮次 | 零点梯度范数 | 胜三对照 | real−无边 | H5 | H10 |
|---|---|---:|---:|---:|---|---:|---|---|
| epilepsiae_1125 | load | 0 | 28 | 0.0962 | False | +0.00929 | False | False |
| epilepsiae_1125 | load | 1 | 16 | 0.1359 | False | +0.01295 | False | False |
| epilepsiae_1125 | load | 2 | 28 | 0.1194 | False | +0.00144 | False | False |
| epilepsiae_1125 | load | 3 | 24 | 0.0998 | False | +0.00292 | False | False |
| epilepsiae_1125 | load | 4 | 6 | 0.0826 | False | +0.00152 | True | True |
| epilepsiae_1125 | participation | 0 | 20 | 0.1904 | True | -0.00386 | False | False |
| epilepsiae_1125 | participation | 1 | 1 | 0.2253 | True | -0.00498 | True | True |
| epilepsiae_1125 | participation | 2 | 20 | 0.2377 | True | -0.00361 | False | False |
| epilepsiae_1125 | participation | 3 | 22 | 0.1634 | False | +0.00220 | False | False |
| epilepsiae_1125 | participation | 4 | 30 | 0.1656 | True | -0.00195 | False | False |
| epilepsiae_253 | load | 0 | 8 | 0.1226 | False | +0.06736 | False | False |
| epilepsiae_253 | load | 2 | 22 | 0.1228 | True | -0.06109 | False | False |
| epilepsiae_253 | load | 3 | 28 | 0.0960 | False | +0.03511 | False | False |
| epilepsiae_253 | load | 4 | 30 | 0.0862 | False | +0.04706 | False | False |
| epilepsiae_253 | participation | 0 | 0 | 0.5552 | False | +0.00000 | False | False |
| epilepsiae_253 | participation | 2 | 0 | 0.2506 | False | +0.00000 | False | False |
| epilepsiae_253 | participation | 3 | 0 | 0.3605 | False | +0.00000 | False | False |
| epilepsiae_253 | participation | 4 | 0 | 0.4687 | False | +0.00000 | False | False |
| yuquan_liyouran | load | 0 | 4 | 0.2123 | False | +0.01213 | True | True |
| yuquan_liyouran | load | 1 | 4 | 0.2276 | False | +0.00284 | True | True |
| yuquan_liyouran | load | 2 | 4 | 0.0779 | True | -0.00480 | False | False |
| yuquan_liyouran | load | 3 | 4 | 0.3269 | False | +0.02240 | False | False |
| yuquan_liyouran | load | 4 | 4 | 0.1774 | False | -0.00023 | True | True |
| yuquan_liyouran | participation | 0 | 0 | 0.5497 | False | +0.00000 | False | False |
| yuquan_liyouran | participation | 1 | 0 | 0.2768 | False | +0.00000 | False | False |
| yuquan_liyouran | participation | 2 | 0 | 0.4016 | False | +0.00000 | False | False |
| yuquan_liyouran | participation | 3 | 0 | 0.2587 | False | +0.00000 | False | False |
| yuquan_liyouran | participation | 4 | 0 | 0.2468 | False | +0.00000 | False | False |
| yuquan_zhangbichen | load | 0 | 14 | 0.0679 | True | -0.00088 | True | True |
| yuquan_zhangbichen | load | 1 | 22 | 0.2757 | False | +0.00902 | False | False |
| yuquan_zhangbichen | load | 2 | 18 | 0.1827 | False | +0.00397 | False | False |
| yuquan_zhangbichen | load | 3 | 20 | 0.1704 | True | -0.04307 | False | False |
| yuquan_zhangbichen | load | 4 | 12 | 0.1109 | True | -0.01528 | True | True |
| yuquan_zhangbichen | participation | 0 | 2 | 1.0617 | False | +0.00330 | False | False |
| yuquan_zhangbichen | participation | 1 | 18 | 0.7682 | False | +0.23248 | False | False |
| yuquan_zhangbichen | participation | 2 | 22 | 1.2900 | False | +0.09159 | False | False |
| yuquan_zhangbichen | participation | 3 | 20 | 1.1360 | True | -0.00252 | False | False |
| yuquan_zhangbichen | participation | 4 | 14 | 1.0377 | True | -0.09245 | False | False |

### 独立支持核对（承重限制）

- epilepsiae_1125/load: 事件平均基于 2904 行；互不重叠 100 事件块仅 33 个；**产物中无逐块对比**。
- epilepsiae_1125/participation: 事件平均基于 2904 行；互不重叠 100 事件块仅 33 个；**产物中无逐块对比**。
- epilepsiae_253/load: 事件平均基于 3159 行；互不重叠 100 事件块仅 34 个；**产物中无逐块对比**。
- epilepsiae_253/participation: 事件平均基于 3159 行；互不重叠 100 事件块仅 34 个；**产物中无逐块对比**。
- yuquan_liyouran/load: 事件平均基于 251 行；互不重叠 100 事件块仅 2 个；**产物中无逐块对比**。
- yuquan_liyouran/participation: 事件平均基于 251 行；互不重叠 100 事件块仅 2 个；**产物中无逐块对比**。
- yuquan_zhangbichen/load: 事件平均基于 835 行；互不重叠 100 事件块仅 8 个；**产物中无逐块对比**。
- yuquan_zhangbichen/participation: 事件平均基于 835 行；互不重叠 100 事件块仅 8 个；**产物中无逐块对比**。


## 7. 已知限制（承重）

### 7.1 T2 无独立时间块对比

T2 产物只有 D_mechanism 上的 next-event 事件平均，**没有任何逐块 contrast**。
由于 exposure 核 ρ=exp(−1/100)，相邻行共享约 100 次事件的历史，
行数远大于独立信息量。以 `epilepsiae_1125` 为例：事件平均基于 2904 行，
而互不重叠的 100-event block 只有 33 个。
R1.6 最小 H3 曾以 independent-block medians 为判据，本实现缺此项。
**因此任何 T2 阳性在补上逐块对比之前只能登记为候选。**

### 7.2 `real_edge_estimable` 混入拟合结果

`estimable` 由四个仪器条件（gradient_finite、gradient_at_zero_norm>1e-8、
exposure_rank==dim、min(exposure_sd)>1e-8）**与** `edge_left_zero_initialisation`
共同构成，后者是拟合结果（selected_epoch>0 才为真）。
聚合器只在 `real_edge_estimable=True` 的 seed 上计中位数与分母，
于是"在梯度健康的前提下主动判定无可用边"的 seed 被**移出分母而非记为阴性**，
方向上**低报阴性证据**。实例：`epilepsiae_253` / participation 4/4 seeds
零点梯度范数 0.2506、满秩、SD 正常，但全部 selected_epoch=0，
按实现记 0 个可估计 seed，正确读法是 4/4 次真实阴性。

### 7.3 H2a 端点退化

`same_prefix_continuation` 的限定条件是"该事件的 first tied group 在被评分集合中
至少出现两次"。本队列 contacts 为 6–52、scored events 以千计，
该条件对几乎所有事件成立，因此该端点与 `continuation` **逐位等值**
（所有患者 / 所有 seed，事件数相同、NLL 差 <1e-12）。
不得作为第三个独立端点计数。

### 7.4 仪器故障排除了一位可能有信息的患者

`epilepsiae_1077` 5 seeds 中 4 个触发非有限梯度守卫，仅 1 个可评分；
该单 seed 的两个区间均显著有利（−0.06448 / −0.02569），
但 1/5 达不到 3/5 门槛因而未进入 T2。这是**仪器原因**而非数据不足。

### 7.5 两个"支持"单元中的一个由安慰剂退化驱动

诊断量 `placebo − no_edge = (real−no_edge) − (real−placebo)`：若为正，说明
state-matched 安慰剂臂比"根本不加边"还差，此时"real 胜过 placebo"不构成暴露有效的证据。

| 单元 | real−no_edge | placebo−no_edge | 种子符号(负/正) | 独立块 |
|---|---:|---:|---|---:|
| `epilepsiae_1125` / participation | **−0.00361** | +0.00295 | 4 / 1 | 33 |
| `yuquan_zhangbichen` / load | −0.00088 | **+0.02857** | 3 / 2 | 8 |

`yuquan_zhangbichen`/load 相对零边基线只有 −0.00088（实质为零），五个种子符号不一致
（−0.04307 / +0.00902 / +0.00397 / −0.04307 / −0.01528），独立块仅 8 个，
其"支持"判定完全由 0.0286 的安慰剂劣化撑起。**不予采信。**

根因：`patient_source_support` 只要求三个中位数同时 <0，
**未要求 real 相对 no_edge 的差距达到任何实质量级**，
因此一个退化的安慰剂臂即可让判据通过。修法：把
"real 必须以非平凡幅度胜过 no_edge"单列为必要条件，
并把 `placebo − no_edge` 作为安慰剂健康度诊断常规报告。

另有两个单元（`epilepsiae_253`/participation、`yuquan_liyouran`/participation）
在零点梯度健康（0.25 / 满秩 / SD 正常）的前提下 4/4 与 5/5 seeds 主动判定无可用边，
`real − no_edge` 恒为 0.00000。按 §7.2 的更正读法，这是**真实阴性**，
不是"不可估计"。

## 8. 运行事件

- **非有限梯度 5 cells / 2 患者**（`epilepsiae_1077` 4、`yuquan_zhaochenxi` 1）：
  按 R1.6 先例显式记录为仪器失败，`analysis_status=NONFINITE_GRADIENT`、
  `stable_checkpoint=False`、保留在 5-seed 分母内。新增窄口径判据
  `is_nonfinite_gradient_failure`：仅精确匹配冻结优化器守卫的错误文本，
  其余错误（shape / 对齐 / checkpoint / 显存）仍直接中止。
- **T2 路径此前从未跑通**：`r1_7_t2.py:89` 调用 `_query_states()` 漏传必需
  keyword-only 参数 `state_permutation`（九个调用点中唯一一个）。
  按其余八处既定写法补 `state_permutation=None`，并新增 AST 静态回归测试
  扫描全部调用点。该 bug 存在于提交 b7eaf0a1，与本轮改动无关。
- **显存不足 3 cells / 1 患者**：`yuquan_zhangbichen` 有 52 contacts（队列最多），
  tied-group mark likelihood 的 elementary symmetric polynomial 随 contacts 增长，
  单 cell 峰值约 6 GB，3 并发耗尽 24 GB。按合同以**降低并发（12→3）**处理，
  未改 batch/chunk，因此数值不变；重跑后峰值 11.2 GB。
- **测试**：126 passed。另记录一项与本证据链无调用关系的脆弱性：
  已退役长尺度路线的 `test_constant_offset_is_absorbed_without_exposure_edge`
  对线程数敏感（OMP=1 通过 0.00015；OMP=2 得 0.0067、OMP=4 得 0.0037，超过 1e-3 判据），
  因该拟合 batch>n、每 epoch 仅一步、20 epochs 远未收敛（intercept 0.559 vs 真值 0.7）。
  项目所有 runner 固定单线程，未修改该测试。

## 9. 下一步

1. **给 T2 加逐块 contrast**（与 R1.7A H1 同款：按 TRAIN 决定长度切分 D_mechanism、
   逐块算 real−各对照、再做块 bootstrap），重跑 38 cells。
   这是 `epilepsiae_1125`/participation 能否升格的判决点。
2. **修 `real_edge_estimable`**：只保留四个仪器条件，把
   `edge_left_zero_initialisation` 移出闸门、单独作为一个报告字段。
3. **R1.7B 探索性扩展**：去掉每数据集前 5 位上限（10→17 位，其余判据不变），
   seeds 5→10，检验 4/10 这个复现率是否稳定。
