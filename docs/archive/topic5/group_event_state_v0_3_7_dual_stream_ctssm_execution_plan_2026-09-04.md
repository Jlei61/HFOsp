# Group-Event State v0.3.7：执行计划

**日期：** 2026-09-04  
**状态：** `PLAN_FROZEN_DO_NOT_RESTART_V035`

## 1. 执行原则

架构搜索只替换 state producer，不改变 H1/H2a/H2b/H3 的 target、anchor、对照和结论门槛。所有 v0.3.7 产物进入独立根；旧 v0.3.5/v0.3.6 结果只读。

推荐新根：`/data/hfosp_group_event_state_v0_3_7/`。

## 2. Work package 0：版本隔离与注册表

1. 固化上一阶段 closeout 和 HOLD；
2. 建立 `checkpoint_registry.json`，分别登记 event encoder/decoder、`B_rate`、`B_mark`、`S_event`、`S_dual`、`S_grid` 和 H3 `M0/M1/M2`；
3. 每个 entry 记录数据时间上限、输入流、是否使用 background、目标、seed、代码 commit 和 normalization provenance；
4. 旧 v0.3.6 `full_credit.py` 不直接升格为生产实现，只可复用无截断/检查点工具。

**完成条件：** 旧 supervisor 无活进程；新旧根互不写入；development/sealed 标志为 false。

## 3. Work package 1：锁定事件表示与成熟 decoder

1. 对 decoder 做嵌套时间泄漏审计：patient offset、contact normalization/vocabulary/order、calibration bias、event normalization、tied-group 统计、detector template 和 checkpoint selection；
2. 只接受完全早于 state TRAIN/INNER/SELECTION 的 decoder bundle；不合格患者按相同时间合同重训；
3. 预计算每个事件的 `u_N`、raw grammar embedding、community/repertoire posterior、waveform/band/delay summaries；
4. 用 TRAIN-only cross-fitting 产生 `u_G` 和 nuisance provenance；
5. 冻结 event representation。

**验证：** 原始事件到 embedding 可逐事件回溯；future mark 不进入 pre-event state；decoder 零调制 parity 逐位一致。

## 4. Work package 2：透明基线

实现并先跑：

- `B_rate`：已有动态 rate/history 基线；
- `B_mark`：对 burden 与 normalized grammar 的固定 10 min–16 h EWMA/pseudo-count；32 h 仅在留出跨度充分时作探索性敏感性；
- background-only 可审计基线。

每个 horizon 的 head 独立选择，但共享同一套 causal history features。输出 count 与 conditional grammar 两套分数，不合并成一个总 accuracy。

**目的：** 先确定长历史信息是否已能被透明统计解释；不把“简单模型赢”当项目失败。

## 5. Work package 3：CT-DSSM 仪器与实现

### 5.1 必做实现

- exact irregular-`dt` diagonal transition；
- event impulse input；
- background zero-order-hold input；
- associative scan 前向/反向；
- fixed stable time bank；
- `S_N/S_G/R_fast/Z_B/uncertainty` 独立字段；
- 小非零 state-readout 初始化。

### 5.2 仪器矩阵

在同一 E253 safe FIT slice 上比较：

1. 旧 gated RNN；
2. fixed mark-EWMA；
3. event-only diagonal CT-DSSM；
4. 5 min hierarchical SSM。

只测梯度到达、event-rate invariance、null-event invariance、72 h 稳定性、吞吐与可拟合性。合成 common-drive/count-feedback/mark-feedback/variable-rate/missingness 用于验证实现和 H3 假阳性控制，不用于判断人体可训练。

**停止条件：** causal audit 失败、event density 改变物理 `tau`、null event 擦除历史、72 h 非有限，必须修代码后重跑；性能不显著不是工程停止条件。

## 6. Work package 4：人体训练性与架构选择

在不读取 development/sealed 的固定 pilot 上，以相同 event embeddings、anchor、目标、seed、有效 optimizer steps 和 head capacity 比较 `B_mark/S_event/S_grid`。优化器真实搜索：encoder/state LR、weight decay、warmup、state/readout init、state width 小/中两档；不得只移植旧学习率。

pilot 按输入完整性、连续长窗供给和 decoder 可用性事前选择。优先核查 E253、E958、E1073、E1077、E1125、E818 的资格，但名单必须由不读取模型结果的 eligibility 表冻结；不能因为结果难看剔除。

所有种子都报告：第一步梯度、参数更新量、FIT/INNER 曲线、预算边界、状态范数和 horizon-specific 样本数。首步最优或末步最优都视作训练诊断，不自动判科学阴性。

## 7. Work package 5：H1 主实验

1. 冻结一个共享 producer；
2. 分别拟合 0.5/2/6/8 h heads，12 h 仅对可估子队列；
3. 在相同 anchor 上比较 `B_rate/B_mark/S_event/S_dual/S_grid`；
4. 运行 block-circular shifted state；
5. 分解 future burden 与 conditional grammar；
6. 报告患者内效应及患者级汇总。

核心图：横轴为物理 horizon，纵轴为相对 `B_mark` 的 held-out log-score；分别画 count 与 conditional grammar，并叠加 correct 与 shifted state。图接口在任何显著性出现前就固定。

## 8. Work package 6：background 双流

每 5 分钟产生 causal non-event background frame。按相同 producer/head 预算比较：

- background-only；
- event-only；
- dual-stream；
- dual-stream 去掉 event-history；
- dual-stream 去掉 background correction。

重点不是“背景越多越好”，而是拆分 common drive 与 event-history 增量。raw background encoder 只在可审计特征版训练稳定后进入容量敏感性。

## 9. Work package 7：H2a 冻结 contact decoder

1. 添加低秩 contextual modulation，作用于 decoder 每一步的 hidden gain/offset、continue/STOP、size 与 contact logits；
2. decoder 主体冻结，先做 `b=0` parity 和 leaked-future oracle；
3. H2a primary 同时冻结 state producer 与 decoder 主体，只训练 state-to-context projection 和低秩 modulation adapter；
4. 比较 prefix-only、`B_mark`、correct `S_obs`、shifted `S_obs`；
5. 主报 same-prefix continuation、STOP/extent、subset、route/order 和 conditional bands；
6. 做 empirical-support interpolation 和 local Jacobian 分析。

另跑一条清楚标记的 interictal-only joint sensitivity：decoder 主体仍冻结，但允许 H2a loss 回传更新 state producer。该 checkpoint 不覆盖 primary registry，且必须重新运行 H1，防止把针对单事件形态重新训练后的状态冒充原本的跨 horizon 状态。

H2a 不以“一次事件是否预测准确”笼统汇总，必须指出状态到底改变了终止、范围、路径还是频带。

## 10. Work package 8：H2b 冻结跨任务

producer、decoder、time bank 与 background features 全部冻结后，才读取 seizure labels：

- fixed-grid discrete survival hazard；
- lead-time-to-seizure；
- early ictal field/path at 6 h、2 h、30 min、5 min；
- baseline 加入 time since previous seizure、postictal/cluster、clock、sleep 和临床干预。

checkpoint registry 中所有预先登记 producer 都进入 H2b，不按 H1 结果筛选。输出以患者和 held-out seizure 为分母；无足够事件或发作为不可估。

## 11. Work package 9：H3 独立生成线

另建命名空间和结果根，禁止复用 observer update 作为 jump：

1. 同构合成验证 M0/M1/M2 的正恢复与无反馈假阳性；
2. 人体先做可估性表和独立 exposure/future blocks；
3. 比较 common-drive M0、count-feedback M1、mark-feedback M2；
4. 所有臂带相同截距、滚动慢水平、background/context 与状态容量；
5. mark replacement 保持事件时刻与数量，count experiment 不匹配掉 count；
6. 报告 signed event-type impulse response 和 held-out future log-score。

H3 不占用早于 H1/H2a/H2b 的首轮主算力，也不因 observer state 阳性自动放行结论。

## 12. 计算与队列策略

- smoke 与正式队列使用不同根；
- instrument checks 通过后，以独立患者/seed 作为 worker 单元并行；
- 先实测显存，再把单卡并发提高到稳定利用率；OOM 只允许降低同卡并发或启用 checkpointing，不静默改变科学 batch、窗口或目标；
- worker 写原子 card，supervisor 只调度/汇总，不拥有训练进程；
- 每个 cell 可恢复，失败、非有限和结构零均保留在分母与审计表。

## 13. 首轮交付物

1. v0.3.7 machine contract 与 checkpoint registry；
2. event representation/decoder 泄漏审计；
3. `B_mark` 与 CT-DSSM instrument report；
4. H1 跨 horizon 主图；
5. H2a same-prefix/context modulation 主图；
6. H2b hazard + early ictal field lead-time 图；
7. H3 M0/M1/M2 图；
8. 白话报告、技术报告、per-patient table、figure metadata 和 `figures/README.md`。

## 14. 当前不启动的内容

- 不重启 v0.3.5 shared queue；
- 不把 v0.3.6 smoke 续成正式人体队列；
- 不直接上 Mamba、attention、input-dependent transition；
- 不打开 development/sealed；
- 不用 H1 漂亮患者决定 H2b/H3 分母。
