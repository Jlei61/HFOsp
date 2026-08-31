# H2b Cross-task Transfer v0.3 执行计划

## 0. 交付目标

本计划只做一件事：判断冻结的间期状态能否跨任务连接到发作风险与发作入口组织。执行顺序固定为：

```text
冻结合同与分母
    ↓
间期 state instrument 资格
    ↓
半合成灵敏度和假阳性校准
    ↓
完整时间轴 nested hazard + prequential OOF
    ↓
tau_z lag-response + OOS 流形—流场
    ↓
有信号后才做 IED-source ablation 和 ictal organisation
```

输出根：

```text
results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_3/
```

v0.2 结果只读；formal/sealed/H3/T2/paper-ready figures 全程不触碰。

## 1. 运行纪律

- 所有任务使用原子 manifest、唯一 task key、heartbeat、dead-owner reclaim 和输入/输出 SHA256；
- 长任务用 `nohup` 或独立 `tmux`/`setsid`，网络断开后继续；
- 每个 worker 固定 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`，避免 CPU 线程乘法；
- 启动并发前实测单 worker RSS/VRAM，以 `floor(0.8 * 可用资源 / 峰值)` 决定 worker 数；
- GPU 作业按患者串行占卡、跨卡并行；OOM 时只降低 batch/启用 checkpointing 并从原子状态续跑，不改变科学配置；
- seed 是优化重复，不是科学样本；至少三 seed 用于 state qualification，最终先在患者内汇总；
- 任何结果表都同时带 `n_patients`、`n_oof_lead_seizures`、`n_segments`、`n_seeds`，不得只报 fit 数。

建议目录：

```text
v0_3/
├── analysis_contract.json
├── manifests/
├── qualification/
├── assay/
├── hazard/
├── lag_response/
├── geometry/
├── source_ablation/
├── phenotype/
├── reports/
└── logs/
```

## 2. Phase A0：合同、设计矩阵与 attrition

### 实现

1. 运行 `scripts/topic5_continuous_marked_state_h2b/freeze_v03_contract.py`，将机器合同连同 SHA256 写入结果根；
2. 建立只读 source inventory：17 位患者、85 cells、75 readable checkpoints 起步，逐项重算 checkpoint/result/config hash；
3. 从原始 coverage 与 seizure crosswalk 重建 5 min anchors，禁止复用未经核实的 provisional eligible 标签；
4. 输出从 85 total 到 contrast-specific evaluable 的 attrition 表；
5. 输出每个设计矩阵的真实列清单和维度，机器断言 `M2=M1+ZP`、`M4=M3+R`；
6. 将 v0.2 的 `primary_chronological` 旧标签只保留作兼容字段，新支持等级根据实际 OOF 发作与 assay power 生成。

### 产物

- `analysis_contract.json`
- `manifests/source_inventory.json`
- `manifests/attrition_by_reason.csv`
- `manifests/seizure_crosswalk.csv`
- `manifests/design_matrix_receipt.json`
- `reports/scientific_route_audit_A0.json`

### 验收

- seizure labels 没有上游梯度路径；
- 所有缺失都有互斥原因；
- M1/M2、M3/M4 使用完全相同 anchors、outcomes 和 outer folds；
- 同一患者不跨 coverage gap carry state；
- 设计矩阵未通过时不拟合任何 outcome model。

## 3. Phase A1：只用间期数据验 state instrument

### 实现模块

建议新增：

- `src/topic5_continuous_marked_state_h2b/v03_state_qualification.py`
- `scripts/topic5_continuous_marked_state_h2b/run_v03_state_qualification.py`
- `tests/topic5_continuous_marked_state_h2b/test_v03_state_qualification.py`

对每个 readable checkpoint/seed 计算：

1. Q1：decoder-standardised effective rank、方差谱、gap/reset 解释度、collapsed/shuffled null；
2. Q2：相同当前窗口下 persistent 与 memoryless 对 held-out future IED timing/mark 的差；
3. Q3：generator drift、observation correction、correction/generator 比、open-loop horizon、reset recovery；
4. Q4：decoder metric autocorrelation/variogram 的 `tau_z` 与区间删失状态；
5. Q5：跨 seed decoder-output CKA、pairwise-distance correlation、Procrustes-aligned trajectory；
6. Q6：加入所有真实可用 clock/sleep/day/last-seizure/segment 后的 interictal increment。

### 运行顺序

先在 `epilepsiae_548`、一个 H1-stable 患者、一个 H1-unstable 患者上做三患者 smoke。检查数值尺度与内存后，再并行扩到全部 75 readable cells。这里不训练新 state，只做冻结模型 forward/诊断，优先用 CPU workers；确需 GPU decoder forward 时按卡分患者。

### 产物

- `qualification/per_checkpoint_metrics.parquet`
- `qualification/per_patient_seed_summary.csv`
- `qualification/state_qualified_manifest.json`
- `qualification/all_frozen_manifest.json`
- `qualification/tau_z_by_patient_seed.csv`
- `reports/scientific_route_audit_A1.json`

### 分流

- 有 state-qualified 患者：进入 A2；
- state-qualified 为空：当前 R1.7B 的 persistent-state 分支收口，进入 R1.8 slow/fast instrument redesign；可保留 all-frozen 表示诊断，但不继续给它增加 seizure heads；
- 只有 Q2/Q3 失败：优先判断 observer overwrite，不把它写成生理无慢状态。

这不是对整个项目的 gate，只是禁止无资格 instrument 承担 persistent-state claim。

## 4. Phase A2：半合成 assay 与 power

### 实现模块

- `src/topic5_continuous_marked_state_h2b/v03_assay.py`
- `scripts/topic5_continuous_marked_state_h2b/run_v03_assay.py`
- `tests/topic5_continuous_marked_state_h2b/test_v03_assay.py`

### 两阶段运行

1. 7 个 world × 100 replicates：只用于抓方向、泄漏、符号和运行时间错误；
2. 修复完成并冻结所有自由度后，7 个 world × 1000 replicates：唯一可用于 type-I/power 验收的批次。

每个 replicate 有固定 seed 与独立 task key；使用向量化 CPU 批次并行。不得因 smoke 结果选择更有利的真实患者、horizon 或 probe。

需要一次性冻结：

- prequential 初始 `K∈{2,3,4,5}`；
- ridge grid；
- residualisation 宽度和惩罚；
- GAM 自由度；
- basin/approach/jump 的统计量；
- 最小相关效应 5%。

### 产物

- `assay/frozen_assay_config.json`
- `assay/replicate_manifest.jsonl`
- `assay/type1_power_summary.json`
- `assay/mechanism_recovery.csv`
- `reports/scientific_route_audit_A2.json`

### 验收

- null 与 observation-only 不产生 state 增量；
- clock-confounded 在调整后回零；
- persistent world 恢复 T、M 和 lag degradation；
- 三类 geometry world 能被各自统计量区分；
- 若 power 不足，记录 `ASSAY_NOT_SENSITIVE`，先修测量工具，不运行并解释真实阴性。

## 5. Phase A3–A4：完整时间轴 nested hazard 与 prequential OOF

### 实现模块

- `src/topic5_continuous_marked_state_h2b/v03_hazard.py`
- `scripts/topic5_continuous_marked_state_h2b/run_v03_hazard.py`
- `tests/topic5_continuous_marked_state_h2b/test_v03_hazard.py`

### 数据构造

1. 在每个 recorded segment 上建立 5 min grid；
2. 每个 anchor 只看过去；ictal 和 120 min postictal 排除；
3. future 30 min 为主 outcome，5/15/60 min 由同一 fitted hazard 导出；
4. `O` 中的 learned code 必须在每个 anchor 重置 persistent history 后计算；
5. `R` 的 residual model 只在 outer-training past folds 拟合；
6. 按 frozen K 做 rolling-origin，预测每一个后续 lead seizure。

### 比较

- T：M2 vs M1；
- M：M4 vs M3；
- 基线分解：M1 vs M0 只说明 current observation 有用，不算 state；
- v0.2 exact risk-set 在同一 OOF seizure 上重算，只作 bridge。

### 产物

- `hazard/full_grid_anchor_manifest.parquet`
- `hazard/prequential_fold_manifest.json`
- `hazard/oof_predictions.parquet`
- `hazard/per_seizure_contrasts.csv`
- `hazard/per_patient_contrasts.csv`
- `hazard/cohort_inference.json`
- `reports/scientific_route_audit_A3_A4.json`

### 验收与分流

- powered T−：关闭当前 R1.7B transfer branch，不增加复杂 probe；
- T+、M−：后续只按 transferable representation 命名；
- T+、M+：进入时刻/几何检验；
- all-frozen 与 state-qualified 必须并排，不能用更漂亮的一层替换另一层。

## 6. Phase A5：`tau_z` lag-response

### 实现

对每个 OOF anchor 固定 `C/H/O/outcome`，只替换 `ZP` 为当前、0.5、1、2、4 倍 `tau_z` 之前的状态。每个 anchor 使用多个匹配 donor；未来状态另表作 acausal falsification。

若 `tau_z` 区间删失或近似常数，该患者不进入时间专属性主量，但保留在 T/M 分母。不得用任意固定 30 min shift 替代患者自身时间尺度。

### 产物

- `lag_response/donor_manifest.parquet`
- `lag_response/per_patient_lag_curve.csv`
- `lag_response/cohort_lag_summary.json`
- `reports/scientific_route_audit_A5.json`

### 解释

- 当前/近 lag 最好，超过 `tau_z` 后衰减：支持局部时间专属性；
- 所有 lag 相同：宽 basin 或非时间专属 representation；
- 只有 5 min 有利且 M 不成立：acute preictal encoder；
- 不要求慢状态逐分钟唯一。

## 7. Phase A6：OOS 流形—流场

### 实现模块

- `src/topic5_continuous_marked_state_h2b/v03_geometry.py`
- `scripts/topic5_continuous_marked_state_h2b/run_v03_geometry.py`
- `tests/topic5_continuous_marked_state_h2b/test_v03_geometry.py`

每个 outer fold：

1. 只用 training fold clean interictal 连续轨迹拟合 scaling、decoder metric 和邻接图；
2. training seizures 可在该 fold 内定义 entry region/direction；
3. held-out preictal/ictal 只 OOS project；
4. 分别计算 basin occupancy/dwell、geodesic approach/flow alignment、off-manifold displacement；
5. 用 clock/sleep/day 匹配的非发作轨迹做 null；
6. 只有满足合同条件才运行 MARBLE；UMAP 图不进入统计表。

### 产物

- `geometry/fold_geometry_receipts.jsonl`
- `geometry/oos_trajectory_metrics.parquet`
- `geometry/basin_results.csv`
- `geometry/approach_results.csv`
- `geometry/abrupt_exit_results.csv`
- `geometry/figures/README.md` 与诊断图（只有实际生成图后写 README）
- `reports/scientific_route_audit_A6.json`

### 验收

必须证明 scaler、metric、graph、entry region 和方向均未见 held-out seizure。D 只能与 T/M 合并升级 claim，不能单独把彩色轨迹称为 H2b。

## 8. Phase A7：IED-source ablation，条件解锁

仅在稳定 T 且至少一个 M/D 分量存在时启动。复用同一 interictal split，训练：

- full background + IED objective；
- background-only；
- IED-shuffled。

三臂同架构、同 state width、同 optimizer budget、同 seed、同 qualification。GPU 前先做单患者峰值 VRAM 测量，再决定并发；不得因为某臂 OOM 而给它更少的有效 optimizer steps。

输出三臂 checkpoint hash、qualification、T/M/D 对比和 paired patient effects。只有 full 稳定胜两个对照，才升级为 IED-specific transfer。

## 9. Phase A8：ictal organisation，条件解锁

先在完全不读取 state 的脚本中冻结连续 target：IED–ictal reuse、early extent/entropy、speed、laterality。禁止按 state 结果重聚类。

对每个 held-out seizure，检验 pre-event state/IED decoder 与该次 ictal recruitment 的相似性是否随接近 onset 增强，并胜过匹配非发作轨迹。稀疏 `broad_ER/gamma_ER` 只作旧结果桥接，不再承担主 endpoint。

## 10. 每阶段强制路线审计

每个 `scientific_route_audit_A*.json` 必须回答机器合同中的七问，并额外记录：

- 当前模块直接回答 T、M、D 或 organisational continuity 中哪一项；
- 该模块若阳性，最强允许表述是什么；
- 该模块若阴性，是生物学阴性、instrument 不合格还是样本不可估计；
- 是否新引入了 seizure supervision、非嵌套 comparator、raw-latent geometry 或伪分母；
- 是否正在用更多模型复杂度替代一个未通过的承重测量。

出现 `SCIENTIFIC_ROUTE_DRIFT` 时，停止相应 claim branch，修复后从最近的有效 receipt 继续；不删除其他预注册探索结果，也不把单一失败变成整个项目的总 gate。

## 11. 最终报告和完成定义

必须生成两个版本：

1. `reports/h2b_v0_3_plain.md`：白话说明问了什么、哪些患者/发作真正可测、T/M/D 各到哪一级、最可能的替代解释；
2. `reports/h2b_v0_3_technical.md`：合同 hash、attrition、design columns、folds、所有估计量、null/power、patient-first inference、route audits、复现命令。

完成不等于阳性。v0.3 的完成条件是：

- A0–A2 测量链闭合；
- 对实际可估计人群完成 A3–A6，或按合同明确在 A1/A2/T 处收口；
- 任何未解锁的 A7/A8 明确记为 `NOT_TRIGGERED`，不是 missing；
- 结论严格落入 claim ladder；
- formal/sealed/H3/T2 仍为 false；
- 代码、合同、报告提交，结果 manifests 可复现且长作业无孤儿进程。

## 12. 第一批最小工作包

下一位 agent 应先完成而不是直接跑大队列：

1. 冻结 `analysis_contract.json`；
2. 实现 A0 design/attrition receipt；
3. 实现 A1 Q1–Q6 和三患者 smoke；
4. 给出 `all_frozen` 与 `state_qualified` 的真实分母；
5. 实现 A2 的 7-world × 100 smoke；
6. 只有 smoke 方向正确后，冻结 assay 并扩 1000 replicates；
7. 在此之前不读取真实 v0.3 seizure outcome performance。

这批工作包结束时，先回答一个白话问题：

> 我们现在拿去预测发作的量，究竟是会跨窗口延续的状态，还是把当前背景重新编码了一遍？

若这一步没有答案，后面的 hazard、流形和发作亚型都不应被包装成原始科学目标的推进。
