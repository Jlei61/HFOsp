# H2b Cross-task Transfer v0.2 技术收口

## 1. Scientific estimand and boundary

Primary estimand：held-out 30-min conditional risk-set log loss 的 `B_state - B_observation`，负值有利。必要解释量为 `persistent - memoryless` 和 `correct-time - matched-wrong-time`。

R1.7B interictal timing+exact-mark checkpoints 只读接入；consumer-side audit 对 85 cells 与 75 个可读 checkpoint/result 重算哈希并通过，但 R1.7B 本身不满足旧 v0.1 的 50-fit formal release gate，故仅作为 exploratory development source。observer、state update、generator、timing/mark decoder 全冻结；seizure loss 仅训练患者内低容量 ridge conditional-risk probe。H1 stability 只作事前分层，不作纳入 gate。formal/sealed、H3/T2、paper-ready figures 全程未触碰。

## 2. Cohort and denominators

|层级|患者数|规则|
|---|---:|---|
|primary chronological|1|30 min 合格发作 ≥10|
|nested LOSO sensitivity|3|5–9|
|descriptive case series|5|2–4|
|not estimable|1|<2|

上游 17 位、85 cells 中 75 checkpoints 可读；13/17 位有 development seizure inventory，10 位完成 raw/background 支持审查，46 个 checkpoint-seed 状态缓存完成。crosswalk 为 124 条 development 发作，绝对时间为 float64，状态按 recorded coverage segment 重置，不跨 gap carry。

主风险表严格固定 5/15/30/60/120 min；30 min 合格发作 ID 固定所有 sensitivity lead 的患者内人群。case/control 同患者、同 coverage segment；ictal 与 120 min postictal 排除。患者 feature width 不同，未跨患者拼接矩阵。

## 3. Main results

### 3.1 Primary chronological

`epilepsiae_548`：10 次 30-min eligible seizures，但最终 TEST 只有 2 risk sets。

|contrast|effect|interpretation|
|---|---:|---|
|state - observation|-0.0212337|置换 95% 零带 [-0.0861538,+0.0500968] 内|
|persistent - memoryless|+0.00709455|不支持 persistent carry|
|correct - wrong|+0.201607|wrong-time 子集降为 LOSO；方向不利|

100 次 time-label permutation 完整，positive synthetic PASS。主层没有建立 H2b。

### 3.2 LOSO and descriptive

30 min 全 checkpoint-available 分层：

|tier|contrast|n|favourable|median|sign p|
|---|---|---:|---:|---:|---:|
|LOSO|state - observation|3|2/3|-0.00108777|1.0|
|LOSO|persistent - memoryless|3|2/3|-0.0490280|1.0|
|LOSO|correct - wrong|1|0/1|+0.201607|1.0|
|descriptive|state - observation|4|4/4|-0.121453|0.125|
|descriptive|persistent - memoryless|4|3/4|-0.109097|0.625|
|descriptive|correct - wrong|4|1/4|+0.138702|0.625|

描述层的 4/4 是探索性方向信号；样本太少、无时刻专属性，不能替代 primary。H1-stable/unstable 分层已在正式报告中逐项给出，但单元分母更小，不作独立 cohort claim。

60-min wrong-time sensitivity 中 `epilepsiae_548` 出现 +2332.72 的极端外推损失；原始 CSV 保留该值，报告标记为数值不稳定并不作解释。

## 4. Explicit non-estimability

早期实现会把 30-min 主效应不存在的患者仍标作 permutation `COMPLETE`，或在 wrong-time 只有一次发作时触发 `KeyError`。修复后：

- `epilepsiae_1073/primary` → `NOT_ESTIMABLE_AT_PRIMARY_LEAD`；
- `epilepsiae_1125/matched_wrong_time` → 同一显式状态；
- 两者 `n_permutations_run=0`、`n_finite_permutations=0`、observed/null summaries 为 null/空；
- 工程完成与科学可估计性分别记录，普通不可估计不阻断其他患者。

全队列 15 个 probe analyses：13 个 `COMPLETE`，2 个 `NOT_ESTIMABLE_AT_PRIMARY_LEAD`。

`epilepsiae_1125/primary` 的完整 100 次置换作业运行约 10 h 13 min：`state-observation=-0.00108777`，零带 `[-0.0576025,+0.0231829]`，`persistent-memoryless=-0.0490280`；8 次发作的 LOSO 结果不支持可靠跨任务增量。

## 5. Secondary frozen phenotype

冻结 target join 得到 185 条 seed-expanded available rows；6 位患者执行，18 个患者×目标单元中仅 2 个有限：

- `epilepsiae_253/frozen_subtype_broad_ER`：descriptive，2 held-out，effect -0.166275；
- `epilepsiae_548/frozen_subtype_broad_ER`：primary，2 held-out，effect -0.459209。

其余 16/18 为 missing target、class support 不足或低样本不可估计。early-recruitment scalar 未在本轮派生；target 未重聚类。phenotype 使用 baseline/observation/state 三臂，wrong-time donor availability 明确不作 phenotype gate。

## 6. Engineering corrections during closeout

1. probe task 增加原子 claim、heartbeat、dead-owner reclaim、live foreign-writer detection、输入/输出 SHA receipt 和断点续跑；
2. 旧长作业完成后才修改 source，保证产物 source SHA 真实；
3. patient-first aggregation 按 patient+lead 对齐 wrong-time，并保留 donor-valid 子集自己的降级 tier；
4. secondary phenotype 修复错误的 wrong-time 必需输入；
5. 主报告将全 checkpoint-available 与 H1 strata 分开，避免无 stratum 标签的重复行；
6. handoff 明确 17 位总清单、16 位 checkpoint-available subjects，并将已退出的 runtime monitor 标为 inactive。

## 7. Acceptance evidence

- H2b tests：112 passed，5 个既有 Transformer warnings；
- final machine audit：`PASS_COMPLETE`；
- R1.7B consumer acceptance audit：`PASS_EXPLORATORY_DEVELOPMENT_SOURCE`，且显式记录 formal v0.1 gate 未满足；
- input manifests：10；state cache cells：46；patient-first rows：34；
- phenotype patients run：6；estimable patient-target cells：2；
- formal/sealed/H3/T2/paper-ready flags：全部 false；
- queue 使用 setsid/nohup、CPU thread limit、patient-separated workers、原子 manifests 和 checkpoint/source hashes。

## 8. Safe claim

当前结果是 development closeout，结论为 **H2b not established**。它既不是跨任务阳性，也不是“生理状态不存在”的证明。后续若继续，应优先增加具有足够 chronological held-out seizures 的新患者，而不是在当前低分母上继续扩展 lead、亚型或模型复杂度。

## 9. Reproduction anchors

- `results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_2/COHORT_RUN_COMPLETE.json`
- `.../reports/machine_audit.json`
- `.../reports/per_patient_lead_results.csv`
- `.../reports/cohort_patient_first_summary.csv`
- `.../reports/phenotype_target_availability.json`
- `.../fits/by_subject/<subject>/<analysis>/risk_probe_machine_audit.json`
