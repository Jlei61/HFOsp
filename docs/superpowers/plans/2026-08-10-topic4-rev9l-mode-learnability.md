# Topic 4 rev9-L mode learnability 执行计划

**Spec：** `docs/superpowers/specs/2026-08-10-topic4-rev9l-mode-learnability-design.md`
**状态：** L0 完成；L1 packet 已冻结，formal fit 待启动
**分支：** `codex/topic4-rev9l-mode-learnability`
**结果目录：** `results/topic4_sef_hfo/data_driven_core_field_rev9_learnability/`

## 执行纪律

- 本线是 development/oracle audit，不读旧 patient held-out 分数，不产生 patient-generalization claim。
- 指标低、支持不足、runaway 和 network variability 是结果；只把合同/hash/shape/非有限实现错误作为硬停止。
- fit、selection、confirmation network seeds 不混用；confirmation 不参与 objective、source 或 optimizer 选择。
- producer 先提交再运行；长任务必须 `systemd-run --user -> nohup`，写 log/status 并通知。
- 所有图目录在图生成后写中文 `README.md`。

## Task L0：objective replay

- [x] 锁定 rev8.1 checkpoint、selection、final arrays、patient reference/target 和 rev9 factorial hashes。
- [x] 对 48 个 fit candidates 重放保留的 global/mode/support 指标，计算 `D_A/D_B/D_weak` 和 Pareto front。
- [x] 单独重放 3 个 selection candidates，不把 fit-only candidate 冒充 selection miss。
- [x] 计算旧 objective 与 `D_A/D_B/D_weak` 的 Spearman 相关。
- [x] 仅对有逐事件数组的 final/rev9 arms 计算完整 descriptor；历史缺失标为 `not_retained`。
- [x] 仅用 patient training blocks 计算 mode A/B block-to-complement reliability、mode proportion 和 dispersion。
- [x] 输出 CSV/JSON、Pareto/objective 图、patient reliability 图和唯一根 `decision.json`。

**L0 结果：** 48 个 fit candidates 中 24 个 support eligible。旧 `joint_loss` 与 mode A loss 的 Spearman
`rho=-0.082, p=0.704`，与 mode B 和 smooth weakest-mode loss 分别为 `rho=0.775/0.840`；fit 库和三个 selection-evaluated
candidates 均没有同时改善 A 且不损 B 的 dominator。因此判定为 `OBJECTIVE_DOES_NOT_PROTECT_MODE_A`，不是已经存在双优候选却
漏选。patient-training block-to-complement reliability 的中位数为 A `0.964`、B `0.984`；A 的 5% 分位更低（`0.772` vs
`0.896`），说明 A 更异质，但 target 仍有足够稳定性进入 forced-capacity assay。旧 patient held-out 未计算分数或 prototype。

L0 解释规则：

- selection-evaluated candidate 同时改善 A 且不损 B，才标记 `OLD_OBJECTIVE_OR_SELECTION_MISS`；
- 旧 objective 与 A 低相关但无 Pareto dominator，标记 `OBJECTIVE_DOES_NOT_PROTECT_MODE_A`；
- patient A block reliability 明显低于 B 时，记录 `PATIENT_MODE_A_TARGET_HETEROGENEOUS`，但不据此停止 L1。

## Task L1：forced-initiation capacity

- [x] 冻结独立 L1 config；component 按 raw Gaussian contribution 选源，control 按冻结中心最近邻选同数 E cells。
- [x] 冻结 primary paired-excess rank window `[100,250] ms`；returned triggered event 只作 secondary，后发自发事件不计。
- [x] 实现 deterministic E-neuron forced-spike adapter 和 sham；单测锁定 spike 时刻、cell identity 和无注入 no-op。
- [x] 实现 one-arm/one-network worker：一次建网和 sham 后复用全部 source/packet pairs，并保存 paired-excess rank/geometry。
- [x] 实现三网络 bounded canary launcher、120 s wait、status/log/notify 和 packet-selection aggregator。
- [x] 冻结 primary mapping A<-component2、B<-component1，以及 component1/2/3 与三个 off-field source 的确定性选取规则。
- [x] Node canary 比较 0.5%/1%/2% `N_E`，冻结最小可读 packet size。
- [ ] Null/Node/Edge/Node+Edge 在新 fit seeds 上运行 400 ms paired assays。
- [ ] 输出 recruitment、precedence、profile、event cloud、r50/r90、duration、size、OOD、return 和 sham difference。
- [ ] 直接判断 ignition 与 propagation，不运行 de novo KMeans。

**L1 canary 结果：** `0.5% * N_E` 已冻结；component 1/2 在 seeds 1001-1003 均为 3/3 可读，18/18 trigger 前 spike
bit-identical，runaway 0。canary 只选 instrument，不进入患者 capacity 结论。正式 config 额外锁定 canary、patient-training
prototype 与 profile reference hash；24 个 one-arm/one-seed jobs 最多并发 18，120 秒检查一次。

## Task L2：relaxed edge oracle（条件性）

- [ ] 仅在 scalar forced propagation 不足时实现四参数 component-pair mapper。
- [ ] 单测 target/source 方向、per-target conservation、exact no-op、topology/delay 不变。
- [ ] 正负 finite difference 建 Jacobian，受约束 ridge 生成候选，最多 16 个 full-SNN evaluations。
- [ ] component 3 固定为零负对照；不开放全 edge matrix，不先扫 beta。

## Task L3：network/optimizer audit

- [ ] 使用 6 fit / 3 selection / 3 confirmation 新 network seeds。
- [ ] 分别求 per-network oracle 与 shared-parameter oracle，报告 `Delta_network`。
- [ ] 已知好解存在后，以相同预算和 CRN 比较 Sobol/local refinement 与 CMA-ES 3 restarts。
- [ ] 参数等价类和多解直接进入 identifiability 结论。

## Task L4：spontaneous confirmation（条件性）

- [ ] 仅在 forced shared capacity 成立后执行。
- [ ] 冻结 candidate、objective、detector 和 classifier；confirmation seeds 不参与前序选择。
- [ ] primary 用 frozen classifier，secondary 用 de novo KMeans。
- [ ] 报每网络 A/B event rate、OOD、四层 mode distance 和 network bootstrap。

## 完成定义

- `decision.json` 对 target/objective、ignition、propagation family、network、optimizer、identifiability 分项作答；
- 现存数据没有保存的指标不补造；
- forced event、spontaneous interictal phenotype、patient blind 和 lifecycle claim 四层分开；
- 每个正式产物能从 clean commit、冻结 config 和记录的 input hashes 重建。
