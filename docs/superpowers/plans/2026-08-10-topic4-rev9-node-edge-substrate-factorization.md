# Topic 4 rev9 node-edge factorization 执行计划

**Spec：** `docs/superpowers/specs/2026-08-10-topic4-rev9-node-edge-substrate-factorization-design.md`
**状态：** 探索性执行已开始
**结果目录：** `results/topic4_sef_hfo/data_driven_core_field_rev9/`

## 执行纪律

- 只把 node 重建错误、edge 结构守恒错误和仿真数值错误作为硬停止。
- ratio/KL/ESS、SNR、响应匹配、mode support 和因果特异性全部连续报告，不设层层 blocker。
- network/noise seed 在 arm 内配对；runaway 保留为结果，不通过筛掉样本美化匹配。
- 长任务由 `systemd-run --user -> /usr/bin/nohup` 启动，必须有 `.log/.status` 和完成通知。
- producer 先提交再正式运行；旧 patient held-out 不参与选择。

## Task 1：rev8.1 图和冻结场零仿真诊断

- [x] Fig4A：3D h landscape、signed `DeltaVtheta=-hd`、全事件 onset density 和 h contours。
- [x] Fig4B：model/patient prototype、patient block variability、hierarchical bootstrap CI 和 benchmark Pareto 图。
- [x] direct readout 统一幅度标尺、图例和 mode shading。
- [x] 50/50 final event profile hash 复现；nearest-component onset sidecar 完成。
- [x] 增加 soft Gaussian responsibility、mode-label permutation 和 pooled-location null。
- [x] 冻结 rev8.1 mode classifier、distance/OOD p99，并保存 de novo KMeans 对照配置。

## Task 2：Node 重建和 Edge mapper v2

- [x] 从原 quantile seed 独立重建 `d_i`、h 和 `Vtheta_0-hd`；禁止从 Vtheta/h 反推。
- [x] 实现 `field_normalized_ee_pair`：source-target pair、跨 delay target log-sum-exp normalization。
- [x] 精确 no-op、零 incoming target、全 AMPA cache invalidation、结构和 incoming-E 守恒诊断。
- [x] 单测锁定 target/source 方向、可分离 target factor 抵消、pair factor 生效和 beta 数值路径。
- [x] 用冻结 rev8.1 artifact 做 node hash/reconstruction formal preflight；float/dtype 均逐值一致，最大误差 0。
- [x] alpha grid 结构审计写入正式 JSON；所有 alpha 保持 topology、delay、E->I/GABA 和 incoming-E。

## Task 3：冻结 small-kick instrument

- [x] 写 `config/topic4_rev9_exploratory.json`，冻结 E-only top-hat、初始 onset 100 ms、duration 18 ms、radius 1 mm。
- [x] 首轮幅度 `{0.25,0.5,1.0}*nu_theta`，5 ms bins，pulse-end 后三个候选窗口。
- [x] 明确 signed/positive response、source/downstream、OLS slope、窗口 tie-break 和 paired event exclusion。
- [x] Node canary seeds `901--903` 完成 57 次仿真；54/54 kick pairs 被 sham 启动事件污染，0 runaway。
- [x] eligibility reconciliation 作废原始 20--40 ms 自动选择：`0/18` site-seed eligible，正式状态无可用窗口。
- [x] 同三条 Node sham onset scan 完成：无事件 seed 数 `[0,1,3,2,2]`，冻结唯一 3/3 安静的 `220 ms`。
- [x] 第二轮 producer 固定复用 e785 的原 network cache，避免 timing 比较混入 connectivity 改变。
- [x] 新 onset 下 Node canary 完成：3/3 sham 安静；42/54 kick 触发事件，0 runaway，仅 3/18 site-seed eligible 且全属 seed 903。
- [x] 作废高剂量下的跨网络冻结解释；20--40 ms 仅保留为单网络 canary candidate。
- [x] 同三张 network、同 220 ms onset 跑 `{0.05,0.10,0.20}*nu_theta` threshold mapping：16/54 event、0 runaway。
- [x] 确认 `0.05*nu_theta` 在 18/18 site-seed 均为非事件；更高剂量显示 component 较早进入 ignition 的探索性差异。
- [x] 发现 broad response interval 误用于排除全部窗口；threshold mapping 逐窗可用数为 16/18、11/18、7/18。
- [x] primary window 改为预定义 0--10 ms first-generation response；后两窗只作传播诊断，不再最大化 downstream mass。
- [x] 最终 canary 使用 `{0.0125,0.025,0.05}*nu_theta`；逐窗可用数 17/18、16/18、15/18，0 runaway，覆盖 3/3 networks。
- [x] 修复长跑结束时才读 provenance 的错误；producer 改为启动前快照。
- [x] 用 944 commit blob、wrapper/systemd unit 和运行参数重建 raw 来源；正式 sidecar provenance 为干净 642e commit。
- [x] 保存 SNR、event/runaway 和排除数；这些是诊断，不是放行门。

## Task 4：Alpha exploratory calibration

- [x] Edge `alpha=0.5, seed=901` smoke 通过：结构守恒、逐窗 readout、启动 provenance 和 nohup 状态链均正常。
- [x] worker 按一个 `arm-alpha-seed` 分片；基于 80 logical CPUs/246 GiB available memory，最多 18 个并发、分两批运行。
- [ ] 并行生成 seeds 904--906 的 Node reference；缺失 network 仅在冻结 engine hashes 一致时构建。
- [ ] seeds `901--906` 扫 `{0,0.25,0.5,1,2,4}`，按 `J_cal` 排出 response-matched `alpha_star`。
- [ ] 最优相邻区间至多两轮 midpoint；beta 默认关闭。
- [ ] 原始单位和标准化分数同时画，附 ratio/KL/ESS、baseline shift 和 paired count。
- [ ] seeds `911--922` 不重选 instrument/alpha，做 out-of-selection 描述性复测。
- [ ] primary family 若仅有径向宽度缺口，再开 beta 小网格；轴向/非径向失配直接保留阴性结果。

## Task 5：四臂因果分解

- [ ] seeds `911--922` 跑 Null / Node / Edge / Node+Edge；Node+Edge 不重调 alpha。
- [ ] 每个 endpoint 计算 `Delta_N`、`Delta_E`、`Delta_NE` 和 `I_Y=Y_NE-Y_N-Y_E+Y_0`。
- [ ] 同时报 frozen classifier 与 de novo KMeans、OOD、AMI、support 和 consensus。
- [ ] 输出 onset、rate、mode proportion、recruitment、precedence、profile、event cloud、duration、size、return。
- [ ] endpoint-level paired bootstrap 和机制表允许 `unidentifiable`。
- [ ] 生成一张机制分解图和一张 Fig4-style direct waveform/KMeans 图，图目录写中文 README。

## Task 6：Component lesion、relocation 和 d interaction

- [ ] 三个 component 全做 direct lesion 和 matched relocation。
- [ ] relocation 报 projection collateral diagnostics，不把整体 field 改变冒充位置特异性。
- [ ] global/neighborhood shuffle 各 5 个预冻结 seeds；positive/negative 各报 raw 与 `sum h|d|` matched。
- [ ] 使用 frozen/de novo 双 mode readout，比较 event rate、mode、onset、profile 和 return。
- [ ] 阴性结果只降级 core 命名，不自动停止稳定 substrate 的 lifecycle 工程探索。

## Task 7：Patient objective development

- [ ] assignment/proportion pool 改为 50 events；JS 使用全 pool。
- [ ] shape metrics 用每 mode 10 events 的多组 hash-locked balanced subsets。
- [ ] patient floor 与模型 estimator 在 event/mode count、block、missingness、PCA 和重复抽样上同构。
- [ ] 保存 frozen/de novo assignment margin、support 和 estimator hashes。
- [ ] 无新 blind unit 时只写 `DEVELOPMENT_ONLY_NO_BLIND_UNIT`，不执行 formal patient optimization。

## Task 8：Substrate bundle 和 lifecycle 计划

- [ ] bundle 包含 h/d/Vtheta、alpha/beta、neuron order、全部 hash、结构诊断、response 和四臂 endpoint。
- [ ] `alpha_star` 标记为 response-matched reference；其他 alpha 是新 exploratory substrate。
- [ ] lesion 阳性才称 mode-specific core；阴性使用 data-driven spatial parameterization。
- [ ] 后续连接性相图沿用既有 Z/M/adaptation 方程，不把静态 h/alpha 改名为 slow state。
- [ ] 每个相图点重新检查 entry、bounded carrier、exit、postictal protection 和 return/recovery。

## 当前最短执行顺序

1. 提交 Node/Edge mapper、测试和本版 spec/plan。
2. 运行 alpha grid 的零仿真结构审计，确认真实 40k-neuron network 上的守恒和畸变曲线。
3. 完成 responsibility/null 与 frozen classifier sidecar。
4. 冻结 instrument，先用 `901--903` nohup canary 画 response surface。
5. 根据 response surface 选 `alpha_star`，再启动 `911--922` 四臂探索。

## 探索轮完成定义

- spec 中公式、真实 engine 语义和 producer 一致；
- node 可精确重建，edge mapper 在真实 network 上保持结构和 target incoming-E；
- 有可复查的结构曲线、local-response surface、四臂 interaction 和 frozen/de novo mode 图；
- 阴性、runaway、不稳定和 OOD 不被隐藏；
- engineering、development、patient blind 和 lifecycle 四层结论分开。
