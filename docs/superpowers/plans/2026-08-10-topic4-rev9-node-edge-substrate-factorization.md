# Topic 4 rev9 node-edge substrate factorization 执行计划

**Spec：** `docs/superpowers/specs/2026-08-10-topic4-rev9-node-edge-substrate-factorization-design.md`  
**目标：** 先完成冻结 h 的因果诊断，再在真实 SNN 方程中把 node core 与 target-normalized E->E edge core 做
matched-local-response 分解；通过后只生成 lifecycle handoff，不在本线改 slow equations。

## 执行纪律

- 结果目录：`results/topic4_sef_hfo/data_driven_core_field_rev9/`，不用 PR 编号。
- 任何长仿真由 `systemd-run --user` 启动 `/usr/bin/nohup`；每个阶段必须有 `.log/.status` 和桌面完成通知。
- 每次正式运行前提交 producer，运行时 `provenance.tracked_modules_dirty=false`。
- network / OU / Poisson seed 在 arm 间配对；一个 arm 出错，同组 paired unit 整组失效。
- 旧 rev8.1 held-out 已消费，只能作历史 read-back，不能作 rev9 选参或 blind gate。

## Task 1：rev8.1 图与冻结候选因果前置

- [x] Fig4A 增加 3D h landscape 旁的 signed `Delta Vtheta=-hd` 神经元图。
- [x] 设计全 50 events 的 event-equal earliest-1% onset density producer，并逐事件核对 final profile hash。
- [x] Fig4A mode A/B 叠加 all-event onset density、source centroids 和 h contours。
- [x] direct readout 改为所有 contacts 共用一个 model-current amplitude scale；contact family 与 mode shading 分开图例。
- [x] Fig4B model profile 实线、patient prototype 虚线、patient recording-block variability band。
- [x] 2x2 matrix 增加 conditional hierarchical bootstrap 95% CI；metadata 明确它条件于冻结 KMeans 标签。
- [x] 增加 `D_curve` vs worst-mode correlation benchmark；control y 只画描述性点估计。
- [x] 正式 nohup producer 完成后检查 JSON/NPZ hash、PNG/PDF、README 和视觉遮挡。
- [x] 根据 `P(component|mode)` 判断是否值得启动 component lesion；关联弱不等于路线自动停止，但必须降级表述。

**Task 1 实测：** 50/50 final events 逐曲线 hash 复现。Mode A 的 17/17 source centroids 最近 component 2；
Mode B 为 component 1 `23/33`、component 2 `10/33`，component 3 为 `0/50`。这足以启动 lesion/relocation，
但仍只是关联。conditional hierarchical bootstrap 的 mode-A matched cell 为 `+0.155 [0.011,0.318]`，mode-B 为
`+0.977 [0.969,0.987]`；CI 条件于冻结 KMeans 标签，不是重新聚类的不确定性，也不改变 rigid benchmark fail。

**产物：**

- `joint_confirmation_rev8_1/all_event_onset_diagnostics.{json,npz}`
- `joint_confirmation_rev8_1/figure_diagnostics.{json,npz}`
- `fig4_data_driven_core_field_rev8_1/figures/fig4a_*`
- `fig4_data_driven_core_field_rev8_1/figures/fig4b_*`

## Task 2：Edge mapper v2

**修改：** `src/topic4_core_connectivity.py`

- [ ] 新增 `field_normalized_ee_pair(net,h,alpha,beta=0,...)`，公式严格按 spec §4。
- [ ] primary `exp(alpha*h_target*h_source)`；geometry term 默认关闭。
- [ ] 同一 target 的 normalization 跨全部 delay bins；只修改 E target 的 E source edge。
- [ ] 删除 stale `ampa_flat`；若 `gaba_flat` 已存在可保留，因为 GABA 未变。
- [ ] 输出 incoming sum、topology、delay、E->I/GABA hash、ratio、KL、ESS diagnostics。
- [ ] 旧 `field_normalized_ee_core` 保持兼容，或明确包装到 additive legacy path；不得静默改变旧 artifact 的定义。

**测试：**

- [ ] 行=target、列=source 的非对称 toy matrix 锁定方向。
- [ ] 纯 target factor 在 normalization 后抵消；`h_t*h_s` pair factor 不抵消。
- [ ] alpha=beta=0 逐值 no-op；权重正、有限。
- [ ] 每个 E target incoming-E error `<=1e-9`。
- [ ] topology/delay/E->I/GABA 不变；flatten cache 重建后使用新权重。
- [ ] 畸变预算拒绝 edge ratio 或 KL 越界候选。

## Task 3：Frozen-field lesion 与 d interaction

**新增：**

- `src/topic4_core_field_causal.py`
- `scripts/run_topic4_rev9_frozen_field_causal.py`
- `config/topic4_rev9_node_edge_factorization.json`

- [ ] 从 frozen theta 解出每个 raw `q_c` 和原 projection lambda。
- [ ] 三个 component 全做 direct lesion；保持原 lambda，不补预算。
- [ ] control location 在运行前从绕 sheet center 的 `+90/-90` 度旋转中选与原场重叠最小的一侧；orientation 同步旋转，matched relocation 调回原 h budget。
- [ ] relocation 保留 Gaussian weight/covariance，只对替换后的 raw q 重新求 `project_to_budget` 的 lambda；不得用整体 q 缩放冒充 matched control。
- [ ] d arms：original、global shuffle、1 mm neighborhood shuffle、positive-only、negative-only。
- [ ] positive/negative 同时输出 raw 与 `sum h|d|` matched。
- [ ] metadata 明确 d 是位置独立的冻结随机抽样；shuffle 只检验 realization dependence，不表述为患者空间配准。
- [ ] 所有 arm 跑 paired seeds，统一读 event rate、mode proportion、Dglobal、D_A/B、onset、duration、size、return。
- [ ] 若 lesion 只改 event rate、不改 onset/mode，不能称 mode-specific core。

## Task 4：Small-kick instrument calibration

**复用：** `src/topic4_propagation_operator.py::build_w_resp` 的 kick-sham、target/source 方向和 coarse-bin response 语法。

**新增：** `scripts/run_topic4_rev9_local_response_calibration.py`

- [ ] 固定 `T=400 ms, t_kick=100 ms, r_kick=1.0 mm`。
- [ ] 位置为三个 Gaussian center + 三个 matched off-field controls。
- [ ] canary 扫 `{0.25,0.5,1.0}*nu_theta` 和三个 post-kick windows。
- [ ] 只用 Node-only seeds `901--903` 选择最小 quasi-linear amplitude 与 first-generation window。
- [ ] 在选择前写 instrument JSON + SHA256；后续 arm 不得重选。
- [ ] kick/sham 同 seed，保存未 clip 的 signed response 与用于正式量的 positive response。
- [ ] kick 或 sham 在 calibration window 内出现 detector-qualified event 时预注册排除该 pair；每位置至少 `5/6` seeds。
- [ ] gate：source/downstream 单调、无 runaway/tonic、响应 SNR 可读。
- [ ] 另跑 `T=2000 ms` no-kick paired baseline，rate/active-fraction p95 改变 >25% fail；短 run event count 只报告。

## Task 5：选择 alpha* 的 matched-local-response calibration

- [ ] seeds `901--906`；arm 为 Null、Node-only 和 Edge-only alpha grid。
- [ ] 先做结构畸变门，再跑 `{0,0.25,0.5,1,2,4}`；禁止仿真已越界候选。
- [ ] 用 spec §7 的 `J_cal` 排序；最优相邻区间只做两轮 midpoint refinement。
- [ ] 输出 per-site/per-seed response vector、paired delta、Node seed IQR scaling 和完整失败原因。
- [ ] equivalence gate：两个 gain ratio、r90、response-map rho、`5/6` seed direction 全过。
- [ ] 若 primary 不过，先裁定失败原因；只有“空间 profile 系统性不匹配且 alpha 已匹配总 gain”才允许 beta secondary。
- [ ] beta 开启时先做 alpha/beta response identifiability；参数相关或 profile 不改善即退回 primary negative result。

**产物：**

- `node_edge_calibration/instrument.json`
- `node_edge_calibration/edge_structure_audit.json`
- `node_edge_calibration/local_response.json`
- `node_edge_calibration/figures/README.md` 与 paired response 图

## Task 6：四臂因果分解

- [ ] 冻结 alpha* 后在 `911--922` 上运行 Null / Node-only / Edge-only / Node+Edge。
- [ ] Node+Edge 使用相同 alpha*，不得为改善 patient score 重调。
- [ ] 先报 baseline rate、event rate、runaway/tonic；再报 mode 与传播，不只筛 readable events。
- [ ] mode 输出固定为 recruitment、precedence、profile、event distribution 四层。
- [ ] 用 paired network bootstrap 比较 Edge-Node、Node+Edge-Node、Node+Edge-Edge。
- [ ] 分类：edge sufficient / node specific / redundant / synergistic / destabilizing / unidentifiable。
- [ ] 生成一张机制分解图和一张 Fig4-style event/KMeans companion；图目录同时写中文 README。

## Task 7：rev9 patient objective 开发与 formal 开启门

- [ ] patient train 按 recording block 建四项 mode distance 和各自 floor。
- [ ] 模型每个候选固定 KMeans `K=2,n_init=100,random_state=0`，Hungarian 对齐 patient-train prototypes；10 个 random-state consensus AMI 中位数 `<0.8` 直接 mode-unstable。
- [ ] precedence 只纳入达到冻结共同参与 support 的 contact pairs，权重与缺失规则写入 artifact。
- [ ] event-cloud PCA 只在 patient train 拟合，95% variance、最多 8 维；64 个 sliced-Wasserstein directions 与变换参数 hash-lock。
- [ ] mode JS 使用自然对数和每 mode 0.5 Jeffreys pseudocount；其 floor 也只来自 patient-train recording-block split-half。
- [ ] 实现 `J_rev9`、fixed n=20 global、fixed n=32 mode、每簇至少 10。
- [ ] 用 patient train split-half 和已有 controls 做 zero-simulation scale sanity；不得用旧 held-out 调 lambda/tau。
- [ ] 没有新 blind unit 时只输出 `DEVELOPMENT_ONLY_NO_BLIND_UNIT`，停止 formal optimization。
- [ ] 若用户另行冻结新 blind unit，再写独立 fit/selection/final seed config，三池不得与 rev8/rev9-A/B 重叠。

## Task 8：Lifecycle handoff

- [ ] 只在 Task 5 equivalence + Task 6 因果可解释时生成 bundle。
- [ ] bundle 含 h/neuron order、alpha/beta、全部 hash、结构守恒、畸变、local response 和 arm metrics。
- [ ] 不在本分支编辑 FCXR-LC3/LC4 slow equations 或正在执行的 lifecycle worktree。
- [ ] lifecycle 消费端把 alpha 作为 static substrate coordinate；动态坐标仍用既有 Z/M/adaptation 合同。
- [ ] 每个相图点重新跑 finite-pulse lifecycle：entry、bounded carrier、exit、postictal protection、return/recovery。
- [ ] runaway、tonic、brief dip、bounded carrier、recovered return 分开记录。

## 决策表

| 结果 | 下一步 |
|---|---|
| edge 无法在畸变预算内匹配 node | 关闭当前 edge family；不做患者优化 |
| edge 匹配局部 gain，但 profile 不匹配 | 只允许一次 beta geometry secondary |
| edge 匹配且复制 node 的 onset/mode | 进入四臂分解，评估 node 是否可替代 |
| Node+Edge runaway/tonic | 判 destabilizing，不以高事件数算成功 |
| lesion/relocation 无位置特异性 | 降级“Gaussian components”为空间参数化，不称 cores |
| d shuffle 不改变 mode | d 不是必要配准层；后续简化 node mechanism |
| 无新 blind patient unit | 只交付 mechanism/development result，不做 patient generalization claim |

## Definition of done

- 图、统计 sidecar、edge mapper、局部响应、四臂和因果 controls 均可由干净 checkout 重建；
- spec 的 target/source 方向、归一化轴和 engine 更新式与代码逐项一致；
- node/edge 的比较基于 matched response，不使用虚假的同参数预算；
- 最弱模式有连续目标保护，平均模式不能掩盖 mode A；
- blind、development、engineering、lifecycle 四层结论在 JSON、README 和报告中明确分开。
