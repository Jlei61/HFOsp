# Topic 4 Archive Index — Model layer

> **Topic 4 formal entry（2026-06-01 起）**：`docs/topic4_sef_itp_framework.md` (**SEF-HFO / SEF-ITP framework v0.2**)
> **上游 SBA framework**：`docs/paper1_framework_sba.md`（v1.1.2 lock 2026-04-30 / PR-7 addendum 2026-05-01）—— SEF-ITP 取代其 BHPN-toy / BHPN-fit toy-mechanism 部分；保留其 P1/P2/P3/P5 红线
> **硬前置**：`docs/topic0_methodology_audits.md` §3.1 phantom-rank 修复 + §5 broad re-derivation roadmap（已解锁；后续 runner 仍必须使用 masked features）
> **范围**：Topic 1 / Topic 3 实验数据之上的模型层；当前主路径 = SEF-HFO（间期 HFO 传播的空间易激场模型）
> **不属于**：Topic 2（事件间 ~2 Hz / refractory + slow modulation；属独立 Paper 2）。

## 当前主路径（2026-06-01 起）

**SEF-HFO**（Spatial Excitability Field model for interictal HFO propagation）：

- 主文档：`docs/topic4_sef_itp_framework.md`
- v0.2 plan：`docs/archive/topic4/sef_hfo_topic4_v2_plan_2026-06-01.md`
- 核心断言：间期群体 HFO = 局部低异质性、各向异性连接、近临界但仍亚阈值的 E-I 易激斑块，在噪声触发下产生的自限性瞬态传播事件；低异质性必须通过 effective gain 实际进入稳定性分析
- 保留 v1 真实数据验收合同（H1–H6）：endpoint compactness / source-sink reversal / mark independence + stable geometry / rate-geometry decoupling / pre-post-ictal endpoint shift / participation-field segregation
- v0.2 建模路线：effective gain → linear dispersion map → finite-pulse response map → 2D rate field + geometry controls → LIF E-I SNN → 抽象慢变量 feasibility bridge
- 文献 framing：具体细胞机制多样，但在中观层收敛到易激性、恢复能力、有限扰动响应和空间招募变化；SEF-HFO 只承接这个抽象动力学层，不把间期 HFO 写成微型发作
- 转向触发点（2026-06-01）：旧 HR/FHN route 仍能作为 sensitivity，但主线不够 sharp；v0.2 把机制收紧为“低异质性必须 gain-closed、事件必须 pulse-validated、几何证据必须 control-disciplined”
- 2026-06-02 review amendment：v0.2 重定位为 **two-stage control-disciplined exploratory mechanism screen**（Stage 1 exploratory screen → freeze 最小机制 → Stage 2 held-out consistency validation，筛选/验证目标不重叠）；三条纪律补丁（operating-point family + 不准抢救、recovery 从 rate 层起跨阶段同构、判别指标改为「方向随连接轴转不随电极杆转」带阈值判据）见 v2 plan 顶部 2026-06-02 amendment。
- 2026-06-03 amendment：「局部」收窄为**时间离散自终止（空间可填满 SOZ 邻域网格）** + 促临界↔稳态回拉拮抗作组织视角 + recovery 并列机制分支硬化（不写成暂不做）+ homogeneous/heterogeneous 不绝对化 + 发作桥接降格为 H5/Phase 3 候选机制（不作 clinical seizure onset 结论）+ coworker1 LIF (`Jlibrary/ei_snn_scaffold/`) = Step 4 前置参考。**Step 0 验收（v2，2026-06-03 晚 user ratify）= 通过 mechanism-scale gate**（换 LIF colored-noise transfer 后，有限脉冲在 LIF-rate field 产生「点着→定向传播→自终止」事件；sigmoid 路线结构性失败）。**口径：过的是 mechanism-scale gate，不是 patient-fitted quantitative match**（尺度一致即可，~70/110ms、cm 级、几十 ms 展布同一量级；定量时长/速度/范围张力降为 Step 1/2 sensitivity）。**Step 1 解锁**，带两候选窗（Brunel-like 短/稳健主线 + 病理高激+recovery envelope-时长 sensitivity）。前一判定（sigmoid 路线 LOCKED）保留作 audit trail。结果链：`step0_results_2026-06-02.md`（sigmoid）+ `lif_transfer_route_2026-06-03.md`（LIF route + 验收 + Step 1 设计）；详见 framework 顶部 Step-0 验收判定 v2。

## 前一代主路径（2026-05-20 至 2026-06-01）

**SEF-ITP v1**（Spatial Excitable Field model for Interictal Template Propagation）：

- 主文档同上，v1 历史审计链保留在顶部 banner 与各 archive。
- 核心断言：间期群体 HFO = 空间组织化病理易激场 θ(x) 被扩散物理反复采样的痕迹。
- 5 phase 路线：Phase 0 phantom-rank 硬前置 → Phase 1 空间几何免费检验 → Phase 2 temporal × geometry 联合 → Phase 3 ictal-adjacent → Phase 4 HR/FHN neural-field toy。
- v1 的真实数据验收合同继续保留；Phase 4 HR/FHN toy 降级为历史探索 / sensitivity。
- 转向触发点（2026-05-20）：识别 BHPN-toy 是循环论证（A_ij 矩阵预编码现象 → Kuramoto 必然演化出双稳态 → "复现"是 tautology）。

## 历史归档

### `layered_model_framework.md`
2026 年 4 月被 `paper1_framework_sba.md` 取代的早期分层模型框架（intra-event Kuramoto 保留 + 不再追求 inter-event 大一统）。SBA 框架的概念前身——保留作为思路演变的历史记录。

### `pr_t4_1_bhpn_toy/` — BHPN-toy plan（**SUPERSEDED 2026-05-20**）
- `pr_t4_1_bhpn_toy_plan_2026-05-01.md` — plan-of-record v2，**已被 SEF-ITP 取代**
  - 顶部有 SUPERSEDED banner 指向 `docs/topic4_sef_itp_framework.md`
  - 保留为历史归档：v2 修订过程（rotating-frame rank / direction-agnostic mark dependence / TDD unit vs integration 拆分）作为方法学训练范例；其中 "k=2 是对称 Hebbian 数学必然，不是 mechanism discovery" 是触发 SEF-ITP 转向的关键认知

### `sef_itp_phase4_v1/` — HR/FHN route（**SUPERSEDED AS MAIN ROUTE 2026-06-01**）
- `stage1_results_2026-05-28.md`、`stage1b_results_2026-05-28.md` — HR 单节点 / burst-envelope 标定结果。
- 保留为历史探索 / sensitivity。主模型路线已切到 SEF-HFO v0.2：effective gain + finite-pulse response + geometry controls first。

## 跨文档链接

- `docs/topic4_sef_itp_framework.md` §9.1 — SBA framework 取代 / 保留范围
- `docs/topic4_sef_itp_framework.md` §13 — 历史文档索引（含本 INDEX 反向链接）
- `docs/topic0_methodology_audits.md` §5h — Topic 4 attractor on masked re-run 是 SEF-ITP Phase 0 子步
