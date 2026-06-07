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
- 2026-06-03 晚 **数学路线更新**：主模型 sigmoid rate field（`F_eff`）→ **LIF-derived rate field（`Φ_LIF(μ,σ)`）**；逻辑链不变只换 transfer，`F_eff` 降级、低异质性后置为 LIF 参数分布。关键更正：真 LIF 工作点稳健稳定（非 near-critical / 无 finite-k Hopf），self_limited_propagation 是非线性可激（全或无）→ Step-0a 目标 = 稳定可激、色散是诊断 finite-pulse 是闸门。新 Step-0 工作：0d 各向异性旋转控制（承重判据未做）+ Φ_LIF 收进 canonical src + σ-dynamics sensitivity + 0e heterogeneity 后置层。详见 `docs/archive/topic4/sef_itp_phase4_v2/lif_rate_field_theory_2026-06-03.md`。
- 2026-06-03 晚 **Step-0 工作 (1)(2)(3) 完成**：0d 承重判据 **PASS**（θ_prop 随连接轴转 <0.1°、isotropic 无轴；commit c183ed6）；`Φ_LIF` 收进 canonical `src/sef_hfo_lif.py`（commit 46f9040 + 4 项工程加固 e95af61：w_ee_mult 贯穿场 / 0d 单轨化 / mean_field 多初值取最低-nuE root + 暴露 wEE×1.4 双稳 / 0d 判据测试）；σ-dynamics sensitivity（commit f86e73a）。(4) 0e heterogeneity = deferred。
- 2026-06-03 晚 **Step 1 开工（锁定合同冻结）**：`docs/archive/topic4/sef_itp_phase4_v2/step1_noise_contract_2026-06-03.md` —— 用户 6 条方法学加固落数值（锁死事件检测器 + seed×amp 网格报比例 + 触发率曲线找宽区间不调点 + 窗口 B 回低-root + recovery-off 俘获失败对照 + 方向先场后真实 pipeline + 噪声下 isotropic+aligned-shaft 必须过不了）。
- 2026-06-06 **SNN 噪声自发离散事件扫描（同质底物，初步，判定降级）**：`docs/archive/topic4/sef_itp_phase4_v2/snn_noise_spontaneous_scan_2026-06-06.md` —— 把 Step-1 rate 场 (drive×σ) NULL 搬到放电真值上对照。**站得住**：σ=0 静默 + 同质放电组织 + 慢空间噪声**稳健地**自发出离散自终止事件（机制上与 rate 场"离散几乎不存在"相反）。**撤回（用户审阅 §7，5 条全实）**："复现 NULL / 事件太快"——率门四处不可靠（标定漏 σ_ref、1.5s 分辨不了低频、率比较分母错配=模拟数所有成核 vs 数据数穿出来的群体事件、tau 没到边界）。异质核（Step 3）更硬的动机是模板多样性（本测试未测）。runner = tracked `scripts/sef_hfo_snn_noise_scan_runner.py`。

## 数据侧（paper-A）：rate vs 传播几何（探索性，与上面的模型侧并行）

- plan：`docs/superpowers/plans/2026-06-06-sef-hfo-soz-localization-rate-vs-geometry.md`（v3/v4/v5 修正块）。
- **2026-06-07 低事件窗读数稳定性 + de novo 发现能力**（探索性，n=28，无 held-out）：`docs/archive/topic4/sef_hfo/low_rate_template_stability_2026-06-07.md`。三层结论，主参照=全程模板（非 rate）：
  - **read-back（已学到模板、短窗读回）**：低事件窗里传播源→汇模板比发放计数更抗采样不足，扣对照后 EXCESS +0.131（25/28，p<1e-4）。**这不是 SOZ 定位优势**（静态定位 + SOZ 内部稳定性早前均 NULL，见 `soz_localization_results_2026-06-07.md`），而是读数时间稳定性。
  - **de novo（短窗从零发现模板）= 负结果**：方向恢复 NULL（ALL +0.022 p=0.10）、端点恢复 NULL（excess −0.064）；一致性臂复现 read-back（epi +0.156）证明非 pipeline 回归。
  - **学习曲线**：失败既因事件少（方向需 ~100 事件饱和）又因安静时段本身状态变了（真实窗落随机基准下 −0.29，rate 同降）。**统一：传播模板要"多事件学习、短窗读回"，不能"短窗从零发现"。**
  - 代码 `src/low_rate_template_stability.py` + `scripts/{run_low_rate_denovo,run_denovo_endpoint_stability,run_denovo_learning_curve,plot_*}.py`；tests 22 全绿。

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

### HR/FHN 节点动力学路线（Phase 4 v1）— **整体 SUPERSEDED 2026-06-01，全部集中在此**
**一句话**：曾用 Hindmarsh-Rose / FHN 抽象节点搭模型；2026-06-01 起主模型切到 EI / LIF-derived rate field（SEF-HFO v0.2），HR 路线整体降级。要找 HR 的东西，全在下面三处（别处不再散落）：
- **文档**：`sef_itp_phase4_v1/stage1_results_2026-05-28.md`、`stage1b_results_2026-05-28.md`（HR 单节点 regime + burst-envelope 标定，均已加 SUPERSEDED 头）
- **spec**：`docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md`（HR/FHN route 设计，已加 SUPERSEDED 头）
- **结果图**：`results/topic4_sef_itp/phase4_hr_route_SUPERSEDED/`（HR 单节点相图 + regime map + envelope 标定图；含 `_SUPERSEDED_README.md`）
- 保留为历史 audit trail / sensitivity，**不是当前路线**。当前路线 = SEF-HFO v0.2（`docs/topic4_sef_itp_framework.md` + `sef_itp_phase4_v2/`，结果 `results/topic4_sef_hfo/`）。

## 跨文档链接

- `docs/topic4_sef_itp_framework.md` §9.1 — SBA framework 取代 / 保留范围
- `docs/topic4_sef_itp_framework.md` §13 — 历史文档索引（含本 INDEX 反向链接）
- `docs/topic0_methodology_audits.md` §5h — Topic 4 attractor on masked re-run 是 SEF-ITP Phase 0 子步
