# Topic 4：SEF-HFO / SEF-ITP Framework —— 间期 HFO 传播的空间易激场模型

> **状态**：**v0.2 plan lock draft 2026-06-01**。Topic 4 主模型路线更新为 **SEF-HFO：低异质性 + 各向异性连接 + 近临界 E-I 易激场 + 慢变量调制**，并在 review 后收紧为 **gain-closed + pulse-validated + control-disciplined** 版本。旧 HR/FHN Phase 4 modeling route 降级为历史探索 / sensitivity，不再是主模型路线。详细计划见 `docs/archive/topic4/sef_hfo_topic4_v2_plan_2026-06-01.md`。
> **2026-06-02 review amendment（two-stage screen 重定位）**：SEF-HFO v0.2 从「强机制确认模型」重定位为 **two-stage control-disciplined exploratory mechanism screen**（Stage 1 exploratory screen → freeze 最小机制 → Stage 2 held-out consistency validation，筛选/验证目标不重叠）；三条核心纪律补丁（① operating-point family + 自洽稳态 + 报「可接受族里多大比例成立」不报「存在一个点」+ 不准用均值阈值/外部输入/连接强度抢救机制；② recovery 从 rate 层就加、0b 扫纯抑制 vs 加恢复变量按实测定、rate 与 LIF 用同一套、恢复时间常数锚真实事件时长；③ 承重判别指标改为「模板方向随连接各向异性轴转、随电极杆旋转不变」带阈值回归判据，isotropic+aligned-shaft 对照必须过不了）详见 `docs/archive/topic4/sef_hfo_topic4_v2_plan_2026-06-01.md` 顶部 2026-06-02 amendment。
> **2026-06-03 amendment（user 讨论敲定：时间离散性 + 促临界↔稳态回拉拮抗视角；建模主线仍限纯间期）**：本轮修正**全部不改 v0.2 核心边界红线**（Topic 4 主模型只解释间期 HFO 群体事件的中观传播组织，**不解释 clinical seizure onset**）。六条：
> 1. **「局部自限」重新表述为「时间上离散自终止」（空间可填满 SOZ 邻域网格）**——修正之前把空间局部性看得过重。建模目标**不是**空间上比网格更小的波，而是在 SOZ 小邻域网格内出现的、时间上离散且能自终止的群体 HFO 传播事件；网格本身 = 病灶及其小邻域，波填满网格仍是解剖学局部。要锚的是传播 lags（= 模板时间结构）与事件包络时长（= 自终止尺度），**不是** HFO 80–250Hz 载波频率（载波仍 out of scope）。
> 2. **recovery 不是首选生物解释，但必须作为并列机制分支进入 Step 0b**（**不**写成暂不做）——沿用 v0.2 plan「0b 扫纯抑制自限 vs 加恢复变量自限、report-both、no-auto-select」合同；是否保留 recovery 由**真实事件时长 / 空间范围 / 自限区宽度**决定。亚临界噪声触发（"Brunel 支"）是离散事件的概念首选路径，但 recovery（"Pinto–Ermentrout 支"）作为并列分支硬性进入对比，不预先砍掉。
> 3. **coworker1 LIF 结果 = Step 4 前置参考（LIF 引擎可行性证据），不是我们的模型闭环**——它证明 LIF 层能产生各向异性行波；距模型闭环仍缺：亚临界离散事件、低异质性 patch、synthetic 走真实事件 pipeline、controls 必须过不了、rate↔LIF 同构。
> 4. **促临界 ↔ 稳态回拉拮抗** 作为看待整个参数方案的组织视角（描述性框架，非新验收）——旋钮归两极：促临界（低阈值异质性 / patch、强 E→E、增益↑、去抑制、Cl⁻ 致 E_GABA↑）vs 回拉（sAHP、自适应阈值、强/快抑制、阈值异质性↑）。同一拮抗在两尺度各跑一遍：事件内（快：可激传播 vs recovery/空间约束掐断）+ 跨时间（慢：促临界漂移 vs 稳态）。recovery = 回拉极的快端，与慢端稳态同源。
> 5. **homogeneous / heterogeneous（不绝对化）**——heterogeneous patch + surround 是**间期自限事件的自然结构**（促临界病理核心被周围回拉机制约束住）；homogeneous 是单相极限，**可能**代表全回拉 / 全促临界 / 或控制条件，**不应直接等同「发作态」**。间期态 = 促临界核心被周围回拉约束住；发作样招募 = 该约束被削弱后、自限安全余量被随机涨落穿越。
> 6. **发作桥接 = H5 / Phase 3 候选机制，降格、不进入当前纯间期主验收**——可能图景：间期事件**既是**噪声涨落、**又可能**留下净促临界痕迹；当自限安全余量被压低时，某次涨落更容易越界成为持续招募。具体是逐事件累积 / 悬停临界 / 还是发作后重置（风险率无记忆 vs 渐增），**须用 seizure 间隔 / 风险率数据判**。**此发作桥接只作后续仿真与 H5 数据检验的候选机制，不作为当前 Topic 4 主模型已解释 clinical seizure onset 的结论。**
>
> **执行序不变**：先做固定工作点上的**纯间期**（异质 patch + 约束，离散自终止 + 传播轴 + 端点几何，全高于 controls），坐实拮抗快端；慢端（促临界漂移 → 发作）作为独立扰动后加、需自己的失败模式。Step 0a/0b 机器已落地（`docs/archive/topic4/sef_itp_phase4_v2/step0_results_2026-06-02.md`）：scaffold 深度稳定 / 主模 k=0；能力体检证明机器能算出有限 k Turing，但 finite-k Hopf（行波）需 Brunel 时间结构、未在 scaffold 出现。
> **Step 0 验收判定（2026-06-03，sigmoid 路线；已被下方 LIF-route v2 supersede，保留作 audit trail）**：Step 0 **机器里程碑 accepted**（模块 + 21 tests + gate + runner，smoke tier 过）；但**解锁 Step 1 的 formal gate 未过**——scaffold 占位参数下 `fraction_with_window=0`、0 候选窗、深度稳定 / 主模 k=0、有限脉冲全 extinction，按纪律「只有落在 self-limited propagation 区的参数进入后续分析」**无参数入选** → **Step 1 保持锁定**。解锁前置：(a) 数据锚定 operating-point family + 单位（provenance→data_locked）；(b) data_locked re-run 0a/0b 取得「移动波前自限 + 正安全余量」候选窗（recovery off/on 并列、dt/L 稳定、报族内通过比例）；(c) 很可能先换 rate 层 LIF 有色噪声 transfer + Brunel 时间结构以预测有限-k Hopf/传播态（rate↔LIF 同构交叉验证）。详见 v2 plan 2026-06-03 amendment + `docs/archive/topic4/sef_itp_phase4_v2/step0_results_2026-06-02.md`。
> **Step 0 验收判定 v2（2026-06-03 晚，user ratify — supersede 上方 sigmoid 路线 LOCKED 判定）**：换 LIF colored-noise transfer（留在 Step 0a/0b 的最小手术）后，**Step 0b 通过 mechanism-scale gate**——LIF-rate field 上有限脉冲产生「点着 → 定向传播（移动波前、顺序招募 = 模板）→ 自终止」事件；sigmoid 路线结构性失败（放不出低静息高增益态 → 不传播）。**口径锁死：Step 0 过的是 mechanism-scale gate，不是 patient-fitted quantitative match**——证明「这种机制能产生我们需要的事件类型」，**不**证明每个时长/速度/范围已拟合真实数据（数据噪声重；IEI/事件间过程本轮未建模，属 Step 1+）。**尺度一致即可**（~70/110ms、cm 级推进、几十 ms 通道展布，同一生理量级）。
> **Step 1 解锁，带两个候选窗进入**：(a) **Brunel-like 分支（主线）**——短(~70ms)、稳健、宽窗、recovery-off，自限靠亚临界回落（参数少、不靠 recovery 精调）；(b) **病理高激 + 快回拉分支（机制 sensitivity）**——时长更贴 envelope(~110ms)，窗窄、recovery 必需（Pinto–Ermentrout 支）。
> **Step 1 验收（粗量级、不精拟合）**：无噪声不自发持续响 + 加噪声出现**离散**事件（非持续振荡）+ 事件能自终止 + 传播方向/通道顺序能被真实 pipeline 稳定读出 + 事件率落到真实量级附近。
> **Step 1 锁定合同（2026-06-03，开工前冻结，用户 6 条加固落数值）**：锁死事件检测器（活跃阈/最短时长/合并间隙/自终止/runaway/sustained/captured-high）+ seed×amp 网格报**比例**（不挑幸运 seed）+ **触发率曲线找宽区间**（不调单点凑率=避免循环论证）+ 窗口 B 回低-root + recovery-off-被俘获失败对照（B 仍是 sensitivity）+ 方向**先场后真实 pipeline**（事件不足不强行聚类）+ **噪声下** isotropic+aligned-shaft 必须过不了。详见 `docs/archive/topic4/sef_itp_phase4_v2/step1_noise_contract_2026-06-03.md`。
> **保留风险（= Step 1/2 sensitivity，不是 Step 0 阻断项）**：速度-尺度张力（regime b 波前 ~0.9mm/ms → ~10cm reach、patch 渡越 ~30ms 偏快）；recovery 窗窄；定量时长未完全锁定；0a 的近临界 Hopf 是阻尼迭代假象（真工作点稳健亚临界、自限传播是 excitable-from-stable-rest 而非 Hopf）。详见 `docs/archive/topic4/sef_itp_phase4_v2/lif_transfer_route_2026-06-03.md`。
> **数学路线更新（2026-06-03 晚，user 理论重新推导 + agent 的 near-critical→excitable 经验更正）**：Topic 4 主模型从「sigmoid rate field（`F_eff`）」升级为「**LIF-derived rate field（`Φ_LIF(μ,σ)` Siegert transfer）**」。逻辑链「population transfer → gain → linear stability → finite-pulse」不变，只换 transfer 实现；`F_eff` 降级为 optional coarse-graining，低异质性**后置**为「LIF 参数分布 p(θ) 收窄、经 `Φ_LIF`/gain/`λ(k)` 检验」（不再硬编码入 Step 0）。机制：sigmoid 低率 gain→0（卡死），LIF 近阈噪声区低率仍高 `∂Φ/∂μ`（活）。**关键更正（经验强制）**：fsolve 后真 LIF 工作点**稳健稳定**（max Re λ≈−0.05、loop gain≈0.58、k=0；**非 near-critical、无 finite-k Hopf**——之前的 Hopf 是阻尼迭代假象），而 self_limited_propagation 是**非线性可激**（全或无、波前推进幅度无关）→ **Step-0a 目标 = 稳健稳定但可激（非 η≈0+）；色散/finite-k-Hopf 是诊断不是闸门，finite-pulse 才是闸门**。**新 Step-0 工作**：(1) **0d 各向异性旋转控制 ✅DONE PASS**（θ_prop 随连接轴转 <0.1°、isotropic 无轴 ratio 1.00；承重判据通过；`scripts/sef_hfo_step0d_anisotropy_control.py`，commit c183ed6 + canonical 化 e95af61）；(2) ✅**`Φ_LIF` 已收进 canonical `src/sef_hfo_lif.py`**（新模块 + TDD，commit 46f9040；4 项工程加固 commit e95af61：w_ee_mult 贯穿场 / 0d 单轨化 / mean_field 多初值取最低-nuE root 并暴露 bistability / 0d 判据测试）；(3) ✅σ-dynamics sensitivity（拉长时长 110→151ms，不解速度-尺度张力，commit f86e73a）；(4) 0e heterogeneity 后置层（Φ_LIF 在多维 θ 分布上积分）= **deferred/recorded**（task 23）。详见 `docs/archive/topic4/sef_itp_phase4_v2/lif_rate_field_theory_2026-06-03.md` + Step-1 合同 `step1_noise_contract_2026-06-03.md`。
> **v0.2 核心边界**：只解释 HFO 群体事件的中观传播组织（event envelope、通道激活顺序、rank template、forward/reverse、identity bias、endpoint geometry、rate-geometry decoupling）；ictal-like recruitment 只作 synthetic feasibility bridge，不解释 clinical seizure onset；不解释 HFO 80-250/500 Hz carrier 的细胞生物物理，不把 template source 拟合成 clinical SOZ。
> **v0.2 承接**：保留 v1 已锁定的真实数据验收合同、cohort tier、phantom-rank 修复纪律、clinical SOZ 不作为拟合标签的红线；替换的是建模机制路线，不是 Topic 1 的实证发现。
>
> **历史状态（v1，保留审计链；建模主线已被 v2 supersede）**：**v1.0.7 lock 2026-05-23 (Phase 2 cohort 收口 + user-return v3 catch ratified)**。**v1.0.7 sub-amendments**: §3.4 H4 v1.1 Stage B scope strict-limit ("背景间期 30-min epochs rate-geometry decoupling", 不蕴含 ictal-adjacent / SOZ claim) + decision-k drift recast as Phase 3 biomarker feasibility (不入 Phase 2 verdict) + MEB k>3 caveat (Phase 3 启动前必须补 4-point sphere 或 MEB 降级 sensitivity); §3.5 H5 reporting unit per-seizure primary + cohort inference hierarchical/subject-clustered + stratify by seizure type/pair/interval + Δk + Δk/baseline_k + spatial radius 联合报 (decision_k_std cross-subject 不可直接比); v1.0.6 sustained: §3.3 H3 三层 verdict (R1+R2+sub-shape) + §3.4 H4 current SUPPLEMENTARY + §3.5 H5 主问题 pivot + 4 metrics tier。**Pending user ratify**: §3.5 H5 verdict 阈值具体值 (≥2/3 primary 或 ≥3/3 primary); §3.3 short-scale POSITIVE memory variant 是否单独 sub-label; H4 I_rate matched null spec (see spec_amendment_2026-05-23.md); endpoint stability OR vs AND combinator 永久锁; epoch_hours 默认 0.5 vs framework prose 1–2h; §3.5 H5 hierarchical model form (LMM vs GEE vs cluster-robust SE)。前置版本 v1.0.6 lock 2026-05-23 (structural clarification); v1.0.5 lock 2026-05-22; **Phase 0 解锁 2026-05-21（lagPatRank phantom-rank broad re-derivation Step 5a–5h 全部完成 + Checkpoint A/B advisor consult 通过）**
> **Phase 2 完结 marker**：Stage B H4 v1.1 cohort 跑通 (23/23 subjects, B1 calibration 0/23 mismatch), 主结论 = "背景间期 30-min epochs, 标准化事件率不稳定度显著高于 rank-endpoint identity 不稳定度 (p=1.4e-6, d=+1.26), participation field 对照同向 (p=9.5e-7, d=+1.50)"。Stage B decision-k drift 作 Phase 3 biomarker 可行性验证, 不入 Phase 2 cohort verdict。下一步 Phase 3 (H5 v1.0.7 spec) per-seizure primary + subject-clustered inference。详见 `docs/archive/topic4/sef_itp_phase2/cohort_run_2026-05-24.md`。
> **版本历史**：v1 lock 2026-05-20 (initial) → v1.0.1 advisor 5-fix 同日 → v1.0.2 user review 6-fix 同日（H2 工程不可行修复 + H1 概念冲突拆分 + H3 verdict naming 修订 + H4 normalized instability + H5 统计合同收紧 + Topic 1/3 cross-doc 实际落字）→ **Phase 0 完成 2026-05-21**（Topic 0 §3.1 phantom-rank 修复跑完所有下游 PR）→ **v1.0.3 lock 2026-05-22**：H2 prose 改 PR-6 swap_check ingest（CLAUDE.md §5+§6 错误三段修复）+ H1 v1.0.7 null pool 扩到全 SEEG implantation → **v1.0.4 lock 2026-05-22**：H2 input order 收紧到 rank-displacement `swap_sweep` variable-k，PR-2/PR-6 fixed top-3 降为 funnel/audit → **v1.0.5 lock 2026-05-22 user tier 拨正**：H1 从 cohort claim 候选 → **sanity / 必要前置**（任何结构化传播模型都会预测端点紧凑，不区分 SEF-ITP）；H2 从 mechanism sanity → **primary cohort claim**（同一空间结构被双向读取的反直觉预测才是 SEF-ITP 真正与替代模型分离的指纹；n=23 cohort 上 source 19/23 binomial p=1.3e-3 vs 50% null、sink 16/23 p=4.7e-2，足以支持 cohort 级别判读；PR-6 plan §3.3 的 mechanism-sanity 锁是针对 PR-6 自己的 n=6 8-subset 测试，**不**继承到 Topic 4 的 n=23 rank-displacement spatial 测试） → **v1.0.6 lock 2026-05-23 (user-return v2 catch surgical clarification)**：
> - **§3.3 H3 scale-stratified amendment**：v1.0.5 字面 "全部 6 条 mark-transition TOST equiv_pass" AND-rule 把生理上的短尺度 refractoriness anti-clustering 作为 framework failure 触发，与 SEF-ITP 自己的 biophysics 内部矛盾。v1.0.6 surgical clarification 拆三层：**R1 (long-scale ≥60s independence required)** + **R2 (endpoint identity stability required)** + biophysical sub-shape (short/mid-scale burst/refractory) descriptive only；原 v1.0.5 AND-rule 保留作 audit trail；详见 §3.3
> - **§3.4 H4 当前 implementation 降级 SUPPLEMENTARY + H4 v1.1 stub**：`compute_local_endpoint` 用 `events_bool[labels==c].mean(axis=0)` 取 participation rate top-k，不是 propagation rank top-k；v1.0.6 起当前 implementation 降级为 supplementary "participation field stability"；H4 v1.1 (rank-based endpoint + per-side spatial radius metrics + decision-k drift, **all 23 subjects computed, stratified by swap_class — v1.0.7 sub-amendment 修正自原 9/23 gating**) Stage B 实施，结果记入新 archive `docs/archive/topic4/sef_itp_phase2/cohort_run_2026-05-24.md`；详见 §3.4
> - **§3.5 H5 主问题重定向**：v1.0.5 "ΔS > 0 AND ΔR > 0 fraction" 没测空间扩散 / endpoint 招募，把"单纯 rate 升高 endpoint 不变"误判为 NULL；v1.0.6 主问题改为 "发作邻近是否出现 endpoint/swap-core 招募或空间扩展，而不是单纯 rate 升高"；4 指标：swap-k node identity Jaccard 下降 (recruitment) + swap-k 空间半径 source/sink 各自扩张 + decision-k Δ 上升 (core recruitment) + HFO rate Δ secondary 标记；统计合同 (time-of-day matched baseline ≥12h, min 30 events, subject-level aggregation, power floor 6 subjects) 沿用 v1.0.5 lock；详见 §3.5
> - **Banner bump trigger**：`docs/archive/topic4/sef_itp_phase2/cohort_run_2026-05-23.md` 顶部 user-return v2 catch banner (`temp_claude_log.md` 2026-05-23 user 长 message) ratifies above three amendments；archive body 保留作为 v1.0 字面 rule 跑出的 v1.0 cohort 历史快照，cross-cite 指向本 v1.0.6 amendment
> **取代**：`docs/paper1_framework_sba.md` 里的 BHPN-toy 数学模型部分（§4 toy model 数学规范 + §6 fitted model BHPN-fit spec 的 toy-mechanism 子句 + §7 dumb baselines 中针对 toy 机制的项）
> **保留**：`paper1_framework_sba.md` 里 P1 / P2 实证 verdict + P3 PR-7 addendum 锁字 + P5 directionality（sin-based）红线
> **硬前置（已解锁 2026-05-21）**：~~`docs/topic0_methodology_audits.md` §3.1 `lagPatRank` phantom-rank 修复 + §5 broad re-derivation roadmap 完成是所有 Phase ≥ 1 的不可越闸门~~ → **Phase 0 完成验收：所有 Phase ≥ 1 实验可以启动**。phantom-rank 修复结论（按 H1–H6 对应）：
> - **H3 (coordinate-free λ₂ metastability) 实质加强**：orig 10/34 (p<0.001) → masked **13/34**（5 个原 NULL/borderline 新增显著：pengzihang/253/442/958/1096；0 个反向丢；like-for-like n=33 子集 9→12）。详见 `docs/archive/topic0/lagpat_phantom_rank/step5h_topic4_attractor_results_2026-05-21.md`
> - **H4 (slow rate modulation; PR-5-B post-ictal recruitment)**：dominant_rate 主信号 `+65.46 events/h, p=0.0013` → `+65.66 events/h, p=0.0004`，direction + magnitude 都保持。详见 `step5e_pr5_results_2026-05-21.md`
> - **H4 联动 (PR-4B L3 high-conf Pearson r)**：orig 唯一显著 finding (n=8, p=0.016, 7/8) **修过版不复现** (p=0.547, 5/8) → 归 fragility-on-small-n，**不**进 H4 evidence base
> - **H1/H2 (PR-6 endpoint anchoring + fwd/rev swap)**：H1 cohort NULL 不变；H2 fwd/rev swap 9/9 → 8/8 都 positive (−1 subject from 5b)；Step 6 held-out swap_class concordance **0.69 → 0.82 实质提升**；rank displacement F_norm/τ/ρ 几乎完全不变。详见 `step5f_pr6_results_2026-05-21.md`
> - **PR-7 antagonistic temporal pairing (H3 mark-independence 联动) ✅ 完成 2026-05-22**：H1 triple-gate cohort-level NULL 不变；**P3 framework-flip gate CLEAR**：on orig P3 cohort (n=6) 用 masked features 重跑，verdict = **INCONCLUSIVE 完全保持 (4/4 flag 与 orig 一致)**；broader cohort (n=8 或 n=30) verdict 形式上滑到 NULL 但归 cohort × power × 5b fwd/rev 翻动互相作用非 phantom-rank 在 P3 statistic 的直接效应。`docs/paper1_framework_sba.md` v1.1.2 **不修订**。详见 `docs/archive/topic0/lagpat_phantom_rank/step5g_pr7_results_2026-05-21.md`
> - **Cohort tier 数字 lock**：Tier 1 cohort 大小见下方 §5；epilepsiae_916 已确认 cohort exit（stable_k 2→4），其他 34 stable_k=2 subject 全部进 Tier 1
> 
> **Phase 0 verdict（一句话）**：phantom-rank 修复**实质性加强**了 SEF-ITP framework 的两条关键 H3/H4 证据；**一条 PR-4B exploratory 翻转**被识别归类（fragility-on-small-n，不入主结论）；H1/H2 大方向都保持，其中 H2 mechanism 信号略缩 magnitude 但方向 100% 保持。
> **范围**：Topic 4 模型层 + Topic 1 within-event 现象的机制解释
> **不属于**：Topic 2 事件间周期（→ Paper 2）、ictal phase transition（→ Paper 3）、HFO 80–250 Hz carrier 振荡机制层

> **历史 v1.0.8 banner amendment 2026-05-27（已被 v2 建模路线 supersede；保留为 HR/FHN route audit trail）**：scope = Phase 4 modeling track 启动条件 + 节点动力学选择 + shaft 控制 + cluster 判据 + aniso D 进 stage 2 + smoke-first 网格 + 3-proxy sensitivity + adaptive gate + event extraction sensitivity sweep。修订 §6.5 三处：
> 1. **Phase 4 prerequisite**：v1.0.7 "Phase 1+2 H6+H1+H3 PASS + Phase 3 NULL" → v1.0.8 "framework structural prerequisite met (phantom-rank 修复 done + Phase 1 runner 落地 done) 即可启动 Phase 4 modeling track；model 输出作 mechanism exploration 层，**不替代** cohort verdict"
> 2. **节点动力学**：v1.0.7 "FHN，不加 Hindmarsh-Rose / Epileptor" → v1.0.8 "**HR (Hindmarsh-Rose) 主**，FHN / 简化 excitable unit 作 sensitivity；不模拟 HFO carrier 80–250Hz"
> 3. **Shaft 采样 + cluster 判据**：v1.0.7 未规定 → v1.0.8 "**强制 shaft 控制**：aligned / orthogonal / offset / angled / random / jittered (≥6 几何) + isotropic θ negative control；**禁止单用 KMeans k=2 作机制成功**；必须并列 split-half + odd-even + forward/reverse Spearman + rank-displacement swap_sweep + principal-curve audit；**isotropic θ + random shaft 也不能通过这套**；3 proxy (P1=x, P2=dx/dt, P3=detrend envelope) 全 PASS 才进 framework mechanism support 栏（取严格版）；L4 principal-curve audit 是 descriptive classifier 不是 hard gate"
>
> 完整 v1 spec 见 `docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md` v0.2。本 banner block 是 v1.0.7 doc 的 surgical 修订，§6.5 内容 v1.0.7 prose 不删除，下方 § 6.5 文末加 "→ v1.0.8 banner amendment 修订上面 3 项" cross-link。

> **历史 Stage 1b 观测单位锁定 2026-05-28（HR route only）**：Phase 4 modeling track 的**节点事件单位 = burst envelope（一簇 spike 合并，事件时刻 = 第一个 spike 起点）**，不是单个 spike excursion；spike-level 仅作 within-burst secondary。理由：真实数据对象是 HFO / group-HFO，不是单神经元 spike；HR 节点被选正因其 bursting。单节点 baseline (I=1.0, r=0.006) 标定确认 σ=0 静息、σ>0 事件稀疏短暂 (excitable-like)，且 envelope 与 spike 确为不同单位 (n_spikes/burst 1.5→2.6)。merge 阈值 `envelope_gap=30` 取自实测 inter-spike gap 谷底，非调参。证据：`docs/archive/topic4/sef_itp_phase4_v1/stage1b_results_2026-05-28.md`；Stage 2+ group event = 多节点 burst-envelope onset 短窗共现。

---

## 0. 一句话承诺（朴素表述，CLAUDE.md §8 大白话风格）

我们假设：**间期 HFO 群体事件不是每次随机走一条新路，而是同一块病理组织附近存在一条固定的传播高速路**。这块组织内部神经元更相似，但“更相似”本身不自动等于“更危险”；它必须先改变群体输入-输出曲线，再改变局部线性稳定性。噪声偶尔点燃这条路，活动沿轴传播后自限熄灭；慢变量把系统推得更危险时，事件出现得更频繁，但不应该重写传播路线。

如果这个图景正确，我们应该看到：通道传播顺序稳定、正反模板共享同一条空间轴、同一 subject 内通道身份偏置很高但高于几何采样对照、事件率可被慢变量调制但传播几何相对稳定。发作样持续招募只作为仿真可行性桥接，不作为真实发作起始解释。

如果这些现象在 synthetic data 里只有靠手工调参或人工挑图才能出现，模型失败。所有 synthetic data 必须走真实模板 pipeline；clinical SOZ 只作 held-out 关系检验，不作反向拟合标签。模型**不**解释 HFO 80-250/500 Hz carrier 振荡是哪来的。

---

## 1. 为什么换掉旧模型

### 1.1 旧 BHPN（Bidirectional Hebbian Phase Network）的根本问题

旧 plan（`docs/archive/topic4/pr_t4_1_bhpn_toy/pr_t4_1_bhpn_toy_plan_2026-05-01.md`，已 SUPERSEDED）的设计是：

1. 把"目标通道顺序"预先塞进一个对称 Hebbian 连接矩阵 `A_ij = ρ · cos(φ*_i − φ*_j)`
2. 让 Kuramoto 方程演化
3. 数学保证它会落到 `φ*_F` 或 `π − φ*_F` 两个对偶 fixed point
4. 宣称"模型复现了正反模板"

**这是循环论证**：

- 现象（两个互逆模板）已经被预先编码进矩阵 `A_ij`
- 模型只是把预先编码的东西"演化"一次显示出来
- **不解释**为什么是这套通道、为什么是这个顺序、为什么发生在 SOZ 区域

这和老 KONWAC paper 犯的是同类病：数学漂亮，机制空洞。BHPN-toy 即使把所有 11 个单元测试 + 7 个集成 sanity 跑过，对评审而言它**仍然只是一个把"对称双稳态"重新画了一遍的装饰**——不是机制解释。

### 1.2 v2 新方向：低异质性 + 各向异性连接 + 近临界 E-I 易激场

朴素思路：

- 间期事件**不是**某个内禀振荡器在跳动，也不是每次随机换路。
- 病理组织附近有一块局部低异质性的 E-I 易激斑块：同类神经元阈值更相似；这个变化必须先进入群体输入-输出曲线，再由实际线性化结果决定它是否更接近临界。
- E→E 连接核是各向异性的；沿长轴传播更容易。这条长轴就是正反模板共享的传播高速路。
- 斑块处在近临界但仍亚阈值的候选工作窗：线性稳定性只给地图，有限幅脉冲仿真才证明它能被点燃并自限熄灭。
- 慢变量主要控制事件被点燃的**频率**，不应该重写传播几何；如果慢变量一加进去模板就完全换路，模型失败。

这个思路把"空间几何"和"E-I 稳定性"同时放回机制核心。它比 v1 的抽象扩散 / HR toy 更 sharp，因为它给出三条可直接证伪的机制杠杆：

- 各向异性轴旋转，模板方向必须跟着旋转。
- 降低局部 threshold variance 后，必须先计算 gain 和线性稳定性是否真的改变；若改变方向成立，成核点才应向该 patch 聚集，而不是只增加全局同步。
- 慢变量改变事件率时，rank geometry 应保持稳定。

v1 已经锁定的真实数据判据继续保留；v0.2 替换的是机制实现路线：先做 effective gain、线性稳定性和 finite-pulse response，再进入 rate field / LIF SNN，而不是直接把 HR/FHN 节点拼成 toy movie。

---

## 2. 核心方程（v2 最小可行形式）

> **2026-06-03 晚更新（见顶部 banner「数学路线更新」+ `docs/archive/topic4/sef_itp_phase4_v2/lif_rate_field_theory_2026-06-03.md`）**：下方 `F_eff`（阈值分布平均的 sigmoid）**已降级为 optional coarse-graining**；主 transfer 改为 **LIF Siegert `Φ_LIF(μ,σ)`**（功能位置相同：input→population transfer→rate；但 LIF 在低静息率仍有高 `∂Φ/∂μ`，sigmoid 没有）。下方 prose 保留作 audit trail + heterogeneity 后置层（`Φ_eff_LIF=∫Φ_LIF(μ,σ;θ)p(θ)dθ`）的概念前身。另：本节标题"近临界工作窗"措辞经更正——真 LIF 工作点是**稳健稳定但可激**，非 near-critical。

第一版不直接上 SNN。先用二维 E-I rate field 找近临界工作窗：

```text
tau_E * dr_E/dt = -r_E + F_E(W_EE * r_E - W_EI * r_I + I_E)
tau_I * dr_I/dt = -r_I + F_I(W_IE * r_E - W_II * r_I + I_I)
```

对均匀稳态线性化，定义局部离失稳边界的距离：

```text
F_a_eff(h; x) = integral f_a(h - phi) p(phi; phi_bar_a(x), sigma_phi,a(x)) dphi
G_a(x) = dF_a_eff(h; x) / dh | h=h0(x)
eta_lin(x) = - max_k Re(lambda(k; G_E(x), G_I(x), W_hat(k), tau_E, tau_I))
```

其中 `eta_lin(x) >> 0` 是小扰动稳定区，`eta_lin(x) approx 0+` 是候选可激窗，`eta_lin(x) < 0` 是小扰动失稳区。`sigma_phi(x)` 不能直接等同于 `eta_lin(x)`；只有当前工作点下实际计算显示 gain 改变使 `eta_lin(x)` 下降，才能说低异质性让 patch 更接近临界。

| 符号 | 朴素含义 |
|---|---|
| `r_E, r_I` | 局部兴奋 / 抑制群体活动 |
| `W_EE` | E→E 连接核；v2 主模型中是椭圆形各向异性核 |
| `W_EI, W_IE, W_II` | 抑制相关连接；抑制核更宽，并保留全局刹车 |
| `sigma_phi(x)` | threshold variance；病理 patch 内降低，表示局部低异质性 |
| `F_eff, G` | 阈值分布进入群体输入-输出曲线后的有效响应和局部 gain |
| `eta_lin(x)` | 由 gain、E/I 参数、输入、慢变量和连接核共同决定的线性稳定性读数 |
| `xi(x,t)` | 噪声触发源，决定哪次事件从哪里成核 |

**关键 framing（v2 反调参锁）**：

- `sigma_phi(x)` 是具体机制；`eta_lin(x)` 是稳定性读数，不是另一个任意病理场。
- 线性稳定性只回答小扰动地图，不证明有限幅事件。自限传播必须由 finite-pulse response map 验证。
- `k=2` 和 raw identity bias 只能作描述性输出；主证据必须高于几何采样 controls。
- 第一版只降低 threshold variance，不同时调 `J_EE`、`E_L`、adaptation、外部输入。
- 各向异性轴、低异质性 patch、慢变量必须做独立扰动实验；每个扰动都要有失败模式。
- 模型**只解释**通道点火的空间-时间次序和群体事件组织，**不解释** HFO carrier 的细胞层振荡来源。

如果不写清楚这一点，模型会退化成“给一个高易激场，然后预测高易激场被点燃”的同义反复。

---

## 3. 六条预测（H1–H6，pre-registered）

每条用三段式：**测了什么 / 怎么测的 / 揭示了什么**。所有数字判据 lock at framework time，**不允许事后调整**。

### 3.1 H1 — Endpoint 是结构化锚点（**sanity / 必要前置**，非 SEF-ITP 区分性预测）

> **v1.0.5 tier（2026-05-22 user 拨正）**: H1 不是 SEF-ITP 的 discriminative 预测——任何"有结构传播"的模型（独立稳定锚点、单向行波、多焦点同步、SEF-ITP 都包括）都会预测 source 通道和 sink 通道各自空间紧凑。H1 在这里的角色是 **prerequisite sanity**：如果连这个都不成立（endpoint 全随机散落），SEF-ITP 连测试条件都不满足，下面 H2/H3/H4 都没意义。**H1 报告 cohort 数字描述（如 31/46 pass-like）作为"前提条件已具备"的事实陈述，不作为支持 SEF-ITP 的核心证据**。SEF-ITP 真正的区分性预测在 H2（source/sink 在两个反向模板间互换）和 H3（同一空间结构被多次独立采样）。

**测了什么**：endpoint 不是散布全脑的随机通道；而是**空间结构化**——分三个独立子测，避免与 H2 概念冲突。

**为什么必须拆**（v1.0.2 修订）：原 v1 把紧凑性定义为 `C_E = mean pairwise distance within (S ∪ K)`。但如果 H2 期望成立（source/sink 是同一条空间轴的两端），那 `S ∪ K` 本身**应该是 elongated 而不是 compact**——v1 度量会把"轴两端结构"误判为"不紧凑"，与 H2 互斥。v1.0.2 拆三层，**每层都与 H2 概念正交**。

**怎么测的**：

#### H1a — within-source compactness

- 每个 cluster 的 source 集合 `S = top-3 source channels`（k=3 lock）内部紧凑
- 度量：`C_S = mean pairwise distance within S`，3D Euclidean 主 + shaft-ordinal / cortical surface / SC distance 并列 sensitivity
- Null：matched random 3-channel sampling 1000 次；匹配条件 = shaft 分布 + participation rate + HFO rate（小 subject 退化规则同 §3.2 旧 H1）
- 检验：subject-level one-sided `C_S < C_S_null`；cohort Wilcoxon

#### H1b — within-sink compactness

- 每个 cluster 的 sink 集合 `K = bottom-3 sink channels` 内部紧凑
- 度量 / null / 检验：同 H1a，但作用在 K 上

#### H1c — endpoint envelope within pathological participation field

- 检验 `E = S ∪ K` 是否位于高参与场内部（不脱离 pathological field）
- 度量：
  - 主：`ratio = mean(d(endpoint_i, centroid(high-participation channels))) / mean(d(non-endpoint participating ch_j, same centroid))`
  - 备选 sensitivity：endpoint convex hull volume / participating channel convex hull volume
- 期望：`ratio ≈ 1`（endpoint 嵌在参与场内部）；`ratio >> 1` 意味 endpoint 脱离参与场
- Null：在 valid_mask=True 参与通道里随机抽 |E| 个，重算 1000 次

**揭示了什么**（H1 整体 PASS / NULL / FAIL 联合解读）：

- **PASS**：H1a 与 H1b 都 PASS（cohort Wilcoxon p<0.05 且 ≥ 3/4 距离同向）+ H1c 不 FAIL → endpoint 是结构化锚点
- **partial PASS**：H1a 或 H1b 一过一不过 → 单端紧凑、另一端散；可能 reverse template 不稳定，或 sink 定义出问题
- **NULL**：三层都不显著 → endpoint 作为锚点的几何意义弱
- **FAIL**（H1c FAIL：endpoint 大幅脱离参与场，`ratio >> 1`）→ endpoint 是分隔群体，**空间场假说受重创**

**敏感性**：每条 H1 子测都额外做 SOZ-mask 内随机抽样（控制 SOZ 选择偏差），并报告四种距离度量（Euclidean / shaft-ordinal / cortical surface / SC via DTI）下结论是否一致。

### 3.2 H2 — Source/sink reversal geometry (rank-displacement swap-k labels)

> **v1.0.4 调查结论（2026-05-22）**: H2 有三层数据，不能混用。**PR-2 / PR-2.5** 回答"有没有两个稳定传播模板"，是模板发现 / 复现层；`candidate_forward_reverse_pairs` 只是描述标签，不是 channel-level H2 输入。**PR-6 endpoint anchoring** 回答"每条模板 fixed top-3 source/sink 是谁"，是端点摘要层，比 PR-2 更接近 H2，但仍是 fixed-k 抽象。**PR-6 supplementary rank-displacement `swap_sweep`** 回答"每个通道在两模板之间怎样换位，哪个 `decision_k` 最能解释 source/sink 反转"，是 per-channel + variable-k + family-wise null 层；这是 Topic 4 建模前最稳的 H2 channel-label 来源。
>
> **当前 H2 input source order**: 用 masked rank-displacement per-subject JSON：`results/interictal_propagation_masked/rank_displacement/per_subject/<dataset>_<subject>.json::primary_pair.swap_sweep` (`swap_class`, `decision_k`, `T_obs`, `p_fw`)；channel label 用 `joint_valid` + `rank_a_dense_full` 经 `src.rank_displacement.derive_swap_endpoint` 派生。PR-2/PR-2.5 只作 provenance / funnel，不作为 H2 label。PR-6 fixed top-3 endpoint 与 `h2_swap_check` 保留为摘要层和 audit trail。**注意：这一步只定义 swap-k 节点，不等于 H2 空间性检验。**
>
> **v1.0.3 修订（2026-05-22）**: H2 不在 Phase 1 内重新发明 set/spatial reversal — 直接 ingest PR-6 anchoring 已锁定的 `h2_swap_check` (per-contact endpoint Jaccard 反向 + 1000-perm null per subject; PR-6 plan §3.3 contract)。v1.0.4 在此基础上进一步收紧：Topic 4 建模前若需要 channel-label，应优先取 rank-displacement swap-k，而不是 PR-6 fixed top-3 或 PR-2 candidate。下方早期 v1.0.2 prose "主分析 1: Set-based reversal Jaccard" + "主分析 2: Spatial reversal centroid distance" + "Sensitivity: PCA axis projection" **已弃用**，保留只为 commit history 可追溯。
>
> **H2 空间性主检验**: rank-displacement 的 `decision_k` 先拆出 T_a source-side 前 `k` 个通道与 sink-side 后 `k` 个通道；随后分别把这两组 `k` 个 swap 节点作为 `members`，在 **all mapped SEEG minus swap endpoint** 的 pool 上无 shaft 约束随机抽同样大小节点集，形成 compactness null。这里不能沿用 H1 的 same-shaft null：H2 问的是 swap-k 节点相对全植入空间是否特殊，source/sink shaft 本身就是机制的一部分，强行 same-shaft 会把 null 抽死并重新制造特殊情况。`source_side` 与 `sink_side` 是主 sanity；`combined_endpoint` (`2k`) 只作辅助描述，因为 source 与 sink 两侧可以天然相距很远，不能把 2k 混成唯一主检验。
>
> **判读 tier（v1.0.5 拨正 2026-05-22）**: H2 是 **primary cohort claim — SEF-ITP 真正反直觉的、与替代模型分离的指纹**。同一空间结构被双向读取意味着 source 节点和 sink 节点不是两个独立的解剖区，而是一对耦合的端点；任何独立稳定锚点 / 多焦点同步 / 单向行波模型都不会预测 source 和 sink 各自空间紧凑这件事在同一组 swap-k 节点上同时成立。**早期 v1.0.3 把 PR-6 plan §3.3 的 mechanism-sanity 锁直接套到 Topic 4 H2 是 cohort 转移错误**：PR-6 plan §3.3 锁的是 PR-6 自己的 n=6 8-subset 测试（PR-2.5 strict subcohort），统计功效根本不允许 cohort claim；Topic 4 v1.0.4 H2 是不同的测试（rank-displacement swap-k + spatial compactness）跑在不同的 cohort（n=23），需要重新评估 tier。n=23 上 source 19/23 binomial p=1.3e-3 vs 50%（严苛的 coin-flip 基线），sink 16/23 p=4.7e-2，both 13/23 p=1.2e-3 vs 25%（双侧独立基线）——足以支持 cohort-level "spatial compactness on swap-k source AND sink sides" claim。
>
> **报告契约 lock**：
> - per subject 报 label 层 (`swap_class` / `decision_k` / `T_obs` / `p_fw` / `swap_endpoint_channels`) + spatial 层 (`source_side` / `sink_side` / `combined_endpoint` compactness verdict)
> - cohort 报：`swap_class_distribution`（label 层 strict / candidate / none 比例）+ 每侧 spatial compactness PASS 比例 + binomial sign-test p 值（against 50% coin-flip null AND against 5% per-subject 5%-α-only null，两个都报）+ "both sides PASS" 联合比例
> - **允许报 cohort-level claim**：在锁定语言下，e.g. "在 n=23 cohort 上，rank-displacement 派生的 swap-k source-side 与 sink-side 节点 each 显著空间紧凑（binomial p<0.05 vs 50% coin-flip null）；13/23 subject 双侧同时 PASS（p=1.2e-3 vs 25% independent baseline）。这支持 SEF-ITP 的对偶端点预测。"
> - **不允许**：报 "H2 PASS" 当作 binary 整体裁决（信息丢失太多）；报 "100% PASS" 或"每个 subject 都通过"（实际是 19/23 + 16/23，要保留分母）；从 spatial 层 PASS 直接跳到 "source/sink reversal 是 cohort 主效应"（spatial 层是必要不充分，label 层 swap_class strict-or-candidate 仅 9/23 才是更狭义的 swap 信号）
>
> **input warning**: 不要从 PR-2 `candidate_forward_reverse_pairs` 开始抓 H2。PR-2 的整模板 Spearman 反向会被共享节点稀释；它只能说明"像正反模板"，不能告诉 Topic 4 哪些通道构成 swap endpoint。若只是复核 PR-6 `h2_swap_check`，可沿用 PR-2.5 `forward_reverse_reproduced` (split-half OR odd-even) ∩ PR-6 audit eligible；若进入 Topic 4 建模前 channel-label，必须转到 rank-displacement `swap_sweep` 的 variable-k 输出。
>
> 设计错误三段叠加（CLAUDE.md §5+§6 经典）：(a) cohort 用错字段（candidate vs reproduced）；(b) 重新发明 PR-6 已有 swap_score helper；(c) 升级 mechanism-sanity tier 到 cohort claim。三条全部修正在 v1.0.3。

---

**以下为 v1.0.2 早期 prose（已弃用，保留 audit trail）**：

**测了什么**：正反两个模板的 source / sink **角色互换**——即正模板里 "最先点火的几个通道" 应该和反模板里 "最后点火的几个通道" **空间上 / 集合上**对应，反之亦然。

**为什么换设计**（v1.0.2 修订）：原 v1 主分析是 PCA 主轴投影。但 Tier 1 eligibility `n_participating ≥ 6` + endpoint k=3 时 `|S ∪ K|` 可吃掉全部 6 个 valid 通道，**排除 endpoint 后 PCA 主轴几乎一定不可定义**（剩 0–2 个点）。v1.0.2 把主分析改为**不需要轴拟合**的 set-based + centroid-distance 两条 reversal index 测验；PCA 投影降级为高通道 subject 的 sensitivity。

#### 主分析 1：Set-based reversal Jaccard

- 每个 fwd/rev cohort subject 算：
  - `J_swap = (Jaccard(S_A, K_B) + Jaccard(K_A, S_B)) / 2`（反向配对 — 期望高）
  - `J_same = (Jaccard(S_A, K_A) + Jaccard(K_B, S_B)) / 2`（同模板首尾 — 期望低，因 source 与 sink 概念上互斥）
- Reversal Index: `R_set = J_swap − J_same`
- 期望：reversal 成立时 `R_set > 0`
- **Null**：在该 subject 的 `E_A ∪ E_B` 通道池里随机重新分配 4 个 role tag（S_A / K_A / S_B / K_B 各 k 个，互不重叠），重算 1000 次得 `R_set_null`
- 检验：subject-level one-sided `R_set > R_set_null_median`；cohort Wilcoxon signed-rank

#### 主分析 2：Spatial reversal centroid distance

- 每个 subject 算 4 个 endpoint 集合的空间 centroid：`c(S_A), c(K_A), c(S_B), c(K_B)` (3D Euclidean)
- 两种距离和：
  - `d_swap = d(c(S_A), c(K_B)) + d(c(K_A), c(S_B))`（反向配对 — 期望小）
  - `d_same = d(c(S_A), c(K_A)) + d(c(K_B), c(S_B))`（同模板首尾 — 期望大，因 source/sink 是轴的两端）
- Spatial Reversal Index: `R_spatial = d_same / (d_swap + d_same)`
- 期望：reversal 成立时 `R_spatial > 0.5`
- **Null**：同上 role tag 重新分配 1000 次
- 检验：subject-level one-sided `R_spatial > R_spatial_null_median`；cohort Wilcoxon signed-rank

**两条主分析联合**：两条都 PASS 才算 H2 PASS（防止单条 false positive）；一条 PASS 一条 NULL 算 partial PASS；两条都 NULL 算 NULL。

#### Sensitivity：PCA axis projection（仅高通道 subject）

- 仅在 `non_endpoint_participating_channels ≥ 4` 的 subject 上跑（**这是新的子 cohort 准入条件，见 §5.3**）
- PCA 主轴用 `valid_mask=True \ (E_A ∪ E_B)` 通道定义（排除 endpoint 避免 double-dip）
- 四组 endpoint 投到主轴，检验 `proj(S_A) ≈ proj(K_B)` 且 `proj(K_A) ≈ proj(S_B)`
- 报告：合格 subject 比例 + 合格 subject 内的 axis-based verdict
- **不作主结论**，只作高通道 subject 的辅助验证；若 sensitivity 跑了且与主分析方向一致 → 强化 H2 PASS；若方向相反 → archive note + 主分析仍以两条 reversal index 为准

**揭示了什么**：

- **PASS**（两条主分析 cohort Wilcoxon 都 p<0.05 同向；如 sensitivity 跑了方向也同）：endpoint 在集合与空间两层面都呈反转关系——强支持"同一空间结构轴被相反方向读取"
- **NULL**（两条主分析都不显著）：endpoint 在两模板间无系统性 reversal 关系
- **FAIL**（方向反转：`R_set < 0` 或 `R_spatial < 0.5`）：endpoint 反而**靠近同极而非互换**——反转几何被证伪

### 3.3 H3 — Mark independence with stable geometry

**测了什么**：连续事件挑同一个模板的频率是不是兼容独立抛硬币；同时 endpoint 集合跨时间稳定。

**怎么测的**（继承 PR-7 addendum 2026-05-01 锁字）：

- 模板选择独立性（marginal-aware null-relative，**不**预设 marginal 50/50）：
  - lag-1 same-template excess（vs N2 marginal-preserving null）
  - 10s / 30s / 60s / 1800s window excess
  - run_length lift
  - cohort bootstrap 95% CI 与 δ_excess = 0.05 比较
- Endpoint 几何稳定性：
  - split-half endpoint set Jaccard recall
  - odd-even block endpoint set Jaccard recall

#### **v1.0.6 surgical clarification 2026-05-23（user-return v2 catch ratified）— 三层 verdict**

User-return v2 catch (`temp_claude_log.md` 2026-05-23 user 长 message)：v1.0.5 字面 "全部 6 条 mark-transition TOST equiv_pass" AND-rule 等价于要求 SEF-ITP "在所有时间尺度上都不允许任何系统性偏置"。但 SEF-ITP framework 自己（"病理结构 + 局部扩散 + 不应期物理"）就预测 sub-1min 必然留下 refractoriness 反聚集痕迹 → v1.0.5 字面 rule 与 SEF-ITP biophysics 自身**内部矛盾**。Phase 2 cohort 实跑 (`docs/archive/topic4/sef_itp_phase2/cohort_run_2026-05-23.md`) 揭示该矛盾：10–30s anti-clustering 触发 "CONTRADICTED" 但方向与 framework 警告的 memory variant 反向。

**v1.0.6 surgical clarification 拆 H3 verdict 为三层**：

| 层 | 时间尺度 | 角色 | 期望方向 |
|---|---|---|---|
| **R1 (long-scale ≥60s independence, REQUIRED gate)** | 60s + 1800s window excess | 主 gate (必要条件) | excess ≈ 0; cohort 95% CI ⊆ ±δ_excess = ±0.05 |
| **R2 (endpoint identity stability, REQUIRED gate)** | split-half + odd-even subject-mean Jaccard | 主 gate (必要条件) | ≥1 of {split-half, odd-even} ≥ 0.7（OR combinator） |
| **Descriptive sub-shape (REPORTED, NOT gate)** | lag-1 + run_length (weak burst-like) + 10–30s anti-clustering (refractoriness) | 描述性，**不入 verdict gate** | **weak** burst-like tendency (small excess +, e.g. cohort lag-1 ~+0.016 and run_length lift ~+0.035) at lag-1 + run_length; refractoriness direction (−) at 10–30s |

**Verdict mapping (v1.0.6 lock)**：

- **SUPPORTED**：R1 ✓ AND R2 ✓
- **NOT_SUPPORTED_GEOMETRY_UNSTABLE**：R1 ✓ but R2 ✗ → 模板几何不可靠，回 Phase 0 查事件检测 / 聚类质量
- **CONTRADICTED**：R1 ✗（长尺度 mark transition CI 偏离 ±δ_excess，**任一方向** — positive long-scale memory 或 negative 长尺度反聚集都触发；LOO 仍 fail 才进 CONTRADICTED）

**LOO discipline (继承 v1.0.5)**：R1 fails 的 metric 还须 leave-one-subject-out cohort 仍 fails 才进 CONTRADICTED；单 subject sensitive 不进 CONTRADICTED。

**OR combinator (advisor catch A 2026-05-23 锁住)**：R2 用 `endpoint_jaccard_split_half OR endpoint_jaccard_odd_even ≥ 0.7`，与 CLAUDE.md / AGENTS.md cross-PR `forward_reverse_reproduced` = split-half OR odd-even 惯例一致；测试 `test_h3_integrated_verdict_or_combinator` 锁住。

**v1.0.6 措辞锁（继承 v1.0.5 红线 §8.9，扩展 scale-stratified 限定）**：

> ✅ "compatible with mark-independent sampling within tested precision **(long-scale ≥60s)**"
> ✅ "biophysically consistent refractoriness at 10–30s (descriptive)"
> ✅ "**weak burst-like tendency** at lag-1 + run_length (small effect size, e.g. lag-1 ~+0.016 / run_length lift ~+0.035 in current cohort; consistent with within-event co-firing; descriptive)"
> ❌ "strong burst" / "burst 主效应" — current cohort effect sizes 都很小，不能写得太满
> ❌ "证明独立 / proves mark-independence"
> ❌ "H3 simple PASS"（扁平 PASS 丢失三相 biophysics 描述）
> ❌ 把 short-scale anti-clustering 当作 CONTRADICTED 触发（v1.0.5 字面 rule 已 SUPERSEDED）

**为什么这不是事后调参拯救 verdict**：v1.0.6 amendment 后，falsifiability 完整保留 — (a) 长尺度 (60s + 1800s) CI 任一方向偏离 ±0.05 + LOO 仍 fail → CONTRADICTED；(b) endpoint 不稳定 → NOT_SUPPORTED_GEOMETRY_UNSTABLE。amendment 只是把 SEF-ITP 自己的 biophysics 预测 (short-scale refractoriness) 从 "失败信号" 改为 "描述性 sub-shape"，移除了与自身预测的内部矛盾。

**v1.0.6 不引入 short-scale POSITIVE memory variant 子 verdict (banner authorization scope lock)**：banner 只授权 scale-stratified sub-bullet (long + short descriptive + endpoint)；如果未来 cohort 出现 short-scale POSITIVE memory (与 refractoriness 反向 — 累积 memory at 10–30s)，是否需要单独 verdict label (e.g., NOT_SUPPORTED_SHORT_SCALE_MEMORY) 留待 user 单独 ratify。当前 v1.0.6 下，short-scale 不论方向都是 descriptive sub-bullet，只有 long-scale 触发 verdict。

**揭示了什么 v1.0.5 verdict mapping (SUPERSEDED by v1.0.6 surgical clarification above, kept for audit trail)**：

H3 的 verdict 标签**特别用** SUPPORTED / NOT SUPPORTED / CONTRADICTED，**不**用 PASS / NULL / FAIL——因为 "PASS for mark-independence" 会被简化为 "证明独立"（fail-to-reject ≠ accept null）：

- **SUPPORTED**：mark transition cohort CI ⊆ ±δ_excess（compatible with independence within tested precision）+ endpoint split-half recall ≥ 0.7 → 与 "随机触发 + 稳定空间骨架" 一致；**不是** "证明独立"
- **NOT SUPPORTED (geometry-unstable variant)**：mark transition compatible 但 endpoint 不稳定 → 模板几何本身不可靠，回 Phase 0 查事件检测 / 聚类质量
- **NOT SUPPORTED (memory variant)**：mark transition CI 偏离 ±δ_excess 但 endpoint 稳定 → 模型需要在 §2 方程里加 short-term memory / refractory / latent state 项；framework 仍活
- **CONTRADICTED**：cohort robust `|excess|` 显著 > δ_excess 且 leave-one-subject-out 仍 > δ_excess → 随机触发假说证伪；framework 改写或回退

**Migration note (v1.0.5 → v1.0.6)**：v1.0.5 的 NOT_SUPPORTED_MEMORY 子 label 在 v1.0.6 下归并入 CONTRADICTED（长尺度 positive memory 会让 R1 fail → CONTRADICTED）。若未来 user 希望保留 long-scale memory variant 与 short-scale anti-clustering 的细分，可单独 ratify 一个 sub-label。v1.0.5 SUPPORTED 子条件 "mark transition cohort CI ⊆ ±δ_excess" 在 v1.0.6 下狭义化为 "R1 (60s + 1800s) CI ⊆ ±δ_excess" — 短尺度不再作 SUPPORTED 必要条件。

**关键纪律**（继承 paper1_framework_sba.md §5.3 P3 v1.1.2 + PR-7 addendum 2026-05-01 lock）：

- H3 措辞硬锁 **"compatible with mark-independent sampling within tested precision"**
- **禁止**写 "证明独立" / "mark-independence PASS" / "证伪 ping-pong" 等 v1.0 错误语
- **禁止**用 leave-one-subject-out 替代 cohort 主判据
- v1.0.2 verdict naming（SUPPORTED 而非 PASS）从命名上就阻断 "PASS=证明" 的误读

### 3.4 H4 — Slow rate modulation 不变形 geometry

**测了什么**：当事件率慢漂（小时级）时，模板几何（centroid + endpoint set）是不是基本不变。

**为什么换设计**（v1.0.2 修订）：原 v1 用 `CV(rate) ≥ 3 × CV(geometry)` 硬阈值。问题：rate 和 geometry similarity 不同量纲；Jaccard / centroid similarity 的 CV 在小 N epoch 下不稳；硬锁 3× 是 framework time 锁了一个**可能没意义的比较**。v1.0.2 改成 "两个量各自对自己的 matched null 标准化后再比较"。

**怎么测的**：

- 把 24h（Yuquan）或多天（Epilepsiae）数据切成 1h–2h epoch（**保留时间顺序**，不打乱）
- 每 epoch 算：`rate(t)` + `template_centroid_similarity(t, global)` + `endpoint_Jaccard(t, global)`
- 计算 **normalized instability** per subject：
  - `I_rate = std(log(rate)) across epochs / sqrt(matched-null variance of log(rate))`
    - matched null: time-shuffle epoch order 1000 次，每次重算 std → 取分布的 variance 当作 baseline
  - `I_geom = std(1 − endpoint_Jaccard) across epochs / sqrt(matched-null variance)`
    - matched null: 每 epoch 内 role-shuffle endpoint 标签（在 valid_mask=True 池子里随机选 |E| 个当 "该 epoch 的 endpoint"），重算 1000 次得 null 分布
- subject-level 检验：`I_rate > I_geom`（one-sided）
- cohort 检验：Wilcoxon signed-rank on `(I_rate − I_geom)` per subject
- **效应大小**：Cohen's d on cohort `(I_rate − I_geom)` 值，median + IQR 同报；**不**锁固定倍数比

**揭示了什么**：

- **PASS**：cohort Wilcoxon p<0.05 同向 + Cohen's d ≥ 0.3（small effect floor）
- **NULL**：cohort Wilcoxon 不显著或 Cohen's d < 0.3 → rate 和 geometry 同尺度漂动，模型需把 geometry 也时变化
- **FAIL**：Cohen's d < 0 且 Wilcoxon p<0.05 反向 → geometry 比 rate 更不稳定，模型架构错

**联动对接**：PR-5 dominant-template absolute rate post-ictal 升高（已落 doc）应与 H4 PASS 一致——若 H4 PASS 而 PR-5 NULL，说明 rate 升高未到 geometry 变形阈值。**待 phantom-rank 修复后 §6.1 §5e 重跑确认 PR-5 方向是否保留**。

#### **v1.0.6 amendment 2026-05-23（user-return v2 catch ratified）— H4 当前 implementation 降级 SUPPLEMENTARY + H4 v1.1 stub**

User-return v2 catch (`temp_claude_log.md` 2026-05-23)：Phase 2 v1.0.0 implementation 里 `src/sef_itp_phase2.py::compute_local_endpoint` 把 per-epoch endpoint 定义为：

```python
ch_mean = events_bool[labels == c].mean(axis=0)
# top-k by mean participation = "高参与率通道"
```

这是 **per-epoch participation rate top-k**（哪些通道这段时间常常出现在事件里），**不是** SEF-ITP H4 原意预测的 **per-epoch propagation rank top-k**（哪些通道在事件里最先点火）。两者物理意义不同：

- 参与率 top-k = 通道在 epoch 内是否经常被采样到的"**参与场**"
- 传播 rank top-k = 通道在事件里最先点火的"**传播端点**"

`load_subject_propagation_events` 返回的有 `ranks` 字段（per-event 每通道的 lag rank），但 `compute_local_endpoint` 没用它。

**v1.0.6 起 H4 当前 implementation 降级为 SUPPLEMENTARY**：

- `docs/archive/topic4/sef_itp_phase2/cohort_run_2026-05-23.md` §2 报告的 PASS 信号 (`circular_shift` null, Wilcoxon p=9.5e-7, Cohen's d=1.50, n=21 finite) 反映 **participation field stability**（参与场稳定），作为 supplementary descriptive evidence
- **不**作为 SEF-ITP H4 关于 propagation endpoint geometry 不漂的 cohort claim 判据
- 任何外部引用 `cohort_summary.json::h4.verdict = "PASS"` 必须以 SUPPLEMENTARY 标注；不上升到 H4 主线判据

**H4 v1.1 stub (canonical 主线，Stage B 实施 → 新 archive `docs/archive/topic4/sef_itp_phase2/cohort_run_2026-05-24.md`)**：

1. **per-epoch rank-based endpoint** — 用 `load_subject_propagation_events`（或 PR-6 anchoring `ranks` 字段）per-epoch 重算 template_rank。**端点语义严格使用 rank 不用 top/bottom**（top/bottom 在 H4 当前实现里被参与率语义污染过，禁用）：
   - **source-side** = **lowest-k channels by rank**（0-indexed ranks 0, 1, ..., k−1 — 最早点火的 k 个通道）
   - **sink-side** = **highest-k channels by rank**（0-indexed ranks N−k, N−k+1, ..., N−1 — 最晚点火的 k 个通道）
   - **禁用** `events_bool.mean(axis=0)` 作为 endpoint 定义（这是 participation rate，不是 propagation rank）
   - **禁用** "top-k source / bottom-k sink" 措辞（top/bottom 词在 H4 v1.0 implementation 已经被 participation-rate top-k 污染；rank-based 必须用 "lowest-rank / highest-rank" 措辞 lock 严格区分）

2. **per-side 4 spatial radius metrics** — 对每个 cluster 的 source-side 和 sink-side **独立**算（避免轴变长与每端散混淆）：
   - (a) **centroid RMS radius** (per side)：该侧 k 点到自己 centroid 的均方距离
   - (b) **mean pairwise distance** (per side)：该侧 k 点两两距离平均
   - (c) **min enclosing ball radius** (per side)：装下该侧 k 点的最小球半径
   - (d) **source-sink centroid distance** (cross-side)：传播轴长度（横向辅助指标，单独报；**不**与 per-side radius 混合作主指标）
   - per-epoch 算每个指标，看时间漂动；matched null = `valid_mask=True` 池子里随机抽 |source| / |sink| 个通道重算 1000 次
   - **禁止**把 source+sink 混成单一半径作主指标（会把"轴变长"和"每端变散"混在一起）

3. ~~**decision-k drift — stratified report by swap_class** (三层 stratified report: high-confidence 9/23 primary + descriptive superset + fall-back k=3 sensitivity)~~ → **SUPERSEDED by v1.0.7 sub-amendment 下方** (Stage B 实际跑了 all 23 subjects 不分 swap_class gate; Stage B decision-k drift 重 frame as Phase 3 biomarker feasibility, **不**入 Phase 2 cohort verdict; 真正的 SEF-ITP cohort claim 在 Phase 3 per-seizure 招募分析 by H5 v1.0.7 spec, 不在 Stage B 背景间期 decision_k drift)。原 prose 三层 stratified report 保留作 audit trail。

4. **subject-level 检验 + cohort 检验** — 保持 v1.0.5 的 normalized instability + Wilcoxon signed-rank + Cohen's d ≥ 0.3 floor 框架，但 `I_geom` 改用 rank-based endpoint radius metrics（每个 spatial metric 独立检验，**多重比较 BH-FDR q<0.10 across 4 spatial metrics**）

5. **k = decision-k vs k = 3 sensitivity**：H4 v1.1 默认 k = decision-k (for swap subjects)；none-swap subjects fall back to k=3；额外 sensitivity sweep on k ∈ {2, 3, 4, 5}（与 framework 红线 §8.6 endpoint k=3 lock 不冲突 — 红线锁的是 H1/H2 主预测 k=3；H4 v1.1 是 SEF-ITP 关于 endpoint 时间漂动的检验，允许 decision-k 自适应）

6. **预期产出 (Stage B)**：`docs/archive/topic4/sef_itp_phase2/cohort_run_2026-05-24.md` 记录 H4 v1.1 cohort verdict (PASS/NULL/FAIL/UNDERPOWERED)；`results/topic4_sef_itp/phase2_temporal_x_geometry/per_subject_v1_1/*.json` 记录 per-subject 4-radius drift + decision-k drift 数字；图按 AGENTS.md figures README 中文规范

**Reuse discipline (CLAUDE.md §6.1 question-match)**：H4 v1.1 不复用 PR-6 anchoring `extract_endpoint_middle` (这是 cohort-level reproducibility 问题，不是 per-epoch drift 问题)；新写 `compute_rank_based_local_endpoint` thin helper + 4 个 spatial radius compute functions + decision-k drift extractor。

**v1.0.5 H4 verdict mapping (PASS/NULL/FAIL)** 在 v1.0.6 下：
- **applied to participation field (supplementary)**：当前 implementation 跑出的 PASS 反映 participation field stability — 这条结论 v1.0.6 起加 supplementary 标注，**不**作为 SEF-ITP H4 propagation endpoint 不漂的判据
- **applied to rank-based endpoint (v1.1 待实施)**：H4 v1.1 跑出的 verdict 才是 SEF-ITP H4 cohort claim 判据；v1.0.5 verdict labels (PASS/NULL/FAIL/UNDERPOWERED) 沿用，但每个 spatial metric BH-FDR 校正

**I_rate matched null spec degeneracy (v1.0.0 implementation 已知问题，pending user decision — independent of v1.0.6 amendment)**：framework v1.0.5 §3.4 prose "shuffle epoch order, recompute std" 数学退化 (std 对置换不变 → null variance = 0)；详见 `docs/archive/topic4/sef_itp_phase2/spec_amendment_2026-05-23.md`。当前 Phase 2 v1.0.0 默认 `circular_shift_within_block`；H4 v1.1 Stage B 沿用该提议直到 user ratify 最终选择。**v1.0.6 amendment 不重复处理此 spec 问题**（独立 deferred decision），但 H4 v1.1 实施代码必须保留 circular_shift 直至 user ratify。

---

#### **v1.0.7 sub-amendment 2026-05-23（user-return v3 catch ratified）— Stage B H4 scope strict-limit + decision-k drift recast as Phase 3 motivation + MEB caveat**

User-return v3 catch (`temp_claude_log.md` 2026-05-23 user 长 message 第 1-3 + 5 段)：v1.0.6 §3.4 H4 v1.1 stub 在 Stage B 实施后跑出 PASS，但 measurement framing / interpretation 必须严格限定，否则会被 over-read 成 "病理空间永久稳定 / SOZ 不扩张"，超出 Stage B 真正测的事的 scope。

**Stage B H4 v1.1 cohort claim scope lock**：

| 项 | 允许的 cohort claim | **禁止的 over-read** |
|---|---|---|
| Stage B 测的事 | 在**背景间期 30 分钟段**内, **标准化事件率不稳定度显著高于 rank-endpoint identity 不稳定度** (Wilcoxon p=1.4×10⁻⁶, Cohen's d=+1.26) | "病理空间形状不变" / "事件率漂动是 endpoint 漂动 X 倍" |
| 措辞 | "在 23 个病人 cohort（21 个 finite）上, 背景间期 30-min epochs 中, 标准化事件率不稳定度显著高于 rank-endpoint identity 不稳定度, 支持慢状态主要调制触发频率、不显著重塑传播端点身份" | "rate 漂动是 geometry 漂动的 3 倍"（raw 量比 framing, 实际是两个 dimensionless 标准化指数, 不能这样表述） |
| Time scale | **背景间期 30 分钟段**, 慢状态调制时间尺度 | ictal-adjacent / inter-seizure interval / 长时间尺度（这些是 Phase 3 重头戏 + future longitudinal cohort） |
| Spatial claim | "endpoint identity 在 30-min epoch 间 bounded variation (CV ~ 0.30, sink 比 source 略大)" 作 background characterization, 不入 cohort verdict | "病理空间稳" / "SOZ 不扩张" / 任何 SOZ 相关 claim |
| Endpoint definition | "rank-based propagation endpoint" (传播 rank top-k); participation-field (events_bool top-k) 作 supplementary 对照 | "rank-based 和 participation-field 都给 PASS → 端点定义不重要"（实际差别意义重大, supplementary 同向只是 robustness sanity） |

**decision-k drift in Stage B = Phase 3 biomarker 可行性验证, 不入 H4 cohort verdict**：

Stage B 对 all 23 subjects 跑了 per-epoch decision_k drift。**这个分析的科学价值不是 Phase 2 cohort claim**，而是：

> Stage B confirms per-epoch decision_k is measurable and shows bounded variation (strict 子组 epoch decision_k range 普遍跨 2-4 个值, 最极端 zhaochenxi 跨 3-13); this **motivates Phase 3 per-seizure recruitment / expansion analysis** as a feasible biomarker candidate.

**禁止 over-read**：
- ❌ "swap-class 之间 decision_k_std 无差异 → SEF-ITP 一致" — 这是组间无差异, 不是 SEF-ITP cohort claim 的证据; 真正的 SEF-ITP claim 在 Phase 3 per-seizure 招募分析
- ❌ "decision_k 稳 = 病理核心数量不变" — 跨 subject decision_k_std 不可直接比 (k=2 时 std=1 跟 k=23 时 std=1 含义完全不同); Stage B 只是验证 metric 可测 + bounded
- ❌ none 组 decision_k 是 "稳定 swap-core 的核心大小" — 实际是 noise/control baseline, 不能跟 strict/candidate 同等解释

**Spatial radius metric caveats (Stage B 报告 + Phase 3 implementation gate)**：

1. **`compute_endpoint_spatial_radius` MEB (min enclosing ball) 当前实现对 k ≤ 3 完整, k ≥ 4 不完整**：
   - 实现枚举 2-point antipodal sphere + 3-point circumsphere (3D 三角形外接圆球), 取最小包含所有点的半径
   - k ≤ 3: 完整 (任何 ≤ 3 点的 MEB 必然由 2-point 或 3-point 候选给出)
   - k ≥ 4: **不完整** — 4 点 MEB 可能由 4-point sphere (4 点都在球面) 给出, 当前实现缺这种穷举; fallback 是 "farthest-from-centroid" 启发, 给的是上界不是真正 MEB
   - **Phase 3 implementation gate**：H5 用 variable decision_k 时 k 可能 > 3, **Phase 3 启动前必须**:
     - 要么补 4-point sphere 枚举 (3D 4 点 MEB: 试 4 点都在球面的唯一外接球, 否则 fall back 到 3-point 子集穷举)
     - 要么把 MEB **降级为 sensitivity metric**, centroid RMS + mean pairwise 升 primary (这两个对任意 k 都是 well-defined, 数学上无歧义)
   - Stage B 报告里 MEB 数字 (cluster 0 source ≈ 10.82 mm, sink ≈ 18.03 mm) 在 k=3 fixed 默认下是正确的; 但 v1.0.6 §3.4 H4 v1.1 stub 提到 "k = decision-k for swap subjects" — 这部分 swap-positive subject 的 MEB 数字未在 Stage B v1.1.0 实施 (Stage B 默认 endpoint_k=3 fixed); future 实施 variable k 时必须先处理 MEB k>3 问题

2. **source vs sink 半径 background characterization** (Stage B 描述性, 不入 cohort claim)：
   - cluster 0 cohort 中位: source centroid RMS ≈ 9.66 mm, sink ≈ 16.69 mm; source mean pairwise ≈ 15.88 mm, sink ≈ 24.89 mm; source-sink centroid distance ≈ 20.95 mm (传播轴长度)
   - 段间 CV 中位 ~ 0.28-0.32 — bounded variation, 不为零
   - **观察**: sink 侧空间分布比 source 侧更分散 (~17 vs ~10 mm 半径量级) — 与"涟漪从紧的源点扩散开"的物理图景一致, 但**不**作 cohort claim, 只作 background characterization

**v1.0.7 banner authorization scope**：user-return v3 catch authorized (a) Stage B H4 cohort claim scope strict-limit to background interictal rate-geometry decoupling, (b) decision-k drift Stage B 重 frame as Phase 3 biomarker feasibility (not cohort verdict), (c) MEB k>3 incomplete caveat + Phase 3 implementation gate, (d) source vs sink background characterization 限定 descriptive only. 措辞锁与 H3 v1.0.6 / H5 v1.0.7 一致 (cohort claim 严格限定 scope, 不 over-read; descriptive sub-shape / sensitivity / motivation 与 primary verdict 严格分层)。

### 3.5 H5 — Endpoint identity shift around seizures（短时尺度，仅可测部分）

#### **v1.0.6 amendment 2026-05-23（user-return v2 catch ratified）— H5 主问题重定向 + 3 primary recruitment/expansion metrics + 1 secondary rate metric**

User-return v2 catch (`temp_claude_log.md` 2026-05-23 user 长 message)：v1.0.5 §3.5 主问题 "subject-level fraction of channels with `ΔS_i > 0 AND ΔR_i > 0`" 只测 endpoint score Δ + rate Δ **各自方向**，**没测**：(a) endpoint identity 招募（新通道临时变成 swap-core）；(b) endpoint 空间扩散（swap-core 空间半径扩张）；(c) 核心病理区数量变化（decision-k Δ）。SEF-ITP 关于"病理核心可能在发作邻近招募更多通道或空间扩展"的预测**没有被 v1.0.5 spec 检验**；并且 v1.0.5 把 rate Δ 与 endpoint Δ 平起平坐做 AND 条件，会把"单纯 rate 升高 endpoint 不变"（其实是 SEF-ITP 整体图景的强证据）误判为 NULL。

**v1.0.6 H5 主问题改为**：

> 发作邻近 (pre-ictal / post-ictal) 是否出现 **endpoint / swap-core 招募或空间扩展**，而不是单纯 rate 升高？

**v1.0.6 H5 指标合同：3 primary recruitment/expansion metrics + 1 secondary rate metric** (peri-ictal window vs time-of-day matched baseline window)：

| # | 指标 | tier | 测什么 | 期望 SEF-ITP 方向 |
|---|---|---|---|---|
| 1 | **swap-k node identity Jaccard ↓** | primary | pre/post-ictal vs baseline window 的 swap-k node set Jaccard；招募新节点 → Jaccard 下降 | Jaccard ↓ (recruitment) |
| 2 | **swap-k 空间半径扩张 (source/sink 各自)** | primary | source-side / sink-side **independently** 算 RMS radius + mean pairwise distance peri vs baseline；空间扩张 → radius ↑ | radius ↑ (spatial expansion) |
| 3 | **decision-k Δ** | primary | swap_sweep `decision_k` peri-ictal vs baseline；核心区招募更多通道 → decision_k ↑ | decision_k ↑ (core recruitment) |
| 4 | **HFO rate Δ** | **secondary** | events/h peri vs baseline | rate ↑ (descriptive 标记 only) |

**为什么把 rate Δ 严格降级 secondary（统计合同 lock，不只是文风）**：v1.0.5 把 rate Δ 与 endpoint Δ 在 binary AND condition 里平起平坐，相当于要求 SEF-ITP "rate 必须升高 AND endpoint 必须扩散" 同时才算 PASS — 但 SEF-ITP 的核心图景是"病理空间结构相对稳定 + 触发率受慢变量调制"，rate ↑ 是 SEF-ITP 兼容的也是 alternatively 假说兼容的（任何 ictal-邻近 excitability 升高都预测 rate ↑），**不是** SEF-ITP 的 discriminative 预测。v1.0.6 严格只用 primary 3 metrics (招募/空间扩展) 作为 SUPPORTED gate；rate Δ 是 descriptive secondary，不入 gate count。

**v1.0.6 H5 verdict mapping (DRAFT — 阈值具体值 X/3 pending user ratify)**：

- **SUPPORTED**：≥X/3 primary metrics (1, 2, 3) cohort Wilcoxon p<0.05 (**BH-FDR q<0.10 across the 3 primary metrics**) in SEF-ITP direction + Cohen's d ≥ 0.3 (per primary metric) + ≥6 qualifying subjects。rate Δ (secondary) descriptively reported alongside，**不**入 gate count。
- **NULL**：< X/3 primary metrics 显著 + 无强反向 → recruitment / expansion 信号 underpowered 或 absent
- **FAIL**：≥X/3 primary metrics 显著反向（Jaccard ↑ identity 稳定, radius ↓ 收缩, decision_k ↓ 核心区收缩）→ 发作邻近反而 endpoint 收缩 / 一致 → 与 SEF-ITP 预测**反向**
- **UNDERPOWERED**：< 6 qualifying subjects (v1.0.5 floor 保留；不达 → "暂无定论"，列入 future cohort 扩展 PR)

**⚠️ X/3 threshold value authorization scope lock (banner authorization boundary)**：user-return v2 catch authorized (a) main question pivot to "招募/空间扩展 vs 单纯 rate 升高", (b) 3 primary metrics + 1 secondary rate metric, (c) rate Δ severely demoted out of SUPPORTED gate. The catch did **not** explicitly authorize the specific threshold value X (e.g., X=2/3 weak majority vs X=3/3 strict all-primary). Both are defensible：
- **X = 2/3 (weak majority)**：允许一项 primary 不显著但其他两项强烈推动 SUPPORTED — more sensitive to partial recruitment patterns；可能 inflate false positives
- **X = 3/3 (strict all-primary)**：要求 recruitment + spatial expansion + core recruitment 全都显著 — 严格 SEF-ITP discriminative prediction；更保守 specifity 更高

**Pending user ratify**：until user explicitly ratifies X，Stage C (Phase 3 H5) implementation must surface this decision before producing cohort verdict。Same authorization-scope lock pattern as §3.3 v1.0.6 "短-scale POSITIVE memory variant not pre-registered" decision。

**v1.0.6 sustained spec items (v1.0.5 lock 不变，沿用)**：

- 数据 gate：per channel × per window `n_events ≥ 30`；不达 channel 在该 subject H5 分析中剔除
- 窗口定义：pre-ictal `[−60min, −5min]`，post-ictal `[+5min, +60min]`，baseline 按下规则
- Baseline 选择：同 subject + 同 hour-of-day ± 2h + 距任何 seizure ≥ 12h + 每 seizure ≥ 5 baseline 窗口（不满足 → 该 seizure 排除）
- ~~Subject 内多 seizure：先 aggregate 到 subject-level (mean across seizures per metric)，**禁止**把每次 seizure 当独立观测~~ → **SUPERSEDED by v1.0.7 amendment 2026-05-23**（见下）
- SOZ 关系 (secondary, 仅在 H5 主分析 SUPPORTED 时报告)：在 `≥X/3 primary metrics 显著的子集 subject` 上查 primary 指标 1-3 涉及的招募通道是否更靠近 clinical SOZ / data-driven SOZ (PR-T3-1 Layer B)；与 ictal early channel overlap；个体 DTI 可用时 SC out-degree 关系；**两套 SOZ 标签都查，不融合**；**禁止**在 underpowered case 上做 SOZ 关系分析

**v1.0.6 measurement source contract**：

- 指标 1 (swap-k Jaccard): 用 `src.rank_displacement.derive_swap_endpoint` per-window 算 swap-k node set; baseline = time-of-day matched window 平均
- 指标 2 (spatial radius): 同 H4 v1.1 的 4 spatial radius metrics (centroid RMS / mean pairwise / per-side)，复用 H4 v1.1 implementation
- 指标 3 (decision-k Δ): 同 H4 v1.1 decision-k drift; per-window 重跑 `swap_sweep`
- 指标 4 (rate Δ): events/h per window，简单计数

**Stage 实施序**：H5 v1.0.6 实施依赖 H4 v1.1 的 spatial radius / decision-k drift helpers (复用 question-match: H4 v1.1 测时间漂动 inv slow rate modulation, H5 v1.0.6 测时间漂动 around seizures — 同类几何量、不同窗口划分)。建议 Stage B 先做 H4 v1.1 (Phase 2 收口)，然后用同样的 helpers 做 H5 v1.0.6 (Phase 3 启动)。

---

#### **v1.0.7 sub-amendment 2026-05-23（user-return v3 catch ratified）— H5 reporting unit + cohort inference 改 per-seizure**

User-return v3 catch (`temp_claude_log.md` 2026-05-23 user 长 message 第 4 段)：v1.0.6 H5 sustained spec 里"Subject 内多 seizure 先 aggregate 到 subject-level"跟 user 当前判断**冲突**。Topic 5 PR-1 与 PR-4 cohort run 已证明 subject 内 seizure type 差异巨大、seizure 间时间波动也很大、相对时间也不能解决。subject-level aggregate 会把这些 within-subject heterogeneity 抹平 → 真正的 ictal-adjacent 招募/扩张信号被 average out。**H5 真正测的事**：peri-ictal 招募 / 空间扩张，这本来就是 per-seizure 现象（不同 seizure 性质不同、不同时段触发的核心区不同），subject 级 aggregate 是错的统计单位。

**v1.0.7 H5 reporting unit + cohort inference lock**：

| 项 | v1.0.6 (SUPERSEDED) | **v1.0.7 (current)** |
|---|---|---|
| Primary reporting unit | subject-level mean across seizures | **每次 seizure 的 peri-ictal window vs matched baseline** |
| Cohort inference | Wilcoxon on subject-level fraction vs 0.25 | **hierarchical / subject-clustered**（mixed-effect model with seizure nested in subject, OR cluster-robust SE on subject id）；**禁止**把同一 subject 多次 seizure 当独立样本 |
| Stratification (mandatory) | (无) | **按 seizure type / seizure pair / inter-seizure interval 分层报**（subject 内 seizure 性质不同必须 separately reported, 不能 average） |
| Subject-level summary | primary statistic | **secondary**（仅作 between-subject heterogeneity sensitivity） |

**Phase 3 主问题 lock (v1.0.7)**：

> 在两次 seizure 之间或 seizure-adjacent window，**swap-k endpoint 的数量和空间范围是否相对 matched baseline 增加**？

这是 per-seizure primary, subject-clustered cohort inference。

**v1.0.7 primary metrics 精确定义（覆盖 v1.0.6 表的 metric tier）**：

| # | 指标 | 类型 |
|---|---|---|
| 1 | **Δdecision_k = decision_k(peri) − decision_k(matched baseline)** | per-seizure primary |
| 2 | **Δdecision_k / baseline_k**（normalized to baseline core size，跨 subject 可比化）| per-seizure primary |
| 3 | **swap-k endpoint identity Jaccard(peri vs baseline) ↓**（招募新节点）| per-seizure primary |
| 4 | **source-side / sink-side spatial radius peri vs baseline**（centroid RMS + mean pairwise + axis distance；MEB 见 §3.4 caveat）| per-seizure primary |
| 5 | HFO rate Δ peri vs baseline | per-seizure **secondary**（excitability state marker 只用作 descriptive context，**不**入 SUPPORTED gate） |

**为什么 Δk + Δk/baseline_k 一起报**：decision_k_std 跨 subject 不可直接比 — k=2 时 std=1 意味着核心区数量翻倍 ±50%，k=23 时 std=1 是 < 5% 变化（user-review catch）。Phase 3 必须用 Δk（绝对变化）、Δk/baseline_k（相对变化）、range（极值）+ spatial radius 联合报，**不**单独用 std 跨 subject 比较。

**v1.0.7 implementation gate**：
- 必须先把 §3.4 H4 v1.1 MEB k>3 caveat 处理掉（要么补 4-point sphere，要么 MEB 降为 sensitivity，centroid RMS + mean pairwise 升 primary）才能跑 Phase 3 spatial radius metric
- swap_class=none 的 subject decision_k 是 noise/control baseline（**不是**稳定 swap-core 的核心大小），Phase 3 这部分 subject 的 Δdecision_k 不能跟 strict/candidate 同等解释 — 要么作 negative control（期望 Δk ≈ 0），要么完全排除 Phase 3 分析
- Phase 3 启动前必须先把 Phase 2 cohort_run_2026-05-24.md 收口 + advisor pass

**v1.0.7 banner authorization scope**：user-return v3 catch authorized (a) reporting unit pivot to per-seizure primary, (b) cohort inference 改 hierarchical / subject-clustered, (c) stratify by seizure type / pair / interval, (d) Δk + Δk/baseline_k + range + spatial radius 联合报, (e) decision_k_std cross-subject 不可直接比, (f) none 组 decision_k 不可与 strict/candidate 同等解释. The catch did **not** explicitly specify the hierarchical model form (linear mixed model vs GEE vs cluster-robust SE) — Stage C implementation must surface this decision before producing Phase 3 cohort verdict. Same authorization-scope lock pattern as §3.3 v1.0.6 + §3.5 v1.0.6 X/3 threshold pending items.

**v1.0.5 H5 主问题 + verdict mapping (SUPERSEDED by v1.0.6 above, kept for audit trail below)**：

**测了什么**：靠近发作的时段（pre-ictal / post-ictal）里，是不是有原本不算 endpoint 的通道临时变成 endpoint，同时 HFO rate 也升高。

**为什么收紧合同**（v1.0.2 修订）：原 v1 只写 `ΔS_i > 0 AND ΔR_i > 0`，没说最小事件数、baseline 怎么挑、subject 内多 seizure 非独立怎么处理、通道多重比较怎么校正——这种合同会变成"看图说话"。v1.0.2 把统计合同显式锁住。

**统计合同（v1.0.2 lock，每一步都不可松动）**：

#### 数据 gate

- **最小事件数**：per channel × per window `n_events ≥ 30`；不达的 channel 在该 subject 的 H5 分析中被剔除
- **窗口定义**：pre-ictal `[−60min, −5min]`，post-ictal `[+5min, +60min]`，baseline 见下
- **窗口选择说明**：刻意比 Topic 2 PR-2.7 broad-rate 窗口 `[−6h, −1h]` 短——H5 测 endpoint identity 短时漂移，不测 broad rate 调制；H5 PASS 后可 sensitivity 用 `[−6h, −1h]` 验证 broad rate 是否同漂

#### Baseline 选择（time-of-day matched，不可松动）

每个 seizure，baseline 窗口规则：

1. 同一 subject 内
2. 与 pre-ictal / post-ictal 窗口同 hour-of-day ± 2h（控制 circadian）
3. 距任何 seizure ≥ 12h
4. 每 seizure 至少配 5 个独立 baseline 窗口

不满足任一条 → 该 seizure 从 H5 排除（不强凑）。

#### Subject 内多 seizure 处理（不可独立计数）

- 每 subject 多 seizure 时，**先 aggregate 到 subject-level**：对每 channel 算 `mean(ΔS_i across seizures)` + `mean(ΔR_i across seizures)`
- **禁止**把每次 seizure 当独立观测——multi-seizure 高度非独立（同 subject 同 SOZ）

#### 多通道比较（不可放过 FDR）

- 第一线 primary：subject-level **scalar statistic** = "fraction of channels with `ΔS_i > 0 AND ΔR_i > 0`"
  - 每 subject 输出一个 0–1 标量
  - H0 期望（ΔS / ΔR 各自独立、各自方向 50/50 chance）：fraction ≈ 0.25
- 备选 sensitivity：subject 内 BH-FDR across channels (q < 0.10) + count significantly-increasing channel 数
- **禁止**裸跑每 channel 单独 t-test 不校正

#### Cohort 检验

- Wilcoxon signed-rank on subject-level fraction `vs 0.25` (one-sided, fraction > 0.25)
- **Power floor**: 至少 6 个 subject 每个有 ≥ 1 qualifying seizure；不达 → 报告 "UNDERPOWERED, no verdict"，**不**给 PASS/NULL/FAIL
- 效应大小：Cohen's d on `(fraction − 0.25)` per subject

#### SOZ 关系（secondary，仅在 H5 主分析 PASS 时报告）

在 `fraction > 0.5` 的 subject 子集上（"明显有 endpoint shift" 的子集），查这些 ΔS>0 ΔR>0 channel 是否：

- 更靠近 clinical SOZ
- 更靠近 data-driven SOZ（`results/data_driven_soz/layer_b_labels/`，**当前不存在，待 PR-T3-1 Layer B 启动**；两个标签都查，不融合）
- 重叠 ictal early propagation channel
- 如果有个体 DTI：更接近 long-range SC out-degree 高的节点

**禁止**在 underpowered case 上做 SOZ 关系分析。

**揭示了什么**：

- **PASS**：cohort Wilcoxon `fraction > 0.25` p<0.05 + Cohen's d ≥ 0.3 + ≥ 6 qualifying subject，且 SOZ-related channel 富集 → 支持 "SOZ recruitment / 扩展" 机制
- **NULL**：cohort fraction 兼容 0.25 → 无系统性 endpoint identity shift
- **FAIL**：cohort fraction 显著 < 0.25 → 发作邻近反而 endpoint identity 更**不**变（与模型预测反向）
- **UNDERPOWERED**：< 6 qualifying subject → "暂无定论"，列入 future cohort 扩展 PR

**长尺度版本 H5-long 列存档**：跨录制周 / 月年级 endpoint 漂移需要纵向 cohort，**当前数据不可测**，作为 future work。Epilepsiae 部分 subject 有多次入院记录，未来如能整合可单独立 PR。

### 3.6 H6 — Participation-field 空间分隔（基础证伪）

**测了什么**：高参与（经常出现在事件里）通道和低参与（很少参与）通道，是不是空间上分得开——high participation 聚成一团，low participation 散在外围 / 远离 core。

**怎么测的**：

- 每通道 `P_i = #events containing i / #total events`（参与率）
- 空间分隔度量（三种并列）：
  1. 两群质心欧氏距离 vs null
  2. Moran's I 空间自相关（要求统计显著）
  3. Silhouette score（h vs l 标签下）
- **Null 构造 lock**：**shaft-stratified shuffle**（同一根电极杆内 shuffle 参与率）作为主 null；全脑 shuffle 作为敏感性
- **与 H1 独立**：H6 不需要 endpoint 定义，可以**先于** endpoint analysis 跑作为基础证伪

**揭示了什么**：

- **PASS**：参与场有空间结构 → 支持"空间组织化病理场"基础断言
- **NULL**：参与率到处都一样，没有空间组织
- **FAIL**（low 和 high 完全混杂）：**空间组织化病理场断言垮**，整个 framework 受重创——回到讨论是否换模型

---

## 4. Phase 1 失败模式矩阵（pre-registered）

H1 / H2 / H6 是 Phase 1 三条核心几何检验。它们的联合 verdict 决定 framework 后续走向。**Phase 1 实操顺序建议**：先跑 H6（基础证伪，不需要 endpoint）→ 过了再跑 H1（三层 H1a/H1b/H1c）→ 过了再跑 H2（两条 reversal index 主分析 + PCA sensitivity）。

**H1 整体 verdict 规则**（v1.0.2 新加）：

- H1 = PASS：H1a + H1b 同时 PASS（≥ 3/4 距离同向）+ H1c 不 FAIL
- H1 = partial PASS：H1a 或 H1b 一过一不过 + H1c 不 FAIL
- H1 = NULL：三层都不显著
- H1 = FAIL：H1c FAIL（endpoint 大幅脱离参与场，`ratio >> 1`），不论 H1a / H1b

**H2 报告规则**（v1.0.5 lock 2026-05-22，旧 v1.0.4 "mechanism sanity only" 已弃用——tier 拨正为 primary cohort claim）：

H2 现在是 **primary cohort claim**，但因为它有两层（label 层 / spatial 层），cohort verdict 必须分层报，不写单一 PASS/NULL/FAIL。

- **label 层** (per subject)：`swap_class` ∈ {`strict`, `candidate`, `none`} + `decision_k` + `T_obs` + `p_fw`。来源是 masked rank-displacement `swap_sweep`。**label 层是 SEF-ITP swap 信号的窄定义**——只有 strict 表示家族错误率控制下确实测出 source/sink 角色反转。
- **spatial 层** (per subject)：source-side `k` 个 swap 节点、sink-side `k` 个 swap 节点 各自做 `PASS / NULL / FAIL_DIFFUSE` compactness 裁决（null = `all mapped SEEG minus swap endpoint` 无 shaft 约束随机抽 `k`）。`combined_endpoint (2k)` 只作辅助描述。**spatial 层是 SEF-ITP 端点几何耦合的宽定义**——swap-k 通道在脑里有空间聚合性。
- **cohort 层 v1.0.5 输出契约**：
  - label 层 cohort：`swap_class_distribution`（strict / candidate / none / unknown）
  - spatial 层 cohort：每侧（source / sink / combined）的 `PASS / NULL / FAIL_DIFFUSE` 比例 + binomial sign-test p 值（against p_null=0.5 coin-flip baseline；against p_null=0.05 per-subject 5%-α baseline，两个都报，不省略 0.5 那个）
  - 双侧联合：`#subjects with source PASS ∩ sink PASS / n` + binomial p (against p_null=0.25 双侧独立基线)
  - 不再报 `exceeds_null_95th` sign-test（旧 v1.0.8 字段，已删）

**允许的判读语言（v1.0.5 lock）**：
- ✅ "在 n=23 cohort 上，rank-displacement 派生的 swap-k source-side 与 sink-side endpoint 各自相对其他 mapped SEEG 节点呈现 cohort-level 空间紧凑性（source 19/23 PASS, binomial p=1.3e-3 vs 50% coin-flip null；sink 16/23 PASS, p=4.7e-2）。13/23 subject 双侧同时 PASS（p=1.2e-3 vs 25% independent baseline）"
- ✅ "spatial 层 cohort claim 支持 SEF-ITP 对偶端点预测：同一组 swap-k 节点同时是空间紧凑的 source 和空间紧凑的 sink"
- ✅ "label 层 9/23 strict-or-candidate 是更狭义的 source/sink role-swap 信号；strict-only 5/23 表明 family-wise null 下严格通过的 subject 数比 spatial 层少——这是窄定义 vs 宽定义的自然差距"
- ✅ "EN: spatial layer cohort claim is supported on n=23: 19/23 source-side and 16/23 sink-side swap-k endpoints show significant spatial compactness; 13/23 both sides simultaneously"
- ❌ "H2 PASS"（信息丢失太多，必须保留分母 + 哪一层）
- ❌ "H2 100% PASS" / "每个 subject 都通过"（实际 22/23 至少一侧 PASS，但 1/23 双侧 NULL，分母不能丢）
- ❌ 从 spatial 层 PASS 直接跳到 "source/sink 角色反转是 cohort 主效应"（spatial 层是必要不充分；要 strict claim role-swap 看 label 层 strict 比例）
- ❌ `combined_endpoint` 当作唯一空间裁决（source 与 sink 可能本身就在传播轨迹两端）

**H2b 方向轴诊断（supplementary，archive-only，2026-05-25 v1.0.2 review-round-1 demoted）**：在 H2 spatial-layer cohort claim 之上，archive-only 的 H2b 方向轴诊断 phase，问"swap 是同一条空间轴反向读取还是两个独立 source"。**Strict verdict (primary read)** 在 23 个可测 subject 上 0 个 axis_reversal，因 decision_k ≈ n_universe / 2 时 per-cluster role-shuffle null 自由度低。**Descriptive shape (supplementary, NOT a conclusion layer)** 在 swap_class∈{strict, candidate} 的 9 个可测 subject 上 5 个 axis_reversal_shaped、0 个 dual_source_shaped、4 个 unclear（含 3 个 PCA 单 shaft 退化）；shape 一致于"非正交"假说，但**注意 scope 红线**：H2b 只能 falsify "cluster A 与 cluster B 是正交无关解剖源" 假说，**不能区分** "同一病理核心轴双向读取" 与 "同轴双端独立 seed"（两者都预测 cos(v_A, −v_B) ≈ +1）。要区分必须做 Round 2（per-event seed 聚类 + rank-distance 梯度 + source 各自 SOZ 关系），未实施。**H2b 不是 cohort claim**；不进 framework v1.0.7 §3 H 主清单。详见 `docs/archive/topic4/sef_itp_direction_axis/{phase_h2b_direction_axis_plan_2026-05-25.md (v1.0.2 §1.5 + §10.5), cohort_run_2026-05-25.md}`。

**Phase 1 失败模式表**（v1.0.5 lock；H1 列"sanity"，H2 列"primary cohort claim"按 spatial 层 cohort 比例填）：

| H6 (participation 分隔, secondary) | H1 (endpoint 锚点, **sanity / 必要前置**) | H2 (swap-k spatial cohort, **primary claim**) | Framework 解读 | 后续动作 |
|---|---|---|---|---|
| PASS | PASS / partial | source ≥70% PASS + sink ≥60% PASS | 完整证据链：基础锚点 + SEF-ITP 区分性预测都成立 | 进 Phase 2，**framework cohort claim 成立** |
| PASS | PASS / partial | source 或 sink < 50% PASS | sanity 通过但 SEF-ITP 区分性预测不成立 | framework 在该 cohort 下被弱化；考虑 H2 label 层 strict 比例 + cohort 扩展 |
| PASS | NULL | source ∩ sink 都 ≥70% PASS | sanity 失败但 spatial 成立——前提条件没满足却得到下游 cohort 信号，**内部矛盾** | 回 Phase 0 查 endpoint 定义 + phantom rank 是否漏修；不能直接接受 H2 cohort claim |
| PASS | FAIL (H1c) | * | endpoint 大幅脱离参与场 | endpoint operational definition 需重新设计；H2 cohort claim 暂搁置 |
| NULL / FAIL | * | * | **基础断言（空间组织化病理场）证伪** | **framework 整体回退**；SEF-ITP 名字作废 |

**重要**：H1 是 sanity，H1 表现弱不等于 framework 失败；但 H1 通过不等于 framework 成立——必须 H2 的 cohort claim 同时成立。**禁止**事后调整 H1 / H2 / H6 数字判据来"挽救" verdict。

---

## 5. Cohort tier 定义（数字化，pending Phase 0）

继承 AGENTS.md 的 cohort tier 纪律。所有 Phase 1+ 数字必须标 Tier 1 / Tier 2，**禁止裸写** "30/30" 或 "35/40"。

### 5.1 Tier 定义（criterion 措辞 lock，数字 TBD pending Phase 0）

**Tier 1（主预测 cohort）**：

- valid-only 重跑后 `stable_k = 2`
- `forward_reverse_reproduced = True`（split-half OR odd-even，继承 CLAUDE.md cross-PR contract）
- 每个 cluster `n_participating_channels ≥ 6`（H1 / H2 主分析最低要求；endpoint k=3 占满 6 个通道仍允许 H1a / H1b 与 H2 set/spatial reversal 跑）
- SEEG/ECoG 3D 坐标可用
- 数字 N = **TBD**（待 Phase 0 `results/interictal_propagation_masked/` 完成后填；当前 phantom-rank-contaminated 版本约 6–8 subject 进 fwd/rev cohort，masked 重跑后预计同量级，但 lock 必须以 valid-only 实跑为准）

**Tier 1+ (H2 PCA sensitivity sub-cohort)**：

- 满足 Tier 1 所有条件
- **额外**：`non_endpoint_participating_channels ≥ 4` per cluster（即 `|valid_mask=True ∩ ¬(S ∪ K)| ≥ 4`）
- 用于 H2 sensitivity PCA axis projection（主分析 reversal index 不需要此条件）
- 数字 N = **TBD**（待 Phase 0；这是 Tier 1 的子集）

**Tier 2（敏感性 cohort）**：

- valid-only 重跑后 `stable_k = 2`
- 不强制 forward/reverse 复现
- 数字 N = **TBD**（best estimate per Topic 0 §3.1：40-subject cohort 中 1 例 stable_k 翻转（epilepsiae_916 2→4），其余 39/40 stable_k 不变 → **Tier 2 ≈ 34 预期，pending lock-in**；高 k cohort 已被独立排除）

**Excluded**：

- valid-only 重跑后 `stable_k ≠ 2`（已知 epilepsiae_916 在 Topic 0 cohort 中 2→4 翻转）
- 参与通道 union 不足
- phantom-rank 修复后 endpoint 集合手动审计不通过

### 5.2 规则

- 主文档 / archive / paper 任何引用 Phase 1+ 数字时**必须**标注 Tier 1 / Tier 2
- Phase 0 完成后 cohort tier 数字 lock，**不允许**为了 H1/H2 显著性反向调整
- 任何 Tier 1 与 Tier 2 数字方向不一致的情况，**主结论以 Tier 1 为准**，Tier 2 作 sensitivity 报告

---

## 6. Phase 0–4 实施路线

### 6.1 Phase 0 —— Phantom-rank 修复 + broad re-derivation 【硬前置】

**目标**：把所有 endpoint / cluster / template centroid 的地基修好。

**关键 bug**：`lagPatRank` 现有数据里没参与某事件的通道也被赋予了有限的 int rank（U-型污染，两端 rank=0 / rank=N−1 都可能被噪声塞通道）。详情：`docs/topic0_methodology_audits.md` §3.1。

**任务清单**（继承 topic0 §5 重跑路线图）：

- [ ] **5a** PR-2 re-cluster on masked → `results/interictal_propagation_masked/per_subject/`
- [ ] **5b** PR-2.5 split-half / odd-even on masked → 同上
- [ ] **Checkpoint A** advisor consult: stable_k flip + forward/reverse 复现集合变化
- [ ] **5c** PR-3 per-cluster MI on masked labels
- [ ] **5d** PR-4A/B/C/D on masked labels
- [ ] **Checkpoint B** advisor consult: PR-4B/D 方向是否反转
- [ ] **5e** PR-5 / PR-5-B on masked（H4 验证基础）
- [ ] **5f** PR-6 endpoint anchoring / swap / Step 6 / rank displacement on masked（H1 / H2 准备）
- [ ] **5g** PR-7 antagonistic pairing on masked（H3 验证基础；**P3 翻转 = framework-level revision，必须停下来 review，不允许默默改写**）
- [ ] **5h** Topic 4 attractor on masked → `results/topic4_attractor_masked/`
- [ ] **5i** cohort tier audit 表 lock：Tier 1 N / Tier 2 N / Excluded N + 数字 lock 到本文 §5

**前置纪律**：

- Phase 1 在 Phase 0 完成前**不启动**
- Phase 0 进度看板维护在 `docs/topic0_methodology_audits.md` §5 重跑路线图
- 如果 Phase 0 实际推进缓慢，本 framework 处于 **"等待修复完成的占位文档"** 状态——这一状态在本文 §0 + paper_overview 上必须诚实声明，**不掩饰**

### 6.2 Phase 1 —— 空间几何检验（免费，不需要建模）

**目标**：先证明"空间场"假设的必要性。

**前置**：Phase 0 完成 + Tier 1 cohort + SEEG 3D 坐标可用。

**Runner 入口（2026-05-21 完工）**：

```
scripts/run_sef_itp_phase1.py --dataset <epilepsiae|yuquan> --subject <sid> \
    --distance-metric euclidean --hypothesis all \
    --output-dir results/topic4_sef_itp/per_subject
```

`load_subject_for_phase1(<phase0a_json_path>)` 把以下 4 个数据源接成 `SubjectPhase1Data`：

| 源 | 路径 | 提供 |
|---|---|---|
| 1. masked Phase 0a JSON | `results/interictal_propagation_masked/per_subject/<dataset>_<sid>.json` | `channel_names`, `adaptive_cluster.candidate_forward_reverse_pairs` |
| 2. masked PR-6 anchoring JSON | `results/interictal_propagation_masked/template_anchoring/per_subject/<dataset>_<sid>.json` | `per_template[].source/sink/valid_mask`（端点为通道名，loader 转 index）|
| 3. lagPat NPZ | yuquan: `/mnt/yuquan_data/yuquan_24h_edf/<sid>/`<br>epilepsiae: `/mnt/epilepsia_data/.../all_data_lns/<sid>/all_recs/` | `events_bool` via `src.interictal_propagation.load_subject_propagation_events` |
| 4. SEEG 坐标 | yuquan: `chnXyzDict.npy`<br>epilepsiae: SQL `electrode.coord_*` + MNI152 1mm MRI affine | `coords (n_ch, 3)` via `src.seeg_coord_loader.load_subject_coords`；mm 强制断言 |

合同关键条款（通道对齐 / 端点 name→index / mm 单位断言 / valid_pool 交集 / pair schema 翻译）+ 8 个集成测试用例 + 完整 results dir layout 见 [`docs/archive/topic4/sef_itp_phase1/phase1_runner_contract_2026-05-21.md`](archive/topic4/sef_itp_phase1/phase1_runner_contract_2026-05-21.md)。所有集成测试 GREEN（2026-05-21）。

实跑 smoke test（7 subject：epilepsiae 1073/548/922/590/1150/1077/1146，yuquan chengshuai）全部产出真实裁决，无 pipeline error。`coord_provenance.normalization_certainty` 字段如实记录 `grid_confirmed_warp_type_unverified`（Epilepsiae）/ `subject_native`（Yuquan），未来 warp field 文档到位再升级到 `mni_normalized_verified`。

**任务顺序**（按基础证伪 → 详细验证）：

1. **H6 participation-field segregation**（优先跑——基础证伪）
2. **H1 endpoint compactness**（四种距离度量并列；matched null）
3. **H2 source/sink reversal geometry**（PCA 主轴独立于 endpoint）
4. **SOZ relation as secondary**：endpoint 与 clinical SOZ + data-driven SOZ（PR-T3-1 输出）并列报告 overlap / distance / enrichment；**不融合**，不作模型硬前提

**产出图**（按 AGENTS.md figures README 中文规范）：

- 每 subject endpoint spatial map（含正反两模板叠图）
- endpoint compactness vs random null violin plot（cohort）
- reversal-axis projection scatter（four endpoint groups on PCA axis）
- participation spatial heatmap（每 subject 一张）
- cohort summary forest plot（H1/H2/H6 effect sizes）

**产出目录**：`results/topic4_sef_itp/phase1_spatial_geometry/{per_subject,cohort_summary.json,figures/,soz_relation/}` — 整个 root 在 Phase 1 启动时新建。完整 dir layout 见上述 archive。

### 6.3 Phase 2 —— Temporal × geometry 联合检验

**目标**：证明"随机触发 + 稳定几何骨架"是否成立。

**前置**：Phase 1 H6 + H1 至少 PASS。

**任务**：

- H3 mark independence + endpoint split-half/odd-even recall（继承 PR-7 addendum 形式）
- H4 CV(rate) vs CV(geometry) across 1h–2h epoch
- 联合 verdict 表（参见 §3.3 H3 揭示什么）

**产出目录**：`results/topic4_sef_itp/phase2_temporal_x_geometry/`

### 6.4 Phase 3 —— 发作邻近 endpoint identity shift

**目标**：H5 短时版本检验。

**前置**：Phase 1 + Phase 2 全 PASS（否则发作邻近也不可信）。

**任务**：

- baseline / pre-ictal / post-ictal endpoint score + HFO rate 联合变化
- 与 clinical SOZ + data-driven SOZ + ictal early channel 的空间关系
- 如有个体 DTI：与 long-range SC out-degree 的关系

**产出目录**：`results/topic4_sef_itp/phase3_ictal_adjacent/`

### 6.5 Phase 4 v2 —— SEF-HFO 最小模型闭环（取代 HR/FHN toy 主线）

**目标**：用最少机制复现真实数据的核心几何：held-out rank stability、split-half / odd-even 稳定、forward / reverse 反相关、full model 高于几何采样 controls、事件率可变但模板几何稳定。`k=2` 和 raw identity bias 是描述性输出，不是 primary success criterion。**不做 patient-specific fitting，不用 clinical SOZ 反向拟合参数，不模拟 HFO carrier。**

**前置**：Topic 0 phantom-rank 修复已完成；真实数据 pipeline 可用于 synthetic data。Phase 1-3 的实证合同继续作为验收标准，但 v2 建模路线不再等待 HR/FHN route。

> **2026-06-03 amendment 适用于本 Step 列表**（详见顶部 banner）：建模目标 = SOZ 小邻域网格内**时间离散自终止**的群体事件（**非**空间次网格波，波可填满网格）；Step 0b 的 recovery 为**并列机制分支**（纯抑制自限 vs 加恢复变量自限 report-both，去留由实测事件时长/空间范围/自限区宽度定）；亚临界噪声触发为离散事件首选路径；发作样 recruitment 只作 H5 / Phase 3 候选机制，**不**入本轮纯间期主验收。
> **→ 2026-06-03 晚 v2 update**：**Step 0b 已通过 mechanism-scale gate**（LIF transfer，非 sigmoid）；**Step 1 解锁**，带两候选窗进入——(a) Brunel-like 短/稳健/recovery-off 主线 + (b) 病理高激+快回拉 envelope-时长 sensitivity。Step 1 验收 = 无噪声不自发持续响 + 噪声→离散事件（非持续振荡）+ 自终止 + 方向/通道顺序可被真实 pipeline 读出 + 事件率粗量级。详见顶部 v2 verdict + `docs/archive/topic4/sef_itp_phase4_v2/lif_transfer_route_2026-06-03.md`。

**Step 0a：effective gain + 线性稳定性分析**

- 把 `sigma_phi(x)` 写进 `F_eff(h;x)`，计算当前工作点下的局部 gain。
- 用二维 E-I rate field 求 `max_k Re(lambda(k))`。
- 必须找到三类区域：小扰动稳定区、候选可激窗、小扰动失稳区。
- 输出 gain map、phase diagram、`max_k Re(lambda(k))` heatmap、`k*` heatmap、candidate working point 和 boundary。

**Step 0b：finite-pulse response map**

- 在预注册 pulse family 上扫描有限幅刺激，分类为 extinction / local bump / self-limited propagation / runaway。
- 定义 `A_event` 和 `A_runaway`。
- 真正的间期候选窗必须满足 `A_event < infinity` 且 `A_runaway - A_event` 有正安全边界。
- 输出完整 response surface，不能只挑一个好看的 pulse。

**Step 1：最小 2D rate field**

- 在 Step 0a/0b 锁定的候选窗加入 OU / Gaussian noise。
- 验收：无噪声无持续事件；加噪声后出现离散、自限、空间范围合理的群体事件；事件落在 self-limited propagation 区；事件率可调到真实数据同一量级。

**Step 2：加入 E→E 各向异性 + controls**

- 把 `W_EE` 改成椭圆核，`ell_parallel > ell_perp`。
- 主验收：held-out rank stability、split-half / odd-even stability、inter-template anti-correlation、finite-pulse self-limited propagation 均高于 controls。
- 必跑 controls：isotropic connection + aligned shaft、anisotropic connection + random shaft、rotated shaft、jittered contacts、shuffled contact identity、multiple shaft orientations。
- `k=2` 和 identity bias 数值范围只报告为 descriptive。

**Step 3：加入局部低异质性 patch**

- 第一版只降低 `sigma_phi(x)`，不同时调多个生物物理旋钮。
- 验收：事件成核点向 patch 聚集；source / endpoint 与 patch 有统计关系；移动 patch 后 source density 随之移动；不强制 source 等于 clinical SOZ。

**Step 4：搬到 LIF E-I SNN**

- 只有 Step 0-3 通过后才启动。
- 用 conductance-based LIF 验证 rate 层机制在 spiking 层仍成立；虚拟 SEEG 用突触电流 proxy + envelope。
- synthetic data 必须走真实 PR-2 / PR-2.5 / endpoint pipeline。

**Step 5：加入慢变量**

- 第一版只用抽象 `q(t)` 调制 `eta(x,t)`。
- 验收：事件率升高但 rank template 不大幅改变；forward / reverse 不需要强时间配对；`eta(x,t) < 0` 时能在仿真中进入 ictal-like recruitment。
- `z_I(t)`、`g_K(t)`、`E_GABA` 漂移只作后续机制分解，不进第一版主分析。
- ictal-like recruitment 只作为 synthetic feasibility bridge，不作为 clinical seizure onset claim。

**Step 6：回到真实数据做预测验证**

- 用一半事件估计传播轴和 endpoint，另一半验证稳定性。
- endpoint / source 与 clinical SOZ、ictal onset、影像病灶、resection zone 只做 held-out 关系检验。
- endpoint 不等于 clinical SOZ 不算模型失败；真正失败是 held-out 内部不稳定，或模型不能复现 identity bias / template stability。

**产出目录**：`results/topic4_sef_hfo/{linear_stability,rate_field,anisotropy,low_heterogeneity_patch,lif_snn,slow_variable_bridge,synthetic_vs_real}/`，含 figures + cohort/stat JSON + figures/README.md。

**历史关系**：`results/topic4_sef_itp/phase4_fhn_toy/` 和 `docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md` 保留为 HR/FHN route 历史探索 / sensitivity，不再是主模型路线。详细 v2 plan 见 `docs/archive/topic4/sef_hfo_topic4_v2_plan_2026-06-01.md`。

---

## 7. 模型支持 / 不支持的解释

### 7.1 模型能解释的现象

1. 为什么同一 subject 内存在稳定传播模板（→ 固定各向异性传播轴 + 固定电极几何；但必须高于几何采样 controls）
2. 为什么正反模板共享 source/sink 几何骨架（→ 同一条轴从两端随机成核，而非两套网络）
3. 为什么 identity bias 高（→ 通道 rank 主要由电极在传播轴上的位置决定；raw 数值只作描述性输出）
4. 为什么模板选择在已测试尺度上近似随机（→ 噪声决定每次从哪端点燃，短时间强配对不是必要机制）
5. 为什么 event rate 可受慢变量调制，而模板几何相对稳定（→ 抽象慢变量调 `eta(x,t)`，不改连接轴）
6. 为什么同一参数族在仿真中可能从自限传播进入持续招募（→ synthetic feasibility bridge，不是 clinical seizure onset claim）

### 7.2 模型暂时**不支持**的强说法（写死 out-of-scope）

1. ❌ endpoint 就是 SOZ 边界
2. ❌ endpoint 一定在 SOZ 内
3. ❌ endpoint 一定是 seizure onset channel
4. ❌ source 是兴奋驱动，sink 是抑制反弹（**HFO 80–250 Hz 不分 E/I**，红线）
5. ❌ 正反模板证明了双稳态 attractor 或强 ping-pong 机制（H3 仅声明 "compatible with mark-independent within tested precision"）
6. ❌ 模板选择已经被证明完全独立（同上）

### 7.3 文献 framing：具体机制多样，中观动力学收敛

这批文献的作用不是“证明我们的 HFO 模型”，而是给 Topic 4 的建模地位划边界：钾离子、钠钾泵、氯离子 / GABA、胶质缓冲、抑制失效、突触短时可塑性、结构连接和局部网络异质性都可能参与癫痫；但在 HFO 群体事件这个尺度上，它们共同改变的是局部网络多容易被点燃、点燃后多快恢复、有限扰动是否会扩展成更大空间招募。

因此 SEF-HFO 不把某一个细胞机制硬塞成“真实慢变量”。它只检验一个更抽象、更可证伪的问题：如果局部组织处在稳定但可激、接近临界但未失控的状态，并且连接有固定空间轴，那么噪声触发的间期 HFO 群体事件是否会留下稳定的通道先后顺序；当慢状态把系统推近边界时，事件率和招募是否可以增加，而传播几何不被重写。

引用分层写法：

- Jirsa 2014 / Chizhov 2018 / Wendling 2005 支撑“间期到发作可以被看成慢状态推动的动力学转变，而不是单一细胞机制的直接结果”。
- Cressman 2009 / Wei 2014 / Ho and Truccolo 2016 支撑“离子、泵、胶质和抑制等不同机制可以收敛到网络稳定性和易激性的变化”。
- Proix 2018 / Naze 2015 / Wang 2016/2017 支撑“空间场、连接结构和周边组织可招募性会塑造传播形态”。
- Chang 2018 / Maturana 2020 / Lepeu 2024 支撑“恢复能力、临界接近和有限扰动响应是比静态稳定性更贴近发作易感性的描述”。
- Zijlmans 2010/2011 / Weiss 2013 只支撑“HFO 是合理观测对象且与癫痫组织有关”，不能用来声称“间期 HFO 就是微型发作”。

安全核心句：

> We do not assume a single cellular mechanism linking interictal and ictal activity. Instead, we use a dynamical abstraction: diverse biophysical mechanisms may converge onto a shared change in local excitability, resilience, and finite-amplitude recruitment. Under this view, interictal group-HFO events are modeled as isolated excursions of a pathological excitable field, whereas ictal-like recruitment corresponds to slow-state-gated clustering or spatial expansion of similar elementary events.

详细文献分层和引用位置见 `docs/archive/topic4/sef_hfo_topic4_v2_plan_2026-06-01.md` §6。

---

## 8. 红线（继承 paper1_framework_sba.md v1.1.2 红线 + 本框架新增）

1. **HFO 80–250 Hz 不分 E/I**：framework / archive / paper / toy / 任何 plan 全文严禁出现 "兴奋驱动 / 抑制反弹 / 证明机制" 表述
2. **H3 措辞锁**："compatible with mark-independent sampling within tested precision"；禁止写 "证明独立"，禁止用 leave-one-subject-out 替代主判据
3. **不预设 clinical SOZ = true SOZ**：H1/H2/H5 的 endpoint 不被假定为 SOZ 边界 / SOZ 内 / seizure onset；与多源 SOZ proxy 关系并列报告
4. **不把低异质性直接等同易激性**：`sigma_phi(x)` 只能表示局部 threshold variance；它必须通过 `F_eff -> G -> lambda -> eta_lin` 实际改变局部稳定性，才允许进入“更接近可激窗”的解释。禁止回退到 v1 的抽象 `θ(x)` 场，或写成 `sigma_phi down => eta_lin down`。
5. **directional predictor 用 sin 不是 cos**：如果未来本 framework 扩展到 P5 等价物（间期 → 发作 directionality），必须用 sin-based 或 rank-based，**禁止** cos-based directed graph（继承 SBA v1.1）
6. **k = 3 主预测，k ∈ {2, 4, 5} 敏感性**：endpoint 定义的 k 在 framework time lock，**禁止**事后调整
7. **H1 三层拆分 lock**：H1 必须拆 H1a（within-source compactness）+ H1b（within-sink compactness）+ H1c（envelope within participation field）；**禁止**回退到 `mean pairwise distance within (S ∪ K)` 的 v1 单一度量（与 H2 概念冲突）
8. **H2 当前主合同 = 通道层角色互换 + source/sink 两侧空间紧凑**：H2 channel-label 必须来自 masked rank-displacement `swap_sweep` 的 variable-k 输出；空间层分别检验 source-side 与 sink-side swap-k 节点相对 `all mapped SEEG minus swap endpoint` 的 compactness。早期 set-based Jaccard + centroid-distance reversal index 只保留为 v1.0.2 audit trail，**禁止**作为当前 H2 主分析复活。
9. **H3 verdict naming = SUPPORTED / NOT SUPPORTED / CONTRADICTED**：**禁止**用 PASS / NULL / FAIL（防止 "PASS = 证明独立" 误读）
10. **H4 normalized instability + Cohen's d ≥ 0.3** lock：**禁止**回退到 `CV(rate) ≥ 3 × CV(geometry)` 的 v1 硬阈值（量纲不同 + CV 不稳）
11. **H5 统计合同 lock**：min event ≥ 30 / time-of-day matched baseline / subject-level aggregation / fraction-vs-0.25 Wilcoxon / power floor 6 subject；**禁止**裸 channel-level t-test 或不校正 FDR
12. **H6 shaft-stratified shuffle 主 null**：lock at framework time，**禁止**事后换 null 形式

---

## 9. 与现有 framework / topic doc 的关系

### 9.1 SBA framework 取代关系

- **取代** `docs/paper1_framework_sba.md` §4（BHPN-toy 数学规范）
- **取代** `docs/paper1_framework_sba.md` §6（fitted model BHPN-fit spec 的 toy mechanism 子句）
- **取代** `docs/paper1_framework_sba.md` §7（5 dumb baselines 中针对 toy 机制的项）
- **保留** `docs/paper1_framework_sba.md` §5.1 P1（template stability，**待 Phase 0 valid-only 重跑后确认**）
- **保留** `docs/paper1_framework_sba.md` §5.2 P2（geometric backbone，**待 Phase 0 valid-only 重跑后确认**）
- **保留** `docs/paper1_framework_sba.md` §5.3 P3 PR-7 addendum 锁字 → 直接映射为本文 H3
- **保留** `docs/paper1_framework_sba.md` §5.4 P4 anatomical anchoring（部分被本框架 H5 短时版本覆盖，长时 P4 PR-8 v2 路径仍单独存在）
- **保留** `docs/paper1_framework_sba.md` §5.5 P5 directional predictor 红线（sin 不是 cos）

### 9.2 Topic 1 / Topic 3 / Topic 4 文档双向链接

- `docs/topic1_within_event_dynamics.md` §2 当前结论加入 SEF-ITP framework 入口链接
- `docs/topic3_spatial_soz_modulation.md` §2 加入与 SEF-ITP H5 关系说明（endpoint identity shift × data-driven SOZ）
- `docs/topic4_sef_itp_framework.md`（本文）= Topic 4 模型层 formal entry
- `docs/paper_overview.md` 索引在 Topic 4 行加入本文链接

### 9.3 Archive 关系

- `docs/archive/topic4/pr_t4_1_bhpn_toy/pr_t4_1_bhpn_toy_plan_2026-05-01.md` 顶部加 **SUPERSEDED** banner 指向本 framework
- `docs/archive/topic4/INDEX.md` 更新指向本 framework（formal entry 不再是 "待写"）

### 9.4 PR-T3-1 数据驱动 SOZ 的关系

- PR-T3-1 Layer B 输出（`results/data_driven_soz/layer_b_labels/`，**当前不存在，待 PR-T3-1 Layer B 启动后产出**；Layer A 已落地于 `results/data_driven_soz/layer_a_ictal_er_rank/`）是 H5 检验的**第二份独立 SOZ 标签**（避免循环验证）
- 任何 H5 / H1 与 SOZ 关系的报告**必须**同时引用 clinical SOZ + data-driven SOZ 两个来源，不融合
- PR-T3-1 v2.2.3 `lambda_fragile` flag 必须随标签一起传递到 H5 下游 reader

---

## 10. 中心段落（草稿，供未来 paper / abstract 使用）

> 我们假设，间期群体 HFO 传播模板来自一块局部低异质性、连接有固定方向、接近临界但仍未失控的 E-I 易激斑块。低异质性本身不是结论；它必须先改变群体输入-输出曲线和局部 gain，再在线性稳定性与有限幅脉冲仿真中表现为“能被点燃、但能自限熄灭”的工作窗。稳定的 endpoint 不被假定为 SOZ 边界，也不被强制假定为 SOZ 内部节点，而是模型预测的传播几何锚点。正反模板来自同一条各向异性传播轴从两端随机成核，而不是两套独立网络或强 ping-pong 机制。慢变量主要调节事件发生率；如果它重写传播几何，模型失败。发作样持续招募只作为仿真可行性桥接，不作为 clinical seizure onset 解释。

英文版（供 abstract / 投稿）：

> We hypothesize that interictal group-HFO propagation templates arise from a locally low-heterogeneity, anisotropically connected E-I excitable patch that is near critical yet still subthreshold. Low heterogeneity is not assumed to imply excitability directly; it must enter the effective population transfer function, alter local gain, and then produce a linear-stability and finite-pulse working regime in which finite perturbations can trigger self-limited propagation without runaway recruitment. Stable source/sink endpoints are not assumed to be SOZ boundaries or restricted to SOZ interior; rather, they are predicted geometric anchors of propagation. Forward/reverse template pairs arise from stochastic nucleation at opposite ends of the same anisotropic propagation axis, not from two independent networks or a forced ping-pong mechanism. Slow modulation changes event probability while the underlying template geometry remains relatively stable; ictal-like recruitment is treated only as a synthetic feasibility bridge, not as an explanation of clinical seizure onset.

---

## 11. 自检清单（v0.2 当前合同 + v1 真实数据合同）

- [x] 单核心断言清晰（局部低异质性 + 各向异性连接 + 近临界但亚阈值 E-I 易激斑块；低异质性必须经 `F_eff -> G -> lambda -> eta_lin` 闭合）
- [x] H1–H6 每条有 verdict 标签 + 数字判据 lock + 三段式描述
- [x] H1 拆三层（H1a within-source / H1b within-sink / H1c envelope-within-field）避免与 H2 概念冲突
- [x] H1 endpoint k=3 主预测 + {2, 4, 5} 敏感性 lock
- [x] H1 四种距离（Euclidean / shaft-ordinal / cortical surface / SC）并列报告 lock
- [x] H2 主合同 = masked rank-displacement variable-k channel labels + source-side / sink-side swap-k spatial compactness；不写单一 "H2 PASS"
- [x] H2 早期 set-based Jaccard / centroid-distance reversal index 与 PCA axis projection 只保留为 v1.0.2 audit trail，不作当前主分析
- [x] H3 verdict naming = SUPPORTED / NOT SUPPORTED / CONTRADICTED（不用 PASS 防误读）
- [x] H3 措辞锁 "compatible with mark-independent within tested precision"（继承 PR-7 addendum 2026-05-01）
- [x] H4 normalized instability vs matched null + Cohen's d ≥ 0.3 lock（v1 CV ≥ 3× 已废）
- [x] H5 统计合同 lock：min event 30 + time-of-day matched baseline + subject aggregation + fraction-vs-0.25 + power floor 6
- [x] H5 短时（pre/post-ictal, hours）vs 长时（months）拆分；长时遗留 future work
- [x] H6 shaft-stratified shuffle 主 null + 全脑 shuffle 敏感性 lock
- [x] H6 优先级最高（基础证伪），实操顺序 H6 → H1 → H2 写入 §4
- [x] Phase 1 失败模式表 pre-register（H6 × H1 × H2 三轴 × 5 verdict 行）
- [x] Cohort tier criterion 措辞锁，数字 TBD pending Phase 0
- [x] Tier 1+ sub-cohort（H2 PCA sensitivity）单独 criterion lock
- [x] Phase 0 phantom-rank 修复硬前置，禁止跳过
- [x] HFO carrier 振荡机制不在本 framework 范围（红线 §8.1）
- [x] `sigma_phi(x)` 必须通过 `F_eff -> G -> lambda -> eta_lin` 闭合，禁止直接写 `sigma_phi down => eta down`
- [x] Phase 4 v0.2 主路线 = effective gain → linear dispersion map → finite-pulse response map → rate field → controls → LIF SNN → 抽象慢变量桥接；HR/FHN route 降级历史 / sensitivity
- [x] `k=2` 与 raw identity bias 降级为 descriptive；主验收必须包含 held-out stability + controls fail
- [x] ictal-like recruitment 只作 synthetic feasibility bridge，不作 clinical seizure onset claim
- [x] 与 PR-T3-1 数据驱动 SOZ 的双标签合同写明
- [x] SBA framework 取代 / 保留范围明确（§9.1）
- [x] Topic 1 §2 + Topic 3 §2 保留 Topic 4 模型层链接；v0.2 主入口指向 SEF-HFO / SEF-ITP formal entry
- [x] Out of scope 包括 Topic 2 事件间周期 / HFO carrier 细胞生物物理 / patient-specific fitting
- [x] CLAUDE.md §8 大白话风格全文遵循（codename 仅作括号补注）

---

## 12. 一句话承诺（结尾）

我们把 Topic 4 的模型层从 **"塞 Hebbian 矩阵让 Kuramoto 演化得到预设吸引子"**，再推进到 **"低异质性 + 各向异性连接 + 近临界 E-I 易激场"**。第一步不是上复杂 SNN，而是先把低异质性写进群体输入-输出曲线，实际计算它是否改变局部 gain 和线性稳定性；线性图只给候选工作区，有限幅脉冲图才证明“能点燃但不失控”。随后才用二维 rate field 检查稳定正反传播几何，并且必须跑电极几何和采样方式的 negative controls。SNN 和慢变量只是后续验证：前者验证 spiking 层能否保留同一几何，后者先用抽象慢变量验证“事件率变、传播几何不变”。所有 synthetic data 必须走真实模板 pipeline；clinical SOZ 只作 held-out 关系检验，不作拟合标签。

---

## 13. 历史文档索引

- `docs/archive/topic4/sef_hfo_topic4_v2_plan_2026-06-01.md` —— **CURRENT v2 plan**，SEF-HFO 主模型路线（线性稳定性 → rate field → LIF SNN → 慢变量桥接）
- `docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md` —— **SUPERSEDED as main route**，HR/FHN Phase 4 route，保留为历史探索 / sensitivity
- `docs/paper1_framework_sba.md` v1.1.2 + PR-7 addendum 2026-05-01 lock —— 上游 SBA framework；本框架取代其 BHPN-toy 部分
- `docs/archive/topic4/pr_t4_1_bhpn_toy/pr_t4_1_bhpn_toy_plan_2026-05-01.md` —— **SUPERSEDED**，BHPN-toy plan-of-record v2，归档
- `docs/archive/topic4/layered_model_framework.md` —— 更早的分层模型框架（已被 SBA 取代）；保留为历史
- `docs/topic0_methodology_audits.md` §3.1 + §5 —— phantom-rank 修复 + broad re-derivation roadmap（本框架硬前置）
- `docs/archive/topic1/propagation/topic4_attractor_diagnostics_step1_results_2026-05-10.md` —— Topic 4 attractor Step 1 结果（principal curve + λ₂ transition）；待 Phase 0 §5h 在 masked features 上重跑
- `docs/topic3_spatial_soz_modulation.md` —— Topic 3 PR-T3-1 数据驱动 SOZ（H5 第二标签来源）
