# M3 local-W kick-calibration fine-scan — recap（2026-06-23）

探索性 SNN fine-scan 收口。**不冻结预注册，不进 μ 相图，不估正式 W。** 本文锁"薄结果"口径，
并按用户 2026-06-23 review 把判据改成 **event-aligned (EA) 为 primary、固定窗为 sensitivity**。

数据：`results/topic4_sef_hfo/m3_local_w/kick_calibration_explore/`（gitignored）。
重判脚本：`scripts/reclassify_m3_ea_primary.py` → `finescan_ea_primary/`。

---

## 朴素话先讲（测了什么 / 怎么测 / 揭示了什么）

**测了什么**：在一块模拟脑组织薄片上戳一下，看激起的活动会不会就地传一小圈然后自己平息（"局部+回静的有限小爆发"，
用来代表发作间期 HFO）。比的是：加一个"病灶核"（把局部一小撮神经元调得更易兴奋）能不能让同一片组织在更小的戳下、
更干净地出这种事件——**注意：不是指望核造一条新路径，而是看核是否把同一传播场推近"招募边界"（降低触发阈值 K_min，但传播形状不变）。**

**怎么测**：每一戳分三路——戳了减没戳（看就地传多远）、单看戳了那条原始记录（看回静还是失控）、单看没戳只有底物（看底物自己会不会无端点火）。
关键改动：把观察窗从"固定时刻"改成"锚在事件真正起点之后"（EA），因为事件起点会随戳/核/种子漂移，固定窗可能截到事件的某个片段、把空白误判成局部。

**揭示了什么**：小幅线性响应那条路（戳越轻响应越成比例）在当前设置下**看不到**；能看到的是一个**有限幅、能回静、不失控的事件**，
但它**不是核独有**的——空白薄片在最重一戳下也能在固定窗里出"像局部"的事件。只有把判据换成事件对齐窗后，
**偏易兴奋的两个窄核（平均阈值<18）才比空白和"平均=18"的核出更干净的局部事件**；方向一致，但样本只有 8 个种子、统计上还不能下结论。

---

## A / B / C 分流（当前设置下）

- **A（small-kick 线性 W_small）**：**不支持**。无单调的小幅线性响应区。
- **B（finite-amplitude W_event）**：**B0 支持；B2 经 ceiling 扩展后被支持；B1/B3 待做**（见下 B0–B3 分层 + Step 4 ceiling 结果）。
- **C（local-W 什么都没有）**：**否定**。确实存在有限幅、回静、不失控的事件。

核特异性现状（**经 kick ceiling 扩展更新**）：原 10-kick scan 只到 1.2，看不清；扩到 1.6 后看到——
**核显著提高同一 kick 下出 EA-local 事件的概率（OR≈4.5×），并在多数 seed 上把触发阈值小幅提前（≈0.1 kick、0 推后），而不是造一条新路径**。
这正是 review 假设的 B2 机制（概率层）。⚠️ 阈值"1.1 vs 1.6"是 cohort 概率曲线在 0.7 这条线的跨越点（最阈值敏感的 lens），**不是 seed-level 阈值稳降 0.5**——seed-level 实际是 ≈0.1 kick 的小幅一致提前（详见 Step 4）。

## Step 4 — kick ceiling 结果（决定性升级，bare + n17.6，kick 1.0–1.6，12 seed）

`scripts/run_m3_kick_calibration.py`（explore，无 engine/detector/阈值改动）→ `kick_ceiling_{bare,n17.6}/`；
重判 → `finescan_ea_primary_ceiling/`。

EA-local-returned 比例（P_EA，12 seed）随 kick：

| kick | bare P_EA | n17.6 P_EA | bare 类 | n17.6 类 |
|------|-----------|------------|---------|----------|
| 1.0 | 0.25 | 0.583 | nonlocal_returned | fixed_only_local |
| 1.1 | 0.583 | **0.75** | nonlocal_returned | **EA_local_returned** |
| 1.2 | 0.417 | 0.833 | fixed_only_local | EA_local_returned |
| 1.3 | 0.583 | 0.833 | fixed_only_local | EA_local_returned |
| 1.4 | 0.583 | 0.917 | fixed_only_local | EA_local_returned |
| 1.6 | **0.833** | 1.0 | **EA_local_returned** | EA_local_returned |

- **阈值，三个 lens 要分清**（artifact：`finescan_ea_primary_ceiling/seed_kmin_shift.json`）：
  - **cohort 概率曲线（headline lens，最阈值敏感）**：当前-kick-局部比例 P_EA≥0.7 的跨越点 = bare **1.6** vs n17.6 **1.1**。
    这条 1.1-vs-1.6 被 0.7 这条线 + bare 曲线在 1.2 的非单调凹陷放大了，**不等于每个 seed 阈值降 0.5**。
  - **cumulative-ever-crossed lens（稳健）**：累计曾跨过 ≥0.7 = bare **1.1** vs n17.6 **1.1** —— gap 几乎消失。
  - **seed-level 配对（最该信）**：逐 seed 首次 EA-local kick，n17.6 vs bare = **7 早 / 5 同 / 0 晚**，
    sign test p=**0.0156**，中位位移 **−0.1 kick**（≈一个网格步，bootstrap CI [−0.1, 0]），1 个 bare seed 在 1.6 内从不跨（censored）。
  - **措辞锁**：核**显著提高同一 kick 下出 EA-local 事件的概率**、并在多数 seed 上把阈值**小幅**提前（≈0.1 kick、0 推后）；
    **不写"seed-level K_min 稳降 0.5"。**
- n17.6 EA-local band = **1.1–1.6 连续**；**全程不失控**（`finescan_ea_primary_ceiling/` reclass：P_runaway=0、P_returned=1 对 bare+n17.6 所有 kick 直到 1.6），bare 在 1.6 也出**同款** EA-local 事件（r95~4mm）——
  同一种事件、核更早点着 = phenotype 不变（W_shape 是否真同 → B1）。
- **汇总检验（reviewer 建议 local~substrate+kick；artifact：`logit_substrate_kick.json`）**：Logit，
  substrate(core) **OR=4.49，95% CI [2.89, 6.97]**；cluster-robust（按 seed）p<1e-4；
  **seed-block 符号置换 p=0.0007**（12 个 seed cluster 偏小，置换是小样本稳健检验，确认 OR 非 cluster-SE 伪影）。
  连续 far_ea 效应量 dz(bare−core) 每个 kick 都 ≈ **1.0–1.34**。**这是 B2 概率层最干净的证据。**
- **仍需保留的限制**：① 只 bare vs n17.6（n17.8 未进 ceiling，但 10-kick scan 中 ≈n17.6）；
  ② **"同一种事件/W_shape 不变" 是从 phenotype（r95、回静、不失控）推断的，未在传播主轴/顺序层面验证 = B1 仍待做（mini-W_event）**；
  ③ kick 是合成外驱，K_min 是模型内部量；④ 逐 kick 二值 perm p 仍不显著（强度在汇总 OR + K_min 分离 + 一致 far 效应量）。

---

## EA-primary 重判结果（finescan_ea_primary/）

定义：seed 算 EA-local-returned ⇔ r95_ea ≤ 6mm 且 far_ea ≤ 0.5 且 returned；P_EA = 占比；EA-local cell ⇔ P_EA ≥ 0.7 且 n_seeds ≥ 6。
（EA 字段跨三窗恒定，已验证。）

| 底物 | kick=1.0 类 | kick=1.2 类 | kick=1.2 P_EA |
|------|-------------|-------------|---------------|
| bare | nonlocal_returned | **fixed_only_local** | 0.50 |
| n17.6 | fixed_only_local | **EA_local_returned** | 0.875 |
| n17.8 | nonlocal_returned | **EA_local_returned** | 0.875 |
| n18.0 | nonlocal_returned | **fixed_only_local** | 0.50 |
| w18.0 | nonlocal_returned | EA_local_returned* | 0.75 |

\* w18.0 含 1/8 自发点火 seed 污染（见下）。

**关键**：
- bare kick=1.2 = `fixed_only_local`（固定窗 0.875 但 EA 只 0.5）→ 固定窗把它误判局部，EA 揭示非局部。
  **所以固定窗 FINITE_THRESHOLD 不能作 W_event 主证据。**
- **EA-primary 下 n17.6 ≈ n17.8**（都在 1.2 EA_local_returned）。早先对抗式验证说"只有 n17.6 核特异"是**固定窗伪影**
  （n17.6 在 kick1.0 固定 P=0.75 但 EA P=0.5）。真正区分轴 = **核平均阈值<18（17.6/17.8 赢）vs =18（bare、n18.0 输）**。

## 配对统计（n17.6 vs bare，EA，**原 10-kick scan n=8 —— 历史记录，已被 Step 4 ceiling n=12 取代**）

> 本节是 ceiling 前的弱版（n=8 只到 kick 1.2）；当前 B2 口径以上面 Step 4 ceiling（n=12，OR=4.49、seed-block 置换 p=7e-4）为准。保留作演进记录。

- 三个 kick（0.85/1.0/1.2）ΔP_EA 都 = **+0.375**，bootstrap 95%CI **[0.125, 0.75]**（不跨 0），
  但 **paired permutation p ≈ 0.25（不显著）**。
- 连续效应量：far_ea 的 dz(bare−core) ≈ **0.95–1.14（大）**；r95_ea 的 dz ≈ 0.6（中）。
- **判定：弱候选**。方向一致、连续空间指标效应量不错，但 n=8 欠功效、二值 ΔP 的 CI 很宽。
  cap 敏感性：n17.6 的 core-specific EA-win 只在 r95cap 6–7mm 成立（对 far cap 稳健，含严格 0.08），≤5.5mm 就消失。
  → 只能写 **weak spatial-locality separation**，不能只靠单条 6mm cap。

## w18.0 更正（重要：推翻 2026-06-23 早先 8h 报告的判断）

早先报告写 "w18.0 不戳基线 = 188.8 = 5.8×bare、恒定自活跃、自燃门误标安静" —— **错误**。逐 seed 数据：
`core_only_downstream`(kick1.2,win22-32) = [19,24,29,32,33,40,51,**1282**]。

- 188.8 是 **8 个 seed 的均值，被那 1 个 1282 离群 seed 拉高**；**中位数 32 ≈ bare**，**7/8 个 seed 安静**。
- `core_only_quiet` 是**多数票**（7/8 安静 → 标 quiet），所以门**判断正确，并非误标**。
- 大幅自发点火（co_ds>100）：窄核 17.6/17.8/18.0 **全 0/8**；只有 w18.0 是 **1/8**（跨 kick 恒定 = 同一 core_only 复用）。
- 结论：**w18.0 = 偶发自发点火（1/8 seed），不是恒定自燃**；它不是"被污染的负对照"，而是"会偶发自燃的正对照"。
  narrow cores 确认**真安静**（0/8）。

**Step 5 门改进的真正方向**：不是"判这个底物自活跃"，而是加一个**per-seed 自发点火 flag**，把那 1/8 个 igniting seed 单独剔除
（多数票会掩盖少数 igniting seed）。

---

## Branch B 分层（新定义，避免过度解释）

- **B0 — finite-event feasibility**：存在有限幅、回静、不失控的事件。**支持。**
- **B1 — event-conditioned propagation operator**：这些事件的早期招募形状稳定、能定义 W_event、比 distance/rate 更能预测传播顺序，
  且核与 bare 的 W_shape 一致。**未做（mini-W_event / Step 6 才能验证）——这是 B2 当前唯一的缺口。**
- **B2 — pathology/permissivity 移动事件边界**：核提高有限事件概率 / 小幅前移触发阈值，**但不改变传播主轴/phenotype**。
  **经 ceiling 扩展后被支持（概率层）**：核 OR=4.49 [2.89,6.97]（seed-block 置换 p=7e-4），逐 seed 阈值 7 早/5 同/0 晚（sign p=0.0156，中位 ≈0.1 kick），无失控，bare 在更高 kick 出同款事件。
  **不写"K_min 稳降 0.5"（那是 cohort 概率曲线 0.7-lens 的放大）。唯一缺口 = W_shape 不变尚未在主轴层面验证（B1）。**
- **B3 — interictal→ictal bridge**：提高 μ 后同一 W_event 从回静有限事件转为持续招募。**未开始（B1 确认前不能做）。**

**本轮口径：B0 yes；B2 概率层 yes（核提高事件概率 OR≈4.5 / 小幅前移阈值 ≈0.1 kick，已统计支持；非 0.5 稳降），但其"W_shape 不变"前提待 B1 验证；B3 not started。**

## 预注册 / 相图门槛

- **可以进入预注册讨论，但不冻结**（preregistration discussion candidate，NOT freeze）。**B2 已到概率层支持**（核 OR≈4.5、seed-block 置换 p=7e-4、ceiling n=12）；
  但 B1（W_shape 不变）未验证，且核效应在阈值上是 ≈0.1 kick 的小幅前移（非 0.5 稳降）——下一步是 mini-W_event，不是冻结、不是相图。
- **下一步是 targeted fine-scan，不是 μ 相图。** 进相图前至少要：EA-primary 有限回静事件稳定 + W_event_shape 可复现 +
  W_event 比 distance/rate 更能预测 event order + 核或 μ 至少能降 K_min 或升有限事件概率 + 固定窗-only 不作主证据。
  相图横轴届时应叫 **Λ_event / R_event**（finite-event recruitment gain），不叫 Λ0(W_small)。

## 执行顺序（用户 review 6 步）

1. 本 recap（锁薄结果口径）— **本文**。
2. EA-primary 重判 + cap 敏感性 + discordance 表 — **完成**（`scripts/reclassify_m3_ea_primary.py`，`finescan_ea_primary/`）。
3. n17.6 vs bare 配对统计（原 n=8）— **完成**（`paired_stats.json`，弱版，已被 ceiling 取代）。
4. kick ceiling 扩展（bare + n17.6，kick 1.0–1.6，12 seed）— **完成**（B2 概率层支持；`finescan_ea_primary_ceiling/`）。
   B2 统计 artifact（P1）— **完成**：`scripts/analyze_m3_ceiling_b2_stats.py`（9 TDD）→ `seed_kmin_shift.json`（seed-level 阈值/sign/survival）+ `logit_substrate_kick.json`（系数表 + seed-block 置换 + cluster bootstrap）。
5. core_only 逐 seed 污染审计（Step B，门按 reframe **不改 substrate 级多数票门**，只在分析层加 per-seed flag + median 汇总）— **完成**（`scripts/audit_m3_core_only_seed_confounds.py`，5 TDD；唯一污染=w18.0 1/8，B2 底物 bare/n17.6 全 0 → B2 不依赖污染 seed）。
6. mini-W_event pilot（B1 验证：W_shape 是否一致/沿 E→E 轴/能预测 order；5 源 × source-specific K50 × 12 seed；K_min·W_shape·P_escape 三件不混）—
   设计稿：[[m3_mini_w_event_design_2026-06-23]]。
   **6a. Step D 基础设施 — 已建 + 已验证 + 已提交**（pilot RUN 仍 PILOT-FIRST 待用户放行）：
   - runner `--kick-xy`（多源 kick，engine 不动；`--kick-xy==core_center` 与现有逐字节 bit-parity，off-center 改变输出=参数有效）— commit a612e3a。
   - runner `--emit-ea-bins` → `ea_net_bins.npz`（W_shape 原料=每 (kick,seed) 的逐 bin 事件对齐差分；默认 OFF 时既有产物逐字节不变）— commit 6e57bdd。
   - `src/sef_hfo_mini_w_event.py`（13 TDD）：`extract_kmin/k50`、`build_w_shape`（源排除+逐seed归一+只取成功seed）、`w_shape_reproducibility`（B1a，bin-shuffle null）、`success_seeds_at_kick`。**复用上游门**：`_ea_local_flag`（reclassify）、`spontaneous_ignition_flag`（Step B），不重写 — commit 4502cb3。
   - `scripts/run_m3_mini_w_pilot.py`：5 源几何（center/±axis@45/±offaxis@135，R_src=4mm，core 中心 [10,10]）+ K_min(q) 图 + center W_shape 可重复性图 + README；geometry 4 TDD、dry-run 10 命令、L=8 smoke 端到端跑通（10/10，无事件时优雅 no_ea_local）— commit 10ea46c。
   **6b. pilot RUN（L=20，2 底物 × 5 源 × 7 kick × 12 seed）— 已跑（2026-06-24）**。
   - **网格 bug + 修复（必须记住）**：首轮用了 `n_bins_per_axis=4`(16 格)，但 ceiling 工作点是 **5×5**(25 格，记在 `thresholds.json::n_bins=25`，**不在** `config.sweep_parameters` 里，所以配置对比漏掉了)。4×4 时薄片正中心 [10,10] 卡在 4 格交界缝，`_spatial_extent` 的半径参考点 `bin_centers[src_bin_idx]` 偏 3.5mm → r95_ea 虚高 6.9 vs 5.2、far 0.63 vs 0.02，中心事件全被误判非局部 → P_EA=0。**根因 = orchestrator 参数填错，引擎一字节没动**。修复=`n_bins_per_axis 5`（commit aad1881，含守卫测试钉死「L/2 必须是格心非交界」）。系统调试见 §systematic-debug-2026-06-24 below。
   - **结果（5×5，干净，精确重现 ceiling）**：K_min(q) center bare **1.6** vs n17.6 **1.1**（= ceiling），偏轴源 bare/n17.6 收敛到 1.0–1.2 → **核降阈是局域的（中心最强、偏轴减弱），不是全局**（偏轴 ~1.0 有边缘约束 caveat，不过度解读）。中心 W_shape 跨 seed 可重复（B1a）：bare 实测相似度 0.85 ≫ 随机 0.24（7 成功 seed）、n17.6 0.90 ≫ 0.25（9 个），两边 PASS。artifacts：`results/topic4_sef_hfo/m3_local_w/mini_w_event/{figures/kmin_by_source.png, center_wshape_repro.png, kmin_by_source.json}`（results gitignored）。
   - **B1 现状**：B1a（W_shape 可重复）中心两底物都过；B1b（沿轴）/B1c（胜过 distance/rate）/B1d（核改阈不改形状的 matched-shape）**仍待做**（off-axis review 后步骤）。**未进 μ 相图。**
   - **每跑事件诊断图（用户 2026-06-24 要求）**：`src/sef_hfo_event_figure.py`（4 TDD）+ 接入 runner（默认每跑出 `figures/event_diagnostic.png`：raster 按离 kick 距离排序看外扩 + 早期逐格差分热图标源格/kick点 + 回静轨迹 kick vs sham；`--no-event-figure` 关、`--event-figure-only` 非破坏式补图）。commit a36ec30（+ DUR_KICK scope 修）。
   - 进 μ 相图前仍需 B1b/c/d + 合成结论过真实 masked lagPat/rank/KMeans pipeline；横轴届时叫 Λ_event/R_event。

---

**一句话（经 ceiling 更新）**：当前分流不是 A 也不是 C。**B0 成立；B2 的"核把同一传播场推近招募边界"在概率层已被统计支持**
——同一 kick 下核出 EA-local 事件的几率 ≈4.5×（OR=4.49 [2.89,6.97]，seed-block 置换 p=7e-4），逐 seed 触发阈值小幅一致前移（7 早 / 5 同 / 0 晚、中位 ≈0.1 kick），全程不失控，bare 在更高 kick 出同款事件。
（⚠️ "1.1 vs 1.6"是 cohort 概率曲线 0.7-lens 的放大，**不是 seed-level 阈值降 0.5**。）**但 B2 还差一条腿：W_shape 不变只从 phenotype 推断、未在传播主轴层面验证（=B1，靠 mini-W_event）。**
所以下一步是 mini-W_event pilot（验证 W_shape 一致 + 出 K_min/W_shape/P_escape 三件不混的图），**不是 μ 相图**；
w18.0 经更正是偶发自发点火（1/8）、非恒定自燃。
