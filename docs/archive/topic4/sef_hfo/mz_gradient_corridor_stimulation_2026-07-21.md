# MZ 病理轴走廊刺激位点比较 — 队列级虚拟 SEEG 仿真

**日期**: 2026-07-21 起（跨 07-22）· **分支**: `codex/topic4-mz-stim-site-frame` · **Tier**: 模型侧机制探索（NOT 临床 / NOT 真实发作 / NOT DBS 疗效）

---

## 摘要（第一性原理朴素话）

**测了什么。** 我们拿一个会"发作"的脉冲神经网络（40000 个神经元的二维薄片），在里面放两个"病灶核"（低阈值、容易点火的两团细胞），核之间是一条"走廊"。病灶核和走廊的位置不是随便定的，而是从每个病人**间期** HFO 传播梯度轴（早→晚的方向场）算出来、映射进这张薄片。然后我们在走廊上不同位置放一对"虚拟电极"，对电极附近的兴奋性细胞做**虚拟抑制**（把它们的点火阈值抬高一段时间，等于临时压住这一小片），比较三个位置：轴的负端、轴的正端、走廊中段。问：**压住走廊中段，是不是比压住两端更能（1）推迟全网失控、（2）减少"跨过中段、烧到另一端"的全局扩散、（3）还保留一些局部小事件、（4）但又不是把所有活动都灭光**？只有四条同时成立，才叫"选择性走廊切断"。

**怎么测的。** 每个病人 × 每个随机种子先跑一遍**不刺激**的基线，看它什么时候失控（连续 100ms、每神经元 ≥120Hz 记为 operational runaway），据此**盲选**刺激窗口（失控时间的 45%–75%），并要求刺激前至少有 3 个"点起来又自己平息"的间期事件。然后对每个刺激位置各跑一遍，除刺激位置外**一切相同**（同网络、同噪声、同刺激前轨迹、同抑制强度、同目标细胞数）。主指标是固定 20 秒内的"受限无失控时间"RRT = min(真实失控时间, 20s) − 刺激关闭时刻；主对比 C_run = RRT(中段) − ½[RRT(负端)+RRT(正端)]。统计单位是**病人**（先在种子内取配对效应、再取病人中位数、再做精确符号翻转检验），只用严格可逆梯度轴的 primary 队列。

**揭示了什么。** 见下文"结果"。上限只能说到："在这个由病人电极布局映射出来的 Z+M 脉冲网络里，走廊中段的虚拟抑制**改变**了模型的 operational runaway 时间和模型传播范围。"不能说成临床疗效、真实发作预防或完整癫痫机制。

（内部代号：Z+M 慢变量候选 `zA_q75_tz5000__mA0p001_tau500`；间期梯度场 `topic5_interictal_template_fields_v1` / `template_propagation_axis_v2`；SNN Stage-5 blessed 底物；operational runaway 120Hz/100ms。）

---

## 1. 科学问题与可证伪假设

在具有**稳定反向**（`relation=reversed`）间期 template-gradient 轴的病人布局中，对病理轴走廊**中段**双极电极对实施虚拟抑制，相比刺激轴**任一端**：
(H1) 更晚达到 operational runaway；(H2) 更少跨越中段抵达另一端的全局传播；(H3) 事件更多局限在局部；(H4) 但不是把所有局部事件消灭。四条同时成立 → 支持"selective corridor disruption"。仅 H1 成立而传播范围不变 → 只能说 timing/site effect。刺激后所有事件消失 → 只能说 global suppression。

## 2. 模型硬合同（冻结，全程不逐病人调参）

- 引擎：`run_loop`（`src/topic4_mz_onset_dynamics.py`）`store_spikes=False` + 轻量 streaming spatial observer + 复用 `LFPRecorder.sample(I_E,I_I)`（电流型虚拟 LFP，每 10 步采样）。**从不分配 T×NE raster**（NE=32000，一次 25s raster 会是 8GB）。
- 慢变量：仅兴奋性神经元的抑制效能 z + 适应 m；冻结候选 `zA_q75_tz5000__mA0p001_tau500`（use_z=use_m=True, I_th_EI=95.1985, tau_z=5000ms, tau_adp=500ms, eta_m=0.0074516）。
- 刺激 = `MZOnsetProbe.set_suppression`（阈值抬高 = 虚拟抑制），delta=+50mV，作用于双极对中点半径 1.5mm 内、各位点**共同支持的固定 N_target 个最近 E 细胞**。
- 失控判据：`score_runaway`（20ms EMA 每神经元率 ≥120Hz 持续 ≥100ms）。
- **未修改任何 guarded engine 文件**（6 个：kick_probe/params/model/connectivity/connectivity_rot/lfp）；`run_observed_loop` 是 `run_loop` 的忠实拷贝 + 只读 observer/LFP 钩子，并有**逐位相等 parity 门**（rate_E/rate_I/慢变量迹与 `run_loop` bit-identical，streamed active-fraction 与 raster active-fraction 完全相等）。

## 3. 几何硬合同（轴只来自间期梯度，绝不来自 endpoint）

- 只读冻结输入 `results/interictal_propagation_masked/template_gradient_fields/per_subject/<subject>.json`（主 checkout，只读，显式 `--input-root`）。
- 只用 `axis_pair.shared_axis.u`、`interictal_field.planes.shared.points`、`interictal_field.contact_order`（+ `shafts`）。**禁用** source/sink centroid、rank-displacement、decision_k、swap endpoints、`template_source_foci`、`register_to_sheet`、fixed top-k、D_AB、任何 ictal 量。模块**不 import** `sef_hfo_subject_placement`（有 AST 测试守卫）。
- 每个记录加载后重算 `interictal_field_fingerprint` 并与存档比对，**fingerprint/contract/shared-plane 任一不符即 fail closed**（不回退 endpoint）。
- Sheet 映射：对 `planes.shared.points` 做**各向同性单一缩放 + 平移**（along→sheet-x，走廊水平，theta_EE=0，全病人一致），把 E→E 各向异性对齐走廊。两病理核放在 along 分布的 **Q10/Q90**、transverse=0。全病人同一冻结 SNN（同 seed、同 net、同核阈值幅度），**只**改核位置和电极/刺激/LFP 几何。
  - 参照：blessed E1146 endpoint 底物 inter-core=13.05mm@θ≈0；本梯度映射 E1146 走廊=11.77mm@θ=0 —— 几何几乎一致，说明映射保留了候选被 bless 的动力学 regime。

## 4. 队列与几何审计（`geometry_audit.csv` / `cohort_manifest.json`）

primary 进入门（全部必须）：`axis_pair_estimable` & `geometry_2d_supported` & `strict_stability_pass` & `relation=reversed` & `shared_axis.status=ok` & `interictal_field.status=ok` & fingerprint 通过 & 至少 3 个**互不重叠**的同杆相邻双极位点 & 两核不重叠（core_sep > 2·core_r=3.0mm）。

**入选 primary（n=4）**: epilepsiae_1146（走廊 11.77mm）、epilepsiae_590（6.55mm）、epilepsiae_958（4.60mm）、yuquan_zhaochenxi（6.65mm）。

**排除（如实记录）**:
- epilepsiae_1084 —— 两核重叠（core_sep 2.96mm ≤ 3.0mm，各向同性 bbox 拟合被横向电极扇面压缩）→ 走廊退化。
- epilepsiae_583 —— 7 触点里只有 1 对同杆相邻双极（其余单触点/非相邻杆）→ 无法组 3 位点。
- epilepsiae_384（sensitivity，non_strict）—— 3 对相邻双极全在一根杆上 → 无法组 3 个互不重叠位点。
- epilepsiae_139 / yuquan_zhangjiaqi —— geometry_unsupported（单杆一维），不进二维刺激主分析。

n=4 ≥ 4 → 队列 GO（几何层面）。双极位点选择：负/正端 = along 最低/最高的相邻对；中段 = 到走廊中心点 (along-center, transverse=0) **二维**最近的相邻对（保证在轴上而非仅 along 居中）；off-axis = along 近中心、|transverse| 最大的合法对。⚠️注意：部分病人电极不在走廊中心取样，中段是"可得最近"而非理想中点（590 中段 HL5-HL6 偏向负端）——配对符号检验仍有效，但个别病人中段-端点对比偏弱，报告时注明。

## 5. 仿真臂与统计

每 subject×seed 至少 4 臂：`baseline_no_stim` + `gradient_endpoint_{negative,positive}` + `gradient_middle`（资源允许再加 `gradient_offaxis_control`）。seeds 1/3/4。T_max=20s。刺激窗口 arm-blind = [0.45, 0.75]×基线失控时间；全臂共用；要求刺激前 ≥3 个可恢复间期事件。基线不合格条件（不逐病人调参）：T_max 内无失控 / 失控前 <3 事件 / 一开始就高放电 / 无稳定间歇事件。

主指标 RRT=min(t_run,T_max)−stim_off；保存真实 t_run + 右删失指示（不把删失说成"发作"）。主对比 C_run=RRT_mid−½(RRT_neg+RRT_pos)；敏感对比 C_best=RRT_mid−max(RRT_neg,RRT_pos)。统计单位=病人：seed 内配对→病人中位数→队列。**报告口径见 §6：中位数=描述性效应；`exact_sign_flip` 是对*均值*统计量的精确符号翻转随机化检验（不是中位数），另附 Wilcoxon；主口径 = balanced complete-case（4 病人都合格的 seeds）+ seed 分层，混合全 seed 只作 secondary。**

跨走廊传播指标：**首选 post-stim、pre-runaway 窗 [stim_off, t_run) 的 far-reach probability / axial span**（`prerunaway_propagation`，从 streaming 轴向活动离线算）；whole-run 的 escape/far/normalized-span/local-preservation 被终末全场 runaway 支配、不区分位点，仅 per-run 保留、不作承重（见 §6）。total-activity ratio 判"是否全灭"。

**⚠️ n=4 的精确检验功耗上限**：4 个病人的精确符号翻转（对均值）两侧 p 最小 = 2/16 = 0.125（单侧 1/16=0.0625）。因此 cohort 层面拿不到 p<0.05；主报告口径是"4 个病人里几个方向一致 + 逐病人值 + 精确 p + 个体异质"（小队列描述性 tier），不是强功耗显著性。

**事件条校准（承重口径）**：冻结候选在本几何里产生的是**缓慢 z-耗竭爬坡**——间期离散可恢复事件从 t=0 起就存在（率迹清晰见到 40–70Hz 爆发后回到 0），但事件幅度随 z 耗竭单调增大（early active-fraction 峰 ~0.05 → late ~0.14），最终在 late 段越过 120Hz runaway。若事件条用**整段** af.max（被 late 爬坡抬到 ~0.14）→ 条 ~0.07 → 漏掉早期小事件（误判"0 个 pre-stim 事件"）。**正确做法**：事件条 = floor + 0.5×(早期 4000ms 窗口 af.max − floor)（z≈未耗竭 ≈ blessed slow-off 间期尺度），条 ~0.022–0.033，逐病人 pre-stim 可恢复事件 6–18 个 → 全部合格。这是对 blessed `slowoff_event_bar` 的近似（省一次全长 slow-off 跑）；未来敏感性可用真 slow-off 复核。
- ⚠️ E1146 走廊最长（11.77mm）→ runaway 最晚（≈19.8s≈T_max）→ RRT headroom 很小（~157ms），近乎删失，逐病人贡献接近 null（非偏倚，只是降功耗）；590/958/zhaochenxi（runaway 9.6–12.2s）headroom 充足。

## 6. 结果

**GO/NO-GO 一句话**：工程与几何 GO、动力学合格 GO，但**主假设阴性**——中段虚拟抑制**不**一致地比端点抑制更能推迟 runaway；真正稳定的结论是**病人间异质**（唯一稳健的中段优势只出现在走廊最短的 E958）。

**测了什么 / 怎么测的**：见 §1–§5。4 例 primary（E1146/E590/E958/yuquan_zhaochenxi）合格（修正事件条后 pre-stim 可恢复间期事件 13–23 个）。完成 = seed 1/3 各 4 病人×5 臂 + seed 4 三病人×5 臂（**E1146 seed-4 baseline 在 T_max 内不失控 → 无刺激窗/臂 → ineligible**）= 56 runs 全绿；~141min、8 workers、无 OOM。

**统计口径（承重，避免误读）**：单位=病人；先 seed 内配对、再对每病人取跨 seed 中位数、再看队列。**中位数=描述性效应；`exact_sign_flip` 检验是对*均值*统计量的精确符号翻转随机化检验（不是对中位数），另附 Wilcoxon signed-rank**。n=4 精确检验两侧功耗天花板 p≥0.125（拿不到 <0.05）。**两套分析集并列报告、不事后钦点 primary**：(a) `all_available` = 预先指定的 seeds 1/3/4（每病人对其合格 seed 取中位数）；(b) `complete_case` = 4 病人都合格的 seed 子集（=seeds 1,3）的 complete-case 敏感性。两者都是"每病人一个中位数"，complete_case **不是加权更重**——它只是丢掉 seed 4（E1146 在 seed 4 无 runaway/臂，而其余 3 人恰在 seed 4 最正），所以队列中位数更负；**两者方向相同**。

**揭示了什么**：
- **两套分析集并列（都阴性、方向一致）**：`all_available`（seeds 1/3/4）C_run 中位数 **−293ms**（均值 −131，**+2/4**，sign-flip-on-mean p=1.000，Wilcoxon 1.000）、C_best −949ms（+1/4，p=0.375）；`complete_case`（seeds 1,3）C_run 中位数 **−613ms**（均值 −306，+2/4，p=1.000）、C_best −1176ms（+1/4，p=0.375）。→ 结论对分析集**稳健**，都是**阴性**（中段中位数上更差、3/4 病人相对最优端点更差）。
- **seed 分层**：seed 1 C_run 中位数 −375（+2/4）、seed 3 −480（+2/4）、seed 4 **+2636（+3/3，但 E1146 缺席 → 不是完整队列反转**，只说明效应有明显 seed 依赖）。
- **逐病人（真正稳定的结果=个体异质，非统一中段优/劣）**：
  - **E958（最短走廊 4.6mm）：3/3 seed 中段 > 平均端点*且* > 最优端点**（balanced C_run +2051、C_best +1778）——**唯一稳健的中段优势**。
  - E590（6.55mm）：3/3 seed 中段 > 平均端点，但相对最优端点几乎持平（balanced C_best −392）。
  - E1146（最长 11.77mm）：两个有效 seed 都偏端点（balanced C_run −2048、C_best −2192）；**seed-3 在 10.3s 失控、有充分 headroom 仍偏端点** → 非 T_max 删失伪迹，与"runaway 由两个低阈值 core 驱动、压 core 的端点比切走廊中段更直接"一致。
  - yuquan_zhaochenxi（6.65mm）：2 seed 偏端点、1 seed 偏中段（signs −1/−1/+1）。
  - → 现在支持的是**个体异质性**，不支持统一的中段优势或劣势。**"短走廊→中段、长走廊→端点"只能列为下一步假设**——n=4，且走廊长度与 core 距离、刺激覆盖、删失共同混杂，不能形成长度依赖结论。
- **跨走廊传播（非承重，已修窗口耦合）**：旧 whole-run escape/far/span 被终末全场 runaway 支配（各臂几乎相同、local 全 0），弃用。**关键坑（审阅指出）**：若每臂各用自己的 [stim_off, t_run)，推迟 runaway 的臂自动获得更长观察窗 → far-reach 不再独立于 RRT。改用**三臂共用的固定 1s 刺激后窗**（`far_delta_common1s` = 中段−平均端点 far-reach probability，与各臂 runaway 时间解耦）：E1146 +0.009、E590 −0.010、E958 −0.010、zhaochenxi +0.007——**全部 ~±0.01、贴近 0，且只有 E958 在多种窗口下符号稳定；E590 的"中段限扩散"不具窗口稳健性**。→ **C 面板已从主图删除**（不作承重结果）。**安全表述：在共同刺激后窗下没有观察到窗口稳健的跨走廊传播位点差异——不能写成"中段刺激不降低跨走廊传播"这种更强主张。**
- **是否只是 global suppression**：否。total-activity ratio 0.73–1.20（4/4），任何位点都没把活动全灭；t_run（RRT）阴性与事件条无关（RRT 基于率 120Hz、不用事件条），相对可信。
- **事件条敏感性（因是看过结果后校准）**：事件条只影响合格判定与事件类指标，不影响 RRT。离线在 (early-window 2000–6000ms × CAL_FRAC 0.3–0.7) 网格上重算（`event_bar_sensitivity.json`）：**window≤5000ms 且 CAL_FRAC≤0.5 时 11/11 subject-seed 全合格**（每例 pre-stim 可恢复事件 ≥6），只有逼近旧"整段 max"伪迹的极端组合（6000ms+0.7）才掉到 7–10/11 → **4/4 admission 对事件条选择稳健**。⚠️此检验用的是下采样 af（~3.1ms bins）近似；definitive control = 真 slow-off baseline（未做）。

**当前安全科学表述**：在这个由病人间期 template-gradient 轴映射出的 Z+M SNN 里，小尺度双极虚拟抑制会适度扰动 operational runaway 的时间，但**没有观察到统一的中段优势**——4 例里只有走廊最短的 E958 稳健地中段优于端点，最长的 E1146 稳健地端点更好，其余混合。**不支持 selective corridor disruption**；走廊长度依赖只是待检验假设；不构成任何临床、真实发作或 DBS 结论。

**最大未解决问题 + 下一步**：(1) 效应小 + n=4 功耗天花板（p≥0.125）→ 小队列描述性阴性、非强证伪；(2) 传播读出仍粗——下一步用逐事件"刺激后跨中段首次到达远端 / 远端事件概率与潜伏期"，且限制在 pre-runaway 窗；(3) 走廊长度 vs core 距离 vs 刺激覆盖 vs 删失混杂，需参数化走廊长度或更大几何队列解耦；(4) 事件条需真 slow-off baseline 复核（本轮只做了下采样近似敏感性）；(5) 剂量×位点交互（+50mV、~520 cells 相对全场 z-耗竭偏小）是下一杠杆。频率 HFS/LFS **未做**（§8）。

## 7. 工程与资源

- OMP/MKL/OPENBLAS/NUMEXPR=1，MALLOC_ARENA_MAX=2；单 launcher fork Pool（40k net 经 COW 在 worker 间共享，RSS≈7.1GiB 主要是共享连接矩阵）；atomic write + fingerprint resume；resource_log.csv 周期记录。
- 时间成本：net build ≈112s/seed；单次全长 20s 运行 ≈35s/sim-second ≈12min（早停在失控则更短）。
- 28 个合同测试全绿（fail-closed fingerprint / cohort==admitted / sign-flip label-only / no-endpoint-axis(AST) / 同杆相邻双极 / middle 不重叠 / 剂量匹配 N_target / 各向同性变换 / pre-stim parity / store_spikes=False≡raster / 删失 / resume fingerprint / 坏 artifact / 无 post-stim 事件配对）。

## 8. HFS/LFS 边界

阈值抬高 clamp 无真实双相脉冲/电场/轴突激活/突触短时抑制/charge balance，因此**不得**称 HFS/LFS，也不得据此比较临床高/低频。频率作为独立后续实验，只有在主队列+统计+QA+报告全部完成且仍剩 ≥90min 才实现，且 site/细胞数/窗口/剂量匹配。
