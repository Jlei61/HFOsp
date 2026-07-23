# Z/M ictal-carrier gate — archive (2026-07-24)

Branch `codex/topic4-m4-snn-native-exit`. Pre-registered design:
`docs/superpowers/specs/2026-07-24-topic4-zm-ictal-carrier-gate-design.md`.
This is the **Z/M-only** archive. The old q_I+S_G+p/H sandbox is quarantined in §7 as ARCHIVE-ONLY.

---

## 0. 朴素话摘要（测了什么 / 怎么测 / 揭示了什么）

**测了什么.** 我们有一张 2 万个兴奋细胞 + 8 千个抑制细胞的二维皮层片子（各向异性连接，来自病人
E1146 的电极几何）。每个兴奋细胞带两个"慢变量"：一个记录"抑制还剩多少劲"（z，反复放电会把它耗掉→
去抑制），一个记录"自己放电后有多累"（m，适应）。片子会自己冒出零星的"发作间期"放电事件；随着 z 被耗
掉，事件越来越密，最后冲进失控高频（runaway）。我们再加一个"全局分裂式抑制池"S_G，想把失控刹住，看它
会不会变成一段真正的"发作"——而且这段发作在虚拟电极（SEEG）上应该是**持续的高频能量**，而不是一串彼此
分开的尖峰。

**怎么测.** 三条臂，都在同一张片子、同一个随机种子（seed 1）上：只有 z/m（bare）、z/m + S_G（sg）、
z/m + S_G + 记忆刹车 H（sgh）。我们把每个细胞的电流按病人真实电极位置加权，读出 15 路虚拟 SEEG（就是
论文里 |I_E|+|I_I| 的 LFP 代理，10 kHz 采样）。然后问一个很具体的问题：核心区放电率在微爆发之间是**掉回
静息**还是**一直保持在高位**；电极上的 30–80 Hz 能量是**持续抬高**还是**一阵一阵回到基线**。判据在跑之前
就锁死（见 spec 的两层门 A 源空间 / B 电极观测）。

**揭示了什么（截至本文，Phase 0 完成，Phase 1 判决待填）.** 目前只有 seed 1 的旧诊断跑（还没上正式高频读
出）能说的是：sg 把失控刹成了一段**长时间反复、逐渐增强的核心爆发串**——全片平均放电率只有 ~4.79 Hz（像
发作间期水平），但核心峰值到 403.6 Hz，核心平均只有 30.8 Hz（即核心大部分时间是安静的、~7% 占空的爆发）。
**这既不是终止，也不能仅凭"全场平均低"就叫发作间期。** 它到底是"持续发作载体"还是"一串 HFO 样爆发"，
必须用锁死的两层门在正式高频读出上判，不能拍脑袋。（内部代号：Z/M lockpoint
`zA_q75_tz5000__mA0p001_tau500`，S_G α_G=16，arm `sg`。）

---

## 1. 底座（当前正确 substrate）

- E1146 `twoend_equal`，`PP.build_substrate(seed)`，L=20 mm，N=40000（NE=32000，NI=8000）。
- 慢变量：`use_z=True, use_m=True, use_qI=False, use_gK=False`（q_I 冻结在 1 → `z·q_I·I_I == z·I_I`，与
  canonical `mz_slow_vars.py` 逐位一致，`tests/test_zm_slow_field_parity.py`）。
- lockpoint `zA_q75_tz5000__mA0p001_tau500`：τ_z=5000, τ_adp=500, η_m=0.001, I_th_EI=q75(slow-off 间期
  E-cell I_I)=1.280（in-run 标定）。
- **不回 q_I / g_K。E→E 拓扑不动。** 本线独立问题：不改 EE 的前提下，抑制侧的空间反馈能否先形成一段持续的
  ictal carrier，再由慢变量退出。

## 2. seed-1 三臂实测（25 ms bin，来自 `results/topic4_sef_hfo/zm_snn_native_exit/*_seed1.npz`）

| arm | 时长 | 分类(旧 termination cls) | all-E mean | all-E peak | core mean | core peak | z_core终 | S_G max | H max |
|---|---|---|---|---|---|---|---|---|---|
| bare | 2.9 s (runaway 截断@2871.8ms) | runaway | 44.8 Hz | 270.6 | 113.8 | 431.2 | 0.702 | – | – |
| sg | 15.0 s | fragment | **4.79 Hz** | 86.6 | 30.8 | **403.6** | 0.343 | 0.159 | – |
| sgh | 25.0 s | fragment | **2.17 Hz** | 39.8 | 18.1 | 319.7 | 0.372 | 0.102 | **0.0349** |

**共性数字纠正（本文锁定口径，纠正早期 memory 混淆）：**
- **SG 是 15 s，不是 25 s**；25 s 是 SGH。
- **SG all-E mean 是 ~4.79 Hz**（不是 ~2.2 Hz）；~2.17 Hz 是 SGH。
- 全片低均值（4.79 / 2.17 Hz）**不能掩盖核心 100–400 Hz 的爆发**：core peak 403.6（sg）/ 319.7（sgh）。
  "全场平均像间期水平" ≠ "这是间期状态"。
- bare 的 runaway 由引擎 Hz 阈值触发，写进 `runaway_early_stop_ms=2871.8`（**不是** `runaway_ms`——后者
  在引擎里根本不存在；harness 读对了字段）。

## 3. 概念纠正（跑正式门之前先把话说清楚）

1. **sensor amplitude ≠ actuator gain ≠ effective load.** H 有三个不同的量：H 感到的输入 `phi_drive`
   （sensor 读数）、耦合强度 `alpha_H`（actuator gain）、真正进膜的负载 `alpha_H · H`（effective load）。
   sgh 的 `H_max=0.0349`、`alpha_H=16` → effective load ≤ 0.56。**"H_max 小" 不等于 "H 无效"**——要看
   effective load 相对分母 `1+alpha_G·S_G+alpha_H·H` 的占比，而不是 H 的绝对值。
2. **不是每个 burst 都叫 IED.** sg 的核心爆发是"S_G 反复把核心同步 reset → 再点火"的产物，是一段候选的
   ictal 内循环（candidate clonic-like inner cycle），不是发作间期离散事件。措辞用 "persistent focal
   recurrent burst train" / "candidate inner cycle"，不用 "IED train"。
3. **不声称旧 q_I substrate 证明了"不存在稳定间期吸引子".** 那是 q_I 沙盒的结论，与当前 Z/M 无关（§7）。
4. **H 的真实输入是 phi_drive，不是 p_max.** 旧诊断图把 `p_max`（p 场的空间最大值标量）标成 "H sensor in"
   是错的。真实驱动是 `phi_drive = mean(Phi(p) over cells where phi > 0.2·pmax)`（active-focus 均值）。
   Phase 0.2 已把 `phi_drive` 与 active-focus 占比正式 trace（`slow_field.py`，off-by-default byte-parity，
   `tests/test_zm_hdrive_diagnostics.py`）。**本线 Phase 1 H 冻结关闭**——H 的建立/终止是 Phase 2 的问题，
   不在此处下"H 建不起来"的结论。

## 4. 两个分类器分离（task §5.4）

旧 M4-2 termination 分类器的 `fragment` 标签描述"活动曲线形状"，**不能**承担"是否存在 ictal carrier"的判
定。新增两个互相独立、词表不重叠的 verdict（`src/topic4_zm_carrier_verdict.py`，合成 fixture 测试
`tests/test_topic4_zm_carrier_verdict.py`）：

- `ictal_carrier_verdict` ∈ {`fail_runaway`, `fail_plateau`, `fail_hfo_like_train`,
  `candidate_source_only`, `candidate_observed_carrier`}。
- `lifecycle_verdict` ∈ {`carrier_not_established`, `no_onset`, `prevention`, `persistent`,
  `terminate_to_silence`, `terminate_then_reignite`, `terminate_and_recover`}——**只有 carrier 通过才允许
  输出 lifecycle candidate**；否则返回 `carrier_not_established` 哨兵。

## 5. Phase 0 工程加固（commit 1）

- **H-drive 观测**：`slow_field.py` 新增 `trace_phi_drive`（真实 H 输入）、`trace_active_frac`（active-focus
  网格占比）、`trace_m_core_mean/surround`；均为观测，spike 输出不变（BASELINE_SHA=`da5fc18c27d5340a` 不动，
  Z/M parity 不破）。
- **verdict 分离** + **pre-registered carrier gate spec**（阈值锁死，跑前冻结）。

## 6. 下一步：Phase 1 carrier 门（正式高频读出）

复用引擎自带 `lfp_recorder=` 钩子（`kick_probe.py:291`，观测-only，不改动力学）+ E1146 15 触点 montage
`S["reg"]["montage_sheet"]`，把 LFP（10 kHz 采样、存 2 kHz、Nyquist 1 kHz > 150 Hz）读回。跑 `bare` / `sg` /
`interictal_ctrl`（H 全关，seed 1 先判），用锁死的两层门判 `ictal_carrier_verdict`。判决填入 §8。

## 7. 🔴 LEGACY q_I + S_G + p/H 沙盒 —— WRONG-SUBSTRATE / ARCHIVE-ONLY（不代表当前 Z/M 模型）

> 以下全部来自 **旧的 field-based q_I + S_G + p/H 模型**（`use_qI=True`），**不是** 当前锁定的 per-neuron
> Z/M。所有科学结论（"H 是干净终止器"、"没有可恢复的间期吸引子"、frozen atlas、t_form=1600、onset=2300）
> **只在 q_I 沙盒内成立**，不得当作 Z/M 结果读。细节见
> `docs/archive/topic4/sef_hfo/m4_snn_native_exit_execution_2026-07-21.md`（§1–§11）与
> memory `project_topic4_m4_snn_native_exit_2026-07-21`。可复用的只有：工程/方法（crash-safe resume、
> provenance、active-focus H sensor），以及 H 作为一个**候选**机制留待在 Z/M 上重测。

## 8. 判决（seed-1，2026-07-24）—— NO-GO：是 HFO 样爆发串，不是持续 carrier

**一句话**：在当前原始各向异性 Z/M SNN 上，S_G 把失控刹成的发作态，在虚拟 SEEG 上是**一串彼此分开、之间
回到基线的 HFO 样爆发**，**不是**一个持续增强的 ictal high-frequency-energy carrier → 走 Path B（先改抑制侧
的空间反馈造 carrier），**不**走 Path A（没有 carrier 就没有 exit 可谈），并**停止**一切 H/α_H/τ_H/burst-count
扫描。

**测了什么.** sg 臂（z/m + 全局分裂抑制池 S_G，15 s，seed 1）产生的核心反复爆发，到底是"发作载体"（电极上
高频能量一直抬高）还是"一串分开的 HFO 尖峰"（能量一阵一阵回到基线）。

**怎么测.** 跑前锁死的两层门（spec §3/§4），在 10 kHz 采样、存 2 kHz（Nyquist 1 kHz > 150 Hz）的虚拟 SEEG 上：
门 A 看源空间核心放电率的最长 macroepisode 是否 ≥2 s 且 occupancy ≥80%（爆发之间不掉回基线）；门 B 看是否
≥2 个触点的 30–80 Hz 能量包络持续抬高（occupancy ≥80%、gap ≤250 ms）。

**揭示了什么（seed-1）.**
- **门 A 失败（`source_not_sustained`）**：核心峰值 455 Hz，但最优 2 s 窗只有 **occupancy 0.17**、最长
  macroepisode 1730 ms（< 2000）——爆发之间核心掉回基线，~83% 时间是安静的。全体均值 4.79 Hz、tail 10.29 Hz。
  （A7 sep=3、recruit=True、flash=False、sat=False：爆发是**局部+招募+区别于间期事件**的，唯独**不持续**。）
- **门 B 失败（0 个持续触点）**：汇侧触点 ICL8–11 的 30–80 Hz 有很强的凸起（峰值 dB 14–32），比源空间"更接近
  持续"（最优触点 occupancy 0.55），但仍 < 0.80 → 没有触点通过 B1。**注意**：电极因为把 ~19 个邻近细胞的
  |电流| 加权平均，包络比脉冲率平滑，所以显得没那么间歇——但两层门一致判"间歇"。
- **Section-8 慢-快（`transient_burst_train`）**：47 次爆发，IBI≈300 ms（cv_tail 0.13＝周期基本平稳），但幅度随
  z_core 从 1.0 耗竭到 0.34 而**持续escalate**（drift 0.55）；S_G 在每次爆发后 ~65 ms 才涌起、随后塌陷＝松弛泵。
  → 这是**瞬态反复爆发串 / 候选 inner cycle**，**不是** limit cycle（无 frozen-slow 重复轨迹）。

**当前能写**：sg = "S_G 有界化的持续局部反复爆发串（escalating transient burst train）"；机制＝单一**全局标量**
S_G 每次把整个核心**同步 reset**（burst→S_G 涌起→whole-core reset→S_G 塌陷→再点火），天然生成窄爆发串；瓶颈
是**缺内在 ictal carrier**，不是缺 exit actuator。

**当前不能写**：ictal attractor / limit cycle / lifecycle 完成；"持续 ictal carrier"；"高核心率＝高频 SEEG 能量"；
"H 建不起来"（H 本阶段关闭，不测）；旧 q_I 关于间期吸引子的任何结论。仍是 seed-1 pilot。

**下一步为什么是 Path B 而不是 exit.** stop condition #4：carrier 门失败后不许再调 H。缺的是**内在载体**，不是
退出器。按 §7 Path B：把单一全局 S_G 拆成 **patchwise 局部抑制反馈 + 较弱全局**，先做 2–8 patch 的 cheap-first
rate 屏幕，看快子系统能否从"全同步松弛振荡"变成"异步 microdomain / 有界空间波 / 持续高频群体载体"；只有 cheap
proxy 通过才移植回全 SNN，且全 SNN 仍须过上面的 A+B 门。**不**以"更强 M"为首选，burst-count H 暂缓。

图：`results/topic4_sef_hfo/zm_ictal_carrier_gate/figures/`（README 逐图说明）。
manifest：`carrier_gate_seed1.json`（每臂 provenance + verdict）。engine SHA `8ef5b60`，readout 2 kHz/Nyquist 1 kHz。

## 9. Path B —— spatial inhibitory carrier 设计 + cheap-first screen（2026-07-24）

**设计（锁定）.** 不改 E→E 拓扑、不动 relay/conductance；只在**抑制侧**把 §8 诊断出的"单一全局标量 S_G 每次
同步 reset 整个核心→窄爆发串"这个机制拆掉：把一个全局池换成**空间分辨的抑制反馈**——patchwise 局部池
（每个微区各自的 S_i）、可选**较弱的全局成分**、可选**空间平滑/低秩共享**。核心假设：局部化后不同微区不再同步
归零→相位错开的 microdomain 活动叠加→群体（电极）高频能量包络持续，而非一起塌到基线。

**cheap-first screen（reduced K-patch rate 模型，`src/topic4_zm_patch_screen.py`，非 SNN）.** 每个 patch =
慢分裂抑制池门控的快放电率松弛振子（自兴奋 w_rec 给快双稳→松弛振荡；τ_s=80 匹配 S_G）。比较四种抑制结构
（K=16、异质 σ_I=0.4、弱耦合 w_c=0.05、4 seeds、OFF 态为绝对基线的群体 occupancy + 跨 patch 同步度 + 是否仍
振荡）：

| 结构 | pop occupancy | synchrony | 仍振荡 | carrier proxy |
|---|---|---|---|---|
| global scalar（homogeneous＝SNN 复现） | **0.52** | +1.00 | 是 | **0/4**（同步爆发串，P 塌到 OFF） |
| global scalar（heterogeneous） | 1.00 | +1.00 | **否**（osc≈0） | 0/4（死不动点，非载体） |
| **patchwise 独立局部池** | **1.00** | **+0.04** | 是 | **4/4** |
| patchwise + 空间平滑（σ=1） | 1.00 | −0.06 | 是 | 4/4 |
| local + weak global（ε=0.2） | 0.96 | +0.09 | 是 | 4/4 |

**揭示了什么.** 便宜模型里**方向性 GO**：把全局标量池换成 patchwise 局部池，把"同步爆发串（P 在爆发间塌到
OFF、occ 0.52）"变成"去同步、仍振荡、群体不再塌到 OFF 的持续载体（occ≈1.0、sync +1.0→+0.04）"。global 要么
同步爆发串、要么（加异质）退成死不动点；local+weak-global 也过（弱全局不重新同步）。K-scan（K=2..64）patchwise
方向稳健。

**能写**：在 reduced rate 模型里，**patchwise（空间分辨）抑制是把 global-S_G burst train 转成持续群体载体的正确
杠杆**；下一步＝把 patchwise 抑制移植回原始各向异性 SNN（**仅抑制侧、不动 E→E**）。**不能写**：这不是 SNN 结果、
不是 LFP 上的 carrier（rate 代理不含带内高频结构，群体仍有 ~1.3× 调制、非平滑纯载体）；移植后**仍须**过预注册的
A+B carrier 门才算数。**不**用"更强 M"、burst-count H 暂缓（stop condition #4）。

图 `results/topic4_sef_hfo/zm_patch_screen/figures/patch_screen.png`（+README）；summary
`patch_screen_summary.json`；测试 `tests/test_topic4_zm_patch_screen.py`（8 green，锁定振荡带 + global>patchwise
同步度 + patchwise 过载体代理）。
