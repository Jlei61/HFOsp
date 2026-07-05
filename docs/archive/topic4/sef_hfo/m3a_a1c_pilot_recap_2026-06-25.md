# M3A-A1c PILOT — dynamic global feedback RESTRAINT screen (2026-06-25, PILOT-FIRST HARD STOP)

> Scope: A1c PILOT only (spec §4.1). MECHANISM-SCREEN go/no-go, **NOT** inhibitory-exhaustion validation
> (that = A2). Results `results/topic4_sef_hfo/m3a_slowvars/a1c_pilot/` (status_a1c_pilot.json + figures).
> Engine hook committed `065e54a` (bit-parity, re-blessed). **STOPPED after pilot — full grid NOT run.**

## 朴素话（测什么 / 怎么测 / 揭示什么）

**测什么**：A1b 是"静态地图"（固定强度的总体抑制）。A1c 给引擎加了**动态全局反馈抑制**：抑制不再是固定旋钮，
而是**网络自己的总放电率经低通滤波后，反过来按比例压住所有兴奋细胞**（Abbott 的真动态：放电越凶刹车越大，看全网
平均、不分位置）。核心问题：**动态反馈能不能把一个静态会失控的角（l2_g1.0，核内放电 ~410Hz）变成"点着后回落"，
同时不把另一个还在工作的活动态（l1_g1.0）一刀压成静默。**

**怎么测**：失控锚 `l2_g1.0`（强回路）+ **工作态保留对照 `l1_g1.0`（弱回路）** × 反馈增益 `{0,8,16,32}` ×
τ{150（事件内）,2000（事件间）} × 3 seeds。
> **`l1_g1.0` 的正确语义（P1-1 修正）**：它**不是**"已经能干净回落的发作样态"——在 gain=0 基线、用绝对回静比量，它
> 自己也偏高（110–338×、三 seed 全失控），只是比强回路锚（570–1101×）弱一档。它是**来自 A1b 的、较弱的工作/活动态**；
> A1c 对它只问一件事：**动态刹车会不会把它从"有活动"直接过压成"静默"**。（早期 A1b 给它贴的"能回落"标签是用 per-event
> 相对 return 判的，那会把钳在高位的平台误判成"回落"——这正是 A1c 改用绝对判据的原因。）

判据用**绝对** tail-to-baseline 比（末 500ms 率 / 基线率 ≤ 1.5 才算"回到基线"，不是相对事件峰值的 return，因为那会把
"钳在高位的平台"误判成"终止"）；且终止必须 **(a) 先点着**（不是一开始就被压死）**(b) I_global 在率下降之前先升**
（动态，不是静态恒定刹车）。**联合判据现已机器化**（`status_a1c_pilot.json::joint_window_by_gain_tau`，P1-2）：每个
(gain,tau) 逐腿记录"锚是否全 seed 终止 / 保留对照是否未被压静默 / I_global 是否领先下降"，"没有联合窗口"由机器导出、
不再靠人工读表——结果 `joint_window_exists = false`。

**揭示什么（诚实，screen 级）**：
1. **均匀动态全局反馈 CANNOT 干净地终止这个核集中型失控。** 强回路锚 `l2_g1.0`：增益 8 → 三 seed 仍 runaway
   （绝对回静比 154–302）；增益 16 → 仍 runaway（116–226）；增益 32 → 终止 2/3（回静比 0.22 / 0.62），第 3 个 seed
   **从没点着**（silent_suppressed）。
2. **磁量级三个量分开讲，且**只在同一个 run 内比**（P1-4，不能用一个范围同时解释 peak 和 ratio，也不能跨 run 配对）**：
   - `I_global_peak`（动态刹车实际峰值电流，强回路锚 l2_g1.0 t150 **三 seed 中位**）：**g8 ≈ 56 → g16 ≈ 79 → g32 ≈ 91**
     （g32 两个终止 seed 91 / 107，第三个 silent seed 1.2 拉低不计；峰值随增益单调升）。
   - `I_I_on_E_p95`（核**自己**产生的抑制电流，E 上 95 分位）：**失控态（g8、g16）run 里 ≈ 370–540**；run 一旦终止
     （g32），核停火，同一 run 的 p95 **塌到 ≈ 96–168**。
   - → **同一 run 内**比：在**失控的 g8/g16 run** 里，刹车峰值（~55–89）只有那个 run 自身核抑制 p95（~370–540）的
     **~0.11–0.18 倍**——刹车压不过正在燃烧的核 → 仍失控。到 **g32**，刹车升到 ~91–107，同时核 p95 随终止塌到 ~96–168，
     于是刹车才追平自己 run 的（正在塌的）核（终止的 g32_s1：刹车 107 vs 同 run 核 p95 96，比值 ~1.1）。**图景是：刹车
     把一个本就在边界附近的强回路推过去，而不是压住一个满火力的失控核。**（旧写法把 g32 的刹车峰值 ~105 和**失控 run** 的
     核 p95 ~370–540 跨 run 配成 "0.2–0.3 倍" —— 那两个数不在同一个 run，已纠正。）
   - `I_global_to_I_I_ratio`（峰值 / E 上抑制**中位数**，锚 cells ≈ **68–123**）很大，**只是因为分母是近乎静默的周边
     细胞中位数（~1）**——它**不**代表刹车压过了核。要看刹车 vs 核，比的是**核 p95**（上一条），不是全 E 中位数。
   - ⚠️ 旧 recap 把"I_global 峰值 ~90–300"写错了：那其实是**回静比尾值**（tail 116–302），不是刹车峰值；已纠正。
3. **能终止失控的那个增益（32）把工作态保留对照过压成静默**：`l1_g1.0` 在 g32 下 2/3 seeds **被压成 silent**，而能让
   `l1_g1.0` 点着-回落的增益（16）**终止不了强回路失控**（锚点仍 runaway）。**没有任何一个增益既终止强回路失控、
   又保住弱回路工作态**（机器判据 `joint_window_exists = false`；g32 那一格三条腿全断：锚未全终止 + 保留对照被压静默
   + I_global 不领先）。
4. **那点 g32 部分终止是动态 / 时序特异的（P1-3 onset-gated 对照判定，§7）**：互相关上 `I_global` 不领先率下降
   （leads-decay≈0，与率峰**同时**而非领先），但放开点火窗后施加**等幅度**刹车的对照显示——恒定刹车 + 时间打乱刹车
   （保留全部峰值幅度，只错位）**都不终止**已点火的失控，**只有**随率同步的动态反馈终止（两 seed sign-consistent）。
   所以终止**需要**刹车峰恰好压在率峰上，不是"一记静态强刹"就能复现。
5. τ=2000（事件间）对单次失控无效（仍 runaway）——符合预期，长 τ 刹不住亚秒级失控。

**结论（P1-1 命名纪律）**：**在这个衬底/量级下，动态全局反馈 CANNOT 干净终止核集中型失控**；机制上是因为失控
集中在核（失控时核区抑制电流高分位 `I_I_on_E_p95` ~500，对应核区放电 ~160–170Hz 远高于全网 p95 ~13–15Hz），而
**空间均匀**的全局反馈对所有 E 一视同仁——要压住 firing 核就得过压周边/弱态。**禁止
写**"Abbott 抑制耗竭机制成立"或"全局反馈终止发作"；只能写"均匀动态全局反馈不是核集中型失控的对的工具"。

（内部归档代号：M3A-A1c, feedback_gain/feedback_tau_ms, EMA α=1−exp(−dt/τ), tail_to_baseline_ratio,
I_global_leads_decay, l2_g1.0 anchor / l1_g1.0 preservation, silent_suppressed vs terminated,
`docs/paper/abbott_model.md` global feedback inhibition / local:global ratio）

## 状态面（figures/a1c_state_surface.png）

```
l2g1.0 (runaway anchor)         τ150: g0=run g8=run  g16=run  g32=TERM(2/3, 1 over-suppressed)
                                τ2000: g0=run        g16=run
l1g1.0 (working-state preserve) τ150: g0=run g8=run  g16=TERM g32=SILENT(over-suppressed)
```
绿格（TERM）不对齐 = 终止强回路失控所需增益（g32）≠ 保住弱回路工作态所需增益（g16）。

## §7 matched-static / time-shuffled 对照（P1-3，判定动态特异 vs 静态强刹）

**动机（P1-4 + leads-decay）**：g32 那点部分终止，`I_global` 不领先率下降 → 当前只能写"不像动态领先终止"。要把它
确定成"等价于一记静态强刹"，需要一个对照：把强回路锚 `l2_g1.0` 的两个 g32 **终止** seed（**s1、s2**），换成两种
**预设刹车**重跑同一个网络（同 seed = 同网络/同泊松抽样）：
- **matched constant brake**：恒定刹车 = 该动态 run `I_global` 的均值（幅度匹配、零时序结构）。
- **time-shuffled feedback**：把动态 `I_global` 序列时间打乱（幅度分布匹配、毁掉与放电率的因果锁定）。

**对抗审查（workflow）后三条加固**（避免把"阻止点火"误读成"终止失控"）：
- **S6 onset-gate**：动态刹车在早期自发点火窗里本来就 ~0（EMA 从 0 起）。所以对照刹车在动态刹车**首次启动前**（峰值 5%）
  一律保持 0，只在**启动后**的窗口施加 → 点火窗在三种条件下都**不被刹** → 对照 tail 低才能解读成"终止了一个**已点火**
  的失控"，而不是"一记从头就在的恒定刹车把点火**摁住了**"。
- **S7 点火门**：记录每个对照 run 的峰值放电率 + `ignited`；`terminated` = **点火了 AND** tail≤gate；没点火却 tail 低
  标记 `prevented_no_ignition`（**不算**终止）。
- **S8 公共基线**：三个 tail 都用**动态 run 的（早期未被刹的）5–50ms 基线**归一，apples-to-apples（不再各用各自被刹压低
  的基线）。另加 `control_informative = (动态自己 tail≤gate AND 点火)` 机器门。

判读：仅当 `control_informative=true`，且对照 run **点火后** tail≤1.5 → 终止**不依赖**因果锁定 → 与 leads-decay=False
一致，可写"终止不是动态特异的"。若对照**没点火**就 tail 低 → 是"阻止点火"，不构成对动态终止的反驳。

> 工程：引擎 off-by-default `fb_override_trace` 钩子（None=逐字节 parity，已 re-bless `8e773d13`，TDD T9 dynamic-path
> 回归[基线取自 pre-edit `065e54a`] + T10 zero-override parity + T11 注入正确性 + T12 mutex）；runner `--fb-control`
> （动态跑完后在同一网络上重跑 onset-gated 预设刹车，写 `fbctrl_*.json` 旁挂，含 `state`/`ignited`/`control_informative`）。

**结果（onset-gated 对照，两个 g32 终止 seed，`control_informative=true`）**：

| seed | 动态反馈 | matched-constant | time-shuffled |
|---|---|---|---|
| s1 | **terminated** (tail 0.22, peak 23Hz) | **runaway** (tail 924, 点火 peak 46) | **runaway** (tail 954, 点火 peak 54) |
| s2 | **terminated** (tail 0.62, peak 22Hz) | **runaway** (tail 573, 点火 peak 49) | **runaway** (tail 586, 点火 peak 56) |

**判读（两个 seed sign-consistent，pilot 级 n=2）**：放开点火窗后施加**等幅度**刹车——恒定刹车（= 动态刹车
post-onset 均值 ~14–17）**不终止**（仍 runaway）；时间打乱刹车（**保留**动态刹车的全部幅度、含 ~90–107 峰值，只
打乱顺序）**也不终止**（仍 runaway）。**只有随放电率同步的动态反馈终止。** 三种条件下都**点火了**（对照 peak 46–56Hz
甚至高于动态 22–23Hz，因为对照没在正确时刻刹住峰），所以这不是"阻止点火"而是"该不该把一个**已点火**的失控刹回来"。
→ **g32 那点部分终止是动态 / 时序特异的**：刹车必须把它的峰值**恰好压在放电率峰上**才终止；摊成均值、或把同样的峰
错位投放，都不行。这与 leads-decay≈0 不矛盾（刹车与率峰**同时**、并非领先，互相关 lead≈0；但时序仍是必要的）。

**⚠️ 纠正**：本 recap 早先一版（旧 from-t0 对照）写过"静态/打乱也终止 → 不是动态特异"——那是 **prevent-ignition 假象**
（恒定 ~15 从 t=0 把点火**摁住**了，对抗审查 workflow S6 命中）。onset-gate 后结论**翻转**：终止**是**动态特异的。

**对 A1c 总结论的影响（重要：headline 不变，sub-claim 翻转）**：headline"均匀动态全局反馈在这个衬底上**没有干净
操作窗口**（终止强回路锚的增益 g32 同时把弱工作态过压成静默，`joint_window_exists=false`）**不变**。但"那点 g32 终止
只是静态强刹"这个 sub-claim **作废**——它其实是**真动态终止**，问题出在**空间均匀**（要把核刹停就得过压周边），不是
"动态没用"。这反而**更**支持 A2 方向：动态时序确有终止作用，把它做成**局部/空间**的（环绕抑制）或**用-依赖动态**
（z/e_GABA 耗竭）才有希望同时终止核又不伤周边。

## 下一步（PILOT-FIRST，交用户定 — 全网格 NOT 跑）

A1c pilot 信号 = "cannot cleanly quench"，按 §4.1 **不进 396-run 全网格**。可选方向：
1. **测全局-均值驱动型失控**（A1b 的 `l1_g0.7` 那类，全程 timeout）——均匀全局反馈对全局-均值型失控**可能**有效
   （spec §5 已要求按 runaway 类型分层）；本 pilot 只测了核集中型。
2. **接受 screen 结论**：均匀全局反馈≠核集中型失控的解；真 Abbott 机制是**局部环绕抑制塌陷**（空间的，非全局）
   或 **z/e_GABA 使用依赖动态耗竭（A2）**。A1c 给的负结果正好把方向推向 A2 / 局部机制。

整段 A1（a1a+a1b+a1c）的移交见 `m3a_a1_handoff_2026-06-25.md`。

关联：[[project_topic4_sef_hfo_m3a_stage3_core]]、`m3a_a1b_state_topography_2026-06-25.md`、
`a1c_dynamic_global_feedback_spec_2026-06-25.md`（vetted spec + P1 修订）、`docs/paper/abbott_model.md`。
