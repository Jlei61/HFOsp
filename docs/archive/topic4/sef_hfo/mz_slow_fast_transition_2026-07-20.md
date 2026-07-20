# MZ 慢–快动力学转变 — archive (2026-07-20)

**Branch** `codex/topic4-mz-slow-fast-transition`（基线 `codex/topic4-mz-onset-dynamics`@7477453）
**Design** `docs/superpowers/specs/2026-07-20-topic4-mz-slow-fast-transition-design.md`（+ plan）
**Tier** model-side mechanism analysis。每个表型是 detection label；口径 = operational runaway（120 Hz / 100 ms），
**不是发作复现**。

---

## 摘要（朴素话）

**测了什么。** 模型里两个"慢旋钮"随间期事件重复而漂：抑制效能被用掉（去抑制 `D = 1 − z̄`），放电疲劳累积
（适应 `a = η_m·m̄ / I_EE`）。之前的 onset-dynamics 线画的是这两个旋钮*怎么漂*。这次问一个正交问题：把这两个旋钮在
漂移路上的某一时刻**冻住不动**，只让快的东西（膜电位、突触、放电）继续跑——这个被冻在原地的快系统，是稳稳待在
"失控边界"下面，还是已经掉过去了？而且那个边界是"陡阈值"、还是"差一脚就能被推过去"、还是"靠噪声运气"、还是
"平滑过渡"？（不预设。）

**怎么测的。** 沿自然漂移注册 8 个时刻（baseline / mid / onset 前 2000/1000/500/200/100 ms / 首次失控），4 个条件
（z-only、runaway τ=500、edge τ=1000、plateau τ=2000）× 3 个 seed。在每个时刻冻住 z/m，对同一个冻结态：① 换 12 份
互相独立的未来噪声，看它自己失控的比例（P_runaway）；② 给全体兴奋细胞加一个统一的、越来越强的短脉冲，看要多强才
能把它点着（点火阈值 ε_c，最高一档 = 阈值间隙的 0.20）；③ 给一个明确亚阈的脉冲，看它多久回到基线（恢复时间 τ_rec）。
再做 z×m 状态互换的反事实（把 m 清零、把 z 复原、早晚 z/m 互换）。

**揭示了什么。** 结果异常干净、四条件三 seed 一致：**只要慢状态还在走廊以下（D 一路到 ~0.08），冻结的快系统就纹丝
不动——12 份噪声里 0 次自发失控，而且我们最强的一巴掌（ε_c 到 0.20）也点不着它**；**慢状态一到走廊（D≈0.087），
它不用任何外力就自己失控（ε_c = 0）**。也就是说，失控是被"慢状态漂到某个特定的 D"这件事触发的，不是快系统本身
是个一触即发的亚稳态。这**排除了"差一脚就被推过去"（finite-amplitude escape）**：如果是那样，越靠近边界、需要的
那一脚应该越小；实测却是走廊以下"一直踢不动"（阈值在我们的量程之外），到走廊突然变成 0。反事实也印证：在 onset
前 100 ms，四条件都还在边界下面，任何 z/m 互换都推不过去；把 m 清零 / 早晚互换都不改变结果 → **离边界多近是 z（D）
说了算，m 不管**（这和 onset-dynamics"适应只推迟到达、不移动边界"一致）。

**一句话的边界口径。** 在 E1146 的注册范围内，冻结的快系统稳稳待在一个陡的去抑制边界（D≈0.087）下面、在那里用我们
最强的统一扰动也点不着；只有当慢状态漂到这个边界，它才自发失控。这**与"慢状态位置控制的陡转变"一致、与"有限幅
逃逸"不一致**。预注册的 result-neutral 分类器多数给 `unresolved`/`smooth`，**唯一原因**是它的"陡转变"判据里还要求
"临界减速（τ_rec 随逼近边界而上升）"，而 τ_rec 恰恰在失控点**删失**（失控没有恢复可言）——所以那一条无法确认，
属分类器**保守**，不是反证。

---

## 1. 自然行为 = 完整复现 onset-dynamics（验证）

| 条件 | 自然 crossing（ms，三 seed）| onset-dynamics 对照 |
|------|------|------|
| z-only(A=0)     | 9337 / 9540 / 9797（vs 锚点 9293.6/9499.3/9757.9，+40ms EMA 暖机偏移）| 3/3 runoff ✓ |
| runaway(τ=500)  | 12999 / 11054 / 12717 | 3/3 delayed runoff（seed1 对 ground truth 12956.2，+43ms）✓ |
| edge(τ=1000)    | seed4=15443；seed1/3 无 crossing | "plateau 2 + run-off 1" = 1/3 ✓ |
| plateau(τ=2000) | 三 seed 全无 crossing | bounded plateau 3/3 ✓ |

`run_loop` 复现无误（z-only crossing 全部落在锁定锚点 +40ms 内），MZ 条件的延后 crossing 也对上 onset-dynamics。

## 2. 冻结快系统 — 每状态 P_runaway / ε_c / τ_rec（seed 平均）

四个条件的 pre-onset 状态**全部** `P_runaway = 0`、`ε_c 删失`（三 seed 一致）；只有 `first_crossing`
（D≈0.089–0.090）`P_runaway = 1`、`ε_c = 0`（zero_runaway，三 seed 一致）。代表（z-only / runaway）：

```
z-only     D≈0.009→0.081(pre-onset)  P=0   ε_c=cens ;  crossing D≈0.090  P=1  ε_c=0
runaway    D≈0.008→0.063(pre-onset)  P=0   ε_c=cens ;  crossing D≈0.089  P=1  ε_c=0
edge/plateau  所有 pre-onset  P=0  ε_c=cens ;  (edge seed4 crossing P=1 ε_c=0；plateau 无 crossing)
```
（全量 91 行见 `results/topic4_sef_hfo/mz_slow_fast_transition/slow_fast_transition_summary.csv`。）
唯一非平凡的边缘点：z-only `pre_onset_100ms`（D≈0.081）P≈0.06、`pre_onset_200ms` τ_rec≈260 ms（临界减速的
一点苗头），但都在走廊以内才发生。

## 3. Result-neutral 分类（预注册 design §5，faithful）

| 条件 | per-seed labels | consensus |
|------|------|------|
| z-only   | unresolved / **dynamical_tipping** / unresolved | seed-inconsistent |
| runaway  | unresolved / unresolved / unresolved | unresolved |
| edge     | smooth_crossover / smooth_crossover / unresolved | seed-inconsistent |
| plateau  | smooth_crossover / smooth_crossover / smooth_crossover | smooth_crossover |

**为什么会 unresolved（不是分类器 bug，是预注册保守）**：design §5 的 `dynamical_tipping` 要求四条同时成立
——(a) P_runaway 陡跳 0→1、(b) ε_c→0、(c) τ_rec 随逼近边界上升、(d) 自然轨迹在此 crossing 而 plateau 停在外面。
实测 crossing 条件 **(a)(b)(d) 无歧义成立**，但 **(c) 在失控点删失**（失控无恢复），加上 pre-onset 的 τ_rec 本身噪声大
不单调，(c) 无法确认 → 落到 `unresolved`。z-only seed3 恰好其 τ_rec 满足了 (c) → 命中 `dynamical_tipping`，另两 seed
没有 → seed-inconsistent（体现 τ_rec 判据的 seed 脆弱，P-step / ε_c pattern 三 seed 是一致的）。plateau/edge-非crossing
的 `smooth_crossover` 实际含义 = "始终在边界以下，没采样到转变"。**按预注册纪律，不把 `unresolved` 事后升级为
`dynamical_tipping` 作 cohort label。**

## 4. Verdict（分层，诚实）

- **主口径（allowed）**：冻结快系统在陡的去抑制边界（D≈0.087，= onset-dynamics runoff 走廊）以下**稳健且不可点火**
  （ε_c 全程超出 0.20 量程），到边界处**自发失控**（ε_c=0、P_runaway=1）。这**与"慢状态位置控制的陡转变"一致、
  与"有限幅逃逸"不一致**（ε_c 不随逼近边界渐降，而是删失→0 的突变）。
- **反事实（allowed）**：pre-onset 100 ms 的 z×m 互换全部 P≈0；**m 不决定离边界多近，只有 z/D 决定**
  （印证 onset-dynamics"适应只推迟到达、不移动边界"）。
- **分类器口径（faithful, 不升级）**：预注册 result-neutral 标签 = z-only/edge `seed-inconsistent`、runaway
  `unresolved`、plateau `smooth_crossover`；`unresolved` 的唯一缺口是 τ_rec 临界减速判据在失控点删失、无法确认。
- **禁止**：写"发作/seizure"、"（热力学意义的）phase transition"（需要有限尺寸 N 标度，未做）、"证明 onset"；
  禁止把 `unresolved` 当 `dynamical_tipping` 的 cohort 主张。

## 5. Caveats / limitations

1. **τ_rec 在失控点删失** → `dynamical_tipping` 第 4 条判据无法确认 → 分类器保守。这是判据与"陡失控无恢复"之间的
   张力，**如实报，未事后改分类器**（改了就是 result-driven tuning）。
2. **反事实分支点（pre_100）在边界以下** → 反事实是 null（在 pre_100 谁都推不过去，无法判别 z vs m 的边界-邻近作用）。
   更有判别力的反事实应分支在各条件自己的 crossing-D 处，**deferred**。
3. **只是观测层**：没有对边界本身做因果操纵。
4. n_replay=12（spec 下限，40k-neuron 共享机算力预算）、单被试 E1146、3 seed。

## 6. 资源 / provenance

- **Substrate** = E1146 narrow twoend_equal，**40000 neurons**（NE=32000）；build ~137 s、峰值 RSS ~6.8 GB / worker。
- **并发**：pilot 后先 4 workers（礼让），用户要求提速后 **12 workers 一波**（内存 gated，实测 used 峰值 ~110 GB /
  251 GB，123 GB cache 可回收 = 非 oom）；全跑 wall ~7.5 h（12 个 40k-neuron 仿真 + 别的 session 一起挤内存带宽）。
- **代码**：新文件复用 onset-dynamics 的 `MZOnsetProbe`/`run_loop` checkpoint-resume（**零 engine 编辑**，
  `git diff --quiet src/snn_engine` 通过）；16 个测试全绿。
- **两个已修 bug**（首单元验证抓到）：natural_tail 截断延后 MZ crossing（1500→6000）；被杀 driver 的 spawn 孤儿进程
  （改进程组管理 `kill -- -<pgid>` + setsid）。
- **产物** `results/topic4_sef_hfo/mz_slow_fast_transition/`：`per_state/`（8 checkpoint × JSON + natural.npz）、
  `counterfactual/`、`matched_d/`、`slow_fast_transition_summary.{csv,json}`、`STATUS.md`、`figures/`（四联图 + README）。

## 7. Non-goals（守住）

无空间 eigenmode / 主轴 / source-sink / field bridge；不改 onset-dynamics 锁定产物；不 push / 不 merge。
