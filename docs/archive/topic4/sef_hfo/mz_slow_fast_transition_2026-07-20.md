# MZ 慢–快动力学转变 — archive (2026-07-20; verdict 2026-07-21 review 后下调)

**Branch** `codex/topic4-mz-slow-fast-transition`（基线 `codex/topic4-mz-onset-dynamics`@7477453）
**Design** `docs/superpowers/specs/2026-07-20-topic4-mz-slow-fast-transition-design.md`（+ plan）
**Tier** model-side mechanism analysis。口径 = operational runaway（120 Hz / 100 ms），**不是发作复现**。

> **验收状态（2026-07-21 审阅后）**：**工程完成、自然表型复现完成、慢—快 operational-runaway 转变*候选*成立；
> 分叉机制尚未验收；finite-amplitude escape 尚未排除。** 早先版本（本文件初稿）把结果写成"慢状态位置控制的陡分叉、
> 排除有限幅逃逸、m 不影响边界、四条件三 seed 异常干净一致"——这些**过强**，已在下面 §4/§5 逐条收回并说明原因。

---

## 摘要（朴素话）

**测了什么。** 模型两个"慢旋钮"随间期事件漂：去抑制 `D = 1 − z̄`、放电适应 `a`。这次问的正交问题：把慢旋钮在
漂移路上某一时刻**冻住**，只让快系统（膜电位/突触/放电）继续跑——它是稳在"失控边界"下面，还是掉过去了？边界是
什么性质（陡阈值/差一脚就过去/噪声运气/平滑）？

**怎么测的。** 沿自然漂移注册 8 个时刻（baseline / mid / onset 前 2000/1000/500/200/100 ms / 首次失控，**时间锚点
= z-only onset**），4 条件（z-only、runaway τ=500、edge τ=1000、plateau τ=2000）× 3 seed。每个时刻冻 z/m 后：换 12 份
独立未来噪声看自发失控率（P_runaway）；全 E 统一阈值下调短脉冲看点火阈值（ε_c，最高 0.20×阈隙）；亚阈脉冲看恢复
时间（τ_rec）；+ z×m 反事实。

**能站住的（solid）。**
1. **自然表型完整复现 onset-dynamics**：z-only 3/3 失控、runaway 3/3 延后失控（自身 crossing 11–13 s）、edge 1/3
   （seed4）、plateau 0/3；z-only crossing 落在锁定锚点 +40 ms 内（`run_loop` 复现确认）。这是最可靠的结论。
2. **一个慢—快状态转换的候选现象**：沿自然漂移，冻结态从"低逃逸概率"过渡到"first-crossing 处高逃逸概率"；在**所采样
   的低活动亚阈状态 + 所测的单一扰动族（全 E、阈值降、10 ms、≤0.20）内**，没点着它们。

**还站不住的（NOT established，早稿说过头，收回）。**
- ❌ "D≈0.087 的陡分叉"：first-crossing 的 D 实测是 **0.08965 ± 0.0017，范围 0.0868–0.0925**（不是单一 0.087）；
  且 MZ 条件的自身 onset 附近**几乎没采样**（见 §5-P0b），Panel B 中橙线是从 D≈0.06 直接连到 crossing 的**插值**。
- ❌ "排除有限幅逃逸"：只测了一个扰动族/幅度量程；且 ignition ladder 出现**非单调**（§5-P1a），标量 ε_c 在那里失效。
- ❌ "m 不决定离边界多近"：反事实分支点在边界以下且**噪声未配对**，是无信息 null（§5-P0c）。
- ❌ "四条件三 seed 异常干净一致 / 处处 0/12"：实测有明显 seed/状态方差（§5-P1b），mean-D **不是充分状态坐标**。
- ❌ 分叉类型（saddle-node / Hopf / basin / slow-passage）**未识别**；`first_crossing` 的 P=1、ε_c=0 **混入了快状态历史**（§5-P0a）。

---

## 1. 自然行为 = 完整复现 onset-dynamics（solid）

| 条件 | 自然 crossing（ms，三 seed）| onset-dynamics 对照 |
|------|------|------|
| z-only(A=0)     | 9337 / 9540 / 9797（vs 锚点 +40ms EMA 暖机）| 3/3 runoff ✓ |
| runaway(τ=500)  | 12999 / 11054 / 12717 | 3/3 delayed（seed1 对 ground truth 12956，+43ms）✓ |
| edge(τ=1000)    | seed4=15443；seed1/3 无 crossing | "plateau 2 + run-off 1" = 1/3 ✓ |
| plateau(τ=2000) | 三 seed 全无 crossing | bounded 3/3 ✓ |

## 2. 冻结快系统 — 每状态 P_runaway / ε_c / τ_rec（seed 平均；**注意 §5 混杂**）

pre-onset 状态**多数** `P_runaway = 0`、`ε_c 删失`；`first_crossing`（D≈0.089）`P=1`、`ε_c=0`。但**不是处处 0**：
- z-only seed3 `pre_onset_100ms`（D=0.0840）：**2/12 runaway，P=0.167，Wilson CI 0.047–0.448**（非零）。
- matched-D=0.08：**mz_runaway seed3 = 12/12（P=1）**，而 seed1/seed4 = 0/12 —— 同一 mean-D 下 seed 差异极大。
全量 91 行见 `results/.../slow_fast_transition_summary.csv`；matched-D 见 `matched_d/`。

## 3. Result-neutral 分类（预注册 design §5，faithful）

| 条件 | per-seed labels | consensus |
|------|------|------|
| z-only   | unresolved / dynamical_tipping / unresolved | seed-inconsistent |
| runaway  | unresolved ×3 | unresolved |
| edge     | smooth_crossover / smooth_crossover / unresolved | seed-inconsistent |
| plateau  | smooth_crossover ×3 | smooth_crossover |

**多数 unresolved/smooth 是保守判定**（design §5 的 dynamical_tipping 要求 τ_rec 临界减速上升，而 τ_rec 在失控点删失、
pre-boundary 又噪声大），但**按纪律不事后升级**。**注意**：这个"保守"叠加了 §5 的采样/混杂问题，不能读作"接近确认 tipping"。

## 4. Verdict（下调，诚实）

**可以说**：MZ 自然慢状态漂移伴随一个从低逃逸概率状态到持续 operational runaway 的**状态转换**；该转换与去抑制水平
**和快状态历史共同相关**。自然表型复现无误。

**不能说**：不能称"D≈0.087 陡分叉/已确认 dynamical tipping"；不能称"已排除 finite-amplitude escape"；不能称"m 不影响
边界"；不能称"四条件干净一致 / mean-D 是充分坐标"；未识别分叉类型。**禁**：发作/thermodynamic phase transition/证明 onset。

## 5. 混杂 / 局限（审阅 P0/P1，逐条 + 实测证据）

- **P0a — first-crossing 混入快状态**：`first_crossing` checkpoint 继承自然轨迹当时的 V/突触/ring/OU，此时快系统已在
  持续高活动上升支。随后 P=1、ε_c=0 只说明它**从高活动态继续 runaway**，不能单独归因于 z/m 达到某值使**低活动**快系统
  失稳。要证慢参数控制的分叉，需在**同一 z/m** 下比较 low / native / high 三种标准化 fast 初值。（未做。代码位点见文末
  code-comment。）
- **P0b — MZ 自身 onset 附近未采样**：所有 pre_onset 锚定 z-only onset。最后一个注册点到自身 crossing 的间隔实测：
  z-only ~139–144 ms，**MZ-runaway 1654–3805 ms**，edge seed4 5785 ms。故 Panel B 橙线 D≈0.06→0.089 是插值，**不能据此
  称陡转变**。（matched-D 只到 0.08 < crossing 0.089，且带同样的快状态混杂。）
- **P0c — 反事实无信息 + 噪声未配对**：分支点 = z-only onset−100 ms，对 MZ 远低于其边界，任何操纵都可能维持 P≈0；且不同
  反事实用**不同 RNG stream**（`branch_rng_state` 把 branch label 纳入 key）。**铁证**：z-only 的 `native_zm` 与 `m→0` 在
  物理上**完全相同**（z-only m≡0），seed3 却得 P=0.0 vs **P=0.25**——差别纯来自噪声抽样，不是 m。故不能判别 z vs m。
- **P1a — 未排除 finite-amplitude + 非单调点火**：只测一个扰动族（全 E、阈值降、10 ms、≤0.20×阈隙）。且 **mz_runaway
  seed3, matched-D=0.08 的 `ignition_ran = [T,T,T,T,F]`**——最大扰动反而不 runaway，违反标量点火阈值单调假设。此处应报
  nonlinear response curve，而非压成 ε_c。
- **P1b — 非"干净一致"**：见 §2（2/12、12/12 vs 0/12、crossing-D 范围 0.0868–0.0925）。这反而提示**有价值的意外**：
  **mean-D 不是充分状态坐标**——快状态历史 / z 空间分布 / 局部电流仍可能决定是否逃逸。
- **P1c — 无可靠临界减速**：τ_rec 大量为 0、非单调、脉冲是否产生可比响应无 gate、baseline band 过宽会把"几乎没被扰动"
  记为立即恢复。故 unresolved 不只是"crossing 删失"，pre-boundary 数据本身也不足。

## 6. 要坐实机制，需补（审阅 §7 最小修改路线）

1. 只补 **MZ 自身 onset 周围**稠密 checkpoint（−1000/−500/−250/−100/−50/−20/0 ms，或 D=0.076–0.094 步长 0.002）。
2. 每个 z/m 状态从 **low / native / high 三种 fast 初值**出发，**同一组未来噪声配对**。
3. **D 上/下扫 + 双初值**：共存+滞回 → saddle-node/bistability；小振荡+振幅连续增长 → Hopf；只有 native ramp 会跑、
   标准低态不跑 → slow-passage / history-dependent transition。
4. 扰动改成**完整 response surface**；只有响应随幅度单调时才定义 ε_c。
5. 重画图：A 对齐**各条件自身 onset**；B 显示 dense P(D)（含 matched-D）；C 明示右删失下界；D 用 matched-boundary +
   paired-noise 反事实。
   另：counterfactual common-random-number 配对、ignition 单调性检查（失败禁输出 ε_c）、matched-D 保存 `a`+fast-state
   summary、独立 `provenance.json`、清理 `run.pid`。

## 7. 资源 / provenance

- Substrate = E1146 narrow twoend_equal，**40000 neurons**（NE=32000）；build ~137 s、峰值 RSS ~6.8 GB/worker。
- 并发：pilot 后 4 workers → 用户要求提速后 **12 workers 一波**（内存 gated，used 峰 ~110/251 GB，非 oom）；wall ~7.5 h
  （内存带宽 bound）。12/12 units 成功、0 failure。
- 代码：新文件复用 `MZOnsetProbe`/`run_loop` checkpoint-resume（**零 engine 编辑**，`git diff --quiet src/snn_engine` 过）；
  16 测试绿。两个已修 bug（首单元验证抓到）：natural_tail 截断延后 MZ crossing（1500→6000）；spawn 孤儿进程（进程组 kill）。
- 产物 `results/topic4_sef_hfo/mz_slow_fast_transition/`。

## 8. Non-goals（守住）

无空间 eigenmode / 主轴 / source-sink / field bridge；不改 onset-dynamics 锁定产物；不 push / 不 merge。
