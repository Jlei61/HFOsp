# M4/MZ conductance 分阶段验收与 agent handoff

日期：2026-07-20

分支：`codex/topic4-mz-conductance`

工作树：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-conductance`

## 1. 当前总判决

本阶段可以验收为：**膜方程和工作点成立，Z bridge 成立，terminal runaway 的时空表型成立；完整 interictal→ictal→recovery lifecycle 不成立。**

因此当前工作不是“模型失败后没有结果”，也不是“已经找到发作态”。它锁定了一个可重复的间期基线和一条明确的失败边界，下一机制必须把 terminal runaway 变成有界且可恢复的高招募态。

## 2. 分阶段 gate

| Stage | 问题 | 判决 | 已有证据 / 边界 |
|---|---|---|---|
| A0 工程合同 | exact conductance 是否 off-by-default、可复现且数值安全 | **ACCEPT** | reversal potential、conductance-dependent `tau_eff`、manifest 与定向测试已锁；旧路径默认不变 |
| A1 工作点 | `L=20` 是否有稳定的自发间期事件工作点 | **ACCEPT** | seed 1/3 均有 returning interictal events；无数值 clipping，`tau_eff_min > 2dt` |
| A2 Z bridge | Z 是否能在事件间留下稳定慢记忆并推动系统接近转换 | **ACCEPT** | seed 1/3 均出现 event-locked `D=1-z` staircase，随后进入 runaway |
| A3 时空转变 | 是否从有序间期传播切换到另一空间表型 | **ACCEPT as terminal-transition phenotype only** | returning event 保留 source→sink 轴向时序；early runaway 招募扩大且方向梯度塌缩 |
| A4 global/local inhibition | protected additive-global GABA 是否形成新发作态 | **NO-GO as lifecycle generator** | 它形成 runaway→near-prevention→suppression bracket，但只移动边界，没有独立慢维度 |
| A5 当前 M | 线性、从低活动即累积的 M 是否提供延迟终止和恢复 | **NO-GO** | 弱时压不住 runaway；稍强时先压掉间期事件或进入 prevention，onset 与 termination 未分离 |
| Final lifecycle | 是否存在 bounded ictal bout、终止、恢复原间期模板 | **NOT ACCEPTED** | 未见 bounded high state、limit cycle/bistability、post-bout recovery 或恢复后的 returning event |

## 3. 当前唯一安全 claim

当前模型在 `L=20` 上能自发产生稳定间期事件、event-locked Z 消耗阶梯，并从有序轴向传播转入广招募、方向性下降的 terminal runaway。conductance 与 protected additive-global inhibition 提供了可解释的局部/全局 restraint bracket，但没有产生有界、可恢复的发作态。

禁止写成：

- 已复现 seizure / ictal attractor；
- 已证明 bistability、Hopf 或 limit cycle；
- 已实现 interictal→ictal→interictal closed lifecycle；
- global inhibition 或当前 M 已解释发作终止；
- early-runaway 空间图就是一个独立、可恢复的 seizure state。

## 4. 阶段图与机器合同

阶段机制图：

- `results/paper-ready-figure/fig_mz_conductance_current_dynamics/figures/mz_conductance_current_dynamics.png`
- `results/paper-ready-figure/fig_mz_conductance_current_dynamics/figures/mz_conductance_current_dynamics.pdf`
- `results/paper-ready-figure/fig_mz_conductance_current_dynamics/figures/mz_conductance_current_dynamics_metadata.json`
- `results/paper-ready-figure/fig_mz_conductance_current_dynamics/figures/README.md`

producer：`scripts/paper_figures/plot_fig_mz_conductance_current_dynamics.py`。图使用同一条 `L=20 / seed=1 / beta=1/12 / Z on / M off` 连续轨迹；metadata 锁定 source artifact、窗口、选择规则和全部空间读出。原始 full/capture simulation 属于 ignored runtime artifact，不进入 git；需要重跑时从已提交的 config 和 runner 重新生成，不能把图像像素当数值输入。

图中阶段事实：

- returning event：onset-axis Spearman `+0.959`，保留 source→sink 时序；
- early runaway：onset-axis Spearman `−0.118`，招募扩大且方向性丢失；
- 连续轨迹：19 个 pre-runaway returning events，runaway time `7180.1 ms`；
- 图只支持 terminal-transition phenotype，不支持 recovery claim。

## 5. 复现锚与验收命令

冻结锚：`L=20`、`gaba_gain=1.125`、protected additive-global `beta=1/12`、`q75`、`tau_z=2.5 s`、primary seeds 1/3。入口与配置：

- `scripts/run_topic4_mz_conductance.py`
- `config/topic4_mz_conductance.yaml`
- `config/topic4_mz_conductance_z_staircase_seed1.json`
- `config/topic4_mz_conductance_z_confirm_seed3.json`
- `config/topic4_mz_conductance_figure_capture_seed1.json`

最小回归测试：

```bash
pytest -q \
  tests/test_mz_conductance.py \
  tests/test_topic4_mz_conductance.py \
  tests/test_mz_slow_vars.py \
  tests/test_topic4_mz_slowvars.py
```

任何新机制必须先复现 A1/A2 锚，再只改变一个机制；不得用外部 pulse、提高 recurrent excitation、放松 runaway 判据或单点 cherry-pick 抢救 lifecycle。

## 6. 下一条独立机制线

本 worktree 下一步不继续细扫 `beta`，也不把当前线性 M 当主恢复机制。优先加入 presynaptic E→E resource `x_j`：

\[
\tau_x\dot x_j=1-x_j-U_x x_j r_j(t),\qquad
W^{EE}_{ij,eff}=W^{EE}_{ij}x_j.
\]

设计目的不是再加一个均匀刹车，而是让传播前沿耗竭已经使用过的 outgoing E→E relay，形成空间 refractory wake。第一轮只做 `U_x × tau_x` cheap grid，M off；先问能否把 terminal runaway 分成“仍 runaway / 一下即灭 / bounded high-recruitment bout”。只有第三类出现后，才进入跨 seed recovery、空间前沿、尾迹恢复和 fast-subsystem hysteresis/limit-cycle 检查。

若 `x_j` 只能产生一下即灭，第二配料才允许是 **thresholded dynamic global brake**；它必须与当前瞬时 protected additive-global restraint 分开命名、分开状态变量、分开 ablation。

## 7. 对其他 agent 的协作边界

- 以本文件作为本分支阶段状态的唯一入口；细节反思见 `mz_conductance_dynamics_reflection_and_next_model_2026-07-20.md`。
- 可读取和复用 frozen workpoint、runner、测试、图 producer 与 gate；不得把 ignored runtime results 当已提交 cohort evidence。
- 另一条并行机制线保持独立，不复制其未验收参数或结论到本分支；比较时只对齐共同 lifecycle gate。
- 未经用户要求，不 merge、不 push、不改其他 worktree；新实验继续控制并发和单任务内存，L=20 长程仿真一次只保留必要 trace。
