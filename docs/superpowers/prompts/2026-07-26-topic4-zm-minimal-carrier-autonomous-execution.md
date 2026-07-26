# Topic 4 Z/M minimal-carrier branch decision — autonomous execution prompt

你现在是这个任务的自主执行 agent。请直接在指定 worktree 内工作，按已锁定的
Revision 3.1 spec 和 implementation plan 完成 **Phase 0–3 branch decision**。
在范围内不需要反复向用户确认；遇到 spec 明定的终止条件时，诚实停机、归档并汇报，
不要为了得到正结果而继续调参。

## 0. 工作位置与读取顺序

工作位置：

```text
/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-m4-snn-native-exit
```

分支必须是：

```text
codex/topic4-m4-snn-native-exit
```

开始前依次完整阅读：

1. `/home/honglab/leijiaxin/HFOsp/AGENTS.md`
2. `docs/archive/topic4/sef_hfo/zm_carrier_exit_line_acceptance_2026-07-26.md`
3. `docs/superpowers/specs/2026-07-26-topic4-zm-minimal-carrier-branch-decision-design.md`
4. `docs/superpowers/plans/2026-07-26-topic4-zm-minimal-carrier-branch-decision.md`
5. `docs/topic4_sef_hfo.md` 中当前 Topic 4 / Z–M 状态
6. 真实实现及既有测试：
   - `scripts/run_zm_snn_native_exit.py`
   - `src/snn_engine/kick_probe.py`
   - `src/snn_engine/slow_field.py`
   - `src/topic4_zm_carrier_gate_v2.py`
   - `tests/test_zm_slow_field_parity.py`
   - `tests/test_snn_shunting.py`
   - `tests/test_snn_gates.py`
   - `tests/test_a1c_feedback.py`
   - `results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json`
7. 只有进入最终绘图任务时再完整阅读 `docs/figure_style_guide.md` 的 Topic 4
   小节。

spec 是科学合同，plan 是执行顺序。两者冲突时先停下记录冲突，不得自行选择一个更容易
通过的版本。不要沿用旧 q_I/g_K sandbox 的语义，也不要把旧 reduced-field 或
excitable-wave spec 当成当前合同。

## 1. 核心科学目标

我们不是为了“造出一个振荡”，也不是为了证明某个附加电流可以把网络压静默。

原始科学问题是：

> 在保留各向异性空间 E→E scaffold、原始二维 E/I spiking 网络、Z/M 慢变量和
> 虚拟电极读出的前提下，反复的间期事件是否能通过 Z 耗竭推动网络进入一个有界、
> 空间组织化、可观测的发作载体；现有慢变量能否把它带入并带出该载体窗口？

本轮不直接实现完整 lifecycle，而是回答它的上游分支问题：

> 在真实 Z/M+\(S_G\) 轨迹经过或邻近的慢状态上，支持 bounded
> stable/metastable carrier 的最小动态子系统究竟是 fast E/I、fast E/I+\(M\)、
> fast E/I+\(S_G\)，还是 fast E/I+\(M+S_G\)？

必须区分四种失败来源：

1. 局部 fast/slow neighbourhood 内根本没有 carrier；
2. carrier 存在，但当前 Z/M/\(S_G\) 慢轨迹没有进入它；
3. carrier 和 entry 存在，但现有慢坐标没有合适的 offset 方向；
4. source-space carrier 存在，但 current-based virtual-SEEG observation
   mapping 不匹配真实 early-ictal 语义。

本轮的成功不是“完整可恢复发作”。本轮的成功是得到一个可复核的、fail-closed 的
branch decision，并说明证据停在哪一层。

## 2. 已知事实与不能退回的旧解释

必须以这些当前事实为起点：

- 正确衬底是 per-neuron Z/M SNN，不是 q_I/g_K。
- 网络为 E1146 `twoend_equal`，\(N=40000\)、\(N_E=32000\)、
  \(N_I=8000\)，E→E 各向异性连接、病理轴和拓扑保持不变。
- 当前锁点为 `zA_q75_tz5000__mA0p001_tau500`：
  \(\tau_z=5000\) ms、\(\tau_m=500\) ms、\(\eta_m=0.001\)、
  \(I_{\mathrm{th},EI}\approx1.28\)。
- bare Z/M 能产生 returning interictal events → recruitment/escalation →
  runaway。这证明进入链存在候选，但 runaway 不是发作吸引子。
- Z/M+\(S_G\)（初始锚点 \(\alpha_G=16\)）把 runaway 整形成有界的低占空比
  recurrent burst train；carrier gate v2.1 将其判为
  `fail_hfo_like_train`，不能叫持续 ictal carrier。
- \(H\) 失败的直接原因是 bursty low-mean state 喂不饱 persistence sensor。
  本轮不继续调 \(H\)。
- local-inhibition reduced field 的结论是 `both_stable`：局部化抑制没有使
  均匀振荡横向失稳。本轮不把该 field 迁回 SNN。

因此禁止写：

- “当前已经存在可控 ictal lifecycle”；
- “Z/M+\(S_G\) 的 burst train 就是 ictal state”；
- “终止到静默等于恢复到间期”；
- “fast E/I 必然是 carrier”；
- “M 必然是 exit variable”；
- “一次 seed-1 负结果证明 Z/M 不可能产生 carrier”。

## 3. 锁定衬底与独立性边界

必须保持：

- `use_z=True`, `use_m=True`, `use_qI=False`, `use_gK=False`;
- no \(H\), no persistence current, no new actuator；
- 不修改 E→E 权重、kernel、各向异性、拓扑或 pathology geometry；
- 不借用并行 E→E 机制线的修改；
- 不迁移被否决的局部抑制 field；
- 不做大而无因果解释的全参数网格；
- 不把 rate toy model 的正结果替代为 SNN 结论。

允许的唯一 guarded engine 修改是：

```text
src/snn_engine/kick_probe.py
```

并且只能加入最小、off-by-default 的 checkpoint/freeze hook。不得复制第二套
integration loop。其他 connectivity、neuron、parameter、LFP guarded engine
文件不能修改。

如果 worktree 内出现无法归因于本任务的改动，先只读核对，不覆盖、不清理、不提交；
若与任务文件冲突则停机报告。

## 4. 执行原则

严格按 plan 的 Task 1 → 14 顺序执行。每项采用 TDD：

1. 先写能捕获科学/工程合同的失败测试；
2. 做最小实现；
3. 运行定向测试；
4. 运行相关回归；
5. 验证真实 artifact/schema/provenance；
6. 做小而单一目的的 commit；
7. 再进入下一 task。

不得把多个尚未验证的机制塞进一个提交。不得为了赶进度跳过 Task 3.5 vertical
slice。不得先跑长仿真后补状态合同。

每次提交前至少运行：

```bash
git diff --check
git status --short
```

只提交本任务文件。不要 push、merge、rebase 或改写 peer 分支；除非用户另行明确要求。

### 4.1 Phase 0 的 P0 合同

先完成完整 dynamic-state inventory 和 canonical config lock。任何影响膜电流但
未分类的状态都是 P0。

checkpoint 必须保留：

- membrane/refractory state；
- synaptic gates/currents；
- recurrent-E current；
- delay ring buffers 和 cursor；
- external/OU state；
- simulator RNG 的完整 bit-generator state；
- per-neuron \(z_i,m_i\) 及其传感状态；
- \(r_E^\mathrm{fast},\mu_G,S_G\)；
- 所有启用的 hidden dynamic features；
- 连续 readout 所需的 observer filters/buffers，但 observer RNG 与 simulator
  RNG 分开。

freeze 的含义只能是：

\[
q(t>t_f)=q(t_f)
\]

同时保留该坐标在膜方程中的 current effect。不能 reset、不能只冻结均值、不能停止
记录但让隐藏状态继续漂移。primary \(S_G\) freeze 必须把
\(r_E^\mathrm{fast},\mu_G,S_G\) 作为一个 family 一起冻结。

### 4.2 Guard hash 与 exact resume

修改 `kick_probe.py` 前：

1. 记录旧 SHA 和精确 diff 基线；
2. 保存可用于 pre/post default-path 比对的 fixture；
3. 确保 hook 关闭时不增加 RNG draw、allocation 或浮点运算。

实现后必须证明：

- pre-edit 与 post-edit default path 的 spike raster、source/current traces、
  final RNG state byte-identical；
- continuous run 与 split→save→restore→continue 的 spike raster、所有 final
  simulator state、source-space traces、current-vSEEG、RNG progression
  byte-identical；
- 测试态包含 active Z/M/\(S_G\)、非空 delay buffer、非零 synaptic/recurrent
  currents 和 nontrivial refractory state；
- trough/rising/peak 三种 fork phase 均覆盖；
- config/engine/schema/state hash 或 \(dt\) 不匹配时 fail closed。

只有全部通过后，才更新
`results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json` 中
`kick_probe.py` 的单个 key，并记录 old/new SHA、测试和理由。

不要写“byte parity 所以无需 re-bless”。guarded file 变更本身就要求按合同更新。

### 4.3 Mandatory vertical slice

Task 3.5 必须用一个短 seed-1 真实状态跑通：

```text
anchor
→ checkpoint
→ restore
→ freeze_all
→ continuation
→ source metrics + current-vSEEG metrics
```

同时验证 observer filter 连续和全部 provenance hash round-trip。输出只能写入
`smoke/`，不能更新 production summary，也不能当 carrier 证据。

vertical slice 失败时立即停在 redesign，不得启动 Tasks 4–14。

## 5. 载体的统计定义

不要用单条轨迹肉眼命名吸引子。对每个 minimal-subsystem × slow-bin × fast-phase ×
future-noise 条件，按 spec 的 beta-binomial 合同估计
\(P_\mathrm{carrier}(8s)\)。

分类必须是：

- `stable_carrier`：posterior median \(>0.8\)，variance/drift bounded，lifetime
  显著超过匹配 IED；
- `metastable_carrier`：\(0.3<P\le0.8\)，lifetime 显著超过匹配 IED，且不反复
  reset 到 rest basin；
- `transient_carrier_like`：\(P\le0.3\) 或 lifetime 与 IED 相当；
- `hfo_like_train`：事件之间反复 reset 到 rest basin；
- `probabilistically_indeterminate`：posterior 跨越关键阈值或样本不足。

formal carrier window 需要：

- stable 或 metastable；
- 至少两个相邻 slow bins 有兼容支持；
- 至少两个 natural fast phases 收敛到同一统计状态；
- 至少 2/3 eligible primary seeds 确认；
- primary seeds 固定为 `{1,3,4}`，seed 1 同时只承担 discovery；
- paired future noise 至少包含 `noise_replay`,
  `noise_resample_1`, `noise_resample_2`；`mean_input_only` 只能作额外诊断。

不要把 8 秒存活本身等同于极限环。稳定性、metastability 和跨 phase/noise/seed
收敛必须分开报告。

## 6. Minimal-subsystem arms

对同一个精确 snapshot，至少实现并核对：

```text
dynamic_replay
freeze_z
freeze_zm
freeze_zsg
freeze_all
dynamic_z_only
```

这些 arm 用来识别最小动态载体：

- `freeze_all` 近似 fast E/I；
- 保留 \(M\) 动态对应 fast E/I+\(M\)；
- 保留 \(S_G\) family 动态对应 fast E/I+\(S_G\)；
- 同时保留 \(M+S_G\) 动态对应 fast E/I+\(M+S_G\)。

不要只按 arm 名推断语义；每个 output manifest 都要落盘实际 freeze policy、snapshot
hash、noise-bank hash、engine/config SHA 和 enabled dynamics。

## 7. 真实观察语义

current-based virtual-SEEG 是 primary；rate proxy 只能做 sensitivity。真实 early-ictal
reference、returning IED、sharp pulse-train null 和 matched synchronized-global-
oscillator null 必须锁成不可变输入，记录 SHA256。

carrier gate 至少覆盖：

- duration、occupancy/duty cycle、energy、spatial extent；
- 按 kernel-width 去相关后的 independent-contact count；
- harmonic-comb concentration；
- spectral entropy / broadband continuity；
- instantaneous-frequency drift；
- burst-interval CV 和 temporal phase coherence；
- wavefront-velocity variability；
- spatial phase entropy；
- axial first passage；
- multivariate rest distance 和 dwell。

固定约 5 Hz 的全场同步振荡即使均值、频率和能量匹配，也必须被 temporal/spatial
organisation null 拒绝。不能把脉冲串谐波的高频能量当持续 ictal broadband carrier。

如果真实 early-ictal artifact 缺失：

- 写 `blocked_reference_artifacts.json`；
- 不用模型或 synthetic 数据伪造真实分布；
- state/source-space forks 可在 `observation_layer_blocked` 标签下继续；
- source-only carrier 可进入 functional-rank、modal、entry/offset diagnostics；
- 但不能授权新 actuator，也不能写 observation-matched ictal carrier。

## 8. Anchor 与 neighbourhood

anchor 必须来自真实动态轨迹，不得手工构造 fast state。使用：

- bins：`pre_entry`, `onset_adjacent`, `bounded_early`,
  `bounded_mid`, `bounded_late`；
- fast phases：`trough`, `rising`, `peak`；
- slow-bin 按 trajectory arc length / quantile 选取，避免不等速轨迹造成时间采样偏差。

如果不足三个 eligible bounded anchors，输出
`insufficient_bounded_anchors`，不能因此进入 Branch F。

visited states 无 carrier 时，必须先做 local neighbourhood audit，且同时使用：

1. coarse decision-coordinate PCA；
2. full-field \([z_i,m_i,S_G]\) PCA，至少保留 3 modes；
3. pathology-axis/core-surround projections。

只有三种表示的结论相容，且至少三个 eligible bounded anchors 的局部 neighbourhood
都没有 carrier，才允许 `branch_F_fast_carrier_repair`。

若 visited state 无 carrier、邻近 state 有 carrier：

```text
branch_T_slow_trajectory_repair
```

若 coarse 与 full-field/pathology-axis 结论冲突：

```text
representation_sensitive_no_branch
```

此时必须停机，不能默认 Branch F。

Branch T/F 都是本轮终点；只报告，不实现。

## 9. 找到 carrier 后的顺序

只有 carrier verdict 成立，才继续：

1. Task 9A standardized slow-coordinate functional-rank；
2. Task 9B trajectory-conditioned modal/operator audit；
3. Task 10 Z-entry probability boundary；
4. Task 11 existing slow-coordinate offset boundary；
5. 仅当所有有效 existing-coordinate offset 都不足时，Task 12 matched offline
   exit-driver selection。

不得把 Task 9B 提前，也不得跳过 functional rank 直接猜 exit variable。

### 9.1 Functional rank

对 \(Z/M/S_G\) 使用无量纲、匹配 perturbation norm 的 sensitivity matrix 和 SVD。
近 rank-1 只能写“carrier neighbourhood 内局部功能共线”，不能写两个变量等价、
全局冗余或可以删除。

### 9.2 Entry

entry 是概率边界，不是一条人工阈值线。必须在匹配 fast phase/noise 下估计 Z 方向
进入 carrier 的概率，并区分：

- Z 的确横跨 entry boundary；
- Z 只相关但不构成有效 entry coordinate；
- 当前动态轨迹没有达到边界。

### 9.3 Offset

offset 必须先检验现有坐标族：

- \(M\) alone；
- \(M+S_G\)；
- \(M+Z\)-recovery。

只有这些都无法形成可达 offset boundary，才允许进入离线 driver selection。若
\(M\)-alone boundary 仅以很小、预注册范围的校准就可达，可判
`branch_M_calibration`；只判决，不在本轮调参实现。

Task 12 的 \(P/A\) 等候选只能作 matched offline counterfactual driver 比较，不能在
本轮写进 SNN 或宣称终止机制成立。

## 10. Fail-closed branch verdict

最终判决器必须是纯函数，只读取已完成 phase 的 manifest/metrics；不能在 adjudication
阶段补跑未注册实验或偷偷改阈值。

至少能输出并区分：

- `blocked_state_inventory`
- `blocked_exact_resume`
- `blocked_reference_artifacts`
- `observation_layer_blocked`
- `insufficient_bounded_anchors`
- `probabilistically_indeterminate`
- `representation_sensitive_no_branch`
- `branch_T_slow_trajectory_repair`
- `branch_F_fast_carrier_repair`
- `branch_M_calibration`
- source-space carrier / observation-space carrier 的 stable 或 metastable 层级
- existing-coordinate offset 可达或不足
- downstream driver-selection recommendation

unknown、missing、ineligible 和 smoke evidence 一律不能自动变成 Branch F 或 GO。

每个 phase 之后都检查 stop rule。得到 terminal verdict 后：

1. 停止启动新仿真；
2. 完成当前结果的 provenance 校验；
3. 只画已完成 phase；
4. 写 archive 和最终报告；
5. 不实现判决出的下一分支。

## 11. 资源与自主运行约束

可持续自主执行约 10 小时，或直到出现 terminal stop / P0 blocker。优先 cheap-first，
但在所有 gate 通过后充分利用 CPU 和内存。

所有进程固定：

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

资源纪律：

- 起初只允许 1 个 full 40k SNN worker；
- 先实测单臂 peak RSS 和 wall time；
- 只有在实测预算允许后，最多并行 2 个 full SNN workers；
- 始终保留 `MemAvailable >= 96 GB`；
- swap 不能增长；
- 每约 2 分钟将 CPU、RSS、MemAvailable、swap、PID、arm、seed、phase 写入耐久日志；
- 达到阈值时停止新 launch，必要时只终止本任务最新启动的 worker；
- 不杀 peer worktree 或用户进程；
- 禁止靠盲目并行造成 OOM；
- 离线分析/绘图可在内存许可下并行，但不能和两个高 RSS SNN 叠加到危险区；
- runner 必须 crash-safe、per-arm 原子写入、可从已验证 manifest 续跑；
- resume 前必须重新核对 config/engine/state/noise SHA，不能仅凭文件存在跳过。

长仿真启动前必须确认 Tasks 1–3.5 全绿。不要使用 `nohup` 后失去 provenance；
后台任务必须有明确 PID、日志、输出路径和完成状态。不要无限等待一个失败进程。

## 12. 输出合同

结果根目录固定为：

```text
results/topic4_sef_hfo/zm_branch_decision/
```

每个 production artifact 至少带：

- git SHA；
- engine/config/state/noise/input artifact SHA；
- seed、arm、slow bin、fast phase、duration、\(dt\)；
- eligibility 和 stop-rule 状态；
- schema/gate version；
- timestamp；
- peak RSS 和 resource-log pointer。

smoke、discovery 和 production 分目录，不能相互覆盖。任何结果文件只有真实生成且通过
provenance 校验后才能被索引或提交。

图必须只表达已经完成的证据。至少按 plan Task 14 输出相应已运行面板，并满足：

- `figures/README.md` 用中文逐图说明“展示什么、关注什么”；
- 更新现有 Figure Index；
- 没运行的 phase 标注 `not run by stop rule`，不能用空图假装完成；
- 可以做 Figure 4/5 风格的诊断图，但在 recovery 未建立前不能称
  paper-ready lifecycle figure；
- 图中明确区分 source-space、current-vSEEG primary、rate sensitivity、
  observed reference/null。

archive 路径按 plan 写入：

```text
docs/archive/topic4/sef_hfo/zm_minimal_carrier_branch_decision_2026-07-XX.md
```

使用实际日期替换 `XX`。

## 13. 最终汇报格式

最终用中文、白话、证据先行，必须包含：

### 一句话 verdict

直接给 exact terminal label，并说明它属于 state、carrier、trajectory、offset 还是
observation 哪一层。

### 测了什么 / 怎么测 / 揭示什么

三段式解释，不用工程术语掩盖科学含义。

### 分层完成度

分别列出：

- exact-resume/state gate；
- source-space carrier；
- observation-space carrier；
- entry；
- offset；
- recovery/lifecycle。

没有测到的写“未建立”，不能用前一层替代后一层。

### 能写 / 不能写

严格按当前证据给 manuscript-safe claim 和 forbidden claim。

### 核心数字与图

列 posterior carrier probability、lifetime、跨 phase/noise/seed 支持、readout
gate、entry/offset boundary 及 uncertainty；附真实图路径，并说明每张图的动力学
语义。

### 工程与复现

列 commits、测试、guard old/new SHA、config SHA、artifact provenance、资源峰值、
残留进程和 worktree 状态。

### 下一步

只能报告 branch verdict 所授权的下一机制方向。不要在本轮继续实现它。

## 14. 立即开始

现在直接执行：

1. 核对 branch、worktree、残留进程和 peer 状态；
2. 建立 plan checklist；
3. 从 Task 1 的失败测试和真实 state inventory 开始；
4. 按 gate 自主推进；
5. 任何 P0/terminal verdict 均按本 prompt 收尾，不等待一个“更漂亮”的正结果。

最重要的原则：

> 保留原始 Z/M、空间各向异性和虚拟电极语义；先判明 carrier 在哪里、慢轨迹是否
> 经过它，再讨论 entry/offset。工程上能运行不等于 ictal lifecycle 成立。
