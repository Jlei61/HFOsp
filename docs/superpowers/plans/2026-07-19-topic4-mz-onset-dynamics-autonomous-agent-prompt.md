# Copy-ready prompt: autonomously execute the Topic 4 MZ early-onset dynamics analysis

Copy the block below into a fresh Codex agent after the active MZ early-field and susceptibility agents have either committed their work or published stable artifacts.

---

你现在负责 Topic 4 的 **MZ 同行版早期启动动力学**。不要停留在计划、代码 smoke test 或一张直观图；你的任务是自主完成从 MZ 慢状态、冻结快系统、非线性点火边界、z 因果反事实、间期事件干预，到 m push-pull 检验的完整、可审计分析。

工作仓库：

```text
/home/honglab/leijiaxin/HFOsp
```

MZ 上游 worktree：

```text
/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-slowvars
```

唯一承重设计合同：

```text
/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-slowvars/docs/superpowers/specs/2026-07-19-topic4-mz-onset-dynamics-phase-portrait-design.md
```

先完整读完该 spec，再执行。不要自行缩成“画一个 z-m 轨迹”或“再扫一轮参数”。

## 核心科学问题

我们要检验的不是“模型会不会 runaway”，而是：

```text
反复可恢复的间期事件
  -> 是否推动 z_i 耗竭 / 有效抑制下降
  -> 是否让同一个 E1146 scaffold 的轴向有限时易感性增强
  -> 是否让全 SNN 的非线性点火阈值下降
  -> 是否使下一次内源波动/事件逃出间期 basin
  -> operational-runaway 早期场是否保留间期模板次序
  -> m_i 是否能提高点火阈值、限制或恢复该状态
```

必须区分：

1. slow-state projected trajectory；
2. frozen fast-system eigenvalue/stability；
3. non-normal finite-time axial gain；
4. full-SNN nonlinear ignition threshold；
5. contact/source-space field bridge。

不能用其中一层替代另一层。

## 第一步：工作区与上游审计

1. 运行并记录：

   ```bash
   git worktree list --porcelain
   git -C /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-slowvars status --short
   git -C /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-slowvars log -8 --oneline --decorate
   ```

2. 当前 MZ worktree 可能仍有两个 agent 的未提交修改：

   - early-field bridge；
   - state-conditioned susceptibility / snapshot observer。

   不得覆盖、移动、stage 或提交它们的 dirty files。

3. 如果上游已经提交：从最新已提交 MZ base 创建独立分支和 worktree：

   ```text
   branch: codex/topic4-mz-onset-dynamics
   worktree: /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-onset-dynamics
   ```

4. 如果上游仍在活动：先做只读 artifact/code 审计和测试基线；不要在同一 dirty 文件上写。等上游形成 commit 后再建独立 worktree。若无法等到稳定 base，把状态写为 `upstream_active`，列出重叠文件和已完成的只读工作，不得偷取未提交修改。

5. 创建独立 worktree 后，只提交本任务文件；不 push、不 merge，不改 Topic 5、论文 Methods 或 early-readout worktree。

## 必须先读的文件

除根目录 `AGENTS.md` 外，按顺序读：

```text
docs/superpowers/specs/2026-07-19-topic4-mz-onset-dynamics-phase-portrait-design.md
docs/superpowers/specs/2026-07-18-topic4-mz-per-neuron-slowvars-design.md
docs/archive/topic4/sef_hfo/mz_slowvars_discovery_2026-07-18.md
docs/superpowers/specs/2026-07-19-topic4-mz-early-field-bridge-design.md
docs/superpowers/specs/2026-07-19-topic4-state-conditioned-spatial-susceptibility-design.md
docs/superpowers/specs/2026-06-27-sef-hfo-m3b-spectral-phase-map-design.md
docs/figure_style_guide.md
```

然后核对真实代码和 artifact，不要只相信文档摘要：

```text
src/snn_engine/mz_slow_vars.py
src/snn_engine/kick_probe.py
src/topic4_mz_slowvars.py
src/topic4_m3b_spectral_phase.py
scripts/run_topic4_mz_slowvars.py
results/topic4_sef_hfo/mz_slowvars/
results/topic4_sef_hfo/mz_early_field_bridge/                  # 若已存在
results/topic4_sef_hfo/state_conditioned_susceptibility/       # 也检查 spec 里的 canonical root
```

artifact 路径若和 spec 不一致，以真实生产者、provenance 和 STATUS 为准，并把差异写入报告。

## 执行原则

- 自主推进，不要每一步都向用户确认。
- 先 cheap smoke、测试和资源审计，再跑 full-density。
- 使用 per-seed/per-state 可恢复产物；重复运行必须幂等。
- 结果方向不是 gate：正、负、反向、seed-inconsistent、right-censored、unresolved 都要完成注册分析并如实报告。
- 不允许看见结果后换 seed、换时窗、换方向、扩大 perturbation 上限或调参数救结果。
- 不把 engineering green、artifact existence、analysis eligibility 和 scientific support 混为一层。
- 需要运行较长任务时持续推进；不要因为第一轮结果不漂亮就停在半成品。
- 只在真实 P0 阻断时停止：关键上游 artifact 缺失、native replay 不一致、数值操作点全 unresolved、或资源无法安全运行。阻断时先穷尽 spec 内 fallback，再报告。

## 实现与分析任务

### A. 状态坐标和 current-aware mapping

实现 spec §5：

- `D_z = 1 - mean(z)`；
- `A = eta_m * mean(m)`；
- all-E、source core、sink core、axis corridor、off-axis、core-excluded；
- 注册状态前 20 ms 内的 `q_eff = sum(z*I_I)/sum(I_I)`；
- `p_deplete = P(I_I >= I_th)`；
- `z_bar` versus `q_eff` 映射审计。

observer 只能积累注册窗口，不得保存 `N x T` 电流矩阵。`q_eff` 是 rate-field 的主 inhibition scale；`z_bar` 是 slow-state 坐标，不要二选一挑更好看的。

### B. 自然轨迹和 projected slow flow

主轨迹 `zA_q75_tz5000`，seeds 1/3/4；敏感性 `zA_q50_tz10000`。

注册 baseline/mid/pre500/pre100/onset 及每个 eligible returning event 的 event-pre/event-post。计算事件触发的 `Delta D`、`Delta q_eff`、`Delta A`。

按 spec 的独立访问数、跨 seed 支持和 sign-agreement 规则决定哪些 `(D,A)` bin 能画 drift arrow/nullcline。支持不足时必须写 `Projected MZ state trajectories`，不能伪装成完整 phase portrait。

### C. 冻结快系统 phase diagram

复用 M3B finite heterogeneous Jacobian，不另写一个不一致的 surrogate。完成：

- 每个自然状态的 operating point；
- `alpha_1`、频率、谱隙、globality、core overlap；
- `G_axis`、`G_perp`、global gain，T=10/30/50 ms，75 ms sensitivity；
- projected propagator `K_T = P_rE exp(JT) E_rE`；
- controlled `(D,A)` grid；
- uniform/rotate/shuffle/z-blocked controls；
- n=8 numerical audit，n=12 final。

始终分开 `leading eigenmode` 和 `non-normal finite-time response`。若 upstream susceptibility 已经完整，复用并验证其 artifact，不重复造轮子；只补本 spec 缺的 current-aware mapping、主轨迹或 MC plane。

### D. 全 SNN 非线性点火阈值

不要为了 state fork 修改受保护的 `kick_probe.py`。采用从 t=0 同 seed 完整 replay，在注册 step 通过 MZ slow object 的 off-by-default schedule 分支：

- 分支前 spike/rate/z/m/current summaries 必须 bit-identical；
- 分支时 freeze z/m；
- 用 `threshold()` hook 对 source、sink、off-axis E target 施加确定性 10-ms threshold-lowering probe；
- amplitude ladder 使用 spec 注册值 `[0,0.025,0.05,0.10,0.20]` threshold-gap units；
- 有 bracket 时固定做两次二分；
- 500-ms horizon；
- 仍用锁定的 120-Hz/100-ms operational-runaway detector；
- 零 probe 已 runaway -> `epsilon_c=0`；最大 probe 未点燃 -> right-censored。

完成 baseline/mid/pre500/pre100 × source/sink/off-axis × seeds 1/3/4。不得只跑 seed1 或只展示最强方向。

### E. z 因果反事实

在 pre500/pre100 做：

- native dynamic；
- native frozen；
- uniform current-matched；
- fixed-seed spatial shuffle；
- reset z=1；
- rotated-90（映射有效时）。

每个分支跑 zero probe 和邻近 `epsilon_c` 的注册 source probe。比较 onset/threshold/early spatial response。回答 global mean 与 spatial pattern 各自贡献什么；不要用普通相关代替 state-matched counterfactual。

### F. 间期事件抑制

使用 fixed slow-off event bar。按时间顺序选择主轨迹中 onset 前最后三个、且结束至少早于 onset 200 ms 的 returning events。

只在 slow-off 上标定一个固定 inhibitory threshold pulse；锁 amplitude 后再进 target trajectory。对每个事件做 no-pulse、event-suppression、quiet-time sham。测 event removal、`Delta D/q_eff`、下一事件、onset shift、下一状态的 `epsilon_c`。

如果无法在不全局压死网络的前提下抑制事件，写 `event_suppression_unresolved`。没有这一层，禁止写“间期事件因果性耗竭 z 并触发转换”。

### G. focused m push-pull

原 Arm C nominal 3x3 不得使用。完成 spec §10 的 focused grid：

```text
z = [zA_q75_tz10000, zA_q75_tz5000]
tau_adp = 2000 ms
target A/I_EE_scale = [0,0.025,0.05,0.10,0.20]
seeds = [1,3,4]
```

eta_m 只能从现有 slow-off calibration recipe 推导。先验证 realized adaptation levels 真正不同。报告 m 是提高点火阈值、全局 suppress、产生 bounded/retriggerable state，还是无稳健效果。

### H. field bridge integration

若 upstream early-field bridge 数值 eligible，把 held-out template-to-early-field 结果按 seed/state 与 `D/q_eff`、`alpha_1`、`G_axis/G_perp`、`epsilon_c` 和 counterfactual 对齐。

最终表格必须分别回答五个问题，不给总 PASS：

1. D/q_eff 是否增加；
2. 轴向 finite-time gain 是否增强/保持；
3. epsilon_c 是否下降；
4. early field 是否与 held-out interictal template 一致；
5. z 反事实是否改变相应结果。

## 输出合同

严格写入：

```text
results/topic4_sef_hfo/mz_onset_dynamics/
```

代码：

```text
config/topic4_mz_onset_dynamics.yaml
src/topic4_mz_onset_dynamics.py
scripts/run_topic4_mz_onset_dynamics.py
scripts/plot_topic4_mz_onset_dynamics.py
tests/test_topic4_mz_onset_dynamics.py
```

报告：

```text
results/topic4_sef_hfo/mz_onset_dynamics/STATUS.md
docs/archive/topic4/sef_hfo/mz_onset_dynamics_2026-07-19.md
```

每个 JSON 写 schema、upstream paths/hashes、git SHA、engine/config hash、candidate/seed/state、命令、stage status 和 censoring。所有中间大矩阵放 `arrays/`，per-seed 放 `per_seed/`，不要散落。

## 测试与资源

先运行 MZ、M3B 和 upstream 相关测试，记录精确命令/计数。必须补齐 spec §14 的 parity、pre-branch identity、freeze、counterfactual invariant、q_eff、operator residual、ignition censoring、phase-arrow eligibility、resume/idempotency 测试。

先短时/低密度 smoke，测 wall time 和 peak RSS，再决定 full-density 并发。长任务使用阶段化、可恢复输出；不要在内存未知时盲目开多个 32k-neuron worker。

## 图形

阅读 `docs/figure_style_guide.md`。主图必须逐 panel 有明确 argument：

1. continuous rate + D/A trajectory；
2. projected `(D,A)` diagram；
3. alpha/gain/epsilon_c；
4. z counterfactual；
5. event intervention + field bridge。

不要画装饰性机制图，不要用大公式占 panel，不要只画 seed1，不要 PASS/FAIL 印章。完整 atlas 放 diagnostic/supplement。PNG/PDF 都生成；眼检后再写中文 `figures/README.md`。

## 结果口径

可以写：linear crossing、finite-amplitude escape、axis-selective susceptibility、uniform global amplification、state-invariant、seed-inconsistent、unresolved，以及各 counterfactual/event/m 结果。

禁止写：模型复现临床发作、runaway 是完整 seizure、z 证明氯积累/GABA failure、eigenvalue crossing 单独证明发作、field correlation 证明因果、原 Arm C 是 graded interaction、单 seed 支撑 bridge、没有稳定 returned 状态却称终止机制成立。

## 自主交付要求

不要在“代码写完”“测试通过”或“图生成”时提前结束。持续推进到 spec §18 的 definition of done，或遇到 spec 定义的真实阻断。

完成时给出：

1. 一句话科学判断；
2. 分层完成度，不合并成总 PASS；
3. 每个核心问题的真实结果与边界；
4. 最大科学缺口；
5. 测试、运行命令、资源使用；
6. 所有 canonical artifact/figure/report 路径；
7. 逐图视觉 QA；
8. commits 和最终 `git status --short`；
9. 明确未运行、失败、right-censored、unresolved 项；
10. 不 push、不 merge。

按逻辑批次提交本任务 owned files，避免 `git add -A` 吞入其他 agent 的文件。建议批次：

```text
1. state observer/intervention infrastructure + tests
2. phase/susceptibility/ignition analysis + tests
3. full runs + artifacts + figures
4. STATUS/archive report/provenance
```

从现在开始执行，不要再给我一份新的散文计划。

---

