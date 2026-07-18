# Copy-ready prompt — autonomous 8-hour MZ early-field bridge run

你在 `/home/honglab/leijiaxin/HFOsp` 仓库工作。请自主执行最多 8 小时，不要等待我回复；遇到非破坏性歧义时按 spec 的保守口径继续，遇到 stop rule 时停止昂贵运行并完成诊断报告。最终只做本地提交，不 push、不 merge、不 rebase、不创建 PR。

## 唯一主目标

在 `codex/topic4-mz-slowvars` 分支上，完成 MZ 的直接空间 bridge：

```text
same-seed slow-off returning interictal events
-> held-out timing template
-> z-only delayed operational runaway
-> onset-locked early virtual-LFP/source activation field
-> template/energy association + spatial null + seeds 1/3/4 consistency
```

核心设计合同：

`docs/superpowers/specs/2026-07-19-topic4-mz-early-field-bridge-design.md`

先完整阅读该文件，再行动。它优先于同目录下两份 `state-conditioned-spatial-susceptibility` 草稿；后两份只可作为 Phase B 参考，今晚不能取代 direct bridge。

## 开始前必须核对

1. 阅读仓库根 `AGENTS.md`、`CLAUDE.md`、`docs/figure_style_guide.md` 的 Topic 4 小节。
2. 切换/确认工作目录：

   ```text
   /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-slowvars
   ```

3. 记录 `git status --short --branch`、`git rev-parse HEAD`、`git worktree list --porcelain`。
4. 当前预期 HEAD 为 `66a4d93`，并已有四份可能未跟踪的用户文档：

   ```text
   docs/superpowers/specs/2026-07-19-topic4-state-conditioned-spatial-susceptibility-design.md
   docs/superpowers/plans/2026-07-19-topic4-state-conditioned-spatial-susceptibility-implementation.md
   docs/superpowers/specs/2026-07-19-topic4-mz-early-field-bridge-design.md
   docs/superpowers/plans/2026-07-19-topic4-mz-early-field-bridge-8h-agent-prompt.md
   ```

   保留它们。不要覆盖、不删除、不把前两份当已实现代码。
5. 只读审计 `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-early-readout`。优先复用其中：

   ```text
   src/early_recruitment_readout.py
   tests/test_early_recruitment_readout.py
   scripts/run_topic4_m3_runaway_readout.py 中的 30--80 Hz contact timing/energy helpers
   scripts/paper_figures/plot_fig_topic4_early_recruitment_readout.py 的 Figure 5 绘图语法
   ```

   该 worktree 有大量未提交修改，只能读，不能改；不要整体复制其 manuscript/docs 变更。需要移植时只移植最小通用代码，并用 `apply_patch` 在 MZ worktree 落盘。
6. 回看真实输入：

   ```text
   results/topic4_sef_hfo/mz_slowvars/per_seed/multiseed_summary.json
   results/topic4_sef_hfo/mz_slowvars/calibration.json
   results/topic4_sef_hfo/mz_slowvars/p3_candidates.json
   scripts/run_topic4_mz_slowvars.py
   src/snn_engine/mz_slow_vars.py
   ```

7. 在日志和最终报告中显式 quarantine Arm C：nominal 3x3 实际只有两个 unique z 配置、m 三档重复；禁止消费 `9/9`、`weak/mid/strong` 或 dose-response 叙述。

## 时间与优先级

总时长上限 8 小时；hour 6.5 后不得再启动新的昂贵仿真，最后 90 分钟留给出图、眼检、测试、报告和 git 审计。

```text
P0  0:00--1:00  preflight、复用审计、测试与 artifact schema
P1  1:00--2:15  fixed-bar detector + MZ bridge producer + unit tests
P2  2:15--5:15  q75/tz5 seeds 1,3,4 slow-off + native runs（最多并行2个）
P3  5:15--6:15  templates、early fields、maxAB/null、cohort aggregation
P4  6:15--7:15  diagnostic figures + original/high-detail visual QA + 中文 README
P5  7:15--8:00  targeted tests、STATUS、archive report、local commits、handoff
```

如果已有 artifact 可验证复用，不要无意义重跑。实现必须支持 `--resume`，逐 seed 原子写出；失败 seed 留下带错误与 provenance 的 artifact 后继续其他 seed。

## P0/P1：最小实现范围

新增命名建议：

```text
config/topic4_mz_early_field_bridge.yaml
src/topic4_mz_early_field_bridge.py
scripts/run_topic4_mz_early_field_bridge.py
scripts/plot_topic4_mz_early_field_bridge.py
tests/test_topic4_mz_early_field_bridge.py
```

不要大重构 MZ runner。优先将纯函数放在 `src/`，runner 只做 I/O、仿真调度和 provenance。

必须测试：

1. slow-off `event_bar_seed` 被 target 原样复用，target 自身 max 不改变 bar；
2. incomplete early window fail-closed；
3. missing contacts 不补值，matched support 正确；
4. odd/even train-held-out split 无事件泄漏；
5. A/B 两模板的 `maxAB` null 在每次 permutation 内重算 max；
6. within-shaft permutation 保持 shaft membership；
7. source-grid toroidal shift 排除零位移并保留 field multiset；
8. onset component 必须包含既有 `t120`；
9. output JSON 可序列化 NaN 为 null 或显式 status；
10. `--resume` 不覆盖已完成且 provenance 匹配的 seed。

可以选择性复用 early-readout 的 generic arrival/energy/Spearman/permutation 代码，不要另写一套语义相近但合同不同的函数。

## P2：注册运行

只跑主候选：

```text
zA_q75_tz5000
seeds 1,3,4
T <= 15000 ms
```

每 seed 跑：

1. matched slow-off + LFP recorder；
2. native z-only delayed-runaway + LFP recorder；
3. 相同 seed、相同 scaffold、相同 observation geometry；
4. slow-off 一次冻结 `event_bar_seed`，native 必须复用；
5. 尽量使用现有 early-stop，但必须保留 `t_recruit+100 ms` 完整窗口；若不完整则 fail-closed，不得用截断窗；
6. 不保存全程 N×T 的 z/m；只保存 bridge 所需派生量和固定 landmark（如实现了 snapshot）；
7. 不跑 q50 sensitivity，除非三 seed 主候选、所有 P3/P4/P5 必交付已完成且剩余超过 90 分钟。

运行前做一个短 smoke，确认 LFP shape、contact ordering、内存和预计 wall time。最多并行两个 full-density SNN，避免 RAM 过载。

## P3：科学读出

严格按 spec 实现：

- primary interictal templates 来自 same-seed slow-off returning events；
- 30--80 Hz contact burst-envelope latency；
- A_to_B / B_to_A 固定方向；
- chronological odd/even held-out；
- `t120` + slow-off-derived `t_recruit`；
- primary 0--50 ms，sensitivities 0--25、25--50、50--100、0--100 ms；
- contact virtual-LFP early-energy + 24x24 source-grid activation field；
- signed Spearman、cosine、early-vs-late quartile、top-k、support、dynamic range、recruited area；
- all-support 与 direct-core-excluded 两套结果；core-excluded support 不足时 fail-closed；
- 每个 contact 周围 1.5 mm 的真实 E-neuron participation audit，禁止把 contact peak 直接写成 local tissue recruitment；
- within-shaft maxAB null、unrestricted contact null、source-grid toroidal-shift maxAB null；
- per seed 结果 + three-seed median/range/sign count，不做 n=3 cohort p-value。

z-only 连续轨迹中的 pre-runaway returning events只做 secondary within-trajectory audit；不足时写 `insufficient_support`，不要用目标 energy 挑事件。

任何负结果、反号、degenerate field、source/contact 不一致都照实输出，不调窗口、不换 seed、不换主候选。

## P4：图

输出：

```text
results/topic4_sef_hfo/mz_early_field_bridge/figures/mz_early_field_bridge_seed1.png
results/topic4_sef_hfo/mz_early_field_bridge/figures/mz_early_field_bridge_multiseed.png
```

seed1 图复用现有 Figure 5 field grammar，但必须诚实显示 trajectory provenance。若 timing event 来自同一 native z-only trajectory，可画单条 continuous trace；若 primary template 来自 matched slow-off，则必须把 slow-off event 与 native runaway 画成两个明确标注的 trace strip，禁止把 slow-off window 阴影画在 native trace 上。图中必须写 `operational runaway` 和 `MZ diagnostic`，不得覆盖：

```text
results/paper-ready-figure/fig5_snn_state_readout/
```

multiseed 图显示每 seed 的 held-out template reproducibility、contact/source effect、fixed windows、null 和 eligibility。未运行/不合格必须可见。

生成后逐张以 original/high detail 打开眼检：contact 顺序、轴方向、两个 field extent、颜色方向、onset/window、seed 曲线、空 panel、clipping、字体和 colorbar。眼检后再写中文 `figures/README.md`，每张图用 `### filename` 开头，2--4 句说明，最后一行 `**关注点**：`。

## 可选项：只有全部主交付完成后

如果 hour 6 前三 seed direct bridge 已完整，且至少还剩 90 分钟，可做以下一项，按顺序：

1. 在 `t_recruit-100 ms`、`t_recruit`、`t120` 保存 z spatial snapshots，计算 spec 的 `global_fraction/local_fraction`；它只是诊断 readout，不进入方程。
2. 跑 q50/tz10 的 seed 1 sensitivity。
3. 只为一个 selected seed/state 构建精简 M3B projected-propagator smoke。

若做第 3 项，必须使用：

```text
K_T = P_rE exp(JT) E_rE
```

并冻结全部 operator 参数；若用 probe dictionary，先正交化。不得把 leading eigenmode 写成 axial，不得把 surrogate z->q mapping 写成已标定生物映射。

今晚禁止：

- broad state-conditioned atlas；
- 全 5 states × 全 controls × 全 horizons；
- exact snapshot/resume engine surgery；
- native/uniform/shuffle/reset 的“state-matched”因果宣称；
- 新 slow variable 或全局分母；
- 修复版 z+m sweep；
- 40 s acceptance；
- 为了漂亮结果重选窗口/方向/seed/candidate。

## 输出与报告

主输出根：

```text
results/topic4_sef_hfo/mz_early_field_bridge/
```

同时写：

```text
results/topic4_sef_hfo/mz_early_field_bridge/STATUS.md
docs/archive/topic4/sef_hfo/mz_early_field_bridge_2026-07-19.md
```

报告必须逐条回答：

1. fixed slow-off bar 是否真正跨状态复用？
2. 每 seed 有多少 A/B training/held-out events，held-out template 是否可复现？
3. `t_recruit` 与 `t120` 各是多少，窗口是否完整？
4. 0--50 ms contact/source fields 是否有足够 support 和 dynamic range？
5. 每 seed `rho_A/rho_B/rho_maxAB`、quartile contrast、top-k 和 null 是什么？
6. 结果是否跨 seed 同号；contact 与 source 是否矛盾？
7. pre-runaway within-trajectory event audit 是否 eligible？
8. 是否只能支持 observation-layer bridge，还是已到 mechanism layer？
9. 哪些 optional 为 completed/failed/not_run？
10. 最大科学缺口和唯一下一步是什么？

措辞边界：只写 `operational runaway/runoff`，不写 clinical seizure；`z_i` 是 phenomenological inhibitory-efficacy variable；virtual-LFP energy 是 model proxy；single-model three-seed 不是 cohort validation。

## 验证与提交

至少运行：

```bash
pytest -q tests/test_mz_slow_vars.py tests/test_topic4_mz_slowvars.py tests/test_early_recruitment_readout.py tests/test_topic4_mz_early_field_bridge.py
git diff --check
git status --short
git diff --stat
git worktree list --porcelain
```

若某个复用测试在 MZ 分支不存在，先移植它对应的最小通用模块与测试，不要用跳过掩盖。

允许按逻辑本地提交：

1. `feat(topic4): add fixed-bar MZ early-field bridge`
2. `feat(topic4): add multiseed MZ bridge diagnostics`
3. `docs(topic4): report MZ early-field bridge`

每次提交前检查精确 staged paths。不要提交 early-readout worktree 的文件，不要 stage 无关 dirty files，不要 push。

## 最终 handoff

用中文、证据优先，按下列格式返回：

```text
一句话判断
完成层级：engineering / numerical / scientific / bridge
三 seed 主结果与 denominator
contact/source/null 结果
Figure 5-compatible diagnostic 是否成功生成
tests 精确计数
代码、artifact、figure、README、report 路径
local commits
failed/not_run 及原因
最大科学缺口
唯一下一步
最终 git/worktree 状态
```

不要只说“运行成功”。即使结果为负，也要完成注册分析和可复现报告。
