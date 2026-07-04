# Topic 5 V3p 实现 — Agent Handoff Prompt

> 交给一个**长时执行 agent**:在隔离 worktree 里按 spec+plan 逐 task 实现 Topic 5 V3p,跑完 n_perm=1000,出图。下面整段可直接作为 agent 的起始 prompt。

---

## 你的任务（一句话）
在隔离 worktree `topic5-v3p-preictal-trajectory`（已建好）里,按已锁的 spec+plan **逐 task TDD 实现 Topic 5 V3p（发作前非轴向轨迹）**,跑完最终 `n_perm=1000`,出 paper-grade 图。用 **superpowers:subagent-driven-development**（每 task 新 subagent + 两段式 review + commit）。

## 核心科学目标（朴素话,先读这段建立直觉）
病人两次发作之间,短暂高频异常放电按一条固定先后顺序在电极间传开——一条走熟的小路（间期传播轴 `G_HFO`）。V3p **只看发作真正开始之前的两分钟**（`P0..P3` = −120~−10s,按每次发作的电生理 onset 锚定,**完全不碰发作本身**),问:小路**之外**的触点（non-axis strict）是不是随发作临近**逐渐爬升**（Theil-Sen 斜率>0）出现——
- **（承重 co-primary）** 非轴向连锁流 `net_offaxis_flux`（H3p-b）+ 最易放大方向 `mode_shift_density`（H3p-c）;
- 且这爬升**专门集中在真非轴向触点**——由**同杆 label-null**裁判（把"哪些算小路外"在杆内打乱重算整条斜率;全场一起升温则打乱后斜率一样大→不显著,只有专门压真非轴向才超 null）。

全程**探索性**、**无发作预测/forecasting 主张**（是趋势描述+null 裁判,不是预测器）;**预注册阴性可接受**（反而加固 Topic 5 主线"间期轴=共享粗骨架、非发作特异 replay"）;**不救个别被试**（1125 只作描述性 case,不升队列主张）。

## 和 V3a 的衔接
- **V3a**（分支 `topic5-v3a-mode-transition`,已实现+跑完 n_perm=1000 → tier 2 偏阴性）问的是"发作后 `I1`（+10~+30s）对比 `P3`（−30~−10s）"两时刻差。命门:`I1` 落在发作**已点着之后**（信噪比塌/饱和/全场淹没）→ 终点脏。**V3p 把终点从"发作后"挪走、改看纯 preictal 段的斜率**——同一科学问题（轴向→非轴向搬迁）的**更干净一枪**。
- V3p **read-only 继承** V3a 全部纯数学机制:`src/topic5_v3_mode_transition.py`（相位窗 `phase_bin_range`/几何 `classify_contacts`+`P_A/P_N`/三 null/avalanche `atm_offdiag`+`net_offaxis_flux`/dynamics `lowrank_var`+`subspace_mode_shift`…）+ `scripts/_topic5_v3_io.py`（`classify_subject_contacts`/`load_subject_phase_envelopes`）。**只加新文件 `src/topic5_v3p_*`+`scripts/run_topic5_v3p_*`+tests;绝不 edit 任何 V3a/V2 文件**（V3a 分支还在收尾,保合并干净）。

## Spec / Plan 路径（source of truth — 每 step boundary 重读对应节,别凭记忆）
- **spec（rev2）**:`docs/superpowers/specs/2026-07-03-topic5-v3p-preictal-trajectory-design.md`
- **plan（rev2,11 task）**:`docs/superpowers/plans/2026-07-03-topic5-v3p-preictal-trajectory.md`
- 继承的 V3a spec/plan（参照,不改）:`docs/superpowers/{specs,plans}/2026-07-02-topic5-v3a-mode-transition*`

## 执行纪律（承重,别漏）
- **Task 顺序 0→1(gate)→2→3→4→5→6→7→8→9→10。** **Task 1 feasibility 是硬门**:(a) lock `min_windows_for_slope`（实测 ~17-18/sz 非约束）;(b) **axis-quality 门校准**——确认 `axis_quality_gate_pass=True` 覆盖全部策展 roster,否则**放宽阈值到不误杀 roster**;(c) 锁 `admitted` 扩展集（`broad_expanded=broad_core∪admitted_epilepsiae`,admitted yuquan 进独立补充）。**narrow <4 qualify → STOP + 报告。**
- **CLAUDE.md §5/§6**:每写函数体**前**重读 plan 对应节的多子句不变量（边界参数传播 / paired-cohort key match / surrogate 构造每一子句 / "reported alongside" 二级字段 / 复用 helper 须 question-match 非 signature-match）。承重 rev1/rev2 合同:**双 span（full+guard）须同过** / **H3p-b HARD rate（逐窗）+lag0（lag1_specific>0）** / **H3p-c HARD phase+block（strong vs weak grade）** / **label-null 主裁 + 回归残差只作 sensitivity（`slope_resid≈0` 不推翻 label-null）** / **gain_shift=slope(gain_nonaxis−gain_axis)** / **N_self_sustain lag1-specific** / **label_null_underpowered 出强阳性分母** / **broad_core 始终并列,tier 4 须 broad_core 同向** / **yuquan 永不 pool**。
- **CLAUDE.md §8**:任何面向用户的 status/recap 先用朴素话讲"测了什么/怎么测/揭示什么",代号只作括号补注。
- **判读**:tier 只在 Task 9 summary 判;`state_v3p_supported=tier≥3`;V3p 最高 tier 4。EXPLORATORY 全程,禁 forecasting 措辞。

## Worktree / 数据 / 运行
- worktree `.worktrees/topic5-v3p-preictal-trajectory`（off V3a HEAD `ac042f3`）已建好,**5 个 gitignored 数据 symlink 已接好**:`results/topic5_ictal_recruitment/ictal_field_long_cache`、`results/interictal_propagation_masked_broad`、`results/spatial_modulation/propagation_geometry_broad`、`results/interictal_propagation_masked/rank_displacement/per_subject`、`results/spatial_modulation/propagation_geometry/observation_readout/real_subjects`。新增 subdir 若读别的 gitignored 数据,照 V3a worktree `readlink` 补 symlink。
- **cohort 实测干净**:broad 9/9 + narrow 7/7 geometry OK、每发作 ~17-18 preictal 窗。integration subject = **253**。
- **new-file commit 用 `git add <files>`**（非 `-am`）;real-data 脚本 `@pytest.mark.integration`+`--outdir`。
- final run:`for ax in narrow broad; do run feasibility/trajectory(--n-perm 1000)/summary; done; plot; pytest 全绿`。narrow 先跑。

## 画图规范（重要 —— 保持和 V3a 一致,但样式未锁）
1. **先读 `docs/figure_style_guide.md` §0 全局硬规则**:paper-grade **自包含**（无 `§X`/`cluster_id`/括号轴标;共享图例;紧坐标;**render→肉眼看→fix 再 commit**）;每个含图目录**必须**配中文 `figures/README.md`（`### filename` + 2-4 句 + 末行 `**关注点**：`）;**append `results/FIGURE_INDEX.md`**。
2. **画图风格直接对照 V3a**:把 V3a 的 `scripts/plot_topic5_v3_summary.py`（在 `topic5-v3a-mode-transition` worktree,产 `v3_mode_transition_{trajectory,summary}.png`）当**样式参照**,产出同一 look（配色/布局/字号/panel 组织)。
3. **⚠️ 样式尚未定稿**:`figure_style_guide.md` §Topic 5 明说 Topic 5 canonical 图型"暂不锁定、探索期、按个案、不强制统一布局";用户口径="现在还在调整,最后会落下来"。所以——**产出 V3a-一致的图即可,不要过度打磨 / 自创新布局 / 反复抠像素**,预期后续有一次统一 restyle 收口。**图的科学内容比样式更重要**:CLAUDE.md §7,每 panel 答一个**独立**科学问题,冗余 panel 砍掉。
4. **V3p Task 10 目标图（2–3 panel)**:(A) per-subject co-primary surplus 斜率——`net_offaxis_flux_surplus_slope` + `mode_shift_density_surplus_slope`,**narrow vs broad_expanded**（broad_core 子集标出）,零线,label-null 显著被试标记;(B) preictal 相位轨迹 `P0→P3`（`mode_shift_density` + `net_offaxis_flux`,均值±IQR,来自 window_detail);(C 可选) per-subject `slope_label_z`（±1.96 参考线)。输出目录 `results/topic5_ictal_recruitment/v3p_preictal_trajectory/{narrow,broad}/figures/`。

## 完成标准
全 11 task commit + `pytest tests/test_topic5_v3p_preictal_trajectory.py -v` 全绿 + `pytest -m integration tests/test_topic5_v3p_integration.py -v` 全绿 + final `n_perm=1000` 跑完两队列 + Task 9 summary tier + Task 10 图（肉眼过）+ `figures/README.md` + `FIGURE_INDEX.md`。跑完用**朴素话**汇报 tier 结果 + honest recap（**探索性别写成结论**;若阴性,写成"数据不支持非轴向搬迁 → 加固共享粗骨架主线";若阳性,写清 strong/weak grade + 两 span + label-null 是否 underpowered）。Task 1 gate 结果、任何 STOP、任何偏离 spec 的决定 → 先报告再继续。
