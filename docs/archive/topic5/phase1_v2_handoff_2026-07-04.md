# Topic 5 V2 Phase-1-v2 Handoff — Candidate Scaffold Refinement

date 2026-07-04 · 状态：**plan + spec 已写、Phase-1 已验收、本阶段未执行**（下一个 agent 在干净 worktree 起）

---

## 1. 核心科学目标

Phase-1 已验收 = **cohort-level candidate early-ictal spatial recruitment scaffold**：间期 HFO-derived geometry 与发作 onset+0–20s 多频带能量场的对齐，在 cohort 层跨频段超过弱/全局空间 null（FWER 后 6/7 primary，唯 ripple_high 不过）+ order null（strong 子集），**NOT ripple-specific**。

**但 Phase-1 只证明了"存在一个 band-generic 空间共结构"，没回答决定这个 scaffold 该怎么解释的三个问题——这三个是必要下一步，不是可选补充：**

```
survive?  —— Gate B/C 后是否仍在？是 broadband / common-field recruitment 还是 residual frequency-specific layer？（1-f / 共场 / confound 未排）
who?      —— 哪些 subject 真有？cohort 6/7 是聚合（narrow 中位仅 2/7），是否少数 subject 驱动？
when?     —— preictal 已存在（静态解剖 scaffold）还是 onset 后才增强（onset-triggered recruitment）？（现只 onset 后 0–20s、clinical onset 可能不准）
```

**tier 天花板（pre-registered）= exploratory candidate scaffold refinement**（**不是 closure**——formal within-shaft Gate A 因 SEEG 杆几何 2/20 仍 unresolved，本阶段不解）。可升级的只是修饰语：`raw-only → residual-robust（W1）→ preictal-present / onset-triggered（W3）→ subject-heterogeneous（W2）candidate scaffold`。**不能**升级成 formal spatial-null positive / HFO-/LVFA-/ripple-specific / timing-order replay / criticality-proven / 机制。

---

## 2. 给执行 agent 的 prompt（可直接粘贴）

> 你在执行 Topic 5 V2 **Phase-1-v2（Candidate Scaffold Refinement）**。**目标不是升级 tier，是决定 Phase-1 已验收的 candidate scaffold 到底该怎么科学解释**——回答三个必要问题：survive?（Gate B/C 后是否仍在、broadband/common-field 还是 residual frequency-specific layer）· who?（哪些 subject 真有、cohort 是否少数驱动）· when?（preictal 已有还是 onset 后才现）。
>
> **按 plan 逐 task 执行**：`docs/superpowers/plans/2026-07-04-topic5-v2-phase1-v2-scaffold-refinement.md`（判据/锁定参数已全部写死）；判读框架见 spec `docs/superpowers/specs/2026-07-02-topic5-v2-phase1b-gate-closure-spec.md`（§EXP）；Phase-1 验收 + 数据/路径见 `docs/archive/topic5/v2_phase1_band_scan_backbone_2026-07-02.md`。用 `superpowers:subagent-driven-development` 逐 task + 两阶段 review，`superpowers:test-driven-development` 写代码。
>
> **执行顺序**：`W2 → W3-dev100/raw → W1-full1000 → W3-final`（W2 最快、不跑新 null 就暴露 subject 异质；W3-raw 快答 preictal 是否已有；W1 最重最承重）。科学写作里三者并列 P0/P0/P0-P1。
>
> **硬约束（LOCK）**：
> - **tier 天花板 = candidate scaffold refinement**；可升修饰语，不能升 formal spatial-null positive / HFO-/ripple-specific / timing replay / criticality / 机制。
> - **统计单位 = subject**（永不把 window 当独立样本；window→seizure→subject→cohort）。
> - **W3 主锚 = EEG onset**（clinical onset 只 sensitivity；两锚不一致以 EEG-onset 作生理主结论）；**W3 主 endpoint = band-generic scaffold score**（7 primary 带中位，per-band 作 descriptive）；**W3 null = subject-level sign-flip**（不用 window-label shuffle 作主 p）；pre-ictal 窗**豁免 ictal_fraction_min**（硬 gate 断言 pre 窗没被过滤没）。
> - **W2 输出连续 profile**（不止 n_sig：low/LVFA/HFA score、band_genericity_index、ripple_rank、n_positive_delta、方向一致性…）+ **三档 subject label**（strong ≥4/7 / directional n_pos≥5/7 / weak）。
> - **W1 Gate B/C outcome 语言已降级**："consistent with broadband/common-field account"（非"证明是 broadband"）；aperiodic 有 QC gate（fit fail<0.2 否则 Gate C 只 descriptive）；Outcome C（ripple 存活）须先过 sanity。
> - **长跑（W1 full-1000）必须用 resumable + setsid**（`run_topic5_v2_nulls.py` 的 `_partial_{feature}/` checkpoint + `setsid bash launcher < /dev/null &`，抗 session teardown——Phase-1 被 teardown 杀 3 次才根治；harness `run_in_background` 扛不过）。
> - 承重定性主张 → 数值阈值 gate（feedback_acceptance_gate）。
>
> **复用（不重写，DRY）**：`src/topic5_v2_band_scan.py` 纯函数 + `scripts/run_topic5_v2_{alignment,nulls,gates}.py`（已含 §2 cohort-perm + §EXP primary-family FWER + §3 strong-subset order + resumable checkpoint + `--feature {common_resid,aperiodic_resid}` + `--min-group`）+ `scripts/build_topic5_v2_{common_resid_cache,aperiodic_cache,confound_maps}.py`（残差 cache builder，只 smoke 过 139、需喂全 20）。band cache 已覆盖 onset−130s（W3 无需重建 cache）。图复用 `scripts/plot_topic5_v2_phase1_figures.py`。
>
> **起步**：开干净 worktree（`superpowers:using-git-worktrees`，off `topic5-v2-phase1`）；artifacts 写 `results/topic5_ictal_recruitment/v2_band_scan/`；队列 = narrow-20 / broad-17（yuquan xuxinyi/zhangkexuan 保留 `anchor=eeg_onset` 标注）。
>
> **禁措辞**：HFO-/LVFA-/ripple-specific · timing-order replay · formal Gate A passed · 超过任何空间随机场 · criticality-proven · 机制。

---

## 3. 路径

| 类别 | 路径 |
|---|---|
| **本阶段 plan** | `docs/superpowers/plans/2026-07-04-topic5-v2-phase1-v2-scaffold-refinement.md` |
| **spec（判读框架 §EXP + §0–§8 锁）** | `docs/superpowers/specs/2026-07-02-topic5-v2-phase1b-gate-closure-spec.md` |
| **Phase-1 验收 archive（数据/tier/路径索引）** | `docs/archive/topic5/v2_phase1_band_scan_backbone_2026-07-02.md` |
| **Phase-1 artifacts（n_perm=1000）** | `results/topic5_ictal_recruitment/v2_band_scan/{narrow,broad}/phase1_{gate,null_raw,alignment_raw}_*` |
| **核心代码** | `src/topic5_v2_band_scan.py` · `scripts/run_topic5_v2_{alignment,nulls,gates}.py` · `scripts/build_topic5_v2_{common_resid_cache,aperiodic_cache,confound_maps}.py` |
| **图脚本 / paper-ready Fig3-Sup1** | `scripts/plot_topic5_v2_phase1_figures.py` · `scripts/paper_figures/plot_fig3_sup1_multiband_field_alignment.py` · `results/paper-ready-figure/fig3_sup1_multiband_field_alignment/figures/` |
| **memory（agent 状态）** | `project_topic5_v2_phase1_accepted_2026-07-04.md` |
| **测试** | `tests/test_topic5_v2_band_scan.py`（34 绿）· `tests/test_topic5_v2_integration.py` · 新建 `tests/test_topic5_v2_phase1_v2.py` |

分支：Phase-1 全部已在用户 dir `topic5-v2-phase1`（HEAD 见 memory）。literal `main` 落后 86 commit（历史积压，未推进）。

---

## 4. 做图规范

**画前必读**：`docs/figure_style_guide.md` §0（硬规则）+ 参考成熟图。**流程 = render → 亲自目视（Read PNG）→ 改 → 再 render**，确认无误才 commit（feedback_figure_self_contained_paper_grade）。

**§0 硬规则（贯穿）**：
- 配色锁定：**顺序量 viridis**（除非 user 另指定——Phase-1 图按 user 要求 F1 用**红蓝 diverging**、心=0.5）；**带正负差值 diverging 红蓝**（0 居中）；**SOZ 黑环 overlay**（绝不作度量输入，图里写 "SOZ overlay only"）。不用 jet/rainbow。
- paper-grade 自洽：**无内部代号**（§X/cluster_id/PR-6/stable_k 不进面向读者的轴/图例/标题）；**紧坐标**；**一张图一套共享 legend/colorbar**；CJK 字符不进图（DejaVu Sans 渲染不出→变方框，用英文）。
- 多面板纪律（§7）：**一面板一个独立科学问题**，同构两角度=冗余删一个。

**Phase-1 三图 = 直接参考模板**（`scripts/plot_topic5_v2_phase1_figures.py`，已按 user 反馈定稿）：
- F1 观测热图：subject×band，**红蓝 diverging 蓝<0.5<红**、显著 cell 标**白星黑边**（path_effects.withStroke，任意底色可见）、**primary\|composite 黑虚线**、末行 cohort 中位带值、标题**一行**。
- F2 per-band null：**violin + 背景散点**（每点=1 subject）、**黑横条=cohort Δ**、**muted red `#c44e52`(过)/gray(n.s.)**、0 线细 gray（不加粗）、字体大（x/y label 12–15）、`*`=family-wise significant、标题**一行**。
- F3 per-subject caveat：横条=每 subject n_sig、色=null 强度档、median/≥4 参考线、legend 底部。

**paper-ready 约定**：
- 目录 `results/paper-ready-figure/<fig_name>/figures/`：PNG + **PDF** + `README.md`（中文逐图 `### filename` + 2–4 句 + 末行 `**关注点**：`）+ `figure_metadata.json`（含 `status` / `forbidden_language` / `panels` / `provenance{code,tests,data,archive,reproduce}`）。**这些都 gitignored**（只存盘），**只 track 画图脚本**（`scripts/paper_figures/plot_*.py`）。
- registry（tracked，须更新）：`results/paper-ready-figure/README.md` + `docs/main_figure_plan.md`。
- 非 paper-ready 的结论图目录：`results/.../figures/` 必须有 `figures/README.md`（中文关注点）；新目录 append `results/FIGURE_INDEX.md`。
- Fig3-Sup1（本 Phase-1 图）= Fig3-A field concordance 的 multi-band supplement（已建，见 §3 路径）；phase1-v2 的新图（W2 profile / W3 trajectory）若要 paper-ready，同 Fig3-Sup 系列命名。

**W3 时间轨迹主图建议**：x=相对 **EEG onset** 时间（−100→+20s）、y=band-generic scaffold score、cohort median ± subject IQR、5 个 time-bin 竖虚线（far/mid/near_pre / peri_onset / early_post）；per-band 轨迹放 supplementary。
