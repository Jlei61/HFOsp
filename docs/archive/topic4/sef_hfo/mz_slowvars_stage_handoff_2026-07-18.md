# M4-MZ 阶段 handoff（2026-07-18）—— 间期 + runoff 已达成（对标 M2 门槛）

> tier = **mechanism screen（探索性）**。所有 phenotype 是**检出标签**，不是发作主张。
> 分支 `codex/topic4-mz-slowvars` @ `.worktrees/topic4-mz-slowvars`（从 main `2d01634`）。**不 push、不动 main、不删旧 M3/M4 结果。**
> 本文件是**阶段 handoff**（可续），不是最终 report；最终 report 待 P3 多 seed + 图收口后写入 `mz_slowvars_discovery_2026-07-18.md`。

## 0. 本阶段的门槛与结论（朴素话）

**门槛（对标 M2，用户 2026-07-18 定）**：这一轮**不要求**"有界可恢复的临界招募态"（`expanded_returned`）。达标条件是模型能同时给出**间期基线**（安静、稀疏、可自恢复的事件）和 **runoff（失控/runaway）**两个 regime——和 M2 被评估的那个层级一致。

**结论：达标（YES），且经 P3 多 seed 复核 seed-robust。**
- **间期基线**：slow-off（两慢变量都关）在 seed 1/3/4 都有稀疏可自恢复间期事件（标定 T=8s 时 20/22/20；P3 T=15s 时 38/40/39），baseline-anchor **PASS 跨 3 seed**。不是均质 sheet 的全或无。
- **runoff/runaway**：只开 z（去抑制）seed=1 有 5/9 失控；失控起始随去抑制**强度/速度**单调延后（q50 强耗竭 1.7s → q75 弱耗竭 9.3s）。**P3 复核：两个失控格在 seed 1/3/4 全 3/3 失控、起始近乎一致**（q50_tz10000 ~4.7–4.9s、q75_tz5000 ~9.3–9.8s）——runoff 跨 seed 稳健。
- **有界扩大态（超出本轮门槛，不作要求）= seed-脆弱**：唯一那格 `zA_q75_tz10000` 只在 seed=1 有界扩大，seed 3/4 退回 interictal_like（1/3）——不是稳健 regime，**不作 bounded 主张**。

即：M4-MZ 复现了"间期 + 失控"两 regime 且跨 seed 稳健（M2 层级达标）；更高的"有界可恢复临界态"未稳健出现（本轮本就不要求）。

## 1. 测了什么 / 怎么测

只对**兴奋（E）神经元**加两个慢变量，其它慢变量（`q_I`/`g_K`/`S_G`/a-shunt/STD）全关：
- `z_i`（抑制效能）：被持续强抑制轰炸 → 有效抑制变弱 → 事件沿病人轴摊大（**push**）。
- `m_i`（放电适应）：放电累积适应电流 → 刹车 → 限制高招募、可能恢复（**pull**）。

同一 E1146 衬底（narrow / template_source / L=20 / dens=100 / AR=2 / 双低阈值核 / 真实电极注册），四臂：slow-off / z-only / m-only / z+m。自发（无 kick），每臂同 seed 同噪声。判读用 slow-off 同 seed 事件分布做基线，7 类标签。

## 2. discovery 结果（seed=1，T=12s，27 格）

phenotype counts：`{runaway:5, expanded_bounded:1, interictal_like:15, suppress:6}`（0 `expanded_returned`）。

| 臂 | 分解 | 机制读法 |
|---|---|---|
| **A（z-only，push）** | 5 runaway + 1 expanded_bounded + 3 interictal | 去抑制多数**失控**；失控起始随耗竭强度/速度延后（1.7→9.3s）。唯一有界扩大 = `zA_q75_tz10000`（最弱最慢耗竭：48 events、峰 61Hz、无失控）——**z 单独**做到，与 m 无关。 |
| **B（m-only，pull）** | 6 suppress + 3 interictal | 纯刹车：只压制或没反应，**从不扩大**。 |
| **C（z+m）** | 全 9 interictal_like | 在测到的角落（强 z + 强 m），m 把 z 推力**抵消回基线**，不是有界扩大态。 |

## 3. 承重局限（必须随结果一起交接）

**arm-C 预注册选格塌进一个角落。** §7 预注册规则本想按实测 z 耗竭取弱/中/强三档 = 目标 `z_min ∈ {0.8,0.5,0.2}`，但实测去抑制没覆盖到"弱"这档（多数格深度耗竭到近 0），三档全落到强档 `q50`；m 侧同理全落到 `ta500_f20`。结果 arm C 实际只测了"强 z × 强 m"一个角落，全 interictal。**⇒ z+m 的中间地带（弱 z 自限 + 分级 m）没测到**，现在**不能下 clean no-go**，只能说"在测到的角落里 m 过刹回基线"。

## 4. 现状与在跑

- **代码全绿**：34 测试过；引擎 6 核心文件 SHA 未变，不 re-bless。
- **标定完成且预注册**：`calibration.json`（`I_th_EI` q50/q75/q90=1.67/95.2/391.7；`eta_m_table`；arm-C 3×3 规则）。
- **P3 多 seed 完成**（`b2lcmmokj` → `per_seed/multiseed_summary.json`；seed 1/3/4、T=15s、4 workers 并行）：跨 seed 表 —— `zA_q75_tz5000` 3/3 runaway（9293/9499/9758ms）、`zA_q50_tz10000` 3/3 runaway（4937/4707/4862ms）、`zA_q90_tz10000` 3/3 interictal、**bounded 格 `zA_q75_tz10000` 只 seed=1（seed 3/4 退 interictal）= 1/3 seed-脆弱**。基线 38/40/39。**本轮到此止，不进 40s acceptance、不做 field concordance。**

## 5. 可报告 / 禁止报告

**可报告**：M4-MZ 在 E1146 上复现"间期基线 + runoff"两 regime（对标 M2 达标，跨 3 seed 基线 + seed-1 runoff）；z 是 push（去抑制→失控/弱档有界扩大）、m 是 pull（纯刹车）、z+m 在强×强角落 m 过刹回基线；bounded/returned 未在 z+m 出现（且 z+m 中间地带未测）。

**禁止报告**：❌ 称任何格为 seizure；❌ 把 runaway 早停截断当发作；❌ 称 z+m clean no-go（中间地带没测）；❌ 从 field 分数反挑参数；❌ 把"轴向保持可接受"当成改 topic4 主文档"必须破轴"框架（分支内口径）。

## 6. 下一步（待用户定）

1. P3 跑完 → 出 3 张图（phenotype map / mechanism traces / spatial recruitment）+ 写最终 archive report（答 8 问 + 三选一）。
2. **修正版 z+m 扫描**（补 §3 塌角缺口）：以自限的弱 z（`zA_q75_tz10000` 档）为 push、叠分级 m，看 m 能否把"z 单独的有界扩大"稳成可再触发的招募态，或把"强 z 的失控"刹成有界而非刹回基线。这是回答同行 z+m 假说的**关键补测**，属新一轮（不在本轮预注册内，明确标 post-hoc）。

## 7. 交接坐标

- branch/worktree：`codex/topic4-mz-slowvars` @ `.worktrees/topic4-mz-slowvars`
- commits（分支内，未 push）：`16fb03f`（代码+测试+标定）、`41f54c1`（discovery 结果 + 并行化 multiseed）
- 代码：`src/snn_engine/mz_slow_vars.py`、`src/topic4_mz_slowvars.py`、`scripts/run_topic4_mz_slowvars.py`（calibrate/rss-audit/discovery/multiseed/capture-figures）、`config/topic4_mz_slowvars.yaml`、`scripts/plot_topic4_mz_slowvars.py`
- 结果：`results/topic4_sef_hfo/mz_slowvars/{calibration.json, discovery_summary.json, per_run.jsonl/csv, p3_candidates.json}`（P3 输出 → `per_seed/`）
- 设计：`docs/superpowers/specs/2026-07-18-topic4-mz-per-neuron-slowvars-design.md`
- 关联：[[project_topic4_m4_divisive_shared_inhibition_2026-07-05]]（S_G 池 bound-not-terminate）、[[project_topic4_m4_3a_discovery_2026-07-10]]（shunt clean no-go）——M4-MZ 是同行提的第三条终止/恢复机制候选。
