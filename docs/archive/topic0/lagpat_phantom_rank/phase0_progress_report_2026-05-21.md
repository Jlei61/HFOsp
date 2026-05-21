# Phase 0 进度白话报告（2026-05-21 09:50）

> Topic 0 broad re-derivation 进行中。本报告同时是 SEF-ITP framework Phase 0 ledger。
> 受众：用户复盘 + 协作者外审 + 半年后回看。
> 代号只作括号补注；正文走 CLAUDE.md §8 三段式。

---

## 0. 整体总览

| Step | 内容 | 状态 | 关键发现（一句话） |
|---|---|---|---|
| **5a** | PR-2 主聚类 | ✅ DONE | "K=2 两类传播模式"结构稳健（34/35），但事件归类有 30% 调整 |
| **5b** | PR-2.5 复现性 | ✅ DONE | 复现等级整体降一档；fwd/rev 集合 5 个换位但全部能追溯到 5a |
| **5c** | PR-3 簇内刻板度 | ✅ DONE | PR-4 panel d "86% identity bias" → **92%（加强）** |
| **5d.1** | PR-4A 昼夜 | ✅ DONE | 昼夜模板占比无差异（与原版同向 NULL） |
| **5d.2.0** | PR-4B Step 0 lag validation | ✅ DONE | 与原版数字基本一致 |
| **5d.2.1** | PR-4B Step 1 rate-state | ✅ DONE | 全 NULL；L1 dominant rho 方向翻 (-0.083 → +0.183, NS) |
| **5d.2.2** | PR-4B Step 23 | ✅ DONE | ⚠ **原版唯一显著 finding (L3 高置信 Pearson r 7/8, p=0.016) 不复现 (5/8, p=0.547)** |
| **5d.3** | PR-4C 发作邻近 | 🔵 34+6/40 (resume in progress) | （预期 NULL，原版已封板阴性） |
| 5d.4 | PR-4D 模板速率分解 | ⛔ SKIPPED | user 决定优先 5e/5f/5g/5h |
| **5e** | PR-5 / PR-5-B 模板招募率 | 🟡 PENDING（即将 N=10 并行启动） | 核心 paper 数字：post-ictal +65.46 events/h, p=0.00128 |
| **5f** | PR-6 endpoint / swap / Step 6 | 🟡 PENDING | SEF-ITP H1/H2 真正瓶颈 |
| **5g** | PR-7 反向模板时间配对 | 🟡 PENDING | SEF-ITP H3 + framework P3 翻转 flag |
| **5h** | Topic 4 attractor | 🟡 PENDING | SEF-ITP framework 解锁条件 |
| 5i | 主文档收口 | 🟡 PENDING | 等 5e-5h 都完成 |

---

## 1. 三段式朴素话总览

### 1.1 测了什么

我们 24h 间期数据里"每个 HFO 群体事件归到哪一类传播模式"这件事，**3 周前发现做这件事用的"通道激活先后名次"数据被 21 年的老代码污染了**：不参与某个事件的通道也被偷偷塞了个假名次（这些假名次集中在 0 和 n_ch-1 两端）。

**Phase 0 任务 = 把所有用了这套数据的分析（PR-2 到 PR-7，共 40 个 subject）按修过的数据重跑一遍，看哪些结论站得住、哪些要改**。

### 1.2 怎么测的

每一个分析步骤都做同一件事：
1. **修过版**=每事件只在"实际参与"的通道之间重排名次，不参与的填中点 0.5；其余算法不变
2. **跑出来的数字** vs 原版做 paired 比较（同一 subject 比、做 Wilcoxon 配对检验）
3. **判定**：方向反 / 强度变 / 显著 vs NULL flip 都要标出

修过版结果走 `results/interictal_propagation_masked/` **旁路目录**，旧的不删——方便对比、不破坏既有归档。

### 1.3 揭示了什么（截止 2026-05-21 09:50）

**好消息——大结构站得住**：
- **"K=2 两类传播模式"** 在 34/35 主线 subject 上不变。phantom 没有制造或抹除"两类"这个结构层结论。
- **legacy MI（每 subject 整体刻板程度）** 40/40 仍显著。
- **PR-4 panel d "簇内 86% identity bias"** 不仅站住，反而**加强到 92%**——phantom 给数据加的是噪声不是身份偏置，去掉它之后簇内刻板度全部由身份排序解释。
- **PR-4A 昼夜模板占比** 原版 NULL，修过版仍 NULL，方向同。
- **PR-4B 全部 cohort Wilcoxon** 原版 NULL，修过版仍 NULL。
- **PR-4B Step 0 通道排名 exact order match** = 1.0（修过版与原版完全一致；这个 sanity 验证了 masked re-rank 的正确性）。

**警示——细颗粒会改**：
- **"哪个事件分到哪一类"** 主线 33 个 subject 中位 Jaccard 0.70，6/33 < 0.5。大约 30% 的事件被重新归类。
- **PR-2.5 复现 grade** 整体降一档（31/9/0 → 26/12/2）。
- **forward-reverse 反向模板对集合** 16/17 → 15/16，**5 个 subject 换位**（3 失：548/620/635；2 得：253/916）。advisor 鉴别测试通过：4/4 非退化换位都能追溯到 5a 的 label 变化，masking 没引入新偏置。
- **L1 dominant cluster rho 方向翻转** -0.083 → +0.183（cohort 同向 25/40，但 Wilcoxon p NS）——原版说"低 rate 阶段 cluster 更紧密"，修过版倾向反向，但两边都没显著，结论 = 仍 NULL。

**Checkpoint B 必须 advisor consult 的红线发现**：
- **PR-4B Step 23 唯一显著结果不复现**：原版 L3 高置信子集 (n=8, dom_r>0.7) Pearson r 高/低 rate 段 delta +0.083，**7/8 同向，p=0.016**（archive 称为"only significant result"）。修过版同一子集 Pearson r delta +0.053，**5/8，p=0.547**，**回到 NS**。

  这条原本是 Topic 1 §4 主文档里"L3 高置信亚组有探索性正向 H4 信号"的唯一来源。现在它没了。可能解释：
  - phantom 在被 dom_r>0.7 选中的"高置信"子集里碰巧聚集了正向 noise，被原版分析当 signal 看
  - 真实的 L3 rate-coupling 不存在（消失版本）
  - cohort 太小（n=8）+ 修过版换 label 后归到子集的 subject 也变了，导致 power 散

  这条**不立刻动主文档**，等 5d.3 PR-4C 跑完 + 5e PR-5 跑完一起做 Checkpoint B advisor consult。

---

## 2. 各 Step 三段式分卡

### Step 5a / PR-2 主聚类（cohort 5h40m wall）

**测了什么** — 每个 subject 的事件被分两类，那"两类"是不是真的不是被假名次驱动的。
**怎么测的** — 修过版重跑同样的 KMeans，比原版 vs 修过版的"哪个事件属哪类"用 Jaccard / AMI / contingency 表对比。
**揭示了什么** — 类别数 (K=2) 在 34/35 主线 subject 上稳健；唯一翻转 epilepsiae_916 (2→4) 是已知 Topic 4 attractor 的 GOF-fail subject。每事件归类层面，主线 33 个可比较 subject 的 Jaccard 中位 0.70，意味着约 30% 事件被重新归类。**关键 sanity**：Spearman ρ(Step 2 audit Δ, PR-2 rerun AMI) = 0.961 (p≈7e-20)——consumer 端跑出来的差跟之前 audit 测的差几乎完全一致，masking 策略没引入偏置。

代号补注：commit `cda1263`，archive `step5a_pr2_results_2026-05-20.md`。

### Step 5b / PR-2.5 复现性（cohort ~70s wall）

**测了什么** — 每个 subject 的事件按时间一半一半切，看两半各自重新跑出来的模板长得像不像，验证 PR-2 聚类不是过拟合。
**怎么测的** — split-half + odd-even-block 两种切法，每种切法在两半上用修过版 KMeans 重新聚类，按 mean_match_corr / assignment_agreement / reproducibility grade 评估。AGENTS.md cross-PR contract：fwd/rev 复现用 split-half OR odd-even 取并集。
**揭示了什么** — grade 分布 31 strong / 9 moderate / 0 weak → **26 / 12 / 2**，整体降一档但无大规模崩塌。fwd/rev 复现集合 16/17 → 15/16，**5 个换位**（3 失 548/620/635，2 得 253/916）。这是 Checkpoint A 关注的红线（5/16 = 31% turnover）。advisor 推荐的鉴别测试做了：4/4 非退化换位 subject 的 PR-2.5 集合变化全部能追溯到 5a 的 label-level shift。**Checkpoint A 通过**。

代号补注：commit `5a33d99`，archive `step5b_pr25_results_2026-05-20.md`。bug fix：_run_pr25 / _run_pr4a 内层 runner 需接收 `use_masked_features` 参数，原本误用了不可见的 `args.masked_features`。

### Step 5c / PR-3 簇内刻板度（无需重跑，从 5a JSON 抽数）

**测了什么** — 修过版 cluster labels 上的"每个 subject 簇内时序刻板程度"和"identity-bias fraction" 是不是变了。
**怎么测的** — 5a 主管道已经把这些数算到 masked per_subject JSON 里。读 40 个 subject 的 orig vs mask 字段做 paired Wilcoxon。
**揭示了什么** —
- **legacy MI（bools-masked，预期不变）**：max |Δ| = 0.000017，40/40 在两版都 p<0.05 → 验证 bools-masked 路径不受 phantom 影响
- **簇内 raw τ**：orig 中位 0.237 → mask 0.291 (Δ +0.054)，**39/40 增强**，p=1.27e-10
- **簇内 centered τ（去掉 identity bias）**：基本不变，p=0.69 NS
- **bias_fraction**：orig 87.9% → mask **92.2%**，p=3.17e-4

phantom 是噪声，不是 identity-bias source。去掉 phantom 后簇内 raw 信号增强但增量全来自 identity 通道排序——**PR-4 panel d "簇内 86% identity bias" 结论被加强**。Topic 1 §3.2 已被 user 同步更新（86 → 92）。

代号补注：commit `519a4b7`，archive `step5c_pr3_results_2026-05-20.md`。

### Step 5d.1 / PR-4A 昼夜模板占比（~3 min wall）

**测了什么** — 修过版 cluster labels 上的"白天 vs 晚上哪一类传播模式更主导"。
**怎么测的** — 每事件按 abs_time → day/night，算每 subject 的 dominant fraction 和 normalized entropy 在白天/晚上的差异，做 cohort Wilcoxon。
**揭示了什么** — dominant_fraction 白天中位 0.606 vs 晚上 0.608，Wilcoxon **p=0.73 NS**；原版 p=0.12 也是 NS；方向一致。"昼夜模板占比漂移弱"原本就是描述层结论，不动。

### Step 5d.2.0 / PR-4B Step 0 绝对 lag validation（~5 min wall）

**测了什么** — 修过版 cluster labels 派生的"每簇主导通道排名"和原版做绝对 lag 通道排名对照是否一致。
**怎么测的** — 用 lagPatRaw 的 per-event min-subtraction 派生 relative_lag，计算 per-cluster Pearson r、exact_order_match_fraction、pairwise_order_concordance。
**揭示了什么** — **exact order match = 1.0**（与原版完全一致；验证 masked re-rank 的正确性）；dominant cluster Pearson r 中位 0.580（原版 0.601，差异 0.02 不显著）；**8/40 subject 通过 dom_r > 0.7 高置信门**（原版 30 subject 中是 8 个，扩展到 40 subject 仍 8 个）。

### Step 5d.2.1 / PR-4B Step 1 rate-state 耦合（~1h30m wall）

**测了什么** — 把事件按所在时间段的 rate 分高/低段，看簇内时序刻板度在高/低 rate 段是否系统不同（如果是真有 rate-state coupling，高 rate 段应该有更强或更弱的簇内一致性）。
**怎么测的** — 每 subject 计算高 rate / 低 rate 段的簇内 raw τ + centered τ + L1 dominant cluster rho；cohort Wilcoxon。
**揭示了什么** — 全部 NULL：
- raw τ delta median +0.007, p=0.197 NS
- centered τ delta +0.003, p=0.057 borderline
- L1 dominant rho median **+0.183**（原版 **-0.083**，方向翻转但两边都 NS）
- 原版结论"L1 L2 cohort 全 null"在修过版**仍 null**——方向单 subject 翻转但 cohort 没改变 verdict

L1 sign flip 是 advisor 建议 Checkpoint B 关注点之一，但本身不动结论。

### Step 5d.2.2 / PR-4B Step 23 — lag span / Pearson + L3 高置信子集（~1h wall）

**测了什么** — 簇内 lag span（每事件在簇内最早最晚通道的时间差）和 per-cluster Pearson r（template 与簇内事件平均 lag 的相关）在高/低 rate 段是否系统不同。这是 PR-4B 的 "L3" 层。
**怎么测的** — Full cohort + 高置信子集（dom_r > 0.7）双口径 Wilcoxon。
**揭示了什么** ——

| metric | orig (n) | mask (n) |
|---|---|---|
| L3 lag_span full-cohort | Δ +0.001, **18/30**, p=0.135 | Δ +0.001, **23/40**, p=0.128 |
| L3 Pearson r full-cohort | Δ +0.033, **17/29**, p=0.265 | Δ +0.008, **20/37**, p=0.309 |
| **L3 Pearson r 高置信 (dom_r>0.7)** | **Δ +0.083, 7/8, p=0.016 ★** | **Δ +0.053, 5/8, p=0.547 NS** |

⚠ **原版唯一显著结果（高置信子集 L3 Pearson r）在修过版不复现**——p 从 0.016 升到 0.547，方向虽仍正但 5/8 比 7/8 弱很多。Topic 1 §3 / archive PR-4B 文档里"L3 高置信子集探索性正向 H4 信号"的唯一来源消失了。

可能解释：
- phantom 在 dom_r > 0.7 子集里碰巧聚集了正向噪声，被原版分析当 signal
- 真实的 L3 rate-coupling 不存在
- n=8 子集 power 太弱，masked 版选中的子集也微调

不立刻动主文档；Checkpoint B advisor consult 时正式判定。

### Step 5d.3 / PR-4C 发作邻近（在跑，34+6/40）

**测了什么** — 把发作前后划成 baseline / pre-ictal / post-ictal 三窗口，每个 subject 算"传播模式 5 指标"在窗口间是否系统变化。
**怎么测的** — 每 seizure 作中心，按主配置 (4/1/1h) + 辅助配置 (2/0.5/1h) 提取窗口，per-cluster τ / 模板距离 / occupancy 等。
**揭示了什么** — 进行中（34 subject 已完成，6 个 resume 在跑）。**预期 NULL**——原版历史已封板阴性（"5 指标 cohort Wilcoxon 全 null，仅 rate_by_template 单一信号显著"），修过版预期同方向。**封口报告 + cohort 数字** 在 PR-4C resume 跑完后由 consolidator 工具 (`scripts/consolidate_pr_cohort_masked.py --pr pr4c`) 重建 cohort artifact。

### Step 5d.4 / PR-4D 模板速率分解（**SKIPPED**，user 决定）

PR-4D 在 Topic 1 主结论里属次级描述层（rate-burst seizure-enrich 9 subject strict-match）。**user 2026-05-21 决定优先 5e/5f/5g/5h**——这些是 SEF-ITP framework H1/H2/H3/H4 的直接对接 + framework P3 翻转 flag 触发点。5d.4 留待 5i 收口 phase 决定是否补。

### Step 5e / PR-5 / PR-5-B 模板招募率（PENDING，N=10 并行即将启动）

**预期测什么** — 在 PR-5-A novel-template gate 通过的前提下，看 dominant template 的绝对事件率 (events/h) 在 baseline / pre-ictal / post-ictal 三窗口是否系统变化。
**为什么是优先级 1** — Topic 1 主文档 §2 核心 paper 数字：原版 `dominant_global` post−baseline median **+65.46 events/h**, p=**0.00128**（Bonferroni-pass）。这条直接进 SEF-ITP H4 解耦预测。
**计划** — N=10 并行 PR-5-A，consolidate 后跑 PR-5-B (cohort gate-dependent，串行)。

### Step 5f / PR-6 endpoint anchoring（PENDING）

**预期测什么** — PR-2 稳定模板的 endpoint (source ∪ sink) 通道是否在解剖上集中到 SOZ；forward/reverse swap 几何关系。
**为什么优先** — SEF-ITP H1 (endpoint spatial compactness) + H2 (source-sink reversal geometry) 直接对接。
**前置工作** — `scripts/run_pr6_template_anchoring.py` 需加 `--masked-features` flag（hardcoded 路径，要 patch），同 `run_pr6_step6.py` + `run_pr6_sup1_rank_entropy.py`。Patch 后才能并行。

### Step 5g / PR-7 反向模板时间配对（PENDING）

**预期测什么** — forward/reverse template 是否在事件级时间上配对（紧邻互呼应）。原版三 metric 全 NULL，framework 表述"compatible with mark-independent within tested precision"。
**为什么优先 + 警示** — SEF-ITP H3 (mark independence + stable geometry) 直接对接。**警示**：如果修过版把 P3 从 INCONCLUSIVE 推到 PASS 或 FAIL，这是 framework-level 修订，不允许默改 `docs/paper1_framework_sba.md`——必须停下来发起 framework review。
**前置工作** — `scripts/run_pr7_template_pairing.py` 需加 `--masked-features` flag。

### Step 5h / Topic 4 attractor（PENDING）

**预期测什么** — 35 stable_k=2 cohort 上重跑 Step 1（principal curve + GOF + label transition λ₂）。
**为什么优先 + 解锁条件** — SEF-ITP framework Phase 1+ 的硬前置（topic4_sef_itp_framework.md §"Phase 1+ 在 Topic 0 phantom-rank 修复完成前不启动"）。
**前置工作** — `scripts/run_attractor_step1.py` 需加 `--masked-features` flag → `build_rank_feature_matrix` 已经支持 `mask_phantom` 参数。

### Step 5i / 主文档 + AGENTS.md 收口（PENDING）

**预期做什么** — 把 Topic 1 主文档 §2 / §3 / §4 / §10 的所有 phantom-era cohort 数字更新到 masked 版本；AGENTS.md "Cross-PR Contract Lookups" 增 phantom-rank lookup（已 done 部分）；将 `use_masked_features=True` 默认翻为 True（旧默认 False = phantom 路径，留 deprecation warning）。

---

## 3. 工具链状态（2026-05-21 新增）

- `scripts/run_pr_parallel.py` — N=10 round-robin parallel launcher（commit `45eb221`）
- `scripts/consolidate_pr_cohort_masked.py` — cohort summary consolidator（supports pr4a / pr4a-followup / pr4c / pr5-gate / pr5-recruitment；同 commit）
- 机器资源：80 cores / 251GB RAM / load ~2，N=10 并行有充足余量
- 已有的 sequential 跑数：5d.3 PR-4C 6 subject resume **不停**，让它跑完

下一步执行：等 PR-4C resume 完成 → 跑 consolidator `--pr pr4c` 重建 cohort artifact → N=10 并行启 PR-5-A → consolidate → PR-5-B → 同步 patch PR-6 / PR-7 / Topic 4 runners → N=10 并行 5f/5g/5h。

---

## 4. Phase 0 接下来 ~ 时间预算

| Step | 估时（并行 N=10） |
|---|---|
| PR-4C resume 完 + consolidate | 30 min |
| 5e PR-5-A 并行 + consolidate | 30 min |
| 5e PR-5-B（cohort-dependent，串行） | 30 min |
| Patch PR-6 / PR-7 / Topic 4 runners | 1-2 hours |
| 5f PR-6 并行 + consolidate | 1-2 hours |
| 5g PR-7 并行 + consolidate | 1-2 hours |
| 5h Topic 4 attractor 并行 | 30 min |
| **Checkpoint B advisor consult**（5d-5g 总览） | 30 min |
| 5i 主文档收口 + AGENTS.md / CLAUDE.md | 半天 |
| **总计** | **~ 一天** |

vs 串行预算 **2-3 天**，并行节省 ~50%-70% wall time。

---

## 5. 还在 Phase 0 之外的开放问题

- PR-7 P3 翻转的 framework-level review 触发条件 — 等 5g 实测后才能判
- Topic 5 yuquan ictal cohort extension (PR-0.1) — 独立于 Topic 0，已 plan 但未执行
- HFO Detector v2 cohort 剩 13 个 subject 未跑（独立 infra）

这些都不在 Phase 0 任务范围内；Phase 0 = "Topic 0 broad re-derivation" only。
