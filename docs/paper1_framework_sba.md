# Paper 1 Framework — Stereotyped Bidirectional Attractor (SBA)

> **状态**：架构性 pre-registration 文档（最高优先级）
> **范围**：Paper 1 整体科学框架，横跨 Topic 1（事件内动力学）与 Topic 3（SOZ 空间归因）
> **不属于**：Topic 2（事件间周期性）— 那是独立 Paper 2
> **版本**：v1.1.2（2026-04-30 当日三次修订）— v1.0 → v1.1 五条结构性修订 + v1.1 → v1.1.1 P3 判据 cohort-level 化 + v1.1.1 → v1.1.2 marginal-aware **audit-triggered correction（在 PR-7 addendum 数据审计后发现 marginal assumption 漏洞、修订 criterion，再跑 addendum lock；技术上是 transparent post-hoc fix，不是从 v1.0 起就存在的 pre-registration）**，均见 §15 Changelog
> **当前 lock 范围**：单核心假设 / 5 条 sharp predictions / toy model 最小数学形式 / fitted model spec / 5 个 dumb baseline / 失败模式判据 / equivalence margin δ_excess
> **后续修改约束**：本文件锁定后，对 P1–P5 的 PASS/NULL/FAIL 判据、toy model 必含组件、baseline 列表、equivalence margin 的修改必须在主文档双源更新，**不允许**事后调整以匹配数据

---

## 1. 这份文档是什么 / 不是什么

**是**：

- Paper 1（合并原 Paper 1 + Paper 3 的"现象 + 机制"双层）的科学架构 pre-registration（含 v1.1.2 audit-triggered correction，详见 §15）
- 单一核心假设 + 5 sharp predictions：**P1 + P2 PASS, P3 INCONCLUSIVE-locked (compatible with mark-independent within tested precision), P4 + P5 pending**
- Toy model（BHPN-toy）与 fitted model（BHPN-fit）的最小数学合同
- 5 条 dumb baseline 的 model-selection 设计
- 框架失败模式与修订路径

**不是**：

- 替代 `docs/topic1_within_event_dynamics.md` / `docs/topic3_spatial_soz_modulation.md` 的科学口径
- 详细数值结果归宿（结果归 archive）
- Topic 2（事件间 ~2 Hz / refractory + slow modulation）的内容载体
- 任何 ictal phase transition / KONWAC adaptive coupling β / pre-ictal synchronization 的 framework（这些归 Paper 3）

---

## 2. 背景与诊断（为什么需要这份框架）

### 2.1 老 paper（KONWAC, 2024）的核心结构性问题

按 reviewer 视角，老 paper 失败有四条结构性原因：

1. **跨时间尺度混叙**：把"事件间 ~2 Hz 周期性"与"事件内 HFO 频段（80–250 Hz）振荡"混在同一个 Kuramoto 框架；当 Topic 2 证伪了"~2 Hz 是内禀振荡"后，整个 Kuramoto 框架被波及——但这是混叙的失败，不是 Kuramoto 在事件内 HFO 尺度上的失败。
2. **零可证伪命题**：模型有足够自由度拟合任何 stereotyped 序列，没有事先的 sharp prediction 被独立检验。
3. **关键 cohort 结论不复现**：power-law IEI（实为 lognormal，30/30）、pre-ictal synchronization（PR4–PR6 cohort null）、bidirectional E3 anecdote（已升级为 8/9 fwd/rev cohort，但老 paper 是 n=1）。
4. **Mechanism overclaim**：HFO 80–250 Hz 不能区分 E/I，老 paper 用"兴奋抑制相变"语言论述 ictal transition 缺物理基础。

### 2.2 当前 PR 序列累积的硬约束

任何后继 framework 必须**同时满足**以下七条 cohort 事实，否则不构成对现有数据的合理描述：

| # | 约束 | 来源 | Cohort |
|---|---|---|---|
| C1 | 模板群体级稳定（30/30 stable adaptive solutions） | PR-2 | 30/30 |
| C2 | stable_k 分布偏向 k=2（27 × k=2，2 × k=4，1 × k=6） | PR-2 | 30/30 |
| C3 | 模板跨时间稳定（split-half + odd-even reproducibility） | PR-2.5 | 23 strong + 7 moderate / 30 |
| C4 | 8/9 k=2 subject 的 fwd/rev pair 跨时间复现互逆关系 | PR-2.5 | 8/9 |
| C5 | fwd/rev 节点级 swap geometry 显著（sign-test p=0.031） | PR-6 Step 4b | n=6 fwd/rev cohort |
| C6 | Forward / reverse template 在 1s–1h 时间窗内**未见 mark dependence**（mark-independent 兼容） | PR-7 | n=6 |
| C7 | 间期事件几何在发作邻近窗口上无稳健变化（cohort null） | PR-4C | 30/30 |

C6 是最强约束：naive 对称 Hebbian-coupled bidirectional storage 几乎都预测短窗 reciprocity；PR-7 addendum (2026-05-01) verdict = INCONCLUSIVE，**1800s window + lag1_same_excess null-relative 干净 PASS**，10/30/60s + run_length_lift cohort CI underpowered at n=6 with structural outliers。**精确措辞**："数据 compatible with mark-independent retrieval；目前的证据不足以正面排除强短窗 reciprocity 模型，但也未观察到支持它的 cohort 信号"——**不**写"直接淘汰一类模型"或"ping-pong 假说被证伪"。任何后继 framework 仍应显式包含让两 attractor 独立调用的机制项（与 1800s + lag1 PASS 一致）。详见 `docs/archive/topic1/pr7_template_pairing/pr7_addendum_p3_equivalence_2026-05-01.md`。

---

## 3. 单一核心假设（H₀）

> **SBA hypothesis (v1.1)**：在 multi-source SOZ proxy 富集的局部脑网络（**pathological core neighborhood**，operationalized as: clinical SOZ ∪ data-driven M1 HFO-onset rate ∪ M2 ER-ratio）中，存在一个低维结构性骨架（structural backbone）。该骨架通过对称（或近似对称）Hebbian-like 耦合可塑性，存储一对几何对偶的稳定 attractor（forward / reverse template）。这对 attractor 在**事件内**（HFO 频段、ms 级）时间尺度上是 phase oscillator network 的双稳态 fixed-point solution。**事件间**触发由一个慢变 latent 变量 s_rate(t)（min–hour 尺度，与 Topic 2 一致）调制 retrieval 触发率（event rate），但 **attractor identity 不由 s_rate(t) 决定**——而是由每次 retrieval 的随机初始相位独立选取，跨 retrieval 之间统计独立。Attractor 端点（source ∪ sink）作为骨架的几何角色节点，应当与多源 SOZ proxy 显著重合；从骨架的 phase pattern φ\* 派生的 *directed* 特征（如 sin(φ_j\* − φ_i\*) 或 source→sink rank gradient）应对发作早期 LFP 传播方向构成 directionally informative prior。

**这条假设不包含**：

- 任何关于 E/I 神经元的论断
- 任何关于 ictal phase transition 的预测（Paper 3 范围）
- 任何关于事件间 ~2 Hz "周期性" 的解释（Paper 2 范围）
- 任何关于全脑网络的论断（仅 pathological core neighborhood 邻域范围，~ SEEG/ECoG 局部视野）
- 任何关于"clinical SOZ = true SOZ"的预设（multi-source proxy 的设计正是承认 clinical 标签可能不完整 / 不可靠）

---

## 4. Toy model — BHPN-toy（已被 SEF-ITP 取代）

> **2026-05-22 归档说明**：BHPN-toy 数学规范 (§4.1–4.5) 已被 `docs/topic4_sef_itp_framework.md` v1.0.2 取代（详见 topic4_sef L5 取代声明）。原 §4 完整内容（节点动力学 + Hebbian 存储 + core node heterogeneity + s_rate/ε_id 解耦 + T1–T7 最小事实清单）保留在 [`docs/archive/paper1/bhpn_toy_and_fit_spec_superseded_2026-04-30.md`](archive/paper1/bhpn_toy_and_fit_spec_superseded_2026-04-30.md) §4。
>
> 当前 modeling spec 以 SEF-ITP H1–H6 框架为准。Paper 1 引用 SBA toy mechanism 仅作为 historical context，正面 modeling 论述走 topic4_sef。

---

## 5. 5 条 sharp predictions（pre-registered）

每条 prediction 必须包含：null hypothesis / PASS / NULL / FAIL 判据 / 对应 PR / 当前 verdict。

### 总表

| # | Prediction | 状态 | PR | Verdict |
|---|---|---|---|---|
| **P1** | Template 跨时间稳定性 | DONE | PR-2.5 | PASS |
| **P2** | Bidirectional pair 共享几何骨架 | DONE | PR-6 Step 4b | PASS |
| **P3** | 短窗 attractor 调用 ≈ mark-independent (TOST equivalence) | DONE | PR-7 + addendum 2026-05-01 | **INCONCLUSIVE**（compatible with mark-independent within tested precision；1800s + lag1 null-relative PASS；10/30/60s + run_length_lift CI underpowered at n=6）。SBA 不被 falsified |
| **P4** | Attractor 角色节点解剖锚定多源 SOZ proxy | TODO | PR-T3-1 → PR-8 v2 | Pending |
| **P5** | 间期 Hebb 网络对发作早期方向性 informative | TODO | PR-9 (新) | Pending |

### 5.1 P1 — Template 跨时间稳定性（DONE）

- **Prediction**：subject-level template 在前/后半数据 split 上 cosine similarity ≥ 0.8 的比例 ≥ 25/30。
- **Null hypothesis**：template 是 finite-sample artifact，跨时间 split 不复现。
- **PASS**：≥ 25/30（包括 strong + moderate）→ 已 lock。
- **NULL**：< 25/30。
- **FAIL（与 SBA 矛盾）**：< 15/30，意味着 subject 内部根本没有可识别的稳定模板，整个 framework 拒绝。
- **Verdict (PR-2.5)**：23 strong + 7 moderate + 0 weak / 30，**PASS**。
- **数据来源**：`docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md` PR-2.5 节。

### 5.2 P2 — Bidirectional pair 共享几何骨架（DONE）

- **Prediction**：在 fwd/rev reproduced subject 上，节点级 source(forward) 与 sink(reverse) 的角色 swap 显著（sign-test p < 0.05）。
- **Null hypothesis**：fwd 与 rev 是独立 template，节点角色无共享结构。
- **PASS**：sign-test p < 0.05 + Wilcoxon paired diff median > 0。
- **NULL**：sign-test p ≥ 0.05。
- **FAIL（与 SBA 矛盾）**：swap geometry 反向（即 source(forward) ≈ source(reverse)），意味着两 template 不是 Hebbian 对称存储下的 fixed-point pair，要求改 framework。
- **Verdict (PR-6 Step 4b, fwd/rev cohort n=6)**：6/6 positive sign，sign-test p=0.031；Wilcoxon p=0.031，median +2 swap-vs-same node count → **PASS**。
- **附注**：cohort-wide n=21 上 Wilcoxon p=0.012 但 sign-test p=0.12（粗 cohort 上证据弱），主 framework claim 在 fwd/rev subset 上立得住。
- **数据来源**：`docs/archive/topic1/pr6_template_anchoring/pr6_template_endpoint_anchoring_plan_2026-04-25.md` §15 Step 4b。

### 5.3 P3 — 短窗 attractor 调用 ≈ mark-independent（v1.1.1：cohort-level TOST + 现实功效声明）

**v1.0 → v1.1 关键修订**：v1.0 把 "Wilcoxon p > 0.05 → PASS" 误用为支持独立的正向证据；这是统计学错误（fail-to-reject ≠ accept null）。v1.1 改用 **two one-sided tests (TOST) 等价性检验** + bootstrap CI。

**v1.1 → v1.1.1 修订（同日识别两个剩余问题）**：
1. v1.1 把 "subject-level |excess| < δ" 设为 PASS 第 1 条判据；但 PR-7 cohort 中 548 是 magnitude outlier（10s = −0.201, 30s = −0.188），单一 subject 字面 PASS 几乎不可能。这导致 v1.1 字面判据**事实上不可达**。事后排除 outlier 又是 rule fitting → 主判据必须改为 cohort-level robust，subject-level 降级 sanity diagnostic。
2. v1.1 列出"PR-7 cohort 实测 |median| ∈ [0.002, 0.029] 远小于 0.05"作为 δ 选取理由之一，是借数据为 δ 背书（轻度 circular）。v1.1.1 把 δ 论证完全限制在 *与 naive coupled storage 仿真预测的对比*，**不引用 PR-7 实测数字**。

#### Equivalence margin lock

**δ_excess = 0.05**（cohort-level robust median 的绝对值）。

**δ 选取理由（pre-registered, 仅来自 scientific equivalence 推理）**：

- naive symmetric-Hebbian-coupled bidirectional storage（**无** random initial phase, **无** i.i.d. 选择）在 BHPN-toy 大 N 仿真上 |excess| ~ 0.3–0.5（toy 可独立验证）
- δ = 0.05 ≈ naive 预测的 1/6 至 1/10，足以与 strong reciprocity 假说定量区分
- δ lock at v1.1.1，**不允许事后调整**；**不引用 PR-7 实测数据为 δ 背书**（避免循环论证：PR-7 的角色是检验数据是否进入 δ 界限，不是帮 δ 选位置）

#### Prediction（v1.1.1）

在 fwd/rev 双 template subject 上，对窗口 Δt ∈ {10s, 30s, 60s, 1800s} 全部满足以下 **cohort-level 三条**：

1. **Cohort median equivalence**：|cohort median(excess(Δt))| < δ_excess（robust median，不是 mean）
2. **Cohort bootstrap CI**：cohort median 的 bootstrap 95% CI ⊆ [−δ_excess, +δ_excess]
3. **Lag-1 / run-length null-relative metrics（v1.1.2 修订：marginal-aware）**：cohort `lag1_same_excess` vs N2 null（N2 保 marginal label 分布）的 bootstrap 95% CI ⊆ [−δ_excess, +δ_excess]；cohort `run_length_lift` vs N2 null 的 bootstrap 95% CI ⊆ [1 − δ_excess, 1 + δ_excess]

#### Subject-level sanity diagnostic（不进 PASS gate, archive only）

报告但**不**作为 PASS / NULL 判据：

- (a) 满足 |excess| < δ_excess 的 subject 占比；按窗口报告
- (b) **leave-one-subject-out** cohort median 是否仍落在 ±δ_excess 内（识别是否单一 outlier 驱动结论）
- (c) **leave-548-out**（pre-specified outlier）cohort median 与 bootstrap CI

**关键纪律**：subject-level sanity 仅用于**解读** outlier 影响，**不允许**用作 PASS 替代。如果 cohort-level 主判据 INCONCLUSIVE 但 leave-548-out PASS，最高档位是"sensitivity-only PASS, archive"（不写主文档主结论）。

#### PASS / INCONCLUSIVE / NULL

| 档位 | 判据 | Paper 1 处理 |
|---|---|---|
| **PASS** | cohort-level 三条全部满足 | 写入主文档；P3 作为 SBA framework 已检验机制约束之一 |
| **INCONCLUSIVE** | cohort median \|excess\| < δ_excess 但 bootstrap CI 跨 ±δ_excess（即 underpowered）| Archive only；主文档维持 "compatible with mark-independent within tested precision (TOST(δ=0.05) cohort CI underpowered at n=6)" |
| **SENSITIVITY-only**（v1.1.1 新增）| cohort 主判据 INCONCLUSIVE，但 leave-one-subject-out / leave-548-out 满足 PASS | Archive only；明确标注"sensitivity"，不写主文档主结论 |
| **NULL** | cohort median \|excess\| > δ_excess 且 leave-one-subject-out cohort median 仍 > δ_excess | reciprocity 是 cohort 普遍现象，非 outlier 驱动；要求改 framework（参考 §8 P3 NULL 行） |

#### 现实预期（pre-stated）

按 PR-7 现有 per-subject 表 + 548 outlier + n=6 + 上述判据，**realistic outcome 大概率是 INCONCLUSIVE 而非 PASS**：

- 点估计 cohort median 落在 ±δ_excess 内的可能性高（已知 PR-7 median 量级 ~0.02）
- bootstrap CI 在 n=6 + 548 outlier 下大概率跨 ±0.05（粗估 10s CI ~ [−0.13, +0.02]，30s 类似）
- TOST 双侧 p < 0.05 难以通过

这是 honest 等价性框架在小 cohort 下的既有代价，**不视为框架失败**。SBA 不会因 P3 INCONCLUSIVE 而 falsified；SBA 只在 P3 NULL（cohort robust |excess| > δ）时被 falsified。

#### v1.1.2 PR-7 addendum 状态（2026-05-01 完成 lock）

- 原 PR-7 v1 仅报告 Wilcoxon p, 未做 TOST + bootstrap CI
- v1.0 PASS verdict **撤回**
- PR-7 addendum 已执行 2026-05-01：`scripts/pr7_addendum_p3_equivalence.py` + `results/interictal_propagation/template_pairing/pr7_addendum_p3.json` + `docs/archive/topic1/pr7_template_pairing/pr7_addendum_p3_equivalence_2026-05-01.md`
- **Verdict = INCONCLUSIVE**（compatible with mark-independent within tested precision；TOST(δ=0.05) cohort CI underpowered at n=6 with structural outliers）

**Addendum 主结果摘要**：

| 测试 | Cohort median | 95% CI | TOST p | 单测 verdict |
|---|---:|---|---:|---|
| excess(10s) | −0.0181 | [−0.1247, +0.0219] | 0.062 | INCONCLUSIVE |
| excess(30s) | −0.0148 | [−0.1103, +0.0104] | 0.061 | INCONCLUSIVE |
| excess(60s) | −0.0101 | [−0.0947, +0.0045] | 0.062 | INCONCLUSIVE |
| **excess(1800s)** | **−0.0002** | **[−0.0015, +0.0002]** | **<0.001** | **PASS** |
| **lag1_same_excess** | **−0.0111** | **[−0.0306, +0.0082]** | **<0.001** | **PASS** |
| run_length_lift | 0.9774 | [0.9396, 1.0293] | 0.115 | INCONCLUSIVE |

**Outlier 结构**：T1 短窗 INCONCLUSIVE 完全由 548 单独 driving（leave-548-out 全 PASS）；T3 run_length_lift INCONCLUSIVE 是结构性的（1073=0.93, 635=0.95, 548=1.05 三方向 outlier，leave-any-one 仍 fail）。SENSITIVITY-only 档位**未触发**（T3 leave-out 仍 fail）。

**SBA framework 状态**：H₀ 单核心假设 + toy model 不受 INCONCLUSIVE 影响；INCONCLUSIVE ≠ falsified；NULL 档位**未触发**。Paper 1 P3 status 写法 lock 为 **"compatible with mark-independent within tested precision (TOST(δ=0.05) cohort CI underpowered at n=6 with structural outliers; 1800s window + lag1 null-relative excess clean PASS)"**——**禁止**写 PASS。详见 `docs/archive/topic1/pr7_template_pairing/pr7_addendum_p3_equivalence_2026-05-01.md` §5–§6。

#### 重要 framing 纪律

- P3 PASS 不等于"证明独立"，而是"在 equivalence margin δ=0.05 内 cohort robust median reciprocity 不显著大于零"
- P3 INCONCLUSIVE 是 honest 不确定，不是负面结果；写为 "compatible with mark-independent"
- 任何"用 leave-548-out 替代主判据"的写法都是 rule fitting，违反框架
- **未测**：alternative burst definitions、rate-state 切换、history-dependent regression、form-(4) latent-state coupling——这些列入 Paper 3 候选 follow-up，**不**进 Paper 1
- **数据来源**：`docs/archive/topic1/pr7_template_pairing/pr7_template_pairing_results_2026-04-29.md` §17 + 待补 PR-7 addendum

### 5.4 P4 — Attractor 角色节点解剖锚定多源 SOZ proxy（TODO）

- **Prediction**：在 fwd/rev reproduced cohort 上，stable template 的 endpoint(source ∪ sink) 节点集合在三种 SOZ 定义（clinical / data-driven M1=HFO-onset rate / data-driven M2=ER-ratio）下，frac(SOZ ∩ endpoint) − frac(SOZ ∩ middle) 的 subject-level Wilcoxon paired test p < 0.05，且至少在 clinical + 一种 data-driven 上方向一致。
- **Null hypothesis**：endpoint 与 SOZ 在解剖上独立，frac 差异 ≈ 0。
- **PASS**：≥ 2/3 SOZ definition 同方向 + 至少 1 个 Wilcoxon p < 0.05 + cohort delta median > 0 + held-out (50/50 time split) 一致。
- **NULL**：所有 SOZ definition 下 Wilcoxon p > 0.10。
- **FAIL（与 SBA 矛盾）**：endpoint 富集到 non-SOZ + middle 富集到 SOZ（方向反向）→ 要求 framework 修正"core = SOZ-anchored"这一连接。
- **Verdict**：Pending — 依赖 PR-T3-1 (data-driven SOZ audit) → PR-8 v2 (held-out + multi-source)。
- **关键执行约束（从 PR-8 v1 §16 继承，写死）**：
  - **Held-out validation 强制**：前 50% 时间数据用于训练 cluster + 决定 polarity 指派；后 50% 用于计算 Δ_event 并跑 H1。再翻过来跑一次取 cohort 一致方向（否则与 PR-6 同数据 partial circularity）。
  - **Multi-source SOZ 协同**：clinical / M1 / M2 三源并列报告，单源显著不构成 PASS（否则 SOZ 标签是 single point of failure）。
  - **`valid_mask` 严格显式传入**：禁止默认 `valid_mask=None` 路径（CLAUDE.md cross-PR contract）。
  - **`forward_reverse_reproduced` OR rule**：split_half OR odd_even（CLAUDE.md cross-PR contract）。
  - **HFO 80–250 Hz 不分 E/I**：framework / archive / paper 全文严禁出现"兴奋驱动 / 抑制反弹 / 证明机制"等表述。
- **结果归档路径**：`results/intra_event_spatial/`（per_subject + cohort + figures），主结果回写本框架文件 §5.4 verdict + 主文档 topic1 §7 + topic3 §7。

### 5.5 P5 — 间期骨架对发作早期方向性 informative（v1.1：directional predictor 修正）

**v1.0 → v1.1 关键修订**：v1.0 错误地把 `A_ij = cos(φ_i\* − φ_j\*)` 当 directed graph 用于预测 directed ictal coherence。cos 是偶函数（cos(a) = cos(−a)），数学上**无方向**，不能预测 i→j vs j→i。v1.1 把 directional predictor 从 A_ij **替换**为从 phase pattern φ\* 直接派生的反对称特征。

- **主 directional predictor (lock at v1.1)**：
  ```
  D_ij = sin(φ_j* − φ_i*)         # signed phase lead
  ```
  解读：D_ij > 0 ⇔ i 在 j 之前到达（i leads j），i.e., i → j 是 forward-leading 方向；sin 反对称（sin(a) = −sin(−a)）保证 D_ij = −D_ji。
- **备选 / sensitivity directional predictor**：
  ```
  R_ij = sign(rank_j^F − rank_i^F)   # source→sink rank gradient (from PR-6 template_rank)
  ```
  R_ij 与 sin-based D_ij 应同向；cohort 上必须报告两种 predictor 的方向一致性比例。
- **Prediction**：subject-level Spearman ρ(D_ij^forward, signed_directed_coherence_ij_ictal) > 0，cohort Wilcoxon p < 0.05；同时 R_ij-based 与 sin-based 在 ≥ 80% subject 上方向一致。
- **Null hypothesis**：间期 phase pattern 与发作传播方向无关；D_ij 不优于 random directed graph baseline。
- **PASS**（必须同时满足）：
  1. Cohort Wilcoxon (one-sided greater) p < 0.05；
  2. 与至少 2 个 dumb directional baseline（见 §7）的 Wilcoxon paired test 显著差异；
  3. Held-out（**leave-one-seizure-out cross-validation**：用 N−1 个 seizure 训练 template + 计算 D_ij，留 1 个 seizure 测试 ictal directionality；roll over all seizures）；
  4. sin-based D_ij 与 rank-based R_ij 在 cohort ≥ 80% subject 同向。
- **NULL**：Wilcoxon p > 0.05 或两种 directional predictor 方向不一致（< 80% 同向）。
- **FAIL（与 SBA 矛盾，并降级 Paper 1 ceiling）**：D_ij 与 ictal directionality **反相关** → framework 必须显式添加 "interictal / ictal directionality decoupling" 或 "reverse-flow ictal" 机制项；Paper 1 投稿目标从 Nat Comms 降至 Brain / eLife。
- **Fallback path（如果 PASS 与 NULL 都不成立）**：降级 P5 为 **"ictal spatial backbone similarity"**（undirected）——比较间期 |A_ij| 与 ictal undirected coherence 的 Spearman ρ。这是老 paper Fig 6 的严格化版本：PASS 时科学贡献减弱（仅承担 sanity）；不写入 framework 主结论，仅在 results doc §sensitivity 报告。Topic 1 §7.10 已识别此风险。
- **Verdict**：Pending — PR-9 立项。
- **关键执行约束（lock at framework v1.1）**：
  - **Held-out across seizures**：leave-one-seizure-out CV；**禁止**同一 seizure 的间期 + 发作做训练 + 检验（否则 trivial 自相关）。
  - **PDC 作为主 directional metric (locked)**：理由：spectral resolution + 标准化 + `mne-connectivity` 现成 API；DTF / time-domain Granger / phase slope index 仅作为 sensitivity，**不允许** verdict 出来后挑 metric。
  - **Dumb directional baseline 比较硬约束**：D_ij 必须显著优于 (i) random directed graph (matched in/out degree) (ii) 对称 |cos(φ_i\* − φ_j\*)| baseline（间期 strength 但无方向，验证 directionality 才是关键不仅是 strength） (iii) chain graph (linear sequential 1→2→...→N)，否则无论自身 p 多小都不算 PASS。
  - **老 paper Fig 6 比较**：PR-9 是 directional 升级，老 paper 是 undirected coherence similarity；两条在 Paper 1 里**不**互相替代——老结果作为 sanity（undirected 是 directed 的 marginal），新结果作为 prediction test。
  - **预防 v1.0 错误**：source code grep 在 PR-9 实现中**必须 raise on usage of `cos(φ-φ)` as directed predictor**；directional 处只能出现 sin-based 或 rank-based。

---

## 6. Fitted model — BHPN-fit（已被 SEF-ITP 取代）

> **2026-05-22 归档说明**：BHPN-fit 完整 spec (§6.1–6.3，含 50/50 时间切分 + ≤6 自由参数表 + F1–F4 v1.1.2 marginal-aware null-relative 形式) 已被 `docs/topic4_sef_itp_framework.md` v1.0.2 取代。原 §6 完整内容保留在 [`docs/archive/paper1/bhpn_toy_and_fit_spec_superseded_2026-04-30.md`](archive/paper1/bhpn_toy_and_fit_spec_superseded_2026-04-30.md) §6。
>
> SEF-ITP H1–H6 取代 F1–F4：geometry / mark-independence / coupling 等 mechanism 检验全部走 SEF-ITP Phase 1 runner。

---

## 7. Dumb baselines（model selection 设计）

每个 baseline 抹掉 BHPN-fit 的一个机制项，用于隔离该机制项的解释力。**fitted model 必须在每个对应 metric 上显著优于对应 baseline**（pre-specified Wilcoxon），否则该机制项不必要。

| Baseline | 抹掉的机制 | 应当失败的 prediction | 失败时意味 |
|---|---|---|---|
| **B1: Random graph** (matched degree distribution) | 全部 Hebbian 结构 | F1 + F3 + P5 | 网络结构信息无用 → framework 拒绝 |
| **B2: Rate-only** (predict mean event rate, ignore template) | 模板信息 | F1 + F3 | 模板对 prediction 无贡献 → 退化为 rate model |
| **B3: Symmetric Hebbian without core** (k_core = 0; 所有节点同质) | core node 异质性 | F4 + P4 | core 不必要 → role node 无解剖含义 |
| **B4: Core without bidirectional storage** (only forward template stored, no reverse fixed point) | 对偶 attractor 性质 | F1（held-out template geometry 仅 match forward, miss reverse） | bidirectional storage 不必要 → fwd/rev 独立 (与 PR-2.5 fwd/rev cohort 矛盾) |
| **B5: Bidirectional + core but deterministic attractor selection** (alternating fwd/rev or strong history bias instead of i.i.d. random initial phase) | random initial phase 机制（§4.4 Process B） | F2 (b) + F2 (c) — null-relative：B5 应当显著 \|lag1_same_excess\| > δ_excess **vs N2 null** 与 \|run_length_lift − 1\| > δ_excess **vs N2 null**（**不**用 marginal-50/50 / geometric(0.5) anchor，因 §6.3 v1.1.2 修订）| random initial phase 不必要 → 数据相对 marginal-preserving null 应呈 deterministic alternation 或 history bias 偏离（与 PR-7 addendum 1800s + lag1_same_excess null-relative 干净 PASS 矛盾；short-window + run_length_lift INCONCLUSIVE 不能在 BHPN-fit B5 比较中作为对 B5 的支持） |

**核心 framing**：每个 baseline 失败的方式 *不同*。B1 失败说"任何结构都 informative"，B2 说"哪怕是模板也要"，B3 说"core 是必须"，B4 说"双稳态是必须"，B5 说"random initial phase i.i.d. selection 是必须"（v1.1.2 修订前误称 "slow gating"——v1.1 拆分 s_rate / ε_id 后，attractor identity 的 i.i.d. 来源是 random initial phase，不是 slow gating）。这是真正的 Popperian model selection，不是装饰。

---

## 8. 失败模式与框架修订路径

| 失败 | 框架解读 | 修订路径 | Paper 1 状态 |
|---|---|---|---|
| **P1 fail** | template 不稳定 | framework 整体崩溃 | abort，Paper 1 重新设计 |
| **P2 fail** | fwd/rev 不共享骨架 | 退化为"两 independent template" | 改 framework：去掉对称 Hebbian 假设；Paper 1 ceiling 大幅降低 |
| **P3 NULL（cohort robust median \|excess\| > δ_excess + leave-one-subject-out 仍 > δ）** | random initial phase 假设不成立；简单对称存储 + i.i.d. selection 模型证伪 | 升级到 ping-pong / asymmetric Hebbian / history-dependent / inhibitory rebound 类模型；要求 LFP / 单元验证 | Paper 1 不能 close；分两段投稿 |
| **P3 INCONCLUSIVE（cohort median \|excess\| < δ 但 bootstrap CI 跨 ±δ）— v1.1.1 likely outcome** | 数据 underpowered（n=6 + 548 outlier），不能区分 strict independence vs 弱 reciprocity | 不写 P3 PASS 进主文档；档位降为 "compatible with mark-independent within tested precision (TOST(δ=0.05) underpowered at n=6)" | Paper 1 ceiling 微降；archive + 主文档 honest 不确定声明；**SBA 不被 falsified** |
| **P3 SENSITIVITY-only（cohort 主 INCONCLUSIVE 但 leave-548-out / leave-one-out PASS）** | outlier-driven uncertainty | Archive 报告；主文档维持 INCONCLUSIVE；**禁止**用 leave-548-out 替代主判据 | 同 P3 INCONCLUSIVE |
| **P4 fail（held-out cohort）** | 解剖锚定不成立 | framework 退化为 mathematical substrate（不带 SOZ 解释） | Paper 1 ceiling 降级；clinical anchor 弱化 |
| **P5 NULL（directional predictor 不优于 baseline）** | 间期 → 发作 directional transfer 不成立 | 走 §5.5 Fallback：降级为 "ictal spatial backbone similarity" (undirected) | Paper 1 仍可投但 ceiling 降至 Brain / eLife |
| **P5 FAIL（D_ij 与 ictal directionality 反相关）** | 间期 backbone 与发作传播 *负* 相关 | framework 必须显式添加 "interictal/ictal directionality decoupling" 机制项 | Paper 1 ceiling 降至 Brain / eLife，Discussion 必须正面讨论 decoupling 机制 |
| **B5 不优于 BHPN-fit** | random initial phase i.i.d. selection 不必要 | 与 P3 NULL 路径合并 | 同 P3 NULL |
| **B3 不优于 BHPN-fit** | core 不必要 | 与 P4 fail 路径合并 | 同 P4 fail |

**重要**：**任何 verdict 都允许在 archive 报告，但只有当 P1 + P2 + (P3 或 P4 或 P5 中至少一条) PASS 时**，主文档（topic1 / topic3）才能写入 framework 级别结论。仅 P1 + P2 PASS（即只完成"现象学 + 几何骨架"）等价于 Paper 1 老 paper Section 1 的严格化版本，不构成"机制 + prediction"的升级。

---

## 9. PR 映射与工作量估算

| Prediction / 组件 | PR | 状态 | 估算工作量 | 依赖 |
|---|---|---|---|---|
| P1 | PR-2.5 | DONE | — | — |
| P2 | PR-6 Step 4b | DONE | — | — |
| P3 | PR-7 | DONE | — | — |
| Toy model T1–T6 | PR-T4-1（新立） | TODO | 2–3 周 | 本框架 |
| P4 | PR-T3-1 → PR-8 v2 | TODO | T3-1 = 2 周；PR-8 v2 = 3 周 | 本框架 + PR-T3-1 |
| P5 | PR-9（新立） | TODO | 3–4 周 | 本框架 |
| Fitted model BHPN-fit + 5 baselines | PR-T4-2（新立） | TODO | 4 周 | PR-T4-1 + P4 + P5 |

总计 ~14–16 周（toy + T3-1 + 8v2 + 9 + fitted + baselines），与 Paper 1 6–9 个月时间线一致。

---

## 10. 命名 / 范围 / Out of scope

### 10.1 命名约定

- **Framework**：SBA = Stereotyped Bidirectional Attractor（科学层）
- **Toy model**：BHPN-toy = Bidirectional Hebbian Phase Network, abstract version
- **Fitted model**：BHPN-fit = subject-specific fitted version

**显式不复用**：

- ❌ KONWAC（已被 Topic 2 部分证伪，且与 ictal phase transition 绑定）
- ❌ Hopfield-Kuramoto（保留作为数学性质引用，但不作为模型 brand）
- ❌ Ping-Pong（PR-7 已证伪 short-window reciprocity）

### 10.2 Paper 1 包含范围

- Topic 1 PR-2 / PR-2.5 / PR-6 / PR-7 全部
- Topic 3 PR-T3-1 + 老 paper Section 1 SOZ-AUC 验证
- 本框架 + BHPN-toy + BHPN-fit + 5 baselines
- PR-9 间期→发作 directionality
- PR-8 v2 解剖锚定
- 临床 SOZ-AUC 跨数据集验证（已有）

### 10.3 Out of scope（写死，不允许进 Paper 1）

- ictal phase transition、KONWAC adaptive coupling β（→ Paper 3）
- pre-ictal synchronization（PR4–PR6 cohort null，已封板）
- ~2 Hz 周期性 / power-law IEI / refractory + slow modulation 解释（→ Paper 2）
- E/I 兴奋抑制机制层（HFO 80–250 Hz 不分；→ 未来 LFP / 单元 work）
- 全脑 Kuramoto / 全脑 connectivity（→ 未来 modeling work）
- subject 548 single-subject case study（→ 独立 follow-up）
- history-dependent marked point process（→ Paper 3 / 独立 follow-up）

---

## 11. Paper 1 论证流（写作顺序）

按本框架，Paper 1 的论证流必须严格按以下顺序：

1. **Introduction**：epilepsy as network、interictal data 价值、老 paper 现象描述层有效但缺可证伪框架
2. **§1 Cohort phenomenology**（PR-2 + PR-2.5）：30/30 stable templates + 8/9 fwd/rev → P1
3. **§2 Geometric backbone of paired attractors**（PR-6 Step 4b）：sign-test p=0.031 → P2
4. **§3 Toy model: BHPN-toy**：minimal mechanism that produces (P1 + P2 + P3 + P4 + P5 prediction set) → 这是 paper 的"机制"层
5. **§4 Sharp predictions derivation from toy**：5 条预先 lock 的 prediction
6. **§5 Test of P3: temporal independence**（PR-7）：mark-independent 一致 + ping-pong 排除 → P3 PASS（这一节展示 paper 的负面结果如何变成 framework 支持）
7. **§6 Test of P4: anatomical anchoring**（PR-T3-1 + PR-8 v2）：multi-source SOZ → P4 verdict
8. **§7 Test of P5: ictal directionality prediction**（PR-9）：interictal A_ij → ictal PDC → P5 verdict
9. **§8 Fitted model + dumb baselines**（BHPN-fit + 5 baselines）：subject-level held-out + 各 mechanism component 必要性
10. **§9 Clinical relevance**：SOZ-AUC 跨数据集 + 多源 SOZ + （如有）surgical outcome
11. **Discussion**：framework survived/falsified components；what's not claimed (E/I, ictal transition, between-event); honest limitations

**写作顺序硬纪律（v1.1）**：

- toy model（§3）必须在 prediction tests（§5–§8）**之前**呈现
- §5 PR-7 数据必须按 PR-7 addendum 重判后的档位呈现：(a) PASS → "P3 equivalence test (TOST cohort-level, δ=0.05) PASS"；(b) INCONCLUSIVE（v1.1.1 现实预期）→ "compatible with mark-independent within tested precision; cohort TOST underpowered at n=6"；(c) NULL → 按 §8 P3 NULL 行处理。**不**以 "Wilcoxon p>0.05 → 独立"、"ping-pong 假说被证伪" 等 v1.0 语言；**不**用 leave-548-out 替代 cohort 主判据
- 所有 mechanism 论断仅限 network / Hebbian-like plasticity / phase oscillator 层；不出现 E/I、兴奋抑制、inhibitory rebound
- §5.5 P5 directional predictor 必须明确为 sin-based D_ij 或 rank-based R_ij；任何 "cos(φ−φ) directed graph" 表述都是 v1.0 错误，必须根除
- 任何 "BHPN-fit 逐事件预测 fwd/rev label" 的论述违反 §4.4 i.i.d. 假设，写作时必须避免

---

## 12. 与 topic docs 的双向链接（必须维护）

- `docs/topic1_within_event_dynamics.md` §2 一句话当前结论 + §10 历史文档索引引用本文件
- `docs/topic3_spatial_soz_modulation.md` §2 一句话当前结论 + §10 历史文档索引引用本文件
- `docs/topic2_between_event_dynamics.md` §1 显式声明本文件不包含 Topic 2 内容，Topic 2 是独立 Paper 2
- 后续 PR-T4-1 / PR-T4-2 / PR-9 / PR-8 v2 / PR-T3-1 的 plan-of-record archive doc 顶部必须 `> 上游：docs/paper1_framework_sba.md` 块

---

## 13. 自检清单（v1.1 lock 状态）

- [x] 单核心假设清晰，无堆叠子假设；不预设 "clinical SOZ = true SOZ"
- [x] 5 sharp predictions 每条有 null + PASS + NULL + FAIL + 对应 PR
- [x] **P3 用 cohort-level TOST equivalence + bootstrap CI 而非 fail-to-reject**；δ_excess = 0.05 lock；**主判据 cohort robust median**（subject-level 仅 sanity diagnostic，不进 PASS gate）；**δ 论证仅来自 scientific equivalence 推理**，不引用 PR-7 实测；INCONCLUSIVE 是 v1.1.1 下小 cohort 的现实可能 outcome，不是失败
- [x] **P5 directional predictor = sin(φ−φ) 或 rank gradient，不是 cos(φ−φ)**；A_ij 仅描述 storage dynamics
- [x] toy model 自由参数 ≤ 6，且每个自由度有 falsification 路径（T6 + T7）
- [x] **toy model s_rate(t) 与 ε_id 统计独立**，分别匹配 Topic 2 慢漂与 PR-7 短窗 i.i.d.，无内部时间尺度矛盾
- [x] fitted model 自由参数 ≤ 6
- [x] **fitted model F1–F4 仅 aggregate-level + conditional**，禁止逐事件 label 预测（与 §4.4 i.i.d. 一致）
- [x] 5 dumb baselines 每个抹掉一个 specific mechanism component
- [x] 失败模式 → framework 修订路径明确（含 P3 NULL / INCONCLUSIVE 区分；P5 NULL / FAIL 区分）
- [x] HFO 80–250 Hz 不分 E/I 限制写入 §5.4 + §10.3
- [x] Topic 2 / Paper 3 范围 explicit out of scope
- [x] 命名不复用 KONWAC / Hopfield-Kuramoto / Ping-Pong
- [x] 已 verified 的 P1, P2 不允许事后调整 PASS 判据；**P3 v1.0 PASS 撤回，待 PR-7 addendum 重判**
- [x] P4 / P5 的 PASS / NULL / FAIL 判据 lock at framework v1.1 time，先于 PR 执行
- [x] held-out validation 强制（P4 split-half + P5 leave-one-seizure-out）
- [x] multi-source SOZ 协同强制（P4）
- [x] dumb baseline 显著性 强制（P5：必须打过 random directed + symmetric strength + chain）
- [x] PR-T4-1 / PR-T4-2 / PR-9 / PR-8 v2 / PR-T3-1 plan-of-record 必须在顶部声明继承 v1.1（不是 v1.0）

---

## 14. 一句话承诺

Paper 1 提交一个 minimal framework，从单一假设（SBA）导出 5 条事先 lock 的 sharp prediction：P1（时间稳定，DONE PASS）+ P2（共享几何骨架，DONE PASS）+ P3（mark-independent, cohort-level TOST equivalence δ=0.05；v1.0 PASS 撤回，v1.1.1 现实预期 INCONCLUSIVE/compatible pending PR-7 addendum）+ P4（解剖锚定，TODO via PR-T3-1 → PR-8 v2）+ P5（directional predictor sin(φ_j\*−φ_i\*)，TODO via PR-9）；并通过 5 个 dumb baseline 隔离每个机制项的必要性。无论 verdict 如何，PASS 路径产出"现象 + 机制可证伪 + 临床锚"的完整故事；INCONCLUSIVE 路径维持 honest 不确定声明（不写 PASS）；NULL 路径按 §8 失败模式表降级或修订 framework，不写入主文档主结论。framework 不论及 E/I 机制、ictal phase transition、事件间 ~2 Hz 周期性——这些是 Paper 2 / Paper 3 / 未来 work 的范围。

---

## 15. Changelog

**当前版本 = v1.1.2**（2026-04-30 当日三次 lock-time 修订：v1.1 initial → v1.1.1 cohort-only P3 + δ 不引数据 → v1.1.2 marginal-aware null-relative）。

完整 changelog（5 条 v1.0→v1.1 结构性硬伤修订 + v1.1.1 两条 outlier-driven 修订 + v1.1.2 PR-7 addendum 触发的 marginal assumption 漏洞修订 + 后续修订协议）见 [`docs/archive/paper1/sba_changelog_v1.1.x_2026-04-30.md`](archive/paper1/sba_changelog_v1.1.x_2026-04-30.md)。

任何对 P1–P5 PASS/NULL/FAIL 判据、δ_excess、baseline 列表、framework 锁定子句的修改，必须先在归档 changelog 新增 v1.x 条目（动因 + 修订 + 影响范围），再同步更新主 doc 对应章节。已 verified 的 P1/P2 PASS 判据不允许事后调整；P3 v1.1.2 cohort-level + null-relative 主判据不允许退回 marginal-50/50 形式。
