# Topic 4：SEF-ITP Framework —— 间期模板传播的空间易激场模型

> **状态**：v1.0.3 framework lock，2026-05-22；**Phase 0 解锁 2026-05-21（lagPatRank phantom-rank broad re-derivation Step 5a–5h 全部完成 + Checkpoint A/B advisor consult 通过）**
> **版本历史**：v1 lock 2026-05-20 (initial) → v1.0.1 advisor 5-fix 同日 → v1.0.2 user review 6-fix 同日（H2 工程不可行修复 + H1 概念冲突拆分 + H3 verdict naming 修订 + H4 normalized instability + H5 统计合同收紧 + Topic 1/3 cross-doc 实际落字）→ **Phase 0 完成 2026-05-21**（Topic 0 §3.1 phantom-rank 修复跑完所有下游 PR）→ **v1.0.3 lock 2026-05-22**：H2 prose 改 PR-6 swap_check ingest（CLAUDE.md §5+§6 错误三段修复：cohort 用错 candidate 字段、重新发明 PR-6 已有 helper、升级 mechanism-sanity tier 为 cohort claim 都修正）+ H1 v1.0.7 null pool 扩到全 SEEG implantation（v1.0.6 lagPat 池循环论证 bug 修正）
> **取代**：`docs/paper1_framework_sba.md` 里的 BHPN-toy 数学模型部分（§4 toy model 数学规范 + §6 fitted model BHPN-fit spec 的 toy-mechanism 子句 + §7 dumb baselines 中针对 toy 机制的项）
> **保留**：`paper1_framework_sba.md` 里 P1 / P2 实证 verdict + P3 PR-7 addendum 锁字 + P5 directionality（sin-based）红线
> **硬前置（已解锁 2026-05-21）**：~~`docs/topic0_methodology_audits.md` §3.1 `lagPatRank` phantom-rank 修复 + §5 broad re-derivation roadmap 完成是所有 Phase ≥ 1 的不可越闸门~~ → **Phase 0 完成验收：所有 Phase ≥ 1 实验可以启动**。phantom-rank 修复结论（按 H1–H6 对应）：
> - **H3 (coordinate-free λ₂ metastability) 实质加强**：orig 10/34 (p<0.001) → masked **13/34**（5 个原 NULL/borderline 新增显著：pengzihang/253/442/958/1096；0 个反向丢；like-for-like n=33 子集 9→12）。详见 `docs/archive/topic0/lagpat_phantom_rank/step5h_topic4_attractor_results_2026-05-21.md`
> - **H4 (slow rate modulation; PR-5-B post-ictal recruitment)**：dominant_rate 主信号 `+65.46 events/h, p=0.0013` → `+65.66 events/h, p=0.0004`，direction + magnitude 都保持。详见 `step5e_pr5_results_2026-05-21.md`
> - **H4 联动 (PR-4B L3 high-conf Pearson r)**：orig 唯一显著 finding (n=8, p=0.016, 7/8) **修过版不复现** (p=0.547, 5/8) → 归 fragility-on-small-n，**不**进 H4 evidence base
> - **H1/H2 (PR-6 endpoint anchoring + fwd/rev swap)**：H1 cohort NULL 不变；H2 fwd/rev swap 9/9 → 8/8 都 positive (−1 subject from 5b)；Step 6 held-out swap_class concordance **0.69 → 0.82 实质提升**；rank displacement F_norm/τ/ρ 几乎完全不变。详见 `step5f_pr6_results_2026-05-21.md`
> - **PR-7 antagonistic temporal pairing (H3 mark-independence 联动) ✅ 完成 2026-05-22**：H1 triple-gate cohort-level NULL 不变；**P3 framework-flip gate CLEAR**：on orig P3 cohort (n=6) 用 masked features 重跑，verdict = **INCONCLUSIVE 完全保持 (4/4 flag 与 orig 一致)**；broader cohort (n=8 或 n=30) verdict 形式上滑到 NULL 但归 cohort × power × 5b fwd/rev 翻动互相作用非 phantom-rank 在 P3 statistic 的直接效应。`docs/paper1_framework_sba.md` v1.1.2 **不修订**。详见 `docs/archive/topic0/lagpat_phantom_rank/step5g_pr7_results_2026-05-21.md`
> - **Cohort tier 数字 lock**：Tier 1 cohort 大小见下方 §5；epilepsiae_916 已确认 cohort exit（stable_k 2→4），其他 34 stable_k=2 subject 全部进 Tier 1
> 
> **Phase 0 verdict（一句话）**：phantom-rank 修复**实质性加强**了 SEF-ITP framework 的两条关键 H3/H4 证据；**一条 PR-4B exploratory 翻转**被识别归类（fragility-on-small-n，不入主结论）；H1/H2 大方向都保持，其中 H2 mechanism 信号略缩 magnitude 但方向 100% 保持。
> **范围**：Topic 4 模型层 + Topic 1 within-event 现象的机制解释
> **不属于**：Topic 2 事件间周期（→ Paper 2）、ictal phase transition（→ Paper 3）、HFO 80–250 Hz carrier 振荡机制层

---

## 0. 一句话承诺（朴素表述，CLAUDE.md §8 大白话风格）

我们假设：**间期群体 HFO 不是随机噪声，而是脑里某个空间组织化的"病理易激区"被反复触发后、扩散物理留下的痕迹**。这个区有它固有的几何形状（形状本身我们不预测——它是真实病理网络的物理结构，是模型输入）；模型预测的是**这个形状被反复采样后，在我们的数据上必然留下哪些可见的指纹**：通道传播顺序的稳定性、source/sink 的几何反转、模板选择的随机性、慢变率与模板几何的解耦、发作邻近的端点身份漂移。

如果这些指纹在数据上被观察到 → 病理易激场假说被支持。
如果指纹大多数没看到 → 假说在我们的数据精度内被证伪，回到讨论是否换模型。

我们**不**预测病理易激区具体长在哪、**不**预先假定它和临床 SOZ 边界重合、**不**解释 HFO 80–250 Hz carrier 振荡是哪来的。

---

## 1. 为什么换掉旧模型

### 1.1 旧 BHPN（Bidirectional Hebbian Phase Network）的根本问题

旧 plan（`docs/archive/topic4/pr_t4_1_bhpn_toy/pr_t4_1_bhpn_toy_plan_2026-05-01.md`，已 SUPERSEDED）的设计是：

1. 把"目标通道顺序"预先塞进一个对称 Hebbian 连接矩阵 `A_ij = ρ · cos(φ*_i − φ*_j)`
2. 让 Kuramoto 方程演化
3. 数学保证它会落到 `φ*_F` 或 `π − φ*_F` 两个对偶 fixed point
4. 宣称"模型复现了正反模板"

**这是循环论证**：

- 现象（两个互逆模板）已经被预先编码进矩阵 `A_ij`
- 模型只是把预先编码的东西"演化"一次显示出来
- **不解释**为什么是这套通道、为什么是这个顺序、为什么发生在 SOZ 区域

这和老 KONWAC paper 犯的是同类病：数学漂亮，机制空洞。BHPN-toy 即使把所有 11 个单元测试 + 7 个集成 sanity 跑过，对评审而言它**仍然只是一个把"对称双稳态"重新画了一遍的装饰**——不是机制解释。

### 1.2 新方向：空间组织化的病理易激场 + 局部扩散物理

朴素思路：

- 间期事件**不是**某个内禀振荡器在跳动
- 是 SOZ 内部某处的局部异常兴奋，**像石头扔进水里激起涟漪**一样，经过场电势 / ephaptic / 局部组织扩散，扫过附近通道
- 通道在事件里的点火先后顺序 = 涟漪扫过它们的先后
- 正反两个模板 = 同一个病理区的"两个端点"各能成为涟漪起点（A→B vs B→A）
- 慢漂的觉醒度 / 状态变量 控制涟漪触发**频率**，**不**控制涟漪走哪条路

这个思路把"空间"放回到机制核心——这正是我们经验证据的核心维度（PR-2 / PR-6 揭示的是空间几何，不是抽象 phase pattern）。模型不再是拟合机器，因为：

- θ(x) 反映**真实但未知**的病理结构，是模型输入
- 模型产出的是 **θ(x) + 扩散物理结合后必然产生的几何指纹**
- 几何指纹有 6 条事先锁定的判据（H1–H6），每条都有 PASS/NULL/FAIL 解读

---

## 2. 核心方程（最小可行形式）

```
∂_t u(x, t) = f(u, v; θ(x)) + D∇² u + I_event(x, t) + η(x, t)
∂_t v(x, t) = g(u, v)
```

| 符号 | 朴素含义 |
|---|---|
| `u(x, t)` | 局部活动变量（点火 / 不点火的连续量） |
| `v(x, t)` | 恢复变量（不点火、休息中） |
| `θ(x)` | 空间非均匀的局部易激性——**模型预设输入**，反映真实但未知的病理结构 |
| `D∇² u` | 局部扩散（场电势 / ephaptic / 短程电耦合） |
| `I_event(x, t)` | 事件触发输入（外部 OR 内部冲动） |
| `η(x, t)` | 随机扰动，决定每次事件从哪个 seed 起 |

**关键 framing**（v1.0 反 tautology 锁）：

- `θ(x)` 是**模型输入**而非模型产出。我们不预测"θ(x) 长啥样"，只预测"**给定 θ(x) 非均匀，加上扩散物理与随机触发，必然产生哪些可观察的几何指纹**"
- 模型**不解释** HFO 80–250 Hz 振荡 carrier 的来源（PV interneuron ripple / ephaptic / population spike envelope 都有可能——那是另一层物理）
- 模型**只解释**通道点火的空间-时间次序

如果不写清楚 θ(x) 是输入：paper 评审会立刻打回 "你预设了高激场，然后预测高激场区域被反复采样——同义反复"。

---

## 3. 六条预测（H1–H6，pre-registered）

每条用三段式：**测了什么 / 怎么测的 / 揭示了什么**。所有数字判据 lock at framework time，**不允许事后调整**。

### 3.1 H1 — Endpoint 是结构化锚点而非随机散点（拆 H1a / H1b / H1c）

**测了什么**：endpoint 不是散布全脑的随机通道；而是**空间结构化**——分三个独立子测，避免与 H2 概念冲突。

**为什么必须拆**（v1.0.2 修订）：原 v1 把紧凑性定义为 `C_E = mean pairwise distance within (S ∪ K)`。但如果 H2 期望成立（source/sink 是同一条空间轴的两端），那 `S ∪ K` 本身**应该是 elongated 而不是 compact**——v1 度量会把"轴两端结构"误判为"不紧凑"，与 H2 互斥。v1.0.2 拆三层，**每层都与 H2 概念正交**。

**怎么测的**：

#### H1a — within-source compactness

- 每个 cluster 的 source 集合 `S = top-3 source channels`（k=3 lock）内部紧凑
- 度量：`C_S = mean pairwise distance within S`，3D Euclidean 主 + shaft-ordinal / cortical surface / SC distance 并列 sensitivity
- Null：matched random 3-channel sampling 1000 次；匹配条件 = shaft 分布 + participation rate + HFO rate（小 subject 退化规则同 §3.2 旧 H1）
- 检验：subject-level one-sided `C_S < C_S_null`；cohort Wilcoxon

#### H1b — within-sink compactness

- 每个 cluster 的 sink 集合 `K = bottom-3 sink channels` 内部紧凑
- 度量 / null / 检验：同 H1a，但作用在 K 上

#### H1c — endpoint envelope within pathological participation field

- 检验 `E = S ∪ K` 是否位于高参与场内部（不脱离 pathological field）
- 度量：
  - 主：`ratio = mean(d(endpoint_i, centroid(high-participation channels))) / mean(d(non-endpoint participating ch_j, same centroid))`
  - 备选 sensitivity：endpoint convex hull volume / participating channel convex hull volume
- 期望：`ratio ≈ 1`（endpoint 嵌在参与场内部）；`ratio >> 1` 意味 endpoint 脱离参与场
- Null：在 valid_mask=True 参与通道里随机抽 |E| 个，重算 1000 次

**揭示了什么**（H1 整体 PASS / NULL / FAIL 联合解读）：

- **PASS**：H1a 与 H1b 都 PASS（cohort Wilcoxon p<0.05 且 ≥ 3/4 距离同向）+ H1c 不 FAIL → endpoint 是结构化锚点
- **partial PASS**：H1a 或 H1b 一过一不过 → 单端紧凑、另一端散；可能 reverse template 不稳定，或 sink 定义出问题
- **NULL**：三层都不显著 → endpoint 作为锚点的几何意义弱
- **FAIL**（H1c FAIL：endpoint 大幅脱离参与场，`ratio >> 1`）→ endpoint 是分隔群体，**空间场假说受重创**

**敏感性**：每条 H1 子测都额外做 SOZ-mask 内随机抽样（控制 SOZ 选择偏差），并报告四种距离度量（Euclidean / shaft-ordinal / cortical surface / SC via DTI）下结论是否一致。

### 3.2 H2 — Source/sink reversal geometry (PR-6 swap_check ingest)

> **v1.0.3 修订（2026-05-22）**: H2 不在 Phase 1 内重新计算 — 直接 ingest PR-6 anchoring 已锁定的 `h2_swap_check` (per-contact endpoint Jaccard 反向 + 1000-perm null per subject; PR-6 plan §3.3 contract)。下方早期 v1.0.2 prose "主分析 1: Set-based reversal Jaccard" + "主分析 2: Spatial reversal centroid distance" + "Sensitivity: PCA axis projection" **已弃用**，保留只为 commit history 可追溯。
>
> **判读 tier lock**: PR-6 plan §3.3 + §15 → H2 是 **directional mechanism sanity, NOT cohort claim**（pre-registered tier，不允许 results 阶段升级）。每 subject 报 swap_score / null_p / exceeds_null_95th；cohort 报 sign-test #subjects exceed null_95th / n_testable + binomial one-sided p (p_null=0.05)；**不给 cohort-level PASS/NULL/FAIL 整合裁决**。
>
> **input cohort**: PR-2.5 `forward_reverse_reproduced` (split-half OR odd-even, per AGENTS.md Cross-PR contract) ∩ PR-6 audit eligible (endpoint defined, valid channels ≥ 6, SOZ JSON 等)，NOT PR-2 `candidate_forward_reverse_pairs`（PR-2 archive 明文："候选描述标签，不是最终机制判定"）。
>
> 设计错误三段叠加（CLAUDE.md §5+§6 经典）：(a) cohort 用错字段（candidate vs reproduced）；(b) 重新发明 PR-6 已有 swap_score helper；(c) 升级 mechanism-sanity tier 到 cohort claim。三条全部修正在 v1.0.3。

---

**以下为 v1.0.2 早期 prose（已弃用，保留 audit trail）**：

**测了什么**：正反两个模板的 source / sink **角色互换**——即正模板里 "最先点火的几个通道" 应该和反模板里 "最后点火的几个通道" **空间上 / 集合上**对应，反之亦然。

**为什么换设计**（v1.0.2 修订）：原 v1 主分析是 PCA 主轴投影。但 Tier 1 eligibility `n_participating ≥ 6` + endpoint k=3 时 `|S ∪ K|` 可吃掉全部 6 个 valid 通道，**排除 endpoint 后 PCA 主轴几乎一定不可定义**（剩 0–2 个点）。v1.0.2 把主分析改为**不需要轴拟合**的 set-based + centroid-distance 两条 reversal index 测验；PCA 投影降级为高通道 subject 的 sensitivity。

#### 主分析 1：Set-based reversal Jaccard

- 每个 fwd/rev cohort subject 算：
  - `J_swap = (Jaccard(S_A, K_B) + Jaccard(K_A, S_B)) / 2`（反向配对 — 期望高）
  - `J_same = (Jaccard(S_A, K_A) + Jaccard(K_B, S_B)) / 2`（同模板首尾 — 期望低，因 source 与 sink 概念上互斥）
- Reversal Index: `R_set = J_swap − J_same`
- 期望：reversal 成立时 `R_set > 0`
- **Null**：在该 subject 的 `E_A ∪ E_B` 通道池里随机重新分配 4 个 role tag（S_A / K_A / S_B / K_B 各 k 个，互不重叠），重算 1000 次得 `R_set_null`
- 检验：subject-level one-sided `R_set > R_set_null_median`；cohort Wilcoxon signed-rank

#### 主分析 2：Spatial reversal centroid distance

- 每个 subject 算 4 个 endpoint 集合的空间 centroid：`c(S_A), c(K_A), c(S_B), c(K_B)` (3D Euclidean)
- 两种距离和：
  - `d_swap = d(c(S_A), c(K_B)) + d(c(K_A), c(S_B))`（反向配对 — 期望小）
  - `d_same = d(c(S_A), c(K_A)) + d(c(K_B), c(S_B))`（同模板首尾 — 期望大，因 source/sink 是轴的两端）
- Spatial Reversal Index: `R_spatial = d_same / (d_swap + d_same)`
- 期望：reversal 成立时 `R_spatial > 0.5`
- **Null**：同上 role tag 重新分配 1000 次
- 检验：subject-level one-sided `R_spatial > R_spatial_null_median`；cohort Wilcoxon signed-rank

**两条主分析联合**：两条都 PASS 才算 H2 PASS（防止单条 false positive）；一条 PASS 一条 NULL 算 partial PASS；两条都 NULL 算 NULL。

#### Sensitivity：PCA axis projection（仅高通道 subject）

- 仅在 `non_endpoint_participating_channels ≥ 4` 的 subject 上跑（**这是新的子 cohort 准入条件，见 §5.3**）
- PCA 主轴用 `valid_mask=True \ (E_A ∪ E_B)` 通道定义（排除 endpoint 避免 double-dip）
- 四组 endpoint 投到主轴，检验 `proj(S_A) ≈ proj(K_B)` 且 `proj(K_A) ≈ proj(S_B)`
- 报告：合格 subject 比例 + 合格 subject 内的 axis-based verdict
- **不作主结论**，只作高通道 subject 的辅助验证；若 sensitivity 跑了且与主分析方向一致 → 强化 H2 PASS；若方向相反 → archive note + 主分析仍以两条 reversal index 为准

**揭示了什么**：

- **PASS**（两条主分析 cohort Wilcoxon 都 p<0.05 同向；如 sensitivity 跑了方向也同）：endpoint 在集合与空间两层面都呈反转关系——强支持"同一空间结构轴被相反方向读取"
- **NULL**（两条主分析都不显著）：endpoint 在两模板间无系统性 reversal 关系
- **FAIL**（方向反转：`R_set < 0` 或 `R_spatial < 0.5`）：endpoint 反而**靠近同极而非互换**——反转几何被证伪

### 3.3 H3 — Mark independence with stable geometry

**测了什么**：连续事件挑同一个模板的频率是不是兼容独立抛硬币；同时 endpoint 集合跨时间稳定。

**怎么测的**（继承 PR-7 addendum 2026-05-01 锁字）：

- 模板选择独立性（marginal-aware null-relative，**不**预设 marginal 50/50）：
  - lag-1 same-template excess（vs N2 marginal-preserving null）
  - 10s / 30s / 60s / 1800s window excess
  - run_length lift
  - cohort bootstrap 95% CI 与 δ_excess = 0.05 比较
- Endpoint 几何稳定性：
  - split-half endpoint set Jaccard recall
  - odd-even block endpoint set Jaccard recall

**揭示了什么**（H3 联合解读 — v1.0.2 verdict naming 修订）：

H3 的 verdict 标签**特别用** SUPPORTED / NOT SUPPORTED / CONTRADICTED，**不**用 PASS / NULL / FAIL——因为 "PASS for mark-independence" 会被简化为 "证明独立"（fail-to-reject ≠ accept null）：

- **SUPPORTED**：mark transition cohort CI ⊆ ±δ_excess（compatible with independence within tested precision）+ endpoint split-half recall ≥ 0.7 → 与 "随机触发 + 稳定空间骨架" 一致；**不是** "证明独立"
- **NOT SUPPORTED (geometry-unstable variant)**：mark transition compatible 但 endpoint 不稳定 → 模板几何本身不可靠，回 Phase 0 查事件检测 / 聚类质量
- **NOT SUPPORTED (memory variant)**：mark transition CI 偏离 ±δ_excess 但 endpoint 稳定 → 模型需要在 §2 方程里加 short-term memory / refractory / latent state 项；framework 仍活
- **CONTRADICTED**：cohort robust `|excess|` 显著 > δ_excess 且 leave-one-subject-out 仍 > δ_excess → 随机触发假说证伪；framework 改写或回退

**关键纪律**（继承 paper1_framework_sba.md §5.3 P3 v1.1.2 + PR-7 addendum 2026-05-01 lock）：

- H3 措辞硬锁 **"compatible with mark-independent sampling within tested precision"**
- **禁止**写 "证明独立" / "mark-independence PASS" / "证伪 ping-pong" 等 v1.0 错误语
- **禁止**用 leave-one-subject-out 替代 cohort 主判据
- v1.0.2 verdict naming（SUPPORTED 而非 PASS）从命名上就阻断 "PASS=证明" 的误读

### 3.4 H4 — Slow rate modulation 不变形 geometry

**测了什么**：当事件率慢漂（小时级）时，模板几何（centroid + endpoint set）是不是基本不变。

**为什么换设计**（v1.0.2 修订）：原 v1 用 `CV(rate) ≥ 3 × CV(geometry)` 硬阈值。问题：rate 和 geometry similarity 不同量纲；Jaccard / centroid similarity 的 CV 在小 N epoch 下不稳；硬锁 3× 是 framework time 锁了一个**可能没意义的比较**。v1.0.2 改成 "两个量各自对自己的 matched null 标准化后再比较"。

**怎么测的**：

- 把 24h（Yuquan）或多天（Epilepsiae）数据切成 1h–2h epoch（**保留时间顺序**，不打乱）
- 每 epoch 算：`rate(t)` + `template_centroid_similarity(t, global)` + `endpoint_Jaccard(t, global)`
- 计算 **normalized instability** per subject：
  - `I_rate = std(log(rate)) across epochs / sqrt(matched-null variance of log(rate))`
    - matched null: time-shuffle epoch order 1000 次，每次重算 std → 取分布的 variance 当作 baseline
  - `I_geom = std(1 − endpoint_Jaccard) across epochs / sqrt(matched-null variance)`
    - matched null: 每 epoch 内 role-shuffle endpoint 标签（在 valid_mask=True 池子里随机选 |E| 个当 "该 epoch 的 endpoint"），重算 1000 次得 null 分布
- subject-level 检验：`I_rate > I_geom`（one-sided）
- cohort 检验：Wilcoxon signed-rank on `(I_rate − I_geom)` per subject
- **效应大小**：Cohen's d on cohort `(I_rate − I_geom)` 值，median + IQR 同报；**不**锁固定倍数比

**揭示了什么**：

- **PASS**：cohort Wilcoxon p<0.05 同向 + Cohen's d ≥ 0.3（small effect floor）
- **NULL**：cohort Wilcoxon 不显著或 Cohen's d < 0.3 → rate 和 geometry 同尺度漂动，模型需把 geometry 也时变化
- **FAIL**：Cohen's d < 0 且 Wilcoxon p<0.05 反向 → geometry 比 rate 更不稳定，模型架构错

**联动对接**：PR-5 dominant-template absolute rate post-ictal 升高（已落 doc）应与 H4 PASS 一致——若 H4 PASS 而 PR-5 NULL，说明 rate 升高未到 geometry 变形阈值。**待 phantom-rank 修复后 §6.1 §5e 重跑确认 PR-5 方向是否保留**。

### 3.5 H5 — Endpoint identity shift around seizures（短时尺度，仅可测部分）

**测了什么**：靠近发作的时段（pre-ictal / post-ictal）里，是不是有原本不算 endpoint 的通道临时变成 endpoint，同时 HFO rate 也升高。

**为什么收紧合同**（v1.0.2 修订）：原 v1 只写 `ΔS_i > 0 AND ΔR_i > 0`，没说最小事件数、baseline 怎么挑、subject 内多 seizure 非独立怎么处理、通道多重比较怎么校正——这种合同会变成"看图说话"。v1.0.2 把统计合同显式锁住。

**统计合同（v1.0.2 lock，每一步都不可松动）**：

#### 数据 gate

- **最小事件数**：per channel × per window `n_events ≥ 30`；不达的 channel 在该 subject 的 H5 分析中被剔除
- **窗口定义**：pre-ictal `[−60min, −5min]`，post-ictal `[+5min, +60min]`，baseline 见下
- **窗口选择说明**：刻意比 Topic 2 PR-2.7 broad-rate 窗口 `[−6h, −1h]` 短——H5 测 endpoint identity 短时漂移，不测 broad rate 调制；H5 PASS 后可 sensitivity 用 `[−6h, −1h]` 验证 broad rate 是否同漂

#### Baseline 选择（time-of-day matched，不可松动）

每个 seizure，baseline 窗口规则：

1. 同一 subject 内
2. 与 pre-ictal / post-ictal 窗口同 hour-of-day ± 2h（控制 circadian）
3. 距任何 seizure ≥ 12h
4. 每 seizure 至少配 5 个独立 baseline 窗口

不满足任一条 → 该 seizure 从 H5 排除（不强凑）。

#### Subject 内多 seizure 处理（不可独立计数）

- 每 subject 多 seizure 时，**先 aggregate 到 subject-level**：对每 channel 算 `mean(ΔS_i across seizures)` + `mean(ΔR_i across seizures)`
- **禁止**把每次 seizure 当独立观测——multi-seizure 高度非独立（同 subject 同 SOZ）

#### 多通道比较（不可放过 FDR）

- 第一线 primary：subject-level **scalar statistic** = "fraction of channels with `ΔS_i > 0 AND ΔR_i > 0`"
  - 每 subject 输出一个 0–1 标量
  - H0 期望（ΔS / ΔR 各自独立、各自方向 50/50 chance）：fraction ≈ 0.25
- 备选 sensitivity：subject 内 BH-FDR across channels (q < 0.10) + count significantly-increasing channel 数
- **禁止**裸跑每 channel 单独 t-test 不校正

#### Cohort 检验

- Wilcoxon signed-rank on subject-level fraction `vs 0.25` (one-sided, fraction > 0.25)
- **Power floor**: 至少 6 个 subject 每个有 ≥ 1 qualifying seizure；不达 → 报告 "UNDERPOWERED, no verdict"，**不**给 PASS/NULL/FAIL
- 效应大小：Cohen's d on `(fraction − 0.25)` per subject

#### SOZ 关系（secondary，仅在 H5 主分析 PASS 时报告）

在 `fraction > 0.5` 的 subject 子集上（"明显有 endpoint shift" 的子集），查这些 ΔS>0 ΔR>0 channel 是否：

- 更靠近 clinical SOZ
- 更靠近 data-driven SOZ（`results/data_driven_soz/layer_b_labels/`，**当前不存在，待 PR-T3-1 Layer B 启动**；两个标签都查，不融合）
- 重叠 ictal early propagation channel
- 如果有个体 DTI：更接近 long-range SC out-degree 高的节点

**禁止**在 underpowered case 上做 SOZ 关系分析。

**揭示了什么**：

- **PASS**：cohort Wilcoxon `fraction > 0.25` p<0.05 + Cohen's d ≥ 0.3 + ≥ 6 qualifying subject，且 SOZ-related channel 富集 → 支持 "SOZ recruitment / 扩展" 机制
- **NULL**：cohort fraction 兼容 0.25 → 无系统性 endpoint identity shift
- **FAIL**：cohort fraction 显著 < 0.25 → 发作邻近反而 endpoint identity 更**不**变（与模型预测反向）
- **UNDERPOWERED**：< 6 qualifying subject → "暂无定论"，列入 future cohort 扩展 PR

**长尺度版本 H5-long 列存档**：跨录制周 / 月年级 endpoint 漂移需要纵向 cohort，**当前数据不可测**，作为 future work。Epilepsiae 部分 subject 有多次入院记录，未来如能整合可单独立 PR。

### 3.6 H6 — Participation-field 空间分隔（基础证伪）

**测了什么**：高参与（经常出现在事件里）通道和低参与（很少参与）通道，是不是空间上分得开——high participation 聚成一团，low participation 散在外围 / 远离 core。

**怎么测的**：

- 每通道 `P_i = #events containing i / #total events`（参与率）
- 空间分隔度量（三种并列）：
  1. 两群质心欧氏距离 vs null
  2. Moran's I 空间自相关（要求统计显著）
  3. Silhouette score（h vs l 标签下）
- **Null 构造 lock**：**shaft-stratified shuffle**（同一根电极杆内 shuffle 参与率）作为主 null；全脑 shuffle 作为敏感性
- **与 H1 独立**：H6 不需要 endpoint 定义，可以**先于** endpoint analysis 跑作为基础证伪

**揭示了什么**：

- **PASS**：参与场有空间结构 → 支持"空间组织化病理场"基础断言
- **NULL**：参与率到处都一样，没有空间组织
- **FAIL**（low 和 high 完全混杂）：**空间组织化病理场断言垮**，整个 framework 受重创——回到讨论是否换模型

---

## 4. Phase 1 失败模式矩阵（pre-registered）

H1 / H2 / H6 是 Phase 1 三条核心几何检验。它们的联合 verdict 决定 framework 后续走向。**Phase 1 实操顺序建议**：先跑 H6（基础证伪，不需要 endpoint）→ 过了再跑 H1（三层 H1a/H1b/H1c）→ 过了再跑 H2（两条 reversal index 主分析 + PCA sensitivity）。

**H1 整体 verdict 规则**（v1.0.2 新加）：

- H1 = PASS：H1a + H1b 同时 PASS（≥ 3/4 距离同向）+ H1c 不 FAIL
- H1 = partial PASS：H1a 或 H1b 一过一不过 + H1c 不 FAIL
- H1 = NULL：三层都不显著
- H1 = FAIL：H1c FAIL（endpoint 大幅脱离参与场，`ratio >> 1`），不论 H1a / H1b

**H2 整体 verdict 规则**（v1.0.2 新加）：

- H2 = PASS：两条 reversal index 主分析（set-based + spatial centroid）cohort Wilcoxon 都 p<0.05 同向
- H2 = partial PASS：一条主分析 PASS 一条 NULL
- H2 = NULL：两条主分析都不显著
- H2 = FAIL：方向反转（`R_set < 0` 或 `R_spatial < 0.5`）
- PCA sensitivity (Tier 1+ sub-cohort)：仅作辅助验证，方向同主分析 → 加强；方向反 → archive note，**不**翻主分析

**Phase 1 失败模式表**（H6 × H1 × H2 三轴 × 5 verdict 行）：

| H6 (participation 分隔) | H1 (endpoint 锚点) | H2 (reversal 几何) | Framework 解读 | 后续动作 |
|---|---|---|---|---|
| PASS | PASS | PASS | 完整证据链：空间场 + 结构化锚点 + 反转读取 | 进 Phase 2 |
| PASS | PASS / partial | NULL / FAIL | 空间场 + 锚点成立，但反转不成立 | endpoint 可能是两个独立稳定锚点；H2 改写为"双锚点 not 同轴反转"；framework 仍活 |
| PASS | NULL | PASS | 空间场成立、reversal 成立，但 endpoint 不构成结构化锚点——内部矛盾 | 回 Phase 0 查 endpoint 定义 + phantom rank 是否漏修 |
| PASS | FAIL (H1c) | * | endpoint 大幅脱离参与场 | endpoint operational definition 需重新设计；framework 部分活，Phase 2/3 可继续但 endpoint 概念需重塑 |
| NULL / FAIL | * | * | **基础断言（空间组织化病理场）证伪** | **framework 整体回退**；SEF-ITP 名字作废；回到讨论是否换模型 |

**重要**：任何 cell 落到 NULL / FAIL 都必须按表执行后续动作。**禁止**事后调整 H1 / H2 / H6 数字判据来"挽救" verdict。

---

## 5. Cohort tier 定义（数字化，pending Phase 0）

继承 AGENTS.md 的 cohort tier 纪律。所有 Phase 1+ 数字必须标 Tier 1 / Tier 2，**禁止裸写** "30/30" 或 "35/40"。

### 5.1 Tier 定义（criterion 措辞 lock，数字 TBD pending Phase 0）

**Tier 1（主预测 cohort）**：

- valid-only 重跑后 `stable_k = 2`
- `forward_reverse_reproduced = True`（split-half OR odd-even，继承 CLAUDE.md cross-PR contract）
- 每个 cluster `n_participating_channels ≥ 6`（H1 / H2 主分析最低要求；endpoint k=3 占满 6 个通道仍允许 H1a / H1b 与 H2 set/spatial reversal 跑）
- SEEG/ECoG 3D 坐标可用
- 数字 N = **TBD**（待 Phase 0 `results/interictal_propagation_masked/` 完成后填；当前 phantom-rank-contaminated 版本约 6–8 subject 进 fwd/rev cohort，masked 重跑后预计同量级，但 lock 必须以 valid-only 实跑为准）

**Tier 1+ (H2 PCA sensitivity sub-cohort)**：

- 满足 Tier 1 所有条件
- **额外**：`non_endpoint_participating_channels ≥ 4` per cluster（即 `|valid_mask=True ∩ ¬(S ∪ K)| ≥ 4`）
- 用于 H2 sensitivity PCA axis projection（主分析 reversal index 不需要此条件）
- 数字 N = **TBD**（待 Phase 0；这是 Tier 1 的子集）

**Tier 2（敏感性 cohort）**：

- valid-only 重跑后 `stable_k = 2`
- 不强制 forward/reverse 复现
- 数字 N = **TBD**（best estimate per Topic 0 §3.1：40-subject cohort 中 1 例 stable_k 翻转（epilepsiae_916 2→4），其余 39/40 stable_k 不变 → **Tier 2 ≈ 34 预期，pending lock-in**；高 k cohort 已被独立排除）

**Excluded**：

- valid-only 重跑后 `stable_k ≠ 2`（已知 epilepsiae_916 在 Topic 0 cohort 中 2→4 翻转）
- 参与通道 union 不足
- phantom-rank 修复后 endpoint 集合手动审计不通过

### 5.2 规则

- 主文档 / archive / paper 任何引用 Phase 1+ 数字时**必须**标注 Tier 1 / Tier 2
- Phase 0 完成后 cohort tier 数字 lock，**不允许**为了 H1/H2 显著性反向调整
- 任何 Tier 1 与 Tier 2 数字方向不一致的情况，**主结论以 Tier 1 为准**，Tier 2 作 sensitivity 报告

---

## 6. Phase 0–4 实施路线

### 6.1 Phase 0 —— Phantom-rank 修复 + broad re-derivation 【硬前置】

**目标**：把所有 endpoint / cluster / template centroid 的地基修好。

**关键 bug**：`lagPatRank` 现有数据里没参与某事件的通道也被赋予了有限的 int rank（U-型污染，两端 rank=0 / rank=N−1 都可能被噪声塞通道）。详情：`docs/topic0_methodology_audits.md` §3.1。

**任务清单**（继承 topic0 §5 重跑路线图）：

- [ ] **5a** PR-2 re-cluster on masked → `results/interictal_propagation_masked/per_subject/`
- [ ] **5b** PR-2.5 split-half / odd-even on masked → 同上
- [ ] **Checkpoint A** advisor consult: stable_k flip + forward/reverse 复现集合变化
- [ ] **5c** PR-3 per-cluster MI on masked labels
- [ ] **5d** PR-4A/B/C/D on masked labels
- [ ] **Checkpoint B** advisor consult: PR-4B/D 方向是否反转
- [ ] **5e** PR-5 / PR-5-B on masked（H4 验证基础）
- [ ] **5f** PR-6 endpoint anchoring / swap / Step 6 / rank displacement on masked（H1 / H2 准备）
- [ ] **5g** PR-7 antagonistic pairing on masked（H3 验证基础；**P3 翻转 = framework-level revision，必须停下来 review，不允许默默改写**）
- [ ] **5h** Topic 4 attractor on masked → `results/topic4_attractor_masked/`
- [ ] **5i** cohort tier audit 表 lock：Tier 1 N / Tier 2 N / Excluded N + 数字 lock 到本文 §5

**前置纪律**：

- Phase 1 在 Phase 0 完成前**不启动**
- Phase 0 进度看板维护在 `docs/topic0_methodology_audits.md` §5 重跑路线图
- 如果 Phase 0 实际推进缓慢，本 framework 处于 **"等待修复完成的占位文档"** 状态——这一状态在本文 §0 + paper_overview 上必须诚实声明，**不掩饰**

### 6.2 Phase 1 —— 空间几何检验（免费，不需要建模）

**目标**：先证明"空间场"假设的必要性。

**前置**：Phase 0 完成 + Tier 1 cohort + SEEG 3D 坐标可用。

**Runner 入口（2026-05-21 完工）**：

```
scripts/run_sef_itp_phase1.py --dataset <epilepsiae|yuquan> --subject <sid> \
    --distance-metric euclidean --hypothesis all \
    --output-dir results/topic4_sef_itp/per_subject
```

`load_subject_for_phase1(<phase0a_json_path>)` 把以下 4 个数据源接成 `SubjectPhase1Data`：

| 源 | 路径 | 提供 |
|---|---|---|
| 1. masked Phase 0a JSON | `results/interictal_propagation_masked/per_subject/<dataset>_<sid>.json` | `channel_names`, `adaptive_cluster.candidate_forward_reverse_pairs` |
| 2. masked PR-6 anchoring JSON | `results/interictal_propagation_masked/template_anchoring/per_subject/<dataset>_<sid>.json` | `per_template[].source/sink/valid_mask`（端点为通道名，loader 转 index）|
| 3. lagPat NPZ | yuquan: `/mnt/yuquan_data/yuquan_24h_edf/<sid>/`<br>epilepsiae: `/mnt/epilepsia_data/.../all_data_lns/<sid>/all_recs/` | `events_bool` via `src.interictal_propagation.load_subject_propagation_events` |
| 4. SEEG 坐标 | yuquan: `chnXyzDict.npy`<br>epilepsiae: SQL `electrode.coord_*` + MNI152 1mm MRI affine | `coords (n_ch, 3)` via `src.seeg_coord_loader.load_subject_coords`；mm 强制断言 |

合同关键条款（通道对齐 / 端点 name→index / mm 单位断言 / valid_pool 交集 / pair schema 翻译）+ 8 个集成测试用例 + 完整 results dir layout 见 [`docs/archive/topic4/sef_itp_phase1/phase1_runner_contract_2026-05-21.md`](archive/topic4/sef_itp_phase1/phase1_runner_contract_2026-05-21.md)。所有集成测试 GREEN（2026-05-21）。

实跑 smoke test（7 subject：epilepsiae 1073/548/922/590/1150/1077/1146，yuquan chengshuai）全部产出真实裁决，无 pipeline error。`coord_provenance.normalization_certainty` 字段如实记录 `grid_confirmed_warp_type_unverified`（Epilepsiae）/ `subject_native`（Yuquan），未来 warp field 文档到位再升级到 `mni_normalized_verified`。

**任务顺序**（按基础证伪 → 详细验证）：

1. **H6 participation-field segregation**（优先跑——基础证伪）
2. **H1 endpoint compactness**（四种距离度量并列；matched null）
3. **H2 source/sink reversal geometry**（PCA 主轴独立于 endpoint）
4. **SOZ relation as secondary**：endpoint 与 clinical SOZ + data-driven SOZ（PR-T3-1 输出）并列报告 overlap / distance / enrichment；**不融合**，不作模型硬前提

**产出图**（按 AGENTS.md figures README 中文规范）：

- 每 subject endpoint spatial map（含正反两模板叠图）
- endpoint compactness vs random null violin plot（cohort）
- reversal-axis projection scatter（four endpoint groups on PCA axis）
- participation spatial heatmap（每 subject 一张）
- cohort summary forest plot（H1/H2/H6 effect sizes）

**产出目录**：`results/topic4_sef_itp/phase1_spatial_geometry/{per_subject,cohort_summary.json,figures/,soz_relation/}` — 整个 root 在 Phase 1 启动时新建。完整 dir layout 见上述 archive。

### 6.3 Phase 2 —— Temporal × geometry 联合检验

**目标**：证明"随机触发 + 稳定几何骨架"是否成立。

**前置**：Phase 1 H6 + H1 至少 PASS。

**任务**：

- H3 mark independence + endpoint split-half/odd-even recall（继承 PR-7 addendum 形式）
- H4 CV(rate) vs CV(geometry) across 1h–2h epoch
- 联合 verdict 表（参见 §3.3 H3 揭示什么）

**产出目录**：`results/topic4_sef_itp/phase2_temporal_x_geometry/`

### 6.4 Phase 3 —— 发作邻近 endpoint identity shift

**目标**：H5 短时版本检验。

**前置**：Phase 1 + Phase 2 全 PASS（否则发作邻近也不可信）。

**任务**：

- baseline / pre-ictal / post-ictal endpoint score + HFO rate 联合变化
- 与 clinical SOZ + data-driven SOZ + ictal early channel 的空间关系
- 如有个体 DTI：与 long-range SC out-degree 的关系

**产出目录**：`results/topic4_sef_itp/phase3_ictal_adjacent/`

### 6.5 Phase 4 —— 最小 2D FHN neural-field toy（**最后才做**）

**目标**：只复现 Phase 1–3 数据已支持的现象，**不做大而全拟合**。

**前置**：Phase 1 + Phase 2 至少 H6 + H1 + H3 PASS；Phase 3 至少 NULL（不是 FAIL）。

**第一版模型只包含**：

- 2D grid（或 cortical sheet 抽象）
- 非均匀 θ(x)：**v1 toy 必须用 generic structured 非均匀场（例如随机高斯随机场 + 单一 hot zone），不是 patient-specific 拟合**。如果 θ(x) 从 Phase 1 单 subject 数据反推，模型立刻退回 BHPN 拟合机器同类问题。**patient-specific θ(x) 拟合若启动，必须单独立 PR 并通过 framework-level review**
- 局部扩散 `D∇² u`
- 单一节点类型：**FHN**（不加 Hindmarsh-Rose / Epileptor）
- 随机 event trigger η(x, t)
- 慢变量 `s_slow(t)` 控制 event probability
- **不**模拟 HFO carrier 80–250 Hz 振荡（FHN 自然频率 ~10 Hz，**不强行匹配**——HFO carrier 是另一层物理）

**模型只需要复现**：

- 稳定 endpoint（对应 H1）
- source/sink reversal（对应 H2）
- mark-independent-like template sampling（对应 H3）
- rate 变化大于 geometry 变化（对应 H4）
- 如果加入慢变 θ(x, t)：可产生 endpoint identity shift（对应 H5）

**Phase 4 显式不做**：

- patient-specific fitting（→ future PR）
- HR / Epileptor 多节点类型（v1 toy 只 FHN）
- 5 dumb baselines 比较（→ 未来 PR-T4-2 等价物，仅当走到那一步）
- HFO carrier 振荡机制
- 跨 subject 参数共享

**产出目录**：`results/topic4_sef_itp/phase4_fhn_toy/`，含 figures + sanity JSON + plain-language README

---

## 7. 模型支持 / 不支持的解释

### 7.1 模型能解释的现象

1. 为什么同一 subject 内存在稳定传播模板（→ θ(x) 非均匀 + 扩散物理）
2. 为什么正反模板共享 source/sink 几何骨架（→ 同一空间轴的反向读取）
3. 为什么模板选择在已测试尺度上近似随机（→ η(x, t) 决定每次 seed）
4. 为什么 event rate 可受慢变量调制，而模板几何相对稳定（→ s_slow 调 `I_event` 不调 θ(x)）
5. 为什么某些 endpoint-like 节点可能和 SOZ 进展 / 发作招募有关（→ θ(x, t) 慢变可产生 endpoint identity shift）
6. 为什么局部异常场遇到长程 SC-rich 区域时可能 transition 到大范围发作（→ 长程 SC 跳线被纳入扩散网络后递归招募；与 Abbott 的抑制波前崩溃说**互补而非冲突**）

### 7.2 模型暂时**不支持**的强说法（写死 out-of-scope）

1. ❌ endpoint 就是 SOZ 边界
2. ❌ endpoint 一定在 SOZ 内
3. ❌ endpoint 一定是 seizure onset channel
4. ❌ source 是兴奋驱动，sink 是抑制反弹（**HFO 80–250 Hz 不分 E/I**，红线）
5. ❌ 正反模板证明了双稳态 attractor（H3 仅声明 "compatible with mark-independent within tested precision"）
6. ❌ 模板选择已经被证明完全独立（同上）

---

## 8. 红线（继承 paper1_framework_sba.md v1.1.2 红线 + 本框架新增）

1. **HFO 80–250 Hz 不分 E/I**：framework / archive / paper / toy / 任何 plan 全文严禁出现 "兴奋驱动 / 抑制反弹 / 证明机制" 表述
2. **H3 措辞锁**："compatible with mark-independent sampling within tested precision"；禁止写 "证明独立"，禁止用 leave-one-subject-out 替代主判据
3. **不预设 clinical SOZ = true SOZ**：H1/H2/H5 的 endpoint 不被假定为 SOZ 边界 / SOZ 内 / seizure onset；与多源 SOZ proxy 关系并列报告
4. **不预设 θ(x) 自然涌现**：θ(x) 是模型预设输入；模型预测的是 θ(x) + 物理产出的几何指纹，**不是** θ(x) 本身（v1.0 反 tautology 锁）
5. **directional predictor 用 sin 不是 cos**：如果未来本 framework 扩展到 P5 等价物（间期 → 发作 directionality），必须用 sin-based 或 rank-based，**禁止** cos-based directed graph（继承 SBA v1.1）
6. **k = 3 主预测，k ∈ {2, 4, 5} 敏感性**：endpoint 定义的 k 在 framework time lock，**禁止**事后调整
7. **H1 三层拆分 lock**：H1 必须拆 H1a（within-source compactness）+ H1b（within-sink compactness）+ H1c（envelope within participation field）；**禁止**回退到 `mean pairwise distance within (S ∪ K)` 的 v1 单一度量（与 H2 概念冲突）
8. **H2 主分析 = 两条 reversal index lock**：set-based Jaccard + spatial centroid distance 是 H2 主分析；PCA axis projection 只能作 Tier 1+ sensitivity；**禁止**把 PCA 升为主分析（Tier 1 工程上跑不动）
9. **H3 verdict naming = SUPPORTED / NOT SUPPORTED / CONTRADICTED**：**禁止**用 PASS / NULL / FAIL（防止 "PASS = 证明独立" 误读）
10. **H4 normalized instability + Cohen's d ≥ 0.3** lock：**禁止**回退到 `CV(rate) ≥ 3 × CV(geometry)` 的 v1 硬阈值（量纲不同 + CV 不稳）
11. **H5 统计合同 lock**：min event ≥ 30 / time-of-day matched baseline / subject-level aggregation / fraction-vs-0.25 Wilcoxon / power floor 6 subject；**禁止**裸 channel-level t-test 或不校正 FDR
12. **H6 shaft-stratified shuffle 主 null**：lock at framework time，**禁止**事后换 null 形式

---

## 9. 与现有 framework / topic doc 的关系

### 9.1 SBA framework 取代关系

- **取代** `docs/paper1_framework_sba.md` §4（BHPN-toy 数学规范）
- **取代** `docs/paper1_framework_sba.md` §6（fitted model BHPN-fit spec 的 toy mechanism 子句）
- **取代** `docs/paper1_framework_sba.md` §7（5 dumb baselines 中针对 toy 机制的项）
- **保留** `docs/paper1_framework_sba.md` §5.1 P1（template stability，**待 Phase 0 valid-only 重跑后确认**）
- **保留** `docs/paper1_framework_sba.md` §5.2 P2（geometric backbone，**待 Phase 0 valid-only 重跑后确认**）
- **保留** `docs/paper1_framework_sba.md` §5.3 P3 PR-7 addendum 锁字 → 直接映射为本文 H3
- **保留** `docs/paper1_framework_sba.md` §5.4 P4 anatomical anchoring（部分被本框架 H5 短时版本覆盖，长时 P4 PR-8 v2 路径仍单独存在）
- **保留** `docs/paper1_framework_sba.md` §5.5 P5 directional predictor 红线（sin 不是 cos）

### 9.2 Topic 1 / Topic 3 / Topic 4 文档双向链接

- `docs/topic1_within_event_dynamics.md` §2 当前结论加入 SEF-ITP framework 入口链接
- `docs/topic3_spatial_soz_modulation.md` §2 加入与 SEF-ITP H5 关系说明（endpoint identity shift × data-driven SOZ）
- `docs/topic4_sef_itp_framework.md`（本文）= Topic 4 模型层 formal entry
- `docs/paper_overview.md` 索引在 Topic 4 行加入本文链接

### 9.3 Archive 关系

- `docs/archive/topic4/pr_t4_1_bhpn_toy/pr_t4_1_bhpn_toy_plan_2026-05-01.md` 顶部加 **SUPERSEDED** banner 指向本 framework
- `docs/archive/topic4/INDEX.md` 更新指向本 framework（formal entry 不再是 "待写"）

### 9.4 PR-T3-1 数据驱动 SOZ 的关系

- PR-T3-1 Layer B 输出（`results/data_driven_soz/layer_b_labels/`，**当前不存在，待 PR-T3-1 Layer B 启动后产出**；Layer A 已落地于 `results/data_driven_soz/layer_a_ictal_er_rank/`）是 H5 检验的**第二份独立 SOZ 标签**（避免循环验证）
- 任何 H5 / H1 与 SOZ 关系的报告**必须**同时引用 clinical SOZ + data-driven SOZ 两个来源，不融合
- PR-T3-1 v2.2.3 `lambda_fragile` flag 必须随标签一起传递到 H5 下游 reader

---

## 10. 中心段落（草稿，供未来 paper / abstract 使用）

> 我们假设，间期群体 HFO 传播模板反映的是一个空间组织化的病理易激场被反复采样。模型中的 θ(x) 代表真实但未知的病理空间结构，是模型输入而非模型产出；模型真正预测的是 θ(x) 与局部扩散、随机触发共同作用后产生的传播模板几何、source/sink 反转结构和时间采样规律。稳定的 endpoint 不被假定为 SOZ 边界，也不被强制假定为 SOZ 内部节点，而是病理场中的几何锚点。正反模板来自同一局部高易激轴在不同随机初始条件下被相反方向读取。慢变量主要调节事件招募概率，而模板几何相对稳定。当前数据可检验的是小时尺度的 endpoint identity shift；月年尺度的 SOZ 漂移需要纵向 cohort，暂不作为主文预测。

英文版（供 abstract / 投稿）：

> We hypothesize that interictal group-HFO propagation templates reflect repeated sampling of a spatially organized pathological excitability field. The local excitability θ(x) is a model input reflecting the unknown true pathological structure; the model's predictions concern the geometric and temporal organization of templates emerging from θ(x) + local diffusion + stochastic event initialization, not θ(x) itself. Stable source/sink endpoints are not assumed to be SOZ boundaries or restricted to SOZ interior; rather, they are geometric anchors embedded in or near the pathological field. Forward/reverse template pairs arise when the same local high-excitability axis is read out in opposite directions under stochastic initial conditions. Slow modulation changes the probability of event recruitment while the underlying template geometry remains relatively stable. Short-timescale endpoint identity shifts around seizures (hours) are testable in current data; long-timescale drift (months) requires longitudinal cohorts not yet available.

---

## 11. 自检清单（v1 lock 状态）

- [x] 单核心断言清晰（H0 病理易激场 + 扩散 + 随机触发，θ(x) 是输入不是产出）
- [x] H1–H6 每条有 verdict 标签 + 数字判据 lock + 三段式描述
- [x] H1 拆三层（H1a within-source / H1b within-sink / H1c envelope-within-field）避免与 H2 概念冲突
- [x] H1 endpoint k=3 主预测 + {2, 4, 5} 敏感性 lock
- [x] H1 四种距离（Euclidean / shaft-ordinal / cortical surface / SC）并列报告 lock
- [x] H2 主分析 = set-based Jaccard reversal + spatial centroid reversal index（两条都跑，都 PASS 才算 H2 PASS）
- [x] H2 PCA axis projection 降为 Tier 1+ sensitivity（要求 `non_endpoint_participating ≥ 4`），不作主分析
- [x] H3 verdict naming = SUPPORTED / NOT SUPPORTED / CONTRADICTED（不用 PASS 防误读）
- [x] H3 措辞锁 "compatible with mark-independent within tested precision"（继承 PR-7 addendum 2026-05-01）
- [x] H4 normalized instability vs matched null + Cohen's d ≥ 0.3 lock（v1 CV ≥ 3× 已废）
- [x] H5 统计合同 lock：min event 30 + time-of-day matched baseline + subject aggregation + fraction-vs-0.25 + power floor 6
- [x] H5 短时（pre/post-ictal, hours）vs 长时（months）拆分；长时遗留 future work
- [x] H6 shaft-stratified shuffle 主 null + 全脑 shuffle 敏感性 lock
- [x] H6 优先级最高（基础证伪），实操顺序 H6 → H1 → H2 写入 §4
- [x] Phase 1 失败模式表 pre-register（H6 × H1 × H2 三轴 × 5 verdict 行）
- [x] Cohort tier criterion 措辞锁，数字 TBD pending Phase 0
- [x] Tier 1+ sub-cohort（H2 PCA sensitivity）单独 criterion lock
- [x] Phase 0 phantom-rank 修复硬前置，禁止跳过
- [x] HFO carrier 振荡机制不在本 framework 范围（红线 §8.1）
- [x] θ(x) 是模型输入而非产出，避免 tautology（红线 §8.4）
- [x] Phase 4 最小 toy 只用 FHN + generic structured θ(x)（不 patient-fit），不加 HR / Epileptor
- [x] 与 PR-T3-1 数据驱动 SOZ 的双标签合同写明
- [x] SBA framework 取代 / 保留范围明确（§9.1）
- [x] Topic 1 §2 + Topic 3 §2 实际加入 SEF-ITP 链接（v1.0.2 修订实际落字）
- [x] Out of scope 包括 Paper 2 / Paper 3 / E/I / patient-specific fitting
- [x] CLAUDE.md §8 大白话风格全文遵循（codename 仅作括号补注）

---

## 12. 一句话承诺（结尾）

我们把 Topic 4 的模型层从 **"塞 Hebbian 矩阵让 Kuramoto 演化得到预设吸引子"** 换成 **"假设一个空间组织化的病理易激区，看它被扩散物理反复采样后必然留下的几何指纹"**。先**免费**在数据上查 6 条指纹（H1–H6：空间紧凑 / 反向几何 / 独立采样 / 慢漂解耦 / 发作邻近漂移 / 参与场分隔），全部 PASS 才上模型代码（Phase 4 最小 FHN toy）。每条假设失败都有明确预先注册的后续动作。Phantom-rank bug 修复（Topic 0 §5）是所有 Phase 的硬前置，没修复任何数字都不可信。

---

## 13. 历史文档索引

- `docs/paper1_framework_sba.md` v1.1.2 + PR-7 addendum 2026-05-01 lock —— 上游 SBA framework；本框架取代其 BHPN-toy 部分
- `docs/archive/topic4/pr_t4_1_bhpn_toy/pr_t4_1_bhpn_toy_plan_2026-05-01.md` —— **SUPERSEDED**，BHPN-toy plan-of-record v2，归档
- `docs/archive/topic4/layered_model_framework.md` —— 更早的分层模型框架（已被 SBA 取代）；保留为历史
- `docs/topic0_methodology_audits.md` §3.1 + §5 —— phantom-rank 修复 + broad re-derivation roadmap（本框架硬前置）
- `docs/archive/topic1/propagation/topic4_attractor_diagnostics_step1_results_2026-05-10.md` —— Topic 4 attractor Step 1 结果（principal curve + λ₂ transition）；待 Phase 0 §5h 在 masked features 上重跑
- `docs/topic3_spatial_soz_modulation.md` —— Topic 3 PR-T3-1 数据驱动 SOZ（H5 第二标签来源）
