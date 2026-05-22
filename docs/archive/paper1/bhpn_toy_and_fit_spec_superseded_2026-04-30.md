# BHPN-toy + BHPN-fit Spec (Superseded by SEF-ITP, Archived 2026-05-22)

> **归档说明**：这是 `docs/paper1_framework_sba.md` §4（BHPN-toy 数学规范）与 §6（BHPN-fit spec）的完整原文，两者于 2026-04-30 lock 在 v1.1.2，**于 2026-05-20 被 SEF-ITP framework (`docs/topic4_sef_itp_framework.md` v1.0.2)取代**。
>
> 取代理由（见 topic4_sef L5）："SEF-ITP framework 取代 paper1_sba §4 toy model 数学规范 + §6 fitted model BHPN-fit spec 的 toy-mechanism 子句 + §7 dumb baselines 中针对 toy 机制的项"。
>
> **保留以下不被 SEF-ITP 取代的部分**仍在 paper1_sba 主 doc：P1 / P2 实证 verdict + P3 PR-7 addendum 锁字 + P5 directionality (sin-based) 红线 + 单核心假设 H₀ + 5 sharp predictions 总表 + 失败模式表 + writing flow + cross-doc links。
>
> 本归档只为历史溯源 + 可能的 BHPN baseline 比对参考；任何当前 modeling 设计以 SEF-ITP 为准。

---

## §4 Toy model — BHPN-toy（最小数学形式）

> **目的**：在最小数学复杂度下证明 SBA hypothesis 是**一个**合理 mechanism——而不是唯一 mechanism。toy model 不拟合任何真实 subject 数据，只演示机制可行性 + 校验 PR-7 / fwd-rev 等硬约束在该 mechanism 下能否同时被满足。

### 4.1 节点动力学

N = 8–12 节点（匹配单 SOZ + nearby 典型 SEEG 通道数），每节点 phase oscillator at HFO band：

```
dθ_i/dt = ω_i + Σ_j A_ij · sin(θ_j - θ_i) + ξ_i(t)
```

- `ω_i ~ N(ω_0, σ_ω)`，`ω_0` 设在 HFO 频段中段（~110 Hz 量级）
- `ξ_i(t)`：弱白噪声
- 时间尺度：每次"event"模拟 ~50–200 ms（与真实 HFO event 长度一致）

### 4.2 连接矩阵（Hebbian 存储）

存储一个目标 phase 模式 `φ* = [φ_1*, ..., φ_N*]`（来自一个 sequential rank pattern）：

```
A_ij = ρ · cos(φ_i* - φ_j*)        (symmetric Hebbian)
```

由 Hopfield-Kuramoto 已知性质：对称 cosine 连接同时支持 `φ*` 与 `π − φ*`（forward / reverse）作为 fixed-point attractor。**这是数学事实，不是经验拟合**。

`ρ` 是单个 coupling gain，**不**对每个 i,j 独立调参。

**严格区分（v1.1 lock）**：
- **A_ij 描述的是动力学耦合**（symmetric storage matrix），cos 是偶函数，**不**承担方向预测。
- **方向信息 在 phase pattern φ\* 中**：从 φ\* 派生的 `D_ij = sin(φ_j\* − φ_i\*)`（sin 反对称）或 `R_ij = sign(rank_j − rank_i)` 才是 directional predictor。
- P5（间期 → 发作 directionality）使用 D_ij / R_ij，**不**使用 A_ij。任何把 cos-based A_ij 当 directed graph 的 prediction 都是 v1.0 的错误，v1.1 已根除。

### 4.3 核节点 heterogeneity（角色型 core）

在 N 节点中标记 `k=2–3` 个为 "core node"。Core 由两种参数化之一定义（toy 阶段任选一种，最小化自由度）：

- **方式 A**：`ω_core` 略高于 `ω_other`（intrinsic excitability 偏置）
- **方式 B**：core 节点 plasticity gain `γ_core > γ_other`（在长期学习阶段更易被纳入模板）

**关键**：core 是 *operational* 定义（参数差异），**不**预设其在真实数据中的解剖位置。Toy model 自然产生 swap node = core node 子集这一性质，再让 fitted model 预测该角色节点应当与 SOZ 显著重叠（P4）。

### 4.4 Attractor selection 与 rate modulation（v1.1 修订：两过程统计独立）

v1.0 把 mark-independence 与慢调制混在同一个 latent state s(t) 里，导致内部时间尺度矛盾（"τ_s ≪ 10s" 与 Topic 2 min-hour modulation 不兼容）。v1.1 显式分离为两个**统计独立**的过程：

**Process A — Event rate modulation s_rate(t)**（min–hour 尺度，与 Topic 2 一致）：

```
ds_rate/dt = -s_rate/τ_rate + η(t)     (OU, τ_rate ~ 10–10000 s)
```

s_rate(t) 调制 retrieval **触发率**（Poisson 率 λ(t) ∝ exp(s_rate(t))），但**不**调制 attractor identity。s_rate(t) 存在的目的是匹配 Topic 2 已观察到的 broad-band slow rate modulation，使 toy 同时与事件间慢漂事实兼容。

**Process B — Per-retrieval attractor identity ε_id**：

每次 retrieval 时初始相位 θ_i(0) ~ uniform[0, 2π]（**i.i.d. across retrievals**），系统沿 BHPN dynamics 落入对称双稳态 basin 之一。在**纯对称 Hebbian toy** 中，basin 几何对称 ⇒ P(forward) ≈ P(reverse) ≈ 0.5（这是 toy 内部 sanity check 的 target，详见 T3 / T5）；**ε_id 跨 retrievals 独立——这是 mark-independence 的唯一源头**。

**Toy vs fitted/real-data marginal asymmetry（v1.1.2 重要 caveat）**：真实数据 cohort marginal P(fwd) 中位 0.498、范围 [0.20, 0.61]（PR-7 addendum, 548 极端 P(fwd)=0.20）。BHPN-fit 必须允许 basin asymmetry（来源可能是：(i) 近似但非完全对称的 Hebbian connection；(ii) 异质 core node 让一个 attractor 偏深；(iii) initial phase 分布弱偏置；(iv) external rate-state 偏置某 basin），但**保持 ε_id i.i.d. 的核心 mechanism 不变**。**P3 / F2 等价性检验对准的是给定 marginal 之后的 i.i.d. 偏离**（null-relative excess vs N2 marginal-preserving null），**不**预设 marginal 50/50。Toy 50/50 是 toy 自身实现的 sanity；real-data marginal asymmetric 不与 SBA mechanism 冲突。

**Refractory τ_ref**：retrieval 完成后短时窗 ~ event duration（50–200 ms）内 phase 重置为 random 且禁止再触发，仅防止单次 retrieval 被两次计数。τ_ref 远短于 PR-7 检验窗（10s–60min），**不**贡献任何 reciprocity 信号。

**关键设计（lock at v1.1）**：mark-independence **不依赖 s_rate(t) 的时间尺度**。即使 s_rate(t) 在 min–hour 尺度上慢漂（与 Topic 2 一致），attractor identity 仍因 random initial phase 保持 i.i.d.。这一架构同时满足 C6 (PR-7 NULL, 10s–60min) 与 Topic 2 慢率漂移观测，**无内部时间尺度矛盾**。s_rate 调 rate，ε_id 调 identity，二者解耦。

### 4.5 Toy model 必须再现的最小事实清单（v1.1）

| # | 观测 | 对应硬约束 |
|---|---|---|
| T1 | 系统在两个对称稳定相位排序之间收敛（fwd / rev） | C1, C2 |
| T2 | 多次 retrieval 的相位排序聚为 k=2 cluster 主导 | C2, C3 |
| T3 | 在固定 s_rate 条件下、**large-N simulation (N ≥ 1000 retrievals)**，连续 retrievals 的 attractor identity 独立采样：lag-1 same-label 率 → 0.5（CI ⊆ [0.45, 0.55]）；run-length 分布 → geometric(0.5)（KS p > 0.05）；equivalence \|excess\| < δ_excess（**toy validation 在 simulation 上跑，不依赖 PR-7 实证 cohort**） | C6 |
| T4 | swap node 与 core node 高度重合 | (P4 prediction 的 toy 内部检验) |
| T5 | s_rate(t) 在 min–hour 尺度上慢漂时，event rate 表现 min–hour 漂移但 attractor identity 仍 i.i.d.（lag-1 still ≈ 0.5） | (与 Topic 2 + C6 同时一致 → 验证两过程解耦) |
| T6 | 关掉 core → fwd/rev 双稳态退化为单一对称态或不稳定 | (core 作用的 toy 内部 falsification) |
| T7 | 把 attractor selection 改为 deterministic alternating 或带强 history bias → lag-1 同标率显著偏离 0.5（同向或反向）| (random initial phase 必要性的 toy 内部 falsification) |

T5 与 (T6 + T7) 是 toy 阶段的内部 sanity：T5 证明两过程解耦合理（与 Topic 2 + PR-7 同时兼容），T6 + T7 证明 **少任何一个组件 toy 就不能再现已观察事实**——因此 toy 是 minimal 的，不是 over-parameterized 的。

---

## §6 Fitted model — BHPN-fit（spec）

### 6.1 输入 / 训练 / 测试切分

- 每个 subject 单独训练（无跨 subject 参数共享）
- **时间切分严格 50/50**：前半训练，后半 held-out 测试；再交换训练 / 测试方向，cohort 取一致方向
- 训练数据：每个 event 的 `lagPatRaw` + `bools` + cluster_labels + 多源 SOZ proxy（PR-T3-1 输出）
- 测试 observable：后半 events 的 (a) cluster label (fwd / rev), (b) event 内 propagation rank, (c) event-to-event reciprocity rate

### 6.2 自由参数（必须低）

| 参数 | 个数 | 说明 |
|---|---|---|
| coupling gain ρ | 1 | 单参数 |
| slow modulation timescale τ_s | 1 | 慢变 OU |
| refractory τ_ref | 1 | 短不应期 |
| core node identity | k ≤ 3 | binary mask, k 由 cohort 整体设定 |
| core ω 偏置 Δω 或 plasticity gain Δγ | 1 | toy 阶段二选一，fitted 沿用 |

**总自由参数 ≤ 6**。任何引入 N×N 自由 connection 的版本不属于 BHPN-fit，归 follow-up。

### 6.3 Predictions on held-out（v1.1.2：仅 aggregate-level + null-relative，禁止逐事件 label 预测）

**v1.0 → v1.1 关键修订**：v1.0 F1 承诺"逐事件预测 fwd/rev label accuracy > 50%" 在框架内部自相矛盾——§4.4 明确规定 attractor identity 由 random initial phase 独立选取（i.i.d. across retrievals），框架本身**禁止**模型预测单个 event 的 label 优于 chance。v1.1 把所有 held-out predictions 限定为 *aggregate-level* + *conditional* 量。

**v1.1 → v1.1.2 关键修订**：v1.1 F2 写 "Marginal P̂(forward) ∈ [0.4, 0.6]" 与 "geometric(0.5) KS"——与 §5.3 同样的 marginal-50/50 假设漏洞。PR-7 addendum 数据审计显示 cohort marginal P(fwd) 中位 0.498、范围 [0.20, 0.61]，548 极端 P(fwd)=0.20。50/50-anchor 与 geometric(0.5) 形式会对 marginal asymmetric subject 系统性 fail，与模型的 i.i.d. 实际语义无关。v1.1.2 把 F2 全部改为 marginal-aware null-relative 形式，与 §5.3 v1.1.2 PR-7 addendum 范围一致。

| # | Observable | Metric (v1.1.2) |
|---|---|---|
| **F1** | held-out 上 forward / reverse template **几何**（每 cluster 平均 rank pattern） | Cosine similarity (training template vs held-out template) > 0.8 + 显著优于 B1 random graph baseline (subject-level Wilcoxon) |
| **F2** (v1.1.2) | held-out 上 cluster label **null-relative 调用结构** | (a) **Marginal stability**：held-out P̂(forward) 与 training P̂(forward) 的 subject-level absolute diff 中位数 < 0.05，**不**预设 P̂ ≈ 0.5；(b) **null-relative same-label 偏向**：held-out lag1_same_excess（vs N2 marginal-preserving null）的 cohort bootstrap CI ⊆ [−δ_excess, +δ_excess]；(c) **null-relative run-length lift**：held-out run_length_lift（vs N2）的 cohort bootstrap CI ⊆ [1−δ_excess, 1+δ_excess]；与 §5.3 v1.1.2 P3 判据形式一致 |
| **F3** | held-out 上 **conditional rank pattern**（已知 cluster label, 预测 event 内 channel rank 顺序） | Per-event Spearman ρ(predicted_rank \| label, observed_rank)，cohort 中位数 > pre-registered 阈值 (TBD in PR-T4-2 plan) + 显著优于 B2 rate-only baseline |
| **F4** | core node identity（从 training 估出）的解剖位置 | 与 multi-source SOZ proxy overlap fraction (Jaccard / hit-rate)，subject-level Wilcoxon vs B3 random k-subset baseline，p < 0.05 |

**逻辑约束（lock at v1.1.2）**：
- F1 + F3 共同承诺"模型抓到了模板几何"（**conditional** prediction：给定 label，预测 within-event 顺序）
- F2 承诺"模型抓到了 i.i.d. 调用结构"（marginal-aware null-relative 形式；模型**不能** overpredict null-relative reciprocity，且 train/test marginal 应稳定）
- F4 承诺"模型 core 与 SOZ proxy 重合"（与 P4 一致）
- **禁止性条款**：(i) 任何 F1–F4 涉及"逐事件预测 fwd/rev label accuracy > 50%" 的扩展违反框架；(ii) 任何 F2 残留 marginal-50/50 / geometric(0.5) anchored 形式都属 v1.1 漏洞，PR-T4-2 plan 必须使用 v1.1.2 spec
- 因 P3 在 PR-7 addendum 后 verdict = INCONCLUSIVE（1800s + lag1_same_excess null-relative 子项 PASS, 短窗 + run_length_lift INCONCLUSIVE），F2 在 fitted model 上复测时各子项**继承**对应 P3 子项判定的可达性预期：F2 (b) lag1_same_excess 子项可期 PASS；F2 (c) run_length_lift 子项现实预期 INCONCLUSIVE（除非 fitted model 把 6 subject 做 large-N 仿真等价扩 cohort）；F2 (a) marginal stability 是新检验，不与 P3 重叠
