# PR-T4-1 计划：BHPN-toy（Bidirectional Hebbian Phase Network — abstract toy model）

> ## ⚠️ SUPERSEDED — 2026-05-20
>
> **本计划已被 `docs/topic4_sef_itp_framework.md` (SEF-ITP framework v1) 取代，不再是 Topic 4 模型层的 plan-of-record。**
>
> **取代原因（朴素表述）**：本 plan 设计的 BHPN-toy 是**循环论证** —— 预先把"目标通道顺序"塞进对称 Hebbian 连接矩阵 `A_ij = ρ·cos(φ*_i − φ*_j)`，让 Kuramoto 演化必然落到 `φ*_F` 或 `π − φ*_F` 两个对偶 fixed point，然后宣称"模型复现了正反模板"。**现象已经被预先编码进矩阵**；模型只是把预编码的东西演化一次显示出来，**不解释**为什么是这套通道、为什么是这个顺序、为什么发生在 SOZ 区域。
>
> **新方向（SEF-ITP）**：把"空间"放回机制核心 —— 假设 SOZ 是一个空间组织化的病理易激区，间期事件是它被扩散物理反复采样后留下的痕迹。模型预测的是 θ(x) + 扩散 + 随机触发产生的几何指纹（H1–H6），不是 θ(x) 本身。
>
> **保留为历史归档的价值**：本 plan 的 v2 修订过程（rotating-frame rank、direction-agnostic mark dependence、TDD 拆分 unit vs integration）作为方法学训练范例保留；其中"k=2 数学几乎必然由对称 Hebbian 保证、不是 mechanism discovery" 这一关键认知是 SEF-ITP 转向的直接触发点。
>
> 下面 v2 plan 原文不动，仅作为历史记录。
>
> ---

> **状态**：plan-of-record v2（2026-05-01 同日修订，待执行）
> **上游 framework**：`docs/paper1_framework_sba.md` v1.1.2（含 audit-triggered correction，2026-05-01 lock）
> **创建日期**：2026-05-01（v1 同日由 review 触发 v2 重写）
> **范围**：Paper 1 §3 toy model **机制可视化为第一优先级 deliverable** + 数学 invariant 单元测试 + large-N stochastic sanity（机制可行性 demo），**不**拟合任何真实 subject 数据
> **下游消费者**：PR-T4-2（fitted BHPN-fit, 等 PR-T4-1 + PR-T3-1 完成后启动）、Paper 1 §3 + §4 figure source
> **预算**：3 周（figure storyboard 锁死后 → math primitives TDD → 6 张主图 → integration runner → doc）

**v1 → v2 修订（2026-05-01 review）**：

| # | v1 问题 | v2 修订 |
|---|---|---|
| 1 | 可视化在 Step 9（"补图"位置） | **图 storyboard 提到 §3 + Step 0/1**，作为 primary deliverable lock 在 plan 时；6 张核心图：(i) Model schematic / (ii) Phase landscape / (iii) Example retrieval trajectories / (iv) 1000-retrieval rank heatmap / (v) Ablation panel / (vi) Slow rate decoupling raster |
| 2 | TT10 "remove core → 双稳态崩掉" 与 Hopfield-Kuramoto 数学矛盾（A_ij = ρ cos(φ−φ) 已数学保证 fwd/rev 双 fixed point；core 影响的是 basin asymmetry / endpoint anatomy，不是 bistability 本身） | **删除 TT10 / 改写 T6**：remove core → P(forward) → 0.5 strict（basin symmetry 恢复），**双稳态仍存在**；endpoint anatomy 的 core 锚定是 framework H₀ illustrative assumption，不是 toy 自然涌现的发现 |
| 3 | T4 把 core 手放 endpoint 后又验证 core ⊆ endpoint，是 tautology | **T4 重 frame 为 illustrative consistency check**：toy 显式声明 core_indices 是 framework H₀ 的 illustrative 输入（v1 不引入 learning stage），T4 验证 toy 配置正确而非"模型发现 core ⊆ endpoint"。机制层面"核作为 endpoint 的合理性"由 P4 PR-8 v2 真实数据检验，不在 toy 范围 |
| 4 | rank 直接 argsort(θ mod 2π)，会被 ω_0 = 110 Hz common rotation + wrap-around 污染 | **rotating-frame relative phase**：ψ_i(t) = θ_i(t) − ω_0·t；rank 由 ψ_i 在 retrieval 后段（最后 20–50 ms 平均）的 circular order 派生；similarity 用 phasor cosine（e^{iψ}）而非 raw rank |
| 5 | TT14 锁定 sign（"naive coupled storage 上 excess > 0.3"）；但 θ=previous+ε 实际产生 same-template persistence（excess < 0），方向反 | **TT14 改 direction-agnostic**：检验 \|mark-dependence\| > δ_excess，多个 deviating mechanism 各产生不同符号偏差（alternating → excess > +δ；sticky / Markov(p_stay=0.8) → excess < −δ；都满足 \|mark\| > δ） |
| 6 | Step 0 创建 figures/README.md 占位（违反 AGENTS.md "图生成后再写 README"） | **Step 0 不创建 README 占位**；6 张图 storyboard 在 §3 plan 锁死，README.md 在 Step 7 figure 实际生成后写 |
| 7 | TDD 14 项混在一起，stochastic large-N 与数学 invariant 不分；stochastic test flaky | **TDD 拆分**：(a) **unit tests**（math invariants, fixed seed, low-noise / 解析对照, 快速 deterministic）放 `tests/test_bhpn_toy.py`；(b) **large-N stochastic sanity** 放 `scripts/run_bhpn_toy.py` integration runner，verdict 写入 `results/bhpn_toy/sanity_*.json` 而非 pytest |

---

## 1. Context — 为什么这是 framework 的下一个执行 PR

Paper 1 论证流是 **toy → predictions → fitted**（framework §11 lock）。toy 必须**先于** P4 / P5 / fitted 完成，因为：

1. Paper 1 §3 / §4 figure source 来自 toy（机制可视化）
2. P3 v1.1.2 等价 margin **δ_excess = 0.05** 的 scientific 论证依赖 toy 在 naive coupled storage 配置下 simulated |excess| ~ 0.3–0.5——这条数字 **目前是声称值，需要 toy 实证支持**
3. PR-T4-2 BHPN-fit 的 5 个 dumb baseline 都需要 toy 上先验证 baseline 各自抹掉哪个 mechanism component 在 simulation 上确实导致对应 prediction 失败
4. PR-7 addendum (2026-05-01) 已 lock P3 = INCONCLUSIVE-but-compatible（1800s + lag1_same_excess null-relative PASS）；toy 必须**独立**演示 i.i.d. 调用机制能产生这种 verdict pattern

**关键纪律（v1.1.2 lock）**：toy validation 走 **large-N simulation (N ≥ 1000 retrievals per subject; M ≥ 5 toy subjects)**，**不**以 PR-7 实证 cohort (n=6) 为 anchor。toy 只回答"机制是否能产生观察到的现象"，不回答"机制是否在真实 cohort 上 fits"——后者归 PR-T4-2 BHPN-fit。

---

## 2. Inheritance declaration（v1.1.2 lock）

| Framework 组件 | 本 PR 实现位置 |
|---|---|
| §3 SBA 单核心假设 | Toy model demo 该假设的可行性 |
| §4.1 节点动力学（phase oscillator at HFO band） | `src/bhpn_toy.py` `simulate_within_event(...)` |
| §4.2 connection matrix（symmetric Hebbian, ρ cos(φ_i\* − φ_j\*)） | `src/bhpn_toy.py` `build_hebbian_connection(...)` |
| §4.3 core node heterogeneity（k=2–3, ω 偏置 OR plasticity gain） | toy parameter `core_indices`, `core_omega_offset` |
| §4.4 Process A: s_rate(t) (OU, min–hour) → event rate modulation | `simulate_event_train(...)` 内部 OU + Poisson |
| §4.4 Process B: per-retrieval ε_id (random initial phase i.i.d.) | `simulate_retrieval(...)` 初始相位 uniform[0, 2π] |
| §4.4 refractory τ_ref (50–200 ms) | retrieval 后 phase 重置 + 抑制窗 |
| §4.5 T1–T7 sanity check 清单 | §4 本计划 + `tests/test_bhpn_toy.py` |
| §6.3 F2 (v1.1.2) marginal-aware null-relative | toy validation 在 BHPN-fit (PR-T4-2) 复用 metric 逻辑 |

**显式不属于 PR-T4-1**：

- 任何拟合真实 subject 数据（→ PR-T4-2）
- δ_excess 的重新设定（lock at v1.1.2）
- 对 framework v1.1.2 任何 prediction 判据的修改
- 任何 learning stage / Hebbian 自更新 / endpoint 自然涌现机制（v1 toy 把 storage + core 当 illustrative 输入；这些归 future PR）

---

## 3. Figure storyboard（v2 lock at plan time, 6 张核心图）

**v2 决定**：toy model 的价值不在"证明能写 Kuramoto"，而在用最少的图让 reviewer 一眼看懂 SBA mechanism。6 张图在 plan 时锁死布局与科学信息，作为 primary deliverable 写入 §6 Step 1（先于 large-N integration runner）。

每张图都对应一个 SBA framework 的概念断言；reviewer 不读代码、不读数值表也能从图判断 toy 是否可信。

### Fig 1 — Model schematic（concept figure）

**信息**：BHPN-toy 的三层结构分开画。

| 子面板 | 内容 |
|---|---|
| 1a | N=10 节点的 spatial layout，标注 k=2 core nodes（颜色高亮） |
| 1b | Hebbian storage matrix `A_ij = ρ cos(φ\*_F[i] − φ\*_F[j])` 的 heatmap（对称） |
| 1c | Directed readout `D_ij = sin(φ\*_F[j] − φ\*_F[i])` 的 heatmap（反对称） |
| 1d | 标注：A_ij is storage（symmetric，不预测方向）；D_ij is directional readout（vs. ictal coherence in P5）；两者从同一 φ\* 派生但用途不同 |

**关注点**：storage matrix ≠ directed predictor；P5 用 D_ij，不用 A_ij（避免 v1.0 cos-based directed graph 错误）。

### Fig 2 — Phase landscape / basin map（mechanism intuition）

**信息**：在 collective coordinates 上画两个 attractor basin。

- 高维 Kuramoto 不能硬画 phase plane；用**投影坐标** q_F = ⟨cos(θ_i − φ\*_F[i])⟩（与 forward template 的 alignment）vs q_R = ⟨cos(θ_i − φ\*_R[i])⟩（与 reverse template 的 alignment）
- 多次 random init 的 trajectory 投影到 (q_F, q_R) 平面，轨迹收敛到 (1, 0) 或 (0, 1) 两个 basin
- 背景 contour 是基于多次仿真的 attractor 频率 density（heatmap）

**关注点**：纯对称 Hebbian 数学保证两个对偶 basin 同时存在；core 影响 basin 偏置（asymmetric 仿真叠 default 仿真做对比 sub-panel）。

### Fig 3 — Example retrieval trajectories（dynamical detail）

**信息**：3–4 条随机初始相位 trajectory 在 100 ms 内的演化。

| 子面板 | 内容 |
|---|---|
| 3a | 旋转坐标 ψ_i(t) = θ_i(t) − ω_0·t，10 节点 line 图，2 example: 一条收敛 fwd, 一条收敛 rev |
| 3b | 同 trajectory 的 phase raster（hilbert 风格 wrapped phase 颜色编码）, 100 ms × N |
| 3c | rank vector 时间演化（rank within event over time），收敛到 forward / reverse template |

**关注点**：rotating-frame ψ_i 演化干净；100 ms 内 phase-locking + 稳定 rank 形成。

### Fig 4 — Rank heatmap of N_retrievals retrievals（老 fig4/5 风格升级）

**信息**：1000 retrievals 的 rank pattern matrix，按 cluster 排序。

| 子面板 | 内容 |
|---|---|
| 4a | rank matrix (channel × retrieval), 列按 cluster label (fwd/rev) 分组 |
| 4b | 旁边叠 cluster centroid template (fwd vs rev rank pattern) bar chart |
| 4c | KMeans(k=2) silhouette + adaptive_k stability（参 PR-2 stable_k 风格） |

**关注点**：k=2 主导 + cluster 内 rank 一致性高 + cluster 间 anti-correlation 显著（与 PR-2.5 fwd/rev 一致）。

### Fig 5 — Ablation panel（mechanism necessity）

**信息**：4 toy variants 的 lag-1 same-label rate + window excess 比较。

| 子面板 | 内容 |
|---|---|
| 5a | bar chart of lag-1 same-label rate: default i.i.d. / no-random-init (sticky) / Markov(p_stay=0.8) / alternating |
| 5b | window excess curve at Δt ∈ {10, 30, 60, 1800} for same 4 variants，叠 framework δ_excess=0.05 line |
| 5c | run_length distribution (geometric fit) for default vs sticky |

**关注点**：default i.i.d. 全部 metric 在 ±δ 内；其他三个 variant **\|mark\| > δ** in 不同方向（v2 修订：alternating excess > +δ；sticky / Markov excess < −δ）。验证 **random initial phase 是 mark-independence 的必要机制项**。这是 framework §7 B5 baseline 在 toy 上的具体演示。

### Fig 6 — Slow rate decoupling（dual-process架构）

**信息**：s_rate(t) 与 ε_id 的统计独立性。

| 子面板 | 内容 |
|---|---|
| 6a | OU 过程 s_rate(t) trace（24h, τ_rate=300s） |
| 6b | event density (5-min binned) 时间序列 + binned-rate autocorrelation curve（lag 5/30/60 min positive，与 Topic 2 一致） |
| 6c | event raster colored by label (fwd/rev) — 视觉上 rate 慢漂但 label 颜色 i.i.d. 撒落 |
| 6d | sliding-window P̂(forward) trace：尽管 rate 漂移 10×，P̂(forward) 维持稳定（**贴 long-time-average ± sampling noise**），与 rate trace 解耦——具体值由 default toy 的 basin asymmetry 决定，可能不等于 0.5；关键是它不随 rate 漂 |

**关注点**：rate 慢漂（min–hour, 与 Topic 2 一致）+ identity 不跟着漂（attractor selection 仍 i.i.d.）。这是 framework v1.1 修订（s_rate / ε_id 解耦）的核心可视化。

---

**Storyboard locked**：6 张图的子面板布局、轴标签、对比 variant 在 plan time 锁死；Step 1 实际生成时禁止重新设计。如发现 figure 设计与 toy mechanism 不匹配，归 v2.1 audit-triggered correction，提交 user 审阅，**不**直接改 storyboard。

---

## 4. Toy model 数学规范（v2 lock at PR-T4-1 plan time）

### 4.1 Within-event 动力学（v2: rotating-frame rank）

```
For each retrieval r:
    θ_i(0) ~ uniform[0, 2π]                                           # i.i.d. across r (Process B)
    For t in [0, T_event] with dt = 1 ms:
        dθ_i/dt = ω_i + Σ_j A_ij · sin(θ_j - θ_i) + ξ_i(t)
    end
    # v2: rotating-frame relative phase (avoid ω_0 common rotation + wrap-around contamination)
    ψ_i(t) = θ_i(t) - ω_0 · t                                         # rotating frame
    ψ_late_i = circular_mean_{t ∈ [T_event - 30ms, T_event]}(ψ_i(t))   # average late-window relative phase
    rank_r[i] = argsort_i(circular_unwrap(ψ_late_i - ψ_late_ref))      # ψ_late_ref = circular mean of ψ_late
    label_r = argmax_k cos_similarity(e^{i·ψ_late}, e^{i·φ*_k})        # phasor cosine, not raw rank
end
```

**rotating-frame 修订理由（v2）**：v1 直接 `argsort(θ mod 2π)` 会被 ω_0 = 110 Hz × 100 ms = 11 full rotations 的 common phase advance + 2π wrap-around 污染。rotating frame ψ_i = θ_i − ω_0·t 把 common rotation 减掉，使 rank 对应的是节点相对于 mean field 的 phase lag/lead。circular average 取 retrieval 后段（最后 30 ms）减小过渡期噪声。similarity 用 phasor cosine（e^{iψ}）而非 raw integer rank，避免 ties / 量化误差。

- N nodes ∈ {8, 10, 12}（可配，default 10 to match SEEG 局部）
- ω_i ~ N(ω_0, σ_ω)，ω_0 = 110 Hz（HFO 频段中段）, σ_ω = 5 Hz
- ξ_i(t) ~ N(0, σ_noise²)，σ_noise = 0.05·ω_0
- T_event = 100 ms (within HFO event 量级)
- dt = 1 ms
- 解析 sanity check（T_event → ∞, σ_noise → 0, ω_i 同质）：ψ_i 应收敛到 φ*_F[i] 或 φ*_R[i]，cosine similarity 到对应 template ≥ 0.99

### 4.2 Connection matrix（symmetric Hebbian, v2 unchanged）

```
target pattern: φ*_F[i] = (rank_F[i] / N) · π                         # forward template
                φ*_R[i] = π − φ*_F[i]                                  # reverse (180° opposite)
A_ij = ρ · cos(φ*_F[i] − φ*_F[j])                                     # Hopfield-Kuramoto symmetric storage
```

- ρ ∈ [0.5, 5.0]（coupling gain, lock at default 2.0 for sanity；扫描范围在 §4.6 sensitivity）
- A_ii = 0（no self-coupling）
- 符号约定：rank_F = [0, 1, ..., N−1]（forward 模板）； rank_R = [N−1, ..., 0]（reverse）
- **数学事实**：A_ij 数学上保证 φ*_F 与 φ*_R 都是 fixed point（Hopfield-Kuramoto 对称存储已知性质）。**bistability 由对称 Hebbian 直接产生，与 core 节点异质性无关**——v1 计划 T6 误将 "remove core → bistability breaks" 写为 sanity，已在 v2 修正（参 §5）。

### 4.3 Core node heterogeneity（v2: illustrative assumption, no learning stage）

**v2 重要 caveat**：toy v1 不引入 learning / endpoint emergence，core 位置是 framework H₀ 的 **illustrative input**。"core ⊆ endpoint" 在 toy 内是配置正确性 (consistency check)，**不是模型自然涌现的发现**。机制层面"为什么核作为 endpoint"的检验由 P4 / PR-8 v2 真实数据承担，不在 toy 范围。任何关于 "toy 演示 core 自然落到 endpoint" 的写法都属于 v1 tautology，v2 已禁止。

实现两种参数化（toy default 选 mode A，§4.6 sensitivity 测试 mode B）：

- **Mode A (default)**：core_omega_offset = 0.05·ω_0，apply 到 k_core ∈ {2, 3} core nodes
- **Mode B (alt)**：plasticity gain γ_core = 1.5·γ_other（仅在长期 storage 学习模拟时启用，v1 不实现）

`core_indices` 在 toy 内是 explicit illustrative 输入参数（v2 default 放在 rank_F 两端：rank 0 + rank N−1，与 PR-6 endpoint anchoring 关注的 source ∪ sink 几何角色对齐）。**由于这是先验放置，T4 是 consistency 而非 discovery**（参 §5）。

### 4.4 Inter-event slow rate modulation (Process A)

```
ds_rate/dt = -s_rate / τ_rate + η(t)             (OU, η white noise)
λ(t) = λ_0 · exp(s_rate(t))                       (Poisson rate at time t)
retrieval times: Poisson(λ(t))
```

- τ_rate ∈ [10, 1000] s（default 300 s，允许 sweep）
- λ_0 = 0.05 Hz（cohort median rate ≈ 3 events/min, matches PR-7 cohort 量级）

**关键设计 (v1.1.2 lock)**：s_rate 调 rate，**不**调 ε_id（attractor identity）。

### 4.5 Refractory τ_ref

每次 retrieval 后 50 ms 内禁止再触发 + phase 重置为 random uniform[0, 2π]。τ_ref ≪ 任何 P3 检验窗（10s），不贡献 reciprocity 信号。

---

## 5. T1–T7 内部 sanity check operationalization（v2 修订）

每条 T 是 toy 的内部 falsification gate：default toy 必须在 **large-N integration runner (N_retrievals ≥ 1000, M_subjects ≥ 5)** 上同时满足 T1 / T2 / T3 / T4 / T5；T6 + T7 是 ablation/falsification 配置，用关掉特定机制项的 toy 变体验证 default 配置中每个机制项的角色。

| ID | Sanity 描述 | Toy 配置 | 通过判据 |
|---|---|---|---|
| **T1** | 系统在两个对称稳定相位排序之间收敛（fwd / rev） | default toy（symmetric Hebbian + core） | ≥ 95% retrievals 的 ψ_late 与 forward-template 或 reverse-template 的 phasor cosine similarity ≥ 0.9（rotating-frame 版） |
| **T2** | 多次 retrieval 的 ψ pattern 聚为 k=2 cluster 主导 | default toy, M=5 subjects, N_retrievals=1000 | KMeans(k=2) 在 phasor 空间 silhouette > 0.5；adaptive cluster (1..6) 的 stable_k = 2 在 ≥ 4/5 subject |
| **T3** | 在 fixed s_rate 下，连续 retrievals attractor identity i.i.d.（**marginal-aware**：default toy with core 可能 P(fwd) ≠ 0.5） | default toy, fixed s_rate=0, N_retrievals=2000 per subject, M=5 | (a) **null-relative** lag-1 same excess: \|lag1_same_empirical − (P̂_fwd² + P̂_rev²)\| < δ_excess（i.i.d. 下 lag1_same 应等于 marginal 平方和；与 framework v1.1.2 §5.3 marginal-aware 一致）；(b) run-length 分布与 geometric(P̂_other) KS p > 0.05（per-label 几何分布，参数由经验 marginal 决定，**不**预设 0.5）；(c) N2-style window shuffle excess(10s, 30s, 60s, 1800s) 全部 \|excess\| < δ_excess (=0.05) |
| **T4** | core 与 endpoint 配置一致（**illustrative consistency check, NOT discovery**） | default toy, M=5, k_core=2，core_indices 显式放在 rank_F 两端 | core_indices ∈ {top-3 ranks ∪ bottom-3 ranks} = 100% by construction（v1 toy 不引入 learning，core 位置是 framework H₀ illustrative 输入；T4 只验 toy 的 storage 几何与 core placement 在 retrieval 后仍一致：core 节点的 ψ_late rank 留在 endpoint 位置而不漂走，而非"模型自然把 core 推到 endpoint"） |
| **T5** | s_rate(t) 慢漂时 event rate 漂移但 attractor identity 仍 i.i.d. | toy with τ_rate=300s, simulation duration ≥ 24h | event rate 5-min binned autocorrelation positive at lag 5/30/60 min（与 Topic 2 一致）；**marginal-aware decoupling**：sliding-window P̂(fwd) 跨 24h 各 1h 窗口的标准差 / 与 24h 平均的 max absolute deviation 都 < δ_excess（即 marginal 稳定，不跟 rate 漂；具体均值由 toy basin asymmetry 决定，**不**预设 0.5）；同时 null-relative lag-1 same excess 与 T3 (a) 同 spec |
| **T6 (v2 重写)** | 关掉 core 异质性（ω_i 同质）→ basin asymmetry 消失但 **bistability 持续** | toy variant: A_ij = ρ cos(...) 但 ω_i ≡ ω_0（无 core ω 偏置） | (a) **bistability persists**（T1 + T2 仍满足，cluster k=2 + silhouette > 0.5）；(b) **basin symmetry 严格化**：P̂(forward) bootstrap CI ⊆ [0.48, 0.52]（与 default toy 可能存在的微弱 asymmetry 对比）；(c) endpoint 的 ψ rank 仍由 stored template φ\*_F 决定，与 core 是否存在解耦。**v1 错误声称 "remove core → bistability breaks"，v2 改正：bistability 是 symmetric Hebbian 的数学后果，core 只调 basin asymmetry / endpoint anatomy** |
| **T7 (v2 重写, direction-agnostic)** | 把 attractor selection 改为 deterministic / history-biased → \|null-relative mark-dependence\| > δ_excess | toy variants: (a) alternating fwd/rev; (b) Markov chain p_stay=0.8; (c) θ_i(0) = previous_template_phase + ε (sticky) | 对每个 variant：\|lag1_same_excess vs N2-style marginal-preserving null\| > δ_excess（=0.05）；window-30s \|excess\| > δ_excess；**符号方向不锁定**——alternating 应给 excess > +δ（opposite-template lift），sticky / Markov 应给 excess < −δ（same-template lift）；toy 实现正确仅要求 \|mark\| > δ |

**Locked (v2)**：
- T1 / T2 / T3 / T5 是 default toy 必须通过的 cohort sanity（M=5 subjects × N_retrievals=1000–2000）
- **T4 是配置 sanity，不是 mechanism discovery**（toy v1 不演示 endpoint emergence；那是 future PR with learning stage）
- T6 验证 default toy 的 core 角色限定在 basin asymmetry / anatomy；**bistability 与 core 解耦**
- T7 验证 default toy 的 random initial phase 是 mark-independence 的必要项；direction-agnostic：only require \|mark\| > δ
- 任何"toy 自然把 core 推到 endpoint"或"remove core 必然破坏双稳态"的写法都是 v1 错误，v2 已禁止

---

## 6. TDD 测试列表（v2 重构：unit + integration 拆分）

**v2 关键变化**：v1 把 14 项 TDD 全混在 `tests/test_bhpn_toy.py`，包含 large-N stochastic sanity（容易 flaky）。v2 严格拆分：

- **Unit tests (tests/test_bhpn_toy.py)**：math invariants + 解析对照 + 固定 seed + low-noise / 单 retrieval 量级，**deterministic, fast, 不应 flaky**
- **Integration runner (scripts/run_bhpn_toy.py)**：large-N stochastic sanity (M=5 × N=1000–2000 retrievals)，verdict 写入 `results/bhpn_toy/sanity_*.json`，**不**走 pytest（runner 自带 PASS/FAIL 自检并退出码）

### 6.1 Unit tests（`tests/test_bhpn_toy.py`，11 项，全部 deterministic）

| ID | 测试函数 | 关注 invariant | 实现说明 |
|---|---|---|---|
| TT1 | `test_build_hebbian_connection_symmetric` | A_ij == A_ji (symmetric); A_ii == 0 | 解析检验，无 random |
| TT2 | `test_build_hebbian_connection_stores_both_attractors_analytically` | 给定 A = ρ cos(φ\*_F − φ\*_F^T)，**解析**验证 φ\*_F 与 φ\*_R = π−φ\*_F 都是 dynamics 的 fixed point（zero-noise, ω_i ≡ ω_0 limit：dθ/dt 在两 template 上同时为 0）| 数值验证 max\|f(φ\*_F)\| < 1e-8 与 max\|f(φ\*_R)\| < 1e-8 |
| TT3 | `test_rotating_frame_rank_invariant_under_common_rotation` | 对 θ_i 全部加同一常数 c：rank 输出不变（rotating frame 把 common rotation 吸收） | 固定 seed 单 retrieval，对比 c=0 / c=π/3 / c=2π·5 输出 rank 一致 |
| TT4 | `test_phasor_cosine_similarity_correct_for_known_pattern` | 给定 ψ = φ\*_F + 小扰动 ε，phasor cosine ≈ 1；给定 ψ = φ\*_R + ε，phasor cosine ≈ 1 (vs reverse)，−1 (vs forward) | 解析单元 |
| TT5 | `test_simulate_within_event_converges_to_template_low_noise` | σ_noise=0, 给定 initial ψ 接近 φ\*_F，T_event=100 ms 后 phasor cosine ≥ 0.99；给定接近 φ\*_R，≥ 0.99 vs reverse | low-noise deterministic, fixed seed |
| TT6 | `test_random_init_picks_basin_in_low_noise_regime` | low-noise default toy, 200 random init, fixed seed → ≥ 95% retrievals 落在某一 basin (cosine sim ≥ 0.9 vs fwd 或 rev) | seeded, deterministic |
| TT7 | `test_remove_core_preserves_bistability` (v2 新, 替代 v1 TT10) | toy variant ω_i ≡ ω_0 (无 core), 200 random init, fixed seed → ≥ 95% 仍落在 fwd 或 rev basin（**bistability 保持**）；同时 P(fwd) 与 0.5 的 absolute deviation < 0.05（**basin symmetry 恢复**） | 直接验证 v1 TT10 错误 |
| TT8 | `test_alternating_init_produces_negative_excess` | 50 retrievals with deterministic alternating init (fwd/rev/fwd/rev...), fixed seed → lag-1 same-label rate < 0.1 (即 \|lag1_excess\| > 0.4)，方向 = opposite-template lift | small N OK, deterministic seed, direction = opposite |
| TT9 | `test_sticky_init_produces_positive_same_label` | 50 retrievals with sticky (θ(0) = prev_template + 0.1·ε), fixed seed → lag-1 same-label rate > 0.7 (即 \|lag1_excess\| > 0.2)，方向 = same-template lift | direction = same |
| TT10 | `test_markov_pstay_produces_history_bias` | 50 retrievals with Markov p_stay=0.8 forced label sequence, fixed seed → lag-1 same-label rate ∈ [0.7, 0.9] | direction = same |
| TT11 | `test_naive_coupled_storage_produces_mark_dependence` (TT14 v2) | toy variant **without random init**（v1 TT14：θ(0) = previous_final_state + ε，**未限定方向**），50 retrievals, fixed seed → \|window-30s null-relative excess\| > 0.3 **OR** \|lag1_excess\| > 0.2，**方向不锁** (direction-agnostic per §5 T7 修订)；输出 single JSON 报告实际方向 | direction-agnostic |

**Unit test 总数 11**（v1 14 - 删 v1 TT5/TT7（迁去 integration）- 删 v1 TT10（错误声称）+ 加 v2 TT3/TT4/TT7 stronger）。所有 unit test 必须 deterministic + low-noise + fixed seed + < 30s 内全部通过。

TT11 (v2 TT14 重写) 关键：把 framework v1.1.2 §5.3 δ_excess scientific 推理（"naive coupled storage 上 |mark-dependence| ~ 0.3–0.5"）从声称值升级为 toy 实证；**direction-agnostic**——alternating / sticky / no-random-init 各产生不同符号偏差，验证只要 random initial phase 缺失就 |mark| > δ。

### 6.2 Integration runner sanity（`scripts/run_bhpn_toy.py`, large-N stochastic）

不走 pytest，由 runner 自带 PASS/FAIL 判定 + 退出码 + 写入 `results/bhpn_toy/sanity_*.json`。

| Sanity ID | 来源 | 配置 | PASS 判据 |
|---|---|---|---|
| S1 (= T1 large-N) | §5 T1 | M=5 × N=1000 retrievals, default toy | ≥ 95% retrievals 落在 fwd 或 rev basin（per subject）；cohort min ≥ 90% |
| S2 (= T2) | §5 T2 | 同上 | KMeans(k=2) silhouette > 0.5 in ≥ 4/5 subjects；adaptive stable_k=2 in ≥ 4/5 |
| S3 (= T3 large-N i.i.d., marginal-aware) | §5 T3 | M=5 × N=2000 retrievals, fixed s_rate=0 | (a) **null-relative** lag-1 same excess: cohort \|lag1_same − (P̂_fwd² + P̂_rev²)\| < δ_excess；(b) run-length geometric(P̂_other) per-label KS p > 0.05 在 ≥ 4/5；(c) window {10/30/60/1800} \|excess\| < δ_excess in ≥ 4/5；**禁止**预设 marginal=0.5 |
| S4 (= T4 illustrative) | §5 T4 | M=5, k_core=2, core 显式放 rank 两端 | core_indices 在 retrieval 后 ψ_late rank 中仍位于 endpoint (top-3 / bottom-3) ≥ 95%（**配置一致性, not discovery**） |
| S5 (= T5 slow rate decoupling, marginal-aware) | §5 T5 | M=5, τ_rate=300s, 24h | event rate 5-min binned autocorr lag 30 min > 0.1；**marginal-aware decoupling**：sliding-window 1h P̂(fwd) 与 24h 平均的 max\|Δ\| < δ_excess (per subject)；null-relative lag-1 same excess (\|lag1_same − P̂_fwd² − P̂_rev²\|) cohort < δ_excess；**禁止**预设 marginal=0.5 |
| S6 (= T6 ablation) | §5 T6 v2 | toy variant ω_i 同质 (no core), M=5, N=1000 | (a) bistability persists：basin coverage ≥ 95%, KMeans silhouette > 0.5；(b) basin symmetry 严格：P̂(fwd) cohort CI ⊆ [0.48, 0.52]；(c) endpoint anatomy 由 stored template 决定（与 default toy 对比 ψ_late rank 排序一致） |
| S7 (= T7 falsification, 3 variants) | §5 T7 v2 | toy variants: alternating / Markov(p=0.8) / sticky; M=5, N=500 each | 每个 variant：\|null-relative lag1_excess\| > δ_excess **OR** \|window-30s excess\| > δ_excess；direction-agnostic；alternating 给 opposite-lift, Markov / sticky 给 same-lift |

Runner 输出：`results/bhpn_toy/sanity_S1_S5.json`（default toy）+ `results/bhpn_toy/sanity_S6_S7.json`（ablation / falsification）。

**关键纪律**：
- runner 失败不会被 pytest 抓到（unit tests 已经全过）；runner 必须自带 PASS/FAIL summary 印 stdout + 写 verdict 字段
- runner 用 fixed RNG seed = 0 + 显式扫描 5 subject seed (1..5)，**不**用 unseeded random
- stochastic CI underpowered 时（per-subject N 不够）：runner 自检并 raise，提示提高 N，**不**降阈值

---

## 7. Step breakdown（v2: figure storyboard 优先于 large-N runner）

**v2 设计原则**：figure storyboard (§3) 是 primary deliverable；先把 6 张图做出来再跑 large-N integration runner。Step 0–4 实现 unit-test-level 的核心 mechanism，Step 5 立即跑出图（用 small-N seeded simulation 演示 mechanism, M=2-3 toy subjects），Step 6 才开 large-N runner。

### Step 0 — 模块脚手架（不创建 figures README 占位）

- [ ] **0.1** 新建 `src/bhpn_toy.py` 模块 + `tests/test_bhpn_toy.py` test stub + `scripts/run_bhpn_toy.py` runner stub + `scripts/plot_bhpn_toy.py` plot stub
- [ ] **0.2** 创建 `results/bhpn_toy/` 目录与 `figures/` 子目录（**不**创建 figures/README.md 占位——按 AGENTS.md 规范图生成后再写；Step 5 完成后才写 README）
- [ ] **0.3** Commit：`feat(pr-t4-1): Step 0 — BHPN-toy scaffolding`

### Step 1 — 数学 primitives + unit tests TT1–TT4

- [ ] **1.1** 实现 `build_hebbian_connection(template_phase, rho)`、`compute_rotating_frame_phase(theta_t, t, omega_0)`、`compute_phasor_cosine_similarity(psi, template_phase)`
- [ ] **1.2** TT1 + TT2（解析 storage） + TT3 (rotating-frame invariance) + TT4 (phasor cosine) GREEN
- [ ] **1.3** Commit：`feat(pr-t4-1): Step 1 — math primitives + unit tests (TT1–TT4)`

### Step 2 — Within-event integrator + retrieval + unit tests TT5–TT7

- [ ] **2.1** 实现 `simulate_within_event(theta_init, omega, A, n_steps, dt, sigma_noise, rng) -> theta_trajectory`（return full 而非仅 final, 供 figure 用 trajectory）
- [ ] **2.2** 实现 `simulate_retrieval(template_phase, omega, A, T_event, dt, sigma_noise, rng) -> dict{psi_late, rank, label, similarity}` （uniform random initial phase, rotating-frame rank）
- [ ] **2.3** TT5 (low-noise convergence) + TT6 (random init basin) + TT7 (remove core preserves bistability) GREEN
- [ ] **2.4** Commit：`feat(pr-t4-1): Step 2 — integrator + retrieval (TT5–TT7)`

### Step 3 — Falsification variants + unit tests TT8–TT11

- [ ] **3.1** 实现 4 个 retrieval-sequence 函数：`simulate_retrieval_sequence_iid(...)`（default uniform random init）、`_alternating(...)`、`_markov(p_stay)`、`_sticky(epsilon)`（v2 TT11 关键，对应 framework naive coupled storage）
- [ ] **3.2** TT8 (alternating → opposite lift) + TT9 (sticky → same lift) + TT10 (Markov → same lift) + TT11 (no-random-init → \|mark\| > δ direction-agnostic) GREEN
- [ ] **3.3** Commit：`feat(pr-t4-1): Step 3 — variants + falsification unit tests (TT8–TT11)`

### Step 4 — Core 异质性 + slow rate

- [ ] **4.1** 实现 `add_core_heterogeneity(omega, core_indices, omega_offset)` + `simulate_event_train(...)`（OU + Poisson + retrievals）
- [ ] **4.2** Unit test：固定 seed，core_indices=[0, N-1] 时 ψ_late rank 在 retrieval 后**保持**两端位置（illustrative consistency, T4 unit-level 版本）
- [ ] **4.3** Unit test：OU + Poisson 在已知 τ_rate=300s 下 5-min binned autocorr lag 30 min 显著 > 0
- [ ] **4.4** Commit：`feat(pr-t4-1): Step 4 — core + slow rate primitives`

### Step 5 — 6 张核心图 (figure storyboard 实现)

**v2 关键 step**：把 §3 storyboard 落地。每张图独立 commit。每图先用 small-N (M=2-3, N=200-500) seeded simulation 生成；如果某图需要更大 N，由该图自带 runner 走 N=1000-2000。

- [ ] **5.1** Fig 1 (model schematic)：`plot_bhpn_toy.py --fig 1`，4 子面板，**non-stochastic**（纯 schematic + matrix heatmap）。Commit。
- [ ] **5.2** Fig 2 (phase landscape / basin map)：M=1 toy, N=500 random init, 投影到 (q_F, q_R) 平面 + density contour. Commit.
- [ ] **5.3** Fig 3 (example retrieval trajectories)：4 trajectories (2 fwd, 2 rev) in rotating frame ψ_i(t)；fixed seed. Commit.
- [ ] **5.4** Fig 4 (rank heatmap, 1000 retrievals)：M=1 toy, N=1000, KMeans(k=2) 重排 + cluster centroid 旁注. Commit.
- [ ] **5.5** Fig 5 (ablation panel, 4 variants)：default i.i.d. / alternating / Markov / sticky 各 N=500, 比较 lag-1 + window excess + run-length distribution；叠 framework δ=0.05 line. Commit.
- [ ] **5.6** Fig 6 (slow rate decoupling)：M=1 toy, 24h simulation, 4 子面板 (OU trace / event rate / label raster / sliding P̂(fwd)). Commit.
- [ ] **5.7** **写** `results/bhpn_toy/figures/README.md`（图全部生成后，按 AGENTS.md 中文 + "关注点"行）。Commit：`feat(pr-t4-1): Step 5 — 6 storyboard figures + Chinese README`

### Step 6 — Large-N integration runner sanity (S1–S7)

- [ ] **6.1** 实现 `scripts/run_bhpn_toy.py --mode {default-sanity | ablation | falsification}` CLI
- [ ] **6.2** Mode `default-sanity`：M=5 × N=2000 retrievals，跑 S1–S5 全部判据，verdict 写入 `results/bhpn_toy/sanity_default.json`，runner 自检并退出码 0/1
- [ ] **6.3** Mode `ablation`：toy without core，M=5 × N=1000，跑 S6 判据 → `sanity_ablation.json`
- [ ] **6.4** Mode `falsification`：alternating / Markov / sticky 三 variants 各 M=5 × N=500，跑 S7 判据 → `sanity_falsification.json`
- [ ] **6.5** Mode `delta-excess-validation`：no-random-init variant 在 toy 上量化 \|mark-dependence\| 大小，verdict 写入 `delta_excess_validation.json`（结果可回写 framework v1.1.2 §15 changelog 作为 toy-validated δ）
- [ ] **6.6** Commit：`feat(pr-t4-1): Step 6 — large-N integration runner (S1–S7 verdicts)`

### Step 7 — 文档收口

- [ ] **7.1** 写 `docs/archive/topic1/pr_t4_1_bhpn_toy_results_<date>.md`（§1 figure storyboard inventory + §2 unit test verdict + §3 integration runner verdict S1–S7 + §4 TT11 / delta_excess validation + §5 framework v1.1.2 一致性 + §6 与 v1 计划差异）
- [ ] **7.2** 在 `docs/topic1_within_event_dynamics.md` §2 + §10 加 PR-T4-1 verdict 一行 + archive 链接
- [ ] **7.3** 如 TT11 / S7 / delta_excess_validation 验证 framework δ_excess 推理，**提案** v1.1.3 audit-triggered correction（写入新 changelog，提交 user 审阅，**不**直接编辑 framework 主体）
- [ ] **7.4** Commit：`docs(pr-t4-1): Step 7 — close out PR-T4-1 v2`

---

## 8. 文件 / 模块 map

新增：

- `src/bhpn_toy.py` — toy module（math primitives + integrator + retrieval + variants + slow rate）
- `tests/test_bhpn_toy.py` — 11 项 unit tests (TT1–TT11)，全部 deterministic + fixed-seed + low-noise + < 30s 通过
- `scripts/run_bhpn_toy.py` — large-N integration runner (4 modes: default-sanity / ablation / falsification / delta-excess-validation)；自带 PASS/FAIL 退出码
- `scripts/plot_bhpn_toy.py` — 6 张图生成（Fig 1–6 per §3 storyboard）
- `docs/archive/topic1/pr_t4_1_bhpn_toy_results_<date>.md` — 完成时写

新增结果目录（按 AGENTS.md 规范）：

```
results/bhpn_toy/
├── sanity_default.json               # S1–S5 default toy large-N sanity
├── sanity_ablation.json              # S6 remove-core variant (bistability persists)
├── sanity_falsification.json         # S7 alternating / Markov / sticky variants
├── delta_excess_validation.json      # TT11 / no-random-init |mark-dependence| 数据
├── figures/
│   ├── README.md                     # 中文 (Step 5.7 写, 不在 Step 0 占位)
│   ├── fig1_model_schematic.png
│   ├── fig2_phase_landscape.png
│   ├── fig3_retrieval_trajectories.png
│   ├── fig4_rank_heatmap_1000_retrievals.png
│   ├── fig5_ablation_panel.png
│   └── fig6_slow_rate_decoupling.png
```

**不新增 / 不动**：framework / topic doc / Topic 2 / 任何 PR-2/3/4/5/6/7 已有 archive 文档（除 §7 Step 7 显式列出的同步更新与 v1.1.3 audit-triggered correction 提案）。

---

## 9. 失败模式与修订路径（v2）

| 失败 | 解读 | 处理 |
|---|---|---|
| TT1–TT10 任一 fail | toy unit-level 实现错误 | 修正 toy 实现，不修 framework |
| TT7 fail（remove core 后 bistability 没保持） | toy implementation 错误（数学上 symmetric Hebbian 必须保持双稳态） | 必修 toy；如果**调试后**仍 fail，意味 v2 §4.2 数学声称错误，归 v1.1.3 framework audit-triggered correction |
| TT11 fail（naive coupled 上 \|mark-dependence\| < δ in 所有 variant 与方向） | framework v1.1.2 §5.3 δ_excess 的"naive predicts \|mark\| ~ 0.3–0.5" 声称值不实 | **不**直接调 δ；先尝试更强 naive 配置（更高 ρ / 更窄 ω 分布 / 更长 sticky persistence）；若仍 fail，提案 v1.1.3 framework audit-triggered correction |
| S1–S5 任一 fail in default-sanity runner | toy mechanism 不够 minimal、large-N 参数空间错位、或单元测试 invariant 与 large-N 行为脱节 | 文档化 fail 配置 + 修正 toy 默认参数 / runner N；不动 framework |
| S6 fail (remove core 后 bistability 没保持 OR P̂(fwd) 不收紧到 0.5) | TT7 unit-level 通过但 large-N 不一致：integrator 数值精度 OR ablation 实现错误 | 必修 |
| S7 fail（deviating mechanism 在 large-N 仍未触发 \|mark\| > δ） | toy variant 实现错误或 N 不够大 | 调整 N + 必要时调试 variant |

**Locked rule (v2)**：本 PR 不允许修改 framework v1.1.2 任何 prediction 判据；任何 toy 揭示的 framework 内部矛盾归 v1.1.3 audit-triggered correction 提案，**新建** changelog entry 并提交 user 审阅，**不**直接编辑 framework 主体。

---

## 10. 自检清单（v2 plan 写完必须勾掉）

- [x] 顶部声明继承 framework v1.1.2（含 audit-triggered correction note）
- [x] toy validation 走 large-N integration runner，**不**以 PR-7 实证 cohort 为 anchor
- [x] §3 figure storyboard 6 张图在 plan time lock，作为 primary deliverable（先于 large-N runner）
- [x] T4 frame 为 illustrative consistency check, **不**声明"toy 自然把 core 推到 endpoint"（v1 tautology 已修复）
- [x] T6 v2 重写：remove core → bistability **保持**，basin symmetry 收紧；**不**声称 "remove core breaks bistability"（v1 错误已修复）
- [x] T7 / TT11 direction-agnostic：\|mark-dependence\| > δ，符号不锁；alternating / sticky / Markov 各产生不同符号偏差
- [x] rank 定义用 rotating frame ψ_i = θ_i − ω_0·t（v1 raw argsort 漏洞已修复）
- [x] TDD 拆分：unit tests (deterministic, fixed seed, low-noise, math invariants) vs integration runner (large-N stochastic sanity)；不混
- [x] Step 0 **不**创建 figures/README.md 占位；README 在 Step 5 figure 实际生成后写
- [x] 11 项 unit tests 每项 invariant 清晰，无 placeholder
- [x] 7 项 integration sanity (S1–S7) lock at plan time
- [x] toy 不拟合任何真实 subject 数据（fitted → PR-T4-2）
- [x] 显式 out of scope：不动 framework v1.1.2、不修 P3/P4/P5 判据、不引入新 prediction、不引入 learning stage
- [x] AGENTS.md 规范：figures/README.md 中文 + "关注点" 行（Step 5.7 写）；results 目录命名按 topic 分类
- [x] 失败模式区分"toy 错误" vs "framework 错误"，后者归 v1.1.3 提案不直改 framework

---

## 11. 与下游 PR 的边界

| 下游 PR | 依赖本 PR | 不涉及 |
|---|---|---|
| **PR-T4-2 BHPN-fit** | 复用 `src/bhpn_toy.py` 的 connection / integrator / retrieval 函数；继承 toy 的 minimal mechanism set | PR-T4-2 拟合真实数据，本 PR 不做 |
| **PR-9 (P5 directionality)** | 不直接依赖；可独立并行 | 本 PR 不实现 sin-based / rank-based directional predictor |
| **PR-T3-1 (data-driven SOZ audit)** | 不直接依赖；可独立并行 | 本 PR 不消费 SOZ 数据 |
| **PR-8 v2 (P4 anchoring)** | 等 PR-T3-1 完成后启动；不直接依赖本 PR | 本 PR 不消费 endpoint 真实数据 |

---

## 12. 一句话承诺（v2）

PR-T4-1 v2 把 BHPN-toy 当作**机制可视化工具**：先锁 6 张核心图作为 primary deliverable（reviewer 不看代码也能从 §3 storyboard 看懂 SBA mechanism），再用 11 项 deterministic unit tests 锁住数学 invariants，再用 integration runner (S1–S7) 检验 large-N stochastic 一致性。toy 不拟合任何真实数据；不引入 learning stage；不预设"core 自然涌现到 endpoint"（那是 P4 / PR-8 v2 真实数据的事）；不修改 framework v1.1.2 任何 prediction 判据；任何 framework 漏洞归 v1.1.3 提案，不直改。
