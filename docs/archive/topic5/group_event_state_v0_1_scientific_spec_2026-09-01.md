# Group-Event State v0.1 — 科学 spec（frozen 2026-09-01）

## 0. 核心问题

**能否从患者完整的间期群体事件序列里，学到一个可跨任务使用的病理网络状态？**

主序列 = 患者按真实时间排列的**完整间期群体事件**。连续背景 SEEG 只是辅助观测，
不是主时钟、不是替代序列、不再是主预测任务。每个时间步是一整个群体事件
（不是单触点 IED、不是 rank step、不是固定一分钟背景窗）。

数据语义见 `group_event_state_v0_1_data_contract_2026-08-31.md`（本文件不重复）。

## 1. 因果结构（由实现强制）

对第 `i` 个事件：

```
1  predict  Δt_i          from  z(t_{i-1}^+)                 # 演化之前
2  evolve   z(t_i^-) = b + (z(t_{i-1}^+) − b)·exp(−Δt_i/τ)   # 真实秒
3  correct  z(t_i^-) ← z(t_i^-) + g(background 观测于 t_i 之前)
4  predict  事件 i 的内容 from z(t_i^-)
5  update   z(t_i^+) = Update(z(t_i^-), Enc(event_i))        # 似然算完之后
```

两条已实现并写成回归测试的**泄漏禁令**（`tests/test_topic5_group_event_state_no_leakage.py`）：

- **timing 头必须读演化之前的状态。** 若先按 `Δt_i` 演化再预测，衰减量本身
  就是答案的函数——`Δt_i` 会通过 `exp(−Δt/τ)` 泄漏进它自己的预测。
- **baseline 的 recent-history 特征必须整体后移一位。** `t[i] − t[i−lag]`
  在 `lag=1` 时**恰好等于** timing 目标；已改为 `t[i−1] − t[i−1−lag]`。

内容端点条件在 `Δt_i` 上是**合法**的：marked point process 分解为
`p(t | past)·p(mark | t, past)`。delay 头对**所有**触点输出 `(μ, σ)`，
似然只在实际参与的触点上累加——头的参数不依赖 participation mask，因此
不构成泄漏。

## 2. 状态

两个时间尺度，都用 `τ = exp(clamp(log τ, log τ_min, log τ_max))`：

| | 维度 | τ 范围 | 初始化（log-uniform） |
|---|---|---|---|
| `z_fast` | 64 | 1 s – 1 h | 10 s – 600 s |
| `z_slow` | 32 | 60 s – 48 h | 30 min – 24 h |

`softplus(log τ)` 会把 τ 压在 ~20 s 以下，本仓库已为此付过一次代价
（Epi-PRSSM v0.1）；本线用 `exp(clamp(·))` 并加了"τ_slow 必须能到小时"的测试。

- fast 更新：`GRUCell([event_emb, z_slow], z_fast)`
- slow 更新：`z_slow + 0.05·σ(gate)⊙tanh(delta)`，**允许双向**，不预设所有 IED 同号推动
- session 边界：状态重置为可学习初值，**状态不跨未观测间隙传播**

## 3. 事件 encoder

输入是**整个患者触点宇宙**（不只参与触点）。四个分支叠加到每个触点的 token 上：

- **waveform**：共享 1-D CNN（stride 4/4/4），三个参考视图各带可学习 view embedding
- **time-frequency**：per-band 包络轨迹 CNN + band summary + cross-band lag；
  不支持的频带乘 0 **并附 mask flag**（missing，不是 0）
- **structural**：participation / exact delay（值 + 名次分数 + finite flag）/
  tied group（组序 + 组大小）/ legacy rank（仅 a2 臂）
- **geometry**：坐标（可得时）+ shaft/序号 + "坐标是否真实"标志位

触点 token 过 1 层 masked multi-head self-attention + FFN，再 masked mean/max 池化
得到 event embedding。

## 4. 端点（**分别报告，禁止合成单一分数**）

| 端点 | 形式 | 读法 |
|---|---|---|
| `timing` | log-normal NLL of Δt | 下一事件何时来 |
| `participation` | per-contact Bernoulli | 哪些触点参与 |
| `group_size` | \|E[size] − size\| | 事件范围 / STOP |
| `delay` | Gaussian NLL on participants | 触点间精确延迟（招募顺序） |
| `band_energy` | Gaussian NLL, per (contact, band) | 分频带能量传播 |
| `band_peak` | Gaussian NLL, per (contact, band) | 分频带峰时 |
| `cross_band_lag` | Gaussian NLL, per (contact, pair) | 跨频带 time-lag |

**判读纪律**：只有 `group_size` / `participation` 改善 → 只能叫 **extent state**；
`delay` / `cross_band_lag` / 招募顺序也改善 → 才接近 **repertoire state**。

## 5. 臂

核心五臂（§九）：

| arm | 输入 | 问题 |
|---|---|---|
| `a1_static_recent_history` | 最近 1/5/20 事件的固定摘要，**无潜状态** | 不用状态能做到多少 |
| `a2_rank_group_state` | participation + legacy 整数 rank | 只有 rank 的旧信息够不够 |
| `a3_delay_group_state` | + tied groups + 精确连续 delay | 精确延迟比 rank 多给了什么 |
| `a4_full_multimodal_state` | + 原生波形 + 多频带 | 波形/频带再多给什么 |
| `a5_full_plus_background` | + 背景 SEEG 修正 | 背景观测是否另有帮助 |

Ablation：`b1_no_real_dt`（事件计数时钟）、`b2_no_waveform`、`b3_no_multiband`、
`b4_memoryless`（每个事件重置状态）、`b5_no_geometry`、`b6_slow_only`（fast 压到 8 维）。

容量不强行对齐（不同臂的输入维度本就不同）；**报告参数量与训练预算**，
并用 `b4_memoryless`（同一 encoder、无状态）承担"是不是只是容量更大"的对照。

## 6. H1 — 是否学到跨事件持续的状态

白话：过去很多次完整群体事件，是否形成一个能持续影响未来事件的内部状态，
而不只是记住最近一次事件？

证据链（**四条都要报**）：

1. `a4` vs `b4_memoryless`：持续状态 vs 同 encoder 无状态
2. **correct-time vs matched wrong-time**：把 test 段的状态轨迹整体置换后重算内容似然。
   置换保持状态的边缘分布不变，只破坏"状态对得上时刻"这件事
3. **历史截断阶梯**：同一个训练好的模型，在评测时每 K 个事件重置一次状态
   （K = 1 / 5 / 20 / 100 / 全 session）。若长历史不优于 K=5，就没有"长状态"
4. `a4` vs `a1_static_recent_history`：状态 vs 已知近期历史的无状态基线

**禁止**把"网络比均值好"读成生理状态。

## 7. H2a — 状态是否改变下一次群体事件的表达

按 §4 七个端点**逐项**报告 patient-first 的臂间差。判读见 §4 末段。
另报两个派生量：招募顺序的 Spearman/Kendall（预测 delay 名次 vs 实测），
以及 tied-group pairwise 一致率。

## 8. H2b — 间期状态能否跨任务预测发作

1. 只用**间期事件**训练 state model
2. **冻结** encoder / state dynamics / state trajectory
3. 发作标签只进入后置 probe
4. patient-first、按时间顺序的 risk set
5. 对照：history only / current observation / memoryless event code / persistent state
6. **按 seizure pattern 分层报告**，不把不同发作型混成一个结论
7. development 阳性**不得**写成 cohort confirmation

H2b 是理想的跨任务验证，**不是 H1 的硬 gate**。
Yuquan 侧注意：seizure 来自 detection 而非临床标注，10/21 患者 0 检出——
只能读"未检出"，不可读"无发作"。

## 9. H3 — IED 是否逐渐塑造未来状态

尺度：100 / 1,000 / 5,000 / 10,000（或该患者覆盖允许的更长），
**按真实连续覆盖计算**，滑窗数不冒充独立样本数。

主问题：控制 pre-event state 之后，过去 IED 的"超出预期部分"是否仍能改善
未来状态或未来事件预测？

对照（针对本仓库 08-26 复审记录的"固定 event jump 饱和后变成免费截距"这一失效）：

- no event edge
- real event update
- **intercept-matched control**（同样多一个自由截距，但不带事件信息）
- non-overlapping delayed exposure
- state-matched placebo
- current-event-only jump

H3 阴性不阻止 H1/H2；短尺度阴性不阻止长尺度探索。

## 10. 硬停止（只有四类）

1. train / validation / test 或 seizure-label 泄漏
2. event、contact、reference 或时间轴错位
3. formal / sealed 分区被误开
4. 原始数据真的缺失且无法追溯

**以下都不是停止项**：单患者阴性、某端点阴性、raw 不赢、background 不赢、
H3 某尺度不赢、某个模型家族训练困难。遇到这些应降低对应结论、试更简单的替代模型、
继续其它假设。

## 11. 预注册的判读语言

- 只有 `group_size` / `participation` 改善 → "extent state"，**不得**写成 repertoire
- 单患者阳性 → 描述性句子，**不得**作为 cohort 主张
- H2b development 阳性 → "development-only"，**不得**写成 cohort confirmation
- CUDA OOM 导致的失败 → `resource_failed`，**永远不得**读成科学阴性
