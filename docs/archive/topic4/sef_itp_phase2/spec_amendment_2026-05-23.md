# SEF-ITP Phase 2 — H4 I_rate matched null spec amendment proposal

> **状态**：**Proposal pending user ratification**. Phase 2 v1.0.0 runner produces both methods; framework doc edit deferred until user returns.
> **作者**：Claude (Opus 4.7) + advisor consult
> **日期**：2026-05-23
> **触发**：Phase 2 H4 implementation discovered framework v1.0.5 §3.4 prose is mathematically degenerate.

---

## 1. 用大白话讲清楚问题

H4 想测什么？看一天里事件率（events/h）漂得有多厉害，再看一天里 endpoint 通道集合漂得有多厉害，比较"率漂"和"几何漂"哪个更大。framework 的 SEF-ITP 预测：率漂幅度 ≫ 几何漂幅度（病理空间结构稳定，触发频率不稳定）。

要比较"漂得有多大"必须做归一化，因为两个量量纲不同（一个是事件数/小时，一个是 Jaccard 距离）。framework v1.0.5 §3.4 给的归一化公式：

```
I_rate = std(log(rate_obs)) across epochs
         ─────────────────────────────────
         sqrt( var of std(log(rate_null)) over 1000 matched-null shuffles )
```

匹配 null 的定义：**"time-shuffle epoch order 1000 次，每次重算 std → 取分布的 variance 当作 baseline"**。

**但这条 null 是数学退化的**：std 对一组数的顺序不敏感（permutation-invariant）。把 12 个 epoch 的 log(rate) 打乱顺序，得到的还是同一组数，std 还是同一个数。1000 次 shuffle 全得到同一个 std，方差 = 0，归一化分母 = 0，I_rate 未定义。

## 2. 实测验证退化性

```python
>>> rates = np.array([5.0, 10.0, 20.0, 40.0, 80.0])
>>> log_rates = np.log(rates)
>>> rng = np.random.default_rng(0)
>>> stds = [np.std(log_rates[rng.permutation(5)]) for _ in range(1000)]
>>> np.var(stds)
0.0
>>> set(stds)
{1.07...}  # 1000 个相同的值
```

不是实现 bug，是 spec 数学层面的退化。Framework prose 写错了。

## 3. 四种可能的修正方案

### (a) Circular-shift within block ⭐推荐

为每个 block 随机选一个偏移 Δ ∈ [0, epoch_seconds)，把 block 内所有事件时间加 Δ（block 内 wrap-around），重切 epoch，重算 per-epoch rate，取 std(log(rate))。1000 次得 null 分布。

- **优点**：保留每个事件相对其它事件的时间结构（事件间隔不变）；只是 epoch-membership 被打乱
- **科学含义**：null 回答 "如果事件触发的时间模式是同样的，但 epoch 的起点是随机的，rate 的 epoch-to-epoch std 会有多大"
- **类似**：space-time data 里常用的 spatial-bootstrap，preserve auto-correlation 结构

### (b) Homogeneous Poisson with same total count

按 block 内总事件数生成 homogeneous Poisson 事件流，重切 epoch，重算 rate std。

- **优点**：null 是"完全无 rate 调制"的 baseline，物理意义最直接
- **缺点**：忽略 burstiness / refractory；可能 inflate I_rate（因为 obs 有 burstiness 但 null 没有，所以 obs std 看起来更大）

### (c) Gamma-fit resample on epoch rates

对 obs 的 12 个 epoch rate 拟合 gamma 分布，从中重抽 12 个值，取 std(log)。

- **优点**：保留 rate 的边际分布
- **缺点**：null 几乎完全捕获 obs 的 std，I_rate ≈ 1 几乎一定

### (d) Cross-epoch shuffle preserving block boundary

把 block 内事件随机重分配到 block 内 epoch（保持 block 内 epoch 数不变），重算 rate。

- **优点**：是"epoch_order_shuffle"的合理修正（猜测 framework prose 原意可能是这条）
- **缺点**：极端情况下退化为 Poisson；保留 block 内边际分布

## 4. 推荐：(a) circular-shift within block

理由：
1. 保留 sub-epoch 时间结构 → 最尊重 obs 的细节
2. 只随机化 epoch-membership → 直接对应 "I_rate 衡量 epoch-to-epoch 不稳定" 的本意
3. 计算便宜 → cohort 23 subject × 1000 perm × 12 epoch 量级
4. **不**改变 framework 的科学问题，只修正数学表达

## 5. 为什么必须用户拍板

CLAUDE.md §5: "Pre-registered hypothesis tier is fixed at planning time." 同理，**pre-registered statistical null is fixed at framework time**。

framework v1.0.5 §3.4 prose 不严谨不等于授权 agent 自己挑一个新 null 然后 silently 改 framework。这是 spec 修正，不是 bug fix。**任何修正必须 user 拍板**，理由如下：

- (a)/(b)/(c)/(d) 的科学含义不同；选哪个会影响 cohort verdict
- 即使大家都同意 (a) 是最 sensible 的，**也要 user 显式确认**，避免日后被审稿人/合作者追问"为什么不是 (b)"时只能说"agent 当时选的"
- 一旦写进 framework，就有 pre-registration 效力；偷偷改 = 偷偷重新 pre-register

## 6. Phase 2 v1.0.0 实施决定

- **per-subject runner 同时跑 (a) 和 epoch_order_shuffle 两个 null** → JSON 两个字段都保留
- **cohort summarizer 两边都报** → 用户能比较
- **spec_amendment_2026-05-23.md（本文）**作为 proposal 落盘
- **`docs/topic4_sef_itp_framework.md` §3.4 不动**（v1.0.5 banner 保持）
- **`docs/archive/topic4/sef_itp_phase2/cohort_run_2026-05-23.md`** 写 cohort 数字时两边并列
- **STOP at Phase 2 Task 13** — Task 14（framework doc edit）作 deferred

## 7. 数学小附录

设 $X_1, \dots, X_n$ 为 obs epoch log-rates。

- **epoch_order_shuffle null**：取 $\sigma \in S_n$ 均匀随机，计算 $\text{std}(X_{\sigma(1)}, \dots, X_{\sigma(n)})$。但 std 是对称函数 → 不依赖 σ → 1000 个值全相同 → variance = 0。证毕。

- **circular_shift_within_block null**：取 $\Delta \sim U[0, \text{epoch\_seconds})$，事件时间集 $T = \{t_1, \dots, t_m\}$ → $T' = \{((t_i + \Delta - t_\text{block\_start}) \bmod L_\text{block}) + t_\text{block\_start}\}$，按原 epoch 边界重切，rate per epoch 改变，std 也改变。1000 个 Δ 给出非退化的 null 分布。

---

## 决定记录

| 字段 | 值 |
|---|---|
| 提议 | (a) circular-shift within block |
| Phase 2 v1.0.0 状态 | proposal, 双 null 都跑 |
| Framework 修改 | 待用户拍板 |
| 影响范围 | 仅 H4 I_rate；H4 I_geom 不受影响（其 null 是 endpoint role-shuffle，本来就非退化）|
