# PR-7 计划：Template Antagonistic Temporal Pairing

> 状态：plan-of-record，2026-04-28（v1）
> 范围：检验 forward/reverse template 在**短时间尺度**上是否构成"拮抗性配对"，即 T_a 之后短时窗内 T_b 出现的频率显著高于慢漂移基线。这是"Ping-Pong"机制叙事中**功能耦合层**的可证伪检验，**不**承担机制层（兴奋-抑制）论断。
> 上游：`docs/topic1_within_event_dynamics.md` §3 / §7（PR-2.5 forward/reverse + PR-6 endpoint geometry）；`docs/archive/topic1/ping_pong_hypothesis_review_2026-04-28.md` §3.3
> 下游：若 PASS，PR-8 candidate = signed rank displacement / continuous pair-geometry；PR-9 candidate = subject typology × PR-5 split

---

## 1. Context — 为什么这是下一个 PR

PR-6 Step 4 / 4b 已经把 cluster-rank 反向翻译成节点级 source/sink swap 几何（forward/reverse subset n=6, swap−same sign-test p=0.031）。这建立了**现象学层 (A)**：两套反向 rank template 共存且节点级 endpoint 角色互换。

下一步必须问的问题不是"swap 是不是真的"（已确认），而是：

- **T_a 与 T_b 是不是在时间上配对出现，还是各自独立？**

如果它们独立（即 T_b 出现频率不依赖最近一次 T_a 是否发生），那"两类模板"只是同一群体事件流被压缩到两个 cluster centroid 上的统计产物，**Ping-Pong 在因果层不成立**。如果它们配对（短时窗 lift > 1，长时窗 lift ≈ 1），则功能耦合层得到支持，可向 inhibitory-restraint / rebound 文献借用机制语言（**仅作文献一致性，不作 HFO 机制结论**——HFO 80–250 Hz 不区分 E/I）。

**这是 Ping-Pong 假说能否成立的最直接可证伪检验**。失败（lift ≈ 1 across all Δt）即关闭机制叙事，论文 framing 退回到 "interictal stereotypy reflects bidirectional traveling waves with no temporal coupling between directions"。

---

## 2. 编号与归档决定

- **新 PR**：PR-7 = **Template Antagonistic Temporal Pairing**
- **不打包**：节点级 signed displacement（PR-8 candidate）/ subject typology × PR-5 split（PR-9 candidate）/ 降维可视化（appendix）**不**进 PR-7
- **范围声明（写死）**：本 PR 只验证**功能耦合层**，**不**做机制归因。机制层（"Ping = excitation, Pang = inhibitory rebound"）仅作文献语言对接，不作数据结论
- **主文档回写**：完成后在 `docs/topic1_within_event_dynamics.md` §7 加一句话指向本计划的结果归档

---

## 3. 假设与统计合同（核心）

### 3.1 数据对象

每个 stable_k=2 forward/reverse-reproduced subject 的输入：
- `event_abs_times: np.ndarray[float, N]` — per-event absolute timestamp (sec since epoch)，由 `load_subject_propagation_events(subject_dir)` 提供
- `cluster_labels: np.ndarray[int, N]` — `adaptive_cluster.labels`，∈ {0, 1}（stable_k=2）
- `block_time_ranges: list[(float, float)]` — recording block 边界，**关键**用于 local-aware null
- `valid_event_indices` — 与 PR-2 一致的过滤（`min_n_participating ≥ 5` 已在 cluster 阶段应用）

**Label normalization**（PR-2 cluster id 任意性问题）：cluster id 0/1 由 KMeans 任意分配。本 PR 不预设方向，把两个 cluster 称作 `T_a` / `T_b`，并报告**对称化**指标（见 §3.3）。

### 3.2 H1 primary — Short-window opposite-template excess

**Per-subject 主统计量（subject-level）**：

对多尺度 Δt ∈ {1s, 5s, **10s**, 30s, 60s, 5min, 30min, 1h}（**完整 grid 见 §3.5**；其中 **10s = primary**，**30s = required sensitivity**，1s / 5s = packing-proximity diagnostic 不进 PASS 判据）：

```
For each event i with cluster_label l_i = c (c ∈ {a, b}):
    window_i = (t_i, t_i + Δt]
    n_opposite_i(Δt) = #{ j : t_j ∈ window_i, l_j ≠ c }
    n_same_i(Δt)     = #{ j : t_j ∈ window_i, l_j = c }
```

跨所有事件聚合：
```
P_opposite(Δt) = sum_i n_opposite_i(Δt) / N_events
P_same(Δt)     = sum_i n_same_i(Δt)     / N_events
```

**Lift 与 excess**：
```
opposite_lift(Δt) = P_opposite(Δt) / E_null[P_opposite(Δt)]
same_lift(Δt)     = P_same(Δt)     / E_null[P_same(Δt)]
excess(Δt)        = opposite_lift(Δt) − same_lift(Δt)
```

`E_null[·]` 由 §4 surrogate hierarchy 给出（**主 null = N2 local-window 30 min**）。报告 `lift` 而非 raw probability，便于跨 subject / 跨 Δt 比较。

**Cohort 主检验**（forward/reverse-reproduced subject，PR-6 §15 H2 cohort 对齐，audit-derived ≤ 8）：

主检验对 **Δt = 10s** 跑 Wilcoxon signed-rank test on `excess(10s)` against 0：
- 单边 alternative = "greater"（短窗预期 opposite > same）
- α = 0.05；二级 sign test
- 配 **Δt = 30s 同方向 sensitivity**：30s `excess` 中位数必须 > 0（不强求显著，但**不能反向**）

**PASS 判据**（**三条必须全部满足**）：
1. `excess(10s)` Wilcoxon p < 0.05（primary）
2. `excess(10s)` sign test p < 0.05（primary）
3. `excess(30s)` 中位数 > 0（required sensitivity，不必显著）

> 增加 30s sensitivity 的理由：10s 离 5s packing window 太近，单点显著不够干净。若 10s 显著但 30s 反向，提示信号被 packing-proximity stickiness 主导（参见 §9.4 confound profile B），不能 claim short-range coupling。

**NULL 接受**：
- `excess(10s)` Wilcoxon p > 0.10 → Ping-Pong 在功能耦合层不成立。论文 framing 转回纯几何描述
- 或 10s 满足但 30s 反向 → 信号是 packing 制品，机制叙事不成立

**1s / 5s 角色**：仅作 packing-proximity diagnostic 报告。如果 1s / 5s 显著且 10s / 30s 都 ≈ 0 → packing window stickiness 制品（confound profile B 触发）。**不**进 H1 PASS / NULL 判据。

### 3.3 H1b — Direction symmetry

Forward/reverse 假说不预设方向（T_a 跟 T_b 与 T_b 跟 T_a 应对称）。报告：
```
asym(Δt) = P(T_b in window | T_a at t) − P(T_a in window | T_b at t)
```
预期 |asym| 小（< 0.5 × |excess|）。若 |asym| 与 |excess| 同量级，提示两个 cluster 在 functional role 上不对等，需进一步 case-series。

仅作 secondary，不进 H1 α 池。

### 3.4 H2 — Negative control on non-forward/reverse subset

对 audit-derived non-forward/reverse h1_eligible subject（PR-6 §15 cohort，n=16）跑同一 H1 流程。**预期 excess(10s) ≈ 0**，因为这些 subject 的两个 cluster centroid 高度共享 endpoint（`J(source_same)`/`J(sink_same)` 在 PR-6 Step 4 测得低，`Spearman(rank_T0, rank_T1) ≈ −0.12`），不构成"两个对立模板"。

**PASS 判据 of H2**：non-fwdrev cohort `excess(10s)` Wilcoxon p > 0.10（即 negative control 守住）。
**FAIL of H2**：non-fwdrev 同样显著 → H1 阳性的因果归因被削弱（可能是事件 burst 普适特征，不是 fwd/rev 特异）。

### 3.5 多尺度衰减 profile（H1 解释力的关键）

**完整 grid**：`Δt ∈ {1s, 5s, 10s, 30s, 60s, 5min, 30min, 1h}`（共 8 个尺度）。

**尺度角色**：

| Δt | 角色 | 进入判据？ |
|---|---|---|
| 1s | packing-proximity diagnostic（远小于 5s packing window） | ❌ 仅诊断 |
| 5s | packing-edge diagnostic（恰在 packing window） | ❌ 仅诊断 |
| **10s** | **H1 primary** | ✅ |
| **30s** | **required sensitivity（必须同方向）** | ✅（方向） |
| 60s | short-range profile | profile only |
| 5min | mid-range profile | profile only |
| 30min | long-range expected ≈ 0 | profile only |
| 1h | very-long-range expected ≈ 0；slow-drive confound 检测 | profile only |

**预期 profile**（Ping-Pong consistent）：
- 短尺度（10s / 30s / 60s）：excess > 0
- 长尺度（30min / 1h）：excess ≈ 0
- 1s / 5s 预期不参与判读，但**应**与 10s 同方向；若 1s/5s 显著而 10s/30s 反向 → confound profile B 触发

**Confound profile A**（slow shared drive）：所有 Δt 都 lift > 1（包括 1h）→ 共享慢调制驱动，不是短程功能耦合。**论文不能 claim Ping-Pong**。应对见 §9.3。

**Confound profile B**（packing artifact）：仅 1s / 5s 上 lift > 1，且 10s / 30s 都 ≈ 0 → group-event 定义内置的 within-window stickiness。**应对见 §9.4**：packing-window sensitivity sweep。

报告 cohort-median `excess(Δt) curve` + 每 subject 单独曲线 + 10s / 30s 的 paired Wilcoxon。

### 3.6 Secondary descriptive — Next-event transition odds

**仅作描述，不进主 α 池**。把"窗口内密度上升"与"下一次事件就倾向反模板"区分开。

**定义**：对每个事件 i（label = c），找到下一个事件 j（按时间最近）：
```
If l_j ≠ c:  this is "opposite next"
Else:        this is "same next"
```
**Per-subject metric**：
```
P_next_opposite = #{i : l_{next(i)} ≠ l_i} / N_events
transition_odds = P_next_opposite / (1 − P_next_opposite)
baseline_odds = (n_opposite / n_total) / (n_same / n_total)  # null assuming i.i.d.
```

**额外**：报告 `time_to_next_opposite` vs `time_to_next_same` 的中位数（per subject + cohort 分布）。

**解释规则**：
- `transition_odds > baseline_odds` 显著 → 下一次事件**直接**倾向反模板（强 Ping-Pong 信号）
- `transition_odds ≈ baseline_odds` 但 `excess(10s) > 0` → 短窗内密度升高来自多事件 burst，不是单跳 alternation
- 两者一致 → 信号既在 next-event 也在窗口尺度，最干净的 Ping-Pong 证据

不进 H1 PASS / NULL 判据。仅写在结果章节作为机制解释的附加证据。

---

## 4. Surrogate 设计（**核心 — 用户 2026-04-28 push back 加强**）

单纯 "fix timestamps, shuffle labels globally" 太弱：总事件率本身有 burst 和慢漂移，全局 label shuffle 会把"高 rate 时段事件天然更密"误判为 T_a → T_b 配对。本 PR 必须并列报告**多个**渐进收紧的 null。

### 4.1 Surrogate hierarchy（per subject 各跑 1000 perm）

| ID | Null 名称 | 实现 | 控制了什么 | 没控制 | 角色 |
|---|---|---|---|---|---|
| **N0** | Global label shuffle | 把所有事件的 cluster_labels 全局随机洗牌 | label 比例 | 局部 rate burst | **sanity ceiling** |
| **N1** | Block-aware label shuffle | 在每个 `block_time_ranges[k]` 内部 shuffle labels | block 内 label 比例 | block 内的局部 rate burst | **sanity / mid-strength** |
| **N2** | Local-window label shuffle | 滑动窗口（默认 30 min）内 shuffle labels；窗口重叠 50% | 局部 30 min rate burst | <30 min 尺度的精细 burst | **主 null（H1 判读用此）** |
| **N3** | Circular shift label sequence | 把整个 label 序列做 circular shift（shift 量 = 随机整数 ∈ [N/10, 9N/10]） | label 序列的 autocorrelation 结构 + 全局 rate | 短程依赖（这正是要测的） | **robustness（必跑）** |
| **N4** | Rate-matched ISI permutation per cluster | 把每个 cluster 的事件时间替换为相同 rate trajectory 但 ISI 重抽（gamma fit per local window） | 每个 cluster 的 marginal rate trajectory | cluster 间 short-range 配对 | **conditional follow-up（不必做）** |

**层级判定**：

| 层 | 内容 | 何时跑 |
|---|---|---|
| 主 null | **N2** | 必跑（H1 PASS / NULL 判据基于此） |
| Robustness | **N3** | 必跑（与 N2 一致 → robust；不一致 → 降级） |
| Sanity | **N0 / N1** | 必跑（ceiling 与 mid-strength 对照，不进判读） |
| Conditional | **N4** | **仅在 N2 阳性但 N3 不一致时**作 follow-up；不作为 PR-7 主交付 |

> **N4 降级理由**：rate-matched ISI per cluster 参数化太重（gamma fit per local window × per cluster），小 cohort（n=6）下容易出现"模型假设决定结果"。把 N4 留作后续 follow-up，避免把 PR-7 主结论绑在一个 fragile parametric null 上。

**报告**：每个 subject 给出 N0 / N1 / N2 / N3 下的 `lift` 与 `excess` 分布（4 个 null + empirical = 5 版本）。**主 H1 判读用 N2**；N2 与 N3 一致 → robust；N2 显著但 N3 不一致 → 触发 N4 follow-up，主结论降级为 exploratory。

### 4.2 Null 实现规范

- **N1 / N2** 必须保持 timestamp 不动，只 permute label。**不**重抽事件时间
- **N2 窗口尺寸**：默认 30 min；`--null-window-min` flag 暴露 sweep（10 / 30 / 60 min）
- **N2 窗口边界**：滑动 step = window/2；事件落入第一个覆盖它的窗口 shuffle 池
- **N3 shift 范围**：[N/10, 9N/10] 避免 trivial shifts；每次 perm 独立抽 shift
- **N4 ISI 重抽**：在 5 min 局部窗口内 fit gamma(α, β) per cluster，按 fit 重抽 ISI（保持局部 cluster-specific rate trajectory）；这是最严格的 null，但参数化最多，仅作 robustness

### 4.3 Permutation budget

- Default `n_perm = 1000` per subject per null
- 多 subject × 4 null × 8 Δt 总成本：~6 subject × 4 null × 8 Δt × 1000 perm ≈ 1.9e5 evaluations。每 perm 是一个 O(N log N) 滑窗扫描；预算 ~10 min/subject GPU 不需要
- N4 仅在条件触发时跑，不计入主预算

---

## 5. Cohort 定义（audit-derived，复用 PR-6 audit 结果）

**关键设计**：直接复用 PR-6 §15 已经跑出来的 `cohort_audit.csv`，不再做独立 audit。

**入选条件（pre-registered，写死，不放宽）**：

1. `endpoint_defined == True`（即 `n_ch ≥ 6`，`stable_k == 2`，centroid 有 polarity）
2. `forward_reverse_reproduced == True`（OR 规则：split-half 或 odd-even 任一复现，PR-2.5 accepted rule）
3. `n_events_total >= 300`（lift 估计的 SE 可控）
4. `min(n_T_a, n_T_b) >= 75`（cluster 平衡，避免少数 cluster 的 anchor 太薄）
5. `n_blocks >= 3` **OR** `total_observed_coverage_hours >= 6`（N2 30 min 窗口有足够窗口数；coverage 用 `block_time_ranges` 求和得）

**关键原则（user 2026-04-28 push back）**：**门槛 pre-registered 写死，不放宽**。如果 fwd/rev cohort 因此从 6 掉到 < 4，按 §9.6 走 case-series，**不**通过放松门槛保 cohort size。

**预期 cohort 大小**（待 Step 1 audit 确认）：
- **H1 primary cohort**：fwd/rev reproduced ∩ 上述 5 条 → 预期 4–6（PR-6 §15 H2 cohort 6 是上界）
- **H2 negative control cohort**：non-fwdrev h1_eligible ∩ 上述 5 条 → 预期 ~13（PR-6 §15 16 是上界）
- **Sensitivity cohort**：所有 endpoint_defined ∩ 上述 5 条 → 预期 ~17（PR-6 §15 21 是上界）

**Step 1 第一动作**：跑 audit 增量，输出 `pr7_cohort_audit.csv`，每行一个候选 subject，列出：
- `n_events_total`、`n_T_a`、`n_T_b`、`min_cluster_n`
- `n_blocks`、`total_coverage_hours`、`max_block_hours`
- `forward_reverse_reproduced`（split-half / odd-even / both）
- 五条入选条件 pass / fail + `exit_reason`（首次失败决定）

**绝不预写 cohort size**。审稿人可在 audit 表里独立验证 inclusion logic。

---

## 6. 代码改动（最小化 + 大量复用）

### 6.1 新增：`src/template_temporal_pairing.py`

独立模块，纯统计层：

```python
def compute_pairing_lift(
    event_abs_times: np.ndarray,
    cluster_labels: np.ndarray,
    delta_t_seconds: float,
    block_time_ranges: List[Tuple[float, float]],
) -> Dict[str, float]:
    """Block-aware lift estimator. Returns symmetric + directional fields:
       {'p_opposite', 'p_same',
        'p_a_to_b', 'p_b_to_a', 'p_a_to_a', 'p_b_to_b',
        'n_a_anchors', 'n_b_anchors', 'n_used'}.
       Cluster labels MUST be in {0, 1} (a=0, b=1) per §3.1 normalization.
       Cross-block events filtered (no spurious cross-block counting)."""

def shuffle_labels_global(labels: np.ndarray, rng) -> np.ndarray: ...

def shuffle_labels_block_aware(
    labels: np.ndarray, event_abs_times: np.ndarray,
    block_time_ranges: List[Tuple[float, float]], rng,
) -> np.ndarray: ...

def shuffle_labels_local_window(
    labels: np.ndarray, event_abs_times: np.ndarray,
    window_seconds: float,
    block_time_ranges: List[Tuple[float, float]],   # MUST be passed; never crosses blocks
    rng,
) -> np.ndarray:
    """N2 main null: 50% overlap (step = window/2), first-covering rule,
       per-block independent. See §4.2 for the precise partition."""

def shuffle_labels_circular(labels: np.ndarray, rng) -> np.ndarray: ...

def resample_isi_per_cluster(*args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """N4 conditional follow-up — RAISES NotImplementedError.
       The gamma-fit-per-window construction in §4.1 has not been built
       because N4 is conditional. Implement before any follow-up trigger."""

def compute_pairing_with_nulls(
    event_abs_times: np.ndarray,
    cluster_labels: np.ndarray,
    block_time_ranges: List[Tuple[float, float]],
    delta_t_grid: List[float] = (1, 5, 10, 30, 60, 300, 1800, 3600),
    n_perm: int = 1000,
    nulls: Tuple[str, ...] = ("N0", "N1", "N2", "N3"),
    n2_window_seconds: float = 1800.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """Per-subject driver. Returns nested dict:
       {
         'empirical': {Δt: pairing_lift_dict (full directional fields)},
         'null': {N0/N1/N2/N3: {Δt: {p_opposite_dist, p_same_dist,
                                     p_a_to_b_dist, p_b_to_a_dist}}},
         'lift': {N0/N1/N2/N3: {Δt: {opposite_lift, same_lift, excess,
                                     a_to_b_lift, b_to_a_lift, asym}}},
       }
       N4 raises if listed in `nulls`; conditional follow-up only."""

def compute_transition_odds(
    event_abs_times: np.ndarray,
    cluster_labels: np.ndarray,
    block_time_ranges: List[Tuple[float, float]],   # block-aware mandatory
) -> Dict[str, float]:
    """Secondary descriptive (§3.6). Cross-block consecutive pairs are NOT
       counted as transitions (a recording-gap of hours is not a neural
       transition). Returns
       {'p_next_opposite', 'transition_odds', 'baseline_odds',
        'time_to_next_opposite_median', 'time_to_next_same_median',
        'n_pairs'}."""

def cohort_paired_test(
    excess_per_subject: Mapping[Hashable, float], alternative: str = "greater",
) -> Dict[str, float]:
    """Wilcoxon + sign test, key-aligned. Returns {'wilcoxon_p',
       'sign_test_p', 'median', 'n'}."""

def evaluate_pass_criteria(
    cohort_excess_10s: Mapping[Hashable, float],
    cohort_excess_30s: Mapping[Hashable, float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Triple gate: 10s Wilcoxon p<α AND sign p<α AND median(30s)>0.
       RAISES ValueError on subject-key mismatch between 10s and 30s dicts —
       paired design requires the same cohort on both sides.
       Returns {'pass': bool, 'wilcoxon_10s', 'sign_10s', 'median_30s',
                'median_30s_positive', 'n_subjects', 'subjects', ...}."""
```

### 6.2 新增：`scripts/run_pr7_template_pairing.py`

- `--audit`：跑 cohort audit，输出 `pr7_cohort_audit.csv`
- `--per-subject`：对 audit-pass subject 跑 `compute_pairing_with_nulls`，输出 `results/interictal_propagation/template_pairing/per_subject/<subject>.json`
- `--cohort`：H1 / H1b / H2 / 多尺度 cohort 统计，输出 `cohort_summary.json`
- `--null-window-min`：N2 窗口尺寸 sweep（默认 30）
- `--n-perm`：默认 1000
- `--seed`：默认 0

### 6.3 新增：`scripts/plot_pr7_template_pairing.py`

详细可视化方案见 **§6.5 Visualization Spec**。脚本只负责实现 §6.5 中各张图的绘制，输出到 `results/interictal_propagation/template_pairing/figures/`。同时生成 `figures/README.md`（中文，AGENTS.md 规范，每图 2–4 句 + "**关注点**:"）。

### 6.4 不动

- `compute_adaptive_cluster_stereotypy` / `build_cluster_templates` / `assign_events_to_templates` 完全不动
- `template_anatomical_anchoring` / PR-6 Step 1–5 输出完全不重跑
- 不引入任何新的 cluster 算法 / template 重定义
- 不动 PR-2 / PR-2.5 / PR-3 / PR-4A/B / PR-5

---

### 6.5 Visualization Spec — 回主文档前的最后一步

**核心要求**（user 2026-04-28）：图必须**直接展示核心假设的结果**，看图就能读出 Ping-Pong PASS / NULL / confound 是哪一种。

**输出目录**：`results/interictal_propagation/template_pairing/figures/`
- 主图 4 张 + Appendix 3 张
- 风格：`src/plot_style.py` Morandi 调色板（与 Topic 2 PPT 5 图集一致）
- 命名：`fig{n}_<short_name>.png`，全部 300 dpi PNG + matching SVG
- README.md：中文，每图 2–4 句 + "**关注点**:" 行（AGENTS.md 规范）

#### 主图 1 — Antagonistic Pairing Cohort Curve（**最核心一张**）

> **直接对应 H1 PASS / NULL 判读**。看一眼这张图就能知道 Ping-Pong 假说成立否。

- **布局**：单图，宽 8 in × 高 5 in
- **X 轴**：Δt，对数尺度，刻度标 `1s, 5s, 10s, 30s, 60s, 5min, 30min, 1h`
- **Y 轴**：`excess(Δt) = opposite_lift − same_lift`（per cohort，中位数）
- **曲线**：
  - **红实线** = fwd/rev cohort 中位数（带阴影 = 25/75 percentile envelope）
  - **灰虚线** = non-fwdrev cohort 中位数（H2 negative control）
  - **黑点线** = excess = 0 baseline
- **垂直线 / 阴影**：
  - 1s / 5s 区域用浅黄阴影 + label "packing-proximity diagnostic"
  - 10s 处加红色虚线 + 注释 "**H1 primary**"
  - 30s 处加红色虚线 + 注释 "**required sensitivity**"
  - 30min / 1h 区域用浅蓝阴影 + label "expected ≈ 0 (slow-drive control)"
- **文本框**（右上角）：
  - `H1 primary (10s): Wilcoxon p = 0.0XX, sign p = 0.0XX, median = 0.XXX`
  - `Required sensitivity (30s): median = 0.XXX (positive ✓ / negative ✗)`
  - `H1 verdict: PASS / NULL / packing confound`
- **可读性**：cohort=fwd/rev 红线**应明显**在短窗 > 0、长窗 ≈ 0；non-fwdrev 灰线在所有 Δt 上 ≈ 0
- **失败模式可视化**：
  - profile A confound：红线在 30min/1h 也 > 0
  - profile B confound：红线只在 1s/5s 黄区 > 0，10s/30s 落到 0
  - H1 NULL：红线全段 ≈ 0
  - H2 FAIL：灰线也短窗 > 0

#### 主图 2 — Per-Subject Null Comparison Grid

> **直接展示每个 subject 的多 null 对比**。审稿人能看到每个 subject 在 4 种 null 下都 robust，还是只在 N2 显著。

- **布局**：fwd/rev cohort 每 subject 一个子图，3 列 × ⌈n/3⌉ 行（n=6 → 2×3 grid）
- **每子图**：
  - X 轴：Δt（log），8 个刻度
  - Y 轴：`opposite_lift` (实线) 与 `same_lift` (虚线)，按 null 着色：
    - N0 浅灰 / N1 中灰 / N2 红 / N3 蓝
  - empirical 值 = 1.0（lift 是相对 null，所以 empirical 在 lift=1）→ 改为画 raw `P_opposite` 与 95% null bands
  - 实际画：每 subject 显示 `excess(Δt)` 在 4 个 null 下的曲线 + N2 的 95% null band（阴影）
- **文本**：每子图标题 `subject_id (n_events, n_T_a / n_T_b)`
- **关注点**：N2 红线显著 > 0 与 N3 蓝线**方向一致** = robust；N2 显著但 N3 不一致 = 触发 N4 follow-up

#### 主图 3 — Direction Asymmetry & Transition Odds

> **同时展示 H1b（方向对称）与 §3.6 secondary（next-event transition）**。

- **左 panel**：H1b direction symmetry scatter
  - X = `opposite_lift(T_a → T_b at 10s)`
  - Y = `opposite_lift(T_b → T_a at 10s)`
  - 每个点 = 一个 fwd/rev subject；对角线 y=x
  - 预期：点应靠近对角线（fwd/rev 假说不预设方向偏好）
- **右 panel**：next-event transition odds vs baseline
  - X = `baseline_odds`（per subject，i.i.d. 假设下）
  - Y = `transition_odds`（per subject，empirical）
  - 对角线 y=x；点在 y > x 一侧 = Ping-Pong 信号在 next-event 层面也成立
  - fwd/rev cohort 红，non-fwdrev cohort 灰对照

#### 主图 4 — Two Exemplar Subjects: Time Series + Pairing

> **故事性图**。挑 2 个 fwd/rev subject（最强信号 + 最弱信号各一），展示原始事件序列 + pairing 结构。

- **布局**：2 行 × 3 列
- **每行（一个 subject）**：
  - **左**（事件时间序列局部，e.g. 30 min 窗口）：上下两轨，T_a 红 raster，T_b 蓝 raster；标记 fwd/rev subject id
  - **中**：同 subject 的 `excess(Δt)` 曲线（重复主图 1 单 subject 版）+ N2 95% null band
  - **右**：raster 之上叠加 transition arrows — T_a 后 10s 内出现 T_b 的事件画连线；视觉化 Ping-Pong
- **故事线**：
  - 强信号 subject（如 fwd/rev 中 swap_score 最高）：连线密集
  - 弱信号 subject：连线稀疏
- **谨慎注释**：标注 "exemplar — not cohort claim"

#### Appendix 1 — N2 window sweep robustness

- 同主图 1 布局，三条红线对应 N2 window ∈ {10, 30, 60} min；预期**三条曲线重合**（H1 结论对 window 选择不敏感）

#### Appendix 2 — Packing window sensitivity

- 若 §9.4 confound profile B 触发，对 packing window ∈ {5, 15, 30}s 重跑 PR-2 cluster 后再做 PR-7，对比 cohort `excess(10s)`；预期**跨 packing window 一致**

#### Appendix 3 — Cohort audit transparency

- 表格图：每 subject 一行，列 `n_events_total / min_cluster_n / n_blocks / coverage_h / fwd_rev_source / pass`，pass 列着色（绿=入选，灰=排除）；审稿人可视化检查 audit logic

---

## 7. Step 拆分

| Step | 名称 | 交付 | 预算 |
|---|---|---|---|
| **PR-7 Step 0** | 主文档 §7 占位回写 + brainstorm doc 链接 | `docs/topic1_within_event_dynamics.md` §7 加 PR-7 一句话占位；review doc commit | 0.25 d |
| **PR-7 Step 1** | 代码层 + TDD（§8） | `src/template_temporal_pairing.py` + 4 个 shuffle helper + `compute_pairing_with_nulls` + `compute_transition_odds` + `evaluate_pass_criteria`；T1–T10 全绿 | 1.5 d |
| **PR-7 Step 2** | Cohort audit + per-subject 跑（fwd/rev cohort） | `pr7_cohort_audit.csv` + audit-pass fwd/rev subject 全部 per-subject JSON（含 N0/N1/N2/N3 4 nulls + transition odds） | 0.5 d |
| **PR-7 Step 3** | H1 primary cohort + 10s/30s 双门 + N3 robustness | fwd/rev cohort `excess(10s)` Wilcoxon + sign test；30s 同方向；N3 一致性 | 0.5 d |
| **PR-7 Step 4** | H2 negative control（non-fwdrev cohort） | non-fwdrev cohort 跑同一流程；H2 守住 / FAIL 判定 | 0.5 d |
| **PR-7 Step 5** | Robustness：null window sweep + N4 conditional follow-up | window 10/30/60 min sweep；N4 仅在 N2/N3 不一致时触发 | 0.5 d |
| **PR-7 Step 6** | **可视化（§6.5 全部 4 主图 + 3 appendix）** | `scripts/plot_pr7_template_pairing.py` + 7 张图 + `figures/README.md` 中文图说 + cohort verdict text 写入主图 1 | 0.75 d |
| **PR-7 Step 7** | 归档 + 主文档结论 | `docs/archive/topic1/pr7_template_pairing_results_2026-05-xx.md` + topic1 §7 一句话结论；本计划落入"历史索引" | 0.5 d |
| **总预算** | | | **5.0 d** |

---

## 8. TDD 测试合同（锁 16 项 — 含 2026-04-28 review 加固的 5 项）

> **2026-04-28 review 加固**：T11–T15 五项是科学合同 audit 后补上的关键测试，针对 N2 partition / cross-block 隔离 / 主 paired-test key-match / direction asymmetry 等容易被实现层悄悄破坏的合同。T16 锁 N4 stub 必须 raise，避免静默使用。

测试文件 `tests/test_pr7_template_pairing.py`：

```
T1. test_compute_pairing_lift_perfectly_alternating:
    times = [0, 1, 2, 3, 4, 5, ...]
    labels = [0, 1, 0, 1, 0, 1, ...]
    Δt = 1.5
    -> p_opposite ≈ 1.0, p_same ≈ 0.0
    -> 断言 opposite/same dominance

T2. test_compute_pairing_lift_independent:
    times = uniform random in [0, 1000]
    labels = bernoulli(0.5)
    Δt = 1.0
    -> p_opposite ≈ 0.5 × rate × Δt, p_same ≈ same
    -> excess ≈ 0 within tolerance

T3. test_compute_pairing_lift_block_boundary_isolation:
    构造 2 blocks，block A 末尾 label=0，block B 起始 label=1
    Δt 跨过 block 边界
    -> 跨 block 的 j 必须不计入 (确认 block_time_ranges 过滤)

T4. test_shuffle_labels_global_preserves_count:
    labels with 30/70 split
    -> shuffled 全局比例不变

T5. test_shuffle_labels_block_aware_within_block_only:
    block A: labels=[0,0,1,1], block B: labels=[1,1,0,0]
    shuffle 后：block A 仍 2x0+2x1，block B 仍 2x1+2x0
    断言跨 block label 不互换

T6. test_shuffle_labels_local_window_proportions:
    构造 [0min, 60min] 的事件，在 30min 窗口内一半 0 一半 1
    shuffle 后每个 30min 窗口的比例不变（容忍 boundary）

T7. test_shuffle_labels_circular_preserves_autocorrelation:
    构造周期 label = [0]*100 + [1]*100，shift 任意量
    后 lag-1 autocorrelation 不变（shift invariance）

T8. test_compute_transition_odds_alternating_vs_independent:
    alternating: P_next_opposite ≈ 1.0, transition_odds → ∞
    independent (50/50): P_next_opposite ≈ 0.5, transition_odds ≈ baseline_odds

T9. test_compute_pairing_with_nulls_full_pipeline:
    构造 antagonistic schedule（0,1,0,1,...）
    -> N0/N1/N2/N3 下 excess(10s) > 0 显著（p<0.05）
    -> 构造 independent schedule，excess ≈ 0 (p>0.5)

T10. test_evaluate_pass_criteria_triple_gate:
    case 1: 10s_wilcoxon_p=0.01, sign_p=0.03, 30s_median=0.05 -> pass=True
    case 2: 10s_wilcoxon_p=0.01, sign_p=0.03, 30s_median=-0.02 -> pass=False (30s reversal)
    case 3: 10s_wilcoxon_p=0.08, sign_p=0.02, 30s_median=0.05 -> pass=False (10s wilcoxon fail)

T11. test_n2_first_covering_partition_with_50pct_overlap (review-加固):
    构造 4 段不同 label 比例，验证 N2 50% overlap + first-covering:
    pool 0 = [0, 1800)，混 0/1; pool 1 = [1800, 2700)，homogeneous; pool 2 = [2700, 3600)
    -> 验证 pool 0 内部 mix，pool 1 / pool 2 各自保持 homogeneous 比例

T12. test_n2_does_not_cross_block_boundaries (review-加固):
    Block A [0, 1000) 全 label 0；gap；Block B [5000, 6000) 全 label 1
    用 window=10000s（远大于 block 距离），跑 N2:
    -> Block A 内仍全 0，Block B 内仍全 1（永不跨 block shuffle）

T13. test_evaluate_pass_criteria_rejects_key_mismatch (review-加固):
    excess_10s = {s1, s2, s3}; excess_30s = {s1, s2, s4}（s3 vs s4 不匹配）
    -> 必须 raise ValueError，提示 paired design 要求 subject-key 一致

T14. test_transition_odds_does_not_cross_blocks (review-加固):
    Block A 全 0，gap 1万s，Block B 全 1
    block-aware: p_next_opposite = 0（block 内全部 same）
    naive 单 block: p_next_opposite = 1/99（cross-block 那一对被错误计入）

T15. test_pairing_lift_direction_asymmetry (review-加固):
    Schedule [0,1,1,1,1,0,1,1,1,1,...]：T_a 后必接 T_b；T_b 大多接 T_b
    -> p_a_to_b > 0.9，p_b_to_a < 0.3，asym = p_a_to_b − p_b_to_a > 0.5

T16. test_resample_isi_per_cluster_raises:
    -> 必须 raise NotImplementedError，stub 不允许沉默返回
```

每条测试 < 30 行；总 16 项 < 500 行。

**注**：N4 (rate-matched ISI per cluster) 实现**故意 raise NotImplementedError**（src/template_temporal_pairing.py），不留 silent stub。N4 仅在 N2/N3 不一致时作 follow-up，届时实现完整 gamma-fit-per-window null 后 加专门 TDD 测试。

---

## 9. 失败合同（pre-registered）

写死 6 类失败 / 边界场景，避免事后 spin：

### 9.1 H1 NULL（cohort excess(10s) Wilcoxon p > 0.10）

→ 仅否定**这一种特定形式**的功能耦合：fwd/rev cohort 在 10s/30s 窗口内没有检测到"反向模板比同模板更易紧随"这一固定时间签名。**不否定**：
- PR-6 已建立的几何相关性（forward/reverse 模板共享同一网络，source/sink swap）
- 因果性本身（观测性事件序列 + 当前固定窗口 excess metric 的 null 不能支持"无因果"这种强论断）
- 慢时间尺度耦合（hours / circadian / seizure proximity 等被本检验设计性排除）

论文 framing（pre-registered）：
- ✅ 可以写："Forward/reverse propagation geometries coexist, but their event timing shows no robust short-window reciprocal coupling at the tested scales (Δt ∈ [10s, 30s])"
- ✅ 可以写："The bouncing-back / short-range reciprocal version of Ping-Pong is rejected; geometric coupling (PR-6 source/sink swap) is preserved"
- ❌ **不**写："两种模板时间上无关 / 无相关 / 无因果"
- ❌ **不**删除 PR-6 几何叙事；只撤回"短时接力"叙事
- 后续 follow-up（独立 PR）：历史依赖模型 P(next_label / hazard_opposite | recent history + local rate + block / state)，检验"加上前一个模板标签是否提高预测"——比固定窗口 excess 更不依赖单一时间尺度

### 9.2 H2 negative control FAIL（non-fwdrev cohort 也显著）

→ H1 阳性可能是事件 burst 普适特征而非 fwd/rev 特异。降级 H1 为 "general short-range clustering of HFO group events" 描述，不进 fwd/rev 机制叙事。

### 9.3 长尺度 lift 也 > 1（confound profile A）

→ 共享慢调制驱动；用 detrend 后重跑（如果 detrended 仍显著 → robust short-range coupling；若不再显著 → 短程 coupling 是慢调制 artifact）。

### 9.4 仅 packing-window 时间尺度上显著（confound profile B）

→ 可能是 group-event 定义内置的 within-window stickiness。**触发条件**：1s / 5s 显著 + 10s / 30s ≈ 0 或反向。**Sensitivity check**（appendix 2）：对不同 packing window（5/15/30s）重跑 PR-2 cluster 后再做 PR-7。如果跨 packing window 一致 → robust；若仅 default 5s 显著 → packing artifact，**主结论降级**为 "compatible with packing window stickiness, not independent short-range coupling"。

### 9.5 30s required sensitivity 反向（即 10s 显著但 30s 中位数 < 0）

→ 信号被 packing-edge 主导，**不**触发 PASS。报告 "10s peak excess but 30s reversal — likely packing-proximity artifact"，论文不能 claim short-range pairing。需走 §9.4 packing window sensitivity sweep。

### 9.6 N2 与 N3 在 H1 primary (10s) 上结论冲突

→ Null 模型敏感性问题。**触发 N4 conditional follow-up**（§4.1）。如果 N4 与 N2 一致 → 报告"信号在 N2 / N4 一致，N3 不一致；N3 可能因 circular shift 失去 cluster-rate 共享结构"；若 N4 与 N3 一致 → **降级为 exploratory**，并列报告所有 null，不用单一 null 做 cohort 判定。

### 9.7 Cohort 太小（fwd/rev reproduced 满足 §5 全 5 条入选条件 n < 4）

→ 走 case-series：每个 subject 单独报告 `excess(Δt) curve` + N0/N1/N2/N3 null bands + transition odds；**不**做 cohort Wilcoxon。论文层面写 "case-series of fwd/rev subjects with template antagonistic pairing"。**不**通过放松 §5 入选门槛保 cohort size。

---

## 10. 已知风险 / 边界

| 风险 | 描述 | 应对 |
|---|---|---|
| **Cohort 功效极有限** | fwd/rev reproduced n ≈ 6；§5 收紧门槛后预期 4–6；Wilcoxon 单边最小 p = 0.016（n=6, W=0）；二项 sign-test 6/6 ≈ p=0.031 | H1 PASS 三条同时（Wilcoxon p<0.05 + sign p<0.05 + 30s median>0）。提前在 plan 锁死，避免事后调整。n<4 走 §9.7 case-series |
| **Cluster id 任意性** | KMeans label 0/1 随机；不影响 H1（excess 对称化），但报告 H1b direction 时必须用 PR-2 inter-cluster Spearman 极性约定（按 first-half 命名 T_a / T_b） | 在 §3.1 已锁死 label normalization；T1 测试覆盖 |
| **Packing window stickiness** | 5–30s 内的 within-window dependency 可能本身就是 packing 制品 | §9.4 sensitivity check + Δt 多尺度 profile 帮助识别 |
| **Block 边界 spurious pairing** | 跨 block 的 t_j ∈ window 计数可能是 spurious | §6.1 实现：`compute_pairing_lift` **必须**只计入同 block 内的 j（block_time_ranges 过滤） |
| **Slow-modulation confound** | Topic 2 PR-2.5 已确认 IEI 慢漂移占比 72%；可能驱动所有 lift > 1 | §9.3 + N2 / N3 / N4 nulls 设计专门控制慢漂移 |
| **机制层 over-claim** | "Ping = excitation, Pang = inhibition" 在 HFO 上不可证 | §2 范围声明写死；论文 framing 仅作文献语言对接，不作 HFO 数据结论 |
| **Forward/reverse subject 内部异质** | n=6 subject 跨 dataset（5 epilepsiae + 1 yuquan）；effect size 可能差异大 | per-subject 报告 + cohort 仅用方向一致性 |

---

## 11. 显式不做（避免 scope creep）

- **不**重做 cluster 算法（KMeans k=2 锁死，复用 PR-2 stable_k）
- **不**重做 template centroid（复用 `adaptive_cluster.clusters[k].template_rank`）
- **不**做节点级 swap 几何（→ PR-8 candidate）
- **不**做 subject typology × PR-5 split（→ PR-9 candidate）
- **不**做 PCA / UMAP / MDS 降维（→ supplementary figure 后续，不进主线）
- **不**做 ER / CUSUM / ictal anchor / Schevon 复现
- **不**改 PR-2 / PR-2.5 / PR-3 / PR-4 / PR-5 / PR-6 任何代码或输出
- **不**做 HFO 80–250 Hz 之外的频段分析
- **不**做 inhibitory restraint 文献的形式化对接（仅在结果讨论时引用）

---

## 12. 与其他 PR 的边界

| PR | 关系 |
|---|---|
| **PR-2 / PR-2.5** | PR-7 输入；不动 |
| **PR-3 / PR-4A** | 不相关（per-subject visualization） |
| **PR-4B (rate state coupling)** | PR-7 不复用；PR-4B L2 是模板内顺序，PR-7 是模板间时序耦合 |
| **PR-4C (seizure proximity)** | PR-4C cohort null 已封板；PR-7 **不**做发作邻近窗口 |
| **PR-4D (rate × type)** | PR-4D 是描述层 envelope；PR-7 是统计层耦合检验，不冲突 |
| **PR-5 (template recruitment)** | PR-5 cohort 是 dominant template post-ictal rate↑；PR-7 fwd/rev cohort 与 PR-5 retained subset 独立 |
| **PR-6 (endpoint anchoring)** | PR-6 给出 cluster centroid 几何；PR-7 复用 cluster_labels；**audit cohort 直接复用 PR-6 §15 H2 cohort** |
| **PR-8 candidate (signed displacement)** | PR-7 PASS 后开 PR-8；PR-8 把"反向型 vs 主导型"做成连续几何 |
| **PR-9 candidate (typology × PR-5)** | PR-8 完成 typology 后，开 PR-9 复跑 PR-5 split |

---

## 13. 历史链接

- `docs/archive/topic1/ping_pong_hypothesis_review_2026-04-28.md` — 整体假说审阅与 roadmap，PR-7 是其中第一刀
- `docs/archive/topic1/pr6_template_endpoint_anchoring_plan_2026-04-25.md` — PR-6 cluster centroid 几何（PR-7 输入合同的来源）
- `docs/archive/topic1/interictal_group_event_internal_propagation.md` — PR-2 / PR-2.5 forward/reverse candidate 完整结果
