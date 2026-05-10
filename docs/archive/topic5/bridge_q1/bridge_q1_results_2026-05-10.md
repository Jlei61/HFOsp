# Topic 1 × Topic 5 Bridge Q1 — 探索性 cohort 结果 (2026-05-10)

> **🔄 PIVOT 2026-05-10 (later same day)**: 本文档记录的 Q1 (state fingerprint) NULL-locked 结果作 phase-1 negative-control 保留，**已被 Q1' (channel-rank correspondence) 取代**作为主 axis。详见 `docs/archive/topic5/bridge_q1prime/bridge_q1prime_results_2026-05-10.md`。


> **Tier**: 探索性，cohort N=9（spec 设计 N=10，1096 因 topic1 PR-2 lagPat lineage data freshness 失效退出）。
> **Verdict**: **NULL-locked**（0/3 windows 通过 PER-WINDOW-PASS）。
> **Spec**: `docs/superpowers/specs/2026-05-10-topic1-topic5-bridge-design.md`
> **Plan**: `docs/superpowers/plans/2026-05-10-topic1-topic5-bridge-q1.md`
> **代码**: `src/topic1_topic5_bridge.py`, `scripts/run_topic1_topic5_bridge.py`, `tests/test_topic1_topic5_bridge.py`
> **结果**: `results/topic1_topic5_bridge/` (cohort_summary.json, per_subject/*, q1b_442_sentinel.json, figures/)

## 1. 一句话结论

在 epilepsiae N=9 探索性 cohort 上，pre-ictal HFO group-event template (T0/T1) 指纹（frac_T0 / switch_rate / last_template）在 same-feature dual gate (α/3=0.0167 AND |effect|>0.10) 下，3 个 sensitivity 窗口（[-15,-1] / [-30,-1] / [-60,-1] min）一致 0 subject 通过。Cohort verdict = **NULL-locked**。442 binary-outlier sentinel descriptive 信号方向正（+0.64 / +0.87 effect on frac_T0），但 n_outlier=1 决定 MW p ≥ 2/(n+1) ≈ 0.13，无法显著。

## 2. Cohort

| 状态 | 数量 | subjects |
|---|---|---|
| 设计 cohort (spec §3.1) | 10 | 1073, 1096, 1146, 253, 548, 590, 635, 916, 922, 958 |
| 实际有效 | 9 | 上面剔除 1096 |
| 1096 退出原因 | 1 | adaptive_cluster.labels 长度 223264 ≠ _valid_event_indices(min_participating=3) 的 223212；52-event 差距，疑 PR-2 stable_k 时刻使用的 lagPat 与现况不一致（topic1 lineage robustness 信号） |
| Q1b sentinel | 1 | 442 |
| broad-band sensitivity | 1 | 1084 |

## 3. Per-window 结果

| Window | n_positive | denom | binomial p (vs p_null=0.049) | PER-WINDOW-PASS |
|---|---|---|---|---|
| [-15.0,-1.0] | 0 | 9 | 1.0000 | False |
| [-30.0,-1.0] | 0 | 9 | 1.0000 | False |
| [-60.0,-1.0] | 0 | 9 | 1.0000 | False |

3-state verdict = **`NULL-locked`** (0/3 windows pass AND all counts ≤ 1).

## 4. Q1b 442 binary-outlier (descriptive only)

| Window | n_outlier | n_main | frac_T0 p | frac_T0 effect |
|---|---|---|---|---|
| [-15.0,-1.0] | 0 | 12 | n/a (skipped) | n/a |
| [-30.0,-1.0] | 1 | 14 | 0.349 | +0.643 |
| [-60.0,-1.0] | 1 | 15 | 0.192 | +0.867 |

**Effect 方向**：sz=9 outlier 在 [-30,-1] 与 [-60,-1] 窗口内 frac_T0 高于 main subgroup（+0.64 / +0.87）。
**统计意义**：n_outlier=1 决定 MW 最小 p ≈ 2/(n+1)；即使 effect 极大也无法跨过 α/3=0.0167。
**结论**：descriptive case study；不构成 cohort claim 也不构成"方法学验证"。

## 5. Q3 stratifier 描述层 (非 α 检验)

9 个 cohort subject 在 swap_class × silhouette 二维上的分布：

| | silhouette > 0.5 | silhouette ≤ 0.5 |
|---|---|---|
| swap real (strict ∪ candidate) | 1073, 1146, 548 | 635, 958 |
| swap none | 590, 916 | 253, 922 |

**所有 4 格全部 0 positive**——topic1 几何强弱与 silhouette 二分都没有把信号挽救出来。

## 6. 三条结构性发现

### 6.1 Audit 估值 5× 高估事件数

| Subject | audit (block-proxy) | 真实 (event-level) | 倍率 |
|---|---|---|---|
| 442 | 21 | 4 | 5.3× |
| 548 | 17 | 3 | 5.7× |
| 922 | 1300 | 1211 | 1.07× |

低-mid rate subject 估值严重虚高；high-rate subject 几乎无差。**未来 audit 不再用 block apportionment proxy**。

### 6.2 Dual gate 在小 subtype-balanced 样本上结构性低功效

| subtype 平衡 | MW 最小 p (perfect separation) | 能否过 α/3=0.0167? |
|---|---|---|
| 4 vs 4 | 2/70 = 0.029 | ❌ |
| 4 vs 5 | 2/126 = 0.0159 | ✅ 临界 |
| 5 vs 5 | 2/252 = 0.008 | ✅ |
| 16 vs 1 (442) | 2/17 = 0.118 | ❌ |

cohort 多数 subject 的 subtype 大小分布是 `{大主流, 1-3 minority}` 或 `{4, 4}`。**locked 阈值在这个 cohort 上几乎不可能触发 PER-WINDOW-PASS，无论真实 effect 多大**——是 power floor 的 NULL，不是科学意义上的 NULL。

### 6.3 1096 lineage robustness 信号

`adaptive_cluster.labels` 与 `_valid_event_indices` 的 size 差 52 events 提示 PR-2 stable_k 落盘时点与当前 lagPat NPZ 版本不同步。Spec 锁的 alignment guard 正确 raise；runner 优雅 skip + 写入 status="failed" JSON。**这是 topic1 数据合同问题，本 PR 把它彻底 surface 出来**。

## 7. 显式不做（locked from spec §6 + 用户决策）

- 不下"bridge 失败"的 framing；NULL-locked = "bridge 3 features 在当前 dual gate + cohort sample size 下不足"
- 不在 NULL-locked 上写 paper-level claim
- Q2 (z-ER channel-onset rank × template rank) 仍 deferred（spec §2.3）
- Stouffer Z secondary 仍 deferred（first-round scope locked）
- 不重新跑 PR-2 cluster pipeline；1096 lineage 修复独立后续 PR

## 8. 推荐下一步（决策点）

按 spec §1 + 用户 step-7 路线图，NULL-locked 触发以下三条候选路径：

1. **接受 NULL，启动 feature menu**（spec `2026-05-10-topic5-subtype-onset-features-menu.md`）找更强 pre-ictal discriminator。**风险**：§6.2 power floor 在新 feature 上仍生效；同 cohort/同阈值大概率仍 NULL。
2. **检讨 dual gate 阈值**：α/3 = 0.0167 是否过严？effect_min = 0.10 在小样本上是否合适？回 spec 改阈值是新决策点，不是隐式调整。
3. **重审 cohort 选择**：subtype 大小过小（< 4 either way）的 subject 是否应预先 sensitivity 排除？这会进一步缩 cohort（可能 N → 5-7）。

**默认建议**：先在主文档（topic1 + topic5）回链本档案，然后用户决定走 (1)/(2)/(3)。

## 9. 文件清单

### 代码
- `src/topic1_topic5_bridge.py` — 核心模块（fingerprint extractor + per-subject test + cohort aggregator + sentinel + Q3 + 5 figure helpers）
- `scripts/run_topic1_topic5_bridge.py` — CLI driver: `setup / per-subject / cohort / sentinel-442 / figures`
- `tests/test_topic1_topic5_bridge.py` — 28 tests

### 数据
- `results/topic1_topic5_bridge/bridge_setup.json` — T0/T1 freeze + audit-rerun marker (12 subjects, 1 dropped = 1096)
- `results/topic1_topic5_bridge/per_subject/*.json` — 12 个 (10 cohort + 442 + 1084)
- `results/topic1_topic5_bridge/cohort_summary.json` — verdict 与 per-window
- `results/topic1_topic5_bridge/q1b_442_sentinel.json` — 442 binary-outlier descriptive
- `results/topic1_topic5_bridge/figures/` — 5 PNGs + README.md (中文)
