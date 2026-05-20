# Step 5b — PR-2.5 split-half / odd-even 修过版重跑结果（2026-05-20）

> 状态：Step 5b 已完成。Checkpoint A 评估**待 advisor consult**（fwd/rev 集合 5/16 翻动属"边界级"）。
> 主入口：`docs/topic0_methodology_audits.md`
> 5a 结果：`./step5a_pr2_results_2026-05-20.md`
> 修过版结果：`results/interictal_propagation_masked/per_subject/*.json`（每个 JSON 顶层 `time_split_reproducibility` 字段）

---

## 1. 三段式朴素话

**测了什么** —— 同一个 subject 的事件按时间一半一半切开（或按 block 奇偶切开），两半分别重新做 KMeans 聚类，看两边算出的 cluster 模板长得像不像、能不能把对方的事件归对类。**这是检验 PR-2 聚类不是过拟合**——如果两半数据各自跑出完全不同的模板，那"K=2 主导"就是凑出来的；如果两半模板高度相似，证明这是数据里真有的结构。修过版用的是去掉假名次的 feature 矩阵。

**怎么测的** ——
1. 用 Step 5a 写好的 masked PR-2 labels 作为 `chosen_k` 和 `adaptive_labels` 的来源。
2. 对每个 subject 做两种 split：
   - **split-half**: 按事件绝对时间排序后前一半 vs 后一半
   - **odd-even-block**: 偶数 block 序号 vs 奇数 block 序号（控制 block 边界效应）
3. 每个 split 的两半都用 `use_masked_features=True` 重新 KMeans 到原 `chosen_k`，匹配两边模板（Hungarian on Spearman correlation）。
4. 计算 `mean_match_corr`（两半模板的 Spearman 相关）和 `assignment_agreement`（用 A 的模板给 B 的事件分类，对的比例）。reproducibility_grade = strong / moderate / weak。
5. 找出"forward-reverse 模板对"在两个 split 里能不能被对应识别 — `forward_reverse_reproduced` 用 split-half OR odd-even 取并集（AGENTS.md cross-PR 合同规则）。

**揭示了什么** ——
- **整体结构稳健**：26/40 仍 strong，12/40 moderate，仅 2/40 weak（huanghanwen, litengsheng）。grade 分布相对原版（31/9/0）整体降了一档，但没有大量 strong → weak 的崩塌。
- **forward-reverse 反向模板对**：orig 16/17 vs masked 15/16，绝对数仅差 1，但**子集成员发生 5 个翻动**（3 失：548/620/635；2 得：253/916）。这是 Checkpoint A 关注的红线之一。
- **916 的 stable_k 翻转伴随 fwd/rev 集合从"无"变"有"**：原版 916 stable_k=2 但 fwd/rev pair 不复现；masked 版 stable_k=4 但 fwd/rev pair 复现（在新的 4-cluster 几何下）。说明 phantom 在 916 上同时压低了 stable_k 选择和反向模板识别。
- audit ↔ rerun 一致性方向继续：Step 5a 已经 ρ=0.961，5b 在 grade 层面也维持同方向（grade 降级的 9 个 subject 主要是原版 strong → 修过版 moderate，符合"phantom 让聚类看上去更稳定"的预期）。

代号补注：PR-2.5, `compute_time_split_reproducibility`, `forward_reverse_reproduced`, AGENTS.md cross-PR contract (split-half OR odd-even rule), 修过版 = `use_masked_features=True`。

---

## 2. Cohort 数字表

### 2.1 Grade 分布

| grade | orig (phantom) | mask | Δ |
|---|---:|---:|---:|
| strong | **31** | **26** | -5 |
| moderate | 9 | 12 | +3 |
| weak | 0 | **2** | +2 |
| total | 40 | 40 | — |

**Grade 翻转明细（9/40 subjects）**：

| sid | orig grade | mask grade | 方向 |
|---|---|---|---|
| epilepsiae_1084 | strong | moderate | ↓ |
| epilepsiae_1150 | strong | moderate | ↓ |
| epilepsiae_620 | strong | moderate | ↓ |
| epilepsiae_916 | strong | moderate | ↓ |
| yuquan_gaolan | strong | moderate | ↓ |
| **yuquan_huanghanwen** | moderate | **weak** | ↓↓ |
| **yuquan_litengsheng** | moderate | **weak** | ↓↓ |
| yuquan_zhangkexuan | moderate | **strong** | ↑ |
| yuquan_zhourongxuan | strong | moderate | ↓ |

7 个降级、1 个降到 weak（huanghanwen / litengsheng）、1 个 zhangkexuan 反向升级。

### 2.2 Forward / reverse pair 复现（split-half OR odd-even）

| | orig | mask |
|---|---:|---:|
| 有 pair available 的 subject 数 | 17 | 16 |
| reproduced（fwd/rev pair 在 split 内被对应识别） | **16/17** | **15/16** |
| 两边都 reproduced（intersection） | — | **13** |
| **orig-only**（masked 后丢了 reproduction） | — | **`548, 620, 635`**（3） |
| **mask-only**（masked 后新增 reproduction） | — | **`253, 916`**（2） |

### 2.3 Match correlation cohort 中位数

| metric | orig (phantom) | mask | Δ |
|---|---:|---:|---:|
| split-half median match_corr | (need recompute) | **0.851** | — |
| odd-even median match_corr | (need recompute) | **0.942** | — |
| split-half median agreement | — | **0.870** | — |
| odd-even median agreement | — | **0.891** | — |

odd-even 高于 split-half 是已知现象（odd/even 切片在每个 block 内部分摊，混合更均匀；split-half 按时间硬切，可能跨过一些状态边界）。

---

## 3. Checkpoint A 评估 — 待 advisor

按 `./rerun_roadmap_2026-05-20.md` §"Checkpoint A" 的合同：

| 触发条件 | 实测 | 评估 |
|---|---|---|
| 5a stable_k flip 主线 cohort 超 3 个 | 1（仅 916） | ✓ 远低于阈值，PASS |
| 5b reproduced 集合大幅重组 | 5/16 = **31% turnover** | ⚠ 边界 |

5/16 turnover 不是显然的"大幅重组"也不是显然的"集合稳定"——属于 plan 没有量化阈值的边界区域。需要 advisor consult 才能定：

- 是 PASS → 进 5c (PR-3 MI on masked labels)
- 还是 PAUSE → 跑 impute 策略 sensitivity（channel_median 或 drop_low_participation）后再 reconcile

**支持 PASS 的证据**：
- 主线 cohort stable_k flip 仅 1 个，结构层稳健
- grade 整体降一档但无大量崩塌
- 翻动方向科学上可解释（916 在两边都"边缘"；548/620/635 是原 phantom 假信号被去掉；253/916 是去掉 phantom 后真信号显出）
- audit↔rerun ρ=0.961 仍然成立，masking 策略本身没引入偏置

**支持 PAUSE 的证据**：
- 5/16 = 31% turnover 在直觉上"不小"
- huanghanwen/litengsheng 从 moderate 掉到 weak（grade 跌两档），值得查
- 没跑 sensitivity 就直接进 5c 风险是：万一 PR-3 MI 出现集体降级，回头不知道是 PR-3 本身的问题还是 masking 还有空间调

---

## 4. 输出位置

- per-subject masked PR-2.5 字段：`results/interictal_propagation_masked/per_subject/<sid>.json` 顶层 `time_split_reproducibility` 字段（40 files）
- cohort summary：`results/interictal_propagation_masked/pr1_cohort_summary.json`（含 PR-2.5 reproducibility grade dist 顶层字段）
- run log：`logs/step5b_pr25_masked_cohort.log`（约 70s wall）

---

## 5. 与 5a 状态联动

5a 的判定：主线 cohort label-level shift 大但 chosen_k 稳健 → 重跑 advisable
5b 的判定：grade 整体降一档，fwd/rev set 5/16 翻动 → consult advisor

整体一致性：phantom 给"事件归类"和"反向模板对识别"都掺了水分；去掉后**结构层（K=2、reproducibility grade）总体仍站得住**，但**具体哪些 subject 满足哪些次级标准会变**。

5c (PR-3 MI) 启动前必须先把 Checkpoint A 通过 / 拒绝的决定写定。
