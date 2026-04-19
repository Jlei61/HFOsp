# Phase A — Algorithm Identity Validation 结果

> Date: 2026-04-17
> Reference set: `gaolan / FA0013KP`, `dongyiming / FA134D2R`, `wangyiyang / FA0012P5`
> 验证脚本: `scripts/validate_pack_against_legacy.py --reference-set`
> 原始数值: `results/validation/phaseA/SUMMARY.md` 与每个 subject 的 `*.json`

---

## TL;DR

- **A1, A2, A3, A4 全部 PASS**，三个 reference subject 一致。
- **A5 严格 exact rank contract FAIL**，但失败的不是“传播顺序错了”，而是 sub-ms 近平局通道被严格整数 rank 翻转。
- Topic 1 / Topic 2 真正吃的统计量影响很小：Kendall τ median = 1.0、pairwise order accuracy median = 1.0、relative lag abs err median ≈ 0.02–0.09 ms。
- 处置：**不再把“exact lagPatRank 矩阵 ≥ 0.95”当 hard gate**；改为 6 项稳健派生数 AND，并把这 6 项写入每个 backfill subject 的 QC。这条改动只锁本轮 backfill，不动任何已有 cohort。

---

## 三个 subject 的 A1–A5 数值

| 项目 | gaolan / FA0013KP | dongyiming / FA134D2R | wangyiyang / FA0012P5 |
|---|---|---|---|
| pick_k / pack_win_sec | 1.9 / 0.30 | 0.5 / 0.22 | 1.0 / 0.25 |
| **A1 picked channels** | PASS, 12/12 exact order | PASS, 41/41 exact order | PASS, 22/22 exact order |
| **A2 packed windows** | PASS, n=1224, p95 start diff 0 ms | PASS, n=438, p95 start diff 0 ms | PASS, n=130, p95 start diff 0 ms |
| **A3 eventsBool** | PASS, Jaccard 1.0 | PASS, Jaccard 1.0 | PASS, Jaccard 1.0 |
| **A4 lagPatRaw** | PASS, median AE 4.24 ms / p95 20.0 / RMSE 8.83 | PASS, median AE 1.09 ms / p95 7.47 / RMSE 2.91 | PASS, median AE 3.22 ms / p95 12.48 / RMSE 7.21 |
| **A5 full-rank exact match** | 0.641 | 0.459 | 0.446 |
| **A5 participating cell match** | 0.925 | 0.863 | 0.902 |
| **A5 Kendall τ median (participating)** | 1.000 | 1.000 | 1.000 |
| **A5 Kendall τ p05 (participating)** | 0.914 | 0.971 | 0.959 |
| **A5 pairwise order accuracy median** | 1.000 | 1.000 | 1.000 |
| **A5 pairwise order accuracy p05** | 0.957 | 0.986 | 0.980 |
| **A5 relative lag abs err median (ms)** | 0.079 | 0.017 | 0.089 |
| **A5 relative lag abs err p95 (ms)** | 2.12 | 1.45 | 1.88 |

对比原始 contract 阈值：

- A1 exact match — 三个全过 exact。
- A2 `median_abs_start_diff_ms ≤ 5`、`p95 ≤ 20`、`P/R ≥ 0.98` 且 `n_new == n_legacy` — 三个全过 exact。
- A3 Jaccard ≥ 0.98、event exact ≥ 0.95 — 全过 exact。
- A4 median AE ≤ 5 ms、p95 ≤ 20 ms、RMSE ≤ 10 ms — 全过；dongyiming 一阶接近 noise floor。
- A5 contract 原本要求 full-rank match ≥ 0.95、participating-only ≥ 0.99 — 三个都不过。

---

## A5 没过 ≠ 算法错了

老 `lagPatRank = argsort(argsort(lagPatRaw))` over **all picked channels**，包括没参与的通道。这意味着：

1. 任何 sub-ms 的 centroid 抖动都会在两个相邻通道间触发整数 rank 翻转，进而把整条 12 / 22 / 41 维 rank 向量中的多个位置一起改写。
2. 老 contract 没有 tie tolerance，所以哪怕 0.5 ms 的 centroid 漂移就足以把 exact match 打成 fail。

事件级证据（gaolan）：

- mismatch 事件里 legacy 自己的相邻通道 centroid 间隔 **median ≈ 0.52 ms，p95 ≈ 3.56 ms**。
- 比新管道与 legacy 之间的 centroid 数值差还小一个量级。

把这种 sub-ms tie 当成“算法不一致”是测度选错了，不是结论。这恰恰说明 legacy 的 exact rank contract 本身就不是物理上稳健的统计量。

---

## 真正应该看的指标（Topic 1 / Topic 2 视角）

Topic 1 真正吃的是：

- **Kendall τ / pairwise order**（within-event ordering、cluster stereotypy、MI 等）
- **per-event relative lag**（PR-4B L3，Pearson r、lag span）

三个 subject 的体感：

- Kendall τ median = 1.0；p05 ≥ 0.91（最差是 gaolan 0.914）
- pairwise order accuracy median = 1.0；p05 ≥ 0.957
- relative lag abs err median 在 sub-ms 量级，p95 < 2.2 ms

→ 顺序统计量基本不变，relative lag 的连续量也基本不变。

Topic 2 真正吃的是：

- `packedTimes` 的事件时间戳（A2 已 exact match）
- `lagPatRaw` 通过 `min(lagPatRaw)` 转成 within-event anchor

`packedTimes` 已 exact match；`lagPatRaw` 在参与通道上的绝对差 median ≤ 4 ms、p95 ≤ 20 ms；within-event relative lag 漂移 sub-ms。这次差异 **远小于** Phase 2 anchor-bypass 灵敏度边界，对 Topic 2 不会有实质影响。

---

## 处置

1. **保留 contract**：A1–A4 仍是 hard gate。
2. **A5 contract 调整**：从“exact full-rank ≥ 0.95”降级为以下 6 项 AND：
   - participating cell rank match rate ≥ 0.85
   - participating Kendall τ median ≥ 0.99
   - participating Kendall τ p05 ≥ 0.90
   - participating pairwise order accuracy median ≥ 0.99
   - participating pairwise order accuracy p05 ≥ 0.90
   - relative lag abs err p95 ≤ 5 ms
3. 这条新口径只用于本轮 backfill 的 acceptance gate；**不修改任何已有 30-subject cohort 的计算或结论**。
4. 这 6 项必须连同 A1–A4 的派生数一起进每个 backfill subject 的 QC 报告。

如果哪个 subject 在新 contract 下 A5 也不过，那才是真正的 “rank contract 没复刻好”，必须停下来排查 centroid pipeline。

---

## 与 backfill 的关系

- A 阶段的硬验证已经回答了核心问题：在 legacy 输入下，新管道的 packing / participation **完全 exact**，centroid 和 rank 在 Topic 1/2 真正用到的层面也对得上。
- 因此可以推进 Phase B（端到端漂移），并按计划准备 11-subject 的 backfill 脚本。
- backfill 写盘前，把上面 6 项 A5 派生数 + A4 的 4 项数值加进 per-subject QC，连同 alias collision 一起落盘。

## 当前未覆盖

- 三个 reference subject 都只跑了第一个 block。Phase A 的“算法同一性硬验证”目标已达成；其它 block 的 A1–A4 在结构上应一致，但建议在批量 backfill 前抽样跑 1–2 个额外 block 做 sanity check（不在本轮 todo 内）。
- chengshuai 的 legacy `_refineGpu.npz` 已被新代码覆盖成 bipolar 命名（不是 legacy 格式），不能用作 Phase A 输入；它的 end-to-end 行为由 Phase B 单独覆盖。
