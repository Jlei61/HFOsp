# Topic 5：频段 × 表示阶梯 × 共线分层 —— 描述性地图（设计合同）

> **定位（锁死）：本轮是地图，不是结论。** 不预注册主格、不下队列主张、不挑显著格。
> 目的是把 "raw / 轴 / field 哪个表示是对的" 这个问题所缺的格子补齐，看清地形，**之后**再另开 spec 决定预注册什么。

---

## 0. 一句话承诺（大白话，CLAUDE.md §8）

我们要看：**发作刚开始时哪些触点更亮，跟这个病人平时（间期）高频事件反复走的那条传播路，到底有多像。**

这件事的答案会随三个旋钮翻转，而我们从来没把三个旋钮同时摊开看过：

1. **用哪个频段看**——1–45 Hz、1–150 Hz、还是高频 60–100 Hz；
2. **用多少几何看**——完全不用触点位置（只把"平时的早晚顺序"和"发作时的亮度"直接比）、还是把触点投到一张平面上抹开成一片再比、还是再铺成网格再比；
3. **拿什么当"如果是随机会怎样"**——把发作亮度在所有触点间随便打乱（粗对照），还是只在**每根电极杆内部**打乱、保住"哪根杆整体更热"这个插电极的几何（严对照）。

并且按**这个病人的 A、B 两条传播方向是不是同一条线上的一来一回**分层看。

**为什么非做不可**：同一批数据，最宽分母 + 1–45 Hz + 严对照 → 看起来完全没信号（p=0.676）；换到"两条方向确实一来一回"的子集 + 1–150 Hz + 同样的严对照 → 强信号（p=0.003）。这两个数都是真的，都在同一个结果文件里。**在把地形看清之前，任何一个都不该被当成结论。**

（内部归档代号：contact-similarity ladder R1/R2/R3、`bb_auc`/`bb150_auc`/`hfa_auc`、within_shaft / channel shuffle null、`axis_cohort.csv::relation`、`template_axis_field`。）

---

## 1. 为什么做这个

### 1.1 现有证据的格子结构

现有结论横跨约 60 个格子（3 频段 × 4 分母 × 2 null × own/shared），且**结论随格子翻转**：

| 分母 | 频段 | 严对照 p（own_maxab） |
|---|---|---|
| 全部二维几何 n=17/18 | 1–45 | 0.676 |
| 全部二维几何 n=17/18 | 1–150 | 0.337 |
| 共线 n=7 | 1–45 | 0.053 |
| **共线 n=7** | **1–150** | **0.003** |
| 共线 n=7 | HFA | 0.042 |

### 1.2 缺的那一格

"raw vs field" 的承重比较（阶梯 R1/R2/R3）**只跑过 1–45 和 HFA、且只在最宽分母 n=18 上**：

- broadband(1–45)：R1=6 / R2=5 / R3=4 过严对照
- HFA：R1=9 / R2=7 / R3=5 过严对照

**从没跑过 1–150；从没按共线分层。** 而 1–150 恰恰是相对 null 余量最大的频段，共线恰恰是唯一让严对照过关的分层。所以目前**无法**回答用户的问题："raw contact、轴、field，哪个是对的？"

### 1.3 阻塞已解除

07-13 的建轴方法比较报告记录 "1–150 Hz 被数据阻塞（cache 无 `bb150_auc`）"。**该记录已过时**：`results/topic5_ictal_recruitment/t0_feature_cache_v2_windows/` 现含 `bb150_auc` + `bb_auc` + `hfa_auc` + `bact`，覆盖 20 人。

---

## 2. 跑什么

三个维度全交叉 + 分层：

- **频段**：`bb_auc`(1–45) / `bb150_auc`(1–150) / `hfa_auc`(60–100)
- **阶梯**：R1 无几何 / R2 同平面核（触点评估）/ R3 网格场
- **null**：`channel`（粗）/ `within_shaft`（杆内，解剖控制）；`anchor_matched` 顺带产出不作主线
- **分层**：`relation` ∈ {reversed, same, different}

顺带沿用阶梯已有的 `R2_sigma_sweep`（核宽 ×0.5/×1/×2）与 `sequence`（Spearman/Kendall）两条既有轨道，不新增。

---

## 3. 输入（锁死）

| 项 | 值 | 说明 |
|---|---|---|
| cache | `results/topic5_ictal_recruitment/t0_feature_cache_v2_windows/` | 20 人，三频段 key + `bact` 齐全 |
| 平面 | `results/spatial_modulation/propagation_geometry/observation_readout/real_subjects/{ds_sid}_t_a.json` 的 `x_norm`/`y_norm`/`sigma_xy` | **不改**，沿用阶梯现有端点轴 readout |
| 分层来源 | `results/topic5_ictal_recruitment/template_axis_field/axis_cohort.csv::relation` | 纯间期定义、冻结于发作读出之前 |
| B | 1000 | 沿用 |
| seed | `RNG_SEED`（脚本现值） | 沿用 |

**三频段共用同一 cache** 是本轮相对旧阶梯（用 `t0_feature_cache`，19 人、无 1–150）的**唯一输入改动**，目的是消除现有频段 panel 里 1–45 n=17 / 1–150 n=18 的分母错配。

**平面为什么不改**：建轴方法已在 07-13 报告验证对发作读出打平（宽带 median margin 梯度 +0.422 vs 端点 +0.422）。沿用现有 readout 记录符合 CLAUDE.md §6 "re-use don't re-invent"。

---

## 4. 分母（已逐个核过 `axis_cohort.csv`）

| 层 | n | 被试 |
|---|---|---|
| **reversed** | **6** | E1084 / E1146 / E384 / E583 / E590 / E958 |
| same | 1 | E548 |
| different | 11 | E1077 / E1096 / E1125 / E1150 / E253 / E442 / E620 / E635 / E922 / yuquan_xuxinyi / yuquan_zhangkexuan |

排除记录（必须显式落表，不得静默）：

- **E139**：`relation=reversed` 但 `n_shafts=1` → 阶梯自身 `single_shaft` gate 排除（杆内 null 无自由度）
- **E916**：`stable_k≠2` → 无 A/B 双轴

**`same` 只有 1 人，不成层**：逐个报，不参与任何层级统计。

---

## 5. 每格报两个数（2026-07-15 用户拍板）

1. **队列层**：`obs_median` vs null median 分布 → `p_upper`
2. **逐人层**：`k/n`（各自 `obs_subject > null_q.p95` 的人数）

**两个数不一致本身就是结果。** 现有表里已出现队列 p=0.003 但逐人只 2/7 过 —— 这是"跨病人一致的小偏移"，不是"几个病人很强"。n=6 时队列 p 极易被一两个病人带动，`k/n` 是揭穿这一点的唯一手段。

---

## 6. 禁止（本轮定位边界）

- **禁**把任何一格写成"结论"或"预注册主结果"。本轮是地图。
- **禁**挑最显著的格子当主格。要锁主格，必须在看完地图后**另开 spec 声明**，并论证该频段/分母的选择**独立于本轮 p 值**。
- **禁**把 1–45 的旧阶梯数字与本轮新数字混报（cache 不同：19 人 vs 20 人）。旧值只作对照列，显式标注来源。
- **禁**把阶梯的 R2/R3（**端点轴**平面）与 `template_axis_field` 的 own/shared（**梯度轴**平面）逐格对齐 —— 不是同一套平面，只可同向比较趋势。
- 仍是 **early-ictal readout，不是 early-ictal-specific**：已有时间负对照显示远端发作前场并不弱于 0–10 s 发作场。本轮不碰这个问题。
- **禁**把共线解释为同一条白质束或因果传播通路。

---

## 7. 产出

```
results/topic5_ictal_recruitment/contact_similarity/frequency_ladder_map/
├── ladder_map.csv          # band × rung × null × stratum × {obs_median, null_median, p_upper, k, n}
├── cohort_summary.json     # 正式聚合入口 + 排除记录 + caveats
├── per_subject/<ds_sid>_<band>.json
└── figures/
    ├── README.md           # 中文，图生成后写
    └── frequency_ladder_map.png
```

---

## 8. 改动清单

### 修改
- `scripts/run_topic5_contact_similarity.py`
  - `ACTIVATION_KEY` 加 `"bb150": "bb150_auc"`
  - `_ctx` 的 cache 目录改为可配置参数，**默认值不变**（`t0_feature_cache`），保持所有既有调用行为逐位不变

### 新增
- `scripts/build_topic5_frequency_ladder_map.py` — 跑 3 频段 × 全队列 + join 分层 + 出表 + 出图
- `tests/test_topic5_frequency_ladder_map.py`

### TDD（先写先失败）
| 测试 | 必须的行为 |
|---|---|
| `test_bb150_activation_key` | `ACTIVATION_KEY["bb150"] == "bb150_auc"`，且 `--activation bb150` 可解析 |
| `test_default_cache_dir_unchanged` | 不传 cache 参数时解析到 `t0_feature_cache`（**回归**：旧行为逐位不变） |
| `test_custom_cache_dir` | 传入自定义 cache 时读取该目录 |
| `test_stratum_join_key_alignment` | 分层 join 按 `subject_id` 对齐；`axis_cohort.csv` 缺失被试显式落 drop，不静默丢 |
| `test_same_stratum_not_aggregated` | `same` 层（n=1）不产出层级统计，只出逐人记录 |
| `test_both_numbers_reported` | 每格同时含 `p_upper` 与 `k`/`n`，缺任一即失败 |

---

## 9. 失败合同（fail-closed）

- cache 缺某频段 key → 该频段显式 drop 记录，不静默跳过
- 被试在 `axis_cohort.csv` 里无 `relation` → 落 `stratum_unknown` drop 记录，不归入任何层
- 单杆 / `stable_k≠2` → 沿用阶梯既有 gate，落排除表
- 任一格 `effective_shuffle_n < 4` → 标 `INSUFFICIENT_NULL`，**不计为通过**（沿用 `subject_null` 既有语义）

---

## 10. 看完地图之后（非本轮范围）

地图出来后需要回答、但**本轮不回答**的问题：

1. 主格锁哪个（频段 × 分母 × null × 表示）——须论证独立于本轮 p 值
2. 1–150 作为主频段的理由是否独立于这些结果（用户 07-14 已为发作能量线锁 1–150，需确认该锁不是被本线结果驱动）
3. `template_axis_field` 归档的主读出（1–45）是否需要按新主格重出
