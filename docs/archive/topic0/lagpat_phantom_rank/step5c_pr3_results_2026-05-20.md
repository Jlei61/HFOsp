# Step 5c — PR-3 per-cluster MI 修过版重跑结果（2026-05-20）

> 状态：Step 5c 完成。无需重新跑——5a 主管道已经把 per-cluster MI / within-cluster centered τ / bias_fraction 都计算到 masked per-subject JSON 里。本文档是 cohort 层面的对比汇总。
> 主入口：`docs/topic0_methodology_audits.md`
> 5a / 5b 结果：`./step5a_pr2_results_2026-05-20.md` / `./step5b_pr25_results_2026-05-20.md`

---

## 1. 三段式朴素话

**测了什么** —— 拿 5a 修过版的 cluster labels（"哪个事件属哪一类"已经重新算好了），看两件事：
1. legacy MI（每个 subject 的整体"通道激活顺序刻板程度 vs 随机置换"的对比）会不会变——这是 PR-3 的核心数字，应该不变（因为 MI 算法本身按 `eventsBool` 做了 mask，理论上 phantom 影响不到它）。
2. 簇内时序刻板程度（within-cluster raw τ 和 centered τ）+ identity-bias fraction（簇内刻板里有多大比例来自"哪几个通道天然位置更靠前"）。**这个会受影响**——phantom 把不同簇的事件混在一起会稀释簇内一致性。

**怎么测的** —— 5a 已经把 masked PR-2 跑了一遍。每个 subject 的 JSON 里 `legacy_mi`、`adaptive_cluster.clusters[].raw_tau/centered_tau`、`within_cluster_centered.mean_*` 都是修过版的数。读 40 个 subject 的 orig vs mask JSON，对每个 metric 做 paired Wilcoxon。

**揭示了什么** ——
- **legacy MI 没变**（max |Δ| = 0.000017；40/40 在 orig 和 mask 下都 p<0.05）。验证了"bools-masked 的 MI 路径确实不受 phantom 污染"这个 audit 前的假设。PR-3 "30/30 MI 显著" 结论站得住。
- **簇内 raw τ 增强**：orig 中位 0.237 → mask 0.291（+0.054），39/40 subject 增强，Wilcoxon p=1.27e-10。说明 phantom 在簇内添加了**随机噪声**——事件被混到错的簇里，所以簇内一致性被稀释。修过版后簇内信号实质增强。
- **簇内 centered τ 没变**：orig 中位 0.021 → mask 0.023（+0.000），21/40 增强，p=0.69 NS。Centered τ 已经把 identity bias 减掉了，剩下的纯传播方向信号不受 phantom 影响。这一致——phantom 是 noise 而不是 identity-bias source。
- **bias_fraction 略增**：orig 中位 0.879 → mask 0.922（+0.017），28/40 增强，p=3.17e-4。簇内刻板里**身份偏置占比从 87.9% 涨到 92.2%**。
- **结论**：PR-4 panel d "簇内 86% identity bias" 的发现**被修过版加强**——去 phantom 后簇内总刻板度更高，但增量全部来自 identity 通道排序（不是事件特异性传播方向）。phantom 用 noise 稀释了 raw signal，但没有冲淡 centered signal。

代号补注：legacy MI = `compute_legacy_mi` (bools-masked), within-cluster centered τ = `compute_within_cluster_centered_tau`, bias_fraction = 1 - centered/raw, PR-4 panel d identity-bias finding (历史 30-subject cohort, 86%)。

---

## 2. Cohort 数字表

n = 40 paired subjects.

### 2.1 legacy MI（PR-3 主指标，bools-masked，预期不变）

| | orig | mask |
|---|---|---|
| max \|Δ MI mean\| | — | **0.000017** |
| n significant (p<0.05) | 40 | 40 |

通过 sanity check：bools-masked 的 MI 计算路径确实不受 phantom 影响。

### 2.2 簇内 stereotypy（受 phantom 影响）

| metric | orig median | mask median | Δ median | 方向 | n 增强 | Wilcoxon p |
|---|---:|---:|---:|---|---:|---:|
| `mean_raw_tau` | 0.237 | **0.291** | **+0.054** | mask STRONGER | **39/40** | **1.27e-10** |
| `mean_centered_tau` (bias-removed) | 0.021 | 0.023 | +0.000 | ≈ flat | 21/40 | 0.69 NS |
| `mean_bias_fraction` (= 1 - centered/raw) | 0.879 | **0.922** | **+0.017** | mask higher bias | **28/40** | 3.17e-4 |

### 2.3 解读

- raw τ 提升 + centered τ 不变 ⇒ **phantom 是 noise，不是 identity-bias source**
- bias_fraction 提升 ⇒ **identity ordering 的解释力在 masked 下占比更高**
- 这与 PR-4 panel d 的 "86% identity bias" 主旨同向（panel d cohort 是 30-subject Tier 0，本 cohort 是 40-subject Tier 2，数字不直接可比，但方向一致）
- **没有任何 metric 出现"方向反转"**——所有方向都和 audit 预测一致

---

## 3. Topic 1 主文档 §3.2 数字需要更新

> ### 3.2 Identity bias 不是小问题，在簇内水平更高
>
> | 层 | raw τ | centered τ | bias fraction |
> |---|---|---|---|
> | 整体 | 0.089 | 0.023 | 0.652 |
> | 簇内 | 0.252 | ≈0.03 | **0.86** |

→ Step 5i 应改成：

> | 层 | raw τ | centered τ | bias fraction |
> |---|---|---|---|
> | 整体 | ~~0.089~~ → TBD（5d 后重算） | TBD | TBD |
> | 簇内（mask） | **0.29** | **0.02** | **0.92** |

但本节先不动主文档（等 5d 全跑完一起更）。

---

## 4. 受影响 PR 状态更新（Topic 0 §3.1 表）

| 下游 | 状态前 | 5c 后状态 |
|---|---|---|
| PR-3 per-cluster MI | YES | **重跑完，cohort MI 数字不变；簇内 stereotypy +0.054（增强），bias_fraction +0.017 → PR-4 panel d 同向加强** |

整体方向：**PR-3 / PR-4 panel d 类主指标全部稳健**，没有崩塌也没有反转，反而部分加强。这说明 phantom rank bug 主要污染的是"WHICH event in which cluster"层面，不是"how much stereotypy"层面的数字。

---

## 5. Checkpoint A 通过依据复盘（advisor 2026-05-20）

advisor 提议的"discriminator check": 5b 翻动的 5 个 subject 中 4 个 non-degenerate 全部能追溯到 5a 的 label-level shift：

| sid | direction | jaccard(5a) | cohort rank/35 | 解释 |
|---|---|---:|---:|---|
| 548 | LOST | 0.70 | 17 | mid-shift, marginal swap | 
| 620 | LOST | 0.81 | 30 | low-shift, threshold-borderline swap |
| 635 | LOST | 0.52 | 10 | severe shift, easily explains |
| 253 | GAINED | 0.38 | **6** | extreme shift, de-phantom reveals real structure |
| 916 | GAINED | N/A | — | degenerate (stable_k 2→4), excluded |

⇒ 5b set turnover 是 5a label restructuring 的下游结果，**masking 没有引入 PR-2.5 层面的新偏置**。Checkpoint A **通过**。

---

## 6. 下一步

立即可启动 **Step 5d** (PR-4A/B/C/D on masked labels)。这是 cohort 主结论里描述层的核心：

- **PR-4A** day/night template occupancy timeline + day/night summaries
- **PR-4B** rate-state coupling L1/L2/L3
- **PR-4C** seizure proximity
- **PR-4D** template-rate decomposition

5d 完跑后是 **Checkpoint B**：审 PR-4B/D 方向是否反转（这两条是 Topic 1 §4 主结论描述层的支柱）。

启动命令（按 PR-4A 启动；PR-4B/C/D 后续）：
```
python scripts/run_interictal_propagation.py --pr4a --masked-features
```
