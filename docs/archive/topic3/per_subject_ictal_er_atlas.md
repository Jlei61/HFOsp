# Per-subject Ictal ER 原子图谱（PR-T3-1 Layer A 跑 cohort 用）

> **目的**：随 Layer A producer (`scripts/run_ictal_er_rank.py`) 增量记录每个 subject 的 ER-rank 输出 + clinical SOZ 标注 + 二者一致性诊断。供 cohort 完成后归纳 subject-level epilepsy pattern 分类参考；不作 cohort-level 一票通过/否决依据。
>
> **措辞合同（重要）**：clinical SOZ (`focus_rel == 'i'`) 与 data-driven ER-rank 都是 *proxy*，不互为 ground truth。本文档**只描述**两套结果与差异的统计/解剖性质，**不**写"data-driven 扩展了 / 修正了 / 真正反映了"等隐含价值判断的句式。任何 inference 必须等 cohort-level 多套独立证据交叉之后才能在 main doc 出现。
>
> **producer-health 与 clinical-concordance 是两件独立的事**，必须分开报告：
> - `producer_health`：data-driven 信号自身是否可用（看 n_ok / s_sz / top-K coverage / tied/unreached 比例）。
> - `clinical_concordance`：data-driven r_sz 排序是否与 clinical i-label 一致（focal vs nonfocal Wilcoxon + focal channel 在 top-K 出现率）。
> 一个高 ≠ 另一个高；一个低 ≠ 另一个低。两者各自独立分类、独立汇报。

---

## Tag 词典（cohort 一致使用，不要 subject-level 自由发挥）

### `producer_health`（4 档）

| tag | 条件 |
|---|---|
| `stable` | `n_ok ≥ 3` AND `s_sz ≥ 0.5` AND top-10 channels 中至少 7/10 满足 `cov ≥ max(3, 0.5 × n_ok)` |
| `moderate` | `n_ok ≥ 3` AND `0.3 ≤ s_sz < 0.5` AND top-10 cov 同上 |
| `unstable` | `n_ok ≥ 3` AND `s_sz < 0.3`（不论 cov） |
| `insufficient` | `n_ok < 3`（含全 baseline_invalid / 全 onset_unreached / 全 onset_tied 之类无法形成 r_sz 的状况）|

### `clinical_concordance`（4 档，仅作诊断元数据，不进 producer 决定）

| tag | 条件 |
|---|---|
| `concordant` | focal r_sz 中位数 < nonfocal AND Wilcoxon one-sided `p < 0.1` AND focal channels 在 top-10 中 ≥ 30% |
| `partial` | 上述三条只满足其中两条，或 ≥ 30% top-10 由 focal strip 邻接 channel 占据（同 strip 不同触点）|
| `discordant` | Wilcoxon `p ≥ 0.1` AND focal 在 top-10 占比 < 30% |
| `not_assessable` | r_sz 中 focal channel < 2 或 nonfocal channel < 2，或 producer_health = `insufficient` |

### Subject-level 综合 tag（cohort summary 用，目前先不下结论）

跑完 30 subject 后，按 `(producer_health, clinical_concordance)` 二维分类：

- (stable, concordant)：data-driven 与 clinical 一致且信号稳——最强证据 case
- (stable, discordant)：data-driven 自身稳但与 clinical 不一致——需要解剖 / 多模态独立交叉
- (moderate / unstable, concordant)：clinical 标注落在 ER-rank 前段，但 r_sz 排序在不同 seizure 间不稳——可能 within-subject 多发作类型
- (insufficient, *)：producer 在该 subject 上 unusable，跳过 Layer B
- 其它组合：归 partial 看具体

---

## 当前完成的 subject

- `epilepsiae/548` — sentinel
- `epilepsiae/916` — sentinel

cohort 24 subject 待 §3.4 doc-fix 完成、A.4 cohort run 跑完后逐个回填。

---

## epilepsiae/548

**Sentinel role**：sentinel_A（PR-6A 已锁，9-subset 中 seizure 最多）

**Clinical SOZ (`focus_rel == 'i'`, n=7)**：`HL7, HL8, HL9, HL10, TBLA1, TBLA2, TBLA3`

### Layer A producer 输出

| ER config | λ (capped?) | n_total | n_loaded | n_ok | n_unreached | n_tied | s_sz |
|---|---|---|---|---|---|---|---|
| gamma_ER | 100.0 (capped) | 31 | 31 | 15 | 16 | 0 | **0.159** |
| broad_ER | 100.0 (capped) | 31 | 31 | 19 | 12 | 0 | **0.097** |

**Top-10 earliest channels — gamma_ER**

| rank | ch | r_sz | cov / n_ok | F/N |
|---|---|---|---|---|
| 1 | HL9 | 11.000 | 7/15 | F |
| 2 | TBRC1 | 12.000 | 11/15 | N |
| 3 | HL7 | 12.500 | 15/15 | F |
| 4 | HL6 | 12.500 | 15/15 | N |
| 5 | TBLC6 | 12.500 | 13/15 | N |
| 6 | TBLB4 | 12.500 | 13/15 | N |
| 7 | HL3 | 12.750 | 10/15 | N |
| 8 | TLRB1 | 14.000 | 14/15 | N |
| 9 | TBLC5 | 14.500 | 13/15 | N |
| 10 | HL8 | 14.500 | 11/15 | F |

**Top-10 earliest channels — broad_ER**

| rank | ch | r_sz | cov / n_ok | F/N |
|---|---|---|---|---|
| 1 | GB8 | 4.500 | 3/19 | N |
| 2 | GC6 | 9.750 | 4/19 | N |
| 3 | TBLA3 | 13.000 | 17/19 | F |
| 4 | HL4 | 13.000 | 16/19 | N |
| 5 | HL10 | 13.000 | 14/19 | F |
| 6 | HL9 | 14.000 | 18/19 | F |
| 7 | TLRA4 | 14.000 | 9/19 | N |
| 8 | HL7 | 14.500 | 16/19 | F |
| 9 | HL5 | 15.000 | 19/19 | N |
| 10 | HL3 | 15.000 | 15/19 | N |

**Focal channel ranks（按 r_sz 升序）**

| ER | 排序 |
|---|---|
| gamma_ER | HL9 (11) · HL7 (12.5) · HL8 (14.5) · HL10 (16) · TBLA3 (22) · TBLA2 (25) · TBLA1 (None, cov=0) |
| broad_ER | TBLA3 (13) · HL10 (13) · HL9 (14) · HL7 (14.5) · HL8 (20) · TBLA2 (20) · TBLA1 (39) |

**focal vs nonfocal r_sz**

| ER | focal median (n) | nonfocal median (n) | Wilcoxon one-sided p |
|---|---|---|---|
| gamma_ER | 15.25 (6) | 26.50 (77) | 0.006 |
| broad_ER | 14.50 (7) | 23.75 (77) | 0.012 |

### Tag 分类

| ER | producer_health | clinical_concordance |
|---|---|---|
| gamma_ER | **unstable** (s_sz=0.16 < 0.3；top-10 cov 9/10 满足 ≥ 7.5) | **concordant** (Wilcoxon p=0.006；focal 在 top-10 = 3/10) |
| broad_ER | **unstable** (s_sz=0.10 < 0.3；top-10 cov 8/10 满足 ≥ 9.5，GB8/GC6 cov=3/4 极低) | **concordant** (Wilcoxon p=0.012；focal 在 top-10 = 4/10) |

### Spatial / 解剖描述（中性）

- gamma_ER 中 ER-rank 早触发集中在 **HL strip 中段（HL3, HL6, HL7, HL8, HL9）+ TBLB4 / TBLC5–6 / TBRC1 / TLRB1 等多个 TB-* depth strip 通道**。clinical i 标注的 HL7–10 中 HL7/8/9 落在 top-10，HL10 排第 14，TBLA3 排第 22；TBLA1 在 gamma_ER 中 cov=0（CUSUM 未触发）。
- broad_ER 中 ER-rank 早触发同样集中在 **HL strip + TBLA3**，但出现 GB8 / GC6 两个 cov 很低（3–4/19）的 grid 通道排到前列，提示 broad band 噪声敏感性更高。
- 两套 ER 之间一致性：**HL strip + TBLA3 在 gamma + broad 中都进 top-10**；GB / GC / TBRC / TBLB / TBLC 等"扩散通道"在两套 ER 中各不相同。
- **重要观察**：focal 与非 focal channel 都在 HL strip 上同段范围（HL3-HL10）触发，按 r_sz 排序无法把 focal 与"focal strip 邻接通道"区分开；clinical-concordance 的"concordant"判定主要靠 Wilcoxon median 偏移 + top-10 中 focal 占 30–40%，不代表 ER-rank 能精确选出 clinical SOZ subset。

### Subject-type cohort 候选 tag（待 cohort 完成后归纳）

`(unstable, concordant)`：focal channels 平均落在 ER-rank 前段，但不同 seizure 之间排序差异大（s_sz 低）。可能反映：
1. 548 内有多种 seizure 起始模式，每次起源 channel 略不同（s_sz 自然低）；
2. clinical i 标注的 HL strip 是 broad ictogenic zone，单 seizure 内最早 channel 在 HL3-HL10 之间漂移，median 上仍偏向 i-subset。

---

## epilepsiae/916

**Sentinel role**：sentinel_B（PR-6A 已锁，普通 k=2、reproducibility=strong、不在 9-subset）

**Clinical SOZ (`focus_rel == 'i'`, n=11)**：`AM7, F8, FT10, PB1, PB2, PB3, PB4, PB5, PB6, PB7, T8`

(`l` 区: 0; `e` 区 n=94：含 PB8–15、AM 余下 14 个、TS 全条 n=12、TB 全条 n=15、AH 全条 n=15、PH 全条 n=12、scalp 余下数十)

### Layer A producer 输出

| ER config | λ (capped?) | n_total | n_loaded | n_ok | n_unreached | n_tied | s_sz |
|---|---|---|---|---|---|---|---|
| gamma_ER | 100.0 (capped) | 51 | 49 | 26 | 23 | 0 | **0.830** |
| broad_ER | 100.0 (capped) | 51 | 49 | 2 | 47 | 0 | **0.590** |

**Top-15 earliest channels — gamma_ER**（top-10 没有 focal 通道，扩到 15 才出现 PB1）

| rank | ch | r_sz | cov / n_ok | F/N |
|---|---|---|---|---|
| 1 | PB15 | 2.750 | 26/26 | N |
| 2 | TS11 | 3.000 | 26/26 | N |
| 3 | TS9 | 3.250 | 26/26 | N |
| 4 | TS10 | 3.250 | 26/26 | N |
| 5 | TS12 | 3.250 | 26/26 | N |
| 6 | AM15 | 4.000 | 26/26 | N |
| 7 | PB14 | 4.750 | 26/26 | N |
| 8 | TB15 | 6.500 | 25/26 | N |
| 9 | TB14 | 6.750 | 26/26 | N |
| 10 | PB13 | 9.000 | 26/26 | N |
| 11 | AM3 | 11.000 | 26/26 | N |
| 12 | AM4 | 11.250 | 26/26 | N |
| 13 | TB2 | 14.000 | 26/26 | N |
| 14 | **PB1** | 14.000 | 26/26 | **F** |
| 15 | TB13 | 14.750 | 26/26 | N |

**Top-10 earliest channels — broad_ER**（n_ok=2，r_sz 信号薄弱仅供参考）

| rank | ch | r_sz | cov / n_ok | F/N |
|---|---|---|---|---|
| 1 | TS9 | 1.250 | 2/2 | N |
| 2 | AM15 | 2.000 | 2/2 | N |
| 3 | TS11 | 2.500 | 2/2 | N |
| 4 | TS12 | 2.500 | 2/2 | N |
| 5 | TS10 | 3.500 | 2/2 | N |
| 6 | TB15 | 4.000 | 2/2 | N |
| 7 | PB15 | 6.000 | 2/2 | N |
| 8 | TB14 | 7.000 | 2/2 | N |
| 9 | PB14 | 7.500 | 2/2 | N |
| 10 | PB2 | 10.000 | 1/2 | F |

**Focal channel ranks（按 r_sz 升序）**

| ER | 排序 |
|---|---|
| gamma_ER | PB1 (14) · PB2 (25) · AM7 (29.75) · PB3 (31) · PB4 (38) · PB7 (45) · PB6 (46) · PB5 (51.5) · F8/FT10/T8 不在 r_sz 的 84 通道 alias map 内 |
| broad_ER | PB2 (10) · PB6 (12) · PB1 (12.5) · PB4 (13) · PB7 (14) · PB5 (15) · PB3 (16) · AM7 (None, cov=0) |

**focal vs nonfocal r_sz**

| ER | focal median (n) | nonfocal median (n) | Wilcoxon one-sided p |
|---|---|---|---|
| gamma_ER | 34.50 (8) | 32.00 (69) | 0.583 |
| broad_ER | 13.00 (7) | 26.00 (55) | 0.004 ¹ |

¹ broad_ER 仅 n_ok=2，r_sz 中位数高度受随机 seizure 选择影响；统计显著但样本太少不进 producer-health 决定。

### Tag 分类

| ER | producer_health | clinical_concordance |
|---|---|---|
| gamma_ER | **stable** (n_ok=26；s_sz=0.83 ≥ 0.5；top-10 cov 全部 ≥ 25/26) | **discordant** (Wilcoxon p=0.583；focal 在 top-10 = 0/10) |
| broad_ER | **insufficient** (n_ok=2 < 3；47/49 onset_unreached) | **not_assessable** (受 producer_health = insufficient 限制) |

### Spatial / 解剖描述（中性）

- gamma_ER 中 ER-rank 最早通道集中在 **PB strip 高号深端触点 (PB13–15)** 与 **TS9–12 / AM15 / TB14–15 等 depth strip 高号触点**；clinical i 标注的浅端 PB1–7 在 r_sz 上排位 14–51.5，AM7 在 29.75，scalp-like F8 / FT10 / T8 不在 84 通道 montage 内。两套结果**无重叠**于 top-10。
- 同一物理 PB strip 上，PB13–15（clinical=e）与 PB1–7（clinical=i）r_sz 差距 ~5×，属同 strip 不同触点对 ER-rank 的反应不同；本文档**不**对此差异作"哪段触点是真起源"判断。
- broad_ER 在 916 上 47/49 onset_unreached；说明该 ER 配置在 916 baseline noise 下 z-score 上不来，CUSUM 累不到 λ=100。仅有的 2 个 ok seizure 中 focal channel (PB2/PB6/PB1/PB4/PB7) 反而排前——但 n_ok=2 信号不可靠，**不**作 within-subject "broad 与 gamma 起源不同" 推论。

### Subject-type cohort 候选 tag（待 cohort 完成后归纳）

`(stable, discordant)`：data-driven r_sz 在 916 内部高度可重复（s_sz=0.83；top-10 cov 全满），但与 clinical i 标注几乎不重叠。两条独立可能解释（不在本文档下结论）：

1. clinical i 标注覆盖范围与 ER-rank 早触发区域物理上不重合（即两套 proxy 实测差异确实存在）；
2. 916 内 gamma + broad 两套 ER 触发模式不一致（broad 仅 2 ok，无法直接对照），可能反映 within-subject 多发作类型。

后续无论 broad_ER 是否 producer-health PASS，本 subject 的 gamma_ER label 进入 Layer B 时必须**同时输出 discordant 标签**，下游 PR 不得默认 "data-driven SOZ ≈ clinical SOZ"。

---

## 待办（cohort 跑完后回填）

- [ ] 24 audit_eligible subjects 各填一节（schema 同上）
- [ ] cohort summary：`(producer_health, clinical_concordance)` 二维分布表
- [ ] cohort summary：subject-level type 分类（数量 + 子类典型 case 链接）
- [ ] λ-cap 现象 cohort 分布：是否所有 subject 都撞顶 100，还是 sentinel-only 现象
- [ ] gamma vs broad 两套 ER 的 producer_health 一致性 cohort 表
- [ ] 决定 Layer B 准入策略：(stable, *) ∪ (moderate, concordant) 还是更宽？等 cohort 分布出来再定
