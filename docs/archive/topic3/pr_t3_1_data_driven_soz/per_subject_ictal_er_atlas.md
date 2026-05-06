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

cohort 16 epilepsiae subject 待 A.4 cohort run 跑完后逐个回填（plan §6.1 v2.2 doc-fix 后 cohort 限定 epilepsiae，详见下文 cohort scope 注）。

---

## v2.2 cohort scope 注（2026-05-04）

**当前 v2.2 cohort = epilepsiae only（15 audit_eligible + 916 sentinel-only = 16 subjects）。**

**为什么**：`extract_seizure_window` 在 `src/ictal_onset_extraction.py:273` 显式 raise 非 epilepsiae。这是 PR-6A 阶段的既有约束。Layer A 的 SeizureWindow 加载链 (`extract_seizure_window` → ER 计算 → CUSUM → r_sz) 完全建立在 PR-6A 接口上，因此 v2.2 cohort 自然继承这个限制。

**audit_eligible 24 subject 中被排除的 9 个 yuquan subject**：
gaolan, huanghanwen, litengsheng, pengzihang, sunyuanxin, xuxinyi, zhangjinhan, zhangkexuan, zhaojinrui

**追跑路径（独立后续 PR）**：实现 yuquan 版本的 `extract_seizure_window`（加载 yuquan EDF + clinical onset 切窗 + baseline 解析 + SOZ channel 通过 `yuquan_soz_core_channels.json` 而非 `focus_rel`）。落地后用相同 `_run_subject_all_ers` + `_build_v2_2_tags` 接口跑完 9 yuquan subject，per-subject JSON 落到同一 `per_subject/` 目录，cohort summary 自动 pickup。

**对当前 cohort 结论的影响**：
- 16 epilepsiae cohort 已经能给出 (producer_health × clinical_concordance) 二维分布的代表性样本；不会因缺 yuquan 而无法判断 v2.2 框架是否 work
- Layer B 初版标签**只覆盖 epilepsiae**；下游 PR (Layer B → topic3 cross-validation) 需要 propagate "yuquan label pending" 标志，避免误以为 yuquan 没有 ER-driven SOZ
- atlas cohort summary 末尾会单列 yuquan 状态行，明确写"待 yuquan extract_seizure_window 后追跑"

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

## epilepsiae/583

**Cohort role**：γ_a primary 双 ER（model case — 唯一双 ER 都 stable/moderate + concordant 的 subject）

**Clinical SOZ (`focus_rel == 'i'`, n=15)**：
- HL strip 浅端：HL1, HL2, HL3, HL5, HL6（HL4 是 e；HL7-10 都是 e）
- TBA strip 深端：TBA1, TBA2（TBA3, TBA4 是 e）
- TBB strip 顶端：TBB1（TBB2-4 是 e）
- TLA strip 深端：TLA1, TLA2, TLA3, TLA4（TLA5-6 是 e）
- TLB strip：TLB1, TLB2, TLB7（TLB3-6, TLB8-12 是 e）

(`l` = 0；`e` n=46，含 5 strip 的非 i 触点 + 全部 scalp + TBC strip 全部)

### Layer A producer 输出

| ER | λ (capped?) | n_total | n_loaded | n_ok | n_ur | n_tied | s_sz |
|---|---|---|---|---|---|---|---|
| gamma_ER | 100.0 (capped) | 23 | 23 | 22 | 1 | 0 | **0.448** |
| broad_ER | 100.0 (capped) | 23 | 23 | 21 | 2 | 0 | **0.625** |

**Top-15 earliest channels — gamma_ER**（topk_total=13 due to boundary tie at r_sz=11.5）

| rank | ch | r_sz | cov / n_ok | F/N |
|---|---|---|---|---|
| 1 | TLB1 | 1.75 | 22/22 | **F** |
| 2 | TBA2 | 6.00 | 21/22 | **F** |
| 3 | TLB7 | 6.50 | 22/22 | **F** |
| 4 | TBB3 | 6.50 | 21/22 | N |
| 5 | TLA2 | 7.25 | 22/22 | **F** |
| 6 | TLA3 | 7.25 | 22/22 | **F** |
| 7 | TBA1 | 7.50 | 21/22 | **F** |
| 8 | HL1 | 8.75 | 20/22 | **F** |
| 9 | HL4 | 11.00 | 21/22 | N |
| 10–13 | HL5 / HL9 / TBA3 / HL8 | 11.50 | 14/22, 17/22, 22/22, 17/22 | F / N / N / N |
| 14 | HL2 | 12.50 | 20/22 | **F** |
| 15 | TLB2 | 14.00 | 21/22 | **F** |

(rank 10-13 是 r_sz=11.5 4-way tie，全部纳入 → topk_total=13；HL5 是这 4 个中唯一 focal)

**Top-10 earliest channels — broad_ER**（无 boundary tie，topk_total=10）

| rank | ch | r_sz | cov / n_ok | F/N |
|---|---|---|---|---|
| 1 | TBA2 | 1.50 | 21/21 | **F** |
| 2 | TBB3 | 3.50 | 21/21 | N |
| 3 | TLA3 | 4.00 | 21/21 | **F** |
| 4 | TLB7 | 4.00 | 21/21 | **F** |
| 5 | TBA1 | 4.00 | 21/21 | **F** |
| 6 | TLB1 | 5.25 | 20/21 | **F** |
| 7 | TBA3 | 6.00 | 21/21 | N |
| 8 | HL1 | 6.00 | 19/21 | **F** |
| 9 | HL2 | 9.75 | 20/21 | **F** |
| 10 | TBB4 | 11.50 | 21/21 | N |

**Focal channel ranks (broad_ER, 完整序)**：

TBA2 (1) · TLA3 (3) · TLB7 (4) · TBA1 (5) · TLB1 (6) · HL1 (8) · HL2 (9) · TLA2 (11) · TLB2 (12) · HL5 (13) · HL3 (19) · HL6 (23) · TLA4 (25) · TBB1 (28, cov 9/21) · TLA1 (29)

### Tag 分类

| ER | producer_health | clinical_concordance |
|---|---|---|
| gamma_ER | **moderate** (n_ok=22, s_sz=0.45 ∈ [0.3, 0.5), top-10 cov 全 ≥ 11) | **concordant** (MWU p=0.0016, focal_in_topk=8/13=61.5%) |
| broad_ER | **stable** (n_ok=21, s_sz=0.625 ≥ 0.5, top-10 cov 全 ≥ 19) | **concordant** (MWU p=0.0013, focal_in_topk=7/10=70%) |

### Spatial / 解剖描述（中性）

- **5 个 focal strip 全部在 ER-rank top-15 出现**：TLB (TLB1=#1, TLB7=#3, TLB2=#15)、TBA (TBA1, TBA2)、TLA (TLA2, TLA3)、HL (HL1, HL5, HL2)、TBB (TBB1 在 broad rank=28, 但 cov=9/21 较低)。
- gamma 与 broad 双 ER **top 通道高度重合**：TLB1, TBA2, TLB7, TLA2/3, TBA1, HL1, HL2 都在双 ER top-10/13 内。
- "Top-K 中的非 focal 通道" 全为 focal-strip 邻接触点：TBB3 (focal=TBB1), HL4/8/9 (focal HL1-3,5-6), TBA3 (focal TBA1-2), TBB4 (focal TBB1)。换句话说 ER-rank 的 top set ≈ clinical i-set 的物理 strip 邻域，没有"远端 outlier"。
- 焦点 channel 中 **TBB1, TLA1, HL6 排序较晚**（rank > 20-30 在 broad，cov 较低）。这些可能是 within-subject seizure 间起源不一致的少数 channel。

### Label assessment（下游可用性）

- **强证据 model case**：双 ER 都 (stable / moderate) + concordant，s_sz、p、focal fraction 全过线。
- 两套 SOZ proxy 在 583 上**广泛同意**：data-driven top-K 几乎完全覆盖 clinical i 的 5 个深部 strip 主体。
- top-K 选 k = |clinical_matched| = 15 → broad 会选出 top-15 全部即 7 focal + 8 非 focal-strip-adjacent。下游若用 broad primary，对照 clinical 时大致一致。
- **缓冲**：不是 1:1 重合 — 焦点 TBB1/TLA1/HL6 排得晚，data-driven label 会 miss 它们；同时会 include strip 邻接的 TBB3/HL4/8/9。下游不应把"data-driven 多出来的 channel"或"data-driven 漏掉的 focal"当 ground truth 偏差，只能当两套 proxy 的边界差异。

### Subject-type cohort 候选 tag

`(stable, concordant)` model case：data-driven r_sz 与 clinical i-label 在主体 region 上一致，且 producer 跨 seizure 高度可重复。在 cohort 里是少数（双 ER 同时 stable/moderate + concordant 唯一 case）。

---

## epilepsiae/139

**Cohort role**：γ_a primary（broad moderate+concordant）+ γ_a drop（gamma unstable+discordant）

**Clinical SOZ (`focus_rel == 'i'`, n=8)**：F8, FT10, HL2, HL3, HL4, HL5, T8, TBA1
- HL strip 中段：HL2-5（HL1, HL6-10 是 e）
- TBA strip 顶端：TBA1（TBA2-4 是 e）
- 4 个 scalp 电极：F8, FT10, T8（这些不在 bipolar montage 的 r_sz 里 → 永远 None）

`l` n=28（含右半球 HRA1-3, HRB1, scalp 多个）；`e` n=21

### Layer A producer 输出

| ER | λ (capped?) | n_total | n_loaded | n_ok | n_ur | n_tied | s_sz |
|---|---|---|---|---|---|---|---|
| gamma_ER | 100.0 (capped) | 6 | 6 | 4 | 2 | 0 | **0.079** |
| broad_ER | 100.0 (capped) | 6 | 6 | 4 | 2 | 0 | **0.346** |

**Top-10 earliest channels — broad_ER**（教科书级 focal cluster）

| rank | ch | r_sz | cov / n_ok | F/N |
|---|---|---|---|---|
| 1 | HL3 | 1.25 | 4/4 | **F** |
| 2 | HL2 | 1.75 | 4/4 | **F** |
| 3 | TBA1 | 5.50 | 4/4 | **F** |
| 4 | TBB2 | 6.25 | 4/4 | N |
| 5 | HRA1 | 6.25 | 4/4 | N |
| 6 | HL4 | 6.50 | 4/4 | **F** |
| 7 | HL5 | 7.25 | 4/4 | **F** |
| 8 | HL7 | 8.50 | 4/4 | N |
| 9 | TBA2 | 9.25 | 4/4 | N |
| 10 | HL6 | 9.25 | 4/4 | N |

**Top-10 earliest channels — gamma_ER**（与 broad 显著不一致，多区域散布）

| rank | ch | r_sz | cov / n_ok | F/N |
|---|---|---|---|---|
| 1 | HRA2 | 2.00 | 3/4 | N (右半球) |
| 2 | TBB4 | 3.00 | 3/4 | N |
| 3 | HRB1 | 4.50 | 2/4 | N (右半球) |
| 4 | HL7 | 5.00 | 3/4 | N |
| 5 | HL6 | 6.50 | 3/4 | N |
| 6 | HL10 | 6.75 | 2/4 | N |
| 7 | HL9 | 8.25 | 2/4 | N |
| 8 | HL2 | 9.00 | 3/4 | **F** |
| 9 | HRA3 | 9.00 | 3/4 | N (右半球) |
| 10 | HL3 | 9.00 | 3/4 | **F** |

### Tag 分类

| ER | producer_health | clinical_concordance |
|---|---|---|
| gamma_ER | **unstable** (n_ok=4 ≥ 3 但 s_sz=0.08 < 0.3) | **discordant** (focal_in_topk=2/10=20% < 30%, MWU p=0.59) |
| broad_ER | **moderate** (n_ok=4, s_sz=0.35 ∈ [0.3, 0.5)) | **concordant** (focal_in_topk=5/10=50%, MWU p=0.0003) |

### Spatial / 解剖描述（中性）

- **broad_ER 给出非常干净的 HL strip + TBA1 picture**：clinical i 的 5 个 in-montage focal channel (HL2-5 + TBA1) 全部进 top-7；剩余 3 个 scalp 焦点 (F8/FT10/T8) 不在 84 通道 montage 内，r_sz 自然为 None。Top-10 里的非 focal 通道 (TBB2, HRA1, HL7, TBA2, HL6) 又都是同 strip 邻接 (TBA, HL) 或左半球 HRA strip 顶端。
- **gamma_ER 与 broad_ER 截然不同**：gamma top-10 早触发集中在右半球 HRA/HRB strip + HL 上端 (HL6-10) + TBB4。Clinical 焦点 HL2/HL3 被推到 rank 8/10。这种 ER 间 disagreement 可能反映 within-subject 多种 seizure type（不同发作起源不同）；n_ok=4 太少不足以稳定 median 排序，是 gamma s_sz=0.08（基本随机）的直接表现。
- **n_ok=4 是边界 case**：刚刚过 producer_health 门（n_ok ≥ 3）。下游使用 broad label 时需注意"4 seizures 平均"在统计上 fragile。

### Label assessment（下游可用性）

- **broad primary 可用且强**：HL strip 中段 + TBA1 与 clinical i 几乎完美对应；非 focal 进 top-10 的全是 strip 邻接，无远端 outlier。
- **gamma drop 合理**：producer 不稳 + cc 不显著 + 多区域散布；信号杂乱不该作 label。
- **缓冲**：4 scalp focal 永远不在 r_sz 中，broad label 不会包含它们。如果下游 PR 需要 scalp electrode 评估，139 不能给。
- 强烈推荐**只用 broad primary**，并标注 cc=concordant + n_ok=4 small-sample caveat。

### Subject-type cohort 候选 tag

ER-asymmetric case：broad band 给 textbook focal cluster，gamma band 给 multi-region。可能的解释：(1) within-subject multiple seizure types，broad 平滑后落在主导 type 上；(2) gamma band 高频特异性强但需要更多 seizure 累积才能稳定排序。

---

## epilepsiae/253

**Cohort role**：γ_a primary（broad moderate+discordant）+ γ_a drop（gamma unstable+not_assessable）

**Clinical SOZ (`focus_rel == 'i'`, n=3)**：HRB2, HRB3, HRC2（**全在右半球**：HR strip cluster）
- HRB strip 中段：HRB2, HRB3
- HRC strip 顶端：HRC2

`l` n=42（很多对侧 + scalp + 其它 strip）；`e` n=6（HRA1-3, HRB1, HRC1, HRC3 — 与 focal 同 strip 上下邻接）

### Layer A producer 输出

| ER | λ (capped?) | n_total | n_loaded | n_ok | n_ur | n_tied | s_sz |
|---|---|---|---|---|---|---|---|
| gamma_ER | 100.0 (capped) | 7 | 5 | 5 | 0 | 0 | **-0.148** |
| broad_ER | 100.0 (capped) | 7 | 5 | 5 | 0 | 0 | **0.384** |

**Top-15 earliest channels — gamma_ER**（HL/HR 全 strip 大杂烩，焦点全 None）

| rank | ch | r_sz | cov / n_ok | F/N |
|---|---|---|---|---|
| 1 | HLC4 | 2.50 | 1/5 | N |
| 2 | HRC5 | 3.00 | 5/5 | N |
| 3 | HLB3 | 3.75 | 2/5 | N |
| 4 | HRA5 | 4.50 | 5/5 | N |
| 5 | HLB1 | 4.50 | 1/5 | N |
| 6-10 | HLC3 / HRC3 / HRB5 / HLC5 / HRA4 | 5.00–5.50 | 3-5/5 | N |
| 11+ | ... | | | N |

**3 个 focal 通道 (HRB2, HRB3, HRC2) 在 gamma 中**: 全部 r_sz=None（CUSUM 在 4-20Hz 基线 + gamma 60-100Hz 检测下 5 个 seizure 都没触发）

**Top-10 earliest channels — broad_ER**

| rank | ch | r_sz | cov / n_ok | F/N |
|---|---|---|---|---|
| 1 | HRA2 | 2.00 | 5/5 | N |
| 2 | HLC5 | 3.00 | 3/5 | N |
| 3 | HLC4 | 3.50 | 3/5 | N |
| 4 | HRC1 | 3.50 | 5/5 | N |
| 5 | HLB5 | 4.00 | 2/5 | N |
| 6 | HRB1 | 4.50 | 5/5 | N |
| 7 | **HRB3** | 7.00 | **1/5** | **F** |
| 8 | HRC4 | 7.00 | 4/5 | N |
| 9 | HRA3 | 7.00 | 5/5 | N |
| 10 | HRA1 | 7.00 | 5/5 | N |

**Focal channel ranks (broad_ER)**：HRB3 (rank 7, **cov 1/5 单 seizure 触发**) · HRB2 (rank 21, cov 5/5) · HRC2 (rank 28, cov 2/5)

### Tag 分类

| ER | producer_health | clinical_concordance |
|---|---|---|
| gamma_ER | **unstable** (s_sz=-0.15 < 0 表示跨 seizure 排序基本随机) | **not_assessable** (3 focal 全 None r_sz → n_focal_with_finite_rsz=0) |
| broad_ER | **moderate** (s_sz=0.38) | **discordant** (focal_in_topk=1/10, MWU p=0.79；HRB3 仅靠 cov=1/5 进 top) |

### Spatial / 解剖描述（中性）

- **clinical i 和 ER-rank 主体早触发区域 不一致**：clinical 标 HRB2/HRB3/HRC2（HR strip 中段），broad ER-rank 早触发集中在 **HRA strip 顶端 (HRA1-3)** + **HLC strip 顶端 (HLC4-5)** + **HRC1, HRB1, HLB5** —— 与 clinical 焦点是同区域 strip 系列但**不同触点**（深 vs 浅 / left HL vs right HR 也有混合）。
- HRB3 在 broad 进入 top-10 但**只在 1/5 seizure 中触发 CUSUM**——是低 coverage spurious early-firer 的典型 case，不应被 label 准入门当强证据。
- HRB2 (cov 5/5 always fires) 排到 rank 21；HRC2 (cov 2/5) 排 28。

### Label assessment（下游可用性）

- **broad primary 可用但需重点带 cc=discordant tag**：data-driven SOZ 落在 HRA + HLC strip 顶端，与 clinical HRB/HRC 不一致。是真实的 proxy disagreement case，类似 916 的 (stable+discordant)，但 producer 弱一档 (moderate vs stable)。
- 下游 PR 用 253 broad label 进 cross-validation 时**绝对不能默认** "data-driven SOZ ≈ clinical SOZ"。
- gamma drop 合理：3 个焦点全部 None r_sz，无法形成 cc 评估，且 s_sz=负 (基本随机)。
- 缓冲：HRB3 在 top-10 但 cov=1/5，建议 Layer B label builder 在 top-K 选择时**用 r_sz_valid_count 过滤** (e.g., 排除 cov < 0.5×n_ok 的 channel)。

### Subject-type cohort 候选 tag

`(moderate, discordant)` proxy-disagreement case：与 916 (stable, discordant) 同类但 producer 弱。data-driven proxy 与 clinical proxy 给出不同的 SOZ 区域候选；不在本文档下"哪个对"判断，但下游 PR 必须双 label 同时报。

---

## epilepsiae/1084

**Cohort role**：γ_a primary（broad moderate+not_assessable）+ γ_a drop（gamma unstable+not_assessable）

**Clinical SOZ (`focus_rel == 'i'`, n=0)**：**临床 SOZ 完全空** — focus_rel 中 i+l+e 全部 0 通道。
- 这意味着该 subject 在 SQL `electrode.focus_rel` 字段无任何标注（不是 i 全空但 l/e 有值，是整体无标注）。
- clinical_concordance 在 1084 上**永远** not_assessable（无对照集）。
- 仅 producer_health 维度可评估。

### Layer A producer 输出

| ER | λ (capped?) | n_total | n_loaded | n_ok | n_ur | n_tied | s_sz |
|---|---|---|---|---|---|---|---|
| gamma_ER | 100.0 (capped) | 94 | 94 | 22 | 72 | 0 | **0.381** |
| broad_ER | 100.0 (capped) | 94 | 94 | 36 | 58 | 0 | **0.487** |

**Top-15 earliest channels — broad_ER**（producer 信号最强的 cohort 之一）

| rank | ch | r_sz | cov / n_ok | strip prefix |
|---|---|---|---|---|
| 1 | GG8 | 2.00 | 36/36 | G (grid) |
| 2 | GF8 | 5.50 | 35/36 | G |
| 3 | GB2 | 7.50 | 28/36 | G |
| 4 | GC3 | 8.50 | 25/36 | G |
| 5 | GG4 | 9.50 | 29/36 | G |
| 6 | GH8 | 10.75 | 24/36 | G |
| 7 | GF7 | 11.00 | 31/36 | G |
| 8 | GA4 | 11.00 | 24/36 | G |
| 9 | GD1 | 11.00 | 27/36 | G |
| 10 | IHA4 | 12.00 | 30/36 | IHA |
| 11–15 | GE6 / GG2 / GH4 / GG3 / GE8 | 12.5–14.0 | 21-32/36 | G |

**Top-15 earliest channels — gamma_ER**

| rank | ch | r_sz | cov / n_ok | strip prefix |
|---|---|---|---|---|
| 1 | GF8 | 2.75 | 14/22 | G |
| 2 | GE8 | 4.00 | 15/22 | G |
| 3 | GG8 | 4.50 | 13/22 | G |
| 4 | GC3 | 4.50 | 7/22 | G |
| 5 | GA3 | 6.00 | 7/22 | G |
| 6 | GE2 | 6.25 | 8/22 | G |
| 7-15 | GG2/GF7/GH8/GG7/GG4/GG5/GH4/GH5/GF5 | 8.5–9.0 | 3-13/22 | G |

### Tag 分类

| ER | producer_health | clinical_concordance |
|---|---|---|
| gamma_ER | **unstable** (s_sz=0.38 但 top-10 cov 多个 < 0.5×22=11 → 实际 fraction-based check 借降一档) | **not_assessable** (n_focal=0) |
| broad_ER | **moderate** (s_sz=0.49 ∈ [0.3, 0.5), top-10 cov 全 ≥ 24) | **not_assessable** (n_focal=0) |

### Spatial / 解剖描述（中性）

- **几乎全部 G-strip channels（grid electrode）**：GG8 顶头早触发（broad cov 36/36 = 100% seizure 都触发它），GF8 次之 (35/36)。GG row（row 8 = top）大量出现：GG8, GF8, GH8, GE8 → 顶端 column 一大片。
- 还有 GB2, GC3, GG4 等深处的 G channel；非 G 的早触发只有 IHA4（IHA = inferior hippocampal anterior？）在 broad rank 10。
- broad 的 top channels 几乎全部满 cov（≥24/36），说明这些 channel 在大多数 seizure 上都触发——是高度可重复的 spatial cluster。
- gamma 的 top 也都在 G strip 但 cov 偏低（多个 < 11 = 0.5×22），暗示 gamma 在某些 seizure 上起源 channel 不一致。

### Label assessment（下游可用性）

- **broad primary 信号强**：cohort 中 producer s_sz 排第二（0.49，仅次于 583 的 0.62）；top-K cov 全满；空间形态高度一致 (G-strip cluster)。
- **致命限制**：**无 clinical 对照**，concordance 维度永远 not_assessable。无法 cohort-level 验证 data-driven SOZ 的"对错"；下游只能当**纯 data-driven SOZ candidate** 用，不能 cross-validate clinical。
- 适合用作 "data-driven SOZ producer 在没有 clinical SOZ 标注的 subject 上仍能产出可重复 spatial cluster" 的 demonstration case。
- gamma drop 合理：cov 偏低 + 与 broad disagree。

### Subject-type cohort 候选 tag

`(moderate, not_assessable)` no-clinical-reference case：producer 内部稳，但 clinical 标注空让二维分类只剩 producer_health 一维。下游 PR 处理这类 subject 必须 fallback 到 "data-driven SOZ candidate, no clinical comparison available" 的语义。

---

## epilepsiae/958 — γ_a sensitivity 双 ER

**Clinical SOZ (n=21)**：GD3-4, GD8, GE3-4, GE8, GF8, GG6, GH6, OPL3-6, TBB2, TBC4-5, TBD4-6 — 跨 G strip 中段 + OPL + TBB/TBC/TBD strip。

| ER | tags | n_ok | s_sz | top-10 focal in | MWU p | 注 |
|---|---|---|---|---|---|---|
| gamma | unstable+concordant (sensitivity) | 14 | 0.04 | 3/10 (TBC4 #5, GE4 #6, TBC5 #7) | 0.078 | s_sz≈0 表示跨 seizure 排序基本随机；但 median 仍把 3 个 focal 推到 top-10 |
| broad | unstable+partial (drop) | 14 | 0.22 | 2/10 (TBD4 #1, OPL3 #3) | 0.012 | partial: p ok 但 frac < 30% |

**Spatial 描述**：958 是大 montage 多 strip subject，clinical SOZ 跨 G strip 中段 + OPL + TBB/C/D。gamma 的 top-10 落在 G/TBC region 但与 broad 的 top-10 (TBD/OPL/G) 几乎不重叠，提示 ER-band sensitivity 不同。

**Label assessment**：gamma sensitivity 可用但 s_sz≈0 是红旗——意味着 r_sz 是单 seizure noise 的"幸运 median"，下游用要带 strong caveat。建议下游 PR 对 958 gamma sensitivity label 加额外 stability override（例如要求 cov ≥ 0.5×n_ok 才进 top-K）。broad partial 不进 Layer B (drop)。

---

## epilepsiae/590 — γ_a sensitivity broad 单 ER

**Clinical SOZ (n=17)**：BLA1-3, BLB1-3, BRA1-3, GC1-2, GD1, HR1-4, GD2 — 跨 BLA + BLB + BRA + GC + GD + HR strip 多触点。

| ER | tags | n_ok | s_sz | top-10 focal in | MWU p | 注 |
|---|---|---|---|---|---|---|
| gamma | unstable+partial (drop) | 7 | 0.15 | 2/10 (GD1 #3, HR1 #4) | 0.075 | 边界 partial; 7 ok seizure 单薄 |
| broad | unstable+concordant (sensitivity) | 9 | 0.17 | 3/10 (BLB1 #4, GC2 #9, GD1 #10) | 0.092 | borderline concordant |

**Spatial 描述**：broad top-10 含 GC1-2/GD1 (focal G strip) + BLB1 + 多个 r_sz=1 cov 的孤立 channel (BRC5, HR7, TRC5)。broad 把焦点 G strip 顶端揉进 top-10，但前 3 名都是 cov=1 的"单 seizure 早起" channel。

**Label assessment**：broad sensitivity 标签**强烈推荐 cov 过滤**——top 3 BRC5/HR7/TRC5 都是 cov=1/9 的 single-seizure outlier，会污染 top-K。Layer B 应在 entry 加 `topk_min_cov_threshold=2` 之类的保护，否则 590 的 label 几乎都是噪声。

---

## epilepsiae/635 — γ_a sensitivity broad 单 ER（broad 强 concordance）

**Clinical SOZ (n=22)**：HL strip 1-14 (除 HL11-12 是 e) + 其它 strip 余 8 通道。HL strip 大段 focal。

| ER | tags | n_ok | s_sz | top-10 focal in | MWU p | 注 |
|---|---|---|---|---|---|---|
| gamma | unstable+discordant (drop) | 10 | 0.03 | 2/10 (TBA2 #3, HL1 #8) | 0.554 | 多孤立 cov=1-3 channel 进 top |
| broad | unstable+concordant (sensitivity) | 13 | 0.06 | **7/10** (TBA2/TBA1/HL1/HRA2/HL2/HL7/HL3) | **8.3e-6** | 极强 concordance 但 s_sz 低 |

**Spatial 描述**：broad top-10 的 7/10 都是 focal，剩 3 个 (HRA5, HRA3, TLA1) 也是 focal-strip 邻接 (HRA, TLA)。MWU p ~ 1e-5 的强统计显著性。但 s_sz=0.06 表示**跨 seizure 不同次的 median 比较稳，但任两 seizure 排序基本独立**。这是个矛盾的 case：cohort-level focal 偏向强，但 within-cohort 没有 stable 排序。

**Label assessment**：broad sensitivity label 信号强但 producer 不稳。建议下游 PR 把 635 broad 当 "high-concordance but low-stability label"，可作 "看 cohort-level data-driven SOZ 是否聚焦在 clinical region"的统计证据，不适合作 "single-subject case study" 的逐通道判定。

---

## epilepsiae/922 — γ_a sensitivity broad 单 ER

**Clinical SOZ (n=19)**：GA2-5, GB1-4, GC1-4, GD1-2 + 余 — G strip 顶端 (rows A-D, columns 1-5) 矩形 cluster。

| ER | tags | n_ok | s_sz | top-10 focal in | MWU p | 注 |
|---|---|---|---|---|---|---|
| gamma | unstable+discordant (drop) | 24 | 0.16 | 2/10 (GA5 #2, GA4 #3) | 0.976 | 排序整体偏向非 focal grid |
| broad | unstable+concordant (sensitivity) | 25 | 0.13 | **6/10** (GC1, GA5, GC2, GA4, GC3, GD1) | 0.0010 | 干净 G-strip cluster |

**Spatial 描述**：broad top-10 的 6/10 都在 GA + GC + GD（与 clinical i-cluster 完全重合的 G strip 行）。non-focal 3/10 在 top 是 GH8, GG8, GH6/7 — G strip 别处的 grid contact，不是远端 outlier。

**Label assessment**：与 635 类似 (高 concordance + 低 s_sz)，但 922 cov 更扎实 (top-10 多 ≥ 18/25)，更可信。下游使用 broad sensitivity 时风险低。

---

## Cohort summary (A.4 v2.2 — 16 epilepsiae subjects, 完成 2026-05-06)

**Run**: 11.3h wall (2026-05-05 18:32 → 2026-05-06 05:51), 16/16 epilepsiae subjects (15 audit_eligible + sentinel 916), 9 yuquan 已 excluded（见上文 cohort scope 注）。

**Run config**: λ_max=100 (撞顶判 stable), bias=0.5, fpr=1/h, hop=0.1s, detection [-5s,+30s], baseline [-300s, min(0,eeg_onset)-60s]。helper v2.2.1 fix (2026-05-06): top-K 在 r_sz tie 时按 channel 名 alphabetical 二级排序，cohort vs sentinel 跨运行确定性保证。

### 全 cohort per-subject 一览

| subject | n_sz | n_focal | g_n_ok | g_s_sz | g_ph | g_cc | g_p | g_ftk | b_n_ok | b_s_sz | b_ph | b_cc | b_p | b_ftk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1073 | 20 | 5 | 9 | 0.17 | unstable | discordant | 0.963 | 0 | 4 | 0.31 | unstable | discordant | 0.915 | 0 |
| 1077 | 9 | 8 | 3 | 0.02 | unstable | discordant | 0.263 | 1 | 2 | -0.34 | insufficient | not_assessable | - | - |
| 1084 | 94 | **0** | 22 | 0.38 | unstable | not_assessable | - | - | 36 | 0.49 | moderate | not_assessable | - | - |
| 1096 | 9 | 8 | 7 | 0.45 | unstable | discordant | 0.995 | 0 | 8 | 0.19 | unstable | discordant | 0.911 | 1 |
| 1146 | 26 | **0** | 5 | -0.04 | unstable | not_assessable | - | - | 7 | 0.12 | unstable | not_assessable | - | - |
| 1150 | 9 | 6 | 5 | -0.07 | unstable | discordant | 0.424 | 1 | 6 | -0.09 | unstable | partial | 0.051 | 2 |
| 139 | 6 | 8 | 4 | 0.08 | unstable | discordant | 0.588 | 2 | 4 | 0.35 | **moderate** | **concordant** | <0.001 | 5 |
| 253 | 7 | 3 | 5 | -0.15 | unstable | not_assessable | - | - | 5 | 0.38 | **moderate** | discordant | 0.787 | 1 |
| 442 | 22 | **0** | 16 | 0.37 | unstable | not_assessable | - | - | 16 | 0.45 | unstable | not_assessable | - | - |
| 548 | 31 | 7 | 15 | 0.16 | unstable | partial† | 0.006 | 3/11 | 19 | 0.10 | unstable | concordant | 0.012 | 4/12 |
| 583 | 23 | 15 | 22 | 0.45 | **moderate** | **concordant** | 0.002 | 8 | 21 | 0.62 | **stable** | **concordant** | 0.001 | 7 |
| 590 | 13 | 17 | 7 | 0.15 | unstable | partial | 0.075 | 2 | 9 | 0.17 | unstable | concordant | 0.092 | 3 |
| 635 | 20 | 22 | 10 | 0.03 | unstable | discordant | 0.554 | 2 | 13 | 0.06 | unstable | concordant | <0.001 | 7 |
| 916 | 51 | 11 | 26 | 0.83 | **stable** | discordant | 0.583 | 0 | 2 | 0.59 | insufficient | not_assessable | - | - |
| 922 | 30 | 19 | 24 | 0.16 | unstable | discordant | 0.976 | 2 | 25 | 0.13 | unstable | concordant | 0.001 | 6 |
| 958 | 16 | 21 | 14 | 0.04 | unstable | concordant | 0.078 | 3 | 14 | 0.22 | unstable | partial | 0.012 | 2 |

`g_/b_` = gamma_ER / broad_ER；`n_ok` = 进 r_sz 的 seizure 数；`s_sz` = 跨 seizure 排序中位 Spearman；`ph` = producer_health；`cc` = clinical_concordance；`p` = focal vs nonfocal **Mann-Whitney U** one-sided（注：rank-sum 家族，以前 plan/atlas 写 "Wilcoxon" 是 loose 用语）；`ftk` = focal_in_topk_count / topk_total（v2.2.2 起为 tie-inclusive，tie 边界扩展时 topk_total 可 > 10）。**0 标灰**的 subject 临床 SOZ 全为 0（focus_rel 中 i+l+e 全空），cc 自动 not_assessable。

†548 gamma 在 v2.2.1 alphabetical-tie-break 下被错标 concordant；v2.2.2 改用 tie-inclusive 后回归 partial（topk_total=11，focal=3，3/11≈27.3% < 30%，但 MWU p=0.006<0.1 仍满足）。这正是 reviewer 指出的 "alphabetical 让 channel 名字决定科学分类" 的具体表现。

### `(producer_health × clinical_concordance)` 二维分布

**gamma_ER（16 cells, v2.2.2 tie-inclusive）**：

| ph \ cc | concordant | partial | discordant | not_assessable | 行总 |
|---|---|---|---|---|---|
| stable | 0 | 0 | **1** (916) | 0 | 1 |
| moderate | **1** (583) | 0 | 0 | 0 | 1 |
| unstable | 1 (958) | 2 (548†, 590) | 7 | 4 | 14 |
| insufficient | 0 | 0 | 0 | 0 | 0 |
| **列总** | **2** | **2** | **8** | **4** | **16** |

**broad_ER（16 cells）**：

| ph \ cc | concordant | partial | discordant | not_assessable | 行总 |
|---|---|---|---|---|---|
| stable | **1** (583) | 0 | 0 | 0 | 1 |
| moderate | **1** (139) | 0 | 1 (253) | 1 (1084) | 3 |
| unstable | 4 (548, 590, 635, 922) | 2 (1150, 958) | 2 | 2 | 10 |
| insufficient | 0 | 0 | 0 | 2 (1077, 916) | 2 |
| **列总** | **6** | **2** | **3** | **5** | **16** |

### Cohort 关键观察

1. **λ-cap 普遍性**：32/32 cells 撞顶 λ=100。整 cohort baseline ER 噪声超过 1/h FPR 预算 at λ=100。**不要把 λ 数值当 producer 区分维度**——λ_max 是上限，不是峰值。
2. **broad_ER 整体 producer health 优于 gamma_ER**：
   - broad: 1 stable + 3 moderate + 10 unstable + 2 insufficient
   - gamma: 1 stable + 1 moderate + 14 unstable + 0 insufficient
   - broad 跨 seizure 排序更可重复（更多 spectral mass → 更平滑 z-score）。代价是高频特异性下降。
3. **broad_ER concordance 也比 gamma 多**：6 vs 3 concordant cells。可能反映 broad band 包含的低频 ictal recruitment 信号在空间上更紧密对齐 clinical SOZ；或是 gamma 在多 seizure 内变异大，median 排序无法稳定锁定 SOZ。
4. **唯一双 ER 都 stable/moderate 的 subject = 583**：gamma moderate+concordant + broad stable+concordant，是 cohort 的 "model case"。
5. **3/16 (1084, 1146, 442) 临床 SOZ 全空**：cc 维度无意义，仅 producer health 可评估。
6. **2/16 (1077, 916) broad_ER insufficient**（n_ok=2 < 3）：broad band 在某些 subject 上 ictal recruitment 触发率极低（47/49 onset_unreached）。1077 同时 gamma s_sz=0.02（基本随机），整体不可用。
7. **`(stable, discordant)` 仅 916 一例**：与本文 916 节的"data-driven r_sz 与 clinical i 标注不一致 但 cohort 内部高度可重复"的现象一致；不是 sentinel-specific 异常，而是 cohort 中可观察但稀有的 case。
8. **gamma 上 producer health 与 clinical concordance 弱相关**：3 个 concordant 中只有 583 是 moderate；548、958 都 unstable。说明"focal channel 在排序前段"有时**不需要** producer 内部稳定也能 statistically 显示。

### Layer B 准入策略 — γ_a 锁定（v2.2.2, 2026-05-06）

**核心原则（重申 v2.2 metadata-only 合同）**：`producer_health` 决定 cell 进不进 Layer B；`clinical_concordance` **不**作 gate，但每个进入的 label 必须**强制携带** cc tag，让下游 PR 看到 concordance 状态自行决定是否信任。

> **v2.2.1 → v2.2.2 Layer B 框架修正**：之前 atlas/plan 的 γ_a 条目内部矛盾——一处写"主标签 = stable ∪ moderate"（这会包含 916 stable+discordant、253 moderate+discordant、1084 moderate+not_assessable），另一处又写"discordant/not_assessable 不进 Layer B"。reviewer 指出这是实质性错误：v2.2 合同明文 cc 只是 metadata，cc 不能既当 gate 又当 metadata。v2.2.2 锁定的 γ_a 改为：**producer_health 单独 gate + cc 永远 tag**。

**γ_a 准入分层定义**：

| 准入分层 | per (subject × ER) 入选条件 |
|---|---|
| **primary**（主标签）| `producer_health ∈ {stable, moderate}`，**不论** clinical_concordance |
| **sensitivity**（敏感性标签）| `producer_health == "unstable"` **AND** `clinical_concordance == "concordant"` |
| **drop**（不进 Layer B）| `producer_health == "insufficient"`，或 `producer_health == "unstable"` 且 cc ∈ {discordant, partial, not_assessable} |

**实测 cohort 应用结果（v2.2.2 tie-inclusive）**：

| ER | primary cells | sensitivity cells | drop cells |
|---|---|---|---|
| **gamma_ER** | 2: 916 (stable+discordant), 583 (moderate+concordant) | 1: 958 (unstable+concordant) | 13 (含 548† unstable+partial 因 v2.2.2 tie-inclusive 改判) |
| **broad_ER** | 4: 583 (stable+concordant), 139 (moderate+concordant), 253 (moderate+discordant), 1084 (moderate+not_assessable) | 4: 548, 590, 635, 922 (unstable+concordant) | 8 |

**Layer B 输出 4 份 JSON**（每份 entry 必须携带其 cc tag + producer detail）：

| 文件 | n_subjects | subjects |
|---|---|---|
| `data_driven_soz_core_channels_gamma_ER_primary.json` | 2 | 916, 583 |
| `data_driven_soz_core_channels_gamma_ER_sensitivity.json` | 1 | 958 |
| `data_driven_soz_core_channels_broad_ER_primary.json` | 4 | 583, 139, 253, 1084 |
| `data_driven_soz_core_channels_broad_ER_sensitivity.json` | 4 | 548, 590, 635, 922 |

**Unique subjects 跨双 ER**：{916, 583, 139, 253, 1084, 548, 958, 590, 635, 922} = 10 subjects 至少在某个 (ER × tier) 进 Layer B。其余 6 epilepsiae subjects (1073, 1077, 1084 gamma 那边, 1096, 1146, 1150, 442) 全部 drop（producer 在 cohort 上不可用）。

**重要 narrative 约束**：

1. **916 gamma 是 stable + discordant**：进 primary，但携带 `clinical_concordance=discordant` tag。下游 PR 看到这条 label 时**必须**意识到 data-driven SOZ 与 clinical i-label 显著不一致（详见本文 916 节）。"用 916 gamma label 进 cross-validation" **不**等于 "用 clinical SOZ"。
2. **253 broad 是 moderate + discordant**：与 916 类似但 producer 较弱。下游 PR 同样需 propagate cc。
3. **1084 broad 是 moderate + not_assessable**：1084 临床 i 标注全空（n_focal=0），cc 无法评估。下游 PR 用 1084 broad label 时只能当"无 clinical 对照的 data-driven SOZ"。
4. **(unstable, concordant) 进 sensitivity 而非 primary**：producer 内部不稳但 median 仍偏向 clinical SOZ，是一个**值得看但不当主结论**的 case；sensitivity JSON 的命名 + 文档说明必须明确提示下游"探索用，不当 primary"。

**进入 Step B.1**：实现 `src/data_driven_soz_pivot.py` label builder（plan §3.8 / §10 B1-B12 TDD）。准入规则、文件命名、entry schema 已锁定，可以直接对接 TDD。

---

## 待办

- [x] 用户选定 Layer B 准入策略：**γ_a 锁定**（2026-05-06）
- [x] 准入策略 commit 到 plan §6.3.1（v2.2.2）
- [x] reviewer fixes commit（tie-inclusive top-K + MWU 命名 + atlas Layer B framing 一致 + 测试覆盖 tie 路径）
- [ ] 进 Step B.1：实现 `src/data_driven_soz_pivot.py` label builder + TDD B1-B12（per plan §3.8 + §10）
- [ ] yuquan 9 subject 等 yuquan extract_seizure_window 后追跑（独立 PR，task #24）
