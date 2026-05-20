# lagPatRank Phantom Pseudo-Rank — Audit Diagnostic（2026-05-20）

> **状态**：Step 1（helper + TDD）+ Step 2（cohort audit）+ Step 3（gate evaluation）已完成。Gate verdict = **Broad re-derivation**。Step 4 (cosmetic) / Step 5a–5i (broad re-derivation execution) **未启动**——本归档只承载诊断结果，重跑授权由 user 单独决定。
> **范围**：本文档报告 cohort gate 决策，**不**修改 Topic 1 / Topic 4 主结论。所有数值口径维持现状直到 Step 5 跑完。
> **代码**：`src/lagpat_rank_audit.py` + `scripts/audit_kmeans_phantom_rank.py` + `scripts/augment_lagpat_audit_masked_stable_k.py` + `scripts/plot_lagpat_phantom_audit.py`
> **测试**：`tests/test_lagpat_rank_audit.py`（5 tests pass 2026-05-20）
> **数据**：`results/lagpatrank_audit/cohort_summary.csv` + per_subject JSON + 3 张 figures + README
> **Plan-of-record**：`docs/superpowers/plans/2026-05-20-lagpatrank-phantom-rank-audit-and-fix.md`

---

## 1. Bug 重确认（实证）

`lagPatRank` 在 legacy producer (`ReplayIED/.../hfo_net.py:289`) 由
`np.argsort(np.argsort(x))` 在 per-event channel center 向量上构造，**无 `eventsBool` 掩码**。
non-participating 通道仍获得 `[0, n_ch-1]` 范围内的有限整数排名，来源于 spectrogram 噪声 centroid。

**chengshuai/FC10477Y_lagPat_withFreqCent.npz（n_ch=8, n_ev=3513，2026-05-20 验证）**：

- ranks 中 NaN 数 = **0**，取值 ∈ {0..7}
- 9064 个 `bools==False` cells 全部有有限整数 rank
- 分布 U-shape：rank=7 占 19.1%、rank=0 占 17.8%、中段每档 ~10%。两端被偏置。

HFOsp 6 个 KMeans 调用点的 `np.where(np.isfinite, ..., 0.0)` guard 对 phantom int 完全无效。

## 2. Audit 设计与 noise-floor anchor

`AMI(original_labels, masked_labels)` 单独看会被「两者共享的 participation pattern」拉高 ceiling。
**Gate 阈值锚定到 per-subject seed-jitter noise floor**：

- `ami_seed_floor_original` = 在 original (phantom-contaminated) features 上跑 5 seed KMeans 的 pairwise median AMI
- `ami_seed_floor_masked` = 同上但在 masked features
- `ami_audit` = AMI(original@seed=0, masked@seed=0)
- **核心 metric**：`ami_audit_minus_floor` = `ami_audit − min(ami_seed_floor_original, ami_seed_floor_masked)`

Pre-registered gate（plan §3.2，2026-05-20 freeze before running）：

| Outcome | 条件 |
|---|---|
| Cosmetic | cohort-median Δ ≥ -0.05 AND n(Δ<-0.10) ≤ 2 AND no stable_k flip |
| Mixed | cohort-median Δ ∈ [-0.15, -0.05) OR 3-8 with Δ<-0.10 OR 1-3 stable_k flip |
| **Broad** | cohort-median Δ < -0.15 OR > 8 with Δ<-0.10 OR ≥4 stable_k flip |

## 3. Cohort gate verdict — Broad re-derivation

n = 40 (20 yuquan + 20 epilepsiae)，全部 status=ok。

### 3.1 主指标

| metric | cohort median | range | 触发条件 |
|---|---|---|---|
| `ami_audit_minus_floor` (Δ) | **-0.599** | [-0.995, -0.133] | < -0.15 → **Broad** ✓ |
| n subjects Δ < -0.10 | **40 / 40** | — | > 8 → **Broad** ✓ |
| n subjects Δ < -0.50 | **27 / 40** | — | — |
| `ami_seed_floor_original` | 0.997 | [0.695, 1.000] | n<0.7: 1 (zhangjinhan, k=6) |
| `ami_seed_floor_masked` | 0.996 | [0.408, 1.000] | high-k subjects 较低 |
| `ami_audit` | 0.368 | [0.000, 0.700] | — |
| `phantom_fraction` | 0.328 | [0.140, 0.458] | ρ(phantom, Δ) = **-0.42, p=0.007** |

**Verdict = Broad re-derivation**（同时满足 cohort-median + n_subjects 两个触发条件）。

### 3.2 Cohort-split：stable_k=2 vs stable_k>2

| cohort | n | median Δ | range Δ | n with Δ<-0.10 | n with stable_k flip |
|---|---|---|---|---|---|
| stable_k=2 | **35** | -0.609 | [-0.995, -0.296] | 35/35 | **1** (epilepsiae_916: 2→4) |
| stable_k>2 | 5 | -0.252 | [-0.676, -0.133] | 5/5 | 3 (huangwanling 4→3, zhaojinrui 5→6, zhangjinhan 6→5) |

### 3.3 stable_k 翻转细节（4/40）

3 个高 k 翻转都在 stable_k>2 cohort 内，且这些 subject 全部 `n_channels_union ≤ 5`，
已被 Topic 4 H3 主分析以独立 PCA-3 ill-posed 理由排除。**对 Topic 1 主线 cohort 无直接影响**。

唯一 stable_k=2 cohort 内的翻转：**epilepsiae_916: 2 → 4**，Δ=-0.813，masked floor=1.000，
masked scan passing_ks = {2, 3, 4}。属 material flip——下游凡是用 916 的 PR-2 stable_k=2 结果都需重审。
（注：916 已是 Topic 4 attractor Step 1 的 GOF-fail subject，`var_explained_curve=0.565`。）

## 4. 方法学含义——这是 finding，不是 noise artefact

**`ami_seed_floor_original` median = 0.997 范围 [0.695, 1.000]** 是关键发现。

它的含义：phantom-driven clustering **在 seed 间是可重现的**——不同 KMeans random_state 总是找到同一组 cluster。
phantom 不是把数据打散变随机，而是定义了一组**与真实排序信号完全不同**的、稳定的、可重现的 cluster——
这些 cluster 的 identity 由参与模式的端点几何 (per-event endpoint rank pollution) 驱动，
不是由通道之间的真实激活顺序驱动。

这是与 Topic 1 PR-4 panel d **identity-bias finding 同性质**的方法学发现：
- PR-4 panel d 已经证实"簇内 86% bias 来自 hub identity ordering"
- 本审计进一步：**簇本身的 identity（哪个事件属哪个簇）也很大程度上由 hub identity ordering + phantom endpoint pollution 决定**
- Topic 1 主线"PR-2 stable_k=2 是间期刻板时序的主导特征"在 **k=2 结构** 这一层面仍成立
  （34/35 stable_k=2 subject 在 masked features 上仍选 k=2），
  但**「哪个事件属哪个簇」的 label-level 结论需要全面重算**

这与 cluster_geometry.py 文档 §1-24 已经标注的 caveat 一致：
"KMeans Euclidean uses ALL channels' ranks (including non-participating positions' fallback ranks)"。
本审计把这条 caveat 从"已知但未量化"升级为"已量化、cohort 全部失败 cosmetic 阈值"。

## 5. 应该重跑的下游 PR（受 Step 5 broad 影响）

根据 Δ 全 cohort < -0.10 + 1 个 stable_k=2 翻转 (916)，下游所有消费 PR-2 cluster labels 的工作都需复跑：

| 下游 | 受影响 | 详情 |
|---|---|---|
| PR-3 per-cluster MI | YES | cluster_events = valid_events[labels == k]，labels 改 → MI 重算 |
| PR-4A day/night template occupancy | YES | 模板从 cluster labels 派生 |
| PR-4B/D rate-state coupling | YES | L1 / L2 / L3 全部用 cluster labels |
| PR-4C seizure proximity | YES | 同 |
| PR-5 / PR-5-B template recruitment shift | YES | `dominant_global` 直接派生自 cluster |
| PR-5-A novel-template gate | YES | gate 跑在 cluster templates 上 |
| PR-6 H1 endpoint anchoring | YES | endpoint 派生自 cluster templates |
| PR-6 H2 forward-reverse swap | YES | swap_class 派生自 cluster templates |
| PR-6 Step 6 held-out | YES | KMeans 同 bug |
| PR-6 supplementary rank displacement | YES | 用 PR-2 template_rank |
| PR-7 antagonistic pairing | YES | event-level lag-1 / run-based 都用 cluster labels |
| Topic 4 attractor Step 1 | YES | build_rank_feature_matrix 同 bug |
| cluster_geometry.py PCA embedding | YES | compute_pca_embedding 同 bug |

不受影响（已 bools-masked）：
- `_legacy_hist_mean_rank` template construction（仅 `template[ci]=ci` fallback 是已知 known issue）
- `_center_rank_matrix` 有 `min_participation=10` gate
- `_multi_seed_tau_summary` tau 计算用 bools mask
- PR-6 `valid_mask` 从 raw bools 派生

## 6. PR-7 P3 framework-level 警示

PR-7 P3 INCONCLUSIVE-locked 是 paper-framework-level decision
（`docs/paper1_framework_sba.md` v1.1.2 + `docs/archive/topic1/pr7_template_pairing/pr7_addendum_p3_equivalence_2026-05-01.md`）。
若 Step 5g masked re-derivation 把 P3 推到 PASS 或 FAIL（任意一侧），**这是 framework revision 而非 numerical update**——
必须先停下来发起 framework 修订 review。

## 7. 下一步

由 user 决定是否进入 Step 5 broad re-derivation（plan §5）。Plan 已经写好分步骤 + 两个 advisor checkpoint：
- Checkpoint A: 5a + 5b 后审 stable_k flip 数 + reproduced 集合变化
- Checkpoint B: 5d 后审 PR-4B/D 主线翻转

Step 4 (cosmetic-only) 路径已被 cohort gate 实证排除，**不再可用**。

如果暂不启动 Step 5，则现有 Topic 1 / Topic 4 主结论数字维持，但每条数字背后都背负
"cluster identity 与 phantom-de-biased re-clustering 共享信息量低于 seed jitter"的方法学 caveat。
中期 PPT 报告时建议在 Topic 1 §3.1 caveat 段加一句 forward reference 到本归档。
