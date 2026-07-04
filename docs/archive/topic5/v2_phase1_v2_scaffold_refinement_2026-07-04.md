# Topic 5 V2 Phase-1-v2 — Candidate Scaffold Refinement (survive? / who? / when?)

date 2026-07-04 · 分支 `topic5-v2-phase1` · tier（pre-registered 天花板，锁定）= **exploratory candidate scaffold refinement**（**不是** formal spatial-null positive / HFO-/LVFA-/ripple-specific / timing-order replay / criticality / 机制）

> 前身 = Phase-1 已验收（`v2_phase1_band_scan_backbone_2026-07-02.md`）：20/17 队列、n_perm=1000，发作 onset 后 0–20s 多频带能量空间场与间期 HFO-derived geometry (G_HFO) 对齐，cohort 层跨频段超过弱/全局空间 null（FWER 6/7 primary）+ order null（strong 子集），band-generic、NOT ripple-specific；但 per-subject 弱、formal within-shaft Gate A 未评估、Gate B/C 未跑、仅 onset 短窗。Phase-1-v2 回答由此打开的三个必要问题：**survive? / who? / when?**

---

## 0. 摘要（朴素话）

**测了什么**：每个癫痫病人的间期 HFO 有一张"平时谁先谁后"的传播顺序几何（G_HFO），平滑成空间图；发作刚起头时电极阵列上各频段的能量也点亮成空间图。核心现象 = 这两张图长得像（Phase-1 已确认 cohort 层像、跨频段都像、非 ripple 独有）。Phase-1-v2 问由此打开的三件事：**这种像扛得住把宽带/1-f 扣掉吗（survive）· 是每个病人都有还是少数撑起来（who）· 发作前就已经在还是点着才现（when）**。

**揭示了什么**：

- **survive?** —— 对齐**不是纯宽带招募**（扣掉所有频段共享的宽带成分后，7 个 primary 频段里还有 4 个、两池都含 α+γ，对齐仍超随机）；但那个"频段特异残差层"**大半其实是 1/f 背景**（扣掉 1/f 后只剩 **γ(gamma_LVFA)** 一段两池都稳）。ripple 段看似扣 1/f 后在 broad 池冒头，一查是**多重比较天花板假象 + 一个采样贴奈奎斯特的病人带偏**，**绝不作 HFO/ripple 特异**。→ 修饰语升到 **residual-robust（有频段特异残差层、非纯宽带）但大半 1/f-可归因、非 ripple 特异**。

- **who?** —— cohort 层的阳性是一个**不均人群的聚合**：典型病人 7 个 primary 频段里只有 2 段稳，≥4/7 的只 6/20（broad 7/17）；没有单一病人特征能预测谁强；频段梯度贴 0（band-generic）。→ 修饰语 **subject-heterogeneous**。

- **when?** —— 骨架**发作前就已经高且平**（远/近发作前无爬升），发作起始处有个**不大的上抬**（broad 池符号翻转支持、narrow 临界），且这个上抬只有锚在**电生理(EEG)起始**时才看得清（锚临床起始会被抹平，因为 EEG 起始常比临床早几十秒）。→ 修饰语 **preictal-present（静态解剖易感场）+ modest onset increment**。

**一句话（最稳口径，可直接引用）**：
> Interictal HFO-derived geometry marks a broad, largely trait-like early-ictal spatial recruitment scaffold that overlaps a shared low-to-mid-frequency (partly aperiodic) field: it is heterogeneous across subjects, present pre-ictally with a modest onset-associated increment (clearer under an EEG-onset anchor), carries a frequency-specific residual layer beyond broadband recruitment (robust only for gamma after 1/f control), and is **not** ripple/HFO-specific.

---

## 1. W1「扛得住吗？」— Gate B/C（residual 存活；n_perm=1000）

**测了什么 / 怎么测的**：把两样平凡的替代解释分别扣掉，再看对齐能不能超过"杆内洗牌"的空间随机场（cohort 层 subject-level permutation of the median + max-over-bands FWER，7 primary 为 family；判据 = `max_over_bands_p < 0.05` 存活）。**注意：formal within-shaft Gate A 仍未评估（cohort 空间 null 强度 = subject_wide_weak），所以看的是各 residual 自己的 cohort-perm FWER 存活，不是 formal gate flag。**

- **common_resid（LOBO 共有场残差）**：每个频段功率减去其余 6 个 primary 频段的平均（= 各频段共享的宽带成分），只留该频段自己的偏离。
- **aperiodic_resid（1/f 残差）**：每个通道把 1/f 背景拟合成直线扣掉，只留超出背景的正的部分。**1/f 拟合质检已过**：cohort median r²≈0.79、失败率 1.7%（<20% 门）→ 检验可信（失败率高才只能描述性）。

**结果（存活频段数 = max_over_bands_p<0.05，7 primary）**：

| feature | narrow (n=20，主·干净扣除梯) | broad (matched n=16*) | 说明 |
|---|---|---|---|
| RAW | 6/7（唯 ripple_high 不过） | **3/7**（δ, α, β）〔matched n=16；n=17 Phase-1 基线 = 6/7〕 | broad 计数受队列大小影响，见下 |
| **common_resid** | **4/7**（δ, α, γ, hg） | **4/7**（θ, α, β, γ） | **α+γ 两池都存活** → 频段特异残差层，**非纯宽带（Outcome B）** |
| **aperiodic_resid** | **1/7**（γ） | **2/7**（γ, ripple_high†） | **γ 两池都存活** = 唯一扛过两道扣除；余大半 1/f-可归因 |

**⚠️ broad 计数受队列大小影响（承重·like-for-like，CLAUDE.md §5）**：broad 三行都算在 **matched n=16** 上。**排除 1146 让 raw 从 n=17 的 6/7 掉到 n=16 的 3/7**——1146 是强正向被试（broad n_sig 7/7）撑高了 raw，而 residual 受其影响小，故 broad 上 common_resid(4/7) 竟 **≥** matched-raw(3/7)。**所以 broad 的"存活频段数"不是干净的逐级扣除梯**。**承重证据不是 broad 计数，而是：(i) narrow(n=20) 干净扣除梯 6→4→1；(ii) 跨池 band-identity——α+γ 过 common_resid 两池、γ 过 aperiodic 两池——与队列大小无关。** Outcome B（common_resid ≥2/7）两池都成立。

\* broad=16（排除 epilepsiae_1146）：排除理由 = **单被试 n_perm=1000 单线程 ~2h（OMP 不助，Python-loop permute）计算受限** + 1146 是**强正向离群**（broad n_sig 7/7）——排除强正向被试是**保守**（matched raw 6/7→3/7 印证只会降低存活）。1146 已在 narrow 全量分析（narrow 里它是 within_shaft_strong；**broad 里是 subject_wide_weak**——within_shaft_strong 是 narrow 属性、不迁移到 broad context）。1146-broad 后台补跑中，结论对其稳健。
† ripple_high broad 的"存活"= **假象**，见 §1.1 sanity。

**判读（accepted wording，LOCK）**：
> 对齐**不是纯宽带招募/共有场**（common_resid 后两池各 4/7 存活、α+γ 一致 → 存在频段特异残差层）——即 **Outcome B**，非 Outcome A。但扣掉 1/f 后只剩 **gamma_LVFA** 两池稳存活，说明那个频段特异层**大半可归因于 1/f 背景**；只有 gamma 扛过宽带+1/f 两道扣除。**（gamma 存活是描述性事实，不构成"LVFA-specific 机制"主张——tier 禁。）**

### 1.1 Outcome C sanity — broad ripple 的 aperiodic "存活" = 假象（DESCRIPTIVE ONLY）

broad-16 aperiodic ripple_high 过 FWER（max_over_bands_p=0.022），触发 Task 1.4 强制 sanity（与 Phase-1"ripple 最弱"矛盾）。**六项全指向假象 → 只描述、绝不写 HFO/ripple 特异**：

1. **多重比较天花板假象**：aperiodic ripple 的 cohort 中位 delta = **0.010 < raw 的 0.036** —— 扣 1/f 让 ripple 对齐**更低**，不是更高；"存活"只是 1/f 把别的频段压下去、max-over-bands 天花板降了，ripple 才**相对**冒头，非绝对增益。
2. **存活不随效应量**：narrow aperiodic ripple 中位 delta **更高（0.052）却没过 FWER**（0.156）。
3. **cohort 异质**：broad 16 人里 9 正 7 负，916/922 强负（−0.25 / −0.27）。
4. **采样边界带偏**：fs=512Hz 的 253（ripple 150–250Hz 贴奈奎斯特 256、edge_safe=220）是最强正向 driver 之一。
5. **池间不一致**：narrow 不显著、broad 才显著。
6. 与 raw（ripple 从未过 FWER）+ Phase-1 矛盾。

拟合 QC 干净（r²0.79、每 sz 排除 18 个工频 bin）→ 不是拟合失败，是**真·天花板假象**。

### 1.2 单-covariate confound（Task 1.4 Step 2-3，descriptive · deferred）

confound maps（每触点 hfo_rate / baseline_band_power / broadband_1_250 / **shaft_position（用杆序索引，非 along_axis_mm，防自证循环）**）已在 20/17 构建（`{narrow,broad}/phase1_confound_maps.json`）。单-covariate confound-adjusted alignment（把 G_HFO 的 typical_rank 对每个 covariate 残差化后重对齐、单-covariate 为主不塞 combined 大模型）是一项 **descriptive robustness follow-up**。**核心 W1 结论（Outcome B、非纯宽带、NOT ripple-specific）不依赖它**——common_resid 的 LOBO 已控宽带成分；confound 只是补充"对齐是否也不被 HFO-率/功率/杆位地形解释"。confound-adjusted 逐 covariate 数值待补（`run_topic5_v2_nulls.py --confound-maps <path>`，descriptive）。

---

## 2. W2「谁有？」— subject phenotype（cohort 是聚合、不均、无单一表型）

**测了什么 / 怎么测的**：纯读 Phase-1 已算好的 per-subject 结果。对每个病人数"7 个 primary 频段里，有几段对齐既为正、又超过该病人自己的空间零假设 p<0.05"（n_sig，满分 7）；算连续画像；看有没有任何单一病人特征能预测 n_sig。

**揭示了什么**：
- **cohort 6/7 是聚合、per-subject 弱**：narrow 每人显著频段数中位 = **np.median 2.5（偶数池 n=20；下中位 2，与 Phase-1 归档"2"一致——同一分布、偶数池取中位法差异，非数据变化）**；**≥4/7 的只 6/20**（broad 中位 3、≥4=7/17）。**count(≥4)=6/20 才是承重（无量纲、跨取中位法一致）的"少数驱动"证据。**
- **两池都稳的只 {1146,1150,384}**；1146 是唯一"杆内空间强 + 多频段阳"个例（narrow 5/7、broad 7/7；**n=1，非 cohort**）。
- **无单一干净表型**：n_sig 对每个**独立**病人特征（发作次数/触点数/原始对齐幅度/跨发作一致性/频段梯度）Spearman 几乎全在 |r|=0.4 门下（唯跨发作一致性 narrow 过 r=0.48、broad 不过 → 不稳健）。过闸看似强的四个特征都是**同一对齐余量向量的再描述**（天然共变、非独立预测），已剔除。
- **频段梯度贴 0（band-generic）**：HFA(80–250)−low(1–13) 对齐差中位≈0，两池都不偏某频段。描述性分桶多数落"band-generic 平"或"弱/缺"，少数偏低频。**非 KMeans/统计 subtype 主张。**

**图**：`figures/phase1_v2_W2_subject_phenotype.png`（A 频段梯度散点按 tier 着色；B 相关性筛查 |r|=0.4 门）。

---

## 3. W3「什么时候？」— peri-ictal trajectory（发作前已在、起始处小增量、EEG 锚更清）

**测了什么 / 怎么测的**：对齐度按"相对 EEG 起始 −100→+20s"滑窗量出（10s 窗 5s 步、5 时间段）。骨架分 = 7 primary 对齐中位（band-generic）。聚合 窗→发作→病人→cohort（每层中位）。主检验 = **对病人配对差做符号翻转置换**（n_perm=20000 两侧；不用窗标签打乱）。双锚：EEG 起始（主）vs 临床起始（敏感性）。

**⚠️ 关键数据发现（承重）**：EEG 起始常比临床起始**早几十秒**（eeg_onset_rel 最多约 −86s；多被试 >20s）→ Phase-1 临床锚的"起始+0–20s"经常已**深入发作**。这坐实 EEG 主锚的理由，也是为什么专门修 icf 锚。

**揭示了什么（描述性）**：
- **发作前已高且平**（近前−远前≈0，p≈1.0/0.80 两池 EEG）→ 非"逐渐逼近阈值加载"，更像**静态/解剖易感场**。
- **起始处不大的上抬**（起始后−远前、起始后−近前）：broad 符号翻转 **p≈0.0046/0.0044（支持）**、narrow 临界（p≈0.06、Wilcoxon<0.05）。
- **EEG 锚比临床清**：上抬在临床锚下被抹平 → 临床起始滞后模糊了真动力学；**以 EEG 起始为生理主结论**。
- band-generic。骨架分 = 每窗现存 primary 频段中位；~7% 远前窗只有 5/7 频段（掉的总是 80–250Hz HFA 两段）；要求全 7 段重算幅度移 ≤~20% 但**无一显著性判决改变**（稳健）。

**图**：`figures/phase1_v2_W3_trajectory.png`（A cohort 轨迹 EEG 主/临床虚线；B 三对比 + 符号翻转 p）。

---

## 4. 整合结论 + tier + 承重 caveat

**整合（三问合起来，最稳口径）**：间期 HFO 几何标记一个**广泛、大半 trait-like 的发作早期空间招募 scaffold**，它叠在一个**共享的低-中频（部分 aperiodic）场**上——**跨被试异质、发作前已存在且起始处有不大增量（EEG 锚更清）、有超出宽带招募的频段特异残差层（1/f 控制后只 gamma 稳）、NOT ripple/HFO-specific**。对应 plan §Supported-conclusion **情况 A/B 之间**（宽带/共场重叠 + 频段特异残差层 + preictal loading 的一个弱版本），**不到情况 C**（不是 subject-specific + state-dependent 的干净 pathological critical mode 候选——那要 Phase-2 criticality）。

**tier 天花板（LOCK，不可越）**：exploratory candidate scaffold refinement。**禁**：formal spatial-null positive · HFO-/LVFA-/ripple-specific · timing-order replay · criticality-proven · 机制 · "过任何空间随机场" · "formal Gate A/B/C passed"。**允许**：band-generic · NOT ripple-specific · suggestive spatial co-structure · candidate early-ictal recruitment scaffold · frequency-specific residual layer（beyond broadband）· preictal-present。

**承重 caveat（放正式报告首页）**：
1. **per-subject 弱**：narrow ≥4/7 仅 6/20（cohort 6/7 是聚合）。
2. **weak/global spatial null 下成立、formal within-shaft Gate A 未评估**（2/20、反保守 likely inflated）——所有"存活"都在弱空间 null 尺度。
3. **W1 residual**：非纯宽带（Outcome B）但大半 1/f-可归因、只 gamma 扛过两道扣除、**NOT ripple-specific**（broad ripple 假象）。
4. **EEG↔clinical 锚差大**（临床锚会误导 when 判读）。
5. **broad = 16**（1146 计算受限排除；保守；后台补跑）。confound（Task 1.4）descriptive、[pending]。

---

## 5. 路径 / artifacts / tests

- **代码**：`scripts/analyze_topic5_v2_subject_phenotype.py`（W2）· `scripts/run_topic5_v2_trajectory.py` + `plot_topic5_v2_W3_trajectory.py`（W3）· `run_topic5_v2_alignment.py`（peri-ictal EEG-frame + `--anchor` + `--config`）· `build_topic5_v2_aperiodic_cache.py`（+QC）· `run_topic5_v2_{nulls,gates}.py`（residual，复用）
- **artifacts**（`results/topic5_ictal_recruitment/v2_band_scan/`）：`phase1_v2_subject_phenotype.csv`（+band_profile）· `periictal/`(eeg)+`periictal_clin/`(clin) window_long + `phase1_v2_alignment_trajectory.csv` + `phase1_v2_trajectory_contrasts.csv` · `{common_resid,aperiodic_resid}_cache/`+`aperiodic_qc.json` · `{narrow,broad}/phase1_{null,gate}_{common_resid,aperiodic_resid}_*`
- **图**：`figures/phase1_v2_W2_subject_phenotype.png` · `figures/phase1_v2_W3_trajectory.png`（+ figures/README.md）
- **tests**：`tests/test_topic5_v2_phase1_v2.py`（W2+W3, 20）· `tests/test_topic5_v2_band_scan.py`（+aperiodic QC, 36）
- **commits**：cd692ce/0465f8a(2.1) f3c6531(2.2) d479001(2.3) [3.1 in 3d6d839] 98c1187(3.2 Step0) c8720bf(3.2b) 8f1ddf1(3.2b-fu) 11575ce(1.1 QC) [+ 1.3/1.4 archive TBD]
- **plan/spec**：`docs/superpowers/plans/2026-07-04-topic5-v2-phase1-v2-scaffold-refinement.md` · `docs/superpowers/specs/2026-07-02-topic5-v2-phase1b-gate-closure-spec.md`（§EXP/§2/§5/§6）
