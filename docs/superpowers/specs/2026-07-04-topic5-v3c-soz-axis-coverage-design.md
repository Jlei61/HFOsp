# Topic 5 V3c — 间期传播轴对临床 SOZ 的空间覆盖，及"轴外扩"触点的发作招募时序

date 2026-07-04 · 状态：**EXPLORATORY，spec（未实现）** · tier=exploratory
· 前身 = V3a（`docs/archive/topic5/v3a_mode_transition_2026-07-04.md`，脆弱阳性）
· 姊妹 = V3b（M3B 模型–数据一致性 H3d，未实现，**命名不同**）
· 命名：**V3c / V3-SOZ-stratification**（审阅 P1-3：不撞 V3b）
· 复用 V3a 代码底座（`scripts/_topic5_v3_io.py`、`src/topic5_v3_mode_transition.py`）+ `src/topic5_ictal_recruitment.py` + `src/seeg_coord_loader.py`
· 设计来源：本会话 brainstorm（用户 16 点 co-design + 决策 1–9 + agent R1–R4 + label-blind assay-QC 探针）

---

## 0. 一句话定位

**间期高频传播轴（`A`）在空间上是否系统性覆盖临床发作起始区（`S`），并定义出一圈结构化的"轴外扩"触点（`A∖S`）——这是承重 primary（纯空间标签 + 空间 null，不依赖 latency）。在此之上，再用发作招募时序问：这些 `A∖S` 触点在发作里是与 SOZ 同期点亮（onset-synchronous），还是 SOZ 之后才点亮（downstream）——这是一个受 assay-质量门控的机制 secondary，latency 不合格不伤 primary。**

论文角色（最稳写法）：*interictal axis spatially covers clinical SOZ and defines a structured peri-SOZ surplus; ictal latency then tests whether that surplus is onset-synchronous or downstream.* 无论 latency 支持 H-A、H-B、还是因 ties/censoring 判不出，这部分都不会失败。

---

## 1. 科学目的（朴素话：测了什么 / 怎么测 / 想揭示什么）

**测了什么。** 把三张**互相独立**的标签叠在每个电极触点上：① 这个触点在不在病人**平时那条高频小放电的传播路线**上（间期传播轴 `A`）；② 临床医生有没有把它圈成**发作起始区**（`S`）；③ 发作时这个触点**多早被点亮**（招募 latency）。问间期这条网络和临床判断在哪儿一致、哪儿分歧，分歧的触点在发作里是早还是晚。

**怎么测。** 先数**覆盖**——这条轴盖没盖住临床 `S`、往外多盖了多少（`A∖S`），并且盖得是不是超出"同样植入几何下的随机轴"。再在**同一次发作里**，比"多盖出来的触点 `A∖S`"和"轴上的临床 SOZ 核 `A∩S`"谁先被点亮。所有对比都跟随机重排比，以病人为单位。

**想揭示什么。** 间期高频网络是不是把临床发作起始区**整个包住、还往外扩了一圈结构化的组织**；那多扩的一圈到底是——
- **和 SOZ 同期亮**（H-A：间期网络可能标出了临床 SOZ 之外、也参与早期发作招募的组织 → 支持"网络指标比临床 SOZ 更贴近真起源"，接术后 outcome capstone）；还是
- **SOZ 之后才亮**（H-B：那圈是下游扩散，医生正确地没圈 → 和现有"轴 = 网络骨架 readout、不是发作起点"的故事一致）。

**两个方向都是有价值的结果，不存在"阴性 = 失败"。**

（内部代号：`A`=`classify_subject_contacts`.`is_axis`（间期传播模板成员）；`S`=`epilepsiae_soz_core_channels.json`；latency=`bb_zt` 首次超阈；坐标=`seeg_coord_loader`。）

### 1.1 为什么 coverage 是 primary、latency 只能是 gated secondary（决策 1 = 1b）

原始设想（V3a 顺问）是让 latency 承重（"axis-surplus 早/晚"）。但两条硬约束把它降级：

1. **latency 在本仓库历史上有承重风险。** Stage-2 招募排序"跨发作不稳、最早触点 tie 在窗口起点"；Stage-2b 峰晚 + 符号翻；z-ER 中后期偏示意。
2. **n=6 下 latency 一旦出现大量 t0 ties / right-censoring / 阈值敏感，primary 直接塌**——而 coverage 仍能作为干净补充分析成立。

label-blind assay-QC 探针（§附录 A.2）实测证实了这个风险是真的（1077 有 56% 轴触点在发作起点已亮、1150 有 44% 从不超阈）。因此：

- **coverage 做 primary**（只靠空间标签 + 空间 null，完全不碰 latency）。
- **latency 做预注册的、受 assay-质量门控的机制 secondary**：`Primary spatial endpoint; pre-specified mechanistic latency endpoint if recruitment timing passes assay-quality criteria.`（这比"看完 sanity 再决定 primary"更抗审稿——避免 data-dependent endpoint selection：assay-QC 是 **label-blind** 的，只判 latency 这个量本身可不可信，**不看** SOZ-vs-surplus 组差。）

---

## 2. 集合语言与概念纪律（审阅 point 3，必须遵守）

**V3c 全程用集合语言，不再用"off-axis"这个词**（除非严格限定它是"not in `A`"还是"non-HFO-active"——两者不同，混用会让审稿人看不清 coverage 和 2×2 退化的关系）。

三个必须分清的集合：

| 记号 | 定义 | 说明 |
|---|---|---|
| `HFO-active` | 间期 HFO 参与度 > 0 的触点 | 与 `A` 不同！是更宽的集合 |
| `A` | `classify_subject_contacts`.`is_axis` = 在间期传播模板里（有限 `typical_rank`） | 承重轴定义；broad top-N≈20 |
| `S` | 临床 SOZ 触点（SOZ JSON），∩ all-clean 池 | 临床标注真相 |

四个交并集（V3c 的全部对象）：

| 记号 | 名称 | 科学含义 |
|---|---|---|
| `A∩S` | axis-covered SOZ | 轴盖住的临床 SOZ 核 |
| `A∖S` | **axis-surplus** | 轴多盖出来的触点（承重对象）|
| `S∖A` | **axis-missed SOZ** | 临床 SOZ 里不在轴上的（635 有 3 个；单独画，不是噪音）|
| `¬A` | non-axis | = off-axis-strict ∪ ambiguous；**V3c 不用此分区做检验** |

**关键澄清（635）**：635 的 3 个"临床 SOZ 但不在传播模板里"的触点是 **`S∖A`（axis-missed SOZ）**，它们是 `HFO-active` 的（放间期高频）、只是不在 propagation template 里（V3a 里归 ambiguous）——**不是** "off-HFO / never fires"。此前"既离轴又 SOZ 的格子按定义就空"这句话用的是 V3a 参与度定义的 off-axis-strict（参与度≤0.10），和 `S∖A` 是两码事。V3c 完全绕开参与度分区，只做 `A`、`S` 的集合运算。

**措辞纪律（审阅 point 13）**：spec 与结果初稿一律用中性词 **`axis-surplus` / `A∖S`**，**不用**"那圈"。只有在 V3c-3 空间聚集分析证明 `A∖S` 确实围绕 SOZ 或沿同杆连续后，讨论里才可用 `peri-SOZ axis extension / axis-adjacent surplus`；若 `A∖S` 是多杆离散点，则明说"扩散而非环绕"。

---

## 3. 数据现状 + 两层 eligibility + cohort

### 3.1 数据现状（实测，见 §附录 A）

- 原 2×2 三格退化：以 V3a 参与度 off-axis 定义，"离轴∩SOZ"全空、按定义空。→ 真信号在单边 `A∖S`。V3c 用集合语言重述为：`S ⊆ A`（多数被试）+ `A∖S` 大（外扩）。
- 覆盖非同义反复：broad 轴下 6/7 被试 `A∩S = S`（100% 覆盖），但 **635 只 7/10**（3 个 `S∖A`）→ coverage 是有方差、可检验的真问题。
- latency assay 非退化但有真 censoring：uniq_ranks 中位 7–11、阈值稳定性 ρ 0.68–0.84（好）；1077 t0=56%、1150 cens=44%（坏，正好该被 QC 门挡）。

### 3.2 两层 eligibility（审阅 point 10：coverage 不被 latency 的 n 拖累）

**Coverage eligibility（V3c-1 用）**：有临床 SOZ 名单 + 可算 broad 轴 + `|S∩pool| ≥ 1` + `|A| ≥ 1`。**不需要 latency**。
→ broad 可分类 SOZ 被试 = **7**：139, 253, 635, 1077, 1096, 1150, 1146（442/958 无 broad cache，进 narrow 敏感度）。

**Latency eligibility（V3c-2 用）**：`|A∖S| ≥ 3` **且** `|A∩S| ≥ 3`（primary 组定义）**且** ≥ 2 发作 **且** ≥ 2 informative 发作（informative 定义见 §5.2 QC）。
→ broad 满足集合门 = 7（同上），再经 label-blind assay-QC 后预计 ~5 assay-valid（139/253/635/1096/1150，其中 1150 需 censoring 敏感度一致；1077 assay-QC 不过 → descriptive-only）。1146 待 QC。

### 3.3 Cohort（broad 主 / narrow 敏感度，永不 pool）

| 用途 | broad（主） | narrow（敏感度） |
|---|---|---|
| Coverage（V3c-1） | 139,253,635,1077,1096,1150,1146（n=7） | 1096,1146,253,442,958（n=5；SOZ-available narrow）|
| Latency（V3c-2） | 集合门 7 → assay-QC 后 ~5 | 253,442,958（n=3，仅方向/effect-size 一致性，不作独立推断）|
| 无 SOZ 名单 掉出 | 384,620,916,**1125**（V3a 最佳被试） | 同 |

narrow 口径（审阅 point 12）：narrow 只作**方向与 effect-size 一致性的敏感度**，不给独立 inferential 语言。写：*The narrow-axis analysis was treated as a sensitivity analysis for directionality and effect-size consistency, not as an independent inferential cohort.* 不一致时不说"失败"，解释为 *narrow axis captures only the highest-confidence propagation core, whereas broad axis captures the extended interictal propagation scaffold.*

**注意：broad 与 narrow 队列成员部分不同、非严格子集。** 三个被试（1096, 253, 1146）两队列都在，对它们 narrow-vs-broad 是真正的 within-subject 轴定义敏感度；442/958 只有 narrow（无 broad cache）。故 narrow 敏感度 = "轴定义变宽/变窄时方向是否一致" + "narrow-only 被试的补充描述"，两成分要在 README 分开标，不混成一句"narrow 复现了 broad"。

---

## 4. V3c-1：间期轴对临床 SOZ 的空间覆盖（**primary**）

**Cohort**：coverage-eligible（broad 主 n=7 / narrow 敏感度 n=5）。

### 4.1 承重 + 描述 endpoints（审阅 point 4：不只 sensitivity，还要 surplus burden）

per-subject 三指标：

| 指标 | 公式 | 角色 | 科学含义 |
|---|---|---|---|
| **SOZ coverage / sensitivity** | `|A∩S| / |S|` | **primary** | 轴是否覆盖临床 SOZ |
| surplus fraction | `|A∖S| / |A|` | 描述（**R1**：半机械）| 轴是否明显外扩 |
| Jaccard | `|A∩S| / |A∪S|` | 描述 | 覆盖与泛化的平衡 |

**R1（agent 补充）**：broad 轴近固定大小（top-N≈20，实测 17–20），`surplus fraction ≈ (20−|A∩S|)/20` 基本由 coverage 决定、**不是独立证据**。coverage 的**特异性**只压在（a）same-shaft null（§4.2）和（b）V3c-3 空间聚集上，**surplus fraction 不承独立权重**。

### 4.2 Null 层级（审阅 point 5：三层，不三个都当 primary）

| null | 保留什么 | 用途 | 解释力 |
|---|---|---|---|
| all-contact spatial shuffle | 仅 `|A|` | 最松 sanity | 只说明非全脑随机 |
| **same-shaft shuffle preserving per-shaft axis counts** | `|A|` + 每杆轴触点数 | **primary coverage null** | 控植入几何 |
| HFO-rate-matched + distance-to-SOZ-matched | + HFO 率分布 / 到 SOZ 距离 | sensitivity | 排"只是 HFO-rich / 离 SOZ 近" |

**primary null = same-shaft shuffle**（每杆内随机选 `k_shaft` 个触点当轴，`k_shaft`=该杆实际轴触点数，逐杆独立；保 `|A|` 与每杆轴数不变；`S` 固定）。1000 次。

**R2（agent 补充，钉进 claim 语言）**：same-shaft null 证的是"**超出植入几何**"，**不是**"超出 HFO-rich"。轴本身从间期 HFO 定义、SOZ 又天然 HFO-rich，same-shaft null 控几何但不控"两者都是高频区"这个巧合。所以 primary coverage 的措辞只能写"轴覆盖 SOZ 超出植入几何预期"；更强的"超出单纯高频富集"**只有 HFO-rate-matched 敏感度 null 才授权**。

### 4.3 推断（§7 通用纪律）

per-subject：观测 coverage 在自己 same-shaft null 的分位（单侧，高）。
cohort：**nested subject-level null（cohort-median null，primary）**——每置换：每被试各自 same-shaft shuffle → 每被试 null coverage → cohort median null coverage；观测 cohort median 在该分布的分位。附：过各自 null 的被试数（binomial vs α=0.05）+ LOSO。

### 4.4 Coverage 的**双条件** claim（审阅 point 4 收尾）

coverage 主张成立需**同时**：
1. `|A∩S|/|S|` 高于 same-shaft null（cohort-median null 显著）；**且**
2. `A∖S` 非空 **且** 空间上有结构（V3c-3：聚集超 same-shaft shuffle）——排"一个超大的轴平凡地盖住 SOZ 但没特异性"。

---

## 5. V3c-2：axis-surplus 的发作招募时序（**gated mechanistic secondary**）

### 5.1 门控原则

latency 只有在通过**预注册、label-blind 的 assay-QC**（§5.2）后，才被称为 mechanistic endpoint 并进入 cohort AUC 推断；不过则降为 descriptive-only，**不伤 V3c-1 primary**。

### 5.2 预注册 label-blind assay-QC（审阅 point 2A + agent 探针定标）

只在轴触点 `A` 上算、**不看 SOZ/surplus 组差**。per-subject 记录并公开（§附录 A.2 已有实测基线）：

| QC 项 | 预注册门槛 | 探针实测（6 broad） |
|---|---|---|
| finite latency 比例 | ≥ 0.40 | 37–64%（1077=37 边缘）|
| t0 / window-start 已亮比例 | ≤ 0.50 | 12–56%（**1077=56 不过**）|
| median unique latency ranks / informative sz | ≥ 4 | 7–11（全过）|
| threshold Spearman（lat@2.0 vs @1.5 & @2.5）| ≥ 0.5 | 0.68–0.84（全过）|
| informative 发作数 | ≥ 2 | — |
| right-censoring 比例 | > 0.40 → **flag**：需 drop-censored & impute-30s 敏感度**同号** | 8–44%（**1150=44 flag**）|

- **informative seizure** = 在 `A∩S ∪ A∖S` union 上：非全 t0、非全 censored、≥ 3 unique ranks。
- **assay-valid subject** = finite≥0.40 ∧ t0≤0.50 ∧ uniq≥4 ∧ thrρ≥0.5 ∧ informative_sz≥2；否则 latency descriptive-only。
- **R4（agent）**：发作间稳定性 QC 盯**组级统计量**（每发作 AUC / 组中位）稳不稳，**不是** per-contact 排序可复现（Stage-2 塌的是后者；primary 是组间对比，能容忍 per-contact 顺序抖动）。

**assay-QC 固定输出 3 图**（审阅 point 2B）：① per-subject latency raster / rank（contacts × seizures，标 `A∩S`/`A∖S`/`S∖A`）；② QC bar（t0/censor/finite/uniq-ranks 率）；③ SOZ-vs-surplus AUC forest（每被试一点 + null 区间）。

### 5.3 Statistic（审阅 point 6：AUC/秩为主，方向写死）

- **primary contrast**：`A∩S` vs `A∖S`（axis-covered SOZ core vs axis-surplus，均在轴内，最干净）。
- **clinical sensitivity contrast**：`S` vs `A∖S`（全临床 SOZ）。
- **primary statistic**：`AUC_late = P(L_surplus > L_SOZ) + 0.5·P(L_surplus = L_SOZ)`（`L_surplus`=`A∖S` latencies；`L_SOZ`=`A∩S` primary / `S` sensitivity）。
  - `≈ 0.5`：surplus 与 SOZ 同步/不可区分；`> 0.5`：surplus 晚（downstream，H-B）；`< 0.5`：surplus 早（临床 SOZ 可能漏早期轴组织，**或** latency artifact → 走 artifact 检查）。
- **interpretive statistic**：`Δt = median(L_surplus) − median(L_SOZ)`（秒级 effect size；AUC 给排序、Δt 给生物意义。AUC=0.65 但 Δt=0.3s 不写强 downstream）。

### 5.4 Censoring 处理（审阅 point 9：作结果的一部分，不只 QC）

per seizure/subject 报：finite / right-censored / left-censored(t0) / largest tie block / unique ranks。

| 类 | primary | sensitivity |
|---|---|---|
| 未招募（right-censored）| 排最后，彼此 tie | (a) 去掉全未招募重跑；(b) 设 30s+ε 重跑 |
| t0（left-censored at 0）| 排最早 | 排除 t0-left-censored 重跑 |

**若排除 t0 后结论翻转 → latency 不能承重（该被试/该队列降 descriptive）。**

### 5.5 Null + 推断

null（同 §4.2 层级思路，label 版）：primary = 同杆 label shuffle within subject（在 `A∩S ∪ A∖S` 上随机换 SOZ-core/surplus 标签，保每杆两类计数）；sensitivity = distance-to-SOZ matched、HFO-rate matched。
推断（§7）：seizure→subject median→cohort median；nested subject-level（cohort-median）null；LOSO；无 pooled channel-level p。

### 5.6 H-A / H-B / Indeterminate（审阅 point 7 + **R3**）

预注册**三段式**解释（不是二分、不用 p>0.05 当 H-A）：

- **H-B supported（downstream）**：cohort median `AUC_late ≥ 0.60` **且** 多数被试同向（≥ ⌈n/2⌉+1 个 `AUC_late > 0.55`）**且** matched null 下显著 **且** `Δt > 2s`（或 > 采样/检测分辨率）。
  措辞：*Axis-surplus contacts were recruited after clinical SOZ contacts, supporting the interpretation that the interictal axis captures a broader propagation scaffold rather than only the seizure onset core.*
- **H-A compatible（onset-synchronous，描述性）**：cohort median `AUC_late ∈ [0.45, 0.55]` **且** 无一致 subject 级偏晚 **且** `|Δt| < 1–2s` **且** QC 显示非"大量 t0 ties 把 AUC 人为压回 0.5"。
  **R3（agent）**：n=6 下**无法**做正式等效检验（TOST 严重欠功率），故 H-A 只能是**描述性 compatibility**、**显式声明不是被证明的等效**。措辞：*compatible with onset-synchronous recruitment*——**禁**直接写"医生漏掉真 SOZ"。
- **Indeterminate**：`AUC≈0.5` 但 ties/censoring 重 → *latency ordering was not sufficiently resolved to distinguish onset-synchronous from downstream recruitment.* **禁**把 nonsignificant 当 H-A。

`S∖A`（axis-missed SOZ，如 635 的 3 个）单独画并解释：若 `S∖A` 早于 `A∖S` → 轴确实漏部分临床 onset 组织；若 `S∖A` 晚/不招募 → 临床 SOZ 较宽、轴反而更聚焦。

---

## 6. V3c-3：axis-surplus 的空间组织（**secondary descriptive**，审阅 point 14）

**Cohort**：coverage-eligible（同 V3c-1）。便宜，且是 coverage 双条件 claim 的第 2 条支撑。

per-subject 指标（坐标来自 `seeg_coord_loader`）：

| 指标 | 含义 |
|---|---|
| number of shafts containing `A∖S` | surplus 集中还是分散 |
| surplus shaft entropy / Gini | 空间聚集程度 |
| distance from `A∖S` to nearest `S` | 是否靠近 SOZ |
| contiguous surplus runs along shaft | 是否沿电极杆连续扩展 |
| observed clustering vs same-shaft shuffle | 是否超随机 |

判读：结构化 peri-SOZ extension（支持 coverage 特异）vs 弥散过度泛化（削弱 coverage 特异）。

---

## 7. 统计推断纪律（审阅 point 11，全 V3c 通用）

- 层级：contact → seizure → subject → cohort。**禁 pooled channel-level / pooled seizure-level p**。
- per-seizure statistic → per-subject median → cohort：报 subject median、IQR、每被试方向、**LOSO**、subject-level null 分位。
- cohort p：**primary = nested subject-level（cohort-median）null**（每置换在每被试内做同杆/匹配 shuffle → per-subject null → cohort median → 观测分位）。**不用** Fisher/Stouffer（n=6 会显得比实际强）。
- broad 主 / narrow 敏感度，**永不 pool**。

---

## 8. 承重 acceptance gates（数值门编码结论；memory: feedback_acceptance_gate_encode_conclusion）

每个承重定性主张 → 一个数值门 + 坏数据回归：

| 主张 | 门 | 坏数据回归 |
|---|---|---|
| coverage 显著（V3c-1 primary）| cohort-median coverage > same-shaft null 95 分位（p<0.05）| 打乱 `S` 后应 NULL |
| coverage 有特异性（双条件）| ∧ V3c-3 聚集 > same-shaft null | 均匀散点 surplus 应不聚集 |
| latency assay-valid | §5.2 五门全过 | 全 t0 / 全 censored 应判 invalid |
| H-B supported | §5.6 四条件全满足 | 随机 latency 应 AUC≈0.5 |
| H-A compatible（描述性）| §5.6 带内 + QC 干净 + **显式非等效** | t0-tie 压出的 0.5 应判 indeterminate |

---

## 9. 禁止 claim（红线）

- **禁**用 primary same-shaft null 声称"覆盖超出 HFO 富集"（R2；只 HFO-rate null 授权）。
- **禁**把 latency nonsignificant 当 H-A（R3；H-A 是描述性 compatibility 非 p>0.05）。
- **禁**在 n=6 说"证明了 onset-synchronous 等效"（无功率做等效检验）。
- **禁**用 "off-axis" 指代 `¬A` 或 `S∖A` 而不严格限定（point 3）。
- **禁**结果前用"那圈 / peri-SOZ"预设空间结构（point 13；等 V3c-3 证）。
- **禁**把 narrow（n=3）当独立 inferential cohort（point 12）。
- **禁**pooled channel-level p（point 11）。
- **禁**直接写"医生漏掉真 SOZ / clinical SOZ 错了"；只能写 *may identify peri-SOZ tissue participating in early ictal recruitment*。

---

## 10. Outcome = future / blocked（审阅 point 14 收尾）

不在 V3c 内做 outcome。等 Yuquan resection/ablation masks + Engel/ILAE outcome 标签到手（E1 capstone，`docs/superpowers/specs/2026-06-13-yuquan-template-resection-outcome-design.md`，当前 blocked：标签不在 repo）后，再做真正的 capstone：**unresected early-surplus burden → poor outcome**——这才能支撑"clinical SOZ missed relevant tissue"。V3c 只在讨论里把它写作 future endpoint，不预设结论。

---

## 11. 复用与工程

### 11.1 复用地图（re-use don't re-invent；CLAUDE.md §6.1 question-match）

| 复用 | 来源 | 用途 | 契合检查 |
|---|---|---|---|
| `classify_subject_contacts(ds, "broad", cfg)` | `scripts/_topic5_v3_io.py` | `A`=is_axis、all_clean 池、`shaft_by_name` | 问题匹配：需要"轴成员 + 杆归属"，正是它的产物 ✓ |
| SOZ JSON | `results/{epilepsiae,yuquan}_soz_core_channels.json` | `S`（monopolar，与 cache 同名空间）| 已验证同名可 join ✓ |
| `bb_zt__{si}` / `bb_relt__{si}` + meta `eeg_onset_rel` | `results/topic5_ictal_recruitment/ictal_field_long_cache/{ds}.npz` | latency 源 | 已验证 §附录 A ✓ |
| `detect_contact_onset_zcross` / `baseline_robust_z` | `src/topic5_ictal_recruitment.py` | 首次超阈（V3c latency helper 薄封装）| null 契约：threshold-crossing，正是所需 ✓ |
| `seeg_coord_loader` | `src/seeg_coord_loader.py` | 距离 null / V3c-3 坐标 | epi+yuquan 双 loader ✓ |
| 同杆 shuffle / subject-level null 模式 | `src/topic5_v3_mode_transition.py`（V3a）| null 骨架 | **注意**：V3a null 是率保/相位/块，V3c 需**新的** same-shaft-axis-count-preserving + label + distance/rate-matched，不能直接套 ✗ → 新写 |

### 11.2 新文件（planned；细分交 writing-plans）

- `src/topic5_v3c_soz_coverage.py`（纯函数）：集合运算（coverage/surplus/jaccard）、same-shaft axis-count-preserving shuffle、label shuffle、distance/rate-matched null、`AUC_late`+`Δt`、censoring tallies、latency assay-QC 判定、空间聚集指标。
- 扩 `scripts/_topic5_v3_io.py` 或新 `scripts/_topic5_v3c_io.py`：`S` loader + 坐标 join + 从 cache 抽 per-contact latency（window/censoring 契约）。
- `scripts/run_topic5_v3c_coverage.py`（V3c-1）、`run_topic5_v3c_latency_qc.py`（label-blind，出 3 图）、`run_topic5_v3c_latency.py`（gated on QC，V3c-2）、`run_topic5_v3c_surplus_spatial.py`（V3c-3）、`run_topic5_v3c_summary.py`（cohort tier + claim 语言选择器）、`plot_topic5_v3c_*.py`。
- `tests/test_topic5_v3c_*.py`（纯函数 TDD：集合运算、null 保持量、AUC tie/censor 处理、QC 门、聚集指标）。
- config：扩 `config/topic5_v3.yaml` 加 `v3c:` 块（window=onset..+30s 主 / +20s & −2s buffer 敏感度、z_threshold=2.0 主 / 1.5&2.5 敏感度、assay-QC 门、n_perm=1000、cohorts broad 主/narrow 敏感度、never_pool）。

### 11.3 分支

在 `topic5-v2-phase1`（当前 HEAD `eac3fed`，已含 V3a io + ictal_recruitment）**off 一个隔离 worktree**（`using-git-worktrees`）。`classify_subject_contacts` 已在本树验证可跑。V3a 自己的 mode-transition 结果在未合并分支，但 V3c 只依赖**代码**底座（已 tracked），不依赖 V3a 结果产物。

### 11.4 产物目录（AGENTS.md Results 规范）

`results/topic5_ictal_recruitment/v3c_soz_axis_coverage/{broad,narrow}/`：per_subject JSON + `cohort_*.csv/json` + `figures/`（含中文 `README.md`：coverage forest / surplus 空间 / latency 3 图 / `S∖A` 图）。旧无并行目录问题（全新分析）。

---

## 12. TDD 优先承重不变量（交 writing-plans 展开）

按 CLAUDE.md §6：以下 plan-prose 不变量各需一条会失败的 TDD：
1. same-shaft null **保** `|A|` 与每杆轴触点数（打印 per-shaft 计数前后相等）。
2. `AUC_late` 的 tie 项 `0.5·P(=)` 正确、censored 排最后 / t0 排最前的秩规则。
3. coverage 双条件：条件 2 不满足时 summary **不得**输出"coverage 特异"。
4. assay-QC **label-blind**：QC 函数签名**不接受** `S`（编译期防泄漏）。
5. subject-key 对齐：coverage/latency/spatial 三处 subject 集合可不同，cohort 合并按 subject-key（`assert` 对齐，不按顺序）。
6. nested null：cohort-median null 每置换 per-subject 独立 shuffle（不是全队列一次 shuffle）。
7. narrow 结果**永不**并入 broad cohort 统计（分文件、分 verdict）。

---

## 附录 A：feasibility 探针数值（本会话，read-only）

### A.1 2×2 集合 join（broad 分类，n=9 SOZ-available 中 broad 可分类 7）

| 被试 | `|all|` | `|A|` | `|S∩pool|` | `|A∩S|` | `|A∖S|` surplus | `|S∖A|` | latElig |
|---|---:|---:|---:|---:|---:|---:|:--:|
| 139 | 41 | 20 | 4 | 4 | 16 | 0 | ✓ |
| 253 | 30 | 20 | 3 | 3 | 17 | 0 | ✓ |
| 635 | 57 | 17 | 10 | **7** | 10 | **3** | ✓ |
| 1077 | 121 | 20 | 4 | 4 | 16 | 0 | ✓ |
| 1096 | 75 | 20 | 6 | 6 | 14 | 0 | ✓ |
| 1150 | 124 | 20 | 3 | 3 | 17 | 0 | ✓ |
| 1146 | 114 | 20 | 14 | PENDING | PENDING | PENDING | ✓(待QC) |
| 442/958 | — | — | — | — | — | — | 无 broad cache → narrow only |
| 384/620/916/1125 | — | — | 无 SOZ | — | — | — | 掉出 |

> **1146 = PENDING（未在探针里跑 broad 分类）**：`|A|`=20 已确认（可 broad 分类），`|S∩pool|`=14 来自 SOZ JSON，但 `|A∩S|`/`|A∖S|`/`|S∖A|` 未实测。执行时由 V3c-1 coverage runner 的真实 join 回填（plan Task 16 Step 3 明确要求）；回填前此行数值不得写进任何结论。

### A.2 label-blind latency assay-QC（broad 轴触点，window onset..+30s，z=2.0）

| 被试 | nAx | nSz | finite% | t0% | cens% | uniqRk | maxTie | thrρ | szStab(s) | assay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 139 | 20 | 4 | 64 | 24 | 12 | 11 | 4 | 0.78 | 3.21 | ✓ |
| 253 | 20 | 6 | 62 | 12 | 25 | 11 | 2 | 0.68 | 9.05 | ✓ |
| 635 | 17 | 17 | 54 | 28 | 18 | 8 | 3 | 0.80 | 7.93 | ✓ |
| 1077 | 20 | 8 | 37 | **56** | 8 | 7 | 11 | 0.82 | 0.24 | **✗ t0** |
| 1096 | 20 | 8 | 56 | 15 | 29 | 10 | 4 | 0.84 | 6.31 | ✓ |
| 1150 | 20 | 8 | 41 | 14 | **44** | 7 | 2 | 0.72 | 7.39 | ⚠ cens-flag |

读法：uniq_ranks 7–11 + thrρ 0.68–0.84 → 排序非退化；1077 t0=56% 不过、1150 cens=44% 需敏感度同号。→ latency 作 gated secondary 预计 assay-valid ~5，印证决策 1=1b。

---

*(spec 结束；下一步 writing-plans 展开实现计划，含 TDD 任务分解与 subagent 分工)*
