# 患者多模态传播语法审计（40 人 masked adaptive-K）

日期：2026-08-15 · 层级：探索性方法学审计（非预注册假设检验）
产出目录：`results/interictal_propagation_masked/multimode_grammar_audit/`
输入：`results/interictal_propagation_masked/per_subject/*.json`（phantom-rank 已修版）+ 原始 lagPat 树重放

---

## 0. 朴素话摘要（先读这段）

我们手上有 40 名植入了颅内电极的癫痫患者。对每个患者，我们把他一整天里成千上万次的间期高频事件，
按"哪根电极先响、哪根后响"分门别类。以前的自动分类程序对 34 个人分出了两类，对 6 个人分出了三到六类。
这次的问题是：那多出来的三到六类，到底是不是"这个人的大脑真的有三到六种不同的传播方式"。

**怎么测的。** 三步。第一步，看看"分出更多类"这件事，跟什么东西一起变。如果它跟大脑有关，
应该跟记录时长、事件多少这些有关；如果它跟仪器有关，应该跟插了几根电极、能看到几个触点有关。
第二步，把每一类的"平均先后顺序"画出来，看这几类彼此差在哪：是差在**方向**（谁先谁后整个反过来），
还是差在**范围**（一次拉进来几个触点）。第三步，看时间上：如果真有个"开关"在几种传播方式之间切换，
那么相邻两次事件用同一类的比例，应该明显偏离"每次都独立地按各类比例抽一次签"的预期。

**看到了什么。** 三件事都指向同一个方向。

一，多出来的类只出现在**电极最少**的人身上。6 个多类患者的可用触点数是 4、4、4、5、5、6；
34 个两类患者是 6 到 52。触点数和类数的相关是 −0.61（P=3×10⁻⁵），而事件数（P=0.45）和记录段数（P=0.95）
跟类数**测不出任何关系**。所有触点数 ≤ 5 的患者（5 人）无一例外都被分成了多类。

二，只有 4 个触点时，"哪个先哪个后"这件事总共只有 **36 种**可能的取法——我们数过，这三名 4 触点患者
把这 36 种**全部占满了**。所以"分成六类"实际上是把 36 个点切成 6 堆，每堆 4–8 个点。
把每一类的原型画出来看，它们就是在轮流枚举"哪个触点最先响 / 哪个最后响"：比如 zhangjinhan 的五类，
恰好就是"F5 最后、F4 最后、F6 最后、G1 最后、G2 最后"这五种。

三，如果完全没有记忆——每次事件独立地按各类占比抽签——相邻两次换类的比例应该等于某个数。
实测跟这个数的差距，六个人分别是 −1.7% 到 +0.7%，34 个两类患者是 −6.3% 到 +1.4%，两组**分不开**（P=0.31）。
单看每一格的转移，全队列最大的一处偏离是 818 的第 4 类"连着再来一次"，比无记忆预期高 4.6 个百分点。
也就是说：**没看到"刚才走了这条路，这次更容易再走这条路"的开关行为。**

还有一件必须说的。有人可能会说"那些多出来的类是不是自然凑成两个方向相反的大家族"。
我们试了：把 K 个原型按所有可能的方式切成两组，取"组内像、组间反"最明显的那一种。
六个人都能切出来。但我们把每一类的触点顺序**随机打乱**再跑同一个流程，发现随机情况下这个流程
有 98.4%–100% 的抽样**照样**给出"两个方向相反的家族"——因为在 4 个触点上，两个随机顺序之间的
相关系数标准差就有 0.58，正负 0.8 完全是常态。实测值全部落在随机分布内部（P = 0.13–0.40）。
所以"两个方向超家族"这个读法**站不住**。

**裁定。** 用户预设的两条路——"特定方向通路"和"通用模式选择器"——**都不满足**，
结论锁定为 `PATIENT_MULTIMODE_MECHANISM_UNRESOLVED`。同时给出一条正面结论：
这 6 个人的"多模式"最兼容的解释是**仪器侧的伪多模态**，但驱动变量不是用户假设的记录段或事件数，
而是**触点数 / 可分辨顺序空间的大小**。

（内部归档代号：`adaptive_cluster.chosen_k`、`stable_k`、`build_masked_kmeans_features`、
`lattice saturation`、`within-block occupancy-preserving permutation null`、`random-ordering superfamily null`、
`silhouette-vs-k slope`、`min_cluster_fraction` gate。以下正文按精度口径书写。）

---

## 1. 一句话裁定

**`PATIENT_MULTIMODE_MECHANISM_UNRESOLVED`** —— 6 名 `chosen_k > 2` 患者的额外模式既不构成稳定的
相反方向超家族（random-ordering null 下 P = 0.13–0.40，且该 null 在 98.4–100% 抽样中本就给出"两个相反家族"），
也无法建立"方向相近但占用/招募不同"这一读法所要求的"方向相近"前提（4–6 触点下 prototype Spearman ρ
在随机顺序下的 sd = 0.45–0.58）；同时 `chosen_k` 与 `n_channels` 强相关（ρ = −0.613，P = 2.6×10⁻⁵）
而与 `n_valid_events`（P = 0.45）、`n_blocks_used`（P = 0.95）无关，
最兼容的解释是**由触点数 / masked 特征格点基数驱动的仪器侧伪多模态**——这是对用户候选解释 4 的修正版
（驱动变量是通道数，不是 recording block 或事件数量）。

---

## 2. K 分布与数据完整性

### 2.1 K 分布（与数据合同**完全一致**，未沿用假设值）

| K | n | 患者 |
|---|---|---|
| 2 | 34 | （其余全部） |
| 3 | 1 | `yuquan_huangwanling` |
| 4 | 2 | `epilepsiae_818`、`epilepsiae_916` |
| 5 | 2 | `yuquan_zhangjinhan`、`yuquan_zhourongxuan` |
| 6 | 1 | `yuquan_zhaojinrui` |

全部 40 名 `chosen_reason == "stable_k"`，`k_range == (2, 8)`。本轮**未重新挑 K**；
`adaptive_cluster.scan` 仅作只读汇报（§2.4）。

### 2.2 必做工程审计（6 项，40/40 通过，0 例排除）

`engineering_audit.json::all_checks_passed == true`，`exclusions == []`。逐项：

| # | 检查 | 实现 | 结果 |
|---|---|---|---|
| 1 | `len(labels) == n_valid_events` | 从原始 lagPat 树重放 `_valid_event_indices(bools, min_participating=3)`，三者互等 | 40/40 |
| 2 | labels 与 block boundaries 顺序一致 | `block_boundaries` 连续且覆盖全部事件；loader `block_ids` 与之逐元素相等；`event_abs_times` 在每个 block 内非降；`valid_events` 严格升序 | 40/40 |
| 3 | 不计算跨 block 转移 | `_within_block_adjacent_pairs(block_ids)` 为**必需参数**（无 `=None` 默认）；observed 与 null 均只在 block 内累计。剔除的跨段相邻对：11–434 对/人 | 40/40 |
| 4 | channel ordering + 每 cluster valid mask | loader `channel_names` 与 JSON `channel_names` 断言相等；prototype 用 `mask_phantom_ranks` 后按 cluster 从原始 `eventsBool` 求参与计数，未参与通道为 NaN 且不进入任何 pairwise 统计。`adaptive_cluster.clusters[*].template_rank` **仅作 provenance 携带**，不参与计算（cross-PR 合同：其 `_legacy_hist_mean_rank` fallback 会给未参与通道赋秩） | 40/40 |
| 5 | K 冻结 | `chosen_k` 从 artifact 读取；断言 `unique(labels) == arange(chosen_k)` | 40/40 |
| 6 | commit / hash / seed / 排除理由 | 见 §2.3 | 已记录 |

> **数据合同遵守**：只使用 `results/interictal_propagation_masked/`（phantom-rank 已修）。
> 所有 rank 量经 `src.lagpat_rank_audit.mask_phantom_ranks` / `build_masked_kmeans_features(impute="event_median")`，
> 与产出端参数一致（`min_shared_channels=3`）。未触碰任何未修复的旧结果目录。

### 2.3 复现信息

- `git_commit`（分析执行时）：`d4bd26778f1e462b6004e11dfe5a7022a102cc25`，branch `codex/topic4-rev10-sa-shaft-aware`
- **⚠️ 并发提交提示**：本会话期间该分支 HEAD 被**其他进程**推进了三次（`7046f8df` → `d4bd267` → `a6dcbdd9`）。
  已核验：`src/interictal_propagation.py`、`src/lagpat_rank_audit.py`、`scripts/run_interictal_propagation.py`、
  `results/interictal_propagation_masked/per_subject/` 在 `7046f8df..a6dcbdd9` 区间内**无改动**且工作区干净，
  因此本分析的输入未被并发提交污染。
- 环境：Python 3.11.5 / numpy 1.26.4 / scipy 1.13.1
- 随机种子：`SEED = 20260815`，每名患者用 `default_rng(SEED + i)`（i 为字典序下标），
  deep dive 用 `SEED+1000+i` / `SEED+5000+i`，addendum 用 `SEED+9000+i`
- 置换/重采样次数：block 内 occupancy-preserving permutation **4096**；global label permutation **4096**；
  recording-block bootstrap **2000**；random-ordering superfamily null **2000**；MI 洗牌地板 **200**
- 每名患者记录 `input_json_sha256`、`raw_ranks_bools_sha256`、`labels_sha256`（见 `engineering_audit.json`）
- **排除理由**：无。40/40 全部进入分析，无任何 subject / event / block 被剔除，
  唯一被系统性排除的是**跨 recording block 的相邻事件对**（这是审计条款 3 要求的，逐人计数已记录）

### 2.4 替代 K（只读汇报，未重新挑 K）

| 患者 | K | 通过双闸门的 k | k=2 silhouette | K 处 silhouette | 增益 | k+1 被哪个闸门拦下 |
|---|---|---|---|---|---|---|
| huangwanling | 3 | 2, 3 | 0.422 | 0.424 | **+0.002** | assignment_stability |
| 818 | 4 | **仅 4** | 0.327 | 0.423 | +0.095 | min_cluster_fraction + stability |
| 916 | 4 | 2, 3, 4 | 0.335 | 0.353 | +0.018 | min_cluster_fraction |
| zhangjinhan | 5 | 2, 3, 4, 5 | 0.285 | 0.336 | +0.051 | min_cluster_fraction + stability |
| zhourongxuan | 5 | 2, 3, 4, 5 | 0.444 | 0.554 | +0.109 | min_cluster_fraction |
| zhaojinrui | 6 | 2, 3, 4, 5, 6 | 0.414 | 0.525 | +0.112 | min_cluster_fraction |

5/6 患者 k=2 同样合法；huangwanling 的 K=3 相对 K=2 只多 **0.002** silhouette。
**只有 818 的 k=2/k=3 真正不合法**（AMI 0.464 / 0.604 < 0.70）。

---

## 3. 选择器机制：为什么恰好是这 6 个人

产出：`k_selection_mechanism.json`。产出端规则是 `chosen_k = argmax(median_silhouette)`（在通过双闸门的 k 中）。
这条规则的行为完全取决于 silhouette 随 k 的走向：

- **silhouette 对 k 的斜率**：K>2 组 **6/6 为正**（+0.010 … +0.050，中位 +0.039）；
  K=2 组 **仅 5/34 为正**（中位 −0.007）。Mann-Whitney（K>2 更大）**P = 5.2×10⁻⁷**。
  斜率 vs `n_channels`：ρ = **−0.694**，P = 6.7×10⁻⁷。
- **`chosen_k` 是否等于最大可行 k**：K>2 组 **6/6 是**；K=2 组 **1/34 是**。
- **`chosen_k` 处的最小簇占比**：K>2 组 0.108–0.198（紧贴 0.10 闸门）；K=2 组 0.276–0.497（离闸门很远）。

机制解读：当触点很少时，masked 特征向量塌缩到少数**完全重合**的格点上，簇内距离含大量精确 0，
于是把格点切得越细 silhouette 越高，K 被一路推高直到撞上 `min_cluster_fraction ≥ 0.10`（5/6）或稳定性闸门（1/6）。
当触点多时，特征空间稠密，细分反而降低 silhouette，argmax 自然落在 k=2。
**因此这 6 个人的 K 是被"上升的 silhouette 曲线 + 10% 最小簇闸门"共同定出来的位置，不是簇质量的真实峰值。**

> 这与记忆中已锁定的 `feedback_silhouette_threshold_high_dim`（silhouette 绝对阈值不可迁移）是同一课的镜像：
> 那次是高维下 silhouette 被压低，这次是**低维强并列下 silhouette 被抬高且随 k 单调上升**。

---

## 4. 六名 K>2 患者逐例结果

术语：`占比` = mode occupancy；`换类率` = block 内相邻事件换 mode 的比例；
`超额` = 实测换类率 − block 内 occupancy-preserving permutation null 均值（4096 次）；
`格点` = masked 特征向量的不同取值数 / 理论可达上限；`招募` = 该 mode 事件参与触点数中位数 / 总触点数。

### 4.1 `yuquan_huangwanling` — K=3，4 触点，**单杆 H2–H5**，55 521 事件 / 12 段

- 占比 0.487 / 0.267 / 0.246，归一化熵 0.954
- 换类率 0.6124，null 0.6269，**超额 −0.0145**（z=−7.5，P=4.9×10⁻⁴）
- **格点 36/36 = 1.00（占满）**，每 mode 11–13 个不同顺序
- 三个 mode 招募**完全相同**（均为 3/4 触点）→ 差异全在顺序，不在范围
- pairwise ρ ∈ [−1.00, +0.40]，1/3 对 < −0.5；maxPartDiff 0.220–0.536
- 端点：3/3 有各自不同的最先触点，2/3 有不同的最后触点，3/3 端点对互异
- superfamily：实测分离 1.100 vs 随机顺序 null 均值 0.806，**P = 0.29**
- 12/12 段全部同时表达 3 个 mode；mode×block Cramér's V = 0.085（null 0.014）
- **⚠️ K=3 相对 K=2 的 silhouette 增益仅 +0.002**

### 4.2 `epilepsiae_818` — K=4，5 触点，3 杆（TBA/TBB/TBC，**无一杆含 ≥3 触点**），11 337 事件 / 222 段

- 占比 0.495 / 0.190 / 0.169 / 0.146，归一化熵 0.898（六人中最低）
- 换类率 0.6044，null 0.6166，**超额 −0.0122**（z=−3.2，P=2.9×10⁻³）
- 格点 234/260 = 0.90，每 mode 49–76 个不同顺序
- 四个 mode 招募**完全相同**（均 3/5）；maxPartDiff 0.379–0.471；最大 shaft 参与差 0.471
- pairwise ρ ∈ [−0.90, +0.60]，2/6 对 < −0.5
- 端点：4/4 最先触点互异（TBB3 / TBC6 / TBC5 / TBB4），端点对 4/4 互异
- superfamily：1.033 vs null 0.706，**P = 0.13**（六人中最接近，但仍不显著）
- 91.0% 段表达 ≥2 个 mode，58.6% 段表达全部 4 个；Cramér's V = 0.303（null 0.140，六人中最高）
- **单格最大记忆效应出现在这里**：mode 4 自转移 0.306 vs block 内 null 0.260，**+0.046**
  （⚠️ 不可与该 mode 0.146 的全局占比对比——那 0.16 的落差大部分是跨段成分差异造成的假象）
- **⚠️ 无任何一杆含 ≥3 触点 → 无法定义杆内传播轴**，axial ρ 全部为 `None`
- **⚠️ 唯一 k=2/k=3 不合法的患者**（AMI 0.464 / 0.604）

### 4.3 `epilepsiae_916` — K=4，6 触点，2 杆（AM1–2 / AH1–4），93 204 事件 / 435 段

- 占比 0.359 / 0.272 / 0.215 / 0.155，归一化熵 0.967
- 换类率 0.7120，null 0.7048，**超额 +0.0072**（z=+5.2，P=4.9×10⁻⁴）——六人中唯一为正
- 格点 1209/1830 = 0.66，每 mode 215–360 个不同顺序（**六人中最不退化**）
- **招募差异真实存在**：mode 2 中位招募 5/6，其余三个 3/6；对 mode 1–2 而言
  maxPartDiff **0.810**、最大 shaft 参与差 **0.784**（AM 杆在 mode 2 中被大量拉入）
- pairwise ρ ∈ [−0.66, +0.66]，2/6 对 < −0.5
- axial ρ（AH 杆）：+1.0 / +1.0 / −0.8 / +0.8
- superfamily：0.667 vs null 0.625，**P = 0.40**（六人中最不显著）
- 97.2% 段 ≥2 个 mode，88.0% 段全 4 个；Cramér's V = 0.180（null 0.068）
- 单格最大偏离仅 0.012

### 4.4 `yuquan_zhangjinhan` — K=5，5 触点，2 杆（F4–F6 / G1–G2），6 156 事件 / 13 段

- 占比 0.265 / 0.225 / 0.208 / 0.182 / 0.120，归一化熵 0.981
- 换类率 0.7767，null 0.7805，**超额 −0.0039**（z=−0.8，**P = 0.47，未偏离无记忆**）
- 格点 254/260 = 0.98，每 mode 50–52 个不同顺序
- 招募 3/5、3/5、4/5、4/5、4/5（两档）；maxPartDiff 0.067–0.529
- **五个 mode 恰好枚举"哪个触点最后放电"**：F5 / F4 / F6 / G1 / G2 各占一个
  （`n_distinct_sink_contacts = 5/5`，而 `n_distinct_source_contacts` 仅 2/5）
- pairwise ρ ∈ [−0.70, +0.90]，3/10 对 < −0.5
- superfamily：0.883 vs null 0.701，**P = 0.19**
- 13/13 段全部表达 5 个 mode；Cramér's V = 0.109（null 0.044）
- **⚠️ 事件数最少（6 156）**

### 4.5 `yuquan_zhourongxuan` — K=5，4 触点，**单杆 G7–G10**，12 238 事件 / 12 段

- 占比 0.277 / 0.257 / 0.196 / 0.155 / 0.114，归一化熵 0.970
- 换类率 0.7758，null 0.7809，**超额 −0.0051**（z=−1.4，**P = 0.16，未偏离无记忆**）
- **格点 36/36 = 1.00（占满）**，每 mode 仅 4–9 个不同顺序
- 招募：mode 4 为 4/4，其余 3/4；对含 mode 4 的配对 maxPartDiff 0.754–0.812
- pairwise ρ ∈ [−1.00, +0.80]，4/10 对 < −0.5
- 端点：4/5 最先触点互异，端点对 5/5 互异
- superfamily：0.917 vs null 0.839，**P = 0.35**
- 12/12 段全部表达 5 个 mode；Cramér's V = 0.038（null 0.030，几乎无 block 结构）
- **⚠️ Path-D legacy-variant 患者**（`YUQUAN_LEGACY_VARIANT_SUBJECTS`，仅有旧 `_lagPat.npz`，
  打包脚本与 `_lagPat_withFreqCent` 主线不同）

### 4.6 `yuquan_zhaojinrui` — K=6，4 触点，**单杆 F5–F8**，46 855 事件 / 13 段

- 占比 0.200 / 0.177 / 0.175 / 0.166 / 0.152 / 0.130，归一化熵 **0.995**（近乎完全均匀）
- 换类率 0.8102，null 0.8269，**超额 −0.0167**（z=−9.6，P=4.9×10⁻⁴；六人中绝对值最大，仍仅 1.7 个百分点）
- **格点 36/36 = 1.00（占满）**，每 mode 仅 **4–8** 个不同顺序 → K=6 就是把 36 个格点切成 6 堆
- 招募：mode 5、6 为 4/4，mode 1–4 为 3/4；maxPartDiff 0.061–0.616
- pairwise ρ ∈ [−0.80, +0.80]，5/15 对 < −0.5
- 端点对 **6/6 互异**（占满 4 触点可表达的 12 种端点对的一半）
- superfamily：0.867 vs null 0.839，**P = 0.39**
- 13/13 段全部表达 6 个 mode；Cramér's V = 0.068（null 0.016）
- **⚠️ Path-D legacy-variant 患者**

---

## 5. 四轴分解：direction / extent / occupancy / switching

### 5.1 Direction（传播方向）

**没有可辨的方向家族结构。**

- superfamily 二分最大化在六人身上都能给出 "组内正、组间负"，但这是**最大化器的必然产物**：
  在 random-ordering null 下，同一流程有 **98.4%–100%** 的抽样也给出"两个相反家族"。
  实测分离度全部落在 null 内部（P = 0.29 / 0.13 / 0.40 / 0.19 / 0.35 / 0.39）。
- 产出端 `candidate_forward_reverse` 判据（`spearman_r < −0.5`）在这些阵列上**不具判别力**：
  实测满足该条件的配对数 1–5 个，random-ordering null 期望 0.62–3.20 个，全部 P = 0.11–0.49。
- 根本原因：m 个触点上两个随机顺序的 Spearman ρ 的 sd = 1/√(m−1)。
  m=4 → 0.577，m=5 → 0.500，m=6 → 0.447。**4 触点时随机顺序的 95% 区间覆盖整条 ρ 轴。**
- block bootstrap 显示二分方案在重采样下高度稳定（观测切分复现率 0.59–1.00 vs 机会率 0.03–0.33），
  但**这只度量原型估计的精度，不度量结构是否存在**——必须与 random-ordering null 一起读。

### 5.2 Extent / 参与结构（招募范围）

**这一轴上有真实差异，且强于方向轴。**

- 六人中 **4 人**（zhaojinrui、zhourongxuan、zhangjinhan、916）存在中位招募档位不同的 mode；
  huangwanling 与 818 的所有 mode 招募中位数完全相同。
- 逐触点参与率差（`max_participation_rate_diff`）最大值：0.536 / 0.471 / **0.810** / 0.529 / **0.812** / 0.616。
- 可解析杆的三人中，最大 shaft 参与差：818 = 0.471、916 = **0.784**、zhangjinhan = 0.293
  （另三人为单杆，该量不适用）。
- **精确信息分解**（label 是 masked 特征向量的确定性函数，故
  `H(mode) = I(mode; 参与集合) + I(mode; 集合内顺序)`，为精确恒等式）：
  六人的"由参与集合解释"份额（洗牌地板校正后）= 19.3% / 25.6% / 33.1% / 18.9% / 14.2% / 13.2%，
  其余 66.9%–86.8% 由"集合内顺序"解释。
  ⚠️ 该分解只在**参与集合空间远小于事件数**时可信；六名 K>2 患者的 MI 洗牌地板均 ≤ 0.31% × H(mode)，
  故可信。相反，高通道数 K=2 患者的地板高达 80–100% × H(mode)（`songzishuo` 38 触点 = 100.0%），
  这些患者的原始"参与集合解释份额"是插值 MI 的**纯偏差**，已在 `cohort_addendum.json` 中给出校正值与地板。
- 端点枚举：六人的 mode 均**一一对应到互异的（最先触点，最后触点）对**（3/3、4/4、4/4、5/5、5/5、6/6）。
  这些 mode 占该阵列可表达端点对总数的 25% / 20% / 13% / 25% / 42% / 50%。
  `modes_per_expressible_endpoint_pair` 与 K 相关 ρ = +0.621，P = 1.9×10⁻⁵。

### 5.3 Occupancy（占用）

- 归一化熵 0.898–0.995，均接近均匀。
- 与 K=2 组（0.851–1.000）**无差异**（中位 0.969 vs 0.974，Mann-Whitney P = 0.64）。
- mode 进入概率与该 mode 的整体占比几乎逐位相等
  （zhaojinrui：entry = [0.195, 0.173, 0.170, 0.163, 0.148, 0.126] vs occupancy = [0.200, 0.177, 0.175, 0.166, 0.152, 0.130]）
  ——这正是无记忆抽签的表现。
- 每段是否都表达多个 mode：huangwanling / zhangjinhan / zhourongxuan / zhaojinrui **100% 的记录段表达全部 mode**；
  916 为 97.2%（全部 mode 88.0%）；818 为 91.0%（全部 mode 58.6%）。

### 5.4 Switching（时间切换）

**没有找到模式选择器行为。**

- 超额换类率：−0.0167 / −0.0122 / +0.0072 / −0.0039 / −0.0051 / −0.0145，
  即最多偏离无记忆预期 1.7 个百分点。P 值中三例达 4096 次置换的下限（4.9×10⁻⁴），
  但那是 n = 10⁴–10⁵ 事件下的必然结果，**效应量才是可读的量**。
- 与 K=2 组无差异：|超额| 中位 0.0097 vs 0.0142，Mann-Whitney **P = 0.31**。
- 逐格转移偏离（实测 − block 内无记忆期望）：全队列单格最大 **+0.046**（818 的 mode 4 自转移），
  其余五人 ≤ 0.031。
- ⚠️ 实测换类率本身在 K>2 组显著更高（0.744 vs 0.454，P = 5.2×10⁻⁷），但这是**纯机械效应**：
  均匀占用下的无记忆换类率 = 1 − 1/K。不可读作发现。
- mode×block 关联（Cramér's V）在 K>2 组**并不更高**（中位 0.097 vs 0.186，P = 0.17），
  这直接**否定**了"recording block 造成伪多模态"这一字面读法。
  ⚠️ 该检验的 P 值在 n = 10⁴–10⁵ 下近乎退化，只能读效应量。

---

## 6. 哪种假设更兼容

按用户预设的判别逻辑逐条判：

**"特定方向 pattern"** —— **不成立**

| 判据 | 结果 |
|---|---|
| K>2 原型主要落入两个稳定的相反方向超家族 | ❌ random-ordering null 下 P = 0.13–0.40；该 null 本身 98.4–100% 抽样即产生"两个相反家族" |
| 新增模式只是同一方向的细分 | ❌ 各 mode 的端点分布在不同触点上（端点对 100% 互异），axial ρ 在 −1…+1 间散开 |
| occupancy 变化主要偏向固定方向家族 | ⛔ 本轮为静态设计，无 occupancy 变化可测，**未检验** |

**"通用模式选择 / 可达性"** —— **不成立（关键前提缺失）**

| 判据 | 结果 |
|---|---|
| 多个 mode 方向相近但 occupancy / 招募范围 / 参与结构明显不同 | ⚠️ **后半句成立**（§5.2：参与率差达 0.81，shaft 参与差达 0.78，招募档位分层）；**前半句无法建立**——4–6 触点下"方向相近"与"方向相反"在统计上不可分辨 |
| mode number / entropy 与方向极性无固定对应 | ⚠️ 成立，但属**退化式成立**：根本不存在可辨的方向极性结构 |
| 模式转换不能由一个二元方向轴解释 | ⚠️ 同上，退化式成立；且转换本身已接近无记忆（§5.4），"转换"没有可解释的结构 |

**→ 两者都不满足，裁定 `PATIENT_MULTIMODE_MECHANISM_UNRESOLVED`。**

**同时给出的正面结论（对候选解释 4 的修正）：** 用户候选 4 的字面版本
（"recording block 或事件数量造成的伪多模态"）**被否定**——K 与 `n_blocks_used`（P = 0.95）、
`n_valid_events`（P = 0.45）均无关，且 K>2 组的 mode×block 关联反而更低（P = 0.17）。
但一个**近亲解释成立**：伪多模态由**触点数 / masked 特征格点基数**驱动。
支持证据链：K vs `n_channels` ρ = −0.613（P = 2.6×10⁻⁵）；
所有 `n_channels ≤ 5` 的患者（5/5）都是 K>2，K>2 组 `n_channels` 上限为 6；
三名 4 触点患者的格点占满率 = 1.00（36/36），每 mode 仅剩 4–13 个不同顺序；
silhouette-vs-k 斜率 K>2 组 6/6 为正、K=2 组 5/34 为正（P = 5.2×10⁻⁷），
且 `chosen_k` 在 K>2 组 6/6 等于最大可行 k、在 K=2 组仅 1/34（§3）。

**在这条修正解释下，"通用模式选择器"与"特定方向通路"两个机制假说都没有得到患者侧支持，
也都没有被排除**——数据的分辨率不足以区分它们。

---

## 7. 替代解释与局限

1. **本审计不能证明这些大脑只有两种传播模式。** 结论是"当前数据无法支撑多于两种"，不是"多于两种不存在"。
   同一批患者若有更大覆盖的植入，可能显示真实的多模式结构。
2. **模式并非均匀格点分箱。** 每个 mode 的最高频顺序占该 mode 事件的 20.2%–47.3%，
   相对"在该 mode 自身格点上均匀"的参考高出 **3.1× – 86.2×**。
   ⚠️ **该量未做零假设**——无法据此区分"真实模板"与"KMeans 在局部最密格点上切出的分箱"。
   这是本审计**最主要的未闭合缺口**；补法是在冻结 K 下对打乱数据重跑同一 KMeans 作对照
   （本轮受"不得重新挑 K"约束未做，且重跑 KMeans 会引入新的一层需要论证的设计）。

   > **【2026-08-16 已闭合】** 该零假设已建成并运行：保住每个事件的招募掩码、事件数与记录段，
   > 只把参与触点之间的先后顺序随机化，再用与产出端逐字相同的聚类在冻结 K 上重跑（每人 512 次，
   > 6/6 患者上该复刻重现冻结标签 AMI=1.000）。结论：**4/6 患者的集中度确实超过这个零假设**
   > （huangwanling +0.055、818 +0.127、916 +0.087、zhaojinrui +0.037，均 P=0.0019 即 512 次抽样下限），
   > zhangjinhan 恰好为 0（−0.0003，P=0.45），zhourongxuan **低于**零假设（−0.073）。
   > 随后又加了第二层零假设（穷举 p! 后按最大熵匹配每个触点的秩边缘，即连"这个触点整体偏早还是偏晚"
   > 也一并保住、只打掉事件内跨触点协同）：**6/6 患者都不再超出**
   > （构造检验合格的 5 人中 0/5 超出）。
   > 即：本条缺口的答案是——模式确实抓住了真实的顺序不均匀性（第一层 4/6 跨过），
   > 但那份不均匀性**完全来自单个触点的早/晚偏好**，不是来自触点之间的协同（第二层 0/5）。
   > 因此**不能**把这些模式称作"传播模板"。本报告 §5–§6 关于
   > "K 跟着触点数走""模式在枚举端点对""没有方向超家族"的结论**不受影响**。
   > 详见 [`multimode_selection_null_and_916_extent_2026-08-16.md`](multimode_selection_null_and_916_extent_2026-08-16.md)。
3. **`epilepsiae_916` 的招募/参与差异是真实的**，且不能用格点退化解释（格点占满率仅 0.66，每 mode 215–360 个顺序）。
   它是六人中唯一在"extent"轴上有非退化结构的患者。
4. **`epilepsiae_818` 的 K=4 是六人中唯一"不可退回 k=2"的**（k=2/k=3 均未过稳定性闸门）。
   但它 5 个触点分在 3 根杆上、无一杆含 ≥3 触点，**无法定义任何杆内传播轴**。
5. **random-ordering null 只随机化顺序，不随机化参与掩码。** 它检验"方向家族结构"，
   不检验"参与集合本身是否有结构"。参与集合的结构性未被此 null 覆盖。
6. **block bootstrap CI 在 Δ招募比例上退化**（该量量化到 1/n_channels 的整数倍，CI 常塌成单点）。
   连续量 `max_participation_rate_diff` 是更可靠的招募差指标。
7. **K>2 组 n = 6**，所有涉及该组的组间对比（换类率、熵、Cramér's V）**功效很低**，
   "无差异"应读作"测不出差异"。相反，K vs `n_channels` 用全部 40 人，功效较好。
8. **两名 K>2 患者（`zhaojinrui`、`zhourongxuan`）属 Path-D legacy-variant**，
   使用旧 `_lagPat.npz` 与不同打包参数（`pickChn_thresh` / `packWinLen`），
   见 `cohort_slice_a2_legacy_variant_2026-05-07.md`。它们同时是格点占满的 4 触点患者。
9. **mode×block Cramér's V 的 P 值在 10⁴–10⁵ 事件下近乎退化**，只有效应量与 null 均值之差可读。
10. **本审计层级为探索性方法学审计**，非预注册假设检验。上述任何一条都不足以进入 topic 主文档的正式口径，
    需先补齐第 2 条的缺口。

---

## 8. 是否值得为 `epilepsiae_916` / `yuquan_zhaojinrui` 建 K=4 / K=6 的 subject-specific SNN

**`yuquan_zhaojinrui`（K=6）：不建议。** 理由：4 个触点全在同一根 F 杆上，
masked 特征格点 36/36 占满，每个 mode 只剩 4–8 个不同顺序，K=6 完全由
"silhouette 随 k 单调上升 + 撞上 10% 最小簇闸门"定出（§3），且 k=2…6 全部合法。
**不存在一个"六种传播语法"的拟合目标**；对它建 K=6 模型等于让模型去拟合一个组合学产物。
叠加 Path-D legacy-variant 的数据谱系风险。

**`epilepsiae_916`（K=4）：不建议按"K=4 四个方向"建，但它是六人中唯一值得动的。** 理由两面：
- 正面：6 触点 / 2 杆，格点占满率 0.66，每 mode 215–360 个不同顺序，是六人中最不退化的；
  且它有**真实且大的招募/参与对比**——mode 2 中位招募 5/6（其余 3/6），
  逐触点参与率差 0.810，AM 杆参与差 0.784。
- 反面：它的 K=4 同样是最大可行 k（k=5 被 10% 闸门拦下），k=2 与 k=3 都合法，
  且 K=4 相对 k=2 的 silhouette 增益仅 **+0.018**；四个原型之间无方向家族结构（P = 0.40，六人中最不显著）。

**建议的替代做法**（如果这条线要继续）：对 `epilepsiae_916` 建**两模式** SNN，
把拟合目标定为**"同一方向下的招募范围 / 跨杆参与差"**（mode 2 拉入 AM 杆、招募 5/6 vs 其余 3/6），
而不是"四个方向"。这恰好对应当前 SNN 里 E→I redistribution 的表型（招募范围扩大），
而且是患者侧唯一在 extent 轴上非退化的观测。

**若一定要一个 K>2 锚点**：`epilepsiae_818` 是唯一 k=2/k=3 均不合法的患者，
但它 5 触点分 3 杆、无一杆 ≥3 触点，无法定义空间轴，对空间组织化的 SNN 是差目标。
两害相权，**本轮结论是：不启动新的 K=4 / K=6 subject-specific SNN 长跑。**

---

## 9. 与 E→I 的关系（必须明确写出）

**患者 SEEG 本身不能识别 E→I 因果机制。**

SEEG 触点记录的是局部场电位的宏观混合信号；兴奋性→抑制性突触权重重分配（E→I redistribution）
与兴奋性→兴奋性重分配（EE redistribution）以及任何其它能改变有效连接的扰动，
在触点级的 rank / 招募观测上**不可区分**。本审计观察到的一切——mode 占用、招募范围、
参与结构、转移统计——都是**表型层**的量，可以与多种底层机制兼容。

因此本文的任何结论都**不构成对 E→I 的支持或否证**。本审计只能做到：
在患者侧寻找与"通用模式选择器"或"特定方向通路"**一致**的证据；本轮**两者都没找到**，
且找到了一个更简单的仪器侧解释（触点数 / 特征格点基数）。
SNN 中"E→I redistribution 把 Mode 2 占比从约 66% 推到 74%"这类陈述，
其因果归属只在模型内部成立，**不得**借本审计向患者数据外推。

---

## 附：产出清单

```
results/interictal_propagation_masked/multimode_grammar_audit/
├── run_multimode_grammar_audit.py       主分析（40 人，合同条款 C1–C9 逐条断言）
├── verify_bootstrap_equivalence.py      block 充分统计量 bootstrap 与直接重算的等价性验证
├── run_kgt2_deep_dive.py                6 名 K>2 患者：superfamily 稳定性 + random-ordering null + 冻结 k-scan
├── run_cohort_addendum.py               端点枚举 + MI 偏差地板 + 组间对比 + mode 内顺序集中度
├── run_k_selection_mechanism.py         冻结 k-scan 的只读机制汇总（silhouette 斜率 / 闸门）
├── plot_multimode_grammar_audit.py      全部图（PNG/PDF/SVG）
├── engineering_audit.json               C1–C6 逐人检查 + hash + 排除记录（0 例）
├── cohort_summary.{csv,json}            40 人一行
├── mode_pairs.csv                       84 对 mode pair
├── kgt2_deep_dive.json
├── cohort_addendum.json
├── k_selection_mechanism.json
├── per_subject/<sid>.json               40 份完整载荷
├── run.log
└── figures/                             README.md + 4 个独立 panel + 完整拼版 + 2 张补充图
```

`verify_bootstrap_equivalence.py` 在 4 名患者 × 40 次随机 block 重采样上通过，
原型最大偏差 1.1×10⁻¹³（浮点噪声量级）。
