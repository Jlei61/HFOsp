# Topic 5 A 线上游 gate · TA/TB 间期传播场反向门（field-reversal gate）

> 日期 2026-07-06 · Topic 5 network-axis（A 线）上游补充 gate
> **状态：`n_perm=1000` 正式全量跑批已完成（2026-07-07），本文档数值即正式全量结果；用户签核完成前，不作为 `docs/topic5_seizure_subtyping.md` 的正式接受结论（CLAUDE.md §5）——主文档暂留 preliminary 指针。与 preliminary `n_perm=300` 的唯一实质差异：`broad` 过门 7/26→6/26（一个边际被试掉门），`narrow` 9/26 不变。**
> 上游：`docs/archive/topic5/axis_alignment_AB_result_2026-06-14.md`（A 线主统计：间期传播轴 ↔ 发作早期激活的轴对齐）
> 设计 spec：`docs/superpowers/specs/2026-07-06-topic5-tatb-field-reversal-design.md`
> 实施 plan：`docs/superpowers/plans/2026-07-06-topic5-tatb-field-reversal.md`
> 代码：`src/topic5_field_reversal.py`（统计核心）+ `scripts/run_topic5_field_reversal.py`（跑批）+ `scripts/plot_topic5_field_reversal.py`（画图）
> 测试：`tests/test_topic5_field_reversal.py`
> 结果根：`results/topic5_ictal_recruitment/field_reversal/`（gitignored，见 §4 复现命令）；图：`results/topic5_ictal_recruitment/field_reversal/figures/`（逐图中文说明见该目录 `README.md`）

---

## 0. 一句话（测了什么 / 怎么测的 / 揭示了什么）

**测了什么**：每个病人间期都会反复出现两种"传播模板"——同一批电极触点，事件有时候按一种先后顺序被激活，有时候按另一种顺序，这里把这两种叫 TA、TB。这次要问一件更几何的事：如果把 TA、TB 各自摊成一整张连续的空间"地形图"（同一批触点位置上，谁先谁后连续插值成一张面），这两张面是不是恰好整体"反过来"——TA 里早的地方在 TB 里变晚、TA 里晚的地方在 TB 里变早，像同一根轴的两个走向？而且反得比"只在同一根电极杆内部把触点顺序打乱"这种最起码的对照更狠？顺带还问了另一件事：把离散触点铺成连续空间场这个动作，是不是真的能让"这个触点大概什么时候被激活"这个估计更准、更不怕噪声——也就是原始动机里"平滑 = 去噪，因而更鲁棒"这句话到底站不站得住。

**怎么测的**：给每个病人建一张 TA、TB 两类事件共用的空间场坐标（同一批触点坐标、同一个平滑带宽、同一张网格），算 TA 场和 TB 场之间的带符号相关（越接近 −1 越像"完全反过来"；不做镜像翻转，翻转会把物理上不同的两个东西硬凑成一样）。零假设 = 只在同一根电极杆内部随机打乱触点的值（杆间顺序、坐标、参与度全不动），重建场、重算相关，重复一千次（`n_perm=1000` 正式全量），看真实观测值落在这个随机分布里的什么位置；同时有一个"全部触点整体打乱"的粗对照，和一个不看 TA/TB 标签、随机切两半事件比较的非推断对照（后者应该长得偏正，用来反衬"TA/TB 是真的不同，不是随便切两半就会有的巧合"）。另外单独测了"摊成场"这一步是不是真的让方向估计更可靠——用留一法：先用训练半段事件，各自建好"离散触点原始均值"和"去掉该触点后其余触点重建的平滑场、在该触点位置取值"两个预测器，再看哪个预测器更准地猜出留出半段里同一批触点的真实均值。两个触点覆盖范围都做了一遍：`broad`（覆盖更广的 20 触点）和 `narrow`（更贴近临床 SOZ 核心的小触点集），全程分开报告、从不合并。

**揭示了什么**：两件事，一真一假，必须分开说清楚。**第一，反向（正面信号，待签核）**：逐病人对着自己的零假设检验，`broad` 6/26、`narrow` 9/26 病人的反向强度显著超过随机水平（二项检验 p 分别约 1.5e-3 和 2.8e-6）——这是本次分析里最站得住的正面信号（**仍勿写「确为」**）：TA、TB 这两个间期传播模板在空间场层面**初步看起来是**同一根轴的两个相反方向（`n_perm=1000` 正式全量复核后仍成立，**待用户签核才能写成 final**）。**第二，「逐触点 LOO 重建」这一层，场不优于触点 self-mean**：用留一法比较，平滑场预测未见触点真实值的准确度，在全部 26 个 `broad`、26 个 `narrow` 病人里都不如直接用离散触点自己的原始均值（配对检验 p≈3.0e-8）。**但要点破一句**：这个检验**天然偏向触点 self-mean**——每个触点的间期均值是大量事件平均出来的高 SNR 量，拿掉它、用邻居插值估它本质是插值，当然更糊。所以它测的是**逐触点重建**、**不是**你原来关心的**传播轴 / 方向的鲁棒性**（1146 那种「电极序列被杆结构带偏、场用坐标纠回几何一致方向」的失败模式，本检验根本没碰）。**方向鲁棒性由 axis-level supplement（§2.7）回答**（Option-B cohort 已跑：读方向要用真实坐标、坐标盲杆折叠会跑偏、平滑不额外加分；定义见 spec §6a）。还有一个细节值得说：如果换一种"整个队列平均起来是不是普遍更负"的问法（配对 Wilcoxon 比较每个病人的观测值和他自己零假设的中位数），`broad` 这个问法下不显著（p≈0.099），只有 `narrow` 显著（p≈0.0002）——这不是矛盾，是两种问法本来就可能给不同答案：前者看"有没有个别病人打穿自己的门槛"，后者看"整个队列平均有没有系统性偏移"。**`n_perm=1000` 正式全量复核已完成（2026-07-07）；用户签核后方可写入主文档正式结论——pending sign-off。**

---

## 1. 结论边界 + 主张分级（spec §2 合同，逐字口径）

- **H_primary（反向门，预注册 primary cohort claim 档）**：per-subject，signed corr(TA_field, TB_field) 显著为负，且强于 within-shaft 重排 null；`broad`/`narrow` 各自二项检验，永不 pool。
- **H_supplement（方向鲁棒性 = §2.7 axis-level）**：坐标读法的轴是否比坐标盲（杆折叠）读法在 held-out 上更好地预测方向——这才是「更鲁棒方向」的直接检验。**per-contact LOO（§2.4）只是逐点重建 sanity，不是方向鲁棒性检验。**
- **三档结论 + 措辞边界**（spec §2 原文口径）：
  - **过**（field 版 K/n 病人过自己 within-shaft null，分底物二项显著）：可写「TA/TB 间期传播场在共享空间轴上反向对齐，超出同杆重排——**TA/TB 是同一根空间 scaffold 的两个相反遍历方向**，membership-robust 的几何 readout」。
  - **不过**：「反向对齐落在空间自相关预期内——forward/reverse 配对在场层部分是空间平滑 artifact」——有价值的阴性，非"无信号"。
  - **"场更鲁棒吗（方向鲁棒性）" = 被测子问题**，由 **axis-level held-out 方向预测（§2.7）**回答：读方向要用真实坐标、坐标盲杆折叠读法会在一部分病人（尤其 narrow 核）跑偏、平滑在坐标之上不额外加分。**field-vs-contact 过 null 病人数差（§2.3）与 per-contact LOO 配对（§2.4）只回答「逐触点重建」sanity、不回答方向鲁棒性**（LOO 天然偏向高 SNR 触点 self-mean）。
  - **禁写（即使反向门过）**：❌「发作会选择 TA 或 TB 极性」；❌「field 证明真实传播方向 / ground truth 方向」；❌ 方向重放；❌ 证明两模板存在（PR-2 gap_perm 已证）；❌ 有效刻画病理网络。上述极性/真方向类主张须 supplement（本文档 §2.4）AND 后续独立 ictal-polarity 检验都过才解锁——本次分析不含 ictal-polarity 检验。

**本次（n_perm=1000 正式全量）落点**：按上面"过"的字面判据，`broad` 与 `narrow` 的逐被试二项检验都显著（§2.1）——满足反向门的通过条件。但因为尚未经用户签核，**精确数值只记录在本 archive；`docs/topic5_seizure_subtyping.md` 与 `results/FIGURE_INDEX.md` 只放不含具体 p 值/分数的定性 preliminary 指针，不进入正式接受档**（CLAUDE.md §5：preliminary numbers go to archive only）。per-contact LOO 重建 sanity：场不优于触点 self-mean（§2.4，**非 axis test**）；**方向 / 轴鲁棒性由 axis-level supplement（§2.7）回答（Option-B cohort 已跑，定义见 spec §6a）**。

---

## 2. Cohort 数字（n_perm=1000 / n_split=200 / loo_split=50，正式全量，2026-07-07）

### 2.1 H_primary 正式判决层：逐被试二项 gate

| 底物 | 候选数 | 推断样本 n_ok | 过 own within-shaft null | binomial p | 观测 corr 中位 |
|---|---|---|---|---|---|
| `broad` | 26 | 26 | **6/26** | **1.5e-3** | −0.613 |
| `narrow` | 35 | 26 | **9/26** | **2.8e-6** | −0.607 |

两底物的逐病人二项检验都远超 0.05 显著性门槛——这是本次分析里正式的判决层，§2.2 的队列可视化不能替代它。

### 2.2 队列层面可视化：配对 Wilcoxon（辅助图，非判决）—— "两个问法不一致"

| 底物 | data 中位 | null 中位 | Wilcoxon(less) p | n(data<null) |
|---|---|---|---|---|
| `broad` | −0.613 | −0.292 | 0.0994（n.s.） | 16/26 |
| `narrow` | −0.607 | −0.037 | 0.0002（\*\*\*） | 20/26 |

这不是矛盾：§2.1 的二项 gate 问"有没有个别病人打穿自己的 5% 门槛"；这里的配对 Wilcoxon 问"整个队列中位数是不是系统性比零假设更负"。`broad` 在前一问法下有 6/26 显著个案，但队列中位数偏移本身没有显著到能拒绝"两分布位置相同"；`narrow` 两种问法都显著。**正式过/不过判定以 §2.1 二项 gate 为准，本节 Wilcoxon 只是 `field_reversal_cohort_stat.png` 的可视化，不是判决**（图注自带同一句话）。

### 2.3 表示层 head-to-head：场 vs 触点二元过关对比（谁通过自己零假设的病人更多）

| 底物 | both_pass | field_only | contact_only | neither | field 过关 | contact 过关 |
|---|---|---|---|---|---|---|
| `broad` | 5 | 1 | 2 | 18 | 6/26 | 7/26（触点略胜 1 个） |
| `narrow` | 9 | 0 | 1 | 16 | 9/26 | 10/26（触点略胜） |

单看"过没过自己零假设"这个二元问题，场没有明显买到更多过关病人——场和触点表现几乎打平。这一层的答案和 §2.4 的连续精度层面答案不同，两者分开看，不合并成一句"场没用"（CLAUDE.md §6.3 pronoun discipline）。

### 2.4 H_supplement（**降级为 sanity**）：per-contact LOO 重建 —— 场不优于触点 self-mean（**非 axis-level robustness test**）

留一法头对头：**contact** 预测 = 训练半段事件里该触点的离散原始均值；**field** 预测 = 训练半段去掉目标触点后、由其余触点重建的平滑场在目标触点位置的取值（LOO，杜绝自我平滑泄漏）；两者都对保留半段真值做跨触点 Spearman 打分。

| 底物 | field_rho 中位 | contact_rho 中位 | contact 赢的病人数 | 配对 Wilcoxon p |
|---|---|---|---|---|
| `broad` | 0.78 | 0.98 | **26/26** | 2.98e-8 |
| `narrow` | 0.70 | 0.99 | **26/26** | 2.98e-8 |

**在全部 26 个 `broad` 病人和全部 26 个 `narrow` 病人里，直接用离散触点原始均值都比平滑场预测更准**（两个底物的配对 Wilcoxon p 都是统计上能拿到的最小值，26/26 同号）。**但这是「逐触点重建」层面，不是「方向 / 轴」的检验**：该检验**天然偏向触点 self-mean**（每个触点均值是大量事件平均的高 SNR 量，邻居插值必糊掉 contact-specific 信息），它问「能不能用邻居重建某触点的值」，**不问「哪种表示给更鲁棒的传播轴」**。原始动机（「平滑是否让**方向估计**更鲁棒」）**本检验无法回答**——须看 axis-level supplement（§2.7，Option-B cohort 已跑，定义见 spec §6a）。**不写「denoising refuted」**（用户审阅 2026-07-06：LOO 换了题，per-contact 重建 ≠ axis robustness）。

### 2.5 Accountability：谁进了队列、为什么（8 类入组结果账目）

- **`broad`**：26 个候选全部 `ok`，8 类账目里除 `ok=26` 外全部是 0，无任何排除。
- **`narrow`**：35 个候选里 26 个 `ok`；被排除的 9 个 = **6 个 `c1_violation`**（标签本身不满足"稳定两簇"`stable_k==chosen_k==2` 前置条件，是标签层的既有事实，非本次分析的统计失败）+ **1 个 `plane_not_built`**（几何平面本身是 status-only 记录、从未建成，Task 8 post-hoc 加了守卫防止崩溃）+ **2 个 `degenerate_null`**（compact-core 可打乱触点太少，弱-null 守卫拦下）；其余 4 类账目（`no_planes`/`load_error`/`cluster_map_ambiguous`/`insufficient_overlap`）均为 0。

`narrow`（更贴近 SOZ 核心的小触点集）确实比 `broad` 更容易在门槛前被挡下——这正是"贴近核心 = 触点更少、更容易退化"的预期代价，是一个真实结果，不是运行失败（spec §8）。两个底物最终都剩 26 个可推断病人，样本量恰好相等只是巧合。

### 2.6 Broad-vs-narrow sensitivity：反向能不能扛到贴近 SOZ 核心的触点

两底物都 `ok` 的 20 个病人：

| | narrow pass | narrow fail |
|---|---|---|
| **broad pass** | 3（both_pass） | 1（broad_only） |
| **broad fail** | 4（narrow_only） | 12（neither） |

多数点落在"两者都负"的象限（signed_corr broad vs narrow 散点，见 `field_reversal_broad_vs_narrow.png`）——反向的**方向**在 `narrow` 下没有消失，不是只在粗覆盖下才出现的远场/粗糙现象（narrow 自己的过关数 9/26 事实上比 broad 的 6/26 还高一些）。但"同时通过各自零假设"的病人只有 3/20——反向的**方向**在两个尺度下相当一致，**统计显著性**却经常各自独立地打穿或打不穿阈值。

---

### 2.7 axis-level robustness（新增，用户 2026-07-06 Option-B）：读方向要用坐标、不是杆序；平滑不额外加分

**测了什么（白话）**：一个病人两套间期模板，每套都能读出「传播往哪个方向走」。三种读法：(a) **坐标轴** = 用触点真实二维坐标对发放先后做直线拟合；(b) **杆折叠轴（坐标盲，shaft-collapsed）** = 把同一根杆内的触点值压成**该杆的一个均值**、丢掉杆内先后，**再用真实坐标拟合方向**——即「只保留在哪根杆、不保留杆内位置」（**⚠️ 这不是「按电极名字/插入顺序排序」**，仍用真实坐标，只是把杆内信息抹平了）；(c) **平滑轴** = 先把触点值铺成空间场再拟合。1146 那个担心：多个 early 触点分布在两根杆上、真入口在两杆之间时，坐标盲的杆折叠读法会误成「A 杆→B 杆」。

**怎么测的**：砍两半事件，一半定方向、另一半验证——三种轴各自「沿轴投影预测另一半发放顺序」的 Spearman ρ（held-out）。null = 每个病人自己几何下、200 个随机方向的 held-out ρ 中位（随机方向该 ≈0）。每种轴对 null 配对 Wilcoxon；坐标 vs 杆序也配对。broad/narrow 分开、TA/TB 折叠、**永不 pool**。

**揭示了什么**：
- **三种读法都超过随机 null**（都抓到真顺序）：broad raw_contact p=1.5e-8（26/26）、sequence p=7.5e-8（25/26）；narrow raw_contact p=3.7e-9（28/28）、sequence p=5.5e-5（20/26）。**但坐标盲（杆序）超得少、边际更散**——紧核里好几个病人 held-out ρ 掉到 0 附近甚至以下（见三联图 narrow 组）。
- **坐标 > 杆序 显著**：broad p=1.2e-3（coord>shaft/shaft>coord/tie=13/2/11）、**narrow p=1.3e-5（19/2/5，更强）**。
- **坐标盲会严重跑偏**：分歧 `angle(sequence, raw_contact)` >45° 的被试 broad **3/26（11.5%）**、narrow **9/26（34.6%，超三分之一）**；个别近正交/相反（E1077 broad TB 164°、E1146 narrow TB 105°，见个案图）。**紧核 SOZ 核尤其容易被杆序带偏**（触点挤、多杆）。`poor_planarity` 标签预测不了它（1077 非 poor-planar 却最强）。
- **平滑不额外加分（如实阴性）**：field vs raw_contact 近平（broad 13/26 field 赢、narrow 8/28，都不显著），两者方向几乎重合（cos 中位 **0.996/0.997**）→ **有坐标就够，平滑不是关键。**
- **TA/TB 轴大致反平行（仅辅助，不承载反向主门）**：两套模板各自拟合出的坐标轴方向大致相反，cos(TA 轴, −TB 轴) 中位 broad **0.61**、narrow **0.67**（`cos_ta_tb_axis_antiparallel_median`，见 `axis_robustness/cohort_summary.json`）——中等强度，**不够强到单独承载「TA/TB 是反向对」的结论**；反向对的正式证据是 §2.1 的 within-shaft null 门（对每个病人自己的零假设），不是这个裸轴夹角。

**分母说明（narrow 28 vs 26）**：narrow 底物里，坐标轴 / 平滑轴的 held-out 打分只要有触点定方向即可用，**n=28**；而「杆折叠 / 坐标-vs-杆序」对比需要至少两根不同电极杆才能定义「杆折叠方向」，**2 个只落在单根电极杆上的病人无法定义杆折叠方向、被剔出杆序分母**，故 sequence / shaft 比较是 **n=26**。broad 底物无此问题（26/26 全可用）。

**结论口径（红线）**：**读传播方向要用真实坐标；坐标盲的杆折叠读法（把每根杆压成一个均值、丢掉杆内位置、再用坐标拟合；≠ 按电极名字排序）在一部分病人（宽池一成、紧核三成）会明显读错、个别接近正交或相反，紧核尤甚；平滑本身不额外加分——是「坐标」救回几何、不是「平滑」。** 原始「场更鲁棒 / 去噪」动机被 §2.4（per-contact LOO）+ 本 §2.7（axis）一起收窄成：**真正救回几何一致方向的是「用坐标」，场平滑没在坐标之上加分。不写「场去噪 / 真实传播轴」。** tier = supplement（方法学 note）。

**工件**：`src/topic5_axis_robustness.py`（三轴 + `axis_angle` + `held_out_axis_score` + `random_axis_null_score`，5 单测）、`scripts/{run,plot}_topic5_axis_robustness.py`、图 `axis_robustness/figures/{axis_three_way_comparison,divergence_distribution,case_axes}.png`（复刻 `plot_fig_topic5_network_extension_null.py` 三联样式）+ `why_coordinates_schematic.png`（示意 + 真实失败案例 E1146 narrow TB：坐标 ρ=+0.73 vs 杆折叠 ρ=−0.29 比随机还差、两轴 105°；`scripts/plot_topic5_axis_schematic.py`）、`axis_robustness/cohort_summary.json`。复现：`python scripts/run_topic5_axis_robustness.py`（`--input-results-root` 指主树 results）。

### 2.8 带宽敏感性（§7，n_perm=1000 队列聚合）：反向不是平滑尺度假象

反向门在 {0.5, 1, 2}×median-nn 三档平滑带宽下各重跑一遍（1.0× = §2.1 主门），看过门病人数、signed_corr 中位是否随带宽漂移。

| 底物 | 0.5× | 1.0×（主门） | 2.0× |
|---|---|---|---|
| `broad` 过门 | 7/26（p=2.2e-4） | **6/26（p=1.5e-3）** | 7/26（p=2.2e-4） |
| `broad` signed_corr 中位 | −0.59 | −0.61 | −0.61 |
| `narrow` 过门 | 10/26（p=2.5e-7） | **9/26（p=2.8e-6）** | 9/26（p=2.8e-6） |
| `narrow` signed_corr 中位 | −0.49 | −0.61 | −0.68 |

三档带宽下过门数和反向强度都稳定（两底物所有档 binomial p 都 <1e-3），**反向不是某一档平滑尺度才冒出来的假象**。（聚合字段 `cohort_summary.json::<substrate>.bandwidth_sweep`，由 `_aggregate_sweep` 生成。）

---

## 3. 图（完整逐图中文说明见 `results/topic5_ictal_recruitment/field_reversal/figures/README.md`）

全部位于 `results/topic5_ictal_recruitment/field_reversal/figures/`：

- `field_reversal_cohort_stat.png`（+ `field_reversal_cohort_stat_metadata.json`）—— 头条图：`broad`|`narrow` 并排 Data-vs-Null violin+箱线+散点配对（§2.2）。
- `accountability.png` —— 谁进了队列、为什么（§2.5）。
- `broad/field_vs_contact_headtohead.png`、`narrow/field_vs_contact_headtohead.png` —— 场 vs 触点二元过关 2×2（§2.3）。
- `broad/loo_reproducibility.png`、`narrow/loo_reproducibility.png` —— 去噪假设关键阴性图（§2.4）。
- `broad/field_reversal_null_forest.png`、`narrow/field_reversal_null_forest.png` —— 逐病人诊断图（辅助，非论文级结论图）。
- `field_reversal_broad_vs_narrow.png` —— broad-vs-narrow sensitivity 散点 + 2×2（§2.6）。
- `broad/per_subject/{epilepsiae_1077,epilepsiae_1125}.png`、`narrow/per_subject/{epilepsiae_1096,yuquan_zhaochenxi}.png` —— 每底物 2 个代表病人（1 个清楚通过 + 1 个"裸相关很负但没打穿自己零假设"的教学案例）。
- `case_1146_mechanism.png` —— **未渲染**（延后，非遗漏，见 §5）。

---

## 4. 复现命令

```bash
# 从任一持有本仓库 checkout 的目录运行（下面用本 worktree 举例）
cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic5-tatb-field-reversal

python scripts/run_topic5_field_reversal.py --cohort --substrate both
# 默认即 n_perm=1000 / n_split=200 / loo_split=50 —— 本文档正式全量数值的参数

python scripts/plot_topic5_field_reversal.py
```

**`--input-results-root` 说明**：`run_topic5_field_reversal.py` 默认 `--input-results-root=/home/honglab/leijiaxin/HFOsp/results`——这是上游几何输入 `spatial_modulation/propagation_geometry{,_broad}`（`GEOM` 字典）所在的共享主树，**不是**运行所在 worktree 自己的 `results/`（本 worktree 恰好也有本地拷贝，但这只是巧合，不能依赖）。若从一个没有这两个几何目录本地拷贝的 worktree/checkout 复现，需显式传 `--input-results-root <持有 propagation_geometry 数据的 results 根>`，否则会读到空/缺失路径。

**n_perm 说明**：本文档全部数值来自 `n_perm=1000` 正式全量跑批（2026-07-07，CLI 默认 `n_split=200`、`loo_split=50`）。此前 preliminary `n_perm=300`（`n_split=150`、`loo_split=30`）已被本次正式全量取代；唯一实质变化 = `broad` 过门 7/26→6/26（一个边际被试在 1000 permutation 下不再打穿自己的 5% 门槛），`narrow` 9/26 与所有其他数字不变。

---

## 5. 已知缺口 + 下一步（pending 用户签核的具体内容）

**正式收口（formal closure）三件套** —— 本 archive 结论从 preliminary 升为 final 并写入 `docs/topic5_seizure_subtyping.md` 正式接受档，须**同时**满足：**(1) 反向门 `n_perm=1000` 全量复核 ✅（2026-07-07 完成）；(2) 带宽敏感性 `{0.5, 1, 2}×median-nn` 聚合成队列级数字 ✅（§2.8）；(3) broad/narrow 各自最终 denominator + exclusion table 定稿 ✅**（exclusion 明细见 §2.5：broad 26/26 无排除、narrow 35→26；axis 层 narrow 分母 28 vs 26 见 §2.7）。**三件套已全部完成——剩余唯一 gate = 用户签核。**以下逐条展开：

- **`n_perm=1000` 已完成（2026-07-07）✅**：正式全量跑批已取代 preliminary `n_perm=300`；唯一变化 `broad` 7/26→6/26（边际被试掉门），`narrow` 9/26 不变。剩余唯一 gate = 用户签核后写入 `docs/topic5_seizure_subtyping.md` 的正式接受档（CLAUDE.md §5）。
- **`case_1146_mechanism.png` 未渲染**：spec §9 机制示意图（原始触点顺序被几何带偏 vs 场给出几何一致轴估计的对比图）本次未生成。核实过 `epilepsiae_1146` 的 `broad` `t_a` 几何记录确实带 `poor_planarity=True`（与 spec 自己对这个个案的警示一致，且该被试两个底物都通过反向门），但左侧面板要求的"朴素电极顺序拟合方向"不是本管线里已经算出的任何量——现有 `along_axis_mm`/`x_norm` 字段本身已经是几何+平滑之后的读出，也就是右侧面板要展示的东西，不能直接当"朴素对照"用。现造一个未经审阅的朴素拟合统计量，超出"能快速核实"的范围。按 spec §9"核实不了就换被试或跳过、不要硬凑预设叙事"的要求，本次选择记录并跳过，留给后续单独决定是否值得为这一张图专门实现该统计量。
- **带宽 sweep 已 cohort 聚合（§2.8）✅**：`{0.5, 1, 2} × median-nn` 三档过门数与 signed_corr 中位都稳定（`_aggregate_sweep` → `cohort_summary.json::<substrate>.bandwidth_sweep`），反向不是某一平滑尺度的假象。
- **narrow 分层分母**：narrow 候选 35（raw availability，spec §8 预告的分层分母）里，9 个因不同原因被挡在门外（§2.5），最终 26 个 `ok`；这是"贴近核心=更容易退化"的真实代价，不是数据墙。

---

## 6. 文件清单

- **代码**：`src/topic5_field_reversal.py`、`scripts/run_topic5_field_reversal.py`、`scripts/plot_topic5_field_reversal.py`
- **测试**：`tests/test_topic5_field_reversal.py`
- **结果**：`results/topic5_ictal_recruitment/field_reversal/`（`cohort_summary.json` + `per_subject/{broad,narrow}/*.json` + `figures/`，全部 gitignored，见 §4 复现命令）
- **图说明**：`results/topic5_ictal_recruitment/field_reversal/figures/README.md`
- **设计 + 计划**：`docs/superpowers/specs/2026-07-06-topic5-tatb-field-reversal-design.md`、`docs/superpowers/plans/2026-07-06-topic5-tatb-field-reversal.md`
