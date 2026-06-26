# Topic 5 A-line — event-resolved interictal axis_bias — PILOT recap

> **日期**: 2026-06-25
> **状态**: PILOT executed (3 epi subjects); **HARD STOP before cohort** — 待 advisor(=用户) sign-off
> **tier**: secondary / exploratory-descriptive（不触碰、不扩展 A-line primary scaffold 结论）
> **上游 spec/plan**: `docs/superpowers/specs/2026-06-25-topic5-event-resolved-axis-bias-design.md` (v2) ·
> `docs/superpowers/plans/2026-06-25-topic5-event-resolved-axis-bias.md`
> **代码**: `src/topic5_event_resolved_alignment.py`（17 tests） · runner `scripts/run_topic5_event_resolved_alignment.py` · 图 `scripts/plot_topic5_event_resolved_alignment.py`
> **输出**: `results/topic5_ictal_recruitment/event_resolved_alignment/`（`per_subject/`, `figures/`, `pilot_summary.json`）

---

## 0. 白话摘要（CLAUDE.md §8 三段式）

**测了什么** — 一个病人平时成千上万次的间期"高频放电传播"已经被自动分成了两类（这里就叫 A 类、B 类）。原来的主线是把同一类成千次事件压成一张"平均传播图"，再看这张平均图跟癫痫发作头几秒的活动图是不是落在同一条空间方向上。这次我们不压平均：把**每一次具体的间期事件单独拿出来**，逐次问"这一次事件像不像发作早期那张图"，这样才能看清同一类内部到底有多分散（有的事件很像、有的不像），以及 A、B 两类各自有没有不同偏向。

**怎么测的** — 关键障碍是一次间期事件只点亮很少几个电极触点——在原来那套窄电极集合里，一半事件只点亮 3 个触点，根本铺不成一张图。我们改用一套更宽的电极集合（每个病人 20 个触点），这样每次事件平均能点亮约 10–14 个触点，够铺一张小图了。每次事件铺成它自己的小图，用**跟原来主线同一个场相关算法**（只看两张图的空间梯度是否落在同一条线上、不分正反方向）跟发作图比，得到这一次事件的一个"对齐分数"（0 到 1）。把同一类所有事件的分数收成一条分布：分布的宽窄 = **这一类内部的事件级离散度**，A、B 两类分布的位置差 = 类间偏向。判断两类差异是不是真的，**把 A/B 标签在每段录音块内部随机打乱上千次当对照（保留每块原来的 A/B 个数→事件级类大小不漂移、块内真实混合也保留），而且打乱后用新标签对应那一类的平面重新取对齐分数（付"按类选平面"的成本）**——因为相邻事件在时间上扎堆、不独立，必须块内打乱才公平。

> **两点口径更正（2026-06-25 用户审阅后）**：(1) **本轮量的是"事件级离散度"（统计散布），不是"短期抑制 STD（short-term depression）"**——某类事件之后同类/反类是否被短期压低、随间隔恢复，那是序列问题（Stage C），本轮没做，**别写"展现 STD 效应"**。(2) "同一个尺子"只指**场相关算法本身**；发作场的估计量本模块用 subject-mean（比 primary 的 per-seizure 折叠更平均），是 secondary 口径，cohort 前要补 per-seizure-median 敏感性。

**揭示了什么** — 这是一次小范围预跑（3 个病人，验证方法可行性），**不是结论**。三件事看清了：(1) 方法在宽电极底物上完全跑得通——每个事件都能算出对齐分数（可用率 100%），而且事件跨越上百段录音块，样本基础扎实；(2) **事件级离散度**确实存在且很大——同一类里逐事件对齐分数从接近 0 一直散到 0.5 以上，平均成单模板时这部分信息就被抹掉了（**再次强调：这是统计离散度，不是 STD 短期抑制机制**）；(3) 两类事件之间确有差异：用块内约束打乱（保类大小+付选平面成本）做对照，3 个病人的 A−B 位置差都超出全部 1000 次打乱（单被试、未校正），其中 2 个连离散度也不同（B 更散）、1 个（922）只位置不同、离散度相当；但"哪一类更像发作"在不同病人间方向不一致（A/B 只是任意两类标签，跨病人比方向得先把模板朝向对齐）。这些都是**描述性、单病人、未做多重比较校正**的——绝不能读成"某类间期事件驱动/重放发作"。

（内部归档代号：A-line primary = `corr_pair_mirror_invariant` 2D field + 4-null + FDR；本 secondary = event-resolved axis_bias，metric M = 逐事件镜像不变场相关 / M1d = 1D 顺序-激活共线 companion / R2 = block-level A/B label-shuffle separation null；底物 = `interictal_propagation_masked_broad` + `lagpat_broad_epilepsiae` + `propagation_geometry_broad`；A/B = `adaptive_cluster.labels` stable_k=2 → `t_a`/`t_b`。）

---

## 1. 这一轮做了什么（build + pilot）

1. **设计经 advisor-proxy 四镜头复审 → spec v2**：v1（"层3 bootstrap 宽度测 std + K-scan"）被一致判为统计错误（bootstrap-of-means 宽度 = 均值标准误 ~1/√K，非类内 std；K-scan 收窄是 CLT 必然，零异质也"通过"）→ 废弃。改为**逐事件分布的直接 IQR/std**测散度。其余 7 处 blocker（缺 A/B 标签打乱 null、忽略块自相关、每类该用自己平面、逐事件 support、cluster↔template 近镜像误映、ictal 估计量口径）全部写入 spec §C1–C10。变更账见 spec §9。
2. **纯函数模块 `src/topic5_event_resolved_alignment.py`（17 TDD 全绿）**：loader+§C1 位置级对齐三重硬校验、cluster↔template signed/margin/bijection 映射、逐事件场度量 M、1D companion M1d（已向量化）、block-level 分离 null R2、participation 诊断、Stage B/C stub。
3. **底物 de-risk（关键）**：实测确认 broad 标签与重载事件**位置级**对齐（复现 producer 模板逐元素相等，1077/1125/922 全 exact），且 broad 每事件参与触点中位 ~10–14（窄底物只有 3）。
4. **pilot 跑通 3 epi 被试 + 出图**；yuquan_zhangbichen 因缺 broad 几何平面（`propagation_geometry_broad` 暂无 yuquan）优雅跳过。
5. **用户审阅后修复（2026-06-25，P0+3×P1，18 TDD 全绿）**：
   - **P0 R2 重写**：旧 null 把每块压成 dominant label 再整块打乱——抹掉块内混合（真实数据块大量混合：1077 149/176、1125 144/158、922 107/114 mixed）、类大小漂移、且没付"按类选平面"成本。新 null = **每事件先在两个类平面下各算一个对齐分（align0/align1）**，观测用各自类平面值；打乱 = **块内约束置换**（保每块 A/B 个数→类大小不漂移+块内混合保留），打乱后**用新标签的平面重新取分**（付选平面成本）。
   - **P1 报告口径**：所有 p 旁只报分析子集数（`n_events_analyzed`）+ 可用块（`M_n_blocks_usable`/`R2_n_blocks`）；全数据仅用于 participation 诊断（键改 `n_blocks_full`/`participation_full`）。
   - **P1 措辞**：本轮=事件级离散度，**非 STD 短期抑制**（=Stage C 序列）；"同尺子"仅指场相关算法，发作场估计量=subject-mean≠primary，cohort 前补 per-seizure-median 敏感性。
   - **用户 pivot（新主诉求）**：每类全事件投影+加权归一化→场图（A 场 | B 场 | 发作场，左图形式），`scripts/plot_topic5_event_resolved_fields.py` → `figures/fields/<被试>_class_fields.png`。

---

## 2. 底物决策证据（窄 vs broad）

| | 窄（原 A-line 底物） | broad（本 pilot 选用） |
|---|---|---|
| 每事件参与触点（1077） | min/median/max = 3/**3**/6；≥6 仅 4.6% | min/median/max = 5/**14**/20；≥6 = 99.7% |
| 逐事件 2D 场度量 M | 几乎全部 `insufficient_overlap`（dead） | **usable fraction = 1.00**（3 被试全 100%） |
| 事件数 | 多（5万+/被试） | 少（345–2万/被试） |
| 队列覆盖 | 18 epi 全（= A-line primary） | 12 epi + 17 yuquan（≠ primary，注脚） |

**结论**：逐事件**场**度量必须走 broad；窄底物只能留给 1D companion（且需 n_ch≥8）。代价 = broad 子队列 ≠ A-line primary 全队列（可比性注脚），事件数较少（已用 n_blocks 报有效样本）。

---

## 3. pilot 结果（3 epi；descriptive，每类下采样 ≤1500 事件，固定种子；n_blocks 按全数据报）

（**修复版 R2 = 块内约束置换 + 双平面再取分**；分析子集每类≤1500 事件，R2 基于分析子集的块）

| 被试 | map margin | M usable | analyzed ev | R2 blocks | A: med(IQR) | B: med(IQR) | A−B Δmed (within-block p) | IQR比 A/B (p) |
|---|---|---|---|---|---|---|---|---|
| 1077 | 1.34 | 1.00 | 2757 | 176 | 0.203 (0.256) | 0.290 (0.333) | −0.087 (0.001*) | 0.77 (0.001) |
| 1125 | 1.30 | 1.00 | 3000 | 140 | 0.301 (0.320) | 0.407 (0.480) | −0.106 (0.001*) | 0.67 (0.001) |
| 922 | 0.84 | 1.00 | 3000 | 111 | 0.431 (0.209) | 0.325 (0.214) | +0.106 (0.001*) | 0.98 (0.76) |

\* p=0.001 = 1/(1000+1) 下限 = 观测超出**全部 1000 次块内约束置换**（单被试、未校正、exploratory）。

读法（朴素，§8）：
- **事件级离散度大**：每类 IQR ≈ 0.2–0.5——逐事件对齐分数远非单一值，散得很开。这正是"平均成单模板会抹掉的东西"。（再次：这是统计离散度，**非 STD 短期抑制**。）
- **类间有差（修复版 null 下仍成立）**：用块内约束置换（保类大小+保块内混合+付选平面成本）后，3 个被试的 A−B 位置差仍超出全部 1000 次置换；离散度差在 1077/1125 成立（B 更散，p=0.001）、922 不成立（IQR 比 0.98，p=0.76）。
- **方向跨被试不一致**：Δmed 在 1077/1125 为负、922 为正——A/B 是任意两类标签（前/反向模板），跨被试比方向必须先做朝向对齐（cohort 前置步骤）。

---

## 3b. class 场 vs 模板 场 对比（用户 2026-06-25 第二轮，全 12 epi broad cohort）

**白话**：把"每类全部事件投影+加权归一化的场"（class 场）和"原来那张平均模板场"（template 场）放一起，用**和 A-line 完全一样的统计、一样的 max 取法**（候选 = A、B 两个场里取对齐更高的那个；对照打乱时也取 max，付选择成本）跟三个目标场比——两个发作前窗（−10..0s、−120..−90s）+ 发作头 0–10s（接 A-line）。

**结果（descriptive，单被试 channel-null 未校正，n_null=100）**：
| 目标窗 | template 超随机 | class 超随机 | class−template Δmedian | class>template |
|---|---|---|---|---|
| 发作前 −10..0s | 5/12 | 6/12 | −0.018 | 5/12 |
| 发作前 −120..−90s | 5/12 | 5/12 | −0.002 | 6/12 |
| 发作头 0..10s（onset，接 A-line） | 7/12 | 6/12 | +0.002 | 7/12 |

**揭示了什么**：**在"聚合成一张场"这个层面，class 场 ≈ template 场**——两者跟发作前/发作场的 max-AB 对齐几乎一样（Δmedian ≈ ±0.02，谁高谁低约各半），没有系统差异。这是意料之中且是好事：模板本来就是这一类事件的平均，所以把事件重新聚合回去得到的就是模板（6 张/被试的场图也是这么看：template A ≈ class A）。**含义：event-resolved 的增量不在"聚合场"层（那等于模板），而在"逐事件"层**（Stage A 的离散度）以及之后的"序列/短期抑制"层（Stage C）。约一半被试在发作前窗也超随机，与"长期静态网络底座（scaffold）在发作前也在"的口径一致（未校正、exploratory）。

工件：`scripts/run_topic5_class_vs_template_alignment.py`（max-AB 统计，resume-able）、`scripts/plot_topic5_class_vs_template_fields.py`（6 面板/被试）；输出 `results/.../event_resolved_alignment/class_vs_template/{per_subject/*.json, figures/*.png, cohort_summary.json}`。

---

## 4. 老实的限制（不可越界）

- pilot = 3 被试、descriptive、未做多重比较校正、未做 LOSO；**不是 cohort 结论**。
- block-shuffle p 触底 0.001 仅说明"观测差异超出本被试所有块级打乱"，是**单被试**陈述，不是队列显著性。
- M 的 ictal 场用 subject-mean（比 primary 的 per-seizure-median 更平均）——口径**不同于** primary，仅作 secondary（spec §C6）。
- 事件下采样到 ≤1500/类（控算力）；散度/位置在此采样下已稳，但 cohort 版需复核采样敏感性。
- M1d（1D 顺序-激活共线）是**比 primary 更接近"重放"的构念**，只作交叉验证、只看类级分布、不点名单个事件、不留方向符号。

---

## 5. 给用户的决策菜单（advisor sign-off 闸门 — 回来再定）

1. **是否进 cohort run**：在 broad 子队列（12 epi + 需补 yuquan broad 几何后的 17 yuquan）上跑 M + R2，配对（先做模板朝向对齐）+ LOSO + 多重比较口径（FDR 或明示 raw exploratory）。
2. **底物/度量微调**：(a) 是否补 per-seizure-median 版 ictal 作 sensitivity；(b) 采样上限是否提高/全量；(c) 是否给 yuquan 建 `propagation_geometry_broad`（当前只有 epi，故 yuquan 被试全部无法跑场度量）。
3. **是否进 Stage B（窗口偏向）/ Stage C（序列效应）**：时间戳已确认可用（`event_abs_times`/`block_ids`），入口现为 stub。需另写 spec/plan。
4. **跨被试方向口径**：cohort 比"哪类更像发作"前，必须先定模板朝向对齐规则（否则 Δmed 符号无意义）。

---

## 6. 工件清单

- 代码：`src/topic5_event_resolved_alignment.py`（**18 tests**，`tests/test_topic5_event_resolved_alignment.py`）— loader+C1、cluster↔template map、双平面逐事件场度量 M、M1d、块内约束分离 null R2、class_aggregate_contact_values（场图用）、Stage B/C stub。
- 驱动：`scripts/run_topic5_event_resolved_alignment.py`（`--pilot` / `--subjects` / `--max-per-class` / `--activation`；**拒绝无 --pilot/--subjects 的隐式 cohort run**）
- 图（小提琴/逐事件分布）：`scripts/plot_topic5_event_resolved_alignment.py`（3 面板/被试）
- **图（场图 = 用户 pivot）**：`scripts/plot_topic5_event_resolved_fields.py` → `figures/fields/<被试>_class_fields.png`（A 类全事件场 | B 类全事件场 | 发作场，同一物理平面，左图形式）
- 图说明：`figures/README.md`（中文，含两类图 + STD/离散度更正口径）
- 数据：`results/topic5_ictal_recruitment/event_resolved_alignment/{per_subject/*.json, figures/**, pilot_summary.json}`

**HARD STOP**：cohort run / cohort 判决未执行，等用户(advisor) sign-off。当前所有结论为 pilot-exploratory。代码与文档**未 commit**（沿用本项目 exploratory 工作"待用户审阅再提交"惯例；当前分支 `topic4-event-extent-audit`）。
