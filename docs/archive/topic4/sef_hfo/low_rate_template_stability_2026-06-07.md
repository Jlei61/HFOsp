# SEF-HFO — 低事件时间窗里，传播模板比发放计数更抗采样不足（正结果，已对抗验证 + 公平性修复后仍成立）

**日期**：2026-06-07
**前置 / 边界结论**：`soz_localization_results_2026-06-07.md`（SOZ 内部静态定位 + 整体稳定性**未**支持几何胜出——这条边界结论保留，与本结果不矛盾）。本分析是**换了尺度**的另一个问题。
**结果文件**：`results/topic4_sef_hfo/low_rate_template_stability/{cohort.json, per_subject/*.json, count_matched_null_all28.json, figures/reproducibility_vs_event_count.png}`。
**代码**：`src/low_rate_template_stability.py` + `scripts/run_low_rate_template_stability.py` + `tests/test_low_rate_template_stability.py`。
**一句话**：在 HFO 事件少的安静时间窗里，"通道响多少次(发放计数)"的排名会真实地随时间漂移、变得不可靠；而"传播谁先谁后(源→汇模板轴)"的排名明显更能复现全程结构。这是真实的**时间结构**漂移，不只是"平滑统计量"的假象。

---

## 这个分析在测什么 / 不在测什么（reference-recovery，不是 de novo discovery）

**它测的是 read-back（参考复现）**：**已知**这个病人**全程**的传播模板后，低事件时间窗里**还能不能读回这条传播轴**。具体：
1. 用全程群体事件先定义一条"源→汇"传播轴（含正反模板的全局逐事件归属，把反向事件翻到同一轴）。
2. 低事件窗里**不重新聚类**，只把该窗的群体事件**投到这条全程轴上**，看窗内传播排序能多大程度复现全程排序。
3. 同时看窗内发放计数排名能多大程度复现全程计数排名。
4. 用"随机抽同样多事件"对照，扣掉"小样本本来就抖"的部分。

**这对当前问题是公平的**：rate 也在复现"它自己的全程计数排名"——两者都是"短窗读回全程指纹"。**不是偷看结论，是一个 reference-recovery 问题。**

**它不测**：低事件窗能不能**从零、不借用全程模板标签、单独重聚类**地发现同一个模板（=de novo short-window discovery）。那是更难、更接近"短时记录能否独立发现模板"的下一层验证（低事件窗可能事件不够、KMeans 不稳），**作为下一层，不拿来否定当前结果**（见文末"下一步"）。

**结论（写回）**：**在全部可计算传播模板的通道集合中，低事件时间窗里，传播先后模板比发放计数更抗采样不足；这个差异既不是单纯小样本计数噪声（count-matched null 已扣），也不是沉默通道被 rate 重罚造成的假象（common-channel 公平版仍成立）。**

---

## 朴素话摘要（测了什么 / 怎么测的 / 揭示了什么）

**测了什么**：把每个病人的录制切成 1 小时一段。安静时段里 HFO 事件很少，"哪个通道响得多"这个排名会抖。问题是：这些低事件时段里，"传播先后顺序形成的模板"还能不能复现全程的模板，比"通道计数"更稳。

**怎么测的**：每个 1 小时窗，在**全部 lagPat 通道**（**不限制致痫区**）上算两件事和全程比——(1) 计数排名 vs 全程计数排名；(2) 传播先后排名 vs 全程模板轴。**公平起见两者在同一批通道上算**（窗内有传播排名 ∧ 全程有排名的通道；否则沉默通道会被计数当并列 0 重罚、却被模板直接丢弃，系统性偏向模板——见护栏 2）。正反两套传播模板会把"源头"平均糊掉，所以用全程的逐事件归属把反向模板的事件翻到同一条"源→汇"轴上再算（全局归属，不在每个窗里偷挑更像的模板）。再按窗事件数分 低/中/高 三档。**关键对照**：把"按时间连续的低事件窗"和"从全程随机抽同样多事件(打散时间)"比——如果连续窗比随机抽更不一致，就是超出抽样噪声的**真实时间漂移**。

**揭示了什么**：
- **低事件窗里，模板复现度(中位 0.88)明显高于计数复现度(0.70)**，且计数抖得厉害（下四分位逼近 0、甚至变负）；事件多时两者并拢到 ~0.96/0.88。
- 跨被试配对检验，**扣掉随机对照后的真实时间漂移 EXCESS 显著 >0**：合并 n=28 中位 **+0.131，25/28 例为正，p≈3e-5**；epilepsiae +0.153 (13/15, p=0.003)；yuquan +0.115 (12/13, p=0.001)。
- **这不是"平滑统计量"假象**：用"随机抽同样多事件"做对照，模板的优势几乎消失（随机对照各事件数档都接近 0）；连续低事件窗差距远大，相减 = 真实时间结构漂移 +0.131。直接证据：真实安静时段里计数复现度会掉到**负值**，随机抽永远不会——只有"哪些通道活跃随时间真的变了"才能造成。
- **这也不是"通道宇宙不公平"假象**（user review）：改成"计数与模板同在参与通道上算"(common-channel)后，结论**不变甚至更干净**(EXCESS +0.131, 25/28)——因为即使只看参与通道，安静窗里计数排名仍真实塌缩(如 590 公平版仍 −0.06)，而模板稳在 +0.74。零膨胀只是小头(参与通道版与全通道版几乎重合，见图虚线)。

（内部归档代号：low_rate_template_stability, template axis aligned-Spearman, rate count-Spearman, count-matched null excess, m_bucket, reversed-template event flip, primary_low_excess）

---

## 关键数字

**主检验（common-channel 公平版；每被试 low-event 窗内配对差的中位数 → 跨被试单边 Wilcoxon；EXCESS=扣 count-matched null 的真实时间漂移=诚实 headline）**：

| 队列 | RAW Δ(模板−计数) | RAW p | **EXCESS（扣随机对照）** | EXCESS 为正 | EXCESS p |
|---|---|---|---|---|---|
| ALL (n=28) | +0.176 (22/28) | <1e-3 | **+0.131** | **25/28** | **3e-5** |
| epilepsiae (n=15) | +0.179 (12/15) | 0.009 | **+0.153** | 13/15 | 0.003 |
| yuquan (n=13) | +0.173 (10/13) | 0.002 | **+0.115** | 12/13 | 0.001 |

**复现度交叉（描述性，非检验统计量）**：low 档 模板 0.88 / 计数 0.70；mid 0.92/0.85；high 0.96/0.88。

**M-graded 时间结构 excess（公平版；效应不均匀，随事件数变化）**：
`M≤2 +0.03(分辨不开,null太宽)` → `M=3-4 +0.09` → `M=5-20 +0.21(峰值,最确凿)` → `M=21-100 +0.14` → `M>100 +0.06(采样充足,趋近0,符合预期)`。

---

## 对抗验证（3 个独立 agent，workflow）

- **独立重算**：用另一套代码逐被试重算，每被试 primary_low_delta 与存档 JSON **4 位小数全等**；队列 headline 完全复现。CONFIRMED。
- **count-matched null（决定性）**：CONFIRMED-WITH-CAVEAT。"平滑统计量"假象被**否定为主因**（随机对照差 +0.025、各档近 0；时间结构 excess +0.140 远超）。
- **bug 审计（6 项）**：CONFIRMED 无 bug——反转对齐用全局逐事件标签(非 per-window 偷挑)、通道索引一致、<3 通道窗丢弃并报数、计数=群体事件参与数(与模板同批事件)、无静态 SOZ 泄漏、全程轴与窗轴同一对齐。
- **common-channel 公平性修复（user review 后追加）**：旧版 rate 在全通道(沉默→并列0)、template 仅在参与通道，对 template 有利。改成两者同在参与通道上算 → 结论**不变更干净**(EXCESS 0.140→0.131, 24/28→25/28, p 仍 <1e-4)。零膨胀是小头；count-matched null + common-channel 双修复后仍稳。

## Secondary：端点集合是否漂（user review，已跑）

主结果看的是**整条传播轴排序**复现；这个补充看离散的 **source/sink 端点集合**（全程 axis 两端各 top-2 = 4 个端点通道，common-channel）在低事件窗里漂不漂，与 rate top-2 overlap 做同样比较 + 同样 count-matched null。结果文件 `cohort_endpoint_overlap.json`。

| | RAW Δ(端点−率) | RAW p | EXCESS（扣 null） | 分布 |
|---|---|---|---|---|
| ALL (n=27) | +0.205 | <1e-3 | 中位 +0.000 / 均值 +0.11 | 11 个恰为 0、13 正、3 负 |
| epilepsiae (15) | +0.267 | 0.002 | 中位 +0.00 | |
| yuquan (12) | +0.138 | 0.006 | +0.167 | |

**诚实读法**：
- **RAW 端点 overlap 远好于 rate top-k**（低档 0.60 vs 0.33），但这**大部分是"平滑统计量/估计量"效应**——count-matched null 也复现了大半（随机抽也是端点比 rate top-k 稳）。
- **离散端点 metric 太粗**：~40%（11/27）被试 EXCESS **恰为 0**（端点集只 4 通道、饱和，时间连续窗与随机窗的 Jaccard 一样，分辨不开）。在能分辨的 16 个里，13 个为正（mean +0.11、最大 +0.60），方向与主轴结果一致；Wilcoxon p=0.002 是这 16 个里的正向显著（中位 0 是被 11 个零拉到的）。
- **结论**：端点集合**方向上**也漂（与主轴一致），但**离散 top-k 端点是粗读数、对约 40% 被试无分辨力**；**真正敏感、稳健的时间结构信号在整条轴排序**（主结果 EXCESS +0.131、25/28、p<1e-4），不在离散端点。**所以主结果用整条轴排序，端点 overlap 作方向佐证而非独立强结论。**

## 护栏 / 局限（必读，诚实）

1. **效应是 M-分级的，不是一致的 28/28**：峰值在 M=5-20；M>100 采样充足时趋近 0（应然）；**M≤2 分辨不开**（null 太宽，subjects 1084/442 不支持也不反对，符合"事件不足这个尺度测不了"的护栏）。报告须 M-分级，不可写成统一效应。
2. **小的"平滑统计量"假象真实存在但是少数**（随机对照各档接近 0）：有界平均(排名)确实比小数目计数稍平滑，略微抬高 RAW 效应量，但不解释它（时间结构 excess +0.131 远超）。**所以诚实 headline 用 EXCESS(+0.131)，不用 RAW(+0.176)。**
3. **"零膨胀/通道宇宙不公平"已修复（user review）**：现版 rate 与 template 同在参与通道上算(common-channel)；旧版"rate 全通道含沉默 0"作为 `rate_repro_allch` 敏感性保留(图中虚线，几乎与公平版重合 → 零膨胀影响小)。结论在公平版下仍成立。
4. **检验统计量是"逐窗配对差再取中位"**，不是"两个分档中位数相减"。后者会误判某些被试(如 liyouran：low 档模板中位<计数中位、但逐窗配对差 +0.28)。描述性交叉 0.85/0.70 不是检验量。
5. **4 个被试为负/不确定**：1150(真·计数>模板)、442/1084(M≤2 噪声)、922(~0)；与 M-分级图景一致，非矛盾。
6. **探索性 + 花园岔路**：本数据上这是第 ~6 个分析方向。主指标(low 档/1h/Spearman/配对差/扣 null)是预设的且两数据集均显著，但**无 held-out 不作确证**；需预注册 + 独立队列。
7. **KMeans 模板分区** random_state=0/n_init=5；24/28 被试为反转模板(reversed=True)，分区在验证子集稳定；弱-K=2 被试换种子可能翻转，未单独做分区稳定性审计。

## 与边界结论的关系（不矛盾）

- 旧（保留）：在 **SOZ 内部**做 SOZ-vs-非SOZ 静态定位 / 整体 top-k 稳定性，几何**没**胜过率。
- 新（本结果）：在**全部 lagPat 通道**上、问"低事件窗里哪个读数更能复现自己的全程结构"，传播模板**显著**比计数抗采样不足，且是真实时间漂移。
- 统一诚实主句：**"HFO 发放计数在安静/短时段会真实漂移、变得不可靠；传播源→汇模板轴是更抗采样不足的读数。这是关于读数时间稳定性的现象，不是 SOZ 内部定位精度的主张。"**

## 下一步（候选 / 分层）

- ✓ count-matched null + M-分级 + common-channel 公平版已正式并入 runner 输出(`cohort.json` 含 `median_low_excess_NULLCORRECTED`/`wilcoxon_p_excess_low`/`m_graded_cohort_excess`)。
- ✓ **(secondary，已跑) 端点集合是否漂**：见上"Secondary"节——方向一致但离散端点是粗读数（~40% 被试 EXCESS=0 分辨不开），真正稳健信号在整条轴排序。
- ✓ **(下一层验证，已跑，NULL) de novo short-window discovery**：见下"De novo 那层（LR-7）"节。
- 30min 窗敏感性（`--window-min 30`）；per-template 事件预算更公平版；"模板缺失通道按不可复现惩罚"更狠 sensitivity。
- 临床意义对接：这个时间稳定性差异是否帮助"短时记录也能定位"——需在独立队列预注册检验。

---

## De novo 那层（LR-7）：短窗独立发现能力 — 负结果

**日期**：2026-06-07
**代码**：`src/low_rate_template_stability.py`（`denovo_window_axis`, `_fast_2means`, `window_recovery_paired`, `count_matched_null_gap_paired`）+ `scripts/run_low_rate_denovo.py` + `scripts/plot_low_rate_denovo.py` + `tests/test_low_rate_denovo.py`（17 tests 全绿）。
**结果**：`results/topic4_sef_hfo/low_rate_template_stability/{cohort_denovo.json, per_subject_denovo/*.json, figures/denovo_recovery_vs_event_count.png}`。

### 科学靶子（三档指标，主次分明）

这层要回答的问题是：**低事件窗只用自己的事件重新算模板，能不能恢复多事件/全程数据里看到的同一条传播结构？** 全程数据是 ground truth；评分始终是"短窗从零算出的结果 vs 全程答案"。

和主结果（read-back）的唯一区别：主结果里每个低事件窗借用全程逐事件归属来定方向；这一层**禁止看全程答案**，只用窗内事件自己重新聚类。数字上 `template_repro_denovo_signed = Spearman(window_denovo_axis, full_axis)`——分子和分母都是明确的：窗内自己算的轴 vs 全程轴。

**三档指标（预先锁定，都报）**：

1. **主指标 — 恢复全程传播方向（signed）**：短窗从零算的源→汇方向，与全程源→汇方向的 Spearman 相关（有符号，方向翻了记负）。这是"能不能独立发现同一条有向传播结构"。
2. **分解指标 — 恢复轴线但不管方向（|·|）**：只问"是不是同一组通道在轴两端"，不问哪头是源。解释失败原因：短窗经常找到了轴线、但方向认反了。
3. **端点 secondary — 恢复源/汇端点集合（无极化 union Jaccard）**：离散 top-k 问题，粗读数，见下方 secondary 节。

**Rate 是操作性参照，不是主对手**：rate 和 de novo template 估计的是不同对象（"哪个通道响得多" vs "传播先后结构"），不应直接作为主竞争对手报告。但作为操作性问题——"如果短窗只能用自己的数据、要选一个 fingerprint，de novo template 是否比发放计数更可靠"——rate 是合理的参照，结果见下。

### 朴素话结论

**测了什么**：低事件窗里，只用这段时间自己的 HFO 事件（不看全程），能不能重新发现全程里那条"哪个通道先放电、哪个后放电"的传播模板。

**怎么测的**：每个 1 小时窗，把窗内事件自己分两堆、自己决定哪堆是源头方向，然后算出一条轴；再和全程轴比相关（signed）。同时也报"只看轴线两端是否一样、不管哪头叫源"（|·|）。**Ground truth 始终是全程轴，不是 rate。**

**揭示了什么**：

- **主指标（能不能从零发现方向）**：低事件窗里，从零恢复全程传播方向很难。低档绝对恢复：read-back 0.88 > de novo |·| 0.64 > de novo signed 0.52。epilepsiae 多天记录 de novo signed 只有 0.36，且三个种子全部不显著（p=0.60，NULL）。
- **分解（轴线 vs 方向）**：约 12% 的窗 signed < 0 而 |·| > 0.5——轴线本身部分恢复，但方向认反了。这些窗集中在少事件（中位 66 事件）+ 有正反模板的被试：少事件时随机偷到大堆 → 方向歧义。这是短窗独立发现的核心困难，不是代码 bug。
- **操作性参照（rate）**：作为短窗 fingerprint，de novo template 在 epilepsiae 里也不比发放计数更可靠（epi low 档 de novo signed 0.36 < rate 0.66）。这说明短窗里"传播先后"这个信号不比"谁响得多"更易提取——不是"模板稳定性差"，而是"从短窗里独立学模板这件事本身很难"。
- **一致性通过（重要对照）**：同一 pipeline 的 read-back 臂复现 epi +0.156（vs 主结果 +0.153）——de novo 的劣势是真实的认知门槛，不是 pipeline 回归。

**科学意义（重要）**：这个负结果实际上**有正面的科学内容**：传播模板是一个需要用多事件/全程数据先学习出来的稳定结构；它不是短窗里随手可发现的结构。这直接支持了主结果里"read-back 读回"这一框架的正当性——read-back 是真实有意义的操作（先建模板，再看短窗能否对齐），而不是一个平凡的自我印证。

### 关键数字

**主指标：低事件窗从零恢复全程方向（de novo signed absolute）**

| 队列 | read-back 绝对水平 | de novo signed 绝对水平 | de novo |·| 绝对水平 | p（de novo signed > null） |
|---|---|---|---|---|
| ALL (n=28) | 0.88 | 0.52 | 0.64 | 0.104 (NULL) |
| epilepsiae (n=15) | 0.83 | **0.36** | 0.48 | 0.598 (NULL) |
| yuquan (n=13) | 0.89 | 0.79 | 0.82 | 0.025 ⚠️ |

⚠️ **yuquan p=0.025 不是真正的正结果**：去掉 gaolan 和 huanghanwen 后 p → 0.087（NULL）。这两个被试的大 excess 来源是 rate_repro 崩溃（gaolan low 窗 rate_repro 中位 −0.34），不是 de novo 成功。加之 yuquan 仅单 seed、每被试低档窗 4-11 个。三重脆弱。

**操作性参照（rate）**：epi low 档 de novo signed 0.36 < rate 0.66；ALL RAW 配对差（de novo - rate）= −0.069（epi −0.167）。作为短窗 fingerprint，de novo template 在 epilepsiae 里甚至不如发放计数。这是一个关于短窗独立学习能力的操作性事实，不是"模板比率差"的结论（两者估计的对象不同）。

**方向失败机制（分解）**：de novo |·| = 0.64 > de novo signed = 0.52，gap = 0.12（低档）→ 0.02（高档）。约 11.7% 的窗 signed < 0 且 |·| > 0.5，集中在 reversed=True 被试（13.0% vs 2.2%）和低事件数窗（中位 66 vs 156）。方向歧义在少事件时最严重，事件多时四线并拢。

**三 seed 稳健性（ALL）**：p = 0.104 / 0.093 / 0.181，全 NULL。

**端点 secondary（无极化 union）**：ALL RAW −0.143，EXCESS +0.000，p=0.246，NULL。

**Read-back 与 de novo 对比（不矛盾，是互补）**：

| | 主结果（read-back） | 本层（de novo） |
|---|---|---|
| **操作定义** | 已知全程模板，短窗投影到全程轴读回 | 短窗只用自己事件，从零算轴并和全程比 |
| **epi low 档 signed 绝对值** | 0.83 | 0.36 |
| **epi 检验** | +0.153 (p=0.003) ✓ | NULL (p=0.60) |
| **科学内容** | 模板是稳定时间读数（需要全程） | 短窗无法独立建立可靠模板 |

**统一主句**：全程传播模板是一个需要多事件数据学习出来的稳定结构——一旦建立，低事件短窗能稳定读回它（read-back +0.131）；但低事件短窗无法从零独立发现它，尤其是方向（epi de novo signed 0.36，NULL）。这两层合在一起，清楚描述了这个结构的性质：**稳定但需先验**。

### 对抗验证摘要（3 agent，2026-06-07）

- 独立重算（sklearn KMeans 独立实现）：1073 low 档 de novo signed 0.343 与 JSON 精确吻合（< 0.01 delta），rate 0.547 > de novo，read-back 0.943。CONFIRMED。
- polarity/sign 审计：abs == |signed| 全 2708 窗误差为 0；reverse-dom 窗集中在低事件数（中位 66）+ reversed=True 被试；direction-pure 被试近零。是真实方向歧义，非 flip bug。CONFIRMED。
- 优化审计（inline）：`_null_m` m≤25 精确，覆盖 53% low 窗；fast-2means 分离测试与 sklearn 一致。CONFIRMED。

（内部归档代号：LR-7, denovo_window_axis, _fast_2means, window_recovery_paired, count_matched_null_gap_paired, peek_cost, polarity/direction ambiguity, seed robustness）

---

## LR-7 补充：方向无关端点稳定性（KMeans-union 方法）

**日期**：2026-06-07
**代码**：`src/low_rate_template_stability.py`（`window_endpoint_stability_denovo`）+ `scripts/run_denovo_endpoint_stability.py` + `tests/test_denovo_endpoint_stability.py`（5 tests 全绿）。
**结果**：`results/topic4_sef_hfo/low_rate_template_stability/cohort_endpoint_stability.json`。

### 重新设计的动机

前一层（signed axis de novo）被方向歧义拖累：正反两套模板混合时，de novo 经常把方向认反。用户提出：直接看端点通道——KMeans k=2 分出两个子集，每个子集取最早/最晚放电的通道，**并集**。并集天然方向无关：正向子集的源头 = 反向子集的汇点，都出现在各自子集的极端位置，合并后就是同一批通道。

### 方法

1. 对窗内事件做 KMeans k=2（无方向信息）
2. 对每个子集：取 top-k 最低秩（最早放电）+ top-k 最高秩（最晚放电）通道
3. **两个子集并集** = 窗内端点候选（方向无关）
4. 和全程端点集（全程轴两端各 top-2 = 4 个通道）比 Jaccard
5. Rate 参照：发放次数最多的 top-4 通道

小窗口（< 4 事件）退化到直接取 naive mean 两端，不做 KMeans。

### 结果

| | low 档绝对 endpoint_J | rate_J | EXCESS（扣 null） | p |
|---|---|---|---|---|
| ALL (n=27) | 0.38 | 0.33 | **−0.064** (4/27) | 0.993 (NULL) |
| epilepsiae | 0.38 | 0.33 | −0.064 (3/15) | 0.950 (NULL) |
| yuquan | 0.36 | 0.24 | −0.051 (1/12) | 0.981 (NULL) |

**绝对水平参照**：
- 纯随机（N=15 通道中随机选 4 个）：Jaccard ≈ 0.15
- 发放计数 rate：0.33
- KMeans-union 端点：0.38（>rate，>随机，但 excess 为负）

**EXCESS 为负的解释**：count-matched null（从全程随机抽同样 M 个事件做同样流程）给出 ~0.44，比实际低事件窗（0.38）更高。意味着**连续安静时段比随机 M 事件更难恢复端点**——可能安静时段事件更均质（只出现一套模板）或 KMeans 分离度更低。

### 两种 de novo 方法收敛到同一结论

| 方法 | low 档绝对复现 | EXCESS | 结论 |
|---|---|---|---|
| Signed 轴（带方向） | epi 0.36 | NULL | 方向识别失败 |
| \|·\| 轴线（去方向） | 0.48 | NULL | 轴线部分恢复 |
| **KMeans-union 端点（本节）** | **0.38** | **NULL/负** | 端点识别也失败 |

去掉方向问题后，结论不变：**短时间安静窗里，无法可靠地从零识别最早/最晚放电的通道。**

### 科学意义

这个结果从负面确认了 read-back 主结果的价值：
- 直接找端点 → 不可靠（0.38，null）
- 先用全程数据建模 → 再投影短窗 → 可靠（read-back +0.131）

"先学习、再读回"不是取巧，是数据告诉我们的必要性：少事件窗里的信号量不足以独立建立稳定的传播结构，无论是方向敏感还是方向无关的方法。
