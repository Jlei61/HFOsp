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

## De novo 那层（LR-7，压力测试，NULL）

**日期**：2026-06-07
**代码**：`src/low_rate_template_stability.py`（`denovo_window_axis`, `_fast_2means`, `window_recovery_paired`, `count_matched_null_gap_paired`）+ `scripts/run_low_rate_denovo.py` + `scripts/plot_low_rate_denovo.py` + `tests/test_low_rate_denovo.py`（17 tests 全绿）。
**结果**：`results/topic4_sef_hfo/low_rate_template_stability/{cohort_denovo.json, per_subject_denovo/*.json, figures/denovo_recovery_vs_event_count.png}`。

### 这层在测什么（不是什么）

和主结果（read-back）的**唯一区别**：主结果里每个低事件窗"借用全程逐事件归属"来定方向（知道答案）；这一层**禁止看答案**——每个窗自己对窗内事件重新分两堆、自己定哪头是源头（更大的那堆定向、更小的堆翻转到同一轴），不依赖全程标签。其余全不变：rate 基线、common-channel 掩码、count-matched null 结构、结果文件。

这样 global（read-back）和 de novo 在**同一批窗、同一批随机抽样**上配对算——两者之差 = **不让偷看的代价（peek_cost）**。

### 朴素话结论

**测了什么**：低事件窗里，"不知道全程模板、自己重聚类"能不能发现出和全程一样的传播方向，并且比"通道发放计数排名"更靠谱。

**怎么测的**：主结果 read-back 在 +0.131，说明"知道全程答案时读回是靠谱的"。这一层把"知道答案"拿掉，看从零发现能不能一样好。预先锁定：带符号的方向恢复是主指标（同口径可以和 +0.131 对比）；另报 |·|（只问轴线不问哪头是源）和无序端点并集；主定义"两者之差 = peek cost"。

**揭示了什么**：

- **绝对恢复水平（能不能发现）**：低事件窗里 read-back = 0.88 >> de novo 带方向 = 0.52，而且 de novo 带方向（0.52）在 epilepsiae 里连发放计数（0.66）都不如——**从零发现是这三者里最差的读数**。
- **配对 RAW 差（更直白的指标）**：de novo 带方向 - 计数，epilepsiae 中位 **−0.167**，ALL **−0.069**。de novo **输给计数**。
- **null 校正（扣掉"小样本聚类本来就不稳"的部分）**：ALL +0.022（16/28，p=0.104，**NULL**）；epilepsiae +0.007（8/15，p=0.598，**NULL**）；yuquan +0.143（8/13，p=0.025，但见下方红旗）。
- **一致性检验通过**：同一 pipeline 的 global（read-back）臂，epilepsiae 一致性 +0.156 vs 主结果 +0.153，误差 0.003——**不是 pipeline 回归**，de novo 劣势是真实差距。

**诚实主句**：低事件窗里，从零自己发现传播模板的方向，**没有比发放计数更靠谱**，带方向指标在 epilepsiae 里 RAW 差 −0.167（反而不如计数）；null 扣完约为 0 意味着"从零聚类的劣势中大头是通用的小样本聚类不稳，不是额外时间漂移"。这是对 read-back 主结果的**压力测试不是推翻**——read-back 的优势依赖"已知全程模板"，这是真实的认知门槛。

### 关键数字

| 队列 | 绝对水平 low（read-back / de novo signed / rate） | RAW 配对差（de novo - rate） | EXCESS（扣 null） | p |
|---|---|---|---|---|
| ALL (n=28) | 0.88 / 0.52 / 0.70 | −0.069 | +0.022 (16/28) | 0.104 (NULL) |
| epilepsiae (n=15) | 0.83 / **0.36** / 0.66 | **−0.167** | +0.007 (8/15) | 0.598 (NULL) |
| yuquan (n=13) | 0.89 / 0.79 / 0.78 | +0.002 | +0.143 (8/13) | 0.025 ⚠️ |

⚠️ **yuquan p=0.025 是伪正结果**：去掉 gaolan 和 huanghanwen 后中位 +0.143 → +0.021、p 0.025 → 0.087（NULL，重算确认）。这两个被试的大 excess 来源是 **rate_repro 崩溃**（gaolan low 窗 rate_repro 中位 −0.34，huanghanwen +0.05）——rate 失灵不等于 de novo 成功。加之 yuquan 仅单 seed、每被试低档窗 4-11 个。三重脆弱。

**Peek cost（不让偷看的代价，配对中位）**：epilepsiae +0.104，ALL +0.004（yuquan 平）——对 epilepsiae 多天记录，拿走全程模板标签、低事件窗几乎全吃掉 read-back 的优势。

**极化（polarity）分解**：de novo |·|（只问轴线） = 0.64 > de novo signed = 0.52，gap = 0.12（低档）→ 0.02（高档）。约 11.7% 的窗 signed < 0 且 |·| > 0.5（翻了方向），这些窗集中在有正反两套模板的被试（reversed=True，13.0% vs reversed=False 2.2%）、且集中在**低事件数**（中位 66 vs 全窗 156）——少事件 → 反向模板半数运气偷到大堆 → 方向认反但轴线本身还可以。这是真实的**方向歧义**，不是 flip 代码 bug（direction-pure 被试 1150/922 该现象近零）。

**M-分级 de novo signed excess**：
`≤2(分辨不开) −0.20` → `3-4 −0.05` → `5-20 +0.12` → `21-100 +0.02` → `>100 0.00`。仅 M=5-20 档轻度正向，其余负或零。

**种子稳健性**（全队列 ALL primary）：seed 0 p=0.104 / seed 1 p=0.093 / seed 2 p=0.181——三个种子全 NULL。

**端点并集 secondary（无极化）**：ALL RAW −0.143，EXCESS +0.000（11/27，p=0.246，NULL）。

### 对抗验证（3 agent，2026-06-07）

- **独立重算**（不导入生产 de novo 函数，用 sklearn KMeans 独立实现）：epilepsiae 1073 low 档 [read-back 0.943, de novo signed 0.343, |·| 0.457, rate 0.547] 与 JSON 精确一致（delta < 0.01）。CONFIRMED。
- **polarity/sign 审计**：abs == |signed| 全 2708 窗误差为 0；reversed-dom 窗集中在 reversed=True 被试、集中于低事件数（中位 66）；direction-pure 被试近零（0/113, 0/24, 0/26, 1/155）——是真实方向歧义而非 flip bug。CONFIRMED（一个 framing 修正：reverse-dom 在低事件数、不是高事件数，见上方纠正）。
- **诚实性批评**：RAW paired gap (EPI −0.167) 必须明报、不能只报 null-corrected +0.022；yuquan p=0.025 是率崩溃伪正、需标注脆弱。已上述纠正纳入。CONFIRMED_WITH_CAVEAT（已修复）。
- **优化审计**（inline 检查）：`_null_m` 对 m≤25 精确（低档 53% 窗覆盖，主指标不受 bucketing 影响）；fast-2means 与 sklearn 分离测试结果一致；单线程修复（OMP_NUM_THREADS=1）去除了 ~29ms/call OpenMP 开销后仍需 M-bucketing 才实用。CONFIRMED。

### 与主结果的关系（不矛盾，有边界）

| | 主结果（read-back） | 本层（de novo） |
|---|---|---|
| **测的是什么** | 已知全程模板，短窗能不能读回 | 不知全程模板，短窗能不能从零发现 |
| **低档 epi 绝对恢复** | 0.83 | 0.36（低于计数 0.66） |
| **epi null-corrected excess** | +0.153 (p=0.003) | +0.007 (p=0.598) |
| **结论** | 时间稳健读数（read-back） | 短时记录无法独立发现 |

**统一诚实主句（两层合并）**："HFO 传播模板在全程记录里是比发放计数更稳的时间读数（read-back +0.131）；但这个优势依赖已知全程模板——如果只有一段短时安静时段、没有全程参考，从零发现传播方向比发放计数更差（epi RAW −0.167）。模板稳健性是一个读数时间稳定性现象；能否用短时记录独立发现，是另一个更难的问题，当前结果否定它。"

（内部归档代号：LR-7, denovo_window_axis, _fast_2means, window_recovery_paired, count_matched_null_gap_paired, peek_cost, polarity-free endpoint union, seed robustness）
