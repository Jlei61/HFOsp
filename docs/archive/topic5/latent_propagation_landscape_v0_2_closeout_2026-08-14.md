# Topic 5.2 latent propagation landscape 与 perturbation response v0.2 收口

> 状态：**SCIENTIFIC CLOSEOUT COMPLETE**  
> 日期：2026-08-14；**2026-08-15 control-referenced 修订**（见 §14）  
> Parent：Topic 5.1 multiscale effective propagation scaffold v0.5  
> 结果根：`results/topic5_latent_propagation_landscape_v0_2/`

## 0. 朴素话摘要（先读这段）

我们把一批已经训练好、**这一轮完全不再改动**的循环网络当成"读心装置"：给它看某位患者一次间期事件里
前几个触点被点亮的先后顺序，问它——事件走到哪一步了、接下来会往哪片脑区蔓延、如果伸手把它的内部状态
推一下，它接下来生成的蔓延图案会不会跟着变。

先说这一轮真正立住的东西。我们准备了四种**连线方式完全不同**的网络（只让近邻连、近邻加学出来的额外
近邻、随机远程连、近邻加学出来的远程连），每一种都用同一位患者的真实事件顺序训练过；另外还有一种把
"前半段和后半段的对应关系打乱"之后训练的对照网络。然后我们不去看它们的连线，而是问一个功能问题：
**在某个真实状态上，轻轻扰动一小片组织，接下来哪些触点的输出会变、变多少？** 把这张"扰动某片组织 →
未来哪些触点响应"的表格拿出来两两比较，结果是：四种真实顺序网络之间**很像（0.75）**，而它们跟打乱
顺序的对照网络**明显不那么像（0.64）**，28 位患者里 24 位是这个方向。更强的一步是：**拿其中三种网络
的平均去预测第四种没参与平均的网络**，也比预测打乱顺序的对照网络准（28 人里 23 人）。这张表格是纯几何
扰动做出来的，**完全没有用到任何拟合出来的方向轴**，所以"大家都被要求解释同一个标签所以自然像"这个
解释在这里不成立。

更进一步：把这张表格换算到触点空间（"扰动某个触点读到的组织 → 未来哪个触点响应"），拿去跟**这位患者
留出来、模型从没见过的真实事件**里"谁跟在谁后面"的统计比。**在保留电极杆归属的随机基线下仍然为正**
（+0.068，28 人里 21 人，校正后 p=0.003）——也就是说这不只是"同一根杆上的触点当然相关"。

但**"这张表格是这位患者独有的"这一条没有立住**：跟别的患者的传播统计比时，别人的数据要先经过一层
空间平滑才能搬到本患者的触点上，而本患者自己的数据没经过这层平滑。把这层平滑对两边配平之后，优势从
+0.235 掉到 +0.086，置信区间跨过 0，28 人里只有 17 人为正。所以现在能说"这个算子抓到了真实的传播
规律"，还不能说"抓到的是这位患者特有的传播规律"。

**看不出的**：给真实顺序并没有让网络更早确定"将往哪去"；推一下状态并没有干净地分别改变"进程"和"方向"；
原来以为"网络会沿着两个方向搬运状态、把偏离压回去"是学来的本事，跟打乱顺序的对照网络一比，对照网络
一样能做到——那是这类网络的结构性质。

最后一个必须单独讲的乌龙：预先写死的输出坐标是"越早被点亮值越大"，而"把状态往前推"天然抬高**更晚**
被点亮的触点，符号是反的，所以那个预注册对比在设计时就注定拿不到正号。翻过来看同一批数字变成强正向，
但那是事后翻的，只能当下一轮的假设。

**一句话**：具体连线图认不出来，但"面向未来传播场的那套算法"可以跨连线方式稳定地算出同一个东西，而且
它对得上留出数据里的真实传播规律；它是不是这位患者独有的，还没证出来。

（内部归档代号：C1–C7 claim ladder、`D_progress`/`D_field`、`v_data_prog` earlyness orientation、
C-suffix order-shuffled arm、ALL_CONTACT/WITHIN_SHAFT/DISTANCE_BIN spatial nulls、
`PATCH_OPERATOR_R1` topology-consensus finite-time propagation operator。）

## 1. 一句话总判断

**本阶段的科学重心不是"找到了一条 progress axis"，而是：有序间期序列不能唯一确定一张 recurrent
connectivity graph；可辨识的对象是一个跨 topology 稳定实现的 future-field 有限时间功能计算。**

冻结 RNN 的 hidden state 同时编码 event 的 ordinal phase 与 continuous future field；但 phase 相关的几何、
tangent transport 和 transverse contraction 在打乱 prefix–suffix 对应的 C-suffix 对照臂里同样成立，因此
它们是这类 leaky recurrent 架构的通用性质，不是真实顺序带来的。真正承重的是两条收敛结果：

1. **axis-fitted**：不同真实顺序 topology 的 future-field finite-time response field 两两相似度 `0.869`，
   对 C-suffix 只有 `0.528`，patient-level margin `+0.3345`（CI `[0.265,0.593]`，27/28）；
2. **axis-free（2026-08-15 补实验）**：改用纯几何 tissue-patch 扰动构造的 `tissue patch → future contact`
   有限时间响应算子，同一比较仍成立（reliability-corrected margin `+0.0758`，CI `[0.048,0.164]`，24/28），
   且三种 topology 的 consensus 能预测第四种 held-out topology（`+0.0693`，23/28）。第二条排除了
   "所有 arm 的 hidden axis 都被拟合去解释同一个 future-field 标签，因此自然相似" 这一构造性解释。

该 consensus 算子映射到触点空间后，与本患者**留出**事件的 prefix→suffix 传播统计对齐，在保留电极杆归属的
空间零模型下仍为正（`+0.0676`，21/28，Holm `0.0034`）。但把身份 null 的一维平滑对两侧配平后，患者特异
margin 降到 `+0.0858` 且 CI 跨 0（17/27），因此 **patient specificity 未成立**。

未成立的还有：真实顺序没有带来更早的 future-field commitment；轨迹没有向条件流形收敛；预注册双轴
perturbation double dissociation 未通过；C5 预注册 progress 方向因语义反号不可辨识；SNN 输入无资格；
early-ictal exploratory alignment 未确认。

**2026-08-15 修订新增的两条边界**（原文缺失，见 §14）：

1. C2 的 tangent transport 与 transverse contraction 只相对 0 成立；相对同一患者的 C-suffix
   order-shuffled arm，四个 primary endpoint 的配对差在 Holm 后全部 UNSUPPORTED
   （progress transport `+0.0013`、field transport `+0.0081`、transverse contraction `+0.0058`）。
   因此它们是**架构性质，不是 order-specific 结果**；`transverse_contraction` 按 spec §5.7 的
   「绝对 gain <1 **且**低于 controls」两条件规则改判 UNSUPPORTED。
2. C4 必须按轴分开写：future-field 轴 margin `+0.3345`（CI `[0.265, 0.593]`，27/28）承重；
   progress 轴 margin `+0.0095` 的中位数 CI 为 `[-0.0050, +0.0492]`，**跨过 0**，且 order-shuffled
   arm 已经达到 0.808 / 0.873 的相似度。

本轮另发现一个必须独立记录的 spec 语义问题：预注册 `v_data_prog` 是静态 **earlyness** field，而正向
hidden tangent 指向更晚 phase。沿 progress 正向扰动提高 later-contact logits 时，投影到 earlyness field
自然可能为负。因此原 C3/C5 primary 保持阴性不改，但 target-free post-hoc laterness sensitivity 呈正，
下一轮应先冻结并确认正确的 progress-output orientation，再决定是否进入 E3。

## 2. 执行范围与工程验收

- 28 位患者、42 fits、630 checkpoint cells；
- 531 个 v0.5 formal cells + 99 个 v0.3 exact-reuse cells；
- 18,900 replay events、187,305 replay steps；
- Pass 1 response-blind sample 为 77,617 events；
- Pass 2 冻结 119,655 个完整 `q=(h,r,k)` reference states；
- axis/control/chord perturbation audit 全通过；
- patch freeze 14,725,260 state–center pairs，主剂量双符号支持 8,813,069；
- patch response 25,585,396 eligible state–center–dose pairs，91,169,079 finite tau branches；
- C7 为 17 patients、167 seizures、334 seizure-axis rows；
- **补实验（2026-08-15）**：patch operator R1 全量重抽取 630/630 cells，`eligible_state_center_dose_pairs`
  与 `finite_state_center_dose_tau` 与冻结的 R0 阶段逐位一致（25,585,396 / 91,169,079），且把 R1 算子
  投影回冻结的 train-only contact axes 可复现 R0 的 `mean_scores`，全队列最大偏差 `2.09e-07`（float32 舍入）；
- 19 项实现测试通过；spec §15 列出 30 项必须实现的测试，其中 **12 项有直接回归测试、18 项仅由所属
  stage audit 的 `PASS` 状态背书、0 项无任何证据**（逐项映射见 `CLOSEOUT_AUDIT.json::spec_test_coverage`）。
  stage-audit 状态是弱证据：它说明该阶段跑完且没有记录到失败，不等于该 invariant 被单独检验；
- parameter-hash 不变性由 630/630 个 per-cell `model_hash_unchanged` / `decoder_hash_unchanged` 实际扫描确认；
- 最终 machine closeout audit 为 `PASS`，未解决 failure 为 0。

12 个 `RECOVERED_FAILURE.json` 是首次 C-suffix reference-freeze 的 fail-closed 留痕；修复后相同冻结合同
全量重跑并通过 630/630 hash、shape、support 与 target-blind audit。它们不从结果树删除，也不计作未解决
failure。

## 3. C1–C9 claim ladder

| Claim | 状态 | 核心结果 | 安全解释 |
|---|---|---|---|
| C1 two-coordinate geometry | **UNSUPPORTED（复合）** | progress、future-field incremental decoding 均支持；real−C-suffix early emergence 不支持 | hidden state 可解码 task variables，但真实顺序没有让 future-field commitment 更早出现 |
| C2 dynamical transport | **UNSUPPORTED（复合）** | progress/field tangent transport 相对 0 支持；transverse contraction 按两条件规则改判 UNSUPPORTED；conditional-manifold convergence 为负；四项相对 order-shuffled arm 全部 UNSUPPORTED | 局部方向被运输，但这是架构性质、非 order-specific，更不能称 attracting propagation channel |
| C3 axis perturbation | **UNSUPPORTED** | `D_progress<0`；`D_field>0` 但 Holm 后 0.0505；high-u empirical chords 为正 secondary | 预注册双重解离未闭合；empirical transplantation 只作 model-internal secondary |
| C4 topology convergence | **SUPPORTED（仅 future-field 轴）** | future-field margin `+0.3345`（CI `[0.265,0.593]`，27/28，Holm 1.5e-8）；progress margin `+0.0095`（CI `[-0.0050,+0.0492]`，18/28，Holm 0.0387）跨 0 | static topology 不可辨识时，不同 real-order topologies 在 future-field 响应上收敛；progress 轴不承重 |
| C5 patient-specific data alignment | **UNSUPPORTED（primary）** | earlyness-signed progress 与 field primary family 未通过；laterness post-hoc 为强正向，但在 shaft/distance 保结构 null 下降到约一半 | 不能正式写 patient-specific validation；sign sensitivity 是新确认实验的依据，引用效应量必须带 null family |
| C6 RNN–SNN convergence | **NOT IDENTIFIABLE** | 两个 SNN source 都是 single-patient diagnostic-only，field values 未打开 | 不是统计阴性，也不能展示单患者漂亮对齐替代 cohort evidence |
| C7 early-ictal alignment | **EXPLORATORY COMPLETE / UNSUPPORTED** | progress earlyness −0.048；laterness +0.048；field +0.018，均未确认 | locked internal exploratory only，不称 prediction/recruitment/confirmation |
| C8 axis-free operator convergence（2026-08-15 补实验） | **SUPPORTED** | 几何扰动算子 real-real `0.7476` vs real-vs-shuffled `0.6407`；reliability-corrected margin `+0.0758`（CI `[0.048,0.164]`，24/28，Holm 4.1e-5）；leave-one-topology-out `+0.0693`（23/28） | 收敛不再可能由"共用同一个 future-field 标签"解释；phase invariance `0.918` 说明它接近一张与事件阶段无关的固定图案 |
| C9 operator ↔ 留出数据 | **PARTIAL** | 保留电极杆的空间零模型下 margin `+0.0676`（CI `[0.018,0.128]`，21/28，Holm `0.0034`）→ **对齐成立**；平滑配对后的患者身份 margin `+0.0858`（CI `[-0.044,0.148]`，17/27）→ **患者特异未成立** | 可以写"算子复现了留出数据里的传播规律"，不能写"这套规律是这位患者独有的" |

## 4. Progress 的最终定位：ordinal phase 是对照坐标，不是病理轴

Progress label 是事件内部的归一化序号：

```text
s = (k - 1) / (K_e - 1)
```

它**不表示**秒、生理传播速度、被招募的组织距离、seizure recruitment fraction，或系统离发作还有多远。
它高度依赖一次事件最终被多少触点观察到、电极覆盖、检测阈值、rank-set 的离散方式、STOP decoder 和事件
截断方式。因此本轮起，它的正式名称是 **ordinal event-phase coordinate**，不是 propagation-progress axis。

它的 held-out 增量确实明显（`ΔR² = +0.1442`，25/28 为正），说明冻结 hidden state 里存在当前 contact
configuration 之外的序列阶段信息。但 progress transport、transverse contraction 和 progress 轴上的
topology convergence **都能被 C-suffix 对照臂复现**。所以它在本轮的作用固定为：

- **nuisance coordinate**：先把事件阶段分离出去，才能问"在相同事件阶段下，hidden state 是否仍然包含未来
  空间模式"，否则 future-field 方向可能只是"某一类事件通常更长 / 采样更偏早期 / 离 STOP 更远"的影子；
- **negative-control coordinate**：它同时证明了一件有价值的事——**并不是任何容易解码的 latent axis 都表现出
  order-dependent convergence**。progress 轴不表现，future-field 轴表现。

因此本轮**不再**把"重新确认 progress 输出方向"当作下一阶段的主任务，也不让它阻断 future-field 或 E3 分支。

## 5. C1：存在可解码几何，但没有 order-specific early commitment

所有可辨识 fits 的 generic tier（28 patients）中：

- progress `P−O` heldout `R²` 增量中位 `+0.14422`，25/28 为正，Holm `P=1.88e-6`；
- future field `PF−P` 中位 `+0.01012`，27/28 为正；
- future field `PF−PF-null` 中位 `+0.01070`，27/28 为正；
- early emergence 的 real-order−C-suffix 中位 `−0.000607`，10/28 为正，Holm `P=0.922`。

14 位 canonical A/B shared-fit 患者得到同样结构：progress 和 future-field geometry 可解码，但 real order
没有比 C-suffix 更早进入 A↔B future-field state。因此本轮可以写 raw hidden states jointly encode phase and
future-field information，不能写 true prefix–suffix association creates an earlier latent commitment。

## 6. C2：局部动力学运输方向，但既不形成吸引通道、也不是学出来的

generic tier 的 patient-level 中位数（绝对值 / 相对同一患者 C-suffix order-shuffled arm 的配对差）：

| endpoint | 绝对中位数（vs 0） | 相对 order-shuffled arm | 正号数 | Holm（control family） |
|---|---|---|---|---|
| progress tangent transport cosine | `0.7867` | `+0.00133` | 14/28 | `0.416` |
| field tangent transport cosine | `0.8955` | `+0.00805` | 18/28 | `0.208` |
| transverse contraction | `+0.1473` | `+0.00583` | 17/28 | `0.290` |
| event-to-conditional-manifold convergence | `−0.03314`（1/28 为正） | `+0.00136` | 15/28 | `0.305` |

两点必须一起读：

1. transverse contraction 的绝对值确实支持 gain `<1`（`1 − 0.853`，CI `[0.095, 0.239]`），但 spec §5.7 对该
   leg 的定义是「绝对 gain `<1` **且**低于 controls」，后半条不成立，因此该 endpoint 现判 UNSUPPORTED。
2. 四个 endpoint 相对 order-shuffled arm 的配对差全部不显著 → `order_specificity = NOT_ORDER_SPECIFIC`。
   一个 leaky RNN 的单步 Jacobian 本身接近 `(1−κ)I + κ·diag(sech²)W`，对任何光滑轴族都会给出高 transport
   cosine，因此「cosine 远大于 0」不是学习到结构的证据。

Teacher-forced 与 deterministic closed-loop transition 的 progress Spearman 为 `0.9951`、field 为 `0.7583`，
说明两种推进方式在低维投影上高度一致；但 closed-loop manifold convergence 仍为 `−0.06413`，28/28
均不朝训练条件流形靠近。因此本轮只能写「局部动力学会沿拟合出的方向搬运状态、并保持横向 gain 低于 1，
且这些性质在打乱顺序的对照臂里同样存在」，不能写 attracting trajectory / propagation channel，也不能写
这是真实顺序带来的。

预注册的另外三族 transport control（phase-shuffled axes、event-shuffled axes、high-variance PCA
directions，spec §5.7）在本轮**未实现**；只有 C-suffix arm 这一族可用。这是一个已知实现缺口，见 §14。

## 7. C3/C4：双轴 primary 未通过，但 axis-fitted topology convergence 清楚

generic tier 的 co-primary：

- `D_progress` 中位 `−0.20377`，0/28 为正；
- `D_field` 中位 `+0.22535`，18/28 为正，raw `P=0.02525`，Holm `P=0.05050`；
- canonical tier 的两项也均未通过。

matched-observable empirical state transplantation 是一项正向 secondary：high-u 相对 small-u chords 的
open、closed 和 terminal future-field response 分别为 `+0.18875`、`+0.24131`、`+0.01775`。这说明真实访问过
的、future-field coordinate 不同的 hidden-state difference 能推动生成 future field，但它不能替代预注册
axis-specific double dissociation。

C4 按轴分开裁定：

| 响应轴 | real-arm 两两相似度 | real-vs-order-shuffled 相似度 | topology margin | margin CI95 | Holm |
|---|---|---|---|---|---|
| future field | `0.8691` | `0.5275` | `+0.33451`（27/28） | `[0.2648, 0.5931]` | `1.49e-8` |
| progress | `0.8731` | `0.8077` | `+0.00945`（18/28） | `[-0.0050, +0.0492]` | `0.0387` |

future-field 轴承重；progress 轴的 margin 中位数 CI 跨过 0，且 order-shuffled arm 已经拿到 0.808 的相似度，
说明 progress 响应场基本由固定观测算子和触点几何决定，不区分 topology 也不区分顺序。因此本轮最强机制层
结果只能写成：

> Distinct recurrent topologies converge on similar finite-time **future-field** responses despite non-identifiable static connectivity.

不得省略 `future-field` 而写成泛指的 functional responses。

## 8. Progress sign-semantics audit

预注册输出轴：

```text
v_data_prog = centered mean start-removed rank field
```

其中 `1-normalized_rank` 越大表示越早；正向 hidden tangent 则定义为 `gamma(s+delta)-gamma(s)`，表示向更晚
phase 推进。这使正向 progress response 对 earlyness field 的投影存在确定的符号冲突。

不改变 primary 的 target-free post-hoc audit 显示：

- `D_progress_laterness` generic tier 中位 `+0.02958`，16/28 为正，raw `P=0.0407`，但 **CI95
  `[-0.0157, +0.0947]` 跨过 0**；canonical A/B tier 中位仅 `+0.00762`，7/14，`P=0.195` —— 即**双轴
  perturbation 的解离在翻符号之后依然没有成立**，翻符号能救的只是 C5 的对齐方向；
- sign-invariant `|progress|-|field spillover|` 中位 `+0.04514`，19/28 为正；
- C5 progress spatial-null margin 从 earlyness `−0.47083` 精确变为 laterness `+0.47083`，25/26 为正；
- C5 progress identity margin 从 `−0.20946` 变为 `+0.20946`，20/26 为正；
- canonical 14 人 laterness spatial/identity margins 也均为正方向。

**效应量强烈依赖空间零模型的选择**（spec §9.2 要求报告，原文缺失，2026-08-15 补齐）。progress 轴
laterness margin 的 patient-level 中位数：

| 零模型族 | generic n=26 | 正号数 | `P` | canonical n=14 |
|---|---|---|---|---|
| synchronized all-contact（预注册 primary） | `+0.4708` | 25/26 | `4.5e-8` | `+0.4052` |
| graph-spectral autocorrelation | `+0.4771` | 25/26 | `7.5e-8` | `+0.5088` |
| within-shaft | `+0.1982` | 23/26 | `6.4e-7` | `+0.1786` |
| distance-bin local | `+0.1792` | 22/26 | `5.5e-6` | `+0.1381` |

方向在四族里都稳（22–25/26，`P ≤ 6e-6`），但**保留电极杆归属或保留局部距离结构之后，效应量掉到约
0.18–0.20，只有预注册 primary 数值的 40%**；也就是说这个对齐里约六成由粗空间结构（哪根杆、离得多近）
解释。原因是预注册 primary 用的 synchronized all-contact 置换会打散一切空间结构，其 null median ≈ 0
（全队列 `|observed − margin| ≤ 0.029`），所以那一档的「null-relative margin」在数值上就等于原始 signed
Spearman。future-field 轴在四族里全部为空（`+0.035` ~ `+0.099`，`P` 0.16–0.41）。

identity null 另有一个不对称：cross-patient 一侧的场都经过归一化传播轴上的一维核平滑，而 same-patient
一侧保持原始触点分辨率。用**同一核、同一 per-pair 带宽**对患者自己的场做 smoothing-matched 对照后，
progress identity margin 由 `−0.2095`（6/26）变为 `−0.1851`（7/26），laterness 侧为 `+0.1851`，19/26，
`P=0.0031`。效应仍在，但原数值被平滑不对称抬高了约 12%。

这不是事后选择患者/phase/axis，也发生在 early-ictal target 解封前；但它是在看到 C3/C5 符号异常后增加的
semantic audit，因此只能写作 hypothesis-generating。科学上最准确的表述不是“没有 patient-specific
progress field”，而是：

> 本轮预注册 progress-output estimand 方向不合语义；正确 laterness orientation 显示出值得独立确认的患者特异信号，但没有资格救回 C3/C5 primary。

## 9. Spatial patch field

Gaussian tissue patch 使用全部 tissue nodes 作为中心，宽度固定为两倍局部 node spacing，剂量为
`0.25/0.5/1.0 local SD`，两种符号都必须通过同一 N0。主剂量 response field 的 real-arm cosine 中位：

- progress：`0.6976`；
- future field：`0.8141`。

跨剂量 cosine 为 `0.9953`（0.25↔0.5）和 `0.9751`（0.5↔1.0）；patient-level sign agreement 约
`0.82`。这说明 model-internal patch susceptibility maps 数值稳定，但它们不是 tissue stimulation maps，
也没有被 C5/C6 外部验证。

## 10. 补实验：topology-consensus finite-time propagation operator（2026-08-15）

### 10.1 为什么需要它

第 7 节的 axis-fitted convergence 有一个构造性依赖：每个 arm 的 hidden future-field direction 都是拟合去
解释**同一个**患者级 `u_e` 标签的，所以"不同 arm 的响应场相似"里天然包含"它们都被要求编码同一个监督变量"
这一部分。本补实验换一个完全不依赖任何拟合方向的扰动来重问同一个问题。

### 10.2 定义

对患者 `p`、arm `a`、tissue patch center `i`、future contact `c`、phase `s`、future lag `tau`：

```text
K_{p,a}(c, i | s, tau) = mean over frozen reference states of
                         [ l^{+i}_{k+tau, c} - l^{-i}_{k+tau, c} ] / (2 * dose)
```

扰动方向是冻结的 Gaussian tissue patch（纯几何），**不使用 future-field label、不使用 progress orientation、
不使用任何 hidden axis**。primary 取 `dose = 0.5 local SD`、`tau ∈ {1,2,3}`、三个 phase 平均。

与审阅意见的一处偏离：状态维用 **mean** 而非 median。原因是冻结的 R0 阶段本来就存 state mean，而投影是
线性的，因此 R1 算子投影回冻结 contact axes 必须**逐位复现** R0 的 `mean_scores`——这条恒等式就是重抽取
忠实性的检验（全队列最大偏差 `2.09e-07`）。

**残留的轴依赖（必须写明）**：每个 state–patch 对是否合格由 support gate 决定，而 gate 在各 arm **自己**
拟合出的坐标空间里评估，因此"哪些对进入平均"弱依赖于 arm。响应量本身不含任何拟合轴。

### 10.3 算子是一张接近固定的图案

- phase invariance（三个 phase 两两相似度的患者中位数）：`0.9179`；
- dose consistency（`0.25` 与 `0.5 local SD` 之间）：`0.9958`；
- split-half reliability（按参考态序号奇偶分半，Spearman-Brown 校正）：real arms `0.9962`、C-suffix `0.9967`。

reliability 接近 1 有一个重要后果：**"C-suffix 相似度低只是因为它更吵" 这个解释被排除了**。

### 10.4 跨 topology 收敛（C8）

| endpoint | 中位数 | CI95 | 正号数 | Holm |
|---|---|---|---|---|
| real-real 两两相似度 | `0.7476` | — | — | — |
| real vs order-shuffled 相似度 | `0.6407` | — | — | — |
| raw margin | `+0.0757` | `[+0.0475, +0.1644]` | 24/28 | `4.13e-5` |
| reliability-corrected margin | `+0.0758` | `[+0.0483, +0.1644]` | 24/28 | `4.13e-5` |
| leave-one-topology-out margin | `+0.0693` | `[+0.0385, +0.1833]` | 23/28 | `4.13e-5` |

最后一行是最关键的一步：用其余三种 topology 的 consensus 去预测**没有参与构造 consensus** 的第四种
topology，仍然比预测 C-suffix 更准。这排除了"每个网络用自己拟合的轴证明自己内部自洽"。

判据与 C4 一致：中位数 `>0`、Holm `<0.05`、**且 bootstrap 中位数 CI 不跨 0**，三条同时满足才记 SUPPORTED。

### 10.5 与留出数据的对齐（C9）

用冻结的观测算子把 tissue-patch 轴映射到触点空间（`read_weight[i,c] = <p_i, H[c,:]>`，逐列归一），得到
`扰动某触点读到的组织 → 未来某触点响应`。经验对照量来自本患者**留出** test events 的
`follow-within-lag-1..3` 频率，减去 within-event rank-shuffle 的期望（该期望有闭式解，按每个事件自身的
rank 多重集算，无需重采样）。

| endpoint | 中位数 | CI95 | 正号数 | 判定 |
|---|---|---|---|---|
| consensus ↔ 留出传播（原始 Spearman 中位数） | `0.1804` | — | — | 描述 |
| all-contact 置换 margin | `+0.2006` | `[+0.115, +0.285]` | 23/28 | 敏感性（该 null 打散全部空间结构） |
| **within-shaft margin（primary）** | `+0.0676` | `[+0.018, +0.128]` | 21/28 | **SUPPORTED**（Holm `0.0034`） |
| distance-bin margin | `+0.0635` | `[+0.017, +0.145]` | 21/28 | 敏感性 |
| consensus 减 order-shuffled arm 的算子 | `+0.1097` | `[+0.024, +0.199]` | 20/28 | 敏感性 |
| identity margin（未配平平滑） | `+0.2348` | `[+0.139, +0.361]` | 22/28 | **不作为 primary** |
| **平滑配对后的 identity margin（primary）** | `+0.0858` | `[−0.044, +0.148]` | 17/27 | **UNSUPPORTED**（CI 跨 0） |

两条必须一起读：

1. **对齐成立**。保留电极杆归属之后 margin 仍为正，说明这不只是"同一根杆上的触点当然彼此相关"。但
   all-contact 那一档（`+0.20`）是把空间结构全打散的基线，**不是空间已控制的效应量**；控制后约为其三分之一。
2. **患者特异未成立**。cross-patient 一侧的传播统计要先经过归一化传播轴上的一维核平滑才能搬到本患者
   触点上，本患者自己的一侧没有这层平滑。用同核、同 per-pair 带宽给本患者做 self-transport 配平后，
   优势由 `+0.235` 降到 `+0.086` 且 CI 跨 0。**这与本轮早些时候在 C5 identity null 上发现的是同一个
   不对称**，这里在设计阶段就配平了，不是事后补的。

### 10.6 没做的那一步

审阅意见里的第五步（删除 / 翻转 consensus 主分量，看是否选择性损害 suffix 预测）**未执行**。它需要在
hidden 空间构造一条与该算子对应的方向并重新 rollout，超出"只用冻结产物"的边界。因此本轮只有**充分性
方向**的证据，**没有必要性证据**；措辞不得写成 "carries the computation"。

## 11. C6：SNN 不可辨识

只读取 producer metadata 后：

- D5.2：E1146、6 networks、8 s、development confirmation；
- D6.3：E1146、12 networks、16 s、fresh-network replication failed；
- 两者都缺显式 `runtime_mode`、≥20 s late-runaway contract、case-series/cohort denominator 和预冻结
  RNN↔SNN field mapping。

因此 `SNN_ALIGNMENT_NOT_IDENTIFIABLE`，SNN field values 未打开。不能把此状态写成 RNN–SNN 不一致，也
不能用 E1146 单病例图代替 C6。

## 12. C7：early-ictal exploratory 未确认

全部 target-free fields、patch maps、geometry mapping、SNN eligibility、sign audit、scorer 与 null manifests
冻结并哈希后，才解封既有 17 patients/167 seizures 的 clinical-onset 后 0–10 s、1–150 Hz broadband
energy field。固定 prediction 不选 best axis/phase/mode。

患者级 all-contact null-relative margin：

- progress earlyness：`−0.04762`，6/17 为正，`P=0.824`；
- target-free frozen laterness sensitivity：`+0.04762`，11/17 为正，`P=0.189`；
- future field：`+0.01786`，9/17 为正，`P=0.391`。

identity margin 也均未确认。该结果只写 locked internal exploratory；不能写 interictal control fields predict
early-ictal recruitment。

## 13. 最终允许与禁止措辞

### 主文允许

> Ordered interictal sequence models did not identify a unique recurrent topology. Instead, distinct
> recurrent implementations converged on a common finite-time future-field response, and this convergence
> was markedly reduced after prefix-suffix associations were disrupted. It also holds for a geometry-only
> tissue-patch response operator that uses no fitted hidden axis, and a consensus built from three
> topologies predicts a held-out topology. The identifiable observation is therefore an order-sensitive
> finite-time response pattern rather than a unique connectivity graph; necessity of a shared low-rank
> internal computation was not established.

中文：

> 有序间期序列不能唯一确定一张 recurrent connectivity graph；多种不同 recurrent implementation 会收敛到
> 相似的 future-field 有限时间功能响应，破坏 prefix–suffix 对应后收敛明显减弱。该收敛在**完全不依赖拟合
> 方向**的几何扰动算子上同样成立，且三种 topology 的 consensus 能预测第四种 held-out topology。因此当前
> 可辨识的是**顺序敏感的有限时间响应规律**，不是唯一连接图；一个跨网络共享、低秩且预测所必需的内部计算
> 尚未建立。

### 可以作为辅助结果

> Frozen recurrent states encode an ordinal event-phase coordinate and continuous future-field information.
> The ordinal phase behaves as a generic sequence coordinate that the order-shuffled control reproduces,
> whereas convergence is selective for the future-field response.

> The consensus operator matches the same patient's held-out interictal propagation transitions beyond a
> shaft-preserving spatial null; its patient specificity is not established once the identity null's
> smoothing is matched on both sides.

### 禁止

- 找到了患者固定的解剖连接通路 / pathological wiring / true patient connectome；
- progress 是癫痫病理轴，或 progress 坐标表示秒、传导速度、组织招募范围；
- TA/TB 是两个真实 attractors；
- RNN 已识别 biological attractor 或 attracting propagation channel；
- progress 与 future-field axes 已完成因果双重解离；
- tangent transport / transverse contraction / progress 轴 topology convergence 是 order-specific 或学出来的；
- functional response field 或 consensus operator 已被确认是 patient-specific；
- 该 operator "carries / 承载" 了计算（2026-08-16 的留一网络删除实验未建立选择性 necessity）；
- RNN 与 SNN 已完成 cohort-level convergence；
- interictal field 已预测 / 确认 early-ictal recruitment；
- early-ictal broadband energy 等同组织 recruitment；
- post-hoc sign reorientation 已救回 C3/C5；
- 把 synchronized all-contact 那一档（progress `0.47` / operator `0.20`）当成"空间已控制"的效应量。

## 14. E3 决策

不再以"再确认 progress 输出方向"作为 E3 前提。当前决策状态：

```text
GENERIC_PROGRESS_GEOMETRY_CLOSED
FUTURE_FIELD_FUNCTIONAL_CONVERGENCE_SUPPORTED
DIRECT_OPERATOR_TO_DATA_LINK_ALIGNMENT_SUPPORTED_PATIENT_SPECIFICITY_PENDING
SHARED_COMPONENT_NECESSITY_UNSUPPORTED
E3_SMOOTH_SUSCEPTIBILITY_NOT_YET_LAUNCHED
```

E3（把高维 operator 压成低参数 smooth susceptibility field）的门槛现在是**患者特异那一条**，不是 progress
方向。最短下一步按优先级：

1. **把患者特异做干净**：identity null 必须在设计阶段就对两侧施加同一 registration/smoothing 算子；
   预注册 effect size 以 shaft-preserving 档为准（约 `+0.07`），不以 all-contact 档（`+0.20`）立门槛；
2. 补齐 spec §5.7 缺失的三族 transport control（phase-shuffled / event-shuffled / high-variance PCA），
   并把"相对 order-shuffled arm 的配对差"写进 C2 的 primary 判据；
3. necessity 已于 2026-08-16 完成但未通过；即使患者身份后续闭合，E3 也只能作为低参数描述模型，不能用
   "必要共同计算"作为启动理由。

## 15. 核心工件

- `results/topic5_latent_propagation_landscape_v0_2/CLAIM_LADDER_ADJUDICATION.json`
- `results/topic5_latent_propagation_landscape_v0_2/CONTROL_REFERENCED_ADDENDUM.json`
- `results/topic5_latent_propagation_landscape_v0_2/spatial_control_field/patch_operator/PATCH_OPERATOR_SUMMARY.json`
- `results/topic5_latent_propagation_landscape_v0_2/spatial_control_field/patch_operator/OPERATOR_{TOPOLOGY_CONVERGENCE,LEAVE_ONE_OUT_CONSENSUS,DATA_ALIGNMENT,PHASE_INVARIANCE,DOSE_CONSISTENCY}.csv`
- `results/topic5_latent_propagation_landscape_v0_2/shared_functional_computation_necessity_v0_2/{CLAIM_ADJUDICATION,FINAL_AUDIT,SUBSPACE_SENSITIVITY_SUMMARY,SECONDARY_SUMMARY}.json`
- `results/topic5_latent_propagation_landscape_v0_2/C5_SPATIAL_NULL_FAMILY_PATIENT_EFFECTS.csv`
- `results/topic5_latent_propagation_landscape_v0_2/C5_SMOOTHING_MATCHED_IDENTITY.csv`
- `results/topic5_latent_propagation_landscape_v0_2/COHORT_PATIENT_TABLE.csv`
- `results/topic5_latent_propagation_landscape_v0_2/CLOSEOUT_AUDIT.json`
- `results/topic5_latent_propagation_landscape_v0_2/axis_perturbation/responses/PROGRESS_SIGN_SEMANTICS_AUDIT.json`
- `results/topic5_latent_propagation_landscape_v0_2/spatial_control_field/patch_response/SPATIAL_PATCH_CONTROL_SUMMARY.json`
- `results/topic5_latent_propagation_landscape_v0_2/SNN_INPUT_ELIGIBILITY.json`
- `results/topic5_latent_propagation_landscape_v0_2/early_ictal_exploratory/EARLY_ICTAL_SUMMARY.json`
- `results/topic5_latent_propagation_landscape_v0_2/paper-ready-figure/latent_landscape_candidate/figures/`

候选图已完成同状态 PNG/PDF 视觉验收与 SVG hash/结构验收，但未分配 paper registry slot，因此保持
closeout candidate，不称 canonical Figure 6。

## 16. 修订记录

### 16.1 2026-08-15 第一轮：control-referenced 修订

本次修订**不重放任何模型、不重选任何 event、不改动任何预注册 primary 的判决**；全部数字由已冻结的
per-patient CSV 重新聚合得到，产出物为 `CONTROL_REFERENCED_ADDENDUM.json` 及两个 patient-level CSV。

| 编号 | 问题 | 处理 |
|---|---|---|
| P0-1 | C2 四个 primary endpoint 只对 0 检验；spec §5.7 要求与 phase-shuffled / event-shuffled / PCA / C-suffix 方向比较，且 `C_perp` 需「gain <1 **且**低于 controls」。`C2_PATIENT_EFFECTS.csv` 里的 `*_real_minus_C_suffix` 列从未进入任何 summary | 在 `summarize_topic5_latent_transport_v0_2.py` 中加入 control family（含 Holm），`transverse_contraction` 改为两条件判据 → SUPPORTED 变 UNSUPPORTED；新增 `order_specificity` 字段并传入 claim ladder。三族未实现的 control 记录为已知缺口（§5 末） |
| P0-2 | C5 只把 synchronized all-contact 一族写进 patient-level summary；spec §9.2 要求同时报告 shaft / distance / autocorrelation 保结构族。该族 null median ≈0，故 margin 数值上等于原始相关，效应量被高估约 2.4 倍 | 新增 `audit_topic5_latent_control_referenced_v0_2.py`，把 `SPATIAL_NULL_ALIGNMENT.csv` 已有的四族聚合到 patient level（两种 tier、两种朝向），写入 §7 表格、claim ladder 与 panel g |
| P1-3 | identity null 只对 cross-patient 一侧施加一维核平滑，same-patient 一侧未施加，构成平滑不对称 | 用同一核、同一 per-pair 带宽构造 self-transport 对照，得到 smoothing-matched identity margin（`−0.1851`，laterness `+0.1851`，19/26，`P=0.0031`） |
| P1-4 | C4 判据只看 `median>0 且 Holm<0.05`，未看 CI；progress 轴 margin CI 跨 0 而被并入"明确支持" | claim ladder 增加 `per_axis` 块与 `ci95_median_excludes_zero`，§3/§6 改为按轴分开裁定，允许措辞加上 `future-field` 限定 |
| P1-5 | §7 只给 generic tier 的 laterness `D_progress`，未给其跨 0 的 CI，也未给 canonical tier 的 null 结果 | §7 补齐两 tier 与 CI，并明确「翻符号救不回双轴解离，只影响 C5 对齐方向」 |
| P1-6 | `CLOSEOUT_AUDIT.json` 的 `contract_evidence_coverage` 把 28 个合同项映射到 13 个 stage-audit 状态（6 项同映射到 `pass1`），`parameter_hash_invariance` 定义为"其他 audit 全 PASS"，与 `all_stage_audits_pass` 同义，无法失败 | 替换为 spec §15 的 30 项显式覆盖表，逐项标注 `AUTOMATED_TEST` / `STAGE_AUDIT_STATUS_ONLY` / `UNCOVERED`；`parameter_hash_invariance` 改为实际扫描 630 个 per-cell `model_hash_unchanged` / `decoder_hash_unchanged`。当前：30 项中 12 项有直接回归测试、18 项仅由 stage-audit 状态背书、0 项无证据 |
| P1-7 | 候选图：panel g/i 把同一批数值乘 `−1` 后当作两条独立分布画（7 条里 4 条是 2 对镜像）；panel e 未标 earlyness 朝向；panel d 用 0 参考线画绝对 cosine；panel c 画两条重叠曲线而非配对差；panel h 是纯文字框；唯一成立的 C4 没有面板；统计面板无不确定性层与 n | 重画：c/d 改画配对差；e 行标注 earliness-signed 并加构造性说明；f 换成 C4 配对图；g/i 只画预注册朝向并改画四族零模型；h 换成资格判据表；全部统计面板补 bootstrap 中位数 CI 与 n。删除单患者 exemplar 面板及其孤儿 source 表 |

修订后重跑并通过：19/19 tests、13→14 项 closeout checks 全 PASS、630/630 parameter-hash 不变、
figure PNG/PDF/SVG 同状态视觉与 hash 验收（PDF 最小字号 5.6 pt）。所有既有 endpoint 的数值在重跑后
逐位一致，唯一变化是新增字段与 `transverse_contraction` 的状态标签。

### 16.2 2026-08-15 第二轮：按审阅意见重定重心 + 补实验

审阅意见的核心是：本阶段的科学重心应从 "progress axis" 移到 "future-field functional convergence"，
并补一个不依赖拟合方向的直接实验。据此：

| 动作 | 内容 |
|---|---|
| 重心 | progress 降级为 ordinal event-phase 的 nuisance / negative-control 坐标（新 §4），不再作为病理轴，也不再作为 E3 的前置门 |
| C4 措辞 | 全部加 `future-field` 限定；明确收敛对象是 finite-time functional response，不是 `W` / edge mask / anatomical pathway |
| 补实验 | 新增 topology-consensus finite-time propagation operator（新 §10）：patch operator R1 全量重抽取 630/630，跨 topology 收敛 + leave-one-topology-out + 与留出传播统计对齐 + 四族空间零模型 + 平滑配对身份 null |
| 新判据 | operator 两族 endpoint 的 SUPPORTED 需同时满足 中位数 `>0`、Holm `<0.05`、bootstrap 中位数 CI 不跨 0——与本轮对 C4 progress 轴施加的标准一致 |
| 结论修改 | 新增 C8（SUPPORTED）与 C9（PARTIAL：对齐成立、患者特异未成立）；E3 决策状态改为三段式，门槛从 progress 方向改为患者特异 |
| 未执行 | necessity 腿（consensus 主分量删除 / 翻转）需要新的 hidden-space rollout，超出冻结产物边界，明确记为缺口 |

**图未改动**。按用户 2026-08-15 的规定（复审默认不动图、要改图必须先问、不许靠加字补科学表述），本轮
只把新数字落到结果 JSON / CSV 与本归档文档；候选图仍是第一轮验收的版本，其 PNG/PDF/SVG hash 与
`FIGURE_VISUAL_QA.json` 记录一致。若要按审阅意见重排面板（progress 相关面板移入 Extended Data、
补 operator 面板），需用户单独批准。

**读现有候选图时需自带的三点**（因为图上没有画）：

1. panel e 的 `2x2` 是描述性 pooled median，预注册双重解离**未通过**；
2. panel f 的 topology convergence 是 **axis-fitted** 版本，其独立性检验在本文档 §10，不在图上；
3. panel g 的 all-contact 档是**未做空间控制**的基线，控制后约为其 40%。

### 16.3 2026-08-16 第三轮：共同响应成分 necessity 删除实验

上一轮 §10 的跨网络响应收敛是充分性证据，本轮用留一网络合同补上 hidden-state 删除实验。最终审计发现初版
v0.1 的删除中心与 support gate 使用了由留出事件完整后半段计算的 future-field coordinate；共同方向本身未读
留出 target，但删除幅度和分支资格受到目标信息影响，因此 v0.1 数值全部作废。v0.2 将中心固定为仅由训练事件
拟合的 phase curve，删除旧缓存的 outcome-derived fields，并全量重跑。

outcome-blind v0.2 主结果仍为 `NECESSITY_UNSUPPORTED`：共同成分绝对 dose-AUC 中位 `+0.000878`
（CI `[−0.000412,+0.002000]`，Holm `P=0.302`）；相对同范数无关方向 `−0.000561`
（Holm `P=0.781`）；相对打乱后半段网络方向 `+0.000050`
（Holm `P=0.552`）。只有 1/4 待测网络满足三项效应方向均为正，预注册要求为 3/4。

累计删除前 2–3 条共同方向后，绝对损失中位分别为 `+0.000763` 与 `+0.001380`，CI 仍跨 0；相对打乱网络
方向分别为 `−0.000026` 与 `−0.000150`。扰动当下存在一般输出损害，事件晚段有较强趋势，但均未建立共同方向
特异性。

因此 §10 的安全解释进一步收窄为：不同连接实现产生相似、可留一外推且与留出间期传播统计对齐的有限时域响应；
不能写这些网络依赖同一个低秩内部成分完成预测。最终方法、数值、P0 修复、工程审计和补充图见
`shared_functional_computation_necessity_v0_2_closeout_2026-08-16.md`。
