# Fig2 时序图形式规格

本文档是 Fig2 真实数据传播时序素材图的单一视觉合同。以后如果任务里出现“按 Fig2 时序图形式”“Fig2 时序图样式”“Fig2 temporal propagation panel”，先读本文件，再改图或跑图。

## 使用场景

这套图用于展示单个 subject 的间期 HFO group events 如何在时间上反复呈现传播顺序，并如何被无监督聚类压缩成两类传播模板。它是 Fig2 的 subject-level 素材图，不是 cohort-level 统计图，也不是模型/SNN readout 图。

本文件只规范 Fig2-A 的长时序 rank/cluster 素材。Fig2-C 的单事件包络 frame 与 TA/TB GIF 另见 `docs/fig2c_interictal_event_envelope_field_spec.md`；不得把两套 renderer、颜色或 claim 混用。

默认数据源是 masked phantom-rank 修正后的真实数据：

```bash
results/interictal_propagation_masked/
```

复现入口：

```bash
python scripts/plot_interictal_propagation.py --masked-features --pr3 --paper-style --max-events 2000
```

单 subject 预览：

```bash
python scripts/plot_interictal_propagation.py --masked-features --pr3 --paper-style --dataset epilepsiae --subjects 958 --max-events 2000
```

如果只想临时预览、不覆盖正式图，使用：

```bash
python scripts/plot_interictal_propagation.py --masked-features --pr3 --preview-style --dataset epilepsiae --subjects 958 --max-events 2000
```

正式输出：

```bash
results/interictal_propagation_masked/figures/per_subject/<dataset>_<subject>_propagation.png
```

预览输出：

```bash
results/interictal_propagation_masked/figures/per_subject/preview_per_subject/<dataset>_<subject>_propagation_preview.png
```

## 科学语义

这套图回答一个局部、可视化层面的科学问题：

> 该 subject 的间期 HFO group events 是否反复落在稳定的传播时序结构上，并且这些事件是否能被分成两类主要模板？

图上允许表达：

- group event 内部存在 channel-level first-to-last rank 顺序；
- 同一 subject 的大量事件在时间轴上反复出现类似的 rank pattern；
- KMeans 后，事件可被分为模板组，Fig2 主素材优先使用 stable k=2 subject；
- TA/TB 是该 subject 内两类传播模板的视觉别名，用于图上阅读，不等于跨 subject 的全局模板编号；
- 右侧 rank distribution / mean rank 只是辅助读出，不是独立统计检验。

图上不允许表达：

- 不把这张图写成 cohort-level 证明；
- 不把 KMeans 聚类本身写成机制解释；
- 不把 TA/TB 写成固定解剖方向、固定病灶方向或跨 subject 可直接相加的类别；
- 不把模型/SNN 输出混进这套真实数据图。

## 四个 panel 的布局

整体是 2 x 2 布局：左列宽、右列窄；上下两行间距紧凑；左右两列保持紧凑但保留清楚间距，避免右侧 rank 列贴住左侧主热图和 tick labels。所有字体使用 paper-view 大字号。

### 左上：Events over time

内容：

- 真实 lagPatRank heatmap，事件按原始时间顺序排列；
- y 轴显示 channel labels；
- 底部附 Day/Night strip；
- 非参与 channel-event cell 用 light gray，不用 viridis phantom rank 颜色；
- 左上角写 subject 名称和事件数，格式为 `<dataset>:<subject> | n=<valid_events>`，无外边框；
- panel title 为 `Events over time`。

语义：

- 展示原始时间序列中的传播 rank 结构；
- Day/Night strip 只提供时间背景，不作为 Fig2 主结论；
- 这张 panel 是“数据长什么样”的入口，不是聚类结论。

### 左下：TA/TB clustered events

内容：

- 同一批事件按 KMeans label 排序；
- y 轴 channel order 必须和左上完全一致；
- 簇分界用粗红色实线，必要时加极淡红色边界带；
- 簇标签写 `TA (n=...)` / `TB (n=...)`，无外边框，放在 heatmap 上方但不压住热图；
- 不在 panel 内另写 `KMeans k=2` 标题；
- x 轴为 `Pop Events (clustered)`。

语义：

- TA/TB 是同一 subject 内两个主要传播模板；
- Fig2 主素材应优先选择 stable k=2 subject；
- 如果 subject 的 `stable_k > 2`，这类图可以作为 atlas/补充诊断，但不应强行当成 Fig2 主 panel 的“两模板”证据。

### 右上：Rank dist.

内容：

- per-channel rank distribution；
- 不重复显示 y 轴 channel labels；
- 标题为 `Rank dist.`，居中、不加粗；
- 右侧放共享 rank colorbar，label 为 `First -> Last`；
- 右侧列要窄，不抢左侧 heatmap 的视觉权重。

语义：

- 辅助说明每个 channel 在事件中的 rank 分布；
- colorbar 语义和左侧 heatmaps 一致，都是 rank early-to-late。

### 右下：Mean rank

内容：

- 两类模板的 per-channel mean rank profile；
- y 轴顺序必须和左侧完全一致；
- 不显示 y 轴 channel labels；
- 不画 legend；
- title 为 `Mean rank`；
- 右下 panel 必须紧凑，避免和左下 x 轴末端 tick 重叠。

语义：

- 用一条紧凑 profile 显示 TA/TB 的平均顺序差异；
- 这是左下 clustered heatmap 的摘要，不是新的分析层。

## 全图标题与统计信息

全图第一行：

```text
<dataset>:<subject> | repro=<grade>
```

全图第二行放补充统计，便于后续裁掉：

```text
n=<valid_events> | tau=<overall_tau> | MI=<mi>, p=<p> | KMeans k=<k> | within-tau=<value> | inter-corr=<value> | forward/reverse: ...
```

不要把这行信息放进左下 heatmap panel 内。

## prompt 版

给 agent 的短 prompt：

```text
请按 Fig2 时序图形式生成真实数据 per-subject propagation 图。先读取 docs/fig2_temporal_propagation_panel_spec.md，然后使用 masked 结果树 results/interictal_propagation_masked，不要用模型/SNN 图。布局为 2x2：左上 Events over time，左下 TA/TB clustered events，右上 Rank dist.，右下 Mean rank。四个 panel 必须使用同一 channel y 轴顺序；右侧两个 panel 不显示 y 轴 channel labels；右侧 rank 列和左侧主图之间保留清楚间距；TA/TB 和左上 subject/n 标注不要外边框；左下不写 KMeans k=2 标题，统计信息放全图标题第二行。正式覆盖 per-subject 图时运行 scripts/plot_interictal_propagation.py --masked-features --pr3 --paper-style --max-events 2000。
```

给 agent 的长 prompt：

```text
任务：按 Fig2 时序图形式调整或生成真实数据 per-subject propagation 图。

先读 docs/fig2_temporal_propagation_panel_spec.md。只使用真实数据 masked 传播结果：results/interictal_propagation_masked/。不要使用 subject-SNN 或 Fig4 模型 readout 图。

图形语义：该图是 Fig2 的 subject-level 素材，用来展示大量间期 HFO group events 的 channel-level first-to-last rank 时序，以及这些事件如何被 KMeans 分成同一 subject 内的两类传播模板 TA/TB。它不是 cohort-level 统计，也不是机制证明。

视觉要求：
1. 2x2 布局，左列宽，右列窄，上下行紧凑；左右列不要贴住，右侧 rank 列和左侧主图之间保留清楚间距。
2. 左上为 Events over time：时间顺序 lagPatRank heatmap，底部 Day/Night strip，左上角写 <dataset>:<subject> | n=<valid_events>，无外边框。
3. 左下为 clustered heatmap：同一 channel order，事件按 KMeans label 排序，粗红实线分隔 TA/TB，标签为 TA (n=...) / TB (n=...)，无外边框，不写 KMeans k=2 panel 标题。
4. 右上为 Rank dist.：不显示 y 轴 labels，标题居中不加粗，右侧放共享 First -> Last colorbar。
5. 右下为 Mean rank：同一 y 轴顺序，不显示 y 轴 labels，不画 legend。
6. 全图第一行标题写 <dataset>:<subject> | repro=<grade>；第二行写 n/tau/MI/KMeans/within-tau/inter-corr/forward-reverse 等补充统计，方便之后整行裁掉。

正式覆盖全 cohort：
python scripts/plot_interictal_propagation.py --masked-features --pr3 --paper-style --max-events 2000

单 subject 检查：
python scripts/plot_interictal_propagation.py --masked-features --pr3 --paper-style --dataset epilepsiae --subjects 958 --max-events 2000
```
