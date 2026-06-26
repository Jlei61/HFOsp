# 主图计划

本文主图围绕两个核心论点组织：

1. 间期 HFO 群体事件是癫痫病理网络的指示器。
2. 间期活动可能是病理网络动态的推动者；这部分主要通过模型和病例场景说明可行机制。

## Fig1: 间期 HFO 群体事件与病理网络读出

### Fig1-A: 原始群体事件示例

**目的**：用最直观的原始信号说明，间期 HFO 不是孤立单通道尖峰，而是跨通道共同出现的群体事件，并且群体内部存在稳定的早晚关系。

**当前验收版本**：

- 输出目录：`results/paper-ready-figure/fig1_hfo_group_event_demo/figures/`
- 正式文件：`yuquan_y1_hfo_group_event_demo.png` / `yuquan_y1_hfo_group_event_demo.pdf`
- 复现入口：`scripts/paper_figures/plot_fig1_hfo_group_event_legacy_style.py`
- 数据来源：Yuquan Y1, `FC10477Q`
- 固定示例事件：packed event indices `22,237,1458`
- 图形合同：左侧为 80-250 Hz stacked bipolar traces；右侧为 legacy-style normalized spectrogram，并用 spec-center 点/线显示群体事件内部时序。

**当前口径**：

这张图只承担现象入口作用，不单独证明 cohort-level 传播模板或机制结论。它应该把读者带到后续 Fig1-B/C/D 的定量结果：群体事件可被定义、可排序、可汇总到病理网络轴。

### Fig1-B: 群体事件定义与传播 rank

**计划内容**：展示从 HFO detections 到 packed group event，再到 channel-level event rank / template 的分析流程。

**需要补齐**：

- 明确使用 masked `lagPatRank` 后的正式 pipeline 输出。
- 选一个 subject-level schematic，而不是堆 cohort 数值。
- 避免把示意图画成方法 supplement；主图只保留读者理解传播 rank 所需的最小链条。

### Fig1-C: 病理网络指示器的 cohort-level 证据

**计划内容**：展示间期 HFO 群体事件的空间组织、SOZ/病灶相关性或网络轴 readout。

**需要补齐**：

- 从 Topic 1/3 当前验收结论里选择最稳的 cohort-level readout。
- 区分“事件存在时序结构”和“该结构指向病理网络”的证据层级。
- 主图只放一个核心统计面板，完整分层表放 supplement。

### Fig1-D: 从指示器到动力学 scaffold

**计划内容**：把间期传播模板和病理网络 scaffold 连接起来，作为后续建模主张的入口。

**需要补齐**：

- 明确哪些内容来自真实数据，哪些只是模型 bridge。
- 不在 Fig1 里提前声称“推动者”机制已经被证明；只说明 Fig1 给出可被模型解释的病理网络读出。

## Fig2-Fig6 暂定分工

### Fig2-Fig3: 间期事件作为病理网络指示器

优先承载真实数据主结果：传播模板、网络轴、SOZ/临床相关性、跨事件稳定性。这里应该是第一核心论点的主要证据区。

### Fig4: 指示器证据整合

Fig3 和 Fig4 可以合并为一张更强的主图，避免重复展示类似的 event/rank/template 信息。合并后应突出一个主问题：间期 HFO 群体事件是否稳定读出病理网络。

### Fig5-Fig6: 间期活动作为推动者的模型与病例场景

当前机制证据还没有收口，因此 Fig5/6 先按建模工作组织。允许呈现几类可能病例场景，但必须清楚区分：

- 真实数据已经支持的 readout；
- 模型能够复现或解释的 dynamics；
- 仍然是假设、需要后续验证的机制。

## 当前执行原则

- 主图脚本统一放在 `scripts/paper_figures/`。
- 主图输出统一放在 `results/paper-ready-figure/`，每个 figure/panel group 单独建目录。
- 每个 figure 输出目录必须有 `figures/README.md`，说明展示目的、正式文件、关注点。
- 主图计划文档只记录正式口径和待补齐内容；详细审阅和数值表继续放 `docs/archive/`。
