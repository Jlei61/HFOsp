# BrainBody workshop：contact-sequence 结果第一节（中文讨论稿）

本节的一句话主张：**间期 HFO 群体事件中的触点传播顺序包含可学习的预测信息，支持将 contact-sequence RNN 用作患者内事件传播的读出模块。**

按全文五页的预算，本节保留一张四面板结果图和三段正文；模型架构放在 Figure 1。下文先列图注，再列结果正文。当前是内容与排图讨论稿，尚未进行全文五页模板排版。

## Figure 2 图注

**Figure 2｜间期事件的触点序列包含可预测的传播结构。**
**a，真实事件与自由续写。** 患者 E1146 的两类传播模式（TA、TB）各展示 12 个留出事件及对应的 RNN 续写。每列为一个事件，每行为一个触点，所有热图采用相同触点顺序；颜色表示归一化的事件内参与顺序，灰色表示未参与。网络仅接收首个触点集合，随后自由生成；TA/TB 标签不作为网络输入。示例沿用原图中模板一致性较高、参与触点数接近典型值的事件，用于展示传播模式。
**b，队列预测增益。** 真实序列模型相对跨事件后缀重分配对照的下一触点负对数似然（NLL）改善量；正值表示真实序列模型更优。
**c，生成传播场的对应。** 剔除给定起始触点后，生成序列的触点平均先后顺序与实测间期传播场的 Spearman 相关，TA/TB 两个相关在患者内取平均；对照为触点身份置换。
**d，模型响应与实测转移的对应。** 模型有限时域扰动响应与留出事件中未来 1–3 个秩步的经验触点转移矩阵之间的非对角 Spearman 相关；对照为同一电极杆内的触点身份置换。响应汇总四种真实顺序网络设计及各自三个随机种子。
b–d 均以患者为统计单位（n = 28）；点为患者，横线和粗竖线分别为中位数和四分位距，c、d 的细线连接同一患者。c、d 每位患者的置换对照取 512 次置换的中位数。P 值来自单侧配对 Wilcoxon 检验，报告原始未校正值。

![Figure 2 contact-sequence 候选图](/home/honglab/leijiaxin/HFOsp/results/topic5_contact_sequence_workshop/figures/figure2_contact_sequence.png)

## 结果正文

### 间期事件内的触点传播顺序具有可学习结构

间期 HFO 群体事件中的触点顺序包含可用于预测后续传播的信息。我们将事件表示为按参与先后排列的触点集合，并在固定触点集合下，以患者内训练的 RNN 预测下一触点及事件终止。对于留出事件，真实序列模型优于跨事件后缀重分配对照：28 名患者中有 24 名的预测改善，下一触点负对数似然的患者中位改善量为 0.0237 nats/decision（P = 3.16 × 10⁻⁵；Fig. 2b）。这表明，事件前段与后续触点之间的关联提供了可学习的预测信息。

这种信息也体现在自由续写的传播结构中。仅给定首个触点集合，网络便能生成呈现不同传播模式的序列（Fig. 2a）。剔除给定起始触点后，生成传播场与实测间期传播场的相关仍高于触点置换对照（患者中位 Spearman ρ：0.189 对 0.002；P = 0.00134；Fig. 2c）。因此，续写保留了部分患者内传播结构。

模型内部的响应也与真实的触点转移结构相对应。有限时域扰动响应与留出事件转移矩阵的相关，在 21/28 名患者中高于同电极杆置换对照，配对相关增量的中位数为 0.0676（P = 0.00172；Fig. 2d）。这些结果支持将 contact-sequence RNN 用作事件内传播的读出模块，为进一步检验事件间状态是否提供额外预测信息建立参照；它们尚未确定具体连接结构的必要性。

## 排图取舍与证据边界（作者讨论，不进入结果正文）

- Figure 1 放整体架构及 contact-sequence readout 在整体框架中的位置；Figure 2 只承担本节的实证证据。
- Figure 2a 沿用原图 B 的真实事件与自由续写；Figure 2b 以 28 人队列增益替换原图 C 的单患者训练曲线；Figure 2c 沿用原图 D 的 seed-removed 传播场对应；Figure 2d 沿用原图 I 的患者级响应对应统计。
- 这节使用已完成的 contact-sequence 模型结果（原 v0.3 / multiscale v0.5 证据链），不将这些数字改称 v0.3.8 状态模型的性能。它为后续状态模型提供读出能力的依据；状态是否改善该读出必须在下一节独立检验。
- 这里的“患者内传播”不等于“已证明 SOZ 内专属机制”：触点集合是固定的回顾性建模字典，并非所有触点都被临床标注为 SOZ。正文不将该结果升级为病理机制、真实解剖连接或发作预测。
- 不在本节加入原图 E–F 的发作场对应或 G–H 的完整扰动/拓扑比较，以便将篇幅用于事件内预测、生成和功能对应这条主线；具体非局部连接的特异优势尚未确认。
- 热图示例是原图筛选出的说明性事件。队列统计使用完整的原定 28 人，不按示例效果选择患者。

## 可复现文件

- [Figure 2 PDF](/home/honglab/leijiaxin/HFOsp/results/topic5_contact_sequence_workshop/figures/figure2_contact_sequence.pdf)
- [Figure 2 SVG](/home/honglab/leijiaxin/HFOsp/results/topic5_contact_sequence_workshop/figures/figure2_contact_sequence.svg)
- [统计、来源和文件哈希](/home/honglab/leijiaxin/HFOsp/results/topic5_contact_sequence_workshop/figure2_contact_sequence_metadata.json)
- [生成脚本](/home/honglab/leijiaxin/HFOsp/scripts/paper_figures/plot_brainbody_contact_sequence_figure2.py)

从已冻结绘图输入重画：

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  /home/honglab/leijiaxin/HFOsp/scripts/paper_figures/plot_brainbody_contact_sequence_figure2.py \
  --source-data /home/honglab/leijiaxin/HFOsp/results/topic5_contact_sequence_workshop/source_data
```

统计核对：三个 P 值、核心效应量和分母均与既有源数据一致；脚本包含事件留出身份、起始触点、触点维度、患者配对及原始统计端点检查。PNG/PDF/SVG 由同一次绘图输出。本候选图尚待作者确认。
