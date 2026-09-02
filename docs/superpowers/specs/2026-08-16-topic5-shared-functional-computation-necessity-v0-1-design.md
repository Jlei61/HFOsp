# Topic 5.2 shared functional computation necessity v0.1

> **SUPERSEDED / 不得作为最终合同引用**：收口审计发现 §5 的条件中心包含由留出事件完整后半段计算的 future-field coordinate。共同方向本身没有读取留出 target，但删除幅度与 support gate 受该量影响。最终执行与结论改用 [v0.2 修复合同](2026-08-16-topic5-shared-functional-computation-necessity-v0-2-design.md)。

**状态**：执行前冻结。  
**目的**：检验四种连接方式不同、但都用真实事件顺序训练的冻结 RNN，是否依赖同一个可泛化的内部功能成分来预测后续触点。

## 1. 科学问题

已有结果只说明：轻微扰动一小片组织后，四种真实顺序网络产生的未来触点响应图彼此相似，而且比“前段和后段对应关系被打乱”的网络更相似。

这仍然只是充分性方向的证据：网络对同类扰动会产生相似响应。它没有证明这个共同响应成分对预测是必要的。

本实验回答更强的问题：

> 只用另外三种真实顺序网络的训练事件定义一个共同功能成分；把该成分从第四种网络的未见事件状态中删除，是否会选择性损害它随后三步的真实触点预测？

如果删除任意同样大小的方向都会造成同样损失，则只能说明 RNN 对扰动敏感。如果只有跨网络共同成分造成更大损失，才支持不同连接实现收敛到同一个有功能意义的计算。

## 2. 冻结对象与数据拆分

- 模型参数、连接掩码、触点读出、STOP head 和 size decoder 全部冻结；运行前后逐 cell 核对参数 hash。
- 四种真实顺序网络分别是：仅近邻连接、近邻加额外短连接、近邻加随机长连接、近邻加训练选择的长连接。
- “后半段打乱”网络只作为顺序信息对照，不进入真实顺序共同成分。
- 共同成分只从 `axis_train` 事件提取。
- 必要性检验只在 `heldout_test` 事件的冻结 reference states 上进行。
- 训练事件的真实后续触点只用于计算预测损失，不参与共同成分方向的选择；未见事件的 target 在方向冻结前不可读取。

## 3. 训练侧功能响应算子

对每个 fit、每个网络和 seed：

1. 从 response-blind 的训练事件清单中按冻结 identity hash 取最多 64 个事件。
2. 每个事件在早、中、晚三个归一化阶段各取一个仍有后续的状态。
3. 在完整 decoder state 的 hidden 部分，对每个 tissue-grid 中心施加固定宽度的 Gaussian 小片区正负扰动；recruited mask 与 rank index 保持不变。
4. 用真实后续输入 teacher-force 三步，保存未来触点 pre-mask logits 的中心差分响应。
5. 对阶段、未来 1–3 步和 seed 求平均，得到矩阵

\[
O_{a}\in\mathbb R^{C\times M},
\]

其中行是未来触点，列是被扰动的 tissue patch。每列减去触点均值，去掉 softmax 不可辨识的共同 logit 平移；每个网络的矩阵再除以 Frobenius norm，避免某个网络只因响应幅度大而支配共同成分。

## 4. 跨网络留一共同成分

对待检验网络 \(a\)，只使用另外三个真实顺序网络：

\[
O_{-a}=\operatorname{median}\{\widetilde O_b:b\neq a\}.
\]

对 \(O_{-a}\) 做 SVD。primary 使用第一右奇异向量 \(v_{-a,1}\)，它给出哪些 tissue patches 共同参与这一未来触点响应。通过待检验网络自身冻结的 Gaussian patch basis 映射回 hidden space：

\[
u_{-a,1}=\operatorname{unit}(v_{-a,1}^{\top}P_a).
\]

该方向没有使用待检验网络自己的响应，也没有使用未见事件的 target。方向正负不影响删除操作。

## 5. 删除操作

在未见事件状态 \(h\) 上，以该阶段冻结的条件中心 \(c\) 为参照：

\[
h_{\alpha}=h-\alpha u\,u^{\top}(h-c),
\qquad \alpha\in\{0.25,0.50,1.00\}.
\]

这不是把连接删掉，也不是重新训练；它只删除当前状态中沿共同成分的事件特异偏移。\(\alpha=1\) 表示完全删除该一维投影。

只保留通过以下数值有效性检查的分支：有限值、node range、局部 kNN 支持和条件流形残差阈值。禁止 clip 或把越界状态强行拉回。

## 6. 对照

每个未见状态都使用等量 hidden displacement，避免“推得更远所以损失更大”的替代解释。

1. **等量正交方向**：从冻结局部 normal controls 中选与共同成分正交、并在即时 logit 改变和一步 hidden gain 上最接近的方向。
2. **高方差方向**：从前三个训练侧 PCA 方向中按同一规则选择。
3. **后半段打乱网络成分**：用同一 fit 的打乱网络训练侧算子第一成分，映射到待检验网络；位移范数与共同成分删除严格相同。
4. **不删除**：原始 hidden state。

对照方向只用于比较，不影响 reference-state 资格或共同成分定义。

## 7. 预测终点

primary 是删除后延迟的真实后续触点 NLL 变化。为避免只测即时 readout，先输入下一个真实 rank set，再评价其后的三个决策：

\[
\Delta L^{\mathrm{delayed}}
=
\operatorname{mean}_{\tau=1}^{3}
\left[L_{\tau}(h_{\alpha})-L_{\tau}(h)\right].
\]

每一步 NLL 在当时尚未出现的触点上计算，并按真实 next-rank set 的 contact 数取平均。

secondary：即时下一步 NLL、STOP probability trajectory、未来 logits 变化范数、剂量单调性，以及前 1/2/3 个共同成分累计删除的敏感性。

## 8. 统计与主判据

聚合顺序固定为：reference state → seed → fit → patient。不能把 state、event、seed 或 fit 当独立患者。

primary 取三档剂量相对不删除的损失曲线 AUC。共同功能计算的 necessity 只有在以下条件同时满足时才判为 `SUPPORTED`：

1. 共同成分删除的 patient-level AUC 大于 0；
2. 共同成分 AUC 大于等量正交对照；
3. 共同成分 AUC 大于打乱网络成分；
4. 以上三个预注册检验经 Holm 校正后均通过；
5. 四个 held-out 网络中至少三个方向一致；
6. 参数 hash 504/504 不变，且有效 reference-state denominator 完整报告。

共同成分对高方差方向的比较是重要 secondary，不加入 conjunctive primary。

## 9. 允许与禁止的结论

若 primary 通过，允许写：

> 不同连接实现的冻结 RNN 收敛到一个跨网络可迁移的功能成分；删除该成分会选择性损害未见事件的后续触点预测。

仍然禁止写：

- 找到了唯一真实连接图或白质通路；
- 该成分是癫痫特异机制；
- 某种长程连接承担了该计算；
- 与 early-ictal 或 SNN 的机制一致已经得到证明。

若只大于 0、但不优于等量对照，则写“generic perturbation sensitivity”；若优于正交对照但不优于打乱网络成分，则写“shared architecture/task sensitivity, order specificity unresolved”；若无稳定损失，则 necessity 不支持，原跨网络响应相似性仍只保留为充分性证据。

## 10. Figure 6 合同

结果出来前不改主图。若 primary 通过：

- 现有 H 保留为“不同连接网络对多次局部扰动产生相似未来响应”；
- 现有 I 移入补充材料；
- 新 I 画成 paired dose-response：共同成分删除、等量正交、打乱网络成分三条 patient-level 曲线，并直接标出 target-minus-control 的患者级分布与 95% CI。

若 primary 不通过，新结果不进入 Figure 6 主图，避免把一般扰动敏感性包装成共同计算的必要性。
