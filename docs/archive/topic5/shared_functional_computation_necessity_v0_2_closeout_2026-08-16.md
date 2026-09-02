# Topic 5.2 冻结 RNN：跨网络共同响应是否为预测所必需（v0.2 最终收口）

## 0. 一句话结论

四种连接方式不同的 RNN 会产生相似、并与留出间期触点跟随结构对齐的有限时域扰动响应；但是，从留出状态中删除由另外三种网络定义的共同响应成分，并没有比删除同样大小的无关成分更稳定地损害未来触点预测。

最终裁定：

```text
CROSS_NETWORK_RESPONSE_CONVERGENCE_SUPPORTED
HELDOUT_INTERICTAL_ALIGNMENT_SUPPORTED
LOW_RANK_SHARED_COMPONENT_NECESSITY_UNSUPPORTED
FIGURE6_MAIN_PANEL_INELIGIBLE
```

## 1. 这项补实验为什么必要

上一轮得到两项结果：

1. 四种真实顺序网络的“组织小片扰动 → 未来触点输出变化”图案彼此更像；
2. 它们的平均响应图案与留出事件中“哪些触点经常跟在另一些触点后面出现”的统计对齐。

这两项说明网络对同类扰动作出相似反应，而且反应与真实数据有关；但仍可能只是四种网络都学到了相似的输入—输出统计。要证明共同计算，还需要回答：

> 模型做后续触点预测时，是否真的依赖这份共同响应对应的内部状态成分？

本轮因此不再比较“响应图像像不像”，而是直接删除共同成分，再看未见事件的预测准确率是否选择性下降。

## 2. 共同成分怎样得到

四种正式网络都使用相同患者的真实事件顺序训练，只改变内部连接约束：仅近邻、额外短连接、随机远连接、数据选择的远连接。第五种网络打乱不同事件的前半段—后半段对应关系，只作顺序对照。TA/TB 标签没有输入任何网络，也没有用于本轮方向定义。

对每个网络，在训练事件的真实状态上逐个扰动组织平面的小片，记录未来 1–3 步每个触点输出改变多少，形成一张“被扰动组织小片 × 未来触点响应”表。

测试某一种网络时，只用另外三种真实顺序网络的训练响应表求共同成分；待测网络自己的响应表和留出事件 target 均不进入方向定义。四种网络轮流留出，共冻结 42 fits × 4 网络 = 168 份方向。

## 3. P0：旧 v0.1 为什么作废

最终代码—合同核对发现，v0.1 虽然没有用留出 target 选择共同方向，却用下面的中心计算删除幅度和 support：

\[
c_{e,k}=\gamma(s_{e,k})+u_e b(s_{e,k}).
\]

其中 \(u_e\) 来自该留出事件完整的后续传播场。换句话说，在评价模型能否预测后半段之前，删除中心已经间接看过后半段。这不会改变共同方向本身，却会改变：

- 状态沿共同方向被删除多少；
- 扰动后状态是否被判为仍在经验支持内；
- 哪些状态进入最终统计。

因此 v0.1 数值不能作为最终结果引用。

v0.2 把中心固定为只用训练事件拟合的阶段轨迹：

\[
\boxed{c_{e,k}=\gamma_{\mathrm{train}}(s_{e,k})}.
\]

旧缓存里的 `event_u` 和 `conditional_center` 在进入删除逻辑前被直接移除；support 的 scale、kNN 和残差阈值也在新中心下重算。留出事件后半段只在共同方向冻结以后，用于计算预测损失。

修复前后第一共同方向几乎不变：168 份方向的 absolute cosine 中位 0.999993，最小 0.995747。变化主要来自删除中心与 support，而不是重新选择了一个有利方向。

## 4. 删除具体做了什么

对留出事件真实 hidden state \(h\)，共同方向 \(u\) 和三档剂量 \(\alpha=0.25,0.50,1.00\)：

\[
h_\alpha=h-\alpha uu^\top(h-\gamma_{\mathrm{train}}(s)).
\]

- 25%：删除该方向偏移的四分之一；
- 50%：删除一半；
- 100%：完全删除这一维偏移；
- 已出现触点、当前步骤、STOP/size bookkeeping 和所有模型参数保持不变。

随后给删除前和删除后的模型输入完全相同的真实后续 rank sets，比较未来第 1–3 步实际 next-contact set 的负对数似然。正值表示删除后预测变差。

每个状态同时比较三类同位移长度对照：与共同方向正交的无关方向、高方差方向、由“后半段对应关系打乱”网络得到的方向。这样可以区分“模型对任何扰动都敏感”和“模型特异依赖共同成分”。

## 5. Primary：没有选择性必要性

患者是统计单位；reference state、seed 和 fit 均先在患者内聚合。28 位患者的结果为：

| 检验 | dose AUC 中位数 | 95% bootstrap CI | 正/负 | 单侧 P | Holm P |
|---|---:|---:|---:|---:|---:|
| 删除共同成分造成的预测损失 | +0.000878 | [−0.000412, +0.002000] | 17/11 | 0.1008 | 0.3024 |
| 比同样大小的无关方向多造成的损失 | −0.000561 | [−0.002025, +0.001092] | 13/15 | 0.7810 | 0.7810 |
| 比打乱后半段网络方向多造成的损失 | +0.000050 | [−0.000261, +0.000591] | 15/13 | 0.2759 | 0.5519 |

共同成分删除有很小的正向趋势，但置信区间跨 0；更关键的是，它没有比同样大小的无关方向造成更大损害，也没有比打乱后半段网络方向更有害。

按四种待测网络分别检查，只有 1/4 网络满足三条效应方向均为正；预注册要求至少 3/4。最终裁定是 `NECESSITY_UNSUPPORTED`。

## 6. 删除更多共同方向也没有闭合

累计删除前 1、2、3 个共同响应成分：

| 删除维数 | 共同成分绝对损失 | 95% CI | 相对打乱后半段方向 | 相对高方差方向 |
|---:|---:|---:|---:|---:|
| 1 | +0.000878 | [−0.000412, +0.002000] | +0.000050 | +0.000797 |
| 2 | +0.000763 | [−0.000559, +0.002364] | −0.000026 | +0.002153 |
| 3 | +0.001380 | [−0.000872, +0.003216] | −0.000150 | +0.002110 |

前 2–3 维的绝对损失 CI 仍跨 0；相对打乱网络方向仍围绕 0。相对高方差方向在 2–3 维时接近显著，但校正后分别为 0.0697 和 0.0701，不能升级为支持。

所以结果不能解释为“共同计算是三维的，之前只删一维太少”。

## 7. 扰动当下和事件阶段说明什么

扰动当下，共同方向删除使下一步输出损失增加：中位 +0.00348，21/28 为正，Holm P=0.0058。但它相对同范数无关方向的差为 −0.000953，说明当下变化不是共同方向特异的。

在事件晚段，未来 1–3 步的绝对损失趋势更强：24 位可评价患者中位 +0.00148，未校正 P=0.0059；但整个 secondary family 校正后 P=0.0704，并且相对无关方向和打乱网络方向均未通过。

因此最稳妥的解释是：删除 hidden-state 成分会即时改变读出，接近事件结束时模型也更脆弱；这些现象没有证明跨网络共同成分是后续预测的特异必要通道。

## 8. 这对“共同功能计算模式”意味着什么

上一轮仍成立的事实是：

- 四种真实顺序网络之间的响应相似度中位 0.7476；
- 真实顺序网络与打乱后半段网络的相似度中位 0.6407；
- reliability-corrected margin +0.0758，24/28 同向；
- 用三种网络的平均响应预测第四种未参与平均的网络，margin +0.0693，23/28 同向；
- 平均响应与留出间期 contact-following 结构在保留电极杆的空间随机基线下仍对齐：+0.0676，21/28，Holm P=0.0034。

这些结果支持：不同连接设计收敛到相似、可外推到第四种网络、并与留出数据相符的有限时域响应规律。

本轮新增的限制是：把这份响应规律压缩成一至三个 hidden-state 方向后，删除它们没有选择性损害预测。因此目前不能把“相似且有数据意义的响应规律”升级为“多个网络依赖同一个低秩内部计算瓶颈”。可能的解释包括：

1. 功能分散在更多可替代维度上；
2. 输出响应相似，但四种网络的内部坐标不同；
3. 从组织小片→触点响应表映射回少数 hidden directions，不足以构成真正的 causal lesion。

这项阴性不推翻响应收敛，却明确限定了它的机制强度。

## 9. Figure 6 处理

Primary 不通过，因此 Figure 6 主图不替换。主图最后一行仍只能支持：多次组织小片扰动产生稳定响应、不同连接设计的响应相似、平均响应与留出间期触点跟随结构对齐。

删除结果进入两面板补充图：

![共同响应删除结果](/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-lbss-rnn-v0-1/results/paper-ready-figure/fig6_interictal_crossstate_response_r5_candidate/figures/supplement_topic5_shared_component_necessity_v0_2.png)

- a：28 位患者的单方向删除结果。三列分别是绝对预测损失、相对同范数无关方向、相对打乱后半段网络方向；白点和黑线是患者中位数与 bootstrap 95% CI。
- b：累计删除前 1、2、3 个共同方向。红线是绝对损失，紫线是相对打乱网络方向，绿线是相对高方差方向；所有关键 CI 均跨 0。

主图 PNG/PDF/SVG hash 均与删除实验前一致。

## 10. 工程验收

- 训练侧响应表：630/630 PASS，8,952,245 个有效 state-patch pairs；
- 留一方向：168/168 PASS；
- 单方向删除：504/504 PASS，699,925 个有效 state-family-dose 分支，1,602,882 个延迟决策；
- 累计 1–3 维删除：504/504 PASS，3,310,720 个延迟决策；
- 模型与 decoder hash：所有单元不变；
- 最大 reference replay error：5.45e-6，小于冻结容忍度 1e-5；
- 单方向最终审计：20/20 PASS；
- 累计删除审计：11/11 PASS；
- 相关回归测试：27/27 PASS；
- 补充图 PNG/PDF/SVG 目视检查通过，SVG 文字保持为文字；
- 并行执行发现的汇总临时文件竞争已修复：分片只写单元，最后由单进程汇总。

## 11. 最终允许与禁止的措辞

允许：

> Different connectivity designs produced convergent finite-time perturbation responses aligned with held-out interictal contact-following structure. Removing the target-free leave-one-network shared component did not selectively impair future-contact prediction.

禁止：

- 多个网络依赖同一个必要低秩计算；
- 共同响应成分承载了传播计算；
- 已识别唯一真实连接图、白质通路或患者特异病理机制；
- 这项删除实验证明了发作期或 SNN 的跨模型机制一致性。

## 12. 核心工件

- `results/topic5_latent_propagation_landscape_v0_2/shared_functional_computation_necessity_v0_2/CLAIM_ADJUDICATION.json`
- `results/topic5_latent_propagation_landscape_v0_2/shared_functional_computation_necessity_v0_2/FINAL_AUDIT.json`
- `results/topic5_latent_propagation_landscape_v0_2/shared_functional_computation_necessity_v0_2/SUBSPACE_FINAL_AUDIT.json`
- `results/topic5_latent_propagation_landscape_v0_2/shared_functional_computation_necessity_v0_2/PRIMARY_INFERENCE.csv`
- `results/topic5_latent_propagation_landscape_v0_2/shared_functional_computation_necessity_v0_2/SUBSPACE_INFERENCE.csv`
- `results/topic5_latent_propagation_landscape_v0_2/shared_functional_computation_necessity_v0_2/SECONDARY_INFERENCE.csv`
- `results/paper-ready-figure/fig6_interictal_crossstate_response_r5_candidate/figures/supplement_topic5_shared_component_necessity_v0_2.{png,pdf,svg}`
