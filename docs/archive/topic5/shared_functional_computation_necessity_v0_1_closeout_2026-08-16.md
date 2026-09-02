# Topic 5.2 冻结 RNN：跨网络共同响应是否为预测所必需（v0.1 收口）

> **SUPERSEDED / 数值不得引用**：最终审计发现 v0.1 的删除中心和 support gate 使用了由留出事件完整后半段计算的 future-field coordinate。方向冻结本身没有泄漏，但删除幅度和分支资格受目标信息影响。最终、无目标泄漏的结果见 [v0.2 收口](shared_functional_computation_necessity_v0_2_closeout_2026-08-16.md)。

## 0. 一句话结论

四种连接方式不同、但都用真实事件顺序训练的 RNN，确实会产生相似的“组织小片扰动 → 未来触点输出变化”图案；
但是，删除这份共同图案对应的内部状态成分，并没有选择性损害模型对留出事件后续触点的预测。

因此当前允许写：

> 不同连接实现会收敛到相似的有限时域响应图案，而且该图案与留出间期事件中的触点跟随关系对齐。

当前不能写：

> 这些网络依赖同一个内部成分完成后续传播预测，或这份共同响应“承载”了计算。

最终裁定：

```text
CROSS_NETWORK_RESPONSE_SIMILARITY_SUPPORTED
HELDOUT_INTERICTAL_ALIGNMENT_SUPPORTED
SHARED_COMPONENT_NECESSITY_UNSUPPORTED
FIGURE6_MAIN_PANEL_INELIGIBLE
```

---

## 1. 这次补实验究竟问了什么

上一轮已经得到两个正向结果：

1. 四种真实顺序网络的扰动响应比它们与“前后半段对应关系被打乱”的网络更相似；
2. 四种网络的平均响应图案与本患者留出间期事件中“哪些触点经常在另一些触点之后出现”的统计对齐。

但这两项都只是“形状相似”和“与数据相关”。它们没有回答：

> 模型做预测时是否真的需要这份跨网络共同响应。

本轮直接做删除实验。若共同响应是多个网络实现同一计算时不可缺少的部分，删除它应当比删除同样大小的无关方向
更明显地增加留出预测损失；若删除后没有选择性损害，则响应相似仍只是充分性或相关性证据。

---

## 2. 五种网络分别是什么

四种正式网络都读取相同患者的真实间期事件顺序，只改变内部可用连接：

| 读者用名称 | 内部代号 | 连接方式 |
|---|---|---|
| 只连附近组织 | L0 | 只有局部连接 |
| 附近连接加额外短连接 | L1 | 局部连接加同数量的较短附加连接 |
| 附近连接加随机远连接 | L2m | 局部连接加统计匹配的随机远连接 |
| 附近连接加数据挑出的远连接 | L3 | 局部连接加由间期任务选择的远连接 |

第五种网络保留每次事件后半段本身，但把不同事件的前半段与后半段重新配对。它只用作“打乱后半段对应关系”的
对照。TA/TB 标签没有进入任何网络输入，也没有用于定义本轮共同成分。

---

## 3. 共同成分是怎样定义的

### 3.1 先得到每个网络的功能响应表

在训练事件的真实 RNN 状态上，依次对组织平面上的高斯小片施加等量正、负扰动。随后继续输入相同的真实后续
rank sets，并记录未来 1–3 步每个触点的 pre-mask logit 变化。正负扰动的中心差分形成一张表：

```text
行：未来哪个触点的预测发生变化
列：扰动了哪个组织小片
值：该未来触点输出改变了多少
```

每一列先减去全部触点的均值，去掉 softmax 无法区分的共同平移；每个网络的整张表再除以 Frobenius norm，避免
响应幅度较大的网络支配比较。

### 3.2 每次都把待测网络排除在定义之外

例如要测试“只连附近组织”的网络，就只用其余三种真实顺序网络的训练事件响应表。三张表逐元素取中位后做
SVD，第一条输出方向定义为跨网络共同响应。然后利用冻结的组织小片基底，把它映射回待测网络的 hidden state。

四种网络轮流被排除，因此得到 42 个 fits × 4 个待测网络 = 168 份冻结方向。方向冻结阶段没有读取待测网络的
响应表，也没有读取任何 test-event target。

---

## 4. 内部状态具体怎样被删除

对一个留出事件中的真实状态 \(\mathbf h\)，先在相同事件阶段和显式状态条件下找到训练状态中心
\(\boldsymbol\mu\)。令冻结共同方向为单位向量 \(\mathbf a\)，删除比例为 \(\alpha\)：

\[
\mathbf h'=
\mathbf h-
\alpha\,\mathbf a\mathbf a^\top
(\mathbf h-\boldsymbol\mu),
\qquad
\alpha\in\{0.25,0.50,1.00\}.
\]

含义是：

- 25%：删除该状态沿共同方向偏离训练中心的四分之一；
- 50%：删除一半；
- 100%：把该方向上的偏离完全消掉；
- 其余 hidden dimensions、已经招募的触点、当前 rank index 和 decoder bookkeeping 均保持不变。

这不是把网络重新训练，也不是删 recurrent edge。它只在一个真实留出状态上做局部状态干预，然后观察模型随后
的触点预测是否变差。

---

## 5. 和什么对照

每个真实状态、每档剂量都使用相同位移长度，并只比较共同可执行的扰动分支：

1. **同样大小的无关方向**：与共同方向正交，并尽量匹配扰动当下的输出变化和一步 hidden gain；
2. **网络本身的高方差方向**：删除训练 hidden states 中变化最大的方向；
3. **打乱后半段网络的方向**：用相同方法从前后半段关系被打乱的网络中提取方向。

主判据要求同时满足：

- 删除共同成分后，未来触点预测损失增加；
- 增加量大于同样大小的无关方向；
- 增加量大于打乱后半段网络的方向；
- 四种待测网络中至少三种方向一致。

三项患者级检验共同做 Holm 校正。患者是统计单位，事件、seed 和 fit 均先在患者内聚合。

---

## 6. 预测损失具体是什么

干预后继续给模型真实的后续输入，比较未来第 1–3 步实际 rank set 的负对数似然。数值单位是
`nats/decision`：

- 正值：删除后预测变差；
- 零：删除前后无可见差异；
- 负值：删除后反而略好。

三档删除比例的损失曲线以 dose AUC 汇总。主 endpoint 不使用扰动当下的输出；当下输出只作为次要敏感性分析。

---

## 7. 主结果：共同成分不是选择性必要成分

28 位患者的结果如下：

| 检验 | dose AUC 中位数 | 95% bootstrap CI | 正/负 | 单侧 P | Holm P |
|---|---:|---:|---:|---:|---:|
| 删除共同成分造成的损失 | −0.000218 | [−0.000891, +0.000843] | 12/16 | 0.685 | 1.000 |
| 相对同样大小的无关方向，多造成的损失 | −0.000501 | [−0.002704, +0.000307] | 12/16 | 0.950 | 1.000 |
| 相对打乱后半段网络方向，多造成的损失 | +0.0000276 | [−0.000291, +0.000512] | 15/13 | 0.347 | 1.000 |

三条结果均围绕零波动，没有一条满足方向和统计门槛。按四种待测网络分别计算时，0/4 网络同时满足三条方向要求；
预注册要求是至少 3/4。

因此结果不是“共同成分有一点作用但样本不足”，而是当前效应量本身接近零，并且相对匹配对照没有选择性。

---

## 8. 不是因为只删了一条方向

又累计删除跨网络共同响应的前 2 条和前 3 条方向。28 位患者中：

| 删除方向数 | 共同成分绝对损失 | 相对打乱后半段方向 | 相对高方差方向 |
|---:|---:|---:|---:|
| 1 | −0.000218 | +0.0000276 | −0.0000468 |
| 2 | +0.0000550 | −0.0000137 | +0.0000596 |
| 3 | −0.000266 | +0.0000152 | +0.000274 |

前 2–3 条方向的所有 95% CI 均跨 0，rank-2/3 sensitivity 的 Holm P 均为 1.0。因此主结果不能解释为
“真正的共同计算是多维的，只删第一维太少”。

---

## 9. 不是被扰动时点平均掉了

### 9.1 扰动当下

扰动当下的共同成分损失中位数为 `−0.0000772`，14/28 为正，Holm P=1.0；相对同样大小的无关方向反而更小
（`−0.00318`）。因此没有“当下明显受损、过一步立即恢复”的证据。

### 9.2 事件早、中、晚

- 早段和中段：共同成分绝对损失均为负方向；
- 晚段：共同成分绝对损失为 `+0.00106`，25 位可评价患者中 17 位为正；
- 但晚段相对无关方向的中位差为 0，相对打乱后半段方向仅 `+0.0000145`，校正后均不显著。

晚段结果只说明模型接近事件结束时对状态删减更敏感，不能说明它特异依赖跨网络共同成分。

---

## 10. 这怎样改变上一轮的科学解释

上一轮的三个事实仍然成立：

1. 四种真实顺序网络两两响应相似度中位数为 `0.7476`，高于它们与打乱后半段网络的 `0.6407`；
2. 分半信度两侧均约为 `0.996`，所以差异不是因为对照网络更吵；
3. 四种网络的平均响应与留出事件触点跟随统计的 shaft-preserving null margin 为 `+0.0676`
   （21/28，Holm P=0.0034）。

本轮新增的限制是：

> 相似的响应图案可以被不同网络产生，也能复现留出数据中的平均传播规律，但当前没有证据表明模型预测必须经过
> 这一个可由 SVD 提取并映射回 hidden state 的共同方向或前三个方向。

这留下三种仍可能存在、但本轮不能区分的解释：

- 计算分散在很多可替代维度上，不存在低秩瓶颈；
- 不同网络的输出响应相似，但内部实现并不使用相同 hidden-state 坐标；
- 从 patch→output 表映射回单一 hidden 方向的近似不足以完成真正的 causal lesion。

所以本轮没有证明“共同计算不存在”，而是明确否定了更强的现有主张：**目前没有证明一个低秩、跨网络共享且对
预测必要的内部成分。**

---

## 11. Figure 6 应怎样处理

### 11.1 主图不替换

预注册规定：necessity primary 不通过，新结果不得进入 Figure 6 主图。当前 r5 主图 PNG/PDF/SVG 的 hash 均保持
不变。现有最后一行只能这样读：

- G：同一组织小片重复扰动后，未来触点输出怎样改变；
- H：四种真实顺序网络的响应图案比打乱后半段网络更相似；
- I：平均响应图案与留出间期事件的触点跟随统计对齐，I 不是删除前后准确率比较。

G–I 合起来支持“功能响应收敛并与数据对齐”，不支持“共同响应是必要计算”。

### 11.2 新补充图

![共同响应删除结果](/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-lbss-rnn-v0-1/results/paper-ready-figure/fig6_interictal_crossstate_response_r5_candidate/figures/supplement_topic5_shared_component_necessity_v0_1.png)

- **a**：每个点是一位患者。第一列是删除共同成分造成的绝对未来预测损失；第二、三列分别扣除同样大小的无关
  方向和打乱后半段网络方向。三个患者级分布均围绕零，黑白圆和误差线是中位数与 95% bootstrap CI。
- **b**：累计删除前 1、2、3 条共同方向。红线是绝对损失；紫、绿线分别是相对打乱后半段方向和高方差方向的
  额外损失。所有区间跨 0，说明扩大到前三条方向也没有恢复选择性 necessity。

这张图适合 Extended Data / Supplement，不适合替换主 Figure 6 的阳性 panel。

---

## 12. 工程验收

| 阶段 | 完成 | 关键检查 |
|---|---:|---|
| 训练事件 patch→future-contact 响应重算 | 630/630 | 9,068,880 个可用 state-patch pairs；未读 test target；模型与 decoder hash 全不变 |
| 留一网络方向冻结 | 168/168 | 42 fits × 4 待测网络；待测网络响应表未参与方向定义 |
| 第一方向留出删除 | 504/504 | 1,645,453 个可用延迟决策；模型与 decoder hash 504/504 不变 |
| 前 1–3 方向累计删除 | 504/504 | 3,410,006 个可用延迟决策；模型与 decoder hash 504/504 不变 |
| 主审计 | PASS，15/15 | test split、方向 hash、共同支持、有限值和 patient-first denominator 全通过 |
| 多方向审计 | PASS，9/9 | rank 1/2/3 均有支持，位移范数误差 0 |
| 次要端点审计 | PASS | 504 cells、即时 endpoint、三阶段 denominator 和主 claim hash 全核对 |

浮点复算分支的最大 baseline 差为 `1.91×10⁻⁶`，低于预先记录的 float32 容忍度 `2.5×10⁻⁶`；所有匹配控制的
位移范数误差为 0。

补充图的 PNG 和 PDF 首页已逐图目检，SVG 保留文字为文字；主 Figure 6 的三种格式 hash 未改变。

---

## 13. 当前允许与禁止的论文措辞

### 可以写

> Despite different recurrent connectivity constraints, models trained on the same ordered interictal task
> produced convergent finite-time perturbation responses that aligned with held-out contact-following structure.

> Targeted removal of the leave-one-network shared response component, including its first three dimensions,
> did not selectively impair held-out future-contact prediction relative to matched controls.

### 不能写

- 多种网络已经证明收敛到同一个必要计算；
- 共同响应成分承载了后续传播预测；
- 删除共同成分选择性破坏了 TA/TB 或 suffix prediction；
- 共同成分是患者特异的癫痫病理机制；
- 相似响应证明了共同解剖通路或共同 recurrent edges；
- 本轮 necessity 阴性否定了所有分布式或高维共同计算。

---

## 14. 核心工件

- 设计：`docs/superpowers/specs/2026-08-16-topic5-shared-functional-computation-necessity-v0-1-design.md`
- 计划：`docs/superpowers/plans/2026-08-16-topic5-shared-functional-computation-necessity-v0-1.md`
- 主判决：`results/topic5_latent_propagation_landscape_v0_2/shared_functional_computation_necessity_v0_1/CLAIM_ADJUDICATION.json`
- 主审计：`results/topic5_latent_propagation_landscape_v0_2/shared_functional_computation_necessity_v0_1/FINAL_AUDIT.json`
- 多方向敏感性：`results/topic5_latent_propagation_landscape_v0_2/shared_functional_computation_necessity_v0_1/SUBSPACE_SENSITIVITY_SUMMARY.json`
- 次要端点：`results/topic5_latent_propagation_landscape_v0_2/shared_functional_computation_necessity_v0_1/SECONDARY_SUMMARY.json`
- Figure 6 决策：`results/paper-ready-figure/fig6_interictal_crossstate_response_r5_candidate/figures/FIGURE6_NECESSITY_DECISION.json`
- 补充图：`results/paper-ready-figure/fig6_interictal_crossstate_response_r5_candidate/figures/supplement_topic5_shared_component_necessity_v0_1.{png,pdf,svg}`
