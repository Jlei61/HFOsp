# Stable Interaction Identifiability v2.1 正式验收

## 验收结论

v2.1 按自身冻结合同完成并验收，不再开放任何 single-fixed-graph 扩展。

- feedback 相对 matched noGraph 的增量存在；
- patient-matched sensitivity + specificity 为 4/6 PASS，另外两位保持未裁决；
- 4 位可辨识患者的 human real-minus-matched-null temporal stability 为 0/4；
- unseen-start 仅有 NLL 5/6，完整 precedence 为 2/6；
- 改用未参与 checkpoint 选择的 test probe 后，D3 仍为 0/6；
- D5、34 人扩展、replication 和 SNN Gate 均未开放。

因此，v2.1 的安全结论是：当前 single fixed contact-feedback graph 没有获得结构特异
证据。该结论不否定 stable backbone、time-varying graph、离散 regimes 或其他高阶
结构。

## 与 v2.2 的关系

v2.2 不是给 v2.1 加 event drive 或扩大 hidden state。它改变了科学时间轴：

- v2.1：rank step 是 recurrence step，描述一场事件内部传播；
- v2.2：一整场事件是 token，event index 是长期状态时间轴；
- v2.2 首先检验 block-wise rank field 是否存在超过估计噪声的时间变化；
- G0 未通过前不实现 ELR-RNN。

机器验收：

`results/topic5_stable_interaction_graph/development/SIG_V2_1_ACCEPTANCE.json`
