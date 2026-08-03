# 患者特异 RNN 两类一致性指标 v0.1

## 指标

1. **间期预测一致性**：模型自由生成与 untouched test20 中所有有向 contact pair 的先后概率相关。它直接衡量模型是否恢复患者自己的传播排序，而不是只看 NLL。
2. **发作期预测一致性**：冻结 RNN-derived contact field 与同患者 clinical-onset 后 0--10 s、1--150 Hz energy field 的 max absolute Spearman。发作 target 不进入训练；它是跨状态空间预测一致性，不是逐次发作路径预测。

## 全 cohort 统计

- 间期共 34 人；图中显示全部 34 人。排除三名 development patients 后，RNN 相对 rank-shuffle 的 precedence correlation 增量中位数为 `0.745`，28/31 为正，`P=2.328e-08`。
- 发作期 exact clinical-onset target 可用者共 16 人；图中显示全部 16 人。排除 E1146 后 primary 15 人中，RNN field 相对 all-contact channel-shuffle null 的一致性 margin 中位数为 `0.167`，13/15 为正，`P=0.02557`。
- 两项患者级一致性强弱在 15 人中不相关：Spearman `rho=0.171`，permutation `P=0.544`。因此队列层面两项均成立，不代表“间期模型拟合最好的人一定具有最强发作对应”。

## 边界

- early-ictal 分母不是 34，因为 18 人没有当前合同所需的 exact clinical-onset target；不能把缺失 target 当成阴性患者。
- 发作期一致性在 all-contact null 下成立；within-shaft sensitivity 和相对完整静态 scaffold 的增量仍未建立。
