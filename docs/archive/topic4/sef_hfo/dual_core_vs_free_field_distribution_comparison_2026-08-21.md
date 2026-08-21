# 手放双 core 与连续自由场的间期分布解释力比较

## 一句话结论

在完全匹配的 Node 预算、网络种子、噪声、事件探测器和患者目标下，连续患者拟合场对患者 held-out 间期事件分布的解释力明显强于手放双 core。优势同时出现在 recruitment、precedence、rank profile 和完整 event cloud，并主要伴随更充分的 SCL/双杆参与。手放双 core 仍能产生直观且稳定的双簇，但“容易看成两个 core”不等于“更接近患者完整事件分布”。

当前状态：

```text
CONTINUOUS_FIELD_DISTRIBUTION_ADVANTAGE
/
HAND_DUAL_CORE_K2_CONTROL_RETAINED
/
PATIENT_BLIND_GENERALIZATION_NOT_TESTED
/
ANATOMICAL_CORE_CLAIM_NOT_SUPPORTED
```

## 科学问题

此前的手放双 core 位于两个传播端点附近，空间解释直接；连续自由场则由患者间期事件优化得到，但形态更复杂。这里要回答的不是哪张场更好看，而是：在相同资源和同一 SNN scaffold 下，哪一种 Node 场能更完整地解释患者真实间期事件的招募、先后关系、模式比例和事件分布。

## 冻结比较合同

- 连续场：冻结的 `v62_density_t050` Node field。
- 手放双 core：中心固定为 `(4.199, 9.129)` mm 与 `(16.479, 3.966)` mm；每张网络选择距任一中心最近的 1129 个 E neurons，形成二值场。
- 两臂均满足 `sum(h)=1129`。直接固定 1.5 mm 半径会随网络实现覆盖 1080–1184 个神经元，因此不用于主比较；等预算场的有效半径为 1.458–1.530 mm。
- 12 张配对网络：seed 1561–1572；每张 20 s；相同空间 OU、`d_i`、位置、拓扑、delay、GABA 和公共 detector。
- EE 与 E-to-I learned edge coefficients 均为精确 no-op；Z/M 关闭。因此本实验只比较 Node field representation，不混入连接或慢变量。
- 患者目标使用冻结的 rev10-SA recording-block split：train 30,049 events/56 blocks；held-out 16,633 events/24 blocks。held-out 已在此前开发中看过，故本轮仍是 development-only，不是新的 patient-blind 验证。
- 正式分布评分每张网络、每个患者模式取 6 个事件；3 个事件/模式为低预算敏感性分析。单杆事件作为真实分布误差保留，不能因 KMeans 不可读而删除。
- KMeans 图只使用 returned、双杆且位于患者 support 内的事件，作为 Fig.4 风格的描述性补充，不能替代完整分布评分。

## 结果

### 完整分布解释力

配对差定义为 `continuous - hand`；误差越低越好。区间为 4096 次 paired network bootstrap 的 5–95% 分位，网络 seed 是独立统计单位。

| 端点 | 连续场减双 core | 90% 区间 | 配对 Wilcoxon p |
|---|---:|---:|---:|
| weakest-mode score | -1.474 | [-2.013, -0.985] | 0.0010 |
| recruitment | -1.186 | [-1.517, -0.794] | 0.0034 |
| precedence | -1.386 | [-1.858, -0.829] | 0.0049 |
| rank profile | -1.052 | [-1.221, -0.878] | 0.00049 |
| event cloud | -1.594 | [-1.929, -1.243] | 0.0010 |
| mode-proportion JS | -0.035 | [-0.055, -0.016] | 0.0122 |

双 core 的 weakest-mode error 中位数为 3.516，连续场为 1.786。模式 A 的平均 mode score 从 2.530 降到 1.289，模式 B 从 3.355 降到 1.987。因此连续场的优势不是只由某一个容易模式驱动。

### 优势来自哪里

- 双杆事件比例：双 core 0.534，连续场 0.803；配对差 +0.268 [0.228, 0.312]。
- SCL 招募比例：双 core 0.434，连续场 0.690；配对差 +0.255 [0.221, 0.294]。
- OOD：双 core 0.385，连续场 0.419；连续场略高 0.035，但区间 [-0.002, 0.070] 跨 0，Wilcoxon p=0.204。

这说明连续场的主要收益不是简单减少 OOD，而是让患者目标中重要的 SCL 和跨杆结构真正进入事件分布，同时改善杆内与跨杆的 precedence。

### KMeans 结果

双 core 的自然 K=2 direction-balanced alignment 为 0.723，连续场为 0.647；配对差 -0.077 [-0.156, 0.001]，Wilcoxon p=0.176。双 core 数值上更容易形成清楚的两簇，但该差异未稳定离开 0；更重要的是，它没有转化为对患者完整事件分布的更好解释。

因此两张图应一起读：KMeans 图回答“是否能切出两个方向簇”，分布图回答“这些事件是否像患者”。前者不能代替后者。

### 低事件预算敏感性

每模式每网络只取 3 个事件时，连续场的 recruitment 和 event-cloud bootstrap 区间仍偏向改善，mode share 和双杆/SCL participation 也保持优势；但 weakest-mode、precedence 和 profile 的区间跨 0，Wilcoxon 也不稳定。这表明正式 6-event 预算下的多层优势需要足够事件数才能分辨，不能外推为任意小样本下都成立。

## 解释与边界

1. 手放双 core 是有价值的简化机制对照。它证明两个固定热点足以生成直观的双模式事件，但不足以解释患者完整的多杆 recruitment、precedence 和 event-cloud 分布。
2. 连续场的额外解释力说明患者数据约束出的空间结构包含超出两个固定热点的信息；当前最直接的证据是双杆参与和 SCL recruitment 的恢复。
3. 这不证明连续场中的峰就是患者解剖 core，也不证明 K=3 Gaussian 参数具有生物学唯一性。比较的是冻结的场函数，不是其参数化解释。
4. 这不涉及 learned EE/E-to-I 连接、Z/M 或发作；这些因素在本轮都关闭。
5. 患者 held-out target 已用于既往开发，结果不能写成 patient-blind generalization。

## 产物与审计

- 主比较图：`results/topic4_sef_hfo/data_driven_dual_core_vs_free_field/figures/dual_core_vs_free_field_explanatory_power.{png,pdf}`
- Fig.4 风格 KMeans 图：`results/topic4_sef_hfo/data_driven_dual_core_vs_free_field/figures/dual_core_vs_free_field_kmeans.{png,pdf}`
- 汇总：`comparison_summary.json`、`per_network_metrics.csv`
- 审计：`comparison_audit.json`，12/12 seed pairs 全部通过，0 failures。
- worker 原始产物保留在同目录 `workers/`，未纳入 Git。

## 下一步

间期 Node substrate 可以冻结为连续场，手放双 core 保留为关键简化对照。进入 EE/E-to-I/Z/M 发作桥接时，应在两种 Node 场上至少保留一个小型 matched control：若只有连续场能同时保留 Fig.4 分布并进入合格发作态，才说明自由场提供了跨状态所需的额外结构；若双 core 同样成立，则更简单的机制解释仍未被排除。
