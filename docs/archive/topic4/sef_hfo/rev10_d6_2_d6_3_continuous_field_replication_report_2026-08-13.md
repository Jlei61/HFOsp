# Topic 4 rev10-D6.2/D6.3 连续场联合方向与独立复制报告

## 一句话裁定

```text
D6_2_LOCAL_JOINT_SIGNAL_NOT_OBSERVED
/
D6_3_NEAR_BOUNDARY_CANDIDATE_NOT_REPLICATED
/
FIG4_DIAGNOSTIC_ONLY
```

旧 slow-off SNN 中，两个正交连续场方向的等权组合在 D6.2 六张网络上接近联合信号边界，但在 D6.3 十二张全新网络上没有复制患者几何改善。当前不能说 data-driven 连续场稳定复现患者 A/B 间期传播模式。

## 科学问题与固定合同

D6.2 不使用 Gaussian component、峰数量或电极坐标来构造场，而是在同一个 18 系数 tensor cubic B-spline 场中组合两个预先冻结的正交方向：一个来自自然 KMeans 信号，另一个来自 patient contact-split cross-fit 信号。7 个场候选共享总 field mass、网络拓扑、common detector、spatial OU 和 16 s 仿真；Edge exact no-op，`beta`、optimizer 与慢变量关闭。

D6.3 不再搜索，只冻结 D6.2 的近边界候选 `(a,b)=(0.5,0.5)`，在 12 张全新网络 `1401–1412` 上与 warm field 成对复制。D6.2 网络不进入复制裁定，network seed 是独立统计单位。

## D6.2 结果

等权组合相对 warm baseline：

| 指标 | warm | `(0.5,0.5)` | paired delta |
|---|---:|---:|---:|
| 自然 KMeans balanced alignment | 0.707 | 0.839 | +0.133，q05 −0.006，q95 0.275 |
| patient cross-fit margin | 0.298 | 0.443 | +0.146，q05 0.079，q95 0.208 |
| worst-mode recruitment error | 0.144 | 0.133 | 改善 +0.033，q05 −0.006，q95 0.072 |
| 支持 K=2 的网络 | 1/6 | 4/6 | — |
| runaway | 0/6 | 0/6 | — |

patient cross-fit 改善有正下界，但自然 KMeans 改善下界略跨 0。因此正式状态是 `REV10D6_2_JOINT_CONTINUOUS_FIELD_SIGNAL_NOT_OBSERVED`，不是通过。

## D6.3 独立复制

| 指标 | warm | 冻结候选 | candidate − warm |
|---|---:|---:|---:|
| 自然 KMeans balanced alignment | 0.728 [0.680, 0.779] | 0.786 [0.709, 0.859] | +0.057 [−0.021, 0.137]；9/12 为正 |
| patient cross-fit margin | 0.390 [0.309, 0.464] | 0.283 [0.150, 0.413] | −0.107 [−0.270, 0.077]；4/12 为正 |
| worst-mode recruitment error | 0.158 | 0.133 | 改善 −0.003 [−0.053, 0.050]；5/12 为正 |
| 支持 K=2 的网络 | 1/12 | 4/12 | 低于预注册 8/12 |
| runaway | 0/12 | 0/12 | 安全通过 |

自然分群均值略升，但 paired bootstrap 下界跨 0；更重要的是 patient cross-fit 平均下降，且 K=2 网络覆盖不足。复制规则明确失败，状态为 `REV10D6_3_JOINT_CONTINUOUS_FIELD_NOT_REPLICATED`。

## Fig.4 风格诊断图

- `fig4a_spatial_ou_direct_readout`：连续场 landscape、同一网络 TA/TB onset density 和 15 个虚拟触点直接波形。
- `fig4b_spatial_ou_kmeans_consistency`：pooled KMeans rank heatmap、rank distribution、模型/患者 prototype 和 cross-fit matrix。

D6.3 图中的 cross-fit matrix 为 `[[+0.80, -0.72], [-0.48, +0.35]]`。第一行清楚，第二模式正匹配仍弱。两图通过工程与视觉 QA，但只能作为 `DIAGNOSTIC_ONLY`，因为 pooled 图形不能覆盖 12 张网络上的复制失败。

## 机制解释与边界

这次阴性结果不是 CMA-ES 没收敛造成的：D6.2/D6.3 直接测试了预先定义的连续场局部方向及其组合，没有运行优化器。它说明在旧 slow-off 方程、固定 OU 和当前 warm field 附近的这个二维低频子空间里，没有观察到可跨网络复制的“自然双簇 + 患者几何”联合改善。

它也不能证明所有连续自由场或目标函数都失败，更不能关闭 active Z+M 机制。D6.3 冻结于共享 Z/M baseline 之前，必须保持为 historical slow-off replication。下一轮若继续自由场，只能继承 `config/topic4_data_driven_snn_baseline_zm_v1.json`，显式使用 `active_z_plus_m`、每候选至少 20 s，并把晚期 runaway 判为无效。

