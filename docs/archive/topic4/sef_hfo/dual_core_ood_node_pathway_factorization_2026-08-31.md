# Topic 4：双 core Node 的 OOD 拟合与 EE/E->I 通路分解

**日期**：2026-08-31

**状态**：`DUAL_CORE_NODE_REPERTOIRE_PARTIAL_SUPPORT / FULL_DISTRIBUTION_NOT_RECOVERED / EE_OOD_FILTER_WITH_YIELD_COST / ETOI_MODE_OCCUPANCY_MODULATOR / JOINT_NONADDITIVE`

**范围**：development-only；不作 patient-blind、解剖 core、临床波形、发作或患者因果机制主张。

## 1. 科学问题

这一轮不再让连续自由场用很多参数贴合观测密度，而是回到一个可解释的最小假设：

> 在冻结的患者间期目标和同一 SNN 框架中，两个离散的 Node 易激 core 是否足以让同一张网络产生患者 Fig.2C 所示的两种传播模式；冻结这个 Node 后，learned EE 和 E->I 重分配分别改变事件分布的哪一部分？

主要判断量是所有 returned events 中落在冻结患者支持域外的比例：

```text
OOD_all_returned = 1 - N(returned and readable and in patient support)
                         / N(all returned)
```

不可读的 returned event 也计为 OOD。这样不能靠只保留少数好看的可读事件降低分数。KMeans 与 GIF 是必要的模式结构和形态审计，但不替代 OOD 主指标。

## 2. 冻结设计

- Node 场严格只有两个二值 core；不使用连续高斯混合，不增加 K。
- 搜索参数只有两个中心坐标和总 Node 数；总预算与冻结参考一致。
- fit 使用 48 个 Sobol 候选，加历史手放双 core 作为同口径锚点。
- fit、selection、confirmation 使用互不重叠的 network seed 池。
- Z/M 关闭；先只恢复间期底物。
- 患者目标固定为 15 个接触点、保留杆身份的两模式支持域。
- 选择顺序为：两模式均出现 -> OOD 最低 -> weakest-mode support distance 最低 -> returned event 数更多。
- Node 冻结后，EE、E->I、EE+E->I 只转移既有 learned coefficient，不重新拟合。

完整合同和执行计划：

- `docs/superpowers/specs/2026-08-30-topic4-dual-core-ood-node-pathways-design.md`
- `docs/superpowers/plans/2026-08-30-topic4-dual-core-ood-node-pathways.md`

## 3. 双 core Node 结果

最终候选为 `dualcore_s39`：

| 项目 | 数值 |
|---|---:|
| core 1 | (1.54, 1.23) mm |
| core 2 | (18.61, 2.42) mm |
| 中心间距 | 17.11 mm |
| Node 数 | 1,499 |

fit 池中，该候选的 `OOD_all_returned=0.378`，历史手放双 core 为 `0.584`。这是同一 fit seed 池上的开发性比较，不是独立确认优势。selection 上候选 OOD 为 0.422。

### 3.1 独立 confirmation

| 指标 | 结果 |
|---|---:|
| 同一网络内出现两种患者支持模式 | 12/12 networks |
| `OOD_all_returned` | 0.462，network bootstrap 90% CI 0.426-0.500 |
| 只在可读 returned events 上的 OOD | 0.356 |
| 不可读 returned fraction | 0.165 |
| returned events | 94.7/network |
| natural KMeans alignment | 0.756，90% CI 0.694-0.813 |
| weakest-mode full error | 3.047 patient-floor units，10/12 networks 可评价 |

这说明两个 core 有稳定的**双模式表示容量**，但没有恢复患者完整事件分布：接近一半 returned events 仍在患者支持域外。

模式占比也没有恢复。患者训练集为 Mode 1/2 = 30.9%/69.1%，模型 confirmation 为 77.3%/22.7%。因此不能用“两个模式都存在”替代“模式分布一致”。固定每模式 10 个事件的次级距离也都远高于患者噪声地板：recruitment 2.25、precedence 4.10、profile 2.15、event cloud 2.66、weakest-mode 3.05。

### 3.2 Fig.2C 形态审计

GIF 从同一个 confirmation network 中按算法各选一个 support distance 最低的事件，不按肉眼挑选。上半部分是 5 ms 窗的真实 E-neuron 放电密度，使用 viridis；下半部分是同一事件的虚拟接触点读出。实线为模型杆内顺序，虚线为患者 prototype。

两种代表事件都显示了清楚且不同的跨杆招募过程；Mode 2 的代表事件尤其接近患者顺序，Mode 1 仍可见局部顺序误差。它证明“模型确实能产生 Fig.2C-like 事件”，但不能抵消总体 46% OOD，也不能替代对全部事件的统计。

- `results/topic4_sef_hfo/data_driven_dual_core_ood/confirmation/figures/dual_core_node_fig2c_mode_check.gif`
- `results/topic4_sef_hfo/data_driven_dual_core_ood/confirmation/figures/dual_core_node_ood_kmeans_confirmation.png`

## 4. 冻结 Node 后的四臂通路分解

四臂使用同一组 12 个 paired network seeds：Node、Node+EE、Node+E->I、Node+EE+E->I。

| arm | OOD，越低越好 | returned/network | Mode 1/2 share | KMeans alignment | weakest-mode error |
|---|---:|---:|---:|---:|---:|
| Node | 0.479 | 94.6 | 76.6% / 23.4% | 0.768 | 3.470 (8/12) |
| +EE | 0.216 | 41.8 | 75.5% / 24.5% | 0.689 (11/12) | 2.440 (6/12) |
| +E->I | 0.508 | 100.3 | 41.6% / 58.4% | 0.695 | 3.482 |
| +EE+E->I | 0.351 | 87.3 | 65.0% / 35.0% | 0.610 | 2.585 |

相对 Node 的 paired network bootstrap 结果：

- **EE**：OOD 降低 0.263，90% CI -0.347 至 -0.181；但 returned events 同时减少 52.8/network。它更像一个有明显产率代价的选择性过滤器，不能只写成传播几何改善。
- **E->I**：OOD 增加 0.030，90% CI -0.004 至 0.064；Mode 2 share 增加 0.350，90% CI 0.304-0.401；returned events 增加 5.75/network。它主要改变模式占用和可达性，没有提高总体分布内比例。
- **EE+E->I**：OOD 降低 0.127，90% CI -0.171 至 -0.084；Mode 2 share 增加 0.116；但 KMeans alignment 降低 0.159，90% CI -0.230 至 -0.088。
- OOD 的二因素 interaction 为 +0.106，90% CI 0.025-0.191；Mode 2 share interaction 为 -0.244，90% CI -0.340 至 -0.152。两条通路不是简单相加，联合臂没有同时保留 EE 的全部 OOD 优势和 E->I 的全部模式占用效应。

最安全的模型内部机制分解是：

```text
Node 固定点火位置和两模式的基本方向几何；
EE 以明显降低事件产率为代价，过滤掉大量患者支持域外事件；
E->I 主要把模式占用推向 Mode 2，并略微增加事件产率；
两者联合时存在拮抗性的非加性交互。
```

这是冻结 SNN 内部的通路效应，不是患者 SEEG 对 EE 或 E->I 因果机制的识别。

通路图：

- `results/topic4_sef_hfo/data_driven_dual_core_ood/pathway/figures/dual_core_node_pathway_factorization.png`

## 5. 与连续自由场和历史手放双 core 的关系

本轮直接回答的是“严格双 core family 在 OOD 主指标下能做到什么”。fit 池里数据驱动定位的双 core 优于历史手放位置，但 confirmation 没有加入同 seed 的手放对照，因此不能把 fit 差异写成独立泛化优势。

此前 `data_driven_dual_core_vs_free_field` 比较使用不同的事件数、可读事件口径和主目标。那一轮连续自由场在完整 shaft-aware 距离上更好，但在 readable-only OOD 与 natural clustering 上没有稳健压过手放双 core。两轮不能直接比较绝对数值。当前结果也没有推翻连续场的表达容量，只说明一个更简单的双 core Node 已能恢复部分 Fig.2C repertoire，但仍留下明确的 full-distribution residual。

## 6. 裁定与下一步

当前裁定不是“拟合完成”，而是：

```text
DUAL_CORE_NODE_REPERTOIRE_PARTIAL_SUPPORT
FULL_PATIENT_EVENT_DISTRIBUTION_NOT_RECOVERED
```

下一轮若继续降低 weakest-mode error，不应先增加 core 数或再次放开无约束自由场。最小修正应针对当前 residual：

1. 保持两个 Node core 冻结，单独校准 Mode 2 occupancy，而不牺牲 Mode 1 OOD。
2. 将 EE 的“降低 OOD”和“压低事件产率”拆开，寻找 matched-yield 的连接表达强度。
3. 将 E->I 作为 mode-occupancy lever 做低维剂量曲线，再与 EE 做小型二维响应面；目标同时包含 OOD、患者 31/69 模式比例和最低事件产率。
4. 每个候选继续用新网络确认，并保留 Fig.2C-style 全神经元 GIF；禁止只展示最像的事件而不报告总体 OOD。

## 7. 产物与复现

- 配置：`config/topic4_dual_core_ood_node_pathways.json`
- 最终统计：`results/topic4_sef_hfo/data_driven_dual_core_ood/final_analysis.json`
- per-network 表：`results/topic4_sef_hfo/data_driven_dual_core_ood/pathway_per_network.csv`
- confirmation 图与 GIF：`results/topic4_sef_hfo/data_driven_dual_core_ood/confirmation/figures/`
- pathway 图：`results/topic4_sef_hfo/data_driven_dual_core_ood/pathway/figures/`

正式运行共 176 个 worker 单元：fit 98、selection 18、confirmation 12、pathway 48；全部完成，无失败单元。所有长跑均由后台 service/nohup controller 管理，阶段完成后退出；最终分析按 network seed 做配对 bootstrap。
