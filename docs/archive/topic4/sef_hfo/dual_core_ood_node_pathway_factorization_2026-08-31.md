# Topic 4：双 core Node 的 OOD 拟合与 EE/E->I 通路分解

**日期**：2026-08-31

**状态**：`DUAL_CORE_NODE_REPERTOIRE_PARTIAL_SUPPORT / PATHWAY_REFIT_OOD_IMPROVEMENT_CONFIRMED / MODE_OCCUPANCY_PARTIAL / NATURAL_KMEANS_NOT_IMPROVED / FULL_DISTRIBUTION_NOT_RECOVERED / NATIVE_HF_CARRIER_NOT_RECOVERED`

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

必须强调：这一轮不是在双 core 上联合优化两条通路。EE 与 E->I 的六维系数行来自旧连续场，四臂只做原样迁移和开关消融；四个 manifest 都明确记录 `pathway_refit=false`。因此联合臂只能检验旧系数在新 Node 上是否可加，不能裁定双 core 上重新学习的联合连接是否有容量。

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

### 4.1 两个 core 是否同时存在

两个 Node core 在每次仿真从第 0 ms 起同时写入同一个二值 `h` 场。代表 confirmation 网络中 1,499 个低阈值 E 节点按最近中心分为 752/747，未发生只加载一个 core 的工程错误。事件中两个区域不同时点火来自空间噪声、背景状态和递归传播的竞争，属于当前机制的预期结果。

但这还不等于“一个 core 因果地产生一个模式”。当前模式标签来自患者分类器，不是 core 标签；在完成 first-core onset、单 core lesion 和 matched relocation 前，只能说两个静态起核区支持了两种传播事件，不能声称两者一一对应。

### 4.2 绝对时间尺度和原生载波审计

零仿真重读现有轨迹显示，患者两模式的触点招募跨度中位数为 42.6 和 48.0 ms，5--95% 区间分别为 17.8--111.1 和 20.2--104.3 ms。四臂的 equal-network 中位跨度为 Node 36、EE 47.5、E->I 32、联合 41 ms。因此当前 GIF 看起来缓慢主要来自 5 ms 模型帧以 8 fps 播放，即 25 倍慢放；绝对传播跨度并未明显慢于患者。

为关闭这一读出缺口，固定 `dualcore_s39`、seed 2430 和 Node-only 动力学，新增了不做时间平滑的 1 ms 区域 E/I population rate，以及 core 中心和 15 个触点的 1 ms 突触电流 proxy。新 worker 与原 confirmation 产物对 event onset、rank、event time、return、active fraction 和 contact envelope 做逐数组 parity；七项完全一致，因此新增记录没有改变原仿真。

50 个完整、returned、患者支持内事件的原始结果一致：两种模式、两个 core 的中位 raw peak count 都是 1，出现至少三个规则周期的比例均为 0；core-center current proxy 也是单个宽脉冲。core 1 相对 annulus 的瞬时峰值比约 2.73--2.77，30--80 Hz 绝对功率比约 4.13--7.38；core 2 对应约 1.36--1.44 和 1.95--2.06。这说明局部瞬时活动和宽频高频能量确实集中，但**高频能量来自陡峭单脉冲，不是原生多周期 HFO carrier**。Welch 主峰固定在 23.4 Hz 只是 256 ms 事件窗的最低有效频率格点附近，不应解释成一个 23 Hz 生理振荡。

因此 Fig.4 的 30--80 Hz 波形仍只能作为统一的 band-limited model-current 显示语法，不能作为模型已产生原生 60 Hz 局部节律的证据。

可复现产物：

- `results/topic4_sef_hfo/data_driven_dual_core_ood/temporal_carrier_audit.json`
- `results/topic4_sef_hfo/data_driven_dual_core_ood/carrier_canary/raw_carrier_analysis.json`
- `results/topic4_sef_hfo/data_driven_dual_core_ood/carrier_canary/figures/dual_core_raw_carrier_canary.png`

### 4.3 冻结 Node 的最小 AMPA/GABA kinetics canary

在 Node、空间 OU、网络拓扑、外部 Poisson 均值和零 pathway redistribution 全部冻结的条件下，只扫描 `tau_d_AMPA={2.0,3.5} ms` 与 `tau_d_GABA={8,12,18} ms`。每格使用相同的 3 个 network seeds 和 8 s 轨迹；该轮是机制可行性 canary，不是正式参数拟合。

| AMPA/GABA (ms) | both modes | OOD | natural KMeans | returned/network | 原生三周期事件 |
|---|---:|---:|---:|---:|---:|
| 2.0/8 | 3/3 | 0.458 | 不可评价 | 13.7 | 0% |
| 2.0/12 | 3/3 | 0.531 | 0.831 (3/3) | 37.0 | 0% |
| **2.0/18** | **3/3** | **0.339** | **0.925 (3/3)** | **38.3** | **0%** |
| 3.5/8 | 1/3 | 0.700 | 不可评价 | 4.0 | 0% |
| 3.5/12 | 3/3 | 0.579 | 0.733 (2/3) | 23.7 | 0% |
| 3.5/18 | 3/3 | 0.542 | 0.768 (3/3) | 39.0 | 0% |

两个结论需要分开：

1. 将 AMPA 衰减从 3.5 ms 缩短到 2.0 ms、保留 GABA 18 ms，在这三个开发 seeds 上同时改善 OOD 和自然 KMeans，且没有明显牺牲事件产率。这是值得做独立 20 s confirmation 的**间期 repertoire 候选**，不是已冻结的新工作点。
2. 六个组合的原生三周期比例全部为 0。单独调快 AMPA/GABA 衰减不足以把当前单脉冲事件变成局部高频振荡；继续扩大纯时间常数网格缺少科学依据。

可复现产物：`results/topic4_sef_hfo/data_driven_dual_core_ood/carrier_kinetics/aggregate.json` 和 `figures/dual_core_carrier_kinetics.png`。

### 4.4 双 core 上的 EE/E->I 表达剂量重标定

在 Node、Z/M、AMPA/GABA kinetics、拓扑、延迟和每个 target 的入射预算全部冻结后，仅缩放已经学得的 EE 与 E->I coefficient row。开发 screen 为 5 个 `g_EE` × 4 个 `g_EtoI` × 3 个共同 network seeds，共 60/60 完成；随后 4 个候选加 paired Node 进入 12 s selection，最终按冻结连续目标选择 `g_EE=0.5, g_EtoI=1.0`。这不是重新学习连接图，只是重新标定两条 learned redistribution 的表达强度。

正式 confirmation 使用 12 个全新 paired network seeds、20 s/seed，共 24/24 完成：

| 指标 | paired Node | `g_EE=0.5, g_EtoI=1.0` |
|---|---:|---:|
| OOD，越低越好 | 0.466 | 0.373 |
| Mode 2 share | 0.188 | 0.358 |
| natural KMeans alignment | 0.693 | 0.645 |
| returned events/network | 93.8 | 95.1 |
| Mode 1/2 招募跨度 | 55.1 / 38.4 ms | 52.0 / 50.7 ms |
| timing-only 三周期事件 | 1.67% | 2.10% |
| 群体同步三周期事件 | 0% | 0% |

以 network seed 为独立单位的 paired bootstrap 显示：OOD 改变为 -0.092，90% CI -0.129 至 -0.054，12 张网络中 11 张降低；Mode 2 share 改变为 +0.170，90% CI 0.110--0.229，11/12 升高。returned event 数改变 +1.33，90% CI -2.25--5.00，说明 OOD 改善不是靠删掉一半事件得到的。相反，natural KMeans 改变为 -0.048，90% CI -0.153--0.067，未改善且方向略差；患者 Mode 2 比例为 0.691，模型仍只有 0.358。因此这是**患者支持范围和模式占用的部分改善**，不是完整间期分布恢复。

原生载波审计还发现旧 timing-only 判据的漏洞：它只要求三个间隔规则的局部峰，没有要求峰代表群体同步；在基线接近零时，单神经元量化脉冲也会入选。保留旧字段作历史诊断后，新增每个周期至少对应 1 ms 内 5% core 神经元同步的幅度条件。两臂在新指标上均为 0%；峰值最大的旧阳性也只达到 48.3 Hz，未越过 50 Hz 群体线，未滤波电流表现为宽脉冲而非多周期振荡。故通路剂量重标定没有恢复原生 HFO carrier。

可复现产物：

- `results/topic4_sef_hfo/data_driven_dual_core_ood/pathway_refit/confirmation/aggregate.json`
- `results/topic4_sef_hfo/data_driven_dual_core_ood/pathway_refit/confirmation/figures/dual_core_pathway_confirmation.png`
- `results/topic4_sef_hfo/data_driven_dual_core_ood/pathway_refit/confirmation/figures/dual_core_pathway_refit_fig2c_mode_check.gif`

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

1. 保持两个 Node core 冻结，先审计两种模式是否由不同 core 优先起核；不要把模式标签直接当 core 标签。
2. EE/E->I 表达剂量重标定已经完成：它稳定改善 OOD 和 Mode 2 占用，但没有改善 natural KMeans，也没有达到患者 31/69 比例。继续扩大同一二维剂量网格的边际价值低。
3. 将绝对招募跨度加入正式端点；原有 0--1 onset 特征继续负责顺序，不再单独代表时间尺度。
4. 原始读出、最小 AMPA/GABA 网格和 EE/E->I expression refit 均已完成，并共同裁定为 `NATIVE_HF_CARRIER_NOT_RECOVERED`。下一步若仍要求局部 HFO carrier，应增加明确能形成 fast E/I recurrent oscillation 的机制自由度；不能继续用 bandpass 后振铃或 timing-only 单神经元峰作为优化量。
5. `AMPA=2.0/GABA=18 ms` 只作为 repertoire 候选进入新网络、20 s confirmation；只有独立确认仍改善 OOD/KMeans，才允许替换原 kinetics。载波阴性不因 repertoire 改善而撤回。
6. 每个候选继续保留 Fig.2C-style 全神经元 GIF；禁止只展示最像的事件而不报告总体 OOD。

## 7. 产物与复现

- 配置：`config/topic4_dual_core_ood_node_pathways.json`
- 最终统计：`results/topic4_sef_hfo/data_driven_dual_core_ood/final_analysis.json`
- per-network 表：`results/topic4_sef_hfo/data_driven_dual_core_ood/pathway_per_network.csv`
- confirmation 图与 GIF：`results/topic4_sef_hfo/data_driven_dual_core_ood/confirmation/figures/`
- pathway 图：`results/topic4_sef_hfo/data_driven_dual_core_ood/pathway/figures/`
- raw carrier：`results/topic4_sef_hfo/data_driven_dual_core_ood/carrier_canary/`
- kinetics canary：`results/topic4_sef_hfo/data_driven_dual_core_ood/carrier_kinetics/`
- pathway refit：`results/topic4_sef_hfo/data_driven_dual_core_ood/pathway_refit/`

初始双 core 与四臂运行共 176 个 worker 单元：fit 98、selection 18、confirmation 12、pathway 48；后续 pathway refit 另有 99 个单元：screen 60、selection 15、paired confirmation 24。全部完成，无失败单元。所有长跑均由后台 service/nohup controller 管理，阶段完成后退出；最终分析按 network seed 做配对 bootstrap。
