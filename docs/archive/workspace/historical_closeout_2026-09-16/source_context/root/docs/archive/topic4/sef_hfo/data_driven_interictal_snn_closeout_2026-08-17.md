# Data-driven interictal SNN 收官（2026-08-17）

## 一句话裁定

这条线可以暂时收口。当前最安全的状态是：

```text
DATA_DRIVEN_INTERICTAL_SUBSTRATE_PARTIAL_SUPPORT
/
STATIC_PATHWAY_EFFECTS_ESTIMATED
/
COHORT_CANONICAL_LAYOUT_ADVANTAGE
/
REAL_GEOMETRY_GENERALIZATION_UNRESOLVED
/
ICTAL_UNIFICATION_SEPARATE
```

患者数据约束的 node field 与局部连接，能够在新的 SNN 网络中产生自发、可返回、具有两簇传播
组织的间期 event-like activity，并在冻结患者分类器下部分接近患者的触点招募顺序。它没有证明
完整患者间期活动已被复现，也没有在真实植入几何下建立队列泛化，更没有证明解剖 core、临床波形
或发作生命周期机制。

**最大缺口**：canonical contact-order layout 上的弱正向队列结果在真实电极几何中消失；同一张
网络同时恢复两种患者模式的比例也未达到预注册门槛。

**下一步**：静态间期 field/edge 不再继续追参。只有出现新的 blind patient unit 或解决真实几何
读出合同后才重开；Z/M 发作统一是独立机制问题，不能把本轮间期拟合当作其正向结果。

## 1. 已经回答了什么

### 1.1 Node 与局部连接的角色

- `Node` 通过患者约束的局部兴奋性场改变点火可达性，是 spontaneous event-like activity 的主要
  substrate。
- 旧 rev9 的 `exp(alpha * h_target * h_source)` E→E 重分配在均匀阈值下不能独立稳定点火；它更像
  已点火活动的条件性 relay/amplifier，而不是第二个点火机制。
- rev11 把连接改成连续、局部、成对的 data-driven mapper。EE 与 E→I 分别学习六个局部特征的
  系数：source/target field、距离和有向位移；拓扑与 delay 不变，每个 target 的 incoming budget
  守恒，I→E/I→I 冻结。
- 结构审计覆盖 48/48 workers，incoming E→E/E→I 最大误差分别小于 `1.6e-11` 和 `6.9e-12`，
  edge ratio 范围为 `0.234–2.279`。

这支持的机制分解是：**Node 主要控制点火可达性；EE 改变局部递归传播几何；E→I 改变模式占比和
患者支持范围。** 后两句来自冻结 pathway ablation 的模式效应，仍是模型内的静态机制候选，不是
患者 SEEG 对 EE/E→I 的直接因果识别。

### 1.2 冻结网络确认

在 12 张新配对网络上，冻结 joint candidate 相对 Node 的 composite score 改善为 `-0.0828`，
90% network-bootstrap CI `[-0.1471, -0.0109]`，候选更低的概率为 `0.968`。但 Node 自身也能通过
部分方向和患者几何对照，因此连接不是产生这些模式的已证明必要条件。

自然 KMeans 与患者冻结模式的对应仍不充分。不能把 supervised classifier 下的 Mode 1/2 直接写成
网络自然恢复了两个患者模式。

### 1.3 EE / E→I pathway confirmation

正式确认使用 12 张全新的 paired networks（1581–1592），每臂 20 s：

- E→I-only 相对 Node：Mode 2 share `+7.66` percentage points，90% CI `[+2.89,+12.21]`；
  OOD `-10.15` points，90% CI `[-13.80,-6.59]`。
- Joint 相对 Node：OOD `-18.13` points，90% CI `[-21.56,-14.93]`；但自然 KMeans match
  同时 `-9.73` points，90% CI `[-15.88,-3.39]`。
- EE-only 没有任何 Figure 4C 主 endpoint 的 90% CI 排除零；它仍改变了 event-aligned recurrent
  currents，但没有形成独立的模式级确认。
- Joint 不能写成 synergy：它改善患者 support 的同时损害自然聚类匹配。

Figure 4C 的星号只表示 paired arm-minus-Node 的 90% network-bootstrap CI 不跨零，未做多重比较
校正；它们不是临床效应或患者层统计。

## 2. 34 人队列给出的边界

- canonical contact-order layout：23/34 人的 held-out weakest-mode loss 低于各自 within-shaft
  relabel null；中位优势 `+0.0063`，Wilcoxon `P=0.043`。
- same-network K=2：15/34 人通过，低于预注册的 50% 门槛；Figure 4E 中与 held-out pass 的交集
  11/34 只是描述性亚群。
- real geometry sensitivity：28 人中只有 14 人与 canonical layout 同号；中位优势 `-0.0007`，
  `P=0.98`。

因此队列结果只能写为：**患者特异模型在目标盲的 canonical shaft-row readout 上出现弱的
development-level held-out advantage，但 same-network dual-mode recovery 不足，且真实几何泛化未解决。**

## 3. 不能写的结论

1. 不能写“data-driven SNN 复现了患者完整间期活动”。当前没有比较临床 SEEG 电压波形、频谱、
   振幅、持续时间和完整事件率分布。
2. 不能写“恢复了患者解剖 core”。场是数据约束的建模 substrate；component lesion、matched
   relocation 和完整的微异质性因果审计没有形成患者解剖定位证据。
3. 不能写“EE 或 E→I 是患者模式的临床机制”。患者 SEEG 只约束传播运动学，模型 ablation 只说明
   当前 SNN scaffold 内的可实现分解。
4. 不能写“静态间期结果已经与发作统一”。Z/M、受控高态、终止和恢复有独立验收合同。

## 4. Figure 4 候选

当前作者选择的 A–J 完整布局保存为：

`results/paper-ready-figure/archive/2026-08-19_pre_final_fig4/fig4/figures/fig4-author-layout-candidate.png`

> 2026-08-19 归档说明：该 A–J 图是本报告当时的作者候选，现已被可重建的 A–H 定稿
> `results/paper-ready-figure/fig4/figures/fig4-complete-layout.png` 取代；本段其余判断保留历史时点口径。

- 尺寸：`1402 × 1236 px`
- SHA256：`d62fca2ea7187412964eb5638ded9b2314d5de1e8b773bd50d697a0efef7c9c9`
- 状态：`AUTHOR_PROVIDED_LAYOUT_CANDIDATE`

它把机制示意、底物几何、直接波形、双模式空间读出、34 人队列、事件热图、rank profile、
cross-fit matrix 和 pathway ablation 放在同一页，科学叙事已经完整。由于仓库中没有该 A–J 拼版的
完整 producer，它是当前 Figure 4 的作者布局候选，而不是完全可重建的 `LOCKED` 资产。现有 A–G
panel 包及其冻结 source/metadata 继续保留，作为可重建的数据来源。

## 5. 执行和资源收口

- pathway confirmation、frozen substrate confirmation 和 34 人 cohort formal run 均已退出；
  当前没有 Topic 4 data-driven worker 在运行。
- 当前 data-driven 分支就是主工作目录，不存在可独立删除的 data-driven worktree。现有
  `topic4-fcxr-lc2` 和 `topic4-m4-snn-native-exit` 属于发作/恢复机制线，前者还有未跟踪脚本，未删除。
- 只删除三处未被 git 跟踪、可由冻结 config/seed 重建的 network cache；worker outputs、JSON、
  figures、spec、config 和 provenance 全部保留。详见
  `data_driven_interictal_cache_cleanup_manifest_2026-08-17.md`。
