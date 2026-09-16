# Stable-repertoire event RNN v2.3.1：六患者 pilot 验收

> **后续状态（2026-08-02）**：本 pilot 的 row-wise shuffle、circular target
> provenance、recency matching、equal-count 和 endpoint 分层问题已由 v2.4 修复。
> 六位患者现仅作为 development set；v2.3.1 的 chronology-specific 读数被 v2.4
> 锁定扩展取代，不能再作为 cohort 结论。

日期：2026-08-01  
状态：**六患者开发 pilot 完成；开放冻结线性阶梯的全队列扩展，不开放更复杂 GRU**

## 1. 一句话判断

此前 split-half / odd-even block 已证明患者内传播模板总体稳定，这确实提供了建立跨事件 RNN 的条件。重新以“一整场事件”为一个时间步后，模型发现较长事件历史对未来传播 repertoire 有信息；但最普遍的信息首先是过去 80 场的**无序组成**，不是复杂非线性顺序计算。三位患者进一步出现跨 `H=20/40` 同向的低维线性顺序增量，GRU 没有提供额外解释。

## 2. 科学对象与数据合同

- 每个 recurrent step 是一整场间期事件，不是事件内部 rank step。
- 目标是未来 `H` 场事件的 mode occupancy、contact mean rank 和 participation 分布。
- 主分析 `L=80, H=20`；预先声明的敏感性为 `L=80, H=40`。
- 只使用旧 `train80`，再按完整 source recording 的真实时间做 60/20/20 train/validation/test；旧 heldout20 未进入。
- `K=2` 稳定模板只在 train sources 上重新拟合。没有读取 A/B、病理轴、几何、SOZ、发作或 SNN 信息。
- source 之间重置状态；正式 future windows 不重叠且不跨 source。

## 3. 五轮修正及其作用

### Round 1：把旧稳定性结果恢复为前提

旧 PR-2.5 的结论是 `23/30 strong + 7/30 moderate + 0 weak`，split-half 和 odd/even block 的模板相关中位数分别为 0.899 和 0.985。v2.2 的 block-mean observability Gate 测的是未按稳定模板分层的边际均值，不能推翻这项结果。本轮在 validation/test 中分别独立重聚类，再与 train-only templates 做 Hungarian matching：validation 6/6 strong；test 4/6 strong、2/6 moderate、0 weak。由此 train-only backbone 的 read-back 才算真正复核完成。

### Round 2：按顺序建立 direct ladder

依次实现 static、recent-window ridge、first-order switching、低维线性 event-state 和 GRU event-state。direct linear 在 H=20 一度 4/6 超过当时最强简单基线；但它同时承担“压缩最近状态”和“学习顺序历史”两个任务，不能据此裁决 recurrent necessity。

### Round 3：补未来窗口可靠性和 H=40

对 validation target window 反复 split-half，分别估计 occupancy、rank、participation 的 sampling reliability。H=40 普遍提高可靠性；chenziyang 和 zhangjiaqi 的 H=20 occupancy 仍较噪，因此所有结论同时报告 H=20/40，不删除患者。

### Round 4：加入同信息集无序长历史基线

R1-L 与 RNN 同样读取过去 80 场，但只使用其无序 repertoire descriptor。加入后，direct linear 相对最强无序基线仅 H=20 1/6、H=40 2/6。此前的大部分增益来自更长历史的组成，而不是顺序本身。

随后把 recurrent 模型改为嵌套增量：

\[
\widehat D_{future}
=
\widehat D_{unordered\;80\;events}
+
\Delta_{ordered\;state}.
\]

这使顺序状态只需解释无序长历史尚未解释的 residual。

### Round 5：嵌套线性、嵌套 GRU 与顺序破坏

每个模型都与同事件集的 within-history shuffle 和 source 内 circular input-target shift 比较。合成数据单测确认：当历史组成固定、只有顺序携带信息时，线性 recurrent correction 能恢复已知信号，而 shuffle 不能。

### Round 6：最终 heldout 边界复核

最终审计发现 source 集合虽由 train80 建立，但旧实现从这些 source 取事件时没有再次与
`event_split==0` 相交；80/20 截点所在 source 的 heldout 尾部因此被带入 test。六位受影响事件数
分别为 1,351、12、74、464、995 和 59。修复后增加 final-index 逐事件断言，所有 R0–R4、
H=20/40、三个 seed 和两个顺序 null 全部重跑，旧 checkpoint 被覆盖。修复后的定性结论未反转，
本报告所有数值均来自修复版。

## 4. 六患者结果

| 比较 | H=20 | H=40 | 判断 |
| --- | ---: | ---: | --- |
| 无序 80 场历史优于最近窗口 | 5/6 | 4/6 | 较长历史分布有预测信息 |
| 嵌套线性优于最强无序基线 | 4/6 | 3/6 | 患者异质，不是普遍 cohort 效应 |
| 嵌套线性优于 history shuffle | 4/6 | 4/6 | 部分患者存在顺序增量 |
| 两个 horizon 都满足上两项 | 3/6 | 3/6 | 922、chenziyang、zhangjiaqi |
| 嵌套 GRU 优于嵌套线性 | 1/6 | 1/6 | 非线性 GRU 不必要 |

六患者 sign test 不支持总体同向结论：嵌套线性相对无序基线的单侧 p 值在 H=20 为 0.344、H=40 为 0.656。因此三患者结果是可复现的开发期异质性，不是六患者 cohort claim。

cluster / moving-block bootstrap 进一步显示：

- `epilepsiae_922`：两个 horizon 的 linear-minus-unordered 和 linear-minus-shuffle 95% CI 均完全小于 0；
- `yuquan_zhangjiaqi`：两个 horizon 同样稳定；
- `yuquan_chenziyang`：两个 horizon 点估计同向；由于 test 只有一个 source recording，linear-vs-unordered 的 moving-block CI 均跨 0。H=40 的 linear-vs-shuffle CI 小于 0，H=20 仍跨 0；
- 增量主要来自未来 contact participation，其次是 mode occupancy；mean rank 的额外改善较小。

## 5. 正确的科学结论

当前结果支持两层结论：

1. 重复间期事件反复采样一个患者特异、跨时间可复现的传播 repertoire；
2. 过去较长一段事件的 repertoire 组成能够预测未来窗口，且少数患者存在超出无序组成的低维线性顺序增量。

当前结果不支持：

- 所有患者都需要一个 chronology-sensitive recurrent state；
- GRU 或更复杂非线性 dynamics 是必要的；
- RNN 恢复了 contact-level 生物网络；
- 间期事件因果塑造了患者网络或证明了可塑性。

## 6. 下一步

开放全 34 人扩展，但只冻结运行最小阶梯：static、recent-H、unordered-L、nested linear、history shuffle 和 circular shift。GRU 不进入全队列。全队列应先报告各患者 future-window reliability，再做 patient-first 统计，并检验顺序增量是否构成可解释亚群；旧 heldout20 仍不能称为全新 confirmatory set。

## 7. 主要产物

- 科学合同：`docs/superpowers/specs/2026-08-01-topic5-stable-repertoire-event-rnn-v2_3.md`
- 主验收：`results/topic5_stable_repertoire_event_rnn/development/acceptance_v2_3_1/FINAL_ACCEPTANCE.json`
- 患者 × horizon 表：`results/topic5_stable_repertoire_event_rnn/development/acceptance_v2_3_1/patient_horizon_summary.csv`
- cluster bootstrap：`results/topic5_stable_repertoire_event_rnn/development/acceptance_v2_3_1/cluster_bootstrap.json`
- 主/敏感性模型：`results/topic5_stable_repertoire_event_rnn/development/{v2_3_1_residual_linear,v2_3_1_residual_gru,v2_3_h40_residual_linear,v2_3_h40_residual_gru}/`
- checkpoints：两个 residual-GRU 目录下 `checkpoints/<subject>/<condition>_seed*.pt`

## 8. 工程验收

- C0 数据合同：H=20 6/6，H=40 6/6；
- train-only stable repertoire read-back：两个 horizon 均 6/6；
- residual-GRU training adequacy：两个 horizon 均 6/6；
- 新模块单测 9 项通过，覆盖 source split、共享 source 的 heldout 尾部排除、future blindness、非重叠 target、order/circular controls、family-balanced score、线性/GRU 输出和已知顺序信号恢复；
- 与 v2.2/SIG/SPF 共用基础设施的相关回归测试共 50 项通过；12 个患者×horizon 公共产物逐事件复核后 heldout 索引为 0，108 个 residual-GRU checkpoint 全部可加载；
- 直接 R3/R4 产物保留为 capacity diagnostics，但由嵌套 v2.3.1 在科学判决上取代。
