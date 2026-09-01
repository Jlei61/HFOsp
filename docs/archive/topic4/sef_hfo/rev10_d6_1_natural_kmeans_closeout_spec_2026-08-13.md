# Topic 4 rev10-D6.1：自然比例 KMeans fresh closeout

## 科学问题

在不增加 core 数、不引入 contact-conditioned field、不开放 edge、beta、topology 或 OU 参数的前提下，判断 D6 连续自由场的局部方向是否能在全新网络中稳定改善 Fig.4C 所要求的自然双模式聚类及患者方向一致性。

本轮只关闭 KMeans 证据层，不等同于完整患者间期活动分布通过。

## 冻结候选

从 D6 的 49 个连续场中，在 fresh seeds 不可见时按四个互补轴冻结：warm baseline、自然 pooled KMeans purity、招募规模、contact-split cross-fit patient margin、三轴 Pareto 折中。若类别 winner 重合，用下一 Pareto 候选补足 5 个不同场。

## 执行合同

- fresh network seeds：1341–1346；前 3 张是预声明 canary，后 3 张是扩展，但一次性全部运行。
- 每个候选每张网络固定 16 s，不允许按候选自适应时长。
- common detector、spatial OU、network topology、edge no-op 和 beta closed 全部冻结。
- 主 KMeans：每张网络使用全部 returned、joint、in-distribution clean events，保留自然 A/B 比例。
- 独立单位：network seed；pooled event 仅作描述。
- supervised-mode balanced bootstrap 只作次级诊断。

## 防循环 patient readout

ICL 和 SCL 各自按杆内顺序奇偶交替分成两个 contact folds。一次在 fold 1 上按 patient prototype 分配事件、在互斥 fold 2 上评价 2x2 Spearman matrix；再交换两 fold。正式 margin 是两次 held-contact matrix 的平均，不能在同一组 contacts 上同时分配和评价。

## 探索性裁定

若预冻结候选相对 warm baseline 在至少 4/6 paired networks 上提高 natural KMeans alignment，且 equal-network cross-fit patient margin 为正、无 runaway，则记为连续场局部 KMeans signal retained。该规则只决定是否值得进入后续低维组合或分布目标，不作为完整 Fig.4 或患者间期活动通过门。

## 结论边界

允许：连续场某个局部方向在 fresh networks 上稳定改变自然双簇结构或患者方向几何。

禁止：data-driven 模型已复现完整患者间期活动、已恢复 patient core、edge/beta 必要、患者 blind generalization 或 ictal lifecycle。
