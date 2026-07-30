### fig6_ordered_history_architecture_audit.png

这张六面板图依次回答：模型究竟学习哪一种事件内历史、这种顺序增量是否跨架构存在、是否超过匹配的 rank-shuffle、显式删除/重排历史是否损伤预测、以及冻结后的有序残差能否在静态结构和无序前缀之外对应 clinical-onset 后 `[0,10] s` 的早期发作能量场。F 固定使用既有论文代表病例 E1146，不按本轮 target 表现挑选。

**关注点**：B/C/D 的统计单位均为患者；B 中 7 个预注册递归家族只有 linear-state 通过 family-wise inference；E 是复用 16 人 106 次发作 target 的条件性静态场检验且增量未建立，不是独立验证；整图中的 state 仅指事件 rank 索引上的现象学状态。

### fig6_ordered_history_architecture_audit.pdf

与 PNG 内容和数据版本完全相同的矢量版本，用于论文排版。

**关注点**：排版时保持六个 panel 的科学分工，不把 F 的代表病例扩写为 cohort 或机制证据。
