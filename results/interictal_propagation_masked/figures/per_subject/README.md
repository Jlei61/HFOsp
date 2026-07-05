### per-subject propagation figure

这组图是 Fig2 时序图形式的真实数据 subject-level 素材。完整视觉合同见 `docs/fig2_temporal_propagation_panel_spec.md`。

复现命令：

```bash
python scripts/plot_interictal_propagation.py --masked-features --pr3 --paper-style --max-events 2000
```

2x2 布局：左列宽、右列窄、上下两行紧凑；右侧 rank 列和左侧主图之间保留清楚间距。

- 左上：`Events over time`，原始 lagPatRank heatmap（时间顺序），底部 Day/Night 条带；左上角写 `<dataset>:<subject> | n=<valid_events>`。
- 左下：clustered heatmap，同一 channel order，事件按 KMeans label 排序；stable k=2 主素材中两类写作 `TA` / `TB`，粗红线分隔。
- 右上：`Rank dist.`，不重复显示 y 轴 channel labels；右侧竖向 colorbar 语义为 `First -> Last`。
- 右下：`Mean rank`，同一 y 轴顺序，不显示 y 轴 channel labels，不画 legend。

**关注点**：先看左上真实时间序列是否有重复 rank 结构，再看左下 TA/TB 聚类后是否形成两类稳定模式；右侧两块只作为 rank 分布和均值摘要，不是单独统计检验。
