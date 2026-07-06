# Fig3 field concordance Data-vs-Null statistic

### field_concordance_cohort_stat.png / field_concordance_cohort_stat.pdf

按参考图风格绘制：每一组都是 `Data` vs `Null` 的 violin + box + subject 点，并用浅灰线连接同一 subject 的配对 Data/Null 值，不显示 subject 名字。三组分别是 `BB 1-45 maxAB`、`BB 1-150 maxAB` 和 `HFA 60-100 maxAB`；都使用当前 maxAB artifact 中可评估的 subject，不写 `All candidates`，也不混入 broad fallback。

**关注点**：BB 1-45 maxAB：n=20，Wilcoxon one-sided p=0.0053，16/20 data>null；BB 1-150 maxAB：n=20，Wilcoxon one-sided p=0.006，15/20 data>null；HFA 60-100 maxAB：n=20，Wilcoxon one-sided p=0.0077，16/20 data>null。这张图展示 cohort-level shift above null；formal pass 仍以 selection-corrected p95/p-value 表为准。`BB 1-150 maxAB` 是新增 sensitivity，原 `bb_auc` 仍是 legacy 1-45 Hz。
