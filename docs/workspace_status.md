# 工作区状态

2026-09-30 整合中。来源盘点、工件清单及核验记录见[本次整合目录](archive/workspace/integration_2026-09-30/)。远端发布完成并重新确认没有运行依赖之后，才移除闲置checkout。

## 保留的工作区

| 工作区 | 保留原因 |
|---|---|
| 主目录 `/home/honglab/leijiaxin/HFOsp` | 当前改图、Figure 4 core数/位置/队列补实验；保留所有未提交改动及四个既存冲突路径。 |
| `.worktrees/topic4-continuous-core-state-r1` | 正在运行的四条Figure 4实验线直接导入其原生引擎及冻结源文件；当前没有进程以此为cwd不等于可删除。 |
| `.worktrees/topic4-substrate-autapse-fix` | 空间SNN原生基底、患者数据包和重建依赖，仍被现行代码引用；保留未提交更改。 |
| `.worktrees/topic5-ges-v033-training-lab` | Topic 5 RNN尚有独立开发/审阅任务与未提交内容；本轮不把它归为已停止的Figure 5补实验。Topic 5不等于Figure 5。 |
| `.worktrees/main-cherry` | 干净main整合与查阅入口；不运行补实验。 |

## 本次待退役的checkout

`topic4-dual-core-z-bifurcation`、`topic4-rev22-postfit`、`topic4-six-rate-heterogeneity`、`/tmp/hfosp-six-rate-publish-20260917`。前两者的既有历史成果已按原状态归档；六群体发布及v11异质性代码、数组、图和报告纳入本次main。Git分支及提交保留。实际移除记录以本次整合目录的 `retirement.json` 为准。

当前默认分岔入口为[空间rate模型](topic4_model_versions.md)，Figure 5为[70点A–F版](current_figure5.md)。不能根据仍保留在历史日志中的RUNNING/Goal ACTIVE字样推断当前进程，也不能根据整合验收推断正式分岔已经建立。
