# 工作区状态

2026-09-30 整合及清理完成。已将核验后的模型与结果推送origin/main；重新确认无进程/打开文件/内存映射引用后，无force移除了四个checkout。Git工作区从9个减至5个：**4个工作/依赖区，另1个干净main整合区**。除主目录外，是3个工作/依赖区加1个main区。来源盘点、工件清单及核验记录见[本次整合目录](archive/workspace/integration_2026-09-30/)。

## 保留的工作区

| 工作区 | 保留原因 |
|---|---|
| 主目录 `/home/honglab/leijiaxin/HFOsp` | 当前改图、Figure 4 core数/位置/队列补实验；保留所有未提交改动及四个既存冲突路径。 |
| `.worktrees/topic4-continuous-core-state-r1` | 正在运行的四条Figure 4实验线直接导入其原生引擎及冻结源文件；当前没有进程以此为cwd不等于可删除。 |
| `.worktrees/topic4-substrate-autapse-fix` | 空间SNN原生基底、患者数据包和重建依赖，仍被现行代码引用；保留未提交更改。 |
| `.worktrees/topic5-ges-v033-training-lab` | Topic 5 RNN尚有独立开发/审阅任务与未提交内容；本轮不把它归为已停止的Figure 5补实验。Topic 5不等于Figure 5。 |
| `.worktrees/main-cherry` | 干净main整合与查阅入口；不运行补实验。 |

## 本次已退役的checkout

`topic4-dual-core-z-bifurcation`、`topic4-rev22-postfit`、`topic4-six-rate-heterogeneity`、`/tmp/hfosp-six-rate-publish-20260917`。前两者的既有历史成果已按原状态归档；六群体发布及v11异质性代码、数组、图和报告纳入本次main。所有本地及远端分支/提交保留，只有可再生缓存随checkout移除。实际记录见[retirement.json](archive/workspace/integration_2026-09-30/retirement.json)。

旧Z分岔工作区被历史加载器引用的三个模块，逐字节核对与main已整合代码一致后，补入主目录原本缺失的 `src/` 路径；原加载器已有的fallback可直接读取，活动脚本本身未改动。旧目录移除后的Figure 5导入也已核查。主目录只新增模型路由说明、身份标签及这三个相同依赖模块，未切换其分支或解决其他任务的冲突。

当前默认分岔入口为[空间rate模型](topic4_model_versions.md)，Figure 5为[70点A–F版](current_figure5.md)。不能根据仍保留在历史日志中的RUNNING/Goal ACTIVE字样推断当前进程，也不能根据整合验收推断正式分岔已经建立。
