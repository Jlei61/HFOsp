# 工作区状态

2026-10-15 已完成工作区盘点和两处临时发布区清理：Git worktree 从 **7 个减至 5 个**。RNN 线按作者要求保留；两个旧 SNN 区仍是现有代码和结果的依赖区。本次没有停止实验进程，也没有清理主目录的未提交工作或四个既存冲突。

## 保留的五个工作区

| 工作区 | 当前用途与保留原因 |
|---|---|
| `/home/honglab/leijiaxin/HFOsp` | 共享工作目录；RNN、患者分析等任务并行使用，另有运行中的控制器。保留现有分支、未提交改动、四个既存冲突及本机资料。 |
| `.worktrees/main-cherry` | main 的整合和查阅入口；本轮统一 Figure 1、3、4 的已接受版本及公共索引。 |
| `.worktrees/topic5-ges-v033-training-lab` | **RNN 并行开发区，作者明确要求保留**；不因提交日期较早而退役，不整体合并其尚在开发的工作内容。 |
| `.worktrees/topic4-continuous-core-state-r1` | 原生空间 SNN 引擎及结果依赖；主目录 26 个代码/配置文件仍引用该路径，包括 `scripts/topic4_zm_onset_cohort/common.py`。175 个未提交文件已在本机另行备份，原位置保持。 |
| `.worktrees/topic4-substrate-autapse-fix` | 原生基底、checkpoint/慢变量及冻结结果依赖；主目录 14 个代码/配置文件仍引用该路径。268 个未提交文件已在本机另行备份，原位置保持。 |

两个 SNN 区暂不能按“长期不提交”删除。相对于本轮开始时的 `origin/main`，经典三方合并预览分别出现49和41个含冲突标记的文件块，涉及历史方法与实现；这些数字不是待完成实验数。旧历史不能作为一个普通代码更新整体覆盖当前主线。继续退役前，须先迁移被引用的确切代码/数据、验证对应消费者，再重新检查活动进程。

## 本轮整合与退役记录

- Figure 4 分支 `codex/fig4-compact-layout-20261009` 的18个提交以 `0f12127b806ce492c0ea99db0e60505882edf81b` 为终点纳入整合。Figure 1/3 保留 `48ffeacabdae555b2d520867b639781f689ecccd` 中的已接受包；三套正式包与各自来源逐文件一致。
- 公共索引的四处冲突按当前版本合并；旧 Figure 3 Timing+Space 组装入口转到当前 A–E producer，旧 A–F 实现保存在历史归档。登记表验证同时支持冻结发布清单，缺失或篡改的导出文件会被拒绝。
- 已无 force 移除 `/tmp/hfosp-fig3-publish-20261010` 与 `/tmp/hfosp-fig4-publish-20261009`；两条本地和远端分支以及原提交均保留。Figure 4 发布区的三份本机覆盖文件先备份，已由整合版本中的 Figure 1/3 正式入口取代。
- 本轮开始时，全部7个工作区的 HEAD 历史都可从至少一个 origin 远端引用到达。**这不表示各工作区的未提交文件已上传。** 两个 SNN 区的新增本机备份仅覆盖175/268个未提交源码与文档文件，不包含全部被忽略的数值结果或 checkpoint；原依赖区及其数据仍保留。

[精确提交、依赖、校验及退役记录](archive/workspace/consolidation_2026-10-15/README.md)。完整本机审计与备份在 `/data/hfosp/workspace_audit_20261015/`。2026-09-30 的历史整合仍见 [integration_2026-09-30](archive/workspace/integration_2026-09-30/)，不能再用其工作区清单推断当前活动状态。
