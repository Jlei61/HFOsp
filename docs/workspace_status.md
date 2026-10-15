# 工作区状态

2026-10-15：HFOsp 现在登记 **4个工作区**。两个 SNN 开发区已整合成一个；RNN、共享主目录和 main-cherry 保留。旧 substrate 原路径仅作冻结归档与历史依赖，不再登记为 HFOsp 开发 worktree。数据未删除，实验进程未停止。

| 工作区 | 用途 |
|---|---|
| `/home/honglab/leijiaxin/HFOsp` | 当前共享工作目录，保留并行任务、现有分支、未提交内容和四个既存冲突。 |
| `.worktrees/topic4-continuous-core-state-r1` | **唯一 SNN 开发区**，整合分支 `codex/snn-unified-20261015`。保留原路径以兼容冻结的运行依赖。 |
| `.worktrees/topic5-ges-v033-training-lab` | **RNN 并行开发区**，按作者要求保留，本次不改动。 |
| `.worktrees/main-cherry` | main 查阅、整合检查与发布入口。旧研究分支的科学版本不能直接覆盖 main。 |

## SNN 整合

`substrate-autapse-fix` 的提交 `336800aabfed51f0d2a8bf008b90fc1d94e0046c` 是 continuous 的 `f01db07cdb1dbb28c2695f54ce9742915747e093` 的祖先，continuous 多5个提交；两者并未分叉。以 continuous 为基础保留其新增机制和未提交源码，加入 substrate 独有的57个文件，并吸收分岔模块和两个分析脚本的后续改进。另从 main 补入三个缺失的粗粒化/延迟模型依赖。整合后的150项相关测试通过。

旧 `.worktrees/topic4-substrate-autapse-fix/` 保留原位置及全部既有文件，退出共享 worktree 登记后使用独立、无 remote、detached HEAD 的本机 Git 元数据保存旧版本身份。它是**冻结归档，不是第二个 SNN 开发区**；新工作转到 continuous。旧路径仍被历史代码引用，暂不删除。完整源文件和结果校验清单留在 `/data/hfosp/workspace_audit_20261015/snn_consolidation/`。

[统一 SNN 入口与验证范围](snn_workspace.md) · [本轮整合记录](archive/workspace/snn_consolidation_2026-10-15/README.md)。本次整合没有修改活动批次锁定的219个输入文件；其中两处主目录绘图源码在整合前已有哈希不匹配，记录在验证说明中，未在本次清理中覆盖。

## 已完成的发布区整理

Figure 1/3/4 正式包及公共索引已在 `49769c3ffea58ccc52189b8f0ac31ccdfee99819` 整合并发布。`/tmp/hfosp-fig3-publish-20261010` 和 `/tmp/hfosp-fig4-publish-20261009` 已关闭，其分支、提交和必要本机覆盖备份保留。

[先前7→5工作区的提交与退役记录](archive/workspace/consolidation_2026-10-15/README.md)仍作为历史记录；当前清单以上方四区为准。2026-09-30 的历史整合记录见 [integration_2026-09-30](archive/workspace/integration_2026-09-30/)。
