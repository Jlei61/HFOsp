# SNN工作区整合，2026-10-15

作者要求最终保留四个工作区。本次将两个SNN开发区的代码统一到 `codex/snn-unified-20261015`，旧substrate原位保留为冻结归档并退出共享worktree登记。

- `file_comparison.json`：合并前非忽略文件的双向精确比较；共有文件仅四处不同。
- `integrated_files.json`：57个独有文件、3个选定后续版本及3个main依赖的来源和SHA256。
- `test_paths.json`、`validation.json`：22个测试文件、150项通过；记录既存冻结清单不匹配及验证边界。
- `retirement.json`：原位退役方式、旧HEAD、Git元数据位置；数据没有删除或搬动。
- `substrate_full_preservation.json`：退役前后16697个原有文件/链接一致。完整逐文件清单及原Git控制数据保存在本机审计目录。

代码整合位于SNN分支，main接收公共入口与整理记录。旧SNN历史与main之间的49/41处合并冲突不是两个SNN区之间的冲突；本次没有用旧科研版本覆盖main中的正式图和模型。

[当前四区清单](../../../workspace_status.md) · [SNN入口](../../../snn_workspace.md)
