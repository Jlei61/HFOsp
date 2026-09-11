# 第一批闲置 worktree 归档与关闭

用户于2026-09-11授权按盘点清单处理第一批4个worktree。完整分支与本地AGENTS改动已提交并push，远端SHA逐一核对；目录移除前再次检查未跟踪文件、忽略文件和进程。分支保留，结果数据与运行任务不在本次删除范围。

| 目录 | 已保存的远端分支 | 提交 |
|---|---|---|
| figure-final-sync-20260905 | [codex/figure-final-sync-20260905](https://github.com/Jlei61/HFOsp/tree/2f46052adad76b1412c70454821be10259ecc0ec) | `2f46052a` |
| topic4-dual-core-mechanism-scan | [codex/topic4-rev22-dci-spec](https://github.com/Jlei61/HFOsp/tree/a50b1982be1c59bf39e50ed8df2759a46d03112d) | `a50b1982` |
| topic4-rev22-final-audit | [codex/topic4-rev22-final-audit](https://github.com/Jlei61/HFOsp/tree/0cfb8e0a9dd47ae7e0857a6054d9d9dee1b18453) | `0cfb8e0a` |
| topic4-rev22-isotropic-fix | [codex/topic4-rev22-isotropic-fix](https://github.com/Jlei61/HFOsp/tree/fc6b0918273840d49886062548582901f90aec4c) | `fc6b0918` |

## main集成范围

本次main仅接收此归档索引、保留清单及两份来源补丁。正式图发布分支本已等同main。其余rev22分支历史保存在上表远端，**其模型源码尚未迁入main运行入口**；main当前对应的rev11实现没有rev22的frozen_graph_aspect_ratio和拓扑覆盖接口，不能孤立套用isotropic修复，也不能未经依赖与科学合同核对整体覆盖。

isotropic独立修复原提交为4d9878cb：椭圆算子参考宽高比应核对学习时冻结图，而非当前被null替换的图。原代码与测试随完整分支保存，补丁仅用于追溯，不是已在main通过测试的声明。后续如迁移rev22，须一并迁移依赖、配置和相应集成测试。

3个工作树唯一未提交改动均为同一份AGENTS共享规范入口，已在各自分支独立提交；shared_guidance.patch.gz保留原始内容。旧指导只作历史来源，不替换当前主规范。

## 关闭状态

当前：远端保全已验证，等待此索引进入main后移除4个checkout。移除前占用总计约1.50GB（不跟随符号链接）；实际释放以执行记录为准。

保留其余11个worktree。本轮未替换Figure5候选、未清理shared results、未处理主目录4个冲突。最新Figure5仍由原活跃任务迭代，后续候选发布与旧模型源码迁移均不在此归档完成声明内。
