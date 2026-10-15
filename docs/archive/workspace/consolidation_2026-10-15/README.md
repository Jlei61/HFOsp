# 2026-10-15 工作区整合与清理

作者要求保留 RNN 并行开发，核查其余工作区的提交，合并可接受内容并推送；论文全文只在本机保留。此次操作使用独立整合 checkout，未对共享主目录或 RNN 区做全局暂存、重置、清理或分支切换。

## 核查记录

- [inventory_before.json](inventory_before.json)：7个工作区的准确 HEAD、分支、未提交状态与保留/退役判断。
- [commit_review.json](commit_review.json)：逐工作区列出尚未进入原 origin/main 的提交；所有原 HEAD 的提交历史在 origin 的某个引用中均已存在。
- [runtime_dependencies.json](runtime_dependencies.json)：两个保留 SNN 区的26/14个代码或配置消费者及内容校验值。
- `topic4-*_commit_patch_comparison.txt`：逐提交 patch-equivalence 比较；没有将旧历史视为可直接快进的 main 更新。
- `topic4-*_merge_preview.json`：经典三方合并预览的冲突文件块；只作静态合并预览，没有更改原 worktree。
- `topic4-*_local_backup_manifest.json`：175/268个未提交文件的本机备份清单。备份与原文件逐字节校验；完整路径见工作区状态页。该备份未包含 ignored 结果数据。
- [accepted_package_comparison.json](accepted_package_comparison.json)：Figure 1/3 对比原 main，Figure 4 对比作者定稿分支，三套正式包未改变。
- [validation.json](validation.json)：实际执行的发布包校验、登记表回归测试及合并文件语法检查。
- [retirement.json](retirement.json)：已移除的两处 checkout、移除前的进程/路径/ignored文件检查、保留的提交和分支；均未使用 force。

## 合并范围

将 Figure 4 的18个已发布提交整合到包含当前 Figure 1/3 的 main，解决 `AGENTS.md`、全局来源登记表、论文图登记表和 paper-ready README 的四处冲突。Figure 1 为 `y1_final_20261010`，Figure 3 为 `visual_alignment_compact_rows_20261010`，Figure 4 为 `core_search_final_20261011`；不改变数值、图件字节或作者验收状态。

验证期间发现旧 Figure 3 Timing+Space 入口仍读取已过期的 A–F 登记结构。该入口已与现行入口一致，旧实现另外归档；登记校验改为验证当前发布清单，测试覆盖当前入口、禁止旧panel回退和缺失/篡改文件。生成的 SVG 与 CSV 按其原始字节保留，Git 属性避免将生成器的格式差异误判成需要改写的源文件。

RNN 区按作者要求不合并、不关闭；两个旧 SNN 区仍有运行与复现依赖，保留并说明原因。主目录的4个既存冲突未改动，正在进行的 RNN 和患者分析未被本次操作接管。本机论文正文、入口页面及相应本机段落均不进入发布提交。
