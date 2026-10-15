# 统一 SNN 开发入口

唯一维护目录：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-continuous-core-state-r1/`。分支：`codex/snn-unified-20261015`。新实验和改动在此进行，仍须遵守各实验的冻结合同；已经启动的批次继续使用其原始依赖。

## 本次整合的选择

- 保留 continuous 的 `src/snn_engine/kick_probe.py`，包括默认关闭的初始电压、观察回调与外部抑制输入接口。
- 两边其他17个共有引擎文件完全一致；保留 continuous 独有的两个 core-input 引擎。
- 加入 substrate 独有的44个脚本、4个源码模块和9个测试。
- `src/topic4_fig5_local_bifurcation.py` 采用 substrate 的后续版本，保留 z_anchor、求积节点缓存和 fold 求解 Jacobian；两个同名分析脚本采用 substrate 的图例/事件选择改进。旧 continuous 文件有本机备份。
- 从 main 提交 `49769c3ffea58ccc52189b8f0ac31ccdfee99819` 原样补入 `topic4_patient_zm_meanfield.py`、`topic4_dual_core_spatial_z.py`、`topic4_dual_core_spatial_z_delay.py`。这些模块此前存在于主目录，旧分岔测试引用的 worktree 已不在，不再依赖该失效路径才能导入。
- 旧结果只补入统一目录缺失的文件：9250个普通文件和2536个符号链接；不覆盖 unified 原有结果。链接仍指向原本的外部数据，不表示这些外部轨迹已上传 Git。

## 兼容性与边界

22个相关测试文件共150项通过，包括连接/autapse、初始状态、core input、checkpoint、事件目标函数、原生活动正则和局部分岔检查。上一轮三随机种子引擎比较中，22个共同输出字段、RNG 状态及 checkpoint 摘要一致；本次没有再改动该引擎。测试通过不升级任何历史结果的科学验收状态，也不表示重新完成了全队列实验。

活动 cohort 的219个冻结输入在本次修改前后逐一一致。其原始 source_snapshot 有两项**整合前既有**不匹配：主目录的 `scripts/paper_figures/plot_fig3_field_concordance_cohort_stat.py` 和 `scripts/paper_figures/plot_fig3b_interictal_ictal_shared_field.py`。本次不更改两者，也不重写冻结清单。

## 旧 substrate 归档

`/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/` 已退出 HFOsp 的 worktree 登记。原有16697个文件/链接逐项校验未改动；源码、结果、外部数据链接、未提交文件及旧 HEAD `336800aabfed51f0d2a8bf008b90fc1d94e0046c` 保留。它使用独立本机 Git 元数据、detached HEAD、无 remote，以保住依赖 git provenance 的旧读取代码；不在此继续开发。原 worktree 元数据也有备份。

完整本机归档记录在 `/data/hfosp/workspace_audit_20261015/snn_consolidation/`。历史结果仍按各自源版本解释，不能把整合后的源码身份倒填到旧实验。
