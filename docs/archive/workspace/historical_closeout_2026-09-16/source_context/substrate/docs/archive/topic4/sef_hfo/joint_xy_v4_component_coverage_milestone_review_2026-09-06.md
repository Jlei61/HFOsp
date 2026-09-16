# V4 双核 XY 分量覆盖实验：四轮里程碑审阅包（2026-09-06）

接手人：本轮执行 Agent（11:31 接手；上一位 Agent 已启动 V4 主服务，本 Agent 接上诊断 watcher、逐轮审计、目视检查并整理本包）。审阅对象：交回上一位 Agent。

## 0. 结论一句话

四轮覆盖实验按预设边界正常结束（控制器自动停在容量审阅点，无失败、无重启），新增 96 个双核位置、24 个八网络候选，**没有任何候选通过任何一项患者分布门槛**；把"参与最好 / rank 最好 / 时差最好"的候选保留进局部搜索和补算之后，参与分量可以做到与母位置持平（门槛 1.2–2 倍），但 V4 的 24 个候选时差分量都在门槛 5.6 倍以上（全池 78 个 ≥4 倍，事件数 ≥64 者 ≥6.7 倍），联合最优仍是 V3 的 r008_003。"搜索覆盖不足"这个解释在参与分量上被检验并**没有**换来联合改善。当前没有合格模型；不进入 E/EI、不进入 Fig5、不启动 V5。

## 1. 朴素话：测了什么、怎么测、揭示了什么

**测了什么。** 一张 20×20 mm 的模拟皮层薄片（32000 个兴奋性、8000 个抑制性神经元，连接按欧氏距离、E→E 连接沿约 −22.8° 的长轴各向异性、不是环面）上放两个"更容易兴奋的小圆斑"（共 1499 个神经元被改了阈值，且不是全部降低：约 27–33% 反而升高）。15 个虚拟电极触点读取放电密度，用与患者数据同一套规则打包成"群体事件"：哪些触点参与、先后顺序、毫秒时差。问题是：只改这两个圆斑的位置（X1/Y1/X2/Y2），能不能让模型事件的整体分布像患者 E10（Epilepsiae 1146）的 30049 个真实间期事件？

**怎么测的。** 模型全部事件和患者全部事件各自映射到同一个固定的 1024 维随机特征空间（参与掩膜 + masked rank + 未裁剪相对时差 + 固定空间矩），算两团点云均值的距离（联合误差），并分别看参与、rank、时差三个分量。"像"的门槛来自患者自己：从患者事件里反复抽 N 个算与全体的距离，取 95% 分位；模型必须比这个自抽样波动还小才算落进患者分布。门槛按"下一个不小于 N 的校准档"取（N 65–127 用 128 档，N ≤64 用 64 档），不足 64 个事件直接不算。位置搜索不改任何模型参数、损失或门槛：每轮 24 个新位置（随机全局起点 + 围绕四个"局部参考"的随机扰动），先用两个训练网络各跑 8 秒初筛，再挑 6 个补另外六个训练网络到八网络、按真实事件数验收。V4 相对 V3 的唯一改动：局部参考和补算提名把参与、rank、时差各自最好的候选也保留下来，检验"三个分量互相牺牲"是不是搜索没搜到造成的。

**揭示了什么。** 四轮里参与专长的子代确实被细化了 17 次、被提名补算 4 次；它们把参与分量做到门槛 1.2–2.1 倍（与母位置 2.1 倍持平），但时差分量 6.7–16 倍、联合 7–16 倍，反而不如联合最优。全池 443 个位置、78 个八网络候选里，没有一个在三个分量上同时优于 V3 的联合最优；时差分量的全池下限（事件数 ≥64）是门槛 6.7 倍，且四轮 96 个新位置没有把它压下去。换句话说，"改位置"这个自由度在当前模型里够不着时差；而原始场目视显示事件多从薄片边缘起始、core 常被经过而非起始，两排触点的相对时序离散度是患者的 3–8 倍。这些指向需要审查模型设定（边界、传播时间尺度、观测桥），不是继续撒位置。

（内部代号：V4 = `joint_rank_space_xy_component_coverage_v4`；三分量 = kernel support / rank_space / timing_space；门槛 = `patient_calibration.json` q95；局部参考 = `local_anchors`；补算 = race / common-seed expansion；终态 = `NEEDS_MODEL_CAPACITY_REVIEW`。）

## 2. 运行快照与完整性

| 项目 | 值 |
|---|---|
| 主服务 | `codex-t4-component-xy-20260906.service`（用户级 transient），MainPID 1529491，11:31:28 启动 → 16:51:24 退出，Result=success，exit 0，NRestarts=0 |
| 诊断 watcher | `codex-t4-component-diagnostics-20260906.service`（unit 文件 `~/.config/systemd/user/`，MemoryHigh 3G / Max 4G），本 Agent 11:45:21 启动，日志 `V4/raw_propagation_audit/watcher.log`，无 Traceback |
| 轮次 | 4/4 完成；终态 `NEEDS_MODEL_CAPACITY_REVIEW`，`goal_remains_active=true` |
| 几何 | 复用 347 + 新增 96（每轮 24）= 443；四轮实际 全局/局部 = 8/16、7/17、19/5、17/7（第 3、4 轮触发停滞规则，随机起点概率升到 0.85） |
| 新仿真 | 336 次 8 s worker（每轮 48 初筛 + 36 补算），exit≠0 = 0，OOM = 0 |
| 复用 | 347 个 V3 几何的评分与轨迹直接复用（无重算）；6 个旧几何被提名时复用其已有两网络轨迹，只补 6 个新网络 |
| 八网络候选 | 78（V3 54 + V4 新几何 18 + V4 补算旧几何 6）；可作局部参考 62 |
| 来源核验 | 冻结输入 3148 项、轨迹文件 696 个（V4 execution 672 + 复用旧轨迹 24）逐一 SHA256 一致；四轮设计/提名/评估重放逐项一致 |
| 各阶段耗时 | 初筛 36 / 37 / 44 / 60 min；补算 30 / 34 / 37 / 34 min；全程 5 h 20 min |
| worker 资源 | 峰值 RSS（VmHWM）6.08 GiB、地址空间峰 6.49 GiB；单 worker 中位耗时 684–1085 s；实际并行 15–17（调度器按 12 GiB/worker + 40 GiB 保留准入） |
| 机器 | 251 GiB 内存（可用始终 ≥187 GiB），80 核（主搜索用 17 核），两块 RTX 3090 全程 0%（worker 为纯 CPU 仿真）；磁盘结束时 129 GiB 空闲；V4 目录 352 MB |
| 未测 | 未做独立的峰值内存压力测试；systemd MemoryMax 未被触及 |

第四轮初筛 60 分钟（其余轮 36–44）的原因是并发负载：审阅方的多参数 pilot 在 15:26–16:11 跑了 14 个 canary worker，系统负载峰值约 22（见 §9 工程问题）。

## 3. 来源清单

代码 / 配置 / 校准（WT `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix`，HEAD `336800aabfed51f0d2a8bf008b90fc1d94e0046c`，分支 `codex/topic4-substrate-autapse-fix`）：

| 项目 | 路径（相对 WT） | SHA256 |
|---|---|---|
| V4 控制器 | scripts/run_topic4_joint_xy_component_search.py | c859abcae73b85b0944a59ac980f098937f98d1ed0acc62f7cbd7da7a79f682f |
| V4 提名/局部参考规则 | src/topic4_xy_component_search.py | 4cd2d9ec517ae236c313473db54f8b798024ae7ac2be509301c2f1964682355f |
| V4 测试 | tests/test_topic4_xy_component_search.py | 9ec523d93a646032be75bbb263400c70900ded92a421b9d96fff033c29351dcc |
| V4 冻结配置 | config/topic4_joint_xy_kernel_v4.json | fdb91945b3e53b36d32dc454885bab5af7d88aaf41ca0baad9bb800f7f98ea08 |
| V4 方案文档 | docs/archive/topic4/sef_hfo/joint_xy_component_coverage_revision_2026-09-06.md | 6f0483df263272828623efa03fe221ce1ec8e8481dc7952101bc22f0833fa23f |
| 患者校准（与 V2/V3 同一文件） | results/…/joint_rank_space_dual_core_search_v4/patient_calibration.json | cba9814d4ce89de5aaa903bedeae66fe6265ce16aa2988e12509fc0c8951b795 |
| 目标合同（3146 项来源哈希） | results/…/v4/objective_contract.json | 2dbe4ae98eaad6f93815c098818f2c31bc37ad4653d906085ff0028bb1c26aaa |
| 核目标 | src/topic4_joint_xy_kernel_objective.py | c8f672acd75daab0747ed826bac54b67b15b76e8cf4f6e401280603f9d8c00d2 |
| 观测/特征 | src/topic4_joint_xy.py | 2b8d7f1329ad9aaf3dfe8d8281bc73b4db73ed5bdcc3af755a49c35de874642a |
| 局部参考资格 | src/topic4_xy_replicated_anchors.py | efb162ce1d27d6f1ff0dc4527105053bdb567f02968236afb14dbc6691b23e51 |
| 底层执行（v1 adaptive） | scripts/run_topic4_joint_xy_adaptive.py | f19e5877d1934838322f70dbc883315a25302e53b1a9dd0c0bf6f3190af6cea3 |
| 诊断 watcher / 路由 | scripts/watch_topic4_xy_component_diagnostics.py / scripts/render_topic4_xy_component_diagnostics.py | 774b4b470092f49bd23682530243e6341789d35e5cc18e73c5c7a6a4e43b15e0 / c8e18ad92f22a5cb9e9b4a3e955f7924e73092947412dd12160ba406eef1c23c |
| 原始传播 producer | scripts/audit_topic4_xy_raw_propagation_video.py | 41296bc08d7e99beb268511d38f45712029bb3c790490148cd2ffea5c6978136 |
| 整轮候选图 producer | scripts/paper_figures/plot_topic4_joint_replicated_expanded.py | 3dc1e62c6cfd5c19f0d2de7454762abff8f4f6a35778fe5bc8a81ca8a16543a5 |
| 患者 GIF | /home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/supplementary-video-1.gif | 5612da9fcc19ac20643b690117ba8313162285785b982789636a7d3fce09c9c5 |
| 患者 cache | /home/honglab/leijiaxin/HFOsp/results/interictal_propagation_masked/event_envelope_fields/epilepsiae_1146_event_envelope_field_cache.npz | d2fd193cbb30c7cd30a8c173c955ed300d08e657bac8eb43f0d29c7b9a0a9a8b |

Seeds：V4 master seed 3028316138（V3 2395812204），每轮 RNG 序列 `[master, round, 0]`；初筛训练网络 2511/2512（8 s），补算共同网络 2513–2518（8 s）；独立确认 seed 从 43000 起（本周期未触发）。网络缓存 `WT/results/topic4_sef_hfo/substrate_autapse_correction/network_cache/`，状态 CORRECTED_GRAPH_VALIDATED（E→E 入度 800、E→I 800、I→E 200，自环 0）。校准 seed 2026090607、64 次抽样、样本档 16/32/64/128/256、q95。

| 档 | joint | D_support | D_order | D_lag ms | direction | kernel support | kernel rank | kernel timing |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.014798 | 0.088371 | 0.010363 | 4.094084 | 0.098101 | 0.017289 | 0.018689 | 0.015973 |
| 128 | 0.008388 | 0.080070 | 0.006356 | 3.077518 | 0.070771 | 0.011747 | 0.012108 | 0.008334 |

本包新增的审计脚本（新文件，不改冻结源码）：

- `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/scripts/audit_topic4_xy_v4_component_cycle.py` — SHA256 `453e11f16bc4a269d14d862ddf6664b23fff1505a0bc19afb9466de85c55ab71`
- `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/scripts/summarize_topic4_xy_v4_component_coverage.py` — SHA256 `75cacc7a76779a2367a648b1aa842d18848f0c5fe8186b20beb26c4d8e0949c4`
- `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/scripts/summarize_topic4_xy_v4_raw_diagnostics.py` — SHA256 `4d95b055c08906f00c6d5109bbb554e30f27b83e965dea47a6ed0c1e2e8c808d`
- `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/scripts/audit_topic4_xy_v4_native_boundary.py` — SHA256 `b0330325f81fc82bd113716662a3793888a6bd9ce152d361b49050bc9e7257d2`
- `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/scripts/summarize_topic4_xy_v4_shaft_relative_timing.py` — SHA256 `929229c9fee1b31c2b965a1388e7d90e765e1eefbb0c4f35d874a8494670b86e`
- `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/scripts/plot_topic4_xy_v4_component_coverage_milestone.py` — SHA256 `6e49332ea7764d84c4c56488fab6dd44db8cf7788fdd18eccdfceddacfb06724`

本包产出（结果根 `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4`）：

- `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/cycle_closure_review.json` — SHA256 `ef3b85e523a4568d6198e693dddcda980841dceae9d56900e7154bb4e47aacba`
- `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/component_coverage_milestone.json` — SHA256 `509fb6062b778c6d0cf3601ebcfafd6674aff9dfa1580dce976d9fe29320e347`
- `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/component_coverage_milestone_eight_network_table.csv` — SHA256 `84e7ce857a8a0aca66bb5099026b9454f7b4e5d6655489de0b423d059aae7033`
- `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/scl_icl_relative_timing_descriptive.json` — SHA256 `101d43912332b54e203773fd3fa9cb210882eca504c4ac4e6d1625e966ced885`
- `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/figures/v4_component_coverage_milestone.png` — SHA256 `eb2333d7a1d612876e0b793fb836c79da44147a6f18f5be795c97d8d62d948d6`
- `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/figures/README.md` — SHA256 `51adedff942526d5f5dc82338371a718c3bb480d2595f252aee55db96a960263`

逐轮原始诊断数值审计：

- 第 1 轮：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/rounds/001/raw_propagation_numeric_audit.json`
- 第 2 轮：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/rounds/002/raw_propagation_numeric_audit.json`
- 第 3 轮：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/rounds/003/raw_propagation_numeric_audit.json`
- 第 4 轮：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/rounds/004/raw_propagation_numeric_audit.json`

逐轮候选图目视记录（独立文件，不改 watcher 锁定输出）：

| 轮 | 候选图目视记录 | 状态 |
|---|---|---|
| 1 | `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/rounds/001/figures/visual_qa.json` | PNG 与渲染 PDF 一致；作者未验收 |
| 2 | `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/rounds/002/figures/visual_qa.json` | PNG 与渲染 PDF 一致；作者未验收 |
| 3 | `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/rounds/003/figures/visual_qa.json` | PNG 与渲染 PDF 一致；作者未验收 |
| 4 | `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/rounds/004/figures/visual_qa.json` | PNG 与渲染 PDF 一致；作者未验收 |

## 4. 逐轮重放审计与八网络结果（全部候选、全部门槛）

## 表 A：逐轮提案与提名重放

| 轮 | 诊断 | 全局/局部 | 各锚点子代数 | 初筛事件数范围 | 越界几何 | 专长提名(参与/rank/时差) | 本轮新位置进入提名 | 合格 |
|---|---|---|---|---|---|---|---|---|
| 1 | multi_anchor_local_plus_random | 8/16 | r007_009×6, xy_whole_sheet_007×6, r008_003×3, r001_004×1 | [9, 24] | 3 | joint_r001_012 / cr001_015 / cr001_023 | 5 | 0 |
| 2 | multi_anchor_local_plus_random | 7/17 | xy_whole_sheet_007×6, r001_004×5, r008_003×2, cr001_015×4 | [11, 23] | 3 | cr002_022 / cr002_023 / cr001_003 | 4 | 0 |
| 3 | increase_random_restart_fraction | 19/5 | xy_whole_sheet_007×1, cr001_015×3, r008_003×1 | [8, 25] | 5 | r006_010 / cr002_015 / cr003_003 | 2 | 0 |
| 4 | increase_random_restart_fraction | 17/7 | xy_whole_sheet_007×4, r001_004×2, r008_003×1 | [7, 25] | 3 | xy_interior_058 / xy_interior_032 / cr004_023 | 3 | 0 |

## 表 B：V4 各轮补算后的八网络结果（全部候选，全部门槛）

| 轮 | 候选 | 来源(提案/锚点) | N | 联合 (门槛) | kernel 参与 | kernel rank | kernel 时差 | D_support | D_order | D_lag ms | 方向 | 初筛→八网络联合 | 结果 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | joint_r001_012 | 局部/— | 75 | 0.141213 (0.008388) | 0.0282/0.0117 | 0.0599/0.0121 | 0.1368/0.0083 | 0.1173/0.0801 | 0.0487/0.0064 | 13.98/3.08 | 0.3107/0.0708 | 0.1711→0.1412 | 未通过，9项失败 |
| 1 | component_r001_015 | 全局/— | 75 | 0.061283 (0.008388) | 0.0468/0.0117 | 0.0462/0.0121 | 0.0617/0.0083 | 0.1034/0.0801 | 0.0314/0.0064 | 13.17/3.08 | 0.2250/0.0708 | 0.0843→0.0613 | 未通过，9项失败 |
| 1 | component_r001_023 | 局部/ws007 | 77 | 0.093856 (0.008388) | 0.1191/0.0117 | 0.0967/0.0121 | 0.0950/0.0083 | 0.1922/0.0801 | 0.0408/0.0064 | 17.03/3.08 | 0.2196/0.0708 | 0.0807→0.0939 | 未通过，9项失败 |
| 1 | component_r001_001 | 局部/r007_009 | 79 | 0.078882 (0.008388) | 0.0428/0.0117 | 0.0546/0.0121 | 0.0805/0.0083 | 0.1122/0.0801 | 0.0313/0.0064 | 14.53/3.08 | 0.2247/0.0708 | 0.0797→0.0789 | 未通过，9项失败 |
| 1 | component_r001_008 | 局部/r007_009 | 49 | 0.114438 (0.014798) | 0.0963/0.0173 | 0.1192/0.0187 | 0.0948/0.0160 | 0.1575/0.0884 | 0.0594/0.0104 | 14.30/4.09 | 0.2301/0.0981 | 0.0988→0.1144 | 未通过，10项失败；事件不足 |
| 1 | component_r001_010 | 全局/— | 59 | 0.086280 (0.014798) | 0.1162/0.0173 | 0.1118/0.0187 | 0.0895/0.0160 | 0.1705/0.0884 | 0.0415/0.0104 | 16.12/4.09 | 0.2173/0.0981 | 0.0857→0.0863 | 未通过，10项失败；事件不足 |
| 2 | component_r002_022 | 局部/ws007 | 76 | 0.134813 (0.008388) | 0.0247/0.0117 | 0.0442/0.0121 | 0.1347/0.0083 | 0.0845/0.0801 | 0.0390/0.0064 | 15.79/3.08 | 0.2891/0.0708 | 0.0906→0.1348 | 未通过，9项失败 |
| 2 | component_r002_023 | 局部/r001_004 | 67 | 0.127347 (0.008388) | 0.0458/0.0117 | 0.0566/0.0121 | 0.1346/0.0083 | 0.1103/0.0801 | 0.0455/0.0064 | 15.49/3.08 | 0.3253/0.0708 | 0.0810→0.1273 | 未通过，9项失败 |
| 2 | component_r001_003 | 局部/ws007 | 64 | 0.133831 (0.014798) | 0.0331/0.0173 | 0.0891/0.0187 | 0.1143/0.0160 | 0.0966/0.0884 | 0.0699/0.0104 | 13.08/4.09 | 0.3603/0.0981 | 0.0940→0.1338 | 未通过，9项失败 |
| 2 | component_r002_009 | 全局/— | 62 | 0.110461 (0.014798) | 0.0999/0.0173 | 0.0722/0.0187 | 0.1162/0.0160 | 0.1347/0.0884 | 0.0787/0.0104 | 17.62/4.09 | 0.2310/0.0981 | 0.0833→0.1105 | 未通过，10项失败；事件不足 |
| 2 | component_r002_002 | 局部/r001_004 | 71 | 0.073286 (0.008388) | 0.0538/0.0117 | 0.0495/0.0121 | 0.0779/0.0083 | 0.1241/0.0801 | 0.0448/0.0064 | 14.95/3.08 | 0.2183/0.0708 | 0.0926→0.0733 | 未通过，9项失败 |
| 2 | replicated_r008_011 | 全局/— | 80 | 0.109137 (0.008388) | 0.1474/0.0117 | 0.0987/0.0121 | 0.1186/0.0083 | 0.1453/0.0801 | 0.0629/0.0064 | 18.44/3.08 | 0.2974/0.0708 | 0.0998→0.1091 | 未通过，9项失败 |
| 3 | replicated_r006_010 | 全局/— | 95 | 0.140644 (0.008388) | 0.0291/0.0117 | 0.0701/0.0121 | 0.1315/0.0083 | 0.0946/0.0801 | 0.0843/0.0064 | 15.73/3.08 | 0.2947/0.0708 | 0.2252→0.1406 | 未通过，9项失败 |
| 3 | component_r002_015 | 局部/r001_004 | 74 | 0.123836 (0.008388) | 0.0440/0.0117 | 0.0404/0.0121 | 0.1355/0.0083 | 0.0961/0.0801 | 0.0288/0.0064 | 18.30/3.08 | 0.3240/0.0708 | 0.2757→0.1238 | 未通过，9项失败 |
| 3 | component_r003_003 | 局部/cr001_015 | 87 | 0.106778 (0.008388) | 0.0378/0.0117 | 0.0570/0.0121 | 0.1059/0.0083 | 0.0862/0.0801 | 0.0453/0.0064 | 14.66/3.08 | 0.3191/0.0708 | 0.0738→0.1068 | 未通过，9项失败 |
| 3 | component_r003_006 | 局部/cr001_015 | 69 | 0.082095 (0.008388) | 0.0335/0.0117 | 0.0307/0.0121 | 0.0923/0.0083 | 0.0833/0.0801 | 0.0230/0.0064 | 15.77/3.08 | 0.2277/0.0708 | 0.0929→0.0821 | 未通过，9项失败 |
| 3 | component_r001_006 | 局部/r008_003 | 70 | 0.068966 (0.008388) | 0.0780/0.0117 | 0.0601/0.0121 | 0.0712/0.0083 | 0.1391/0.0801 | 0.0527/0.0064 | 17.55/3.08 | 0.2285/0.0708 | 0.0966→0.0690 | 未通过，9项失败 |
| 3 | replicated_r007_007 | 局部/— | 71 | 0.092243 (0.008388) | 0.1032/0.0117 | 0.0915/0.0121 | 0.0972/0.0083 | 0.1680/0.0801 | 0.0599/0.0064 | 15.78/3.08 | 0.2534/0.0708 | 0.0948→0.0922 | 未通过，9项失败 |
| 4 | xy_interior_058 | sobol_4d/— | 72 | 0.242625 (0.008388) | 0.0303/0.0117 | 0.0670/0.0121 | 0.2387/0.0083 | 0.0973/0.0801 | 0.0819/0.0064 | 20.40/3.08 | 0.3608/0.0708 | 0.1890→0.2426 | 未通过，9项失败 |
| 4 | xy_interior_032 | sobol_4d/— | 85 | 0.106412 (0.008388) | 0.0294/0.0117 | 0.0396/0.0121 | 0.1104/0.0083 | 0.0702/0.0801 | 0.0510/0.0064 | 16.82/3.08 | 0.2698/0.0708 | 0.1394→0.1064 | 未通过，8项失败 |
| 4 | component_r004_023 | 局部/ws007 | 61 | 0.111795 (0.014798) | 0.0204/0.0173 | 0.0523/0.0187 | 0.1067/0.0160 | 0.0745/0.0884 | 0.0514/0.0104 | 15.11/4.09 | 0.3079/0.0981 | 0.0995→0.1118 | 未通过，9项失败；事件不足 |
| 4 | component_r004_007 | 全局/— | 81 | 0.067176 (0.008388) | 0.0854/0.0117 | 0.0691/0.0121 | 0.0696/0.0083 | 0.1761/0.0801 | 0.0360/0.0064 | 14.42/3.08 | 0.1819/0.0708 | 0.0875→0.0672 | 未通过，9项失败 |
| 4 | component_r004_020 | 全局/— | 78 | 0.070271 (0.008388) | 0.0804/0.0117 | 0.0616/0.0121 | 0.0777/0.0083 | 0.1495/0.0801 | 0.0850/0.0064 | 15.62/3.08 | 0.2018/0.0708 | 0.0817→0.0703 | 未通过，9项失败 |
| 4 | component_r001_021 | 全局/— | 67 | 0.089022 (0.008388) | 0.0256/0.0117 | 0.0472/0.0121 | 0.0918/0.0083 | 0.0684/0.0801 | 0.0666/0.0064 | 13.95/3.08 | 0.2329/0.0708 | 0.1085→0.0890 | 未通过，8项失败 |

## 表 C：V3+V4 全部八网络候选按联合误差排序（前 15）

| 排名 | 候选 | 周期 | N | 档 | 联合/门槛 | 参与核比 | rank 核比 | 时差核比 | D_support 比 | D_order 比 | D_lag 比 | 方向比 | 失败项数 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | replicated_r008_003 | V3 | 75 | 128 | 0.057722/0.008388 | 6.42 | 4.89 | 6.91 | 1.78 | 7.30 | 4.43 | 2.80 | 9 |
| 2 | replicated_r006_013 | V3 | 61 | 64 | 0.059542/0.014798 | 2.91 | 2.31 | 4.03 | 1.19 | 6.06 | 3.31 | 1.87 | 10 |
| 3 | component_r001_015 | V4_new_geometry | 75 | 128 | 0.061283/0.008388 | 3.99 | 3.81 | 7.41 | 1.29 | 4.93 | 4.28 | 3.18 | 9 |
| 4 | replicated_r007_009 | V3 | 69 | 128 | 0.065655/0.008388 | 4.94 | 3.72 | 8.56 | 1.63 | 5.22 | 4.96 | 2.62 | 9 |
| 5 | component_r004_007 | V4_new_geometry | 81 | 128 | 0.067176/0.008388 | 7.27 | 5.71 | 8.35 | 2.20 | 5.66 | 4.69 | 2.57 | 9 |
| 6 | component_r001_006 | V4_new_geometry | 70 | 128 | 0.068966/0.008388 | 6.64 | 4.97 | 8.54 | 1.74 | 8.29 | 5.70 | 3.23 | 9 |
| 7 | replicated_r007_005 | V3 | 77 | 128 | 0.069507/0.008388 | 10.19 | 7.35 | 8.78 | 2.09 | 4.77 | 4.51 | 2.98 | 9 |
| 8 | replicated_r005_020 | V3 | 70 | 128 | 0.070242/0.008388 | 6.65 | 5.20 | 9.08 | 1.59 | 9.46 | 4.96 | 3.20 | 9 |
| 9 | component_r004_020 | V4_new_geometry | 78 | 128 | 0.070271/0.008388 | 6.84 | 5.09 | 9.32 | 1.87 | 13.37 | 5.07 | 2.85 | 9 |
| 10 | replicated_r002_007 | V3 | 76 | 128 | 0.070465/0.008388 | 7.09 | 5.76 | 9.17 | 1.91 | 8.68 | 4.21 | 2.85 | 9 |
| 11 | replicated_r002_013 | V3 | 73 | 128 | 0.071172/0.008388 | 11.87 | 7.99 | 9.37 | 2.59 | 6.92 | 5.23 | 3.21 | 9 |
| 12 | xy_whole_sheet_062 | V3 | 75 | 128 | 0.071579/0.008388 | 7.74 | 6.04 | 9.59 | 2.21 | 8.61 | 5.71 | 3.00 | 9 |
| 13 | component_r002_002 | V4_new_geometry | 71 | 128 | 0.073286/0.008388 | 4.58 | 4.09 | 9.35 | 1.55 | 7.05 | 4.86 | 3.08 | 9 |
| 14 | replicated_r002_014 | V3 | 91 | 128 | 0.074324/0.008388 | 4.43 | 4.07 | 8.85 | 1.21 | 3.30 | 4.32 | 3.48 | 9 |
| 15 | replicated_r007_014 | V3 | 80 | 128 | 0.074358/0.008388 | 5.28 | 4.62 | 9.86 | 1.52 | 6.85 | 4.75 | 2.64 | 9 |

## 表 D：分量极值（可作局部参考的八网络候选）

- V3（n=42）：joint=`replicated_r008_003` joint 0.057722 [sup 0.0754, rank 0.0592, tim 0.0576]；support=`xy_whole_sheet_007` joint 0.119138 [sup 0.0245, rank 0.0502, tim 0.1128]；rank_space=`replicated_r001_004` joint 0.124271 [sup 0.0302, rank 0.0276, tim 0.1302]；timing_space=`replicated_r008_003` joint 0.057722 [sup 0.0754, rank 0.0592, tim 0.0576]
- V4（n=20）：joint=`component_r001_015` joint 0.061283 [sup 0.0468, rank 0.0462, tim 0.0617]；support=`component_r002_022` joint 0.134813 [sup 0.0247, rank 0.0442, tim 0.1347]；rank_space=`component_r003_006` joint 0.082095 [sup 0.0335, rank 0.0307, tim 0.0923]；timing_space=`component_r001_015` joint 0.061283 [sup 0.0468, rank 0.0462, tim 0.0617]
- all（n=62）：joint=`replicated_r008_003` joint 0.057722 [sup 0.0754, rank 0.0592, tim 0.0576]；support=`xy_whole_sheet_007` joint 0.119138 [sup 0.0245, rank 0.0502, tim 0.1128]；rank_space=`replicated_r001_004` joint 0.124271 [sup 0.0302, rank 0.0276, tim 0.1302]；timing_space=`replicated_r008_003` joint 0.057722 [sup 0.0754, rank 0.0592, tim 0.0576]

最差分量比最小的候选：
- replicated_r006_021 (V3)：最差分量 6.68×；参与 2.15× / rank 1.84× / 时差 6.68×；联合 6.37×
- replicated_r008_003 (V3)：最差分量 6.91×；参与 6.42× / rank 4.89× / 时差 6.91×；联合 6.88×
- component_r001_003 (V4_new_geometry)：最差分量 7.15×；参与 1.92× / rank 4.77× / 时差 7.15×；联合 9.04×
- component_r001_015 (V4_new_geometry)：最差分量 7.41×；参与 3.99× / rank 3.81× / 时差 7.41×；联合 7.31×
- component_r004_007 (V4_new_geometry)：最差分量 8.35×；参与 7.27× / rank 5.71× / 时差 8.35×；联合 8.01×
- component_r001_006 (V4_new_geometry)：最差分量 8.54×；参与 6.64× / rank 4.97× / 时差 8.54×；联合 8.22×

支配 V3 联合最优三分量者： []

## 表 E：锚点子代贡献

| 轮 | 锚点 | 角色 | 子代数 | 子代到锚点距离 mm | 子代最新联合(八网络者已补算) | 子代被提名 |
|---|---|---|---:|---|---|---|
| 1 | replicated_r008_003 | joint,timing_space | 3 | 0.4, 5.0, 2.3 | 0.069, 0.137, 0.143 | component_r001_006 |
| 1 | xy_whole_sheet_007 | support | 6 | 1.5, 1.2, 3.3, 1.2, 0.9, 3.2 | 0.134, 0.104, 0.214, 0.120, 0.189, 0.094 | component_r001_003, component_r001_023 |
| 1 | replicated_r001_004 | rank_space | 1 | 1.7 | 0.161 | — |
| 1 | replicated_r007_009 | 联合补位 | 6 | 5.5, 4.3, 3.6, 6.3, 3.0, 1.9 | 0.079, 0.135, 0.114, 0.227, 0.149, 0.122 | component_r001_001, component_r001_008 |
| 2 | replicated_r008_003 | joint,timing_space | 2 | 0.9, 1.9 | 0.186, 0.107 | — |
| 2 | xy_whole_sheet_007 | support | 6 | 0.9, 1.3, 0.4, 4.4, 3.6, 0.7 | 0.120, 0.193, 0.226, 0.138, 0.116, 0.135 | component_r002_022 |
| 2 | replicated_r001_004 | rank_space | 5 | 2.3, 0.6, 2.3, 3.8, 0.4 | 0.073, 0.124, 0.132, 0.210, 0.127 | component_r002_002, component_r002_015, component_r002_023 |
| 2 | component_r001_015 | 联合补位 | 4 | 3.8, 1.2, 0.7, 3.6 | 0.235, 0.138, 0.125, 0.136 | — |
| 3 | replicated_r008_003 | joint,timing_space | 1 | 3.7 | 0.162 | — |
| 3 | xy_whole_sheet_007 | support | 1 | 2.0 | 0.196 | — |
| 3 | replicated_r001_004 | rank_space | 0 |  |  | — |
| 3 | component_r001_015 | 联合补位 | 3 | 1.6, 5.0, 4.1 | 0.107, 0.082, 0.146 | component_r003_003, component_r003_006 |
| 4 | replicated_r008_003 | joint,timing_space | 1 | 2.5 | 0.105 | — |
| 4 | xy_whole_sheet_007 | support | 4 | 1.3, 0.8, 0.7, 0.6 | 0.136, 0.151, 0.119, 0.112 | component_r004_023 |
| 4 | replicated_r001_004 | rank_space | 2 | 1.4, 5.4 | 0.096, 0.142 | — |
| 4 | component_r001_015 | 联合补位 | 0 |  |  | — |

## 表 F：原始传播诊断数值审计（每轮全部事件）

| 轮 | 候选 | N | native/readout 时差 MAE 中位/最大 ms | 对 TA 顺序不一致率中位 | 对 TB 中位 | 对 TA 参与差异中位 | 每事件触点数中位 | 边带/内部活跃比 | 目视 |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| 1 | joint_r001_012 | 75 | 1.83 / 30.7 | 0.396 | 0.704 | 1 | 14 | 1.18–1.27 | 未 |
| 1 | component_r001_015 | 75 | 1.95 / 16.7 | 0.366 | 0.636 | 3 | 12 | 1.14–1.23 | 已 |
| 1 | component_r001_023 | 77 | 1.99 / 17.7 | 0.364 | 0.725 | 3 | 12 | 1.13–1.21 | 未 |
| 1 | component_r001_001 | 79 | 1.94 / 15.3 | 0.450 | 0.600 | 4 | 11 | 1.08–1.20 | 未 |
| 1 | component_r001_008 | 49 | 1.76 / 13.2 | 0.629 | 0.412 | 2 | 13 | 1.11–1.20 | 未 |
| 1 | component_r001_010 | 59 | 1.89 / 38.8 | 0.237 | 0.810 | 3 | 12 | 1.14–1.23 | 未 |
| 2 | component_r002_022 | 76 | 1.99 / 39.8 | 0.383 | 0.684 | 1 | 14 | 1.16–1.23 | 未 |
| 2 | component_r002_023 | 67 | 1.83 / 21.4 | 0.340 | 0.765 | 1 | 14 | 1.12–1.21 | 未 |
| 2 | component_r001_003 | 64 | 1.79 / 11.4 | 0.442 | 0.667 | 1 | 14 | 1.15–1.28 | 未 |
| 2 | component_r002_009 | 62 | 1.59 / 13.8 | 0.393 | 0.673 | 4 | 11 | 1.13–1.19 | 未 |
| 2 | component_r002_002 | 71 | 2.04 / 12.2 | 0.368 | 0.692 | 4 | 11 | 1.26–1.38 | 未 |
| 2 | replicated_r008_011 | 80 | 1.83 / 20.2 | 0.346 | 0.731 | 3 | 12 | 1.11–1.22 | 未 |
| 3 | replicated_r006_010 | 95 | 1.78 / 11.0 | 0.526 | 0.531 | 1 | 14 | 1.22–1.25 | 未 |
| 3 | component_r002_015 | 74 | 1.82 / 13.6 | 0.236 | 0.772 | 1 | 14 | 1.14–1.22 | 未 |
| 3 | component_r003_003 | 87 | 1.98 / 18.3 | 0.414 | 0.650 | 3 | 12 | 1.09–1.24 | 未 |
| 3 | component_r003_006 | 69 | 1.91 / 50.9 | 0.343 | 0.735 | 3 | 12 | 1.08–1.22 | 未 |
| 3 | component_r001_006 | 70 | 1.89 / 12.9 | 0.332 | 0.724 | 4 | 11 | 1.15–1.32 | 未 |
| 3 | replicated_r007_007 | 71 | 1.96 / 41.9 | 0.246 | 0.765 | 4 | 11 | 1.27–1.35 | 未 |
| 4 | xy_interior_058 | 72 | 1.96 / 24.8 | 0.573 | 0.424 | 2 | 13 | 1.04–1.19 | 已 |
| 4 | xy_interior_032 | 85 | 1.92 / 33.8 | 0.418 | 0.686 | 3 | 12 | 1.15–1.26 | 已 |
| 4 | component_r004_023 | 61 | 2.08 / 21.1 | 0.409 | 0.667 | 2 | 13 | 1.18–1.27 | 已 |
| 4 | component_r004_007 | 81 | 2.01 / 26.0 | 0.446 | 0.605 | 3 | 12 | 1.14–1.24 | 已 |
| 4 | component_r004_020 | 78 | 1.84 / 10.9 | 0.360 | 0.693 | 4 | 11 | 1.11–1.26 | 已 |
| 4 | component_r001_021 | 67 | 1.93 / 21.4 | 0.460 | 0.643 | 3 | 12 | 1.30–1.44 | 已 |

## 参考位置轨迹

- 第 1 轮前：replicated_r008_003 0.057722；轮后：replicated_r008_003 0.05772181365811115
- 第 2 轮前：replicated_r008_003 0.057722；轮后：replicated_r008_003 0.05772181365811115
- 第 3 轮前：replicated_r008_003 0.057722；轮后：replicated_r008_003 0.05772181365811115
- 第 4 轮前：replicated_r008_003 0.057722；轮后：replicated_r008_003 0.05772181365811115


说明：表 B 各格为"值/门槛"；同一轮内事件数不足 64 的候选使用 64 档门槛，其比例不能与 128 档候选直接排名。表 C 的 replicated_r006_013（N=61，64 档）按原始联合排第二只是跨档效应，它不满足局部参考的事件数条件。表 E 的"子代最新联合"对已补算子代是八网络值。

## 5. 参与提名与局部参考到底贡献了什么

1. **参与专长确实进入了搜索。** 参与锚点 xy_whole_sheet_007 四轮共得到 6+6+1+4 = 17 个子代（距母位置 0.4–4.4 mm），其中 4 个被提名补算（component_r001_023 第 1 轮、component_r001_003 与 component_r002_022 第 2 轮、component_r004_023 第 4 轮）。参与专长提名每轮都实际发生（第 1 轮 joint_r001_012、第 2 轮 r002_022、第 3 轮 r006_010、第 4 轮 xy_interior_058）。
2. **子代保住了参与，没有换来时差。** 四个被补算的参与锚点子代八网络参与核为 0.0204–0.1191（母位置 0.0245）；其中 r002_022 0.0247、r001_003 0.0331、r004_023 0.0204，与母位置持平或更好，但它们的时差核 0.095–0.135（母位置 0.113）、联合 0.094–0.135（母位置 0.119）。参与和时差在这些位置上没有一起动。
3. **分量极值四轮后全部仍是 V3 的候选**：联合/时差 r008_003（0.057722）、参与 xy_whole_sheet_007（0.0245）、rank r001_004（0.0276）。V4 自己的极值：联合/时差 component_r001_015（0.061283）、参与 component_r002_022、rank component_r003_006（0.0307）。
4. **没有任何候选在三个分量上同时优于 V3 联合最优**（支配集为空）。"最差分量比"最小的仍是 V3 的 r006_021（6.68 倍）和 r008_003（6.91 倍）；V4 最好的是 component_r001_003（7.15 倍）与 component_r001_015（7.41 倍）。
5. **专长保留规则的固定成本**：每轮至少 1–2 个补算名额给了单分量极端、联合比同轮最好候选差 3–5 倍的候选（第 3 轮 r006_010、c002_015；第 4 轮 xy_interior_058 联合 0.2426 为四轮最差）。这是规则设计使然，不是错误；但它说明按分量保留并不便宜。
6. **初筛排名仍不稳定**：24 个提名候选中 11 个八网络联合比两网络初筛差（最大变差 0.081→0.127、0.094→0.134；最大变好 0.225→0.141）；专长提名尤其如此（第 2 轮三个专长提名全部变差）。八网络补算是必要的，但它只影响"挑谁"，不改变"没有人合格"。
7. **参考位置四轮未动**：r008_003（0.057722）始终是联合最优；四轮各自最好新候选为 0.0613 / 0.0733 / 0.0690 / 0.0672。

## 6. 原始传播诊断：数值审计与目视记录

### 6.1 全事件数值审计（每轮六候选、全部事件，无筛选）

见 §4 表 F。四轮 24 个候选 1744 个事件记录（不是独立重复：八网络共享 seed 2511–2518）。native 1 mm 格质心与 readout 质心的逐事件成对时差误差中位数 1.6–2.0 ms，最大 11–51 ms；对未配对示例 TA 的顺序不一致率中位数 0.24–0.63、对 TB 0.41–0.81；边带（外侧 2 mm）单位神经元活跃量为内部的 1.04–1.44 倍（描述性）。

### 6.2 两排触点相对时序（描述性，全事件；脚本 `summarize_topic4_xy_v4_shaft_relative_timing.py`）

| 来源 | 事件数 | 两排都参与比例 | SCL−ICL 组中位时差 ms（中位 [IQR]） | 首 SCL − 首 ICL ms 中位 | SCL 参与率 | ICL 参与率 |
|---|---:|---:|---|---:|---:|---:|
| 患者训练分布 | 30049 | 0.98 | -6.2 [16.6] | +3.1 | 0.80 | 0.80 |
| component_r001_001 | 79 | 0.81 | -23.7 [70.6] | -1.4 | 0.64 | 0.80 |
| component_r001_003 | 64 | 0.95 | +26.0 [82.6] | +28.5 | 0.82 | 0.91 |
| component_r001_006 | 70 | 0.73 | -11.3 [85.4] | +2.5 | 0.60 | 0.80 |
| component_r001_008 | 49 | 0.69 | +14.2 [74.8] | +23.2 | 0.56 | 0.97 |
| component_r001_010 | 59 | 0.64 | +8.5 [104.4] | +15.9 | 0.51 | 0.86 |
| component_r001_015 | 75 | 0.89 | -11.0 [75.8] | -3.0 | 0.67 | 0.85 |
| component_r001_021 | 67 | 0.96 | -7.6 [71.7] | +7.6 | 0.81 | 0.79 |
| component_r001_023 | 77 | 0.70 | +20.3 [107.4] | +31.8 | 0.44 | 0.95 |
| component_r002_002 | 71 | 0.77 | -1.2 [87.5] | +10.7 | 0.60 | 0.84 |
| component_r002_009 | 62 | 0.95 | -15.3 [73.1] | -10.3 | 0.91 | 0.73 |
| component_r002_015 | 74 | 0.91 | -2.0 [103.6] | +12.5 | 0.73 | 0.92 |
| component_r002_022 | 76 | 0.92 | -12.8 [92.4] | -5.5 | 0.83 | 0.88 |
| component_r002_023 | 67 | 0.91 | +5.4 [87.2] | +18.4 | 0.84 | 0.91 |
| component_r003_003 | 87 | 0.92 | +2.9 [91.7] | +18.1 | 0.71 | 0.84 |
| component_r003_006 | 69 | 0.81 | -25.0 [71.3] | -7.1 | 0.68 | 0.81 |
| component_r004_007 | 81 | 0.74 | +5.9 [68.3] | +17.6 | 0.46 | 0.88 |
| component_r004_020 | 78 | 0.76 | -9.9 [86.6] | +2.5 | 0.58 | 0.81 |
| component_r004_023 | 61 | 0.97 | -8.9 [88.4] | +1.4 | 0.82 | 0.86 |
| joint_r001_012 | 75 | 0.97 | +8.7 [73.3] | +21.1 | 0.93 | 0.87 |
| replicated_r006_010 | 95 | 0.95 | +14.2 [83.8] | +15.0 | 0.89 | 0.86 |
| replicated_r007_007 | 71 | 0.85 | +0.8 [82.4] | +15.8 | 0.59 | 0.80 |
| replicated_r008_011 | 80 | 0.86 | +0.1 [97.2] | +8.9 | 0.70 | 0.79 |
| xy_interior_032 | 85 | 0.92 | -16.7 [93.7] | -3.5 | 0.74 | 0.85 |
| xy_interior_058 | 72 | 0.97 | -13.3 [126.5] | -11.0 | 0.89 | 0.84 |

患者训练分布里两排触点的相对时序几乎固定（SCL−ICL 组中位 −6.2 ms，IQR 16.6 ms，98% 事件两排都参与）；模型各候选的中位数在 −25 到 +26 ms 之间随位置变化、IQR 50–127 ms、两排都参与的比例 64–97%。**失败形态是排间相对时序的离散度是患者的 3–8 倍（两个方向都有），不是固定方向的晚到**——我最初从一个事件的目视写成"SCL 系统性晚到"，被这张全事件表推翻，已更正。这正是时差核惩罚的量，四轮位置搜索没有把它压下来。

### 6.3 目视记录：看了什么、没看什么

覆盖规则（事先声明）：最终一轮六候选各看最低 seed 首个时间事件的全部 125 帧原始场（用 cuda_env Pillow 11.1.0 解码，80 ms/帧核验，裁出原始场面板拼成 5 张 25 帧分镜逐帧看）+ 一张合成帧 PNG + 全事件 PNG；此外第 1 轮的 component_r001_015（V4 最均衡候选）同样看了 125 帧。第 1–3 轮其余候选只做数值审计，保留完整自动诊断（GIF/CSV/PNG/PDF 均在 `raw_propagation_audit/<候选>/`）。**没有看**：任何候选的第二个及之后事件、其他 seed 的事件、逐候选的 PDF 渲染（整轮候选图的 PDF 已逐轮渲染核对）。

| 候选 | 轮 | seed / 事件 / 窗 ms | 帧数 | 触点数 | 对 TA 顺序不一致率 / 时差 MAE ms / 参与差异 | 记录文件 |
|---|---|---|---:|---:|---|---|
| component_r001_015 | 1 | 2511 / 第 1 个 / 865–1115 | 125 | 15 | 0.316 / 35.2 / 0 | `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/raw_propagation_audit/component_r001_015/figures/visual_qa.json` |
| component_r001_021 | 4 | 2511 / 第 1 个 / 1276–1526 | 125 | 9 | 0.229 / 17.8 / 6 | `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/raw_propagation_audit/component_r001_021/figures/visual_qa.json` |
| component_r004_007 | 4 | 2511 / 第 1 个 / 1302–1552 | 125 | 13 | 0.809 / 41.5 / 2 | `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/raw_propagation_audit/component_r004_007/figures/visual_qa.json` |
| component_r004_020 | 4 | 2511 / 第 1 个 / 1190–1440 | 125 | 10 | 0.432 / 18.3 / 5 | `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/raw_propagation_audit/component_r004_020/figures/visual_qa.json` |
| component_r004_023 | 4 | 2511 / 第 1 个 / 1571–1821 | 125 | 15 | 0.465 / 19.3 / 0 | `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/raw_propagation_audit/component_r004_023/figures/visual_qa.json` |
| xy_interior_032 | 4 | 2511 / 第 1 个 / 678–928 | 125 | 8 | 0.179 / 12.8 / 7 | `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/raw_propagation_audit/xy_interior_032/figures/visual_qa.json` |
| xy_interior_058 | 4 | 2511 / 第 1 个 / 856–1106 | 125 | 15 | 0.351 / 30.7 / 0 | `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/raw_propagation_audit/xy_interior_058/figures/visual_qa.json` |

逐候选原始场观察（细节见各 `visual_qa.json`）：

- **component_r001_015**（第 1 轮，V4 最均衡候选；两 core (5.9,13.9)/(11.1,10.2)）：前 30 帧安静；−18 ms 左边缘（x 0–2，y 8–9）先亮，−6 ms 下边缘另起第二波前，两者在 +10 至 +20 ms 合成沿 ICL 行的斜带向右行进（对应 readout 中 ICL11→ICL1 有序）；+66 ms 起左上角第二段活动覆盖 core A、沿 SCL 行经两 core 之间到 core B 再到右边缘，+160 ms 后消失。三个起始点都在边缘；15 触点全参与但 SCL 比 ICL 晚 75–100 ms（本事件），对 TA 时差 35 ms。
- **xy_interior_058**（第 4 轮参与专长；两 core 都在右侧 x≈15.6–15.9）：−11.5 ms 左边缘（距 core 约 14 mm）先亮，+2 ms 下边缘另起波前，合成 L 形带向右行进；+38 至 +50 ms 抵达两 core 并使其同时活跃；+70 至 +108 ms 第二段活动在 SCL 行上方左侧退向左上角；+130 ms 后第三段沿上边缘自右向左。15 触点全参与，对 TA 时差 30.7 ms、不一致率 0.35。
- **xy_interior_032**（第 4 轮 rank 专长；两 core 相邻在 SCL 行附近）：窗口最初 14 帧右下角有上一段活动残余；−22 ms 右上角边缘活动与 −18 ms core B 内部微弱活动几乎同时出现，−12 至 −4 ms core A 也亮、两 core 之间成带；+8 ms 后带碎裂成多个碎片退向各边缘。readout 只有 8 触点（SCL6–9 与 ICL1–4，ICL5–11 缺失）。全事件缺失模式与其他候选相反（ICL 侧常缺、SCL 侧常在）。
- **component_r004_023**（第 4 轮时差专长、参与锚点子代；core A 紧贴 SCL9/8，core B 在 ICL 行下方）：−29 ms 右上角边缘先亮；−13 至 −9 ms core A 内部自行亮起（此时右上角波前仍在约 10 mm 外），团块覆盖 SCL9/8/7 后分裂，一支向右下成斜带抵达 ICL 行，+25 ms core B 随斜带抵达而活跃，+29 至 +37 ms 沿 ICL 行向两侧铺开，+67 ms 后消失。15 触点全参与，SCL 0–10 ms、ICL 30–50 ms 且 ICL 内顺序不单调；对 TA 时差 19.3 ms、不一致率 0.47。
- **component_r004_007**（第 4 轮联合最优新候选；core A 近 SCL9，core B 在两排之间）：**七个候选里唯一一个 core 先亮且当时没有并发边缘波前的首事件**——−21 ms core A 内部先亮，−9 至 −1 ms 沿两 core 连线扩展抵达 core B；+7 ms 右下角另起边缘活动；+11 至 +49 ms core B 团块向右下扫过 ICL 行右端并与右下角活动汇合；+51 至 +99 ms 一个团块沿 ICL 行自右向左移到左边缘消失。readout 13 触点（缺 SCL9/8），ICL 行自右向左，与 TA 反向（不一致率 0.81）、更像 TB（0.29）。只是一个事件，不能当分布结论。
- **component_r004_020**（第 4 轮全局起点；两 core 水平并排在两排之间）：−55 ms 右上角边缘先亮并沿上、右边缘扩展（距 core 10–12 mm）；−22 ms core B、−8 ms core A 依次被激活，−2 ms 两 core 之间成横贯的带；+1 至 +41 ms 带碎裂成 4–5 个团块分别退向左边缘、右边缘、左上角和沿 ICL 行到右下角，+65 ms 后安静。readout 10 触点（缺 SCL6、ICL10/11、ICL2/3）。readout 敏感性图是六候选中最干净的。
- **component_r001_021**（第 4 轮联合补位；core B 贴着下边缘、core A 在两排之间偏上）：−32 ms core B 内部先亮（当时无其他活动），沿下边缘/ICL 行向两侧扩展并在左边缘留下持续团块；+2 至 +28 ms core A 随之亮起并沿 SCL 方向拉长分裂；+30 至 +58 ms 碎片退向左上角与右边缘，ICL 行另有一条向右的带，+90 ms 后安静。core 先亮但该 core 本身贴边，与边界效应不可区分。readout 9 触点（缺 ICL1、ICL5–9）。


**共同形态。** 七个候选的首事件里，4 个（component_r001_015、xy_interior_058、component_r004_020，以及 component_r004_023 的最早活动）最早出现在薄片边缘或角落，距两个 core 4–14 mm，core 区域是"被波前经过"或随后被激活；3 个出现 core 区域先亮：xy_interior_032（与右上角边缘波前同时）、component_r001_021（先亮的 core 本身贴着下边缘）、component_r004_007（core A 干净先亮、沿两 core 连线到 core B，是唯一一个没有并发边缘波前的例子）。几乎每个 250 ms 事件窗都包含 2–3 段空间上分离的活动，readout 把它们合成一个"事件"，事件末段总是退到某个边缘或角落。这与边带活跃比 >1 的描述统计一致。**这不能据此断言边界伪影或独立源**（原始网格是 1 mm/2 ms 内活跃 E 神经元计数，不是单神经元 raster；一个事件也不是分布），但它是位置搜索改变不了的模型级形态。

## 7. 失败原因分层

1. **事件不足**：只影响 4/24 个候选（N=49、59、62、61），且这些候选按更宽的 64 档也没有任何一项通过。不是主因。
2. **患者分布失败**：24/24 候选 9 项分布门槛（联合、D_support、D_order、D_lag、方向、四个核分量）中至少 8 项未通过；唯一通过的项目是 D_support（向量参与距离）在 3 个 V4 候选上（V3 池已有 4 个），且它们的参与核分量仍未通过。这是主失败面。
3. **网络间不稳定**：初筛→八网络联合变化 −45% 到 +65%；专长提名尤其不稳。影响"挑谁"，不改变结论。
4. **readout 局限**：native/readout 逐事件中位差 1.6–2.1 ms、最大 51 ms（边缘触点）；readout 没有系统性扭曲顺序，也没有掩盖原始场里的边缘起始和多波前。不是主因。
5. **优化覆盖不足**：这是 V4 专门检验的解释。参与专长进入了局部搜索与补算，参与分量能被保住，但时差分量与联合没有随之改善；四个分量极值仍是 V3 候选；无人支配 V3 联合最优。→ 在参与分量上，这个解释**没有**得到支持。四维空间没有穷尽，因此这也不是"数学上无解"的证明。
6. **容量嫌疑**：全池 78 个八网络候选的时差核分量全部 ≥4 倍门槛（事件数 ≥64 者 ≥6.7 倍），排间相对时序离散度 3–8 倍于患者，事件起始多在边缘。位置自由度看起来不控制时差分量——这是"需要审查冻结模型/边界/观测桥"的证据，与审阅方独立两轮审阅（`joint_xy_v4_two_round_independent_review_2026-09-06.md`）的优先级判断一致：冻结模型能否实现所需分布 > 优化器噪声与联合排名取舍 > XY 范围不够。

## 8. P0 / P1

- **P0 科学**：无合格模型；时差分量对 XY 不敏感（全池下限 6.7 倍门槛，四轮 96 个新位置未压下）。
- **P0 科学**：目视的七个首事件里 4 个起始在薄片边缘/角落、core 被经过而非起始，3 个 core 先亮者中只有 1 个没有并发边缘波前；事件末段总退到边缘；边带活跃比 1.04–1.44（描述性）。有限薄片非环面连接的边界效应嫌疑，位置搜索无法消除；需要配对对照才能判断 core 在总体分布上的作用。
- **P1 科学**：排间相对时序离散度 3–8 倍于患者（两个方向都有），不是固定晚到；这是时差核失败的具体形态。
- **P1 科学**：专长保留规则每轮消耗 1–2 个补算名额在单分量极端、联合差 3–5 倍的候选上。
- **P1 科学**：初筛两网络排名不稳（24 个提名里 11 个八网络后变差）。审阅方建议的 12 位置×4 网络初筛可作为下一版本的候选改动，本包不执行。
- **P0 工程（跨 Agent 干扰，由本 Agent 造成）**：审阅方的多参数 pilot（`codex-t4-multidimensional-pilot-20260906.service`，15:26:50 启动）把整个 `scripts/` 目录 1625 个文件写进它的源码锁，包括本 Agent 新建的 6 个审计脚本。16:17:30 本 Agent 为修一个 KeyError 改了自己的 `scripts/audit_topic4_xy_v4_component_cycle.py`，16:17:31 该 pilot 控制器因"locked runtime source changed"报错退出（status.json 为 ERROR，`execution/paired_round1/` 0 个 worker，正式配对轮未开始；其 canary 14 个 worker 与 observer parity 2 个 worker 已在 16:11–16:14 完成）。本 Agent 未触碰该目录、未重启该服务。根因是两个 Agent 在同一 worktree 里各自的控制器锁了重叠的文件集；按其 README，"在相同源码和输入下重新执行控制器即可跳过已验证产物"——但源码已变，需审阅方决定是否以当前源码重新加锁。**本 Agent 在此之后不再改动该 worktree 里的任何脚本。**
- **P1 工程**：调度器按 12 GiB/worker 预留 + 40 GiB 保留准入，使并行数封顶 17（实测峰值 6.1 GiB），CPU 只用 17/80 核、GPU 0%；补算最后一波只有 2 个 worker。冻结配置内不可改；下一版本可把 `screen_minimum_worker_gib` 降到 10（=代码里 1.5×峰值的自适应值）→ 21 并行，初筛/补算各 2 波，每轮约 66→50 分钟。
- **P1 工程**：第四轮初筛被并发 canary 拖慢到 60 分钟（其余轮 36–44）。同一机器上的两条仿真线需要协调调度。
- **P1 工程**：V4 watcher 未包含 V3 的边带活动审计，本包以 `audit_topic4_xy_v4_native_boundary.py` 手动补齐（同一算法，仅参数化结果根）。

## 9. 结论与下一步最小实验（不在本包执行）

**结论**：继续同构 XY 搜索的收益证据不足；应转向"冻结模型/观测能否实现所需分布"的审查。三条互斥解释的最小可区分实验：

A. **时差分量对什么敏感（优先）**。固定 r008_003 与 component_r001_015 两个位置、固定 8 个训练网络（2511–2518）、固定 VTH realization / readout / 损失 / 门槛，单因素改一个传播时间参数一档（E→E 传导延迟或轴突速度；或 E→E 各向异性比）。判据：时差核变化是否超过四轮位置搜索的全部散布（0.058–0.24）；同时看排间时序 IQR 是否从 50–100 ms 收向 16 ms。变的只有一个参数；其余全部固定。
B. **边界效应检验**。同一位置（r008_003）在 24×24 mm（触点几何不变、居中）或环面连接下重跑 2 个训练网络。判据：首事件起始位置是否仍在边缘、边带活跃比是否回到 1、参与/时差分量是否变化。变的只有薄片尺寸或边界条件。
C. **core 控制能力配对对照**（与审阅方 §6.2 一致）。同网络、同噪声、同读出，对照当前 VTH、去掉 core VTH 调制、与旧手放一致的阈值截断；判据：core 是否可重复地改变患者相关分布的任一分量。不声称必要性。
D. **不建议**：V5 同构 XY 搜索；调 E/EI 做机制归因；Fig5。若将来重搜位置，先采用审阅方建议的 12 位置×4 网络初筛并降低 worker 预留。

## 10. 是否进入 E/EI 与 Fig5

**不进入。** 三个前提（患者分布门槛、独立六网络确认、原始传播验收）一项也没满足；本周期没有确认尝试、没有 `qualified_substrate.json`、没有 `fig5_followup/`。

## 11. 文件清单（绝对路径）

- 本报告：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/docs/archive/topic4/sef_hfo/joint_xy_v4_component_coverage_milestone_review_2026-09-06.md`
- 结果根：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4`
- 闭环审计：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/cycle_closure_review.json`（EXHAUSTED_UNQUALIFIED_CYCLE_REPLAY_VERIFIED）
- 覆盖汇总：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/component_coverage_milestone.json`、`component_coverage_milestone_eight_network_table.csv`
- 排间时序：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/scl_icl_relative_timing_descriptive.json`
- 里程碑图：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/figures/v4_component_coverage_milestone.{png,pdf,svg}` + `figures/README.md` + `figures/v4_component_coverage_milestone_metadata.json`
- 逐轮：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/rounds/00N/{design,scores,race_nomination,analysis,raw_propagation_numeric_audit}.json`、`rounds/00N/figures/{candidate_after_event_expansion.*,README.md,visual_qa.json}`
- 逐候选原始诊断：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/raw_propagation_audit/<候选>/{summary.json,all_event_comparisons.csv,all_event_contacts.csv,native_boundary_activity.json,automated_diagnostic_completion.json,figures/*}`；目视记录在 `figures/visual_qa.json`（7 个候选）
- 交接与并发文档：`docs/archive/topic4/sef_hfo/joint_xy_v4_next_milestone_handoff_2026-09-06.md`（本任务的输入）、`joint_xy_v4_two_round_independent_review_2026-09-06.md`（审阅方，截止第 2 轮）、`results/topic4_sef_hfo/multidimensional_interictal_pilot_round1/`（审阅方 pilot，状态 ERROR，见 §8）
