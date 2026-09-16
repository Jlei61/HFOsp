# Group-Event State v0.4.0：G0 执行记录（接口定义、边界修复与机器合同）

日期：2026-09-07。本文件记录 **G0 已完成并已验证** 的内容，以及 G1/G2 的运行安排。
版本保持 v0.4.0。E1125 全程为开发资料，本文件不含任何人体效果结论。

主合同：[v0.4 科学 spec](group_event_state_v0_4_epilepsy_state_scientific_spec_2026-09-07.md)。
执行合同：[v0.4.0 首包与预算](group_event_state_v0_4_0_first_evidence_package_2026-09-07.md)。
Agent 指令：[执行 prompt](group_event_state_v0_4_0_agent_execution_prompt_2026-09-07.md)。

- 代码入口：`src/topic5_group_event_state/v040/`、`scripts/run_group_event_state_v040.py`
- 配置：`config/group_event_state_v040_first_package.json`
- 结果根：`/data/hfosp_group_event_state_epilepsy_state_v040`
- 复用：v0312 的 latent core（`numerics.py`）、事件编码器与 `SlowState`（`model.py`）、
  `conditional_set_lp` / `identity_units`（`frozen.py`）、`controls_mask` / `available_recent`
  （`seizure.py`）、`inventory_crosswalk`（`spatial_transfer.py`）、v0311 `block_event_tables`。

## 1. 相对 v0.3.12 的实际改动

| 位置 | 改动 | 理由 |
|---|---|---|
| `v040/data.py:build_split` | FIT 权限追加 `release<=fit_end` | 分钟包随其闭合 1 小时块发布；仅按包结束筛选不足以证明发布合法 |
| `v040/conditions.py` | 新增共同条件 C（10 字段）并以同一接口给五臂与全部消费者 | 旧 head 只接时钟，背景基线无法与状态臂同权 |
| `v040/model.py:Readout` | 条件维 `LATENT+2` → `LATENT+12` | 同上；`HistoryReference.latent` 同时接目标时钟、C 与已逝时间 |
| `v040/objective.py` | 形态评分改为逐分量路径边缘化 + 族/事件/窗口等权 | spec §6.1 |
| `v040/objective.py` | 旧整包联合分数保留为 `legacy_packet_joint_component_score` | spec §6.1，仅监测 |
| `v040/train.py` | 选模改为 `J_inner = 0.5·spatial@30 + 0.5·morphology@30` | spec §9.1；count/load/旧 1min selection 仅监测 |
| `v040/train.py:run_paired_inner` | 两个 INNER 同进程、共同评估网格、共同 LR 里程碑、按平均 J 选步 | 工作包 §3.1，禁止拼两个 argmin |

## 2. 发布支持与冻结的短历史窗口

按 spec §5.1 的预登记规则只读时间/支持元数据（split_seed=20260906、stride=30、h=30min），
不读损失、未来形态或发作效果。

| stage | 主时距 query | 30min 中位 coverage / 合格 / 合格率 | 120min 中位 coverage / 合格 / 合格率 |
|---|---:|---|---|
| INNER0 | 12 | 0.000 / 1 / 0.083 | 0.158 / 1 / 0.083 |
| INNER1 | 24 | 0.083 / 0 / 0.000 | 0.550 / 18 / 0.750 |
| OUTER（仅描述） | 123 | 0.000 / 61 / 0.496 | 0.725 / 115 / 0.935 |

**两个候选都没有在两个 INNER 上达到 80% 法定人数**，按预定规则以最长候选**替换**，
故 **H_short = 120 分钟，仍为五臂**，30 分钟只作可用性/时延诊断。
与 spec 文内表格逐格一致（release-aware FIT 修复不改变本表，因为支持审计用 `valid_packet`）。

**必须保留的局限**：INNER0 在两小时规则下只有 1/12 合格（0.083）。
不得写成“两小时资格已全面通过”。

### 2.1 短窗是否真的截断前向历史

| stage | 2h 截断生效的 query | 占比 | 生效时中位移除事件数 |
|---|---:|---:|---:|
| INNER0 | 7 / 43 | 0.163 | 37 |
| INNER1 | 64 / 74 | 0.865 | 3255 |
| OUTER | 352 / 376 | 0.936 | 5325 |

即 `S_marks-short` 与 `S_marks` 在 OUTER 上 93.6% 的 query 读到的历史确实不同；
但在 INNER0 上只有 16.3%，与 §2 的 INNER0 支持弱是同一件事。

### 2.2 memory_support

预定义为「短窗 coverage≥0.5、支持末端年龄≤H/2，且短窗之前另有 ≥30 分钟合法已发布曝光」。
INNER0 = 0/12，INNER1 = 17/24，OUTER = 113/123；其中**较早事件数为 0 的 memory_support query 为 0 个**，
即较早历史归因的分层里没有“纯安静较早窗”。

### 2.3 训练截止修复

| stage | 仅按包结束的 FIT 包 | 加 release 约束后 | 移除的迟发布包 | 占比 |
|---|---:|---:|---:|---:|
| inner0 | 1706 | 1669 | 37 | 2.17% |
| inner1 | 2191 | 2139 | 52 | 2.37% |
| outer | 2914 | 2871 | 43 | 1.48% |

修复后 `release_within_cutoff` 与 `event_release_within_cutoff` 在三个 stage 全为真。
scaler、trait、模板与统计量都经 `train_packet` 读取，因此同时受该边界约束。
count 与 marks 当前共用一个闭合块 release；两个年龄字段分别导出，但**不臆造更早的 count 发布时间**。

## 3. 四张机器可读合同表

`/data/hfosp_group_event_state_epilepsy_state_v040/contracts/`

- `support.json` — 逐 query 支持、四类支持分类、H_short 决定与依据、memory_support 分层、训练截止依赖、局限清单
- `training_objectives.json` — 七个目标的分布/坐标/支持/权重/训练-选模-仅评价身份 + 代码入口；
  形态估计量的四层汇总顺序；保留的 legacy 分数；条件字段布局
- `state_export.json` — Q=(m24, P24×24, C, 合法元数据)；明确不导出 `c64` 与事件内记忆；
  H1/H2a-A/H2a-B/S-A/S-B/共同 C 各自的固定读取方式；三类表示状态的判定依据
- `consumer_routes.json` — 条件集合归一语义与 community 规模、供体匹配规则、真实前缀禁用输入、
  S-A 的 13 个功能坐标与整体量定义、S-B 的 community 内 Spearman 主口径

## 4. G0 接口检查（`contracts/g0_checks.json`，六项全过）

| 检查 | 结果 |
|---|---|
| 形态估计量独立重算 | 逐事件最大误差 **1.6e-07**，窗口聚合最大误差 **1.2e-07**（容差 2e-05）；92 事件 / 11 窗口；3 个事件无任何有效形态族 → 计为无支持，不填 0 |
| 训练权重公式 | 独立重算差 **2.8e-08**；batch 含 0/1/2/10/22 事件的目标，逐窗口有效事件数 {1,2,9,21,22} 权重相同 |
| 集合归一枚举 | 16 个 (K_c) 组合，最大质量误差 **1.2e-07**；K=0 与 K=全集正确判为确定性、不计入单位 |
| 训练截止 | 三个 stage 均无越界（见 §2.3） |
| 重放与随机流 | 连续 vs 分批状态最大差 **1.2e-10**，协方差/条件/时距子集/分块打分差**恰为 0** |
| 极端 NLL | 全部有限；窗口级最大值 count 4.38 / spatial 6.60 / morphology 1.38 / load 4.13 nats，无病态尾部 |

极端 NLL 的检查是在初始化处做的；训练后需按同一入口复检。

## 5. 两个合成任务的生成器验证（拟合前）

`v040/synthetic.py`，每任务 3 个实现（901/902/903）。目标由**可读条件集合直接生成**再加独立噪声。

在真实生成数据上回归 `band_ratio[:,0] ~ [1, z_rate, sin, cos, z_old]`：

| 任务 | 实现 | z_old 系数 | 残差 sd | partial corr(target, z_old \| readable) |
|---|---|---:|---:|---:|
| conditional_zero | 901/902/903 | +0.015 / −0.007 / −0.003 | 1.004 / 1.002 / 1.002 | +0.015 / −0.007 / −0.003 |
| organization_positive | 901/902/903 | +1.015 / +0.993 / +0.997 | 1.004 / 1.002 / 1.002 | +0.712 / +0.702 / +0.705 |

即注册的标准化旧内容系数（1 与 0）与独立目标噪声 sd（1）在数据里可复核。
`z_old` 与粗通道相关：count −0.006、load −0.009、coarse rate −0.036（驱动器已对可读基
底做投影正交化，否则有限 bin 数会造成 ~0.17 的偶然相关）。
旧汇总窗口取 `[t−360min, t−240min)`，在 h=30 的 query 上**100% 已发布**，且不与 2 小时短窗重叠。

**不得声称**：3 个实现校准了错误率或排除了小效应；也不认证人体双 INNER 程序。

## 6. D-local 逐字段结论（`contracts/dlocal.json`）

130 张测量卡；每块实测处理耗时中位 **45.4 s**（1.9–83.0 s），
而观测到的发布滞后中位 **3600 s**（139–3600 s）。

| 字段 | 右上下文 | 依赖 | 结论 |
|---|---|---|---|
| participation / contact_ok | 事件上下文窗 | 事件级 | 有界可验证 |
| 每分钟事件数 | 分钟包边界 | 事件级 | 有界可验证 |
| core_seconds_raw / 事件结束 | 事件上下文窗 | 事件级 | 有界可验证 |
| relative_delay_s / tied_group_id（同步组） | 整个事件（rank 归一） | 事件级 | 有界可验证 |
| delay_iqr_s / delay_span_s | 整个事件 | 事件级（需 ≥4 个参与延迟） | 有界可验证 |
| band_log_energy / centroid / peak | 闭合 200 秒处理段 | 200 秒段 | **需局部重算**，步骤已列 |
| cross_band_lag_s | 闭合 200 秒处理段 | 200 秒段 | **需局部重算** |
| band_ratio / signed_xlag（目标） | 继承上两项 | 200 秒段 | **需局部重算** |
| 波形描述量 | 事件上下文窗 | 事件级 | 有界可验证 |
| block background summary | 整个 1 小时块 | 块级 | 块绑定，但**不进入 producer 输入路径**，故不约束任一拟合臂的 D-local |

**同原始时间重算比较**（4 个 FIT 内块，从原始 cache 用 `block_event_tables` 重算）：
band_ratio / signed_xlag / delay_iqr / participation / contact_tokens 的最大绝对差
**全部为 0.0**，缺失模式完全一致。事件集合差异保留而非取交集：
block 0/3 无差异；block 1 有 409 个 cache 事件、block 2 有 203 个在打包流中不存在，
**全部（409/409、203/203）落在已登记的发作/发作后排除区间内**；
反向差异（打包流中有而 cache 无）为 16/4/11/13 个，即跨 200 秒段被保留但细 marks 缺失的事件。

**结论**：producer 实际读取的每个字段右上下文都有界（事件级或 200 秒段级）。
观测到的 0–60 分钟发布滞后是**一小时打包约定**，不是内在测量时延。
本包**没有**产出整段记录的低时延重发布，也**没有**在局部发布流上重拟合任何臂；
这不阻断 D-delayed 主线。

## 7. G1/G2 运行安排与预算

计划文件 `plan.json` 共 47 个任务：

| 任务 | 计划数 |
|---|---:|
| 人体 paired INNER（5 臂 × 各 2 个 INNER 轨迹） | 5（=10 次拟合） |
| 人体最终重拟合（5 臂 × 3 seed） | 15 |
| **主体拟合合计** | **25** |
| 合成（2 任务 × 3 实现 × 3 臂） | 18 |
| S_marks 冻结消费者包（3 seed） | 3 |

优先级把 `S_marks` 的 paired INNER → 首 seed 最终重拟合 → 消费者链排在最前，
以便 G2 不等其余四臂。每 GPU 一个训练进程：GPU0 跑人体链，GPU1 跑合成。

实测吞吐（本机同时有其他 topic4 作业，load ≈ 22，CPU 争用会拉长墙钟）：
人体 paired（两条轨迹同步推进）约 3.35 s/update；合成单臂约 0.65 s/update + 每 50 步约 11 s 评估。

### 7.1 冒烟验证（quick 模式，全链路走通）

`paired INNER → recipe → 最终重拟合 → 冻结消费者` 全链路在 quick 配置下跑通：
H2a-A COMPLETE、正确时刻 COMPLETE、H2a-B COMPLETE、S-A DESCRIPTIVE、S-B DESCRIPTIVE、
S-C NOT_RUN（缺前瞻风险分母，只限制 S-C）；`producer_unchanged=True`。

**quick 模式下四个条件集合 head 全部选中第 0 步**（`selected_step=0`，权重初始化为零 ⇒
四个 head 输出完全相同、三个对比恰为 0.0，S-B 四个条件的 Spearman 也完全相同）。
这是 20 步只做一次验证检查的必然结果，不是缺陷；正式运行用 400 步 / 20 次检查、
且事件密度高 6 倍。**但必须在正式结果里逐个 head 报告 `step_zero_validation`、
`selected_step`、`parameter_change_l2`、`learned`** —— 若正式运行仍选第 0 步，
那是「读出退化为截距」的真实结果（spec §6.2），要照实写，不得调参调到阳性为止。

### 7.2 启动记录

2026-09-07 21:0x 冻结源码（digest `b580de0c…`）并启动：
GPU0 跑人体链（paired_inner / final / consumers），GPU1 跑合成（6 个世界已生成，18 次拟合排队），
窗口 30 小时，每 GPU 一个训练进程，全部可恢复。

## 8. 已知局限（进入报告，不得省略）

1. INNER0 的两小时支持弱（1/12 合格，截断仅 16.3% 生效），较早历史归因在该 origin 上支持有限。
2. count 与 marks 共用块 release；异步发布无法在当前 packets 上分辨。
3. 临床 reset 年龄是事后标注，只支持离线条件分析；已在 `state_export.json` 单列，
   且以同一接口给全部臂，不构成状态臂的私有优势。
4. E1125 的 community 规模为 HR=6 / TBA=1 / TBB=2，**只有 HR 满足 S-B 的每组 ≥3 触点**，
   community 内 Spearman 实际由单一 community 承载。
5. 自由续接臂需要初始组织状态参数，而冻结的 `topic5_wiring_economy_rnn.rollout` 不暴露该参数；
   `v040/consumers.py:rollout_with_h0` 逐字复制其全部决策规则、仅新增该参数。此为签名与
   合同散文不一致处，已显式登记，不作静默修补。
6. H2a-B 的**自有登记预算**（工作包里的 400 步上限管的是条件身份 head，不是这一项）：
   h0 适配器拟合 ≤1200 个事件 / ≤200 步；打分 ≤3000 个事件；
   **自由续接生成另按 ≤300 个事件抽稀**——生成是逐事件的 Python 循环、每步都要与设备同步，
   3000 个事件时实测超过 10 分钟，而它只用于两个描述量（剩余触点召回率、长度误差）。
   所有抽稀都是在读任何分数之前按固定时间步长确定的。
   冻结解码器为 `we_decoder/formal_units/epilepsiae_1125__own_a/L3_LOCAL_PLUS_LEARNED_LR/seed0`
   （`best_checkpoint_eligible=True`、`target_values_read=False`、按记录时间切分），
   触点宇宙 7 个（TBB1/TBB2/HR9/HR10/HR12/HR13/HR14），与本 producer 的 9 触点集按名字对齐。
7. 极端 NLL 检查目前只在初始化处完成，训练后需按同一入口复检。

---

**本文件只记录 G0 与运行安排。** 五个差值、H2a-A/B、S-A、S-B 的结果，
以及丰富信息 / 持续历史 / 冻结状态连接的分项判断，待 G1–G3 完成后另行归档。
工作包完成不等于科学闭环完成。
