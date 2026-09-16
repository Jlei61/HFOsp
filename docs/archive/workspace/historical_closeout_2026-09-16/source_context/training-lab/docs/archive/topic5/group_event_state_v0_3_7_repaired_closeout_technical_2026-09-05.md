# Group-Event State v0.3.7 修复后技术收口

**日期：** 2026-09-05  
**状态：** `ENGINEERING_ACCEPTED_SCIENCE_DEVELOPMENT_ONLY`

## 1. 冻结科学边界

- `S_obs(t)` 是过去事件、非事件背景和已知 context 的因果预测摘要；observer update 不是生理 jump。
- `Z_phys(t)` 只存在于 H3 独立生成模型；M0/M1/M2 分别表示 common drive、count feedback 和 mark feedback。
- `S_N` 与 `S_G` 分别承担 future burden 和 conditional grammar；grammar 必须控制未来事件总数与有效 exposure。
- H1 的共享 producer 跨 0.5/2/6/8 h 使用同一状态；horizon-specific head 可以独立。
- H2a 冻结严格时间切分的 contact-sequence decoder；H2b 在所有间期特征冻结后才读取 seizure outcome。
- 人体主时间常数 bank 上限为 16 h；32 h 只保留为 instrument design check。

## 2. 修复验收

第一轮人体结果作废的根因，经保存权重和训练轨迹复核为真实问题：`B_mark` 在 4/5 患者权重严格为零，H2a 在 3/4 患者调制输出严格为零，且原 optimizer search 未搜索控制臂参数。

最终 `repaired_v5` 实现：

1. 48 格 control search 先运行并冻结 `baseline_warm_init`；
2. 264 格 model search 仅在冻结控制地基上比较；
3. 新增臂显式保留 parent-parity checkpoint；
4. 训练充分性只由 first-step gradient、训练期间参数移动和 budget exhaustion 判断；
5. 人体增益、selected-at-origin 和患者一致性与训练充分性分开报告；
6. H2a 零初始化只用于保证冻结 decoder parity，warm-up=5、patience=40，并记录 peak modulation；
7. H3 岭网格扩到 `1e4`，显式包含 zero edge 并记录上界饱和；
8. 每张卡保存 producer source SHA256；综合汇总拒绝混合代码版本。

修复后全部承重 H1/H2a 路径 `path_explored_fraction=1.0`，budget exhaustion 为 0 或极低；H2a leaked oracle 20/20 非父选择。故真实状态臂回退 parent checkpoint 时可解释为 INNER 不支持增量，而非“没有训练”。

## 3. 执行矩阵与审计

| 阶段 | 完成/总数 |
|---|---:|
| control + model optimizer search | 312/312 |
| H1 optimized | 75/75 |
| H2a frozen decoder primary | 60/60 |
| H2a joint sensitivity | 20/20 |
| H2b frozen transfer | 25/25 |
| H3 one-step human | 20/20 |
| H3 persistent human | 20/20 |
| 正式结果卡合计 | 220 |

机器审计：

- `all_cards_match_current_code=true`；缺 provenance 卡数 0；
- `development_targets_read_any=false`；
- `sealed_partition_opened_any=false`；
- `h3_observer_checkpoint_used_as_jump_any=false`；
- H2b 25/25 在 outcome 前完成 feature freeze；
- 监督器选择的 9 个测试文件为 80 passed；新增控制修复文件单独复跑为 30 passed。

## 4. H1 结果

### 4.1 event-only 与 grid state

患者级五 seed 中位数：

| 患者 | `S_event-B_mark` | constant-unexplained | correct-time | 判读 |
|---|---:|---:|---:|---|
| E1077 | 0.000 | 0.000 | 0.000 | 无稳定增量 |
| E1125 | +0.358 | -0.083 | +0.007 | 增益由阶段/常数水平解释 |
| E253 | 0.000 | 0.000 | 0.000 | 无增量 |
| E916 | 0.000 | 0.000 | 0.000 | 无增量 |
| E958 | +0.100 | -0.005 | -0.001 | 非时刻特异，常数已解释 |

`S_grid` 的对应数值与 `S_event` 几乎一致，未提供独立的更慢动态证据。

### 4.2 dual state

| 患者 | persistent background-current | beyond constant | correct-time | event after background |
|---|---:|---:|---:|---:|
| E1077 | +0.415 | +0.053 | +0.017 | 0.000 |
| E1125 | +0.357 | -0.015 | +0.004 | +0.027；但 event beyond constant=-0.019 |
| E253 | +0.011 | -0.004 | -0.002 | 0.000 |
| E916 | -0.004 | -0.003 | +0.002 | 0.000 |
| E958 | +0.057 | -0.089 | +0.002 | -0.083 |

仅 E1077 同时满足 persistent-over-current、beyond-constant 和 correct-time 三个 development 条件。其 event-history-after-background 为零，故候选来源是背景流而非离散 IED 历史。

### 4.3 独立窗口

五分钟重叠 anchor 只用于患者内 loss 估计，不是统计分母。SELECTION 中每位患者的互不重叠窗约为：0.5 h 42–48、2 h 10–12、6 h 3–4、8 h 2–3。6/8 h 只作探索，患者是队列统计单位。

## 5. H2a 结果

冻结 decoder 的 oracle grammar sensitivity 为：E1077 `+0.441`、E1125 `+0.552`、E253 `+0.506`、E958 `+0.271`。真实状态通路均发生参数移动，因此测量通路具有灵敏度。

但三类状态均未形成一致阶梯：

- event state：E253 相对 `B_mark` `+0.0068`，但相对 static grammar `-0.0029`、constant-unexplained `-0.0048`；其 correct-time `+0.0291` 不能单独承担结论。
- grid state：E1125 相对 `B_mark` `+0.0025`，其余承重增量为 0；E253 多数为负。
- dual state：E253 相对 static grammar `+0.0070` 且 correct-time `+0.0246`，但相对 `B_mark` 仅 `+0.0003`，且 E1077 H1 候选在 H2a 为 0。
- E958 dual rich-mark gain `+0.310` 缺少同端点 constant/shift controls，且主 grammar 阶梯为零，只记为探索线索。
- joint sensitivity 20/20 完成，但没有同一模型同时建立 H1 与 H2a。

判决：`H2A_NOT_ESTABLISHED_WITH_ASSAY_SENSITIVITY_CONFIRMED`。

## 6. H2b 结果

| 患者 | FIT/INNER/SELECTION seizures | 风险任务 | early field/path |
|---|---:|---|---|
| E1077 | 1/3/2 | not estimable | not estimable |
| E1125 | 6/0/0 | not estimable | not estimable |
| E253 | 1/1/0 | not estimable | not estimable |
| E916 | 32/2/1 | single-seizure descriptive | not estimable |
| E958 | 3/1/11 | repeated held-out estimable | not estimable |

E958 的 event/grid 相对 mark history 分别为 `-0.00066/-0.00064`，dual 相对 background 为 `+0.00002`，correct-time 均未改善。E916 只有 1 次 held-out 发作，不作推断。

判决：`H2B_NOT_ESTABLISHED_MOSTLY_DENOMINATOR_LIMITED`。早期 ictal field/path 是未检验，不是阴性。

## 7. H3 结果

### 7.1 仪器

一步 instrument 9/9 完成并通过 common-drive count/mark false-positive controls、count feedback recovery 和 mark feedback recovery。persistent instrument 同样通过 exact-null、count 和 mark recovery，以及 fitted wrong-time capacity control。两者均不推断 human trainability 或 biological feedback。

### 7.2 一步人体模型

| 患者 | count→future background | correct-delayed | mark→future background |
|---|---:|---:|---:|
| E1077 | -0.094 | -0.089 | 0.000 |
| E1125 | +0.009 | +0.025 | 0.000 |
| E253 | +0.026 | +0.029 | 0.000 |
| E958 | -0.084 | -0.123 | -0.001 |

M0 common-drive 在 E1125 与 E253 的 INNER 上分别改善约 `0.048–0.068` 和 `0.064–0.080`，selected step 为 25 与 75–100；因此两位正向候选不是由完全未训练的 M0 产生。

对默认 M0 回退 origin 的 E1077/E958 另做 20 张敏感性卡：E958 在 `1e-4/3e-4` 下均为 5/5 非 origin，INNER 中位改善 `0.0124/0.0102`，但人体未来背景效应仍为 `-0.0840/-0.0839`；负方向保留。E1077 两档均为 0/5 非 origin，故其结果只说明没有正向候选，M0 本身未获得额外 common-drive 增益。

### 7.3 persistent 人体模型

| 患者 | 可估尺度 | persistent count over one-step background | real over **floored** wrong-time null | (撤回) real over raw fitted placebo | mark over one-step background |
|---|---|---:|---:|---:|---:|
| E1077 | 2 h | 0.000 | **-0.094** | 0.000 | 0.000 |
| E1125 | 2/6 h | +0.080 | **+0.089** | ~~+0.757~~ | -0.224 |
| E253 | 2/6 h | +0.0018 | **+0.0018** | +0.0018 | 0.000 |
| E958 | 2 h | -0.065 | **-0.149** | -0.056 | -0.274 |

**错时安慰剂封底（2026-09-05 审阅修复）。** 同容量的拟合安慰剂在 INNER 上自选岭系数，因此可能过拟合并在 SELECTION 上比「完全没有这条边」还差；此时 `real - placebo` 度量的是安慰剂的过拟合，不是真实边的能力。四位患者中三位出现该现象且 5/5 seed 一致（E1125 安慰剂比无边模型差 `0.663`，同一批留出行、同一支撑 n=288）。因此可报的对比把零假设封底到无边模型：`min(placebo, M0) - real`。封底后 E1125 的数字与嵌套 `M1_persistent_over_M0_background` 一致，为 `+0.089`。原表中的 `+0.757` 已撤回。

E1125 仍是唯一同时具有明显 persistent-over-one-step、封底后仍为正的 wrong-time 对比、M1-over-M0 background (`+0.089`) 且五个 seed 同向的候选；**承重量级是 `+0.089`，不是 `+0.757`**。E253 仅为微小、患者内一致的弱提示。

### 7.4 错时对照本身的质量（审阅补记）

H1、H2b 与 persistent H3 共用同一条供体规则：在每个留出覆盖段内循环滚动一半。两项此前未记录的性质：

| 患者 | 供体时间距离（中位） | 留出段数 | 最长留出段 | 错时臂比常数臂还差的时间尺度 |
|---|---:|---:|---:|---|
| E1077 | 6.42 h | 3 | 12.83 h | 无 |
| E1125 | 12.00 h | 1 | 23.92 h | 0.5 / 2 / 6 / 8 h |
| E253 | 12.00 h | 1 | 24.00 h | 0.5 / 2 / 6 / 8 h |
| E916 | 9.33 h | 2 | 18.58 h | 2 / 6 / 8 h |
| E958 | 3.25 h | 4 | 7.83 h | 2 h |

供体距离由覆盖段的破碎程度决定，不由设计决定，因此 `correct_time_gain` **只可在患者内解读，不可跨患者比大小**。错时臂分数差于常数臂时，它给的是矛盾信息而非无信息，会撑大优势——H1 的 E1077 候选在四档上均无此现象，故不受影响。

本队列**结构上无法构造时钟相位对齐（相差整日）的错时对照**：留出期最长单段为 12.8–24.0 h，段内不存在相隔 24 h 的锚点对。此项记为不可得，不得默认已通过。

昼夜混淆已直接检验并排除：三次谐波时间基底（仅在 FIT 上拟合）只解释背景状态方差的 `0.96%`（事件状态 `-0.58%`），且扣除时钟成分后的残差状态在 0.5 h 与 2 h 上仍给出正的正确时刻优势（`+0.023` / `+0.028`）。

persistent ridge 的扩展惩罚网格中，所有最终非零选择均位于内部，zero edge 也是显式候选；没有上界饱和。一步 M2 mark 路径仍有 14/20 张卡选择惩罚上界，因此 mark-specific 零值不得升级成有力生物学阴性。

判决：`E1125_COUNT_HISTORY_DIRECTIONAL_DEPENDENCE_CANDIDATE`；最高允许措辞为 feedback-like directional dependence，不是干预因果。

## 8. 最终结论与下一步

工程结论：修复通过，正式重跑完成；第一轮 135 张卡及 `repaired_v2/v3/v4` 人体数值继续保持 superseded。

科学结论：

1. H1 只获得 E1077 的 background-driven time-specific predictive-state 候选；
2. 离散 event/mark history 没有建立超出透明历史、常数水平和背景的动态状态；
3. H2a 无支持但仪器灵敏；H2b 无支持且大部分不可估；
4. H3 获得 E1125 count-history-to-background persistent directional dependence 候选，E253 为弱提示；mark-specific feedback 未建立。

下一轮若扩展，优先级应是：扩大 H1 background candidate 和 H3 E1125 型候选的患者覆盖；按发作而非锚点设计 H2b 可估性；保留 S_N/S_G 分解与冻结 contact decoder，不再回到只预测 event rate 的单任务。

## 9. 机器产物

- 汇总：`/data/hfosp_group_event_state_v0_3_7/repaired_v5/final_reports/integrated_summary_v2.json`
- manifest：`/data/hfosp_group_event_state_v0_3_7/repaired_v5/final_reports/manifest_v3.json`
- 完成状态：`/data/hfosp_group_event_state_v0_3_7/repaired_v5/supervisor/completion_status.json`
- H3 M0 训练性敏感性：`/data/hfosp_group_event_state_v0_3_7/repaired_v5/h3_m0_trainability_sensitivity/summary.json`
- 核心图：`/data/hfosp_group_event_state_v0_3_7/repaired_v5/final_reports/figures/`
