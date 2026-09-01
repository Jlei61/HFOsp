# Group-Event State v0.2：三条假设共同科学合同

状态：**冻结设计输入，development-only**  
代码基线：`7c847246`；v0.2 独立工作树 `/tmp/hfosp_group_event_state_v02`  
数据继续只读使用 `/data/hfosp_group_event_state_v0_1/dataset`；新结果只写
`results/epi_prssm/group_event_state/v0_2/`。

## 0. 共同目标

我们不是要证明“RNN 能预测下一次事件”，而是要从患者按真实时间排列的完整间期群体事件中判断：

1. **H1/H2a**：是否存在跨很多事件、跨真实时间持续的预测状态；它是否改变一片未来群体事件的网络表达？
2. **H2b**：这个完全从间期任务中学到并冻结的状态，能否跨任务预测距离下一次发作、发作风险和发作入口类型？
3. **H3**：群体 IED 的出现和内容是否对之后的慢状态轨迹留下增量、长期影响？

三条线可并行探索，任何一条阴性都不 gate 另外两条；但 H2b/H3 使用哪个 checkpoint、哪条状态轨迹，必须写入不可变 manifest。

## 1. 一个 timestep 是什么

一个 timestep 是一次完整间期群体事件，不是单触点尖峰、rank step 或固定一分钟窗。

每个事件输入包括全部患者触点上的：participation、基于 `lagPatRaw` 频谱质心的参与触点相对延迟、10 ms tied groups、原生事件窗 waveform 视图、多频带包络/峰时/能量/cross-band lag、静态几何与可用性 mask。连续背景 SEEG 只作辅助 observation。

测试按事件顺序因果运行：先预测事件，再用真实观察到的事件更新状态。所有 teacher-forced 一步结果必须明确标作 filtering；从同一 anchor 不读取中间真实事件的多未来预测另行标作 forecasting。

## 2. fast 与 slow 当前到底有什么区别

| | `z_fast` | `z_slow` |
|---|---|---|
| 维度 | 64 | 32 |
| 衰减范围 | 1 s–1 h | 1 min–48 h |
| 初始化范围 | 10 s–10 min | 30 min–24 h |
| 事件更新 | `GRUCell([event,z_slow],z_fast)`，每事件可大幅改写 | `z_slow + 0.05 × gate × delta`，每事件只作小残差更新 |
| 共同点 | 都在真实 `dt` 间向学习到的 bias 指数衰减；都由每次事件驱动；都进入预测头 | 同左 |

这只是**架构先验**，不是已经识别出的生理时间尺度：

- fast 与 slow 在 1 min–1 h 有重叠；fast 也可能携带慢信息，slow 也可能编码短程统计；
- 当前 head 拼接两者，reset/wrong-time 同时扰动两者，旧结果不能归因于 slow；
- `tau` 能达到小时只证明模型有表达能力，训练后的 `tau` 或坐标未必被数据识别；
- 潜坐标可旋转，单维曲线不作承重结果。承重对象优先使用状态读出的未来 repertoire/participation/risk 等功能量。

因此 v0.2 必须分别报告 fast-only、slow-only、fast+slow 的冻结读出增量和多未来预测；“slow”一词在通过这些检验前只指代码模块名。

## 3. v0.1 已发现的实现与解释问题

1. **warm-up 终态被丢弃**：旧代码先 warm 再 test，但 `run_sequence` 每次重新初始化；`full_session` 实际只是 test 段内 carry。v0.2 已改成显式传入/返回 `(z_fast,z_slow,since_reset)`，并从当前 recorded session 起点因果 replay。
2. **reset 不是生理状态证据**：post-hoc reset 只说明同一训练模型依赖递归轨迹，且带测试分布改变。必须补同 encoder、同输入、独立训练的 memoryless 模型。
3. **K=100 不能叫饱和**：当前只有 `{1,20,100,full}`，且一步端点、8人低功效。事件数还混杂事件率。以后同时用事件数和真实时间。
4. **wrong-time 不是 matched**：旧实现是 test 状态行的任意 permutation。v0.2 每个 anchor 使用同患者、同 session 的 5–10 个 donor，并匹配昼夜、近期 rate、距上次事件、近期 size/STOP、coverage 与 session position。
5. **一步 teacher forcing 不是慢状态**：每次预测后都读取真实事件。慢状态必须从同一 anchor 预测多个未来事件/时间窗。
6. **简单近期统计是科学候选**：若最近 1/5/20 事件的低维统计追平 RNN，应报告“简单统计可能足够”，而不是把它写成失败。
7. **延迟语义已澄清**：a3 的连续延迟来自 participant-masked `lagPatRaw` 频谱质心，不是原始波形；设置本身保留。原生 waveform/multiband 只在 a4 进入。
8. **H2b 当前新线尚未运行**：旧脚本只取发作前最后一次事件做 case/control AUC，不是逐事件 distance/hazard；必须重写主端点。
9. **H3 不能读取架构定义当发现**：当前 slow 每次事件后必然更新。观察到 `z_slow` 改变是同义反复；必须比较能否解释未见未来，以及 event-preserving controls。
10. **训练并行也可能截断长历史**：`n_streams=8` 把 TRAIN 人为切成八段并分别初始化。它是吞吐优化，但对数千事件记忆未必科学中性；H1 agent 必须做 `n_streams=1` 或 session-preserving sensitivity。

### 3.1 本项目此前已经踩过、三条线都必须知道的坑

- `softplus(log tau)` 曾把想要的 300 s 初始化变成约 5.7 s，并使模型在预算内无法表示小时状态；继续使用 `exp(clamp(log_tau))`，但“范围覆盖小时”不等于“数据识别了小时尺度”。
- “无状态”臂曾让 STOP/participation head 偷看到 state；每个 no-state/memoryless 臂都要用输入扰动回归测试证明所有 head 真看不到 carry。
- 图零假设曾跨五个代码包并静默平均重复 run；任何成对比较必须锁相同 source/checkpoint/config hash，重复 payload 报错而不是平均。
- Yuquan seizure ID 与规范表字符串不通用；按 recording code 显式 crosswalk，逐发作核对 onset，禁止直接字符串 inner join 静默丢患者。
- 长窗资格曾按粗 `event_session` 计数，而真正窗口按 recording coverage segment 建立；所有 H2b/H3 支持必须用与最终 estimator 完全相同的 coverage/gap 逻辑。
- H3 固定 event jump 曾饱和成免费截距；real arm 必须有 fitted-intercept/count/rate matched control。delayed control 与真实 exposure 窗必须严格不重叠。
- ridge 正则曾因未按 Gram/样本尺度归一而在时间外外推发散；线性 probe 必须标准化、报告相对条件数/目标位移，并把远坏于 intercept baseline 的拟合标为不可估计。
- 三个 seed 曾生成 byte-identical payload；交付时检查初始化、训练顺序和输出 hash 真正不同。seed 是重复拟合，不是患者分母。
- synthetic recovery、测试全绿和工程 `PASS_COMPLETE` 只证明仪器按合同运行，不证明人体 H1/H2/H3 成立。

## 4. 共同数据与切分

- ictal-overlap 事件排除；preictal 间期事件保留。
- train/validation/test 仍按事件时序 70/10/20；同时报告各 split 的真实小时数、session 数、事件率和发作数。
- 不跨未记录 gap 传播状态。split 若切在 session 中间，validation/test 必须从该 session 起点 replay 并传入终态。
- 事件行可以进入似然，但患者和独立发作/不重叠未来窗才是统计分母。
- formal/sealed 分区继续关闭；所有数字均为 development。

## 5. 共同比较纪律

- patient-first：先 seed 内/患者内，再跨患者；事件数不冒充患者数。
- 一步 timing、mark、未来块、seizure 与 perturbation 分开报告，不合成一个总分。
- 每个方向性数字同时给出患者数、seed spread、有效独立时间块/发作数。
- 参数更新、selected epoch、梯度、非有限值、OOM 是训练资格信息，不是科学结论。
- synthetic 只作实现和灵敏度校准，不作人体证据、不作继续探索的 gate。

## 6. 三 agent 的写权限与交付接口

### Agent A：H1/H2a + 共同状态生产者

拥有 `src/topic5_group_event_state/` 的 core state/evaluation 修改权；写
`results/.../v0_2/h1_h2a/` 与 `shared/checkpoint_manifest.json`、`shared/state_trajectory_manifest.json`。

### Agent B：H2b

core 模型只读；新增代码放 `scripts/topic5_group_event_state/v02_h2b_*` 或独立 H2b 模块；写
`results/.../v0_2/h2b/`。可以先用 v0.1 checkpoint 调通并明确标 `plumbing_only`，人体承重数字只认 shared manifest 中的 v0.2 checkpoint。

### Agent C：H3

core 模型只读；新增代码放 `scripts/topic5_group_event_state/v02_h3_*` 或独立 H3 模块；写
`results/.../v0_2/h3/`。可并行实现轨迹分解和 perturbation，但承重人体结果绑定 shared manifest。

三个 agent 都不得覆盖别人的 manifest、报告或结果根。共同 schema 修改先写 additive 字段，禁止在下游运行中静默改旧字段语义。

## 7. 运行与报告

- 长任务由单一 queue owner 通过 nohup/setsid 或 tmux 持久运行；原子输出、幂等跳过、可恢复。
- 使用实测显存决定 GPU slot；OOM 只降低当前 job 并发/chunk，不改变科学数据，失败记 `resource_failed`。
- CPU worker 固定 `OMP/MKL/OPENBLAS/NUMEXPR=1`；按物理盘限并发，避免寻道竞争。
- 每条线最后各交付白话版和技术版；白话版不能省略 estimand、分母、teacher forcing/open-loop、fast/slow 与关联/因果边界。
