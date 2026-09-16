# 给夜间 Agent 的执行交接：v0.3.10

## 可直接复制的任务指令

> 按本交接和链接的科学spec，自主工作8–10小时。目标是验证训练是否充分、公平比较F/L/N与有限容量、把同一冻结状态接到未训练contact/形态任务，并按稀缺发作合同做分层分析。不要承诺阳性。尽早启动已冻结的B0合成LR队列占用两卡，同时实现并验证v0310训练修复；不要让有待运行的合格任务因为无人接续而停下。所有新实验写新根，保留旧失败与旧报告，不开development/sealed，不用发作结果反选上游，不修改正在运行的源码快照。根据现场吞吐完成完整配对组，9h停新增长任务，9.5h保存训练，10h以内交付实际完成/未完清单、逐窗/逐簇结果和中文结论。不要把耐心停止、测试通过、GPU满载或多个seed当作科学结论。

科学依据唯一入口：[冻结spec](group_event_state_v0_3_10_training_and_rare_seizure_spec_2026-09-05.md)。先读[本次复核](group_event_state_v0_3_9_training_review_2026-09-05.md)，不要从最初的v038阳性摘要继续。

## 交接时实际状态

- 九个2400预算的合成复核已校验，N-over-L中位数−0.008185，N-over-F+0.073232，L-over-F+0.086236。全部INNER耐心停止；训练充分性仍未证实。
- 报告配对、停止原因、source/score核验代码已修；相应回归与原汇总测试18/18通过。
- 两张3090上的11个容量工程预检已完成、无OOM；只读真实FIT，未评分SELECTION或发作。它们不是人体科学训练结果。
- B0的18个逐卡命令与9个依赖源码快照已生成；**PREPARED_NOT_STARTED**。本交接不表示整夜训练已经启动。
- **v0310人体训练器、全夜监督器和新发作评价器尚未实现。** 下面的现成入口只运行B0；不能把它的COMPLETE当成本晚全部完成。
- 当前worktree有大量既有未提交文件；根checkout另有冲突。不要清理、reset、覆盖或提交整棵工作树。没有推送/发布任务。

## 路径与已有入口

工作区：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-ges-v033-training-lab`。
Python：`/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`。
输出根：`/data/hfosp_group_event_state_v0_3_10_overnight_design`。
计划：该根`night_plan.json`；准备队列：`bootstrap_manifest.json`；容量预检：`preflight/gpu{0,1}_capacity_profile.json`。

旧正式数据根：`/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired`。
仅重用`human_data_v2/epilepsiae_{1096,1125,253}.pt`及其匹配测量依赖；hash已写night_plan。
旧最终人体统计在`final_reports/summary_main.json`；最新报告在`final_reports_training_review/`；旧contact接口对话的产物在`/data/hfosp_rnn_v038_contact_bridge_20260905/`。

|用途|现有实现，可读后复用|
|---|---|
|严格九卡审计|`scripts/audit_group_event_state_v039_budget_extension.py`、`src/topic5_group_event_state/v039/instrument_audit.py`|
|F/L/N与损失|`src/topic5_group_event_state/v039/transition.py`|
|旧人体训练，需新版本修复|`scripts/train_group_event_state_v039_human.py`|
|历史数据与时序|`src/topic5_group_event_state/v039/human_data.py`、`scripts/build_group_event_state_v039_human_data.py`|
|冻结导出与真实历史梯度|`scripts/export_group_event_state_v039_frozen_state.py`、`scripts/audit_group_event_state_v039_human_gradient.py`|
|contact与细表达|`scripts/train_group_event_state_v039_contact_transfer.py`、`scripts/probe_group_event_state_v039_expression.py`|
|旧发作支持盘点，不能直接当新方案|`scripts/audit_group_event_state_v039_seizure_support.py`|
|已有有限监督器结构|`scripts/supervise_group_event_state_v039_transfer_closure.py`|

现有human CLI只接受H0.5/8，未接readout hidden；即使直接传`--width 64`，也不等于完成本版64→128的配方。新参数必须真正接入每层，库存表由实例化模型导出。

## 开始后的前30分钟

1. 读工作区AGENTS和科学spec；核对进程、两卡、磁盘、night_plan/source hash。把实际t0、PID、deadline、主机/GPU/torch版本写`night_runtime.json`。别杀其他任务。
2. 先启动下面的B0。该入口每卡1进程，每卡独立目录，最多2小时，1.5h后停止派新卡；某卡出现失败即隔离，不自动重试。运行中可继续CPU代码修复。不要同时在相同GPU另开主训练；B0释放卡后才交给全夜调度。
3. 冻结新的v0310代码位置及修改清单，先实现日志/恢复/微批/hidden/调度，保持数据与科学边界。用最小FIT测试验证，不读取留出分数做调试。

现成命令（当前cwd须为上述工作区）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
scripts/prepare_group_event_state_v0310_bootstrap.py \
--root /data/hfosp_group_event_state_v0_3_10_overnight_design --execute
```

用PTY/session或可靠后台方式保留该进程，记录PID。它不是全夜总监督器；B0结束后必须接U1等队列。不要再次执行以覆盖已有`bootstrap_status.json`；失败或截止后先审阅attempt，必要重跑走新目录。CPU的`OMP/MKL/OPENBLAS/NUMEXPR=1`，若pandas触发GLIBCXX错误使用CUDA环境的lib路径，不替换HOME环境变量。

B0不支持exact resume：它沿用已核对的旧源码，目的是新训练器开发期间执行有限LR敏感性。必须按family/LR/优化seed配对，注明共用data seed39001；不同LR预算/耐心规则的结果不能解释为同一优化轨迹continuation。

## 1–3小时：先让训练可信

创建`src/topic5_group_event_state/v0310/`及版本化入口（这是待实现路径，当前不存在）。固定配方见spec第4节，不临时挑优化器到有利结果。每张卡必须能回答：用了哪些样本、有效更新多少、每层多大、权重动了多少、FIT/INNER怎样变化、降LR之后是否还有改善、最佳与最终是否不同、是否预算受限。

必要检查一次通过后就进入有限队列，不循环追求更多形式测试：

- 中断恢复与连续训练的RNG、batch序列、权重/分数对齐；不把缺optimizer的旧checkpoint当精确恢复。
- 同128样本的整批与微批累积梯度/更新对齐；mask与多头归一化一致。
- 比较的一臂达到预算即保留受限标签；源码、target、split、通道顺序不一致拒绝合并。
- L/N同初始A/B，残差零输出导致第一步observer无梯度是预期；后续应有真实梯度/更新诊断。
- F保留完整七核；hidden32/64/128确实对应模型实例；参数库存包括背景、残差、常数，而不只observer。
- 使用完整训练器最大batch/H/宽度试跑，评估分批；nvidia-smi总显存预算18GiB、保留6GiB。工程预检的46MiB不是总内存上限。
- 长历史真实事件重放/有限差分、采样发布时刻截断不变性；不能以detach节省显存。

将测试、recipe与源码hash写`training_preflight.json`后才启动U1。若P0实现耗时超过2h，优先完成核心训练诊断与较小完整配对组，登记计划差异；不要跳过检查或延长到无限夜跑。

## 3–8小时：按冻结顺序推进，不追阳性

U1容量27卡 → U2另外两seed18卡 → T1的H2九卡 → U3背景条件27卡 → T1的H0.5九卡。具体维度、患者、选择规则以night_plan/spec为准。

同一paired group内F/L/N都要完成或同时记不完整，不能只交付最快/最好的一臂。以实际运行速度更新预计耗时；完整配对组预计无法在截止前结束时，不新开长作业。预算/任务状态不能从科学分数反推。

一旦某患者上游配方冻结，可构建contact缓存并准备冻结探针，CPU与GPU交叠。下游结果不得反馈上游选择。至少保住M1的9个event-only患者×seed状态及对应对照，优先直接训练contact identity，STOP独立。

先在CPU盘点S1的发作簇、查询覆盖和匹配集合，不计算状态－发作结果。全部上游实际完成/关闭清单、recipe、checkpoint hash冻结后，才统一计算新的S-A/B/C；不等待H1阳性、不筛掉负主状态。允许少发作病例描述，不把重叠锚点算复现。

若数据支持和时间允许，P2按独立合成data seeds → H16/H24 → 单视图迁移的顺序执行。无支持的H16/H24写NOT_ESTIMABLE，GPU接下一个登记任务。任何未列出的扩展写为后续建议，本晚不自动增殖新grid。

## 最后1–2小时：证据核验与交付

9h停止新增长作业，9.5h保存latest，10h内交付。若实际进度只能做到U1/U2，也要明确报告完成了什么、哪些更高层问题仍未获得有效检验。已完成有限工作包可以关闭；原始科学闭环不能随执行完成自动关闭。

交付文件至少包括：

- `night_closeout.json`：计划、实际、未运行原因、失败attempt、时间/GPU使用，不写零失败掩盖重跑。
- `training_audit.csv`：每卡训练充分性向量、每层参数、LR、更新/梯度/饱和/预算、source/data/target/split hash。
- `paired_physical_windows.csv`、`contact_endpoint_scores.csv`：逐物理窗、同prefix分数、分端点分母与冻结来源。
- `seizure_episode_ledger.csv`：raw onset→簇→历史支持→匹配/留出/前推资格逐条映射；不是只有NOT_ESTIMABLE一句话。
- `same_checkpoint_evidence.csv`：同患者、同seed、同checkpoint的H1/长历史/contact/细形态/发作结果，禁止跨人拼接。
- `report_zh.md`：四个原问题分别支持到哪里、训练/功效边界、旧结论哪些撤回；不打任意科学完成度总分。
- 图及对应producer、metadata、hash；图实际产生后写`figures/README.md`并检查最终图。没有图也不要造空README假装完成。

CPU监督器检查每30s，机器资源快照每5min；只在有实质进展/失败/完成时通知用户。没有人为要求固定频率刷屏。

## 不可跨越的解释边界

- 耐心停止不等于充分收敛；初始checkpoint不等于程序坏，也不证明事件已学会。
- L或F有效都是有效科学结果，不为N必须阳性设计筛选。
- 计数与招募共同训练的好分数，不等于未训练病理表达迁移；背景贡献不等于事件贡献。
- 一个稀缺发作病例可以有有价值的轨迹/空间对应，但不是临床预测器；留一簇法若用未来训练只能称回顾性迁移。
- 发作前空间预测不能偷偷给ictal prefix、真实K或未来触点选择；只有粗ROI头不能写逐触点预测。
- 每次测量块写入observer不等于每个IED改变生理状态。H3需要独立观测和识别设计。
