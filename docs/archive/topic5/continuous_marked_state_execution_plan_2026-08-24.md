# Epi-PRSSM-Raw v1 执行计划

**日期：** 2026-08-24
**版本：** v1.4（dual-clock long-sequence revision）
**科学合同：** `continuous_marked_state_scientific_spec_2026-08-24.md`
**结果根：** `results/epi_prssm/continuous_marked_state/r0_1/`
**原则：** 探索优先、版本隔离、开发分区运行、普通阴性不停项目、核心映射和封条异常才停。

## 1. 本轮 8–10 小时目标

1. 收口三患者 R0.2 architecture triage，不再扩到 34 人。
2. 建立 B0–B3 的同样本、同 head、exact-time + mark Bridge 数据集和可恢复 runner。
3. 在 6 位预先指定开发患者先跑 B0/B1 和轻量 raw-derived B2/B3；明确这是 Bridge-E0，不冒充最终 raw Transformer 结论。
4. 建立 T1/T2 的最小方程、forward-state carry、correction-off 与 same-capacity placebo 单元测试。
5. GPU 释放后，启动宽 Transformer observation embedding 或轻量 raw branch 的开发作业；CPU 同时跑 Bridge 和 T1/T2 toy/真实小样本。
6. 写白话报告、技术报告、状态表和可恢复 handoff。

## 2. 版本隔离

- R0.2 历史输出仍在 `results/epi_prssm/raw_seeg_state/r0_1`，聚合时必须按 `revision + package_hash + arm + seed` 去重；不得再把它叫 H1–H3 结果。
- 新代码放 `src/topic5_continuous_marked_state/`；新脚本放 `scripts/topic5_continuous_marked_state/`。
- 新结果只写 `results/epi_prssm/continuous_marked_state/r0_1/`。
- dirty worktree 中 paper-ready figures、Topic 3/4 文档及其删除项全部视为用户资产，不读取后覆盖、不恢复、不提交。

## 3. 数据工作包 D0

### D0.1 冻结输入

- event marks：`results/epi_prssm/v0_1/cache/cohort_v0_1.pt`；
- full event stream 是 T1/T2 recurrent timeline 的主输入：development 内每个合法相邻 IED transition 都保留；observation 缺失只改 mask，不删除 event；
- raw cache 通过 `contract.raw_cache_dir()` 解析；当前历史存储根名称与 dataset 名反向，必须信 manifest/subject 映射，不能按 mount 名猜路径；
- split：`results/epi_prssm/raw_seeg_state/r0_1/data/split_manifest.json`，34/34 的 train/validation last-epoch 与 `results/epi_prssm/v0_1/manifests/SPLIT_MANIFEST.json` 精确一致；
- recorded intervals：`results/epi_prssm/v0_1/recorded_intervals/<subject>.npz`。

### D0.2 pilot 6

沿用预先按数据支持而非结果选择的：`epilepsiae_{620,958,139}`、`yuquan_{huanghanwen,zhangjiaqi,hanyuxuan}`。

### D0.3 Bridge 样本

单位是事件 \(e\)：用事件 \(e\) 之前的历史和 background observation 预测下一事件 \((t_{e+1},m_{e+1})\)。跨 session、记录缺口、split boundary 的 pair 删除。所有 observation 必须结束在当前事件以前，IED core 用 mask 而非未来插值。

输出每患者：

- `bridge_features.npz`；
- `feature_manifest.json`（时间范围、event 数、mask 比例、source hash）；
- `split_audit.json`（max train time、min validation time、sealed bound）。

### D0.4 T1 regular observation stream

Bridge 的事件锚点 observation 不能充当 T1 连续观测。T1 另建独立于 IED 是否发生的规则时钟：第一版每 60 s 一个 anchor（执行层近似；冻结目标仍为 30 s），只读取 anchor 前 30 s，mask 已知 IED core、坏道和缺口；PCA/标准化只在 TRAIN 拟合。随后按绝对时间把 observation anchors 与 full-event timeline 合并。

## 4. Bridge-E0 实验工作包 B

### B0–B3

统一 head：log-normal exact-time density + contact-wise mark likelihood；同一个优化器、正则和种子集合。第一轮种子 0–2。

- B0 explicit history；
- B1 history + explicit masked spectral/variance/autocorrelation；
- B2 history + masked raw-derived embedding；
- B3 history + both。

主输出：validation joint NLL/event、timing NLL/event、mark NLL/event；辅助为 next-time MAE、participation BCE、order loss。按患者先聚合，不按事件池化推断队列方向。

Bridge-E0 的 raw-derived embedding 只用于快速判断信息方向；若显示 raw increment，再换成宽 Transformer embedding 重跑 B2/B3。若没有 raw increment，仍保留 B1/T1-spectral 路线。

## 5. T1 工作包

实现并先测试：

1. stable \(K=\Omega-Q\) 与任意 \(\Delta t\) 的 matrix exponential；
2. observation correction 与 generator propagation 分函数；
3. explicit history builder 无 trainable parameter；
4. point-process timing 与 contact mark 共用 \(z(t_e^-)\)；
5. chunk 间 state carry、gradient detach、不 reset；
6. correction-off rollout、wrong-time swap、clamp/reset。

Bridge 的 observation-complete 行不得直接用作 recurrent timeline。T1/T2 数据层必须保留 development 区间内每一个相邻 IED transition；raw/spectral 不可用时只把 `observation_available=false`，不能删除事件或 exposure jump。

优化按 Stage A/B/C：冻结 event-only baseline → 缓存 observation 上训练小 state core → Bridge raw 增量成立后才解冻 raw encoder 最后一层。

## 6. T2 工作包

1. 用 training-only cross-fit expected load 得到 \(\eta_e\)；
2. H3-S0 先运行 \(\tau\to0\) current-event limit 与 1 s、6 s burst controls；随后用固定 25/50/100/200/400 events 的 count clock 定位多事件窗口，并为每个 count window 配一个按 TRAIN median IEI 换算的 physical-time clock；真正 T2 用显式 event-jump 作为同参数量 current-event arm，不用刚性近零 ODE；
3. T2-real 和同容量 time-shifted innovation placebo 使用相同初始化、参数量和时间线；
4. 主对比为 correction-off 后 5/10/20 events 的 joint/timing/mark NLL，并强制拆 participation/order/STOP；STOP-only 只能解释为 termination/extent，不能叫 repertoire；
5. 分三层比较：real vs placebo 回答是否有预测性 forcing；distributed vs event-jump 回答是否超过单个 IED；physical-time vs matched event-count 回答是否识别到真实时间常数；
6. 不用“T1 AND T2 任一失败就全停”的 gate。

窗口选择在固定网格完成前即冻结：50/100/200 events 为 R2 主窗口，25 为 fast control，400 为 long-memory sensitivity。完整五档曲线用于描述宽记忆带和检查单调性，不按患者、source 或端点挑事后最佳档；普通阴性不会阻断另一 source/clock/端点继续运行。

在完整 T2 generator 前先运行 **H3-S0 full-event exposure screen**：用 pre-event covariates 训练内 cross-fit 当前 IED load/participation innovation，分别构造 current-event limit、秒级 controls、physical-time exposure、matched event-count exposure 与只取更早事件、无 circular wrap 的延迟 placebo；在完整当前事件历史之上比较下一事件 timing/mark。physical 网格 748/748、rate-matched count clock 408/408 均已完成；固定 25/50/100/200/400-events 的 count 340 + matched physical 340 cells 也已完成并通过同包、样本、history、封条与 219-cell 精确重跑审计。结果支持约 25–200-event accumulation，但 physical time 未在拆分端点上胜 count clock；分钟数只能作为 rate-matched label。该筛查用于冻结下一版 windows，不作 generator 因果结论。

## 7. 运行顺序与资源

### CPU（可并行但控制 I/O）

- 非交互 shell 必须先把环境自己的 C++ runtime 放到动态库搜索首位：`CMS_CONDA_LIB=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib`，再 `export LD_LIBRARY_PATH="${CMS_CONDA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"`；否则当前系统库缺 `GLIBCXX_3.4.29`，pandas 入口会在启动前失败；
- 特征构建最多 4 个患者并行，`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`；
- Bridge GLM 每患者独立，最多 6 worker；
- Bridge 线性 head 使用零初始化、全批量 LBFGS；跨 seed 逐位一致后不把 seed 重复当作独立证据，科学统计单位仍为患者；
- 正则强度只在每位患者 chronological TRAIN 的末 20% 上，从预先固定的 6 档网格选择；随后用全部 TRAIN 重拟合，development validation 只评价一次；
- 原子写 manifest，已有 DONE 且 hash 一致则跳过；失败不覆盖旧文件。

### GPU（RTX 3090 单卡）

- 同时只允许 1 个训练作业；启动前读 `nvidia-smi`；
- 87-contact Conformer 实测 batch 2 仍可异步 OOM，修订上限为 batch 1 或 contact-token budget <= 180；
- CUDA 调试作业启用同步报错一次，正式长跑关闭；
- 每作业记录 peak memory、effective batch、optimizer steps 和 checkpoint hash。

### 持久化

- 长任务用 `nohup + setsid` 或现有 tmux `epi_prssm_v01`；
- queue、worker、monitor 分离，monitor 不拥有 worker；
- 明确 PID 文件，禁止 `pkill -f`；
- 网络波动不影响本地任务；不在运行时下载模型或依赖。

## 8. 探索性判断点，不是总 gate

- B1/B2/B3 都不增量：Bridge 收口为 no increment，仍可推进 event-only T2。
- B1 阳性、B2 不额外：使用显式 spectral observer，不强行上 raw encoder。
- timing-only：转为 event-propensity state，同时保留 mark 阴性。
- mark-only：优先 H1/H2a repertoire state。
- T2-real 不胜 placebo：H3 当前时间尺度阴性或不可辨识，T1/H2a 不受牵连。
- distributed exposure 不胜 current-event limit：保留当前事件 innovation association，不写成累积 IED exposure。
- distributed event-count 胜 current-event、但 physical 不胜 count：写 recent-event-count accumulation，不写分钟级生理时间常数。
- 仅以下情况暂停相应工作包：sealed boundary 违规、患者/通道/event 映射不唯一、核心 likelihood 实现错误、输出包混用、同一 checkpoint 被不同臂复用。

## 9. 8–10 小时验收物

- 冻结 spec 和本 plan；
- R0.2 closeout evidence card；
- Bridge feature/split audit；
- B0–B3 至少 3 位患者开发结果，目标为 6 位；
- T1/T2 toy recovery + 至少一位真实患者 smoke；
- `RUN_STATUS.json`、`CURRENT_HANDOFF.md`；
- `reports/plain_report_2026-08-24.md`；
- `reports/technical_report_2026-08-24.md`。

如果 10 小时时长结束时长作业仍在跑，报告必须把“已完成”“正在跑”“尚未启动”分开，提供 PID、日志、恢复命令和当前可安全主张；不得用预计结果补表。

## 10. 下一执行版本 R1：真正的 T1，pilot 6

### R1.1 数据与 observer

1. 在同一规则 30 s clock 上冻结 spectral、raw Transformer、combined 三类 observation embedding；每个 anchor 只读前 30 s，IED core \(\pm1\) s 主分析遮蔽。当前 60 s clock 只属于本轮 E0 近似，不带入 R1。
2. raw observer 复用 R0.2 中较稳的 wide Transformer **架构与可选初始化**，不复用其 frequency-forecast latent 作为 state，也不把旧频谱任务当科学监督。先做 frozen-backbone warm-up；随后因 E0 已在 3/6 pilot 显示 raw-only 或 raw-beyond-spectral 增量，按预定 Stage C 解冻最后一个 encoder block，以 `0.1 × LR_state` 在同一个 joint IED timing+mark likelihood 上有限微调。Conformer 不作为 R1 主臂，但保留为后续架构 sensitivity，而不是当前 gate。
3. 保存 all-contact raw 到 event-contact 的显式映射、mask 比例、时间边界与 source hash；同一患者三类 observer 必须共享 event timeline。

### R1.2 最终 joint event model

1. timing 从当前 conditional log-normal prototype 升级为对真实 recorded intervals 积分的 point-process likelihood；缺口不进入 survival integral。
2. mark 接回同包、去重的 sequential contact-RNN；在接入前将 tied group 锁为显式 group identity 下的精确无序、无放回 subset likelihood，禁止有放回近似；状态 adapter 分别报告 participation、order、STOP、same-prefix continuation。
3. event-history baseline 每患者只拟合一次并冻结；T0/T1 共用同一个 checkpoint。T1 主体训练 observation correction、稳定 generator 和低秩 state adapters；raw/combined observer 只在 frozen warm-up 后按上条有限解冻最后一层，不允许全 backbone 与 state 同时从头剧烈变化。
4. 主模型 `state_dim=8`；combined observer 的 `state_dim=16` 只作容量敏感性，不参与模型选择。

### R1.3 pilot 矩阵与判读

- T1：6 patients × 3 observers × 3 seeds = 54 fits；每患者先聚合 seed。
- T0：6 个共享 frozen baselines；不得为 observer 重拟合。
- diagnostics 不另起模型：filtered、matched wrong-time swap、reset/clamp、real-time interval shuffle、anchor 后 H5/H10/H20 correction-off。
- filtered 阳性但 correction-off 阴性：`predictive filter`；correction-off、swap、reset 同向后才称 `autonomous predictive state estimate`。
- timing、participation、order、STOP 分开收口；任一普通阴性不终止其他端点或患者。

## 11. 下一执行版本 R2：T2 generator 与 H3

### R2.1 两类 exposure source 与两类 clock

- scalar load innovation：标量 \(\eta_e^L\)，\(B_\tau\) 为 state 向量；
- contact participation innovation：向量 \(\eta_e^P\)，使用固定 rank-2 exposure→state 映射；current-event 与 cumulative arms 的 rank、参数量完全相同。

每个固定尺度单独训练，不在一个模型内自由竞争多个 kernel。主参数改为固定事件数 \(N\)，不再把分钟数当作已识别尺度；每个 \(N\) 同时运行 event-count kernel 与按该患者 TRAIN median IEI 换算的 rate-matched physical-time kernel。

R2 主集合预先固定为 \(N\in\{50,100,200\}\)；\(N=25\) 是 fast-history control，\(N=400\) 是 long-memory sensitivity。五档均保留在 H3-S0 报告中，但不允许以单患者或单端点的最低 validation NLL 选择 R2 窗口。

另设不依赖固定-N结果的 fixed-physical sensitivity：\(\tau\in\{10,60,360\}\) min。每个 \(\tau\) 同时运行 physical kernel 与患者内 \(N_i=\tau/\widetilde{\Delta t}_{TRAIN,i}\) 的 matched count kernel，用来直接检查跨患者共同的分钟时钟。该层完整报告但不与主 fixed-N 家族合并，不作为是否继续其他 H1–H3 分支的 gate。

### R2.2 训练与 controls

1. 从同一个 T1 checkpoint 克隆，第一轮冻结 observer、decoder、history baseline 与 \(K\)，只训练 exposure edge；第二轮才允许小学习率联合微调 state core。
2. 每个 distributed window 包含 count-real/count-delayed/count-state-matched 与 physical-real/physical-delayed/physical-state-matched 六臂；每个 source 另有共享的 current-event jump real/state-matched 两臂。两种 clock 都只在 IED 时刻通过同一个低秩 \(B_s s_e\) operator 更新 \(z\)，共享 observer、decoder、\(K\)、forcing rank 与初始化；连续 \(u\to z\) forcing 只作后续敏感性，不能直接与离散 count arm 比输赢。
3. seed 0 discovery 使用 6 patients × [2 sources × 3 主窗口 × 2 clocks × 3 arms + 2 sources × 2 event-jump arms] = **240 fits**；seed 1/2 replication 各重复同一冻结矩阵，共 **480 fits**。25/400-event controls 若在 R2 generator 中追加，则单列为 sensitivity，不回写主矩阵。多 worker 按患者并行，科学单位仍是患者，seed 不充当患者。
   fixed-physical sensitivity 另为 6 patients × 2 sources × 3 physical windows × 2 clocks × 3 arms = **216 discovery fits**，seed1/2 共 **432 replication fits**；共享主矩阵的 event-jump/T1 checkpoints，不重复计数。主矩阵与该 sensitivity 合计为 456 discovery + 912 replication fits。
4. anchor 后关闭未来 observation correction、保留真实未来 event forcing；比较 H5/H10/H20。
5. validation 起点使用全部更早 TRAIN 观测和事件做无梯度 causal warm-start；split 只截断梯度和禁止拟合 validation，不重置 (z)、exposure 或 explicit history。只有真实 gap/session boundary 才 reset，并逐患者持久化 warm-start 长度与最后一次可用观测时间。

### R2.3 允许的结论

- real 不胜 placebo：该尺度无可检出的 generator forcing；
- cumulative 不胜 current-event jump：仅支持单事件作用；
- count distributed 胜 current-event、physical 不胜 count：recent-event-count shaping；
- physical distributed 进一步胜 matched count：才允许报告 physical-time shaping；
- timing-only：event-propensity shaping；
- STOP-only：event-termination/extent shaping；
- participation/order 或 same-prefix continuation：functional repertoire shaping。

pilot 6 全部完成后才判断是否扩到 34 人；扩展是资源选择，不是把 pilot 阴性变成项目总 gate。

## 12. R3：冻结状态后的发作层

1. T1/T2 完全冻结，不使用 seizure loss 回训状态。
2. 主层使用 27 patients/361 seizures；203 次近两小时有 IED 的高可观测层仅作 sensitivity。
3. 359/361 已有精确 crosswalk；未匹配两例不强行映射。所有发作总体为主，subtype 只在患者内支持足够时探索；broad_ER/gamma_ER 真正双组患者仅 6/4 人。
4. 严格六维 case matching 在本队列 0/361 可行，不再作为主设计；使用连续 patient-within probe、pseudo-onset、leave-one-patient/seizure 与 coverage gradient。
5. H2b 报 T1 state 的额外关联；H3b 报 T2-specific exposure-sensitive component 的额外关联。二者均不写成 IED 导致发作。
6. held-out preictal raw 在训练/预训练中保持封存，但评估时由冻结 observer 因果读取到各 5/15/30/60/120-min lead 为止；绝不读取 lead 之后或 onset 后数据。主层报告这种真实可用的 filtered state，另以提前 anchor 后 correction-off rollout 作为 autonomous-risk-state sensitivity。

## 13. 最终扩队列与正式分区

- 34 人 development 扩展前冻结 observer family、state dimension、时间尺度集合、主端点和 figure contract。
- 正式检验分区只在上述选择全部锁定后一次打开；不得用它选择 \(\tau\)、患者、亚型或图面板。
- 每个阶段均交付 plain/technical 两版报告、machine-readable evidence card、运行 manifest、同状态图 metadata 与 `CURRENT_HANDOFF.md`。
