# 可直接交给执行 Agent 的 prompt：多事件分布学习 v2.1

请实施下面的已修订设计，持续推进到 G4 的科学审阅包，不停在建议、controller 写好或两个聚类标签都出现。常规实现、排错和合同内运行已授权；不自动扩大搜索、不自动冻结模型或进入 Fig.5，不 commit/push，不覆盖其他任务或旧结果。

## 工作点与必读合同

工作目录：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix`。

先读本目录适用 AGENTS.md，以及以下文件（以下相对路径均相对此工作目录）：

1. `docs/archive/topic4/sef_hfo/same_network_multievent_distribution_design_v2_1_2026-09-07.md`：完整科学与执行合同，本 prompt 为执行摘要。
2. `config/topic4_multievent_distribution_search_v2_1.json`：机器可读设计，**不是旧 worker 可直接读取的 execution_config**，需要适配。
3. `results/topic4_sef_hfo/multievent_distribution_search_v2_1/revision_manifest.json`：旧输入哈希和五类上游来源说明。
4. `results/topic4_sef_hfo/multievent_same_network_propagation_review/scientific_review.md`：此前多事件/原生场审阅，理解为什么训练改善不足以证明传播恢复。

目标是在**一个固定网络的一段自主连续运行中**生成多次间期事件，使单事件参与、rank、时差与空间特征的分布接近患者。不能跨不同网络拼出“两类能力”，也不要求单次事件同时符合两类模板。TB 右下起始及其后续空间路径是提名后的验证问题，不写入 loss、初始化/core 区域偏好或事件专属刺激。不要增加序列损失；本轮不声称恢复模式切换动力学。

目前交付的是修订设计、配置和来源 manifest，**v2.1 的 D_off 实现、两个离线扫描、完整 controller 和新物理链尚未因本次修订完成**。启动时查看实际进程和产物，避免重复其他任务刚完成的工作；不能把旧 v2 的8项检查当成新目标已经通过验证。

## 可复用资产与禁止误用

- `src/topic4_multievent_distribution_objective.py` 与 v2 `training_objective.pkl` 实现的是旧 D16。保留 phi、psi、患者目标、固定聚类器和正尺度，创建独立版本的 D_off 实现/payload。pickle 会依赖导入类，不能只原地改旧类使旧 pickle 重读后悄悄改变含义。
- `results/topic4_sef_hfo/multievent_distribution_search_v2/initial_candidate_manifest.json` 已有48条件、master seed **3854897931**。校验继承哈希，保留真实候选和顺序，不再生成一套“新的首批48”。旧 manifest 的 config_sha256 指向旧配置，继承关系用新 manifest 表达，不改旧哈希。
- **不要把 `scripts/prepare_topic4_multievent_distribution_search.py` 当 v2.1 启动入口直接重跑**：它面向旧目标/旧准备流程。写新的版本化入口或明确隔离的适配器；不要让旧资格检查阻塞 G0/G1。
- 复用 `scripts/run_topic4_multidimensional_worker.py`、`src/topic4_observation_repaired.py` 和 `results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2/observation_contract.json`；训练事件来自冻结修复读出，不以其他 lineage onset 替代质心表。
- 可参考上一轮 execution_config：`results/topic4_sef_hfo/multidimensional_interictal_pilot_round1/execution/paired_round1/execution_config.json`。新任务使用独立输出目录 `results/topic4_sef_hfo/multievent_distribution_search_v2_1/`，保留旧12秒结果作为历史诊断。

## G0：先得到可信的24秒物理轨迹

从现有48清单预先固定历史基线与两个随机条件，legacy 单元 `(2511,2511)`，各24秒；先小并发测量，再按真实资源增加。G0不等待损失比例/散布扫描，不等待患者波形小包，不要求出现两类或拟合变好。实现足以派发这三次的最小 adapter，检查参数实际作用、原生场与原始输出、实际时长、冻结读出和资源记录。

worker 已有 `--seed`、`--topology-seed`、`--dynamics-seed` 接口；split 模式 `--seed` 要与 topology_seed 一致。训练两单元使用 `(2511,2511)`、`(2512,2512)`。生成 worker 配置时同步其 `search.fit_network_seeds`、确认 topology 池和 `search.dynamics_seeds`，不能只有外层 JSON 更新而内部仍拒绝新 seed。输出和恢复索引使用 `candidate_topo_T_dyn_D`；核查真实 CLI 和配置再调用，不猜测完整命令。

每个条件的边权变换从不可变基线计算，不能从上一个候选继续乘。以实际 mask、delta_vtheta、vtheta 核查 core 交换等价性；同一拓扑下相同物理场可复用但保留提案身份，不能只排序中心就宣称所有图都等价。逐参数核对 EE/EI/IE 和 tau_GABA 真实作用、固定图 EE 重分配的入度总权重守恒。记录变换前后实际加权边位移主轴、轴比、距离分布及各目标总输入误差，按稀疏块累积，避免为审计复制全边表。近各向同性不硬报轴角。GABA 衰减仍非积分剂量匹配，不能直接归因于纯时间常数。

资源为**最多8 worker，不能默认8个全开**。RLIMIT_AS=18GiB 是虚拟地址空间限制，不是RSS或不会OOM的保证。测量整个进程树 RSS/PSS、虚拟空间、峰值与输出/临时文件增长；计入分析、压缩、GIF、其他作业未来增长和至少40GiB系统余量，磁盘至少留30GiB并预估剩余产物。cgroup 只有在实际授权的 memory controller 生效时使用，否则进程树监测＋保守准入。工程失败停止新派发并收尾，不终止其他任务。

`physical_status` 和 `execution_status` 分别保存。已满足物理 runaway 可与 OOM 同时成立；数据不足则物理状态 unknown，不把OOM自动解释为runaway。合格canary且物理/读出/时长/seed/输出合同一致可复用进G1。仅更换离线loss通常允许重评分，不因此重跑物理；旧12秒不能充作24秒。

## G1：完成48×2，同时落实两个小扫描和患者追溯

继续固定48条件×2单元，共96次（含可复用canary）。正常低事件或runaway均保留，不按传播好坏停批次。训练payload与优化器只含规定的训练信息，不读取路径验证、PROBE支持表或GIF结果作排名。

修订主统计量为：

`D_off(X,t) = ||mean(X)-t||² - mean_i||X_i-mean(X)||²/(N-1)`。

称作“去自配对的经验训练统计量”。负值有效，禁止 max(0)、abs 或 sqrt；对连续相关事件不称无偏生成分布估计。phi、psi_k、患者目标及0.5/0.5保持；正尺度固定为 `a_global=0.0425770294722167`、`a_modes=0.21306191302831015`，不重新用D_off的CAL中位数作分母。每单元独立计算两分量后，两单元等权平均，不池化事件。

旧D16仅作有限事件评分诊断。同一候选任一训练单元 N<16 时正式loss=null、不排名、不对剩余单元单独取平均；数学上D_off的N下限为2，16仅为本轮约定。保留实际N、逐模式数、原尺度和归一化的D_off/D16及单元明细。

与物理运行并行完成两项小扫描，细节沿用完整设计：

1. 常数维二项枚举＋完整特征混合比例扫描：包含p=0.2/0.1/0.02与患者p，同一随机抽样并列新旧分数，N=16/32/64、每点128重采样。
2. 固定参与mask的模式内时间散布缩放0/0.5/0.75/1/1.25/1.5，重算rank/核特征，不以截断改变时间；记录标签变化与不确定性。这只检测时间/rank变异，不冒充全部生物散布。

补充紧贴改动的检查：D_off与显式i≠j求和一致；负值在序列化和排序中不被截断；正尺度不变；缺事件不发生部分单元平均；恢复不重抽/重复派发；改变验证产物不改变提名；同图换噪声不改变静态数组；延迟批次不依赖完成顺序。不要扩为几十项前置门槛。若扫描揭示明确系统性目标缺陷，保留G1轨迹，只暂停G2并报告决定性问题，不回退D16或按曲线挑权重。

现在就恢复患者原producer、event ID、块、窗口与通道映射，冻结每模式32个及各32备用ID；每块最多8、尽量≥4块，实际不足如实记录。抽样不看路线或候选表现。可复用主工作区 `/home/honglab/leijiaxin/HFOsp/scripts/plot_topic5_interictal_event_envelope_field.py` 的低层 `load_events/build_event`，先核对真实接口；不调用默认方向挑例入口，不用rank近邻猜ID，不覆盖正式视频。

患者小包按数据完整性立即恢复，但候选比较等提名后进行。保存完整包络、时间轴、窗口、有效质量/mask、t10/t50/t90、左右首末10ms边界质量、截断状态和不可读原因。窗口截断时分位时刻只解释为窗口内条件量。平衡的32+32不估计自然模式频率，频率沿用完整合格事件表。原始文件缺失只暂停对应患者验证，继续仿真与模型展示。

## G2：固定排序后进行唯一一批16条件自适应搜索

在生成后代前写定 ranking freeze：D_off实现/payload版本、原正尺度、两单元聚合、N16规则与缺失分数语义。保留11维原范围、几何规则、Z/M关闭和固定噪声；不重建更大参数网。

每群体8目标，从持久化父代快照生成整批，DE/rand/1/bin、F=0.6、CR=0.8。派发前落盘目标索引、互异且排除目标的3个donor、所有随机数、至少一维变异的交叉mask、边界反射及几何拒绝的提案顺序。两个群体共16后代完成后统一更新，不允许先完成worker改变其他后代。

几何有效但INSUFFICIENT_EVENTS的向量可作donor/目标，不伪造数值loss；可评分后代替换不可评分父代；均可评分才按更低L_off替换，相等保留父代；均不可评分保留父代而不宣称拟合优劣。工程未完成要先恢复或说明，不把它当已完成的低事件。重启只恢复已落盘计划和随机数。

保留已有Sobol提案索引与检查前缀的真实几何拒绝率，称随机化空间填充设计，不宣称完整2^m网平衡。64条件池旁路并列L_off、旧v2 L16、原round1 pooled joint及提名差异，只有L_off进入优化。这能说明评分如何改变选择，不能拆分新目标与更宽范围的因果贡献。

每群体从全部已完成可评分候选按L_off提名前两名，物理等价去重、按榜递补，最多4候选；与历史、support_rank、old_joint三个baseline合并去重后最多7条件。保存 nomination 文件后才打开候选验证。

## G3：2张新拓扑×2段噪声，最多28次确认

拟定组合：`(6101,7101),(6101,7102),(6102,7101),(6102,7102)`，第一个数为topology_seed，第二个为dynamics_seed。先查本任务实际使用记录；冲突时一次性分配未用ID并冻结全组合，不能看结果再换。所有条件同样配对，每次24秒，总预算不增。

固定网络包含位置、E/I身份、边、时延、静态异质性、core成员、阈值偏移和变换后权重。同一条件、同一拓扑两次重演必须数组一致；相同确定性初态，只换全局OU、空间OU和外部Poisson实现。当前worker/substrate已有分离seed接口，优先复用；检查初态与随机流真实实现，保留legacy调用语义，不因为观测代码修改而无意改变随机流。不同参数条件当然可有不同权重或阈值。

逐运行保留实际N和逐模式数，以患者匹配实际N及块结构的重采样变异作参照。**不要求每次都看见两类**；少数类未观察到时报告支持有限，条件传播指标NOT_ESTIMABLE，不宣称机制不存在。标签用于分组，另看模式参与—时差联合结构、到两个患者簇的绝对/相对距离、散布、原生场与完整包络。四次重演不足以分解全部随机效应。

用已有轨迹给四个6秒段的数量、比例、实际有效时长和间隔；按qualifying interval中点归段，跨段间隔归后事件并标记，标注既有0.5秒burn-in。检验两类是否只分居启动/后期，不把此诊断接入loss。

## G4：交付能辨认差距的多事件证据

预先固定代表组合为数值最小topology/dynamics；按时间顺序展示该组合全部检测窗口，并展示同一拓扑另一动力学重演，避免只看一个漂亮事件或只看一段噪声。其余组合有完整逐事件表和原生场抽帧，发现差异再补相关窗口GIF并说明选择原因。每个组合均有完整24秒概览；标记训练排除、检测窗口和跳过时间，不隐藏runaway或不相容事件。

对照 Fig.2C/Supplementary Video 1 的 TA/TB 与预先固定的患者小包；示例不是患者全部方差。时间对齐使用既有窗口/观测定义，不为视觉吻合调时间或空间轴。最早质心/t10都不是神经元起燃；如中部质心偏早，要用完整包络和原生场分清早起始、持续短或末端长尾。

验证GIF所有帧可解码、时间和事件覆盖齐全，实际打开原生场/读出/患者帧目视审查，保存可点击PNG/PDF/GIF和相应事件表；图生成后按仓库要求写中文figures/README。Agent自查完成不等于用户人工验收。

中文审阅报告回答：训练分布是否改善；患者条件分布是否相容；同一张图换噪声后是否保留能力；偏差来自读出时间表征、原生传播还是事件支持不足；哪些层因原始数据缺失不可判断。不能把两个标签共存称为两个起源核，不能从联合参数搜索推导机制唯一性。

最多128训练＋28确认＝156次正式24秒运行，兼容canary计入，不新增代数或确认预算；工程重试单列原因、占用与产物，不能作为额外候选搜索。G4结束为 `ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW`。若同图噪声改变后支持不稳，提出针对随机性/证据量的小实验；若质心好但早期包络错误，提出时间表征检查；若原生场缺传播分支，再提针对core或连接的配对干预。**只提出有证据指向的下一步，本轮不自动接新搜索或Fig.5。**

环境：`/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`；`LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib`；`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。交付实际启动命令、配置、输入来源、完成数、物理/工程失败数和未完成层；此prompt不提供尚不存在的controller命令来假装可以直接运行。
