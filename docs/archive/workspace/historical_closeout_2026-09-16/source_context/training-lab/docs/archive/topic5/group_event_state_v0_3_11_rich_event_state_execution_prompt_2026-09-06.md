# v0.3.11 rev3 执行prompt：丰富事件状态的自主工作窗口

将下方完整正文交给执行Agent即可。本文是执行指令，规范细节以所链接rev3 spec为准；旧v0311 handoff和273次矩阵均不生效。

---

你负责实现并执行Group-Event State v0.3.11 rev3的首个自主工作窗口。请主动完成已授权的实现、修复、必要验证、实验和交付，不停在计划或“已启动”。从开始执行时记录t0，工作窗口默认10小时，8小时开始准备交付。时间是工作预算，不是必须在10小时内获得阳性或完成原始科学闭环。

本次目标是：**接通丰富群体IED历史→因果状态过滤→真实未来空间/形态/数量预测→同checkpoint冻结细触点及可估发作分析，交付可信的完成子集、限制和可恢复的剩余队列。** 若使用goal工具，把它设为有界开发工作包；已有未完成goal先识别其范围，不覆盖他人的goal。本次工作包交付可以完成，原始科学闭环仍按实际证据开放。

## 1. 必读与工作范围

主规格：
`/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-ges-v033-training-lab/docs/archive/topic5/group_event_state_v0_3_11_rich_event_state_execution_spec_2026-09-06.md`

前轮问题来源：
`/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-ges-v033-training-lab/docs/archive/topic5/group_event_state_v0_3_10_independent_scientific_review_2026-09-06.md`

工作树：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-ges-v033-training-lab`。
新输出：`/data/hfosp_group_event_state_rich_event_identification_v0311/`。
Python：`/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`。

先检查当前AGENTS、git status、活动进程、GPU及已存在的新版输出；保留所有既有未提交工作。新模块使用独立`src/topic5_group_event_state/v0311/`及明确新runner，不能原样调用旧trainer后只换版本号。已有新版实现则核实是否符合rev3后续用，不重复从零开发。

旧v0310、v039 repaired和independent_review根只读。只用E1096/E1125/E253已授权、已打开的资料；新划分不开放原development/sealed。不得提交、推送、改写原final_reports或终止其他任务。若用隔离checkout，需要保留当前所依赖的未提交源码清单，不能只检出旧HEAD再冒称同一版本。

实施、可逆排错、必要的同合同实验和调度已经由本prompt授权，不重复索取这些许可。新数据范围、primary/科学目标变化或合同外不可逆动作另行说明；等待只暂停依赖它的分支。不得为了填GPU启动无科学解释的任务。

## 2. 本版已经确定的改变

- 首轮P_stats/P_marks配对；P_marks真正进入编码器，保留触点、相对时序、频带/波形、事件顺序与dt。分钟是慢更新时钟，不是丢掉细事件结构的理由。
- 三主目标为数量、粗空间、条件形态/传播。总负荷次要；细contact identity未直接监督producer，STOP独立。
- q显式维护m/P，观测证据更新完整预测分布；不是给所有样本相同创新。主训练没有rev2的潜路径条件KL、KL warm-up和绝对后验方差下界。
- 使用spec的Gaussian证据过滤和严格因果预测损失；它是学习的近似过滤器，不能报告精确Bayesian后验/ELBO。
- L/N共享精确线性OU子步＋非线性残差分裂；主24维完整协方差。不要恢复一分钟显式Euler后仅检查D>0。
- 首批2h有梯度前缀，持续状态与严格历史臂分开。0.5/2/8h重拟合后续接入；24h不是首轮前置。
- 主体是12次S-E＋6次S-ID＝18次；加2个指定消融为20次紧凑开发包。旧273次不是待执行清单。

## 3. 先核实数据，再实现最小完整链

复用`v039/human_data.py`及raw/cache接口时，注意其rich字段按闭合块发布；`build_group_event_state_v039_fit_measurement.py`包含闭合200s段测量和小时available_time。不能把缓存时间改早冒充因果。核对participation/channel names、relative_delay_s、tied_group_id、band_features、cross_band_lag_s、waveform_*、has_waveform的来源和可用时刻。

创建measurement_manifest和split_manifest，包含字段单位、频带、质量/缺测、原始支持及release_time；所有FIT-only词表/模板/缩放有来源。先做30个覆盖边界的真实未来截断检查，合法核心工作继续时补至100；原始缺字段只限制对应端点，不伪造波形、延迟或临床标签。

新接口建议分为packets/encoding/filtering/dynamics/observation/training/evaluation/transfer，名称可按实际代码调整。先接通一位患者全链，不只训练count而把形态和identity留空。预定顺序E1125、E1096、E253；这是开发排队顺序，不按新outer结果换患者。

实现spec中的先预测整个测量区间、保存预测、包发布后评分、再同化；count使用区间rate×exposure积分。条件事件表达标明已知事件时刻/K/prefix。缺测不作零事件；发作/postictal排除区间reset或排除跨越历史，不当普通技术gap。

## 4. 核心验证和合成任务

以下是会使结果无效的核心检查，不是要求所有结构世界先通过：

1. 解析线性高斯核：不规则dt、强/弱观测、gap与恢复，m/P及创新似然对参考Kalman；FP64相对误差目标1e-6。检查新观测可收缩继承方差，缺测方差按模型趋稳。另做同训练器的学习版，区别核正确与神经证据有效。
2. 分裂求解器：L的精确OU参照，N的半步/四分之一步分布和效应稳定性。随机流耦合，不把重抽噪声当积分误差。
3. 形态信息世界：保持率/粗composition不变，慢变量只调制shape/传播；固定I-L-G1，P_stats/P_marks×旧/新目标×2个独立数据实现＝8次小拟合。另测同通路identity注入/零效应。它们是接口/鉴别初筛，不能叫正式功效已校准。
4. 数据权限、概率归一化、相同评分支持；微批累积对整批标量目标的梯度；激活重计算与恢复的随机流；单写者认领。
5. FIT真实损失对早期0–0.5/0.5–2h数据的梯度及有限扰动。outer梯度只能最终冻结后描述，不触发选模/调参。

真实数据通路、概率/梯度核合法后，即可运行依赖已满足的人体分支，不等全部动力学真值世界。若某个正对照无鉴别力，限制它对应的科学结论，继续其他合法任务；若实际代码错误，则修复受影响链并弃用受影响卡。

## 5. 主体、消融、冻结读出与优先队列

种子20260906；复核种子20260907/20260908。

- S-E：3患者×P_stats/P_marks×I-L-G1/C-N-G1＝12次；同一canonical时间前推段，三目标一致。I-L与C-N多因素不同，只筛查表达能力，不能直接证明N或耦合。
- S-ID：3患者×P_stats/P_marks×I-L-G1＝6次，1个分散连续留出fold。保留这6次，不能再次只做单段前推。
- E1125、P_marks、I-L-G1、canonical S-E：新增旧目标消融1次、新增传播目标及其原始输入同时移除的跨视图消融1次；详见spec。字段不可估就报不可估，不按结果更换患者。

每个主体与充分拟合的截距/时钟、近期率、完整marked-history、常数状态比较，统一支持。比较双方同等优化机会；记录每档LR/预算、实际参数更新和所选初始化。不能在相同480步截断两臂，再把L收敛慢说成N更好。

一次有梯度的连续episode过滤服务多个起点，单起点rollout到最大lead后切四个时距；不要为128个起点各自重放完整历史，或把共享前缀重复计权。参数更新后旧图/旧参数状态缓存失效。主优化和MC规则完全按spec。

每份冻结checkpoint及时导出：逐包先验、1/5/30/120min、HOLD/EVOLVE/RELAX、细contact identity、同支持常数/marked-history。固定K集合概率需正确归一化，STOP分开；组内identity控制粗community。只有未发生的未来不得输入，目标视图的合法过去在普通主体中可以使用。

producer按间期INNER冻结后，CPU立即做可估S-A/S-B，比较固定空间trait、近期历史、trait＋state。发作不足仍交个案/台账，不伪造风险分母；不等七家族、多fold全部完成。提前看到发作不能反选producer；受其影响的新版本明确属于后续开发。

剩余优先级：实际bug修复 → 承重对照续训 → 同患者缺失配对 → 冻结细迁移/发作 → 6次S-ID与2个定向消融 → 9次严格0.5/2/8h历史 → 按已冻结INNER家族的6次优化重复。历史/复核共最多15次优先扩展；结构补充最多12次、必须成科学上明确的配对。不要根据outer阳性给赢家增加预算。

## 6. GPU、恢复及监控

现场检查两张3090的可用性，OMP/MKL/OPENBLAS/NUMEXPR线程限制避免争抢。每卡先1worker；以真实最长shape、完整forward/backward/optimizer/MC设置profile，峰值≤18GiB/24GiB。只有总显存仍合格、吞吐提升至少20%才每卡2worker。microbatch先1个episode/16起点并累积，OOM优先减微批/并发，不暗改历史和有效权重。

首个FIT 300–500更新和一次完整训练后重算ETA；不要把旧F/L/N耗时按数量直接缩放为新版保证。并发有数据准备、CPU读出和报告，GPU优先填已有合格任务，不跑无关负载。

任务身份含配置/数据/split/源码版本；原子claim、单写者、临时写后原子发布。running尚未出card不等于pending。恢复包含模型、optimizer、LR、全部RNG、episode/起点位置及阶段；故意恢复核验后再依赖续跑。配置变化不能沿用旧身份覆盖结果。

每30s看进程/显存/队列，合格待办存在但5min没有训练就定位并补位。监控正常时安静；失败、关键科学变化、阶段完成或需用户输入时通知。不要仅启动后台任务便结束整个工作窗口；同时持续修复和汇总。若没有合法GPU待办，记录具体科学/数据依赖，不为“满载”取消检验。

## 7. 窗口交付与退出规则

t0+8h开始冻结阶段结果、准备报告；9h后不新开预计无法达到安全checkpoint的任务；最迟10h交付可恢复状态。若时间不足，不共同截断臂后叫收敛，也不删失败/不可估卡。全部有价值的窗口任务提前完成可以提前交付，不为凑时长重复实验。

交付新根中的：

- 中文科学报告：输入包含什么、哪些表达有增量、HOLD/EVOLVE/RELAX差别、历史长度、同checkpoint形态/发作结果、能改变哪些旧结论。
- 机器摘要和共同物理窗表：所有成功、失败、未估计、预算停止与仪器无鉴别力分别标记；患者/日期/物理块/事件/发作簇/种子分母都保留。
- 训练与逐层参数表、过滤/求解器证据、来源和恢复信息。
- 有数据时才产PNG/PDF/中文README，Agent自查后标待用户目视；不编造故事性同前缀例图。
- 剩余execution_plan、精确续跑命令、检查点位置、最高优先配对及按实测更新的ETA。需要安全暂停的本任务进程保存恢复状态；不留下无人负责的写入者，不影响其他任务。

最后回答：原始状态识别、时间组织、冻结发作三个层级分别到了哪里。工程工作包完成不等于科学闭环完成；非零梯度、后验更新、三个tau及一个总分不证明IED生理反馈。

只要仍在窗口内、任务已授权且可推进，就继续实施和排错，不反复询问常规许可。遇到权限/科学目标必须变更时说明具体依赖，其余独立任务继续。不要提交或推送。
