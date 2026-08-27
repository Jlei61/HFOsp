# Continuous Marked State R1.7A / T2 R2.0 冻结合同

## 1. 本阶段只回答什么

R1.7A 在完全未参与架构、优化器、epoch 预算、阈值或长尺度选择的
development 患者上，检验冻结的 R1.6 配置能否复现两件事：状态是否比逐窗口即时读码更有用，
以及正确时刻的状态是否比同条件错误时刻状态更有用。只有通过这两层、且具体改善落在
first-subset、later-continuation 或 same-prefix continuation 的患者，才进入 T2 R2.0。

T2 R2.0 只问：条件于 IED 前状态和已知历史后，最近约 100 次 IED 的意外载荷或意外空间组成，
是否仍增量改变下一次 IED 的表达。它不检验一千至一万次尺度，也不再运行六小时 boxcar。

## 2. R1.6 验收与旧路线退役

- R1.6 仅验收为优化器、训练充分性和可识别性诊断完成，以及 `epilepsiae_384` 的单患者
  development 支持；不外推为队列证据。
- R1.5 的 epoch-0 比较见过选择数据，旧 no-update 不具科学阴性解释力。
- H3-long 的 N=1,000/3,000/10,000 与六小时 boxcar 退役：旧实现混有免费截距、结构零、
  名义尺度与有效时间尺度不符、独立窗口不足或拟合发散。旧数值只留作审计史，不进入新汇总。

## 3. R1.7A 队列冻结

先排除所有参与过旧决策的患者，再要求不少于 6 个触点、TRAIN 1,000 次事件、validation 300 次
事件、TRAIN 6 小时和 validation 1.5 小时记录支持。各数据集按 validation 事件数、记录时长和
患者名的固定顺序取前 5 人，共 10 人。选择脚本不得读取任何模型结果。

冻结患者为：`epilepsiae_1073`, `epilepsiae_1077`, `epilepsiae_1125`,
`epilepsiae_1146`, `epilepsiae_253`, `yuquan_liyouran`, `yuquan_xuxinyi`,
`yuquan_zhangbichen`, `yuquan_zhaochenxi`, `yuquan_wangyiyang`。

## 4. 训练与评价

- 输入、exact timing + tied-group mark likelihood、explicit observer、`dz=8` 和稳定生成器不变。
- 公共配置固定为 R1.6：prefix 12 passes、LR `1e-3`、chunk 128；alignment `8+8` passes、
  state LR `3e-4`、observer ratio 0.1、chunk 32；AdamW、warm-up 0.1、clip 5、无 weight decay。
- 每位患者固定 seeds 0--4；所有 seed 都汇报，seed 只表示优化离散度。
- TRAIN 内仍按记录顺序使用 0--60%、60--80%、80--100% 三段选择；development validation
  不参与模型或 epoch 选择。
- development validation 按实际记录时长分开：前 60% 为 `D_state`，仅检验 H1/H2a；后 40%
  为 `D_mechanism`，只给 T2。跨缺口的墙钟时间不计入 60/40。
- H1 主比较为 persistent−memoryless 与 correct−5 matched wrong-time。
- H2a 主端点为 first subset、later continuation 和 same-prefix continuation；STOP 与 timing 次要。
- 患者效应先取五 seed 中位数；科学区间使用仅由 TRAIN 确定长度的 session/连续时间块 bootstrap。

## 5. T2 R2.0

- 仅对 `D_state` 同时显示 persistence、correct-time 与健康非零 T1 的患者运行。
- `D_mechanism` 必须有可评分 next-event support；100-event block 少于 5 个只报个案。
- source 固定为 load innovation 与去除 load/size 后的 participation-composition innovation；
  两者均只用 TRAIN cross-fit。
- exposure 为 `x_e=rho*x_(e-1)+eta_e`, `rho=exp(-1/100)`，中心与尺度只取 TRAIN。
- 四臂：T1 no-edge、real cumulative、state-matched non-overlap、current-event-only。
- 不给 exposure 自由截距；冻结 observer、K、history 与 decoder，只拟合 signed low-rank `B`，并要求
  `B=0` 时目标梯度非零。
- 主评价只在 `D_mechanism` 的 next event；H5/H10 one-shot 为次要，不开未来 observation correction。
- 在 N=100 出现可靠增量前，不运行 N>=1,000、物理时间 family 或 seizure probe。

## 6. 结论边界

普通阴性不阻断其他患者。formal/sealed 分区、发作 probe 与 paper-ready 图始终关闭。R1.7A 仍是
development replication；T2 阳性也只能称条件增量支持，不能直接称生物学因果机制。
