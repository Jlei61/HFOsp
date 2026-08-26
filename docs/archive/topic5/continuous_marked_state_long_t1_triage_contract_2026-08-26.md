# 长序列 target-trained T1 分诊合同（2026-08-26）

## 目标

在不打开正式检验分区的前提下，检验先前长尺度 H3 患者在 R1.3 的完整
IED timing + sequential mark 目标训练后，能否形成跨窗口预测记忆。此阶段先修测量
工具，不把增加事件窗口或 seed 当作 H3 证据。

## 固定患者与理由

- `yuquan_hanyuxuan`：R1.2 中 predictive + persistent 为 7/7，是正参照；
- `yuquan_chenziyang`：R1.2 中训练启动但 development validation 不利，是边缘失败参照；
- `yuquan_chengshuai`：拥有当前最强的 10,000/15,000-event 连续支持，但 R1.2 为 epoch 0，是高事件量失败参照。

名单在读取 R1.3 结果前冻结。彭子航与 E922 暂不进入第一批，避免在确认仪器方向前
扩展高密度作业；普通阴性不改变上述三位的执行顺序。

## 模型与训练

- 使用 R1.3 target-trained explicit observer；目标仍是完整 recorded-time timing likelihood
  与 exact tied-group sequential mark likelihood；
- 每位 3 个优化起点，分别从同编号的 R1.2 checkpoint 初始化；必须报告 checkpoint
  hash 的 distinct 数，seed 不是生物学重复；
- observer alignment 4 epoch + joint alignment 4 epoch；在 TRAIN 内 chronological
  inner-validation 选轮次，development validation 只在训练结束后评分；
- generator 保持冻结，避免同时改变状态坐标、观测器和自然演化；
- 第一批只跑 explicit。只有 explicit 形成可用状态后，raw increment 才作为配对二级实验，
  不用 raw 阴性阻断 H3。

## 证据分层

逐患者、逐 seed 分别报告：

1. `selected_total_epoch > 0`：target alignment 实际发生；
2. persistent 胜 memoryless：存在跨窗口预测记忆；
3. correct-time 胜 matched wrong-time：具有时刻专属性；
4. first subset、later continuation、STOP/size 分解：状态信息落在哪个 IED 端点。

前两项同时成立才可进入探索性 H3。第三项决定结论命名强度，但不是其他探索的总 gate。

## H3 进入条件与主对照

T1 完成后才计算每个候选窗口的真正不重叠 TRAIN/validation 整窗数。这里的“整窗”是
真实 N-event 暴露与向前平移 1,000 events 的 causal-delayed 暴露所需历史的并集，
即按 `N + 1,000` events 的支持区间计数，不能只按名义 N 计数。只有前状态可用且
独立支持不退化时才启动新的 H3；否则如实收口为分诊结果，不继续堆 N。

H3 仅接受：

- `real_minus_intercept_matched`；
- `real_minus_causal_delayed`。

`real_minus_no_edge` 只作免费截距伪迹诊断。实际时间尺度必须同时报告名义 N/小时、
50%/90% 权重时间和不重叠整窗数。下一 exposure 至少包含 load 与
participation/repertoire composition 两类独立定义。

## 资源与恢复

- cache 最多 2 个 CPU/I/O worker；GPU fit 最多 2 个 worker；每 worker 1 个 CPU thread；
- fit 保留现有 chunk 自动减半 OOM 恢复；结果按 `result.json::COMPLETE` 断点续跑；
- 后台队列使用独立 session，主会话或网络断开不影响；
- 正式检验分区、seizure probe、paper-ready figures 均保持关闭。

## 输出

输出根：

`results/epi_prssm/continuous_marked_state/r1/r1_3_long_t1_triage/`

必须包含 `STATUS.json`、`summary.json`、逐作业日志、逐 seed checkpoint/result，以及
完成后的白话和技术报告。
