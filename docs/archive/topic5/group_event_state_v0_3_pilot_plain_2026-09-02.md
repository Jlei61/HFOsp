# 群体间期事件状态 v0.3.1：审阅后收口报告（白话版）

## 一句话

v0.3.1 把 split、有效记录时间、冻结 contact grammar、连续时间扫描和 open-loop 评分接通了，但没有完成一次能够裁定慢状态的实验。最准确的状态是：**仪器跑通，state learning 未决。**

## 为什么旧结论必须更正

旧分析比较的是 `state S` 单独预测和显式历史 `H` 单独预测。真正的问题却应该是：在同一个 `H` 基础上，加动态状态是否还有增量，即 `H+S` 是否优于 `H`。这个主比较没有运行。

旧 wrong-time 比较也只证明某些 state 数值与时刻有关；它没有回答这些数值是否在 `H` 之外仍有信息。旧图和旧报告因此不能再写成‘H1/H2a 阴性’。

## 三位患者分别说明什么

- `epilepsiae_1146`：选中 epoch [10, 10, 11]；event encoder / state / adapter 相对更新 0.00365 / 0.00005 / 0.51746。5/30 分钟 correct-time 相对 shifted 有利，但 state 单独仍未超过 H；三个 seed 都在预算末端，最多支持‘存在时刻相关信息’，不能支持 residual state。
- `yuquan_pengzihang`：选中 epoch [0, 0, 0]；event encoder / state / adapter 相对更新 0.00138 / 0.00001 / 0.01861。三个 seed 都选择第一个训练 epoch，120 分钟无可评分 anchor；这是当前合同不适配/未形成有效更新，不是 state 阴性。
- `yuquan_zhangkexuan`：选中 epoch [0, 0, 0]；event encoder / state / adapter 相对更新 0.00037 / 0.00000 / 0.01527。三个 seed 都选择第一个训练 epoch，correct 与 shifted 几乎相同，120 分钟严重失校准；更像训练塌缩，不是生物学阴性。

## 新增审计发现

1. adapter 不是数学上的全零死区：投影权重非零，初始 gate 为 sigmoid(-4)≈0.018，已有梯度记录也非零。但实际输出调制量没有被测量，而且全局 gradient clipping 在大多数 chunk 触发，所以有效 state path 是否真正学起来仍未知。
2. grammar 在每个时刻对 state 做 LayerNorm，可能删除有意义的幅度；state update 又能任意混合旧 state，因此所谓固定时间尺度只是 nominal label。这两点使当前 latent 不能按 5/30/120/360 分钟生理时间常数解释。
3. TBPTT 实现确实在事件数或30分钟任一先达到时切 chunk，旧技术报告写成 AND 是错误措辞；但120分钟 loss 的梯度只能回传30分钟，长 horizon 的差表现不能直接解释为没有长状态。
4. validation/test 会从合法 segment 起点按当前 checkpoint 重放，没有发现 stale-state 复用；5分钟只是每个 segment 开头不评分，不是每个 chunk 都丢数据。但5分钟不足以单独初始化120分钟通道。
5. 未来事件数高度过度离散：三位患者各 split/horizon 的 variance/mean 为 7.1–384.8，Poisson 明显不合适。
6. 旧聚合器曾根据 development-test 上是否靠近 fitted-intercept 决定某个 seed 是否进入分母。现在已改成所有有限分数都保留，审计只加 flag，不再删除；旧 filtered 数值只留作 deprecated provenance。

## 正式收口

v0.3.1 的状态固定为 `V0_3_1_PILOT_CLOSED_MAJOR_REVISION`。允许写：nested split、有效 exposure、冻结 grammar、连续时间扫描和 open-loop 评分已完成端到端联调；1146 有有限的时刻相关诊断信号。

不允许写：没有慢状态、H1/H2a 已被否定、当前 fixed-τ 对应真实生理尺度、非零 state norm 证明学到了状态、或者 H2b/H3 已经可以开始解释。

已读取的80–100% development test 已参与架构审阅，今后不能再充当最终独立检验。正式/封存分区仍未打开。下一版的首要比较固定为 `H+S_correct` vs `H`、`H+S_correct` vs `H+S_shifted`、以及 dynamic `S` vs TRAIN mean `S`。

机器汇总：`/data/hfosp_group_event_state_v0_3/pilot/summary_v0_3_1_closeout.json`
