# Topic 5 有序间期历史与架构控制：综合验收

## 一句话结论

本轮验收状态为 **COMPLETE_LAYERED_ORDER_AND_CROSS_STATE_AUDIT**。当前 RNN 线检验的是单个间期群体事件内部的 rank-step history，而不是跨小时的发作倒计时。target-blind 选出的 `linear_state` 支持顺序增量，但 7 个预注册递归家族中只有这一个通过 family-wise inference，故跨架构稳定性和 early-ictal 条件增量均未建立。

## 1. 实际做了什么

1. 审计 34 人、864,163 个合格间期群体事件的绝对时间，以及 16 人 106 次 clinical-onset target 的配对关系。
2. 在完全不读取发作 target 的阶段，比较 static、unordered prefix、first-order、linear state、vanilla RNN、GRU 和预注册的 low-rank r0/r1/r2/r4。
3. 对最佳非 GRU 模型补做同架构 within-event rank shuffle，并对冻结状态执行 reverse、drop-first 和 rank 后 reset。
4. 模型和表示冻结后，才读取既有 `[0,10] s`、`1–150 Hz` early-ictal contact energy，检验 ordered field 是否在 static participation 与 unordered-prefix field 之外仍有增量。

## 2. 时间语义审计

- 现有模型每一步是同一个群体事件内的 recruitment rank set；每个事件开始时状态清零。
- 106 次发作只有 46 条不同的 causal pre-seizure histories，仅 6 位患者至少有 3 条不同历史。
- 因此 across-event history–seizure 分支没有被包装成 106 个独立样本；它不是本轮 primary。

## 3. 间期自监督架构结果

最佳非 GRU `linear_state` 相对 static、last-set 和 unordered-prefix 的患者中位 NLL 增益依次为 0.0832、0.0446 和 0.0257。其中相对 unordered-prefix 有 26/34 患者方向为正，名义 Wilcoxon p=5.31e-05；正式解释使用下表跨 7 个预注册递归家族的 maxT 校正 P。

相对同架构 rank-shuffle，其真实顺序增益为 0.0419，31/34 为正，名义 p=1.87e-07；由于该架构先按 unordered 对照被挑中，这一项是 selection-aware sensitivity，不作为独立确认。GRU 相对 unordered 的中位增益为 0.0106，相对 rank-shuffle 为 0.0306。

安全解释是：**事件内部真实顺序可被一个简单线性递归状态利用，而且不依赖 GRU 门控；但该增量尚未跨至少两个递归家族稳定复现。** 因此结果是 architecture-dependent 的序列证据，不是 hidden manifold 的独立脑相似性验证，也不能写成跨架构普遍规律。

| 架构 | 中位 NLL gain vs unordered | 正向患者 | 名义 P | maxT P |
|---|---:|---:|---:|---:|
| Linear state | 0.0257 | 26/34 | 5.31e-05 | 0.00032 |
| Vanilla RNN | 0.0008 | 17/34 | 0.426 | 0.887 |
| GRU | 0.0106 | 21/34 | 0.0563 | 0.183 |
| Low-rank r=0 | 0.0158 | 20/34 | 0.0213 | 0.052 |
| Low-rank r=1 | -0.0033 | 16/34 | 0.306 | 0.972 |
| Low-rank r=2 | 0.0138 | 25/34 | 0.00192 | 0.0836 |
| Low-rank r=4 | 0.0038 | 18/34 | 0.306 | 0.688 |

## 4. 容量公平性敏感性

为避免把较小参数量误当成较弱的架构，本轮另以 GRU(h=32) 的 11,246 个参数为参照，补跑 linear-state(h=64) 与 vanilla-RNN(h=48)；两者参数量均在参照的 10% 以内。该分析只检查固定 hidden-size 结果是否受模型容量驱动，不参与 target-blind 模型选择。

| 敏感性模型 | hidden | 参数量 | 中位 NLL gain vs unordered | 正向患者 | 名义 P | Holm P |
|---|---:|---:|---:|---:|---:|---:|
| linear_state_parammatched_h64 | 64 | 10158 | 0.0270 | 24/34 | 8.9e-05 | 0.000178 |
| vanilla_rnn_parammatched_h48 | 48 | 10318 | 0.0061 | 21/34 | 0.0647 | 0.0647 |

## 5. 有效历史深度与历史干预

前一轮冻结的同架构历史窗口实验已经给出历史深度上限：H2 相对 H1 的中位 NLL 增益为 0.0172（32/34），H3 相对 H2 为 0.0113（29/34），但 full history 相对 H3 为 -0.0010（P=0.436）。匹配的 ordered H3 相对 H3 rank-shuffle 增益为 0.0261（27/34，P=0.00361）。

因此当前数据支持的是最近 2–3 个 rank set 的 bounded short history，而不是无界 full-history memory。

已在 34 人、3 seeds 上比较 ordered、reverse prefix、drop earliest，以及在第 1/2/3 个 rank set 后 reset（代码索引 0/1/2）。所有 eligible contact mask 始终由完整真实 prefix 决定，因此干预只改变进入 recurrent state 的历史，不会把已出现触点错误放回候选集。

这些结果回答“模型是否真的使用该段历史”，不能解释为生物恢复时间常数。

| 干预 | 中位 NLL 代价 | 正向患者 | one-sided P |
|---|---:|---:|---:|
| reverse_prefix | 0.0357 | 32/34 | 5.82e-10 |
| drop_earliest | 0.0068 | 28/34 | 1.17e-05 |
| reset_after_rank_0 | 0.0068 | 28/34 | 1.17e-05 |
| reset_after_rank_1 | 0.0076 | 31/34 | 2.88e-06 |
| reset_after_rank_2 | 0.0205 | 27/34 | 1.06e-05 |

在 selected model 的真实 contact-logit readout 方向上，局部一步 retention 中位数为 0.5075，readout alignment 为 0.9997，局部 Jacobian spectral radius 为 0.5266。这些只是 rank-step 上的输出相关记忆诊断；不同架构的 hidden 坐标不可直接逐单元比较。

## 6. 低维结果的解释边界

既有 frozen hidden-state audit 中，ordered GRU 的 effective rank 中位数为 1.8784，但 rank-shuffle GRU 同样只有 1.3358。因此低维性不能单独支持“二维癫痫状态流形”；本轮把主要证据放在 heldout 顺序增量和显式历史干预上。

## 7. early-ictal 条件增量

在 static participation 与 unordered-prefix field 条件下，ordered field 的 absolute partial-r margin 患者中位数为 0.0346，n=12，p=0.515。

相对 matched rank-shuffle，ordered field 的 absolute partial-r 差值中位数为 0.0085，6/12 为正，p=0.604。

两项都没有通过冻结门，因此 **ordered-history 对 early-ictal field 的条件增量未建立**。该 target 已在前序工作中读取，故即使结果为阳性也只能称为 reused-target internal validation。输出仍是患者固定的 contact field，不是逐次发作预测器。

## 8. 与论文核心目标的关系

- **未偏移**：输入仍为原始 SEEG 简化后的 contact-rank event；主问题仍是间期有序传播信息是否与发作早期静态能量招募场共享。
- **主动收窄**：不再把低 effective dimension、本身的二维轨迹或 hidden PC 解释为真实脑流形。
- **与 SNN 分工清楚**：RNN 只负责数据驱动的 history-state identification；SNN 单独检验局部抑制/慢变量机制能否生成相似的状态转移。
- **未新开 IEI 主线**：绝对时间只用于因果配对和伪重复审计，不作为预测输入。

## 9. 图与产物

- Paper-ready candidate：`results/paper-ready-figure/fig6_ordered_history_architecture_audit/figures/fig6_ordered_history_architecture_audit.png`
- 机器验收：`results/topic5_ordered_history_architecture_audit/FINAL_ACCEPTANCE.json`
- 测试与独立复算审计：`results/topic5_ordered_history_architecture_audit/TEST_AUDIT.json`
- 架构表、干预表与 early-ictal 条件统计：`results/topic5_ordered_history_architecture_audit/analysis/`
- 四个正式阶段均保存逐 20 秒资源日志；最终验收 JSON 汇总其最低可用内存、峰值显存、GPU 利用率和温度。

## 10. 最终用语边界

允许写：

> 在 target-blind 选出的线性递归模型中，有序间期 recruitment history 对 heldout next-contact prediction 提供了超越静态结构、last-set 和无序前缀的增量；该证据具有架构依赖性，对 early-ictal contact field 的条件增量未建立。

禁止写：

> RNN 发现了二维癫痫脑状态流形、恢复了真实 E/I 回路，或学得了连续时间发作倒计时。

## 11. 审阅批注与下一步

- **信息控制通过，但结论分层**：linear-state 同时超过 static、last-set、unordered 和同架构 rank-shuffle，说明单个事件内部的 recruitment order 不是普通参与频率的重述；但 1/7 家族通过意味着它尚不是跨架构稳定规律。
- **架构结论不是“需要 GRU”**：参数匹配的 linear-state 仍阳性，vanilla RNN 和 GRU 均未通过相同的 family-wise 标准。当前最简、最诚实的表示是 bounded linear event-indexed state，而不是更深的门控网络或强制 low-rank/Dale 约束。
- **early-ictal 主桥没有得到新增支持**：冻结 ordered residual 在 static + unordered 之外不超过 contact shuffle，也不超过 matched rank-shuffle。论文中可以保留既有 static contact-field correspondence，但不能把本轮序列状态写成其新增解释来源。
- **没有偏到 IEI 或发作倒计时**：所有训练状态都在事件边界清零；绝对时间只用于因果配对与伪重复审计。现有 106 次发作只有 46 条不同 causal histories，不足以把 across-event 分支当作 106 个独立预测样本。
- **停止继续刷当前模型**：不再用更多 seeds、hidden size、low-rank rank 或 loss 权重追 early-ictal 阳性。只有独立 clinical-onset cohort，或每位患者至少三条真正不同且可因果配对的 pre-seizure histories，才值得重新开放跨事件/跨状态训练。
