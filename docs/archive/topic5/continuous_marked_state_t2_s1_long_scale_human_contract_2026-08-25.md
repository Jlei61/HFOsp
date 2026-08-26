# T2-S1 长尺度人体实验冻结合同

**冻结日期：** 2026-08-25
**冻结对象：** 人体 development one-step 实验；合成标定已独立完成，不参与人体参数选择。
**结果根：** `results/epi_prssm/continuous_marked_state/r1/t2_s1_long_scale/`

## 1. 科学问题

> 在已经估计出的下一事件前状态以及固定 event-history 之外，过去大量 IED 的异常负荷累积，
> 是否仍然能改善下一场 IED 的发生时间和完整空间表达预测？

这是 H3 的最小残余边筛查，不是递归生成器机制证明。它从同一个 pre-event T1 state 出发，
只在当前事件结束后施加一次候选 exposure edge，再开放环预测下一事件。

## 2. 为什么修改尺度

H3 可能不是单次或百次 IED 的效应，而是上千乃至上万次的微小累积。冻结三个事件尺度：

- `N=100`：短尺度参照；
- `N=1000`：本轮主要长尺度；
- `N=10000`：目标长尺度，但只有已有同合同 T1 checkpoint 且单个无缺口记录段历史完整时运行。

事件数不等于固定物理时间。严格审计中，`N=1000` 的历史中位时长为：620 约 3.28 h，
958 约 1.22 h；同一个 N 在不同患者代表不同小时数，必须并排报告。

## 3. 可观测性与固定患者

历史不能跨未记录区间。固定 R1.3 三人中：

- 620：`N=1000` 有 779 个 validation 候选下一事件；
- 958：`N=1000` 有 1,426 个 validation 候选下一事件；
- 黄瀚文：单段最多 285 个既往事件，`N=1000` 不可测；
- 三人均不能测试 `N=10000`。

因此本轮人体 T2-S1 固定为 620 与 958、seeds 0/1/2、`N in {100,1000}`，共 12 fits。
这不是按结果挑患者，而是在人类结果打开前按记录支持和 R1.3 checkpoint 可用性确定。
张家齐是六人池中唯一可测试 `N=10000` 的患者（5,715 个 validation 候选下一事件），
但在其具备同合同 target-trained T1 前暂缓；缺少 checkpoint 不作为 H3 阴性。

## 4. 冻结 T1

- 每个 subject/seed 使用其 formal R1.3 `explicit` checkpoint；
- raw 臂结果不 gate T2，避免 raw 普通阴性阻断 H3；
- T1 generator、observation correction、timing/mark readout 全部冻结；
- 当前事件使用严格 pre-event `z(t_e^-)`；
- 所有 candidate edge 从同一 T1 state 出发，不让任一臂重估 T1。

## 5. Exposure

先仅用 scalar signed load innovation，避免在人体结果前选择高维 participation 表示：

1. TRAIN 内以 pre-event state 和固定 history 岭回归当前 event load；
2. 残差除以 TRAIN residual SD，得到 signed innovation；
3. 同一无缺口记录段内累积最近 N 个 innovation，并除以 `sqrt(N)`；
4. validation load 不参与期望模型拟合；
5. participation exposure 作为独立后续 secondary，不由 load 结果决定是否执行。

## 6. 四个等支持臂

1. `no_edge`：候选边固定为零；
2. `real_cumulative`：真实最近 N 次 signed load innovation；
3. `state_matched_placebo`：从 TRAIN donor 池选 pre-event state/history 最接近但不在
   本地正负 N 事件邻域的累积 exposure；validation donor 只能来自 TRAIN；
4. `current_event_only`：只使用当前事件的 signed innovation。

四臂必须逐元素共享 current/next event、quadrature、history 和 mark support。

## 7. 模型与评估

候选边只有一个 8 维 signed vector：

`z(t_e^+) = z(t_e^-) + B_x x_e`。

之后冻结 observation correction，用原 T1 generator 流到下一事件以及四点 quadrature node，
计算 exact survival likelihood 和完整 tied-group sequential mark likelihood。

- TRAIN 尾部 20% 选择 edge epoch 0–15，再在完整 TRAIN refit；
- primary endpoint：下一事件 joint timing + full mark NLL；
- secondary：timing、selecting size、STOP、first subset、later continuation；
- 先 seed 内完成，再患者内取 seed 中位数；两位患者不称队列证据；
- `real - no_edge` 与 `real - state_matched_placebo` 都报告；current-event 是尺度定位，
  不是 AND gate。

## 8. 允许结论

- real 胜 no-edge 且胜 state-matched：累积 IED exposure 在当前 T1 state/history 外含
  下一事件增量，支持开发级 H3a；
- real 只改善 STOP/size：支持 extent/termination 方向，不称 network route；
- real 改善 subset/continuation：才接近 repertoire-shaping 方向；
- current-event 不胜而 N=1000 胜：支持累积而非单 IED 尺度；
- N=100 阴性、N=1000 阳性：是预注册尺度分离；
- 两尺度均阴性：只说明本仪器与这两位患者未见增量，不能排除 `N=10000`；
- 任何阳性仍是预测性残余边，不称因果生理机制。

## 9. 工程无效条件

仅以下情况使对应 fit 重跑：正式分区打开、checkpoint/hash 不一致、历史跨 recorded gap、
validation outcome 进入 load expectation、四臂 support 不同、NaN/Inf 或合成真值不能恢复。
普通阴性和患者异质性不阻断其他探索。

## 10. `N=10000` 后续合同方向

本轮 one-step likelihood 是最小仪器，不足以单独代表“长期塑形”。后续万次尺度只在张家齐
等具备完整连续历史的患者上运行，并先训练与 R1.3 相同合同的 target-trained T1。主结局改为
未来 observation 更新后状态相对冻结 T1 自然 flow 的偏移；下一事件 likelihood 降为辅助结局。
同时报告 `N` 和实际小时数，不把 event-count 窗口解释成统一生理时间常数。

本轮 one-step 还属于 **conditional residual-edge screen**：它先条件于当前
`z(t_e^-)`，再问历史 exposure 是否仍有增量。如果长期 IED 效应已经通过背景 observation
进入当前状态，这个条件化会控制掉 H3 的中介路径。因此本轮零结果不能检验 IED 对当前状态的
总累积效应。万次实验必须把起点移到累积窗口之前：固定共同起始状态，分别沿真实 exposure
序列、no-edge 自然 flow 和反事实 exposure 序列推进，再预测窗口末端由未来背景 observation
估计的状态；不能继续只在窗口末端给当前状态加一次汇总 exposure。主误差在冻结 T1 decoder
的 timing、STOP/size、subset 和 continuation 读出空间计算，raw latent norm 仅作敏感性，
避免把任意 latent 坐标尺度误当作生理状态改变。

## 11. 2026-08-26 长尺度解释补充（不追溯修改本轮结果）

用户提出 H3 可能要累积数千至上万次 IED 才形成微小状态改变。该方向进入下一阶段主实验，
但可观测性审计同时表明：张家齐的 `N=10000` 历史中位仅约 6.04 小时，而黄瀚文的
`N=100` 已约 8.49 小时。因此“事件数很多”和“物理时间很长”不是同一个尺度。

下一阶段只增加一个正交辅助臂，不扩成时间尺度大网格：

- 主臂：最近 `N=10000` 次 signed innovation；
- 辅助臂：最近约 6 小时内的 signed innovation，使用同一预处理、归一化、状态起点与结局；
- 两臂都从累积窗开始前的共同状态出发，以窗口末端 observation-inferred state 为直接目标，
  比较真实 exposure 序列、no-edge natural flow 与反事实 exposure 序列；主评分使用冻结
  decoder 的 timing/mark 读出，latent norm 只作敏感性；
- 若只有固定事件数臂有利，倾向 event-count accumulation；若只有固定时间臂有利，倾向
  physical-time integration；二者都只作 development 级预测解释。

这一补充不改变已经运行的 `N in {100,1000}` one-step 合同，也不把本轮普通阴性升级为
生物学排除。
