# Continuous marked-state R1.3 / T2 长尺度总效应技术报告

> **2026-08-26 代码复审更正**：逐条更正见 `docs/archive/topic5/continuous_marked_state_t2_long_total_post_review_corrections_2026-08-26.md`，
> 事后补算的仪器诊断见 `t2_long_total_effect/reports/post_review_audit.json`。
> 主结论（张家齐不可检验）不变；§3 §4 §5 §6 §8 有更正段落。
> 本文所描述的模块版本已由 `t2_long_total_effect_decoder_space_v1` 升级为
> `..._v2_intercept_matched`；磁盘上 6 个人体结果仍是 v1（结论不变，未重跑 GPU）。

## 1. 冻结范围与科学定位

- 上一阶段 R1.3 revision：`r1_3_full_raw_temporal_exact_target_isolated_increment_v2`。
- 本阶段 revision：`t2_long_total_effect_decoder_space_v1`。
- 预注册合同：`docs/archive/topic5/continuous_marked_state_t2_long_total_effect_contract_2026-08-26.md`。
- 患者：`yuquan_zhangjiaqi`；三 seed；两个固定尺度；正式 test 分区未打开。
- 科学对象：长期 `IED exposure sequence -> future state` total-effect candidate；不再是窗口末端条件于当前 state 的 one-step residual edge。

上一阶段的 H1/H2a 结果保持独立：三位 formal R1.3 患者 persistent-memoryless 3/3 有利，strict matched wrong-time 2/3 有利，first subset 和 continuation 均 3/3 有利。当前阶段只扩展 H3，不重估这些分母。

## 2. 张家齐同合同 T1

### 2.1 数据与初始化

- R1.2 full design：2,318 个 observation anchors，28,965 个 TRAIN events，9,655 个 validation events；TRAIN 15.05 h，validation 4.52 h。
- 新 R1.3 cache 独立冻结 `explicit_normalised.npy` 和 `contact_mask.npy`；所有 anchor 可重读，design hash 与 R1.2 manifest 一致。
- common state/generator/readout 从 audited R1.2 explicit seed-0 checkpoint 初始化；三 R1.3 seed 使用相同起点，再执行相同 exact timing + tied-group mark target alignment。
- 训练预算与 formal explicit R1.3 一致：observer alignment 2 epoch + joint alignment 2 epoch；stable generator 冻结。

### 2.2 选择结果

三个 seed 的 deterministic explicit 轨迹逐项相同：

| total epoch | inner-TRAIN chronological validation joint NLL |
|---:|---:|
| 0 | 10.2100695 |
| 1 | 10.2101176 |
| 2 | 10.2103741 |
| 3 | 10.2111807 |
| 4 | 10.2118964 |

3/3 选择 epoch 0。关键 gradient max 非零：state readout 0.4328、spatial fusion 0.02856、observation correction 0.02689。故 no-update 不是断梯度、OOM 或脚本提前退出，而是当前更新没有通过 TRAIN 内 chronological selection。

选择 epoch 0 后，persistent-memoryless 与 strict correct-wrong 的 joint/subset contrast 均精确为 0；所有 state-to-event readout 保持零，decoder rank = 0。

**更正（比原文更强）**：不只是读出为零。检查 checkpoint 得到
`state.generator.{omega_raw, q_raw, mu}` 与三个 readout 的
`max|θ − θ_init|` 全部为 0，即**整个状态模型停在构造函数默认值**；上游 R1.2 的
`selected_epochs` 同样是 0。因此本文 §3 中的「冻结 T1 generator `K`」指的是一个
**未经训练的默认演化**，不是拟合出来的流。这也意味着本轮启动前，磁盘上已有的
R1.2 结果就已经预示了高风险。

## 3. 长 exposure 算子

每次事件输入二维向量：

```text
x_j = [1, load_j - E(load_j | pre-event state_j, deterministic history_j)]
```

第二维只用 TRAIN 拟合条件均值并以 TRAIN residual SD 标准化。冻结 T1 generator `K`。对于窗口 `[s,e)`，长期 jump 的终点总贡献是：

```text
Delta z_e = sum_{j=s}^{e-1} exp(K * (t_e-t_j)) B x_j
```

实现没有为每个窗口做 10,000 步重复 rollout，而是在每个连续 coverage segment 内建立一次递推 prefix operator；任一窗口通过“终点 prefix 减去起点 prefix 的自然传播”精确得到。单元测试逐元素验证该算子与显式逐事件 rollout 等价。

`B` 为 `2 x 8`，共 16 个参数。

**更正 A（免费截距）**：occurrence 那一列是 `sum_j exp(K (t_e - t_j))`，即按时间
加权的事件计数。当窗口远长于 `K` 的时间常数时它饱和，跨窗口几乎恒定——张家齐
validation 上的相对波动只有 0.115（N=10000）/ 0.161（约 6 h）。于是 real 与
delayed 各自获得一个 no-edge 没有的 8 维自由状态截距。用同一套代码在
**完全无暴露信息**的目标上验证：常数偏移目标给出 `real − no_edge = −445.5`，
线性漂移目标给出 `−31.4`。因此 `real_minus_no_edge` / `delayed_minus_no_edge`
已降级为伪迹量；主对比改为 `real_minus_intercept_matched` 与
`real_minus_causal_delayed`（后者本来就同时参数匹配与截距匹配，
两臂共用完全相同的 occurrence block，单元测试已锁）。

**更正 B（实际时间尺度）**：`K = -(softplus(-4) + 1/2880) I`，时间常数 54.06 分钟
（= 构造函数默认值）。因此两个尺度实测：

| 尺度 | 名义事件数 | 有效加权事件数 | 50% 权重 | 90% 权重 | >1h 权重占比 |
|---|---:|---:|---:|---:|---:|
| N=10000 | 10,000 | 2,409 | 0.50 h | 1.59 h | 0.23 |
| 约 6 h | 9,848 | 2,285 | 0.49 h | 1.61 h | 0.23 |

两个「独立尺度」在数值上是同一台仪器。禁止再把它写成「一万次 / 六小时总效应」。

拟合目标不是 latent norm，而是冻结 decoder 的四个等权块：timing、STOP、non-zero size、contact/subset。各块只用 TRAIN target variation 定标。ridge 只在 TRAIN chronological 80/20 中从 `{1e-4, 1e-2, 1, 100}` 选择，再在全部 TRAIN 窗重拟合。

反事实保留所有 occurrence，只把 load innovation 因果延迟 1,000 次；不使用未来 exposure，参数量与真实臂完全相同。

## 4. Synthetic recovery

26,000 个事件的模拟中，N=10,000 的中位物理长度为 6.45 h，TRAIN/validation 窗口分别为 7,000/8,000。

- mixed true edge：real 同时胜 no-edge 与 delayed，`cos(B_true, B_fit)=0.999982`；
- occurrence-only truth：real 与 delayed 均胜 no-edge，符合预注册解释规则；
- null truth：real-no-edge decoder total contrast `+0.000317`，未制造方向有利结果；
- 五项 synthetic acceptance 全部通过。

**更正（v1 acceptance 抓不到主要失效模式）**：上述 null 是白噪声目标，
其 `target_delta` 均值为 0，因此免费截距无从发挥；判据 `|real − no_edge| < 0.02`
在构造上不可能失败。v2 已新增两个 exposure-free 但有均值 / 有漂移的零真值场景，
判据改为单侧（真实臂不得赢过截距对照臂），并加一条坏数据回归
（`offset_null_reproduces_intercept_artefact`：若那个 −445 的伪迹消失，说明算子
变了，降级理由必须重新推导）。v2 合成 **10/10 通过**，关键数字：

| 场景 | real − intercept | real − delayed | real − no_edge |
|---|---:|---:|---:|
| mixed_true_edge | −3.74 | −2.39 | −16884.8 |
| occurrence_only | −3.69 | +0.004 | −51091.4 |
| null（白噪声） | +0.00002 | +0.0003 | +0.0003 |
| null_with_state_offset | +0.373 | +0.015 | **−445.5** |
| null_with_slow_drift | +0.451 | +0.305 | **−31.4** |

该合成只证明仪器能区分已知真值，不提供人体 H3 证据。

## 5. 人体窗口与结果

| 尺度 | TRAIN windows | validation windows | validation next-event pairs | median hours | median events |
|---|---:|---:|---:|---:|---:|
| fixed N=10,000 | 11,905 | 4,715 | 4,714 | 5.9275 | 10,000 |
| fixed ~6 h | 7,922 | 5,991 | 5,990 | 5.9996 | 9,849 |

全部窗口位于同一 recorded coverage segment；跨 gap 数为 0（v1 里这是写死的字面量，
v2 改为由 `count_windows_crossing_segment` 真算，复审时对冻结产物补算结果同为 0）。
validation endpoint 全部落在 `[train_end, dev_end)`，历史只向过去延伸。

**更正（窗口数不是样本量）**：窗口逐事件滑动，相邻两个只差一个事件。该患者只有
两段 recorded coverage（10.68 h / 22,905 events，全 TRAIN；中断 2.37 h；
8.89 h / 15,715 events，含 6,060 TRAIN + 9,655 validation）。N=10000 要求同段内
≥11,001 events，因此 4,715 个 validation 窗口全部落在第二段最后 1.63 h。

| 尺度 | validation windows | endpoint span | effective independent windows |
|---|---:|---:|---:|
| N=10000 | 4,715 | 1.632 h | **1.81** |
| 约 6 h | 5,991 | 2.160 h | **2.40** |

TRAIN 侧分别为 7.09 / 4.81。用于选 ridge 的 inner-validation 是 TRAIN endpoint 的
后 20%，约 1.28 h ≈ 1.4 个记忆核长——它不是一个真正的 hold-out。
因此「数据量足够」不成立：事件数足够，独立窗口严重不足。

三 seed、两个尺度的 decoder rank 均为 0。因此每个 arm 的预测状态、decoder-space primary、latent sensitivity 与 exact next-event secondary 都由结构保证相同：

| contrast | N=10,000 | ~6 h |
|---|---:|---:|
| real - no-edge decoder total | 0.0000 | 0.0000 |
| real - delayed decoder total | 0.0000 | 0.0000 |
| real - no-edge next-event joint | 0.0000 | 0.0000 |
| real - no-edge timing / mark / STOP / first subset / continuation | 全部 0 | 全部 0 |

这些不是估计得到的生物学零效应，而是上游 T1 no-update 造成的 structural zero。机器汇总状态为 `UNTESTABLE_T1_INSTRUMENT_DEGENERATE`，`admissible_seeds=0`。三 seed 只是实现稳定性检查，不能当作三位生物学重复。

## 6. 与短尺度 T2-S1 的关系

- 先前 N=1000（**更正**）：620 与 958 的 `real_cumulative` 与 `state_matched_placebo`
  两条边在全部 12 个人体拟合里 `selected_epoch` 均为 0，边向量保持零初始化，
  因此 `real_minus_no_edge` 与 `real_minus_state_matched_placebo` **恒等于 0.0**。
  判据是 `< 0`，恒零永远不可能满足——原报告的「0/2」是构造保证的，不是估计出来的。
  正确写法是分母 0/0：暴露边在任何被检查的 epoch 都没有改善 TRAIN 内选择集。
  同轮 `current_event_only` 臂选到 epoch 5–15 且对比非零，证明机器本身没坏。
  聚合器已加 `n_structural_zero` / `n_estimated`，新汇总为
  `n_structural_zero=2, n_estimated=0, n_favourable_negative=0`。
- 当前 N=10,000 / 6 h：设计上改为从共同起点 rollout 到 observation-informed target state，能够测 total-effect candidate；但唯一高事件量患者没有有效 T1，因此未产生可解释的人体比较。

二者合起来只允许写（**更正措辞**）：当前没有人体 H3 支持；短尺度是「暴露边未被
拟合出来」的 0/0 而不是阴性，长尺度是 T1 退化导致的不可检验，两者都不能排除
IED 通过更长时间或当前 state 中介产生作用。

## 7. 工程验收

- T1：3/3 COMPLETE；长人体：2 scales x 3 seeds = 6/6 COMPLETE；synthetic COMPLETE。
- 相关测试：复审前 77 passed；复审后 **87 passed**（新增 10 项：约 6 h 窗口路径、
  跨段计数、delayed 与 real 共用 occurrence block、截距臂吸收无暴露偏移、
  两种退化读出不得过闸、单块触底不得误封整轮、记忆核审计、独立窗口审计、placebo 排除覆盖 validation），
  12 个已知 Transformer nested-tensor warning；无失败。
- log 扫描：无 traceback、OOM、split violation 或 sealed flag true。
- formal test opened：false；paper-ready figures modified：false。
- 长作业由可重入 queue 执行；完成结果采用原子 JSON/NPZ，重复运行不会混 seed/checkpoint。

## 8. 科学验收与下一步

本阶段工程完成度高；人体 H3 不验收为阴性，状态为“上游 T1 不可用导致未决”。
不建议通过增加 epoch、挑 seed 或增加 exposure 网格来追逐阳性。

**更正**：synthetic 仪器在 v1 判据下的「可验收」结论过强——v1 的 null 抓不到免费
截距这一主要假阳性方式。v2 补齐后 10/10 通过，可验收的对象是
`real_minus_intercept_matched` 与 `real_minus_causal_delayed` 这两个对比，
不包括 `real_minus_no_edge`。

下一最小实验是把 total-effect 合同迁移到已有非退化 formal R1.3 T1 的 620 和 958，
但**不是「完全相同」的合同**：主对比换成截距对照臂；先测这两位患者自己的生成器
时间常数再定窗口（实测：620 完全停在初始值 54.06 分钟，958 全部模态 53.5–55.1 分钟、黄瀚文 54.9–55.0 分钟，
即三位患者都在同一量级）；端点跨度与有效独立窗口数必须事前报出。
另注：`t2_s1_long_scale` 的 12 个人体结果早于 placebo donor 排除修复，
重跑前不得引用 placebo 对比。这样先回答“有效 candidate state 上的数小时、数千事件总效应”，再决定是否寻找更多同时满足有效 T1 与 N=10,000 支持的患者。H2b/H3b 与 seizure probe 在此之前继续保持关闭。
