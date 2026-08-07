# Topic 5 SPF-RNN v0.4：公平对照修复与六患者开发收口（2026-07-30）

合同：
`docs/superpowers/specs/2026-07-30-topic5-shared-propagation-field-rnn-v0_1.md`

Canonical 结果：
`results/topic5_shared_propagation_field/development/ladder_pilot_v0_4/`

> **2026-07-31 后续审阅**：likelihood 校准、nested learning curve、
> observable-residual stability、既有 SNN 只读辨识、latent-d sensitivity 和
> field-utilization 诊断已完成。Gate 最终状态与显式 `k_dir=2` 的 SNN 方向
> 重算见
> `docs/archive/topic5/shared_propagation_field_rnn_multiround_review_2026-07-31.md`。
> 本文保留 v0.4 初始收口时的历史快照。

## 1. 一句话结论

在六名 target-blind pilot 患者、三个 seed、八个公平对照全部训练充分后，
autonomous shared field（M4）虽然超过静态模型 6/6、超过普通 stationary
first-order Markov 5/6，但它没有超过 phase-matched Markov mixture（0/6）
或低维时间模板（0/6）。给 field 加相同进度时钟后也只在 2/6 超过时间模板。

因此旧比较中的“时钟不公平”已经排除，但 autonomous-field 主张仍未获得支持。
本线按 development bounded negative 收口，不进入 34 人全队列。

## 2. 本轮修复

### 2.1 公平信息集

旧阶梯只含 stationary Markov，而 M3 直接读取 `t/(T-1)`。本轮加入：

- M1-phase：带进度的一阶 Markov
- M2-phase：带进度的 mixture Markov
- M4-phase：带进度的非自主 field 诊断

M4-phase 不是主模型；它成功也不能支持 autonomous-field claim。

### 2.2 独立开发测试集

旧 train80 内按时间顺序分为 70% inner train、15% monitor validation 和
15% development test。monitor 选 checkpoint；development test 只在八个
模型全部选完后评分。旧 heldout20 未进入拟合、选择或评分。

### 2.3 训练充分性与可复现

- 每个模型使用与阶梯顺序无关的固定 seed。
- 最佳点过早标为 `EARLY_OPTIMUM_UNVERIFIED`，从同一初始化以 0.2 倍学习率
  自动复核。
- patience 只被达到 `1e-4` 相对量级的改善重置，末位小数抖动不再造成假未收敛。
- 每个 run 保存 model、最佳点 optimizer、两次尝试记录和 validation curve。
- 聚合器同时核对 config SHA 与 source SHA。

本轮共触发 5 次低学习率复核：M1 1 次、M4 3 次、M4-phase 1 次；复核后
144/144 fits 全部为 `CONVERGED` 或 M0 的 `NO_FREE_PARAMETERS`。

### 2.4 评价修复

- 主表改为 NLL / suffix decision。
- latent 模型用 4 次 importance-weighted marginal-likelihood estimate；
  full-event posterior 只作 importance proposal。
- 同时报告 4 次 future-blind prior-predictive NLL。
- 每个模型做 4 次独立自由 rollout，报告 repertoire 均值与 SD。
- 逐 step NLL 单独保存，用于检查优势是否只来自早期或末期。

## 3. 完整事件 likelihood

下表为三个 seed 先在患者内折叠后的 development-test NLL / suffix decision；
越低越好。

| patient | M0 | M1 | M1-phase | M2 | M2-phase | M3 | M4 | M4-phase |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epilepsiae_1096 | 1.2837 | 1.2278 | 1.2185 | 1.2174 | 1.2079 | **1.2019** | 1.2259 | 1.2275 |
| epilepsiae_620 | 1.5055 | 1.4144 | 1.4009 | 1.3914 | 1.3761 | **1.3421** | 1.3807 | 1.3560 |
| epilepsiae_922 | 1.3967 | 1.3681 | 1.2638 | 1.3292 | **1.2373** | 1.2384 | 1.3163 | 1.2374 |
| yuquan_chenziyang | 1.7877 | 1.3663 | 1.3481 | 1.3533 | **1.3282** | 1.3544 | 1.3796 | 1.3606 |
| yuquan_zhangjiaqi | 1.3021 | 1.2468 | 1.2393 | 1.2202 | 1.2020 | 1.1946 | 1.2190 | **1.1906** |
| yuquan_zhangkexuan | 2.6975 | 2.2914 | 2.2289 | 2.2514 | **2.1745** | 2.1817 | 2.2552 | 2.2205 |

患者内比较（左减右，负数表示左模型更好）：

| 对比 | 中位差 | 左模型更好 |
| --- | ---: | ---: |
| M4 − M0 | −0.1040 | 6/6 |
| M4 − M1 | −0.0307 | 5/6 |
| M4 − M1-phase | +0.0169 | 2/6 |
| M4 − M2 | +0.0013 | 3/6 |
| M4 − M2-phase | +0.0347 | 0/6 |
| M4 − M3 | +0.0319 | 0/6 |
| M4-phase − M3 | +0.0100 | 2/6 |

这说明：

1. 完整事件确实不是纯静态 participation。
2. 相对普通 stationary M1 的优势不能直接归因于自主 field，因为加入进度和
   离散路径后，M2-phase 在 6/6 患者上优于 M4。
3. M3 在 6/6 上优于 M4；M4-phase 也未稳定超过 M3，所以“只是少了一块时钟”
   不能再解释 autonomous M4 的全部失败。

## 4. 估计量与自由生成复核

pure prior-predictive sensitivity 与上表方向一致：

- M4 − M3：中位 +0.0314，M4 胜 0/6
- M4 − M2-phase：中位 +0.0369，M4 胜 0/6
- M4-phase − M3：中位 +0.0092，M4-phase 胜 2/6

latent primary NLL 的 MC SD 很小：

- M3：中位 0.00028，最大 0.00091
- M4：中位 0.00019，最大 0.00080
- M4-phase：中位 0.00021，最大 0.00189

因此主要差值不是单次 Monte Carlo 抖动。

四次自由 rollout 的 mean precedence correlation：

- M3：0.932
- M4：0.861
- M4-phase：0.915

M4 没有在第二个 endpoint 上形成与 likelihood 相反的稳定优势。少数单患者或
单 seed 生成较好，只能作异质性描述。

## 5. SNN 既有资产审计

按用户要求，本轮没有运行任何 SNN simulation。只读审计输出：

`results/topic5_shared_propagation_field/snn_positive_control/existing_artifact_reuse/`

| family | files/seeds | raw events | model-ready | 历史 reported direction |
| --- | ---: | ---: | ---: | --- |
| source-only | 20 | 246 | 180 | 98 forward / 0 reverse |
| sink-only | 20 | 237 | 230 | 0 forward / 110 reverse |
| paired source/sink | 21 | 222 | 222 | 103 forward / 119 reverse |

审计为每个 readout/figdata partner 和生成的 rank-event NPZ 保存 SHA。现有数据
已经包含 source/sink 方向翻转与 paired repertoire，无需重跑 SNN。

上表前两行是历史 payload 中保存的 reported sign；后续审阅发现 subject-SNN
runner 修改 imported globals 并不会改变已绑定的 Python 默认参数。2026-07-31
以保存 ranks/coordinates 和显式 `k_dir=2, eps=2 mm` 重算后，source-only 为
158/0/22（forward/reverse/unreadable），sink-only 为 0/217/13，paired
仍为 103/119/0。历史 reported sign 已单独保留，不再作为 canonical direction。

机制口径修正：

> 方向由低阈值 pathological kernel/core 位于哪一端产生；E→E 各向异性影响
> 传播通道，但去掉各向异性不应单独抹去方向。

所以旧 `AR=1` isotropic probe 不是 direction-erasure negative control。现有
本文冻结时的资产状态是 `INPUTS_READY / RNN_IDENTIFIABILITY_NOT_SCORED`，
不是 G0 PASS。2026-07-31 的后续复核认定该设计受 legacy pooling、未建立
`N_min` 和 first-rank lookup 捷径影响，因而从 RNN Gate 删除，不能据此判
G0 正或负。

## 6. Gate 判决

- 工程合同：PASS
- SNN artifact availability：PASS（只读输入已齐）
- G0 RNN identifiability：NOT SCORED（本文冻结时；后续改为
  `REMOVED_FROM_RNN_GATE_NOT_EVALUABLE_FROM_ROUND5`）
- G1 autonomous full-event generation：**development stop condition met**
- G2 stable observable structure：NOT RUN
- G3 one structure, many trajectories：NOT RUN
- 34 人正式扩展：STOP

没有必要再通过 mixture-of-fields、更多 GRU、低秩约束或发作 readout 挽救
autonomous-field 主张。事件数 learning curve 若将来运行，只能回答新的
sample-efficiency 问题，不能反转本次 full-data gate。

## 7. 允许与禁止表述

允许：

> 在六名 target-blind development patients 中，完整 rank-event 组织稳定超出
> 静态 participation 和普通 stationary first-order Markov；但在匹配事件进度
> 与离散路径后，一个自主共享 latent field 没有提供额外解释。

禁止：

- 已辨识患者特异稳定传播场
- one structure, many trajectories 已成立
- RNN 已恢复 SNN 连接或预测 contact lesion
- 时间模板是脑内真实机制
- 六人 development pilot 是 cohort-confirmed negative

## 8. 工程验收

- 相关测试：31 passed
- Python compile：通过
- 18/18 run states：`COMPLETE`
- 4 个 leakage flags：全部 false
- config SHA：`cfc376836276273e745e880d8e238b88529d707dcdb89a149f445891888632fc`
- cohort state：
  `results/topic5_shared_propagation_field/development/ladder_pilot_v0_4/LADDER_PILOT_STATE.json`

早期 v0.2 和中断的 v0.3 只保留为修复审计证据，不进入任何模型比较。
