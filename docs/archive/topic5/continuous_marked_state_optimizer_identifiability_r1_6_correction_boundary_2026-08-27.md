# Continuous Marked State R1.6：旧结论更正边界

本文只规定哪些旧说法被 R1.6 覆盖，以及新的安全表述。旧产物保留作 provenance，不删除、不改写。

## 1. R1.5 epoch 0 / no-update

### 旧说法

“模型选择 epoch 0，因此没有可训练的 persistent state。”

### 更正

R1.5 的 epoch 0 已在完整 TRAIN refit，随后却拿 TRAIN 尾部作为 target-alignment 的 inner selection；epoch 0 已见过选择数据，比较不公平。旧 no-update 不再承担科学阴性，只保留为选择与优化失败的证据。

### 新边界

R1.6 用 TRAIN 内 0–60%、60–80%、80–100% 三段时间选择。六位患者 18/18 短段过拟合通过、30/30 最终训练轨迹均有训练改善；未见后段不改善者归为泛化失败或当前模型不可识别，而不是“没有状态”。

## 2. R1.5/H3-long 的 `ZERO_GRADIENT`

### 旧说法

“IED→state 边为零，所以 H3 没有作用。”

### 更正

旧张家齐 T1 的 `state_timing`、`state_contact`、`state_size` 权重全为零，H3 real edge 的 matrix 与 intercept 在零点的梯度均精确为 0。该边结构上无法更新，调整 H3 学习率也不能补救。

### 新边界

旧 `ZERO_GRADIENT` 全部降级为不可估计的仪器失败，不作为 H3 生物学阴性。只有来自选择安全、非零且稳定 T1 的 H3 重跑才有解释资格。

## 3. 张克轩的“稳定 persistent T1”

### 旧说法

“张克轩是 R1.5 唯一稳定且正确时刻特异的 persistent state 患者。”

### 更正

R1.6 公共配置的五 seed 结果显示：正确时刻 5/5 优于匹配错误时刻，中位 joint NLL 差 `−0.01278`；但 persistent 5/5 输给 memoryless，中位差 `+0.02507`。

### 新边界

撤回“稳定 persistent T1”。保留的证据是：张克轩存在具有时刻专属性的 observation-conditioned code，但没有证明跨窗口 carry 比每窗独立编码更好。

## 4. E384、程帅和陈子阳

- `epilepsiae_384`：3/5 stable checkpoints，两个独立确认 seeds 均稳定。新定级为 **development-level optimization-robust support**；这是唯一可进入最小 H3 的患者。
- `yuquan_chengshuai`：1/5 stable，独立确认 0/2。定级为 **optimizer-sensitive support**。
- `yuquan_chenziyang`：1/5 stable，独立确认 0/2，量级很小。定级为 **optimizer-sensitive support**。

不得把上述 seed 比例换算为患者阳性率，也不得把 E384 单例写成队列结论。

## 5. E1096 与张家齐

两人均为训练 5/5 改善、短段过拟合 3/3 通过，但完整 stable checkpoint 为 0/5。因此安全定级是：

> 当前模型和数据划分下的泛化失败或不可识别。

不得写成“患者不存在状态”，也不再追加小修小补的 optimizer 网格来寻找患者特异最好配置。

## 6. R1.6 最小 H3

E384 的 load 与 participation 共 6 个单元都没有通过 full-control 主判据。load 3/3 对所有控制均不利；participation 仅一个 seed 的逐事件平均胜过部分对照，未胜因果延迟/趋势，且独立时间块均不利。每个 seed 只有 2 个独立 validation 单元。

新的安全结论是：

> E384、N=1000 的最小 H3 未获得支持且效力有限；H3 仍未决。

不得写成“IED 不改变状态”，也不得外推到数千至上万次 IED 的更长累积尺度。

## 7. 不受本轮改写的部分

- 旧 event-history state swap / same-prefix continuation 作为独立 development predictive evidence 保留，但不等于 R1.6 raw/explicit persistent state 已在队列复现；
- H2b 的发作前探索性结果本轮未重跑，不增不减；
- formal test、sealed partition、seizure probe 与 paper-ready figures 未触碰。
