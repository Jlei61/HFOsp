# Group-Event State v0.3.1 审阅后正式收口

**日期：** 2026-09-02
**状态：** `V0_3_1_PILOT_CLOSED_MAJOR_REVISION`
**安全结论：** `instrument complete, state learning unresolved`

## 1. 这轮到底完成了什么

三位预定义患者、每位三个 seed 的 development pilot 已完成。nested physical-time split、有效记录 exposure、冻结 contact grammar、连续时间状态扫描、checkpoint-specific replay 和 open-loop 评分均能端到端运行。正式/封存分区没有打开。

这轮没有完成一次可以裁定 H1/H2a 的 state-learning 实验。旧实验比较的是候选状态 `S` 单独与显式历史 `H`，而真正需要的 residual 比较是：

1. `H+S_correct` vs `H`；
2. `H+S_correct` vs `H+S_shifted`；
3. `H+S_dynamic` vs `H+mean(S_train)`。

三项在 v0.3.1 中都没有运行。因此不能把旧结果写成“没有慢状态”或“H1/H2a 阴性”。

## 2. 三位患者的正确读法

- `epilepsiae_1146`：正确时刻状态相对 shifted state 在 5/30 分钟有利，但状态单独仍未超过显式历史，三个 seed 又都在预算末端。它最多说明轨迹含有有限的时刻相关信息，不能证明存在 `H` 之外的 residual state。
- `yuquan_pengzihang`：三个 seed 都选择第一个训练 epoch，120 分钟没有可评分 anchor。当前合同没有产生有效更新，不能计作 state 阴性。
- `yuquan_zhangkexuan`：三个 seed 都选择第一个训练 epoch，correct 与 shifted 几乎相同，120 分钟严重失校准。更像训练塌缩和 count 模型失配，不能计作生物学阴性。

旧 mark contrast 也不构成 H2a 阴性：两位患者几乎停在初始解，另一位又缺少承重的 `H+S` 配对。

## 3. 本次已经修正的承重问题

### 3.1 纠正主 estimand 和结论

机器汇总、白话报告、技术报告、图 payload 与图合同现已统一：v0.3.1 的旧 count/mark 数值只放在 `v0_3_1_diagnostics` 中，H1/H2a/H2b/H3 主接口全部保持 `not yet run`。空 panel 表示没有运行正确实验，不表示零效应。

### 3.2 development-test 不再决定统计分母

旧聚合器用 development-test 上是否接近 fitted-intercept 决定一个 seed 是否进入对比。该规则会让观察到的 test 表现反过来改变分母，现已删除：

- 所有有限的配对分数都进入汇总；
- post-hoc intercept audit 只保留 flag；
- 旧 filtered contrast 和旧 admissible count 明确标为 `deprecated`；
- 新增回归测试，保证 flag 不会再改变 scored-pair denominator。

### 3.3 修正实现审计的事实边界

- adapter 不是数学上的全零死区：末层投影为非零小随机初始化，gate 为 `sigmoid(-4)≈0.018`，已记录梯度也非零；但有效 logit 调制与 Jacobian 尚未测量，所以功能可训练性仍未决。
- TBPTT 实际在事件数或物理时间任一上限先达到时切块，并非旧报告所写的 AND；但 120 分钟目标只能回传 30 分钟梯度，仍存在 credit-assignment 不一致。
- validation/development-test 会用当前 checkpoint 从合法 segment 起点重新 replay，未发现 stale trajectory；5 分钟 burn-in 是 segment-level，不是每个 chunk 重置。
- 当前逐时刻 LayerNorm 会压掉状态幅度，自由 state-to-state mixing 又会改变 nominal time scale，因此 5/30/120/360 分钟标签不能解释为被数据识别出的生理时间常数。
- 对已有三患者、各 split/horizon 的 future-count 审计显示 variance/mean 为 7.1–384.8，Poisson 明显失配。

### 3.4 关闭被消费的独立检验身份

80–100% development-test 已用于本轮架构审阅，今后不得再作为最终独立检验。现有结果仍可用于 development 诊断，但不能通过改名恢复独立性。

## 4. 哪些问题没有在旧版本里“补丁式修掉”

以下属于新科学模型，而不是 v0.3.1 报告修辞，因此明确留给 v0.3.2；本轮不偷偷重训：

- 将显式 `H` 固定为基础模型，只学习 residual marked state；
- 用受约束的 12 维 leaky bank 取代自由 recurrent mixing；
- 去除 per-time LayerNorm，改用 TRAIN-only 固定标准化；
- 拆开 static grammar calibration、history adapter 与 dynamic state adapter；
- 以 30 分钟 negative-binomial count 为首个 state-learning 主目标；
- 延长或重构 120 分钟 credit assignment；
- 事前冻结 endpoint eligibility，不再按 solver/test 结果改变分母；
- 重建非 transductive contact vocabulary 后再进行 sealed evaluation；
- detector refractory、feature availability 与 event-window overlap 的完整测量审计；
- H2b frozen transfer 与 H3 feedback 模型比较。

这些未完成项不降低本次 closeout 的真实性，但禁止把“工程问题已列出”误写成“新模型已经修好”。

## 5. 旧产物如何使用

- 原始 checkpoint 和逐 seed 结果：保留，作为 v0.3.1 development provenance。
- `summary_main.json`：旧口径，冻结为历史产物，不再作为当前权威汇总。
- `summary_v0_3_1_closeout.json`：当前唯一机器汇总。
- 同日白话版和技术版：由修正后的 finalizer 重建，为当前文字口径。
- core-evidence figures：只展示最终科学接口；v0.3.1 的 H1/H2a 主 panel 保持 `not yet run`。

## 6. 最终验收

可以验收：v0.3.1 的工程联调、审阅后统计分母修复、实现事实审计、报告与图接口收口。

不能验收：residual predictive state、慢生理状态、H2a repertoire modulation、H2b seizure transfer 或 H3 IED feedback。

下一轮只能从 v0.3.2 新合同开始；不能原样扩 6 人，也不能继续把当前 development-test 当作独立证据。
