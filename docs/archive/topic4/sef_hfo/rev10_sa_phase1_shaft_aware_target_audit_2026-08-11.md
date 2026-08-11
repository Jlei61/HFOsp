# Topic 4 rev10-SA 第一阶段审计：shaft-aware target 与历史重评分

日期：2026-08-11

# 审阅结论

## 1. 一句话判断

rev10-SA 第一阶段已经回答了“旧目标是否看不见 SCL”这个问题：答案是**是**。新的目标能稳定识别 SCL 删除、跨杆 timing 改变和 0/4 到 4/4 的连续恢复；但现有历史产物还不能证明“旧 objective 选错了某个已有 field”，因为真正的 48 个 field-fit 候选没有保存逐事件固定 contact 数据。

当前可继续进入 SA5 contact detectability。不能直接进入 field optimizer、Edge、`beta` 或 topology 扩展。

## 2. 完成程度

> **rev10-SA 第一阶段完成度：85/100；完整 dual-shaft recovery 完成度：约 40/100**

已完成：

- SA0 的 15-contact/shaft 合同、55/6/44 无序 pair 类和哈希；
- patient-training-only 的 fixed-contact timing 与 ordinal-compatible 双目标；
- 分杆 recruitment、四状态 precedence、分杆 profile 和 event cloud；
- 四个零仿真正控；
- patient direction 与 event extent 的 factorization 审计；
- rev8.1 final、rev9 factorial、hand dual-core、filament、L2/L3 retained artifact 的统一重评分。

未完成：

- SA5 虚拟电极可检测性；
- SA6 fixed-budget SCL relocation 与 dual-shaft capacity；
- 新 field 拟合、独立 selection network、重新 Edge factorization；
- 新 patient blind unit。

## 3. P0 / P1 关键问题

### P0：低 AMI 不能被写成新 target “通过”

新的 shaft-aware consensus KMeans 很稳定，但与旧 A/B 的 AMI 只有 `0.0112`。预设 stop 确实触发。后续附加审计显示，它主要按事件招募范围分组，而不是重现传播方向。

处理：旧 A/B 继续作为方向模式；shaft recruitment extent 和 event cloud 作为每个方向模式内部的连续因子。这个修改属于 patient-training-only 的探索性 amendment，不是盲验证。

### P1：旧 field-selection miss 无法回顾性证明

48 个 rev8.1 field-fit history 候选只保留了聚合分数，没有逐事件 contact identity。不能从 mean curve 反推 SCL recruitment 或 cross-shaft precedence。

处理：状态固定为 `OLD_OBJECTIVE_FIELD_SELECTION_MISS_NOT_TESTABLE`。下一步用 matched relocation 直接测试容量，不再从缺失 artifact 推断。

### P1：Null/Edge 的高 SCL 招募不是患者模式支持

Null 与 Edge 的 SCL recruited-contact fraction 为 `0.577/0.542`，但 OOD 为 `92.3%/83.3%`。它们只能说明均匀网络偶尔会出现 SCL 活动，不能说明恢复患者 A/B。

处理：图中用斜线灰柱标记 `NOT_EVALUABLE`，不参与患者模式排名。

## 4. 科学性结果

### 4.1 新目标是否真的看到两杆

四个正控全部按预期响应：

- 删除全部 SCL 显著恶化 SCL recruitment、SCL-SCL/ICL-SCL precedence 和 event cloud；
- 只平移跨杆 timing 时，杆内 precedence 基本不变，跨杆项显著改变；
- 0/4、1/4、2/4、3/4、4/4 恢复 SCL 时，recruitment、cross precedence 和总分连续改善；
- shared-axis 坐标塌缩不改变 fixed-contact/shaft feature。

因此此前担心的“单杆 contact 数量更多，优化器靠 ICL 内 rank 得到好分”已经被目标层修复。它过去确实可能发生；新目标不会再让缺失 SCL 以 NaN 方式消失。

### 4.2 患者模式应如何定义

shaft-aware K=2 对 event extent 的预测 AUC 为 `0.9519`，对旧 A/B 只有 `0.6181`。旧方向 × 新 extent 的四格均有大量事件：

```text
5082 / 4193
8668 / 12106
```

在 `recording block × ICL count × SCL count` 内严格配平后，每次每模式 `7590` 个事件，旧 A/B 的三类 precedence 都高于置换 null。方向和范围是两个因子，不应强迫一个平面 KMeans 同时表达。

### 4.3 历史模型重新评分

| 历史 family | multishaft fraction | SCL recruited-contact fraction | patient mode status |
|---|---:|---:|---|
| rev8.1 final | 0.000 | 0.000 | development-only |
| rev9 Node | 0.000 | 0.000 | development-only |
| rev9 Node+Edge | 0.000 | 0.000 | development-only |
| hand dual-core | 0.077 | 0.019 | development-only |
| Stage 2 filament | 0.022 | 0.011 | development-only |
| rev9 Edge | 0.889 | 0.542 | not evaluable, OOD 0.833 |
| rev9 Null | 0.923 | 0.577 | not evaluable, OOD 0.923 |

L2 的 64 个候选和 L3 的 57 个候选全部为零 SCL recruitment。旧目标与新 weakest-mode score 的 Spearman 相关为：

```text
L2: rho=0.364, n=62, p=0.0037
L3: rho=0.129, n=57, p=0.338
```

说明旧目标与新问题不等价；但由于 fit-history 缺事件级数据，不能声称已经找到“被旧目标错过的现成 field”。

## 5. 工程性结果

- FULL_TIMING 和 ORDINAL_COMPATIBLE 使用不同 patient embedding/floors，禁止比较绝对分数；
- L2 `768` 行、L3 `2052` 行 retained envelope 的重新提取 rank 与保存 rank 完全一致；
- missing SCL 保留为四状态 precedence 的“未共同招募”，不删除 pair；
- 当前定向单元测试 `9/9` 通过；
- SA4 使用 `systemd-run --user -> nohup`，独立日志、退出码、完成标记和桌面通知，修正版退出码为 0。

当前 artifact provenance 仍显示 runtime dirty，因为 producer 与配置尚未提交。正式 SA5 启动前必须先提交并用 clean commit 重建必要 sidecar。

## 6. 图

### Shaft-aware 正控

![rev10-SA shaft-aware positive controls](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/data_driven_core_field_rev10_sa/shaft_aware_target/figures/rev10_sa_shaft_aware_positive_controls.png)

### Direction 与 event extent 分解

![rev10-SA direction extent factorization](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/data_driven_core_field_rev10_sa/shaft_aware_target/figures/rev10_sa_direction_extent_factorization.png)

### 历史 artifact 重评分

![rev10-SA historical artifact rescore](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/data_driven_core_field_rev10_sa/shaft_aware_target/figures/rev10_sa_historical_artifact_rescore.png)

## 7. 最小修改路线

1. 提交并冻结 SA0-SA5 的代码、配置和文档，重建 clean provenance。
2. 运行六张网络的 SA5，同一网络内给 15 个 contact 注入同半径、同数量 exact spike packet。
3. 若 SCL 神经响应正常但 current readout 低，先修 observation；若两者都正常，进入 SA6。
4. SA6 先跑 component-3 SCL relocation、matched off-shaft 和小型 mass/width grid，不开 optimizer。
5. 只有 SA6 找到 dual-shaft feasible region，才做低维 field allocation fit；`beta` 和 Edge 继续关闭。

## 8. 当前状态

```text
OBJECTIVE_SHAFT_BLINDNESS_CONFIRMED
/
DIRECTION_AND_EXTENT_FACTORS_BOTH_SUPPORTED_EXPLORATORY
/
FROZEN_LEARNED_NODE_FIELDS_HAVE_ZERO_SCL_SUPPORT
/
OLD_OBJECTIVE_FIELD_SELECTION_MISS_NOT_TESTABLE
/
VIRTUAL_CONTACT_OBSERVATION_PENDING
/
DUAL_SHAFT_FIELD_CAPACITY_PENDING
```

这条线现在最核心的问题已经从“换优化器是否能救 mode A”转为两个可分辨的机制问题：SCL 是否在 observation 层可读，以及固定预算 K=3 field 是否存在一个能同时支持 ICL/SCL 的区域。
