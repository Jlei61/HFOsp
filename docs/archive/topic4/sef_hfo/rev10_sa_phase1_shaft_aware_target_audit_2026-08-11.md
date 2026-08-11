# Topic 4 rev10-SA 第一阶段审计：shaft-aware target 与历史重评分

日期：2026-08-11

# 审阅结论

## 1. 一句话判断

rev10-SA 第一阶段已经回答了“旧目标是否看不见 SCL”这个问题：答案是**是**。新的目标能稳定识别 SCL 删除、跨杆 timing 改变和 0/4 到 4/4 的连续恢复；但现有历史产物还不能证明“旧 objective 选错了某个已有 field”，因为真正的 48 个 field-fit 候选没有保存逐事件固定 contact 数据。

SA5 contact detectability 已经完成并排除 SCL observation 主限制。当前可继续进入 SA6 fixed-budget dual-shaft field canary；仍不能直接进入 field optimizer、Edge、`beta` 或 topology 扩展。

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
- 当前 SA0-SA6 定向单元测试 `19/19` 通过；
- SA4 使用 `systemd-run --user -> nohup`，独立日志、退出码、完成标记和桌面通知，修正版退出码为 0。

最终 clean freeze 由 commit `c6bde4b4` 重建。target、direction/extent 和 SA4 三份 sidecar 均记录该 commit 且 `runtime_dirty=false`；数值与提交前审计一致。

## 6. 图

### Shaft-aware 正控

![rev10-SA shaft-aware positive controls](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/data_driven_core_field_rev10_sa/shaft_aware_target/figures/rev10_sa_shaft_aware_positive_controls.png)

### Direction 与 event extent 分解

![rev10-SA direction extent factorization](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/data_driven_core_field_rev10_sa/shaft_aware_target/figures/rev10_sa_direction_extent_factorization.png)

### 历史 artifact 重评分

![rev10-SA historical artifact rescore](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/data_driven_core_field_rev10_sa/shaft_aware_target/figures/rev10_sa_historical_artifact_rescore.png)

### SA5 contact detectability

![rev10-SA contact detectability](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/data_driven_core_field_rev10_sa/dual_shaft_canary/contact_detectability/figures/rev10_sa_contact_detectability.png)

六张网络均使用每 contact `160` 个、半径 `1 mm` 的等量 E-cell packet。SCL/ICL current gain 为 `0.961 [0.934, 0.986]`，local neural response 为 `0.953 [0.942, 0.985]`，两杆 detector margin 均为 100% 正值。结论是 `SCL_READOUT_NOT_PRIMARY_LIMIT`：SCL 在同等局部活动下可被当前 virtual-contact readout 正常读出。

## 7. 最小修改路线

1. 从 clean commit 启动 21-candidate SA6 canary，三张新网络、12 workers、120 秒状态等待。
2. 同时比较 frozen、component-3 SCL relocation、matched off-shaft 和 `2x3x3` mass/width grid。
3. 每张网络运行 ICL-A、ICL-B、SCL 三种 forced source 和一条短 spontaneous 轨迹。
4. 只有 SA6 找到 dual-shaft feasible region，才做低维 field allocation fit；`beta` 和 Edge 继续关闭。

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
SCL_READOUT_NOT_PRIMARY_LIMIT
/
DUAL_SHAFT_FIELD_CAPACITY_RUNNING_NEXT
```

这条线现在已排除 observation 层解释。剩余的最近问题是固定预算 K=3 field 是否存在一个能同时支持 ICL/SCL 的区域；在这个问题回答前，不比较优化器，也不开放连接参数。
