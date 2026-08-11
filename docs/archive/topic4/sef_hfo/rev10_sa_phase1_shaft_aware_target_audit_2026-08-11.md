# Topic 4 rev10-SA 第一阶段审计：shaft-aware target 与历史重评分

日期：2026-08-11

# 审阅结论

## 1. 一句话判断

rev10-SA 第一阶段已经回答了“旧目标是否看不见 SCL”这个问题：答案是**是**。新的目标能稳定识别 SCL 删除、跨杆 timing 改变和 0/4 到 4/4 的连续恢复；但现有历史产物还不能证明“旧 objective 选错了某个已有 field”，因为真正的 48 个 field-fit 候选没有保存逐事件固定 contact 数据。

SA5 contact detectability 已经完成并排除 SCL observation 主限制。SA6 随后只完成了 fixed-K3 component-3 relocation canary；它不是连续自由场容量测试。当前进入 SA6F continuous-field correction，仍不直接开放 Edge、`beta` 或 topology。

## 2. 完成程度

> **rev10-SA SA0-SA6 完成度：90/100；完整 dual-shaft recovery 完成度：约 55/100**

已完成：

- SA0 的 15-contact/shaft 合同、55/6/44 无序 pair 类和哈希；
- patient-training-only 的 fixed-contact timing 与 ordinal-compatible 双目标；
- 分杆 recruitment、四状态 precedence、分杆 profile 和 event cloud；
- 四个零仿真正控；
- patient direction 与 event extent 的 factorization 审计；
- rev8.1 final、rev9 factorial、hand dual-core、filament、L2/L3 retained artifact 的统一重评分。

未完成：

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

### SA6 fixed-K3 component-3 relocation canary

![rev10-SA dual-shaft capacity](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/data_driven_core_field_rev10_sa/dual_shaft_canary/dual_shaft_capacity/figures/rev10_sa_dual_shaft_capacity.png)

`21` 个固定 K=3 fields 在 `3` 张新网络上全部完成，共 `63/63` workers 成功、零 runaway。没有任何候选让 ICL mode-A 或 mode-B source 招募 SCL；因此 mode A/B 的 SCL recruitment floor excess 分别固定在 `2.81/3.88`，全部高于 1。

这不是因为 SCL field 没有真正加上去。最强候选 `grid_mid_w35_s4p5` 在 SCL contact 1 mm 邻域达到：

```text
mean h                         0.424
median h                       0.447
h >= 0.5                      42.2%
mean delta Vtheta             -0.216 mV
threshold-lowered neurons      69.7%
```

但该候选的 ICL-A -> SCL 和 ICL-B -> SCL 招募仍均为 `0`。反向的 SCL packet 在 `2/3` 网络能触及至少一个 ICL contact，不过平均只招募 `0.061` 的 ICL contacts。短 spontaneous 总共检测到 `14` 个事件，全部返回、`0` 个 multishaft event。

因此最窄的结论是：**固定两个 ICL Gaussian、只移动并调节第三个 component 的 2+1 分配没有产生 ICL→SCL 传播。** 它既不能外推成所有 K=3 fields 失败，也不能裁定连续自由场容量。

## 7. 最小修改路线

1. 先运行无 K、无 component/峰数约束的 continuous B-spline field canary。
2. 以 4×4 控制面作为 matched-DoF primary，6×6 只作分辨率敏感性；控制系数不是 core。
3. 只有 continuous field 仍不能产生 ICL→SCL 时，才运行 packet-amplitude 与总 field-budget curve。
4. 若上述实验仍显示方向性不可达，再设计 directional route-support；`beta` 继续关闭。

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
FIXED_K3_COMPONENT3_RELOCATION_CANARY_NEGATIVE
/
LOW_RESOLUTION_CONTINUOUS_INITIALIZATION_NO_CROSS_SHAFT_SUPPORT
/
NO_K_CONTINUOUS_CONNECTED_FIELD_FAILS_CROSS_SHAFT_AT_FIXED_PACKET_AND_BUDGET
```

SA6F 已完成 `37 x 3 = 111` 个无 K B-spline workers，零 runaway，但没有
ICL-to-SCL recruitment。该结果仍不能裁定连续场或 connectivity：最强 SCL 邻域
平均 `h` 只有 `0.128`，明显低于 constrained K3 canary 的 `0.424`。一个 4x4
候选把 mode-A ICL precedence excess 从约 `2.99` 降到 `0.393`，说明连续场能
改变杆内限制项，但没有解决 SCL。

SA6G 已完成 `8 x 3 = 24` 个 workers，全部 clean、零 runaway。四个已连接场的
真实 bridge mean `h=0.528-0.907`，bridge-near `h>=0.5` 比例为
`74.4-98.0%`；但两个 ICL source 在所有场和所有网络中均为 0 个 SCL contacts。
SCL 反向只平均触及约 1/11 个 ICL contacts。5 个短 spontaneous events 全部来自
disconnected narrow controls，全部返回、无 multishaft event。

因此已排除“固定 K/core 数”和“桥上场太弱”作为当前阴性的主要解释，但还不能直接写
connectivity family fail：packet amplitude 与总 field budget 尚未扫。下一步先做小型
packet-amplitude curve，再视结果做 budget curve；只有持续 ICL→SCL=0 才开放
directional route-support，`beta` 继续关闭。

## 9. Observation-boundary correction: SA6H

后续审计撤回了 SA6F/SA6G 的“自由场”称谓。SA6F 的 B-spline 系数由患者接触点上的
recruitment target 拟合，SA6G 的支撑直接来自观测到的两条杆路径；二者都会让信息量随
电极覆盖变化，只能保留为 observation-conditioned capacity diagnostics，不能说明患者数据
从无偏二维场中恢复出 substrate。

SA6H 回到 Stage 3 已学出的明确场，但只把它作为全 sheet warm start。新的 latent field
使用整张 `20 x 20 mm` sheet 上的 real Fourier `cos/sin` 基底和 stationary isotropic
residual；field builder 不接收 contact、shaft、patient onset、mode label 或 forced source。
患者信息只在 8-s spontaneous SNN 仿真完成后，经虚拟电极 readout、冻结 mode classifier
和 shaft-aware training objective 进入评分。

初始开发库为 `V0/V1/V2` 共 21 个候选，在配对网络 `1031-1033` 上运行；结果用于构造
小型 `V3` refinement，再以新网络确认。由于患者 held-out 已经使用，整个 SA6H 仍只能写
development-only recovery，不能写 patient blind generalization。Edge、`beta` 和 topology
在 observation-invariant Node field 冻结前继续关闭。

### SA6H initial result

初轮 `21 x 3 = 63` 个 8-s spontaneous workers 全部从 clean commit `dd9ae9ac`
完成，无 worker failure、无 runaway。旧 K3 与它的均匀 Fourier 投影分别产生 `34/33`
个事件；投影 `h` RMSE 为 `0.0103`，top-5% support Jaccard 为 `0.952`，说明表示替换
没有破坏旧场。uniform field 为 0 事件，是有效负对照。

但初轮没有恢复 shaft-aware patient repertoire。谱 warm start 的 weak-mode score 为
`5.397`；表面 winner `v1_pair01_plus` 为 `5.407`，只因 OOD 从 `0.037` 降到 0 才在总分
上领先 `0.009`，不能视为场改善。9 个 support-eligible candidates 的 SCL recruitment
均为 0，mode A/B SCL recruitment excess 固定为 `5.413/4.105`。KMeans 与 frozen labels
的 AMI 为 1，但 model-A prototype 与 patient A 明显不符，证明双簇稳定不等于患者模式复现。

本轮主要限制是搜索半径：旧 warm field 在场外比峰值低约 `6-7` 个 log units，V1/V2
扰动至多约 1 RMS，足以改变 A/B occupancy，甚至造成单模式 collapse，却不足以让候选在
SCL recruitment 上产生变化。V3 改为 4x4 全 sheet 等距 allocation scan；16 个位置在不读取
contact/shaft 的情况下冻结，同一平滑方向投影回 Fourier 场，由仿真后的 patient objective
选择。这些是 optimizer probes，不是新增 core 或 K。

### SA6H V3 correction and V4 handoff

V3 `63/63` workers 在 `c933986b` 完成，零失败、零 runaway。原聚合仍选择旧场，但原因
不是所有新场都没有 SCL：位置 07/10 分别产生 `39/21` 个 SCL-only 事件，位置 12 产生
`13/43` 个双杆事件，位置 09 产生 `4/12` 个双杆事件。原 aggregator 在 shaft-aware loss
之前调用旧 shared-axis rank curve；SCL-rich events 被变成不可用或 OOD，因此目标入口仍然
shaft-blind。这个结果只能解释为 continuous Node field 具有 SCL activation capacity，不能
解释为患者双杆 repertoire 已恢复。

patient full shaft-aware KMeans 与旧 A/B 的 AMI 只有 `0.011`，因为前者主要区分招募范围，
后者是传播方向。使用 patient train、按 recording block 隔离的监督式 A/B 分类器，6 折
balanced accuracy 为 `0.939-0.957`，pooled `0.945`，AUC `0.990`。因此 V4 把方向身份和
双杆参与因子化：每个事件都分配 A/B，同时 OOD 只作惩罚不作删除；所有事件另算 ICL-only、
joint、SCL-only。

V3 Fourier 坐标的条件数约 `1e8`，不适合继续优化。V4 改为不读取观测的 `14 x 14` uniform
cubic B-spline continuous field，条件数 `27.6`；Stage 3 warm 的 `h` RMSE `0.0098`，top-5%
Jaccard `0.966`。V4 冻结 50 个候选：warm、uniform negative、16 个全 sheet 位置各两个温和
幅度，以及 8 对 observation-free 平滑随机场。Edge、`beta`、topology 继续关闭。

### SA6H V4 result and V4.1 bridge

V4 从 clean commit `48d49318` 完成 `50/50` workers，零失败、零 runaway。
数值线程已限制为每 worker 1 个，16 并发时内存余量超过 170 GiB。50 个候选均没有
joint ICL+SCL event；只有 6 个候选出现少量 SCL-only activity。因此原自动输出的
`v4_alloc_02_a1p0` 只是 joint penalty 全部相同时的 route scalar minimum，不是科学
winner。修正状态为：

```text
REV10SA_V4_NO_JOINT_SHAFT_CANDIDATE
```

这一阴性不能归因于 spline family 或 optimizer。V4 实际测试的是 `1.0 x Stage3 warm`
加幅度 1/2、宽 3 mm 的温和扰动，没有覆盖 V3 已知产生 SCL/joint events 的
`0.5 x warm + amplitude 4, width 2.5 mm` 区域。下一步 V4.1 不根据结果挑位置，而把 V3
全部 21 个场逐一投影到稳定 `18 x 18` uniform B-spline 坐标。全库预检最大 `h` RMSE
`0.00316`、最小相关 `0.99975`、最小 top-5% Jaccard `0.971`、条件数 `27.53`；field
builder 仍不读取 contact、shaft、patient event 或 V3 score。V4.1 先验证表示桥接，再开放
V5 连续系数优化；不增加 K/core，Edge 与 `beta` 继续关闭。

V4.1 随后完成 `21/21` workers，零失败、零 runaway。uniform 09 在同 seed 上从 V3 的
`3 joint + 3 ICL-only` 变为 spline 的 `2 joint + 4 ICL-only`；uniform 12 从
`2 joint + 9 SCL-only` 变为 `4 joint + 9 SCL-only`。因此稳定 spline 保留了旧场的
shaft capacity，但最佳 joint fraction 仍只有 `0.333`，远低于患者约 `0.95`，且事件云
高 OOD。这是表示桥接通过，不是患者模式恢复。

V5 使用冻结规则从训练 seed 选四个锚点：Stage 3 参考、两个最高 joint 场 uniform 09/12、
一个额外最低 route 场 uniform 06。患者训练目标只作为 optimizer feedback；18×18 均匀
basis、每处空间分辨率和正则化不变。对 6 对锚点分别构建 latent-linear 与 density-mixture
两条连续路径，每条取 `t=0.25/0.5/0.75`，加四个锚点共 40 个场。未观测区域仍是平滑先验
延拓，不能声称由患者数据识别。V5 先在 1031 拟合，再把多样 Pareto subset 带到
1032/1033；不增加 K/core，Edge、`beta` 和 topology 继续关闭。
