# Group-Event State v0.3.5 完整执行报告（技术版）

## 0. 审阅修正（2026-09-04）

**因果重跑产物**（结果根目录 `/data/hfosp_group_event_state_v0_3_5_causal`）：q(t) 已改为只用因果的记录段位置，下文全部数字来自修复后的完整链条。

审阅代码时发现三处承重问题，已修正代码并加回归测试：

1. **（P0）动态负荷 q(t) 用了未来信息。**"记录段位置"特征原来算的是 (t − 段起点) / (段终点 − 段起点)，用到了覆盖段的**结束时刻**。而覆盖段的结束时刻多数正好是下一次发作的起点（审计 2026-09-04：E548 27/42、E922 21/29、E1125 14/21、E1146 13/27、E1096 9/23、E384 8/16、E253 7/21、E583 2/7 个覆盖段结束在发作起点上），所以它等价于"离下一次发作或断录还有多远"的倒计时。q(t) 是 W2–W6 的共同上游：H1 的动态负荷增益、H2b 的风险层、H3 的共同驱动臂以及 m(t) 的输入都受影响；8 位患者中 6 位在这个特征上学到了非零权重，E1096 上它是 5 分钟负荷模型里绝对值最大的权重。现已改为只用"距本段开始已过去的时间"（H3 的两个段位置项同样改法），并按 spec §11 的全局停止条件把整条链在独立目录重跑。
2. **（P1）"正确时刻 vs 错时"对照的锚点不一致。**错时臂只在有远距离供体的锚点上打分，正确时刻臂却在全部锚点上打分，动态负荷（E1096 5 min：136 vs 46 个锚点）、事件语法、功能读出（E548 120 min：5580 vs 3240 个值）和辅助头四处都直接相减。现在四处都在同一批锚点上比较；旧卡片里无法配平的对照记为缺失而不是沿用错配数字。
3. **（P1）H3 数值可采信规则只单向检查。**原规则只拦"子模型误差超过父模型 4 倍"，没有拦"父模型本身发散"；6 h 尺度里 +20 / +87 的"增益"正是父模型发散造成的。规则改为双向 4 倍，旧卡片按新规则重新判定。

另有两处不改数字但需注意的表述问题：`mark_only`/`static_only` 在训练好的物理头和语法臂里是"联合模型关掉 q"的消融，不是单独训练的臂；stepwise 正对照与 E922 的不可估记录沿用原运行。


## 1. 结论边界

本阶段是完整 development execution，不是 formal confirmation。工程完成、可估性、assay sensitivity 与人体科学支持分别报告。所有 seed 先在患者内取中位；seed 不是独立样本。epoch-0 或仅 1/3 seed 更新的状态保留在长表，但不进入 learned-state 主分母。

## 2. 数据、拆分与模型

- 本轮注册输入：8 位患者。完整记录 591,628 次群体 IED / 891.8 小时有效覆盖；**80% 边界之前实际用于训练与评价的是 445,015 次事件 / 713.4 小时**。完整 41 人底座未被冒充为本轮已训练队列。
- 注册患者：E253, E922, E1096, E548, E583, E1146, E384, E1125；E922 因成熟 decoder 在注册评价窗无事件，W3–W6 状态链不可估。
- FIT 20–60%、rolling INNER 60–70%、一次性 SELECTION 70–80%；development/formal/sealed 均未读取。
- `q(t)`：固定 2 min/10 min/30 min/2 h/8 h 因果 bank，负二项 future-count likelihood，静态截距为嵌套特例。
- `m(t)`：完整群体事件编码；pre-event state 预测当前/未来，事件观测后才更新；future rollout 不用真实未来事件 teacher forcing。
- frozen contact decoder：每个 recurrence step 使用低秩 FiLM、contact-specific shift 与 STOP shift；decoder 主干不更新。
- H2b：5-min grid discrete survival 和 early-ictal field/path；间期 producer 冻结。
- H3：不重叠 exposure+30-min future block，M0/M1/M2 参数槽位相同；M0 已含冻结 pre-state/q、duration、clock 和 segment position。

## 3. 训练与优化审计

完整事件状态搜索为 6 recipes × 4 subjects × 3 seeds = 72 units。`compact` 由 FIT/INNER 患者中位 rank 选出；相对 base 的 inner delta 为 −0.0007739。最终 21 units 均完成且无 OOM retry；预算审计没有 final-two-epoch unit，结论为 `ORIGINAL_BUDGET_ADEQUATE`。这排除了“普遍只因步数太少”的解释，但 E1096 0/3 和 E583 1/3 仍是患者级训练/选择失败。

| 患者 | 更新的 seed | 中位选择轮次 | 主状态分母 |
|---|---:|---:|---|
| E253 | 2/3 | 2 | 是 |
| E922 | 0/0 | NA | 否 |
| E1096 | 0/3 | 0 | 否 |
| E548 | 3/3 | 12 | 是 |
| E583 | 1/3 | 0 | 否 |
| E1146 | 3/3 | 7 | 是 |
| E384 | 3/3 | 5 | 是 |
| E1125 | 3/3 | 5 | 是 |

## 4. 机器汇总

```json
{
  "dynamic_5min": {
    "n": 8,
    "n_positive": 5,
    "n_negative": 1,
    "n_zero": 2,
    "n_nonzero": 6,
    "median": 0.11728286743164062,
    "two_sided_sign_p": 0.21875
  },
  "dynamic_30min": {
    "n": 8,
    "n_positive": 4,
    "n_negative": 2,
    "n_zero": 2,
    "n_nonzero": 6,
    "median": 0.10818767547607422,
    "two_sided_sign_p": 0.6875
  },
  "dynamic_120min": {
    "n": 7,
    "n_positive": 4,
    "n_negative": 2,
    "n_zero": 1,
    "n_nonzero": 6,
    "median": 0.004423618316650391,
    "two_sided_sign_p": 0.6875
  },
  "learned_rate_residual_5min": {
    "n": 8,
    "n_positive": 5,
    "n_negative": 3,
    "n_zero": 0,
    "n_nonzero": 8,
    "median": 0.0071839094161987305,
    "two_sided_sign_p": 0.7265625
  },
  "background": {
    "n": 8,
    "n_positive": 2,
    "n_negative": 0,
    "n_zero": 6,
    "n_nonzero": 2,
    "median": 0.0,
    "two_sided_sign_p": 0.5
  },
  "next_event_grammar": {
    "n": 5,
    "n_positive": 4,
    "n_negative": 1,
    "n_zero": 0,
    "n_nonzero": 5,
    "median": 0.0012679696083068848,
    "two_sided_sign_p": 0.375
  },
  "next20_grammar": {
    "n": 5,
    "n_positive": 3,
    "n_negative": 2,
    "n_zero": 0,
    "n_nonzero": 5,
    "median": 0.0005020499229431152,
    "two_sided_sign_p": 1.0
  },
  "next20_morphology": {
    "n": 5,
    "n_positive": 3,
    "n_negative": 2,
    "n_zero": 0,
    "n_nonzero": 5,
    "median": 0.002876307176592263,
    "two_sided_sign_p": 1.0
  },
  "next20_morphology_correct_time": {
    "n": 5,
    "n_positive": 5,
    "n_negative": 0,
    "n_zero": 0,
    "n_nonzero": 5,
    "median": 0.0033934100823529434,
    "two_sided_sign_p": 0.0625
  },
  "next20_morphology_over_mean": {
    "n": 5,
    "n_positive": 3,
    "n_negative": 2,
    "n_zero": 0,
    "n_nonzero": 5,
    "median": 0.006464369148115212,
    "two_sided_sign_p": 1.0
  },
  "physical_30min_morphology": {
    "n": 5,
    "n_positive": 3,
    "n_negative": 2,
    "n_zero": 0,
    "n_nonzero": 5,
    "median": 0.00025471046819435655,
    "two_sided_sign_p": 1.0
  },
  "physical_30min_correct_time": {
    "n": 5,
    "n_positive": 3,
    "n_negative": 2,
    "n_zero": 0,
    "n_nonzero": 5,
    "median": 0.0034543093781786993,
    "two_sided_sign_p": 1.0
  },
  "physical_30min_over_mean": {
    "n": 5,
    "n_positive": 2,
    "n_negative": 3,
    "n_zero": 0,
    "n_nonzero": 5,
    "median": -0.0003694300546257384,
    "two_sided_sign_p": 1.0
  },
  "h2b_risk_rate": {
    "n": 4,
    "n_positive": 1,
    "n_negative": 3,
    "n_zero": 0,
    "n_nonzero": 4,
    "median": -0.003993794981942736,
    "two_sided_sign_p": 0.625
  },
  "h2b_risk_mark": {
    "n": 4,
    "n_positive": 1,
    "n_negative": 3,
    "n_zero": 0,
    "n_nonzero": 4,
    "median": -0.0014220354857906459,
    "two_sided_sign_p": 0.625
  },
  "h2b_risk_correct_time": {
    "n": 4,
    "n_positive": 0,
    "n_negative": 4,
    "n_zero": 0,
    "n_nonzero": 4,
    "median": -0.00039954643647852756,
    "two_sided_sign_p": 0.125
  },
  "h2b_risk_over_mean": {
    "n": 4,
    "n_positive": 1,
    "n_negative": 3,
    "n_zero": 0,
    "n_nonzero": 4,
    "median": -0.0002684384295868758,
    "two_sided_sign_p": 0.625
  },
  "h2b_field_mark": {
    "n": 2,
    "n_positive": 0,
    "n_negative": 0,
    "n_zero": 2,
    "n_nonzero": 0,
    "median": 0.0,
    "two_sided_sign_p": null
  },
  "h3_burden_30min": {
    "n": 5,
    "n_positive": 3,
    "n_negative": 2,
    "n_zero": 0,
    "n_nonzero": 5,
    "median": 0.004622930426683491,
    "two_sided_sign_p": 1.0
  },
  "h3_mark_30min": {
    "n": 5,
    "n_positive": 2,
    "n_negative": 3,
    "n_zero": 0,
    "n_nonzero": 5,
    "median": -0.10522124905976671,
    "two_sided_sign_p": 1.0
  },
  "h3_burden_2h": {
    "n": 3,
    "n_positive": 2,
    "n_negative": 1,
    "n_zero": 0,
    "n_nonzero": 3,
    "median": 0.023294653937654952,
    "two_sided_sign_p": 1.0
  },
  "h3_mark_2h": {
    "n": 2,
    "n_positive": 1,
    "n_negative": 1,
    "n_zero": 0,
    "n_nonzero": 2,
    "median": -0.023902296616248736,
    "two_sided_sign_p": 1.0
  },
  "h3_burden_1k": {
    "n": 3,
    "n_positive": 1,
    "n_negative": 2,
    "n_zero": 0,
    "n_nonzero": 3,
    "median": -0.02263356823006392,
    "two_sided_sign_p": 1.0
  },
  "h3_mark_1k": {
    "n": 3,
    "n_positive": 1,
    "n_negative": 2,
    "n_zero": 0,
    "n_nonzero": 3,
    "median": -0.18953936879476585,
    "two_sided_sign_p": 1.0
  },
  "h3_burden_5k": {
    "n": 0,
    "n_positive": 0,
    "n_negative": 0,
    "n_zero": 0,
    "n_nonzero": 0,
    "median": null,
    "two_sided_sign_p": null
  },
  "h3_mark_5k": {
    "n": 0,
    "n_positive": 0,
    "n_negative": 0,
    "n_zero": 0,
    "n_nonzero": 0,
    "median": null,
    "two_sided_sign_p": null
  }
}
```

### 4.1 动态负荷患者表（5 min）

| 患者 | q(t)−静态 | 学习残差−q(t) |
|---|---:|---:|
| E1096 | +0.9376 | +0.0106 |
| E1125 | +0.3858 | +0.0063 |
| E1146 | +0.0223 | -0.0011 |
| E253 | +0.2122 | +0.0081 |
| E384 | +0.0000 | -0.0735 |
| E548 | -0.6658 | -0.1925 |
| E583 | +0.6064 | +0.0110 |
| E922 | +0.0000 | +0.0359 |

### 4.2 next-event contact grammar（稳健训练患者）

| 患者 | m(t)−q(t) | 正确时刻−错时 | m(t)−FIT 均值 |
|---|---:|---:|---:|
| E1125 | +0.0005 | +0.0001 | -0.0004 |
| E1146 | -0.0039 | -0.0002 | -0.0006 |
| E253 | +0.0014 | +0.0004 | +0.0002 |
| E384 | +0.0013 | +0.0000 | +0.0010 |
| E548 | +0.0208 | +0.0097 | +0.0139 |

### 4.3 30-min seizure risk（有阳性窗且稳健训练患者）

| 患者 | q(t)−临床 | m(t)−q(t) | 正确时刻−错时 | m(t)−FIT 均值 |
|---|---:|---:|---:|---:|
| E1146 | +0.0091 | -0.0012 | -0.0000 | -0.0005 |
| E253 | -0.0024 | -0.0063 | -0.0086 | -0.0052 |
| E384 | -0.0056 | +0.0111 | -0.0003 | +0.0036 |
| E548 | -0.0107 | -0.0017 | -0.0005 | -0.0001 |

### 4.4 H3 独立块汇总

| 暴露尺度 | burden 可采信患者 | burden 为正 | burden 中位 | mark 可采信患者 | mark 为正 | mark 中位 |
|---|---:|---:|---:|---:|---:|---:|
| 30 min | 5 | 3/5 | +0.0046 | 5 | 2/5 | -0.1052 |
| 2 h | 3 | 2/3 | +0.0233 | 2 | 1/2 | -0.0239 |
| 6 h | 1 | 0/1 | -0.0878 | 1 | 1/1 | +0.0898 |
| 1,000 events | 3 | 1/3 | -0.0226 | 3 | 1/3 | -0.1895 |
| 5,000 events | 0 | 0/0 | NA | 0 | 0/0 | NA |

所有 H3 原始拟合均保留。新增反馈臂在两种情况下记为不可采信（两条都不看增益符号）：一、INNER 或 SELECTION 的 MSE 非有限，或与嵌套父模型相差 4 倍以上（**任一方向**——父模型自己发散、子模型有界，同样排除）；二、父或子任一臂的 MSE 超过零模型 4 倍。零模型为 FIT 均值预测器；审阅前写出的卡片未存该字段，回退到"FIT 标准化后结局量方差按构造为 1"的单位参照。6 h 低支持结果不进入核心效应图。5k/10k 设计没有满足 ≥4/2/2 FIT/INNER/SELECTION 独立块的患者。

## 5. Assay sensitivity 与科学读法

future-oracle 在 E253/E548/E583 的 contact grammar 改善分别约 +0.0442/+0.0375/+0.0150，说明状态到逐步 decoder 的路径不是结构零。但 oracle 是泄漏答案的正对照，只证明接口灵敏，不证明真实状态存在。真实 `m(t)` 必须同时看：超过 `q(t)`、correct-time 超过 block-shift、超过 FIT-period mean；三项不同步时不能升级为时刻特异状态。

H2b 风险行若 `n_positive=0`，仍保留在 `h2b_risk.csv` 以审计校准，但不进入风险方向统计。early-field 的基本分母是 held-out seizure；只有 E548/E1146 有可估读出。H3 的基本分母是不重叠时间块，endpoint 先在患者内合并后再数患者。

## 6. 运行中发现并修复的 P0/P1

1. H2b person-period label 使用 `[left,right)`，而 seizure onset 正好定义 coverage segment 右边界，导致训练阳性全为零；改为 `(left,right]` 并加边界回归测试。
2. H2b horizon eligibility 要求完整随访，错误删除了 horizon 内已经观察到 seizure 的 anchor；改为“完整随访或在截尾前已观察到发作”。
3. selection 随访硬限制在 80% 边界，禁止把 80% 后无事件时间当作 negative exposure。
4. functional 120-min 缺少 block-shift/period-mean 时，显式输出 `NOT_ESTIMABLE` arm。
5. finalizer 只对有至少一个阳性 seizure window 的 patient×horizon 计算 primary risk summary。
6. functional/seizure maintenance CLI 改为跟随 card 中登记的 trajectory，不再猜本地同名文件。

## 7. 产物与复现

- `dynamic_baseline.csv/json`：静态、确定性 q、学习残差、block-shift。
- `state_training.csv`：每位患者实际更新 seed 和主状态资格。
- `stepwise_decoder.csv/json`、`stepwise_oracle.csv`：逐步 frozen-decoder 接口与正对照。
- `h1_h2a_{grammar,functional,auxiliary}.csv`：next-1/5/20 与 5/30/120 min 全端点。
- `h2b_{risk,field,support}.csv`：风险、early field/path 和 seizure 分母。
- `h3_{models,innovation}.csv`：M0/M1/M2 与 observer innovation。
- 每张核心图同时有 PNG、矢量 PDF 和 metadata；`figures/README.md` 给出逐图阅读合同。
- 所有 `correct_time_gain_over_shift` 列自 2026-09-04 起为同锚点对照（错时臂与正确时刻臂都只在有远距离供体的锚点上打分）；`correct_time_support_matched=False` 的行表示旧卡片无法配平、对照记为缺失。
- `h3_models.csv` 的 `*_admissible` 列由 finalizer 从各臂 MSE 重新判定：双向 4 倍父子比值，加上父/子任一臂超过零模型 MSE 4 倍即不可采信。零模型为 FIT 均值预测器（卡片 `null_fit_mean`）；审阅前写出的卡片没有该字段，回退到"标准化结局量方差按构造为 1"的单位参照，`absolute_reference` 列记录用的是哪一种。

运行 `scripts/supervise_group_event_state_v035_reporting.py` 可重建审计、机器表、图和两份报告。最终 targeted test suite 为 66 passed。所有 `development_targets_read`/`sealed_partition_opened` 标志为 false。
