# Step 1 解锁可行性综合（Exploration 1 + 2 合一，2026-06-03）

> 合并两个探索：Exploration 1（率层色散能否重现 coworker1 的行波机制，`exploration1_rate_lif_dispersion_2026-06-03.md`）+ Exploration 2（从 Topic 1/2 真实数据取锚定数）。**结论：有条件绿灯**——data_locked re-run 路径有充分动机、配方清晰，但 Step 1 仍**保持锁定**，直到 re-run 真的跑出候选窗。

## 一句话判定

**机制可达 + 数据兼容 + 配方清晰，但还有两件事要靠"扫"或"假设"而非"测出来"。** 所以这是"可行性确认"，不是"Step 1 已解锁"。

## 两个探索怎么对上的

**Exploration 1（机制）**：率层色散里，**慢抑制（抑制衰减 ~18ms）是把"原地一起动"变成"会移动的振荡（有限-k Hopf）"的钥匙**；对照——换成 scaffold 的快抑制（2ms）→ 振荡消失、退回静止。这解释了我 scaffold 为什么拿不到行波。**但这是借 Brunel 的，不是数据测出来的。**

**Exploration 2（数据锚）**：真实间期 HFO 群体事件的时间尺度——
- 单次事件包络 ~100–300ms（打包窗 500ms 是上限；通道激活展布的每被试最大值中位 178ms，典型更短 ~50–150ms）
- 事件率 ~209 次/小时；事件间隔中位 3.3s（重尾对数正态，硬下限 ~0.18s = 不应期代理）
- 点过程 ~2Hz 峰**不是内禀振荡器**——完全由"不应期 renewal + 慢速率漂移"解释（21/21 被试）
- 慢漂移在**数小时尺度**，同时调制事件率和参与通道数（互相关 0.74）

**对上了**：把网络坐在 Hopf 分岔**下方一点（亚临界）**，噪声踢出会移动又自己衰减的瞬态波包——响几个 ~20–29Hz 周期（~40ms/周期 × 几个周期）= **~100–300ms 的事件包络**，波在 cm 级电极覆盖范围内扫过 = **~50–178ms 的通道激活展布**（cm 网格 + v≈0.3mm/ms 给 ~33–100ms，落在数据区间），事件按噪声触发**~秒级复发**，整体被**数小时慢漂移**调制。**数据"不是振荡器、是不应期+慢漂移"这一条，恰恰主动支持"亚临界可激"而非"持续振荡"——正是我们要坐的位置。**

## data_locked 锚定表（Exploration 2 给的，带出处）

| 量 | 数据值 | 锚到模型的哪个 | 出处 |
|---|---|---|---|
| 事件包络时长 | ~100–300ms | 振铃时长 / 离临界距离（recovery 时间常数若开） | `pr4b_lag_validation` + 检测器 50–200ms |
| 通道激活展布 | 中位 178ms（典型 ~50–150ms） | 传播速度（延迟 + 核 + 网格） | `pr4b_lag_validation::relative_lag_max` |
| **网格尺度 L** | **cm 级（= SOZ 电极覆盖范围，不是 1–2mm）** | 空间域大小 | 见下"注意 2" |
| 不应期 / dead-time | ~0.18–0.5s | 触发不应期 | `iei_fit.iei_min` |
| 事件率 | ~209 次/h（基线） | 噪声 + 亚临界设定的触发率 | `event_periodicity::n_events/dur` |
| 慢漂移 | 数小时，调制率 + 参与数（xcorr 0.74） | Step-5 慢变量 S(t) | `event_periodicity_analysis §5.6/5.8` |
| 参与通道数 | 中位 10（IQR 7–16） | 事件空间范围 | `pr1_subject_summary` |
| 结构 vs 动态占比 | bias_fraction ~0.71 | 模板"固定骨架"占主导 | `pr1_cohort_summary::bias_fraction_median` |

## 必须"扫"或"假设"、数据测不出的（诚实清单）

- **E-I 比、离阈距离、绝对突触权重、单通道基线发放率**——数据链里没有，必须扫（= framework 要求的 operating-point family）。
- **突触抑制衰减时间常数本身**——Exploration 2 测不到。所以 Exploration 1 的"慢抑制"**只能作为 Brunel 锚定的假设带进去 + 扫一个范围**，不能说成"数据已确认"。模型在这个假设下能否重现事件级时间尺度（包络/展布/不应期），是它的**间接验证**，不是直接测量。

## 两个要小心的点

1. **慢抑制是假设不是测量**（如上）——这是 advisor 当初要 hold 到 Exploration 2 的那个点。结论：数据**不反对**慢抑制（事件级时间尺度兼容），但也**没法直接确认**它。带着扫。
2. **空间尺度要用电极覆盖范围（cm），不能照搬 coworker1 的 1–2mm 小网格**。在 1–2mm 网格上 Brunel 速度给 ~38ms 传播，比实测 ~50–178ms 快；换成 cm 级网格（真实电极覆盖）后 ~33–100ms，就对上了。所以 L 和传导速度要**联合锚**，不是照搬 Brunel 的小网格。

## 对 Step 1 的意义

**强烈倾向 data_locked re-run 可行**，配方已清晰：(a) 时间结构锚到慢抑制（假设/扫）+ 事件包络/展布/不应期（实测）；(b) 空间用 cm 级电极覆盖网格；(c) E-I / 离阈距 / 增益作 operating-point family 扫；(d) 各向异性 E→E。在某个工作点拿到"亚临界 + 噪声 → 会移动的自限波包"应当出现 → 即 Step 0b 的 self-limited PROPAGATION + 正余量 = 候选窗 → 解锁 Step 1。

**但 Step 1 仍锁定**：以上是可行性确认，不是 re-run 本身。真正解锁要等 data_locked 的 0a/0b re-run **实际跑出**候选窗（带"族里多大比例通过 + recovery off/on 并列 + dt/L 敏感性稳定"）。

（内部归档代号：finite-k Hopf via slow GABA、sub-critical noise-kicked packet、event envelope/relative_lag_max/iei_min/iei_median、bias_fraction、slow drift S(t)、operating-point family、data_locked provenance、cm-scale grid vs coworker1 1-2mm；Exploration 1 = `exploration1_rate_lif_dispersion_2026-06-03.md`，Exploration 2 = Topic 1/2 artifact archaeology）
