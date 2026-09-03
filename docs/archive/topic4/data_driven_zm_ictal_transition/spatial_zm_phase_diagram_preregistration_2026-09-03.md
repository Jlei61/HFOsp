# 空间 Z/M + 持续 OU：相图与分岔审计预注册

**Date:** 2026-09-03  
**Branch:** `codex/topic4-zm-phase-diagram`  
**Status:** `PREREGISTERED_BEFORE_NEW_PHASE_POINT_RESULTS`

## 1. 科学问题

当前 frozen-q 粗扫显示：`q=0.825` 为低活动支，`q=0.800` 出现高率但未完全招募的
中间状态，`q=0.775` 为近饱和 tonic plateau。该结果只定位了一个陡峭 operating-regime
edge，不能区分：

1. 同一参数下低、高两支并存的 saddle-node-like bistability；
2. 唯一稳态随 q 陡峭但连续变化的 crossover；
3. OU 驱动下的 noise-induced / metastable switching；
4. 振荡支的 Hopf-like onset。

本轮首先在**同一冻结患者来源网络**上做双初态 frozen-q SNN 分支图；只有 SNN 出现稳定双支后，
才启动 mean-field fixed-point continuation + Jacobian，以判定能否正式命名动力系统 bifurcation。

## 2. 模型与控制参数

- 网络、learned E→E/E→I、位置、接触点及 OU 统计合同全部继承 `tonic_b0_v2`，不得重拟合。
- primary 模型：hybrid Z/q–M/gK，`eta_m=0.02`、`tau_m=12.5 ms`。
- mechanism ablation：q-only，`eta_m=0`；它回答 M 是否改变支结构，不替换 primary。
- 实际控制量是 `q_clamp`：每个 phase point 设置 `q_init=q_min=q_clamp` 且
  `freeze_q=true`。**不得把动态轨迹里的 `q_min` 参数扫描称为分岔扫描。**
- primary environment：与 Fig. 5 相同的 stationary spatial OU，`sigma_rate=0.10/ms`、
  `tau=20 ms`、`ell=0.38 mm`。同一 phase point 的低/高初态共享未来 Poisson/OU 随机流。
- OU-off 只能作为噪声消融；SNN 仍含外源 Poisson 抽样，因此 OU-off SNN 也不等于数学确定性系统。

## 3. 双初态与配对

- low initial state：seed 1842 exact checkpoint at 200 ms。
- high initial state：seed 1842 exact checkpoint at 600 ms。
- checkpoint 只提供膜电位、突触电流、delay ring 与 M 状态；开始 phase point 前将整个 q field
  明确替换为该点的 `q_clamp`。
- 对每个 `noise_seed`，低/高初态必须使用同一个重新初始化的网络 Poisson RNG、空间 OU 当前场与
  OU innovation RNG；不得让 200 ms 与 600 ms checkpoint 原有的不同未来噪声冒充初态效应。
- 所有 learned edges 保持 100%；无 kick、无 onset-triggered drive、无 30–80 Hz 注入。
- 运行在隔离 worktree 中；被 `.gitignore` 排除的冻结 `results/` 输入从
  `/home/honglab/leijiaxin/HFOsp` 只读解析。输入身份仍由逐文件 SHA256 决定，不以目录名代替。

## 4. 扫描顺序

### Stage 0：instrument + branch canary

- `q_clamp ∈ {0.790, 0.805, 0.820}`；
- initial state ∈ {low, high}；
- `noise_seed=9101`；
- 每条 1200 ms，前 400 ms 为 state-reset transient，末 800 ms 打分。

Stage 0 只检验 exact state override、matched future noise、支分类和资源成本，不用于正式结论。

### 2026-09-03 Stage 0 后补充声明（在补充模拟启动前写入）

Stage 0 实测在 `q=0.820` 为 `INTERMEDIATE/INTERMEDIATE`，在 `q=0.790` 为
`INTERMEDIATE/TONIC_HIGH`，尚缺同一 `eta_m=0.02` 下的两端单稳锚点。故补跑
`q_clamp ∈ {0.770, 0.840}`、同一 `noise_seed=9101`、low/high 双初态、1200 ms/400 ms
burn-in。该补充不修改任何分类阈值，只用于给后续边缘加密建立外侧 bracket；仍不进入正式
多 seed 结论。

`q=0.840` low-start 通过 LOW，但 high-start 的 median rate=82.2 Hz，仅比冻结 LOW 上限高
2.2 Hz，故没有形成双 LOW 锚点。在看到更外侧数据前再声明一次单步扩展到 `q=0.860`，其余合同
不变；此步仍是 bracket 工程层，不进入多 seed 结论。

### Stage 0d：q-only 共存单点

Stage 0/0b/0c 表明 `eta_m=0.02` 下的初值分离主要落在宽的 intermediate 区，尚无严格
`LOW/HIGH` 同点共存。旧的 q-only frozen-q locator 已显示 low-start 在 `q=0.825` 仍属低态，
而该点从未做 high-start 对照。因此在任何 q-only 密网格之前，先固定
`q=0.825, eta_m=0`，用 noise seed 9101、low/high 两个 checkpoint、1200 ms 总时长和末
800 ms 打分窗做单点共存检验。这个检验若阳性，只记为单 seed 的经验共存候选；若阴性，也不
据此否定其他 q 点或更长驻留时间下的共存。

该扩展在运行前登记，不改分类阈值、噪声或初始状态定义。

Stage 0d 完成后，`q=0.825, eta_m=0` 得到 `LOW/INTERMEDIATE`；high-start 从约 289 Hz
回落，末窗 median=86.25 Hz，没有保持 tonic-high。按照二分而不是密扫的原则，在运行前追加
`q=0.8125, eta_m=0` high-start。只有该臂达到 `TONIC_HIGH`，才补同点 low-start；否则继续
向较低 q 二分。此顺序只节省明显无共存可能的配对臂，不改变状态判据。

Stage 0e 的 `q=0.8125, eta_m=0` high-start 再次衰减，末窗 median=100.75 Hz，分类为
`INTERMEDIATE`。结合旧 q-only low-start locator 的 `q=0.825` 低态、`q=0.800` 中间态，
high-branch 存活边界与 low-start 逃逸边界都可能位于 `0.800–0.8125` 的窄区间。故在运行前
追加二分点 `q=0.80625`，仍先跑 high-start；只有达到 `TONIC_HIGH` 才补 low-start 来检验
严格共存。其他合同不变。

### Stage 1：一维 q 分支

- 初始固定网格：`q_clamp=0.770..0.840`，步长 0.0025；
- low/high 双初态；paired noise seeds `9101/9102/9103`；
- 每条 2500 ms，前 1000 ms burn-in，末 1500 ms 打分；
- 若相邻格分类不同，在该格间追加二分点，直到 q 宽度 ≤0.00125；
- 不得因图形好看删 seed 或移动阈值。

### 2026-09-03 Stage 1 资源修订（任何正式 Stage 1 运行前）

Stage 0 实测单条 1200 ms arm 需要约 6–35 分钟，而且连续二分揭示宽的 `INTERMEDIATE`
区，不是预想中的干净二态跳变。原矩形方案为 29 q × 2 eta × 3 seeds × 2 initial states =
348 条 2500 ms 运行；按当前成本直接执行既不成比例，也会把大量预算花在已知无严格共存可能的
格上。因此原矩形 Stage 1 在**尚无一条正式运行**时冻结为 superseded，改为以下自适应顺序：

1. 在 `eta_m=0` 先分别二分 high-start tonic survival edge 与 low-start escape edge；
2. pilot seed 9101 的二分宽度先收至 0.003125；`INTERMEDIATE` 不自动算边界，必要时先延长驻留；
3. 只有 pilot 同一点得到 low=`LOW`、high=`TONIC_HIGH`，才在该 overlap 候选上补
   seeds 9102/9103 和 2500/1000 ms 正式分母；
4. q-only 边界清楚后，再用相同顺序测 `eta_m=0.02` 的位移；
5. classifier、matched-noise、checkpoint、OU、边和 3/3 robust rule 全部不变。

这只改变计算资源的投放顺序，不改变科学阳性门，也不允许删掉不利 seed。

### Stage 2：二维 Z/M 相图

只有 Stage 1 出现至少 `BISTABLE_CANDIDATE` 才启动。横轴为 `D=1-q_clamp`，纵轴为
`eta_m ∈ {0, 0.01, 0.02, 0.04, 0.08}`；q 只扫 Stage 1 边界两侧各 0.015。每格仍为
low/high 双初态 × 3 paired seeds。

## 5. 预先冻结的状态量与分类

打分窗内先用 20 ms moving average 计算群体率：

- `LOW`（Stage 0 原始 v1，已由下方正式前量表修订取代）：median E rate ≤80 Hz，q95 <120 Hz，
  median active-E <0.50；
- `TONIC_HIGH`：median E rate ≥300 Hz，median active-E ≥0.85，median recruited-sheet ≥0.85，
  joint global duty ≥0.80；
- `INTERMEDIATE`：其余有限、数值稳定状态；
- `UNSTABLE`：NaN/Inf、超过 refractory ceiling 或积分合同失败。

同一 `q_clamp × eta_m × noise_seed` 的双初态联合分类：

- `LOW_MONOSTABLE_CANDIDATE`：low-start 与 high-start 均为 `LOW`；
- `HIGH_MONOSTABLE_CANDIDATE`：两者均为 `TONIC_HIGH`；
- `BISTABLE_CANDIDATE`：low-start=`LOW` 且 high-start=`TONIC_HIGH`；
- `REVERSE_SPLIT`：low-start=`TONIC_HIGH` 且 high-start=`LOW`，视为审计失败/未收敛；
- `MIXED_OR_UNRESOLVED`：其余组合。

跨三 paired seeds：3/3 同为 `BISTABLE_CANDIDATE` 才叫 robust SNN bistability candidate；2/3
只能叫 stochastic/metastable candidate；≤1/3 不支持双稳态。

### 2026-09-03 Stage 0 量表修订（正式 Stage 1 前冻结）

Stage 0 揭示 `q95<120 Hz` 会把“低 median、低全局 duty、只有短暂间期样事件”的状态误判成
`INTERMEDIATE`：例如 `q=0.860` high-start 的 median=49.7 Hz、global duty=0.051，但仅因
q95=124.5 Hz 失败。这与本轮低态允许每秒约 4–5 个离散事件的科学定义不一致。

因此 LOW 的 q95 条款在正式 Stage 1 前替换为与既有 runaway detector 同尺度的持续性条款：
20-ms 平滑群体率不得 `>=120 Hz` 连续达到 100 ms。median rate ≤80 Hz 与 median active-E
<0.50 保持不变；q95 继续报告但只作诊断。TONIC_HIGH 的四条门完全不变。旧 q95 标签保留在
每个 worker JSON 的 `classification_history`，新标签标为 `event_tolerant_low_v2`。这是 construct
修正，不是为了制造 L/H 点；修订发生在任何 Stage 1 正式点启动之前。

## 6. 什么结果允许叫“分岔”

SNN 双支图即使出现 hysteresis，也只写 **empirical SNN bistability/hysteresis**。要正式写
`saddle-node bifurcation`，还必须：

1. 在与 SNN 参数映射冻结的 deterministic mean-field/rate reduction 中求出 low/high fixed points；
2. 用 pseudo-arclength continuation 找到中间不稳定支；
3. Jacobian 有一个实特征值在转折点穿零；
4. SNN branch/hysteresis 区与 mean-field fold 区方向一致。

若共轭复特征值穿过虚轴且振荡振幅从零附近出现，才考虑 Hopf。若只有 OU 打开时发生随机跃迁，
则结论必须是 noise-induced/metastable transition。`phase transition` 另需预注册 order parameter 与
finite-size scaling，不由本轮自动获得。

## 7. 产出与 fail-closed

- 科学结果根：`results/topic4_sef_hfo/data_driven_zm_phase_diagram/`；
- worker JSON/NPZ：`phase_points/`；
- 聚合 CSV/JSON：结果根；
- 图：`figures/`，并在图生成后写中文 `README.md`；
- 若缺任一双初态、matched-noise 证据或 3-seed 分母，不生成 paper-facing bifurcation panel；
- preliminary coarse locator 只能标 `DIAGNOSTIC_COARSE_SCAN`，不能标 bifurcation。
