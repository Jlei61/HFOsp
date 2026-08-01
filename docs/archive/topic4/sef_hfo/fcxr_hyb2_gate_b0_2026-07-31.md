# FCXR-HYB2 Gate B0 —— 在预注册阈值下通过（膜层判据），转 Gate A0

- 日期：2026-07-31（2026-08-01 按审阅修订：撤回 `Q_on` 泄漏、B0 判据重定域到膜层）
- 分支：`codex/topic4-fcxr-hyb2`（worktree `.worktrees/topic4-fcxr-hyb2`，未 push / 未 merge）
- plan of record：`docs/superpowers/plans/2026-07-31-topic4-fcxr-hyb2.md`
- spec：`docs/superpowers/specs/2026-07-31-topic4-fcxr-hyb2-event-limited-recruitment-design.md`
- 结果根：`results/topic4_sef_hfo/mz_full_conductance_spatial_relay/hyb2_event_limited_recruitment/`

## 0. 一句话

在**预注册的 calibration 半段阈值**（`Q_on` = 112.505 / 173.429）下，Gate B0 两个 seed 均判
`BASELINE_PRACTICALLY_INVISIBLE`：seed1 在 validation 半段出现**一次极稀少激活**
（20.34 s，占空比 1.077e-4，全部落在最后四分之一段），seed3 **未激活**；两个 seed 的间期事件
统计与 off 臂**逐位相同**。判据已按审阅重定域到**膜层**——`q_v` 是隐藏传感器，只有越过
`Q_on` 后生成的 `R_evt` 才进入膜方程。

**允许的表述**：在预注册的 calibration-half 阈值下，seed1 出现极稀少的 validation 激活、
seed3 未激活；两个 seed 均未观察到间期事件统计扰动（24 s、两个连接 seed）。
**不得写**："一次都没启动"、"逐位证明完全隐形"。

## 1. 判决表（`gate_b0_seed{1,3}.json`）

| 子句 | 地位 | seed1 | seed3 | 门 |
|---|---|---|---|---|
| `R_evt` 活跃占空比 | **判据** | 1.077e-4 PASS | 0.0 PASS | ≤ 0.01 |
| `R_evt` 分段占空比（无持续抬升） | **判据** | [0, 0, 0, 4.31e-4] PASS | [0,0,0,0] PASS | 每段 ≤ 0.01 |
| 间期事件统计落在 baseline 带内 | **判据** | 逐位相同 PASS | 逐位相同 PASS | — |
| 数值安全 | **判据** | PASS | PASS | — |
| `q_v` onset 前残留 | 诊断，不投票 | 0.2669 | 0.1096 | （参考 0.01）|
| `q_v` 地板漂移 | 诊断，不投票 | +0.00857 | −0.00937 | （参考 0.01）|

事件统计"逐位相同"是字面意义：seed1 34→34、IEI CV 0.631760→0.631760、中位时长 10.0→10.0 ms；
seed3 67→67、0.525396→0.525396、11.0→11.0 ms。

分段占空比是**精确导出**而非估计：`t_gate` 是首次有体素越过 `Q_on` 的块，在它之前 `R_evt`
恒为零，所以早于 `t_gate` 的每一段占空比精确为 0，全部占空比集中在其后的段。

## 2. 为什么 `q_v` 的两条不再投票（审阅裁定，两个独立理由）

**(a) 层级不对等。** HYB1 栽的是**直接进入膜方程**的钾地板棘轮；`q_v` 是隐藏传感器，它的地板
再高也不影响任何一个细胞，除非越过 `Q_on` 变成 `R_evt`。把 HYB1 的否决权原样移植到隐藏状态
上不是保层级的翻译。膜层的对应读数是分段 `R_evt` 占空比，**沿用同一个 1% 界，不引入新阈值**。

**(b) onset 前残留是坏尺子。** 它的推导（`q_peak·exp(−GAP/τ_R) ≤ 0.01·Q_on`）前提是 gap 内
`e_v ≡ 0`，而 §4.1 定义 `b_v := Q99_t[s_v]`，按构造保证每个体素 1% 的块高于自己的背景
（实测占空比中位数 0.0074、90 分位 0.0103）。图
`figures/b0_gap_resolved_envelope_seed1.png` 摊开整条空档期（只采距上次 offset ≥ 4τ_R 的点，
此时余量 < 2%）：

| 距 onset | q99/`Q_on` | 贡献 gap 数 |
|---|---|---|
| 200 ms | ≈0.40 | 23 |
| 100 ms | 0.0155 | 32 |
| **30–75 ms** | **0.0044–0.0077** | 33 |
| 25 ms | 0.047 | 33 |
| **2 ms（原采样点）** | **0.244** | 33 |

**包络确实清空了**（30–75 ms 静息窗过 0.01）。原采样点高，是因为它坐在**下一次事件自己的
局部爬升**上；空档期深处的高值来自**够不到全局判据条、因而不算事件的局部活动**。两者都在
执行器阈值以下。两条仍在每次判决里报告，所以这个降级是可审计的。

## 3. 撤回：`Q_on` 数据泄漏（2026-08-01 审阅 P0）

2026-07-31 的一次提交把 `Q_on` 从"calibration 前 12 s 的事件峰值"改成"全 24 s 的事件峰值"，
理由是 plan §4.2 第 2 步写的是 `max_{event, occupied voxel}`。**这是错的**：§4.2 **第 1 步**
写明"在 calibration 半段上"，spec:207 更把 calibration/validation 分离写成强制、第 3 步明说
"用**未参与定阈值的** validation half 检查 false activation"。当时是 grep `Q_on` 读的，
第 1 步不含这个字串因而没被读到。

后果：seed1 的 `Q_on` 被 validation 半段的峰值抬到 169.846，**高于 validation 最大值 154.405**，
于是"执行器在间期一次没启动"成了循环验证。当时还加了一条测试把这个偏差锁成"正确行为"。

已撤回：`event_peak_values` 的 `calibration_end_ms` 是**必填、无默认**（有默认就等于给调用方
一条静默恢复泄漏路径的口子，与 PR-6 `valid_mask=None` 同一失效模式），三个调用点全部传值；
那条测试反向改成"validation 事件不得进入阈值"。`finalize` 无-load 分支改为使用精确 float64
存值、并用 `q_max_series` 独立重导做**交叉校验**（该序列存为 float32，故校验带 float32 容差）。

被取代的产物在 `superseded/`：`calibration_lock_fullrecord_Q_on_LEAKED.json`、
`gate_b0_seed{1,3}_fullrecord_Q_on.json`。

## 4. 离线重判，不重跑 baseline

用预注册阈值跑出的那两条轨迹**本来就在盘上**（`superseded/gate_b0_seed{1,3}.json`，
`code_commit=58cccf41`），管线确定性，重跑 40k baseline 只是花 70 分钟复现已有数字。
新增离线阶段 `gate-b0-readjudicate`（无仿真），并带守卫：若归档运行的 `Q_on` 与锁不符则拒判。

- **seed3 在两次运行中是同一条轨迹**：两次 `Q_on` 都是 173.429、`q` 从未达到它，没有任何电流
  被加进去，所以两次不可能不同。它的 `q` 诊断取自后一次运行（用的是契约的联合统计量）。
- **seed1** 以预注册那次为准；`q` 诊断从记录的 calibration load 重算（开环传感器臂），并在
  JSON 的 `q_diagnostic_source` 字段标明。

## 5. 附带记录

- `_git_sha()` 报的是 HEAD，对脏工作树不可靠：`calibration_seed3.json` 记的 commit 与 seed1
  不同，但两者的 `cmd_calibration` **逐字节相同**（已 diff 验证），seed3 只是跑在 load-save
  进工作树之前。
- Gate A0 的前置检查一度仍在比对已退休的状态名，导致两臂都拒绝启动；现在生产者与消费者共用
  同一个常量 `H2.B0_PASS_STATUS`，并有测试锁住。
- 本轮两次 40 分钟仿真死在**归约阶段**（重复 dict 关键字；抽纯函数后调用点残留已删局部名），
  故新增 `scripts/hyb2_preflight_static.sh`：发射前跑 pyflakes 未定义名 + 单元套件，毫秒级。

## 6. 下一步

Gate A0（两条 9 s，off / on 臂）—— 判"招募执行器在高负荷态是否真的有效"，三分判决
（输入不足 / 天花板混淆 / 有效或无效）。之后按审阅意见：先用 M-off 12 格拓扑筛出**至多两个**
具备"自发 onset → 有界高态 → X 终止 → 恢复"拓扑的（近）候选，各补一条固定 M-on 的 24 s，
再检验 M 是否把高态从密集事件串变成持续、宽带、去同步但广泛招募的振荡平台；
**只差波形门的点不因 M 未解锁而提前淘汰**；最后才跑 seed3 与未见噪声。
