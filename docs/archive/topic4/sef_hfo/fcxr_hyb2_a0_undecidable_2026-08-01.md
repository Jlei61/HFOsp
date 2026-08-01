# FCXR-HYB2 收口：基线门通过，但所有锁定 A0 输入均被空间天花板混淆

- 日期：2026-08-01
- 分支：`codex/topic4-fcxr-hyb2`（未 push / 未 merge）
- plan of record：`docs/superpowers/plans/2026-07-31-topic4-fcxr-hyb2.md`（含运行前锁定的 rev2）
- spec：`docs/superpowers/specs/2026-07-31-topic4-fcxr-hyb2-event-limited-recruitment-design.md`
- 结果根：`results/topic4_sef_hfo/mz_full_conductance_spatial_relay/hyb2_event_limited_recruitment/`

## 0. 最终判决

HYB2 在 Gate B0 通过：事件尺度执行器在两个间期 seed 上没有造成可测的事件统计扰动。Gate A0
没有得到执行器“有效”或“无效”的判决，而是得到 **`A0_UNDECIDABLE_ALL_LEVELS`**：初始 q50
点和 rev2 预先锁定的 S25/S50/S75 三档，在执行器关闭时都已经招募超过 90% 的占据体素，空间
extent 没有足够动态范围承载“至少提高 10%”的因果检验。

按预注册停机规则，12 格生命周期短屏、长窗七门、M 波形臂、seed3/unseen-noise confirmatory
均未解锁。因此本 sprint **没有得到、也没有否定可控发作 lifecycle**。安全结论是：HYB2
解决了 HYB1 的跨事件基线棘轮，但当前 A0 输入/读出组合无法识别它是否保留 B2.1 的招募效应。

## 1. 本轮修复后成立的 Gate B0

审阅发现旧实现把 validation 半段事件纳入 `Q_on`，使 seed1 阈值从预注册的 112.505 被抬到
169.846，形成循环验证。修复后 `Q_on` 只来自 calibration 前 12 s：seed1 = 112.5047，
seed3 = 173.4290。已有正确阈值轨迹被离线重判，无需重复 40k baseline。

| seed | 膜层 `R_evt` 活跃占空比 | 门 | 事件统计 | 判决 |
|---|---:|---:|---|---|
| 1 | 1.077e-4 | ≤0.01 | 与 off 臂相同 | `BASELINE_PRACTICALLY_INVISIBLE` |
| 3 | 0 | ≤0.01 | 与 off 臂相同 | `BASELINE_PRACTICALLY_INVISIBLE` |

`q_v` 是隐藏传感器，不直接进入膜方程；其 onset 前残留和地板漂移保留为诊断项，不再错误地
继承 HYB1 对膜电流棘轮的否决权。完整订正见
`docs/archive/topic4/sef_hfo/fcxr_hyb2_gate_b0_2026-07-31.md`。

## 2. Gate A0 rev2：三档全部不可判

q50 初始点已在运行前按旧合同判为 `A0_CEILING_CONFOUNDED`（930/1024 体素）。为避免在一个已
饱和输入上误判机制，rev2 在新仿真前锁定了三条规则：只用已经锁定的 S25/S50/S75 Z 轴；
eligibility 只由 off 臂决定；主判决取最弱 eligible 档。若三档均不可判，禁止继续降 Z、修改
窗口或提高 90% 天花板。

| Z 档 | `S_Z` | `I_th_EI` | `t_gate` (ms) | off 体素 | off 体素占比 | max `R_evt` off/on | 末 1 s E 率 off/on (Hz) | 判决 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| S25 | 0.2315 | 37.487 | 1518.5 | 923/1024 | 90.14% | 0.232/0.232 | 5.094/5.094 | `A0_CEILING_CONFOUNDED` |
| S50 | 0.2882 | 13.606 | 1510.5 | 925/1024 | 90.33% | 1.363/1.363 | 6.121/6.121 | `A0_CEILING_CONFOUNDED` |
| S75 | 0.3597 | 3.655 | 1517.5 | 924/1024 | 90.23% | 2.159/2.243 | 11.491/11.495 | `A0_CEILING_CONFOUNDED` |

三档均越过 0.9×1024 = 921.6 的体素天花板；无一档触发 300 Hz runaway，clip 均为 0，且
`t_gate` 后均有超过 7 s 的观察窗。因此失败不是输入不足或数值问题，而是空间 extent 的动态
范围不足。S25 只比天花板多 1.4 个体素，但门是预注册的，不能按结果移动。

off/on 的参与细胞、体素和半径在三档内相同。由于 off 臂已经接近全域，这些相同值同时兼容
“执行器对 extent 无效”和“执行器有效但 extent 被天花板截住”，当前设计不能区分两者。

## 3. 科学含义

### 已经做对的

1. 事件执行器的短记忆和阈值化设计消除了 HYB1 那种跨事件浓度棘轮；基线门在两个 seed 上通过。
2. `R_evt` 最大值沿 S25→S75 从 0.23 墠到约 2.24，说明 A0 确实覆盖了弱到中等执行器暴露；
   “不可判”不是因为执行器始终没有打开。
3. eligibility 完全由 off 臂预先决定，未用 on/off 效果选择最有利档位，避免了结果导向选点。

### 仍然没有做到的

1. 没有证明 ELR 能提高空间招募，也没有进入无 kick 的间期→有界高态→终止→统计恢复闭环。
2. 没有检验高态是否为持续、宽带、去同步但广泛招募的振荡平台；M 没有解锁。
3. 没有 seed3/unseen-noise 的 lifecycle 复现，也没有 paper-ready 的 pre/ictal/post 三窗主图。

## 4. 工程与资源

六条 rev2 A0 仿真均为 9 s、无 kick、无 clip、finite，峰值 self RSS 约 11.1 GiB/条。并行六条时
合计常驻约 41 GiB，运行期间 MemAvailable 约 199–209 GiB，swap 未增长，未触碰 sibling 进程。
所有长任务使用 stage sentinel/lock 和 wall guard；结束后无本任务仿真残留。

图：`figures/hyb2_gate_summary.png`；中文图注：`figures/README.md`。

## 5. 下一代设计应改什么

不能在 HYB2 合同内继续“救”A0。下一代应把**状态发生**与**执行器效力**分开：先在一个不会
天然铺满全部体素、但能稳定产生有界高态的 reduced/toy 或局部输入上做 actuator causal assay；
再回 40k E1146 底座检验 lifecycle。若仍以 40k 空间范围作主读出，必须在运行前另锁一个不受
全域参与天花板支配的量（例如 onset-latency gradient、招募速度或持续高能量的接触覆盖），并用
同步全域负控证明它能区分“广泛但同步”与“传播式招募”。这属于新 spec，不是本 sprint 的阈值修补。

## 6. 允许与禁止措辞

允许：`engineering green`；Gate B0 accepted；Gate A0 `A0_UNDECIDABLE_ALL_LEVELS`；HYB2 消除
了 HYB1 的间期棘轮；当前空间 extent assay 在所有锁定输入上被天花板混淆。

禁止：`NO_GO_EVENT_LIMITED_ACTUATOR`；执行器无效；得到/否定 seizure lifecycle；limit cycle、
bistability、Hopf；复现患者离子浓度或真实发作传播。
