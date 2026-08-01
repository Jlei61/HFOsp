# FCXR-LC2 实施计划 —— Phase 0–3：迟滞 carrier、负荷终止与间期统计恢复

日期：2026-08-01

状态：**IMPLEMENTATION PLAN CANDIDATE — 未授权执行**

分支：`codex/topic4-fcxr-lc2`

基点：`3c2fc86a`（FCXR-HYB2 正式收口）

Spec：`docs/superpowers/specs/2026-08-01-topic4-fcxr-lc2-hysteretic-carrier-design.md`

> **spec 与 plan 冲突 = execution blocker。** 不设“spec 优先”或“plan 优先”；发现冲突必须先同步
> 修订两份文件并重新审阅。本文只落实 spec 授权的 Phase 0–3，不扩大科学范围。

> **硬边界**：本计划不授权 40k lifecycle，不授权 K/Na、HYB2/ELR、recruited-area A、完整
> eigenmode、最终论文图、AdEx/EIF 或修改六个 blessed engine 文件。任何阶段失败均按预注册标签
> 收口，不用更多旋钮“救活”。

---

## 0. 本轮唯一验收目标

先在实际 RC1 LIF transfer function 标定的 reduced system 中证明：

```text
stable low branch
  + stable finite H-supported high branch
  + nonzero hysteresis interval
  + X-mediated offset surface
  + X-protected Z recovery path
```

只有这套几何完整成立，才允许把唯一锁定参数迁移到动力学匹配的小型空间 SNN，验证无 kick 的
`interictal -> ictal -> postictal -> returning IED`。工程 green、单个高态、一次终止或一张好看的
轨迹都不等于完成。

---

## 1. 执行图与解锁关系

```text
T0 provenance/preflight
  ├─ T1 empirical target + window locks
  └─ T2 real dynamotype lock
          ↓
T3 RC1/LIF transfer calibration
          ↓
T4 reduced-model implementation + synthetic continuation tests
          ↓
T5 small-SNN scaling feasibility + H-sensor calibration only
          ↓
T6 H-only continuation ── fail -> H_HYSTERESIS_NO_GO / stop
          ↓
T7 M carrier-survival gate ── fail -> CARRIER_POSITIVE_MORPHOLOGY_NEGATIVE / stop
          ↓
T8 X offset/postictal continuation
          ↓
T9 dynamic-Z closed slow path ── fail -> reduced-stage stop
          ↓
T10 local-H engine path + parity/contracts
          ↓
T11 small spatial SNN development, G0–G6
          ↓
T12 locked confirmation seeds
          ↓
T13 archive/status/diagnostic figures
```

T1/T2 的真实窗若不完整，不阻塞 T3–T9 的**机制开发**，但阻塞 observation-matched morphology
阈值、T11 的 G3/G6 终局和任何患者匹配措辞。该状态必须写为 `EMPIRICAL_TARGET_INCOMPLETE`，不能
用旧 HEO 阈值临时代替。

---

## 2. 目录、文件与单一真源

结果根：

```text
results/topic4_sef_hfo/fcxr_lc2/
  STATUS.md
  run_manifest.json
  candidate_verdict.json
  resource_log.jsonl
  empirical_lock/
  transfer/
  reduced/
  small_snn/
  figures/
    README.md
```

新代码建议：

```text
src/topic4_fcxr_lc2_empirical.py
src/topic4_fcxr_lc2_transfer.py
src/topic4_fcxr_lc2_reduced.py
src/topic4_fcxr_lc2_gates.py
scripts/build_topic4_fcxr_lc2_empirical_lock.py
scripts/run_topic4_fcxr_lc2_transfer.py
scripts/run_topic4_fcxr_lc2_continuation.py
scripts/run_topic4_fcxr_lc2_small_snn.py
scripts/plot_topic4_fcxr_lc2_diagnostics.py
tests/test_topic4_fcxr_lc2_{empirical,transfer,reduced,gates}.py
tests/test_mz_lc2_h.py
```

不把科学判决散落到 runner。所有分类和 gate 进入纯函数模块并有 synthetic bad-data regression；
runner 只负责输入、调度、sentinel 和落盘。

---

## 3. T0 —— provenance、边界和 artifact preflight（零仿真）

### 3.1 必查内容

1. 当前分支必须由 `3c2fc86a` 分出；HYB2 worktree clean，最终 archive 可解析。
2. 六个 blessed engine 文件记录启动 sha256；每 stage 重验。
3. 核验 LC1 X、HEO2/3 M、RC1 saturation 的 source commit、config 与 artifact sha256。
4. 核验 E1146 SQL、raw/head、15-contact montage、onset/offset 和相邻 block 连续性。
5. 核验现有 HEO1 `coop_A` 在 LC2 primary 中恒为 0；缺少显式字段即响亮失败。
6. 核验 confirmatory connection/noise seeds 尚未出现在 LC2 参数选择产物中。

### 3.2 输出

```text
run_manifest.json
empirical_lock/artifact_preflight.json
reduced/upstream_parameter_provenance.json
```

任一承重 artifact 缺失或 checksum 漂移，标 `ARTIFACT_PREFLIGHT_FAILED` 并停止相关下游；禁止靠
文件名相似推断来源。

---

## 4. T1/T2 —— 真实目标、returning windows 与 dynamotype lock

### 4.1 T1 数据流

复用 `scripts/run_heo_gate_on_real_seizure.py` 的 loader、SQL truth 和 15-contact local-CAR 合同，新增
只读的 window builder：

1. 从同一患者的非发作时段选择至少 3 个 returning-interictal population-event windows；选择规则
   在读模型结果前冻结，保存 detector version、事件 ID、绝对时间和 block checksum。
2. 生成 seizure onset `[0,3] s`、established `[3,18] s`、pre-offset `[-5,0] s`、postictal
   `[0,15] s`、recovery `[15,60] s`。
3. 若窗口跨 block，只能按 SQL recording/block 时间连续性拼接；gap、重复、采样率或 contact 顺序不一致
   即失败。
4. sharp pulse-comb null 独立生成；不能从真实 IED 或模型 candidate 反推其参数。

### 4.2 T2 dynamotype

只记录可观测量：onset abruptness、amplitude-from-zero、early ISI trend、offset slowing/abruptness、
DC availability。若采集链不能证明 DC，标签保持合并，不强分 saddle-node/subcritical Hopf。

### 4.3 输出与测试

输出 spec §3.2 的五个 JSON，并测试：

- contact order/local CAR 一致；
- 绝对与相对时间可逆；
- onset/offset 不跨 SQL gap；
- returning windows 不与 seizure/postictal 重叠；
- pulse comb、returning IED、early ictal 三类不会共用同一 label；
- raw/head/SQL checksum 和采样率进入每个 window record。

T1/T2 只锁**真实目标**，不输出模型 GO/NO-GO。

---

## 5. T3 —— 实际 RC1/LIF transfer calibration

### 5.1 被测对象

必须使用 RC1 最终膜方程：feedforward additive、recurrent E→E conductance、recurrent smooth
saturation、`coop_A=0`、H/M/X/Z 全 off 或冻结在声明值。解析 `src/sef_hfo_lif.py` 只作 null。

### 5.2 probe 设计

在固定噪声流下，分层测量：

1. 单细胞 E/I：二维 `(I_E,I_I)` response surface；
2. recurrent conductance：`g_raw -> g_eff -> firing`，覆盖 baseline、transition、high、ceiling；
3. frozen Z/M/X：在同一 operating point 做中央差分 sensitivity；
4. E/I delay 与 effective noise：由短扰动 impulse/step response 估计；
5. refractory ceiling、distance-to-threshold 和 local gain；
6. held-out grid 点只用于验收，不参与拟合。

同一输入点至少重复 3 个 noise streams；surface 拟合必须单调、边界有界，并保存 bootstrap。不得用
任意 Wilson–Cowan sigmoid 替代实际响应。

### 5.3 hard gate

预注册误差在运行前由 baseline replicate variability 给出：held-out absolute error 不得超过
`max(2*replicate_MAD, 10% dynamic range)`，且 transition 区符号/斜率一致。否则
`TRANSFER_CALIBRATION_UNRESOLVED`，T4 以后停止。

输出：

```text
transfer/rc1_transfer_probe.npz
transfer/rc1_transfer_surface.json
transfer/rc1_transfer_bootstrap.json
transfer/transfer_gate.json
```

---

## 6. T4 —— reduced system 与 continuation 仪器

### 6.1 先测仪器，后测 LC2

对以下 synthetic systems 做回归：

- saddle-node normal form；
- subcritical/supercritical Hopf 区分；
- 有/无 bistable window；
- stable/unstable branch；
- 长 transient 但无 attractor；
- coarse step 造成的假 hysteresis；
- transfer bootstrap 足以覆盖表面差异的“不可判”例。

必须同时用 forward/backward sweep、pseudo-arclength、Jacobian eigenvalues、最小奇异值、双 basin time
integration。少一层不得给 `fold/Hopf/bistable` canonical label。

### 6.2 数值合同

- ODE solver 至少两种 tolerance/step 设置复现分支；
- continuation step 减半后 fold 坐标变化小于 transfer bootstrap uncertainty；
- analytic Jacobian 或 automatic/finite-difference Jacobian 交叉核验；
- 每个候选分支保存状态、稳定性、局部 gain、ceiling distance；
- 参数和状态带单位；无量纲化必须能逆变换到 engine-drive/Hz/ms。

输出：

```text
reduced/continuation_instrument_gate.json
reduced/reduced_model_contract.json
```

仪器未过，不能把后续空结果写成模型 NO-GO。

---

## 7. T5 —— 小型缩放可行性与 H-sensor calibration（不跑 lifecycle）

这一 stage 只为把 reduced H 坐标映射回真实 local recurrent-drive 分布；不是跳过 reduced gate。

### 7.1 规模选择

依次检查 `N={4000,8000,16000}`，按 spec §7 保持几何无量纲比、E/I 比、expected in-degree、权重
均值/方差、delay 和 RC1 normalization。选择满足合同的最小 N；如果概率需截断到 1，直接判该 N
不可行。

### 7.2 H-sensor-only probe

在 H 输出严格关闭 (`rho_H=0`) 时，仍离线/旁路记录每个候选 `tau_H={50,100,200} ms` 对
`gErec_raw_i(t)` 的 exact exponential low-pass，得到各自 `Q99.9` 作为 `theta_H`。同一轨迹同时测
H-off baseline 与 frozen-state transfer parity。

不得为了得到更好 theta 改 event bar、drop cells 或只取静默窗。若 baseline 不匹配 RC1 accepted
band，判 `SMALL_SNN_SCALING_BLOCKED`，不进入 T6。

输出：

```text
small_snn/scaling_feasibility.json
small_snn/h_sensor_calibration.json
small_snn/h_sensor_trace_summary.npz
```

---

## 8. T6 —— H-only 迟滞 continuation

### 8.1 固定候选盒

严格使用 spec §5.3：

```text
tau_H = {50,100,200} ms
theta_H = corresponding H-off Q99.9
k_H=0.1*theta_H
rho_H/g_sat = {0.25,0.50,0.75}
M = X = 0
d = frozen continuation coordinate
```

不得扩大盒子；不得用 waveform 挑点。若多个通过，按最小 rho、最短 tau、字典序。

### 8.2 两层 continuation

1. `(d,rho_H)` 找 low/high branch、fold/Hopf/ceiling；
2. 对唯一候选在 `(d,x)` 上做 offset-load unfolding，当前先把 x 当冻结抑制坐标，不接动态 X。

H high branch 必须在 M/X off 时独立存在。16 Hz Hopf 可同时存在，但不能是唯一高态；common-mode
oscillation 不计作迟滞证据。

### 8.3 判决

输出 `reduced/h_geometry_gate.json`：

- `H_GEOMETRY_ACCEPTED`：spec §6.2 七条全过；
- `H_HYSTERESIS_NO_GO`：只有 ramp/runaway/saturation/common Hopf；
- `FAST_TRANSFER_FUNCTION_REPAIR_REQUIRED`：迟滞存在但 high branch 是 refractory plateau；
- `H_GEOMETRY_UNRESOLVED`：continuation/transfer uncertainty 无法区分。

后二者不得偷写成机制阴性。非 accepted 一律停止，不实现 SNN H。

---

## 9. T7 —— M carrier-survival 与形态方向锁

### 9.1 reduced survival

在唯一 H geometry 上打开 mean adaptation：`tau_M=250 ms`，`g_M/eta_M` 按该 high branch recurrent
drive 的 10% force-match 解析确定。只问 high branch 是否还存在、hysteresis margin 是否仍高于
uncertainty；不在 reduced model 宣称宽带或去同步。

### 9.2 小型 SNN morphology probe 的授权边界

只有 reduced survival 通过，才允许在 T11 development trajectory 中比较：

- per-cell M_i；
- matched total-load mean-field M；
- M-off。

不扫描 HEO3 patch placement，不复用旧绝对 `eta_m=0.354`。若 per-cell M 只产生长 burst-silence 并
摧毁 carrier，标 `CARRIER_POSITIVE_MORPHOLOGY_NEGATIVE`。

输出：`reduced/m_survival_gate.json` 与锁定的 force-match provenance。

---

## 10. T8 —— X offset 与 postictal continuation

### 10.1 代码符号回归

低维负荷坐标固定 `x=1-mean(x_relay)`。必须用 synthetic trajectory 证明：

- sustained y 超阈 -> `x_relay` 下降 -> reduced x 上升；
- x 上升降低 recurrent availability；
- `x=0` 对应 relay 完全可用；
- 不允许把 `x_relay` 直接作为“终止负荷上升”。

### 10.2 参数和唯一轴

复用 LC1 锁值：`tau_y=120 ms, K_X=5, n=4, x_min(relay)=0.1,
tau_x_down=1000 ms`；仅比较 `tau_x_up={5000,10000} ms`。`y_gate` 用新 H-off baseline 的 Q99.9，
不沿用旧绝对值 76.6/85.3。

### 10.3 continuation/gate

对唯一 H/M 点检查 spec §6.4 七条。X-off matched control 只置 relay neutral，其他状态/噪声相同。
若 5 s 和 10 s 都能 offset 但都无法给 Z recovery 足够保护，标
`OFFSET_POSITIVE_RECOVERY_NEGATIVE`；不加 X→Z。

输出：

```text
reduced/x_offset_surface.json
reduced/x_recovery_gate.json
```

---

## 11. T9 —— dynamic Z 与 autonomous closed slow path

### 11.1 Z 选择规则

沿用现有 Z 方程和 `tau_z_down/up`，只解析选择 `I_th_EI`：

- baseline replay 前 8 s 不跨 T6 的 `d_on`；
- 在锁定开发时窗内有非零 spontaneous crossing probability；
- 规则在结果前写死；不能扩成 hazard 网格。

若无阈值同时满足，标 `Z_ENTRY_CALIBRATION_UNRESOLVED`。

### 11.2 closed path

接回真实 `dot z` 与 `dot x`，从 autonomous ODE/随机 reduced system 运行，不 kick、不 step 参数。检查：

1. repeated events 使 d 增加并跨 onset surface；
2. H high branch 建立；
3. X 在 onset 后积累并跨 offset surface；
4. offset 时 low branch 稳定；
5. X 保护期内 Z 回到 safe side；
6. X 衰减后仍留在 low/interictal basin；
7. pre/post event-statistic distribution 距离回到 bootstrap band。

输出 `reduced/closed_slow_path_gate.json`。只有 `REDUCED_LIFECYCLE_GEOMETRY_ACCEPTED` 解锁 T10。

---

## 12. T10 —— local H 实现与工程合同

### 12.1 唯一实现

只修改非 blessed `src/snn_engine/mz_slow_vars.py`，新增与旧 `coop_A` 分开的字段，例如：

```text
use_h_lc2, tau_h_lc2, theta_h_lc2, k_h_lc2, rho_h_lc2
h_lc2_E
```

H 使用同一 target 的 `gErec_raw_i` exact exponential low-pass；`gH` 在 RC1 tanh saturation 之前相加；
`B(V)=1`。不得新建 W、读 core mask、global rate 或 seizure label。

### 12.2 一步因果顺序必须锁死

在 step n：膜电流使用 `h(t_n^-)`；本步计算得到的 `gErec_raw(t_n)` 只用于更新
`h(t_{n+1})`。不得在同一步先更新 H 再反作用膜电流。snapshot 必须保存 H 和所需 cache，使 restart
后逐位一致。

### 12.3 必测合同

- `rho_H=0` 与 RC1 raster/trace byte-identical；
- H-off 与 old `coop_A=0` 不引入额外 current；
- exact exponential 与解析常值输入一致；
- one-step causality；
- same-W-path equivalence；
- E-only H（因为 H 是 recurrent E→E）；
- finite/nonnegative/shape validation；
- determinism、snapshot/restart；
- H 与 X/M/Z 组合开关无 mutex 泄漏；
- old HEO1 `coop_A` 与 LC2 H primary 互斥；
- 六个 blessed hashes 不变。

工程合同未过，只能写 engineering blocked，不能给 H scientific NO-GO。

---

## 13. T11 —— 小型空间 SNN development 与 G0–G6

### 13.1 参数冻结与 seeds

只迁移 T6–T9 的唯一参数点和 T5 的最小可行 N。development connection seeds 固定 2 个；noise seeds
在 run manifest 开跑前写死。不得依据第一条轨迹增加新 H/M/X/Z 候选。

### 13.2 运行长度与模块顺序

先分阶段最小验证，再跑 nominal lifecycle：

1. RC1/H-sensor baseline；
2. H-on G0 + frozen-d basin probe（kick 只标 basin，不计 lifecycle）；
3. H+M morphology probe；
4. H+M+X frozen-d offset probe；
5. H+M+X+dynamic-Z nominal no-kick trajectory。

nominal 每条至少 30 s，目标时序为 `>=8 s pre + 1–5 s high + postictal + >=8 s recovered`。
如 onset 尚未发生，不因 wall time 延长到任意长度；最长开发窗在 run manifest 预锁，超过即
`NO_SPONTANEOUS_ONSET_IN_LOCKED_WINDOW`。

### 13.3 七门必须逐门输出

```text
small_snn/g0_baseline.json
small_snn/g1_onset.json
small_snn/g2_carrier.json
small_snn/g3_morphology.json
small_snn/g4_spatial.json
small_snn/g5_offset.json
small_snn/g6_recovery.json
small_snn/development_verdict.json
```

关键纪律：

- G0 失败就不跑 lifecycle；
- G1 需 dynamic-Z spontaneous onset，matched Z-frozen 不进入；
- G2 保存 spec §8.1 全套 tonic-saturation 指标；
- G3 所有指标同一个 200–300 ms active window 合取，classifier 必须拒绝 pulse-comb、returning IED、
  HEO1 16 Hz 三个坏数据；
- G4 用 first-passage/newly recruited area/front/axis latency，不用 occupied-volume 投票；
- G5 X-on 与 matched X-off；
- G6 比较 pre/recovered 多变量统计邻域，不把静默或固定节律叫恢复。

开发阶段只要任一 seed 触发数值不安全、saturated tonic branch 或角色错位，停止，不用另一 seed
“投票盖过”。科学 gate 要求两 development seeds 同方向；不一致标 `DEVELOPMENT_SEED_UNRESOLVED`。

---

## 14. T12 —— confirmation seeds（只在七门全部通过后）

参数、阈值、窗口、分类器、最长时窗全部冻结后，才揭盲至少 3 个 connection/noise seed。每个 seed 跑：

- nominal no-kick lifecycle；
- 七门内必要的 Z-frozen、X-off 和 mean-field-M matched controls。

不在本 plan 跑完整 H-off/isotropic/axis-rotation Phase-4 矩阵。若 confirmation 不复现，终局是
`SMALL_SNN_LIFECYCLE_NOT_CONFIRMED`，不得返回 development 调参。

输出 `small_snn/confirmation_verdict.json` 与逐 seed provenance。

---

## 15. T13 —— 图、归档和允许措辞

只画诊断图，不画最终论文图：

1. `empirical_target_lock.png`：真实 returning/early-ictal/offset/postictal 窗与阈值；
2. `reduced_hysteresis_geometry.png`：branches、on/off surfaces、closed slow path；
3. `small_snn_seven_gate_diagnostic.png`：仅在 T11 有数据后展示 G0–G6；
4. `recovery_state_distance.png`：pre/post 统计距离和 returning IED。

图实际生成后才写 `figures/README.md`，逐图 2–4 句中文并含 `**关注点**：`。失败阶段没有输入就不
画占位图。

归档：

```text
docs/archive/topic4/sef_hfo/fcxr_lc2_<terminal_label>_2026-08-XX.md
```

措辞分层固定：

```text
engineering
transfer calibrated / unresolved
H geometry accepted / no-go / unresolved
reduced lifecycle geometry accepted / not established
small-SNN candidate / confirmed / not confirmed
40k lifecycle not tested
```

即便 small SNN confirmed，也只能称“动力学匹配的缩小空间 SNN 生命周期候选”，不能称 E1146 40k
已复现或论文目标完成。

---

## 16. OOM、nohup 与并行纪律

### 16.1 先测资源，再定 worker

每一种新 N/时长先单 worker smoke，记录 peak RSS 和 wall/sim-second。worker 数：

```text
W_mem = floor((MemAvailable - 96 GiB safety reserve) / (1.35 * measured_peak_RSS))
W = min(W_mem, stage hard cap)
```

hard cap：`T>=20 s: W<=4`，`T<20 s: W<=6`；若 sibling 40k 在跑，按总实测占用继续下调。不能因为
机器总内存大就跳过单实例测量。

### 16.2 线程与 swap

所有 worker 设置：

```text
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

- submit 前 `MemAvailable >= 96 GiB` 且至少 `2 * measured_peak_RSS`；
- swap 相对 stage baseline `+256 MiB`：停止提交新任务；
- swap `+512 MiB` 且继续上升：只终止**本任务最新** worker，绝不碰 sibling；
- 任一 worker RSS 超单实例预估 1.5 倍：终止该 worker并标 resource failure；
- 不用过量 trace；per-cell trace 仅保存预注册 core/axis/off-axis 与必要抽样，其余在线归约。

### 16.3 detached 执行

超过 10 分钟的仿真必须：

- `setsid nohup ... > nohup_<stage>.log 2>&1 &`；
- `launcher_<stage>.pid`，且等待/终止只读 PID，不用 `pgrep -f`；
- stage-scoped `flock`；
- `RUNNING_<run> / DONE_<run> / FAILED_<run>` sentinel；
- per-run wall kill guard、resource_log、exit code；
- 每次 teardown 前检查本任务残留 PID。

网络断线不是失败；DONE/FAILED 和原子落盘 JSON 是唯一完成依据。不得删除、kill、renice 或修改任何
sibling worktree/process。

---

## 17. 提交策略与停机交付

建议按逻辑提交，不积成一个大提交：

1. `docs: lock LC2 design and implementation plan`
2. `test: add LC2 empirical and transfer contracts`
3. `feat: lock E1146 empirical target windows`
4. `feat: calibrate RC1 transfer surface`
5. `test: validate LC2 continuation instrument`
6. `feat: map LC2 H geometry`
7. `feat: add off-by-default local H engine path`
8. `feat: run LC2 small-SNN seven-gate validation`
9. `docs: archive LC2 terminal verdict`

每个 scientific gate 都可成为合法终点。合法终点必须同时交付：terminal label、最大阻断、允许/禁止
claim、tests、blessed hashes、资源、sentinel、artifact/figure/README、commit stack、无残留进程。

---

## 18. 开跑前最终 checklist

- [ ] spec 与 plan 状态均经用户签核为 DESIGN LOCK / EXECUTION AUTHORIZED；
- [ ] HYB2 保持 closed，LC2 branch/base 正确；
- [ ] empirical raw/SQL/montage preflight 通过；
- [ ] confirmatory seeds 未泄漏；
- [ ] transfer probe/held-out gate 在运行前锁定；
- [ ] H/M/X/Z 候选和 selection rule 与 spec 完全一致；
- [ ] `x=1-x_relay` 符号测试存在；
- [ ] `coop_A=0` 与 `B(V)=1` 明示；
- [ ] 40k/K/HYB2/A/eigenmode/final figure 均未授权；
- [ ] OOM/nohup/flock/sentinel 合同可运行；
- [ ] 六个 blessed hashes 记录；
- [ ] spec/plan cross-check 无差异。

checklist 未全过，不得启动第一条仿真。
