# FCXR-LC2-Core 实施计划 —— 快速回答 H basin、X offset 与 Z recovery

日期：2026-08-01

状态：**IMPLEMENTATION PLAN CANDIDATE — 未授权执行**

Spec：`docs/superpowers/specs/2026-08-01-topic4-fcxr-lc2-core-design.md`

分支：`codex/topic4-fcxr-lc2`

> **spec/plan 冲突 = execution blocker。** 同日旧 LC2 Phase 0–3 plan 已标记
> `SUPERSEDED — DO NOT EXECUTE`。本 plan 不得与旧 T1–T12 混跑。

---

## 0. 本轮执行顺序

```text
R0  equation/data-flow vertical slice + H-off parity
 ↓
R1  H sensor separability + feasible tau interval
 ↓ fail: H_SENSOR_NOT_SEPARABLE / stop
R2  targeted three-region transfer + H/X reduced geometry
 ↓ fail: H_LOOP_NO_BISTABILITY / H_HIGH_BRANCH_SATURATED /
         X_HAS_NO_OFFSET_AUTHORITY / stop
R3  40k frozen basin forks
 ↓
R4  analytic X-protection time + dynamic Z/H/X lifecycle, seed1
 ↓ positive: CORE_LIFECYCLE_CANDIDATE
R5  two prelocked replication seeds
 ↓ positive: CORE_LIFECYCLE_REPLICATED
R6  archive + handoff to deferred Phenotype
```

不先做 empirical/dynamotype，不做 M，不做小网络 scaling，不做空间/宽带/患者 gate。

---

## 1. 结果根与代码边界

```text
results/topic4_sef_hfo/fcxr_lc2_core/
  STATUS.md
  run_manifest.json
  candidate_verdict.json
  resource_log.jsonl
  r0_vertical_slice/
  r1_sensor/
  r2_reduced/
  r3_frozen_forks/
  r4_lifecycle/
  r5_replication/
  figures/
    README.md
```

建议新增：

```text
src/topic4_fcxr_lc2_core.py
src/topic4_fcxr_lc2_core_gates.py
scripts/run_topic4_fcxr_lc2_core.py
scripts/plot_topic4_fcxr_lc2_core.py
tests/test_topic4_fcxr_lc2_core.py
tests/test_mz_lc2_h.py
```

唯一既有代码改动候选：`src/snn_engine/mz_slow_vars.py`。当前审计证明 hook 足够，不改 blessed
`kick_probe.py/params.py/model.py/connectivity.py/connectivity_rot.py/lfp.py`。若实现时事实与审计不符，
停止并修订 spec；不得在错误位置绕接 H。

---

## 2. R0 —— 方程、接口和 100–500 ms vertical slice

### 2.1 先写失败测试

新增配置/状态，命名与旧 `coop_A` 分离，例如：

```text
use_h_lc2
tau_h_lc2
theta_h_lc2
k_h_lc2
rho_h_lc2
h_lc2_E
```

最少测试：

1. `rho_h_lc2=0` 对 RC1 raster、膜 trace、RNG state 逐位一致；
2. `I_E_rec` 经 X presynaptic scatter 后再进入 `gA_raw`；
3. H 输入等于 post-X `gA_raw`，不是 pre-X edge sum、总 I_E 或 saturation 后 gErec；
4. membrane step n 使用 `h_n^-`，`gA_raw,n` 只影响 `h_{n+1}`；
5. constant-input exact exponential；
6. `h=0` 时 normalized sigmoid output 严格 0；
7. H 在 RC1 tanh 之前相加；
8. snapshot/restart 与 deterministic replay；
9. `coop_A>0` 与 LC2 H primary 互斥；
10. finite/nonnegative/shape/units validation。

### 2.2 sensor-only 与 active smoke

- 100 ms H-sensor-only：state 演化、membrane 完全不变；
- 500 ms active H：人工 constant/recurrent input，验证 H 有效且不产生 NaN/clip；
- 500 ms X-on：验证 X 降低 gA 后 H source 同步下降，旧 H 只按 tau 衰减。

输出：

```text
r0_vertical_slice/dataflow_contract.json
r0_vertical_slice/parity.json
r0_vertical_slice/smoke.json
```

R0 工程不通过，不得把后续失败写成科学 NO-GO。

---

## 3. R1 —— H sensor 可分离性

### 3.1 artifact preflight

四个 canonical provenance 在本 plan 中直接锁定，不能在同类轨迹中另挑：

| 状态 | canonical artifact | 当前 sha256 |
|---|---|---|
| RC1 returning IED | `LC1_ROOT/baseline_trace_seed1.npz` | `b6204332e6a62bcfbf04f268149b057bef42d4eb7219495cbf07097fef8f286e` |
| LC1 q75 dense train | `LC1_ROOT/runs/20260722T171901.346631Z_2583352_f56b721_zonly_seed1_q75_T24000/zonly_traces.npz` | `082e362b192434a259b3bf2431af865db82ef2c5d138d96fb4a559717adc9649` |
| HEO1 sustained state | `HEO_ROOT/high_energy_oscillatory_branch/screen_cells/gq0.999_A8_D0.15_nokick_trace.npz` | `cfd50a44c7fd689f0cb01d3ca0010656b3f2062010e7c06ba8e9c4ba913a72a7` |
| HEO2 fast-10% | `HEO_ROOT/broadband_diagnostic/arms/dyn_tau250_frac0.1_trace.npz` | `2995f7490ebec4bc3f39ae37be79215cec600204fe02ceab650bdf357ea35582` |

根路径：

```text
LC1_ROOT=/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-fcxr-lc1/results/topic4_sef_hfo/mz_full_conductance_spatial_relay/lifecycle_closure
HEO_ROOT=/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-fcxr-heo1/results/topic4_sef_hfo/mz_full_conductance_spatial_relay
```

RC1 的配套 `baseline_contract_seed1.json` sha256 锁为
`fd3e0d05ef730c30a484a071046e6a92d8f5e775b2035646dc89f4b4e8367c53`；HEO1/HEO2 配套 JSON 分别为
`18c2a76e4f3d6733d079b89cc1649b274ba5d0a0b1c1e6df534255dbe3f38967` 与
`a4c4b915a85168d63270e79fe34cbbedc69875ab658f565ff1f0c890e1acfaae`。

执行时核验存在性、sha256、config、seed；任一漂移响亮失败，不自动重选：

- RC1 accepted interictal baseline；
- LC1 q75 dense event train；
- HEO1 sustained 16 Hz state；
- HEO2 fast-10% intermittent/clonic state。

现有 NPZ 没有足够的逐细胞 `gA_raw(t)`，因此不能声称“零仿真重分析”。运行同配置 sensor-only replay，
要求 rate/LFP/事件摘要落入原 artifact 的 deterministic/accepted tolerance；不匹配即
`SENSOR_REPLAY_INVALID` 并扣留科学判决。

### 3.2 控制 trace 体积

每条 run 预先固定 4096 个均匀抽样 E cells 记录 `gA_raw`，另存 core/axis/off-axis 摘要但不参与全局
quantile 投票。采样 seed 在 manifest 先锁；1 ms block-average、float32。baseline 8 s，其余各 5 s。

禁止保存 40k×dt 全量 trace。在线同时记录 full-population gA mean/q50/q90/q99 作为采样代表性诊断。

### 3.3 tau 可行区

对保存 trace 在 `[5,2000] ms` 做 24 点 log bracket，再对 pass/fail 边界二分到 0.5 ms；这是同一数据的
数值求根，不是 SNN 参数 sweep。

按 Core spec 计算 bootstrap `L_tau/U_tau`：L 来自 accepted RC1 IED，U 取 HEO1 与 HEO2 established
high 各自 Q0.10 的较小值；LC1 q75 dense train 只作过渡诊断。并报告 next-onset residual、
shortest-gap residual、high-trough residual。pass 集必须包含一个非空连通区；多个连通区取包含最小
tau 的区间，避免无必要长记忆。

锁定：

```text
tau_H = geometric midpoint
theta_H = (L+U)/2
k_H = (U-L)/(2 ln 9)
```

输出：

```text
r1_sensor/sensor_replay_manifest.json
r1_sensor/gA_sampled_traces.npz
r1_sensor/h_sensor_separability.json
r1_sensor/h_parameter_lock.json
```

无可行区直接写 `H_SENSOR_NOT_SEPARABLE`、出一张诊断图并停止。

---

## 4. R2 —— targeted transfer 与 H/X reduced geometry

### 4.1 只测三个区域

从 RC1/H-off 选择并在 run 前写入 operating-point contract：

- baseline low；
- transition（介于 accepted low 与 existing high 的 recurrent drive）；
- existing finite high/tonic。

每区做短 frozen probe，测 firing response、local slope、recurrent saturation derivative、refractory-ceiling
distance；每个点至少 3 个固定 noise repeats。不给完整二维 transfer surface。

### 4.2 rho 解析选点

用 measured slope 解 `rho_crit`，只评估：

```text
rho = 1.05 * rho_crit
rho = 1.25 * rho_crit
```

先 multi-start equilibrium、forward/backward slow sweep 和 low/high initialization。多个候选取较小 rho。
两点都没有双 basin，写 `H_LOOP_NO_BISTABILITY`，不扩盒。

### 4.3 候选后正式确认

只有候选出现才运行 pseudo-arclength、Jacobian/min singular value、step/tolerance sensitivity。需要确认：

- stable low；
- unstable separatrix；
- stable finite high；
- hysteresis margin 大于数值/重复不确定性；
- high local gain 非零且不靠 refractory ceiling。

若只得到 ceiling plateau，写 `H_HIGH_BRANCH_SATURATED`。

### 4.4 X 必须走乘性 relay

在同一方程中 frozen sweep `ell_X=1-mean(x_relay)`；X 同时降低 fast gA 和 H source，不加 `-g_Xx`。
求 `ell_X,off` 与 `ell_X,release`。可达 relay 区间内 high basin不消失，写
`X_HAS_NO_OFFSET_AUTHORITY`。

输出：

```text
r2_reduced/targeted_transfer.json
r2_reduced/h_loop_gain.json
r2_reduced/h_branches.json
r2_reduced/x_offset_geometry.json
r2_reduced/geometry_verdict.json
```

Reduced 不生成 IED，只保存由现有 SNN event statistics 推出的 d drift corridor。

---

## 5. R3 —— 原 40k substrate 的 frozen basin forks

### 5.1 为什么直接 40k

沿用已验收连接和实际计时。2–5 s 40k fork 的成本低于建立并验证三种 small-N scaling，而且避免
缩放改变 in-degree、noise 和 basin。固定 connection seed 1；不以结果切换 N。

### 5.2 snapshot/fork 矩阵

共享相同网络、noise state 和 frozen coordinates，逐条 5 s：

```text
A-low   healthy d, low h,  ell_X=0
A-high  healthy d, high h, ell_X=0
B       susceptible d, low h,  ell_X=0
C       susceptible d, high h, ell_X=0
D       susceptible d, high h, ell_X>ell_X,off
D-low   matched susceptible d, low h, same ell_X
```

`healthy/susceptible/h_high/ell_X,off` 全由 R2 映射，不从 fork 结果调整。空间场按 spec 锁定：d 使用
既有 `p_i`；h_low/high 使用 R1 baseline/established-high post-X gA 模板并只匹配 reduced 均值；X load
使用 LC1 established-high relay 模板。缺模板就做同配置 sensor-only replay，不以 uniform field 代替。

### 5.3 判读

- A-low/A-high 都回 low；
- B 保持 low；
- C 在 1 s settling 后至少持续 2 s finite high；
- D 回 low，D-low 仍 low；
- C high 非 runaway、非 clip、非 refractory ceiling，local gain 非零；
- B/C 分离大于同一 basin noise variability。

这些是 basin probe，允许 high-H initialization；不得称 spontaneous lifecycle。

输出逐 arm summary、state snapshot 和 `r3_frozen_forks/fork_verdict.json`。R2 positive 但 R3 不复现，
判 `H_LOOP_NO_BISTABILITY` 或 `X_HAS_NO_OFFSET_AUTHORITY` 时必须注明是 SNN mapping failure。

---

## 6. R4 —— X protection 解析锁定与 seed1 dynamic lifecycle

### 6.1 tau_X_up 不扫 5/10 s

从 R3 获取 `ell_X,0/ell_X,release`。做一条 X-clamped-low、dynamic-Z recovery fork，测
`T_Z^upper95`。按 Core spec 计算：

```text
tau_X_up_min = T_Z_upper95 / ln(ell_X0_lower95 / ell_Xrelease_upper95)
tau_X_up = 1.10 * tau_X_up_min
```

最长允许保护窗在运行前锁为 20 s；推导值更长或分母非正，写
`OFFSET_POSITIVE_RECOVERY_NEGATIVE`。`tau_x_down=1000 ms` 与 LC1 sensor 参数不扫。

### 6.2 Z entry

用 H-off pre-Z inhibitory trace 和 R2 `d_on` 解析 I_th。目标：前 8 s 不跨，锁定最长 trajectory
内跨越。最多两个验证点；禁止 q50/q75 周围扩网格。

### 6.3 nominal trajectory

M=0、无 kick、无 reset、无 parameter step，seed1。最长时窗在 manifest 先锁，至少能容纳：

```text
>=8 s interictal
spontaneous onset
bounded H high state
X-mediated offset
postictal protection
>=8 s returning-IED assessment
```

### 6.4 C0–C4

1. C0：前 8 s event rate/IEI CV/duration/participation 与 RC1 accepted band 相容；
2. C1：dynamic Z 无 kick onset；同 snapshot Z-frozen 不进入；
3. C2：high finite、非 ceiling，持续到 X offset；不要求患者频谱；
4. C3：同 onset snapshot 的 X-off 明显延长或到 cap；
5. C4：post returning IED observable distance 落入 pre-vs-pre bootstrap band。

Recovery primary 只含 event rate、IEI、duration、participation、vSEEG energy。`h/ell_X/z` 单独报告，
不混成总分。永久静默、固定节律和快速复燃均失败。

C0–C4 全过输出 `CORE_LIFECYCLE_CANDIDATE`；否则按唯一承重失败映射到 Core spec 的七个标签。

---

## 7. R5 —— 两个预锁 replication seeds

在 R4 开跑前就把两个 replication connection/noise seeds 的 ID 和 hash 写入 sealed manifest，但不运行、
不参与选择。R4 candidate 后才依次执行；所有 H/X/Z 参数、时窗、bar 和判据冻结。

两个 seed 均通过 C0–C4，输出 `CORE_LIFECYCLE_REPLICATED`。任一个失败，保留 candidate 层结论并
明确 replication negative；不得返回 seed1 调参。

这不是患者 phenotype confirmation，只是 Core mechanism replication。

---

## 8. R6 —— 图、归档与停止

最多四张诊断图：

1. `h_sensor_separability.png`；
2. `h_x_reduced_geometry.png`；
3. `frozen_basin_forks.png`；
4. `core_lifecycle.png`（仅 R4 有真实输入后）。

图实际生成后写 `figures/README.md`，每张2–4句中文并含 `**关注点**：`；失败阶段不画占位。

归档：

```text
docs/archive/topic4/sef_hfo/fcxr_lc2_core_<terminal_label>_2026-08-XX.md
```

只在 `CORE_LIFECYCLE_REPLICATED` 后，把 Phenotype spec 从 locked-out 改成 design candidate；仍需单独
审阅和 plan，不自主继续 M/E1146/spatial。

---

## 9. 资源、OOM 与 nohup

### 9.1 worker

- 每类 run 先单实例测 peak RSS 和 wall/sim-second；
- 40k `T>=20 s` 严格 1 worker；
- 40k 2–5 s fork 默认最多 2 workers，且只有
  `MemAvailable >= 96 GiB + 2*1.35*measured_RSS`、无重负载 sibling 才开第二个；
- 线程全部钉死为1：OMP/OpenBLAS/MKL/NUMEXPR；
- sensor trace 只存预锁4096 cells，禁止全40k时间矩阵。

### 9.2 swap/进程

- stage baseline swap `+256 MiB`：停止提交；
- `+512 MiB` 且继续升：只终止本任务最新 worker；
- RSS 超 smoke 1.5倍：终止本 worker并标 resource failure；
- 不 kill/renice/改 sibling worktree/process。

### 9.3 detached

超过10分钟必须 `setsid nohup`，配 stage `flock`、launcher PID、RUNNING/DONE/FAILED sentinel、wall
guard 和 `resource_log.jsonl`。等待按 PID/sentinel，不用 `pgrep -f`。

---

## 10. 测试、提交与最终汇报

每阶段至少：相关新测试 + 现有 MZ/RC1/LC1 回归 + blessed sha256 + `git diff --check`。建议提交：

1. `docs: replace LC2 waterfall with core discovery contract`
2. `test: lock LC2 H dataflow and parity`
3. `feat: add post-relay local H vertical slice`
4. `feat: adjudicate LC2 H sensor separability`
5. `feat: map LC2 H/X reduced geometry`
6. `feat: run frozen 40k basin forks`
7. `feat: test dynamic LC2 core lifecycle`
8. `docs: archive LC2 core terminal verdict`

终报必须逐项回答：完成到 R 几、七标签之一、H sensor 区间、rho/branch、四组 forks、C0–C4、seed
replication、tests/hashes、RSS/swap/worker、nohup/sentinel/残留、artifact/figure/README、commit stack、
允许/禁止 claim。合法 NO-GO 是完成，不得为填满运行时间扩展到 Phenotype。

---

## 11. 开跑前 checklist

- [ ] Core spec/plan 同步签核；旧 LC2 两文件保持 superseded；
- [ ] post-X gA 数据流测试存在；
- [ ] `-g_Xx` 已从 reduced equation 删除；
- [ ] M/E1146/spatial 不在 Core gate；
- [ ] sensor replay configs/sha/采样 cells 在结果前锁定；
- [ ] tau/theta/k/rho 解析规则冻结；
- [ ] replication seeds 在 R4 前 sealed；
- [ ] 40k 资源和 nohup 合同可执行；
- [ ] spec/plan 无冲突。

未全过，不启动 R0 之后的仿真。
