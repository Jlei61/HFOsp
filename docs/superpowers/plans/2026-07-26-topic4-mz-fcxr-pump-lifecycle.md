# FCXR pump-lifecycle P0–P2 implementation plan

日期：2026-07-26

状态：**IMPLEMENTATION PLAN；本文件不授权立即启动长仿真**

设计来源：
`docs/superpowers/specs/2026-07-26-topic4-mz-fcxr-pump-lifecycle-design.md`

## 0. 目标与不可越界范围

在锁定的 E1146 / L=20 / N=40000 FCXR-HEO substrate 上，只新增：

```text
per-cell dimensionless spike load u_i
→ nonlinear pump activation phi(u_i)
→ baseline-compensated electrogenic outward current
```

本 plan 只执行 P0–P2：

```text
P0 = Gate I-a instrument validity + non-blocking Gate I-b diagnostic
P1 = frozen topology + branch-conditioned slow flow
P2 = dynamic causal lifecycle + spatial preservation + empirical readout
```

执行依赖不是按章节编号串行：

```text
hard path:
Task 1 → 2 → 3 → 5 → Task 7/I-a → 8 → 9 → 10 → 11 → 12

diagnostic side path:
Task 6 → Task 7/I-b                       # 不阻塞 hard path

empirical side path:
Task 4 → Gate E target/holdout            # 可并行，只在 Gate E 使用
```

不执行：

- `M` waveform shaping；
- `X` relay depletion；
- global/area feedback；
- drive/connectivity/cooperative-gain rescue；
- exact 40k full-state Floquet；
- P3/P4 roadmap。

## 1. 工作区与文件边界

### 1.1 新 worktree

执行时从当前已推送的
`codex/topic4-mz-fcxr-heo1`
创建：

```text
branch:   codex/topic4-mz-fcxr-pump-lifecycle
worktree: /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-fcxr-pump-lifecycle
```

开始前记录：

- base commit；
- `git status --short`；
- `git worktree list --porcelain`；
- 6 个 blessed engine hashes；
- sibling 40k jobs；
- `MemAvailable` / swap baseline。

### 1.2 预计改动

修改：

- `src/snn_engine/mz_slow_vars.py`
- `tests/test_mz_slow_vars.py`

新建：

- `src/topic4_mz_fcxr_pump.py`
- `src/topic4_mz_fcxr_response.py`
- `src/topic4_mz_fcxr_pump_lifecycle.py`
- `scripts/build_topic4_e1146_target_distribution.py`
- `scripts/run_topic4_mz_fcxr_pump.py`
- `scripts/plot_topic4_mz_fcxr_pump.py`
- `tests/test_topic4_mz_fcxr_pump.py`
- `tests/test_topic4_mz_fcxr_response.py`
- `tests/test_topic4_mz_fcxr_pump_lifecycle.py`

明确不改：

- `src/snn_engine/kick_probe.py`
- `src/snn_engine/lfp.py`
- `src/snn_engine/params.py`
- `src/snn_engine/model.py`
- `src/snn_engine/connectivity.py`
- `src/snn_engine/connectivity_rot.py`

若 P0 证明没有 guarded-engine edit 就无法构造合格读出，停止并报告，不自行 re-bless。

## 2. RNG 合同

当前 substrate seed 同时影响 connectivity 和运行 RNG。新 runner 必须拆开：

```text
connectivity_seed
noise_seed
initial_state_seed
perturbation_seed
```

锁定：

- development：`connectivity_seed=1, noise_seed=101`；
- baseline calibration：noise `201`；
- baseline held-out equivalence：noise `202`；
- confirmatory noise：`301,302,303`；
- connectivity seeds：`1,3`；
- initial state：当前 deterministic reset，记录 `initial_state_seed=0`；
- perturbation/shuffle：`7001`。

构建 connectivity 后，运行前显式：

```python
S["net"]["rng"] = np.random.default_rng(noise_seed)
```

paired counterfactuals 从 t=0 重跑；干预前 trajectory prefix 必须 hash-identical。

## 3. Task 1 — 纯 pump 数学与 TDD

### Files

- Create `src/topic4_mz_fcxr_pump.py`
- Create `tests/test_topic4_mz_fcxr_pump.py`

### Tests first

1. `pump_activation(u,h=3)=u^3/(1+u^3)`，非负、有限、单调、光滑；
2. `u=0 → phi=0`；
3. spike jump 不乘 `dt`；
4. clearance 乘 `dt/tau_N`；
5. update 后 `u≥0`；
6. zero-spike state monotonically clears；
7. same `phi` 同时驱动 clearance 与 membrane pump；
8. excess current 是 `Imax*(phi-p0)`，**没有 positive part**；
9. `phi=p0 → excess=0`；
10. `phi<p0` 允许 negative excess compensation；
11. `h` primary 只允许 3；
12. raw recurrent conductance 不出现在 primary load function。

### Implement

Pure functions：

```python
pump_activation(u, h=3)
step_spike_load(u, spike_mask, *, a_load, tau_N, dt, h=3)
excess_pump_current(u, p0, *, Imax, h=3)
```

### Verify

```bash
pytest -q tests/test_topic4_mz_fcxr_pump.py
```

无仿真。

## 4. Task 2 — `MZSlowVars` pump plugin

### Files

- Modify `src/snn_engine/mz_slow_vars.py`
- Modify `tests/test_mz_slow_vars.py`

### Config

新增 off-by-default fields：

```text
use_pump=False
pump_sensor_only=False
pump_a_load=0.0
pump_tau_ms=0.0
pump_Imax=0.0
pump_h=3
pump_p0_E=None
pump_record_calibration=False
pump_interventions=None
```

state：

```text
u_pump_E             shape (NE,)
pump_phi_sum_E       shape (NE,), calibration only
pump_phi_count
trace_u_mean/max
trace_phi_mean/max
trace_pump_excess_mean/max/min
pump snapshots       landmark only
```

### Tests first

1. `use_pump=False` full engine byte-parity；
2. pump only acts on E cells；
3. sensor-only updates `u` but membrane output unchanged；
4. `Imax>0` requires finite `p0_E.shape==(NE,)`；
5. membrane uses pre-step `u(t-)`；
6. spike update occurs in `slow.step` after membrane；
7. p0 compensation is applied as `-Imax*phi + Imax*p0`；
8. no rectification at `phi-p0=0`；
9. calibration observer does not change spikes；
10. pump intervention off path byte-parity；
11. scheduled current knockout leaves u dynamics active；
12. scheduled load reset sets `u` to supplied baseline field；
13. scheduled load injection changes u only at registered step；
14. default M/Z/X behavior unchanged；
15. `test_existing_ZMX_update_order_unchanged()`：既有 Z/M/X membrane→spike→slow-update 顺序和逐步值不变；
16. snapshot stores only selected `u_E` vectors，不存 `N_cell×T`。

### Implementation

- 在 `membrane_terms` 的 E-cell drive 中减去 excess pump current；
- 在 `step(spk,labels,dt)` 中按锁定因果顺序更新 `u_pump_E`；
- `mz_slow_vars.py` 仅作为 off-by-default plugin container；不得重排、合并或改写既有 Z/M/X update path；
- scheduled intervention 用 integer steps，不用浮点时间比较；
- intervention schedule 只由 runner 构造；
- 保留 `M=off`、`X=1`。

### Verify

```bash
pytest -q tests/test_mz_slow_vars.py tests/test_topic4_mz_fcxr_pump.py
pytest -q tests/test_mz_full_conductance_spatial_relay.py \
          tests/test_topic4_mz_fcxr_heo1.py \
          tests/test_topic4_mz_fcxr_heo2.py \
          tests/test_topic4_mz_fcxr_heo3.py
```

仍不启动 40k。

## 5. Task 3 — Baseline calibration、shrinkage 与 equivalence

### Files

- Extend `src/topic4_mz_fcxr_pump.py`
- Extend `tests/test_topic4_mz_fcxr_pump.py`
- Create `scripts/run_topic4_mz_fcxr_pump.py` with stages `p0-baseline`, `p0-equivalence`

### Pure functions

```python
rate_decile_groups(baseline_rate_E)
fit_p0_shrinkage(phi_block_means, spike_counts, groups)
apply_p0_shrinkage(raw_p0, group_p0, weights)
block_equivalence_margins(baseline_blocks)
evaluate_baseline_equivalence(off_metrics, on_metrics, margins)
required_ied_count(block_metrics, minimum=20)
```

### Shrinkage contract

- calibration noise seed `201`；
- baseline 分多个非重叠 blocks；
- rate-decile grouping 仅使用 pump-off baseline；
- shrinkage strength 只在 calibration blocks 内以预注册的 inner block-CV prediction error 选择；
- 不使用 source/sink/axis labels；
- p0 raw/group/shrunken 全保存；
- equivalence margin 只由 calibration pump-off block-to-block variability 定义，并在查看 final held-out pump-on 结果前落盘；
- final held-out equivalence 使用 noise seed `202`；该轨迹不得参与 grouping、shrinkage strength、equivalence margin 或 threshold 的拟合，也不得重拟合。

### Cheap order

1. `N≈1000`, 1–2 s smoke；
2. `L=20`, 1 s plumbing smoke；
3. `L=20` event-count-driven sensor-only calibration；
4. pump-off/pump-on held-out equivalence。

### Gate I-a baseline

所有 primary baseline metrics 落入 block-derived equivalence margin；否则：

- 不进入 response operator；
- 不调 lifecycle 参数抢救；
- 只允许修 baseline compensation / shrinkage。

### Artifacts

```text
baseline_variability.json
pump_baseline_calibration.json
pump_baseline_equivalence.json
p0_E.npz
```

## 6. Task 4 — Real E1146 target distribution（Gate E preparation，non-blocking）

### Files

- Create `scripts/build_topic4_e1146_target_distribution.py`
- Add pure target helpers to `src/topic4_mz_fcxr_pump_lifecycle.py`
- Add tests to `tests/test_topic4_mz_fcxr_pump_lifecycle.py`

### Inputs

先按 Epilepsiae SQL / seizure inventory 建 eligible seizure table，不复用 HEO1 的单 block 作为全集。

对每场 eligible seizure：

- local-CAR / montage 合同与现有真实 runner 一致；
- interictal baseline、onset、early、established windows；
- 六频段中 `1–80 Hz` 为主，`80–150 Hz` 描述；
- PLV / recruitment / sharpness / burst rate；
- contact-level 与 seizure-level分层保存。

### Holdout

- 按时间排序；
- 最后一场 eligible seizure 作为 holdout；
- 若 eligible denominator 不足以支持 holdout，写 `exploratory_only=true`；
- threshold/tolerance 不得读取 holdout。

本 Task 可与 P0/P1 并行，但不属于 Gate I-a 输入。target extraction 失败或 denominator 不足只使 Gate E `UNRESOLVED/exploratory`，不得阻止 Gate T/C。

### Tests

1. window 不跨 recording gap；
2. seizure 与 interictal baseline 不泄漏；
3. holdout 不进入 threshold fit；
4. contact ordering 与 model montage 对齐；
5. 80–150 Hz 不进入 primary broadband count；
6. synthetic 3 Hz spiky broadband target 通过；
7. narrow 16 Hz state 不通过。

### Artifact

`real_target_distribution.json`。

## 7. Task 5 — Virtual-SEEG component audit

### Files

- Add non-blessed observer helpers to `src/topic4_mz_fcxr_pump.py`
- Extend `scripts/run_topic4_mz_fcxr_pump.py --stage p0-readout`
- Extend pump tests

### Constraint

不修改 blessed `lfp.py`。复用其 electrode weights，但在 pump observer 中在线聚合：

```text
legacy_abs
excitatory_component
inhibitory_component
pump_component
no_direct_pump
all_components
```

不保存 per-cell×time component matrices。

先做 capability audit：列出 `MZSlowVars`/runner 在不改 blessed engine 时实际可取得的逐细胞量、符号、driving-force 与采样时点。上述名字均为 virtual-SEEG proxy，不得写成物理 forward voltage。若现有 hook 不能构造带符号的 `no_direct_pump`，不得从绝对值反推符号，也不得在本 sprint 修改 blessed `lfp.py`；写出 `READOUT_NOT_IDENTIFIABLE`，Gate I-a fail，并提出独立 engine-change spec。

### Tests

1. observer off 不改变 spikes；
2. component sum identity；
3. `no_direct_pump` 不含 pump；
4. `all-no_direct == pump_component`；
5. pump current 单独的低频正弦不能让 `no_direct_pump` broadband；
6. contact weights/order 与 `LFPRecorder` 一致；
7. early-stop 后 trace 长度一致。

### Gate I-a readout

Gate E primary 指标必须在 `no_direct_pump` 上计算。若改善只出现在 `all_components`：

`READOUT_CONTAMINATION`，停止 empirical claim。

## 8. Task 6 — Empirical finite-time response operator（Gate I-b diagnostic）

### Files

- Create `src/topic4_mz_fcxr_response.py`
- Create `tests/test_topic4_mz_fcxr_response.py`
- Extend runner stage `p0-operator`

### Pure API

```python
classify_dynamical_regime(cycle_features, ic_labels, replay_labels)
build_coarse_basis(geometry, n_grid=20)
bin_observables(rate_E, rate_I, gE_eff, gI, geometry)
paired_response(delta_plus, delta_minus, epsilon)
fit_finite_time_operator(X0, X1, regularization)
operator_modes(A_delta, basis)
finite_time_svd(A_delta)
mode_projections(mode, common, axial, transverse, core_diff)
```

### Synthetic TDD

1. deterministic repeated cycles classify as `DETERMINISTIC_PERIODIC_CANDIDATE`；
2. frequency-diffusing replay-sensitive cycles classify as `STOCHASTIC_OSCILLATORY_REGIME`；
3. finite-lived high branch classifies as `METASTABLE_HIGH_ACTIVITY`；
4. known diagonal map recovers eigen ordering；
5. known nonnormal map has stable eigenvalues but gain >1；
6. common/axial/transverse basis orthogonalization；
7. `±epsilon` cancels even-order bias；
8. common random noise cancels matched noise；
9. mismatched noise fails repeatability；
10. `20×20` uniform field stays uniform；
11. mode sign ambiguity does not change projection magnitude。

### Small-network validation

- `N≈1000` required；
- `epsilon` linearity at three amplitudes；
- repeated noise replay；
- compare `20×20` vs coarser basis；
- `N≈4000` only if mode ordering ambiguous。

### 40k run

Landmarks：

- baseline/interictal；
- pre-onset；
- established 16 Hz branch；
- pre-offset candidate（P2 后补）。

Exact Floquet 只写 `eligible/not_eligible`，不在本 plan 实现 saltation tangent。

40k coarse observables 必须在 non-blessed observer 中在线做空间 binning/rolling aggregation；禁止保存 `N_cell × N_time` 全量矩阵。若某一候选 observable 在当前 hook 不可识别，删去该分量并在 operator artifact 记录，而不是用未验证 proxy 替代。

该 Task 的失败不阻止 P1/P2 lifecycle；失败时写 `gate_Ib.status=UNRESOLVED/FAIL`，撤回 response-mode/eigenmode claim 和最终图右侧 susceptibility panel。

### Artifacts

```text
finite_time_operator_smallnet.json
finite_time_operator_40k.json
operator_landmark_fields.npz
dynamical_regime_classification.json
```

## 9. Task 7 — Gate I-a / I-b 分层 adjudicator

### Files

- Add `adjudicate_gate_Ia()` and `adjudicate_gate_Ib()` to `src/topic4_mz_fcxr_pump_lifecycle.py`
- Add tests
- Runner stage `p0-adjudicate`

### Gate I-a required inputs

- pump-off parity / existing ZMX update-order contract；
- baseline calibration/equivalence；
- readout audit。

### Gate I-b diagnostic inputs

- small/40k operator repeatability；
- dynamical-regime / exact-Floquet eligibility。

### Output

```text
gate_Ia.json
status ∈ {PASS, FAIL_PARITY, FAIL_UPDATE_ORDER, FAIL_BASELINE,
          FAIL_READOUT_IDENTIFIABILITY, FAIL_READOUT_CONTAMINATION,
          UNRESOLVED}

gate_Ib.json
status ∈ {PASS, FAIL_OPERATOR, UNRESOLVED, NOT_RUN}
```

只有 `gate_Ia.status=PASS` 解锁 P1。Gate I-b 不阻塞 lifecycle，只控制 response-mode claim。

## 10. Task 8 — Activity-shaped frozen `Z×P` topology maps

### Files

- Extend `src/topic4_mz_fcxr_pump.py`
- Extend runner stages `p1-sensor-field`, `p1-map`
- Extend tests

### Sensor-only field

`Imax=0`，从 established high branch 记录：

```text
u_high_0p5s
u_high_1s
u_high_2s
u_high_3s
u_baseline
```

构造：

```python
u_rho = u0 + rho_u * (u_high - u0)
p_excess = phi(u_rho) - p0
```

`rho_u` 只是 field-construction parameter；所有 branch map、slow-flow 和 figure 的正式横轴均使用 `mean_excess_pump_activation`，不得标成 raw `N/u`。

### Maps

每个 field 做：

- activity-shaped；
- mean-matched uniform；
- value-matched spatial shuffle。

轴：

```text
rho_Z
mean_excess_pump_activation
```

IC：

- low；
- high；
- kick-release only for basin mapping。

seed1 discovery；Gate T 候选出现后才用 seed3 边界确认。

### Stop

若 pump 同时移除 low/high，或 high 永不被选择性移除，立即 `TOPOLOGY_NO_GO`。

## 11. Task 9 — Branch-conditioned slow flow

### Files

- Add slow-flow functions to `src/topic4_mz_fcxr_pump_lifecycle.py`
- Extend tests
- Runner stage `p1-slow-flow`

### API

```python
branch_conditioned_flow(z_trace, pump_activation_trace, branch_mask, dt)
project_load_field(u_minus_u0, spatial_basis)
low_safe_after_reset(z_state, frozen_map)
adjudicate_gate_T(topology, slow_flow)
```

### Tests

1. synthetic closed excursion flow passes；
2. monotone Z countdown fails stationarity；
3. high-branch flow away from exit fails；
4. pump release outside low-safe region fails；
5. uniform and shaped fields with same mean remain distinguishable；
6. no forced Hopf/fold label。

### Artifact

```text
frozen_topology_map.json
branch_slow_flow.json
gate_T.json
```

只有 Gate T PASS 解锁 P2。

## 12. Task 10 — Dynamic stationarity 与 onset controls

### Files

- Extend runner stages `p2-stationarity`, `p2-development`
- Extend lifecycle pure module/tests

### Stationarity

burn-in 直到：

- 至少预锁 `N_IED`；
- block rate/IEI/Z/u 无显著单调趋势；
- observation start 不决定 onset time。

### Declustered control

记录 baseline/rare-cluster 的 Z sensor drive；构造保持 mean 与 marginal distribution、破坏 cluster ordering 的 block-shuffled replay。比较：

- onset probability；
- onset latency；
- slow-state excursion。

若 no-kick onset 仍是固定倒计时：

- 标签改为 `AUTONOMOUS_SLOW_DRIFT`；
- 不写 spontaneous；
- 可继续测试 pump termination，但不能过 spontaneous-onset 子门。

### Parameter choice

- 从 Gate T 反推 `Imax`；
- 从 load accumulation / release 反推 `a_load,tau_N`；
- dynamic Z 候选必须先过 stationarity；
- one-axis `±20%`，不做 Cartesian grid。

## 13. Task 11 — Causal counterfactuals

### Files

- Extend `MZSlowVars` scheduled interventions
- Extend runner stage `p2-causal`
- Extend tests

### Intervention implementation

每个 paired run 从 t=0 重放相同 noise：

```text
control
pump_current_knockout
preoffset_load_reset
preoffset_load_sufficiency_injection
postoffset_load_reset
termination_combined
termination_pump_only_z_frozen
termination_z_only_pump_disabled
termination_neither
uniform_matched_injection
shuffle_matched_injection
```

四个 termination-decomposition 臂从同一个 established-high snapshot 重放：

- `combined`：native Z recovery + native pump；
- `pump_only_z_frozen`：Z 固定在 ictal/permissive field，pump 保持动态；
- `z_only_pump_disabled`：Z 正常恢复，`u→u0` 且 pump current 关闭；
- `neither`：Z 固定、`u→u0`、pump current 关闭。

这些臂不扫新参数；需要新增 `freeze_z_field` scheduled intervention，并验证 freeze 只停止 Z evolution、不改变 pump/noise/spikes 的 intervention 前 prefix。

干预前 hash：

```text
rate prefix
spike-count prefix
mean Z/u prefix
landmark state checksum
```

必须一致。

### Tests

1. all four decomposition arms have identical pre-intervention prefix hashes；
2. `pump_only_z_frozen` keeps the supplied Z field bitwise fixed while `u/pump` continues；
3. `z_only_pump_disabled` has zero pump membrane contribution while Z evolves；
4. `neither` freezes Z and removes pump without changing future noise replay；
5. synthetic offset times classify pump-dominant / cooperative / Z-dominant / non-identifiable correctly。

### Causal gate

- knockout 延迟/阻止 offset；
- preoffset reset 延迟 offset；
- preoff field 提前注入足以提前 termination；
- postoffset reset 缩短 postictal suppression；
- shaped/uniform/shuffle 区分空间承重。
- 四臂给出 `PUMP_DOMINANT_EXIT` 或 `COOPERATIVE_Z_PUMP_EXIT`；
- 若为 `Z_DOMINANT_EXIT`，pump termination claim fail；若 neither 也 offset，则该 snapshot 的因果分解记为 `NON_IDENTIFIABLE`。

如果只有时间先后、反事实失败：

`TEMPORAL_ASSOCIATION_ONLY`，Gate C fail。

## 14. Task 12 — Statistical return、holdout 与 Gate S/E

### Files

- Extend pure lifecycle module
- Extend runner stages `p2-confirm`, `p2-adjudicate`
- Extend tests

### Statistical return

- baseline pilot 决定 `N_IED≥20`；
- block equivalence margins；
- late recovery event-count-driven；
- time-block bootstrap；
- no-event tail 永远不能算恢复。

### Holdout

锁参后：

```text
connectivity {1,3}
× noise {301,302,303}
```

不得用 holdout 调参。

### Gate S

检查：

- non-simultaneous ignition；
- early axis preference；
- interictal/early-ictal scaffold alignment；
- recovery 后 forward/reverse probe；
- `U_parallel/U_perp`；
- common/axial/transverse response；
- shaped vs uniform/shuffle。

### Gate E

用 real holdout：

- 9D trajectory；
- 3–8 Hz sharp burst；
- 1–80 Hz broadband；
- phase/recruitment sequence；
- `no_direct_pump` primary。

### Outputs

```text
stationarity_gate.json
lifecycle_verdict_*.json
causal_counterfactuals_*.json
spatial_preservation_*.json
holdout_summary.json
gate_C.json
gate_S.json
gate_E.json
candidate_verdict.json
```

## 15. Task 13 — Figures、STATUS 与归档

### Files

- Create `scripts/plot_topic4_mz_fcxr_pump.py`
- Write results `figures/README.md`
- Write archive under `docs/archive/topic4/sef_hfo/`

### Figures

1. instrument + baseline equivalence；
2. real target distribution；
3. virtual-SEEG component decomposition；
4. finite-time response modes；
5. frozen topology + slow-flow arrows；
6. lifecycle + paired causal traces；
7. statistical-return equivalence；
8. spatial scaffold preservation。

只有 `I-a+T+C+S` 通过才生成 lifecycle candidate；只有再过 E 且 I-b 支持 response-mode claim，才生成包含右侧 susceptibility panel 的 paper-ready 四栏图。

### STATUS wording

分层写：

```text
engineering
Gate I-a
Gate I-b
Gate T
Gate C
Gate S
Gate E
```

不得把上游 gate PASS 自动传递为下游 scientific PASS。

## 16. Runner / nohup / OOM

统一入口：

```bash
python scripts/run_topic4_mz_fcxr_pump.py --stage <stage> --confirm-run
```

stage：

```text
p0-baseline
p0-equivalence
p0-readout
p0-operator
p0-adjudicate
p1-sensor-field
p1-map
p1-slow-flow
p2-stationarity
p2-development
p2-causal
p2-confirm
p2-adjudicate
```

长任务：

```bash
setsid nohup env \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  python scripts/run_topic4_mz_fcxr_pump.py \
  --stage <stage> --confirm-run \
  > <run_dir>/nohup.log 2>&1 < /dev/null &
```

限制：

- `T<20s` 最多 2 workers；
- `T≥20s` 1 worker；
- swap `+256 MiB` 停提交；
- swap `+512 MiB` 且增长，或 MemAvailable `<2×peak`：停自己的最新任务；
- PID/log/sentinel/resource log 必须齐；
- 不杀 sibling；
- 不保存 `N_cell×T` dense pump state。

## 17. Commit plan

建议分批：

1. `test: lock dimensionless pump and baseline compensation contracts`
2. `feat: add off-by-default per-cell pump slow state`
3. `feat: add pump baseline calibration and equivalence gate`
4. `feat: build E1146 target distribution and readout audit`
5. `feat: add empirical finite-time response operator`
6. `feat: add activity-shaped topology and slow-flow maps`
7. `feat: add dynamic lifecycle and causal counterfactuals`
8. `feat: add holdout spatial/empirical adjudication`
9. `docs: archive FCXR pump P0-P2 verdict`

每次提交前：

```bash
git diff --check
pytest -q <targeted tests>
```

最终：

```bash
pytest -q \
  tests/test_mz_slow_vars.py \
  tests/test_topic4_mz_fcxr_pump.py \
  tests/test_topic4_mz_fcxr_response.py \
  tests/test_topic4_mz_fcxr_pump_lifecycle.py \
  tests/test_mz_full_conductance_spatial_relay.py \
  tests/test_topic4_mz_fcxr_heo1.py \
  tests/test_topic4_mz_fcxr_heo2.py \
  tests/test_topic4_mz_fcxr_heo3.py
```

## 18. 最终停机判决

任何 gate fail 都允许形成合格 bounded-negative：

- Gate I-a fail：instrument / compensation 不成立；
- Gate I-b fail：撤回 response-mode claim，但不阻止 lifecycle；
- Gate T fail：没有选择性 exit 或 slow flow 不闭合；
- Gate C fail：pump 无因果 termination / memory；
- Gate S fail：时间闭环但空间 scaffold 丢失；
- Gate E fail：lifecycle scaffold 存在，但与真实 E1146 波形/轨迹不兼容。

P3/P4 永远不在本 plan 内自动解锁。
