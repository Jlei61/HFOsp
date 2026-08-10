# FCXR-LC5 实施计划：逐细胞 episode-load / pump

日期：2026-08-10

状态：**IMPLEMENTATION PLAN LOCKED FOR U0–U2。U3/U4 条件性，U2 未过不得执行。**

对应 spec：`docs/superpowers/specs/2026-08-10-topic4-fcxr-lc5-per-cell-episode-pump-design.md`

## 1. Definition of done

本轮初始实施完成不等于 lifecycle 成功。U0–U2 的完成条件是：

1. LC4e/f 口径和图修订落档；
2. 新 `rectified_excess` U current 与历史 pump 行为被测试隔离；
3. 一条 no-kick natural-entry exact capture 同时产生三个 tau 的 sensor-only load state；
4. 完成严格匹配的 pump-off + 3×3 high-state fork，并对最多两个候选完成 onset+4 s / late visited-state 复核；
5. 按 spec §8 输出标签、判决、图、README、资源日志和 archive；
6. 只在 U2 与移动状态复核都通过时写出 `U3_AUTHORISED.json`；否则 STOP。

## 2. 开工前审计

### T0.1 工作区与 provenance

```bash
git status --short
git branch --show-current
git log -1 --oneline
git worktree list --porcelain
```

必须记录：HEAD、六个 blessed hash、`src/snn_engine/mz_slow_vars.py` hash、LC4f raw artifacts hash、per-cell separation artifact hash。现有用户文件 `scripts/nohup_subject_capture.sh` 不纳入本 sprint、不得修改或提交。

### T0.2 上游 closeout

确认以下措辞已落盘并重画：

- LC4e = closed-loop shared, not cumulative-dose matched；
- LC4f 0.380 = archived late-bout reference, not universal boundary；
- current X implementation closed, X family not globally falsified；
- lifecycle not established。

### T0.3 历史 pump 审计

必须逐条核对：

- `src/topic4_mz_fcxr_pump.py::step_spike_load`；
- `src/snn_engine/mz_slow_vars.py::_pump_step/_pump_excess_E`；
- snapshot / load intervention / causal order tests；
- 2026-07-27 `FAIL_BASELINE` 是 signed-centered actuator 的结果；
- constitutive Na/K line 的 `UNRESOLVED_CALIBRATION` 不等于本 U family 的机制 NO-GO。

产物：`u0_lineage_audit.json`。

## 3. T1 — TDD：新增 rectified excess，不改历史语义

### T1.1 API

在非 blessed `MZSlowVarsConfig` 新增显式模式，例如：

```python
pump_excess_mode: Literal["signed_centered", "rectified_excess"] = "signed_centered"
```

默认值必须保持历史行为；LC5 runner 必须显式写 `rectified_excess`。不允许把旧 `_pump_excess_E()` 静默改成 rectified。

### T1.2 必须先失败再实现的测试

1. `Phi(u)<p0` 时 rectified current 精确为 0；
2. `Phi(u)>p0` 时为 `Imax*(Phi-p0)`；
3. signed historical mode 仍返回负 excess；
4. clearance 与 membrane current 使用同一个 `Phi`；
5. membrane 使用 pre-step load，spike 从下一步生效；
6. E-only current，I cells 不变；
7. `use_pump=False` full-engine byte parity；
8. state snapshot/load restore 不丢 `u_E`；
9. invalid mode fail loudly；
10. JSON sanitizer 覆盖 `numpy.bool_`、`numpy.float*`、小 ndarray、Path；
11. atomic bundle 任一文件校验失败时不发布半套正式产物；
12. H source 仍是有效 recurrent `gErec_raw`，U 不直接 reset H。

### T1.3 多 tau observer

实现一个纯 observer，在同一 spike stream 上同时更新 tau 3/8/15 s 的三个 per-cell u field。它不得改变膜、RNG draw 或 event detection。优先放新模块，不给 blessed engine 增加分支；若必须挂到 `MZSlowVars.step`，需 off-by-default parity test。

T1 产物：`u1_engine_contract.json`。

## 4. T2 — 解析 calibration，先锁数再跑

从已归档 pump-off high-state / per-cell separation 读取候选 rate 只做 preflight；最终 `r_hi_ref` 必须来自 T3 fresh capture。

锁定函数：

```text
a_load(tau) = 0.5 / (r_hi_ref * tau)
Imax(tau, Gamma) = Gamma * median(I_EE_force) / median([Phi(u)-p0]+)
```

单位测试必须用 Hz↔ms 显式换算。输出每个 tau 的：

- `a_load`；
- estimated per-cell equilibrium activation q50/q95/q99/max；
- divergent fraction；
- interictal activation/current leakage prediction；
- 9 个解析 Imax。

任何 `q99≥0.90`、divergent fraction >0、分母≤0、Imax 非有限，记 `U_SCALE_NOT_IDENTIFIABLE` 并停止，不得换 median 为 mean。

## 5. T3 — U1 no-kick exact capture

### T3.1 启动

单独 stage：`u1_capture`。只允许 1 个 40k worker，`setsid nohup` 启动。运行前写：

```text
u1_capture.pid
U1_RUNNING.json
resource_log.jsonl
```

模拟采用 accepted LC4f nominal entry，pump actuator off；多 tau observer 只读。保存至少 onset、onset+1s、+4s、late 三个 exact states。若 22 s 没有 onset，写 `U1_ENTRY_NOT_REPRODUCED.json` + FAILED/STOP。

### T3.2 Prefix validation

与 accepted no-pump reference 比较：

- external input hash；
- first onset；
- pre-onset event count/ledger；
- rate/activity prefix；
- Z/H traces。

只读 observer 必须逐位不改变主轨迹；否则 `OBSERVER_CONTAMINATES_TRAJECTORY`。

### T3.3 Artifact transaction

发布前必须同时存在并校验：

```text
u1_capture_traces.npz
u1_capture_summary.json
u1_event_ledger.json
states/onset.pkl
states/onset_plus_1s.pkl
states/onset_plus_4s.pkl
states/late.pkl
u1_noise_provenance.json
U1_DONE.json
```

正式文件用 atomic rename；禁止 NPZ 成功、JSON 失败后靠手工重建冒充一次完整事务。

## 6. T4 — 回填 candidate lock

T3 完成后才运行。用 fresh `r_hi_ref` 和 `I_EE_force` 回填：

```text
candidate_lock.json
```

内容包括 equation hash、3 个 tau、3 个 Gamma、每格 a_load/Imax、p0 derivation hash、source state hash、noise hash、所有 gate。lock 后修改代码或数值必须 fail loudly。

## 7. T5 — U2 3×3 high-state fork

### T5.1 运行顺序

先跑 pump-off control；它若不能维持 8 s high carrier，整个 source snapshot 作废。

其后顺序固定，避免只看“最好点”：

```text
tau 8s: Gamma 0.10, 0.25, 0.40
tau 3s: Gamma 0.10, 0.25, 0.40
tau 15s: Gamma 0.10, 0.25, 0.40
```

每条从同一 onset+1s exact state fork；D/Z frozen、H dynamic、X=1、M=0、外源输入相同。严格单 worker、每格独立 DONE sentinel。早停只允许：numerical failure；或已连续低态 ≥2 s 后完成 post-offset observation。

### T5.2 每格最小输出

- rate / activity fraction；
- `u/Phi/I_U` q10/q50/q90/q99；
- H、`I_EE_force`、I/E balance；
- offset candidate 与 1 s guard；
- refractory fraction、clip、tau_eff；
- exact source/noise/code hashes；
- label 与 label reason。

### T5.3 判决器坏数据回归

在读真数据前，合成测试必须区分：

- sustained high；
- rate-only suppression；
- genuine offset；
- burst-silence loop；
- immediate over-suppression；
- numerical collapse。

并重放历史 LC4f negative，确保不会把 persistent high 判成 offset。

## 8. T6 — 聚合、图、人工轨迹检查

生成：

```text
u2_authority_map.json
u2_candidate_verdict.json
figures/u2_episode_pump_authority.png/.pdf
figures/README.md
```

图至少四 panel：

1. 3×3 label map；
2. pump-off、正候选或最接近候选的 rate；
3. `I_U → I_EE_force → H` 时间顺序；
4. interictal/high/post-offset per-cell activation distributions。

必须目视检查原始轨迹。classifier 通过但图是短 trough，人工改为 `UNRESOLVED_CLASSIFIER_CONFLICT`，不得强判 GO。

## 9. T7 — U2 gate、移动状态复核与 stop

先按 spec §8.2 原样判定。若通过，最多两个候选各自在 onset+4 s 与 late exact snapshots 上复核；D/Z 冻结各自状态，H 动态，其他合同与 T5 相同。只有 primary 在三个 visited states 都 `BOUNDED_OFFSET` 才写：

```text
U3_AUTHORISED.json
```

否则写对应 STOP：

```text
U_AUTHORITY_NO_GO_IN_CURRENT_H.json
NO_ROBUST_U_WINDOW.json
H_BYPASS_OR_CARRIER_MISALIGNMENT.json
FAST_ADAPTATION_LIKE_NOT_EPISODE_OFFSET.json
U_AUTHORITY_NOT_ROBUST_TO_VISITED_SLOW_STATE.json
```

任一 STOP 后不得自动加大 `Gamma`、延长 grid、打开 M、改 H 或跑 70 s。

## 10. T8 — 条件性 U3 自然 lifecycle

仅在 `U3_AUTHORISED.json` 存在且 hashes 与 HEAD 一致时执行。最多 primary + 一个 sensitivity，每条 70 s，no kick，机制 t=0 始终在线。

先 development noise/connection；通过完整 lifecycle gate 后才运行独立 connection seed。任何参数选择发生后，confirmation seed 只确认，不回调参数。

输出：

```text
u3_lifecycle_summary.json
u3_pre_ictal_events.json
u3_post_ictal_events.json
u3_rest_distance.json
figures/u3_lifecycle.png/.pdf
figures/README.md
```

最终判决必须分开：entry / bounded carrier / offset / postictal / D recovery / returning IED distribution。缺一项即不得称完整 lifecycle。

## 11. T9 — 条件性 U4 M morphology

不在 U0–U2 自动执行范围。只有 U3 完整闭环后另行锁一个小 plan；禁止在本计划中顺手打开。

## 12. 测试矩阵

至少运行：

```bash
pytest -q tests/test_topic4_mz_fcxr_pump.py
pytest -q tests/test_mz_slow_vars.py
pytest -q tests/test_topic4_fcxr_lc4*.py
pytest -q tests/test_topic4_fcxr_lc5*.py
```

再运行所有 MZ/FCXR 相关回归。full-repo pytest 若因 gitignored artifact 在 worktree 缺失而失败，必须逐项归因，不能用“多数通过”覆盖真实代码失败。

## 13. 资源与 nohup

- T≥20 s 仿真严格 1 worker；短 fork 也默认 1 worker；
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`；
- 启动前记录 MemAvailable、swap、sibling PID/命令；
- swap delta ≥256 MiB 停止新提交，≥512 MiB 终止最新 worker并写 RESOURCE_STOP；
- 每 stage 使用独立 flock，不使用 `pgrep -f` 判断自身；只按登记 PID；
- wall-kill：capture 2 h、每个 8 s fork 90 min、70 s lifecycle 4 h；超时先 checkpoint 再终止；
- 网络断开后 `setsid nohup` 必须继续；结束后 DONE/FAILED sentinel 能由新会话读取；
- 不轮询占用前台会话；检查时读 sentinel/resource log。

## 14. 提交边界

建议提交顺序：

1. `docs(topic4-lc4): tighten LC4e/f closeout claims`
2. `docs(topic4-lc5): lock per-cell episode-load design and plan`
3. `test(topic4-lc5): pin rectified load-current contract`
4. `feat(topic4-lc5): add episode-load observer and atomic artifacts`
5. `feat(topic4-lc5): capture exact natural-entry source state`
6. `feat/topic4-lc5`: run and adjudicate U2 authority screen`

未经用户授权不 push/merge/rebase。不得提交用户的 `scripts/nohup_subject_capture.sh`。
