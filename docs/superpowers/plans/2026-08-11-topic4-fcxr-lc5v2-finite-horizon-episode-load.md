# FCXR-LC5v2 finite-horizon episode-load — implementation plan

对应 spec：`docs/superpowers/specs/2026-08-11-topic4-fcxr-lc5v2-finite-horizon-episode-load-design.md`

状态：**LOCKED FOR CORE EXPLORATION**

## T0 — lineage 与新结果根

1. 读取并 hash U1a summary、sparse spikes、rate fields、recurrent force、onset exact state；
2. 验证 `u1_prefix_validation=PASS` 与 `source_type=escalating_saturated_source`；
3. 新结果根：`results/topic4_sef_hfo/fcxr_lc5v2_finite_episode/`；
4. 旧 `U_SCALE_NOT_IDENTIFIABLE` 只封锁 LC5v1 runner，不封锁独立 LC5v2 runner。

输出 `lineage_lock.json`。

## T1 — 纯离线 finite calibration

新增 `src/topic4_fcxr_lc5_finite_episode.py`：

- `replay_phi_summary(stream,tau,a,dt,windows,snapshots)`；
- `solve_a_for_window_target(...)`；
- `estimate_p0(...)`；
- `calibrate_episode_dose(...)`；
- `attach_load_to_state(...)`。

二分 bracket 从 `a=0` 开始倍增，直到 window median phi 跨过 0.5；最多 40 次，绝对 target error
`<=5e-4`。同一输入 bitwise deterministic。离线 replay 可按 step 稀疏更新，但必须与 engine 离散方程
保持同一因果顺序。执行采用 spec 锁定的 `dt_cal=1 ms`、activation sampling=5 ms；用 U1 首秒
对 `dt=0.05 ms` 原方程做误差审计，审计数值随 candidate lock 落盘。

baseline `p0` 第一版使用既有 `choose_p0_shrinkage`；若 W_B 不能形成 >=3 个非重叠块，使用预锁的
四个 1 s blocks。禁止看 U2 outcome 后换 shrinkage。

recurrent force denominator 从已保存 per-second block force 读取，W_E 使用 blocks 12、13；分子在同一
2 s 窗逐步积分。生成 `finite_episode_calibration.json`、`u_fields_tau3_8_15.npz`、
`candidate_prelock.json`。

单元测试：单调、二分、单位、window 边界、engine parity、state attachment、dose ratio、病理细胞不删。

## T2 — U2 runner 最小闭环

新增 `scripts/run_topic4_fcxr_lc5v2.py`，stages：`audit`、`calibrate`、`u2a-control`、
`u2a-gamma025`、`u2a-gamma010`、`u2a-gamma040`、`adjudicate`。

每个 U2 arm：

- 从同一 raw onset exact state 加载；
- template 开 `use_pump=True`、`rectified_excess`、锁定 `a/tau/p0/Imax`；
- 将对应 onset `u_i` 写入 state；
- Z/H 动态，X frozen 1，M off；
- 7 s，单 worker，counter/RNG state 连续；
- 保存 rate、D/H、u/phi/IU、gErec raw/effective、IEE force、event ledger 和 final state；
- 每 1 s rolling checkpoint，atomic publish。

control 必须复现原 pump-off onset prefix；不一致则 `U2_CONTROL_MISMATCH`，停止。

## T3 — 顺序判读

先 control，再 Gamma 0.25。比较：

- saturation entry time / saturation fraction；
- high dwell 与 offset；
- rate、IU、IEE force、H source/H 的时间顺序；
- achieved dose；
- numerical safety。

若 0.25 与 control 有核心分离，补 0.10、0.40；否则先判断 dose 是否实际送达，再决定是否按 spec
追加 0.60。不得默认跑满。

### T3.1 — 2026-08-12 低剂量边界定位

已有结果显示 `Gamma=0.025--0.25` 全为 `IMMEDIATE_SUPPRESSION`，且 exact-load 审计排除了
粗时间步重放污染。按同步修订后的 spec，只运行：

```text
tau8_gamma0001
tau8_gamma0003
tau8_gamma0005
tau8_gamma0010
```

四臂使用同一输入/状态/判决器并可资源安全地并行；每臂内部仍为单 worker、单线程。
完成后生成统一 branch table。出现 `CONTAINED_HIGH_NO_OFFSET` 或
`FINITE_EXCURSION_OFFSET` 即停止强度轴并进入 tau 比较；四点均无中间类则停止继续细化。

输出 `u2a_branch_map.json` 与诊断图；只有核心动力学结果出现后再跑广泛回归和确认 seed。

## T4 — 工程合同

- `mz_slow_vars.py` 如不需修改则保持 hash；若修改必须单独 mechanism hash；
- 六个 blessed engine 文件不改；
- pump-off/control exact prefix parity；
- `setsid nohup` + arm-scoped flock + PID + sentinel；
- 用户已授权 OOM-safe 并行；最多 4 个独立 U2 arm，按实测 6.8 GiB/arm 的 1.5 倍预算，
  启动时 `MemAvailable` 至少为总预算 3 倍，swap 增量门不变；
- 资源守卫在每个 simulation chunk 后执行；
- 用户文件 `scripts/nohup_subject_capture.sh` 不碰；
- 不 push/merge。

## Definition of done（本轮核心）

1. finite calibration 有唯一、可复现的 `a_U/p0/Imax`；
2. control 与 Gamma 0.25 至少完成并可比较；
3. 回答 U 是否把 escalating saturation 改成 contained/finite excursion；
4. 有核心分离才扩点；
5. 结果、资源、进程和措辞边界写入 STATUS。
