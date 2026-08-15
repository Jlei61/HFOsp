# FCXR-LC6B frozen-slow causal atlas — implementation plan

Spec: `docs/superpowers/specs/2026-08-15-topic4-fcxr-lc6b-frozen-slow-causal-atlas-design.md`
Config: `config/topic4_fcxr_lc6b_frozen_slow_atlas.json`
结果根: `results/topic4_sef_hfo/fcxr_lc6b_frozen_slow_atlas/`

## 1. Definition of done

第一批 8 条 clamp 全部完成（或按 §6 的注册规则明确判为 `RIGHT_CENSORED`），并且：

- 每条臂写出 exact-state 起点哈希、未来输入哈希、graph/config 哈希、分类标签与全部 §7 读数；
- 同一源快照的四条臂 `external_input_sha256` 实测逐位相等；
- `h_lc2_frozen_E is None` 的既有路径通过 byte-parity 回归；
- `clamp_fork_summary.json` + `run_manifest.json` + `STATUS.md` + `figures/{lc6b_clamp_forks.png,pdf}` + `figures/README.md` 齐备；
- archive doc 落地；
- 按 spec §10 的 A/B/C 自动决策产出下一步：A → 落 H-EFF/H-CAP spec/plan（不执行）；B → 进入 natural-path atlas；C → 补 onset+1 s 点。

## 2. T0 preflight

- [ ] T0.1 确认工作树除用户脚本 `scripts/nohup_subject_capture.sh` 外干净；`git diff --check` 干净；`pytest -q tests/test_topic4_fcxr_lc6*.py` 全绿。
- [ ] T0.2 确认两个源 checkpoint 存在且 `state_hash` 与 `trajectories/C0/summary.json::pinned_checkpoints` 记录一致；确认 `t` 换算出的绝对时间等于 manifest 的 `actual_ms`（13000 / 15000 ms）。
- [ ] T0.3 记录 `MemAvailable` 与 LC6A 实测单 worker peak RSS（`resource_log.jsonl::natural_arm_measured_peak_rss_gib = 6.891`），据此定并发数。

## 3. T1 引擎：H 冻结钩子（TDD）

新增 `MZSlowVarsConfig.h_lc2_frozen_E: np.ndarray | None = None`，紧邻 `h_lc2_init_E`。

实现约束：

- `membrane_terms` **一个字符都不改**；
- `step()` 里 LC2 H 的更新块条件从 `if c.use_h_lc2:` 改成 `if c.use_h_lc2 and c.h_lc2_frozen_E is None:` —— `None` 时逐字符等价；
- `__init__` 在 `h_lc2_init_E` 之后安装冻结场（与 `z_frozen_E` / `x_relay_frozen_E` 一致）；
- `_validate_config` 硬检查：1-D、shape 在 `__init__` 检查、finite、`>= 0`、必须 `use_h_lc2=True`（否则该场是静默死旋钮）。

先写测试再写实现：

- [ ] TDD-1 `h_lc2_frozen_E=None` 时，一段既有配置的多步运行与改动前 **byte-identical**（比较 `state_hash` 与 spike sha256）。
- [ ] TDD-2 H 冻结后逐步逐位不变：跑 N 步，每步后 `h_lc2_E` 的 `tobytes()` 与初值完全相同。
- [ ] TDD-3 H 冻结时 `_h_source_lc2_E` 仍在更新（source trace 不被关掉），且 `trace_h_lc2_mean` 逐点等于冻结值。
- [ ] TDD-4 `h_lc2_frozen_E` 配 `use_h_lc2=False` 抛错；shape 错、非有限、负值都抛错。
- [ ] TDD-5 D/Z 冻结后逐步逐位不变（`use_z=False` + `z_frozen_E`），并确认 `membrane_terms` 仍在用该 z 调制（不是被绕过）。
- [ ] TDD-6 D 与 H 同时冻结时两者都逐位不变。

## 4. T2 clamp helper（TDD）

新增 `src/topic4_fcxr_lc6b_clamp.py`：

```
apply_slow_clamp(state, *, clamp_d: bool, clamp_h: bool) -> FCXRLoopState
```

- 深拷贝（`clone_loop_state`），绝不写穿调用方；
- `clamp_d` → `cfg.z_frozen_E = slow.z[:NE].copy()`；`cfg.use_z = False`；
- `clamp_h` → `cfg.h_lc2_frozen_E = slow.h_lc2_E.copy()`；`use_h_lc2` 保持 `True`；
- 两个都 `False` 时返回的克隆与输入 `state_hash` 相同且 cfg 未被改动（`NAT` 臂）；
- 硬检查冻结场 shape/dtype/finite/range；
- 返回同时给出 `clamp_config_sha256`（对被改动的 cfg 字段做规范化哈希），供 runner 区分
  "config 差异" 与 "state 差异"。

**为什么不直接用 `src/topic4_fcxr_lc3_dxprobe.freeze_dynamic_state`**：它的问题是
"把 D 和 X 一起按到**选定值**做相图网格"，会连带改写 `cfg.use_x` / `x_relay_frozen_E`，
并且没有 H 这一路。本轮的问题是"把 D 和/或 H 按在**快照自身**的值上，X 一个字节都不碰"。
两句话不同 → 写新的，不复用。

- [ ] TDD-7 `clamp_d=False, clamp_h=False` → `state_hash` 不变、`use_z`/`use_h_lc2`/`z_frozen_E`/`h_lc2_frozen_E` 全部与输入相同。
- [ ] TDD-8 clamp 后原 state 未被修改（深拷贝证明）。
- [ ] TDD-9 exact checkpoint load → clamp → 续跑 → 末状态可再存再载入且哈希自洽。
- [ ] TDD-10 同一起点、只有 clamp 配置不同的两条臂，`external_input_sha256` 相等，而 `state_hash` 不等 —— 即 runner 不会把 config 差异误写成 state 差异。

## 5. T3 runner

新增 `scripts/run_topic4_fcxr_lc6b_clamp_forks.py`，stage 为 `run` / `finalize`。

`run --snapshot {S2,S4} --arm {NAT,H_CLAMP,D_CLAMP,DH_CLAMP}`：

1. 校验 execution manifest（`experiment_id = fcxr_lc6b_frozen_slow_atlas`）与 blessed engine hash；
2. `NAT._fresh_system` 建 C0 系统 → `U2.PM._seed_template` 建模板 → `U2.load_into` 载入 exact checkpoint → 校验 `state_hash`；
3. `apply_slow_clamp`；
4. 分 1 s chunk 续跑 6000 ms，`input_sink=ExactInputHasher`、`spike_sink=SparseSpikeBinaryWriter`、`membrane_term_sink=NaturalCurrentObserver`；
5. 每 chunk 后写 `progress.json` + 滚动 exact checkpoint；每 chunk 复查 mechanism source hash；
6. `AtomicStageBundle` 原子提交 `summary.json` + `spikes.npz` + `traces.npz`；
7. 新 checkpoint 按 spec §4.1 的元数据合同写出。

`finalize`：聚合 8 条臂 → `clamp_fork_summary.json` → 判 §10 的 A/B/C → 出图 + `figures/README.md` → `run_manifest.json` + `STATUS.md`。

- [ ] TDD-11 runner 的 `_classify` 在合成输入上给出 §8.3 判决树的每一个分支（8 个标签各一例）。
- [ ] TDD-12 `_classify` **不接受** D/H 轨迹作为参数（签名层面就杜绝 §8.1 的复用陷阱）。
- [ ] TDD-13 checkpoint 元数据合同的七个字段齐备且 `snapshot_time_ms` 与 `t·dt` 一致。

## 6. T4 执行

- 8 条臂，`setsid nohup` + stage-scoped `flock` + PID file + `RUNNING/DONE/FAILED` sentinel。
- 线程固定 `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1`。
- 并发 4，需满足 `4 × 6.891 GiB + margin < MemAvailable`。
- 建立 completion waiter 后**停止主动轮询**。
- 注册单次延长（spec §6.2）：`RIGHT_CENSORED / STILL_ESCALATING` 的臂从自身 exact 末状态再续 4000 ms 一次。

## 7. 三类硬停机（与 spec §9 一一对应）

- **G1** exact-state / input / hash integrity
- **G2** numerical integrity
- **G3** resource / checkpoint integrity

科学结果（saturation / silence / low / bounded）**不是** gate，不得中止其余预注册主臂。

## 8. 授权边界

本 plan 不授权：U、M、full lifecycle、termination stimulation map、大型组合参数搜索、
CP-S/CP-L、threshold heterogeneity、global E→I tail、H-EFF/H-CAP 的**执行**（只允许写 spec/plan）。
不 push / merge / rebase。不改动或提交 `scripts/nohup_subject_capture.sh`。
