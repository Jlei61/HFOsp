# FCXR-LC1 — 动态慢反馈生命周期闭环（design + plan lock）

日期：2026-07-22
分支：`codex/topic4-mz-fcxr-lc1`（worktree `.worktrees/topic4-mz-fcxr-lc1`），基线 = `6819643`（FCXR-RC1 Stage D bounded-negative 收口）。
上游合同：Stage-D 收口 `docs/archive/topic4/sef_hfo/mz_fcxr_stage_d_branch_map_2026-07-22.md` §6/§7；Stage-D 设计 `docs/superpowers/plans/2026-07-20-topic4-mz-fcxr-stage-d.md`。
执行合同（authoritative）：用户 FCXR-LC1 长时执行 prompt（2026-07-22）。本文件是该 prompt 的**实现级契约落地**，不覆盖 prompt；冲突以 prompt 为准。
结果根：`results/topic4_sef_hfo/mz_full_conductance_spatial_relay/lifecycle_closure/`
状态：**E0 设计锁（尚未跑任何仿真）。**

---

## 0. 唯一承重目标

在同一个患者特异空间 scaffold 上、同一次连续仿真中，由**动态慢变量自主**完成：

> 统计间期（稀疏、不规则、自终止 IED）→ 有界发作样活动 → 自主终止（允许 post-ictal 静默）→ 统计间期恢复（returning IED，事件统计 + 空间 readout 回到 baseline band，无漂移/无复燃）。

- "间期"不要求固定节律/固定点/周期轨道——稳定背景邻域 / 稳定概率分布即可。
- 一次高发放片段、一次 runaway、一次踢后瞬变、一次末段静默都**不**等于目标。
- 允许 GO / 分阶段 NO-GO / 工程阻断。禁止扩参数海、改锁定底座、弱化验收。

---

## 1. 锁死不动的快系统（RC1，本轮不动）

`subject=epilepsiae_1146`，`montage=narrow`，`L=20`，`N=40000`（`NE=32000`），`dt=0.05 ms`，`drive=0.6`，`g=3.6`，`E_E=58`，`V_match=18`，`gaba_gain=1.125`，`g_sat=21.6`，primary seeds `1,3`，`M=off`，`phi=off`。

膜方程（`mz_slow_vars.membrane_terms`，full_conductance，`_fc_cfg`）：
- feedforward AMPA 加性（`ff_conductance=False`）；
- recurrent E→E 电导朝 `E_E`（`rec_conductance=True`）；
- recurrent-only 平滑饱和 `g_rec_eff = g_sat·tanh(g_rec_raw/g_sat)`（`rec_sat_g=g_sat=21.6`）。

**禁止**调 drive / 连接强度 / `g_sat` / `gaba_gain` / 空间核 / 阈值异质抢救结果；不重开 M/phi/dynamic global brake；不改 blessed engine（`kick_probe.py, params.py, model.py, connectivity.py, connectivity_rot.py, lfp.py`，SHA 见 `results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json`），除非无法绕开的 P0——若必须改，先停并报告，不自行 re-bless。

**已存在、直接复用（`mz_slow_vars.py` 非 blessed）**：
- 动态 Z：`use_z=True`，`z_inf = H(I_th_EI − I_I)`（硬 Heaviside，strict `>=` 处耗竭），Euler `z += (dt/tau_z)(z_inf−z)`，`z∈[0,1]`。`I_TH_EI = 95.19851312666987`（锁）。
- X relay：`use_x=True`（要求 full_conductance + split-exc），`y_j` 传感器（`tau_y`，每 E-spike +`1000/tau_y`），`x_inf = 1 − (1−x_min)·Hill([y−y_gate]₊; K_y, hill_n)`，**因果** `ee_relay_send = x_relay(t−)`（本帧散射前快照，`kick_probe.py:361/445-450` 按源缩放**外向** E→E 边）。
- X↔M1-STD 互斥：`kick_probe.py:214` `if relay_on and ee_std_on: raise ValueError`（blessed，已有）。

---

## 2. 唯一引擎改动 = §4.3 非对称 X 动力学（`mz_slow_vars.py`，off-by-default）

新增 config 字段：`tau_x_down: float|None = None`、`tau_x_up: float|None = None`。
`step()` 中把当前对称更新（`mz_slow_vars.py:371`）
```
self.x_relay += (x_inf - self.x_relay) * (1.0 - np.exp(-dt / c.tau_x))
```
改为：
```
if c.tau_x_down is None and c.tau_x_up is None:
    self.x_relay += (x_inf - self.x_relay) * (1.0 - np.exp(-dt / c.tau_x))   # 逐比特还原（byte-parity）
else:
    tau_sel = np.where(x_inf < self.x_relay, c.tau_x_down, c.tau_x_up)       # deplete->down, recover->up
    self.x_relay += (x_inf - self.x_relay) * (1.0 - np.exp(-dt / tau_sel))
```
`_validate_config`（`if c.use_x:` 块内）新增：both-or-neither + 正性。

### 2.1 契约条款（deep-contract-verify；每条 = 一个 test，`tests/test_mz_slow_vars.py`）

| # | 条款 | 验证 |
|---|---|---|
| C1 | `tau_x_down is None AND tau_x_up is None` → X 更新逐比特等于旧对称路径 | `test_asym_x_off_byte_parity_symmetric` |
| C2 | `x_inf < x_relay`（耗尽）用 `tau_x_down` | `test_asym_x_depletion_uses_tau_down` |
| C3 | `x_inf >= x_relay`（恢复）用 `tau_x_up` | `test_asym_x_recovery_uses_tau_up` |
| C4 | `ee_relay_send = x_relay(t−)` 因果顺序不变（send 快照在 y/x 更新前） | `test_asym_x_causal_send_snapshot_unchanged` |
| C5 | 只改 `x_relay`（长度 NE，E-only）；I cell / E→I / I→E / I→I 不受影响 | `test_asym_x_scales_only_outgoing_ee` |
| C6 | `use_x` + `ee_std_u>0` 仍互斥 raise（blessed guard 未破坏） | `test_asym_x_mutex_with_m1_std_preserved` |
| C7 | `tau_x_down<=0` / `tau_x_up<=0` / 只给一个（另一 None）→ ValueError | `test_asym_x_invalid_timescale_fail_fast` |
| C8 | 同 config+inputs → `x_relay` 轨迹逐比特一致 | `test_asym_x_deterministic` |
| C9 | `x_relay ∈ [x_min,1]` 在非对称锤打下保持 | `test_asym_x_bounded_unit_interval` |

**设计不变量（非 config gate，E4 runner 验证 + log）**：`tau_x_down < tau_z <= tau_x_up`。理由：把 Z↔X 的 tau 关系硬编进 mz_slow_vars 是层错（config 不知道 use_z 与 use_x 联用意图）；由 spec/runner 负责选参并记录。

---

## 3. 生命周期状态机分类器（`src/topic4_mz_fcxr_lifecycle.py`，纯逻辑，TDD on synthetic）

### 3.1 状态标签集（≥10）
`INTERICTAL_BASELINE / DENSE_EVENT_TRAIN / ICTAL_LIKE_BOUNDED / TERMINATED_REFRACTORY / RECOVERED_INTERICTAL / PERMANENT_SILENCE / RAPID_RELAPSE / RUNAWAY / NUMERICAL_UNSAFE / UNRESOLVED`。

### 3.2 输入契约
一次连续 run 归约为**有序窗口序列** `windows`（每窗 `win_ms`，如 1000ms）：每窗 `dict`：
- `occ`：该窗 300ms rolling 率高于间期 band 上界（`baseline_roll_hi`）的时间占比；
- `event_rate_hz`、`iei_ms`、`duration_ms`、`participation`、`peak_rate_hz`：该窗事件统计；
- `recruit_frac`：该窗招募范围（超 baseline P90 = 扩展）；
- `numerical_unsafe`：该窗数值门。
加 `baseline_band`（E1 建立：各量的 accepted 统计带 lo/hi + P90）与全局 `runaway` / `numerical_unsafe`。

### 3.3 分类器契约条款（每条 = test，`tests/test_topic4_mz_fcxr_lifecycle.py`）
- **L1 安全优先**：任一窗 `numerical_unsafe` 或全局 runaway → `NUMERICAL_UNSAFE` / `RUNAWAY`（先于一切）。
- **L2 间期窗**：事件统计落 `baseline_band` 且 `occ < ELEVATED_OCC` 且 `event_rate_hz>0` → interictal。
- **L3 密事件/发作窗**：`occ ≥ HIGH_OCC` 且 `recruit_frac` 超 P90 且有界 → ictal-like；`ELEVATED_OCC ≤ occ < HIGH_OCC` → dense train。
- **L4 生命周期序列**：需**顺序**满足 pre-ictal interictal（≥`PRE_MS`，默认 8000ms）→ ictal-like bounded（≥`ICTAL_MS`，默认 1000ms，且 recruit 超 P90，有界）→ 自主 termination（occ 掉出高区，非手工）→ recovery（≥`POST_MS`，默认 8000ms）。全满足 → `RECOVERED_INTERICTAL`（承重）。
- **L5 anti-cheat（§6 #12，最重要）**：recovery 窗必须有 **returning events**：`event_rate_hz > 0` 且落入 `baseline_band`。整段 post 窗 `event_rate_hz==0`（静默）→ `PERMANENT_SILENCE`，**绝不**判 RECOVERED。仅 `occ` 掉到 band 以下**不**充分。
- **L6 复燃**：termination 后短时（<`RELAPSE_GUARD_MS`）occ 再入高区 → `RAPID_RELAPSE`。
- **L7 未闭合**：有 ictal-like 但无合格 recovery（且非 silence/relapse）→ `TERMINATED_REFRACTORY`（终止但未统计返回）或 `UNRESOLVED`。
- **L8 因果**：`RECOVERED_INTERICTAL` 额外要求（在 E4 真 run 层校验，非纯分类器）：终止前 X 有延迟累积（`D_X` 在 ictal 段末上升）、Z 先回安全邻域后 X 才恢复近 1。

### 3.4 慢变量相图坐标
`D_Z(t) = Σ p_i(1−z_i)/Σ p_i`，`D_X(t) = Σ p_j(1−x_j)/Σ p_j`（`p` = onset-depletion 权重）。有效算子按**列**（presynaptic）缩放：`J_eff^EE ~ D_gain · D_sech² · W_EE · D_x`。

---

## 4. Phase 计划（pilot-first；重活全部 nohup detached + 资源看门狗）

- **E0** 工程合同：本 spec + C1–C9 tests（红）→ 实现 → pytest 绿 + engine-bless 绿 + smoke；lifecycle 分类器 L1–L7 synthetic tests；`--confirm-run` 门。**不进 nohup。**
- **E1** 统计间期合同：seed1/3 slow-off baseline（各 24s，串行），切非重叠 8s 窗 → `baseline_contract.json`（事件率/IEI/时长/参与度/峰值/core-axis-off participation/rolling occ/energy field/空间模板 concordance/数值安全 + 各量统计带 + P90）。Gate：seed1/3 都保住工作点、零 clip、`tau_eff_min≥2dt`、无 runaway、有足够 returning events。失败即停。
- **E2** 动态 Z-only：`use_z=on`，X 中性/off，M/phi/kick off。用**既有** calibration（`results/topic4_sef_hfo/mz_slowvars/calibration.json`）：**q75 主** = `I_th_EI=95.198`/`tau_z=5000`（`zA_q75_tz5000`，mid depletion）；**q50 sensitivity** = `I_th_EI=1.665`/`tau_z=10000`（`zA_q50_tz10000`，strong depletion）。`_fc_cfg` 硬编 I_th_EI=95.198→q50 需 `cfg["I_th_EI"]=1.665` override。`D_Z(t)` 的 p_i 权重取 `zA_q75_tz5000` snapshot（`load_onset_depletion_pi`）。问：先保≥8s 间期？自主进亚稳密事件区？有界？`D_Z(t)` event-locked 阶梯 or 硬阈值伪跳变？Gate：两档都进不去 → clean NO-GO；硬阈值伪跳变 → 停线并提 smooth-Z fallback（本轮不实现）。
- **E3** X sensor separation：`use_x=true, x_min=1`（X 对连接中性，只跑 `y_j`）。比 D=0 间期 vs D≈0.15 密事件区（seed1 主/seed3 确认）。主值 `tau_y=120ms, hill_n=4, y_gate=baseline y q99.9`；分不开才试 `tau_y∈{80,120,200}`。Gate：baseline IED 基本不越门、dense train 持续越门、core→axis→off 招募顺序可解释、seed1/3 同向；严重重叠 → `SENSOR_NO-GO`。
- **E4** 动态 Z+X pilot-first：只扫 `x_min`(3 有界 bracket) × `tau_x_down`∈{0.5,1,2}s × `tau_x_up`∈{tau_z, 2·tau_z}。先用 D≈0.15 下短静态/冻结-X control 找"能明显降 dense occ"的区间再取三档（不按最终 lifecycle cherry-pick）。seed1 cheap-first；中心 12–20s pilot 确认有 onset+X activation 再扩相邻格；T≥20s 严格单 worker；≤12 长格点；找到合格候选即停扩参数海转 seed3；primary 候选**无 kick**，kick 仅 secondary basin diagnostic。

### 4.1 生命周期正式验收（§11）
pre-ictal ≥8s 稀疏不规则自终止 IED 落 seed band（X 不提前明显耗尽）→ ictal-like bounded ≥1s（occ 明显超间期上界、recruit 超 P90、轴外/更广招募、有界零 clip 无 ceiling、非同一 IED 轻微加密）→ termination（无 reset/无持续驱动/无参数切换、自主退出高 occ、X 终止前延迟累积）→ recovery（允许先 silence，再 ≥8–12s：returning IED + 事件统计/空间模板回 band、无永久沉默、无快速复燃、Z 先回安全邻域后 X 才恢复近 1）→ 复现（seed3 同向；`x_min` 或 X-tau 至少一个 ±20% 邻域检查）。energy-field bridge / eigenmode / stimulation **仅** lifecycle 通过后做。

---

## 5. 文件与输出

代码：`src/topic4_mz_fcxr_lifecycle.py`、`scripts/run_topic4_mz_fcxr_lifecycle.py`、`tests/test_topic4_mz_fcxr_lifecycle.py`；引擎改动仅 `src/snn_engine/mz_slow_vars.py`（+ `tests/test_mz_slow_vars.py` 的 C1–C9）。
结果根：`results/topic4_sef_hfo/mz_full_conductance_spatial_relay/lifecycle_closure/`：`STATUS.md`、`run_manifest.json`、`baseline_contract.json`、`sensor_separation.json`、`z_only_summary.json`、`lifecycle_grid.json`、`candidate_verdict.json`、`runs/<run_id>/`、`figures/README.md`、`figures/{lifecycle_diagnostic,slow_phase_portrait}.png`。lifecycle 未过 → 只出诊断图，不出伪 Figure 5。

## 6. 资源 / nohup 合同（§13/§14）

复用 `run_topic4_mz_fcxr.py`：`_launcher_lock`（`OUT_ROOT/.l20_launcher.lock`）、`_assert_engine_blessed`、`_meminfo`、`_resource_log`、`_plan_workers`、`_fc_cfg`、`_write_json/_npz`、`PP.build_substrate`。
**更严于现有** `_plan_workers`：T≥20s → **强制 1 worker**（不因无 sibling 放宽到 2）、全程 ≤2、OMP/BLAS/NUMEXPR=1；`_plan_workers` 返回 0 → 停止提交（禁 `max(1,·)`）。
watchdog（新增，相对本 launcher baseline）：每 30s 记 `resource_log.jsonl`；soft `MemAvail<64GiB` 或 `swap−baseline≥256MiB` → 不提新 cell + 完成当前 + `RESOURCE_PAUSED`；hard `<32GiB` 或 `≥512MiB` → SIGTERM 自己的 pool + `RESOURCE_ABORTED`（绝不 kill sibling）。长 run 串行、`E_spk_bool` 不落盘、算完即 `gc`、Z/X 只存 O(N) 预注册快照、连续 trace 下采样 ~4k–8k。
nohup 仅长跑；tests/smoke/`--dry-run`/bless/worktree 核对/flock/baseline 全绿后才 detached，用 `setsid env OMP...=1 python -u ... --confirm-run --workers 1`，立即写 `launcher.pid` + `RUNNING.json`；DONE/RESOURCE_PAUSED/ABORTED sentinel；原子写 JSON/NPZ。

## 7. Stop rules（§15，任一 = 合格 NO-GO）

baseline 工作点失败 / Z-only 不能同时保间期+进密区 / X 分不开普通 IED 与持续活动 / X onset 前耗尽 / 候选全短 blip / 全永久沉默 / 终止即复燃 / 只有 kick 才有 lifecycle / post-ictal 无 returning IED / early-ictal 空间扩展不成立 / seed3 不复现 / 只单一精确参数点成立 / 数值或资源门失败。
**禁止本轮引入**：M / phi / dynamic global inhibition / 新 drive / 新连接强度 / 新 `g_sat` sweep / 第 3–4 慢变量。X 选择性成立但不能终止 → 写"下一 sprint 可能需要 thresholded global recovery"建议，不叠加。

措辞分层（不得混写）：engineering green / baseline accepted / sensor accepted / lifecycle candidate / lifecycle confirmed / bounded-negative(no-go)。无真实 statistical recovery 前禁称 seizure lifecycle / limit cycle / bistability / 可恢复发作。
