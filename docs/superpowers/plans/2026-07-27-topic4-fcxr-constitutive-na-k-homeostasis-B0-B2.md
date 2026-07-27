# FCXR-ION 实施计划：Phase B0–B2

日期：2026-07-27

状态：**IMPLEMENTATION PLAN。本文件不授权立即启动 40k 长仿真；每个仿真 stage 需 `--confirm-run`。**

设计来源（唯一入口）：
`docs/superpowers/specs/2026-07-27-topic4-fcxr-constitutive-na-k-homeostasis-design.md`（rev2）

上一代际终局：
`docs/archive/topic4/sef_hfo/mz_fcxr_pump_lifecycle_gate_Ia_2026-07-27.md`

---

## 0. 范围与执行图

只执行 spec 的 **B0 + B1 + B2**，产出 **Gate H** 与 **Gate B** 两个判决。

**不执行**：B3（局部钾微扰定 `g_K_ion`）、B4（冻结高分支定 `eta_pump`）、三维相图、动态生命周期、
因果分解（四臂泵分解 / 钾钳制 / Na 钳制 / Z 消融 / 爆后 reset）、空间响应模态、数据一致性判据、
任何 `Cl`/`Ca`/双室扩展。

执行图**不是**按任务编号串行：

```
硬路径:  T1 -> T3 -> T4 ------\
                              +--> T5 -> T6(Gate H) -> T7 -> T8 -> T9(Gate B) -> T10
前置路径: T2 (方向读出功率) ---/        ^                                  ^
                                       |                                  |
                            T2 失败 => 不得进入 T8/T9        T6 失败 => 不得进入 40k
```

- **T1、T3、T4 无仿真**；**T2 一次 40k（11 s）**；**T5–T7 只用小网络**；**T8–T9 才动 40k**。
- **两个 B0 待验证项（spec §16 末尾）是 T1 与 T2**，二者任一不通过即停在 B0，不进入 B1/B2。

---

## 1. 工作区与文件边界

### 1.1 worktree

从当前分支 `codex/topic4-mz-fcxr-pump-lifecycle` 的 HEAD 创建：

```text
branch:   codex/topic4-fcxr-ion
worktree: /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-fcxr-ion
```

开始前记录：base commit、`git status --short`、`git worktree list --porcelain`、
六个 blessed engine sha256、sibling 40k 进程、`MemAvailable` / swap 基线。

### 1.2 预计改动

新建：

```text
src/snn_engine/ion_homeostasis.py          # 离子状态 + 数学 + IonHomeostaticMZAdapter
src/topic4_fcxr_ion.py                     # 纯分析：出处表、可行性门、起始位点读出、Gate H/B 判决
scripts/run_topic4_fcxr_ion.py             # 统一 runner
scripts/plot_topic4_fcxr_ion.py            # 诊断图
tests/test_ion_homeostasis.py
tests/test_topic4_fcxr_ion.py
```

修改：**无**。本 plan 不修改任何既有源文件（包括 `mz_slow_vars.py`）——离子层通过**包装**接入。

**明确不改（六个 blessed）**：`kick_probe.py` / `lfp.py` / `params.py` / `model.py` /
`connectivity.py` / `connectivity_rot.py`。若证明必须改，**停止并另写受审阅的 guarded-engine-change spec**。

### 1.3 一个已核实的工程事实（进入 T5 的设计约束）

`params.py` **没有** tonic-bias 旋钮（只有 `J_ext_E`/`J_ext_I` 与 `nu_ext_ratio`）。
因此 spec §7.2 允许的两个 nuisance 参数 `I_bias_E` / `I_bias_I` **由 adapter 作为常数加在 `drive` 上**，
与离子电流走同一条通路。**不改 `params.py`，也不改 `Params` 的任何默认值。**

---

## 2. RNG 与工作点合同

```text
connectivity_seed   1（development）；1 与 3（confirmatory）
noise_seed          202  = B0-2 方向读出功率前置（复现已接受的泵关臂，可直接对照）
                    401  = B2 development
                    402,403,404 = confirmatory unseen（与 B2 development 互斥）
initial_state_seed  0（deterministic reset）
```

`r0` 锁定来源（spec §7.1）：
`results/topic4_sef_hfo/mz_full_conductance_spatial_relay/pump_lifecycle/pump_baseline_equivalence.json`
→ `per_arm.pump_off.pooled.mean_rate_hz = 4.1581 Hz`（primary）。

**恰好一次闭合迭代**：锁 `r0` → 调两个 bias → 测得 `r0'` → **重算一次** `q_ion`/`J_Na_rest` → 再跑一次 →
**冻结**。残差硬门 `|r0' − r0| / r0 ≤ 10%`。超出即 NO-GO，不得继续迭代。

---

## 3. Task 1 — B0-1：引擎电压单位链路核算（无仿真）

**这是 spec §4.3 ⚠️ 的待验证项，直接决定 `g_K_ion` 的量级。**

### Files

- Create `src/topic4_fcxr_ion.py`（本 task 只加单位核算部分）
- Create `tests/test_topic4_fcxr_ion.py`
- Runner stage `b0-units`

### 已知的正面证据（进入 task 时已核实，仍必须走完整链路）

`params.py` 逐行标注单位：`V_th = 18.0 # mV`、`V_reset = 11.0 # mV`、`J_ext_E = 0.455 # mV`。
即引擎是 Brunel 型电流 LIF，`V` 相对静息，阈值 18 mV、复位 11 mV。

### 必须核算的完整链路（缺一不可）

1. `V` 的零点是**漏电反转**（`V_L = 0`），阈值 18 mV、复位 11 mV；
2. FCXR 电导膜路径 `V_inf = (drive + g_rev)/(1 + g_rel)` 中，`drive` 与 `g_rev` **同为 mV 量纲**，
   因此往 `drive` 加一个 mV 量纲的电流是**单位自洽**的；
3. 力匹配锚点 `v_match = 18` 与 `E_E = 58` 在同一坐标下自洽；
4. **本 substrate 的 `e_gaba` 与 `e_k` 都被 FCXR 配置设为 0（= 漏电反转）**，而生理钾反转约在静息下方
   37 mV。离子层**不改动**这两个既有反转，只把 `ΔE_K` 作为附加电流注入（spec §4.3），
   因此不会与既有 sAHP/GABA 通路冲突——**这一点必须在产物里显式记录**，否则读者会误以为离子层
   把 K 反转改到了 −37。

### Tests first

1. `test_engine_voltage_scale_is_mV`：从 `params.py` 读出并断言 `V_th=18`、`V_reset=11` 且注释单位为 mV；
2. `test_conductance_membrane_drive_is_mV_dimensioned`：构造已知 `drive`/`g_rel`/`g_rev`，
   断言 `V_inf` 与手算一致，且往 `drive` 加 `Δ` 使 `V_inf` 变化 `Δ/(1+g_rel)`；
3. `test_delta_EK_maps_one_to_one_into_engine_units`：给定 `K_o` 变化，断言注入电流 = `g_K_ion·ΔE_K`
   且 `g_K_ion = 1` 时 1 mV 的钾反转移动 = 1 个引擎单位；
4. `test_ion_layer_does_not_touch_existing_e_gaba_or_e_k`：断言配置里既有的 `e_gaba`/`e_k` 未被修改。

### Verify / Gate

```bash
pytest -q tests/test_topic4_fcxr_ion.py -k units
python scripts/run_topic4_fcxr_ion.py --stage b0-units
```

产物 `b0_voltage_unit_audit.json`，字段含 `status ∈ {CONFIRMED, NOT_CONFIRMED}`。
**`NOT_CONFIRMED` 即停在 B0**，并把 `g_K_ion` 的量级问题写成独立议题，不得带着未确认的单位进入 B1。

---

## 4. Task 2 — B0-2：起始位点方向读出 + 功率前置（一次 40k）

**这是 spec §9 B-real 的硬前置。** rev1 的方向读出在已接受底座上不可执行
（22 个事件里只有 2 个两核都有参与，逐块 `1,0,0,0,1`），必须先证明新读出有功率。

### Files

- Extend `src/topic4_fcxr_ion.py`（起始位点读出）
- Extend `tests/test_topic4_fcxr_ion.py`
- Runner stage `b0-direction-power`

### 读出定义（预注册，本 task 内锁死）

对每个自终止事件：取事件窗内**最早发放的前 5%** 参与细胞，算其位置质心，按到两个注册核中心的距离归属
（`source` / `sink`）；若两核距离之差小于核半径的 20% 则记为 `ambiguous`，不计入。

- 每个事件都可评分（**不要求两核同时参与**），这正是与 rev1 读出的关键差别；
- 报告 `n_scoreable`、`frac_source`、`frac_sink`、`frac_ambiguous`。

### Tests first（合成数据，不跑网络）

1. `test_initiation_site_scores_every_event_not_only_two_sided_ones`：构造只有单核参与的事件，
   断言 rev1 式读出得 `NaN` 而新读出仍可评分；
2. `test_initiation_site_assigns_by_earliest_5pct_centroid`：合成一个从 source 核扩散出去的事件，
   断言归属为 `source`，反之亦然；
3. `test_ambiguous_when_centroid_is_equidistant`：质心落在两核中垂线上 → `ambiguous`；
4. `test_power_precondition_is_a_hard_gate`：`n_scoreable < 20` 时判 `INSUFFICIENT_POWER`，
   **不得**返回可用阈值。

### 仿真

一次 40k：arm-C 泵关配置（**无**离子层）、`noise_seed = 202`、`T = 11 s`，与已接受的泵关臂**逐参数一致**，
因此同时是一次**复现性检查**（`mean_rate_hz` 应复现 4.158 Hz）。

### Gate（B0-2）

```text
n_scoreable >= 20  且  frac_source > 0  且  frac_sink > 0
```

**不满足即必须先加长窗口或换读出**，不得带着一个在自己底座上都测不出来的判据进入 T8/T9。
阈值只在本前置通过之后、在 B2 development seed 上预锁。

产物：`b0_direction_power.json`。

---

## 5. Task 3 — B0-3：纯离子数学 + TDD（无仿真）

### Files

- Extend `src/topic4_fcxr_ion.py`
- Extend `tests/test_topic4_fcxr_ion.py`

### 纯函数

```python
pump_flux(Na_i, K_o, *, rho, Na_half, s_Na, K_half, s_K)
glia_uptake(K_o, *, G_glia, half=18.0, slope=2.5)
bath_clearance(K_o, *, eps, k_o_inf)
K_i_from_Na_i(Na_i, *, K_i0, Na_i0)
E_K(K_o, K_i, *, RTF=26.64)
q_ion_from_f(f, *, I_pump_0, r0)          # -> (q_ion, J_Na_rest)
```

### Tests first（每条都是 spec §3 的一个子句）

1. 泵方程逐字复现参考形式；`I_pump(18.0, 4.0) == 0.02016 mM/s`（4 位有效数字）；
2. 泵对 `Na_i` 与对 `K_o` 都**单调递增**、有界于 `rho`；
3. 胶质项在高钾处**饱和**趋于 `G_glia`（回归：不得被并成线性项）；
4. `K_i = 140 + (18 − Na_i)` 代数闭合；
5. `E_K(4.0, 140.0) == −94.71 mV`（2 位小数）；
6. `q_ion_from_f`：`J_Na_rest = 3·I_pump_0·(1−f) ≥ 0` **对所有 f∈(0,1] 成立**（可行性门）；
7. `f` 越界（≤0 或 >1）**抛异常**，不得静默 clip；
8. **静息不动点**：无 spike、`Na_i = 18`、`K_o = 4` 时，`d[Na]_i/dt` 与 `d[K]_o/dt` 都为 0
   （中心化形式的直接检验）；
9. **`h` 与半激活常数不得跨文献混用**：出处表里每个 inherited 项都能追到单一来源标签。

### Verify

```bash
pytest -q tests/test_topic4_fcxr_ion.py
```

---

## 6. Task 4 — B0-4：出处表与解析可行性门落成产物（无仿真）

把 spec §3 的表格变成**机器可读、可重跑**的产物，而不是文档里的声明。

### Files

- Extend `src/topic4_fcxr_ion.py`（`PARAM_TABLE` + `analytic_feasibility()`）
- Runner stage `b0-provenance`

### 内容

- `PARAM_TABLE`：每项含 `value / unit / equation / source / kind ∈ {inherited, derived, effective}`；
- `analytic_feasibility(f, r0)`：复算 spec §3.2 的表（一次普通事件的 `ΔK_o`/`ΔE_K`、
  持续 50 Hz 的稳态 `K_o`/`ΔE_K`），并检查三条硬门：
  `J_Na_rest ≥ 0`、静息是不动点、所有浓度 > 0。

### Tests

1. `test_every_inherited_param_has_a_single_source_label`；
2. `test_no_effective_param_is_labelled_inherited`（防止把自己的闭合假设伪装成继承）；
3. `test_analytic_feasibility_reproduces_the_spec_table`（对 `f ∈ {1.0, 0.5, 0.25}` 三行逐值比对）；
4. `test_feasibility_gate_rejects_negative_rest_leak`。

产物：`b0_parameter_provenance.json`、`b0_analytic_feasibility.json`。

---

## 7. Task 5 — B1-1：`ion_homeostasis.py` + adapter（小网络）

### Files

- Create `src/snn_engine/ion_homeostasis.py`
- Create `tests/test_ion_homeostasis.py`

### 状态

```text
Na_i_all        (N,)        E 与 I 都有
K_o_grid        (32, 32)    primary
pump_flux_all   (N,)
E_K_all         (N,)
grid_spikes_E / grid_spikes_I
cell_to_grid    (N,)  int   由 net["pos"] 预计算
n_per_grid      (32,32) int 占用数（空格 -> 源项 0）
```

### `IonHomeostaticMZAdapter` 委托合同（spec §6）

引擎协议表面（已逐条枚举）必须**全部直通**到被包装的 `MZSlowVars`：

```text
capability : uses_conductance_membrane  uses_split_excitation  uses_ee_relay  uses_shunt
attribute  : cfg   ee_relay_send   nE   q_I
call       : membrane_terms  apply_currents  threshold  step
```

顺序：

1. `membrane_terms(...)` → 先**原样**调 `mz.membrane_terms(...)`；再对 **E 和 I 全体**
   `drive += I_bias_a + g_K_ion*(E_K − E_K_0) − eta_pump*(I_pump − I_pump_0)`，
   用**上一个离子块**的 `E_K`/`I_pump`；`g_rel`/`g_rev` **一个字节不动**；
2. `threshold(...)` → 直通；
3. `step(spk, labels, dt)` → **先**原样调 `mz.step(...)`（既有 Z/M/X 顺序与逐步值不变），
   **再**累积 E/I 网格 spike；每 `dt_ion/dt` 步更新一次离子状态；
4. `apply_currents(...)` → 直通；本代际只走电导膜路径，若被触发则 **raise**（"stub 必须响亮失败"）。

### Tests first（每条对应 spec §13 的一行）

```text
1  adapter-off byte-parity: ions 关闭时整条 simulate_kick 与裸 MZSlowVars 逐位相同
2  existing Z/M/X update order unchanged（复用上一 sprint 的逐步值测试）
3  I-cell coupling is a CURRENT：显式验证离子项在 I 细胞上确实生效（防"只作用于 E 却报告 E/I"）
4  capability passthrough：四个谓词 + cfg/ee_relay_send/nE/q_I 全部直通
5  apply_currents raises（不静默返回可信数值）
6  resting equilibrium：无 spike -> Na_i、K_o 不动
7  single-spike Na/K update：一次 spike 的 Na 增量 = q_ion，K 增量 = beta*q_ion/n_g
8  3:2 pump flux identity：同一 I_pump 在 Na 方程系数 3、K 方程系数 2*beta
9  finite-volume K budget closure：源 − (泵回收+清除+胶质) − 扩散净通量 与 Δ总胞外钾 相对误差 < 1e-10
10 zero-flux boundary：扩散净通量恒为 0（角/边格邻居数 2/3）
11 empty-voxel handling：n_g = 0 时源项与泵项为 0，清除与扩散照常
12 grid-resolution：16/32/64 的总预算与粗粒化场一致（**不**要求逐格相同）
13 multi-rate convergence：dt_ion ∈ {0.25,0.5,1,2} ms 收敛
14 checkpoint/restart identity
15 no negative concentrations（safety bound 只 fail-fast，不作饱和器）
16 blessed sha256 未变
```

### Verify

```bash
pytest -q tests/test_ion_homeostasis.py
pytest -q tests/test_mz_slow_vars.py tests/test_mz_full_conductance_spatial_relay.py \
          tests/test_topic4_mz_fcxr_heo1.py tests/test_topic4_mz_fcxr_heo2.py \
          tests/test_topic4_mz_fcxr_heo3.py
```

**仍不启动 40k。**

---

## 8. Task 6 — B1-2：Gate H（`N≈1000`，再 `N≈4000`）

### Files

- Add `adjudicate_gate_H()` to `src/topic4_fcxr_ion.py`
- Runner stage `b1-gate-h`

### 三项检验（spec §9，措辞已订正：**不得**暗示 ion-conserving）

1. **ODE balance residual**：无 spike 时静息不动点正确，`|d[Na]_i/dt|`、`|d[K]_o/dt|` → 0；
2. **finite-volume K budget closure**：相对误差 < 1e-10，零通量边界净通量为 0；
3. **pump 3:2 flux identity**。

另需：baseline 泵通量**非零**；局部扰动能恢复；网格/`dt_ion`/checkpoint 一致；
离子层关闭时旧引擎 byte-parity；无负浓度、未触 safety bound。

产物：`gate_H.json`，`status ∈ {PASS, FAIL_EQUILIBRIUM, FAIL_BUDGET, FAIL_STOICHIOMETRY,
FAIL_PARITY, FAIL_NUMERICAL, UNRESOLVED}`。

**Gate H 不 PASS 不进入 40k。**

---

## 9. Task 7 — B1-3：在小网络上选 `f`（三点）

`f ∈ {1.0, 0.5, 0.25}`，**只在这三点里选**，不做连续搜索。

判据（预注册）：单次事件的钾瞬态**可测但会恢复**；重复事件簇能时间积分；
普通事件**不**产生持续钾累积。

产物：`b1_f_selection.json`（含三点的完整对照，不只写选中的那个）。

---

## 10. Task 8 — B2-1：40k bias 重标定 + 一次闭合迭代

### 只允许两个 nuisance 参数

`I_bias_E`、`I_bias_I`（由 adapter 加在 `drive` 上，§1.3）。
**没有第三个 fallback**：两个 bias 无法恢复 baseline 即判 **NO-GO**（spec 停机条件 4）。

### 搜索策略（禁止二维笛卡尔网格）

先用 **4 s 短探针**（约 6 min/次）做一维化搜索：先固定 `I_bias_I` 调 `I_bias_E` 使 E 群体率
接近 `r0`，再调 `I_bias_I` 使 E/I 率比回到泵关臂的比值；两轮交替，**上限 12 次探针**。
超过 12 次仍未进入容差 → NO-GO。

### 一次闭合迭代（spec §7.1）

锁 `r0 = 4.1581 Hz` → 调 bias → 测 `r0'` → **重算一次** `q_ion`/`J_Na_rest` → 再跑一次 11 s 验证 →
**冻结**。硬门 `|r0' − r0|/r0 ≤ 10%`，且离子变量块间无显著趋势。

产物：`b2_bias_calibration.json`、`b2_closure_iteration.json`。

---

## 11. Task 9 — B2-2：Gate B 判决 + confirmatory

### B-real（优先级 1，binding）

| 判据 | 真实目标产物 | 层级 |
|---|---|---|
| 两个稳定传播模板存在，模型两核对应其 source foci | `results/interictal_propagation_masked/per_subject/epilepsiae_1146.json` → `adaptive_cluster.stable_k` | **模板层** |
| 间期稀疏性与不规则性 | 同上 `propagation_stereotypy` / `temporal_dynamics` | 模板层 |
| 事件在**两个注册核**上都会起始 | T2 锁定的起始位点读出 | 模型侧 |

> **禁止写成"复现了双向传播"或"复现了通道层角色互换"** —— E1146 的
> `candidate_forward_reverse_pairs = null`，通道层互换分数未 populated（spec §9）。

### B-model（优先级 2，工程参考，对照 arm-C 泵关臂）

- `Na_i`/`K_o` 块间平稳，无缓慢倒计时；
- 事件率 / 间隔中位数 / 间隔变异系数落在泵关臂块间容差内（UNDERPOWERED 项不作等价证据）；
- 源 / 汇 / 轴外放电份额落在容差内；
- **普通事件不产生全 sheet 钾波**：远离事件的格 `ΔK_o` < 事件格的 10%；
- baseline 泵不饱和；普通事件后 `Na_i`/`K_o` 可恢复。

### Confirmatory

`connectivity {1,3} × unseen noise {402,403,404}` = **6 条轨迹**；**bias 只在 development seed 401 上调**，
confirmatory 不得用于调参。

产物：`gate_B.json`、`candidate_verdict.json`。

---

## 12. Task 10 — 图、STATUS 与归档

结果根：

```text
results/topic4_sef_hfo/mz_full_conductance_spatial_relay/ion_homeostasis/
```

必须有：`STATUS.md`（分层写 engineering / Gate H / Gate B，上游 PASS 不自动下传）、
`run_manifest.json`、`resource_log.jsonl`、`b0_*.json`、`gate_H.json`、`gate_B.json`、
`candidate_verdict.json`、`figures/` + 中文逐图 `README.md`。

诊断图（**只画已通过的门**，缺输入就跳过、不画占位）：

1. `b0_feasibility_and_units.png` — 解析可行性（三点 `f` 的间期/高态动态范围）+ 单位链路核算；
2. `gate_H_homeostasis.png` — 静息不动点、预算闭合残差、3:2 恒等式、网格/多时间尺度收敛；
3. `gate_B_interictal_substrate.png` — 新 substrate 的间期指标 vs 泵关臂容差 + 两核起始位点分布。

**生命周期候选图与四栏 paper 图一律不生成**（B3/B4 未授权）。

归档：`docs/archive/topic4/sef_hfo/fcxr_ion_B0_B2_<verdict>_<date>.md`。

---

## 13. Runner / nohup / OOM

统一入口：

```bash
python scripts/run_topic4_fcxr_ion.py --stage <stage> --confirm-run
```

stage：`b0-units` `b0-provenance` `b0-direction-power` `b1-smallnet` `b1-gate-h` `b1-select-f`
`b2-bias` `b2-validate` `b2-confirm` `b2-adjudicate`

长任务一律：

```bash
setsid nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python scripts/run_topic4_fcxr_ion.py --stage <stage> --confirm-run \
  > <run_dir>/nohup_<stage>.log 2>&1 < /dev/null &
```

配 `RUNNING_<stage>.json` / `DONE_<stage>.json` / `FAILED_<stage>.json` / `<stage>.pid` /
`resource_log.jsonl`，并用**按 stage 分片的单实例锁**（网络恢复后不会重复提交同一 stage）。

### 实测成本（本 sprint 的 40k 计时，用于排期）

| 运行 | 仿真时长 | 实测 wall |
|---|---:|---:|
| 传感器标定（含离线扫描） | 11 s | 1290 s |
| 泵关臂 | 11 s | 952 s |
| 泵开臂 | 11 s | 1291 s |

即约 **87–117 s wall / 每秒仿真**。据此排期：T2 ≈ 16 min；T8 探针 ≤ 12 × 6 min ≈ 72 min +
两次 11 s 验证 ≈ 40 min；T9 confirmatory 6 × 11 s ≈ 2 h（单 worker）。**B2 合计约 3.5–4 h。**

### 资源门（与上一 sprint 相同）

- `T < 20 s`：最多 2 workers；`T ≥ 20 s`：严格 1 worker；
- **已有 sibling 40k 时进一步降低，不与其争抢**；
- swap delta `> 256 MiB` 停止提交；`> 512 MiB` 且继续上升，或 `MemAvailable < 2×` 单 run peak：
  只停自己最新的任务；写 `RESOURCE_PAUSED.json` / `ABORTED.json`；
- **不杀 sibling / user 进程**；不保存 `N_cell × T` dense 离子状态（只存 landmark 场与流式标量）。

> **计划撰写时的实测占用（2026-07-27）**：另一 worktree 有 2 个 20 s 的 40k 任务在跑，
> swap 相对本 sprint 基线涨约 89 MiB（在 256 MiB 门槛内），`MemAvailable` 约 247 GB。
> **执行 T2/T8/T9 前必须重新核对 sibling 占用并据此降并发。**

---

## 14. Commit plan

```text
1  test: lock constitutive Na/K ion math and provenance table
2  feat: add engine voltage-unit audit and initiation-site direction readout
3  feat: add ion homeostasis state + IonHomeostaticMZAdapter (off-by-default)
4  feat: add Gate H homeostasis adjudication
5  feat: add 40k bias recalibration with one closure iteration
6  feat: add Gate B interictal substrate adjudication
7  docs: archive FCXR-ION B0-B2 verdict
```

每次提交前：

```bash
git diff --check
pytest -q <targeted tests>
```

最终回归：

```bash
pytest -q tests/test_ion_homeostasis.py tests/test_topic4_fcxr_ion.py \
          tests/test_mz_slow_vars.py tests/test_topic4_mz_fcxr_pump.py \
          tests/test_topic4_mz_fcxr_pump_lifecycle.py \
          tests/test_mz_full_conductance_spatial_relay.py \
          tests/test_topic4_mz_fcxr_heo1.py tests/test_topic4_mz_fcxr_heo2.py \
          tests/test_topic4_mz_fcxr_heo3.py
```

**不 push、不 merge、不 rebase。**

---

## 15. 停机判决

任何一处不通过都是合格的 bounded negative，不需要抢救：

- **T1 fail**：引擎电压单位链路不自洽 → `g_K_ion` 量级无依据，停在 B0；
- **T2 fail**：方向读出在自己底座上功率不足 → 先解决读出，不得进入 B2；
- **Gate H fail**：离子稳态或数值合同不成立 → 不进入 40k；
- **Gate B fail**：构成性离子 substrate 拿不回合法间期工作点 → 归档为
  "constitutive Na/K on this substrate cannot recover the interictal working point"，
  **不是**"离子机制被否定"。

**触发 spec §11 任一停机条件即停并归档，不得自动转入下一机制、不得扩大网格、
不得动 B3/B4，也不得引入第三个 nuisance 参数。**
