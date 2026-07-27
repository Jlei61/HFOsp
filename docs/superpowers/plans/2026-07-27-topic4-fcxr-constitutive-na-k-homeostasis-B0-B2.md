# FCXR-ION 实施计划：Phase B0–B2

日期：2026-07-27（rev3，随 spec rev4 同步；异质初始化器、空格合同、T7 五条门重写）

状态：**IMPLEMENTATION PLAN。本文件不授权立即启动 40k 长仿真；每个仿真 stage 需 `--confirm-run`。**

设计来源（唯一入口）：
`docs/superpowers/specs/2026-07-27-topic4-fcxr-constitutive-na-k-homeostasis-design.md`（**rev4**）

⚠️ spec rev4 把方程改为**偏差形式**（rev3 的常数形式在空格上破坏静息不动点）。**本 plan 的所有离子方程以 rev4 为准**；
rev2 的 `J_Na_rest = 3I₀(1−f)` 与「110 倍动态范围」均已作废，dial 也已从 `f` 改名重定义为 `f'`。

**rev3（第三轮审阅）改动**：

| 位置 | 改动 |
|---|---|
| T2 §4 | 顺带落盘**逐细胞基线率场** `b0_baseline_rate_field.npz` + sha256，作为异质初始化器的输入（零额外仿真） |
| T3 §5 | 新增偏差形式导数与 `heterogeneous_steady_state`；新增**空格静息不动点**与**异质初始化残差**两条测试（各带反向回归） |
| T4 §6 | 可行性表的比对候选改为 `{0.5, 1.0, 2.0}`；可行性门改为"静息不动点（含空格）" |
| T5 §7 | 初始状态改为异质预平衡；**修正测试 11 的空格合同**（旧写法会让空格以 +0.28221 mM/s 自己积钾） |
| T6 §8 | Gate H 新增第 4 项异质初始化残差（**逐细胞/逐格 q95/q99/max，群体均值不算通过**）+ 两个新 FAIL 分支 |
| T7 §9 | 候选集 → `{0.5, 1.0, 2.0}`（`1.0` primary）；**五条门全部重写**（旧三条经实算全部失效）；tie-break 由"取最大"改为"最接近 1.0" |
| T8 §10 | 探针与验证运行改用异质预平衡；块间趋势须逐细胞/逐格报告 |

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

### 1.3 上游 artifact preflight（P0-4，**T2 之前的第 0 步**）

**新建 worktree 后，两条上游 lineage 默认都不在当前目录下**（`results/` 是 gitignore 的）：

| 需要的 artifact | 实际所在 |
|---|---|
| 泵关臂基线（`r0`、块指标、容差） | **只在** pump-lifecycle worktree 的 `results/.../pump_lifecycle/pump_baseline_equivalence.json` |
| E1146 模板层 / 通道层传播产物 | **只在**主 checkout 的 `results/interictal_propagation_masked/...` |
| HEO1 slow-off baseline contract（T8 若需间期带） | **只在** heo1 worktree |

runner **必须**实现 `resolve_artifact(rel_path)`：按 `[本 worktree, PUMP_ROOT, HEO1_ROOT, MAIN_ROOT]`
顺序查找，命中即记录**绝对路径 + sha256 + mtime + schema 关键字段**；全未命中则**响亮失败**
（`SystemExit`），**不得**回退到相对路径、**不得**静默用默认值。

产物 `b0_artifact_preflight.json` 每个输入含：`resolved_abs_path` / `sha256` / `mtime` /
`schema_ok` / `root_used`。**preflight 不通过即停，不进入任何 stage。**

> 这不是假想风险：本 sprint 已踩过一次——HEO1 的 baseline contract 在新 worktree 里不存在，
> 靠一次「只构造不仿真」的干跑才在花掉一小时之前发现。

### 1.4 一个已核实的工程事实（进入 T5 的设计约束）

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

**恰好一次闭合迭代**：锁 `r0` → 调两个 bias → 测得 `r0'` → **重算一次** `q_ion`（`J_Na,0`/`J_K,0`
是常数，**不**随迭代改变）→ 再跑一次 → **冻结**。残差硬门 `|r0' − r0| / r0 ≤ 10%`。
超出且合法边界未被括住 → `UNRESOLVED_CALIBRATION`；被括住且无可行解 → `NO_GO_BASELINE`（§10）。

---

## 3. Task 1 — B0-1：引擎电压单位链路核算（无仿真）

**这是 spec §4.3 ⚠️ 的待验证项。注意它只定 `g_K_ion` 的量纲，不定其数值** ——
`g_K_ion = 1` 是一个明示的 effective reference normalization（spec rev3 §4.3），由 B3 标定。

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
3. `test_delta_EK_injection_is_dimensionally_consistent`：给定 `K_o` 变化，断言注入量 = `g_K_ion·ΔE_K`
   且量纲与 `drive` 一致；**不断言 `g_K_ion` 必须为 1**（那是归一化选择，不是核算结论）；
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
（`core_A` / `core_B`；**不再叫 source/sink**——E1146 的方向性未被确立，旧名夹带无证据的方向主张）；
若两核距离之差小于核半径的 20% 则记为 `ambiguous`，不计入。

- 每个事件都可评分（**不要求两核同时参与**），这正是与 rev1 读出的关键差别；
- 报告 `n_scoreable`、`frac_A`、`frac_B`、`frac_ambiguous`。

### Tests first（合成数据，不跑网络）

1. `test_initiation_site_scores_every_event_not_only_two_sided_ones`：构造只有单核参与的事件，
   断言 rev1 式读出得 `NaN` 而新读出仍可评分；
2. `test_initiation_site_assigns_by_earliest_5pct_centroid`：合成一个从 `core_A` 扩散出去的事件，
   断言归属为 `core_A`，反之亦然；
3. `test_ambiguous_when_centroid_is_equidistant`：质心落在两核中垂线上 → `ambiguous`；
4. `test_power_precondition_is_a_hard_gate`：`n_scoreable < 20` 时判 `INSUFFICIENT_POWER`，
   **不得**返回可用阈值。

### 仿真

一次 40k：arm-C 泵关配置（**无**离子层）、`noise_seed = 202`、`T = 11 s`，与已接受的泵关臂**逐参数一致**，
因此同时是一次**复现性检查**（`mean_rate_hz` 应复现 4.158 Hz）。

**这次运行同时落盘异质初始化器的输入（rev3 新增，闭合 spec §4.2c）**：本 task 已经要跑这条泵关轨迹，
顺带写出**逐细胞基线放电率场** `r_i`（E 与 I 分开，燃烧期之后的窗口平均）即可，**零额外仿真成本**：

- 产物 `b0_baseline_rate_field.npz`：`rate_E`（`N_E` 个 float32）、`rate_I`、`cell_voxel`（细胞→格索引）、
  `n_cells_per_voxel`（含**哪些格是空的**）、窗口定义、`noise_seed`；
- 记录该文件的 **sha256**，T5/T6/T8 的初始化器必须引用同一个哈希；
- 体量约 32000 + 8000 个 float32 ≈ 160 KB，不触碰"不保存 `N_cell × T` dense state"的资源合同。

**没有这个率场就不得进入 T8**：单一全局速率的标量稳态会留下 11 s 暴露不出来的缓慢空间重排（spec §4.2c）。

### Gate（B0-2）

```text
n_scoreable >= 20  且  frac_A > 0  且  frac_B > 0
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
background_fluxes(I_pump_0, beta)         # -> (J_Na_0 = 3*I0,  J_K_0 = 2*beta*I0)   常数
q_ion_from_fprime(fp, *, J_Na_0, r0)      # -> q_ion = J_Na_0 * fp / r0
interictal_steady_state(q_ion, r)         # -> (Na_star, K_o_star)  2D 联立解（同质，仅用于解析表）

# rev3 新增（spec §4.1/§4.2 偏差形式 + §4.2c 异质初始化器）
dNa_dt(Na_i, K_o, spikes, *, q_ion)                    # = q_ion*S - 3*(I_pump - I_pump_0)
dKo_dt(K_o, r_bar, I_pump_bar, n_cells, K_o_nb, ...)   # 全部偏差项；n_cells==0 时 spike 项与泵偏差项均为 0
heterogeneous_steady_state(rate_E, rate_I, cell_voxel, *, q_ion)
    # -> (Na_star[per cell], K_o_star[per voxel], n_iter, residual)  §4.2c 的不动点迭代
```

### Tests first（每条都是 spec §3 的一个子句）

1. 泵方程逐字复现参考形式；`I_pump(18.0, 4.0) == 0.02016 mM/s`（4 位有效数字）；
2. 泵对 `Na_i` 与对 `K_o` 都**单调递增**、有界于 `rho`；
3. 胶质项在高钾处**饱和**趋于 `G_glia`（回归：不得被并成线性项）；
4. `K_i = 140 + (18 − Na_i)` 代数闭合；
5. `E_K(4.0, 140.0) == −94.71 mV`（2 位小数）；
6. `background_fluxes`：`J_Na_0 = 0.06047`、`J_K_0 = 0.28221 mM/s`（5 位有效数字），**与 `f'` 无关**；
7. **静息不动点（核心回归）**：无 spike、`Na_i = 18`、`K_o = 4` 时
   `d[Na]_i/dt` 与 `d[K]_o/dt` **都精确为 0**（到机器精度）。
   **两条反向回归**，分别锁死两次修正、防止将来有人"简化"回去：
   (a) 去掉背景通量（rev2 写法）→ `d[K]_o/dt` 必须回到 `−2β·I₀ ≈ −0.2822`；
   (b) 用 rev3 的常数写法（保留常数 `J_K,0` 但把空格泵项置零）→ 空格必须回到 `+0.28221`；
7b. **空格静息不动点（rev3 新增，spec §4.2 空格合同）**：`n_cells = 0` 且 `K_o = 4` 时
   `d[K]_o/dt` **精确为 0**。这是 rev3 的 P0：空格是采样缺口不是无组织区，
   泵项取未解析组织的静息值 `I_pump_0`，只把 spike 超额置零；
7c. **异质初始化残差（rev3 新增，spec §4.2c）**：给一个 E/I 与空间都不均匀的合成率场，
   `heterogeneous_steady_state` 的输出代回 `dNa_dt`/`dKo_dt` 后，
   **逐细胞与逐格的 max |残差|** 都 < 1e-8；且断言含空格的率场也收敛（空格取 `I_pump_0`）。
   **同时附一条反向回归**：用单一全局标量稳态初始化同一个异质率场，
   逐细胞残差的 q99 必须**显著非零**——证明这条测试真的能区分两种初始化；
8. `f'` ≤ 0 **抛异常**，不得静默 clip；`f'` > 1 **允许**（它是倍数不是比例）；
9. `interictal_steady_state`：`f'=1, r=r0` 给 `(Na*, K_o*) ≈ (20.07, 4.11)`，
   `r=50 Hz` 给 `K_o* ≈ 5.28`（2 位小数）；
10. **量纲**：`dt_ion` 加倍时连续通量项的增量加倍、spike 项不变（防 1000 倍的 ms↔s 错误）；
11. **半激活常数不得跨文献混用**：出处表里每个 inherited 项都能追到单一来源标签。

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
  `J_Na,0 > 0`（构造恒成立）、**无 spike 态是精确不动点**、所有浓度 > 0。

### Tests

1. `test_every_inherited_param_has_a_single_source_label`；
2. `test_no_effective_param_is_labelled_inherited`（防止把自己的闭合假设伪装成继承）；
3. `test_analytic_feasibility_reproduces_the_spec_table`（对 `f' ∈ {0.5, 1.0, 2.0}` 三行逐值比对；
   spec §3.2 表里另外两行 `0.25` / `4.0` 是**参考行不是候选**，一并比对但标注 out-of-candidate-set）；
4. `test_feasibility_gate_rejects_broken_rest_fixed_point`（含空格那一支）。

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
n_per_grid      (32,32) int 占用数（空格 -> 只把 spike 超额置零；泵取 I_pump_0，见 spec §4.2 空格合同）
```

**初始状态不是常数**：`Na_i_all` / `K_o_grid` 由 §4.2c 的 `heterogeneous_steady_state`
从 T2 落盘的 `b0_baseline_rate_field.npz` 生成（引用其 sha256），**不得**用单一全局标量稳态填充。

### `IonHomeostaticMZAdapter` 委托合同（spec §6 rev3）

**rev1 列的 `nE` / `q_I` / `uses_shunt` 在真实 `MZSlowVars` 上并不存在**（已在实例上 `hasattr` 验证；
只有 `NE`）。合成它们会**翻转引擎的 `hasattr` 守卫、改变执行路径**。因此 adapter
**不列白名单**，改用 `__getattr__` 委托——天然保留"缺席"语义——只覆盖三个调用：

```python
def __getattr__(self, name):        # 未覆盖的一切原样委托；不存在的仍然不存在
    return getattr(self.mz, name)
```

顺序：

1. `membrane_terms(...)` → 先**原样**调 `mz.membrane_terms(...)`；再对 **E 和 I 全体**
   `drive += I_bias_a + g_K_ion*(E_K − E_K_0) − eta_pump*(I_pump − I_pump_0)`，
   用**上一个离子块**的 `E_K`/`I_pump`；`g_rel`/`g_rev` **一个字节不动**。
   **B0–B2 中 `eta_pump = 0`**（spec rev3 §4.3），该项恒为 0；
2. `apply_currents(...)` → 与 1 **对称**：先委托，再加**同一个**离子电流。
   （rev1 曾同时写成"直通"和"触发即 raise"，自相矛盾，已删。）
3. `threshold(...)` → 由 `__getattr__` 直通；
4. `step(spk, labels, dt)` → **先**原样调 `mz.step(...)`（既有 Z/M/X 顺序与逐步值不变），
   **再**累积 E/I 网格 spike；每 `dt_ion/dt` 步更新一次离子状态。

### Tests first（每条对应 spec §13 的一行）

```text
1  adapter-off byte-parity: ions 关闭时整条 simulate_kick 与裸 MZSlowVars 逐位相同
2  existing Z/M/X update order unchanged（复用上一 sprint 的逐步值测试）
3  I-cell coupling is a CURRENT：显式验证离子项在 I 细胞上确实生效（防"只作用于 E 却报告 E/I"）
4  capability passthrough：存在的（NE/cfg/ee_relay_send/三个 uses_* 谓词）全部直通
4b absent attributes stay absent：hasattr(adapter,'nE'/'q_I'/'uses_shunt') 必须为 False
5  apply_currents 与 membrane_terms 对称加同一离子电流（不 raise、不静默失效）
5b full-conductance 集成测试：走真实 simulate_kick 的电导分支，而非只做单元级调用
6  resting equilibrium：无 spike -> Na_i、K_o 不动
6b EMPTY-voxel resting equilibrium：n_g = 0 且 K_o = 4 时 dK_o/dt 精确为 0（rev3 的 P0-1 回归）
6c heterogeneous init residual：用 T2 的真实率场初始化后，逐细胞与逐格残差的
   q95/q99/max 都过门；反向回归=单一全局标量初始化时 q99 显著非零
7  single-spike Na/K update：一次 spike 的 Na 增量 = q_ion，格内 K 增量 = beta*q_ion/n_g
7b deviation form is exact：无 spike 时两个导数精确为 0，且该性质**与 occupancy 无关**
8  3:2 pump flux identity：同一 I_pump 在 Na 方程系数 3、K 方程系数 2*beta
9  finite-volume K budget closure：源 − (泵回收+清除+胶质) − 扩散净通量 与 Δ总胞外钾 相对误差 < 1e-10
10 zero-flux boundary：扩散净通量恒为 0（角/边格邻居数 2/3）
11 empty-voxel handling：n_g = 0 时**只有 spike 超额项为 0**；泵项取未解析组织的静息值
   I_pump_0（偏差为 0），清除、胶质与扩散照常。**不得**把泵项整体置零——那等于宣称空格无组织，
   会让空格以 +0.28221 mM/s 自己积钾（rev3 的 P0-1）
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
   **含空格**（`n_g = 0` 必须同样为 0，rev3 的 P0-1）；
2. **finite-volume K budget closure**：相对误差 < 1e-10，零通量边界净通量为 0；
3. **pump 3:2 flux identity**；
4. **异质初始化残差（rev3 新增，spec §4.2c）**：用 T2 的真实率场初始化后立即取残差 ——
   **逐细胞** `|d[Na]_i/dt|` 与**逐格** `|d[K]_o,g/dt|` 的 **q95 / q99 / max** 全部过门。

> **群体均值平稳一律不算通过。** 异质基底上均值可以平得很好，同时底下有一个 11 s 完全看不见的
> 缓慢空间重排（`tau_Na = 54.4 s`）。`gate_H.json` 必须逐项落盘 q95/q99/max，**不得只写均值**。

另需：baseline 泵通量**非零**；局部扰动能恢复；网格/`dt_ion`/checkpoint 一致；
离子层关闭时旧引擎 byte-parity；无负浓度、未触 safety bound。

产物：`gate_H.json`，`status ∈ {PASS, FAIL_EQUILIBRIUM, FAIL_EMPTY_VOXEL, FAIL_INIT_RESIDUAL,
FAIL_BUDGET, FAIL_STOICHIOMETRY, FAIL_PARITY, FAIL_NUMERICAL, UNRESOLVED}`。

**Gate H 不 PASS 不进入 40k。**

---

## 9. Task 7 — B1-3：在小网络上选 `f'`（三点，判据已数值化）

`f' ∈ {0.5, 1.0, 2.0}`，**只在这三点里选**，不做连续搜索。**`f' = 1.0` 是 primary normalization**，
0.5 / 2.0 是对称的 sensitivity bracket。

> **rev3 改动（第三轮审阅）**：旧候选集 `{1.0, 0.5, 0.25}` 里 `0.25` 在间期只抬 `E_K` 0.31 mV（2% `V_th`），
> 过弱；`f' = 4` 在持续 50 Hz 下抬 22.7 mV（126% `V_th`），超出约化 LIF 的有效区。故取 `{0.5, 1.0, 2.0}`。

### 9.1 小网络网格必须按 occupancy 对齐 40k（P1）

40k 在 32×32 上是 **39.1 细胞/格**；小网络沿用 32×32 会大量空格，离子局部统计不可比。按同一 occupancy 缩放：

| 网络 | 网格 | 细胞/格 |
|---|---|---:|
| N ≈ 1000 | **5 × 5** | 40.0 |
| N ≈ 4000 | **10 × 10** | 40.0 |
| N = 40000 | **32 × 32** | 39.1 |

### 9.2 刺激协议（预注册，执行前锁死）

同一小网络、同一噪声种子，**全部三段都从异质解析预平衡态起步**（`tau_Na = 54.4 s`，spec §3.2b/§4.2c）。

小网络**不能**直接用 T2 那个 40k 率场（网络规模与网格都不同）。本 task 先跑一段**同配置、离子层关闭**的
短静息运行取本网络自己的逐细胞率场，再喂给 `heterogeneous_steady_state`；该率场与其 sha256
一并写入 `b1_f_selection.json`。5×5 / 10×10 的 occupancy 已按 §9.1 对齐 40k（40.0 vs 39.1 细胞/格），
正常情况下没有空格，但**空格分支仍须走 §4.2 的合同**（不得因"小网络没空格"而不实现）：

1. **静息段** 20 s，无刺激 —— 量 `Na_i`/`K_o` 的基线波动；
2. **单事件段**：`t = 25 s` 施加一次局部 kick（沿用既有 `kick_center`/`R_KICK`/`DUR_KICK`），之后 **30 s** 无刺激；
3. **事件簇段**：`t = 60 s` 起以 **200 ms 间隔连打 5 次**同样 kick，之后 **30 s** 无刺激。

### 9.3 数值判据与 tie-break（rev3 重写：旧的三条门全部不可靠）

**先说为什么重写。** 用锁定的常数把 rev2 的三条门各算一遍，三条**都**在构造上失效：

| 旧门 | 实算 | 失效方式 |
|---|---|---|
| `ΔK_o ≥ 5σ` | `σ` 由初始化质量决定，没有下界 | 初始化越好 `σ` 越小，门越容易过；**测的是初始化，不是钾信号** |
| `Na_i` 20 s 内回到静息 **2%** | 单事件单细胞 `ΔNa` = 0.015–0.087 mM，门 = **0.36 mM** | 门比它要检验的信号大 **4–25 倍**，恒过 |
| `Na_i` 20 s 内"回到" | `tau_Na = 54.4 s`，20 s 只衰减 **30.8%** | 若门收紧到真能卡住，则**构造上不可满足** |
| `ΔK_o` 第 5 次 ≥ 第 1 次 **1.5×** | `tau_Ko = 0.655 s`，200 ms 间隔的**纯线性叠加**已给 **2.97×** | 门低于线性预测，过它**不构成**超线性累积的证据 |

根因是**两个变量差 83 倍时间常数**（`tau_Ko = 0.655 s`、`tau_Na = 54.4 s`），却被塞进同一个 20 s 窗口：
对 K 太长（30 个时间常数，残留 5e-14，不可能失败），对 Na 太短（0.37 个时间常数，不可能成功）。

**rev3 的门**——每条按各自时间常数定，且都有**绝对**下界（不只相对 `σ`）：

| 判据 | 变量 | 数值门 |
|---|---|---|
| **可测** | `K_o` | 单事件参与格 `ΔK_o` 峰值 ≥ `max(5σ_rest, 0.15 mM)`。绝对地板 0.15 mM = `E_K` 抬高 1 mV = 5.6% `V_th`，即"至少要能被膜看见" |
| **安全（新增，双侧）** | `K_o` | 同一峰值 ≤ **0.90 mM**（= `ΔE_K` 5.4 mV = 30% `V_th`）。**单次**间期事件就把膜推过阈值的三成，说明尺度选过头，不是好候选 |
| **会恢复 (K)** | `K_o` | 单事件后 **3 s**（= 4.6 `tau_Ko`）内回到该格静息均值 **1σ** 内 |
| **会恢复 (Na)** | `Na_i` | 判**事件诱发超额**（`Na_i(t) −` 该细胞 kick 前 5 s 的自身基线），**不是**判绝对浓度：(a) 峰后 20 s 内单调不增（允许 1σ 抖动），(b) 20 s 的衰减比例落在解析预测 30.8% 的 **[0.5×, 1.5×] = [15.4%, 46.2%]** 内。偏高=清除过快（`rho` 标定错），偏低/不降=积累失控 |
| **能积分** | `K_o` | 事件簇第 5 次 `ΔK_o` 峰值 / 第 1 次 ≥ **2.97×** 的 **0.8 倍 = 2.38×**（2.97 是 200 ms 间隔下的纯线性叠加预测）。同时报告实测比值与 2.97 的比，**> 1 才是超线性**，`b1_f_selection.json` 必须显式记录这一比值 |

**五条全过才 admissible。**

**tie-break（rev3 改）**：多个 `f'` 同时可接受时取**最接近 `f' = 1.0`** 者。
> rev2 的"取最大"会系统性偏向更强的钾正反馈——那正是最容易把后续 B3/B4 推成 runaway 的方向，
> 等于用一条 tie-break 规则预先偏置机制结论。primary 是 1.0，另外两点是 sensitivity。

**三点全不可接受** → `NO_GO_ION_SCALE`，不得扩大候选集、不得连续搜索、不得放宽上表任何一格。

产物 `b1_f_selection.json` 必须记录**三点 × 五条门的完整对照**（含每格的实测值与门值），不只写选中的那个。

## 10. Task 8 — B2-1：40k bias 重标定 + 一次闭合迭代

### 只允许两个 nuisance 参数

`I_bias_E`、`I_bias_I`（由 adapter 加在 `drive` 上，§1.3）。
**没有第三个 fallback**：两个 bias 无法恢复 baseline 即判 **NO-GO**（spec 停机条件 4）。

### 搜索策略与判决分档（P1，rev2 订正）

**rev1 把「12 次探针不收敛」直接判成科学 NO-GO 是错的** —— 那只证明校准器失败，不证明不存在可行解。

1. **预锁合法边界** `I_bias_E, I_bias_I ∈ [−2, +2]` 引擎单位（= mV；`V_th = 18`，即阈值的 ±11%）；
2. **共同噪声有限差分**估 2×2 rate-response Jacobian `∂(r_E, r_I)/∂(I_bias_E, I_bias_I)`（4 次探针，同一噪声实现）；
3. **bounded Newton / trust-region** 更新，最多两轮（每轮 2 次验证探针）；
4. 探针总数上限 **12 次**（4 s 短探针，约 6 min/次）。

| 情形 | 判决 |
|---|---|
| 进入容差 | `CALIBRATED` |
| 12 次用尽仍未收敛，但合法边界**未被完整括住** | **`UNRESOLVED_CALIBRATION`**（校准器失败，**不是**机制结论） |
| 合法边界被完整括住且区间内**不存在**可行解 | `NO_GO_BASELINE`（此时才是 spec 停机条件 4） |

⚠️ **4 s 探针只能估快速率响应，不能验收 Na 稳态**（`tau_Na = 54.4 s`）。探针阶段**必须**用 §4.2c 的
**异质**解析预平衡把 `Na_i`/`K_o` 置于该 bias 下的预测稳态（输入 = T2 的 `b0_baseline_rate_field.npz`，
按当前 bias 的率响应线性外推；引用其 sha256）；最终稳态检查只在 11 s 验证运行里做。

**块间趋势必须逐细胞 / 逐格报告 q95/q99/max，不能只报群体均值**——`tau_Na` 是 11 s 窗口的 5 倍，
均值平稳完全可以掩盖一个还没走完的空间重排。11 s 只能验证"初始化点附近稳定"，
**不能**证明系统已达稳态；`b2_closure_iteration.json` 必须显式写下这条限制。

### 一次闭合迭代（spec §7.1）

锁 `r0 = 4.1581 Hz` → 调 bias → 测 `r0'` → **重算一次** `q_ion`（`J_Na,0`/`J_K,0` 是常数，不参与迭代）→ 再跑一次 11 s 验证 →
**冻结**。硬门 `|r0' − r0|/r0 ≤ 10%`，且离子变量块间无显著趋势；不达标按上表分档判决。

产物：`b2_bias_calibration.json`、`b2_closure_iteration.json`。

---

## 11. Task 9 — B2-2：Gate B 判决 + confirmatory

### B-real（优先级 1，binding）

| 判据 | 真实目标产物 | 层级 |
|---|---|---|
| 两个稳定传播模板存在，模型两核对应其 source foci | `results/interictal_propagation_masked/per_subject/epilepsiae_1146.json` → `adaptive_cluster.stable_k` | **模板层** |
| 间期稀疏性与不规则性 | 同上 `propagation_stereotypy` / `temporal_dynamics` | 模板层 |
| 事件在两个注册核（`core_A` / `core_B`）上都会起始 | T2 锁定的起始位点读出 | 模型侧 |

**方向判据的数值门（P1；rev1 的 `frac_A>0 && frac_B>0` 太弱，一次偶然事件即可过门）**：

```
每条轨迹：n_scoreable >= 20  且  min(frac_A, frac_B) >= 0.15
6 条 confirmatory 聚合：>= 5/6 条满足上式（多数不够，必须近乎一致）
```

阈值 0.15 在 T2 功率前置通过后、于 development seed 401 上**预锁**，confirmatory 只判一次，不得回头调。

> **层级纪律（spec §9 rev3 已逐字段核实）**：E1146 模板层 `candidate_forward_reverse_pairs = []`（空列表）；
> 通道层**有** `swap_class = strict`（`decision_k = 7`，`p = 0.025`），但 `fwd_rev_reproduced = None`
> ——复现性合同未确认，且通道层属描述性 / 机制兜底档。
> 因此**禁止**写成「复现了双向传播」（rev1 的错），也**禁止**写成「E1146 没有互换证据」（rev2 的错）。
> 允许的只有：「保留了两个稳定模板对应的两个起始核」。

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
- **T7 = `NO_GO_ION_SCALE`**：三个 `f'` 都不满足可测/安全/恢复(K)/恢复(Na)/积分 五条门 → 归档，
  不得扩大候选集、不得改 tie-break、不得放宽任何一格；
- **T8 = `UNRESOLVED_CALIBRATION`**：校准器未收敛但可行域未被排除 → **不是**机制结论，
  须另行改进校准器，不得写成 baseline NO-GO；
- **Gate H fail**：离子稳态或数值合同不成立 → 不进入 40k。含两个 rev3 新增分支：
  `FAIL_EMPTY_VOXEL`（空格不是不动点）与 `FAIL_INIT_RESIDUAL`（异质初始化残差过不了 q95/q99/max）；
- **Gate B fail**：构成性离子 substrate 拿不回合法间期工作点 → 归档为
  "constitutive Na/K on this substrate cannot recover the interictal working point"，
  **不是**"离子机制被否定"。

**触发 spec §11 任一停机条件即停并归档，不得自动转入下一机制、不得扩大网格、
不得动 B3/B4，也不得引入第三个 nuisance 参数。**
