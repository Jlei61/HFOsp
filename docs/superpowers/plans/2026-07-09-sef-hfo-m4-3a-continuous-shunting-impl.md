# M4-3A 连续 shunting 恢复变量（n→a）实现 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 SNN 引擎加一个连续、活动驱动、baseline-centered、**电导型 shunting** 的慢恢复变量 `n→a`，并搭出 M4-3A go/no-go sweep（含 P0 离线标定、early/late 两窗 retrigger、机制 ablation），回答"这个慢变量能不能 clean-terminate M4-1 有界持续态"。

**Architecture:** `n(x,t)` 是活动负荷场（照 `q_I`/`g_K` 模式进 `SpatialSlowField`），`a=a_max·Π(n)` 是 shunt 强度。**form (A) 电导型**：`a` 经 `g_A=α_A·a` 进 `kick_probe` 的 M4 内联膜更新（带 reversal `E_A` 结构上不去抑制），次要减法偏置 `-η_A·a` 进 `apply_currents`。ODE 核心抽成独立模块 `src/sef_hfo_m4_load_shunt.py`，**离线 P0 标定与在线网络共用同一实现**（DRY）。sweep/retrigger 复用 M4-2 的 `--p1-sweep` 机器，只换机制变量并扩 early 窗。

**Tech Stack:** Python 3, numpy, pytest, multiprocessing (fork-COW)。引擎 `src/snn_engine/{slow_field,kick_probe}.py`；判读 `src/sef_hfo_m4_termination.py`；runner `scripts/run_m4_dynamic_qi.py`。

**Spec of record:** `docs/superpowers/specs/2026-07-09-sef-hfo-m4-3-continuous-shunting-axis-coordinate-design.md`（rev2）。本 plan 只实现 **M4-3A**（spec §3.1/§3.2 form A、§4、§6、§7、§8）。**M4-3B（`K_graph=F(W_EE)` graph-kernel + 三负控）是独立子系统、另开 plan**（spec §9）。

**Worktree root:** `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-m4-divisive-sg/`（所有相对路径以此为根；从此根运行命令）。

## Global Constraints

每个 task 的要求都隐含包含本节。以下值逐字来自 spec，实现时不得改动：

- **Parity 红线（承重）：** `use_A=False`（或 `k_n=0.0`）时引擎必须**逐字节等旧引擎**——照现有 `use_qI`/`use_SG` 的 `use_<var> bool + 零值 strength 旋钮` 门控模式（`slow_field.py` 模块 docstring：`k_q=0,k_K=0,q_init=1 → byte-identical to slow=None`）。
- **`uses_shunt()` 定义（P1-1 修：parity gate 的唯一判据）：** `uses_shunt() ⟺ (use_A AND k_n≠0 AND alpha_A≠0)`——**必须含 `k_n≠0`**：`k_n=0` 时 `a` 恒为 0，`kick_probe` 走**literal 旧路**（不进电导分支，避免 `decay_V**(1+0)` 与 `decay_V` 的非位等）。parity gate 一律以 `uses_shunt()==False` 判定；不要用 `use_A and alpha_A` 漏掉 `k_n`。
- **re-bless（P1-2 修：真实路径 + 命令）：** guard = `src/sef_hfo_snn_engine_guard.py`（`record_versions`/`assert_versions`）；快照文件 = **`results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json`**（git-tracked，非 ignored）；gate 测试 = `tests/test_a1c_feedback.py::test_T8_engine_blessed`。**只有改 `kick_probe.py`（Task 5）才 re-bless**——`slow_field.py` **不在** guard 列表（`kick_probe/params/model/connectivity/connectivity_rot/lfp`），故 Task 4 无需 re-bless。re-bless 前 off-path parity 测试必须先过。
- **铁律 D2：`a` 绝不整体除 signed net current。** form (A) 电导：`V_inf=(I_net+g_A·E_A)/(1+g_A)`，`E_A` = 阈下 reversal（复用引擎 `e_gaba`）。含抑制的 `I_net` 被 reversal **夹回静息**、不产生去抑制。
- **M4 工作点（固定，不扫）：** `k_q=0.10, alpha_G=16, ee_std off, g_K off`（`use_gK=False, k_K=0.0`）。核对 `run_m4_dynamic_qi.py` cfg build（约 line 184）。
- **baseline-center（D8）：** `ũ_n=[u_n-u_{n,0}]_+`（rectified），`u_{n,0}` = Arm0 安静间期长期均值的**固定常数**（非在线、非 ictal sensor）。
- **Clamps / caps：** `n∈[n_min,n_max]`，`a∈[0,a_max]`，`g_A∈[0,g_A_max]`。ODE 用 forward-Euler，`dt/τ` 单位统一为 **ms**（引擎 dt=0.1ms）。
- **命名防撞：** **绝不**把负荷时标命名 `tau_a`——`tau_a` 已是现有 rate-EMA 常数（`slow_field.py`：`1-exp(-dt/tau_a)`）。负荷时标用 `tau_n`；场变量用 `self.n_load` / `self.a_shunt`；config 前缀清晰。引擎命名约定 = `use_<var>` + 零值 strength 旋钮（照 `use_qI`/`use_gK`/`use_SG`）。
- **T 两级（D9）：** discovery `T=15000`；acceptance `T=40000` + post-offset 静默 10–20s 无 rebound/runaway（候选 go / 边界 cell 必跑）。
- **seed 分母（D10）：** primary = Arm0 判为 `bounded-persist` 的 seed（**计算得出**，每 seed 同 run 内跑 Arm0→分类）；Arm0=fragment/runaway 的 seed（M4-2 里 seed 4）只作 stress，不进 primary go-fraction。
- **go 判据（D5/§7）：** `go(cell) = terminate_clean AND retrigger_early=="attenuated" AND retrigger_probe=="reignite_bounded"`（`retrigger_probe` = late 窗；Task 7 additive schema）。
- **sensor-free 数值门（§6.1）：** `Δa_IED>0` 且 `≥2·σ_baseline` 且 `≥0.5%·a_max`；`⟨a⟩_interictal < a_block`；`R_A=Δa_ictal(1s)/Δa_IED ≥ 5`（起步 bar）。
- **σ_n（§3.3）：** 默认 `sigma_n = sigma_q = 1.5`（宽）；次级扫 `{0.75, 1.5, 2.25}`。
- **Framing 锁：** 结果措辞 "actual M4-3A **SIMULATION**"，绝不 "real data"。no-go 是**合法**结果（加强怀疑、非已证 D_EE，须先过 M4-3B smoke）。

---

## File Structure

- **`src/sef_hfo_m4_load_shunt.py`** (Create) — 独立 ODE 核心：`LoadShuntParams`、`hill_pi`、`load_shunt_step`、`event_triggered_a_response`、`compute_R_A`。离线 P0 与在线 `SpatialSlowField.step` 共用。单一职责 = "负荷→shunt 的 elementwise 动力学 + a-响应度量"。
- **`scripts/run_m43a_p0_calibration.py`** (Create) — P0a 离线标定 CLI（**proxy**）：喂 rate trace 过 ODE，出 4-regime 表 + sensor-free 硬门（`Δa_IED`/`R_A`/`soft_gate_fail`/`sensor_free_pass`），写标定 params JSON。
- **`src/snn_engine/slow_field.py`** (Modify) — `SpatialSlowFieldConfig` 加 n/a config；`__init__` alloc `n_load`/`a_shunt`/`_Kn`/traces；`step` 加 n/a 演化（门控 `use_A and k_n!=0`）；`apply_currents` 加 `-η_A·a` 减法项；新增 `uses_shunt()`/`shunt_g_at_E()` 访问器。
- **`src/snn_engine/kick_probe.py`** (Modify) — M4 内联膜更新（约 line 316-317）改电导 shunt；re-bless `results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json`（guard=`src/sef_hfo_snn_engine_guard.py`）。
- **`src/sef_hfo_m4_termination.py`** (Modify) — `retrigger_verdict` 拆 4 态；`run_cell_with_retrigger` 扩 early+late 两窗。
- **`scripts/run_m4_dynamic_qi.py`** (Modify) — 加 `--m43a-sweep`（(α_A×τ_n) 网格 + per-seed Arm0 + go early/late + 诊断记录）与 `--m43a-ablation`（shunt/subtractive/hybrid）。
- **Tests:** `tests/test_m4_load_shunt.py`、`tests/test_slow_field_na.py`、`tests/test_kick_probe_shunt.py`、`tests/test_m4_termination_early_late.py`、`tests/test_m43a_runner_smoke.py`。

**注意（AGENTS.md）：** 下文行号来自 2026-07-09 核对，names drift——每次 Modify 前先 grep 确认函数位置，别硬信行号。

---

### Task 1: 负荷→shunt ODE 核心（`sef_hfo_m4_load_shunt.py`）

**Files:**
- Create: `src/sef_hfo_m4_load_shunt.py`
- Test: `tests/test_m4_load_shunt.py`

**Interfaces:**
- Produces: `LoadShuntParams`（frozen dataclass，字段 `tau_n,k_n,rho_n,n_base,n50,hill_h,a_max,u_n0,n_min,n_max`）；`hill_pi(n, p) -> ndarray`；`load_shunt_step(n, u_n, dt, p) -> (n_new, a)`（elementwise，标量/1D/2D 通用）。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_m4_load_shunt.py
import numpy as np
import pytest
from src.sef_hfo_m4_load_shunt import LoadShuntParams, hill_pi, load_shunt_step


def _p(**kw):
    base = dict(tau_n=20000.0, k_n=1.0, rho_n=0.0, n_base=0.0, n50=0.5,
                hill_h=2.0, a_max=1.0, u_n0=0.0, n_min=0.0, n_max=10.0)
    base.update(kw)
    return LoadShuntParams(**base)


def test_hill_pi_monotone_and_bounded():
    p = _p()
    xs = np.array([0.0, 0.25, 0.5, 1.0, 4.0])
    pi = hill_pi(xs, p)
    assert np.all(np.diff(pi) > 0)          # strictly increasing in n
    assert pi[0] == 0.0                       # at n_base -> 0
    assert np.all(pi < 1.0) and pi[-1] > 0.9  # saturates toward 1


def test_baseline_center_rectifies_subthreshold_drive():
    # u_n below the set-point u_n0 must not build load (rectified to 0)
    p = _p(u_n0=0.3, k_n=5.0)
    n, a = load_shunt_step(np.array(0.0), np.array(0.2), dt=1.0, p=p)
    assert n == 0.0 and a == 0.0              # drive 0.2 < u_n0 0.3 -> no build, decays/stays at n_base


def test_quiet_baseline_decays_to_n_base_and_a_zero():
    p = _p(n_base=0.0, u_n0=0.0)
    n = np.array(2.0)                          # start elevated
    for _ in range(200000):                    # many ms of zero drive
        n, a = load_shunt_step(n, np.array(0.0), dt=1.0, p=p)
    assert n == pytest.approx(0.0, abs=1e-3)
    assert a == pytest.approx(0.0, abs=1e-3)


def test_sustained_drive_accumulates_load_and_shunt():
    p = _p(u_n0=0.0, k_n=1.0, tau_n=20000.0)
    n = np.array(0.0)
    for _ in range(5000):                       # 5 s of sustained drive
        n, a = load_shunt_step(n, np.array(1.0), dt=1.0, p=p)
    assert n > 0.3 and a > 0.1                  # load and shunt rose measurably


def test_clamps_hold():
    p = _p(k_n=1e6, n_max=3.0, a_max=0.8)
    n, a = load_shunt_step(np.array(0.0), np.array(1.0), dt=1.0, p=p)
    assert n <= 3.0 and 0.0 <= a <= 0.8


def test_validate_rejects_bad_params():
    with pytest.raises(ValueError):
        _p(tau_n=0.0).validate()
    with pytest.raises(ValueError):
        _p(n50=0.0).validate()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_m4_load_shunt.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.sef_hfo_m4_load_shunt'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sef_hfo_m4_load_shunt.py
"""M4-3A load->shunt recovery variable (n -> a).

Continuous, activity-driven, baseline-centered slow variable. `n` is an abstract
activity load; `a = a_max * Pi(n)` is the shunt strength. This elementwise ODE is
shared by the offline P0 calibration AND SpatialSlowField.step (network) so both
paths use one implementation (DRY).

Sign contract (spec D2): `a` NEVER divides a signed net current. This module only
produces `a`; the divisive (conductance) / subtractive coupling lives in the engine.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LoadShuntParams:
    tau_n: float          # ms, load recovery toward n_base (SLOW; > tau_q)
    k_n: float            # load build rate on baseline-centered drive
    rho_n: float          # load consumption via Pi(n)
    n_base: float         # baseline n0 (Hill offset)
    n50: float            # Hill half-point (on n - n_base)
    hill_h: float         # Hill exponent
    a_max: float          # shunt ceiling
    u_n0: float = 0.0     # baseline drive set-point (homeostatic constant, from Arm0)
    n_min: float = 0.0    # clamp
    n_max: float = 10.0   # clamp

    def validate(self) -> None:
        if self.tau_n <= 0:
            raise ValueError("tau_n must be > 0")
        if self.n50 <= 0:
            raise ValueError("n50 must be > 0")
        if self.hill_h <= 0:
            raise ValueError("hill_h must be > 0")
        if self.a_max < 0:
            raise ValueError("a_max must be >= 0")
        if self.n_min > self.n_max:
            raise ValueError("n_min must be <= n_max")


def hill_pi(n, p: LoadShuntParams):
    """Continuous pump/conductance activation Pi(n) in [0,1). NOT a seizure sensor."""
    x = np.maximum(np.asarray(n, float) - p.n_base, 0.0) ** p.hill_h
    return x / (p.n50 ** p.hill_h + x)


def load_shunt_step(n, u_n, dt: float, p: LoadShuntParams):
    """One forward-Euler step of the load ODE + shunt readout.

    dn/dt = -(n - n_base)/tau_n + k_n * [u_n - u_n0]_+ - rho_n * Pi(n)
    a     = a_max * Pi(n_new)

    Elementwise: works on scalars, 1D traces (P0), or 2D grids (SpatialSlowField).
    Returns (n_new, a), both clamped.
    """
    n = np.asarray(n, float)
    u_tilde = np.maximum(np.asarray(u_n, float) - p.u_n0, 0.0)   # baseline-centered, rectified
    dn = -(n - p.n_base) / p.tau_n + p.k_n * u_tilde - p.rho_n * hill_pi(n, p)
    n_new = np.clip(n + dt * dn, p.n_min, p.n_max)
    a = np.clip(p.a_max * hill_pi(n_new, p), 0.0, p.a_max)
    return n_new, a
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_m4_load_shunt.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_m4_load_shunt.py tests/test_m4_load_shunt.py
git commit -m "feat(m4-3a): load->shunt ODE core (n->a, baseline-centered)"
```

---

### Task 2: a-响应度量（`Δa_IED` / `R_A`，§6.1 sensor-free 门）

**Files:**
- Modify: `src/sef_hfo_m4_load_shunt.py`（追加 2 个函数）
- Test: `tests/test_m4_load_shunt.py`（追加）

**Interfaces:**
- Consumes: 无（对 1D `a` trace + 事件索引操作）。
- Produces: `event_triggered_a_response(a_trace, event_idx, dt, *, pre_ms, post0_ms, post1_ms) -> float`；`compute_R_A(delta_a_ictal, delta_a_ied) -> float`。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_m4_load_shunt.py  (append)
from src.sef_hfo_m4_load_shunt import event_triggered_a_response, compute_R_A


def test_event_triggered_a_response_positive_bump():
    dt = 1.0
    a = np.zeros(1000)
    a[500:600] = 0.4                            # a bump right after t=500
    delta = event_triggered_a_response(a, [500], dt, pre_ms=100, post0_ms=10, post1_ms=90)
    assert delta == pytest.approx(0.4, abs=1e-9)


def test_event_triggered_skips_events_without_full_window():
    dt = 1.0
    a = np.zeros(200)
    # event at 10 has no room for pre=100 -> skipped; event at 100 is fine
    a[100:150] = 0.2
    delta = event_triggered_a_response(a, [10, 100], dt, pre_ms=100, post0_ms=0, post1_ms=50)
    assert delta == pytest.approx(0.2, abs=1e-9)


def test_event_triggered_raises_when_no_usable_event():
    a = np.zeros(50)
    with pytest.raises(ValueError):
        event_triggered_a_response(a, [10], 1.0, pre_ms=100, post0_ms=0, post1_ms=50)


def test_R_A_ratio_and_soft_gate_flag():
    assert compute_R_A(0.5, 0.05) == pytest.approx(10.0)
    assert compute_R_A(0.5, 0.0) == float("inf")     # IED did not move a -> soft ictal gate
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_m4_load_shunt.py -q -k "event_triggered or R_A"`
Expected: FAIL with `ImportError: cannot import name 'event_triggered_a_response'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sef_hfo_m4_load_shunt.py  (append)
def event_triggered_a_response(a_trace, event_idx, dt, *, pre_ms, post0_ms, post1_ms):
    """Mean a in [t+post0, t+post1] minus mean a in [t-pre, t], averaged over events.
    a_trace is 1D (per-step). Events without a full pre/post window are skipped.
    Raises if no event has a full window."""
    a = np.asarray(a_trace, float)
    pre = int(round(pre_ms / dt)); p0 = int(round(post0_ms / dt)); p1 = int(round(post1_ms / dt))
    deltas = []
    for i in event_idx:
        i = int(i)
        if i - pre < 0 or i + p1 > a.size:
            continue
        deltas.append(a[i + p0:i + p1].mean() - a[i - pre:i].mean())
    if not deltas:
        raise ValueError("no events with a full pre/post window")
    return float(np.mean(deltas))


def compute_R_A(delta_a_ictal, delta_a_ied):
    """Duty-cycle ratio: sustained-ictal a-accumulation over single-IED a-bump.
    R_A >> 1 is the sensor-free signature (spec 6.1). Returns inf when the IED did
    not move a at all (delta_a_ied <= 0) -> caller flags a soft ictal gate."""
    if delta_a_ied <= 0:
        return float("inf")
    return float(delta_a_ictal / delta_a_ied)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_m4_load_shunt.py -q`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_m4_load_shunt.py tests/test_m4_load_shunt.py
git commit -m "feat(m4-3a): event-triggered a-response + R_A (sensor-free gate)"
```

---

### Task 3: P0a 离线标定 CLI（proxy；`run_m43a_p0_calibration.py`）

> **P1-4（proxy 标注）：** 在线定义是 `u_n = K_n * r_E`（空间慢场驱动）；本 task 的 runbook 喂的是 `p1_sweep_traces.npz:*__rate`（全局 sheet-mean rate），是 **cheap proxy = P0a**，可探参数形状，但**不能锁最终 `u_n0`/`n50`**。真正锁参数前必须用 Task 4 的 `trace_un_mean`（field-derived `u_n`）从 Arm0 导出/重放（= P0b，见 Task 10 runbook）。

**Files:**
- Create: `scripts/run_m43a_p0_calibration.py`
- Test: `tests/test_m43a_p0_calibration.py`

**Interfaces:**
- Consumes: `LoadShuntParams`, `load_shunt_step`, `event_triggered_a_response`, `compute_R_A`（Task 1/2）。
- Produces: `run_a_trace(u_series, dt, p) -> a_series`（1D→1D）；`calibrate_regimes(regime_series, dt, p, event_idx=None, a_block=None) -> {table, metrics, gate}`。**`gate`（P1-3 sensor-free 硬门）字段** = `sigma_baseline, delta_a_ied, soft_gate_fail, baseline_jitter_pass, magnitude_pass, delta_ied_pass, R_A, R_A_pass, a_interictal_mean, interictal_block_pass, sensor_free_pass`。CLI 写 `<out>/p0_calibration.json`。

> **P1-3 铁律：** `Δa_IED ≤ 0`（IED 完全不冲 `a`）= 软 ictal gate = **硬 fail**，`soft_gate_fail=True`、`sensor_free_pass=False`——**不能靠 `compute_R_A` 的 `inf` 过门**（`R_A_pass` 显式要求 `delta_a_ied > 0`）。`interictal_block_pass` 需要网络测得的 `a_block`；`a_block=None` 时为 `None`（pending，非 pass），`sensor_free_pass` 因此为 `False`（P0a proxy 不能完全认证）。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_m43a_p0_calibration.py
import numpy as np
import pytest
from src.sef_hfo_m4_load_shunt import LoadShuntParams
from scripts.run_m43a_p0_calibration import run_a_trace, calibrate_regimes


def _p():
    # rho_n=0 so sustained ictal keeps accumulating (R_A>1); k_n small so a single IED
    # bumps a measurably WITHOUT saturating (keeps the gate booleans meaningful).
    return LoadShuntParams(tau_n=20000.0, k_n=0.0008, rho_n=0.0, n_base=0.0, n50=0.4,
                           hill_h=2.0, a_max=1.0, u_n0=0.05, n_min=0.0, n_max=10.0)


def test_run_a_trace_shapes_and_quiet_stays_low():
    dt = 1.0
    u_quiet = np.full(3000, 0.05)                   # == u_n0 -> rectified to ~0 drive
    a = run_a_trace(u_quiet, dt, _p())
    assert a.shape == u_quiet.shape
    assert a.max() < 1e-2                            # quiet baseline: a low & stable


def test_calibrate_regimes_table_directions():
    dt = 1.0
    quiet = np.full(6000, 0.05)
    ied = quiet.copy(); ied[3000:3050] = 1.5        # one brief high-duty spike
    ictal = np.concatenate([np.full(1000, 0.05), np.full(5000, 1.2)])  # sustained
    post = np.concatenate([np.full(3000, 1.2), np.full(3000, 0.05)])   # drops to quiet
    out = calibrate_regimes(
        {"quiet": quiet, "isolated_ied": ied, "bounded_ictal": ictal, "post_offset": post},
        dt, _p(), event_idx={"isolated_ied": [3000]})
    assert out["table"]["quiet"]["a_max"] < out["table"]["bounded_ictal"]["a_max"]
    assert out["table"]["isolated_ied"]["delta_a_ied"] > 0        # IED nudges a
    assert out["metrics"]["R_A"] > 1.0                            # sustained >> single IED
    assert out["table"]["post_offset"]["a_end"] <= out["table"]["post_offset"]["a_mid"]  # decays (never rises)
    assert out["gate"]["soft_gate_fail"] is False                # IED did move a


def test_soft_ictal_gate_hard_fails_even_if_R_A_inf():
    # P1-3: an IED that does NOT move a (delta<=0) must HARD-fail, not pass via R_A==inf.
    dt = 1.0
    quiet = np.full(6000, 0.05)
    ied = quiet.copy()                                            # identical to quiet -> no a bump at all
    ictal = np.concatenate([np.full(1000, 0.05), np.full(5000, 1.2)])
    out = calibrate_regimes({"quiet": quiet, "isolated_ied": ied, "bounded_ictal": ictal},
                            dt, _p(), event_idx={"isolated_ied": [3000]})
    assert out["gate"]["delta_a_ied"] <= 0.0
    assert out["gate"]["soft_gate_fail"] is True
    assert out["gate"]["R_A_pass"] is False                       # inf must NOT sneak through
    assert out["gate"]["sensor_free_pass"] is False


def test_interictal_block_pass_requires_a_block():
    dt = 1.0
    quiet = np.full(6000, 0.05); ied = quiet.copy(); ied[3000:3050] = 1.5
    ictal = np.concatenate([np.full(1000, 0.05), np.full(5000, 1.2)])
    out = calibrate_regimes({"quiet": quiet, "isolated_ied": ied, "bounded_ictal": ictal},
                            dt, _p(), event_idx={"isolated_ied": [3000]}, a_block=None)
    assert out["gate"]["interictal_block_pass"] is None           # can't certify without network a_block
    assert out["gate"]["sensor_free_pass"] is False               # P0a proxy cannot fully pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_m43a_p0_calibration.py -q`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` on `run_m43a_p0_calibration`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/run_m43a_p0_calibration.py
"""P0a offline calibration for the M4-3A load->shunt variable (spec 4.0, 6.1).

Feed real M4-1 Arm0 rate traces (4 regimes) through the n->a ODE WITHOUT running the
network, check a(t) against the calibration table, and evaluate the sensor-free HARD
gate (Delta_a_IED, R_A). Produces calibrated params so the network sweep (Task 8) does
not waste budget, and so n50 is not a soft ictal gate.

PROXY (P1-4): the runbook feeds '{label}__rate' (global sheet-mean rate) as u_n, which
is a cheap proxy (= P0a). Locking u_n0/n50 needs the field-derived u_n (Task 4
trace_un_mean) from an Arm0 replay (= P0b). Trace source:
results/topic4_m4_dynamic_p1_sweep*/p1_sweep_traces.npz (Arm0 label = 'p1_arm0').
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from src.sef_hfo_m4_load_shunt import (
    LoadShuntParams, load_shunt_step, event_triggered_a_response, compute_R_A)


def run_a_trace(u_series, dt, p: LoadShuntParams):
    """Integrate the ODE over a 1D drive series; return the a(t) series."""
    u = np.asarray(u_series, float)
    n = p.n_base
    a_out = np.empty(u.size, dtype=float)
    for t in range(u.size):
        n, a = load_shunt_step(n, u[t], dt, p)
        a_out[t] = float(a)
    return a_out


def calibrate_regimes(regime_series, dt, p: LoadShuntParams, event_idx=None, a_block=None):
    """Run each regime's drive through the ODE, summarize a(t), and evaluate the sensor-free
    HARD gate (P1-3). a_block = network-measured a level that blocks an IED kick (P0b); None
    -> block check pending (not a pass). Returns {'table', 'metrics', 'gate'}."""
    event_idx = event_idx or {}
    table, a_by_regime = {}, {}
    for name, u in regime_series.items():
        a = run_a_trace(u, dt, p)
        a_by_regime[name] = a
        row = {"a_max": float(a.max()), "a_mean": float(a.mean()),
               "a_mid": float(a[a.size // 2]), "a_end": float(a[-1])}
        if name in event_idx:
            row["delta_a_ied"] = event_triggered_a_response(
                a, event_idx[name], dt, pre_ms=200, post0_ms=10, post1_ms=200)
        table[name] = row

    metrics, gate = {}, {}
    d = table.get("isolated_ied", {}).get("delta_a_ied")
    if d is not None and "bounded_ictal" in a_by_regime:
        ic = a_by_regime["bounded_ictal"]
        n1s = int(round(1000.0 / dt))
        delta_ictal = float(ic.max() - ic[:min(n1s, ic.size)].mean())   # 1s+ ictal accumulation
        metrics["R_A"] = compute_R_A(delta_ictal, d)
        sigma_bl = float(a_by_regime.get("quiet", ic).std())
        gate["sigma_baseline"] = sigma_bl
        gate["delta_a_ied"] = float(d)
        gate["soft_gate_fail"] = bool(d <= 0.0)                          # IED did not move a -> hidden sensor
        gate["baseline_jitter_pass"] = bool(d >= 2.0 * sigma_bl)
        gate["magnitude_pass"] = bool(d >= 0.005 * p.a_max)
        gate["delta_ied_pass"] = bool(d > 0.0 and gate["baseline_jitter_pass"] and gate["magnitude_pass"])
        gate["R_A"] = metrics["R_A"]
        gate["R_A_pass"] = bool(d > 0.0 and gate["R_A"] >= 5.0)          # d>0 guard: inf can't sneak through
        if "quiet" in a_by_regime:
            gate["a_interictal_mean"] = float(a_by_regime["quiet"].mean())
            gate["interictal_block_pass"] = (bool(gate["a_interictal_mean"] < a_block)
                                             if a_block is not None else None)
        else:
            gate["a_interictal_mean"] = None
            gate["interictal_block_pass"] = None
        gate["sensor_free_pass"] = bool(
            gate["delta_ied_pass"] and gate["R_A_pass"] and (not gate["soft_gate_fail"])
            and gate["interictal_block_pass"] is True)                   # None (pending) / False -> not certified
    return {"table": table, "metrics": metrics, "gate": gate}


def _load_regime(npz_path, key, seg=None):
    arr = np.asarray(np.load(npz_path)[key], float)
    if seg is not None:
        arr = arr[seg[0]:seg[1]]
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--tau-n", type=float, default=20000.0)
    ap.add_argument("--k-n", type=float, default=1.0)
    ap.add_argument("--rho-n", type=float, default=0.1)
    ap.add_argument("--n50", type=float, default=0.4)
    ap.add_argument("--hill-h", type=float, default=2.0)
    ap.add_argument("--a-max", type=float, default=1.0)
    ap.add_argument("--u-n0", type=float, default=0.0)
    ap.add_argument("--a-block", type=float, default=None)   # P0b: network-measured IED-block a level
    # --regime name=npz:key[:start:end]  (repeatable); --event name=idx[,idx]
    ap.add_argument("--regime", action="append", default=[])
    ap.add_argument("--event", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    p = LoadShuntParams(tau_n=a.tau_n, k_n=a.k_n, rho_n=a.rho_n, n_base=0.0, n50=a.n50,
                        hill_h=a.hill_h, a_max=a.a_max, u_n0=a.u_n0)
    p.validate()
    regimes = {}
    for spec in a.regime:
        name, rhs = spec.split("=", 1)
        parts = rhs.split(":")
        npz, key = parts[0], parts[1]
        seg = (int(parts[2]), int(parts[3])) if len(parts) == 4 else None
        regimes[name] = _load_regime(npz, key, seg)
    events = {}
    for spec in a.event:
        name, idx = spec.split("=", 1)
        events[name] = [int(x) for x in idx.split(",")]
    out = calibrate_regimes(regimes, a.dt, p, events, a_block=a.a_block)
    out["params"] = p.__dict__
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "p0_calibration.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_m43a_p0_calibration.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/run_m43a_p0_calibration.py tests/test_m43a_p0_calibration.py
git commit -m "feat(m4-3a): P0a offline n->a calibration + sensor-free hard gate"
```

---

### Task 4: `SpatialSlowField` 加 n/a 场（config/alloc/step/traces/accessors + η_A 减法项）

**Files:**
- Modify: `src/snn_engine/slow_field.py`（`SpatialSlowFieldConfig` 约 line 50-93；`__init__` 约 230-237；`apply_currents` 约 254-278；`step` 约 285-311；`validate` 约 108-110）
- Test: `tests/test_slow_field_na.py`

**Interfaces:**
- Consumes: `LoadShuntParams`, `load_shunt_step`（Task 1）。
- Produces: config 字段 `use_A,k_n,tau_n,rho_n,n_base,n50,hill_h,a_max,alpha_A,eta_A,sigma_n,u_n0,n_min,n_max,g_A_max`；`self.n_load`/`self.a_shunt`/`self.trace_n_mean`/`self.trace_a_mean`；方法 `uses_shunt()->bool`、`shunt_g_at_E()->ndarray(nE)`。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_slow_field_na.py
import numpy as np
import pytest
from src.snn_engine.slow_field import SpatialSlowField, SpatialSlowFieldConfig


def _mk(**kw):
    # minimal field; reuse existing required ctor args as the codebase defines them.
    # (grep SpatialSlowField.__init__ for the exact signature: n_grid, L, posE, posI, nE, nI ...)
    cfg = SpatialSlowFieldConfig(use_qI=True, k_q=0.0, use_gK=False, k_K=0.0, **kw)
    return cfg


def test_shunt_off_by_default_is_byte_parity():
    # use_A default False -> a_shunt stays 0, n_load stays n_base, shunt_g_at_E all zero
    cfg = _mk()
    assert cfg.use_A is False
    fld = _make_field(cfg)                      # helper in this test file (see Step 3 note)
    for _ in range(100):
        fld.step(_zero_spikes(fld), dt=0.1)
    assert np.all(fld.a_shunt == 0.0)
    assert np.allclose(fld.n_load, cfg.n_base)
    assert np.all(fld.shunt_g_at_E() == 0.0)
    assert fld.uses_shunt() is False


def test_k_n_zero_is_parity_path_even_with_alpha_A():
    # P1-1: k_n=0 -> a==0 forever -> uses_shunt() MUST be False so kick_probe stays literal.
    cfg = _mk(use_A=True, k_n=0.0, alpha_A=2.0)
    fld = _make_field(cfg)
    for _ in range(100):
        fld.step(_driving_spikes(fld), dt=0.1)
    assert np.allclose(fld.n_load, cfg.n_base)   # k_n=0 -> load never evolves
    assert np.all(fld.a_shunt == 0.0)
    assert fld.uses_shunt() is False             # P1-1: k_n=0 keeps the byte-parity path


def test_uses_shunt_true_only_when_all_three_on():
    assert _make_field(_mk(use_A=True, k_n=1.0, alpha_A=2.0)).uses_shunt() is True
    assert _make_field(_mk(use_A=True, k_n=1.0, alpha_A=0.0)).uses_shunt() is False   # no conductance
    assert _make_field(_mk(use_A=False, k_n=1.0, alpha_A=2.0)).uses_shunt() is False  # field off


def test_shunt_g_clips_and_tracks_a():
    cfg = _mk(use_A=True, k_n=1.0, alpha_A=5.0, a_max=1.0, g_A_max=3.0)
    fld = _make_field(cfg)
    for _ in range(20000):                        # drive load up
        fld.step(_driving_spikes(fld), dt=0.1)
    g = fld.shunt_g_at_E()
    assert g.shape == (fld.nE,)
    assert np.all((g >= 0.0) & (g <= 3.0))        # clipped to g_A_max
    assert g.max() > 0.0                          # shunt engaged


def test_eta_A_subtractive_term_only_when_on():
    # apply_currents subtracts eta_A*a on E cells; with a==0 it's a no-op (parity)
    cfg = _mk(use_A=True, k_n=0.0, eta_A=1.0)     # a stays 0 -> no subtraction
    fld = _make_field(cfg)
    I_E = np.ones(fld.nE + fld.nI); I_I = np.zeros(fld.nE + fld.nI)
    out0 = fld.apply_currents(I_E.copy(), I_I.copy())
    assert np.allclose(out0[:fld.nE], I_E[:fld.nE])  # a==0 -> unchanged
```

> **Step 3 note for implementer:** add `_make_field(cfg)`, `_zero_spikes(fld)`, `_driving_spikes(fld)` helpers at the top of the test file. Grep `SpatialSlowField(` usage in existing tests (e.g. `tests/test_snn_*`, `tests/test_slow_field*`) to copy the exact constructor call (grid size, positions, nE/nI). Keep the field tiny (e.g. n_grid=16, nE≈64) for speed.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_slow_field_na.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'use_A'`

- [ ] **Step 3: Write minimal implementation**

In `SpatialSlowFieldConfig` (after the `g_K` block, ~line 69), add:

```python
    # ---- n(x,t) load -> a(x,t) shunt field, M4-3A ----
    use_A: bool = False        # master gate; False -> byte-parity
    k_n: float = 0.0           # load build rate; 0 -> OFF -> byte-parity
    tau_n: float = 20000.0     # ms, load recovery (SLOW; keep > tau_q)
    rho_n: float = 0.0         # load consumption via Pi(n)
    n_base: float = 0.0        # Hill offset / baseline load
    n50: float = 0.5           # Hill half-point
    hill_h: float = 2.0        # Hill exponent
    a_max: float = 1.0         # shunt ceiling
    alpha_A: float = 0.0       # divisive conductance gain: g_A = alpha_A * a
    eta_A: float = 0.0         # subtractive bias gain: -eta_A * a (E cells)
    sigma_n: float = 1.5       # mm, K_n width (default = sigma_q, WIDE)
    u_n0: float = 0.0          # baseline drive set-point (constant, from Arm0)
    n_min: float = 0.0
    n_max: float = 10.0
    g_A_max: float = 20.0      # conductance cap
```

In `validate()` (~line 108), append:

```python
        if self.use_A and self.sigma_n <= 0.0:
            raise ValueError("sigma_n must be > 0 when use_A")
```

At top of `slow_field.py` imports (~line 34), add:

```python
from src.sef_hfo_m4_load_shunt import LoadShuntParams, load_shunt_step
```

In `__init__` (after `self.g_K` alloc, ~line 231-237), add:

```python
        self.n_load = np.full((n, n), self.cfg.n_base, dtype=float)
        self.a_shunt = np.zeros((n, n), dtype=float)
        self._Kn = isotropic_gaussian(n, L, self.cfg.sigma_n)
        self.trace_n_mean = []
        self.trace_a_mean = []
        self.trace_un_mean = []                     # P1-4: field-derived drive u_n (for P0b param lock)
```

Add a params helper + accessors as methods on the class:

```python
    def _load_shunt_params(self):
        c = self.cfg
        return LoadShuntParams(tau_n=c.tau_n, k_n=c.k_n, rho_n=c.rho_n, n_base=c.n_base,
                               n50=c.n50, hill_h=c.hill_h, a_max=c.a_max, u_n0=c.u_n0,
                               n_min=c.n_min, n_max=c.n_max)

    def uses_shunt(self) -> bool:
        """True iff the conductance shunt actually couples (P1-1). Needs use_A AND k_n!=0
        (else a==0 forever -> must take the literal parity path in kick_probe) AND alpha_A!=0.
        The parity gate keys on uses_shunt()==False; do NOT drop the k_n!=0 term."""
        return bool(self.cfg.use_A and self.cfg.k_n != 0.0 and self.cfg.alpha_A != 0.0)

    def shunt_g_at_E(self) -> np.ndarray:
        """Per-E-neuron conductance g_A = alpha_A * a, clipped to [0, g_A_max].
        Returns zeros(nE) when shunt off -> kick_probe takes the literal parity branch."""
        if not self.uses_shunt():
            return np.zeros(self.nE, dtype=float)
        aE = self.a_shunt[self._iyE, self._ixE]
        return np.clip(self.cfg.alpha_A * aE, 0.0, self.cfg.g_A_max)
```

In `step()` (after the `g_K` block, before the `trace_qI_mean` append ~line 310), add:

```python
        if cfg.use_A:                                               # M4-3A load -> shunt
            u_n = convolve_periodic(self.rE, self._Kn)              # field-derived drive K_n * rE (EMA rE)
            self.trace_un_mean.append(float(u_n.mean()))            # P1-4: dump real u_n even when k_n=0 (P0b lock)
            if cfg.k_n != 0.0:                                      # evolve load ONLY when active -> parity when k_n=0
                self.n_load, self.a_shunt = load_shunt_step(self.n_load, u_n, dt, self._load_shunt_params())
        self.trace_n_mean.append(float(self.n_load.mean()))
        self.trace_a_mean.append(float(self.a_shunt.mean()))
```

> **Parity note (P1-4):** computing `u_n` (a convolution) when `use_A=True, k_n=0` is a **read-only** side calculation — it never feeds `load_shunt_step`, so `n_load`/`a_shunt`/spikes/`q_I` are byte-identical to the old engine. This lets a short Arm0 replay (`use_A=True, k_n=0, alpha_A=0, eta_A=0` → `uses_shunt()==False` → membrane parity) export the true field-derived `u_n` for P0b param-locking (P1-4). `trace_un_mean` is only appended under `use_A`; the parity red-line configs (`use_A=False`) leave it empty.

In `apply_currents()` E-cell block (after the `- eta_K*gK_E` term, ~line 266), add:

```python
        if self.cfg.use_A and self.cfg.eta_A != 0.0:               # M4-3A subtractive bias (E only)
            aE = self.a_shunt[self._iyE, self._ixE]
            out[:nE] -= self.cfg.eta_A * aE
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_slow_field_na.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Run the FULL existing slow_field / snn test suite for regression**

Run: `python -m pytest tests/ -q -k "slow_field or snn or m4"`
Expected: PASS — no regression (off-by-default fields must not change existing behavior).

- [ ] **Step 6: Commit**

```bash
git add src/snn_engine/slow_field.py tests/test_slow_field_na.py
git commit -m "feat(m4-3a): n->a slow field in SpatialSlowField (off-by-default parity)"
```

---

### Task 5: `kick_probe` M4 内联膜更新改电导 shunt（form A）+ re-bless

**Files:**
- Modify: `src/snn_engine/kick_probe.py`（M4 内联分支 `if slow is not None:` 约 line 316-317；`e_gaba` 解析约 line 110）
- Modify: `results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json`（P1-2：真实 guard 快照路径，git-tracked；re-bless）
- Test: `tests/test_kick_probe_shunt.py`（+ 复用 `tests/test_a1c_feedback.py::test_T8_engine_blessed` 作 re-bless gate）

**Interfaces:**
- Consumes: `slow.uses_shunt()`, `slow.shunt_g_at_E()`, `slow.nE`（Task 4）；`e_gaba`（= `p.E_gaba`，阈下 reversal）。
- Produces: 无新 public API；改膜更新语义。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kick_probe_shunt.py
import numpy as np
from src.snn_engine.kick_probe import simulate_kick   # confirm import path via grep


def test_shunt_off_matches_baseline_bit_exact():
    """use_A off (uses_shunt()==False): M4 path must be byte-identical to pre-change."""
    # Build two identical small M4 runs; one with the n/a field present but use_A off.
    # Assert spike rasters are bit-identical. (See Step 3 note for the fixture builder.)
    res_a, res_b = _run_pair(use_A=False)
    assert np.array_equal(res_a["spk"], res_b["spk"])


def test_shunt_on_pulls_membrane_toward_reversal():
    """With g_A>0 the effective V_inf moves toward E_A (rest) vs the un-shunted drive."""
    I_net = np.array([5.0, 5.0]); V = np.array([5.0, 5.0]); decay_V = 0.9
    g = np.array([0.0, 4.0]); E_A = 0.0
    V_inf = (I_net + g * E_A) / (1.0 + g)
    Vtmp = V_inf + (V - V_inf) * decay_V ** (1.0 + g)
    assert Vtmp[1] < Vtmp[0]                    # shunted cell driven closer to E_A
```

> **Step 3 note:** `_run_pair(use_A)` builds a tiny E1146-like M4 config (grep `run_m4_dynamic_qi.py` `run_arm` / `simulate_kick` call, ~line 192, for the arg set), runs `simulate_kick` twice with a fixed seed and `slow` = SpatialSlowField (use_A off in both), and returns the two result dicts. If constructing a full net is too heavy for a unit test, instead unit-test the branch math directly (as `test_shunt_on_pulls_membrane_toward_reversal` does) and rely on Task 10's smoke for the integrated parity check — but the off-path bit-parity assertion MUST run somewhere before re-bless.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_kick_probe_shunt.py -q`
Expected: FAIL (shunt not wired; or fixture builder missing)

- [ ] **Step 3: Write minimal implementation**

First ensure `e_gaba` (sub-threshold reversal) is resolved BEFORE the slow branch. Near line 110, confirm/add:

```python
    E_A = p.E_gaba if e_gaba is None else e_gaba    # sub-threshold shunt reversal, reused for a-shunt
```

Replace the M4 inline branch (~line 316-317):

```python
    if slow is not None:
        Vtmp = I_net + (V - I_net) * decay_V
```

with:

```python
    if slow is not None:
        if slow.uses_shunt():                                  # M4-3A conductance a-shunt (form A)
            g = np.zeros_like(V)
            g[:slow.nE] = slow.shunt_g_at_E()                  # E-only; I cells g=0 -> parity
            V_inf = (I_net + g * E_A) / (1.0 + g)              # a NEVER divides signed net (D2): reversal-clamped
            Vtmp = V_inf + (V - V_inf) * decay_V ** (1.0 + g)
        else:
            Vtmp = I_net + (V - I_net) * decay_V               # literal pre-change path -> byte parity
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_kick_probe_shunt.py -q`
Expected: PASS

- [ ] **Step 5: Run FULL engine parity/regression suite BEFORE re-bless**

Run: `python -m pytest tests/ -q -k "snn or kick or engine or slow_field or m4"`
Expected: PASS — the off-path (`uses_shunt()==False`) must remain bit-identical. If any existing byte-parity/golden test fails, STOP and fix — do not re-bless over a real parity break.

- [ ] **Step 6: Re-bless the engine fingerprint (P1-2: exact path + command)**

Only `kick_probe.py` changed and it IS in the guard list, so re-snapshot the tracked set (other engine files' hashes are unchanged; `slow_field.py` is not guarded, so its edits in Task 4 needed no re-bless):

```bash
python -c "import json; from src.sef_hfo_snn_engine_guard import record_versions; \
p='results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json'; \
old=json.load(open(p)); json.dump(record_versions(list(old.keys())), open(p,'w'), indent=2); \
print('re-blessed', p)"
```

Then confirm the re-bless gate + off-path parity are green:

Run: `python -m pytest tests/test_a1c_feedback.py::test_T8_engine_blessed tests/test_kick_probe_shunt.py -q`
Expected: PASS — the JSON's `src/snn_engine/kick_probe.py` sha now matches the edited file; off-path parity intact. (If T8 still fails, the JSON did not update — verify you wrote the real path above, not a bare `engine_versions.json`.)

- [ ] **Step 7: Commit**

```bash
git add src/snn_engine/kick_probe.py results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json tests/test_kick_probe_shunt.py
git commit -m "feat(m4-3a): conductance a-shunt in M4 membrane update (form A) + re-bless"
```

---

### Task 6: `retrigger_verdict` 拆 4 态（reignite_bounded / attenuated / runaway / not_run）

**Files:**
- Modify: `src/sef_hfo_m4_termination.py`（`retrigger_verdict` 约 line 100-118；调用点）
- Test: `tests/test_m4_termination_early_late.py`

**Interfaces:**
- Consumes: 无。
- Produces: `retrigger_verdict(...)` 返回值集合从 `{pass,fail,not_run}` 变为 `{reignite_bounded, attenuated, runaway, not_run}`。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_m4_termination_early_late.py
import numpy as np
import pytest
from src.sef_hfo_m4_termination import retrigger_verdict


def _post(peak, tail):
    a = np.full(400, 0.0); a[50:100] = peak; a[-80:] = tail
    return a


def test_not_run_when_not_terminate_clean():
    # class != terminate_clean returns not_run before baseline/ref_peak are ever needed
    assert retrigger_verdict("persist", _post(0.6, 0.0)) == "not_run"


def test_attenuated_when_kick_fizzles():
    # P1-5: real signature is (class, post_af, baseline=, ref_peak=); amp = ref_peak - baseline internally
    v = retrigger_verdict("terminate_clean", _post(0.1, 0.0), baseline=0.0, ref_peak=1.0)
    assert v == "attenuated"                              # peak 0.1 < baseline + 0.5*amp -> refractory


def test_runaway_when_tail_stays_high():
    v = retrigger_verdict("terminate_clean", _post(0.9, 0.9), baseline=0.0, ref_peak=1.0)
    assert v == "runaway"                                # tail 0.9 >= baseline + 0.8*amp -> stayed high


def test_reignite_bounded_when_fires_then_falls():
    v = retrigger_verdict("terminate_clean", _post(0.9, 0.0), baseline=0.0, ref_peak=1.0)
    assert v == "reignite_bounded"                       # rose then fell -> bounded re-event
```

> **P1-5 (verified signature):** `retrigger_verdict(termination_class, post_af=None, baseline=None, ref_peak=None, *, reig_frac=0.5, runaway_tail_frac=0.8, tail_ms=None, bin_ms=5.0)`. It computes `amp = ref_peak - baseline` **internally** and raises `ValueError` if `terminate_clean` with `baseline`/`ref_peak` None. Pass `ref_peak`, NOT `amp`. Only the three return strings change (below); the signature and math are unchanged.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_m4_termination_early_late.py -q`
Expected: FAIL — current code returns `"pass"`/`"fail"`, not the 4 new labels.

- [ ] **Step 3: Write minimal implementation**

In `retrigger_verdict` change ONLY the three return strings (verified lines 114, 117, 118); leave the signature, the `amp = ref_peak - baseline` line, and every condition untouched:

```python
    if float(post_af.max()) < baseline + reig_frac * amp:  # fizzle: kick did not re-ignite an event
        return "attenuated"                                # was "fail" -> postictal refractory (P1-5/Task 6)
    tail_bins = int(round(tail_ms / bin_ms)) if tail_ms else max(1, post_af.size // 5)
    if float(post_af[-tail_bins:].mean()) >= baseline + runaway_tail_frac * amp:  # stayed high -> runaway
        return "runaway"                                   # was "fail"
    return "reignite_bounded"                              # was "pass" -> re-ignited AND came back down
```

Update the docstring at line 100-104 (`pass / fail / not_run`) to the 4-label set `reignite_bounded / attenuated / runaway / not_run`.

Update the pass-2-runaway short-circuit in `run_cell_with_retrigger` (line 154-156): `out["retrigger_probe"] = "runaway"` (was `"fail"`).

Update the two runner call sites in `scripts/run_m4_dynamic_qi.py` (grep `retrigger_probe`, ~line 256 and 308): the M4-2 `--p1-sweep` `go = ... AND retrigger_probe == "pass"` becomes `== "reignite_bounded"` (keeps M4-2's single-probe semantics working; M4-3A uses the early+late composite in Task 8).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_m4_termination_early_late.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Regression — existing M4-2 termination tests**

Run: `python -m pytest tests/ -q -k "termination or m4_2 or retrigger"`
Expected: PASS (update any test asserting the old `"pass"`/`"fail"` strings to the new labels — these are the same-semantics rename).

- [ ] **Step 6: Commit**

```bash
git add src/sef_hfo_m4_termination.py scripts/run_m4_dynamic_qi.py tests/test_m4_termination_early_late.py
git commit -m "feat(m4-3a): split retrigger_verdict into reignite_bounded/attenuated/runaway"
```

---

### Task 7: `run_cell_with_retrigger` 加 early 窗（P1-5：additive，offset 来自 classify）

**Files:**
- Modify: `src/sef_hfo_m4_termination.py`（`run_cell_with_retrigger` line 121-165）
- Test: `tests/test_m4_termination_early_late.py`（追加）

**Interfaces:**
- Consumes: `classify_termination`、`retrigger_verdict`（Task 6）。
- Produces: `run_cell_with_retrigger(run_fn, bin_ms, *, recovery_ms=5000.0, recovery_factor=3.0, reprobe_boost=1.0, probe_window_ms=3000.0, baseline_af=None, early_offset_ms=None)` → 现有 dict（`termination_class, retrigger_probe, offset_ms, t_kick2_ms, peak, baseline_af, runaway_ms, pass2_runaway_ms`）**+ 新增 `retrigger_early`、`t_early_ms`（仅当 `early_offset_ms` 给出）**。`early_offset_ms=None`（默认）= **M4-2 逐行不变**：`retrigger_probe` 仍是唯一的 late 探针（`offset + recovery_factor*recovery_ms`，本就 ≈offset+10s）。

> **P1-5 铁律：** `offset_ms`/`peak`/`baseline` **一律**来自 `classify_termination(af1, ...)` 的 `info`，**绝不从 `run_fn` 结果读**（`run_fn` 只回 `af` / 可选 `runaway_ms` / `baseline_af`，**没有** `offset_ms`）。`retrigger_verdict` 传 `ref_peak=info["peak"]`、`baseline=base`（Task 6 的真实签名）。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_m4_termination_early_late.py  (append)
from src.sef_hfo_m4_termination import run_cell_with_retrigger


def _clean_trace(n):
    """Deterministic terminate_clean trace of length n: quiet, plateau [100,400)@0.9, then quiet.
    Absolute positions -> _clean_trace(a)[:a] == _clean_trace(b)[:a], so pre-probe identity holds."""
    a = np.zeros(int(n)); a[100:400] = 0.9
    return a


def test_early_refractory_late_recovers():
    calls = {"n": 0}

    def run_fn(t_kick2, boost, min_T):
        if t_kick2 is None:
            return {"af": _clean_trace(1200), "baseline_af": 0.0}    # NOTE: no 'offset_ms' key
        calls["n"] += 1
        af = _clean_trace(int(min_T) + 10)                          # pass-2 must run to >= min_T
        i = int(round(t_kick2))
        if t_kick2 > 5000.0:                                        # late -> bounded re-event
            af[i:i + 60] = 0.9; af[i + 60:i + 200] = 0.0
        else:                                                       # early -> fizzle
            af[i:i + 60] = 0.05
        return {"af": af, "baseline_af": 0.0}

    out = run_cell_with_retrigger(run_fn, bin_ms=1.0, recovery_ms=5000.0, recovery_factor=2.0,
                                  probe_window_ms=3000.0, baseline_af=0.0, early_offset_ms=750.0)
    assert out["termination_class"] == "terminate_clean"
    assert out["offset_ms"] == pytest.approx(400.0)                 # P1-5: from classify, not run_fn
    assert out["retrigger_early"] == "attenuated"
    assert out["retrigger_probe"] == "reignite_bounded"            # late == the existing single probe
    assert calls["n"] == 2                                          # one early + one late pass-2


def test_early_offset_none_is_m4_2_parity():
    def run_fn(t_kick2, boost, min_T):
        if t_kick2 is None:
            return {"af": _clean_trace(1200), "baseline_af": 0.0}
        af = _clean_trace(int(min_T) + 10); i = int(round(t_kick2))
        af[i:i + 60] = 0.9; af[i + 60:i + 200] = 0.0
        return {"af": af, "baseline_af": 0.0}
    out = run_cell_with_retrigger(run_fn, bin_ms=1.0, recovery_ms=5000.0, recovery_factor=2.0,
                                  probe_window_ms=3000.0, baseline_af=0.0)   # early_offset_ms default None
    assert out["retrigger_probe"] == "reignite_bounded"
    assert "retrigger_early" not in out                            # no early probe when not requested


def test_pre_probe_identity_enforced():
    def bad_run_fn(t_kick2, boost, min_T):
        if t_kick2 is None:
            return {"af": _clean_trace(1200), "baseline_af": 0.0}
        af = _clean_trace(int(min_T) + 10); af[0] += 1.0           # corrupt the pre-probe prefix
        return {"af": af, "baseline_af": 0.0}
    with pytest.raises(RuntimeError, match="pre-probe identity"):
        run_cell_with_retrigger(bad_run_fn, bin_ms=1.0, baseline_af=0.0, early_offset_ms=750.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_m4_termination_early_late.py -q -k "early or parity or pre_probe"`
Expected: FAIL — `early_offset_ms` kwarg unknown / no `retrigger_early` key.

- [ ] **Step 3: Write minimal implementation**

Lift the existing single-probe body (verified lines 146-164) into a helper, then call it for the late probe (always) and the early probe (only when requested). The late path stays byte-identical to today:

```python
def _probe_once(run_fn, af1, bin_ms, t2, reprobe_boost, probe_window_ms, base, ref_peak):
    """One same-seed pass-2 at t2. Enforces pre-probe identity + fail-closed window; returns
    (verdict, pass2_runaway_ms). Mirrors the verified lines 146-164 exactly."""
    min_T = t2 + probe_window_ms
    res2 = run_fn(t2, reprobe_boost, min_T)
    af2 = np.asarray(res2["af"], float)
    i2 = int(round(t2 / bin_ms))
    i_ov = min(af1.size, af2.size, i2)
    if not np.array_equal(af1[:i_ov], af2[:i_ov]):
        raise RuntimeError("pre-probe identity violated: pass-2 prefix != pass-1 for t < t_kick2")
    pass2_runaway = res2.get("runaway_ms")
    if pass2_runaway is not None:                       # re-kick drove pass-2 into runaway (early-stop truncates)
        return "runaway", pass2_runaway
    probe_bins = int(round(probe_window_ms / bin_ms))
    if i2 + probe_bins > af2.size:                      # FAIL-CLOSED: non-runaway short pass-2 = contract violation
        raise RuntimeError(
            f"retrigger probe window did not fit: need pass-2 length >= {i2 + probe_bins} bins "
            f"(t_kick2={t2:.0f}ms + probe {probe_window_ms:.0f}ms), got {af2.size}.")
    return retrigger_verdict("terminate_clean", post_af=af2[i2:i2 + probe_bins],
                             baseline=base, ref_peak=ref_peak), pass2_runaway


def run_cell_with_retrigger(run_fn, bin_ms, *, recovery_ms=5000.0, recovery_factor=3.0,
                            reprobe_boost=1.0, probe_window_ms=3000.0, baseline_af=None,
                            early_offset_ms=None):
    res1 = run_fn(None, 0.0, None)
    af1 = np.asarray(res1["af"], float)
    base = baseline_af if baseline_af is not None else res1.get("baseline_af")
    cls, info = classify_termination(af1, bin_ms, baseline=base, runaway_ms=res1.get("runaway_ms"))
    out = dict(termination_class=cls, offset_ms=info["offset_ms"], t_kick2_ms=None,
               peak=info["peak"], baseline_af=info["baseline"], runaway_ms=info["runaway_ms"],
               retrigger_probe="not_run", pass2_runaway_ms=None)
    if cls != "terminate_clean":
        return out
    t2 = info["offset_ms"] + recovery_factor * recovery_ms          # late probe (= existing single probe)
    out["retrigger_probe"], out["pass2_runaway_ms"] = _probe_once(
        run_fn, af1, bin_ms, t2, reprobe_boost, probe_window_ms, base, info["peak"])
    out["t_kick2_ms"] = t2
    if early_offset_ms is not None:                                 # M4-3A: add an early refractory probe
        t_early = info["offset_ms"] + early_offset_ms
        out["retrigger_early"], _ = _probe_once(
            run_fn, af1, bin_ms, t_early, reprobe_boost, probe_window_ms, base, info["peak"])
        out["t_early_ms"] = t_early
    return out
```

> `_probe_once` is the verified lines 146-164 lifted verbatim (same pre-probe guard, same pass-2-runaway short-circuit → now `"runaway"` per Task 6, same fail-closed raise). The late call reproduces today's behavior exactly; the early call is purely additive. **Do not read `offset_ms` from `run_fn` (P1-5).**

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_m4_termination_early_late.py -q`
Expected: PASS (7 passed — 4 verdict + 3 harness)

- [ ] **Step 5: Regression — M4-2 single-probe path unchanged**

Run: `python -m pytest tests/ -q -k "termination or m4_2 or retrigger"`
Expected: PASS — `early_offset_ms=None` default keeps every existing M4-2 caller byte-identical.

- [ ] **Step 6: Commit**

```bash
git add src/sef_hfo_m4_termination.py tests/test_m4_termination_early_late.py
git commit -m "feat(m4-3a): additive early retrigger window (late=existing probe; offset from classify)"
```

---

### Task 8: runner `--m43a-sweep`（(α_A×τ_n) 网格 + per-seed Arm0 + go early/late + 诊断记录）

**Files:**
- Modify: `scripts/run_m4_dynamic_qi.py`（新 `_run_m43a_sweep`/`_m43a_cell_worker`，仿 `_run_p1_sweep` 约 line 318-355、`_p1_cell_worker` 约 289-315；cfg build 仿 line 184；CLI 约 392-397 + dispatch 432-438）
- Test: `tests/test_m43a_runner_smoke.py`

**Interfaces:**
- Consumes: Task 4 config（`use_A,alpha_A,eta_A,tau_n,...`）；Task 7 `run_cell_with_retrigger`（early/late）。
- Produces: CLI `--m43a-sweep`；输出 `<out>/m43a_sweep_summary.json`（rows 恒含 `label,alpha_A,tau_n,termination_class,retrigger_probe,go,D_A_mean,D_A_p95,D_A_max,seed`；`retrigger_early`/`t_early_ms` 仅 `terminate_clean` cell 有，Task 7 additive）+ `<out>/m43a_sweep_traces.npz`。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_m43a_runner_smoke.py
import json, subprocess, sys, os, glob


def test_m43a_sweep_tiny_runs_and_writes_schema(tmp_path):
    """1 seed, 2x2 grid, very short T -> pipeline runs, JSON schema present, Arm0 row exists."""
    out = tmp_path / "m43a_smoke"
    cmd = [sys.executable, "scripts/run_m4_dynamic_qi.py", "--m43a-sweep", "--confirm-run",
           "--seed", "1", "--T", "800",
           "--m43a-alpha-grid", "0,4", "--m43a-tau-grid", "2000,20000",
           "--m43a-workers", "2", "--out", str(out)]
    subprocess.run(cmd, check=True, cwd=os.getcwd())
    summ = json.load(open(glob.glob(str(out / "*summary.json"))[0]))
    rows = summ["rows"]
    labels = {r["label"] for r in rows}
    assert any(l.startswith("m43a_arm0") for l in labels)          # per-seed Arm0 present
    for r in rows:
        for key in ("label", "alpha_A", "tau_n", "termination_class",
                    "retrigger_probe", "go", "seed"):     # retrigger_early is conditional (terminate_clean only)
            assert key in r
    assert summ["provenance"]["seed"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_m43a_runner_smoke.py -q`
Expected: FAIL — `--m43a-sweep` not a recognized argument.

- [ ] **Step 3: Write minimal implementation**

Add module-level M4-3A constants near the existing `SIGMA_Q, TAU_Q, ...` (~line 50):

```python
# M4-3A load->shunt defaults (spec Global Constraints); tau_n SLOW (> TAU_Q)
M43A_TAU_N, M43A_N50, M43A_HILL_H, M43A_A_MAX = 20000.0, 0.4, 2.0, 1.0
M43A_RHO_N, M43A_K_N, M43A_ETA_A, M43A_UN0 = 0.1, 1.0, 0.0, 0.0
M43A_GA_MAX = 20.0
```

Add a cfg builder that mirrors the op-point at line 184 but turns on the shunt:

```python
def _m43a_cfg(k_q, alpha_G, alpha_A, tau_n, *, eta_A=M43A_ETA_A, sigma_n=SIGMA_Q):
    return SpatialSlowFieldConfig(
        use_qI=True, use_gK=False, k_q=k_q, k_K=0.0, sigma_q=SIGMA_Q, sigma_K=0.5,
        q_min=Q_MIN, q_init=1.0, tau_q=TAU_Q, tau_a=TAU_A,
        use_SG=True, alpha_G=alpha_G, beta_SG=0.0,           # match M4 op-point (grep exact SG args)
        use_A=True, k_n=M43A_K_N, tau_n=tau_n, rho_n=M43A_RHO_N, n_base=0.0,
        n50=M43A_N50, hill_h=M43A_HILL_H, a_max=M43A_A_MAX,
        alpha_A=alpha_A, eta_A=eta_A, sigma_n=sigma_n, u_n0=M43A_UN0, g_A_max=M43A_GA_MAX,
        r0_psi=0.0, r50_psi=R50_PSI, n_psi=N_PSI, p_pool=P_POOL,
        tau_mu=TAU_MU, tau_S=TAU_S, S_max=S_MAX)
```

Add worker + driver mirroring `_p1_cell_worker` / `_run_p1_sweep`. Cell tuple = `(label, k_q, alpha_G, alpha_A, tau_n, base_T, rf, rb)`. Arm0 = `alpha_A=0` (a off, `uses_shunt()` False → M4-1 bounded baseline). Each cell:

```python
def _m43a_cell_worker(cell):
    label, k_q, alpha_G, alpha_A, tau_n, base_T, rf, rb = cell
    try:
        S = _S["S"]
        cfg = _m43a_cfg(k_q, alpha_G, alpha_A, tau_n)
        def run_fn(t2, boost, min_T):
            return run_arm(S, cfg, t_kick2=t2, KICK_BOOST2=boost, base_T=base_T,
                           min_T=min_T, dump_shunt_trace=True)   # dump a/n traces (Task 4 traces)
        verdict = run_cell_with_retrigger(run_fn, BIN_MS, recovery_ms=max(tau_n, TAU_Q),
                                          recovery_factor=rf, reprobe_boost=rb,
                                          baseline_af=None, early_offset_ms=750.0)
        go = (verdict["termination_class"] == "terminate_clean"
              and verdict.get("retrigger_early") == "attenuated"
              and verdict["retrigger_probe"] == "reignite_bounded")   # retrigger_probe = late window (Task 7)
        row = {"label": label, "alpha_A": alpha_A, "tau_n": tau_n, "go": bool(go), **verdict}
        # diagnostics: divisive denominator D_A = 1 + alpha_A * a (per spec 8)
        a_tr = verdict.get("_a_trace")
        if a_tr is not None:
            D_A = 1.0 + alpha_A * a_tr
            row.update(D_A_mean=float(D_A.mean()), D_A_p95=float(np.percentile(D_A, 95)),
                       D_A_max=float(D_A.max()))
        return row
    except Exception as e:
        return {"label": label, "termination_class": "ERROR", "go": False, "error": repr(e)}
```

Driver builds `cells = [("m43a_arm0", k_q, alpha_G, 0.0, tau_grid[0], base_T, rf, rb)] + [(f"m43a_a{a:g}_tau{int(t)}", k_q, alpha_G, a, t, base_T, rf, rb) for a in alpha_grid for t in tau_grid]`, runs the same `mp.Pool(min(workers, len(cells)))` COW pattern as `_run_p1_sweep`, stamps `seed`, writes `m43a_sweep_summary.json` with `provenance=dict(seed, subject, montage, git_sha, argv, T)` and `m43a_sweep_traces.npz`.

Add CLI + dispatch (mirror `--p1-sweep`):

```python
    ap.add_argument("--m43a-sweep", action="store_true")
    ap.add_argument("--m43a-alpha-grid", default="2,4,8")
    ap.add_argument("--m43a-tau-grid", default="5000,20000,40000")
    ap.add_argument("--m43a-workers", type=int, default=5)   # OOM-safe
    ...
    if a.m43a_sweep:
        _run_m43a_sweep(a); return
```

> Grep the real `run_arm` signature (~line 192) and thread a `dump_shunt_trace` flag mirroring the existing `trace_xdep`/`dump_ee_std_trace` plumbing so `run_arm` returns `_a_trace`/`_n_trace` (from `slow.trace_a_mean`). Reuse the existing `--confirm-run` guard, thread-pinning (`OMP/OPENBLAS/MKL_NUM_THREADS=1`), and provenance helpers verbatim.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_m43a_runner_smoke.py -q`
Expected: PASS (1 passed) — the tiny sweep completes and the schema is present.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_m4_dynamic_qi.py tests/test_m43a_runner_smoke.py
git commit -m "feat(m4-3a): --m43a-sweep (alpha_A x tau_n, per-seed Arm0, early/late go)"
```

---

### Task 9: runner `--m43a-ablation`（shunt-only / subtractive-only / hybrid）

**Files:**
- Modify: `scripts/run_m4_dynamic_qi.py`（加 `_run_m43a_ablation`）
- Test: `tests/test_m43a_runner_smoke.py`（追加）

**Interfaces:**
- Consumes: `_m43a_cfg`（Task 8）。
- Produces: CLI `--m43a-ablation`（在给定 `(α_A, τ_n)` 点跑 3 变体）；输出 `<out>/m43a_ablation.json`（3 行 + matched-η_A 计算记录）。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_m43a_runner_smoke.py  (append)
def test_m43a_ablation_three_variants(tmp_path):
    out = tmp_path / "abl"
    cmd = [sys.executable, "scripts/run_m4_dynamic_qi.py", "--m43a-ablation", "--confirm-run",
           "--seed", "1", "--T", "800", "--m43a-abl-alpha", "6", "--m43a-abl-tau", "20000",
           "--out", str(out)]
    subprocess.run(cmd, check=True, cwd=os.getcwd())
    abl = json.load(open(glob.glob(str(out / "*ablation.json"))[0]))
    variants = {r["variant"] for r in abl["rows"]}
    assert variants == {"shunt_only", "subtractive_only", "hybrid"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_m43a_runner_smoke.py -q -k ablation`
Expected: FAIL — `--m43a-ablation` unknown.

- [ ] **Step 3: Write minimal implementation**

```python
def _run_m43a_ablation(a):
    _require_confirm(a)                                     # reuse existing --confirm-run guard
    S = _build_substrate(a)                                  # reuse existing substrate builder (grep)
    k_q, alpha_G, alpha_A, tau_n = a.p1_kq, a.p1_alpha_g, a.m43a_abl_alpha, a.m43a_abl_tau
    # 1) shunt-only: alpha_A>0, eta_A=0
    r_sh = _m43a_eval(S, k_q, alpha_G, alpha_A, tau_n, eta_A=0.0, base_T=a.T)
    # matched-subtractive: set eta_A so mean subtractive removal ~ shunt-only mean drive removal.
    # shunt removes on average (1 - 1/D_A_mean) of depolarizing drive; match via logged I_dep_mean.
    eta_matched = float(alpha_A * r_sh.get("a_mean", 0.0) * r_sh.get("I_dep_mean", 1.0)) \
        if r_sh.get("a_mean") else 0.0
    # 2) subtractive-only: alpha_A=0, eta_A=matched
    r_sub = _m43a_eval(S, k_q, alpha_G, 0.0, tau_n, eta_A=eta_matched, base_T=a.T)
    # 3) hybrid: alpha_A>0, eta_A=matched
    r_hy = _m43a_eval(S, k_q, alpha_G, alpha_A, tau_n, eta_A=eta_matched, base_T=a.T)
    rows = [dict(variant="shunt_only", **r_sh),
            dict(variant="subtractive_only", eta_A=eta_matched, **r_sub),
            dict(variant="hybrid", eta_A=eta_matched, **r_hy)]
    _write_json(a.out, "m43a_ablation.json",
                dict(rows=rows, alpha_A=alpha_A, tau_n=tau_n, eta_matched=eta_matched,
                     provenance=_provenance(a)))
```

`_m43a_eval(...)` wraps one `run_cell_with_retrigger` on a single `(alpha_A, tau_n, eta_A)` cfg and returns its verdict dict + `a_mean`/`I_dep_mean` diagnostics. Add CLI `--m43a-ablation`, `--m43a-abl-alpha`, `--m43a-abl-tau` and dispatch, mirroring Task 8.

> The matched-η_A recipe is a first-order match on mean depolarizing-drive removal; log `eta_matched` and the two mean-removal quantities so the reviewer can verify the match. Prediction (spec D11): shunt_only + hybrid → terminate_clean window; subtractive_only → persist/suppress/fragment.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_m43a_runner_smoke.py -q -k ablation`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_m4_dynamic_qi.py tests/test_m43a_runner_smoke.py
git commit -m "feat(m4-3a): --m43a-ablation (shunt-only vs matched-subtractive vs hybrid)"
```

---

### Task 10: 全链路冒烟 + off-parity 门 + campaign runbook

**Files:**
- Create: `tests/test_m43a_end_to_end.py`
- Create: `docs/superpowers/plans/m43a_runbook.md`（execution 步骤，非科学结论）

**Interfaces:**
- Consumes: 全部前置 task。
- Produces: 一个证明"pipeline 跑通 + use_A off 逐字节 parity + JSON schema"的集成测试；一份 runbook（P0 → P1 discovery → 40s acceptance → ablation 的确切命令，供用户执行真实 campaign）。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_m43a_end_to_end.py
import numpy as np


def test_use_A_off_byte_parity_end_to_end():
    """A full tiny M4 run with use_A=False must be bit-identical to the same run built
    without the n/a fields at all (spec parity red-line)."""
    res_off = _tiny_m4_run(use_A=False)         # n/a config present, gate off
    res_ref = _tiny_m4_run_reference()          # pre-M4-3A config path
    assert np.array_equal(res_off["spk"], res_ref["spk"])
    assert np.allclose(res_off["trace_qI_mean"], res_ref["trace_qI_mean"])


def test_shunt_on_changes_dynamics():
    res_on = _tiny_m4_run(use_A=True, alpha_A=8.0, k_n=1.0)
    res_off = _tiny_m4_run(use_A=False)
    assert not np.array_equal(res_on["spk"], res_off["spk"])   # shunt actually couples
```

> `_tiny_m4_run` / `_tiny_m4_run_reference` build the smallest real E1146-style M4 config (grep `run_arm`); reference = the same net stepped through the pre-change path (e.g. a config where the n/a fields are absent or use_A hard-off). If a true "reference" pre-change run is impractical, assert parity against a golden raster captured on `main`/pre-Task-4 and stored under `tests/fixtures/`.

- [ ] **Step 2: Run test to verify it fails, then implement fixtures to pass**

Run: `python -m pytest tests/test_m43a_end_to_end.py -q`
Expected: FAIL first (fixtures missing), then PASS after adding the tiny-run helpers.

- [ ] **Step 3: Write the runbook (execution, not results)**

```markdown
# docs/superpowers/plans/m43a_runbook.md
# M4-3A execution runbook (run order per spec 10; results need user eyeball)

## P0a offline calibration — PROXY (no network; global-rate drive)
python scripts/run_m43a_p0_calibration.py --dt 0.1 \
  --regime quiet=results/topic4_m4_dynamic_p1_sweep/p1_sweep_traces.npz:p1_arm0__rate:0:20000 \
  --regime bounded_ictal=results/topic4_m4_dynamic_p1_sweep/p1_sweep_traces.npz:p1_arm0__rate \
  --regime isolated_ied=<short-IED-run>.npz:rate --event isolated_ied=<idx> \
  --regime post_offset=<suppress-cell>.npz:rate \
  --out results/topic4_m43a_p0a/
# gate (P1-3): sensor_free_pass requires delta_a_ied>0 AND >=2*sigma AND >=0.5%*a_max AND R_A>=5
#             AND interictal_block_pass. soft_gate_fail (delta<=0) is a HARD fail (R_A=inf does NOT pass).
# P0a is PROXY only -> sensor_free_pass stays False until P0b provides field-derived u_n + a_block.

## P0b field-derived u_n lock (P1-4): replay Arm0 dumping the real drive
python scripts/run_m4_dynamic_qi.py --m43a-sweep --confirm-run --seed 1 --T 20000 \
  --m43a-alpha-grid 0 --m43a-tau-grid 20000 --out results/topic4_m43a_p0b_arm0/
# the Arm0 cell dumps trace_un_mean (field-derived u_n = K_n * rE). Set u_n0 = its long-run mean;
# re-run P0a with --u-n0 <that> and --a-block <from a small IED-kick probe> to certify sensor_free_pass.

## P1 discovery sweep (T=15000), per seed in {1,3,4}
for s in 1 3 4; do
  python scripts/run_m4_dynamic_qi.py --m43a-sweep --confirm-run --seed $s --T 15000 \
    --m43a-alpha-grid 2,4,8 --m43a-tau-grid 5000,20000,40000 --m43a-workers 5 \
    --out results/topic4_m43a_p1_seed$s/
done
# primary denominator = seeds whose m43a_arm0 classifies bounded-persist (compute from output).

## 40s acceptance for candidate go / boundary cells (T=40000)
python scripts/run_m4_dynamic_qi.py --m43a-sweep --confirm-run --seed <s> --T 40000 \
  --m43a-alpha-grid <candidate a> --m43a-tau-grid <candidate tau> --m43a-workers 3 \
  --out results/topic4_m43a_accept_seed<s>/
# go requires: terminate_clean at 40s + no post-offset rebound + early attenuated + late reignite_bounded.

## Mechanism ablation at the best candidate point
python scripts/run_m4_dynamic_qi.py --m43a-ablation --confirm-run --seed <s> --T 40000 \
  --m43a-abl-alpha <a> --m43a-abl-tau <tau> --out results/topic4_m43a_ablation/
# expect: shunt_only+hybrid clean; subtractive_only persist/suppress/fragment.
```

- [ ] **Step 4: Run the full M4 + M4-3A test suite green**

Run: `python -m pytest tests/ -q -k "m4 or slow_field or kick or snn or termination or m43a"`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add tests/test_m43a_end_to_end.py docs/superpowers/plans/m43a_runbook.md
git commit -m "test(m4-3a): end-to-end off-parity gate + execution runbook"
```

---

## Self-Review

**1. Spec coverage (rev2 sections → task):**
- §3.1 baseline-centered `n→a` ODE + clamps → Task 1 (+ §6.1 metrics Task 2). ✓
- §3.2 form (A) 电导 shunt（不除 signed net、reversal E_A）→ Task 5; `-η_A a` 减法 → Task 4. ✓
- §3.3 `σ_n=σ_q=1.5` 默认 + `{0.75,1.5,2.25}` 次级 → Task 4 config default + Task 8 grid arg. ✓
- §4.0 P0a proxy 离线标定 + sensor-free 硬门 booleans → Task 3. ✓
- §4.1 (α_A×τ_n) 网格、op-point、per-seed Arm0 → Task 8. ✓
- §4.2 40s acceptance + early/late retrigger + ablation → Task 7 (early/late), Task 8 (T=40000 via `--T`), Task 9 (ablation), Task 10 (runbook). ✓
- §6.1 `Δa_IED`/`R_A`/`⟨a⟩<a_block` 数值门 → Task 2 + Task 3（`gate` booleans；`Δa_IED≤0`=硬 fail，inf 不过门）. ✓
- §7 go = terminate_clean AND early attenuated AND late reignite_bounded → Task 6 (labels), Task 8 (composite). ✓
- §8 parity 红线 / re-bless / n/a trace 复用 `trace_qI_mean` 模式 / D_A 记录 → Tasks 4,5,8,10. ✓
- Global: `use_gK=False` op-point, `tau_a` 命名防撞, `use_<var>+零旋钮` 门控 → Task 4/8 constraints. ✓
- **Out of scope (correctly):** M4-3B graph-kernel `K_graph` + 三负控 (spec §9), M4-3C 闭环 → separate plan. ✓

**2. Placeholder scan:** 无 TBD/TODO；每个 code step 有完整代码。若干 "grep to confirm signature/line" 是**故意**的（AGENTS.md：names drift；引擎构造签名 must be read from source, not fabricated）——已在 Step-3 note 里点名要确认的具体符号，非占位。

**3. Type consistency:** `LoadShuntParams` 字段（Task 1）↔ `_load_shunt_params()`（Task 4）一致；`uses_shunt() = use_A and k_n!=0 and alpha_A!=0`（Task 4，P1-1）↔ `kick_probe` 分支门 + parity gate（Task 5）一致；`shunt_g_at_E()`（Task 4）↔ `kick_probe` 调用（Task 5）一致；`retrigger_verdict` 4-label 集（Task 6）↔ `run_cell_with_retrigger` 产 `retrigger_probe`(late)/`retrigger_early`（Task 7）↔ runner go 判据 `retrigger_probe=="reignite_bounded" and retrigger_early=="attenuated"`（Task 8）一致；`retrigger_verdict` 用 `ref_peak`（非 `amp`）跨 Task 6/7 一致；cell 8-tuple（Task 8）内部自洽。

**4. Review round P1 fixes（本轮）：** P1-1 `uses_shunt()` 加 `k_n!=0`（parity gate 修矛盾，Task 4）；P1-2 `engine_versions.json` 真实路径 `results/topic4_sef_hfo/snn_heterogeneity/` + 具体 re-bless 命令 + T8 gate（Task 5，且确认 `slow_field.py` 不在 guard→Task 4 免 re-bless）；P1-3 P0 `gate` booleans + `Δa_IED≤0` 硬 fail（inf 不过门，Task 3）；P1-4 P0a=proxy 标注 + `trace_un_mean` field-derived `u_n` + P0b 锁参步骤（Task 3/4/10）；P1-5 retrigger `offset`/`peak`/`baseline` 一律来自 `classify_termination` info（非 `run_fn`）+ `ref_peak` 真实签名 + early 窗 additive（`early_offset_ms=None`→M4-2 逐行不变，Task 6/7）。

**已知需实现者从源码确认（非本 plan 能捏造的）：** `SpatialSlowField.__init__` 精确构造签名、`run_arm` 精确签名 + trace 回传方式、`retrigger_verdict` 精确 kwargs、`engine_versions.json` re-bless 命令、runner 里 `_build_substrate`/`_provenance`/`_require_confirm`/`--confirm-run` 的确切名字。每个都在对应 Step 里点名 grep。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-09-sef-hfo-m4-3a-continuous-shunting-impl.md`. Two execution options:

**1. Subagent-Driven (recommended)** — 每个 task 派一个 fresh subagent，task 间 review，快速迭代。**特别适合本 plan**：Task 5（引擎 + re-bless）与 Task 8（runner sweep）风险最高，task 间 review 能在 re-bless / 大 sweep 前卡住 parity 破坏。

**2. Inline Execution** — 本 session 内用 executing-plans 批量执行 + checkpoint review。

**Which approach?**
