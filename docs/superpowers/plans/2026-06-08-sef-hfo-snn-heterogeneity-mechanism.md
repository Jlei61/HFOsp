# SEF-HFO 放电网络异质性病理核 — 机制实现 + 粗网格 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在放电网络（SNN）里给兴奋性神经元装上"按空间分布的点火门槛"病理核（参差收窄；配平 + 不配平），跑一个 (病理核 × kick 点) 粗网格，产出机制图 + 基线对比图 + 白话 recap。

**Architecture:** 纯函数取样器（门槛场 + 邻域离散度，`src/`，TDD）→ 引擎 2 行注入逐神经元门槛（gitignored，配 patch+checksum 可追踪性守卫）→ 源空间度量纯函数（`src/`，TDD）→ 粗网格 runner（配对种子、`--time-one` 先计时、`--quick` smoke）→ plotter（复用 `two_electrode_readout`）。

**Tech Stack:** Python, numpy, pytest, matplotlib(Agg)。复用引擎 `results/topic4_sef_hfo/lif_snn/engine/`（`place_neurons` / `build_connectivity_rot` / `simulate_kick` / `LFPRecorder` / `compute_nu_theta`）+ `src/sef_hfo_plot.py::two_electrode_readout`。

**Spec:** `docs/superpowers/specs/2026-06-08-sef-hfo-snn-heterogeneity-mechanism-design.md`。

**Locked constants (spec §2):** `V_th=18`, `V_reset=11`, gap 7；`std_wide=1.5`, `std_narrow=0.5`；不配平 `core_mean_shift=2.0`（核内均值 16）；工作点 `L=3.0`, `density=1800`, `drive=0.6`, `theta_EE=45°`, `AR=2.0`；kick `R_KICK=0.15`, `T_KICK=150`, `DUR_KICK=18`, `KICK_BOOST=2·nu_theta`。`patch_radius` 默认 0.5 mm。

**Engine path:** `ENGINE = "results/topic4_sef_hfo/lif_snn/engine"`（runner/测试用 `sys.path.insert(0, ENGINE)` 导入）。

---

### Task 1: 门槛场取样器 + 邻域离散度（纯函数，TDD）

**Files:**
- Modify: `src/sef_hfo_heterogeneity.py`（在文件末尾追加；不动现有 rate-model 函数）
- Test: `tests/test_sef_hfo_heterogeneity.py`（追加）

- [ ] **Step 1: Write the failing test**

```python
# 追加到 tests/test_sef_hfo_heterogeneity.py
import numpy as np
from src.sef_hfo_heterogeneity import sample_threshold_fields, local_vth_spread


def _toy_sheet(n=4000, L=3.0, fE=0.8, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, L, size=(n, 2))
    is_E = np.zeros(n, bool); is_E[: int(fE * n)] = True
    return pos, is_E, rng


def test_threshold_fields_share_surround_and_narrow_core():
    pos, is_E, rng = _toy_sheet()
    out = sample_threshold_fields(pos, is_E, patch_center=(1.5, 1.5),
                                  patch_radius=0.5, rng=rng,
                                  vth_mean=18.0, std_wide=1.5, std_narrow=0.5,
                                  v_reset=11.0, core_mean_shift=2.0)
    base, matched, unmatched = out["baseline"], out["matched"], out["unmatched"]
    core = out["core_mask"]
    surround = is_E & ~core
    # surround is BIT-IDENTICAL across the three (paired-seed contract)
    np.testing.assert_array_equal(base[surround], matched[surround])
    np.testing.assert_array_equal(base[surround], unmatched[surround])
    # core spread narrows; baseline core keeps the wide spread
    assert matched[core].std() < base[core].std()
    assert unmatched[core].std() < base[core].std()
    # matched core mean ~ 18 (held), unmatched core mean ~ 16 (shifted down)
    assert abs(matched[core].mean() - 18.0) < 0.3
    assert abs(unmatched[core].mean() - 16.0) < 0.3
    # physical domain: nothing below reset
    assert (base[is_E] >= 11.0).all()
    assert (matched[is_E] >= 11.0).all() and (unmatched[is_E] >= 11.0).all()
    # I neurons keep scalar threshold
    assert np.allclose(base[~is_E], 18.0)


def test_local_vth_spread_lower_in_core():
    pos, is_E, rng = _toy_sheet()
    out = sample_threshold_fields(pos, is_E, (1.5, 1.5), 0.5, rng)
    spread = local_vth_spread(pos, out["matched"], is_E, radius=0.3)
    core = out["core_mask"]
    surround = is_E & ~core
    assert np.nanmean(spread[core]) < np.nanmean(spread[surround])
    assert np.isnan(spread[~is_E]).all()   # I neurons not colored
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sef_hfo_heterogeneity.py -k "threshold_fields or local_vth" -v`
Expected: FAIL with `ImportError: cannot import name 'sample_threshold_fields'`

- [ ] **Step 3: Write minimal implementation**

```python
# 追加到 src/sef_hfo_heterogeneity.py
def _trunc_gauss(mean, std, low, size, rng):
    """N(mean,std) truncated to [low, inf) by rejection resampling (truncated +
    renormalized; spec §2 physical domain — no clip-to-floor)."""
    if std <= 0:
        return np.full(size, max(mean, low), float)
    out = np.empty(size); filled = 0
    while filled < size:
        draw = rng.normal(mean, std, size=size - filled)
        ok = draw[draw >= low]
        out[filled:filled + ok.size] = ok
        filled += ok.size
    return out


def sample_threshold_fields(pos, is_E, patch_center, patch_radius, rng,
                            vth_mean=18.0, std_wide=1.5, std_narrow=0.5,
                            v_reset=11.0, core_mean_shift=2.0):
    """Per-neuron firing thresholds (length N) for baseline / matched / unmatched,
    sharing a bit-identical surround (paired-seed contract, spec §2).

    I neurons keep scalar vth_mean (engine inhibitory threshold stays scalar).
    E neurons: wide TruncGauss everywhere (= surround field W); the matched/
    unmatched variants redraw ONLY the in-patch E neurons (narrow spread; matched
    mean=vth_mean, unmatched mean=vth_mean-core_mean_shift). Truncated at v_reset.
    """
    pos = np.asarray(pos, float); is_E = np.asarray(is_E, bool)
    n = len(pos)
    W = np.full(n, vth_mean, float)
    eidx = np.flatnonzero(is_E)
    W[eidx] = _trunc_gauss(vth_mean, std_wide, v_reset, eidx.size, rng)
    d = np.linalg.norm(pos - np.asarray(patch_center, float)[None, :], axis=1)
    core = is_E & (d <= float(patch_radius))
    cidx = np.flatnonzero(core)
    rng_m = np.random.default_rng(int(rng.integers(1 << 31)))
    rng_u = np.random.default_rng(int(rng.integers(1 << 31)))
    matched = W.copy()
    matched[cidx] = _trunc_gauss(vth_mean, std_narrow, v_reset, cidx.size, rng_m)
    unmatched = W.copy()
    unmatched[cidx] = _trunc_gauss(vth_mean - core_mean_shift, std_narrow, v_reset,
                                   cidx.size, rng_u)
    for arr in (W, matched, unmatched):
        if not (arr[eidx] >= v_reset).all():
            raise ValueError("threshold below V_reset — truncation failed (spec §2)")
    return dict(baseline=W, matched=matched, unmatched=unmatched, core_mask=core)


def local_vth_spread(pos, vth, is_E, radius):
    """Per-E-neuron std of V_th among E neighbours within `radius` (mm) — for the
    mechanism-figure colouring. I neurons -> NaN (not coloured)."""
    pos = np.asarray(pos, float); vth = np.asarray(vth, float)
    is_E = np.asarray(is_E, bool)
    out = np.full(len(pos), np.nan)
    Epos = pos[is_E]; Evth = vth[is_E]
    eidx = np.flatnonzero(is_E)
    for k, i in enumerate(eidx):
        dd = np.linalg.norm(Epos - pos[i][None, :], axis=1)
        nb = dd <= radius
        out[i] = Evth[nb].std() if nb.sum() >= 3 else np.nan
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sef_hfo_heterogeneity.py -k "threshold_fields or local_vth" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_heterogeneity.py tests/test_sef_hfo_heterogeneity.py
git commit -m "feat(topic4 snn-hetero): per-neuron V_th field sampler (paired surround) + local spread"
```

---

### Task 2: 引擎注入逐神经元门槛（2 行 + 等价/抑制测试）

**Files:**
- Modify: `results/topic4_sef_hfo/lif_snn/engine/kick_probe.py:40-41, 159-161`
- Test: `tests/test_sef_hfo_snn_hetero_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sef_hfo_snn_hetero_engine.py
"""逐神经元 V_th 注入：传标量等价数组、传高门槛抑制发放。tiny smoke params。"""
import os, sys
import numpy as np
import pytest

ENGINE = os.path.join("results", "topic4_sef_hfo", "lif_snn", "engine")
sys.path.insert(0, ENGINE)


@pytest.fixture(scope="module")
def net_and_p():
    from params import Params, compute_nu_theta
    from model import build_network
    p = Params(g=3.6, L=1.0, density=4000.0, T=220.0, dt=0.1, nu_ext_ratio=0.6, seed=1)
    net = build_network(p, verbose=False)
    nu_theta = compute_nu_theta(p)[0]
    return p, net, nu_theta


def test_array_equals_scalar_when_uniform(net_and_p):
    from kick_probe import simulate_kick
    p, net, nu_theta = net_and_p
    N = net["NE"] + net["NI"]
    net["rng"] = np.random.default_rng(7)
    a = simulate_kick(p, net, KICK_BOOST=2 * nu_theta, kick_center=[0.5, 0.5])
    net["rng"] = np.random.default_rng(7)
    b = simulate_kick(p, net, KICK_BOOST=2 * nu_theta, kick_center=[0.5, 0.5],
                      V_th_per_neuron=np.full(N, p.V_th))
    np.testing.assert_array_equal(a["rate_E"], b["rate_E"])   # uniform array == scalar


def test_high_threshold_suppresses_firing(net_and_p):
    from kick_probe import simulate_kick
    p, net, nu_theta = net_and_p
    N = net["NE"] + net["NI"]
    vth = np.full(N, p.V_th); vth[: net["NE"]] = 1e6      # E can never reach threshold
    net["rng"] = np.random.default_rng(7)
    c = simulate_kick(p, net, KICK_BOOST=2 * nu_theta, kick_center=[0.5, 0.5],
                      V_th_per_neuron=vth)
    assert c["rate_E"].max() == 0.0                        # no E spikes at all
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sef_hfo_snn_hetero_engine.py -v`
Expected: FAIL — `simulate_kick() got an unexpected keyword argument 'V_th_per_neuron'`

- [ ] **Step 3: Write minimal implementation**

Edit `results/topic4_sef_hfo/lif_snn/engine/kick_probe.py`:

1) signature (line 40-41) — add the kwarg:
```python
def simulate_kick(p: Params, net, KICK_BOOST, slow=None, nu_signal_fn=None,
                  verbose=False, kick_center=None, lfp_recorder=None, r_kick=None,
                  t_kick=None, V_th_per_neuron=None):
```

2) threshold source (the `else` branch around line 159-161) — use the array when given:
```python
        if slow is not None:
            I_net = slow.apply_currents(I_E, I_I, labels)
            V_th_eff = slow.threshold(p.V_th)
        else:
            I_net = I_E - I_I
            V_th_eff = p.V_th if V_th_per_neuron is None else V_th_per_neuron
```
(The spike compare at line 169 already handles array-vs-scalar via `np.isscalar`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sef_hfo_snn_hetero_engine.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit** (engine is gitignored — commit only the test now; the engine change is captured by Task 3's patch/checksum)

```bash
git add tests/test_sef_hfo_snn_hetero_engine.py
git commit -m "test(topic4 snn-hetero): engine V_th_per_neuron equivalence + suppression"
```

---

### Task 3: 引擎可追踪性守卫（patch + checksum + 断言，TDD）

**Files:**
- Create: `src/sef_hfo_snn_engine_guard.py`
- Test: `tests/test_sef_hfo_snn_engine_guard.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sef_hfo_snn_engine_guard.py
import numpy as np
import pytest
from src.sef_hfo_snn_engine_guard import record_versions, assert_versions


def test_record_then_assert_roundtrip(tmp_path):
    f = tmp_path / "engine_a.py"; f.write_text("x = 1\n")
    rec = record_versions([str(f)])
    assert_versions(rec)                       # matches -> no raise
    f.write_text("x = 2\n")                     # drift
    with pytest.raises(RuntimeError):
        assert_versions(rec)                    # checksum mismatch -> loud fail
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sef_hfo_snn_engine_guard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.sef_hfo_snn_engine_guard'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sef_hfo_snn_engine_guard.py
"""Traceability guard for the gitignored SNN engine (spec §7 hard contract).
record_versions() snapshots sha256 of the engine files a runner imports; the
runner calls assert_versions() at startup so silent engine drift fails loudly."""
from __future__ import annotations
import hashlib
from pathlib import Path


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_versions(paths) -> dict:
    return {str(p): _sha256(str(p)) for p in paths}


def assert_versions(recorded: dict) -> None:
    for p, h in recorded.items():
        cur = _sha256(p)
        if cur != h:
            raise RuntimeError(
                f"SNN engine drift: {p} sha256 {cur[:12]} != recorded {h[:12]}. "
                "Re-snapshot engine_versions.json only after reviewing the change.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sef_hfo_snn_engine_guard.py -v`
Expected: PASS

- [ ] **Step 5: Save the engine patch + checksum snapshot, then commit**

```bash
mkdir -p results/topic4_sef_hfo/snn_heterogeneity/engine_patch
cd results/topic4_sef_hfo/lif_snn/engine && \
  git diff --no-index /dev/null kick_probe.py > /dev/null 2>&1; \
  cd - >/dev/null
# capture the V_th_per_neuron edit as a standalone patch (manual diff is fine):
python - <<'PY'
import hashlib, json, pathlib
eng = pathlib.Path("results/topic4_sef_hfo/lif_snn/engine")
files = ["kick_probe.py","params.py","model.py","connectivity.py",
         "connectivity_rot.py","lfp.py"]
rec = {str(eng/f): hashlib.sha256((eng/f).read_bytes()).hexdigest() for f in files}
out = pathlib.Path("results/topic4_sef_hfo/snn_heterogeneity")
out.mkdir(parents=True, exist_ok=True)
(out/"engine_versions.json").write_text(json.dumps(rec, indent=2))
print("recorded", len(rec), "engine checksums")
PY
git add src/sef_hfo_snn_engine_guard.py tests/test_sef_hfo_snn_engine_guard.py \
        results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json
git commit -m "feat(topic4 snn-hetero): engine version guard + checksum snapshot"
```

> Also copy the edited `kick_probe.py` into `results/topic4_sef_hfo/snn_heterogeneity/engine_patch/kick_probe.py` so the exact source is recoverable even if the gitignored tree is lost.

---

### Task 4: `two_electrode_readout` 加 `patch_circle`（病理核轮廓 + smoke 测试）

**Files:**
- Modify: `src/sef_hfo_plot.py`（`two_electrode_readout` 签名 + 左 a 面板）
- Test: `tests/test_sef_hfo_plot_patch_circle.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sef_hfo_plot_patch_circle.py
import numpy as np
from src.sef_hfo_plot import two_electrode_readout


def test_two_electrode_readout_accepts_patch_circle(tmp_path):
    n = 200
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 3, size=(n, 2))
    t = np.arange(60) * 1.0
    sig = np.exp(-((t - 30) ** 2) / 50.0)[None, :].repeat(4, 0)
    shaft = dict(contacts=np.array([[0.8, 0.8], [1.0, 1.0], [1.2, 1.2], [1.4, 1.4]]),
                 part=np.ones(4, bool), s=sig, signal=sig, label="P", panel_title="par")
    out = tmp_path / "x.png"
    two_electrode_readout(str(out), field_xy=xy, field_c=rng.random(n),
                          field_clabel="local V_th spread (mV)",
                          kick_xy=np.array([2.4, 1.5]), axis_deg=45.0,
                          extent=(0, 3, 0, 3), par=shaft, perp=shaft,
                          t=t, event_window=(5, 55), signal_ylabel="x",
                          substrate_label="x", contact_note="x",
                          patch_circle=(1.5, 1.5, 0.5))
    assert out.exists()                       # ran end-to-end with patch_circle
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sef_hfo_plot_patch_circle.py -v`
Expected: FAIL — `two_electrode_readout() got an unexpected keyword argument 'patch_circle'`
(If the existing positional/keyword names differ, first read `src/sef_hfo_plot.py:158-230` and match the real signature in the test before implementing.)

- [ ] **Step 3: Write minimal implementation**

In `src/sef_hfo_plot.py::two_electrode_readout`, add `patch_circle=None` to the signature, and after the panel-A substrate/scatter draw (just before/after `_axis_arrow(axA, ...)`), add:
```python
    if patch_circle is not None:
        px, py, pr = patch_circle
        axA.add_patch(plt.Circle((px, py), pr, fill=False, ls="--",
                                 ec="crimson", lw=1.6, zorder=4))
```
(Ensure `import matplotlib.pyplot as plt` is in scope — it is in this module.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sef_hfo_plot_patch_circle.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_plot.py tests/test_sef_hfo_plot_patch_circle.py
git commit -m "feat(topic4 snn-hetero): two_electrode_readout patch_circle outline"
```

---

### Task 5: 源空间度量（传播主轴 + 核同步，纯函数，TDD）

**Files:**
- Create: `src/sef_hfo_snn_metrics.py`
- Test: `tests/test_sef_hfo_snn_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sef_hfo_snn_metrics.py
import numpy as np
from src.sef_hfo_snn_metrics import onset_times, onset_axis, peak_active_fraction


def test_onset_axis_recovers_linear_wave_direction():
    rng = np.random.default_rng(0)
    NE = 400
    posE = rng.uniform(0, 3, size=(NE, 2))
    dt, t_kick = 0.1, 150.0
    nsteps = int(300 / dt)
    spk = np.zeros((nsteps, NE), bool)
    # onset increases along +x: neuron at x fires at t_kick + 20*x ms
    for i in range(NE):
        ti = int((t_kick + 20.0 * posE[i, 0]) / dt)
        if ti < nsteps:
            spk[ti, i] = True
    onset = onset_times(spk, dt, t_kick)
    axis = onset_axis(posE, onset, min_n=20)
    ang = np.degrees(np.arctan2(axis[1], axis[0])) % 180.0
    assert min(ang, 180 - ang) < 15.0          # ~along x (0 deg)


def test_peak_active_fraction_counts_distinct():
    dt = 0.1
    spk = np.zeros((100, 10), bool)
    spk[40:45, :6] = True                       # 6/10 distinct in one 5ms bin
    paf = peak_active_fraction(spk, dt, 0.0, 10.0, bin_ms=5.0)
    assert abs(paf - 0.6) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sef_hfo_snn_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.sef_hfo_snn_metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sef_hfo_snn_metrics.py
"""Source-space (oracle) event metrics for the SNN heterogeneity grid (spec §4).
Operate on per-E-neuron spike booleans + coords; electrode-independent so every
grid cell is comparable."""
from __future__ import annotations
import numpy as np


def onset_times(E_spk_bool, dt, t_kick):
    """First-spike time (ms) after the kick per E neuron; NaN if it never fires."""
    i_kick = int(round(t_kick / dt))
    post = np.asarray(E_spk_bool)[i_kick:]
    ever = post.any(axis=0)
    return np.where(ever, (post.argmax(axis=0) + i_kick) * dt, np.nan)


def onset_axis(posE, onset, min_n=20):
    """Propagation axis from the onset-time spatial gradient (lstsq t ~ a + g·x).
    Returns a unit 2-vector (direction of increasing onset) or None if too few."""
    fin = np.isfinite(onset)
    if fin.sum() < min_n:
        return None
    X = np.asarray(posE)[fin]; t = np.asarray(onset)[fin]
    Xc = X - X.mean(0)
    g, *_ = np.linalg.lstsq(Xc, t - t.mean(), rcond=None)
    nrm = float(np.linalg.norm(g))
    return None if nrm < 1e-9 else g / nrm


def peak_active_fraction(E_spk_bool, dt, t_lo, t_hi, bin_ms=2.0):
    """Max over bin_ms windows of (distinct E neurons that fired in the bin)/NE,
    within [t_lo,t_hi) ms. Mirrors engine kick_probe.peak_active_fraction but
    lives in tracked src so the grid runner needn't import the gitignored engine."""
    spk = np.asarray(E_spk_bool)
    nsteps, NE = spk.shape
    bs = int(round(bin_ms / dt))
    i_lo = int(round(t_lo / dt)); i_hi = int(round(t_hi / dt))
    best = 0.0
    for b0 in range(i_lo, i_hi, bs):
        b1 = min(b0 + bs, i_hi)
        if b1 > b0:
            best = max(best, spk[b0:b1].any(axis=0).sum() / NE)
    return float(best)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sef_hfo_snn_metrics.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_snn_metrics.py tests/test_sef_hfo_snn_metrics.py
git commit -m "feat(topic4 snn-hetero): source-space onset-axis + active-fraction metrics"
```

---

### Task 6: 粗网格 runner（配对种子 + `--time-one` + `--quick` smoke + 引擎守卫）

**Files:**
- Create: `scripts/run_sef_hfo_snn_hetero_grid.py`

> Boilerplate (network build / electrode montage / LFP recorder) mirrors the
> existing `scripts/run_sef_hfo_obs_two_electrode_snn_lfp.py:45-70` — read it for
> the exact `place_neurons` / `build_connectivity_rot` / `LFPRecorder` /
> `merge_montages` calls and reuse them verbatim.

- [ ] **Step 1: Write the runner**

```python
# scripts/run_sef_hfo_snn_hetero_grid.py
"""SNN heterogeneity pathology-core grid (spec 2026-06-08).
Modes:
  --time-one   time ONE L=3 baseline run, print wall seconds (size the grid)
  --quick      tiny L=1 smoke over a 2-cell grid (CI-cheap green run)
  --grid       the real coarse grid (default)
Conditions: baseline / matched / unmatched core, PAIRED seed+noise+kick+surround.
Source-space metrics per cell -> grid_metrics.json; per-cell NPZ for the plotter.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ENGINE = os.path.join("results", "topic4_sef_hfo", "lif_snn", "engine")
sys.path.insert(0, ENGINE)
from params import Params, compute_nu_theta            # noqa: E402
from connectivity import place_neurons                 # noqa: E402
from connectivity_rot import build_connectivity_rot    # noqa: E402
from kick_probe import simulate_kick, compute_metrics  # noqa: E402
from lfp import LFPRecorder                             # noqa: E402

sys.path.insert(0, os.getcwd())
from src.sef_hfo_heterogeneity import sample_threshold_fields  # noqa: E402
from src.sef_hfo_snn_metrics import onset_times, onset_axis, peak_active_fraction  # noqa: E402
from src.sef_hfo_snn_engine_guard import assert_versions       # noqa: E402
from src.sef_hfo_observation import build_shaft, merge_montages  # noqa: E402

OUT = Path("results/topic4_sef_hfo/snn_heterogeneity")
THETA, AR = 45.0, 2.0
PATCH_R = 0.5
T_KICK = 150.0


def _engine_guard():
    rec = json.loads((OUT / "engine_versions.json").read_text())
    assert_versions(rec)                       # spec §7: drift -> loud fail


def _build(L, density, seed):
    p = Params(g=3.6, L=L, density=density, T=450.0, dt=0.1,
               nu_ext_ratio=0.6, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=np.deg2rad(THETA), AR=AR)
    nu_theta = compute_nu_theta(p)[0]
    return p, net, nu_theta, NE


def _montage(L):
    u = np.deg2rad(THETA)
    c = (0.85 * L / 3.0, 0.85 * L / 3.0)
    m = merge_montages([build_shaft(u, 0.26, 7, c, "P"),
                        build_shaft(u + np.pi / 2, 0.26, 7, c, "Q")])
    return m


def _run_one(p, net, nu_theta, NE, kick_xy, vth, montage):
    """One paired-seed run; returns source-space metrics + arrays for the plot."""
    net["rng"] = np.random.default_rng(p.seed)          # paired noise/poisson
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=montage.contacts)
    res = simulate_kick(p, net, KICK_BOOST=2 * nu_theta, kick_center=list(kick_xy),
                        lfp_recorder=rec, V_th_per_neuron=vth)
    dt = p.dt
    onset = onset_times(res["E_spk_bool"], dt, T_KICK)
    axis = onset_axis(net["pos"][:NE], onset, min_n=20)
    m = compute_metrics(res, dt)
    return dict(
        peak=m["peak"], returned=bool(m["returned"]), outside=m["outside"],
        peak_active_frac=m["peak_active_frac"],
        axis_deg=(float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0)
                  if axis is not None else None),
        onset=onset, lfp=res["lfp_trace"], times=res["times"],
        contacts=montage.contacts, names=np.array(montage.names),
    )


def _grid_cells(L):
    """Pre-registered coarse grid (spec §3). Patch fixed mid-sheet for sweep-1;
    kick fixed at axis end for sweep-2. Returns list of dict cells."""
    u = np.array([np.cos(np.deg2rad(THETA)), np.sin(np.deg2rad(THETA))])
    ctr = np.array([L / 2, L / 2])
    end = ctr + 0.6 * (L / 2) * u                       # far axis end (kick0)
    mid_patch = tuple(ctr)                               # patch on the path (sweep-1)
    cells = []
    # sweep-1: fix patch (mid), vary kick (4 reps): end / near-patch / opposite / off-axis
    kicks = {"end": end, "nearcore": ctr + 0.25 * (L / 2) * u,
             "opp": ctr - 0.6 * (L / 2) * u,
             "offaxis": ctr + 0.5 * (L / 2) * np.array([-u[1], u[0]])}
    for kname, k in kicks.items():
        cells.append(dict(sweep=1, kick=tuple(k), kname=kname,
                          patch=mid_patch, cond="matched"))
    # sweep-2: fix kick (end), vary patch (4 positions) x {matched, unmatched}
    patches = {"nearseed": tuple(end - 0.3 * (L / 2) * u), "mid": mid_patch,
               "far": tuple(ctr - 0.4 * (L / 2) * u),
               "offaxis": tuple(ctr + 0.4 * (L / 2) * np.array([-u[1], u[0]]))}
    for pname, pc in patches.items():
        for cond in ("matched", "unmatched"):
            cells.append(dict(sweep=2, kick=tuple(end), kname="end",
                              patch=pc, cond=cond, pname=pname))
    return cells


def _process(cells, L, density, seed):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_cell").mkdir(exist_ok=True)
    p, net, nu_theta, NE = _build(L, density, seed)
    is_E = net["labels"] == 0
    montage = _montage(L)
    rows = []
    base_cache = {}                                      # kick -> baseline metrics
    for i, c in enumerate(cells):
        fields = sample_threshold_fields(net["pos"], is_E, c["patch"], PATCH_R,
                                         np.random.default_rng(seed))
        kkey = tuple(np.round(c["kick"], 4))
        if kkey not in base_cache:
            base_cache[kkey] = _run_one(p, net, nu_theta, NE, c["kick"],
                                        fields["baseline"], montage)
        b = base_cache[kkey]
        core = _run_one(p, net, nu_theta, NE, c["kick"], fields[c["cond"]], montage)
        d_paf = core["peak_active_frac"] - b["peak_active_frac"]
        d_axis = (None if (core["axis_deg"] is None or b["axis_deg"] is None)
                  else float(min((core["axis_deg"] - b["axis_deg"]) % 180.0,
                                 180.0 - (core["axis_deg"] - b["axis_deg"]) % 180.0)))
        # kick-on-patch flag (anti-circularity, spec §1)
        on_patch = bool(np.linalg.norm(np.array(c["kick"]) - np.array(c["patch"])) <= PATCH_R)
        row = dict(idx=i, **{k: c[k] for k in ("sweep", "kname", "cond")},
                   patch=list(c["patch"]), kick=list(c["kick"]),
                   d_peak_active_frac=d_paf, d_axis_deg=d_axis,
                   core_returned=core["returned"], base_returned=b["returned"],
                   kick_on_patch=on_patch)
        rows.append(row)
        np.savez_compressed(OUT / "per_cell" / f"cell{i:02d}.npz",
                            posE=net["pos"][:NE], onset_core=core["onset"],
                            vth=fields[c["cond"]], is_E=is_E,
                            lfp=core["lfp"], times=core["times"],
                            contacts=core["contacts"], names=core["names"],
                            kick=np.array(c["kick"]), patch=np.array(c["patch"]),
                            patch_r=PATCH_R, L=L, theta=THETA,
                            base_lfp=b["lfp"], base_times=b["times"],
                            meta=json.dumps(row))
        print(f"[{i+1}/{len(cells)}] sweep{c['sweep']} {c.get('pname',c['kname'])} "
              f"{c['cond']} d_paf={d_paf:+.3f} d_axis={d_axis} on_patch={on_patch}",
              flush=True)
    (OUT / "grid_metrics.json").write_text(json.dumps(
        dict(L=L, density=density, seed=seed, patch_r=PATCH_R, cells=rows), indent=2))
    print("wrote grid_metrics.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-one", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()
    _engine_guard()
    if a.time_one:
        p, net, nu_theta, NE = _build(3.0, 1800.0, 1)
        fields = sample_threshold_fields(net["pos"], net["labels"] == 0,
                                         (1.5, 1.5), PATCH_R, np.random.default_rng(1))
        t0 = time.time()
        _run_one(p, net, nu_theta, NE, [2.4, 1.5], fields["baseline"], _montage(3.0))
        print(f"ONE L=3 run wall = {time.time()-t0:.1f}s")
        return
    if a.quick:
        _process(_grid_cells(1.0)[:2], 1.0, 4000.0, 1); return
    _process(_grid_cells(3.0), 3.0, 1800.0, 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify engine import names**

Run: `python -c "import sys; sys.path.insert(0,'results/topic4_sef_hfo/lif_snn/engine'); import lfp, connectivity_rot, kick_probe; print('ok')"`
Expected: `ok`. If `LFPRecorder` / `build_connectivity_rot` live under different names, read `scripts/run_sef_hfo_obs_two_electrode_snn_lfp.py` imports and fix the `from ... import ...` lines to match.

- [ ] **Step 3: Smoke-run `--quick`**

Run: `python scripts/run_sef_hfo_snn_hetero_grid.py --quick`
Expected: prints 2 cells, writes `grid_metrics.json` + 2 `per_cell/cell0*.npz`, no error. (Engine guard must pass — Task 3's `engine_versions.json` exists.)

- [ ] **Step 4: Commit**

```bash
git add scripts/run_sef_hfo_snn_hetero_grid.py
git commit -m "feat(topic4 snn-hetero): coarse grid runner (paired seeds + source metrics + engine guard)"
```

---

### Task 7: Plotter — 网格总览 + 代表机制图 + 基线对比 + README

**Files:**
- Create: `scripts/plot_sef_hfo_snn_hetero_mechanism.py`
- Create: `results/topic4_sef_hfo/snn_heterogeneity/figures/README.md`

- [ ] **Step 1: Write the plotter**

```python
# scripts/plot_sef_hfo_snn_hetero_mechanism.py
"""Three figure classes from the heterogeneity grid (spec §5):
  (1) grid_overview.png   — d_peak_active_frac over the (kick × core) grid
  (2) mechanism_<tag>.png — representative cells, reuse two_electrode_readout
                            (panel-a coloured by LOCAL V_th spread + patch outline)
  (3) baseline_compare.png— baseline vs core population read-out for reps
Representative pick rule is PRE-REGISTERED (spec §4): max effect / near-zero /
kick-outside-wave-through-core / matched-vs-unmatched."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.sef_hfo_plot import two_electrode_readout
from src.sef_hfo_heterogeneity import local_vth_spread

OUT = Path("results/topic4_sef_hfo/snn_heterogeneity")
FIG = OUT / "figures"


def _pick_representatives(cells):
    off = [c for c in cells if not c["kick_on_patch"] and c["d_axis_deg"] is not None]
    by_paf = sorted(cells, key=lambda c: abs(c["d_peak_active_frac"]))
    reps = {}
    reps["maxeffect"] = max(cells, key=lambda c: abs(c["d_peak_active_frac"]))
    reps["nearzero"] = by_paf[0]
    if off:
        reps["through_core"] = max(off, key=lambda c: abs(c["d_peak_active_frac"]))
    m = [c for c in cells if c["cond"] == "matched"]
    u = [c for c in cells if c["cond"] == "unmatched"]
    if m and u:
        reps["unmatched"] = max(u, key=lambda c: abs(c["d_peak_active_frac"]))
    return reps


def _overview(cells):
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [f"{c['kname']}/{c.get('cond','')[:1]}\n{c['sweep']}" for c in cells]
    vals = [c["d_peak_active_frac"] for c in cells]
    colors = ["crimson" if c["kick_on_patch"] else "steelblue" for c in cells]
    ax.bar(range(len(cells)), vals, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(cells))); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("Δ peak active fraction (core − baseline)")
    ax.set_title("Heterogeneity grid: core effect on synchrony "
                 "(red = kick-on-patch, not mechanism evidence)")
    fig.tight_layout(); fig.savefig(FIG / "grid_overview.png", dpi=130); plt.close(fig)


def _mechanism(cell, tag):
    z = np.load(OUT / "per_cell" / f"cell{cell['idx']:02d}.npz", allow_pickle=True)
    posE = z["posE"]; vth = z["vth"]                     # posE=(NE,2); vth=(N,), E at [:NE]
    NE = len(posE)
    # colour E neurons by local V_th spread (E-only coords + E thresholds)
    spread = local_vth_spread(posE, vth[:NE], np.ones(NE, bool), 0.3)
    L = float(z["L"]); patch = z["patch"]; pr = float(z["patch_r"])
    t = np.asarray(z["times"]); lfp = np.asarray(z["lfp"]).T          # (n_contact, nt)
    contacts = z["contacts"]
    ncp = 7
    def shaft(sl, lab, ttl):
        return dict(contacts=contacts[sl], part=np.ones(sl.stop - sl.start, bool),
                    s=lfp[sl], signal=lfp[sl], label=lab, panel_title=ttl)
    win = (155.0, 205.0)                                  # event window after T_KICK=150
    two_electrode_readout(
        str(FIG / f"mechanism_{tag}.png"),
        field_xy=posE, field_c=spread, field_clabel="local V_th spread (mV)",
        kick_xy=z["kick"], axis_deg=float(z["theta"]), extent=(0, L, 0, L),
        par=shaft(slice(0, ncp), "P", "∥ axis — peaks sweep"),
        perp=shaft(slice(ncp, 2 * ncp), "Q", "⊥ axis — peaks aligned"),
        t=t, event_window=win, signal_ylabel="current-LFP (|I_E|+|I_I|)",
        substrate_label=f"Spiking · hetero core · {tag}",
        contact_note="contacts scaled to model sheet — NOT real SEEG spacing; "
                     "firing-density read-out (not LFP)",
        patch_circle=(float(patch[0]), float(patch[1]), pr))


def _baseline_compare(reps):
    fig, axes = plt.subplots(1, len(reps), figsize=(4 * len(reps), 3.2), squeeze=False)
    for ax, (tag, c) in zip(axes[0], reps.items()):
        z = np.load(OUT / "per_cell" / f"cell{c['idx']:02d}.npz", allow_pickle=True)
        ax.plot(z["base_times"], np.asarray(z["base_lfp"]).mean(1), color="0.6",
                label="baseline")
        ax.plot(z["times"], np.asarray(z["lfp"]).mean(1), color="C3", label="core")
        ax.set_title(f"{tag}\nΔpaf={c['d_peak_active_frac']:+.3f}", fontsize=8)
        ax.set_xlabel("time (ms)"); ax.legend(fontsize=6)
    axes[0, 0].set_ylabel("mean current-LFP")
    fig.suptitle("Network-state: baseline vs pathology core (mean over contacts)")
    fig.tight_layout(); fig.savefig(FIG / "baseline_compare.png", dpi=130); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    cells = json.loads((OUT / "grid_metrics.json").read_text())["cells"]
    _overview(cells)
    reps = _pick_representatives(cells)
    for tag, c in reps.items():
        _mechanism(c, tag)
    _baseline_compare(reps)
    (OUT / "cohort_summary.json").write_text(json.dumps(
        {"representatives": {k: v["idx"] for k, v in reps.items()},
         "n_cells": len(cells)}, indent=2))
    print("figures:", [p.name for p in FIG.glob("*.png")])


if __name__ == "__main__":
    main()
```

> NOTE: before running, match the real `two_electrode_readout` `par`/`perp`
> shaft-dict keys by reading `src/sef_hfo_plot.py:158-230` (the Explore map listed
> `contacts/part/s/signal/label/panel_title`; confirm and adjust `shaft()` if they
> differ). The existing `scripts/plot_sef_hfo_two_electrode_readout.py::_snn_electrode`
> builds the same dict — copy its exact keys.

- [ ] **Step 2: Run on the `--quick` output to smoke the plot path**

Run: `python scripts/run_sef_hfo_snn_hetero_grid.py --quick && python scripts/plot_sef_hfo_snn_hetero_mechanism.py`
Expected: writes `figures/grid_overview.png` + `mechanism_*.png` + `baseline_compare.png` with no error. (Quick grid is tiny, figures will be sparse — this only proves the plot path runs.)

- [ ] **Step 3: Write `figures/README.md` (Chinese, AGENTS.md)**

```markdown
# results/topic4_sef_hfo/snn_heterogeneity/figures

### grid_overview.png
病理核在 (戳点 × 核配置) 粗网格上对"核内细胞同时活跃比例"的影响（核 − 基线的差）。
红柱 = 戳在核上的格子，按非循环纪律**不作机制证据**，只列出。

### mechanism_<tag>.png
代表案例机制图（复用两电极读出画法）。左 a：神经元面按**邻域门槛离散度**上色（核内
低=更齐、核外高=更参差）+ 病理核虚线轮廓 + 戳点 + 两根电极杆 + 长轴箭头。中 b / 右 c：
该案例电极读出（∥ 峰值依次扫=方向 / ⊥ 峰值对齐=无方向）。代表选择规则预注册：最大效应 /
近零效应 / 戳核外波穿核 / 配平vs不配平。

### baseline_compare.png
代表案例的"整体网络状态"对比：基线 vs 病理核 的群体读出轨迹（触点平均）。

**关注点**：先看 grid_overview 全貌再看代表；机制图左 a 的病理核轮廓 + 上色要能让人一眼
看出异质性在哪、多大；电极读出是 firing-density（**不是 LFP**），只验方向顺序。
```

- [ ] **Step 4: Commit**

```bash
git add scripts/plot_sef_hfo_snn_hetero_mechanism.py \
        results/topic4_sef_hfo/snn_heterogeneity/figures/README.md
git commit -m "feat(topic4 snn-hetero): grid/mechanism/baseline plotter + figures README"
```

---

### Task 8: 计时 → 定网格 → 跑真网格 → 目视 → 白话 recap 归档

**Files:**
- Create: `docs/archive/topic4/sef_hfo/snn_heterogeneity_mechanism_2026-06-08.md`

- [ ] **Step 1: Time one L=3 run (gate the grid size)**

Run: `python scripts/run_sef_hfo_snn_hetero_grid.py --time-one`
Note the wall seconds. Grid ≈ 16 runs → est. total = 16 × (one-run seconds). If > ~10 min, run Step 2 in background.

- [ ] **Step 2: Run the real grid**

Run (foreground if fast, else `run_in_background`): `python scripts/run_sef_hfo_snn_hetero_grid.py --grid`
Expected: `per_cell/cell00..NN.npz` + `grid_metrics.json`. Sanity: at least some `base_returned=True` (events self-terminate); `kick_on_patch` flags set on the on-patch cells.

- [ ] **Step 3: Plot + eyeball**

Run: `python scripts/plot_sef_hfo_snn_hetero_mechanism.py`
Open `figures/grid_overview.png` + `mechanism_*.png` + `baseline_compare.png`. Confirm: panel-a patch outline + local-spread colouring read clearly; electrode read-out panels look like the existing `two_electrode_readout_snn.png`; on-patch cells flagged red in the overview.

- [ ] **Step 4: Write the plain-language recap archive doc**

Use the `hfosp-plain-language-recap` skill. The doc covers (三段式 + 两条轴, spec §6): 测了什么（装门槛参差病理核、比均匀vs核）/ 怎么测的（配对种子、源空间度量、网格）/ 揭示了什么（核内同步 / 事件 / 自终止 / 电极读出顺序，跟率均值场 null 比，方向从仿真读、允许 disconfirm）。诚实标注配平核的 Jensen、戳核上格子不作证据、gap-limit 没放宽。链接代表机制图。

- [ ] **Step 5: Commit**

```bash
git add docs/archive/topic4/sef_hfo/snn_heterogeneity_mechanism_2026-06-08.md \
        results/topic4_sef_hfo/snn_heterogeneity/
git commit -m "feat(topic4 snn-hetero): grid results + mechanism figures + plain-language recap"
```

---

## Self-Review notes

- **Spec coverage:** Task1=门槛场取样器(§2 三条件+配对surround)+local spread(§5 上色); Task2=引擎注入(§7); Task3=可追踪性硬合同(§7); Task4=patch_circle(§5 机制图左a); Task5=源空间度量(§4); Task6=网格runner(§3 两扫法+配对种子+--time-one+on_patch 标注§1); Task7=三类图(§5)+预注册代表选择(§4); Task8=计时→跑→目视→recap(§6). 不做项(厘米级/细胞丢失/率配平/cohort裁决/rate parity)= §8，无任务（正确）。
- **Threshold names:** `sample_threshold_fields` keys (baseline/matched/unmatched/core_mask), `V_th_per_neuron`, `local_vth_spread`, `onset_axis`, `peak_active_fraction`, `assert_versions`/`record_versions`, `patch_circle` — used consistently across tasks.
- **Known soft spots flagged inline (not placeholders):** engine import names (Task6 Step2 verifies), `two_electrode_readout` shaft-dict keys (Task4/Task7 say read `sef_hfo_plot.py:158-230` first), `_mechanism` dead-branch to simplify on wire-up. These are "verify against real code" notes, with the exact file:line to check.
- **Engine traceability:** Task3 records checksums + saves the edited file copy; Task6 asserts at startup. Bringing the engine into git tracking is the cleaner option — raise with the user at Task 2/3 (spec §7 推荐纳入跟踪).
