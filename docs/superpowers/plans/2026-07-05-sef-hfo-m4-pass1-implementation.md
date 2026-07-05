# M4 divisive shared inhibitory pool `S_G` — Pass-1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the M4 divisive shared inhibitory pool `S_G` to the SNN engine — a delayed global pool that *divides* recurrent-E gain (not a subtractive scalar) — up to (but NOT including) the scientific `q_core × alpha_G` phase-plane run.

**Architecture:** Keep the combined AMPA current `I_E` accumulation in `simulate_kick` **exactly as is** (byte-identical); additionally track a **recurrent-only** current `I_E_rec` (a second accumulator fed only by delay-ring arrivals). The divisive effect is applied inside `SpatialSlowField.apply_currents` as a *subtraction* of the removed recurrent current `ΔI_rec = I_E_rec · α_G S_G/(1+α_G S_G)` on E cells. When `α_G S_G = 0` this term is exactly 0, so parity is exact — superior to a literal two-way `s_E` split (which breaks FP associativity). The pool state (`μ_G`, `S_G`) and the recruitment sensor (`Ψ_G → A_G`) live in `SpatialSlowField`, reusing its fast rate field `rE_fast`, all gated on `use_SG` (OFF by default → byte-identical to today).

**Tech Stack:** Python, NumPy, pytest. SNN engine at `src/snn_engine/` (imported in tests via `sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))`).

## Global Constraints

- **OFF-by-default byte-parity (承重).** `use_SG=False` (default) MUST leave engine output byte-identical to today: no allocation, no RNG draw, no float touch on the default path. All M4 code sits inside `if ...use_SG:` branches. Verified by the existing spike-output fixtures (`test_a1c_feedback.py` T9-style `spk_sha`) + the m3a-v2.2 parity tests.
- **Exact parity at `α_G S_G = 0` even with `use_SG=True`.** Because the combined `I_E` accumulation is never touched and the divisive term is a subtraction of an exactly-zero quantity when `α_G S_G = 0`.
- **Re-bless after editing `kick_probe.py`.** `test_a1c_feedback.py::test_T8_engine_blessed` sha256-checks `src/snn_engine/kick_probe.py` against `results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json`. After editing, update that sha256 (re-bless) — but ONLY after the T9 output-parity fixtures still pass.
- **Divide recurrent E only, never feedforward.** `ΔI_rec` uses `I_E_rec` (delay-ring arrivals only), applied to E cells only. External/kick drive (`ext*ext_incr`) is never divided.
- **Naming guard.** The config already has an unrelated `beta_G` (h_G proxy phase-plane term, `slow_field.py:90`). The M4 subtractive-pool coefficient MUST be named `beta_SG` (NOT `beta_G`). Pool activation is `mu_G`; do not reuse `u_G`/`h_G`.
- **Canonical sensor (rev4 spec §3):** `A_G = [⟨Ψ_G(r_E)^p⟩_x]^{1/p}`, `Ψ_G(r)=[r−r0]_+^n / (r50^n+[r−r0]_+^n)`. Default `p ∈ [2,4]` (focal-sensitive), swept as a diagnostic; `p=1` is the area/mean limit. No `M/B/P95/Φ_G`.
- **Spec of record:** `docs/superpowers/specs/2026-07-05-sef-hfo-m4-divisive-shared-inhibition-design.md` (rev4). Re-read §3/§4/§5/§6/§9.1 at each task boundary (CLAUDE.md §5).
- **STOP LINE:** Execute Tasks 1–5 only. Task 6 (the scientific phase-plane) is planned for review but MUST NOT be run until the user reviews plan + implementation.

---

## File Structure

- **Modify** `src/snn_engine/kick_probe.py` — add gated recurrent-only AMPA accumulator (`s_E_rec`, `I_E_rec`); pass `I_E_rec` to `slow.apply_currents`. (Task 1)
- **Modify** `src/snn_engine/slow_field.py` — add stateless sensor helpers `psi_recruit`, `pnorm_pool`; add M4 config fields + pool state + `step` advance + divisive term in `apply_currents`. (Tasks 2–4)
- **Modify** `results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json` — re-bless `kick_probe.py` sha256. (Task 1)
- **Create** `tests/test_m4_shared_inhibition.py` — M4 unit + field + smoke tests. (Tasks 2–5)
- **Create (Task 6, DO NOT RUN)** `src/sef_hfo_m4_phaseplane.py` + `scripts/run_m4_phaseplane.py` — `q_core × alpha_G` sweep + §9.1 go/no-go. Scaffolded for review; execution gated.

---

## Task 1: Recurrent-only AMPA accumulator in `simulate_kick` (gated) + re-bless

**Files:**
- Modify: `src/snn_engine/kick_probe.py` (state init near `:168`; loop near `:234-257`)
- Modify: `results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json`
- Test: `tests/test_m4_shared_inhibition.py`

**Interfaces:**
- Consumes: `simulate_kick(p, net, KICK_BOOST, slow=None, ...)`; the `slow` object may expose `slow.cfg.use_SG` (bool).
- Produces: when `getattr(getattr(slow,"cfg",None),"use_SG",False)` is True, a per-step `I_E_rec` (recurrent-only AMPA current, shape `(N,)`) is computed and passed as the 4th positional-or-keyword arg to `slow.apply_currents(I_E, I_I, labels, I_E_rec)`. Default path passes nothing (byte-identical).

- [ ] **Step 1: Write the failing parity test** (default path unchanged) in `tests/test_m4_shared_inhibition.py`

```python
import os, sys, hashlib
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))
sys.path.insert(0, ROOT)
from kick_probe import simulate_kick  # noqa: E402
from model import build_network, Params  # noqa: E402  (adjust import to the engine's net builder if named differently)


def _tiny_net(T=120.0, seed=0):
    p = Params()
    p.T = T
    net = build_network(p, seed=seed)   # adjust to the actual builder used by other engine tests
    return p, net


def test_default_path_byte_identical_without_SG():
    # simulate_kick with slow=None must be unaffected by the M4 edit.
    p, net = _tiny_net()
    net["rng"] = np.random.default_rng(1)
    a = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9)
    p, net = _tiny_net()
    net["rng"] = np.random.default_rng(1)
    b = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9)
    assert hashlib.sha1(a["E_spk_bool"].tobytes()).hexdigest() == \
           hashlib.sha1(b["E_spk_bool"].tobytes()).hexdigest()
```

> NOTE: replace `build_network`/`Params` import with whatever `tests/test_m3a_v2_2_global_recovery.py` uses (read it first: `_net(...)` helper). The point of this test is a determinism/parity anchor for the default path; keep it aligned with the existing engine test's net builder.

- [ ] **Step 2: Run it — expect PASS now (pre-edit determinism anchor)**

Run: `python -m pytest tests/test_m4_shared_inhibition.py::test_default_path_byte_identical_without_SG -q`
Expected: PASS (this anchors default determinism before the edit).

- [ ] **Step 3: Add the gated recurrent-only accumulator to `simulate_kick`**

In `src/snn_engine/kick_probe.py`, after the state init `s_I = np.zeros(N); I_I = np.zeros(N)` (~`:169`), add:

```python
    # ---- M4: recurrent-only AMPA accumulator (OFF by default -> no alloc/float touch). Tracks the
    # recurrent (delay-ring) component of I_E separately so the shared pool can DIVIDE only recurrent
    # E input. The combined I_E accumulation below is untouched (byte-parity). ----
    track_rec = bool(getattr(getattr(slow, "cfg", None), "use_SG", False))
    if track_rec:
        s_E_rec = np.zeros(N); I_E_rec = np.zeros(N)
```

Inside the loop, right after the combined `s_E += ext * ext_incr` / `I_E = s_E + (I_E - s_E)*decay_IE` block (~`:246-250`), add:

```python
        if track_rec:
            s_E_rec *= decay_sE
            s_E_rec += ring_sE[slot]                     # RECURRENT arrivals only (no ext)
            I_E_rec = s_E_rec + (I_E_rec - s_E_rec) * decay_IE
```

Then change the slow call (`~:257`) from `I_net = slow.apply_currents(I_E, I_I, labels)` to:

```python
            I_net = slow.apply_currents(I_E, I_I, labels,
                                        I_E_rec if track_rec else None)
```

- [ ] **Step 4: Run parity test — expect PASS (default path still byte-identical)**

Run: `python -m pytest tests/test_m4_shared_inhibition.py::test_default_path_byte_identical_without_SG tests/test_a1c_feedback.py -q`
Expected: default-path parity PASS; `test_T9_dynamic_path_unchanged_by_override_param` PASS; **`test_T8_engine_blessed` FAILS** (source hash changed — expected).

- [ ] **Step 5: Re-bless `kick_probe.py`**

```bash
cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-m4-divisive-sg
python - <<'PY'
import hashlib, json
kp = "src/snn_engine/kick_probe.py"
ev = "results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json"
rec = json.load(open(ev))
rec["src/snn_engine/kick_probe.py"] = hashlib.sha256(open(kp,"rb").read()).hexdigest()
json.dump(rec, open(ev,"w"), indent=2)
print("re-blessed:", rec["src/snn_engine/kick_probe.py"])
PY
```

- [ ] **Step 6: Run the full a1c suite — expect all PASS**

Run: `python -m pytest tests/test_a1c_feedback.py -q`
Expected: PASS (T8 now matches, T9 unchanged).

- [ ] **Step 7: Commit**

```bash
git add src/snn_engine/kick_probe.py results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json tests/test_m4_shared_inhibition.py
git commit -m "feat(m4): gated recurrent-only AMPA accumulator in simulate_kick + re-bless"
```

---

## Task 2: Stateless recruitment sensor helpers `psi_recruit`, `pnorm_pool`

**Files:**
- Modify: `src/snn_engine/slow_field.py` (add near the existing stateless helpers `saturation`/`aq_drive`, ~`:131`)
- Test: `tests/test_m4_shared_inhibition.py`

**Interfaces:**
- Produces:
  - `psi_recruit(r, r0, r50, n) -> ndarray` : elementwise Hill recruitment `[r−r0]_+^n / (r50^n + [r−r0]_+^n)`, range `[0,1)`.
  - `pnorm_pool(z, p) -> float` : `[mean(z**p)]**(1/p)` over all elements; `z` assumed in `[0,1]`, `p>=1`.

- [ ] **Step 1: Write the failing tests**

```python
from slow_field import psi_recruit, pnorm_pool  # add to the imports at top of the test file


def test_psi_recruit_hill_shape():
    r = np.array([0.0, 1.0, 2.0, 100.0])
    z = psi_recruit(r, r0=0.0, r50=1.0, n=2.0)
    assert np.isclose(z[0], 0.0)                 # background not recruited
    assert np.isclose(z[1], 0.5)                 # r=r50 -> half recruited
    assert z[3] > 0.99 and np.all(z <= 1.0)      # saturates to 1
    assert np.isclose(psi_recruit(0.4, 0.5, 1.0, 2.0), 0.0)  # sub-threshold clipped


def test_pnorm_pool_mean_and_focal_limits():
    # p=1 == plain mean (soft area); larger p weights toward the peak (focal).
    z = np.zeros(100); z[:1] = 1.0             # one hot cell among 100
    assert np.isclose(pnorm_pool(z, 1.0), 0.01)          # area/mean
    assert pnorm_pool(z, 4.0) > pnorm_pool(z, 1.0)       # focal-sensitive
    assert np.isclose(pnorm_pool(z, 4.0), 0.01 ** 0.25)  # (k/N)^(1/p)
    z2 = np.full(100, 0.5)
    assert np.isclose(pnorm_pool(z2, 3.0), 0.5)          # uniform field: p-invariant
```

- [ ] **Step 2: Run — expect FAIL (ImportError)**

Run: `python -m pytest tests/test_m4_shared_inhibition.py::test_psi_recruit_hill_shape tests/test_m4_shared_inhibition.py::test_pnorm_pool_mean_and_focal_limits -q`
Expected: FAIL (`cannot import name 'psi_recruit'`).

- [ ] **Step 3: Implement the helpers** in `src/snn_engine/slow_field.py` (below `aq_drive`, ~`:143`)

```python
def psi_recruit(r, r0, r50, n):
    """Per-location recruitment nonlinearity Psi_G(r)=[r-r0]_+^n/(r50^n+[r-r0]_+^n) (rev4 spec §3).
    Elementwise; range [0,1). Sub-threshold background (r<=r0) -> 0."""
    x = np.maximum(np.asarray(r, dtype=float) - r0, 0.0)
    xn = x ** n
    return xn / (r50 ** n + xn)


def pnorm_pool(z, p):
    """Pooled drive A_G=[<z^p>_x]^(1/p) over ALL elements (rev4 spec §3). z in [0,1], p>=1.
    p=1 -> soft recruited-area mean; larger p -> focal-sensitive (mean<->max knob)."""
    z = np.asarray(z, dtype=float)
    return float(np.mean(z ** p)) ** (1.0 / p)
```

- [ ] **Step 4: Run — expect PASS**

Run: `python -m pytest tests/test_m4_shared_inhibition.py::test_psi_recruit_hill_shape tests/test_m4_shared_inhibition.py::test_pnorm_pool_mean_and_focal_limits -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/snn_engine/slow_field.py tests/test_m4_shared_inhibition.py
git commit -m "feat(m4): psi_recruit + pnorm_pool recruitment sensor helpers"
```

---

## Task 3: M4 pool state (`mu_G`, `S_G`) + config + `step` advance (gated)

**Files:**
- Modify: `src/snn_engine/slow_field.py` (`SpatialSlowFieldConfig` ~`:44-90`; `__init__` ~`:183-206`; `step` ~`:253-271`)
- Test: `tests/test_m4_shared_inhibition.py`

**Interfaces:**
- Consumes: `psi_recruit`, `pnorm_pool` (Task 2); `firing_rate_field` (existing); `self.rE_fast`.
- Produces: `SpatialSlowFieldConfig` gains fields `use_SG:bool=False, alpha_G:float=0.0, beta_SG:float=0.0, r0_psi:float=0.0, r50_psi:float=1.0, n_psi:float=2.0, p_pool:float=3.0, tau_mu:float=40.0, tau_S:float=120.0, S_max:float=1.0`. `SpatialSlowField` gains state `self.mu_G:float=0.0`, `self.S_G:float=0.0`, traces `self.trace_muG/trace_SG/trace_AG`. `step()` advances them when `use_SG`.

- [ ] **Step 1: Write failing field tests**

```python
from slow_field import SpatialSlowField, SpatialSlowFieldConfig


def _pool_field(use_SG=True, alpha_G=0.0, **cfgkw):
    # small 2D sheet; E and I positions on an L x L grid
    L = 4.0; nE = 64; nI = 16
    rngp = np.random.default_rng(0)
    posE = rngp.uniform(0, L, size=(nE, 2)); posI = rngp.uniform(0, L, size=(nI, 2))
    cfg = SpatialSlowFieldConfig(use_SG=use_SG, alpha_G=alpha_G, tau_mu=40.0, tau_S=120.0,
                                 r0_psi=0.0, r50_psi=1.0, n_psi=2.0, p_pool=3.0, **cfgkw)
    return SpatialSlowField(nE + nI, 16.5, posE, posI, L, cfg=cfg), nE, nI


def test_SG_builds_and_is_bounded_under_activity():
    f, nE, nI = _pool_field()
    dt = 0.1
    # drive: all E cells "spike" every step for 400 steps -> A_G high -> mu_G,S_G climb, stay bounded
    spk = np.zeros(nE + nI, dtype=bool); spk[:nE] = True
    for _ in range(4000):
        f.step(spk, labels=None, dt=dt)
    assert 0.0 < f.S_G <= f.cfg.S_max + 1e-9
    assert 0.0 < f.mu_G <= 1.0 + 1e-9
    assert f.S_G > 0.1                       # genuinely built, not stuck at 0


def test_SG_off_by_default_no_pool_evolution():
    f, nE, nI = _pool_field(use_SG=False)
    spk = np.zeros(nE + nI, dtype=bool); spk[:nE] = True
    for _ in range(1000):
        f.step(spk, labels=None, dt=0.1)
    assert f.S_G == 0.0 and f.mu_G == 0.0      # OFF -> no evolution
```

- [ ] **Step 2: Run — expect FAIL** (`unexpected keyword 'use_SG'`)

Run: `python -m pytest tests/test_m4_shared_inhibition.py::test_SG_builds_and_is_bounded_under_activity tests/test_m4_shared_inhibition.py::test_SG_off_by_default_no_pool_evolution -q`
Expected: FAIL.

- [ ] **Step 3a: Add config fields** to `SpatialSlowFieldConfig` (after the `beta_G` proxy field, ~`:90`):

```python
    # ---- M4 divisive shared inhibitory pool S_G (rev4 spec §3-§5; OFF by default -> byte-parity) ----
    use_SG: bool = False       # master gate; False -> no pool alloc/evolution, apply_currents unchanged
    alpha_G: float = 0.0       # divisive strength: I_rec_E -> I_rec_E/(1+alpha_G*S_G)
    beta_SG: float = 0.0       # OPTIONAL small subtractive pool current (arm 1/3); NOT beta_G (that is the h_G proxy)
    r0_psi: float = 0.0        # Psi_G recruitment onset
    r50_psi: float = 1.0       # Psi_G half-recruitment
    n_psi: float = 2.0         # Psi_G steepness
    p_pool: float = 3.0        # A_G p-norm exponent (2-4 focal; 1 = area/mean); swept diagnostic
    tau_mu: float = 40.0       # ms, pool activation low-pass (fast)
    tau_S: float = 120.0       # ms, pool output low-pass
    S_max: float = 1.0         # pool output ceiling
```

Add validation in `validate()` (after the h_G checks, ~`:125`):

```python
        for nm, v in (("alpha_G", self.alpha_G), ("beta_SG", self.beta_SG), ("p_pool", self.p_pool),
                      ("tau_mu", self.tau_mu), ("tau_S", self.tau_S), ("S_max", self.S_max),
                      ("r50_psi", self.r50_psi), ("n_psi", self.n_psi)):
            if v < 0.0 if nm in ("alpha_G", "beta_SG") else v <= 0.0:
                raise ValueError(f"{nm} must be {'>= 0' if nm in ('alpha_G','beta_SG') else '> 0'}, got {v}")
```

- [ ] **Step 3b: Add pool state** in `SpatialSlowField.__init__` (after the h_G state block, ~`:206`):

```python
        # ---- M4 shared inhibitory pool (rev4 spec §4) ----
        self.mu_G = 0.0
        self.S_G = 0.0
        self.trace_muG = []; self.trace_SG = []; self.trace_AG = []
```

- [ ] **Step 3c: Advance the pool** in `step()`. Change the fast-EMA guard so `rE_fast` is computed for `use_SG` too, and add the pool block. Replace the `if cfg.use_hG:` opener (~`:253-256`) so `rE_fast` is hoisted:

```python
        if cfg.use_hG or cfg.use_SG:                          # FAST (tau_s) EMA needed by h_G and/or S_G
            if self._alpha_s is None:
                self._alpha_s = 1.0 - np.exp(-dt / cfg.tau_s)
            self.rE_fast += self._alpha_s * (rE_inst - self.rE_fast)
        if cfg.use_hG:                                        # §B6 global recovery (unchanged below)
```

Then DELETE the now-duplicated two `rE_fast` lines that were inside the old `if cfg.use_hG:` block (the `self._alpha_s`/`self.rE_fast +=` pair), since they are hoisted above. After the whole `if cfg.use_hG:` block (~`:271`, before `self._t += dt`), add:

```python
        if cfg.use_SG:                                        # §4 M4 pool advance
            z_G = psi_recruit(self.rE_fast, cfg.r0_psi, cfg.r50_psi, cfg.n_psi)
            A_G = pnorm_pool(z_G, cfg.p_pool)
            self.mu_G += dt * (-self.mu_G + A_G) / cfg.tau_mu           # forward Euler, matches h_G style
            self.mu_G = float(np.clip(self.mu_G, 0.0, 1.0))
            self.S_G += dt * (-self.S_G + cfg.S_max * self.mu_G) / cfg.tau_S
            self.S_G = float(np.clip(self.S_G, 0.0, cfg.S_max))
            self.trace_AG.append(A_G); self.trace_muG.append(self.mu_G); self.trace_SG.append(self.S_G)
```

- [ ] **Step 4: Run — expect PASS**

Run: `python -m pytest tests/test_m4_shared_inhibition.py::test_SG_builds_and_is_bounded_under_activity tests/test_m4_shared_inhibition.py::test_SG_off_by_default_no_pool_evolution -q`
Expected: PASS.

- [ ] **Step 5: Run m3a parity — expect still PASS** (M4 additions gated; use_hG unaffected)

Run: `python -m pytest tests/test_m3a_v2_2_global_recovery.py tests/test_m3a_v2_spatial_slowvars.py -q -k "not visual_diagnostic"`
Expected: PASS (the `visual_diagnostic` figure test is a known pre-existing failure, unrelated).

- [ ] **Step 6: Commit**

```bash
git add src/snn_engine/slow_field.py tests/test_m4_shared_inhibition.py
git commit -m "feat(m4): mu_G/S_G pool state + config + step advance (gated on use_SG)"
```

---

## Task 4: Divisive term in `apply_currents` (`I_E_rec`) + arms

**Files:**
- Modify: `src/snn_engine/slow_field.py` (`apply_currents` ~`:208-219`)
- Test: `tests/test_m4_shared_inhibition.py`

**Interfaces:**
- Consumes: `self.mu_G`, `self.S_G`, `cfg.alpha_G`, `cfg.beta_SG`.
- Produces: `apply_currents(self, I_E, I_I, labels=None, I_E_rec=None) -> ndarray`. When `use_SG` and `I_E_rec is not None`, E-cell output has `ΔI_rec = I_E_rec[:nE]·(alpha_G·S_G/(1+alpha_G·S_G)) + beta_SG·S_G` subtracted. `I_E_rec=None` (all non-M4 callers) → unchanged.

- [ ] **Step 1: Write failing tests** (exact parity at α_G=0; divides recurrent-only; ΔI algebra)

```python
def test_apply_currents_exact_parity_when_alpha_zero():
    f, nE, nI = _pool_field(use_SG=True, alpha_G=0.0)
    f.S_G = 0.7                                    # even with S_G>0, alpha_G=0 -> ZERO divisive term
    N = nE + nI
    I_E = np.linspace(1, 2, N); I_I = np.linspace(0, 1, N); I_E_rec = np.linspace(0, 0.5, N)
    out_with = f.apply_currents(I_E, I_I, labels=None, I_E_rec=I_E_rec)
    out_none = f.apply_currents(I_E, I_I, labels=None, I_E_rec=None)
    assert np.array_equal(out_with, out_none)      # BYTE-exact: alpha_G*S_G=0 -> term is exactly 0


def test_apply_currents_divides_recurrent_E_only():
    f, nE, nI = _pool_field(use_SG=True, alpha_G=2.0)
    f.S_G = 0.5                                     # D_G = 1 + 2*0.5 = 2.0
    N = nE + nI
    I_E = np.full(N, 3.0); I_I = np.zeros(N); I_E_rec = np.full(N, 1.0)
    out = f.apply_currents(I_E, I_I, labels=None, I_E_rec=I_E_rec)
    # E cells: I_net = I_E - q_I*I_I - eta_K*g_K - I_E_rec*(alpha*S/(1+alpha*S))
    #        = 3 - 0 - 0 - 1*(1.0/2.0) = 2.5    (equivalently I_ff + I_rec/D_G = 2 + 1/2)
    assert np.allclose(out[:nE], 2.5)
    # I cells untouched: I_E - I_I = 3
    assert np.allclose(out[nE:], 3.0)


def test_beta_SG_subtractive_arm():
    f, nE, nI = _pool_field(use_SG=True, alpha_G=0.0, beta_SG=0.4)
    f.S_G = 0.5
    N = nE + nI
    I_E = np.full(N, 3.0); I_I = np.zeros(N); I_E_rec = np.full(N, 1.0)
    out = f.apply_currents(I_E, I_I, labels=None, I_E_rec=I_E_rec)
    assert np.allclose(out[:nE], 3.0 - 0.4 * 0.5)   # only the subtractive pool term
    assert np.allclose(out[nE:], 3.0)
```

- [ ] **Step 2: Run — expect FAIL** (`apply_currents` has no `I_E_rec` param)

Run: `python -m pytest tests/test_m4_shared_inhibition.py -q -k apply_currents_or_beta` (adjust `-k` to the three test names)
Expected: FAIL (TypeError / assertion).

- [ ] **Step 3: Modify `apply_currents`** (`src/snn_engine/slow_field.py` ~`:208`):

```python
    def apply_currents(self, I_E, I_I, labels=None, I_E_rec=None):
        """I_net = I_E - q_I*I_I - eta_K*g_K - eta_G*h_G for E cells; I_E - I_I for I cells.
        M4 (use_SG + I_E_rec given): additionally subtract the removed recurrent current
        dI_rec = I_E_rec[:nE]*(alpha_G*S_G/(1+alpha_G*S_G)) + beta_SG*S_G  (divide recurrent E only).
        alpha_G*S_G=0 and beta_SG=0 -> dI_rec exactly 0 -> byte-parity."""
        qI_E = self.q_I[self._iyE, self._ixE]
        gK_E = self.g_K[self._iyE, self._ixE]
        out = np.asarray(I_E, float) - np.asarray(I_I, float)
        nE = self.nE
        hG_eff = self.h_G if self.cfg.use_hG else 0.0
        out[:nE] = (I_E[:nE] - qI_E * I_I[:nE]
                    - self.cfg.eta_K * gK_E
                    - self.cfg.eta_G * hG_eff)
        if self.cfg.use_SG and I_E_rec is not None:
            aS = self.cfg.alpha_G * self.S_G
            frac = aS / (1.0 + aS)                                  # aS=0 -> 0 (exact)
            out[:nE] -= np.asarray(I_E_rec, float)[:nE] * frac + self.cfg.beta_SG * self.S_G
        return out
```

- [ ] **Step 4: Run — expect PASS**

Run: `python -m pytest tests/test_m4_shared_inhibition.py -q -k "apply_currents or beta_SG"`
Expected: PASS (all three).

- [ ] **Step 5: Commit**

```bash
git add src/snn_engine/slow_field.py tests/test_m4_shared_inhibition.py
git commit -m "feat(m4): divisive recurrent-gain term + beta_SG subtractive arm in apply_currents"
```

---

## Task 5: End-to-end smoke — `use_SG=False`≡old, and `alpha_G>0` reduces recurrent runaway

**Files:**
- Test: `tests/test_m4_shared_inhibition.py`

**Interfaces:**
- Consumes: `simulate_kick` (Task 1), `SpatialSlowField`/`SpatialSlowFieldConfig` with `use_SG` (Tasks 3-4).

- [ ] **Step 1: Write the smoke tests**

```python
def _slow_for(p, net, **cfgkw):
    NE, NI = net["NE"], net["NI"]
    posE = net["pos"][net["labels"] == 0]; posI = net["pos"][net["labels"] == 1]
    cfg = SpatialSlowFieldConfig(**cfgkw)
    return SpatialSlowField(NE + NI, p.V_th, posE, posI, p.L, cfg=cfg)


def test_use_SG_off_matches_slow_none_engine_output():
    # A SpatialSlowField with everything neutral (use_SG=False, k_q=0,k_K=0,use_hG=False)
    # must give the SAME engine output as slow=None (byte-parity of the gated path).
    p, net = _tiny_net(T=150.0)
    net["rng"] = np.random.default_rng(3)
    a = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9)
    p, net = _tiny_net(T=150.0)
    net["rng"] = np.random.default_rng(3)
    slow = _slow_for(p, net)      # defaults: all mechanisms off, use_SG=False
    b = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, slow=slow)
    assert hashlib.sha1(a["E_spk_bool"].tobytes()).hexdigest() == \
           hashlib.sha1(b["E_spk_bool"].tobytes()).hexdigest()


def test_alpha_G_reduces_total_E_spikes_vs_neutral_pool():
    # With the pool ON but alpha_G=0 vs alpha_G large under a kick that would otherwise recruit hard:
    # divisive pool should NOT increase, and for large alpha_G should reduce, total E spikes.
    def total(alpha_G):
        p, net = _tiny_net(T=300.0)
        net["rng"] = np.random.default_rng(7)
        slow = _slow_for(p, net, use_SG=True, alpha_G=alpha_G, tau_mu=30.0, tau_S=80.0,
                         r0_psi=0.0, r50_psi=1.0, n_psi=2.0, p_pool=3.0, S_max=1.0)
        res = simulate_kick(p, net, KICK_BOOST=8.0, r_kick=1.5,
                            V_th_per_neuron=np.full(net["NE"] + net["NI"], 16.5), slow=slow)
        return float(res["E_spk_bool"].sum())
    n0, n_big = total(0.0), total(6.0)
    assert n_big <= n0 + 1e-9      # divisive pool does not increase recruitment; large alpha reduces it
```

- [ ] **Step 2: Run — expect the parity test PASS; tune the smoke if needed**

Run: `python -m pytest tests/test_m4_shared_inhibition.py -q`
Expected: `test_use_SG_off_matches_slow_none_engine_output` PASS. `test_alpha_G_reduces_total_E_spikes_vs_neutral_pool` should PASS; if the tiny net doesn't recruit enough for the pool to bite, raise `KICK_BOOST`/`T` or lower `r50_psi` until `n0` is a non-trivial count, THEN assert the inequality (document the chosen values inline).

- [ ] **Step 3: Full targeted suite green**

Run: `python -m pytest tests/test_m4_shared_inhibition.py tests/test_a1c_feedback.py tests/test_m3a_v2_2_global_recovery.py -q -k "not visual_diagnostic"`
Expected: PASS (except the known unrelated `visual_diagnostic` figure test).

- [ ] **Step 4: Commit**

```bash
git add tests/test_m4_shared_inhibition.py
git commit -m "test(m4): end-to-end smoke — use_SG off == slow=None; alpha_G does not increase recruitment"
```

- [ ] **Step 5: STOP for user review.** Post a summary: files changed, test status, and the diff of the divisive term. Do NOT proceed to Task 6.

---

## Task 6 (SIM GATE — DO NOT EXECUTE until user review): `q_core × alpha_G` phase-plane + §9.1 go/no-go

> This task is written for completeness so the reviewer sees the full Pass-1 shape. **It runs simulations (the scientific experiment).** Per the user's instruction, STOP after Task 5; run Task 6 only after the plan + implementation are reviewed.

**Files:**
- Create: `src/sef_hfo_m4_phaseplane.py` — `derive_core_mask(...)`, `q_core(...)`, per-cell classifier (`persist/bounded/act_frac/S_grad/F_off/core_overlap/globality`), `classify_cell(...)` implementing §9.1 TRIVIAL-A/B exclusions + `go(cell)`.
- Create: `scripts/run_m4_phaseplane.py` — sweep `q_core ∈ [q_min,1] × alpha_G ∈ [0,alpha_max]` for arm 0 (baseline) / arm 1 (`beta_SG`) / arm 2 (`alpha_G` divisive); write `results/topic4_m4/phase_plane_qcore_alpha.{json,png}` + `figures/README.md`.

**Pre-req (also gated):** derive `m_core` from an arm-0 kick's first-activation map; generate arm-0 TRIVIAL-A/TRIVIAL-B reference instances and **calibrate** `theta_core/theta_glob/theta_off` to exclude them; LOCK the values into the spec's §9.1 calibration table BEFORE the sweep.

**Success gate (§9.1):** go(plane) = ≥ K_min contiguous go(cell) present in arm 2 but NOT arm 1, none explained by TRIVIAL-A/TRIVIAL-B. A clean no-go is a valid result.

---

## Self-Review notes

- **Spec coverage:** §5 divisive membrane → Task 4; §6 AMPA split → Task 1; §3 sensor → Task 2; §4 pool → Task 3; §7 arms (0/1/2) → Task 4 (`alpha_G`/`beta_SG`) + Task 5 smoke; §8.1/§9.1 → Task 6 (gated). §10.2 (Pass-2 Jacobian, κ_k) is NOT in Pass-1 (gated on criticality merge) — out of scope by design.
- **Byte-parity:** Task 1 (fixtures + re-bless) + Task 3 Step 5 (m3a parity) + Task 5 (`use_SG` off == `slow=None`).
- **Naming:** `beta_SG` (not `beta_G`), `mu_G` (not `u_G`). Confirmed against `slow_field.py:90` collision.
- **Import anchoring:** the `_tiny_net`/`build_network`/`Params` names in the tests MUST be reconciled with the actual helper used in `tests/test_m3a_v2_2_global_recovery.py` / `tests/test_a1c_feedback.py` (`_net(...)`); read those first at Task 1 Step 1.
