# SEF-HFO Step 0a/0b (Delayed Linear Stability + Finite-Pulse Screen) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the Step 0 core of the SEF-HFO two-stage exploratory mechanism screen — a **delayed** effective-gain → linear-dispersion map (0a) and a finite-pulse response map (0b) — that decides whether a self-limited, gain-closed, control-disciplined interictal working window exists, behind a hard go/no-go gate.

**Architecture:** Three `src/` modules: `sef_hfo_stability.py` (transfer function, gain, self-consistent operating point, **augmented per-mode dispersion matrix** carrying synaptic kinetics + Erlang-chain conduction delay + recovery, `eta_lin`, delay-validation, low-heterogeneity screen); `sef_hfo_field.py` (params, kernels, 2-D field integrator whose state mirrors the dispersion model — rate + synaptic + delay-chain + recovery); `sef_hfo_pulse.py` (wavefront-aware response classifier, adaptive amplitude thresholds, pulse family). Two CLI runners produce JSON + figures under `results/topic4_sef_hfo/`.

**Delays (2026-06-02 decision):** delays/kinetics enter **as auxiliary linear states**, not as a transcendental `e^{-λd}` factor. Each connection's temporal response = Erlang-`n` conduction-delay chain (n stages, rate `n/d`; n→∞ → pure delay `d`) followed by an exponential synaptic filter (`τ_AMPA` for E-presynaptic, `τ_GABA` for I-presynaptic). The per-mode dispersion is then the **eigenvalues of a constant matrix** (`np.linalg.eigvals`) — robust, no missed-root hazard — and the field integrator needs only extra ODE states, no delay history buffer. The two layers are therefore the **same model** by construction.

**Tech Stack:** Python 3, numpy, scipy (`optimize.fsolve`, `optimize.brentq`), pytest, matplotlib.

**Reference (align to this):** Bachschmid-Romano, Hatsopoulos & Brunel 2026, *A Spatially Structured Spiking Network Model of Beta Traveling Waves…* (bioRxiv 2026.03.18.712701). Same machinery — 2×2 self-consistent dispersion `λ(k)`, external-drive phase diagram, anisotropic E→E selecting a propagation axis, `|I^E|+|I^I|` LFP proxy. **Key distinction to preserve:** their event is a near-global Turing–Hopf traveling wave; ours is a *localized self-limited* transient. So 0b's recovery-OFF / recovery-ON comparison is exactly: sub-critical-near-Hopf (Brunel branch) vs adaptation-driven localized pulse (Pinto–Ermentrout branch). Anchor synaptic/membrane timescales and connectivity footprint to their Table 1 at the *formal* run.

---

## Governing contract (implements the 2026-06-02 amendment + this review round)

Freeze the **contracts, tests, and discipline**; **NOT** parameter values (every `SEFParams` default and config value is a TEST-ONLY SCAFFOLD). The exploratory *output* is which parameter regions pass.

- **Patch A:** self-consistent operating point; admissible family fixed from data BEFORE the σ_φ sweep (structured provenance; scaffold cannot pass the formal gate); screen reports the **fraction of the family** where lowering σ_φ moves closer to the boundary, never "exists a point"; only σ_φ may differ (no rescue).
- **Patch B:** recovery is a switchable rate-layer state; 0b runs recovery OFF and ON and **reports both, no auto-select**; recovery timescale anchored to measured event duration at the formal run.
- **Finite-pulse necessity:** 0a (delayed dispersion) only locates candidate windows; only parameters with a `self_limited_propagation` response and a **positive `A_runaway − A_event` margin** pass.
- **Discriminator/reuse (patch C/D/E)** are Step 2/4 scope — intentionally **NOT** in this plan.

**Review-round fixes folded in (commit-blocking):** (1) homogeneous-patch approximation, position labels dropped; (2) structured data-locked provenance gate; (3) 0b consumes 0a's operating-point family, reports a fraction; (4) recovery report-both, no auto-pick; (5) units-anchoring requirement gates the formal run; (6) L/grid + dt + adaptive-amplitude sensitivity; (7) `fsolve` convergence + multi-start + multiplicity; (8) dt-sensitivity (coupled to Erlang n); (9) field-linearization ↔ eigenvalue consistency; (10) wavefront/centroid + global-mode classifier.

**Commits** end with the repo trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` (omitted from examples below).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/sef_hfo_field.py` | `SEFParams` (rate + synaptic τ + delay d/n + recovery, all SCAFFOLD); kernels; FFT convolution; `F_eff` lookup; field integrator with synaptic + Erlang-delay + recovery states |
| `src/sef_hfo_stability.py` | `f`/`f'`; `F_eff`/`gain`; self-consistent operating point (+convergence/multistart/multiplicity); analytic kernel FT; **augmented per-mode dispersion matrix**; `eta_lin`; delay validation (n-convergence + bounded transcendental cross-check); low-het screen + no-rescue |
| `src/sef_hfo_pulse.py` | wavefront-aware `classify_response` (+ global-mode class); `run_pulse`; adaptive `amplitude_thresholds`; `PULSE_FAMILY` (radii × durations, single location) |
| `config/sef_hfo_operating_points.json` | admissible family + **structured provenance** (`source`, `locked_before_sweep`, `derived_from`, `locked_at`, `cohort`, `hash`) |
| `scripts/run_sef_hfo_step0a_stability.py` | 0a: delayed dispersion over family, phase diagram, k*/ω*, n-convergence, screen |
| `scripts/run_sef_hfo_step0b_pulse.py` | 0b: pulse map over **0a candidate family**, recovery OFF+ON (report both), L/grid+dt sensitivity |
| `tests/test_sef_hfo_stability.py` | sign-flip; fixed-point + convergence; matrix reduction; n-convergence; transcendental cross-check; screen + no-rescue |
| `tests/test_sef_hfo_field.py` | kernel normalization; steady-state hold; recovery wiring; **field-linearization ↔ eigenvalue** consistency |
| `tests/test_sef_hfo_pulse.py` | classifier 5 regimes (incl. global flash); adaptive-threshold logic |
| `tests/test_sef_hfo_step0_gate.py` | structured-provenance formal gate; units-anchoring; self-limited-window margin |
| `results/topic4_sef_hfo/{linear_stability,finite_pulse}/` | outputs (+ Chinese `figures/README.md`) |
| `docs/archive/topic4/sef_itp_phase4_v2/step0_results_2026-06-02.md` | go/no-go writeup |

---

## Task 1: Scaffold — params (rate + synaptic + delay + recovery), structured provenance, dirs

**Files:** Create `src/sef_hfo_field.py`, `config/sef_hfo_operating_points.json`, `tests/test_sef_hfo_field.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_sef_hfo_field.py
from src.sef_hfo_field import SEFParams

def test_params_scaffold_invariants():
    p = SEFParams()
    assert p.ell_perp < p.ell_par            # propagation axis exists
    assert p.sigma_I > p.ell_par             # wide inhibition
    assert p.tau_AMPA < p.tau_GABA           # fast excitation, slow inhibition
    assert p.b_a == 0.0                      # recovery OFF by default (switchable)
    assert p.erlang_n >= 1
```

- [ ] **Step 2: Run → fail** (`ImportError`): `pytest tests/test_sef_hfo_field.py::test_params_scaffold_invariants -v`

- [ ] **Step 3: Implement**

```python
# src/sef_hfo_field.py
"""SEF-HFO Step-0 rate field. ALL numeric defaults are TEST-ONLY SCAFFOLDS
(Step 0a/0b screen them); formal results require data-anchored units (see gate).
See docs/archive/topic4/sef_hfo_topic4_v2_plan_2026-06-01.md 2026-06-02 amendment."""
from dataclasses import dataclass

@dataclass(frozen=True)
class SEFParams:
    n: int = 64
    L: float = 64.0
    tau_E: float = 1.0
    tau_I: float = 2.0
    tau_a: float = 20.0           # recovery timescale (anchor to event duration at formal run)
    tau_AMPA: float = 0.5         # E-presynaptic synaptic filter (fast)
    tau_GABA: float = 2.0         # I-presynaptic synaptic filter (slow)
    delay_d: float = 1.0          # conduction delay (Erlang chain mean)
    erlang_n: int = 2             # Erlang stages approximating the delay (n->inf = pure delay)
    J_EE: float = 1.0
    J_EI: float = 1.0
    J_IE: float = 1.0
    J_II: float = 0.5
    ell_par: float = 6.0
    ell_perp: float = 2.0
    axis_angle: float = 0.0
    sigma_I: float = 10.0
    sigma_IE: float = 4.0
    sigma_II: float = 4.0
    gamma_global: float = 0.2
    b_a: float = 0.0              # recovery strength; 0 = OFF
    beta: float = 4.0
    phi_bar: float = 0.0
    sigma_phi: float = 1.0
```

- [ ] **Step 4: Run → pass.**

- [ ] **Step 5: Structured-provenance config + dirs**

```json
// config/sef_hfo_operating_points.json
{
  "provenance": {
    "source": "scaffold",
    "locked_before_sweep": false,
    "derived_from": "PLACEHOLDER background-rate band; replace with data-fixed band",
    "locked_at": null,
    "cohort": null,
    "hash": null
  },
  "operating_points": [
    {"I_E": 0.20, "I_I": 0.10}, {"I_E": 0.30, "I_I": 0.10},
    {"I_E": 0.40, "I_I": 0.15}, {"I_E": 0.50, "I_I": 0.20},
    {"I_E": 0.60, "I_I": 0.25}
  ]
}
```
```bash
mkdir -p results/topic4_sef_hfo/linear_stability/figures results/topic4_sef_hfo/finite_pulse/figures
```

- [ ] **Step 6: Commit** — `feat(topic4 sef-hfo step0): SEFParams (delay/synaptic/recovery) + structured-provenance config`

---

## Task 2: Population transfer function + gain (patch A sign-flip)

**Files:** Create `src/sef_hfo_stability.py`, `tests/test_sef_hfo_stability.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_sef_hfo_stability.py
import numpy as np
from src.sef_hfo_stability import F_eff, gain, _f

def test_Feff_reduces_to_single_unit_when_sigma_zero():
    for h in [-1.0, 0.0, 1.5]:
        assert abs(F_eff(h, 0.0, 1e-6, 4.0) - _f(h, 4.0)) < 1e-4

def test_gain_sign_flips_between_steep_point_and_tail():
    g_steep_small = gain(0.0, 0.0, 0.3, 4.0); g_steep_large = gain(0.0, 0.0, 2.0, 4.0)
    assert g_steep_small > g_steep_large                 # steep point: lower sigma -> higher gain
    g_tail_small = gain(6.0, 0.0, 0.3, 4.0); g_tail_large = gain(6.0, 0.0, 2.0, 4.0)
    assert g_tail_small < g_tail_large                   # deep tail: sign flips
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement**

```python
# src/sef_hfo_stability.py
"""SEF-HFO Step-0a delayed linear stability. See 2026-06-02 amendment."""
import numpy as np
from dataclasses import replace, fields

def _f(x, beta): return 1.0 / (1.0 + np.exp(-beta * x))
def _fprime(x, beta):
    s = _f(x, beta); return beta * s * (1.0 - s)

_Z = np.linspace(-8.0, 8.0, 2001); _PZ = np.exp(-0.5 * _Z**2) / np.sqrt(2 * np.pi)

def F_eff(h, phi_bar, sigma_phi, beta):
    return np.trapz(_f(h - (phi_bar + sigma_phi * _Z), beta) * _PZ, _Z)

def gain(h, phi_bar, sigma_phi, beta):
    return np.trapz(_fprime(h - (phi_bar + sigma_phi * _Z), beta) * _PZ, _Z)
```

- [ ] **Step 4: Run → pass. Step 5: Commit** — `feat(topic4 sef-hfo step0a): transfer function + gain (operating-point sign-flip)`

---

## Task 3: Self-consistent operating point (convergence + multi-start + multiplicity)

**Files:** Modify `src/sef_hfo_stability.py`, `tests/test_sef_hfo_stability.py`

- [ ] **Step 1: Failing tests**

```python
# add to tests/test_sef_hfo_stability.py
from src.sef_hfo_field import SEFParams
from src.sef_hfo_stability import self_consistent_operating_point, F_eff

def test_operating_point_self_consistent_and_converged():
    p = SEFParams()
    op = self_consistent_operating_point(p, 0.4, 0.15)
    assert op["converged"] is True
    assert abs(F_eff(op["h_E0"], p.phi_bar, p.sigma_phi, p.beta) - op["r_E0"]) < 1e-6
    assert abs(F_eff(op["h_I0"], p.phi_bar, p.sigma_phi, p.beta) - op["r_I0"]) < 1e-6

def test_operating_point_flags_multiplicity():
    # strong recurrent excitation + low heterogeneity can be bistable -> >1 distinct root
    p = SEFParams(J_EE=4.0, sigma_phi=0.3, beta=8.0)
    op = self_consistent_operating_point(p, 0.5, 0.05)
    assert "n_distinct_roots" in op and op["n_distinct_roots"] >= 1
    if op["n_distinct_roots"] > 1:
        assert op["bistable"] is True
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement**

```python
# add to src/sef_hfo_stability.py
from scipy.optimize import fsolve

def self_consistent_operating_point(p, I_E, I_I, guesses=None, tol=1e-8):
    """Solve r0 = F_eff(W_hat(0) r0 + I) from multiple initial guesses; report
    convergence, residual, and root multiplicity (near-critical bistability)."""
    def resid(r):
        rE, rI = r
        hE = p.J_EE * rE - p.J_EI * rI + I_E - p.b_a * rE
        hI = p.J_IE * rE - p.J_II * rI + I_I
        return [rE - F_eff(hE, p.phi_bar, p.sigma_phi, p.beta),
                rI - F_eff(hI, p.phi_bar, p.sigma_phi, p.beta)]
    if guesses is None:
        guesses = [(a, b) for a in (0.02, 0.2, 0.5, 0.9) for b in (0.02, 0.2, 0.5)]
    roots = []
    for g in guesses:
        sol, info, ier, _ = fsolve(resid, g, full_output=True)
        if ier == 1 and max(abs(np.array(resid(sol)))) < tol and (sol >= -1e-6).all():
            if not any(np.allclose(sol, r, atol=1e-4) for r in roots):
                roots.append(sol)
    if not roots:
        return {"converged": False, "n_distinct_roots": 0}
    rE0, rI0 = max(roots, key=lambda r: r[0])     # pick high-activity root deterministically
    hE0 = p.J_EE * rE0 - p.J_EI * rI0 + I_E - p.b_a * rE0
    hI0 = p.J_IE * rE0 - p.J_II * rI0 + I_I
    return {"r_E0": float(rE0), "r_I0": float(rI0), "h_E0": float(hE0), "h_I0": float(hI0),
            "converged": True, "n_distinct_roots": len(roots), "bistable": len(roots) > 1}
```

- [ ] **Step 4: Run → pass. Step 5: Commit** — `feat(topic4 sef-hfo step0a): operating point with convergence + multiplicity check (review #7)`

---

## Task 4: Augmented per-mode dispersion matrix (synaptic kinetics + Erlang delay + recovery) → eta_lin

**Files:** Modify `src/sef_hfo_stability.py`, `tests/test_sef_hfo_stability.py`

- [ ] **Step 1: Failing tests**

```python
# add to tests/test_sef_hfo_stability.py
from src.sef_hfo_stability import (gaussian_hat, build_dispersion_matrix, eta_lin)

def test_matrix_reduces_to_rate_jacobian_without_kinetics():
    # erlang_n=0, tau_syn=0, b_a=0 => augmented matrix == bare 2x2 rate Jacobian
    p = SEFParams(erlang_n=0, tau_AMPA=0.0, tau_GABA=0.0, b_a=0.0)
    op = self_consistent_operating_point(p, 0.4, 0.15)
    M = build_dispersion_matrix(p, op, kpar=0.5, kperp=0.0)
    assert M.shape == (2, 2)
    G_E = gain(op["h_E0"], p.phi_bar, p.sigma_phi, p.beta)
    G_I = gain(op["h_I0"], p.phi_bar, p.sigma_phi, p.beta)
    WEE = p.J_EE * gaussian_hat(0.5, 0.0, p.ell_par, p.ell_perp)
    assert abs(M[0, 0] - (-1.0 + G_E * WEE) / p.tau_E) < 1e-12

def test_no_coupling_decoupled_eigenvalues():
    p = SEFParams(J_EE=0, J_EI=0, J_IE=0, J_II=0, erlang_n=0, tau_AMPA=0.0, tau_GABA=0.0)
    op = self_consistent_operating_point(p, 0.3, 0.1)
    ev = np.sort(np.linalg.eigvals(build_dispersion_matrix(p, op, 0.5, 0.0)).real)
    assert np.allclose(ev, np.sort([-1/p.tau_E, -1/p.tau_I]), atol=1e-9)

def test_stronger_EE_less_stable_at_fixed_operating_point():
    op = self_consistent_operating_point(SEFParams(), 0.4, 0.15)
    k = np.linspace(-2, 2, 25)
    assert eta_lin(SEFParams(J_EE=1.5), op, k) < eta_lin(SEFParams(J_EE=0.5), op, k)
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement**

```python
# add to src/sef_hfo_stability.py
def gaussian_hat(kpar, kperp, ell_par, ell_perp):
    return np.exp(-0.5 * (ell_par**2 * kpar**2 + ell_perp**2 * kperp**2))

def _inhib_hat(p, kpar, kperp):
    wide = (1.0 - p.gamma_global) * gaussian_hat(kpar, kperp, p.sigma_I, p.sigma_I)
    return wide + (p.gamma_global if (kpar == 0.0 and kperp == 0.0) else 0.0)

def _connection_specs(p):
    # (post, pre, sign, J, kernel_hat, tau_syn[by presynaptic transmitter])
    return [
        ("E", "E", +1, p.J_EE, gaussian_hat(0, 0, p.ell_par, p.ell_perp), p.tau_AMPA, "EE"),
        ("E", "I", -1, p.J_EI, None, p.tau_GABA, "EI"),
        ("I", "E", +1, p.J_IE, None, p.tau_AMPA, "IE"),
        ("I", "I", -1, p.J_II, None, p.tau_GABA, "II"),
    ]

def _kernel_hat(p, tag, kpar, kperp):
    if tag == "EE": return gaussian_hat(kpar, kperp, p.ell_par, p.ell_perp)
    if tag == "EI": return _inhib_hat(p, kpar, kperp)
    if tag == "IE": return gaussian_hat(kpar, kperp, p.sigma_IE, p.sigma_IE)
    if tag == "II": return gaussian_hat(kpar, kperp, p.sigma_II, p.sigma_II)

def _chain_rates(p, tau_syn):
    """Erlang-n conduction-delay stages (rate n/d) then one synaptic stage (1/tau_syn)."""
    rates = []
    if p.erlang_n > 0 and p.delay_d > 0:
        rates += [p.erlang_n / p.delay_d] * p.erlang_n
    if tau_syn > 0:
        rates += [1.0 / tau_syn]
    return rates

def build_dispersion_matrix(p, op, kpar, kperp):
    """Constant linear matrix whose eigenvalues are the growth rates at mode (kpar,kperp).
    State = [r_E, r_I, (per-connection delay+synaptic chain stages...), (recovery a)]."""
    G = {"E": gain(op["h_E0"], p.phi_bar, p.sigma_phi, p.beta),
         "I": gain(op["h_I0"], p.phi_bar, p.sigma_phi, p.beta)}
    tau = {"E": p.tau_E, "I": p.tau_I}
    idx = {"E": 0, "I": 1}; nxt = 2; chains = []
    for (post, pre, sign, J, _, tau_syn, tag) in _connection_specs(p):
        weight = J * _kernel_hat(p, tag, kpar, kperp)
        rates = _chain_rates(p, tau_syn)
        sidx = list(range(nxt, nxt + len(rates))); nxt += len(rates)
        chains.append((post, pre, sign, weight, sidx, rates))
    rec_idx = None
    if p.b_a > 0: rec_idx = nxt; nxt += 1
    M = np.zeros((nxt, nxt))
    M[0, 0] = -1.0 / p.tau_E; M[1, 1] = -1.0 / p.tau_I
    for (post, pre, sign, weight, sidx, rates) in chains:
        out = sidx[-1] if sidx else idx[pre]
        M[idx[post], out] += G[post] * sign * weight / tau[post]
        src = idx[pre]
        for j, (si, rj) in enumerate(zip(sidx, rates)):
            M[si, si] += -rj; M[si, src] += rj; src = si
    if rec_idx is not None:
        M[rec_idx, rec_idx] = -1.0 / p.tau_a; M[rec_idx, 0] = 1.0 / p.tau_a
        M[0, rec_idx] += G["E"] * (-p.b_a) / p.tau_E
    return M

def max_growth_rate(p, op, k_grid):
    best = -np.inf
    for kpar in k_grid:
        for kperp in k_grid:
            ev = np.linalg.eigvals(build_dispersion_matrix(p, op, float(kpar), float(kperp)))
            best = max(best, float(ev.real.max()))
    return best

def eta_lin(p, op, k_grid):
    return -max_growth_rate(p, op, k_grid)

def leading_mode(p, op, k_grid):
    """Return (k*, omega*, max Re lambda) for the dominant mode (for k*/frequency reporting)."""
    best = (-np.inf, 0.0, 0.0)
    for kpar in k_grid:
        ev = np.linalg.eigvals(build_dispersion_matrix(p, op, float(kpar), 0.0))
        j = int(np.argmax(ev.real))
        if ev.real[j] > best[0]:
            best = (float(ev.real[j]), float(kpar), float(abs(ev.imag[j])))
    return {"max_re": best[0], "k_star": best[1], "omega_star": best[2]}
```

- [ ] **Step 4: Run → pass (all three). Step 5: Commit** — `feat(topic4 sef-hfo step0a): augmented delayed dispersion matrix + eta_lin`

---

## Task 5: Delay validation — n-convergence at the leading mode + bounded transcendental cross-check

**Files:** Modify `src/sef_hfo_stability.py`, `tests/test_sef_hfo_stability.py`

- [ ] **Step 1: Failing tests**

```python
# add to tests/test_sef_hfo_stability.py
from src.sef_hfo_stability import erlang_n_convergence, transcendental_max_re

def test_n_convergence_at_leading_mode():
    # leading-mode growth rate must converge as Erlang n increases (advisor gate:
    # distributed delay over-stabilizes; check convergence AT the leading mode).
    p = SEFParams(); op = self_consistent_operating_point(p, 0.4, 0.15)
    k = np.linspace(-2, 2, 25)
    conv = erlang_n_convergence(p, op, k, n_values=(1, 2, 4, 8))
    assert conv["converged"]                       # |Δ max_re| between top n's below tol
    assert conv["recommended_n"] <= 8

def test_transcendental_cross_check_one_slice():
    # bounded complex-box scan of the EXACT delayed dispersion D(lambda,k)=0 must agree
    # with the augmented-matrix max Re lambda on one coarse slice (catches algebra errors).
    p = SEFParams(erlang_n=16); op = self_consistent_operating_point(p, 0.4, 0.15)
    k0 = 0.5
    m_matrix = float(np.linalg.eigvals(build_dispersion_matrix(p, op, k0, 0.0)).real.max())
    m_exact = transcendental_max_re(p, op, k0, re_lo=-3.0, re_hi=0.5, im_hi=8.0)
    assert abs(m_matrix - m_exact) < 0.05
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement**

```python
# add to src/sef_hfo_stability.py
def erlang_n_convergence(p, op, k_grid, n_values=(1, 2, 4, 8), tol=1e-3):
    """Leading-mode growth rate vs Erlang n. Returns smallest n past which it stops moving."""
    vals = [(n, max_growth_rate(replace(p, erlang_n=n), op, k_grid)) for n in n_values]
    deltas = [abs(vals[i][1] - vals[i - 1][1]) for i in range(1, len(vals))]
    converged = bool(deltas and deltas[-1] < tol)
    rec = next((n for (n, _), d in zip(vals[1:], deltas) if d < tol), n_values[-1])
    return {"values": vals, "deltas": deltas, "converged": converged, "recommended_n": rec}

def _char_det(lam, p, op, kpar, kperp):
    """EXACT delayed dispersion determinant D(lambda,k) with e^{-lambda d}/(1+lambda tau_syn)."""
    G = {"E": gain(op["h_E0"], p.phi_bar, p.sigma_phi, p.beta),
         "I": gain(op["h_I0"], p.phi_bar, p.sigma_phi, p.beta)}
    def H(tau_syn):
        return np.exp(-lam * p.delay_d) / (1.0 + lam * tau_syn)
    WEE = p.J_EE * gaussian_hat(kpar, kperp, p.ell_par, p.ell_perp)
    WEI = p.J_EI * _inhib_hat(p, kpar, kperp)
    WIE = p.J_IE * gaussian_hat(kpar, kperp, p.sigma_IE, p.sigma_IE)
    WII = p.J_II * gaussian_hat(kpar, kperp, p.sigma_II, p.sigma_II)
    a = (1 + p.tau_E * lam) - G["E"] * WEE * H(p.tau_AMPA)
    b = G["E"] * WEI * H(p.tau_GABA)
    c = -G["I"] * WIE * H(p.tau_AMPA)
    d = (1 + p.tau_I * lam) + G["I"] * WII * H(p.tau_GABA)
    return a * d - b * c

def transcendental_max_re(p, op, k0, re_lo, re_hi, im_hi, n_re=60, n_im=60):
    """Bounded rightmost-root estimate: scan a complex box for sign changes of Re/Im D,
    return the largest Re(lambda) of detected roots. Robust (no naive Newton wandering)."""
    res = np.linspace(re_lo, re_hi, n_re); ims = np.linspace(0.0, im_hi, n_im)
    best = -np.inf
    for i in range(n_re - 1):
        for j in range(n_im - 1):
            corners = [_char_det(complex(res[a], ims[b]), p, op, k0, 0.0)
                       for a in (i, i + 1) for b in (j, j + 1)]
            if (min(c.real for c in corners) < 0 < max(c.real for c in corners) and
                    min(c.imag for c in corners) < 0 < max(c.imag for c in corners)):
                best = max(best, res[i])           # root bracketed in this cell
    return best
```

- [ ] **Step 4: Run → pass. Step 5: Commit** — `feat(topic4 sef-hfo step0a): delay validation (n-convergence + bounded transcendental cross-check)`

---

## Task 6: Low-heterogeneity screen + no-rescue guard (patch A)

**Files:** Modify `src/sef_hfo_stability.py`, `tests/test_sef_hfo_stability.py`

- [ ] **Step 1: Failing tests**

```python
# add to tests/test_sef_hfo_stability.py
import pytest
from dataclasses import replace
from src.sef_hfo_stability import _assert_only_sigma_phi_differs, screen_low_heterogeneity_effect

def test_no_rescue_guard():
    p = SEFParams()
    with pytest.raises(ValueError):
        _assert_only_sigma_phi_differs(p, replace(p, sigma_phi=0.5, phi_bar=0.3))
    _assert_only_sigma_phi_differs(p, replace(p, sigma_phi=0.5))   # ok

def test_screen_reports_fraction_over_whole_family():
    p = SEFParams(); fam = [(0.3, 0.1), (0.4, 0.15), (0.5, 0.2)]; k = np.linspace(-2, 2, 17)
    out = screen_low_heterogeneity_effect(p, fam, 0.5, k)
    assert out["n_admissible"] == len(fam) == len(out["per_point"])
    assert abs(out["fraction_closer"] - np.mean([d["closer_to_critical"] for d in out["per_point"]])) < 1e-12
    with pytest.raises(ValueError):
        screen_low_heterogeneity_effect(p, fam, 2.0, k)           # patch must LOWER sigma_phi
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement**

```python
# add to src/sef_hfo_stability.py
def _assert_only_sigma_phi_differs(p_base, p_patch):
    for fld in fields(p_base):
        if fld.name == "sigma_phi": continue
        if getattr(p_base, fld.name) != getattr(p_patch, fld.name):
            raise ValueError(f"forbidden rescue: '{fld.name}' changed alongside sigma_phi")

def screen_low_heterogeneity_effect(p_base, admissible_operating_points, sigma_phi_patch, k_grid):
    if sigma_phi_patch >= p_base.sigma_phi:
        raise ValueError("patch must LOWER sigma_phi")
    p_patch = replace(p_base, sigma_phi=sigma_phi_patch)
    _assert_only_sigma_phi_differs(p_base, p_patch)
    per = []
    for I_E, I_I in admissible_operating_points:
        ob = self_consistent_operating_point(p_base, I_E, I_I)
        opp = self_consistent_operating_point(p_patch, I_E, I_I)
        eb = eta_lin(p_base, ob, k_grid); ep = eta_lin(p_patch, opp, k_grid)
        per.append({"I_E": I_E, "I_I": I_I, "eta_baseline": eb, "eta_patch": ep,
                    "closer_to_critical": bool(ep < eb)})
    return {"fraction_closer": float(np.mean([d["closer_to_critical"] for d in per])),
            "n_admissible": len(per), "per_point": per}
```

- [ ] **Step 4: Run → pass. Step 5: Commit** — `feat(topic4 sef-hfo step0a): low-heterogeneity screen + no-rescue guard (patch A)`

---

## Task 7: Step 0a CLI runner + figures + README

**Files:** Create `scripts/run_sef_hfo_step0a_stability.py`, `results/topic4_sef_hfo/linear_stability/figures/README.md`

- [ ] **Step 1: Runner**

```python
# scripts/run_sef_hfo_step0a_stability.py
"""Step 0a: delayed dispersion + n-convergence + low-heterogeneity screen, with the
framework Step-0a output contract (topic4_sef_itp_framework.md:810-822):
phase diagram (+candidate boundary + data-locked family), growth/k* heatmap, gain & low-het shift."""
import argparse, json
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src.sef_hfo_field import SEFParams
from src.sef_hfo_stability import (self_consistent_operating_point, eta_lin, leading_mode, gain,
                                   build_dispersion_matrix, erlang_n_convergence,
                                   screen_low_heterogeneity_effect)
OUT = Path("results/topic4_sef_hfo/linear_stability")
PHASE_CMAP = ListedColormap(["#3182bd", "#fee08b", "#de2d26"])   # stable / candidate / unstable

def phase_idx(eta, tol=1e-2):
    return 0 if eta > tol else (2 if eta < -tol else 1)

def phase_label(eta, tol=1e-2):
    return ["stable", "candidate_excitable", "unstable"][phase_idx(eta, tol)]

def plot_phase_diagram(p, family, k, out):                       # framework: phase diagram + boundary
    ie = np.linspace(0.1, 0.7, 13); ii = np.linspace(0.05, 0.30, 11)
    M = np.full((len(ii), len(ie)), np.nan)
    for a, I_I in enumerate(ii):
        for b, I_E in enumerate(ie):
            op = self_consistent_operating_point(p, I_E, I_I)
            if op.get("converged"): M[a, b] = phase_idx(eta_lin(p, op, k))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(M, origin="lower", aspect="auto", cmap=PHASE_CMAP, vmin=-0.5, vmax=2.5,
              extent=[ie[0], ie[-1], ii[0], ii[-1]])
    ax.plot([f[0] for f in family], [f[1] for f in family], "ko", ms=7)
    ax.set_xlabel("background drive I_E"); ax.set_ylabel("inhibitory drive I_I"); ax.set_title("Step 0a phase diagram")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE_CMAP(i)) for i in range(3)] + \
              [plt.Line2D([], [], marker="o", color="k", ls="")]
    ax.legend(handles, ["stable", "candidate excitable", "unstable", "data-locked family"], loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "figures" / "step0a_phase_diagram.png", dpi=140); plt.close(fig)

def plot_growth_kmap(p, op, out):                                # framework: max Re(lambda) heatmap + k*
    kk = np.linspace(-3, 3, 61); G = np.empty((len(kk), len(kk)))
    for a, kperp in enumerate(kk):
        for b, kpar in enumerate(kk):
            G[a, b] = np.linalg.eigvals(build_dispersion_matrix(p, op, float(kpar), float(kperp))).real.max()
    a0, b0 = np.unravel_index(np.argmax(G), G.shape); lim = float(np.abs(G).max())
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(G, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=[kk[0], kk[-1], kk[0], kk[-1]])
    ax.plot(kk[b0], kk[a0], "k*", ms=15, label=f"k*=({kk[b0]:.2f}, {kk[a0]:.2f})")
    ax.set_xlabel("k_parallel (along axis)"); ax.set_ylabel("k_perp (across axis)"); ax.legend(fontsize=8)
    ax.set_title("max Re(lambda) over wavevector"); fig.colorbar(im, ax=ax, label="max Re(lambda)")
    fig.tight_layout(); fig.savefig(out / "figures" / "step0a_growth_kmap.png", dpi=140); plt.close(fig)

def plot_gain_lowhet_shift(p, family, sigma_patch, k, fraction_closer, out):   # framework: gain map (+patch A)
    pp = replace(p, sigma_phi=sigma_patch); rows = []
    for I_E, I_I in family:
        ob = self_consistent_operating_point(p, I_E, I_I); opp = self_consistent_operating_point(pp, I_E, I_I)
        if not (ob.get("converged") and opp.get("converged")): continue
        rows.append((I_E, gain(ob["h_E0"], p.phi_bar, p.sigma_phi, p.beta),
                     gain(opp["h_E0"], pp.phi_bar, pp.sigma_phi, pp.beta),
                     eta_lin(p, ob, k), eta_lin(pp, opp, k)))
    x = [r[0] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x, [r[1] for r in rows], "o-", label=f"baseline (sigma_phi={p.sigma_phi})")
    ax[0].plot(x, [r[2] for r in rows], "s--", label=f"low-het patch (sigma_phi={sigma_patch})")
    ax[0].set_xlabel("I_E"); ax[0].set_ylabel("population gain G_E"); ax[0].set_title("Gain at operating point"); ax[0].legend(fontsize=8)
    ax[1].plot(x, [r[3] for r in rows], "o-", label="baseline"); ax[1].plot(x, [r[4] for r in rows], "s--", label="low-het patch")
    ax[1].axhline(0, color="k", lw=.8); ax[1].set_xlabel("I_E"); ax[1].set_ylabel("eta_lin (distance to boundary)")
    ax[1].set_title(f"Does lowering heterogeneity drop eta_lin? (not automatic) — closer at {fraction_closer:.0%} of points")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "figures" / "step0a_gain_and_lowhet_shift.png", dpi=140); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/sef_hfo_operating_points.json")
    ap.add_argument("--sigma-phi-patch", type=float, default=0.5)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    fam = [(d["I_E"], d["I_I"]) for d in cfg["operating_points"]]
    p = SEFParams(); k = np.linspace(-3, 3, 61); kc = np.linspace(-3, 3, 31)   # kc: coarser for the 2-D phase grid
    rows = []
    for I_E, I_I in fam:
        op = self_consistent_operating_point(p, I_E, I_I)
        if not op.get("converged"):
            rows.append({"I_E": I_E, "I_I": I_I, "converged": False}); continue
        eta = eta_lin(p, op, k); lm = leading_mode(p, op, k)
        rows.append({"I_E": I_E, "I_I": I_I, "converged": True, "bistable": op["bistable"],
                     "eta_lin": eta, "phase": phase_label(eta), "k_star": lm["k_star"], "omega_star": lm["omega_star"]})
    conv = erlang_n_convergence(p, self_consistent_operating_point(p, *fam[len(fam) // 2]), k)
    screen = screen_low_heterogeneity_effect(p, fam, args.sigma_phi_patch, k)
    candidates = [{"I_E": r["I_E"], "I_I": r["I_I"]} for r in rows if r.get("phase") == "candidate_excitable"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "step0a_stability.json").write_text(json.dumps(
        {"provenance": cfg["provenance"], "sigma_phi_baseline": p.sigma_phi, "sigma_phi_patch": args.sigma_phi_patch,
         "per_point": rows, "erlang_n_convergence": conv, "low_heterogeneity_screen": screen,
         "candidate_operating_points": candidates}, indent=2, default=float))
    plot_phase_diagram(p, fam, kc, OUT)
    rep = (candidates[0]["I_E"], candidates[0]["I_I"]) if candidates else fam[len(fam) // 2]
    plot_growth_kmap(p, self_consistent_operating_point(p, rep[0], rep[1]), OUT)
    plot_gain_lowhet_shift(p, fam, args.sigma_phi_patch, k, screen["fraction_closer"], OUT)
    print(f"[step0a] fraction_closer={screen['fraction_closer']:.3f}; "
          f"erlang converged={conv['converged']} rec_n={conv['recommended_n']}; candidates={len(candidates)}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run (smoke)** — `python scripts/run_sef_hfo_step0a_stability.py`; writes `step0a_stability.json` + 3 figures (`step0a_phase_diagram.png`, `step0a_growth_kmap.png`, `step0a_gain_and_lowhet_shift.png`). The phase-diagram 2-D grid is the heaviest part; for a fast smoke run lower `SEFParams.n` or the grid resolution.

- [ ] **Step 3: Figures README (Chinese)**

```markdown
<!-- results/topic4_sef_hfo/linear_stability/figures/README.md -->
# Step 0a 带延迟的线性稳定性图（对应框架 Step-0a 输出合同）

### step0a_phase_diagram.png
背景驱动二维网格（横 I_E、纵 I_I）上的相图：蓝=小扰动稳定、黄=候选可激、红=失稳；黑点是数据锁定的工作点族。
**关注点**：数据锁定的工作点落在哪个区——必须有点落在黄色"候选可激"带（间期工作窗的前提），且这条带和"失稳红区"的边界清晰。

### step0a_growth_kmap.png
一个代表性候选工作点上，最大增长率 `max Re(lambda)` 在波矢 (k∥沿轴, k⊥跨轴) 上的热图，黑星是最不稳模 `k*`。
**关注点**：最不稳模是不是出现在**有限 k**（不是 k=0），且偏向沿轴方向——这对应"沿传播轴的有限波长结构"；纯 k=0 失稳意味着全局同步而非空间传播。

### step0a_gain_and_lowhet_shift.png
左：工作点族上降低阈值异质性前后的群体增益 G_E；右：同样前后的 `eta_lin`（离边界距离），标题给出"有多大比例的点真的更靠近边界"。
**关注点**：降低异质性**不是**自动让系统更危险——只有右图虚线（低异质 patch）确实压到实线下方的那些点才算数（patch A：报比例不报存在），比例低就如实记录适用范围。

> 另：JSON 里 `erlang_n_convergence.converged` 必须为真（Erlang 延迟偏稳，n 不够会把失稳边界算得太晚）；`bistable` 标记的工作点要单独看（近临界可能多个稳态）。
```

- [ ] **Step 4: Commit** — `feat(topic4 sef-hfo step0a): dispersion + n-convergence + screen runner + figure`

---

## Task 8: Field integrator (synaptic + Erlang-delay + recovery states, mirrors the dispersion model)

**Files:** Modify `src/sef_hfo_field.py`, `tests/test_sef_hfo_field.py`

- [ ] **Step 1: Failing tests**

```python
# add to tests/test_sef_hfo_field.py
import numpy as np
from dataclasses import replace
from src.sef_hfo_field import build_kernels, make_Feff_lookup, integrate_field
from src.sef_hfo_stability import self_consistent_operating_point

def test_kernels_normalized():
    K = build_kernels(SEFParams(n=32, L=32.0))
    for name, k in K.items(): assert abs(k.sum() - 1.0) < 1e-9, name

def test_field_holds_at_fixed_point():
    p = SEFParams(n=32, L=32.0); op = self_consistent_operating_point(p, 0.4, 0.15)
    act = integrate_field(p, op, 0.4, 0.15, stim_fn=lambda t: 0.0, dt=0.05, t_max=5.0)
    assert np.max(np.abs(act[-1] - op["r_E0"])) < 1e-3

def test_recovery_lowers_steady_response_under_constant_drive():
    # WIRING test (not phenomenon): recovery's negative feedback lowers the uniform
    # steady response. Whether recovery enables self-limited propagation is 0b DATA.
    base = SEFParams(n=16, L=16.0)
    def steady(p):
        op = self_consistent_operating_point(p, 0.2, 0.1)
        return float(integrate_field(p, op, 0.2, 0.1, stim_fn=lambda t: 0.3,
                                     dt=0.05, t_max=120.0)[-1].mean())
    assert steady(replace(base, b_a=1.0, tau_a=10.0)) < steady(replace(base, b_a=0.0)) - 1e-4
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement** — field state mirrors `build_dispersion_matrix`: per connection a spatial convolution then an Erlang-delay + synaptic temporal chain (extra state grids), plus recovery.

```python
# add to src/sef_hfo_field.py
import numpy as np
from src.sef_hfo_stability import F_eff, _connection_specs, _kernel_hat, _chain_rates

def _grid(n, L):
    x = (np.arange(n) - n // 2) * (L / n); return np.meshgrid(x, x, indexing="ij")

def anisotropic_gaussian(n, L, ell_par, ell_perp, angle):
    X, Y = _grid(n, L); u = np.cos(angle)*X + np.sin(angle)*Y; v = -np.sin(angle)*X + np.cos(angle)*Y
    g = np.exp(-(u**2)/(2*ell_par**2) - (v**2)/(2*ell_perp**2)); return g/g.sum()

def isotropic_gaussian(n, L, sigma):
    X, Y = _grid(n, L); g = np.exp(-(X**2 + Y**2)/(2*sigma**2)); return g/g.sum()

def uniform_kernel(n):
    k = np.ones((n, n)); return k/k.sum()

def build_kernels(p):
    return {"EE": anisotropic_gaussian(p.n, p.L, p.ell_par, p.ell_perp, p.axis_angle),
            "EI": ((1-p.gamma_global)*isotropic_gaussian(p.n, p.L, p.sigma_I) + p.gamma_global*uniform_kernel(p.n)),
            "IE": isotropic_gaussian(p.n, p.L, p.sigma_IE),
            "II": isotropic_gaussian(p.n, p.L, p.sigma_II)}

def convolve_periodic(field, kernel):
    return np.real(np.fft.ifft2(np.fft.fft2(field) * np.fft.fft2(np.fft.ifftshift(kernel))))

def make_Feff_lookup(p, h_min=-30.0, h_max=30.0, npts=6001):
    hs = np.linspace(h_min, h_max, npts)
    return hs, np.array([F_eff(h, p.phi_bar, p.sigma_phi, p.beta) for h in hs])

def F_eff_grid(h, lookup):
    return np.interp(h, lookup[0], lookup[1])

def integrate_field(p, op, I_E, I_I, stim_fn, dt, t_max):
    """2-D E-I field. Each connection: spatial conv (kernel) -> Erlang-delay+synaptic
    temporal chain (extra state grids) -> postsynaptic drive. Mirrors build_dispersion_matrix."""
    n = p.n; rE = np.full((n, n), op["r_E0"]); rI = np.full((n, n), op["r_I0"])
    a = np.full((n, n), op["r_E0"]); K = build_kernels(p); lut = make_Feff_lookup(p)
    rate_of = {"E": rE, "I": rI}
    specs = _connection_specs(p)
    chain = {tag: [np.full((n, n), 0.0) for _ in _chain_rates(p, ts)]   # init at 0 perturbation level
             for (_, _, _, _, _, ts, tag) in specs}
    # seed chains at steady convolved input so the fixed point holds
    for (post, pre, sign, J, _, ts, tag) in specs:
        steady_in = convolve_periodic(rate_of[pre], K[tag])
        for s in chain[tag]: s[:] = steady_in
    nsteps = int(round(t_max / dt)); rec = np.empty((nsteps, n, n))
    for t in range(nsteps):
        stim = stim_fn(t * dt)
        drive_E = I_E + (stim if np.ndim(stim) else stim) - p.b_a * a
        drive_I = I_I * np.ones((n, n))
        for (post, pre, sign, J, _, ts, tag) in specs:
            inp = convolve_periodic(rate_of[pre], K[tag]); rates = _chain_rates(p, ts)
            src = inp
            for j, rj in enumerate(rates):
                chain[tag][j] = chain[tag][j] + dt * rj * (src - chain[tag][j]); src = chain[tag][j]
            out = chain[tag][-1] if rates else inp
            (drive_E if post == "E" else drive_I)[...] += sign * J * out
        rE = rE + dt/p.tau_E * (-rE + F_eff_grid(drive_E, lut))
        rI = rI + dt/p.tau_I * (-rI + F_eff_grid(drive_I, lut))
        a = a + dt/p.tau_a * (-a + rE); rate_of["E"], rate_of["I"] = rE, rI
        rec[t] = rE
    return rec
```

- [ ] **Step 4: Run → pass (3 tests). Step 5: Commit** — `feat(topic4 sef-hfo step0b): field integrator mirroring delayed dispersion (synaptic+delay+recovery)`

> **Note (advisor coupling, review #8):** the fastest timescale is now `min(tau_E, tau_AMPA, delay_d/erlang_n)`. Keep `dt` well below it (default scaffold `dt=0.05`); larger `erlang_n` requires smaller `dt`. The dt-sensitivity gate (Task 12) enforces this.

---

## Task 9: Field-linearization ↔ eigenvalue consistency + FFT-vs-analytic kernel

**Files:** Modify `tests/test_sef_hfo_field.py`

- [ ] **Step 1: Failing tests**

```python
# add to tests/test_sef_hfo_field.py
from src.sef_hfo_stability import build_dispersion_matrix, gaussian_hat
from src.sef_hfo_field import anisotropic_gaussian

def test_fft_kernel_matches_analytic_hat():
    # discrete FFT of the real-space kernel must match the analytic gaussian_hat at grid k
    p = SEFParams(n=64, L=64.0); g = anisotropic_gaussian(p.n, p.L, p.ell_par, p.ell_perp, 0.0)
    ghat = np.fft.fft2(np.fft.ifftshift(g)).real
    kx = 2*np.pi*np.fft.fftfreq(p.n, d=p.L/p.n)
    m = 3; analytic = gaussian_hat(kx[m], 0.0, p.ell_par, p.ell_perp)
    assert abs(ghat[m, 0] - analytic) < 1e-2

def test_field_linearization_matches_dispersion_eigenvalue():
    # DECISIVE consistency (advisor): seed a small mode-k perturbation, measure the field's
    # growth rate, confirm it matches build_dispersion_matrix's max Re eigenvalue.
    p = SEFParams(n=64, L=64.0); op = self_consistent_operating_point(p, 0.4, 0.15)
    n, L = p.n, p.L; x = (np.arange(n) - n//2) * (L/n)
    m = 2; kpar = 2*np.pi*m/L
    eps = 1e-4
    def stim_fn(t): return 0.0
    base = integrate_field(p, op, 0.4, 0.15, stim_fn, dt=0.02, t_max=6.0)
    # perturb initial E by eps*cos(kpar x); measure growth of that Fourier component
    pert = eps*np.cos(kpar*x)[:, None]*np.ones((1, n))
    act = integrate_field_with_ic(p, op, 0.4, 0.15, stim_fn, dt=0.02, t_max=6.0, dE0=pert)
    amp = np.abs(np.fft.fft2(act - op["r_E0"])[:, m, 0])
    t = np.arange(len(amp))*0.02; lo, hi = len(amp)//4, len(amp)//2
    rate_meas = np.polyfit(t[lo:hi], np.log(amp[lo:hi] + 1e-30), 1)[0]
    rate_pred = float(np.linalg.eigvals(build_dispersion_matrix(p, op, kpar, 0.0)).real.max())
    assert abs(rate_meas - rate_pred) < 0.05
```

- [ ] **Step 2: Run → fail** (needs `integrate_field_with_ic`).

- [ ] **Step 3: Implement** — add an initial-condition variant (factor the loop body of `integrate_field` so both share it; here add the thin wrapper):

```python
# add to src/sef_hfo_field.py  (refactor integrate_field to accept optional dE0=None and add:)
def integrate_field_with_ic(p, op, I_E, I_I, stim_fn, dt, t_max, dE0):
    return integrate_field(p, op, I_E, I_I, stim_fn, dt, t_max, dE0=dE0)
```
(Modify `integrate_field` signature to `(..., t_max, dE0=None)` and after seeding `rE`, do `if dE0 is not None: rE = rE + dE0`.)

- [ ] **Step 4: Run → pass. Step 5: Commit** — `feat(topic4 sef-hfo step0b): field-linearization ↔ dispersion eigenvalue consistency (review #9, advisor gate)`

---

## Task 10: Wavefront-aware response classifier (+ global-mode class)

**Files:** Create `src/sef_hfo_pulse.py`, `tests/test_sef_hfo_pulse.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_sef_hfo_pulse.py
import numpy as np
from src.sef_hfo_pulse import classify_response

def _act(extents, centroids, n=16):
    """Build (T,n,n) E-activity with prescribed active-fraction and centroid-x per frame."""
    T = len(extents); a = np.zeros((T, n, n)); cells = n*n
    for t, (frac, cx) in enumerate(zip(extents, centroids)):
        k = int(round(frac*cells)); c0 = int(cx) % n
        a[t, :, c0:min(n, c0 + max(1, k//n + 1))] = 1.0
    return a

def test_classifier_five_regimes():
    n = 16; stim = np.zeros((n, n)); stim[:, 7:9] = 1.0; rest = 0.0
    kw = dict(stim_mask=stim, rest_level=rest, runaway_frac=0.5, return_frac=0.2, detect=0.5)
    extinction = _act([0.06, 0.06, 0.0, 0.0], [8, 8, 8, 8])
    local_bump = _act([0.06]*6, [8]*6)
    self_lim   = _act([0.06, 0.18, 0.28, 0.0], [8, 10, 12, 12])     # extent grows AND centroid moves, returns
    runaway    = _act([0.06, 0.25, 0.55, 0.80], [8, 9, 9, 9])
    flash      = _act([0.06, 0.40, 0.40, 0.0], [8, 8, 8, 8])         # extent grows but centroid does NOT move
    assert classify_response(extinction, **kw) == "extinction"
    assert classify_response(local_bump, **kw) == "local_bump"
    assert classify_response(self_lim,   **kw) == "self_limited_propagation"
    assert classify_response(runaway,    **kw) == "runaway"
    assert classify_response(flash,      **kw) == "global_synchronous"   # NOT propagation
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement** — a propagating event must have a *moving compact front* (centroid displacement), not merely growing extent (which a synchronous flash also has):

```python
# src/sef_hfo_pulse.py
"""SEF-HFO Step-0b finite-pulse response: wavefront-aware classifier, adaptive thresholds."""
import numpy as np

def _centroid_x(frame, rest, detect):
    above = frame > (rest + detect)
    if not above.any(): return np.nan
    xs = np.where(above)[1]; return float(xs.mean())

def classify_response(activity, stim_mask, rest_level, runaway_frac=0.5, return_frac=0.2,
                      detect=0.05, travel_cells=1.0):
    above = activity > (rest_level + detect)
    extent = above.reshape(above.shape[0], -1).mean(axis=1)
    stim_extent = float(stim_mask.mean()); max_extent = float(extent.max()); final_extent = float(extent[-1])
    cx = np.array([_centroid_x(f, rest_level, detect) for f in activity])
    cx0 = next((c for c in cx if not np.isnan(c)), np.nan)
    moved = np.nanmax(np.abs(cx - cx0)) if not np.isnan(cx0) else 0.0
    grew = max_extent > 1.5 * stim_extent
    returned = final_extent <= return_frac * max(max_extent, 1e-12)
    if max_extent >= runaway_frac:
        return "runaway"
    if grew and moved < travel_cells:
        return "global_synchronous"          # extent grew but front did NOT travel -> not propagation
    if grew and moved >= travel_cells and not returned:
        return "runaway"
    if grew and moved >= travel_cells and returned:
        return "self_limited_propagation"
    if returned:
        return "extinction"
    return "local_bump"
```

- [ ] **Step 4: Run → pass (5 regimes). Step 5: Commit** — `feat(topic4 sef-hfo step0b): wavefront-aware classifier with global-synchronous class (review #10)`

---

## Task 11: Pulse driver (homogeneous-patch approx) + adaptive amplitude thresholds

**Files:** Modify `src/sef_hfo_pulse.py`, `tests/test_sef_hfo_pulse.py`

- [ ] **Step 1: Failing test**

```python
# add to tests/test_sef_hfo_pulse.py
from src.sef_hfo_pulse import amplitude_thresholds

def test_adaptive_amplitude_thresholds_separates_self_limited():
    # gate quantity is A_self_limited (NOT A_event): local_bump / global_synchronous
    # must NOT count toward the safety margin. Here: extinction<0.8; local_bump[0.8,1.0);
    # self-limited[1.0,2.0); runaway>=2.0.
    def fake(A):
        if A < 0.8: return "extinction"
        if A < 1.0: return "local_bump"
        if A < 2.0: return "self_limited_propagation"
        return "runaway"
    out = amplitude_thresholds(fake, a_lo=0.2, a_hi=3.0, n_coarse=12, n_refine=6)
    assert abs(out["A_self_limited"] - 1.0) < 0.15
    assert abs(out["A_runaway"] - 2.0) < 0.15
    assert out["A_event"] < out["A_self_limited"]            # local_bump detected earlier, separate
    assert abs(out["safety_margin"] - (out["A_runaway"] - out["A_self_limited"])) < 1e-9
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement** — single stimulation location (homogeneous-patch approximation; position/axis discrimination is Step 3), radii × durations family, coarse-then-refine amplitude search (review #1, #6):

```python
# add to src/sef_hfo_pulse.py
from src.sef_hfo_field import build_kernels, make_Feff_lookup, integrate_field, _grid

DETECTABLE = ("local_bump", "self_limited_propagation", "global_synchronous", "runaway")
PULSE_FAMILY = {"radii": (3.0, 6.0), "durations": (1.0, 3.0)}   # single location (patch-approx)

def _disk_mask(p, r):
    X, Y = _grid(p.n, p.L); return ((X**2 + Y**2) <= r**2).astype(float)

def run_pulse(p, op, I_E, I_I, r, T, A, dt, t_max, return_activity=False, **cls_kw):
    mask = _disk_mask(p, r)
    def stim_fn(t): return (A * mask) if t < T else 0.0 * mask
    act = integrate_field(p, op, I_E, I_I, stim_fn, dt, t_max)
    label = classify_response(act, mask, op["r_E0"], **cls_kw)
    return (label, act) if return_activity else label

def _first_with(label_fn, a_grid, predicate):
    for A in sorted(a_grid):
        if predicate(label_fn(A)):
            return float(A)
    return np.inf

def amplitude_thresholds(label_fn, a_lo, a_hi, n_coarse=12, n_refine=6):
    """REAL coarse->refine: label_fn(A) MUST run a fresh pulse at A (no nearest-label
    lookup). Separately resolve A_event (first detectable), A_self_limited (first
    'self_limited_propagation' — the gate quantity), A_runaway (first 'runaway').
    safety_margin = A_runaway - A_self_limited (NOT A_event): local_bump / global
    synchronous must not count toward the margin (review #3)."""
    coarse = list(np.linspace(a_lo, a_hi, n_coarse))
    def resolve(predicate):
        a_coarse = _first_with(label_fn, coarse, predicate)
        if not np.isfinite(a_coarse):
            return np.inf
        lo = max(a_lo, a_coarse - (a_hi - a_lo) / n_coarse)
        fine = np.linspace(lo, a_coarse, n_refine)        # endpoint a_coarse already satisfies
        return _first_with(label_fn, fine, predicate)     # real pulses at fine amplitudes
    A_event = resolve(lambda L: L in DETECTABLE)
    A_self = resolve(lambda L: L == "self_limited_propagation")
    A_run = resolve(lambda L: L == "runaway")
    return {"A_event": A_event, "A_self_limited": A_self, "A_runaway": A_run,
            "safety_margin": A_run - A_self}
```

- [ ] **Step 4: Run → pass. Step 5: Commit** — `feat(topic4 sef-hfo step0b): pulse driver (patch-approx, single location) + adaptive thresholds (review #1,#6)`

---

## Task 12: Step 0b CLI — scan 0a candidate family, recovery OFF+ON (report both), L/grid+dt sensitivity

**Files:** Create `scripts/run_sef_hfo_step0b_pulse.py`, `results/topic4_sef_hfo/finite_pulse/figures/README.md`

- [ ] **Step 1: Runner** (consumes `step0a_stability.json` candidate operating points — review #3; reports both recovery settings — review #4; L/grid + dt sensitivity — review #6,#8):

```python
# scripts/run_sef_hfo_step0b_pulse.py
"""Step 0b: finite-pulse response map over 0a's CANDIDATE operating points, recovery OFF
and ON (both reported, no auto-select), L/grid + dt sensitivity, plus the framework
Step-0b output contract: full response surface + margin waterfall + example snapshots."""
import argparse, json
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src.sef_hfo_field import SEFParams
from src.sef_hfo_stability import self_consistent_operating_point
from src.sef_hfo_pulse import PULSE_FAMILY, run_pulse, amplitude_thresholds, _centroid_x
OUT = Path("results/topic4_sef_hfo/finite_pulse")
A_LO, A_HI, A_SURF = 0.2, 3.0, 12
CLASSES = ["extinction", "local_bump", "self_limited_propagation", "global_synchronous", "runaway"]
CLS_CMAP = ListedColormap(["#dddddd", "#9ecae1", "#31a354", "#fdae6b", "#de2d26"])

def _memo_label_fn(p, op, I_E, I_I, r, T, dt, t_max):
    """A REAL pulse runner with a cache (fixes the nearest-label hack): every distinct A
    runs a fresh simulation; repeats are reused. amplitude_thresholds' refine therefore
    runs genuine pulses at fine amplitudes."""
    cache = {}
    def fn(A):
        key = round(float(A), 4)
        if key not in cache:
            cache[key] = run_pulse(p, op, I_E, I_I, r, T, A, dt, t_max)
        return cache[key]
    return fn

def scan_point(p, I_E, I_I, dt, t_max):
    op = self_consistent_operating_point(p, I_E, I_I)
    if not op.get("converged"): return None
    a_surf = list(np.linspace(A_LO, A_HI, A_SURF))
    cells = []
    for r in PULSE_FAMILY["radii"]:
        for T in PULSE_FAMILY["durations"]:
            lf = _memo_label_fn(p, op, I_E, I_I, r, T, dt, t_max)
            grid = [lf(A) for A in a_surf]                       # response-surface row (real sims)
            thr = amplitude_thresholds(lf, A_LO, A_HI)           # real refine reuses cache
            has_win = (np.isfinite(thr["A_self_limited"]) and np.isfinite(thr["safety_margin"])
                       and thr["safety_margin"] > 0)
            cells.append({"r": r, "T": T, **thr, "grid": grid, "has_self_limited_window": bool(has_win)})
    return {"I_E": I_E, "I_I": I_I, "op": op, "a_surf": a_surf, "cells": cells}

def family_summary(p, family, dt, t_max):
    pts = [sp for (I_E, I_I) in family if (sp := scan_point(p, I_E, I_I, dt, t_max)) is not None]
    for sp in pts:
        sp["has_window"] = any(c["has_self_limited_window"] for c in sp["cells"])
    n = len(pts); k = sum(sp["has_window"] for sp in pts)
    return {"n_candidates": n, "n_with_window": k, "fraction_with_window": (k / n if n else 0.0), "points": pts}

def _find_point(summ, I_E, I_I):
    return next((sp for sp in summ["points"] if sp["I_E"] == I_E and sp["I_I"] == I_I), None)

def plot_response_surface(off, on, out):
    rep = next((sp for sp in off["points"] if sp["has_window"]), off["points"][0] if off["points"] else None)
    if rep is None: return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (name, summ) in zip(axes, [("recovery OFF", off), ("recovery ON", on)]):
        sp = _find_point(summ, rep["I_E"], rep["I_I"])
        if sp is None: continue
        mat = np.array([[CLASSES.index(l) for l in c["grid"]] for c in sp["cells"]])
        ax.imshow(mat, aspect="auto", cmap=CLS_CMAP, vmin=-0.5, vmax=4.5,
                  extent=[A_LO, A_HI, len(sp["cells"]) - 0.5, -0.5])
        ax.set_yticks(range(len(sp["cells"])))
        ax.set_yticklabels([f"r={c['r']:.0f} T={c['T']:.0f}" for c in sp["cells"]])
        ax.set_xlabel("pulse amplitude"); ax.set_title(f"{name}  (operating point I_E={rep['I_E']:.2f})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLS_CMAP(i)) for i in range(5)]
    fig.legend(handles, [c.replace("_", " ") for c in CLASSES], loc="lower center", ncol=5, fontsize=8)
    fig.suptitle("Step 0b finite-pulse response surface (representative candidate point)")
    fig.tight_layout(rect=[0, 0.07, 1, 1]); fig.savefig(out / "figures" / "step0b_response_surface.png", dpi=140)
    plt.close(fig)

def _best_cell(sp):
    fin = [c for c in sp["cells"] if np.isfinite(c["safety_margin"])]
    pos = [c for c in fin if c["safety_margin"] > 0]
    return (max(pos, key=lambda c: c["safety_margin"]) if pos
            else (max(fin, key=lambda c: c["safety_margin"]) if fin else sp["cells"][0]))

def plot_margin_waterfall(off, on, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (name, summ) in zip(axes, [("recovery OFF", off), ("recovery ON", on)]):
        pts = summ["points"]; x = np.arange(len(pts))
        a_sl = [c["A_self_limited"] if np.isfinite(c["A_self_limited"]) else np.nan
                for c in (_best_cell(sp) for sp in pts)]
        a_ru = [c["A_runaway"] if np.isfinite(c["A_runaway"]) else np.nan
                for c in (_best_cell(sp) for sp in pts)]
        ax.vlines(x, a_sl, a_ru, color="0.75", lw=6)             # safety-margin band
        ax.plot(x, a_sl, "o", color="#31a354", label="A_self_limited")
        ax.plot(x, a_ru, "s", color="#de2d26", label="A_runaway")
        ax.set_xticks(x); ax.set_xticklabels([f"{sp['I_E']:.2f}" for sp in pts], rotation=90, fontsize=7)
        ax.set_xlabel("candidate operating point (I_E)"); ax.set_title(name)
    axes[0].set_ylabel("pulse amplitude"); axes[0].legend(fontsize=8)
    fig.suptitle("Step 0b safety margin per candidate point  (band = A_runaway - A_self_limited)")
    fig.tight_layout(); fig.savefig(out / "figures" / "step0b_margin_waterfall.png", dpi=140); plt.close(fig)

def _find_example(summ, label):
    for sp in summ["points"]:
        for c in sp["cells"]:
            for A, lbl in zip(sp["a_surf"], c["grid"]):
                if lbl == label:
                    return (sp["I_E"], sp["I_I"], sp["op"], c["r"], c["T"], A)
    return None

def plot_example_snapshots(p, off, dt, t_max, out):
    rows = [(lbl, _find_example(off, lbl)) for lbl in
            ["self_limited_propagation", "global_synchronous", "runaway"]]
    rows = [(lbl, ex) for lbl, ex in rows if ex is not None]
    if not rows: return
    nt = 4; fig, axes = plt.subplots(len(rows), nt, figsize=(3 * nt, 3 * len(rows)))
    axes = np.atleast_2d(axes)
    for i, (lbl, (I_E, I_I, op, r, T, A)) in enumerate(rows):
        _, act = run_pulse(p, op, I_E, I_I, r, T, A, dt, t_max, return_activity=True)
        idxs = np.linspace(0, len(act) - 1, nt).astype(int)
        for j, ti in enumerate(idxs):
            axes[i, j].imshow(act[ti], cmap="magma")
            cx = _centroid_x(act[ti], op["r_E0"], 0.05)
            if not np.isnan(cx): axes[i, j].axvline(cx, color="cyan", lw=1.2)   # centroid -> moves if propagating
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([]); axes[i, j].set_title(f"t={ti*dt:.0f}", fontsize=8)
        axes[i, 0].set_ylabel(lbl.replace("_", " "), fontsize=8)
    fig.suptitle("Step 0b example responses: snapshots + centroid (cyan) — propagation moves, flash does not")
    fig.tight_layout(); fig.savefig(out / "figures" / "step0b_example_snapshots.png", dpi=130); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0a", default="results/topic4_sef_hfo/linear_stability/step0a_stability.json")
    ap.add_argument("--dt", type=float, default=0.05); ap.add_argument("--t-max", type=float, default=40.0)
    ap.add_argument("--tau-a", type=float, default=15.0)
    args = ap.parse_args()
    a0 = json.loads(Path(args.stage0a).read_text())
    family = [(d["I_E"], d["I_I"]) for d in a0.get("candidate_operating_points", [])] \
             or [(d["I_E"], d["I_I"]) for d in a0["per_point"] if d.get("converged")]
    base = SEFParams()
    off = family_summary(replace(base, b_a=0.0), family, args.dt, args.t_max)
    on = family_summary(replace(base, b_a=1.0, tau_a=args.tau_a), family, args.dt, args.t_max)
    sens = {"half_dt": family_summary(replace(base, b_a=0.0), family, args.dt / 2, args.t_max)["fraction_with_window"],
            "smaller_L": family_summary(replace(base, b_a=0.0, n=48, L=48.0), family, args.dt, args.t_max)["fraction_with_window"]}
    OUT.mkdir(parents=True, exist_ok=True)
    def _strip(s):   # drop bulky 'op' before JSON (keep 'grid' for provenance of the surface)
        return {**s, "points": [{k: v for k, v in sp.items() if k != "op"} for sp in s["points"]]}
    (OUT / "step0b_pulse.json").write_text(json.dumps(
        {"family_size": len(family), "runs": {"recovery_off": _strip(off), "recovery_on": _strip(on)},
         "sensitivity": sens, "recovery_decision": "REPORT_BOTH_no_auto_select"}, indent=2, default=float))
    plot_response_surface(off, on, OUT)
    plot_margin_waterfall(off, on, OUT)
    plot_example_snapshots(replace(base, b_a=0.0), off, args.dt, args.t_max, OUT)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["off", "on"], [off["fraction_with_window"], on["fraction_with_window"]]); ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of candidate points with self-limited window (+margin)")
    ax.set_title("Step 0b: recovery OFF vs ON (report both)")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "step0b_window_fraction.png", dpi=140); plt.close(fig)
    print(f"[step0b] off={off['fraction_with_window']:.2f} on={on['fraction_with_window']:.2f} sens={sens}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run (smoke)** — `python scripts/run_sef_hfo_step0b_pulse.py --t-max 25`; writes `step0b_pulse.json` + 4 figures (`step0b_response_surface.png`, `step0b_margin_waterfall.png`, `step0b_example_snapshots.png`, `step0b_window_fraction.png`). **Compute-heavy** (every amplitude is a real field integration); for smoke, run on a small candidate family / small `SEFParams.n`.

- [ ] **Step 3: Figures README (Chinese)**

```markdown
<!-- results/topic4_sef_hfo/finite_pulse/figures/README.md -->
# Step 0b 有限脉冲响应图（局部均匀斑块近似）

> 本步是空间均匀近似、单一刺激位置——中心/轴端/离轴的区分留到 Step 3 加空间 patch 后。所有幅度都是真实仿真（不是用粗扫标签近似）。

### step0b_response_surface.png
代表性候选工作点上的**完整响应面**：横轴脉冲幅度、纵轴脉冲（半径×时长）组合、颜色是五类响应（熄灭/局部鼓包/自限传播/全局同步/失控），左右分"纯抑制"和"加恢复变量"。
**关注点**：绿色（自限传播）有没有形成一条夹在"局部鼓包"和"失控"之间的带——这条带存在、且上方"失控"出现得明显更晚，才说明有真正的间期工作窗；不能只挑一个好看的脉冲。

### step0b_margin_waterfall.png
每个候选工作点的 `A_self_limited`（刚出现自限传播的幅度）、`A_runaway`（刚失控的幅度）和两者之间的安全余量带，左右 off/on。
**关注点**：余量带为正且不太窄的工作点有几个——这是 go/no-go 的直接依据；注意余量是 `A_runaway − A_self_limited`，**不含**局部鼓包/全局同步。

### step0b_example_snapshots.png
各挑一个自限传播 / 全局同步 / 失控的例子，画时间快照，青色竖线是激活质心位置。
**关注点**：自限传播那行的青线应**逐帧移动**（波前在走），全局同步那行青线基本**不动**（只是原地一起亮）——这就是把"传播"和"同步闪光"肉眼分开的判据。

### step0b_window_fraction.png
候选工作点里"存在自限窗+正余量"的**比例**，off / on **并列**（不自动选）。
**关注点**：off / on 哪套比例更高，且 JSON 里 `sensitivity`（半 dt、小 L）是否一致——不一致说明是数值/边界假象，不能信。
```

- [ ] **Step 4: Commit** — `feat(topic4 sef-hfo step0b): pulse map over 0a candidate family + recovery report-both + sensitivity (review #3,#4,#6,#8)`

---

## Task 13: Go/no-go gate — structured-provenance formal tier + units anchoring + window margin

**Files:** Create `tests/test_sef_hfo_step0_gate.py`, `docs/archive/topic4/sef_itp_phase4_v2/step0_results_2026-06-02.md`

- [ ] **Step 1: Gate tests**

```python
# tests/test_sef_hfo_step0_gate.py
"""Step 0 go/no-go gate. SMOKE tier: structural checks on scaffold runs.
FORMAL tier (unlocks Step 1): requires data-locked provenance + anchored units."""
import json
from pathlib import Path
A = Path("results/topic4_sef_hfo/linear_stability/step0a_stability.json")
B = Path("results/topic4_sef_hfo/finite_pulse/step0b_pulse.json")

def test_smoke_screen_is_fraction_and_n_converged():
    d = json.loads(A.read_text())
    assert 0.0 <= d["low_heterogeneity_screen"]["fraction_closer"] <= 1.0
    assert d["erlang_n_convergence"]["converged"], "Erlang delay n not converged -> dispersion unreliable"

def test_smoke_step0b_sensitivity_stable():
    d = json.loads(B.read_text())
    off = d["runs"]["recovery_off"]["fraction_with_window"]
    assert abs(off - d["sensitivity"]["half_dt"]) <= 0.2, "dt-sensitive -> numerical, not physical"
    assert abs(off - d["sensitivity"]["smaller_L"]) <= 0.2, "L-sensitive -> boundary wrap artifact"

def test_formal_gate_requires_data_locked_provenance_and_window():
    """FORMAL tier: skip on scaffold runs; enforce on data-locked runs."""
    d = json.loads(A.read_text()); prov = d["provenance"]
    if prov.get("source") != "data_locked":
        import pytest; pytest.skip("scaffold run: NOT a formal Step-1-unlock pass")
    assert prov["locked_before_sweep"] is True and prov["hash"]
    b = json.loads(B.read_text())
    assert max(b["runs"]["recovery_off"]["fraction_with_window"],
               b["runs"]["recovery_on"]["fraction_with_window"]) > 0, \
        "no self-limited window with positive margin -> Step 1 stays LOCKED (a finding)"
```

- [ ] **Step 2: Run** — `python scripts/run_sef_hfo_step0a_stability.py && python scripts/run_sef_hfo_step0b_pulse.py --t-max 25 && pytest tests/test_sef_hfo_step0_gate.py -v`. Smoke tests pass/inform; formal test SKIPS on scaffold. **If `fraction_with_window`==0, that's a valid finding — Step 1 does not start.**

- [ ] **Step 3: Results writeup (plain-language-led; fill empirical lines after running)** — `docs/archive/topic4/sef_itp_phase4_v2/step0_results_2026-06-02.md`:

```markdown
# SEF-HFO Step 0 结果 + go/no-go 闸门（2026-06-02，带延迟版）

## 测了什么
跑大网络前先问两件事：(1) 把一块组织的神经元阈值变整齐（降低异质性），在一族用真实数据框定的背景工作点上，有多大**比例**真的更靠近"一点就着"的边界（不是找一个能用的点）；(2) 给一束有限脉冲，活动是"传一段又自己熄灭"，还是点不着 / 一点就失控 / 只是全局同步闪一下。这一版把**突触快慢和传导延迟**也放进了稳定性计算。

## 怎么测的
0a：每个工作点解自洽稳态，把突触动力学和延迟当成附加线性状态，算带延迟的色散，得到 `eta_lin` 和最不稳模的波数/频率；并验证 Erlang 近似的延迟级数 n 已经收敛。0b：对 0a 的候选工作点逐点打脉冲，要求活动是**会移动的紧凑波前**（不是原地长大的同步闪光）才算传播，并要求失控幅度明显高于点出事件的幅度。recovery 关/开两套并列报，不自动选。

## 揭示了什么
[实跑后填：低异质性筛选在候选族里约 …% 成立；候选族里约 …% 存在自限窗+正余量；recovery off/on 各…；很多还是 global_synchronous 还是 self_limited……；闸门 PASS 解锁 Step 1 / 或没过则如实记录适用范围、不抢救。]

## 与 Brunel 2026 的关系
方法（自洽 2×2 色散、外驱相图、各向异性 E→E 选轴、|I^E|+|I^I| LFP 代理）与 Bachschmid-Romano/Hatsopoulos/Brunel 2026 一致；区别是他们的事件是近全局 Turing–Hopf 行波，我们要的是局部自限瞬态——所以 0b 的 recovery off/on 正好对应"亚临界近 Hopf"与"恢复变量局部脉冲"两条机制。

（内部归档代号：Step 0a/0b、delayed dispersion、erlang_n、eta_lin、operating-point family、fraction_closer、finite-pulse、A_event/A_runaway/safety_margin、global_synchronous、recovery off/on、Brunel 2026）
```

- [ ] **Step 4: Commit** — `feat(topic4 sef-hfo step0): go/no-go gate (smoke vs data-locked formal) + units anchoring + writeup`

---

## Self-Review

**Spec coverage:** delayed dispersion (Tasks 4–5) ✓; patch A screen+no-rescue (Task 6) ✓; patch B switchable recovery + report-both (Tasks 8,12) ✓; finite-pulse window+margin gate (Tasks 11–13) ✓. Review fixes #1 (Task 11 single location + README) #2 (Task 13 provenance tiers) #3 (Task 12 family scan) #4 (Task 12 report-both) #5 (Task 13 units anchoring) #6 (Tasks 11,12 adaptive+sensitivity) #7 (Task 3) #8 (Task 8 note + Task 12 dt-sens) #9 (Task 9) #10 (Task 10) — all mapped. Advisor gates: n-convergence (Task 5), bounded transcendental (Task 5), field-linearization consistency (Task 9), n↔dt (Task 8 note + Task 12).

**Visualization + gate-correctness round (2026-06-02):** framework Step-0a/0b output contract (`topic4_sef_itp_framework.md:810-822`) now met — 0a emits phase diagram (+candidate boundary + data-locked family), growth/`k*` heatmap, gain & low-het shift (Task 7); 0b emits full response surface, margin waterfall, example snapshots with centroid trail (Task 12), per your Brunel-style argument chain (structure → phase → dispersion → simulation example → proxy). Two gate-polluting bugs fixed: (a) amplitude refinement now runs **real pulses** at fine amplitudes via a cached real label_fn (no nearest-coarse-label lookup) — Tasks 11–12; (b) `A_self_limited` separated from `A_event`, and **`safety_margin = A_runaway − A_self_limited`** so local-bump / global-synchronous do not inflate the margin — Tasks 11,13. Figures follow the paper-grade self-contained discipline (legends, plain-term axes, no codebase jargon).

**Placeholder scan:** only the `[实跑后填…]` empirical line in the writeup (data the run produces). No code placeholders.

**Type consistency:** `SEFParams` fields, `self_consistent_operating_point` keys (`r_E0/r_I0/h_E0/h_I0/converged/bistable/n_distinct_roots`), `build_dispersion_matrix` (shared by stability + field via `_connection_specs/_chain_rates/_kernel_hat`), classifier labels (`extinction/local_bump/self_limited_propagation/global_synchronous/runaway`), `amplitude_thresholds` keys — consistent across tasks.

**Known scope cut (logged, review #5):** scaffold timescales are dimensionless toys; the *formal* gate (Task 13) requires data-anchored units (synaptic/membrane τ from Brunel Table 1; `tau_a` from measured HFO group-event duration; rates in Hz) before any result enters a topic doc.

---

## Execution Handoff

**Plan saved to `docs/superpowers/plans/2026-06-02-topic4-sef-hfo-step0-stability-pulse-plan.md`. Two options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks.
**2. Inline Execution** — executing-plans, batch with checkpoints.

**Which approach?**
