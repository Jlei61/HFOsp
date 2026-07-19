# MZ slow–fast dynamical transition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline). Steps use `- [ ]`.

**Goal:** Build a new-file analysis that freezes the natural MZ slow state `{z_i, m_i}` at registered
checkpoints and evolves only the fast spiking system, measuring perturbation-free escape probability
`P_runaway`, global nonlinear ignition threshold `ε_c`, and recovery time `τ_rec`, plus state-matched M/Z
counterfactuals, then classifies the transition result-neutrally and produces a 4-panel figure + archive.

**Architecture:** Reuse `MZOnsetProbe` / `run_loop` (checkpoint/resume) / `score_runaway` /
`epsilon_c_from_ladder` from `src/topic4_mz_onset_dynamics.py` (no engine edits). New pure functions in
`src/topic4_mz_slow_fast_transition.py`; new runner + plotter + config + tests + results tree.

**Tech Stack:** Python, numpy, scipy, matplotlib, PyYAML, multiprocessing (Pool), pytest.

## Global Constraints

- Binding design: `docs/superpowers/specs/2026-07-20-topic4-mz-slow-fast-transition-design.md`.
- Seeds 1,3,4. `dt=0.1`. η_m=0.0074516, I_th_EI=95.19851312666987, tau_z=5000. Onsets O_s: 9293.6/9499.3/9757.9.
- Runaway = 120 Hz / 100 ms (`score_runaway`). Operational runaway only; never "seizure".
- `OMP/MKL/OPENBLAS_NUM_THREADS=1`. Workers memory-gated `min(nproc−2, floor((avail−max(30GB,25%))/peakRSS))`.
- No engine edits (6 guarded files), no push/merge, no `git add -A`. Atomic per-job writes; separate aggregate.
- Results root `results/topic4_sef_hfo/mz_slow_fast_transition/`. Topic-named files, no PR numbers in dir names.

---

### Task 1: config + module/runner/test skeletons

**Files:** Create `config/topic4_mz_slow_fast_transition.yaml`, `src/topic4_mz_slow_fast_transition.py`
(imports + `SCHEMA_VERSION="mz-slow-fast-transition-1.0"`), `tests/test_topic4_mz_slow_fast_transition.py`,
`scripts/run_topic4_mz_slow_fast_transition.py` (CLI stub with `pilot/run/aggregate` subparsers, `--confirm-run`
gate), `scripts/plot_topic4_mz_slow_fast_transition.py` (stub).

- [ ] Config carries: `conditions` (4, exact cfg from design §1), `seeds`, `onsets`, checkpoints (design §2.1),
  `matched_d_targets:[0.02,0.04,0.06,0.08]`, `p_runaway:{n_replay:20,horizon_ms:500}`,
  `ignition:{probe_ms:10,amplitude_ladder:[0,0.025,0.05,0.10,0.20],bisection:2,runaway_hz:120,runaway_dur_ms:100}`,
  `recovery:{amp:0.02,pulse_ms:10,horizon_ms:500,band_k:1.0,pre_window_ms:200}`, `counterfactual` (design §4),
  `natural_tail_ms:1500`, `runaway_hz/dur`.
- [ ] Module imports `MZOnsetProbe, run_loop, LoopState, score_runaway, epsilon_c_from_ladder` from
  `src.topic4_mz_onset_dynamics` (path-insert `src/snn_engine` mirror already in that module).
- [ ] Commit `feat(topic4-mzsf): scaffolding — config + module/runner/plot/test skeletons`.

---

### Task 2: `branch_rng_state` + `wilson_ci` (pure, TDD)

**Files:** Modify `src/topic4_mz_slow_fast_transition.py`, `tests/test_topic4_mz_slow_fast_transition.py`.

**Produces:**
- `branch_rng_state(seed:int, cond:str, state:str, idx:int) -> dict` — PCG64 `bit_generator.state` for an
  independent future-noise branch, deterministic in inputs.
- `wilson_ci(k:int, n:int, z:float=1.96) -> tuple[float,float]`.

- [ ] **Step 1 (test):**
```python
import numpy as np
from src.topic4_mz_slow_fast_transition import branch_rng_state, wilson_ci

def test_branch_rng_state_deterministic_and_independent():
    a = branch_rng_state(1, "mz_runaway", "pre_onset_100ms", 0)
    a2 = branch_rng_state(1, "mz_runaway", "pre_onset_100ms", 0)
    b = branch_rng_state(1, "mz_runaway", "pre_onset_100ms", 1)
    assert a == a2                                   # deterministic
    assert a != b                                    # distinct branch idx -> distinct stream
    g = np.random.default_rng(0); g.bit_generator.state = a   # swappable into PCG64
    x = g.standard_normal(5)
    g2 = np.random.default_rng(0); g2.bit_generator.state = b
    assert not np.allclose(x, g2.standard_normal(5))

def test_wilson_ci_bounds():
    lo, hi = wilson_ci(0, 20); assert 0.0 <= lo <= hi <= 1.0 and lo == 0.0 or lo >= 0.0
    lo, hi = wilson_ci(20, 20); assert hi <= 1.0
    lo1, hi1 = wilson_ci(5, 20); lo2, hi2 = wilson_ci(15, 20)
    assert lo2 > lo1                                  # monotone in k
```
- [ ] **Step 2:** run → FAIL (import error). **Step 3:** implement:
```python
import numpy as np

def branch_rng_state(seed, cond, state, idx):
    ss = np.random.SeedSequence([int(seed), abs(hash(cond)) % (2**31), abs(hash(state)) % (2**31), int(idx)])
    return np.random.default_rng(np.random.PCG64(ss)).bit_generator.state

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))
```
Note: `hash()` of str is salted per-process — replace with a stable hash (`hashlib.sha256(cond.encode())`) so
`branch_rng_state` is reproducible ACROSS processes (workers). Use `int.from_bytes(sha256(...)[:4])`.
- [ ] **Step 4:** run → PASS. **Step 5:** commit `feat(topic4-mzsf): branch_rng_state + wilson_ci`.

---

### Task 3: `recovery_time` (pure, TDD)

**Produces:** `recovery_time(rate_hz, dt, pulse_off_idx, band_lo, band_hi, *, smooth_ms=20.0, min_hold_ms=50.0)
-> float|None` — first time (ms after pulse offset) the 20-ms-EMA rate re-enters `[band_lo,band_hi]` and stays
for `min_hold_ms`; None if never.

- [ ] **Step 1 (test):** decaying-exponential trace returns finite ~expected; flat-elevated (never returns) →
  None; already-in-band at pulse_off → ~0.
```python
def test_recovery_time_returns_finite_for_decay():
    dt = 0.1; n = 6000; t = np.arange(n) * dt
    rate = 5.0 + 20.0 * np.exp(-(t) / 50.0)           # decays toward 5 Hz
    rt = recovery_time(rate, dt, pulse_off_idx=0, band_lo=4.0, band_hi=6.0, min_hold_ms=50.0)
    assert rt is not None and 80.0 < rt < 400.0

def test_recovery_time_censored_when_never_returns():
    rate = np.full(3000, 40.0)
    assert recovery_time(rate, 0.1, 0, band_lo=4.0, band_hi=6.0) is None
```
- [ ] Steps 2–4 (fail→implement EMA + sustained-in-band scan→pass). **Step 5:** commit.

---

### Task 4: `state_step_schedule`, `matched_d_times`, `classify_transition` (pure, TDD)

**Produces:**
- `state_step_schedule(onset_ms, dt) -> dict[str,int]` (design §2.1 eight keys, `first_crossing` filled later).
- `matched_d_times(D_trace, t_ms, targets) -> dict[float, float|None]` (first t where D≥target).
- `classify_transition(per_state:list[dict], *, natural_crosses:bool, plateau_outside:bool) -> dict`
  → `{"label":..., "features":{...}}` with 5 labels per design §5.

- [ ] **Step 1 (tests):** one synthetic `per_state` per label (steep P step + ε_c→0 + τ_rec↑ → `dynamical_tipping`;
  low P but ε_c↓ → `finite_amplitude_escape`; smooth P rise → `noise_driven_escape`; flat everything →
  `smooth_crossover`; empty/all-censored → `unresolved`), assert label + non-empty features.
- [ ] Steps 2–4. Classifier is transparent thresholds over `[D, P_runaway, eps_c, tau_rec]` (document each
  threshold inline). **Step 5:** commit `feat(topic4-mzsf): schedules + result-neutral transition classifier`.

---

### Task 5: runner core — one `(condition, seed)` job

**Files:** Modify `scripts/run_topic4_mz_slow_fast_transition.py`.

**Interfaces / Produces:** `run_unit(cond_id, seed, cfg, out_root, resume) -> path` that writes
`per_state/<cond>_seed<S>_<state>.json`, `per_state/<cond>_seed<S>_natural.npz`,
`counterfactual/<cond>_seed<S>.json`, `matched_d/<cond>_seed<S>.json`.

Sub-steps (each its own `- [ ]`, integration glue — reuse primitives, no engine edits):
- [ ] `build_S(seed)` via `run_m4_phaseplane.build_substrate`; `mzcfg(cond_id)` from config.
- [ ] **Natural trajectory, chained capture:** loop the registered checkpoint steps ascending; for each segment
  `run_loop(start=prev_ck, n_steps=Δ, capture_final=True, store_spikes=True)` with a fresh un-frozen
  `MZOnsetProbe`; keep each `ck` + the running `rate_E`, `slow.trace_z_mean`, `slow.trace_adap_current`. After
  the last pre-onset checkpoint, continue to `O_s + natural_tail_ms` (or condition's own crossing via
  `early_stop_runaway`) to fill `first_crossing`. Save natural npz (t, D, a, rate, crossing_ms, event marks).
  Cross-check D_max/crossing against existing `results/.../mz_onset_dynamics/per_seed/` npz (sanity log).
- [ ] **Per checkpoint** (from its captured `ck`): (a) `P_runaway`: 20 forks, each a `copy.deepcopy(ck)` with
  `rng_state = branch_rng_state(...)`, `run_loop(start=fork_ck, n_steps=horizon, early_stop_runaway=True)`,
  `score_runaway`; count → P + `wilson_ci`. (b) `ε_c`: global `set_probe(target_E=all_E, delta=a·gap)` ladder
  + bisection (native noise `start=ck`), `epsilon_c_from_ladder`. (c) `τ_rec`: one subthreshold global probe,
  `recovery_time` on the returned rate. Write per-state JSON atomically (`.tmp`→`os.replace`).
- [ ] **Counterfactuals** at `pre_onset_100ms` ck: for each of 5 (z,m) settings, overwrite copied `ck.slow.z[:NE]`
  / `.m[:NE]` from the `pre_100` and `mid_fraction` snapshots, run P_runaway + ε_c + τ_rec. Write JSON.
- [ ] **Matched-D:** from natural D-trace get `matched_d_times`; at each reached target re-capture a ck (resume
  chain to that step) and run P_runaway + ε_c. Write JSON.
- [ ] `--resume` skips any per-state JSON that already exists. Commit `feat(topic4-mzsf): per-(cond,seed) runner`.

---

### Task 6: pilot subcommand + resource measurement

- [ ] `pilot`: run ONE (cond=mz_plateau, seed=1) at ONE checkpoint (`pre_onset_100ms`) with `n_replay=4`;
  wrap in `resource.getrusage`/`tracemalloc` sampling; print `peak_rss_gb`, `wall_s`, `wall_per_1e5_steps`.
  Also emit a projected full-run estimate. Commit `feat(topic4-mzsf): pilot + resource probe`.

---

### Task 7: parallel driver + aggregate

- [ ] `run --all --workers W`: build the 12 `(cond,seed)` unit list; `multiprocessing.Pool(W)` (spawn), each
  worker sets threads=1 and calls `run_unit`; per-unit fail-loud (re-raise with unit tag). Default `W` computed
  from live `psutil.virtual_memory().available` and measured `peak_rss` (fallback: `--workers` explicit).
- [ ] `aggregate` (no sim): read all per-state/counterfactual/matched_d JSON → `slow_fast_transition_summary.csv`
  (one row per cond,seed,state) + `.json` (per-condition `classify_transition`), `STATUS.md`, `provenance.json`
  (git sha, engine SHAs asserted == baseline, config/module hashes). Commit `feat(topic4-mzsf): driver + aggregate`.

---

### Task 8: 4-panel figure + README

- [ ] Read `docs/figure_style_guide.md` Topic 4 section; reuse palette/fonts from
  `plot_topic4_mz_onset_dynamics.py` + `plot_topic4_mz_onset_tau.py`. Fixed colors runaway/edge/plateau/z-only.
- [ ] `plot_topic4_mz_slow_fast_transition.py`: Panel A (onset-aligned R_E/D/a), B (P_runaway vs D + CI +
  crossing), C (ε_c + τ_rec vs state, censoring marks), D (counterfactual P_runaway/ε_c). PNG + PDF.
- [ ] Visual check (render, eyeball, fix). Write Chinese `figures/README.md` (per-panel argument + 关注点).
  Commit `feat(topic4-mzsf): four-panel figure + README`.

---

### Task 9: archive doc + verification + final commits

- [ ] `docs/archive/topic4/sef_hfo/mz_slow_fast_transition_2026-07-20.md` — plain-language (§8) verdict:
  测了什么 / 怎么测的 / 揭示了什么 + per-condition classification + resource peak + provenance + non-goals.
- [ ] `pytest -q tests/test_topic4_mz_slow_fast_transition.py` green; `git diff --check` clean;
  engine SHAs unchanged. Logical commits; worktree clean. No push/merge.

---

## Self-Review

- **Spec coverage:** design §1 conditions→T1/T5; §2 checkpoints→T4/T5; §3 probes→T2/T3/T5; §4 counterfactuals→T5;
  §5 classifier→T4; §6 reuse→T1/T2/T5; §7 files→all; §8 resources→T6/T7; §9 figure→T8; §10 tests→T2–T4/T5; §11
  non-goals→respected (no spatial code). Covered.
- **Placeholder scan:** pure-function tasks carry full code; integration tasks carry exact reuse calls + file
  paths. No TBD.
- **Type consistency:** `branch_rng_state`→dict swapped into `LoopState.rng_state`; `wilson_ci`→tuple;
  `recovery_time`→float|None; `classify_transition`→dict{label,features}. Consistent across T5 usage.
- **Fix noted:** `branch_rng_state` must use a stable hash (hashlib), not builtin `hash()` (per-process salt).
