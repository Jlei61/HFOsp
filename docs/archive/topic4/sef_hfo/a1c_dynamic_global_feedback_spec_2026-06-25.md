# A1c **dynamic global feedback RESTRAINT screen** — spec（workflow 产出 + 用户审查修订 2026-06-25）

> 来源：8-agent design workflow（3 设计 lens → judge 合成 → 3 skeptic 对抗 → finalize）。
> 设计 alternatives + 对抗发现见同目录 a1c_design_provenance_2026-06-25.md。

## ★ P1 用户审查修订（2026-06-25，覆盖下文相应处，实现以此为准）

- **P1-1 命名/结论纪律（最重要）**：A1c = **dynamic global feedback RESTRAINT screen**，**不是**抑制性耗竭机制。
  Abbott 的真耗竭是 Cl⁻/z 这类"使用依赖抑制变弱"（见 `docs/paper/abbott_model.md:438`），那是 **A2**（z/e_GABA 动态耗竭，
  见 `m3a_a1b_state_topography_2026-06-25.md:65`）。**允许结论**："global feedback CAN / CANNOT quench the MEASURED
  runaway corner under this substrate"。**禁止结论**："inhibitory exhaustion mechanism validated"。下文 §0/§5 凡说
  "Abbott 机制成立"一律按此降级。
- **P1-2 pilot 必须含 `l1_g1.0` preservation control**：A1c 成功 = runaway 变成"点着后回落" **且**已有的 working
  seizure-like 态（l1_g1.0）**不被压死**。behavior pilot（§4.1）除 `l2_g1.0` 外，**必须加** `l1_g1.0 × gain{0,2,8} ×
  tau{150}（稳则+2000）× 3 seeds`。full grid 保留 preservation control。
- **P1-3 magnitude calibration 必须出实际电流读数**：pilot 强制 `--dump-fb`（或 `--dump-drive`）同时写 **E 细胞上
  I_I、I_E 的 median/p95 + core/surround 分层 + `I_global / median(I_I_on_E)` 比值**。pilot report **先报这个 ratio**，
  再决定是否重标 gain ladder（feedback_gain 单位极易偏 10×–1000×；全 silent / 完全无效可能只是单位错，不是机制结果）。
- **P1-4 动态 vs 静态再加一个对照**：`I_global leads decay`（§5.4）是必要但不充分（事件可能本就自衰减、EMA 只是跟着升）。
  pilot 找到候选 terminating rung 后，**再补一个 matched-static / frozen-feedback 对照**（同量级的恒定 I_global）。
  完整对照做好前，结论只能写 **"consistent with dynamic feedback"**，**不得**写 "ruled out static brake"。
- **P1-5 TDD 实现前补齐**：补 T2（三类 caller parity）、T8（assert_versions）、T6 加 `slow != None` 互斥、T5 改成
  **确定性 recompute**（从 rate_E 重算期望 EMA·gain，逐步比对 dump 的 I_global_trace，证因果延迟+EMA 数学，非靠随机有无 spike）。

---

I now have all load-bearing facts confirmed against the code. Key confirmations:

- Runner call site is **line 589**, `slow=slow` is threaded — A1b runaway corners run `slow=None` (A1b sets `_a1b_on` via weight lesion, not slow layer). So injecting in the default `membrane_step` branch (line 246) is sufficient.
- `--global-ei-scale` (default 1.0, line 431) is A1b's **static** counterpart — it scales GABA input to every E target via `a1b_weight_lesion` (line 552). A1c is its dynamic, rate-driven counterpart. This gives the calibration anchor Skeptic 3 demanded.
- `returned` is **per-event-peak-relative** (line 228) — Skeptic 2 Hole 1 confirmed.
- `BASELINE_MS = (5.0, 50.0)` (line 60), `event_bar_mode` default `record_peak` (line 453), `--dump-fullfield` gates `r95`/`seizure_like` (line 665).
- Engine guard is whole-file sha256 (`f7676b78…`); any in-file edit forces re-bless — Skeptic 1 Hole 1 confirmed.
- 38 files reference `simulate_kick`; default kwargs `feedback_gain=0.0` keep them bit-parity.

Here is the final vetted spec.

---

# A1c — DYNAMIC GLOBAL FEEDBACK INHIBITION — FINAL IMPLEMENTATION SPEC (design only; execute step by step)

## 0. What A1c is, in one plain paragraph (for the human before coding)

**测了什么**：A1b 已经画出一张"静态地图"——把一个固定强度的、不分位置的总体抑制（`--global-ei-scale`，对每个兴奋细胞的 GABA 输入统一加权）从弱调到强，看网络是"刹不住地跑飞（runaway）/ 中间出现像发作的大同步事件但还能回落 / 太强到全程静默"。A1c 换一个机制：抑制不再是固定旋钮，而是**网络自己的总体放电率经过一个低通滤波后，反过来按比例压住所有兴奋细胞**（Abbott/Liou-Schevon 2020 的动态全局反馈抑制：放电越凶、刹车越大，且这刹车看的是全网平均、不分位置）。

**怎么测的**：取 A1b 里**有完整读数、确实测到的那个跑飞角**（`l2_g1.0`，core 放电 414 Hz，return≈0.02，5 个种子全是 runaway），把动态反馈打开，逐级加大反馈增益。如果 Abbott 的预测对，应该出现一个**中间增益窗**：事件还能点着（不是被压死），但被反馈在事件进行中追上并**主动终止**，而不是无限跑飞。

**揭示了什么**：不是 "PASS/FAIL"，而是"在这个 readout 和这些量级下，**动态全局反馈能 / 不能**把一个静态会跑飞的角落变成能自我终止的发作样事件"——而且要能跟"加个静态常数刹车也能做到"区分开（动态判据：反馈信号在率下降之前先升起来）。

---

## 1. EXACT ENGINE HOOK

### 1.1 Mechanism (the only new equation)

A per-step scalar running estimate of the global E firing **rate** (Hz, intensive, NE-invariant) drives a uniform inhibitory current added onto **E cells only**:

```
# state (init once, before the loop):
r_ema = 0.0                                  # filtered global E rate proxy (Hz)
alpha_fb = 1.0 - exp(-dt / feedback_tau_ms)  # EXACT low-pass coeff (engine convention, NOT dt/tau)

# top of step t, BEFORE membrane_step (consume the PREVIOUS step's count -> one-step causal delay):
I_global = feedback_gain * r_ema             # scalar >= 0 (r_ema >= 0 since counts >= 0)
I_fb = I_global on E cells, 0 on I cells

# membrane update for E cells uses an effective inhibition I_I + I_fb:
#   I_net = I_E - (I_I + I_fb)   for E cells   (I cells: I_fb=0 -> unchanged)

# at the EXISTING recorder line (after spikes are computed for step t):
rate_E[t] = spk[:NE].sum()                   # (existing line, unchanged)
r_inst_hz = rate_E[t] / NE / (dt * 1e-3)     # count -> Hz  (matches the runner's /NE/DT*1e3 readout)
r_ema += alpha_fb * (r_inst_hz - r_ema)      # EMA update, consumed at TOP of step t+1
```

**Why each clause (folding skeptic fixes):**
- **EMA on Hz, not count** (Skeptic 3 Hole 3): `feedback_gain` multiplies an intensive Hz proxy. A raw count would make `gain` silently scale with `NE ∝ density·L²` and break any future L change. `r_inst_hz = count/NE/(dt·1e-3)` is exactly the runner's Hz convention (`/NE/DT*1e3`), so `r_ema` is directly comparable to the A1b `global_E_rate_mean_hz` status numbers.
- **`alpha = 1 − exp(−dt/τ)`, NOT `dt/τ`** (Skeptic 2 Hole 4): copies the engine's own verbatim convention (`ee_std_recover_factor` line 64, `decay_sE` line 120, `ou_a` line 149). `dt/τ` is only the first-order Taylor approx; using the exact form means the τ axis label is the real time constant at every τ rung.
- **Consume-at-top / update-at-recorder** (Skeptic 2 Hole 3): `I_global` at step t is a function of spikes at steps ≤ t−1 only. This is a true one-step causal delay (same discipline as the delay ring). It forbids the acausal within-step loop where I_global at t would depend on spk at t.
- **E-only injection** (Abbott contract): `I_fb` uses the existing `is_E = labels==0` mask (line 156, already in scope). One scalar broadcast to ALL E cells — never core-restricted. The *driving signal* `r_ema` is total-E-count-derived (core-dominated at a runaway by construction) — this is correct Abbott (I_global responds to total network rate); document it so a reviewer doesn't read core-dominance of the *signal* as a spatial leak (Skeptic 2 Hole 7).

### 1.2 Injection point + off-by-default gate (bit-parity)

The hook is gated by `fb_on = feedback_gain > 0.0`, mirroring the M1 `ee_std_on = ee_std_u > 0.0` template (line 174) **verbatim in spirit**: at `fb_on=False`, **no state init, no per-step EMA, no I_fb array, no change to the membrane call** — the default path is the literal pre-edit code.

Concretely, inside `simulate_kick`:

```python
# new kwargs (defaults => bit-parity):  feedback_gain=0.0, feedback_tau_ms=0.0

# --- A1c init (gated; mirrors M1 line 174-178: no alloc/no RNG/no float touch when off) ---
fb_on = feedback_gain > 0.0
if fb_on:
    assert feedback_tau_ms > 0.0, "feedback_gain>0 requires feedback_tau_ms>0"
    assert slow is None, "A1c rides the default current-based membrane_step; slow must be None"
    assert not shunt_gaba, "A1c rides the default current-based membrane_step; shunt_gaba must be False"
    r_ema = 0.0
    alpha_fb = float(1.0 - np.exp(-dt / feedback_tau_ms))
    NE_f = float(NE); inv_dt_ms = 1.0 / (dt * 1e-3)
```

In the membrane block (around line 243-247), keep the existing `slow is not None` and default branches untouched; add the A1c case **inside the else (slow is None) branch only**, gated on `fb_on`:

```python
else:
    if fb_on:
        I_fb = np.where(is_E, feedback_gain * r_ema, 0.0)        # scalar*0 column on I cells
        Vtmp = membrane_step(V, I_E, I_I + I_fb, decay_V,
                             shunt_gaba=shunt_gaba, e_gaba=e_gaba, g_gaba_scale=g_gaba_scale)
    else:
        Vtmp = membrane_step(V, I_E, I_I, decay_V,                # <-- LITERAL pre-edit call
                             shunt_gaba=shunt_gaba, e_gaba=e_gaba, g_gaba_scale=g_gaba_scale)
```

Then at the recorder (after line 257), gated:

```python
if fb_on:
    r_ema += alpha_fb * (rate_E[t] / NE_f * inv_dt_ms - r_ema)
```

**Parity guarantees (folding Skeptic 1 Holes 1-2, Skeptic 2 Hole 2):**
- The `fb_on=False` path is textually the pre-edit `membrane_step(V, I_E, I_I, …)` call — it does **not** route through `I_I + I_fb`. No new array allocation, no new float op, **no new `rng` draw anywhere on the gain=0 path** (the OU `xi` and Poisson streams stay in lockstep — this is the real parity killer if violated).
- "Bit-parity ⇒ no re-bless" is **FALSE** for an in-file edit: the guard is whole-file sha256 over `kick_probe.py` (current `f7676b78…`). ANY character added to the file changes the hash and every M1/M2/M3/A1b/A1c run refuses to start until re-blessed. So "bit-parity" here means a **numeric-output obligation** (gain=0 produces byte-identical traces), proven by a regression test (§2), which **gates** the re-bless.

### 1.3 Re-bless step (ordered, mandatory)

1. Make the edit (new kwargs + gated block).
2. Run the §2 numeric-parity regression — must pass `np.array_equal` on continuous + count recorders AND identical `rng.bit_generator.state`.
3. **Only then** recompute sha256 of `kick_probe.py` and overwrite **only** the `"src/snn_engine/kick_probe.py"` key in `results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json` (current `f7676b78…` → new).
4. Commit engine edit + re-bless in one commit with the regression test green.

---

## 2. TDD — TESTS TO WRITE FIRST (before any engine edit)

Write these against a tiny network (small L / few steps) for speed; the parity tests are the gate.

**T1 — gain=0 bit-parity, continuous recorder (THE re-bless gate).** Build one net, fixed seed. Run `simulate_kick(..., feedback_gain=0.0)` (new default) and compare against a **frozen pre-edit baseline** captured before the edit (pickle the `lfp_trace` + `E_spk_bool` + `rate_E` from the current engine). Assert:
- `np.array_equal(lfp_trace_new, lfp_trace_baseline)` — continuous (`V`/`I`-derived) recorder, NOT just `rate_E`. Two membrane trajectories can share an integer count per bin while differing sub-threshold (Skeptic 1 Hole 2b); the continuous LFP catches that.
- `np.array_equal(E_spk_bool_new, E_spk_bool_baseline)` and `np.array_equal(rate_E_new, rate_E_baseline)`.
- **RNG-stream invariant**: capture `net["rng"].bit_generator.state` after a gain=0 run vs after a pre-edit run from the same seed — must be identical (proves zero added `rng` draws). (Skeptic 2 Hole 2.)

**T2 — gain=0 across representative callers.** Repeat T1 for (a) one M1 cell (`ee_std_u>0`), (b) one A1b cell (`global_ei_scale≠1.0`, slow=None), (c) one plain cm-spontaneous cell. Each at `feedback_gain=0.0` must be byte-identical to its pre-edit output. This proves the 38 existing call sites stay bit-parity via the new default kwargs.

**T3 — `feedback_tau_ms=0.0` with `feedback_gain=0.0` does NOT divide by zero.** Assert the short-circuit is gated on **gain**, never on tau (default `feedback_tau_ms=0.0` would NaN `1−exp(−dt/0)`). Run default kwargs; assert no NaN/inf in outputs. (Skeptic 3 Hole 6.)

**T4 — EMA coefficient correctness.** Unit-test the EMA math directly: feed a known constant rate `r*` for many steps into the update; assert `r_ema → r*` and that the time to reach `(1−1/e)·r*` equals `feedback_tau_ms` within one `dt` (validates `1−exp(−dt/τ)`, not `dt/τ`). (Skeptic 2 Hole 4.)

**T5 — causal delay (no within-step loop).** Construct a 2-step toy where step t's spikes are nonzero only at t=k; assert `I_global` first becomes nonzero at step k+1, never at step k. (Skeptic 2 Hole 3.)

**T6 — mutual-exclusion asserts fire.** Assert `simulate_kick(feedback_gain=0.5, slow=<non-None>)` raises; same for `shunt_gaba=True`; same for `feedback_tau_ms=0` with `gain>0`. (Skeptic 2 Hole 2 / engine integrity.)

**T7 — monotone braking sanity (small, fast).** On a tiny excitable net at a gain that produces a visible effect, assert mean E rate is monotonically **non-increasing** in `feedback_gain` over {0, small, large} (the brake brakes). Pure direction sanity, not a science verdict.

**T8 — runner regression: `assert_versions` passes after re-bless.** After re-blessing engine_versions.json, the runner's startup `assert_versions` (line 73) must not raise.

---

## 3. RUNNER CLI KNOBS (`scripts/run_sef_hfo_snn_cm_spontaneous_readout.py`)

Add two args + a fail-fast guard + thread into the single `simulate_kick` call at line 589.

```python
ap.add_argument("--feedback-gain", type=float, default=0.0,
                help="A1c dynamic global feedback inhibition gain (0=off=bit-parity). "
                     "I_global = gain * EMA(global E rate Hz), injected on E cells only.")
ap.add_argument("--feedback-tau-ms", type=float, default=0.0,
                help="A1c EMA low-pass time constant (ms); required >0 if --feedback-gain>0.")
ap.add_argument("--dump-fb", action="store_true",
                help="A1c: dump per-1ms-binned I_global + global E rate trace (readout-only).")
```

Fail-fast at parse (before network build), folding mutual-exclusion:
```python
if a.feedback_gain > 0.0:
    if a.feedback_tau_ms <= 0.0:
        raise SystemExit("--feedback-gain>0 requires --feedback-tau-ms>0")
    if a.slow_var != "none" or a.shunt_gaba:
        raise SystemExit("A1c (--feedback-gain) is incompatible with --slow-var / --shunt-gaba "
                         "(it rides the default current-based membrane path)")
```

Thread into the line-589 call:
```python
res = simulate_kick(p, net, KICK_BOOST=0.0, ..., dump_drive=a.dump_drive,
                    feedback_gain=a.feedback_gain, feedback_tau_ms=a.feedback_tau_ms)
```

Record both into the readout JSON config block (grid provenance), parallel to how `global_ei_scale` is recorded (line 843).

**Mandatory grid flags** (fold Skeptic 3 Hole 4): the A1c grid driver MUST pass `--dump-fullfield` (so `r95`/`seizure_like` are reachable) and `--event-bar-mode prefix_peak` — BUT see §5 for why the **primary** termination read is raw-rate, not the bar.

---

## 4. CONCRETE SWEEP GRID (within L20 budget)

Substrate (fixed): Stage-3 `twoend_equal` core, `L=20 density=100 theta=45`, `core-mean=17.5 core-std=1.0 core-r=1.5 sep-frac=0.7 drive=0.6`. `T=8000ms`. Runner `slow=None`, `shunt_gaba=False`.

**A1b anchor cells (re-anchored per Skeptic 1 Hole 1 / Skeptic 3 Holes 1-2):**

| cell | A1b state | n_seeds in A1b | role in A1c |
|---|---|---|---|
| **`l2_g1.0`** | **runaway (coreR=414, ret=0.02)** | **5 (MEASURED)** | **PRIMARY anchor** — measured runaway with full readouts to contrast against |
| `l1_g1.0` | seizure_like (3/5 sz, ret=0.90) | 5 | **seizure-PRESERVATION control** — A1c must NOT abolish this working return-event regime |
| `l1_g0.7` | runaway (timeout, n_seeds=0) | 0 (placeholder) | SECONDARY only — global-runaway type; must be re-run at gain=0 with completion to get a trace |

> Why drop g0.7 as primary: those cells have **no readout JSON** (timeout placeholder via `setdefault`, analyze line 111-114). "Control = runaway" there is a tautology (timeout always labels runaway) and termination would be indistinguishable from "the run stopped hanging." Use the **measured** `l2_g1.0` so the gain=0 baseline is a real classifiable trace whose `coreR`/`ret`/E-rate you can show shrinking. `l2_g1.0` is **core-concentrated** runaway (coreR=414 vs global=26) — a global-feedback signal may struggle to see it; that is itself an Abbott-relevant measurement (local:global ratio), so **stratify the verdict by runaway type** (§5) and keep `l1_g0.7` as the global-mean-driven secondary.

**Feedback gain** (master knob, log-spaced): `{0.0 (parity ref), 0.5, 1, 2, 4, 8}` — but calibrate magnitude first (§4.1).

**Feedback tau** — **two regimes, both required** (Skeptic 2 Hole 3 / Skeptic 3 Hole 7): a τ ≫ event width can't terminate a sub-second runaway and a NULL there is a τ-artifact, not Abbott-falsification.
- **within-event regime**: `{50, 150}` ms (straddles the ~tens-of-ms event/kick timescale, `DUR_KICK=18`, `SETTLE_MS=50`).
- **inter-event regime**: `{2000}` ms (Abbott seconds-scale; sets inter-event spacing, brakes the *next* event).

**Seeds**: `≥5` at the primary anchor + preservation control (match A1b's 5-seed hot cells; the twoend_equal substrate is the documented finite-size-fragile one). 3 seeds only for the secondary g0.7 sanity.

### 4.1 PILOT-FIRST HARD STOP (do this before any full grid)

**Magnitude calibration pilot** (Skeptic 3 Hole 3): before sweeping, find a `feedback_gain` where `I_global` at the runaway E rate is comparable to the static `global_ei_scale` increment that A1b already shows converts seizure_like→interictal (the 1.0→1.3 step). Run `l2_g1.0` at `tau=150`, a 3-gain coarse ladder `{0, 2, 8}`, seed 1, with `--dump-fb`. Eyeball:
- (a) `gain=0` reproduces measured runaway (coreR high, E-rate pinned/rising to T_end);
- (b) `I_global` trace rises into a current magnitude on the order of the inhibitory currents (not 1000× too weak/strong — if it is, rescale the gain ladder);
- (c) some gain bends the E-rate down.

**Behavior pilot**: `l2_g1.0` × `{gain 0, 2, 8}` × `{tau 150, 2000}` × 3 seeds = 18 runs. Eyeball the §5 figures:
- gain=0 = runaway (positive trace, not a missing file);
- some intermediate gain at the **within-event** tau self-terminates while E-bar is high;
- the inter-event tau (2000) brakes spacing but may NOT terminate the single runaway excursion — confirm via the I_global trace whether the filter even rises within the event.

**STOP and report to human after pilot.** Only on a clear (a)+(b)+(c) signal proceed to the full grid. If compute tight, drop the secondary g0.7 cells first, then the `gain=0.5` rung.

**Full grid size** (after pilot GO): {l2_g1.0 primary, l1_g1.0 preservation} × {gain 0,0.5,1,2,4,8} × {tau 50,150,2000} × 5 seeds = 360 runs; + {l1_g0.7 secondary} × {gain 0,2,8} × {tau 150,2000} × 3 = 36. ~396 L20 runs. If over budget: drop tau=50 (keep 150 as the within-event rung) → 264 + 36.

---

## 5. FALSIFIABLE TERMINATION TEST (thresholds + gain=0 control)

**Tier**: MECHANISM-SCREEN go/no-go on the Abbott termination prediction. Report as *"dynamic global feedback CAN / CANNOT terminate the static-runaway corner at intermediate gain, under this readout and these magnitudes"* — **never** "proves Abbott." This is screen-grade, not a formal verdict (per the prior SNN discipline: screen ≠ verdict).

### 5.1 The gain=0 control (per anchor, per seed)

Run the same corner at `feedback_gain=0.0` to **completion** (the loop is `for t in range(nsteps)` — it never infinite-hangs; "timeout" is wall-clock only, so run it under a wall-clock cap large enough to finish T=8000 at L20). Record the **affirmative runaway signature** (Skeptic 3 Hole 5): `rate_E[last 500ms].mean() / pre_event_baseline` (baseline = `compute_metrics` window, reuse `BASELINE_MS`), `coreR`, full `rate_E` trace showing it stays elevated to T_end. The control is a **measured** runaway, not a missing-file label.

### 5.2 ABSOLUTE termination gate (the central anti-artifact, Skeptic 2 Hole 1)

`return_to_baseline_fraction` / `returned` is **per-event-peak-relative** (`post.min() ≤ RETURN_FRAC·peak`, line 228) — it reads `returned=True` for a clamped-but-elevated **plateau** (event overshoots a high new fixed point). That cannot distinguish Abbott-termination from Abbott-failure-into-sustained-seizure. Therefore the PASS gate uses an **absolute** tail-to-baseline ratio:

- Define `tail_to_baseline_ratio = rate_E[last sustained window].mean() / baseline_abs`, baseline_abs from the gain=0-matched pre-event `BASELINE_MS` window.
- **Hard pre-condition**: `tail_to_baseline_ratio ≤ 1.5` sustained for ≥500ms to T_end. Report the raw ratio alongside the discrete label always.
- `T_terminate` = last time `rate_E` drops below `1.5·baseline_abs` and STAYS (sustained window ≥500ms or ≥3·tau_membrane); `NaN` if never returns.

### 5.3 Discrete-label PASS (reuse `_state` verbatim, but never alone)

Use the existing `_state` classifier verbatim (no threshold drift). But guard its two failure modes:
- **`silent` conflates "terminated a real seizure" with "clamped, never ignited"** (Skeptic 1 Hole 6). Require `active_E_fraction_peak` in the seizure band AND `gr ≥ 0.3` at SOME point — i.e. the event provably ignited — before crediting "termination." A run that never crossed the event bar at any gain is labeled `suppressed (no ignition)`, distinct from `terminated`.
- **`seizure_like` needs `r95>8` which needs `--dump-fullfield`** (Skeptic 3 Hole 4). Mandate the flag so the label is reachable; require the terminating run to be `seizure_like` (large event that returns), not `interictal_like` (a shrunk axial blip = over-suppression, not Abbott-termination).
- **`collision_rate` is defined only over RETURNED events** (Skeptic 3 Hole 5) → it does not exist on the runaway side → it CANNOT be a termination discriminator. Demote to descriptive sidecar only.

### 5.4 DYNAMIC-vs-STATIC discriminator (the thing only a dynamic brake can show)

A constant DC brake also makes a runaway finite. To claim **dynamic** termination (Skeptic 3 Hole 7 / Skeptic 1 Hole 5):
- **`I_global` must LEAD the rate decay**: cross-correlation of `I_global(t)` against `−d(rate_E)/dt` shows I_global rising **before** the E-rate starts falling. A slow-tau bystander EMA rises *after* the peak (on the event's own decay) — that fails the lead test and is NOT credited. Replace the weak "rises during the event" with "rises *before* the decay."

### 5.5 PASS / FALSIFIED / INCONCLUSIVE (sign-consistent across ALL seeds)

**PASS** requires, sign-consistent across all seeds at fixed tau:
1. gain=0 control is **measured** runaway (affirmative signature §5.1; NOT a timeout placeholder; NOT `setdefault`);
2. some **intermediate** gain flips to `seizure_like` with `tail_to_baseline_ratio ≤ 1.5` AND finite `T_terminate` AND the event provably ignited (`gr≥0.3`, `active_E_fraction_peak` in seizure band) — i.e. not silent/suppressed;
3. **I_global leads the rate decay** (§5.4) — the dynamic signature a static brake can't show;
4. highest gain → `silent` **only if reached *through* an ignite-then-terminate rung** as gain increases (§5.3); if gain jumps runaway→silent with NO intervening self-terminating cell, that is FALSIFIED, not a monotone dose-response.

**FALSIFIED** if: runaway jumps straight to silent with no self-terminating window; OR control isn't a measured runaway (corner mis-specified); OR I_global does not lead the decay at any terminating gain (looks static/bystander).

**INCONCLUSIVE** if: termination is seed-fragile (e.g. 3/5); OR only the inter-event tau "terminates" while the I_global trace shows the filter never rose within the event (τ-artifact / the substrate self-terminated and the slow filter was a bystander); OR the core-concentrated `l2_g1.0` runaway is unreachable by the global signal while the global-driven `l1_g0.7` is — report **stratified by runaway type**: "global-mean-driven runaway [terminates/doesn't]; core-concentrated runaway [terminates/doesn't]."

**Anti-rule** (Skeptic 1 Hole 5): a run **completing within budget is NOT termination evidence**. Completion-vs-timeout is a compute artifact. Termination requires the measured `rate_E` absolute-tail gate (§5.2), full stop.

### 5.6 Separate `completed` / `timeout` from `runaway` (Skeptic 2 Hole 4)

Record per run: `completed: bool`, `n_events_truncated`, and tag `runaway` only if (completed-and-tail-elevated) OR (timed-out with `rate_E[-200ms:].mean() > RETURN_FRAC·event_peak`, i.e. still rising at the cut). A run that terminated its event and was then cut off during quiescence is **not** runaway (this `setdefault`-style conflation biases the whole test toward a false NULL).

---

## 6. NEW READOUTS

Written to `results/topic4_sef_hfo/m3a_slowvars/a1c_grid/` (parallel to `a1b_grid/`), plus `status_a1c.json` (parallel to `status_a1b.json`), `figures/` + Chinese `figures/README.md` per repo standard.

Per-run JSON (in addition to the existing A1b activity readouts):
- `feedback_gain`, `feedback_tau_ms`, `alpha_fb` (provenance);
- `I_global_trace` + `global_E_rate_hz_trace` per-1ms bin (readout-only, `--dump-fb`, empty/absent at gain=0);
- `tail_to_baseline_ratio` (absolute, §5.2) — the central anti-plateau number;
- `T_terminate` (NaN if never), `baseline_abs`, `peak_E_rate`, `core_E_rate_mean_hz`, `global_E_rate_mean_hz`;
- `I_global_leads_decay` (cross-corr lead sign, §5.4) + lead lag value;
- `self_terminated` flag = `T_terminate` finite AND `tail_to_baseline_ratio ≤ 1.5` AND gain=0 control of SAME corner+seed was elevated-to-T_end AND event ignited;
- `ignited` flag (`active_E_fraction_peak` in seizure band) — to separate `suppressed (no ignition)` from `terminated`;
- `completed`, `n_events_truncated` (§5.6);
- raw `_state` label + ALL raw metrics alongside (so a boundary artifact can't manufacture a discrete PASS).

Figures (each one independent question, §7 figure discipline):
1. **A1c state surface**: gain × tau heatmap of `_state` for the primary anchor (gain=0 column = A1b baseline) + the preservation-control cell as a second panel showing it is NOT abolished. (Question: where in (gain,tau) does runaway become self-terminating, and does the working seizure regime survive?)
2. **Allow-then-quench money figure**: overlay `rate_E(t)` + `I_global(t)` for one terminating cell, with `T_terminate` marked and the I_global-leads-decay lag annotated. (Question: does the brake rise before the rate falls — dynamic, not static?)

Cross-check termination from the **raw `rate_E` sustained-below-baseline** (bar-independent), NOT from the bar-gated `events` list — the `prefix_peak` bar can be polluted by a spontaneous event in the calibration window or by the EMA's charging transient (Skeptic 2 Hole 6 / Skeptic 1 Hole 7). Make raw-rate the PRIMARY termination criterion; the active-fraction `events`/`seizure_like` label is secondary confirmation. Where possible calibrate the event bar from the **gain=0 control of the same corner** (`fixed_bar` with the control's bar) so braked and unbraked events sit on one feedback-independent scale.

---

## 7. ADVERSARIAL HOLES — FOLD-IN CHECKLIST (every guard, mapped)

| # | Hole | Guard in this spec |
|---|---|---|
| S1-1 | whole-file sha256 ⇒ re-bless mandatory for any in-file edit; "off-by-default ⇒ no re-bless" is wrong | §1.2/§1.3: re-bless is a numeric-output obligation gated by §2 regression; re-bless ordered step |
| S1-2 | short-circuit on wrong line; RNG/alloc is the real parity surface | §1.2: gate ENTIRE block on `fb_on`; gain=0 takes literal pre-edit call; T1 RNG-state invariant |
| S2-3 | EMA off-by-one / acausal within-step loop | §1.1: consume-at-top, update-at-recorder; T5 causal test |
| S2-4 / S1 | `alpha=dt/τ` mis-scales the filter | §1.1: `alpha = 1−exp(−dt/τ)` (engine convention); T4 |
| S3-3 | count-not-Hz ⇒ gain NE-dependent + uncalibrated magnitude | §1.1: EMA on Hz; §4.1 magnitude pilot vs static `global_ei_scale` step |
| S1-1 / S3-1,2 | named runaway corners have n_seeds=0 (timeout placeholder); two runaway mechanisms | §4: re-anchor on MEASURED `l2_g1.0`; stratify verdict by core-vs-global runaway type; §5.1 affirmative control |
| S2-1 | per-event-relative `returned` reads a plateau as terminated | §5.2: ABSOLUTE `tail_to_baseline_ratio ≤ 1.5` sustained, raw-rate primary |
| S1-6 / S3-4 | `silent`/`interictal_like` conflate termination with over-suppression / never-ignited; `seizure_like` needs `--dump-fullfield` | §5.3: require ignition + `seizure_like` (not interictal blip); mandate `--dump-fullfield`; `suppressed (no ignition)` label |
| S3-5 | `collision` defined only on returned events ⇒ not a discriminator | §5.3: demote collision to descriptive sidecar |
| S3-7 / S1-5 | static DC brake also makes runaway finite; completion ≠ termination | §5.4 I_global-LEADS-decay; §5.5 anti-rule: completion is not termination |
| S2-4 / S3-6 | seed-fragility; timeout→runaway conflation | §4 ≥5 seeds + sign-consistency; §5.6 separate completed/timeout from runaway |
| S2-6 / S1-7 | prefix_peak polluted by spontaneous event / EMA charging transient | §6: raw-rate primary; fixed_bar from gain=0 control where possible |
| S2-7 | I_global driving signal is core-dominated | §1.1: documented as correct Abbott (global-rate response); injection is uniform on all E, never core-restricted |
| S3-6 (engine) | `feedback_tau_ms=0` default ÷0 | §1.2 gate on gain; T3 |
| §6 figure | redundant panels | §6: 2 independent-question panels (state surface; allow-then-quench dynamic signature) |

---

## Relevant absolute paths

- Engine hook: `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-m3/src/snn_engine/kick_probe.py` (`simulate_kick` kwargs; membrane block lines 243-247; recorder line 257; `is_E` line 156; M1 gate template line 174)
- Guard + re-bless target: `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-m3/src/sef_hfo_snn_engine_guard.py`; `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-m3/results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json` (key `src/snn_engine/kick_probe.py = f7676b78…`)
- Runner: `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-m3/scripts/run_sef_hfo_snn_cm_spontaneous_readout.py` (call line 589; `--global-ei-scale` static counterpart line 431; `BASELINE_MS` line 60; `--dump-fullfield` line 446; `--event-bar-mode` line 452)
- State classifier + timeout setdefault: `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-m3/scripts/analyze_m3a_a1b.py` (`_state` line 65; setdefault runaway line 111-114)
- `returned` per-event-peak-relative def: `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-m3/src/sef_hfo_events.py` (line 228; `RETURN_FRAC=0.2` line 30; `SETTLE_MS=50` line 34)
- A1b anchors: `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-m3/results/topic4_sef_hfo/m3a_slowvars/a1b_grid/status_a1b.json` (measured runaway `l2_g1.0` n_seeds=5 coreR=414; timeout-placeholder `l*_g0.7` n_seeds=0)
- Output root (to create): `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-m3/results/topic4_sef_hfo/m3a_slowvars/a1c_grid/` + `status_a1c.json` + `figures/README.md`