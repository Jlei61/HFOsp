# Reduced 2-D `S_L(x)+S_G` field — transversal-instability screen (design, revision 3, 2026-07-25)

**Status: calibration-informed design, LOCKED before the confirmatory field screen.** (Not "pre-registered"
in the strict sense: the Fix-A operating region `β≈2–4` and the existence of ~111 oscillatory sets were seen
in an offline 0-D probe BEFORE this spec was written. Everything downstream of the Phase-A lock — arms,
thresholds, verdict — is fixed before the confirmatory runs.) Branch `codex/topic4-m4-snn-native-exit`.
Reduced **rate field only** — NO SNN, NO H/termination, NO E→E change. Rev-2 incorporates the second review:
the pool is now **dual (divisive + subtractive)**, a gating **Phase 0** mean-field check is added, and the
transverse readout is a proper **per-mode 3×3 Floquet** analysis. Companion carrier archive
`docs/archive/topic4/sef_hfo/zm_ictal_carrier_gate_2026-07-24.md` §9-§10; predecessor
`src/topic4_zm_patch_screen.py`.

---

## 0. The one question (scoped)

On a **fixed anisotropic excitatory scaffold** with a **matched inhibition budget**, does making the
inhibitory pool **spatially local** (`S_L(x)`) render the **synchronised burst-train orbit transversally
unstable** — so the population settles into a **recoverable, long-time-bounded, high-duty phase-staggered**
state — where a single **global** scalar `S_G` at the same budget keeps it synchronised?

Carrier gate §8-§10: the Z/M+global-`S_G` SNN makes an HFO **burst train** (dense ~0.6 s clusters that never
connect into a ≥2 s macroepisode), hypothesised to be a **single global scalar resetting the whole core
synchronously**. This screen tests, in a reduced 2-D field, whether spatially-resolved inhibition removes that
synchronous reset.

**⚠️ SCOPE CONDITIONAL (rev-3, review P0-1).** The `−β·S` subtractive term is a **NEW mechanism on this
line**: the Z/M `sg` arm that produced the burst-train result ran `alpha_G=16, beta_SG=0.0` (verified in
`sg_seed1.json::config`), i.e. **purely divisive**. The engine *supports* `beta_SG`, but it was never engaged.
Phase 0 showed the purely-divisive oscillator has **no synchronised orbit at all**, so the subtractive term is
what *creates* the orbit here. Therefore this screen answers:

> *Conditional on a NEW dual (divisive + subtractive) pool oscillator substrate, does the spatial rank of the
> inhibitory feedback control synchrony vs phase staggering?*

It **cannot** answer: *"does merely localising the CURRENT Z/M+`S_G` global pool produce a carrier?"* — because
the current Z/M+`S_G` (β=0) has no orbit to localise. This line is a **new inhibition-side mechanism arm**, not
a seamless spatialisation of the existing model. The 4-arm decomposition (§5) makes that attribution explicit.

**Non-goals** (unchanged, restated §11): the full Z/M onset→termination lifecycle; a quantitative
`ξ ↔ z` equivalence; that the SNN has a clinical ictal carrier. A pass buys ONLY a migration justification:
*what must migrate is the spatial STRUCTURE of local inhibition — not a termination current, not an E→E change.*

---

## 1. Model — fast E-rate field with a DUAL (divisive + subtractive) pool

One fast E-rate variable `r(x,t) ≥ 0` per cell on an `n×n` lattice (`n=32` primary, `n=64` central-candidate
sensitivity) over `L=20 mm`. Reuse `sef_hfo_field.convolve_periodic` + a new anisotropic kernel; reuse the SNN
pooling nonlinearities (§3).

```
u(x)     = I0(ξ)  +  [ w_rec·r(x) + w_c·(K_E ∗ r)(x) ] / (1 + α·S_eff(x))  −  β·S_eff(x)  −  θ
τ_a ṙ(x) = −r(x) + F(u(x)) ,        F(u) = [u]_+ / (u_half + [u]_+)     (a_max=1, u_half=0.5)
```

- **DUAL inhibition. `α·S` divides the recurrent excitation only** (this part matches the SNN `S_G` acting on
  `I_E_rec`, `alpha_G=16`); **`−β·S` subtracts on the membrane and is NEW on this line** (the engine exposes
  `beta_SG` but the `sg` arm ran `beta_SG=0`; see the §0 scope conditional). **Why dual is required**
  (Phase-0 finding, §6.0): with the divisive term ALONE,
  the external drive `I0` props the high state up so the pool cannot produce a full OFF/reset → no synchronised
  orbit exists (a uniform-mean-field scan over `W0∈[2,24], α∈[0.5,16], I0∈[0,2], θ∈[0.3,0.8]` found 0
  oscillatory sets). Adding `−β·S` restores a clean relaxation orbit (trough→0, period ~150–220 ms ≈ the SNN
  ~300 ms IBI; 111 oscillatory sets). The subtractive reset is what ties synchrony to the pool's spatial rank
  (global `−β·S_G` resets every cell identically = synchronous; local `−β·S_L(x)` resets each microdomain =
  can desynchronise).
- `w_rec` = local self-recurrence; `w_c·(K_E∗r)` = non-local anisotropic recurrence; `K_E(0)=0` (no
  self-double-count), `K_E` renormalised to `Σ=1` (§4). On the uniform manifold the total recurrent gain is
  `W0 = w_rec + w_c`.
- **The local/non-local SPLIT IS DERIVED, NOT SET** (review P0-2): `w_rec = W0·q_cell`, `w_c = W0·(1−q_cell)`,
  where `q_cell` = the fraction of the continuous `K_E` mass falling inside ONE lattice cell (fine-quadrature
  coarse-graining). Computed: **`q_cell = 0.226` at `n=32`** (0.625 mm cells) and **`0.077` at `n=64`** — so it
  scales correctly with resolution. A hand-set `w_frac=0.5` would MORE THAN DOUBLE the self-excitation and
  manufacture per-cell microdomain oscillators (an EE-structure change in disguise, violating "EE untouched").
  `q_cell` is locked into `phaseA_lock.json` and never re-chosen from `λ_⊥` results; a `w_frac` sensitivity
  (`0.5×q_cell`, `q_cell`, `2×q_cell`) is run for the central candidate only.
- `S_eff(x)` is the pool the cell sees, set by the arm (§5).

## 2. Excitability coordinate `ξ`

`ξ ∈ [0,1]` is a **frozen excitability coordinate that monotonically represents inhibition depletion**
(higher `ξ` = more depleted = more excitable): `I0(ξ) = I_base + κ_ξ·ξ`. It scales **excitability only** and
MUST NOT scale `S_L`/`S_G` (that would fuse use-dependent inhibition depletion with spatial containment and
make the staggered state's origin unattributable). It is a reduced-order excitability abstraction (Proix et al.
PMC5852068, for motivation only), **NOT** a term-by-term `z` replica; `κ_ξ` is uncalibrated and `ξ` is never
called "frozen z", and there is no strict `ξ=1−z` claim.

## 3. Inhibition pools — nonlinearity-THEN-pool (matches the SNN)

Matches `slow_field.py:577-581` (`z_G=psi_recruit(r)` per-location, THEN `A_G=pnorm_pool(z_G)`, THEN
two-stage low-pass `μ→S`). Decide per-location whether it is strongly active FIRST, then pool, so quiescent
surround does not recruit the pool:

```
Ψ(r) = psi_recruit(r; r0=0, r50=0.4, n=2)                          (reuse slow_field.psi_recruit)
A_L(x) = [ (K_σS ∗ Ψ(r)^p)(x) ]^{1/p}         (local drive; p = p_pool = 3)
A_G    = [ ⟨ Ψ(r)^p ⟩_x ]^{1/p}               (global drive; reuse slow_field.pnorm_pool)
τ_μ μ̇_L = −μ_L + A_L(x) ,  τ_S Ṡ_L = −S_L + S_max·μ_L
τ_μ μ̇_G = −μ_G + A_G    ,  τ_S Ṡ_G = −S_G + S_max·μ_G
```
`τ_μ=30, τ_S=80, S_max=1` (SNN values). Both pooling kernels normalised to `Σ=1`.

## 4. Kernels

- **`K_E` — the real SNN anisotropic kernel, coarse-grained onto the lattice, FIXED.** Rotated
  elliptical-exponential (matching `connectivity_rot` + `params.py:68` `l_EE=0.38 mm`, `AR=2`):
  `K_E(x) ∝ exp(−√((u/l∥)² + (v/l⊥)²))` with `(u,v)` rotated to the source→sink axis `θ_EE`,
  `l∥ = l_EE·√AR ≈ 0.537 mm`, `l⊥ = l_EE/√AR ≈ 0.269 mm`; `K_E(0)=0`; renormalise `Σ K_E = 1`. **Not modified
  in this line.** Because `l_EE` is sub-lattice at `n=32` (`0.625 mm/cell`), `n=64` (`0.3125 mm/cell`) is run
  as a resolution sensitivity for the central candidate; the anisotropy `AR=2` and orientation are the
  load-bearing features.
- **`K_σS` — inhibition pooling kernel.** Normalised isotropic Gaussian at a **pre-registered coarse-grained
  containment scale** `σ_S` (primary `σ_S = 2.0 mm`, broader than `K_E`; the SNN has no native `S_L` pool —
  direct inhibition ~0.25 mm, slow-resource `σ_q=1.5 mm` — so this is a NEW mechanism scale, not "matching the
  SNN"). One primary `σ_S` is locked; a `σ_S ∈ {1.0, 2.0, 4.0} mm` sensitivity is run for the central
  candidate ONLY; `σ_S` is never re-chosen to favour local (Liou et al. PMC7089769, motivation only).

## 5. FOUR arms — mechanism decomposition (review P0-1)

Both pooling kernels normalised; identical `Ψ, p, τ_μ, τ_S`. Arms 2-4 share `(α, β)` and differ ONLY in the
**spatial rank of the pool the cells see**; arm 1 is the β-ablation that pins what the new term contributes:

| # | arm | `S_eff(x)` | `β` | role |
|---|---|---|---|---|
| 1 | `div_global` | `S_G` (scalar) | **0** | **reduced baseline of the CURRENT Z/M+`S_G`** (`alpha_G=16, beta_SG=0`) — expected: no orbit (Phase-0 finding) |
| 2 | `dual_global` | `S_G` (scalar) | `β_lock` | shows the NEW subtractive term is what creates the synchronised orbit |
| 3 | `dual_local` | `S_L(x)` (field) | `β_lock` | the test arm |
| 4 | `dual_mixed` | `(1−ε_G)·S_L(x) + ε_G·S_G`, `ε_G=0.2` | `β_lock` | local + weak global |

Arm 1 vs 2 is the **mechanism attribution** (β creates the orbit); arm 2 vs 3/4 is the **spatial-rank test**
(the actual question). Reporting must never collapse these two contrasts into one claim.

On the uniform manifold `r(x)=r̄(t) ⇒ S_L(x)=S_G=S̄(t)`, so `S_eff=S̄` and **arms 2-4 share the identical
`(r̄,μ̄,S̄)` mean-field on ANY uniform trajectory** — same `(α,β)`, same synchronised orbit ⇒ they differ ONLY
transversally. The comparison isolates spatial STRUCTURE, not inhibition STRENGTH.


## 6. Dynamical analysis

### 6.0 Phase 0 — mean-field recalibration + NO-ORBIT STOP (gating; run FIRST)
Uniform 3-state `(r̄, μ̄, S̄)` (identical across arms):
```
τ_a ṙ̄ = −r̄ + F(I0(ξ) + W0·r̄/(1+α·S̄) − β·S̄ − θ) ,  τ_μ μ̄̇ = −μ̄ + Ψ(r̄) ,  τ_S S̄̇ = −S̄ + S_max·μ̄
```
Continuation over the grid `W0∈{2,3,4,6}, α∈{1,2,4}, β∈{0,1,2,4,8}, θ∈{0.4,0.5,0.6}`, sweeping
`I0∈[0.5,2.0]` (`β=0` included so the divisive-only ablation is measured, not assumed).

**MINIMAL-INTERVENTION selection rule (locked; review P0-1) — NOT "widest window / strongest oscillator":**
among configs whose oscillatory `I0` points form a **single contiguous segment** of the `I0` sweep with
`n_points ≥ 5`, choose in this strict lexicographic order:
1. **smallest `β`** (least new mechanism);
2. then smallest `|α − 16|`-rank … i.e. smallest deviation from the SNN anchor `α_anchor = 16` — implemented as
   smallest `|log2(α/16)|`; ties → smaller `α`;
3. then smallest `|W0 − W0_anchor|` with `W0_anchor = 2`; ties → smaller `W0`;
4. then `θ` closest to 0.5; ties → smaller `θ`;
5. final tie-break: lexicographic `(W0, α, β, θ)` — deterministic, never dict-iteration order.
Stop as soon as a config satisfies {periodic orbit, OFF trough, numerical convergence}; do not keep searching
for a "better" oscillator.

**Contiguity + robustness (review P0-3):** oscillatory `I0` points are split into contiguous runs; only ONE
run may be used; the 5 `ξ` levels are taken from that run's **interior** (drop the two boundary points, to stay
off the bifurcation edges). The chosen operating point must give the **same orbit classification at `dt` and
`dt/2`**; otherwise take the next candidate in the ordering.

**Lock the operating point `(W0, α, β, θ)` and the `ξ→I0` map** judged on the MEAN-FIELD ONLY (arm-independent;
never using local/mixed results). **If no config with a contiguous ≥5-point orbit segment exists → immediate
NO-GO; the 2-D field is NOT built.** (This is why "inherit the K-patch calibration" is wrong: recurrent-only
division is a different oscillator.) Mean-field is 3-D; `μ̄=A(r̄)`, `S̄=S_max·μ̄` may be substituted only for the
fixed-point equation, never for the orbit / Jacobian / continuation.

### 6.1 Mean-field structure
Fixed points, oscillation window in `ξ`, mean-field Jacobian (3×3) vs `ξ` — establishes the uniform orbit the
transverse analysis linearises around. A mean nullcline alone cannot prove a spatial staggered attractor.

### 6.2 Transverse Floquet — THE primary criterion (per 2-D mode, 3×3 monodromy)
On the synchronised periodic orbit `(r_0(t), μ_0(t), S_0(t))`, linearise the FIELD for a spatial-Fourier
perturbation `(δr, δμ, δS) ∝ e^{i(k_x x + k_y y)}`. Convolutions become multiplications by `K̂_E(k)`,
`K̂_σS(k)`; **the global pool `S_G` responds only at `k=0`**, so for `k≠0` global-only sees a frozen pool,
while the local pool `S_L` responds at every `k` via `K̂_σS(k)`. For each mode:
1. integrate the **3×3 time-periodic variational system** over one orbit period `T`;
2. form the `3×3` **monodromy matrix**; take the max-magnitude Floquet multiplier `ρ_max`;
3. `λ_⊥(k_x,k_y) = T^{-1}·log|ρ_max|`.
Because `K_E` is anisotropic, DO NOT collapse to scalar `|k|` or average over equal `|k|`: save the full 2-D
`λ_⊥` heatmap, the most-unstable mode `k*`, and its angle vs `θ_EE`. **Sign requires a numerical margin:** the
same sign at `dt` and `dt/2`, and `|λ_⊥| >` the discretisation-error floor. A full-field small-perturbation
growth-rate estimate is an **independent sanity check**, NOT the primary estimator.
Target: `global-only λ_⊥(k)<0 ∀k`; `local/mixed λ_⊥(k*)>0` at some `k*` in a finite `ξ` window.

### 6.3 Nonlinear transverse-instability protocol (fixes the invariant-manifold trap)
A translation-symmetric system with a uniform IC keeps `r(x,t)=r̄(t)` forever (invariant manifold). Therefore:
- **Homogeneous parameters** (no per-cell heterogeneity) — primary transverse test measures genuine lateral
  instability, not param-noise / IC-bias desync.
- Initialise **all fields to a fixed uniform-orbit phase point** `(r_0(t*), μ_0(t*), S_0(t*))` (so `μ_L,S_L,
  μ_G,S_G` start at the orbit value, NOT zero — avoids a slow-pool startup transient being read as growth),
  then add a **fixed zero-mean small perturbation to `r` only**: `r(x,0)=r_0(t*)+ε·δr(x)`, `⟨δr⟩=0`,
  `ε=10⁻⁴×` amplitude.
- **4 pre-fixed perturbation seeds**, the SAME set across all arms.

## 7. Metrics (locked; post burn-in = first 25% discarded)
- **occupancy** — `P(t)=⟨r⟩_x`; fraction above `0.20·P95` (OFF-state absolute baseline; reuse
  `population_occupancy`).
- **energy floors** — `P95 ≥ 0.1·a_max` AND `mean P_local ≥ 0.5·mean P_global` (a staggered state must carry
  real energy, not just avoid zero).
- **active_area_frac** — fraction of **all** cells with temporal oscillation amplitude ≥ `0.1·a_max`.
- **oscillatory fraction** — fraction of **all** cells (denominator = every cell, not the active subset)
  completing ≥ 10 cycles with normalised peak-to-trough `p2p/a_max ≥ 0.20`. Cycle phase via
  **upward-crossing / cycle interpolation** (relaxation-oscillation-appropriate); Hilbert phase reported as a
  sensitivity only (non-sinusoidal bursts break Hilbert monotonicity).
- **R_phase** — spatial Kuramoto `R(t)=|⟨e^{iφ_x(t)}⟩_x|` computed at each time, then **median over the
  acceptance window**.
- **pairwise correlation** — mean pairwise temporal correlation across active cells.
- **period** — dominant cycle period per arm.
- **`λ_⊥(k)`** — §6.2.

## 8. Pre-registered acceptance gate

**Phase 0** (§6.0): orbit exists → continue; no orbit → NO-GO (no field built).

**Phase A** — lock excitability levels FIRST. Use the mean-field (arm-independent) to fix **5 `ξ` levels evenly
inside the chosen contiguous segment's INTERIOR**. Write `phaseA_lock.json`: spec SHA, `(W0,α,β,θ)`, segment
definition, the 5 `ξ`/`I0` levels, `q_cell`/`w_frac` provenance, the 4 perturbation seeds, solver + `dt` +
grid `n`, kernel hashes. **The lock is write-once: if the file already exists the runner FAILS CLOSED** (never
silently overwrites); re-locking requires an explicit human deletion. Phase B reads this lock ONLY; levels are
never re-chosen after seeing local/mixed results.

**Cheap-first ordering (review P1-3):** Phase B runs the **Floquet map for every level FIRST** (seconds), and
only then launches nonlinear runs — restricted to the target window (`global λ_⊥<0 & local/mixed λ_⊥>0`) or a
pre-registered subcritical-diagnostic window. If no level shows a target window, write the taxonomy verdict and
STOP without the 30/60 s sweeps.

**Phase B** — reduced-field GO iff BOTH:
- **(i) Nonlinear screen:** local-only OR mixed passes in **≥3 CONSECUTIVE of the 5 levels**, each in **≥3/4
  fixed perturbation seeds**, ALL of:
  1. **energy** — occupancy ≥ 0.80 AND the §7 energy floors.
  2. **genuine local oscillation** — active_area_frac ≥ 0.50 AND oscillatory fraction ≥ 0.50 (over all cells),
     `p2p/a_max ≥ 0.20` (excludes a high fixed plateau + a tiny-active-set loophole).
  3. **spatial desync** — median-over-time `R_phase < 0.50` AND pairwise correlation < 0.50.
  4. **period band** — local period within `0.5–2×` the matched global period (excludes slow drift).
  5. **phase-reset return** — full-state reset (`r,μ_L,S_L,μ_G,S_G`) to a uniform-orbit phase point + the same
     `10⁻⁴` `r`-perturbation; criteria 2+3 re-reached within 5 s and held to the end.
  6. **long-time** — 30 s screen, last 10 s no drift to synchrony / silence / saturated plateau; the central
     passing level re-confirmed at **60 s**, and at `dt/2` and `n=64`.
  7. **global-only control** — at the same levels, global-only remains a **synchronised oscillation**
     (`R_phase ≥ 0.80`). If global-only goes silent or to a non-oscillating plateau at a level, that level has
     **no valid matched comparison** (excluded) — NOT counted as "global failed to synchronise".
- **(ii) Transverse Floquet (§6.2):** `local/mixed λ_⊥(k*)>0` while `global λ_⊥<0` in the SAME window
  (numerical sign margin met).

## 9. Verdict taxonomy (4-cell; no over-broad NO-GO)
Report the pair `(global transverse stability, local transverse stability)`:
- **global-stable / local-unstable** + Phase-B(i) → **GO** (target mechanism; migrate).
- **global-unstable / local-stable** → reverse result (local suppresses spatial modes MORE — a real finding).
- **both-stable** → no transverse instability route in the tested window.
- **both-unstable** → the substrate is transversally unstable regardless of inhibition rank.
- A recoverable staggered state with `λ_⊥<0` (nonlinear pass but linearly stable orbit) → **subcritical /
  finite-amplitude staggered candidate** — this refutes ONLY the "instability-from-the-synchronised-orbit"
  path, NOT the whole local-inhibition hypothesis.

## 10. Migration rule (out of scope here, for the record)
Only on GO: seed-1 SNN 3-arm (global-`S_G` / local-`S_L(x)` / mixed, inhibition-side only, dual divisive+
subtractive to match this field, E→E untouched, H off) judged by `carrier_gate_v2.1`. No large `S_L` SNN grid.

## 11. Forbidden claims
- **No "localising the current Z/M+`S_G` global pool produces a carrier"** — the current `sg` arm is β=0 and has
  no orbit; every result here is conditional on the NEW dual-pool substrate (§0). Report arm 1-vs-2 (β creates
  the orbit) and arm 2-vs-3/4 (spatial rank) as SEPARATE contrasts, never merged.
- No "SNN has a carrier / lifecycle / termination".
- No "`ξ = frozen z`" and no strict `ξ=1−z`.
- No "phase-staggered relay observed" — it is the hypothesis under test.
- No `K_E` / E→E modification; no calling `σ_S` an existing SNN scale.
- "synchronised **burst-train orbit**" / "candidate carrier substrate", never "synchronised ictal carrier".
- A reduced-field GO = migration justification, not an SNN mechanism proof.

## 12. Code / outputs / engineering
- `src/topic4_zm_field_meanfield.py` — Phase-0 uniform 3-state RHS + orbit detector + continuation (tested).
- `src/topic4_zm_field_screen.py` — 2-D field (Fix A) + anisotropic elliptical-exp `K_E` + per-mode 3×3
  Floquet estimator + upward-crossing phase / `R_phase`; reuse `psi_recruit`/`pnorm_pool` (slow_field),
  `convolve_periodic`/`isotropic_gaussian` (sef_hfo_field), `population_occupancy` (patch screen).
- `scripts/run_topic4_zm_field_screen.py` (Phase 0 → A lock → B), `scripts/plot_topic4_zm_field_screen.py`,
  `tests/test_topic4_zm_field_{meanfield,screen}.py` (kernel normalisation + anisotropy; pooling order; matched-
  budget uniform-manifold identity across arms; Phase-0 orbit detection incl. a NO-ORBIT case; `λ_⊥(k)` sign on
  a constructed stable vs unstable orbit + `dt`/`dt/2` margin; `R_phase` on in/anti-phase fields; the §7 metrics
  incl. the tiny-active-set + plateau loopholes).
- **Engineering:** `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`; arms run sequentially, arrays released between
  arms; **streaming metrics** — never hold `60 s × time × 32 × 32 × all-states`; field traces downsampled to
  ~5 ms for figures; primary `dt` locked with a `dt/2` central-candidate convergence check. Outputs →
  `results/topic4_sef_hfo/zm_field_screen/` (+ `figures/README.md`). Provenance JSONs carry git SHA + git_dirty
  + module SHA256 + `phaseA_lock` hash (as in `carrier_gate_v2.1`).

## 13. References (per review; motivation only, do NOT support the expected sign)
- Proix et al., PMC5852068 — neural-field reduced-order frozen-slow projection (supports `ξ` as an excitability
  abstraction, not `ξ↔z` equivalence).
- Liou et al., PMC7089769 — local vs global inhibition topology (supports studying inhibition spatial rank, not
  that local necessarily destabilises the synchronised orbit).
