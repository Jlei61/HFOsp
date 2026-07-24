# Reduced 2-D `S_L(x)+S_G` field — transversal-instability screen (design, revised per review, 2026-07-24)

**Status: design, pre-registered. LOCKED before any run.** Branch `codex/topic4-m4-snn-native-exit`.
This is the reduced-model gate the seed-1 SNN pilot is contingent on. It runs a **rate field only** — NO
SNN, NO H/termination, NO E→E topology change. Companion: carrier archive
`docs/archive/topic4/sef_hfo/zm_ictal_carrier_gate_2026-07-24.md` §9-§10; the K-patch predecessor
`src/topic4_zm_patch_screen.py`.

---

## 0. The one question (scoped)

On a **fixed anisotropic excitatory scaffold** with a **matched mean inhibition budget**, can a **local**
inhibition field `S_L(x)` **destabilise the transverse (spatial) modes** of the synchronised ictal carrier
and settle into a **recoverable, long-time-bounded, high-duty phase-staggered** state — where a single
**global** scalar `S_G` at the same mean budget leaves the carrier synchronised?

The carrier gate (§8-§10 of the archive) showed the Z/M+global-`S_G` SNN makes an HFO **burst train**: dense
but ~0.6 s clusters that never connect into a ≥2 s macroepisode. The mechanism hypothesis is that a **single
global scalar** resets the whole core synchronously each cycle. This screen tests, in a reduced 2-D field,
whether **spatially-resolved** inhibition removes that synchronous reset — i.e. whether the synchronised
periodic orbit is **transversally unstable** under local (not global) inhibition, and whether the resulting
phase-staggered state is a bounded attractor.

**This screen does NOT claim**: the full Z/M onset→termination lifecycle; a quantitative equivalence between
the reduced excitability coordinate and the SNN's `z`; that the SNN has a clinical ictal carrier. A pass only
buys a clean migration reason: *what must migrate is the spatial STRUCTURE of local inhibition, not a
termination current and not an E→E change.*

---

## 1. Model — fast E-rate field

One fast E-rate variable `r(x,t) ≥ 0` per cell on an `n×n` lattice (`n=32`, matching the SNN slow-field grid)
over the `L×L` sheet (`L=20 mm`). Reuse `src/sef_hfo_field.convolve_periodic` + a new
`anisotropic_gaussian` kernel; reuse the SNN pooling nonlinearities (§3).

```
τ_a ṙ(x) = −r(x) + F( I0(ξ)  +  [ w_rec·r(x) + w_c·(K_E ∗ r)(x) ] / (1 + α_L·S_L(x) + α_G·S_G)  −  θ )
```

- `F(u) = a_max·[u]_+ / (u_half + [u]_+)` — saturating (bounds `r`), as in the K-patch model.
- **Divisive containment acts on the RECURRENT excitation only** (`w_rec·r + w_c·K_E∗r`), matching the SNN
  `S_G` acting on `I_E_rec`. The external drive `I0(ξ)` is OUTSIDE the division.
- `w_rec` = **local self-recurrence**; `w_c·(K_E∗r)` = **non-local anisotropic recurrence**, with `K_E(0)=0`
  so the two terms do not double-count the self weight.
- **Fast-field parameters inherit the K-patch calibrated oscillatory regime** (`topic4_zm_patch_screen`:
  `τ_a=10, w_rec=2, θ=0.5, a_max=1, u_half=0.5`; `α_ref` at the divisive scale `g_S=16`). `I_base` and `κ_ξ`
  are fixed in Phase A so the 5 `ξ` levels land inside the mean-field oscillation window (§6.1, §8-A).

## 2. Excitability coordinate `ξ` (NOT "frozen z")

The reduced rate field has no per-cell `z`; `z`-depletion (use-dependent inhibition loss) raises
excitability. So the frozen-slow axis is an **independent excitability coordinate**:

```
ξ = 1 − z ,      I0(ξ) = I_base + κ_ξ·ξ        (equivalently θ_eff(ξ) = θ0 − κ_ξ·ξ)
```

`ξ` scales **excitability only**; it MUST NOT scale `S_L`/`S_G` (that would fuse use-dependent inhibition
depletion with spatial containment and make the phase-staggered state's origin unattributable). Until `κ_ξ`
is calibrated from SNN frozen-`z` data, the variable is reported as the **frozen excitability coordinate ξ**,
never as "frozen z". This is a legitimate reduced-order projection of the model's slow variable onto a
neural-field excitability parameter (per review: Proix et al., PMC5852068), not a term-by-term replica.

## 3. Inhibition fields — pooling order is nonlinearity-THEN-pool

Matches the SNN (`slow_field.py:577-581`: `z_G = psi_recruit(r); A_G = pnorm_pool(z_G); μ_G,S_G low-pass`).
Decide **per location** whether it is strongly active, THEN pool — so quiescent surround does not recruit the
pool:

```
Ψ(r) = psi_recruit(r; r0=0, r50=0.4, n=2)                       (reuse slow_field.psi_recruit)
A_L(x) = [ (K_σS ∗ Ψ(r)^p)(x) ]^{1/p}          (local drive; p = p_pool = 3)
A_G    = [ ⟨ Ψ(r)^p ⟩_x ]^{1/p}                (global drive; reuse pnorm_pool over the whole field)
τ_μ μ̇_L = −μ_L + A_L(x) ,   τ_S Ṡ_L = −S_L + S_max·μ_L
τ_μ μ̇_G = −μ_G + A_G   ,   τ_S Ṡ_G = −S_G + S_max·μ_G
```

`τ_μ=30, τ_S=80, S_max=1` (SNN values). Both pooling kernels are **normalised to sum 1** (see §5).

## 4. Kernels

- **`K_E` — anisotropic, fixed to the SNN scaffold.** Anisotropic Gaussian, aspect ratio `AR=2.0` along the
  source→sink axis `theta_deg` (the E1146 values `build_connectivity_rot(theta_EE, AR=2.0)` uses), `K_E(0)=0`.
  Parallel width `σ_E∥`, perpendicular `σ_E⊥ = σ_E∥/AR`. **Not modified in this line** (that would break the
  transfer back to the anisotropic SNN scaffold). Default `σ_E∥ = 1.0 mm` (sanity-anchor to the SNN E→E length;
  the ABSOLUTE scale is secondary to `AR` and to being narrower than `K_σS`).
- **`K_σS` — inhibition pooling kernel.** Normalised **isotropic broad** Gaussian, `σ_S` anchored to the
  inhibition spatial scale (default `σ_S = 3.0 mm`, broader than `σ_E`, matching the SNN's wide-inhibition /
  narrow-excitation structure). Local/global inhibition topologies can independently gate local instability,
  recruitment, and propagation containment (per review: Liou et al., PMC7089769) — this is the structure being
  tested, not merely an extra global negative current.

## 5. Three arms + matched inhibition budget

Both pooling kernels normalised (sum 1); identical `Ψ, p, τ_μ, τ_S`. Then:

| arm | `α_L` | `α_G` |
|---|---|---|
| global-only | 0 | `α_ref` |
| local-only | `α_ref` | 0 |
| mixed (local + weak global) | `(1−ε_G)·α_ref` | `ε_G·α_ref` |

`ε_G = 0.2` (v1). **`α_L + α_G = α_ref` in ALL three arms.** On the uniform manifold `r(x)=r̄(t)`,
`S_L(x)=S_G=S̄(t)`, so every arm has the identical divisive factor `1 + α_ref·S̄` on **any** uniform
trajectory (not just at one reference state). ⇒ **the three arms share the exact same mean-field / synchronised
orbit; they differ ONLY in the transverse (spatial-mode) stability of that orbit.** That is the whole design:
the comparison isolates spatial STRUCTURE, not inhibition STRENGTH.

## 6. Dynamical analysis — transversal instability is the core criterion

### 6.1 Mean-field (uniform manifold)
Reduce to `(r̄, S̄)` (same for all arms): find fixed points, the **oscillation window in `ξ`**, and the
mean-field Jacobian. A bare mean nullcline can only tell us there is a uniform oscillation — it CANNOT prove a
spatial staggered attractor.

### 6.2 Transverse Floquet-like growth rate `λ_⊥(k)` — THE key readout
On the synchronised periodic orbit, linearise for a spatial-Fourier perturbation of wavenumber `k`. Because the
kernels enter as convolutions, mode `k` sees `K̂_E(k)` and `K̂_σS(k)`; `S_G` (global) contributes only to `k=0`,
while `S_L` (local) contributes at every `k`. Numerically estimate `λ_⊥(k)` as the **early exponential growth
rate of the spatial-mode amplitude** `|r̂_k(t)|` over the first few cycles (before nonlinear saturation),
averaged over the fixed perturbation seeds. **Expectation to be tested (not assumed):**
```
global-only:   λ_⊥(k) < 0   for all k>0        (all transverse modes decay → stays synchronised)
local / mixed: λ_⊥(k*) > 0   at some k*>0 in a finite ξ window   (a transverse mode grows → lateral instability)
```
Then nonlinear saturation must **bound** the grown mode into a sustained phase-staggered state (NOT runaway,
NOT collapse). The ξ window where `local λ_⊥(k*)>0` while `global λ_⊥<0` is the window the §8 nonlinear screen
runs in.

### 6.3 Transversal-instability numerical protocol (fixes the "uniform IC can't desync" trap)
A perfectly translation-symmetric system with a uniform IC keeps `r(x,t)=r̄(t)` forever — the uniform manifold
is invariant, so even a transversally-unstable orbit stays synchronised numerically. Therefore:
- **Homogeneous parameters** (no per-cell heterogeneity) for the primary transverse-stability test — so what we
  measure is genuine lateral instability of the orbit, not param-noise / IC-bias desync.
- Seed the aligned limit cycle with a **fixed zero-mean small perturbation**: `r(x,0) = r̄(0) + ε·δr(x)`,
  `⟨δr⟩_x = 0`, `ε ≈ 10⁻⁴ ×` local amplitude.
- **4 pre-fixed perturbation seeds**, the SAME set across all arms (never a different perturbation per arm).
- Record per-`k` early growth rates.

## 7. Metrics (locked definitions)
All are computed **post burn-in** = the first **25%** of each run discarded (matches the patch screen
`settle_frac=0.25`).
- **occupancy** — population `P(t)=⟨r⟩_x`; fraction above `floor_frac·P95` (`floor_frac=0.20`) with the OFF
  state as the absolute baseline (reuse `topic4_zm_patch_screen.population_occupancy`).
- **R_phase** — Kuramoto order parameter of the per-cell oscillation phases (analytic-signal phase of
  `r(x,t)−⟨r⟩_t` per active cell): `R = |⟨e^{iφ_x}⟩_x|`. `R≈1` synchronised, `R≈0` desynchronised.
- **pairwise correlation** — mean pairwise temporal correlation across active cells (as in the patch screen).
- **local-oscillation** — per active cell: number of completed cycles and normalised peak-to-trough amplitude.
- **`λ_⊥(k)`** — §6.2 (early-growth estimate over the first ~5 cycles).
- **active cell** — temporal oscillation amplitude ≥ `AMP_MIN = 0.1 × population peak` (excludes quiescent +
  dead-plateau cells).

## 8. Pre-registered acceptance gate

### Phase A — lock the excitability levels FIRST (no post-hoc selection)
1. Use the **global-only** arm to locate the synchronised-oscillation window in `ξ`.
2. Pre-fix **5 `ξ` levels inside that window** (evenly spaced).
3. All arms, all perturbation seeds, and the phase-reset test use **exactly these 5 levels**. Levels are NOT
   re-chosen after seeing local/mixed results.

### Phase B — reduced-field GO conditions
local-only OR mixed must pass in **≥3 CONSECUTIVE of the 5 preset levels**, and at each such level in **≥3/4
fixed perturbation seeds**, ALL of:
1. **Sustained energy** — occupancy ≥ **0.80**.
2. **Genuine local oscillation** — ≥ **50%** of active cells complete ≥ **10 cycles** with normalised
   peak-to-trough amplitude ≥ **0.20** (excludes a high fixed plateau that would pass occupancy trivially).
3. **Spatial desynchronisation** — post burn-in **median `R_phase` < 0.50 AND pairwise correlation < 0.50**.
4. **Phase-reset return** — re-align all cells to a uniform state + the same small perturbation; the staggered
   criterion (2+3) is re-reached within **5 s** and holds to the end of the run (the staggered state is an
   attractor, not a one-off transient).
5. **Long-time hold** — 30 s screen with the last 10 s showing no drift to synchrony, silence, or a saturated
   plateau; the **central** passing level is re-confirmed at **60 s**.
6. **global-only control** — at the same levels, global-only must remain a **synchronised oscillation**
   (`R_phase ≥ 0.80`). If global-only instead goes silent or to a non-oscillating plateau at a level, that level
   has **no valid matched comparison** (excluded), it is NOT counted as "global failed to synchronise".

**Verdict.** GO (→ SNN migration) iff Phase-B holds for local-only or mixed AND the §6.2 transverse analysis
shows `local λ_⊥(k*)>0 / global λ_⊥<0` in the same window. Otherwise **NO-GO**: the desynchronised-local-inhibition
hypothesis fails even in the reduced 2-D field, and we do not migrate.

## 9. Migration rule (out of scope for this spec, stated for the record)
Only on a full Phase-B + §6.2 GO: seed-1 SNN 3-arm (global-`S_G` / local-`S_L(x)` / mixed, inhibition-side only,
E→E untouched, H off) judged by `carrier_gate_v2.1` (source + virtual-SEEG). H/exit stays out until a carrier
passes. A large patchwise/`S_L` SNN parameter grid is NOT run.

## 10. Non-goals / forbidden claims
- No "the SNN has a carrier / lifecycle / termination".
- No "ξ = frozen z" (it is an excitability abstraction; `κ_ξ` uncalibrated).
- No "phase-staggered relay observed" — the relay is the hypothesis this screen TESTS, not a result.
- No modification of `K_E` / E→E strength.
- A reduced-field GO is a **migration justification**, not a mechanism proof in the SNN.

## 11. Code / outputs
- `src/topic4_zm_field_screen.py` — the field model + `anisotropic_gaussian` + `λ_⊥(k)` estimator + `R_phase`;
  reuses `psi_recruit`/`pnorm_pool` (slow_field) + `convolve_periodic`/`isotropic_gaussian` (sef_hfo_field) +
  `population_occupancy` (patch screen). Off-by-default parity not applicable (new standalone rate model).
- `scripts/run_topic4_zm_field_screen.py` (Phase-A level lock + Phase-B screen + phase-reset + 60 s confirm,
  provenance: git SHA/dirty + module hashes + locked levels + seeds), `scripts/plot_topic4_zm_field_screen.py`.
- `tests/test_topic4_zm_field_screen.py` (kernel normalisation + anisotropy; pooling order; matched-budget
  uniform-manifold identity across arms; `λ_⊥(k)` sign on a constructed stable/unstable orbit; R_phase on
  synthetic in/anti-phase fields; occupancy/oscillation/desync metrics).
- Outputs → `results/topic4_sef_hfo/zm_field_screen/` (+ `figures/README.md`). Cheap rate field; OMP=1.

## 12. References (per review)
- Proix et al., neural-field reduced-order frozen-slow-variable projection — supports treating `ξ` as an
  excitability abstraction (PMC5852068).
- Liou et al., local vs global inhibition topology controlling local instability / recruitment / propagation
  containment (PMC7089769).
