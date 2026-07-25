# Excitable-wave ictal carrier — pre-registered design (2026-07-26)

**Status: LOCKED before any run.** Branch `codex/topic4-m4-snn-native-exit`.
Supersedes the *mechanism question* of the reduced-field screen
(`zm_reduced_field_screen_2026-07-25.md` = `both_stable` NO-GO), **not** its engineering discipline —
every hard-won rule from that round (fail-closed lock, parameter-encoded caches, mutation-catching tests,
stubs that raise, claim-only-what-was-computed) is carried forward verbatim in §9.

---

## 0. Why the question changes

The previous line asked: *does a spatially uniform oscillation spontaneously break into a phase-staggered
pattern when inhibition is localised?* Answer, over the **complete** 513-mode spectrum: **no** — no positive
growth band at any representable wavelength. That closed one specific route to a carrier.

It also showed *why* that route was the wrong shape: the reduced field had **no explicit inhibitory
population, no E–I delay, and a single narrow inhibitory scale**, and its oscillation only existed because a
**new subtractive term** was bolted on. A uniform oscillator asked to disperse its own phase is not how
cortical seizure activity is thought to spread.

This line asks the **excitability** question instead:

> On the ORIGINAL anisotropic E/I substrate with the real Z/M depletion semantics, does a **local** trigger
> produce a **propagating / re-entrant wave** whose spatial relay keeps virtual-electrode energy continuous —
> rather than dying out, flashing the whole field at once, or running away?

This is a different mechanism class (excitable medium, finite-velocity front) and is testable at the reduced
level before any spiking cost.

**What stays frozen (non-negotiable, keeps this line independent of the parallel E→E line):**
- E→E kernel shape, anisotropy and length: elliptical-exponential, `l_EE = 0.38 mm`, `AR = 2` (`l∥ = 0.537`,
  `l⊥ = 0.269`), `K_E(0) = 0`, `Σ = 1`. **Not tuned, not swept, not rotated.**
- **No artificial oscillator**: the `−β·S` subtractive pool from the previous line is REMOVED. If this
  substrate oscillates it must do so through E–I interaction, not through a bolted-on term.
- **No H / termination actuator** (upstream stop #4 still in force).
- Z/M keeps its real semantics: `z ∈ [0,1]` **scales inhibition** (depletion ⇒ disinhibition), `m` is a
  subtracted adaptation current. `z` is NOT re-abstracted into an arbitrary drive parameter.

---

## 1. Model — two-population rate field with explicit fast I

Fields `r_E(x,t), r_I(x,t) ≥ 0` on an `n×n` lattice over `L = 20 mm` (periodic), `n = 48` primary.

```
τ_E ṙ_E = −r_E + F_E[ I_ext + W_EE·(K_E ∗ r_E)/(1 + α_G·S_G) − z·W_EI·(K_EI ∗ r_I)(t−d) − η_m·m − θ_E ]
τ_I ṙ_I = −r_I + F_I[ W_IE·(K_IE ∗ r_E)(t−d) − W_II·(K_II ∗ r_I) − θ_I ]
```

- `F_•(u) = [u]_+ / (u_half + [u]_+)`, `u_half = 0.5`, ceiling 1 (bounded, as before).
- **`τ_I < τ_E`** — the explicit fast inhibitory population the previous field lacked. `τ_E = 10 ms`,
  `τ_I = 4 ms`.
- **`d` = E–I delay** (ms), applied to the *cross*-population terms only (E→I and I→E), implemented as a
  ring buffer. `d = 2 ms` primary. `d = 0` is an explicit ablation arm (§5).
- **Inhibitory spatial scale is broader than excitatory**: `K_EI`, `K_IE`, `K_II` are normalised isotropic
  Gaussians of width `σ_I`. Primary `σ_I = 1.5 mm` — anchored to the SNN's slow-resource footprint
  `sigma_q = 1.5 mm`; it is a **coarse-grained containment scale**, NOT a claim about monosynaptic
  inhibitory axon length (~0.25 mm). Sensitivity `σ_I ∈ {0.75, 1.5, 3.0}` for the central candidate only.
- `α_G·S_G` = the divisive global pool, kept from the SNN (`alpha_G = 16`); `S_G` is FROZEN in Phase 1.
- Local/non-local split of `K_E` uses the same derived `q_cell` quadrature as the previous line (never
  hand-set): `w_rec = W_EE·q_cell`, `w_c = W_EE·(1−q_cell)`, `K_E(0)=0`.

**Phase-1 frozen slow state**: `(z, m, S_G)` are uniform frozen scalars. No slow dynamics in this sprint.

## 2. The three staged questions (only Q1–Q2 are in scope this sprint)

1. **Entry** — as `z` depletes, does a local trigger cross a propagation threshold?
2. **Carrier** — does that produce a finite-velocity propagating / re-entrant front giving **≥ 2 s** of
   continuous electrode-band energy, as opposed to a whole-field synchronous flash or a train of separated
   discrete events?
3. **Exit** *(NOT this sprint)* — do `m`/`z` recovery and `S_G` remove the propagation condition and return
   the substrate to a region that can still generate irregular interictal events?

**This sprint delivers the frozen-state atlas for Q1 and the wave classification for Q2 only.** Q3 and any
slow-variable closed loop are out of scope and must not be claimed.

## 3. Phase 0 — is the substrate EXCITABLE at all? (hard gate)

The whole question is malformed unless the uniform two-population system has a **stable low-activity rest
state with a threshold**. Reduce to the space-independent `(r̄_E, r̄_I)` system (all kernels sum to 1) and,
over a pre-registered grid `W_EE ∈ {2,3,4,6}, W_EI ∈ {1,2,4}, W_IE ∈ {2,4,8}, W_II ∈ {0,1,2}`,
`θ_E ∈ {0.4,0.5,0.6}`, `θ_I = 0.5`, at the atlas-centre `(z,m,S_G) = (1.0, 0, 0)`, classify each point:

- **`excitable`** — a stable low fixed point (`r̄_E < 0.05`) AND a super-threshold pulse produces a large
  transient excursion (`peak r̄_E ≥ 0.3`) that RETURNS to the low state; AND a sub-threshold pulse does not.
- **`oscillatory`** — no stable fixed point (limit cycle). ⚠️ Not what this line wants; a self-oscillating
  medium re-opens the previous line's question.
- **`bistable_saturated`** — a super-threshold pulse leaves the system in a high state permanently.
- **`inexcitable`** — even a large pulse decays without a large excursion.

**Selection rule (minimal intervention, locked, applied to `excitable` points only)** — in strict order:
1. smallest `W_EE` (least excitatory gain);
2. then smallest `|log2(W_EI/2)|` (closest to a balanced E/I ratio), ties → smaller `W_EI`;
3. then smallest `W_IE`, then smallest `W_II`, then `θ_E` closest to 0.5;
4. final deterministic lexicographic tie-break `(W_EE, W_EI, W_IE, W_II, θ_E)`.
Stop at the first satisfying point; do NOT keep searching for a "better" excitable regime.

**Robustness**: the chosen point must retain the `excitable` class at `dt` and `dt/2`.
**NO-ORBIT-STYLE STOP**: if the grid contains **no** `excitable` point → immediate NO-GO; the 2-D field is
NOT built, and we report that this substrate (with `K_E` frozen) is not an excitable medium in the tested
window. **Do not respond by tuning `K_E`.**

## 4. Phase A — the frozen state-fork atlas (write-once lock)

Lock, before any 2-D run, into `phaseA_lock.json`: spec SHA, the Phase-0 operating point, the atlas grid,
kick parameters, `q_cell`, seeds, `dt`, `n`, kernel hashes, git SHA + dirty flag.

**Atlas axes** (frozen slow state; 3-D grid):
- `z ∈ {1.00, 0.85, 0.70, 0.55, 0.40, 0.25}` — inhibitory efficacy from intact to strongly depleted.
- `m ∈ {0, 2, 5, 10}` — adaptation load (`η_m = 0.001` as in the SNN lockpoint ⇒ current `η_m·m`).
- `S_G ∈ {0, 0.05, 0.15}` — frozen global divisive pool.

**Kick** (the local trigger): a disk of radius `r_kick = 1.5 mm` at a fixed site, added to `I_ext` with
amplitude `A_kick` for `d_kick = 20 ms`. `A_kick` is calibrated ONCE at the atlas centre as the smallest
amplitude that produces a super-threshold local response there, then FROZEN across the atlas (so the atlas
varies only in slow state, not in trigger strength). A **sub-threshold control** at `0.5·A_kick` is run at
every atlas point.

## 5. Arms (ablations; all share the frozen kernels and the locked operating point)

| arm | change | purpose |
|---|---|---|
| `full` | as §1 | the candidate |
| `no_delay` | `d = 0` | is the E–I delay load-bearing for propagation? |
| `narrow_inh` | `σ_I = 0.375 mm` (≈ `K_E` scale) | is the BROAD inhibitory scale load-bearing? |
| `no_I` | `W_EI = W_IE = 0` (E-only, inhibition removed) | sanity: without inhibition it must run away — if it does NOT, the substrate is mis-calibrated |

## 6. Pre-registered classification of the response to a kick

Per atlas point × arm × seed, from `r_E(x,t)` over `T = 4000 ms`:

- **front arrival** at radius `ρ` = first time the ring-mean `r_E` at distance `ρ` from the kick centre
  exceeds `0.2 × (its own peak over the run)`, for `ρ ∈ {2,4,6,8} mm`.
- **wave speed** `v` = slope of a least-squares fit of `ρ` vs arrival time (mm/s), reported with `R²`.
- **active area fraction** `A(t)` = fraction of cells with `r_E ≥ 0.1·a_max`.
- **duration** = length of the longest span with `A(t) ≥ 0.05`.

Classes (evaluated in this order):

1. **`runaway`** — `A(t) ≥ 0.80` sustained for ≥ 500 ms, or `A(T_end) ≥ 0.5`.
2. **`whole_field_flash`** — the front reaches `ρ = 8 mm` within **≤ 20 ms** of the kick (i.e. faster than
   any finite-velocity front the lattice can resolve) OR arrival-time fit `R² < 0.5` with all arrivals inside
   20 ms — activation without a spatial gradient.
3. **`propagating_wave`** — arrivals at ≥ 3 of the 4 radii, monotone in `ρ`, fit `R² ≥ 0.8`, speed
   `v ∈ [20, 2000] mm/s` (a physiologically loose but finite band), and `A(t)` peaks below 0.80.
4. **`sustained_wave`** — as `propagating_wave` AND `duration ≥ 2000 ms` (the carrier target).
5. **`no_propagation`** — none of the above (front never reaches 4 mm, or decays back).

**Sub-threshold control must classify as `no_propagation`** at any point claimed as a wave; if the
half-amplitude kick propagates too, the point is **not** threshold behaviour and is excluded.

## 7. Phase-B GO / NO-GO (pre-registered)

**GO** (→ justifies restoring slow variables, then an SNN pilot) iff ALL:
1. A **non-empty, contiguous** region of the `(z, m, S_G)` atlas classifies as `propagating_wave` or
   `sustained_wave` in the `full` arm, in **≥ 3 of 4 seeds** per point.
2. That region is **bounded on both sides**: adjacent atlas points (in the `z` direction) include at least
   one `no_propagation` and at least one `runaway`/`whole_field_flash` — i.e. a genuine window, not an edge
   artefact of the grid.
3. The sub-threshold control is `no_propagation` throughout the region.
4. At least one point in the region reaches `sustained_wave` (≥ 2 s), OR — if none does — the verdict is
   the weaker `propagating_but_not_sustained`, which is **not** a GO for the SNN and instead defines the
   next question.

**Verdict vocabulary** (single source of truth, adjudicated by a pure function):
`no_excitable_regime` (Phase-0 stop) · `no_propagation_anywhere` · `flash_only` · `runaway_only` ·
`propagating_but_not_sustained` · `sustained_wave_window` (= GO) · `no_evidence`.

**Ablations are reported as SEPARATE contrasts** (never merged into the main verdict): `no_delay` and
`narrow_inh` say *which ingredient the propagation needed*; `no_I` is a calibration sanity check.

## 8. Forbidden claims
- ❌ "the SNN has a seizure / carrier / lifecycle" — this is a reduced field; no SNN, no virtual SEEG here.
- ❌ any Exit / termination / recovery claim (Q3 is out of scope).
- ❌ "propagation proves the clinical mechanism" — a finite-velocity front in a rate field is a *candidate*.
- ❌ claiming coverage beyond the atlas actually run, or beyond the classified radii/`T` window.
- ❌ tuning `K_E`, adding a subtractive oscillator, or touching H in response to a negative result.
- ❌ merging the ablation contrasts with the main verdict.

## 9. Engineering rules carried forward (all learned the hard way last round)
1. `phaseA_lock.json` is **write-once AND fail-closed**: reuse only if spec SHA / operating point / grid /
   `dt` / seeds all still match; otherwise `exit(3)`.
2. Every cached artefact's filename encodes **every parameter that changes the trajectory** (`T`, `dt`, `n`,
   arm, seed, atlas point) — a short diagnostic run must never be resumable as a production run.
3. Any linearisation added later must be verified by **finite difference against the independently-written
   field RHS** (mutation-catching), not against itself.
4. Unimplemented acceptance criteria **raise `NotImplementedError`** — never look functional.
5. Provenance in every output JSON: git SHA, dirty flag, module SHA256s, lock hash, exact parameters.
6. **Cheap-first**: Phase 0 (0-D, seconds) → atlas coarse pass → only then long/at-scale runs.
7. Claim only what was computed; state the scanned ranges next to every conclusion.
8. `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`; memory floor 64 GB; per-arm resume.

## 10. Outputs
`src/topic4_zm_excitable_{meanfield,field,verdict}.py`, `scripts/run_topic4_zm_excitable_atlas.py`,
`scripts/plot_topic4_zm_excitable_atlas.py`, `tests/test_topic4_zm_excitable_*.py`;
results → `results/topic4_sef_hfo/zm_excitable_wave/` (+ Chinese `figures/README.md`, FIGURE_INDEX row).
