# Figure 5 Supplementary spatial dynamics — plotting contract

**Status**: Figure 5 candidate companion figures, frozen 2026-07-19.  
**Main-figure relation**: the current Figure 5 V1 candidate (`fig_mz_early_bridge`) shows the
virtual-SEEG observation bridge. The two figures specified here are separate dynamics supplements;
they explain how spatial susceptibility and the leading linear mode change on the approach to the
same operational-runoff regime. They do not replace or enlarge the Figure 5 main canvas.

## 1. Scientific object

Both figures use candidate `zA_q50_tz10000`, seeds 1/3/4. The state source is the **continuous MZ
z-only SNN trajectory**. Captured neuronal `z_E` is mapped to the fixed E1146-aligned coarse sheet and
converted to inhibitory efficacy in a **frozen-q heterogeneous M3B E/I rate-field surrogate**.

This contract is therefore:

> actual MZ z-only slow-state timestamps → frozen-q rate-field operator → spatial perturbation and
> Jacobian-mode readouts.

It is not the historical `q_I/g_K` formula, not a perturbation of left/right inhibitory pools, and not
a direct perturbation or eigendecomposition of the full spiking network.

## 2. Figure 5 candidate — Supplementary 1

**Formal stem**: `figure5_supplementary_1_spatial_perturbation_response`.

### Estimand

Apply the same source-core Gaussian E-rate kick to the resolved baseline state and the state 100 ms
before operational runoff. For each frozen state,

\[
\delta x(t)=\exp(J_s t)B_E\,\delta r_E(0),
\]

where `J_s` is the state-specific rate-field Jacobian. This is one fixed input, not a separately
optimized perturbation at each time or state.

### Panel contract

- **a**: `Δr_E(x,y,t)` at 5, 15, 30 and 50 ms, baseline above pre-onset. All eight maps share one
  diverging scale. Up/down triangles mark the fixed source and remote endpoint.
- **b–c**: magnitude of the same response projected onto the scaffold axis as position × time. The
  dashed and dotted lines mark source and remote endpoint; they do not encode fitted wavefronts.
- **d**: whole-field response norm, normalized to the kick at `t=0`.
- **e**: cumulative remote/source response-energy ratio.

The threshold-based first-arrival regression remains in `time_response_summary.json`; it is not drawn
on the main canvas. It may support sequential recruitment when eligible, but does not by itself prove
a continuously travelling wavefront.

### Visual contract

- canvas width 7.2 in, 300 dpi; PDF text remains editable (`fonttype=42`);
- no super-title, grid, prose annotation or statistical paragraph on canvas;
- panel letters are lower-case and left aligned;
- baseline is neutral grey `#555555`; pre-onset is ochre `#C88719`; do not reuse the manuscript's
  red/blue template-A/template-B semantic colors;
- map colormap is `PuOr_r` with one symmetric scale; kymographs use one shared `magma` scale;
- only short reader-facing labels are allowed: `Time`, `Axis position`, `Response norm`, `Energy ratio`.

### Caption-safe claim

> The same localized perturbation decays near its source at baseline but recruits a broader axial
> response and accumulates more remote response energy 100 ms before operational runoff.

## 3. Figure 5 candidate — Supplementary 2

**Formal stem**: `figure5_supplementary_2_eigenmode_dynamics`.

### Estimand

At captured MZ timestamps, solve the frozen-q rate-field operating point, construct its Jacobian and
extract the leading eigenvalue and the non-negative E-loading of its real mode or complex-pair
invariant subspace. This is the instantaneous asymptotic mode; it is distinct from finite-time SVD
input `V1`, output `U1`, and the fixed-kick response in Supplementary 1.

For a complex pair with E loading `φ_E`, define `p(x)=|φ_E(x)|²`. The plotted metrics are:

- **Stability**: `Re λ`; zero is the linear-instability boundary.
- **Persistence**: `τ=-1/Re λ` for resolved stable states.
- **Axis**: signed second-moment anisotropy of `p`, aligned to the registered scaffold axis.
- **Globality**: `(Σp)²/(N Σp²)`.
- **Overlap**: normalized dot product with the preceding non-negative mode loading.

### Panel contract

- **a–d**: representative seed-1 mode fields at baseline, midpoint, −500 ms and −20 ms. All maps
  share one scale. The exact time to operational runoff is printed in the title.
- **e**: median `Re λ` with seed range across seeds 1/3/4.
- **f**: median damping time `τ` with seed range.
- **g**: median Axis, Globality and stepwise Overlap.

The runoff endpoint is not plotted as zero. If the frozen equilibrium is unresolved, it stays absent;
analysing the true transition requires a time-dependent tangent operator or a model with a resolved
post-transition attractor.

### Visual contract

- canvas width 7.2 in, 300 dpi, editable PDF text;
- no grid, long x/y labels, frequency panel, seed legend, claim text or fit annotation;
- mode fields use `magma` with one shared scale;
- Stability is purple `#7B3294`; Persistence and Overlap are teal `#2A9D8F`; Axis is ochre
  `#C88719`; Globality is grey `#555555`;
- uncertainty is a low-alpha seed range, not a confidence interval; the caption must call it a range.

### Caption-safe claim

> Along actual MZ z-only slow-state timestamps, the leading frozen-q rate-field mode becomes less
> damped and reorganizes from a nearly global loading to an axis-aligned loading before operational
> runoff; the resolved branch approaches but does not cross `Re λ=0`.

## 4. Shared claim boundary

Allowed:

- fixed scaffold, different slow states show different spatial susceptibility;
- the pre-runoff state supports a more persistent and spatially extended fixed-kick response;
- the leading frozen-q rate-field mode becomes axial and weakly damped before runoff.

Forbidden:

- calling the right-hand state an ictal or clinical seizure state;
- claiming a full interictal-to-seizure-to-recovery cycle;
- calling the fields full-SNN eigenmodes or full-SNN perturbations;
- treating the kymograph as proof of a continuous wavefront;
- declaring Hopf, fold, bistability or causal `z` mechanism from these two figures alone;
- treating operational runoff as seizure onset.

## 5. Reproduction and artifacts

- plotting-only entry: `scripts/paper_figures/plot_figure5_supplementary_spatial_dynamics.py`;
- scientific producer: `scripts/run_topic4_state_conditioned_susceptibility.py`;
- accepted numeric sidecars:
  `results/topic4_sef_hfo/state_conditioned_susceptibility/{time_response,eigenmode_timecourse}_arrays.npz`;
- accepted summaries:
  `results/topic4_sef_hfo/state_conditioned_susceptibility/{time_response,eigenmode_timecourse}_summary.json`;
- formal figures:
  `results/paper-ready-figure/fig5_mz_spatial_dynamics_supplementary/figures/`.

The plotting entry must fail if an accepted sidecar is absent. It may not silently rerun the SNN,
change the state labels, substitute a different candidate, or recompute an alternative estimand.
