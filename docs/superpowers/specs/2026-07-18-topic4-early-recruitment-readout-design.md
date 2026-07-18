# Topic 4 early-recruitment readout — shared infrastructure design

Date: 2026-07-18
Status: implementation design v1
Scope: M2 reduced rate-field adapter first; later adapters may consume the collaborator `m+z` SNN without changing the readout contract.

## 1. Scientific question

The readout asks one narrow question:

> Does the spatial order expressed by a self-limited interictal perturbation predict the spatial distribution of early post-critical recruitment energy on the same fixed scaffold?

It does not classify a complete seizure, explain termination, or upgrade a runaway trajectory to a clinical ictal event. M2 continues to own `linear_ignition` and `nonlinear_spread`; this module adds a third, orthogonal measurement: `interictal_arrival_to_early_energy`.

## 2. Three objects that must remain separate

1. **Interictal reference field**: a physical core kick on a qualified stable low branch. At each source location or contact, arrival is the first half-peak crossing of the positive kick-minus-control response.
2. **Early recruitment field**: the mean squared positive kick-minus-control E-rate in a fixed early window after the same physical kick at `at_crossing` or the separately labelled `pre_runaway` state.
3. **Critical eigenmode**: retained as M2's ignition-locus diagnostic. It is not substituted for either dynamic field above.

The primary early window is 10–50 ms, centered on the established HFO-scale non-normal response whose axial elongation peaks near 30 ms. Windows 5–30 ms and 10–75 ms are sensitivities. Window selection is configuration, never chosen from the resulting correlation.

## 3. Signal and eligibility contract

For kicked and no-kick control trajectories at the same operating point,

\[
\Delta r_E(x,t)=r_E^{kick}(x,t)-r_E^{control}(x,t),\qquad
a(x,t)=\max(\Delta r_E(x,t),0).
\]

The interictal arrival time is

\[
\tau(x)=\min\{t:a(x,t)\geq 0.5\max_t a(x,t)\}.
\]

A location participates only if its peak exceeds both an absolute numerical floor and 10% of the largest spatial peak. The early-energy proxy is

\[
E_{early}(x)=\operatorname{mean}_{t\in W} a(x,t)^2.
\]

This is an **excess-rate energy proxy**, not simulated broadband LFP power. SNN adapters may later supply a current-based or spectral signal while retaining the same arrival/energy/comparison schema.

If numerical escape/saturation occurs before the configured window ends, that window is ineligible and its energy field is not scored. A truncated runaway is never treated as an early-field positive.

## 4. Spatial levels

The same calculation is performed twice:

- **source space**: all M2 grid cells;
- **contact space**: both dynamic fields are passed through the same Gaussian sampler and the accepted E1146 subject-SNN montage/contact order. No parametric cross montage is allowed in the subject-facing output.

The M2 grid is registered to the accepted E1146 `template_source` sheet by one fixed similarity
transform. The M2 perturbation core maps to the source-A focus; a preregistered point 2.5 mm along
the M2 E→E axis maps to the source-B/sink endpoint. The transform is applied to coordinates only;
field values and dynamics are not interpolated or refit from the resulting association. This is a
reduced-field observation adapter, not a claim that the full E1146 SNN has already produced an ictal
trajectory.

All comparisons use the intersection of finite interictal arrival, finite target energy, and the reference participation mask. A source-excluded sensitivity removes cells or contacts directly overlapping the perturbation core.
At contact level, "direct core" is defined from the transformed binary M2 core sampled through the
same observation kernel (`core_contact_loading >= 0.5`). It is deliberately not defined by the three
E1146 template-source contact labels: those labels mark the clinical/template source region in the
mechanism panel, whereas the loading mask measures actual injection contamination in this adapter.

## 5. Statistics

For each target state and window, report:

- `arrival_energy_spearman`: expected negative when earlier locations are hotter;
- `earliness_energy_spearman = -arrival_energy_spearman`: expected positive;
- standardized field cosine between `-arrival` and energy;
- overlap between the `k` earliest reference locations and `k` highest-energy target locations;
- usable support size and variance/degeneracy status.

Contact-space permutation controls:

1. unrestricted contact shuffle;
2. within-shaft shuffle, preserving each shaft's energy multiset.

Both nulls permute the target energy field while leaving the interictal arrival field fixed. P-values are one-sided for positive `earliness_energy_spearman`. The null result is reported, not used to hide an observed effect that has the wrong sign.
When the full constrained permutation space contains at most 50,000 labelings, it is enumerated
exactly; larger spaces use a fixed-seed Monte Carlo estimate and report that method explicitly.

## 6. M2 adapter

The adapter reuses M2's actual slow-space bracket and frozen-Jacobian machinery:

- primary interictal reference: `last_qualified_low`;
- reference sensitivity: `first_qualified_low`;
- target 1: the localized `at_crossing` operating point;
- target 2: `pre_runaway` at fraction 0.85 of the same bracket; this is a visualization probe that recruits the full accepted E1146 montage before numerical escape, not a complete seizure;
- primary perturbation: positive physical `core_kick`, `eps_rel=0.05`;
- negative polarity and critical-eigenvector kicks remain controls outside the primary readout.

M2 continues to localize the slow-state crossing on its frozen 6×6 diagnostic grid. Because that
grid leaves fewer than three recruited cells outside the direct perturbation core, the same slow
coordinates are re-solved on a minimally refined 8×8 grid for the spatial readout. The adapter
reports both grid sizes and recomputes `alpha1` on the readout grid. This is a resolution sensitivity
of the M2 state coordinate, not a replacement or silent reinterpretation of M2's 6×6 eigenmode.

The output is a JSON summary plus an NPZ carrying times, dynamic fields, arrival maps, energy maps, masks, coordinates, and contact names. The adapter must not modify M2's existing ignition/spread verdict.

### 6.1 M3 runaway adapter

The figure-facing runaway field is not taken from M2 `pre_runaway`. A separate computation producer
reuses the accepted E1146 M3A-v2.1 continuous `q_I build-up → runaway` protocol exactly
(`k_q=0.10`, `q_min=0.05`, `kick_boost=5.0`, `r_kick=0.6`, `T=1500 ms`, seed 1). It exports the
continuous virtual-LFP trace, pulse schedule, q_I summaries, the operational runaway onset, and
contact-space fields without changing the SNN engine.

Runaway onset retains the accepted operational definition: the first 100-ms interval in which at
least 80% of samples of the 20-ms-smoothed E rate exceed 120 Hz. This is the observed transition on
a q_I-depletion trajectory; the current artifact does not independently solve an analytic
separatrix value `q_I*`.

The early-runaway energy window begins at that onset and ends at the earliest of onset+100 ms, the
next scheduled external pulse, or the trace end. For the accepted trajectory this is
1109.8–1209.8 ms; the next pulse is at 1210.0 ms. Contact energy is the mean squared positive excess
absolute virtual-LFP relative to the pre-runaway median.

## 7. Figure 5 candidate contract

The M3 onset-locked figure is registered as the current Figure 5 candidate. Its
paper argument is `same scaffold, different state`: one interictal-like event
expresses a contact order on the fixed E1146 scaffold, and the early operational-
runaway window expresses a concordant energy gradient on the same continuous
trajectory. Candidate status does not promote runaway to a clinical seizure or
close termination/recovery. The shorter canonical paper-facing contract is
`docs/fig5_snn_state_readout_spec.md`.

The computation runner writes no figure. A plotting-only producer consumes its M3 artifact on the
accepted E1146 montage and does not label panels A/B/C:

1. **Top, full width — one continuous build-up-to-runaway recording.** Display the full 0–1500 ms
   virtual-SEEG trace. Shade only the one pre-runaway event actually used by the lower-left field;
   do not draw pre-runaway peak points or propagation lines. Mark operational runaway onset with a red dashed line and the
   energy averaging interval with one light-red span. Display the signed 30–80 Hz component so the
   approximately 50-Hz burst cycles remain visible. Scale each contact by its own pre-runaway 95th
   percentile absolute amplitude; exclude runaway from that display scale and allow it to clip.
   Put the legend in a dedicated row rather than inside the trace axes.
2. **Bottom left — single-event interictal recruitment order.** Select the endpoint source of the
   last scheduled pulse before runaway, then take its last qualifying local event
   (`5<=peak_hz<=120`, `active_frac>=0.02`, before onset−20 ms). Within that exact shaded window,
   rank each finite virtual contact by its 30–80 Hz burst-envelope peak latency. Display the
   resulting `1..N` contact rank with continuous min–max interpolation and `viridis`
   (dark=early, yellow=late); the colorbar maximum is the participating contact count, not
   milliseconds. Keep the accepted E1146 registered montage fixed and draw all 15 electrodes with
   one black-outline convention. Neuron grain uses first-spike order from the identical event, not
   a multi-event median. Keep the colorbar close to its own field panel.
3. **Bottom right — early-runaway energy.** Display onset-locked mean contact excess virtual-LFP
   energy using `Blues` (dark=high). It is a static window statistic and carries no zigzag. Use all
   15 finite energy contacts on the identical fixed montage and extent; restore raw model energy on
   the colorbar rather than showing only normalized 0–1 values. Use the same black electrode rim
   and keep this colorbar close to its own field. The association statistic remains restricted to
   contacts with finite values in both panels (15 in the selected baseline run).

Both lower fields reuse the paper Fig3-B paired-field grammar: equal square panels, the identical
fixed E1146 registered plane, complete montage and extent. Because the model sheet spans only
20 mm, the data figure's 6-mm display kernel is rescaled to 3.0 mm rather than copied literally.
Contact-space Gaussian interpolation retains a smoothly fading sampling-confidence alpha; a hard
polygon/source-support edge is forbidden. No shaft connectors or core rings are drawn.

The smooth wash and granular layer have different provenance. The wash is the virtual-SEEG contact
readout. The grain comes directly from the same run's `E_spk_bool` and all 8,000 E-neuron positions:
gray points show the simulated substrate; left colored points show first-spike latency for
neurons firing in the identical single interictal event; right colored points show per-neuron firing rate
inside the identical early-runaway window. Contact interpolation must never be sampled back onto
neurons and presented as neural activity. No GIF is produced.

For the accepted trajectory the transition-side source is tempB and the earlier reference responses
occur at 265 and 535 ms. The shared-contact earliness–energy Spearman is descriptive only: it is a
single conditioned trajectory on one observation layer, not a cohort p-value or an independent
mechanism test. The M2 10–50 ms association JSON remains a separate numerical diagnostic and must
not be conflated with this M3 onset-locked figure.

## 8. Interpretation matrix

- Positive at source and contact levels, stable across windows: model-side spatial bridge candidate.
- Positive only in source space: projection/geometry bottleneck; no model-to-SEEG bridge.
- Positive only after including the direct core: injection-location confound; propagation inheritance not established.
- Sensitive to window or perturbation magnitude: descriptive only.
- Ineligible because the early window saturates: operating point/readout unresolved, not a negative correlation.
- No association under a clean window: the current M2 transition does not explain the empirical interictal-to-early-energy relation.

No threshold in this infrastructure promotes the result to a seizure-mechanism claim.
