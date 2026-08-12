# Topic 4 rev10-D4.1: fresh-network forced route dose confirmation

## Scientific question

D4 found one uniform-grid source at `(18, 6) mm` that produced clean mode A responses in all three development networks. D4.1 asks whether that route-access result transfers to fresh network realizations, and what minimum synchronized packet scale is required. It does not test spontaneous initiation.

## Frozen comparison

- A source: `(18, 6) mm`, selected by D4 and frozen before the fresh networks.
- B source: `(2, 14) mm`, the exact geometric mirror around the sheet center and a D4 clean-B source.
- Packet fractions: `0.000625`, `0.00125`, `0.0025`, `0.005` of E, approximately 20, 40, 80, and 160 neurons.
- Fresh network seeds: `1231-1236`.
- Node field: frozen `v62_density_t050`; all edge and dynamic-state mechanisms off.
- Detector and patient-training direction classifier remain frozen.

Each network uses one sham and resets the same RNG state for every source-dose trial. The direct injected frame is removed before electrode classification. Zero response is valid data.

## Primary decision

For a source-dose response to be clean it must have bit-identical pre-trigger spikes, no runaway, a returned detector-qualified event, recruitment on both shafts, patient-support OOD pass, and the pre-frozen expected A/B label.

The smallest packet dose is accepted only when both sources pass in at least `5/6` fresh networks at that same dose. Failure at `0.005` downgrades D4 to a development-network localization. Success confirms forced route capacity across network realizations, not spontaneous interictal activity.

## Claim boundary

The A source was selected using the D4 patient-labeled outcome map, so this is development-only confirmation. A forced packet is an intervention and cannot replace the accepted Fig.4 direct-waveform and KMeans pair. Those figures can be replaced only after an observation-invariant continuous mechanism generates both modes spontaneously within the same network pool.

## Frozen secondary timing audit

The primary run returned `5/6` clean A and `4/6` clean B responses at 160 E neurons, while both sources had `6/6` correct directions and `6/6` joint-shaft recruitment. The preregistered verdict remains `NOT_CONFIRMED`. A secondary audit therefore reruns only sham+A+B at the already frozen maximum dose, saves every detector interval and active-fraction trace, and asks whether the missing formal responses occurred after the 40 ms latency window or overlapped sham events. This audit may attribute the failure but cannot alter the primary verdict or select a new dose/source.

### Timing audit outcome

The two missing B responses both began at `141 ms`, exactly 1 ms beyond the frozen 100-140 ms window. In the full paired response window, A and B each had `6/6` returned responses, `6/6` expected directions, and `6/6` joint-shaft recruitment; none of the 12 responses overlapped a sham event. A had `5/6` patient-support passes because seed 1235 lay marginally outside the frozen OOD boundary. The formal D4.1 verdict therefore remains `REV10D4_1_FRESH_NETWORK_FORCED_AB_ROUTE_NOT_CONFIRMED`, while the secondary mechanism status is `REV10D4_1_FORMAL_GATE_FAIL_BUT_BROAD_ROUTE_TIMING_SUPPORTED`.

This supports an exploratory accessibility experiment: the scaffold can carry either direction when synchronously initiated, but no evidence yet shows that an endogenous or continuous stochastic mechanism reaches those route basins. The next experiment must be statistically translation invariant and must not center a core or drive patch at the D4 outcome-selected source.
