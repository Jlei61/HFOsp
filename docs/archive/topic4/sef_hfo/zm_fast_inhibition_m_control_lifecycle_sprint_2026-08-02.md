# Z/M fast inhibition × M × finite control lifecycle sprint

**Date:** 2026-08-02  
**Final status:** `NO_GO_FULL_ICTAL_LIFECYCLE`  
**Accepted deliverable:** `ACCEPT_BOUNDED_FAST_INHIBITION_M_CONTROL_MAP`

## 1. Plain-language verdict

On the original seed-1 Z/M spiking substrate, fast inhibitory dynamics can
change the high-activity branch, M can reduce or prevent some high-activity
episodes, and a finite threshold pulse can suppress firing almost completely
during the pulse.  None of these interventions produced a credible controlled
ictal lifecycle.

The missing object remains a sustained, bounded, non-tonic and spatially
structured ictal carrier with a reachable recovery basin.  The observed states
were either a near-synchronous relaxation-burst train or a spreading tonic
plateau.  M converted part of the burst-train family into a lower-density burst
train; finite control produced an acute dip followed by return to the same
train.

## 2. Fixed substrate and tested mechanisms

The pathological anisotropic E→E graph, weights, orientation, original Z
equation, S_G implementation, noise law, pre-entry checkpoint and virtual-SEEG
geometry were unchanged.

This line tested only the inhibitory/exit side:

1. presynaptic I→E depression, applied at inhibitory-spike emission before the
   delayed weighted event is enqueued;
2. inhibitory-neuron threshold adaptation;
3. the existing M current at `g_M in {0,1,3,10,30}` and
   `tau_M in {0.5,2} s`;
4. a finite all-E threshold uplift, without state reset or parameter switching.

## 3. Fast inhibitory phase map

The pre-registered 36-cell inventory ended with 16 completed full-dynamic
trajectories, eight adaptive cancellations and twelve cells not run after the
adaptive stop.  The completed set was balanced between I→E-depression-only and
combined I→E-depression plus I-adaptation arms.

- 5/16 were classified as `relaxation_burst_train`;
- 11/16 were classified as `spreading_plateau`;
- no completed condition produced a sustained high-energy state with both low
  common-mode dominance and continuous virtual-SEEG occupancy.

Thus fast inhibitory state is accepted as a branch-control coordinate, but not
as an ictal-carrier solution.  The stage acceptance and implementation
corrections are separately archived in
`fast_inhibitory_phase_map_acceptance_2026-08-02.md`.

## 4. M response surface

All 36 promoted conditions completed: four fast phenotypes crossed with nine M
coordinates.

- Weak or fast M generally left the persistent state intact.
- `g_M=10, tau_M=2 s` produced three paired episode-offset candidates at
  5.95--7.98 s and deep-gap burst tails.
- Only one of these contained any returning-event candidate; none recovered the
  pre-event distribution.
- Strong slow M (`g_M=30, tau_M=2 s`) prevented onset in two I→E conditions,
  but failed to terminate the high-common-mode combined conditions.

The correct interpretation is phenotype-dependent movement toward an exit
direction, not a universal M termination mechanism.  The same M coordinate can
be prevention for one fast branch and ineffective for another.

## 5. Forty-five-second native follow-up

The closest condition (`I→E tau_D=439 ms, d*=0.8227, g_M=10,
tau_M=2 s`) was extended to 45 s.

The macroepisode detector marked an onset at 0.50 s and a sustained-density
offset at 5.95 s.  Visual and event-level inspection showed that bursts
continued throughout the full 45 s.  The result was a dense-to-sparse burst
train transition, not ictal termination followed by a postictal corridor.

After the detector's offset:

- 74 event candidates occurred;
- 40.5% individually matched the frozen 15-event interictal reference;
- median duration, peak, participation, core/surround ratio and IEI were within
  the development tolerances;
- the pre-registered distribution-recovery decision still failed;
- M remained high at 70.25 mV-equivalent at the end, Z ended at 0.588 after a
  maximum of 0.670, and the I→E resource ended at 0.836.

Therefore this is not return to the original interictal working point.  It is
evidence that the current detector's `durable_offset` means offset of a
sustained high-density macroepisode; it must not be called seizure termination
when a burst train continues.

## 6. Finite-control result

The original control-clock bug was corrected before the accepted panel: the
manifest time is relative to the pre-entry checkpoint, whereas the resumed SNN
uses the checkpoint's absolute engine time.  All accepted artifacts use
`relative_to_pre_entry_checkpoint_v2`; the five earlier unversioned artifacts
are invalid engineering artifacts.

Calibration found `u_ref=4 mV`.  In the six-cell dose panel, all controlled and
uncontrolled traces were exactly identical before the pulse.

| dose | duration | paired core reduction | longest global zero-rate dwell | durable exit |
|---:|---:|---:|---:|---:|
| 0.5 u_ref | 50 ms | 41.6% | 2 ms | no |
| 0.5 u_ref | 200 ms | 38.9% | 30 ms | no |
| 1.0 u_ref | 50 ms | 52.4% | 2 ms | no |
| 1.0 u_ref | 200 ms | 45.8% | 60 ms | no |
| 1.5 u_ref | 50 ms | 99.9% | 42 ms | no |
| 1.5 u_ref | 200 ms | 99.96% | 192 ms | no |

The strongest 200-ms pulse exceeded the 100-ms silencing guard but still did
not create a durable offset.  Activity returned to the same relaxation-burst
train.  The negative result is therefore not explained by insufficient control
amplitude: acute suppression did not reach a persistent recovery basin.

## 7. Engineering and provenance closure

- emission-time I→E resource/delay semantics are locked by a deterministic
  two-spike delayed test;
- conditionally active state inventory includes I→E resource and inhibitory
  adaptation only when enabled;
- workers write heartbeat, peak RSS and durable terminal receipt;
- `ranking_v2` replaces the historical scalar ranking for candidate selection;
- resumed control uses the corrected absolute engine clock;
- immutable matching artifacts are now validated and reused instead of being
  reported as failed overwrite attempts;
- the unified ledger contains 84 unique configurations: 64 successful runs,
  eight adaptive cancellations and twelve pre-registered not-run cells;
- stage accounting is fast 36, M 36, calibration 5, dose 6 and native long-run 1;
- 88 targeted Z/M lifecycle, control, checkpoint and ledger tests pass;
- no lifecycle simulation process remains.

## 8. Core artifacts

- `results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint/figures/fast_inhibitory_phase_map.png`
- `results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint/figures/m_response_surface.png`
- `results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint/figures/control_dose_response.png`
- `results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint/figures/trajectory_i2e__tauD439__d0.8227__s1__T45s__gM10__tauM2000.png`
- `results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint/unified_run_ledger.json`
- `results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint/native_long45_analysis.json`

The figures directory contains the required Chinese `README.md` inventory.

## 9. Claim boundary and next decision

Safe claim:

> On the unchanged Z/M SNN substrate, fast inhibitory state selects between a
> relaxation-burst and spreading-plateau branch.  M and finite threshold control
> can reduce high activity, but the tested system has no demonstrated path from
> a data-consistent ictal carrier through durable offset back to the original
> interictal distribution.

Forbidden claims include controlled seizure termination, autonomous ictal
lifecycle, postictal recovery, or clinical stimulation efficacy.

Further M-gain or threshold-uplift sweeps should stop.  The next mechanism must
first change the fast carrier itself: preserve continuous macro-energy while
breaking the current global common mode and avoiding the all-off gaps of the
relaxation train.  Exit and control should only be reopened after such a carrier
exists on the unchanged pathological scaffold.
