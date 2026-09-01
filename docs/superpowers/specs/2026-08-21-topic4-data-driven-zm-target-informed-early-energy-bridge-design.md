# Topic 4 rev5: target-informed Z/M fit to the E1146 early-ictal energy field

Date: 2026-08-21
Status: execution contract
Branch: `codex/topic4-data-driven-zm-ictal-transition`
Relation to rev4: rev4 remains the model-internal discovery audit. Rev5 is a separate,
explicitly target-informed development analysis and may not be described as blind discovery.

## 1. Motivation and scientific question

Fig.4 provides an E1146 substrate learned from interictal events: the continuous node field,
topology, delays, virtual-contact geometry and local E-to-E/E-to-I redistribution. Earlier Fig.5
development runs showed that activating the per-neuron Z/M feedback can move the same network
from returned event-like activity into a broad high-activity state. They also showed that reducing
the expression of the learned E-to-I redistribution to roughly 2--5% can make this transition
look more globally oscillatory. That observation does not identify a Z/M mechanism: the E-to-I
dose changes the fast recurrent substrate and indirectly changes the inhibitory-current signal
that drives Z.

Rev5 therefore asks a narrower question:

> With the accepted data-driven substrate kept intact, can calibration of Z/M parameters alone
> produce a qualified model-ictal state whose onset-locked contact-energy gradient resembles the
> distribution observed across E1146 seizures?

The answer has two ordered parts:

1. the trajectory must first enter a sustained, broad and faster model state;
2. only qualified trajectories are compared with the patient early-ictal target.

The clinical target participates in work-point selection. The result is
`DEVELOPMENT_ONLY_TARGET_INFORMED_ZM_BRIDGE`, not prospective prediction, blind validation or a
patient-specific biological mechanism.

## 2. Mechanism identity

For E cells the active slow system is

```text
tau_z dz_i/dt = H(I_th - I_i^I) - z_i
dm_i/dt       = -m_i/tau_m + sum_k delta(t-t_i^k)
I_net,i       = I_E,i - z_i I_I,i - eta_m m_i.
```

`I_th_EI` is the historical configuration name. Operationally it is compared with inhibitory
current received by E cells; it must not be confused with the learned E-to-I AMPA pathway.

The learned E-to-I dose multiplies the coefficient row that redistributes existing E-source
weights onto I targets. It preserves topology, delays and target-wise incoming E budget. A 5%
dose is therefore 5% expression of the learned spatial redistribution, not 5% of E-to-I synapses
or total inhibitory strength.

## 3. Frozen substrate

The primary rev5 arm freezes all of the following to the accepted Fig.4 substrate:

- E/I positions and cell labels;
- continuous node field `h` and reconstructed thresholds;
- AMPA/GABA topology and delay assignment;
- learned E-to-E and E-to-I coefficient rows at dose `1.0`;
- target-wise incoming pathway budgets;
- spatial OU drive and seed construction;
- exact 15-contact montage, names, shafts, coordinates and order;
- LFPRecorder current-proxy definition.

The historical 2% and 5% E-to-I trajectories remain labelled calibrated-edge comparators. They
may be rescored and visualized, but they cannot win the primary Z/M-only fit.

## 4. Parameters that may be fitted

Only four slow coordinates may vary:

```text
s_I       = I_th / 95.19851312666987
tau_z     = Z recovery time constant
tau_m     = M adaptation time constant
G_m       = eta_m * tau_m
eta_m     = G_m / tau_m
```

Primary bounded development ranges are:

```text
s_I       in [0.65, 1.05]
tau_z     in [2500, 10000] ms
tau_m     in [125, 1000] ms
G_m/G_m0 in [0.25, 2.0],  G_m0 = 0.007451594355587098 * 500
```

Initial conditions remain `z=1`, `m=0`. Background drive, node field, edge doses, detector and
readout are not optimizer variables. Parameter values are represented in log space except
`s_I`. Existing completed Z/M-only candidates are rescored before any new simulation.

## 5. Patient early-ictal target

### 5.1 Contact and spectral contract

The clinical feature follows the Fig.3 contract:

- E1146, CAR reference;
- exact-name join to all 15 frozen contacts;
- `1--150 Hz` log power;
- 1 s spectral window and 0.5 s hop;
- per-contact robust-z against EEG-onset `[-120,-90] s`;
- `[-10,0] s` pre-onset and `[0,10] s` early-ictal vectors;
- no rank transform, sign flip, mirror selection, contact subset or axis refit.

There are 25 complete exact-band E1146 seizures in the current checkpoint inventory. Seizure 2
is retained only as the accepted display example. The other 24 seizures form the development
target distribution. The producer must freeze the exact inventory and hashes rather than rely on
the number written in prose.

### 5.2 Frozen target summaries

For every contact and endpoint the target stores the 24-seizure values, median, bootstrap
interval and robust scale. It additionally stores:

- global median robust-z energy;
- positive-contact fraction;
- across-contact IQR;
- exact contact vector;
- shaft-balanced contact vector;
- demeaned early-minus-pre increment;
- TA-axis slope and early-four minus late-four contrast;
- leave-one-seizure-out field agreement;
- split-half reliability.

Patient `10--150 Hz` is a required sensitivity. Seizure 2 values
`shared_a_signed=0.719127` and direct early-rank correlation `0.570884` are parity checks only.

The numerical model-versus-patient loss uses the matched `10--150 Hz` patient target. The
Fig.3-native `1--150 Hz` target remains the clinical primary display and a required sensitivity;
it is not directly subtracted from a model vector computed in a different band.

## 6. Model-ictal qualification

Rev5 reuses `MODEL_ICTAL_ELIGIBLE_V2`. Patient data cannot rescue a failed trajectory.

Let `t_ictal` be 100 ms before the unchanged 120 Hz/100 ms operational detector. A trajectory is
eligible only if the complete V2 early interval `[t_ictal+100, t_ictal+1100] ms` exists and:

1. at least 80% of post-onset 20 ms windows have both `F_E >= 0.5` and
   `F_sheet >= 0.5`;
2. median 20 ms-smoothed population E rate is at least twice its paired, same-seed Z/M-off
   low-state reference;
3. over `[t_ictal+500, t_ictal+1000] ms`, median contact spectral centroid rises from the paired
   reference by at least 5 Hz and by a factor of 1.25;
4. no non-finite state or simulator failure occurs through the analysis interval.

The 70/80/90% duty, activity thresholds, bin sizes, onset shifts, population frequency and
current-proxy amplitude remain mandatory sensitivities. A rate crossing alone is not eligible.

## 7. Model baseline and energy readout

### 7.1 Paired baseline

Each network/noise seed has one Z/M-off reference using the identical substrate and stochastic
construction. Non-overlapping 500 ms windows from the reference form a per-contact log-power
baseline. The same median/MAD robust-z transform is applied to every Z/M candidate sharing that
seed. A candidate may not estimate its own baseline from a pre-transition interval already on the
Z/M buildup trajectory.

### 7.2 Primary model band

The quantitative model band is `10--150 Hz`, using the current-based 15-contact recorder and a
500 ms Welch estimate. The signed `30--80 Hz` trace is display-only. Sensitivities use
`1--150 Hz` where the saved duration and detrending support it, plus firing-density contact and
population-rate spectra to ensure that signed-current cancellation is not mistaken for absent
network oscillation.

### 7.3 State-defined readout time

The patient target does not choose a frame. Starting at `t_ictal`, scan 500 ms windows on a fixed
25 ms grid and select the earliest complete window satisfying the model-internal broad-state and
frequency clauses. This is `W_read`. If none exists, the trajectory is not bridge-evaluable.

The plotted field is the mean contact log power over `W_read`; the plotted activity snapshot is
the centre of `W_read`. All candidate comparisons use the same algorithm. The full onset-locked
time course is retained so a transient maximum cannot be hidden.

## 8. Target-informed loss

Only model-ictal eligible trajectories receive a finite bridge score.

### 8.1 Energy burden

`D_energy` compares model and patient distributions of global median robust-z, positive-contact
fraction and contact IQR. Each term is scaled by the patient bootstrap IQR with a nonzero
split-half floor. For the discrete positive-contact fraction, the scale floor is one contact
(`1/15`); this prevents an all-positive patient sample from creating a zero-denominator loss
while retaining the exact observed fraction difference.

### 8.2 Contact energy field

For contact `i`, let `P_i` be robust-z energy in `W_read` and `T_i` the 24-seizure target median.
The primary contact error is

```text
e_i = abs(P_i - T_i) / max(IQR_bootstrap(T_i), floor_i, epsilon).
```

Errors are averaged separately within ICL and SCL and combined with `LSE_0.25`, so the eleven ICL
contacts cannot hide failure on four SCL contacts. Exact-contact Spearman, cosine similarity,
TA-axis slope and early-minus-late contrast are continuous diagnostics; no mirror or sign
reselection is allowed.

### 8.3 Transition increment

For both model and patient, subtract the contact median from the early-minus-pre vector. Compare
the two vectors with patient-bootstrap-scaled L1 distance. This prevents a candidate from winning
only because the static interictal axis was already present before onset.

### 8.4 Time-course diagnostic

`D_time` compares the normalized progression of global energy, TA similarity and spatial
increment over fixed pre/early landmarks. It does not match patient seconds to model
milliseconds. Because this rev5 experiment is authorized specifically to fit the *early* energy
gradient, `D_time` is a frozen post-selection diagnostic and does not enter work-point selection.
It may motivate a later lifecycle experiment but cannot silently change the rev5 winner.

### 8.5 Combined objective

```text
J_field        = LSE_0.25(D_contact, D_increment)
J_early_bridge = mean(D_energy, J_field)
                 + LSE_0.25(D_energy, J_field)
                 + R_robustness
```

Model-ictal failure and missing readout windows receive no finite score. Seed robustness is
implemented as the predeclared 2/3 selection and confirmation requirement rather than an
arbitrary additive penalty. Onset sensitivity is reported after freezing. The score never rewards
higher absolute patient correlation alone. Every component and unscaled vector is saved.

## 9. Fit, selection and confirmation

### Stage 0: zero-simulation rescore

Rescore all completed full-dose Z/M candidates and the frozen 2%/5% comparators. This establishes
whether a new run is necessary and validates the readout producer.

### Stage 1: Z/M-only fit canary

Use common random numbers on the frozen full-dose substrate. Start with a deterministic local
design around the best existing full-dose point, prioritizing the previously underexplored
`s_I x tau_z` plane. Do not launch a high-dimensional optimizer before the local design yields at
least one model-ictal eligible candidate.

If the first plane separates the requirements rather than satisfying them, freeze one bounded
crossing refinement using only model-internal evidence. The observed crossing is: `tau_z=2500 ms`
passes the frequency clause but misses one-second recruitment duty, whereas slower `tau_z` can
sustain recruitment but loses the frequency increase. Therefore the only authorized refinement
holds `s_I=0.8`, `tau_z=2500 ms` and scans `tau_m={250,500,1000} ms` by
`G_m/G_m0={0.5,1.0,1.5}`, with `eta_m=G_m/tau_m`. Patient bridge scores may not choose this grid
or stop individual cells.

The bounded grid may add exactly one three-cell corner-cross at `tau_m=62.5 ms` with the same
three `G_m` ratios if, and only if, the initial nine cells produce no eligible point while the
historical `tau_m=62.5, tau_z=5000 ms` point passes recruitment duty and the `tau_z=2500 ms`
points pass frequency. This combines two model-internal one-clause successes; it is not selected
from patient scores. Failure of this final cross closes Z/M-only fitting at full learned-edge
expression for rev5.

### Stage 2: selection

Run the top eligible Z/M-only candidates on a predeclared selection seed set. Rank by median
`J_early_bridge`, then worst-seed `J_early_bridge`, then distance from the exact Z/M reference. A candidate
must remain model-ictal eligible on most selection seeds. The patient target is fixed throughout.

### Stage 3: frozen confirmation

Freeze the winning parameters and `W_read` algorithm before running new confirmation seeds.
Confirmation reports the complete score distribution and may not trigger parameter retuning.

The 2% and 5% edge-dose arms are historical comparators, not eligible winners. If only those arms
match the requested morphology/field, the result is
`ZM_ONLY_FIT_INSUFFICIENT_EDGE_EXPRESSION_DEPENDENT`.

## 10. Selection-aware controls

Before interpreting the minimum fitted loss:

1. rerun the complete candidate/window selection against contact-permuted patient targets;
2. use within-shaft permutations and spatial-gradient-preserving surrogate targets;
3. compare against static-axis amplitude-only and uniform-energy model controls;
4. report the null distribution of the minimum `J_early_bridge`, not only the selected candidate's
   nominal correlation;
5. compare the winner with 2%/5% edge-dose comparators without allowing those comparators to alter
   the primary Z/M-only conclusion.

## 11. Figure and movie contract

The primary Fig.5 candidate retains the accepted visual grammar:

- continuous 15-contact readout with global recruitment strip and a visible transition;
- projected Z/M trajectory;
- exact interictal event field beside the `W_read` early model-ictal energy field;
- model-versus-patient target field and per-contact comparison;
- no clinical units on model current and no clinical seizure label.

A companion GIF uses the prior qI/gK three-panel grammar:

```text
Z/M slow-state map | 2D E-neuron activity | uninterrupted 15-contact readout
```

The GIF must show pre-transition returned events, transition, broad recruitment and at least
1 s of the qualified state. It is generated for the frozen Z/M-only winner and the 2%/5%
comparators from their own uninterrupted trajectories.

## 12. Required artifacts

```text
results/topic4_sef_hfo/data_driven_zm_ictal_transition/target_informed_bridge_v1/
  clinical_target.json
  clinical_target_vectors.npz
  target_provenance.json
  existing_candidate_rescore.{json,csv}
  fit_manifest.json
  fit_results.{json,csv}
  selection_results.json
  WORKPOINT_TARGET_INFORMED_FROZEN.json
  confirmation_results.json
  selection_aware_null.json
  final_report.md

results/paper-ready-figure/fig5_target_informed_zm_bridge/figures/
  fig5-target-informed-zm-bridge.{png,pdf,svg}
  fig5-target-informed-zm-bridge.gif
  fig5-target-informed-zm-bridge-metadata.json
  README.md
```

Every result records git commit, dirty state, source hashes, patient inventory, contact order,
seed role, Z/M parameters, edge doses, onset/readout windows and score version.

## 13. Interpretation and stop rules

| Result | Interpretation/action |
|---|---|
| no full-dose Z/M candidate becomes model-ictal | close primary fit; do not optimize patient field on non-ictal trajectories |
| model-ictal but field loss remains at selection-aware null | model state found; patient early-energy bridge unsupported |
| field fits but increment/time fail | persistent static scaffold or amplitude fit, not transition organization |
| Z/M-only winner confirms | development-stage target-informed Z/M bridge candidate |
| only 2%/5% comparator succeeds | bridge depends on attenuating learned E-to-I redistribution |
| confirmation loses model-ictal morphology | work point unstable; no final Fig.5 freeze |
| no unseen patient/seizure unit | retain development-only, non-generalization language |
