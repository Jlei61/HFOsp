# Topic 4 MZ-divisive lifecycle: current-based Z/M + dynamic recurrent-gain pool

**Status:** execution complete; Stage-1 containment clean no-go, Stage 2 not opened by stop rule

**Branch:** `codex/topic4-mz-divisive-lifecycle`
**Result root:** `results/topic4_sef_hfo/mz_divisive_lifecycle/`

Final report: `docs/archive/topic4/sef_hfo/mz_divisive_lifecycle_pilot_2026-07-19.md`.

## 1. Scientific question

The existing current-based MZ model can move the fixed E1146 scaffold from returning interictal-like
events toward prevention, a bounded elevated plateau, or operational runaway. The existing M4 shared
pool can convert runaway into a bounded sustained third state, but that state does not terminate.

This branch tests one narrow conjunction:

> Can a fast activity-dependent divisor stabilize the Z-mediated escape into a finite recruited state,
> after which the slower spike-driven M variable removes that state and returns the same autonomous
> trajectory to its original interictal basin?

This is not an Abbott/Liou replication. It is a current-based mechanism-composition test. The parallel
`topic4-mz-conductance` branch exclusively owns conductance Z/M, global-GABA topology, and optional
dynamic threshold `phi`; this branch must not implement those mechanisms.

## 2. Locked equation

For E cells,

\[
\tau_E \dot V_i = -V_i + I^{E,ff}_i
 + \frac{I^{E,rec}_i}{1+\alpha_G S_G}
 - z_i I^I_i - \eta_m m_i .
\]

For I cells, the membrane remains unchanged:

\[
\tau_I \dot V_i = -V_i + I^E_i-I^I_i.
\]

The existing per-neuron slow variables are reused exactly:

\[
\tau_z\dot z_i=H(I_{th}^{EI}-I^I_i)-z_i,
\qquad
\dot m_i=-m_i/\tau_m+\sum_k\delta(t-t_i^k).
\]

The existing M4 pool is reused:

\[
\Psi(r)=\frac{[r-r_0]_+^n}{r_{50}^n+[r-r_0]_+^n},\qquad
A_G=\langle\Psi(r)^p\rangle^{1/p},
\]

\[
\tau_\mu\dot\mu_G=-\mu_G+A_G,\qquad
\tau_S\dot S_G=-S_G+S_{max}\mu_G.
\]

`S_G` divides recurrent excitation only. It does not divide feed-forward excitation, signed net current,
or inhibitory current. `z` and `m` act on E cells only. The first version locks `beta_SG=0` and adds no
`q_I`, `g_K`, `h_G`, conductance shunt, STD, or `phi`.

## 3. Division of dynamical roles

- `z`: slow access variable. Event-driven loss of inhibitory efficacy must move the trajectory toward
  the recruited basin without changing parameters or applying a kick.
- dynamic `S_G`: fast containment variable. It may stabilize a bounded high branch or an oscillatory
  high-state envelope, but is not presumed to terminate it.
- `m`: slow exit and refractory variable. It is only tested after an m-off bounded high state exists.

The desired object is not necessarily a permanent seizure limit cycle. The stronger admissible result is
a frozen-slow-state bounded high attractor or bursting orbit that exists over a finite Z/M window, with
the slow trajectory crossing distinct entry and exit boundaries.

## 4. Two pre-registered pool sensors

The sensors are separate arms, not winner-picked formulations.

1. **Legacy-M4 anchor (`p=3`)**: peak/core-sensitive. This is the minimal reuse test and establishes
   whether current Z/M can close the already observed M4 bounded state.
2. **Area-integrator arm (`p=1`)**: `A_G=<Psi(r)>`, a soft recruited-area fraction. This tests whether a
   pool that continues to grow with recruited area supplies a missing containment/exit gradient.

Only the `p=1` arm may be described as spatial-integrator-like, and only if `A_G/S_G` demonstrably tracks
recruited area. Neither arm is a literal GABA conductance.

Locked sensor start: `r0=0`, `r50=0.4`, `n=2`, `tau_mu=30 ms`, `tau_S=80 ms`, `S_max=1`. An observer run
must show that the sensor rises during the recruited shoulder but is not tonically occupied by ordinary
interictal events. If this fails, recalibrate `r50` from slow-off and Z-runaway traces before changing
`alpha_G`.

## 5. Cheap-first execution ladder

All primary runs are spontaneous: `KICK_BOOST=0`, fixed parameters, fixed scaffold, and the same-seed
noise stream across arms.

### Stage 0: exactness and observer

- full-engine slow-off parity;
- standalone MZ versus composite with `use_SG=False` parity;
- composite with `use_SG=True, alpha_G=0` spike/RNG parity to standalone MZ;
- observer traces for slow-off and `zA_q50_tz10000`.

### Stage 1: containment with M off

Primary access anchor: `I_th=1.6652801609959704`, `tau_z=10000 ms`, which gives operational runaway at
about 4.7--4.9 s in seeds 1/3/4.

- `p=3`: `alpha_G = [8, 12, 16, 20, 24, 32]`;
- `p=1`: `alpha_G = [8, 16, 24, 32, 48, 64]`.

Seed 1, `T=10 s`, exact 120 Hz / 100 ms runaway early-stop. If a bounded strip appears, test the
candidate and its two neighbours in seeds 3/4 before adding M. Only if no candidate appears but the
sensor is valid may `tau_S=[40,120,200] ms` be tested at the best `alpha_G`; do not open a broad
`r50 x p x tau_mu x tau_S x alpha` Cartesian sweep.

**Adaptive boundary refinement registered after the first Stage-1 wave (2026-07-19):** the observer
passed for both sensors, `alpha=0` reproduced runaway, and every registered `alpha>=8` produced only
returning/amplified IED trains rather than a sustained recruited epoch. The unresolved transition is
therefore bracketed by `[0,8]`. Before invoking the no-go rule, run the same seed/T/thresholds at
`alpha=[1,2,3,4,5,6,7]` for each sensor. This is a one-dimensional boundary localization; no M,
sensor, time-constant, or phenotype threshold changes are allowed. If it still jumps directly from
runaway to IED-like activity, the minimal composition is a clean containment no-go and M is not added.

**Longevity diagnostic registered after boundary readout (2026-07-19):** no cell crossed the original
sustained-recruitment gate, but `p=1, alpha=2` and `p=3, alpha=1` showed a slow, monotone rise in 1-s
rolling rate through the 10-s endpoint. Run only these two cells to 20 s with the unchanged runaway
detector. This is not a relaxation of the phenotype gate: it tests whether the apparent bounded burst
train is merely delayed runaway. Any runaway, or a still-rising endpoint without a settled bounded
window, closes the minimal composition as no-go. Only a settled, non-runaway 20-s trajectory may be
taken to independent seeds.

The already registered `tau_S=[40,120,200] ms` sensitivity is the final allowed rescue test after this
fixed-80-ms no-go. It is evaluated only at the two boundary coordinates above, at 15 s, with no other
parameter changes. A non-runaway cell must also have a settled last-3-s rolling-rate slope before it can
proceed to a 20-s confirmation. If this six-cell test yields only delayed runaway or IED-like return, this
branch stops; do not add M or open a new grid.

### Stage 2: exit with weak M

Only bounded m-off cells are eligible. Lock `tau_m=2000 ms` and scan the weak, previously unresolved
adaptation range:

`eta_m = [0, 0.00186, 0.00373, 0.00745, 0.01118, 0.01863]`.

Seed 1 uses `T=15 s` and no high-rate early-stop for a candidate termination cell: a genuinely bounded
high epoch must be allowed to return. The exact m-off and SG-off controls remain paired to the same seed.

### Stage 3: confirmation and mechanism controls

For a termination candidate:

- seeds 1/3/4 at 20 s;
- dynamic pool versus matched clamped pool;
- M off, SG off, and Z off;
- early post-offset probe must be attenuated and a later probe must regain bounded excitability;
- only after this short confirmation: 40 s repeated-cycle test and spatial/LFP readout.

## 6. Phenotype and acceptance contract

The exact runaway detector remains a 20 ms smoothed population rate sustained at or above 120 Hz for
100 ms. A recruited epoch requires at least 250 ms above 20 Hz, so ordinary short IED-like transients do
not count.

Report plateau and bursting separately. A bounded recruited trace is not called bursting unless the
recruited interval lasts at least 1 s, has at least four resolved peaks, modulation index at least 0.30,
and a non-DC spectral peak in 0.5--20 Hz. These are fixed screen descriptors, not proof of a limit cycle.

The lifecycle gate requires all of:

1. slow-off retains returning interictal-like events;
2. Z causes autonomous entry without a kick or parameter switch;
3. the recruited interval is bounded and lasts at least 1 s;
4. M-off does not terminate the same state, while M-on returns for at least 2 s to the same-seed
   slow-off rate band without reset;
5. no rebound runaway during pool decay;
6. early re-trigger is attenuated and late re-trigger is bounded;
7. the result replicates in at least two of three primary seeds, with seed 4 reported as stress;
8. state-fork/frozen-slow analysis locates separable entry and exit boundaries before any downstream
   workflow treats the trajectory as an ictal lifecycle.

Until item 8, the safe term is `autonomous lifecycle candidate`, not Hopf, limit cycle, tonic-clonic, or
spontaneous seizure.

## 7. Stop rules

- Any parity failure: stop before simulation.
- Invalid observer (`S_G` absent during recruitment or tonic during slow-off): stop the alpha sweep and
  recalibrate the sensor.
- All `alpha_G<=32` in the p=3 anchor remain runaway, or jump directly from runaway to suppression with
  no bounded interval: M is not added to that arm.
- No bounded m-off high state in either registered sensor arm: clean no-go for the minimal composition.
- Weak M only prevents entry or leaves a non-returning flat plateau: clean no-go; do not rescue by adding
  conductance, `phi`, STD, or a large unregistered grid in this branch.
- The first credible termination candidate switches priority from discovery to cross-seed and mechanism
  controls; the remaining broad grid is cancelled.

## 8. Resource contract

- `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` before NumPy import.
- Parent builds the substrate and recurrent edge cache once; Linux `fork` shares them copy-on-write.
- `chunksize=1`, `maxtasksperchild=1`; workers return metrics plus downsampled traces only.
- Screen runs do not save I-cell rasters, LFP, drive snapshots, or spatial movies.
- Worker cap is `min(12, floor((MemAvailable-96 GiB)/(1.2*measured_peak_RSS)))`; before a measured RSS
  exists, use 6.5 GiB per 10 s worker. Never launch when `MemAvailable<96 GiB`.
- The initial wave is capped at 8 workers while the conductance launcher is active. Preserve at least
  96 GiB available RAM and abort launching a new wave if swap use rises materially.
- One launcher lock and a start manifest are mandatory so a live run is distinguishable from a stale
  result directory.

## 9. Claim boundary and downstream release

The three blocked workflows remain readout-only until the lifecycle gate passes. Engineering completion,
a bounded rate ceiling, or one successful seed cannot release them. A negative result is still useful:
it would show that combining the existing Z/M slow variables with an already validated recurrent-gain
stabilizer is insufficient, sharply motivating the separate conductance/topology branch.
