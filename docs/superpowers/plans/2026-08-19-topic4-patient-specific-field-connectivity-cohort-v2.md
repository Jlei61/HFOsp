# Topic 4 patient-specific field/connectivity cohort v2 execution plan

**Status:** `READY_FOR_IMPLEMENTATION_AND_UNATTENDED_RUN`

**Spec:** [`2026-08-19-topic4-patient-specific-field-connectivity-cohort-v2.md`](../specs/2026-08-19-topic4-patient-specific-field-connectivity-cohort-v2.md)

## 1. Execution order

### Phase 0: immutable inventory

1. Hash the 34 frozen target JSON/NPZ pairs and 34 layout records.
2. Verify train/held-out recording-block separation and masked contact order.
3. Verify that 28 subjects contain `real_coords_sheet` and list the 6 missing.
4. Mark E1146 as `DEVELOPMENT_SOURCE`, while retaining its reconstruction run.
5. Freeze the uniform spline/Fourier basis hash and local-connectivity feature
   names before any patient score is computed.

Outputs:

```text
results/topic4_sef_hfo/patient_specific_field_connectivity_cohort_v2/
  DATA_MANIFEST.json
  GEOMETRY_ELIGIBILITY.json
  SEARCH_BASIS.json
  controller.status
```

### Phase 1: engineering canary

Run three geometry-diverse patients (`epilepsiae_590`, `epilepsiae_958`,
`yuquan_chengshuai`) through one complete fit generation and one selection seed.
This canary may stop only for engineering/safety failure. Its patient scores do
not alter the frozen objective or search size.

Required checks:

- candidate fields differ by hash and have exact fixed mass;
- field builder never reads contact coordinates;
- EE/E-to-I incoming totals are conserved;
- topology, delays and GABA hashes are unchanged;
- active Z/M trace exists;
- late runaway is invalid;
- real contact order equals the target order exactly;
- one failed worker does not delete the candidate from history.

### Phase 2: patient-specific fit

For each of the 28 geometry-eligible patients:

1. start or resume its CMA-ES checkpoint;
2. run six generations of ten candidates;
3. use one common fit seed within each generation;
4. checkpoint after every generation;
5. retain all invalid and low-support candidates with reasons;
6. rank only on patient training blocks.

The scheduler is cohort-global and resource-aware. It starts no worker unless:

- host `MemAvailable >= 150 GiB`;
- output filesystem has at least 120 GiB free;
- one-minute load is below 75% of logical CPU count.

It then chooses

```text
min(8, floor((MemAvailable - 96 GiB) / 8 GiB))
```

workers, never fewer than one. It sleeps 600 s while resources are unavailable;
there is no tight polling loop.

### Phase 3: independent selection

For each patient, rerun its four best distinct fit candidates on seeds 1911 and
1912. Select by mean training objective. Save selection regret relative to the
fit winner and candidate rank stability across seeds.

### Phase 4: held-out confirmation

Run the frozen patient winner on seeds 1921-1924. Only here load held-out mode
descriptors. Compute:

- weakest-mode loss;
- matched within-shaft null;
- natural KMeans two-mode verdict;
- OOD;
- per-seed event and mode counts;
- patient-level aggregate with equal network weight.

No retry, candidate substitution or parameter change is allowed after a
held-out score exists.

### Phase 5: frozen-winner mechanism replay

Without changing the winner field, run paired replays:

- Node only;
- Node + EE;
- Node + E-to-I;
- Node + EE + E-to-I;
- joint winner with Z/M off.

These are causal parameter ablations of the fitted substrate. They do not
reselect a winner.

### Phase 6: figures and report

Generate the two required Fig.4-style figures for every evaluable patient, then
produce one compact cohort panel. Every figures directory gets a Chinese
`README.md`. Rendered PNG and PDF are visually inspected; tests alone do not
accept a figure.

The final report separates:

- 28 fitted real-geometry patients;
- 27 non-E1146 patients for the cross-patient primary summary;
- 6 geometry-missing target-only patients;
- E1146 reconstruction;
- optimization completion, model evaluability and scientific recovery.

## 2. Search budget

Nominally per fitted patient:

| phase | simulations |
| --- | ---: |
| six fit generations | 60 |
| four candidates x two selection seeds | 8 |
| one winner x four confirmation seeds | 4 |
| frozen mechanism replays | 10 |
| total | 82 |

The full 28-patient ceiling is 2,296 simulations, plus the engineering canary.
Existing output is reused only when config, candidate, subject, seed, runtime
commit and input hashes all match.

## 3. Background lifecycle

The controller is launched as:

```text
systemd-run --user -> nohup -> resource-aware Python supervisor
```

It writes one-line state transitions to `controller.status`, appends detailed
logs to `run_logs/supervisor.log`, and sends `notify-send` on canary completion,
full completion or failure. `DONE.json` is written only after aggregation and
figure metadata finish.

The controller must survive shell exit, network interruption and workstation
sleep. It resumes from per-subject optimizer checkpoints and completed worker
hashes rather than restarting a generation.

## 4. Acceptance language

Allowed after a positive result:

> Patient-specific continuous node fields and local excitatory redistribution,
> fitted only on training interictal events, reproduced both held-out propagation
> modes in a subset/cohort of real implant geometries.

Not allowed:

- global optimum;
- recovered anatomical lesion/core;
- all 34 patients fitted when six lack geometry;
- patient generalization without a new blind unit;
- waveform or clinical HFO reproduction from rank-order evidence alone.
