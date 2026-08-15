# Data-driven SNN cohort: canary acceptance and formal freeze (2026-08-15)

Status: `CANARY_ACCEPTED`; formal 34-subject run launched at commit `96618174`.
This document records the preflight verdict and the three pre-registration
corrections made before any formal patient score existed. It carries no cohort
result.

## 1. What the canary was for

The canary is a capacity preflight on six geometry-diverse subjects, not a
cohort result. Its job: show that one shared pretrained morphology can be
simulated and read through montages of very different contact counts without
the detector quietly failing on the small ones, and that the accepted figures
render. `epilepsiae_1146` is the development source subject and is excluded
from every cross-subject transfer statistic.

## 2. Canary outcome

16/16 workers complete at `96198f72`, no runaway, all provenance frozen,
figures on disk. `fit_selection.json` reports `CANARY_FIT_EVALUABLE` with
6/6 subjects evaluable and 5/5 transfer subjects evaluable, above the frozen
minimum of 4.

Event support by montage size (median over all 16 workers):

| subject | contacts | median recruited | readable event fraction |
|---|---|---|---|
| `epilepsiae_1077` | 6 | 4.0 | 0.61 |
| `yuquan_huanghanwen` | 10 | 6.0 | 0.70 |
| `epilepsiae_1146` | 15 | 9.25 | 0.77 |
| `epilepsiae_590` | 16 | 10.5 | 0.79 |
| `yuquan_songzishuo` | 38 | 23.0 | 0.87 |
| `yuquan_zhangbichen` | 52 | 36.25 | 0.76 |

Recruitment holds a roughly constant 60–70 % of contacts from 6 to 52
contacts, so there is no contact-count-dependent detector failure. The
memory sentinel alone (`rot000_node`, seed 1661) had shown a median of one
recruited contact on the six-contact montage; that is the worst single arm,
not the canary behaviour, and reading the sentinel as representative would
have been wrong.

Audit artifact: `results/topic4_sef_hfo/data_driven_snn_cohort_v1/multisubject_canary/canary_acceptance_audit.json`.

## 3. A confound the canary surfaced

The raw same-minus-crossed margin falls monotonically with contact count:
`epilepsiae_1077` (6 contacts) is highest, `yuquan_zhangbichen` (52) lowest.
That is what a rank correlation over six contacts does on its own, not a
statement about the model. The formal endpoint is null-relative and its null
is drawn inside the same montage, so the inflation cancels within each
subject; the aggregator now also reports Spearman(subject delta, contact
count) as a diagnostic so the cancellation can be checked rather than assumed.
This is a reported diagnostic, not a gate, and was added before any formal
patient score existed.

The canary also shows the natural-KMeans margin falling well below the
supervised margin on the two largest montages (0.53 → 0.13 and 0.42 → 0.19).
A `SAME_NETWORK_K2_INSUFFICIENT` cohort outcome is therefore a live
possibility and must be reported as such if it occurs.

## 4. Three pre-registration corrections

### 4.1 The within-shaft null did not have 64 alternatives for everyone

Four subjects do not have 64 distinct within-shaft permutations:
`epilepsiae_583` has 3, `epilepsiae_1073` has 5, `epilepsiae_1077` has 11 and
`epilepsiae_253` has 23. The original implementation filled 64 rows by drawing
with replacement, which advertised a resolution those montages do not have —
for `epilepsiae_583` the smallest reachable permutation p-value is 1/4.

The non-identity group is now enumerated exactly when it is no larger than the
request and distinct rows are drawn otherwise; `effective_null_size` and
`minimum_reachable_p` are stored per subject and drawn on the layout figure.
Padding back to 64 with replacement is forbidden.

Consequence for the endpoint: the subject-level gate is "held-out weakest-mode
loss below the matched null median", not a per-subject permutation p-value.
`epilepsiae_583`, `1073` and `1077` could never reach p ≤ 0.05 on this null and
would have failed for montage reasons rather than model reasons.

### 4.2 The canonical layout compressed the axis the claim rests on

The canonical contact-order layout fixed contact pitch at 1 mm. Montages with
few distinct ordinals ended up inside a 2 mm strip (`epilepsiae_1077`,
`epilepsiae_253`) while their real-geometry counterparts spread the same
contacts over 16 mm — on the within-shaft ordinal axis, which is exactly the
axis the contact-order claim lives on. That both weakened the primary arm and
confounded the canonical-versus-real sensitivity contrast with montage extent.

The ordinal axis now fills the usable sheet the way the real-geometry
projection fills its largest-variance axis. The same edit fixed
`ordinals.min(initial=0)`, which measured the span from zero rather than from
the smallest ordinal.

Residual difference, stated rather than smoothed over: both layouts fill 16 mm
on their principal axis, but the canonical shaft axis is uniformly 16 mm while
the real minor axis ranges from 0.3 mm to 15 mm. The canonical layout gives
every subject the same maximal readout; the real-geometry arm tests whether
that uniformity changes the direction of the effect.

### 4.3 The scorer aligned model and patient by shape only

Matching array shapes let a reordered montage score model contact *i* against a
different patient contact with nothing raised. The scorer now requires the
target to carry its own contact order and checks it element by element.

## 5. Formal design as frozen

- 24 shared candidates: 4 rotations x 2 reflections x {Node, Node+EE,
  Node+EE+EtoI}, generated before any patient score. The canary may not prune
  this library: it shares five subjects with the formal cohort, so its
  alignment scores would leak.
- Three selection stages, all frozen in the config: all candidates on the fit
  seeds, each subject's best two on the selection seeds, each subject's single
  choice on fresh confirmation seeds. Stages A and B only ever see the patient
  training split.
- One simulation feeds all 34 canonical and all 28 real-geometry montages.
- 20,000 ms, frozen absolute detector, beta closed, Z/M off.
- Subject-level endpoint: held-out weakest-mode loss below the matched
  within-shaft null median, on confirmation seeds only.
- Cohort gates: at least 60 % of the 34 subjects pass, two-sided Wilcoxon on
  the subject-level null-relative deltas at alpha 0.05, same-network two-cluster
  structure reported separately, and canonical and real-geometry effects must
  agree in direction.

## 5.1 Formal memory sentinel (first real formal worker)

`rot000idn_node` seed 1661 completed at 20,000 ms with no runaway in 1331 s
(22 min), peak RSS 13.5 GiB, network cache hit, 595 KiB of output. It produced
128 detected events and wrote all 62 layout records (34 canonical + 28 real).

Event support on the canonical layout, which is what the primary arm reads:

- every one of the 34 subjects has at least 49 events with three or more
  recruited contacts, against a frozen minimum of six;
- the recruited-contact fraction is flat against montage size,
  Spearman 0.247, p = 0.158, so the canonical layout does not introduce a
  contact-count-dependent readout;
- the two weakest subjects, `epilepsiae_139` and `yuquan_zhangjiaqi`, are the
  two single-shaft cases whose canonical layout is a straight line with no
  second axis; they recruit a median of 0.5 contacts per event but still yield
  53 readable events each.

## 6. Engineering notes

- Reading 890 contacts through the frozen envelope sampler costs about 34
  minutes per worker, more than the 20-second simulation it reads, because an
  all-true boolean mask copies the whole frame matrix once per contact. A
  separate batched module computes the same distance-weighted average as one
  chunked matrix product in under two minutes, agreeing to 1e-12 with identical
  contact ordering. It is a separate module because the observation module is
  in the provenance chain of runs still in flight. BLAS blocks differently per
  chunk shape, so the chunk size is frozen in the config and recorded in worker
  provenance.
- Known provenance gap, pre-existing and shared with the canary: worker
  provenance is captured before `src/sef_hfo_snn_adapter` is imported, so that
  module's hash is not in `runtime_module_sha256`. Not changed mid-flight.
- The volume was at 99 % (13 GiB free) when the run was launched, so the
  controller pauses below 6 GiB rather than failing and the workers were built
  to drop the per-contact envelope. During stage A roughly 170 GiB was freed
  outside this line by clearing the bulk artifacts under
  `data_driven_core_field/` and `data_driven_core_field_rev9/`; 180 GiB is now
  free. All five inputs this run pins survived with hashes matching the config,
  including `data_driven_core_field/config/stage_config.json` and the rev9
  detector audit, and no worker failed across the cleanup.

## 7. Claim boundary

Nothing here supports a cohort claim. The canonical layout is a target-blind
contact-order readout, not patient anatomy, and a positive formal result could
only support recovery of held-out contact-order structure above matched
within-shaft nulls — never "34 patients were reproduced", anatomical cores,
clinical waveform identity, patient-blind generalization or seizure mechanism.
