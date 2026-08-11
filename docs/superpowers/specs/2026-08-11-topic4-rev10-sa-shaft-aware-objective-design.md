# Topic 4 rev10-SA: shaft-aware patient objective and dual-shaft field recovery

## 1. Scientific objective

The only objective of rev10-SA is to determine whether patient-training data can
drive the existing fixed-budget `K=3` node field to cover both ICL and SCL and to
reproduce within- and cross-shaft propagation structure. Edge calibration,
`beta`, topology expansion, optimizer comparison, slow variables, and patient
generalization are outside the first phase.

The accepted rev9-L state is:

```text
OBJECTIVE_SHAFT_BLINDNESS_CONFIRMED
/
FROZEN_FIELD_HAS_NO_SCL_SUPPORT
/
MODE_A_WITHIN_ICL_MISMATCH_REMAINS
/
MULTISHAFT_FIELD_AND_EDGE_CAPACITY_UNRESOLVED
```

rev9-L is engineering-complete, but its negative capacity statement is bounded
to the frozen ICL-biased field, the shaft-collapsed observation, and the tested
finite static-edge library. It does not establish that a shaft-aware learned
field or edge realization cannot reproduce the patient modes.

## 2. Development and blind boundary

All targets, floors, controls, and candidate comparisons in rev10-SA use patient
training recording blocks only. The old rev8.1 held-out blocks have already been
used and are not blind. No patient held-out score may be read by this phase.

Without a newly frozen patient data unit, the strongest eventual status is:

```text
DEVELOPMENT_ONLY_SHAFT_AWARE_RECOVERY
```

## 3. SA0 contact and shaft contract

The canonical contact order is frozen once and shared by patient and model
artifacts. Each contact record contains:

```text
contact_name
contact_index
shaft_id
contact_number
within_shaft_order_by_shared_axis
sheet_xy_mm
shared_axis_coordinate_mm
```

The readout contract also records `rx`, `Rr`, timing threshold, participation
margin, and any detector constants that can change contact eligibility.

Pair classes use unordered pairs:

```text
ICL-ICL: 55
SCL-SCL: 6
ICL-SCL: 44
```

The producer writes canonical SHA256 values for contact geometry, shaft
assignment, pair classes, and readout parameters. `readable` remains an
event-level onset-extraction property. It is not a synonym for multishaft
coverage.

## 4. SA1 patient target

Primary labels remain the frozen patient-training A/B labels from rev8.1. This
separates a representation audit from a redefinition of the biological target.

For event `e` and contact `i`:

```text
m[e,i] = 1 when the contact participates
u[e,i] = (t[e,i] - min_e(t)) / (max_e(t) - min_e(t) + eps)
```

The primary patient feature is:

```text
x_e = [m, m*u, f_ICL, f_SCL, delta_first, delta_valid]
```

where `delta_first` is the normalized first-SCL minus first-ICL onset and
`delta_valid` prevents a missing shaft from being silently imputed as a real
zero lag. Contact identity is never projected onto the old shared one-dimensional
axis.

Patient-train-only standardization and PCA retain 95% variance with a frozen
dimension cap. K=2 consensus KMeans is secondary. Its AMI to the old A/B labels,
recording-block stability, proportions, and a K=3 exploratory solution are
reported. `AMI < 0.8` stops model optimization and yields
`PATIENT_MODE_DEFINITION_UNRESOLVED`.

### 4.1 Direction and event-extent factorization amendment

The pre-specified AMI stop was triggered (`AMI=0.0112`). Model optimization was
not resumed on that result. A subsequent patient-training-only exploratory audit
showed that the stable shaft-aware K=2 partition is predominantly an event-extent
split rather than a replacement for the old propagation-direction labels:

```text
extent AUC for old A/B: 0.6181
extent AUC for shaft-aware K=2: 0.9519
old A/B x extent-cluster cells: 5082 / 4193 / 8668 / 12106
```

After exact matching within `recording block x recruited ICL count x recruited
SCL count`, old A/B retained ICL-ICL and ICL-SCL precedence contrasts far above
the within-stratum permutation null. The amended development target therefore
uses:

```text
primary mode identity = frozen old A/B direction labels
within-mode continuous factor = shaft recruitment extent and event cloud
flat shaft-aware K=2 as patient mode = rejected
```

This amendment is explicitly post-training-data and exploratory. It resolves the
meaning of the low AMI for development work; it is not a preregistered patient
mode redefinition and provides no blind-generalization evidence.

### 4.2 Legacy timing boundary

Patient files retain `lag_raw`, but several historical model artifacts retain
only dense ranks. Historical rescoring therefore has three explicit tiers:

1. `FULL_TIMING`: contact onset or contact envelope is retained; all descriptors
   can be evaluated against the primary target.
2. `ORDINAL_COMPATIBLE`: only ranks are retained; recruitment and precedence are
   primary, while profile/event-cloud use a separately calibrated ordinal target
   and cannot be reported as full timing recovery.
3. `NOT_RESCORABLE`: per-event contact identity was not retained.

No interpolation or inverse mapping may manufacture contact timing from a pooled
rank curve.

## 5. SA2 shaft-balanced descriptors

### 5.1 Recruitment

Recruitment MAE is computed separately for ICL and SCL. Each raw distance is
converted to excess-noise units using its own matched-count patient block floor:

```text
E = max(0, (D - floor_median) / (floor_q95 - floor_median + eps))
```

The mode recruitment loss is the centered smooth maximum of ICL and SCL excess.
The 11 ICL contacts therefore cannot numerically overwhelm the 4 SCL contacts.

### 5.2 Precedence

Each unordered pair has a four-state distribution:

```text
[i before j, j before i, tie, not jointly recruited]
```

The final state makes missing SCL support part of the distance rather than a NaN
that disappears. Pair-wise Jensen-Shannon divergence is averaged separately for
ICL-ICL, SCL-SCL, and ICL-SCL, calibrated with class-specific patient floors,
then combined with a centered smooth maximum.

### 5.3 Profile and event cloud

The profile contains separate contact-identity means of `m*u` for ICL and SCL,
plus cross-shaft first-onset offset and validity. These three parts receive
separate floors and a smooth worst-part combination.

The event-cloud distance is computed in the patient-train-only shaft-aware PCA
space. Model data only transform through the frozen standardizer/PCA. The old
one-dimensional normalized rank curve remains historical diagnostics only.

### 5.4 Mode and global objective

For mode `k`:

```text
D_k = mean(D_rec,k, D_prec,k, D_prof,k, D_dist,k)
```

The development objective is:

```text
J_SA = E_global + 2 * smooth_max(D_A, D_B) + 0.25 * E_JS + R
```

Forced source-capacity experiments do not contain a spontaneous mode-occupancy
estimate, so `E_JS` is reported but omitted from their selection score. This is
not replaced by a hard multishaft event gate.

## 6. SA3 zero-simulation controls

Before any SNN run, the patient target must respond correctly to:

1. censoring all SCL contacts;
2. shifting only SCL timing while preserving masks and within-shaft ordering;
3. restoring all combinations of 0/4 through 4/4 SCL contacts, with mean score
   improving monotonically across restoration level;
4. collapsing shared-axis coordinates while retaining contact and shaft identity.

The new feature must remain unchanged by control 4, while a synthetic ICL-only
event and an axis-matched SCL-only event must remain distinct.

Controls are exploratory measurements with component-specific expected effects,
not a collection of unrelated hard gates. Only a metric implementation that is
insensitive to SCL censoring or cross-shaft timing stops execution.

## 7. SA4 historical artifact audit

Every requested family is inventoried, even when full rescoring is impossible:

- rev8.1 field fit candidates;
- frozen rev8.1 final candidate;
- Node and Node+Edge;
- L2 and repeated L3 Sobol candidates;
- hand dual-core;
- Stage 2 filament.

Outputs distinguish `FULL_TIMING`, `ORDINAL_COMPATIBLE`, and `NOT_RESCORABLE`.
At minimum, every artifact with per-event contact ranks reports ICL/SCL
recruitment, the three precedence classes, multishaft participation, and mode-A
within-ICL error. A historical candidate can be called
`OLD_OBJECTIVE_SELECTION_MISS` only if its retained data support the full new
objective and it improves on independent model selection units.

### 7.1 Completed SA4 adjudication

SA4 recovered exact rank parity for all retained full-timing forced-source rows
(`768` L2 rows and `2052` repeated L3 rows). The frozen rev8.1 final, rev9 Node,
and Node+Edge events have zero SCL recruitment. Hand dual-core and Stage 2
filament have only sparse multishaft events (`7.7%` and `2.2%`). Null and Edge
show high SCL recruitment, but their OOD fractions are `92.3%` and `83.3%`, so
their patient-mode matrices are not evaluable.

All `64` L2 and `57` retained L3 candidates have zero SCL recruitment. The old
and shaft-aware rankings correlate only weakly in L2 (`rho=0.36`) and are not
associated in L3 (`rho=0.13`). This supports objective mismatch, but does not
identify a missed field candidate: all `48` rev8.1 field-fit history entries lack
per-event fixed-contact values and are `NOT_RESCORABLE`.

The frozen SA4 verdict is:

```text
FROZEN_LEARNED_NODE_FIELDS_HAVE_ZERO_SCL_SUPPORT
/
RIGID_CONTROLS_HAVE_SPARSE_SCL_SUPPORT
/
EDGE_NULL_MULTISHAFT_EVENTS_ARE_PATIENT_MODE_NOT_EVALUABLE
/
OLD_OBJECTIVE_FIELD_SELECTION_MISS_NOT_TESTABLE
```

## 8. Conditional SA5-SA8 execution

SA5 contact detectability and SA6 dual-shaft field canaries are designed only
after SA0-SA4 close. Long simulations use managed `systemd-run --user -> nohup`,
bounded workers, 120-second waits, status/log files, and completion notification.

SA6 keeps Node-only, `K=3`, total field mass, topology, `d_i`, and detector fixed.
It compares frozen field, component-3 matched SCL relocation, matched off-shaft
relocation, and a small deterministic SCL mass/width scan. Formal field
optimization and a new node-edge factorization remain conditional on a positive
dual-shaft capacity canary.

The old `alpha=0.75` is invalid after any change to `h`. `beta` stays closed.

### 8.1 SA5 contact-detectability design

SA5 uses a uniform-threshold Null substrate, six frozen network seeds, and the
same network/OU/Poisson realization for sham and every contact packet. Within
each network, all 15 contacts use a `1.0 mm` disk and the same number of nearest
E neurons; the common count is the minimum support across contacts, capped at
`0.5%` of E neurons. The engine's exact `forced_spike_mask` hook injects one
packet at `100 ms`.

The assay jointly reports local neural spike excess and the current-based
`LFPRecorder` response at the stimulated contact. It also records raw kernel
support, normalized-weight ESS, baseline noise, current peak, current SNR, and
the margin above one common definition (`sham mean + 5 SD`). SCL/ICL summaries
are first computed within each network and then equally weighted across the six
networks.

The `0.5` SCL/ICL ratio is an exploratory branch reference, not a new blocker:

```text
neural retained, current weak -> VIRTUAL_CONTACT_OBSERVATION_FAIL
neural weak                  -> SCL_LOCAL_NETWORK_RESPONSE_LIMIT
both retained                -> SCL_READOUT_NOT_PRIMARY_LIMIT
```

SA5 completed on six networks from a clean commit. Every network used exactly
`160` E neurons per contact packet. The equal-network SCL/ICL ratios were
`0.961 [0.934, 0.986]` for current gain and `0.953 [0.942, 0.985]` for local
neural response; all ICL and SCL contacts had positive detector margin. The
frozen branch status is therefore `SCL_READOUT_NOT_PRIMARY_LIMIT`.

SA6 is consequently cleared to launch. It remains Node-only, fixed `K=3`,
fixed total field mass, fixed topology and `d_i`. The exploratory candidate set
contains exactly 21 fields: frozen, component-3 matched SCL relocation,
matched off-shaft relocation, and `2 centers x 3 masses x 3 longitudinal
widths`. Three new network seeds (`1031-1033`) each receive equal `0.5%` E-cell
packets from ICL mode-A, ICL mode-B, and SCL source locations, plus one paired
short spontaneous run. The common population detector is frozen at
`0.0195703125` before candidate execution. No Edge, `alpha`, `beta`, topology,
or optimizer parameter is opened.

SA6 uses patient floors at three events per old direction mode. The
`excess<=1` SCL recruitment line, support in at least two of three paired
networks, and absence of runaway define an exploratory capacity reference, not
a patient blind acceptance gate. Mode-B ICL structure is reported beside the
frozen baseline instead of becoming another blocker.
