# Topic 4 rev10-SA: shaft-aware patient objective and dual-shaft field recovery

## 1. Scientific objective

The only objective of rev10-SA is to determine whether patient-training data can
first test whether the existing fixed-budget `K=3` node field can cover both ICL and SCL and then, after the representation audit below, replace component allocation with a continuous non-component field to
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

### 8.2 SA6 result and boundary

All `21 candidates x 3 networks` completed from commit `b066026f`; all 63
workers exited successfully and no forced or spontaneous run entered runaway.
No candidate recruited any SCL contact after either ICL direction source, so
the floor-excess SCL recruitment errors remained `2.81` for mode A and `3.88`
for mode B. The tested-grid status is:

```text
DUAL_SHAFT_FIELD_CAPACITY_NOT_FOUND_IN_TESTED_GRID_CANARY
```

This is not a general fixed-budget family failure. The strongest tested SCL
allocation (`w_SCL=0.35`, `sigma_parallel=4.5 mm`) produced mean `h=0.424`
and mean `delta Vtheta=-0.216 mV` within 1 mm of SCL contacts, yet ICL-to-SCL
recruitment remained zero. Conversely, an SCL packet reached at least one ICL
contact in `2/3` networks for that field. The current canary therefore points
to directional cross-shaft access, packet threshold, or field-amplitude budget
as the next discriminants; it does not justify a high-dimensional SA7 fit.

The next minimal experiment is a non-optimized mechanism ladder: packet
amplitude curve on the strongest field, then SCL mass/total-budget curve, each
with matched off-shaft controls. Only persistent ICL-to-SCL failure after that
ladder justifies opening a directional route-support family. `beta` remains
closed because no result has isolated a width/delay residual.

### 8.3 Representation correction: SA6F continuous field

The preceding SA6 did not test a free field. Its producer fixed the first two
Gaussian components and assigned component 3 to the SCL relocation/weight scan.
Its negative result is therefore renamed:

```text
FIXED_K3_COMPONENT3_RELOCATION_CANARY_NEGATIVE
```

It cannot adjudicate continuous-field capacity. SA6F removes `K`, component
identity, and peak-count constraints from the primary representation. The
latent field is a tensor-product cubic B-spline surface

```text
s(x,y) = sum_uv c_uv B_u(x) B_v(y)
q(x,y) = exp(s(x,y))
h       = project_to_fixed_mass(q)
```

The `4x4` primary surface has 16 coefficients and one mass-projection-redundant
constant direction, hence 15 effective degrees of freedom, approximately
matched to the old K=3 Gaussian's 17 parameters. A `6x6` surface is only a
resolution sensitivity. Spline coefficients are numerical field controls, not
cores; the field may produce any number of extrema or connected regions.

Patient-training contact recruitment initializes the surface with equal total
ICL/SCL fitting weight. No spline basis is assigned to a shaft. Mode-A and
mode-B forced sources come from their patient earliest-ICL centroids rather
than frozen Gaussian centers. K=3 remains one historical benchmark only.

SA6F reuses network seeds `1031-1033` from the constrained SA6 so field
representation is the paired experimental factor and the existing 500-MB
network caches are reused. These are development networks, not independent
confirmation seeds.

SA6F precedes packet-amplitude, total-budget, Edge, `beta`, topology, and formal
field optimization. Its result is exploratory and cannot support patient
generalization.

#### SA6F result and interpretation correction

All `37 x 3 = 111` workers completed without runaway. Neither the `4x4`
matched-DoF fields nor the `6x6` resolution fields produced forced
ICL-to-SCL recruitment. This is not yet a continuous-family or connectivity
failure. The strongest tested continuous initialization reached only mean
`h=0.128` within 1 mm of SCL contacts, whereas the preceding constrained K3
SCL allocation reached `0.424`. Exact fixed-mass projection diluted the broad
low-resolution spline surfaces, and some field mass remained between or away
from the observed contact paths.

SA6F is therefore frozen as:

```text
LOW_RESOLUTION_CONTINUOUS_INITIALIZATION_NO_CROSS_SHAFT_SUPPORT
```

One `4x4` field reduced mode-A ICL precedence excess from approximately `2.99`
to `0.393`, showing that a continuous field can materially change the limiting
within-ICL pattern. SCL recruitment excess nevertheless remained unchanged.
This is a useful initialization lead, not a fitted patient solution.

### 8.4 No-K continuous support capacity control: SA6G

SA6G does not add Gaussian cores or increase biological `K`. It constructs a
continuous positive-control field from distance to the observed shaft paths:

```text
d(x, Gamma) = minimum distance from x to the continuous path Gamma
q(x)        = exp[-d(x, Gamma)^2 / (2 sigma^2)]
h           = project_to_fixed_mass(q)
```

`Gamma` is either the union of the observed ICL and SCL polylines or that union
plus the shortest cross-shaft bridge. The line segments encode the capacity
control geometry only; they are not components, peaks, or inferred cores.
Widths are `0.10, 0.20, 0.35, 0.50 mm`, giving eight fields at the unchanged
mass budget. Preflight on a dense sheet gives mean projected `h=0.535-1.000`
within 0.25 mm of the path, so this control closes the field-strength ambiguity
left by SA6F.

SA6G remains Node-only with frozen topology, `d_i`, detector, network seeds,
packet sources, and total field mass. It is an exploratory capacity control,
not the patient-fit field. The formal patient fit remains a free continuous
spline/RKHS field optimized directly with the shaft-aware SNN objective and a
smoothness penalty; basis resolution is numerical discretization, never a core
count. `beta`, Edge, and topology remain closed until SA6G is adjudicated.

#### SA6G result

All `8 x 3 = 24` workers completed from clean commit `425e8b36`, with no
runaway. In the four connected fields, actual bridge mean `h` was
`0.528-0.907`; `74.4-98.0%` of bridge-near neurons had `h>=0.5`. Despite this,
both patient-derived ICL forced sources recruited zero SCL contacts in every
candidate and every network. SCL forcing reached only approximately one of 11
ICL contacts on average (`0.061`). No connected field generated a spontaneous
event in the 400-ms canary; the disconnected narrow controls generated five
events, all returned and none were multishaft.

The frozen status is:

```text
NO_K_CONTINUOUS_CONNECTED_FIELD_FAILS_CROSS_SHAFT_AT_FIXED_PACKET_AND_BUDGET
```

This result removes fixed core count and weak bridge field as explanations for
the current failure. It still does not prove a connectivity-family failure,
because packet amplitude and total excitability budget remain frozen. The next
small experiment is a paired packet-amplitude curve on the connected fields,
followed only if needed by a total-budget curve. If ICL-to-SCL access remains
zero while local ICL propagation and SCL-to-ICL response persist, a directional
route-support family is justified. `beta` remains closed because the residual
is reachability, not demonstrated width or delay mismatch.

### 8.5 Observation-boundary correction and SA6H

The SA6F/SA6G results remain valid capacity diagnostics, but neither is a free
learned field. SA6F fitted B-spline coefficients at observed contacts using
patient recruitment targets. SA6G defined support from observed shaft paths.
Both therefore place more latent information where the observation is denser.
Their corrected roles are:

```text
SA6F = OBSERVATION_CONDITIONED_BSPLINE_CAPACITY_DIAGNOSTIC
SA6G = OBSERVATION_CONDITIONED_PATH_REACHABILITY_CONTROL
```

They cannot adjudicate recovery of the unobserved two-dimensional substrate.
Their geometries are forbidden as a formal fit initialization or prior.

SA6H returns to the clearly learned Stage 3 field, but uses it only as a warm
start. The formal latent field is defined over the complete `20 x 20 mm` sheet
with real Fourier features:

```text
s(x) = sum_k [a_k cos(k dot x) + b_k sin(k dot x)]
q(x) = exp(s(x))
h(x) = project_to_fixed_mass(q)
```

The constant mode is absent because fixed-mass projection makes it
unidentifiable. Frequencies are selected by isotropic radius; every frequency
has both sine and cosine phase. The stationary residual prior gives equal phase
variance and depends only on frequency magnitude. Basis bandwidth is numerical
resolution, not a core count; there is no component identity or peak-count
constraint.

The observation boundary is strict:

- field construction may use uniform sheet coordinates, the old Stage 3 field
  as a whole-sheet warm start, a stationary prior, and fixed total mass;
- it may not use contact coordinates, shaft paths, patient onset density,
  patient labels, or patient-derived forced sources;
- patient information enters only after spontaneous SNN simulation through the
  virtual-electrode readout, frozen mode classifier, and shaft-aware objective.

The initial library has `21` candidates. `V0` contains the exact Stage 3
benchmark, a uniform negative control, and its uniform-sheet spectral
projection. `V1` has three antithetic low-frequency residual pairs. `V2` has
six antithetic stationary multiscale residual pairs. Each candidate runs for
8 s without a kick on the same three development networks, using the same mass,
`d_i`, connectivity, and common absolute detector.

Selection protects spontaneous two-mode support and safety, then minimizes the
shaft-aware weakest-mode score plus a continuous OOD penalty. Roughness is a
reported tie-break only. De novo KMeans stability, KMeans/frozen-label AMI,
mode counts, direct model-current traces, and patient/model prototypes are
mandatory outputs; KMeans alone is not the objective.

The old patient held-out block has already been read. SA6H therefore remains
development-only on fresh network seeds. Edge, `beta`, and topology stay closed
until an observation-invariant Node field is frozen and confirmed.

#### SA6H initial result and V3 correction

All `21 x 3 = 63` initial workers completed from clean commit `dd9ae9ac`
without runaway or worker failure. The uniform field produced no spontaneous
events. The old K3 field produced `34` events and its uniform spectral
projection produced `33`, while the projection preflight gave field-`h` RMSE
`0.0103` and top-5% support Jaccard `0.952`. The spectral representation
therefore preserves the old learned field without using it as a component
model.

The initial search did not recover the shaft-aware patient repertoire. The
spectral warm start had weakest-mode score `5.397`; the selected V1 candidate
had `5.407` before OOD penalty and won only because its OOD was `0` versus
`0.037`. This `0.009` total-score margin is not evidence of an improved field.
Every one of the nine support-eligible candidates had zero SCL recruitment in
the six fixed events per mode, with identical SCL recruitment excess
`5.413/4.105` for modes A/B. KMeans was stable and agreed with the frozen labels
(`AMI=1`) for the eligible candidates, but the model-A prototype remained far
from patient A. Thus stable two-cluster geometry and patient recovery are
empirically separated.

The negative is not yet a failure of the Fourier family or shaft-aware loss.
The old warm field is approximately `6-7` log units below its peaks away from
support, whereas V1/V2 perturbations were at most one whole-sheet RMS. They
changed mode occupancy and often caused single-mode collapse, but did not give
the objective a candidate with variable SCL recruitment. The limitation is the
tested search radius around the warm start.

V3 therefore performs an observation-invariant allocation scan. It attenuates
the warm field and projects one identical smooth log-field direction at every
location of a `4 x 4` uniform sheet grid. The 16 locations are fixed before and
without reading contacts or shaft paths. The patient objective may select among
them only after spontaneous simulation. These directions are optimizer probes
in the same Fourier field, not biological cores or a new `K`.

#### V3 result: capacity signal and objective-entry correction

All `21 x 3` V3 workers completed from clean commit `c933986b` with zero
failure and zero runaway. The original aggregation again selected the initial
field (`5.407`). That selection is invalid for the new scientific question:
the aggregator first converted events to the old shared-axis rank curve and
discarded events for which that curve was unavailable. SCL-rich candidates
therefore reached the shaft-aware objective as `usable=0` or OOD=1.

The raw events nevertheless establish a bounded capacity result. Uniform
locations 07, 09, 10, 11, 12, 14, and 15 produced SCL events. Most were
shaft-switching rather than cross-shaft: location 07 produced 39 SCL-only and
zero joint events, while location 10 produced 21 SCL-only and zero joint
events. The strongest joint candidates were location 12 (`13/43`) and location
09 (`4/12`), far below the patient A/B joint-shaft fractions (`0.957/0.983`).
Thus continuous Node allocation can make SCL active, but the tested fields do
not yet reproduce patient multishaft events.

Old A/B direction and full shaft-aware KMeans must not be conflated. Patient
shaft-aware KMeans is stable but has AMI `0.011` to old A/B because it is
dominated by recruitment extent. A patient-training-only logistic direction
classifier in the same shaft-aware embedding preserves old A/B with
recording-block-held-out balanced accuracy `0.945` and AUC `0.990`. V4 freezes
this supervised direction classifier, assigns every model event, and reports
class-conditional OOD without deleting events. The objective is factorized
into:

1. all-event joint-shaft participation;
2. old A/B direction support and mode-conditioned shaft-aware distance;
3. OOD and event-support penalties;
4. de novo KMeans as a diagnostic only.

The half-period Fourier coordinates are also retired for optimization. Their
uniform-grid condition number is approximately `1e8`, with coefficients up to
`7.7e4`; maps are deterministic in float64, so V3 remains a valid capacity
diagnostic, but coefficient distance and roughness are not trustworthy. V4
uses a `14 x 14` uniform cubic B-spline log field (`195` effective DoF), whose
condition number is `27.6`. It projects the Stage 3 field with `h` RMSE `0.0098`
and top-5% support Jaccard `0.966`. Candidate fields comprise the spline warm,
a uniform negative control, `16 x 2` identical whole-sheet allocation probes,
and eight antithetic pairs of observation-free smooth random residuals. No
contact, shaft, onset, or mode coordinate enters field construction.

#### V4 result and V4.1 representation bridge

All 50 V4 candidates completed on common network seed 1031 with zero worker
failure and zero runaway. The field builder remained observation-free, but the
screen did not enter the known V3 capacity region: V4 used the full Stage 3 warm
field plus log-amplitude 1/2 probes of width 3 mm, whereas V3 used `0.5 x warm`
plus amplitude 4 at width 2.5 mm. Six V4 candidates produced at least one
SCL-only event, but no candidate produced a joint ICL+SCL event. Therefore the
scalar minimum is diagnostic only and the frozen verdict is:

```text
REV10SA_V4_NO_JOINT_SHAFT_CANDIDATE
```

When all joint fractions are zero, the joint penalty is constant and the
remaining scalar rank reduces to route/support/OOD differences. The aggregator
must set `selected_candidate_id=null`; it may display the scalar minimum but
must not call it a balanced or eligible winner.

V4.1 tests representation continuity before any new optimizer. Every one of the
21 frozen V3 spectral fields is projected on a uniform sheet into an `18 x 18`
cubic B-spline field. No V3 score, winner identity, contact, shaft, onset, or
label is used to choose a source field. Preflight over the complete library must
retain all 21 candidates and satisfy maximum field-`h` RMSE below 0.005. The
observed preflight is RMSE `0.00316`, minimum correlation `0.99975`, minimum
top-5% Jaccard `0.971`, and design condition number `27.53`.

V4.1 uses the same network seed 1031 as a paired dynamics bridge. At least one
joint event is required before any candidate can be called eligible. Passing
the bridge establishes that the stable continuous coordinates preserve the old
capacity signal; it does not establish patient recovery. Only then may V5
optimize continuous spline/KL coefficients on separated fit and selection
network pools. `K`, Edge, `beta`, topology, and observation-conditioned support
priors remain closed.

V4.1 completed `21/21` workers with zero failure and zero runaway. The stable
spline bridge preserved the V3 shaft partition on the paired seed. In
particular, uniform 09 changed from `3 joint + 3 ICL-only` to
`2 joint + 4 ICL-only`, while uniform 12 changed from
`2 joint + 9 SCL-only` to `4 joint + 9 SCL-only`. The selected exploratory
bridge field is uniform 12 with joint fraction `4/13=0.308`; uniform 09 has
`2/6=0.333`. Both remain far below patient train joint fractions near 0.95, and
both have high OOD. The safe result is stable-coordinate capacity preservation,
not patient repertoire recovery.

V5 is an adaptive continuous interpolation round on fit seed 1031. Its four
anchors are selected by a frozen non-spatial rule: the Stage 3 reference, the
two highest-joint V4.1 candidates, and one additional lowest-route candidate.
The resulting source IDs are uniform 09, uniform 12, uniform 06, and the Stage
3 reference. This use of patient-training score is optimizer feedback; it does
not change the uniform `18 x 18` basis or add parameters near contacts.

For every unordered anchor pair, V5 freezes three interior fractions under two
continuous paths:

```text
s_t = (1-t) s_left + t s_right
q_t = (1-t) exp(s_left) + t exp(s_right),  s_t = log(q_t)
```

Together with the four anchors this gives 40 coefficient-unique fields. The
first path interpolates the latent log field; the second preserves a union of
positive density support. Neither path defines components or cores. Values on
unobserved sheet regions remain smooth-prior extensions and are not identified
by patient data. V5 is fit-only; a diverse subset must be evaluated on network
seeds 1032/1033 before any fresh-seed confirmation.

V5 completed `40/40` workers on fit seed 1031 with zero failure and zero
runaway. Latent-linear interpolation did not produce a joint-shaft event in
any of its 18 interior fields. Density-mixture interpolation did: the fit
minimum `density(uniform09, uniform12; t=0.25)` produced 6 events, including
3 joint-shaft events, with direction counts `3/3` and OOD fraction `0.50`.
This is a sparse fit-network signal, not recovery. It suggests that retaining
the union of positive field support is more useful than translating the log
field between endpoints, but the event denominator is too small to interpret
the spatial extrema or mode proportions biologically.

V5.1 freezes eight fields before reading any selection-network outcome: the
V5 fit minimum, joint-positive training anchors, all V5 Pareto fields, and all
density-mixture points on the fit winner's anchor pair. The field basis and
coefficient locations remain uniform over the sheet; patient observations are
used only to rank complete simulated fields. Every candidate is run on both
network seeds 1032 and 1033. A field is selection-eligible only when it has at
least two pooled joint-shaft events and at least one joint-shaft event in each
network. This is one minimal cross-network realizability requirement, not a
claim that the patient event distribution has been recovered. No V5.1 result
may alter the frozen library, open Edge or beta, or move basis functions toward
contacts.

V5.2 is a fresh-development-network confirmation, not another search. It
freezes three complementary fields before seeds 1041-1043 are read: the V5.1
score winner, the eligible V5.1 field with the largest pooled joint-event
support, and the original Stage 3 field. Each field is evaluated once per
network. Confirmation requires at least three pooled joint events spanning at
least two of three networks. The primary report is the full per-network table;
the scalar minimum remains secondary because direction distance, joint support,
and event yield trade off strongly.

The V5.2 pooled verdict is not sufficient. Post-run eventwise cross-tabulation
showed a shaft-partition solution: for the score winner, direction A had
`0/8` joint events and all 8 were OOD, whereas direction B had `18/18` joint
events and all 18 were in-distribution. The joint-support anchor showed the
same split (`0/27` versus `31/31`). Therefore KMeans/direction AMI of 1 reflects
shaft participation, not recovery of two patient directions. The corrected
selection unit is now a mode-specific event satisfying both joint-shaft
participation and the frozen class-conditional patient support. Every future
candidate must provide such events for A and B separately; pooled joint count
cannot compensate for a missing mode.

V6 is a one-dimensional continuous-field boundary refinement, not a new field
family. The frozen V5 library contains no field with patient-supported joint
events in both directions. Uniform 12 supplies joint in-distribution B events,
while the density path from uniform 12 toward uniform 06 first supplies one
joint in-distribution A event at `t=0.25` after B support has disappeared. V6
samples this same complete-field path at increments of 0.025 over `t=[0,0.25]`.
No contact coordinate is used to place a parameter. Fit eligibility requires
at least one joint in-distribution event in each direction; if none exists, the
tested path has no observed coexistence window and no selection round follows.
