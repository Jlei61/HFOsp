# Topic 5 ordered-history conditional-information and architecture audit v0.1

## 1. Scientific question

The primary estimand is operational rather than a literal finite-sample mutual
information estimate:

\[
\Delta_{\mathrm{order}}
=
\operatorname{NLL}(\text{static + unordered prefix})
-
\operatorname{NLL}(\text{ordered prefix}).
\]

It asks whether contact recruitment order inside an interictal group event
contains held-out next-set information beyond patient/contact structure and
the unordered set of already recruited contacts.

The RNN is a neural-data-constrained predictive system-identification model.
Its state is an event-indexed phenomenological coordinate. It is not a
cell-level circuit, a biological E/I state, or a continuous-time slow
variable.

## 2. Two temporal axes must not be mixed

### Primary: within-event rank history

- one recurrent step = one recruitment rank set inside one group event;
- state resets at every group-event boundary;
- chronological train80/heldout20 remains frozen for each patient;
- no ictal target is read during fitting or model selection.

### Exploratory: across-event pre-seizure history

- one recurrent step = one complete interictal group-event field;
- timestamps may be used only to establish causal order and eligibility;
- IEI is not an input and no real-time time constant is estimated;
- repeated seizures with the same last available event are one distinct
  history unit, not independent samples;
- this branch cannot become a primary result without an independent target
  cohort.

## 3. Frozen information controls

For the primary within-event task:

1. `static_contact_hazard`: contact/patient scaffold, no prefix identity;
2. `unordered_prefix`: DeepSets-style pooling of the recruited prefix;
3. `last_set_first_order`: current recruitment front only;
4. `within_event_rank_shuffle`: preserve participants, destroy within-event
   order;
5. `ordered_prefix`: true rank-set order;
6. evaluation interventions: reverse observed prefix, reset state after ranks
   1/2/3, and delete the earliest rank set.

`event_order_shuffle`, `block_shuffle`, and `history_target_circular_shift`
are not controls for a state that resets between group events. They are
reserved for the exploratory across-event branch.

The accepted matched-window experiment is frozen evidence for history depth:
H2 exceeds H1, H3 exceeds H2, full history does not exceed H3, and ordered H3
exceeds matched H3 rank shuffle. It is reused rather than retrained. Hence the
candidate scientific object is bounded two-to-three-rank history, even when a
full-history implementation is retained as an architecture reference.

## 4. Frozen architecture ladder

All neural models share the contact encoder, local patient offset, action
decoder, STOP definition, event-balanced NLL, outer-patient training and
held-out-patient calibration.

1. static MLP;
2. unordered DeepSets-style prefix MLP;
3. first-order last-set MLP;
4. linear state-space recurrence;
5. vanilla tanh rate RNN;
6. GRU;
7. stable low-rank leaky RNN with ranks 0, 1, 2 and 4.

Hidden size is 32. Parameter counts are reported rather than asserted to be
identical. The primary comparison is ordered recurrence versus the unordered
prefix model; architecture comparisons determine whether this increment is
specific to gating, available to a linear state, or stable across recurrent
families.

Capacity fairness is evaluated as a target-sealed sensitivity, not a new
selection pass. With the accepted GRU at hidden size 32 (11,246 parameters),
the linear state model is repeated at hidden size 64 and the vanilla RNN at
hidden size 48, placing both totals within 10% of the GRU while preserving the
encoder, local-offset interface, decoder, loss and optimizer. The fixed
hidden-size ladder remains primary because a larger hidden state changes the
state-dimensional prior; the matched-parameter ladder checks whether an
apparent architecture difference is merely a capacity difference. Its two
comparisons are reported with Holm correction and do not participate in model
selection.

No Dale's-law constraint is used.

## 5. Statistics and claims

- Patient is the primary statistical unit; seeds are repeated optimization
  measurements and are collapsed within patient.
- Primary endpoint: held-out event-balanced next-set/STOP NLL.
- Secondary endpoints: top-1 next-set accuracy, STOP Brier score, and
  contact-rank distribution error.
- Report median paired patient difference, sign count, exact/paired Wilcoxon,
  and bootstrap confidence interval.
- Because the target-facing nongated model is chosen from the architecture
  ladder, protect all recurrent-vs-unordered median gains with joint patient
  sign flips and a maximum statistic across the tested recurrent families.
  The selected model's nominal P is not an independent confirmation.
- Do not use a global hard gate to erase informative partial results.

Allowed strongest conclusion:

> Ordered within-event recruitment history contains cross-architecture
> predictive information beyond static contact structure and an unordered
> prefix, if this increment is positive for at least two recurrent families
> and survives the matched within-event order null.

Disallowed conclusions:

- a two-dimensional biological seizure manifold was discovered;
- a GRU hidden unit is an excitatory or inhibitory population;
- rank-step persistence is a biological time constant;
- a visually low-dimensional trajectory independently validates the model.

## 6. Early-ictal readout

The frozen strict target remains clinical-onset `[0,10] s`, `1–150 Hz`,
baseline-normalized contact energy (16 patients, 106 seizures). It is reused,
not independent confirmation.

The early-ictal analysis asks whether a frozen ordered-history field adds to:

1. static train80 participation;
2. regularized unordered contact/rank summaries;
3. matched shuffled-order fields.

Any readout must be selected without target values, evaluated patient-first,
and use all-contact/channel-shuffle inference as the primary null. Hidden
PC dimensionality is descriptive only. State analysis is restricted to
readout-relevant directions and explicit history interventions. For the
selected nongated recurrence, report the local rank-step Jacobian only after
projection through the fitted hidden-to-contact-logit map; call it
event-indexed retention, never a biological time constant.

## 7. Stop rules

- Engineering failure, leakage, fingerprint drift, NaN, or incomplete LOSO:
  stop and repair.
- If linear, vanilla, GRU and low-rank models all fail to exceed unordered
  prefix, close the ordered-state claim.
- If recurrence beats unordered prediction but no frozen ordered residual
  improves the reused early-ictal target, retain a within-event sequence result
  and do not claim cross-state history transfer.
- Across-event seizure-history analysis remains exploratory unless at least
  three distinct causal histories exist per included patient and duplicate
  histories are grouped.
