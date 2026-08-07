# Topic 5 matched event-innovation recurrent transition v3.1

- **Status:** successor identification contract; code and synthetic calibration authorized;
  human test conditional on the frozen V3.0 train/validation handoff
- **Time step:** one complete interictal event
- **Primary scientific object:** an observable rank/precedence state with a patient-specific
  stable backbone and low-dimensional transition
- **Primary question:** whether event innovation must enter the latent state transition after
  observation and filtering capacity are held fixed
- **Output root:** `results/topic5_event_innovation_state_space/v3_1/`

## 1. Scientific identification question

V3.0 tests whether valid event innovations have future-directed and cumulative associations
with later propagation-state displacement. That result alone can still be an observer effect:
the event may simply improve estimation of a state that would have evolved anyway.

V3.1 asks the narrower mechanism-identification question:

> Given the same observation model, latent dimension and filtering update, does held-out
> sequence prediction require event innovation to enter the recurrent transition itself?

The comparison is nested. The event-driven model may not receive a richer decoder, a more
flexible measurement update, more history, different targets or different anchors.

## 2. Entry and independence contract

Implementation and synthetic tests may start immediately. A human transition test is opened
only if the predeclared V3.0 train/validation handoff detects a stable Goal 2 or Goal 3
rank/precedence signal. The V3.0 human test is not searched to decide model form.

V2.7 is not an entry Gate. SNN, ictal data, geometry, SOZ, pathological-axis labels and old
heldout20 remain forbidden. SNN is an independent model line.

## 3. Inherited observable contract

V3.1 inherits without refitting choices after test inspection:

- continuity units and reset/carry rules;
- train80 event inventory and chronological splits;
- rank/precedence primary, mode secondary and participation tertiary families;
- stable rank backbone `mu_0` and contact-space readout;
- validation-selected `K in {1,2,3,4}` and mandatory `K=6/8/full-rank` diagnostics;
- source-balanced dense training and non-overlap validation/test anchors;
- family-specific, blocked-cross-fitted innovations and innovation-validity labels;
- V3.0 nuisance, null and patient-first inference rules.

The observable state remains

\[
\mu_e=\mu_0+Lz_e,
\qquad
G_{e,ij}=\sigma(\mu_{e,j}-\mu_{e,i}).
\]

Only changes in `mu` and `G` are interpreted. Raw latent coordinates and recurrent matrices
are not contact graphs.

## 4. Shared observation and filtering model

All continuous models use the same observation family and future-blind filter:

\[
z_e^-=Az_{e-1}^+ + \eta_e,
\qquad
y_e\sim p_C(y_e\mid z_e^-),
\qquad
\nu_e=\operatorname{Innov}(y_e,z_e^-),
\qquad
z_e^+=z_e^-+K_f\nu_e.
\]

`C`, the family-specific likelihoods, `K_f`, state dimension, state/noise parameterization,
normalization, optimizer, regularization grid and training budget are shared. When joint
fitting is required, shared parameters are tied in one nested implementation; they cannot be
separately tuned for the event-driven arm.

## 5. Frozen model set

### T0 — fixed backbone

No time-varying latent state.

### T1 — autonomous observer-only transition

\[
z_{e+1}^-=Az_e^+ + \eta_{e+1}.
\]

The event updates the posterior estimate through the shared filter but has no additional
transition input.

### T2 — event-driven transition

\[
z_{e+1}^-=Az_e^+ + B\nu_e + \eta_{e+1}.
\]

The only added scientific term is `B nu_e` in the transition. The primary observable impulse
is the induced change in future `mu` and `G`, not `B` itself.

### T3 — discrete switching control

A small validation-selected switching state model uses the same observable families, anchor
set and comparable effective state capacity. It tests whether a few stable repertoire states
explain the sequence without a continuous innovation-driven transition.

No GRU/Transformer sweep, contact-graph recovery, mixture of autonomous fields or SNN prior is
part of V3.1.

## 6. Fairness and identifiability checks

Before human test, synthetic systems must verify:

1. T2 does not beat T1 when events only reveal an autonomous hidden state.
2. T2 recovers a known innovation-driven transition and observable impulse.
3. T3 wins or ties for true discrete switching.
4. T2 advantage disappears under state-matched innovation donors, source-coherent chronology
   nulls and safe shifts.
5. T1 and T2 have identical observation/filter preprocessing, anchor support and training
   adequacy; added effective capacity is reported and controlled by nested validation.

The V3.1 observable impulse must agree in sign/subspace and horizon profile with the frozen
V3.0 local projection. Disagreement prevents a state-transition interpretation.

## 7. Human test and inference

The 34-patient analysis remains exploratory. Training may use dense source-balanced anchors;
validation and primary test use non-overlap anchors; dense test is sensitivity only. Seeds
and folds are combined within patient.

The primary continuous patient effect is

\[
\Delta_{\mathrm{transition}}
=
\operatorname{Score}(T1)-\operatorname{Score}(T2),
\]

on held-out rank/precedence propagation score, so positive values favor the event-driven
transition. T2 is also compared with T3 and all V3.0 state-matched, chronology,
future-versus-past and nuisance controls.

Cohort reporting uses the patient median, bootstrap 95% CI, favorable-patient count,
two-sided Wilcoxon signed-rank and sign test. Patients are not assigned conjunctive PASS/FAIL
labels and result-defined subtypes are forbidden.

## 8. Interpretation levels

- **Observer-only:** T1 is sufficient; events improve state estimation but no separate
  transition input is required.
- **Discrete switching:** T3 is sufficient; the repertoire switches among a few states.
- **Predictive but unidentified:** T2 improves prediction but fails observable agreement or
  chronology/state-matched specificity.
- **Event-associated low-rank update:** T2 improves over T1 and T3, passes the frozen
  specificity tests and agrees with V3.0 observable impulses.

The last outcome licenses only: "event innovations are associated with subsequent low-rank
changes in the patient-specific propagation state." `Supports activity-dependent shaping`
requires frozen independent replication plus the V3.0 cumulative dose, alignment,
cancellation and compatible IEI-decay evidence. Causal plasticity remains prohibited.

## 9. Machine-readable release checklist

Human test requires:

1. the V3.0 train/validation handoff record;
2. shared T1/T2 parameter registry and equality checks;
3. synthetic observer-only, event-driven and switching recovery;
4. frozen anchor, dimension, optimizer, regularization and budget grids;
5. frozen observable-impulse agreement metric;
6. frozen null and cohort-inference rules;
7. config, source-list, code and input hashes.
