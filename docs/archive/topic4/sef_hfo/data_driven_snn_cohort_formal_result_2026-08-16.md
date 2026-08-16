# Data-driven SNN cohort: formal result (2026-08-16)

Status: `SAME_NETWORK_K2_INSUFFICIENT`, with
`OBSERVATION_LAYOUT_DEPENDENCE_UNRESOLVED` also not met. **Two of the frozen
gates failed. The pre-registered positive claim is not licensed.**

Run pinned at commit `96618174`; 152 workers across three stages, no worker
failed and no run went runaway. Design, denominators and all thresholds were
frozen before any formal patient score existed; see
`data_driven_snn_cohort_canary_and_formal_freeze_2026-08-15.md`.

Artifacts: `results/topic4_sef_hfo/data_driven_snn_cohort_v1/formal/`
(`cohort_result.json`, `conditioning_value_audit.json`,
`cohort_subjects_canonical.csv`, `figures/`).

## 1. The question, in plain terms

Can a spiking network conditioned on a patient's own interictal data reproduce,
on recording blocks it never saw, *which contact fires in which position* —
well enough to beat the same contacts relabelled inside their own shaft?

Primary readout: a target-blind contact-order layout for all 34 subjects.
Sensitivity readout: the real 3-D implant geometry for the 28 that have one.

## 2. Primary endpoint: passes its own gate, but barely and only on one layout

| quantity | value |
|---|---|
| subjects beating their matched within-shaft shuffle | 23/34 = 67.6 % (gate 60 %) |
| median advantage over the shuffle | +0.00631 |
| the same as a fraction of the null level (null median 0.193) | 3.3 % |
| Wilcoxon signed-rank on subject deltas | p = 0.0426 |
| sign test on the same deltas | p = 0.0576 |
| per-subject permutation p, median over four networks | 0.342 |
| subjects reaching p ≤ 0.05 on their own null | 3/34 |

The frozen gate is met, but every companion statistic says the effect is small
and marginal: the more conservative sign test does not clear 0.05, the typical
subject's own permutation test is nowhere near significance, and the median
advantage is three percent of the quantity being compared.

Confounds the canary told us to check both came back clean, so the effect is
not an artefact of montage size or event yield:

- advantage versus in-distribution event count: Spearman −0.171, p = 0.33;
- advantage versus contact count: Spearman −0.101, p = 0.57 — the raw-margin
  inflation the canary showed did cancel inside the matched null, as designed;
- dropping the top event-count quartile: n = 25, median +0.00754, p = 0.042.

One robustness flag: the per-network pass fractions are 0.65, 0.65, 0.59 and
0.68, so on confirmation network 1683 alone the cohort would sit below its own
60 % gate.

## 3. Gate not met: one network rarely holds both modes

15/34 subjects (44.1 %) had a single confirmation network produce two
reproducible clusters that each match a distinct patient mode, against a frozen
floor of 50 %. Per-subject counts over the four networks: 8 subjects at 4/4,
7 at 3/4, 12 at 2/4, 3 at 1/4, 4 at 0/4.

The median-performing subject shows what the failure looks like from inside.
`epilepsiae_590` passed its own shuffle test, and its unsupervised clusters
correlate with the patient templates at **+0.06** (model mode A against patient
TA) and **+0.56** (model mode B against TB), with a silhouette of 0.16. One
patient mode is recovered; the other is not, and the two model clusters are
essentially the two signs of a single axis rather than two distinct modes.

## 4. Gate not met: the effect does not survive the readout geometry

| | contact-order layout (34) | real implant geometry (28) |
|---|---|---|
| median advantage | +0.00754 | **−0.00065** |
| Wilcoxon p | 0.043 | **0.981** |
| subjects with a positive advantage | 23/34 | 13/28 |
| subjects passing their own gate | 67.6 % | 46.4 % |

Sign agreement between the two layouts is **14/28 — exactly chance**. The
per-subject advantage on one layout carries no information about the advantage
on the other.

Two readings survive this data and it cannot separate them:

1. the contact-order effect is an artefact of an idealised readout — parallel
   shaft rows filling the sheet, ordinals evenly spaced;
2. the real geometry is a lower-signal readout — its minor axis ranges from
   0.3 mm to 15 mm against a uniform 16 mm for the canonical layout — and
   simply cannot detect an effect this small.

Because it cannot separate them, the pre-registered status is exactly right:
observation-layout dependence is *unresolved*, not resolved against the model.

## 5. Was the patient conditioning load-bearing?

Asked properly, on held-out data. Comparing each subject's own selected
candidate against one shared candidate (`rot090idn_node`, the one most subjects
selected) applied to everybody:

- pass fraction 67.6 % with each subject's own choice, 55.9 % with the shared
  one;
- excluding the 6 subjects whose own choice already *is* the shared candidate:
  19/28 did better with their own choice, median advantage +0.00604,
  Wilcoxon p = 0.045.

So there is weak evidence that per-subject candidate choice helps — but it
helps recover an effect that itself does not survive the geometry swap. The
conditioning result inherits that limit and cannot be quoted on its own.

**A number that must not be quoted**: on the *training* split the same
comparison gives median +0.0273, 28/34 improved, p = 3.8e-6. That comparison is
circular — a minimum over twenty-four candidates beats a fixed candidate on the
split that chose it even under exchangeable noise — so its sign and its p-value
are what the null predicts, not evidence.

## 6. What may and may not be said

May be said:

> Across 34 stable-bidirectional subjects, patient-conditioned continuous-field
> spiking networks matched held-out contact-order structure slightly better than
> matched within-shaft contact-identity nulls under an idealised contact-order
> readout (23/34, median +0.0063, p = 0.043), but the advantage vanished under
> the patients' real implant geometry (median −0.0007, p = 0.98, sign agreement
> 14/28), and fewer than half the subjects had a single network hold two
> clusters matching distinct patient modes (15/34).

May not be said: that the cohort was reproduced; that patient interictal
activity was recovered; that the two-mode repertoire was reproduced; that the
model localises anything anatomical; that the contact-order result is a
cohort-level positive finding. The canonical layout is a target-blind readout
of contact order, not anatomy, and the one arm that used real anatomy showed
nothing.

## 7. Where this leaves the line

The shared-morphology transfer family, as frozen here, does not clear the bar.
Two directions remain open and are *not* claimed to be promising by this run:

- the pre-registered fallback in the plan — a candidate library that reads no
  patient contact, rank or mode at all (full-sheet continuous B-spline plus
  low-frequency spectral), which this run did not test;
- resolving the layout dependence directly, by giving the real-geometry arm the
  same effective coverage the canonical arm has, so that a null result there
  means the effect is absent rather than undetectable.

Neither should be started as a rescue of this result. The result stands as a
two-gate failure.
