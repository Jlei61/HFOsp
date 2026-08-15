# Topic 4 data-driven SNN cohort execution plan

## C0: target and geometry audit

- Add a producer for all 34 masked stable-K=2 subjects.
- Split patient data by recording block before fitting masked KMeans.
- Freeze per-subject TA/TB descriptors and train/held-out samples.
- Build a geometry-only PCA placement for every coordinate-eligible subject.
- Write `cohort_eligibility.csv`, `cohort_target_audit.json`, per-subject
  JSON/NPZ files and a diagnostic figure with PNG/PDF/README.
- Tests must cover phantom ranks, block leakage, label alignment, geometry sign
  determinism and variable contact counts.

## C1: shared candidate library

- Generate every continuous field and EE/E-to-I coefficient candidate without
  reading patient targets or contact geometry.
- Hash the complete library before the first patient score is computed.
- Include Node-only and uniform-field controls plus contact-permutation nulls.
- Keep topology, delays, total field budget and incoming pathway budgets frozen.

## C2: six-subject canary

- Use subjects with 6, 10, 15, 16, 38 and 52 contacts.
- Run four candidates on two fit and two selection network seeds for 12 s.
- Require at least four subjects with evaluable same-network K=2 and no
  contact-count-dependent detector failure.
- Render the two accepted Fig.4-style panels for each evaluable subject and
  inspect actual PNG/PDF output.

## C3: formal cohort run

- Freeze candidate library, detector, fit/selection/confirmation seeds and
  subject list after C2.
- Run 20 s simulations on fresh confirmation seeds.
- Use managed `systemd-run --user -> nohup` workers, one numeric thread each,
  measured-RSS memory admission and 300-s controller waits.
- Do not launch while the rev11 pathway confirmation is running.
- A completion monitor aggregates, audits, draws the final cohort figure and
  sends a desktop notification.

## C4: adjudication

- Report 34 target subjects, geometry-eligible denominator and every exclusion.
- Use subject as the independent unit.
- Separate supervised patient geometry, natural KMeans, event support and OOD.
- Select the representative figure subject by median performance.
- Write the final claim only after the figure, sidecars and held-out audit agree.
