# interictal_propagation vs masked — diagnostic comparison artefacts

This directory holds **only** the orig-vs-masked diagnostic comparison artefacts
for the Topic 0 lagPatRank phantom-rank audit (see
`docs/topic0_methodology_audits.md` §3.1 + §5).

## Contents

- `pr2_comparison.csv` — per-subject PR-2 cluster label shift (orig vs masked)
- `pr2_comparison_summary.md` — narrative summary of the table above
- `figures/cluster_fraction_shift.{png,pdf}` — max |orig − masked| cluster fraction per subject
- `figures/label_jaccard_distribution.{png,pdf}` — PR-2 label-level Jaccard distribution + audit-vs-PR-2 AMI scatter
- `figures/README.md` — Chinese description of the two figures (关注点 included)

## Where the masked data tree lives now (2026-05-22 D1 migration)

The masked re-derivation data tree (per-subject JSONs, cohort_summary,
template_anchoring/, rank_displacement/, template_pairing/,
pr6_step6_held_out_template/, template_share_switching/) was moved from the
old nested location (`interictal_propagation_vs_masked/interictal_propagation_masked/`)
to its canonical top-level path:

    results/interictal_propagation_masked/

This aligns with the AGENTS.md "parallel directory" convention for Topic 0
audit-triggered reruns. All downstream scripts and archive docs reference
that new top-level path.
