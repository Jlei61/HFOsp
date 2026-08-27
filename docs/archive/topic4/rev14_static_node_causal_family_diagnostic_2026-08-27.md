# Topic 4 rev14 static Node causal-family diagnostic

## Safe conclusion

The frozen `exact_off` static Node field generates a stable pooled K=2 split
and retains the frozen old A/B patient direction geometry on contact-split
cross-fit in all three networks. It is nevertheless only a partial
reconstruction: one network has poor natural-KMeans alignment, more than half
of isolated causal families are OOD, dual-shaft participation is low and the
matched full-distribution score remains above the patient block-floor scale.
Rev14 must improve this static Node field before EE, E-to-I or Z/M are changed.

## Frozen analysis contract

- No new SNN simulation was run.
- Primary observations are returned, isolated complete causal families; all
  members of overlap-connected episodes are excluded. Source-evaluable events
  form a separate topology sidecar and are not required by the contact loss.
- OOD, single-shaft and missing-SCL events remain in the patient-training loss.
- Patient training arrays and the frozen rev11 direction classifier are used.
  Patient held-out arrays are not loaded.
- Networks are scored separately and then averaged with equal network weight.
- Natural KMeans and contact-split cross-fit are diagnostics, not fit targets.

An initial Phase 0 draft incorrectly replaced old A/B direction identity with
the shaft-aware K=2 extent labels. Their training AMI is 0.011, so that mapping
is invalid. The draft negative cross-fit matrix and `J14=2.4853` are
superseded. This archive reports the corrected old-A/B result only; a regression
test now locks the primary mode definition.

## Counts and patient-training score

Across seeds 2311--2313 there are 86 primary isolated families and 72
Fig.4-readable families. Per-network counts are 26/31/29 and 24/26/22.

The equal-network legacy soft diagnostic is 1.4688. Mode-specific mean errors
are 0.9326 and 1.1669; weakest-mode LSE is 1.0771. Occupancy JS is 0.0087,
contrast alignment is 0.3906 and contrast loss is 0.6094.

This legacy number is not the rev14 optimization reference. The frozen
`topic4_rev14_j14_v1` matched 6-versus-6 objective is 2.0527. Its mode means are
1.2076 and 1.3024, weakest-mode LSE is 1.2795, confidence-adjusted readable
in-support effective supports are 4.97 and 7.98, overlap fraction is 0.3832 and
support loss is 0.6895. Per-network overlap fractions are 0.366, 0.311 and
0.473. All rev14
candidates must be compared with this same objective and sampling contract;
the 1.4688 legacy diagnostic cannot select a field.

The first anti-cheating draft reused pseudo-model events in its patient
reference and is superseded. The frozen control instead uses disjoint recording
blocks for pseudo-model and patient reference, applies every degeneration to
the same pseudo-model events, and repeats the comparison at three objective
seeds. Sparse or OOD events remain in the contact-distance loss but do not count
as patient-mode support.

The clean block-disjoint control gives `J14=0.8222` for a patient-training
pseudo-model versus a disjoint patient-training reference. SCL censoring rises
to 1.4812, a repeated single event to 2.0122, a single-mode sample to 2.6524,
fully ambiguous assignments to 3.0125 and zero events to 3.0966. Every
degeneration is worse at each of three frozen objective seeds.

## Historical zero-simulation rescore

All 366 complete 20-s static-Node trajectories from Stages Z, AG, AK and AL
were re-audited and rescored without running the SNN. Their worker field,
mapping, mechanism-off state, clean runtime provenance and NPZ hashes all
match the frozen manifests. The inventory contains 12,381 contact-primary
families, including 2,482 with fewer than three finite contacts; 4,642 members
of overlap-connected episodes are excluded.

Within the original stage-specific seed pools, the best selectable fields are:

- Stage Z: `stage_z_g11_p`, `J14=1.9884`, versus anchor 2.1979;
- Stage AG: `stage_ag_g09_m`, `J14=1.9807`, versus anchor 2.2149;
- Stage AL: `stage_al_m100_d050`, `J14=2.0069`, versus anchor 2.1058.

Stage AK remains diagnostic-only. These raw values are not compared across
seed pools and no historical candidate replaces `exact_off` as the rev14 CRN
comparator. The Stage-AG winner differs from the earlier draft because sparse
and OOD families no longer inflate mode-evidence support.

## KMeans and patient geometry

Pooled natural KMeans produces clusters of 33 and 39 events with seed AMI 1.0,
silhouette 0.313 and balanced alignment 0.875. This pooled summary hides
network heterogeneity: balanced alignment is 0.417, 0.885 and 0.846 for the
three networks. The K2-versus-K1 held-out GMM diagnostic is negative and is not
used as a gate.

Using the frozen old A/B direction labels, the equal-network contact-split
matrix is:

```text
[[ 0.5476, -0.7048],
 [-0.5575,  0.4524]]
```

The per-network signed margins are 0.3155, 0.4286 and 0.5417. Therefore the
direction geometry survives contact splitting, although this does not rescue
the OOD, dual-shaft and full-distribution failures.

## Support diagnostics

OOD fractions are 0.577, 0.516 and 0.552, or 47/86 pooled. Joint ICL-SCL
participation is 0.115, 0.129 and 0.241. The current field therefore generates
many internally structured events outside the frozen patient support and only
rarely recruits both shafts as one isolated causal family.

## Artifacts

- `results/topic4_sef_hfo/data_driven_node_dualmode_rev14/static_node_causal_family_diagnostic/exact_off_static_node_rescore.json`
- `results/topic4_sef_hfo/data_driven_node_dualmode_rev14/static_node_causal_family_diagnostic/exact_off_static_node_fig4_bundle.npz`

The JSON records all input hashes, original event indices, runtime-path hashes,
the analysis commit and runtime dirty state. It also separates
`contact_primary`, `topology_primary` and `fig4_kmeans_readable` masks. The NPZ
is the sole bundle for
subsequent Fig.4-style rendering of this diagnostic.

## Decision

```text
STATIC_NODE_DIRECTION_GEOMETRY_PARTIAL_PASS
/
PATIENT_SUPPORT_AND_FULL_DISTRIBUTION_FAIL
/
STATIC_NODE_FIELD_REFINEMENT_REQUIRED
```

This result does not authorize changes to connectivity or slow variables and
does not use patient held-out data.
