# Topic 4 rev14 static Node causal-family diagnostic

## Safe conclusion

The frozen `exact_off` static Node field generates a stable pooled K=2 split,
but it does not reconstruct the patient's two propagation modes on fresh
networks. The positive pooled KMeans result is limited by poor per-network
alignment, negative contact-split patient cross-fit margins in all three
networks, more than half of isolated causal families being OOD, and low
dual-shaft participation. Rev14 must therefore improve the static Node field
before EE, E-to-I or Z/M are changed.

## Frozen analysis contract

- No new SNN simulation was run.
- Primary observations are returned, source-evaluable, isolated complete
  causal families; all members of overlap-connected episodes are excluded.
- OOD, single-shaft and missing-SCL events remain in the patient-training loss.
- Patient training arrays and the frozen rev11 direction classifier are used.
  Patient held-out arrays are not loaded.
- Networks are scored separately and then averaged with equal network weight.
- Natural KMeans and contact-split cross-fit are diagnostics, not fit targets.

## Counts and patient-training score

Across seeds 2311--2313 there are 86 primary isolated families and 72
Fig.4-readable families. Per-network counts are 26/31/29 and 24/26/22.

The equal-network legacy soft diagnostic is 2.1137. Mode-specific mean errors are
0.7035 and 1.7659; weakest-mode LSE is 1.5962. Occupancy JS is 0.0087, but this
apparently favorable occupancy does not rescue the weak mode: contrast
alignment is only 0.1392 and contrast loss is 0.8608.

This legacy number is not the rev14 optimization reference. The frozen
`topic4_rev14_j14_v1` matched 6-versus-6 objective is 2.4853. Its mode means are
0.9015 and 1.8201, weakest-mode LSE is 1.6533, confidence-adjusted effective
supports are 14.78 and 12.31, overlap fraction is 0.3832 and support loss is
0.5049. Per-network overlap fractions are 0.366, 0.311 and 0.473. All rev14
candidates must be compared with this same objective and sampling contract;
the 2.1137 legacy diagnostic cannot select a field.

## KMeans and patient geometry

Pooled natural KMeans produces clusters of 33 and 39 events with seed AMI 1.0,
silhouette 0.313 and balanced alignment 0.875. This pooled summary hides
network heterogeneity: balanced alignment is 0.417, 0.885 and 0.846 for the
three networks. The K2-versus-K1 held-out GMM diagnostic is negative and is not
used as a gate.

The equal-network contact-split matrix is:

```text
[[-0.7214, -0.6857],
 [ 0.4956,  0.5635]]
```

The per-network signed margins are -0.6643, -0.6857 and -0.8143. Therefore the
model clusters do not recover the patient mode geometry on contacts withheld
from assignment.

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
STATIC_NODE_K2_INTERNAL_STRUCTURE_PRESENT
/
PATIENT_DUALMODE_GEOMETRY_NOT_RECOVERED
/
STATIC_NODE_FIELD_REFIT_REQUIRED
```

This result does not authorize changes to connectivity or slow variables and
does not use patient held-out data.
