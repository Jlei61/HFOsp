# Axis-only seven-band experiment: endpoint vs gradient-primary field concordance

date: 2026-07-19
status: DESIGN (approved, pending spec review)
relates-to: `docs/archive/topic5/fig3_ictal_gradient_r3_full_recompute_run_form_2026-07-18.md`

## 1. Question

Did the seven-band ictal field-concordance change — old **endpoint** F2 "6/7 pass FWER"
→ current **gradient** "2/7 pass FWER" — come from the **axis definition**?

Answer it the only defensible way: compute **endpoint-axis R3** under the *identical*
locked pipeline as the current gradient R3, then **directly contrast the subject-level
margins per band (paired)**. Never infer an axis difference from "one version significant,
the other not" (significance-difference fallacy).

## 2. Held constant — identical to the gradient run (the "everything else")

Reused verbatim from `scripts/run_topic5_figure3_ictal_grid_rebuild.py` at N=161:

- cohort: 17 subjects / 167 seizures (`all_phenotype_matched`), strict 106 / gamma 61;
- activation: baseline robust-z log band power, clinical onset `[0,10] s`, the 7 primary
  bands from `config/topic5_v2_phase1.yaml`;
- common finite mask: `contact_order ∩ caches ∩ finite in all 7 bands ∩ finite in BB150
  anchor`, fail-closed `>= 6` — **the BB150-anchor finiteness constraint stays even though
  the anchor is not scored, so the mask is byte-identical to the gradient run**;
- σ rule: one σ per subject (`subject_fixed`), applied to A and B, all bands, observed and
  every null draw;
- grid: adaptive per-plane, y-symmetric, **N=161 primary** (81 sensitivity);
- null: coherent all-contact 1000-perm shuffle, seed 20260718; **the same physical-contact
  mapping per (subject, seizure, draw) reused across bands and A/B** (band-independent seed);
- readout: A/B `maxAB` with candidate-specific overlap gate + abs-max mirror;
- fold: seizure→subject median before any cohort test (no pooling);
- cohort stats: seven-band coherent maxT pFWER + coherent cohort spatial-null permutation p.

## 3. The one thing that changes — the axis

For `--axis endpoint`, `SubjectField` builds the **endpoint plane per template** instead of
reading the frozen gradient planes:

```
cores_T = build_endpoint_cores(rank_T, valid_mask, k_primary=3)   # top-3 source / sink
fr      = compute_axis_frame(coords, cores_T.source_idx, cores_T.sink_idx)
u       = normalize(fr.sink_centroid - fr.source_centroid)        # the endpoint (swap) axis
points  = make_normalized_plane(coords, u, ...)                   # SAME plane builder, different u
```

- `build_endpoint_cores` / `compute_axis_frame` from `src.propagation_skeleton_geometry`
  (the canonical swap/main-analysis axis; k_primary=3 confirmed). `valid_mask` = all
  `contact_order` contacts (they carry positive support in both templates by the frozen-field
  keep rule), matching the canonical `build_endpoint_cores(rank, ones, k_primary=3)` usage.
- `make_normalized_plane` from `src.topic5_template_axis_field` — the **same** normalized-plane
  builder the gradient uses; only the axis direction `u` differs, so plane normalization is
  identical.
- endpoint is **per-template A/B for all 17 subjects** (own-style routing): endpoint-A from
  `rank_a`, endpoint-B from `rank_b`; separate planes, separate support/earliness.
- `support_a/b`, `earliness_a/b`, `coords`, `rank_a/b` are the **same frozen-field values** as
  the gradient run — the endpoint changes only the projection plane, not which contacts /
  support / earliness feed the field.
- σ per subject = median-nearest-neighbour spacing on the endpoint-A plane points
  (`subject_fixed`: same σ for A and B), i.e. the identical σ *rule*, a different σ *value*.
- fingerprint gate: still verify the frozen field `fingerprint_sha256` (source data unchanged).

## 4. Confound — stated explicitly everywhere

Endpoint is per-template A/B; gradient-**primary** is shared-else-own. So this contrast
measures the **endpoint package** (endpoint-cores axis + per-template routing) against the
**gradient-primary package** — it does **not** isolate the axis alone. Every report/figure
caption says so. A pure axis-only isolation (matched per-template routing on both) is a
possible follow-up, deliberately **out of scope** here (user chose endpoint-vs-primary).

## 5. Implementation

**5.1 Runner (`scripts/run_topic5_figure3_ictal_grid_rebuild.py`) — additive flags:**
- `--axis {gradient,endpoint}` (default `gradient`; preserves current behaviour).
- `--score-bands-only`: keep the anchor in the common mask (mask unchanged) but skip scoring
  the BB150 anchor activation and skip the parent pooled/strict/gamma groups; compute only the
  7 bands. Endpoint runs are seven-band-only with the identical mask.
- `SubjectField` gains an endpoint branch: when `axis == "endpoint"`, build endpoint planes
  as in §3, set `route = "endpoint"`, `pts_a/pts_b` = endpoint-A/B points, σ from endpoint-A.
  When a template's endpoint cores are degenerate (`< 3` valid source or sink, or a zero-length
  axis), fail closed for that subject and record the reason (no silent fallback to gradient).

**5.2 Endpoint calc root:**
`results/topic5_ictal_recruitment/field_concordance_grid_endpoint_axis/n161_endpoint/`
(gradient primary at `.../field_concordance_grid_method_sensitivity/n161_subject_fixed/`
is reused, never re-run).

**5.3 Comparison (`scripts/run_topic5_axis_only_endpoint_vs_gradient.py`):**
Load both runs' `multiband_cohort.csv` + `multiband_subject.csv` +
`multiband_subject_null_draws.npz`. Emit:
- per-band side-by-side: endpoint vs gradient `D` / Δ-cohort / `coherent_cohort_spatial_null_p`
  / `seven_band_maxt_pfwer`;
- **direct per-band subject-level margin contrast** = per subject/band
  `margin_endpoint[s,b] - margin_gradient[s,b]` (margin = `D - Nmed`), with median effect, IQR,
  n positive, paired two-sided Wilcoxon, and subject sign-flip p — folded per band and folded
  band→subject. This is the answer.
- output `axis_only_endpoint_vs_gradient_{per_band.csv,contrast.csv,summary.json}`.

**5.4 Figure (`scripts/paper_figures/plot_fig3_axis_only_endpoint_vs_gradient.py`):**
- Panel A: seven bands, endpoint vs gradient per-subject Δ (two coloured groups per band) +
  cohort bar per axis + each axis' own seven-band maxT-pFWER star;
- Panel B: direct per-band margin contrast (endpoint − gradient), per-subject points + cohort
  bar + paired-test star;
- caption states the axis+routing confound and "gradient primary / endpoint sensitivity".
- staging `results/paper-ready-figure/fig3_ictal_field_concordance_grid_method_sensitivity/axis_only/`.

## 6. Tests (TDD)

New tests in `tests/test_topic5_gradient_grid_field.py` (pure) and a runner check:
1. endpoint axis wiring: for a synthetic rank/coords, the endpoint axis `u` equals
   `normalize(sink_centroid - source_centroid)` from `build_endpoint_cores(k=3)`, and
   `make_normalized_plane(coords, u)` places contacts along it; endpoint plane ≠ gradient plane.
2. degenerate endpoint cores fail closed (raise / mark unavailable), never fall back to gradient.
3. **identical-pipeline invariant**: gradient and endpoint runs (n_perm=20) produce the SAME
   event list, the SAME common mask, and the SAME per-event coherent permutation hashes
   (`permutation_mapping_audit_summary.csv`) — only plane/score differ.
4. `--score-bands-only` keeps the anchor in the mask (mask identical to a full run) but omits
   anchor scoring and parent groups.

## 7. Success criteria

- Endpoint seven-band R3 computed under the identical pipeline, with test-3 confirming the
  event list / mask / coherent-perm hashes match the gradient run.
- The **direct per-band subject-level margin contrast** (endpoint − gradient) reported with
  paired p — the headline result, not a "6/7 vs 2/7" star comparison.
- Figure + comparison CSVs + a short plain-language discussion.
- Gradient stays primary; endpoint is a sensitivity; the axis+routing confound is explicit.

## 8. Non-goals (YAGNI)

- No pure axis-only isolation (matched routing) — deferred.
- No pooled/BB/gamma parent groups for endpoint (seven-band only, user-chosen scope).
- No re-run of the gradient primary (reuse `n161_subject_fixed`).
- No σ-multiplier or k-sweep; k_primary=3 fixed (canonical).
