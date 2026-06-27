# Field-aligned swap subjects -> subject-specific SNN placement plan

**Date:** 2026-06-26  
**Status:** planning / case-selection gate, not a simulation result  
**Scope:** Topic 4 Stage 3 subject-specific SNN placement, using Topic 5 field alignment and PR6 masked rank-displacement swap-k nodes.

> **Review revisions (2026-06-26, agent audit).** v1 numbers all verified against artifacts
> (`results/interictal_propagation_masked{,_broad}/rank_displacement/per_subject/*`,
> `results/topic5_ictal_recruitment/axis_alignment/*`). Five corrections folded in below; the
> v1 case-selection text is preserved with strike-through context so the change is traceable:
>
> 1. **Montage-consistency contract (§2, §3).** v1 pulled swap from the *narrow* tree but
>    selected E253/E635 on *broad* field. Those cannot both hold for one subject. On the
>    montage where field passes, E253's swap is a strict **left-vs-right hippocampus** split
>    (k=10), not two small cores; E635 is a strict multi-shaft split. A broad masked
>    rank-displacement tree EXISTS and must be used when the field/geometry are broad.
> 2. **Cohort trade-off.** Whole-cohort scan: no subject has BOTH a clean primary-band
>    (broadband) field pass AND a clean unilateral two-small-core swap on a consistent montage.
>    "Two small cores" is largely a narrow-montage artifact. E958 (subdural grid) is the only
>    clean case. Expect this trade-off; do not hunt for a perfect 2nd case.
> 3. **Success criterion (§3C/§3D).** "Model produces two ordered templates" is ~geometry-forced
>    on a single EE axis (prior `core_model_s3_brakeoff` NULL: "仪器对齐非机制重现, 单一连接轴使
>    k=2 半被迫"). Success = **beats the §3D nulls**, not "produces two templates".
> 4. **Bidirectional method (§3C).** One-network two-equal-core spontaneous alternation is the
>    documented twoend_equal failure mode (one-core-dominance). Primary route = run each core
>    separately then pool (validated `pooled_bidir`); one-network is an optional harder test.
> 5. **Claim tier (§1).** Per-subject descriptive scaffold (n=2-3 case), NOT a cohort claim that
>    swap defines a causal pathological axis. Anatomical blocker: a single 2D excitable sheet
>    cannot represent a cross-midline bilateral axis (E253-broad).

---

## 1. Core scientific goal

We want a small number of patient-specific cases where two conditions both hold:

1. The interictal propagation field and early ictal activation field share a strong within-subject axis.
2. The same subject has PR6 masked rank-displacement evidence for source/sink role exchange between the two interictal templates.

The modeling hypothesis is deliberately narrow:

> The rank-displacement swap-k source-side and sink-side nodes identify two ends of a pathological axis. If an SNN sheet is placed on the patient's contact plane, lower-threshold heterogeneity cores placed at those two ends, with the long E->E axis aligned between them, should reproduce the key SEEG interictal template timing readout.

This does **not** claim that interictal HFO drives seizures. It is a scaffold/readout bridge: the same patient-specific axis may support interictal template switching and early ictal recruitment.

**Claim tier (fixed at planning time).** This is a **per-subject descriptive existence/illustration** at n=2-3 cases, NOT a cohort claim. The cohort-level swap work can only falsify "orthogonal source/sink"; it cannot positively confirm "same causal axis" (that needs per-event seed clustering + rank-distance gradient + source-SOZ asymmetry — not in scope here). So the wording for any result is "on this subject the swap-derived two-core substrate does / does not reproduce the readout", never "swap defines the pathological axis at cohort level". The figure is case-based (§5), not a significance board.

---

## 2. Input contracts

### Montage-consistency rule (binding — added at review)

**Swap, field, and geometry for one subject MUST come from the same montage** (all narrow, or all broad). Mixing montages is a resolution-level error (CLAUDE.md §6.2): the swap-k source/sink *selection* changes between montages, so positioning narrow-montage swap nodes in broad-montage geometry mixes two different answers.

- **Narrow montage** = max_ab field (`*_max_ab_B1000.json`) + narrow swap (`results/interictal_propagation_masked/rank_displacement/...`) + narrow geometry (`propagation_geometry/...`).
- **Broad montage** = broad field (`*_broad_B2000.json`) + **broad swap (`results/interictal_propagation_masked_broad/rank_displacement/...`)** + broad geometry (`propagation_geometry_broad/...`).

Per-subject montage assignment (driven by which montage the field passes on):

| Subject | Montage | Why |
| --- | --- | --- |
| `E958` | **narrow** | field passes only on narrow (hfa_max_ab); no broad artifacts exist; swap+geom self-consistent |
| `E1146` | **narrow** | field passes on narrow broadband (bb_max_ab +0.285); swap+geom narrow |
| `E253` | **broad** | field passes only on broad; **then swap = broad strict k=10 (L-vs-R hippocampus), NOT narrow candidate k=2** |
| `E635` | **broad** | field passes only on broad; **then swap = broad strict k=9 multi-shaft** |

### Field-alignment input

Use current Topic 5 A-line field contract, not stale template-A-only:

- Narrow geometry / maxAB:
  - `results/topic5_ictal_recruitment/axis_alignment/axis_alignment_broadband_max_ab_B1000.json`
  - `results/topic5_ictal_recruitment/axis_alignment/axis_alignment_hfa_max_ab_B1000.json`
- Broad lagPat geometry:
  - `results/topic5_ictal_recruitment/axis_alignment/axis_alignment_broadband_broad_B2000.json`
  - `results/topic5_ictal_recruitment/axis_alignment/axis_alignment_hfa_broad_B2000.json`

Screening display statistic:

`field_margin = real_median_abs_corr - channel_null_p95`

`field_margin > 0` means the subject beats its own channel-null 95% threshold for that candidate. For case selection this is acceptable. If this becomes a formal OR/max result, the null must repeat the same OR/max selection.

### Swap-k input

Use masked rank-displacement per-subject JSON, **from the montage assigned above**:

- narrow-montage subjects: `results/interictal_propagation_masked/rank_displacement/per_subject/<dataset>_<subject>.json`
- broad-montage subjects: `results/interictal_propagation_masked_broad/rank_displacement/per_subject/<dataset>_<subject>.json`

Required fields:

- `pairs[0].swap_sweep.swap_class`
- `pairs[0].swap_sweep.decision_k`
- `pairs[0].swap_sweep.T_obs`
- `pairs[0].swap_sweep.p_fw`
- `pairs[0].rank_a_dense_full`
- `pairs[0].channel_names`

Source-side = lowest `decision_k` ranks in `rank_a_dense_full`.  
Sink-side = highest `decision_k` ranks in `rank_a_dense_full`.  
The combined endpoint set is useful for audit, but Stage 3 placement must keep source-side and sink-side as two separate cores.

---

## 3. Selected subjects

### Primary case set

| Subject | Why selected | Field evidence | Swap evidence | T_a source-side core | T_a sink-side core | Main caveat |
| --- | --- | --- | --- | --- | --- | --- |
| `epilepsiae_958` | Best first case: strict swap, small cores, strong reverse rank geometry | `hfa_max_ab`: margin `+0.102` (`|r|=0.733`, channel-null95 `0.631`), `n_ch=16`, `n_sz=12` | `strict`, `T_obs=0.750`, `p_fw=0.005`, `decision_k=3`, Spearman `rho=-0.806`, F_norm `0.922` | `GD8`, `GF6`, `GE7` | `GG7`, `OPL5`, `GH7` | Field pass is HFA sensitivity, not broadband primary |
| `epilepsiae_253` | Strong field margin and very small two-end cores | `hfa_broad`: margin `+0.263` (`|r|=0.827`, channel-null95 `0.564`), `n_ch=20`, `n_sz=6` | `candidate`, `T_obs=0.667`, `p_fw=0.059`, `decision_k=2`, Spearman `rho=-0.762`, F_norm `0.875` | `HRC2`, `HRA3` | `HRA2`, `HLC1` | Swap is candidate, not strict; seizure count is modest |
| `epilepsiae_1146` | Broadband/maxAB field pass plus strict swap; good stress-test case | `bb_max_ab`: margin `+0.285` (`|r|=0.822`, channel-null95 `0.537`), `n_ch=15`, `n_sz=25` | `strict`, `T_obs=0.653`, `p_fw=0.025`, `decision_k=7`, Spearman `rho=-0.464`, F_norm `0.857` | `SCL9`, `ICL11`, `ICL9`, `SCL8`, `SCL7`, `ICL10`, `ICL8` | `ICL7`, `ICL6`, `ICL5`, `ICL4`, `ICL3`, `ICL1`, `ICL2` | `decision_k=7/15`; this is a broad endpoint strip, not two tiny cores |

### Montage-consistent re-characterization (review — supersedes E253/E635 rows above)

The v1 table above pulled E253/E635 swap from the narrow tree but their field from broad. On the montage where field actually passes (broad), the swap is different:

| Subject | Montage | Verified swap (consistent montage) | Geometry reality | Verdict |
| --- | --- | --- | --- | --- |
| `E958` | narrow | strict, `T_obs=0.750`, `p_fw=0.005`, k=3; source `GD8,GF6,GE7` / sink `GG7,GH7,OPL5` | subdural **grid**: source = D/E/F columns, sink = G/H columns → clean 2D gradient, 6/6 swap nodes in geometry | **CLEAN — primary, lead case** (caveat: field is HFA-only, fails broadband-primary −0.029) |
| `E1146` | narrow | strict, `T_obs=0.653`, `p_fw=0.025`, k=7 | sink = `ICL1–7`, source = `ICL8–11 + SCL7–9` → **within-shaft strip; inter-core axis ≈ shaft direction** (cannot dissociate connectivity-axis from shaft) | **stress-test only** — broadband-primary field is strongest of all cases, but axis is shaft-degenerate |
| `E253` | broad | **strict, `T_obs=0.818`, `p_fw=0.001`, k=10** (NOT narrow candidate k=2) | source = 10 RIGHT-hippocampus contacts, sink = 9 LEFT-hippocampus + HRA1 → **bilateral, cross-midline** | **NOT an SNN-substrate case as-is** — a single 2D excitable sheet cannot represent a cross-midline E→E axis; see §3-fork |
| `E635` | broad | **strict, `T_obs=0.718`, `p_fw=0.002`, k=9** (NOT narrow candidate k=5) | source/sink span TB/TL/HL/HR (bilateral, multi-shaft) | broad multi-shaft, also bilateral — same blocker as E253 |

**Cohort fact (whole-cohort scan).** No subject has BOTH a clean primary-band (broadband) field pass AND a clean unilateral two-small-core swap on a consistent montage. Clean small-core swaps live on the narrow montage (field tends to fail there); field passes require the broad montage (swap becomes broad/bilateral). E958 is the sole clean case because it is a dense 2D subdural grid.

**§3-fork — RESOLVED 2026-06-26 (user): Option B.** Drop E253 as an SNN case. Run **E958 + E1146** as two contrasting regimes:
- **E958** = clean two-separated-core case (subdural grid, source D/E/F columns vs sink G/H corner).
- **E1146** = **full-propagation sampling contrast** (user framing): the ICL shaft genuinely samples the *complete* lesion-activity propagation end-to-end, so the within-shaft strip is not a defect but a legitimate contrast geometry — electrodes that catch the whole pathway. The axis-rotation null (§3D) is still mandatory to confirm the readout follows the modeled axis rather than the shaft.
- Honest framing: "clean small-core + broadband-primary field" do not co-occur in this cohort; E958 carries the clean-two-core regime (field HFA-only), E1146 carries the broadband-field + full-pathway regime.

~~Option A — keep E253 on broad montage as hemispheric axis~~ (rejected: 58 mm cross-midline, no single 2D sheet).

### Backup / sensitivity case

| Subject | Why backup | Field evidence | Swap evidence | Caveat |
| --- | --- | --- | --- | --- |
| `epilepsiae_635` | Good field margin, more seizures than E253, moderate core size | `hfa_broad`: margin `+0.194` (`|r|=0.734`, channel-null95 `0.540`), `n_ch=17`, `n_sz=17` | `candidate`, `T_obs=0.667`, `p_fw=0.130`, `decision_k=5`, Spearman `rho=-0.491`, F_norm `0.840` | Mostly same-shaft / shaft-dominated; swap weaker than E253 |

### Negative controls

These are useful controls, not positive modeling cases:

- `epilepsiae_139`: strict swap (`T_obs=1.000`, `p_fw=0.011`) but field margin is negative (`-0.113`). **Caveat: narrow swap is single-shaft (`HL6/7/8` source vs `HL2/3/4` sink) — shaft-degenerate like E1146; and field failure may be low-power (only n_sz=4 seizures), not true field absence.**
- `epilepsiae_620`: candidate swap (`T_obs=0.600`, `p_fw=0.115`) but field margin is negative (`-0.089`, n_sz=6). Cleaner negative control than E139 (more seizures).

If the subject-specific SNN bridge also works in these negative controls, the bridge is probably too flexible.

---

## 4. Stage 3 plan: interictal field-guided SNN placement

### Stage 3A. Geometry preflight

For each selected subject:

1. Load contact plane from the field source that supported selection.
   - `E958` / `E1146`: narrow geometry, `results/spatial_modulation/propagation_geometry/observation_readout/real_subjects/*_t_{a,b}.json`.
   - `E253`: broad geometry, `results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects/*_t_{a,b}.json`.
2. Load swap-k source/sink channels from masked rank-displacement.
3. Verify channel name overlap between field plane and rank-displacement channels.
4. Compute two core centroids in the contact plane:
   - source core centroid = mean coordinates of source-side swap-k nodes.
   - sink core centroid = mean coordinates of sink-side swap-k nodes.
5. Reject or down-tier a case if:
   - either core has fewer than 2 mapped contacts,
   - source and sink centroids are nearly identical,
   - all contacts collapse onto one degenerate line with no off-axis readout,
   - model readout electrodes cannot sample both cores and the inter-core axis.

Output should be a per-subject geometry audit table before any simulation.

### Stage 3B. Subject-specific SNN substrate

For each surviving subject:

1. Place a 2D SNN sheet on the patient's field plane.
2. Set two lower-threshold (`V_th` lower / higher excitability) heterogeneity cores at the source-side and sink-side centroids.
3. Set the long E->E connectivity axis along the line connecting the two core centroids.
4. Place virtual SEEG contacts at the patient contact coordinates, not at hand-picked ideal positions.
5. Keep global model parameters fixed across subjects where possible; only geometry and electrode locations are subject-specific.

**Pin the validated operating point (added at review).** Do NOT re-explore the regime per subject. Reuse the blessed spontaneous operating point that produced clean spontaneous bidirectional events in `core_model_s3_brakeoff`: core threshold mean ≈ **17.5 mV**, **wide** dispersion (std ≈ 1.5), density 100, prefix_peak detection calibration, brake OFF. Only the following are subject-specific (geometry-derived, not free-tuned): sheet size `L` (scaled to the contact-plane extent), focus separation `sep_frac` (from the two core centroids), EE axis angle `theta` (from the inter-core vector), aspect ratio `AR` (from the contact-plane aspect). Engine params stay fixed; re-bless `engine_versions.json` if the engine is touched.

**Code gap (executability).** Two pieces are NEW/PARTIAL: (a) `from_real_geometry()` is a `NotImplementedError` stub — implement it to consume the precomputed `x_norm/y_norm` (no 3D→2D registration needed; 2D projection already exists in the geometry JSON); (b) the runner's `montage()` hard-codes ∥/⊥ ideal bars — bypass it and feed patient coords via `LFPRecorder(sites=<patient_xy>)`. Reuse `propagation_skeleton_geometry.compute_axis_frame / build_endpoint_cores / core_radii / perp_spread` and `sef_hfo_observation.endpoint_centroid_axis`.

This is the main improvement over the previous generic Stage 3: the two foci are not arbitrary endpoints, but patient-derived swap-k endpoint regions.

### Stage 3C. Minimal simulation gates

Run in this order and stop at the first hard failure:

1. **One-core positive controls**
   - Activate source-side core only.
   - Activate sink-side core only.
   - Expected: the two runs produce opposite rank/readout directions on the virtual SEEG contacts.
2. **Bidirectional template record (primary route = separate-then-pool)**
   - **Primary, validated route:** run the source-core-only and sink-core-only activations separately, then *pool* the two event sets into one synthetic subject (the `pooled_bidir` method that actually produced stable_k=2 / strict swap in prior work). This is the route that worked.
   - **Optional, harder test:** both cores present in one network, spontaneous alternation. Expect the documented `twoend_equal` failure mode — one-core-dominance, collisions, imbalanced fwd/rev (prior brakeoff got 9 fwd / 6 rev, repro only moderate). Do not treat a one-network failure as a refutation of the substrate; it is a known difficulty.
   - Metrics: event count, local/global/collision classes, source label, readable axis fraction, source-core vs other-core ignition.
3. **Template readout comparison**
   - Cluster model events with the same masked feature discipline used in real data.
   - Compare model template rank fields to the subject's real T_a/T_b rank fields.
   - **Not a success metric by itself:** "model produces two readable templates whose source/sink ordering matches the swap sides" is ~geometry-forced on a single EE axis (prior `core_model_s3_brakeoff` NULL: a single connection axis makes k=2 half-forced — that is instrument alignment, not mechanism reproduction). Report it descriptively, but the bar for success is §3D (beat the nulls).
4. **SEEG key timing check**
   - Virtual SEEG trace/rank readout should reproduce the coarse ordering of the real interictal template pair.
   - Do not require point-by-point replay.

Stop rule: if one-core controls fail, do not interpret two-core spontaneous output. That would mean the observation/readout geometry is not capable of seeing the modeled axis.

### Stage 3D. Required controls — THIS is the success criterion

Because "produces two ordered templates" is half-forced by single-axis geometry (§3C.3), the only meaningful success statement is **the real swap-core placement beats all four nulls below on readout match**. State the success/failure of a case in terms of these margins, not in terms of "did it produce templates".

Each positive case must include:

- **Core-location null:** move the two cores to non-swap contacts with matched distance/shaft count.
- **Axis-rotation null:** keep cores but rotate or isotropize the long E->E axis.
- **No-heterogeneity null:** same E->E axis, no low-`V_th` cores.
- **Single-core null:** only one low-`V_th` core, to show that two-template behavior is not forced by the readout alone.

Success means the real swap-core placement beats these controls on readout match, not merely that it produces events.

---

## 5. Figure plan

Use the visual grammar of:

`results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/figures/core_model_s3_brakeoff.png`

But make it patient-specific.

Per subject, one figure:

1. **Panel A: real patient field plane**
   - contacts in `(x_norm, y_norm)` or mm plane,
   - color = interictal `typical_rank`,
   - source-side swap nodes outlined in one color,
   - sink-side swap nodes outlined in another,
   - early ictal activation field shown as contour or side-by-side mini-panel.
2. **Panel B: SNN substrate**
   - two low-`V_th` cores overlaid on sheet,
   - long E->E axis drawn between core centroids,
   - virtual SEEG contacts placed at patient coordinates.
3. **Panel C: readout traces**
   - forward-like and reverse-like spontaneous examples,
   - contacts ordered by real field-axis coordinate,
   - event onset markers and model source label.
4. **Panel D: template/rank comparison**
   - real T_a/T_b rank heatmap vs model event-cluster rank heatmap,
   - report template Spearman / mirror-invariant field alignment,
   - show controls as compact margin bars.

The figure should be descriptive and case-based. It should not look like a cohort significance board.

---

## 6. Stage 5 plan: ictal timing extension

Stage 5 should start only after at least one Stage 3 subject passes the one-core controls and produces readable interictal template-like events.

Question:

> Under a seizure-like state change on the same patient-specific substrate, does recruitment over virtual SEEG contacts follow the early ictal activation field better than geometry/null controls?

Proposed state change:

- raise global drive and/or lower inhibition to model recruitment susceptibility,
- keep the same two swap-derived low-`V_th` cores and same E->E long axis,
- do not add seizure-specific hand-tuned seeds.

Primary Stage 5 readout:

- virtual contact activation order / AUC over early window,
- compare to real ictal broadband activation field (`0-10s`) using the same mirror-invariant field statistic,
- report whether the same substrate that explains interictal template timing also predicts early ictal recruitment order.

Controls:

- same Stage 3 core-location null,
- axis-rotation null,
- no-core null,
- state-change-only null.

Allowed claim if successful:

> A patient-specific swap-axis substrate can reproduce both interictal template readout and early ictal recruitment readout at the coarse field level.

Forbidden claim:

> Interictal HFO causes the seizure, or the seizure replays the exact interictal route.

---

## 7. Minimal next action (loop-safe ordered task list)

Do **not** run a cohort or long simulation first. STOP markers mean a human decision is required before proceeding — a loop must not auto-resolve them.

1. **[safe, simulation-free] Subject-geometry audit (Stage 3A).** Build a montage-consistent geometry builder that, per subject (using the §2 montage assignment), loads `t_a/t_b`, intersects swap source/sink with the geometry channels, computes the two core centroids, and emits the §3A audit table: per-core mapped-contact count, inter-core centroid distance (mm), core radii (`core_radii`), perpendicular spread (`perp_spread`), cross-region/cross-midline flag, and the §3A reject/down-tier verdict. Run for E958, E1146, E253(broad), E635(broad). *(In progress this review turn — outputs the numbers that confirm §3 verdicts.)*
2. **[safe] Static patient field maps** with swap-k source/sink overlays (color = `typical_rank`, source/sink outlined, SOZ ring overlay per figure_style_guide). Eyeball before any sim.
3. **§3-fork RESOLVED (2026-06-26, user): Option B = E958 + E1146.** E1146 = full-propagation-sampling contrast.
4. **Operating point (2026-06-26 derivation, flag for user review):** rescale each patient contact plane ISOTROPICALLY into the blessed `L=20` sheet (preserve relative geometry: inter-core ratio, electrode layout, axis), keep the blessed spontaneous core params (core_mean 17.5, core_std 1.5 wide, core_r 1.5, density 100, drive 0.6, prefix_peak detection, brake OFF). Rationale: the model is a generic excitable sheet, not a literal cortex reconstruction; what transfers is the *relative* core/electrode topology, and the blessed dynamics were tuned at L=20 (L-sensitive per M1). `theta` = scaled inter-core vector angle; `center` = midpoint of the two scaled core centroids; cores placed directly at the scaled source/sink centroids.
5. **[DONE 2026-06-26]** Code gaps (§3B): `from_real_geometry()` implemented (2D-precomputed path; 3D still raises) in `src/sef_hfo_observation.py`; new `src/sef_hfo_subject_placement.py` (load + isotropic sheet registration, channel-overlap loud-fail, frame-consistent); `LFPRecorder(sites=)` + read-out wired in new `scripts/run_sef_hfo_subject_snn.py` (reuses cm_spontaneous glue + engine, **no engine edit → no re-bless**). TDD: `tests/test_sef_hfo_subject_placement.py` 9/9 green incl. E958 real-data smoke.
6. **[EXECUTED + CORRECTED 2026-06-26]** Subject runs via `run_sef_hfo_subject_snn.py`. **Core-placement correction (user review):** v1 used the rank-displacement swap `decision_k=7` source/sink *centroids* — those are broad 7-channel strips pulled toward the axis MIDDLE (E1146: ~9 mm apart at plane-fit), so v1 then ARTIFICIALLY core-anchored to 12 mm and used m17.0 (drift from stage3). **Corrected = `--placement template_source`:** the two cores = the **earliest-3 electrodes of EACH template** by field `typical_rank` (core A = t_a source = `SCL9/ICL9/ICL11` one end; core B = t_b source = `ICL1/ICL2/ICL3` other end — the two true ends of the swap axis, = the user's "two template sources"). Plane-fit of the REAL geometry then gives ~13 mm separation NATURALLY (≈ blessed sep 0.7) with ALL contacts retained — no anchoring hack. Params back to stage3: m17.5, std1.0, core_r1.5. `template_source_foci()` in `src/sef_hfo_subject_placement.py`.
   - **k_dir=2 is a load-bearing sparse-electrode relaxation (sensitivity required):** patient electrodes are sparser than the blessed ∥/⊥ bars, so events light only ~5-6 contacts; the standard k_dir=3 estimator (needs ≥6 participating, 3 early + 3 late) returns 0 direction for the E1146 source leg. Document "patient-sparse readout uses k_dir=2" everywhere; report the k_dir=3 fallback counts.
   - **Honesty tier:** separate-core driving (source-only / sink-only, the `pooled_bidir` route) = INSTRUMENT ALIGNMENT (can the patient electrode geometry read the two opposite templates). Spontaneous twoend (both cores, one network) tests the stronger mechanism claim; v1 swap-placement twoend showed robust one-core dominance (6/6 seeds all-reverse, 0 forward). Never write "model spontaneously reproduces the swap" from the separate-then-pool result.
   - **E958 (subdural grid) NEGATIVE, honestly framed:** events stay local (n_part 3-4) even at FULL plane-fit coverage (16/16 valid) → no readable direction; the ~10 mm grid spacing under-samples the model's local spontaneous events. Not purely a core-anchored coverage artifact (that is an additional, separate effect). Figure: `negative_epilepsiae_958.png`.
7. If E958 one-core controls pass: bidirectional record via separate-then-pool (§3C.2 primary route), then the §3D nulls, then the first patient-specific `core_model_s3`-style figure.
8. Only after E958 passes §3D: extend to E1146 (shaft-strip stress) and the resolved §3-fork case; Stage 5 only after a Stage 3 case clears §3D.


---

## Figure pipeline (end-to-end, executed 2026-06-26)

Three stages; each consumes the previous artifact, no re-sim downstream.

**Stage 1 — simulation + virtual-SEEG readout.** `scripts/run_sef_hfo_subject_snn.py`
- Placement (`--placement template_source`, helper `src/sef_hfo_subject_placement.py`):
  two low-V_th cores = the **earliest-`k` electrodes of each interictal template** (field
  `typical_rank`; `template_source_foci`), i.e. the two template sources at the two axis ends;
  **real-geometry plane-fit** (`register_to_sheet`, no core-anchoring) → cores ~13 mm apart
  naturally, all contacts kept. Substrate = blessed stage3 (`build_connectivity_rot` theta=axis
  angle, `sample_core_field` core_mean 17.5 / std 1.0 / r 1.5, density 100, drive 0.6), engine
  reused (no edit → no re-bless). Virtual electrodes = patient contacts via `LFPRecorder(sites=)`.
- Modes (`--lesion`): `source` (ignite core A → forward), `sink` (core B → reverse),
  `twoend_equal` (both cores → spontaneous, seed-dependent bidirectional).
- Read-out `--k-dir 2` (load-bearing sparse-electrode relaxation; k_dir=3 fallback reported).
- Outputs per tag: `results/topic4_sef_hfo/field_swap_subject_snn/{readout_<tag>.json, figdata_<tag>.npz}`
  (figdata carries vth/foci/posE/contacts/lfp_trace/events/rep_fwd/rep_rev).

**Stage 2 — Fig4A (mechanism + propagation + readout).** `scripts/paper_figures/plot_fig_subject_snn.py`
- 1-row-4-col (style_guide §T4): `mechanism | tempA source | tempB source | electrode readout`.
- mechanism + readout from the `twoend` run (spontaneous, both cores, both directions shaded);
  tempA/tempB event panels from the `source`/`sink` runs.
- Output `results/paper-ready-figure/fig_subject_snn_<subject>/figures/fig_subject_snn_<subject>.{png,pdf}` + metadata.

**Stage 3 — Fig4B (KMeans=2 readout verification).** `scripts/paper_figures/plot_fig_subject_snn_kmeans2.py`
- Consumes the SAME twoend readout (clean directional events, `sign != None & n_part >= 2*k_dir`);
  unsupervised `compute_adaptive_cluster_stereotypy(k=2, use_masked_features)`.
- Drawn with the **mature canonical plotters** (`_plot_rank_heatmap` / `_plot_rank_histogram` /
  `_plot_cluster_boundaries` / `_plot_cluster_rank_fig4` from `scripts/plot_interictal_propagation.py`,
  same as the Topic-1a per_subject figure). 3 blocks, heatmap leftmost, y-axis aligned (cluster-rank
  re-inverted), rank axis cropped to the actual max rank, no suptitle (stats → metadata).
- Output `..._kmeans2.{png,pdf}` + metadata.

**Per-subject tags (E1146):** twoend `epilepsiae_1146_twoend_equal_tsrc_s3`, source
`epilepsiae_1146_source_tsrc_s1`, sink `epilepsiae_1146_sink_tsrc_s1`.

**Reproduce all three:**
```
python scripts/run_sef_hfo_subject_snn.py --subject epilepsiae_1146 --lesion twoend_equal --seed 3 --tag epilepsiae_1146_twoend_equal_tsrc_s3
python scripts/run_sef_hfo_subject_snn.py --subject epilepsiae_1146 --lesion source --seed 1 --tag epilepsiae_1146_source_tsrc_s1
python scripts/run_sef_hfo_subject_snn.py --subject epilepsiae_1146 --lesion sink   --seed 1 --tag epilepsiae_1146_sink_tsrc_s1
python scripts/paper_figures/plot_fig_subject_snn.py
python scripts/paper_figures/plot_fig_subject_snn_kmeans2.py
```
(Stage-1 runs default to template_source / k_dir2 / m17.5; ~1 core each, single-threaded.)
