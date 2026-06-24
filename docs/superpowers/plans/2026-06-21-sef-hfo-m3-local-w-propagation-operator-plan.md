# M3B model↔SEEG field-bridge plan

> Status: re-centered 2026-06-24. Round 1 now attacks the model↔SEEG bridge directly.
> The previous version made round 1 = model-side W estimation (estimate W three ways +
> resolution sweep). An asset audit (below) showed that is the wrong first move; the bridge
> machinery, the data, and the geometry nulls already exist, and the model's structural axis
> is known exactly. So round 1 is the bridge, with the "same field, two gains" readout folded
> into the same machinery, and the model-side W estimator demoted.
> Scope: M3B only. M3B asks whether the model's propagation field (W / W_eff) is a quantitative
> readout of field structure AND whether it bridges to the real SEEG interictal/ictal propagation
> axes. It does NOT test the slow-variable seizure mechanism (that is M3A).

## 0. Why this revision (the re-centering)

Round 1 of the previous plan defaulted to model-side W instrument work (audit W objects →
estimate W three ways → resolution sweep). An asset audit (2026-06-24) flips that for three reasons:

1. **The model's structural axis is exact, not something to estimate.** E→E connectivity is
   already anisotropic — `src/snn_engine/params.py`: `rho_EE=0.6`, elliptical-exponential kernel,
   long axis along the (1,1) diagonal = **45°**. So `W_struct`'s axis = that ellipse's major axis,
   read straight off connectivity. No estimation needed.
2. **The recurring "W ≈ distance / not resolved" was a resolution artifact, not science.** B1b
   already diagnosed it: bin = 4 mm, event r95 ≈ 5 mm → the recruitment shape is dominated by the
   4 orthogonal neighbours, so the readout physically cannot resolve the 45° diagonal that is there
   by construction. Polishing a finer W estimator chases the artifact, not the field.
3. **The bridge metric, the data, and the geometry-stripping nulls already exist** (see §3). The
   bridge is almost pure reuse; building a cleaner model-side W is not the bottleneck.

Therefore: **round 1 = the bridge** (does the model's scaffold land inside the real SEEG scaffold
distribution, after stripping geometry?), with the **"same field, two gains" readout folded into the
same comparison machinery**. Model-side W estimation and resolution sweeps are demoted (§6).

## 1. Hard boundary (carried over from the M3 split — do not relax)

- M3A answers the slow-variable mechanism; M3B answers the W readout + data bridge. They are separate.
- **No `h(W)`-coupled threshold permissivity as a mechanism.** The old `V_th_eff = V_th0 − δ·μ·h(W)`
  path is a historical control / negative result only.
- `W_eff(s) = D(s) · W_struct`: slow state changes effective **gain** (the diagonal `D`), not the
  structural scaffold `W_struct`. M3B may consume M3A's `s_slow`; it does not invent it.
- Source-column / target-row convention: `W[p, q] = q → p`.
- Event window is **event-aligned**; fixed windows are sensitivity only.
- `R4a` (W-aligned sustained recruitment) is the only seizure-like bridge candidate; `R4b` (tonic
  full-field runaway) is never written as seizure-like.
- Any rank/lag input that touches real SEEG goes through the **masked** lagPat/rank pipeline, never
  the phantom-contaminated raw ranks.
- Do not call `W_event` a proven directional propagation operator.

## 2. Core claim and its two legs

**Core claim**: the interictal propagation field and the ictal propagation field are the **same
scaffold at two effective gains**; the model's scaffold (the 45° anisotropy) reproduces the real
SEEG scaffold.

- **Leg-bridge (PRIMARY, round 1)**: the model's propagation field, observed through a virtual-SEEG
  pipeline, lands **inside** the real SEEG interictal cohort's own field-similarity distribution, and
  survives the same geometry/anatomy nulls the data side used. It is also compared to the real
  ictal-early axis.
- **Leg-readout ("same field, two gains", folded into round 1, secondary)**: across a range of
  effective recruitment gain, the field **shape/axis is invariant** (shape-stable) while the gain
  rises (gain-variant) — tested with the **same** comparison metric.
- **Deferred (Lane B2, after M3A)**: `W_eff(s)` with a real `s_slow`, event-accumulation, and the
  slow-state phase-transition figure. Gated; untouched by this revision.

## 3. Assets to REUSE (do not reinvent)

Round 1 is mostly wiring existing pieces together. Confirm each path/signature before use.

| Bridge part | Existing asset |
|---|---|
| Model↔cohort metric | **`compare_model_to_cohort(model_record, real_records, X, Y, sigma_xy, s_thresh, overlap_min)`** — `src/propagation_contact_plane_readout.py:370`. Built for exactly "where does one model record fall in the real cohort distribution": (a) scalar placement (model percentile + robust z for `axis_length_mm`, `transverse_width_mm`, `early_zone_spread`, `late_zone_spread`, `early_late_centroid_distance_norm`, `rank_vs_xnorm_spearman`); (b) field placement (model-to-real median mirror-invariant corr vs the real-to-real distribution, subject-first folded). No p-values (posterior-predictive, spec §9). |
| Field similarity (bidirectional) | `corr_pair_mirror_invariant(F1,S1,F2,S2,...)` — `propagation_contact_plane_readout.py:285`. Returns `max(corr_identity, corr_yflip)` — handles the axis being bidirectional. |
| Source→sink axis frame | `compute_axis_frame(coords, source_idx, sink_idx)` — `src/propagation_skeleton_geometry.py:75`. Returns `along_axis`/`off_axis` per channel. |
| Held-out axis validity | `split_half_axis_validation(...)` — `propagation_skeleton_geometry.py:535` (half-A frame predicts half-B rank; bootstrap CI). |
| Geometry/anatomy nulls | Topic 5 A-line **four-tier nulls** — `topic5_axis_alignment.py`, pure functions: `channel` / `within_shaft` / `anchor_matched` / `joint`. The A-line verdict to match: coarse skeleton survives the channel null (FDR q≈0.02); fine alignment survives the joint null only for 60–100 Hz HFA. |
| Real interictal axis (data) | `results/spatial_modulation/propagation_geometry/components/path_axis/per_subject/<subject>.json` (per channel: `along_axis_mm`, `off_axis_mm`, `stereotypy_excess`, `role`, `shaft`, `name`) + `cohort_summary.json`. |
| Real ictal-early axis (data) | `results/topic5_ictal_recruitment/axis_alignment/{timing_rate/, axis_alignment_FINAL.json, axis_alignment_FINAL.md}`. |
| Model axis ground truth | `src/snn_engine/params.py` `rho_EE=0.6` → 45°; readout via `principal_axis` in `src/sef_hfo_b1_validation.py`. |
| Model → virtual electrodes | the existing virtual-SEEG observation layer (the "cm 四对照 at L=20" work). **Locate the module and confirm its path before building** — the audit did not surface the exact file. |

**§6.1 question-match check (the null is the contract).** `compare_model_to_cohort`'s baseline is the
real cohort's own internal field-similarity distribution; my bridge question is "does the model's
field sit inside what real subjects achieve with each other?" — these match, so the helper is the
right shape. But `compare_model_to_cohort` alone does NOT strip geometry (its baseline is cohort-
internal). The geometry confound is handled separately by the four-tier nulls. Round 1 uses **both**:
`compare_model_to_cohort` for placement + the four-tier null pass for "beats geometry".

## 4. The model's structural axis (exact, free) — and the only real instrument question

- `W_struct` axis is analytic: the `rho_EE=0.6` ellipse major axis at 45°. We do not estimate it.
- The **only** instrument question that matters for the bridge is the **observation pipeline**: the
  virtual-electrode montage must be able to *resolve* that axis. From B1b, a 4–5 mm isotropic readout
  cannot. So the montage needs ≥2 non-parallel shafts with spacing fine enough (relative to the event
  scale) to express the diagonal, matched to the real SEEG montage geometry as closely as the source
  space allows.
- **Internal sanity gate**: the virtual readout must recover an axis within ~25° of the known 45°
  before any bridge claim. If it cannot, the bridge is untestable for resolution reasons (a documented
  caveat), not a science negative.

## 5. Round-1 tasks (all reuse + thin adapters; PILOT-FIRST)

### Task 0: Freeze the round-1 scope
- [ ] `STATUS.md`: round 1 = bridge (primary) + same-field-two-gains readout (folded), NOT model-side
  W estimation; resolution sweep dropped; W_eff(s) deferred to M3A.
- [ ] Record the two findings from §0 (exact 45° axis; binned-W is a resolution artifact).

### Task 1: Model → virtual-SEEG record adapter (the load-bearing reuse)
The bridge is only fair if the model goes through an observation pipeline **analogous to how the SEEG
records were built**. The real `path_axis` records come from: cluster template rank → source/sink
cores → `compute_axis_frame` → per-channel `along_axis_mm`/`off_axis_mm`.
- [ ] Run the model (one accepted Stage-3/M3 substrate), collect spontaneous events.
- [ ] Project onto a virtual-electrode montage (§4) that resolves the 45° axis.
- [ ] Build a per-"subject" model record with the **same schema** the real records + `compare_model_to_cohort`
  consume: per-electrode coords + recruitment rank, plus the scalar metrics
  (`axis_length_mm`, `transverse_width_mm`, `early/late_zone_spread`,
  `early_late_centroid_distance_norm`, `rank_vs_xnorm_spearman`) via `compute_axis_frame` on the
  model's source/sink.
- [ ] **Sanity gate**: recovered axis within ~25° of 45° (§4). Stop and recap if it fails.

### Task 2: The bridge (PRIMARY)
- [ ] `compare_model_to_cohort(model_record, real interictal cohort)` → scalar-placement percentiles
  + field-placement percentile in the real-to-real distribution.
- [ ] Run the model record against the **four-tier nulls** (channel / within_shaft / anchor_matched /
  joint) exactly as the A-line did — the bridge claim requires beating at least the channel
  (geometry-stripped) null, matching the data-side bar.
- [ ] Also compare the model axis to the **ictal-early** axis (`axis_alignment` artifacts).
- [ ] Report model-vs-cohort placement BEFORE any aggregate verdict; do not collapse legs (§6.3 pronoun
  discipline) — state "field-placement percentile X AND beats null Y", not a bare "bridge PASS".

### Task 3: "Same field, two gains" readout (folded into the same machinery, secondary)
- [ ] Produce model records at a **range of effective recruitment gain** and test whether the field
  shape/axis is invariant (each gain level lands in the real cohort AND consecutive gains are
  mirror-invariant-correlated to each other via `corr_pair_mirror_invariant`) while the gain rises.
- [ ] **Gain source (pilot decides)**: primary = endogenous event-recruitment-size bins. **Caveat**:
  the static-μ pilot found spontaneous event size fairly flat (~12 bins), so the endogenous dynamic
  range may be too small. Fallbacks, explicitly labeled **instrument/probe, NOT mechanism**:
  kick-strength sweep (read the *propagated* shape, not the radial seed), or μ as a controlled gain
  knob. Pick in the pilot based on which gives adequate gain range; log the choice.
- [ ] Negative pattern to report honestly: if shape changes with gain, the `D(s)·W_struct` form is
  wrong (the network rewires its route at high gain) — that falsifies the "same scaffold" claim.

### Task 4: Figures + verdict
- [ ] Figures (each answers one question, §7 figure discipline): (1) the model's virtual-SEEG record
  + recovered axis vs the known 45°; (2) model placement inside the real interictal cohort (scalar +
  field); (3) shape stability across the gain range. Chinese `figures/README.md`.
- [ ] Verdict categories:
  - **B-PASS field bridge**: model field lands in the real cohort AND beats the geometry null AND
    aligns with the ictal-early axis.
  - **B-PASS placement-only**: model lands in the cohort but does not beat the geometry null (shared
    geometry, not shared scaffold) — bounded.
  - **B-BOUNDED NEGATIVE**: readout resolves the model axis but it does not bridge the data.
  - **B-UNTESTABLE (resolution)**: virtual readout cannot recover the 45° axis (Task 1 sanity fails).

## 6. Demoted / dropped (was the old round 1)

- **Estimate W three ways (old Task 2)** — dropped as round 1. `W_struct` is exact (§4); `W_resp/W_step`
  small-kick response was already negative; `W_event` was the resolution-artifact path. Keep the
  three-object *separation* as a definitional note only, not a round-1 deliverable.
- **Resolution sweep `n_bins ∈ {5,9,11}` (old Task 3)** — **dropped**. The sweep's information value is
  low (it only weakens/strengthens a caveat, never resolves it). Per user: if any resolution check is
  ever warranted, run the **maximum** resolution directly, once — never a graded sweep.
- **`W_event` estimator / event-conditioned W** — optional appendix only, not round 1.

## 7. Lane B2 (after M3A provides `s_slow`) — gated, unchanged

- Bin events by slow-state quantile; estimate `W_eff(s)` empirically or as `D(s)·W_step`.
- Compute `Lambda_eff(s)` or finite-event recruitment gain `R_event(s)`; test shape stability
  `corr(vec(W_shape(s)), vec(W_shape(0)))`.
- Expected positive: `s_slow ↑` → `Lambda_eff/R_event ↑`, `W_shape` stable, R-class R2/R3 → R4a.
- Event-accumulation readout (`load_p(t) = [W_shape · a(t)]_p`) and the slow-state phase-transition
  figure live here. Do not start without a valid M3A `s_slow`.

## 8. Stop rules

Stop and recap before any expensive run if:
- Task 1 sanity fails (virtual readout cannot recover the 45° axis) — bridge is untestable, recap.
- the model lands in the cohort but cannot beat the geometry null — write placement-only, do not
  escalate to "field bridge".
- the Topic 5 axis contract turns out stale on re-read (read `docs/topic0_methodology_audits.md`,
  `docs/topic5_seizure_subtyping.md`, `docs/paper_overview.md` first).
- Lane B2 is attempted before M3A delivers `s_slow`.

## 9. Plain-language claim template

Allowed if supported:

> The structural field scaffold is fixed by the model's E→E anisotropy. Observed through a virtual
> SEEG montage, the model's propagation field falls inside the distribution that real patients'
> interictal fields span, and it stays in that distribution — same axis — as the effective recruitment
> gain rises. So interictal and ictal look like one scaffold read at two gains, not two routes.

Not allowed: "W causes seizure" · "W_event is a proven directional operator" · "h(W)-threshold μ
reproduced the interictal-to-seizure transition" · a bare "bridge PASS" that hides whether it beat
the geometry null.
