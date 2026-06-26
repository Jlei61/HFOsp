# Topic 5 Event-Resolved Interictal axis_bias (Secondary) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or executing-plans. Before each task body, invoke `hfosp-deep-contract-verify` and **re-read the cited spec §** (CLAUDE.md §5). Steps use `- [ ]`.

**Goal:** Build an event-resolved extension of the Topic 5 A-line: per-interictal-event ictal-field alignment, split by A/B cluster label, on the dense broad substrate — measuring within-class dispersion (the std effect) directly. Keep the A-line primary untouched.

**Architecture:** Pure-function module reusing the A-line/geometry primitives (`mask_phantom_ranks`, `compute_axis_frame`, `make_field_record`, `R_smooth_rank`, `corr_pair_mirror_invariant`, the activation shuffles). A pilot-first runner; cohort run is a hard-stop awaiting the human advisor. Stages B/C are stubs.

**Tech Stack:** Python, numpy, scipy.stats, sklearn (labels already on disk), pytest. No new deps.

**Spec:** `docs/superpowers/specs/2026-06-25-topic5-event-resolved-axis-bias-design.md` (v2). Re-read the cited § at each task boundary.

**Note on test counts:** cumulative counts are indicative; `pytest tests/test_topic5_event_resolved_alignment.py -q` fully green is the gate.

## Global Constraints (verbatim from spec §4/§5)

- substrate_primary = **broad** (`results/interictal_propagation_masked_broad/per_subject/<ds>_<subj>.json` labels + `results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects/<ds_sid>_t_{a,b}.json` planes + broad-pool lagPat NPZ); substrate_companion = narrow (M1d only, n_ch≥8).
- min_participating=3 (valid-event gate); MIN_PART_EVENT=5; CHANNEL_HEADROOM=3; OVERLAP_MIN=25; cluster_map_margin=0.30; class_sep_null = A/B label **block-level** shuffle preserving class sizes, N_PERM=1000; RNG_SEED=20260625.
- ictal_reference = subject-mean bb_auc over `eligible_idxs` (read from `{ds_sid}.json` sidecar, NOT npz); labeled "more-averaged estimator ≠ primary's per-seizure-median".
- All dispersion/null **block-aware** (report n_blocks); forbid per-event-count precision language. Real & null use the **same** mirror-invariant + sign-free reduction.
- tier = exploratory; cohort run forbidden until advisor (user) sign-off. Stage B/C stubs raise `NotImplementedError`.

---

## File Structure

- Create `src/topic5_event_resolved_alignment.py` — pure functions (loader+contract checks, cluster↔template map, M field metric, M1d 1D metric, block label-shuffle null, diagnostics, Stage B/C stubs).
- Create `tests/test_topic5_event_resolved_alignment.py` — one test per §C contract + metric behavior.
- Create `scripts/run_topic5_event_resolved_alignment.py` — `--pilot/--subjects/--substrate/--activation/--out`.
- Create `scripts/plot_topic5_event_resolved_alignment.py` — per-subject 3-panel + `figures/README.md`.
- Output root `results/topic5_ictal_recruitment/event_resolved_alignment/` (`per_subject/`, `figures/`, `pilot_summary.json`).

**Reused (do NOT reimplement):** `src.interictal_propagation.{load_subject_propagation_events,_valid_event_indices}`; `src.lagpat_rank_audit.mask_phantom_ranks`; `src.propagation_skeleton_geometry.compute_axis_frame`; `src.topic5_axis_alignment.{make_field_record,channel_shuffle}`; `src.propagation_contact_plane_readout.{R_smooth_rank,corr_pair_mirror_invariant,smooth_field,S_THRESH}`.

---

## Phase 0 — Broad coverage + participation audit (de-risk the substrate before code)

### Task 0: Broad-substrate audit script
Re-read spec §1 (data reality table) + §8.1.

**Files:** Create `scripts/audit_topic5_event_resolved_substrate.py` (one-off audit; no TDD needed — it only reads + prints/writes a CSV).

**Interfaces:** Produces `results/topic5_ictal_recruitment/event_resolved_alignment/substrate_audit.csv` with columns `dataset,subject,substrate,n_channels,n_valid_events,n_blocks,median_n_part,frac_ge5,frac_ge6,frac_ge10,has_broad_labels,has_broad_planes,in_aline_cohort`.

- [ ] Step 1: For each broad-labelled subject (`interictal_propagation_masked_broad/per_subject/*.json`, stable_k==chosen_k==2), load broad-pool lagPat via `load_subject_propagation_events`, reconstruct `valid_ev`, compute participation distribution (median n_part, frac≥{5,6,10}), n_blocks (unique block_ids over valid events), and whether broad planes (`propagation_geometry_broad/.../<ds_sid>_t_a.json`+`_t_b.json`) exist. Cross-flag against A-line cohort (`axis_alignment_broadband_B1000.json::per_subject` status ok).
- [ ] Step 2: Run it; eyeball the CSV. **Gate:** confirm ≥3 pilot subjects have broad labels + broad planes + median n_part ≥ ~8. Record findings inline in the eventual pilot recap.

> NOTE: broad-pool lagPat NPZ location must be resolved here (narrow uses `/mnt/.../all_recs/*_lagPat_withFreqCent.npz`; broad pool was produced into a results tree — confirm the exact dir the broad labels were computed from, e.g. `results/lagpat_broad_epilepsiae*/<subj>/` for epi, `results/lagpat_broad/<subj>/` for yuquan). The loader must be fed the SAME dir that produced the broad labels, or §C1 will (correctly) raise.

---

## Phase 1 — Module (TDD)

### Task 1: `load_event_labels_ranks` + §C1 positional alignment (three hard raises)
Re-read spec §3 intro + §C1/§C5/§C6.

**Files:** Create `src/topic5_event_resolved_alignment.py`; Test `tests/test_topic5_event_resolved_alignment.py`.

**Interfaces:**
- Produces `load_event_labels_ranks(dataset, subject, *, broad=True, masked_dir=None, lagpat_dir=None) -> dict` with keys `masked (|C|,n_valid)`, `bools (|C|,n_valid)`, `labels (n_valid,)`, `valid_ev`, `event_abs_times (n_valid,)`, `block_ids (n_valid,)`, `channel_names`, `n_blocks`. Raises `ValueError` on any §C1 mismatch.

- [ ] Step 1: Write failing tests:
  - `test_c1_channel_names_mismatch_raises`: synthetic loader returns channel_names ≠ JSON → raises.
  - `test_c1_cluster_count_mismatch_raises`: labels partition counts ≠ JSON `clusters[k].n_events` → raises.
  - `test_c1_template_mismatch_raises`: rebuilt per-cluster nanmean template far from JSON `template_rank` → raises.
  - `test_c1_happy_path`: aligned synthetic inputs → returns dict with `masked` already phantom-masked (`mask_phantom_ranks`), `labels` length == valid count.
  (Use a tiny synthetic NPZ + JSON fixture in `tests/`; monkeypatch the loader + json load.)
- [ ] Step 2: Run → fail (function undefined).
- [ ] Step 3: Implement. Default dirs: epi broad lagpat = `results/lagpat_broad_epilepsiae/<subj>`, yuquan broad = `results/lagpat_broad/<subj>`; labels = `results/interictal_propagation_masked_broad/per_subject/<ds>_<subj>.json`. Load broad JSON (`adaptive_cluster.{labels,clusters,stable_k,chosen_k,channel_names}`); assert `stable_k==chosen_k==2`. Load lagPat via `load_subject_propagation_events(lagpat_dir)` → keep **raw `ranks`** (for C1.3) AND `masked=mask_phantom_ranks(ranks,bools)`; `valid_ev=_valid_event_indices(bools,3)`; subset both `[:,valid_ev]`. Then §C1: (1) `channel_names==json_names` elementwise else raise; (2) for k in {0,1}: `int((labels==k).sum())==clusters[k]["n_events"]` else raise; (3) **reproduce producer template**: `tr_k = argsort(argsort(_legacy_hist_mean_rank(ranks_raw[:, valid_ev[labels==k]], bools_raw[:, valid_ev[labels==k]])))`; require `list(tr_k)==clusters[k]["template_rank"]` (exact; fallback rank-corr ≥0.99) else raise. **Use raw ranks, not masked, for this clincher** (verified exact on 1077/1125/922; masked nanmean only reaches 0.83/0.61 and is NOT the producer's aggregation).
- [ ] Step 4: Run → pass.
- [ ] Step 5: Commit `feat(topic5): event-resolved loader with positional label-alignment guards (C1)`.

### Task 2: `map_clusters_to_templates` (§C2 signed + margin + bijection)
Re-read spec §C2.

**Interfaces:** Produces `map_clusters_to_templates(cluster_templates, t_a_rank, t_b_rank, *, margin=0.30) -> {"map": {0:"t_a"|"t_b",1:...}, "diag_minus_offdiag": float, "ambiguous": bool}`. `cluster_templates` = the two per-cluster rebuilt templates from Task 1.

- [ ] Step 1: Failing tests: `test_c2_clean_map` (clear diagonal → map + not ambiguous); `test_c2_near_mirror_ambiguous` (t_a,t_b anti-correlated, weak diagonal → ambiguous True); `test_c2_bijection_enforced` (both clusters argmax same template → ambiguous True).
- [ ] Step 2: Run → fail.
- [ ] Step 3: Implement: build 2×2 **signed** Spearman corr (cluster_i vs t_a/t_b), pick argmax per cluster, require bijection AND `diag_mean−offdiag_mean ≥ margin`; else ambiguous.
- [ ] Step 4–5: pass; commit `feat(topic5): cluster↔template signed/margin/bijection map (C2)`.

### Task 3: `per_event_field_alignment` (M — §3.1, §C3/C4/C8)
Re-read spec §3.1 + §C3/§C4/§C8 + §C6. Invoke hfosp-deep-contract-verify (boundary params: per-class plane, per-event support, pinned sigma, identical reduction).

**Interfaces:**
- Consumes Task-1 dict + Task-2 map + per-class plane records (`plane_a, plane_b`: each has per-channel `x_norm,y_norm,name` from broad geometry) + `sigma_a, sigma_b` (each class's full-channel template sigma) + `ictal_field` (`{F,S}` subject-mean bb_auc field on matched channels).
- Produces `per_event_field_alignment(...) -> {"per_event":[{event_idx,abs_time,block_id,label,n_part,align,status}], "usable_fraction":float, "n_blocks":int}`.

- [ ] Step 1: Failing tests:
  - `test_m_uses_class_own_plane`: a B-labelled event is built on `plane_b` not `plane_a` (assert the channel coords used come from plane_b).
  - `test_m_support_is_event_participation`: per-event field support reflects THIS event's participation (1/NaN), not aggregate support (feed an aggregate-support plane and assert it's ignored).
  - `test_m_sigma_pinned`: sigma passed in is used (not re-derived) — assert call to R_smooth_rank receives `sigma_a`/`sigma_b`.
  - `test_m_overlap_gate`: an event with <OVERLAP_MIN overlap → status `insufficient_overlap`, excluded from usable.
  - `test_m_same_reduction_signfree`: align == |corr_pair_mirror_invariant(...)| (abs of the mirror-invariant result).
- [ ] Step 2: Run → fail.
- [ ] Step 3: Implement per spec §3.1 pseudocode. For each valid event: pick plane by label→template map; matched = channels in plane ∩ ictal ∩ participating; `rec = make_field_record(matched, masked[participating,e])` with support forced to event participation; `F_e,S_e = R_smooth_rank(rec, X,Y, sigma=sigma_class, S_THRESH)`; `r = corr_pair_mirror_invariant(F_e,S_e,F_ictal,S_ictal,overlap_min=OVERLAP_MIN)`; `align=abs(r["corr"])` if not insufficient else status. Track block_id; compute usable_fraction + n_blocks over usable.
- [ ] Step 4–5: pass; commit `feat(topic5): per-event mirror-invariant field alignment M (C3/C4/C8)`.

### Task 4: `per_event_1d_alignment` (M1d — §3.2, replay-adjacent, headroom gate)
Re-read spec §3.2 + §6 (replay wording) + §C8.

**Interfaces:** `per_event_1d_alignment(masked, bools, valid_ev, labels, ictal_by_ch, channel_names, *, min_part=5, headroom=3, n_perm=1000, rng) -> {"per_event":[{event_idx,label,n_part,align1d,null_p}], "n_channels":int, "eligible":bool, "usable_fraction":float}`. NO sign field.

- [ ] Step 1: Failing tests: `test_m1d_headroom_gate` (n_ch=6 → eligible False, no per-event output); `test_m1d_signfree_no_sign` (output has no `sign` key); `test_m1d_per_event_null_within_participating` (null shuffles a_e only within P_e, n_part-matched).
- [ ] Step 2: Run → fail.
- [ ] Step 3: Implement: eligible only if `n_ch ≥ min_part+headroom`; per event with `min_part ≤ n_part ≤ n_ch−headroom` and finite a_e: `align1d=abs(spearman(masked[P_e,e], a[P_e]))`; null = permute `a[P_e]` n_perm times → fraction ≥ observed. No sign stored.
- [ ] Step 4–5: pass; commit `feat(topic5): replay-adjacent 1D companion M1d with channel-headroom gate (C8)`.

### Task 5: `class_separation_block_null` (R2 — §3.3, §C7 block-level)
Re-read spec §3.3 R2 + §C7.

**Interfaces:** `class_separation_block_null(align_by_event, labels, block_ids, *, n_perm=1000, rng) -> {"delta_median_obs":float,"delta_median_null_p":float,"disp_ratio_obs":float,"disp_ratio_null_p":float,"n_blocks":int,"size_matched":{...}}`.

- [ ] Step 1: Failing tests: `test_r2_block_level_shuffle` (labels permuted by whole block, not per event — assert events in same block keep same shuffled label); `test_r2_preserves_class_sizes` (null draws keep n_A,n_B marginals); `test_r2_size_matched_reported` (down-sample larger class to n_min for width comparison).
- [ ] Step 2: Run → fail.
- [ ] Step 3: Implement: observed Δmedian(A,B) and dispersion ratio (IQR_A/IQR_B); null = permute the **block→label** assignment preserving counts, recompute; p = right/two-sided fraction; also report size-matched width.
- [ ] Step 4–5: pass; commit `feat(topic5): block-level A/B label-shuffle separation null (R2/C7)`.

### Task 6: `participation_diagnostics` + Stage B/C stubs (§C10)
**Interfaces:** `participation_diagnostics(bools, labels, block_ids) -> {per class: n_events,n_blocks,median_n_part,frac_ge*}`; `stage_b_window_bias(*a,**k)`/`stage_c_sequential_effects(*a,**k)` raise `NotImplementedError`.

- [ ] Step 1: Failing tests: diagnostics counts; `test_stage_b_stub_raises`, `test_stage_c_stub_raises`.
- [ ] Step 2–4: implement; pass.
- [ ] Step 5: Commit `feat(topic5): participation diagnostics + Stage B/C stubs (C10)`.

---

## Phase 2 — Pilot runner + figures (NO cohort run)

### Task 7: `run_topic5_event_resolved_alignment.py --pilot`
Re-read spec §8 (hard stops). 

**Files:** Create `scripts/run_topic5_event_resolved_alignment.py`.

- [ ] Step 1: Implement runner: for each PILOT subject — load via Task 1; map via Task 2 (skip if ambiguous, record); build per-class planes from broad geometry + per-class sigma; build subject-mean ictal bb_auc field (read `eligible_idxs` from sidecar, §C6); run M (Task 3), M1d on narrow if n_ch≥8 (Task 4), R2 (Task 5), diagnostics (Task 6); write `per_subject/<ds_sid>.json` + append `pilot_summary.json`. **Refuse** to run without `--pilot`/`--subjects` (no implicit cohort). 
- [ ] Step 2: Run `--pilot`. Verify per_subject JSONs + pilot_summary written; eyeball usable fractions + n_blocks.
- [ ] Step 3: Commit `feat(topic5): event-resolved pilot runner (pilot-first, no cohort)`.

### Task 8: `plot_topic5_event_resolved_alignment.py` + figures/README.md
Re-read spec §7 (panel discipline) + figure memories (paper-grade, self-contained).

- [ ] Step 1: Implement 3-panel per-subject figure (a: M A/B align hist/violin + usable frac + n_blocks; b: R2 Δ vs label-shuffle null band; c: participation broad-vs-narrow). Self-contained labels (no §X / cluster_id jargon).
- [ ] Step 2: Render for pilot subjects; eyeball; fix; re-render. Write `figures/README.md` (中文, per-figure 关注点).
- [ ] Step 3: Commit `feat(topic5): event-resolved pilot figures + README`.

---

## Phase 3 — HARD STOP + recap (advisor=user gate)

### Task 9: Pilot recap + memory
- [ ] Write `docs/archive/topic5/event_resolved_axis_bias_pilot_<date>.md` with §0 三段式 (plain-language, hfosp-plain-language-recap), the substrate decision evidence, per-subject usable fractions / n_blocks / dispersion, and the explicit decision menu for the user (cohort run? substrate tweak? Stage B/C?). Append a row to `results/FIGURE_INDEX.md`.
- [ ] Update memory (project file) with pilot outcome + the v1→v2 design correction.
- [ ] **STOP.** Do not run cohort; do not write cohort verdicts. Await user sign-off.

---

## Self-Review (run after writing; per skill)

- Spec coverage: §3.1→Task3; §3.2→Task4; §3.3 R2→Task5, R1/diag→Task6/7; §C1→Task1; §C2→Task2; §C3/C4/C8→Task3; §C5→Task1/3; §C6→Task7; §C7→Task5; §C9→(cohort, deferred); §C10→Task6; §8 pilot-first→Task7/9. Stage B/C = stubs (Task6), full plan deferred. ✓
- Placeholders: none — each task has concrete tests + signatures + contract bindings; full code written in implementation (plan cites pseudocode in spec §3).
- Type consistency: `load_event_labels_ranks` dict keys reused verbatim by Tasks 3/4/5/7. `align`/`align1d`/`delta_median_obs` names consistent. ✓
