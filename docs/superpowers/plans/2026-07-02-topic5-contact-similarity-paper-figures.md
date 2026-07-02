# Topic5 Contact-Similarity — paper-grade figure redesign

> Implementer: superpowers:subagent-driven-development. `- [ ]` steps.

**Goal:** Replace the current dense/unintuitive contact-similarity figures with 3 clean paper-grade figures telling the honest narrow-口径 story: **method → direct contact-rank correspondence → grid/3D add nothing.**

**Branch/worktree:** `topic5-cs-figures` off `main 0472224` (has contact-sim + R2b code + data). Inputs via `--input-results-root /home/honglab/leijiaxin/HFOsp/results`.

**口径 (all captions, verbatim discipline):** show "spatially-weighted contact rank sequence captures the same coarse interictal↔ictal spatial scaffold as the gridded field; grid & native-3D add no distinguishable info." **NEVER** "predicts / characterizes the epileptic pathological network." Single-subject panels are illustrative; the honest cohort null-context sits beside them.

**Style:** follow `docs/figure_style_guide.md` (sequential=viridis; tight axes; shared legend; self-contained labels — no bare §/cluster-id/code axis labels). Chinese `figures/README.md` per AGENTS.md. Representative subject = **epilepsiae_1146** (15 contacts, 2 shafts, full coverage, high obs, passes R3 null).

**Inputs (all present):** interictal = `typical_rank` from `<AXIS_DIR>/epilepsiae_1146_t_a.json` (template A) + `_t_b.json` (template B), with plane `x_norm/y_norm/support`; ictal = `bb_auc` per contact from `<root>/topic5_ictal_recruitment/t0_feature_cache/epilepsiae_1146.npz` (`bb_auc__{idx}` per seizure; use the per-contact mean over seizures, matching the runner). Cohort metrics from `<CAN>/cohort_summary_{band}.json` (R1/R2/R3 `within_shaft.{obs_subject, null_q.p95}`) and `<CAN>/r2b_summary_{band}.json` (`R2_nm`, `R2b`, `r2b_minus_r2nm`). Gridded field via `R_smooth_rank(make_field_record(matched, vals), X, Y, sigma, S_THRESH)` + `make_plane_grid()`. Spatial weighting at contacts via `kernel_smooth_at_contacts`. `<CAN>` = `<root>/topic5_ictal_recruitment/contact_similarity`; `<AXIS_DIR>` = `<root>/spatial_modulation/propagation_geometry/observation_readout/real_subjects`.

**Reuse (import, don't reinvent):** `run_topic5_contact_similarity._ctx` (matched channels + per-seizure vectors + plane + sigma) to build the representative subject's context; `make_field_record`/`R_smooth_rank`/`make_plane_grid`/`S_THRESH`, `kernel_smooth_at_contacts`.

Output all to `<CAN>/figures/`: `fig1_spatial_weighting_schematic.png`, `fig2_rank_comparison.png`, `fig3_vs_field.png` (+ hfa variants where a band applies). Update `figures/README.md`.

---

### Task 1: Shared loader + Fig 2 (direct rank comparison — the main result figure)

**Files:** Create `scripts/plot_topic5_cs_paper.py` (shared `_load_subject_ctx(ds_sid, band, root)` + `fig2(...)`); Test `tests/test_plot_topic5_cs_paper.py` (smoke: renders non-empty PNG, panels have titles/labels).

**Fig 2 layout (2 panels):**
- **Left — rankdisp-style rank comparison (epilepsiae_1146):** contacts sorted along the propagation axis (T_a source→sink, as `plot_rank_displacement.py` sorts by `argsort(rank_a)`). Plot, per contact: **spatially-weighted interictal rank — template A in red (`#B71C2B`), template B in blue (`#1F4E9C`)** — and, on a paired track/twin axis, the **ictal early-broadband-energy rank** (rank of the spatially-weighted `bb_auc`). Reader sees the two interictal rank templates vs the ictal energy rank; annotate which template (A/B) is the maxAB match + its |corr|. Weighting = `kernel_smooth_at_contacts` on the plane at `sigma_xy`. Ranks normalized 0..1, low=early=source.
- **Right — cohort consistency vs null (null-比-null, mode a):** one row per subject (ordered by R2 obs), showing **R2 within_shaft obs** (spatially-weighted contact similarity) as a dot vs its **within-shaft-shuffle null p95** as a tick/bar (from `cohort_summary_broadband.json`); overlay R1 (unweighted) obs as a light open marker so the reader sees weighting raises obs AND the null (only a minority clear the line). Mark which subjects clear null. y-axis = |maxAB similarity|; caption states the honest reading.

- [ ] Step 1: write the smoke test (renders, panels labeled). Step 2: run→fail. Step 3: implement `_load_subject_ctx` + `fig2`. Step 4: render `fig2_rank_comparison.png` (broadband), structural-check (PNG > few KB, titles/labels/legend set). Step 5: commit `feat(topic5): paper Fig2 rank comparison + shared loader`.

---

### Task 2: Fig 1 (spatial-weighting method schematic)

**Files:** add `fig1(...)` to `scripts/plot_topic5_cs_paper.py`.

**Fig 1 layout (3 steps, left→right, epilepsiae_1146 plane):**
1. **输入**: contacts as dots at real plane positions (`x_norm`×`y_norm`), colored (viridis) by a per-contact interictal rank. Title "① 一个序列 = 每触点一个值,放到它的空间位置".
2. **空间加权**: same dots faint; pick 1–2 focal contacts, draw the Gaussian kernel as a translucent circle (radius≈σ=median-NN spacing) + faint lines to neighbors with width∝weight `exp(-d²/2σ²)`. Title "② 每触点新值 = 自己 + 邻近触点按距离(高斯核)加权平均".
3. **输出**: same dots recolored by the kernel-smoothed rank. Title "③ 空间平滑后的形状".
Pure method cartoon — no stats, no claim. Shared colorbar.

- [ ] Step 1: implement `fig1`. Step 2: render `fig1_spatial_weighting_schematic.png`, structural-check. Step 3: commit `feat(topic5): paper Fig1 spatial-weighting schematic`.

---

### Task 3: Fig 3 (vs field + native-3D)

**Files:** add `fig3(...)` to `scripts/plot_topic5_cs_paper.py`.

**Fig 3 layout (2 panels):**
- **Left — contact-weighted vs gridded field (epilepsiae_1146):** two spatial maps side by side of the SAME quantity (weighted interictal rank): (i) at contacts (`kernel_smooth_at_contacts`, dots), (ii) the 81×81 gridded field (`R_smooth_rank`, imshow with support mask) — visually the same shape. Title conveys "不铺网格(触点) vs 铺网格(场):同一形状".
- **Right — cohort: grid & 3D add nothing:** two scatters (or one 2-series): R2(触点核) vs R3(场) `obs_subject` per subject (from `cohort_summary`), and R2b(native-3D) vs R2_nm(2D-plane) per subject (from `r2b_summary`); both hug the y=x diagonal → grid/3D add no distinguishable info. Annotate the cohort SESOI equivalence (grid_delta and r2b_minus_r2nm medians+CI, both negligible). Diagonal reference line; ±SESOI band.

- [ ] Step 1: implement `fig3`. Step 2: render `fig3_vs_field.png`, structural-check. Step 3: commit `feat(topic5): paper Fig3 vs-field + native-3D`.

---

### Task 4: README + verify

- [ ] Update `<CAN>/figures/README.md` (Chinese, AGENTS.md format): one `### <filename>` block per figure, 2–4 句 + `**关注点**：`, 守窄口径. Run `pytest tests/test_plot_topic5_cs_paper.py -q`. Commit `feat(topic5): paper figures README`.
- [ ] NOTE: final VISUAL inspection is the user's — produce + structural-check only.

## Acceptance
Three PNGs render; each panel self-contained + labeled; captions strictly narrow-口径 (no pathological-network claim); representative-subject panels paired with honest cohort null-context; follow figure_style_guide.
