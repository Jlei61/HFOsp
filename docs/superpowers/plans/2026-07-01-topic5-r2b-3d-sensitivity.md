# Topic5 R2b native-3D Sensitivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.

**Goal:** Add a native-3D contact-kernel rung (R2b) to the contact-similarity ladder as a defensive sensitivity check: does the "same-plane geometry helps" result survive when we use native 3D Euclidean distance instead of the normalized 2D contact-plane projection? Primary comparison is `R2b − R2_nm` on a common coord-mapped channel subset.

**Architecture:** Extend the existing `src/topic5_contact_similarity.py` kernel to n-D, add R2b (native-3D) + R2_nm (2D-plane, no-mirror) computed on the SAME coord-mapped subset, via an augment path that rebuilds per-subject context FROM SOURCE artifacts (T0 cache + axis records + 3D coords) and reuses the stored R1/R2/R3 summary only for a cross-check. No R3 recompute.

**Tech Stack:** Python, numpy, scipy, matplotlib, pytest. Reuses `src/topic5_contact_similarity.py`, `src/seeg_coord_loader.py`, `src/propagation_skeleton_geometry.py`, `scripts/run_topic5_contact_similarity.py`.

## Global Constraints (inherit the ladder's; new ones flagged NEW)

- Inherit: `B=1000`, `seed=20260614`, `MIN_CH=6`, `MIN_SHAFTS=2`, `MIN_FINITE_PER_SZ=6`, `SESOI=0.05`, sign-free `maxAB=max(|sim_A|,|sim_B|)`, per-seizure→`median_s` fold (`_p95_med`), per-draw maxAB recompute, within-shaft primary null.
- **NEW — base branch:** work on `topic5-r2b-3d` (off local `main` `f31f572`, which HAS the contact-similarity code). Inputs/stored results via `--input-results-root /home/honglab/leijiaxin/HFOsp/results`.
- **NEW — R2b geometry:** native 3D mm coords from `load_subject_coords`; `sigma_3d = median 3D nearest-neighbor spacing` (mm); **no plane-mirror** (3D coords are anatomical/absolute — no transverse-PCA-sign ambiguity). Per-subject only, NEVER pool cross-dataset point clouds (Yuquan fs-native-RAS-mm vs Epilepsiae MNI152-mm).
- **NEW — units hard gate (P1-3):** call `assert_coord_result_is_mm_for_main_analysis(result)`; `allow_voxel_fallback=False`. Non-mm / loader raise / coord missing → that channel unmapped; if the mapped subset is too small → subject `r2b_status=NA`. NO silent fallback.
- **NEW — common subset (P1-1):** `R2b` and `R2_nm` MUST be computed on the SAME `coord_mapped` channel subset (channels present in BOTH the 2D axis record AND with valid mm 3D coords). The common subset must still satisfy `n_ch>=6`, `n_shafts>=2`, per-seizure `finite>=6`; else `r2b_status=NA` (do not force). Never compare full-channel stored R2 against reduced-channel R2b.
- **NEW — mirror handling (decision 1, user-approved):** primary = `R2b(3d,no-mirror) − R2_nm(2d,no-mirror)` (mirror held constant = off). Secondary DESCRIPTIVE field only = `R2b − R2_main(mirror,full)` to connect readers to the original R1/R2/R3 main figure; NOT a primary verdict.
- **NEW — augment from source (P1-2, decision 2):** rebuild each subject's context (matched channels, per-seizure `bb_auc`/`hfa` vectors, `bact` anchors, 3D coords) from source artifacts and run the FULL within-shaft null for R2_nm/R2b. Reuse the stored `cohort_summary_{activation}.json` R1/R2/R3 ONLY as a cross-check (R3 stays as-is; do NOT recompute the 81×81 grid). "全量" = all cohort subjects, not a full re-run of R3.
- Tier: sensitivity/robustness. Conclusion language守窄口径 — "2D-plane vs native-3D geometry"; NEVER "characterizes pathological network".

---

### Task 0: Base worktree (DONE by controller)

Worktree `.worktrees/topic5-r2b-3d` on branch `topic5-r2b-3d` off local `main` `f31f572`; contact-similarity code + `seeg_coord_loader` present; inputs reachable via `/home/honglab/leijiaxin/HFOsp/results`. No action for implementers.

---

### Task 1: Tighten conclusion language to the accepted narrow口径 (P1 verckept doc-fix)

**Files:** Modify `docs/archive/topic5/contact_similarity_ladder_2026-07-01.md`; Modify `results/topic5_ictal_recruitment/contact_similarity/figures/README.md`.

- [ ] **Step 1:** In the archive doc's conclusion section, add an explicit scope block (verbatim):
  - Supports ONLY: "Spatially weighted contact-level similarity captures the same coarse interictal–ictal spatial scaffold as the gridded field readout, indicating the field result is driven mainly by local spatial smoothing rather than grid interpolation." — a useful spatial readout / sensitivity metric.
  - Does NOT support: "effectively characterizes the epileptic pathological network." Evidence: within-shaft pass counts DROP R1→R3 (6/5/4 bb, 9/7/5 hfa) — no increased cohort-positive evidence.
  - Upgrade to a "pathological network" claim requires clinical validation (SOZ/resection/outcome, propagation endpoint, cross-window stability). The R2b 3D sensitivity (this plan) is a defensive check that the result is not a 2D-projection artifact, NOT such a clinical upgrade.
- [ ] **Step 2:** Mirror a one-line scope note into `figures/README.md` (Chinese, keep AGENTS.md format).
- [ ] **Step 3:** Commit `docs(topic5): tighten contact-similarity conclusion to accepted narrow口径`.

---

### Task 2: Generalize `kernel_smooth_at_contacts` to n-D

**Files:** Modify `src/topic5_contact_similarity.py`; Test `tests/test_topic5_contact_similarity.py`.

**Interface (unchanged signature, generalized body):** `kernel_smooth_at_contacts(values, source_pts, eval_pts, support, sigma)` — `source_pts`/`eval_pts` may be (n,2) OR (n,3). Distance is Euclidean over ALL coord columns.

- [ ] **Step 1: Write failing tests** (append to the test file):

```python
def test_kernel_3d_hand_weights():
    # 3 contacts in 3D; hand-compute the Nadaraya-Watson output at one eval point
    pts = np.array([[0.,0.,0.],[1.,0.,0.],[0.,0.,2.]]); vals = np.array([1.,2.,3.]); sup = np.ones(3); sigma = 1.0
    out = kernel_smooth_at_contacts(vals, pts, pts, sup, sigma)
    sig2 = 2.0*sigma*sigma
    # eval at pts[0]: d2 to each = [0,1,4]; w=exp(-d2/sig2)
    w = np.exp(-np.array([0.,1.,4.])/sig2); exp0 = (w*vals).sum()/w.sum()
    assert np.isclose(out[0], exp0)

def test_kernel_2d_regression_unchanged():
    # 2D path must be numerically identical to before generalization (protect the cross-check test)
    rng = np.random.default_rng(0); pts = rng.random((6,2)); vals = rng.random(6); sup = np.ones(6)
    out = kernel_smooth_at_contacts(vals, pts, pts, sup, 0.3)
    # recompute with an explicit 2-col Euclidean reference
    ref = np.array([ ( (np.exp(-(((pts-pts[i])**2).sum(1))/(2*0.3**2)) * vals).sum()
                       / np.exp(-(((pts-pts[i])**2).sum(1))/(2*0.3**2)).sum() ) for i in range(6)])
    assert np.allclose(out, ref)

def test_kernel_nan_coords_excluded():
    # a source contact with NaN coord must not contribute to any weight
    pts = np.array([[0.,0.],[np.nan,np.nan],[1.,0.]]); vals=np.array([1.,9.,2.]); sup=np.ones(3)
    out = kernel_smooth_at_contacts(vals, pts, pts, sup, 0.5)
    assert np.isfinite(out[0]) and np.isfinite(out[2])  # value 9 (NaN-coord) must not leak in

def test_kernel_sigma_nonpositive_raises():
    pts=np.array([[0.,0.],[1.,0.]]); vals=np.array([1.,2.]); sup=np.ones(2)
    with pytest.raises((ValueError,)):
        kernel_smooth_at_contacts(vals, pts, pts, sup, 0.0)
```

- [ ] **Step 2:** Run → confirm the new tests fail (3D hand-weights / nan-coords / sigma-raise) and the 2D regression passes only after generalization.
- [ ] **Step 3: Implement** — generalize `d2` to all columns; exclude NaN-coord sources; raise on `sigma<=0`:

```python
def kernel_smooth_at_contacts(values, source_pts, eval_pts, support, sigma):
    v = np.asarray(values, float); sup = np.asarray(support, float)
    src = np.asarray(source_pts, float); ev = np.asarray(eval_pts, float)
    if not (sigma > 0):
        raise ValueError(f"sigma must be > 0, got {sigma}")
    sig2 = 2.0 * float(sigma) * float(sigma)   # MUST match smooth_field
    coord_ok = np.isfinite(src).all(axis=1)     # NaN-coord sources never contribute
    out = np.full(ev.shape[0], np.nan)
    fin = np.isfinite(v) & coord_ok
    for i in range(ev.shape[0]):
        if not np.isfinite(ev[i]).all():
            continue
        d2 = ((src[coord_ok] - ev[i]) ** 2).sum(axis=1)   # n-D Euclidean
        w = sup[coord_ok] * np.exp(-d2 / sig2)
        if w.sum() <= 1e-12:
            continue
        fi = fin[coord_ok]
        wf = w[fi]
        if wf.sum() > 1e-12:
            out[i] = float((wf * v[coord_ok][fi]).sum() / wf.sum())
    return out
```
(Keep behavior byte-identical for the all-finite 2D case — the existing `test_kernel_matches_smooth_field_on_grid` must still pass.)

- [ ] **Step 4:** Run the FULL `tests/test_topic5_contact_similarity.py` → all prior 20 + new pass. **Also add** `median_nn_spacing(pts)` helper (n-D) with a test that all-identical coords → `median_nn=0` → callers must treat as NA (helper returns 0.0; the R2b builder raises/NA on `sigma_3d<=0`).
- [ ] **Step 5:** Commit `feat(topic5): n-D contact kernel (2D+3D) + guards`.

---

### Task 3: R2b (native-3D) + R2_nm (2D no-mirror) on the common coord-mapped subset

**Files:** Create `scripts/augment_topic5_r2b_3d.py`; Test `tests/test_augment_topic5_r2b_3d.py`.

**Consumes:** `kernel_smooth_at_contacts` (n-D), `contact_corr`/`_pearson_over_contacts`/`subject_null`/`fold_subject` (existing), `run_topic5_contact_similarity._ctx`-style loaders (matched channels, per-seizure vectors, bact anchors), `load_subject_coords`+`assert_coord_result_is_mm_for_main_analysis` (`src/seeg_coord_loader.py`), `parse_shaft`.

**Per-subject contract (encode exactly):**
1. Rebuild context FROM SOURCE (same loaders as `run_topic5_contact_similarity._ctx`): matched channels + names, per-seizure `{activation}__idx` vectors (≥6 finite), `bact__idx` anchors, plane `x_norm/y_norm`/support/`sigma_xy`.
2. `cr = load_subject_coords(ds, subj, matched_names, allow_voxel_fallback=False)`; `assert_coord_result_is_mm_for_main_analysis(cr)`. Build `coord_mapped_mask` (finite mm xyz).
3. **common subset** = matched channels with `coord_mapped_mask` True. Require `n_common>=6`, `n_shafts_common>=2` (via `parse_shaft` on common names), per-seizure finite-on-common `>=6`; else `r2b_status="NA"` with a reason, skip stats.
4. On the common subset, recompute with the SAME null harness (`subject_null`, within-shaft, per-draw maxAB, `_p95_med` fold, B=1000, seed):
   - `R2_nm` = 2D-plane kernel, mirror=False, sigma=`sigma_xy` (plane) — **recomputed on common subset**.
   - `R2b` = native-3D kernel on `cr.coords_array[common]` (mm), mirror=False, `sigma_3d=median_nn_spacing(coords3d_common)` (raise/NA if `<=0`).
5. Deltas: primary `r2b_minus_r2nm = R2b.obs_subject − R2_nm.obs_subject` (both no-mirror, common subset). Secondary descriptive `r2b_minus_r2main = R2b.obs_subject − stored R2(mirror,full).within_shaft.obs_subject`.
6. Cross-check: assert stored R1/R2/R3 present for the subject (from `cohort_summary_{activation}.json`); record `r3_obs` for provenance. Do NOT recompute R3.

- [ ] **Step 1: Write failing tests** (fixture-based, machine-independent where possible; one real-subject smoke gated on data):
  - common-subset eligibility: a synthetic subject where dropping no-coord channels leaves `n_common<6` → `r2b_status="NA"`.
  - units gate: monkeypatch `load_subject_coords` to return `coord_units="voxel"` → `assert_...` raises → subject `r2b_status="NA_units"` (caught, not crash).
  - R2_nm==R2b when 3D coords are exactly the 2D plane embedded in z=const (degenerate check): with `coords3d=[x,y,0]` and `sigma` matched, `R2b≈R2_nm` (sanity that the only difference is the coordinate space).
  - real-subject smoke (skip if data absent): `augment_subject("epilepsiae_1146", activation="broadband", B=20, input_results_root=MAIN)` → finite `r2b_minus_r2nm`, `r2b_status=="ok"`, `coord_units=="mm"`.
- [ ] **Step 2:** Run → fail (module missing).
- [ ] **Step 3: Implement** `augment_topic5_r2b_3d.py` per the contract above (`augment_subject(...)` + `main()` with `--activation --B 1000 --seed 20260614 --input-results-root --out-dir --subjects`). Reuse `run_topic5_contact_similarity`'s loaders (import them; do not reimplement).
- [ ] **Step 4:** Run tests → pass. Manual smoke on 1 subject to `/tmp`.
- [ ] **Step 5:** Commit `feat(topic5): R2b native-3D + R2_nm on common coord subset (augment, from source)`.

---

### Task 4: Coverage table + figure + README

**Files:** Modify `scripts/augment_topic5_r2b_3d.py` (write coverage CSV); Modify `scripts/plot_topic5_contact_similarity.py` (or a small new plot); Modify `figures/README.md`.

- [ ] **Step 1:** `main()` writes `r2b_coverage_{activation}.csv` with columns: `subject_id, n_matched_2d, n_coord_mapped_3d, n_common, n_shafts_common, coord_space, coord_units, r2b_status, missing_channels`. Plus `r2b_summary_{activation}.json` with per-subject `{R2_nm, R2b, r2b_minus_r2nm, r2b_minus_r2main, r2b_status}` + cohort `r2b_minus_r2nm` median/CI/`grid_negligible`-style SESOI verdict + n_ok.
- [ ] **Step 2:** Figure: add an `R2b` point to the geometry ladder AND an `R2b − R2_nm` per-subject panel (zero line; NA subjects greyed). Update `figures/README.md` (Chinese) describing R2b + the coverage caveat.
- [ ] **Step 3:** Render + structural check (PNG non-empty, labels). Commit `feat(topic5): R2b coverage table + figure + README`.

---

### Task 5: Full cohort augment run + cross-check + archive + merge

- [ ] **Step 1:** Run all 18 (both bands) via augment:
```
python scripts/augment_topic5_r2b_3d.py --activation broadband --B 1000 --input-results-root /home/honglab/leijiaxin/HFOsp/results --out-dir results/topic5_ictal_recruitment/contact_similarity
python scripts/augment_topic5_r2b_3d.py --activation hfa       --B 1000 --input-results-root /home/honglab/leijiaxin/HFOsp/results --out-dir results/topic5_ictal_recruitment/contact_similarity
```
- [ ] **Step 2:** Cross-check: for each subject the augment's recorded stored-R3 obs equals `cohort_summary_{band}.json` `real_median_abs_corr` (consistency; the augment must not have drifted the base). Verify R2_nm is finite for all `r2b_status=ok` subjects.
- [ ] **Step 3:** Regenerate figures from the augment outputs; verify coverage table (report how many subjects/channels dropped for missing coords).
- [ ] **Step 4:** Update `docs/archive/topic5/contact_similarity_ladder_2026-07-01.md` with the R2b results + the 3-way verdict (`R2b≈R2_nm`→2D suffices; `>`→3D extra info; `<`→narrow to 2D-plane), 守窄口径. Append FIGURE_INDEX note if a new figure dir.
- [ ] **Step 5:** Full test suite for the touched files. Commit. Then merge `topic5-r2b-3d` into `main` (fast-forward — it branches off `f31f572`).

---

## Acceptance gates (lock before running)
- Primary: `R2b − R2_nm` cohort paired Δ + bootstrap CI + SESOI(0.05) equivalence verdict, on `r2b_status=ok` subjects only.
- Coverage reported (n_ok / NA breakdown, channels dropped). Verdict language守窄口径 (2D-plane vs native-3D; never "pathological network").
- Bad-data regressions: units=voxel→NA; n_common<6→NA; sigma_3d<=0→NA/raise; NaN coords excluded from weights.

## Self-Review
- Spec coverage: P0 base(T0) ✓; P1-1 common-subset(T3 contract 3) ✓; P1-2 augment-from-source(Global + T3 contract 1) ✓; P1-3 units gate(Global + T3 contract 2) ✓; decision-1 R2_nm+secondary(Global + T3 step5) ✓; decision-2 augment(Global) ✓; kernel tests(T2) ✓; coverage table(T4) ✓.
