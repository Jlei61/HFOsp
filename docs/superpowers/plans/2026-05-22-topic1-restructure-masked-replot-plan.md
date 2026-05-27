# Topic 1 主文档重排 + masked 数据补图 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `docs/topic1_within_event_dynamics.md` to surface the two post-phantom-fix headline conclusions (K=2 dominant stereotypy + forward/reverse template swap geometry) while honoring CLAUDE.md §5 pre-registered tier discipline, and regenerate all topic0-affected figures on the masked (phantom-fixed) data tree at the canonical `results/interictal_propagation_masked/` top-level path.

**Architecture:** Three coupled deliverables executed in order — (D1) `git mv` the nested masked data tree to top-level and repair all path references; (D2) add `--masked-features` flag to 6 plot scripts mirroring the existing `plot_rank_displacement.py:73` pattern, run Batch 1 (4 figure groups blocking doc rewrite) with user eyeball checkpoint, then Batch 2 (3 secondary figure groups in parallel with D3); (D3) full TOC rewrite of topic1 main doc with §-anchor migration table.

**Tech Stack:** Python (matplotlib, numpy, pandas), bash + git, plain-text markdown. No new dependencies. All plot scripts already exist; this plan adds flags and rewrites a single doc.

**Source spec:** [`docs/superpowers/specs/2026-05-22-topic1-restructure-masked-replot-design.md`](../specs/2026-05-22-topic1-restructure-masked-replot-design.md) — read §0 three guardrails BEFORE writing §2 / §3 of the new topic1 doc.

---

## Phase 0 — D1: Directory migration

### Task 1: git mv the nested masked tree to top-level

**Files:**
- Move from: `results/interictal_propagation_vs_masked/interictal_propagation_masked/`
- Move to: `results/interictal_propagation_masked/`

- [ ] **Step 1: Verify source exists and destination is free**

Run:
```bash
ls -d results/interictal_propagation_vs_masked/interictal_propagation_masked/ \
  && ls -d results/interictal_propagation_masked/ 2>&1 | head -3
```
Expected: source listed; destination output contains `No such file or directory`.

- [ ] **Step 2: Run git mv**

Run:
```bash
git mv results/interictal_propagation_vs_masked/interictal_propagation_masked \
       results/interictal_propagation_masked
```
Expected: no output (git mv silent on success).

- [ ] **Step 3: Verify new structure**

Run:
```bash
ls results/interictal_propagation_masked/ | head -10 \
  && echo "---" \
  && ls results/interictal_propagation_vs_masked/
```
Expected: first listing has `per_subject pr1_cohort_summary.json template_anchoring template_pairing rank_displacement template_share_switching pr6_step6_held_out_template ...`; second listing has only `figures pr2_comparison.csv pr2_comparison_summary.md`.

### Task 2: Sweep and replace all repo references to the nested path

**Files:** all `*.md` and `*.py` referencing the nested path.

- [ ] **Step 1: Find every reference**

Run:
```bash
grep -rn "interictal_propagation_vs_masked/interictal_propagation_masked" \
  --include="*.md" --include="*.py" .
```
Expected: list of files containing the nested reference. Save this list; you'll re-run after replacement to verify 0 hits.

- [ ] **Step 2: Replace in markdown docs**

For each `*.md` file from Step 1, replace `interictal_propagation_vs_masked/interictal_propagation_masked` with `interictal_propagation_masked` using your editor (sed-style global replace per file). Most likely files:
- `AGENTS.md`
- `docs/topic0_methodology_audits.md`
- `docs/archive/topic0/lagpat_phantom_rank/*.md` (12+ step5*.md files; also `checkpoint_b_report_2026-05-21.md`, `phase0_progress_report_2026-05-21.md`, `plain_chinese_report_2026-05-20.md`, `rerun_results_2026-05-21.md`, `rerun_roadmap_2026-05-20.md`, `diagnostic_2026-05-20.md`)

Command pattern (run per-file or globally):
```bash
sed -i 's|interictal_propagation_vs_masked/interictal_propagation_masked|interictal_propagation_masked|g' AGENTS.md docs/topic0_methodology_audits.md docs/archive/topic0/lagpat_phantom_rank/*.md
```

- [ ] **Step 3: Replace in Python scripts**

For any `*.py` from Step 1, replace identically:
```bash
sed -i 's|interictal_propagation_vs_masked/interictal_propagation_masked|interictal_propagation_masked|g' $(grep -rl "interictal_propagation_vs_masked/interictal_propagation_masked" --include="*.py" scripts/ src/ tests/ 2>/dev/null)
```

- [ ] **Step 4: Verify 0 dangling references**

Run (same command as Step 1):
```bash
grep -rn "interictal_propagation_vs_masked/interictal_propagation_masked" \
  --include="*.md" --include="*.py" .
```
Expected: **0 lines of output**. If any remain, fix them manually.

- [ ] **Step 5: Verify masked path tests still pass**

Run:
```bash
pytest tests/test_attractor_masked_features.py tests/test_pr5_masked_path_routing.py tests/test_pr6_masked_path_routing.py tests/test_pr7_masked_path_routing.py -v 2>&1 | tail -20
```
Expected: all tests PASS. These tests assert the run-scripts use `results/interictal_propagation_masked/`, which is now the real path (was an unwritten convention before). If they fail, investigate before proceeding.

### Task 3: Add `vs_masked/README.md` explaining the residual contents

**Files:**
- Create: `results/interictal_propagation_vs_masked/README.md`

- [ ] **Step 1: Write the README**

Create `results/interictal_propagation_vs_masked/README.md` with:

```markdown
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
```

- [ ] **Step 2: Verify README exists**

Run:
```bash
test -f results/interictal_propagation_vs_masked/README.md && echo "OK"
```
Expected: `OK`.

### Task 4: Commit Phase 0 — D1 directory migration

- [ ] **Step 1: Stage and commit**

Run:
```bash
git add -A results/interictal_propagation_masked results/interictal_propagation_vs_masked \
        AGENTS.md docs/topic0_methodology_audits.md docs/archive/topic0/lagpat_phantom_rank/ \
        scripts/ tests/
git status --short
```

Then commit:
```bash
git commit -m "$(cat <<'EOF'
chore(results): promote interictal_propagation_masked/ to top-level parallel dir

Per AGENTS.md "parallel directory" convention, masked re-derivation outputs
should live at results/interictal_propagation_masked/ (sibling to the
phantom-contaminated results/interictal_propagation/), not nested under
results/interictal_propagation_vs_masked/.

- git mv the nested tree to the canonical top-level path
- sed-replace all references in docs (AGENTS.md, topic0_methodology_audits.md,
  docs/archive/topic0/lagpat_phantom_rank/*.md) and scripts
- vs_masked/ retains only the orig-vs-masked diagnostic artefacts
  (pr2_comparison.csv + pr2_comparison_summary.md + figures/2 diagnostic plots)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit succeeds. If pre-commit hooks fail, fix issues, re-stage, create a NEW commit (do not amend).

- [ ] **Step 2: Verify tests still pass after commit**

Run:
```bash
pytest tests/test_attractor_masked_features.py tests/test_pr5_masked_path_routing.py tests/test_pr6_masked_path_routing.py tests/test_pr7_masked_path_routing.py -v 2>&1 | tail -10
```
Expected: all PASS.

---

## Phase 1 — Add `--masked-features` to 6 plot scripts

> **Pattern to mirror** (already implemented in `scripts/plot_rank_displacement.py:64-84` + `scripts/plot_rank_displacement.py:1006-1010`): module-level `RES_DIR` / `FIG_DIR` declared as legacy path; `_apply_masked_paths()` function swaps them to the `_masked` tree via `global` assignment; argparse exposes `--masked-features`; `main()` calls `_apply_masked_paths()` before any path use when flag is set.
>
> All 6 tasks below follow the same shape. The differences are: (a) which path globals to swap, (b) the script's argparse layout, (c) when in `main()` to call the swap.

### Task 5: Add `--masked-features` to `plot_interictal_propagation.py`

**Files:**
- Modify: `scripts/plot_interictal_propagation.py` lines 40–46 (module-level paths) and the `__main__` block (need to read it first)

- [ ] **Step 1: Inspect current path globals and main block**

Run:
```bash
sed -n '38,50p' scripts/plot_interictal_propagation.py
sed -n '1685,1699p' scripts/plot_interictal_propagation.py
```
Note the exact current values of `RESULTS_DIR`, `FIG_DIR`, `PER_SUBJECT_DIR`, `PR3_FIG_DIR`, `PR4A_FIG_DIR`, and the `if __name__ == "__main__":` block structure. (Lines may shift; trust the search not the line numbers.)

- [ ] **Step 2: Replace module-level path block**

In `scripts/plot_interictal_propagation.py`, find this block (around lines 40–46):

```python
RESULTS_DIR = Path("results/interictal_propagation")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PER_SUBJECT_DIR = RESULTS_DIR / "per_subject"
PR3_FIG_DIR = FIG_DIR / "per_subject"
PR4A_FIG_DIR = FIG_DIR / "per_subject"
PR4A_FIG_DIR.mkdir(parents=True, exist_ok=True)
```

Replace with:

```python
# Legacy (non-masked) paths. `_apply_masked_paths()` swaps these to the
# `_masked` parallel tree (Topic 0 §3.1 phantom-rank rerun, 2026-05-21).
RESULTS_DIR = Path("results/interictal_propagation")
FIG_DIR = RESULTS_DIR / "figures"
PER_SUBJECT_DIR = RESULTS_DIR / "per_subject"
PR3_FIG_DIR = FIG_DIR / "per_subject"
PR4A_FIG_DIR = FIG_DIR / "per_subject"


def _apply_masked_paths() -> None:
    """Reassign module-level path globals to the `_masked` parallel tree.

    Mirrors scripts/plot_rank_displacement.py:_apply_masked_paths so the
    plotting script consumes the masked per-subject JSONs / cohort_summary
    and writes figures to results/interictal_propagation_masked/figures/.
    """
    global RESULTS_DIR, FIG_DIR, PER_SUBJECT_DIR, PR3_FIG_DIR, PR4A_FIG_DIR
    RESULTS_DIR = Path("results/interictal_propagation_masked")
    FIG_DIR = RESULTS_DIR / "figures"
    PER_SUBJECT_DIR = RESULTS_DIR / "per_subject"
    PR3_FIG_DIR = FIG_DIR / "per_subject"
    PR4A_FIG_DIR = FIG_DIR / "per_subject"


# `mkdir` deferred to inside `main()` so `_apply_masked_paths()` can run first.
```

Note the **two semantic changes** that go with the path block move:
1. The `FIG_DIR.mkdir(...)` and `PR4A_FIG_DIR.mkdir(...)` at module-import time are removed — they would create the *legacy* dir before `_apply_masked_paths()` runs. They will be re-created in `main()` after the path swap.
2. Add the `_apply_masked_paths()` function definition.

- [ ] **Step 3: Find or create the `main()` / `__main__` block**

Run:
```bash
sed -n '/^if __name__/,$p' scripts/plot_interictal_propagation.py
```

If a `main()` function or `if __name__ == "__main__":` block exists, note its argparse layout. If a `main()` function does NOT exist (the script may be a top-level run), wrap the bottom-level execution in `def main():` first.

- [ ] **Step 4: Add `--masked-features` to argparse and call `_apply_masked_paths()` early**

In the script's argparse setup (inside `main()` or `__main__`), add:

```python
parser.add_argument(
    "--masked-features",
    action="store_true",
    help="Consume masked PR-2 cluster outputs and write figures under "
         "results/interictal_propagation_masked/figures/. Mirrors "
         "scripts/plot_rank_displacement.py --masked-features.",
)
```

Then, immediately after `args = parser.parse_args()` (and BEFORE any path-using call), add:

```python
if args.masked_features:
    _apply_masked_paths()

FIG_DIR.mkdir(parents=True, exist_ok=True)
PR4A_FIG_DIR.mkdir(parents=True, exist_ok=True)
```

If the script has no argparse at all (it may just run top-to-bottom), wrap the bottom in:

```python
def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__ or "plot_interictal_propagation")
    parser.add_argument("--masked-features", action="store_true", ...)
    args = parser.parse_args()
    if args.masked_features:
        _apply_masked_paths()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PR4A_FIG_DIR.mkdir(parents=True, exist_ok=True)
    # ... (existing top-level execution code, indented one level) ...
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Smoke test — flag is recognized and help text shows it**

Run:
```bash
python scripts/plot_interictal_propagation.py --help 2>&1 | grep -A1 "masked-features"
```
Expected: `--masked-features` flag listed with its help text.

- [ ] **Step 6: Smoke test — masked path is used when flag set**

Add a temporary one-line check (or use python -c):

```bash
python -c "
import sys; sys.argv = ['plot_interictal_propagation.py', '--masked-features']
import scripts.plot_interictal_propagation as m
m._apply_masked_paths()
print('RESULTS_DIR:', m.RESULTS_DIR)
print('FIG_DIR:', m.FIG_DIR)
assert str(m.RESULTS_DIR) == 'results/interictal_propagation_masked'
assert str(m.FIG_DIR) == 'results/interictal_propagation_masked/figures'
print('OK')
"
```
Expected: paths print as `results/interictal_propagation_masked` + `results/interictal_propagation_masked/figures` + `OK`.

If `import scripts.plot_interictal_propagation` fails (no `__init__.py` in scripts/), use:

```bash
python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('plot_interictal_propagation', 'scripts/plot_interictal_propagation.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m._apply_masked_paths()
print('RESULTS_DIR:', m.RESULTS_DIR); print('FIG_DIR:', m.FIG_DIR)
assert str(m.RESULTS_DIR) == 'results/interictal_propagation_masked'
print('OK')
"
```

- [ ] **Step 7: Commit Task 5 alone**

```bash
git add scripts/plot_interictal_propagation.py
git commit -m "feat(plot): add --masked-features flag to plot_interictal_propagation

Mirrors scripts/plot_rank_displacement.py:_apply_masked_paths so the
PR-2/PR-3 cohort + per_subject + heatmap + pr4a_daynight plots can be
regenerated on the Topic 0 masked feature tree.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 6: Add `--masked-features` to `plot_pr6_template_anchoring.py`

**Files:**
- Modify: `scripts/plot_pr6_template_anchoring.py` line 56 (module-level paths) + `main()` at line 741

- [ ] **Step 1: Replace module-level paths**

In `scripts/plot_pr6_template_anchoring.py`, find:

```python
RESULTS_DIR = Path("results/interictal_propagation/template_anchoring")
FIG_DIR = RESULTS_DIR / "figures"
PER_SUBJECT_DIR = RESULTS_DIR / "per_subject"
```

Replace with:

```python
# Legacy (non-masked) paths. `_apply_masked_paths()` swaps these to the
# `_masked` parallel tree (Topic 0 §3.1 phantom-rank rerun, 2026-05-21).
RESULTS_DIR = Path("results/interictal_propagation/template_anchoring")
FIG_DIR = RESULTS_DIR / "figures"
PER_SUBJECT_DIR = RESULTS_DIR / "per_subject"


def _apply_masked_paths() -> None:
    """Reassign module-level path globals to the `_masked` parallel tree."""
    global RESULTS_DIR, FIG_DIR, PER_SUBJECT_DIR
    RESULTS_DIR = Path("results/interictal_propagation_masked/template_anchoring")
    FIG_DIR = RESULTS_DIR / "figures"
    PER_SUBJECT_DIR = RESULTS_DIR / "per_subject"
```

- [ ] **Step 2: Add `--masked-features` flag in `main()` argparse**

In `scripts/plot_pr6_template_anchoring.py:741-750`, find the argparse block in `main()`. Add the flag immediately after the existing `--example-dataset` arg:

```python
ap.add_argument(
    "--masked-features",
    action="store_true",
    help="Consume masked PR-6 outputs (results/interictal_propagation_masked/"
         "template_anchoring/) and write figures next to them.",
)
```

Then, immediately after `args = ap.parse_args()`, add:

```python
if args.masked_features:
    _apply_masked_paths()
```

This must happen BEFORE `cohort = load_cohort()` and `FIG_DIR.mkdir(...)`.

- [ ] **Step 3: Smoke test**

```bash
python scripts/plot_pr6_template_anchoring.py --help 2>&1 | grep "masked-features"
```
Expected: line containing `--masked-features`.

- [ ] **Step 4: Path swap verification**

```bash
python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('m', 'scripts/plot_pr6_template_anchoring.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m._apply_masked_paths()
assert str(m.RESULTS_DIR) == 'results/interictal_propagation_masked/template_anchoring'
print('OK')
"
```
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add scripts/plot_pr6_template_anchoring.py
git commit -m "feat(plot): add --masked-features flag to plot_pr6_template_anchoring

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 7: Add `--masked-features` to `plot_pr6_swap_cluster_rank_multiples.py`

**Files:**
- Modify: `scripts/plot_pr6_swap_cluster_rank_multiples.py`

- [ ] **Step 1: Inspect current path globals**

Run:
```bash
grep -nE "^[A-Z_]+ ?=.*results.*interictal|argparse|add_argument" scripts/plot_pr6_swap_cluster_rank_multiples.py | head -20
```
Note the current path declarations and argparse layout.

- [ ] **Step 2: Apply the same pattern as Task 6**

Replace path declarations with legacy + `_apply_masked_paths()` swap function. Add `--masked-features` argparse flag. Call `_apply_masked_paths()` immediately after `args = parser.parse_args()`.

The exact path names to swap: whichever module-level Path globals point to `results/interictal_propagation/...`. In the swap function, replace `interictal_propagation` with `interictal_propagation_masked` (everything else identical).

- [ ] **Step 3: Smoke test**

```bash
python scripts/plot_pr6_swap_cluster_rank_multiples.py --help 2>&1 | grep "masked-features"
```
Expected: line containing `--masked-features`.

- [ ] **Step 4: Path swap verification** — same shape as Task 6 Step 4, with `plot_pr6_swap_cluster_rank_multiples.py` substituted.

- [ ] **Step 5: Commit**

```bash
git add scripts/plot_pr6_swap_cluster_rank_multiples.py
git commit -m "feat(plot): add --masked-features flag to plot_pr6_swap_cluster_rank_multiples

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 8: Add `--masked-features` to `plot_pr6_step6.py`

**Files:**
- Modify: `scripts/plot_pr6_step6.py`

Same pattern as Task 7 (inspect → replace paths + add swap function → add argparse flag → smoke test → commit). The swap target path is `results/interictal_propagation_masked/pr6_step6_held_out_template/` (mirror whatever the legacy script uses, with `interictal_propagation_masked` substituted).

- [ ] **Step 1: Inspect** — `grep -nE "^[A-Z_]+ ?=.*results.*interictal|argparse|add_argument" scripts/plot_pr6_step6.py | head -20`
- [ ] **Step 2: Apply pattern**
- [ ] **Step 3: Smoke test**
- [ ] **Step 4: Path swap verification**
- [ ] **Step 5: Commit** with message `feat(plot): add --masked-features flag to plot_pr6_step6`

### Task 9: Add `--masked-features` to `plot_pr6_sup1_rank_entropy.py`

**Files:**
- Modify: `scripts/plot_pr6_sup1_rank_entropy.py`

Same pattern as Task 8. Note: this script is "Topic 4 preflight, descriptive only" per the archive (see `docs/archive/topic1/propagation/pr6_supplementary_rank_entropy_results_2026-05-10.md`); regenerating it is for completeness, NOT to support the new §2 / §3 narrative.

- [ ] **Step 1: Inspect**
- [ ] **Step 2: Apply pattern**
- [ ] **Step 3: Smoke test**
- [ ] **Step 4: Path swap verification**
- [ ] **Step 5: Commit** with message `feat(plot): add --masked-features flag to plot_pr6_sup1_rank_entropy`

### Task 10: Add `--masked-features` to `plot_topic1_pr4_ppt.py`

**Files:**
- Modify: `scripts/plot_topic1_pr4_ppt.py`

Same pattern as Task 8. PPT panels are described as descriptive layer in the existing main doc (PR-4 PPT panel d → §4.5 secondary diagnostic). They go to Batch 2 (after eyeball).

- [ ] **Step 1: Inspect**
- [ ] **Step 2: Apply pattern**
- [ ] **Step 3: Smoke test**
- [ ] **Step 4: Path swap verification**
- [ ] **Step 5: Commit** with message `feat(plot): add --masked-features flag to plot_topic1_pr4_ppt`

---

## Phase 2 — D2 Batch 1: regenerate the four headline figure groups

> Goal: produce the 4 figure groups that the new §2 / §3.2 narrative will reference, so D3 can cite real file paths. Each invocation writes into `results/interictal_propagation_masked/.../figures/`. After all four are produced, write a README per new figures dir, then STOP and request user eyeball before Phase 3.

### Task 11: Regenerate PR-2/3 cohort + per_subject figures on masked data

**Files:**
- Produced: `results/interictal_propagation_masked/figures/cohort_propagation_summary.png` + `figures/per_subject/<dataset>_<subject>_propagation.png` (40 expected)

- [ ] **Step 1: Run plot_interictal_propagation with masked flag**

```bash
python scripts/plot_interictal_propagation.py --masked-features 2>&1 | tail -30
```
Expected: cohort + per-subject figures generated under `results/interictal_propagation_masked/figures/`. Last lines should reference written files. No traceback.

- [ ] **Step 2: Verify outputs**

```bash
ls results/interictal_propagation_masked/figures/cohort_propagation_summary.png \
   results/interictal_propagation_masked/figures/per_subject/ 2>&1 | head -10
```
Expected: cohort_summary file exists; per_subject/ directory has 40-ish `_propagation.png` files.

### Task 12: Regenerate PR-6 template_anchoring main figure on masked data

**Files:**
- Produced: `results/interictal_propagation_masked/template_anchoring/figures/pr6_template_pair_geometry_main.png` (+ supp coreness/jaccard/fwdrev if you pass `--all`)

- [ ] **Step 1: Run plot_pr6_template_anchoring --main with masked flag**

```bash
python scripts/plot_pr6_template_anchoring.py --masked-features --main 2>&1 | tail -10
```
Expected: `pr6_template_pair_geometry_main.png` written under masked path.

- [ ] **Step 2: Verify**

```bash
ls -la results/interictal_propagation_masked/template_anchoring/figures/pr6_template_pair_geometry_main.png
```
Expected: file exists, non-zero size.

### Task 13: Regenerate PR-6 swap cluster rank multiples on masked data

**Files:**
- Produced: `results/interictal_propagation_masked/template_anchoring/figures/pr6_supp_swap_cluster_rank_multiples.{png,pdf}` + `_nonstrong.{png,pdf}` (mirror legacy output filenames)

- [ ] **Step 1: Run with masked flag**

```bash
python scripts/plot_pr6_swap_cluster_rank_multiples.py --masked-features 2>&1 | tail -10
```
Expected: figure files written under masked template_anchoring/figures/.

- [ ] **Step 2: Verify**

```bash
ls results/interictal_propagation_masked/template_anchoring/figures/pr6_supp_swap_cluster_rank_multiples* 2>&1 | head -5
```

### Task 14: Regenerate PR-6 rank_displacement cohort heatmap on masked data

**Files:**
- Produced: `results/interictal_propagation_masked/rank_displacement/figures/*.{png,pdf}`

- [ ] **Step 1: Run with masked flag** (script already has the flag — Task 5–10 don't touch it)

```bash
python scripts/plot_rank_displacement.py --masked-features 2>&1 | tail -10
```
Expected: cohort_displacement_heatmap + per_subject figures written under masked rank_displacement/figures/.

- [ ] **Step 2: Verify**

```bash
ls results/interictal_propagation_masked/rank_displacement/figures/ 2>&1 | head -10
```

### Task 15: Write `figures/README.md` for each new masked figures dir

**Files:**
- Create: `results/interictal_propagation_masked/figures/README.md`
- Create: `results/interictal_propagation_masked/template_anchoring/figures/README.md`
- Create: `results/interictal_propagation_masked/rank_displacement/figures/README.md`

- [ ] **Step 1: Write `results/interictal_propagation_masked/figures/README.md`**

```markdown
# interictal_propagation (masked) — PR-2/PR-3 figures

Figures regenerated on the Topic 0 phantom-rank masked feature tree
(2026-05-22 D2 Batch 1). Source data: `../per_subject/*.json` +
`../pr1_cohort_summary.json`. Generator: `scripts/plot_interictal_propagation.py
--masked-features`. Phantom-fix details: `docs/topic0_methodology_audits.md` §3.1.

### cohort_propagation_summary.png
6-panel cohort summary on masked features: stable_k distribution, within-cluster
τ vs overall τ uplift, identity bias raw/centered, mixture taxonomy. Tier defaults
to Tier 1 (n=33 primary); see `_n33.png` / `_n40.png` for explicit tier variants
if regenerated. **关注点**：stable_k 分布是否保持 dominant k=2（Tier 0 27/30）；
within-cluster τ uplift 是否方向保持；identity bias (raw vs centered) 是否方向保持。

### per_subject/<dataset>_<subject>_propagation.png
Per-subject 3-panel propagation summary (rank-time scatter colored by cluster +
template overlay + MI distribution). 40 subjects expected. **关注点**：每张图
template overlay 颜色对齐 cluster；右下角 MI distribution 是否仍 significant。
```

- [ ] **Step 2: Write `results/interictal_propagation_masked/template_anchoring/figures/README.md`**

```markdown
# PR-6 template anchoring (masked) — endpoint geometry figures

Figures regenerated on the Topic 0 phantom-rank masked feature tree
(2026-05-22 D2 Batch 1). Generator: `scripts/plot_pr6_template_anchoring.py
--masked-features --all` + `scripts/plot_pr6_swap_cluster_rank_multiples.py
--masked-features`. Phantom-fix details: `docs/topic0_methodology_audits.md` §3.1;
PR-6 masked rerun verdicts: `docs/archive/topic0/lagpat_phantom_rank/step5f_pr6_results_2026-05-21.md`.

### pr6_template_pair_geometry_main.png
PR-6 6-panel main: subject swap-vs-same paired test + node anatomy + fwd/rev
small multiples + endpoint geometry summary. **关注点**：H2 fwd/rev 子集 sign-test
是否 cleanly positive（masked 期望 8/8 vs orig 9/9）；H1 SOZ anchoring cohort
Wilcoxon 仍 NULL（masked p ≈ 0.388）。

### pr6_supp_coreness_sensitivity.png / pr6_supp_endpoint_jaccard_per_subject.png / pr6_supp_fwdrev_small_multiples.png
Supplementary: endpoint metric sensitivity (top-3 vs coreness top-20%), split-half/odd-even
endpoint Jaccard per subject, fwd/rev subject pair small-multiples. **关注点**：endpoint metric
变化下 7/20 direction-discordant 仍存在；endpoint Jaccard split-half 0.71 / odd-even 0.93
方向保持。

### pr6_supp_swap_cluster_rank_multiples.{png,pdf} / _nonstrong.{png,pdf}
Variable-k swap classifier dual-tier small-multiples. strict (T_obs ≥ 0.5, p_fw < 0.05) +
candidate (0.05 ≤ p < 0.20) tier. **关注点**：masked strict 9/28 / candidate 6/28
（orig 10/35 + 8/35）；strict 子集是否仍主要落在 PR-2.5 fwd/rev-reproduced 集合。
```

- [ ] **Step 3: Write `results/interictal_propagation_masked/rank_displacement/figures/README.md`**

```markdown
# PR-6 rank displacement (masked) — continuous swap geometry figures

Figures regenerated on the Topic 0 phantom-rank masked feature tree
(2026-05-22 D2 Batch 1). Generator: `scripts/plot_rank_displacement.py
--masked-features`. Phantom-fix details: `docs/topic0_methodology_audits.md` §3.1.

### cohort_displacement_heatmap.{png,pdf}
Cohort rank displacement heatmap (rows = subjects sorted by Kendall τ, columns
= channels by template rank). swap_class strict markers indicate the n_strict
subset; dashed reference at 2/3. **关注点**：cohort F_norm median ≈ 0.79 (masked),
τ median ≈ −0.24, ρ(F_norm, τ) ≈ −0.92 (强负相关持平). PR-2.5 reproduced subjects
should still cluster in the upper τ band.

### per_subject/<dataset>_<subject>_*.png
Per-subject rank displacement small-multiples. **关注点**：每张图列按 rank_T_a_dense
排序避免 sorting bias；Δr sign anchor 仅 subject 内部有效。
```

- [ ] **Step 4: Verify all three READMEs exist**

```bash
for d in results/interictal_propagation_masked/figures \
         results/interictal_propagation_masked/template_anchoring/figures \
         results/interictal_propagation_masked/rank_displacement/figures; do
  test -f "$d/README.md" && echo "OK: $d/README.md" || echo "MISSING: $d/README.md"
done
```
Expected: 3 `OK:` lines.

### Task 16: Commit Batch 1 figures + READMEs and STOP for user eyeball

- [ ] **Step 1: Stage and commit**

```bash
git add results/interictal_propagation_masked/figures/ \
        results/interictal_propagation_masked/template_anchoring/figures/ \
        results/interictal_propagation_masked/rank_displacement/figures/
git status --short results/interictal_propagation_masked/ | head -20
git commit -m "$(cat <<'EOF'
feat(figures): D2 Batch 1 — regenerate 4 headline figure groups on masked tree

Topic 0 phantom-rank masked rerun (2026-05-22 D2 Batch 1):
- PR-2/3 cohort 6-panel + 40 per-subject propagation
- PR-6 template_anchoring main 6-panel + supp coreness/jaccard/fwdrev
- PR-6 swap cluster rank multiples (strict + nonstrong)
- PR-6 rank_displacement cohort heatmap + per-subject

Each new figures dir has a Chinese README per AGENTS.md figures convention.
Old phantom-contaminated figures at results/interictal_propagation/figures/
remain untouched (will be marked SUPERSEDED in Phase 3).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 2: REQUEST USER EYEBALL** (hard checkpoint — do not proceed to Phase 3 without explicit user OK)

Stop and tell the user: "Batch 1 figures committed. Please eyeball:
- `results/interictal_propagation_masked/figures/cohort_propagation_summary.png`
- `results/interictal_propagation_masked/template_anchoring/figures/pr6_template_pair_geometry_main.png`
- `results/interictal_propagation_masked/template_anchoring/figures/pr6_supp_swap_cluster_rank_multiples.png`
- `results/interictal_propagation_masked/rank_displacement/figures/cohort_displacement_heatmap.png`

Confirm visual sanity (axes labels, colormap, no NaN/empty panels, fwd/rev sign-test direction matches advisor's expectation of 8/8 positive on masked). Reply OK to proceed to Phase 3 (doc rewrite) or request fixes."

**Per CLAUDE.md §7 figure discipline: do NOT skip this checkpoint.**

---

## Phase 3 — D3: Rewrite `docs/topic1_within_event_dynamics.md`

> Re-read the spec §0 three guardrails before each section: `docs/superpowers/specs/2026-05-22-topic1-restructure-masked-replot-design.md`. Re-read the §2.1–§2.4 wording draft in the spec; it is the source of truth for §2 in this rewrite.

### Task 17: Save the current doc as a backup and write the new §0–§1

**Files:**
- Modify: `docs/topic1_within_event_dynamics.md`
- Backup: skip (git history is the backup; do not litter the tree with `.bak` files)

- [ ] **Step 1: Open the current main doc and read end-to-end**

Use the Read tool on `docs/topic1_within_event_dynamics.md` (you may need 2 reads, file is 526 lines). Internalize the existing PR-2/2.5/3/4*/5/6/7 narrative before rewriting; do NOT lose any cited result number — only reorganize.

- [ ] **Step 2: Write the new §0 §-anchor migration table at the very top**

Replace the current document (or use Edit to splice) so the file starts with:

```markdown
# Topic 1：间期事件内部时序结构

> 状态：当前正式入口（**2026-05-22 重排**：把 K=2 stability + fwd/rev swap geometry 升到第一序结论；详见 §0 §-anchor 迁移表）
> 范围：只讨论单个间期群体事件内部的时序组织，包括传播刻板性与事件级同步性。
> **Paper 1 架构性 framework**：`docs/paper1_framework_sba.md`（SBA framework：单核心假设 + 5 sharp predictions + 失败模式）。本 topic 的 PR-2 / PR-2.5 / PR-6 / PR-7 / 待立 PR-9 全部受该框架统辖。
> **Topic 4 模型层 framework（2026-05-20 lock）**：`docs/topic4_sef_itp_framework.md`（SEF-ITP：空间易激场模型，H1–H6 + Phase 0–4 路线）。**Phase 0 解锁 2026-05-21**：phantom-rank 修复 5a–5h 完成 + coord loader v3.1 落地。

---

## 0. §-anchor 迁移表（旧 §X → 新 §Y）

旧版主文档（pre-2026-05-22）累积了大量历史细节，把 PR-2 stable_k 现象学和 PR-6
swap geometry 都淹没在 §3.1 / §7.10 的细节里。本次重排目的：把"K=2 大范围间期稳定
刻板时序" + "fwd/rev template start-rank ↔ end-rank swap geometry" 两条 phantom-fix
后仍稳健的主结论升到 §2 / §3.2，其余慢调制 / 发作邻近 / SOZ anchoring / mark
dependence 全部 cohort-level NULL 集中放 §2.3。

archive 文档不在本次重排范围内，原始 §-anchor 引用按下表映射：

| 旧 § | 新 § | 备注 |
|---|---|---|
| §2 一句话当前结论 | §2.1–§2.4 | 重写 + 拆分按"现象层 / 机制层 / NULL / exploratory" 四段 |
| §3.1 内部传播刻板性 | §3.1（合并到 K=2 stability 段） | 与 §3.1b 合并 |
| §3.1b 数据合同/聚类稳定性 | §3.1 | 数字保留 |
| §3.1b.1 Tier 1/Tier 2 表 | §3.1 末尾 | 表不动 |
| §3.1c PR-3/PR-4A occupancy | §3.4（降级到描述层） | |
| §3.1d Cluster geometry viz | §3.3（缩短） | trilateration + bimodality audit |
| §3.2 Identity bias 簇内 | §3.1 末段 | masked rerun 加强（92.2%） |
| §3.3–§3.4 synchrony | §3.4 | 合并 |
| §4 当前最可信结果 | §5 | 合并到风险段 |
| §5 仍未解决的问题 | §5 | 保留 |
| §6 三类读数框架 | archive | 移走，主 doc 只在 §4 PR-by-PR 表的 PR-4 行链接 |
| §7.1–§7.4 PR-4 验收 | §4 PR-by-PR 表 | 一段总结 + archive 链接 |
| §7.5 优先级 | §6 推荐下一步 | 缩短 |
| §7.6–§7.9 模型层 / 子集 | archive | 主 doc 删，保留 archive |
| §7.10 PR-6 endpoint anchoring | §3.2 + §4 PR-6 行 | 主要内容上提到 §3.2，verdict 表移到 §4 |
| §7.11 PR-7 pairing | §4 PR-7 行 | 一段总结 |
| §8 代码与结果入口 | §7 | 更新到 masked path |
| §9 跨 topic 边界 | §8 | 不动 |
| §10 历史文档索引 | §9 | 瘦身，删 SUPERSEDED 链接 |
| §11 文档整理里程碑 | §9 末尾 | 加 2026-05-22 entry |

---

## 1. 这个 topic 只回答什么问题

本 topic 只回答两个问题：

1. 单个群体事件内部，不同通道的激活顺序是否稳定、是否刻板、是否存在多种主要传播模式。
2. 单个事件内部的同步性指标在发作前后是否表现出系统性变化。

它**不**回答：

- 事件与事件之间的 IEI / PSD / rate modulation：那是 `docs/topic2_between_event_dynamics.md`
- 慢调制发生在 SOZ 还是 non-SOZ：那是 `docs/topic3_spatial_soz_modulation.md`

---
```

- [ ] **Step 3: Verify the new §0 / §1 is in place**

```bash
sed -n '1,80p' docs/topic1_within_event_dynamics.md
```
Expected: file starts with the new §0 migration table and §1.

### Task 18: Rewrite §2 (the four-section first-order conclusion block)

**Files:**
- Modify: `docs/topic1_within_event_dynamics.md` §2 (the "一句话当前结论" bullet list)

> **Re-read spec §0 guardrails before writing this section.** §2.2 wording must say "机制层" not "cohort PASS"; §2.1 must enumerate the K=2 multi-mode minority across all three Tiers; §2 numbers must come from masked rerun (Step 5f).

- [ ] **Step 1: Replace the existing §2 block**

Find the old `## 2. 一句话当前结论` and the bullets under it (currently a single long list of `- **Paper 1 framework...**`, `- **传播刻板性**`, etc.). Replace with the spec §2 wording draft, verbatim:

```markdown
## 2. 当前最强结论

Topic 0 phantom-rank 修过后（2026-05-21 phase 0 broad re-derivation 完成，见
[docs/topic0_methodology_audits.md §3.1](topic0_methodology_audits.md) + step5a–h），
本 topic 在 masked 数据上有两条互锁的结论：

### 2.1 现象层：K=2 dominant stereotyped sequence（PR-2 / PR-2.5）

间期群体事件**普遍**存在 within-event 时序刻板性，最常见的压缩是 **k=2**（Tier 0
n=30：27/30 + 2×k=4 + 1×k=6；Tier 1 n=33：30/33 + 2×k=4 + 1×k=6；Tier 2 n=40：
35/40 + 2×k=4 + 2×k=5 + 1×k=6，含两个 4-ch path-D 极端 outlier）。模板在时间分块
复现上 **23/30 strong + 7/30 moderate + 0/30 weak**；split-half 中位模板相关 0.899，
odd/even block 0.985。masked rerun 不改变这条结论（簇内 raw τ +0.054，39/40 同向
p=1.27e-10；bias_fraction 87.9 → **92.2%**；详见
[step5c PR-3 results](archive/topic0/lagpat_phantom_rank/step5c_pr3_results_2026-05-20.md)）。

### 2.2 机制层：forward/reverse template swap geometry（PR-6 H2 + Step 6）

在 k=2 subject 中有一个稳定的 **机制层**（不是 cohort-level）现象：少数 subject 的
两个 template 之间，T_a 的 start-rank 通道在 T_b 里变成 end-rank 通道（"swap"）。
三个独立 masked 测量互相印证：

- **PR-6 H2 forward/reverse 子集 (n=8/8 on masked)** sign-test cleanly positive
  （fwd/rev 几何 swap；orig 9/9）
- **PR-6 Step 6 held-out swap_class concordance 0.69 → 0.82**（masked rerun
  实质提升，n=28 like-for-like）——同一通道 swap 类型在前半 / 后半时间分块上稳定
- **PR-6 supplementary rank displacement swap_class** strict 10→9 + candidate 8→6
  (masked) 维持相似分布；Kendall τ 与 F_norm 的强负相关 ρ = −0.92 不变

**重要：这是机制层证据，不是 cohort-level claim。** PR-6 plan
[`pr6_template_endpoint_anchoring_plan_2026-04-25.md`](archive/topic1/pr6_template_anchoring/pr6_template_endpoint_anchoring_plan_2026-04-25.md)
§3.3 把 H2 swap pre-register 为 "directional mechanism sanity, not cohort claim"。
PR-6 H1 (SOZ anchoring) cohort Wilcoxon **仍 NULL**（masked p=0.388 → 与 orig
一致）；node anatomy h1_eligible secondary Wilcoxon p=0.014 → 0.059
(**masked 进一步弱化**，方向保持)。所以 swap 是稳健的几何现象，但**不能**写成
"间期 HFO 模板锚定 SOZ"。

### 2.3 cohort-level NULL 列表（masked 重跑后仍 null）

- **PR-4** 慢调制三层全 cohort null（L1 模板混合 / L2 模板内顺序一致性 / L3 模板内
  相对时延结构），L3 高置信子集 (n=8) Pearson r p=0.016 → **0.547**（masked，
  exploratory tier 失效；不入主结论）
- **PR-4C** 模板内部几何无稳健发作邻近调制（封板阴性）
- **PR-5-B share** composition diagnostic 不复制 panel d（masked share post-base
  p=0.86，进一步弱化）
- **PR-6 H1** SOZ anchoring cohort 仍 NULL；H3 focus_rel i/l/e 仍 NULL
- **PR-7** mark dependence 三类 metric 全 NULL（compatible with mark-independent
  within tested precision，**不**等于证明独立）

唯一未在 cohort 维度 null 的：**PR-5-B 候选 A dominant template 绝对事件率
post_minus_baseline** main p=0.0004 (masked), median +65.66 ev/h Bonferroni-pass。

### 2.4 同步性线唯一 exploratory 信号

Epilepsiae 区域分层 `phase_e` pre>post p=0.012, r=0.31。其余全 null。

---
```

- [ ] **Step 2: Verify §2 wording satisfies guardrails (manual checklist)**

Re-read your §2.1 / §2.2 against the spec §0:

- [ ] §2.1 lists Tier 0 stable_k (27+2+1) AND Tier 1 (30+2+1) AND Tier 2 (35+2+2+1) — three Tiers, multi-mode minorities enumerated
- [ ] §2.2 says "**机制层**" (mechanism layer), not "cohort PASS" or "primary finding"
- [ ] §2.2 cites masked numbers: strict 10→9, candidate 8→6, concordance 0.69→0.82, fwd/rev 8/8 (was 9/9), H1 p=0.388 NULL, node anatomy 0.014→0.059
- [ ] §2.2 explicitly states "不能写成'间期 HFO 模板锚定 SOZ'" — preserves negative-result discipline
- [ ] §2.3 lists masked L3 p=0.547 (was 0.016) — flagged as exploratory tier loss
- [ ] §2.4 retains the single phase_e exploratory signal

If any item fails, fix inline before proceeding.

### Task 19: Rewrite §3 (the evidence chain in new order)

**Files:**
- Modify: `docs/topic1_within_event_dynamics.md` §3 (the "核心证据链" block)

- [ ] **Step 1: Plan the new §3 outline**

The new §3 has 4 subsections (per spec §2 TOC):

- §3.1 K=2 stereotyped sequence — merge old §3.1 + §3.1b + §3.1b.1 + §3.2 (identity bias)
- §3.2 forward/reverse swap geometry — promote old §7.10 + §7.10 末尾 Step 6/supp 表 + old archive §8 swap classifier
- §3.3 cluster geometry viz — old §3.1d, shortened to 1 paragraph + archive link
- §3.4 occupancy / synchrony — merge old §3.1c + §3.3 + §3.4, shortened

- [ ] **Step 2: Rewrite §3.1 (K=2 stereotypy)**

Replace old §3 / §3.1 / §3.1b / §3.1b.1 / §3.2 with:

```markdown
## 3. 证据骨架

> ### ⚠️ Topic 0 phantom-rank 重跑状态（2026-05-22 已完成）
>
> §3 所有 PR-2/2.5/3/4*/5/6/7 + Topic 4 attractor 的数字均经 Topic 0 §3.1
> phantom-rank 修过版 broad re-derivation 验证。**总判读**：所有 primary cohort
> verdict 方向保持；3 条加强（PR-3 bias_fraction 87.9→92.2%、PR-6 Step 6
> swap_class concordance 0.69→0.82、Topic 4 λ₂ 10→13/34）；1 条 exploratory
> tier loss（PR-4B L3 high-conf n=8 p=0.016→0.547）；4 条 secondary metric flip
> （PR-5 share+extended+transition + PR-6 node anatomy h1_eligible，全不进主
> Bonferroni 池）。**0 个 framework-level revision 触发**。完整 PR-by-PR impact
> 表 + step5a–h 链接见 [docs/topic0_methodology_audits.md §3.1 + §5](topic0_methodology_audits.md)。

> ### Cohort tier 注解（2026-05-07 起强制双轨/三层口径）
>
> 本 topic 共有 **三个 cohort 层**：
>
> | 层 | n | 谱系 | 适用范围 |
> |---|---|---|---|
> | **Tier 0** | 30 | 21 年 cusignal vintage 全链 | 历史 framework lock 时刻 |
> | **Tier 1 — primary** | 33 | 30 vintage + 3 lineage-adjacent | PR-1/PR-2/PR-2.5/PR-6 主表 |
> | **Tier 2 — extended** | 40 | 33 + 7 legacy variant | sensitivity / lineage robustness only |
>
> 写作合同：**Tier 1 > Tier 0 > Tier 2**。混合引用必须显式标注；不允许裸写 30/30
> 让读者猜。详见 [archive cohort slice 文档](archive/topic1/propagation/cohort_slice_a2_legacy_variant_2026-05-07.md)。

### 3.1 K=2 dominant stereotyped sequence

来自 `lagPatRank + eventsBool + chnNames` 的 cluster-aware 分析（Topic 0 phantom 修过版）。

**Tier 0 cohort 主表（SBA framework P1/P2 lock 时刻）**：

- `30/30` subject 的 pairwise Kendall `τ` 分布呈多模态
- KMeans(`k=2`) 后，簇内 `τ` 中位数 `0.250` 显著高于整体 `τ = 0.089`
- `29/30` subject within-cluster `τ > overall τ`
- `30/30` legacy MI permutation 显著，复现老论文结论
- `stable_k` 分布 `27 × k=2`、`2 × k=4`、`1 × k=6`（dominant compression at k=2，
  multi-mode minority **必须保留报告**）
- Adaptive within-cluster `τ` 中位数 `0.252`，相对整体 uplift 中位数 `+0.100`
- PR-2.5 时间切片复现：`23/30 strong`、`7/30 moderate`、`0 weak`；split-half 中位
  模板相关 `0.899`，odd/even block `0.985`
- `9` 个 k=2 subject 带 `candidate_forward_reverse` 对（inter-cluster `r < -0.5`），
  其中 `8/9` 跨时间切片可复现互逆关系

合理口径：**一个 subject 常常有多条主要传播路径，每条路径内部仍然刻板。**

**Tier 1 / Tier 2 当前 cohort 数字（2026-05-07 起 PR-1/PR-2/PR-2.5/PR-6 主表）**：

| 指标 | Tier 1 (n=33) | Tier 2 (n=40) |
|---|---|---|
| `n_strict_mixture` | 30 | 30（不变） |
| `n_possible_mixture` | 3 | 10 |
| `mean_tau_median` | 0.0884 | 0.0845 |
| `bias_fraction_median` | 0.6568 | 0.7110 |
| `stable_k_distribution` | `{2:30, 4:2, 6:1}` | `{2:35, 4:2, 5:2, 6:1}` |
| PR-2.5 grade | strong=26, moderate=7 | strong=31, moderate=9 |
| PR-2.5 forward_reverse `n_with_pairs / n_reproduced` | 14 / 13 | 17 / 16 |
| PR-6 H1 pooled wilcoxon_greater p | 0.388 (n=23 eligible) | 0.223 (n=28 eligible) |
| PR-6 H1 Yuquan-only p | 0.344 (n=10) | 0.107 (n=15) |

历史与详细数值：
- Slice A1: [`docs/archive/topic1/propagation/cohort_slice_a1_2026-05-06.md`](archive/topic1/propagation/cohort_slice_a1_2026-05-06.md)
- Slice A2: [`docs/archive/topic1/propagation/cohort_slice_a2_legacy_variant_2026-05-07.md`](archive/topic1/propagation/cohort_slice_a2_legacy_variant_2026-05-07.md)

**Identity bias（簇内 86%，masked rerun 加强到 92.2%）**：

| 层 | raw τ | centered τ | bias fraction |
|---|---|---|---|
| 整体 | 0.089 | 0.023 | 0.652 |
| 簇内 | 0.252 | ≈0.03 | **0.86（orig）→ 0.922（masked）** |

簇内 bias 意味着每个传播模式内部约 86% 通道排序一致性来自固有激活位置，仅 14%
是事件特异性传播动力学。**stereotypy 主要由结构性通道排序驱动**；masked rerun
进一步加强（92.2%，phantom 是噪声不是身份偏置）。masked rerun 细节：
[step5c PR-3 results](archive/topic0/lagpat_phantom_rank/step5c_pr3_results_2026-05-20.md)。

**Masked 主图**：[results/interictal_propagation_masked/figures/cohort_propagation_summary.png](../results/interictal_propagation_masked/figures/cohort_propagation_summary.png)
+ [per_subject/*_propagation.png](../results/interictal_propagation_masked/figures/per_subject/)。
旧版 phantom-contaminated 主图：[results/interictal_propagation/figures/cohort_propagation_summary.png](../results/interictal_propagation/figures/cohort_propagation_summary.png)
（**SUPERSEDED 2026-05-22**，详 README.md）。
```

- [ ] **Step 3: Rewrite §3.2 (forward/reverse swap geometry)**

Insert after §3.1:

```markdown
### 3.2 forward/reverse template swap geometry（PR-6 主线）

> 完整合同：[`pr6_template_endpoint_anchoring_plan_2026-04-25.md`](archive/topic1/pr6_template_anchoring/pr6_template_endpoint_anchoring_plan_2026-04-25.md)
> Pivot 来源（2026-04-25）：原 PR-6-A multi-anchor consensus / ictal-onset alignment
> 已冻结归档（sentinel `548/916` 证伪稳定 ictal anchor + Schroeder 2020 / Wenzel
> 2017 / Pinto 2023 / Bailey 2021 文献支持），新主线 = stable template endpoint
> (source ∪ sink) anatomical anchoring。详细 brainstorm:
> [`pr6_direction_brainstorm_2026-04-25.md`](archive/topic1/pr6_template_anchoring/pr6_direction_brainstorm_2026-04-25.md)。

Pivot 后四条互锁证据（按 advisor §0 tier 规则：H2 swap 是 **机制层 mechanism
sanity**，不是 cohort PASS）：

#### 3.2.1 PR-6 Step 4b node-level swap–same 几何

Audit-derived 主 cohort **n=21（13 epilepsiae + 8 yuquan，masked rerun n=21 持平）**。

| 检验 | masked 结果 | 旧版 (orig) |
|---|---|---|
| H1 endpoint vs middle pooled Wilcoxon p | 0.388 (NULL) | 0.42 (NULL，方向保持) |
| Step 4b swap−same paired Wilcoxon, h1_eligible | 0.012 → **0.059** (borderline，secondary metric 弱化) | 0.012 (was nominal) |
| H2 forward/reverse subset sign-test | **8/8 cleanly positive** (masked) | 9/9 (orig，1 例进 phantom NULL) |
| Step 5b endpoint Jaccard (split-half / odd-even) | 0.71 / 0.93 (持平) | 0.71 / 0.93 |
| Endpoint metric sensitivity (top-3 vs coreness 20%) | 7/20 direction-discordant + 1 one-is-zero | 同 |

**Cohort verdict**：单 metric H1 cannot determine whether stable templates anchor
clinical SOZ；**更稳健的现象 = pair-level swap geometry**，且 swap geometry **不能**由
subject-level SOZ enrichment 解释（详 §3.2.4）。

**Masked 主图**：
[`results/interictal_propagation_masked/template_anchoring/figures/pr6_template_pair_geometry_main.png`](../results/interictal_propagation_masked/template_anchoring/figures/pr6_template_pair_geometry_main.png)。

#### 3.2.2 PR-6 supplementary rank displacement + variable-k swap classifier

| 指标 | masked | orig |
|---|---|---|
| Cohort F_norm median | 0.789 | 0.800 |
| Cohort Kendall τ median | −0.24 | −0.20 |
| Spearman ρ(F_norm, τ) | **−0.921** | −0.916 |
| Variable-k swap_class strict (T_obs ≥ 0.5, p_fw < 0.05) | **9** / 28 | 10 / 35 |
| Variable-k swap_class candidate (0.05 ≤ p < 0.20) | **6** / 28 | 8 / 35 |
| Variable-k swap_class others | 19 / 28 | 17 / 35 |

ρ = −0.92 的强负相关表示 F_norm 与 τ 共线（两量都派生自同一组 displacement 距离差，
是 sanity / consistency check，非独立 finding）。strict 子集 10/9 主要落在 PR-2.5
fwd/rev-reproduced 集合。详:
- [`pr6_supplementary_rank_displacement_results_2026-05-06.md`](archive/topic1/propagation/pr6_supplementary_rank_displacement_results_2026-05-06.md)
  §8（variable-k dual-tier 详细数）
- [`step5f_pr6_results_2026-05-21.md`](archive/topic0/lagpat_phantom_rank/step5f_pr6_results_2026-05-21.md)
  （masked rerun 主数）

**Masked 主图**：[`results/interictal_propagation_masked/rank_displacement/figures/cohort_displacement_heatmap.png`](../results/interictal_propagation_masked/rank_displacement/figures/cohort_displacement_heatmap.png)
+ [`pr6_supp_swap_cluster_rank_multiples.png`](../results/interictal_propagation_masked/template_anchoring/figures/pr6_supp_swap_cluster_rank_multiples.png)。

#### 3.2.3 PR-6 Step 6 held-out time validation

Tier 分布 strong/moderate/weak/fail：

| 指标 | masked (n=28) | orig (n=35) |
|---|---|---|
| strong fraction | **17/28 = 61%** | 20/35 = 57% |
| template_spearman median | 0.88 | 0.92 |
| endpoint_position_recall median | **0.83**（完全相同） | 0.83 |
| **swap_class_concordant fraction** | **0.82**（实质提升） | 0.69 |

**Step 6 swap_class concordance 0.82 是 masked 数据上"held-out swap geometry 跨时间
稳定"的最相关 metric**。详:
[`pr6_step6_held_out_template_results_2026-05-10.md`](archive/topic1/propagation/pr6_step6_held_out_template_results_2026-05-10.md)
+ [`step5f_pr6_results_2026-05-21.md`](archive/topic0/lagpat_phantom_rank/step5f_pr6_results_2026-05-21.md)。

#### 3.2.4 swap × clinical SOZ set-relationship（旧 supp §9）

Primary strict ∩ informative n=5: sign test p=0.500 (masked sign p=0.66 → 仍 NULL),
median enrichment +0.042, bootstrap CI [−0.071, +0.098]。Sensitivity candidate ∩
informative n=5: 4/5 positive, median +0.127。Typology: `E⊊S`=0; `S⊊E`=4; 多数
`partial`。**方向上 swap 倾向"覆盖且包含"clinical SOZ，但 strict tier cohort 仍 NULL。**

Channel-selection circular caveat：lagPat 已对 SOZ 富集，量化的是 high-HI 区内 swap
是否进一步富集 SOZ，不是全脑关系。

详:
[`pr6_supplementary_swap_clinical_soz_plan_2026-05-08.md`](archive/topic1/propagation/pr6_supplementary_swap_clinical_soz_plan_2026-05-08.md)
+ [`pr6_supplementary_rank_displacement_results_2026-05-06.md`](archive/topic1/propagation/pr6_supplementary_rank_displacement_results_2026-05-06.md)
§9。
```

- [ ] **Step 4: Rewrite §3.3 (cluster geometry viz, shortened)**

Insert after §3.2:

```markdown
### 3.3 Cluster geometry 可视化（trilateration + bimodality audit）

PR-2 / PR-2.5 cluster decomposition 的描述性补强（2026-05-06）。20 Epilepsiae subjects
入选（18 Yuquan 因 data freshness 问题暂排）。

- silhouette median = **0.460**（range 0.182–0.671，5/20 < 0.3）
- KMeans-vs-template-matching agreement median = **0.892**（range 0.769–0.955，8/20 < 0.85）
- Marginal-x bimodality: **11/20 BIMODAL / 8/20 AMBIGUOUS / 1/20 UNIMODAL**（`442` 是
  UNIMODAL counter-example，**必须报告**）
- 不推翻 PR-2/2.5/3/4/5/6/7 任何主结论；纯描述层；不进 SBA framework P1–P5

详细数字、bimodality 检验机制、failure 合同、follow-up：
- Plan: [`cluster_geometry_viz_plan_2026-05-06.md`](archive/topic1/propagation/cluster_geometry_viz_plan_2026-05-06.md)
- Results: [`cluster_geometry_viz_results_2026-05-06.md`](archive/topic1/propagation/cluster_geometry_viz_results_2026-05-06.md)

> **桥到 Topic 4 (attractor diagnostics)**：在 §3.3 cluster geometry 之上做的
> attractor-class 诊断（principal curve + GOF + KMeans-axis 夹角 + coordinate-free
> PR-2 label λ₂）已完成 Step 0 + Step 1 + sensitivity。35 例 stable_k=2 cohort，
> 34 进 H3 主分析。**未进主结论**（Step 2 Λ_gap 未跑）。masked rerun：λ₂ 10/34 →
> 13/34（实质提升）。详:
> [`topic4_attractor_diagnostics_step1_results_2026-05-10.md`](archive/topic1/propagation/topic4_attractor_diagnostics_step1_results_2026-05-10.md)
> + [`step5h_topic4_attractor_results_2026-05-21.md`](archive/topic0/lagpat_phantom_rank/step5h_topic4_attractor_results_2026-05-21.md)。
```

- [ ] **Step 5: Rewrite §3.4 (occupancy + synchrony, descriptive layer)**

Insert after §3.3:

```markdown
### 3.4 描述层：occupancy 漂移 + event-level synchrony

#### 3.4.1 PR-3 / PR-4A：固定模板可视化与 occupancy 漂移（**Tier 0**，未重算）

- PR-3 论文级 6-panel cohort 图已固定（n=30 framework lock 版）；簇内 identity-bias
  median = 86%（masked rerun 92.2%）
- PR-4A day/night occupancy timeline: dominant fraction Wilcoxon `p=0.124`、entropy
  `p=0.245`、TV distance median `0.019`（n=30）→ **占比的昼夜漂移整体较弱**
- 模板投射 agreement 中位数 `0.888`；只有 `3/30`（`chengshuai`、`253`、`818`）
  低于 `0.8`
- 描述层结果，**不是强机制结论**

PR-4D `rate×type` decomposition 是 PR-4A 的补强（不再"平滑 occupancy"，改用固定模板
分解后绝对事件率 envelope + 计数堆叠柱）。Cohort dominant template rate fraction
中位数 `0.584`（range `0.262–0.866`）；`25/30` subject 至少出现一次主导模板交叉。

#### 3.4.2 Event-level synchrony cohort 统计

PR4–PR6 以 seizure interval 为统计单位，主指标是 `phase`：

- `phase_all` post vs pre：`p = 0.279`
- `phase_core` post vs pre：`p = 0.967`
- within-interval trajectory：`phase_all p = 0.589`，`phase_core p = 0.643`
- event rate：`p = 0.361`

cohort level **不**支持"发作后去同步重置"或"发作前同步性恢复"。

#### 3.4.3 Topic 1 唯一值得继续追的 synchrony 信号

Epilepsiae 区域分层 `phase_e`：`p = 0.012`，`r = 0.31`，方向 `pre > post`，Bonferroni
校正后仍勉强保留。**`phase_i p=0.646`、`phase_l p=0.543` 全 null**。

这是 synchrony 线中唯一可称为 exploratory-significant 的结果。

---
```

- [ ] **Step 6: Verify §3 structure**

```bash
grep -nE "^### 3\.[0-9]" docs/topic1_within_event_dynamics.md
```
Expected: `3.1`, `3.2`, `3.2.1`, `3.2.2`, `3.2.3`, `3.2.4`, `3.3`, `3.4`, `3.4.1`, `3.4.2`, `3.4.3` headings present.

### Task 20: Write the new §4 PR-by-PR status table

**Files:**
- Modify: `docs/topic1_within_event_dynamics.md` (insert after §3.4)

- [ ] **Step 1: Write §4 PR-by-PR table**

Insert after §3.4:

```markdown
## 4. PR-by-PR status 表

主文档不再重述每个 PR 的合同 / 阈值 / metric 公式；每条 PR 1 段当前判定 + masked
rerun verdict + archive 链接。

| PR | tier | 当前判定（masked） | 关键数 | archive 入口 |
|---|---|---|---|---|
| **PR-2 / PR-2.5** | primary | **ACCEPTED**：K=2 dominant stereotypy stays；split-half 23/30 strong；PR-2.5 fwd/rev `8/9` reproduced | bias_fraction 87.9→92.2% (加强) | [`interictal_group_event_internal_propagation.md`](archive/topic1/propagation/interictal_group_event_internal_propagation.md) |
| **PR-3** | primary | **ACCEPTED**：6-panel cohort fig 已重画 (n=33 default) | cohort_propagation_summary.png masked 已生成 | 同上 |
| **PR-4A / PR-4D** | descriptive | **ACCEPTED**：occupancy / `rate×type` 描述层 | dominant rate fraction median 0.584 | 同上 + ".cursor/rules/topic1-within-event-dynamics.mdc" L1/L2/L3 guardrails |
| **PR-4B** | NULL（cohort）| **NULL stays**：L1/L2/L3 cohort null；L3 high-conf 子集 (n=8) Pearson r p=0.016→0.547 masked exploratory tier loss | n=8 Wilcoxon 最小可能 p=0.004 (power floor) | 同上 |
| **PR-4C** | NULL（cohort，封板）| **CLOSED 2026-04-19**：propagation 五指标 cohort null 封板；唯一稳健信号 `rate_by_template` 已 升级为 PR-5 | aux 1/15 + main 1/15 (跨配置不一致) | [`pr4c_seizure_proximity_review_2026-04-17.md`](archive/topic1/propagation/pr4c_seizure_proximity_review_2026-04-17.md) §9 |
| **PR-5-A / PR-5-B** | primary | **ACCEPTED 2026-04-20 / STRONG (masked verified)**：dominant template 绝对事件率 post_minus_baseline median +65.66 ev/h (masked main p=0.0004 Bonferroni-pass)；§4.5 share composition diagnostic masked 进一步弱化 (p=0.86 NULL) | retained masked main n=27 / aux n=26 | [`pr5_template_recruitment_plan_2026-04-20.md`](archive/topic1/pr5_template_recruitment/pr5_template_recruitment_plan_2026-04-20.md) §11 + [`step5e_pr5_results_2026-05-21.md`](archive/topic0/lagpat_phantom_rank/step5e_pr5_results_2026-05-21.md) |
| **PR-6 (endpoint anchoring)** | mechanism sanity | **H1 NULL stays + H2 mechanism cleanly positive**：详 §3.2 | swap_class concordance 0.69→0.82, fwd/rev 8/8 | [`pr6_template_endpoint_anchoring_plan_2026-04-25.md`](archive/topic1/pr6_template_anchoring/pr6_template_endpoint_anchoring_plan_2026-04-25.md) + [`step5f_pr6_results_2026-05-21.md`](archive/topic0/lagpat_phantom_rank/step5f_pr6_results_2026-05-21.md) |
| **PR-7 (template pairing)** | NULL（已测时间尺度）| **NULL stays**：mark dependence 三类 metric 全 cohort null；compatible with mark-independent within tested precision；P3 cohort-level equivalence INCONCLUSIVE-locked | masked h1_primary n=8 Wilcoxon p=1.000 / sign p=1.000 / median(10s)=−0.037 | [`pr7_template_pairing_results_2026-04-29.md`](archive/topic1/pr7_template_pairing/pr7_template_pairing_results_2026-04-29.md) §17 + [`pr7_addendum_p3_equivalence_2026-05-01.md`](archive/topic1/pr7_template_pairing/pr7_addendum_p3_equivalence_2026-05-01.md) + [`step5g_pr7_results_2026-05-21.md`](archive/topic0/lagpat_phantom_rank/step5g_pr7_results_2026-05-21.md) |

**模型层（未来 PR-T4-1 / PR-T4-2 / PR-9 等）状态**：硬启动条件未达成；维持冻结；
不绑 PR 编号。详 §6 推荐下一步。

---
```

### Task 21: Rewrite §5 (risks) / §6 (next steps) / §7 (code entry) / §8 (cross-topic) / §9 (history index)

**Files:**
- Modify: `docs/topic1_within_event_dynamics.md` (replace remaining old §4/§5/§6/§7/§8/§9/§10/§11)

- [ ] **Step 1: Rewrite §5 / §6 / §7 / §8 / §9**

Replace the old §4 / §5 / §6 / §7 / §8 / §9 / §10 / §11 with:

```markdown
## 5. 已知风险与未解问题

- SOZ > non-SOZ 的传播优势仍偏弱，更像探索性趋势；PR-6 H1 cohort masked 仍 NULL
- centered rank 可能过度校正；`soz_source_erased` 仅 `3/30`，必须和 raw 结果并列
- PR-4A occupancy 已给出 day/night 描述性答案，但不能升级为正式发作邻近主结论
- `candidate_forward_reverse` 仅是描述标签（`inter-cluster Spearman r < -0.5`），
  不是生理机制
- 高 k subject 中 `818` 与 `zhangjinhan` 仍需 `n_participating` 匹配子样本验证
- 固定模板投射 agreement 整体够高，但 `chengshuai`、`253`、`818` 这 3 个 subject
  解释时间轨迹细节时需谨慎
- synchrony 线最大风险是"把 null 写得太花"；最诚实说法是 **总体 null，局部
  extra-focal `phase_e` 线索待验证**
- propagation 与 synchrony 是不同统计对象，文档里必须并列而非混写
- **PR-6 H2 swap geometry 是 mechanism sanity tier**：plan §3.3 pre-registered，
  禁止升级为 cohort-level claim 或 "independent publishable finding"
- **PR-7 NULL 不等于证明 mark independence**：未测 alternative burst definitions、
  rate-state / seizure-proximity switching、form (4) latent-state coupling、
  history-dependent regression。措辞限定 "compatible with mark-independent within
  tested precision"，禁止写 "PASS"
- **PR-4B / PR-5 share / PR-6 node anatomy**：phantom 修过版 4 条 secondary metric
  flip（详 §2.3 + Topic 0 §3.1）；这些 metric 已退出主 Bonferroni 池
- **`epilepsiae/1096` pre-existing contamination**：PR-6 H1 pooled `valid_mask_source=fallback_all_valid`
  inherits 进 Tier 1/2 双轨 H1 p；要 fix 需单独 sensitivity PR

---

## 6. 推荐的下一步

1. **PR-6 Step 6 follow-up** (mechanism sanity tier): held-out endpoint vs per-seizure
   onset channel set Jaccard（subject-level Wilcoxon），用真实 onset 通道而非 SOZ JSON 二值化
2. **fwd/rev subject 的 ictal propagation 直接对比**：reuse Topic 3 PR-2.5 onset
   estimation，把 T0 source / T1 sink 与真实 ictal LFP 传播方向比对
3. **PR-4B HC subset (n=8) × PR-6 endpoint 联动**：高放电期 dominant cluster 的 endpoint
   是否更靠 SOZ（descriptive，不开 cohort gate）
4. **pre-ictal vs baseline endpoint anatomy**：reuse PR-2.7 seizure-triggered window，
   看 endpoint 集合是否偏移
5. **未来模型层 (PR-T4-1 BHPN-toy / PR-T4-2 BHPN-fit / SEF-ITP Phase 1)**：硬前置见
   `docs/paper1_framework_sba.md` + `docs/topic4_sef_itp_framework.md`；Phase 0 已解锁
   (2026-05-21)，Phase 1 可启动，但需先完成 `load_subject_for_phase1()` integration PR

**Out of scope（保持冻结）**：在当前数据上重新调 endpoint 定义；重做 PR-2/3/4/4B
核心 cluster pipeline；引入 ictal anchor / ER / CUSUM；做"先挑 hub 再重跑 PR-4C"
的 double-dipping replay。

---

## 7. 代码与结果入口

### 内部传播

- 文档：[`docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md`](archive/topic1/propagation/interictal_group_event_internal_propagation.md)
- 代码：`src/interictal_propagation.py`、`src/lagpat_rank_audit.py`
- 脚本（**全部支持 `--masked-features` flag**）：
  - `scripts/run_interictal_propagation.py`、`scripts/run_pr6_template_anchoring.py`、
    `scripts/run_pr6_step6.py`、`scripts/run_rank_displacement.py`、
    `scripts/run_pr7_template_pairing.py`、`scripts/pr7_addendum_p3_equivalence.py`、
    `scripts/run_pr5b_share_extended.py`、`scripts/run_pr5_transition_windows.py`
  - 绘图：`scripts/plot_interictal_propagation.py`、`scripts/plot_pr6_template_anchoring.py`、
    `scripts/plot_pr6_swap_cluster_rank_multiples.py`、`scripts/plot_rank_displacement.py`、
    `scripts/plot_pr6_step6.py`、`scripts/plot_pr6_sup1_rank_entropy.py`、
    `scripts/plot_topic1_pr4_ppt.py`、`scripts/plot_pr7_template_pairing.py`、
    `scripts/plot_template_share_switching.py`
- 结果（**phantom-fixed 主路径**）：`results/interictal_propagation_masked/`
- 结果（**phantom-contaminated 历史路径**）：`results/interictal_propagation/`
  （README.md 已标 SUPERSEDED，保留作为 archive evidence）

### 事件级同步性

- 文档：[`docs/archive/topic1/synchrony/interictal_synchrony_preliminary_report_2026-04-03.md`](archive/topic1/synchrony/interictal_synchrony_preliminary_report_2026-04-03.md)
- 代码：`src/interictal_synchrony.py`、`src/interictal_synchrony_aggregation.py`、
  `src/interictal_synchrony_analysis.py`
- 脚本：`scripts/pr6_interictal_sync_figures.py`
- 结果：`results/interictal_synchrony/analysis/combined/`

---

## 8. 与其他 topic 的边界

- "`~2 Hz` 峰是不是真的"或"IEI serial correlation 说明什么" → [`docs/topic2_between_event_dynamics.md`](topic2_between_event_dynamics.md)
- "SOZ 和 non-SOZ 到底差在哪里" → [`docs/topic3_spatial_soz_modulation.md`](topic3_spatial_soz_modulation.md)
- 同时涉及"传播是否真实"和"慢调制是否发生在 SOZ" → 先分别读 topic 1 和 topic 3，不要混成一个问题
- "ictal subtyping / Topic 5 cross-link" → [`docs/topic5_seizure_subtyping.md`](topic5_seizure_subtyping.md)

---

## 9. 历史文档索引

> 仅保留**正式 archive 入口**。SUPERSEDED 链接（PR-6-A 三份 doc 等）不再列在主 doc，
> 协作者按 archive 内部 SUPERSEDED 块查找。

### Framework 层

- [`docs/paper1_framework_sba.md`](paper1_framework_sba.md) — Paper 1 架构性 framework
  （SBA 单核心假设 + P1–P5 + BHPN-toy/fit + 5 dumb baseline + 失败模式）
- [`docs/topic4_sef_itp_framework.md`](topic4_sef_itp_framework.md) — Topic 4 模型层
  framework（SEF-ITP H1–H6 + Phase 0–4 路线）
- [`docs/archive/topic4/pr_t4_1_bhpn_toy/pr_t4_1_bhpn_toy_plan_2026-05-01.md`](archive/topic4/pr_t4_1_bhpn_toy/pr_t4_1_bhpn_toy_plan_2026-05-01.md)
  — PR-T4-1 BHPN-toy plan-of-record

### Topic 0 phantom rerun (2026-05-21 phase 0 complete)

- [`docs/topic0_methodology_audits.md`](topic0_methodology_audits.md) §3.1 + §5 —
  cohort-level audit + step5a–h 索引
- step5a (PR-2) / step5b (PR-2.5) / step5c (PR-3) / step5d.3 (PR-4C) / step5e (PR-5)
  / step5f (PR-6) / step5g (PR-7) / step5h (Topic 4 attractor) — 见
  [`docs/archive/topic0/lagpat_phantom_rank/`](archive/topic0/lagpat_phantom_rank/)
- [`checkpoint_b_report_2026-05-21.md`](archive/topic0/lagpat_phantom_rank/checkpoint_b_report_2026-05-21.md)
  — phase 0 advisor consult summary

### PR-level archive

- [`interictal_group_event_internal_propagation.md`](archive/topic1/propagation/interictal_group_event_internal_propagation.md)
  — PR-2/2.5/3/4*/5 详细数 + 合同
- [`pr4c_seizure_proximity_review_2026-04-17.md`](archive/topic1/propagation/pr4c_seizure_proximity_review_2026-04-17.md)
  — PR-4C 主+辅助配置全量审阅 + §9 封板
- [`pr5_template_recruitment_plan_2026-04-20.md`](archive/topic1/pr5_template_recruitment/pr5_template_recruitment_plan_2026-04-20.md)
  — PR-5 完整计划 + §11 复跑结论
- [`pr6_template_endpoint_anchoring_plan_2026-04-25.md`](archive/topic1/pr6_template_anchoring/pr6_template_endpoint_anchoring_plan_2026-04-25.md)
  — PR-6 正式入口 (H1/H1b/H2/H3 + audit-derived cohort + 8 TDD)
- [`pr6_direction_brainstorm_2026-04-25.md`](archive/topic1/pr6_template_anchoring/pr6_direction_brainstorm_2026-04-25.md)
  — PR-6 pivot 决策 brainstorm
- [`pr6_step6_held_out_template_results_2026-05-10.md`](archive/topic1/pr6_step6_held_out_template_results_2026-05-10.md)
  — PR-6 Step 6 held-out time validation
- [`pr6_supplementary_rank_displacement_plan_2026-05-06.md`](archive/topic1/pr6_supplementary_rank_displacement_plan_2026-05-06.md)
  + [`pr6_supplementary_rank_displacement_results_2026-05-06.md`](archive/topic1/pr6_supplementary_rank_displacement_results_2026-05-06.md)
  — PR-6 supp 连续 rank displacement + §8 variable-k swap §9 SOZ set-relationship
- [`pr6_supplementary_swap_clinical_soz_plan_2026-05-08.md`](archive/topic1/pr6_supplementary_swap_clinical_soz_plan_2026-05-08.md)
  — §9 plan
- [`pr6_supplementary_rank_entropy_plan_2026-05-10.md`](archive/topic1/pr6_supplementary_rank_entropy_plan_2026-05-10.md)
  + [`pr6_supplementary_rank_entropy_results_2026-05-10.md`](archive/topic1/pr6_supplementary_rank_entropy_results_2026-05-10.md)
  — PR-6 supp1 first-rank entropy (Topic 4 preflight)
- [`pr7_template_antagonistic_pairing_plan_2026-04-28.md`](archive/topic1/pr7_template_pairing/pr7_template_antagonistic_pairing_plan_2026-04-28.md)
  + [`pr7_template_pairing_results_2026-04-29.md`](archive/topic1/pr7_template_pairing/pr7_template_pairing_results_2026-04-29.md)
  + [`pr7_addendum_p3_equivalence_2026-05-01.md`](archive/topic1/pr7_template_pairing/pr7_addendum_p3_equivalence_2026-05-01.md)
  — PR-7 plan + results + addendum
- [`ping_pong_hypothesis_review_2026-04-28.md`](archive/topic1/ping_pong_hypothesis_review_2026-04-28.md)
  — Ping-Pong 假说三层分离 + PR roadmap
- [`cluster_geometry_viz_plan_2026-05-06.md`](archive/topic1/propagation/cluster_geometry_viz_plan_2026-05-06.md)
  + [`cluster_geometry_viz_results_2026-05-06.md`](archive/topic1/propagation/cluster_geometry_viz_results_2026-05-06.md)
  — §3.3 cluster geometry viz
- [`topic4_attractor_diagnostics_step1_results_2026-05-10.md`](archive/topic1/propagation/topic4_attractor_diagnostics_step1_results_2026-05-10.md)
  — Topic 1 → Topic 4 attractor 桥
- [`midterm_ppt_outline_2026-05-10.md`](archive/topic1/midterm_ppt_outline_2026-05-10.md)
  — 中期组会 PPT 大纲
- Cohort slice / synchrony / topic 5 bridge / cross-link 等 — 见 [`docs/archive/topic1/`](archive/topic1/)
  目录

### 文档整理里程碑

- **2026-04-20**：主文档大幅瘦身（442 行 → ~270 行），按"主文档只放正式口径，过程性
  细节回链 archive"原则重写。
- **2026-05-22**：主文档重排（~526 行 → ~300 行）。把 K=2 stability + fwd/rev swap
  geometry 升到 §2 第一序结论；PR-6 swap supersedes PR-2/2.5 在 swap geometry 上的
  描述深度；topic0 phantom rerun masked figures 已落地 `results/interictal_propagation_masked/`
  顶层（D1 git mv）；旧 phantom-contaminated `results/interictal_propagation/figures/`
  保留作为 archive evidence，README 标 SUPERSEDED。重排原则：surface conclusions
  + honor pre-registered tier discipline (CLAUDE.md §5)。
```

- [ ] **Step 2: Verify total doc structure**

```bash
grep -nE "^## [0-9]\. " docs/topic1_within_event_dynamics.md
```
Expected: §0 §1 §2 §3 §4 §5 §6 §7 §8 §9 — 10 top-level sections.

```bash
wc -l docs/topic1_within_event_dynamics.md
```
Expected: ~280–320 lines (down from 526).

### Task 22: Cross-reference sanity check

**Files:** none modified; pure validation.

- [ ] **Step 1: grep for accidentally-orphaned old anchors**

```bash
grep -rnE "topic1_within_event_dynamics\.md#(7\.10|7\.11|3\.1c|3\.1d|3\.1b\.1)" \
  docs/ AGENTS.md CLAUDE.md 2>/dev/null
```
Expected: 0 lines (or list of archive docs that reference old anchors — these are OK,
the §0 migration table in the new main doc covers the mapping).

- [ ] **Step 2: Verify all archive paths cited in new §3 / §9 actually exist**

```bash
for path in \
  docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md \
  docs/archive/topic1/pr6_template_anchoring/pr6_template_endpoint_anchoring_plan_2026-04-25.md \
  docs/archive/topic1/pr6_template_anchoring/pr6_direction_brainstorm_2026-04-25.md \
  docs/archive/topic1/propagation/pr6_supplementary_rank_displacement_results_2026-05-06.md \
  docs/archive/topic1/propagation/pr6_step6_held_out_template_results_2026-05-10.md \
  docs/archive/topic1/pr7_template_pairing/pr7_template_pairing_results_2026-04-29.md \
  docs/archive/topic1/pr7_template_pairing/pr7_addendum_p3_equivalence_2026-05-01.md \
  docs/archive/topic0/lagpat_phantom_rank/step5c_pr3_results_2026-05-20.md \
  docs/archive/topic0/lagpat_phantom_rank/step5e_pr5_results_2026-05-21.md \
  docs/archive/topic0/lagpat_phantom_rank/step5f_pr6_results_2026-05-21.md \
  docs/archive/topic0/lagpat_phantom_rank/step5g_pr7_results_2026-05-21.md \
  docs/archive/topic0/lagpat_phantom_rank/step5h_topic4_attractor_results_2026-05-21.md \
  docs/archive/topic0/lagpat_phantom_rank/checkpoint_b_report_2026-05-21.md ; do
  test -f "$path" && echo "OK: $path" || echo "MISSING: $path"
done
```
Expected: all `OK:`. If any `MISSING:`, fix the path in the doc or note the actual path.

- [ ] **Step 3: Confirm new figure paths exist**

```bash
for path in \
  results/interictal_propagation_masked/figures/cohort_propagation_summary.png \
  results/interictal_propagation_masked/template_anchoring/figures/pr6_template_pair_geometry_main.png \
  results/interictal_propagation_masked/template_anchoring/figures/pr6_supp_swap_cluster_rank_multiples.png \
  results/interictal_propagation_masked/rank_displacement/figures/cohort_displacement_heatmap.png ; do
  test -f "$path" && echo "OK: $path" || echo "MISSING: $path"
done
```
Expected: all `OK:` (these were generated in Phase 2 Tasks 11–14).

### Task 23: Mark old phantom-contaminated figures as SUPERSEDED

**Files:**
- Modify: `results/interictal_propagation/figures/README.md`

- [ ] **Step 1: Read current README.md**

```bash
sed -n '1,15p' results/interictal_propagation/figures/README.md
```

- [ ] **Step 2: Prepend SUPERSEDED block**

Use Edit tool to add the following at the very top of `results/interictal_propagation/figures/README.md`:

```markdown
> ## ⚠️ SUPERSEDED 2026-05-22
>
> These figures were generated on the **phantom-contaminated** `lagPatRank` feature
> tree (legacy `argsort(argsort(x))` returns finite int ranks for non-participating
> channels — see [`docs/topic0_methodology_audits.md §3.1`](../../../docs/topic0_methodology_audits.md)).
>
> **Canonical replacement** (masked / phantom-fixed):
> [`results/interictal_propagation_masked/figures/`](../../interictal_propagation_masked/figures/)
> + [`results/interictal_propagation_masked/template_anchoring/figures/`](../../interictal_propagation_masked/template_anchoring/figures/)
> + [`results/interictal_propagation_masked/rank_displacement/figures/`](../../interictal_propagation_masked/rank_displacement/figures/)
>
> This directory is **retained as archive evidence** for the phantom-fix audit
> (per AGENTS.md "parallel directory" convention) — do NOT cite these figures
> in new paper text. Cite the masked path instead.
>
> ---
>
```

(Leave the rest of the existing README content intact below the SUPERSEDED block.)

- [ ] **Step 3: Verify**

```bash
head -20 results/interictal_propagation/figures/README.md
```
Expected: SUPERSEDED block at top.

### Task 24: Commit Phase 3 — D3 doc rewrite + SUPERSEDED note

- [ ] **Step 1: Stage and commit**

```bash
git add docs/topic1_within_event_dynamics.md results/interictal_propagation/figures/README.md
git status --short docs/ results/
git commit -m "$(cat <<'EOF'
docs(topic1): restructure to surface K=2 + fwd/rev swap as first-order conclusions

Full TOC rewrite per spec docs/superpowers/specs/2026-05-22-topic1-restructure-masked-replot-design.md.

§2 new four-section first-order conclusion block:
- 2.1 K=2 dominant stereotyped sequence (PR-2/2.5) — masked rerun strengthens
  bias_fraction 87.9→92.2%, dominant compression at k=2 enumerated across
  Tier 0/1/2 (multi-mode minority preserved per CLAUDE.md §8 plain-language rule)
- 2.2 forward/reverse template swap geometry (PR-6 H2 mechanism sanity, NOT
  cohort PASS — honors pre-registered tier per CLAUDE.md §5 + advisor guardrail).
  Three masked measurements: fwd/rev 8/8 sign-test, Step 6 swap_class concordance
  0.69→0.82, rank displacement strict 9 + candidate 6 (was 10 + 8)
- 2.3 cohort-level NULL list (PR-4 / PR-4C / PR-5-B share / PR-6 H1 / PR-7)
- 2.4 phase_e exploratory signal

§3 evidence chain: §3.1 K=2 stability (merged old §3.1+3.1b+3.1b.1+3.2),
§3.2 PR-6 swap geometry (promoted from old §7.10 with masked numbers),
§3.3 cluster geometry viz (shortened), §3.4 occupancy + synchrony descriptive.

§4 PR-by-PR status table (1 row per PR, archive links).
§5–§9 risks / next steps / code entry / cross-topic / history index — shortened.

§0 §-anchor migration table at top maps old §X → new §Y for downstream callers.

Old phantom-contaminated figures at results/interictal_propagation/figures/
marked SUPERSEDED in README; canonical masked figures at
results/interictal_propagation_masked/figures/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4 — D2 Batch 2: regenerate secondary figures (parallel with / after D3)

> Optional but recommended for completeness. None of these figures are referenced
> as primary evidence in the new §2 / §3; they are descriptive-layer / PPT panels
> that support narrative depth without unblocking D3.

### Task 25: Regenerate PR-1 heatmap + per_subject_mi + pr4a daynight on masked data

**Files:**
- Produced: `results/interictal_propagation_masked/figures/pr1_propagation_heatmap_examples.png`,
  `results/interictal_propagation_masked/figures/pr4a_daynight_group_analysis.png`,
  `results/interictal_propagation_masked/figures/per_subject_mi/`

- [ ] **Step 1: Inspect plot_interictal_propagation argparse for relevant subcommands**

```bash
python scripts/plot_interictal_propagation.py --help 2>&1 | tail -20
```
Identify the subcommands that produce heatmap_examples / per_subject_mi / daynight (likely `--heatmap-examples`, `--per-subject-mi`, `--pr4a` or similar; if no flags, defaults run them).

- [ ] **Step 2: Run with masked flag**

If the script has subcommand flags:
```bash
python scripts/plot_interictal_propagation.py --masked-features \
  --heatmap-examples --per-subject-mi --pr4a 2>&1 | tail -10
```

If it just runs everything:
```bash
python scripts/plot_interictal_propagation.py --masked-features --all 2>&1 | tail -10
```

- [ ] **Step 3: Verify**

```bash
ls results/interictal_propagation_masked/figures/pr1_propagation_heatmap_examples.png \
   results/interictal_propagation_masked/figures/pr4a_daynight_group_analysis.png \
   results/interictal_propagation_masked/figures/per_subject_mi/ 2>&1 | head -10
```

### Task 26: Regenerate PR-6 supplementary figures (coreness / jaccard / fwdrev) on masked

**Files:**
- Produced: `results/interictal_propagation_masked/template_anchoring/figures/pr6_supp_coreness_sensitivity.png`,
  `pr6_supp_endpoint_jaccard_per_subject.png`, `pr6_supp_fwdrev_small_multiples.png`

- [ ] **Step 1: Run with --all flag**

```bash
python scripts/plot_pr6_template_anchoring.py --masked-features --all 2>&1 | tail -10
```

- [ ] **Step 2: Verify**

```bash
ls results/interictal_propagation_masked/template_anchoring/figures/ | head -20
```

### Task 27: Regenerate PR-6 step6 figures on masked data

**Files:**
- Produced: `results/interictal_propagation_masked/pr6_step6_held_out_template/figures/*`

- [ ] **Step 1: Run with masked flag**

```bash
python scripts/plot_pr6_step6.py --masked-features 2>&1 | tail -10
```

- [ ] **Step 2: Verify**

```bash
ls results/interictal_propagation_masked/pr6_step6_held_out_template/figures/ 2>&1 | head -10
```

### Task 28: Regenerate PR-6 supplementary rank entropy + PR-4 PPT on masked

**Files:**
- Produced: `results/interictal_propagation_masked/pr6_sup1_rank_entropy/figures/*`,
  `results/interictal_propagation_masked/figures/ppt/*`

- [ ] **Step 1: Run rank entropy**

```bash
python scripts/plot_pr6_sup1_rank_entropy.py --masked-features 2>&1 | tail -10
```

- [ ] **Step 2: Run PR-4 PPT**

```bash
python scripts/plot_topic1_pr4_ppt.py --masked-features 2>&1 | tail -10
```

- [ ] **Step 3: Verify both**

```bash
ls results/interictal_propagation_masked/pr6_sup1_rank_entropy/figures/ 2>&1 | head -5
ls results/interictal_propagation_masked/figures/ppt/ 2>&1 | head -5
```

### Task 29: Commit Batch 2 figures + final close-out

- [ ] **Step 1: Stage and commit**

```bash
git add results/interictal_propagation_masked/
git status --short results/
git commit -m "$(cat <<'EOF'
feat(figures): D2 Batch 2 — regenerate secondary figures on masked tree

Topic 0 phantom-rank masked rerun, secondary/descriptive figures:
- PR-1 heatmap examples + per_subject_mi
- PR-4A daynight occupancy
- PR-6 supplementary (coreness sensitivity / endpoint Jaccard / fwd-rev multiples)
- PR-6 Step 6 held-out time figures
- PR-6 sup1 rank entropy (Topic 4 preflight, descriptive)
- PR-4 PPT panels

None of these are referenced as primary evidence in the rewritten topic1 §2/§3;
they exist for completeness of the masked tree.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist (executor: run after Task 29)

- [ ] grep -r "interictal_propagation_vs_masked/interictal_propagation_masked" . returns 0 lines
- [ ] All 6 plot scripts have `--masked-features` flag (smoke-test each with `--help`)
- [ ] `tests/test_*_masked_path_routing.py` (4 tests) all PASS
- [ ] `docs/topic1_within_event_dynamics.md` line count ≤ 320 (was 526)
- [ ] §0 §-anchor migration table covers every old §-anchor from the 2026-04-20 version
- [ ] §2.2 contains the phrases "机制层" AND "不是 cohort-level claim" AND "H1 (SOZ anchoring) cohort Wilcoxon 仍 NULL"
- [ ] §2.1 enumerates stable_k distributions for Tier 0 AND Tier 1 AND Tier 2
- [ ] All figure paths cited in §3 exist on disk under `results/interictal_propagation_masked/`
- [ ] `results/interictal_propagation/figures/README.md` has SUPERSEDED block at top

If any item fails: fix and create a follow-up commit. Do not amend the closing
Phase 4 commit.
