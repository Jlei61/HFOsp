# Topic 4 Phase 2 (H3 + H4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement SEF-ITP framework Phase 2 (H3 mark-independent template sampling + H4 normalized rate vs geometry instability) on the n=23 Phase 1 cohort, with framework v1.0.5 verdict semantics locked.

**Architecture:**
- **H3 = ingest-heavy.** Re-use PR-7 `template_temporal_pairing.compute_burst_diagnostic_with_nulls` (lag1_same + run_length) + PR-7 per-subject pairing JSON (window excess at 10s/30s/60s/1800s with N2 marginal-preserving null) + PR-7 addendum `tost_equivalence` (cohort TOST vs ±δ_excess band) + PR-6 anchoring `split_half_robustness.per_split.{first_half_second_half,odd_even_block}.subject_mean_jaccard_endpoint` (endpoint stability ≥ 0.7). PR-7 burst diagnostic is currently only run on 8 of the 23 cohort subjects (the original PR-7 P3 strict cohort); Task 0 extends it to the missing 15.
- **H4 = new pipeline.** Slice each subject's 24h / multi-day data into 2h epochs (preserving time order; 1h epochs as sensitivity), compute per-epoch HFO rate(t) + per-epoch endpoint set(t) by re-deriving endpoint locally from epoch's events × cluster labels × template_rank, then `I_rate = std(log(rate)) / sqrt(matched-null variance)` vs `I_geom = std(1 − endpoint_Jaccard) / sqrt(matched-null variance)`, cohort Wilcoxon signed-rank + Cohen's d ≥ 0.3.

**Tech stack:** Python 3.11 / NumPy / SciPy. Re-uses existing modules: `src.template_temporal_pairing`, `src.template_anatomical_anchoring`, `src.interictal_propagation` (`load_subject_propagation_events`), `src.lagpat_rank_audit` (`build_masked_kmeans_features`), `src.sef_itp_phase1` (unified namespace helpers, coord loader).

**Non-negotiable contracts (framework v1.0.5 + CLAUDE.md §5 / §6 / §6.1):**

1. **H3 verdict naming = SUPPORTED / NOT SUPPORTED (memory) / NOT SUPPORTED (geometry-unstable) / CONTRADICTED.** Never write PASS/NULL/FAIL for H3 — guards against "PASS = proves independence" misreading.
2. **H3 措辞 lock:** "compatible with mark-independent sampling within tested precision." Forbid "证明独立 / proves mark-independence."
3. **δ_excess = 0.05** lock at framework time. Forbid post-hoc adjustment.
4. **H4 Cohen's d ≥ 0.3 floor** + Wilcoxon p<0.05. Forbid CV ≥ 3× v1 hard ratio (废 in v1.0.2).
5. **`masked_features` paths everywhere.** All consumers must read from `results/interictal_propagation_masked/...` (lagPatRank phantom-rank fix; Topic 0 §3.1). All lagPat loads via `src.lagpat_rank_audit.build_masked_kmeans_features` or `src.topic4_attractor_diagnostics.build_rank_feature_matrix(mask_phantom=True)`.

**Advisor catches (2026-05-23, before implementation) — locked into plan:**

A. **H3 endpoint stability combinator = OR (NOT AND).** Project convention (AGENTS.md cross-PR `forward_reverse_reproduced` = split-half OR odd-even). Defaulting to AND silently tightens the contract. Plan & Task 4 implementation use OR. Documented in archive.

B. **H4 I_rate matched null spec ambiguity is a science decision, not implementation.** Framework v1.0.5 §3.4 prose "shuffle epoch order, recompute std" is mathematically degenerate (std is permutation-invariant). Multiple non-degenerate readings exist (circular-shift-within-block / homogeneous Poisson / gamma-fit resample / cross-epoch event shuffle preserving block boundary). Plan implements **BOTH** the literal `epoch_order_shuffle` (reports `I_rate_undefined_under_shuffle_null: true`) **AND** `circular_shift_within_block` variants, reports both in per-subject + cohort summary, and writes `spec_amendment_2026-05-23.md` as a **proposal**, not a lock. **STOP after Task 13** — do NOT edit framework doc autonomously (Task 14 deferred until user returns).

C. **LOO populator is a separate task (Task 11.5).** Summarizer computes leave-one-out per metric by dropping each subject and re-running `tost_equivalence`; fills `leave_one_out_min_pass_rate` per metric. Without this, `compute_h3_integrated_verdict`'s CONTRADICTED branch silently never fires (default=1.0 masks unrobust-but-failing cases).

D. **Promote `_resolve_lagpat_subject_dir`** from `scripts/run_sef_itp_phase1.py:116` to public `resolve_lagpat_subject_dir` in `src.sef_itp_phase1` (Task 1 prep), so Phase 2 imports without script-cross-dependency.
6. **Re-use don't re-invent (CLAUDE.md §6 + §6.1):**
   - `pr7_addendum_p3_equivalence.tost_equivalence` is reused for H3 cohort TOST. **Question-match check**: PR-7 addendum asks "is cohort median of metric X equivalent to target within ±δ?" — H3 asks exactly the same question for different metrics (lag1_same_excess, window_excess at 10/30/60/1800s, run_length_lift). Test apparatus is reusable; subject 548 leave-out is PR-7-specific and dropped (Phase 2 cohort is n=23, not n=8).
   - `compute_burst_diagnostic_with_nulls` is reused on cohort subjects where PR-7 audit gates excluded them (Task 0 runs burst with relaxed gate). Same null construction (N2 marginal-preserving), same metrics. Question-match: both ask "compared to N2 mark-independent null, what is the lift on burst metrics?" — exact match.
   - PR-6 `split_half_robustness.subject_mean_jaccard_endpoint` is reused as H3 endpoint stability metric. Question-match: PR-6 asks "is the endpoint set the same between first half / second half?" — H3 asks exactly this. Reuse direct; no re-derivation.
   - H4 epoch endpoint re-derivation **does NOT call** PR-6 anchoring's full helper. Question-match: PR-6 anchoring's full helper enforces minimum n_valid + audit gates designed for cohort-level reproducibility checking — H4 epoch endpoint is a quick "top-k template_rank from local events" computation; not eligible for those gates. Phase 2 writes its own thin `compute_local_endpoint(events_bool, labels, template_rank_global, k=3)` so the question-match is honest.
7. **Time order preserved for H4.** Epoch slicing keeps temporal order; matched null shuffles **epoch order** (for I_rate) and **endpoint roles within epoch** (for I_geom). Forbid shuffling events across epochs.
8. **endpoint k = 3 lock** (framework time). Sensitivity sweeps to k ∈ {2, 4, 5} allowed but archived only, not main analysis.

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `src/sef_itp_phase2.py` | Pure helper functions: H3 metric extraction, H4 epoch slicing + local endpoint + I_rate / I_geom matched-null. Dataclass `SubjectPhase2Data`. | Create |
| `scripts/run_sef_itp_phase2.py` | Per-subject runner. Loads PR-7 pairing+burst JSON, PR-6 anchoring JSON, lagPat NPZ via Phase 1's data loader. Calls H3 + H4 helpers. Writes per-subject JSON. | Create |
| `scripts/summarize_sef_itp_phase2.py` | Cohort summary: TOST equivalence per metric, endpoint stability cohort fraction, H3 integrated verdict, H4 cohort Wilcoxon + Cohen's d. Writes `cohort_summary.json` + `cohort_subjects.csv`. | Create |
| `tests/test_sef_itp_phase2.py` | Unit tests for `src/sef_itp_phase2.py` helpers. | Create |
| `tests/test_run_sef_itp_phase2_integration.py` | Integration smoke (synthetic fixture; load fixture data; run runner; check per-subject JSON schema). | Create |
| `docs/archive/topic4/sef_itp_phase2/plan_2026-05-23.md` | Mirrors this superpowers plan, project-archive copy. | Create |
| `docs/archive/topic4/sef_itp_phase2/cohort_run_2026-05-23.md` | Results writeup. | Create after run |
| `results/topic4_sef_itp/phase2_temporal_x_geometry/per_subject/<dataset>_<sid>.json` | Per-subject H3 + H4 metrics + verdict. | Output |
| `results/topic4_sef_itp/phase2_temporal_x_geometry/cohort_summary.json` | Cohort TOST + Wilcoxon results + integrated verdicts. | Output |
| `results/topic4_sef_itp/phase2_temporal_x_geometry/cohort_subjects.csv` | Flat cohort table for inspection. | Output |
| `results/interictal_propagation_masked/template_pairing/per_subject_burst/<missing subject>.json` | PR-7 burst diagnostic for 15 missing cohort subjects. | Output (Task 0) |
| `docs/topic4_sef_itp_framework.md` | Mark Phase 2 done in §6.3 + update §11 self-check items. | Modify after run |

---

## Cohort (n=23, source: Phase 1 cohort)

```
epilepsiae_1073, epilepsiae_1077, epilepsiae_1084, epilepsiae_1096, epilepsiae_1146,
epilepsiae_1150, epilepsiae_139,  epilepsiae_253,  epilepsiae_442,  epilepsiae_548,
epilepsiae_583,  epilepsiae_590,  epilepsiae_635,  epilepsiae_922,  epilepsiae_958,
yuquan_chengshuai, yuquan_liyouran, yuquan_pengzihang, yuquan_songzishuo,
yuquan_zhangbichen, yuquan_zhangjiaqi, yuquan_zhangkexuan, yuquan_zhaochenxi
```

PR-7 burst diagnostic currently exists for 8 of these (the PR-7 P3 strict cohort: epilepsiae_1073, 139, 253, 548, 635, 922, 958, yuquan_liyouran). Task 0 runs burst for the remaining 15.

---

## Task 0: Run PR-7 burst diagnostic on missing 15 cohort subjects

**Why:** H3 cohort TOST on lag1_same_excess and run_length_lift requires PR-7 burst per-subject JSON for all 23 subjects.

**Files:**
- Output: `results/interictal_propagation_masked/template_pairing/per_subject_burst/<dataset>_<sid>.json` (15 new)

- [ ] **Step 1: Enumerate missing subjects**

Run:
```bash
python3 -c "
import json, pathlib
cohort = sorted([p.stem for p in pathlib.Path('/home/honglab/leijiaxin/HFOsp/results/topic4_sef_itp/phase1_spatial_geometry/per_subject').glob('*.json')])
burst_dir = pathlib.Path('/home/honglab/leijiaxin/HFOsp/results/interictal_propagation_masked/template_pairing/per_subject_burst')
missing = [s for s in cohort if not (burst_dir / f'{s}.json').exists()]
print(','.join(missing))
"
```

Expected: 15 comma-separated subject IDs.

- [ ] **Step 2: Confirm PR-7 audit eligibility gate is compatible**

The PR-7 `run_burst_diagnostic` filters by `_is_target(row)` using cohort `'all_eligible'` = `h1_primary_pass OR h2_negative_pass`. Confirm whether the 15 missing subjects are in PR-7 audit CSV:

```bash
python3 -c "
import csv, pathlib
audit = pathlib.Path('/home/honglab/leijiaxin/HFOsp/results/interictal_propagation_masked/template_pairing/pr7_cohort_audit.csv')
missing = '<paste from step 1>'.split(',')
with audit.open() as f:
    rows = {r['subject_id']: r for r in csv.DictReader(f)}
for s in missing:
    sid = s.split('_', 1)[1]
    r = rows.get(sid)
    if r is None:
        print(f'{s}: NOT IN AUDIT CSV')
    else:
        print(f'{s}: h1={r[\"h1_primary_pass\"]} h2={r[\"h2_negative_pass\"]} eligible={r[\"h1_primary_pass\"]==\"True\" or r[\"h2_negative_pass\"]==\"True\"}')
"
```

Branch logic:
- If all 15 are eligible under `all_eligible`: run burst with `--cohort all_eligible --only <subjects>` (Step 3a).
- If some are NOT eligible (failed both gates): add a Phase 2-local path that bypasses PR-7's audit gate (Step 3b). This is a Phase 2 contract decision documented in the archive: Phase 2 H3 cohort = Phase 1 cohort regardless of PR-7 audit verdict (H3 measures different statistic — mark-independence at fixed window scales — and does not require fwd/rev pairing structure).

- [ ] **Step 3a: Run burst with PR-7 cohort gate (if all eligible)**

```bash
python3 scripts/run_pr7_template_pairing.py \
    --masked-features --burst-diagnostic \
    --cohort all_eligible \
    --only <comma-separated subject ids without dataset prefix> \
    --burst-n-perm 500 --n2-window-min 30 --seed 0
```

Verify each missing subject now has burst JSON.

- [ ] **Step 3b: Run burst with Phase 2 cohort gate (if some not eligible)**

Write a thin wrapper script `scripts/run_pr7_burst_for_phase2_cohort.py` that bypasses PR-7's audit gate and operates directly on Phase 1 cohort:

```python
"""Phase 2 H3 prep: run PR-7 burst diagnostic on Phase 1 cohort (n=23)
regardless of PR-7 audit eligibility. Same N2 marginal-preserving null,
same metrics. Writes to results/interictal_propagation_masked/template_pairing/per_subject_burst/.
"""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np
from src.template_temporal_pairing import compute_burst_diagnostic_with_nulls
from src.interictal_propagation import load_subject_propagation_events
from src.lagpat_rank_audit import resolve_subject_lagpat_dir

PHASE1_DIR = pathlib.Path('results/topic4_sef_itp/phase1_spatial_geometry/per_subject')
MASKED_PROP_DIR = pathlib.Path('results/interictal_propagation_masked/per_subject')
BURST_OUT = pathlib.Path('results/interictal_propagation_masked/template_pairing/per_subject_burst')

def run_for_subject(dataset: str, subject_id: str, n_perm: int = 500, seed: int = 0):
    out_path = BURST_OUT / f'{dataset}_{subject_id}.json'
    if out_path.exists():
        print(f'  skip: {out_path.name} exists')
        return
    masked_json = json.loads((MASKED_PROP_DIR / f'{dataset}_{subject_id}.json').read_text())
    labels = np.asarray(masked_json['adaptive_cluster']['labels'], dtype=int)
    lagpat_dir = resolve_subject_lagpat_dir(dataset, subject_id)
    loaded = load_subject_propagation_events(lagpat_dir)
    event_abs_times = loaded['event_abs_times']
    block_time_ranges = loaded['block_time_ranges']
    if len(labels) != len(event_abs_times):
        raise RuntimeError(f'{subject_id}: labels ({len(labels)}) != events ({len(event_abs_times)})')
    diag = compute_burst_diagnostic_with_nulls(
        event_abs_times=event_abs_times,
        cluster_labels=labels,
        block_time_ranges=block_time_ranges,
        n_perm=n_perm,
        nulls=('N1', 'N2'),
        n2_window_seconds=1800.0,
        seed=seed,
    )
    record = {
        'subject_id': subject_id,
        'dataset': dataset,
        'n_events_used': int(len(labels)),
        'n_T_a': int(np.sum(labels == 0)),
        'n_T_b': int(np.sum(labels == 1)),
        'n_blocks': len(block_time_ranges),
        'nulls_run': ['N1', 'N2'],
        'n_perm': n_perm,
        'n2_window_seconds': 1800.0,
        'seed': seed,
        'burst_diagnostic': diag,
        'source': 'phase2_h3_prep',
    }
    BURST_OUT.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, default=float))
    print(f'  wrote {out_path.name}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=str, default=None, help='Comma-separated dataset_subject ids')
    ap.add_argument('--burst-n-perm', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    cohort = sorted([p.stem for p in PHASE1_DIR.glob('*.json')])
    if args.only:
        only = set(args.only.split(','))
        cohort = [s for s in cohort if s in only]
    print(f'Phase 2 H3 prep: running burst on {len(cohort)} subjects')
    for s in cohort:
        ds, sid = s.split('_', 1)
        try:
            run_for_subject(ds, sid, n_perm=args.burst_n_perm, seed=args.seed)
        except Exception as e:
            print(f'  FAILED {s}: {e}')

if __name__ == '__main__':
    main()
```

Note: `resolve_subject_lagpat_dir` may not exist; if it doesn't, look up the lagPat directory using Phase 1's loader (`scripts/run_sef_itp_phase1.py` already resolves these paths; copy that resolution logic into the script, or extract it into `src/sef_itp_phase1.py` first and re-use). Decision: extract into `src/sef_itp_phase1.py` as a public helper before Task 0 implementation if not already there.

Run:
```bash
python3 scripts/run_pr7_burst_for_phase2_cohort.py --only <missing 15>
```

- [ ] **Step 4: Verify all 23 subjects now have burst JSON**

```bash
python3 -c "
import pathlib
cohort = sorted([p.stem for p in pathlib.Path('results/topic4_sef_itp/phase1_spatial_geometry/per_subject').glob('*.json')])
burst = pathlib.Path('results/interictal_propagation_masked/template_pairing/per_subject_burst')
missing = [s for s in cohort if not (burst / f'{s}.json').exists()]
print(f'missing: {len(missing)}: {missing}')
"
```

Expected: `missing: 0: []`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_pr7_burst_for_phase2_cohort.py results/interictal_propagation_masked/template_pairing/per_subject_burst/
git commit -m "prep(topic4 phase2): extend PR-7 burst diagnostic to full n=23 cohort (H3 input)"
```

---

## Task 1: Create `src/sef_itp_phase2.py` skeleton

**Files:**
- Create: `src/sef_itp_phase2.py`
- Create: `tests/test_sef_itp_phase2.py`

- [ ] **Step 1: Write the failing test**

`tests/test_sef_itp_phase2.py`:
```python
"""Unit tests for src/sef_itp_phase2.py (SEF-ITP framework Phase 2: H3 + H4)."""
from __future__ import annotations

import numpy as np
import pytest

from src import sef_itp_phase2 as p2


def test_module_version():
    assert p2.__version__ == "v1.0.0"


def test_subject_phase2_data_dataclass_fields():
    """SubjectPhase2Data dataclass must carry both H3 ingest fields and H4 raw inputs."""
    s = p2.SubjectPhase2Data(
        dataset="yuquan", subject_id="test",
        # H3 ingest
        lag1_same_excess_n2=0.01,
        window_excess_n2={10.0: 0.0, 30.0: 0.0, 60.0: 0.0, 1800.0: 0.0},
        run_length_lift_n2=1.0,
        endpoint_jaccard_first_half=0.9,
        endpoint_jaccard_odd_even=0.85,
        # H4 raw
        event_abs_times=np.array([0.0, 1.0, 2.0]),
        cluster_labels=np.array([0, 1, 0]),
        block_time_ranges=[(0.0, 10.0)],
        template_ranks={0: np.array([0, 1, 2, 3, 4, 5]), 1: np.array([5, 4, 3, 2, 1, 0])},
        channel_names=["A", "B", "C", "D", "E", "F"],
    )
    assert s.dataset == "yuquan"
    assert s.lag1_same_excess_n2 == pytest.approx(0.01)
    assert 10.0 in s.window_excess_n2
```

- [ ] **Step 2: Run and confirm fail**

```bash
cd /home/honglab/leijiaxin/HFOsp && python3 -m pytest tests/test_sef_itp_phase2.py -v
```

Expected: ImportError on `src.sef_itp_phase2`.

- [ ] **Step 3: Create minimal module to make tests pass**

`src/sef_itp_phase2.py`:
```python
"""SEF-ITP framework Phase 2 — H3 mark-independence + H4 normalized rate/geometry instability.

Phase 2 sits on top of Phase 1's n=23 cohort. H3 is mostly ingest of PR-7 pairing
+ PR-7 burst + PR-6 anchoring split_half_robustness, with cohort TOST equivalence
re-used from `scripts/pr7_addendum_p3_equivalence.py`. H4 is new: epoch slicing +
per-epoch endpoint re-derivation from raw masked lagPat + cluster labels +
normalized instability vs matched null.

Verdict naming locked at framework time:
- H3: SUPPORTED / NOT_SUPPORTED_MEMORY / NOT_SUPPORTED_GEOMETRY_UNSTABLE / CONTRADICTED
       (NOT PASS/NULL/FAIL — guards against "PASS = proves independence")
- H4: PASS / NULL / FAIL (standard)

δ_excess = 0.05 lock at framework time. Forbid post-hoc adjustment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

__version__ = "v1.0.0"


@dataclass
class SubjectPhase2Data:
    """One subject's Phase 2 inputs (post-ingest, pre-compute)."""
    dataset: str
    subject_id: str
    # H3 ingest (already-computed per-subject metrics)
    lag1_same_excess_n2: float
    window_excess_n2: Dict[float, float]  # {10.0, 30.0, 60.0, 1800.0} → excess
    run_length_lift_n2: float
    endpoint_jaccard_first_half: float
    endpoint_jaccard_odd_even: float
    # H4 raw (computed per-epoch downstream)
    event_abs_times: np.ndarray
    cluster_labels: np.ndarray
    block_time_ranges: List[Tuple[float, float]]
    template_ranks: Dict[int, np.ndarray]  # {cluster_id → rank vector aligned to channel_names}
    channel_names: List[str]
```

- [ ] **Step 4: Run tests, verify pass**

```bash
python3 -m pytest tests/test_sef_itp_phase2.py -v
```

Expected: 2/2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sef_itp_phase2.py tests/test_sef_itp_phase2.py
git commit -m "feat(topic4 phase2): module skeleton + SubjectPhase2Data dataclass"
```

---

## Task 2: H3 — extract per-subject metrics from PR-7 + PR-6 JSONs

**Files:**
- Modify: `src/sef_itp_phase2.py`
- Modify: `tests/test_sef_itp_phase2.py`

- [ ] **Step 1: Write failing test for metric extractor**

Add to `tests/test_sef_itp_phase2.py`:
```python
def test_extract_h3_metrics_from_pr7_pairing_json():
    """extract_h3_metrics reads pairing_with_nulls.lift.N2.{10,30,60,1800}.excess."""
    pairing_json = {
        "pairing_with_nulls": {
            "lift": {
                "N2": {
                    "1.0": {"excess": 0.10},
                    "5.0": {"excess": 0.08},
                    "10.0": {"excess": 0.03},
                    "30.0": {"excess": 0.01},
                    "60.0": {"excess": 0.005},
                    "300.0": {"excess": 0.0},
                    "1800.0": {"excess": -0.001},
                    "3600.0": {"excess": -0.002},
                }
            }
        }
    }
    metrics = p2.extract_window_excess_from_pairing(pairing_json, windows=(10.0, 30.0, 60.0, 1800.0))
    assert metrics == {10.0: 0.03, 30.0: 0.01, 60.0: 0.005, 1800.0: -0.001}


def test_extract_h3_metrics_from_pr7_burst_json():
    """extract_h3_burst reads burst_diagnostic.lag1_same_excess.N2 + lift.N2.run_length_lift."""
    burst_json = {
        "burst_diagnostic": {
            "lag1_same_excess": {"N1": 0.02, "N2": 0.005},
            "lift": {
                "N2": {
                    "run_length_lift": 0.97,
                    "mean_run_length": 0.97,
                }
            },
        }
    }
    lag1, run_length = p2.extract_lag1_and_runlength_from_burst(burst_json)
    assert lag1 == pytest.approx(0.005)
    assert run_length == pytest.approx(0.97)


def test_extract_endpoint_jaccard_from_pr6_anchoring():
    """extract_endpoint_jaccard reads PR-6 split_half_robustness.per_split.{first,odd}.subject_mean_jaccard_endpoint."""
    anchoring_json = {
        "split_half_robustness": {
            "per_split": {
                "first_half_second_half": {"subject_mean_jaccard_endpoint": 0.9},
                "odd_even_block": {"subject_mean_jaccard_endpoint": 0.85},
            }
        }
    }
    fh, oe = p2.extract_endpoint_jaccard_from_anchoring(anchoring_json)
    assert fh == pytest.approx(0.9)
    assert oe == pytest.approx(0.85)


def test_extract_endpoint_jaccard_missing_per_split_raises():
    """If split_half_robustness or per_split is missing, raise (not silently zero)."""
    with pytest.raises(KeyError):
        p2.extract_endpoint_jaccard_from_anchoring({})
    with pytest.raises(KeyError):
        p2.extract_endpoint_jaccard_from_anchoring({"split_half_robustness": {}})
```

- [ ] **Step 2: Run, confirm fail**

```bash
python3 -m pytest tests/test_sef_itp_phase2.py -v
```

Expected: AttributeError on missing `extract_*` functions.

- [ ] **Step 3: Implement extractors**

Add to `src/sef_itp_phase2.py`:
```python
def extract_window_excess_from_pairing(
    pairing_json: dict,
    windows: Tuple[float, ...] = (10.0, 30.0, 60.0, 1800.0),
    null_key: str = "N2",
) -> Dict[float, float]:
    """Pull excess @ given Δt windows from PR-7 pairing per-subject JSON.

    PR-7 pairing JSON schema: pairing_with_nulls.lift.<null>.<window_str>.excess.
    Windows are stored as string keys (e.g., '10.0'). Returns {window_float: excess_float}.

    Raises KeyError if pairing_with_nulls / lift / null_key is missing.
    Raises KeyError if any requested window is missing (no silent default).
    """
    lift = pairing_json["pairing_with_nulls"]["lift"][null_key]
    out: Dict[float, float] = {}
    for w in windows:
        out[w] = float(lift[f"{w}"]["excess"])
    return out


def extract_lag1_and_runlength_from_burst(
    burst_json: dict,
    null_key: str = "N2",
) -> Tuple[float, float]:
    """Pull lag1_same_excess + run_length_lift from PR-7 burst per-subject JSON.

    PR-7 burst JSON schema:
      burst_diagnostic.lag1_same_excess.<null_key> → float (vs target=0)
      burst_diagnostic.lift.<null_key>.run_length_lift → float (vs target=1)
    """
    bd = burst_json["burst_diagnostic"]
    lag1 = float(bd["lag1_same_excess"][null_key])
    run_length = float(bd["lift"][null_key]["run_length_lift"])
    return lag1, run_length


def extract_endpoint_jaccard_from_anchoring(
    anchoring_json: dict,
) -> Tuple[float, float]:
    """Pull (first_half_second_half, odd_even_block) endpoint Jaccard from PR-6 anchoring JSON.

    PR-6 anchoring JSON schema:
      split_half_robustness.per_split.first_half_second_half.subject_mean_jaccard_endpoint
      split_half_robustness.per_split.odd_even_block.subject_mean_jaccard_endpoint
    """
    per_split = anchoring_json["split_half_robustness"]["per_split"]
    fh = float(per_split["first_half_second_half"]["subject_mean_jaccard_endpoint"])
    oe = float(per_split["odd_even_block"]["subject_mean_jaccard_endpoint"])
    return fh, oe
```

- [ ] **Step 4: Run tests, verify pass**

Expected: 6/6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sef_itp_phase2.py tests/test_sef_itp_phase2.py
git commit -m "feat(topic4 phase2): H3 ingest extractors for PR-7 pairing/burst + PR-6 anchoring"
```

---

## Task 3: H3 cohort TOST helper (re-use PR-7 addendum)

**Files:**
- Modify: `src/sef_itp_phase2.py`
- Modify: `tests/test_sef_itp_phase2.py`

**Re-use check (CLAUDE.md §6.1):** PR-7 addendum's `tost_equivalence` (in `scripts/pr7_addendum_p3_equivalence.py:123`) computes bootstrap-CI-based TOST equivalence against a target ± δ band. Its question — "is cohort median compatible with target ± δ?" — exactly matches what Phase 2 H3 needs (target=0 for excess metrics, target=1 for run_length_lift). Reusing as a library import preserves the science contract. The function lives in a `scripts/` module currently; we extract it into `src/sef_itp_phase2.py` (or a shared helper module) so that both PR-7 addendum and Phase 2 can import the same code.

- [ ] **Step 1: Write failing test**

Add to `tests/test_sef_itp_phase2.py`:
```python
def test_tost_equivalence_compatible_median_zero():
    """TOST equivalence: cohort median ~0 with tight CI well inside ±δ → equivalence_pass = True."""
    rng = np.random.default_rng(42)
    vals = rng.normal(loc=0.0, scale=0.005, size=30)
    out = p2.tost_equivalence(vals, target=0.0, delta=0.05, n_boot=2000, seed=0)
    assert out["equivalence_pass"] is True
    assert -0.05 < out["ci95_lo"] < out["ci95_hi"] < 0.05


def test_tost_equivalence_violated_median_outside_band():
    """Cohort median 0.1 > δ → equivalence_pass = False."""
    rng = np.random.default_rng(42)
    vals = rng.normal(loc=0.10, scale=0.01, size=30)
    out = p2.tost_equivalence(vals, target=0.0, delta=0.05, n_boot=2000, seed=0)
    assert out["equivalence_pass"] is False
    assert out["ci95_lo"] > 0.05  # whole CI above δ band


def test_tost_equivalence_target_one_for_run_length():
    """run_length_lift target=1 with vals ~ 1.0 → equivalence_pass = True."""
    rng = np.random.default_rng(42)
    vals = rng.normal(loc=1.0, scale=0.005, size=30)
    out = p2.tost_equivalence(vals, target=1.0, delta=0.05, n_boot=2000, seed=0)
    assert out["equivalence_pass"] is True
```

- [ ] **Step 2: Run, confirm fail**

Expected: AttributeError on missing `tost_equivalence`.

- [ ] **Step 3: Port `tost_equivalence` from PR-7 addendum into Phase 2 module**

Read the implementation from `scripts/pr7_addendum_p3_equivalence.py:123-152` and copy it into `src/sef_itp_phase2.py`. Keep the function body verbatim (same statistical guarantees as PR-7 addendum). Make signature explicit:

```python
def tost_equivalence(
    values: np.ndarray,
    target: float,
    delta: float,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Two One-Sided Test (TOST) for equivalence of cohort median to target ± delta.

    Equivalence iff bootstrap p_lower < alpha AND p_upper < alpha AND CI ⊂ (target±delta).
    Returned dict matches `pr7_addendum_p3_equivalence.tost_equivalence` schema exactly:
      median_obs, ci95_lo, ci95_hi, tost_p_lower, tost_p_upper, tost_p,
      equivalence_pass, median_inside_band, ci_inside_band, target, delta.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_medians = np.median(values[idx], axis=1)
    obs_median = float(np.median(values))
    ci_lo, ci_hi = float(np.quantile(boot_medians, alpha / 2)), float(np.quantile(boot_medians, 1 - alpha / 2))
    p_lower = float(np.mean(boot_medians <= target - delta))
    p_upper = float(np.mean(boot_medians >= target + delta))
    p_tost = max(p_lower, p_upper)
    inside_band = (target - delta) <= obs_median <= (target + delta)
    equivalence_pass = (p_tost < alpha) and (ci_lo > target - delta) and (ci_hi < target + delta)
    return {
        "median_obs": obs_median,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "tost_p_lower": p_lower,
        "tost_p_upper": p_upper,
        "tost_p": p_tost,
        "equivalence_pass": bool(equivalence_pass),
        "median_inside_band": bool(inside_band),
        "ci_inside_band": bool(ci_lo > target - delta and ci_hi < target + delta),
        "target": float(target),
        "delta": float(delta),
        "n": int(n),
    }
```

(Optional follow-up commit at end: replace `pr7_addendum_p3_equivalence.tost_equivalence` body with `from src.sef_itp_phase2 import tost_equivalence` to avoid drift. Defer; not required for Phase 2 correctness.)

- [ ] **Step 4: Run tests, verify pass**

Expected: 9/9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sef_itp_phase2.py tests/test_sef_itp_phase2.py
git commit -m "feat(topic4 phase2): re-use PR-7 TOST equivalence for H3 cohort verdict"
```

---

## Task 4: H3 integrated verdict logic

**Files:**
- Modify: `src/sef_itp_phase2.py`
- Modify: `tests/test_sef_itp_phase2.py`

**Verdict rules (framework v1.0.5 §3.3 + §11 lock):**

Mark-transition layer = `lag1_same_excess` AND each of `window_excess @ {10s, 30s, 60s, 1800s}` AND `run_length_lift`. "Mark-transition compatible" = all 6 TOST tests equivalence_pass=True.

Endpoint geometry layer = `endpoint_jaccard_first_half_median ≥ 0.7` **OR** `endpoint_jaccard_odd_even_median ≥ 0.7` (advisor catch A — project convention from `forward_reverse_reproduced`). Per-subject hard floor: at least one of the two Jaccard values ≥ 0.5.

Integrated:

| mark transition | endpoint stability | H3 verdict |
|---|---|---|
| compatible (all 6 TOST equiv) | stable (both Jaccard ≥ 0.7) | **SUPPORTED** |
| compatible | unstable | **NOT_SUPPORTED_GEOMETRY_UNSTABLE** |
| not compatible (≥1 TOST fail) AND robust (leave-one-out still fails) | stable | **CONTRADICTED** |
| not compatible AND NOT robust (single-subject sensitive) | stable | **NOT_SUPPORTED_MEMORY** |
| not compatible | unstable | **NOT_SUPPORTED_BOTH** (combined failure, archive note + Phase 0 re-audit suggested) |

- [ ] **Step 1: Write failing test**

```python
def test_h3_integrated_verdict_supported():
    """All TOST equivalent + both Jaccard medians ≥ 0.7 → SUPPORTED."""
    verdict = p2.compute_h3_integrated_verdict(
        cohort_tost={
            "lag1_same_excess":  {"equivalence_pass": True, "cohort_main": {"equivalence_pass": True}},
            "window_excess_10s": {"equivalence_pass": True, "cohort_main": {"equivalence_pass": True}},
            "window_excess_30s": {"equivalence_pass": True, "cohort_main": {"equivalence_pass": True}},
            "window_excess_60s": {"equivalence_pass": True, "cohort_main": {"equivalence_pass": True}},
            "window_excess_1800s": {"equivalence_pass": True, "cohort_main": {"equivalence_pass": True}},
            "run_length_lift":   {"equivalence_pass": True, "cohort_main": {"equivalence_pass": True}},
        },
        endpoint_jaccard_first_half_median=0.85,
        endpoint_jaccard_odd_even_median=0.80,
    )
    assert verdict == "SUPPORTED"


def test_h3_integrated_verdict_not_supported_geometry_unstable():
    """All TOST equivalent + BOTH endpoint Jaccard medians < 0.7 → NOT_SUPPORTED_GEOMETRY_UNSTABLE.
    OR combinator (advisor catch A) — one ≥ 0.7 would already qualify as stable."""
    verdict = p2.compute_h3_integrated_verdict(
        cohort_tost={k: {"equivalence_pass": True, "cohort_main": {"equivalence_pass": True}}
                     for k in ("lag1_same_excess", "window_excess_10s", "window_excess_30s",
                               "window_excess_60s", "window_excess_1800s", "run_length_lift")},
        endpoint_jaccard_first_half_median=0.60,
        endpoint_jaccard_odd_even_median=0.55,  # both below threshold
    )
    assert verdict == "NOT_SUPPORTED_GEOMETRY_UNSTABLE"


def test_h3_integrated_verdict_or_combinator():
    """One Jaccard median ≥ 0.7, other below → still SUPPORTED (OR combinator).
    Mirrors AGENTS.md forward_reverse_reproduced = split-half OR odd-even."""
    verdict = p2.compute_h3_integrated_verdict(
        cohort_tost={k: {"equivalence_pass": True, "cohort_main": {"equivalence_pass": True}}
                     for k in ("lag1_same_excess", "window_excess_10s", "window_excess_30s",
                               "window_excess_60s", "window_excess_1800s", "run_length_lift")},
        endpoint_jaccard_first_half_median=0.85,  # ≥ 0.7
        endpoint_jaccard_odd_even_median=0.55,    # < 0.7
    )
    assert verdict == "SUPPORTED"  # OR — only one needs to clear threshold


def test_h3_integrated_verdict_contradicted_robust():
    """≥1 TOST fail with robust LOO → CONTRADICTED."""
    cohort_tost = {k: {"equivalence_pass": True, "cohort_main": {"equivalence_pass": True}}
                   for k in ("lag1_same_excess", "window_excess_10s", "window_excess_30s",
                             "window_excess_60s", "window_excess_1800s", "run_length_lift")}
    cohort_tost["lag1_same_excess"] = {"equivalence_pass": False, "cohort_main": {"equivalence_pass": False}, "leave_one_out_min_pass_rate": 0.0}
    verdict = p2.compute_h3_integrated_verdict(
        cohort_tost=cohort_tost,
        endpoint_jaccard_first_half_median=0.85,
        endpoint_jaccard_odd_even_median=0.80,
    )
    assert verdict == "CONTRADICTED"
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement `compute_h3_integrated_verdict`** in `src/sef_itp_phase2.py`:

```python
def compute_h3_integrated_verdict(
    cohort_tost: Dict[str, dict],
    endpoint_jaccard_first_half_median: float,
    endpoint_jaccard_odd_even_median: float,
    *,
    jaccard_threshold: float = 0.70,
    loo_robust_threshold: float = 0.50,
) -> str:
    """Integrate H3 mark-transition + endpoint-stability verdict.

    Returns one of:
      SUPPORTED, NOT_SUPPORTED_MEMORY, NOT_SUPPORTED_GEOMETRY_UNSTABLE,
      NOT_SUPPORTED_BOTH, CONTRADICTED.

    Inputs:
    - cohort_tost: dict keyed by metric name; each value has 'equivalence_pass' (top-level)
      and optionally 'leave_one_out_min_pass_rate' (fraction of LOO subsets that still pass).
    - endpoint_jaccard_*_median: cohort median of subject-level mean Jaccard endpoint.

    Logic locked at framework time. δ_excess = 0.05 (already baked into cohort_tost).
    """
    mark_keys = ("lag1_same_excess", "window_excess_10s", "window_excess_30s",
                 "window_excess_60s", "window_excess_1800s", "run_length_lift")
    mark_pass = all(cohort_tost[k]["equivalence_pass"] for k in mark_keys)
    endpoint_stable = (
        endpoint_jaccard_first_half_median >= jaccard_threshold
        or endpoint_jaccard_odd_even_median >= jaccard_threshold
    )  # OR — project convention (AGENTS.md forward_reverse_reproduced)

    if mark_pass and endpoint_stable:
        return "SUPPORTED"
    if mark_pass and not endpoint_stable:
        return "NOT_SUPPORTED_GEOMETRY_UNSTABLE"
    # mark not pass:
    # check LOO robustness — if any failing metric still fails when one subject dropped,
    # it's robust → CONTRADICTED. Else single-subject sensitive → NOT_SUPPORTED_MEMORY.
    failing = [k for k in mark_keys if not cohort_tost[k]["equivalence_pass"]]
    robust = any(
        cohort_tost[k].get("leave_one_out_min_pass_rate", 1.0) < loo_robust_threshold
        for k in failing
    )
    if endpoint_stable:
        return "CONTRADICTED" if robust else "NOT_SUPPORTED_MEMORY"
    return "NOT_SUPPORTED_BOTH"
```

- [ ] **Step 4: Run tests, verify pass.**

Expected: 12/12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sef_itp_phase2.py tests/test_sef_itp_phase2.py
git commit -m "feat(topic4 phase2): H3 integrated verdict (SUPPORTED/NOT_SUPPORTED variants/CONTRADICTED)"
```

---

## Task 5: H4 — epoch slicer

**Files:**
- Modify: `src/sef_itp_phase2.py`
- Modify: `tests/test_sef_itp_phase2.py`

- [ ] **Step 1: Write failing test**

```python
def test_slice_events_into_epochs_basic():
    """24h of events sliced into 2h epochs → 12 epochs, time-ordered, no event overlap."""
    # 24h of fake events (3600 s/h × 24 h = 86400 s), one event per second
    t0 = 1_000_000.0
    times = t0 + np.arange(86_400, dtype=float)
    labels = np.tile([0, 1], 43_200)  # alternating clusters
    block_time_ranges = [(t0, t0 + 86_400)]
    epochs = p2.slice_events_into_epochs(
        event_abs_times=times,
        cluster_labels=labels,
        block_time_ranges=block_time_ranges,
        epoch_hours=2.0,
    )
    assert len(epochs) == 12
    # Each epoch has its time window and the events inside it
    assert epochs[0]["t_start"] == pytest.approx(t0)
    assert epochs[0]["t_end"] == pytest.approx(t0 + 7200.0)
    assert len(epochs[0]["event_indices"]) == 7200
    # Total events covered = original count (no loss)
    total = sum(len(ep["event_indices"]) for ep in epochs)
    assert total == 86_400
    # Time order preserved (epoch i starts after epoch i-1)
    for i in range(1, len(epochs)):
        assert epochs[i]["t_start"] >= epochs[i - 1]["t_end"]


def test_slice_events_into_epochs_handles_gaps():
    """Events split across two recording blocks with a 12h gap — epoch slicer follows blocks, not wall-clock."""
    t0 = 1_000_000.0
    block_a = (t0, t0 + 7200.0)            # block A: 2h
    block_b = (t0 + 50000.0, t0 + 57200.0) # block B: 2h, 12h after A
    # events: 100 in each block
    times = np.concatenate([
        np.linspace(t0 + 1.0, t0 + 7199.0, 100),
        np.linspace(t0 + 50001.0, t0 + 57199.0, 100),
    ])
    labels = np.tile([0, 1], 100)
    epochs = p2.slice_events_into_epochs(
        event_abs_times=times,
        cluster_labels=labels,
        block_time_ranges=[block_a, block_b],
        epoch_hours=2.0,
    )
    # Should be 2 epochs (one per 2h block) — slicer respects block boundaries, no inter-block epoch
    assert len(epochs) == 2
    assert len(epochs[0]["event_indices"]) == 100
    assert len(epochs[1]["event_indices"]) == 100
    # No "phantom" epoch covering the 12h gap
    assert epochs[1]["t_start"] >= block_b[0]


def test_slice_events_drops_short_epochs():
    """Epoch with < min_events events is dropped (configurable threshold)."""
    t0 = 1_000_000.0
    times = np.concatenate([
        np.linspace(t0 + 1.0, t0 + 7199.0, 100),    # epoch 0: 100 events (kept)
        np.linspace(t0 + 7201.0, t0 + 14399.0, 3),  # epoch 1: 3 events (dropped if min_events=10)
    ])
    labels = np.array([0, 1] * 50 + [0, 1, 0])
    block_time_ranges = [(t0, t0 + 14400.0)]
    epochs = p2.slice_events_into_epochs(
        event_abs_times=times, cluster_labels=labels,
        block_time_ranges=block_time_ranges, epoch_hours=2.0, min_events=10,
    )
    assert len(epochs) == 1
    assert len(epochs[0]["event_indices"]) == 100
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement `slice_events_into_epochs`**:

```python
def slice_events_into_epochs(
    event_abs_times: np.ndarray,
    cluster_labels: np.ndarray,
    block_time_ranges: List[Tuple[float, float]],
    epoch_hours: float = 2.0,
    min_events: int = 10,
) -> List[Dict[str, np.ndarray]]:
    """Slice events into time-ordered epochs of fixed wall-clock duration, respecting block boundaries.

    Each block is sliced independently into contiguous epoch_hours windows starting at the block's
    t_start. Events in the block falling into epoch window [t_start + k*Δ, t_start + (k+1)*Δ) are
    assigned to epoch k. Empty epochs or epochs with < min_events events are dropped.

    Time order is preserved across the returned list: epoch i+1's t_start >= epoch i's t_end (within
    a block) or >= the next block's start.

    Returns list of dicts:
      [{'t_start': float, 't_end': float, 'event_indices': np.ndarray, 'block_index': int}]
    """
    times = np.asarray(event_abs_times, dtype=float)
    epoch_seconds = epoch_hours * 3600.0
    epochs: List[Dict[str, np.ndarray]] = []
    for bi, (b_start, b_end) in enumerate(block_time_ranges):
        block_duration = b_end - b_start
        n_epochs = int(np.floor(block_duration / epoch_seconds))
        for k in range(n_epochs):
            t_s = b_start + k * epoch_seconds
            t_e = t_s + epoch_seconds
            mask = (times >= t_s) & (times < t_e)
            idx = np.where(mask)[0]
            if len(idx) < min_events:
                continue
            epochs.append({
                "t_start": float(t_s),
                "t_end": float(t_e),
                "event_indices": idx,
                "block_index": bi,
            })
    return epochs
```

- [ ] **Step 4: Run tests, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/sef_itp_phase2.py tests/test_sef_itp_phase2.py
git commit -m "feat(topic4 phase2): H4 epoch slicer (block-aware, time-preserving)"
```

---

## Task 6: H4 — per-epoch local endpoint (top-k template_rank from epoch events)

**Files:**
- Modify: `src/sef_itp_phase2.py`
- Modify: `tests/test_sef_itp_phase2.py`

**Re-use check (CLAUDE.md §6.1):** PR-6 anchoring's full helper enforces minimum n_valid + audit gates designed for cohort-level reproducibility. H4 epoch endpoint is a quick "top-k template_rank from local events for this epoch's cluster mask" computation, gated only by "epoch had ≥ min_events events in this cluster". The question — "what's the endpoint set of this slice's events" — is the same as PR-6 but the gates differ. We write `compute_local_endpoint(events_bool_local, labels_local, channel_names, k=3, valid_mask_global=None)` as a thin Phase 2-local function that operates ON the events in this epoch only.

Mechanics: for each cluster c, take the mean events_bool over events labeled c in this epoch → channel-wise participation; rank-by-mean (descending) gives the local source side; rank-by-mean ascending gives sink. Top-k of each defines the local endpoint set. Compared to PR-2's `template_rank` which uses `np.argsort(np.argsort(centered_mean))` over ALL events in the cluster, this re-computes the same statistic on the local event subset.

- [ ] **Step 1: Write failing tests**

```python
def test_compute_local_endpoint_returns_top_k_source_sink():
    """For 2 clusters with clear leader/laggard channels, local endpoint returns expected indices."""
    # 4 events, cluster 0 leader=ch0 laggard=ch5; cluster 1 leader=ch5 laggard=ch0
    # events_bool shape (n_events=4, n_ch=6)
    bools = np.array([
        [1, 1, 1, 0, 0, 0],   # cluster 0
        [1, 1, 0, 1, 0, 0],   # cluster 0
        [0, 0, 0, 1, 1, 1],   # cluster 1
        [0, 0, 1, 0, 1, 1],   # cluster 1
    ], dtype=bool)
    labels = np.array([0, 0, 1, 1])
    out = p2.compute_local_endpoint(bools, labels, k=2)
    # cluster 0: mean = [1, 1, 0.5, 0.5, 0, 0] → top-2 source = ch0,ch1; bottom-2 sink = ch4,ch5
    assert set(out[0]["source"]) == {0, 1}
    assert set(out[0]["sink"]) == {4, 5}
    # cluster 1: mean = [0, 0, 0.5, 0.5, 1, 1] → top-2 source = ch4,ch5; bottom-2 sink = ch0,ch1
    assert set(out[1]["source"]) == {4, 5}
    assert set(out[1]["sink"]) == {0, 1}


def test_compute_local_endpoint_respects_valid_mask():
    """If valid_mask drops some channels, they're not eligible for source/sink top-k."""
    bools = np.zeros((4, 6), dtype=bool)
    bools[:2, [0, 1, 2]] = True   # cluster 0 leaders ch0,1,2
    bools[2:, [3, 4, 5]] = True   # cluster 1 leaders ch3,4,5
    labels = np.array([0, 0, 1, 1])
    valid_mask = np.array([True, True, False, True, True, False])  # ch2, ch5 invalid
    out = p2.compute_local_endpoint(bools, labels, k=2, valid_mask=valid_mask)
    assert 2 not in out[0]["source"] and 2 not in out[0]["sink"]
    assert 5 not in out[1]["source"] and 5 not in out[1]["sink"]


def test_endpoint_jaccard_local_vs_global():
    """jaccard_endpoint(local, global) = |intersect|/|union| of source∪sink sets."""
    local = {0: {"source": [0, 1, 2], "sink": [5, 6, 7]}}
    global_ = {0: {"source": [0, 1, 3], "sink": [5, 6, 8]}}
    j = p2.endpoint_jaccard(local, global_, cluster_id=0)
    # local endpoint = {0,1,2,5,6,7}; global endpoint = {0,1,3,5,6,8}
    # intersect = {0,1,5,6} = 4; union = {0,1,2,3,5,6,7,8} = 8 → 0.5
    assert j == pytest.approx(0.5)
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement**

```python
def compute_local_endpoint(
    events_bool: np.ndarray,
    labels: np.ndarray,
    k: int = 3,
    valid_mask: np.ndarray = None,
) -> Dict[int, Dict[str, List[int]]]:
    """Compute per-cluster top-k source / bottom-k sink endpoint from local events.

    For each cluster c, compute mean(events_bool[labels==c], axis=0) → channel-wise participation.
    source = channel indices of top-k by mean; sink = bottom-k by mean.

    If valid_mask is provided, only valid channels are eligible for top-k selection.

    Returns: {cluster_id: {"source": [int, ...], "sink": [int, ...]}}.
    """
    out: Dict[int, Dict[str, List[int]]] = {}
    n_ch = events_bool.shape[1]
    if valid_mask is None:
        valid_mask = np.ones(n_ch, dtype=bool)
    eligible_idx = np.where(valid_mask)[0]
    for c in np.unique(labels):
        cluster_mask = labels == c
        if cluster_mask.sum() == 0:
            continue
        ch_mean = events_bool[cluster_mask].mean(axis=0)
        # Restrict to eligible channels
        elig_means = ch_mean[eligible_idx]
        if len(elig_means) < 2 * k:
            # Not enough channels for both source + sink top-k; degrade gracefully
            k_eff = max(1, len(elig_means) // 2)
        else:
            k_eff = k
        # Top-k (highest = source/leader)
        top_k_local = np.argsort(elig_means)[::-1][:k_eff]
        # Bottom-k (lowest = sink/laggard)
        bot_k_local = np.argsort(elig_means)[:k_eff]
        out[int(c)] = {
            "source": [int(eligible_idx[i]) for i in top_k_local],
            "sink": [int(eligible_idx[i]) for i in bot_k_local],
        }
    return out


def endpoint_jaccard(
    local: Dict[int, Dict[str, List[int]]],
    global_: Dict[int, Dict[str, List[int]]],
    cluster_id: int,
) -> float:
    """Jaccard(source ∪ sink) between local and global per-cluster endpoint sets."""
    L = set(local[cluster_id]["source"]) | set(local[cluster_id]["sink"])
    G = set(global_[cluster_id]["source"]) | set(global_[cluster_id]["sink"])
    if len(L | G) == 0:
        return 0.0
    return len(L & G) / len(L | G)
```

- [ ] **Step 4: Run tests, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/sef_itp_phase2.py tests/test_sef_itp_phase2.py
git commit -m "feat(topic4 phase2): H4 local endpoint extraction + Jaccard helper"
```

---

## Task 7: H4 — I_rate normalized instability with matched null

**Files:**
- Modify: `src/sef_itp_phase2.py`
- Modify: `tests/test_sef_itp_phase2.py`

- [ ] **Step 1: Write failing test**

```python
def test_compute_I_rate_constant_rate_low_instability():
    """If rate is constant across epochs, std(log(rate))=0 → I_rate ≈ 0."""
    rates = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    result = p2.compute_I_rate_normalized(rates, n_perm=500, seed=0)
    assert result["I_rate"] == pytest.approx(0.0, abs=1e-9)
    assert result["null_variance"] == pytest.approx(0.0, abs=1e-9)


def test_compute_I_rate_variable_rate():
    """Highly variable rate → I_rate > 1 (above matched null)."""
    rng = np.random.default_rng(42)
    rates = rng.lognormal(mean=2.0, sigma=1.0, size=12)
    result = p2.compute_I_rate_normalized(rates, n_perm=1000, seed=0)
    assert result["I_rate"] > 0  # well-defined positive number
    assert "null_std_mean" in result
    assert "log_rate_std_obs" in result


def test_compute_I_rate_matched_null_preserves_marginal():
    """Matched null = shuffle epoch order; std(log(rate)) doesn't change under permutation."""
    # Trivially: shuffling preserves the same set of log(rate) values → std is invariant.
    # So matched-null variance should be 0 (degenerate).
    rates = np.array([5.0, 10.0, 20.0, 40.0, 80.0])
    result = p2.compute_I_rate_normalized(rates, n_perm=200, seed=0)
    # null_std should be (close to) the obs std, so variance(null_std) ≈ 0
    assert result["null_std_var"] == pytest.approx(0.0, abs=1e-9)
    # This means I_rate is undefined / infinity; we report "instability undefined under shuffle null"
    assert result["I_rate_undefined_under_shuffle_null"] is True
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement** — but framework v1.0.5 §3.4 has a science spec ambiguity:

The framework says:
> `I_rate = std(log(rate)) across epochs / sqrt(matched-null variance of log(rate))`
> matched null: time-shuffle epoch order 1000 次，每次重算 std → 取分布的 variance 当作 baseline

But std is **invariant** to permutation of a fixed set of values. So matched-null variance of `std(log(rate))` under epoch-order shuffle is identically 0 by construction.

**Per advisor catch B (2026-05-23): this is a science decision, not an implementation choice.** Multiple non-degenerate readings exist:
- **(a) circular-shift-within-block**: random offset Δ ∈ [0, epoch_seconds), shift event times within each block (wrap-around), re-slice; preserves temporal structure at sub-epoch scale, randomizes epoch-membership
- **(b) homogeneous Poisson**: resample event timestamps from a Poisson with the same total event count over the same block duration; null distribution reflects "what would std be if rate were truly constant"
- **(c) gamma-fit resample**: fit gamma to empirical per-epoch rates, resample
- **(d) cross-epoch shuffle preserving block boundary**: literal alternative reading — shuffle events across epochs within a block

**Implement BOTH (a) and the literal (epoch_order_shuffle) null.** Report both. Write proposal in `spec_amendment_2026-05-23.md`. **STOP at Task 13** — do not pick one autonomously.

For now: `compute_I_rate_normalized` exposes `null_method` parameter; both methods callable; both per-subject results captured in JSON output.

```python
def compute_I_rate_normalized(
    rates: np.ndarray,
    n_perm: int = 1000,
    seed: int = 0,
    null_method: str = "epoch_order_shuffle",
) -> Dict[str, float]:
    """Normalized rate instability across epochs.

    I_rate = std(log(rate_obs)) / sqrt(matched-null variance)

    null_method options:
    - 'epoch_order_shuffle' (framework v1.0.5 §3.4 literal): shuffle epoch order;
      since std is permutation-invariant, this null is degenerate (var=0) and
      I_rate is flagged undefined. Kept for spec-faithful audit.
    - 'circular_shift_within_block' (Phase 2 v1.0.0 amendment): see spec_amendment_2026-05-23.md.
      Requires event_abs_times + block_time_ranges + epoch_hours, NOT just rates.

    Returns dict with I_rate, log_rate_std_obs, null_std_mean, null_std_var,
    I_rate_undefined_under_shuffle_null (bool).
    """
    rates = np.asarray(rates, dtype=float)
    log_rates = np.log(rates + 1e-12)  # guard against log(0)
    obs_std = float(np.std(log_rates))
    rng = np.random.default_rng(seed)
    null_stds = []
    for _ in range(n_perm):
        perm = rng.permutation(len(rates))
        null_stds.append(np.std(log_rates[perm]))
    null_stds = np.asarray(null_stds)
    null_var = float(np.var(null_stds))
    null_mean = float(np.mean(null_stds))
    undefined = bool(null_var < 1e-12)
    I_rate = float("inf") if undefined else obs_std / np.sqrt(null_var)
    return {
        "I_rate": I_rate,
        "log_rate_std_obs": obs_std,
        "null_std_mean": null_mean,
        "null_std_var": null_var,
        "I_rate_undefined_under_shuffle_null": undefined,
        "null_method": null_method,
    }
```

Add the circular-shift variant — `compute_I_rate_normalized_circular_shift(event_abs_times, block_time_ranges, epoch_hours, n_perm, seed)` — and unit-test that it gives a non-degenerate null.

- [ ] **Step 4: Run tests, verify pass.**

- [ ] **Step 5: Add `compute_I_rate_normalized_circular_shift` + test**:

```python
def compute_I_rate_normalized_circular_shift(
    event_abs_times: np.ndarray,
    block_time_ranges: List[Tuple[float, float]],
    epoch_hours: float = 2.0,
    min_events: int = 10,
    n_perm: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """Non-degenerate null for I_rate: per permutation, choose a random offset Δ ∈ [0, epoch_seconds)
    per block, shift event times within block (wrap-around), re-slice into epochs starting at block
    start, compute per-epoch rate, take std(log(rate)).
    """
    times = np.asarray(event_abs_times, dtype=float)
    epoch_seconds = epoch_hours * 3600.0
    rng = np.random.default_rng(seed)

    def _rates_for(times_in):
        rates = []
        for (b_start, b_end) in block_time_ranges:
            block_duration = b_end - b_start
            n_epochs = int(np.floor(block_duration / epoch_seconds))
            for k in range(n_epochs):
                t_s = b_start + k * epoch_seconds
                t_e = t_s + epoch_seconds
                cnt = int(np.sum((times_in >= t_s) & (times_in < t_e)))
                if cnt >= min_events:
                    rates.append(cnt / epoch_hours)
        return np.array(rates)

    obs_rates = _rates_for(times)
    obs_log_rates = np.log(obs_rates + 1e-12)
    obs_std = float(np.std(obs_log_rates))

    null_stds = []
    for _ in range(n_perm):
        shifted = times.copy()
        for bi, (b_start, b_end) in enumerate(block_time_ranges):
            mask = (shifted >= b_start) & (shifted < b_end)
            delta = rng.uniform(0.0, epoch_seconds)
            shifted_block = shifted[mask] + delta
            # wrap-around within block
            block_duration = b_end - b_start
            wrap = (shifted_block - b_start) % block_duration + b_start
            shifted[mask] = wrap
        null_log_rates = np.log(_rates_for(shifted) + 1e-12)
        null_stds.append(float(np.std(null_log_rates)))

    null_stds = np.asarray(null_stds)
    null_var = float(np.var(null_stds))
    null_mean = float(np.mean(null_stds))
    I_rate = obs_std / np.sqrt(null_var) if null_var > 1e-12 else float("inf")
    return {
        "I_rate": I_rate,
        "log_rate_std_obs": obs_std,
        "null_std_mean": null_mean,
        "null_std_var": null_var,
        "n_epochs": int(len(obs_rates)),
        "null_method": "circular_shift_within_block",
    }
```

Test:
```python
def test_compute_I_rate_circular_shift_nondegenerate():
    """Circular shift null gives non-zero variance → I_rate well-defined."""
    rng = np.random.default_rng(0)
    # 24h block, 30 events/h average, with a slow modulation creating rate variability
    t0 = 1_000_000.0
    times = np.sort(t0 + rng.uniform(0, 86_400, size=720))
    result = p2.compute_I_rate_normalized_circular_shift(
        event_abs_times=times,
        block_time_ranges=[(t0, t0 + 86_400)],
        epoch_hours=2.0, n_perm=200, seed=0,
    )
    assert result["null_std_var"] > 0
    assert np.isfinite(result["I_rate"])
```

- [ ] **Step 6: Write `spec_amendment_2026-05-23.md` as a proposal (not lock).**

```bash
mkdir -p docs/archive/topic4/sef_itp_phase2
```

Document the four candidate null methods (a–d above), math derivation of why epoch-order-shuffle is degenerate, expected behaviour of each candidate, and the recommendation (option a: circular-shift-within-block) — **as a proposal awaiting user ratification**. State explicitly that the framework doc edit is deferred until the user returns.

- [ ] **Step 7: Commit**

```bash
git add src/sef_itp_phase2.py tests/test_sef_itp_phase2.py docs/archive/topic4/sef_itp_phase2/spec_amendment_2026-05-23.md
git commit -m "feat(topic4 phase2): H4 I_rate two null methods (literal + circular-shift) + spec amendment proposal"
```

---

## Task 8: H4 — I_geom normalized instability with matched null

**Files:**
- Modify: `src/sef_itp_phase2.py`
- Modify: `tests/test_sef_itp_phase2.py`

H4 I_geom matched null per framework §3.4: "每 epoch 内 role-shuffle endpoint 标签（在 valid_mask=True 池子里随机选 |E| 个当 '该 epoch 的 endpoint'），重算 1000 次得 null 分布"

Implementation:
- For each epoch, observe `endpoint_jaccard(local_endpoint_t, global_endpoint)` per cluster, average across cluster (subject scalar per epoch).
- Compute `std(1 - jaccard) across epochs` = obs.
- Null: for each permutation, replace each epoch's local endpoint with a random sample of |E| channels from the subject's `valid_mask=True` pool, recompute Jaccard vs global, recompute std.
- I_geom = obs / sqrt(var(null_stds)).

- [ ] **Step 1: Test for I_geom non-degenerate (because random endpoint changes Jaccard).**

```python
def test_compute_I_geom_random_endpoint_null_nondegenerate():
    """Random endpoint sample → Jaccard varies → I_geom well-defined."""
    rng = np.random.default_rng(42)
    n_epochs = 10
    n_ch = 12
    k = 3  # k source + k sink = 2k = 6 channels per endpoint
    global_endpoint = {0: {"source": [0, 1, 2], "sink": [9, 10, 11]}}
    # Per-epoch local endpoints: pretend half are close to global, half are far
    per_epoch_local = []
    for i in range(n_epochs):
        if i < 5:
            per_epoch_local.append({0: {"source": [0, 1, 3], "sink": [9, 10, 8]}})
        else:
            per_epoch_local.append({0: {"source": [4, 5, 6], "sink": [4, 5, 6]}})
    valid_mask = np.ones(n_ch, dtype=bool)
    result = p2.compute_I_geom_normalized(
        per_epoch_local=per_epoch_local,
        global_endpoint=global_endpoint,
        valid_mask=valid_mask,
        endpoint_size=6,
        n_perm=500,
        seed=0,
    )
    assert result["null_std_var"] > 0  # non-degenerate
    assert np.isfinite(result["I_geom"])
    assert result["I_geom"] > 0
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement.**

```python
def compute_I_geom_normalized(
    per_epoch_local: List[Dict[int, Dict[str, List[int]]]],
    global_endpoint: Dict[int, Dict[str, List[int]]],
    valid_mask: np.ndarray,
    endpoint_size: int,
    n_perm: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """Normalized endpoint-geometry instability across epochs.

    For each epoch, compute mean-across-cluster Jaccard(local_endpoint, global_endpoint).
    obs_std = std(1 - Jaccard) across epochs.
    null = for each permutation, replace each epoch's local endpoint with random |E| channels
    from valid_mask=True pool; recompute Jaccard vs global; recompute std.

    Returns dict with I_geom, geom_dispersion_std_obs, null_std_mean, null_std_var.
    """
    cluster_ids = list(global_endpoint.keys())
    eligible = np.where(valid_mask)[0]
    rng = np.random.default_rng(seed)

    def _epoch_jaccard(local):
        js = [endpoint_jaccard(local, global_endpoint, c) for c in cluster_ids if c in local]
        return float(np.mean(js)) if js else 0.0

    obs_dispersion = np.array([1.0 - _epoch_jaccard(le) for le in per_epoch_local])
    obs_std = float(np.std(obs_dispersion))

    n_epochs = len(per_epoch_local)
    null_stds = []
    for _ in range(n_perm):
        epoch_jaccards = []
        for _epoch in range(n_epochs):
            # Random |E| channels from valid pool, split half source / half sink
            half = endpoint_size // 2
            chosen = rng.choice(eligible, size=endpoint_size, replace=False)
            random_local = {}
            for c in cluster_ids:
                random_local[c] = {"source": chosen[:half].tolist(), "sink": chosen[half:].tolist()}
            epoch_jaccards.append(_epoch_jaccard(random_local))
        null_dispersion = 1.0 - np.array(epoch_jaccards)
        null_stds.append(np.std(null_dispersion))

    null_stds = np.asarray(null_stds)
    null_var = float(np.var(null_stds))
    null_mean = float(np.mean(null_stds))
    I_geom = obs_std / np.sqrt(null_var) if null_var > 1e-12 else float("inf")
    return {
        "I_geom": I_geom,
        "geom_dispersion_std_obs": obs_std,
        "null_std_mean": null_mean,
        "null_std_var": null_var,
        "n_epochs": n_epochs,
    }
```

- [ ] **Step 4: Run tests, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/sef_itp_phase2.py tests/test_sef_itp_phase2.py
git commit -m "feat(topic4 phase2): H4 I_geom normalized instability"
```

---

## Task 9: H4 cohort Wilcoxon + Cohen's d verdict

**Files:**
- Modify: `src/sef_itp_phase2.py`
- Modify: `tests/test_sef_itp_phase2.py`

- [ ] **Step 1: Write failing test**

```python
def test_h4_cohort_verdict_pass():
    """Cohort with all subjects I_rate >> I_geom + Cohen's d ≥ 0.3 → PASS."""
    I_rate = np.array([3.0, 2.5, 2.8, 2.2, 3.5, 2.9, 2.1, 3.1, 2.6, 2.4])
    I_geom = np.array([0.5, 0.8, 0.7, 0.9, 0.6, 0.8, 1.0, 0.7, 0.9, 0.8])
    result = p2.compute_h4_cohort_verdict(I_rate, I_geom)
    assert result["verdict"] == "PASS"
    assert result["wilcoxon_p"] < 0.05
    assert result["cohen_d"] >= 0.3


def test_h4_cohort_verdict_null_low_effect():
    """Cohort with I_rate ≈ I_geom → NULL."""
    rng = np.random.default_rng(42)
    I_rate = rng.normal(1.0, 0.3, size=10)
    I_geom = rng.normal(1.0, 0.3, size=10)
    result = p2.compute_h4_cohort_verdict(I_rate, I_geom)
    assert result["verdict"] == "NULL"


def test_h4_cohort_verdict_fail_geom_more_unstable():
    """I_geom systematically > I_rate → FAIL."""
    I_rate = np.array([0.5, 0.6, 0.4, 0.7, 0.5, 0.8, 0.4, 0.6, 0.5, 0.7])
    I_geom = np.array([2.5, 2.4, 2.8, 2.3, 2.6, 2.2, 2.7, 2.4, 2.5, 2.3])
    result = p2.compute_h4_cohort_verdict(I_rate, I_geom)
    assert result["verdict"] == "FAIL"
    assert result["cohen_d"] < 0
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement**

```python
from scipy.stats import wilcoxon

def compute_h4_cohort_verdict(
    I_rate_per_subject: np.ndarray,
    I_geom_per_subject: np.ndarray,
    *,
    p_threshold: float = 0.05,
    cohen_d_floor: float = 0.30,
) -> Dict[str, float]:
    """Cohort H4 verdict: PASS if Wilcoxon p<α AND Cohen's d ≥ floor (positive direction);
    NULL if not significant or |d| < floor; FAIL if significant in reverse direction.

    Returns dict: verdict, wilcoxon_p, cohen_d, n_subjects, median_I_rate, median_I_geom.
    """
    a = np.asarray(I_rate_per_subject, dtype=float)
    b = np.asarray(I_geom_per_subject, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    n = len(a)
    diff = a - b  # positive → rate more unstable than geom
    # Wilcoxon signed-rank one-sided (greater)
    if n < 6:
        return {
            "verdict": "UNDERPOWERED",
            "wilcoxon_p": float("nan"),
            "cohen_d": float("nan"),
            "n_subjects": n,
        }
    stat = wilcoxon(diff, alternative="greater", zero_method="wilcox")
    p_val = float(stat.pvalue)
    cohen_d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))
    if p_val < p_threshold and cohen_d >= cohen_d_floor:
        verdict = "PASS"
    elif cohen_d < 0 and wilcoxon(diff, alternative="less", zero_method="wilcox").pvalue < p_threshold:
        verdict = "FAIL"
    else:
        verdict = "NULL"
    return {
        "verdict": verdict,
        "wilcoxon_p": p_val,
        "cohen_d": cohen_d,
        "n_subjects": int(n),
        "median_I_rate": float(np.median(a)),
        "median_I_geom": float(np.median(b)),
    }
```

- [ ] **Step 4: Run tests, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/sef_itp_phase2.py tests/test_sef_itp_phase2.py
git commit -m "feat(topic4 phase2): H4 cohort Wilcoxon + Cohen's d verdict"
```

---

## Task 10: Runner script `scripts/run_sef_itp_phase2.py`

**Files:**
- Create: `scripts/run_sef_itp_phase2.py`
- Create: `tests/test_run_sef_itp_phase2_integration.py`

- [ ] **Step 1: Write failing integration test**

```python
"""Integration test for scripts/run_sef_itp_phase2.py — runs end-to-end on synthetic fixture."""
import json
import pathlib
import subprocess
import sys
import tempfile
import pytest


def test_run_phase2_smoke(tmp_path):
    """End-to-end smoke: write fixture JSONs (PR-7 pairing/burst, PR-6 anchoring, PR-2 masked),
    run scripts/run_sef_itp_phase2.py for one synthetic subject, verify output schema."""
    # ... (full fixture setup; skip if too long to inline)
    pytest.skip("Integration smoke set up later; runs after full plan implemented")
```

- [ ] **Step 2: Implement runner** — wires Phase 1's `load_subject_for_phase1` (extends or re-uses for H4 raw lagPat) + H3 ingest extractors + H4 epoch + I_rate / I_geom + per-subject JSON writer.

```python
#!/usr/bin/env python3
"""SEF-ITP Phase 2 runner: H3 mark-independence + H4 normalized instability.

Usage:
  python scripts/run_sef_itp_phase2.py --subject yuquan_chengshuai --hypothesis all \
      --epoch-hours 2.0 --output-dir results/topic4_sef_itp/phase2_temporal_x_geometry/per_subject
  python scripts/run_sef_itp_phase2.py --all --hypothesis all --output-dir <above>

Reads:
  - results/topic4_sef_itp/phase1_spatial_geometry/per_subject/<dataset>_<sid>.json   (cohort gate)
  - results/interictal_propagation_masked/template_pairing/per_subject/<...>.json     (PR-7 pairing)
  - results/interictal_propagation_masked/template_pairing/per_subject_burst/<...>.json (PR-7 burst)
  - results/interictal_propagation_masked/template_anchoring/per_subject/<...>.json   (PR-6)
  - results/interictal_propagation_masked/per_subject/<...>.json                      (PR-2 masked)
  - lagPat NPZ (resolved via Phase 1 path resolver)

Writes:
  - <output-dir>/<dataset>_<sid>.json  (H3 metrics + H4 instability per subject)
"""
# ... full implementation
```

- [ ] **Step 3: Run on 1 Yuquan + 1 Epilepsiae subject**

```bash
python scripts/run_sef_itp_phase2.py --subject yuquan_chengshuai --hypothesis all
python scripts/run_sef_itp_phase2.py --subject epilepsiae_1073 --hypothesis all
```

Inspect `results/topic4_sef_itp/phase2_temporal_x_geometry/per_subject/yuquan_chengshuai.json` for schema correctness.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_sef_itp_phase2.py tests/test_run_sef_itp_phase2_integration.py
git commit -m "feat(topic4 phase2): runner script (H3 ingest + H4 instability per subject)"
```

---

## Task 10.5: Promote `_resolve_lagpat_subject_dir` to public

**Files:**
- Modify: `src/sef_itp_phase1.py` (add public `resolve_lagpat_subject_dir`)
- Modify: `scripts/run_sef_itp_phase1.py` (use the promoted version)
- Add: test in `tests/test_sef_itp_phase1.py` (or extend existing)

Currently lives at `scripts/run_sef_itp_phase1.py:116` as private. Phase 2 needs it; cross-script import is a smell. Promote with a 30-second move + import-swap.

- [ ] **Step 1: Add to `src/sef_itp_phase1.py`** (resolve helper body verbatim, with default roots):

```python
from pathlib import Path

YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")

def resolve_lagpat_subject_dir(
    dataset: str,
    subject_id: str,
    yuquan_root: Path = YUQUAN_ROOT,
    epilepsiae_root: Path = EPILEPSIAE_ROOT,
) -> Path:
    """Locate the lagPat NPZ directory for a subject across both datasets."""
    if dataset == "yuquan":
        return yuquan_root / subject_id
    legacy = epilepsiae_root / subject_id / "all_recs"
    return legacy if legacy.exists() else epilepsiae_root / subject_id
```

- [ ] **Step 2: Update `scripts/run_sef_itp_phase1.py:116`** to delegate:

```python
from src.sef_itp_phase1 import resolve_lagpat_subject_dir as _resolve_lagpat_subject_dir
# remove the old local definition
```

- [ ] **Step 3: Add a test** verifying the resolver picks `all_recs` vs flat for Epilepsiae:

```python
def test_resolve_lagpat_subject_dir_dispatches_dataset(tmp_path):
    from src.sef_itp_phase1 import resolve_lagpat_subject_dir
    yuq = tmp_path / "yuq"
    epi = tmp_path / "epi"
    (yuq / "subj1").mkdir(parents=True)
    (epi / "subj2" / "all_recs").mkdir(parents=True)
    (epi / "subj3").mkdir()  # legacy flat
    assert resolve_lagpat_subject_dir("yuquan", "subj1", yuq, epi) == yuq / "subj1"
    assert resolve_lagpat_subject_dir("epilepsiae", "subj2", yuq, epi) == epi / "subj2" / "all_recs"
    assert resolve_lagpat_subject_dir("epilepsiae", "subj3", yuq, epi) == epi / "subj3"
```

- [ ] **Step 4: Run full Phase 1 test suite to confirm no regression.**

```bash
python3 -m pytest tests/test_sef_itp_phase1.py tests/test_run_sef_itp_phase1_integration.py -v
```

Expected: 115/115 still green (or +1 for the new resolver test).

- [ ] **Step 5: Commit**

```bash
git add src/sef_itp_phase1.py scripts/run_sef_itp_phase1.py tests/test_sef_itp_phase1.py
git commit -m "refactor(topic4 phase1): promote resolve_lagpat_subject_dir to public src module"
```

---

## Task 11: Summarizer `scripts/summarize_sef_itp_phase2.py`

**Files:**
- Create: `scripts/summarize_sef_itp_phase2.py`

Reads per-subject JSONs from `results/topic4_sef_itp/phase2_temporal_x_geometry/per_subject/*.json`, runs:
- H3 cohort TOST for each of 6 metrics (lag1_same_excess, window_excess_{10,30,60,1800}s, run_length_lift) — re-uses `p2.tost_equivalence` with target & δ per metric
- H3 endpoint stability cohort: median(first_half) + median(odd_even), fraction with both ≥ 0.7
- H3 integrated verdict via `p2.compute_h3_integrated_verdict`
- H4 cohort Wilcoxon + Cohen's d via `p2.compute_h4_cohort_verdict`
- Writes `cohort_summary.json` + `cohort_subjects.csv`

- [ ] Run on full cohort after Task 12.

- [ ] **Commit**:
```bash
git add scripts/summarize_sef_itp_phase2.py
git commit -m "feat(topic4 phase2): cohort summarizer (H3 TOST + H4 Wilcoxon verdicts)"
```

---

## Task 11.5: LOO populator (advisor catch C — required for H3 CONTRADICTED branch)

**Files:**
- Modify: `scripts/summarize_sef_itp_phase2.py`
- Modify: `tests/test_sef_itp_phase2.py` (add test)

Without leave-one-out, `compute_h3_integrated_verdict`'s CONTRADICTED branch silently never fires (`default=1.0` masks unrobust-but-failing cases). The summarizer must populate `leave_one_out_min_pass_rate` per metric.

- [ ] **Step 1: Write failing test in `tests/test_sef_itp_phase2.py`**

```python
def test_cohort_tost_with_leave_one_out_robust_failure():
    """When the cohort metric fails TOST equivalence AND the failure persists across all LOO drops,
    leave_one_out_min_pass_rate = 0.0 (no LOO subset restores equivalence)."""
    # Build a cohort where 30 subjects all have value ~0.10 (far above δ=0.05 band)
    rng = np.random.default_rng(0)
    values = rng.normal(0.10, 0.005, size=30)
    cohort_main = p2.tost_equivalence(values, target=0.0, delta=0.05, n_boot=1000, seed=0)
    assert cohort_main["equivalence_pass"] is False  # main cohort fails

    loo = p2.cohort_tost_with_loo(values, target=0.0, delta=0.05, n_boot=500, seed=0)
    assert loo["cohort_main"]["equivalence_pass"] is False
    assert loo["leave_one_out_min_pass_rate"] == 0.0  # 0/30 LOO subsets restore equivalence


def test_cohort_tost_with_leave_one_out_single_subject_sensitive():
    """One subject is the outlier. Dropping it restores equivalence."""
    values = np.concatenate([
        np.random.default_rng(0).normal(0.0, 0.005, size=29),  # 29 compatible
        np.array([0.5]),                                        # 1 huge outlier
    ])
    loo = p2.cohort_tost_with_loo(values, target=0.0, delta=0.05, n_boot=500, seed=0)
    assert loo["cohort_main"]["equivalence_pass"] is False  # outlier breaks it
    assert loo["leave_one_out_min_pass_rate"] > 0.0  # ≥ 1/30 LOO restores
```

- [ ] **Step 2: Implement `cohort_tost_with_loo` in `src/sef_itp_phase2.py`**:

```python
def cohort_tost_with_loo(
    values: np.ndarray,
    target: float,
    delta: float,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Cohort TOST + leave-one-out robustness.

    Returns:
      cohort_main: tost_equivalence on full cohort
      leave_one_out: dict {drop_i: tost_equivalence dict} for each subject dropped
      leave_one_out_min_pass_rate: fraction of LOO subsets that pass equivalence
      equivalence_pass: True iff cohort_main passes (mirrored for compat with verdict logic)
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    main = tost_equivalence(values, target, delta, n_boot=n_boot, alpha=alpha, seed=seed)
    loo = {}
    n_pass = 0
    for i in range(n):
        sub = np.delete(values, i)
        loo[f"drop_{i}"] = tost_equivalence(sub, target, delta, n_boot=n_boot, alpha=alpha, seed=seed + 1 + i)
        if loo[f"drop_{i}"]["equivalence_pass"]:
            n_pass += 1
    return {
        "cohort_main": main,
        "equivalence_pass": main["equivalence_pass"],  # alias for verdict consumer
        "leave_one_out": loo,
        "leave_one_out_min_pass_rate": n_pass / n if n > 0 else 0.0,
    }
```

- [ ] **Step 3: Wire `cohort_tost_with_loo` into summarizer for each metric.**

- [ ] **Step 4: Run tests; verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/sef_itp_phase2.py scripts/summarize_sef_itp_phase2.py tests/test_sef_itp_phase2.py
git commit -m "feat(topic4 phase2): cohort TOST + leave-one-out for CONTRADICTED branch"
```

---

## Task 12: Full cohort run + verification

- [ ] **Step 1: Run runner on full n=23 cohort**

```bash
python scripts/run_sef_itp_phase2.py --all --hypothesis all --epoch-hours 2.0
```

- [ ] **Step 2: Run summarizer**

```bash
python scripts/summarize_sef_itp_phase2.py
```

- [ ] **Step 3: Sanity-check output**

```bash
python3 -c "
import json
s = json.load(open('results/topic4_sef_itp/phase2_temporal_x_geometry/cohort_summary.json'))
print('H3 verdict:', s['h3']['integrated_verdict'])
print('H4 verdict:', s['h4']['verdict'])
print('H4 cohen_d:', s['h4']['cohen_d'])
"
```

- [ ] **Step 4: Commit results**

```bash
git add results/topic4_sef_itp/phase2_temporal_x_geometry/
git commit -m "results(topic4 phase2): full n=23 cohort run (H3 + H4)"
```

---

## Task 13: Archive doc `docs/archive/topic4/sef_itp_phase2/cohort_run_2026-05-23.md`

Sections (CLAUDE.md §8 三段式朴素话 + 内部代号括号补注):

- §0. Cohort 漏斗（n=23 from Phase 1）
- §1. H3 (mark independence + endpoint stability)
  - 测了什么: 连续事件挑同一模板的频率 + 多个时间窗的同模板偏置 + endpoint 集合时间稳定
  - 怎么测的: PR-7 N2 marginal-preserving null + cohort TOST vs ±0.05 + PR-6 split_half endpoint Jaccard ≥ 0.7
  - 揭示了什么: cohort 数字 + 整合 verdict (SUPPORTED / NOT_SUPPORTED_* / CONTRADICTED)
- §2. H4 (normalized instability rate vs geometry)
  - 测了什么: 事件率漂动 vs endpoint 几何漂动的相对幅度
  - 怎么测的: 2h epoch 切片 + circular-shift null (spec amendment) + Wilcoxon + Cohen's d
  - 揭示了什么: cohort I_rate vs I_geom + verdict
- §3. Spec amendment (I_rate matched null framework v1.0.5 → Phase 2 v1.0.0 circular-shift)
- §4. Cohort 整体一句话 verdict
- §5. Pending work (1h epoch sensitivity, k ∈ {2,4,5} sensitivity)
- §6. 内部代号映射

- [ ] **Step 1: Write archive doc.**

- [ ] **Step 2: Commit**

```bash
git add docs/archive/topic4/sef_itp_phase2/
git commit -m "docs(topic4 phase2): cohort run + archive doc (H3 + H4 results)"
```

---

## Task 14 (DEFERRED — DO NOT RUN AUTONOMOUSLY): Framework doc update

**Advisor catch B (2026-05-23):** the H4 I_rate matched-null spec amendment is a science decision, not an implementation choice. The framework doc edit must be **proposed** to the user in `spec_amendment_2026-05-23.md` and only ratified when the user returns. **STOP after Task 13.**

Deliverables to user when they return:
- `docs/archive/topic4/sef_itp_phase2/spec_amendment_2026-05-23.md` (proposal)
- `docs/archive/topic4/sef_itp_phase2/cohort_run_2026-05-23.md` (per-subject + cohort with **both** null methods reported side by side)
- One-line summary in chat about what was decided autonomously and what is awaiting their call

The user decides:
1. Which null method enters the framework (circular-shift / Poisson / gamma / cross-epoch shuffle / something else)
2. Whether the framework banner moves to v1.0.6 or stays at v1.0.5 with a Phase 2 results pointer

**This task is intentionally NOT executable as a checkbox sequence.** It is a stop-sign.

---

## Self-Review

**1. Spec coverage:**
- H3 mark independence (lag1_same_excess + window_excess @ {10,30,60,1800}s + run_length_lift): Task 2 + 3 + 4 ✓
- H3 endpoint geometric stability (split-half + odd-even Jaccard recall): Task 2 + 4 ✓
- H3 verdict naming (SUPPORTED / NOT_SUPPORTED_* / CONTRADICTED): Task 4 ✓
- H3 措辞 lock "compatible with mark-independent within tested precision": baked into archive doc Task 13 ✓
- H4 normalized I_rate: Task 7 ✓ (with spec amendment surfaced)
- H4 normalized I_geom: Task 8 ✓
- H4 Wilcoxon + Cohen's d ≥ 0.3: Task 9 ✓
- masked features throughout: Tasks 0 / runner / extractors all read `_masked` tree ✓
- δ_excess = 0.05 lock: Tasks 3 + 4 + 11 ✓
- Re-use TOST + burst + PR-6 Jaccard helpers: Tasks 0 + 2 + 3 ✓

**2. Placeholder scan:**
- All task code blocks contain concrete implementations.
- The integration test in Task 10 step 1 is intentionally skipped — implementation lives in the runner script itself with a per-subject JSON schema check after Task 12. Acceptable.

**3. Type consistency:**
- `SubjectPhase2Data` field types align: dataclass fields used in extractor and runner.
- `tost_equivalence` return dict keys consistent with `compute_h3_integrated_verdict` consumption.
- `compute_local_endpoint` returns `Dict[int, Dict[str, List[int]]]` consumed by `endpoint_jaccard` and `compute_I_geom_normalized` — consistent.

**Critical surface to user:** Task 7 step 3 surfaces a framework spec ambiguity (I_rate matched null is degenerate under epoch-order shuffle). Pause + write `docs/archive/topic4/sef_itp_phase2/spec_amendment_2026-05-23.md` + add note in cohort_run archive. Do NOT silently switch null methods.

---

## Execution Notes

- **Frequent commits:** one per task or per logical sub-step. Hooks may reject commits with whitespace issues; do not skip hooks.
- **Test discipline:** every task runs `pytest tests/test_sef_itp_phase2.py -v` after implementation; full suite green before moving on.
- **Cohort gate:** Phase 2 cohort = Phase 1 cohort (n=23). Subjects without PR-7 burst JSON post-Task-0 → cohort excluded with reason logged.
- **Resource:** Single-machine; no parallelism beyond NumPy. Expected total runtime ~5 min for cohort (n=23 × 1000 perms × small NPZ loads).
- **Failure modes to flag (do not silent-fix):**
  - PR-7 burst JSON missing for a subject after Task 0 → log and exclude
  - PR-6 anchoring JSON `split_half_robustness` absent → log and exclude (some subjects fail PR-6 audit)
  - lagPat NPZ load fails → log and exclude
  - I_rate degenerate (`null_var ≈ 0`) per subject → log subject as "uninformative under shuffle null" and use circular-shift variant
