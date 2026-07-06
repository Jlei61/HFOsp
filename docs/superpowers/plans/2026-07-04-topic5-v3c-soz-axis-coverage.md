# Topic 5 V3c — SOZ / axis coverage + gated latency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the interictal HFO propagation axis `A` spatially covers clinical SOZ `S` and defines a structured axis-surplus `A∖S` (primary), then — gated on a label-blind latency assay-QC — whether `A∖S` is recruited onset-synchronously with, or downstream of, the axis-covered SOZ core `A∩S` (secondary).

**Architecture:** Two pure-function modules (`topic5_v3c_coverage.py` = label-space set ops + spatial nulls; `topic5_v3c_latency.py` = timing/censoring/AUC), one IO helper (`_topic5_v3c_io.py`), and six runners/plotters. Reuses the frozen V3a classifier (`classify_subject_contacts`), the V3a within-shaft null primitive (`label_permute`), the recruitment cache (`bb_zt`), and `seeg_coord_loader`. Subject→cohort inference throughout; broad primary / narrow sensitivity, never pooled.

**Tech Stack:** Python 3, numpy, scipy.stats (spearmanr, wilcoxon), pytest, matplotlib. All artifacts under `results/topic5_ictal_recruitment/v3c_soz_axis_coverage/`.

**Spec:** `docs/superpowers/specs/2026-07-04-topic5-v3c-soz-axis-coverage-design.md` (read §2 set language, §4 coverage, §5 latency+QC, §6 spatial, §8 acceptance gates before each milestone).

## Global Constraints

- **Set language only** (spec §2): use `A` (`classify_subject_contacts`→`is_axis`), `S` (SOZ JSON ∩ all_clean pool), `A∩S`/`A∖S`/`S∖A`. Never use "off-axis" for `¬A` or `S∖A`.
- **Subject-first inference** (spec §7): contact→seizure→subject→cohort. NO pooled channel-level or seizure-level p-values. Cohort p = nested subject-level (cohort-median) null.
- **broad primary / narrow sensitivity, NEVER pool** (spec §3.3). Separate output dirs, separate verdicts.
- **coverage does not depend on latency** (spec §3.2): two eligibility tiers. V3c-1 runs on all SOZ+axis subjects; V3c-2 gated additionally on set-thresholds + assay-QC.
- **assay-QC is label-blind** (spec §5.2): QC functions must NOT accept `S` (SOZ). Enforced by signature.
- **latency primary null threshold** z_cross=2.0; window onset..+30s; hop 0.1s; sustain 3 frames. Sensitivity: z∈{1.5,2.5}, window 20s.
- **Never silently drop a subject** (V3a convention): a failing subject still emits a row with a `[skip]`/reason; exceptions are caught per-subject in runners.
- **Reuse, don't reinvent** (CLAUDE.md §6.1): `label_permute`, `shaft_constrained_permute`, `load_subject_coords`, `classify_subject_contacts`, `load_v3_config` are imported, not re-implemented.
- **tier = exploratory**; negatives acceptable; no forecasting; outcome = future/blocked (spec §10).

---

## File Structure

- `config/topic5_v3.yaml` — **modify**: add `v3c:` block (§Task 1).
- `src/topic5_v3c_coverage.py` — **create**: `coverage_metrics`, `coverage_null_distribution`, `surplus_spatial_metrics`, `distance_null_distribution`.
- `src/topic5_v3c_latency.py` — **create**: `first_crossing_latency`, `latency_seconds`, `encode_latency_for_rank`, `censoring_tallies`, `rank_diagnostics`, `threshold_stability`, `assay_valid`, `auc_late`, `delta_t`, `auc_null_distribution`.
- `scripts/_topic5_v3c_io.py` — **create**: `V3C_SUBJECTS`, `load_soz`, `axis_soz_join`, `extract_latency_matrix`, `load_axis_coords`.
- `scripts/run_topic5_v3c_coverage.py` — **create**: V3c-1 runner.
- `scripts/run_topic5_v3c_latency_qc.py` — **create**: label-blind assay-QC runner.
- `scripts/run_topic5_v3c_latency.py` — **create**: V3c-2 runner (gated).
- `scripts/run_topic5_v3c_surplus_spatial.py` — **create**: V3c-3 runner.
- `scripts/run_topic5_v3c_summary.py` — **create**: cohort tier + claim-language selector.
- `scripts/plot_topic5_v3c.py` — **create**: QC 3-fig + coverage forest + surplus spatial + S∖A.
- `tests/test_topic5_v3c_coverage.py`, `tests/test_topic5_v3c_latency.py`, `tests/test_topic5_v3c_io.py` — **create**.

---

## Milestone A — Config + coverage set-ops + coverage null

### Task 1: Add `v3c:` config block

**Files:**
- Modify: `config/topic5_v3.yaml` (append top-level `v3c:` block)
- Test: `tests/test_topic5_v3c_coverage.py`

**Interfaces:**
- Consumes: `src.topic5_v3_mode_transition.load_v3_config()` (already reads the whole yaml).
- Produces: `load_v3_config()["v3c"]` dict with the keys asserted below.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_v3c_coverage.py
from src.topic5_v3_mode_transition import load_v3_config

def test_v3c_config_keys():
    v = load_v3_config()["v3c"]
    assert v["z_cross"] == 2.0 and v["window_sec"] == 30.0 and v["hop_sec"] == 0.1
    assert v["assay_qc"]["t0_frac_max"] == 0.50 and v["assay_qc"]["finite_frac_min"] == 0.40
    assert v["interpretation"]["auc_hb_min"] == 0.60 and v["interpretation"]["auc_ha_band"] == [0.45, 0.55]
    assert v["latency"]["min_surplus"] == 3 and v["latency"]["min_covered_soz"] == 3
    assert v["spatial"]["min_subjects_for_primary"] == 3
    assert v["cohorts"]["primary"] == "broad" and v["nulls"]["n_perm"] == 1000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_v3c_coverage.py::test_v3c_config_keys -v`
Expected: FAIL with `KeyError: 'v3c'`

- [ ] **Step 3: Append the block to `config/topic5_v3.yaml`**

```yaml
v3c:
  hop_sec: 0.1
  z_cross: 2.0
  z_cross_sensitivity: [1.5, 2.5]
  window_sec: 30.0
  window_sec_sensitivity: 20.0
  onset_buffer_sec: -2.0          # sensitivity: window start = onset + buffer
  sustain_frames: 3
  coverage:
    min_soz: 1
    min_axis: 1
  spatial:
    min_subjects_for_primary: 3   # cohort spatial claim needs >=3 coord-eligible subjects (review P1)
  latency:
    min_surplus: 3                # |A∖S| >= 3
    min_covered_soz: 3            # |A∩S| >= 3 (primary contrast group)
    min_seizures: 2
    min_informative_seizures: 2
  assay_qc:
    finite_frac_min: 0.40
    t0_frac_max: 0.50
    uniq_ranks_min: 4
    thr_spearman_min: 0.5
    cens_frac_flag: 0.40          # > this -> censoring-sensitive (needs sensitivity agreement)
    informative_min_unique_ranks: 3
  interpretation:
    auc_hb_min: 0.60
    subject_hb_auc_min: 0.55
    delta_t_hb_min_sec: 2.0
    auc_ha_band: [0.45, 0.55]
  nulls:
    n_perm: 1000
    seed: 20260704
  cohorts:
    primary: broad
    replication: narrow
    never_pool: true
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_v3c_coverage.py::test_v3c_config_keys -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/topic5_v3.yaml tests/test_topic5_v3c_coverage.py
git commit -m "feat(topic5-v3c): add v3c config block (coverage/latency/QC/interpretation)"
```

---

### Task 2: `coverage_metrics` set operations

**Files:**
- Create: `src/topic5_v3c_coverage.py`
- Test: `tests/test_topic5_v3c_coverage.py`

**Interfaces:**
- Produces: `coverage_metrics(axis_names: list[str], soz_names: list[str]) -> dict` with keys
  `coverage` (|A∩S|/|S|, nan if |S|==0), `surplus_fraction` (|A∖S|/|A|, nan if |A|==0),
  `jaccard` (|A∩S|/|A∪S|, nan if union empty), `n_axis`, `n_soz`, `n_covered`, `n_surplus`,
  `n_missed`, and sorted lists `covered` (A∩S), `surplus` (A∖S), `missed` (S∖A).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_v3c_coverage.py  (append)
from src.topic5_v3c_coverage import coverage_metrics
import math

def test_coverage_metrics_basic():
    m = coverage_metrics(["a", "b", "c", "d"], ["b", "c", "e"])
    assert m["n_axis"] == 4 and m["n_soz"] == 3
    assert m["covered"] == ["b", "c"] and m["surplus"] == ["a", "d"] and m["missed"] == ["e"]
    assert m["coverage"] == 2 / 3
    assert m["surplus_fraction"] == 2 / 4
    assert m["jaccard"] == 2 / 5          # |A∩S|=2, |A∪S|={a,b,c,d,e}=5

def test_coverage_metrics_empty_soz():
    m = coverage_metrics(["a", "b"], [])
    assert math.isnan(m["coverage"]) and m["n_missed"] == 0 and m["surplus"] == ["a", "b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_v3c_coverage.py::test_coverage_metrics_basic -v`
Expected: FAIL with `ModuleNotFoundError: src.topic5_v3c_coverage`

- [ ] **Step 3: Create `src/topic5_v3c_coverage.py`**

```python
"""Topic 5 V3c — label-space set operations + spatial nulls (PURE, no I/O).

Set language (spec §2): A = interictal axis contacts; S = clinical SOZ ∩ pool.
This module never touches time; latency lives in topic5_v3c_latency.py.
"""
from __future__ import annotations

import numpy as np


def coverage_metrics(axis_names: list, soz_names: list) -> dict:
    """Coverage of clinical SOZ S by interictal axis A, plus surplus/jaccard.

    coverage = |A∩S|/|S| (sensitivity); surplus_fraction = |A∖S|/|A| (spec R1:
    near-mechanical for fixed |A|, descriptor only); jaccard = |A∩S|/|A∪S|.
    """
    A = set(axis_names)
    S = set(soz_names)
    covered = sorted(A & S)
    surplus = sorted(A - S)
    missed = sorted(S - A)
    union = A | S
    n_a, n_s = len(A), len(S)
    return {
        "coverage": (len(covered) / n_s) if n_s else float("nan"),
        "surplus_fraction": (len(surplus) / n_a) if n_a else float("nan"),
        "jaccard": (len(covered) / len(union)) if union else float("nan"),
        "n_axis": n_a, "n_soz": n_s,
        "n_covered": len(covered), "n_surplus": len(surplus), "n_missed": len(missed),
        "covered": covered, "surplus": surplus, "missed": missed,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_v3c_coverage.py -k coverage_metrics -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/topic5_v3c_coverage.py tests/test_topic5_v3c_coverage.py
git commit -m "feat(topic5-v3c): coverage_metrics set ops (coverage/surplus/jaccard/missed)"
```

---

### Task 3: `coverage_null_distribution` (same-shaft axis-count-preserving null)

**Files:**
- Modify: `src/topic5_v3c_coverage.py`
- Test: `tests/test_topic5_v3c_coverage.py`

**Interfaces:**
- Consumes: `src.topic5_v3_mode_transition.label_permute(axis_names, nonaxis_names, shaft_by_name, rng)` (within-shaft label shuffle; preserves per-shaft axis count exactly).
- Produces: `coverage_null_distribution(axis_names, all_clean, soz_names, shaft_by_name, *, n_perm, rng) -> np.ndarray` of length `n_perm` (null coverage values). Under each permutation, the axis label is reshuffled within shafts across ALL non-axis clean contacts (`all_clean ∖ A`), preserving `|A|` and per-shaft axis counts; `S` stays fixed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_v3c_coverage.py  (append)
import numpy as np
from src.topic5_v3c_coverage import coverage_null_distribution

def _shaft(name):  # "H1".."H4" -> "H"; "G12" -> "G"
    return "".join(c for c in name if not c.isdigit())

def test_coverage_null_preserves_axis_count_and_size():
    all_clean = ["H1", "H2", "H3", "H4", "G1", "G2", "G3", "G4"]
    axis = ["H1", "H2", "G1"]                      # per-shaft: H=2, G=1
    soz = ["H1", "H2"]
    shaft = {n: _shaft(n) for n in all_clean}
    null = coverage_null_distribution(axis, all_clean, soz, shaft, n_perm=200, rng=0)
    assert null.shape == (200,)
    # coverage = |A_null ∩ {H1,H2}| / 2 ; A_null always has H=2 of {H1..H4}, G=1 of {G1..G4}
    assert set(np.unique(null)).issubset({0.0, 0.5, 1.0})
    assert null.max() == 1.0 and null.min() == 0.0     # both extremes reachable within shaft H

def test_coverage_null_soz_shuffle_regression():
    # If S is shuffled to non-axis positions, observed coverage should sit inside the null (not high)
    all_clean = ["H1", "H2", "H3", "H4"]
    axis = ["H1", "H2"]; shaft = {n: "H" for n in all_clean}
    null_when_soz_offaxis = coverage_null_distribution(axis, all_clean, ["H3", "H4"], shaft, n_perm=200, rng=1)
    # observed coverage(axis={H1,H2}, soz={H3,H4}) = 0; null spans 0..1 -> observed not above null
    assert null_when_soz_offaxis.mean() > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_v3c_coverage.py -k coverage_null -v`
Expected: FAIL with `ImportError: cannot import name 'coverage_null_distribution'`

- [ ] **Step 3: Add to `src/topic5_v3c_coverage.py`**

```python
from src.topic5_v3_mode_transition import _coerce_rng, label_permute  # noqa: E402


def coverage_null_distribution(
    axis_names: list, all_clean: list, soz_names: list, shaft_by_name: dict,
    *, n_perm: int, rng,
) -> np.ndarray:
    """Same-shaft null: reshuffle the axis label within shafts across all clean
    contacts (preserves |A| and per-shaft axis count), recompute coverage of the
    FIXED soz set. Controls implant geometry (spec §4.2 primary null; R2: proves
    'beyond geometry', not 'beyond HFO-rich' — that needs the rate-matched null).
    """
    rng = _coerce_rng(rng)
    S = set(soz_names)
    n_s = len(S)
    axis_set = set(axis_names)
    nonaxis = [n for n in all_clean if n not in axis_set]
    out = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        new_axis, _ = label_permute(axis_names, nonaxis, shaft_by_name, rng)
        out[i] = (len(set(new_axis) & S) / n_s) if n_s else float("nan")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_v3c_coverage.py -k coverage_null -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/topic5_v3c_coverage.py tests/test_topic5_v3c_coverage.py
git commit -m "feat(topic5-v3c): coverage_null_distribution via same-shaft label_permute"
```

---

## Milestone B — V3c-1 coverage runner

### Task 4: SOZ loader + axis/SOZ join (IO)

**Files:**
- Create: `scripts/_topic5_v3c_io.py`
- Test: `tests/test_topic5_v3c_io.py`

**Interfaces:**
- Consumes: `scripts._topic5_v3_io.classify_subject_contacts(ds_sid, cohort, cfg)` (returns `is_axis`, `all_clean`, `shaft_by_name`, `cache_names`, `meta`, `geometry_sufficient`).
- Produces:
  - `V3C_SUBJECTS: dict[str, list[str]]` with keys `broad`, `narrow`.
  - `load_soz(dataset: str, subject: str) -> list[str]` (dataset ∈ {epilepsiae, yuquan}; [] if subject absent).
  - `axis_soz_join(cls: dict, soz_list: list) -> dict` returning `coverage_metrics(cls["is_axis"], S)` where `S = [n for n in soz_list if n in set(cls["all_clean"])]`, plus `soz_in_pool` (the intersected S).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_v3c_io.py
from scripts._topic5_v3c_io import load_soz, axis_soz_join, V3C_SUBJECTS

def test_v3c_subject_lists():
    assert "epilepsiae_1146" in V3C_SUBJECTS["broad"]
    assert "epilepsiae_442" in V3C_SUBJECTS["narrow"]
    assert "epilepsiae_442" not in V3C_SUBJECTS["broad"]     # no broad cache (spec §3.3)

def test_load_soz_epilepsiae():
    s = load_soz("epilepsiae", "1146")
    assert "ICL1" in s and len(s) == 14
    assert load_soz("epilepsiae", "9999") == []              # absent -> empty, not crash

def test_axis_soz_join_intersects_pool():
    cls = {"is_axis": ["a", "b"], "all_clean": ["a", "b", "c"]}
    j = axis_soz_join(cls, ["a", "z"])                        # z not in pool -> dropped
    assert j["soz_in_pool"] == ["a"] and j["n_covered"] == 1 and j["coverage"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_v3c_io.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts._topic5_v3c_io`

- [ ] **Step 3: Create `scripts/_topic5_v3c_io.py`**

```python
"""Topic 5 V3c — SOZ join + latency-matrix IO (reuses V3a classifier)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._topic5_v3_io import CACHE, classify_subject_contacts  # noqa: E402
from src.topic5_v3c_coverage import coverage_metrics  # noqa: E402

SOZ_JSON = {
    "epilepsiae": _ROOT / "results/epilepsiae_soz_core_channels.json",
    "yuquan": _ROOT / "results/yuquan_soz_core_channels.json",
}

# broad = broad-classifiable SOZ subjects (442/958 lack broad cache -> narrow only, spec §3.3)
V3C_SUBJECTS = {
    "broad": ["epilepsiae_139", "epilepsiae_253", "epilepsiae_635", "epilepsiae_1077",
              "epilepsiae_1096", "epilepsiae_1150", "epilepsiae_1146"],
    "narrow": ["epilepsiae_1096", "epilepsiae_1146", "epilepsiae_253",
               "epilepsiae_442", "epilepsiae_958"],
}


def load_soz(dataset: str, subject: str) -> list:
    """Clinical SOZ contact names for one subject; [] if the subject is absent."""
    path = SOZ_JSON[dataset]
    data = json.loads(path.read_text())
    return list(data.get(subject, []))


def axis_soz_join(cls: dict, soz_list: list) -> dict:
    """coverage_metrics(A, S) with S restricted to the all-clean pool; adds soz_in_pool."""
    pool = set(cls["all_clean"])
    soz_in_pool = [n for n in soz_list if n in pool]
    m = coverage_metrics(cls["is_axis"], soz_in_pool)
    m["soz_in_pool"] = soz_in_pool
    return m
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_v3c_io.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/_topic5_v3c_io.py tests/test_topic5_v3c_io.py
git commit -m "feat(topic5-v3c): SOZ loader + axis/SOZ pool join + cohort subject lists"
```

---

### Task 5: V3c-1 coverage runner (per-subject null + cohort-median null + LOSO)

**Files:**
- Create: `scripts/run_topic5_v3c_coverage.py`
- Test: `tests/test_topic5_v3c_coverage.py`

**Interfaces:**
- Consumes: `classify_subject_contacts`, `axis_soz_join`, `load_soz`, `coverage_null_distribution`, `load_v3_config`, `V3C_SUBJECTS`.
- Produces: `coverage_subject_row(ds_sid, cohort, cfg) -> dict` (per-subject coverage + observed null percentile) and `cohort_median_null(subject_obs, subject_nulls) -> dict` (observed cohort-median coverage percentile in the per-perm cohort-median null). Writes `results/topic5_ictal_recruitment/v3c_soz_axis_coverage/<cohort>/coverage_subject.csv` + `coverage_cohort.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_v3c_coverage.py  (append)
import numpy as np
from scripts.run_topic5_v3c_coverage import cohort_median_null

def test_cohort_median_null_percentile():
    # 3 subjects, each with an obs coverage and a null array; cohort stat = median over subjects
    subject_obs = [1.0, 1.0, 0.7]
    rng = np.random.default_rng(0)
    subject_nulls = [rng.uniform(0, 0.6, size=500) for _ in range(3)]   # nulls well below obs
    res = cohort_median_null(subject_obs, subject_nulls)
    assert res["obs_cohort_median"] == 1.0
    assert res["p_value"] < 0.01 and res["n_perm"] == 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_v3c_coverage.py::test_cohort_median_null_percentile -v`
Expected: FAIL with `ModuleNotFoundError: scripts.run_topic5_v3c_coverage`

- [ ] **Step 3: Create `scripts/run_topic5_v3c_coverage.py`**

```python
"""V3c-1: interictal-axis coverage of clinical SOZ (primary). Subject-first,
same-shaft null, cohort-median null + LOSO. broad primary / narrow sensitivity.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._topic5_v3_io import classify_subject_contacts  # noqa: E402
from scripts._topic5_v3c_io import V3C_SUBJECTS, axis_soz_join, load_soz  # noqa: E402
from src.topic5_v3_mode_transition import load_v3_config  # noqa: E402
from src.topic5_v3c_coverage import coverage_null_distribution  # noqa: E402

OUT = _ROOT / "results/topic5_ictal_recruitment/v3c_soz_axis_coverage"
COLS = ["subject", "cohort", "n_axis", "n_soz", "n_covered", "n_surplus", "n_missed",
        "coverage", "surplus_fraction", "jaccard", "coverage_null_p",
        "null_q05", "null_q50", "null_q95", "eligible"]   # q05/q50/q95 for the forest figure


def coverage_subject_row(ds_sid: str, cohort: str, cfg: dict) -> dict:
    dataset, subj = ds_sid.split("_", 1)
    row = {c: float("nan") for c in COLS}
    row.update({"subject": ds_sid, "cohort": cohort, "eligible": False})
    try:
        cls = classify_subject_contacts(ds_sid, cohort, cfg)
        j = axis_soz_join(cls, load_soz(dataset, subj))
        vc = cfg["v3c"]
        eligible = j["n_soz"] >= vc["coverage"]["min_soz"] and j["n_axis"] >= vc["coverage"]["min_axis"]
        null = coverage_null_distribution(
            cls["is_axis"], cls["all_clean"], j["soz_in_pool"], cls["shaft_by_name"],
            n_perm=vc["nulls"]["n_perm"], rng=vc["nulls"]["seed"],
        ) if eligible else np.array([])
        p = float((np.sum(null >= j["coverage"]) + 1) / (null.size + 1)) if null.size else float("nan")
        q05, q50, q95 = (np.percentile(null, [5, 50, 95]) if null.size else (np.nan, np.nan, np.nan))
        row.update({k: j[k] for k in ("n_axis", "n_soz", "n_covered", "n_surplus", "n_missed",
                                      "coverage", "surplus_fraction", "jaccard")})
        row.update({"coverage_null_p": p, "null_q05": float(q05), "null_q50": float(q50),
                    "null_q95": float(q95), "eligible": bool(eligible),
                    "_null": null, "_obs": j["coverage"]})
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] {ds_sid} ({cohort}): {type(exc).__name__}: {exc}", flush=True)
    return row


def cohort_median_null(subject_obs: list, subject_nulls: list) -> dict:
    """Nested subject-level (cohort-median) null (spec §7): per perm take the median
    across subjects of that perm's null coverage; compare to the observed cohort median.
    """
    obs_med = float(np.median(subject_obs))
    n_perm = min(len(n) for n in subject_nulls)
    stacked = np.vstack([n[:n_perm] for n in subject_nulls])   # (n_subj, n_perm)
    perm_medians = np.median(stacked, axis=0)                  # (n_perm,)
    p = float((np.sum(perm_medians >= obs_med) + 1) / (n_perm + 1))
    return {"obs_cohort_median": obs_med, "p_value": p, "n_perm": n_perm,
            "n_subjects": len(subject_obs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["broad", "narrow"], required=True)
    args = ap.parse_args()
    cfg = load_v3_config()
    outdir = OUT / args.cohort
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [coverage_subject_row(s, args.cohort, cfg) for s in V3C_SUBJECTS[args.cohort]]

    elig = [r for r in rows if r.get("eligible")]
    cohort = {}
    if elig:
        cohort = cohort_median_null([r["_obs"] for r in elig], [r["_null"] for r in elig])
        # LOSO only defined with >=2 subjects (review P2: leave-one-out of 1 -> empty vstack)
        if len(elig) >= 2:
            cohort["loso"] = [
                {"dropped": elig[k]["subject"],
                 **cohort_median_null([r["_obs"] for i, r in enumerate(elig) if i != k],
                                      [r["_null"] for i, r in enumerate(elig) if i != k])}
                for k in range(len(elig))]
            cohort["loso_status"] = "ok"
        else:
            cohort["loso"] = []
            cohort["loso_status"] = "not_enough_subjects"
        cohort["n_pass_own_null"] = int(sum(r["coverage_null_p"] < cfg["v3c"]["nulls"].get("alpha", 0.05)
                                            for r in elig if np.isfinite(r["coverage_null_p"])))

    with open(outdir / "coverage_subject.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in COLS})
    (outdir / "coverage_cohort.json").write_text(json.dumps(cohort, indent=2))
    print(f"[done] {args.cohort}: {len(elig)}/{len(rows)} eligible; cohort={cohort.get('p_value')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test + a real smoke run**

Run: `pytest tests/test_topic5_v3c_coverage.py::test_cohort_median_null_percentile -v`
Expected: PASS
Run: `python scripts/run_topic5_v3c_coverage.py --cohort broad`
Expected: prints `[done] broad: 7/7 eligible; cohort=<float>`; writes `coverage_subject.csv` (635 shows n_missed=3) + `coverage_cohort.json`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_topic5_v3c_coverage.py tests/test_topic5_v3c_coverage.py
git commit -m "feat(topic5-v3c): V3c-1 coverage runner (same-shaft + cohort-median null + LOSO)"
```

---

## Milestone C — latency assay (label-blind)

### Task 6: `first_crossing_latency` (finite / t0 / censored)

**Files:**
- Create: `src/topic5_v3c_latency.py`
- Test: `tests/test_topic5_v3c_latency.py`

**Interfaces:**
- Produces: `first_crossing_latency(z_trace_1d, relt, onset, *, z_cross, window_sec, sustain_frames) -> tuple[str, float]`.
  Window = `relt ∈ [onset, onset+window_sec]`. If the first in-window frame is already `>= z_cross` → `("t0", 0.0)` (left-censored). Else first frame `i` with `z[i:i+sustain_frames]` all `>= z_cross` → `("finite", relt[i]-onset)`. Else `("censored", nan)`.
- Also `latency_seconds(kind, sec) -> float` (finite→sec, t0→0.0, censored→nan) and
  `encode_latency_for_rank(kind, sec, *, window_sec) -> float` (finite→sec, t0→0.0, censored→window_sec+1.0).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_v3c_latency.py
import numpy as np
from src.topic5_v3c_latency import first_crossing_latency, latency_seconds, encode_latency_for_rank

RELT = np.round(np.arange(-5.0, 30.001, 0.1), 3)   # onset at 0.0

def test_first_crossing_finite():
    z = np.zeros_like(RELT); z[(RELT >= 5.0)] = 3.0            # crosses at +5s, sustained
    kind, sec = first_crossing_latency(z, RELT, 0.0, z_cross=2.0, window_sec=30.0, sustain_frames=3)
    assert kind == "finite" and abs(sec - 5.0) < 1e-6

def test_first_crossing_t0():
    z = np.full_like(RELT, 3.0)                                # already hot at onset
    kind, sec = first_crossing_latency(z, RELT, 0.0, z_cross=2.0, window_sec=30.0, sustain_frames=3)
    assert kind == "t0" and sec == 0.0

def test_first_crossing_censored_and_transient():
    z = np.zeros_like(RELT); z[(RELT >= 5.0) & (RELT < 5.15)] = 3.0   # 2 frames only -> not sustained
    kind, sec = first_crossing_latency(z, RELT, 0.0, z_cross=2.0, window_sec=30.0, sustain_frames=3)
    assert kind == "censored" and np.isnan(sec)

def test_encodings():
    assert latency_seconds("finite", 5.0) == 5.0 and latency_seconds("t0", 0.0) == 0.0
    assert np.isnan(latency_seconds("censored", float("nan")))
    assert encode_latency_for_rank("censored", float("nan"), window_sec=30.0) == 31.0
    assert encode_latency_for_rank("t0", 0.0, window_sec=30.0) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_v3c_latency.py -v`
Expected: FAIL with `ModuleNotFoundError: src.topic5_v3c_latency`

- [ ] **Step 3: Create `src/topic5_v3c_latency.py`**

```python
"""Topic 5 V3c — recruitment-latency assay, censoring, AUC (PURE, no I/O).

first_crossing_latency distinguishes finite / t0 (left-censored, already hot at
onset) / censored (never sustained-crosses in window) — the t0/censored split
IS a spec-§5.4 result, not just QC, so we keep it (detect_contact_onset_zcross
only returns detected/unreached).
"""
from __future__ import annotations

import numpy as np


def first_crossing_latency(z_trace_1d, relt, onset, *, z_cross, window_sec, sustain_frames):
    z = np.asarray(z_trace_1d, dtype=float)
    relt = np.asarray(relt, dtype=float)
    m = (relt >= onset) & (relt <= onset + window_sec)
    idx = np.nonzero(m)[0]
    if idx.size < sustain_frames:
        return ("censored", float("nan"))
    zt = z[idx]
    zt = np.where(np.isfinite(zt), zt, -np.inf)
    if zt[0] >= z_cross:
        return ("t0", 0.0)
    for i in range(zt.size - sustain_frames + 1):
        if np.all(zt[i:i + sustain_frames] >= z_cross):
            return ("finite", float(relt[idx[i]] - onset))
    return ("censored", float("nan"))


def latency_seconds(kind: str, sec: float) -> float:
    """Seconds for Δt (finite→sec, t0→0.0, censored→nan)."""
    if kind == "finite":
        return float(sec)
    if kind == "t0":
        return 0.0
    return float("nan")


def encode_latency_for_rank(kind: str, sec: float, *, window_sec: float) -> float:
    """Sortable value for AUC (finite→sec, t0→earliest 0.0, censored→last window+1)."""
    if kind == "finite":
        return float(sec)
    if kind == "t0":
        return 0.0
    return float(window_sec) + 1.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_v3c_latency.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/topic5_v3c_latency.py tests/test_topic5_v3c_latency.py
git commit -m "feat(topic5-v3c): first_crossing_latency with finite/t0/censored + rank encodings"
```

---

### Task 7: QC pure functions (`censoring_tallies`, `rank_diagnostics`, `threshold_stability`, `assay_valid`)

**Files:**
- Modify: `src/topic5_v3c_latency.py`
- Test: `tests/test_topic5_v3c_latency.py`

**Interfaces:**
- `censoring_tallies(kinds: list[str]) -> dict` → `finite_frac`, `t0_frac`, `cens_frac` (over all contacts).
- `rank_diagnostics(secs: np.ndarray) -> dict` → `uniq_ranks` (distinct finite seconds, 3-dp), `max_tie_block`.
- `threshold_stability(secs_primary: np.ndarray, secs_alt: np.ndarray) -> float` → Spearman over pairs finite in both (nan if <4 pairs or zero variance).
- `assay_valid(qc: dict, cfg: dict) -> bool` → applies the 5 gates (spec §5.2). `qc` carries `finite_frac`, `t0_frac`, `uniq_ranks_med`, `thr_spearman`, `n_informative`. **Signature takes NO `S` — label-blind.**

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_v3c_latency.py  (append)
from src.topic5_v3c_latency import censoring_tallies, rank_diagnostics, threshold_stability, assay_valid
from src.topic5_v3_mode_transition import load_v3_config

def test_censoring_tallies():
    t = censoring_tallies(["finite", "finite", "t0", "censored"])
    assert t["finite_frac"] == 0.5 and t["t0_frac"] == 0.25 and t["cens_frac"] == 0.25

def test_rank_diagnostics_ties():
    d = rank_diagnostics(np.array([1.0, 1.0, 2.0, 3.0]))
    assert d["uniq_ranks"] == 3 and d["max_tie_block"] == 2

def test_threshold_stability_monotone():
    a = np.array([1.0, 2.0, 3.0, 4.0]); b = np.array([1.1, 2.2, 2.9, 4.5])
    assert threshold_stability(a, b) > 0.9

def test_assay_valid_gates():
    cfg = load_v3_config()
    good = {"finite_frac": 0.6, "t0_frac": 0.2, "uniq_ranks_med": 8, "thr_spearman": 0.8, "n_informative": 4}
    assert assay_valid(good, cfg) is True
    bad_t0 = {**good, "t0_frac": 0.56}                         # 1077-like
    assert assay_valid(bad_t0, cfg) is False
    bad_finite = {**good, "finite_frac": 0.37}
    assert assay_valid(bad_finite, cfg) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_v3c_latency.py -k "tallies or rank_diag or threshold_stab or assay_valid" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add to `src/topic5_v3c_latency.py`**

```python
from scipy.stats import spearmanr  # noqa: E402


def censoring_tallies(kinds: list) -> dict:
    n = len(kinds)
    if n == 0:
        return {"finite_frac": float("nan"), "t0_frac": float("nan"), "cens_frac": float("nan")}
    return {
        "finite_frac": sum(k == "finite" for k in kinds) / n,
        "t0_frac": sum(k == "t0" for k in kinds) / n,
        "cens_frac": sum(k == "censored" for k in kinds) / n,
    }


def rank_diagnostics(secs) -> dict:
    finite = np.asarray(secs, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"uniq_ranks": 0, "max_tie_block": 0}
    vals, counts = np.unique(np.round(finite, 3), return_counts=True)
    return {"uniq_ranks": int(vals.size), "max_tie_block": int(counts.max())}


def threshold_stability(secs_primary, secs_alt) -> float:
    a = np.asarray(secs_primary, dtype=float); b = np.asarray(secs_alt, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 4 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return float("nan")
    return float(spearmanr(a[mask], b[mask]).correlation)


def assay_valid(qc: dict, cfg: dict) -> bool:
    """Label-blind assay gate (spec §5.2). Takes NO SOZ labels by contract."""
    g = cfg["v3c"]["assay_qc"]
    lat = cfg["v3c"]["latency"]
    return bool(
        qc["finite_frac"] >= g["finite_frac_min"]
        and qc["t0_frac"] <= g["t0_frac_max"]
        and qc["uniq_ranks_med"] >= g["uniq_ranks_min"]
        and (np.isfinite(qc["thr_spearman"]) and qc["thr_spearman"] >= g["thr_spearman_min"])
        and qc["n_informative"] >= lat["min_informative_seizures"]
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_v3c_latency.py -k "tallies or rank_diag or threshold_stab or assay_valid" -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/topic5_v3c_latency.py tests/test_topic5_v3c_latency.py
git commit -m "feat(topic5-v3c): label-blind assay-QC pure functions (tallies/ranks/threshold/gate)"
```

---

### Task 8: Latency-matrix extraction from cache (IO)

**Files:**
- Modify: `scripts/_topic5_v3c_io.py`
- Test: `tests/test_topic5_v3c_io.py`

**Interfaces:**
- Produces: `extract_latency_matrix(ds_sid, cfg, names, *, thresholds) -> list[dict]` — one dict per eligible seizure: `{"idx": si, "kinds": {thr: [kind per name]}, "secs": {thr: [sec per name]}}`. Uses `bb_zt__{si}`/`bb_relt__{si}` + `meta["seizure"][si]["eeg_onset_rel"]`; rows ordered by `names`. Window/sustain from `cfg["v3c"]`.

- [ ] **Step 1: Write the failing test** (real cache smoke — marked integration)

```python
# tests/test_topic5_v3c_io.py  (append)
import pytest
from src.topic5_v3_mode_transition import load_v3_config
from scripts._topic5_v3c_io import extract_latency_matrix

@pytest.mark.integration
def test_extract_latency_matrix_shapes():
    cfg = load_v3_config()
    names = ["GA1", "GA2", "GA3"]                              # any 3 real cache names for 139
    mats = extract_latency_matrix("epilepsiae_139", cfg, names, thresholds=[2.0, 1.5])
    assert len(mats) >= 1
    m0 = mats[0]
    assert set(m0["kinds"].keys()) == {2.0, 1.5}
    assert len(m0["kinds"][2.0]) == 3 and len(m0["secs"][2.0]) == 3
    assert all(k in ("finite", "t0", "censored") for k in m0["kinds"][2.0])

@pytest.mark.integration
def test_extract_latency_matrix_fails_closed_on_missing_contact():
    # P1-4: a contact absent from the cache MUST raise, never silently shift
    # the row->name alignment (which would misassign one contact's latency).
    cfg = load_v3_config()
    with pytest.raises(ValueError, match="absent from cache"):
        extract_latency_matrix("epilepsiae_139", cfg, ["GA1", "NOT_A_REAL_CONTACT"], thresholds=[2.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_v3c_io.py::test_extract_latency_matrix_shapes -v -m integration`
Expected: FAIL with `ImportError: cannot import name 'extract_latency_matrix'`

- [ ] **Step 3: Add to `scripts/_topic5_v3c_io.py`**

```python
from src.topic5_v3c_latency import first_crossing_latency  # noqa: E402


def extract_latency_matrix(ds_sid: str, cfg: dict, names: list, *, thresholds: list) -> list:
    """Per eligible seizure, per contact in `names`, first-crossing latency at each
    threshold (window/sustain from cfg['v3c']). Rows ordered 1:1 with `names`.

    P1-4 FAIL-CLOSED: every name MUST exist in the cache channel list. A missing
    contact raises ValueError rather than silently shifting the row->name
    alignment (which would assign one contact's latency to another — a science
    contamination bug). `names` always come from all_clean / soz_in_pool, both
    derived from cache channels, so a miss means an upstream bug, not normal data.
    """
    vc = cfg["v3c"]
    data = np.load(CACHE / f"{ds_sid}.npz", allow_pickle=True)
    meta = json.loads((CACHE / f"{ds_sid}.json").read_text())
    cache_names = [str(x) for x in data["channels"]]
    name_to_row = {n: i for i, n in enumerate(cache_names)}
    missing = [n for n in names if n not in name_to_row]
    if missing:
        raise ValueError(f"{ds_sid}: latency requested for contacts absent from cache: {missing}")
    rows = [name_to_row[n] for n in names]     # 1:1 with names (fail-closed above)
    out = []
    for si in meta.get("eligible_idxs", []):
        zk, rk = f"bb_zt__{si}", f"bb_relt__{si}"
        sz = meta.get("seizure", {}).get(str(si))
        if zk not in data.files or rk not in data.files or sz is None:
            continue
        onset = float(sz["eeg_onset_rel"])
        relt = np.asarray(data[rk], dtype=float)
        Z = np.asarray(data[zk], dtype=float)
        kinds, secs = {}, {}
        for thr in thresholds:
            kk, ss = [], []
            for r in rows:
                kind, sec = first_crossing_latency(
                    Z[r], relt, onset, z_cross=thr,
                    window_sec=vc["window_sec"], sustain_frames=vc["sustain_frames"])
                kk.append(kind); ss.append(sec)
            kinds[thr] = kk; secs[thr] = ss
        out.append({"idx": si, "kinds": kinds, "secs": secs})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_v3c_io.py -k extract_latency_matrix -v -m integration`
Expected: PASS — both `test_extract_latency_matrix_shapes` and `test_extract_latency_matrix_fails_closed_on_missing_contact` (need `results/.../ictal_field_long_cache/epilepsiae_139.npz`)

- [ ] **Step 5: Commit**

```bash
git add scripts/_topic5_v3c_io.py tests/test_topic5_v3c_io.py
git commit -m "feat(topic5-v3c): extract_latency_matrix from bb_zt cache (multi-threshold)"
```

---

### Task 9: Label-blind assay-QC runner

**Files:**
- Create: `scripts/run_topic5_v3c_latency_qc.py`
- Test: manual real run (validated against §附录 A.2 probe numbers)

**Interfaces:**
- Consumes: `classify_subject_contacts`, `axis_soz_join` (only for `n_covered`/`n_surplus` set-gate), `extract_latency_matrix`, `censoring_tallies`, `rank_diagnostics`, `threshold_stability`, `assay_valid`.
- Produces: per-subject QC JSON + `qc_subject.csv` under `<cohort>/latency_qc/`. **Computes latency ONLY on axis contacts `A` (label-blind — does not read SOZ for the QC metrics; SOZ used only to report the set-gate eligibility columns).**

- [ ] **Step 1: Write the runner**

```python
"""V3c-2 pre-check: label-blind latency assay-QC on axis contacts (spec §5.2).
Emits QC metrics that gate whether latency is a mechanistic endpoint. Does NOT
compute any SOZ-vs-surplus contrast (endpoint-blind).
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._topic5_v3_io import classify_subject_contacts
from scripts._topic5_v3c_io import V3C_SUBJECTS, extract_latency_matrix, axis_soz_join, load_soz
from src.topic5_v3_mode_transition import load_v3_config
from src.topic5_v3c_latency import (assay_valid, censoring_tallies, rank_diagnostics,
                                    threshold_stability)

OUT = _ROOT / "results/topic5_ictal_recruitment/v3c_soz_axis_coverage"
COLS = ["subject", "cohort", "n_axis", "n_covered", "n_surplus", "n_sz_used",
        "finite_frac", "t0_frac", "cens_frac", "uniq_ranks_med", "max_tie_med",
        "thr_spearman", "n_informative", "sz_stability_std", "assay_valid", "cens_flag"]


def qc_subject(ds_sid: str, cohort: str, cfg: dict) -> dict:
    dataset, subj = ds_sid.split("_", 1)
    vc = cfg["v3c"]
    row = {c: float("nan") for c in COLS}; row.update({"subject": ds_sid, "cohort": cohort})
    try:
        cls = classify_subject_contacts(ds_sid, cohort, cfg)
        A = cls["is_axis"]
        j = axis_soz_join(cls, load_soz(dataset, subj))
        thrs = [vc["z_cross"]] + list(vc["z_cross_sensitivity"])
        mats = extract_latency_matrix(ds_sid, cfg, A, thresholds=thrs)
        all_kinds, uniq_l, tie_l, rho_l, szmed_l, n_info = [], [], [], [], [], 0
        for m in mats:
            kinds = m["kinds"][vc["z_cross"]]; secs = np.array(m["secs"][vc["z_cross"]], float)
            all_kinds += kinds
            rd = rank_diagnostics(secs); uniq_l.append(rd["uniq_ranks"]); tie_l.append(rd["max_tie_block"])
            fin = secs[np.isfinite(secs)]
            if fin.size:
                szmed_l.append(float(np.median(fin)))
            if rd["uniq_ranks"] >= vc["assay_qc"]["informative_min_unique_ranks"] and \
               any(k == "finite" for k in kinds):
                n_info += 1
            for alt in vc["z_cross_sensitivity"]:
                rho = threshold_stability(secs, np.array(m["secs"][alt], float))
                if np.isfinite(rho):
                    rho_l.append(rho)
        tal = censoring_tallies(all_kinds)
        qc = {"finite_frac": tal["finite_frac"], "t0_frac": tal["t0_frac"],
              "uniq_ranks_med": float(np.median(uniq_l)) if uniq_l else 0.0,
              "thr_spearman": float(np.median(rho_l)) if rho_l else float("nan"),
              "n_informative": n_info}
        row.update({
            "n_axis": len(A), "n_covered": j["n_covered"], "n_surplus": j["n_surplus"],
            "n_sz_used": len(mats), "finite_frac": tal["finite_frac"], "t0_frac": tal["t0_frac"],
            "cens_frac": tal["cens_frac"], "uniq_ranks_med": qc["uniq_ranks_med"],
            "max_tie_med": float(np.median(tie_l)) if tie_l else float("nan"),
            "thr_spearman": qc["thr_spearman"], "n_informative": n_info,
            "sz_stability_std": float(np.std(szmed_l)) if len(szmed_l) >= 2 else float("nan"),
            "assay_valid": bool(assay_valid(qc, cfg)),
            "cens_flag": bool(tal["cens_frac"] > vc["assay_qc"]["cens_frac_flag"]),
        })
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] {ds_sid} ({cohort}): {type(exc).__name__}: {exc}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["broad", "narrow"], required=True)
    args = ap.parse_args()
    cfg = load_v3_config()
    outdir = OUT / args.cohort / "latency_qc"; outdir.mkdir(parents=True, exist_ok=True)
    rows = [qc_subject(s, args.cohort, cfg) for s in V3C_SUBJECTS[args.cohort]]
    with open(outdir / "qc_subject.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader()
        for r in rows: w.writerow({c: r[c] for c in COLS})
    print(f"[done] {args.cohort} assay-QC; valid={[r['subject'] for r in rows if r.get('assay_valid')]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run + validate against probe baseline**

Run: `python scripts/run_topic5_v3c_latency_qc.py --cohort broad`
Expected: `qc_subject.csv` where 139 t0_frac≈0.24, 1077 t0_frac≈0.56 & assay_valid=False, 1150 cens_flag=True; `assay_valid` True for ~4 (139,253,635,1096). (Matches spec §附录 A.2.)

- [ ] **Step 3: Commit**

```bash
git add scripts/run_topic5_v3c_latency_qc.py
git commit -m "feat(topic5-v3c): label-blind latency assay-QC runner (validated vs probe)"
```

---

## Milestone D — V3c-2 latency (gated)

### Task 10: `auc_late` + `delta_t` + `auc_null_distribution`

**Files:**
- Modify: `src/topic5_v3c_latency.py`
- Test: `tests/test_topic5_v3c_latency.py`

**Interfaces:**
- `auc_late(surplus_vals: np.ndarray, soz_vals: np.ndarray) -> float` = `P(surplus>soz) + 0.5·P(surplus==soz)` over all pairs (vals = `encode_latency_for_rank` outputs). nan if either empty.
- `delta_t(surplus_secs: np.ndarray, soz_secs: np.ndarray) -> float` = `nanmedian(surplus_secs) − nanmedian(soz_secs)` (secs = `latency_seconds`; censored already nan).
- `auc_null_distribution(surplus_vals, soz_vals, shaft_by_name, surplus_names, soz_names, *, n_perm, rng) -> np.ndarray` — within-shaft relabel of surplus/soz over `A∩S ∪ A∖S` (via `label_permute`), recompute `auc_late` (spec §5.5 primary label null).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_v3c_latency.py  (append)
from src.topic5_v3c_latency import auc_late, delta_t, auc_null_distribution

def test_auc_late_direction():
    soz = np.array([1.0, 2.0, 3.0]); surplus = np.array([4.0, 5.0, 6.0])  # surplus later
    assert auc_late(surplus, soz) == 1.0                                   # H-B extreme
    assert auc_late(soz, surplus) == 0.0                                   # surplus earlier
    assert auc_late(np.array([2.0, 2.0]), np.array([2.0, 2.0])) == 0.5     # all ties

def test_delta_t_seconds():
    assert delta_t(np.array([5.0, 7.0]), np.array([1.0, 3.0])) == 4.0
    assert np.isfinite(delta_t(np.array([5.0, np.nan]), np.array([1.0])))  # nan (censored) skipped

def test_auc_null_preserves_and_varies():
    shaft = {n: "H" for n in ["H1", "H2", "H3", "H4"]}
    surplus_names = ["H1", "H2"]; soz_names = ["H3", "H4"]
    sv = np.array([10.0, 10.0]); zv = np.array([0.0, 0.0])                 # obs AUC=1.0
    null = auc_null_distribution(sv, zv, shaft, surplus_names, soz_names, n_perm=200, rng=0)
    assert null.shape == (200,) and 0.0 <= np.median(null) <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_v3c_latency.py -k "auc or delta_t" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add to `src/topic5_v3c_latency.py`**

```python
from src.topic5_v3_mode_transition import _coerce_rng, label_permute  # noqa: E402


def auc_late(surplus_vals, soz_vals) -> float:
    s = np.asarray(surplus_vals, dtype=float); z = np.asarray(soz_vals, dtype=float)
    if s.size == 0 or z.size == 0:
        return float("nan")
    gt = np.sum(s[:, None] > z[None, :])
    eq = np.sum(s[:, None] == z[None, :])
    return float((gt + 0.5 * eq) / (s.size * z.size))


def delta_t(surplus_secs, soz_secs) -> float:
    s = np.asarray(surplus_secs, dtype=float); z = np.asarray(soz_secs, dtype=float)
    return float(np.nanmedian(s) - np.nanmedian(z))


def auc_null_distribution(surplus_vals, soz_vals, shaft_by_name, surplus_names, soz_names,
                          *, n_perm, rng) -> np.ndarray:
    """Within-shaft relabel of surplus/soz-core over A∩S ∪ A∖S, preserving per-shaft
    surplus count; recompute auc_late (spec §5.5 primary label null)."""
    rng = _coerce_rng(rng)
    val_by_name = {**{n: float(v) for n, v in zip(surplus_names, surplus_vals)},
                   **{n: float(v) for n, v in zip(soz_names, soz_vals)}}
    out = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        new_surplus, new_soz = label_permute(surplus_names, soz_names, shaft_by_name, rng)
        out[i] = auc_late(np.array([val_by_name[n] for n in new_surplus]),
                          np.array([val_by_name[n] for n in new_soz]))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_v3c_latency.py -k "auc or delta_t" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/topic5_v3c_latency.py tests/test_topic5_v3c_latency.py
git commit -m "feat(topic5-v3c): auc_late + delta_t + within-shaft AUC null"
```

---

### Task 11: V3c-2 latency runner (gated on set-thresholds + assay_valid)

**Files:**
- Create: `scripts/run_topic5_v3c_latency.py`
- Test: `tests/test_topic5_v3c_latency.py` (gating unit) + real run

**Interfaces:**
- Consumes: `classify_subject_contacts`, `axis_soz_join`, `load_soz`, `extract_latency_matrix`, `encode_latency_for_rank`, `latency_seconds`, `auc_late`, `delta_t`, `auc_null_distribution`, plus the QC CSV from Task 9 (reads `assay_valid` per subject).
- Produces: `latency_eligible(join, qc_valid, cfg) -> bool` (set-gate `|A∖S|≥min_surplus ∧ |A∩S|≥min_covered_soz ∧ assay_valid`). Per-subject: per-seizure `AUC_late` (primary `A∩S` vs `A∖S`; clinical-sensitivity `S` vs `A∖S`; censor-sensitivity `drop_censored`/`exclude_t0`), subject-median AUC + signed `Δt`, within-shaft null percentile. **`latency_cohort.json` MUST carry `obs_cohort_median_auc`, `p_value`, `auc_null_q05/q50/q95`, `delta_t_med` (SIGNED), `subject_delta_t`, `auc_drop_censored_med`, `auc_exclude_t0_med`, `sensitivity_concordant`, `subject_aucs`** — these are the contract Task 14 consumes (missing → Task 14 fails closed).

- [ ] **Step 1: Write the failing test (gating logic)**

```python
# tests/test_topic5_v3c_latency.py  (append)
from scripts.run_topic5_v3c_latency import latency_eligible
from src.topic5_v3_mode_transition import load_v3_config

def test_latency_eligible_gate():
    cfg = load_v3_config()
    ok = {"n_surplus": 10, "n_covered": 4}
    assert latency_eligible(ok, True, cfg) is True
    assert latency_eligible(ok, False, cfg) is False           # assay invalid -> descriptive only
    assert latency_eligible({"n_surplus": 2, "n_covered": 4}, True, cfg) is False   # too few surplus
    assert latency_eligible({"n_surplus": 10, "n_covered": 2}, True, cfg) is False  # too few A∩S
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_v3c_latency.py::test_latency_eligible_gate -v`
Expected: FAIL with `ModuleNotFoundError: scripts.run_topic5_v3c_latency`

- [ ] **Step 3: Create `scripts/run_topic5_v3c_latency.py`**

```python
"""V3c-2 (gated mechanistic secondary): ictal recruitment timing of axis-surplus.
Primary contrast A∩S vs A∖S; sensitivity S vs A∖S. Gated on set-thresholds AND
label-blind assay_valid (from latency_qc). Subject-first, within-shaft AUC null.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._topic5_v3_io import classify_subject_contacts
from scripts._topic5_v3c_io import V3C_SUBJECTS, extract_latency_matrix, axis_soz_join, load_soz
from src.topic5_v3_mode_transition import load_v3_config
from src.topic5_v3c_latency import (auc_late, auc_null_distribution, delta_t,
                                    encode_latency_for_rank, latency_seconds)

OUT = _ROOT / "results/topic5_ictal_recruitment/v3c_soz_axis_coverage"
COLS = ["subject", "cohort", "n_covered", "n_surplus", "n_sz_used", "auc_primary",
        "auc_drop_censored", "auc_exclude_t0", "auc_sensitivity_allsoz",
        "delta_t_sec", "auc_null_p", "eligible"]


def latency_eligible(join: dict, qc_valid: bool, cfg: dict) -> bool:
    lat = cfg["v3c"]["latency"]
    return bool(qc_valid and join["n_surplus"] >= lat["min_surplus"]
                and join["n_covered"] >= lat["min_covered_soz"])


def _assay_valid_map(cohort: str) -> dict:
    p = OUT / cohort / "latency_qc" / "qc_subject.csv"
    out = {}
    if p.exists():
        import csv as _csv
        for r in _csv.DictReader(p.open()):
            out[r["subject"]] = (r["assay_valid"] == "True")
    return out


def _auc_variant(by, group_soz, group_surplus, window_sec, mode):
    """AUC_late(surplus,soz) under a censoring-sensitivity mode (spec §5.4):
    'primary' (censored->last, t0->first), 'drop_censored' (exclude censored from
    both groups), 'exclude_t0' (exclude t0 from both groups). nan if a group empties.
    """
    def keep(n):
        k = by[n][0]
        if mode == "drop_censored":
            return k != "censored"
        if mode == "exclude_t0":
            return k != "t0"
        return True
    su = [n for n in group_surplus if keep(n)]
    so = [n for n in group_soz if keep(n)]
    sv = np.array([encode_latency_for_rank(*by[n], window_sec=window_sec) for n in su])
    zv = np.array([encode_latency_for_rank(*by[n], window_sec=window_sec) for n in so])
    return auc_late(sv, zv)


def latency_subject_row(ds_sid, cohort, cfg, qc_map) -> dict:
    dataset, subj = ds_sid.split("_", 1)
    vc = cfg["v3c"]; row = {c: float("nan") for c in COLS}
    row.update({"subject": ds_sid, "cohort": cohort, "eligible": False})
    try:
        cls = classify_subject_contacts(ds_sid, cohort, cfg)
        j = axis_soz_join(cls, load_soz(dataset, subj))
        eligible = latency_eligible(j, qc_map.get(ds_sid, False), cfg)
        covered, surplus, soz_all = j["covered"], j["surplus"], j["soz_in_pool"]
        names = covered + surplus + [n for n in soz_all if n not in set(covered)]
        mats = extract_latency_matrix(ds_sid, cfg, names, thresholds=[vc["z_cross"]])
        aucs_p, aucs_dc, aucs_xt0, aucs_s, dts, nulls = [], [], [], [], [], []
        for m in mats:
            kinds = m["kinds"][vc["z_cross"]]; secs = m["secs"][vc["z_cross"]]
            by = {n: (kinds[i], secs[i]) for i, n in enumerate(names)}
            aucs_p.append(_auc_variant(by, covered, surplus, vc["window_sec"], "primary"))
            aucs_dc.append(_auc_variant(by, covered, surplus, vc["window_sec"], "drop_censored"))
            aucs_xt0.append(_auc_variant(by, covered, surplus, vc["window_sec"], "exclude_t0"))
            aucs_s.append(_auc_variant(by, soz_all, surplus, vc["window_sec"], "primary"))  # clinical sensitivity: S vs A∖S
            sv = np.array([latency_seconds(*by[n]) for n in surplus])   # signed Δt: surplus − covered (>0 = surplus later = H-B)
            zv = np.array([latency_seconds(*by[n]) for n in covered])
            dts.append(delta_t(sv, zv))
            if eligible:
                svr = np.array([encode_latency_for_rank(*by[n], window_sec=vc["window_sec"]) for n in surplus])
                zvr = np.array([encode_latency_for_rank(*by[n], window_sec=vc["window_sec"]) for n in covered])
                nulls.append(auc_null_distribution(svr, zvr, cls["shaft_by_name"], surplus, covered,
                                                   n_perm=vc["nulls"]["n_perm"], rng=vc["nulls"]["seed"]))
        auc_p = float(np.nanmedian(aucs_p)) if aucs_p else float("nan")
        dt_med = float(np.nanmedian(dts)) if dts else float("nan")
        null_med = np.median(np.vstack(nulls), axis=0) if nulls else np.array([])
        p = float((np.sum(null_med >= auc_p) + 1) / (null_med.size + 1)) if null_med.size else float("nan")
        row.update({"n_covered": j["n_covered"], "n_surplus": j["n_surplus"], "n_sz_used": len(mats),
                    "auc_primary": auc_p,
                    "auc_drop_censored": float(np.nanmedian(aucs_dc)) if aucs_dc else float("nan"),
                    "auc_exclude_t0": float(np.nanmedian(aucs_xt0)) if aucs_xt0 else float("nan"),
                    "auc_sensitivity_allsoz": float(np.nanmedian(aucs_s)) if aucs_s else float("nan"),
                    "delta_t_sec": dt_med, "auc_null_p": p, "eligible": bool(eligible),
                    "_auc": auc_p, "_dt": dt_med, "_null_med": null_med,
                    "_auc_dc": float(np.nanmedian(aucs_dc)) if aucs_dc else float("nan"),
                    "_auc_xt0": float(np.nanmedian(aucs_xt0)) if aucs_xt0 else float("nan")})
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] {ds_sid} ({cohort}): {type(exc).__name__}: {exc}", flush=True)
    return row


def _same_side(a, b, band):
    lo, hi = band
    side = lambda x: (1 if x > hi else (-1 if x < lo else 0)) if np.isfinite(x) else 99
    return side(a) == side(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["broad", "narrow"], required=True)
    args = ap.parse_args()
    cfg = load_v3_config(); qc_map = _assay_valid_map(args.cohort)
    band = cfg["v3c"]["interpretation"]["auc_ha_band"]
    outdir = OUT / args.cohort / "latency"; outdir.mkdir(parents=True, exist_ok=True)
    rows = [latency_subject_row(s, args.cohort, cfg, qc_map) for s in V3C_SUBJECTS[args.cohort]]
    elig = [r for r in rows if r.get("eligible") and np.isfinite(r.get("_auc", float("nan")))]
    cohort = {}
    if elig:
        n_perm = min(r["_null_med"].size for r in elig)
        perm_med = np.median(np.vstack([r["_null_med"][:n_perm] for r in elig]), axis=0)
        obs = float(np.median([r["_auc"] for r in elig]))
        dc_med = float(np.median([r["_auc_dc"] for r in elig]))
        xt0_med = float(np.median([r["_auc_xt0"] for r in elig]))
        cohort = {
            "obs_cohort_median_auc": obs, "n_perm": int(n_perm), "n_subjects": len(elig),
            "p_value": float((np.sum(perm_med >= obs) + 1) / (n_perm + 1)),
            "auc_null_q05": float(np.percentile(perm_med, 5)),
            "auc_null_q50": float(np.percentile(perm_med, 50)),
            "auc_null_q95": float(np.percentile(perm_med, 95)),
            "subject_aucs": {r["subject"]: r["_auc"] for r in elig},
            # P1-2: Δt aggregation (H-B needs the SIGNED cohort median; missing -> summary fails closed)
            "delta_t_med": float(np.median([r["_dt"] for r in elig])),
            "subject_delta_t": {r["subject"]: r["_dt"] for r in elig},
            # P1-5: censor/t0 sensitivity AUC cohort medians + sign concordance vs primary
            "auc_drop_censored_med": dc_med, "auc_exclude_t0_med": xt0_med,
            "sensitivity_concordant": bool(_same_side(obs, dc_med, band) and _same_side(obs, xt0_med, band)),
        }
    with open(outdir / "latency_subject.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader()
        for r in rows: w.writerow({c: r[c] for c in COLS})
    (outdir / "latency_cohort.json").write_text(json.dumps(cohort, indent=2))
    print(f"[done] {args.cohort} latency: {len(elig)} eligible; cohort_auc={cohort.get('obs_cohort_median_auc')} "
          f"delta_t_med={cohort.get('delta_t_med')} concordant={cohort.get('sensitivity_concordant')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test + real run**

Run: `pytest tests/test_topic5_v3c_latency.py::test_latency_eligible_gate -v`
Expected: PASS
Run: `python scripts/run_topic5_v3c_latency_qc.py --cohort broad && python scripts/run_topic5_v3c_latency.py --cohort broad`
Expected: `latency_subject.csv` (1077 eligible=False via assay gate), `latency_cohort.json` with `obs_cohort_median_auc`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_topic5_v3c_latency.py tests/test_topic5_v3c_latency.py
git commit -m "feat(topic5-v3c): V3c-2 gated latency runner (AUC A∩S vs A∖S + within-shaft null)"
```

---

## Milestone E — V3c-3 surplus spatial organization

### Task 12: `surplus_spatial_metrics` + `distance_null_distribution`

**Files:**
- Modify: `src/topic5_v3c_coverage.py`
- Test: `tests/test_topic5_v3c_coverage.py`

**Interfaces:**
- `surplus_spatial_metrics(surplus_names, soz_names, coords_by_name, shaft_by_name) -> dict` → `n_shafts_with_surplus`, `shaft_gini` (Gini of surplus counts per shaft), `max_contiguous_run` (longest run of consecutive surplus on any shaft, by trailing integer), `mean_min_dist_to_soz` (nan if coords missing).
- `distance_null_distribution(surplus_names, axis_names, soz_names, coords_by_name, shaft_by_name, *, n_perm, rng) -> np.ndarray` — same-shaft relabel of surplus over axis, recompute `mean_min_dist_to_soz` (spec §6 distance null; empty array if no coords).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_v3c_coverage.py  (append)
from src.topic5_v3c_coverage import surplus_spatial_metrics, distance_null_distribution

def test_surplus_spatial_contiguous_and_distance():
    coords = {"H1": np.array([0., 0, 0]), "H2": np.array([1., 0, 0]), "H3": np.array([2., 0, 0]),
              "S1": np.array([0., 0, 0])}
    m = surplus_spatial_metrics(["H1", "H2", "H3"], ["S1"], coords, {n: "H" for n in ["H1", "H2", "H3"]})
    assert m["n_shafts_with_surplus"] == 1 and m["max_contiguous_run"] == 3
    assert m["mean_min_dist_to_soz"] == (0.0 + 1.0 + 2.0) / 3

def test_distance_null_empty_without_coords():
    null = distance_null_distribution(["H1"], ["H1", "H2"], ["S1"], {}, {"H1": "H", "H2": "H"},
                                      n_perm=50, rng=0)
    assert null.size == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_v3c_coverage.py -k "spatial or distance_null" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add to `src/topic5_v3c_coverage.py`**

```python
def _shaft_and_num(name):
    num = "".join(c for c in name if c.isdigit())
    return name[: len(name) - len(num)], (int(num) if num else -1)


def _gini(counts) -> float:
    x = np.sort(np.asarray(counts, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.sum((np.arange(1, n + 1)) * x) - (n + 1) * x.sum()) / (n * x.sum()))


def _mean_min_dist(surplus_names, soz_names, coords_by_name) -> float:
    sc = [coords_by_name[n] for n in soz_names if n in coords_by_name]
    su = [coords_by_name[n] for n in surplus_names if n in coords_by_name]
    if not sc or not su:
        return float("nan")
    sc = np.vstack(sc)
    return float(np.mean([np.min(np.linalg.norm(sc - p[None, :], axis=1)) for p in su]))


def surplus_spatial_metrics(surplus_names, soz_names, coords_by_name, shaft_by_name) -> dict:
    per_shaft = {}
    for n in surplus_names:
        per_shaft.setdefault(shaft_by_name[n], []).append(_shaft_and_num(n)[1])
    max_run = 0
    for nums in per_shaft.values():
        s = sorted(x for x in nums if x >= 0)
        run = best = 1 if s else 0
        for a, b in zip(s, s[1:]):
            run = run + 1 if b == a + 1 else 1
            best = max(best, run)
        max_run = max(max_run, best)
    return {
        "n_shafts_with_surplus": len(per_shaft),
        "shaft_gini": _gini([len(v) for v in per_shaft.values()]),
        "max_contiguous_run": int(max_run),
        "mean_min_dist_to_soz": _mean_min_dist(surplus_names, soz_names, coords_by_name),
    }


def distance_null_distribution(surplus_names, axis_names, soz_names, coords_by_name,
                               shaft_by_name, *, n_perm, rng) -> np.ndarray:
    if not coords_by_name or not any(n in coords_by_name for n in soz_names):
        return np.array([])
    rng = _coerce_rng(rng)
    covered = [n for n in axis_names if n not in set(surplus_names)]
    out = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        new_surplus, _ = label_permute(surplus_names, covered, shaft_by_name, rng)
        out[i] = _mean_min_dist(new_surplus, soz_names, coords_by_name)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_v3c_coverage.py -k "spatial or distance_null" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/topic5_v3c_coverage.py tests/test_topic5_v3c_coverage.py
git commit -m "feat(topic5-v3c): surplus spatial metrics + distance-to-SOZ null (coords-gated)"
```

---

### Task 13: V3c-3 surplus-spatial runner (coords via seeg_coord_loader, graceful skip)

**Files:**
- Create: `scripts/run_topic5_v3c_surplus_spatial.py`
- Modify: `scripts/_topic5_v3c_io.py` (add `load_axis_coords`)
- Test: `tests/test_topic5_v3c_io.py`

**Interfaces:**
- Produces: `load_axis_coords(dataset, subject, names) -> dict[str, np.ndarray]` — wraps `seeg_coord_loader.load_subject_coords`; returns `{}` on `FileNotFoundError`/`ValueError` (MRI/SQL missing) so V3c-3 falls back to shaft-only metrics (no distance null). Runner writes `<cohort>/surplus_spatial/surplus_subject.csv` **and `surplus_spatial_cohort.json`** (`p_value` = cohort-median distance lower-tail null, `n_spatial_eligible`, `loso`) — Task 14 reads this for the coverage double-condition (§4.4).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_v3c_io.py  (append)
from scripts._topic5_v3c_io import load_axis_coords

def test_load_axis_coords_missing_returns_empty(monkeypatch):
    import scripts._topic5_v3c_io as io
    def boom(*a, **k):
        raise FileNotFoundError("no MRI")
    monkeypatch.setattr(io, "load_subject_coords", boom)
    assert load_axis_coords("epilepsiae", "999", ["A1"]) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_v3c_io.py::test_load_axis_coords_missing_returns_empty -v`
Expected: FAIL with `ImportError: cannot import name 'load_axis_coords'`

- [ ] **Step 3: Add `load_axis_coords` to `scripts/_topic5_v3c_io.py` + create the runner**

```python
# add to scripts/_topic5_v3c_io.py
from src.seeg_coord_loader import load_subject_coords  # noqa: E402


def load_axis_coords(dataset: str, subject: str, names: list) -> dict:
    """{name: ras_mm coord} for `names`; {} if MRI/SQL missing (V3c-3 falls back
    to shaft-only metrics — no silent coord fabrication)."""
    try:
        res = load_subject_coords(dataset, subject, names)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[coords-skip] {dataset}_{subject}: {type(exc).__name__}: {exc}", flush=True)
        return {}
    out = {}
    coords = res.coords_array_in_requested_order      # (n, 3), NaN for missing
    mask = res.mapped_mask_in_requested_order          # (n,) bool, index-aligned to names
    for i, n in enumerate(names):
        if bool(mask[i]) and np.all(np.isfinite(coords[i])):
            out[n] = np.asarray(coords[i], dtype=float)
    return out
```

```python
# scripts/run_topic5_v3c_surplus_spatial.py
"""V3c-3 (secondary descriptive): spatial organization of axis-surplus A∖S.
Shaft spread + contiguous runs + distance-to-SOZ (coords-gated) vs same-shaft null.
Emits surplus_spatial_cohort.json (cohort-median distance null) for the coverage
DOUBLE-condition in Task 14 (spec §4.4).
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._topic5_v3_io import classify_subject_contacts
from scripts._topic5_v3c_io import V3C_SUBJECTS, axis_soz_join, load_axis_coords, load_soz
from src.topic5_v3_mode_transition import load_v3_config
from src.topic5_v3c_coverage import distance_null_distribution, surplus_spatial_metrics

OUT = _ROOT / "results/topic5_ictal_recruitment/v3c_soz_axis_coverage"
COLS = ["subject", "cohort", "n_surplus", "n_shafts_with_surplus", "shaft_gini",
        "max_contiguous_run", "mean_min_dist_to_soz", "dist_null_p"]


def surplus_row(ds_sid, cohort, cfg) -> dict:
    dataset, subj = ds_sid.split("_", 1)
    row = {c: float("nan") for c in COLS}; row.update({"subject": ds_sid, "cohort": cohort})
    try:
        cls = classify_subject_contacts(ds_sid, cohort, cfg)
        j = axis_soz_join(cls, load_soz(dataset, subj))
        coords = load_axis_coords(dataset, subj, cls["all_clean"])
        m = surplus_spatial_metrics(j["surplus"], j["soz_in_pool"], coords, cls["shaft_by_name"])
        null = distance_null_distribution(j["surplus"], cls["is_axis"], j["soz_in_pool"], coords,
                                          cls["shaft_by_name"], n_perm=cfg["v3c"]["nulls"]["n_perm"],
                                          rng=cfg["v3c"]["nulls"]["seed"])
        # lower observed distance than null -> surplus closer to SOZ than random -> structured
        p = (float((np.sum(null <= m["mean_min_dist_to_soz"]) + 1) / (null.size + 1))
             if null.size and np.isfinite(m["mean_min_dist_to_soz"]) else float("nan"))
        row.update({"n_surplus": j["n_surplus"], **{k: m[k] for k in
                    ("n_shafts_with_surplus", "shaft_gini", "max_contiguous_run", "mean_min_dist_to_soz")},
                    "dist_null_p": p, "_null": null, "_obs": m["mean_min_dist_to_soz"]})
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] {ds_sid} ({cohort}): {type(exc).__name__}: {exc}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["broad", "narrow"], required=True)
    args = ap.parse_args()
    cfg = load_v3_config()
    outdir = OUT / args.cohort / "surplus_spatial"; outdir.mkdir(parents=True, exist_ok=True)
    rows = [surplus_row(s, args.cohort, cfg) for s in V3C_SUBJECTS[args.cohort]]
    with open(outdir / "surplus_subject.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader()
        for r in rows: w.writerow({c: r[c] for c in COLS})

    # P1-1: cohort-median distance null feeds the coverage DOUBLE-condition (spec §4.4).
    # A subject is spatial-eligible only if it has coords -> a finite distance null.
    elig = [r for r in rows if isinstance(r.get("_null"), np.ndarray) and r["_null"].size
            and np.isfinite(r.get("_obs", float("nan")))]
    cohort = {"n_spatial_eligible": len(elig), "n_subjects": len(rows)}
    if elig:
        n_perm = min(r["_null"].size for r in elig)
        perm_med = np.median(np.vstack([r["_null"][:n_perm] for r in elig]), axis=0)
        obs = float(np.median([r["_obs"] for r in elig]))
        cohort.update({
            "obs_cohort_median_dist": obs, "n_perm": int(n_perm),
            "p_value": float((np.sum(perm_med <= obs) + 1) / (n_perm + 1)),   # lower-tail (closer = structured)
            "dist_null_q05": float(np.percentile(perm_med, 5)),
            "dist_null_q50": float(np.percentile(perm_med, 50)),
            "dist_null_q95": float(np.percentile(perm_med, 95)),
        })
        # LOSO only defined with >=2 subjects (review P2: leave-one-out of 1 -> empty vstack)
        if len(elig) >= 2:
            cohort["loso"] = [
                {"dropped": elig[k]["subject"],
                 "p_value": float((np.sum(
                     np.median(np.vstack([r["_null"][:n_perm] for i, r in enumerate(elig) if i != k]), axis=0)
                     <= np.median([r["_obs"] for i, r in enumerate(elig) if i != k])) + 1) / (n_perm + 1))}
                for k in range(len(elig))]
            cohort["loso_status"] = "ok"
        else:
            cohort["loso"] = []
            cohort["loso_status"] = "not_enough_subjects"
    (outdir / "surplus_spatial_cohort.json").write_text(json.dumps(cohort, indent=2))
    print(f"[done] {args.cohort} surplus-spatial ({len(rows)} subjects, {len(elig)} spatial-eligible); "
          f"cohort_dist_p={cohort.get('p_value')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test + real run**

Run: `pytest tests/test_topic5_v3c_io.py::test_load_axis_coords_missing_returns_empty -v`
Expected: PASS
Run: `python scripts/run_topic5_v3c_surplus_spatial.py --cohort broad`
Expected: `surplus_subject.csv`; `[coords-skip]` lines are acceptable (distance columns = nan, shaft columns finite).

- [ ] **Step 5: Commit**

```bash
git add scripts/run_topic5_v3c_surplus_spatial.py scripts/_topic5_v3c_io.py tests/test_topic5_v3c_io.py
git commit -m "feat(topic5-v3c): V3c-3 surplus spatial runner + coord loader (graceful skip)"
```

---

## Milestone F — summary (tier + claim language) + figures

### Task 14: `run_topic5_v3c_summary.py` — verdict + claim-language selector

**Files:**
- Create: `scripts/run_topic5_v3c_summary.py`
- Test: `tests/test_topic5_v3c_latency.py` (verdict pure fn)

**Interfaces:**
- Produces: `interpret_latency(cohort_auc, subject_aucs, delta_t_med, null_p, sensitivity_concordant, cfg) -> str` returning one of `"H-B_supported"`, `"H-A_compatible"`, `"surplus_earlier_unverified"`, `"indeterminate"` (spec §5.6 + review P1-3 signed Δt / P1-5 concordance). Reads coverage/latency/**surplus_spatial** cohort JSONs; **enforces the coverage double-condition** (`coverage_primary_pass = coverage_sig ∧ spatial_sig`, `coverage_claim_level ∈ {none, beyond_implantation_geometry, specific_axis_soz_organization}`, R2-pinned) and **fails closed** if `latency_cohort.json` lacks `delta_t_med`/`sensitivity_concordant`. Writes `<cohort>/v3c_summary.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_v3c_latency.py  (append)
from scripts.run_topic5_v3c_summary import interpret_latency
from src.topic5_v3_mode_transition import load_v3_config

def test_interpret_latency_four_way():
    cfg = load_v3_config()
    # H-B: AUC>=0.60, majority>0.55, SIGNED delta_t>=+2, null sig, sensitivity concordant
    assert interpret_latency(0.66, [0.6, 0.7, 0.58, 0.62], 3.0, 0.01, True, cfg) == "H-B_supported"
    # P1-3: identical but delta_t NEGATIVE (contradictory) -> NOT H-B (abs() would wrongly pass)
    assert interpret_latency(0.66, [0.6, 0.7, 0.58, 0.62], -3.0, 0.01, True, cfg) != "H-B_supported"
    # P1-5: H-B numbers but censor/t0 sensitivity NOT concordant -> NOT H-B
    assert interpret_latency(0.66, [0.6, 0.7, 0.58, 0.62], 3.0, 0.01, False, cfg) != "H-B_supported"
    # H-A compatible (descriptive): AUC in [0.45,0.55], small |delta_t|
    assert interpret_latency(0.50, [0.49, 0.51, 0.50], 0.5, 0.6, True, cfg) == "H-A_compatible"
    # surplus EARLIER (low tail): AUC<=0.40 and delta_t<=-2 -> distinct category, not indeterminate
    assert interpret_latency(0.34, [0.3, 0.35, 0.4], -3.0, 0.2, True, cfg) == "surplus_earlier_unverified"
    # indeterminate: mixed direction, null not sig, outside HA band
    assert interpret_latency(0.57, [0.7, 0.4, 0.6], 1.0, 0.3, True, cfg) == "indeterminate"

def test_spatial_primary_ok_requires_min_subjects():
    from scripts.run_topic5_v3c_summary import _spatial_primary_ok
    cfg = load_v3_config()
    assert _spatial_primary_ok({"n_spatial_eligible": 4, "p_value": 0.01}, cfg) is True
    assert _spatial_primary_ok({"n_spatial_eligible": 1, "p_value": 0.01}, cfg) is False  # too few subjects
    assert _spatial_primary_ok({"n_spatial_eligible": 4, "p_value": 0.20}, cfg) is False  # null not sig
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_v3c_latency.py::test_interpret_latency_four_way -v`
Expected: FAIL with `ModuleNotFoundError: scripts.run_topic5_v3c_summary`

- [ ] **Step 3: Create `scripts/run_topic5_v3c_summary.py`**

```python
"""V3c summary: coverage double-condition verdict + gated latency interpretation
(H-B_supported / H-A_compatible / indeterminate, spec §5.6, R3 descriptive H-A)
+ claim-language string. broad primary / narrow sensitivity (separate files).
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.topic5_v3_mode_transition import load_v3_config

OUT = _ROOT / "results/topic5_ictal_recruitment/v3c_soz_axis_coverage"


def interpret_latency(cohort_auc, subject_aucs, delta_t_med, null_p, sensitivity_concordant, cfg) -> str:
    """4-way (spec §5.6 + review P1-3/P1-5). H-B needs SIGNED delta_t>=+thr (surplus
    later) AND censor/t0 sensitivity concordant. surplus_earlier_unverified is the
    low-tail (surplus earlier) case — kept distinct from indeterminate. H-A is a
    DESCRIPTIVE compatibility statement (R3), never proven equivalence.
    """
    it = cfg["v3c"]["interpretation"]; alpha = 0.05
    subj = np.asarray(subject_aucs, dtype=float)
    n = subj.size
    maj = int(np.floor(n / 2) + 1)
    lo, hi = it["auc_ha_band"]
    n_late = int(np.sum(subj > it["subject_hb_auc_min"]))
    # H-B: surplus LATER — signed delta_t must be positive (P1-3) AND sensitivity concordant (P1-5)
    if (cohort_auc >= it["auc_hb_min"] and n_late >= maj
            and delta_t_med >= it["delta_t_hb_min_sec"] and null_p < alpha
            and bool(sensitivity_concordant)):
        return "H-B_supported"
    # surplus EARLIER than SOZ core (low tail) — distinct from indeterminate; needs artifact check
    if cohort_auc <= (1.0 - it["auc_hb_min"]) and delta_t_med <= -it["delta_t_hb_min_sec"]:
        return "surplus_earlier_unverified"
    # H-A compatible (descriptive): AUC in band, no consistent late bias, small |delta_t|
    consistent = np.all(subj <= it["subject_hb_auc_min"]) if n else False
    if lo <= cohort_auc <= hi and consistent and abs(delta_t_med) < 2.0:
        return "H-A_compatible"
    return "indeterminate"


CLAIM = {
    "H-B_supported": "Axis-surplus contacts were recruited after clinical SOZ contacts, "
                     "supporting the interpretation that the interictal axis captures a broader "
                     "propagation scaffold rather than only the seizure onset core.",
    "H-A_compatible": "Axis-surplus recruitment latency was compatible with onset-synchronous "
                      "recruitment relative to the axis-covered SOZ core (descriptive; n too small "
                      "for a formal equivalence test).",
    "surplus_earlier_unverified": "Axis-surplus contacts appeared to be recruited BEFORE the "
                      "axis-covered SOZ core; this low-latency-tail pattern requires a t0 "
                      "left-censoring artifact check before interpretation and is reported unverified.",
    "indeterminate": "First-threshold recruitment latency was not sufficiently resolved to "
                     "distinguish onset-synchronous from downstream surplus recruitment.",
}

# R2: primary same-shaft null only licenses 'beyond implantation geometry', NOT 'beyond HFO-rich'
# (that needs the follow-up HFO-rate-matched null). Wording is pinned accordingly.
COVERAGE_CLAIM = {
    "specific_axis_soz_organization": "The interictal propagation axis covered clinical SOZ beyond "
        "implantation geometry AND its non-SOZ surplus was spatially structured (closer to SOZ than a "
        "same-shaft random axis), indicating specific axis-SOZ spatial organization.",
    "beyond_implantation_geometry": "The interictal propagation axis covered clinical SOZ beyond "
        "implantation geometry; surplus spatial structure was NOT established, so specificity beyond a "
        "geometric coincidence is not claimed.",
    "none": "SOZ coverage by the interictal axis did not exceed the same-shaft geometric null.",
}


def _require(d: dict, keys: list, where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"{where} missing required fields {missing} (fail-closed, review P1-2)")


def _spatial_primary_ok(spa: dict, cfg: dict) -> bool:
    """Surplus spatial structure counts toward the coverage double-condition only when
    ENOUGH coord-eligible subjects support it (review P1: a 1-subject spatial claim is
    too fragile to license 'specific axis-SOZ organization') AND the cohort-median
    distance null passes."""
    return bool(spa.get("n_spatial_eligible", 0) >= cfg["v3c"]["spatial"]["min_subjects_for_primary"]
                and spa.get("p_value", 1.0) < 0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["broad", "narrow"], required=True)
    args = ap.parse_args()
    cfg = load_v3_config(); base = OUT / args.cohort

    def _load(p):
        return json.loads(p.read_text()) if p.exists() else {}
    cov = _load(base / "coverage_cohort.json")
    lat = _load(base / "latency/latency_cohort.json")
    spa = _load(base / "surplus_spatial/surplus_spatial_cohort.json")

    # P1-1: coverage DOUBLE-condition (spec §4.4) — significant coverage AND structured surplus.
    # spatial leg also gated on n_spatial_eligible (review P1: >=min_subjects_for_primary).
    coverage_sig = bool(cov.get("p_value", 1.0) < 0.05)
    spatial_sig = _spatial_primary_ok(spa, cfg)
    coverage_primary_pass = coverage_sig and spatial_sig
    coverage_claim_level = ("specific_axis_soz_organization" if coverage_primary_pass
                            else ("beyond_implantation_geometry" if coverage_sig else "none"))

    interp = "not_run"
    if lat.get("subject_aucs"):
        # P1-2 fail-closed: these fields are the Task-11 contract; missing => raise, don't default
        _require(lat, ["obs_cohort_median_auc", "subject_aucs", "delta_t_med", "p_value",
                       "sensitivity_concordant"], "latency_cohort.json")
        interp = interpret_latency(lat["obs_cohort_median_auc"], list(lat["subject_aucs"].values()),
                                   lat["delta_t_med"], lat["p_value"], lat["sensitivity_concordant"], cfg)

    summary = {
        "cohort": args.cohort,
        "coverage_significant": coverage_sig, "coverage_p": cov.get("p_value"),
        "surplus_spatial_significant": spatial_sig, "surplus_spatial_p": spa.get("p_value"),
        "coverage_primary_pass": coverage_primary_pass,
        "coverage_claim_level": coverage_claim_level,
        "coverage_claim": COVERAGE_CLAIM[coverage_claim_level],
        "latency_interpretation": interp, "latency_claim": CLAIM.get(interp, ""),
        "latency_delta_t_med": lat.get("delta_t_med"),
        "latency_sensitivity_concordant": lat.get("sensitivity_concordant"),
    }
    (base / "v3c_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test + real run**

Run: `pytest tests/test_topic5_v3c_latency.py::test_interpret_latency_four_way -v`
Expected: PASS
Run: `python scripts/run_topic5_v3c_summary.py --cohort broad`
Expected: prints summary JSON with `coverage_primary_pass` (only True when BOTH coverage null AND surplus-spatial null pass), `coverage_claim_level`, `latency_interpretation`, `latency_delta_t_med`, `latency_sensitivity_concordant`. (If `latency_cohort.json` is missing a contract field, it raises `KeyError … fail-closed` — that is the intended P1-2 guard, not a bug.)

- [ ] **Step 5: Commit**

```bash
git add scripts/run_topic5_v3c_summary.py tests/test_topic5_v3c_latency.py
git commit -m "feat(topic5-v3c): summary — coverage double-condition gate + signed-Δt 4-way latency verdict"
```

---

### Task 15: Figures (QC 3-fig + coverage forest + surplus spatial + S∖A) + README

**Files:**
- Create: `scripts/plot_topic5_v3c.py`
- Create: `results/topic5_ictal_recruitment/v3c_soz_axis_coverage/<cohort>/figures/README.md` (written after figures render)

**Interfaces:**
- Consumes: the CSVs/JSONs from Tasks 5, 9, 11, 13. Produces PNGs: `qc_raster_<subj>.png`, `qc_bars.png`, `coverage_null_forest.png`, `latency_auc_forest.png`, `surplus_spatial.png`.

- [ ] **Step 1: Write `scripts/plot_topic5_v3c.py`**

Render (matplotlib; follow `docs/figure_style_guide.md`; paper-grade self-contained per memory `feedback_figure_self_contained_paper_grade`): (1) coverage null forest — per-subject observed coverage vs its same-shaft null 5–95% interval; (2) latency AUC forest — per-subject AUC_late vs within-shaft null interval, 0.5 reference line, assay-invalid subjects greyed; (3) QC bars — t0/censor/finite/uniq per subject; (4) surplus spatial — shaft-gini + mean-dist-to-SOZ; (5) `S∖A` callout for 635. Each reads the cohort CSVs and writes PNG to `<cohort>/figures/`.

```python
# scripts/plot_topic5_v3c.py  (skeleton — fill each panel per the list above)
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "results/topic5_ictal_recruitment/v3c_soz_axis_coverage"

def coverage_forest(cohort):
    base = OUT / cohort
    df = pd.read_csv(base / "coverage_subject.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    y = np.arange(len(df))
    # same-shaft null 5-95% interval per subject (null_q05/q95 now emitted by Task 5)
    for i, r in df.iterrows():
        if np.isfinite(r["null_q05"]) and np.isfinite(r["null_q95"]):
            ax.plot([r["null_q05"], r["null_q95"]], [i, i], color="0.7", lw=6, solid_capstyle="butt",
                    zorder=1, label="same-shaft null 5-95%" if i == 0 else None)
    ax.scatter(df["coverage"], y, c="k", zorder=3, label="observed |A∩S|/|S|")
    ax.set_yticks(y); ax.set_yticklabels(df["subject"])
    ax.set_xlabel("SOZ coverage by interictal axis"); ax.set_xlim(0, 1.05)
    ax.set_title(f"V3c-1 coverage ({cohort} primary)"); ax.legend(loc="lower left", fontsize=8)
    (base / "figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(base / "figures/coverage_null_forest.png", dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cohort", choices=["broad", "narrow"], required=True)
    args = ap.parse_args()
    coverage_forest(args.cohort)
    # latency_auc_forest(args.cohort); qc_bars(args.cohort); surplus_spatial(args.cohort)
    print(f"[done] figures for {args.cohort}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Render for broad**

Run: `python scripts/plot_topic5_v3c.py --cohort broad`
Expected: PNGs in `broad/figures/`. **User visually inspects each (memory: render→eyeball→fix before commit).**

- [ ] **Step 3: Write `figures/README.md`** (Chinese, per-figure 2–4 sentences + `**关注点**:`; AGENTS.md Results standard).

- [ ] **Step 4: Commit**

```bash
git add scripts/plot_topic5_v3c.py "results/topic5_ictal_recruitment/v3c_soz_axis_coverage/broad/figures/README.md"
git commit -m "feat(topic5-v3c): figures (coverage forest / AUC forest / QC bars / surplus) + README"
```

---

## Milestone G — cohort run + docs

### Task 16: Full cohort run (broad + narrow) + archive doc + main-doc pointer

**Files:**
- Create: `docs/archive/topic5/v3c_soz_axis_coverage_2026-07-04.md` (results, plain-language + numbers)
- Modify: `docs/topic5_seizure_subtyping.md` (add §3.9 pointer)
- Modify: `results/FIGURE_INDEX.md` (append figure dir)

**Interfaces:** none (execution + documentation).

- [ ] **Step 1: Run the full pipeline both cohorts**

```bash
for c in broad narrow; do
  python scripts/run_topic5_v3c_coverage.py --cohort $c
  python scripts/run_topic5_v3c_latency_qc.py --cohort $c
  python scripts/run_topic5_v3c_latency.py --cohort $c
  python scripts/run_topic5_v3c_surplus_spatial.py --cohort $c
  python scripts/run_topic5_v3c_summary.py --cohort $c
  python scripts/plot_topic5_v3c.py --cohort $c
done
```
Expected: all CSVs/JSONs/PNGs under both `broad/` and `narrow/`.

- [ ] **Step 2: Full test suite green**

Run: `pytest tests/test_topic5_v3c_coverage.py tests/test_topic5_v3c_latency.py tests/test_topic5_v3c_io.py -v`
Expected: all PASS (integration tests need the cache mount).

- [ ] **Step 3: Write archive doc** (plain-language §8 三段式 abstract; coverage double-condition verdict; latency interpretation with R2/R3 caveats; per-subject tables; forbidden-claim list from spec §9). **Invoke `hfosp-plain-language-recap` before the abstract.** **Backfill spec §附录 A.1 row 1146** (`|A∩S|`/`|A∖S|`/`|S∖A|` = PENDING) with the real numbers from `broad/coverage_subject.csv`; if any pre-run appendix estimate disagrees with the executed join, the executed value wins.

- [ ] **Step 4: Add pointer to `docs/topic5_seizure_subtyping.md`** as §3.9 (one paragraph, plain-language, links archive) + append figure dir to `results/FIGURE_INDEX.md`.

- [ ] **Step 5: Commit**

```bash
git add docs/archive/topic5/v3c_soz_axis_coverage_2026-07-04.md docs/topic5_seizure_subtyping.md results/FIGURE_INDEX.md
git commit -m "docs(topic5-v3c): archive results + main-doc §3.9 pointer + figure index"
```

---

## Self-Review (against spec)

**Spec coverage:** V3c-1 coverage → Tasks 2/3/5 (metrics/null/runner). V3c-2 latency → Tasks 6/7/8/9/10/11 (crossing/QC/extract/QC-runner/AUC/gated-runner). V3c-3 spatial → Tasks 12/13. Coverage double-condition (§4.4) → **enforced** in Task 14 (`coverage_primary_pass = coverage_sig ∧ spatial_sig`) consuming Task 13's `surplus_spatial_cohort.json`. Null hierarchy (§4.2): primary same-shaft = Task 3; HFO-rate/distance sensitivity = distance in Task 12 (HFO-rate-matched null = **follow-up sensitivity, not in first cut** — flagged below). Inference (§7) → cohort-median null in Tasks 5/11/13. H-A/H-B/surplus-earlier/indeterminate (§5.6, R3) → Task 14. Claim language (§9) → Task 14 CLAIM + COVERAGE_CLAIM. Outcome future/blocked (§10) → not built (correct). broad/narrow never pool → separate `--cohort` dirs everywhere.

**Review fixes applied (P1-1..P1-5, 2026-07-04):** P1-1 coverage double-condition is now a real gate in Task 14 (not a note) fed by Task 13's new `surplus_spatial_cohort.json`; P1-2 Task 11 emits `delta_t_med`/`subject_delta_t` and Task 14 fails closed (`_require`) if absent; P1-3 H-B uses SIGNED `delta_t_med >= +thr` (not `abs`) and a distinct `surplus_earlier_unverified` category; P1-4 `extract_latency_matrix` raises on missing contacts (no silent row/name misalignment) with a fail-closed test; P1-5 Task 11 emits `drop_censored`/`exclude_t0` sensitivity AUCs + `sensitivity_concordant`, which Task 14 requires for H-B. Null quantiles (`null_q05/50/95`, `auc_null_q05/50/95`, `dist_null_q05/50/95`) now emitted for the forest figures.

**Known scope cut (flagged, not silent):** the **HFO-rate-matched sensitivity null** (spec §4.2 third tier, §5.5) is NOT implemented in Tasks 1–16 — only the same-shaft primary null (Task 3) and the distance-to-SOZ null (Task 12) are. This is an intentional first-cut boundary; a follow-up task must add `rate_matched_null` (bin contacts by interictal HFO participation from `cls["participation"]`, permute within bin) before the coverage claim can be worded "beyond HFO-rich" (R2). Documented in the archive doc's "next steps".

**Placeholder scan:** Task 8 now ships only the real fail-closed `extract_latency_matrix` (placeholder removed per review P1-4). No TODO/TBD.

**Type consistency:** `axis_soz_join` returns `coverage_metrics` keys (`covered`/`surplus`/`missed`/`n_*`) + `soz_in_pool` — consumed consistently in Tasks 5/9/11/13. `first_crossing_latency` returns `(kind, sec)` consumed by `latency_seconds`/`encode_latency_for_rank` in Tasks 9/11. `extract_latency_matrix` dict shape (`kinds`/`secs` keyed by threshold) consistent Tasks 9/11. `label_permute(axis, nonaxis, shaft, rng)` signature matches its definition in `topic5_v3_mode_transition.py`.

---

## Execution Handoff

**Mode (user-chosen 2026-07-04): INLINE execution in the current worktree** (`superpowers:executing-plans`), NOT subagent-driven. Rationale: the failure surface here is field contracts, real artifacts, the 1146 backfill, and wording gates — splitting to subagents risks conflating "runs" with "claimable". Advance milestone A→G with small commits.

**Boundaries carried into execution:**
- Every `git commit` step here is authorized as the per-task TDD rhythm (user opted in to milestone commits).
- Coverage is primary and never depends on latency; latency is gated on label-blind assay-QC; broad primary / narrow sensitivity, never pooled.
- HFO-rate-matched null is a flagged follow-up; until built, coverage wording stays "beyond implantation geometry", never "beyond HFO-rich".
- Backfill spec §附录 A.1 row 1146 from the executed `broad/coverage_subject.csv` (Task 16).
- Post-implementation, run `superpowers:requesting-code-review` before merging the branch.
