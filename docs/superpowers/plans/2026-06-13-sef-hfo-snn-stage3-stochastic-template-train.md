# Stage 3 — Long-record stochastic template train Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run two equal low-threshold foci in ONE cm-SNN, let background noise spontaneously ignite a long event train from both ends, and test whether the real blind pipeline recovers two opposite templates whose label sequence shows no strong ping-pong or persistence at model scale.

**Architecture:** Extend the existing spontaneous read-out runner with a `twoend_equal` mode (two equal cores, one network, no kick). Per event we assign a **core-level** hidden source label and a collision flag, and write ONE single-network legacy lagPat record + a per-event sidecar. Analysis is **per-run** (each network = one synthetic subject): masked PR-2/2.5/rank-displacement for structure, a collision-censored timing wrapper over `template_temporal_pairing` for label-independence, synthetic-label controls for test validity, and a small entry-jitter helper. A **hard pilot checkpoint** pins parameters before any multi-seed long run.

**Tech Stack:** Python, numpy, pytest; reuses `results/topic4_sef_hfo/lif_snn/engine` (checksum-guarded), `src/sef_hfo_*`, `src/interictal_propagation.py`, `src/rank_displacement.py`, `src/template_temporal_pairing.py`.

**Spec:** `docs/superpowers/specs/2026-06-13-sef-hfo-snn-stage3-stochastic-template-train-design.md`

---

## File Structure

- **Create** `src/sef_hfo_stage3.py` — pure, unit-testable helpers: `core_onset`, `assign_hidden_label`, `collision_free_blocks`, `synthetic_label_sequence`, `entry_jitter_stats`.
- **Create** `src/sef_hfo_stage3_analysis.py` — `analyze_structure_one_record` (masked PR-2/2.5/swap + purity + shared-active corr), `analyze_timing` (collision-censored timing + synthetic controls).
- **Modify** `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py` — add `twoend_equal` lesion mode, core-level onset, collision flag, per-event sidecar writer, real-time packedTimes.
- **Create** `scripts/run_sef_hfo_snn_cm_spontaneous_stage3.py` — orchestrator: `--pilot` (1–2 seeds, no conclusions) and multi-seed; per-run reporting, sign-consistency, NO pooled p-value.
- **Create** `scripts/plot_sef_hfo_snn_cm_spontaneous_stage3.py` — core figure + timing panel (main vs synthetic controls).
- **Create** `tests/test_sef_hfo_stage3.py` — unit tests for the pure helpers + analysis wiring + the censoring invariant.

Phasing (the pilot is a HARD STOP):
- **Phase 0** pure helpers (TDD).
- **Phase 1** runner `twoend_equal` + sidecar + asserts → **pilot run → STOP for user review**.
- **Phase 2** one-record structure analysis.
- **Phase 3** collision-censored timing + synthetic controls.
- **Phase 4** orchestrator + per-run gates → multi-seed (only after pilot approved).
- **Phase 5** figure.
- **Phase 6 (OPTIONAL/stretch)** engine forced-ping-pong control.

---

## Phase 0 — Pure helpers (TDD)

### Task 1: `core_onset` — first time a core's active fraction crosses threshold

**Files:**
- Create: `src/sef_hfo_stage3.py`
- Test: `tests/test_sef_hfo_stage3.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from src.sef_hfo_stage3 import core_onset

def test_core_onset_returns_first_crossing_time():
    # active fraction per bin; bin width 1.0 ms; threshold 0.1
    af = np.array([0.0, 0.02, 0.05, 0.2, 0.5, 0.1])
    # first bin >= 0.1 is index 3 -> time 3*1.0 = 3.0
    assert core_onset(af, bin_w=1.0, frac_threshold=0.1) == 3.0

def test_core_onset_none_if_never_crosses():
    af = np.array([0.0, 0.01, 0.02])
    assert core_onset(af, bin_w=1.0, frac_threshold=0.1) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sef_hfo_stage3.py -k core_onset -v`
Expected: FAIL (ImportError / function not defined).

- [ ] **Step 3: Write minimal implementation**

```python
"""Stage 3 pure helpers: core-level onset, hidden-label assignment, collision
censoring into block ranges, synthetic-label controls, entry-jitter stats.
All functions here are deterministic and unit-tested; no simulation, no I/O."""
from typing import List, Optional, Sequence, Tuple
import numpy as np


def core_onset(active_fraction: np.ndarray, bin_w: float,
               frac_threshold: float) -> Optional[float]:
    """First time (ms) a core's E active-fraction series crosses frac_threshold.
    Returns None if it never crosses."""
    af = np.asarray(active_fraction, float)
    idx = np.flatnonzero(af >= frac_threshold)
    return float(idx[0] * bin_w) if idx.size else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sef_hfo_stage3.py -k core_onset -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_stage3.py tests/test_sef_hfo_stage3.py
git commit -m "feat(stage3): core_onset helper (core-level event onset)"
```

### Task 2: `assign_hidden_label` — which end ignited first, or collision

**Files:**
- Modify: `src/sef_hfo_stage3.py`
- Test: `tests/test_sef_hfo_stage3.py`

- [ ] **Step 1: Write the failing test**

```python
from src.sef_hfo_stage3 import assign_hidden_label

def test_assign_hidden_label_neg_leads():
    assert assign_hidden_label(10.0, 20.0, delta_onset=3.0) == "neg"

def test_assign_hidden_label_pos_leads():
    assert assign_hidden_label(20.0, 10.0, delta_onset=3.0) == "pos"

def test_assign_hidden_label_collision_when_simultaneous():
    assert assign_hidden_label(10.0, 12.0, delta_onset=3.0) == "collision"

def test_assign_hidden_label_ambiguous_when_no_crossing():
    assert assign_hidden_label(None, 10.0, delta_onset=3.0) == "ambiguous"
    assert assign_hidden_label(None, None, delta_onset=3.0) == "ambiguous"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sef_hfo_stage3.py -k assign_hidden_label -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
def assign_hidden_label(onset_neg: Optional[float], onset_pos: Optional[float],
                        delta_onset: float) -> str:
    """neg/pos by which core crossed first; collision if within delta_onset;
    ambiguous if a core never crossed (caller also maps unreadable-axis -> ambiguous)."""
    if onset_neg is None or onset_pos is None:
        return "ambiguous"
    if abs(onset_neg - onset_pos) <= delta_onset:
        return "collision"
    return "neg" if onset_neg < onset_pos else "pos"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sef_hfo_stage3.py -k assign_hidden_label -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_stage3.py tests/test_sef_hfo_stage3.py
git commit -m "feat(stage3): assign_hidden_label (core-level, collision-aware)"
```

### Task 3: `collision_free_blocks` — the P1-1 censor mechanism

**Files:**
- Modify: `src/sef_hfo_stage3.py`
- Test: `tests/test_sef_hfo_stage3.py`

This is the load-bearing P1-1 fix. Clean events between two censored events form ONE block; censored (collision/ambiguous) events fall OUTSIDE every block so `_assign_blocks` maps them to −1 and no run/transition spans them.

- [ ] **Step 1: Write the failing test**

```python
from src.sef_hfo_stage3 import collision_free_blocks

def test_collision_free_blocks_segments_at_censored_events():
    # times (seconds), clean flags: event 2 is a collision (censored)
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    clean = np.array([True, True, False, True, True])
    blocks, block_id = collision_free_blocks(times, clean)
    # two segments: [0,1] and [3,4]; censored event -> -1
    assert blocks == [(0.0, 1.0), (3.0, 4.0)]
    assert list(block_id) == [0, 0, -1, 1, 1]

def test_collision_free_blocks_no_block_spans_a_censor_time():
    # a single clean event isolated between two censors -> its own 1-event block
    times = np.array([0.0, 1.0, 2.0])
    clean = np.array([False, True, False])
    blocks, block_id = collision_free_blocks(times, clean)
    assert blocks == [(1.0, 1.0)]
    assert list(block_id) == [-1, 0, -1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sef_hfo_stage3.py -k collision_free_blocks -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
def collision_free_blocks(event_times: np.ndarray, clean_for_timing: np.ndarray
                          ) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """Partition events into maximal runs of CONSECUTIVE clean events (time order).
    Each run -> one (t_first, t_last) block. Censored events (clean_for_timing
    False) get block_id -1 and break the run, so they fall outside every block
    range and no transition is ever counted across them.

    Returns (block_time_ranges, block_id_per_event) aligned to the INPUT order.
    """
    times = np.asarray(event_times, float)
    clean = np.asarray(clean_for_timing, bool)
    order = np.argsort(times, kind="stable")
    block_id = np.full(times.size, -1, dtype=int)
    blocks: List[Tuple[float, float]] = []
    cur: List[int] = []  # indices (into original) of current run

    def _close():
        if cur:
            ts = times[cur]
            blocks.append((float(ts.min()), float(ts.max())))
            for j in cur:
                block_id[j] = len(blocks) - 1
            cur.clear()

    for i in order:
        if clean[i]:
            cur.append(i)
        else:
            _close()
    _close()
    return blocks, block_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sef_hfo_stage3.py -k collision_free_blocks -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_stage3.py tests/test_sef_hfo_stage3.py
git commit -m "feat(stage3): collision_free_blocks (censor collisions as sequence boundaries)"
```

### Task 4: `synthetic_label_sequence` — test-validity controls (P1-5)

**Files:**
- Modify: `src/sef_hfo_stage3.py`
- Test: `tests/test_sef_hfo_stage3.py`

- [ ] **Step 1: Write the failing test**

```python
from src.sef_hfo_stage3 import synthetic_label_sequence

def test_synthetic_alternating_preserves_marginal_and_alternates():
    labels = np.array([0, 0, 0, 1, 1, 1])
    out = synthetic_label_sequence(labels, mode="alternating",
                                   rng=np.random.default_rng(0))
    assert sorted(out.tolist()) == sorted(labels.tolist())   # marginal preserved
    # equal counts -> fully alternating -> zero same-adjacent pairs
    assert int(np.sum(out[:-1] == out[1:])) == 0

def test_synthetic_sticky_makes_two_runs():
    labels = np.array([0, 1, 0, 1, 0, 1])
    out = synthetic_label_sequence(labels, mode="sticky",
                                   rng=np.random.default_rng(0))
    assert sorted(out.tolist()) == sorted(labels.tolist())
    # sticky = all of one then all of the other -> exactly 2 runs
    n_runs = 1 + int(np.sum(out[1:] != out[:-1]))
    assert n_runs == 2

def test_synthetic_shuffle_preserves_marginal():
    labels = np.array([0, 0, 1, 1, 1])
    out = synthetic_label_sequence(labels, mode="shuffle",
                                   rng=np.random.default_rng(0))
    assert sorted(out.tolist()) == sorted(labels.tolist())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sef_hfo_stage3.py -k synthetic -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
def synthetic_label_sequence(labels: np.ndarray, mode: str,
                             rng: np.random.Generator) -> np.ndarray:
    """Re-arrange a binary label array holding the MARGINAL COUNTS fixed.
    mode='alternating' -> maximal ping-pong; 'sticky' -> two maximal runs;
    'shuffle' -> random permutation (independent). Event TIMES are unchanged
    by the caller; only the label order is replaced."""
    labels = np.asarray(labels, int)
    if mode == "shuffle":
        return rng.permutation(labels)
    classes, counts = np.unique(labels, return_counts=True)
    if classes.size != 2:
        raise ValueError(f"synthetic controls require 2 classes, got {classes}")
    a, b = int(classes[0]), int(classes[1])
    na, nb = int(counts[0]), int(counts[1])
    if mode == "sticky":
        return np.array([a] * na + [b] * nb, dtype=int)
    if mode == "alternating":
        # interleave, then append the remainder of the larger class
        k = min(na, nb)
        head = np.empty(2 * k, dtype=int)
        head[0::2] = a; head[1::2] = b
        tail = [a] * (na - k) if na > nb else [b] * (nb - k)
        return np.concatenate([head, np.array(tail, dtype=int)])
    raise ValueError(f"unknown mode {mode}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sef_hfo_stage3.py -k synthetic -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_stage3.py tests/test_sef_hfo_stage3.py
git commit -m "feat(stage3): synthetic_label_sequence (alternating/sticky/shuffle controls)"
```

### Task 5: `entry_jitter_stats` — entry-jitter (secondary)

**Files:**
- Modify: `src/sef_hfo_stage3.py`
- Test: `tests/test_sef_hfo_stage3.py`

- [ ] **Step 1: Write the failing test**

```python
from src.sef_hfo_stage3 import entry_jitter_stats

def test_entry_jitter_stats_single_fixed_contact_has_zero_dispersion():
    s = entry_jitter_stats(["A0", "A0", "A0"])
    assert s["n_unique"] == 1
    assert s["top1_fraction"] == 1.0

def test_entry_jitter_stats_wandering_group():
    s = entry_jitter_stats(["A0", "A1", "A0", "A2", "A0", "A1"])
    assert s["n_unique"] == 3
    assert 0.0 < s["top1_fraction"] < 1.0
    # top-3 covers everything here
    assert s["top3_fraction"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sef_hfo_stage3.py -k entry_jitter -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
def entry_jitter_stats(first_contacts: Sequence[str]) -> dict:
    """Distribution of the per-event first-active contact for one direction.
    A single fixed contact -> n_unique 1, top1 1.0 (locked, no jitter).
    A wandering small group -> n_unique>1 with high top-k coverage."""
    fc = [c for c in first_contacts if c is not None]
    n = len(fc)
    if n == 0:
        return {"n": 0, "n_unique": 0, "top1_fraction": float("nan"),
                "top3_fraction": float("nan"), "counts": {}}
    vals, cnts = np.unique(np.array(fc, dtype=object), return_counts=True)
    order = np.argsort(cnts)[::-1]
    cnts_sorted = cnts[order]
    return {
        "n": n,
        "n_unique": int(vals.size),
        "top1_fraction": float(cnts_sorted[0]) / n,
        "top3_fraction": float(cnts_sorted[:3].sum()) / n,
        "counts": {str(vals[i]): int(cnts[i]) for i in range(vals.size)},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sef_hfo_stage3.py -k entry_jitter -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_stage3.py tests/test_sef_hfo_stage3.py
git commit -m "feat(stage3): entry_jitter_stats (first-contact dispersion)"
```

---

## Phase 1 — Runner `twoend_equal` + sidecar + asserts → PILOT (hard stop)

### Task 6: Add `twoend_equal` lesion mode + core-level onset + collision + sidecar

**Files:**
- Modify: `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py`
- Test: `tests/test_sef_hfo_stage3.py` (tiny-network smoke + assert tests)

Reuse the existing structure (`build_lesion_vth`, `simulate_kick(KICK_BOOST=0)`, `detect_events`, `read_event`, `per_neuron_onset`). Add:

1. `twoend_equal` branch in `build_lesion_vth`: two cores at neg_xy and pos_xy, **same** `core_mean` (no dephase), `vth = min(cf_neg.vth, cf_pos.vth)`, returns `core_mask_neg`, `core_mask_pos` separately (needed for per-core onset).
2. New `core_active_fraction(E_spk_bool, core_mask_idx, dt, bin_ms)` → per-core active fraction inside an event window (reuse the binning of `active_fraction`).
3. Per event: `onset_neg = core_onset(core_af_neg_in_window, bin_w, frac_threshold)`, same for pos; `frac_threshold = max(0.01, N_MIN/ n_core_cells)`; `hidden = assign_hidden_label(...)`; if `read_event` axis is unreadable -> `hidden="ambiguous"`. `clean_for_timing = hidden in {"neg","pos"} and returned and axis_err<25 and n_part>=PART_MIN`.
4. Write packedTimes in **real seconds** (`t_on/1000.0`, `t_off/1000.0`), NOT arbitrary slots.
5. Write a sidecar JSON `sidecar_<tag>.json`: list of per-event dicts with the §6 schema; the i-th sidecar entry MUST align with the i-th column of the lagPat record (assert).

- [ ] **Step 1: Write the failing smoke + assert test**

```python
import os, json, subprocess, numpy as np

def test_twoend_equal_smoke_writes_aligned_sidecar(tmp_path):
    # tiny, fast network: small L, short T -> just check the contract, NOT science.
    # --out routes ALL output under tmp_path so the test never pollutes results/.
    env = dict(os.environ, OMP_NUM_THREADS="4")
    out = subprocess.run(
        ["python3", "scripts/run_sef_hfo_snn_cm_spontaneous_readout.py",
         "--L", "8", "--density", "60", "--T", "600", "--lesion", "twoend_equal",
         "--core-mean", "14.0", "--core-std", "1.5", "--seed", "1",
         "--delta-onset", "20", "--tag", "smoke_te", "--out", str(tmp_path)],
        capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    side = json.load(open(os.path.join(str(tmp_path), "sidecar_smoke_te.json")))
    ev = side["events"]
    # de-flake: core-mean 14.0 is well below the ignition boundary so the tiny net
    # reliably fires; if this ever returns 0 events, raise T / lower core-mean.
    assert len(ev) >= 1, "smoke net produced no events; lower core-mean or raise T"
    rec = np.load(os.path.join(str(tmp_path), "record", "smoke_te",
                  "model_smoke_te_lagPat_withFreqCent.npz"), allow_pickle=True)
    packed = np.load(os.path.join(str(tmp_path), "record", "smoke_te",
                     "model_smoke_te_packedTimes_withFreqCent.npy"))
    # ASSERT 1: every event has a contiguous event_id from 0
    assert [e["event_id"] for e in ev] == list(range(len(ev)))
    # ASSERT 2: sidecar length == number of record columns (column alignment)
    assert len(ev) == rec["lagPatRank"].shape[1]
    # ASSERT 3: valid hidden label + required onset fields on every event
    for e in ev:
        assert e["hidden_source_label"] in {"neg", "pos", "collision", "ambiguous"}
        assert "core_onset_neg" in e and "core_onset_pos" in e
        assert "clean_for_timing" in e
    # ASSERT 4 (HARD, blocks the Stage-2 synthetic-interleave failure): packedTimes[:,0]
    # MUST equal sidecar t_on/1000 element-for-element, in the SAME order.
    side_t = np.array([e["t_on"] / 1000.0 for e in ev], float)
    assert np.allclose(packed[:, 0], side_t), "packedTimes are not the real event seconds"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sef_hfo_stage3.py -k twoend_equal_smoke -v`
Expected: FAIL (`twoend_equal` not a valid choice).

- [ ] **Step 3: Implement in the runner**

Add `"twoend_equal"` to the `--lesion` choices. In `build_lesion_vth`, add:

```python
    if lesion == "twoend_equal":
        cf1 = sample_core_field(net["pos"], is_E, neg_xy, core_r,
                                np.random.default_rng(seed + 7),
                                core_mean=core_mean, core_std=core_std, base_mean=18.0)
        cf2 = sample_core_field(net["pos"], is_E, pos_xy, core_r,
                                np.random.default_rng(seed + 8),
                                core_mean=core_mean, core_std=core_std, base_mean=18.0)
        # return BOTH core masks so per-core onset can be computed
        return (np.minimum(cf1["vth"], cf2["vth"]),
                cf1["core_mask"], cf2["core_mask"], [neg_xy, pos_xy])
```

(For backward compat keep the other branches returning the existing 3-tuple; in `main`, detect the 4-tuple for `twoend_equal`.) Then add, near `per_neuron_onset`:

```python
from src.sef_hfo_stage3 import core_onset, assign_hidden_label  # noqa: E402

N_MIN = 5            # min core cells to count an onset (pilot-pinned)
DELTA_ONSET_MS = None  # set from --delta-onset; pilot-pinned

def core_active_fraction(E_spk_bool, core_e_idx, dt, bin_ms, t_on, t_off):
    # core_e_idx are E-NEURON column indices (NOT full-network indices); see below.
    bs = max(1, int(round(bin_ms / dt)))
    s, e = int(t_on / dt), int(t_off / dt)
    seg = E_spk_bool[s:e][:, core_e_idx]
    nb = seg.shape[0] // bs
    binned = seg[:nb * bs].reshape(nb, bs, -1).any(axis=1)
    return binned.mean(axis=1)   # fraction of THIS core's E cells active per bin
```

**P1-5 — E-only indexing.** `spk = res["E_spk_bool"]` has NE columns (E neurons only), but `sample_core_field`'s `core_mask` is over the FULL network (length N = NE+NI). Slice to the E block before indexing the spike matrix:

```python
core_neg_e_idx = np.flatnonzero(core_mask_neg[:NE])   # E-only columns of E_spk_bool
core_pos_e_idx = np.flatnonzero(core_mask_pos[:NE])
n_core_cells_neg = len(core_neg_e_idx); n_core_cells_pos = len(core_pos_e_idx)
```

In `main`, for each event compute `onset_neg/onset_pos` via `core_onset` on each core's in-window active fraction with `frac_threshold = max(0.01, N_MIN / n_core_cells)` (use each core's own `n_core_cells`), then `hidden = assign_hidden_label(onset_neg, onset_pos, DELTA_ONSET_MS)`; override to `"ambiguous"` if `rd["axis_err"] is None`; set `collision_reason` accordingly (`simultaneous_onset` / `unreadable_axis` / `no_core_crossing` / `none`). Set `clean_for_timing = hidden in {"neg","pos"} and returned and rd["axis_err"] is not None and rd["axis_err"] < 25 and rd["n_part"] >= PART_MIN`. Append to a `sidecar_events` list (in record-column order) with fields: `event_id, t_on, t_off, event_peak_t, hidden_source_label, core_onset_neg, core_onset_pos, collision_reason, clean_for_timing, n_part, axis_err, sign, readability`. Write packedTimes with real seconds (`t_on/1000, t_off/1000`) and dump `sidecar_<tag>.json`.

Add args: `--delta-onset` (float, ms; required for twoend_equal), `--n-min` (int, default 5), `--out` (str, default the canonical `OUT`; route `readout_*.json`, `sidecar_*.json`, `per_event/`, and `record/` under it so tests can use `tmp_path`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sef_hfo_stage3.py -k twoend_equal_smoke -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_sef_hfo_snn_cm_spontaneous_readout.py src/sef_hfo_stage3.py tests/test_sef_hfo_stage3.py
git commit -m "feat(stage3): twoend_equal runner mode + core-level hidden label + sidecar"
```

### Task 7: PILOT RUN — pin parameters, NO science conclusions → STOP

**Files:** none new — a run + a short report.

- [ ] **Step 1: Run the pilot (1–2 seeds, main operating point)**

```bash
python3 scripts/run_sef_hfo_snn_cm_spontaneous_readout.py \
  --L 20 --density 100 --T 15000 --lesion twoend_equal \
  --core-mean 17.0 --core-std 1.5 --seed 1 --delta-onset 30 --tag pilot_te_s1
```

- [ ] **Step 2: Read the pilot numbers from `readout_pilot_te_s1.json` + `sidecar_pilot_te_s1.json`** and record:
  - clean events per end (`neg`/`pos`), target ≥ 30 (≥ 50 preferred);
  - collision rate = `n_collision / n_events`, target < 20–30 %;
  - whether the surround stays quiet (`true_inter_event_floor` small) — not mush;
  - the typical inter-event interval (median gap between clean events) → fixes the `delta_t_grid`;
  - whether `DELTA_ONSET_MS=30` and `N_MIN=5` separate the ends sensibly (eyeball the `core_onset_neg/pos` spread).

- [ ] **Step 3: STOP. Report to the user.**

> Pilot numbers only — NO stable_k / swap / independence conclusions yet. Report clean/end, collision rate, mush check, typical IEI, and proposed pinned `T`, `delta_t_grid`, `DELTA_ONSET_MS`, `N_MIN`. Get explicit approval before any multi-seed long run (per the spec §8 route and the user's "不要直接长跑").

(No commit — this is a checkpoint.)

---

## Phase 2 — One-record structure analysis

### Task 8: `analyze_structure_one_record`

**Files:**
- Create: `src/sef_hfo_stage3_analysis.py`
- Test: `tests/test_sef_hfo_stage3.py`

Refactor the masked-pipeline body of `pool_and_cluster_spontaneous.py` to take ONE record directory (drop the two-run concatenation), and add purity + shared-active corr.

- [ ] **Step 1: Write the failing test** (uses a synthetic 2-template record built in the test)

```python
from src.sef_hfo_stage3_analysis import analyze_structure_one_record

def _write_synth_record(tmpdir, n_ch=8, n_per=20):
    # two opposite rank templates -> should cluster k=2 with swap
    import numpy as np
    base_fwd = np.arange(n_ch, dtype=float)
    base_rev = base_fwd[::-1].copy()
    rng = np.random.default_rng(0)
    cols = []
    hidden = []
    for _ in range(n_per):
        cols.append(base_fwd + rng.normal(0, 0.3, n_ch)); hidden.append("neg")
    for _ in range(n_per):
        cols.append(base_rev + rng.normal(0, 0.3, n_ch)); hidden.append("pos")
    rank = np.column_stack(cols)
    bools = np.ones_like(rank, bool)
    names = [f"A{i}" for i in range(n_ch)]
    sub = tmpdir / "synth"; sub.mkdir()
    np.savez(sub / "synth_lagPat_withFreqCent.npz", lagPatRank=rank,
             eventsBool=bools, lagPatRaw=rank, chnNames=np.array(names), start_t=0.0)
    np.save(sub / "synth_packedTimes_withFreqCent.npy",
            np.column_stack([np.arange(2*n_per)*1.0, np.arange(2*n_per)*1.0 + 0.05]))
    return str(sub), hidden

def test_structure_recovers_two_clusters_and_purity(tmp_path):
    sub, hidden = _write_synth_record(tmp_path)
    res = analyze_structure_one_record(sub, hidden_labels=hidden)
    assert res["chosen_k"] == 2
    assert res["cluster_hidden_purity"] >= 0.9
    assert res["n_shared_active"] >= 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sef_hfo_stage3.py -k structure -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
"""Stage 3 per-run analysis: structure (masked PR-2/2.5/swap + purity + shared-
active corr) and timing (collision-censored label-independence + controls)."""
import sys, os
import numpy as np
sys.path.insert(0, os.getcwd())
from src.interictal_propagation import (
    load_subject_propagation_events, compute_adaptive_cluster_stereotypy,
    compute_time_split_reproducibility, build_cluster_templates, _valid_event_indices)
from src.rank_displacement import compute_swap_score_sweep


def analyze_structure_one_record(record_dir, hidden_labels):
    """hidden_labels is FULL-length (one entry per record event, in record-column
    order). compute_adaptive_cluster_stereotypy returns labels indexed into
    valid_events (NOT full) — so we rebuild labels_full (-1 for invalid) and use it
    for purity/templates/timing. PR-2.5 keeps the valid-length labels + valid_ev."""
    ev = load_subject_propagation_events(record_dir)
    R = np.asarray(ev["ranks"], float); B = np.asarray(ev["bools"], bool)
    names = list(ev["channel_names"])
    eat = np.asarray(ev["event_abs_times"], float)
    bid = np.asarray(ev.get("block_ids", np.zeros(R.shape[1], int)))
    n_events = R.shape[1]
    assert len(hidden_labels) == n_events, "hidden_labels must be full record length"

    pr2 = compute_adaptive_cluster_stereotypy(R, B, names, use_masked_features=True)
    chosen_k = int(pr2["chosen_k"])
    labels_valid = np.array(pr2.get("labels", []), dtype=int)

    # P1-1: valid_events recomputed with the SAME min_participating the PR-2 used
    # (min_shared_channels default = 3). Build labels_full aligned to ALL events.
    valid_ev = _valid_event_indices(B, min_participating=3)
    assert labels_valid.size == valid_ev.size, "labels are valid-length; alignment broken"
    labels_full = np.full(n_events, -1, dtype=int)
    labels_full[valid_ev] = labels_valid

    try:  # PR-2.5 keeps the valid-length labels + valid_ev (its own contract)
        pr25 = compute_time_split_reproducibility(R, B, eat, bid, chosen_k, labels_valid,
                                                  valid_ev, use_masked_features=True)
    except Exception as e:
        pr25 = {"error": repr(e)}

    purity = float("nan"); n_shared = 0; corr = float("nan"); swap = {"skipped": True}
    if chosen_k == 2 and labels_valid.size:
        # purity vs hidden: only events that are BOTH validly clustered AND non-collision
        hl = np.array(hidden_labels, dtype=object)
        keep = np.isin(hl, ["neg", "pos"]) & (labels_full >= 0)
        if keep.any():
            h01 = (hl[keep] == "pos").astype(int)
            lab = labels_full[keep]
            purity = float(max((lab == h01).mean(), (lab == (1 - h01)).mean()))
        templates = build_cluster_templates(R, B, labels_full, 2)  # full-length labels
        t0, t1 = templates[0], templates[1]
        vm0 = np.isfinite(t0) & (t0 >= 0); vm1 = np.isfinite(t1) & (t1 >= 0)
        shared = vm0 & vm1
        n_shared = int(shared.sum())
        if n_shared >= 2:
            corr = float(np.corrcoef(t0[shared], t1[shared])[0, 1])  # DESCRIPTIVE only
        try:
            swap = compute_swap_score_sweep(t0, t1, vm0, vm1, n_perm=1000, seed=0)
        except Exception as e:
            swap = {"error": repr(e)}

    # entry-jitter input: per-event first-active contact = min-rank participating channel
    first_contacts = []
    for j in range(n_events):
        part = np.flatnonzero(B[:, j])
        first_contacts.append(names[int(part[np.argmin(R[part, j])])] if part.size else None)

    return dict(
        chosen_k=chosen_k, stable_k=pr2.get("stable_k"),
        cluster_sizes=[int((labels_full == c).sum()) for c in range(chosen_k)],
        cluster_hidden_purity=purity,
        n_shared_active=n_shared, inter_cluster_corr_shared=corr,   # corr is sanity-only
        forward_reverse_reproduced=_or_rule(pr25),
        swap_class=swap.get("swap_class") if isinstance(swap, dict) else None,
        decision_k=swap.get("decision_k") if isinstance(swap, dict) else None,
        valid_event_indices=valid_ev.tolist(),
        first_contacts=first_contacts,       # FULL length; per-event min-rank contact
        labels_full=labels_full.tolist())   # FULL length; -1 = invalid event


def _or_rule(pr25):
    """Topic 1 contract: pass = split-half OR odd-even (both reported)."""
    if not isinstance(pr25, dict):
        return None
    sp = pr25.get("splits", {})
    flags = {k: v.get("forward_reverse_reproduced")
             for k, v in (sp.items() if isinstance(sp, dict) else [])
             if isinstance(v, dict)}
    passed = any(bool(x) for x in flags.values()) if flags else None
    return {"pass_or": passed, "splits": flags}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sef_hfo_stage3.py -k structure -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_stage3_analysis.py tests/test_sef_hfo_stage3.py
git commit -m "feat(stage3): one-record structure analysis (purity + OR-rule + shared-active corr sanity)"
```

---

## Phase 3 — Collision-censored timing + synthetic controls

### Task 9: `analyze_timing` — model-scale label-independence, collisions censored

**Files:**
- Modify: `src/sef_hfo_stage3_analysis.py`
- Test: `tests/test_sef_hfo_stage3.py`

The decisive invariant test: a forced-alternating control must be flagged dependent, an independent shuffle must not — proving the test has teeth AND that censoring works.

- [ ] **Step 1: Write the failing test**

```python
from src.sef_hfo_stage3_analysis import analyze_timing
from src.sef_hfo_stage3 import synthetic_label_sequence, collision_free_blocks

def test_timing_flags_alternating_not_shuffle():
    rng = np.random.default_rng(0)
    times = np.arange(120) * 1.0            # 120 events, 1 s apart
    clean = np.ones(120, bool)
    base = np.array([0, 1] * 60)            # marginal 60/60
    alt = synthetic_label_sequence(base, "alternating", rng)
    shuf = synthetic_label_sequence(base, "shuffle", rng)
    grid = [1.0, 2.0, 4.0]
    ra = analyze_timing(times, alt, clean, grid, n_perm=200, seed=0
                        )["views"]["censored"]["by_null"]["N1"]
    rs = analyze_timing(times, shuf, clean, grid, n_perm=200, seed=0
                        )["views"]["censored"]["by_null"]["N1"]
    # ALTERNATING flagged dependent: lag1_same below the null CI; run-length lift < 1
    assert ra["lag1_same"]["inside_ci"] is False
    assert ra["lag1_same"]["empirical"] < ra["lag1_same"]["null_ci"][0]
    assert ra["mean_run_length"]["lift"] < 0.9
    # SHUFFLE reads independent: lag1_same inside the null CI
    assert rs["lag1_same"]["inside_ci"] is True

def test_timing_flags_sticky():
    rng = np.random.default_rng(1)
    times = np.arange(120) * 1.0
    clean = np.ones(120, bool)
    sticky = synthetic_label_sequence(np.array([0, 1] * 60), "sticky", rng)
    r = analyze_timing(times, sticky, clean, [1.0, 2.0, 4.0], n_perm=200, seed=0
                       )["views"]["censored"]["by_null"]["N1"]
    # STICKY flagged dependent: lag1_same ABOVE the null CI; run-length lift > 1
    assert r["lag1_same"]["inside_ci"] is False
    assert r["lag1_same"]["empirical"] > r["lag1_same"]["null_ci"][1]
    assert r["mean_run_length"]["lift"] > 1.5

def test_timing_does_not_count_transitions_across_a_collision():
    # AAAA | (censored) | BBBB -> 2 blocks; lag1_same ~1 within blocks (no A->B
    # transition counted across the censor), and the censored label never enters null.
    times = np.arange(9) * 1.0
    labels = np.array([0, 0, 0, 0, -1, 1, 1, 1, 1])  # -1 = invalid/censored event
    clean = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1], bool)
    res = analyze_timing(times, labels, clean, [1.0], n_perm=50, seed=0)
    assert res["views"]["censored"]["empirical"]["lag1_same"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sef_hfo_stage3.py -k timing -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
from src.template_temporal_pairing import (
    compute_runs, compute_run_metrics, compute_lag1_same_fraction,
    compute_pairing_lift, _generate_null_labels)
from src.sef_hfo_stage3 import collision_free_blocks

N2_WINDOW = 1e18   # N2 disabled by default (NEVER the 1800s patient default on a 15s record)


def _metrics(times, labels, blocks, delta_t_grid):
    """Empirical label-timing metrics for ONE block partition. labels MUST be the
    clean, binary, in-block labels (no collision/-1 placeholders)."""
    rm = compute_run_metrics(compute_runs(times, labels, blocks))
    out = {"lag1_same": compute_lag1_same_fraction(times, labels, blocks),
           "mean_run_length": rm["mean_run_length"]}
    for dt in delta_t_grid:
        pl = compute_pairing_lift(times, labels, dt, blocks)
        out[f"pairing_excess@{dt}"] = pl["p_opposite"] - pl["p_same"]   # ~0 if independent
    return out


def _timing_for_blocks(times, labels, blocks, delta_t_grid, n_perm, nulls, seed):
    """Empirical metrics + per-null PERCENTILE CI (P1-2). One local null loop keeps
    full null distributions (the shared burst diagnostic only exposes mean/std).
    Reuses the SAME metric + _generate_null_labels primitives — no shared-module edit."""
    emp = _metrics(times, labels, blocks, delta_t_grid)
    by_null = {}
    for i, nm in enumerate(nulls):
        rng = np.random.default_rng(seed + i)
        dists = {k: [] for k in emp}
        for _ in range(n_perm):
            sh = _generate_null_labels(nm, labels, times, blocks, rng, N2_WINDOW)
            for k, v in _metrics(times, sh, blocks, delta_t_grid).items():
                dists[k].append(v)
        d = {}
        for k, e in emp.items():
            a = np.asarray(dists[k], float); a = a[np.isfinite(a)]
            lo = float(np.percentile(a, 2.5)) if a.size else float("nan")
            hi = float(np.percentile(a, 97.5)) if a.size else float("nan")
            med = float(np.median(a)) if a.size else float("nan")
            d[k] = {"empirical": e, "null_ci": [lo, hi], "null_median": med,
                    "inside_ci": (bool(lo <= e <= hi) if np.isfinite(e) else None)}
        if np.isfinite(emp["mean_run_length"]) and d["mean_run_length"]["null_median"]:
            d["mean_run_length"]["lift"] = float(
                emp["mean_run_length"] / d["mean_run_length"]["null_median"])  # ~1 indep
        by_null[nm] = d
    return {"n_blocks": len(blocks), "delta_t_grid": list(delta_t_grid),
            "empirical": emp, "by_null": by_null}


def analyze_timing(event_times, cluster_labels_full, clean_for_timing,
                   delta_t_grid, n_perm=500, nulls=("N0", "N1", "N3"), seed=0):
    """Collision-censored label-independence at MODEL scale (spec §2, §3.2).
    `cluster_labels_full` is FULL-length (-1 for invalid events). P1-3: only
    clean AND validly-clustered (binary) events are passed to the null — collisions
    NEVER enter the shuffle; their effect is ONLY the block boundary (P1-1). Two
    views: censored (primary, collisions break runs) and clean_only (naive single
    block) contrast. Gate points: lag1_same inside null CI, pairing_excess inside
    CI, run_length lift ~1 (P1-3)."""
    times = np.asarray(event_times, float)
    labels = np.asarray(cluster_labels_full, int)
    keep = np.asarray(clean_for_timing, bool) & (labels >= 0)     # P1-3 filter
    blocks, _ = collision_free_blocks(times, keep)                 # P1-1 boundaries from full timeline
    t = times[keep]; y = labels[keep]                              # binary only -> no null pollution
    censored = _timing_for_blocks(t, y, blocks, delta_t_grid, n_perm, nulls, seed)
    one_block = [(float(t.min()), float(t.max()))] if t.size else []
    clean_only = _timing_for_blocks(t, y, one_block, delta_t_grid, n_perm, nulls, seed)
    a0 = censored["empirical"]["lag1_same"]; b0 = clean_only["empirical"]["lag1_same"]
    agree = (None if (a0 != a0 or b0 != b0)
             else bool(np.sign(a0 - 0.5) == np.sign(b0 - 0.5)))
    return {"n_kept": int(keep.sum()),
            "views": {"censored": censored, "clean_only": clean_only},
            "views_agree_sign": agree}
```

Gate read-out (spec §5): "independent" at a null = `lag1_same.inside_ci` True, every `pairing_excess@Δt.inside_ci` True, and `mean_run_length.inside_ci` True (equivalently lift ≈ 1). A control must violate these; the `test_timing_flags_alternating_not_shuffle` test enforces it.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sef_hfo_stage3.py -k timing -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_stage3_analysis.py tests/test_sef_hfo_stage3.py
git commit -m "feat(stage3): collision-censored model-scale timing + control invariant test"
```

---

## Phase 4 — Orchestrator + per-run gates → multi-seed

### Task 10: `run_..._stage3.py` orchestrator (per-run, no pooled p-value)

**Files:**
- Create: `scripts/run_sef_hfo_snn_cm_spontaneous_stage3.py`
- Test: manual multi-seed run (after pilot approval)

- [ ] **Step 1: Implement the orchestrator**

```python
"""Stage 3 orchestrator: run twoend_equal across N independent networks (connection
seed FREED), analyze each run on its OWN (structure + timing + synthetic controls +
entry-jitter), apply per-run gates, write stage3_summary.json. NO pooled p-value —
report per-run values + sign consistency (spec §3.2, user lock 2026-06-13)."""
import os, json, subprocess
import numpy as np
from src.sef_hfo_stage3 import synthetic_label_sequence, entry_jitter_stats
from src.sef_hfo_stage3_analysis import analyze_structure_one_record, analyze_timing

OUT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
# PINNED BY PILOT (Task 7) — fill in after the checkpoint:
T, DELTA_ONSET, N_MIN, IEI = 15000.0, 30.0, 5, None   # IEI seconds -> delta_t_grid

def run_one(seed):
    assert IEI is not None, "pin IEI from the pilot before running"
    tag = f"stage3_s{seed}"
    subprocess.run(["python3", "scripts/run_sef_hfo_snn_cm_spontaneous_readout.py",
        "--L", "20", "--density", "100", "--T", str(T), "--lesion", "twoend_equal",
        "--core-mean", "17.0", "--core-std", "1.5", "--seed", str(seed),
        "--delta-onset", str(DELTA_ONSET), "--n-min", str(N_MIN), "--tag", tag],
        check=True)
    side = json.load(open(os.path.join(OUT, f"sidecar_{tag}.json")))
    events = side["events"]
    hidden = [e["hidden_source_label"] for e in events]
    clean = np.array([e["clean_for_timing"] for e in events], bool)
    times = np.array([e["t_on"] / 1000.0 for e in events], float)  # seconds
    rec = os.path.join(OUT, "record", tag)
    struct = analyze_structure_one_record(rec, hidden)             # hidden is FULL length
    labels_full = np.array(struct["labels_full"], int)
    grid = [IEI, 2 * IEI, 4 * IEI]

    # main timing on full labels (analyze_timing filters to clean & labelled internally)
    main = analyze_timing(times, labels_full, clean, grid)

    # synthetic controls: SAME full timeline, SAME clean flags / collision boundaries
    # (P1 round-3) — ONLY the kept-event labels are re-ordered; collisions still censor.
    keep = clean & (labels_full >= 0)
    rng = np.random.default_rng(seed)
    ctrl = {}
    for m in ("alternating", "sticky", "shuffle"):
        y_synth = synthetic_label_sequence(labels_full[keep], m, rng)  # kept labels, time order
        labels_ctrl = np.full_like(labels_full, -1)
        labels_ctrl[keep] = y_synth
        ctrl[m] = analyze_timing(times, labels_ctrl, clean, grid)      # same blocks as main

    # entry-jitter (secondary) per direction, on clean events
    fc = struct.get("first_contacts", [None] * len(events))
    ej = {d: entry_jitter_stats([fc[i] for i in range(len(events))
                                 if hidden[i] == d and clean[i]])
          for d in ("neg", "pos")}

    n_neg = int(sum(1 for i in range(len(events)) if hidden[i] == "neg" and clean[i]))
    n_pos = int(sum(1 for i in range(len(events)) if hidden[i] == "pos" and clean[i]))
    coll_rate = sum(1 for h in hidden if h == "collision") / max(1, len(hidden))
    return dict(seed=seed, n_clean_neg=n_neg, n_clean_pos=n_pos, collision_rate=coll_rate,
                structure=struct, timing_main=main, timing_controls=ctrl, entry_jitter=ej)

def main():
    seeds = [1, 2, 3, 4, 5, 6]
    runs = [run_one(s) for s in seeds]
    # per-run gates; combined = sign consistency, NOT a pooled p-value
    json.dump(dict(seeds=seeds, runs=runs), open(os.path.join(OUT, "stage3_summary.json"), "w"),
              indent=2, default=lambda o: None)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: After pilot approval, fill the PINNED constants and run**

Run: `python3 scripts/run_sef_hfo_snn_cm_spontaneous_stage3.py`
Expected: `stage3_summary.json` with per-run structure + timing + controls.

- [ ] **Step 3: Verify the per-run gates by hand** (spec §5): clean/end ≥30, collision <20–30 %, stable_k==2, purity ≥0.9, swap strict/candidate, and for the main run's censored view `lag1_same.inside_ci` / every `pairing_excess@Δt.inside_ci` / `mean_run_length.inside_ci` all True (≈ independent), AND every run's `alternating`+`sticky` controls have those `inside_ci` False (caught). Report sign consistency across the 6 runs — NOT a pooled p-value.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_sef_hfo_snn_cm_spontaneous_stage3.py
git commit -m "feat(stage3): multi-seed orchestrator (per-run gates, sign consistency, no pooled p)"
```

---

## Phase 5 — Figure

### Task 11: `plot_..._stage3.py`

**Files:**
- Create: `scripts/plot_sef_hfo_snn_cm_spontaneous_stage3.py`
- Create/Update: `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/figures/README.md`

- [ ] **Step 1: Implement** the figure: reuse the Stage-2 core figure layout (a/b maps + electrode read-out + KMeans k=2 panel) for one representative run, PLUS a **timing panel**: the hidden-label sequence over time, and for **main vs alternating vs sticky vs shuffle** side by side, each metric's empirical value with its null **percentile CI** as an error bar — `lag1_same` (CI band ≈ 0.5 under independence), `mean_run_length` (with its null CI; lift annotated, ≈1 under independence), and `pairing_excess@Δt` for each Δt (CI band ≈ 0). The eye should see the main run's points landing **inside** the null CI bands while every control point lands **outside** (caught). Use the `views["censored"]["by_null"]["N1"]` numbers as the primary panel.
- [ ] **Step 2: Render, eyeball** (paper-grade, self-contained, no internal codenames in labels — per the figure memory), fix, re-render.
- [ ] **Step 3: Write `figures/README.md`** entry (Chinese, 2–4 sentences + `**关注点**：`).
- [ ] **Step 4: Commit.**

---

## Phase 6 — OPTIONAL/stretch: engine forced-ping-pong control

Only if the synthetic controls leave doubt about *physical* realizability of dependence. Requires injecting event-triggered, position-dependent external rate (via `simulate_kick`'s `nu_signal_fn`/`slow` hooks, NOT by editing the checksum-guarded engine loop). Heavy; deferred. The synthetic-label controls (Phase 3) already carry the falsifiability load.

---

## Self-Review

**Spec coverage:**
- §1 simulation (twoend_equal, core-onset label) → Task 6 ✓
- §2 collision-as-boundary + two views → Task 3 (blocks) + Task 9 (timing) ✓; *clean-only naive contrast view* → add to Task 9 reporting (compute timing a second time with collisions dropped-and-concatenated; both reported).
- §3.1 structure per-run + purity + shared-active corr (sanity, not gate) → Task 8 ✓
- §3.2 model-scale Δt + null points (lift∋1, excess∋0) → Task 9 ✓
- §3.3 entry-jitter → Task 5 (helper) + Task 10 (wired per direction) ✓
- §4 synthetic controls → Task 4 + Task 9/10 ✓; engine control → Phase 6 (optional) ✓
- §5 gates → Task 10 Step 3 ✓
- §6 sidecar schema + asserts (event_id, alignment, real times) → Task 6 ✓
- §8 pilot-first hard stop → Task 7 ✓

**Gap found & fixed inline:** §2's "clean-only naive" contrast view was described but not coded → now implemented in Task 9 (`analyze_timing` computes both `views["censored"]` (primary) and `views["clean_only"]` (contrast) and a `views_agree_sign` flag).

**Placeholder scan:** `T/DELTA_ONSET/N_MIN/IEI` in Task 10 are explicitly PILOT-PINNED (Task 7 gates them) — labelled, not stray TBDs. No other placeholders.

**Type consistency:** helper names (`core_onset`, `assign_hidden_label`, `collision_free_blocks`, `synthetic_label_sequence`, `entry_jitter_stats`) and analysis fns (`analyze_structure_one_record`, `analyze_timing`) are referenced consistently across Tasks 6/8/9/10.

### Review round 2 (2026-06-13) — fixes folded in

- **P1-1 label alignment:** `compute_adaptive_cluster_stereotypy` returns labels indexed into `valid_events`, not all events. Task 8 now rebuilds `labels_full` (-1 = invalid) with `_valid_event_indices(B, min_participating=3)` (matching `min_shared_channels` default), asserts `len(labels)==valid_ev.size`, uses `labels_full` for purity/templates/timing, keeps valid-length labels only for the PR-2.5 call. Returns `labels_full` (not `labels`); Task 10 reads `labels_full`.
- **P1-2 timing gate complete:** Task 9 now reports `lag1_same`, `mean_run_length` (+lift), and `pairing_excess@Δt`, each with a null **percentile CI** across nulls **N0/N1/N3** (one local null loop reusing the metric + `_generate_null_labels` primitives; no shared-module edit).
- **P1-3 null pollution:** Task 9 filters to `clean & (labels_full>=0)` BEFORE the null; collisions/invalid never enter the shuffle (binary labels only). Their only effect is the block boundary (P1-1).
- **P1-4 hard packedTimes assert:** Task 6 test asserts `packed[:,0] == sidecar t_on/1000` element-wise (not just monotone), and routes output under `tmp_path` via a new `--out`.
- **P1-5 E-only mask:** Task 6 slices `core_mask[:NE]` before indexing `E_spk_bool`.
- **Eng fixes:** Task 4 alternating assert rewritten (`sum(out[:-1]==out[1:])==0`); Task 10 controls operate on the matched kept subset (no length mismatch); default nulls include N0; entry-jitter wired per direction from `struct["first_contacts"]`.

### Review round 3 (2026-06-13) — fixes folded in

- **P1 (control block structure):** Task 10 synthetic controls now run on the FULL timeline with the SAME `clean` flags (collisions still censor as block boundaries); only the kept-event labels are re-ordered (`labels_ctrl[keep] = y_synth`, −1 elsewhere). The control is now a strict same-conditions control (same times, same blocks, relabeled only).
- **P2 (sticky test):** Task 9 gains `test_timing_flags_sticky` — sticky must be flagged dependent (lag1_same above null CI, run-length lift > 1).
- **P2 (flaky smoke):** Task 6 smoke uses core-mean 14.0 / T 600 (well below the ignition boundary → reliable firing) and asserts `len(ev) >= 1` with a clear remediation message.
- **P2 (figure metric names):** Phase 5 figure synced to the real Task 9 output — empirical ± null percentile CI for `lag1_same`, `mean_run_length` (+lift), and `pairing_excess@Δt`, main vs the three controls.
