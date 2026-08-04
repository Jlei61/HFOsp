# FCXR-LC3 event ledger — IMPLEMENTATION PLAN

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the per-event ledger that turns a no-kick trajectory from "was there an onset" into "how many interictal events, carrying how much load, were needed to reach onset".

**Architecture:** The ledger is pure post-processing. Every input already exists in memory when a reconnaissance row finishes — the full-resolution rate array, the 1 ms active-fraction series the event detector runs on, the complete 250 ms full-field snapshot series, the region masks and the lifecycle bout. Today all but a handful of selected snapshots are discarded and the rate is decimated 200x before it reaches disk. So this adds no simulation: it adds a pure function in `src/`, unit-tested against synthetic arrays, and one call site in the reconnaissance runner.

**Tech Stack:** Python 3.11, numpy, pytest. No new dependencies.

## Global Constraints

- Design of record: `docs/superpowers/specs/2026-08-04-topic4-fcxr-lc3-event-driven-pivot-design.md`.
- Regional decomposition over `core_A / core_B / axial / off_axis` is **mandatory** for every slow-variable readout. A whole-array mean is never a sufficient report — the slow-flow probes measured a mean `X` drift of `+0.033/s` while both cores depleted, because 85% of cells are off-axis.
- Calibration constants are taken from the frozen LC1 baseline contract, never re-derived: `floor_af = 3.125e-05`, event bar `0.03978125`, `af_bin_ms = 1.0`, `win_ms = 1000.0`.
- Accumulation bar is `>= 3` returning events before onset.
- Both doses are always reported together; neither may stand alone.
- Integration step `dt = 0.05 ms`; snapshot cadence `250 ms`.
- Every new `src/` symbol must be importable without touching a 40k substrate, so it can be tested in seconds.
- Tests run with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1` and the interpreter at `/home/honglab/leijiaxin/anaconda3/bin/python`.

## File Structure

- **Create `src/topic4_fcxr_lc3_ledger.py`** — every pure function. One responsibility: turn detected events plus slow-field snapshots into an auditable per-event ledger. Knows nothing about files, substrates or simulation.
- **Create `tests/test_topic4_fcxr_lc3_ledger.py`** — synthetic-array unit tests, no 40k network.
- **Modify `scripts/run_topic4_fcxr_lc3_recon.py`** — build the snapshot table and call `build_event_ledger` where the record is assembled, then persist `event_ledger` in the row JSON and the full snapshot table in the row NPZ.

---

### Task 1: Per-event dose on both scales

**Files:**
- Create: `src/topic4_fcxr_lc3_ledger.py`
- Test: `tests/test_topic4_fcxr_lc3_ledger.py`

**Interfaces:**
- Produces: `event_dose_af(af, af_bin_ms, event, floor_af) -> float`, `event_dose_rate(rate_hz, dt_ms, event, r_base_hz) -> float`. `event` is a dict with `t_on`/`t_off` in ms.

- [ ] **Step 1: Write the failing tests**

```python
def test_af_dose_subtracts_the_frozen_floor_and_clips_at_zero():
    af = np.array([0.0, 0.10, 0.20, 0.10, 0.0])       # 1 ms bins
    ev = dict(t_on=1.0, t_off=3.0)
    # (0.10-0.05) + (0.20-0.05) + (0.10-0.05) = 0.25, times 1 ms
    assert event_dose_af(af, 1.0, ev, 0.05) == pytest.approx(0.25)

def test_af_dose_is_zero_when_the_event_never_clears_the_floor():
    af = np.array([0.05, 0.04, 0.05])
    assert event_dose_af(af, 1.0, dict(t_on=0.0, t_off=2.0), 0.05) == 0.0

def test_rate_dose_integrates_at_the_full_step_not_the_decimated_one():
    # 12 ms event at dt=0.05 ms is 241 samples; a 10 ms decimation would see 2.
    rate = np.full(400, 30.0); rate[:100] = 2.0; rate[341:] = 2.0
    ev = dict(t_on=5.0, t_off=17.0)
    got = event_dose_rate(rate, 0.05, ev, 2.0)
    assert got == pytest.approx(28.0 * 241 * 0.05)

def test_rate_dose_clips_below_baseline_to_zero():
    rate = np.full(100, 1.0)
    assert event_dose_rate(rate, 0.05, dict(t_on=0.0, t_off=1.0), 5.0) == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_topic4_fcxr_lc3_ledger.py -v`
Expected: FAIL, `ImportError` / name not defined.

- [ ] **Step 3: Implement**

```python
def _slice(n, t_on_ms, t_off_ms, step_ms):
    i0 = max(0, int(round(float(t_on_ms) / float(step_ms))))
    i1 = min(int(n), int(round(float(t_off_ms) / float(step_ms))) + 1)
    return i0, i1


def event_dose_af(af, af_bin_ms, event, floor_af) -> float:
    """Active-fraction dose, on the same series and floor the detector uses."""
    a = np.asarray(af, dtype=float)
    i0, i1 = _slice(a.size, event["t_on"], event["t_off"], af_bin_ms)
    if i1 <= i0:
        return 0.0
    return float(np.clip(a[i0:i1] - float(floor_af), 0.0, None).sum() * float(af_bin_ms))


def event_dose_rate(rate_hz, dt_ms, event, r_base_hz) -> float:
    """Population-rate dose at the full integration step."""
    r = np.asarray(rate_hz, dtype=float)
    i0, i1 = _slice(r.size, event["t_on"], event["t_off"], dt_ms)
    if i1 <= i0:
        return 0.0
    return float(np.clip(r[i0:i1] - float(r_base_hz), 0.0, None).sum() * float(dt_ms))
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_topic4_fcxr_lc3_ledger.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/topic4_fcxr_lc3_ledger.py tests/test_topic4_fcxr_lc3_ledger.py
git commit -m "feat(topic4): per-event dose on the active-fraction and rate scales"
```

---

### Task 2: Regional snapshot table

**Files:**
- Modify: `src/topic4_fcxr_lc3_ledger.py`
- Test: `tests/test_topic4_fcxr_lc3_ledger.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `REGION_KEYS`, `regional_means(field, masks) -> dict`, `snapshot_table(snapshots, dt_ms, masks) -> list[dict]` where each row is `dict(t_ms, label, D, H, X, y)` and each of `D/H/X/y` is a `dict` over `REGION_KEYS` plus `"all"`. Rows are sorted by `t_ms`.

- [ ] **Step 1: Write the failing tests**

```python
def _masks():
    m = {k: np.zeros(8, bool) for k in REGION_KEYS}
    m["core_A"][0:2] = True; m["core_B"][2:4] = True
    m["axial"][4:6] = True; m["off_axis"][6:8] = True
    return m

def test_regional_means_split_by_mask_and_keep_the_whole_array_too():
    got = regional_means(np.array([1., 1., 2., 2., 3., 3., 4., 4.]), _masks())
    assert got == pytest.approx(dict(core_A=1.0, core_B=2.0, axial=3.0, off_axis=4.0, all=2.5))

def test_snapshot_table_is_sorted_by_time_and_converts_steps_to_ms():
    snaps = {
        "t500": dict(step=10000, z_E=np.zeros(8), h_E=np.ones(8), x_E=np.full(8, 0.5), y_E=np.zeros(8)),
        "t250": dict(step=5000,  z_E=np.ones(8),  h_E=np.zeros(8), x_E=np.ones(8),     y_E=np.ones(8)),
    }
    table = snapshot_table(snaps, 0.05, _masks())
    assert [r["t_ms"] for r in table] == [250.0, 500.0]
    assert table[0]["label"] == "t250"
    assert table[0]["D"]["all"] == pytest.approx(0.0)   # D = 1 - z
    assert table[1]["D"]["all"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_topic4_fcxr_lc3_ledger.py -k "regional or snapshot_table" -v`
Expected: FAIL, name not defined.

- [ ] **Step 3: Implement**

```python
REGION_KEYS = ("core_A", "core_B", "axial", "off_axis")


def regional_means(field, masks) -> dict:
    """Regional means plus the whole-array mean.

    The whole-array value is carried only as context: the slow-flow probes measured
    a mean X drift opposite in sign to both cores, so a mean alone inverts the result.
    """
    a = np.asarray(field, dtype=float)
    if set(masks) != set(REGION_KEYS):
        raise ValueError(f"masks must be exactly {REGION_KEYS}")
    out = {}
    for key in REGION_KEYS:
        mask = np.asarray(masks[key], dtype=bool)
        if mask.shape != a.shape:
            raise ValueError(f"mask {key} shape {mask.shape} != field {a.shape}")
        if not mask.any():
            raise ValueError(f"region {key} is empty")
        out[key] = float(a[mask].mean())
    out["all"] = float(a.mean())
    return out


def snapshot_table(snapshots, dt_ms, masks) -> list:
    """Regional D/H/X/y for every retained full-field snapshot, ordered in time."""
    rows = []
    for label, snap in snapshots.items():
        rows.append(dict(
            t_ms=float(snap["step"]) * float(dt_ms), label=str(label),
            D=regional_means(1.0 - np.asarray(snap["z_E"], float), masks),
            H=regional_means(snap["h_E"], masks),
            X=regional_means(snap["x_E"], masks),
            y=regional_means(snap["y_E"], masks),
        ))
    rows.sort(key=lambda r: (r["t_ms"], r["label"]))
    return rows
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_topic4_fcxr_lc3_ledger.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/topic4_fcxr_lc3_ledger.py tests/test_topic4_fcxr_lc3_ledger.py
git commit -m "feat(topic4): regional snapshot table for the event ledger"
```

---

### Task 3: Bracketing snapshots and entry classification

**Files:**
- Modify: `src/topic4_fcxr_lc3_ledger.py`
- Test: `tests/test_topic4_fcxr_lc3_ledger.py`

**Interfaces:**
- Consumes: `snapshot_table` rows from Task 2.
- Produces: `bracketing_snapshots(table, t_on_ms, t_off_ms) -> (pre_or_None, post_or_None)`, `classify_entry(n_returning_before_onset, onset_ms) -> str`, `ACCUMULATION_BAR = 3`.

- [ ] **Step 1: Write the failing tests**

```python
def test_bracketing_picks_last_before_onset_and_first_after_offset():
    table = [dict(t_ms=t) for t in (0.0, 250.0, 500.0, 750.0, 1000.0)]
    pre, post = bracketing_snapshots(table, 510.0, 522.0)
    assert pre["t_ms"] == 500.0 and post["t_ms"] == 750.0

def test_bracketing_returns_none_when_no_snapshot_exists_on_a_side():
    table = [dict(t_ms=500.0)]
    pre, post = bracketing_snapshots(table, 100.0, 120.0)
    assert pre is None and post["t_ms"] == 500.0
    pre, post = bracketing_snapshots(table, 900.0, 920.0)
    assert pre["t_ms"] == 500.0 and post is None

@pytest.mark.parametrize("n,onset,expected", [
    (0, 1000.0, "ONE_SHOT"), (1, 1000.0, "ONE_SHOT"), (2, 1000.0, "AMBIGUOUS_2"),
    (3, 1000.0, "CUMULATIVE"), (9, 1000.0, "CUMULATIVE"),
    (0, None, "NO_ONSET"), (7, None, "NO_ONSET"),
])
def test_entry_class_never_collapses_the_accumulation_question(n, onset, expected):
    assert classify_entry(n, onset) == expected
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_topic4_fcxr_lc3_ledger.py -k "bracketing or entry_class" -v`
Expected: FAIL, name not defined.

- [ ] **Step 3: Implement**

```python
ACCUMULATION_BAR = 3


def bracketing_snapshots(table, t_on_ms, t_off_ms):
    """Nearest snapshot at or before the event start, and at or after its end."""
    pre = None
    for row in table:
        if row["t_ms"] <= float(t_on_ms):
            pre = row
        else:
            break
    post = next((row for row in table if row["t_ms"] >= float(t_off_ms)), None)
    return pre, post


def classify_entry(n_returning_before_onset, onset_ms) -> str:
    """Entry class; reported alongside the count, never in place of it."""
    if onset_ms is None:
        return "NO_ONSET"
    n = int(n_returning_before_onset)
    if n >= ACCUMULATION_BAR:
        return "CUMULATIVE"
    if n == 2:
        return "AMBIGUOUS_2"
    return "ONE_SHOT"
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_topic4_fcxr_lc3_ledger.py -v`
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add src/topic4_fcxr_lc3_ledger.py tests/test_topic4_fcxr_lc3_ledger.py
git commit -m "feat(topic4): snapshot bracketing and entry classification"
```

---

### Task 4: The ledger itself

**Files:**
- Modify: `src/topic4_fcxr_lc3_ledger.py`
- Test: `tests/test_topic4_fcxr_lc3_ledger.py`

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces: `build_event_ledger(*, events, af, af_bin_ms, floor_af, rate_hz, dt_ms, r_base_hz, table, onset_ms, offset_ms, total_ms) -> dict` with keys `schema, calibration, onset_ms, offset_ms, total_ms, n_events, n_returning, n_events_before_onset, n_returning_before_onset, entry_class, Q_af_to_onset, Q_rate_to_onset, first_non_returning_index, events, post_offset`.

- [ ] **Step 1: Write the failing tests**

```python
def _ledger_case(onset_ms):
    events = [dict(t_on=100.0 + 1000.0 * k, t_off=110.0 + 1000.0 * k,
                   dur_ms=10.0, peak_ext=0.05, returned=True) for k in range(4)]
    af = np.zeros(5000); rate = np.full(100000, 2.0)
    for k in range(4):
        af[100 + 1000 * k: 111 + 1000 * k] = 0.10
        rate[int((100 + 1000 * k) / 0.05): int((111 + 1000 * k) / 0.05)] = 30.0
    masks = _masks()
    snaps = {}
    for i, t in enumerate(np.arange(0.0, 4500.0, 250.0)):
        z = np.full(8, 1.0 - 0.01 * i)
        snaps[f"t{int(t)}"] = dict(step=int(round(t / 0.05)), z_E=z, h_E=np.full(8, 0.02 * i),
                                   x_E=np.full(8, 1.0), y_E=np.zeros(8))
    return events, af, rate, snapshot_table(snaps, 0.05, masks)

def test_four_returning_events_before_onset_read_as_cumulative_entry():
    events, af, rate, table = _ledger_case(4200.0)
    led = build_event_ledger(events=events, af=af, af_bin_ms=1.0, floor_af=0.05,
                             rate_hz=rate, dt_ms=0.05, r_base_hz=2.0, table=table,
                             onset_ms=4200.0, offset_ms=None, total_ms=5000.0)
    assert led["n_returning_before_onset"] == 4
    assert led["entry_class"] == "CUMULATIVE"
    assert led["first_non_returning_index"] is None
    assert [e["phase"] for e in led["events"]] == ["pre_onset"] * 4

def test_cumulative_dose_is_monotone_and_both_scales_are_present():
    events, af, rate, table = _ledger_case(4200.0)
    led = build_event_ledger(events=events, af=af, af_bin_ms=1.0, floor_af=0.05,
                             rate_hz=rate, dt_ms=0.05, r_base_hz=2.0, table=table,
                             onset_ms=4200.0, offset_ms=None, total_ms=5000.0)
    q_af = [e["Q_af"] for e in led["events"]]
    q_rate = [e["Q_rate"] for e in led["events"]]
    assert q_af == sorted(q_af) and q_rate == sorted(q_rate)
    assert all(e["dose_af"] > 0 and e["dose_rate"] > 0 for e in led["events"])
    assert led["Q_af_to_onset"] == pytest.approx(q_af[-1])

def test_no_onset_is_reported_as_such_not_as_one_shot():
    events, af, rate, table = _ledger_case(None)
    led = build_event_ledger(events=events, af=af, af_bin_ms=1.0, floor_af=0.05,
                             rate_hz=rate, dt_ms=0.05, r_base_hz=2.0, table=table,
                             onset_ms=None, offset_ms=None, total_ms=5000.0)
    assert led["entry_class"] == "NO_ONSET"
    assert led["Q_af_to_onset"] is None

def test_per_event_slow_state_keeps_regions_whose_mean_would_invert_them():
    # core falls, off-axis rises; the whole-array mean rises. The ledger must show both.
    events = [dict(t_on=300.0, t_off=310.0, dur_ms=10.0, peak_ext=0.05, returned=True)]
    af = np.zeros(1000); af[300:311] = 0.10
    rate = np.full(20000, 2.0); rate[6000:6220] = 30.0
    masks = _masks()
    def snap(step, core_x, off_x):
        x = np.empty(8); x[0:4] = core_x; x[4:8] = off_x
        return dict(step=step, z_E=np.zeros(8), h_E=np.zeros(8), x_E=x, y_E=np.zeros(8))
    table = snapshot_table({"a": snap(5000, 0.90, 0.50), "b": snap(10000, 0.70, 0.95)},
                           0.05, masks)
    led = build_event_ledger(events=events, af=af, af_bin_ms=1.0, floor_af=0.05,
                             rate_hz=rate, dt_ms=0.05, r_base_hz=2.0, table=table,
                             onset_ms=None, offset_ms=None, total_ms=1000.0)
    delta = led["events"][0]["delta"]["X"]
    assert delta["all"] > 0            # the mean says X recovered
    assert delta["core_A"] < 0         # both cores actually depleted
    assert delta["core_B"] < 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_topic4_fcxr_lc3_ledger.py -k ledger -v`
Expected: FAIL, name not defined.

- [ ] **Step 3: Implement**

```python
LEDGER_SCHEMA = "fcxr-lc3-event-ledger-1.0"


def _delta(pre, post):
    if pre is None or post is None:
        return None
    return {var: {key: float(post[var][key] - pre[var][key]) for key in post[var]}
            for var in ("D", "H", "X", "y")}


def _phase(event, onset_ms, offset_ms):
    if onset_ms is None or event["t_off"] < float(onset_ms):
        return "pre_onset"
    if offset_ms is not None and event["t_on"] > float(offset_ms):
        return "post_offset"
    return "ictal"


def build_event_ledger(*, events, af, af_bin_ms, floor_af, rate_hz, dt_ms,
                       r_base_hz, table, onset_ms, offset_ms, total_ms) -> dict:
    """Per-event ledger: how many events, carrying how much load, and the slow state
    they left behind. Every slow readout is regional; the whole-array mean is context."""
    rate = np.asarray(rate_hz, dtype=float)
    rows, q_af, q_rate = [], 0.0, 0.0
    for k, ev in enumerate(events, start=1):
        d_af = event_dose_af(af, af_bin_ms, ev, floor_af)
        d_rate = event_dose_rate(rate, dt_ms, ev, r_base_hz)
        q_af += d_af; q_rate += d_rate
        i0, i1 = _slice(rate.size, ev["t_on"], ev["t_off"], dt_ms)
        pre, post = bracketing_snapshots(table, ev["t_on"], ev["t_off"])
        rows.append(dict(
            index=k, t_on_ms=float(ev["t_on"]), t_off_ms=float(ev["t_off"]),
            dur_ms=float(ev["dur_ms"]), peak_ext=float(ev["peak_ext"]),
            returned=bool(ev["returned"]),
            peak_rate_hz=float(rate[i0:i1].max()) if i1 > i0 else 0.0,
            dose_af=d_af, dose_rate=d_rate, Q_af=q_af, Q_rate=q_rate,
            phase=_phase(ev, onset_ms, offset_ms),
            pre=None if pre is None else dict(
                t_ms=pre["t_ms"], lag_ms=float(ev["t_on"] - pre["t_ms"]),
                **{v: pre[v] for v in ("D", "H", "X", "y")}),
            post=None if post is None else dict(
                t_ms=post["t_ms"], lag_ms=float(post["t_ms"] - ev["t_off"]),
                **{v: post[v] for v in ("D", "H", "X", "y")}),
            delta=_delta(pre, post),
        ))
    before = [r for r in rows if r["phase"] == "pre_onset"]
    n_ret_before = sum(1 for r in before if r["returned"])
    non_ret = next((r["index"] for r in rows if not r["returned"]), None)
    after = [r for r in rows if r["phase"] == "post_offset" and r["returned"]]
    iei = [after[i]["t_on_ms"] - after[i - 1]["t_on_ms"] for i in range(1, len(after))]
    return dict(
        schema=LEDGER_SCHEMA,
        calibration=dict(floor_af=float(floor_af), af_bin_ms=float(af_bin_ms),
                         dt_ms=float(dt_ms), r_base_hz=float(r_base_hz),
                         accumulation_bar=ACCUMULATION_BAR,
                         r_base_definition="pre-onset quiet median population rate of this run"),
        onset_ms=onset_ms, offset_ms=offset_ms, total_ms=float(total_ms),
        n_events=len(rows), n_returning=sum(1 for r in rows if r["returned"]),
        n_events_before_onset=len(before), n_returning_before_onset=n_ret_before,
        entry_class=classify_entry(n_ret_before, onset_ms),
        Q_af_to_onset=(before[-1]["Q_af"] if before and onset_ms is not None else None),
        Q_rate_to_onset=(before[-1]["Q_rate"] if before and onset_ms is not None else None),
        first_non_returning_index=non_ret,
        events=rows,
        post_offset=dict(
            n_returning=len(after),
            durations_ms=[r["dur_ms"] for r in after],
            participation=[r["peak_ext"] for r in after],
            iei_ms=iei,
        ),
    )
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_topic4_fcxr_lc3_ledger.py -v`
Expected: 19 passed.

- [ ] **Step 5: Commit**

```bash
git add src/topic4_fcxr_lc3_ledger.py tests/test_topic4_fcxr_lc3_ledger.py
git commit -m "feat(topic4): build the per-event ledger with regional slow state"
```

---

### Task 5: Persist the ledger from the reconnaissance runner

**Files:**
- Modify: `scripts/run_topic4_fcxr_lc3_recon.py` — the import block, and the record assembly around the existing `field_summaries` / `_write_npz` / `record` block.
- Test: `tests/test_topic4_fcxr_lc3_ledger.py`

**Interfaces:**
- Consumes: `build_event_ledger`, `snapshot_table` from Tasks 2 and 4.
- Produces: `record["event_ledger"]` in each `recon_noise<seed>.json`, and `snapshot_t_ms` / `snapshot_<var>_<region>` arrays in the row NPZ.

- [ ] **Step 1: Write the failing test**

```python
def test_recon_runner_persists_the_ledger_and_the_full_snapshot_table():
    import ast, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "scripts", "run_topic4_fcxr_lc3_recon.py")).read()
    tree = ast.parse(src)
    body = ast.dump(tree)
    # the bearing measurement must reach disk, not just exist in memory
    assert "build_event_ledger" in body
    assert "event_ledger" in body
    assert "snapshot_t_ms" in body
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_topic4_fcxr_lc3_ledger.py -k persists -v`
Expected: FAIL on `assert "build_event_ledger" in body`.

- [ ] **Step 3: Wire it in**

Add to the import block of `scripts/run_topic4_fcxr_lc3_recon.py`:

```python
from src.topic4_fcxr_lc3_ledger import (  # noqa: E402
    build_event_ledger,
    snapshot_table,
)
```

Immediately after `masks = GEO._region_masks(S)` in `_run_once`, build the table and
the ledger. `r_base` is the run's own pre-onset quiet median, so it is auditable:

```python
    full_table = snapshot_table(slow_final.snapshots, E01.DT, masks)
    quiet_end = int(round((onset_ms if onset_ms is not None else total_ms) / E01.DT))
    r_base_hz = float(np.median(rate_e[:max(quiet_end, 1)]))
    event_ledger = build_event_ledger(
        events=events, af=_af, af_bin_ms=_af_dt, floor_af=float(baseline["floor_af"]),
        rate_hz=rate_e, dt_ms=E01.DT, r_base_hz=r_base_hz, table=full_table,
        onset_ms=onset_ms,
        offset_ms=(None if bout is None
                   else float((bout[1] + 1) * baseline["band"]["win_ms"])),
        total_ms=total_ms,
    )
```

Add to the `_write_npz(...)` call, so the continuous slow trajectory survives:

```python
        snapshot_t_ms=np.asarray([r["t_ms"] for r in full_table], np.float32),
        **{f"snapshot_{var}_{region}":
           np.asarray([r[var][region] for r in full_table], np.float32)
           for var in ("D", "H", "X", "y")
           for region in ("core_A", "core_B", "axial", "off_axis", "all")},
```

Add to `record`:

```python
        event_ledger=event_ledger,
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_topic4_fcxr_lc3_ledger.py -v` then the full LC3 suite:

```bash
python -m pytest -q tests/test_topic4_fcxr_lc3.py tests/test_topic4_fcxr_lc3_geometry.py \
  tests/test_topic4_fcxr_lc3_slowflow.py tests/test_topic4_fcxr_lc3_recon.py \
  tests/test_topic4_fcxr_lc3_spatial.py tests/test_topic4_fcxr_lc3_xcal.py \
  tests/test_topic4_fcxr_lc3_finalize.py tests/test_topic4_fcxr_lc3_ledger.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_topic4_fcxr_lc3_recon.py tests/test_topic4_fcxr_lc3_ledger.py
git commit -m "feat(topic4): persist the event ledger and the full snapshot table"
```

---

## Re-run note

`scripts/run_topic4_fcxr_lc3_recon.py` is inside the reconnaissance execution lock, so
Task 5 moves that lock's git head and completed rows stop satisfying the resume check
(`_run_once` compares `source_lock_git_head`). Land Task 5 only when no reconnaissance
row is in flight, or accept that in-flight work is re-run. The ledger adds no simulation:
a re-run costs exactly what the original run cost, and returns the bearing measurement
the original could not produce.
