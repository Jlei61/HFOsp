# Topic 5 slow-state two-timescale interictal repertoire model v4.0 — implementation plan (rev2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find, per patient, the smallest number of consecutive interictal events at which the
local propagation repertoire rises above that patient's own chance level; use that as the
model step; fit a slow latent state over those blocks; adjudicate each patient separately.

**Architecture:** Phase 0 resolves recording coverage from metadata (never event density) and
builds session/block manifests. Phase 1 measures three within-window agreements over
independently tiled windows and derives two distinct scales — `N_obs` (finest legible
resolution, which becomes the model step) and `N_break` (where windows start mixing states) —
plus a `state_geometry` verdict against nulls that KMeans is not invariant to. Phase 2 fits a
two-layer model whose emission factorises into participation and conditional ordering with
ties, calibrated first on synthetic sequences. Phase 3 adjudicates each admitted patient.

**Tech Stack:** Python 3.11 (`/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`),
NumPy, SciPy, pandas, PyYAML, PyTorch (CPU), pytest.

**Spec:** `docs/superpowers/specs/2026-08-04-topic5-slow-state-two-timescale-v4_0.md` (rev2).
Section references below point there.

## Global Constraints

- **Model step = `N_obs,p`**, the *smallest* window beating the patient's own chance level.
  Never the largest window below `N_break`. Spec §6.1.
- **Recording coverage from metadata only.** Reuse `build_source_segments`; never infer
  coverage from first/last event. Spec §4.1.
- **Ties come from `event_group_ids`**, never from equal `event_local_rank` values. A tied pair
  contributes 0.5 to precedence. Spec §6.2.
- **Session join threshold:** 300 s on *metadata* gaps, with matching continuity group and
  montage hash.
- **`metadata_gap_seconds` and `event_silence_seconds` are separate fields**, never conflated.
- **State carries across gaps with uncertainty growing in `Δt`.** Within-session transitions
  are primary; cross-gap transitions are a separately labelled stratum. Spec §4.3.
- **200 random splits are one window's uncertainty, not 200 replicates.** Primary windows are
  non-overlapping within a session; the statistical unit is the patient. Spec §6.4.
- **Scale pattern:** `BELOW_CHANCE`\* → `RELIABLE`\+ → `CHRONOLOGY_BREAK`\*. A leading
  below-chance run is expected and must not force unresolved. Spec §6.3.
- **Dwell is an interval** `[N_last_reliable, N_break)`, never a point. Spec §6.5.
- **`N` grid:** `{50,100,200,500,1000,2000}`; `N=5000` diagnostic only.
  **`ΔT` grid:** `{300,900,1800,3600,7200,14400,21600}` s.
- **Geometry nulls:** per-dimension circular shift (cluster), phase surrogate (continuous).
  Never block-row shuffling. Spec §7.
- **Emission factorises** as `p(m_e|z_b)·p(r_e,ties|m_e,z_b)`. No dense Gaussian. Spec §8.1.
- **Causal recursion locked:** `h_b = update(h_{b-1}, X_b)`, `X_{b+1} ~ p(X|h_b)`. Spec §8.2.
- **Event rate is conditioned on, not predicted**, in the primary. Spec §9.
- **No `session_index_normalised`.** Elapsed recording time plus a per-recording intercept.
- **Admission = `N_obs` exists AND ≥40 blocks.** Forty is a coverage floor only; the
  interpretable regime comes from the synthetic calibration. Spec §11–12.
- **Phase 0 never edits the spec.** Disagreement writes `PHASE0_RECONCILIATION.md` and stops.
- **No absolute reliability threshold.** Every scale decision is against the patient's own null.
- **Forbidden inputs at every phase:** old heldout20, ictal, SNN, SOZ, geometry, A-B axis.
- **Workers are configurable, default 8.** Raise only after peak RSS is recorded.
- **Output root:** `results/topic5_slow_state_two_timescale/v4_0/`.
- **V2.7 / V3.0 / V3.1 frozen artifacts are never rewritten.** V3.1 stays `NOT_TRIGGERED`.
- Every runner writes JSON atomically and records `runner_sha256` plus the SHA256 of each
  `src/` module it imports.

## File structure

| File | Responsibility |
| --- | --- |
| `config/topic5_slow_state_v4_0.yaml` | frozen grids, thresholds, support floors, workers |
| `src/topic5_source_intervals.py` | metadata-derived source intervals, extracted from the v3.0 audit script |
| `src/topic5_slow_state_sessions.py` | session join, block tiling, gap semantics, transition strata |
| `src/topic5_slow_state_repertoire.py` | three primary descriptors with ties and support floors |
| `src/topic5_slow_state_windows.py` | independent window enumeration per scale |
| `src/topic5_slow_state_scale.py` | three agreements, `N_obs`, `N_break`, dwell interval |
| `src/topic5_slow_state_geometry.py` | geometry verdict against circular-shift / phase nulls |
| `src/topic5_slow_state_model.py` | encoder, pooling, transition, factorised decoder, nuisances |
| `src/topic5_slow_state_templates.py` | secondary template occupancy and dispersion |
| `src/topic5_slow_state_acceptance.py` | per-patient checks, dwell null, regime power lookup |
| `scripts/run_topic5_slow_state_phase0_manifest.py` | Phase 0 |
| `scripts/run_topic5_slow_state_phase1_scale.py` | Phase 1 |
| `scripts/run_topic5_slow_state_synthetic_calibration.py` | Phase 2 prerequisite |
| `scripts/freeze_topic5_slow_state_phase2_release.py` | release gate |
| `scripts/run_topic5_slow_state_phase2_model.py` | Phase 2 |
| `scripts/accept_topic5_slow_state_v4_0.py` | cohort prevalence adjudication |

Tests mirror each module as `tests/test_<module>.py`.

---

## rev3 amendments — apply these before continuing

The second design review (2026-08-05, verdict `CONDITIONAL_GO_TASK7_PHASE0`) found four
contract defects and two engineering defects in the Tasks 1-6 implementation. Task 6 is
`CONDITIONAL_ACCEPTED`; Phase 0 (Task 7) may run on all 34 once these land; **Phase 1 (Task 8)
does not open until R3-C is implemented**. Spec rev3 §0 carries the same list.

Every test below ships with its exact numeric fixture and the exact assertion, and every one
carries a deliberate-break instruction. Three times in Tasks 5 and 6 a prose acceptance
criterion was satisfied in form and defeated in substance, so a test nobody has watched fail
is not accepted as evidence.

### R3-A — `window_state`: a family tie is not reliability

`src/topic5_slow_state_scale.py`. Among above-chance families the outcome is three-way:

| chronology vote among above-chance families | window |
| --- | --- |
| strict majority say break | `CHRONOLOGY_BREAK` |
| strict majority say no break | `RELIABLE` |
| neither, a tie | `UNRESOLVED_FAMILY_DISCORDANCE` |

Add the new label to the module's exported alphabet and to `scale_states`' known set.

- [ ] Test `test_a_family_level_tie_is_discordance_not_reliable`: exactly two above-chance
      families, one whose chronological value sits below its random-half `alpha` quantile and
      one whose does not. Assert `== "UNRESOLVED_FAMILY_DISCORDANCE"`, and assert
      `!= "RELIABLE"` on the same call.
- [ ] Deliberate break: restore `return RELIABLE` as the final fallthrough, watch this test
      FAIL, revert, watch it pass. Paste all four outputs.

### R3-B — `scale_states`: majority, not mode

- [ ] Drop `UNRESOLVED_FAMILIES`, `UNRESOLVED_FAMILY_DISCORDANCE` and any other non-evaluable
      verdict before counting; they never vote.
- [ ] Re-check `min_windows` against the SURVIVING evaluable count, not the original count.
- [ ] Emit `BELOW_CHANCE` / `RELIABLE` / `CHRONOLOGY_BREAK` only on a strict majority of the
      evaluable windows; otherwise emit `UNRESOLVED_MIXED_WINDOWS`.
- [ ] Test `test_two_of_five_windows_cannot_name_a_scale`: input
      `["RELIABLE","RELIABLE","CHRONOLOGY_BREAK","CHRONOLOGY_BREAK","UNRESOLVED_FAMILIES"]`
      with `min_windows=4`. One unevaluable is dropped, leaving 4 evaluable, which still meets
      the minimum; 2 of 4 is not a strict majority. Assert `"UNRESOLVED_MIXED_WINDOWS"`.
- [ ] Test `test_a_three_way_split_is_mixed_not_the_modal_label`: input
      `["RELIABLE","RELIABLE","BELOW_CHANCE","CHRONOLOGY_BREAK","UNRESOLVED_FAMILIES"]` with
      `min_windows=4`. Assert `"UNRESOLVED_MIXED_WINDOWS"` — 2 of 4 is not a majority even
      though `RELIABLE` is the mode.
- [ ] Test `test_dropping_unevaluable_windows_can_take_a_scale_below_the_minimum`: input
      `["RELIABLE","RELIABLE","RELIABLE","UNRESOLVED_FAMILIES","UNRESOLVED_FAMILIES"]` with
      `min_windows=4`. Only 3 evaluable survive. Assert `"UNRESOLVED_TOO_FEW_WINDOWS"`, not
      `"RELIABLE"`.
- [ ] Deliberate break: restore the mode-with-tiebreak implementation, watch the first two
      tests FAIL, revert, watch them pass.

### R3-C — two scale curves, backbone and residual (BLOCKS TASK 8)

Spec §6.6. The raw descriptors are dominated by the patient's stable backbone, so scale curves
computed on them can re-prove the existing split-half result and present it as a slow-state
timescale. `window_agreements` gains a `residualise` argument:

- `residualise=False` — raw curve, quality control, yields `N_obs_backbone`.
- `residualise=True` — the patient's global per-contact and per-pair main effects, estimated on
  the TRAIN portion only and passed in, are subtracted from every window's descriptors before
  the agreements are computed. Yields `N_obs_state`.

Downstream: block size is `max(N_obs_backbone, N_obs_state)`; `N_break` comes from the residual
chronology curve only. A patient whose residual is never estimable is `slow state not
observable`, never `no stable network`.

- [ ] `estimate_backbone(train_repertoires) -> dict` returning the per-contact and per-pair
      main effects, fitted on train windows only.
- [ ] Test `test_residualising_removes_a_constant_backbone_offset`: build 40 windows whose
      participation and mean rank are a fixed backbone vector plus i.i.d. noise of scale 0.01.
      Assert the raw `mean_rank` agreement median exceeds 0.95 while the residualised one falls
      below 0.3 — the backbone alone produces near-perfect raw agreement that carries no
      slow-state information.
- [ ] Test `test_residualising_preserves_a_real_local_deviation`: same backbone, but windows
      1-20 carry a systematic deviation `+0.5` on contacts 0-2 and windows 21-40 carry `-0.5`.
      Assert the residualised chronological agreement across that boundary is negative while
      the raw one stays above 0.8.
- [ ] Test `test_the_backbone_is_estimated_on_train_windows_only`: change only the held-out
      windows and assert `estimate_backbone` returns an identical result.
- [ ] Deliberate break: fit the backbone on all windows instead of train only, watch the third
      test FAIL, revert, watch it pass.

### R3-D — three elapsed-time fields on a block

`src/topic5_slow_state_sessions.py`. `delta_t_from_previous` is currently first-event minus
previous-last-event, which is an inter-block inter-event interval, not the elapsed time between
two slow-state observations. Replace it with three named fields:

- `transition_delta_t` — block centre to block centre. This is what the transition consumes.
- `inter_block_gap` — this block's first event minus the previous block's last event.
- `metadata_gap_seconds` — unobserved wall time across a session boundary, or `None` within one
  session.

- [ ] Test `test_transition_delta_t_is_centre_to_centre_not_edge_to_edge`: one session, events
      at `t = 0,1,2,3,10,11,12,13`, `block_events=4`. Block 0 spans 0-3 (centre 1.5), block 1
      spans 10-13 (centre 11.5). Assert `transition_delta_t == 10.0` and
      `inter_block_gap == 7.0`. These differ, so an implementation using either one for the
      other fails.
- [ ] Test `test_metadata_gap_is_none_within_one_session`: assert `metadata_gap_seconds is None`
      for a block whose previous block is in the same session.
- [ ] Deliberate break: set `transition_delta_t = inter_block_gap`, watch the first test FAIL,
      revert, watch it pass.
- [ ] Update every existing caller and test that referenced `delta_t_from_previous`.

### R3-E — the inventory test measures the environment, not the code

`tests/test_topic5_source_intervals.py::test_a_record_without_an_inventory_row_fails_loudly`
reads `results/epilepsiae_block_inventory.csv`, a data artifact that is not in git. A clean
checkout of this branch runs 109 tests and this one fails.

- [ ] Rewrite it hermetically: build a two-row inventory CSV in `tmp_path` with the columns the
      resolver requires, point the config at it, and assert `RuntimeError` for a record name
      absent from that CSV.
- [ ] Verify by running the file from a clean `git archive` extraction, not from the working
      tree. Paste that command and its output.

### R3-F — Task 7 must prove every event is placed exactly once

Add to Task 7's contract, and to its tests:

- [ ] Test `test_every_included_event_is_assigned_to_exactly_one_session`: the concatenation of
      all sessions' `event_indices` is a permutation of the analysis-eligible index set — no
      duplicate, no omission. Assert on sorted arrays, not on counts alone.
- [ ] Test `test_every_blocked_event_is_assigned_to_exactly_one_block`: the concatenation of all
      blocks' `event_indices` plus the dropped remainders equals the per-session event set.
- [ ] Phase 0 output carries `phase_role = "coverage_only_no_slow_state_inference"`.

---

## Task 1: Frozen configuration

**Files:** Create `config/topic5_slow_state_v4_0.yaml`; Test `tests/test_topic5_slow_state_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: the YAML every runner loads. Keys: `contract`, `dataset_root`,
  `source_mapping_root`, `epilepsiae_block_inventory`, `yuquan_block_inventory`,
  `output_root`, `session_join_seconds`, `event_window_grid`, `diagnostic_only_windows`,
  `clock_window_grid`, `min_windows_per_scale`, `min_events_for_descriptor`,
  `min_participation_count`, `min_pair_coparticipation_count`, `min_resolved_families`,
  `tie_tolerance_seconds`, `min_blocks_for_admission`, `split_fractions`, `null_draws`,
  `random_half_draws`, `null_seed`, `alpha`, `multiplicity_correction`,
  `state_dimension_grid`, `geometry_cluster_grid`, `model_seeds`, `default_workers`,
  `synthetic_grid`, `forbidden_inputs`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_slow_state_config.py
from pathlib import Path

import yaml

CONFIG = Path("config/topic5_slow_state_v4_0.yaml")


def _load():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_session_join_is_three_hundred_seconds_on_metadata_gaps():
    assert _load()["session_join_seconds"] == 300.0


def test_primary_event_grid_excludes_five_thousand():
    config = _load()
    assert config["event_window_grid"] == [50, 100, 200, 500, 1000, 2000]
    assert config["diagnostic_only_windows"] == [5000]


def test_clock_grid_covers_five_minutes_to_six_hours():
    assert _load()["clock_window_grid"] == [300, 900, 1800, 3600, 7200, 14400, 21600]


def test_a_scale_needs_a_minimum_number_of_independent_windows():
    # 200 random splits are one window's uncertainty, not 200 replicates
    assert _load()["min_windows_per_scale"] >= 5


def test_support_floors_exist_for_contacts_and_for_pairs():
    config = _load()
    assert config["min_participation_count"] >= 5
    assert config["min_pair_coparticipation_count"] >= 5


def test_at_least_two_primary_families_must_resolve():
    assert _load()["min_resolved_families"] == 2


def test_admission_gate_is_forty_blocks():
    assert _load()["min_blocks_for_admission"] == 40


def test_default_workers_is_conservative_until_peak_rss_is_measured():
    assert 4 <= _load()["default_workers"] <= 8


def test_no_absolute_reliability_threshold_is_configured():
    assert "reliability_threshold" not in CONFIG.read_text(encoding="utf-8")


def test_forbidden_inputs_are_all_declared_true():
    forbidden = _load()["forbidden_inputs"]
    for key in ("old_heldout20", "ictal_or_snn", "soz_or_geometry", "ab_or_axis_labels"):
        assert forbidden[key] is True


def test_synthetic_calibration_grid_spans_states_dwell_noise_and_block_count():
    grid = _load()["synthetic_grid"]
    assert grid["n_states"] == [2, 3, 4, 5]
    assert grid["dwell_blocks"] == [2, 5, 10, 20, 50]
    assert len(grid["noise_levels"]) == 3
    assert grid["n_blocks"] == [40, 80, 160, 320]
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest -q tests/test_topic5_slow_state_config.py`
Expected: FAIL — config file missing.

- [ ] **Step 3: Write the config**

```yaml
contract: topic5_slow_state_two_timescale_v4_0
dataset_root: results/topic5_interictal_rank_distribution/dataset_v0_4
source_mapping_root: results/topic5_event_indexed_evolving_rank_field/development/input_audit/per_subject
epilepsiae_block_inventory: results/epilepsiae_block_inventory.csv
yuquan_block_inventory: results/dataset_inventory/yuquan_block_inventory.csv
output_root: results/topic5_slow_state_two_timescale/v4_0

session_join_seconds: 300.0
event_window_grid: [50, 100, 200, 500, 1000, 2000]
diagnostic_only_windows: [5000]
clock_window_grid: [300, 900, 1800, 3600, 7200, 14400, 21600]
min_windows_per_scale: 5

min_events_for_descriptor: 20
min_participation_count: 5
min_pair_coparticipation_count: 5
min_resolved_families: 2
tie_tolerance_seconds: 0.0

min_blocks_for_admission: 40
split_fractions: [0.60, 0.20, 0.20]

null_draws: 200
random_half_draws: 200
null_seed: 20260804
alpha: 0.05
multiplicity_correction: holm

state_dimension_grid: [1, 2, 3, 4]
geometry_cluster_grid: [2, 3, 4, 5]
model_seeds: [11, 12, 13]
default_workers: 8

synthetic_grid:
  n_states: [2, 3, 4, 5]
  dwell_blocks: [2, 5, 10, 20, 50]
  noise_levels: [0.2, 0.5, 1.0]
  n_blocks: [40, 80, 160, 320]
  seeds: [0, 1, 2, 3, 4]

forbidden_inputs:
  old_heldout20: true
  ictal_or_snn: true
  soz_or_geometry: true
  ab_or_axis_labels: true
```

- [ ] **Step 4: Run to verify it passes**

Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add config/topic5_slow_state_v4_0.yaml tests/test_topic5_slow_state_config.py
git commit -m "feat(topic5-v4): freeze grids, support floors and worker default"
```

---

## Task 2: Metadata source intervals — extraction, not reimplementation

**Files:**
- Create: `src/topic5_source_intervals.py`
- Modify: `scripts/audit_topic5_event_innovation_v3_0_phase0.py`
- Test: `tests/test_topic5_source_intervals.py`

**Why extraction:** `build_source_segments` already exists in the v3.0 audit script and its
docstring states the rule this task must honour — *"Resolve source intervals from
inventories/EDF headers, never event density."* Reimplementing it from event times is the
defect this task fixes. Move it verbatim into `src/`, leave a re-export in the audit script so
frozen v3.0 behaviour is bit-identical, and confirm that with a test.

**Interfaces:**
- Consumes: `config["epilepsiae_block_inventory"]`, `config["yuquan_block_inventory"]`.
- Produces:
  `build_source_segments(subject, source_ids, record_names, config) -> tuple[tuple[SourceSegment, ...], list[dict]]`,
  plus a re-export of `SourceSegment` for callers' convenience.

**Correction applied 2026-08-04 after a blocked implementation attempt:** an earlier draft of
this task claimed `SourceSegment` was a namedtuple defined in the audit script and should be
moved. It is neither. `SourceSegment` is a **frozen dataclass** already defined in
`src/topic5_event_innovation_data.py`, already listed in that module's `__all__`, already
imported by the audit script, and also used by `assign_continuity_units` in the same frozen
v3.0 module. **Do not move it.** Only `build_source_segments` and its private helpers move.
Because it is a dataclass and not a namedtuple it has no `_fields`; field introspection uses
`dataclasses.fields`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_source_intervals.py
import inspect

import numpy as np
import pytest


def test_the_shared_module_exposes_the_metadata_resolver():
    import dataclasses

    from src.topic5_source_intervals import SourceSegment, build_source_segments

    assert "never event density" in build_source_segments.__doc__
    # SourceSegment is a frozen dataclass owned by topic5_event_innovation_data and only
    # re-exported here, so introspect it as a dataclass rather than as a namedtuple
    names = [field.name for field in dataclasses.fields(SourceSegment)]
    assert names[:3] == ["source_id", "start_time", "stop_time"]


def test_source_segment_is_re_exported_not_redefined():
    from src import topic5_event_innovation_data as owner
    from src import topic5_source_intervals as shared

    assert shared.SourceSegment is owner.SourceSegment


def test_the_v3_0_audit_script_now_imports_rather_than_duplicates():
    from scripts import audit_topic5_event_innovation_v3_0_phase0 as audit
    from src import topic5_source_intervals as shared

    assert audit.build_source_segments is shared.build_source_segments


def test_interval_bounds_do_not_come_from_event_times():
    from src.topic5_source_intervals import build_source_segments

    source = inspect.getsource(build_source_segments)
    for forbidden in ("event_abs_time", "event_time"):
        assert forbidden not in source, f"{forbidden} would reintroduce event-density bounds"


def test_a_record_without_an_inventory_row_fails_loudly():
    from src.topic5_source_intervals import build_source_segments

    with pytest.raises(RuntimeError):
        build_source_segments(
            "epilepsiae_9999",
            np.array(["ghost"]),
            np.array(["ghost"]),
            {"epilepsiae_block_inventory": "results/epilepsiae_block_inventory.csv"},
        )
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `ModuleNotFoundError: No module named 'src.topic5_source_intervals'`.

- [ ] **Step 3: Move the code**

Cut `build_source_segments`, `_inventory_rows`, `_yuquan_inventory_fallback` and
`_montage_hash` from `scripts/audit_topic5_event_innovation_v3_0_phase0.py` into
`src/topic5_source_intervals.py` unchanged. **`SourceSegment` stays where it is** — import it
into the new module from `src.topic5_event_innovation_data` and re-export it. In the audit
script replace the moved functions with:

```python
from src.topic5_source_intervals import (  # noqa: F401
    SourceSegment,
    build_source_segments,
)
```

- [ ] **Step 4: Verify the frozen v3.0 audit still behaves identically**

Run: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest -q tests/test_audit_topic5_event_innovation_v3_0_phase0.py`
Expected: PASS with the same test count as before the move. If any behaviour changes, revert
and stop — the v3.0 audit is frozen.

- [ ] **Step 5: Run the new test**

Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add src/topic5_source_intervals.py tests/test_topic5_source_intervals.py \
        scripts/audit_topic5_event_innovation_v3_0_phase0.py
git commit -m "refactor(topic5): extract metadata source-interval resolver for reuse"
```

---

## Task 3: Sessions, blocks and transition strata

**Files:** Create `src/topic5_slow_state_sessions.py`; Test `tests/test_topic5_slow_state_sessions.py`

**Interfaces:**
- Consumes: `SourceSegment` rows from Task 2 (as dicts or namedtuples),
  `config["session_join_seconds"]`.
- Produces:
  - `build_sessions(segments, *, join_seconds) -> list[dict]` — keys `session_index:int`,
    `segment_ids:tuple[str,...]`, `t_start:float`, `t_end:float` (**metadata** bounds),
    `continuity_group:str`, `montage_hash:str`.
  - `assign_events(sessions, event_times, event_record_names) -> list[dict]` — adds
    `event_indices:np.ndarray`, `n_events:int`, `first_event_time:float | None`,
    `last_event_time:float | None`.
  - `session_gaps(sessions) -> list[dict]` — keys `left_session:int`, `right_session:int`,
    `metadata_gap_seconds:float`, `event_silence_seconds:float`,
    `observed_events_during_gap:bool` (always `False`).
  - `build_blocks(sessions, *, block_events, event_times) -> list[dict]` — keys
    `block_index:int`, `session_index:int`, `event_indices:np.ndarray`, `t_start:float`,
    `t_end:float`, `delta_t_from_previous:float | None`,
    `transition_stratum:str | None` in `{"within_session","cross_gap",None}`.
  - `dropped_remainders(sessions, *, block_events) -> list[dict]` — keys `session_index:int`,
    `n_dropped:int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_slow_state_sessions.py
import numpy as np
import pytest

from src.topic5_slow_state_sessions import (
    assign_events,
    build_blocks,
    build_sessions,
    dropped_remainders,
    session_gaps,
)


def _seg(sid, start, stop, group="g", montage="m"):
    return {
        "source_id": sid, "start_time": start, "stop_time": stop,
        "continuity_group": group, "montage_hash": montage,
    }


def test_sessions_use_metadata_bounds_not_event_bounds():
    # the recording runs 0-1000 s but the only events are at 400-500 s;
    # the session must still be 1000 s long
    sessions = build_sessions([_seg("a", 0.0, 1000.0)], join_seconds=300.0)
    assert sessions[0]["t_start"] == 0.0 and sessions[0]["t_end"] == 1000.0
    with_events = assign_events(sessions, np.array([400.0, 500.0]), np.array(["a", "a"]))
    assert with_events[0]["t_start"] == 0.0 and with_events[0]["t_end"] == 1000.0
    assert with_events[0]["first_event_time"] == 400.0


def test_a_quiet_but_recorded_stretch_is_not_a_gap():
    sessions = build_sessions([_seg("a", 0.0, 5000.0)], join_seconds=300.0)
    assert session_gaps(sessions) == []


def test_metadata_gap_and_event_silence_are_reported_separately():
    segments = [_seg("a", 0.0, 1000.0), _seg("b", 2000.0, 3000.0)]
    sessions = assign_events(
        build_sessions(segments, join_seconds=300.0),
        np.array([100.0, 900.0, 2100.0, 2900.0]),
        np.array(["a", "a", "b", "b"]),
    )
    gap = session_gaps(sessions)[0]
    assert gap["metadata_gap_seconds"] == pytest.approx(1000.0)
    assert gap["event_silence_seconds"] == pytest.approx(1200.0)
    assert gap["event_silence_seconds"] > gap["metadata_gap_seconds"]
    assert gap["observed_events_during_gap"] is False


def test_segments_within_the_join_threshold_merge():
    segments = [_seg("a", 0.0, 1000.0), _seg("b", 1200.0, 2000.0)]
    assert len(build_sessions(segments, join_seconds=300.0)) == 1


def test_segments_with_a_different_montage_never_merge():
    segments = [_seg("a", 0.0, 1000.0, montage="m1"), _seg("b", 1100.0, 2000.0, montage="m2")]
    assert len(build_sessions(segments, join_seconds=300.0)) == 2


def test_segments_in_a_different_continuity_group_never_merge():
    segments = [_seg("a", 0.0, 1000.0, group="g1"), _seg("b", 1100.0, 2000.0, group="g2")]
    assert len(build_sessions(segments, join_seconds=300.0)) == 2


def test_blocks_never_span_a_session_boundary():
    segments = [_seg("a", 0.0, 100.0), _seg("b", 10000.0, 10100.0)]
    times = np.concatenate([np.arange(6.0), 10000.0 + np.arange(6.0)])
    names = np.array(["a"] * 6 + ["b"] * 6)
    sessions = assign_events(build_sessions(segments, join_seconds=300.0), times, names)
    blocks = build_blocks(sessions, block_events=4, event_times=times)
    assert len(blocks) == 2
    assert {b["session_index"] for b in blocks} == {0, 1}


def test_the_first_block_after_a_gap_is_labelled_cross_gap():
    segments = [_seg("a", 0.0, 100.0), _seg("b", 10000.0, 10100.0)]
    times = np.concatenate([np.arange(8.0), 10000.0 + np.arange(8.0)])
    names = np.array(["a"] * 8 + ["b"] * 8)
    sessions = assign_events(build_sessions(segments, join_seconds=300.0), times, names)
    blocks = build_blocks(sessions, block_events=4, event_times=times)
    assert [b["transition_stratum"] for b in blocks] == [
        None, "within_session", "cross_gap", "within_session",
    ]


def test_delta_t_bridges_the_gap_rather_than_resetting_to_zero():
    segments = [_seg("a", 0.0, 100.0), _seg("b", 10000.0, 10100.0)]
    times = np.concatenate([np.arange(4.0), 10000.0 + np.arange(4.0)])
    names = np.array(["a"] * 4 + ["b"] * 4)
    sessions = assign_events(build_sessions(segments, join_seconds=300.0), times, names)
    blocks = build_blocks(sessions, block_events=4, event_times=times)
    assert blocks[0]["delta_t_from_previous"] is None
    assert blocks[1]["delta_t_from_previous"] == pytest.approx(10000.0 - 3.0)


def test_session_remainders_are_dropped_and_counted_not_padded():
    segments = [_seg("a", 0.0, 100.0)]
    times = np.arange(10.0)
    names = np.array(["a"] * 10)
    sessions = assign_events(build_sessions(segments, join_seconds=300.0), times, names)
    assert len(build_blocks(sessions, block_events=4, event_times=times)) == 2
    assert dropped_remainders(sessions, block_events=4) == [
        {"session_index": 0, "n_dropped": 2}
    ]


def test_block_size_below_two_is_rejected():
    segments = [_seg("a", 0.0, 100.0)]
    times = np.arange(10.0)
    sessions = assign_events(
        build_sessions(segments, join_seconds=300.0), times, np.array(["a"] * 10)
    )
    with pytest.raises(ValueError):
        build_blocks(sessions, block_events=1, event_times=times)
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — module missing.

- [ ] **Step 3: Write the implementation**

```python
# src/topic5_slow_state_sessions.py
"""Sessions and blocks for the slow-state contract.

Session bounds come from metadata source intervals, never from the first and last
detected event: a normally recorded stretch in which no HFO was detected is recorded
time with no events, not missing data.

Two gap quantities are kept apart.  `metadata_gap_seconds` is unrecorded wall time.
`event_silence_seconds` is the span between the last event of one session and the first
of the next, and is always at least as large.  Neither ever means "no events occurred".
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _field(segment: Any, name: str) -> Any:
    return segment[name] if isinstance(segment, Mapping) else getattr(segment, name)


def build_sessions(
    segments: Sequence[Any], *, join_seconds: float
) -> list[dict[str, Any]]:
    ordered = sorted(
        segments, key=lambda s: (float(_field(s, "start_time")), str(_field(s, "source_id")))
    )
    sessions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for segment in ordered:
        start = float(_field(segment, "start_time"))
        stop = float(_field(segment, "stop_time"))
        group = str(_field(segment, "continuity_group"))
        montage = str(_field(segment, "montage_hash"))
        joinable = (
            current is not None
            and start - current["t_end"] <= float(join_seconds)
            and current["continuity_group"] == group
            and current["montage_hash"] == montage
        )
        if joinable:
            current["t_end"] = max(current["t_end"], stop)
            current["segment_ids"].append(str(_field(segment, "source_id")))
        else:
            if current is not None:
                sessions.append(current)
            current = {
                "t_start": start,
                "t_end": stop,
                "segment_ids": [str(_field(segment, "source_id"))],
                "continuity_group": group,
                "montage_hash": montage,
            }
    if current is not None:
        sessions.append(current)
    for index, session in enumerate(sessions):
        session["session_index"] = index
        session["segment_ids"] = tuple(session["segment_ids"])
    return sessions


def assign_events(
    sessions: Sequence[Mapping[str, Any]],
    event_times: Sequence[float],
    event_record_names: Sequence[str],
) -> list[dict[str, Any]]:
    times = np.asarray(event_times, dtype=float)
    names = np.asarray(event_record_names).astype(str)
    output = []
    for session in sessions:
        member = np.flatnonzero(np.isin(names, np.asarray(session["segment_ids"])))
        member = member[np.argsort(times[member], kind="stable")]
        row = dict(session)
        row["event_indices"] = member
        row["n_events"] = int(member.size)
        row["first_event_time"] = float(times[member].min()) if member.size else None
        row["last_event_time"] = float(times[member].max()) if member.size else None
        output.append(row)
    return output


def session_gaps(sessions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for left, right in zip(sessions, sessions[1:]):
        last = left.get("last_event_time")
        first = right.get("first_event_time")
        rows.append(
            {
                "left_session": int(left["session_index"]),
                "right_session": int(right["session_index"]),
                "metadata_gap_seconds": float(right["t_start"] - left["t_end"]),
                "event_silence_seconds": (
                    float(first - last) if last is not None and first is not None else None
                ),
                "observed_events_during_gap": False,
            }
        )
    return rows


def build_blocks(
    sessions: Sequence[Mapping[str, Any]],
    *,
    block_events: int,
    event_times: Sequence[float],
) -> list[dict[str, Any]]:
    size = int(block_events)
    if size < 2:
        raise ValueError("block_events must be at least 2")
    times = np.asarray(event_times, dtype=float)
    blocks: list[dict[str, Any]] = []
    previous_session: int | None = None
    previous_end: float | None = None
    for session in sessions:
        indices = np.asarray(session["event_indices"])
        for start in range(0, indices.size - size + 1, size):
            member = indices[start : start + size]
            t_start = float(times[member].min())
            t_end = float(times[member].max())
            if previous_session is None:
                stratum, delta = None, None
            else:
                stratum = (
                    "within_session"
                    if int(session["session_index"]) == previous_session
                    else "cross_gap"
                )
                delta = float(t_start - previous_end)
            blocks.append(
                {
                    "block_index": len(blocks),
                    "session_index": int(session["session_index"]),
                    "event_indices": member,
                    "t_start": t_start,
                    "t_end": t_end,
                    "delta_t_from_previous": delta,
                    "transition_stratum": stratum,
                }
            )
            previous_session = int(session["session_index"])
            previous_end = t_end
    return blocks


def dropped_remainders(
    sessions: Sequence[Mapping[str, Any]], *, block_events: int
) -> list[dict[str, Any]]:
    size = int(block_events)
    rows = []
    for session in sessions:
        total = int(np.asarray(session["event_indices"]).size)
        dropped = total if total < size else total % size
        if dropped:
            rows.append(
                {"session_index": int(session["session_index"]), "n_dropped": dropped}
            )
    return rows
```

- [ ] **Step 4: Run to verify it passes**

Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_slow_state_sessions.py tests/test_topic5_slow_state_sessions.py
git commit -m "feat(topic5-v4): metadata-bounded sessions, blocks and transition strata"
```

---

## Task 4: Primary repertoire descriptors with ties and support floors

**Files:** Create `src/topic5_slow_state_repertoire.py`; Test `tests/test_topic5_slow_state_repertoire.py`

**Interfaces:**
- Consumes: `event_local_rank`, `event_participation`, **`event_group_ids`** slices.
- Produces:
  - `local_repertoire(rank, participation, group_ids, *, min_participation_count, min_pair_count) -> dict`
    with `participation_rate:np.ndarray`, `masked_mean_rank:np.ndarray`,
    `precedence:np.ndarray`, `pair_index:list[tuple[int,int]]`,
    `contact_support:np.ndarray`, `pair_support:np.ndarray`, `n_supported_contacts:int`,
    `n_supported_pairs:int`, `n_events:int`, `status:str`.
  - `family_agreement(left, right) -> dict[str, float | None]` with keys `participation`,
    `mean_rank`, `precedence` — **no `combined` key**.
  - `resolved_families(agreement) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_slow_state_repertoire.py
import numpy as np
import pytest

from src.topic5_slow_state_repertoire import (
    family_agreement,
    local_repertoire,
    resolved_families,
)


def _block(order, groups, n_events, participation=None):
    rank = np.tile(np.asarray(order, float), (n_events, 1))
    gids = np.tile(np.asarray(groups, np.int16), (n_events, 1))
    part = (
        np.ones_like(rank, dtype=np.uint8)
        if participation is None
        else np.tile(np.asarray(participation, np.uint8), (n_events, 1))
    )
    return rank, part, gids


def test_ties_come_from_group_ids_not_from_equal_rank_values():
    # contacts 0 and 1 carry different normalised ranks but share a recruitment group,
    # so they are tied and precedence must be exactly one half
    rank, part, gids = _block([0.1, 0.2, 0.9], [0, 0, 1], 20)
    out = local_repertoire(rank, part, gids, min_participation_count=5, min_pair_count=5)
    assert out["precedence"][out["pair_index"].index((0, 1))] == pytest.approx(0.5)


def test_equal_rank_values_in_different_groups_are_not_treated_as_tied():
    rank, part, gids = _block([0.5, 0.5, 0.9], [0, 1, 2], 20)
    out = local_repertoire(rank, part, gids, min_participation_count=5, min_pair_count=5)
    assert out["precedence"][out["pair_index"].index((0, 1))] != pytest.approx(0.5)


def test_precedence_is_one_when_the_earlier_group_always_comes_first():
    rank, part, gids = _block([0.1, 0.5, 0.9], [0, 1, 2], 20)
    out = local_repertoire(rank, part, gids, min_participation_count=5, min_pair_count=5)
    assert out["precedence"][out["pair_index"].index((0, 2))] == pytest.approx(1.0)


def test_a_contact_below_the_participation_floor_is_excluded_not_averaged():
    rank = np.tile(np.array([0.1, 0.5, 0.9]), (20, 1))
    gids = np.tile(np.array([0, 1, 2], np.int16), (20, 1))
    part = np.ones((20, 3), np.uint8)
    part[:-2, 2] = 0  # contact 2 participates in only 2 of 20 events
    out = local_repertoire(rank, part, gids, min_participation_count=5, min_pair_count=5)
    assert np.isnan(out["masked_mean_rank"][2])
    assert out["n_supported_contacts"] == 2


def test_a_pair_below_the_co_participation_floor_is_excluded():
    rank = np.tile(np.array([0.1, 0.5, 0.9]), (20, 1))
    gids = np.tile(np.array([0, 1, 2], np.int16), (20, 1))
    part = np.ones((20, 3), np.uint8)
    part[:10, 1] = 0
    part[10:, 2] = 0  # contacts 1 and 2 never co-participate
    out = local_repertoire(rank, part, gids, min_participation_count=5, min_pair_count=5)
    assert np.isnan(out["precedence"][out["pair_index"].index((1, 2))])


def test_a_contact_that_never_participates_gets_nan_not_a_phantom_number():
    rank = np.tile(np.array([0.1, 5.0, 0.9]), (20, 1))
    gids = np.tile(np.array([0, 1, 2], np.int16), (20, 1))
    part = np.ones((20, 3), np.uint8)
    part[:, 1] = 0
    out = local_repertoire(rank, part, gids, min_participation_count=5, min_pair_count=5)
    assert np.isnan(out["masked_mean_rank"][1])
    assert out["participation_rate"][1] == 0.0


def test_agreement_reports_each_family_separately_and_has_no_combined_key():
    rank, part, gids = _block([0.1, 0.4, 0.7, 0.9], [0, 1, 2, 3], 20)
    left = local_repertoire(rank, part, gids, min_participation_count=5, min_pair_count=5)
    agreement = family_agreement(left, left)
    assert set(agreement) == {"participation", "mean_rank", "precedence"}
    assert agreement["mean_rank"] == pytest.approx(1.0)


def test_reversed_orderings_disagree_maximally_on_rank():
    rank_a, part, gids = _block([0.1, 0.4, 0.7, 0.9], [0, 1, 2, 3], 20)
    rank_b, _, gids_b = _block([0.9, 0.7, 0.4, 0.1], [3, 2, 1, 0], 20)
    left = local_repertoire(rank_a, part, gids, min_participation_count=5, min_pair_count=5)
    right = local_repertoire(rank_b, part, gids_b, min_participation_count=5, min_pair_count=5)
    assert family_agreement(left, right)["mean_rank"] == pytest.approx(-1.0)


def test_a_single_resolved_family_cannot_stand_in_for_the_repertoire():
    assert resolved_families({"participation": 0.9, "mean_rank": None, "precedence": None}) == 1
    assert resolved_families({"participation": 0.9, "mean_rank": 0.8, "precedence": None}) == 2
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — module missing.

- [ ] **Step 3: Write the implementation**

```python
# src/topic5_slow_state_repertoire.py
"""The three primary repertoire descriptors of one event window.

Ties are read from `event_group_ids`, never from equal `event_local_rank` values: the
stored rank is a normalised rank among participating contacts, so equal values do not
recover the recruitment-group structure.  Contacts sharing a group are simultaneous and
contribute 0.5 to precedence.

Each family is returned separately and there is no combined score, so no caller can let
one surviving correlation coefficient stand in for the whole repertoire.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.stats import spearmanr

FAMILIES = ("participation", "mean_rank", "precedence")


def local_repertoire(
    rank_field: np.ndarray,
    participation: np.ndarray,
    group_ids: np.ndarray,
    *,
    min_participation_count: int,
    min_pair_count: int,
) -> dict[str, Any]:
    rank = np.asarray(rank_field, dtype=float)
    part = np.asarray(participation).astype(bool)
    groups = np.asarray(group_ids)
    if not (rank.shape == part.shape == groups.shape):
        raise ValueError("rank, participation and group_ids must share a shape")
    n_events, n_contacts = rank.shape

    counts = part.sum(axis=0).astype(float)
    rate = counts / float(n_events) if n_events else np.zeros(n_contacts)
    with np.errstate(invalid="ignore"):
        summed = np.where(part, rank, 0.0).sum(axis=0)
        mean_rank = np.where(counts > 0, summed / np.maximum(counts, 1.0), np.nan)
    mean_rank = np.where(counts >= int(min_participation_count), mean_rank, np.nan)

    pair_index: list[tuple[int, int]] = []
    precedence: list[float] = []
    pair_support: list[int] = []
    for i in range(n_contacts):
        for j in range(i + 1, n_contacts):
            both = part[:, i] & part[:, j]
            support = int(both.sum())
            pair_index.append((i, j))
            pair_support.append(support)
            if support < int(min_pair_count):
                precedence.append(np.nan)
                continue
            left, right = groups[both, i], groups[both, j]
            earlier = float(np.sum(left < right))
            tied = float(np.sum(left == right))
            precedence.append((earlier + 0.5 * tied) / support)

    supported_contacts = int(np.sum(counts >= int(min_participation_count)))
    supported_pairs = int(np.sum(np.asarray(pair_support) >= int(min_pair_count)))
    status = "RESOLVED"
    if supported_contacts < 3:
        status = "UNRESOLVED_TOO_FEW_CONTACTS"
    elif supported_pairs < 3:
        status = "UNRESOLVED_TOO_FEW_PAIRS"
    return {
        "participation_rate": rate,
        "masked_mean_rank": mean_rank,
        "precedence": np.asarray(precedence, dtype=float),
        "pair_index": pair_index,
        "contact_support": counts,
        "pair_support": np.asarray(pair_support, dtype=float),
        "n_supported_contacts": supported_contacts,
        "n_supported_pairs": supported_pairs,
        "n_events": int(n_events),
        "status": status,
    }


def _agree(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if int(keep.sum()) < 3:
        return None
    x, y = a[keep], b[keep]
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    value = spearmanr(x, y).statistic
    return None if not np.isfinite(value) else float(value)


def family_agreement(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, float | None]:
    return {
        "participation": _agree(left["participation_rate"], right["participation_rate"]),
        "mean_rank": _agree(left["masked_mean_rank"], right["masked_mean_rank"]),
        "precedence": _agree(left["precedence"], right["precedence"]),
    }


def resolved_families(agreement: Mapping[str, float | None]) -> int:
    return int(sum(agreement.get(name) is not None for name in FAMILIES))
```

- [ ] **Step 4: Run to verify it passes**

Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_slow_state_repertoire.py tests/test_topic5_slow_state_repertoire.py
git commit -m "feat(topic5-v4): tie-aware repertoire descriptors with support floors"
```

---

## Task 5: Independent window enumeration

**Files:** Create `src/topic5_slow_state_windows.py`; Test `tests/test_topic5_slow_state_windows.py`

**Interfaces:**
- Consumes: sessions from Task 3.
- Produces:
  - `tile_event_windows(sessions, *, window_events) -> list[dict]` — non-overlapping, tiled
    from each session start; keys `window_index:int`, `session_index:int`,
    `event_indices:np.ndarray`, `offset_fraction:float` (always `0.0`).
  - `tile_clock_windows(sessions, event_times, *, window_seconds, min_events) -> list[dict]` —
    non-overlapping wall-clock tiles inside a session, keeping only tiles with at least
    `min_events` events.
  - `sliding_event_windows(sessions, *, window_events, offsets) -> list[dict]` — sensitivity
    only; every row has `offset_fraction != 0.0`.
  - `scale_is_evaluable(windows, *, minimum) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_slow_state_windows.py
import numpy as np

from src.topic5_slow_state_windows import (
    scale_is_evaluable,
    sliding_event_windows,
    tile_clock_windows,
    tile_event_windows,
)


def _session(index, indices, t_start=0.0, t_end=1000.0):
    return {
        "session_index": index,
        "event_indices": np.asarray(list(indices)),
        "t_start": t_start,
        "t_end": t_end,
    }


def test_primary_windows_do_not_overlap():
    windows = tile_event_windows([_session(0, range(100))], window_events=20)
    seen = np.concatenate([w["event_indices"] for w in windows])
    assert len(windows) == 5
    assert len(set(seen.tolist())) == seen.size


def test_primary_windows_never_span_two_sessions():
    windows = tile_event_windows(
        [_session(0, range(30)), _session(1, range(30, 60))], window_events=20
    )
    assert len(windows) == 2
    assert {w["session_index"] for w in windows} == {0, 1}


def test_primary_windows_are_marked_as_zero_offset():
    windows = tile_event_windows([_session(0, range(100))], window_events=20)
    assert all(w["offset_fraction"] == 0.0 for w in windows)


def test_sliding_windows_are_labelled_as_sensitivity_and_do_overlap():
    windows = sliding_event_windows(
        [_session(0, range(100))], window_events=20, offsets=[0.5]
    )
    assert windows
    assert all(w["offset_fraction"] == 0.5 for w in windows)


def test_clock_windows_tile_wall_time_not_event_count():
    times = np.concatenate([np.arange(0.0, 10.0), np.arange(100.0, 110.0)])
    session = _session(0, range(20), t_start=0.0, t_end=200.0)
    windows = tile_clock_windows([session], times, window_seconds=100.0, min_events=5)
    assert len(windows) == 2
    assert len(windows[0]["event_indices"]) == 10


def test_a_clock_window_with_too_few_events_is_dropped_not_padded():
    times = np.array([0.0, 1.0, 150.0])
    session = _session(0, range(3), t_start=0.0, t_end=200.0)
    assert tile_clock_windows([session], times, window_seconds=100.0, min_events=5) == []


def test_a_scale_with_too_few_independent_windows_is_not_evaluable():
    assert scale_is_evaluable([{}] * 4, minimum=5) is False
    assert scale_is_evaluable([{}] * 5, minimum=5) is True


def test_two_hundred_random_splits_are_not_two_hundred_windows():
    # guards the rev1 defect: split count must never be mistaken for window count
    windows = tile_event_windows([_session(0, range(100))], window_events=50)
    assert len(windows) == 2
    assert scale_is_evaluable(windows, minimum=5) is False
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — module missing.

- [ ] **Step 3: Write the implementation**

```python
# src/topic5_slow_state_windows.py
"""Independent windows, and what does not count as one.

Two hundred random splits estimate one window's uncertainty.  They are not two hundred
replicates, and no cohort or patient-level count may be built from them.  Primary windows
are non-overlapping tiles inside a session; sliding offsets exist only as sensitivity and
carry a non-zero `offset_fraction` so they can never be pooled with the primary by
accident.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def tile_event_windows(
    sessions: Sequence[Mapping[str, Any]], *, window_events: int
) -> list[dict[str, Any]]:
    size = int(window_events)
    if size < 2:
        raise ValueError("window_events must be at least 2")
    rows: list[dict[str, Any]] = []
    for session in sessions:
        indices = np.asarray(session["event_indices"])
        for start in range(0, indices.size - size + 1, size):
            rows.append(
                {
                    "window_index": len(rows),
                    "session_index": int(session["session_index"]),
                    "event_indices": indices[start : start + size],
                    "offset_fraction": 0.0,
                }
            )
    return rows


def sliding_event_windows(
    sessions: Sequence[Mapping[str, Any]],
    *,
    window_events: int,
    offsets: Sequence[float],
) -> list[dict[str, Any]]:
    size = int(window_events)
    rows: list[dict[str, Any]] = []
    for offset in offsets:
        if float(offset) == 0.0:
            raise ValueError("offset 0.0 is the primary tiling, not a sensitivity offset")
        shift = int(round(float(offset) * size))
        for session in sessions:
            indices = np.asarray(session["event_indices"])
            for start in range(shift, indices.size - size + 1, size):
                rows.append(
                    {
                        "window_index": len(rows),
                        "session_index": int(session["session_index"]),
                        "event_indices": indices[start : start + size],
                        "offset_fraction": float(offset),
                    }
                )
    return rows


def tile_clock_windows(
    sessions: Sequence[Mapping[str, Any]],
    event_times: Sequence[float],
    *,
    window_seconds: float,
    min_events: int,
) -> list[dict[str, Any]]:
    times = np.asarray(event_times, dtype=float)
    span = float(window_seconds)
    rows: list[dict[str, Any]] = []
    for session in sessions:
        indices = np.asarray(session["event_indices"])
        start = float(session["t_start"])
        end = float(session["t_end"])
        edge = start
        while edge + span <= end:
            member = indices[
                (times[indices] >= edge) & (times[indices] < edge + span)
            ]
            if member.size >= int(min_events):
                rows.append(
                    {
                        "window_index": len(rows),
                        "session_index": int(session["session_index"]),
                        "event_indices": member,
                        "offset_fraction": 0.0,
                        "t_start": edge,
                        "t_end": edge + span,
                    }
                )
            edge += span
    return rows


def scale_is_evaluable(windows: Sequence[Any], *, minimum: int) -> bool:
    return len(windows) >= int(minimum)
```

- [ ] **Step 4: Run to verify it passes**

Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_slow_state_windows.py tests/test_topic5_slow_state_windows.py
git commit -m "feat(topic5-v4): independent window enumeration and evaluability gate"
```

---

## Task 6: Two scales, pattern matching and the dwell interval

**Files:** Create `src/topic5_slow_state_scale.py`; Test `tests/test_topic5_slow_state_scale.py`

**Interfaces:**
- Consumes: Task 4 descriptors, Task 5 windows.
- Produces:
  - `window_agreements(rank, participation, group_ids, *, random_half_draws, null_draws, seed, floors) -> dict`
    with `random_half:dict[str,list[float]]`, `chronological:dict[str,float|None]`,
    `contact_null:dict[str,list[float]]` — one entry per family in
    `("participation","mean_rank","precedence")`.
  - `window_state(agreements, *, alpha, min_resolved_families) -> str` in
    `{"BELOW_CHANCE","RELIABLE","CHRONOLOGY_BREAK","UNRESOLVED_FAMILIES"}`.
  - `scale_states(windows_states, *, min_windows) -> str` — majority state over independent
    windows, or `"UNRESOLVED_TOO_FEW_WINDOWS"`.
  - `select_scales(states) -> dict` with `n_obs:int|None`, `n_break:int|None`,
    `n_last_reliable:int|None`, `dwell_interval:tuple[int,int|None]|None`, `status:str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_slow_state_scale.py
from src.topic5_slow_state_scale import select_scales

B, R, C, F = "BELOW_CHANCE", "RELIABLE", "CHRONOLOGY_BREAK", "UNRESOLVED_TOO_FEW_WINDOWS"


def test_n_obs_is_the_smallest_reliable_scale_not_the_largest():
    out = select_scales({50: B, 100: R, 200: R, 500: C, 1000: C})
    assert out["n_obs"] == 100
    assert out["n_break"] == 500
    assert out["status"] == "SCALE_RESOLVED"


def test_a_leading_run_of_below_chance_windows_does_not_force_unresolved():
    # rev1's contiguous-prefix rule wrongly returned unresolved here
    out = select_scales({50: B, 100: B, 200: R, 500: C})
    assert out["status"] == "SCALE_RESOLVED"
    assert out["n_obs"] == 200


def test_the_dwell_is_an_interval_between_the_last_reliable_and_the_break():
    out = select_scales({50: B, 100: R, 200: R, 500: C})
    assert out["n_last_reliable"] == 200
    assert out["dwell_interval"] == (200, 500)


def test_a_dwell_interval_is_open_ended_when_no_break_is_reached():
    out = select_scales({50: B, 100: R, 200: R, 500: R})
    assert out["n_break"] is None
    assert out["dwell_interval"] == (500, None)


def test_reliability_returning_after_a_break_is_reported_not_coerced():
    out = select_scales({50: R, 100: C, 200: R})
    assert out["status"] == "UNRESOLVED_NONMONOTONE"
    assert out["n_obs"] is None


def test_no_reliable_scale_anywhere_is_unresolved_scale():
    out = select_scales({50: B, 100: B, 200: B})
    assert out["status"] == "UNRESOLVED_SCALE"
    assert out["n_obs"] is None


def test_scales_with_too_few_windows_are_skipped_not_counted_as_failures():
    out = select_scales({50: F, 100: R, 200: R, 500: C})
    assert out["status"] == "SCALE_RESOLVED"
    assert out["n_obs"] == 100


def test_a_scale_gap_from_too_few_windows_does_not_break_monotonicity():
    out = select_scales({50: B, 100: R, 200: F, 500: R, 1000: C})
    assert out["status"] == "SCALE_RESOLVED"
    assert out["n_obs"] == 100
    assert out["n_break"] == 1000
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — module missing.

- [ ] **Step 3: Write `select_scales` (complete code — this is where rev1 was wrong)**

```python
# src/topic5_slow_state_scale.py  (excerpt: the corrected selector)
"""Two scales, not one.

`N_obs` is the smallest window at which the repertoire rises above this patient's own
chance level; it becomes the model step, because a state that persists for 500 events cut
into 500-event blocks leaves one block per state and no dwell to observe.

`N_break` is the smallest larger window at which the chronological halves agree worse than
random halves — where a window starts averaging over a state change.

Scales with too few independent windows are skipped, not counted as failures, and a
leading run of below-chance scales is expected rather than disqualifying.
"""
from __future__ import annotations

from typing import Any, Mapping

BELOW, RELIABLE, BREAK = "BELOW_CHANCE", "RELIABLE", "CHRONOLOGY_BREAK"
TOO_FEW = "UNRESOLVED_TOO_FEW_WINDOWS"
UNRESOLVED_FAMILIES = "UNRESOLVED_FAMILIES"
#: States meaning "this scale was not evaluated". They are dropped before the pattern is
#: matched. Counting them as failures would throw a whole patient to NONMONOTONE because of
#: one unevaluable scale - most likely the smallest, where the support floors bite hardest.
NOT_EVALUATED = (TOO_FEW, UNRESOLVED_FAMILIES)


def select_scales(states: Mapping[int, str]) -> dict[str, Any]:
    evaluated = [
        (size, states[size]) for size in sorted(states) if states[size] not in NOT_EVALUATED
    ]
    labels = [state for _, state in evaluated]
    empty = {
        "n_obs": None,
        "n_break": None,
        "n_last_reliable": None,
        "dwell_interval": None,
    }
    if RELIABLE not in labels:
        return {**empty, "status": "UNRESOLVED_SCALE"}

    first = labels.index(RELIABLE)
    last = len(labels) - 1 - labels[::-1].index(RELIABLE)
    leading_ok = all(state == BELOW for state in labels[:first])
    middle_ok = all(state == RELIABLE for state in labels[first : last + 1])
    trailing_ok = all(state == BREAK for state in labels[last + 1 :])
    if not (leading_ok and middle_ok and trailing_ok):
        return {**empty, "status": "UNRESOLVED_NONMONOTONE"}

    n_obs = evaluated[first][0]
    n_last_reliable = evaluated[last][0]
    n_break = evaluated[last + 1][0] if last + 1 < len(evaluated) else None
    return {
        "n_obs": n_obs,
        "n_break": n_break,
        "n_last_reliable": n_last_reliable,
        "dwell_interval": (n_last_reliable, n_break),
        "status": "SCALE_RESOLVED",
    }
```

`window_agreements`, `window_state` and `scale_states` follow the Interfaces block:
`window_state` requires at least `min_resolved_families` families to resolve; a family is
above chance when its random-half median exceeds its own null `q95`; the window is
`CHRONOLOGY_BREAK` when the chronological value sits below the random-half distribution at
`alpha` in a majority of resolved families.

- [ ] **Step 4: Run to verify it passes**

Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_slow_state_scale.py tests/test_topic5_slow_state_scale.py
git commit -m "feat(topic5-v4): N_obs/N_break separation with pattern matching"
```

---

## Task 7: Phase 0 runner and reconciliation gate

**Files:** Create `scripts/run_topic5_slow_state_phase0_manifest.py`;
Test `tests/test_run_topic5_slow_state_phase0_manifest.py`

**Interfaces:**
- Consumes: Task 1 config, Tasks 2–3 modules.
- Produces: `phase0/PHASE0_STATE.json`, `session_manifest.csv`, `session_gaps.csv`,
  `block_feasibility.csv`, `per_subject/<subject>.json`, and — only on disagreement —
  `phase0/PHASE0_RECONCILIATION.md`.
- Exposes `reconcile(measured, rationale, output, tolerance) -> dict` with key `agrees:bool`,
  and `assert_reconciled(report) -> None` which raises `SystemExit` when `agrees` is `False`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_topic5_slow_state_phase0_manifest.py
import inspect

import pytest

from scripts import run_topic5_slow_state_phase0_manifest as phase0


def test_reconciliation_is_written_when_measurement_disagrees_with_the_spec(tmp_path):
    report = phase0.reconcile(
        measured={"median_events_per_session": 900},
        rationale={"median_events_per_session": 615},
        output=tmp_path,
        tolerance=0.10,
    )
    assert report["agrees"] is False
    text = (tmp_path / "PHASE0_RECONCILIATION.md").read_text(encoding="utf-8")
    assert "615" in text and "900" in text


def test_reconciliation_never_edits_the_spec_file():
    source = inspect.getsource(phase0.reconcile)
    assert "specs/" not in source
    assert "docs/superpowers" not in source


def test_execution_stops_when_reconciliation_fails():
    with pytest.raises(SystemExit):
        phase0.assert_reconciled({"agrees": False})


def test_execution_continues_when_measurement_matches_within_tolerance(tmp_path):
    report = phase0.reconcile(
        measured={"median_events_per_session": 640},
        rationale={"median_events_per_session": 615},
        output=tmp_path,
        tolerance=0.10,
    )
    assert report["agrees"] is True
    assert not (tmp_path / "PHASE0_RECONCILIATION.md").exists()
    phase0.assert_reconciled(report)
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — module missing.

- [ ] **Step 3: Write the runner**

`reconcile` compares each shared key within a relative tolerance, writes
`PHASE0_RECONCILIATION.md` listing both values and the reason when any key disagrees, and
never touches any path under `docs/`. `main()` loads the config, resolves source intervals
per subject via Task 2, builds sessions/gaps/blocks via Task 3, writes the four
deliverables atomically with `runner_sha256` and module hashes, then calls
`assert_reconciled`.

- [ ] **Step 4: Run to verify it passes**

Expected: PASS (4 tests).

- [ ] **Step 5: Execute Phase 0 on all 34 patients**

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 nohup \
  /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/run_topic5_slow_state_phase0_manifest.py \
  > results/run_logs/topic5_v4_phase0_$(date +%Y%m%d).log 2>&1 &
```

- [ ] **Step 6: STOP if `PHASE0_RECONCILIATION.md` was written**

Report both numbers and wait for a contract decision. Do not proceed to Phase 1 and do not
edit the spec. Spec §5's numbers came from event density and are *expected* to move once
metadata intervals are used, so this branch is likely.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_topic5_slow_state_phase0_manifest.py \
        tests/test_run_topic5_slow_state_phase0_manifest.py
git commit -m "feat(topic5-v4): phase 0 metadata manifest with reconciliation gate"
```

---

## Task 8: Phase 1 runner

**Files:** Create `scripts/run_topic5_slow_state_phase1_scale.py`;
Test `tests/test_run_topic5_slow_state_phase1_scale.py`

**Interfaces:**
- Consumes: Tasks 1, 3–6, and the Phase 0 manifest.
- Produces: `phase1/PHASE1_STATE.json`, `phase1/scale_per_patient.csv`,
  `phase1/agreement_curves.json`, `phase1/per_subject/<subject>.json` with `n_obs`,
  `n_break`, `dwell_interval`, `delta_t_obs`, `delta_t_break`, `n_windows_per_scale`,
  `status`, `n_blocks_at_n_obs`, `admitted`, `admission_reason`.
- Exposes `model_step(scales) -> int | None`, `admit(scales, blocks, minimum) -> tuple[bool,str]`,
  `selection_scales(states, diagnostic_only) -> dict`, `holm(pvalues, n_scales) -> list[float]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_topic5_slow_state_phase1_scale.py
from scripts import run_topic5_slow_state_phase1_scale as phase1


def test_the_model_step_is_n_obs_not_the_last_reliable_scale():
    assert phase1.model_step({"n_obs": 100, "n_last_reliable": 500, "n_break": 1000}) == 100


def test_admission_requires_a_scale_and_forty_blocks():
    assert phase1.admit({"status": "SCALE_RESOLVED", "n_obs": 100}, blocks=60, minimum=40) == (
        True, "ADMITTED",
    )
    assert phase1.admit({"status": "SCALE_RESOLVED", "n_obs": 100}, blocks=39, minimum=40) == (
        False, "UNRESOLVED_INSUFFICIENT_BLOCKS",
    )
    assert phase1.admit({"status": "UNRESOLVED_SCALE", "n_obs": None}, blocks=900, minimum=40) == (
        False, "UNRESOLVED_SCALE",
    )
    assert phase1.admit(
        {"status": "UNRESOLVED_NONMONOTONE", "n_obs": None}, blocks=900, minimum=40
    ) == (False, "UNRESOLVED_NONMONOTONE")


def test_a_patient_without_a_scale_is_never_called_stationary():
    _, reason = phase1.admit({"status": "UNRESOLVED_SCALE", "n_obs": None}, blocks=900, minimum=40)
    assert "STATIONARY" not in reason.upper()


def test_diagnostic_only_windows_never_enter_scale_selection():
    assert phase1.selection_scales(
        {50: "RELIABLE", 5000: "RELIABLE"}, diagnostic_only=[5000]
    ) == {50: "RELIABLE"}


def test_holm_correction_is_applied_to_reported_per_scale_statistics():
    corrected = phase1.holm([0.001, 0.02, 0.30], n_scales=13)
    assert corrected[0] <= corrected[1] <= corrected[2]
    assert corrected[0] == min(1.0, 0.001 * 13)
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — module missing.

- [ ] **Step 3: Write the runner**

The per-patient worker loops the event grid and the clock grid, enumerating **primary**
windows via Task 5, computing agreements via Task 6, reducing each scale to a state, and
selecting `n_obs` / `n_break`. Sliding-offset windows are computed separately and stored under
a `sensitivity` key, never mixed into the primary. Pool size comes from
`config["default_workers"]` and is overridable with `--workers`.

- [ ] **Step 4: Run to verify it passes**

Expected: PASS (5 tests).

- [ ] **Step 5: Measure peak RSS on three patients before the full run**

```bash
/usr/bin/time -v /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/run_topic5_slow_state_phase1_scale.py --workers 1 \
  --subjects epilepsiae_1146 yuquan_zhangkexuan epilepsiae_635 2>&1 | grep 'Maximum resident'
```

Record the value in the commit message. Choose the worker count so that
`workers × peak_RSS < 40 GiB` and `workers <= min(24, cores - 2)`.

- [ ] **Step 6: Execute Phase 1 under nohup with the chosen worker count.**

- [ ] **Step 7: Report the admission denominator before reading anything else**

Print how many of 34 were admitted and the reason for every patient that was not. This is the
denominator for the whole contract and is recorded before any Phase 2 result exists.

- [ ] **Step 8: Commit**

```bash
git add scripts/run_topic5_slow_state_phase1_scale.py \
        tests/test_run_topic5_slow_state_phase1_scale.py
git commit -m "feat(topic5-v4): phase 1 two-scale estimation and admission gate"
```

---

## Task 9: State geometry with valid nulls

**Files:** Create `src/topic5_slow_state_geometry.py`; Test `tests/test_topic5_slow_state_geometry.py`

**Interfaces:**
- Consumes: block descriptor vectors at `N_obs`.
- Produces:
  - `circular_shift_null(block_vectors, rng) -> np.ndarray` — each dimension independently
    circularly shifted.
  - `phase_surrogate_null(block_vectors, rng) -> np.ndarray` — per-dimension phase
    randomisation.
  - `geometry_verdict(block_vectors, *, cluster_grid, null_draws, seed, train_index) -> dict`
    with `verdict:str`, `best_k:int|None`, `cluster_stability:float|None`,
    `cluster_null_q95:float|None`, `lag1_autocorrelation:float|None`,
    `continuous_null_q95:float|None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_slow_state_geometry.py
import inspect

import numpy as np

from src.topic5_slow_state_geometry import (
    circular_shift_null,
    geometry_verdict,
    phase_surrogate_null,
)


def test_row_shuffling_is_not_used_because_kmeans_ignores_row_order():
    source = inspect.getsource(geometry_verdict)
    assert "circular_shift_null" in source
    assert "phase_surrogate_null" in source


def test_the_cluster_null_preserves_each_dimension_marginal():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 4))
    null = circular_shift_null(data, rng)
    for column in range(4):
        assert np.allclose(np.sort(data[:, column]), np.sort(null[:, column]))


def test_the_cluster_null_destroys_cross_dimensional_co_occurrence():
    rng = np.random.default_rng(1)
    latent = rng.normal(size=(400, 1))
    data = np.hstack([latent, latent + 0.01 * rng.normal(size=(400, 1))])
    null = circular_shift_null(data, rng)
    assert abs(np.corrcoef(data[:, 0], data[:, 1])[0, 1]) > 0.9
    assert abs(np.corrcoef(null[:, 0], null[:, 1])[0, 1]) < 0.5


def test_the_continuous_null_preserves_smoothness_but_breaks_pairing():
    rng = np.random.default_rng(2)
    walk = np.cumsum(rng.normal(size=(400, 2)), axis=0)
    null = phase_surrogate_null(walk, rng)

    def lag1(x):
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])

    assert lag1(null[:, 0]) > 0.8
    assert abs(np.corrcoef(walk[:, 0], null[:, 0])[0, 1]) < 0.5


def test_two_well_separated_clusters_are_called_clustered():
    rng = np.random.default_rng(3)
    data = np.vstack([
        rng.normal(loc=-4.0, scale=0.3, size=(60, 4)),
        rng.normal(loc=+4.0, scale=0.3, size=(60, 4)),
    ])
    out = geometry_verdict(
        data, cluster_grid=[2, 3], null_draws=100, seed=1, train_index=np.arange(72)
    )
    assert out["verdict"] == "few_stable_clusters"
    assert out["best_k"] == 2


def test_a_smooth_random_walk_is_called_continuous():
    rng = np.random.default_rng(4)
    walk = np.cumsum(rng.normal(scale=0.5, size=(200, 4)), axis=0)
    out = geometry_verdict(
        walk, cluster_grid=[2, 3, 4], null_draws=100, seed=1, train_index=np.arange(120)
    )
    assert out["verdict"] == "continuous_trajectory"
    assert out["lag1_autocorrelation"] > 0.8


def test_independent_noise_is_undetermined():
    rng = np.random.default_rng(5)
    out = geometry_verdict(
        rng.normal(size=(200, 4)), cluster_grid=[2, 3], null_draws=100, seed=1,
        train_index=np.arange(120),
    )
    assert out["verdict"] == "undetermined"


def test_k_and_scaling_are_chosen_on_train_only():
    rng = np.random.default_rng(6)
    data = np.vstack([
        rng.normal(loc=-4.0, scale=0.3, size=(60, 3)),
        rng.normal(loc=+4.0, scale=0.3, size=(60, 3)),
    ])
    train = np.arange(72)
    first = geometry_verdict(data, cluster_grid=[2, 3], null_draws=50, seed=1, train_index=train)
    perturbed = data.copy()
    perturbed[72:] += 100.0  # only the held-out tail changes
    second = geometry_verdict(
        perturbed, cluster_grid=[2, 3], null_draws=50, seed=1, train_index=train
    )
    assert first["best_k"] == second["best_k"]


def test_too_few_blocks_returns_undetermined_without_raising():
    out = geometry_verdict(
        np.zeros((5, 3)), cluster_grid=[2], null_draws=10, seed=1, train_index=np.arange(3)
    )
    assert out["verdict"] == "undetermined"
    assert out["best_k"] is None
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — module missing.

- [ ] **Step 3: Write the implementation**

```python
# src/topic5_slow_state_geometry.py  (excerpt: the two nulls rev1 got wrong)
"""Geometry verdict, with nulls KMeans is not invariant to.

rev1 shuffled block row order and re-ran KMeans.  KMeans is invariant to row order, so
that destroys nothing and is not a cluster null.  The cluster null instead shifts each
descriptor dimension independently, preserving every marginal and every within-dimension
autocorrelation while destroying the cross-dimensional co-occurrence a real state
produces.  The continuous null randomises phase per dimension, preserving smoothness
while breaking the correct temporal pairing between blocks.
"""
from __future__ import annotations

import numpy as np


def circular_shift_null(block_vectors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    data = np.asarray(block_vectors, dtype=float)
    out = np.empty_like(data)
    for column in range(data.shape[1]):
        shift = int(rng.integers(1, data.shape[0])) if data.shape[0] > 1 else 0
        out[:, column] = np.roll(data[:, column], shift)
    return out


def phase_surrogate_null(block_vectors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    data = np.asarray(block_vectors, dtype=float)
    out = np.empty_like(data)
    for column in range(data.shape[1]):
        series = data[:, column]
        spectrum = np.fft.rfft(series - series.mean())
        phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape)
        phases[0] = 0.0
        if series.size % 2 == 0:
            phases[-1] = 0.0
        surrogate = np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), n=series.size)
        out[:, column] = surrogate + series.mean()
    return out
```

`geometry_verdict` standardises using train-only statistics, fits KMeans for each `k` on the
train rows, scores split-half adjusted Rand as `cluster_stability`, compares it with the
`q95` of `null_draws` `circular_shift_null` repetitions, and compares lag-1 autocorrelation
with the `q95` of `phase_surrogate_null` repetitions. Clustered when stability beats its
null; continuous when it does not but autocorrelation beats its null; otherwise undetermined.
Fewer than `2 * max(cluster_grid) + 2` rows returns `undetermined` without raising.

- [ ] **Step 4: Run to verify it passes**

Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_slow_state_geometry.py tests/test_topic5_slow_state_geometry.py
git commit -m "fix(topic5-v4): replace row-shuffle null with circular-shift and phase surrogates"
```

---

## Task 10: Factorised emission and locked causal recursion

**Files:** Create `src/topic5_slow_state_model.py`; Test `tests/test_topic5_slow_state_model.py`

**Interfaces:**
- Consumes: blocks from Task 3.
- Produces:
  - `EventEncoder(n_contacts, hidden)` → `forward(mask, rank, group_ids) -> Tensor[n_events, hidden]`.
  - `BlockPool()` → `forward(u) -> Tensor[2*hidden]` (location and dispersion).
  - `SlowTransition(hidden, mode, n_states=None)` →
    `forward(h, pooled, delta_t) -> tuple[Tensor, Tensor]` returning `(h_next, log_scale)`
    where `log_scale` is monotone in `delta_t`; the semi-Markov variant also exposes
    `dwell_logits` and `gap_marginal_steps(delta_t)`.
  - `FactorisedDecoder(hidden, n_contacts, n_nuisance)` with `participation_head`,
    `ordering_head` and
    `log_likelihood(h, mask, rank, group_ids, nuisance) -> Tensor[]`.
  - `rollout_states(encoder, pool, transition, blocks) -> list[Tensor]` where element `b`
    is the state used to score block `b` and depends only on blocks `< b`.
  - `fit_slow_state(blocks, *, mode, seed, n_states, epochs, nuisance=None) -> dict` with
    `states:list[int]`, `held_out_log_likelihood:float`, `dwell_lengths:list[int]`,
    `n_parameters:int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_slow_state_model.py
import inspect

import numpy as np
import torch

from src.topic5_slow_state_model import (
    BlockPool,
    EventEncoder,
    FactorisedDecoder,
    SlowTransition,
    fit_slow_state,
    rollout_states,
)


def test_the_emission_factorises_into_participation_and_conditional_ordering():
    decoder = FactorisedDecoder(hidden=4, n_contacts=5, n_nuisance=4)
    assert hasattr(decoder, "participation_head")
    assert hasattr(decoder, "ordering_head")


def test_a_non_participating_contact_is_scored_by_the_mask_not_by_a_rank():
    torch.manual_seed(0)
    decoder = FactorisedDecoder(hidden=4, n_contacts=3, n_nuisance=4)
    h, nuisance = torch.zeros(4), torch.zeros(4)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    groups = torch.tensor([[0, 1, -1]])
    a = decoder.log_likelihood(h, mask, torch.tensor([[0.1, 0.9, 0.0]]), groups, nuisance)
    b = decoder.log_likelihood(h, mask, torch.tensor([[0.1, 0.9, 99.0]]), groups, nuisance)
    assert torch.allclose(a, b, atol=1e-6)


def test_participating_late_and_not_participating_get_different_likelihoods():
    torch.manual_seed(0)
    decoder = FactorisedDecoder(hidden=4, n_contacts=3, n_nuisance=4)
    h, nuisance = torch.zeros(4), torch.zeros(4)
    late = decoder.log_likelihood(
        h, torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([[0.1, 0.5, 0.9]]),
        torch.tensor([[0, 1, 2]]), nuisance,
    )
    absent = decoder.log_likelihood(
        h, torch.tensor([[1.0, 1.0, 0.0]]), torch.tensor([[0.1, 0.5, 0.0]]),
        torch.tensor([[0, 1, -1]]), nuisance,
    )
    assert not torch.allclose(late, absent, atol=1e-4)


def test_tied_contacts_are_not_scored_as_an_ordered_pair():
    torch.manual_seed(0)
    decoder = FactorisedDecoder(hidden=4, n_contacts=3, n_nuisance=4)
    h, nuisance = torch.zeros(4), torch.zeros(4)
    mask, rank = torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([[0.1, 0.1, 0.9]])
    tied = decoder.log_likelihood(h, mask, rank, torch.tensor([[0, 0, 1]]), nuisance)
    ordered = decoder.log_likelihood(h, mask, rank, torch.tensor([[0, 1, 2]]), nuisance)
    assert not torch.allclose(tied, ordered, atol=1e-4)


def test_block_likelihood_sums_over_every_event():
    torch.manual_seed(0)
    decoder = FactorisedDecoder(hidden=4, n_contacts=4, n_nuisance=4)
    h, nuisance = torch.zeros(4), torch.zeros(4)
    mask, rank = torch.ones(7, 4), torch.rand(7, 4)
    groups = torch.arange(4).repeat(7, 1)
    total = decoder.log_likelihood(h, mask, rank, groups, nuisance)
    piecewise = sum(
        decoder.log_likelihood(h, mask[i : i + 1], rank[i : i + 1], groups[i : i + 1], nuisance)
        for i in range(7)
    )
    assert torch.allclose(total, piecewise, atol=1e-5)


def test_the_target_block_never_enters_the_state_that_predicts_it():
    torch.manual_seed(0)
    blocks = [
        {"mask": torch.ones(5, 3), "rank": torch.rand(5, 3),
         "groups": torch.arange(3).repeat(5, 1), "delta_t": 60.0}
        for _ in range(6)
    ]
    encoder, pool, transition = EventEncoder(3, 4), BlockPool(), SlowTransition(4, "continuous")
    states = rollout_states(encoder, pool, transition, blocks)
    altered = [dict(block) for block in blocks]
    altered[3]["rank"] = torch.rand(5, 3)
    altered_states = rollout_states(encoder, pool, transition, altered)
    assert torch.allclose(states[3], altered_states[3], atol=1e-6)
    assert not torch.allclose(states[4], altered_states[4], atol=1e-6)


def test_transition_uncertainty_grows_with_the_gap():
    torch.manual_seed(0)
    transition = SlowTransition(hidden=4, mode="continuous")
    h, pooled = torch.randn(4), torch.randn(8)
    _, near = transition(h, pooled, torch.tensor(10.0))
    _, far = transition(h, pooled, torch.tensor(100000.0))
    assert float(far) > float(near)


def test_the_state_is_not_reset_at_a_session_boundary():
    torch.manual_seed(0)
    transition = SlowTransition(hidden=4, mode="continuous")
    pooled, gap = torch.randn(8), torch.tensor(100000.0)
    first, _ = transition(torch.randn(4), pooled, gap)
    second, _ = transition(torch.randn(4) + 5.0, pooled, gap)
    assert not torch.allclose(first, second, atol=1e-4)


def test_semi_markov_marginalises_switches_inside_an_unobserved_gap():
    torch.manual_seed(0)
    transition = SlowTransition(hidden=4, mode="semi_markov", n_states=3)
    assert transition.dwell_logits.shape[0] == 3
    assert transition.gap_marginal_steps(torch.tensor(100000.0)) > transition.gap_marginal_steps(
        torch.tensor(60.0)
    )


def test_nuisance_never_reaches_the_transition():
    assert "nuisance" not in inspect.signature(SlowTransition.forward).parameters


def test_fit_recovers_two_states_from_a_synthetic_switching_sequence():
    rng = np.random.default_rng(0)
    blocks = []
    for index in range(80):
        state = 0 if (index // 20) % 2 == 0 else 1
        prob = np.array([0.9, 0.5, 0.2]) if state == 0 else np.array([0.2, 0.5, 0.9])
        order = np.array([0.1, 0.5, 0.9]) if state == 0 else np.array([0.9, 0.5, 0.1])
        blocks.append({
            "mask": torch.tensor((rng.random((30, 3)) < prob).astype(np.float32)),
            "rank": torch.tensor(np.tile(order, (30, 1)), dtype=torch.float32),
            "groups": torch.arange(3).repeat(30, 1),
            "delta_t": 60.0,
            "true_state": state,
        })
    out = fit_slow_state(blocks, mode="semi_markov", seed=0, n_states=2, epochs=200)
    inferred = np.asarray(out["states"])
    truth = np.asarray([b["true_state"] for b in blocks])
    assert max((inferred == truth).mean(), (inferred != truth).mean()) > 0.85


def test_fit_does_not_shred_a_stationary_sequence_into_short_runs():
    rng = np.random.default_rng(1)
    blocks = [
        {"mask": torch.tensor((rng.random((30, 3)) < 0.7).astype(np.float32)),
         "rank": torch.tensor(np.tile([0.1, 0.5, 0.9], (30, 1)), dtype=torch.float32),
         "groups": torch.arange(3).repeat(30, 1), "delta_t": 60.0}
        for _ in range(80)
    ]
    out = fit_slow_state(blocks, mode="semi_markov", seed=0, n_states=2, epochs=200)
    assert float(np.median(out["dwell_lengths"])) > 10.0
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — module missing.

- [ ] **Step 3: Write the implementation**

`FactorisedDecoder.log_likelihood` = per-contact Bernoulli on the mask conditioned on
`[h, nuisance]`, **plus** a Plackett-Luce ordering term over participating contacts only, in
which contacts sharing a group form one tied unit whose internal order contributes no term.
Non-participating contacts contribute only their Bernoulli term, which is what the
"rank is meaningless when absent" test pins.

`rollout_states` returns `h_{b-1}` as the state used to score block `b`. `SlowTransition`
returns `(h_next, log_scale)` with `log_scale = softplus(w · log1p(delta_t) + c)`. The
semi-Markov variant exposes `gap_marginal_steps(delta_t) = 1 + floor(delta_t / dwell_scale)`
and marginalises that many unobserved transitions.

**Sanity check during implementation:** verify the tied-group Plackett-Luce normalisation
against a brute-force enumeration over all orderings of three contacts before proceeding.

- [ ] **Step 4: Run to verify it passes**

Expected: PASS (12 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_slow_state_model.py tests/test_topic5_slow_state_model.py
git commit -m "feat(topic5-v4): factorised participation+ordering emission with causal lock"
```

---

## Task 11: Observation-layer nuisances

**Files:** Modify `src/topic5_slow_state_model.py` and `tests/test_topic5_slow_state_model.py`

**Interfaces:**
- Consumes: block times, the per-dataset timezone contract already used by
  `src/topic5_propagation_drift_diurnal.py`.
- Produces: `build_nuisance(block, timezone_name) -> np.ndarray` of length 4 —
  `[sin(2π·hour/24), cos(2π·hour/24), log1p(events_per_second), elapsed_recording_hours]`;
  `FactorisedDecoder(hidden, n_contacts, n_nuisance, n_recordings=1)` gaining a
  **per-recording intercept embedding** consumed via a separate `recording_index=0` argument.

  **Both new arguments must default**, because Task 10's tests construct
  `FactorisedDecoder(hidden=..., n_contacts=..., n_nuisance=...)` and call
  `log_likelihood(h, mask, rank, groups, nuisance)` with five positional arguments. Adding a
  required parameter here would break Task 10's suite, and a task that breaks an earlier
  task's tests has not been implemented correctly.

- [ ] **Step 1: Write the failing test**

```python
def test_hour_is_encoded_cyclically_so_23h_and_0h_are_adjacent():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.topic5_slow_state_model import build_nuisance

    def at(hour):
        stamp = datetime(2009, 7, 1, hour, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
        return {"t_start": stamp, "t_end": stamp + 600.0, "n_events": 20,
                "elapsed_recording_seconds": 3600.0}

    late = build_nuisance(at(23), "Europe/Berlin")
    early = build_nuisance(at(0), "Europe/Berlin")
    noon = build_nuisance(at(12), "Europe/Berlin")
    assert np.linalg.norm(late[:2] - early[:2]) < np.linalg.norm(late[:2] - noon[:2])


def test_nuisance_carries_event_rate_not_raw_count():
    from src.topic5_slow_state_model import build_nuisance

    fast = build_nuisance(
        {"t_start": 0.0, "t_end": 100.0, "n_events": 20, "elapsed_recording_seconds": 0.0},
        "Europe/Berlin",
    )
    slow = build_nuisance(
        {"t_start": 0.0, "t_end": 1000.0, "n_events": 20, "elapsed_recording_seconds": 0.0},
        "Europe/Berlin",
    )
    assert fast[2] > slow[2]


def test_recording_identity_is_an_intercept_not_a_linear_covariate():
    from src.topic5_slow_state_model import build_nuisance

    block = {"t_start": 0.0, "t_end": 600.0, "n_events": 20, "elapsed_recording_seconds": 0.0}
    # the continuous nuisance vector must not encode recording index at all
    assert len(build_nuisance(block, "Europe/Berlin")) == 4


def test_the_recording_intercept_is_an_embedding_that_generalises_by_pooling():
    torch.manual_seed(0)
    decoder = FactorisedDecoder(hidden=4, n_contacts=3, n_nuisance=4, n_recordings=5)
    assert hasattr(decoder, "recording_intercept")
    assert decoder.recording_intercept.num_embeddings == 5


def test_elapsed_recording_time_is_present_and_monotone():
    from src.topic5_slow_state_model import build_nuisance

    early = build_nuisance(
        {"t_start": 0.0, "t_end": 600.0, "n_events": 20, "elapsed_recording_seconds": 0.0},
        "Europe/Berlin",
    )
    later = build_nuisance(
        {"t_start": 0.0, "t_end": 600.0, "n_events": 20, "elapsed_recording_seconds": 72000.0},
        "Europe/Berlin",
    )
    assert later[3] > early[3]


def test_the_decoder_conditions_on_nuisance_so_hour_can_be_partialled_out():
    torch.manual_seed(0)
    decoder = FactorisedDecoder(hidden=4, n_contacts=3, n_nuisance=4, n_recordings=2)
    h, mask = torch.zeros(4), torch.ones(2, 3)
    rank, groups = torch.rand(2, 3), torch.arange(3).repeat(2, 1)
    day = decoder.log_likelihood(
        h, mask, rank, groups, torch.tensor([0.0, 1.0, 0.5, 0.2]), recording_index=0
    )
    night = decoder.log_likelihood(
        h, mask, rank, groups, torch.tensor([0.0, -1.0, 0.5, 0.2]), recording_index=0
    )
    assert not torch.allclose(day, night, atol=1e-5)
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `build_nuisance` undefined; `FactorisedDecoder` has no `n_recordings`.

- [ ] **Step 3: Write the implementation**

```python
def build_nuisance(block, timezone_name: str) -> np.ndarray:
    """Observation-layer nuisances. Recording identity is NOT in this vector.

    rev1 encoded a normalised session index here, which forced a discrete recording
    identity into a linear trend and could not generalise to an unseen recording.  The
    recording enters as a pooled intercept embedding on the decoder instead.
    """
    from src.topic5_propagation_drift_diurnal import epoch_to_local_hour

    hour = epoch_to_local_hour(float(block["t_start"]), timezone_name)
    span = max(float(block["t_end"]) - float(block["t_start"]), 1e-6)
    rate = float(block["n_events"]) / span
    return np.asarray(
        [
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            float(np.log1p(rate)),
            float(block["elapsed_recording_seconds"]) / 3600.0,
        ],
        dtype=float,
    )
```

`FactorisedDecoder.__init__` gains `n_recordings` and a
`recording_intercept = nn.Embedding(n_recordings, 1)`; `log_likelihood` gains a
`recording_index` argument added to both heads' logits. `SlowTransition.forward` is **not**
changed.

- [ ] **Step 4: Run to verify it passes**

Expected: PASS (18 tests in this file).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_slow_state_model.py tests/test_topic5_slow_state_model.py
git commit -m "feat(topic5-v4): cyclic-hour, rate and recording-intercept nuisances"
```

---

## Task 12: Mandatory synthetic calibration

**Files:** Create `scripts/run_topic5_slow_state_synthetic_calibration.py`;
Test `tests/test_run_topic5_slow_state_synthetic_calibration.py`

**Interfaces:**
- Consumes: `fit_slow_state` from Task 10, `config["synthetic_grid"]`.
- Produces: `synthetic/CALIBRATION_STATE.json`, `synthetic/recovery_grid.csv` with columns
  `n_states`, `dwell_blocks`, `noise`, `n_blocks`, `seed`, `state_recovery`,
  `dwell_recovery`, `false_structure_rate`; and
  `interpretable_regime(n_states, dwell, n_blocks, *, table, minimum) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_topic5_slow_state_synthetic_calibration.py
from scripts import run_topic5_slow_state_synthetic_calibration as calib


def test_the_grid_covers_every_spec_cell():
    cells = calib.grid_cells({
        "n_states": [2, 3], "dwell_blocks": [5, 10],
        "noise_levels": [0.2, 0.5], "n_blocks": [40, 80], "seeds": [0, 1],
    })
    assert len(cells) == 2 * 2 * 2 * 2 * 2


def test_a_stationary_sequence_contributes_a_false_structure_rate_not_a_recovery():
    row = calib.simulate_cell(n_states=1, dwell_blocks=0, noise=0.2, n_blocks=80, seed=0)
    assert row["state_recovery"] is None
    assert 0.0 <= row["false_structure_rate"] <= 1.0


def test_interpretable_regime_is_false_for_few_blocks_and_long_dwell():
    table = {(2, 50, 40): 0.10, (2, 5, 320): 0.95}
    assert calib.interpretable_regime(2, 50, 40, table=table, minimum=0.8) is False
    assert calib.interpretable_regime(2, 5, 320, table=table, minimum=0.8) is True


def test_an_unmeasured_regime_is_not_silently_called_interpretable():
    assert calib.interpretable_regime(5, 50, 40, table={}, minimum=0.8) is False
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — module missing.

- [ ] **Step 3: Write the simulator and sweep**, reusing `fit_slow_state`. A `n_states=1` cell
  measures how often structure is invented in a stationary sequence.

- [ ] **Step 4: Run to verify it passes**

Expected: PASS (4 tests).

- [ ] **Step 5: Execute the calibration under nohup with the measured worker count.**

- [ ] **Step 6: Commit**

```bash
git add scripts/run_topic5_slow_state_synthetic_calibration.py \
        tests/test_run_topic5_slow_state_synthetic_calibration.py
git commit -m "feat(topic5-v4): synthetic recovery calibration and interpretable-regime map"
```

---

## Task 13: Secondary template descriptors

**Files:** Create `src/topic5_slow_state_templates.py`; Test `tests/test_topic5_slow_state_templates.py`

Spec §6.2 promises template occupancy and within-template dispersion as reported secondary
quantities. This task delivers them so the promise is met, and pins that they cannot reach any
decision.

**Interfaces:**
- Produces: `fit_templates(train_blocks, *, k_grid, null_draws, seed) -> dict` with `k:int|None`,
  `centroids:np.ndarray|None`, `stability:float|None`, `null_q95:float|None`, `status:str`,
  `decides_scale_or_admission:bool` (always `False`);
  `template_occupancy(block_vectors, templates) -> dict` with `occupancy:list[float]`,
  `dispersion:float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_slow_state_templates.py
import inspect

import numpy as np
import pytest

from src.topic5_slow_state_templates import fit_templates, template_occupancy


def test_templates_are_fitted_on_train_blocks_only():
    assert "train_blocks" in inspect.signature(fit_templates).parameters


def test_k_is_chosen_against_a_null_not_by_a_constant():
    rng = np.random.default_rng(0)
    out = fit_templates(rng.normal(size=(200, 4)), k_grid=[2, 3], null_draws=100, seed=1)
    assert out["null_q95"] is not None


def test_pure_noise_yields_unresolved_templates():
    rng = np.random.default_rng(1)
    out = fit_templates(rng.normal(size=(200, 4)), k_grid=[2, 3], null_draws=100, seed=1)
    assert out["status"] == "TEMPLATES_UNRESOLVED"


def test_occupancy_sums_to_one_and_dispersion_is_non_negative():
    rng = np.random.default_rng(2)
    data = np.vstack([
        rng.normal(loc=-3.0, scale=0.2, size=(80, 3)),
        rng.normal(loc=+3.0, scale=0.2, size=(80, 3)),
    ])
    templates = fit_templates(data, k_grid=[2], null_draws=50, seed=1)
    out = template_occupancy(data[:40], templates)
    assert sum(out["occupancy"]) == pytest.approx(1.0, abs=1e-6)
    assert out["dispersion"] >= 0.0


def test_secondary_quantities_are_flagged_as_non_deciding():
    rng = np.random.default_rng(3)
    templates = fit_templates(rng.normal(size=(120, 3)), k_grid=[2], null_draws=50, seed=1)
    assert templates["decides_scale_or_admission"] is False
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement**, reusing `circular_shift_null` from Task 9 for the template null.
- [ ] **Step 4: Run to verify it passes** — 5 tests PASS.
- [ ] **Step 5: Commit**

```bash
git add src/topic5_slow_state_templates.py tests/test_topic5_slow_state_templates.py
git commit -m "feat(topic5-v4): secondary template occupancy and dispersion"
```

---

## Task 14: Per-patient acceptance with the underpowered verdict

**Files:** Create `src/topic5_slow_state_acceptance.py`; Test `tests/test_topic5_slow_state_acceptance.py`

**Interfaces:**
- Consumes: Task 10 fit output, Task 12 `interpretable_regime`.
- Produces: `patient_verdict(checks, *, interpretable_regime) -> str`;
  `dwell_null(labels, *, draws, seed) -> dict` with `observed_median_dwell`,
  `null_median_dwell`, `p_value`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_slow_state_acceptance.py
import numpy as np
import pytest

from src.topic5_slow_state_acceptance import dwell_null, patient_verdict

ALL_PASS = {
    "beats_estimation_noise": True, "beats_stationary_null": True,
    "reproduces_across_seeds": True, "state_conditioned_distributions_differ": True,
    "dwell_beats_random_label_null": True,
}
ALL_FAIL_BUT_NOISE = {
    "beats_estimation_noise": True, "beats_stationary_null": False,
    "reproduces_across_seeds": True, "state_conditioned_distributions_differ": False,
    "dwell_beats_random_label_null": False,
}


def test_a_patient_passing_every_check_in_an_interpretable_regime_is_identifiable():
    assert patient_verdict(ALL_PASS, interpretable_regime=True) == "identifiable"


def test_a_failing_patient_in_an_uninterpretable_regime_is_underpowered_not_stationary():
    assert patient_verdict(
        ALL_FAIL_BUT_NOISE, interpretable_regime=False
    ) == "UNDERPOWERED_FOR_ITS_REGIME"


def test_a_failing_patient_in_an_interpretable_regime_is_stationary():
    assert patient_verdict(ALL_FAIL_BUT_NOISE, interpretable_regime=True) == "stationary"


def test_a_patient_that_cannot_beat_estimation_noise_is_coverage_limited():
    checks = {name: False for name in ALL_PASS}
    assert patient_verdict(checks, interpretable_regime=True) == "unresolved_coverage"


def test_states_without_dwell_are_a_continuous_candidate_not_switching():
    checks = {**ALL_PASS, "dwell_beats_random_label_null": False}
    assert patient_verdict(checks, interpretable_regime=True) == "continuous_state_candidate"


def test_dwell_null_preserves_label_marginals_while_destroying_runs():
    out = dwell_null(np.array([0] * 40 + [1] * 40), draws=200, seed=0)
    assert out["observed_median_dwell"] == pytest.approx(40.0)
    assert out["null_median_dwell"] < 5.0
    assert out["p_value"] < 0.01


def test_dwell_null_does_not_call_an_alternating_sequence_long_dwelling():
    assert dwell_null(np.array([0, 1] * 40), draws=200, seed=0)["p_value"] > 0.05
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement.** `patient_verdict` checks `beats_estimation_noise` first — a
  patient that fails it is `unresolved_coverage`, never `stationary`, because that is a
  coverage statement not a biological one. A patient that fails the state checks is
  `stationary` **only** when its regime is interpretable; otherwise
  `UNDERPOWERED_FOR_ITS_REGIME`. `dwell_null` permutes the label sequence keeping marginal
  counts fixed and compares median run length.
- [ ] **Step 4: Run to verify it passes** — 7 tests PASS.
- [ ] **Step 5: Commit**

```bash
git add src/topic5_slow_state_acceptance.py tests/test_topic5_slow_state_acceptance.py
git commit -m "feat(topic5-v4): per-patient verdicts including underpowered regime"
```

---

## Task 15: Release gate, Phase 2 runner and cohort adjudication

**Files:** Create `scripts/freeze_topic5_slow_state_phase2_release.py`,
`scripts/run_topic5_slow_state_phase2_model.py`, `scripts/accept_topic5_slow_state_v4_0.py`;
Tests `tests/test_freeze_topic5_slow_state_phase2_release.py`,
`tests/test_accept_topic5_slow_state_v4_0.py`

**Interfaces:**
- `release.CHECKLIST` — the eleven spec §18 item names, including
  `synthetic_calibration_and_regime_map`.
- `release.model_family(verdict) -> list[str]`; `release.validate(checklist) -> None`;
  `release.assert_no_outcome(output_dir) -> None`.
- `accept.cohort_prevalence(verdicts, deep_stratum=None) -> dict`;
  `accept.assign_levels(verdicts, dwell_pass) -> dict[str, str | None]`;
  `accept.summarise_undetermined(arms) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_freeze_topic5_slow_state_phase2_release.py
import pytest

from scripts import freeze_topic5_slow_state_phase2_release as release


def test_model_family_follows_the_frozen_geometry_verdict():
    assert release.model_family("few_stable_clusters") == ["semi_markov"]
    assert release.model_family("continuous_trajectory") == ["continuous"]
    assert release.model_family("undetermined") == ["semi_markov", "continuous"]


def test_an_unknown_geometry_verdict_fails_loudly():
    with pytest.raises(ValueError):
        release.model_family("mystery")


def test_release_requires_the_synthetic_calibration_to_exist():
    checklist = {name: True for name in release.CHECKLIST}
    checklist["synthetic_calibration_and_regime_map"] = False
    with pytest.raises(RuntimeError):
        release.validate(checklist)


def test_release_refuses_if_a_phase2_result_already_exists(tmp_path):
    (tmp_path / "PHASE2_STATE.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError):
        release.assert_no_outcome(tmp_path)
```

```python
# tests/test_accept_topic5_slow_state_v4_0.py
import pytest

from scripts import accept_topic5_slow_state_v4_0 as accept


def test_cohort_reports_prevalence_not_a_mean_effect():
    verdicts = ["identifiable"] * 6 + ["stationary"] * 4 + ["unresolved_coverage"] * 24
    out = accept.cohort_prevalence(verdicts)
    assert out["n_total"] == 34
    assert out["prevalence_identifiable"] == pytest.approx(6 / 34)
    assert "mean_effect" not in out


def test_unresolved_and_underpowered_patients_stay_in_the_denominator():
    out = accept.cohort_prevalence(
        ["identifiable"] * 2 + ["UNDERPOWERED_FOR_ITS_REGIME"] * 10 + ["unresolved_coverage"] * 22
    )
    assert out["n_total"] == 34
    assert out["n_underpowered"] == 10


def test_underpowered_is_not_mapped_onto_an_evidence_level():
    levels = accept.assign_levels(
        {"a": "identifiable", "b": "UNDERPOWERED_FOR_ITS_REGIME"},
        dwell_pass={"a": True, "b": False},
    )
    assert levels["a"] == "S3"
    assert levels["b"] is None


def test_both_arms_are_reported_when_the_geometry_verdict_was_undetermined():
    out = accept.summarise_undetermined(
        {"semi_markov": {"verdict": "identifiable"}, "continuous": {"verdict": "stationary"}}
    )
    assert set(out) == {"semi_markov", "continuous"}
    assert "selected" not in out


def test_deep_stratum_is_reported_separately_and_cannot_replace_the_cohort():
    out = accept.cohort_prevalence(
        ["identifiable"] * 4 + ["stationary"] * 30,
        deep_stratum=["epilepsiae_1096", "epilepsiae_1073", "epilepsiae_958", "epilepsiae_922"],
    )
    assert out["deep_stratum_size"] == 4
    assert out["n_total"] == 34
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement all three scripts.** The Phase 2 runner fits each admitted patient
  with the released family or families and the three configured seeds, on a chronological
  60/20/20 split, scoring within-session and cross-gap transitions separately. The acceptance
  script is read-only with respect to fitting and refuses to run if
  `PHASE2_RELEASE_STATE.json` is absent or its hashes do not match.
- [ ] **Step 4: Run to verify they pass** — 9 tests PASS.
- [ ] **Step 5: Run the whole scoped suite**

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest -q \
  $(rg --files tests | rg 'topic5_(slow_state|source_intervals)' | tr '\n' ' ')
```

- [ ] **Step 6: Commit**

```bash
git add scripts/freeze_topic5_slow_state_phase2_release.py \
        scripts/run_topic5_slow_state_phase2_model.py \
        scripts/accept_topic5_slow_state_v4_0.py tests/
git commit -m "feat(topic5-v4): release gate, phase 2 runner and prevalence adjudication"
```

---

## Task 16: Archive report and index

**Files:** Create `docs/archive/topic5/slow_state_two_timescale_v4_0_phase1_2_<YYYY-MM-DD>.md`;
Modify `docs/archive/topic5/INDEX.md`

- [ ] **Step 1: Write the report** in Chinese, opening with the three-part plain-language
  summary required by `CLAUDE.md` §8, archive code names only in a trailing parenthetical.
  Must state: the admission denominator and every exclusion reason; `N_obs` and the dwell
  **interval** distributions; geometry verdict counts and which nulls were used; per-patient
  evidence levels with `UNDERPOWERED_FOR_ITS_REGIME` reported separately from `stationary`;
  within-session versus cross-gap results separately; the deep stratum separately; and the
  spec §2 corrections to the V2.7 / V3.0 wording.
- [ ] **Step 2: Register in `INDEX.md`** at the top of `## 主线（network-axis pivot）`.
- [ ] **Step 3: Verify before claiming completion**

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest -q \
  $(rg --files tests | rg 'topic5_(slow_state|source_intervals|event_innovation|propagation_drift|stateful_event_rnn_v2_7)' | tr '\n' ' ')
git diff --check
python3 -c "
import json,hashlib,pathlib
def s(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
r='results/topic5_event_innovation_impulse_response/v3_0/'
rel=json.load(open(r+'HUMAN_TEST_RELEASE_STATE.json'))
assert s(r+'V3_1_HANDOFF_STATE.json')==rel['inputs_sha256']['handoff']
assert json.load(open(r+'V3_1_HANDOFF_STATE.json'))['status']=='NOT_TRIGGERED'
print('v3.0/v3.1 frozen artifacts intact')
"
```

- [ ] **Step 4: Commit** `docs(topic5-v4): archive phase 1-2 acceptance and register in index`

---

## Known limits of this plan, stated rather than hidden

- **Tasks 7, 8, 12, 13, 14 and 15 describe implementations in prose.** Every test body is
  written in full, so behaviour is pinned; a complete semi-Markov forward-backward or a worker
  pool written inline would make the plan unusable. Tasks 1–6 and 9–11 carry complete code for
  the parts where a wrong implementation would fail silently.
- **Spec §5's feasibility numbers came from event density and will change** once §4.1 metadata
  intervals are used. Task 7 exists to detect that; it writes `PHASE0_RECONCILIATION.md` and
  stops rather than editing the contract. This branch is expected, not exceptional.
- **Runtime is unestimated.** Phase 1 sweeps 6 event scales × 7 clock scales × 200 random
  halves × 200 null draws per patient, dominated by `local_repertoire` at
  \(O(\text{contacts}^2)\). Task 8 Step 5 measures peak RSS on three patients before choosing
  the worker count; ceilings are `workers × peak_RSS < 40 GiB` and `min(24, cores - 2)`.
- **The tied-group Plackett-Luce term is the least standard piece.** Its four Task 10 tests pin
  what matters, but the normalisation over tied groups must be checked against a brute-force
  enumeration on three contacts during implementation, as Task 10 Step 3 instructs.
- **`window_agreements` cost is quadratic in contacts and linear in draws.** If Phase 1 proves
  too slow, the admissible lever is reducing `null_draws` / `random_half_draws` — never
  reducing the number of independent windows, which is the statistical unit.

## Out of scope for this plan

Phase 4 (block innovation → switching hazard, spec §15) runs only for patients adjudicated
`identifiable`, and its plan is written after the Phase 3 denominator is known. Writing it now
would be planning against an unknown cohort.
