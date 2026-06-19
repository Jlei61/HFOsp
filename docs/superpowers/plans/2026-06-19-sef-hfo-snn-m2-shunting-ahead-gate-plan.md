# M2 faithful test: conductance shunting + ahead-of-front recruitment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** **First (Task 0, data-side, NO model change): measure, in SEEG CONTACT space, how far each real interictal HFO group event's participating channels spread along the accepted source→sink axis vs a shaft-matched sampling null. This is a methodological CONSTRAINT on what the observation layer can show — NOT a tissue-space claim and NOT a stop-gate on the model.** Task 0 (executed 2026-06-19, n=23): contact footprint is axially LONG (~92% of the recruited pool's axial range) AND ≈ the shaft-matched null → contact-space extent is largely sampling-determined; this CANNOT prove or disprove tissue-space self-limit. Then faithfully test the M2 front-inhibition gate by fixing the two implementation gaps the 2026-06-19 audit found — the engine's inhibition is current-subtraction (not conductance shunting) and the gate's inhibition is recruited AT the front (not ahead) — with direct ahead-of-front + axial-clamp diagnostics, and judge it against a **two-layer** target (full-field tissue self-limit AND virtual-SEEG reproduces Task 0's contact-space AF/LR), NOT "shrink the contact footprint to a segment".

**Architecture:** Two default-OFF engine additions. (1) **Conductance shunting**: replace the current-LIF membrane update `V→(I_E−I_I)` with a conductance form `V→(I_E + g_I·E_gaba)/(1+g_I)` where `g_I=g_gaba_scale·I_I` is the GABA conductance — this gates spike initiation regardless of excitatory drive (a strongly-driven axial cell is clamped toward `E_gaba`, not merely offset). (2) **Wide E→I recruitment gate**: extra E→I edges (wide kernel) so the front recruits inhibitory cells AHEAD of itself, complementing the existing wide I→E veto. Both gated so the default path is bit-identical. Then re-test with shunting + ahead-recruitment + the existing I→E gate + E→E recovery, with an operating-point sanity gate first (shunting changes the E/I balance).

**Tech Stack:** Python, NumPy; `src/snn_engine/` LIF substrate; pytest; existing readout/analysis scripts (`run_sef_hfo_snn_cm_spontaneous_readout.py`, `event_field_geometry`).

## Global Constraints

- **Default-OFF bit-parity is mandatory.** With `shunt_gaba=False` (default) AND `ei_gate_scale=0` (default), the engine must be **spike-identical** to the pre-edit engine (small-net spike SHA `da5fc18c`). Every new mechanism gated; no new RNG draw / float touch on the default path.
- **Engine edit → re-bless required.** `kick_probe.py` + `connectivity_rot.py` are in the runner's engine guard. After editing, re-bless ONLY the worktree's own `results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json` — all model work happens inside `.worktrees/topic4-m1` (see the tree-strategy constraint below); do NOT touch main's copy. The bit-parity test must pass BEFORE re-blessing.
- **PILOT-FIRST hard stop.** Do NOT launch any grid until (a) the operating-point sanity (Task 5) shows shunting-on still spontaneously ignites with a quiet rest, and (b) a single gate-on cell (Task 6 pilot) shows the mechanism changes axial extent.
- **Operating-point caveat.** Shunting changes the membrane balance (it raises effective leak → shortens τ_eff). The current-LIF operating point (drive, thresholds) was tuned without it. Task 5 MUST re-verify the base workpoint before any gate claim; if shunting kills ignition or destabilizes rest, re-tune drive / `g_gaba_scale` there, NOT inside the gate pilot.
- **Naming (verified `params.py`):** `l_IE/C_IE` = E→I (E source → I target); `l_EI/C_EI` = I→E (I source → E target). The existing M2 gate (`gate_scale/l_gate/c_gate`) widens the **I→E veto**. This plan's new recruitment gate (`ei_gate_scale/l_ei_gate/c_ei_gate`) widens the **E→I** side.
- **Claim discipline (two-layer, contact-vs-tissue — REVISED 2026-06-19 after Task 0 review).** The faithful-test verdict is "with conductance shunting + ahead-of-front recruitment, the model {does / does not} produce events that **(layer 1, full-field/tissue)** are spatially FINITE — bounded extent, L-invariant (no growth L20→L32), not boundary-filling, not tonic / full-sheet / seizure-like global recruitment — **AND (layer 2, virtual-SEEG)** after the SAME virtual-SEEG sampling reproduce Task 0's CONTACT-space AF/LR (axially long, close to the shaft-matched null)". The target is NOT "the contact footprint shrinks to a segment" — Task 0 shows the real contact footprint is long + sampling-dominated, so a model whose contact footprint is short would MISmatch the data. NOT a seizure claim. The prior "change mechanism" verdict stays WITHDRAWN until this two-layer test runs.
- **Tree strategy (CORRECTED 2026-06-19 — P0).** The ONLY tree with the full mechanism stack — E→E recovery (`ee_std_u`), wide I→E gate (`gate_scale`), `reach_axis_mm` readout — is the worktree **`.worktrees/topic4-m1`** (branch `topic4-snn-m1-recovery`). The `main` checkout currently has NONE of them (gate / recovery / runner flags all absent — verified 2026-06-19). So **ALL work in this plan happens INSIDE that worktree**: `cd .worktrees/topic4-m1` first; every file path below is relative to the worktree root; re-bless the worktree's own `engine_versions.json`. **Do NOT `cp` engine files from main → worktree** (main lacks recovery+gate and would clobber them). Reconciling/merging the worktree back to a canonical branch is a SEPARATE later step (user's call), not part of this plan.
- **Coexistence guard (mandatory — P0).** `simulate_kick` must accept BOTH `shunt_gaba` AND `ee_std_u` (recovery) at once, and the runner `config` must record BOTH. A test asserts the full combo (shunting + recovery + gate) runs and all three appear in the readout config — so the "+recovery" leg can't be silently dropped when shunting is added.
- **Data-audit CONSTRAINT (P0-science — NOT a model stop-gate; REVISED 2026-06-19).** Task 0 (data-side, NO model change) runs FIRST and constrains how the model is JUDGED, not whether it runs. It measures the real CONTACT-space footprint (how far events spread along the axis among sampled, ever-participating channels, vs a shaft-matched null). It CANNOT speak to tissue-space self-limit: the denominator is the ever-participating broad-channel pool (a contact-space quantity), and observed≈null means the footprint is largely sampling-determined. So Task 0 does NOT decide "is axial self-limit real in tissue"; it tells the model what its virtual-SEEG projection must look like (Task 7 layer 2: long + sampling-like, not a short segment).

---

### Task 0: DATA-SIDE AUDIT — the CONTACT-space footprint of real events on the accepted axis (vs a shaft-matched sampling null); calibrates the model's virtual-SEEG acceptance (no model change; NOT a model stop-gate)

**Why first:** the M2 model must be JUDGED against what the observation layer (SEEG contact space) actually shows, so we measure that first. Task 0 quantifies, in contact space, how far each event spreads along the accepted axis vs a shaft-matched sampling null — this CALIBRATES the model's virtual-SEEG acceptance (Task 7 layer 2). It is NOT a tissue-space test and does NOT gate whether the model runs (the tissue/full-field self-limit question is Task 7 layer 1). Existing skeleton geometry (`results/topic4_sef_hfo/skeleton_geometry/`) says most subjects have a reproducible axis (axis_length median ~24mm; perp_spread.rms median ~12mm) — but that perp_spread is participating-channel transverse spread (confounded by implant sampling + participation rate), and endpoint/source/sink are template/subject-level, NOT event-level footprint. So existing data supports "propagation has an axis" but does NOT answer "does each event fill the axis." This audit answers it at the EVENT level.

**Files:**
- Create: `src/topic4_event_extent_audit.py` (pure metrics)
- Create: `scripts/run_topic4_event_extent_audit.py` (load real axis + events, per-subject summary + figure)
- Test: `tests/test_topic4_event_extent_audit.py`
- Output: `results/topic4_sef_hfo/event_extent_audit/{per_subject.csv, cohort_summary.json, figures/event_extent.png + README.md}`.

**Data contract (PINNED — P1; do NOT hand-parse the NPZ, do NOT invent paths, reuse the canonical loaders):**
- **Primary broad source dirs (and ONLY these):** epilepsiae spine = `results/lagpat_broad_epilepsiae/<subject>/`; yuquan = `results/lagpat_broad/<subject>/`. EXCLUDE the sensitivity / sweep variants `results/lagpat_broad_epilepsiae_{k00,k05,km10,topn40}/` and `results/lagpat_broad_dyn/` (different channel pools — not the canonical broad pool).
- **Events (per-event participating channels):** `src.interictal_propagation.load_subject_propagation_events(subject_dir)` → per-event `eventsBool` + `channel_names` (it already handles withFreqCent/7ch fallback, block ordering by `start_t`, and shape truncation). Do NOT read the NPZ `eventsBool`/`chnNames` by hand — orientation (channel×event) + masking are the loader's contract.
- **Coords + channel-name alignment:** `src.seeg_coord_loader.load_subject_coords(ds, subj, channel_names)` → `coords_array_in_requested_order` (n,3, NaN for unmapped), `mapped_mask_in_requested_order` (bool), `coord_space`. Coords come back IN the requested `channel_names` order (alignment is built in); assert `coords.shape == (len(names), 3)`.
- **Axis frame = the ACCEPTED reproducible axis (NOT recomputed on the broad pool):** read `results/topic4_sef_hfo/skeleton_geometry/per_subject/<ds>_<subj>.json`; take the `source_core` / `sink_core` channel names from its `channels` list (entries with those `role`s) + `axis_length_mm`. Build the frame from those cores' coords (source/sink centroids → unit axis) and PROJECT the broad-pool event channels onto it with the same projection as `compute_axis_frame` (`along = (p − src_c)·u`, `off = ‖⊥‖`). Rationale: the skeleton axis was derived on the narrow mount pool — reusing its accepted source/sink centroids and projecting the broad channels keeps `along`/`off` and the axis in ONE coord frame (projection is pure geometry in a shared `coord_space`). Record `template_source` + `coord_space` from the JSON as per-subject provenance.
- **Eligibility spine (cohort denominator):** only subjects whose skeleton per_subject JSON has a real axis — `status` ok AND `degenerate_axis == False` AND finite `axis_length_mm` AND both source_core and sink_core present in the broad `channel_names`. ≈19 epilepsiae + ≈7 yuquan (the existing skeleton cards). A subject failing this is recorded with `excluded_reason`, not silently dropped.
- **Per-event exclusions (tallied by reason):** empty / degenerate NPZ blocks; events with participating coord-mapped `n_part < 5` (p5/p95 extent needs ≥5 points — record `n_part` 3–4 as a `low_n` sensitivity tier, NOT in the primary stat); channels with NaN coords (`~mapped_mask`).
- **Per-subject diagnostics (REQUIRED columns in `per_subject.csv`):** `n_events_total`, `n_events_used`, `n_events_excluded`, `excluded_reason` breakdown (`empty_block` / `low_n_part` / `no_axis` / `core_unmapped`), `n_channels_in_pool`, `n_channels_mapped`, `coord_space`, `template_source`.

**Interfaces:**
- Produces: `event_extent(along, off, axis_length) -> dict(axial_span, lateral_span, axial_fraction, lateral_ratio)` where `axial_span=p95(along)-p5(along)`, `lateral_span=p95(off)-p5(off)`, `axial_fraction=axial_span/axis_length`, `lateral_ratio=lateral_span/axial_span`. Inputs are the along-/off-axis coordinates (mm) of an event's PARTICIPATING channels.
- Produces: `matched_null_extent(along_all, off_all, n_part, axis_length, n_draw, rng, *, shaft=None, shaft_counts=None, rate=None) -> dict` — `n_draw` random same-subject eligible-channel subsets of size `n_part`, returning the null distribution of the same metrics under up to THREE matching modes (reported in layers): (1) **uniform** (`shaft=None, rate=None`) = same-subject + same n_part; (2) **rate** (`rate=`) = participation-rate-weighted draw; (3) **shaft_matched** (`shaft=` + the event's per-shaft counts `shaft_counts`) = draw the SAME number of channels from each shaft as the real event (controls the implant-sampling confound, via `propagation_skeleton_geometry.parse_shaft`). Returns `{mode: {axial_fraction_med, lateral_ratio_med, axial_fraction[], lateral_ratio[]}}` for every mode whose inputs are present (uniform always). The runner reports all three; the primary verdict uses the MOST CONSERVATIVE (smallest observed−null gap) — typically `shaft_matched`.
- Produces: `event_shaft_counts(names) -> dict(shaft -> count)` — the event's per-shaft channel counts (via `parse_shaft`, unparseable dropped); feeds `matched_null_extent(shaft=, shaft_counts=)`.
- Produces: `cohort_verdict(per_subject, rng, *, n_boot=2000, min_subjects=10) -> dict(verdict, AF, LR, axial_ci, lateral_ci, axial_wilcoxon_p, ...)` — the PRE-REGISTERED Step-9 gate as a tested pure function. Per-subject records carry `axial_obs/axial_null/lateral_obs/lateral_null` (subject median observed vs shaft_matched-null median); Δ = obs−null (confinement ⇒ Δ<0), "below null" ⇔ Δ-mean bootstrap-CI upper bound < 0. Returns one of `AXIAL_EXTENDED_LATERAL_NARROW` / `AXIAL_SEGMENT` / `SAMPLING_ARTIFACT` / `INCONCLUSIVE` per the Step-9 thresholds.

- [ ] **Step 1: Lock the data APIs against the PINNED data contract above** (inspection step, not a placeholder — the names feed Steps 6–7). In a comment block at the top of `run_topic4_event_extent_audit.py`, confirm and record: the broad source dirs (epi `lagpat_broad_epilepsiae`, yuquan `lagpat_broad`; variants excluded); `load_subject_propagation_events` (events + names); `load_subject_coords(ds, subj, names) → coords_array_in_requested_order / mapped_mask / coord_space`; and the accepted-axis source_core / sink_core read from `skeleton_geometry/per_subject/<ds>_<subj>.json`. Add the two fail-loud asserts: `coords.shape == (len(names), 3)`, and `set(source_core) | set(sink_core) ⊆ set(channel_names)` (a core channel missing from the broad pool is a real contract break, `core_unmapped`, NOT a benign skip).

- [ ] **Step 2: Write failing test for `event_extent`** (`tests/test_topic4_event_extent_audit.py`)

```python
import numpy as np
from src.topic4_event_extent_audit import event_extent, matched_null_extent

def test_event_extent_axially_full_laterally_narrow():
    # 20 channels spread 0..24mm along axis, +-1mm off -> fills axis, narrow lateral
    along = np.linspace(0, 24, 20); off = np.array([-1, 1] * 10)
    e = event_extent(along, off, axis_length=24.0)
    # p5->p95 spans 90% of a uniform full-coverage run -> ceiling ~0.90 (>> the 0.225 segment
    # case); p5-p95 is deliberately outlier-robust, so the "fills" threshold is 0.85, not 0.9.
    assert e["axial_fraction"] > 0.85         # fills the axis
    assert e["lateral_ratio"] < 0.2           # narrow sideways

def test_event_extent_axial_segment():
    # channels only over 0..6mm of a 24mm axis -> covers a SEGMENT
    along = np.linspace(0, 6, 12); off = np.array([-1, 1] * 6)
    e = event_extent(along, off, axis_length=24.0)
    assert e["axial_fraction"] < 0.35         # only a segment of the axis

def test_matched_null_reports_three_modes_and_respects_shaft_counts():
    rng = np.random.default_rng(0)
    along_all = np.linspace(0, 24, 12); off_all = np.tile([-1.0, 1.0], 6)
    shaft = np.array(["A"] * 6 + ["B"] * 6, object)   # 6 on each shaft
    rate = np.r_[np.full(6, 3.0), np.full(6, 1.0)]     # shaft A participates more
    out = matched_null_extent(along_all, off_all, n_part=5, axis_length=24.0,
                              n_draw=50, rng=rng, shaft=shaft,
                              shaft_counts={"A": 4, "B": 1}, rate=rate)
    assert set(out) == {"uniform", "rate", "shaft_matched"}     # all three layers present
    # shaft_matched draws 4 from A + 1 from B every time -> n_part=5 honored, never borrows
    assert len(out["shaft_matched"]["axial_fraction"]) == 50
    # a shaft asking for more than its eligible pool yields NO valid draws (skipped, not borrowed)
    bad = matched_null_extent(along_all, off_all, n_part=9, axis_length=24.0, n_draw=20,
                              rng=rng, shaft=shaft, shaft_counts={"A": 9, "B": 0})
    assert len(bad["shaft_matched"]["axial_fraction"]) == 0
```

- [ ] **Step 3: Run, verify fail** (`ImportError`).

- [ ] **Step 4: Implement `event_extent` + `matched_null_extent`** in `src/topic4_event_extent_audit.py`:

```python
import numpy as np

def event_extent(along, off, axis_length):
    along = np.asarray(along, float); off = np.asarray(off, float)
    axial = float(np.percentile(along, 95) - np.percentile(along, 5))
    lateral = float(np.percentile(off, 95) - np.percentile(off, 5))
    return dict(axial_span=axial, lateral_span=lateral,
                axial_fraction=axial / max(axis_length, 1e-9),
                lateral_ratio=lateral / max(axial, 1e-9))

def matched_null_extent(along_all, off_all, n_part, axis_length, n_draw, rng,
                        *, shaft=None, shaft_counts=None, rate=None):
    """Null distribution of event_extent over n_draw random size-n_part subsets of a
    subject's eligible (coord-mapped) channels, under up to three matching modes — to test
    whether an event's observed confinement is below random same-n_part sampling:
      uniform      : same-subject, same n_part, uniform draw (always returned)
      rate         : participation-rate-weighted draw (returned iff `rate` given)
      shaft_matched: draw shaft_counts[s] channels from each shaft s (returned iff
                     `shaft` + `shaft_counts` given; controls the implant-sampling confound).
    Returns {mode: {axial_fraction_med, lateral_ratio_med, axial_fraction[], lateral_ratio[]}}.
    shaft_matched SKIPS (records, never borrows cross-shaft) a draw where a shaft lacks
    enough eligible channels, so the shaft null can't silently relax the matching."""
    along_all = np.asarray(along_all, float); off_all = np.asarray(off_all, float)
    idx_all = np.arange(len(along_all)); k = min(n_part, len(idx_all))
    out = {}

    def _draw_metrics(draw_fn):
        af, lr = [], []
        for _ in range(n_draw):
            pick = draw_fn()
            if pick is None or len(pick) < 2:
                continue
            e = event_extent(along_all[pick], off_all[pick], axis_length)
            af.append(e["axial_fraction"]); lr.append(e["lateral_ratio"])
        return dict(axial_fraction_med=float(np.median(af)) if af else float("nan"),
                    lateral_ratio_med=float(np.median(lr)) if lr else float("nan"),
                    axial_fraction=af, lateral_ratio=lr)

    out["uniform"] = _draw_metrics(lambda: rng.choice(idx_all, size=k, replace=False))
    if rate is not None:
        p = np.asarray(rate, float); p = p / p.sum()
        out["rate"] = _draw_metrics(lambda: rng.choice(idx_all, size=k, replace=False, p=p))
    if shaft is not None and shaft_counts is not None:
        shaft = np.asarray(shaft, object)
        by_shaft = {s: idx_all[shaft == s] for s in shaft_counts}
        def _shaft_draw():
            picks = []
            for s, c in shaft_counts.items():
                pool = by_shaft.get(s, np.array([], int))
                if len(pool) < c:
                    return None  # not enough eligible on this shaft -> skip (don't borrow)
                picks.append(rng.choice(pool, size=c, replace=False))
            return np.concatenate(picks) if picks else None
        out["shaft_matched"] = _draw_metrics(_shaft_draw)
    return out
```

- [ ] **Step 5: Run, verify pass** (the pure-logic suite: `event_extent`, the 3-mode / shaft-count `matched_null_extent`, `event_shaft_counts`, and the 5 `cohort_verdict` branches — 9 tests).

- [ ] **Step 6: Write the runner** `scripts/run_topic4_event_extent_audit.py`: for each eligibility-spine subject (accepted skeleton axis), build the axis frame from the accepted source_core/sink_core centroids (per the data contract) and project every broad-pool channel coord → (along, off). For each real lagPat event take its participating coord-mapped channels (`n_part ≥ 5`), compute `event_extent` over their (along, off); also `matched_null_extent` with n_draw ≥ 200 over the subject's eligible coord-mapped channels — passing `rate=` (per-channel participation fraction) AND `shaft=`/`shaft_counts=` (the event's per-shaft channel counts via `parse_shaft`) so all three null modes are produced. Aggregate per subject: median + IQR of axial_fraction / lateral_ratio, and the per-event observed−null gap under EACH mode (carry the `shaft_matched` gap as the primary/conservative one). Write the per-subject diagnostics columns from the data contract.

- [ ] **Step 7: Run it** on the broad pool; write `per_subject.csv` + `cohort_summary.json`.

- [ ] **Step 8: Figure + README** — one figure (per CLAUDE.md §7 / figure-discipline): panel A = per-subject distribution of `axial_fraction` (do events fill the axis? observed vs matched-null); panel B = `lateral_ratio` (how narrow sideways? observed vs null). `figures/README.md` (中文, per AGENTS.md) explaining the two panels + 关注点.

- [ ] **Step 9: Interpret — CONTACT-space methodological constraint, NOT a model stop-gate (REVISED 2026-06-19 after review)** — write the numbers + this framing into `cohort_summary.json`. Define over the eligibility-spine subjects: `AF` = cohort median of (per-subject median `axial_fraction`); `LR` = cohort median of (per-subject median `lateral_ratio`); `Δ` = paired subject-level `median(per-subject observed − shaft_matched-null median)` with a bootstrap 95% CI + Wilcoxon. **These describe the SEEG CONTACT footprint (sampled, ever-participating channels, denominator = recruited-territory axial extent) — they do NOT measure tissue-space self-limit.** Read-out:
  - **`AF ≥ 0.75` AND `Δ` ≈ 0 (CI includes 0, or excludes it but |Δ| negligible) →** contact footprint is axially LONG and ≈ random shaft-matched sampling → footprint is sampling-dominated; the data give NO strong evidence for a short on-axis segment IN CONTACT SPACE, and CANNOT decide tissue-space self-limit either way. A methodological constraint, NOT a negation of tissue self-limit. **This is what Task 0 found (EXECUTED 2026-06-19, n=23: AF=0.915, LR=0.561; axial Δ CI (−0.044,−0.003) but |Δ|≈0.03 negligible & Wilcoxon p=0.056; lateral Δ CI includes 0; formal verdict INCONCLUSIVE).**
  - **`AF ≤ 0.5` AND axial `Δ` clearly below null →** contact footprint short AND more confined than sampling → only THEN is there contact-space evidence consistent with an on-axis segment (still not proof of tissue self-limit; would need the virtual-SEEG forward model). (Not observed.)

  **What Task 0 SETS, not decides:** it does NOT decide whether Tasks 1–7 run — they do; the self-limit question lives in tissue / full-field and is judged at **Task 7 layer 1**. Task 0 CALIBRATES **Task 7 layer 2**: the model's virtual-SEEG projection must reproduce THIS contact-space AF/LR (axially long, ≈ shaft-null), NOT a short segment. A model whose contact footprint is short would MISmatch the data.

  **口径 / placement (REVISED):** Task 0 is a methodological constraint / model-data bridge result → archive under **Topic 4 observation/model-data bridge or a methodological supplement**, NOT a Topic 3 main result. If referenced in Topic 3, ONLY as a "SEEG spatial footprint is strongly constrained by shaft/contact sampling" caveat/control — never as a SOZ / spatial-localization finding. Commit Task 0 before starting any model task.

---

### Task 1: Extract the membrane update into a pure helper (current path; bit-parity anchor)

> **All Tasks 1–7 run INSIDE `.worktrees/topic4-m1` (canonical base; see Global Constraints tree strategy). `cd` there first; paths below are relative to the worktree root. Re-bless the worktree's own `engine_versions.json`.**

**Files:**
- Modify: `src/snn_engine/kick_probe.py` (extract the V-update lines into a module-level pure function; call it in the loop)
- Test: `tests/test_snn_shunting.py`

**Interfaces:**
- Produces: `membrane_step(V, I_E, I_I, decay_V, *, shunt_gaba=False, e_gaba=11.0, g_gaba_scale=0.0) -> np.ndarray`. With `shunt_gaba=False` returns EXACTLY `(I_E - I_I) + (V - (I_E - I_I)) * decay_V` (the current code).

- [ ] **Step 1: Capture the pre-edit baseline SHA** (if not already `da5fc18c…`)

```bash
cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-m1
python3 - <<'PY'
import sys, hashlib, numpy as np; sys.path.insert(0, "src/snn_engine")
from params import Params; from connectivity import place_neurons
from connectivity_rot import build_connectivity_rot; from kick_probe import simulate_kick
p = Params(L=6.0, density=100.0, T=300.0, dt=0.1, nu_ext_ratio=0.6, seed=1)
rng = np.random.default_rng(1); pos, labels, NE, NI = place_neurons(p, rng)
net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0)
net["rng"] = np.random.default_rng(1)
res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE+NI, 18.0))
print("BASELINE_SHA", hashlib.sha1(res["E_spk_bool"].tobytes()).hexdigest()[:16])
PY
```
Expected: `da5fc18c27d5340a`. Record it as `BASELINE_SHA` in the test file.

- [ ] **Step 2: Write the failing test** (`tests/test_snn_shunting.py`)

```python
import hashlib, os, sys
import numpy as np
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))
from params import Params
from connectivity import place_neurons
from connectivity_rot import build_connectivity_rot
from kick_probe import simulate_kick, membrane_step

BASELINE_SHA = "da5fc18c27d5340a"

def test_membrane_step_current_path_matches_formula():
    V = np.array([12.0, 15.0]); I_E = np.array([20.0, 5.0]); I_I = np.array([4.0, 1.0])
    decay = np.array([0.99, 0.99])
    I_net = I_E - I_I
    expected = I_net + (V - I_net) * decay
    np.testing.assert_allclose(membrane_step(V, I_E, I_I, decay), expected)

def _sha():
    p = Params(L=6.0, density=100.0, T=300.0, dt=0.1, nu_ext_ratio=0.6, seed=1)
    rng = np.random.default_rng(1); pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0)
    net["rng"] = np.random.default_rng(1)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE+NI, 18.0))
    return hashlib.sha1(res["E_spk_bool"].tobytes()).hexdigest()[:16]

def test_extraction_preserves_bit_parity():
    assert _sha() == BASELINE_SHA
```

- [ ] **Step 3: Run, verify fail** — `python3 -m pytest tests/test_snn_shunting.py -q` → FAIL (`cannot import name 'membrane_step'`).

- [ ] **Step 4: Add the pure helper + call it** in `kick_probe.py`. Add near the top (module level):

```python
def membrane_step(V, I_E, I_I, decay_V, *, shunt_gaba=False, e_gaba=11.0, g_gaba_scale=0.0):
    """One LIF membrane update. Default (shunt_gaba=False) = current-based LIF, BIT-IDENTICAL
    to the pre-2026-06-19 engine: V_inf = I_E - I_I; V -> V_inf + (V - V_inf)*decay_V.

    shunt_gaba=True = conductance-based SHUNTING inhibition: GABA is a conductance
    g_I = g_gaba_scale*max(I_I,0) pulling V toward the reversal e_gaba, so it gates spike
    initiation regardless of excitatory drive magnitude:
        V_inf = (I_E + g_I*e_gaba) / (1 + g_I);  V -> V_inf + (V - V_inf)*decay_V**(1+g_I).
    (decay_V**(1+g_I) == exp(-dt*(1+g_I)/tau_m) since decay_V = exp(-dt/tau_m): shunting also
    shortens the effective membrane time constant.)"""
    if not shunt_gaba:
        I_net = I_E - I_I
        return I_net + (V - I_net) * decay_V
    g_I = g_gaba_scale * np.maximum(I_I, 0.0)
    V_inf = (I_E + g_I * e_gaba) / (1.0 + g_I)
    return V_inf + (V - V_inf) * decay_V ** (1.0 + g_I)
```

Replace the loop's V-update (`I_net = I_E - I_I` ... `Vtmp = I_net + (V - I_net) * decay_V`) with: keep `I_net`/`V_th_eff` logic for the `slow` branch, but in the `else` (slow=None) branch compute `Vtmp = membrane_step(V, I_E, I_I, decay_V)` (default args → current path). Leave the `slow is not None` branch untouched.

- [ ] **Step 5: Run, verify pass** — both tests PASS (parity SHA unchanged).

- [ ] **Step 6: Run engine smoke** — `python3 -m pytest tests/ -k "snn or gate or step0" -q` → PASS (no regression).

- [ ] **Step 7: Commit** (no re-bless yet — kick_probe.py changed; re-bless after Task 3 once the flag exists and parity is re-confirmed).

```bash
git add src/snn_engine/kick_probe.py tests/test_snn_shunting.py
git commit -m "refactor(topic4 snn): extract membrane_step pure helper (current path, bit-parity)"
```

---

### Task 2: Conductance shunting in the membrane helper (default OFF)

**Files:**
- Modify: `src/snn_engine/kick_probe.py` (helper already supports `shunt_gaba`; verify behavior)
- Test: `tests/test_snn_shunting.py`

**Interfaces:** Consumes `membrane_step` from Task 1.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_snn_shunting.py`)

```python
def test_shunt_g0_reduces_to_leak_toward_drive():
    # g_gaba_scale=0 (or I_I=0) under shunting: V relaxes toward I_E with decay_V (no inhibition)
    V = np.array([12.0]); I_E = np.array([20.0]); I_I = np.array([5.0]); decay = np.array([0.9])
    out = membrane_step(V, I_E, I_I, decay, shunt_gaba=True, e_gaba=11.0, g_gaba_scale=0.0)
    np.testing.assert_allclose(out, I_E + (V - I_E) * decay)   # g_I=0 -> V_inf=I_E

def test_shunting_gates_high_drive_below_threshold():
    # KEY: a strongly-driven cell (I_E=30, well above V_th=18). Current-subtraction with I_I=10
    # settles at I_E-I_I=20 > V_th (drive wins -> fires). Shunting with g_I=10 settles at
    # (30+10*11)/11 = 12.7 < V_th (clamped toward e_gaba -> spike-initiation gated).
    decay = np.array([np.exp(-0.1 / 20.0)])
    Vc = np.array([11.0]); Vs = np.array([11.0])
    for _ in range(3000):
        Vc = membrane_step(Vc, np.array([30.0]), np.array([10.0]), decay, shunt_gaba=False)
        Vs = membrane_step(Vs, np.array([30.0]), np.array([10.0]), decay,
                           shunt_gaba=True, e_gaba=11.0, g_gaba_scale=1.0)
    assert Vc[0] > 18.0    # current-subtraction: drive overwhelms inhibition
    assert Vs[0] < 18.0    # shunting: clamped below threshold regardless of drive

def test_shunting_changes_engine_spikes():
    # full-engine: shunt on (via runner param in Task 3); here just the helper diverges from current
    V = np.array([15.0]); I_E = np.array([22.0]); I_I = np.array([6.0]); decay = np.array([0.9])
    cur = membrane_step(V, I_E, I_I, decay, shunt_gaba=False)
    sh = membrane_step(V, I_E, I_I, decay, shunt_gaba=True, e_gaba=11.0, g_gaba_scale=0.5)
    assert not np.allclose(cur, sh)
```

- [ ] **Step 2: Run, verify fail/pass** — the helper from Task 1 already implements shunting, so these PASS immediately. To confirm they DISCRIMINATE (not vacuous), temporarily set `g_gaba_scale=0` in `test_shunting_gates_high_drive_below_threshold` and confirm it FAILS (Vs→20), then restore. Document this check in a comment.

- [ ] **Step 3: Commit**

```bash
git add tests/test_snn_shunting.py
git commit -m "test(topic4 snn): conductance shunting gates high-drive cells (vs current-subtraction)"
```

---

### Task 3: Wire shunting into simulate_kick + runner CLI + re-bless

**Files:**
- Modify: `src/snn_engine/params.py` (add `E_gaba` default)
- Modify: `src/snn_engine/kick_probe.py` (`simulate_kick` gains `shunt_gaba=False, e_gaba=None, g_gaba_scale=0.0`; pass to `membrane_step` in the slow=None branch)
- Modify: `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py` (CLI `--shunt-gaba`, `--e-gaba`, `--g-gaba-scale`; pass to `simulate_kick`; config provenance)
- Modify (after parity passes): `results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json` (re-bless)
- Test: `tests/test_snn_shunting.py`, `tests/test_step0_selflimit.py`

**Interfaces:**
- Produces: `simulate_kick(..., shunt_gaba=False, e_gaba=None, g_gaba_scale=0.0)` — when `shunt_gaba=False` (default) the V-update is the current path; `e_gaba=None` defaults to `p.V_reset`.

- [ ] **Step 1: Add `E_gaba` to Params** — in `src/snn_engine/params.py` add `E_gaba: float = 11.0  # mV GABA reversal (= V_reset; shunting/near-rest)`. (Not used unless shunt_gaba on; no effect on default path.)

- [ ] **Step 2: Failing test — default-off parity through simulate_kick** (append)

```python
def test_simulate_kick_shunt_off_is_bit_identical():
    p = Params(L=6.0, density=100.0, T=300.0, dt=0.1, nu_ext_ratio=0.6, seed=1)
    rng = np.random.default_rng(1); pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0)
    net["rng"] = np.random.default_rng(1)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE+NI, 18.0),
                        shunt_gaba=False)
    assert hashlib.sha1(res["E_spk_bool"].tobytes()).hexdigest()[:16] == BASELINE_SHA

def test_simulate_kick_shunt_on_changes_spikes():
    p = Params(L=6.0, density=100.0, T=300.0, dt=0.1, nu_ext_ratio=0.6, seed=1)
    rng = np.random.default_rng(1); pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0)
    net["rng"] = np.random.default_rng(1)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE+NI, 18.0),
                        shunt_gaba=True, e_gaba=11.0, g_gaba_scale=0.5)
    assert hashlib.sha1(res["E_spk_bool"].tobytes()).hexdigest()[:16] != BASELINE_SHA
```

- [ ] **Step 3: Run, verify fail** — FAIL (`simulate_kick() got unexpected keyword 'shunt_gaba'`).

- [ ] **Step 4: Add params to `simulate_kick`** — signature `shunt_gaba=False, e_gaba=None, g_gaba_scale=0.0`. After the signature, `e_gaba = p.V_reset if e_gaba is None else e_gaba`. In the `slow is None` branch replace `Vtmp = membrane_step(V, I_E, I_I, decay_V)` with `Vtmp = membrane_step(V, I_E, I_I, decay_V, shunt_gaba=shunt_gaba, e_gaba=e_gaba, g_gaba_scale=g_gaba_scale)`. (decay_V is per-neuron already.)

- [ ] **Step 5: Run, verify pass** — both PASS (off = SHA, on ≠ SHA).

- [ ] **Step 6: Runner CLI** — in `run_sef_hfo_snn_cm_spontaneous_readout.py` add args:

```python
    ap.add_argument("--shunt-gaba", action="store_true",
                    help="M2 faithful: conductance-based shunting inhibition (default OFF=current-LIF).")
    ap.add_argument("--e-gaba", type=float, default=None, help="GABA reversal (mV); None=V_reset.")
    ap.add_argument("--g-gaba-scale", type=float, default=0.0,
                    help="GABA conductance scale (I_I -> g_I); required >0 with --shunt-gaba.")
```
Pass to the `simulate_kick(...)` call: `shunt_gaba=a.shunt_gaba, e_gaba=a.e_gaba, g_gaba_scale=a.g_gaba_scale`. Add `shunt_gaba=a.shunt_gaba, e_gaba=a.e_gaba, g_gaba_scale=a.g_gaba_scale` to the `config=dict(...)`.

- [ ] **Step 7: CLI + COEXISTENCE GUARD (P0)** (extend `tests/test_step0_selflimit.py`): assert `--shunt-gaba` in `--help`. **Then run a tiny combo smoke and assert all three legs coexist in the readout `config`:**

```bash
python3 scripts/run_sef_hfo_snn_cm_spontaneous_readout.py --lesion twoend_equal \
  --core-mean 17.5 --core-std 1.0 --sep-frac 0.7 --drive 0.6 --L 20 --T 500 \
  --shunt-gaba --g-gaba-scale 0.5 --ee-std-u 0.2 --ee-std-tau-ms 200 \
  --gate-scale 0.5 --l-gate 1.5 --c-gate 150 --tag coexist_smoke --out /tmp/m2_coexist
python3 -c "import json; c=json.load(open('/tmp/m2_coexist/readout_coexist_smoke.json'))['config']; \
assert all(k in c for k in ['shunt_gaba','g_gaba_scale','ee_std_u','gate_scale','l_gate']), c; print('coexist OK', {k:c[k] for k in ['shunt_gaba','g_gaba_scale','ee_std_u','gate_scale']})"
```
Expected: `coexist OK {'shunt_gaba': True, 'g_gaba_scale': 0.5, 'ee_std_u': 0.2, 'gate_scale': 0.5}` — shunting + recovery + gate all recorded, none silently dropped. No crash; engine guard passes after re-bless (Step 8).

- [ ] **Step 8: Re-bless + commit**

```bash
python3 - <<'PY'
import json, hashlib; from pathlib import Path
F="results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json"; d=json.load(open(F))
for k in ("src/snn_engine/kick_probe.py","src/snn_engine/params.py"):
    d[k]=hashlib.sha256(Path(k).read_bytes()).hexdigest()
json.dump(d,open(F,"w"),indent=2); print("re-blessed")
PY
git add src/snn_engine/kick_probe.py src/snn_engine/params.py scripts/run_sef_hfo_snn_cm_spontaneous_readout.py tests/test_snn_shunting.py tests/test_step0_selflimit.py results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json
git commit -m "feat(topic4 M2): conductance shunting inhibition in cm-SNN (default off, bit-parity, re-bless)"
```

---

### Task 4: Wide E→I recruitment gate in connectivity_rot (default OFF)

**Files:**
- Modify: `src/snn_engine/connectivity_rot.py` (mirror the existing I→E gate, but on the E→I / AMPA-to-I branch)
- Modify: `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py` (CLI `--ei-gate-scale`, `--l-ei-gate`, `--c-ei-gate`; into `_gate_kw`)
- Modify (after parity): `engine_versions.json` (re-bless connectivity_rot.py)
- Test: `tests/test_snn_ei_recruit_gate.py`

**Interfaces:**
- Produces: `build_connectivity_rot(..., ei_gate_scale=0.0, l_ei_gate=None, C_ei_gate=None)`. When `ei_gate_scale>0`, for each I TARGET add `C_ei_gate` extra E sources within `l_ei_gate` (wide), weight `ei_gate_scale*w_IE`, so the front recruits I AHEAD. Default 0 → no extra edges → bit-parity.

- [ ] **Step 1: Failing tests** (`tests/test_snn_ei_recruit_gate.py`) — mirror `tests/test_snn_ie_gate.py` structure: (a) `ei_gate_scale=0` default → spike SHA `da5fc18c`; (b) `ei_gate_scale>0` → extra AMPA edges onto **I targets only** (rows ≥ NE in `ampa_by_delay`), E-target AMPA edges unchanged; (c) wider `l_ei_gate` → larger mean E→I edge distance; (d) `ei_gate_scale>0` requires `l_ei_gate` and `C_ei_gate` (ValueError).

```python
# helper mirrors tests/test_snn_ie_gate.py but counts AMPA (ampa_by_delay) edges by target E/I
def _ei_ampa_counts(net, NE):
    e=i=0
    for m in net["ampa_by_delay"]:
        coo=m.tocoo(); e+=int((coo.row<NE).sum()); i+=int((coo.row>=NE).sum())
    return e,i
def test_ei_gate_on_adds_ampa_edges_to_I_targets_only():
    _,net0,NE,NI,_=_net(); _,net1,_,_,_=_net(ei_gate_scale=0.5,l_ei_gate=1.5,C_ei_gate=100)
    e0,i0=_ei_ampa_counts(net0,NE); e1,i1=_ei_ampa_counts(net1,NE)
    assert i1>i0 and e1==e0   # adds E->I (I targets) only; E->E (E targets) unchanged
```

- [ ] **Step 2: Run, verify fail** (`unexpected keyword 'ei_gate_scale'`).

- [ ] **Step 3: Implement** — in `build_connectivity_rot`: add params; pre-loop guard (`ei_gate_scale>0 requires l_ei_gate and C_ei_gate`); in the AMPA block, for **I targets** (`not a_is_E`), after the existing E→I append, add (gated on `ei_gate_scale>0`):

```python
        if (not a_is_E) and ei_gate_scale > 0.0:
            ce = _sample_partners(pt, posE, C_ei_gate, l_ei_gate, 0.0, rng, self_local=None)
            if ce.size:
                de = np.linalg.norm(posE[ce] - pt, axis=1)
                a_rows.append(np.full(ce.size, i)); a_cols.append(ce)
                a_w.append(np.full(ce.size, (w_IE * ei_gate_scale) * jump_ampa[i]))
                a_dly.append(p.tau0 + de * inv_vdt)
```
(`jump_ampa`, `w_IE`, `posE`, `inv_vdt`, `p.tau0` are in scope; gated on `ei_gate_scale>0` so default = no extra rng → parity.)

- [ ] **Step 4: Run, verify pass** (4 tests).

- [ ] **Step 5: Runner CLI** — add `--ei-gate-scale/--l-ei-gate/--c-ei-gate`; extend `_gate_kw` with `ei_gate_scale=a.ei_gate_scale, l_ei_gate=a.l_ei_gate, C_ei_gate=a.c_ei_gate`; add to config.

- [ ] **Step 6: Re-bless connectivity_rot.py + smoke + commit** (same re-bless snippet for `connectivity_rot.py`; tiny `--ei-gate-scale 0.5 --l-ei-gate 1.5 --c-ei-gate 100` run, no crash).

```bash
git commit -m "feat(topic4 M2): wide E->I ahead-of-front recruitment gate (default off, bit-parity, re-bless)"
```

---

### Task 4.5: Mechanism diagnostics — I-spike recording + ahead-of-front lead + clamp check (P1)

**Why:** result metrics (`reach_axis`) alone can't tell WHY a gate works/fails. We must directly show (a) inhibitory cells fire AHEAD of the E front (not synchronous/late), and (b) shunting actually clamps the high-drive AXIAL E cells below threshold (not just suppresses low-drive lateral cells, Q3). Without this, a gate "success" could be global suppression and a "failure" could be ahead-not-implemented vs shunt-too-weak — indistinguishable.

**Files:**
- Modify: `src/snn_engine/kick_probe.py` (record `I_spk_bool` when `dump_i_spikes=True`; optional peak-frame `I_E`/`I_I` snapshot when `dump_drive=True`; both parity-safe READOUT-only, no dynamics change)
- Modify: `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py` (CLI `--dump-i-spikes`, `--dump-drive`; pass through)
- Create: `src/topic4_gate_diag.py` (pure analyses)
- Test: `tests/test_topic4_gate_diag.py`

**Interfaces:**
- Produces: `front_lead_by_axis(E_spk_bool, I_spk_bool, along_E, along_I, n_bins, dt) -> dict(bin_along, t_E_onset_ms, t_I_onset_ms, I_lead_ms)` — per axial bin, first-E-spike vs first-I-spike time; `I_lead_ms = t_E_onset - t_I_onset` (>0 = I ahead of the E front).
- Produces: `clamp_check(I_E, I_I, along_E, axis_unit, e_gaba, g_gaba_scale, v_th) -> dict` — for the front (high-drive axial) E cells, current-path target `I_E - I_I` vs shunting target `(I_E + g_I*e_gaba)/(1+g_I)`; reports fraction of axial-front E cells whose shunting target < v_th (gated) but current target ≥ v_th (would fire). This is the direct Q2/Q3 mechanism evidence.

- [ ] **Step 1: Failing test — parity-safe I-spike recording + the lead metric** (`tests/test_topic4_gate_diag.py`)

```python
import numpy as np
from src.topic4_gate_diag import front_lead_by_axis, clamp_check

def test_front_lead_I_ahead_of_E():
    # 3 axial bins; in bin 2 I fires at t-index 5, E front arrives at t-index 9 -> I leads 4 steps
    T, NE, NI = 20, 6, 6; dt = 0.5
    E = np.zeros((T, NE), bool); I = np.zeros((T, NI), bool)
    along_E = np.array([0,0,5,5,10,10.]); along_I = np.array([0,0,5,5,10,10.])
    I[5, 2] = True      # I in middle bin fires early
    E[9, 2] = True      # E front reaches middle bin later
    out = front_lead_by_axis(E, I, along_E, along_I, n_bins=3, dt=dt)
    j = np.argmin(np.abs(np.array(out["bin_along"]) - 5.0))
    assert out["I_lead_ms"][j] > 0   # I ahead of E in that bin

def test_clamp_check_shunting_gates_axial_front():
    # axial-front E cell: I_E=30 (strong drive), I_I=10. current target 20>=v_th=18 (fires);
    # shunting target (30+10*11)/11=12.7<18 (gated).
    I_E = np.array([30.0]); I_I = np.array([10.0]); along_E = np.array([12.0])
    out = clamp_check(I_E, I_I, along_E, np.array([1.0,0.0]), e_gaba=11.0, g_gaba_scale=1.0, v_th=18.0)
    assert out["frac_axial_gated_by_shunt"] == 1.0
```

- [ ] **Step 2: Run, verify fail** (ImportError).

- [ ] **Step 3: Implement `src/topic4_gate_diag.py`** — `front_lead_by_axis`: bin cells by `along`; per bin take first time-index any E (resp I) cell fires; convert to ms (`*dt`); `I_lead_ms = t_E - t_I` (nan if either never fires). `clamp_check`: select axial-front E cells (top quartile of `along`, i.e. ahead of centroid); `g_I = g_gaba_scale*max(I_I,0)`; `cur = I_E - I_I`; `sh = (I_E + g_I*e_gaba)/(1+g_I)`; `frac_axial_gated_by_shunt = mean((sh < v_th) & (cur >= v_th))`.

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Add `I_spk_bool` recording to `simulate_kick`** — signature `dump_i_spikes=False, dump_drive=False`. When `dump_i_spikes`, allocate `I_spk_bool=np.zeros((nsteps, NI), bool)` and set `I_spk_bool[t]=spk[NE:]` in the record block; add to the return dict. When `dump_drive`, store `I_E`/`I_I` at the peak-active frame. **Recording only — no dynamics touched → bit-parity preserved.** Test: `dump_i_spikes=False` → spike SHA still `da5fc18c`; `dump_i_spikes=True` → SHA unchanged AND `"I_spk_bool"` present.

- [ ] **Step 6: Runner CLI `--dump-i-spikes`/`--dump-drive`** + pass to `simulate_kick`; re-bless (kick_probe.py changed again); commit.

---

### Task 5: Operating-point sanity — shunting AND gate-on both change the E/I balance (PILOT-FIRST gate, 4-cell)

**Files:**
- Output: `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/m2_shunt_opsanity/`

**Interfaces:** Consumes the runner (Tasks 3–4).

- [ ] **Step 1: Run the base workpoint with shunting on, NO gate**, scanning `g_gaba_scale` (shunting strength must be calibrated so it neither kills ignition nor leaves rest noisy):

```bash
cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-m1
OUT=results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/m2_shunt_opsanity; mkdir -p $OUT
for G in 0.0 0.25 0.5 1.0; do
  python3 scripts/run_sef_hfo_snn_cm_spontaneous_readout.py --lesion twoend_equal \
    --core-mean 17.5 --core-std 1.0 --sep-frac 0.7 --drive 0.6 --L 20 --T 8000 \
    --event-bar-mode prefix_peak --cal-prefix-ms 3000 --dump-fullfield \
    --shunt-gaba --g-gaba-scale $G --tag opsan_g$G --out $OUT 2>&1 | tail -2
done
```

- [ ] **Step 2: GATE — inspect, do NOT proceed unless a `g_gaba_scale>0` cell still has discrete events with a quiet inter-event baseline** (i.e. `events>0`, not tonic, not silent). If shunting at all `g>0` kills ignition → first re-tune drive UP at that shunting level (e.g. `--drive 0.7/0.8`) to restore spontaneity; pick the `(g_gaba_scale, drive)` with discrete events as the shunting workpoint. **Record the chosen `(g_gaba_scale, drive)` — Task 6 uses it.** If no `(g, drive)` gives discrete events, STOP and report (shunting destabilizes this substrate; that itself is a finding).

- [ ] **Step 3: 4-cell gate-on sanity (P1 — the gates ALSO change the balance).** At the Task-5 shunting workpoint `(g*, drive*)`, run four cells and tabulate, so Task 6 can tell a real full-field self-limit apart from "killed ignition / went tonic":

```text
(i)   shunting-only          (no gate)
(ii)  + I->E veto only       (--gate-scale .. --l-gate .. --c-gate ..)
(iii) + E->I recruit only    (--ei-gate-scale .. --l-ei-gate .. --c-ei-gate ..)
(iv)  + both gates
```
For each record `n_events`, `true_inter_event_floor` (quiet-baseline active fraction), `n_clean_forward`/`n_clean_reverse` (bidirectional readout preserved?), hidden core source counts (neg/pos clean). **Interpretation rule for Task 6:** an event-count drop with a RISING `true_inter_event_floor` = quasi-tonic (gate failed); a count drop toward silence = killed ignition (gate failed); only "events persist, return, bidirectional preserved, AND `reach_axis` is BOUNDED / finite (not growing with L, not boundary-filling)" counts as a full-field self-limit (Task 6 layer 1). Record this 4-cell table; it is the denominator for the Task 6 verdict.

---

### Task 6: Faithful gate pilot — shunting + ahead-recruitment + I→E gate (+ recovery in worktree)

**Files:**
- (No porting / no `cp` — Tasks 1–4 were already done INSIDE `.worktrees/topic4-m1` per Global Constraints; the worktree already carries E→E recovery `--ee-std-u`, the I→E gate, and the re-blessed `engine_versions.json`.)
- Output (worktree): `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/m2_shunt_gate_pilot/`.

- [ ] **Step 1: (no porting — all prior tasks were IN the worktree per Global Constraints).** Confirm `tests/test_snn_shunting.py tests/test_snn_ei_recruit_gate.py tests/test_topic4_gate_diag.py` pass in the worktree and its `engine_versions.json` is re-blessed. The worktree already carries E→E recovery (`ee_std_u`), so the full **shunting + I→E veto + E→I recruit + recovery** combo is available here without touching main.

- [ ] **Step 2: Pilot grid** at the Task-5 shunting workpoint `(g*, drive*)`, twoend_equal focal (gives bidirectional), `--dump-fullfield --dump-fwd-rev-reps`, T=8000:
  - control: `--shunt-gaba --g-gaba-scale g*` , gate OFF (I→E gate + E→I recruit both off) — the shunting-only base.
  - combo: `+ --gate-scale {0.4,0.6} --l-gate 1.5 --c-gate 150 --ei-gate-scale {0.4,0.6} --l-ei-gate 1.5 --c-ei-gate 150` (I→E veto AND ahead E→I recruitment both wide).
  - L20 first; then L32 (`--prune-radius 4.3`) for the best L20 combo (L-invariance).
  - In the worktree, add `--ee-std-u 0.2 --ee-std-tau-ms 200` (full shunting+gate+recovery combo).
  - Optional finer readout on the best combo: `--nc 9 --pitch 2 --k-dir 2` (resolves the radially-narrow events).

- [ ] **Step 3: Acceptance — LAYER 1 (full-field / tissue self-limit).** From the fullfield JSONs (`event_field_geometry` fields), per readable event (`n_part≥7`, or ≥5 at finer readout) measure the model's TRUE spatial extent: `reach_axis_mm`, `r95_mm`, `edge_margin_mm`, clean fwd/rev. **Layer-1 PASS = the event is spatially FINITE: extent BOUNDED and L-INVARIANT (reach_axis / r95 at L32 ≈ at L20, NOT growing to fill the larger sheet), `edge_margin_mm > 0` (does NOT touch the boundary), a quiet inter-event baseline (not tonic), and not a full-sheet / seizure-like global recruitment** — cross-check the Task-5 Step-3 4-cell sanity (count drop with rising inter-event floor = tonic, not self-limit). **The target is FINITE + bounded, NOT "short / well below the focus distance"** — the old contact-segment target is DROPPED (see Claim discipline; Task 0 shows the real contact footprint is long + sampling-dominated).

- [ ] **Step 3b: Acceptance — LAYER 2 (virtual-SEEG contact footprint STATISTICALLY CONSISTENT with Task 0).** Take the combo cell's virtual-SEEG observation artifact (the cm-SNN observation layer emits `*_lagPat_withFreqCent.npz` + `_montage.json` contact coords; reuse `scripts/run_model_contact_plane_readout.py::build_model_record` to load `eventsBool` / `chnNames` / contact coords). Project model contact coords onto the MODEL source→sink axis; run the Task-0 PURE metrics (`src.topic4_event_extent_audit.event_extent` + shaft_matched `matched_null_extent`, same recruited-territory denominator) to get the model's PER-EVENT contact AF and LR distributions. **Layer-2 PASS = the model's per-event AF (and LR) distribution is STATISTICALLY CONSISTENT with Task 0's real per-event distribution** — reference persisted in `cohort_summary.json::reference_distribution.{axial_fraction,lateral_ratio}`. Test = two-sample (KS or Mann–Whitney) on model-vs-real; **PASS = NOT rejected (`p > α`)**. **LOOSENED / permissive: lenient α (0.01)** so layer 2 fails ONLY when the model footprint is CLEARLY different from the data (e.g. a short-segment model with AF ≪ 0.9). Report alongside a descriptive overlap / effect-size (distribution overlap coefficient or median difference) so a small-n non-rejection is not over-read. **Honest caveat: non-rejection = "can't distinguish", NOT proof of identity** (absence of evidence) — the overlap/effect-size is the companion. A short-AF model (≪ Task 0) is rejected → layer-2 FAIL.

- [ ] **Step 4: Mechanism diagnostics (P1, MANDATORY — run on the combo cell with `--dump-i-spikes --dump-drive`)**:
  - **Ahead-of-front (Q1):** `front_lead_by_axis` (Task 4.5) → `I_lead_ms` per axial bin AHEAD of the front. Report the distribution. If `I_lead_ms ≤ 0` (I synchronous/late) the gate is NOT a true ahead-of-front brake even if `reach_axis` shrank → say so (then any containment is global/local suppression, not front-pinning).
  - **Shunting clamps the AXIAL amplifier (Q2/Q3):** `clamp_check` (Task 4.5) → `frac_axial_gated_by_shunt` = fraction of high-drive axial-front E cells whose shunting target is below threshold while the current-subtraction target is above. >0 here is the direct evidence shunting gates the axial cells (not just lateral).
  - **Three cell groups (Q3):** split cells into (a) axial-leading E, (b) lateral E, (c) far-ahead I; report shunt-ON vs shunt-OFF `V_inf` (from dumped `I_E`/`I_I`), spike probability, first-spike latency for each. The Q3 question is answered only if the AXIAL-leading group is actually suppressed under shunting (not just the lateral group).
  - These three feed the Task 7 verdict; a `reach_axis` change WITHOUT ahead-lead + axial-clamp evidence is "suppression, mechanism unclear", NOT "ahead-of-front shunting brake works".

---

### Task 7: Verdict + result archive

**Files:**
- Create: `docs/archive/topic4/sef_hfo/m2_shunting_gate_result_<date>.md`

- [ ] **Step 0 (from Task 0 — REVISED): set the layer-2 target, do NOT stop the model.** Task 0 (executed 2026-06-19) found the real CONTACT footprint is axially LONG (~92%) and ≈ shaft-matched sampling (INCONCLUSIVE; sampling-dominated). This does NOT make the model unnecessary and is NOT "reframe to lateral". It SETS the layer-2 acceptance: the model's virtual-SEEG footprint must look the SAME (long + sampling-like). The load-bearing model question is the TWO-LAYER one (full-field finite self-limit AND virtual-SEEG reproduces Task 0 AF/LR), NOT "does the contact footprint shrink to a segment".
- [ ] **Step 1: Classify** the combo vs the Task-5 4-cell sanity + Task-6 Step-3 (layer 1) / Step-3b (layer 2) / Step-4 (mechanism):
  - **GATE WORKS (faithful, two-layer)** — LAYER 1 PASS (full-field events FINITE: reach_axis bounded + L-invariant + `edge_margin>0` + not tonic/full-sheet, NOT via killing ignition) **AND** LAYER 2 PASS (virtual-SEEG contact AF/LR consistent with Task 0) **AND** the Step-4 mechanism diagnostics support it (`I_lead_ms > 0` ahead-of-front AND `frac_axial_gated_by_shunt > 0` / axial-leading group clamped) → the M2 ahead-of-front shunting brake produces a tissue-FINITE event that still looks data-like after sampling. Proceed to a full sweep for all-5.
  - **FULL-FIELD FINITE BUT MISMATCHES SAMPLING** — Layer 1 PASS but Layer 2 FAIL (e.g. virtual-SEEG contact AF too short ≪ Task 0's ~0.9) → the model self-limits MORE than the observation layer shows; the gate is too strong / wrong scale → tune (gate width / shunt strength) before any verdict.
  - **MECHANISM UNCLEAR** — extent changes but `I_lead_ms ≤ 0` or axial group NOT clamped → global/local suppression, not front-pinning; do NOT claim the gate; investigate (timing / shunt strength).
  - **STILL FAILS (full-field not finite)** — Layer 1 FAIL even with shunting + ahead recruit (extent still grows with L / fills the sheet / goes tonic) AND the Step-4 diagnostics confirm the mechanism WAS instantiated (I ahead + axial cells clamped) → the faithful mechanism was tested and is insufficient to bound the full field → the prior "change mechanism" verdict is now EARNED. Pivot (structural barrier per the 2026-06-19 recap).
- [ ] **Step 2: Write the result** with the §8 plain-language abstract + the口径 lock: the self-limit claim is TWO-LAYER (full-field finite + virtual-SEEG reproduces Task 0's contact AF/LR), the contact footprint is sampling-dominated (Task 0), NOT "the gate shrinks the SEEG footprint to a segment"; record what was tested (shunting + ahead-recruit + I→E gate (+recovery)), the workpoint, and both layers' results. Update `m2_stage_recap_2026-06-19.md` (collaborator recap) + the M0/M2 memory.

---

## Self-Review

1. **Spec coverage:** DATA audit = CONTACT-space methodological constraint that CALIBRATES the model's layer-2 acceptance (Task 0, NOT a model stop-gate; pinned broad-pool source + canonical loaders + accepted-axis projection, 3-mode matched null incl. shaft-matched, pre-registered AF/LR thresholds; executed 2026-06-19 = INCONCLUSIVE / contact-long + sampling-like) ✓; TWO-LAYER model acceptance — layer 1 full-field finite self-limit (Task 6 Step 3) + layer 2 virtual-SEEG reproduces Task 0 AF/LR (Task 6 Step 3b) ✓; Q2 shunting (Tasks 1–3) ✓; Q1/GapA ahead-recruitment (Task 4) ✓; ahead-of-front + axial-clamp DIRECT diagnostics (Task 4.5 + Task 6 Step 4 — P1) ✓; operating-point caveat for shunting AND gate-on (Task 5 incl. 4-cell sanity — P1) ✓; Q3 axial-vs-lateral-vs-far-ahead cell groups (Task 6 Step 4 — P1, no longer assumed) ✓; Q4 (EE shortcut — ruled out in audit) ✓; coexistence guard shunt+recovery+gate (Task 3 Step 7 — P0) ✓; faithful re-test + verdict (Tasks 6–7) ✓.
2. **Tree strategy (P0 fixed):** all model tasks run IN `.worktrees/topic4-m1` (the only tree with recovery+gate+reach_axis); NO `cp` from main (which lacks them, verified 2026-06-19). Global Constraints + Task 1 banner + Task 6 Step 1 all state this.
3. **Placeholder scan:** the only `<...>` is the result-doc date (Task 7); `(g*, drive*)` is the explicit Task-5 output; Task 0 Step 1 is a real data-API inspection step (not a placeholder). No TODO/TBD.
4. **Type consistency:** `membrane_step(...)`, `shunt_gaba/e_gaba/g_gaba_scale`, `ei_gate_scale/l_ei_gate/C_ei_gate`, `front_lead_by_axis(...)`, `clamp_check(...)`, `event_extent(...)`, `matched_null_extent(...)` consistent across engine/runner/tests; bit-parity SHA `da5fc18c` (worktree default-off) is the single anchor across all parity tests; `I_lead_ms`/`frac_axial_gated_by_shunt` are the Task-6-Step-4 → Task-7 link.
5. **Bit-parity hazard:** every new mechanism gated (`shunt_gaba=False`, `ei_gate_scale=0` defaults) + `dump_i_spikes`/`dump_drive` are readout-only; parity tested at helper / `simulate_kick` / `build_connectivity_rot` level BEFORE each re-bless, in the worktree.
6. **Operating-point hazard (P1 widened):** Task 5 re-validates spontaneity/quiet-rest for shunting AND for each gate (4-cell sanity), so a gate "containment" can't be a hidden tonic/killed-ignition artifact.
7. **Verdict integrity (two-layer):** Task 7 "gate works" requires LAYER 1 (full-field finite: reach_axis bounded + L-invariant + edge_margin>0 + not tonic) AND LAYER 2 (virtual-SEEG contact AF/LR consistent with Task 0) AND the ahead-of-front lead + axial-clamp mechanism diagnostics — `reach_axis` alone is insufficient, and a SHORT contact footprint is now a layer-2 FAILURE (must match Task 0's long + sampling-like footprint), not a success. Task 0 is a contact-space CONSTRAINT, not a tissue-space stop-gate.
