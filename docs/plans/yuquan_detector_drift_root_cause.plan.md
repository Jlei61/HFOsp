# Yuquan Detector Source Drift — Root Cause + Fix Plan

> **Status:** investigation complete, fix plan ready for execution
>
> **Owner:** Linus + Phase B follow-up
>
> **Trigger:** Phase B (`yuquan_lagpat_phaseB_results.md`) found that the new
> `_gpu.npz` for gaolan inflates raw `events_count` by ~3× vs legacy `_gpu.npz`,
> with a strongly non-uniform per-channel pattern (`B'13/B'14/A'10` ≈ 5.9× while
> `D/D'` ≈ 1.3×). This rewrites the picked-channel set and breaks the L1/L3/L4
> drift contract for gaolan. Earlier hypothesis "it's just `pick_k`" is wrong.
>
> **Contract reaffirmed:** *if legacy is not scientifically broken, recover the
> legacy choice exactly.* Only deviate from legacy when legacy itself is wrong.

---

## 0. Tldr

| Tier | Item | What it is | Action |
|---|---|---|---|
| 🔴 must-fix | **D13** notch list missing 250 Hz | new code uses `[50,100,150,200]`, legacy uses `[50,100,150,200,250]` | add 250 Hz |
| 🔴 must-fix | **D14** notch implementation | new = IIR `iirnotch(Q=30)`; legacy = FIR `firwin(801, ±2 Hz)` per band | switch back to legacy FIR |
| 🔴 must-fix | **R02** refine `pick_k` semantics | new code feeds **pack-stage** per-subject `pick_k` (gaolan=1.9) into the **refine** stage. Legacy refine is **global `pickChn_thresh=1.0`** for every subject. | refine writes 1.0; pack keeps per-subject |
| 🟡 should-fix | **D04** chunk size for thresholds | legacy 200 s; new GPU 600 s, CPU 50 s | pin to 200 s |
| 🟡 should-fix | **D03** resample factors | legacy `up=2, down=round(2*fs/800)`; new `Fraction(800,fs).limit_denominator(1000)` | pin to legacy formula |
| 🟢 nice-to-have | **D15** CPU bandpass | legacy = FIR firwin(201) forward; new CPU path = Butter-3 sosfiltfilt | only matters when running CPU detector |
| 🟢 nice-to-have | **D18** chunk-edge events | legacy rejects when both side windows empty; new accepts | small per-boundary inflation |
| 🟢 nice-to-have | **D21** chunk overlap merge | legacy concatenates; new uses 2 s overlap + per-channel merge | boundary semantics drift |
| ⚖️ decision | **D12** drop ≥3 channels | legacy bug: drops nothing if `len(drop_chns)>2`; new code drops correctly | decide: replicate bug for cohort consistency, or document as legacy-only artefact |

> Per-subject differentiation in legacy: **NONE in detector or refine**. Only the
> pack stage uses per-subject `pick_k` and `pack_win_sec`. So the drift cannot be
> "we forgot a per-subject parameter" — it is purely *implementation* drift.

---

## 1. Evidence: legacy parameter audit

Source: `ReplayIED/inter_events/yuquan_24h_perPatientAnalysis_dropRef/`

### 1.1 Detector — `p16_cuda_24h_bipolar.py`

All numerical parameters are **global constants**; only `drop_chns` is per-subject
(via `subs_drop_info`). The misleading `# default 2.0,2.0 / weiwei 1.5 1.5`
comment at `:469-:470` is dead — the live `__main__` always re-asserts
`rel_thresh=2.0, abs_thresh=2.0`.

```
bandpass        = [80, 250] Hz
resample_to     = 800 Hz
segment_time    = 200 s   (per-segment median window)
rel_thresh      = 2.0     (per-channel median multiplier)
abs_thresh      = 2.0     (per-segment median multiplier)
side_thr        = 1.5     (denoise side-mean ratio)
min_gap         = 20 ms
min_last        = 50 ms
max_last        = 200 ms
notch_freqs     = [50, 100, 150, 200, 250]
notch_filter    = firwin(801, [(f-2)/nyq, (f+2)/nyq], pass_zero=True)  forward fftconvolve
bandpass_filter = firwin(201, …)  forward fftconvolve  (NOT filtfilt)
envelope        = sum_k |hilbert(bandpass_k)|, k = 9 sub-bands of 20 Hz inside [80,250]
                  (NO normalization — the divide-by-mean line is commented out)
common_average  = NOT applied (line :412 commented)
reference       = bipolar (adjacent same-shaft) + per-subject drop
resample_poly   = up=2, down=round(2*fs/800)
chunk_overlap   = none (segments concat via cat_chns_times)
```

Detector entrypoint per 200 s GPU batch:

```python
find_high_enveTimes_cu(
    batch_enve, biRef_chns,
    fs=800, rel_thresh=2.0, abs_thresh=2.0,
    min_gap=20, min_last=50, max_last=200,
    start_time=seg_i*200,
)   # side_thr = 1.5 read as global from inside the function
```

### 1.2 Refine — `p16_refine_chns_bySyn.py`

Also entirely global — **no per-subject branch**.

```
pickChn_thresh = 1.0   # mean+1·std on summed events_count, both passes
extL           = 30 ms
packWinLen     = 300 ms
chnsThr        = 0.5
internal fs    = 500 Hz (hard-coded inside get_packedEventsTimes_overThresh; bug, but uniform)
non_overlap    = pick_noOverlap_timeRanges(..., 2)
```

### 1.3 Pack — `P16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool.py`

**Per-subject** dicts at `:493-:499`. Active mapping (`sub_pickT_list`,
`sub_packWL_list`):

| subject | pack `pick_k` | pack `pack_win_sec` |
|---|---|---|
| gaolan | **1.9** | 0.300 |
| dongyiming | 0.5 | 0.220 |
| wangyiyang | 1.0 | 0.250 |
| sunyuanxin | 1.0 | 0.250 |
| chenziyang | 1.0 | 0.300 |
| hanyuxuan | 1.0 | 0.300 |
| huanghanwen | 1.0 | 0.200 |
| litengsheng | 1.0 | 0.300 |
| xuxinyi | 0.7 | 0.200 |
| zhangjinhan | 1.0 | 0.200 |

Note: legacy pack `pick_k` is the **second** application of mean+k·std (after
refine has already applied k=1.0 to thin the channel set). New code has collapsed
these two passes into one and used the pack value for both — that is the R02
bug.

---

## 2. Evidence: new code diff (legacy ↔ new)

Symbols: ✅ identical · ⚠️ different but probably benign · ❌ different and
likely affects `events_count`.

### 2.1 Detector

| ID | Param | Legacy | New | Ref (new) | Match |
|---|---|---|---|---|---|
| D01 | bandpass | `[80,250]` | `[80,250]` | `subject_params.json:16` → `run_hfo_detection.py:165-169` → `hfo_detector.py:42` | ✅ |
| D02 | resample target | 800 Hz | 800 Hz | `subject_params.json:9` | ✅ |
| D03 | resample factors | `up=2, down=round(2*fs/800)` | `Fraction(800,fs).limit_denominator(1000)` | `preprocessing.py:2480-2492` | ❌ for fs ≠ 2000. e.g. fs=2048 → legacy 819.2 Hz, new exact 800 Hz |
| D04 | threshold chunk size | 200 s | GPU 600 s, CPU 50 s | `run_hfo_detection.py:166`; `hfo_detector.py:52-53,245-252` | ❌ chunk-level medians shift |
| D05 | rel_thresh | 2.0 | 2.0 | `bqk_utils.py:161` | ✅ |
| D06 | abs_thresh | 2.0 | 2.0 | `bqk_utils.py:161` | ✅ |
| D07 | side_thr | 1.5 | 1.5 | `bqk_utils.py:178-200, 498-522` | ✅ |
| D08 | min_gap | 20 ms | 20 ms | `bqk_utils.py:148` | ✅ |
| D09 | min_last | 50 ms | 50 ms | `bqk_utils.py:165-171` | ✅ |
| D10 | max_last | 200 ms | 200 ms | `bqk_utils.py:166-169` | ✅ |
| D11 | bipolar reref order | bipolar→drop→resample→notch→bandpass→envelope | bipolar→drop→resample→notch→envelope | `preprocessing.py:2347-2421` | ✅ |
| D12 | drop ≥3 channels | legacy bug: silently NOT applied | applied correctly | `preprocessing.py:2392-2398` | ❌ for weiwei (5), 1084 (5), 916 (11), 590 (3); ✅ for gaolan (1) |
| D13 | notch freqs | `[50,100,150,200,250]` | `[50,100,150,200]` | `preprocessing.py:2125-2128` | ❌ **missing 250** |
| D14 | notch impl | FIR firwin(801, ±2 Hz), forward | IIR `iirnotch(Q=30)` zero-phase | `preprocessing.py:1713-1720` (CPU), `:1788-1803` (GPU) | ❌ family + width + phase all differ |
| D15 | bandpass impl | FIR firwin(201) forward fftconvolve | GPU: same FIR-201 ✅; CPU: Butter-3 sosfiltfilt | `bqk_utils.py:414-441` (GPU), `:368-375` (CPU) | ⚠️ only matters on CPU |
| D16 | envelope sub-bands | sum 9 × 20 Hz `|hilbert|` | identical | `bqk_utils.py:330-348, 426-441` | ✅ |
| D17 | envelope normalization | none (commented out) | none | `bqk_utils.py:557, 565` | ✅ |
| D18 | event side-window when empty | reject (NaN side_mean) | accept | `bqk_utils.py:154-205, 443-526` | ⚠️ small inflation |
| D19 | common-average | NOT applied | NOT applied | `preprocessing.py:2347-2367` | ✅ |
| D20 | threshold reference window | per-segment (200 s) | per-chunk (50/600 s) | `bqk_utils.py:156` | ❌ same design, different size (= D04) |
| D21 | chunk overlap & remerge | none | 2 s overlap + per-chan sort + `merge_timeRanges(min_gap)` | `hfo_detector.py:250-294` | ⚠️ boundary semantics |

### 2.2 Refine

| ID | Param | Legacy | New | Ref (new) | Match |
|---|---|---|---|---|---|
| R02 | refine `pickChn_thresh` (both passes) | **1.0 global** | takes pack's per-subject `pick_k` (gaolan **1.9**, dongyiming 0.5, …) | `subject_params.json:25-49` → `run_hfo_detection.py:243` → `save_refine_gpu_npz` | ❌ violates legacy contract |
| R03 | extL | 30 ms | 30 ms | `group_event_analysis.py:974, 1067` | ✅ |
| R04 | packWinLen | 300 ms | 300 ms | `group_event_analysis.py:972, 1066` | ✅ |
| R05 | chnsThr | 0.5 | 0.5 | `group_event_analysis.py:974, 1068` | ✅ |
| R06 | internal fs | 500 Hz (legacy bug) | 500 Hz (legacy bug preserved) | `group_event_analysis.py:975, 1069` | ✅ |
| R07 | non-overlap rule | drop wins ≥2 s, both-sides on overlap | identical | `group_event_analysis.py:367-370, 976, 1070` | ✅ |
| R08 | rasterization fs | 1000 Hz | 1000 Hz | `group_event_analysis.py:982` | ✅ |
| R09 | selection method | mean + k·std on raw counts | identical algorithm | `group_event_analysis.py:958-964, 984-990` | ✅ |
| R10 | initial-pass count source | sum events_count from `*_gpu.npz` | identical | `group_event_analysis.py:1038-1042, 1046-1050` | ✅ |

---

## 3. Mapping evidence to gaolan's 3× drift

The non-uniform inflation pattern (`B'13/B'14/A'10` ≈ 5.9×, `D/D'` ≈ 1.3×) is
exactly what a missing 250 Hz notch + IIR-Q30 notch would produce:

- **Missing 250 Hz notch (D13)** → 250 Hz mains harmonic sits *on the bandpass
  upper edge* and leaks straight into the [240,250] sub-band envelope. Channels
  closer to the line-noise source (often distal contacts on long shafts — exactly
  `B'13/B'14/A'10`) get the largest envelope inflation.
- **IIR-Q30 vs FIR-±2Hz (D14)** → at 200 Hz the IIR notch is ~6.7 Hz wide (over-
  removes signal); at 250 Hz the IIR notch is *not present at all*, so the
  channel-non-uniform asymmetry is amplified.

Other drift sources (D03, D04, D15, D18, D21) explain residual %-level drift
but not the 3× factor.

`pick_k` (R02) does **not** affect raw `_gpu.npz` `events_count` at all — it
only reshapes the picked set in `_refineGpu.npz`. So **R02 explains the picked-
channel-set drift but not the raw-count drift**.

---

## 4. Fix plan (phased)

### 4.1 Phase D1 — minimum fix, highest ROI

Goal: align everything that has high evidence and low risk. Single-subject
A/B test on gaolan should answer "is this enough?".

Changes (mechanical, no algorithmic re-design):

1. **R02 fix — split refine `pick_k` from pack `pick_k`.**
   - Add new `subject_params.json` field `refine_pick_k`, default **1.0** for
     all subjects.
   - `scripts/run_hfo_detection.py:243` reads `refine_pick_k` (default 1.0)
     instead of `pick_k`.
   - The existing per-subject `pick_k` field is renamed in spirit (kept as
     `pick_k` for backward compat) and now used **only** in the pack stage.
   - Add unit test asserting that for gaolan the refine pass uses 1.0 and the
     pack pass uses 1.9.
2. **D13 fix — notch list adds 250 Hz.**
   - `src/preprocessing.py:2125-2128` change `self.notch_freqs = [50,100,150,200]`
     → `[50,100,150,200,250]`.
   - Or equivalently expose `notch_freqs` as a constructor arg and pass
     `[50,100,150,200,250]` from `run_hfo_detection.py:206`.
3. **D14 fix — notch implementation back to FIR firwin(801, ±2 Hz).**
   - Add a new path in `_apply_notch_cpu` / `_apply_notch_gpu`: if a
     `notch_kind="fir_legacy"` flag is set, build per-band `firwin(801,
     [(f-2)/nyq, (f+2)/nyq], pass_zero=True)` and forward `fftconvolve`.
   - Default the flag to `"fir_legacy"` for the yuquan detector path. Keep the
     IIR path available for non-legacy uses (e.g. CAR-based pipelines).
   - Why a flag instead of replacing: avoid breaking other callers ("Never break
     userspace").

### 4.2 Phase D1 validation

A/B on gaolan. Run new detector on **one 200 s block** with two configurations:

- `cfg_new_aligned` = current new code + D13 + D14 + R02 fixes
- `cfg_new_baseline` = current new code unchanged

Compare both against legacy `_gpu.npz` for the same block. Acceptance:

| Metric | Threshold |
|---|---|
| `|sum(events_count_aligned) − sum(events_count_legacy)| / legacy` | ≤ 5 % |
| Per-channel `|aligned − legacy| / legacy`, p95 | ≤ 25 % |
| Pearson r between aligned and legacy per-channel counts | ≥ 0.95 |
| `B'13/B'14/A'10` inflation factor | ≤ 1.3× (was 5.9×) |
| `D/D'` deflation factor | within ±20 % of 1.0 |

If pass → freeze D1 and move to §4.5 cohort rerun. If fail → Phase D2.

### 4.3 Phase D2 — chunk size + resample

Add (only if D1 fails):

4. **D04 fix — pin chunk size to 200 s.**
   - `scripts/run_hfo_detection.py:166` set `gpu_chunk = 200.0` and
     `cpu_chunk = 200.0` for the legacy-aligned path. Memory cost manageable
     (legacy already ran 200 s at fs=800).
   - Alternatively, expose as a config knob and set in `subject_params.json`.
5. **D03 fix — resample factors.**
   - Add `_legacy_resample_factors(fs, target=800)` returning `(up=2,
     down=round(2*fs/target))`. Use this in `preprocessing.py:2480-2492` when
     `legacy_resample=True`. Default the flag for yuquan path.

A/B same as §4.2.

### 4.4 Phase D3 — CPU path + edge handling (only if D2 still fails)

6. **D15 fix — CPU bandpass back to FIR-201 forward.** (only if anyone runs CPU
   detector)
7. **D18 fix — reject events with empty side window.** Set `side_mean = NaN`
   when both side windows are empty, propagate the existing reject path.
8. **D21 fix — drop chunk overlap, use legacy concat.** (more invasive — only
   if necessary)

### 4.5 Cohort rerun + Phase B re-validation

After whichever phase produces an acceptable A/B:

- Rerun `run_hfo_detection` with the aligned config on **all 11 backfill
  subjects** to regenerate `_gpu.npz` and `_refineGpu.npz`.
- For the **3 reference subjects** (gaolan/dongyiming/wangyiyang), rerun
  `scripts/validate_drift_new_detector.py` (Phase B) on the same 30 common
  blocks. Acceptance:

| Metric | Threshold |
|---|---|
| L1 picked-channel Jaccard | ≥ 0.85 in ≥ 3/3 subjects |
| L2 packed window count ratio | within [0.85, 1.15] in ≥ 3/3 |
| L3 median n_participating shift | within ±15 % in ≥ 3/3 |
| L4 median lag span shift | within ±15 % in ≥ 3/3 |

If pass → 41-subject cohort can replace 30-subject as main conclusion (subject
to L5/L6 still showing no Topic 1/2 drift).

If fail on any subject → that subject moves to a `staging` cohort with explicit
caveat in the manuscript. Do **not** widen the acceptance thresholds — the
contract is recovery, not relaxation.

---

## 5. Decision points (need user sign-off before execution)

### 5.1 D12 legacy drop ≥3 bug — replicate or document?

- Legacy code has `if len(drop_chns) == 0 or len(drop_chns) > 2: do nothing`.
  This means weiwei (5), 1084 (5), 916 (11), 590 (3) actually had **zero**
  channels dropped in the published 30-subject results.
- New code drops them correctly.
- **Affects:** weiwei is in the cohort (existing legacy `_lagPat.npz` exists
  → 30-subject side). 1084, 916, 590 are epilepsiae (different pipeline,
  unaffected). For yuquan backfill (the 11), this bug does NOT affect any of
  them (none has > 2 drops).
- **Options:**
  - **A (replicate)** — add a `legacy_drop_bug=True` flag, skip drop when
    `len > 2`. Cohort consistency, but propagates a known scientific bug.
  - **B (document)** — keep new code's correct behaviour, document that for
    `weiwei` the published result has un-dropped reference contacts. Limit:
    breaks "Never break userspace" for future refresh runs of the legacy 30
    cohort.
  - **C (defer)** — current backfill scope (11 subjects) doesn't touch any
    `len > 2` subject, so this decision can be punted to whenever someone
    re-runs the full 30-subject cohort.

> Linus default: **C (defer)**. Don't fix what isn't broken in scope.

### 5.2 R02 fix scope

- Refine fix is **mandatory** if we want to claim contract recovery.
- It will change `_refineGpu.npz` for **all 8 yuquan subjects with a
  per-subject `pick_k` ≠ 1.0** (huangwanling 3.0, xuxinyi 0.7, zhangbichen
  0.5, zhangjiaqi 1.7, zhaochenxi 0.5, zhangkexuan 0.5, gaolan 1.9, dongyiming
  0.5).
- Of these, only **gaolan** and **dongyiming** are in the current 11-backfill
  scope. The other 6 are in the existing 30-subject cohort.
- Question: do we also rerun the refine pass for those 6 to bring legacy 30
  cohort under the contract? Or leave them as-is and only enforce on backfill?

> Linus default: **enforce on backfill only**. Re-running existing 30-subject
> refine touches the published `_refineGpu.npz` and risks breaking downstream
> caches — out of scope for this PR.

### 5.3 Phase D2/D3 trigger

- Run only if D1 A/B fails the §4.2 threshold.
- Do NOT pre-emptively change D03/D04/D15/D18/D21 — minimise blast radius.

---

## 6. Failure contract (what we will NOT do)

- We will **not** widen the L1/L3/L4 thresholds in §4.5. If legacy alignment
  cannot recover them, the 41-subject cohort stays as a sensitivity analysis,
  not a main result.
- We will **not** add per-subject detector parameters that do not exist in
  legacy. Legacy has none and we are obligated to match.
- We will **not** silently revert the centroid_power=2.0 fallback or any other
  prior fix that aligned us with legacy. All changes here are *additional*
  alignment, not undoing prior alignment.
- If a fix would break the 30-subject legacy cohort's downstream caches, it is
  gated behind a config flag and applied only on the backfill code path.

---

## 7. Out of scope

- Re-running the full 30-subject cohort with the aligned detector (would
  require deletion of published `_gpu.npz`, `_refineGpu.npz`, `_lagPat.npz`).
- Re-deriving the legacy per-subject `pick_k` values (we trust the legacy
  operator's hand-tuned dict as ground truth).
- Investigating the legacy `fs=500` hard-coded synchrony binning — already
  preserved in new code, has no scientific impact under tested conditions.

---

## 8. Execution checklist

- [ ] §5.1 user decision: D12 (default = defer)
- [ ] §5.2 user decision: R02 scope (default = backfill only)
- [ ] §4.1 implement D13 + D14 + R02 (Phase D1 changes)
- [ ] §4.2 A/B on gaolan single block, write report
       `docs/plans/yuquan_detector_drift_phaseD1_results.md`
- [ ] §4.3/§4.4 conditional escalation
- [ ] §4.5 cohort rerun (11 backfill subjects) + Phase B re-validation,
       update `docs/plans/yuquan_lagpat_phaseB_results.md`
- [ ] update `docs/plans/yuquan_lagpat_backfill_validation.plan.md` §555
       prerequisites if they overlap with new fields (`refine_pick_k`)
