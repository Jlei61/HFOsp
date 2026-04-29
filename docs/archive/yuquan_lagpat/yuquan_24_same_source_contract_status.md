# Yuquan 24-subject same-source lagPat contract — implementation status

Status: **structural cohort PASS (2026-04-23) + dual-track numerical audit complete (Track A done, Track B 6/14 refine-stable subjects strict-ε pass, γ ablation done 2026-04-26)**.

This document captures the state of the Yuquan same-source `lagPat` /
`packedTimes` contract after the 2026Q2 closure pass driven by the plan
`/.cursor/plans/yuquan-24-contract_2ad18e1e.plan.md`. Numerical-equivalence
results land separately under `dual_track_audit_2026-04-26.md`; this doc
links to that archive in §"Numerical equivalence audit (next step)".

## Cohort definition

The same-source Yuquan cohort is **21 unique subjects** (the plan's "24" is the
3 reference subjects double-counted across the main-cohort and backfill
buckets). The single source of truth is the constant
`scripts/run_yuquan_lagpat_backfill.py::YUQUAN_SAME_SOURCE_SUBJECTS`:

| group | count | subjects |
|---|---|---|
| Reference (have legacy `lagPat`) | 3 | `gaolan`, `dongyiming`, `wangyiyang` |
| Main cohort (have legacy `lagPat`) | 11 | `chenziyang`, `hanyuxuan`, `huanghanwen`, `huangwanling`, `litengsheng`, `sunyuanxin`, `xuxinyi`, `zhangjinhan`, `chengshuai`, `liyouran`, `zhangjiaqi` ⁱ |
| Backfill-only (legacy skipped) | 7 | `pengzihang`, `songzishuo`, `zhangbichen`, `zhangkexuan`, `zhaochenxi`, `zhaojinrui`, `zhourongxuan` |

ⁱ **Footnote — `zhangjiaqi` cohort-row correction (2026-04-26):** Earlier
versions of this table listed `zhangjiaqi` under "Backfill-only (legacy
skipped)". A direct disk audit shows `zhangjiaqi` actually has 13 legacy
`<raw>/<stem>_gpu.npz`, a legacy `<raw>/_refineGpu.npz`, and 13 backup
`_lagPat.npz` files — i.e. it has full legacy artifacts. It is therefore
moved into "Main cohort". The column-2 count goes from `Reference 3 +
Main 10 + Backfill-only 8 = 21` to `Reference 3 + Main 11 + Backfill-only
7 = 21`. **Important caveat:** zhangjiaqi's legacy `_gpu.npz` was produced
under a **monopolar** reference (`['A1', 'A2', …]`, 116 chn) while the
new pipeline uses **bipolar** (`['A1-A2', 'A2-A3', …]`, 128 chn). Although
left-contact alias-collapse maps both to the same channel name, the
underlying signals differ → `zhangjiaqi` is **scheme-divergent** at the
detector level (≈ +7.5 ms global onset shift, 0.2% tight match) and is
explicitly excluded from the Track A detector-equivalence claim. See
`dual_track_audit_2026-04-26.md` §2.5 for full evidence.

`scripts/_phaseE2_run_packing.sh --list` prints this list at runtime
(updated to reflect the corrected cohort split).

## What changed in this closure pass

### 1. `unify-pack-config`

- `config/subject_params.json` now carries every Yuquan pack-stage parameter:
  - `_defaults.pack_win_sec = 0.300` (legacy `sub_packWL_list` default)
  - `_defaults.pack_drop_channels = []` (legacy module-level `drop_chns`)
  - per-subject `pack_win_sec` overrides for the 12 subjects that diverged
  - `wangyiyang.pack_top_n = 22` with explicit `_pack_top_n_status` note
- `scripts/run_yuquan_lagpat_backfill.py::resolve_subject_pack_params`
  is now the single loader; `LEGACY_SUBJECT_PARAMS` and `SUBJECT_DROP_CHNS`
  in-script tables are deleted.
- Pinned by `tests/test_yuquan_pack_params_config.py` (4 tests, all pass).

### 2. `close-write-contract`

- `pack_one_record` now performs a hard `start_time` validation against EDF
  `meas_date.timestamp()` with a 1.0 s threshold; mismatches are recorded as
  `skip_reason=start_time_mismatch` and never write.
- `run_subject` now returns `(summary, manifest)`; `main` writes both.
- `summary.json` (schema `yuquan_lagpat_backfill_v2_2026Q2`) carries:
  - `write_status` (`ok` / `partial_ok` / `all_failed` / `no_inputs`)
  - `n_blocks_total/written/skipped/error/missing_gpu_npz`
  - `n_alias_collisions` + `n_alias_collisions_in_picked_max/min`
  - `median_n_participating`, `median_lag_span_ms`
  - `start_time_validation_overall_pass`
  - `legacy_block_presence_diff` with `regressions` and `extras_written`
  - per-block `skip_reason` and `start_time_validation`
- `manifest.json` carries the subject-level write provenance: refine npz
  path/mtime/size, alias collisions, alias outer drops, per-block edf+gpu
  file stats, write paths, and backup events.

### 3. `audit-legacy-presence`

- `_legacy_block_presence_diff` is computed per subject and embedded in
  `summary.json`. It buckets EDF stems into `legacy_present_status` /
  `legacy_absent_status` by run status and surfaces:
  - `regressions`: legacy had `_lagPat.npz` for the block but new pipeline
    didn't write one (red flag)
  - `extras_written`: new pipeline wrote a block legacy didn't have
- The cohort-level audit (see §5) rolls these up.

### 4. `resolve-special-cases`

- `wangyiyang.pack_top_n = 22` is the only explicit pack-stage special case.
- It is **option A in plan terminology**: kept as an explicit, audit-grade
  per-subject contract. The decision and the deferred option-B follow-up are
  recorded in `wangyiyang_pack_top_n_decision.md`.
- Surfaced explicitly in:
  - `summary.json::params.pack_top_n` and `summary.json::records[].pack_top_n`
  - `manifest.json::params_resolved.pack_top_n`
  - cohort audit `cohort_audit.md::Explicit pack-stage special cases`

### 5. `scale-to-24-yuquan`

- `scripts/_phaseE2_run_packing.sh` now drives off
  `YUQUAN_SAME_SOURCE_SUBJECTS` (21 subjects), supports `DRY_RUN=1` /
  `ONLY_SUBJECTS="..."` env vars, continues past per-subject failure but
  exits non-zero on any failure, and supports `--list` for inspection.
- Cross-category dry-runs verified the v2 contract on:
  - `gaolan` `FA0013KP` (reference)
  - `chenziyang` `FC1047XY` (main cohort with legacy `lagPat`)
  - `pengzihang` `FA1349ZH` (backfill-only)

  All three returned `write_status=ok`, `start_time_validation` pass,
  `alias_in_picked=0`, and produced both `summary.json` and `manifest.json`
  in the expected schema.

### 6. `ship-cohort-audit`

- `scripts/audit_yuquan_lagpat_contract.py` is a read-only audit that
  aggregates per-subject `summary.json` + `manifest.json` plus inputs and
  emits:
  - `cohort_audit.json`
  - `cohort_audit.md`
- Cohort verdict is `PASS` iff all 21 subjects:
  - have `summary.json` and `manifest.json`
  - point at `results/hfo_detection/<subject>/_refineGpu.npz` (same-source)
  - pass `start_time` validation across every block
  - have zero alias collisions in picked aliases
  - have zero `legacy_block_presence_diff.regressions`

## Current audit verdict

Status: **PASS** (post-batch, 2026-04-23 23:51 UTC batch + 2026-04-25 audit).

Production batch: `logs/yuquan_lagpat_pack/_master_20260423T080346Z.log`
finished at `2026-04-23 23:51:28` with `ok=21 failed=0`.

Audit re-run on 2026-04-25 against the post-batch
`results/lagpat_backfill/`:

```
cohort_pass=True
subjects=21 with_summary=21 with_manifest=21 st_pass=21
alias_red=0 regression_red=0
```

Cohort rollup:

- 21/21 subjects have both `summary.json` and `manifest.json` under the
  v2 schema (`yuquan_lagpat_backfill_v2_2026Q2`)
- 21/21 subjects pass `start_time_validation` across every block (1.0 s
  threshold against EDF `meas_date.timestamp()`)
- 21/21 subjects have zero alias collisions in the picked aliases
- 21/21 subjects have zero `legacy_block_presence_diff.regressions`
- 21/21 subjects point at the same-source detector
  (`results/hfo_detection/<subject>/_refineGpu.npz`)

Block-level totals across the cohort:

- total blocks: 260
- written:      255
- skipped:      5
- error:        0
- missing gpu:  0

Documented per-subject `partial_ok` skips (none indicate drift; all
match legacy behavior or explicit option-A contracts):

- `dongyiming` 11/12 (1 skipped) — legacy never wrote that block either
- `wangyiyang` 9/12 (3 skipped) — explicit `pack_top_n=22` option A; see
  `wangyiyang_pack_top_n_decision.md`
- `liyouran` 11/12 (1 skipped) — legacy never wrote that block either

Explicit pack-stage special cases surfaced by the audit:

- `wangyiyang` (pack_top_n=22) — the only subject with an explicit cap

Audit artifacts:

- `results/lagpat_backfill/_audit/cohort_audit/cohort_audit.json`
- `results/lagpat_backfill/_audit/cohort_audit/cohort_audit.md`

The codepath cohort verdict is closed. Downstream Topic 1/2/3 work may
now describe the Yuquan intermediate-file slice as
"21-subject same-source (closed-codepath)". Numerical equivalence
against the legacy ground truth is a separate next-step audit (see
"Numerical equivalence audit (next step)" below).

## Pre-batch acceptance review (this snapshot)

Run before the operator hits the green button:

- 25/25 unit tests pass (`tests/test_legacy_align_detector.py`,
  `tests/test_legacy_lagpat_centroid.py`,
  `tests/test_legacy_refine.py`,
  `tests/test_yuquan_pack_params_config.py`,
  `tests/test_legacy_block_presence_diff.py`).
- All 21 same-source subjects have a complete same-source detector
  inventory (`_refineGpu.npz` + `_gpu.npz` count == EDF count).
- Cross-category dry-runs (`gaolan` reference, `chenziyang` main cohort,
  `pengzihang` backfill-only) pass with `write_status=ok`,
  `start_time_validation_overall_pass=true`, alias collisions in picked = 0.

Two structural fixes landed during pre-batch acceptance:

1. **`_legacy_block_presence_diff` semantics fix.** Previously the audit
   treated a `<raw_dir>/<stem>_lagPat.npz` as legacy evidence even when it
   was a pre-existing v1 backfill output, which would inflate
   `legacy_present` and deflate `regressions` after the second batch run.
   New rule: `.legacy_backup/` is the single source of truth once it
   exists; otherwise `raw_dir` (untouched subject) is. Pinned by
   `tests/test_legacy_block_presence_diff.py`.
2. **Pre-batch inventory snapshot tool.**
   `scripts/snapshot_yuquan_lagpat_inventory.py` records per-file
   path/size/mtime (and optional sha256) for every EDF / lagPat /
   packedTimes file under `<raw>/` and `<raw>/.legacy_backup/`. Run it
   immediately before the production batch so any mid-flight failure has a
   recorded ground truth to compare against.

## Pre-batch backup state (from inventory snapshot)

Two well-defined source kinds across the 21 subjects:

- `legacy_source_kind = legacy_backup_dir` (11 subjects — already
  backfilled once). `.legacy_backup/<stem>_lagPat.npz` is the canonical
  legacy snapshot. `raw_dir/<stem>_lagPat.npz` is a v1 backfill output and
  will be overwritten atomically by v2.
- `legacy_source_kind = raw_dir_untouched` (10 subjects — chenziyang,
  hanyuxuan, huanghanwen, huangwanling, litengsheng, sunyuanxin, xuxinyi,
  zhangjinhan, chengshuai, liyouran). `raw_dir/<stem>_lagPat.npz` is the
  legacy snapshot and the first batch run will move it into
  `.legacy_backup/` before writing the v2 output.

### Per-subject legacy ground-truth locations

The path in the last column is the canonical place to find each subject's
**original** legacy `_lagPat.npz` / `_packedTimes.npy` files. Topic 1/2/3
should never need to read these directly — they go through the live
`raw_dir/<stem>_*` files — but if a v2 batch result is ever in question
the legacy ground truth lives here:

| subject | source kind | n_edf | legacy lagPat (truth) | ground-truth path |
|---|---|---:|---:|---|
| `gaolan` | legacy_backup_dir | 12 | 12 | `/mnt/yuquan_data/yuquan_24h_edf/gaolan/.legacy_backup/` |
| `dongyiming` | legacy_backup_dir | 12 | 9 | `/mnt/yuquan_data/yuquan_24h_edf/dongyiming/.legacy_backup/` |
| `wangyiyang` | legacy_backup_dir | 12 | 9 | `/mnt/yuquan_data/yuquan_24h_edf/wangyiyang/.legacy_backup/` |
| `chenziyang` | raw_dir_untouched | 12 | 12 | `/mnt/yuquan_data/yuquan_24h_edf/chenziyang/` (raw, untouched) |
| `hanyuxuan` | raw_dir_untouched | 13 | 13 | `/mnt/yuquan_data/yuquan_24h_edf/hanyuxuan/` (raw, untouched) |
| `huanghanwen` | raw_dir_untouched | 12 | 12 | `/mnt/yuquan_data/yuquan_24h_edf/huanghanwen/` (raw, untouched) |
| `huangwanling` | raw_dir_untouched | 12 | 12 | `/mnt/yuquan_data/yuquan_24h_edf/huangwanling/` (raw, untouched) |
| `litengsheng` | raw_dir_untouched | 16 | 16 | `/mnt/yuquan_data/yuquan_24h_edf/litengsheng/` (raw, untouched) |
| `sunyuanxin` | raw_dir_untouched | 12 | 12 | `/mnt/yuquan_data/yuquan_24h_edf/sunyuanxin/` (raw, untouched) |
| `xuxinyi` | raw_dir_untouched | 13 | 13 | `/mnt/yuquan_data/yuquan_24h_edf/xuxinyi/` (raw, untouched) |
| `zhangjinhan` | raw_dir_untouched | 13 | 13 | `/mnt/yuquan_data/yuquan_24h_edf/zhangjinhan/` (raw, untouched) |
| `chengshuai` | raw_dir_untouched | 12 | 12 | `/mnt/yuquan_data/yuquan_24h_edf/chengshuai/` (raw, untouched) |
| `liyouran` | raw_dir_untouched | 12 | 11 | `/mnt/yuquan_data/yuquan_24h_edf/liyouran/` (raw, untouched, 1 EDF lacks legacy lagPat — historical, not new) |
| `pengzihang` | legacy_backup_dir | 12 | 12 | `/mnt/yuquan_data/yuquan_24h_edf/pengzihang/.legacy_backup/` |
| `songzishuo` | legacy_backup_dir | 12 | 10 | `/mnt/yuquan_data/yuquan_24h_edf/songzishuo/.legacy_backup/` |
| `zhangbichen` | legacy_backup_dir | 11 | 5 | `/mnt/yuquan_data/yuquan_24h_edf/zhangbichen/.legacy_backup/` |
| `zhangjiaqi` | legacy_backup_dir | 13 | 13 | `/mnt/yuquan_data/yuquan_24h_edf/zhangjiaqi/.legacy_backup/` |
| `zhangkexuan` | legacy_backup_dir | 12 | 12 | `/mnt/yuquan_data/yuquan_24h_edf/zhangkexuan/.legacy_backup/` |
| `zhaochenxi` | legacy_backup_dir | 12 | 12 | `/mnt/yuquan_data/yuquan_24h_edf/zhaochenxi/.legacy_backup/` |
| `zhaojinrui` | legacy_backup_dir | 13 | 13 | `/mnt/yuquan_data/yuquan_24h_edf/zhaojinrui/.legacy_backup/` |
| `zhourongxuan` | legacy_backup_dir | 12 | 12 | `/mnt/yuquan_data/yuquan_24h_edf/zhourongxuan/.legacy_backup/` |

Important: where `legacy lagPat (truth) < n_edf`, the gap reflects what
legacy actually wrote, **not** new-pipeline drift:

- `dongyiming` 9/12, `wangyiyang` 9/12, `songzishuo` 10/12, `zhangbichen`
  5/11 — those EDFs never had a legacy lagPat in the first place; the
  v2 batch will write them as net-new blocks (recorded as
  `extras_written` in the cohort audit, not as regressions).
- `liyouran` 11/12 — same situation, raw_dir_untouched edition.

### Strong ground-truth snapshot file

Captured via `scripts/snapshot_yuquan_lagpat_inventory.py --with-hash`,
i.e. every legacy file's path + size + mtime + sha256:

- `results/lagpat_backfill/_audit/inventory_snapshots/yuquan_lagpat_inventory_pre_20260423T080200Z.json`
- `results/lagpat_backfill/_audit/inventory_snapshots/yuquan_lagpat_inventory_pre_20260423T080200Z.md`

All lagPat-related audit outputs (inventory snapshots + cohort audit)
live under one root: `results/lagpat_backfill/_audit/`.

## Operational follow-up

The codepath for the 24-subject contract is closed. The remaining work is
operational, not structural:

1. Capture a fresh inventory snapshot (with hashes if you want strong
   ground truth):

   ```bash
   python scripts/snapshot_yuquan_lagpat_inventory.py --with-hash
   ```

2. Run the full 21-subject batch in production mode:

   ```bash
   bash scripts/_phaseE2_run_packing.sh
   ```

   This will write to each subject's raw EDF dir (with `.legacy_backup/`
   stashing as needed) under the new schema. Logs land in
   `logs/yuquan_lagpat_pack/`. Wall-clock estimate: ~10 records per subject
   times ~60–120 s each times 21 subjects ≈ 4–6 hours single-threaded.

3. Re-run the audit against `results/lagpat_backfill/`:

   ```bash
   python scripts/audit_yuquan_lagpat_contract.py
   ```

   Expected outcome: `cohort_pass=True`. Any subject failing validation
   will be flagged in `cohort_audit.md` with a one-line diagnosis derived
   from its `summary.json::write_status` /
   `legacy_block_presence_diff`. The diff is now correct for both
   already-backfilled and untouched subjects.

4. If any `legacy_block_presence_diff.regressions` are non-empty, treat
   them as Phase B-style detector drift and either:
   - accept them as documented divergence (record in this file), or
   - bump them into a follow-up `option-B re-validation` doc (analogous
     to `wangyiyang_pack_top_n_decision.md`).

Until step 3 reports `cohort_pass=True`, downstream work in Topic 1/2/3
**must not** describe the Yuquan intermediate-file slice as "24-subject
same-source" — only the closed-codepath cohort is the right description.

## Explicit non-goals of this pass

- No Epilepsiae work in this pass.
- No re-running of the full Yuquan batch as part of the codepath closure
  itself (handled separately as the operational follow-up).
- No re-running of the option-B drop-`pack_top_n` test for `wangyiyang`
  (handled in `wangyiyang_pack_top_n_decision.md`).
- No edits to Topic 1/2/3 scientific text — that depends on the audit
  flipping to `PASS` first.

## File-level note: `lagPatRaw` is a key, not a separate file

The five legacy intermediate products of the same-source pipeline are:

- `<record>_gpu.npz`              (detector output)
- `<record>_refineGpu.npz`        (refine selection)
- `<record>_packedTimes.npy`      (group-event packing)
- `<record>_lagPat.npz`           (centroid lag pattern)

Inside `<record>_lagPat.npz` the new pipeline writes the same five keys
the legacy pipeline did:

```
keys = ['lagPatRaw', 'lagPatRank', 'eventsBool', 'chnNames', 'start_t']
```

`lagPatRaw` is **a key inside `lagPat.npz`**, not a standalone
`lagPatRaw.npz` file. Anyone tracing the contract under the assumption
that the cohort should produce a fifth `*_lagPatRaw.npz` will not find
one — it is `lagPat.npz['lagPatRaw']` and it is on the legacy
stitched-per-200s-segment timeline (not on the wall-clock timeline).
This is how the legacy producer
(`p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py`)
shipped the file.

## Numerical equivalence audit — dual-track (Track A + Track B + Phase γ)

The cohort verdict above is **structural** (file presence, schema,
start-time, alias collisions, regressions). The structural verdict
**does not** verify that the per-event `lagPatRank` / `lagPatRaw`
arrays match the legacy ground truth at the level of physical
floating-point error.

Numerical equivalence is being audited along two independent tracks +
a root-cause ablation phase. **Full evidence is in the archive
`dual_track_audit_2026-04-26.md`**; this section quotes the verdicts
and links back.

### Track A — Detector event-level attribution (DONE)

Goal: classify the documented ±10–20% events_count drift between new
and legacy `_gpu.npz` (phaseD) at the event level — is it physical-FP
threshold-edge drift on a fixed detection logic, or is it a coarse
logic divergence (channel mapping error / global time shift)?

Cohort coverage (21 subjects, exact bucketing):

| bucket | n | subjects |
|---|---|---|
| **detector-comparable** (≥1 legacy gpu_npz, ≥80% record coverage) | 13 | gaolan, dongyiming, wangyiyang, chenziyang, hanyuxuan, huanghanwen, huangwanling, litengsheng, sunyuanxin, xuxinyi, zhangjinhan, chengshuai, liyouran |
| **scheme-divergent (excluded from detector-equivalence claim)** | 1 | zhangjiaqi |
| **degenerate_evidence** (only 1/12 records have a legacy gpu) | 1 | pengzihang |
| **no_legacy_gpu** | 6 | songzishuo, zhangbichen, zhangkexuan, zhaochenxi, zhaojinrui, zhourongxuan |
| **总计** | 21 | ✓ |

Findings — read the per-subject table together with the cohort
verdict; the verdict alone is misleading.

What is established for the 13 detector-comparable subjects:

- alias-collapsed `legacy_containment_in_new` median ≈ 1.000 (legacy
  channels are present in the new alias set; new can be a strict
  superset because new bipolar pairs include channels legacy didn't);
- `|global_onset_shift|` = 0.00 ms across the bucket — **no global
  coarse time shift**;
- max per-subject unmatched fraction ≤ 14.8% — within the
  phaseD-documented ±20% events_count drift budget. Residuals are
  consistent with **threshold-sensitive FP drift**: events near the
  detector threshold flip in/out of the set under FP-level differences,
  the rest match.

What is **NOT** established:

- **Tight event-level equivalence** (1-sample / 1.25 ms onset match) is
  NOT achieved for all 13. Only 4 subjects (`gaolan`, `wangyiyang`,
  `sunyuanxin`, `xuxinyi`) reach the script's ≥ 0.85 tight-match
  threshold; the remaining 9 sit between 0.51 and 0.84. The bulk of
  matches land in the medium / loose buckets (≤ 6.25 / ≤ 25 ms),
  consistent with FP-level drift in the envelope filter and threshold
  decision but **not** a per-event one-to-one reproduction.

So the precise Track A claim is:

> 13 detector-comparable subjects show no global coarse shift and
> residuals consistent with threshold-sensitive drift; tight
> event-level equivalence is not established for all 13.

1/13 (`zhangjiaqi`) is excluded entirely with cause: legacy uses a
**monopolar** reference, new uses **bipolar**; alias-collapse maps
both to the same channel name but the underlying signals differ →
**+7.5 ms global onset shift, 0.2% tight match, 61% unmatched events**.
Reference-scheme divergence-by-design, not a new-pipeline bug.

**Cohort verdict label**: `coarse_logic_divergence`. **Reading
caveat**: the label is triggered by the script's combined Layer 1 gate
(containment ≥ 0.95 AND |shift| ≤ 1.25 ms AND tight_min ≥ 0.85). The
gate is failed by `zhangjiaqi`'s +7.5 ms shift AND by the lower
tight-match fractions on most subjects. The label itself does NOT
mean the detector behaviour is broadly wrong — read it together with
the per-subject table and the precise claim above.

Reports:
- `results/lagpat_backfill/_audit/detector_attribution/cohort_detector_attribution.{json,md}`
- `results/lagpat_backfill/_audit/detector_attribution/per_subject/<subject>.json`

Implementation:
- `scripts/audit_yuquan_detector_event_match.py`
- `tests/test_yuquan_detector_event_match.py` (10 tests)

### Track B — Pack+lagPat replay numerical preflight on `gaolan` (DONE)

Goal: hold detector + refine drift constant (use legacy
`_refineGpu.npz` + legacy `<raw>/<stem>_gpu.npz` as inputs) and let the
new pack + lagPat code compete only on its own numerical drift vs the
legacy `<raw>/.legacy_backup/<stem>_lagPat.npz`.

Comparator design (3-layer verdict):

- **strict**: `lagPatRaw` maxabs ≤ 1 ns + `lagPatRank` exact + pack-stage exact.
- **phaseA_baseline**: `lagPatRaw` {median ≤ 5 ms, p95 ≤ 20 ms, RMSE ≤ 10 ms} + `lagPatRank` full-event match ≥ 0.95 (Phase A's already-validated thresholds from `validate_pack_against_legacy.py`).
- **pack_stage_only**: `chnNames` + `packedTimes` + `eventsBool` exact (ignoring centroid-stage drift).

Two-axis alignment (this is the central correctness invariant): rows
by `chnNames` (exact set + permute), columns by `packedTimes` (1:1
nearest-onset within `pack_win_sec/2 = 150 ms`); array diffs run
**only** on aligned indices. Provenance gate: comparator refuses to
verdict any record whose manifest reports a `gpu_npz_used` /
`refine_npz_used` under `DETECT_ROOT` (= a same-source masquerading
as a replay).

Preflight result on `gaolan` (12 records, 2026-04-25):

| 层级 | 通过 |
|---|---:|
| **pack-stage exact** | **12/12** ✓ |
| Phase A baseline | 3/12 |
| strict | 0/12 |

`lagPatRaw` maxabs ranges **1.5–67 ms** with strong positive correlation
to record event count: ≤70 events → ≤8 ms; 550–1682 events → 37–67 ms
maxabs. This is **not** physical FP — it is a real numerical drift in
the centroid stage that scales with the number of stft frames.

Implementation:
- `scripts/run_yuquan_legacy_refine_replay.py` (replay driver, requires
  explicit `--legacy-refine-root` / `--legacy-gpu-root` / `--out-root`)
- `scripts/audit_yuquan_legacy_refine_replay.py` (comparator with
  3-layer verdict + provenance gate)
- `scripts/run_yuquan_lagpat_backfill.py` (`run_subject` refactored to
  accept the same path-injection points; same-source CLI default
  unchanged)
- `tests/test_run_subject_path_overrides.py` (6 tests),
  `tests/test_yuquan_legacy_refine_replay_audit.py` (10 tests)

Preflight artifacts:
- `results/lagpat_backfill_legacy_refine_replay/gaolan/{*_lagPat.npz, *_packedTimes.npy, summary.json, manifest.json}`
- `/tmp/track_b_preflight/cohort_replay_audit.{json,md}` (will be moved
  to `results/lagpat_backfill/_audit/legacy_refine_replay/preflight/`
  during the topical commit).

### Phase γ — Root-cause ablation + cohort run (DONE 2026-04-26)

γ found and fixed two independent code bugs, then re-ran the 14-subject
cohort. **Code-level Track B parity is established** for refine-stable
subjects; the 8 fail subjects fail due to data archaeology, not code.

**γ.0 provenance**: confirmed legacy pack stage active path is scipy
CPU `iirnotch + filtfilt` / `butter(3) + filtfilt` (GPU branches in
the legacy script are commented out). phaseD's "GPU vs CPU FP" cause
applies to the **detector** stage only, not to pack. **GPU port is
NOT a candidate**.

**γ.1 ablation finding**: on `FA0013L8` (low drift, 38 events) and
`FA0013KP` (high drift, 1224 events), L1–L4 (bipolar reref →
resample_poly → notch → band) match bit-for-bit between Branch P
(production) and Branch L (literal legacy reimpl). Branch L matches
R0 (`.legacy_backup`) to within float64 ε (3.3e-16 / 1.4e-14 s).
Branch P diverges at **L5 stitched signal**: one boundary sample is
dropped per high-drift record because
`(end_sec - start_sec) * sfreq` rounds slightly below an integer in
float64, and the previous `+1e-12` epsilon was insufficient.

**γ.1 fix #1** (`src/group_event_analysis.py:284`):
`build_stitched_window_signal` switches to legacy literal path
`batch_t = start_sec + np.arange(n) / sfreq` + boolean union mask,
removing the FP epsilon trap. No algorithm / parameter change.

**γ.4 cohort run (14 subjects, 176 records) revealed second bug**:
`alias_bipolar_to_left_with_arbitration` over-removed channels for
subjects whose 2021-era legacy `_refineGpu.npz` already used
single-electrode `chns_names` (e.g. `['A1', 'A2', ...]`) instead of
bipolar pair (`['A1-A2', 'A2-A3', ...]`). The function's
"outermost-shaft alias drop" was designed for bipolar-pair input and
became destructive on single-electrode input.

**γ.4b fix #2** (`scripts/run_yuquan_lagpat_backfill.py:312`):
`alias_bipolar_to_left_with_arbitration` auto-detects input schema
via `any('-' in name for name in chns_names)`. Bipolar pair input
keeps the original outer-drop behavior (preserves same-source
contract). Single-electrode input skips outer-drop (already
alias-collapsed by legacy refine). 247 in-scope tests still green.

**γ.5 final cohort verdict (Track B legacy-refine replay)**:

| bucket | subjects | pass-eligible records | strict ε pass |
|---|---|---:|---:|
| **refine-stable + strict pass** | 6 | 74 | 68 ✓ |
| **refine drift / uncomparable (excluded from numerical claim)** | 8 | 102 | 0 |
| **not_replayed (no legacy refineGpu)** | 7 | 0 | n/a |
| total | 21 | 176 | 68 |

Refine-stable PASS subjects: `gaolan` 12/12, `dongyiming` 9/12 (3 records
have no legacy backup), `wangyiyang` 9/12 (option-A `pack_top_n=22`
with 3 explicit skips), `sunyuanxin` 12/12, `xuxinyi` 13/13,
`zhangjinhan` 13/13.

Refine-drift FAIL subjects: 7 (`chenziyang`, `hanyuxuan`,
`huanghanwen`, `huangwanling`, `litengsheng`, `chengshuai`,
`liyouran`) had `<raw>/_refineGpu.npz` regenerated by the new
detection pipeline on 2026-04-09/10, overwriting the 2021-era refine
legacy used. The 1 additional fail (`zhangjiaqi`) shows ε-level
events_count drift between its 2021 refine and what legacy-era
threshold logic implied — H1 sits 235 counts above the current
`mean + 1.7*std` threshold, but legacy's pick excluded H1.

Cohort verdict label remains `fail` because the strict criterion
demands all-records-pass cohort-wide. **Read together with the
bucketing** the Track B pack+lagPat code-correctness claim is:

> Code path proven: when `<raw>/_refineGpu.npz` matches the 2021-era
> legacy refine, new pack+lagPat reproduces `.legacy_backup` to
> within float64 ε (6/14 subjects, 68/74 pass-eligible records).
> Other 8/14 subjects fail purely because their refine inputs are
> not the same data legacy used.

**Reports**:
- `results/lagpat_backfill/_audit/legacy_refine_replay/cohort_replay_audit.{json,md}`
- `results/lagpat_backfill/_audit/legacy_refine_replay/per_subject/<subject>.json`
- `results/lagpat_backfill/_audit/pack_layer_ablation/gaolan/{FA0013L8,FA0013KP}_diff_report.json`
- Full evidence + decision tree + bucket table: `dual_track_audit_2026-04-26.md`

### Status table (final)

| 项 | 状态 |
|---|---|
| Structural cohort PASS (21/21 same-source) | ✅ DONE (2026-04-23) |
| `run_subject` path-injection refactor + tests | ✅ DONE |
| Track A — detector event-level attribution | ✅ DONE |
| Track B — pack+lag replay preflight on gaolan | ✅ DONE |
| γ.0 — provenance verification | ✅ DONE |
| γ.1 — layer ablation tool + run | ✅ DONE |
| γ.1 fix — `build_stitched_window_signal` legacy literal path | ✅ DONE |
| γ.4 — Track B 14-subject cohort run | ✅ DONE |
| γ.4b fix — `alias_bipolar_to_left_with_arbitration` auto-detect | ✅ DONE |
| γ.5 — final audit verdict (6/14 strict pass) | ✅ DONE |
| Status doc + topical commit | 🟡 PENDING (this commit) |

The canonical cohort claim has updated to its final form:

- **Structural contract closed.** 21/21 subjects produced
  `_lagPat.npz` + `_packedTimes.npy` under a single closed code path;
  all schema / start-time / alias / presence checks pass.
- **Detector residuals consistent with threshold-sensitive drift on
  13/21.** No global coarse time shift; max per-subject unmatched
  fraction ≤ 14.8%. **Tight event-level (1-sample) equivalence is
  not established for all 13** — only 4 subjects clear the ≥ 0.85
  tight-match bar; the rest sit in the 0.51–0.84 band. `zhangjiaqi`
  scheme-divergent; `pengzihang` + 6 no-legacy-gpu uncomparable.
- **Pack+lagPat code-level parity proven on 6/14 refine-stable
  subjects, 68/74 records, ε-level**: when fed legacy refine + legacy
  gpu_npz, new code reproduces `.legacy_backup` to within float64 ε.
- **8/14 Track B subjects uncomparable** due to refine drift between
  2021 legacy era and 2026 (overwritten or ε-drifted), not a code
  defect.

