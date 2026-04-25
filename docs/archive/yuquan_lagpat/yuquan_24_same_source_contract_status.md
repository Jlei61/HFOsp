# Yuquan 24-subject same-source lagPat contract — implementation status

Status: **code path closed, batch re-run pending**.

This document captures the state of the Yuquan same-source `lagPat` /
`packedTimes` contract after the 2026Q2 closure pass driven by the plan
`/.cursor/plans/yuquan-24-contract_2ad18e1e.plan.md`.

## Cohort definition

The same-source Yuquan cohort is **21 unique subjects** (the plan's "24" is the
3 reference subjects double-counted across the main-cohort and backfill
buckets). The single source of truth is the constant
`scripts/run_yuquan_lagpat_backfill.py::YUQUAN_SAME_SOURCE_SUBJECTS`:

| group | count | subjects |
|---|---|---|
| Reference (have legacy `lagPat`) | 3 | `gaolan`, `dongyiming`, `wangyiyang` |
| Main cohort (have legacy `lagPat`) | 10 | `chenziyang`, `hanyuxuan`, `huanghanwen`, `huangwanling`, `litengsheng`, `sunyuanxin`, `xuxinyi`, `zhangjinhan`, `chengshuai`, `liyouran` |
| Backfill-only (legacy skipped) | 8 | `pengzihang`, `songzishuo`, `zhangbichen`, `zhangjiaqi`, `zhangkexuan`, `zhaochenxi`, `zhaojinrui`, `zhourongxuan` |

`scripts/_phaseE2_run_packing.sh --list` prints this list at runtime.

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

## Numerical equivalence audit (next step)

The cohort verdict above is **structural** (file presence, schema,
start-time, alias collisions, regressions). It does **not** verify
that the per-event `lagPatRank` / `lagPatRaw` arrays match the legacy
ground truth in `.legacy_backup/<stem>_lagPat.npz` to within a
floating-point physical-error tolerance.

The next step is a separate read-only audit that, for every block in
the 21-subject cohort, pairs:

- `<raw>/<stem>_lagPat.npz`            (new v2 output, post-batch)
- `<raw>/.legacy_backup/<stem>_lagPat.npz` (legacy ground truth)

and reports per-record / per-subject / cohort-level array distances:

- `chnNames` set-identity (alias-mapped)
- `eventsBool` exact equality
- `lagPatRank` exact equality (ordinal, no physical noise allowed)
- `lagPatRaw` max-abs and median-abs distance against a published
  threshold; gap between threshold and observed max
- `start_t` |delta| ≤ 1.0 s

This audit is tracked separately and lives under
`results/lagpat_backfill/_audit/numerical_equivalence/`. Until it
reports `cohort_numerical_pass=True`, the cohort claim is "structural
contract closed; numerical equivalence pending".
