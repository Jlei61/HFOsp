# wangyiyang `pack_top_n=22` — explicit pack-stage cap decision

Status: **option A (keep cap, elevate to explicit per-subject contract)**, recorded as carry-over from legacy backfill script `LEGACY_SUBJECT_PARAMS`.

## Where it lives now

- `config/subject_params.json::yuquan.wangyiyang.pack_top_n = 22`
- Documented inline via `_pack_top_n_status` field on the same entry.
- Pinned by `tests/test_yuquan_pack_params_config.py::test_per_subject_pack_params_match_legacy_snapshot`.
- Surfaced in every backfill run via:
  - `summary.json::params.pack_top_n`
  - `summary.json::records[].pack_top_n`
  - `manifest.json::params_resolved.pack_top_n`

## Why it is set

Canary debugging during the original backfill found one real legacy-covered edge event in record `FA0012PF` that disappears when the post-Phase-D detector + new refine pipeline admits **2 extra channels** at the pack stage (24 picked aliases instead of legacy's 22).

The downstream `chns_thr = 0.5` rule converts the picked-alias cardinality into a per-window participation threshold via `ceil(chns_thr * n_picked)`:
- legacy 22 picks → threshold 11
- new pipeline 24 picks → threshold 12

The +1 threshold quietly excludes a single legacy event whose participating channel count happens to land exactly at 11. Capping the pack-stage picked set back to the legacy cardinality of 22 restores that event without creating any new pack windows for records that legacy never had a `lagPat` for.

The cap is **pack-stage only**: the upstream `*_gpu.npz` and `_refineGpu.npz` are unchanged.

## Why it stays in (option A)

1. The cap is the minimum surgical fix that restores legacy block-presence parity for `wangyiyang`. Removing it would re-introduce a per-subject regression visible in `legacy_block_presence_diff.regressions` for at least record `FA0012PF`.
2. With option A in place, every block of `wangyiyang` either reproduces the legacy `lagPat` set or is explicitly skipped via `skip_reason`, and the pack-stage divergence from a "uniform 24-subject rule" is one number (`pack_top_n=22`) that lives in config and is auditable.
3. The carry-over keeps Topic 1/2 `wangyiyang` numbers stable while we evaluate option B in a separate pass.

## Option B — deferred follow-up

Removing `pack_top_n` for `wangyiyang` and re-quantifying drift is **explicitly deferred** to a follow-up. The required steps are:

1. Re-run packing for `wangyiyang` with the cap removed.
2. Compare against the legacy `lagPat` set:
   - block-presence diff (expect at least `FA0012PF` regression)
   - per-block `n_events` delta
   - per-event Pearson r / Kendall τ on `lagPatRaw`
3. Decide if the marginal scientific cost of losing those events is acceptable for a uniformly cap-free 24-subject contract.
4. If acceptable, drop `pack_top_n` from config, update this note, and re-run the cohort audit.

This option is intentionally **not** part of the current 24-subject contract closure pass because:
- it requires re-running the same heavy pack stage on `wangyiyang`,
- it requires re-running the Phase B drift validator on `wangyiyang` to land a fresh, signed-off statistical comparison, and
- it does not block any of the other 20 same-source subjects.

## Audit hooks

- The cohort-level audit (`scripts/audit_yuquan_lagpat_contract.py`) emits a row for every subject whose resolved params include `pack_top_n`, so the cap can never silently propagate to another subject.
- Any change to `wangyiyang.pack_top_n` (or to the cohort membership) will break `tests/test_yuquan_pack_params_config.py` and force the change to be acknowledged.
