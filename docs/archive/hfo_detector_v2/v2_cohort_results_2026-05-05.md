# v2 cohort results (in progress)

Phase 3 cohort rebuild — single-subject smoke gate before full 20-subject run.

| subject | sfreq | records (smoke) | total_events | layer A | layer B | layer C | notes |
|---------|-------|-----------------|--------------|---------|---------|---------|-------|
| 635     | 1024  | 1 (`63500102_0000`) | 2866    | SKIPPED (raw not adjacent) | (pending) | (pending) | smoke 2026-05-06; GPU; 97.6s wall (1h record, 57 ch CAR) |

Subject 635 actually has **123 records** (~123h continuous, 1h each). Task 3.3
ran with `--smoke` to exercise the pipeline plumbing (output dir routing, GPU
default, NPZ format) on the first record only; the full 123-record run is
deferred to Task 3.4 cohort dispatch.

## Phase 3 status

- Task 3.1 ✅ — parallel script defaults updated (`results/hfo_detector_v2/`, `N_JOBS=1`)
- Task 3.2 ✅ — Epilepsiae GPU default set (`--gpu/--no-gpu` tristate; Yuquan stays CPU)
- Task 3.3 ✅ — 635 single-record smoke run completed
- Task 3.4 ⏳ — cohort run pending (separate background dispatch)

## Task 3.3 smoke deviations from plan

1. **Used `--smoke` flag.** Plan estimated 1–12 records / ~30–60min for subject 635;
   actual is 123 records (~10h) which is full-cohort scale, not smoke. Per advisor
   review, ran `--smoke` (1 record, 97.6s) since Phase 2 Task 2.3 already
   confirmed determinism on 635 (`count_match=true, max_t_diff=0.0`).
2. **No `_refineGpu.npz` produced.** `--smoke` skips refine. Refine path will be
   exercised by Task 3.4 cohort run on all 20 subjects (including 635 full).
3. **Layer A extractor returned `records: []` (1 skipped).** Reason:
   `raw .head/.data not adjacent` to v2 gpu.npz output (now under
   `results/hfo_detector_v2/635/`, not `/mnt/epilepsia_data/...`). This is the
   known Phase 3 design issue documented in Phase 2 Task 2.2 docstring; the
   Layer A extractor's path-resolution refactor is deferred to Phase 4.

## Smoke output

- `results/hfo_detector_v2/635/63500102_0000_gpu.npz` (51.6 KB)
  - keys: `whole_dets`, `chns_names`, `events_count`, `start_time`,
    `reference_type`, `bipolar_pairs`
  - reference_type=`car`, 57 channels, 2866 events total
  - top-5 events_count = `[133, 33, 17, 22, 22]`
  - start_time = 1250773219.0 (matches `start_ts=2009-08-20 13:00:19.000`)
- `results/hfo_detector_v2/validation/layer_a_635.json`
  - `records: 0`, `skipped: 1` (`error: 'raw .head/.data not adjacent'`)

## Determinism

Phase 2 Task 2.3 already verified twice-run determinism on 635:
`count_match=True, max_t_diff=0.0`. Not re-run for Task 3.3 smoke.

## Next: Task 3.4

Wrapper script: `scripts/run_epilepsiae_detection_parallel.sh` with defaults
already routed to `results/hfo_detector_v2/`, N_JOBS=1. Full 20-subject cohort
expected wall clock ~10h; dispatch as background job.
