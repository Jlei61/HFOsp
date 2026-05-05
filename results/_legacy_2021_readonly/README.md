# Legacy 2021 read-only reference

This directory holds the 2021-vintage Epilepsiae HFO detection artifacts
generated with the legacy `cusignal` GPU stack on `niking314` PC. They are
preserved verbatim as a historical reference.

## Status

- **READ-ONLY.** Do not regenerate, overwrite, or move files inside this tree.
- Canonical md5 manifest: `_manifest_v2.csv` (4355 .npz + 4039 .head, schema: relpath,size_bytes,mtime,md5). Original `_manifest.csv` (size+mtime only, .npz subset) preserved for traceability.
- Verified: 2026-05-05 via `Task 0.1` of v2 cohort rebuild plan.

## Why preserved

- Citation source for any reference back to 21 年 paper figures / numbers.
- Diagnostic anchor when investigating per-subject discrepancies.
- NOT a 1:1 validation target for the v2 detector — see v2 specification
  + validation contract under `docs/archive/hfo_detector_v2/`.

## Not preserved here

The recursive Path A diagnostics (commits 6027281, 85f5a29) confirmed the
21 年 results cannot be bit-reproduced on modern stacks. Detector v2 is the
forward-going main pipeline. See:

- `docs/archive/hfo_detector_v2/v2_specification.md`
- `docs/archive/hfo_detector_v2/v2_validation_contract.md`
- `docs/archive/hfo_detector_v2/v2_cohort_rebuild_plan_2026-05-05.md`
