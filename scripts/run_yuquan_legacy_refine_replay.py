"""Track B — Yuquan legacy-refine + legacy-gpu replay driver.

Wraps `run_yuquan_lagpat_backfill.run_subject` with the path-injection
contract: refine npz comes from `<legacy_refine_root>/<subject>/_refineGpu.npz`
and per-record gpu npz comes from `<legacy_gpu_root>/<subject>/<stem>_gpu.npz`.
Replay output goes under `<out_root>/<subject>/`, never to the production
raw tree.

Coverage (verified against the disk on 2026-04-25): 14 subjects, 171 records
have the (legacy refineGpu ∩ legacy gpu_npz ∩ .legacy_backup lagPat) triple.
Subjects without a legacy `_refineGpu.npz` are skipped with a recorded
reason.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_yuquan_lagpat_backfill import (  # noqa: E402
    DATA_ROOT,
    YUQUAN_SAME_SOURCE_SUBJECTS,
    run_subject,
)


REPLAY_ROOT_DEFAULT = REPO_ROOT / "results" / "lagpat_backfill_legacy_refine_replay"
LEGACY_ROOT_DEFAULT = DATA_ROOT  # /mnt/yuquan_data/yuquan_24h_edf


def _has_legacy_refine(subject: str, legacy_refine_root: Path) -> bool:
    return (legacy_refine_root / subject / "_refineGpu.npz").exists()


def _replay_subject(
    subject: str,
    *,
    legacy_refine_root: Path,
    legacy_gpu_root: Path,
    out_root: Path,
) -> Dict[str, object]:
    """Run one subject through `run_subject` with legacy paths injected.

    Returns a small per-subject status dict (the full summary + manifest
    are persisted by this function — see below).
    """
    if not _has_legacy_refine(subject, legacy_refine_root):
        return {
            "subject": subject,
            "status": "skipped",
            "skip_reason": "no_legacy_refine_npz",
        }

    out_dir = out_root / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        summary, manifest = run_subject(
            subject,
            legacy_refine_root=legacy_refine_root,
            legacy_gpu_root=legacy_gpu_root,
            out_dir=out_dir,
            backup_dir=None,                 # NEVER write a .legacy_backup in replay
            same_source_assertion=False,
        )
    except FileNotFoundError as exc:
        return {
            "subject": subject,
            "status": "error",
            "skip_reason": f"file_missing: {exc}",
            "elapsed_sec": round(time.time() - t0, 1),
        }

    summary_path = out_dir / "summary.json"
    manifest_path = out_dir / "manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))

    return {
        "subject": subject,
        "status": "ok",
        "write_status": summary["write_status"],
        "n_blocks_written": summary["n_blocks_written"],
        "n_blocks_total": summary["n_blocks_total"],
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "elapsed_sec": round(time.time() - t0, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Track B — Yuquan legacy-refine + legacy-gpu replay driver"
    )
    parser.add_argument(
        "--legacy-refine-root", type=str, default=str(LEGACY_ROOT_DEFAULT),
        help="Root containing <subject>/_refineGpu.npz (default: DATA_ROOT)."
    )
    parser.add_argument(
        "--legacy-gpu-root", type=str, default=str(LEGACY_ROOT_DEFAULT),
        help="Root containing <subject>/<stem>_gpu.npz (default: DATA_ROOT)."
    )
    parser.add_argument(
        "--out-root", type=str, default=str(REPLAY_ROOT_DEFAULT),
        help="Replay output root (default: results/lagpat_backfill_legacy_refine_replay)."
    )
    parser.add_argument(
        "--only-subject", action="append", default=None,
        help="Restrict to listed subject(s); repeatable."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Replay every same-source subject that has a legacy _refineGpu.npz."
    )
    args = parser.parse_args()

    legacy_refine_root = Path(args.legacy_refine_root)
    legacy_gpu_root = Path(args.legacy_gpu_root)
    out_root = Path(args.out_root)

    # Defensive guard: must not write replay output into the live raw tree.
    if out_root == DATA_ROOT:
        print(f"FATAL: --out-root {out_root} equals DATA_ROOT {DATA_ROOT}; "
              f"refusing to overwrite the production raw tree.", file=sys.stderr)
        return 2

    if args.only_subject and args.all:
        print("ERROR: pass --only-subject or --all, not both.", file=sys.stderr)
        return 2
    if not args.only_subject and not args.all:
        print("ERROR: must pass either --only-subject <name> (repeatable) or --all.",
              file=sys.stderr)
        return 2

    if args.all:
        subjects: Sequence[str] = list(YUQUAN_SAME_SOURCE_SUBJECTS)
    else:
        subjects = list(args.only_subject)
        for s in subjects:
            if s not in YUQUAN_SAME_SOURCE_SUBJECTS:
                print(f"WARN: {s} not in YUQUAN_SAME_SOURCE_SUBJECTS — proceeding anyway.")

    out_root.mkdir(parents=True, exist_ok=True)

    cohort_status: List[Dict[str, object]] = []
    n_ok = 0
    n_failed = 0
    n_skipped = 0
    t_start = time.time()
    for i, subject in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}] === replay subject={subject} ===")
        status = _replay_subject(
            subject,
            legacy_refine_root=legacy_refine_root,
            legacy_gpu_root=legacy_gpu_root,
            out_root=out_root,
        )
        cohort_status.append(status)
        if status["status"] == "ok":
            n_ok += 1
            print(f"  write_status={status['write_status']} "
                  f"ok={status['n_blocks_written']}/{status['n_blocks_total']} "
                  f"elapsed={status['elapsed_sec']}s")
        elif status["status"] == "skipped":
            n_skipped += 1
            print(f"  SKIPPED: {status['skip_reason']}")
        else:
            n_failed += 1
            print(f"  ERROR: {status.get('skip_reason')}")

    cohort_status_path = out_root / "_cohort_replay_status.json"
    cohort_status_path.write_text(json.dumps({
        "cohort_size": len(subjects),
        "n_ok": n_ok,
        "n_skipped": n_skipped,
        "n_failed": n_failed,
        "legacy_refine_root": str(legacy_refine_root),
        "legacy_gpu_root": str(legacy_gpu_root),
        "out_root": str(out_root),
        "elapsed_sec_total": round(time.time() - t_start, 1),
        "subjects": cohort_status,
    }, indent=2, ensure_ascii=False, default=str))

    print(f"\nreplay DONE ok={n_ok} skipped={n_skipped} failed={n_failed}")
    print(f"cohort status: {cohort_status_path}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
