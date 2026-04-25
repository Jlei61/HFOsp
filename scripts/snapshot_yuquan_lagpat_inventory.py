"""Read-only pre-batch snapshot of Yuquan lagPat / packedTimes file inventory.

Run this BEFORE any batch packing run that writes into the raw EDF dirs.
Captures, for every subject in the same-source cohort:
  - per-file path, size, mtime, sha256 in `<raw_dir>/`
  - per-file path, size, mtime, sha256 in `<raw_dir>/.legacy_backup/`
  - whether `.legacy_backup/` already exists
  - per-EDF coverage flags (raw lagPat present, raw packedTimes present,
    legacy_backup lagPat present, legacy_backup packedTimes present)

Output: a single JSON file (default
`results/lagpat_backfill/_audit/inventory_snapshots/yuquan_lagpat_inventory_pre_<timestamp>.json`)
plus a short markdown summary alongside it. The script does NOT touch any
file other than the output paths.

Why this exists: once a v2 batch run starts, it will atomically rewrite
`<raw_dir>/<stem>_lagPat.npz` and (for subjects whose `.legacy_backup/` is
empty) move pre-existing legacy files into it. If any of that goes wrong
mid-flight, the only ground truth is what we recorded BEFORE the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_yuquan_lagpat_backfill import (  # noqa: E402
    DATA_ROOT,
    YUQUAN_SAME_SOURCE_SUBJECTS,
)


def _sha256(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                buf = fh.read(chunk)
                if not buf:
                    break
                h.update(buf)
        return h.hexdigest()
    except OSError:
        return None


def _file_record(path: Path, with_hash: bool) -> Dict[str, object]:
    rec: Dict[str, object] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return rec
    st = path.stat()
    rec["size_bytes"] = int(st.st_size)
    rec["mtime"] = float(st.st_mtime)
    if with_hash:
        rec["sha256"] = _sha256(path)
    return rec


def snapshot_subject(subject: str, with_hash: bool) -> Dict[str, object]:
    raw = DATA_ROOT / subject
    bkp = raw / ".legacy_backup"
    out: Dict[str, object] = {
        "subject": subject,
        "raw_dir": str(raw),
        "raw_dir_exists": raw.exists(),
        "legacy_backup_dir": str(bkp),
        "legacy_backup_dir_exists": bkp.exists(),
        "legacy_source_kind": (
            "legacy_backup_dir" if bkp.exists() else "raw_dir_untouched"
        ),
    }
    if not raw.exists():
        out["edfs"] = []
        return out

    edf_paths = sorted(raw.glob("*.edf"))
    edf_stems = [p.stem for p in edf_paths]

    raw_lagpat = {
        p.stem.replace("_lagPat", ""): p for p in raw.glob("*_lagPat.npz")
    }
    raw_packed = {
        p.stem.replace("_packedTimes", ""): p
        for p in raw.glob("*_packedTimes.npy")
    }
    bkp_lagpat = {
        p.stem.replace("_lagPat", ""): p for p in bkp.glob("*_lagPat.npz")
    } if bkp.exists() else {}
    bkp_packed = {
        p.stem.replace("_packedTimes", ""): p
        for p in bkp.glob("*_packedTimes.npy")
    } if bkp.exists() else {}

    edfs_records: List[Dict[str, object]] = []
    for stem in edf_stems:
        rec = {
            "stem": stem,
            "edf_present": True,
            "raw_lagpat": _file_record(
                raw_lagpat.get(stem, raw / f"{stem}_lagPat.npz"), with_hash
            ),
            "raw_packed": _file_record(
                raw_packed.get(stem, raw / f"{stem}_packedTimes.npy"), with_hash
            ),
            "bkp_lagpat": _file_record(
                bkp_lagpat.get(stem, bkp / f"{stem}_lagPat.npz"), with_hash
            ),
            "bkp_packed": _file_record(
                bkp_packed.get(stem, bkp / f"{stem}_packedTimes.npy"), with_hash
            ),
        }
        edfs_records.append(rec)

    extras_in_raw = sorted(
        set(raw_lagpat.keys()) - set(edf_stems)
    )
    extras_in_bkp = sorted(set(bkp_lagpat.keys()) - set(edf_stems))

    out.update({
        "n_edfs": len(edf_stems),
        "n_raw_lagpat": len(raw_lagpat),
        "n_raw_packed": len(raw_packed),
        "n_bkp_lagpat": len(bkp_lagpat),
        "n_bkp_packed": len(bkp_packed),
        "extras_in_raw_lagpat_no_edf": extras_in_raw,
        "extras_in_bkp_lagpat_no_edf": extras_in_bkp,
        "edfs": edfs_records,
    })
    return out


def render_markdown(report: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Yuquan lagPat / packedTimes pre-batch inventory snapshot")
    lines.append("")
    lines.append(f"- captured: `{report['captured_at']}`")
    lines.append(f"- data_root: `{report['data_root']}`")
    lines.append(f"- with_hash: {report['with_hash']}")
    lines.append(f"- output: `{report['snapshot_path']}`")
    lines.append("")
    lines.append("| subject | source | n_edf | raw_lp | raw_pt | bkp_lp | bkp_pt | extra_raw | extra_bkp |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in report["subjects"]:
        lines.append(
            f"| `{s['subject']}` "
            f"| {s['legacy_source_kind']} "
            f"| {s.get('n_edfs', 0)} "
            f"| {s.get('n_raw_lagpat', 0)} "
            f"| {s.get('n_raw_packed', 0)} "
            f"| {s.get('n_bkp_lagpat', 0)} "
            f"| {s.get('n_bkp_packed', 0)} "
            f"| {len(s.get('extras_in_raw_lagpat_no_edf', []))} "
            f"| {len(s.get('extras_in_bkp_lagpat_no_edf', []))} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-batch Yuquan lagPat inventory snapshot")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path. Default: "
             "results/lagpat_backfill/_audit/inventory_snapshots/"
             "yuquan_lagpat_inventory_pre_<UTC-stamp>.json",
    )
    parser.add_argument(
        "--with-hash",
        action="store_true",
        help="Include sha256 of every file (slow, ~minutes for the full cohort).",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject ids (defaults to "
             "YUQUAN_SAME_SOURCE_SUBJECTS).",
    )
    args = parser.parse_args()

    subjects = (
        tuple(args.subjects.split(",")) if args.subjects
        else YUQUAN_SAME_SOURCE_SUBJECTS
    )
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (
        REPO_ROOT / "results" / "lagpat_backfill" / "_audit" / "inventory_snapshots"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.out) if args.out
        else out_dir / f"yuquan_lagpat_inventory_pre_{stamp}.json"
    )
    md_path = out_path.with_suffix(".md")

    print(
        f"snapshotting {len(subjects)} subjects "
        f"with_hash={args.with_hash} -> {out_path}"
    )
    snaps = []
    for s in subjects:
        snap = snapshot_subject(s, with_hash=args.with_hash)
        snaps.append(snap)
        print(
            f"  {s:14s} src={snap['legacy_source_kind']:22s} "
            f"n_edf={snap.get('n_edfs', 0):>3} "
            f"raw_lp={snap.get('n_raw_lagpat', 0):>3} "
            f"bkp_lp={snap.get('n_bkp_lagpat', 0):>3}"
        )

    report = {
        "schema_version": "yuquan_lagpat_inventory_v1_2026Q2",
        "captured_at": datetime.now(tz=timezone.utc).isoformat(),
        "data_root": str(DATA_ROOT),
        "snapshot_path": str(out_path),
        "with_hash": bool(args.with_hash),
        "subjects": snaps,
    }

    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    md_path.write_text(render_markdown(report))
    print(f"wrote: {out_path}")
    print(f"wrote: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
