"""Rebuild PR-1 cohort summary from per-subject JSONs without re-running PR-1.

Use after partial PR-1 runs (e.g. only 3 new subjects). Reads
`results/interictal_propagation/per_subject/*.json` and writes a fresh
cohort summary that includes all eligible records.

By default, every `<dataset>_<subject>.json` under `--per-subject-dir`
is included. Pass `--manifest` to restrict to an explicit allow-list
(text file, one `dataset/subject` per line, `#` for comments). Always
prints the final inclusion list so stale JSON is loud, not silent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from interictal_propagation import summarize_propagation_cohort  # noqa: E402


VALID_DATASETS = ("yuquan", "epilepsiae")


def _save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any) -> Any:
        try:
            import numpy as np

            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
        except Exception:
            pass
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=_default))


def _load_manifest(path: Path) -> Set[str]:
    keys: Set[str] = set()
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "/" not in line:
            raise SystemExit(
                f"Manifest entry {line!r} missing dataset/subject form (expected `yuquan/<sub>` or `epilepsiae/<sub>`)"
            )
        ds, _ = line.split("/", 1)
        if ds not in VALID_DATASETS:
            raise SystemExit(f"Manifest entry {line!r} dataset {ds!r} not in {VALID_DATASETS}")
        keys.add(line)
    if not keys:
        raise SystemExit(f"Manifest {path} has no entries")
    return keys


def _discover(per_subject_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for f in sorted(per_subject_dir.glob("*.json")):
        stem = f.stem
        if "_" not in stem:
            continue
        dataset, subject = stem.split("_", 1)
        if dataset not in VALID_DATASETS:
            continue
        out[f"{dataset}/{subject}"] = f
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject-dir", default="results/interictal_propagation/per_subject")
    ap.add_argument("--out-summary", default="results/interictal_propagation/pr1_subject_summary.json")
    ap.add_argument("--out-cohort", default="results/interictal_propagation/pr1_cohort_summary.json")
    ap.add_argument(
        "--manifest",
        default=None,
        help="Optional allow-list file (one `dataset/subject` per line). When set, ONLY these "
        "subjects are aggregated; missing entries fail loudly. Without it, every "
        "<dataset>_<subject>.json is discovered.",
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Explicit `dataset/subject` keys to drop (after manifest / discovery).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List the inclusion / drop set without writing output JSONs.",
    )
    args = ap.parse_args()

    per_subject_dir = Path(args.per_subject_dir)
    discovered = _discover(per_subject_dir)
    if not discovered:
        raise SystemExit(f"No per-subject JSON found in {per_subject_dir}")

    if args.manifest:
        wanted = _load_manifest(Path(args.manifest))
        missing = sorted(wanted - set(discovered.keys()))
        if missing:
            raise SystemExit(f"Manifest missing per-subject JSONs: {missing}")
        selected_keys: List[str] = sorted(wanted)
    else:
        selected_keys = sorted(discovered.keys())

    excluded = set(args.exclude)
    bad_excludes = excluded - set(selected_keys)
    if bad_excludes:
        raise SystemExit(f"--exclude keys not in selection: {sorted(bad_excludes)}")
    final_keys = [k for k in selected_keys if k not in excluded]

    if args.manifest:
        not_in_manifest = sorted(set(discovered.keys()) - set(selected_keys))
        if not_in_manifest:
            print(f"  ignored (not in manifest, n={len(not_in_manifest)}): {not_in_manifest}")
    if excluded:
        print(f"  excluded by --exclude: {sorted(excluded)}")
    print(f"Including {len(final_keys)} subjects:")
    for k in final_keys:
        print(f"  + {k}")

    if args.dry_run:
        print("[dry-run] no output written")
        return

    subject_results: Dict[str, Dict[str, Any]] = {}
    for key in final_keys:
        f = discovered[key]
        try:
            data = json.loads(f.read_text())
        except Exception as exc:
            raise SystemExit(f"Failed to parse {f}: {exc}")
        subject_results[key] = data

    cohort = summarize_propagation_cohort(subject_results)

    _save(subject_results, Path(args.out_summary))
    _save(cohort, Path(args.out_cohort))

    print(f"Wrote {args.out_summary}")
    print(f"Wrote {args.out_cohort}")
    print(f"Cohort n_subjects = {cohort.get('n_subjects')}")


if __name__ == "__main__":
    main()
