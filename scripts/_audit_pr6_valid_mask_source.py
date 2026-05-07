#!/usr/bin/env python3
"""Audit PR-6 per-subject anchoring outputs for `valid_mask_source` provenance.

Cross-PR contract (`docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md`):

> Default `valid_mask=None` for full-data input restores the buggy
> "all channels valid" path — this is silent and only catchable by audit.

When PR-6 driver's `_resolve_cluster_data()` returns None (channel-name
mismatch / event-count drift), `valid_mask_source = "fallback_all_valid"`.
That path is the polluted one cross-PR contract warns about.  This audit
flags any Path D subject that fell into that branch — they must be
excluded from H1 reporting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PATH_D_SUBJECTS = {
    "zhangkexuan", "pengzihang", "songzishuo", "zhangbichen",
    "zhaochenxi", "zhaojinrui", "zhourongxuan",
}


def _name_to_subject(name: str) -> tuple[str, str]:
    stem = name.replace(".json", "")
    if "_" not in stem:
        return ("", stem)
    return tuple(stem.split("_", 1))  # type: ignore[return-value]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: _audit_pr6_valid_mask_source.py <per_subject_dir>", file=sys.stderr)
        return 2
    per_subject_dir = Path(sys.argv[1])
    if not per_subject_dir.exists():
        # BLOCKER mode: audit run with no per_subject dir means PR-6 driver
        # never ran (or wrote elsewhere). Don't silently pass — fail loudly.
        print(
            f"  ✗ per_subject dir does not exist: {per_subject_dir}",
            file=sys.stderr,
        )
        print(
            "  → PR-6 --per-subject must run before audit. Failing.",
            file=sys.stderr,
        )
        return 3

    files = sorted(per_subject_dir.glob("yuquan_*.json")) + sorted(
        per_subject_dir.glob("epilepsiae_*.json")
    )
    if not files:
        # BLOCKER mode: audit run with empty dir means PR-6 produced nothing
        # eligible. Don't silently pass.
        print(
            f"  ✗ no per-subject anchoring outputs in {per_subject_dir}",
            file=sys.stderr,
        )
        print(
            "  → Empty audit cannot guarantee no contamination. Failing.",
            file=sys.stderr,
        )
        return 3

    print(f"{'subject':<28s} | {'valid_mask_source':<22s} | {'h1_eligible':<11s} | path-D?")
    print("-" * 80)
    path_d_contaminated = []
    other_contaminated = []
    missing_field = []
    for f in files:
        ds, sub = _name_to_subject(f.name)
        d = json.load(f.open())
        audit = d.get("audit") or {}
        src = audit.get("valid_mask_source", "<missing>")
        h1 = audit.get("h1_primary_eligible", False)
        is_path_d = ds == "yuquan" and sub in PATH_D_SUBJECTS
        marker = "*" if is_path_d else " "
        print(f"{ds + '/' + sub:<28s} | {src:<22s} | {str(h1):<11s} | {marker}")
        if src == "fallback_all_valid":
            if is_path_d:
                path_d_contaminated.append(f"{ds}/{sub}")
            else:
                other_contaminated.append(f"{ds}/{sub}")
        elif src == "<missing>":
            missing_field.append(f"{ds}/{sub}")

    print()
    if path_d_contaminated:
        print("!!! Path D subjects with fallback_all_valid (cross-PR contract violation):")
        for s in path_d_contaminated:
            print(f"    - {s}")
        print("    These subjects MUST be excluded from H1 in archive §5.4.")
        return 1
    print("OK: No path-D subjects fell back to fallback_all_valid.")

    if other_contaminated:
        print()
        print(f"  Note: {len(other_contaminated)} non-path-D subject(s) on fallback path:")
        for s in other_contaminated:
            print(f"    - {s}")
        print("  These pre-date Path D and are documented elsewhere.")

    if missing_field:
        print()
        print(f"  Note: {len(missing_field)} subject(s) with missing valid_mask_source field:")
        for s in missing_field[:5]:
            print(f"    - {s}")
        if len(missing_field) > 5:
            print(f"    ... and {len(missing_field) - 5} more")
        print("  Likely from older PR-6 run pre-dating valid_mask_source field. Re-run --per-subject if needed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
