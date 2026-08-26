#!/usr/bin/env python3
"""Confirm the median-interval caliper actually ran, rather than trusting that it did.

A producer stage that silently skipped its work is exactly how the first attempt at
this leg failed, so the evidence is checked rather than assumed.
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

from _common import OUTPUT_ROOT, atomic_write_json  # noqa: E402

OUT = OUTPUT_ROOT / "seizure_link_preictal"


#: below this share of hard-calipered seizures the set is a mixture of matched and
#: fallback pairs, and a downstream contrast cannot be read as balanced
CALIPER_MAJORITY = 0.9


def _verdict(matched: int, fallback: int, missing: int) -> str:
    if matched + fallback + missing == 0:
        return "NO_SEIZURE_RECORDS_FOUND"
    if missing:
        return "CALIPER_NOT_APPLIED"
    share = matched / (matched + fallback)
    if share >= CALIPER_MAJORITY:
        return "CALIPER_APPLIED_AND_BALANCED"
    if matched:
        return "CALIPER_PARTIAL"
    return "CALIPER_INADMISSIBLE"


def main() -> None:
    files = sorted(glob.glob(str(OUT / "per_subject/*.json")))
    if not files:
        raise SystemExit("no per-subject output at all")
    newest = max(files, key=lambda f: Path(f).stat().st_mtime)
    age_hours = (time.time() - Path(newest).stat().st_mtime) / 3600.0

    matched = fallback = missing = 0
    reasons: dict[str, int] = {}
    for path in files:
        record = json.loads(Path(path).read_text())
        for block in (record.get("per_seizure") or []):
            if "median_iei_matched" not in block:
                missing += 1
                continue
            if block.get("median_iei_caliper_applied"):
                matched += 1
            else:
                fallback += 1
                reason = block.get("median_iei_fallback_reason") or "unstated"
                reasons[reason] = reasons.get(reason, 0) + 1

    summary = {
        "contract": "topic5_epi_prssm_caliper_verification",
        "n_per_subject_files": len(files),
        "newest_output_age_hours": round(age_hours, 2),
        "n_seizures_with_caliper_applied": matched,
        "n_seizures_on_fallback": fallback,
        "n_seizures_without_the_field": missing,
        "fallback_reasons": reasons,
        "share_with_caliper_applied": (matched / (matched + fallback)
                                       if matched + fallback else None),
        "verdict": _verdict(matched, fallback, missing),
        "verdict_note": "ACTIVE was fail-open: it fired whenever the field existed, "
                        "whether or not the balance was achieved, so a run in which "
                        "half the seizures fell back to the soft cost still read as "
                        "'the caliper is on'. The verdict now separates a set that was "
                        "actually balanced from one that mostly was not.",
    }
    atomic_write_json(OUT / "CALIPER_VERIFICATION.json", summary)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    # PARTIAL is not a pass: it means the reported set mixes balanced pairs with
    # pairs whose balance was abandoned, and no downstream contrast can separate them
    if summary["verdict"] != "CALIPER_APPLIED_AND_BALANCED":
        raise SystemExit(f"caliper verification: {summary['verdict']} "
                         f"(share balanced = {summary['share_with_caliper_applied']})")


if __name__ == "__main__":
    main()
