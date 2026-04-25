"""Cohort-level audit of the Yuquan 24-subject same-source lagPat contract.

Read-only. Aggregates the per-subject `summary.json` / `manifest.json` written
by `scripts/run_yuquan_lagpat_backfill.py` plus inputs from `data_root` and
`results/hfo_detection/` into a single cohort-level audit report.

Key outputs (per `--out-dir`):
- `cohort_audit.json`  : machine-readable, full per-subject + cohort rollup
- `cohort_audit.md`    : human-readable acceptance report

Audit dimensions (one row per subject in the markdown):
  - in_yuquan_same_source_24  : sourced from constant
  - has_summary / has_manifest
  - same_source_detector       : refine_npz mtime / size present
  - write_status               : ok / partial_ok / all_failed / no_inputs
  - blocks (total / written / skipped / error / missing_gpu_npz)
  - alias_collisions (total / max_in_picked)
  - start_time_validation_overall_pass
  - legacy_block_presence: n_legacy_present / n_legacy_absent / n_regressions / n_extras
  - pack_top_n_set         : True if subject has explicit cap (auditable special case)
  - median_n_participating / median_lag_span_ms (aggregate across blocks)

Cohort-level rollup checks (FAIL = red flag in markdown):
  - all 21 same-source subjects have summary + manifest
  - alias_collisions in picked = 0 across cohort (else red)
  - start_time_validation_overall_pass = True for every subject
  - no `legacy_block_presence_diff.regressions` for subjects that have legacy
    lagPat
  - any subject with `pack_top_n` explicitly set is listed under
    "explicit_pack_special_cases" so it cannot silently propagate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_yuquan_lagpat_backfill import (  # noqa: E402
    DATA_ROOT,
    DETECT_ROOT,
    RESULTS_ROOT,
    SUBJECT_PARAMS_PATH,
    YUQUAN_SAME_SOURCE_SUBJECTS,
    resolve_subject_pack_params,
)


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        return {"_load_error": repr(e)}


def audit_subject(subject: str, results_root: Path) -> Dict[str, object]:
    summary_path = results_root / subject / "summary.json"
    manifest_path = results_root / subject / "manifest.json"
    summary = _load_json(summary_path)
    manifest = _load_json(manifest_path)

    detect_dir = DETECT_ROOT / subject
    refine_npz = detect_dir / "_refineGpu.npz"
    raw_dir = DATA_ROOT / subject

    try:
        params = resolve_subject_pack_params(subject)
        params_resolvable = True
        params_error = None
    except Exception as e:
        params = {}
        params_resolvable = False
        params_error = repr(e)

    row: Dict[str, object] = {
        "subject": subject,
        "in_yuquan_same_source_24": subject in YUQUAN_SAME_SOURCE_SUBJECTS,
        "params_resolvable": params_resolvable,
        "params_error": params_error,
        "params_resolved": params,
        "pack_top_n_set": "pack_top_n" in params,
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "has_summary": summary is not None,
        "has_manifest": manifest is not None,
        "raw_dir": str(raw_dir),
        "raw_dir_exists": raw_dir.exists(),
        "refine_npz_exists": refine_npz.exists(),
        "_load_errors": {},
    }
    if isinstance(summary, dict) and "_load_error" in summary:
        row["_load_errors"]["summary"] = summary["_load_error"]
        summary = None
    if isinstance(manifest, dict) and "_load_error" in manifest:
        row["_load_errors"]["manifest"] = manifest["_load_error"]
        manifest = None

    if summary is not None:
        row.update({
            "schema_version": summary.get("schema_version"),
            "write_status": summary.get("write_status"),
            "n_blocks_total": summary.get("n_blocks_total"),
            "n_blocks_written": summary.get("n_blocks_written"),
            "n_blocks_skipped": summary.get("n_blocks_skipped"),
            "n_blocks_error": summary.get("n_blocks_error"),
            "n_blocks_missing_gpu_npz": summary.get("n_blocks_missing_gpu_npz"),
            "n_alias_collisions": summary.get("n_alias_collisions"),
            "n_alias_collisions_in_picked_max": summary.get("n_alias_collisions_in_picked_max"),
            "median_n_participating": summary.get("median_n_participating"),
            "median_lag_span_ms": summary.get("median_lag_span_ms"),
            "start_time_validation_overall_pass": summary.get("start_time_validation_overall_pass"),
        })
        diff = summary.get("legacy_block_presence_diff", {}) or {}
        row["legacy_block_presence"] = {
            "n_legacy_present": diff.get("n_legacy_present"),
            "n_legacy_absent": diff.get("n_legacy_absent"),
            "n_regressions": len(diff.get("regressions", []) or []),
            "n_extras_written": len(diff.get("extras_written", []) or []),
            "regressions": diff.get("regressions", []) or [],
            "extras_written": diff.get("extras_written", []) or [],
        }
        # Sanity: same-source detector means the manifest's refine_npz path
        # equals our DETECT_ROOT/<subject>/_refineGpu.npz. Catch any drift.
        if manifest is not None:
            ref_path = (manifest.get("refine_npz") or {}).get("path")
            row["same_source_detector"] = (
                ref_path == str(refine_npz) and refine_npz.exists()
            )
        else:
            row["same_source_detector"] = None
    else:
        row.update({
            "schema_version": None,
            "write_status": None,
            "n_blocks_total": None,
            "n_blocks_written": None,
            "n_blocks_skipped": None,
            "n_blocks_error": None,
            "n_blocks_missing_gpu_npz": None,
            "n_alias_collisions": None,
            "n_alias_collisions_in_picked_max": None,
            "median_n_participating": None,
            "median_lag_span_ms": None,
            "start_time_validation_overall_pass": None,
            "legacy_block_presence": None,
            "same_source_detector": None,
        })
    return row


def build_cohort_report(results_root: Path) -> Dict[str, object]:
    rows = [audit_subject(s, results_root) for s in YUQUAN_SAME_SOURCE_SUBJECTS]

    n_subjects = len(rows)
    n_with_summary = sum(1 for r in rows if r["has_summary"])
    n_with_manifest = sum(1 for r in rows if r["has_manifest"])
    n_same_source = sum(1 for r in rows if r.get("same_source_detector") is True)
    n_st_pass = sum(1 for r in rows if r.get("start_time_validation_overall_pass") is True)
    n_alias_red = sum(
        1 for r in rows
        if r.get("n_alias_collisions_in_picked_max") not in (None, 0)
    )
    n_regressions_red = sum(
        1 for r in rows
        if r.get("legacy_block_presence", {})
        and (r["legacy_block_presence"].get("n_regressions") or 0) > 0
    )
    pack_top_n_subjects = [r["subject"] for r in rows if r.get("pack_top_n_set")]

    n_blocks_total = sum((r.get("n_blocks_total") or 0) for r in rows)
    n_blocks_written = sum((r.get("n_blocks_written") or 0) for r in rows)
    n_blocks_skipped = sum((r.get("n_blocks_skipped") or 0) for r in rows)
    n_blocks_error = sum((r.get("n_blocks_error") or 0) for r in rows)
    n_blocks_missing = sum((r.get("n_blocks_missing_gpu_npz") or 0) for r in rows)

    cohort_pass = (
        n_with_summary == n_subjects
        and n_with_manifest == n_subjects
        and n_same_source == n_subjects
        and n_st_pass == n_subjects
        and n_alias_red == 0
        and n_regressions_red == 0
    )

    return {
        "schema_version": "yuquan_lagpat_cohort_audit_v1_2026Q2",
        "cohort_size": n_subjects,
        "cohort_subjects": list(YUQUAN_SAME_SOURCE_SUBJECTS),
        "results_root": str(results_root),
        "subject_params_path": str(SUBJECT_PARAMS_PATH),
        "data_root": str(DATA_ROOT),
        "detect_root": str(DETECT_ROOT),
        "rollup": {
            "n_subjects": n_subjects,
            "n_with_summary": n_with_summary,
            "n_with_manifest": n_with_manifest,
            "n_same_source_detector": n_same_source,
            "n_start_time_validation_pass": n_st_pass,
            "n_subjects_with_alias_collisions_in_picked": n_alias_red,
            "n_subjects_with_legacy_block_regressions": n_regressions_red,
            "explicit_pack_special_cases": pack_top_n_subjects,
            "n_blocks_total": n_blocks_total,
            "n_blocks_written": n_blocks_written,
            "n_blocks_skipped": n_blocks_skipped,
            "n_blocks_error": n_blocks_error,
            "n_blocks_missing_gpu_npz": n_blocks_missing,
            "cohort_pass": cohort_pass,
        },
        "subjects": rows,
    }


def render_markdown(report: Dict[str, object]) -> str:
    rollup = report["rollup"]
    lines: List[str] = []
    lines.append("# Yuquan same-source lagPat cohort audit")
    lines.append("")
    lines.append(f"- schema: `{report['schema_version']}`")
    lines.append(f"- cohort size: {report['cohort_size']}")
    lines.append(f"- results root: `{report['results_root']}`")
    lines.append(f"- subject params: `{report['subject_params_path']}`")
    lines.append(f"- detect root: `{report['detect_root']}`")
    lines.append(f"- data root: `{report['data_root']}`")
    lines.append("")
    flag = "PASS" if rollup["cohort_pass"] else "FAIL"
    lines.append(f"## Cohort verdict: **{flag}**")
    lines.append("")
    lines.append("| check | value | required |")
    lines.append("|---|---|---|")
    lines.append(f"| subjects with `summary.json` | {rollup['n_with_summary']} | {rollup['n_subjects']} |")
    lines.append(f"| subjects with `manifest.json` | {rollup['n_with_manifest']} | {rollup['n_subjects']} |")
    lines.append(f"| same-source detector verified | {rollup['n_same_source_detector']} | {rollup['n_subjects']} |")
    lines.append(f"| `start_time` validation pass | {rollup['n_start_time_validation_pass']} | {rollup['n_subjects']} |")
    lines.append(f"| subjects w/ alias collision in picked | {rollup['n_subjects_with_alias_collisions_in_picked']} | 0 |")
    lines.append(f"| subjects w/ legacy block regressions | {rollup['n_subjects_with_legacy_block_regressions']} | 0 |")
    lines.append("")
    lines.append("## Block totals")
    lines.append("")
    lines.append(f"- total blocks: {rollup['n_blocks_total']}")
    lines.append(f"- written:      {rollup['n_blocks_written']}")
    lines.append(f"- skipped:      {rollup['n_blocks_skipped']}")
    lines.append(f"- error:        {rollup['n_blocks_error']}")
    lines.append(f"- missing gpu:  {rollup['n_blocks_missing_gpu_npz']}")
    lines.append("")
    lines.append("## Explicit pack-stage special cases")
    lines.append("")
    if rollup["explicit_pack_special_cases"]:
        for s in rollup["explicit_pack_special_cases"]:
            lines.append(f"- `{s}` (pack_top_n set in config; see `docs/archive/yuquan_lagpat/`)")
    else:
        lines.append("- (none — all subjects use the uniform config rule)")
    lines.append("")
    lines.append("## Per-subject status")
    lines.append("")
    lines.append(
        "| subject | write | blocks ok/total | skip | err | miss | alias_in_picked | st_ok | "
        "legacy present | regressions | extras | top_n |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in report["subjects"]:
        bp = r.get("legacy_block_presence") or {}
        top_n = r.get("params_resolved", {}).get("pack_top_n", "")
        lines.append(
            f"| `{r['subject']}` "
            f"| {r.get('write_status') or '-'} "
            f"| {r.get('n_blocks_written')}/{r.get('n_blocks_total')} "
            f"| {r.get('n_blocks_skipped')} "
            f"| {r.get('n_blocks_error')} "
            f"| {r.get('n_blocks_missing_gpu_npz')} "
            f"| {r.get('n_alias_collisions_in_picked_max')} "
            f"| {r.get('start_time_validation_overall_pass')} "
            f"| {bp.get('n_legacy_present')} "
            f"| {bp.get('n_regressions')} "
            f"| {bp.get('n_extras_written')} "
            f"| {top_n} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Cohort-level Yuquan lagPat contract audit")
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(RESULTS_ROOT),
        help="Directory holding per-subject summary.json / manifest.json. "
             "Defaults to results/lagpat_backfill (the production location).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(
            REPO_ROOT / "results" / "lagpat_backfill" / "_audit" / "cohort_audit"
        ),
        help="Where to write `cohort_audit.json` and `cohort_audit.md`. "
             "Defaults to results/lagpat_backfill/_audit/cohort_audit so all "
             "lagPat audit outputs live under one root.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_cohort_report(results_root)

    json_path = out_dir / "cohort_audit.json"
    md_path = out_dir / "cohort_audit.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    md_path.write_text(render_markdown(report))

    rollup = report["rollup"]
    print(f"wrote: {json_path}")
    print(f"wrote: {md_path}")
    print(
        f"cohort_pass={rollup['cohort_pass']} "
        f"subjects={rollup['n_subjects']} "
        f"with_summary={rollup['n_with_summary']} "
        f"with_manifest={rollup['n_with_manifest']} "
        f"st_pass={rollup['n_start_time_validation_pass']} "
        f"alias_red={rollup['n_subjects_with_alias_collisions_in_picked']} "
        f"regression_red={rollup['n_subjects_with_legacy_block_regressions']}"
    )
    return 0 if rollup["cohort_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
