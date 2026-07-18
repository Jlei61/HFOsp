#!/usr/bin/env python3
"""Fail-closed batch driver for two-dimensional shared-gradient Fig3-B material.

The canonical denominator is not "records containing shared keys".  A subject
must have a fingerprint-valid frozen record, both ``shared_a/shared_b``, and
``geometry_2d_supported=true`` with at least two shafts and effective rank >= 2
for both axes.  Missing derived seizure input is reported separately from a
missing seizure inventory.  ``own_a/own_b`` are never a fallback.

Every run writes producer and renderer outputs only under
``runs/<run_id>/artifacts/``. Default full runs validate that immutable artifact
set, then replace the canonical CSV/JSON and finally the manifest as the sole
completion pointer. Explicit ``--subjects`` and interrupted runs never overwrite
canonical artifacts, indexes, or the manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_topic5_signed_broadband_similarity import (  # noqa: E402
    _load_frozen_shared,
)
from scripts.plot_topic5_signed_broadband_similarity_timecourse import (  # noqa: E402
    _eligibility_status,
)
from src.topic5_template_axis_field import (  # noqa: E402
    scorers_from_interictal_record,
)

FROZEN_FIELD_DIR = (
    ROOT / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
)
FIELD_DIR = ROOT / "results/topic5_ictal_recruitment/field_dynamics_signed"
PAPER_DIR = ROOT / "results/paper-ready-figure/fig3_peri_onset_field_similarity"
FIG_DIR = PAPER_DIR / "figures"
INDEX_CSV = PAPER_DIR / "fig3_peri_onset_subject_index.csv"
INDEX_JSON = PAPER_DIR / "fig3_peri_onset_subject_index.json"
MANIFEST_JSON = PAPER_DIR / "fig3_peri_onset_run_manifest.json"
RUN_DIR = PAPER_DIR / "runs"

TIMECOURSE_SCRIPT = ROOT / "scripts/plot_topic5_signed_broadband_similarity_timecourse.py"
PAPER_SCRIPT = ROOT / "scripts/paper_figures/plot_fig3_peri_onset_field_similarity.py"

TIMECOURSE_ARGS = [
    "--start-sec", "-120", "--stop-sec", "20",
    "--band-lo", "1", "--band-hi", "150",
    "--window-sec", "10", "--step-sec", "2",
]
TIMECOURSE_SUMMARY = (
    "{sid}_signed_broadband_1_150Hz_similarity_timecourse_"
    "m120_p20_10s_step2s_summary.json"
)
TIMECOURSE_PER_SEIZURE = (
    "{sid}_signed_broadband_1_150Hz_similarity_timecourse_"
    "m120_p20_10s_step2s_per_seizure.csv"
)
PAPER_SUMMARY = "{sid}_peri_onset_field_similarity_paper_ready_summary.json"

GENERATED_STATUSES = {"complete_ok", "partial_ok", "severely_partial"}
INDEX_COLUMNS = [
    "subject",
    "status",
    "reason_code",
    "drop_reason",
    "geometry_quality_tier",
    "inventory_n",
    "eligible_cache_path",
    "n_eligible",
    "n_seizures",
    "n_seizure_drops",
    "coverage_fraction",
    "n_windows",
    "maxAB_median_of_window_medians",
    "maxAB_median_of_window_variances",
    "signed_A_median_of_window_medians",
    "signed_B_median_of_window_medians",
    "source_csv",
    "figure_png",
    "figure_pdf",
]


def _record_subject_id(record: dict) -> str:
    return f"{record.get('dataset')}_{record.get('subject')}"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _frozen_audit() -> list[dict]:
    rows = []
    for fp in sorted(FROZEN_FIELD_DIR.glob("*.json")):
        record = json.loads(fp.read_text())
        ds_sid = _record_subject_id(record)
        identity_ok = ds_sid == fp.stem
        models = (record.get("interictal_field") or {}).get("field_models") or {}
        has_shared = {"shared_a", "shared_b"}.issubset(models)
        fingerprint_valid = False
        fingerprint_error = None
        if has_shared and identity_ok:
            try:
                scorers_from_interictal_record(record)
                fingerprint_valid = True
            except Exception as exc:
                fingerprint_error = f"{type(exc).__name__}: {exc}"
        pair = record.get("axis_pair") or {}
        axes = [pair.get("axis_a") or {}, pair.get("axis_b") or {}]
        geometry_2d = (
            pair.get("geometry_2d_supported") is True
            and all(int(axis.get("n_shafts", 0)) >= 2 for axis in axes)
            and all(int(axis.get("effective_rank", 0)) >= 2 for axis in axes)
        )
        rows.append({
            "subject": ds_sid,
            "path": _display_path(fp),
            "identity_valid": identity_ok,
            "shared_pair": has_shared,
            "fingerprint_valid": fingerprint_valid,
            "fingerprint_error": fingerprint_error,
            "geometry_2d_supported": geometry_2d,
            "fingerprint_sha256": (
                record.get("interictal_field") or {}
            ).get("fingerprint_sha256"),
        })
    return rows


def _discover_subjects(frozen_audit: list[dict] | None = None) -> list[str]:
    audit = _frozen_audit() if frozen_audit is None else frozen_audit
    subjects = sorted(
        row["subject"]
        for row in audit
        if row["identity_valid"]
        and row["shared_pair"]
        and row["fingerprint_valid"]
        and row["geometry_2d_supported"]
    )
    if not subjects:
        raise FileNotFoundError(
            "no fingerprint-valid, two-dimensional frozen records with "
            f"shared_a/shared_b under {FROZEN_FIELD_DIR}"
        )
    return subjects


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)


def _tail(text: str | None, n: int = 1) -> str:
    lines = [line for line in (text or "").strip().splitlines() if line.strip()]
    return "\n".join(lines[-n:]) if lines else ""


def _blank_record(ds_sid: str) -> dict:
    record = {column: "" for column in INDEX_COLUMNS}
    record["subject"] = ds_sid
    return record


def _process_subject(
    ds_sid: str,
    *,
    run_field_dir: Path,
    run_figure_dir: Path,
) -> dict:
    record = _blank_record(ds_sid)
    try:
        field_record, _shared = _load_frozen_shared(ds_sid)
    except Exception as exc:
        record.update({
            "status": "ineligible_geometry_or_field",
            "reason_code": "ineligible_geometry_or_field",
            "drop_reason": f"{type(exc).__name__}: {exc}",
            "_stage": "frozen_field",
        })
        return record

    pair = field_record["axis_pair"]
    record["geometry_quality_tier"] = (
        "strict_2d" if pair.get("strict_stability_pass") else "non_strict_2d"
    )
    eligibility = _eligibility_status(ds_sid)
    record["inventory_n"] = eligibility["inventory_n"]
    record["eligible_cache_path"] = eligibility["cache_path"] or ""
    record["n_eligible"] = len(eligibility["eligible_idxs"])
    if not eligibility["eligible_idxs"]:
        record.update({
            "status": "blocked_input",
            "reason_code": eligibility["reason_code"],
            "drop_reason": (
                f"{eligibility['reason_code']} "
                f"(inventory_n={eligibility['inventory_n']})"
            ),
            "_stage": "input_eligibility",
        })
        return record

    timecourse = _run([
        sys.executable,
        str(TIMECOURSE_SCRIPT),
        "--subject",
        ds_sid,
        *TIMECOURSE_ARGS,
        "--out-dir",
        str(run_field_dir),
    ])
    if timecourse.returncode != 0:
        record.update({
            "status": "processing_drop",
            "reason_code": "timecourse_failed",
            "drop_reason": (
                _tail(timecourse.stderr)
                or _tail(timecourse.stdout)
                or "timecourse failed (no output)"
            ),
            "_stage": "timecourse",
            "_detail": _tail(
                (timecourse.stdout or "") + "\n" + (timecourse.stderr or ""), 10
            ),
        })
        return record

    source_csv = run_field_dir / TIMECOURSE_PER_SEIZURE.format(sid=ds_sid)
    paper_run = _run([
        sys.executable,
        str(PAPER_SCRIPT),
        "--subject",
        ds_sid,
        "--source-csv",
        str(source_csv),
        "--out-dir",
        str(run_figure_dir),
    ])
    if paper_run.returncode != 0:
        record.update({
            "status": "processing_drop",
            "reason_code": "renderer_failed",
            "drop_reason": _tail(paper_run.stderr) or "paper figure failed (no output)",
            "_stage": "paper_figure",
            "_detail": _tail(
                (paper_run.stdout or "") + "\n" + (paper_run.stderr or ""), 10
            ),
        })
        return record

    paper_path = run_figure_dir / PAPER_SUMMARY.format(sid=ds_sid)
    if not paper_path.exists():
        record.update({
            "status": "processing_drop",
            "reason_code": "paper_summary_missing",
            "drop_reason": f"paper summary missing: {paper_path.name}",
            "_stage": "paper_figure",
        })
        return record

    paper = json.loads(paper_path.read_text())
    status = paper.get("coverage_status")
    if status not in GENERATED_STATUSES:
        raise RuntimeError(f"{ds_sid}: invalid coverage status {status}")
    readouts = paper["readouts"]
    record.update({
        "status": status,
        "reason_code": "",
        "n_eligible": paper["n_eligible_requested"],
        "n_seizures": paper["n_seizures"],
        "n_seizure_drops": paper["n_seizure_drops"],
        "coverage_fraction": paper["coverage_fraction"],
        "n_windows": paper["n_windows"],
        "maxAB_median_of_window_medians": (
            readouts["maxAB_abs"]["median_of_window_medians"]
        ),
        "maxAB_median_of_window_variances": (
            readouts["maxAB_abs"]["median_of_window_variances"]
        ),
        "signed_A_median_of_window_medians": (
            readouts["signed_A"]["median_of_window_medians"]
        ),
        "signed_B_median_of_window_medians": (
            readouts["signed_B"]["median_of_window_medians"]
        ),
        "source_csv": paper["source_csv"],
        "figure_png": paper["outputs"]["png"],
        "figure_pdf": paper["outputs"]["pdf"],
    })
    return record


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        delete=False,
    )
    handle.close()
    return Path(handle.name)


def _denominator_flow(frozen_audit: list[dict], records: list[dict]) -> dict:
    shared = [r for r in frozen_audit if r["identity_valid"] and r["shared_pair"]]
    fingerprint = [r for r in shared if r["fingerprint_valid"]]
    geometry = [r for r in fingerprint if r["geometry_2d_supported"]]
    return {
        "n_frozen_records": len(frozen_audit),
        "n_shared_pair": len(shared),
        "n_fingerprint_valid_shared": len(fingerprint),
        "n_geometry_2d_shared": len(geometry),
        "n_inventory_available": sum(
            int(record.get("inventory_n") or 0) > 0 for record in records
        ),
        "n_eligible_cache_ready": sum(
            int(record.get("n_eligible") or 0) > 0 for record in records
        ),
        "n_generated": sum(
            record.get("status") in GENERATED_STATUSES for record in records
        ),
        "n_complete_ok": sum(
            record.get("status") == "complete_ok" for record in records
        ),
        "n_partial_ok": sum(
            record.get("status") == "partial_ok" for record in records
        ),
        "n_severely_partial": sum(
            record.get("status") == "severely_partial" for record in records
        ),
        "n_blocked_input": sum(
            record.get("status") == "blocked_input" for record in records
        ),
        "geometry_excluded_subjects": [
            record["subject"]
            for record in fingerprint
            if not record["geometry_2d_supported"]
        ],
    }


def _index_payload(
    records: list[dict],
    *,
    run_id: str,
    run_complete: bool,
    planned_subjects: list[str],
    frozen_audit: list[dict],
    canonical_run: bool,
) -> dict:
    n_generated = sum(
        record.get("status") in GENERATED_STATUSES for record in records
    )
    return {
        "figure": "Fig3-B peri-onset field similarity — per-subject material pool",
        "generated_by": "scripts/paper_figures/run_fig3_peri_onset_all_subjects.py",
        "run_id": run_id,
        "run_complete": run_complete,
        "canonical_run": canonical_run,
        "artifact_publication": {
            "mode": "immutable_run_scoped_artifacts",
            "artifact_root": str(
                (RUN_DIR / run_id / "artifacts").relative_to(ROOT)
            ),
            "canonical_completion_pointer": str(MANIFEST_JSON.relative_to(ROOT)),
            "publish_order": "validate run artifacts, replace canonical index, replace canonical manifest last",
        },
        "planned_subjects": planned_subjects,
        "n_planned": len(planned_subjects),
        "n_processed_records": len(records),
        "tier": (
            "two-dimensional per-subject descriptive material; "
            "NOT a formal cohort statistic"
        ),
        "contract": {
            "band_hz": [1.0, 150.0],
            "time_range_sec": [-120.0, 20.0],
            "window_sec": 10.0,
            "step_sec": 2.0,
            "normalization": (
                "per-channel baseline robust-z (1-150 Hz summed spectrogram "
                "log power, notch-filtered input at 50/100/150/200 Hz; "
                "no extra FFT-bin line mask)"
            ),
            "panel_a": "raw shared-plane max(|r_A|, |r_B|) trajectory",
            "panel_b": "signed shared-plane template A/B polarity sidecar",
            "field_plane": "shared only",
            "field_scorers": ["shared_a", "shared_b"],
            "own_field_fallback": False,
            "geometry_2d_required": True,
        },
        "denominator_flow": _denominator_flow(frozen_audit, records),
        "n_subjects": len(records),
        "n_generated": n_generated,
        "n_not_generated": len(records) - n_generated,
        "subjects": records,
    }


def _write_index(
    records: list[dict],
    csv_path: Path,
    json_path: Path,
    *,
    run_id: str,
    run_complete: bool,
    planned_subjects: list[str],
    frozen_audit: list[dict],
    canonical_run: bool,
) -> dict:
    payload = _index_payload(
        records,
        run_id=run_id,
        run_complete=run_complete,
        planned_subjects=planned_subjects,
        frozen_audit=frozen_audit,
        canonical_run=canonical_run,
    )
    csv_tmp = _temporary_sibling(csv_path)
    json_tmp = _temporary_sibling(json_path)
    try:
        with csv_tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=INDEX_COLUMNS,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            for record in records:
                writer.writerow({
                    column: record.get(column, "") for column in INDEX_COLUMNS
                })
        json_tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        os.replace(csv_tmp, csv_path)
        os.replace(json_tmp, json_path)  # JSON is the completion marker
    finally:
        csv_tmp.unlink(missing_ok=True)
        json_tmp.unlink(missing_ok=True)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_payload(index_payload: dict) -> dict:
    artifacts = []
    for record in index_payload["subjects"]:
        if record.get("status") not in GENERATED_STATUSES:
            continue
        source_csv = ROOT / record["source_csv"]
        source_summary = source_csv.with_name(
            source_csv.name.replace("_per_seizure.csv", "_summary.json")
        )
        source_aggregate = source_csv.with_name(
            source_csv.name.replace("_per_seizure.csv", "_aggregate.csv")
        )
        figure_png = ROOT / record["figure_png"]
        paper_summary = figure_png.with_name(
            PAPER_SUMMARY.format(sid=record["subject"])
        )
        paths = (
            ("source_per_seizure_csv", source_csv),
            ("source_aggregate_csv", source_aggregate),
            ("source_summary", source_summary),
            ("figure_png", figure_png),
            ("figure_pdf", ROOT / record["figure_pdf"]),
            ("paper_summary", paper_summary),
        )
        for role, artifact in paths:
            if not artifact.exists():
                raise FileNotFoundError(artifact)
            artifacts.append({
                "subject": record["subject"],
                "role": role,
                "path": str(artifact.relative_to(ROOT)),
                "size_bytes": artifact.stat().st_size,
                "sha256": _sha256(artifact),
            })
    return {
        "contract": "fig3_peri_onset_shared_gradient_2d_v1",
        "run_id": index_payload["run_id"],
        "run_complete": index_payload["run_complete"],
        "canonical_run": index_payload["canonical_run"],
        "artifact_publication": index_payload["artifact_publication"],
        "denominator_flow": index_payload["denominator_flow"],
        "artifacts": artifacts,
    }


def _write_manifest(index_payload: dict, path: Path) -> None:
    manifest = _manifest_payload(index_payload)
    tmp = _temporary_sibling(path)
    try:
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help=(
            "explicit subject list; writes only a run-scoped progress/final index "
            "and never overwrites the canonical index"
        ),
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="run subjects without writing progress or canonical indexes",
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    frozen_audit = _frozen_audit()
    canonical_run = args.subjects is None
    subjects = (
        _discover_subjects(frozen_audit)
        if canonical_run
        else list(dict.fromkeys(args.subjects or []))
    )
    if not subjects:
        raise ValueError("no subjects requested")

    run_id = args.run_id or _run_id()
    progress_dir = RUN_DIR / run_id
    run_artifact_dir = progress_dir / "artifacts"
    run_field_dir = run_artifact_dir / "field_dynamics_signed"
    run_figure_dir = run_artifact_dir / "figures"
    progress_csv = progress_dir / "fig3_peri_onset_subject_index_progress.csv"
    progress_json = progress_dir / "fig3_peri_onset_subject_index_progress.json"
    run_index_csv = progress_dir / "fig3_peri_onset_subject_index.csv"
    run_index_json = progress_dir / "fig3_peri_onset_subject_index.json"
    run_manifest_json = progress_dir / "fig3_peri_onset_run_manifest.json"
    print(
        f"processing {len(subjects)} subject(s); run_id={run_id}; "
        f"canonical_run={canonical_run}",
        flush=True,
    )

    records: list[dict] = []
    for index, ds_sid in enumerate(subjects, start=1):
        print(f"[{index}/{len(subjects)}] {ds_sid} ...", flush=True)
        record = _process_subject(
            ds_sid,
            run_field_dir=run_field_dir,
            run_figure_dir=run_figure_dir,
        )
        records.append(record)
        if record["status"] in GENERATED_STATUSES:
            print(
                f"    {record['status']} n={record['n_seizures']}/"
                f"{record['n_eligible']} maxAB="
                f"{record['maxAB_median_of_window_medians']:.4f}",
                flush=True,
            )
        else:
            print(
                f"    {record['status']} [{record.get('_stage', '?')}] "
                f"{record['drop_reason']}",
                flush=True,
            )
        if not args.no_index:
            _write_index(
                records,
                progress_csv,
                progress_json,
                run_id=run_id,
                run_complete=False,
                planned_subjects=subjects,
                frozen_audit=frozen_audit,
                canonical_run=False,
            )

    n_generated = sum(
        record["status"] in GENERATED_STATUSES for record in records
    )
    if not args.no_index:
        _write_index(
            records,
            progress_csv,
            progress_json,
            run_id=run_id,
            run_complete=True,
            planned_subjects=subjects,
            frozen_audit=frozen_audit,
            canonical_run=False,
        )
        if canonical_run:
            payload = _write_index(
                records,
                run_index_csv,
                run_index_json,
                run_id=run_id,
                run_complete=True,
                planned_subjects=subjects,
                frozen_audit=frozen_audit,
                canonical_run=True,
            )
            _write_manifest(payload, run_manifest_json)
            # Run artifacts are immutable.  The top-level manifest is replaced
            # last and is the sole canonical completion pointer.
            _write_index(
                records,
                INDEX_CSV,
                INDEX_JSON,
                run_id=run_id,
                run_complete=True,
                planned_subjects=subjects,
                frozen_audit=frozen_audit,
                canonical_run=True,
            )
            _write_manifest(payload, MANIFEST_JSON)

    print(
        f"DONE: {n_generated}/{len(records)} generated; "
        f"{len(records) - n_generated} not generated",
        flush=True,
    )
    if not args.no_index:
        print(f"run index: {progress_json}", flush=True)
        if canonical_run:
            print(f"run manifest: {run_manifest_json}", flush=True)
            print(f"canonical index: {INDEX_JSON}", flush=True)
            print(f"manifest: {MANIFEST_JSON}", flush=True)


if __name__ == "__main__":
    main()
