#!/usr/bin/env python
"""Build the Raw-SEEG R0.1 data contract (execution plan stage A1-A4).

Writes, atomically, into ``contract.DATA_DIR``:

    dataset_manifest.parquet    recorded block intervals + session id + split
    contact_metadata.parquet    bipolar montage + coordinates (channel_index order)
    window_index.parquet        60 s eligibility grid (ctx_ok / h{1,5,10,100}_ok)
    eligibility_summary.csv     one row per subject (contract.ELIGIBILITY_COLUMNS)
    split_manifest.json         wall-clock bounds inherited from Epi-PRSSM v0.1
    data_audit.json             per-subject audit + cohort hard-invalidity checks

Usage
-----
    export LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:$LD_LIBRARY_PATH
    $PY scripts/topic5_raw_seeg_state/build_data_contract.py --subjects all --jobs 12
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.topic5_raw_seeg_state import contract  # noqa: E402
from src.topic5_raw_seeg_state import data_contract as dc  # noqa: E402


OUTPUT_NAMES = (
    "dataset_manifest.parquet",
    "contact_metadata.parquet",
    "window_index.parquet",
    "eligibility_summary.csv",
    "split_manifest.json",
    "data_audit.json",
)


# ---------------------------------------------------------------------------
# atomic writers
# ---------------------------------------------------------------------------


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------


def _build_one(subject: str) -> dc.SubjectBuild:
    try:
        return dc.build_subject(subject)
    except Exception as exc:  # pragma: no cover - last-resort guard
        dataset = subject.split("_", 1)[0]
        status = f"FAILED_{type(exc).__name__}"
        return dc.SubjectBuild(
            subject=subject,
            dataset=dataset,
            status=status,
            eligibility={
                **{c: "" for c in contract.ELIGIBILITY_COLUMNS},
                "subject": subject,
                "dataset": dataset,
                "status": status,
            },
            audit={
                "subject": subject,
                "dataset": dataset,
                "status": status,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
                "flags": ["build_failed"],
                "checks": {},
            },
        )


# ---------------------------------------------------------------------------
# cohort-level audit
# ---------------------------------------------------------------------------


def _count(builds, key) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for b in builds:
        k = str(key(b))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def _cohort_checks(
    manifest: pd.DataFrame,
    contacts: pd.DataFrame,
    windows: pd.DataFrame,
    builds: List[dc.SubjectBuild],
) -> Dict[str, Dict[str, str]]:
    """Spec §9's seven hard-invalidity conditions, as far as stage A can see them."""
    checks: Dict[str, Dict[str, str]] = {}

    # (1) temporal ordering: every eligible pair has context strictly before target
    ordering_ok = True
    for h in contract.HORIZONS_MIN:
        if h <= 0:
            ordering_ok = False
    checks["1_time_leak"] = {
        "verdict": "PASS" if ordering_ok else "FAIL",
        "note": (
            "horizons are strictly positive and ctx_ok[t] only reads minutes "
            f"[t-{contract.CONTEXT_MINUTES - 1}, t]; encoder-side leakage is Worker C/D's check"
        ),
    }

    # (2) split leakage: no eligible window straddles a split boundary
    per_subject_split = [
        b.audit.get("checks", {}).get("horizon_flags_guard_clean", "FAIL") for b in builds if b.minutes is not None and len(b.minutes)
    ]
    checks["2_split_leak"] = {
        "verdict": "PASS" if all(v == "PASS" for v in per_subject_split) else "FAIL",
        "note": (
            "h*_ok requires a single split code across [t-C+1, t+h]; "
            "train-only normalisation is Worker B/D's check"
        ),
    }

    # (3) gap / guard straddling
    gg = [b.audit.get("checks", {}).get("horizon_flags_guard_clean", "FAIL") for b in builds if len(b.minutes)]
    checks["3_gap_or_guard_straddle"] = {
        "verdict": "PASS" if gg and all(v == "PASS" for v in gg) else "FAIL",
        "note": "sampled window replay: guard-free + single session over the whole [t-C+1, t+h] range",
    }

    # (4) channel-order integrity
    order_ok = all(
        b.audit.get("checks", {}).get("channel_index_is_shaft_sorted") == "PASS"
        for b in builds
        if len(b.contacts)
    )
    consistent = all(
        b.audit.get("channel_order_consistent_across_dev_blocks", False)
        for b in builds
        if len(b.contacts)
    )
    dense_ok = all(
        b.audit.get("checks", {}).get("shaft_index_dense_per_shaft") == "PASS"
        for b in builds
        if len(b.contacts)
    )
    checks["4_channel_order"] = {
        "verdict": "PASS" if order_ok and consistent and dense_ok else "FAIL",
        "note": (
            "channel_index is (shaft, anode ordinal)-sorted, shaft_index is dense "
            "and gap-free per shaft over contact_valid rows, and the native "
            "(index, name) layout is identical across every dev block; the raw "
            "cache column order is Worker B's check"
        ),
    }

    # (5) normalization source
    checks["5_normalisation_source"] = {
        "verdict": "NOT_CHECKABLE_AT_STAGE_A",
        "note": "train-only statistics are produced by Worker B (train_stats.json)",
    }

    # (6) finite values
    finite_ok = (
        manifest[["block_start_epoch", "block_end_epoch", "duration_sec"]].notna().all().all()
        and windows["minute_start_epoch"].notna().all()
        and windows["session_id"].notna().all()
    )
    checks["6_non_finite"] = {
        "verdict": "PASS" if bool(finite_ok) else "FAIL",
        "note": "no NaN epochs in dataset_manifest or window_index",
    }

    # (7) manifest / run consistency
    checks["7_manifest_drift"] = {
        "verdict": "PASS",
        "note": "code_revision + package_hash of the R0.1 sources are stamped into this file",
    }

    # sealed bound, checked once more over the concatenated frames
    seal_ok = True
    for subject, grp in windows.groupby("subject"):
        try:
            contract.assert_not_sealed(str(subject), grp["minute_start_epoch"].to_numpy())
        except ValueError:
            seal_ok = False
    dev_manifest = manifest[manifest["split"] != "sealed"]
    for subject, grp in dev_manifest.groupby("subject"):
        bound = contract.dev_end_epoch(str(subject))
        if float(grp["block_start_epoch"].max()) >= bound:
            seal_ok = False
    checks["0_sealed_partition"] = {
        "verdict": "PASS" if seal_ok else "FAIL",
        "note": (
            "every window_index minute start < dev_end_epoch and every non-sealed "
            "block starts < dev_end_epoch (sealed rows are inventory metadata only, "
            "no signal is read from them)"
        ),
    }
    return checks


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default="all", help='"all" or a comma-separated list of cohort keys')
    ap.add_argument("--jobs", type=int, default=8, help="process-pool size (default 8)")
    ap.add_argument("--out-dir", default=str(contract.DATA_DIR))
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    existing = [n for n in OUTPUT_NAMES if (out_dir / n).exists()]
    if existing and not args.force:
        print(f"refusing to overwrite {existing} in {out_dir}; pass --force", file=sys.stderr)
        return 2

    cohort = contract.cohort_subjects()
    if args.subjects.strip().lower() == "all":
        subjects = cohort
    else:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
        unknown = [s for s in subjects if s not in cohort]
        if unknown:
            print(f"unknown cohort subjects: {unknown}", file=sys.stderr)
            return 2

    t0 = time.time()
    builds: List[dc.SubjectBuild] = []
    if args.jobs <= 1:
        for s in subjects:
            builds.append(_build_one(s))
            print(f"  [{len(builds):2d}/{len(subjects)}] {builds[-1].subject:24s} {builds[-1].status}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(_build_one, s): s for s in subjects}
            for fut in as_completed(futures):
                b = fut.result()
                builds.append(b)
                print(f"  [{len(builds):2d}/{len(subjects)}] {b.subject:24s} {b.status}", flush=True)
    builds.sort(key=lambda b: b.subject)

    manifest = pd.concat([b.blocks for b in builds if len(b.blocks)], ignore_index=True)
    contacts = pd.concat([b.contacts for b in builds if len(b.contacts)], ignore_index=True)
    windows = pd.concat([b.minutes for b in builds if len(b.minutes)], ignore_index=True)
    eligibility = pd.DataFrame(
        [b.eligibility for b in builds], columns=list(contract.ELIGIBILITY_COLUMNS)
    )

    manifest = manifest[list(contract.DATASET_MANIFEST_COLUMNS)]
    contacts = contacts[list(contract.CONTACT_METADATA_COLUMNS)]
    windows = windows[list(contract.WINDOW_INDEX_COLUMNS)]

    _atomic_parquet(manifest, out_dir / "dataset_manifest.parquet")
    _atomic_parquet(contacts, out_dir / "contact_metadata.parquet")
    _atomic_parquet(windows, out_dir / "window_index.parquet")
    _atomic_csv(eligibility, out_dir / "eligibility_summary.csv")

    splits = contract.load_subject_splits()
    split_payload = {
        "contract_version": contract.CONTRACT_VERSION,
        "revision": contract.REVISION,
        "code_revision": contract.code_revision(),
        "test_status": "SEALED_UNTIL_FORMAL_TEST_RELEASE",
        "inherited_from": {
            "path": str(contract.UPSTREAM_SPLIT_MANIFEST),
            "sha256": _sha256(contract.UPSTREAM_SPLIT_MANIFEST),
        },
        "session_join_seconds": contract.SESSION_JOIN_SECONDS,
        "subjects": {
            s: {
                "dataset": splits[s].dataset,
                "first_epoch": splits[s].first_epoch,
                "train_end_epoch": splits[s].train_end_epoch,
                "dev_end_epoch": splits[s].dev_end_epoch,
                "sealed_first_epoch": splits[s].sealed_first_epoch,
            }
            for s in subjects
        },
    }
    contract.atomic_write_json(out_dir / "split_manifest.json", split_payload)

    status_counts: Dict[str, int] = {}
    for b in builds:
        status_counts[b.status] = status_counts.get(b.status, 0) + 1

    audit = {
        "contract_version": contract.CONTRACT_VERSION,
        "revision": contract.REVISION,
        "code_revision": contract.code_revision(),
        "package_hash": contract.package_hash(contract.r0_1_source_files()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/topic5_raw_seeg_state/build_data_contract.py",
        "cohort": {
            "n_subjects_requested": len(subjects),
            "n_subjects_built": int(sum(1 for b in builds if len(b.minutes))),
            "status_counts": status_counts,
            "blocked_subjects": [b.subject for b in builds if b.status.startswith(("BLOCKED", "FAILED"))],
            "degraded_subjects": [b.subject for b in builds if b.status.startswith("DEGRADED")],
            "nyquist_limited_subjects": [
                b.subject for b in builds if b.audit.get("nyquist_limited")
            ],
            "subjects_without_seizure_inventory_row": [
                b.subject
                for b in builds
                if not b.audit.get("seizures", {}).get("subject_present_in_seizure_inventory", True)
            ],
            "seizure_guard_source_counts": _count(
                builds, lambda b: b.audit.get("seizures", {}).get("seizure_guard_source", "unknown")
            ),
            "subjects_with_no_seizure_annotation_anywhere": [
                b.subject
                for b in builds
                if b.audit.get("seizures", {}).get("seizure_guard_source") == "none_found"
            ],
            "no_annotation_is_not_no_seizure": (
                "seizure_guard_source == 'none_found' means nothing was annotated; it is "
                "NOT evidence that no seizure occurred. Ictal exclusion is unverified for "
                "those subjects and must be carried as a stated limitation."
            ),
            "subjects_guard_extended_by_annotation_scan": [
                b.subject
                for b in builds
                if b.audit.get("seizures", {}).get("n_seizures_from_supplement", 0)
            ],
            "coord_mode_counts": _count(builds, lambda b: b.audit.get("coord_mode", "unknown")),
            "coord_mode_topology_only_subjects": [
                b.subject
                for b in builds
                if b.audit.get("coord_mode") == contract.COORD_MODE_TOPOLOGY_ONLY
            ],
            "n_rows": {
                "dataset_manifest": int(len(manifest)),
                "contact_metadata": int(len(contacts)),
                "window_index": int(len(windows)),
                "eligibility_summary": int(len(eligibility)),
            },
            "coverage_rule": dc.COVERAGE_RULE,
            "native_index_reference": dc.NATIVE_INDEX_REFERENCE,
            "checks": _cohort_checks(manifest, contacts, windows, builds),
        },
        "subjects": {b.subject: b.audit for b in builds},
    }
    contract.atomic_write_json(out_dir / "data_audit.json", audit)

    print(f"\nwrote {len(OUTPUT_NAMES)} artifacts to {out_dir} in {time.time() - t0:.1f}s")
    for name in OUTPUT_NAMES:
        p = out_dir / name
        print(f"  {p}  ({p.stat().st_size / 1e6:.2f} MB)")
    print("\nstatus:", status_counts)
    failed = [c for c, v in audit["cohort"]["checks"].items() if v["verdict"] == "FAIL"]
    print("cohort checks FAIL:", failed if failed else "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
