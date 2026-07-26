#!/usr/bin/env python3
"""Build the v0.4 34-patient all-interictal masked-rank dataset.

This builder deliberately does not open any ictal target table or cache.
The 13 future target patients are recorded only as routing metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_topic5_interictal_operator_dataset import build_subject  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eligible_subjects(cfg: dict) -> tuple[list[str], pd.DataFrame]:
    audit_path = ROOT / cfg["cohort"]["audit"]
    audit = pd.read_csv(audit_path)
    eligible_column = str(cfg["cohort"]["eligible_column"])
    if eligible_column not in audit.columns:
        raise RuntimeError(f"cohort audit missing {eligible_column}")
    eligible = audit[audit[eligible_column].astype(bool)].copy()
    subjects = sorted(eligible["sid"].astype(str).unique())
    expected = int(cfg["cohort"]["expected_subjects"])
    if len(subjects) != expected:
        raise RuntimeError(f"expected {expected} eligible subjects, got {len(subjects)}")
    counts = eligible["dataset"].astype(str).value_counts().to_dict()
    expected_counts = {
        "epilepsiae": int(cfg["cohort"]["expected_epilepsiae"]),
        "yuquan": int(cfg["cohort"]["expected_yuquan"]),
    }
    if counts != expected_counts:
        raise RuntimeError(f"dataset count drift: expected {expected_counts}, got {counts}")
    return subjects, eligible


def _future_target_subjects(cfg: dict) -> list[str]:
    path = ROOT / cfg["cohort"]["candidate_prefix_attrition"]
    frame = pd.read_csv(path, usecols=["subject", "prefix_field_pass"])
    passed = frame["prefix_field_pass"].astype(str).str.lower().isin(("true", "1", "yes"))
    subjects = sorted(frame.loc[passed, "subject"].astype(str).unique())
    expected = int(cfg["cohort"]["candidate_subjects_expected"])
    if len(subjects) != expected:
        raise RuntimeError(f"future target routing drift: expected {expected}, got {len(subjects)}")
    return subjects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_interictal_rank_distribution_v0_4.yaml",
    )
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text())
    out_dir = (
        args.out_dir
        if args.out_dir is not None and args.out_dir.is_absolute()
        else ROOT / args.out_dir
        if args.out_dir is not None
        else ROOT / cfg["outputs"]["dataset"]
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects, cohort_rows = _eligible_subjects(cfg)
    future_targets = _future_target_subjects(cfg)
    if args.subjects:
        requested = [str(value) for value in args.subjects]
        unknown = sorted(set(requested) - set(subjects))
        if unknown:
            raise RuntimeError(f"requested subjects outside frozen 34-person cohort: {unknown}")
        subjects = requested

    rows = []
    for index, subject in enumerate(subjects, start=1):
        print(f"[v0.4 phase0 {index}/{len(subjects)}] {subject}", flush=True)
        try:
            row = build_subject(
                subject,
                cfg,
                set(future_targets),
                out_dir,
                overwrite=args.overwrite,
                force_all_interictal=True,
            )
        except Exception as exc:
            row = {
                "dataset": subject.split("_", 1)[0],
                "subject": subject,
                "candidate_target_patient": subject in set(future_targets),
                "status": "failed",
                "reason": f"{type(exc).__name__}:{str(exc)[:400]}",
            }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "subject_audit.csv", index=False)
    ok = audit[audit["status"].astype(str) == "ok"]
    full_run = len(subjects) == int(cfg["cohort"]["expected_subjects"])
    cohort_pass = bool(
        full_run
        and len(ok) == int(cfg["cohort"]["expected_subjects"])
        and ok["subject"].nunique() == int(cfg["cohort"]["expected_subjects"])
    )
    manifest = {
        "contract": cfg["contract"]["name"],
        "contract_version": cfg["contract"]["version"],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "source_spec": cfg["contract"]["source_spec"],
        "source_spec_sha256": _sha256(ROOT / cfg["contract"]["source_spec"]),
        "source_cohort_audit": cfg["cohort"]["audit"],
        "source_cohort_audit_sha256": _sha256(ROOT / cfg["cohort"]["audit"]),
        "n_subjects_requested": int(len(subjects)),
        "n_subjects_ok": int(len(ok)),
        "n_subjects_failed": int(len(audit) - len(ok)),
        "n_events_ok": int(ok["n_events"].sum()) if "n_events" in ok else 0,
        "cohort_subjects": sorted(ok["subject"].astype(str).tolist()),
        "future_ictal_target_subjects": future_targets,
        "future_ictal_target_subjects_in_cohort": sorted(
            set(future_targets) & set(ok["subject"].astype(str))
        ),
        "full_source_pool_requested": full_run,
        "cohort_pass": cohort_pass,
        "split_contract": "chronological first 80% calibration, last 20% held out",
        "event_pool": "all fail-closed definite-interictal blocks for all 34 subjects",
        "recurrence_time": "within-one-interictal-event recruitment pseudo-time only",
        "target_values_read": False,
        "ab_or_kmeans_labels_read": False,
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    done = {
        "status": "complete" if cohort_pass else "failed",
        "cohort_pass": cohort_pass,
        "n_subjects_ok": int(len(ok)),
        "n_events_ok": manifest["n_events_ok"],
        "ictal_target_opened": False,
    }
    (out_dir / "PHASE0_DONE.json").write_text(
        json.dumps(done, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not cohort_pass:
        raise SystemExit("v0.4 Phase 0 cohort gate failed")
    print(json.dumps(done, indent=2), flush=True)


if __name__ == "__main__":
    main()
