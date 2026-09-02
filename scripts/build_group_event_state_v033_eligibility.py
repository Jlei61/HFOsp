#!/usr/bin/env python3
"""Build endpoint/horizon estimability from real coverage and frozen assay power."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v032_eval.contract import atomic_json, load_eval_config  # noqa: E402
from src.topic5_group_event_state.v032_eval.timeline import load_eval_timeline  # noqa: E402
from src.topic5_group_event_state.v033_evaluator import eligibility as G  # noqa: E402


def _one(job: tuple[str, str, dict]) -> tuple[dict, list[dict]]:
    subject, config_path, requirements = job
    cfg = load_eval_config(Path(config_path))
    tl = load_eval_timeline(subject, cfg)
    support = G.subject_support(tl)
    tuple_requirements = {(str(k).split("|", 1)[0], int(str(k).split("|", 1)[1])): v
                          for k, v in requirements.items()}
    return support, G.eligibility_rows(subject, support, tuple_requirements)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "config/topic5_group_event_state_v032_eval.json")
    ap.add_argument("--power", type=Path,
                    default=ROOT / "results/group_event_state/v0_3_3/evaluator_assay/d0_d4_power_curve.json")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/group_event_state/v0_3_3/evaluator_assay/eligibility_by_endpoint_horizon.json")
    ap.add_argument("--shared", type=Path,
                    default=Path("/data/hfosp_group_event_state_v0_3_3/shared/eligibility/eligibility_by_endpoint_horizon.json"))
    args = ap.parse_args()
    cfg = load_eval_config(args.config)
    if args.subjects:
        subjects = sorted(set(args.subjects))
    else:
        table = Path(cfg["data_root"]) / "measurement/patient_learnability_table.csv"
        subjects = sorted(r["subject"] for r in csv.DictReader(table.open()))
    curves = json.loads(args.power.read_text()) if args.power.exists() else {"curves": []}
    requirements = G.requirements_from_power_curves(curves, tier="medium")
    serial_requirements = {f"{view}|{horizon}": value for (view, horizon), value in requirements.items()}
    jobs = [(subject, str(args.config), serial_requirements) for subject in subjects]
    supports, rows = {}, []
    with ProcessPoolExecutor(max_workers=min(max(1, args.workers), len(jobs))) as pool:
        for subject, result in zip(subjects, pool.map(_one, jobs)):
            support, subject_rows = result
            supports[subject] = support
            rows.extend(subject_rows)
            print(f"{subject}: {len(subject_rows)} endpoint/horizon rows", flush=True)
    payload = {
        "format": "group_event_state_v0_3_3_eligibility_by_endpoint_horizon",
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": cfg["_config_sha256"],
        "power_source": str(args.power) if args.power.exists() else None,
        "power_requirement_status": "available" if requirements else "power_curve_pending",
        "requirements": serial_requirements,
        "rows": rows,
        "support_by_subject": supports,
        "evidence_label": "HUMAN_SUPPORT_ONLY_NO_RESULT",
        "sealed_partition_opened": False,
    }
    atomic_json(args.out, payload)
    atomic_json(args.shared, payload)
    print(json.dumps({"status": "complete", "subjects": len(subjects), "rows": len(rows),
                      "power_requirement_status": payload["power_requirement_status"]}, indent=2))


if __name__ == "__main__":
    main()
