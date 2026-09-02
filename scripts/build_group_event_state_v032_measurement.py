#!/usr/bin/env python3
"""Measurement-layer manifests and a-priori eligibility for Group-Event State v0.3.2.

Runs before any model output exists.  One worker per patient (CPU only).
"""
from __future__ import annotations

import argparse
import csv
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v032_eval.contract import (  # noqa: E402
    EvalPaths, atomic_json, atomic_text, load_eval_config, now_iso, source_commit,
)
from src.topic5_group_event_state.v032_eval.eligibility import (  # noqa: E402
    compute_eligibility, eligibility_csv_row,
)
from src.topic5_group_event_state.v032_eval.exposure import (  # noqa: E402
    contact_support_manifest, detector_provenance_audit, exposure_manifest,
    nontransductive_support_manifest, refractory_manifest,
)
from src.topic5_group_event_state.v032_eval.state_registry import write_expected_schema  # noqa: E402
from src.topic5_group_event_state.v032_eval.timeline import load_eval_timeline  # noqa: E402


def _block_inventory(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for row in csv.DictReader(path.open()):
        s = row["subject"]
        entry = out.setdefault(s, {"n_detector_channels": 0, "montage_provenance": set()})
        try:
            entry["n_detector_channels"] = max(entry["n_detector_channels"], int(float(row.get("n_detector_channels") or 0)))
        except ValueError:
            pass
        entry["montage_provenance"].add(row.get("montage_provenance", ""))
    for entry in out.values():
        entry["montage_provenance"] = "|".join(sorted(v for v in entry["montage_provenance"] if v))
    return out


def _worker(args: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    subject, config_path, inventory = args
    started = time.time()
    try:
        cfg = load_eval_config(Path(config_path))
        tl = load_eval_timeline(subject, cfg)
        result = {
            "subject": subject,
            "status": "ok",
            "timeline": tl.summary(),
            "exposure": exposure_manifest(tl, cfg),
            "refractory": refractory_manifest(tl, cfg),
            "contact_support": contact_support_manifest(tl, cfg),
            "nontransductive": nontransductive_support_manifest(
                tl, cfg,
                hardware_detector_channels=inventory.get("n_detector_channels") or None,
                montage_provenance=inventory.get("montage_provenance"),
            ),
            "eligibility": compute_eligibility(tl, cfg),
            "seconds": time.time() - started,
        }
    except Exception as exc:  # report, never hide
        result = {"subject": subject, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc(), "seconds": time.time() - started}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/topic5_group_event_state_v032_eval.json")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cfg = load_eval_config(args.config)
    paths = EvalPaths.from_config(cfg)
    paths.ensure()
    write_expected_schema(paths.shared)
    subjects = args.subjects or sorted(p.name for p in Path(cfg["dataset_root"]).iterdir() if (p / "index.json").exists())
    inventory = _block_inventory(Path(cfg["session_inventory"]).with_name("block_inventory.csv"))
    status_path = paths.measurement / "STATUS.json"
    atomic_json(status_path, {"stage": "measurement", "status": "running", "started": now_iso(),
                              "subjects": subjects, "completed": [], "failed": []})
    jobs = [(s, str(args.config), inventory.get(s, {})) for s in subjects]
    results: dict[str, dict[str, Any]] = {}
    with get_context("spawn").Pool(processes=max(1, min(args.workers, len(jobs)))) as pool:
        for res in pool.imap_unordered(_worker, jobs):
            results[res["subject"]] = res
            atomic_json(status_path, {
                "stage": "measurement", "status": "running", "updated": now_iso(), "subjects": subjects,
                "completed": sorted(s for s, r in results.items() if r["status"] == "ok"),
                "failed": sorted(s for s, r in results.items() if r["status"] != "ok"),
            })
            print(f"[{now_iso()}] {res['subject']}: {res['status']} ({res['seconds']:.1f}s)", flush=True)
    ok = {s: r for s, r in results.items() if r["status"] == "ok"}
    failed = {s: {"error": r["error"], "traceback": r["traceback"]} for s, r in results.items() if r["status"] != "ok"}
    meta = {"format_version": "group_event_state_v0_3_2", "generated": now_iso(), "source_commit": source_commit(),
            "config_sha256": cfg["_config_sha256"], "n_patients": len(ok), "failed": failed}

    atomic_json(paths.measurement / "valid_exposure_manifest.json",
                {**meta, "format": "group_event_state_v0_3_2_valid_exposure_manifest",
                 "patients": {s: r["exposure"] for s, r in ok.items()}})
    atomic_json(paths.measurement / "detector_refractory_manifest.json",
                {**meta, "format": "group_event_state_v0_3_2_detector_refractory_manifest",
                 "patients": {s: r["refractory"] for s, r in ok.items()}})
    atomic_json(paths.measurement / "time_varying_contact_support.json",
                {**meta, "format": "group_event_state_v0_3_2_time_varying_contact_support",
                 "patients": {s: r["contact_support"] for s, r in ok.items()}})
    atomic_json(paths.measurement / "nontransductive_support_manifest.json",
                {**meta, "format": "group_event_state_v0_3_2_nontransductive_support_manifest",
                 "measurement_layer_nested_contract": "prefix_vocabulary_on_legacy_event_stream",
                 "patients": {s: r["nontransductive"] for s, r in ok.items()}})
    audit = detector_provenance_audit(cfg, {
        s: {"detector_reference": r["nontransductive"]["detector_reference"],
            "montage_provenance": r["nontransductive"]["montage_provenance"],
            "hardware_detector_channels": r["nontransductive"]["hardware_detector_channels"],
            "legacy_vocabulary_n": r["nontransductive"]["legacy_vocabulary"]["n"],
            "prefix_vocabulary_n": r["nontransductive"]["prefix_vocabulary"]["n"],
            "minimum_possible_iei_seconds": r["refractory"]["structural_refractory"]["minimum_possible_iei_seconds"],
            "fraction_iei_below_2x_core": r["refractory"]["structural_refractory"]["fraction_iei_below_2x_core"]}
        for s, r in ok.items()})
    atomic_json(paths.measurement / "detector_provenance_audit.json", {**meta, **audit})
    eligibility = {s: r["eligibility"] for s, r in ok.items()}
    elig_payload = {**meta, "format": "group_event_state_v0_3_2_endpoint_eligibility",
                    "frozen_before_any_model_result": True, "patients": eligibility,
                    "cohort": {
                        "n_patients": len(eligibility),
                        "n_eligible_count_30min": sum(e["eligibility"]["count_30min_primary"]["eligible"] for e in eligibility.values()),
                        "n_eligible_count_120min": sum(e["eligibility"]["count_120min_secondary"]["eligible"] for e in eligibility.values()),
                        "n_eligible_h2a": sum(e["eligibility"]["h2a_positive_k_prefix"]["eligible"] for e in eligibility.values()),
                        "eligible_count_30min": sorted(s for s, e in eligibility.items() if e["eligibility"]["count_30min_primary"]["eligible"]),
                        "eligible_count_120min": sorted(s for s, e in eligibility.items() if e["eligibility"]["count_120min_secondary"]["eligible"]),
                        "eligible_h2a": sorted(s for s, e in eligibility.items() if e["eligibility"]["h2a_positive_k_prefix"]["eligible"]),
                    }}
    atomic_json(paths.measurement / "endpoint_eligibility.json", elig_payload)
    atomic_json(paths.shared / "endpoint_eligibility.json", elig_payload)
    rows = [eligibility_csv_row(e) for _s, e in sorted(eligibility.items())]
    if rows:
        import io
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        atomic_text(paths.measurement / "patient_learnability_table.csv", buf.getvalue())
    atomic_json(status_path, {"stage": "measurement", "status": "complete" if not failed else "complete_with_failures",
                              "finished": now_iso(), "n_ok": len(ok), "failed": sorted(failed)})
    print(json.dumps(elig_payload["cohort"], indent=2))


if __name__ == "__main__":
    main()
