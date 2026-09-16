#!/usr/bin/env python3
"""Audit whether existing frozen contact decoders satisfy the v0.3.7 outer-time contract.

This script reads only interictal input manifests and decoder metadata.  It does
not load DEVELOPMENT, seizure outcomes, or sealed targets.  Existing bundles
that are safe from STATE_SELECTION may still fail the stricter v0.3.7 rule that
patient-local fitting must stop before STATE_TRAIN begins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v035.contracts import (  # noqa: E402
    DECODER_ROOT,
    INPUT_ROOT,
    V035_DECODER_FITS,
)
from src.topic5_group_event_state.v037.contracts import atomic_json, sha256_file  # noqa: E402


ARM = "L3_LOCAL_PLUS_LEARNED_LR"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_subject(subject: str, decoder_root: Path, input_root: Path) -> dict:
    fit = V035_DECODER_FITS[subject]
    cache = decoder_root / "cache" / fit
    manifest_path = input_root / subject / "manifest_v3.json"
    required = [
        cache / "events.npz",
        cache / "events_raw.npz",
        cache / "provenance.json",
        cache / "plane.npz",
        manifest_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return {"subject": subject, "fit_id": fit, "status": "MISSING", "missing": missing}

    manifest = _read(manifest_path)
    provenance = _read(cache / "provenance.json")
    bounds = {key: float(value) for key, value in manifest["report"]["phase_boundaries_epoch"].items()}
    with np.load(cache / "events.npz", allow_pickle=False) as stored:
        times = np.asarray(stored["event_abs_time"], dtype=np.float64)
        split = np.asarray(stored["split"], dtype=np.int8)
    used = split >= 0
    decoder_max = float(times[used].max()) if np.any(used) else None
    split_counts = {str(int(value)): int(count) for value, count in zip(*np.unique(split, return_counts=True))}
    seeds = []
    for seed in (0, 1, 2):
        unit = decoder_root / "formal_units" / fit / ARM / f"seed{seed}"
        metrics_path = unit / "metrics.json"
        done_path = unit / "DONE.json"
        if not metrics_path.exists() or not done_path.exists():
            seeds.append({"seed": seed, "status": "MISSING", "unit_dir": str(unit)})
            continue
        metrics = _read(metrics_path)
        seeds.append({
            "seed": seed,
            "status": "PASS" if (
                metrics.get("best_checkpoint_eligible") is True
                and metrics.get("target_values_read") is False
                and metrics.get("converged") is True
            ) else "FAIL",
            "best_epoch": metrics.get("best_epoch"),
            "mask_freeze_epoch": metrics.get("mask_freeze_epoch"),
            "target_values_read": metrics.get("target_values_read"),
            "checkpoint_sha256": sha256_file(unit / "weights.pt") if (unit / "weights.pt").exists() else None,
            "metrics_sha256": sha256_file(metrics_path),
            "unit_dir": str(unit),
        })

    source_split = provenance.get("v034_recorded_time_split")
    source_split_rebuilt = source_split is not None
    selection_safe = bool(
        decoder_max is not None
        and decoder_max < bounds["70pct"]
        and all(row["status"] == "PASS" for row in seeds)
    )
    strict_pre_state_fit = bool(
        decoder_max is not None
        and decoder_max < bounds["20pct"]
        and all(row["status"] == "PASS" for row in seeds)
    )
    unresolved = []
    if not source_split_rebuilt:
        unresolved.append("cache lacks v0.3.4 recorded-time re-split provenance")
    # v0.3.4 copied these arrays byte-identically from the legacy cache.  Their
    # mere presence is not proof of leakage, but primary registration needs a
    # producer audit showing that no full-period distribution entered weights.
    for name in ("plane.npz", "train_only_modes.npz"):
        if (cache / name).exists():
            unresolved.append(f"{name} copied from legacy cache; producer-time scope not proven here")
    unresolved.append("contact vocabulary comes from full cache; must be justified as montage metadata, not outcome-derived support")

    return {
        "subject": subject,
        "fit_id": fit,
        "status": "STRICT_PRIMARY_ELIGIBLE" if strict_pre_state_fit else (
            "SELECTION_SAFE_ONLY" if selection_safe else "INELIGIBLE"
        ),
        "scope": provenance.get("scope"),
        "decoder_max_used_time": decoder_max,
        "state_fit_start_20pct": bounds["20pct"],
        "state_inner_start_60pct": bounds["60pct"],
        "state_selection_start_70pct": bounds["70pct"],
        "strict_pre_state_fit": strict_pre_state_fit,
        "selection_safe": selection_safe,
        "source_split_rebuilt": source_split_rebuilt,
        "split_counts": split_counts,
        "seeds": seeds,
        "unresolved_provenance": unresolved,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "cache_provenance_sha256": sha256_file(cache / "provenance.json"),
        "development_targets_read": False,
        "seizure_targets_read": False,
        "sealed_partition_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-root", type=Path, default=DECODER_ROOT)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument(
        "--subjects",
        nargs="*",
        choices=tuple(V035_DECODER_FITS),
        default=list(V035_DECODER_FITS),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/data/hfosp_group_event_state_v0_3_7/audits/decoder_provenance.json"),
    )
    args = parser.parse_args()
    rows = [audit_subject(subject, args.decoder_root, args.input_root) for subject in args.subjects]
    summary = {
        key: sum(row.get("status") == key for row in rows)
        for key in ("STRICT_PRIMARY_ELIGIBLE", "SELECTION_SAFE_ONLY", "INELIGIBLE", "MISSING")
    }
    payload = {
        "format": "group_event_state_v0_3_7_decoder_provenance_audit_v1",
        "contract": (
            "strict primary requires every patient-local decoder fit/selection time before the "
            "20pct state-FIT boundary; selection-safe-only bundles may be used for parity or "
            "diagnosis but not v0.3.7 primary"
        ),
        "rows": rows,
        "summary": summary,
        "development_targets_read": False,
        "seizure_targets_read": False,
        "sealed_partition_opened": False,
    }
    atomic_json(args.out, payload)
    print(json.dumps({"out": str(args.out), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
