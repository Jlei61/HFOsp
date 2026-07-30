#!/usr/bin/env python
"""Build the frozen Figure 6 target bands in one spectrogram pass per seizure.

The cache is separate from the accepted Topic 5 caches. It never overwrites
them and it does not compute a prediction label. Labels are constructed later,
after a prefix-only interictal axis has been frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/hfosp_fig6_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfosp_fig6_mpl")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", "/tmp/hfosp_fig6_mne")

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_topic5_v2_band_cache import build_subject
from src.topic5_v2_band_scan import load_phase1_config


SPECS = [
    ("low_1_8", 1.0, 8.0, True, "primary"),
    ("broad_1_150", 1.0, 150.0, True, "sensitivity"),
    ("gamma_30_80", 30.0, 80.0, True, "negative_control"),
    ("high_gamma_80_150", 80.0, 150.0, True, "negative_control"),
]


def candidate_subjects(parent_table: Path):
    df = pd.read_csv(parent_table)
    rows = df[df["group_id"] == "all_phenotype_matched"]
    return sorted(rows["subject"].astype(str).unique())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "config/topic5_state_conditioned_predictor.yaml")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument(
        "--onset-only-post-sec",
        type=float,
        default=60.0,
        help="clinical-relative post window; 60 s covers the audited +35.94 s EEG-onset lag",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    out_root = ROOT / cfg["outputs"]["target_cache"]
    parent = ROOT / cfg["cohort"]["parent_event_table"]
    audit = ROOT / cfg["cohort"]["eligibility_audit"]
    subjects = args.subjects or candidate_subjects(parent)
    phase_cfg = load_phase1_config()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "contract": cfg["contract"]["name"],
        "config": str(args.config.relative_to(ROOT)),
        "config_sha256": sha256(args.config),
        "source_spec": cfg["contract"]["source_spec"],
        "bands": [
            {"name": n, "lo": lo, "hi": hi, "half_open": half, "role": role}
            for n, lo, hi, half, role in SPECS
        ],
        "alignment_note": "stored rel_t is clinical-relative; primary EEG-relative slicing occurs downstream",
        "baseline_note": "cached affine robust-z is re-referenced downstream to fixed EEG [-120,-90] s",
        "onset_only_post_sec_requested": float(args.onset_only_post_sec),
        "producer_sha256": sha256(Path(__file__)),
        "subjects_requested": subjects,
    }
    manifest_name = (
        "cache_manifest.json"
        if args.subjects is None
        else "cache_manifest_subset_" + hashlib.sha1(
            ",".join(subjects).encode("utf-8")
        ).hexdigest()[:10] + ".json"
    )
    (out_root / manifest_name).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[fig6-target-cache] {len(subjects)} subjects -> {out_root}", flush=True)
    for subject in subjects:
        output = out_root / "cache" / f"{subject}.npz"
        if output.exists() and not args.restart:
            print(f"[skip] {subject}: exists", flush=True)
            continue
        print(f"[build] {subject}", flush=True)
        try:
            build_subject(
                subject,
                "broad",
                SPECS,
                {"low_1_8"},
                phase_cfg,
                out_root,
                audit,
                "analysis_eligible",
                args.onset_only_post_sec,
            )
        except Exception as exc:
            print(f"[error] {subject}: {type(exc).__name__}: {exc}", flush=True)
    print("FIG6 TARGET CACHE DONE", flush=True)


if __name__ == "__main__":
    main()
