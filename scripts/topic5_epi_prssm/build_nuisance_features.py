#!/usr/bin/env python3
"""Causal multi-scale rate and interval features, one row per event.

Topic 2 established that the event rate itself drifts slowly, that roughly three
quarters of the serial correlation in inter-event intervals comes from slow drift,
and that Epilepsiae rate autocorrelation is still positive at eight hours.  A
latent-state claim therefore has to beat these observable time variables, not
merely beat a static repertoire.

Every feature is a strict look-back: at event ``e`` only events at or before ``e``
contribute, so the feature is available to an online system at that moment.  The
features are recomputed here on the current 34-patient cohort and the current
chronological split; no statistic is imported from an earlier cohort.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.topic5_epi_prssm.cohort import cohort_subjects, load_tensors  # noqa: E402
from src.topic5_epi_prssm.contracts import (  # noqa: E402
    OUTPUT_ROOT, atomic_write_json, code_revision, package_hash, sha256_file,
)
from src.topic5_epi_prssm.seizure_labels import TIMEZONE_OFFSET_HOURS  # noqa: E402

OUT = OUTPUT_ROOT / "nuisance_features"
#: look-back windows in seconds: 30 min, 2 h, 4 h, 8 h
RATE_WINDOWS = (1800.0, 7200.0, 14400.0, 28800.0)
FEATURE_NAMES = (
    "log_iei", "log_rate_1800s", "log_rate_7200s", "log_rate_14400s", "log_rate_28800s",
    "log_median_iei_7200s", "coverage_7200s", "time_of_day_sin", "time_of_day_cos",
    "session_position",
)


def build(subject: str) -> dict:
    patient = load_tensors([subject])[0]
    times = np.asarray(patient.event_time, dtype=np.float64)
    n = len(times)
    delta = patient.delta_t.cpu().numpy().astype(np.float64)
    session = np.asarray(patient.meta["session_index"])

    features = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    features[:, 0] = np.log1p(delta)
    index = np.arange(n)
    for j, window in enumerate(RATE_WINDOWS):
        lo = np.searchsorted(times, times - window, side="left")
        features[:, 1 + j] = np.log1p((index - lo) / (window / 3600.0))
    lo2h = np.searchsorted(times, times - 7200.0, side="left")
    median_iei = np.full(n, np.nan)
    coverage = np.zeros(n)
    for e in range(n):
        start = lo2h[e]
        if e - start >= 3:
            median_iei[e] = float(np.median(np.diff(times[start:e + 1])))
        edges = np.linspace(times[e] - 7200.0, times[e], 11)
        coverage[e] = float((np.histogram(times[start:e + 1], bins=edges)[0] > 0).mean())
    features[:, 5] = np.log1p(np.nan_to_num(median_iei, nan=float(np.nanmedian(median_iei))
                                            if np.isfinite(median_iei).any() else 0.0))
    features[:, 6] = coverage
    hour = ((times + TIMEZONE_OFFSET_HOURS[patient.dataset] * 3600.0) / 3600.0) % 24.0
    features[:, 7] = np.sin(2 * np.pi * hour / 24.0)
    features[:, 8] = np.cos(2 * np.pi * hour / 24.0)
    position = np.zeros(n)
    for value in np.unique(session):
        mask = session == value
        span = mask.sum()
        position[mask] = np.arange(span) / max(span - 1, 1)
    features[:, 9] = position

    # standardise on the train partition only, then freeze
    train = (patient.split.cpu().numpy() == 0)
    mean = features[train].mean(axis=0)
    scale = features[train].std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardised = (features - mean) / scale

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_subject").mkdir(parents=True, exist_ok=True)
    path = OUT / "per_subject" / f"{subject}.npz"
    np.savez_compressed(path, features=standardised.astype(np.float32),
                        raw_features=features, feature_names=np.asarray(FEATURE_NAMES),
                        train_mean=mean.astype(np.float32), train_scale=scale.astype(np.float32),
                        event_time=times)
    return {"subject": subject, "n_events": int(n), "n_features": len(FEATURE_NAMES),
            "standardised_on": "train partition only", "path": str(path),
            "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()
    rows = []
    for subject in (args.subjects or list(cohort_subjects())):
        rows.append(build(subject))
        print(f"  {subject:22s} {rows[-1]['n_events']:7d} events", flush=True)
    atomic_write_json(OUT / "NUISANCE_FEATURE_MANIFEST.json", {
        "contract": "topic5_epi_prssm_v0_1_nuisance_features",
        "why": "Topic 2 shows the event rate drifts slowly and that its autocorrelation is still "
               "positive at eight hours; a latent-state claim must beat these observable time "
               "variables, so they are provided as a matched baseline and as a residualisation "
               "set",
        "recomputed_on": "the current 34-patient cohort and the current chronological split; "
                         "no statistic is imported from an earlier cohort",
        "causality": "strict look-back: at event e only events at or before e contribute",
        "feature_names": list(FEATURE_NAMES), "rate_windows_seconds": list(RATE_WINDOWS),
        "n_subjects": len(rows), "subjects": rows,
        "code_revision": code_revision(), "package_hash": package_hash(),
    })
    print(f"\n{len(rows)} subjects written")


if __name__ == "__main__":
    main()
