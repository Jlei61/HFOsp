#!/usr/bin/env python3
"""Freeze the 24-seizure E1146 early-energy target for Topic 4 rev5."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_clinical_onset_shared_field_concordance import (  # noqa: E402
    MIN_BASELINE_FRAMES,
    _extract_bounds,
    clinical_relative_times,
    highest_valid_broadband_upper,
)
from scripts.run_topic5_eeg_onset_shared_field_concordance import (  # noqa: E402
    _eeg_offset_from_inventory,
)
from scripts.run_topic5_t0_eligibility import ICTAL_REFERENCE  # noqa: E402
from scripts.run_topic5_tspectral_field_concordance import (  # noqa: E402
    SPECTRAL_WINDOW_SEC,
    _extract_log_band_power,
)
from src.ictal_onset_extraction import extract_seizure_window  # noqa: E402
from src.topic4_fig5_target_informed_bridge import (  # noqa: E402
    SCHEMA_ID,
    bootstrap_patient_summary,
    jsonable,
)
from src.topic5_ictal_recruitment import bipolar_alias_label  # noqa: E402
from src.topic5_tspectral_field_concordance import (  # noqa: E402
    DISTAL_BASELINE_EEG_SEC,
    aggregate_complete_windows,
    distal_baseline_robust_z,
    exact_name_align_matrix,
)


FIELD_RECORD = ROOT / "results/interictal_propagation_masked/template_gradient_fields/per_subject/epilepsiae_1146.json"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory_at_root(results_root, dataset, sid):
    candidates = [
        Path(results_root) / "dataset_inventory" / f"{dataset}_seizure_inventory.csv",
        Path(results_root) / f"{dataset}_seizure_inventory.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"missing seizure inventory under {results_root}")
    onset_field = "clin_onset_epoch" if dataset == "epilepsiae" else "eeg_onset_epoch"
    with path.open() as handle:
        rows = [row for row in csv.DictReader(handle)
                if row.get("subject") == sid and row.get(onset_field)]
    return sorted(rows, key=lambda row: float(row[onset_field]))


def _extract_two_windows(subject, seizure_idx, record, bands, *, results_root):
    dataset, sid = subject.split("_", 1)
    inventory = _inventory_at_root(results_root, dataset, sid)
    row = inventory[int(seizure_idx)]
    eeg_rel = _eeg_offset_from_inventory(dataset, row)
    pre_sec, post_sec = _extract_bounds(eeg_rel)
    seizure = extract_seizure_window(
        f"{dataset}/{sid}", int(seizure_idx), pre_sec=pre_sec,
        post_sec=post_sec, reference=ICTAL_REFERENCE[dataset],
        results_root=results_root)
    target_names = [str(v) for v in record["interictal_field"]["contact_order"]]
    raw_names = [bipolar_alias_label(name) for name in seizure.ch_names]
    if len(raw_names) != len(set(raw_names)):
        raise ValueError("raw contact aliases are not unique")
    raw_index = {name: i for i, name in enumerate(raw_names)}
    matched = [name for name in target_names if name in raw_index]
    if matched != target_names:
        raise ValueError(f"seizure {seizure_idx} lacks exact 15-contact montage")
    signal = seizure.signal[[raw_index[name] for name in matched]]
    upper = highest_valid_broadband_upper(seizure.fs)
    if not np.isclose(upper, 150.0):
        raise ValueError(f"seizure {seizure_idx} is not exact 1-150 Hz")
    overrides = {name: tuple(map(float, band)) for name, band in bands.items()}
    powers, times_crop = _extract_log_band_power(
        signal, seizure.fs, list(bands), band_hz_override=overrides)
    times = clinical_relative_times(times_crop, seizure.pre_sec)
    baseline = (
        float(eeg_rel + DISTAL_BASELINE_EEG_SEC[0]),
        float(eeg_rel + DISTAL_BASELINE_EEG_SEC[1]),
    )
    out = {}
    for name in bands:
        robust = distal_baseline_robust_z(
            powers[name], times, baseline, min_frames=MIN_BASELINE_FRAMES)
        aligned = exact_name_align_matrix(record, matched, robust["delta"])
        rows, complete = aggregate_complete_windows(
            aligned["values"], times,
            np.asarray([[-10.0, 0.0], [0.0, 10.0]], float),
            spectral_window_sec=SPECTRAL_WINDOW_SEC)
        if not np.all(complete) or not np.all(np.isfinite(rows)):
            raise ValueError(f"seizure {seizure_idx} has incomplete target windows")
        out[name] = rows
    return out, {
        "seizure_idx": int(seizure_idx),
        "seizure_id": str(seizure.seizure_id),
        "sample_rate_hz": float(seizure.fs),
        "eeg_onset_minus_clinical_sec": float(eeg_rel),
        "baseline_clinical_sec": list(baseline),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/topic4_data_driven_zm_target_informed_bridge_v1.json")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    patient = config["patient_target"]
    checkpoints = Path(patient["checkpoint_root"])
    files = sorted(checkpoints.glob("seizure_*.json"))
    if len(files) != int(patient["expected_complete_seizures"]):
        raise RuntimeError(f"expected 25 complete checkpoints, found {len(files)}")
    indices = [int(path.stem.split("_")[-1]) for path in files]
    display = int(patient["display_seizure_idx"])
    development = [idx for idx in indices if idx != display]
    if len(development) != 24:
        raise RuntimeError("display exclusion did not leave 24 development seizures")
    record = json.loads(FIELD_RECORD.read_text(encoding="utf-8"))
    names = [str(v) for v in record["interictal_field"]["contact_order"]]
    shafts = [str(v) for v in record["interictal_field"]["shafts"]]
    bands = {"primary_1_150": [1.0, 150.0], "sensitivity_10_150": [10.0, 150.0]}
    extracted = {}
    metadata = []
    for idx in indices:
        extracted[idx], meta = _extract_two_windows(
            patient["subject"], idx, record, bands,
            results_root=patient["canonical_results_root"])
        metadata.append(meta)
        print(json.dumps({"seizure_idx": idx, "status": "extracted"}), flush=True)
    arrays = {}
    summaries = {}
    for band_name in bands:
        pre = np.asarray([extracted[idx][band_name][0] for idx in development])
        early = np.asarray([extracted[idx][band_name][1] for idx in development])
        arrays[f"{band_name}_pre"] = pre
        arrays[f"{band_name}_early"] = early
        arrays[f"{band_name}_display_pre"] = extracted[display][band_name][0]
        arrays[f"{band_name}_display_early"] = extracted[display][band_name][1]
        summaries[band_name] = bootstrap_patient_summary(
            pre, early, draws=int(patient["bootstrap_draws"]),
            seed=int(patient["bootstrap_seed"]))
    out = ROOT / config["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "clinical_target_vectors.npz",
        contact_names=np.asarray(names, dtype="U16"),
        shaft_ids=np.asarray(shafts, dtype="U8"),
        development_seizure_indices=np.asarray(development, int),
        display_seizure_idx=np.asarray(display, int),
        **arrays,
    )
    payload = {
        "schema_id": SCHEMA_ID,
        "status": "CLINICAL_TARGET_FROZEN",
        "claim_boundary": "development-only target-informed calibration",
        "subject": patient["subject"],
        "contact_names": names,
        "shaft_ids": shafts,
        "complete_seizure_indices": indices,
        "development_seizure_indices": development,
        "display_seizure_idx": display,
        "bands_hz": bands,
        "windows_sec": {"pre": [-10.0, 0.0], "early": [0.0, 10.0]},
        "summaries": summaries,
        "per_seizure_metadata": metadata,
        "source_hashes": {
            "config": _sha256(config_path),
            "field_record": _sha256(FIELD_RECORD),
            "checkpoints": {path.name: _sha256(path) for path in files},
        },
    }
    (out / "clinical_target.json").write_text(
        json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")
    (out / "target_provenance.json").write_text(
        json.dumps({
            "schema_id": SCHEMA_ID,
            "source_hashes": payload["source_hashes"],
            "display_excluded_from_development": True,
            "n_complete": len(indices),
            "n_development": len(development),
        }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
