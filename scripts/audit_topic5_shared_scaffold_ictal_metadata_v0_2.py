#!/usr/bin/env python3
"""Freeze early-ictal file/contact denominators without reading energy values."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readout-config",
        type=Path,
        default=ROOT / "config/topic5_shared_scaffold_ictal_readout_v0_2.yaml",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=ROOT / "config/topic5_shared_scaffold_propagation_rnn_v0_2.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    readout_config = yaml.safe_load(args.readout_config.resolve().read_text())
    training_config = yaml.safe_load(args.training_config.resolve().read_text())
    output = (
        args.output_root.resolve()
        if args.output_root
        else ROOT / readout_config["output_root"]
    )
    audit_root = output / "target_audit"
    target_root = Path(readout_config["target_cache_root"]).resolve()
    dataset_root = (
        Path(training_config["dataset_artifact_root"]).resolve()
        / training_config["dataset_root"]
    )
    primary = list(map(str, readout_config["primary_subjects"]))
    supportive = str(readout_config["supportive_subject"])
    subjects = primary + [supportive]
    if len(set(subjects)) != 16 or supportive in primary:
        raise RuntimeError("ictal denominator must be 15 primary plus one supportive")

    rows = []
    target_member = f"{readout_config['target_key']}.npy"
    for subject in subjects:
        dataset_path = dataset_root / "per_subject" / f"{subject}.npz"
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)
        with np.load(dataset_path, allow_pickle=False) as data:
            model_contacts = np.asarray(data["contact_names"]).astype(str)
        files = sorted((target_root / f"outer_{subject}").glob(f"{subject}__*.npz"))
        if not files:
            raise RuntimeError(f"{subject}: no target files")
        for seizure_index, path in enumerate(files):
            with zipfile.ZipFile(path) as archive:
                members = set(archive.namelist())
            if target_member not in members or "contact_names.npy" not in members:
                raise RuntimeError(f"{path}: required target/contact members absent")
            # NPZ access is lazy; only contact labels are deserialized here.
            with np.load(path, allow_pickle=False) as data:
                target_contacts = np.asarray(data["contact_names"]).astype(str)
            if len(set(target_contacts.tolist())) != len(target_contacts):
                raise RuntimeError(f"{path}: duplicate target contact labels")
            joined = [name for name in model_contacts if name in set(target_contacts)]
            rows.append(
                {
                    "subject": subject,
                    "primary": subject in primary,
                    "supportive_only": subject == supportive,
                    "seizure_index": seizure_index,
                    "seizure_id": path.stem.split("__", 1)[-1],
                    "artifact_path": str(path),
                    "artifact_sha256": sha256_file(path),
                    "target_member_present": True,
                    "target_energy_values_read": False,
                    "n_model_contacts": int(len(model_contacts)),
                    "n_target_contacts": int(len(target_contacts)),
                    "n_exact_joined_contacts": int(len(joined)),
                    "exact_join_eligible": len(joined)
                    >= int(readout_config["minimum_exact_joined_contacts"]),
                    "joined_contact_names": "|".join(joined),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["subject", "seizure_index"])
    if not bool(frame.exact_join_eligible.all()):
        failed = frame.loc[~frame.exact_join_eligible, ["subject", "seizure_id"]]
        raise RuntimeError(f"ineligible exact joins:\n{failed.to_string(index=False)}")
    audit_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(audit_root / "ictal_metadata_inventory.csv", index=False)
    summary = {
        "contract": readout_config["contract"],
        "target_values_read": False,
        "target_values_sealed": True,
        "primary_subjects": primary,
        "supportive_subject": supportive,
        "n_primary_subjects": len(primary),
        "n_total_subjects": len(subjects),
        "n_seizures": int(len(frame)),
        "n_contacts_min": int(frame.n_exact_joined_contacts.min()),
        "n_contacts_median": float(frame.n_exact_joined_contacts.median()),
        "n_contacts_max": int(frame.n_exact_joined_contacts.max()),
        "target_key_present_but_not_read": str(readout_config["target_key"]),
        "target_cache_root": str(target_root),
        "readout_config_sha256": sha256_file(args.readout_config.resolve()),
        "training_config_sha256": sha256_file(args.training_config.resolve()),
        "inventory_sha256": sha256_file(audit_root / "ictal_metadata_inventory.csv"),
    }
    atomic_json(audit_root / "TARGET_SEAL.json", summary)
    print(json.dumps({"status": "COMPLETE", **summary}, allow_nan=False))


if __name__ == "__main__":
    main()
