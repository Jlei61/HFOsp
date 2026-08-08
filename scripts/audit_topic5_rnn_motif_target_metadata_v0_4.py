"""Filename-only inventory of the reused early-ictal benchmark."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


EXPECTED_PRIMARY = {
    "epilepsiae_1077", "epilepsiae_1084", "epilepsiae_1096", "epilepsiae_1125",
    "epilepsiae_1150", "epilepsiae_139", "epilepsiae_253", "epilepsiae_384",
    "epilepsiae_442", "epilepsiae_548", "epilepsiae_590", "epilepsiae_620",
    "epilepsiae_635", "epilepsiae_922", "epilepsiae_958",
}
SUPPORTIVE = "epilepsiae_1146"
TARGET_KEY = "target_1_150"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    target_root = args.target_cache_root.resolve()
    input_manifest = json.loads((out_root / "INPUT_MANIFEST.json").read_text())
    interictal = set(input_manifest["cohort"])
    field_root = Path(input_manifest["input_roots"]["field"])
    available = {
        path.name.removeprefix("outer_"): len(list(path.glob(f"{path.name.removeprefix('outer_')}__*.npz")))
        for path in target_root.glob("outer_*") if path.is_dir()
    }
    primary_join = sorted(interictal & EXPECTED_PRIMARY & set(available))
    missing_from_interictal = sorted(EXPECTED_PRIMARY - interictal)
    subjects = primary_join + ([SUPPORTIVE] if SUPPORTIVE in interictal and SUPPORTIVE in available else [])
    inventory = []
    for subject in subjects:
        record = json.loads((field_root / f"{subject}.json").read_text())
        model_contacts = [str(value) for value in record["interictal_field"]["contact_order"]]
        for path in sorted((target_root / f"outer_{subject}").glob(f"{subject}__*.npz")):
            with zipfile.ZipFile(path) as archive:
                members = set(archive.namelist())
            if f"{TARGET_KEY}.npy" not in members or "contact_names.npy" not in members:
                raise RuntimeError(f"missing target/contact member: {path}")
            # NPZ arrays are lazy.  Only the label vector is deserialized here;
            # target_1_150 is deliberately never indexed before authorization.
            with np.load(path, allow_pickle=False) as data:
                target_contacts = np.asarray(data["contact_names"]).astype(str).tolist()
            joined = [name for name in model_contacts if name in set(target_contacts)]
            inventory.append({
                "subject": subject, "primary": subject in primary_join,
                "supportive": subject == SUPPORTIVE,
                "seizure_id": path.stem.split("__", 1)[-1],
                "artifact_path": str(path), "artifact_sha256": sha256(path),
                "target_member_present_but_not_read": True,
                "n_model_contacts": len(model_contacts),
                "n_target_contacts": len(target_contacts),
                "n_exact_joined_contacts": len(joined),
                "exact_join_eligible": len(joined) >= 6,
                "joined_contact_names": "|".join(joined),
            })
    inventory_path = out_root / "early_ictal_metadata_inventory.csv"
    with inventory_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
        writer.writeheader(); writer.writerows(inventory)
    if not inventory or not all(row["exact_join_eligible"] for row in inventory):
        raise RuntimeError("at least one frozen early-ictal exact join has fewer than six contacts")
    payload = {
        "contract": "topic5_rnn_motif_early_ictal_metadata_inventory_v0_4",
        "target_cache_root": str(target_root),
        "target_arrays_deserialized": False,
        "target_values_read": False,
        "expected_primary_n": 15,
        "actual_primary_join_n": len(primary_join),
        "actual_primary_join": primary_join,
        "supportive_available": SUPPORTIVE in interictal and SUPPORTIVE in available,
        "supportive_subject": SUPPORTIVE,
        "missing_expected_primary_from_interictal_model_cohort": missing_from_interictal,
        "target_key_present_but_not_read": TARGET_KEY,
        "seizure_file_counts_filename_only": {
            subject: sum(row["subject"] == subject for row in inventory) for subject in subjects
        },
        "n_primary_seizures": sum(row["primary"] for row in inventory),
        "n_supportive_seizures": sum(row["supportive"] for row in inventory),
        "exact_joined_contacts_min": min(row["n_exact_joined_contacts"] for row in inventory),
        "exact_joined_contacts_median": float(np.median([row["n_exact_joined_contacts"] for row in inventory])),
        "exact_joined_contacts_max": max(row["n_exact_joined_contacts"] for row in inventory),
        "inventory_csv_sha256": sha256(inventory_path),
        "join_status": "COHORT_MISMATCH_REPORTED_BEFORE_UNSEAL" if len(primary_join) != 15 else "EXPECTED_JOIN",
        "interpretation": (
            "The locked 21-patient wiring-model cohort lacks five patients in the 15-patient "
            "early-ictal benchmark; the external model comparison therefore uses the actual "
            "10-patient primary intersection without changing either cohort after target access."
        ),
    }
    (out_root / "EARLY_ICTAL_METADATA_INVENTORY.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({key: payload[key] for key in (
        "expected_primary_n", "actual_primary_join_n", "supportive_available",
        "missing_expected_primary_from_interictal_model_cohort", "target_values_read"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
