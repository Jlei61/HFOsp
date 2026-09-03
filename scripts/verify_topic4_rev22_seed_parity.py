"""rev22-DCI Task 2 parity verdict: compare a split-seed worker NPZ against an archived one.

Every array present in the archived NPZ must be present in the candidate NPZ and be
byte-identical (same dtype, shape and bytes). Arrays only present in the candidate
(``topology_seed``, ``dynamics_seed``) are listed but do not affect the verdict.

Usage:
  python scripts/verify_topic4_rev22_seed_parity.py --candidate <new.npz> --archived <old.npz> [--json <report.json>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_ARCHIVED = Path(
    "/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/data_driven_dual_core_mechanism_atlas"
    "/confirmation/workers/dualcore_s39_reference_seed_2521.npz"
)


def _digest(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def compare(candidate: Path, archived: Path) -> dict:
    with np.load(candidate, allow_pickle=False) as new, np.load(archived, allow_pickle=False) as old:
        report = {"candidate": str(candidate), "archived": str(archived), "arrays": {},
                  "missing_in_candidate": [], "extra_in_candidate": sorted(set(new.files) - set(old.files))}
        for key in old.files:
            if key not in new.files:
                report["missing_in_candidate"].append(key)
                continue
            a, b = new[key], old[key]
            same = a.dtype == b.dtype and a.shape == b.shape and _digest(a) == _digest(b)
            report["arrays"][key] = {"identical": bool(same), "dtype": str(b.dtype), "shape": list(b.shape)}
    mismatched = [k for k, v in report["arrays"].items() if not v["identical"]]
    report["mismatched"] = mismatched
    report["status"] = (
        "SEED_PARITY_BYTE_IDENTICAL" if not mismatched and not report["missing_in_candidate"]
        else "SEED_PARITY_FAILED"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--archived", type=Path, default=DEFAULT_ARCHIVED)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    report = compare(args.candidate, args.archived)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "n_arrays": len(report["arrays"]),
                      "mismatched": report["mismatched"],
                      "missing_in_candidate": report["missing_in_candidate"],
                      "extra_in_candidate": report["extra_in_candidate"]}, indent=2))


if __name__ == "__main__":
    main()
