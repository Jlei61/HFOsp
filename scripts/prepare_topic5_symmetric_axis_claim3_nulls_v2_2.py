#!/usr/bin/env python3
"""Freeze the 256 per-patient random physical axes for formal Claim 3."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_symmetric_axis_propagation_state_v2_2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _claim2_passed() -> dict[str, Any]:
    path = BASE / "formal/analysis/CLAIM2_STATUS.json"
    if not path.is_file():
        raise SystemExit("Claim 2 status is absent; Claim 3 remains locked")
    status = json.loads(path.read_text(encoding="utf-8"))
    if (
        status.get("status") != "complete"
        or status.get("claim2_next") != "PASS"
        or status.get("claim2_future") != "PASS"
        or status.get("next_stage_allowed") is not True
    ):
        raise SystemExit("both frozen Claim-2 endpoints must PASS before Claim 3")
    return status


def main() -> None:
    config_path = ROOT / "config/topic5_symmetric_axis_propagation_state_v2_2.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    claim2 = _claim2_passed()
    lock_path = BASE / "formal/PHYSICAL_AXIS_FORMAL_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    subjects = list(map(str, lock["subjects"]))
    null_seed = int(cfg["statistics"]["null_seed"])
    n_directions = int(cfg["statistics"]["random_directions"])
    if (
        len(subjects) != 22
        or null_seed != 20260726
        or n_directions != 256
    ):
        raise SystemExit("formal random-axis inventory drifted from v2.2")

    output = BASE / "formal/claim3_random_axis_nulls"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(null_seed)
    files = {}
    for subject in subjects:
        axes = rng.normal(size=(n_directions, 3))
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        # The kernel is sign invariant.  Save a unique canonical representative.
        anchor = np.argmax(np.abs(axes), axis=1)
        sign = np.sign(axes[np.arange(n_directions), anchor])
        axes *= np.where(sign < 0, -1.0, 1.0)[:, None]
        path = output / f"{subject}.npy"
        temporary = output / f".{subject}.{os.getpid()}.npy"
        np.save(temporary, axes.astype("<f8"), allow_pickle=False)
        temporary.replace(path)
        files[subject] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "shape": list(axes.shape),
            "unit_norm_max_abs_error": float(
                np.max(np.abs(np.linalg.norm(axes, axis=1) - 1.0))
            ),
        }

    manifest = {
        "contract": cfg["contract"]["name"],
        "version": cfg["contract"]["version"],
        "status": "locked",
        "null_seed": null_seed,
        "algorithm": (
            "numpy.default_rng(seed); subjects in PHYSICAL_AXIS_FORMAL_LOCK order; "
            "normal(0,1) vectors; unit normalization; sign canonicalization"
        ),
        "n_subjects": len(subjects),
        "directions_per_subject": n_directions,
        "subjects": subjects,
        "files": files,
        "physical_lock_sha256": sha256(lock_path),
        "claim2_status_sha256": sha256(
            BASE / "formal/analysis/CLAIM2_STATUS.json"
        ),
        "claim2_endpoints": claim2["endpoints"],
        "target_values_read": False,
    }
    atomic_json(output / "RANDOM_AXIS_NULL_LOCK.json", manifest)
    print(json.dumps({"status": "locked", "n_subjects": 22, "n": 256}, indent=2))


if __name__ == "__main__":
    main()
