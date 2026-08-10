"""Build the training-only matched-count descriptor floor for rev9-L L2."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.run_topic4_core_field_stage3_profile_round1 import axial_map  # noqa: E402
from scripts.run_topic4_rev9l_objective_replay import (  # noqa: E402
    _load_reference,
    _patient_training_arrays,
)
from src.topic4_component_pair_search import (  # noqa: E402
    patient_descriptor_floor,
)
from src.topic4_core_field_profile import fit_profile_modes  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "config/topic4_rev9l_component_pair_edge.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _provenance(expected_commit):
    paths = set()
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py":
            continue
        try:
            paths.add(str(path.relative_to(ROOT)))
        except ValueError:
            continue
    paths.add(str(Path(__file__).resolve().relative_to(ROOT)))
    paths = sorted(paths)
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True).strip()
    current = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT,
        text=True).strip()
    hashes = {path: _sha256(ROOT / path) for path in paths}
    expected_hashes = {
        path: hashlib.sha256(subprocess.check_output(
            ["git", "show", f"{expected}:{path}"], cwd=ROOT)).hexdigest()
        for path in paths
    }
    if dirty or current != expected or any(
            hashes[path] != expected_hashes[path] for path in paths):
        raise RuntimeError("descriptor-floor producer differs from expected commit")
    return {
        "git_commit": current,
        "expected_git_commit": expected,
        "runtime_modules_dirty": False,
        "runtime_module_sha256": hashes,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    objective = config["objective"]
    if (objective.get("patient_training_only") is not True
            or objective.get("patient_heldout_permitted") is not False):
        raise RuntimeError("L2 objective is no longer training-only")
    for name, record in config["inputs"].items():
        if _sha256(record["path"]) != record["sha256"]:
            raise RuntimeError(f"L2 input hash changed: {name}")

    reference = _load_reference(config["inputs"]["profile_reference"]["path"])
    axial = axial_map()
    contact_names = sorted(axial, key=axial.get)
    patient = _patient_training_arrays(reference, contact_names)
    modes = fit_profile_modes(patient["curves"], reference)
    if modes.get("status") != "ok":
        raise RuntimeError("patient training modes are no longer reconstructable")
    with np.load(config["inputs"]["patient_training_target"]["path"],
                 allow_pickle=False) as loaded:
        if (not np.allclose(
                modes["prototypes"], loaded["patient_train_mode_prototypes"],
                atol=1e-7)
                or not np.array_equal(
                    modes["cluster_counts"], loaded["patient_train_mode_counts"])):
            raise RuntimeError("patient training target changed")

    floor, samples, sampled_blocks = patient_descriptor_floor(
        patient["curves"], patient["ranks"], modes["labels"],
        patient["block_ids"], reference,
        n_per_mode=int(objective["n_model_events_per_mode"]),
        repeats=int(objective["bootstrap_repeats"]),
        seed=int(objective["bootstrap_seed"]),
        scale_minimum=float(objective["floor_scale_minimum"]))
    output_path = Path(objective["floor_output"])
    sample_path = output_path.with_name(
        "patient_training_descriptor_floor_samples.npz")
    arrays = {"sampled_blocks": sampled_blocks}
    for mode in ("A", "B"):
        for metric, values in samples[mode].items():
            arrays[f"{mode}_{metric}"] = np.asarray(values, np.float64)
    _atomic_npz(sample_path, **arrays)
    payload = {
        "status": "REV9L_L2_PATIENT_TRAINING_FLOOR_COMPLETE",
        "scientific_role": (
            "training-only matched-count floor; patient held-out was not read"),
        "sampling_unit": objective["sampling_unit"],
        "n_events_per_mode_per_draw": int(objective["n_model_events_per_mode"]),
        "bootstrap_repeats": int(objective["bootstrap_repeats"]),
        "bootstrap_seed": int(objective["bootstrap_seed"]),
        "n_patient_training_events": int(len(patient["curves"])),
        "n_patient_training_blocks": int(len(np.unique(patient["block_ids"]))),
        "n_patient_events_by_mode": {
            "A": int(np.sum(np.asarray(modes["labels"]) == 0)),
            "B": int(np.sum(np.asarray(modes["labels"]) == 1)),
        },
        "floor": floor,
        "samples": {"path": str(sample_path), "sha256": _sha256(sample_path)},
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "patient_heldout_scores_computed": False,
        "provenance": _provenance(args.expected_commit),
    }
    atomic_write_json(payload, output_path)
    print(json.dumps({
        "status": payload["status"],
        "output": str(output_path),
        "n_patient_training_events": payload["n_patient_training_events"],
        "n_patient_training_blocks": payload["n_patient_training_blocks"],
        "patient_heldout_scores_computed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
