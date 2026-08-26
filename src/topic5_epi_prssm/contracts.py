"""Frozen constants, paths, hashing and the fail-closed forbidden-input audit.

Every number in ``FROZEN`` was fixed before any Epi-PRSSM outcome was computed.
Values inherited from ``config/topic5_slow_state_v4_0.yaml`` are marked with
their provenance so a later reader can tell a re-used frozen constant from a
new one.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

CONTRACT = "topic5_epi_prssm_v0_1"
CONTRACT_VERSION = "0.1"

DATASET_ROOT = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4/per_subject"
SOURCE_MAPPING_ROOT = (
    ROOT / "results/topic5_event_indexed_evolving_rank_field/development/input_audit/per_subject"
)
EPILEPSIAE_BLOCK_INVENTORY = ROOT / "results/epilepsiae_block_inventory.csv"
YUQUAN_BLOCK_INVENTORY = ROOT / "results/dataset_inventory/yuquan_block_inventory.csv"
EPILEPSIAE_SEIZURE_INVENTORY = ROOT / "results/epilepsiae_seizure_inventory.csv"
YUQUAN_SEIZURE_ROOT = ROOT / "results/seizure_detection"
ONSET_REGISTRY = (
    ROOT / "results/topic5_clinical_onset_source_annotation_v0_1/annotation_registry.csv"
)

OUTPUT_ROOT = ROOT / "results/epi_prssm/v0_1"


FROZEN: dict[str, Any] = {
    # --- inherited, already frozen upstream -------------------------------
    # chosen from the measured metadata-gap distribution in v4.0 before any
    # v4.0 or Epi-PRSSM outcome existed (config/topic5_slow_state_v4_0.yaml).
    "session_join_seconds": 300.0,
    # v4.0 split_fractions [0.60, 0.20, 0.20]; realised here by cutting the
    # dataset_v0_4 80% calibration partition 75/25 in chronological order and
    # keeping the dataset's own last-20% partition sealed as the test.
    "split_fractions": (0.60, 0.20, 0.20),
    "tie_tolerance_seconds": 0.0,
    # --- new for Epi-PRSSM, fixed here before any run ----------------------
    "state_dim_H": 8,
    "observer_dim": 16,
    "event_hidden_dim": 16,
    "burn_in_events": 20,
    "tbptt_length": 64,
    "open_loop_horizons": (5, 10, 20, 40),
    "state_reset_horizons": (1, 2, 5, 10, 20, 40, 80),
    "min_events_for_eligibility": 400,
    "min_contacts_for_eligibility": 5,
    "min_train_events_for_baseline": 200,
    "max_integration_substeps": 32,
    "max_substep_seconds": 60.0,
    "resource_tau_grid_seconds": (60.0, 300.0, 1800.0, 7200.0),
    "exposure_tau_primary_seconds": (300.0, 1800.0, 7200.0),  # fast / medium / slow
    "exposure_tau_sensitivity_seconds": (300.0, 900.0, 1800.0, 3600.0, 7200.0),
    "exposure_event_count_sensitivity": (5, 10, 20, 40, 80),
    "breadth_seeds": (11, 12, 13),
    "confirmation_seeds": (11, 12, 13, 14, 15),
    "bootstrap_draws": 10000,
    "bootstrap_seed": 20260818,
    "preictal_window_seconds": 1800.0,
    "pseudo_onset_draws": 200,
    "min_last_ied_to_onset_seconds": 0.0,
    "max_last_ied_to_onset_seconds": 3600.0,
}


# --------------------------------------------------------------------------
# forbidden inputs
# --------------------------------------------------------------------------

#: Fields, files and concepts that must never reach the interictal state model
#: as input, target, gate or model-selection signal.  ``geometry`` is NOT on
#: this list: spec section 4 explicitly authorises the symmetric contact-geometry
#: Laplacian as graph support.  SOZ remains forbidden.
FORBIDDEN_INPUTS: tuple[str, ...] = (
    "soz",
    "soz_core_channels",
    "ictal",
    "seizure_label",
    "seizure_time",
    "time_to_seizure",
    "early_ictal",
    "snn",
    "ta_tb",
    "ab_axis",
    "axis_label",
    "kmeans_label",
    "template_label",
    "old_heldout20",
    "test_split",
    "future_mark",
    "future_load",
)

#: Substrings whose appearance in a loaded artefact path means the run has read
#: something it must not have read.
FORBIDDEN_PATH_TOKENS: tuple[str, ...] = (
    "soz_core_channels",
    "seizure_detection",
    "epilepsiae_seizure_inventory",
    "annotation_registry",
    "topic4_attractor",
    "snn_engine",
)


class ForbiddenInputError(RuntimeError):
    """Raised when a forbidden field or artefact reaches a state-learning path."""


@dataclass(frozen=True)
class LeakageGuard:
    """Fail-closed guard carried by every state-learning entry point.

    ``allow_seizure_side`` is only ever true inside Goal 3, and only after
    ``INTERICTAL_MODEL_FREEZE.json`` exists on disk.
    """

    stage: str
    allow_seizure_side: bool = False

    def check_fields(self, fields: Iterable[str]) -> None:
        bad = sorted({f for f in fields if _is_forbidden_field(f, self.allow_seizure_side)})
        if bad:
            raise ForbiddenInputError(
                f"{self.stage}: forbidden fields reached a state-learning path: {bad}"
            )

    def check_path(self, path: str | os.PathLike[str]) -> None:
        text = str(path)
        bad = sorted({t for t in FORBIDDEN_PATH_TOKENS if t in text})
        if bad and not self.allow_seizure_side:
            raise ForbiddenInputError(f"{self.stage}: forbidden artefact read: {text} ({bad})")

    def check_split(self, split_names: Iterable[str]) -> None:
        names = {str(s) for s in split_names}
        if "test" in names:
            raise ForbiddenInputError(
                f"{self.stage}: the untouched test partition may not be consumed here "
                "(Hard Gate C). Release it only through FORMAL_TEST_RELEASE.json."
            )


def _is_forbidden_field(field: str, allow_seizure_side: bool) -> bool:
    lowered = str(field).lower()
    seizure_side = {"ictal", "seizure_label", "seizure_time", "time_to_seizure", "early_ictal"}
    for token in FORBIDDEN_INPUTS:
        if token in lowered:
            if allow_seizure_side and token in seizure_side:
                continue
            return True
    return False


# --------------------------------------------------------------------------
# hashing / atomic io
# --------------------------------------------------------------------------


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes() + str(array.dtype).encode() + str(array.shape).encode()).hexdigest()


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=_jsonable, ensure_ascii=False).encode()
    ).hexdigest()


def code_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - only if git is unavailable
        return "unknown"


def package_hash() -> str:
    """Hash of every tracked source file of this package (the code revision of the model)."""
    here = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(here.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"not json serialisable: {type(value)!r}")


def jsonable(value: Any) -> Any:
    """Recursively coerce numpy / path / tuple structures into json types."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(jsonable(payload), stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return target


def atomic_write_text(path: str | os.PathLike[str], text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return target


def atomic_write_csv(path: str | os.PathLike[str], frame) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, target)
    return target
