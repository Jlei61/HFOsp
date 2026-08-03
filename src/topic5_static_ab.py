"""Exact-name loader for the frozen Topic 5 static A/B fields."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _centered_field(values: np.ndarray) -> np.ndarray:
    field = np.asarray(values, dtype=np.float64)
    return field - np.nanmean(field)


def load_frozen_static_scaffold(
    artifact_root: Path,
    subject: str,
    contact_names: np.ndarray,
) -> dict[str, np.ndarray]:
    """Load the paper A/B fields and align them by exact contact name."""

    path = (
        Path(artifact_root)
        / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
        / f"{subject}.json"
    )
    payload = json.loads(path.read_text())
    if payload.get("status") != "ok" or payload["interictal_field"].get("status") != "ok":
        raise RuntimeError(f"{subject}: frozen interictal scaffold unavailable")
    field = payload["interictal_field"]
    source_names = np.asarray(field["contact_order"]).astype(str)
    target_names = np.asarray(contact_names).astype(str)
    if len(set(source_names)) != len(source_names) or len(set(target_names)) != len(target_names):
        raise RuntimeError(f"{subject}: duplicate contact names prevent exact join")
    source_index = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in source_names if name not in set(target_names)]
    if missing:
        raise RuntimeError(
            f"{subject}: frozen scaffold contacts absent from rank dataset: {missing}"
        )
    valid = np.asarray([name in source_index for name in target_names], dtype=bool)
    if int(valid.sum()) < 3:
        raise RuntimeError(f"{subject}: fewer than three exact scaffold contacts")

    def align(values: np.ndarray) -> np.ndarray:
        source = _centered_field(np.asarray(values, dtype=np.float64))
        if len(source) != len(source_names):
            raise RuntimeError(f"{subject}: frozen scaffold value length drift")
        aligned = np.full(len(target_names), np.nan, dtype=np.float64)
        for target_index, name in enumerate(target_names):
            if name in source_index:
                aligned[target_index] = source[source_index[name]]
        return aligned

    model = field["field_models"]
    return {
        "scaffold_valid": valid,
        "scaffold_field_a": align(model["own_a"]["template_field"]),
        "scaffold_field_b": align(model["own_b"]["template_field"]),
    }
