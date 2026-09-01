"""Observation layouts and matched nulls for the formal Topic 4 SNN cohort.

The patient targets themselves are frozen upstream; this module only adds the
two observation layouts and the within-shaft permutation null on top of them,
so a layout rebuild can never move a patient target.
"""
from __future__ import annotations

import hashlib

import numpy as np

from src.topic4_canonical_shaft_layout import canonical_shaft_layout
from src.topic4_data_driven_cohort_formal import within_shaft_null_contract


def subject_null_seed(subject_id: str, *, base_seed: int) -> int:
    """Derive a frozen per-subject null seed that does not depend on order."""
    digest = hashlib.sha256(str(subject_id).encode("utf-8")).hexdigest()
    return int(base_seed) + int(digest[:8], 16) % 1_000_000


def build_subject_layout(subject_id: str, contact_order: list[str], *,
                         real_coords_sheet: np.ndarray | None,
                         n_permutations: int, base_seed: int,
                         sheet_size_mm: float, margin_mm: float) -> dict:
    """Build the canonical layout, the sensitivity layout and the null."""
    names = [str(value) for value in contact_order]
    layout = canonical_shaft_layout(
        names, sheet_size_mm=float(sheet_size_mm), margin_mm=float(margin_mm),
    )
    seed = subject_null_seed(subject_id, base_seed=base_seed)
    null = within_shaft_null_contract(
        names, n_permutations=int(n_permutations), seed=seed,
    )
    real = None
    if real_coords_sheet is not None:
        real = np.asarray(real_coords_sheet, float)
        if real.shape != (len(names), 2):
            raise ValueError(
                f"real geometry for {subject_id} does not span the contact order"
            )
    arrays = {
        "contact_order": np.asarray(names, dtype="U64"),
        "canonical_coords_sheet": np.asarray(layout["coords_sheet"], float),
        "canonical_shaft_ids": np.asarray(layout["shaft_ids"], dtype="U16"),
        "canonical_within_shaft_ordinals": np.asarray(
            layout["within_shaft_ordinals"], int,
        ),
        "within_shaft_null_permutations": np.asarray(null["permutations"], int),
    }
    if real is not None:
        arrays["real_coords_sheet"] = real
    record = {
        "subject_id": str(subject_id),
        "n_contacts": len(names),
        "canonical_layout": {
            "layout_type": layout["layout_type"],
            "contact_pitch_mm": layout["contact_pitch_mm"],
            "n_shafts": layout["n_shafts"],
            "shaft_order": layout["shaft_order"],
            "uses_event_ranks": layout["uses_event_ranks"],
            "uses_mode_labels": layout["uses_mode_labels"],
            "anatomical_interpretation": layout["anatomical_interpretation"],
            "x_span_mm": float(
                layout["coords_sheet"][:, 0].max() - layout["coords_sheet"][:, 0].min()
            ),
            "y_span_mm": float(
                layout["coords_sheet"][:, 1].max() - layout["coords_sheet"][:, 1].min()
            ),
        },
        "real_geometry_layout": None if real is None else {
            "available": True,
            "x_span_mm": float(real[:, 0].max() - real[:, 0].min()),
            "y_span_mm": float(real[:, 1].max() - real[:, 1].min()),
        },
        "within_shaft_null": {
            key: value for key, value in null.items() if key != "permutations"
        },
    }
    return {"arrays": arrays, "record": record}
