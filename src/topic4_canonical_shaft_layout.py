"""Target-blind canonical contact layout and shaft-balanced cohort losses."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from src.propagation_skeleton_geometry import parse_shaft


def contact_shaft_contract(contact_names: list[str]) -> dict:
    """Parse unique contact names without reading any event-derived quantity."""
    names = [str(name) for name in contact_names]
    if not names:
        raise ValueError("contact names cannot be empty")
    if len(names) != len(set(names)):
        raise ValueError("contact names must be unique")
    parsed = [parse_shaft(name) for name in names]
    invalid = [name for name, value in zip(names, parsed) if None in value]
    if invalid:
        raise ValueError(f"unparseable contact names: {invalid}")
    shafts = [str(value[0]) for value in parsed]
    ordinals = np.asarray([int(value[1]) for value in parsed], dtype=int)
    shaft_order = sorted(set(shafts))
    return {
        "contact_names": names,
        "shaft_ids": shafts,
        "within_shaft_ordinals": ordinals,
        "shaft_order": shaft_order,
        "n_shafts": len(shaft_order),
    }


def canonical_shaft_layout(contact_names: list[str], *, sheet_size_mm: float = 20.0,
                           margin_mm: float = 2.0) -> dict:
    """Place shafts on fixed parallel rows while preserving contact ordinals.

    The ordinal axis is stretched to fill the usable sheet, exactly as the
    real-geometry projection stretches its largest-variance axis.  A fixed
    physical pitch would instead leave montages with few distinct ordinals
    inside a 2-mm strip while their real-geometry counterparts spread over
    16 mm, so the canonical-versus-real sensitivity contrast would confound
    contact arrangement with montage extent on the very axis that carries the
    contact-order claim.
    """
    contract = contact_shaft_contract(contact_names)
    usable = float(sheet_size_mm) - 2.0 * float(margin_mm)
    if usable <= 0.0:
        raise ValueError("canonical layout dimensions must be positive")
    ordinals = contract["within_shaft_ordinals"]
    ordinal_span = int(ordinals.max() - ordinals.min())
    pitch = usable / float(max(ordinal_span, 1))
    center_ordinal = 0.5 * float(ordinals.min() + ordinals.max())
    x = 0.5 * float(sheet_size_mm) + (ordinals - center_ordinal) * pitch
    shaft_to_y = {}
    shaft_order = contract["shaft_order"]
    if len(shaft_order) == 1:
        shaft_to_y[shaft_order[0]] = 0.5 * float(sheet_size_mm)
    else:
        rows = np.linspace(float(margin_mm), float(sheet_size_mm) - margin_mm,
                           len(shaft_order))
        shaft_to_y = dict(zip(shaft_order, rows.tolist()))
    y = np.asarray([shaft_to_y[shaft] for shaft in contract["shaft_ids"]], float)
    coords = np.column_stack([x, y])
    if (not np.isfinite(coords).all() or np.any(coords < -1e-9)
            or np.any(coords > float(sheet_size_mm) + 1e-9)):
        raise RuntimeError("canonical contact layout escaped the SNN sheet")
    return {
        **contract,
        "coords_sheet": coords,
        "layout_type": "canonical_parallel_shaft_rows_v2_span_filling",
        "contact_pitch_mm": float(pitch),
        "uses_event_ranks": False,
        "uses_mode_labels": False,
        "anatomical_interpretation": False,
    }


def balanced_recruitment_error(model: np.ndarray, patient: np.ndarray,
                               shaft_ids: list[str]) -> float:
    """Equal-weight shafts after averaging contact error within each shaft."""
    model = np.asarray(model, float)
    patient = np.asarray(patient, float)
    shafts = np.asarray(shaft_ids, object)
    if model.shape != patient.shape or model.shape != shafts.shape:
        raise ValueError("recruitment arrays and shaft ids must align")
    errors = [
        float(np.mean(np.abs(model[shafts == shaft] - patient[shafts == shaft])))
        for shaft in sorted(set(shafts.tolist()))
    ]
    return float(np.mean(errors))


def shaft_pair_classes(contact_names: list[str]) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Return canonical unordered contact pairs and their shaft-pair classes."""
    contract = contact_shaft_contract(contact_names)
    pairs = np.asarray(list(combinations(range(len(contact_names)), 2)), dtype=int)
    shaft_ids = contract["shaft_ids"]
    classes = [
        tuple(sorted((shaft_ids[left], shaft_ids[right])))
        for left, right in pairs
    ]
    return pairs, classes


def balanced_precedence_error(model: np.ndarray, patient: np.ndarray,
                              contact_names: list[str],
                              pair_indices: np.ndarray) -> float:
    """Equal-weight shaft-pair classes after averaging contact-pair error."""
    model = np.asarray(model, float)
    patient = np.asarray(patient, float)
    pair_indices = np.asarray(pair_indices, int)
    canonical_pairs, classes = shaft_pair_classes(contact_names)
    if (model.shape != patient.shape or model.ndim != 2
            or model.shape[1] != 3):
        raise ValueError("precedence arrays must align as (pair, three states)")
    if not np.array_equal(pair_indices, canonical_pairs):
        raise ValueError("precedence pair ordering differs from canonical contract")
    by_class = defaultdict(list)
    pair_errors = np.mean(np.abs(model - patient), axis=1)
    for pair_class, error in zip(classes, pair_errors):
        by_class[pair_class].append(float(error))
    return float(np.mean([
        np.mean(by_class[key]) for key in sorted(by_class)
    ]))
