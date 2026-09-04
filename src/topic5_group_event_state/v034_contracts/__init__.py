"""Leakage-safe contracts for Group-Event State v0.3.4.

This package deliberately contains no model training code.  It fixes the
physical-time anchor, causal baseline, level-control provenance and structural
estimability contracts before any v0.3.4 GPU job is allowed to consume them.
"""

from .anchors import (
    AnchorRecord,
    build_fixed_time_anchors,
    independent_window_count,
    validate_anchor_records,
)
from .baseline import BaselineMatrix, build_multiscale_history
from .eligibility import audit_array_capabilities, endpoint_rows
from .levels import (
    LevelControl,
    fit_train_mean_adapter,
    rolling_prefix_level,
    selection_period_mean_input_oracle,
)

__all__ = [
    "AnchorRecord",
    "BaselineMatrix",
    "LevelControl",
    "audit_array_capabilities",
    "build_fixed_time_anchors",
    "build_multiscale_history",
    "endpoint_rows",
    "fit_train_mean_adapter",
    "independent_window_count",
    "rolling_prefix_level",
    "selection_period_mean_input_oracle",
    "validate_anchor_records",
]
