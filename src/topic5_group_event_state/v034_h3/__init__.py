"""H3 estimability and control contracts for Group-Event State v0.3.4.

This package deliberately contains no human-model trainer.  It answers the
question that must come first: whether a proposed exposure/future comparison
has enough independent, fully observed support and whether M0/M1/M2 are
comparable without re-introducing the free-intercept failure from 2026-08-26.
"""

from .controls import (
    ControlDefinition,
    audit_replacement_event_overlap,
    event_window_overlap_fraction,
    interval_overlap_fraction,
    rolling_prefix_slow_level,
    selection_period_mean_oracle,
)
from .estimability import (
    BlockSupport,
    CoveragePiece,
    audit_event_count_design,
    audit_physical_window_design,
)
from .model_contract import FeedbackArmContract, build_feedback_arm_contracts, validate_arm_contracts
from .optimization import OptimizerTraceAudit, audit_optimizer_trace, optimizer_scale_equivalent
from .ridge import RidgeFit, fit_scale_stable_ridge
from .schema import SCHEMA_VERSION, build_machine_report, validate_machine_report

__all__ = [
    "BlockSupport",
    "ControlDefinition",
    "CoveragePiece",
    "FeedbackArmContract",
    "OptimizerTraceAudit",
    "RidgeFit",
    "SCHEMA_VERSION",
    "audit_event_count_design",
    "audit_replacement_event_overlap",
    "audit_physical_window_design",
    "audit_optimizer_trace",
    "build_feedback_arm_contracts",
    "build_machine_report",
    "event_window_overlap_fraction",
    "fit_scale_stable_ridge",
    "interval_overlap_fraction",
    "optimizer_scale_equivalent",
    "rolling_prefix_slow_level",
    "selection_period_mean_oracle",
    "validate_arm_contracts",
    "validate_machine_report",
]
