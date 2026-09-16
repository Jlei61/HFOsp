"""Group-Event State v0.3.7 continuous-time observer primitives.

The package deliberately reserves ``observer`` for causal predictive summaries.
H3 generative physiological states live in a separate future namespace and may
not reuse these classes as evidence of an event-driven physiological jump.
"""

from .baselines import MarkEWMAOutput, fixed_mark_ewma, mark_ewma_features_at_queries
from .contracts import (
    ARCHITECTURE_VERSION,
    CheckpointEntry,
    VisibilityStamp,
    assert_causal_visibility,
    update_checkpoint_registry,
)
from .ctssm import (
    BackgroundCTSSMOutput,
    CTSSMOutput,
    DiagonalEventCTSSM,
    DualStreamCTSSMOutput,
    DualStreamEventCTSSM,
    GridBackgroundCTSSM,
    dual_stream_features_at_queries,
    affine_associative_scan,
    affine_sequential_scan,
    zoh_diagonal_step,
)

__all__ = [
    "ARCHITECTURE_VERSION",
    "BackgroundCTSSMOutput",
    "CTSSMOutput",
    "CheckpointEntry",
    "DiagonalEventCTSSM",
    "DualStreamCTSSMOutput",
    "DualStreamEventCTSSM",
    "GridBackgroundCTSSM",
    "dual_stream_features_at_queries",
    "MarkEWMAOutput",
    "VisibilityStamp",
    "affine_associative_scan",
    "affine_sequential_scan",
    "assert_causal_visibility",
    "fixed_mark_ewma",
    "mark_ewma_features_at_queries",
    "update_checkpoint_registry",
    "zoh_diagonal_step",
]
