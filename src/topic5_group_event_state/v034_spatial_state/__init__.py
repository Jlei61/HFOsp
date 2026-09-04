"""v0.3.4 spatial predictive-state pilot.

This package is intentionally separate from the v0.3.3 training laboratory.
It trains the spatial view first and does not expose development, seizure or
sealed targets.
"""

from .contracts import (
    ArchConfig,
    EVALUATION_SUBJECTS,
    OptimizerConfig,
    SEED_CONTRACT,
    TrainConfig,
    build_evaluation_release_gate,
    build_human_release_gate,
    build_locked_recipe_manifest,
    require_evaluation_release_gate,
    require_synthetic_recovery,
    require_human_release_gate,
)
from .model import SpatialStateModel

__all__ = [
    "ArchConfig",
    "EVALUATION_SUBJECTS",
    "OptimizerConfig",
    "SEED_CONTRACT",
    "TrainConfig",
    "SpatialStateModel",
    "build_human_release_gate",
    "build_locked_recipe_manifest",
    "build_evaluation_release_gate",
    "require_synthetic_recovery",
    "require_human_release_gate",
    "require_evaluation_release_gate",
]
