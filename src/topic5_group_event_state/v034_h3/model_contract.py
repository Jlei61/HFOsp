"""Matched M0/M1/M2 feedback-arm definitions.

All arms have the same fitted state intercept and the same low-rank edge
template.  What differs is the causal source supplied to that edge.  This
prevents the real arm from receiving the free state offset that invalidated the
August 26 comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FeedbackArmContract:
    name: str
    scientific_role: str
    source: str
    state_dim: int
    source_rank: int
    fitted_state_intercept: bool
    rolling_prefix_slow_level: bool
    source_is_event_feedback: bool
    source_is_causal: bool
    trainable_parameters: int

    def as_dict(self) -> dict:
        return asdict(self)


def build_feedback_arm_contracts(*, state_dim: int = 16, source_rank: int = 4) -> tuple[FeedbackArmContract, ...]:
    if state_dim <= 0 or source_rank <= 0:
        raise ValueError("state_dim and source_rank must be positive")
    # One state intercept plus one rank->state map.  Source projections are
    # fixed TRAIN-only constructions, so parameter counts are exactly equal.
    n_params = int(state_dim + state_dim * source_rank)
    common = dict(
        state_dim=int(state_dim), source_rank=int(source_rank), fitted_state_intercept=True,
        rolling_prefix_slow_level=True, source_is_causal=True, trainable_parameters=n_params,
    )
    arms = (
        FeedbackArmContract(
            name="M0_common_drive", scientific_role="common-drive/readout-only comparator",
            source="causal background plus rolling-prefix slow-level residual basis",
            source_is_event_feedback=False, **common,
        ),
        FeedbackArmContract(
            name="M1_burden_feedback", scientific_role="event burden feedback",
            source="count/load innovation conditional on pre-event state and rolling slow level",
            source_is_event_feedback=True, **common,
        ),
        FeedbackArmContract(
            name="M2_mark_feedback", scientific_role="mark/waveform-specific feedback",
            source="rate/time-preserving mark and waveform innovation projected to fixed TRAIN rank",
            source_is_event_feedback=True, **common,
        ),
    )
    validate_arm_contracts(arms)
    return arms


def validate_arm_contracts(arms: tuple[FeedbackArmContract, ...] | list[FeedbackArmContract]) -> None:
    names = {a.name for a in arms}
    required = {"M0_common_drive", "M1_burden_feedback", "M2_mark_feedback"}
    if names != required:
        raise ValueError(f"need exactly {sorted(required)}, got {sorted(names)}")
    counts = {a.trainable_parameters for a in arms}
    intercepts = {a.fitted_state_intercept for a in arms}
    ranks = {a.source_rank for a in arms}
    dims = {a.state_dim for a in arms}
    if len(counts) != 1 or intercepts != {True} or len(ranks) != 1 or len(dims) != 1:
        raise ValueError("M0/M1/M2 must have identical parameter template and fitted intercept")
    if not all(a.rolling_prefix_slow_level and a.source_is_causal for a in arms):
        raise ValueError("every primary H3 arm must include causal rolling slow-level control")
