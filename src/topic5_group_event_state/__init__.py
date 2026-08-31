"""Group-event-driven latent state models for Topic 5.

The package treats one complete interictal group event as one causal sequence
step.  Continuous background SEEG is an optional observation of the state; it
is not the state clock and it is not a replacement for the event sequence.
"""

from .contract import (
    ANALYSIS_BANDS_HZ,
    EVENT_CONTEXT_POST_SECONDS,
    EVENT_CONTEXT_PRE_SECONDS,
    EventSourcePointer,
    relative_participant_delay,
)

__all__ = [
    "ANALYSIS_BANDS_HZ",
    "EVENT_CONTEXT_POST_SECONDS",
    "EVENT_CONTEXT_PRE_SECONDS",
    "EventSourcePointer",
    "relative_participant_delay",
]
