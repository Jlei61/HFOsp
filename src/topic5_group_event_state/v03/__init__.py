"""Group-event predictive state v0.3.

The package keeps three scientific objects separate:

``FrozenContactGrammar``
    A patient-specific within-event decoder.  It is calibrated on outer TRAIN
    and then frozen.  Gradients may pass *through* it to the state adapter.

``FixedTimescaleEventState``
    The only cross-event memory.  It evolves in physical seconds and is updated
    only after the current event has been scored.

``PointProcessTerms``
    The event-time likelihood, including survival evidence from recorded
    event-free intervals.  Gaps and excluded seizure/postictal intervals never
    enter this likelihood.
"""

from .grammar import (
    FrozenContactGrammar,
    GrammarInputs,
    build_train_only_grammar,
    load_legacy_grammar,
)
from .point_process import PointProcessTerms, interval_point_process_terms
from .state import FixedTimescaleEventState, StateConfig

__all__ = [
    "FixedTimescaleEventState",
    "FrozenContactGrammar",
    "GrammarInputs",
    "PointProcessTerms",
    "StateConfig",
    "build_train_only_grammar",
    "interval_point_process_terms",
    "load_legacy_grammar",
]
