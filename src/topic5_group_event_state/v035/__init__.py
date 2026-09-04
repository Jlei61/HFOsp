"""Group-event predictive-state v0.3.5.

The package deliberately separates the causal rate/recording-stage process
``q(t)`` from the full-event content process ``m(t)``.  Public runners write
only below the v0.3.5 result root and never broaden an older version's data
contract.
"""

from .contracts import CORE_HORIZONS_SECONDS, RATE_TAUS_SECONDS, V035_SUBJECTS

__all__ = ["CORE_HORIZONS_SECONDS", "RATE_TAUS_SECONDS", "V035_SUBJECTS"]
