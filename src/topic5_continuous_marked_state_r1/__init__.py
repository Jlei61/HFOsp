"""R1 exact continuous marked-state measurement instrument.

R1 is intentionally isolated from the accepted R0.1 package.  It supplies the
recorded-time point-process likelihood and the exact tied-group mark law before
any persistent-state experiment is allowed to run.
"""

from .contract import REVISION

__all__ = ["REVISION"]
