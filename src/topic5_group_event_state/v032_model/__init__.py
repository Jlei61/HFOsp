"""Group-Event State v0.3.2 -- model side.

Residual marked-history predictive state: a 12-dimensional constrained leaky
bank written by event tokens and decayed in physical time, read out as a
residual on top of the explicit history baseline ``log mu_H`` supplied by the
evaluation agent.  Nothing in this package names the state a physiological
slow variable; that reading is not licensed by this instrument.
"""

PACKAGE_VERSION = "0.3.2-model"
