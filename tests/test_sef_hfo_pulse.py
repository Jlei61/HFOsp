# tests/test_sef_hfo_pulse.py
import numpy as np
from src.sef_hfo_pulse import classify_response

def _act(extents, centroids, n=16):
    """Build (T,n,n) E-activity with prescribed active-fraction and centroid-x per frame.
    frac=0 -> a truly empty frame (real extinction); the active block is CENTERED on cx so
    the measured centroid tracks cx (a growing-but-stationary flash keeps centroid fixed).
    NOTE: deviation from plan-verbatim helper, which floored width at 1 column (so frac=0
    never extinguished) and grew the block rightward (so a stationary flash's centroid
    drifted) -- both made the helper unable to realize the extinction / global_synchronous
    cases. The classifier itself is unchanged. See step0_results writeup."""
    T = len(extents); a = np.zeros((T, n, n)); cells = n*n
    for t, (frac, cx) in enumerate(zip(extents, centroids)):
        k = int(round(frac*cells))
        if k <= 0:
            continue                                   # true extinction frame
        width = max(1, int(round(k / n)))              # columns (each column = n cells)
        c0 = int(cx) % n
        lo = max(0, c0 - width // 2); hi = min(n, lo + width)
        a[t, :, lo:hi] = 1.0
    return a

def test_classifier_five_regimes():
    n = 16; stim = np.zeros((n, n)); stim[:, 7:9] = 1.0; rest = 0.0
    kw = dict(stim_mask=stim, rest_level=rest, runaway_frac=0.5, return_frac=0.2, detect=0.5)
    extinction = _act([0.06, 0.06, 0.0, 0.0], [8, 8, 8, 8])
    local_bump = _act([0.06]*6, [8]*6)
    self_lim   = _act([0.06, 0.18, 0.28, 0.0], [8, 10, 12, 12])     # extent grows AND centroid moves, returns
    runaway    = _act([0.06, 0.25, 0.55, 0.80], [8, 9, 9, 9])
    flash      = _act([0.06, 0.40, 0.40, 0.0], [8, 8, 8, 8])         # extent grows but centroid does NOT move
    assert classify_response(extinction, **kw) == "extinction"
    assert classify_response(local_bump, **kw) == "local_bump"
    assert classify_response(self_lim,   **kw) == "self_limited_propagation"
    assert classify_response(runaway,    **kw) == "runaway"
    assert classify_response(flash,      **kw) == "global_synchronous"   # NOT propagation
