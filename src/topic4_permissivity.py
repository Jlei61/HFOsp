"""M3 W-coupled permissivity threshold delta (Task 2, spec §5.3 U3).

Returns a per-neuron threshold INCREMENT vector to be added to V_th_per_neuron
*before* simulate_kick.  Pure helper — zero engine imports.

Contract:
  - I cells (last NI entries): delta = 0 (never moved).
  - E cells (first NE entries):
        delta[i] = -delta_theta * mu * h_eff[bin_of_cell[i]]
    Sign is NEGATIVE = lowers threshold = raises permissivity.
  - control='none'    -> h_eff = h  (primary)
  - control='uniform' -> h_eff = ones (uniform-mu control, C5)
  - control='shuffle' -> h_eff = h permuted across bins via rng (shuffled-h control, C5)
  - mu=0 => returned vector is exactly all-zeros (bit-parity with engine baseline M3_BASE_SHA).
"""
from __future__ import annotations
import numpy as np


def permissivity_vth_delta(
    h: np.ndarray,
    bin_of_cell: np.ndarray,
    NE: int,
    NI: int,
    *,
    mu: float,
    delta_theta: float,
    control: str = "none",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return threshold increment vector of length NE+NI.

    Parameters
    ----------
    h : (n_bins,) array — per-bin susceptibility map (from h_field(W_resp, scheme)).
    bin_of_cell : (NE,) int array — bin index for each E cell.
    NE, NI : number of E and I cells.
    mu : permissivity scalar (>= 0).  mu=0 -> exact zero vector.
    delta_theta : threshold shift magnitude in mV.
    control : 'none' | 'uniform' | 'shuffle'.
    rng : numpy Generator; required when control='shuffle'.
    """
    delta = np.zeros(NE + NI, dtype=float)

    if mu == 0.0:
        return delta  # bit-parity: engine is unchanged

    if control == "none":
        h_eff = h
    elif control == "uniform":
        h_eff = np.ones(len(h), dtype=float)
    elif control == "shuffle":
        if rng is None:
            raise ValueError("rng must be provided for control='shuffle'")
        h_eff = rng.permutation(h)
    else:
        raise ValueError(f"Unknown control: {control!r}")

    delta[:NE] = -delta_theta * mu * h_eff[bin_of_cell]
    return delta
