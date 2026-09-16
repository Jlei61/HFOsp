"""Same serial floating additions, precomputed integer delay-ring slot lookup."""
import numpy as np
from numba import njit


@njit(cache=True)
def scatter(ring,sources,indptr,dst,delay,weight,absolute_step,gain=1.0):
    slots=np.empty(ring.shape[0],dtype=np.int64)
    for lag in range(ring.shape[0]):
        slots[lag]=(absolute_step+lag)%ring.shape[0]
    for source in sources:
        for edge in range(indptr[source],indptr[source+1]):
            ring[slots[delay[edge]],dst[edge]]+=weight[edge]*gain
