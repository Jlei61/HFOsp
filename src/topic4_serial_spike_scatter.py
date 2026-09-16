"""Serial, order-preserving spike scatter; no fast-math or parallel reductions."""
from numba import njit


@njit(cache=True)
def scatter(ring, sources, indptr, dst, delay, weight, absolute_step, gain=1.0):
    for source in sources:
        for edge in range(indptr[source], indptr[source+1]):
            ring[(absolute_step+delay[edge]) % ring.shape[0], dst[edge]] += weight[edge]*gain
