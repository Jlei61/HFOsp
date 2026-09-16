"""Experimental target-local updates retaining each ring cell's addition order.

Unused by physical workers. Incoming edges retain stable original flat-edge order;
the SNN's source list must be sorted and unique. No fastmath or parallel reduction.
"""
import numpy as np
from numba import njit


def prepare(indptr,dst,delay,weight,n_targets):
    source=np.repeat(np.arange(len(indptr)-1,dtype=np.int32),np.diff(indptr))
    order=np.argsort(dst,kind='stable')
    counts=np.bincount(dst,minlength=n_targets)
    target_ptr=np.r_[0,np.cumsum(counts)].astype(np.int64)
    return target_ptr,source[order],delay[order],weight[order]


@njit(cache=True)
def gather(ring,sources,target_ptr,in_source,in_delay,in_weight,n_sources,absolute_step,gain=1.):
    active=np.zeros(n_sources,np.bool_)
    for source in sources:active[source]=True
    slots=np.empty(ring.shape[0],np.int64)
    for lag in range(ring.shape[0]):slots[lag]=(absolute_step+lag)%ring.shape[0]
    for target in range(len(target_ptr)-1):
        for edge in range(target_ptr[target],target_ptr[target+1]):
            if active[in_source[edge]]:
                ring[slots[in_delay[edge]],target]+=in_weight[edge]*gain
