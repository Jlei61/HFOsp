"""Target-owned CPU accumulation with native source/edge addition order."""
import numpy as np
from numba import njit, prange, set_num_threads


@njit(parallel=True,cache=True)
def accumulate(ring,active,ptr,source,delay,weight,step,gain):
    for target in prange(len(ptr)-1):
        for edge in range(ptr[target],ptr[target+1]):
            if active[source[edge]]:
                slot=(step+delay[edge])%ring.shape[0]
                product=weight[edge]*gain
                ring[slot,target]=ring[slot,target]+product


class TargetScatter:
    def __init__(self,serial,threads=8,threshold=100000):
        self.serial=serial;self.threshold=threshold;self.items={}
        set_num_threads(threads)

    def __call__(self,ring,fired,indptr,dst,delay,weight,step,gain=1.):
        count=np.sum(indptr[fired+1]-indptr[fired])
        if count<self.threshold:
            return self.serial(ring,fired,indptr,dst,delay,weight,step,gain)
        key=id(ring)
        signature=tuple(id(a) for a in [indptr,dst,delay,weight])
        if key not in self.items:
            source=np.repeat(np.arange(len(indptr)-1,dtype=np.int32),np.diff(indptr))
            order=np.argsort(dst,kind='stable')
            ptr=np.r_[0,np.cumsum(np.bincount(dst,minlength=ring.shape[1]))].astype(np.int64)
            self.items[key]=(signature,ptr,source[order],delay[order],weight[order].astype(np.float64),np.zeros(len(indptr)-1,np.bool_))
        old,ptr,source,dly,w,active=self.items[key]
        assert old==signature
        active.fill(False);active[fired]=True
        accumulate(ring,active,ptr,source,dly,w,int(step),float(gain))

