"""Same source-ordered summation as EventDelay, without per-edge division."""
import numpy as np
from numba import njit
from topic4_kinetic_delay import EventDelay


@njit(cache=True)
def schedule_mean(indptr,offset,weight,activity,base,ring):
    size=ring.size
    for src in range(activity.size):
        value=activity[src]
        if value==0.:
            continue
        for e in range(indptr[src],indptr[src+1]):
            at=base+offset[e]
            if at>=size:
                at-=size
            ring[at]+=weight[e]*value


class FlatMeanDelay(EventDelay):
    def __init__(self,weights,n,depth):
        super().__init__(weights,None,n,depth)
        self.flat_edges=[(ptr,d.astype(np.int64)*(2*n)+target,w) for ptr,target,d,w,_ in self.edges]

    def push(self,step,e,i):
        base=(step%(self.depth+1))*(2*self.n)
        for pop,activity in enumerate((e,i)):
            schedule_mean(*self.flat_edges[pop],activity,base,self.mean[pop].reshape(-1))
