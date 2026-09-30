"""Event-driven evaluation of spatial delay convolutions (same operators)."""
import numpy as np
from numba import njit


@njit(cache=False)
def schedule(indptr, target, delay, weight, square, activity, step, mean_ring, variance_ring, stochastic):
    depth = mean_ring.shape[0]
    for src in range(activity.size):
        value = activity[src]
        if value == 0.:
            continue
        for e in range(indptr[src], indptr[src+1]):
            slot = (step+delay[e])%depth
            mean_ring[slot, target[e]] += weight[e]*value
            if stochastic:
                variance_ring[slot, target[e]] += square[e]*value


class EventDelay:
    def __init__(self, weights, squares, n, depth):
        self.n, self.depth = n, depth
        self.stochastic = squares is not None
        self.mean = np.zeros((2,depth+1,2*n))
        self.var = np.zeros_like(self.mean)
        self.edges = []
        for paths in [('ee','ie'),('ei','ii')]:
            sources,targets,delays,ws,qs = [],[],[],[],[]
            for pop, key in enumerate(paths):
                w = weights[key].tocsr(); coo=w.tocoo()
                if squares is not None:
                    q=squares[key].tocsr()
                    assert np.array_equal(w.indptr,q.indptr) and np.array_equal(w.indices,q.indices)
                    qs.append(q.data)
                else:
                    qs.append(np.zeros_like(w.data))
                sources.append(coo.col%n); targets.append(coo.row+pop*n)
                delays.append(coo.col//n+1); ws.append(w.data)
            src=np.concatenate(sources); order=np.argsort(src,kind='stable')
            ptr=np.r_[0,np.cumsum(np.bincount(src,minlength=n))].astype(np.int64)
            self.edges.append((ptr,np.concatenate(targets)[order].astype(np.int32),
                               np.concatenate(delays)[order].astype(np.int32),
                               np.concatenate(ws)[order],np.concatenate(qs)[order]))

    def take(self, step):
        slot=step%(self.depth+1)
        means=self.mean[:,slot].copy(); variances=self.var[:,slot].copy()
        self.mean[:,slot]=0.; self.var[:,slot]=0.
        return means,variances

    def push(self, step, e, i):
        for pop,activity in enumerate((e,i)):
            schedule(*self.edges[pop],activity,step,self.mean[pop],self.var[pop],self.stochastic)
