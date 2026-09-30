"""Ordered accumulation with a target-major delay ring for cache locality."""
import numpy as np
from numba import njit, prange, set_num_threads


@njit(parallel=True, cache=True)
def target_scatter(ring, active, ptr, source, delay, weight, step, gain):
    for target in prange(len(ptr)-1):
        for edge in range(ptr[target],ptr[target+1]):
            if active[source[edge]]:
                slot=(step+delay[edge])%ring.shape[1]
                product=weight[edge]*gain
                ring[target,slot]=ring[target,slot]+product


@njit(cache=True)
def source_scatter(ring, fired, ptr, target, delay, weight, step, gain):
    for i in range(len(fired)):
        source=fired[i]
        for edge in range(ptr[source],ptr[source+1]):
            slot=(step+delay[edge])%ring.shape[1]
            product=weight[edge]*gain
            ring[target[edge],slot]=ring[target[edge],slot]+product


class Incoming:
    def __init__(self, ring, ptr, dst, delay, weight):
        self.ring=ring
        self.local=np.ascontiguousarray(ring.T)
        self.signature=tuple(id(a) for a in [ptr,dst,delay,weight])
        sources=np.repeat(np.arange(len(ptr)-1,dtype=np.int32),np.diff(ptr))
        order=np.argsort(dst,kind='stable')
        self.ptr=np.r_[0,np.cumsum(np.bincount(dst,minlength=ring.shape[1]))].astype(np.int64)
        self.source,self.delay,self.weight=sources[order],delay[order],weight[order].astype(np.float64)
        self.active=np.zeros(len(ptr)-1,np.bool_)

    def before_step(self, step):
        row=int(step)%self.ring.shape[0]
        self.ring[row]=self.local[:,row]
        self.local[:,row]=0.

    def flush(self):
        self.ring[:]=self.local.T


class Manager:
    def __init__(self, threads=8, threshold=1000000):
        set_num_threads(threads)
        self.items={};self.threshold=threshold

    def scatter(self, ring, fired, ptr, dst, delay, weight, step, gain=1.):
        key=id(ring)
        if key not in self.items:self.items[key]=Incoming(ring,ptr,dst,delay,weight)
        item=self.items[key]
        assert item.ring is ring and item.signature==tuple(id(a) for a in [ptr,dst,delay,weight])
        count=np.sum(ptr[fired+1]-ptr[fired])
        if count<self.threshold:
            source_scatter(item.local,fired,ptr,dst,delay,weight,int(step),float(gain))
        else:
            item.active.fill(False);item.active[fired]=True
            target_scatter(item.local,item.active,item.ptr,item.source,item.delay,item.weight,int(step),float(gain))

    def before_step(self, step):
        for item in self.items.values():item.before_step(step)

    def flush(self):
        for item in self.items.values():item.flush()


def wrap_simulator(original, device_index=None):
    def simulate(p,net,*args,**kwargs):
        assert kwargs.get('fast_scatter') is True and kwargs.get('ee_std_u',0.)==0.
        import checkpoint
        import src.topic4_serial_spike_scatter as serial
        scatter0,capture0=serial.scatter,checkpoint.capture
        manager=Manager()
        nu=kwargs.get('nu_signal_fn')
        if nu is None:
            nu_theta=original.__globals__['compute_nu_theta'](p)[0]
            constant_rate=p.nu_ext_ratio*nu_theta
            nu=lambda tm:constant_rate
        def before_input(tm):
            manager.before_step(round(tm/p.dt));return nu(tm)
        def capture(*a,**kw):
            manager.flush();return capture0(*a,**kw)
        kwargs['nu_signal_fn']=before_input
        serial.scatter=manager.scatter;checkpoint.capture=capture
        try:return original(p,net,*args,**kwargs)
        finally:
            manager.flush();serial.scatter=scatter0;checkpoint.capture=capture0
    return simulate
