"""Experimental CUDA target ownership with original per-cell addition order.

Not installed into any scientific worker by importing this module. Explicit
round-to-nearest multiply and add avoid fused or reordered accumulation.
"""
import numpy as np
from numba import cuda
from numba.cuda import libdevice


@cuda.jit
def _accumulate(ring, active, ptr, source, delay, weight, step, gain):
    target = cuda.grid(1)
    if target < ptr.size-1:
        for edge in range(ptr[target], ptr[target+1]):
            if active[source[edge]]:
                slot = (step+delay[edge]) % ring.shape[0]
                product = libdevice.dmul_rn(weight[edge], gain)
                ring[slot,target] = libdevice.dadd_rn(ring[slot,target], product)


@cuda.jit
def _zero_row(ring, row):
    target = cuda.grid(1)
    if target < ring.shape[1]:
        ring[row,target] = 0.


class Incoming:
    def __init__(self, ring, indptr, dst, delay, weight):
        assert ring.dtype == np.float64 and ring.ndim == 2
        assert len(dst) == len(delay) == len(weight) == indptr[-1]
        assert np.all(delay >= 0) and np.all(delay < ring.shape[0])
        self.ring = ring
        self.signature = tuple(id(a) for a in [indptr,dst,delay,weight])
        self.active = np.zeros(len(indptr)-1, np.bool_)
        source = np.repeat(np.arange(len(indptr)-1,dtype=np.int32), np.diff(indptr))
        # Stable flat-edge order is the original ascending-source/within-row
        # accumulation order even for duplicate target/delay pairs.
        order = np.argsort(dst,kind='stable')
        ptr = np.r_[0,np.cumsum(np.bincount(dst,minlength=ring.shape[1]))].astype(np.int64)
        arrays = [ptr, source[order], delay[order], weight[order].astype(np.float64)]
        required = ring.nbytes+self.active.nbytes+sum(a.nbytes for a in arrays)
        free,_ = cuda.current_context().get_memory_info()
        if required > min(8*2**30, free-2*2**30):
            raise MemoryError('Bounded GPU experiment memory allowance exceeded')
        self.d_ring = cuda.to_device(ring)
        self.d_active = cuda.to_device(self.active)
        self.d_ptr,self.d_source,self.d_delay,self.d_weight = [cuda.to_device(a) for a in arrays]
        self.blocks = (ring.shape[1]+127)//128

    def accumulate(self, sources, step, gain):
        assert not len(sources) or (sources[0]>=0 and sources[-1]<len(self.active))
        assert len(sources)<2 or np.all(sources[1:]>sources[:-1])
        self.active.fill(False)
        self.active[sources] = True
        self.d_active.copy_to_device(self.active)
        _accumulate[self.blocks,128](self.d_ring,self.d_active,self.d_ptr,
            self.d_source,self.d_delay,self.d_weight,int(step),float(gain))

    def before_step(self, step):
        row = int(step) % self.ring.shape[0]
        self.d_ring[row].copy_to_host(self.ring[row])
        _zero_row[self.blocks,128](self.d_ring,row)

    def flush(self):
        self.d_ring.copy_to_host(self.ring)


class Manager:
    def __init__(self):
        self.items = {}

    def scatter(self, ring, sources, indptr, dst, delay, weight, absolute_step, gain=1.):
        key = id(ring)
        if key not in self.items:
            self.items[key] = Incoming(ring,indptr,dst,delay,weight)
        item = self.items[key]
        assert item.ring is ring
        assert item.signature == tuple(id(a) for a in [indptr,dst,delay,weight])
        item.accumulate(sources,absolute_step,gain)

    def before_step(self, step):
        for item in self.items.values(): item.before_step(step)

    def flush(self):
        for item in self.items.values(): item.flush()


def wrap_simulator(original, device_index=1):
    """Explicit opt-in wrapper; original engine source stays untouched.

    The native rate callback runs before ring consumption on every step,
    including silent ones. Native capture sees fully synchronized rings.
    """
    def simulate(p, net, *args, **kwargs):
        assert kwargs.get('fast_scatter') is True
        assert kwargs.get('ee_std_u',0.) == 0.
        cuda.select_device(device_index)
        import checkpoint
        import src.topic4_serial_spike_scatter as serial
        previous_scatter = serial.scatter
        previous_capture = checkpoint.capture
        manager = Manager()
        nu_fn = kwargs.get('nu_signal_fn')
        if nu_fn is None:
            nu_theta = original.__globals__['compute_nu_theta'](p)[0]
            constant_rate = p.nu_ext_ratio*nu_theta
            nu_fn = lambda tm: constant_rate

        def before_input(tm):
            manager.before_step(round(tm/p.dt))
            return nu_fn(tm)

        def capture(*capture_args, **capture_kwargs):
            manager.flush()
            return previous_capture(*capture_args, **capture_kwargs)

        kwargs['nu_signal_fn'] = before_input
        serial.scatter = manager.scatter
        checkpoint.capture = capture
        try:
            return original(p,net,*args,**kwargs)
        finally:
            # Includes observer/stop exceptions: external checkpoint already has
            # synchronized complete state, and the native module is restored.
            manager.flush()
            serial.scatter = previous_scatter
            checkpoint.capture = previous_capture
            manager.items.clear()
            cuda.current_context().deallocations.clear()
    return simulate
