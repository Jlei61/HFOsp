"""Fixed-time segments and exact cached derivatives for multiple shooting.

All segments retain the full spatial/delayed state and dynamic M. Derivative
caches may share one GPU allocation; their float64 values live on the host
between uses. Only the storage location changes, never their coefficients.
"""
from common import np
from onset_cubic_section import weights,derivative_weights
from onset_cached_tangent import CachedTangent
from onset_tangent_cuda import Tangent
from fine_rate_frozen_Z_fields import capture,restore
import gc


def split_step(T,dt):
    q=T/dt
    if abs(q-round(q))<1e-9:q=float(round(q))
    n=int(np.floor(q))
    return n,q-n


def fixed_time(A,x,T):
    e=A.e;c=A.c
    assert A.admissible(x),'No clipping of physical segment states'
    restore(e,A.state(x));n,a=split_step(T,e.dt)
    assert n>=2
    whole,tail=divmod(n-1,round(10/e.dt))
    for _ in range(whole):e.chunk()
    for _ in range(tail):e.step()
    e.cp.cuda.get_current_stream().synchronize();points=[c.pack(capture(e))]
    for _ in range(3):
        e.step();e.cp.cuda.get_current_stream().synchronize();points.append(c.pack(capture(e)))
    value=sum(w*q for w,q in zip(weights(a),points))
    slope=sum(w*q for w,q in zip(derivative_weights(a),points))/e.dt
    return value,slope


def fixed_times(A,x,times):
    """Several exact-time readouts of one uninterrupted original flow.

    Use the identical four-state cubic convention as fixed_time. Only
    repeated prefixes are shared; every requested full state is retained.
    Callers must verify parity against independent fixed_time evaluations.
    """
    e=A.e;c=A.c
    assert A.admissible(x)
    times=np.asarray(times,float)
    assert times.ndim==1 and np.isfinite(times).all()
    pairs=[split_step(t,e.dt) for t in times]
    n=np.array([p[0] for p in pairs]);alpha=np.array([p[1] for p in pairs])
    assert n.min()>=2
    needed=sorted(set((n[:,None]+np.arange(-1,3)).ravel().tolist()))
    restore(e,A.state(x));current=0;points={}
    chunk=round(10/e.dt)
    for target in needed:
        whole,tail=divmod(target-current,chunk)
        for _ in range(whole):e.chunk()
        for _ in range(tail):e.step()
        e.cp.cuda.get_current_stream().synchronize()
        points[target]=c.pack(capture(e));current=target
    values=[]
    for m,a in zip(n,alpha):
        p=[points[int(m+j)] for j in range(-1,3)]
        values.append((sum(w*q for w,q in zip(weights(a),p)),
                       sum(w*q for w,q in zip(derivative_weights(a),p))/e.dt))
    return values


class SegmentDerivative:
    def __init__(self,A,x,T,shared=None,cached=True,
                 tangent_class=Tangent,cached_tangent_class=CachedTangent):
        self.A=A;self.x=x.copy();self.T=T;self.cached=cached
        e=A.e;n,self.alpha=split_step(T,e.dt)
        restore(e,A.state(x))
        self.t=cached_tangent_class(e,n+2,cache_buffer=shared) if cached else tangent_class(e)
        self.host=None
        if cached:
            # Graphs are captured only after replacing the allocation, so
            # all recorded cache pointers refer to the shared GPU buffer.
            self.host=self.t.cache.get()
            self.shared=shared if shared is not None else self.t.cache
        self.t.graph()
        self.whole,tail=divmod(n-1,round(10/e.dt))
        with self.t.stream:
            self.t.stream.begin_capture()
            for _ in range(tail):self.t.step()
            self.tail=self.t.stream.end_capture()

    def __call__(self,v):
        A=self.A;e=A.e;t=self.t;c=A.c
        if self.cached:
            self.t.cache.set(self.host);e.cp.cuda.get_current_stream().synchronize()
        restore(e,A.state(self.x));c.set_tangent(t,v)
        for _ in range(self.whole):t.chunk()
        self.tail.launch(t.stream);t.stream.synchronize();points=[c.tangent(t)]
        for _ in range(3):
            t.step();e.cp.cuda.get_current_stream().synchronize();points.append(c.tangent(t))
        return sum(w*q for w,q in zip(weights(self.alpha),points))
