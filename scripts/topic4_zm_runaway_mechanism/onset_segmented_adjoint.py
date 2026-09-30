"""Transpose of the original segmented full delayed-state return derivative."""
from common import np
from onset_cached_adjoint import CachedAdjoint
from onset_cubic_section import weights
from onset_segment_flow import split_step


class SegmentAdjoint:
    def __init__(self, derivative):
        assert derivative.cached
        self.J=derivative;self.A=derivative.A
        self.n,self.alpha=split_step(derivative.T,self.A.e.dt)
        self.reverse=CachedAdjoint(derivative.t);self.reverse.graph()
        self.whole,self.tail=divmod(self.n-1,round(10/self.A.e.dt))

    def __call__(self,w):
        J=self.J;A=self.A;e=A.e;c=A.c;reverse=self.reverse
        # The shared cache must contain this segment's exact coefficients.
        J.t.cache.set(J.host);e.cp.cuda.get_current_stream().synchronize()
        alpha=weights(self.alpha)
        reverse.set_covector(c,alpha[3]*w,c.tick+self.n+2)
        for j in [2,1,0]:
            reverse.step();e.cp.cuda.get_current_stream().synchronize()
            reverse.add_covector(c,alpha[j]*w)
        for _ in range(self.whole):reverse.chunk()
        for _ in range(self.tail):reverse.step()
        e.cp.cuda.get_current_stream().synchronize()
        assert int(reverse.clock.get()[0])==c.tick
        return reverse.covector(c)


class SegmentedPoincareAdjoint:
    def __init__(self,derivative):
        self.J=derivative
        self.parts=[SegmentAdjoint(j) for j in derivative.derivatives]

    def __call__(self,w):
        J=self.J
        q=w-J.normal*float(J.slope@w)/J.den
        for transpose in reversed(self.parts):q=transpose(q)
        return q
