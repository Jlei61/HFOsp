"""Full Poincare derivative with segmented variational storage.

The return time remains the original implicit section crossing. Storage
partitioning changes neither that nonlinear return nor any physical state.
"""
from common import np, log
from onset_poincare_corrector import SectionReturn
from onset_cubic_section import CubicSectionDerivative
from onset_segment_flow import fixed_times, SegmentDerivative
from onset_tangent_cuda import Tangent
from onset_cached_tangent import CachedTangent
from fine_rate_frozen_Z_fields import capture, restore


def partition(A, x, T, count):
    h=round(T/count/A.e.dt)*A.e.dt
    durations=np.r_[np.full(count-1,h),T-(count-1)*h]
    assert count>=2 and min(durations)>2*A.e.dt
    values=fixed_times(A,x,np.cumsum(durations)[:-1])
    states=[A.state(x)]+[A.state(y) for y,_ in values]
    assert all(A.admissible(A.c.pack(s)) for s in states)
    return states,durations


def cycle_coordinates(A, T, count):
    states,_=partition(A,A.xref,T,count)
    power=sum(((A.c.raw(s)**2)*(A.c.weight**2)).sum(1) for s in states)/count
    A.c.scale=np.maximum(np.sqrt(power)[:,None],A.c.floor)
    A.xref=A.c.pack(A.base)
    restore(A.e,A.base); points=[A.xref]
    for _ in range(2):
        A.e.step(); A.e.cp.cuda.get_current_stream().synchronize()
        points.append(A.c.pack(capture(A.e)))
    velocity=(-3*points[0]+4*points[1]-points[2])/(2*A.e.dt)
    A.normal=velocity/np.linalg.norm(velocity)


class SegmentedPoincareDerivative:
    def __init__(self,A,x,T,slope,count,tangent_class=Tangent,cached_tangent_class=CachedTangent,verify_partition=True):
        self.A=A; self.normal=A.normal; self.slope=slope.copy()
        self.den=float(self.normal@self.slope); assert self.den>0
        states,durations=partition(A,x,T,count)
        self.parts=[]
        for s,h in zip(states,durations):
            B=SectionReturn(s,A.e,float(h)); B.c.scale=A.c.scale.copy(); B.xref=B.c.pack(s)
            self.parts.append(B)
        self.derivatives=[None]*count; shared=None
        for j in np.argsort(-durations):
            B=self.parts[j]
            J=SegmentDerivative(B,B.xref,float(durations[j]),shared=shared,
                                tangent_class=tangent_class,cached_tangent_class=cached_tangent_class)
            self.derivatives[j]=J; shared=J.shared
            log('SEGMENTED POINCARE CACHE',j)
        if not verify_partition:
            self.partition_check=dict(status='NOT_REPEATED_AFTER_SAME_ROOT_FIRST_ITERATION_PASS',
                scope='Same full derivative and cache construction; only the redundant comparison is omitted. The first Newton iteration, final spectral map, nonlinear residual and independent phase checks remain required.')
            return
        # Compare the composed return derivative against a single original,
        # uncached variational return, including its implicit return time.
        rng=np.random.default_rng(925515); v=x*rng.normal(size=x.size)
        v-=self.normal*(self.normal@v); v/=np.linalg.norm(v)
        actual=self(v)
        fixed_actual=actual-self.slope*self.last_return_time_derivative
        original=CubicSectionDerivative(A,x,T,slope,tangent_class=tangent_class); expected=original(v)
        fixed_expected=expected-self.slope*original.last_return_time_derivative
        error=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-15))
        fixed_error=float(np.linalg.norm(fixed_actual-fixed_expected)/max(np.linalg.norm(fixed_expected),1e-15))
        self.partition_check=dict(status='PASS' if max(error,fixed_error)<1e-10 else 'FAIL',relative_error=error,
            fixed_time_relative_error=fixed_error,
            comparison='Full composed implicit Poincare derivative and fixed-time derivative versus original unsegmented uncached CubicSectionDerivative')
        assert max(error,fixed_error)<1e-10,self.partition_check
        log('SEGMENTED POINCARE ORIGINAL CHECK',self.partition_check)

    def fixed_time(self,v):
        q=v.copy()
        for J in self.derivatives:q=J(q)
        return q

    def __call__(self,v):
        q=self.fixed_time(v)
        dot=float(self.normal@q)
        self.last_return_time_derivative=-dot/self.den
        return q-self.slope*dot/self.den
