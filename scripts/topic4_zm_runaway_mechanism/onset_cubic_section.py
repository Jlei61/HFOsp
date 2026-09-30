"""Cubic dense output for the unchanged full-state delayed-flow return.

Only the numerical interpolation of the section crossing changes. Four actual
neighboring time steps interpolate every state and canonical history lag. The
variational return uses the identical weights and implicit crossing derivative.
"""
from common import np
from onset_poincare_corrector import SectionReturn
from onset_tangent_cuda import Tangent
from fine_rate_frozen_Z_fields import capture,restore
from scipy.optimize import brentq


def weights(a):
    return np.array([-a*(a-1)*(a-2)/6,
        (a+1)*(a-1)*(a-2)/2,
        -(a+1)*a*(a-2)/2,(a+1)*a*(a-1)/6])


def derivative_weights(a):
    return np.array([-(3*a*a-6*a+2)/6,
        (3*a*a-4*a-1)/2,
        -(3*a*a-2*a-2)/2,(3*a*a-1)/6])


class CubicSectionReturn(SectionReturn):
    def __call__(self,x):
        e=self.e;c=self.c
        assert self.admissible(x),'No inadmissible or clamped input state'
        restore(e,self.state(x))
        lo=max(2,int(np.floor(self.period-self.halfwidth)))
        hi=int(np.ceil(self.period+self.halfwidth))
        whole=(lo-1)//10
        for _ in range(whole):e.chunk()
        for _ in range(lo-1-10*whole):
            self.one_ms.launch(e.stream);e.stream.synchronize()
        previous=capture(e)
        self.one_ms.launch(e.stream);e.stream.synchronize()
        left=capture(e);u=c.pack(left);g=float(self.normal@(u-self.xref))
        if g>=0:raise RuntimeError('Section crossing precedes the declared search window')
        for tm in range(lo+1,hi+1):
            self.one_ms.launch(e.stream);e.stream.synchronize()
            right=capture(e);v=c.pack(right);h=float(self.normal@(v-self.xref))
            if g<=0<h:break
            previous=left;left=right;u=v;g=h
        else:raise RuntimeError('No positively oriented section return in declared window')
        # Start one millisecond before the left bracket so the four-point
        # interpolant includes the preceding actual step even at alpha~0.
        restore(e,previous);u=c.pack(previous);previous_u=None
        g=float(self.normal@(u-self.xref));nms=round(1/e.dt)
        for k in range(1,2*nms+1):
            e.step();e.cp.cuda.get_current_stream().synchronize()
            v=c.pack(capture(e));h=float(self.normal@(v-self.xref))
            if k>nms and g<=0<h:
                assert previous_u is not None
                e.step();e.cp.cuda.get_current_stream().synchronize()
                points=[previous_u,u,v,c.pack(capture(e))]
                scalar=np.array([self.normal@(q-self.xref) for q in points])
                a=brentq(lambda b:weights(b)@scalar,0.,1.,xtol=1e-14)
                w=weights(a);dw=derivative_weights(a)
                value=sum(b*q for b,q in zip(w,points))
                self.last_time_slope=sum(b*q for b,q in zip(dw,points))/e.dt
                period=tm-2+(k-1+a)*e.dt
                self.calls+=1
                assert abs(self.normal@(value-self.xref))<1e-9
                assert self.normal@self.last_time_slope>0
                return value,dict(period_ms=float(period),fraction=float(a),
                    section_residual=float(self.normal@(value-self.xref)),
                    section_speed=float(self.normal@self.last_time_slope),
                    initial_phase=float(self.normal@(x-self.xref)),interpolation='cubic_four_actual_steps')
            previous_u=u;u=v;g=h
        raise RuntimeError('Fine cubic section bracket failed')


class CubicSectionDerivative:
    def __init__(self,A,x,T,slope,tangent_class=Tangent):
        self.A=A;self.x=x.copy();self.T=T;self.slope=slope.copy()
        self.normal=A.normal;self.den=float(self.normal@self.slope);assert self.den>0
        e=A.e;restore(e,A.state(x));self.t=tangent_class(e);self.t.graph()
        n=int(np.floor(T/e.dt));self.alpha=T/e.dt-n
        self.whole=(n-1)//round(10/e.dt);tail=(n-1)%round(10/e.dt)
        with self.t.stream:
            self.t.stream.begin_capture()
            for _ in range(tail):self.t.step()
            self.tail=self.t.stream.end_capture()
        self.calls=0

    def __call__(self,v):
        A=self.A;e=A.e;t=self.t;c=A.c
        restore(e,A.state(self.x));c.set_tangent(t,v)
        for _ in range(self.whole):t.chunk()
        self.tail.launch(t.stream);t.stream.synchronize();points=[c.tangent(t)]
        for _ in range(3):
            t.step();e.cp.cuda.get_current_stream().synchronize();points.append(c.tangent(t))
        fixed=sum(a*q for a,q in zip(weights(self.alpha),points))
        self.last_return_time_derivative=-float(self.normal@fixed)/self.den
        self.calls+=1
        return fixed+self.slope*self.last_return_time_derivative


def polynomial_check():
    nodes=np.array([-1.,0.,1.,2.])
    for a in np.linspace(0,1,17):
        for p in range(4):
            assert abs(weights(a)@(nodes**p)-a**p)<1e-14
            target=p*a**(p-1) if p else 0.
            assert abs(derivative_weights(a)@(nodes**p)-target)<1e-14


if __name__=='__main__':polynomial_check()
