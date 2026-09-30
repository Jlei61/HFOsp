"""Memory-bounded local cubic interpolation on a uniform recorded time grid."""
import numpy as np


class LocalCubic:
    def __init__(self,t,y,axis=0):
        assert axis==0
        self.t=np.asarray(t);self.y=np.asarray(y);self.dt=float(self.t[1]-self.t[0])
        assert len(t)>=4 and np.max(abs(np.diff(t)-self.dt))<1e-8*self.dt
    def __call__(self,t,nu=0):
        assert nu in [0,1]
        target=np.asarray(t);flat=target.ravel();out=np.empty((len(flat),)+self.y.shape[1:])
        for start in range(0,len(flat),256):
            stop=min(start+256,len(flat));x=(flat[start:stop]-self.t[0])/self.dt
            i=np.clip(np.floor(x).astype(int),1,len(self.t)-3);u=x-i
            if nu==0:
                weights=[-u*(u-1)*(u-2)/6,(u+1)*(u-1)*(u-2)/2,
                         -(u+1)*u*(u-2)/2,(u+1)*u*(u-1)/6]
            else:
                weights=[(-3*u*u+6*u-2)/(6*self.dt),(3*u*u-4*u-1)/(2*self.dt),
                         (-3*u*u+2*u+2)/(2*self.dt),(3*u*u-1)/(6*self.dt)]
            shape=(-1,)+(1,)*(self.y.ndim-1)
            value=np.zeros((stop-start,)+self.y.shape[1:])
            for shift,w in zip([-1,0,1,2],weights):value+=self.y[i+shift]*w.reshape(shape)
            out[start:stop]=value
        return out.reshape(target.shape+self.y.shape[1:])


if __name__=='__main__':
    from pathlib import Path
    import json
    t=np.linspace(0,10,2001);x=np.linspace(0,10,3007)
    y=np.stack([t**3,t**2,t,np.ones_like(t)],axis=1);f=LocalCubic(t,y)
    true=np.stack([x**3,x**2,x,np.ones_like(x)],axis=1)
    deriv=np.stack([3*x*x,2*x,np.ones_like(x),np.zeros_like(x)],axis=1)
    error=float(abs(f(x)-true).max());de=float(abs(f(x,1)-deriv).max())
    assert error<1e-10 and de<1e-8,(error,de)
    print(json.dumps(dict(polynomial_error=error,polynomial_derivative_error=de)))
