"""Fourth-order variational integration with cubic interpolation of delays."""
from numba import njit
import numpy as np

@njit(cache=True)
def cubic_coeff(f):
    return np.array([-f*(f-1)*(f-2)/6,(f+1)*(f-1)*(f-2)/2,-(f+1)*f*(f-2)/2,(f+1)*f*(f-1)/6])

@njit(cache=True)
def monodromy4(x,dt,di,dj,delays,val,u,var,tr,rise,decay,D):
    hist=x[12:].reshape(D+1,6).copy();r=x[:6].copy();h=x[6:12].copy();head=0
    ks=np.empty((3,len(val)),np.int64);weights=np.empty((3,len(val),4))
    for stage in range(3):
        for e in range(len(val)):
            d=delays[e]-.5*stage;k=int(np.floor(d));ks[stage,e]=k
            weights[stage,e]=cubic_coeff(d-k)*val[e]
    for step in range((len(u)-1)//2):
        mus=np.zeros((3,6))
        for stage in range(3):
            for e in range(len(val)):
                for j in range(4):
                    index=head+ks[stage,e]+j-1
                    if index>D:index-=D+1
                    mus[stage,di[e]]+=weights[stage,e,j]*hist[index,dj[e]]
        c=hist[head].copy();k=2*step
        a=(-r+var[k]@r+u[k]*mus[0])/tr;b=(r-h)/rise;cc=(h-c)/decay
        r2=r+.5*dt*a;h2=h+.5*dt*b;c2=c+.5*dt*cc
        a2=(-r2+var[k+1]@r2+u[k+1]*mus[1])/tr;b2=(r2-h2)/rise;cc2=(h2-c2)/decay
        r3=r+.5*dt*a2;h3=h+.5*dt*b2;c3=c+.5*dt*cc2
        a3=(-r3+var[k+1]@r3+u[k+1]*mus[1])/tr;b3=(r3-h3)/rise;cc3=(h3-c3)/decay
        r4=r+dt*a3;h4=h+dt*b3;c4=c+dt*cc3
        a4=(-r4+var[k+2]@r4+u[k+2]*mus[2])/tr;b4=(r4-h4)/rise;cc4=(h4-c4)/decay
        r+=dt/6*(a+2*a2+2*a3+a4);h+=dt/6*(b+2*b2+2*b3+b4);cn=c+dt/6*(cc+2*cc2+2*cc3+cc4)
        head=(head-1)%(D+1);hist[head]=cn
    out=np.empty_like(x);out[:6]=r;out[6:12]=h
    for d in range(D+1):out[12+6*d:18+6*d]=hist[(head+d)%(D+1)]
    return out
