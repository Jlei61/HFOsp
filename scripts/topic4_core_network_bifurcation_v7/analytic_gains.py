"""Derivatives of the frozen transfer implementation, including its quadrature.

This does not change the transfer function or the saved nonlinear orbits.
It avoids finite-difference gain errors amplified during long saddle visits.
"""
from common import *
import numpy as np,math
from numba import njit

@njit(cache=True)
def batch_gains(mu,ve,vi,tm,ref,threshold,tw,xx,qw,reset):
    out=np.zeros((3,len(mu),6));rootpi=math.sqrt(math.pi)
    for k in range(len(mu)):
        for i in range(6):
            total=ve[k,i]+vi[k,i];variance=max(total,1e-14);sig=math.sqrt(variance)
            mixed=(ve[k,i]*4.2+vi[k,i]*19.)/tm[i];root=math.sqrt(max(mixed,0.));shift=1.0325*root
            dsE=1.0325*4.2/(2*tm[i]*root) if root>0 else 0.
            dsI=1.0325*19./(2*tm[i]*root) if root>0 else 0.
            inv2var=1/(2*variance) if total>1e-14 else 0.
            for j in range(threshold.shape[1]):
                if tw[i,j]==0.:continue
                width=(threshold[i,j]-reset)/(2*sig);s0=0.;s1=0.;su=0.;overflow=False
                for l in range(len(xx)):
                    u=(reset*(1-xx[l])/2+threshold[i,j]*(1+xx[l])/2-mu[k,i]+shift)/sig
                    if u>26.:overflow=True;break
                    if u<-20.:
                        y=-u;t=1/(2*y*y)
                        z=(1-t+3*t*t-15*t**3+105*t**4)/(rootpi*y)
                        dz=(1/y**2-1.5/y**4+3.75/y**6-13.125/y**8+59.0625/y**10)/rootpi
                    else:
                        z=math.exp(u*u)*math.erfc(-u);dz=2*u*z+2/rootpi
                    s0+=qw[l]*z;s1+=qw[l]*dz;su+=qw[l]*dz*u
                if overflow:continue
                inv=1/(ref[i]+tm[i]*rootpi*width*s0);factor=-tw[i,j]*tm[i]*rootpi*inv*inv
                out[0,k,i]+=factor*(-width*s1/sig)
                out[1,k,i]+=factor*width*(s1*dsE/sig-inv2var*(s0+su))
                out[2,k,i]+=factor*width*(s1*dsI/sig-inv2var*(s0+su))
    return out

def gains(s,mu,ve,vi):
    return batch_gains(np.ascontiguousarray(mu),np.ascontiguousarray(ve),np.ascontiguousarray(vi),s.tm,s.ref,s.threshold,s.tw,s.x,s.qw,s.p['V_reset'])

if __name__=='__main__':
    from periodic import Orbit
    path=OUT/'periodic/long_period_tail/T2200_N16384.npz';z=np.load(path);s=System();r=z['r'];T=float(z['T']);o=Orbit(s,float(z['g']),len(r));H,*_=o.kernels(T)
    vals=[s.ext_mu+o.mean(r,H),s.ext_var+(r[:,:3]@o.Q[:,:3].T)*s.tm,(r[:,3:]@o.Q[:,3:].T)*s.tm]
    exact=gains(s,*vals);freq=2j*np.pi*np.arange(len(r)//2+1)/T
    def derivative(x):return np.fft.irfft(freq[:,None]*np.fft.rfft(x,axis=0),n=len(r),axis=0)
    target=derivative(o.phi(*vals));dv=np.array([derivative(v) for v in vals]);rows=[]
    for eps in (1e-4,1e-5,1e-6,1e-7):
        finite=[]
        for j in range(3):
            h=eps*np.maximum(abs(vals[j]),1.);hi=vals.copy();lo=vals.copy();hi[j]=vals[j]+h;lo[j]=vals[j]-h;finite.append((o.phi(*hi)-o.phi(*lo))/(2*h))
        finite=np.array(finite);rows.append(dict(step=eps,max_gain_difference=float(abs(finite-exact).max()),chain_rule_error=float(abs((finite*dv).sum(0)-target).max())))
    row=dict(source=str(path),analytic_chain_rule_error=float(abs((exact*dv).sum(0)-target).max()),finite_difference=rows)
    write('analytic_gain_validation.json',row);print(row,flush=True)
