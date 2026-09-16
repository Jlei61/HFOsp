"""Numerical integration used only to initialize periodic-orbit solves."""
from model import System,OUT
import numpy as np,json,math
from numba import njit
from scipy.signal import find_peaks

@njit(cache=True)
def transfer(mu,ve,vi,tm,ref,threshold,tw,xx,qw,reset,colored=True):
    ans=np.zeros(6)
    for i in range(6):
        sig=math.sqrt(max(ve[i]+vi[i],1e-14))
        shift=1.0325*math.sqrt(max((ve[i]*4.2+vi[i]*19.)/tm[i],0.)) if colored else 0.
        a=(reset-mu[i]+shift)/sig
        for j in range(threshold.shape[1]):
            if tw[i,j]==0.:continue
            b=(threshold[i,j]-mu[i]+shift)/sig
            integ=0.
            for k in range(len(xx)):
                u=(a+b)/2+(b-a)/2*xx[k]
                if u>26.:
                    integ=math.inf;break
                if u<-20.:
                    y=-u;t=1/(2*y*y)
                    z=(1-t+3*t*t-15*t**3+105*t**4)/(math.sqrt(math.pi)*y)
                else:z=math.exp(u*u)*math.erfc(-u)
                integ+=qw[k]*z
            integ*=(b-a)/2
            ans[i]+=tw[i,j]/(ref[i]+tm[i]*math.sqrt(math.pi)*integ)
    return ans

@njit(cache=True)
def evolve(nsteps,dt,W,Q,tr,tm,ref,rise,decay,extmu,extvar,area,sign,threshold,tw,xx,qw,reset,r0,pulse):
    D=len(W);hist=np.empty((D+1,6));r=r0.copy();h=r0.copy();c=r0.copy()
    for i in range(D+1):hist[i]=r
    out=np.empty((nsteps,6));dout=np.empty((nsteps,18));head=0
    ar=np.exp(-dt/rise);ad=np.exp(-dt/decay);at=np.exp(-dt/tr)
    for k in range(nsteps):
        if k==int(300/dt):r+=pulse
        mu=extmu.copy();ve=extvar.copy();vi=np.zeros(6)
        for d in range(D):
            old=hist[(head-d-1)%(D+1)]
            for i in range(6):
                for j in range(6):mu[i]+=tm[i]*W[d,i,j]*area[j]*sign[j]*old[j]
        for i in range(6):
            for j in range(6):
                if j<3:ve[i]+=tm[i]*Q[i,j]*r[j]
                else:vi[i]+=tm[i]*Q[i,j]*r[j]
        ph=transfer(mu,ve,vi,tm,ref,threshold,tw,xx,qw,reset)
        # Exponential Euler. Simulated extrema are not continuation solutions.
        rn=at*r+(1-at)*ph
        hn=ar*h+(1-ar)*r
        cn=ad*c+(1-ad)*h
        r=rn;h=hn;c=cn;head=(head+1)%(D+1);hist[head]=c
        out[k]=r;dout[k,:6]=mu;dout[k,6:12]=ve;dout[k,12:]=vi
    return out,dout

def simulate(s,g,T=4000,pulse=.02,r0=None):
    if r0 is None:r0=np.array([.0002,.00018,0,0,0,0])
    W,Q=s.weights(g)
    return evolve(round(T/s.dt),s.dt,W,Q,s.tr,s.tm,s.ref,s.rise,s.decay,s.ext_mu,s.ext_var,s.area,s.sign,s.threshold,s.tw,s.x,s.qw,s.p['V_reset'],r0,np.array([pulse,pulse,0,0,0,0]))

if __name__=='__main__':
    s=System();dest=OUT/'dynamics';dest.mkdir(exist_ok=True)
    # Agreement of the fast transfer with the defining scipy implementation.
    rng=np.random.default_rng(13);errs=[]
    for k in range(100):
        rr=np.exp(rng.uniform(-10,-2,6));mu,ve,vi=s.moments(rr,1.)
        fast=transfer(mu,ve,vi,s.tm,s.ref,s.threshold,s.tw,s.x,s.qw,s.p['V_reset'])
        errs.append(abs(fast-s.phi(mu,ve,vi)).max())
    print('TRANSFER MAX ERROR',max(errs),flush=True)
    rows=[]
    for g in [.7,.85,1.,1.12,1.15,1.3]:
        rr,dd=simulate(s,g,T=6000)
        np.savez_compressed(dest/f'g{g:g}.npz',r=rr,moments=dd,dt=s.dt,g=g)
        pk,_=find_peaks(rr[:,0],height=.005,distance=round(50/s.dt));pk=pk[pk*s.dt>2000]
        row=dict(g=g,mean_hz=(rr[-20000:].mean(0)*1000).tolist(),min_hz=(rr[-20000:].min(0)*1000).tolist(),max_hz=(rr[-20000:].max(0)*1000).tolist(),period_ms=(np.diff(pk)*s.dt).tolist())
        rows.append(row);print(json.dumps(row),flush=True)
    (OUT/'dynamics_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
