"""Joint deterministic delay-system exploration and full-state orbit seeds.

Heun integration uses the physical v2 delay array and fixed native DC area.
Time averages from these trajectories are diagnostics, not continued orbits.
"""
from common import *
import numpy as np
from numba import njit
from dynamics import transfer
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
import argparse

@njit(cache=True)
def heun(nsteps,dt,di,dj,lag,frac,val,Q,tr,tm,ref,rise,decay,extmu,extvar,threshold,tw,xx,qw,reset,r,h,hist):
    hist=hist.copy();r=r.copy();h=h.copy();head=0;D=len(hist)
    out=np.empty((nsteps+1,6));out[0]=r
    for k in range(nsteps):
        mu0=extmu.copy();mu1=extmu.copy()
        for e in range(len(val)):
            old=(head-lag[e])%D;older=(old-1)%D;newer=(old+1)%D
            mu0[di[e]]+=val[e]*((1-frac[e])*hist[old,dj[e]]+frac[e]*hist[older,dj[e]])
            mu1[di[e]]+=val[e]*((1-frac[e])*hist[newer,dj[e]]+frac[e]*hist[old,dj[e]])
        ve=extvar.copy();vi=np.zeros(6)
        for i in range(6):
            for j in range(6):
                if j<3:ve[i]+=tm[i]*Q[i,j]*r[j]
                else:vi[i]+=tm[i]*Q[i,j]*r[j]
        ph=transfer(mu0,ve,vi,tm,ref,threshold,tw,xx,qw,reset)
        dr=(ph-r)/tr;dh=(r-h)/rise;dc=(h-hist[head])/decay
        rp=r+dt*dr;hp=h+dt*dh;cp=hist[head]+dt*dc
        ve=extvar.copy();vi=np.zeros(6)
        for i in range(6):
            for j in range(6):
                if j<3:ve[i]+=tm[i]*Q[i,j]*rp[j]
                else:vi[i]+=tm[i]*Q[i,j]*rp[j]
        ph=transfer(mu1,ve,vi,tm,ref,threshold,tw,xx,qw,reset)
        r+=dt*.5*(dr+(ph-rp)/tr);h+=dt*.5*(dh+(rp-hp)/rise)
        cnew=hist[head]+dt*.5*(dc+(hp-cp)/decay)
        head=(head+1)%D;hist[head]=cnew;out[k+1]=r
    final=np.empty_like(hist)
    for d in range(D):final[(-d)%D]=hist[(head-d)%D]
    return out,r,h,final

def orbit_history(s,path,dt,phase=.0):
    z=np.load(path);r=z['r'];T=float(z['T']);N=len(r);freq=2j*np.pi*np.arange(N//2+1)/T
    h=np.fft.irfft(np.fft.rfft(r,axis=0)/(1+freq[:,None]*s.rise),n=N,axis=0)
    c=np.fft.irfft(np.fft.rfft(h,axis=0)/(1+freq[:,None]*s.decay),n=N,axis=0)
    tt=np.arange(N+1)*T/N
    sr=CubicSpline(tt,np.r_[r,r[:1]],bc_type='periodic');sh=CubicSpline(tt,np.r_[h,h[:1]],bc_type='periodic');sc=CubicSpline(tt,np.r_[c,c[:1]],bc_type='periodic')
    D=int(np.ceil(s.delay[-1]/dt))+3;hist=np.empty((D,6))
    for d in range(D):hist[(-d)%D]=sc((phase*T-d*dt)%T)
    return sr(phase*T),sh(phase*T),hist

def simulate(s,g,state,dt=.1,duration=8000.):
    W,Q=s.weights(g);dd,di,dj=np.nonzero(W)
    d=s.delay[dd]/dt;lag=np.floor(d+1e-10).astype(np.int64);frac=d-lag;frac[abs(frac)<1e-9]=0
    assert lag.min()>=1
    val=W[dd,di,dj]*s.tm[di]*s.area[dj]*s.sign[dj]
    return heun(round(duration/dt),dt,di,dj,lag,frac,val,Q,s.tr,s.tm,s.ref,s.rise,s.decay,s.ext_mu,s.ext_var,s.threshold,s.tw,s.x,s.qw,s.p['V_reset'],*state)

def recurrence(r,dt):
    n=len(r);tail=r[max(0,n-round(4000/dt)):];amp=np.maximum(np.ptp(tail,axis=0),.001)
    # At large J the cores oscillate on a high tonic background, so peak
    # prominence must use the oscillation amplitude, not the absolute rate.
    # Prefer surround E when it is recruited: core A may have two peaks in
    # one full-network cycle, while surround E has one strong population pulse.
    span=np.ptp(tail,axis=0);reference=2 if span[2]>.1*span[1] else 1
    if span.max()<1e-8:return dict(period_ms=None,recurrence_error=None,classification='near_equilibrium')
    pk,_=find_peaks(r[:,reference],height=tail[:,reference].min()+span[reference]*.25,prominence=max(1e-7,span[reference]*.12),distance=max(1,round(10/dt)))
    pk=pk[pk>n-round(4000/dt)]
    if len(pk)<4:return dict(period_ms=None,recurrence_error=None,B_peak_intervals_ms=(np.diff(pk)*dt).tolist())
    grid=np.arange(n)*dt; tt=grid[-min(round(2000/dt),n//3)::max(1,round(.5/dt))]
    ref=np.column_stack([np.interp(tt,grid,r[:,i]) for i in range(6)])
    def loss(T):
        shifted=np.column_stack([np.interp(tt-T,grid,r[:,i]) for i in range(6)])
        return np.mean(((ref-shifted)/amp)**2)
    trials=[]
    for mult in range(1,min(5,len(pk)-1)+1):
        T=float(np.median((pk[mult:]-pk[:-mult])*dt))
        fit=minimize_scalar(loss,bounds=(T*.94,T*1.06),method='bounded',options={'xatol':1e-7})
        trials.append(dict(period_ms=float(fit.x),recurrence_error=float(np.sqrt(fit.fun)),multiple=mult))
        if fit.fun<1e-8:break
    best=min(trials,key=lambda x:x['recurrence_error'])
    return dict(**best,trials=trials,reference_group=int(reference),reference_peak_intervals_ms=(np.diff(pk)*dt).tolist(),reference_peaks_ms=(pk*dt).tolist())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--g',type=float,nargs='+',default=[1.175,1.18,1.2,1.25,1.3,1.4,1.5,1.6]);ap.add_argument('--dt',type=float,default=.1);ap.add_argument('--duration',type=float,default=8000);args=ap.parse_args()
    s=System();state=orbit_history(s,V3/'periodic/g1.17500000_N2048.npz',args.dt)
    dest=OUT/'trajectories';dest.mkdir(exist_ok=True);summary=[]
    for g in args.g:
        path=dest/f'g{g:.8f}_dt{args.dt:g}.npz'
        if path.exists():
            z=np.load(path);r=z['r'];state=(z['final_r'],z['final_h'],z['final_history'])
        else:
            r,rf,hf,hist=simulate(s,g,state,args.dt,args.duration);state=(rf,hf,hist)
            if not np.isfinite(r).all():raise RuntimeError(f'Nonfinite simulation at {g}')
            np.savez_compressed(path,r=r,dt=args.dt,g=g,final_r=rf,final_h=hf,final_history=hist)
        rec=recurrence(r,args.dt);tail=r[-round(4000/args.dt):]
        row=dict(g=g,dt_ms=args.dt,duration_ms=args.duration,mean_hz=(tail.mean(0)*1000).tolist(),min_hz=(tail.min(0)*1000).tolist(),max_hz=(tail.max(0)*1000).tolist(),recurrence=rec,source=str(path))
        summary.append(row);write(f'exploration_dt{args.dt:g}.json',summary);print('EXPLORATION',json.dumps(row),flush=True)

if __name__=='__main__':main()
