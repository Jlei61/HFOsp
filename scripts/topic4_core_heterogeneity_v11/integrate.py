"""RK4 method of steps; retain the complete filtered-rate delay history."""
from common import *
from dynamics import transfer
from numba import njit
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
import time


@njit(cache=True)
def rhs(y, history, head, stage, dt, delays, weights, di, dj, Q,
        tr, rise, decay, tm, ref, extmu, extvar, threshold, tw, xx, qw, reset):
    mu=extmu.copy(); ve=extvar.copy(); vi=np.zeros(6)
    size=len(history)
    for k in range(len(delays)):
        x=delays[k]/dt-stage; lag=int(np.floor(x)); frac=x-lag
        old=(1-frac)*history[(head-lag)%size,dj[k]]+frac*history[(head-lag-1)%size,dj[k]]
        mu[di[k]]+=weights[k]*old
    for i in range(6):
        for j in range(6):
            if j<3:ve[i]+=tm[i]*Q[i,j]*y[j]
            else:vi[i]+=tm[i]*Q[i,j]*y[j]
    ph=transfer(mu,ve,vi,tm,ref,threshold,tw,xx,qw,reset)
    ans=np.empty(18)
    ans[:6]=(ph-y[:6])/tr
    ans[6:12]=(y[:6]-y[6:12])/rise
    ans[12:]=(y[6:12]-y[12:])/decay
    return ans


@njit(cache=True)
def steps(n, dt, stride, y, history, head, delays, weights, di, dj, Q,
          tr, rise, decay, tm, ref, extmu, extvar, threshold, tw, xx, qw, reset):
    out=np.empty((n//stride,6)); idx=0
    for k in range(n):
        a=rhs(y,history,head,0.,dt,delays,weights,di,dj,Q,tr,rise,decay,tm,ref,extmu,extvar,threshold,tw,xx,qw,reset)
        b=rhs(y+dt*a/2,history,head,.5,dt,delays,weights,di,dj,Q,tr,rise,decay,tm,ref,extmu,extvar,threshold,tw,xx,qw,reset)
        c=rhs(y+dt*b/2,history,head,.5,dt,delays,weights,di,dj,Q,tr,rise,decay,tm,ref,extmu,extvar,threshold,tw,xx,qw,reset)
        d=rhs(y+dt*c,history,head,1.,dt,delays,weights,di,dj,Q,tr,rise,decay,tm,ref,extmu,extvar,threshold,tw,xx,qw,reset)
        y=y+dt*(a+2*b+2*c+d)/6
        head=(head+1)%len(history);history[head]=y[12:]
        if (k+1)%stride==0:out[idx]=y[:6];idx+=1
    return out,y,history,head


def constant_state(s, r, dt):
    history=np.tile(r,(int(np.ceil(s.delay[-1]/dt))+3,1))
    return np.tile(r,3),history,0


def orbit_state(s, r, T, dt):
    N=len(r);freq=2j*np.pi*np.arange(N//2+1)/T; R=np.fft.rfft(r,axis=0)
    H=R/(1+freq[:,None]*s.rise)
    C=H/(1+freq[:,None]*s.decay)
    h=np.fft.irfft(H,n=N,axis=0)[0]; c=np.fft.irfft(C,n=N,axis=0)[0]
    size=int(np.ceil(s.delay[-1]/dt))+3
    weights=np.ones(len(freq));weights[1:-1]=2
    values=(np.exp(-np.arange(size)[:,None]*dt*freq[None,:])@(C*weights[:,None])).real/N
    history=np.empty_like(values)
    for k in range(size): history[(-k)%size]=values[k]
    return np.r_[r[0],h,c],history,0


def simulate(s,g,duration_ms=6000,dt=.05,state=None,r0=None,save_dt=.5):
    if dt>s.delay[0]+1e-12:raise ValueError('RK stages require past delayed states')
    if state is None:
        if r0 is None:r0=np.array([.0002,.00018,1e-10,1e-8,1e-8,1e-10])
        state=constant_state(s,r0,dt)
    W,Q=s.weights(g);dd,di,dj=np.nonzero(W)
    delays=s.delay[dd];weights=W[dd,di,dj]*s.tm[di]*s.area[dj]*s.sign[dj]
    y,h,head=state
    stride=round(save_dt/dt);n=round(duration_ms/dt);n-=n%stride
    out,y,h,head=steps(n,dt,stride,y.copy(),h.copy(),head,delays,weights,di,dj,Q,
        s.tr,s.rise,s.decay,s.tm,s.ref,s.ext_mu,s.ext_var,s.threshold,s.tw,s.x,s.qw,s.p['V_reset'])
    return out,(y,h,head)


def describe(rate, sample_dt=.5):
    r=rate[len(rate)//2:]*1000
    if not np.isfinite(r).all():return dict(kind='unresolved',reason='nonfinite')
    lo=r.min(0);hi=r.max(0);mean=r.mean(0)
    row=dict(mean_hz=mean.tolist(),min_hz=lo.tolist(),max_hz=hi.tolist(),observed_ms=len(r)*sample_dt)
    if max(hi-lo)<.005:
        row.update(kind='equilibrium_candidate',period_ms=None,recurrence_error_hz=None)
        return row
    pop=int(np.argmax(hi-lo));pk,_=find_peaks(r[:,pop],prominence=max(.01,.05*(hi[pop]-lo[pop])),distance=max(2,round(5/sample_dt)))
    recurrence=[]
    if len(pk)>=3:
        for order in range(1,min(9,len(pk)-1)):
            lag=int(round(np.median(pk[order:]-pk[:-order])))
            if lag<1 or 3*lag>len(r):continue
            size=min(2*lag,len(r)//3)
            t=np.arange(len(r));spl=CubicSpline(t,r,axis=0)
            probe=np.arange(len(r)-size,len(r),2.)
            def error(delta):
                return float(np.max(np.sqrt(np.mean((spl(probe)-spl(probe-delta))**2,axis=0))))
            opt=minimize_scalar(error,bounds=(max(1,lag-2),lag+2),method='bounded',options={'xatol':1e-5})
            recurrence.append((opt.fun,opt.x))
    candidates=[q for q in recurrence if q[0]<.1]
    if candidates:
        err,lag=min(candidates,key=lambda q:q[1]);kind='periodic_candidate'
    elif recurrence:err,lag=min(recurrence);kind='unresolved'
    else:err,lag=None,None;kind='unresolved'
    pattern=('both_burst' if max(lo[:2])<5 else 'A_burst_B_high' if lo[0]<5 else 'B_burst_A_high' if lo[1]<5 else 'both_high')
    row.update(kind=kind,pattern=pattern,period_ms=lag*sample_dt if lag else None,
               recurrence_error_hz=err,peak_count=len(pk),reference_population=pop)
    return row


if __name__=='__main__':
    s=System(1);start=time.monotonic();r,state=simulate(s,1.38,duration_ms=500)
    print('compile_and_500ms_s',time.monotonic()-start,flush=True)
    start=time.monotonic();r,state=simulate(s,1.38,duration_ms=6000,state=state)
    print('6000ms_s',time.monotonic()-start,describe(r),flush=True)
