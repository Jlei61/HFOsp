"""Isolated mixed E/I response diagnostic, not a native-network replay or fit.

Independent compound-Poisson inputs match the frozen graph's first two moments.
This tests the local transfer approximation on its own independence assumption.
"""
from common import *
from rate import phi,table_build
from prepare import quadrature
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter
from numba import njit
import time

@njit(cache=True)
def microscopic(threshold,types,tm,ref,jext,nu,rates,jumps,dt):
    np.random.seed(2026091607)
    N=len(threshold);steps=len(rates);voltage=np.full(N,11.);refractory=np.zeros(N,np.int64)
    hE=np.zeros(N);hI=np.zeros(N);cE=np.zeros(N);cI=np.zeros(N)
    out=np.zeros((steps,2));current=np.zeros((steps,2,2))
    arE=np.exp(-dt/.7);arI=np.exp(-dt);adE=np.exp(-dt/3.5);adI=np.exp(-dt/18.)
    for k in range(steps):
        for i in range(N):
            a=types[i]
            e=np.random.poisson(rates[k,a,0]*dt)*jumps[k,a,0]
            inh=np.random.poisson(rates[k,a,1]*dt)*jumps[k,a,1]
            ex=(np.random.poisson(nu[k,a]*dt) if a==0 else nu[k,a]*dt)*jext[a]
            hE[i]=hE[i]*arE+tm[a]/.7*(e+ex);hI[i]=hI[i]*arI+tm[a]*inh
            cE[i]=hE[i]+(cE[i]-hE[i])*adE;cI[i]=hI[i]+(cI[i]-hI[i])*adI
            refractory[i]=max(0,refractory[i]-1)
            if refractory[i]==0:
                current_i=cE[i]-cI[i];voltage[i]=current_i+(voltage[i]-current_i)*np.exp(-dt/tm[a])
                if voltage[i]>=threshold[i]:
                    out[k,a]+=1;voltage[i]=11.;refractory[i]=round(ref[a]/dt)
            else:voltage[i]=11.
            current[k,a,0]+=cE[i];current[k,a,1]+=cI[i]
    return out,current

def main():
    started=time.time();dt=.1;steps=30000
    old=np.load(ROOT/'results/topic4_sef_hfo/core_burst_bifurcation_v2_20260915/projected_graph.npz')
    native=np.load(V10/'native/a/trajectory.npz');sizes=np.bincount(native['region'],minlength=6)
    rbar=native['six_group_counts_2ms'][1000:].mean(0)/sizes/2.
    scale=np.ones((6,6));scale[0,0]=scale[1,1]=J
    W=old['W'].sum(0)*scale;Q=old['Q'].sum(0)*scale**2
    lam=np.zeros((2,2));jump=np.zeros((2,2))
    for a,pop in enumerate([0,3]):
        for b,sl in enumerate([slice(0,3),slice(3,6)]):
            m=W[pop,sl]@rbar[sl];v=Q[pop,sl]@rbar[sl]
            jump[a,b]=v/m;lam[a,b]=m*m/v
    # Causal zero-order hold: a native 2-ms spike-count bin becomes available
    # at its right edge; it drives the next 2-ms interval, then the native delay.
    source=np.zeros((steps,6));source[20:]=np.repeat(native['six_group_counts_2ms'][:1499]/sizes/2.,20,axis=0)
    mean=np.zeros((steps,2,2));variance=mean.copy()
    for a,pop in enumerate([0,3]):
        for b,indices in enumerate([range(3),range(3,6)]):
            for j in indices:
                mean[:,a,b]+=lfilter(np.r_[0.,old['W'][:,pop,j]*scale[pop,j]],[1.],source[:,j])
                variance[:,a,b]+=lfilter(np.r_[0.,old['Q'][:,pop,j]*scale[pop,j]**2],[1.],source[:,j])
    jump=np.divide(variance,mean,out=np.zeros_like(mean),where=mean>1e-20)
    rates=np.divide(mean**2,variance,out=np.zeros_like(mean),where=variance>1e-20)
    cfg=read(OUT/'model_config.json');nu=np.full((steps,2),cfg['signal_per_ms']);jext=np.array([.455,.85])
    p=cfg['params'];drive=runtime.CoreOUMixture(np.array([0,1]),cfg['signal_per_ms'],.95,0.,dt,p['tau_n'],p['sigma_n'],848101)
    for k in range(steps):nu[k,0]=max(0.,cfg['signal_per_ms']+drive.step(k*dt)[0])
    tm=np.array([20.,10.]);ref=np.array([2.,1.]);tau=np.array([[5.,2.5],[20.,10.]])
    th=[np.tile(old['vtheta'][old['region']==a],8) for a in [0,3]]
    types=np.r_[np.zeros(len(th[0]),np.int64),np.ones(len(th[1]),np.int64)];count=np.bincount(types)
    write(OUT/'mixed_response_protocol.json',dict(status='PRESPECIFIED_BEFORE_ISOLATED_RESPONSE',
        source_population_rates_per_ms=rbar,target_populations=['Core A E','Core A I'],sample_sizes=count,
        input='native seed848101 six-population rate history, 2-ms causal hold, each original delay; independent moment-matched afferents',
        source_window_ms=[0,3000],source_time_resolution_ms=2.,OU='same current core A OU and deterministic I external drive',
        hypothesis='Does the existing static mixed-receptor transfer and constant relaxation reproduce independent-afferent LIF population dynamics?',
        limits='Moment-matched isolated ensembles, not the original recurrent SNN; no intercell recurrent correlations or spatial claim; neither tau choice is fitted or selected.'))
    spikes,currents=microscopic(np.r_[tuple(th)],types,tm,ref,jext,nu,rates,jump,dt)
    actual=spikes/count/dt*1000;currents/=count[None,:,None]
    table,lo,step=table_build();nodes,tw=zip(*(quadrature(x,32) for x in th));nodes=np.array(nodes);tw=np.array(tw)
    predicted=np.zeros((steps,2,2));h=np.zeros((2,2));c=h.copy();r=np.zeros((2,2))
    for k in range(steps):
        drive=rates[k]*jump[k];drive[:,0]+=nu[k]*jext
        h=h*np.exp(-dt/np.array([.7,1.]))+dt*tm[:,None]/np.array([.7,1.])*drive
        c=h+(c-h)*np.exp(-dt/np.array([3.5,18.]))
        var=tm[:,None]*rates[k]*jump[k]**2;var[0,0]+=tm[0]*jext[0]**2*nu[k,0]
        ph=phi(c[:,0]-c[:,1],var[:,0],var[:,1],tm,ref,nodes,tw,table,lo,step,11.)
        r=np.exp(-dt/tau)*r+(1-np.exp(-dt/tau))*ph;predicted[k]=r*1000
    smooth=gaussian_filter1d(actual,50,axis=0);smooth_prediction=gaussian_filter1d(predicted,50,axis=0);rows=[]
    for a,name in enumerate(['Core A E','Core A I']):
        plateaus=[]
        for start,end in [(500,1500),(1500,3000)]:
            sl=slice(start*10,end*10);plateaus.append(dict(window_ms=[start,end],microscopic_mean_hz=float(actual[sl,a].mean()),closure_means_hz=predicted[sl,:,a].mean(0)))
        transients=[]
        for start,end in [(500,1500),(1500,3000)]:
            sl=slice(start*10,end*10);refcurve=smooth[sl,a]
            transients.append(dict(window_ms=[start,end],normalized_mse=np.mean((smooth_prediction[sl,:,a]-refcurve[:,None])**2,axis=0)/np.mean(refcurve**2) if np.mean(refcurve**2)>1. else None))
        rows.append(dict(population=name,window_means=plateaus,transients=transients))
    np.savez_compressed(OUT/'mixed_response.npz',microscopic_rate_hz=actual,closure_rate_hz=predicted,currents=currents,dt_ms=dt,tau_choices_ms=tau)
    write(OUT/'mixed_response.json',dict(status='DEVELOPMENT_DIAGNOSTIC_NOT_NETWORK_VALIDATION',rows=rows,seconds=time.time()-started,
        response_error_definition='Both microscopic and predicted curves receive the same 5 ms Gaussian smoothing; means use raw curves.'))
    print(json.dumps(safe(rows),indent=2),flush=True)

if __name__=='__main__':main()
