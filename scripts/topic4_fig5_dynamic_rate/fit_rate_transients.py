"""Bounded local identification, with a held-out input waveform.

Compare a linear complex-rate closure with a positive nonlinear rate/phase
closure sharing the same measured stationary transfer and local poles. The
nonlinear form is inspired by QIF rate equations, not exact for this LIF model.
"""
from common import *
from rate_unit import RateTransfer
from numba import njit
from scipy.optimize import least_squares
import time
import argparse


@njit(cache=True)
def predict(f,cv,pop,parameters,nonlinear):
    T,C=f.shape;out=np.zeros((T,C));raw_min=0.
    for c in range(C):
        p=pop[c];a_scale=parameters[p*2];low=parameters[p*2+1]
        tau=20. if p==0 else 10.
        rate=f[0,c];alpha=a_scale*f[0,c]*(cv[0,c]/.22)**2+low*np.exp(-f[0,c]*tau)/tau
        aux=-alpha/2 if nonlinear else 0.
        for k in range(T):
            F=f[k,c];alpha=a_scale*F*(cv[k,c]/.22)**2+low*np.exp(-F*tau)/tau
            if not nonlinear:
                a=np.exp(-.1*alpha);co=np.cos(.1*2*np.pi*F);si=np.sin(.1*2*np.pi*F)
                x=rate-F;rate=F+a*(co*x+si*aux);aux=a*(co*aux-si*x)
                value=.5*(rate+np.sqrt(rate*rate+1e-10))
            else:
                # Exact Riccati flow at constant input within a native step.
                # Its integral, rather than a point sample of a narrow peak,
                # is the spike-rate readout for temporal binning.
                w=complex(aux,np.pi*rate);eq=complex(-alpha/2,np.pi*F)
                ratio=(w-eq)/(w+eq);new_ratio=ratio*np.exp(2*eq*.1)
                new_w=eq*(1+new_ratio)/(1-new_ratio)
                integral=eq*.1-np.log((1-new_ratio)/(1-ratio))
                value=integral.imag/(np.pi*.1)
                rate=new_w.imag/np.pi;aux=new_w.real
            raw_min=min(raw_min,rate)
            if not np.isfinite(rate) or not np.isfinite(value) or value < -1e-7:
                out[k:,c]=1e4;break
            out[k,c]=value*1000
    return out,raw_min


def main(a):
    folder=OUT/'local_response'/a.label;folder.mkdir(parents=True,exist_ok=False)
    a=np.load(OUT/'local_response/transient_validation_v1/traces.npz')
    u=a['input_mv'];native=a['native_rate_1ms'];spec=a['specs'];pop=spec[:,0].astype(int)
    t=RateTransfer();fs=[];cvs=[]
    for c in range(len(spec)):
        f,cv=t.row(u[:,c],float(spec[c,1]),pop[c]);fs.append(f);cvs.append(cv)
    f=np.array(fs).T.copy();cv=np.array(cvs).T.copy()
    train=np.arange(3);test=np.arange(3,6);n10=native.reshape(-1,10,6).mean(1)
    scale=np.sqrt(np.mean(n10[:,:3]**2,axis=0))
    write(folder/'contract.json',dict(training='Three pulse responses, E threshold15/18 and I18',
        validation='Three multisine responses, not passed to the optimizer',
        parameters='Four shared damping factors: active/low-rate for E and I. No individual theta or network/D fit.',
        bounds=[.05,20.],max_evaluations_per_start=65,starts=2,
        candidates=['linear complex rate','positive nonlinear rate and phase'],
        scope='Local response structural check only. Both candidates are approximations, not accepted SNN reductions.'))
    rows=[];traces={};started=time.time()
    for nonlinear in (False,True):
        def residual(logp):
            y,_=predict(f[:,:3].copy(),cv[:,:3].copy(),pop[:3],np.exp(logp),nonlinear)
            y10=y.reshape(-1,100,3).mean(1)
            return ((y10-n10[:,:3])/scale).ravel()
        best=None
        for initial in ([1.,1.,1.,1.],[1.,5.,1.,5.]):
            res=least_squares(residual,np.log(initial),bounds=(np.log(.05),np.log(20.)),max_nfev=65)
            if best is None or res.cost<best.cost:best=res
        params=np.exp(best.x);y,minimum=predict(f,cv,pop,params,nonlinear)
        y10=y.reshape(-1,100,6).mean(1);stats=[]
        for c in range(6):
            stats.append(dict(population=int(pop[c]),theta=float(spec[c,1]),input=spec[c,2],
                role='training' if c<3 else 'held_out_waveform',
                native_mean_hz=float(native[:,c].mean()),model_mean_hz=float(y[:,c].mean()),
                normalized_rmse_10ms=float(np.linalg.norm(y10[:,c]-n10[:,c])/np.linalg.norm(n10[:,c])),
                rmse_10ms_hz=float(np.sqrt(np.mean((y10[:,c]-n10[:,c])**2)))))
        key='nonlinear' if nonlinear else 'linear';traces[key]=y.reshape(-1,10,6).mean(1)
        rows.append(dict(model=key,parameters=params,optimizer_success=bool(best.success),cost=best.cost,
            optimizer_evaluations=best.nfev,minimum_raw_rate_per_ms=float(minimum),responses=stats))
        print(rows[-1],flush=True)
    np.savez_compressed(folder/'traces.npz',native_rate_1ms=native,input_mv=u,**traces)
    write(folder/'result.json',dict(status='LOCAL_IDENTIFICATION_COMPLETE',rows=rows,wall_s=time.time()-started,
        network_equivalence='NOT_TESTED',bifurcation='NOT_STARTED'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--label',default='dynamic_rate_identification_exactflow_v2');main(ap.parse_args())
