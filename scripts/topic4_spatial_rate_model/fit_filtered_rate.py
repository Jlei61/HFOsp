"""Test one additional finite input-memory state using local responses only.

The earlier multisine diagnostic is now explicitly development data. New local
waveforms and the second network input remain separate validation tasks.
"""
from common import *
from rate_unit import RateTransfer
from fit_rate_transients import predict
from numba import njit
from scipy.optimize import least_squares
import time


@njit(cache=True)
def filter_inputs(u,pop,taus,dt=.1):
    out=np.empty_like(u)
    for c in range(u.shape[1]):
        value=u[0,c];decay=np.exp(-dt/taus[pop[c]])
        for k in range(len(u)):
            value=decay*value+(1-decay)*u[k,c];out[k,c]=value
    return out


def main():
    folder=OUT/'local_response/filtered_rate_identification_v1';folder.mkdir(parents=True,exist_ok=False)
    a=np.load(OUT/'local_response/transient_validation_v1/traces.npz');u=a['input_mv'];native=a['native_rate_1ms'];spec=a['specs'];pop=spec[:,0].astype(int)
    tr=RateTransfer();n10=native.reshape(-1,10,6).mean(1);scale=np.sqrt(np.mean(n10**2,axis=0))
    old=read(OUT/'local_response/dynamic_rate_identification_exactflow_v2/result.json');rows=[];outputs={};start=time.time()
    write(folder/'contract.json',dict(training='Six earlier prescribed-current pulse/multisine responses; former held-out waveform is now development data',
        validation='New chirp/ramp waveform and second spatial-network input; not used here',
        extra_state='One effective-input memory variable per rate group, du_f/dt=(u-u_f)/tau_f',
        free_parameters='E/I active damping, low-rate damping and input memory time; six globally shared parameters',
        bounds={'damping':[.05,20.],'memory_ms':[.02,20.]},starts=2,max_evaluations=90,
        native_biology='Unchanged; only the local dynamic transfer is identified',
        selection='Retain additional memory only if it improves independent local response and autonomous network behavior'))
    def drive(logp):
        pars=np.exp(logp);filtered=filter_inputs(u,pop,pars[4:]);fs=[];cvs=[]
        for c in range(6):
            f,cv=tr.row(filtered[:,c],float(spec[c,1]),pop[c]);fs.append(f);cvs.append(cv)
        return pars,np.array(fs).T.copy(),np.array(cvs).T.copy()
    for kind in ('linear','nonlinear'):
        oldp=np.array(next(x for x in old['rows'] if x['model']==kind)['parameters'])
        nonlinear=kind=='nonlinear'
        def residual(logp):
            p,f,cv=drive(logp);y,_=predict(f,cv,pop,p[:4],nonlinear)
            return ((y.reshape(-1,100,6).mean(1)-n10)/scale).ravel()
        best=None
        for tau in (1.,5.):
            initial=np.r_[np.clip(oldp,.050001,19.99999),[tau,tau]]
            res=least_squares(residual,np.log(initial),bounds=(np.log([.05]*4+[.02]*2),np.log([20.]*6)),max_nfev=90,diff_step=1e-4)
            if best is None or res.cost<best.cost:best=res
        p,f,cv=drive(best.x);y,minrate=predict(f,cv,pop,p[:4],nonlinear);y10=y.reshape(-1,100,6).mean(1)
        row=dict(model=kind,parameters=p[:4],input_memory_ms=p[4:],cost=best.cost,
            optimizer_success=bool(best.success),evaluations=best.nfev,minimum_raw_rate_per_ms=float(minrate),
            normalized_rmse_10ms=[float(np.linalg.norm(y10[:,i]-n10[:,i])/np.linalg.norm(n10[:,i])) for i in range(6)])
        rows.append(row);outputs[kind]=y.reshape(-1,10,6).mean(1);print(row,flush=True)
    np.savez_compressed(folder/'traces.npz',native_rate_1ms=native,input_mv=u,**outputs)
    write(folder/'result.json',dict(status='LOCAL_FILTERED_RATE_FIT_COMPLETE',rows=rows,wall_s=time.time()-start,
        independent_validation='PENDING',network_acceptance='NOT_ESTABLISHED'))


if __name__=='__main__':main()
