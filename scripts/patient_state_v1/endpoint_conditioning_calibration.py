"""Can informative interval endpoints mimic changing fitted OU timescales?

Exact Gaussian OU bridges use actual interictal event times and clinical ends.
The reference draws Gaussian stationary endpoints (hence the original OU law).
Two diagnostics instead condition endpoints to independently chosen +/-2 or3SD.
These are endpoint-selection diagnostics, not first-passage DDM simulations:
paths may have earlier excursions, and no seizure type or hazard is fitted.
"""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from numba import njit
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import fit,laplace,slice_data
from scripts.patient_state_v1.advanced_controls import history_data
from scripts.patient_state_v1.review_predictions import selected
from scripts.patient_state_v1.analyze_first import best_fits

OUT=RUN/'endpoint_conditioning_calibration_v1_34'

@njit(cache=True)
def simulate_bridge(t,dt,reset,terminal,b,gamma,tau,sd,strength,seed):
    np.random.seed(seed);x=np.empty(len(t));y=np.empty(len(t),np.int64);endpoint=0.;last=0.
    for i in range(len(t)):
        remaining=terminal[i]-t[i]
        if reset[i]:
            endpoint=sd*np.random.normal() if strength==0 else sd*strength*(1. if np.random.random()<.5 else -1.)
            if np.isfinite(remaining):
                c=np.exp(-remaining/tau);mean=c*endpoint;variance=sd*sd*(-np.expm1(-2*remaining/tau))
            else:mean=0.;variance=sd*sd
        elif np.isfinite(remaining):
            a=np.exp(-dt[i]/tau);c=np.exp(-remaining/tau);q=sd*sd*(-np.expm1(-2*dt[i]/tau));vrem=sd*sd*(-np.expm1(-2*remaining/tau));den=q*c*c+vrem
            mean=a*last+q*c/den*(endpoint-c*a*last);variance=q*vrem/den
        else:
            a=np.exp(-dt[i]/tau);mean=a*last;variance=sd*sd*(-np.expm1(-2*dt[i]/tau))
        x[i]=mean+np.sqrt(max(variance,0.))*np.random.normal();h=0. if reset[i] else (y[i-1]-.5)*np.exp(-dt[i]/(1/3600));p=1/(1+np.exp(-(b+x[i]+gamma*h)));y[i]=int(np.random.random()<p);last=x[i]
    return y,x

def inputs():
    d=dict(np.load(RUN/'observations.npz'));origin=float(d['origin_epoch']);sz=json.loads((RUN/'seizures.json').read_text());ends=np.array([(s['onset']-origin)/3600 for s in sz]+[np.inf]);terminal=ends[d['epoch']];assert np.all(terminal-d['t']>=.25/3600-1e-7)
    source=selected(RUN/'advanced_controls_v1_2/fits','full','ou_history')[0];return d,terminal,source

def canary():
    t=np.tile(np.array([0.,.05,.2]),30000)+np.repeat(np.arange(30000)*1.,3);reset=np.tile(np.array([True,False,False]),30000);dt=np.tile(np.array([0.,.05,.15]),30000);terminal=np.repeat(np.arange(30000)+.3,3);_,x=simulate_bridge(t,dt,reset,terminal,-.8,0.,.5,.6,0.,1034001);samples=x.reshape(-1,3);reference=.6**2*np.exp(-abs(np.array([0.,.05,.2])[:,None]-np.array([0.,.05,.2])[None,:])/.5);err=float(np.max(abs(np.cov(samples,rowvar=False)-reference)));mean=float(np.max(abs(samples.mean(0))));assert err<.012 and mean<.012
    write_json(OUT/'numerical_canary.json',dict(status='PASS',n_independent_gaussian_endpoint_bridges=30000,max_covariance_error=err,max_mean_error=mean,reference='Marginalizing stationary Gaussian endpoints restores the stationary irregular-time OU Gaussian law',limits='This verifies the endpoint-conditioned generator, not first-passage survival or a seizure mechanism'))

def worker(job):
    path=OUT/'runs'/f"{job['condition']}_{job['rep']:03d}.json"
    if path.exists():return json.loads(path.read_text())
    started=time.time()
    try:
        d,end,source=inputs();b,g,lt,ls=source['theta'];y,x=simulate_bridge(d['t'],d['dt'],d['reset'],end,b,g,np.exp(lt),np.exp(ls),job['strength'],job['seed']);d['y']=y;hd=history_data(d);folds=json.loads((RUN/'splits.json').read_text());scopes=[('full',len(y))]+[(f"fold{f['fold']}",f['train_end']) for f in folds];fits=[]
        for history in [False,True]:
            for scope,n in scopes:
                train=slice_data(hd if history else d,0,n);p=np.clip(train['y'].mean(),1e-5,1-1e-5);bias=np.log(p/(1-p));candidates=[]
                for tau0 in [.1,1.,6.]:
                    if not history:r=fit(train,'ou',initial=[bias,np.log(tau0),np.log(.6)],maxiter=150)
                    else:
                        initial=np.array([bias,.8,np.log(tau0),np.log(.6)])
                        def fun(t):return -laplace(np.r_[t[:2],0.,t[2:]],train,True)
                        opt=minimize(fun,initial,method='L-BFGS-B',bounds=[(-8,8),(-8,8),(np.log(1/60),np.log(24)),(np.log(.01),np.log(5))],options={'maxiter':180,'ftol':1e-10,'eps':2e-5,'maxls':25});r=dict(theta=opt.x,loglik=-opt.fun,success=bool(opt.success),message=str(opt.message),nfev=int(opt.nfev))
                    candidates.append(r)
                best=max(candidates,key=lambda r:r['loglik']);fits.append(dict(scope=scope,history=history,best=best,candidates=candidates))
        result=dict(status='COMPLETE',job=job,fits=fits,n_events=len(y),tb_fraction=float(y.mean()),latent_quantiles=np.quantile(x,[.01,.1,.5,.9,.99]),elapsed=time.time()-started)
        if job['rep']==0:path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),labels=y,latent_state=x,terminal_hours=end)
    except Exception:result=dict(status='FAILED',job=job,traceback=traceback.format_exc(),elapsed=time.time()-started)
    write_json(path,result);return result

def review():
    rows=[];prefix=[]
    for p in (OUT/'runs').glob('*.json'):
        r=json.loads(p.read_text());assert r['status']=='COMPLETE';j=r['job']
        for history in [False,True]:
            fits={f['scope']:f['best'] for f in r['fits'] if f['history']==history};delta=fits['fold0']['theta'][-2]-fits['full']['theta'][-2]
            rows.append(dict(condition=j['condition'],rep=j['rep'],history=history,delta_log_tau_prefix1_minus_full=delta,all_selected_success=all(f['success'] for f in fits.values()),tb_fraction=r['tb_fraction']))
            for scope,f in fits.items():prefix.append(dict(condition=j['condition'],rep=j['rep'],history=history,scope=scope,tau_minutes=np.exp(f['theta'][-2])*60,sd=np.exp(f['theta'][-1]),success=f['success']))
    df=pd.DataFrame(rows);assert len(df)==384;df.to_csv(OUT/'prefix_statistics.csv',index=False);pd.DataFrame(prefix).to_csv(OUT/'fitted_parameters.csv',index=False);best=best_fits();patient={False:best['fold0','ou']['theta'][-2]-best['full','ou']['theta'][-2],True:selected(RUN/'advanced_controls_v1_2/fits','fold0','ou_history')[0]['theta'][-2]-selected(RUN/'advanced_controls_v1_2/fits','full','ou_history')[0]['theta'][-2]};reports=[]
    for (condition,history),part in df.groupby(['condition','history']):
        values=part.delta_log_tau_prefix1_minus_full.to_numpy();n_above=int(np.sum(values>=patient[history]));q=np.quantile(values,[.025,.5,.975]);successful=part[part.all_selected_success];reports.append(dict(condition=condition,history=bool(history),n=len(part),patient=float(patient[history]),lower=float(q[0]),median=float(q[1]),upper=float(q[2]),n_at_least_patient=n_above,add_one_upper_tail=(1+n_above)/(1+len(part)),n_all_selected_success=len(successful),success_only_n_at_least_patient=int(np.sum(successful.delta_log_tau_prefix1_minus_full>=patient[history]))))
    pd.DataFrame(reports).to_csv(OUT/'calibration_comparison.csv',index=False);write_json(OUT/'scientific_audit.json',dict(status='COMPLETE',results=reports,claim='Synthetic endpoint-selection diagnostic at frozen parameters and observed timestamps/clinical segmentation',not_claimed='Pinned endpoints are not absorbing boundaries: bridges allow earlier excursions. No actual seizure type, hazard, reset rule, or physiological drift has been inferred.',interpretation='If endpoint conditioning reproduces the prefix statistic it supplies a possible observational explanation, not its identification in this patient. Failure only constrains the specified bridge selections.'))
    print(pd.DataFrame(reports).to_string(index=False))

def main():
    canary();_,_,source=inputs();jobs=[]
    for condition,strength in [('gaussian_endpoint',0.),('selected_endpoint_2sd',2.),('selected_endpoint_3sd',3.)]:
        for rep in range(64):jobs.append(dict(condition=condition,strength=strength,rep=rep,seed=1034100+rep))
    write_json(OUT/'contract.json',dict(question='Can informative clinical interval endpoints, rather than physically varying OU parameters, produce the observed prefix-time-constant difference?',source=source,observed_inputs='Actual event times, coverage and clinical interval ends fixed; all interictal labels are newly generated; actual seizure types are not used',generators='Same OU-history parameters. Gaussian stationary endpoints reproduce the ordinary stationary per-epoch OU law; selected endpoints are independently +/-2 or3stationary SD. Final uncensored interval is ordinary OU.',exact_bridge='OU Gaussian transitions conditioned on one terminal value; no first-passage or survival constraint',n_sequences=192,n_selected_fits=1536,n_initial_fit_attempts=4608,fit='Same ordinary OU/OU-history families used on patient;4prefix/full scopes and3initial time constants, selection by training likelihood only',readout='Previously defined first-prefix minus full log time constant;64replicates per generator, all retained with optimizer sensitivity reported',scope='Synthetic specificity/calibration extension only. This does not change patient fitting, SNN, Z/M, or accepted biological interpretation.'))
    write_json(OUT/'queue.json',jobs)
    with ProcessPoolExecutor(max_workers=32) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(done=i+1,total=len(jobs),condition=r['job']['condition'],rep=r['job']['rep'],status=r['status'],elapsed=r['elapsed'])),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_sequences=192,finished_unix=time.time()));review()

if __name__=='__main__':main()
