"""Learn the nuisance short-mark-memory time constant, leaving the OU SDE fixed.

The earlier one-second decay was fixed. This bounded diagnostic asks whether
that restriction falsely assigns short dependence to the slower continuous state.
History is an observation term, not a force or a physical reset of the state.
"""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import laplace,filter_adf,slice_data
from scripts.patient_state_v1.advanced_controls import history_data
from scripts.patient_state_v1.review_predictions import selected
from scripts.patient_state_v1.review_activity_clock import paired

OUT=RUN/'learned_history_decay_v1_32'

def data_at(d,seconds):
    q={k:np.array(v,copy=True) for k,v in d.items() if np.ndim(v)>0}
    h=np.r_[0.,q['y'][:-1]/q['n'][:-1]-.5]*np.exp(-q['dt']/(seconds/3600));h[q['reset']]=0
    q['x']=np.column_stack([np.ones(len(h)),h,np.zeros(len(h))]);return q

def expand(t):return np.r_[t[:2],0.,t[2:4]]

def worker(j):
    path=OUT/'fits'/f"{j['id']}.json"
    if path.exists():return json.loads(path.read_text())
    started=time.time()
    try:
        d=dict(np.load(RUN/'observations.npz'));train=slice_data(d,0,j['end']);base=selected(RUN/'advanced_controls_v1_2/fits',j['scope'],'ou_history')[0]['theta'];initial=np.r_[base,np.log(j['history0'])]
        bounds=[(-8,8),(-8,8),(np.log(1/60),np.log(24)),(np.log(.01),np.log(5)),(np.log(.05),np.log(300))]
        def fun(t):return -laplace(expand(t),data_at(train,np.exp(t[-1])),True)
        opt=minimize(fun,initial,method='L-BFGS-B',bounds=bounds,options={'maxiter':180,'ftol':1e-10,'eps':2e-5,'maxls':25})
        f=filter_adf(expand(opt.x),data_at(d,np.exp(opt.x[-1])),True,order=64)
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),**f)
        r=dict(status='COMPLETE',job=j,theta=opt.x,loglik=-opt.fun,success=bool(opt.success),message=str(opt.message),nfev=int(opt.nfev),elapsed=time.time()-started)
    except Exception:r=dict(status='FAILED',job=j,traceback=traceback.format_exc(),elapsed=time.time()-started)
    write_json(path,r);return r

def review():
    best={}
    for p in (OUT/'fits').glob('*.json'):
        r=json.loads(p.read_text());assert r['status']=='COMPLETE';key=r['job']['scope']
        if key not in best or r['loglik']>best[key][1]['loglik']:best[key]=(p,r)
    assert len(best)==4;pars=[];records=[];d=np.load(RUN/'observations.npz')
    for scope,(p,r) in sorted(best.items()):
        t=r['theta'];old=selected(RUN/'advanced_controls_v1_2/fits',scope,'ou_history')[0]
        pars.append(dict(scope=scope,b=t[0],gamma=t[1],tau_minutes=np.exp(t[2])*60,sd=np.exp(t[3]),history_decay_seconds=np.exp(t[4]),loglik=r['loglik'],gain_fixed_history=r['loglik']-old['loglik'],success=r['success'],source=str(p)))
        if scope=='full':continue
        lo,hi=r['job']['end'],r['job']['test_end'];z=np.load(p.with_suffix('.npz'));pp=np.clip(z['predict_tb'][lo:hi],1e-12,1-1e-12);y=d['y'][lo:hi]
        records.append(pd.DataFrame(dict(model='learned_history',fold=int(scope[4:]),index=np.arange(lo,hi),hour=d['t'][lo:hi],y=y,p_tb=pp,score=y*np.log(pp)+(1-y)*np.log1p(-pp))))
    params=pd.DataFrame(pars);params.to_csv(OUT/'selected_parameters.csv',index=False);pred=pd.concat(records,ignore_index=True);pred.to_csv(OUT/'forward_predictions.csv.gz',index=False);old=pd.read_csv(RUN/'all_forward_predictions.csv.gz');rows=[]
    for baseline in ['constant','ewma','ou','ou_history']:
        for width in [1,6]:rows.append(dict(model='learned_history',baseline=baseline,**paired(pred,old[old.model==baseline],width)))
    scores=pd.DataFrame(rows);scores.to_csv(OUT/'forward_summary.csv',index=False)
    write_json(OUT/'scientific_audit.json',dict(status='COMPLETE',selected_fits=4,all_selected_optimizer_success=bool(params.success.all()),fixed_history_forward_increment_6h=scores[(scores.baseline=='ou_history')&(scores.block_hours==6)].to_dict('records')[0],limits='Development forward test with frozen full-record labels; observation-history dependence is not a neural force. Parameter uncertainty not yet calibrated.'))
    print(params.to_string(index=False));print(scores[scores.block_hours==6].to_string(index=False))

def main():
    d=dict(np.load(RUN/'observations.npz'));q=data_at(d,1.);old=history_data(d);assert np.array_equal(q['x'],old['x']);cut=1000;assert np.array_equal(data_at(slice_data(d,0,cut),2.)['x'],data_at(d,2.)['x'][:cut]);assert np.min(d['dt'][~d['reset']])*3600>=.25-1e-6
    write_json(OUT/'numerical_canary.json',dict(status='PASS',one_second_exact_original_history=True,causal_prefix_identity=True,all_previous_events_completed=True))
    folds=json.loads((RUN/'splits.json').read_text());scopes=[dict(scope='full',end=len(d['y']))]+[dict(scope=f"fold{f['fold']}",end=f['train_end'],test_end=f['test_end']) for f in folds];jobs=[]
    for scope in scopes:
        for i,h in enumerate([.1,1.,10.,100.]):jobs.append(dict(id=f"{scope['scope']}_{i}",history0=h,**scope))
    write_json(OUT/'contract.json',dict(question='Is fixing the short-mark-memory decay at1second responsible for inferred slow-state persistence?',state='Same single stationary OU SDE; no IED reset',observation='logit P(TB_i)=b+x_i+gamma*(previous_label-.5)*exp(-delta_seconds/tau_history); history zero at clinical exclusion boundaries',parameters='Original4 parameters plus one log history decay in[log0.05,log300]seconds; same existing bounds for all inherited parameters',selection='Training-only marginal Laplace likelihood;4 starts per full/prefix fit; all16157forward events retained',n_fits=16,numerics='Exact irregular OU transitions;64-node Gaussian moment filter for forward predictions; extra-parameter numerical/posterior validation required before mechanism claims',scope='Nuisance observation-dependence diagnostic, not neural feedback, a new SNN, or physical evidence accumulation'))
    with ProcessPoolExecutor(max_workers=16) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(done=i+1,total=16,id=r['job']['id'],status=r['status'],success=r.get('success'),elapsed=r['elapsed'])),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_fits=16,finished_unix=time.time()));review()

if __name__=='__main__':main()
