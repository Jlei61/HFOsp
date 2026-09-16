"""v1.16: causal total-activity covariate in the two-scale mark observation.

This checks an observation-model alternative to changing physiological drift.
Event timing remains conditioned on; no causal neural effect is identified.
"""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.activity_mark_controls import prepare
from scripts.patient_state_v1.model import slice_data
from scripts.patient_state_v1.two_timescale import filter_kernel,filter2
from scripts.patient_state_v1.analyze_first import metrics
OUT=RUN/'two_scale_activity_v1_16'

def dataset(end,minutes,carry):
    d,info=prepare(end,minutes);d['x']=d['x'][:,[0,2,1]].copy()
    if carry:d['slow_dt']=np.r_[0,np.diff(d['t'])];d['slow_reset']=np.r_[True,np.zeros(len(d['t'])-1,bool)]
    else:d['slow_dt']=d['dt'].copy();d['slow_reset']=d['reset'].copy()
    return d,info

def filtering(t,d):
    nc=d['x'].shape[1];tf,sdf,sds,ts=np.exp(t[nc:]);nodes,w=hermgauss(40);m,v,p,ll=filter_kernel(d['dt'],d['reset'],d['slow_dt'],d['slow_reset'],d['y'],d['n'],d['x']@t[:nc],tf,sdf,ts,sds,nodes,w/np.sqrt(np.pi));return dict(mean=m,variance=v,predict_tb=p,loglik_terms=ll,loglik=ll.sum())

def fit(d,initial):
    nc=d['x'].shape[1];bounds=[(-8,8)]*nc+[(np.log(1/3600),np.log(3)),(np.log(.01),np.log(3)),(np.log(.001),np.log(5)),(np.log(.25),np.log(96))]
    constraint=dict(type='ineq',fun=lambda t:t[-1]-t[nc]-np.log(4));opt=minimize(lambda t:-filtering(t,d)['loglik'],initial,method='SLSQP',bounds=bounds,constraints=[constraint],options={'maxiter':150,'ftol':1e-7,'eps':2e-5});return dict(theta=opt.x,loglik=-opt.fun,success=opt.success,message=opt.message,feasible=bool(constraint['fun'](opt.x)>=-1e-6),nfev=opt.nfev)

def worker(j):
    p=OUT/'fits'/f"{j['id']}.json"
    if p.exists():return json.loads(p.read_text())
    start=time.time()
    try:
        d,info=dataset(j['end'],j['minutes'],j['carry']);train=slice_data(d,0,j['end']);q=train['y'].mean();initial=np.r_[np.log(q/(1-q)),.7,0.,np.log(j['tf']),np.log(.4),np.log(.6),np.log(j['ts'])];r=fit(train,initial);f=filtering(r['theta'],d);r.update(status='COMPLETE',job=j,covariate_info=info,elapsed=time.time()-start)
        if j['scope']!='full':r['forward']=metrics(f['predict_tb'][j['end']:j['test_end']],d['y'][j['end']:j['test_end']],d['n'][j['end']:j['test_end']])
        p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p.with_suffix('.npz'),**f)
    except Exception:r=dict(status='FAILED',job=j,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(p,r);return r

def main():
    d,_=dataset(28125,1,True);tiny=slice_data(d,0,1000);t=np.array([-.8,.7,np.log(.003),np.log(.4),np.log(.6),np.log(2.)]);expanded=np.r_[t[:2],0.,t[2:]];a=filtering(expanded,tiny);b=filter2(t[:-1],tiny,np.exp(t[-1]),True);assert np.max(abs(a['predict_tb']-b['predict_tb']))<1e-12
    # Prefix identity proves no future labels or event counts enter the covariate.
    from scripts.patient_state_v1.activity_mark_controls import causal_activity
    tt=d['t'];groups=np.cumsum(d['reset']);aa=causal_activity(tt,groups,500,1/60);bb=causal_activity(tt[:1000],groups[:1000],500,1/60);assert np.array_equal(aa[:1000],bb)
    write_json(OUT/'numerical_canary.json',dict(status='PASS',zero_activity_max_probability_difference=np.max(abs(a['predict_tb']-b['predict_tb'])),prefix_activity_identity=True))
    folds=json.loads((RUN/'splits.json').read_text());scopes=[dict(scope='full',end=len(d['y']))]+[dict(scope=f"fold{f['fold']}",end=f['train_end'],test_end=f['test_end']) for f in folds];jobs=[]
    for scope in scopes:
        for carry in [False,True]:
            for minutes in [1,5]:
                for i,(tf,ts) in enumerate([(.003,1.5),(.02,6.)]):jobs.append(dict(id=f"{scope['scope']}_carry{int(carry)}_w{minutes}_{i}",carry=carry,minutes=minutes,tf=tf,ts=ts,**scope))
    write_json(OUT/'contract.json',dict(question='Does conditioning the mark probability on causal total event activity explain apparent excess persistence?',equation='p(TB_i)=sigmoid(b + gamma*h_i + c*log(recent_total_rate/training_rate) + fast_i + background_i)',state='Two OU components with free ordered times, unchanged from v1.13',data='Same strict interictal events and chronological folds; covariate baseline learned from training prefix only; missing coverage initializes at training mean',activity_windows_minutes=[1,5],not_claimed='Conditional association, not a neural causal coupling or autonomous event-time model',n_jobs=len(jobs)))
    with ProcessPoolExecutor(max_workers=24) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(complete=i+1,total=len(jobs),id=r['job']['id'],status=r['status'],elapsed=r['elapsed'],loglik=r.get('loglik'),success=r.get('success'))),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs)))
if __name__=='__main__':main()
