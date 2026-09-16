"""Does state-dependent observation timing imitate a changing mark timescale?

Generate new event times and labels on fixed patient exposure, then repeat the
same nested-prefix conditional-mark fits. All parameters remain stationary.
This calibrates omission of informative times at actual fitted coupling strength.
"""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.generate import simulate
from scripts.patient_state_v1.joint_two_state import unpack
from scripts.patient_state_v1.model import fit,laplace,slice_data
from scripts.patient_state_v1.advanced_controls import history_data
OUT=RUN/'joint_prefix_calibration_v1_25'

def sequence(j):
    origin=float(np.load(RUN/'observations.npz')['origin_epoch']);ex=pd.read_csv(RUN/'exposure.csv');inv=json.loads((RUN/'seizures.json').read_text());end=ex.end_epoch.max()-origin;intervals=[];lo=0.
    for v in inv:
        hi=v['onset']-origin
        if hi>lo:intervals.append((max(lo,0),hi))
        lo=v['offset']-origin
    if end>lo:intervals.append((max(lo,0),end))
    b,a,c,ts,ss,tr,sr=unpack(j['theta'],j['coupled']);times=[];labels=[]
    for epoch,(lo,hi) in enumerate(intervals):
        if hi<=0:continue
        t,y,*_=simulate(hours=(hi-lo)/3600,step=.5,seed=j['seed']+epoch*10000,a=a,b=b,tau_r=tr,sd_r=sr,tau_s=ts,sd_s=ss,c=c,kind=2,gamma=0.,deadtime=.25);t+=lo
        for v in ex.itertuples():
            aa,bb=v.start_epoch-origin,v.end_epoch-origin
            if bb<=lo or aa>=hi:continue
            ok=(t>=aa)&(t+.25<=bb);times.append(t[ok]);labels.append(y[ok])
    t=np.concatenate(times);y=np.concatenate(labels);order=np.argsort(t);t=t[order]+origin;y=y[order].astype(int);assert np.all(np.diff(t)>=.25-1e-6);epoch=np.searchsorted(np.array([v['offset'] for v in inv]),t,side='right');reset=np.r_[True,np.diff(epoch)!=0];dt=np.r_[0,np.diff(t)/3600];dt[reset]=0
    return dict(t=(t-origin)/3600,dt=dt,reset=reset,y=y,n=np.ones(len(y)),x=np.column_stack([np.ones(len(y)),np.zeros((len(y),2))]),epoch=epoch,time_epoch=t)

def worker(j):
    p=OUT/'runs'/f"c{int(j['coupled'])}_{j['rep']:03d}.json"
    if p.exists():return json.loads(p.read_text())
    start=time.time()
    try:
        d=sequence(j);ev=np.load(RUN/'observations.npz');folds=json.loads((RUN/'splits.json').read_text());cuts=[float(ev['origin_epoch'])+3600*float(ev['t'][f['train_end']-1])+.25 for f in folds];scopes=[('full',len(d['y']))]+[(f'fold{i}',int(np.searchsorted(d['time_epoch']+.25,cut,side='right'))) for i,cut in enumerate(cuts)];rows=[]
        for scope,end in scopes:
            data=slice_data(d,0,end);b=np.log((data['y'].mean()+1e-6)/(1-data['y'].mean()+1e-6))
            for model in ['ou','ou_history']:
                dd=history_data(data) if model=='ou_history' else data;starts=[]
                for tau in [.2,2.]:
                    if model=='ou':r=fit(dd,'ou',np.r_[b,np.log(tau),np.log(.6)])
                    else:
                        opt=minimize(lambda t:-laplace(np.r_[t[:2],0.,t[2:]],dd,True),np.r_[b,.2,np.log(tau),np.log(.6)],method='L-BFGS-B',bounds=[(-8,8),(-8,8),(np.log(1/60),np.log(24)),(np.log(.01),np.log(5))],options={'maxiter':150,'ftol':1e-10,'eps':2e-5});r=dict(theta=opt.x,loglik=-opt.fun,success=bool(opt.success),message=str(opt.message))
                    starts.append(r)
                selected=max(starts,key=lambda r:r['loglik']);rows.append(dict(scope=scope,model=model,n_events=end,selected=selected,initial_fits=starts))
        result=dict(status='COMPLETE',job=j,n_events=len(d['y']),tb_fraction=float(d['y'].mean()),fits=rows,elapsed=time.time()-start);p.parent.mkdir(parents=True,exist_ok=True)
        if j['rep']==0:np.savez_compressed(p.with_suffix('.npz'),**d)
    except Exception:result=dict(status='FAILED',job=j,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(p,result);return result

def main():
    ev=np.load(RUN/'observations.npz');assert 'origin_epoch' in ev.files
    jobs=[];sources={}
    for coupled in [False,True]:
        rs=[json.loads(p.read_text()) for p in (RUN/'joint_two_state_v1_20/fits').glob(f'b5_full_c{int(coupled)}_*.json')];r=max(rs,key=lambda r:r['loglik']);sources[str(coupled)]=r
        for rep in range(64):jobs.append(dict(coupled=coupled,theta=r['theta'],rep=rep,seed=940000+rep))
    write_json(OUT/'contract.json',dict(question='Can informative observation at the patient-fitted joint coupling produce the observed prefix dependence of conditional-mark timescales, without time-varying latent parameters?',source_fits=sources,n_sequences=128,n_selected_prefix_fits=1024,replicates_per_generator=64,control='Independent and coupled rate/mode models, same seed set and actual fixed exposure/exclusions',generated='All interictal event times and labels are new; no replay or fitted seizure type',analysis='OU and OU-history fitted on same elapsed-time prefixes as patient; two starts each',limits='Fitted joint generators already fail some patient marginals; this is a calibration of a particular alternative explanation, not overall model acceptance. Effective packing support is approximate.'))
    write_json(OUT/'queue.json',jobs)
    with ProcessPoolExecutor(max_workers=20) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(done=i+1,total=128,coupled=r['job']['coupled'],rep=r['job']['rep'],status=r['status'],elapsed=r['elapsed'],events=r.get('n_events'))),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_sequences=128))

if __name__=='__main__':main()
