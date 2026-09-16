"""Calibrate apparent double wells against known single-OU label sequences."""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.special import expit
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.joint_counts import prepare
from scripts.patient_state_v1.nonlinear_drift import evaluate,fit_model
OUT=RUN/'nonlinear_calibration_v1_21'

def best_real():
    best={}
    for p in (RUN/'nonlinear_drift_v1_21/fits').glob('*.json'):
        r=json.loads(p.read_text());j=r['job'];key=(j['seconds'],j['scope'],j['kind'])
        if r['status']=='COMPLETE' and (key not in best or r['loglik']>best[key]['loglik']):best[key]=r
    return best

def worker(j):
    p=OUT/'fits'/f"{j['id']}.json"
    if p.exists():return json.loads(p.read_text())
    start=time.time()
    try:
        d=prepare(j['seconds']);truth=np.asarray(j['truth']);rng=np.random.default_rng(j['seed']);mu=truth[0];tau,sd=np.exp(truth[1:]);s=np.empty(len(d['n']));v=mu
        for i in range(len(s)):
            if d['reset'][i]:v=mu+sd*rng.normal()
            else:
                rho=np.exp(-d['dt'][i]/tau);v=mu+rho*(v-mu)+sd*np.sqrt(-np.expm1(-2*d['dt'][i]/tau))*rng.normal()
            s[i]=v
        d['y']=rng.binomial(d['n'].astype(np.int64),expit(s)).astype(float);q=d['y'].sum()/d['n'].sum();b=np.log(q/(1-q))
        initials=[np.r_[b,truth[1:]]] if j['kind']=='ou' else [np.array([b,np.log(.5),np.log(.7),-1.]),np.array([b,np.log(2.),np.log(1.5),8.])]
        choices=[fit_model(d,j['kind'],t) for t in initials];r=dict(max(choices,key=lambda a:a['loglik']));r.update(status='COMPLETE',job=j,elapsed=time.time()-start,initial_fits=choices)
    except Exception:r=dict(status='FAILED',job=j,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(p,r);return r

def resolution(j):
    p=OUT/'resolution'/f"{j['id']}.json";start=time.time()
    try:
        d=prepare(j['seconds']);r=evaluate(np.array(j['theta']),d,j['kind'],j['points'],j['step']);r={k:v for k,v in r.items() if np.ndim(v)==0};r.update(status='COMPLETE',job=j,elapsed=time.time()-start)
    except Exception:r=dict(status='FAILED',job=j,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(p,r);return r

def main():
    best=best_real();jobs=[];checks=[]
    for sec in [15,60]:
        truth=best[(sec,'full','ou')]['theta']
        for rep in range(64):
            for kind in ['ou','quartic']:jobs.append(dict(id=f'b{sec}_rep{rep:03d}_{kind}',seconds=sec,rep=rep,kind=kind,seed=710000+sec*1000+rep,truth=truth))
        for kind in ['ou','quartic']:
            for points in [96,192,384]:
                for step in [15.,3.]:checks.append(dict(id=f'b{sec}_{kind}_g{points}_dt{step}',seconds=sec,kind=kind,theta=best[(sec,'full',kind)]['theta'],points=points,step=step))
    write_json(OUT/'contract.json',dict(question='How often does a fitted single-well OU produce an apparently preferred quartic or negative-curvature potential after latent inference?',truth='Exact OU at observation-bin centers; actual patient counts/exposure/exclusions fixed; synthetic labels binomial(n,sigmoid(s))',replicates_per_bin=64,bins_seconds=[15,60],fit='Matched OU and quartic inference; quartic two starts with opposite potential shapes',readout='Patient likelihood gain and inferred curvature versus synthetic null; no standard chi-square calibration assumed',limits='OU-null calibration conditions on observed counts and uses binned mark model; it does not identify a neural potential',resolution_checks=len(checks),n_fits=len(jobs)))
    with ProcessPoolExecutor(max_workers=20) as ex:
        fs=[ex.submit(resolution,j) for j in checks]+[ex.submit(worker,j) for j in jobs]
        for i,f in enumerate(as_completed(fs)):
            r=f.result();print(json.dumps(dict(done=i+1,total=len(fs),id=r['job']['id'],status=r['status'],elapsed=r['elapsed'])),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_fits=len(jobs),n_checks=len(checks)))
if __name__=='__main__':main()
