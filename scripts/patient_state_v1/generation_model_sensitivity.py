"""Check whether generation misfit is confined to a particular fitted point."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
import scripts.patient_state_v1.free_conditional_generation as generator
OUT=RUN/'generation_model_sensitivity_v1_18'

def main():
    generator.OUT=OUT;jobs=[];sources={};best={}
    for p in (RUN/'two_timescale_free_tau_v1_13/fits').glob('full_*hist0_adf_*.json'):
        r=json.loads(p.read_text());j=r['job'];k=j['carry']
        if r['status']=='COMPLETE' and r['feasible'] and (k not in best or r['loglik']>best[k][0]['loglik']):best[k]=(r,p)
    for carry,(r,p) in best.items():
        model=f'nohistory_adf_carry{int(carry)}';theta=np.r_[r['theta'][0],0.,r['theta'][1:]].tolist();sources[model]=str(p)
        for rep in range(128):jobs.append(dict(model=model,carry=carry,theta=theta,rep=rep,seed=540000+int(carry)*1000+rep))
    profile={}
    for p in (RUN/'free_tau_profile_v1_14/fits').glob('*.json'):
        r=json.loads(p.read_text());j=r['job'];k=(j['carry'],j['tau'])
        wanted=j['method']=='adf' and j['tau'] in ([1.,1.5,2.25,3.5] if j['carry'] else [1.,1.5,2.25])
        if wanted and r['status']=='COMPLETE' and (k not in profile or r['loglik']>profile[k][0]['loglik']):profile[k]=(r,p)
    for i,((carry,tau),(r,p)) in enumerate(sorted(profile.items())):
        model=f'profile_history_carry{int(carry)}_tau{tau}';theta=np.r_[r['theta'],np.log(tau)].tolist();sources[model]=str(p)
        for rep in range(128):jobs.append(dict(model=model,carry=carry,theta=theta,rep=rep,seed=542000+i*1000+rep))
    write_json(OUT/'contract.json',dict(question='Do label-generation discrepancies persist after dropping direct history or moving within the approximate background-time profile support?',conditions=sources,n_replicates=128,scope='Conditional on observed event times; profile-point sensitivity is not posterior predictive sampling',selection='No-history ADF full fits and predefined -1.92-reference background grid points; not chosen using seizure types or generation similarity'))
    with ProcessPoolExecutor(max_workers=24) as ex:
        for i,f in enumerate(as_completed([ex.submit(generator.worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(complete=i+1,total=len(jobs),model=r['job']['model'],adjacent_excess=r['adjacent_excess'])),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_runs=len(jobs)))
if __name__=='__main__':main()
