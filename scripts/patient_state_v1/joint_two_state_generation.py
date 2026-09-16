"""Joint latent activity/mode generation on the patient's fixed coverage."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.joint_two_state import unpack
from scripts.patient_state_v1.generate import simulate
import scripts.patient_state_v1.matched_generation as matched
OUT=RUN/'joint_two_state_generation_v1_20'

def main():
    # New additive-rate interface must reduce exactly to the old independent one.
    cfg=dict(hours=.1,step=.5,seed=880001,a=6.,b=-.8,tau_r=.1,sd_r=1.,tau_s=.3,sd_s=.6,c=0.,gamma=0.,deadtime=.25)
    old=simulate(**cfg,kind=0);new=simulate(**cfg,kind=2);assert all(np.array_equal(a,b) for a,b in zip(old,new));write_json(OUT/'numerical_canary.json',dict(status='PASS',zero_coupling_exact_seeded_equivalence=True))
    matched.OUT=OUT;jobs=[];sources={}
    for coupled in [False,True]:
        rs=[json.loads(p.read_text()) for p in (RUN/'joint_two_state_v1_20/fits').glob(f'b5_full_c{int(coupled)}_*.json')];r=max(rs,key=lambda r:r['loglik']);sources[str(coupled)]=r;b,a,c,ts,ss,tr,sr=unpack(r['theta'],coupled)
        for rep in range(128):
            config=dict(hours=24.,step=.5,deadtime=.25,a=a,b=b,tau_r=tr,sd_r=sr,tau_s=ts,sd_s=ss,c=c,kind=2,gamma=0.);jobs.append(dict(version='joint_two_state',model='coupled' if coupled else 'independent',rep=rep,seed=850000+rep,config=config))
    write_json(OUT/'contract.json',dict(question='Does the jointly fitted state-rate association generate both patient event density and mode statistics?',source_fits=sources,n_sequences=256,coverage='Actual coverage and excluded ictal intervals fixed; all interictal event times and labels generated',state='Independent stationary initial draws at interictal intervals, as in inference; no biological reset claimed',no_event_replay=True,selection='All replicates retained; same seed set in both conditions',limits='Laplace parameters and effective packing support remain approximations; matching rates alone is insufficient'))
    with ProcessPoolExecutor(max_workers=20) as ex:
        for i,f in enumerate(as_completed([ex.submit(matched.worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(done=i+1,total=len(jobs),model=r['job']['model'],rep=r['job']['rep'],status=r['status'],rate=r.get('rate_per_hour'),tb=r.get('tb_fraction'))),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_runs=len(jobs)))
if __name__=='__main__':main()
