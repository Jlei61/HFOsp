"""A factorized numerical reference and generated sequences for ADF refits."""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.joint_two_state import unpack
from scripts.patient_state_v1.renewal import prepare
from scripts.patient_state_v1.grid_renewal import evaluate as grid_rate
from scripts.patient_state_v1.nonlinear_drift import evaluate as grid_mark
from scripts.patient_state_v1.joint_two_state_filter import evaluate as adf
import scripts.patient_state_v1.matched_generation as matched
OUT=RUN/'joint_adf_audit_v1_23'

def reference(j):
    p=OUT/'reference'/f"{j['id']}.json";start=time.time()
    try:
        d=prepare(5,.25);t=np.array(j['theta']);rate=grid_rate(d,np.r_[t[1],t[4:]],j['points']);mark=grid_mark(np.r_[t[0],t[2:4]],d,'ou',384,3.);approx=adf(t,d,False,80,True);r=dict(status='COMPLETE',job=j,elapsed=time.time()-start,rate_grid_loglik=rate['loglik'],mark_grid_loglik=mark['loglik'],factorized_reference_loglik=rate['loglik']+mark['loglik'],adf_loglik=approx['loglik'],rate_grid_mass_loss=rate['max_transition_mass_loss'],rate_grid_edge_mass=rate['max_posterior_edge_mass'],mark_grid_edge_mass=mark['stationary_edge_mass'])
    except Exception:r=dict(status='FAILED',job=j,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(p,r);return r

def main():
    assert json.loads((RUN/'joint_adf_refit_v1_23/status.json').read_text())['status']=='COMPLETE';best={}
    for p in (RUN/'joint_adf_refit_v1_23/fits').glob('full_*.json'):
        r=json.loads(p.read_text());key=r['job']['coupled']
        if key not in best or r['loglik']>best[key]['loglik']:best[key]=r
    laplace=max([json.loads(p.read_text()) for p in (RUN/'joint_two_state_v1_20/fits').glob('b5_full_c0_*.json')],key=lambda r:r['loglik']);sources={'laplace':laplace['theta'],'adf':best[False]['theta']};gridfit=max([json.loads(p.read_text()) for p in (RUN/'renewal_grid_refit_v1_7/fits').glob('g1_full_m512_*.json')],key=lambda r:r['loglik']);t=np.array(laplace['theta']);t[1]=gridfit['theta'][0];t[4:]=gridfit['theta'][1:];sources['rate_grid_refit']=t;checks=[]
    for source,t in sources.items():
        for points in [512,1024,2048]:checks.append(dict(id=f'{source}_g{points}',source=source,theta=t,points=points))
    matched.OUT=OUT;jobs=[]
    for coupled,r in best.items():
        b,a,c,ts,ss,tr,sr=unpack(r['theta'],coupled)
        for rep in range(128):jobs.append(dict(version='adf_refit',model='coupled' if coupled else 'independent',rep=rep,seed=850000+rep,config=dict(hours=24.,step=.5,deadtime=.25,a=a,b=b,tau_r=tr,sd_r=sr,tau_s=ts,sd_s=ss,c=c,kind=2,gamma=0.)))
    write_json(OUT/'contract.json',dict(question='Does a better ADF objective correspond to a better controlled likelihood and generated data?',numerical_control='At c=0 the two latent states factor exactly. Compare deterministic rate-grid likelihood plus mark-grid likelihood against ADF at the same parameters.',numerical_refinement='Rate grids 512/1024/2048; mark grid384 with max3-second propagation step; 5-second observations identical',generation='128 matched-exposure autonomous event sequences per ADF-refitted model; same seed set as Laplace-generated comparison',limits='Reference factorization is valid only at c=0; it cannot by itself validate the coupled likelihood',n_reference_checks=len(checks),n_generated=len(jobs)))
    with ProcessPoolExecutor(max_workers=20) as ex:
        fs=[ex.submit(reference,j) for j in checks]+[ex.submit(matched.worker,j) for j in jobs]
        for i,f in enumerate(as_completed(fs)):
            r=f.result();print(json.dumps(dict(done=i+1,total=len(fs),id=r['job'].get('id',r['job'].get('model')),status=r['status'],elapsed=r.get('elapsed'))),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_reference_checks=len(checks),n_generated=len(jobs)))
if __name__=='__main__':main()
