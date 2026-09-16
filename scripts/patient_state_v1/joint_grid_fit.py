"""Bounded direct joint-grid refinement; no change to SNN or model family.

One local optimization from the better independently evaluated Laplace point.
Every objective evaluation is preserved, then the best candidate is checked at
twice the state-grid resolution. A budget exit is not optimizer convergence.
"""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,cupy as cp
from scipy.optimize import minimize
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.joint_state_grid import evaluate,canary
from scripts.patient_state_v1.renewal import prepare
OUT=RUN/'joint_grid_refit_v1_26'

class BudgetEnd(Exception):pass

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=1);args=ap.parse_args();cp.cuda.Device(args.gpu).use();canary();d=prepare(5,.25);choices=[json.loads(p.read_text()) for p in (RUN/'joint_two_state_v1_20/fits').glob('b5_full_c1_*.json')];source=max(choices,key=lambda r:r['loglik']);started=time.time();deadline=min(started+2.5*3600,1788968996+8.25*3600);bounds=[(-8,8),(np.log(.01),np.log(1e7)),(-5,5),(np.log(1/60),np.log(24)),(np.log(.01),np.log(5)),(np.log(1/3600),np.log(24)),(np.log(.01),np.log(5))];records=[];best=None
    write_json(OUT/'contract.json',dict(question='Does directly refining the full joint likelihood reduce the approximation-induced parameter and generation mismatch?',family='Same seven-parameter two-OU joint observation model as v1.20; no extra latent dimension',scope='Full-data development fit, one local start at existing Laplace solution; no new forward validation claimed',source=source,grid=256,final_grid=512,observation_seconds=5,max_objective_calls=200,max_optimizer_iterations=30,deadline_unix=deadline,selection='Numerical joint likelihood only, no seizure labels or generated-summary selection',limits='Grid and effective packing approximation remain; a local or budget-limited result is not a global optimum. Full-data refinement does not inherit old prefix-fit validation.'))
    def fun(theta):
        nonlocal best
        if time.time()>deadline or len(records)>=200:raise BudgetEnd()
        r=evaluate(d,theta,True,256,256,False);record=dict(index=len(records),theta=theta.tolist(),**r,elapsed_total=time.time()-started);records.append(record);write_json(OUT/'evaluations'/f'{len(records)-1:04d}.json',record)
        if best is None or r['loglik']>best['loglik']:best=record;write_json(OUT/'best_so_far.json',best)
        print(json.dumps(dict(evaluation=len(records),loglik=r['loglik'],best=best['loglik'],elapsed=time.time()-started)),flush=True)
        return -r['loglik']
    status='RUNNING';optimizer=None
    try:
        opt=minimize(fun,np.array(source['theta']),method='L-BFGS-B',bounds=bounds,options={'maxiter':30,'maxfun':190,'ftol':1e-10,'gtol':2e-3,'eps':2e-4,'maxls':15});optimizer=dict(success=bool(opt.success),message=str(opt.message),theta=opt.x,loglik=-opt.fun,nfev=int(opt.nfev),nit=int(opt.nit));status='OPTIMIZER_COMPLETE' if opt.success else 'OPTIMIZER_LIMIT_OR_FAILURE'
    except BudgetEnd:status='BUDGET_LIMIT'
    assert best is not None
    initial=records[0];reference=json.loads((RUN/'joint_grid_refinement_v1_24/evaluations/full_laplace_c1_g256.json').read_text());assert abs(initial['loglik']-reference['loglik'])<1e-7
    fine=evaluate(d,best['theta'],True,512,512,True);write_json(OUT/'result.json',dict(status=status,optimizer=optimizer,source=source,initial_grid=initial,best_grid=best,refined_grid=fine,n_evaluations=len(records),elapsed=time.time()-started,generated_validation='PENDING',forward_validation='NOT_RUN for this full-data-refined parameter set'))
    print(json.dumps(dict(status=status,n_evaluations=len(records),initial_loglik=initial['loglik'],best_loglik=best['loglik'],fine_loglik=fine['loglik'],elapsed=time.time()-started)),flush=True)

if __name__=='__main__':main()
