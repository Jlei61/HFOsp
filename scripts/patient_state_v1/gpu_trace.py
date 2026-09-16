"""Independent particle audit of filtered states and actual forward probabilities."""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from numba import cuda
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.gpu_particles import evaluate_trace
from scripts.patient_state_v1.analyze_first import best_fits,metrics
from scripts.patient_state_v1.model import filter_adf

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--model',choices=['ou','ou_cycle'],required=True);ap.add_argument('--replicates',type=int,default=64);args=ap.parse_args()
    cuda.select_device(args.gpu);data=dict(np.load(RUN/'observations.npz'));best=best_fits();folds=json.loads((RUN/'splits.json').read_text());out=RUN/'gpu_traces';out.mkdir(exist_ok=True)
    scopes=['full']+[f"fold{f['fold']}" for f in folds];summary=[]
    audit_path=out/f'{args.model}_audit.json'
    old=json.loads(audit_path.read_text()).get('rows',[]) if audit_path.exists() else []
    for scope in scopes:
        previous=[row for row in old if row['scope']==scope]
        if previous and (out/f'{scope}_{args.model}.npz').exists():summary.append(previous[0]);continue
        r=best[scope,args.model];t=np.array(r['theta']);par=t if args.model=='ou_cycle' else np.r_[t[0],0,0,t[1:]];start=time.time()
        ll,tr=evaluate_trace(data,np.tile(par,(args.replicates,1)),1024,seed=267000+100*scopes.index(scope)+args.gpu)
        mean=tr.mean(axis=0);se=tr.std(axis=0)/np.sqrt(len(tr));a=filter_adf(t,data,args.model=='ou_cycle')
        np.savez_compressed(out/f'{scope}_{args.model}.npz',particle_mean=mean,particle_mc_se=se,loglik_replicates=ll)
        row=dict(scope=scope,model=args.model,particles=1024,replicates=args.replicates,elapsed=time.time()-start,
                 state_mean_rmse=np.sqrt(np.mean((mean[:,1]-a['mean'])**2)),probability_rmse=np.sqrt(np.mean((mean[:,3]-a['predict_tb'])**2)),
                 probability_abs_error_quantiles=np.quantile(abs(mean[:,3]-a['predict_tb']),[.5,.9,.99,1]),
                 state_abs_error_quantiles=np.quantile(abs(mean[:,1]-a['mean']),[.5,.9,.99,1]))
        if scope!='full':
            f=folds[int(scope.replace('fold',''))];lo,hi=f['test_start'],f['test_end'];row['particle_forward']=metrics(mean[lo:hi,3],data['y'][lo:hi],data['n'][lo:hi]);row['adf_forward']=metrics(a['predict_tb'][lo:hi],data['y'][lo:hi],data['n'][lo:hi])
        summary.append(row);write_json(audit_path,dict(status='RUNNING' if len(summary)<4 else 'COMPLETE',rows=summary));print(json.dumps(dict(scope=scope,model=args.model,elapsed=row['elapsed'],state_rmse=float(row['state_mean_rmse']),probability_rmse=float(row['probability_rmse']))),flush=True)

if __name__=='__main__':main()
