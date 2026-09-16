"""Resolve joint likelihood ranking and held-out gains by deterministic refinement."""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,cupy as cp
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.joint_state_grid import evaluate,canary
from scripts.patient_state_v1.renewal import prepare
OUT=RUN/'joint_grid_refinement_v1_24'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=0);args=ap.parse_args();cp.cuda.Device(args.gpu).use();canary();d=prepare(5,.25);jobs=[]
    for size in [128,256,512]:
        for method in ['laplace','adf']:
            for coupled in [False,True]:
                folder='joint_two_state_v1_20' if method=='laplace' else 'joint_adf_refit_v1_23';pattern=f'b5_full_c{int(coupled)}_*.json' if method=='laplace' else f'full_c{int(coupled)}_*.json';choices=[json.loads(p.read_text()) for p in (RUN/folder/'fits').glob(pattern)];r=max(choices,key=lambda a:a['loglik']);jobs.append(dict(id=f'full_{method}_c{int(coupled)}_g{size}',scope='full',method=method,coupled=coupled,grid=size,source=r,end=len(d['n'])))
    inv=json.loads((RUN/'seizures.json').read_text());ev=np.load(RUN/'observations.npz');folds=json.loads((RUN/'splits.json').read_text());cuts=[inv[int(ev['epoch'][f['test_start']])-1]['offset'] for f in folds];stops=cuts[1:]+[float(d['hi'][-1])]
    for size in [256,512]:
        for fold,(cut,stop) in enumerate(zip(cuts,stops)):
            for coupled in [False,True]:
                choices=[json.loads(p.read_text()) for p in (RUN/'joint_two_state_v1_20/fits').glob(f'b5_fold{fold}_c{int(coupled)}_*.json')];r=max(choices,key=lambda a:a['loglik']);jobs.append(dict(id=f'fold{fold}_laplace_c{int(coupled)}_g{size}',scope=f'fold{fold}',method='laplace',coupled=coupled,grid=size,source=r,end=int(np.searchsorted(d['hi'],stop,side='right')),test_lo=int(np.searchsorted(d['hi'],cut,side='right'))))
    write_json(OUT/'contract.json',dict(question='Are joint-observation parameter rankings and held-out gains supported by a resolved full joint state likelihood?',grids=[128,256,512],observation_seconds=5,full_points='Laplace and ADF fits, each with independent or coupled observations',forward='Laplace training-prefix parameters; strict held-out intervals, 256/512 grids',factorization='At c=0 compare full joint grid with separate one-dimensional references',n_jobs=len(jobs),limits='Finite grid and 5-second observation approximation still require explicit refinement assessment; this evaluates existing parameters, not a new optimized physical model'))
    write_json(OUT/'queue.json',jobs)
    for i,j in enumerate(jobs):
        p=OUT/'evaluations'/f"{j['id']}.json"
        if p.exists():continue
        print(json.dumps(dict(job=j['id'],phase='START',index=i+1,total=len(jobs))),flush=True);data={k:v[:j['end']].copy() for k,v in d.items() if np.ndim(v)>0};r=evaluate(data,j['source']['theta'],j['coupled'],j['grid'],j['grid'],True,j['scope']!='full');terms=r.pop('loglik_terms',None);r.update(status='COMPLETE',job=j)
        if terms is not None:p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p.with_suffix('.npz'),loglik_terms=terms,test_lo=j['test_lo'],test_hi=j['end'])
        write_json(p,r);print(json.dumps(dict(job=j['id'],phase='COMPLETE',loglik=r['loglik'],elapsed=r['elapsed'])),flush=True);cp.get_default_memory_pool().free_all_blocks()
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs)))
if __name__=='__main__':main()
