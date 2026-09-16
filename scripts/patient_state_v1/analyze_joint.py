"""Compare binned shared and independent latent states on identical forward outcomes."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.joint_counts import OUT,prepare,filter_model

def worker(r):
    j=r['job'];path=OUT/'filtered'/f"b{j['seconds']}_{j['scope']}_{r['model']}.npz"
    if path.exists():return str(path)
    d=prepare(j['seconds']);f=filter_model(np.array(r['theta']),d,r['model'],order=64)
    path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**f);return str(path)

def main():
    best={}
    for p in (OUT/'fits').glob('*.json'):
        r=json.loads(p.read_text())
        if r['status']!='COMPLETE':continue
        k=(r['job']['seconds'],r['job']['scope'],r['model'])
        if k not in best or r['loglik']>best[k]['loglik']:best[k]=r
    write_json(OUT/'best_fits.json',{str(k):v for k,v in best.items()})
    with ProcessPoolExecutor(max_workers=20) as ex:
        for f in as_completed([ex.submit(worker,r) for r in best.values()]):print(f.result(),flush=True)
    folds=json.loads((RUN/'splits.json').read_text());eventdata=np.load(RUN/'observations.npz');rows=[]
    for seconds in (60,300):
        d=prepare(seconds)
        for fold in folds:
            scope=f"fold{fold['fold']}";start=best[seconds,scope,'mark']['job']['end']
            nextfold=fold['fold']+1
            end=best[seconds,f'fold{nextfold}','mark']['job']['end'] if nextfold<len(folds) else len(d['n'])
            arrays={m:dict(np.load(OUT/'filtered'/f'b{seconds}_{scope}_{m}.npz')) for m in ('mark','rate','shared')}
            for model in ('shared','independent'):
                ll=arrays['shared']['loglik_terms'] if model=='shared' else arrays['mark']['loglik_terms']+arrays['rate']['loglik_terms']
                rate=arrays['shared']['predicted_rate'] if model=='shared' else arrays['rate']['predicted_rate']
                tbrate=arrays['shared']['predicted_tb_rate'] if model=='shared' else arrays['rate']['predicted_rate']*arrays['mark']['prior_tb_probability']
                n=d['n'][start:end];ex=d['exposure'][start:end];y=d['y'][start:end]
                rows.append(dict(seconds=seconds,fold=fold['fold'],model=model,n_bins=end-start,n_events=n.sum(),observed_hours=ex.sum(),loglik=ll[start:end].sum(),
                                 loglik_per_hour=ll[start:end].sum()/ex.sum(),observed_total_rate=n.sum()/ex.sum(),predicted_total_rate=np.dot(rate[start:end],ex)/ex.sum(),
                                 observed_tb_rate=y.sum()/ex.sum(),predicted_tb_rate=np.dot(tbrate[start:end],ex)/ex.sum(),
                                 count_mae=np.mean(abs(rate[start:end]*ex-n)),tb_count_mae=np.mean(abs(tbrate[start:end]*ex-y)),
                                 predicted_rate_over_observed_capacity=int(np.sum(rate[start:end]>14400))))
    pd.DataFrame(rows).to_csv(OUT/'forward_joint_scores.csv',index=False)
    write_json(OUT/'analysis_status.json',dict(status='COMPLETE_APPROXIMATE_FILTER',n_best_fits=len(best),n_forward_comparisons=len(rows),
        evidence_scope='binned selected-event counts and composition; no sub-bin interval recovery',numerical_review='GH Gaussian filter and Laplace fit; independent particle check pending'))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':main()
