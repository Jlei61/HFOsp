"""Strict pre-interval training for held-out effective point-process likelihood.

Earlier exploratory count fits cut at the first test event, including up to 6.9
minutes of test-interval silence in rate training. Freeze parameters before the
preceding seizure offset instead; event-conditional mark fits are unaffected.
"""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.renewal import prepare
from scripts.patient_state_v1.grid_renewal import evaluate
from scripts.patient_state_v1.grid_rate_fit import worker

OUT=RUN/'renewal_forward_v1_10'

def main():
    folds=json.loads((RUN/'splits.json').read_text());inv=json.loads((RUN/'seizures.json').read_text());data=np.load(RUN/'observations.npz');d=prepare(5,.25);cuts=[inv[int(data['epoch'][f['test_start']])-1]['offset'] for f in folds];ends=cuts[1:]+[float(d['hi'][-1])];jobs=[];audit=[]
    for f,start,stop in zip(folds,cuts,ends):
        end=int(np.searchsorted(d['hi'],start,side='right'));test_end=int(np.searchsorted(d['hi'],stop,side='right'));old=float(data['origin_epoch'])+data['t'][f['test_start']]*3600;rate=d['n'][:end].sum()/d['exposure'][:end].sum()
        initial=[np.log(rate),np.log(.2),np.log(3)]
        for i,shift in enumerate((0.,-.5)):
            t=np.array(initial);t[0]+=shift;jobs.append(dict(id=f"strict_v2_g5_fold{f['fold']}_m512_{i}",seconds=5,scope=f"strict_fold{f['fold']}",end=end,points=512,initial=t,train_cutoff_epoch=start,test_end_epoch=stop,test_end=test_end,fold=f['fold']))
        audit.append(dict(fold=f['fold'],old_cutoff_epoch=old,corrected_cutoff_epoch=start,removed_test_silence_minutes=(old-start)/60,n_train_events=int(d['n'][:end].sum()),n_test_events=int(d['n'][end:test_end].sum())))
    write_json(OUT/'cutoff_audit.json',dict(status='CORRECTION_APPLIED',folds=audit,explanation='Exploratory count models included initial test silence before the first test event. Corrected fits use pre-interval cutoffs. Mark-only fits did not use this silent exposure.'))
    write_json(OUT/'queue.json',jobs);results=[]
    with ProcessPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(worker,j) for j in jobs]):
            r=f.result();results.append(r);print(json.dumps(dict(id=r['job']['id'],status=r['status'],elapsed=r['elapsed'],loglik=r.get('loglik'))),flush=True)
    scores=[]
    for fold in range(3):
        choices=[r for r in results if r['job']['fold']==fold and r['status']=='COMPLETE'];r=max(choices,key=lambda q:q['loglik']);j=r['job'];lo,hi=j['end'],j['test_end'];filt=evaluate(d,r['theta'],1024,return_terms=True);rate=d['n'][:lo].sum()/d['exposure'][:lo].sum();constant=d['n'][lo:hi]*np.log(rate)-d['exposure'][lo:hi]*rate
        scores.append(dict(fold=fold,n_events=int(d['n'][lo:hi].sum()),observed_hours=d['physical_exposure'][lo:hi].sum(),ou_log_density=filt['loglik_terms'][lo:hi].sum(),constant_log_density=constant.sum(),theta=r['theta'],points=1024))
        np.savez_compressed(OUT/f'fold{fold}_filter.npz',mean=filt['mean'],variance=filt['variance'],loglik_terms=filt['loglik_terms'],test_lo=lo,test_hi=hi)
    write_json(OUT/'forward_scores.json',scores);write_json(OUT/'status.json',dict(status='COMPLETE',n_fits=len(results),likelihood='Held-out event-time log density, per-hour hazard units; absolute values depend on time units, model differences do not'))

if __name__=='__main__':main()
