"""Evaluate new observation and drift variants on strict held-out intervals."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.renewal import prepare
OUT=RUN/'joint_nonlinear_review_v1_21'

def bootstrap(df):
    group=df.groupby('block')[['gain','events','hours']].sum();arr=group.to_numpy();rng=np.random.default_rng(223322);ii=rng.integers(0,len(arr),(4000,len(arr)));draw=arr[ii].sum(axis=1);q=np.quantile(draw[:,0]/draw[:,1],[.025,.975]);return dict(gain_per_event=float(arr[:,0].sum()/arr[:,1].sum()),lower=float(q[0]),upper=float(q[1]),n_events=int(arr[:,1].sum()),hours=float(arr[:,2].sum()),n_blocks=len(arr))

def main():
    OUT.mkdir(exist_ok=True);folds=json.loads((RUN/'splits.json').read_text());inv=json.loads((RUN/'seizures.json').read_text());ev=np.load(RUN/'observations.npz');cuts=[inv[int(ev['epoch'][f['test_start']])-1]['offset'] for f in folds];allrows=[];summary=[]
    for family in ['joint','nonlinear']:
        for sec in (([1,5,15] if (RUN/'joint_two_state_filter_v1_20/fine_status.json').exists() else [5,15]) if family=='joint' else [15,60]):
            d=prepare(sec,.25);ends=cuts[1:]+[float(d['hi'][-1])]
            variants=[(q,rf) for q in [40,80] for rf in [False,True]] if family=='joint' else [(0,False)]
            for q,rf in variants:
                frames=[]
                for fold,(start,stop) in enumerate(zip(cuts,ends)):
                    lo=np.searchsorted(d['hi'],start,side='right');hi=np.searchsorted(d['hi'],stop,side='right');terms=[];source=[]
                    for variant in [False,True]:
                        if family=='joint':p=RUN/'joint_two_state_filter_v1_20/evaluations'/f'b{sec}_fold{fold}_c{int(variant)}_q{q}_rf{int(rf)}.json'
                        else:
                            kind='quartic' if variant else 'ou';choices=[(p,json.loads(p.read_text())) for p in (RUN/'nonlinear_drift_v1_21/fits').glob(f'b{sec}_fold{fold}_{kind}_*.json')];p,r=max(choices,key=lambda a:a[1]['loglik'])
                        r=json.loads(p.read_text());assert r['status']=='COMPLETE';f=np.load(p.with_suffix('.npz'));terms.append(f['loglik_terms'][lo:hi]);source.append(str(p))
                    frame=pd.DataFrame(dict(fold=fold,bin_index=np.arange(lo,hi),gain=terms[1]-terms[0],events=d['n'][lo:hi],hours=d['physical_exposure'][lo:hi],block=np.floor(d['t'][lo:hi]/6).astype(int)));frames.append(frame);allrows.append(dict(family=family,seconds=sec,order=q,rate_first=rf,fold=fold,gain=float(frame.gain.sum()),events=int(frame.events.sum()),sources=source))
                frame=pd.concat(frames,ignore_index=True);assert int(frame.events.sum())==16157;result=bootstrap(frame);summary.append(dict(family=family,seconds=sec,order=q,rate_first=rf,**result));frame.to_csv(OUT/f'{family}_b{sec}_q{q}_rf{int(rf)}_terms.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'forward_summary.csv',index=False);write_json(OUT/'fold_scores.json',allrows);write_json(OUT/'scientific_audit.json',dict(status='COMPLETE',n_test_events=16157,cutoffs='Parameters fitted strictly before test interictal interval; same held-out intervals in all comparisons',joint='Joint event-time and bin-label density, includes silent exposure; sequential Gaussian projections audited at 40/80 nodes and both orders',nonlinear='Quartic versus OU with identical bins and numerical grid; does not establish neural bistability',uncertainty='Paired resampling of 6-hour elapsed-time blocks; exploratory uncertainty from one developing patient record',selection='Only training likelihood selects starts'))
    print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()
