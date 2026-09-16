"""Forecast future marks from state information available before a fixed horizon.

Only models without direct previous-label terms are used: forecasting those terms
would require a model for unobserved intervening event times and marks. We do not
silently set an unobserved autoregressive input to zero and call it a full forecast.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from scipy.special import expit
from numpy.polynomial.hermite import hermgauss
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.analyze_first import best_fits,metrics
OUT=RUN/'two_scale_horizon_v1_16'
def main():
    OUT.mkdir(exist_ok=True);data=np.load(RUN/'observations.npz');ev=pd.read_csv(RUN/'events.csv');iv=json.loads((RUN/'seizures.json').read_text());folds=json.loads((RUN/'splits.json').read_text());base=best_fits();times=ev.start_epoch.to_numpy();ends=ev.end_epoch.to_numpy();offsets=np.array([s['offset'] for s in iv]);origin=float(data['origin_epoch']);nodes,w=hermgauss(48);w/=np.sqrt(np.pi);best={};records=[]
    for directory,kind in [('two_timescale_free_tau_v1_13','ou2'),('brownian_fast_marks_v1_12','brownian')]:
        for p in (RUN/directory/'fits').glob('fold*.json'):
            r=json.loads(p.read_text());j=r['job']
            if r['status']!='COMPLETE' or j['history'] or (kind=='ou2' and j['method']!='adf') or (kind=='brownian' and j['drift']):continue
            k=(j['scope'],kind,j['carry'])
            if k not in best or r['loglik']>best[k][0]['loglik']:best[k]=(r,p)
    for fold in folds:
        scope=f"fold{fold['fold']}";train_end=ends[fold['train_end']-1];models={'constant':(base[scope,'constant'],None,'constant',False),'ou':(base[scope,'ou'],None,'ou',False)}
        from scripts.patient_state_v1.model import filter_adf
        one=filter_adf(np.array(base[scope,'ou']['theta']),dict(data))
        for (sc,k,carry),(r,p) in best.items():
            if sc==scope:models[f'{k}_carry{int(carry)}']=(r,np.load(p.with_suffix('.npz')),k,carry)
        for minutes in [0,1,5,15,60,120]:
            target=np.arange(fold['test_start'],fold['test_end']);cut=times[target]-minutes*60;valid=(cut>=train_end)&(np.searchsorted(offsets,cut,side='right')==data['epoch'][target]);target=target[valid];cut=cut[valid];previous=np.searchsorted(ends,cut,side='right')-1;assert np.all(previous>=0);elapsed=(times[target]-times[previous])/3600;same=data['epoch'][previous]==data['epoch'][target]
            for name,(r,f,kind,carry) in models.items():
                t=np.array(r['theta']);b=t[0]
                if kind=='constant':p=np.full(len(target),expit(b))
                elif kind=='ou':
                    tf,sd=np.exp(t[-2:]);a=np.exp(-elapsed/tf);m=np.where(same,a*one['mean'][previous],0);v=np.where(same,a*a*one['variance'][previous]+sd*sd*(-np.expm1(-2*elapsed/tf)),sd*sd);p=expit(b+m[:,None]+np.sqrt(2*v[:,None])*nodes)@w
                else:
                    tf,sdf=np.exp(t[1:3]);a=np.exp(-elapsed/tf);m0=np.where(same,a*f['mean'][previous,0],0);v0=np.where(same,a*a*f['variance'][previous,0]+sdf*sdf*(-np.expm1(-2*elapsed/tf)),sdf*sdf)
                    if kind=='ou2':
                        sds,ts=np.exp(t[3:5]);c=np.exp(-elapsed/ts);use=same|carry;m1=np.where(use,c*f['mean'][previous,1],0);v1=np.where(use,c*c*f['variance'][previous,1]+sds*sds*(-np.expm1(-2*elapsed/ts)),sds*sds)
                    else:
                        sigma=np.exp(t[3]);c=np.ones(len(target));use=same|carry;age=(times[target]-np.where(data['epoch'][target]>0,offsets[np.maximum(data['epoch'][target]-1,0)],origin))/3600;m1=np.where(use,f['mean'][previous,1],0);v1=np.where(use,f['variance'][previous,1]+sigma*sigma*elapsed,r['job']['sd0']**2+sigma*sigma*age)
                    cross=np.where(same,a*c*f['variance'][previous,2],0);v=np.maximum(v0+v1+2*cross,1e-12);p=expit(b+(m0+m1)[:,None]+np.sqrt(2*v[:,None])*nodes)@w
                y=data['y'][target];p=np.clip(p,1e-12,1-1e-12);records.append(pd.DataFrame(dict(fold=fold['fold'],model=name,horizon_minutes=minutes,index=target,hour=data['t'][target],p_tb=p,y=y,score=y*np.log(p)+(1-y)*np.log1p(-p))))
    table=pd.concat(records,ignore_index=True);table.to_csv(OUT/'predictions.csv.gz',index=False);scores=table.groupby(['fold','model','horizon_minutes']).agg(n_events=('index','size'),loglik=('score','sum')).reset_index();scores.to_csv(OUT/'scores.csv',index=False)
    # Pair within each horizon and use chronological 6h blocks, retaining all events.
    intervals=[];rng=np.random.default_rng(516016)
    for horizon,g in table.groupby('horizon_minutes'):
        base=g[g.model=='constant']
        for model,m in g.groupby('model'):
            if model=='constant':continue
            d=m.merge(base,on=['fold','index'],suffixes=('_m','_b'),validate='one_to_one');assert len(d)==len(m)==len(base);d['delta']=d.score_m-d.score_b;d['block']=np.floor(d.hour_m/6).astype(int);num=np.zeros(4000);den=np.zeros(4000)
            for _,part in d.groupby('fold'):
                b=part.groupby('block').agg(delta=('delta','sum'),n=('index','size'));ix=rng.integers(0,len(b),(4000,len(b)));num+=b.delta.to_numpy()[ix].sum(1);den+=b.n.to_numpy()[ix].sum(1)
            q=np.quantile(num/den,[.025,.975]);intervals.append(dict(model=model,horizon_minutes=horizon,mean_gain=d.delta.mean(),lower=q[0],upper=q[1],n_events=len(d)))
    pd.DataFrame(intervals).to_csv(OUT/'block_uncertainty.csv',index=False);write_json(OUT/'contract.json',dict(status='COMPLETE',models='No direct previous-label emission term: single OU, free two-OU (ADF), Brownian+fast OU without common drift',horizons_minutes=[0,1,5,15,60,120],forecast='Future observed-event label, conditional on no intervening clinical seizure; no intervening labels or event times consumed',reason_history_models_excluded='Their future history term depends on unobserved intervening events; setting it to zero would not marginalize the fitted model',scope='Development-data forward mode forecast, not event-time or seizure prediction'))
    print(pd.DataFrame(intervals).to_string(index=False))
if __name__=='__main__':main()
