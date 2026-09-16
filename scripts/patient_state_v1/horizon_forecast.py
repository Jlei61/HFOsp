"""Forecast future marks without assimilating any intervening events."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.special import expit
from numpy.polynomial.hermite import hermgauss
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import filter_adf
from scripts.patient_state_v1.analyze_first import best_fits,metrics

def main():
    data=dict(np.load(RUN/'observations.npz'));ev=pd.read_csv(RUN/'events.csv');iv=json.loads((RUN/'seizures.json').read_text());folds=json.loads((RUN/'splits.json').read_text());best=best_fits();origin=float(data['origin_epoch']);times=ev.start_epoch.to_numpy();ends=ev.end_epoch.to_numpy();offsets=np.array([r['offset'] for r in iv]);nodes,w=hermgauss(32);w/=np.sqrt(np.pi);rows=[];blocks=[]
    for f in folds:
        scope=f"fold{f['fold']}";train_end=ends[f['train_end']-1]
        for model in ('constant','cycle','ou','ou_cycle'):
            theta=np.array(best[scope,model]['theta']);nc=3 if 'cycle' in model else 1
            filt=filter_adf(theta,data,'cycle' in model) if model.startswith('ou') else None
            for minutes in (0,1,5,15,60):
                target=np.arange(f['test_start'],f['test_end']);cut=times[target]-minutes*60
                epochcut=np.searchsorted(offsets,cut,side='right');valid=(cut>=train_end)&(epochcut==data['epoch'][target]);target=target[valid];cut=cut[valid]
                previous=np.searchsorted(ends,cut,side='right')-1
                if filt is None:p=expit(data['x'][target,:nc]@theta[:nc])
                else:
                    tau,sd=np.exp(theta[-2:]);m=np.zeros(len(target));v=np.full(len(target),sd*sd);ok=previous>=0
                    ok[ok]&=data['epoch'][previous[ok]]==data['epoch'][target[ok]]
                    elapsed=(times[target[ok]]-times[previous[ok]])/3600;rho=np.exp(-elapsed/tau)
                    m[ok]=rho*filt['mean'][previous[ok]];v[ok]=rho*rho*filt['variance'][previous[ok]]+sd*sd*(-np.expm1(-2*elapsed/tau))
                    eta=data['x'][target,:nc]@theta[:nc];p=expit(eta[:,None]+m[:,None]+np.sqrt(2*v[:,None])*nodes)@w
                y=data['y'][target];n=data['n'][target];row=dict(fold=f['fold'],model=model,horizon_minutes=minutes,**metrics(p,y,n));rows.append(row)
                score=y*np.log(np.clip(p,1e-12,1))+(n-y)*np.log(np.clip(1-p,1e-12,1))
                for hour in np.unique(np.floor(data['t'][target])):
                    mask=np.floor(data['t'][target])==hour;blocks.append(dict(fold=f['fold'],model=model,horizon_minutes=minutes,hour=hour,n_events=n[mask].sum(),loglik=score[mask].sum()))
    pd.DataFrame(rows).to_csv(RUN/'horizon_forecasts.csv',index=False);pd.DataFrame(blocks).to_csv(RUN/'horizon_forecast_hour_blocks.csv',index=False)
    write_json(RUN/'horizon_forecast_audit.json',dict(status='COMPLETE',no_intervening_mark_updates=True,parameters_trained_before_forecast_origin=True,
         restrict_to_uninterrupted_interictal_epoch=True,forecast_is='future mark conditional on an event being observed; not event occurrence or seizure prediction',n_rows=len(rows)))
    print(pd.DataFrame(rows).groupby(['model','horizon_minutes'])[['n_events','loglik']].sum().to_string(),flush=True)

if __name__=='__main__':main()
