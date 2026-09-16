"""Reuse unbiased importance weights to vary state-amplitude regularization."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.review_importance_posterior import summarize
ROOT=RUN/'state_amplitude_prior_sensitivity_v1_30'

def main():
    base=RUN/'independent_importance_posterior_v1_28';audit=json.loads((base/'scientific_audit.json').read_text());rows=[];reports=[];pending=[]
    for model in ['ou','ou_history']:
        accepted=next(r for r in audit['results'] if r['model']==model and r['target']=='particle')
        if accepted['status']!='IMPORTANCE_DIAGNOSTICS_PASS':pending.append(model);continue
        z=np.load(base/model/'posterior_samples.npz');t=z['theta'];names=json.loads((base/model/'contract.json').read_text())['source']['parameter_names'];physical=t.copy();physical[:,-2]=60*np.exp(t[:,-2]);physical[:,-1]=np.exp(t[:,-1]);pnames=names[:-2]+['tau_minutes','stationary_sd']
        for scale in [.5,1.,2.]:
            lw=z['particle_logweight']-.5*np.exp(2*t[:,-1])*(1/scale**2-1);r,_,_=summarize(t,lw,names);p,_,_=summarize(physical,lw,pnames);means=[];errs=[];rep_ess=[]
            for rep in np.unique(z['replicate']):
                ok=z['replicate']==rep;rr,_,_=summarize(t[ok],lw[ok],names);rep_ess.append(rr['importance_ess']);means.append([s['mean'] for s in rr['summary']]);errs.append([s['mcse'] for s in rr['summary']])
            means=np.array(means);errs=np.array(errs);zmax=0.
            for i in range(len(means)):
                for j in range(i):zmax=max(zmax,float(np.max(abs(means[i]-means[j])/np.sqrt(errs[i]**2+errs[j]**2))))
            good=bool(r['importance_ess']>=1000 and r['maximum_weight']<=.01 and r['pareto_k']<.7 and min(rep_ess)>=100 and max(s['mcse_over_sd'] for s in r['summary'])<.05 and zmax<4);reports.append(dict(model=model,amplitude_prior_scale=scale,status='DIAGNOSTICS_PASS' if good else 'DIAGNOSTICS_NOT_PASSED',importance_ess=r['importance_ess'],max_weight=r['maximum_weight'],pareto_k=r['pareto_k'],minimum_replicate_ess=min(rep_ess),max_pairwise_rep_z=zmax))
            for item in p['summary']:rows.append(dict(model=model,amplitude_prior_scale=scale,accepted_numerical=good,**item))
    ROOT.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(ROOT/'parameter_intervals.csv',index=False);write_json(ROOT/'scientific_audit.json',dict(status='COMPLETE' if not pending else 'PARTIAL_PENDING_BASE_IMPORTANCE',models_pending=pending,targets='Stationary state SD half-normal scale0.5,1,2 with same [.01,5] truncation; all other priors unchanged',method='Reweight each unbiased particle likelihood by new amplitude prior / original proposal; prior normalizing constants cancel within each target, no evidence comparison made',diagnostics=reports,scope='Sensitivity of model-conditional parameter uncertainty to amplitude shrinkage; no extra mode or per-event initialization parameters',limits='Robustness to this prior change is not model adequacy or physiological identifiability. Targets failing the same importance diagnostics are not accepted estimates.'))
    print(pd.DataFrame(rows).query("parameter in ['tau_minutes','stationary_sd']").to_string(index=False));print(json.dumps(reports,indent=2))

if __name__=='__main__':main()
