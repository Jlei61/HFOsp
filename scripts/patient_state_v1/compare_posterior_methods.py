"""Cross-check numerical posterior summaries only for chains passing their own gate."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd,arviz as az
from scripts.patient_state_v1.common import RUN,write_json
out=RUN/'posterior_method_comparison';out.mkdir(exist_ok=True)
tab=pd.read_csv(RUN/'independent_importance_posterior_v1_28/parameter_intervals.csv');diagnostics=json.loads((RUN/'posterior_diagnostics/status.json').read_text());rows=[];skipped=[]
for r in diagnostics['results']:
    if r['model'] not in ['ou','ou_history']:continue
    name=r['model']
    if not r['acceptance_gate_pass']:skipped.append(dict(model=name,reason=r['status'],iteration=r['iteration']));continue
    folder=RUN/'particle_posterior_v1_2'/name;contract=json.loads((folder/'contract.json').read_text());z=np.load(folder/'checkpoint.npz');a=z['samples'][contract['warmup']+1:r['iteration']+1].transpose(1,0,2)
    for j,param in enumerate(contract['parameter_names']):
        values=a[:,:,j]
        if param=='log_tau_hours':param='tau_minutes';values=np.exp(values)*60
        if param=='log_stationary_sd':param='stationary_sd';values=np.exp(values)
        ir=tab[(tab.model==name)&(tab.target=='particle')&(tab.parameter==param)].iloc[0];mcse=np.asarray(az.mcse(values,method='mean')).item();mean=float(values.mean());q=np.quantile(values,[.025,.5,.975]);combined=np.hypot(mcse,ir.mcse)
        rows.append(dict(model=name,parameter=param,pmmh_mean=mean,pmmh_mcse=mcse,pmmh_lower=q[0],pmmh_median=q[1],pmmh_upper=q[2],importance_mean=ir['mean'],importance_mcse=ir.mcse,mean_difference=mean-ir['mean'],difference_over_combined_mcse=(mean-ir['mean'])/combined,chain_iteration=r['iteration']))
pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False);write_json(out/'scientific_audit.json',dict(status='COMPLETE',eligible_models=sorted(set(r['model'] for r in rows)),skipped_unconverged_chains=skipped,max_abs_mean_difference_over_combined_mcse=max(abs(r['difference_over_combined_mcse']) for r in rows),scope='Descriptive independent-computation check under same conditional model, not an added biological or generative acceptance gate. PMMH summaries transformed per draw, autocorrelation-aware MCSE recomputed for each transformed parameter. Failed chains excluded.'))
print(pd.DataFrame(rows).to_string(index=False))
