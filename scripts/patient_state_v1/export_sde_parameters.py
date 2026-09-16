"""Export OU restoring and diffusion coefficients from accepted joint samples.

Transform every weighted draw; do not transform interval endpoints as though
time and amplitude were independent. These units belong to the logistic state.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from scipy.special import logsumexp
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.review_importance_posterior import quantile
OUT=RUN/'model_parameter_sheet'

def main():
    audit=json.loads((RUN/'independent_importance_posterior_v1_28/scientific_audit.json').read_text());rows=[]
    for model in ['ou','ou_history']:
        info=next(r for r in audit['results'] if r['model']==model and r['target']=='particle');assert info['status']=='IMPORTANCE_DIAGNOSTICS_PASS';z=np.load(info['posterior_samples']);t=z['theta'];lw=z['particle_logweight'];w=np.exp(lw-logsumexp(lw));values=dict(baseline_log_odds=(t[:,0],'TB log-odds'),tau_minutes=(60*np.exp(t[:,-2]),'minutes'),restoring_coefficient=(np.exp(-t[:,-2]),'1/hour'),stationary_sd=(np.exp(t[:,-1]),'TB log-odds'),diffusion_coefficient=(np.exp(t[:,-1])*np.sqrt(2*np.exp(-t[:,-2])),'TB log-odds/sqrt(hour)'))
        if model=='ou_history':values['one_second_history_coefficient']=(t[:,1],'TB log-odds per history unit')
        for parameter,(x,unit) in values.items():
            lo,med,hi=quantile(x,w,[.025,.5,.975]);rows.append(dict(model=model,parameter=parameter,median=med,lower=lo,upper=hi,units=unit))
    OUT.mkdir(parents=True,exist_ok=True);df=pd.DataFrame(rows);df.to_csv(OUT/'sde_parameters.csv',index=False)
    write_json(OUT/'contract.json',dict(status='COMPLETE',equation='ds=kappa*(b-s)dt+q dW_t; P(TB_i|s_i,h_i)=sigmoid(s_i+gamma*h_i)',time_unit='hours',state_unit='TB log-odds; observation slope fixed at1',definitions='kappa=1/tau; q=stationary_sd*sqrt(2/tau); h_i=(previous_label-.5)*exp(-delta_seconds/1second), zero at new clinical interval',prior='Same accepted particle importance targets; intervals preserve joint parameter dependence by transforming each weighted draw',scope='Model-conditional statistical coefficients. Not an inferred voltage, E/I ratio, long-range conductance, or SNN input amplitude. IED observations do not reset the physical state.',correspondence='Code residual x=s-b has dx=-kappa*xdt+q dW. Positive x raises TB relative to baseline; s>0 is the50% slow-propensity threshold only when history=0.'))
    print(df.to_string(index=False))

if __name__=='__main__':main()
