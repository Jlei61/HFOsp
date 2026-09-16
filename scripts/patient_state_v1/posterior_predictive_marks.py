"""Propagate accepted numerical parameter uncertainty into conditional marks.

Observed event times/exposure stay fixed and cannot be counted as generated
success. Each replicate has a posterior parameter draw and an independent new
latent path/label sequence. This is a model check, not held-out validation.
"""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.special import logsumexp
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
import scripts.patient_state_v1.conditional_mark_generation as conditional
OUT=RUN/'posterior_predictive_marks_v1_29'

def main():
    audit=json.loads((RUN/'independent_importance_posterior_v1_28/scientific_audit.json').read_text());jobs=[];accepted=[];pending=[];sources={}
    for model in ['ou','ou_history']:
        matches=[r for r in audit['results'] if r['model']==model and r['target']=='particle']
        if not matches or matches[0]['status']!='IMPORTANCE_DIAGNOSTICS_PASS':pending.append(model);continue
        accepted.append(model);path=RUN/'independent_importance_posterior_v1_28'/model/'posterior_samples.npz';z=np.load(path);lw=z['particle_logweight'];w=np.exp(lw-logsumexp(lw));rng=np.random.default_rng(1012900+(1000 if model=='ou_history' else 0));indices=rng.choice(len(w),512,p=w);sources[model]=dict(path=str(path),diagnostics=matches[0])
        for rep,index in enumerate(indices):
            t=z['theta'][index];parameters=[float(t[0]),float(t[1]) if model=='ou_history' else 0.,float(np.exp(t[-2])),float(np.exp(t[-1])),1.,0.];jobs.append(dict(model=model,rep=rep,seed=1013000+(1000 if model=='ou_history' else 0)+rep,parameters=parameters,posterior_sample_index=int(index)))
    if not accepted:write_json(OUT/'status.json',dict(status='PENDING_ACCEPTED_IMPORTANCE',models_pending=pending));return
    write_json(OUT/'contract.json',dict(question='Do mode-sequence discrepancies survive posterior parameter uncertainty?',n_replicates_per_model=512,models=accepted,sources=sources,parameters='Independent draws from raw particle-importance empirical posterior, with replacement; each gets independent new process noise and labels',conditioned='Exact patient event times, exposure, and clinical exclusion boundaries are fixed inputs',generated='Mode labels only; rate/count/interval quantities are excluded from success metrics',scope='Full-data posterior predictive model check, not new held-out evidence; no SNN or Z/M changes',pending_models=pending));write_json(OUT/'queue.json',jobs);conditional.OUT=OUT
    with ProcessPoolExecutor(max_workers=24) as ex:
        for i,f in enumerate(as_completed([ex.submit(conditional.worker,j) for j in jobs])):
            r=f.result();assert r['status']=='COMPLETE'
            if (i+1)%32==0:print(json.dumps(dict(done=i+1,total=len(jobs),model=r['job']['model'],rep=r['job']['rep'])),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE' if not pending else 'PARTIAL_PENDING_IMPORTANCE',models=accepted,n_runs=len(jobs),models_pending=pending,finished_unix=time.time()))

if __name__=='__main__':main()
