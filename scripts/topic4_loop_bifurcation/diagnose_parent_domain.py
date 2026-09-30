#!/usr/bin/env python3
"""Diagnose known validation failure; explicitly not a fresh acceptance set."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time
import numpy as np
import torch
from numba import set_num_threads
from campaign import ROOT,read,write,sha
from conductance_static_v2 import Response
from audit_topic4_loop_conductance_response import monte_carlo
from calibrate_topic4_loop_conductance_static import base_features


def main():
    source=ROOT/'conductance_static_v2';out=ROOT/'parent_domain_diagnostic'
    out.mkdir(exist_ok=True);assert not (out/'contract.json').exists()
    with np.load(source/'validation_predictions_locked.npz') as d:
        pars=np.repeat(d['pars'][270:271],6,axis=0)
        g=np.array([0.,.01,.03,float(d['g'][270]),.1,.3])
    f,b=base_features(pars,g);model=Response().double()
    model.load_state_dict(torch.load(source/'locked_model.pt',map_location='cpu',weights_only=False)['model'])
    with torch.no_grad():pred=model(torch.tensor(f),torch.tensor(b)).numpy()
    write(out/'contract.json',dict(status='KNOWN_FAILURE_DIAGNOSIS_NOT_ACCEPTANCE',
        question='Is v2 failure270 inherited from the g0 parent outside its static noise-table support?',
        source_index=270,source_sha256=sha(__file__),new_MC_seed=927294,
        source_inputs=dict(x=float((pars[0,0]-11)/7),sigmaE=float(np.sqrt(pars[0,2])/7),sigmaI=float(np.sqrt(pars[0,3])/7)),
        parent_table_support=dict(sigmaE_max=6.5,sigmaI_max=8.5),g=g.tolist(),
        prediction_Hz=pred.tolist(),replicates=4096,record_ms=2000,burn_ms=500,
        interpretation='Known test point selected for error diagnosis. This is development evidence, not independent validation of a new model. No fitted parameter changes.'))
    start=time.time();set_num_threads(24)
    counts=monte_carlo(pars,g,4096,20000,5000,927294);rates=counts/2.
    means=rates.mean(1);sem=rates.std(1,ddof=1)/np.sqrt(4096)
    np.savez_compressed(out/'diagnostic.npz',pars=pars,g=g,counts=counts,prediction_Hz=pred)
    write(out/'result.json',dict(status='COMPLETE',g=g.tolist(),measured_Hz=means.tolist(),
        SEM_Hz=sem.tolist(),prediction_Hz=pred.tolist(),error_Hz=(pred-means).tolist(),
        paired_change_from_g0_Hz=(rates-rates[:1]).mean(1).tolist(),
        elapsed_s=time.time()-start,acceptance_claim=False))
    print(read(out/'result.json'),flush=True)


if __name__=='__main__':main()
