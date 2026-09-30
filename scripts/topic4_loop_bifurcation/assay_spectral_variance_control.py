#!/usr/bin/env python3
"""Separate total variance/E-I covariance from prescribed spectral shape."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from campaign import read,write,sha
from observe_source_aggregation import OUT as SOURCE
from observe_source_time_structure import OUT as TEMPORAL
from phase_lif_mc import run

OUT=TEMPORAL/'prescribed_spectrum_response/total_variance_control'

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    spectrum=read(TEMPORAL/'prescribed_spectrum_response/result.json')
    local=read(SOURCE/'local_response_factorial_exact_thresholds/result.json')
    inp=dict(np.load(SOURCE/'local_response_factorial_exact_thresholds/inputs.npz'))
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_VARIANCE_CONTROL',
        question='Does remaining improvement with a full netcurrent spectrum come only from its different totalvariance (including EIcovariance), or require spectralshape?',
        design='Same six spectralassay targets, same originalthresholds/mean/M/Z/K. Retain earlier independentAMPA/GABA filtershapes but scale both channelvariances by the one analyticallydetermined ratio needed to equal observedNETcurrentvariance, including EIcovariance. This is a prescribedvariance diagnostic, never a fitted autonomousnoise gain.1024replicas,16sphaseawarecounts,seed929501.',
        limits='Comparison isolates spectralshape versus totalvariance only within Gaussian prescribedinput assays; finite2snativewindow andperiodic spectrum remain. Not coupledvalidation.',
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    pars=[];rows=[]
    for r in spectrum['rows']:
        cell=r['cell'];i=int(np.flatnonzero(inp['cells']==cell)[0]);old=local['rows'][4*i+3];p=inp['pars'][4*i+3].copy()
        variance=np.array(old['raw_IE_II_variance']);h=1+old['conductance'];Z=old['held_Z']
        original=(variance[0]+Z*Z*variance[1])/h**2
        desired=r['expected_effective_current_variance'];scale=desired/original
        p[2:4]*=scale;p[11:17]*=np.sqrt(scale);pars.append(p)
        rows.append(dict(cell=cell,native_rate_Hz=r['native_rate_Hz'],
            original_twochannel_effective_variance=float(original),matched_net_variance=float(desired),
            prescribed_ratio=float(scale),unmatched_lowpass_rate_Hz=r['measured_variance_lowpass_Gaussian_rate_Hz'],
            spectrum_rate_Hz=r['measured_spectrum_Gaussian_rate_Hz'],spectrum_MC_SEM_Hz=r['MC_SEM_Hz']))
    pars=np.array(pars);a=run(pars,1024,16000,1000,929501,device=1,phase_ms=1000,stream_mode=1)
    assert not a[:,:,3].any();rates=a[:,:,2]/16
    for i,r in enumerate(rows):r.update(variance_matched_lowpass_rate_Hz=float(rates[i].mean()),MC_SEM_Hz=float(rates[i].std(ddof=1)/32))
    np.savez_compressed(OUT/'counts.npz',counts=a[:,:,2].astype('i4'),pars=pars)
    write(OUT/'result.json',dict(status='COMPLETE_TOTAL_VARIANCE_CONTROL',rows=rows,data_prescribed_not_autonomous=True,formal_bifurcation_allowed=False,producer_sha256=sha(__file__)))
    print(rows,flush=True)

if __name__=='__main__':main()
