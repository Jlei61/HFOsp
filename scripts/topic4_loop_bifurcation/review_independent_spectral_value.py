#!/usr/bin/env python3
"""Interpret the independent value at its actual fixed input, retaining failures."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import shutil
import numpy as np
from scipy import sparse
from campaign import ROOT,read,write,sha
from validate_individual_spectral_candidate import OUT,SOURCE
from conditional_density_inputs import OPS


def main():
    d=read(OUT/'result.json');assert d['status']=='COMPLETE_INDEPENDENT_FINITE_WINDOW_VALUE'
    raw=dict(np.load(OUT/'parameters.npz'));r=np.load(OUT/'generation_1/source_rate_Hz.npy')
    native=np.load(SOURCE/'generation_0/source_rate_Hz.npy');res=dict(np.load(OUT/'independent_residuals.npz'))
    filtered_sem=np.load(OUT/'generation_1/source_filtered_variance_SEM.npy')
    p=read(OPS/'prepared.json')['params'];means=[];variances=[];sems=[]
    for q,(kind,ix) in enumerate([('ampa',slice(0,32000)),('gaba',slice(32000,40000))]):
        W=sparse.load_npz(SOURCE/f'original_{kind}_jump.npz');Q=W.multiply(W)
        means.append(W@(res['rate_residual_Hz'][ix]*.0001)/(1-np.exp(-.1/p['tau_r_'+kind.upper()])))
        variances.append(Q@res['filtered_variance_residual'][q,ix])
        sems.append(np.sqrt(Q.multiply(Q)@(filtered_sem[q,ix]**2)))
    means,variances,sems=np.array(means),np.array(variances),np.array(sems)
    display=raw['display'][:32000];counts=np.bincount(display,minlength=400)
    field=lambda v:np.bincount(display,weights=v[:32000],minlength=400)/np.maximum(counts,1)
    error=field(r)-field(native);E=np.arange(40000)<32000;rows=[]
    for name,mask in [('allE',E),('coreA',E&(raw['region']==0)),('coreB',E&(raw['region']==1)),('surroundE',E&(raw['region']==2)),('I',~E)]:
        rows.append(dict(region=name,native_development_rate_Hz=float(native[mask].mean()),
            independent_output_rate_Hz=float(r[mask].mean()),
            projected_recurrent_IE_II_mean_residual_RMS_mV=np.sqrt(np.mean(means[:,mask]**2,axis=1)).tolist(),
            projected_recurrent_IE_II_variance_residual_RMS_mV2=np.sqrt(np.mean(variances[:,mask]**2,axis=1)).tolist(),
            projected_variance_output_MCSEM_RMS_mV2=np.sqrt(np.mean(sems[:,mask]**2,axis=1)).tolist()))
    np.savez_compressed(OUT/'projected_residuals.npz',recurrent_mean_residual=means,
        recurrent_variance_residual=variances,recurrent_variance_output_MCSEM=sems,independent_E_field_Hz=field(r))
    result=dict(status='COMPLETE_INDEPENDENT_REVIEW',rows=rows,
        actual_native_development_field_RMS_Hz=float(np.sqrt(np.average(error**2,weights=counts))),
        counterpart='The base collect uses candidate X12 as generation0, so its native_reference labels do not denote a native trajectory. This review explicitly uses originalnative42-44s for the development field comparison.',
        uncertainty='Projection SEM uses independent target/replica output estimators conditional on the fixed supplied X12. It excludes numerical uncertainty of X12, source crosscorrelations omitted by the approximation, finite-window bias and native variability.',
        judgement='Do not accept a certified root at the fresh-output sampling precision: E/I rates and filteredsourcevariances have residuals beyond that SEM, with6SEM outliers and zeroSEM nonzero residuals. This does not by itself establish structural closure failure: X12 was estimated from16 common-stream replicas and remains a noisy approximate candidate. Native same-background controls are required separately.',
        formal_bifurcation_allowed=False,root_certified=False,producer_sha256=sha(__file__))
    write(OUT/'review.json',result);shutil.copy2(__file__,OUT/'review_producer.py');print(result,flush=True)


if __name__=='__main__':main()
