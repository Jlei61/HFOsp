#!/usr/bin/env python3
"""Descriptive input-moment discrepancy at the second observed history."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import shutil
import numpy as np
from scipy import sparse
from campaign import ROOT, read, write, sha
from high_history_spectral_value import OUT as SOURCE, N
from conditional_density_inputs import OPS


def main():
    out=SOURCE/'input_moment_review';out.mkdir(exist_ok=True)
    raw=dict(np.load(SOURCE/'parameters.npz'));p=read(OPS/'prepared.json')['params']
    native=dict(np.load(ROOT/'native_K9p35_high_history_source_spectra/cell_statistics.npz'))
    moments=native['per_cell_mean_moments'].mean(0)
    rates=np.load(SOURCE/'generation_0/source_rate_Hz.npy')
    response=np.load(SOURCE/'generation_1/source_rate_Hz.npy')
    spec=np.load(SOURCE/'generation_0/source_PSD.npy',mmap_mode='r')
    H=np.load(SOURCE/'filter_power.npy');w=np.full(N//2+1,2.);w[[0,-1]]=1.
    sourcevar=np.zeros((2,40000))
    for lo in range(0,40000,128):
        for q in range(2):sourcevar[q,lo:lo+128]=np.asarray(spec[lo:lo+128])@(H[q]*w/N**2)
    predicted=[];mean=[]
    for q,(kind,ss) in enumerate([('ampa',slice(0,32000)),('gaba',slice(32000,40000))]):
        W=sparse.load_npz(SOURCE/f'original_{kind}_jump.npz')
        mean.append(W@(rates[ss]*.0001)/(1-np.exp(-.1/p['tau_r_'+kind.upper()])))
        predicted.append(W.multiply(W)@sourcevar[q,ss])
    # Same finite-period Fourier convention as the supplied source assay.
    # External Poisson is white and has a DC fluctuation component, unlike the
    # demeaned source records, so its transfer sum retains the DC frequency.
    extvar=raw['nu_per_ms']*.1*raw['jump_external']**2*float(H[0]@w/N)
    extmean=raw['jump_external']*raw['nu_per_ms']*.1/(1-np.exp(-.1/p['tau_r_AMPA']))
    predicted=np.array(predicted);mean=np.array(mean)
    predicted[0]+=extvar;mean[0]+=extmean
    actual=np.array([moments[2]-moments[0]**2,moments[3]-moments[1]**2])
    covariance=moments[4]-moments[0]*moments[1]
    net_actual=actual[0]+raw['Z']**2*actual[1]-2*raw['Z']*covariance
    net_independent=predicted[0]+raw['Z']**2*predicted[1]
    rows=[];E=np.arange(40000)<32000
    masks=[('allE',E),('coreA',E&(raw['region']==0)),('coreB',E&(raw['region']==1)),('I',~E)]
    top=np.argsort(abs(response[:32000]-rates[:32000]))[-20:]
    masks.append(('largest20_individual_E_rate_discrepancies',np.isin(np.arange(40000),top)))
    for label,mask in masks:
        rows.append(dict(region=label,targets=int(mask.sum()),
            predicted_IE_II_variance_mean_mV2=predicted[:,mask].mean(1).tolist(),
            native_IE_II_variance_mean_mV2=actual[:,mask].mean(1).tolist(),
            native_IE_II_covariance_mean_mV2=float(covariance[mask].mean()),
            native_E_minus_ZI_variance_mean_mV2=float(net_actual[mask].mean()),
            independent_E_minus_ZI_variance_mean_mV2=float(net_independent[mask].mean()),
            predicted_minus_native_IE_II_mean_RMS_mV=np.sqrt(np.mean((mean[:,mask]-moments[:2,mask])**2,axis=1)).tolist(),
            response_minus_native_rate_RMS_Hz=float(np.sqrt(np.mean((response[mask]-rates[mask])**2)))))
    np.savez_compressed(out/'moments.npz',predicted_mean=mean,predicted_variance=predicted,
        native_variance=actual,native_covariance=covariance,native_net_variance=net_actual,
        independent_net_variance=net_independent,largest_discrepancy_cells=top)
    result=dict(status='COMPLETE_SECOND_HISTORY_MOMENT_DESCRIPTION',rows=rows,
        protocol='Native72-74s variable-background moments versus native-source diagonal2s periodic spectral prediction with fixed expected external mean. Finite-window filter boundary and background differences remain; not a matched intervention.',
        interpretation='Moment differences describe remaining approximation error, not a causal allocation to cross-source or E/I covariance. The measured native E/I covariance is retained explicitly; changing it also changes the joint temporal statistics. No noise multiplier fit or model modification.',
        scope='No new simulation, fixedpoint or stability claim. Original second-history response stays unchanged.',producer_sha256=sha(__file__))
    write(out/'result.json',result);shutil.copy2(__file__,out/'producer.py');print(result,flush=True)


if __name__=='__main__':main()
