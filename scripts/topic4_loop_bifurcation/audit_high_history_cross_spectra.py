#!/usr/bin/env python3
"""Measured source cross spectra, with original delays, at the high anchor.

This is a read-only diagnosis. Native source phases are not supplied to a new
simulation or used as a replacement autonomous closure.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import shutil,time
import numpy as np
from scipy import fft,sparse
from campaign import ROOT,read,write,sha
from high_history_spectral_value import OUT as SOURCE,N
from conditional_density_inputs import OPS
import run_topic4_loop_zk_conditional as native

OUT=SOURCE/'cross_spectral_diagnostic'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_READONLY_CROSS_SPECTRAL_AUDIT',created_epoch=time.time(),
        question='Does omission of measured source cross spectra explain the input variance missing in the second-history diagonal approximation?',
        design='Reuse the72-74s complete native source spike trains, original40000-cell graph and actual delay bins. At the union of39 preobserved targets and20 largest first-response E rate discrepancies, project complex source Fourier amplitudes with each edge delay, then filter with original synapses. Compare full source cross terms with squared-weight diagonal terms on the identical2s circular convention.',
        QA='Sourceperiodograms match frozen inputPSDs; originalgraphidentity; full-power Parseval; at preobservedtargets reconstruct pure recurrentII waveform after0.3s boundary burn. ForIE decompose observedtotal into reconstructedrecurrent and residualexternal, preserving their covariance.',
        interpretation='Descriptive finite-window source-correlation decomposition. Selected20 cells are error diagnostics, not an independent acceptance set. Supplying measured source phases does not define a self-generated recurrent model or certify a bifurcation. No newnative simulation or physical change.',
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    started=time.time();observed=ROOT/'native_K9p35_high_history_source_spectra'
    traces=dict(np.load(observed/'target_traces.npz'));top=np.load(SOURCE/'input_moment_review/moments.npz')['largest_discrepancy_cells']
    cells=np.unique(np.r_[traces['cells'],top]);raw=np.load(SOURCE/'parameters.npz')
    p=read(OPS/'prepared.json')['params'];rates=np.load(SOURCE/'generation_0/source_rate_Hz.npy')
    sp=sparse.load_npz(observed/'source_spikes_0p1ms.npz').tocsr()
    reference=np.load(SOURCE/'generation_0/source_PSD.npy',mmap_mode='r')
    F=np.empty((40000,N//2+1),dtype='c16');source_error=0.
    for lo in range(0,40000,128):
        x=sp[lo:lo+128].toarray().astype(float);x-=x.mean(1,keepdims=True)
        f=fft.rfft(x,axis=1,workers=2);f[:,0]=0;F[lo:lo+len(x)]=f
        source_error=max(source_error,float(abs(abs(f)**2-reference[lo:lo+len(x)]).max()))
    assert source_error<1e-7
    write(OUT/'progress.json',dict(status='BUILDING_ORIGINAL_DELAY_GRAPH',pid=os.getpid(),updated_epoch=time.time()))
    sim,_,_,identity=native.base.old.setup(9108405)
    assert identity==read(OPS/'prepared.json')['graph_identity']
    phase=np.exp(-2j*np.pi*np.arange(N//2+1)/N)
    weight=np.full(N//2+1,2.);weight[[0,-1]]=1
    signals=[];fullvar=[];diagvar=[];means=[];parseval=[]
    for q,(kind,ss) in enumerate([('ampa',slice(0,32000)),('gaba',slice(32000,40000))]):
        matrices=sim.net[kind+'_by_delay'];assert matrices[0].nnz==0
        C=np.zeros((len(cells),N//2+1),dtype='c16')
        for d,W in enumerate(matrices):
            W=W.tocsr()[cells]
            if W.nnz:C+=(W@F[ss])*phase[None,:]**d
        ar=np.exp(-.1/p['tau_r_'+kind.upper()]);ad=np.exp(-.1/p['tau_d_'+kind.upper()])
        H=(1-ad)/((1-ar*phase)*(1-ad*phase));C*=H
        W=sparse.load_npz(SOURCE/f'original_{kind}_jump.npz').tocsr()[cells]
        sourcevar=np.asarray(reference[ss])@(abs(H)**2*weight/N**2)
        diagonal=W.multiply(W)@sourcevar
        variance=abs(C)**2@weight/N**2
        mean=W@(rates[ss]*.0001)/(1-ar)
        signal=fft.irfft(C,n=N,axis=1,workers=2)+mean[:,None]
        error=float(abs(signal.var(1)-variance).max());assert error<1e-8
        signals.append(signal);fullvar.append(variance);diagvar.append(diagonal);means.append(mean);parseval.append(error)
        write(OUT/'progress.json',dict(status='PROJECTED_'+kind.upper(),targets=len(cells),updated_epoch=time.time()))
    signal=np.array(signals);fullvar=np.array(fullvar);diagvar=np.array(diagvar)
    index=np.searchsorted(cells,traces['cells']);cut=3000
    II=signal[1,index,cut:];nativeII=traces['IE_II_V_M'][cut:,1,:].T
    waveform_error=float(abs(II-nativeII).max())
    # Exact agreement here also checks the sign and indexing of physical delays.
    assert waveform_error<1e-8, waveform_error
    rec=signal[0,index,cut:];total=traces['IE_II_V_M'][cut:,0,:].T
    ext=total-rec
    recvar=rec.var(1);extvar=ext.var(1);cross=np.mean((rec-rec.mean(1,keepdims=True))*(ext-ext.mean(1,keepdims=True)),axis=1)
    budget_error=float(abs(total.var(1)-(recvar+extvar+2*cross)).max());assert budget_error<1e-8
    moments=np.load(observed/'cell_statistics.npz')['per_cell_mean_moments'].mean(0)
    observedvar=np.array([moments[2]-moments[0]**2,moments[3]-moments[1]**2])[:,cells]
    ext_expected=raw['nu_per_ms'][cells]*.1*raw['jump_external'][cells]**2*float(np.load(SOURCE/'filter_power.npy')[0]@weight/N)
    region=raw['region'][cells];rows=[]
    for label,mask in [('selected_all',np.ones(len(cells),bool)),('selected_coreA',(cells<32000)&(region==0)),
                       ('selected_coreB',(cells<32000)&(region==1)),('largest20_errors',np.isin(cells,top))]:
        if not mask.any():continue
        rows.append(dict(region=label,targets=int(mask.sum()),
            diagonal_recurrent_IE_II_variance_mV2=diagvar[:,mask].mean(1).tolist(),
            full_cross_recurrent_IE_II_variance_mV2=fullvar[:,mask].mean(1).tolist(),
            cross_source_contribution_IE_II_mV2=(fullvar-diagvar)[:,mask].mean(1).tolist(),
            expected_external_IE_variance_mV2=float(ext_expected[mask].mean()),
            observed_total_IE_II_variance_mV2=observedvar[:,mask].mean(1).tolist()))
    np.savez_compressed(OUT/'projection.npz',cells=cells,full_recurrent_variance=fullvar,
        diagonal_recurrent_variance=diagvar,observed_total_variance=observedvar,
        reconstructed_recurrent_currents=signal,preobserved_cells=traces['cells'],
        trimmed_recurrent_IE_variance=recvar,trimmed_external_IE_variance=extvar,
        trimmed_recurrent_external_covariance=cross,trimmed_total_IE_variance=total.var(1))
    result=dict(status='COMPLETE_MEASURED_SOURCE_CROSS_SPECTRAL_DIAGNOSIS',rows=rows,
        QA=dict(source_PSD_max_abs_error=source_error,original_graph_identity=True,
                Parseval_max_abs_error=parseval,pure_II_waveform_after_0p3s_max_abs_error=waveform_error,
                IE_recurrent_external_variance_budget_error=budget_error),
        reference='Native72-74s, circular2s Fourier reconstruction; waveformQA trims first0.3s so delayed source and synapse initial conditions no longer affect the record. Mean selected moments use full2s and retain boundary effects.',
        scope='Measured source correlations describe the missing input structure. No endogenous cross-spectral closure, formal stability or bifurcation established; no data-forced simulation is promoted to a predictive model.',
        elapsed_s=time.time()-started,producer_sha256=sha(__file__))
    write(OUT/'result.json',result);shutil.copy2(__file__,OUT/'producer.py')
    write(OUT/'progress.json',dict(status='COMPLETE',updated_epoch=time.time()));print(result,flush=True)


if __name__=='__main__':main()
