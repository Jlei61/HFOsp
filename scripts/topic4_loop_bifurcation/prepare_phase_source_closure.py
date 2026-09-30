#!/usr/bin/env python3
"""Construct a source-level phase operator before testing any free closure."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import shutil,time
import numpy as np
from scipy import fft,sparse
from campaign import ROOT,read,write,sha
from conditional_density_inputs import OPS
from measure_coherent_phase_component import OUT as MEASURED,L
from high_history_spectral_value import OUT as SOURCE,N
import run_topic4_loop_zk_conditional as native

OUT=ROOT/'high_history_phase_source_closure'


def project_phase(folder,probability,params):
    """Periodic mean with original delay residues and original discrete filters."""
    waves=[];means=[]
    z=np.exp(-2j*np.pi*np.arange(L//2+1)/L)
    for kind,ss in [('ampa',slice(0,32000)),('gaba',slice(32000,40000))]:
        jumps=np.zeros((40000,L))
        for d in range(L):
            W=sparse.load_npz(folder/f'{kind}_delay_mod22_{d:02d}.npz')
            jumps+=W@np.roll(probability[ss],d,axis=1)
        ar=np.exp(-.1/params['tau_r_'+kind.upper()]);ad=np.exp(-.1/params['tau_d_'+kind.upper()])
        H=(1-ad)/((1-ar*z)*(1-ad*z))
        signal=fft.irfft(fft.rfft(jumps,axis=1,workers=2)*H,n=L,axis=1,workers=2)
        dc=signal.mean(1);means.append(dc);waves.append(signal-dc[:,None])
    return np.array(means),np.array(waves)


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_SOURCE_PHASE_OPERATOR',created_epoch=time.time(),
        question='Can measured coherent target currents be represented by source phase probabilities passed through the original spatial graph and delays, with a source-defined residual spectrum?',
        design='Exactly one22-step candidate from the measured454.5Hz peak. Build and retain originalgraph operators grouped by delay modulo22. Source conditionalphase probabilities and residual PSD come from the existing72-74s developmentrecord only. No parameter/period fitting, native replay, or free model run in this preparation.',
        next_gate='Compare this source-level projection to the completed59target current decomposition. Then test the local response with independent source residual spectra before any bounded allcell self-generated update.',
        limits='A fixedperiod is a candidate representation, not spontaneous frequency selection or a physical time simulation. Source residual cross correlations are omitted. A local fit or convergence of a statistical map cannot certify physical stability or bifurcation.',
        producer_sha256=sha(__file__),period_steps=L,formal_bifurcation_allowed=False))
    started=time.time();operators=OUT/'operators';operators.mkdir()
    write(OUT/'progress.json',dict(status='BUILDING_ORIGINAL_GRAPH_ONCE',pid=os.getpid(),updated_epoch=time.time()))
    sim,_,_,identity=native.base.old.setup(9108405)
    assert identity==read(OPS/'prepared.json')['graph_identity']
    graphqa={}
    for kind,nsource in [('ampa',32000),('gaba',8000)]:
        matrices=sim.net[kind+'_by_delay'];assert matrices[0].nnz==0
        total=sparse.csr_matrix((40000,nsource))
        for residue in range(L):
            W=sparse.csr_matrix((40000,nsource))
            for d in range(residue,len(matrices),L):W=W+matrices[d].tocsr()
            W.sort_indices();sparse.save_npz(operators/f'{kind}_delay_mod22_{residue:02d}.npz',W)
            total+=W
        original=sparse.load_npz(SOURCE/f'original_{kind}_jump.npz')
        diff=(total-original).tocsr();error=float(abs(diff.data).max()) if diff.nnz else 0.
        assert error<1e-12
        graphqa[kind]=dict(nnz=int(total.nnz),original_combined_weights_max_abs_error=error)
    del sim
    p=read(OPS/'prepared.json')['params']
    d=dict(np.load(MEASURED/'source_phase_statistics.npz'));prob=d['phase_spike_probability']
    mean,wave=project_phase(operators,prob,p)
    target=dict(np.load(MEASURED/'target_phase_inputs.npz'));cells=target['cells']
    error=wave[:,cells]-target['phase_wave']
    projectionqa=dict(wave_RMS_mV=np.sqrt(np.mean(error**2,axis=(1,2))).tolist(),
        wave_max_abs_mV=np.max(abs(error),axis=(1,2)).tolist(),
        measured_wave_RMS_mV=np.sqrt(np.mean(target['phase_wave']**2,axis=(1,2))).tolist(),
        mean_max_abs_mV=np.max(abs(mean[:,cells]-target['mean_recurrent']),axis=1).tolist(),
        caveat='Phase-conditioning a finite20000step circular record does not commute exactly with22step delays/filters because20000 is not divisible by22. Differences are measured, not silently set tozero.')
    generation=OUT/'generation_0';generation.mkdir()
    np.save(generation/'phase_spike_probability.npy',prob)
    np.save(generation/'source_rate_Hz.npy',prob.mean(1)*10000)
    np.save(generation/'current_phase_wave.npy',wave)
    np.save(generation/'current_mean.npy',mean)
    spikes=sparse.load_npz(ROOT/'native_K9p35_high_history_source_spectra/source_spikes_0p1ms.npz').tocsr()
    psd=np.lib.format.open_memmap(generation/'residual_PSD.npy',mode='w+',dtype='f8',shape=(40000,N//2+1))
    phase=np.arange(N)%L;weights=np.full(N//2+1,2.);weights[[0,-1]]=1;parseval=0.
    for lo in range(0,40000,128):
        hi=min(lo+128,40000);x=spikes[lo:hi].toarray().astype(float)-prob[lo:hi][:,phase]
        x-=x.mean(1,keepdims=True);P=abs(fft.rfft(x,axis=1,workers=2))**2;P[:,0]=0
        parseval=max(parseval,float(abs(P@weights/N**2-x.var(1)).max()));psd[lo:hi]=P
    psd.flush();assert parseval<1e-12
    for name in ['parameters.npz','filter_power.npy','original_ampa_jump.npz','original_gaba_jump.npz']:
        (OUT/name).symlink_to(SOURCE/name)
    result=dict(status='COMPLETE_SOURCE_PHASE_OPERATOR_PREPARATION',graph_identity=identity,
        graphQA=graphqa,source_residual_Parseval_error=parseval,phase_projection_QA=projectionqa,
        native_development_allE_rate_Hz=float((prob[:32000].mean(1)*10000).mean()),
        local_response_required=True,elapsed_s=time.time()-started,producer_sha256=sha(__file__))
    write(OUT/'result.json',result);shutil.copy2(__file__,OUT/'preparation_producer.py')
    write(OUT/'progress.json',dict(status='COMPLETE_PREPARATION_ONLY',updated_epoch=time.time()));print(result,flush=True)


if __name__=='__main__':main()
