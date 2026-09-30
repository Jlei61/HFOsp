#!/usr/bin/env python3
"""One source-phase decomposition, chosen from the measured 454.5 Hz peak."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import shutil,time
import numpy as np
from scipy import fft,sparse
from campaign import ROOT,read,write,sha
from compare_local_correlated_inputs import OUT as LOCAL,SOURCE,N

OUT=ROOT/'high_history_coherent_phase_component'
L=22


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(LOCAL/'phase_only_followup_v2/result.json')['status']=='COMPLETE_FIXED_POWER_RANDOM_PHASE_RESPONSE'
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_PHASE_COMPONENT_DECOMPOSITION',created_epoch=time.time(),
        question='How much of the missing current structure is a repeatable22-step phase component, rather than stationary Gaussian variance?',
        selection='Largest cross-source IE variance spectral bin is454.5Hz in the measured2srecord. The proposed2.2ms phase period is22 original0.1ms steps, consistent with that finite-frequency bin. Fixed before this decomposition; no period search or fit to rate error.',
        design='Read-only59target recurrentcurrent projection and all40000 existing native source trains. Phase means at time-index mod22, phase-residual covariance and exact variance budget. Report phase explainability percell and perregion, including fraction remaining outside this one phase component.',
        limitations='One native highhistory2s window. A spectral peak near a timestep multiple is a model-level observation, not a physiological frequency or a certified periodic orbit. Period locking and timestep sensitivity remain separate questions. This decomposition supplies no newautonomous model.',
        period_steps=L,dt_ms=.1,producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    inp=dict(np.load(LOCAL/'inputs.npz'));cells=inp['cells']
    signals=np.load(SOURCE/'cross_spectral_diagnostic_v2/projection.npz')['reconstructed_recurrent_currents']
    phase=np.arange(N)%L;count=np.bincount(phase,minlength=L)
    profiles=np.stack([signals[:,:,phase==q].mean(2) for q in range(L)],axis=2)
    recurrent_coherent=profiles[:,:,phase]
    residual=signals-recurrent_coherent
    error=float(abs(signals.var(2)-recurrent_coherent.var(2)-residual.var(2)).max());assert error<1e-9
    conditional_residual=max(float(abs(residual[:,:,phase==q].mean(2)).max()) for q in range(L));assert conditional_residual<1e-10
    # The infinite repeated phase mean is centered to zero. Keep the original
    # supplied DC separately; report finite-record unequal-phase-count offset.
    cyclemean=profiles.mean(2);wave=profiles-cyclemean[:,:,None]
    C=fft.rfft(residual,axis=2,workers=2);C[:,:,0]=0
    np.savez_compressed(OUT/'target_phase_inputs.npz',cells=cells,phase_wave=wave,
        residual_complex=C,mean_recurrent=inp['mean_recurrent'],cycle_mean_offset=cyclemean-inp['mean_recurrent'],
        full_variance=signals.var(2),coherent_variance=recurrent_coherent.var(2),residual_variance=residual.var(2),
        phase_counts=count,native_rate_Hz=inp['native_rate_Hz'])
    sp=sparse.load_npz(ROOT/'native_K9p35_high_history_source_spectra/source_spikes_0p1ms.npz').tocsr()
    assign=sparse.csr_matrix((np.ones(N),(np.arange(N),phase)),shape=(N,L))
    conditional=(sp@assign).toarray()/count[None,:]
    mean=np.asarray(sp.sum(1)).ravel()/N;variance=mean*(1-mean)
    phasevar=np.average((conditional-mean[:,None])**2,axis=1,weights=count)
    fraction=np.divide(phasevar,variance,out=np.zeros_like(variance),where=variance>0)
    assert fraction.min()>=0 and fraction.max()<1+1e-12
    raw=np.load(SOURCE/'parameters.npz');region=raw['region'];E=np.arange(40000)<32000;active=mean>.0001
    rows=[]
    for label,mask in [('allE',E),('coreA',E&(region==0)),('coreB',E&(region==1)),('I',~E)]:
        m=mask&active
        rows.append(dict(region=label,active_targets=int(m.sum()),
            phase_explained_source_variance_median=float(np.median(fraction[m])),
            active_fraction_phase_explained_above_half=float(np.mean(fraction[m]>.5)),
            total_variance_weighted_phase_fraction=float(phasevar[m].sum()/variance[m].sum())))
    np.savez_compressed(OUT/'source_phase_statistics.npz',phase_spike_probability=conditional,
        variance_explained_fraction=fraction,source_mean_spikes_per_step=mean,phase_counts=count)
    selected=np.isin(cells,inp['largest_discrepancy_cells']);full=signals.var(2);coherent=recurrent_coherent.var(2)
    result=dict(status='COMPLETE_22_STEP_PHASE_COMPONENT_DESCRIPTION',source_rows=rows,
        selected20_IE_II_full_variance_mV2=full[:,selected].mean(1).tolist(),
        selected20_IE_II_coherent_variance_mV2=coherent[:,selected].mean(1).tolist(),
        selected20_IE_II_residual_variance_mV2=residual.var(2)[:,selected].mean(1).tolist(),
        source_phase_period_s=L*.0001,
        QA=dict(exact_variance_budget_max_error=error,conditional_residual_mean_max_error=conditional_residual,
            finite_record_to_uniform_phase_mean_offset_max_mV=float(abs(cyclemean-inp['mean_recurrent']).max())),
        limits=read(OUT/'contract.json')['limitations'],producer_sha256=sha(__file__))
    write(OUT/'result.json',result);shutil.copy2(__file__,OUT/'producer.py');print(result,flush=True)


if __name__=='__main__':main()
