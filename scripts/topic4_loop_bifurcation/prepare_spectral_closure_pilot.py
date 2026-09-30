#!/usr/bin/env python3
"""Prepare one bounded, individual-source stationary spectral closure pilot."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import shutil, time
import numpy as np
from scipy import fft, sparse
from campaign import ROOT, read, write, sha
from conditional_density_inputs import OPS
from coupled_density_exit import ADAPTED
from observe_source_aggregation import OUT as OBS, INITIAL
from observe_source_time_structure import OUT as TEMPORAL
from native_target_thresholds import load
import run_topic4_loop_zk_conditional as native

OUT = ROOT / 'individual_source_spectral_pilot'
N = 20000
REPLICAS = 16
GENERATIONS = 3


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'contract.json').exists()
    write(OUT / 'contract.json', dict(status='REGISTERED_BEFORE_SELF_GENERATED_SPECTRA',
        created_epoch=time.time(), question='Can a diagonal-source Gaussian spectral approximation preserve the relevant asymmetric native state when its source means and spectra are updated from its own outputs?',
        scope='One held spatial Z/K point, meanZ0.21/K9.35, initialized from the completed asymmetric native42-44s source trains. Exactly three parallel-Jacobi generations, no relaxation, root solver or parameter sweep.',
        retained='Original40000 individual source/target identities, original synaptic weights and filters, actual individual thresholds, local dynamic native M. External independent Poisson shot noise has fixed per-cell expected rates averaged over native42-44s. This removes the native time-varying external modulation and is a distinct conditional protocol.',
        initialization='Native source means and demeaned2s autospectra initialize generation0 only. Generations1-3 read solely the preceding candidate output means and spectra for recurrent neural drive. Native recordings are retained only as development references.',
        protocol=dict(generations=GENERATIONS, replicas=REPLICAS, dt_ms=.1, record_steps=N,
            burn_steps=30000, additional_burn_uniform_steps=[0,10000], seed=929531,
            source_PSD='Mean |rFFT(x-mean(x))|^2, DC0; Nyquist has single real component.',
            recurrent_input='Independent E/I Gaussian Fourier coefficients after original squared-weight spectral projection. Two-second periodic recurrent signal; fresh external Poisson throughout each local assay.',
            M='Native Euler decay and post-spike increment, initial M=source mean Hz at tauM1000ms, full dynamic burn and record.',
            G='Stationary Graw=30clip((meanEHz*dt/[15*(1-exp(-dt/15))]-200)/300,0,1), held during each local assay. Not physical-time feedback.',
            streams='Same Gaussian/external seeds across iterations as numerical common random numbers; independent target replicas, not native seeds.'),
        approximation='Distinct-source cross spectra, E/I cross spectra and shared-input correlations omitted. Fixed axonal delays drop from individual-source autospectral power; original delays remain in the native system and would be required for temporal stability. Finite periodic spectrum may phase-lock local outputs.',
        evidence='Compare spatial field, cores, feedback activation segment and counterfactual Z drift; report input-output changes, PSD residuals and first/second1s stationarity. This pilot has no prespecified numerical tolerance for certification and cannot pass the native correspondence gate.',
        stop='Stop after three complete generations or on implementation/finite-value failure. Spectral iteration convergence is not physical dynamic stability. Failure redirects analysis, not noise-gain fitting or acceptance relaxation.',
        formal_bifurcation_allowed=False, counts_as_autonomous_loop=False,
        source=str(TEMPORAL/'source_spikes_0p1ms.npz'), producer_sha256=sha(__file__)))
    assert read(TEMPORAL/'observer_audit.json')['status'] == 'PASS'
    geo = dict(np.load(ADAPTED/'geometry.npz')); prep = read(OPS/'prepared.json'); p = prep['params']
    theta, theta_meta = load(); state = native.read_pickle(INITIAL)['engine']
    with np.load(OBS/'cell_statistics.npz') as a:
        nu = a['per_cell_mean_external_per_ms'].mean(0)
        original_moments = a['per_cell_mean_moments'].mean(0)
    E = np.arange(40000) < 32000
    tm = np.where(E,p['tau_m_E'],p['tau_m_I'])
    z = state['slow']['z']; k = np.r_[state['termination_mechanism']['sahp_g'],np.zeros(8000)]
    assert np.all(z[~E]==1) and np.isfinite(nu).all() and nu.min()>=0
    np.savez_compressed(OUT/'parameters.npz', Z=z, K=k, theta=theta, tm=tm,
        ref_steps=np.rint(np.where(E,p['tau_ref_E'],p['tau_ref_I'])/.1),
        jump_external=tm/p['tau_r_AMPA']*np.where(E,p['J_ext_E'],p['J_ext_I']),
        nu_per_ms=nu, initial_V=state['V'], initial_ref=state['ref'], initial_M=state['slow']['m'],
        region=geo['group_region'][geo['cell_group']], display=geo['group_cell'][geo['cell_group']],
        original_moments=original_moments)
    write(OUT/'threshold_identity.json',theta_meta)
    shutil.copy2(__file__,OUT/'preparation_producer.py')
    sp = sparse.load_npz(TEMPORAL/'source_spikes_0p1ms.npz').tocsr()
    assert sp.shape==(40000,N)
    gen=OUT/'generation_0';gen.mkdir()
    psd=np.lib.format.open_memmap(gen/'source_PSD.npy',mode='w+',dtype='f8',shape=(40000,N//2+1))
    rates=np.asarray(sp.sum(1)).ravel()/2.
    weights=np.full(N//2+1,2.);weights[[0,-1]]=1.
    parseval=0.
    for lo in range(0,40000,128):
        x=sp[lo:lo+128].toarray().astype(float); x-=x.mean(1,keepdims=True)
        power=abs(fft.rfft(x,axis=1,workers=2))**2;power[:,0]=0
        parseval=max(parseval,float(abs(power@weights/N**2-x.var(1)).max()))
        psd[lo:lo+len(x)]=power
    psd.flush();np.save(gen/'source_rate_Hz.npy',rates)
    assert parseval<1e-12
    write(OUT/'progress.json',dict(status='PREPARING_ORIGINAL_GRAPH',pid=os.getpid(),updated_epoch=time.time()))
    sim,_,_,identity=native.base.old.setup(9108405)
    assert identity==prep['graph_identity']
    rows=[]
    for kind in ['ampa','gaba']:
        matrices=sim.net[kind+'_by_delay'];w=sum(matrices).tocsr();w.sort_indices()
        overlap=int(sum(m.nnz for m in matrices)-w.nnz)
        assert overlap==0
        sparse.save_npz(OUT/f'original_{kind}_jump.npz',w)
        rows.append(dict(kind=kind,nnz=int(w.nnz),delay_matrix_count=len(matrices),
            repeated_source_target_across_delays=overlap,units='Native synaptic gate jump per source spike'))
    phase=np.exp(-2j*np.pi*fft.rfftfreq(N,d=.0001)*.0001)
    filters=[]
    for kind in ['AMPA','GABA']:
        ar,ad=np.exp(-.1/p['tau_r_'+kind]),np.exp(-.1/p['tau_d_'+kind])
        filters.append(abs((1-ad)/((1-ar*phase)*(1-ad*phase)))**2)
    np.save(OUT/'filter_power.npy',np.array(filters))
    # Independent check against the preceding observed-source variance analysis.
    previous=dict(np.load(TEMPORAL/'spectral_analysis/source_spectra.npz'))
    err=[]
    for q in range(2):
        var=np.asarray(psd)@(weights*np.array(filters)[q]/N**2)
        err.append(float(abs(var-previous['source_filtered_variance'][q]).max()))
    assert max(err)<1e-10
    write(gen/'complete.json',dict(status='COMPLETE_NATIVE_INITIAL_GUESS_ONLY',mean_E_rate_Hz=float(rates[E].mean())))
    result=dict(status='PREPARED',source_Parseval_max_abs_error=parseval,
        prior_source_autovariance_max_abs_errors=err,original_graph=rows,individual_thresholds=True,
        producer_sha256=sha(__file__))
    write(OUT/'preparation_qa.json',result);write(OUT/'progress.json',dict(status='PREPARED',updated_epoch=time.time()))
    print(result,flush=True)


if __name__=='__main__': main()
