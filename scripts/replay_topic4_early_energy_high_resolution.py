#!/usr/bin/env python3
"""One same-noise12s measurement replay with full0.1ms spatial spike counts."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
from datetime import datetime
import json
import pickle
import fcntl
import numpy as np
import psutil
import run_topic4_m_parameter_modes as core

ROOT=core.ROOT
ORIGINAL=core.OUT
WINDOW=ROOT/'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913'
OUT=WINDOW/'early_energy_high_resolution_replay'


def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'measurement_replay.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'qa.json').exists():
        assert core.read(OUT/'qa.json')['status']=='PASS';return
    plan_path=OUT/'plan.json'
    if not plan_path.exists():
        assert datetime.now().astimezone()<datetime.fromisoformat(core.read(WINDOW/'window.json')['deadline'])
        core.write(plan_path,dict(purpose='Resolve measured temporal aliasing of1ms native field power.',
            maximum_measurement_replays=1,horizon_s=12.,same_existing_condition='e0_t0_s9108401',
            eta_M=.005,tau_M_s=1.,seed=9108401,field_recording_ms=.1,
            new_biological_conditions=0,new_F_samples=0,original_M40_endpoints_unchanged=True,
            reason_for_additional_recording='Same sampled E low-band RMS changed16.95-fold under1ms binning in the second high state; original per-cell0.1ms spikes showed2.1ms periodic firing.',
            baseline_s=[1.,8.59],target_s=[10.59,11.59],device=1))
    assert psutil.virtual_memory().available/2**30>=68
    qa_path=WINDOW/'cuda_ordered_scatter_qa/full_network_qa.json';qa=core.read(qa_path)
    assert qa['status']=='PASS' and qa['full_engine_state_bitwise_identical'] and qa['all_observations_bitwise_identical']
    assert core.sha(qa['replacement'])==qa['replacement_sha256']
    protocol=core.read(ORIGINAL/'protocol.json')
    for p,h in protocol['source_hashes'].items():assert core.sha(p)==h,p
    source_job=core.read(ORIGINAL/'jobs/e0_t0_s9108401.json')
    job={**source_job,'name':'native_readout_0p1ms','horizon_s':12.}
    core.write(OUT/'protocol.json',protocol)
    (OUT/'jobs').mkdir(exist_ok=True);core.write(OUT/'jobs/native_readout_0p1ms.json',job)
    from src.topic4_cuda_ordered_scatter import wrap_simulator
    gpu=wrap_simulator(core.old.simulate_kick,device_index=1)
    field_path=OUT/'field_0p1ms.npy';population_path=OUT/'population_0p1ms.npy'
    resumed=(OUT/'runs/native_readout_0p1ms/checkpoint.pkl').exists()
    if resumed:
        assert field_path.exists() and population_path.exists()
    field=np.lib.format.open_memmap(field_path,mode='r+' if resumed else 'w+',dtype=np.uint16,shape=(120000,400))
    population=np.lib.format.open_memmap(population_path,mode='r+' if resumed else 'w+',dtype=np.uint16,shape=(120000,2))
    def simulate(p,net,*args,**kwargs):
        ne=int(net['NE']);assert ne==32000 and p.dt==.1
        cells=core.old.spatial_cell_index(net['pos'][:ne],n_grid=20,sheet_l_mm=p.L)
        original_observer=kwargs['spike_observer']
        def observe(tm,spk):
            k=round(tm/.1);assert 0<=k<120000
            values=np.bincount(cells[spk[:ne]],minlength=400)
            field[k]=values
            population[k]=[values.sum(),spk[ne:].sum()]
            original_observer(tm,spk)
            if (k+1)%10000==0:
                field.flush();population.flush()
        kwargs['spike_observer']=observe
        try:return gpu(p,net,*args,**kwargs)
        finally:field.flush();population.flush()
    core.old.simulate_kick=simulate;core.OUT=OUT
    core.write(OUT/'runtime.json',dict(pid=os.getpid(),device=1,executor=str(Path(__file__).resolve()),
        executor_sha256=core.sha(__file__),original_worker_sha256=core.sha(core.__file__),
        source_original_job=source_job,new_independent_samples=0))
    result=core.worker(job);assert result['end_s']==12.
    core.OUT=ORIGINAL
    import plot_topic4_m_parameter_modes as plot
    a=plot.load(OUT/'runs/native_readout_0p1ms',end_step=120000)
    b=plot.load(WINDOW/'early_Z_lookup_dense_figures/runs/early_z_refill_s9108401',end_step=120000)
    keys=['time_ms','spikes_1ms','regions_1ms','field_1ms','raster','slow_time_ms',
          'Z','M','currents','lfp_time_ms','lfp_raw','inputs']
    for key in keys:assert np.array_equal(a[key],b[key]),key
    assert np.array_equal(field.reshape(12000,10,400).sum(1),a['field_1ms'])
    assert np.array_equal(population.reshape(12000,10,2).sum(1),a['spikes_1ms'])
    assert np.array_equal(field.sum(1),population[:,0])
    core.write(OUT/'qa.json',dict(status='PASS',observations_bitwise_equal_to_existing_trajectory=keys,
        raw_to_1ms_field_exact=True,raw_to_1ms_population_exact=True,raw_spatial_counts_conserved=True,
        field_path=str(field_path),field_sha256=core.sha(field_path),shape=list(field.shape),
        population_path=str(population_path),population_sha256=core.sha(population_path),
        true_dt_ms=.1,recording_end_s=12.,source=str(WINDOW/'early_Z_lookup_dense_figures/runs/early_z_refill_s9108401'),
        measurement_replay_not_new_trial=True,new_F_samples=0,scientific_trial_followup_not_shortened=True))
    print(json.dumps(dict(status='COMPLETE_MEASUREMENT_REPLAY_PASS',end_s=12.,new_samples=0)))


if __name__=='__main__':main()
