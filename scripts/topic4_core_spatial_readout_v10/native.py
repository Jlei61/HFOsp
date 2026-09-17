"""Bounded native SNN replay with frozen contact and full-sheet readouts.

The closure's coexisting orbits are not unique SNN initial conditions. These
three cold-start runs test the corresponding three distinct parameter values;
they must not be relabelled as four verified SNN attractors.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, inspect, json, sys, time, subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/topic4_sef_hfo/core_spatial_readout_v10_20260916'
V7=ROOT/'results/topic4_sef_hfo/core_network_bifurcation_v7_20260916'
sys.path.insert(0,str(ROOT/'scripts/topic4_core_network_bifurcation_v7'))
import native as v7
engine=v7.native
from src.topic4_streaming_spike_readout import RecorderNumpy
PARAMETERS={'a':1.355,'b':1.34936498373232,'cd':1.38}

def run(key,duration=12000.):
    folder=OUT/'native'/key;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return
    start=time.time();g=PARAMETERS[key]
    engine.write(folder/'progress.json',dict(status='BUILDING',g=g))
    # setup uses the immutable baseline cache; no current candidate is substituted.
    sub,groups,loading,det,applied,cores=engine.setup(g,1,848101)
    assert applied['threshold']['n_raised']==0
    engine.write(folder/'applied_physics.json',applied)
    ref=json.loads((V7/'native/per_run/J1.38_s848101/result.json').read_text())
    assert engine.sha(inspect.getfile(engine.ORIGINAL))==ref['frozen_engine_sha256']
    engine.Observer=v7.Observer
    engine.NumpyProxy=lambda shape:RecorderNumpy(shape,sub.positions_e,sub.montage,sub.params.dt)
    result,observer=engine.simulate(sub,groups,loading,det,cores[0],848101,duration,folder/'progress.json')
    spikes=result['E_spk_bool'];movie=spikes.native();env,envdt,_=spikes.envelope()
    arrays=observer.arrays()
    # E refractory time is 2 ms: each neuron can contribute at most once per
    # 2-ms movie bin. Verify counts, rather than merely relying on that fact.
    field=movie['activity_counts']
    assert np.array_equal(field.sum((1,2)),arrays['six_group_counts_2ms'][:,:3].sum(1))
    parent=engine.builder.base.rt.read(engine.builder.base.PARENT)
    contract=engine.builder.base.rt.load_observation_contract(parent)
    assert list(sub.contact_names)==contract['contact_names']
    ob=engine.builder.base.observe(env,float(envdt),contract)
    mu=np.asarray(ob['centroid_ms']).reshape(-1,len(sub.contact_names))
    evaluator=engine.builder.base.rt.load_evaluator(parent)
    labels,support,dist=engine.builder.base.rt.classify_with_both_modes(evaluator,mu)
    arrays.update(contact_envelope=env.T.astype(np.float32),contact_envelope_dt_ms=envdt,
        contact_names=np.array(sub.contact_names),contact_xy_mm=sub.contact_xy,
        positions_E=sub.positions_e,positions_I=sub.positions_i,
        sheet_activity_counts=field,sheet_activity_frame_ms=movie['frame_ms'],
        centroid_ms=mu,recruitment_ms=ob['recruitment_ms'],
        primary_event_indices=np.array(ob['primary_event_indices'],int),event_mode=labels,
        event_support=support,event_distance_modes=dist,
        core_index_E=cores[0],core_index_I=cores[1],vtheta=sub.vtheta)
    if key in ('a','cd'):
        with np.load(V7/f'native/per_run/J{g:g}_s848101/trajectory.npz') as old:
            for name in ('spike_counts_2ms','active_counts_10ms','six_group_counts_2ms','exact_spike_time_ms','exact_spike_cell'):
                assert np.array_equal(arrays[name],old[name]),name
        parity='Exact full-duration spike/count parity with V7'
    else:parity='Same frozen executor; newly evaluated exact PD1 daughter parameter'
    np.savez_compressed(folder/'trajectory.npz',**arrays)
    engine.builder.base.rt.write(folder/'observation.json',ob)
    engine.write(folder/'observation_contract.json',contract)
    engine.write(folder/'result.json',dict(status='COMPLETE',key=key,g=g,topology=2511,noise=848101,
        duration_ms=observer.nsteps*sub.params.dt,burnin_ms=2000,wall_s=time.time()-start,
        native_closure_state_match='NOT_ESTABLISHED',initial_state='original native cold start',
        paired_reduced_states=['c','d'] if key=='cd' else [key],parity=parity,
        n_detected=len(ob['events']),n_primary=ob['n_primary_events'],
        field_count_equals_all_E_spikes=True,runaway_early_stop_ms=result.get('runaway_early_stop_ms'),
        engine_path=inspect.getfile(engine.ORIGINAL),engine_sha256=engine.sha(inspect.getfile(engine.ORIGINAL)),
        arrays_sha256=engine.sha(folder/'trajectory.npz')))
    print(key,'COMPLETE',time.time()-start,flush=True)

def batch():
    from concurrent.futures import ThreadPoolExecutor,as_completed
    (OUT/'logs').mkdir(parents=True,exist_ok=True)
    def one(key):
        with (OUT/'logs'/f'native_{key}.log').open('w') as log:
            code=subprocess.run([sys.executable,__file__,'--key',key],stdout=log,stderr=subprocess.STDOUT).returncode
        return dict(key=key,returncode=code)
    done=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for fut in as_completed([pool.submit(one,key) for key in PARAMETERS]):
            done.append(fut.result());engine.write(OUT/'native_batch_status.json',dict(completed=len(done),total=3,runs=done))
    assert all(r['returncode']==0 for r in done),done

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--key',choices=list(PARAMETERS));p.add_argument('--batch',action='store_true');a=p.parse_args()
    batch() if a.batch else run(a.key)
