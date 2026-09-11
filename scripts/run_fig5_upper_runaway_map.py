#!/usr/bin/env python3
"""Finite-preparation native-delay Z x M map above the earlier fold scan."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
import json
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_topic4_dual_core_spatial_z_bifurcation import load_z_map, atomic_json, atomic_npz, sha256
from scripts.run_topic4_dual_core_spatial_z_stability_assay import _substrate
from scripts.run_topic4_dual_core_oscillatory_phase_map import _terminal_low_seed, _branch_seed, _perturbed
from src.topic4_dual_core_spatial_z_delay import build_coarse_delay_operators, simulate_delayed_ou_trajectory
from src.topic4_dual_core_spatial_z import path_state, regional_rates_hz
from src.topic4_dual_core_oscillation_phase import population_cycle_modulation
from src.topic4_patient_zm_meanfield import load_patient_coarse_model

BASE = Path('/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z')
WORK = {}


def classify_tail(trace, regional, *, dt_ms, gate):
    n = int(round(gate['tail_duration_ms']/dt_ms))
    trace = np.asarray(trace, float)
    if len(trace) < n or not np.all(np.isfinite(trace)):
        raise ValueError('complete finite terminal window required')
    tail = trace[-n:]
    half = np.array_split(tail, 2)
    drift = float(np.mean(half[1])-np.mean(half[0]))
    mean = float(tail.mean())
    stationary = abs(drift) <= gate['maximum_absolute_half_tail_drift_hz']
    runaway = (stationary and mean >= gate['minimum_population_mean_rate_hz']
               and min(regional.values()) >= gate['minimum_each_regional_rate_hz'])
    label = 'tonic_runaway' if runaway else ('bounded' if stationary and mean < 250 else 'unresolved')
    return {'state':label, 'population_hz':mean, 'half_tail_drift_hz':drift,
            'stationary_gate':stationary, 'regional_hz':regional, 'runaway':bool(runaway)}


def cell(task):
    s, gain, prep = task
    c, model, zm, ops = (WORK[k] for k in ('config','model','zmap','ops'))
    tag = f's{s:.3f}_m{gain:g}_{prep}'.replace('.', 'p')
    out = WORK['out']/'per_condition'/tag
    if out.with_suffix('.json').exists():
        old = json.loads(out.with_suffix('.json').read_text())
        if old['analysis_hash'] != WORK['analysis_hash']:
            raise RuntimeError('resume configuration drift')
        return old
    initial = (_terminal_low_seed(WORK['branch']) if prep == 'pre_fold' else
               _branch_seed(WORK['branch'], prefix='outer_high', parameter=s))
    initial = _perturbed(initial, n_cells=model.n_cells, fraction=.01, floor=.0001)
    za, zb, zs, field = path_state(zm, s)
    z2 = zm.z_second_moment_field(z_a=za,z_b=zb,z_surround=zs)
    eta = WORK['eta']*gain
    result = simulate_delayed_ou_trajectory(
        model,ops,initial,z_field=field,z_second_moment=z2,
        ou_rate_e=np.zeros((int(round(c['duration_ms']/ops.dt_ms)),model.n_cells),np.float32),
        tail_steps=int(round(c['tail_ms']/ops.dt_ms)),eta_m=eta,tau_m_slow_ms=500.)
    regional = regional_rates_hz(model,zm,result['tail_mean_rates'][:model.n_cells])
    record = classify_tail(result['mean_e_rate_hz'],regional,dt_ms=ops.dt_ms,gate=WORK['gate'])
    record.update({'s':s,'m_gain_scale':gain,'eta_m':eta,'preparation':prep,
                   'Z_core':za,'Z_surround':zs,'analysis_hash':WORK['analysis_hash'],
                   'mean_M_final':float(np.mean(result['final_adaptation_state'])),
                   'modulation':population_cycle_modulation(result['mean_e_rate_hz'][-10000:],dt_ms=ops.dt_ms)})
    record['modulation']['cycle_profile_hz'] = record['modulation']['cycle_profile_hz'].tolist()
    atomic_npz(out.with_suffix('.npz'),time_ms=np.arange(len(result['mean_e_rate_hz']))*ops.dt_ms,
               population_hz=result['mean_e_rate_hz'],M=result['mean_adaptation_state'],
               final_rates=result['final_rates'],final_M=result['final_adaptation_state'],
               tail_mean_rates=result['tail_mean_rates'])
    record['npz']={'path':str(out.with_suffix('.npz')),'sha256':sha256(out.with_suffix('.npz'))}
    atomic_json(record,out.with_suffix('.json'))
    return record


def main():
    start=time.time()
    cfgpath=ROOT/'config/fig5_upper_runaway_map_20260905.json'
    c=json.loads(cfgpath.read_text())
    gatepath=ROOT/'config/topic4_dual_core_runaway_boundary_v1.json'
    bc=json.loads(gatepath.read_text())
    modelpath=BASE/'deterministic_meanfield/dualcore_topology_2542_ngrid10.npz'
    zpath=modelpath.with_suffix('.zmap.npz')
    branchpath=BASE/'bifurcation/dualcore_spatial_z_bifurcation.npz'
    revpath=ROOT/'config/topic4_rev21_dual_core_zm_transition.json'
    model=replace(load_patient_coarse_model(modelpath),tau_gaba_ms=9.)
    substrate,transition,_,manifest=_substrate(revpath,Path('/home/honglab/leijiaxin/HFOsp'),2542)
    ops=build_coarse_delay_operators(substrate,model)
    if not np.isclose(ops.dt_ms,c['dt_ms']): raise RuntimeError('native dt drift')
    sources={str(p):sha256(p) for p in [cfgpath,gatepath,modelpath,zpath,branchpath,revpath,transition,manifest,Path(__file__)]}
    import hashlib
    analysis_hash=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    out=BASE/'upper_runaway_map_20260905'
    (out/'per_condition').mkdir(parents=True,exist_ok=True)
    atomic_json({'status':'FROZEN_BEFORE_EXECUTION','config':c,'sources':sources,'analysis_hash':analysis_hash},out/'frozen_contract.json')
    with np.load(branchpath,allow_pickle=False) as a: branch={k:a[k] for k in a.files}
    WORK.update(config=c,model=model,zmap=load_z_map(zpath),ops=ops,branch=branch,
                gate=bc['predeclared_tonic_runaway_gate'],eta=bc['dynamics']['eta_m'],out=out,analysis_hash=analysis_hash)
    tasks=[(s,g,p) for s in c['s_values'] for g in c['m_gain_scales'] for p in c['preparations']]
    rows=[]
    with ProcessPoolExecutor(max_workers=24,mp_context=mp.get_context('fork')) as pool:
        for future in as_completed([pool.submit(cell,t) for t in tasks]):
            row=future.result(); rows.append(row)
            print(json.dumps({'completed':len(rows),'total':len(tasks),'s':row['s'],'gain':row['m_gain_scale'],'prep':row['preparation'],'state':row['state'],'seconds':round(time.time()-start)}),flush=True)
    atomic_json({'status':'UPPER_RUNAWAY_FINITE_PREPARATION_MAP_COMPLETE','config':c,'gate':WORK['gate'],
                 'rows':sorted(rows,key=lambda r:(r['m_gain_scale'],r['s'],r['preparation'])),
                 'sources':sources,'analysis_hash':analysis_hash,'wall_seconds':time.time()-start},out/'summary.json')


if __name__=='__main__': main()
