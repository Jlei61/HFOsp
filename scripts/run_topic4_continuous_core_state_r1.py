#!/usr/bin/env python3
"""Frozen historical geometry with one continuous state; one auditable worker."""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src/snn_engine')]
import numpy as np
from src import topic4_initial_state_runtime as rt
from src.topic4_continuous_core_state import ContinuousIState, ou_path, spatial_groups
from src.topic4_initial_state import RunObserver, array_sha256, event_times_ms
from src.topic4_zm_ictal_transition import make_external_drive
from src.topic4_node_dualmode import sheet_activity_movie
from src.topic4_observation_repaired import observe
from src.sef_hfo_snn_adapter import snn_event_envelope
from kick_probe import simulate_kick

DESIGN = ROOT/'config/topic4_continuous_core_state_r1.json'


class ProgressObserver(RunObserver):
    def __init__(self, progress_path, job, **kwargs):
        super().__init__(**kwargs)
        self.progress_path, self.job = progress_path, job

    def observe(self, t, tm, *args):
        super().observe(t, tm, *args)
        if (t+1) % round(1000./self.dt_ms) == 0:
            rt.write(self.progress_path, dict(status='SIMULATING', job=self.job,
                                              simulated_ms=tm+self.dt_ms, updated_unix=time.time()))


def build(design, seed):
    return rt.build_frozen_substrate(design, design['topology_seed'], seed)


def control_for(sub, cand, design, job, duration):
    groups, indices, loading, radius = spatial_groups(sub, cand['node_field']['centers_mm'])
    dt = sub.params.dt; n = round(duration/dt)
    z = (ou_path(n, dt, design['state']['tau_ms'], job['state_seed']) if job['kind']=='ou'
         else np.full(n, float(job['z'])))
    control = ContinuousIState(indices, loading, z, amplitude=design['state']['amplitude'],
                              dt_ms=dt, seed=job['coupling_seed'],
                              warmup_ms=design['state']['warmup_ms'], ramp_ms=design['state']['ramp_ms'])
    return control, groups, radius


def simulate(sub, transition, design, job, control, observer=None):
    sub.net['rng'] = np.random.default_rng(job['dynamics_seed'])
    drive = make_external_drive(sub, transition['spatial_ou'], job['dynamics_seed'])
    result = simulate_kick(sub.params, sub.net, KICK_BOOST=0., t_kick=1e9,
                           V_th_per_neuron=sub.vtheta, slow=None, early_stop_runaway=True,
                           external_e_rate_drive=drive, step_observer=observer, external_i_state=control)
    return result, drive


def qualification(design):
    out = Path(design['output_root']); start=time.time()
    job=design['jobs'][0]
    sub, cand, transition, execution, audit, network = build(design, job['dynamics_seed'])
    # Independently compare the actual frozen reference arrays before any run.
    with np.load(design['sources']['baseline_worker_arrays']['path']) as z:
        comparisons = {k:np.array_equal(np.asarray(v,np.float32),np.asarray(z[k],np.float32))
                       for k,v in [('vtheta',sub.vtheta),('h',sub.h_e),('positions_E',sub.positions_e)]}
    if not all(comparisons.values()):
        raise RuntimeError(f'baseline identity mismatch: {comparisons}')
    sub.params.T=250.
    plain,_=simulate(sub,transition,design,job,None)
    zero_design=copy.deepcopy(design);zero_design['state']['amplitude']=0.
    co,groups,radius=control_for(sub,cand,zero_design,job,250.)
    off,_=simulate(sub,transition,design,job,co)
    # Warm-up is disabled only in this input qualification, not in formal runs.
    active_design=copy.deepcopy(design);active_design['state']['warmup_ms']=0.
    ca,_,_=control_for(sub,cand,active_design,job,250.)
    active,_=simulate(sub,transition,design,job,ca)
    tests=dict(reference_arrays_equal=all(comparisons.values()),
               zero_state_exact_spikes=np.array_equal(plain['E_spk_bool'],off['E_spk_bool']),
               zero_state_exact_rates=np.array_equal(plain['rate_E'],off['rate_E']),
               active_background_counts_equal=co.audit()['legacy_external_counts_sha256']==ca.audit()['legacy_external_counts_sha256'],
               rate_budget_conserved=ca.maximum_relative_total_rate_error<1e-12,
               state_changes_target_I_counts=ca.changed_counts>0)
    record=dict(status='PASS' if all(tests.values()) else 'FAIL',tests=tests,
                candidate_id=cand['candidate_id'],geometry_radius_mm=radius,
                group_counts={k:len(v) for k,v in groups.items()},
                static_array_identity=rt.static_identity(sub,audit),
                active_audit=ca.audit(),wall_seconds=time.time()-start,
                peak_rss_gib=rt.peak_rss_gib(),source_hashes=rt.loaded_source_hashes())
    rt.write(out/'qualification.json',record)
    if not all(tests.values()): raise RuntimeError('physical qualification failed')
    print(json.dumps(rt.json_safe(record)),flush=True)


def worker(design, job):
    out=Path(design['output_root']);qual=rt.read(out/'qualification.json')
    if qual['status']!='PASS':raise RuntimeError('qualification not passed')
    for relative,digest in qual['source_hashes'].items():
        if rt.sha(ROOT/relative)!=digest:raise RuntimeError(f'qualified source changed: {relative}')
    start=time.time();stem=job['id'];work=out/'workers';work.mkdir(exist_ok=True)
    if (work/(stem+'.json')).exists():
        old=rt.read(work/(stem+'.json'))
        if (old.get('status')=='COMPLETE' and old['design_sha256']==rt.sha(DESIGN)
                and rt.sha(work/(stem+'.npz'))==old['arrays_sha256']): return
        raise RuntimeError('existing worker is not a verified completion')
    rt.write(work/(stem+'.progress.json'),dict(status='BUILDING',job=job))
    sub,cand,transition,execution,pa,network=build(design,job['dynamics_seed'])
    identity=rt.static_identity(sub,pa)
    if identity!=qual['static_array_identity']:raise RuntimeError('substrate differs from qualified baseline')
    sub.params.T=job['duration_ms'];dt=sub.params.dt;n=round(sub.params.T/dt)
    control,groups,radius=control_for(sub,cand,design,job,sub.params.T)
    observer=ProgressObserver(work/(stem+'.progress.json'),job,dt_ms=dt,n_e=sub.n_e,n_total=sub.n_e+sub.n_i,groups=groups,
                         n_steps=n,segment_ms=1000.,trace_ms=1.)
    rt.write(work/(stem+'.progress.json'),dict(status='SIMULATING',job=job,started_unix=time.time()))
    result,drive=simulate(sub,transition,design,job,control,observer)
    observed=observer.finish();trace=observed['trace'];spikes=result['E_spk_bool']
    actual_ms=len(spikes)*dt
    rt.write(work/(stem+'.progress.json'),dict(status='OBSERVING',job=job,actual_duration_ms=actual_ms))
    movie=sheet_activity_movie(spikes,sub.positions_e,dt_ms=dt,frame_ms=2.,bin_mm=1.,sheet_mm=20.)
    env,envdt,_=snn_event_envelope(spikes,sub.positions_e,sub.montage,dt)
    del spikes,result['E_spk_bool']
    contract=rt.load_observation_contract(design)
    observation=observe(env,float(envdt),contract)
    mu=np.asarray(observation['centroid_ms'],float).reshape(-1,len(contract['contact_names']))
    evaluator=rt.load_evaluator(design);objective=rt.load_objective(design)
    labels,support,dist=rt.classify_with_both_modes(evaluator,mu)
    times,primary=event_times_ms(observation)
    features=np.full((len(mu),len(objective.target_global)),np.nan,np.float32)
    readable=np.isfinite(mu).sum(1)>=2
    if readable.any():features[readable]=objective.embedding(mu[readable]).astype(np.float32)
    events=[]
    for i,e in enumerate(observation['events']):
        t0=float(e['qualifying_interval_ms'][0])
        step=min(round(t0/dt),control.n_seen-1)
        events.append(dict(event_index=i,event_time_ms=float(times[i]),qualifying_start_ms=t0,
                           window_ms=e['window_ms'],primary_eligible=e['primary_eligible'],
                           exclusion_reasons=e['primary_exclusion_reasons'],
                           prolonged=e['prolonged'],mode=int(labels[i]),support=int(support[i]),
                           distance_modes=dist[i].tolist(),n_contacts=int(np.isfinite(mu[i]).sum()),
                           z_at_detection_start=float(control.z[step]),q_at_detection_start=float(control.q[step])))
    rt.atomic_npz(work/(stem+'.npz'),contact_envelope=np.asarray(env,np.float32),
                  contact_envelope_dt_ms=np.asarray(envdt),contact_names=np.asarray(sub.contact_names),
                  contact_xy_mm=sub.contact_xy,positions_E=np.asarray(sub.positions_e,np.float32),
                  h=np.asarray(sub.h_e,np.float32),vtheta=np.asarray(sub.vtheta,np.float32),
                  sheet_activity_counts=movie['activity_counts'],sheet_activity_frame_ms=np.asarray(2.),
                  centroid_ms=mu,primary_event_indices=primary,event_mode=labels,event_support=support,
                  event_distance_modes=dist,event_time_ms=times,event_phi=features,
                  state_z=control.z[:control.n_seen],state_q=control.q[:control.n_seen],state_dt_ms=np.asarray(dt),
                  I_target_indices=control.indices,I_loading=control.loading,
                  I_actual_counts=control.actual_counts,I_baseline_counts=control.base_counts,
                  I_expected_actual=control.expected_actual,I_expected_baseline=control.expected_base,
                  rate_E=np.asarray(result['rate_E'],np.float32),
                  **{f'trace_{k}':v for k,v in trace.items()},
                  **{f'group_{k}':v for k,v in groups.items()})
    payload=dict(status='COMPLETE',job=job,candidate_id=cand['candidate_id'],design_sha256=rt.sha(DESIGN),
                 actual_duration_ms=actual_ms,runaway_early_stop_ms=result.get('runaway_early_stop_ms'),
                 physical_status='RUNAWAY' if result.get('runaway_early_stop_ms') is not None else 'COMPLETE_NO_RUNAWAY',
                 static_array_identity=identity,state_audit=control.audit(),events=events,
                 group_counts={k:len(v) for k,v in groups.items()},radius_mm=radius,
                 n_detected=len(events),n_primary=len(primary),
                 observation_boundary=observation['boundary_or_low_window_support'],
                 input_segments=observed['segments'],wall_seconds=time.time()-start,
                 peak_rss_gib=rt.peak_rss_gib(),arrays_sha256=rt.sha(work/(stem+'.npz')))
    rt.write(work/(stem+'.json'),payload)
    rt.write(work/(stem+'.progress.json'),dict(status='COMPLETE',job=job,n_primary=len(primary)))
    print(json.dumps({k:payload[k] for k in ['status','job','physical_status','n_primary','wall_seconds']}),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--qualify',action='store_true');ap.add_argument('--job')
    args=ap.parse_args();design=rt.read(DESIGN)
    if args.qualify:qualification(design)
    else:worker(design,next(j for j in design['jobs'] if j['id']==args.job))

if __name__=='__main__':main()
