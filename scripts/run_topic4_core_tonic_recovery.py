"""Versioned tonic-core input assay, reusing frozen connectivity-v2 physics/readout.

Only an explicit constant E-core rate offset is added through the existing engine
hook. Conditional Poisson variance changes with its mean; OU parameters and the
outside deterministic mean remain fixed. No event or label reaches this drive.
"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
from scripts import run_topic4_core_connectivity_search as base
from scripts import run_topic4_propagation_recovery_night as night
from params import compute_nu_theta
rt=base.rt
SCRIPT=Path(__file__).resolve()
PLAN=night.OUT/'tonic_input_plan.json'

class ConstantCoreRateOffset:
    def __init__(self,core_index,signal_per_ms,scale):
        if not np.isfinite(scale) or not 0<scale<=1:raise ValueError('tonic assay scale must be in (0,1]')
        self.values=np.where(np.asarray(core_index)>=0,(scale-1)*float(signal_per_ms),0.)
        self.values.setflags(write=False)
    def step(self,time_ms):return self.values

def worker(stage,cid,topology_seed,seed,duration):
    plan=rt.read(PLAN)
    for path,h in plan['source_snapshot'].items():
        if rt.sha(path)!=h:raise RuntimeError(f'frozen source changed: {path}')
    candidate=rt.read(base.OUT/'candidates'/f'{cid}.json');scale=float(candidate.get('core_mean_rate_scale',1.))
    unit=base.unit_dir(stage,cid,topology_seed,seed);unit.mkdir(parents=True,exist_ok=True)
    network_record=rt.read(base.OUT/'confirmation/frozen_networks.json') if int(topology_seed)!=2511 else None
    design=dict(candidate=candidate,stage=stage,topology_seed=int(topology_seed),dynamics_seed=int(seed),duration_ms=float(duration),runaway=plan['runaway'],
        physics=plan['physics'],source_snapshot=plan['source_snapshot'],analysis_burnin_ms=plan['analysis']['burnin_ms'])
    path=unit/'design.json'
    if path.exists() and rt.read(path)!=rt.json_safe(design):raise RuntimeError('resume changes physical unit')
    rt.write(path,design);work=unit/'workers';work.mkdir(exist_ok=True);stem='trajectory'
    if (work/(stem+'.json')).exists():
        if base.complete(work/(stem+'.json')):return
        raise RuntimeError('existing result is not a verified completion')
    start=time.time();rt.write(work/(stem+'.progress.json'),dict(status='BUILDING',started_unix=start))
    sub,groups,loading,deterministic,applied,(core_e,core_i)=base.build_unit(candidate,int(topology_seed),int(seed),network_record)
    signal=float(sub.params.nu_ext_ratio*compute_nu_theta(sub.params)[0])
    drive=None if scale==1 else ConstantCoreRateOffset(core_e,signal,scale)
    applied['physics']=plan['physics']['version'];applied['input'].update(core_mean_rate_scale=scale,
        baseline_signal_per_ms=signal,core_mean_signal_per_ms=signal*scale,
        core_constant_offset_per_ms=(scale-1)*signal,outside_mean_signal_per_ms=signal,
        OU_sigma_n=sub.params.sigma_n,OU_tau_ms=sub.params.tau_n,
        hook='None (exact legacy call)' if drive is None else 'external_e_rate_drive: constant offset only on core E',
        poisson='conditional count variance equals conditional mean; this variance is not held fixed when mean changes')
    rt.write(unit/'applied_physics.json',applied)
    sub.params.T=float(duration);dt=sub.params.dt;n=round(sub.params.T/dt);n_total=sub.n_e+sub.n_i
    sub.net['rng']=np.random.default_rng(int(seed))
    job=dict(candidate=cid,stage=stage,topology_seed=int(topology_seed),dynamics_seed=int(seed),duration_ms=float(duration))
    observer=base.ProgressObserver(work/(stem+'.progress.json'),job,dt_ms=dt,n_e=sub.n_e,n_total=n_total,groups=groups,n_steps=n,segment_ms=1000.,trace_ms=1.)
    audit=base.RateAudit({k:groups[k] for k in ['coreAE','coreBE','surroundE','allI']},n_total,dt)
    rw=plan['runaway'];kwargs=dict(KICK_BOOST=0.,t_kick=1e9,V_th_per_neuron=sub.vtheta,slow=None,early_stop_runaway=True,
        es_thresh_hz=rw['es_thresh_hz'],es_dur_ms=rw['es_dur_ms'],post_runaway_record_ms=rw['post_runaway_record_ms'],global_ou_loading=loading,
        deterministic_external_mask=deterministic,step_observer=observer,afferent_rate_observer=audit)
    if drive is not None:kwargs['external_e_rate_drive']=drive
    result=base.simulate_streaming(base.simulate_kick,sub.params,sub.net,positions=sub.positions_e,montage=sub.montage,**kwargs)
    if audit.max_outside_deviation!=0:raise RuntimeError('outside input leakage')
    rates=audit.arrays();v=rates['values'];columns=list(rates['columns'])
    sig=v[:,columns.index('tonic_signal_per_ms')];xi=v[:,columns.index('global_xi_per_ms')]
    rate_errors={g:float(np.max(abs(v[:,columns.index(g+'_mean_rate_per_ms')]-np.maximum(scale*sig+xi,0)))) for g in ['coreAE','coreBE']}
    if max(rate_errors.values())>1e-12:raise RuntimeError(f'core mean not applied: {rate_errors}')
    spikes=result['E_spk_bool'];actual_ms=len(spikes)*dt;movie=spikes.native();env,envdt,_=spikes.envelope();del spikes,result['E_spk_bool']
    observed=observer.finish();trace=observed['trace'];parent=rt.read(base.PARENT);contract=rt.load_observation_contract(parent)
    observation=base.observe(env,float(envdt),contract);mu=np.asarray(observation['centroid_ms'],float).reshape(-1,len(contract['contact_names']))
    ev=rt.load_evaluator(parent);objective=rt.load_objective(parent);labels,support,dist=rt.classify_with_both_modes(ev,mu);times,primary=base.event_times_ms(observation)
    features=np.full((len(mu),len(objective.target_global)),np.nan,np.float32);readable=np.isfinite(mu).sum(1)>=2
    if readable.any():features[readable]=objective.embedding(mu[readable]).astype(np.float32)
    events=[dict(event_index=i,event_time_ms=float(times[i]),qualifying_start_ms=float(e['qualifying_interval_ms'][0]),qualifying_interval_ms=e['qualifying_interval_ms'],
        window_ms=e['window_ms'],primary_eligible=e['primary_eligible'],exclusion_reasons=e['primary_exclusion_reasons'],prolonged=e['prolonged'],mode=int(labels[i]),
        support=int(support[i]),distance_modes=dist[i].tolist(),n_contacts=int(np.isfinite(mu[i]).sum())) for i,e in enumerate(observation['events'])]
    steps_seen=max(int(observed['n_observed_steps']),1)
    rt.atomic_npz(work/(stem+'.npz'),contact_envelope=np.asarray(env,np.float32),contact_envelope_dt_ms=np.asarray(envdt),contact_names=np.asarray(sub.contact_names),
        contact_xy_mm=sub.contact_xy,positions_E=np.asarray(sub.positions_e,np.float32),positions_I=np.asarray(sub.positions_i,np.float32),h=np.asarray(sub.h_e,np.float32),
        vtheta=np.asarray(sub.vtheta,np.float32),core_index_E=core_e,core_index_I=core_i,sheet_activity_counts=movie['activity_counts'],sheet_activity_frame_ms=np.asarray(movie['frame_ms']),
        centroid_ms=mu,recruitment_ms=np.asarray(observation['recruitment_ms'],float),primary_event_indices=np.asarray(primary,int),event_mode=labels,event_support=support,
        event_distance_modes=dist,event_time_ms=times,event_phi=features,rate_E=np.asarray(result['rate_E'],np.float32),rate_I=np.asarray(result['rate_I'],np.float32),
        input_expected_per_step=(observed['ext_segment_sums'].sum(0)/steps_seen).astype(np.float32),input_is_stochastic=~deterministic,
        input_rate_audit_columns=rates['columns'],input_rate_audit_values=rates['values'],**{f'trace_{k}':v for k,v in trace.items()},**{f'group_{k}':v for k,v in groups.items()})
    payload=dict(status='COMPLETE',job=job,candidate_id=cid,stage=stage,physics_version=plan['physics']['version'],design_sha256=rt.sha(path),
        applied_physics_sha256=rt.sha(unit/'applied_physics.json'),actual_duration_ms=actual_ms,runaway_early_stop_ms=result.get('runaway_early_stop_ms'),
        physical_status='RUNAWAY' if result.get('runaway_early_stop_ms') is not None else 'COMPLETE_NO_RUNAWAY',static_array_identity=applied['identity'],
        external_input=result['external_input'],input_segments=observed['segments'],events=events,group_counts=applied['group_counts'],n_detected=len(events),n_primary=len(primary),
        observation_boundary=observation['boundary_or_low_window_support'],maximum_outside_rate_deviation=audit.max_outside_deviation,core_rate_application_max_error=rate_errors,
        wall_seconds=time.time()-start,peak_rss_gib=rt.peak_rss_gib(),arrays_sha256=rt.sha(work/(stem+'.npz')))
    rt.write(work/(stem+'.json'),payload);rt.write(work/(stem+'.progress.json'),dict(status='COMPLETE',job=job,n_primary=len(primary)))
    print(json.dumps({k:payload[k] for k in ['status','job','physical_status','n_primary','wall_seconds','peak_rss_gib']}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['worker']);ap.add_argument('--stage',required=True);ap.add_argument('--candidate',required=True)
    ap.add_argument('--topology',type=int,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--duration',type=float,required=True)
    a=ap.parse_args();worker(a.stage,a.candidate,a.topology,a.seed,a.duration)
