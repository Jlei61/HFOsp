"""One native 60 s evaluation, followed by the exact frozen E1146 objective."""
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import argparse
import pickle
import time
from modeling.scripts.topic4_core_count_optimization.common import *
from modeling.scripts.topic4_core_count_optimization.physics import build_unit, CoreCountOU, InputAudit, base, rt
from modeling.src.snn_engine.params import compute_nu_theta
from modeling.src.topic4_observation_repaired import observe

def execute(k, restart, cid, phase='train', noise=None):
    verify_sources()
    plan = read(OUT / 'plan.json')
    d = run_dir(k, restart)
    c = read(OUT / 'canary_candidates' / f'core{k}.json') if phase == 'canary' else read(d / 'candidates' / f'{cid}.json')
    noise = plan['training_noise'] if noise is None else noise
    unit = OUT / 'canary' / f'core{k}' if phase == 'canary' else d / ('units' if phase == 'train' else 'confirmation') / (cid if phase == 'train' else f'{cid}_{noise}')
    unit.mkdir(parents=True, exist_ok=True)
    if (unit / 'result.json').exists():
        if sha(unit / 'trajectory.npz') != read(unit / 'result.json')['arrays_sha256']:
            raise RuntimeError('Completed output mutated')
        return
    duration = plan['canary_ms'] if phase == 'canary' else plan['duration_ms']
    started = time.time()
    job = dict(core_count=k, restart=restart, candidate=cid, phase=phase, topology_seed=plan['topology_seed'], dynamics_seed=noise, duration_ms=duration)
    write(unit / 'design.json', dict(job=job, candidate=c, plan_sha256=sha(OUT / 'plan.json')))
    write(unit / 'progress.json', dict(status='BUILDING', job=job, started_unix=started))
    sub, groups, loading, deterministic, applied, (ec, ic) = build_unit(c, plan['topology_seed'], noise)
    if len(applied['threshold']['members']) != k or min(applied['threshold']['members']) == 0:
        raise RuntimeError('Empty/missing core')
    if applied['threshold']['n_raised'] or np.any(loading[sub.n_e:]):
        raise RuntimeError('Threshold/input contract')
    with (OUT / 'objective.pkl').open('rb') as f:
        objective = pickle.load(f)
    if list(sub.contact_names) != objective.names:
        raise RuntimeError('Historical montage changed')
    np.testing.assert_array_equal(applied['threshold']['centers_mm'], c['centers_mm'])
    weights = applied['graph']['stage_audits']['weights']
    for name, factor in weights['factors'].items():
        if factor != c['parameters'][name]:
            raise RuntimeError('Pathway factor ignored: ' + name)
        block = weights['blocks'][name]
        if block['n_edges'] and (not np.isclose(block['weight_after'], block['weight_before'] * factor, rtol=1e-09)):
            raise RuntimeError('Pathway dose: ' + name)
    axis = applied['graph']['kernel']['theta_deg']
    if not np.isclose(axis, applied['reference_kernel']['theta_deg'] + c['vector'][-1]):
        raise RuntimeError('Axis ignored')
    signal = float(sub.params.nu_ext_ratio * compute_nu_theta(sub.params)[0])
    drive = CoreCountOU(ec, signal, c['core_mean_rate_scale'], c['core_ou_correlation'], sub.params.dt, sub.params.tau_n, sub.params.sigma_n, noise)
    loading *= 0.0
    applied.update(input_mixture=drive.metadata(), duration_ms=duration, core_count=k, objective_sha256=sha(OUT / 'objective.pkl'), observer_sha256=sha(OUT / 'observer.json'))
    write(unit / 'applied_physics.json', applied)
    sub.params.T = duration
    dt = sub.params.dt
    nt = sub.n_e + sub.n_i
    sub.net['rng'] = np.random.default_rng(noise)
    observer = base.ProgressObserver(unit / 'progress.json', job, dt_ms=dt, n_e=sub.n_e, n_total=nt, groups=groups, n_steps=round(duration / dt), segment_ms=1000.0, trace_ms=1.0)
    audit_groups = {name: index for name, index in groups.items() if name.endswith('E') or name == 'allI'}
    audit = InputAudit(audit_groups, nt, dt, drive)
    result = base.simulate_streaming(base.simulate_kick, sub.params, sub.net, positions=sub.positions_e, montage=sub.montage, KICK_BOOST=0.0, t_kick=1000000000.0, V_th_per_neuron=sub.vtheta, slow=None, early_stop_runaway=True, **base.RUNAWAY, global_ou_loading=loading, deterministic_external_mask=deterministic, step_observer=observer, afferent_rate_observer=audit, external_e_rate_drive=drive)
    if audit.max_outside_deviation != 0 or audit.max_I_deviation != 0 or max(audit.max_core_error.values()) > 1e-12:
        raise RuntimeError('Input audit failed')
    spikes = result.pop('E_spk_bool')
    actual = len(spikes) * dt
    movie = spikes.native()
    env, envdt, _ = spikes.envelope()
    del spikes
    observed = observer.finish()
    trace = observed['trace']
    physical = 'RUNAWAY' if result.get('runaway_early_stop_ms') is not None else 'COMPLETE_NO_RUNAWAY'
    if physical == 'COMPLETE_NO_RUNAWAY' and abs(actual - duration) > dt:
        raise RuntimeError('Incomplete simulation')
    obs = None if phase == 'canary' else observe(env, float(envdt), read(OUT / 'observer.json'))
    mu = np.empty((0, len(objective.names))) if obs is None else np.asarray(obs['centroid_ms']).reshape(-1, len(objective.names))
    primary = np.array([], int) if obs is None else np.array([i for i in obs['primary_event_indices'] if obs['events'][i]['window_ms'][0] >= plan['burnin_ms'] and obs['events'][i]['window_ms'][1] <= actual], int)
    save_npz(unit / 'trajectory.npz', contact_envelope=np.asarray(env, np.float32), contact_envelope_dt_ms=np.asarray(envdt), contact_names=np.asarray(sub.contact_names), contact_xy_mm=np.asarray(sub.contact_xy), positions_E=np.asarray(sub.positions_e, np.float32), positions_I=np.asarray(sub.positions_i, np.float32), vtheta=np.asarray(sub.vtheta, np.float32), core_index_E=ec, core_index_I=ic, sheet_activity_counts=movie['activity_counts'], sheet_activity_frame_ms=np.asarray(movie['frame_ms']), centroid_ms=mu, primary_event_indices=primary, rate_E=np.asarray(result['rate_E'], np.float32), rate_I=np.asarray(result['rate_I'], np.float32), **{f'trace_{name}': value for name, value in trace.items()}, **{f'group_{name}': value for name, value in groups.items()})
    score = dict(J=None, status='CANARY_NO_FIT', groups={}, N=0) if phase == 'canary' else objective.score(mu[primary], device='cpu')
    if physical != 'COMPLETE_NO_RUNAWAY':
        score.update(J=None, status=physical)
    score.update(candidate=cid, core_count=k, restart=restart, epoch=c['epoch'], stage=c['stage'], vector=c['vector'], phase=phase, noise=noise, objective_sha256=sha(OUT / 'objective.pkl'), wall_seconds=time.time() - started)
    write(unit / 'score.json', score)
    payload = dict(status='COMPLETE', job=job, physical_status=physical, actual_duration_ms=actual, arrays_sha256=sha(unit / 'trajectory.npz'), N_primary=len(primary), N_detected=len(mu), input_audit=audit.summary(), events=[] if obs is None else obs['events'], wall_seconds=time.time() - started, peak_rss_gib=rt.peak_rss_gib())
    write(unit / 'result.json', payload)
    if phase == 'train':
        write(d / 'scores' / f'{cid}.json', score)
    write(unit / 'progress.json', dict(status='COMPLETE', job=job, N_primary=len(primary), J=score['J']))
    print(json.dumps(safe(dict(job=job, J=score['J'], physical_status=physical, wall_seconds=payload['wall_seconds']))), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cores', type=int, required=True)
    p.add_argument('--restart', type=int, default=0)
    p.add_argument('--candidate', default='canary')
    p.add_argument('--phase', choices=['train', 'canary', 'confirmation'], default='train')
    p.add_argument('--noise', type=int)
    a = p.parse_args()
    execute(a.cores, a.restart, a.candidate, a.phase, a.noise)
