#!/usr/bin/env python3
"""Native-dt Z/M trajectories and paired checkpoint probes on the new XY graph."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src/snn_engine'):
    sys.path.insert(0, str(path))

import numpy as np
from src.topic4_xy_fig5_followup import (
    array_sha, binned_rates, classify_trajectory, critical_state, make_slow,
    read, region_labels, sha, verify_hashes, verify_lock, write)
from src.topic4_zm_ictal_transition import build_substrate, make_external_drive
from src.snn_engine.kick_probe import simulate_kick
from src.snn_engine.lfp import LFPRecorder
from src.snn_engine.checkpoint import load as load_checkpoint, save as save_checkpoint
from src.sef_hfo_events import detect_events


def build(handoff, job):
    verify_hashes(handoff['input_hashes'])
    candidate = handoff['candidate']
    cfg = read(handoff['transition_config'])
    topo = job['topology_seed']
    mechanism = candidate['mechanisms']
    mapping = candidate['node_mapping']
    substrate = build_substrate(
        cfg, 'joint_04_control', topo, cache_dir=handoff['network_cache'],
        ee_dose=0., etoi_dose=0., node_candidate_override=candidate['node_field'],
        node_depth_shrinkage=mapping['signed_depth_shrinkage'], node_gain=mapping['node_gain'],
        ee_ellipse_angle_deg=mechanism['ellipse_angle_deg'],
        ee_ellipse_aspect_ratio=mechanism['ellipse_aspect_ratio'],
        ee_ellipse_reference_angle_deg=mechanism['ellipse_reference_angle_deg'],
        ee_ellipse_reference_aspect_ratio=mechanism['ellipse_reference_aspect_ratio'],
        artifact_root=Path(handoff['artifact_root']), topology_seed=topo,
        dynamics_seed=job['dynamics_seed'], network_cache_record=handoff['networks'][str(topo)])
    if np.any(substrate.edge_coefficients != 0):
        raise RuntimeError('nonzero learned edge coefficients in the VTH-only Fig5 base')
    if not np.isclose(substrate.params.dt, .1):
        raise RuntimeError('native dt changed')
    regions = region_labels(substrate.positions_e, substrate.h_e, candidate['node_field']['centers_mm'])
    fingerprint = {
        'graph_sha256': handoff['networks'][str(topo)]['sha256'],
        'vtheta_sha256': array_sha(substrate.vtheta),
        'positions_E_sha256': array_sha(substrate.positions_e),
        'h_E_sha256': array_sha(substrate.h_e),
        'region_counts': np.bincount(regions, minlength=3).tolist(),
        'contact_names': list(substrate.contact_names), 'native_dt_ms': float(substrate.params.dt),
        'base_candidate_id': 'joint_04_control', 'learned_ee_etoi_coefficients_zero': True}
    return substrate, cfg, regions, fingerprint


def simulate(substrate, transition, regions, job, *, resume=None, forced_ids=None,
             checkpoint_steps=None, checkpoint_sink=None, duration=None):
    substrate.params.T = float(job['duration_ms'] if duration is None else duration)
    substrate.net['rng'] = np.random.default_rng(job['dynamics_seed'])
    slow = make_slow(substrate, job['config'], regions)
    drive = make_external_drive(substrate, transition['spatial_ou'], job['dynamics_seed'])
    recorder = LFPRecorder(substrate.params, substrate.net['pos'], substrate.net['labels'],
                           sites=substrate.contact_xy)
    offset = 0. if resume is None else float(resume['absolute_time_ms'])
    packet = {}
    if forced_ids is not None:
        mask = np.zeros(substrate.n_e + substrate.n_i, dtype=bool)
        mask[forced_ids] = True
        packet = dict(forced_spike_mask=mask, forced_spike_ms=offset)
    result = simulate_kick(
        substrate.params, substrate.net, KICK_BOOST=0., t_kick=1e9,
        V_th_per_neuron=substrate.vtheta, slow=slow, lfp_recorder=recorder,
        early_stop_runaway=resume is None, es_thresh_hz=300., es_dur_ms=1000.,
        post_runaway_record_ms=1000., external_e_rate_drive=drive,
        checkpoint_steps=checkpoint_steps, checkpoint_sink=checkpoint_sink,
        resume_state=resume, time_offset_ms=offset, **packet)
    if not np.isfinite(result['lfp_trace']).all() or not np.isfinite(slow.z).all() or not np.isfinite(slow.m).all():
        raise RuntimeError('nonfinite simulation state or readout')
    return result, slow


def trajectory_arrays(substrate, regions, result, slow):
    dt = float(substrate.params.dt)
    times, rates = binned_rates(result['E_spk_bool'], regions, dt)
    times += float(result['times'][0])
    trace = {f'slow_{k}': v for k, v in slow.trace_arrays().items()}
    trace.update(slow.region_arrays())
    if any(not np.isfinite(v).all() for v in trace.values()):
        raise RuntimeError('nonfinite slow trace')
    # 1-ms box means retain the raw current proxy separately from rate/slow variables.
    width = int(round(1. / dt))
    n = len(result['lfp_trace']) // width
    lfp = result['lfp_trace'][:n * width].reshape(n, width, -1).mean(axis=1).astype(np.float32)
    arrays = dict(time_ms=times, rates_hz=rates.astype(np.float32), lfp=lfp,
                  lfp_time_ms=result['times'][0] + (np.arange(n) + .5),
                  contact_xy=substrate.contact_xy, contact_names=np.asarray(substrate.contact_names),
                  positions_E=substrate.positions_e.astype(np.float32), region_E=regions,
                  vtheta_E=substrate.vtheta[:substrate.n_e].astype(np.float32), **trace)
    return arrays


def save_arrays(path, arrays):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with open(tmp, 'wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)
    return {'path': str(path), 'sha256': sha(path)}


def trajectory_job(substrate, transition, regions, job):
    result, slow = simulate(substrate, transition, regions, job)
    arrays = trajectory_arrays(substrate, regions, result, slow)
    outcome = classify_trajectory(arrays['time_ms'], arrays['rates_hz'], job['duration_ms'])
    outcome['engine_detection_ms'] = result['runaway_early_stop_ms']
    cmrun = substrate.extras['cmrun']
    active, active_dt = cmrun.active_fraction(result['E_spk_bool'], substrate.params.dt, cmrun.BIN_MS)
    cutoff = (outcome['onset_ms'] - 200. if outcome['onset_ms'] is not None else outcome['observed_ms'])
    events = detect_events(active[:max(0, int(cutoff / active_dt))], active_dt,
                           event_on_frac=substrate.detector_threshold)
    returned = [e for e in events if e['returned'] and e['t_on'] >= 200.]
    outcome['pretransition_returned_population_excursions'] = len(returned)
    outcome['qualified_pretransition'] = (outcome['qualified_pretransition'] and len(returned) >= 2)
    arrays.update(active_fraction=np.asarray(active, dtype=np.float32),
                  active_time_ms=np.arange(len(active)) * active_dt)
    return arrays, {'trajectory': outcome, 'critical_state': critical_state(arrays, outcome['onset_ms']),
                    'population_excursion_diagnostics': events,
                    'excursion_contract': 'Existing active-fraction detector; not causal families or propagation-template labels. Return assessed before onset minus 200 ms; omit first 200 ms.',
                    'rate_E_native_sha256': array_sha(result['rate_E']),
                    'lfp_native_sha256': array_sha(result['lfp_trace']),
                    'external_drive': result['external_e_rate_drive']}


def probe_job(substrate, transition, regions, job, out_dir):
    parent_path = Path(job['parent_result']['path'])
    if sha(parent_path) != job['parent_result']['sha256']:
        raise RuntimeError('probe parent trajectory changed')
    parent = read(parent_path)
    if not parent['trajectory']['qualified_pretransition']:
        raise RuntimeError('probe requires a qualified pretransition trajectory')
    if parent['job']['config'] != job['config']:
        raise RuntimeError('probe mechanism differs from its parent')
    protocol = job['probe']
    dt = float(substrate.params.dt)
    states_ms = {'baseline': protocol['baseline_ms'],
                 'pre_onset': parent['trajectory']['onset_ms'] - protocol['pre_onset_offset_ms']}
    if not states_ms['pre_onset'] > states_ms['baseline']:
        raise RuntimeError('baseline/pre-onset checkpoints overlap')
    steps = {int(round(t / dt)): key for key, t in states_ms.items()}
    checkpoints = {}
    def sink(step, state):
        key = steps[step]
        path = out_dir / 'checkpoints' / f'{job["job_id"]}_{key}.npz'
        digest = save_checkpoint(state, path)
        checkpoints[key] = {'path': str(path), 'sha256': digest,
                            'time_ms': float(state['absolute_time_ms'])}
    result, slow = simulate(substrate, transition, regions, job,
                            checkpoint_steps=steps, checkpoint_sink=sink)
    if (array_sha(result['rate_E']) != parent['rate_E_native_sha256']
            or array_sha(result['lfp_trace']) != parent['lfp_native_sha256']):
        raise RuntimeError('checkpoint replay is not identical to the parent trajectory')
    if set(checkpoints) != set(states_ms):
        raise RuntimeError('missing replay checkpoint')
    arrays = trajectory_arrays(substrate, regions, result, slow)
    del result, slow
    gc.collect()
    sites = np.array([[x, y] for y in protocol['site_coordinates_mm'] for x in protocol['site_coordinates_mm']])
    rows = []
    for state_name, ckpt in checkpoints.items():
        if sha(ckpt['path']) != ckpt['sha256']:
            raise RuntimeError('checkpoint changed')
        state = load_checkpoint(ckpt['path'])
        sham, sham_slow = simulate(substrate, transition, regions, job, resume=state,
                                    duration=protocol['duration_ms'])
        # Retain compact paired reference; the entire sham spike movie is unnecessary.
        sham_counts = sham['E_spk_bool'].sum(axis=0).astype(float)
        sham_rate = sham['rate_E'].copy()
        arrays[f'{state_name}_sham_rate_hz'] = sham_rate.astype(np.float32)
        del sham, sham_slow
        responses = []
        for site_index, site in enumerate(sites):
            distances = ((substrate.positions_e - site)**2).sum(axis=1)
            ids = np.argsort(distances, kind='stable')[:protocol['forced_E_neurons']]
            probe, probe_slow = simulate(substrate, transition, regions, job, resume=state,
                                         forced_ids=ids, duration=protocol['duration_ms'])
            counts = probe['E_spk_bool'].sum(axis=0).astype(float)
            delta = counts - sham_counts
            difference = probe['rate_E'] - sham_rate
            n50 = int(round(50. / dt))
            rows.append({'state': state_name, 'site_index': site_index, 'site_mm': site.tolist(),
                         'checkpoint_sha256': ckpt['sha256'], 'packet_requested_E': len(ids),
                         'packet_collision_E': probe['forced_spike_collision_count'],
                         'packet_ids_sha256': array_sha(ids),
                         'packet_radius_mm': float(np.sqrt(distances[ids].max())),
                         'signed_excess_spikes_per_E': float(delta.mean()),
                         'early_50ms_excess_hz': float(difference[:n50].mean()),
                         'late_50ms_excess_hz': float(difference[-n50:].mean()),
                         'positive_excess_neuron_fraction': float(np.mean(delta > 0)),
                         'negative_excess_neuron_fraction': float(np.mean(delta < 0))})
            responses.append(difference.astype(np.float32))
            del probe, probe_slow
        arrays[f'{state_name}_probe_minus_sham_hz'] = np.asarray(responses)
        del state
        gc.collect()
    arrays['probe_sites_mm'] = sites
    return arrays, {'checkpoints': checkpoints, 'replay_bitwise_equal': True,
                    'probe_rows': rows, 'parent_result': job['parent_result'],
                    'trajectory': parent['trajectory'], 'critical_state': parent['critical_state']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--runtime-lock', type=Path, required=True)
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--memory-gib', type=float, required=True)
    args = parser.parse_args()
    cap = int(args.memory_gib * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    lock = verify_lock(args.runtime_lock)
    job_hash, handoff_hash = sha(args.job), sha(args.handoff)
    job, handoff = read(args.job), read(args.handoff)
    if handoff['runtime_lock_sha256'] != sha(args.runtime_lock):
        raise RuntimeError('handoff uses a different runtime')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    substrate, transition, regions, fingerprint = build(handoff, job)
    if job['kind'] == 'trajectory':
        arrays, details = trajectory_job(substrate, transition, regions, job)
    elif job['kind'] == 'probe':
        arrays, details = probe_job(substrate, transition, regions, job, args.out.parent.parent)
        parent = read(job['parent_result']['path'])
        if fingerprint != parent['substrate_fingerprint']:
            raise RuntimeError('probe substrate differs from parent')
    else:
        raise ValueError('unknown job kind')
    record = save_arrays(args.out.with_suffix('.npz'), arrays)
    verify_lock(args.runtime_lock)
    if sha(args.job) != job_hash or sha(args.handoff) != handoff_hash:
        raise RuntimeError('job or handoff changed during execution')
    write(args.out, {'status': 'FIG5_WORKER_COMPLETE', 'job': job, 'job_sha256': job_hash,
                     'handoff_sha256': handoff_hash, 'runtime_lock_sha256': sha(args.runtime_lock),
                     'substrate_fingerprint': fingerprint, 'arrays': record, **details,
                     'wall_seconds': time.time() - started,
                     'max_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                     'claim_boundary': lock['claim_boundary']})
    print(f'Completed {job["job_id"]}', flush=True)


if __name__ == '__main__':
    main()
