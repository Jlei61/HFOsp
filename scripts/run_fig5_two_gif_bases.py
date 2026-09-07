#!/usr/bin/env python3
"""Two fixed GIF substrates: inherited Z/M reference and exact paired probes.

No substrate search. Failure to reach runaway is retained as an outcome.
"""
import argparse
import gc
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import run_topic4_xy_fig5_worker as old
from src.topic4_xy_fig5_followup import read, write, sha, array_sha
from src.topic4_multidimensional_parameters import apply_parameters
from src.topic4_node_dualmode import sheet_activity_movie
from src.topic4_forced_source_capacity import exclude_injected_packet_frame

SOURCE = ROOT / 'results/topic4_sef_hfo/multidimensional_interictal_pilot_round1'
OUTPUT = ROOT / 'results/topic4_sef_hfo/fig5_two_gif_bases'
IDS = ['support_rank__vth_low', 'old_joint__tau_d_GABA_ms_high']


def run(cid):
    out = OUTPUT / cid
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    def status(stage, **extra):
        write(out / 'status.json', dict(candidate_id=cid, stage=stage,
              elapsed_s=time.time()-started, **extra))
    config_path = SOURCE / 'execution/paired_round1/execution_config.json'
    config = read(config_path)
    manifest = read(config['candidate_manifest'])
    candidate = next(c for c in manifest['candidates'] if c['candidate_id'] == cid)
    parent_path = SOURCE / f'execution/paired_round1/workers/{cid}_seed_2511.json'
    parent = read(parent_path)
    transition_path = ROOT / config['inputs']['transition_config']['path']
    handoff = dict(candidate=candidate, transition_config=str(transition_path),
        artifact_root='/home/honglab/leijiaxin/HFOsp', network_cache=config['network_cache'],
        networks=config['corrected_networks'], input_hashes={
            str(transition_path):config['inputs']['transition_config']['sha256']})
    reference = read(ROOT / 'config/topic4_xy_round1_fig5_followup.json')['zm_reference']
    job = dict(job_id=cid, topology_seed=2511, dynamics_seed=2511,
               duration_ms=12000., config=reference)
    write(out / 'protocol.json', dict(candidate=candidate, job=job,
        parent_result=dict(path=str(parent_path), sha256=sha(parent_path)),
        comparison='Two different complete substrates, not a single-factor comparison.',
        state_rule='Reference at 1000 ms; pre-onset = qualified onset minus 250 ms. If no qualified onset, later reference at 10000 ms, explicitly not pre-onset.',
        pulse='16 nearest E cells, same 3x3 sites at x,y = 4,10,16 mm; 200 ms paired sham; injected frame excluded.',
        scientific_role='Exploratory single fixed graph and noise seed; no clinical seizure or accepted patient fit claim.'))
    status('BUILDING_AND_VERIFYING_SUBSTRATE')
    substrate, transition, regions, fingerprint = old.build(handoff, job)
    audit = apply_parameters(substrate, candidate['dynamic_parameters'])
    if audit != parent['multidimensional_parameter_audit']:
        # JSON stores numpy bools as 0/1; Python dictionary equality handles those.
        raise RuntimeError('effective multidimensional substrate differs from GIF parent')
    fingerprint['dynamic_parameter_audit'] = audit
    off = dict(job, duration_ms=100., config=dict(reference, use_z=False, use_m=False))
    result, slow = old.simulate(substrate, transition, regions, off)
    movie = sheet_activity_movie(result['E_spk_bool'], substrate.positions_e,
        dt_ms=.1, frame_ms=2., bin_mm=1., sheet_mm=20.)['activity_counts']
    with np.load(parent['arrays']['path']) as original:
        matched = np.array_equal(movie, original['sheet_activity_counts'][:len(movie)])
    if not matched:
        raise RuntimeError('Z/M-off first 100 ms does not reproduce original GIF sheet movie')
    write(out / 'substrate_verification.json', dict(fingerprint=fingerprint,
        dynamic_parameters_exact=True, original_sheet_prefix_bitwise_equal=True,
        prefix_duration_ms=100., parent_arrays=parent['arrays']))
    del result, slow, movie
    gc.collect()
    status('RUNNING_ZM_REFERENCE', original_sheet_prefix_bitwise_equal=True)
    arrays, detail = old.trajectory_job(substrate, transition, regions, job)
    record = dict(candidate_id=cid, job=job, substrate_fingerprint=fingerprint, **detail)
    record['arrays'] = old.save_arrays(out / 'trajectory.npz', arrays)
    write(out / 'trajectory.json', record)
    onset = record['trajectory']['onset_ms']
    qualified = record['trajectory']['qualified_pretransition']
    states = {'reference':1000.}
    if qualified:
        states['pre_onset'] = onset - 250.
    elif record['trajectory']['observed_ms'] >= 10200.:
        states['later_reference'] = 10000.
    else:
        # Do not create a pre-onset label for an immediate runaway.
        write(out / 'probe_unavailable.json', dict(reason='No sufficiently long pretransition or later reference', trajectory=record['trajectory']))
        status('COMPLETE_NO_VALID_STATE_COMPARISON')
        return
    status('REPLAYING_CHECKPOINTS', trajectory=record['trajectory'])
    steps = {int(round(t/.1)):key for key,t in states.items()}
    checkpoints = {}
    def sink(step, state):
        key = steps[step]
        path = out / f'checkpoint_{key}.npz'
        checkpoints[key] = dict(path=str(path), sha256=old.save_checkpoint(state,path))
    replay, slow = old.simulate(substrate, transition, regions, job,
        checkpoint_steps=steps, checkpoint_sink=sink)
    if array_sha(replay['rate_E']) != record['rate_E_native_sha256'] or array_sha(replay['lfp_trace']) != record['lfp_native_sha256']:
        raise RuntimeError('checkpoint replay differs from parent')
    replay_rate = replay['rate_E'].copy()
    replay_lfp = replay['lfp_trace'].copy()
    del replay, slow
    gc.collect()
    sites = np.array([[x,y] for y in [4.,10.,16.] for x in [4.,10.,16.]])
    probes = dict(sites_mm=sites, positions_E=substrate.positions_e,
                  contact_xy=substrate.contact_xy, region_E=regions)
    rows = []
    for name, t in states.items():
        state = old.load_checkpoint(checkpoints[name]['path'])
        sham, slow = old.simulate(substrate, transition, regions, job, resume=state, duration=200.)
        begin = int(round(t/.1))
        if not np.array_equal(sham['rate_E'],replay_rate[begin:begin+2000]) or not np.array_equal(sham['lfp_trace'],replay_lfp[begin:begin+2000]):
            raise RuntimeError('sham continuation differs from parent')
        del slow
        fields, curves, early = [], [], []
        for index, site in enumerate(sites):
            status('RUNNING_PAIRED_PROBES', state=name, site=index, trajectory=record['trajectory'])
            ids = np.argsort(((substrate.positions_e-site)**2).sum(axis=1),kind='stable')[:16]
            probe, slow = old.simulate(substrate,transition,regions,job,
                resume=state,forced_ids=ids,duration=200.)
            packet = np.zeros(substrate.n_e,bool); packet[ids] = True
            descendant = exclude_injected_packet_frame(probe['E_spk_bool'],sham['E_spk_bool'],packet,trigger_step=0)
            full = descendant.sum(axis=0,dtype=np.int32)-sham['E_spk_bool'].sum(axis=0,dtype=np.int32)
            first = descendant[:500].sum(axis=0,dtype=np.int32)-sham['E_spk_bool'][:500].sum(axis=0,dtype=np.int32)
            count = descendant.sum(axis=1,dtype=np.int32)-sham['E_spk_bool'].sum(axis=1,dtype=np.int32)
            curves.append(count.reshape(200,10).sum(axis=1))
            fields.append(full); early.append(first)
            rows.append(dict(state=name,site_index=index,extra_spikes_200ms=int(full.sum()),
                             extra_spikes_50ms=int(first.sum()),collision_count=int(probe['forced_spike_collision_count'])))
            del probe,slow,descendant
        probes[name+'_full_field'] = np.asarray(fields)
        probes[name+'_early_field'] = np.asarray(early)
        probes[name+'_extra_spikes_per_ms'] = np.asarray(curves)
        del sham,state
        gc.collect()
    write(out/'probe.json', dict(candidate_id=cid, states_ms=states, rows=rows,
        arrays=old.save_arrays(out/'probe.npz',probes), replay_exact=True, sham_continuation_exact=True,
        checkpoints=checkpoints, injected_frame_excluded=True, all_sites_retained=True,
        qualified_pretransition=qualified, low_reference_not_required_quiescent=True))
    status('COMPLETE', trajectory=record['trajectory'])


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',choices=IDS,required=True)
    a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
    try:
        run(a.candidate)
    except Exception as exc:
        write(OUTPUT/a.candidate/'failure.json',dict(error=repr(exc)))
        raise
