#!/usr/bin/env python3
"""Extend the exact baseline checkpoint to the prespecified PRE-onset window."""
from pathlib import Path
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_topic4_rev21_fig5_checkpoints import (
    load_selected_contract, build_selected_substrate, sha256)
from scripts.run_topic4_zm_perturbation_worker import _continue
from src.snn_engine import checkpoint as ckpt
from src.topic4_core_field_runner import atomic_write_json


def main():
    start = time.time()
    art = Path('/home/honglab/leijiaxin/HFOsp')
    base = art / 'results/topic4_sef_hfo/data_driven_dual_core_zm_transition'
    perturb = Path('/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z/perturbation')
    source = base / 'timescale/workers/rev21_ts_tz3000_ta500_topology_2542_dynamics_2642.json'
    config, candidate, meta, npz = load_selected_contract(
        ROOT/'config/topic4_rev21_dual_core_zm_transition.json',
        base/'timescale/candidate_manifest.json', source, art)
    manifest_path = perturb/'checkpoints/checkpoint_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    baseline = manifest['checkpoints']['low_activity']
    if sha256(Path(baseline['path'])) != baseline['sha256']:
        raise RuntimeError('baseline checkpoint changed')
    transition, substrate = build_selected_substrate(
        config, candidate, topology_seed=2542, dynamics_seed=2642, artifact_root=art)
    state = ckpt.load(baseline['path'])
    target = float(meta['model_ictal_rev21']['landmarks']['w_pre_ms'][0])
    offset = float(state['absolute_time_ms'])
    dt = float(substrate.engine['dt'])
    step = int(round(target/dt))
    captured = {}
    replay, _ = _continue(
        substrate, transition, state, duration_ms=target-offset+dt,
        checkpoint_steps=[step],
        checkpoint_sink=lambda i,s: captured.setdefault(i,s))
    with np.load(npz, allow_pickle=False) as a:
        ref = a['transition_rate_E_hz_raw'][int(round(offset/dt)):int(round(offset/dt))+len(replay['rate_E'])]
    if not np.array_equal(np.asarray(replay['rate_E'], np.float32), ref):
        raise RuntimeError('continued source trajectory is not bit-exact')
    if step not in captured:
        raise RuntimeError('pre-onset checkpoint not captured')
    out = perturb/'preonset_20260905/checkpoints'
    out.mkdir(parents=True, exist_ok=True)
    path = out/'dualcore_rev21_pre_onset.npz'
    digest = ckpt.save(captured[step], path)
    manifest['checkpoints']['pre_onset'] = {
        'path':str(path), 'sha256':digest, 'time_ms':target}
    manifest['state_contract']['pre_onset'] = (
        'start of the previously frozen pre-onset window, 500 ms before scientific onset; '
        'chosen before observing any perturbation response')
    manifest['parent_manifest'] = {'path':str(manifest_path), 'sha256':sha256(manifest_path)}
    manifest['pre_onset_replay_exact'] = True
    manifest['wall_seconds_pre_onset_extension'] = time.time()-start
    atomic_write_json(manifest, str(out/'checkpoint_manifest.json'))
    print(json.dumps({'status':'PRE_ONSET_CHECKPOINT_EXACT', 'time_ms':target,
                      'manifest':str(out/'checkpoint_manifest.json'), 'seconds':time.time()-start}), flush=True)


if __name__ == '__main__':
    main()
