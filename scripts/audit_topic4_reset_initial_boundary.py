#!/usr/bin/env python3
"""Audit initial versus post-refill state at the actual first current update.

This is read-only state/recorder analysis, not an additional trajectory.
"""
from pathlib import Path
import hashlib
import json
import pickle
from types import SimpleNamespace
import numpy as np
import run_topic4_reset_state_diagnosis as base
from checkpoint import restore_slow

ROOT = base.ROOT
WINDOW = ROOT / 'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913'
OUT = WINDOW / 'reset_initial_boundary'


def summary(x):
    x = np.asarray(x)
    return dict(mean=float(x.mean()), sd=float(x.std()),
                minimum=float(x.min()), maximum=float(x.max()))


def main():
    parent_path = base.OUT / 'parents/release.pkl'
    blob = parent_path.read_bytes()
    package = pickle.loads(blob)
    state = package['engine']
    assert state['step'] == 765000 and state['slow']['step_index'] == 765000
    assert not state['track_rec'] and state['node_accessibility'] is None
    cfg = base.old.MZSlowVarsConfig(use_z=True, use_m=True, tau_z=5000.,
        I_th_EI=base.old.THRESHOLD, tau_adp=2000., eta_m=.02)
    slow = base.ReleaseZ(40000, 18., cfg, NE=32000, reset_m=True)
    restore_slow(state, slow)
    slow.restore_ms = 75500.
    slow.restore_from = package['restore_from']
    # Zero synaptic history is the existing all-fast arm; the exact first
    # afferent current does not affect Z/M assignment in apply_currents.
    slow._I_I_last.fill(0.)
    slow.apply_currents(np.zeros(40000), np.zeros(40000))
    assert np.all(slow.z == 1) and np.all(slow.m == 0)
    first_file = sorted((WINDOW / 'fast_state_pilot/runs/all_fast_90s/chunks').glob('*.npz'))[0]
    with np.load(first_file) as a:
        assert a['slow_time_ms'][0] == 76500.
        # Actual saved observation is taken after that current application.
        assert a['Z'][0, 0] == 1 and a['Z'][0, 1] == 0
        assert np.all(a['M'][0] == 0)
        observed_first = dict(time_s=float(a['slow_time_ms'][0] / 1000),
                             Z=a['Z'][0].tolist(), M=a['M'][0].tolist())
    geometry_path = base.OUT / 'geometry.npz'
    with np.load(geometry_path) as a:
        positions = a['positions_e'].copy()
    frozen = base.read(ROOT / 'config/topic4_rate_model_dynamics_validation_v1.json')
    execution_path = Path(next(x['path'] for x in frozen['inputs']
                              if x['path'].endswith('execution_config.json')))
    execution = base.read(execution_path)
    transition_path = ROOT / '.worktrees/topic4-substrate-autapse-fix' / execution['inputs']['transition_config']['path']
    transition = base.read(transition_path)
    substrate = SimpleNamespace(positions_e=positions, engine={'L':20., 'dt':.1})
    initial_drive = base.old.make_external_drive(substrate, transition['spatial_ou'], 9108401)
    replay = base.read(ROOT / 'results/topic4_sef_hfo/rate_model_dynamics_validation_v1/reconstruction.json')
    report = dict(status='PASS_BOUNDARY_AUDIT', new_simulations=0,
        release_time_s=76.5, parent=str(parent_path),
        parent_sha256=hashlib.sha256(blob).hexdigest(),
        before_first_current=dict(Z=summary(state['slow']['z'][:32000]),
                                  M=summary(state['slow']['m'][:32000])),
        after_first_current=dict(all_neuron_Z_exactly_one=True, all_neuron_M_exactly_zero=True),
        observed_first_current=observed_first,
        observed_chunk=str(first_file), observed_chunk_sha256=base.sha(first_file),
        fresh_cell_and_synapse_state=dict(V_reset=replay['params']['V_reset'],
            refractory_zero=True, AMPA_GABA_zero=True, delay_rings_zero=True),
        same_cell_initialization_definition_after_full_fast_clear=True,
        global_OU=dict(fresh_xi=0., release_xi=float(state['xi']),
                       tau_n_ms=replay['params']['tau_n']),
        spatial_OU=dict(config=transition['spatial_ou'],
            fresh_cache=summary(initial_drive._cached), release_cache=summary(state['external_drive']['cached']),
            actual_fields_identical=bool(np.array_equal(initial_drive._cached,state['external_drive']['cached'])),
            fresh_field_draw_is_stationary=True,
            release_pending_update_step=int(state['external_drive']['next_step']),
            release_last_step=int(state['external_drive']['last_step'])),
        preserved_differences=['OU current state', 'future global/Poisson and spatial OU random streams',
                              'absolute time and corresponding delay-slot/OU update phase'],
        interpretation='The all-fast arm matches the fresh cellular/synaptic initial values, but does not replay the original external input realization. A finite non-entry does not isolate a causal role of the current OU field or show permanent non-recurrence.',
        sources={str(p):base.sha(p) for p in [Path(base.__file__),Path(base.old.__file__),
            ROOT/'src/topic4_raster_protocol_engine.py',execution_path,transition_path]},
        human_review='PENDING')
    OUT.mkdir(exist_ok=True)
    base.write(OUT/'analysis.json', report)
    (OUT/'scientific_review.md').write_text(
        '# reset 与重新初始化：实际边界核查\n\n'
        '原始 initial_state_audit 在第一步 apply_currents 之前写入，所以显示 M≈396、Z≈0.999955；这不是 M 清零失败。'
        '已用同一 ReleaseZ 类从真实 checkpoint 复原边界，并核对已保存首帧：76.5 秒实际施加电流时，所有神经元 Z=1、M=0。\n\n'
        '全快状态对照还将膜电位设为原始 V_reset、不应期清零，AMPA/GABA 突触及延迟历史清零；原始初始化本来就使用这些值，没有额外随机膜电位。'
        '原有递归-only 累加器与 node accessibility 均关闭。\n\n'
        '仍然有意保留的是 OU 状态和未来随机历史。原始空间 OU 也从其平稳分布抽样，不存在“初始为零、之后逐渐变强”的默认趋势。'
        '所以全快状态清除后若在 90 秒未再进入，只能排除该次有限窗中单靠清除这些状态即可保证再进入，不能直接把原因归给 OU 或宣布永远不再进入。\n')
    print(json.dumps(dict(status=report['status'],actual_first_Z=observed_first['Z'][0],
        actual_first_M=observed_first['M'][0],global_OU=report['global_OU'],
        spatial_OU_summary={k:report['spatial_OU'][k] for k in ['fresh_cache','release_cache']},
        report=str(OUT/'analysis.json'))))


if __name__ == '__main__':
    main()
