"""Scientific handoff gates, censoring, memory admission and native replay parity."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src/snn_engine'):
    sys.path.insert(0, str(p))
from src.topic4_xy_fig5_followup import (
    Fig5RegionSlowVars, admission_slots, binned_rates, choose_development_candidate,
    classify_trajectory, parameter_grid, read, write)
from src.snn_engine.mz_slow_vars import MZSlowVars, MZSlowVarsConfig


def reports():
    seeds = list(range(10, 16))
    chosen = dict(candidate_id='a', domain='interior', selection_eligible=True, J_round1=1.)
    other = dict(chosen, candidate_id='b', J_round1=2.)
    units = [dict(seed=s, runaway=False, J_direction=2.) for s in seeds]
    a = dict(chosen, units=units, all_components_estimable=True, n_returned=30)
    b = dict(other, units=copy.deepcopy(units), all_components_estimable=True, n_returned=30, J_round1=.1)
    nominees = {'candidates': [dict(r, mechanisms={'g_EE': 0., 'g_EtoI': 0., 'Z_M': 'off'},
                                  node_field={'target_count': 1499}) for r in (chosen, other)]}
    return {'candidates': [other, chosen]}, {'candidates': [b, a]}, nominees, seeds


def test_choose_before_confirmation_even_when_other_has_better_confirmation():
    selection, confirm, nominees, seeds = reports()
    candidate, _, _ = choose_development_candidate(selection, confirm, nominees, seeds)
    assert candidate['candidate_id'] == 'a'


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'runaway', 'nonfinite', 'ineligible'])
def test_reject_failed_confirmation_without_falling_back(failure):
    selection, confirm, nominees, seeds = reports()
    a = confirm['candidates'][1]
    if failure == 'missing': a['units'].pop()
    if failure == 'duplicate': a['units'][-1]['seed'] = a['units'][0]['seed']
    if failure == 'runaway': a['units'][0]['runaway'] = True
    if failure == 'nonfinite': a['units'][0]['J_direction'] = float('nan')
    if failure == 'ineligible': a['selection_eligible'] = False
    with pytest.raises(ValueError, match='NO_RESELECTION'):
        choose_development_candidate(selection, confirm, nominees, seeds)


def test_reject_mechanism_drift():
    selection, confirm, nominees, seeds = reports()
    nominees['candidates'][0]['mechanisms']['g_EE'] = 1.25
    with pytest.raises(RuntimeError, match='VTH-only'):
        choose_development_candidate(selection, confirm, nominees, seeds)


def test_tonic_regional_gate_and_censoring():
    t = (np.arange(250) + .5) * 20
    rates = np.ones((250, 4)) * 2
    rates[150:] = [400, 420, 420, 398]
    got = classify_trajectory(t, rates, 5000)
    assert got['runaway'] and got['onset_ms'] == 3000 and got['qualified_pretransition']
    rates[:, 3] = 100
    got = classify_trajectory(t, rates, 5000)
    assert got['right_censored'] and got['onset_ms'] is None
    assert classify_trajectory(t[:100], rates[:100], 5000)['classification'] == 'EARLY_STOP_UNRESOLVED'


def test_transient_plateau_followed_by_return_is_not_terminal_tonic():
    t = (np.arange(250) + .5) * 20
    rates = np.ones((250, 4)) * 2
    rates[80:160] = 400
    assert not classify_trajectory(t, rates, 5000)['runaway']


def test_rate_units_and_regions():
    spikes = np.zeros((40, 6), dtype=bool)
    spikes[:, 0] = True
    regions = np.array([0, 0, 1, 1, 2, 2])
    t, rates = binned_rates(spikes, regions, 1., 20.)
    np.testing.assert_allclose(t, [10, 30])
    np.testing.assert_allclose(rates[0], [1000/6, 500, 0, 0])


def test_memory_admission_reserves_future_growth_and_never_forces_one_worker():
    assert admission_slots(39, 40, 25, [], 24) == 0
    assert admission_slots(100, 40, 25, [5, 5], 24) == 0
    assert admission_slots(150, 40, 25, [5, 5], 24) == 2
    assert admission_slots(250, 40, 25, [25]*24, 24) == 0


def test_full_parameter_design_includes_fixed_reference_and_controls():
    rows = parameter_grid(read(ROOT / 'config/topic4_xy_round1_fig5_followup.json'))
    assert len(rows) == 18 and len({r['config_id'] for r in rows}) == 18
    assert rows[0]['tau_z'] == 5000 and rows[0]['I_th_EI'] == 95.19851312666987
    assert not rows[-2]['use_z'] and not rows[-2]['use_m']
    assert not rows[-1]['use_z'] and rows[-1]['use_m']


def test_recorder_does_not_change_spatial_ZM_dynamics():
    cfg = MZSlowVarsConfig(use_z=True, use_m=True, I_th_EI=20, tau_z=50, tau_adp=10,
                          eta_m=.2, trace_stride_steps=2)
    region = np.tile([0, 1, 2], 4)
    original = MZSlowVars(16, 18, cfg, NE=12, core_mask_E=region < 2)
    traced = Fig5RegionSlowVars(16, 18, cfg, NE=12, core_mask_E=region < 2, region_E=region)
    rng = np.random.default_rng(42)
    for _ in range(300):
        e, i = rng.random((2, 16)) * 100
        spikes = rng.random(16) < .2
        np.testing.assert_array_equal(original.apply_currents(e, i), traced.apply_currents(e, i))
        original.step(spikes, None, .1); traced.step(spikes, None, .1)
    np.testing.assert_array_equal(original.z, traced.z)
    np.testing.assert_array_equal(original.m, traced.m)
    assert traced.region_arrays()['region_z'].shape == (150, 3)


def test_full_native_engine_new_recorder_and_checkpoint_sham_packet_parity(tmp_path):
    from params import Params
    from connectivity import place_neurons, build_connectivity
    from src.snn_engine.kick_probe import simulate_kick
    from src.snn_engine.lfp import LFPRecorder
    from src.snn_engine.checkpoint import save, load
    from src.topic4_spatial_ou_drive import SpatialOUDrive, SpatialOUConfig
    p = Params(L=1., density=120., T=40., dt=.1, seed=42, nu_ext_ratio=1.)
    rng = np.random.default_rng(42)
    pos, labels, ne, ni = place_neurons(p, rng)
    net = build_connectivity(p, pos, labels, ne, ni, rng, verbose=False)
    region = np.arange(ne) % 3
    cfg = MZSlowVarsConfig(use_z=True, use_m=True, I_th_EI=20, tau_z=50, tau_adp=10,
                          eta_m=.02, trace_stride_steps=10)
    def run(traced=True, resume=None, packet=False, sink=None):
        net['rng'] = np.random.default_rng(73)
        slow = (Fig5RegionSlowVars(ne+ni, p.V_th, cfg, NE=ne, core_mask_E=region < 2, region_E=region)
                if traced else MZSlowVars(ne+ni, p.V_th, cfg, NE=ne, core_mask_E=region < 2))
        drive = SpatialOUDrive(pos[:ne], p.L, p.dt, SpatialOUConfig(mode='local',
                    sigma_rate_per_ms=.1, tau_ms=20, ell_mm=.38, seed=573))
        recorder = LFPRecorder(p, pos, labels, sites=np.array([[.3, .3], [.7, .7]]))
        extra = {}
        if packet:
            mask = np.zeros(ne+ni, dtype=bool); mask[:4] = True
            extra = dict(forced_spike_mask=mask, forced_spike_ms=20.)
        return simulate_kick(p, net, 0., t_kick=1e9, slow=slow, lfp_recorder=recorder,
                             external_e_rate_drive=drive, resume_state=resume,
                             time_offset_ms=0. if resume is None else 20.,
                             checkpoint_steps=[200] if sink else None, checkpoint_sink=sink, **extra)
    original = run(traced=False)
    checkpoint = tmp_path / 'state.npz'
    whole = run(sink=lambda step, state: save(state, checkpoint))
    np.testing.assert_array_equal(original['E_spk_bool'], whole['E_spk_bool'])
    np.testing.assert_array_equal(original['lfp_trace'], whole['lfp_trace'])
    p.T = 20.
    state = load(checkpoint)
    resumed = run(resume=state)
    np.testing.assert_array_equal(whole['E_spk_bool'][200:], resumed['E_spk_bool'])
    np.testing.assert_array_equal(whole['lfp_trace'][200:], resumed['lfp_trace'])
    packet1 = run(resume=state, packet=True)
    packet2 = run(resume=state, packet=True)
    np.testing.assert_array_equal(packet1['E_spk_bool'], packet2['E_spk_bool'])
    assert packet1['E_spk_bool'][0, :4].all()


def test_wait_does_not_launch_on_early_final_report(tmp_path):
    from scripts.chain_topic4_xy_round1_fig5 import wait_upstream
    upstream, out = tmp_path / 'search', tmp_path / 'fig5'
    write(upstream / 'status.json', {'status': 'RUNNING', 'phase': 'confirmation', 'complete': 1, 'total': 12})
    write(upstream / 'final_search_report.json', {'status': 'ROUND1_COMPLETE_AWAITING_SCIENTIFIC_REVIEW'})
    lock = tmp_path / 'lock.json'; write(lock, {'hashes': {}})
    assert not wait_upstream(upstream, out, {'upstream_service': 'nonexistent-test-unit'}, lock, once=True)
    assert read(out / 'status.json')['status'] == 'WAITING_UPSTREAM'


def test_completed_handoff_validates_all_units_and_resolves_local_config(tmp_path, monkeypatch):
    import scripts.chain_topic4_xy_round1_fig5 as chain
    from src.topic4_xy_fig5_followup import COMPLETE, sha
    search, out = tmp_path / 'search', tmp_path / 'fig5'
    local, artifact = tmp_path / 'local', tmp_path / 'artifact'
    monkeypatch.setattr(chain, 'ROOT', local)
    monkeypatch.setattr(chain, 'ART', artifact)
    selection, confirmation, nominees, seeds = reports()
    networks = {}
    for seed in seeds:
        graph = tmp_path / f'graph_{seed}.bin'; graph.write_bytes(str(seed).encode())
        networks[str(seed)] = {'path': str(graph), 'sha256': sha(graph), 'status': 'CORRECTED_GRAPH_VALIDATED',
                              'pathways': {'E_to_E': {'self_edges': 0, 'exact_expected_degree': True}}}
    # The input exists only in the worktree, matching the actual transition config.
    transition = local / 'config/transition.json'; write(transition, {'fixture': True})
    config = {'inputs': {'transition_config': {'path': 'config/transition.json', 'sha256': sha(transition)}},
              'corrected_networks': networks, 'network_cache': str(tmp_path / 'cache')}
    cp = search / 'confirmation/execution_config.json'; write(cp, config)
    mp = search / 'confirmation/candidate_manifest.json'
    write(mp, {'config_sha256': sha(cp), 'candidates': nominees['candidates']})
    sp = search / 'confirmation/runtime_snapshot.json'
    write(sp, {'input_hashes': {str(cp): sha(cp), str(mp): sha(mp)}})
    for row in confirmation['candidates']:
        for unit in row['units']:
            jp = search / 'confirmation/workers' / f'{row["candidate_id"]}_seed_{unit["seed"]}.json'
            ap = jp.with_suffix('.npz'); ap.parent.mkdir(parents=True, exist_ok=True); ap.write_bytes(b'fixture')
            write(jp, {'status': 'REV12ND_NODE_WORKER_COMPLETE', 'arrays': {'path': str(ap), 'sha256': sha(ap)},
                       'provenance': {'source_hash_snapshot': {'sha256': sha(sp)}}})
            unit.update(worker_sha256=sha(jp), graph_sha256=networks[str(unit['seed'])]['sha256'])
    write(search / 'selection/aggregate.json', selection)
    write(search / 'confirmation/aggregate.json', confirmation)
    nominees['selection_complete_before_confirmation'] = True
    write(search / 'confirmation_nominees.json', nominees)
    write(search / 'confirmation/completion.json', {'status': 'ALL_WORKERS_COMPLETE', 'jobs': 12})
    write(search / 'round1_relative_position_contract.json', {'fixture': True})
    write(search / 'final_search_report.json', {'status': COMPLETE, 'confirmation': confirmation,
          'round_contract_sha256': sha(search / 'round1_relative_position_contract.json')})
    write(search / 'status.json', {'status': COMPLETE})
    lock = out / 'runtime_lock.json'; write(lock, {'hashes': {}})
    plan = {'confirmation_seeds': seeds, 'analysis_topology_seeds': seeds[:3],
            'candidate_rule': 'fixture', 'claim_boundary': 'fixture'}
    handoff = chain.make_handoff(search, out, plan, lock)
    result = read(handoff)
    assert result['candidate']['candidate_id'] == 'a'
    assert result['transition_config'] == str(transition)
    assert not result['final_substrate_frozen'] and not result['author_acceptance']
    assert chain.make_handoff(search, out, plan, lock) == handoff
    cp.write_text('{}')
    with pytest.raises(RuntimeError, match='changed'):
        chain.make_handoff(search, out, plan, lock)
