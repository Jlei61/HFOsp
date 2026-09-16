from __future__ import annotations

import numpy as np
import torch
import pytest

from scripts.audit_group_event_state_v038_dual_credit import bin_gradient, event_query
from src.topic5_group_event_state.v037.ctssm import DualStreamEventCTSSM
from src.topic5_group_event_state.v034_spatial_state.we_decoder import per_event_scores
from src.topic5_wiring_economy_rnn import build_event_tensors
from scripts.finalize_group_event_state_v038 import _joint_contrasts, _h2b, _h1_family, _adapter_qualified, _fixed_endpoint_pair_evidence, _credit, ENDPOINTS
from scripts.finalize_group_event_state_v038 import _floored_control, _control_harm, _code_provenance_audit


def test_harmful_controls_are_floored_before_seed_aggregation():
    assert _floored_control(control_gain=.55, baseline_gain=.122) == .122
    assert _floored_control(control_gain=.036, baseline_gain=.489) == .036
    assert _floored_control(control_gain=.55, baseline_gain=-.1) == -.1
    assert _control_harm(control_gain=.55, baseline_gain=.122)
    for missing in (None, float('nan'), float('inf')):
        assert _floored_control(control_gain=missing, baseline_gain=.2) is None
        assert _floored_control(control_gain=.2, baseline_gain=missing) is None
        assert _control_harm(control_gain=missing, baseline_gain=.2) is None


def test_provenance_separates_queue_splits_from_within_patient_seed_mixtures():
    def card(scope, seed, sha):
        return dict(subject='patient', seed=seed, _audit_scope=scope, _audit_family='event',
                    code_provenance=dict(source_file='trainer.py', source_sha256=sha))
    result = _code_provenance_audit([card('short', 1, 'a'), card('long', 1, 'b'), card('long', 2, 'c'),
                                   dict(subject='missing', status='NOT_ESTIMABLE')])
    assert result['cards_without_code_provenance'] == 1
    assert result['missing_cards'][0]['status'] == 'NOT_ESTIMABLE'
    groups = {row['scope']: row for row in result['within_comparison_versions']}
    assert groups['long']['mixed_versions']
    assert not groups['short']['mixed_versions']
    assert groups['long']['seeds_by_sha256'] == {'b': [1], 'c': [2]}


def test_marginal_seed_majorities_do_not_form_complete_candidate():
    cards = [{'seed': i, 'primary_contrasts': dict(zip(('a', 'b', 'c'), values))}
             for i, values in enumerate(((1, 1, -1), (1, 1, 1), (-1, -1, 1), (1, 1, -1), (-1, -1, 1)))]
    assert _joint_contrasts(cards, ('a', 'b', 'c'))['passing_seeds'] == [1]
    assert not _joint_contrasts(cards, ('a', 'b', 'c'))['stable_positive_3_of_5']
    with pytest.raises(ValueError, match='duplicate seed'):
        _joint_contrasts(cards + [cards[0]], ('a',))


def test_h2b_requires_complete_same_seed_risk_contrasts():
    keys = ('event_only_state_gain_over_mark_history', 'event_dynamic_gain_over_fit_period_mean',
            'event_correct_time_gain_over_shift')
    cards = [{'subject': 'test', 'seed': 20260903 + i,
              'distance_survival': {'support': {'status': 'ESTIMATED', 'seizures_by_phase': {'SELECTION': 3}}},
              'primary_contrasts': dict(zip(keys, values))}
             for i, values in enumerate(((1, 1, -1), (1, 1, 1), (-1, -1, 1), (1, 1, -1), (-1, -1, 1)))]
    result = _h2b(cards)['rows'][0]
    assert result['risk_joint_evidence']['event']['passing_seeds'] == [20260904]
    assert not result['any_repeatable_cross_task_candidate']


def test_h1_cannot_win_against_budget_exhausted_random_control():
    def score(value):
        return {'total': value, 'endpoints': {name: value for name in ENDPOINTS}}
    cards = []
    for i in range(5):
        stages = {name: {'first_step_gradient_norm': 1.0, 'peak_parameter_delta_from_stage_start': 1.0,
                         'training_budget_exhausted': name == 'random' and i < 3}
                  for name in ('q', 'bmark', 'state', 'random')}
        cards.append({'subject': 'test', 'seed': 20260903 + i, 'horizons_seconds': [1800], 'stages': stages,
                      'selection_scores': {name: {'by_horizon': {'1800': score(value)}}
                                           for name, value in [('B_mark', 2), ('S_event', 1), ('S_event_constant', 2), ('random_frozen', 2)]},
                      'time_shift_by_horizon': {'1800': {'gain': 1, 'correct': score(1), 'shifted': score(2)}},
                      'independent_windows_by_horizon': {'1800': {'independent_windows': 3}}})
    row = _h1_family(cards, 'event', {})['rows'][0]
    assert row['joint_evidence']['n_joint_positive'] == 2
    assert not row['directional_dynamic_state_candidate']
    for card in cards:
        card['stages']['random']['training_budget_exhausted'] = False
    assert _h1_family(cards, 'event', {})['rows'][0]['directional_dynamic_state_candidate']
    # Predictive improvements from initialized filters cannot establish a
    # learned event observer when lineage/replay evidence is absent.
    assert not _h1_family(cards, 'event', {})['rows'][0]['learned_event_dynamic_state_candidate']


def test_adapter_budget_edge_requires_completed_original_patience():
    card = {'stages': {'state': {'epochs_run': 120, 'selected_epoch': 110, 'patience_epochs': 40}}}
    assert not _adapter_qualified(card, 'state')
    card['stages']['state']['selected_epoch'] = 80
    assert _adapter_qualified(card, 'state')


def test_dynamic_dual_with_learned_event_weights_does_not_imply_event_temporal_value():
    def score(value):
        return {'total': value, 'endpoints': {name: value for name in ENDPOINTS}}
    cards = []
    for seed in range(20260903, 20260908):
        stages = {name: {'first_step_gradient_norm': 1., 'peak_parameter_delta_from_stage_start': 1.,
                         'training_budget_exhausted': False}
                  for name in ('q', 'bmark', 'event', 'background_current', 'background_state', 'random')}
        cards.append({'subject': 'test', 'seed': seed, 'horizons_seconds': [21600], 'stages': stages,
                      'selection_scores': {name: {'by_horizon': {'21600': score(value)}} for name, value in
                                           [('B_mark_current_background', 2), ('S_dual', 1), ('S_dual_constant_all', 2), ('random_event', 2)]},
                      'time_shift_by_horizon': {'21600': {'gain': 1, 'correct': score(1), 'shifted': score(2)}},
                      'independent_windows_by_horizon': {'21600': {'independent_windows': 3}},
                      '_state_lineage': {'checkpoint_sha256': 'frozen', 'modules': {'event': {'selected_changed': True}}},
                      '_replay_evidence': {'full_endpoint_replay_qualified': True},
                      '_branch_control_evidence': {'score_parity': {'all_endpoint_total_auditable': True},
                                                   'by_horizon': {'21600': {'event': {
                                                       'dynamic_over_branch_constant': -1.,
                                                       'correct_time_over_branch_shift': 1.,
                                                       'increment_over_persistent_background': 1.}}}}})
    row = _h1_family(cards, 'dual', {})['rows'][0]
    assert row['directional_dynamic_state_candidate']
    assert row['composite_with_learned_event_input_joint_evidence']['n_joint_positive'] == 5
    assert row['learned_event_joint_evidence']['n_joint_positive'] == 0
    assert not row['learned_event_dynamic_state_candidate']


def test_changing_endpoint_pairs_across_seeds_is_not_fixed_multi_endpoint_replication():
    common = [{'seed': seed, 'checks': {'trained': True}} for seed in range(5)]
    # Every seed passes two endpoints; no fixed pair passes three seeds.
    endpoints = {'a': {0, 1, 2, 3}, 'b': {0, 1, 4}, 'c': {2, 3, 4}}
    result = _fixed_endpoint_pair_evidence(common, endpoints)
    assert result['joint_evidence']['n_joint_positive'] == 2
    assert result['passing_pairs'] == []
    endpoints['b'].add(2)
    result = _fixed_endpoint_pair_evidence(common, endpoints)
    assert result['passing_pairs'] == ['a+b']


def test_h2b_positive_margins_without_instrument_and_lineage_remain_unqualified():
    keys = ('event_only_state_gain_over_mark_history', 'event_dynamic_gain_over_fit_period_mean', 'event_correct_time_gain_over_shift')
    cards = [{'subject': 'test', 'seed': 20260903 + i,
              'distance_survival': {'support': {'status': 'ESTIMATED', 'seizures_by_phase': {'SELECTION': 3}}},
              'primary_contrasts': {key: 1.0 for key in keys}} for i in range(5)]
    result = _h2b(cards, enforce_evidence=True)['rows'][0]
    assert result['risk_joint_evidence']['event']['n_joint_positive'] == 5
    assert not result['risk_candidates']['event']


def test_suffix_excludes_known_prefix_but_keeps_stop_at_prefix():
    batch = build_event_tensors(np.array([[0, 1, 2, 3], [0, 1, 3, 2]], dtype=np.int16))
    logits = torch.zeros_like(batch['x']); stops = torch.zeros_like(batch['valid'], dtype=torch.float32)
    changed = logits.clone(); changed[:, 0, 1] = 9.0
    stop_changed = stops.clone(); stop_changed[:, 0] = -9.0
    original = per_event_scores(logits, stops, batch, observed_prefix_groups=2)
    altered = per_event_scores(changed, stop_changed, batch, observed_prefix_groups=2)
    for key in original:
        torch.testing.assert_close(original[key], altered[key], rtol=0, atol=0)
    assert bool((per_event_scores(changed, stop_changed, batch)['grammar']
                 < per_event_scores(logits, stops, batch)['grammar']).all())
    terminal = build_event_tensors(np.array([[0, 1, -1, -1]], dtype=np.int16))
    logits = torch.zeros_like(terminal['x']); stops = torch.zeros_like(terminal['valid'], dtype=torch.float32)
    correct_stop = stops.clone(); correct_stop[:, 1] = 9.0
    before = per_event_scores(logits, stops, terminal, observed_prefix_groups=2)
    after = per_event_scores(logits, correct_stop, terminal, observed_prefix_groups=2)
    assert after['stop_bce'].item() < before['stop_bce'].item()
    assert after['n_predict'].item() == 0


def test_history_deletion_removes_event_mass_and_preserves_empty_bin_identity():
    torch.manual_seed(4)
    model = DualStreamEventCTSSM(1, 1, taus_seconds=(600.0,), burden_channels_per_tau=1,
                                  grammar_channels_per_tau=1)
    times = torch.tensor([1.0, 500.0], dtype=torch.float64)
    query = torch.tensor([700.0], dtype=torch.float64)
    marks = torch.tensor([[3.0], [5.0]])
    original = event_query(model, times, marks, marks, query)
    assert torch.equal(original, event_query(model, times, marks, marks, query, torch.ones(2)))
    removed = event_query(model, times, marks, marks, query, torch.tensor([0.0, 1.0]))
    isolated = event_query(model, times[1:], marks[1:], marks[1:], query)
    torch.testing.assert_close(removed, isolated)
    assert removed[-1] < original[-1]


def test_credit_bins_use_physical_age_and_keep_unavailable_gradient_zero():
    ages = np.array([0, 7200, 21600, 28800, 57600, 115200], dtype=float)
    result = bin_gradient(ages, [torch.ones(6, 2)])
    for name in ('0-0.5h', '2-6h', '6-8h', '8-16h', '16-32h', '>=32h'):
        assert result[name]['observations'] == 1
        np.testing.assert_allclose(result[name]['gradient_mass'], np.sqrt(2), rtol=1e-6)
    assert all(row['gradient_mass'] == 0 for row in bin_gradient(ages, [None]).values())


def test_legacy_credit_without_audited_anchors_is_unestimated_not_zero_effect():
    cards = [{'subject': 'test', 'seed': seed, 'horizon_hours': 8, 'audited_anchors': 0,
              'endpoint_credit': {name: {'fraction_beyond_2h': 0., 'fraction_beyond_6h': 0.} for name in ENDPOINTS},
              'selected_state_training': {}} for seed in range(5)]
    row = _credit(cards)['rows'][0]
    assert row['endpoint_credit']['count']['fraction_beyond_6h']['median'] is None
    assert row['endpoint_credit']['count']['fraction_beyond_6h']['estimated_seeds'] == 0
    assert not row['trained_multi_hour_credit_candidate']
