"""Scientific reporting must fail closed on unpaired or unverified instruments."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.topic5_group_event_state.v039.instrument_audit import convergence_audit, known_truth_margin, sha256

REPORT = Path(__file__).resolve().parents[1] / 'scripts/report_group_event_state_v039_closure.py'


def test_support_label():
    spec = importlib.util.spec_from_file_location('v039_report', REPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.support_label([{'n_selection_anchors': 24}] * 3) == '24'
    assert module.support_label([{'n_anchors': 19}, {'n_anchors': 24}]) == '19–24'
    assert module.support_label([]) == '不可估'


def pair():
    return [dict(family=f, case='nonlinear_transition', experiment='joint', seed=1,
                 truth={'seed': 39001}, held_out={'total': score}, selected_step=480,
                 optimization_limited=True, stop_reason='BUDGET_LIMIT') for f, score in [('L', 3.8), ('N', 3.7)]]


def test_mixed_budget_stops_are_not_convergence():
    rows = pair()
    rows[0].update(optimization_limited=False, stop_reason='INNER_PATIENCE')
    result = known_truth_margin(rows, 'nonlinear_transition', 'N', 'L')
    assert result['margin'] == pytest.approx(.1)
    assert result['any_arm_budget_limited']
    assert not result['converged_margin']
    assert not result['all_arms_patience_stopped']


def test_patience_and_missing_metadata_are_not_convergence():
    rows = pair()
    for r in rows:
        r.update(optimization_limited=False, stop_reason='INNER_PATIENCE')
    assert known_truth_margin(rows, 'nonlinear_transition', 'N', 'L')['all_arms_patience_stopped']
    assert not known_truth_margin(rows, 'nonlinear_transition', 'N', 'L')['converged_margin']
    for r in rows:
        del r['seed']
        del r['optimization_limited']
    result = known_truth_margin(rows, 'nonlinear_transition', 'N', 'L')
    assert result['n_seeds'] is None
    assert not result['all_arms_patience_stopped']


@pytest.mark.parametrize('mutation', ['seed', 'data_seed', 'duplicate'])
def test_unpaired_or_duplicate_summary_rows_rejected(mutation):
    rows = pair()
    if mutation == 'seed': rows[0]['seed'] = 2
    if mutation == 'data_seed': rows[0]['truth']['seed'] = 39002
    if mutation == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError): known_truth_margin(rows, 'nonlinear_transition', 'N', 'L')


def fixtures(root):
    folder = root / 'instruments_convergence_audit'
    folder.mkdir()
    for seed in (20260905, 20260906, 20260907):
        for family, total in [('F', 3.72), ('L', 3.64), ('N', 3.65)]:
            path = folder / f'nonlinear_{family}_seed{seed}'
            checkpoint, scores = path.with_suffix('.pt'), path.with_suffix('.npz')
            checkpoint.write_bytes(b'checkpoint')
            np.savez(scores, total=np.array([total, total]), count=np.array([3., 3.]),
                     recruitment=np.array([total-3, total-3]), episode_id=np.array([20, 21]))
            card = dict(status='COMPLETE', family=family, seed=seed, case='nonlinear_transition',
                        config=dict(seed=seed, family=family, max_steps=2400, patience=6),
                        truth={'seed': 39001}, input_sha256='same', source_hashes={'runner.py': 'same'},
                        selected_step=1000, steps_run=1240, selected_inner=4.,
                        stop_reason='INNER_PATIENCE', optimization_limited=False,
                        training_curve=[{'step': 1000, 'inner': 4.}] +
                                       [{'step': t, 'inner': 4.1} for t in range(1040, 1241, 40)],
                        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                        scores=str(scores), scores_sha256=sha256(scores), elapsed_seconds=10,
                        held_out=dict(total=total, count=3., recruitment=total-3))
            path.write_text(json.dumps(card))
    return folder


def test_validated_pairing_and_direction(tmp_path):
    fixtures(tmp_path)
    result = convergence_audit(tmp_path)
    assert result['all_patience_stopped']
    assert result['N_over_L']['median'] == pytest.approx(-.01)
    assert result['N_over_F']['positive'] == result['L_over_F']['positive'] == 3
    assert result['n_data_seeds'] == 1
    assert not result['converged_margin']


@pytest.mark.parametrize('mutation', ['missing', 'status', 'stops', 'source', 'scores', 'input', 'recipe', 'episodes'])
def test_bad_audit_evidence_rejected(tmp_path, mutation):
    folder = fixtures(tmp_path)
    path = folder / 'nonlinear_N_seed20260905'
    card = json.loads(path.read_text())
    if mutation == 'missing':
        path.unlink()
    else:
        if mutation == 'status': card['status'] = 'RUNNING'
        if mutation == 'stops': card['stop_reason'] = 'BUDGET_LIMIT'
        if mutation == 'source': card['source_hashes']['runner.py'] = 'changed'
        if mutation == 'scores': card['held_out']['total'] = 0
        if mutation == 'input': card['input_sha256'] = 'other'
        if mutation == 'recipe': card['config']['max_steps'] = 2600
        if mutation == 'episodes':
            with np.load(card['scores']) as saved: arrays = dict(saved)
            arrays['episode_id'] = np.array([21, 22])
            np.savez(card['scores'], **arrays)
            card['scores_sha256'] = sha256(card['scores'])
        path.write_text(json.dumps(card))
    with pytest.raises(ValueError): convergence_audit(tmp_path)


def test_absent_audit(tmp_path):
    assert convergence_audit(tmp_path) is None
