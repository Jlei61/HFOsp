#!/usr/bin/env python3
"""Machine closeout: plan vs actual, failures, resource use, four-state verdicts.

'Training finished', 'tests passed', 'GPUs busy' and 'three seeds' are never
substituted for a scientific conclusion (spec section 10).
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from src.topic5_group_event_state.v0310 import audit
from src.topic5_group_event_state.v0310 import queue_plan as Q
from src.topic5_group_event_state.v035.contracts import atomic_json

VERDICTS = ('REPRODUCIBLE_PREDICTIVE_INCREMENT', 'NO_INCREMENT_SEEN_POWER_AND_TRAINING_STILL_LIMITED',
            'NOT_ESTIMABLE', 'EFFECTIVE_NEGATIVE_WITH_CALIBRATED_SENSITIVITY')


def sha(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root
    out = root / 'final_reports'; out.mkdir(parents=True, exist_ok=True)
    runtime = load_json(root / 'night_runtime.json') or {}
    queue = load_json(root / 'queue_state.json') or {}
    boot = load_json(root / 'bootstrap_status.json') or {}
    preflight = load_json(root / 'training_preflight.json') or {}
    throughput = load_json(root / 'throughput_probe.json') or {}
    ledger = load_json(root / 'seizure' / 'seizure_ledger_summary.json') or {}
    aggregation = load_json(out / 'aggregation_summary.json') or {}
    cards = audit.load_cards(root / 'human_v0310')
    common = queue.get('common_recipe') or {}
    planned = {phase: len(Q.phase_cells(phase, common or {s: 'R1' for s in Q.SUBJECTS}))
               for phase in Q.DISPATCH_ORDER}
    actual = {}
    for card in cards:
        phase = Path(card['_path']).parent.parent.name
        actual.setdefault(phase, {}).setdefault(card['status'], 0)
        actual[phase][card['status']] += 1
    # The scheduler was restarted mid-night, so its in-memory history restarts
    # too. Merge every recorded run rather than reporting only the last one.
    runs = [queue] + [load_json(q) for q in sorted(root.glob('queue_state_run*.json'))
                      + sorted(root.glob('queue_state_before*.json'))]
    seen, finished = set(), []
    for run in runs:
        for record in (run or {}).get('finished', []):
            token = (record['id'], round(record.get('elapsed_seconds', 0)))
            if token in seen:
                continue
            seen.add(token); finished.append(record)
    failures = [r for r in finished if r.get('status') not in ('COMPLETE', 'WALL_TIME_LIMITED')]
    gpu_seconds = sum(r.get('elapsed_seconds', 0) for r in finished)
    snapshots = []
    snapshot_path = root / 'resource_snapshots.jsonl'
    if snapshot_path.exists():
        snapshots = [json.loads(line) for line in snapshot_path.read_text().splitlines() if line.strip()]

    groups = audit.group_cards(cards)
    complete_triples = sum(1 for g in groups.values() if {c['family'] for c in g} == {'F', 'L', 'N'})
    scored = [c for c in cards if c['metrics'].get('2', {}).get('status') == 'SCORED'
              and c['metrics']['2'].get('gain_over_floored_control') is not None]
    gains = np.array([c['metrics']['2']['gain_over_floored_control'] for c in scored], float)
    any_limited = any(c['training_sufficiency_vector']['any_arm_budget_limited'] or
                      c['training_sufficiency_vector']['any_arm_wall_time_limited'] for c in cards)

    def verdict_event_information():
        if not scored:
            return dict(verdict='NOT_ESTIMABLE', reason='no held-out cell was scored')
        positive = int((gains > 0).sum())
        per_patient = {}
        for card in scored:
            entry = per_patient.setdefault(card['subject'], dict(n=0, n_positive=0, values=[],
                                                                n_scored_anchors=None,
                                                                n_two_hour_windows=None))
            value = card['metrics']['2']['gain_over_floored_control']
            entry['n'] += 1; entry['n_positive'] += int(value > 0); entry['values'].append(round(value, 5))
            entry['n_scored_anchors'] = card['metrics']['2']['denominators']['n_anchors']
            entry['n_two_hour_windows'] = card['metrics']['2']['denominators']['n_two_hour_physical_windows']
            # A gain of exactly zero because the fit never left its initialisation
            # is a different statement from a gain near zero after real training.
            entry['n_initial_checkpoint_selected'] = entry.get('n_initial_checkpoint_selected', 0) + int(
                bool((card.get('optimisation_diagnosis') or {}).get('initial_checkpoint_selected')))
        for entry in per_patient.values():
            entry['median'] = float(np.median(entry['values']))
        return dict(verdict='NO_INCREMENT_SEEN_POWER_AND_TRAINING_STILL_LIMITED'
                    if not (positive == len(gains) and len(gains) >= 6)
                    else 'REPRODUCIBLE_PREDICTIVE_INCREMENT',
                    n_arms=len(gains), n_positive=positive,
                    median_gain=float(np.median(gains)), min_gain=float(gains.min()),
                    max_gain=float(gains.max()),
                    training_limited=bool(any_limited), per_patient=per_patient,
                    n_arms_that_never_left_initialisation=int(sum(
                        bool((c.get('optimisation_diagnosis') or {}).get('initial_checkpoint_selected'))
                        for c in scored)),
                    zero_gain_note='an arm whose best checkpoint is its initialisation scores exactly zero '
                                   'against the frozen background because it IS the frozen background; '
                                   'that is not the same statement as a trained state failing to help',
                    denominator_note='the held-out denominator differs a lot between patients; a patient '
                                     'with a handful of scored anchors carries far less weight than one '
                                     'with a hundred, and no cohort statistic is formed from three patients',
                    reason='an effective negative would additionally require a pre-registered minimum '
                           'effect with end-to-end sensitivity, which this night did not calibrate')

    verdicts = dict(
        does_event_content_add_predictive_information=verdict_event_information(),
        is_learned_history_integration_needed=dict(
            verdict='NOT_ESTIMABLE' if complete_triples == 0 else
                    'NO_INCREMENT_SEEN_POWER_AND_TRAINING_STILL_LIMITED',
            n_complete_family_triples=complete_triples,
            note='reported from the paired family contrast table; any budget-limited arm forbids a '
                 'sufficiency claim'),
        is_a_nonlinear_transition_needed=dict(
            verdict='NOT_ESTIMABLE' if complete_triples == 0 else
                    'NO_INCREMENT_SEEN_POWER_AND_TRAINING_STILL_LIMITED',
            note='a nonlinear arm with no advantage may equally reflect identifiability or power'),
        does_the_same_state_transfer=dict(
            verdict='NOT_ESTIMABLE' if not (root / 'm1_transfer' / 'm1_transfer_index.json').exists()
                    else 'NO_INCREMENT_SEEN_POWER_AND_TRAINING_STILL_LIMITED',
            note='jointly trained count and coarse recruitment do not establish untrained expression transfer'),
        is_there_pre_seizure_information=dict(
            verdict='NOT_ESTIMABLE' if not list((root / 'seizure').glob('seizure_scores_H*.json')) else
                    'NO_INCREMENT_SEEN_POWER_AND_TRAINING_STILL_LIMITED',
            eligible_clusters={k: v.get('n_eligible_first_onsets') for k, v in
                               (ledger.get('subjects') or {}).items()},
            note='per-episode description only; no clinical predictor is claimed'),
        do_events_change_physiology=dict(
            verdict='NOT_ESTIMABLE',
            boundary_check=load_json(root / 'h3_boundary' / 'h3_boundary.json'),
            note='observer updates and history gradients are computational facts; a physiological test '
                 'needs low-latency measurement and an independent future observation'))

    deliverables = {}
    for name in ('training_audit.csv', 'layer_inventory.csv', 'phase_representativeness.csv',
                 'phase_representativeness.json', 'paired_family_contrasts.csv', 'paired_physical_windows.csv',
                 'contact_endpoint_scores.csv', 'same_checkpoint_evidence.csv', 'bootstrap_lr_audit.csv',
                 'bootstrap_lr_audit.json', 'aggregation_summary.json', 'report_zh.md'):
        path = out / name
        deliverables[name] = dict(exists=path.exists(), sha256=sha(path),
                                  bytes=path.stat().st_size if path.exists() else 0)
    for name, path in (('seizure_episode_ledger.csv', root / 'seizure' / 'seizure_episode_ledger.csv'),
                       ('seizure_ledger_summary.json', root / 'seizure' / 'seizure_ledger_summary.json'),
                       ('training_preflight.json', root / 'training_preflight.json'),
                       ('training_contract_clauses.json', root / 'training_contract_clauses.json'),
                       ('night_runtime.json', root / 'night_runtime.json'),
                       ('queue_state.json', root / 'queue_state.json'),
                       ('throughput_probe.json', root / 'throughput_probe.json'),
                       ('h3_boundary.json', root / 'h3_boundary' / 'h3_boundary.json')):
        deliverables[name] = dict(exists=path.exists(), sha256=sha(path),
                                  bytes=path.stat().st_size if path.exists() else 0)
    for name, path in (('figures/figure_index.json', out / 'figures' / 'figure_index.json'),
                       ('figures/prefix_example_index.json', out / 'figures' / 'prefix_example_index.json')):
        deliverables[name] = dict(exists=path.exists(), sha256=sha(path),
                                  bytes=path.stat().st_size if path.exists() else 0)
    for path in sorted((out / 'figures').glob('*.png')):
        deliverables['figures/' + path.name] = dict(exists=True, sha256=sha(path), bytes=path.stat().st_size)

    payload = dict(
        status='COMPLETE', schema='v0310_night_closeout_v1', timestamp=time.time(),
        runtime=runtime, wall_hours_elapsed=(time.time() - runtime.get('t0_epoch', time.time())) / 3600,
        plan=dict(planned_cells=planned, dispatch_order=list(Q.DISPATCH_ORDER),
                  common_recipe=common, note='upper bounds, never a promise that all of them run'),
        actual=dict(by_phase=actual, n_cards=len(cards), n_complete_family_triples=complete_triples,
                    scheduler_status=queue.get('status'), scheduler_phase=queue.get('phase'),
                    workers_per_gpu=queue.get('workers_per_gpu'),
                    observed_seconds=queue.get('observed_seconds')),
        not_run=dict(reason='wall-clock admission control stops a paired group that cannot finish before '
                            'the no-new-long-jobs deadline; remaining registered work is listed here',
                     phases_not_reached=[p for p in Q.DISPATCH_ORDER if p not in actual],
                     admission_stops=queue.get('admission_stops', []),
                     gpu_quarantine=queue.get('consecutive_failures', {}),
                     registered_but_not_implemented=Q.NOT_IMPLEMENTED_PHASES),
        failed_attempts=failures,
        incidents=[dict(
            time='2026-09-06T03:18+08:00', severity='corrected',
            what='the scheduler dispatched two processes for the same cell, because a phase whose pending '
                 'list had drained was re-planned while its cells were still running, and a running cell '
                 'has no finished card yet to exclude it.',
            introduced_by='an in-flight change earlier the same night that let the next phase be planned '
                          'before the current one fully drained, to stop compute slots idling at phase '
                          'boundaries',
            damage='one cell had two concurrent processes writing one directory (the second resumed from '
                   'the first checkpoint mid-training); one further duplicate was harmless because the '
                   'first process had already written its card and the second returned immediately.',
            fix='a cell id is dispatched at most once per scheduler process; the affected cell directories '
                'were discarded and re-run from scratch, and 43 finished cards were left untouched.',
            cost='about ten minutes of compute and a two-minute gap with idle cards',
            verification='after the restart, one scheduler process, one process per cell, no duplicate ids '
                         'in the running or finished lists'),
        ],
        bootstrap=dict(status=boot.get('status'), completed=len(boot.get('completed', [])),
                       pending=boot.get('pending'), n_planned=18,
                       undispatched_not_rescued=True,
                       undispatched_decision='the bootstrap window is a registered stopping rule (two hours, '
                                             'no new dispatch after ninety minutes). The one undispatched '
                                             'cell was left unfinished rather than re-dispatched after the '
                                             'window closed, even though it is cheap, because relaxing a '
                                             'registered stop after seeing the results is exactly what the '
                                             'contract forbids.',
                       audit=load_json(out / 'bootstrap_lr_audit.json')),
        preflight=dict(all_passed=preflight.get('all_passed'),
                       tests=[dict(test=t['test'], passed=t['passed']) for t in preflight.get('tests', [])],
                       memory_probe=preflight.get('memory_probe')),
        throughput_probe=throughput,
        resources=dict(gpu_process_seconds=round(gpu_seconds), n_resource_snapshots=len(snapshots),
                       last_snapshot=snapshots[-1] if snapshots else None),
        aggregation=aggregation,
        what_the_non_positive_result_can_and_cannot_settle=dict(
            established=('the partitions differ markedly in event distribution and in real covered time. '
                         'For one patient the fitting and held-out medians are nearly identical (60 vs 58) '
                         'while the means differ threefold, because the partitions hold different '
                         'proportions of high-firing stretches; that patient\'s checkpoint was selected on '
                         '3.2 de-duplicated hours in which 95 per cent of windows exceed the entire '
                         'fitting period maximum. Measured in state space, every frozen state places '
                         'essentially all held-out queries on one side of the fitted-period median, one to '
                         'eight fitted standard deviations away.'),
            two_views_of_one_thing=('the state drift and the target-distribution difference are most '
                                    'likely the same phenomenon measured twice, not two independent '
                                    'findings'),
            mechanism_not_identified=('rhythm phase sampling, a real state change, and a measurement or '
                                      'condition change are all compatible with these numbers and call '
                                      'for different fixes'),
            not_established=[
                'that a circadian mismatch is proven: a monotone rise inside a short window reproduces the '
                'same clock correlation without any 24-hour cycle',
                'that the recent-versus-future association truly reverses: local phase, a few high values '
                'or a mixture of distributions reproduce the sign change',
                'that the baseline comparison is invalid: when the available inputs carry no information a '
                'constant is a legitimate opponent, and losing to it still means the state pathway gave no '
                'reliable increment',
                'that the predictable share is only 0.005: the event-only parent sees clock and coverage '
                'only, the absolute loss contains irreducible uncertainty, and a weak parent can also '
                'reflect its representation, its optimisation or its evaluation support'],
            conclusion=('the current no-increment result cannot adjudicate whether interictal events carry '
                        'pathological state'),
            before_any_remedy=('firing level, slow rhythms and their changes may themselves express the '
                               'pathological state, so removing them to improve prediction can delete the '
                               'physiology under study. Which variation is to be preserved and explained '
                               'must be settled before deciding what to condition on'),
            evidence=['final_reports/phase_representativeness.csv', 'final_reports/state_drift.csv'],
            reframed_next_questions=[
                'state identification: does the long history help explain subsequent event morphology, '
                'spatial organisation and load beyond what recent statistics explain, keeping both the '
                'absolute level and the conditional shape',
                'state evolution: with no future events read, how far ahead does that state predict, '
                'reported separately for near-term, thirty minutes and two hours, since a failure at one '
                'lead does not generalise to no state'],
            a_five_minute_target_is_not_yet_warranted=('success at a short lead may only track the local '
                                                       'level; whether the long history adds information '
                                                       'has to be asked separately')),
        power_calibration=dict(
            status='POWER_NOT_CALIBRATED',
            registered_requirement='a real-patient power calibration must keep the real missingness, event '
                                   'density, publication delay and physical-window correlation, and run the '
                                   'full training -> INNER selection -> frozen probe chain; handing a probe '
                                   'the true factor directly is only an optimistic upper bound',
            why_not_done='it is not part of the fixed queue in the frozen spec and building it tonight would '
                         'have been a new grid, which the contract forbids',
            consequence='a non-positive human result tonight must be read as "no increment seen at this '
                        'power", never as a scientific negative'),
        verdicts=verdicts, verdict_states=list(VERDICTS),
        deliverables=deliverables,
        provenance=dict(committed=False, pushed=False,
                        reason='the handoff carries no publish task and forbids committing the working '
                               'tree, which already held unrelated uncommitted work',
                        traceability='every card, figure and report carries the sha256 of the source files '
                                     'that produced it, and the preflight record binds the gate to those '
                                     'same hashes',
                        new_source_files=sorted(str(q) for q in [
                            Path('src/topic5_group_event_state/v0310/__init__.py'),
                            Path('src/topic5_group_event_state/v0310/history.py'),
                            Path('src/topic5_group_event_state/v0310/objective.py'),
                            Path('src/topic5_group_event_state/v0310/trainer.py'),
                            Path('src/topic5_group_event_state/v0310/audit.py'),
                            Path('src/topic5_group_event_state/v0310/queue_plan.py'),
                            Path('src/topic5_group_event_state/v0310/seizure.py'),
                            Path('src/topic5_group_event_state/v0310/seizure_scores.py'),
                            Path('scripts/train_group_event_state_v0310_human.py'),
                            Path('scripts/test_group_event_state_v0310_preflight.py'),
                            Path('scripts/schedule_group_event_state_v0310_night.py'),
                            Path('scripts/export_group_event_state_v0310_frozen_state.py'),
                            Path('scripts/run_group_event_state_v0310_m1_transfer.py'),
                            Path('scripts/build_group_event_state_v0310_seizure_ledger.py'),
                            Path('scripts/score_group_event_state_v0310_seizure.py'),
                            Path('scripts/probe_group_event_state_v0310_h3_boundary.py'),
                            Path('scripts/probe_group_event_state_v0310_throughput.py'),
                            Path('scripts/audit_group_event_state_v0310_bootstrap_lr.py'),
                            Path('scripts/aggregate_group_event_state_v0310_night.py'),
                            Path('scripts/plot_group_event_state_v0310_night.py'),
                            Path('scripts/closeout_group_event_state_v0310_night.py')])),
        forbidden_substitutions=['training finished', 'all tests passed', 'GPUs fully loaded',
                                 'three optimisation seeds'],
        scheduler_restarts=len([q for q in root.glob('queue_state_run*.json')]) +
                           len([q for q in root.glob('queue_state_before*.json')]),
        note='zero failures are not claimed; the duplicate-dispatch incident above and its re-runs are '
             'part of the record, and every attempt from every scheduler run is merged here')
    atomic_json(out / 'night_closeout.json', payload)
    print(json.dumps(dict(n_cards=len(cards), complete_triples=complete_triples,
                          verdicts={k: v['verdict'] for k, v in verdicts.items()}), ensure_ascii=False),
          flush=True)


if __name__ == '__main__':
    main()
