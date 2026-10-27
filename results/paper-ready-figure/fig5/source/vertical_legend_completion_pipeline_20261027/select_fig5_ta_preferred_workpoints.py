#!/usr/bin/env python3
"""Select minimum-J single-core workpoints subject to signed rho_TA > rho_TB."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
from pathlib import Path
import shutil
import sys
import time

REPO = Path('/home/honglab/leijiaxin/HFOsp')
FROZEN = REPO/'.worktrees/topic4-continuous-core-state-r1/scripts/topic4_cohort_core_count_20261024'
EXECUTION = REPO/'.worktrees/topic4-continuous-core-state-r1/scripts/topic4_cohort_core_count_execution_20261026'
sys.path[:0] = [str(FROZEN), str(EXECUTION), str(REPO)]
from common import ROOT as RUN, read, write, sha, np, run_dir, verify_sources
from template_report import readout
from export_selected_mode_events import export_one


def preferred(rhos):
    values = np.asarray(rhos, float)
    return bool(np.isfinite(values).all() and values[0]-values[1] > 1e-12)


def select_patient(task):
    sid, stream, parent, output = task
    output = Path(output)
    scores = sorted((s for s in stream['included_scores'] if s['J'] is not None and np.isfinite(s['J'])),
                    key=lambda s: (s['J'], s['epoch']))
    tested, selected = [], None
    for score in scores:
        unit = run_dir(1, 0, sid)/'units'/score['candidate']
        cache = output/'readouts'/sid/(score['candidate']+'.json')
        if cache.exists():
            result = read(cache)
        elif score['candidate'] == parent.get('candidate'):
            result = dict(parent, source=str(unit))
            write(cache, result)
        else:
            result = dict(readout(sid, unit), source=str(unit))
            write(cache, result)
        expected_hash = read(unit/'result.json')['arrays_sha256']
        if result.get('trajectory_sha256') is None:
            # Non-estimable readouts return before the legacy helper adds provenance.
            assert sha(unit/'trajectory.npz') == expected_hash
            result['trajectory_sha256'] = expected_hash
            write(cache, result)
        assert result['trajectory_sha256'] == expected_hash
        eligible = preferred(result['matched_rhos'])
        tested.append(dict(candidate=score['candidate'], epoch=score['epoch'], J=score['J'],
                           matched_rhos=result['matched_rhos'], eligible=eligible, readout=str(cache)))
        if eligible:
            selected = score
            break
    row = dict(parent)
    if selected is None:
        row.update(status='NO_TA_PREFERRED_CANDIDATE', matched_rhos=[None, None],
                   candidate=None, training_J=None, selected_epoch=None, event_export=None)
        for key in ['source', 'trajectory_sha256', 'n_events', 'mode_counts',
                    'model_cluster_for_TA_TB', 'raw_cluster_patient_matrix']:
            row.pop(key, None)
    else:
        unit = run_dir(1, 0, sid)/'units'/selected['candidate']
        dest = output/'per_subject'/sid/'core1'
        if selected['candidate'] == parent.get('candidate') and parent.get('event_export'):
            exported = parent
            event_export = parent['event_export']
        else:
            exported = read(dest/'summary.json') if (dest/'summary.json').exists() else export_one(sid, unit, dest)
            event_export = str(dest)
        assert exported['trajectory_sha256'] == read(unit/'result.json')['arrays_sha256']
        np.testing.assert_allclose(exported['matched_rhos'], result['matched_rhos'], rtol=0, atol=1e-12)
        assert preferred(exported['matched_rhos'])
        row.update(exported, subject=sid, core_count=1, candidate=selected['candidate'],
                   training_J=selected['J'], selected_epoch=selected['epoch'],
                   source=str(unit), event_export=event_export)
    audit = dict(subject=sid, rule='finite signed rho_TA > rho_TB + 1e-12; minimum J then epoch',
        candidate_budget=len(stream['included_scores']), scorable_candidates=len(scores),
        evaluated_in_increasing_J=tested,
        untested_higher_J_candidates=len(scores)-len(tested),
        early_stop='First eligible candidate is the constrained minimum because candidates are ordered by (J, epoch)',
        previous_candidate=parent.get('candidate'), selected_candidate=row.get('candidate'),
        previous_rhos=parent['matched_rhos'], selected_rhos=row['matched_rhos'],
        previous_J=parent.get('training_J'), selected_J=row.get('training_J'),
        changed=parent.get('candidate') != row.get('candidate'), feasible=selected is not None)
    write(output/'per_subject'/sid/'selection_audit.json', audit)
    return row, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-candidate', type=Path, default=REPO/'results/paper-ready-figure/fig5/candidates/joint_template_matching_20261027')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--completed', action='store_true')
    args = parser.parse_args()
    verify_sources()
    args.output.mkdir(parents=True, exist_ok=True)
    input_dir = args.output/'inputs'
    selection = read(args.parent_candidate/'source/cohort_selection.json')
    if not (input_dir/'parent_report.json').exists():
        if args.completed:
            report = read(RUN/'template_recovery_summary.json')
            assert report['status'] == 'TRAINING_COMPLETE_REVIEW_PENDING'
            assert report['training_complete'] == report['training_planned'] == 1600
            streams = []
            for row in report['subject_arms']:
                scores = sorted([read(p) for p in (run_dir(row['core_count'], 0, row['subject'])/'scores').glob('*.json')], key=lambda s: s['epoch'])
                assert [s['epoch'] for s in scores] == list(range(1, 33))
                streams.append(dict(subject=row['subject'], core_count=row['core_count'], scores=scores,
                                    included_scores=scores, completed_prefix=32))
            prefix = dict(common_budget=32, planned_budget=32, streams=streams, frozen_unix=time.time())
        else:
            report = read(args.parent_candidate/'source/interim_core_count_summary.json')
            prefix = read(args.parent_candidate/'source/interim_selection.json')
        write(input_dir/'parent_report.json', report)
        write(input_dir/'search_prefix.json', prefix)
        write(input_dir/'cohort_selection.json', selection)
        write(input_dir/'policy.json', dict(schema='fig5_ta_preferred_selection_v1',
            author_request_date='2026-10-27', core1_constraint='signed rho_TA > signed rho_TB + 1e-12; both finite',
            core1_objective='minimum original training J among eligible candidates',
            core2_selection='unchanged minimum original training J',
            no_template_relabelling=True, no_absolute_correlations=True,
            no_extra_rho_threshold=True, no_TB_ceiling=True,
            no_new_simulations=True, infeasible_subject='retained with missing 1-core readout',
            inference='Post-hoc asymmetric selection-policy comparison; selection includes TB, so weaker TB is partly conditioned by design, not independent evidence of a core-count mechanism'))
    report = read(input_dir/'parent_report.json')
    prefix = read(input_dir/'search_prefix.json')
    selection = read(input_dir/'cohort_selection.json')
    subjects = selection['included_subjects']
    lookup = {(r['subject'], r['core_count']): r for r in report['subject_arms']}
    streams = {(r['subject'], r['core_count']): r for r in prefix['streams']}
    tasks = [(sid, streams[sid, 1], lookup[sid, 1], str(args.output)) for sid in subjects]
    audits = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row, audit in pool.map(select_patient, tasks):
            lookup[row['subject'], 1] = row
            audits.append(audit)
            print(row['subject'], row.get('candidate'), row['matched_rhos'], flush=True)
    rows = [lookup[r['subject'], r['core_count']] for r in report['subject_arms']]
    constrained_selection = copy.deepcopy(prefix)
    for stream in constrained_selection['streams']:
        key = stream['subject'], stream['core_count']
        selected_id = lookup[key].get('candidate')
        stream['selected'] = next((s for s in stream['included_scores'] if s['candidate'] == selected_id), None)
        stream['selected_by'] = ('minimum training J subject to finite signed rho_TA > rho_TB + 1e-12'
            if key[1] == 1 and key[0] in subjects else 'unchanged minimum training J')
    constrained_selection['selection_policy'] = read(input_dir/'policy.json')
    write(args.output/'selection.json', constrained_selection)
    result = copy.deepcopy(report)
    result.update(status='AUTHOR_REQUESTED_INTERIM', cohort_comparison_complete=False,
        subject_arms=rows, common_budget=prefix['common_budget'], planned_budget=32,
        training_used=len(rows)*prefix['common_budget'],
        snapshot_unix=report.get('snapshot_unix', prefix['frozen_unix']),
        training_search_complete=bool(args.completed), constraint_applied_unix=time.time(),
        selection_policy=read(input_dir/'policy.json'), selection_audit=str(args.output/'selection_audit.json'),
        parent_selection_sha256=report.get('selection_sha256'),
        selection_sha256=sha(args.output/'selection.json'),
        comparison='TA-preferred constrained single-core workpoint versus unconstrained minimum-J two-core workpoint',
        analysis_status='POST_HOC_ASYMMETRIC_SELECTION', confirmations=[],
        independent_noise_confirmations_used=0,
        confirmation_scope='No new-noise replays are used in this conditional training readout; original nominations do not validate reselected core1 workpoints')
    for row in result['subject_arms']:
        row.update(interim=True, training_used=prefix['common_budget'], training_planned=32)
    result['statistics'] = []
    for k in (1, 2):
        for mode, name in enumerate(['TA–MTA', 'TB–MTB']):
            values = np.asarray([r['matched_rhos'][mode] for r in rows if r['core_count'] == k], float)
            finite = values[np.isfinite(values)]
            result['statistics'].append(dict(core_count=k, mode=name, n=len(finite), missing=len(values)-len(finite),
                median=float(np.median(finite)) if len(finite) else None,
                q25_q75=np.quantile(finite, [.25, .75]).tolist() if len(finite) else [None, None]))
    write(args.output/'selection_audit.json', dict(policy=read(input_dir/'policy.json'),
        subjects=audits, n_subjects=len(subjects), n_feasible=sum(r['feasible'] for r in audits),
        n_changed=sum(r['changed'] for r in audits), budget=prefix['common_budget']))
    write(args.output/'summary.json', result)
    shutil.copy2(Path(__file__), args.output/'producer.py')
    print(json.dumps(dict(output=str(args.output), feasible=sum(r['feasible'] for r in audits),
                         changed=sum(r['changed'] for r in audits), new_simulations=0)), flush=True)


if __name__ == '__main__':
    main()
