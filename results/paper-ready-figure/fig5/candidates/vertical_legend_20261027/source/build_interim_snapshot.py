"""Author-requested interim F readout at a common completed search prefix."""
import argparse
import csv
import sys
import time
from pathlib import Path

FROZEN = Path('/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-continuous-core-state-r1/scripts/topic4_cohort_core_count_20261024')
sys.path.insert(0, str(FROZEN))
from common import ROOT, read, write, sha, np, run_dir, verify_sources
from export_selected_mode_events import export_one


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    verify_sources()
    plan = read(ROOT/'plan.json')
    dest = args.output
    selection_path = dest/'selection.json'
    if not selection_path.exists():
        streams = []
        for sid in plan['subjects']:
            for k in (1, 2):
                run = run_dir(k, 0, sid)
                scores = sorted([read(p) for p in (run/'scores').glob('*.json')], key=lambda s: s['epoch'])
                epochs = {s['epoch'] for s in scores}
                assert len(epochs) == len(scores)
                prefix = 0
                while prefix+1 in epochs:
                    prefix += 1
                streams.append(dict(subject=sid, core_count=k, scores=scores, completed_prefix=prefix))
        budget = min(s['completed_prefix'] for s in streams)
        assert 0 < budget < plan['epochs']
        for stream in streams:
            eligible = [s for s in stream['scores'] if s['epoch'] <= budget]
            assert [s['epoch'] for s in eligible] == list(range(1, budget+1))
            good = [s for s in eligible if s['J'] is not None and np.isfinite(s['J'])]
            best = min(good, key=lambda s: (s['J'], s['epoch'])) if good else None
            stream.update(included_scores=eligible, selected=best,
                          selected_by='minimum training J within common completed prefix; never by template rho')
        write(selection_path, dict(status='AUTHOR_REQUESTED_INTERIM', author_request='2026-10-26: generate a version from current results',
            frozen_unix=time.time(), common_budget=budget, planned_budget=plan['epochs'],
            subjects=plan['subjects'], streams=streams, live_status_at_snapshot=read(ROOT/'status.json')))
    selection = read(selection_path)
    rows = []
    for stream in selection['streams']:
        sid, k, best = stream['subject'], stream['core_count'], stream['selected']
        row = dict(subject=sid, core_count=k, interim=True,
                   training_completed=len(stream['scores']), training_used=selection['common_budget'],
                   training_planned=plan['epochs'], matched_rhos=[None, None], status='NO_SCORABLE_CANDIDATE')
        if best is not None:
            unit = run_dir(k, 0, sid)/'units'/best['candidate']
            saved = dest/'per_subject'/sid/f'core{k}'
            cached = saved/'summary.json'
            result = read(cached) if cached.exists() else export_one(sid, unit, saved)
            assert result['trajectory_sha256'] == sha(unit/'trajectory.npz')
            row.update(result, core_count=k, candidate=best['candidate'], training_J=best['J'],
                       selected_epoch=best['epoch'], event_export=str(saved))
        rows.append(row)
        print(sid, 'core', k, row['status'], flush=True)
    statistics = []
    for k in (1, 2):
        values = np.asarray([r['matched_rhos'] for r in rows if r['core_count'] == k], float)
        for mode, name in enumerate(['TA–MTA', 'TB–MTB']):
            v = values[np.isfinite(values[:, mode]), mode]
            statistics.append(dict(core_count=k, mode=name, n=len(v), missing=25-len(v),
                median=float(np.median(v)) if len(v) else None,
                q25_q75=np.quantile(v, [.25, .75]).tolist() if len(v) else None))
    report = dict(status='AUTHOR_REQUESTED_INTERIM', cohort_comparison_complete=False,
        common_budget=selection['common_budget'], planned_budget=plan['epochs'], subject_denominator=25,
        training_used=50*selection['common_budget'], subject_arms=rows, statistics=statistics,
        confirmations=[], independent_noise_confirmations_used=0, statistical_unit='patient',
        mode_order=['TA–MTA', 'TB–MTB'], selection_sha256=sha(selection_path),
        snapshot_unix=selection['frozen_unix'], comparison='Same completed search prefix in every patient and both core-count arms.',
        limitations=['Interim training-selected working points; optimization and independent-noise confirmation remain in progress.',
                    'Fixed model-only K=2 readout is descriptive; it does not demonstrate two natural modes.',
                    'One workpoint per patient and core count supplies both template correlations.'])
    write(dest/'summary.json', report)
    fields = ['subject', 'core_count', 'training_used', 'training_completed', 'candidate', 'selected_epoch',
              'training_J', 'status', 'n_events', 'rho_TA_MTA', 'rho_TB_MTB']
    with (dest/'subjects.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {name: row.get(name) for name in fields}
            record.update(rho_TA_MTA=row['matched_rhos'][0], rho_TB_MTB=row['matched_rhos'][1])
            writer.writerow(record)
    print('Snapshot ready:', dest, 'common budget:', selection['common_budget'], flush=True)


if __name__ == '__main__':
    main()
