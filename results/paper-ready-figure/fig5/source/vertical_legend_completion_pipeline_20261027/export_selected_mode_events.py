"""Persist the historical F readout at nominated one/two-core working points."""
import argparse
import csv
import sys
from pathlib import Path

FROZEN_CODE = Path('/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-continuous-core-state-r1/scripts/topic4_cohort_core_count_20261024')
sys.path.insert(0, str(FROZEN_CODE))
from common import ROOT, REPO, read, write, sha, save_npz, np, hashlib, run_dir, verify_sources
sys.path.insert(0, str(REPO))
from src.topic4_template_recovery import fit_two_templates, align_profiles
from src.topic4_dual_mode_recovery import event_ranks

PATIENT = Path('/data/hfosp/topic4_sef_hfo/cohort_dual_mode_recovery_20260928/patient_templates')


def export_one(sid, unit, dest):
    result = read(unit/'result.json')
    assert sha(unit/'trajectory.npz') == result['arrays_sha256']
    with np.load(PATIENT/sid/'templates.npz') as z:
        names = z['names'].astype(str).tolist()
        patient = z['profiles']
        patient_counts = z['contact_event_counts']
    with np.load(unit/'trajectory.npz') as z:
        model_names = z['contact_names'].astype(str).tolist()
        assert len(set(model_names)) == len(names) and set(model_names) == set(names)
        order = [model_names.index(n) for n in names]
        ids = z['primary_event_indices']
        times = z['centroid_ms'][ids][:, order]
        xy = z['contact_xy_mm'][order]
    ranks = event_ranks(times)
    seed = int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16)
    model = (fit_two_templates(ranks, seed=seed) if len(ranks) >= 2 else
             dict(status='INSUFFICIENT_MODEL_EVENTS'))
    labels = np.full(len(ids), -1, int)
    raw_labels = np.full(len(ids), -1, int)
    aligned = dict(matched_rhos=[None, None])
    arrays = dict(contact_names=names, contact_xy_mm=xy, model_event_ids=ids,
                  model_centroid_ms=times, model_ranks=ranks, patient_profiles=patient,
                  patient_contact_counts=patient_counts)
    if model['status'] == 'ESTIMABLE':
        aligned = align_profiles(model['profiles'], patient)
        order = aligned['model_cluster_for_TA_TB']
        raw_labels = model['labels']
        labels = np.argsort(order)[raw_labels]
        arrays.update(model_profiles=model['profiles'][order],
                      model_contact_counts=model['contact_event_counts'][order])
    arrays.update(model_raw_cluster_labels=raw_labels, model_TA_TB_labels=labels)
    save_npz(dest/'templates.npz', **arrays)
    with (dest/'events.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['event_id', 'mode', 'participating_contacts', 'first_centroid_ms', 'last_centroid_ms'])
        writer.writeheader()
        for event, label, row in zip(ids, labels, times):
            finite = row[np.isfinite(row)]
            writer.writerow(dict(event_id=int(event), mode=['MTA', 'MTB'][label] if label >= 0 else 'UNASSIGNED',
                participating_contacts=len(finite), first_centroid_ms=float(finite.min()) if len(finite) else None,
                last_centroid_ms=float(finite.max()) if len(finite) else None))
    status = model['status']
    if status == 'ESTIMABLE' and not np.isfinite(np.asarray(aligned['matched_rhos'], float)).all():
        status = 'PARTIAL_MATCH'
    summary = dict(subject=sid, source=str(unit), trajectory_sha256=result['arrays_sha256'],
        status=status, **aligned, n_events=len(ids),
        mode_counts=[int((labels == k).sum()) for k in [0, 1]],
        definition='Historical F propagation-order modes: model-only masked K=2 then one-to-one naming. These are not independently fitted geometric propagation angles.',
        statistical_unit='patient; individual events are within-patient observations')
    write(dest/'summary.json', summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--require-complete', action='store_true')
    args = ap.parse_args()
    verify_sources()
    plan = read(ROOT/'plan.json')
    destination = ROOT/'selected_mode_events'
    rows = []
    for sid in plan['subjects']:
        for k in (1, 2):
            run = run_dir(k, 0, sid)
            scores = [read(p) for p in (run/'scores').glob('*.json')]
            nomination = run/'nomination.json'
            if len(scores) != plan['epochs'] or not nomination.exists():
                if args.require_complete:
                    raise RuntimeError('Nomination unavailable: '+sid+' core '+str(k))
                continue
            selected = read(nomination)['selected']
            good = [s for s in scores if s['J'] is not None]
            expected = min(good, key=lambda r: (r['J'], r['epoch']))['candidate'] if good else None
            assert selected == expected
            if selected is None:
                rows.append(dict(subject=sid, core_count=k, status='NO_SCORABLE_CANDIDATE'))
                continue
            for phase, unit in [('training', run/'units'/selected)] + [
                    (f'noise_{noise}', run/'confirmation'/f'{selected}_{noise}') for noise in plan['confirmation_noises']]:
                if not (unit/'result.json').exists():
                    if args.require_complete:
                        raise RuntimeError('Replay unavailable: '+str(unit))
                    continue
                dest = destination/'per_subject'/sid/f'core{k}'/phase
                summary = export_one(sid, unit, dest)
                cached = read(run/'template_readout.json') if phase == 'training' else read(unit/'template_readout.json')
                assert summary['status'] == cached['status']
                np.testing.assert_allclose(np.asarray(summary['matched_rhos'], float), np.asarray(cached['matched_rhos'], float), rtol=0, atol=1e-12, equal_nan=True)
                rows.append(dict(subject=sid, core_count=k, phase=phase, candidate=selected,
                                 status=summary['status'], directory=str(dest)))
    write(destination/'manifest.json', dict(status='COMPLETE' if args.require_complete else 'PARTIAL',
        rows=rows, primary_statistic_unchanged=True, patient_count=len(plan['subjects'])))
    print('Exported', len(rows), 'selected working-point records')


if __name__ == '__main__':
    main()
