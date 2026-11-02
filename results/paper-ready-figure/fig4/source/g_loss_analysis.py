"""Freeze and reproduce the three actual training components at all G workpoints."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from i_j_analysis import mapped_features, d_off

HERE = Path(__file__).resolve().parent
GROUPS = ('rank_pattern', 'local_and_interrod_timing', 'participation')
SYMBOLS = ('r', 'o', 'p')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rows = list(csv.DictReader((HERE/'B_visible_workpoints.csv').open()))
    spec = json.loads((HERE/'i_j_frozen_spec.json').read_text())
    with np.load(HERE/'i_j_frozen_parameters.npz') as z:
        pars = dict(z)
    with np.load(HERE/'i_events.npz') as z:
        names = z['contact_names'].tolist()
    frozen_path = HERE/'g_loss_events.npz'
    scores_dir = HERE/'g_original_scores'
    scores_dir.mkdir(exist_ok=True)
    arrays = dict(np.load(frozen_path)) if frozen_path.exists() else {'contact_names': np.array(names)}
    output, provenance = [], []
    assert len(rows) == 194 and len({r['candidate'] for r in rows}) == 194
    for row in rows:
        cid = row['candidate']
        trajectory = Path(row['source'])
        score_path = trajectory.parents[5]/'scores'/(cid+'.json')
        frozen_score = scores_dir/(cid+'.json')
        if not frozen_score.exists():
            frozen_score.write_bytes(score_path.read_bytes())
        score = json.loads(frozen_score.read_text())
        assert score['objective_sha256'] == spec['original_pickle_sha256']
        assert score['source_sha256'] == row['source_sha256']
        if cid+'_centroid_ms' not in arrays:
            assert sha(trajectory) == row['source_sha256']
            raw = json.loads(trajectory.read_text())
            assert raw['actual_duration_ms'] == 60000
            assert raw['physical_status'] == 'COMPLETE_NO_RUNAWAY'
            with np.load(trajectory.with_suffix('.npz')) as z:
                assert z['contact_names'].tolist() == names
                ix = np.array([int(i) for i in z['primary_event_indices']
                               if raw['events'][int(i)]['window_ms'][0] >= 1500
                               and raw['events'][int(i)]['window_ms'][1] <= 60000])
                arrays[cid+'_centroid_ms'] = z['centroid_ms'][ix]
                arrays[cid+'_event_indices'] = ix
        t = arrays[cid+'_centroid_ms']
        assert len(t) == int(row['N']) == score['N']
        features = mapped_features(t, pars, spec)
        components = [d_off(features[g], pars[g+'_target'])/spec['scales'][g] for g in GROUPS]
        recorded = [score['groups'][g]['scaled'] for g in GROUPS]
        np.testing.assert_allclose(components, recorded, rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.mean(components), float(row['J']), rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.mean(recorded), score['J'], rtol=0, atol=1e-12)
        output.append(dict(candidate=cid, stage=int(row['stage']), N=len(t),
                           **{'L_'+s: v for s, v in zip(SYMBOLS, recorded)}, J=score['J']))
        provenance.append(dict(candidate=cid, stage=int(row['stage']),
                               trajectory_source=str(trajectory), trajectory_sha256=row['source_sha256'],
                               score_source=str(score_path), score_sha256=sha(frozen_score),
                               maximum_component_error=float(np.max(np.abs(np.array(components)-recorded)))))
    if not frozen_path.exists():
        np.savez_compressed(frozen_path, **arrays)
    with (HERE/'g_training_loss_profiles.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    report = dict(status='PASS', profile_count=len(output), all_components_recomputed=True,
                  formula='J = (L_r + L_o + L_p) / 3',
                  component_definition='L_g = D_off,g / s_g; original fixed feature map, patient target and CAL scale',
                  groups=GROUPS, objective_sha256=spec['original_pickle_sha256'],
                  identical_objective_all_194_profiles=True, source_selection='Same 194 visible workpoints, stages 6-16',
                  statistical_unit='One native-SNN trajectory per workpoint; selected primary events after 1.5-s burnin',
                  stage_interpretation='Historical search stages, not independent optimization repeats or a monotone learning curve. Stages 13-16 also optimized J_joint; this panel shows only their common three-component J.',
                  no_new_simulations=True, maximum_component_error=max(r['maximum_component_error'] for r in provenance),
                  records=provenance)
    (HERE/'g_loss_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='records'})


if __name__ == '__main__':
    main()
