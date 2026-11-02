"""Numerical and render-level checks for the unified loss revision."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from i_panel import render_i, GROUPS
from c_panel import render_c

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
OLD = OUT.parent/'snn_layout_g_clear_lines_20261026'
FORMAL = OUT.parent.parent
ROOT = Path('/home/honglab/leijiaxin/HFOsp')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    grows = list(csv.DictReader((HERE/'g_training_loss_profiles.csv').open()))
    irows = list(csv.DictReader((HERE/'i_angle_response.csv').open()))
    assert len(grows)==194 and len(irows)==13
    glookup = {r['candidate']:r for r in grows}
    gv = np.array([[float(r['L_'+g]) for g in ('r','o','p')] for r in grows])
    np.testing.assert_allclose(gv.mean(1),[float(r['J']) for r in grows],rtol=0,atol=1e-12)
    iv = np.array([[float(r['J_component_'+g]) for g in GROUPS] for r in irows])
    total = np.array([float(r['optimizer_J']) for r in irows])
    np.testing.assert_allclose(iv.mean(1),total,rtol=0,atol=1e-12)
    shared = [r for r in irows if r['candidate'] in glookup]
    assert len(shared)==12
    for r in shared:
        g = glookup[r['candidate']]
        np.testing.assert_allclose([float(g['L_'+v]) for v in ('r','o','p')],
                                   [float(r['J_component_'+v]) for v in GROUPS],rtol=0,atol=1e-12)
    with np.load(HERE/'i_bootstrap_draws.npz') as z:
        draws = z['scaled_components_13']
    np.testing.assert_allclose(draws.var(0,ddof=1),
        [[float(r['J_component_'+g+'_bootstrap_variance']) for g in GROUPS] for r in irows],rtol=1e-12,atol=1e-14)
    joint_variance = draws.mean(2).var(0,ddof=1)
    np.testing.assert_allclose(joint_variance,[float(r['J_bootstrap_variance']) for r in irows],rtol=1e-12,atol=1e-14)
    covariance = np.array([np.cov(draws[:,i,:],rowvar=False,ddof=1) for i in range(13)])
    np.testing.assert_allclose(joint_variance,covariance.sum((1,2))/9,rtol=1e-12,atol=1e-14)
    fig=plt.figure(figsize=(320/25.4,232.52/25.4),dpi=140)
    ax=render_i(fig,(255/320,15/232.52,51/320,51/232.52))
    assert len(fig.axes)==1 and ax.get_ylabel()=='Loss'
    lines={l.get_gid():l for l in ax.lines}
    for j,g in enumerate(('r','o','p')):
        np.testing.assert_array_equal(lines['i_L_'+g].get_ydata(),iv[:,j])
    np.testing.assert_array_equal(lines['i_total_J'].get_ydata(),total)
    result=render_c(fig,(164.43/320,144.765915178571/232.52,151/320,80.790178571429/232.52))
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    labels={t.get_gid():t for t in fig.texts if str(t.get_gid()).startswith('loss_label_')}
    boxes=[labels['loss_label_'+g].get_window_extent(renderer) for g in ('Rank pattern','Timing','Participation')]
    assert not any(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:])
    assert [v['label'] for v in result['metadata']['display']['loss_terms']]==['Rank pattern','Timing','Participation']
    plt.close(fig)
    arrays=[]
    for p in (OLD/'source').rglob('*'):
        if p.suffix in ('.npz','.csv'):
            rel=p.relative_to(OLD/'source')
            assert sha(p)==sha(HERE/rel),str(rel)
            arrays.append(str(rel))
    preserved={k:sha(OLD/f'figures/fig4-panel{k}.png')==sha(OUT/f'figures/fig4-panel{k}.png') for k in 'abcdefhi'}
    assert all(preserved.values()),preserved
    formal=json.loads((FORMAL/'release_manifest.json').read_text())
    for name,meta in formal['files'].items():
        assert sha(FORMAL/name)==meta['sha256'],name
    registry=json.loads((OUT/'candidate_registry.json').read_text())
    assert not registry['text_outside_canvas']
    report=dict(status='PASS',G_profiles=194,G_components=582,G_raw_event_reproduction='source/g_loss_validation.json',
                I_angle_points=13,I_matches_12_shared_G_workpoints=True,
                I_reference_point='g1_axis3_plus is the separately retained earlier reference; not in the 194 G workpoints',
                total_equals_three_component_mean=True,I_single_loss_axis=True,
                I_component_and_total_SD_verified=True,total_variance_includes_covariance=True,
                C_loss_label_collisions=False,C_names_match_G_I=True,
                unchanged_prior_numeric_files=len(arrays),unchanged_panel_PNG=preserved,
                formal_manifest_files_unchanged=len(formal['files']),text_outside_canvas=[])
    (OUT/'qa/training_loss_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__':
    main()
