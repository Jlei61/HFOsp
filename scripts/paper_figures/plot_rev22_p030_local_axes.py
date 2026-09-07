#!/usr/bin/env python3
"""Measure p030 incoming EE edge tensors by target location, on both graph versions."""
from pathlib import Path
import hashlib
import json
import pickle
import sys
from unittest.mock import patch

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

ROOT = Path(__file__).resolve().parents[2]
ART = Path('/home/honglab/leijiaxin/HFOsp')
for p in (ROOT, ROOT/'src/snn_engine'):
    sys.path.insert(0, str(p))
from src.topic4_zm_ictal_transition import build_substrate, load_round_config
from src.topic4_manual_dual_core import budget_matched_dual_core_h


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def geometry(sums):
    weight, xx, xy, yy = sums
    moment = np.array([[xx, xy], [xy, yy]])/weight
    vals, vectors = np.linalg.eigh(moment)
    if not np.isfinite(vals).all() or vals[0] <= 0:
        raise RuntimeError('invalid edge second moment')
    axis = vectors[:, -1]
    return {'angle_deg': float((np.degrees(np.arctan2(axis[1], axis[0]))+90)%180-90),
            'aspect_ratio': float(np.sqrt(vals[1]/vals[0]))}


def measure(sub, n_bins=10):
    pos = np.asarray(sub.net['pos'], float)
    ne = sub.n_e
    xybin = np.minimum((pos[:ne]/20*n_bins).astype(int), n_bins-1)
    target_bin = xybin[:, 1]*n_bins+xybin[:, 0]
    moments = np.zeros((n_bins*n_bins, 4))
    self_edges = 0
    for matrix in sub.net['ampa_by_delay']:
        coo = matrix.tocoo(copy=False)
        keep = coo.row < ne
        rows, cols, w = coo.row[keep], coo.col[keep], coo.data[keep]
        if np.any(cols >= ne):
            raise RuntimeError('non-E source in AMPA graph')
        dx, dy = (pos[cols]-pos[rows]).T
        bins = target_bin[rows]
        for j, values in enumerate((w, w*dx*dx, w*dx*dy, w*dy*dy)):
            moments[:, j] += np.bincount(bins, weights=values, minlength=n_bins*n_bins)
        self_edges += int(np.count_nonzero(rows == cols))
    counts = np.bincount(target_bin, minlength=n_bins*n_bins)
    cells = []
    for b in range(n_bins*n_bins):
        if counts[b] == 0:
            raise RuntimeError('empty target bin')
        cells.append({'x_mm': (b%n_bins+.5)*20/n_bins,
                      'y_mm': (b//n_bins+.5)*20/n_bins,
                      'n_targets': int(counts[b]), **geometry(moments[b])})
    return {'global': geometry(moments.sum(0)), 'cells': cells,
            'self_edges': self_edges, 'bin_width_mm': 20/n_bins}


def main():
    audit_path = ROOT/'results/topic4_sef_hfo/substrate_autapse_correction/graph_rebuild_audit.json'
    manifest_path = ART/'results/topic4_sef_hfo/data_driven_dual_core_interictal_identifiability/response_fit/final_execution_candidate_manifest.json'
    config_path = ROOT/'config/topic4_rev22_dci_transition_execution.json'
    audit = json.loads(audit_path.read_text())
    candidate = next(c for c in json.loads(manifest_path.read_text())['candidates'] if c['candidate_id']=='dci_p030')
    m = candidate['mechanisms']
    config = load_round_config(config_path)
    results = {}
    for arm, key in [('legacy', 'old_cache'), ('corrected', 'new_cache')]:
        record = audit[key]
        graph_path = Path(record.get('frozen_cache_path', record.get('path')))
        graph_sha = record.get('cache_sha256', record.get('sha256'))
        if sha(graph_path) != graph_sha:
            raise RuntimeError('graph hash changed')
        with open(graph_path, 'rb') as f:
            cached = pickle.load(f)
        def loader(*args, **kwargs):
            return cached['net'], cached['NE'], cached['NI'], True, {'path': str(graph_path), 'sha256': graph_sha}
        with patch('scripts.run_topic4_rev9_node_kick_canary._load_network', loader):
            sub = build_substrate(config, 'joint_04_control', 2511,
                cache_dir=str(graph_path.parent), ee_dose=m['g_EE'], etoi_dose=m['g_EtoI'],
                node_candidate_override=candidate['node_field'],
                ee_ellipse_angle_deg=m['ellipse_angle_deg'],
                ee_ellipse_aspect_ratio=m['ellipse_aspect_ratio'],
                ee_ellipse_reference_angle_deg=m['ellipse_reference_angle_deg'],
                ee_ellipse_reference_aspect_ratio=m['ellipse_reference_aspect_ratio'],
                artifact_root=ART, topology_seed=2511, dynamics_seed=2511)
        del cached
        results[arm] = {**measure(sub), 'graph': {'path': str(graph_path), 'sha256': graph_sha}}
        centers = np.asarray(candidate['node_field']['centers_mm'])
        _, field = budget_matched_dual_core_h(sub.positions_e, centers, target_count=1499)
        del sub
        print(arm, results[arm]['global'], flush=True)
    contact_path = ART/'results/topic4_sef_hfo/data_driven_core_field_rev10_sa/shaft_aware_target/contact_shaft_contract.json'
    contacts = np.array([c['sheet_xy_mm'] for c in json.loads(contact_path.read_text())['contacts']])
    out = ROOT/'results/topic4_sef_hfo/rev22_candidate_axis_review'
    figures = out/'figures'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    fig.subplots_adjust(left=.07, right=.94, bottom=.20, top=.80, wspace=.25)
    fig.suptitle('p030: measured connection axes throughout the sheet', fontsize=17, weight='bold', y=.97)
    fig.text(.07,.90,'Incoming E-to-E edge moments, grouped by target position. Topology 2511; original edge-core locations.',fontsize=10)
    for ax, (arm, data) in zip(axes, results.items()):
        for cell in data['cells']:
            x, y, ar, angle = cell['x_mm'], cell['y_mm'], cell['aspect_ratio'], cell['angle_deg']
            ax.add_patch(Ellipse((x,y),1.65,1.65/ar,angle=angle,
                         edgecolor='#245C3F',facecolor='#245C3F',alpha=.30,lw=.6))
            v = .825*np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
            ax.plot([x-v[0],x+v[0]],[y-v[1],y+v[1]],color='#245C3F',lw=.8)
        for c in centers:
            ax.add_patch(Circle(c,field['distance_cutoff_mm'],edgecolor='#8757A0',facecolor='#8757A0',alpha=.23))
        ax.plot(centers[:,0],centers[:,1],ls=':',color='#8757A0',lw=1.2)
        ax.scatter(contacts[:,0],contacts[:,1],s=14,facecolors='white',edgecolors='#777777',zorder=4)
        g=data['global']
        label='Original graph' if arm=='legacy' else 'Autapse-corrected graph'
        ax.set_title(f"{label}\nGlobal axis {g['angle_deg']:.2f}°  |  AR {g['aspect_ratio']:.3f}",fontsize=12)
        ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xticks=[0,5,10,15,20],yticks=[0,5,10,15,20],xlabel='x (mm)',ylabel='y (mm)')
    fig.text(.07,.065,'Ellipse major lengths are normalized; they show local axis and aspect ratio, not connection range.\nEach ellipse pools targets in one 2 × 2 mm bin. Purple circles mark the two cores.',fontsize=10)
    stem=figures/'rev22_p030_local_weighted_axes'
    for ext in ('png','pdf','svg'):
        fig.savefig(stem.with_suffix('.'+ext),dpi=220)
    plt.close(fig)
    report={'candidate_id':'dci_p030','topology_seed':2511,'arms':results,
            'scope':'Weighted, uncentered edge second moment; not spike propagation direction. Local pooling by target, not source. Single topology, with no dynamical inference.',
            'source_hashes':{str(p):sha(p) for p in (audit_path,manifest_path,config_path,contact_path,Path(__file__),ROOT/'src/topic4_zm_ictal_transition.py')},
            'output_hashes':{str(stem.with_suffix('.'+ext)):sha(stem.with_suffix('.'+ext)) for ext in ('png','pdf','svg')}}
    (out/'p030_local_axis_measurement.json').write_text(json.dumps(report,indent=2)+'\n')
    readme = figures/'README.md'
    if readme.exists():
        readme.write_text(readme.read_text().split('\n### rev22_p030_local_weighted_axes')[0])
    with open(readme,'a') as f:
        f.write('\n### rev22_p030_local_weighted_axes.png / .pdf / .svg\n将 p030 实际重加权后的 E→E 连接按靶细胞所在的2×2 mm网格分组，逐格计算加权边二阶矩的主轴和轴比。左右使用相同种子、位置和候选参数，仅原始采样图与排除自连接后的图不同；椭圆长径统一，不表达真实连接距离。\n**关注点**：连接各向异性遍布全场，边缘可改变局部方向；主轴是结构统计，不是活动传播方向。\n')


if __name__=='__main__':
    main()
