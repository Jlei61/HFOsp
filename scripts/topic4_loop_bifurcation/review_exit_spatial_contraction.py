#!/usr/bin/env python3
"""Read-only decomposition of the matched exit's spatial activity loss."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import shutil
import time
import numpy as np
from campaign import ROOT, read, write, sha
from coupled_density_exit import ADAPTED

OUT = ROOT/'exit_spatial_contraction_review'


def main():
    assert read(ROOT/'mean_boundary_correspondence/analysis/result.json')['retained']
    assert read(ROOT/'mean_boundary_resolution/analysis/result.json')['retained']
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    write(OUT/'contract.json', dict(status='REGISTERED_READONLY_SPATIAL_DECOMPOSITION', created_epoch=time.time(),
        question='During the matchedK9.5 exit, how much of the populationrate decline accompanies contraction of the high-rate spatial region versus reduced firing within it?',
        observable='Actual400native spatialbins, weighted by fixedEcellcounts, incommon100mswindows0-2.2s. Above100Hz primary descriptivearea;50and200Hz check whether the spatialinterpretation depends on thisdisplaythreshold. Compute high-bin weightedarea, conditionalhigh/lowmeans andexactmean decomposition.',
        baseline='First0-.1s window ofeachsamecondition trajectory; native comparedwithR64andR256 withouttimealignment. No model fit or new simulation.',
        unit='Timewindows andspacebins within the same conditionalexperiment; not independentnative seeds. High-ratebin area is not the fraction of individualspikingcells, nor a propagationdirection estimator.',
        claim_limit='Descriptive contraction duringfixedZ/Kexit. Does not establish an anatomicalaxis cause, frontdepinning bifurcation, or a closedone-dimensional frontcoordinate.', producer_sha256=sha(__file__)))
    d = dict(np.load(ROOT/'mean_boundary_correspondence/analysis/readouts.npz'))
    q = dict(np.load(ROOT/'mean_boundary_resolution/analysis/readouts.npz'))
    geo = np.load(ADAPTED/'geometry.npz');E = geo['population'] == 0
    weights = np.bincount(geo['group_cell'][E], weights=geo['group_size'][E], minlength=400);weights /= weights.sum()
    fields = {'Native': d['native_field'][:440].reshape(22, 20, 400).mean(1),
              'R64': d['model_field'][:2200].reshape(22, 100, 400).mean(1),
              'R256': q['R256_field'][:2200].reshape(22, 100, 400).mean(1)}
    records = {};rows = []
    for name, field in fields.items():
        allmean = field@weights
        for threshold in [50, 100, 200]:
            hot = field >= threshold;area = hot@weights
            high = np.divide((field*hot)@weights, area, out=np.full(22, np.nan), where=area > 0)
            low = np.divide((field*~hot)@weights, 1-area, out=np.full(22, np.nan), where=area < 1)
            rebuilt = area*np.nan_to_num(high)+(1-area)*np.nan_to_num(low)
            assert np.max(abs(rebuilt-allmean)) < 1e-10
            records[name+'_'+str(threshold)] = np.stack([allmean, area, high, low], axis=1)
            selected = []
            for i in [0, 10, 15, 18, 19, 20, 21]:
                selected.append(dict(window_s=[round(i*.1, 3), round((i+1)*.1, 3)], all_E_Hz=float(allmean[i]),
                    high_bin_weighted_fraction=float(area[i]), high_bin_mean_Hz=None if not np.isfinite(high[i]) else float(high[i]),
                    low_bin_mean_Hz=None if not np.isfinite(low[i]) else float(low[i])))
            rows.append(dict(source=name, threshold_Hz=threshold, selected_windows=selected))
    np.savez_compressed(OUT/'readouts.npz', time_s=(np.arange(22)+.5)*.1, E_weights=weights, **records)
    write(OUT/'result.json', dict(status='COMPLETE_READONLY_SPATIAL_DECOMPOSITION', rows=rows, exact_rate_decomposition=True,
        statistical_limit='Weightedspatial-bin area, not a direct per-neuron activityfraction; noindependentreplicate inference.',
        formal_bifurcation_allowed=False, agent_visual='PENDING', human_visual='PENDING', producer_sha256=sha(__file__)))
    plot(records);shutil.copy2(__file__, OUT/'producer.py')


def plot(records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.2), layout='constrained');t = (np.arange(22)+.5)*.1
    for name, color in [('Native', 'black'), ('R64', '#8266ad'), ('R256', '#c07935')]:
        x = records[name+'_100']
        for a, col, scale in zip(ax, [0, 1, 2], [1, 100, 1]):a.plot(t, x[:, col]*scale, color=color, lw=1.2, label=name)
    for a, ylabel in zip(ax, ['All-E rate (Hz)', 'E-weighted high-rate area (%)', 'Mean rate within high-rate bins (Hz)']):a.set(xlim=(0, 2.2), xlabel='Time from high history (s)', ylabel=ylabel)
    ax[0].legend(frameon=False);ax[1].set_title('Spatial bins above 100 Hz', fontsize=10)
    ax[2].set_title('Undefined once high-rate area is zero', fontsize=9)
    fig.suptitle('Spatial contribution to activity loss at held K = 9.5', weight='bold')
    for ext in ['png', 'svg']:fig.savefig(ROOT/f'figures/exit_spatial_contraction.{ext}', dpi=180)
    plt.close(fig)
    p = ROOT/'figures/README.md';title = '### exit_spatial_contraction.png / exit_spatial_contraction.svg'
    if title not in p.read_text():
        with p.open('a') as f:f.write('\n\n'+title+'\n在相同100毫秒窗口内，将全E均率分解为高率空间范围、该范围内均率及范围外均率；主图阈值100Hz，50和200Hz读出保存在JSON。三条线为原生、64复制及256复制，无时间对齐。\n**关注点**：高率空间范围按固定E细胞数加权，不等于逐细胞活跃比例；该描述不认证连接轴因果、前沿分岔或一维闭合。\n')


if __name__ == '__main__':main()
