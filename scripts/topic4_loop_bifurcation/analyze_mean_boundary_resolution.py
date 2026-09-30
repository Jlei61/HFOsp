#!/usr/bin/env python3
"""Inspect the one-point resolution check on a shared physical clock."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import shutil
import time
import numpy as np
from scipy import sparse
from campaign import ROOT, read, write, sha
from coupled_density_exit import ADAPTED
from mean_boundary_resolution import OUT, BOUNDARY
from analyze_mean_boundary import first_low


def main(wait):
    dest = OUT/'analysis';dest.mkdir(exist_ok=True)
    while not (OUT/'result.json').exists():
        if read(OUT/'progress.json')['status'] == 'FAILED':
            write(dest/'progress.json', dict(status='STOPPED_ON_WORKER_FAILURE'));return
        write(dest/'progress.json', dict(status='WAITING_ONE_R256_RUN', pid=os.getpid(), updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    geo = dict(np.load(ADAPTED/'geometry.npz'));E = geo['population'] == 0;sizes = geo['group_size']
    masks = [E]+[E & (geo['group_region'] == q) for q in range(3)]
    counts = np.bincount(geo['group_cell'][E], weights=sizes[E], minlength=400)
    proj = sparse.coo_matrix((sizes[E]/counts[geo['group_cell'][E]], (geo['group_cell'][E], np.flatnonzero(E))), shape=(400, len(E))).tocsr()
    data = {}
    for key, folder in [('R64', BOUNDARY/'model/upper'), ('R256', OUT)]:
        chunks = sorted((folder/'chunks').glob('*.npz'))[:40];assert len(chunks) == 40;blocks = []
        for index, path in enumerate(chunks):
            assert tuple(map(int, path.stem.split('_'))) == (100*index, 100*(index+1))
            with np.load(path) as a:
                v = a['group_output'].astype(float)
                rates = np.stack([np.average(v[:, 0, m], weights=sizes[m], axis=1) for m in masks], axis=1)
                drift = np.stack([np.average((v[:, 8, m]-v[:, 1, m])/5, weights=sizes[m], axis=1) for m in masks], axis=1)
                field = np.asarray((proj@v[:, 0].T).T)
                assert np.allclose(field@counts/counts.sum(), rates[:, 0], atol=1e-5)
                blocks.append(dict(rate=rates, drift=drift, field=field, R=a['global_R_Hz'], G=30*a['global_s']))
        data[key] = {k: np.concatenate([d[k] for d in blocks]) for k in blocks[0]}
        data[key]['time'] = np.arange(1, 4001)/1000
    a, b = data['R64'], data['R256'];ta = first_low(a['time'], a['R']);tb = first_low(b['time'], b['R'])
    delta = b['field'][3000:].mean(0)-a['field'][3000:].mean(0)
    field_error = float(np.sqrt(np.average(delta**2, weights=counts)))
    drift_error = (b['drift'][3000:].mean(0)-a['drift'][3000:].mean(0)).tolist()
    bins = b['field'].reshape(40, 100, 400).mean(1)-a['field'].reshape(40, 100, 400).mean(1)
    transient = np.sqrt(np.average(bins**2, weights=counts, axis=1))
    guards = dict(first_low_time=ta is not None and tb is not None and abs(tb-ta) <= .1,
        both_tail_quiet=bool(max(a['rate'][3000:, :3].mean(0).max(), b['rate'][3000:, :3].mean(0).max()) < 5),
        tail_spatial_RMS=field_error <= 5, tail_core_Zdrift=max(abs(x) for x in drift_error[1:3]) <= .005)
    result = dict(status='COMPLETE_BOUNDARY_NUMERICAL_RESOLUTION', guards=guards, retained=all(guards.values()),
        first100ms_low_s=dict(R64=ta, R256=tb), shift_s=None if ta is None or tb is None else tb-ta,
        tail_spatial_RMS_Hz=field_error, tail_Zdot_difference=drift_error,
        unaligned100ms_spatial_RMS_Hz=transient.tolist(), maximum_transient_RMS_Hz=float(transient.max()),
        interpretation='One physical point and shared initialhistory, foursecond resolution test. More numericalcopies do not add native biologicalreplicates. Does not prove largeRlimit, time-step convergence or branchstability.',
        formal_bifurcation_allowed=False, agent_visual='PENDING', human_visual='PENDING', producer_sha256=sha(__file__))
    write(dest/'result.json', result)
    np.savez_compressed(dest/'readouts.npz', **{key+'_'+k: v for key, d in data.items() for k, v in d.items()}, transient_RMS=transient)
    plot(data, transient);shutil.copy2(__file__, dest/'producer.py')
    write(dest/'progress.json', dict(status='COMPLETE', updated_epoch=time.time()));print(result, flush=True)


def plot(data, transient):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    for label, color in [('R64', '#8266ad'), ('R256', '#c07935')]:
        d = data[label];ax[0].plot(d['time'], d['R'], color=color, lw=1, label=label)
        for core, ls in [(1, '-'), (2, '--')]:ax[1].plot(d['time'], d['drift'][:, core], color=color, ls=ls, lw=1)
    ax[0].set_yscale('symlog', linthresh=1)
    ax[0].set(xlabel='Time (s)', ylabel='Causal E rate (Hz)', xlim=(0, 4));ax[0].axhline(5, color='.6', ls=':', lw=.8)
    ax[0].legend(frameon=False);ax[1].set(xlabel='Time (s)', ylabel='Core dZ/dt if released (1/s)', title='Core A solid; Core B dashed', xlim=(0, 4))
    ax[1].axhline(0, color='.6', lw=.8)
    ax[2].plot((np.arange(40)+.5)*.1, transient, color='black');ax[2].set(xlabel='Time (s)', ylabel='Spatial field RMS difference (Hz)', title='Common 100 ms bins; no time shift', xlim=(0, 4))
    fig.suptitle('Exit at held K = 9.5: numerical replica resolution', weight='bold')
    for ext in ['png', 'svg']:fig.savefig(ROOT/f'figures/mean_boundary_resolution.{ext}', dpi=180)
    plt.close(fig)
    p = ROOT/'figures/README.md';title = '### mean_boundary_resolution.png / mean_boundary_resolution.svg'
    if title not in p.read_text():
        with p.open('a') as f:f.write('\n\n'+title+'\n相同K9.5条件、完整高态初值及外源规律，只将数值复制从64增到256，比较退出时刻、核心收支及未作时间对齐的100毫秒空间差。此图使用共同物理时钟，不以相位平移掩盖时间误差。\n**关注点**：复制分辨率证据仅限该工作点四秒，不是增加原生种子，也不是稳定性或分岔认证。\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--wait', action='store_true');main(p.parse_args().wait)
