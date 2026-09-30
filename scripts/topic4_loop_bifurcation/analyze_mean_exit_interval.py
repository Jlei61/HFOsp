#!/usr/bin/env python3
"""Read the four registered physical-time probes; never infer stability from persistence."""
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
from mean_exit_interval import OUT, JOBS
from analyze_mean_boundary import first_low


def projection():
    geo = dict(np.load(ADAPTED/'geometry.npz'))
    E = geo['population'] == 0; sizes = geo['group_size']
    masks = [E]+[E & (geo['group_region'] == q) for q in range(3)]
    counts = np.bincount(geo['group_cell'][E], weights=sizes[E], minlength=400)
    proj = sparse.coo_matrix((sizes[E]/counts[geo['group_cell'][E]],
        (geo['group_cell'][E], np.flatnonzero(E))), shape=(400, len(E))).tocsr()
    return sizes, masks, counts, proj


def load_case(name, sizes, masks, counts, proj):
    folder = OUT/'runs'/name
    assert read(folder/'implementation_qa.json')['status'] == 'PASS'
    assert read(folder/'result.json')['full_resume_exceptK_exact']
    chunks = sorted((folder/'chunks').glob('*.npz')); assert len(chunks) == 100
    blocks = []
    for index, path in enumerate(chunks):
        assert tuple(map(int, path.stem.split('_'))) == (100*index, 100*(index+1))
        with np.load(path) as a:
            v = a['group_output'].astype(float)
            assert v.shape == (100, 9, len(sizes))
            rate = np.stack([np.average(v[:, 0, m], weights=sizes[m], axis=1) for m in masks], axis=1)
            drift = np.stack([np.average((v[:, 8, m]-v[:, 1, m])/5, weights=sizes[m], axis=1) for m in masks], axis=1)
            M = np.stack([np.average(v[:, 2, m], weights=sizes[m], axis=1) for m in masks], axis=1)
            field = np.asarray((proj@v[:, 0].T).T)
            assert np.allclose(field@counts/counts.sum(), rate[:, 0], atol=1e-5)
            blocks.append(dict(rate=rate, drift=drift, M=M, field=field, R=a['global_R_Hz'], G=30*a['global_s']))
    d = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    d['time'] = np.arange(1, 10001)/1000
    return d


def describe(name, d, counts):
    history, K = JOBS[name]; windows = []
    for lo, hi in [(0, 5), (5, 10)]:
        s = slice(lo*1000, hi*1000)
        ten = d['rate'][s].reshape(-1, 10, 4).mean(1)
        one = d['rate'][s].reshape(-1, 1000, 4).mean(1)
        spatial = d['field'][s].reshape(-1, 100, 400).mean(1)
        area = (spatial >= 100)@counts/counts.sum()
        jointquiet = np.all(ten[:, :3] < 5, axis=1)
        windows.append(dict(interval_s=[lo, hi], mean_rate_Hz=d['rate'][s].mean(0).tolist(),
            one_second_min_rate_Hz=one.min(0).tolist(), one_second_max_rate_Hz=one.max(0).tolist(),
            final_minus_first_one_second_rate_Hz=(one[-1]-one[0]).tolist(),
            mean_core_Zdot_if_released=d['drift'][s, 1:3].mean(0).tolist(),
            jointquiet_fraction_10ms=float(jointquiet.mean()),
            E_weighted_area_of_100ms_bins_ge100Hz=float(area.mean()),
            causal_R_range_Hz=[float(d['R'][s].min()), float(d['R'][s].max())],
            G_range=[float(d['G'][s].min()), float(d['G'][s].max())]))
    tail = windows[-1]
    label = 'QUIET_FINAL_FIVE_SECONDS' if tail['jointquiet_fraction_10ms'] >= .95 else 'ACTIVITY_REMAINS_FINITE_WINDOW'
    return dict(name=name, history=history, held_mean_K=K, windows=windows,
        descriptive_outcome=label, first100ms_causal_R_le5_s=first_low(d['time'], d['R']),
        statistical_unit='One copied-network conditional trajectory; not an independent native seed.',
        stability_certified=False)


def main(wait):
    dest = OUT/'analysis'; dest.mkdir(exist_ok=True)
    while not all((OUT/'runs'/n/'result.json').exists() for n in JOBS):
        failed = [n for n in JOBS if (OUT/'runs'/n/'progress.json').exists()
                  and read(OUT/'runs'/n/'progress.json')['status'] == 'FAILED']
        if failed:
            write(dest/'progress.json', dict(status='STOPPED_ON_WORKER_FAILURE', failed=failed)); return
        write(dest/'progress.json', dict(status='WAITING_FOUR_REGISTERED_PROBES', pid=os.getpid(), updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    sizes, masks, counts, proj = projection(); data = {}; rows = []; reference_rng = None
    for name in JOBS:
        data[name] = load_case(name, sizes, masks, counts, proj)
        rows.append(describe(name, data[name], counts))
        np.savez_compressed(dest/f'{name}_readouts.npz', **data[name])
        with np.load(OUT/'runs'/name/'final_state.npz') as state:
            rng = {k: state[k] for k in ['rng', 'external_rng', 'clock']}
            if reference_rng is None:reference_rng = rng
            else:
                for k in rng:assert np.array_equal(rng[k], reference_rng[k]), (name, k)
            assert int(state['clock'][0]) == 200000
            with np.load(OUT/'fields'/f'{name}.npz') as fields:
                assert np.array_equal(state['state'][:32000, :, 6], np.broadcast_to(fields['Z'][:, None], (32000, 64)))
                assert np.array_equal(state['state'][:32000, :, 7], np.broadcast_to(fields['K'][:, None], (32000, 64)))
    result = dict(status='COMPLETE_FOUR_REGISTERED_CONDITIONAL_PROBES', rows=rows,
        all_future_numerical_RNGs_paired_bitwise=True, full_ZK_fields_held_bitwise=True,
        regions=['All E', 'Core A E', 'Core B E', 'Surround E'],
        limits='Previous K9.35/K9.5 references precede these four probes in the numerical stream. Only the four new probes share the same future random streams. Finite persistence and quiet are descriptive outcomes, not certified stable branches; no autonomous loop was added.',
        formal_bifurcation_allowed=False, agent_visual='PENDING', human_visual='PENDING', producer_sha256=sha(__file__))
    write(dest/'result.json', result); plot(data, rows, counts)
    shutil.copy2(__file__, dest/'producer.py')
    write(dest/'progress.json', dict(status='COMPLETE', updated_epoch=time.time()))
    print(result, flush=True)


def plot(data, rows, counts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    colors = ['#344a80', '#d47b25', '#94588f', '#428f81']
    fig, ax = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
    for row, color in zip(rows, colors):
        name=row['name']; d=data[name]; K=row['held_mean_K']; tail=row['windows'][-1]
        label=f"{row['history'].capitalize()} → K {K:g}"
        ax[0, 0].plot(d['time'], d['R'], color=color, lw=.9, label=label)
        for core, ls in [(1, '-'), (2, '--')]:
            ax[0, 1].plot(np.arange(50, 10000, 100)/1000, d['rate'][:, core].reshape(-1, 100).mean(1), color=color, ls=ls, lw=.9)
            ax[0, 2].plot(np.arange(50, 10000, 100)/1000, d['drift'][:, core].reshape(-1, 100).mean(1), color=color, ls=ls, lw=.9)
        marker='s' if row['history']=='quiet' else 'o'
        ax[1, 0].scatter(K, tail['mean_rate_Hz'][0], c=color, marker=marker, s=45)
        ax[1, 1].scatter(K, tail['E_weighted_area_of_100ms_bins_ge100Hz']*100, c=color, marker=marker, s=45)
        ax[1, 2].scatter(K, tail['mean_core_Zdot_if_released'][0], c=color, marker=marker, s=45)
        ax[1, 2].scatter(K, tail['mean_core_Zdot_if_released'][1], facecolors='none', edgecolors=color, marker=marker, s=70)
    # Historical points remain open and unconnected. They use earlier noise streams.
    refs=[]
    for case, K, fname, prefix in [('high', 9.35, ROOT/'dynamic_mean_history_pair/analysis/high_readouts.npz', 'model_'),
                                 ('asymmetric', 9.35, ROOT/'dynamic_mean_history_pair/analysis/asymmetric_readouts.npz', 'model_'),
                                 ('high', 9.5, ROOT/'mean_boundary_correspondence/analysis/readouts.npz', 'model_')]:
        with np.load(fname) as z:
            rate=z[prefix+'rate'][5000:].mean(0); drift=z[prefix+'drift'][5000:].mean(0)
            f=z[prefix+'field'][5000:].reshape(-1, 100, 400).mean(1)
            area=((f>=100)@counts/counts.sum()).mean()
        marker='^' if case=='asymmetric' else 'o'
        ax[1, 0].scatter(K, rate[0], facecolors='none', edgecolors='.45', marker=marker, s=60)
        ax[1, 1].scatter(K, area*100, facecolors='none', edgecolors='.45', marker=marker, s=60)
        ax[1, 2].scatter([K, K], drift[1:3], facecolors='none', edgecolors='.45', marker=marker, s=60)
    ax[0, 0].set(ylabel='Causal E rate (Hz)', title='Complete-state continuation')
    ax[0, 0].legend(frameon=False, fontsize=8)
    ax[0, 1].set(ylabel='Core rate (Hz)', title='Core A solid; Core B dashed')
    ax[0, 2].set(ylabel='Core dZ/dt if released (1/s)', title='Z held: resource balance only')
    ax[0, 2].axhline(0, color='.6', lw=.7)
    for axis in ax[0]:axis.set(xlim=(0,10), xlabel='Time after K change (s)')
    for axis in ax[0, :2]:axis.set_ylim(bottom=0)
    ax[1, 0].set(ylabel='All-E mean rate (Hz)', title='Measured 5–10 s response', ylim=(-8,215))
    ax[1, 1].set(ylabel='E-weighted high-rate area (%)', title='100 ms spatial bins ≥100 Hz', ylim=(-2,52))
    ax[1, 2].set(ylabel='Core dZ/dt if released (1/s)', title='A filled; B open for new probes')
    ax[1, 2].axhline(0, color='.6', lw=.7)
    for axis in ax[1]:axis.set(xlabel='Held mean K / gL', xlim=(9.337,9.513), xticks=[9.35,9.40,9.45,9.50])
    fig.suptitle('Exit-related conditional responses at one fixed spatial Z field', weight='bold')
    fig.text(.5, -.018, 'Grey open symbols: earlier references (different future stream). Points are measured responses; stability is not certified.', ha='center', fontsize=9)
    for ext in ['png','svg']:fig.savefig(ROOT/f'figures/mean_exit_interval.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    centers=np.load(ROOT/'native_slices/geometry.npz')['centers_mm']
    fig, axes=plt.subplots(1,4,figsize=(12,3.6),layout='constrained')
    for axis, row in zip(axes, rows):
        f=data[row['name']]['field'][5000:].mean(0)
        im=axis.imshow(f.reshape(20,20),origin='lower',extent=(0,20,0,20),vmin=0,vmax=500,cmap='magma',interpolation='nearest')
        for xy in centers:axis.add_patch(Circle(xy,1.5,fill=False,edgecolor='#00c3c5',lw=.9))
        axis.set(title=f"{row['history'].capitalize()} → K {row['held_mean_K']:g}",xlabel='x (mm)',ylabel='y (mm)',xticks=[0,10,20],yticks=[0,10,20])
    fig.colorbar(im,ax=axes.tolist(),shrink=.7,label='E rate, 5–10 s (Hz)')
    fig.suptitle('Spatial responses: same held Z field, graph and future input streams',weight='bold')
    for ext in ['png','svg']:fig.savefig(ROOT/f'figures/mean_exit_interval_spatial.{ext}',dpi=180)
    plt.close(fig)
    p=ROOT/'figures/README.md'
    with p.open('a') as f:
        for stem, description in [('mean_exit_interval', '在已核验退出区间内，从完整高态补三个K点，并把完整静默态的K退回9.35；四条十秒轨迹共用未来随机数。图中仅有实际条件响应点，另标旧参考，显示双核率、资源收支及空间招募面积。'),
                                  ('mean_exit_interval_spatial', '四条已登记条件试验后五秒的原生空间坐标活动场。核心、色标和场族保持一致，空间格率按E细胞数加权。')]:
            f.write(f'\n\n### {stem}.png / {stem}.svg\n{description}\n**关注点**：固定Z不发生实际资源恢复；短时持续不认证吸引子或稳定分支，跨历史点不连成虚构不稳定支，人工待审。\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)
