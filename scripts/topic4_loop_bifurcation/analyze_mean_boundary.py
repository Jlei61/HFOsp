#!/usr/bin/env python3
"""Matched finite-time upper-exit comparison, without a bifurcation label."""
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
from analyze_native import analyze, original
from mean_boundary_correspondence import OUT, MODEL, NATIVE_OUT, NAME, model as implementation


def first_low(t, r):
    count = 0
    for i, x in enumerate(r):
        count = count+1 if x <= 5 else 0
        if count == 100:return float(t[i-99])
    return None


def main(wait):
    dest = OUT/'analysis';dest.mkdir(exist_ok=True)
    while not ((NATIVE_OUT/'result.json').exists() and (MODEL/'upper/result.json').exists()
               and read(MODEL/'upper/result.json').get('both_RNGs_paired_with_K9p35', False)):
        write(dest/'progress.json', dict(status='WAITING_NATIVE_AND_MODEL_UPPER_EXIT', pid=os.getpid(), updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    native_row, drive = analyze(NATIVE_OUT, NAME)
    n = dict(np.load(NATIVE_OUT/'extended_analysis'/f'{NAME}_readouts.npz'))
    folder = NATIVE_OUT/'runs'/NAME
    me = original.load(folder/'mechanism_chunks', ['time_ms', 'global_E_rate_Hz', 'global_raw_conductance_ratio'])
    dr = original.load(folder/'conditional_drift_chunks', ['time_ms', 'values'])
    nt = me['time_ms']/1000-72.;dt = dr['time_ms']/1000-72.
    use = (nt >= -1e-9) & (nt < 10-1e-9);dd = (dt > 1e-9) & (dt <= 10+1e-9)
    native = dict(field=n['field_rate_5ms_Hz'], rate=n['rate_5ms_Hz'], time=nt[use],
        R=me['global_E_rate_Hz'][use], G=me['global_raw_conductance_ratio'][use], drift=dr['values'][dd, :, 0], drift_time=dt[dd])
    assert native['field'].shape == (2000, 400) and native['R'].shape == (10000,) and native['drift'].shape == (500, 4)
    inputs = []
    for p in sorted((implementation.base.MATCHED/'runs/high_history_constant_background/chunks').glob('*.npz')):
        with np.load(p) as z:inputs.append(z['inputs'])
    reference_inputs = np.concatenate(inputs)
    assert reference_inputs.shape == (100, 4) and drive.shape == (100, 3)
    assert np.array_equal(reference_inputs[:, 0], 72000+np.arange(100)*100)
    assert np.array_equal(drive, reference_inputs[:, 1:])
    geo = dict(np.load(ADAPTED/'geometry.npz'));E = geo['population'] == 0;sizes = geo['group_size']
    masks = [E]+[E & (geo['group_region'] == q) for q in range(3)]
    counts = np.bincount(geo['group_cell'][E], weights=sizes[E], minlength=400)
    proj = sparse.coo_matrix((sizes[E]/counts[geo['group_cell'][E]], (geo['group_cell'][E], np.flatnonzero(E))), shape=(400, len(E))).tocsr()
    chunks = sorted((MODEL/'upper/chunks').glob('*.npz'));assert len(chunks) == 100;blocks = []
    for index, path in enumerate(chunks):
        assert tuple(map(int, path.stem.split('_'))) == (100*index, 100*(index+1))
        with np.load(path) as a:
            v = a['group_output'].astype(float)
            rates = np.stack([np.average(v[:, 0, m], weights=sizes[m], axis=1) for m in masks], axis=1)
            drift = np.stack([np.average((v[:, 8, m]-v[:, 1, m])/5, weights=sizes[m], axis=1) for m in masks], axis=1)
            field = np.asarray((proj@v[:, 0].T).T)
            assert np.allclose(field@counts/counts.sum(), rates[:, 0], atol=1e-5)
            blocks.append(dict(rate=rates, drift=drift, field=field, R=a['global_R_Hz'], G=30*a['global_s']))
    model = {k: np.concatenate([a[k] for a in blocks]) for k in blocks[0]};model['time'] = np.arange(1, 10001)/1000
    windows = []
    for lo, hi in [(0, 5), (5, 10)]:
        s = slice(lo*1000, hi*1000);ns = slice(lo*200, hi*200);ds = slice(lo*50, hi*50)
        delta = model['field'][s].mean(0)-native['field'][ns].mean(0)
        windows.append(dict(interval_s=[lo, hi], model_rate=model['rate'][s].mean(0).tolist(), native_rate=native['rate'][ns].mean(0).tolist(),
            rate_difference=(model['rate'][s].mean(0)-native['rate'][ns].mean(0)).tolist(),
            weighted_field_RMS_Hz=float(np.sqrt(np.average(delta**2, weights=counts))),
            model_Zdot=model['drift'][s].mean(0).tolist(), native_Zdot=native['drift'][ds].mean(0).tolist(),
            Zdot_difference=(model['drift'][s].mean(0)-native['drift'][ds].mean(0)).tolist()))
    tail = windows[-1];tn = first_low(native['time'], native['R']);tm = first_low(model['time'], model['R'])
    guards = dict(tail_spatial=tail['weighted_field_RMS_Hz'] <= 10,
        tail_core_rate=max(abs(x) for x in tail['rate_difference'][1:3]) <= 10,
        tail_core_drift=max(abs(x) for x in tail['Zdot_difference'][1:3]) <= .01,
        tail_core_drift_sign=bool(np.array_equal(np.sign(tail['model_Zdot'][1:3]), np.sign(tail['native_Zdot'][1:3]))),
        both_quiet=all(x < 5 for x in tail['model_rate'][:3]+tail['native_rate'][:3]),
        first_low_time=tn is not None and tm is not None and abs(tn-tm) <= .25)
    result = dict(status='COMPLETE_MATCHED_UPPER_EXIT_COMPARISON', windows=windows, guards=guards, retained=all(guards.values()),
        first100ms_causal_R_le5_s=dict(native=tn, model=tm), native_sparse_inputs_paired_exact=True,
        model_both_RNG_paired=True, core_Z_held_not_actual_recovery=True,
        interpretation='Conditional fixed-spatial-field response from the same highhistory, comparingK9.35 with9.5. A transition and positive counterfactual coreZ drift do not certify a fold or prove that the natural trajectory exits by crossing it.',
        formal_bifurcation_allowed=False, agent_visual='PENDING', human_visual='PENDING', producer_sha256=sha(__file__))
    np.savez_compressed(dest/'readouts.npz', **{'native_'+k: v for k, v in native.items()}, **{'model_'+k: v for k, v in model.items()})
    write(dest/'result.json', result);plot(native, model);shutil.copy2(__file__, dest/'producer.py')
    write(dest/'progress.json', dict(status='COMPLETE', updated_epoch=time.time()));print(result, flush=True)


def plot(native, model):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    ref = dict(np.load(ROOT/'dynamic_mean_history_pair/analysis/high_readouts.npz'))
    fig, ax = plt.subplots(2, 3, figsize=(12, 8), layout='constrained')
    for key, label, color, d in [('native', 'Native', 'black', native), ('model', 'Leading recurrent mean', '#8266ad', model)]:
        ax[0, 0].plot(d['time'], d['R'], color=color, lw=1, label=label)
        for core, ls in [(1, '-'), (2, '--')]:ax[0, 1].plot(d.get('drift_time', d['time']), d['drift'][:, core], ls=ls, color=color, lw=.9)
        low = ref[key+'_rate'];low = low[len(low)//2:, 0].mean();upper = d['rate'][len(d['rate'])//2:, 0].mean()
        ax[0, 2].plot([9.35, 9.5], [low, upper], marker='o', color=color, ls=':', label=label)
    ax[0, 0].axhline(5, color='.6', ls=':', lw=.8);ax[0, 0].set_yscale('symlog', linthresh=1)
    ax[0, 0].set(xlabel='Time from high history (s)', ylabel='Causal E rate (Hz)', title='Held K = 9.5; Z unchanged', xlim=(0, 10))
    ax[0, 0].legend(frameon=False, fontsize=8)
    ax[0, 1].axhline(0, color='.6', lw=.8);ax[0, 1].set(xlabel='Time from high history (s)', ylabel='Core dZ/dt if released (1/s)', title='Core A solid; Core B dashed', xlim=(0, 10))
    ax[0, 2].set(xlabel='Held mean K / gL', ylabel='All-E mean rate, 5–10 s (Hz)', title='Measured conditional response', xticks=[9.35, 9.5])
    ax[0, 2].text(.5, .9, 'Dotted lines guide the eye;\nno certified branch', ha='center', va='top', transform=ax[0, 2].transAxes, fontsize=9)
    centers = np.load(ROOT/'native_slices/geometry.npz')['centers_mm']
    fields = [ref['native_field'][1000:].mean(0), native['field'][1000:].mean(0), model['field'][5000:].mean(0)]
    for axis, label, f in zip(ax[1], ['Native K = 9.35', 'Native K = 9.5', 'Leading mean K = 9.5'], fields):
        im = axis.imshow(f.reshape(20, 20), origin='lower', extent=(0, 20, 0, 20), vmin=0, vmax=500, cmap='magma', interpolation='nearest')
        for xy in centers:axis.add_patch(Circle(xy, 1.5, fill=False, edgecolor='#00c3c5', lw=.9))
        axis.set(title=label, xlabel='x (mm)', ylabel='y (mm)', xticks=[0, 10, 20], yticks=[0, 10, 20])
    fig.colorbar(im, ax=ax[1].tolist(), shrink=.8, label='E rate, 5–10 s (Hz)')
    fig.suptitle('Exit-related conditional transition and the sign of core resource balance', weight='bold')
    for ext in ['png', 'svg']:fig.savefig(ROOT/f'figures/mean_boundary_correspondence.{ext}', dpi=180)
    plt.close(fig)
    p = ROOT/'figures/README.md';title = '### mean_boundary_correspondence.png / mean_boundary_correspondence.svg'
    if title not in p.read_text():
        with p.open('a') as f:f.write('\n\n'+title+'\n从相同完整高态出发，在同一真实Z场、固定外源和G/M动态下，比较K均值9.35与9.5的有限条件响应；9.5另有原生与候选模型配对。显示退出时间、核心资源收支符号及实际末窗空间场。\n**关注点**：Z仍被固定，正收支表示若释放可恢复；点间线只辅助阅读，不是已认证分支，不能把该条件变化当作自主退出机制的全部证明。\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--wait', action='store_true');main(p.parse_args().wait)
