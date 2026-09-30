#!/usr/bin/env python3
"""Fixed-window native comparison for two ten-second conditional histories."""
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
from analyze_native import original
from dynamic_mean_history_pair import OUT, CASES, base


def main(wait):
    dest = OUT/'analysis';dest.mkdir(exist_ok=True)
    while not all((OUT/c/'result.json').exists() for c in CASES):
        if any((OUT/c/'progress.json').exists() and read(OUT/c/'progress.json')['status'] == 'FAILED' for c in CASES):
            write(dest/'progress.json', dict(status='STOPPED_ON_WORKER_FAILURE'));return
        write(dest/'progress.json', dict(status='WAITING_TWO_HISTORY_TESTS', pid=os.getpid(), updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    geo = dict(np.load(ADAPTED/'geometry.npz'));E = geo['population'] == 0;sizes = geo['group_size']
    opsgeo = np.load(base.OPS/'geometry.npz')
    assert np.array_equal(geo['cell_group'], opsgeo['cell_group']) and np.array_equal(sizes, opsgeo['group_size'])
    masks = [E]+[E & (geo['group_region'] == q) for q in range(3)]
    counts = np.bincount(geo['group_cell'][E], weights=sizes[E], minlength=400)
    proj = sparse.coo_matrix((sizes[E]/counts[geo['group_cell'][E]], (geo['group_cell'][E], np.flatnonzero(E))), shape=(400, len(E))).tocsr()
    data = {};rows = []
    for case, (_, initial_step, name) in CASES.items():
        assert read(OUT/case/'implementation_qa.json')['status'] == 'PASS'
        chunks = sorted((OUT/case/'chunks').glob('*.npz'));assert len(chunks) == 100
        blocks = []
        for index, path in enumerate(chunks):
            lo, hi = map(int, path.stem.split('_'));assert (lo, hi) == (100*index, 100*(index+1))
            with np.load(path) as a:
                value = a['group_output'].astype(float)
                rates = np.stack([np.average(value[:, 0, m], weights=sizes[m], axis=1) for m in masks], axis=1)
                drift = np.stack([np.average((value[:, 8, m]-value[:, 1, m])/5, weights=sizes[m], axis=1) for m in masks], axis=1)
                field = np.asarray((proj@value[:, 0].T).T)
                assert np.allclose(field@counts/counts.sum(), rates[:, 0], atol=1e-5)
                blocks.append(dict(rate=rates, drift=drift, field=field, R=a['global_R_Hz'], G=30*a['global_s']))
        model = {k: np.concatenate([d[k] for d in blocks]) for k in blocks[0]};model['time'] = np.arange(1, 10001)/1000
        folder = base.MATCHED/'runs'/name;n = dict(np.load(base.MATCHED/'extended_analysis'/f'{name}_readouts.npz'))
        me = original.load(folder/'mechanism_chunks', ['time_ms', 'global_E_rate_Hz', 'global_raw_conductance_ratio'])
        dr = original.load(folder/'conditional_drift_chunks', ['time_ms', 'values'])
        start = initial_step/10000;mt = me['time_ms']/1000-start;dt = dr['time_ms']/1000-start
        use = (mt >= -1e-9) & (mt < 10-1e-9);dd = (dt > 1e-9) & (dt <= 10+1e-9)
        native = dict(field=n['field_rate_5ms_Hz'], rate=n['rate_5ms_Hz'], R=me['global_E_rate_Hz'][use],
            G=me['global_raw_conductance_ratio'][use], time=mt[use], drift=dr['values'][dd, :, 0], drift_time=dt[dd])
        assert native['field'].shape == (2000, 400) and native['R'].shape == (10000,) and native['drift'].shape == (500, 4)
        assert (native['R'] < 200).all() and (native['G'] < .1).all()
        windows = []
        for lo, hi in [(0, 5), (5, 10)]:
            s = slice(lo*1000, hi*1000);ns = slice(lo*200, hi*200);ds = slice(lo*50, hi*50)
            delta = model['field'][s].mean(0)-native['field'][ns].mean(0)
            windows.append(dict(interval_s=[lo, hi], model_rate_Hz=model['rate'][s].mean(0).tolist(), native_rate_Hz=native['rate'][ns].mean(0).tolist(),
                rate_difference_Hz=(model['rate'][s].mean(0)-native['rate'][ns].mean(0)).tolist(),
                weighted_field_RMS_Hz=float(np.sqrt(np.average(delta**2, weights=counts))),
                model_Zdot=model['drift'][s].mean(0).tolist(), native_Zdot=native['drift'][ds].mean(0).tolist(),
                Zdot_difference=(model['drift'][s].mean(0)-native['drift'][ds].mean(0)).tolist()))
        guards = dict(both_windows_spatial=all(w['weighted_field_RMS_Hz'] <= 10 for w in windows),
            both_windows_core_rate=all(max(abs(x) for x in w['rate_difference_Hz'][1:3]) <= 10 for w in windows),
            both_windows_core_drift=all(max(abs(x) for x in w['Zdot_difference'][1:3]) <= .01 for w in windows),
            whole_R_below200=bool((model['R'] < 200).all()), whole_Graw_below_point1=bool((model['G'] < .1).all()))
        rows.append(dict(case=case, windows=windows, guards=guards, retained=all(guards.values()),
            R_range=[float(model['R'].min()), float(model['R'].max())], G_range=[float(model['G'].min()), float(model['G'].max())]))
        data[case] = (native, model)
        np.savez_compressed(dest/f'{case}_readouts.npz', **{'native_'+k: v for k, v in native.items()}, **{'model_'+k: v for k, v in model.items()})
    with np.load(OUT/'high/final_state.npz') as h, np.load(OUT/'asymmetric/final_state.npz') as a:
        paired = {k: np.array_equal(h[k], a[k]) for k in ['rng', 'external_rng']}
    assert all(paired.values())
    result = dict(status='COMPLETE_TWO_TEN_SECOND_HISTORY_COMPARISONS', rows=rows, paired_numerical_rng=paired,
        both_histories_retained=all(r['retained'] for r in rows), formal_bifurcation_allowed=False,
        interpretation='Same graph/held ZK/input law; different complete endogenous histories. These test conditional finite-time correspondence and history retention, not stability, infinite-time attractors, noise convergence or an autonomous cycle.',
        agent_visual='PENDING', human_visual='PENDING', producer_sha256=sha(__file__))
    write(dest/'result.json', result);plot(data);shutil.copy2(__file__, dest/'producer.py')
    write(dest/'progress.json', dict(status='COMPLETE', updated_epoch=time.time()));print(result, flush=True)


def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(3, 4, figsize=(14, 10), layout='constrained')
    centers = np.load(ROOT/'native_slices/geometry.npz')['centers_mm']
    for j, (case, (native, model)) in enumerate(data.items()):
        for label, color, d in [('Native', 'black', native), ('Leading recurrent mean', '#8266ad', model)]:
            ax[0, j*2].plot(d['time'], d['R'], lw=.8, color=color, label=label)
            ax[0, j*2+1].plot(d['time'], d['G'], lw=.8, color=color)
            for core, ls in [(1, '-'), (2, '--')]:
                ax[1, j*2].plot(np.arange(1, len(d['rate'])+1)*10/len(d['rate']), d['rate'][:, core], lw=.5, color=color, ls=ls)
                ax[1, j*2+1].plot(d.get('drift_time', d['time']), d['drift'][:, core], lw=.7, color=color, ls=ls)
        ax[0, j*2].set(title=case.capitalize()+' history', ylabel='Causal E rate (Hz)');ax[0, j*2].legend(frameon=False, fontsize=8)
        ax[0, j*2+1].set(title='Feedback remains free', ylabel='Global G / gL')
        ax[1, j*2].set(title='Core A solid; Core B dashed', ylabel='Core rate (Hz)')
        ax[1, j*2+1].set(title='Z held: drift if released', ylabel='Core dZ/dt (1/s)')
        for axis in ax[:2, j*2:j*2+2].ravel():axis.set(xlim=(0, 10), xlabel='Time from native state (s)')
        for k, (label, d) in enumerate([('Native', native), ('Leading recurrent mean', model)]):
            f = d['field'][len(d['field'])//2:].mean(0)
            axis = ax[2, j*2+k];im = axis.imshow(f.reshape(20, 20), origin='lower', extent=(0, 20, 0, 20), vmin=0, vmax=500, cmap='magma', interpolation='nearest')
            for xy in centers:axis.add_patch(Circle(xy, 1.5, fill=False, edgecolor='#00c3c5', lw=.9))
            axis.set(title=label+' (5–10 s)', xlabel='x (mm)', ylabel='y (mm)', xticks=[0, 10, 20], yticks=[0, 10, 20])
    fig.colorbar(im, ax=ax[2].tolist(), shrink=.8, label='E rate (Hz)')
    fig.suptitle('Conditional dynamics at identical Z/K: longer observation and a second history', weight='bold')
    for ext in ['png', 'svg']:fig.savefig(ROOT/f'figures/dynamic_mean_history_pair.{ext}', dpi=180)
    plt.close(fig)
    p = ROOT/'figures/README.md';title = '### dynamic_mean_history_pair.png / dynamic_mean_history_pair.svg'
    if title not in p.read_text():
        with p.open('a') as f:f.write('\n\n'+title+'\n相同固定Z/K及外源规律下，从双核高态和不对称态的完整原生历史出发，各作十秒自由递归平均输入检验。比较原生全局反馈、双核率、资源收支及后五秒空间场。\n**关注点**：数值复制不是独立原生种子；有限条件对应不认证稳定分支或自主闭环，人工待审。\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--wait', action='store_true');main(p.parse_args().wait)
