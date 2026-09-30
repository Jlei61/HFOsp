#!/usr/bin/env python3
"""Compare the bounded physical-time source tests with their native anchor."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import shutil
import time
import numpy as np
from scipy import sparse
from campaign import ROOT, read, write, sha
from dynamic_individual_source_pilot import OUT, MODES, MATCHED, OPS
from analyze_native import original
from coupled_density_exit import ADAPTED

DEST = OUT/'analysis'


def main(wait):
    DEST.mkdir(exist_ok=True)
    while not (OUT/'result.json').exists():
        if (OUT/'supervisor.json').exists() and read(OUT/'supervisor.json')['status'] == 'FAILED':
            write(DEST/'progress.json', dict(status='STOPPED_ON_PILOT_FAILURE'));return
        write(DEST/'progress.json', dict(status='WAITING_FIXED_PHYSICAL_TIME_PAIR', pid=os.getpid(), updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    assert read(OUT/'result.json')['status'] == 'COMPLETE_TWO_PHYSICAL_TIME_PILOTS_ANALYSIS_PENDING'
    geo = dict(np.load(ADAPTED/'geometry.npz'))
    original_geo = np.load(OPS/'geometry.npz')
    assert np.array_equal(geo['cell_group'], original_geo['cell_group'])
    assert np.array_equal(geo['group_size'], original_geo['group_size'])
    # Operator source grouping uses a 40x40 spatial index; the unchanged
    # native figure observer uses the already audited 20x20 display mapping.
    sizes = geo['group_size'];E = geo['population'] == 0
    masks = [E]+[E & (geo['group_region'] == q) for q in range(3)]
    counts = np.bincount(geo['group_cell'][E], weights=sizes[E], minlength=400)
    proj = sparse.coo_matrix((sizes[E]/counts[geo['group_cell'][E]],
        (geo['group_cell'][E], np.flatnonzero(E))), shape=(400, len(E))).tocsr()
    name = 'high_history_constant_background';folder = MATCHED/'runs'/name
    n = dict(np.load(MATCHED/'extended_analysis'/f'{name}_readouts.npz'))
    field_n = n['field_rate_5ms_Hz'][:200];rates_n = n['rate_5ms_Hz'][:200]
    me = original.load(folder/'mechanism_chunks', ['time_ms', 'global_E_rate_Hz', 'global_raw_conductance_ratio'])
    dr = original.load(folder/'conditional_drift_chunks', ['time_ms', 'values'])
    mt = me['time_ms']/1000-72.;md = (mt >= 0) & (mt < 1)
    dt = dr['time_ms']/1000-72.;dd = (dt > 0) & (dt <= 1)
    native = dict(R=me['global_E_rate_Hz'][md], G=me['global_raw_conductance_ratio'][md], time=mt[md],
        field=field_n, rate=rates_n, drift=dr['values'][dd, :, 0], drift_time=dt[dd])
    assert native['R'].shape == (1000,) and native['drift'].shape == (50, 4)
    data = {};rows = []
    for mode in MODES:
        assert read(OUT/mode/'implementation_qa.json')['status'] == 'PASS'
        a = dict(np.load(OUT/mode/'trajectory.npz'));value = a['group_output'].astype(float)
        rates = np.stack([np.average(value[:, 0, m], weights=sizes[m], axis=1) for m in masks], axis=1)
        drift = np.stack([np.average((value[:, 8, m]-value[:, 1, m])/5, weights=sizes[m], axis=1) for m in masks], axis=1)
        field = np.asarray((proj@value[:, 0].T).T)
        assert np.allclose(field@counts/counts.sum(), rates[:, 0], atol=1e-5)
        assert np.allclose(a['cell_rate_Hz'][:, :32000].mean(1), rates[:, 0], atol=2e-4)
        d = dict(R=a['global_R_Hz'], G=30*a['global_s'], time=a['elapsed_time_ms']/1000,
            field=field, rate=rates, drift=drift);data[mode] = d;windows = []
        for lo, hi in [(0, .1), (.1, .8), (.8, 1.)]:
            s = slice(round(lo*1000), round(hi*1000));ns = slice(round(lo*200), round(hi*200))
            nd = (native['drift_time'] > lo+1e-9) & (native['drift_time'] <= hi+1e-9)
            delta = field[s].mean(0)-field_n[ns].mean(0)
            windows.append(dict(interval_s=[lo, hi], native_rate_Hz=rates_n[ns].mean(0).tolist(),
                density_rate_Hz=rates[s].mean(0).tolist(),
                rate_difference_Hz=(rates[s].mean(0)-rates_n[ns].mean(0)).tolist(),
                weighted_field_RMS_Hz=float(np.sqrt(np.average(delta**2, weights=counts))),
                native_counterfactual_Zdot_per_s=native['drift'][nd].mean(0).tolist(),
                density_counterfactual_Zdot_per_s=drift[s].mean(0).tolist(),
                density_causal_R_range=[float(d['R'][s].min()), float(d['R'][s].max())],
                density_Graw_range=[float(d['G'][s].min()), float(d['G'][s].max())]))
        tail = windows[-1]
        guards = dict(tail_field_RMS=tail['weighted_field_RMS_Hz'] <= 10,
            tail_each_core_rate=max(abs(v) for v in tail['rate_difference_Hz'][1:3]) <= 10,
            whole_R_below200=bool((d['R'] < 200).all()), whole_Graw_below_point1=bool((d['G'] < .1).all()))
        rows.append(dict(mode=mode, windows=windows, relevance_guards=guards,
            development_relevance_retained=all(guards.values()), formal_correspondence_certified=False))
    result = dict(status='COMPLETE_TWO_PHYSICAL_TIME_COMPARISONS', rows=rows,
        interpretation='One-second physical dynamics at one native conditional high-state anchor. Replicas are numerical particles, not native seeds. Independent Gaussian source fluctuations remain approximate; retaining source identity does not alone certify covariance, autonomous loop, periodic solution or stability.',
        formal_bifurcation_allowed=False, producer_sha256=sha(__file__), agent_visual='PENDING', human_visual='PENDING')
    write(DEST/'result.json', result)
    np.savez_compressed(DEST/'readouts.npz', native_field=native['field'], native_rate=native['rate'],
        native_R=native['R'], native_G=native['G'], native_drift=native['drift'],
        **{mode+'_'+key: a for mode, d in data.items() for key, a in d.items()})
    plot(native, data, counts, geo)
    shutil.copy2(__file__, DEST/'producer.py');write(DEST/'progress.json', dict(status=result['status'], updated_epoch=time.time()))
    print(rows, flush=True)


def plot(native, data, counts, geo):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.2), layout='constrained')
    colors = ['black', '#bc7336', '#327ba4'];labels = ['Native', 'Grouped sources', 'Individual sources']
    for color, label, d in zip(colors, labels, [native]+[data[m] for m in MODES]):
        ax[0, 0].plot(d['time'], d['R'], color=color, label=label, lw=1)
        ax[0, 1].plot(d['time'], d['G'], color=color, lw=1)
        t = d.get('drift_time', d['time'])
        for j, ls in [(1, '-'), (2, '--')]:
            ax[0, 2].plot(t, d['drift'][:, j], ls=ls, color=color, lw=1)
    ax[0, 0].axhline(200, color='.6', ls=':', lw=.8)
    ax[0, 0].legend(frameon=False, fontsize=8)
    ax[0, 2].axhline(0, color='.6', ls=':', lw=.8)
    for a, label in zip(ax[0], ['Causal E rate (Hz)', 'Global G / gL', 'Core dZ/dt if released (1/s)']):
        a.set(xlim=(0, 1), xlabel='Time since native state (s)', ylabel=label)
    ax[0, 2].set_title('Core A solid; Core B dashed', fontsize=10)
    centers = np.load(ROOT/'native_slices/geometry.npz')['centers_mm']
    for a, label, d in zip(ax[1], labels, [native]+[data[m] for m in MODES]):
        field = d['field'][round(.8*len(d['field'])):].mean(0)
        im = a.imshow(field.reshape(20, 20), origin='lower', extent=(0, 20, 0, 20),
            vmin=0, vmax=500, cmap='magma', interpolation='nearest')
        for xy in centers:a.add_patch(Circle(xy, 1.5, fill=False, edgecolor='#00c3c5', lw=.9))
        a.set(title=label+' (0.8–1.0 s)', xlabel='x (mm)', ylabel='y (mm)', xticks=[0, 10, 20], yticks=[0, 10, 20])
    fig.colorbar(im, ax=ax[1].tolist(), shrink=.8, label='E rate (Hz)')
    fig.suptitle('Free physical dynamics: source identity with full native delays', weight='bold')
    for ext in ['png', 'svg']:fig.savefig(ROOT/f'figures/dynamic_individual_source_pilot.{ext}', dpi=180)
    plt.close(fig)
    title = '### dynamic_individual_source_pilot.png / dynamic_individual_source_pilot.svg'
    p = ROOT/'figures/README.md'
    if title not in p.read_text():
        with p.open('a') as f:f.write('\n\n'+title+'\n从同一原生高态完整初态出发，比较源群平均与保留逐细胞源身份的物理时间动力学，沿用原图延迟、逐细胞阈值及相同固定外源。Z/K固定，膜、突触、不应期、M/R/G自由演化；没有规定振荡周期。\n**关注点**：检查双核、空间招募、全局反馈及核资源漂移是否对应原生；一秒开发工作点检验不等于分岔或稳定性认证，独立Gaussian源近似仍待验证。\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--wait', action='store_true');main(parser.parse_args().wait)
