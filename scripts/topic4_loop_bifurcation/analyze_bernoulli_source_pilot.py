#!/usr/bin/env python3
"""Paired readout of the one recurrent-variance correction."""
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
from dynamic_bernoulli_source_pilot import OUT
from dynamic_individual_source_pilot import OUT as BASE


def main(wait):
    dest = OUT/'analysis';dest.mkdir(exist_ok=True)
    while not (OUT/'result.json').exists():
        if read(OUT/'supervisor.json')['status'] == 'FAILED':
            write(dest/'progress.json', dict(status='STOPPED_ON_PILOT_FAILURE'));return
        write(dest/'progress.json', dict(status='WAITING_ONE_VARIANCE_ASSAY', pid=os.getpid(), updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    assert read(OUT/'implementation_qa.json')['status'] == 'PASS'
    geo = dict(np.load(ADAPTED/'geometry.npz'));E = geo['population'] == 0;sizes = geo['group_size']
    masks = [E]+[E & (geo['group_region'] == q) for q in range(3)]
    counts = np.bincount(geo['group_cell'][E], weights=sizes[E], minlength=400)
    projection = sparse.coo_matrix((sizes[E]/counts[geo['group_cell'][E]],
        (geo['group_cell'][E], np.flatnonzero(E))), shape=(400, len(E))).tocsr()
    ref = dict(np.load(BASE/'analysis/readouts.npz'));a = dict(np.load(OUT/'trajectory.npz'))
    value = a['group_output'].astype(float)
    rates = np.stack([np.average(value[:, 0, m], weights=sizes[m], axis=1) for m in masks], axis=1)
    drift = np.stack([np.average((value[:, 8, m]-value[:, 1, m])/5, weights=sizes[m], axis=1) for m in masks], axis=1)
    field = np.asarray((projection@value[:, 0].T).T)
    assert np.allclose(field@counts/counts.sum(), rates[:, 0], atol=1e-5)
    windows = []
    for lo, hi in [(0, .1), (.1, .8), (.8, 1.)]:
        s = slice(round(lo*1000), round(hi*1000));n = slice(round(lo*200), round(hi*200))
        delta = field[s].mean(0)-ref['native_field'][n].mean(0)
        windows.append(dict(interval_s=[lo, hi], rate_Hz=rates[s].mean(0).tolist(),
            native_rate_Hz=ref['native_rate'][n].mean(0).tolist(),
            rate_difference_Hz=(rates[s].mean(0)-ref['native_rate'][n].mean(0)).tolist(),
            weighted_field_RMS_Hz=float(np.sqrt(np.average(delta**2, weights=counts))),
            counterfactual_Zdot_per_s=drift[s].mean(0).tolist(),
            causal_R_range=[float(a['global_R_Hz'][s].min()), float(a['global_R_Hz'][s].max())],
            Graw_range=[float(30*a['global_s'][s].min()), float(30*a['global_s'][s].max())]))
    tail = windows[-1];guards = dict(tail_field_RMS=tail['weighted_field_RMS_Hz'] <= 10,
        tail_each_core_rate=max(abs(v) for v in tail['rate_difference_Hz'][1:3]) <= 10,
        whole_R_below200=bool((a['global_R_Hz'] < 200).all()), whole_Graw_below_point1=bool((30*a['global_s'] < .1).all()))
    with np.load(OUT/'final_state.npz') as end, np.load(BASE/'individual_source/final_state.npz') as old:
        assert np.array_equal(end['rng'], old['rng']), 'The two variants consumed different numerical random streams'
        assert np.array_equal(end['state'][:, :, 6:8], old['state'][:, :, 6:8])
        p = end['source_history']*.1;assert p.min() >= 0 and p.max() <= 1
        moment = dict(E_mean_probability=float(p[:, :32000].mean()), E_mean_squared_probability=float((p[:, :32000]**2).mean()))
    result = dict(status='COMPLETE_BERNOULLI_VARIANCE_COMPARISON', windows=windows, relevance_guards=guards,
        development_relevance_retained=all(guards.values()), final_RNG_state_paired_bitwise=True, held_Z_K_paired_bitwise=True,
        last_delay_window_empirical_probability=moment,
        interpretation='Paired1s conditional physical dynamics; only recurrent independent source variance changed from p to p(1-p). Numerical replica probability and Gaussian residuals remain approximations. No source covariance, physical stability, periodic branch or autonomous loop certification.',
        formal_bifurcation_allowed=False, producer_sha256=sha(__file__), agent_visual='PENDING', human_visual='PENDING')
    write(dest/'result.json', result)
    np.savez_compressed(dest/'readouts.npz', field=field, rates=rates, drift=drift, R=a['global_R_Hz'], G=30*a['global_s'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(12, 7.5), layout='constrained');grid = fig.add_gridspec(2, 6)
    ar = fig.add_subplot(grid[0, :3]);ag = fig.add_subplot(grid[0, 3:])
    t = np.arange(1, 1001)/1000
    labels = ['Native', 'Poisson recurrent variance', 'Bernoulli recurrent variance'];colors = ['black', '#ba7738', '#287faa']
    for label, color, r, g in zip(labels, colors,
        [ref['native_R'], ref['individual_source_R'], a['global_R_Hz']],
        [ref['native_G'], ref['individual_source_G'], 30*a['global_s']]):
        ar.plot(t, r, c=color, label=label, lw=1);ag.plot(t, g, c=color, lw=1)
    ar.axhline(200, color='.6', ls=':', lw=.8);ar.legend(frameon=False, fontsize=8)
    ar.set(xlabel='Time since native state (s)', ylabel='Causal E rate (Hz)', xlim=(0, 1))
    ag.set(xlabel='Time since native state (s)', ylabel='Global G / gL', xlim=(0, 1))
    axes = [];centers = np.load(ROOT/'native_slices/geometry.npz')['centers_mm']
    for j, (label, f) in enumerate(zip(labels, [ref['native_field'][160:].mean(0), ref['individual_source_field'][800:].mean(0), field[800:].mean(0)])):
        ax = fig.add_subplot(grid[1, 2*j:2*j+2]);axes.append(ax)
        im = ax.imshow(f.reshape(20, 20), origin='lower', extent=(0, 20, 0, 20), vmin=0, vmax=500, cmap='magma', interpolation='nearest')
        for xy in centers:ax.add_patch(Circle(xy, 1.5, fill=False, edgecolor='#00c3c5', lw=.9))
        ax.set(title=label, xlabel='x (mm)', ylabel='y (mm)', xticks=[0, 10, 20], yticks=[0, 10, 20])
    fig.colorbar(im, ax=axes, shrink=.8, label='E rate, 0.8–1.0 s (Hz)')
    fig.suptitle('Physical source probabilities: test one recurrent noise assumption', weight='bold')
    for ext in ['png', 'svg']:fig.savefig(ROOT/f'figures/dynamic_bernoulli_source_pilot.{ext}', dpi=180)
    plt.close(fig);shutil.copy2(__file__, dest/'producer.py')
    title = '### dynamic_bernoulli_source_pilot.png / dynamic_bernoulli_source_pilot.svg';path = ROOT/'figures/README.md'
    if title not in path.read_text():
        with path.open('a') as f:f.write('\n\n'+title+'\n保留逐细胞源及完整时延，从同一完整高态初始状态比较递归输入方差p与p(1-p)，其余原生参数、外源及数值随机流相同。上排检查全局率和G，下排显示相同0–500Hz色标的末200ms空间均率。\n**关注点**：判断统计近似能否解释错误的核外招募及反馈开启；没有引入新的生物机制，仍非分岔认证，人工待审。\n')
    write(dest/'progress.json', dict(status=result['status'], updated_epoch=time.time()));print(result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--wait', action='store_true');main(p.parse_args().wait)
