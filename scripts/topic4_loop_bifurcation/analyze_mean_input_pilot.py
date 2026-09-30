#!/usr/bin/env python3
"""Retain original relevance guards for the leading recurrent mean diagnostic."""
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
from dynamic_mean_input_pilot import OUT
from dynamic_individual_source_pilot import OUT as FIRST
from dynamic_poisson_external_pilot import OUT as PREVIOUS


def main(wait):
    dest = OUT/'analysis';dest.mkdir(exist_ok=True)
    while not (OUT/'result.json').exists():
        if read(OUT/'supervisor.json')['status'] == 'FAILED':
            write(dest/'progress.json', dict(status='STOPPED_ON_PILOT_FAILURE'));return
        write(dest/'progress.json', dict(status='WAITING_ONE_LEADING_MEAN_DIAGNOSTIC', pid=os.getpid(), updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    assert read(OUT/'implementation_qa.json')['status'] == 'PASS'
    geo = dict(np.load(ADAPTED/'geometry.npz'));E = geo['population'] == 0;sizes = geo['group_size']
    masks = [E]+[E & (geo['group_region'] == q) for q in range(3)]
    counts = np.bincount(geo['group_cell'][E], weights=sizes[E], minlength=400)
    proj = sparse.coo_matrix((sizes[E]/counts[geo['group_cell'][E]],
        (geo['group_cell'][E], np.flatnonzero(E))), shape=(400, len(E))).tocsr()
    ref = dict(np.load(FIRST/'analysis/readouts.npz'));a = dict(np.load(OUT/'trajectory.npz'))
    prev = dict(np.load(PREVIOUS/'analysis/readouts.npz'));value = a['group_output'].astype(float)
    rates = np.stack([np.average(value[:, 0, m], weights=sizes[m], axis=1) for m in masks], axis=1)
    drift = np.stack([np.average((value[:, 8, m]-value[:, 1, m])/5, weights=sizes[m], axis=1) for m in masks], axis=1)
    field = np.asarray((proj@value[:, 0].T).T)
    assert np.allclose(field@counts/counts.sum(), rates[:, 0], atol=1e-5)
    windows = []
    for lo, hi in [(0, .1), (.1, .8), (.8, 1.)]:
        s = slice(round(lo*1000), round(hi*1000));n = slice(round(lo*200), round(hi*200))
        delta = field[s].mean(0)-ref['native_field'][n].mean(0)
        windows.append(dict(interval_s=[lo, hi], rate_Hz=rates[s].mean(0).tolist(), native_rate_Hz=ref['native_rate'][n].mean(0).tolist(),
            rate_difference_Hz=(rates[s].mean(0)-ref['native_rate'][n].mean(0)).tolist(),
            weighted_field_RMS_Hz=float(np.sqrt(np.average(delta**2, weights=counts))),
            counterfactual_Zdot_per_s=drift[s].mean(0).tolist(),
            causal_R_range=[float(a['global_R_Hz'][s].min()), float(a['global_R_Hz'][s].max())],
            Graw_range=[float(30*a['global_s'][s].min()), float(30*a['global_s'][s].max())]))
    tail = windows[-1];guards = dict(tail_field_RMS=tail['weighted_field_RMS_Hz'] <= 10,
        tail_each_core_rate=max(abs(v) for v in tail['rate_difference_Hz'][1:3]) <= 10,
        whole_R_below200=bool((a['global_R_Hz'] < 200).all()), whole_Graw_below_point1=bool((30*a['global_s'] < .1).all()))
    result = dict(status='COMPLETE_LEADING_MEAN_DIAGNOSTIC_COMPARISON', windows=windows, relevance_guards=guards,
        development_relevance_retained=all(guards.values()), both_RNGs_paired=read(OUT/'result.json')['both_numerical_RNGs_paired_bitwise'],
        interpretation='One1s leading recurrent mean diagnostic: physical source probabilities generate the recurrent mean with full delays and exact Poisson external law. Added recurrent residual noise deliberately omitted. Agreement would require further independent state, longer physical-time and finite-native-noise correspondence before formal analysis.',
        formal_bifurcation_allowed=False, agent_visual='PENDING', human_visual='PENDING', producer_sha256=sha(__file__))
    write(dest/'result.json', result);np.savez_compressed(dest/'readouts.npz', field=field, rates=rates, drift=drift,
        R=a['global_R_Hz'], G=30*a['global_s'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained');t = np.arange(1, 1001)/1000
    for label, color, rr, gg in [('Native', 'black', ref['native_R'], ref['native_G']),
        ('With independent recurrent residual', '#4b8d52', prev['R'], prev['G']),
        ('Leading recurrent mean', '#8266ad', a['global_R_Hz'], 30*a['global_s'])]:
        axes[0, 0].plot(t, rr, color=color, label=label, lw=1);axes[0, 1].plot(t, gg, color=color, lw=1)
    axes[0, 0].axhline(200, color='.6', lw=.8, ls=':');axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].set(xlabel='Time since native state (s)', ylabel='Causal E rate (Hz)', xlim=(0, 1))
    axes[0, 1].set(xlabel='Time since native state (s)', ylabel='Global G / gL', xlim=(0, 1))
    centers = np.load(ROOT/'native_slices/geometry.npz')['centers_mm']
    for ax, label, f in zip(axes[1], ['Native', 'Leading recurrent mean'], [ref['native_field'][160:].mean(0), field[800:].mean(0)]):
        im = ax.imshow(f.reshape(20, 20), origin='lower', extent=(0, 20, 0, 20), vmin=0, vmax=500, cmap='magma', interpolation='nearest')
        for xy in centers:ax.add_patch(Circle(xy, 1.5, fill=False, edgecolor='#00c3c5', lw=.9))
        ax.set(title=label, xlabel='x (mm)', ylabel='y (mm)', xticks=[0, 10, 20], yticks=[0, 10, 20])
    fig.colorbar(im, ax=axes[1].tolist(), shrink=.8, label='E rate, 0.8–1.0 s (Hz)')
    fig.suptitle('Free recurrent mean: separate its dynamics from the residual approximation', weight='bold')
    for ext in ['png', 'svg']:fig.savefig(ROOT/f'figures/dynamic_mean_input_pilot.{ext}', dpi=180)
    plt.close(fig);shutil.copy2(__file__, dest/'producer.py')
    title = '### dynamic_mean_input_pilot.png / dynamic_mean_input_pilot.svg';p = ROOT/'figures/README.md'
    if title not in p.read_text():
        with p.open('a') as f:f.write('\n\n'+title+'\n相同完整原生高态、原图/延迟及Poisson外源下，只省略另外添加的独立递归残余噪声，递归均值仍由自己的源放电概率自由产生。上排比较因果率和G，下排核对末200ms真实空间场。\n**关注点**：区分平均反馈动力学与残余近似误差；这是leading mean诊断，不等于证明原生残余为零或已获得稳定分支，人工待审。\n')
    write(dest/'progress.json', dict(status=result['status'], updated_epoch=time.time()));print(result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--wait', action='store_true');main(p.parse_args().wait)
