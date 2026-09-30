#!/usr/bin/env python3
"""Match completed native controls to the two bounded spectral responses."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, shutil, time
from pathlib import Path
import numpy as np
from campaign import ROOT, read, write, sha

OUT = ROOT / 'matched_spectral_histories_review'
NATIVE = ROOT / 'native_K9p35_constant_background_pair_v2'
SPECTRA = [ROOT / 'individual_source_spectral_independent_value', ROOT / 'high_history_spectral_value']
NAMES = ['asymmetric_history_constant_background', 'high_history_constant_background']


def main(wait=False):
    if wait:
        while not all((p / 'result.json').exists() for p in [NATIVE] + SPECTRA):
            for p in [NATIVE] + SPECTRA:
                if (p / 'supervisor.json').exists() and read(p / 'supervisor.json')['status'] == 'FAILED':
                    raise RuntimeError(f'Upstream failed: {p}')
            time.sleep(10)
    OUT.mkdir(exist_ok=True)
    native = read(NATIVE / 'result.json')
    assert native['paired_recorded_inputs_exact'] and native['final_external_RNG_and_xi_exact']
    assert all(read(NATIVE / f'input_override_audit{i}.json')['steps'] == 100000 for i in range(2))
    raw = dict(np.load(SPECTRA[0] / 'parameters.npz'))
    highraw = dict(np.load(SPECTRA[1] / 'parameters.npz'))
    for k in ['Z', 'K', 'theta', 'nu_per_ms', 'display', 'region']:
        assert np.array_equal(raw[k], highraw[k]), k
    display = raw['display'][:32000]; counts = np.bincount(display, minlength=400)
    regions = raw['region']; E = np.arange(40000) < 32000
    rows = []; arrays = {}
    for i, (name, folder) in enumerate(zip(NAMES, SPECTRA)):
        result = read(folder / 'generation_1/complete.json')
        rate = np.load(folder / 'generation_1/source_rate_Hz.npy')
        st = np.concatenate([np.load(folder / f'generation_1/part{j}_statistics.npz')['replica_statistics'] for j in range(2)])
        sem = (st[:, :, 0] + st[:, :, 1]).std(1, ddof=1) / 2 / np.sqrt(st.shape[1])
        model = np.bincount(display, weights=rate[:32000], minlength=400) / np.maximum(counts, 1)
        field_sem = np.sqrt(np.bincount(display, weights=sem[:32000]**2, minlength=400)) / np.maximum(counts, 1)
        actual = np.load(NATIVE / f'{name}_tail_field_Hz.npy')
        assert model.shape == actual.shape == (400,)
        delta = model - actual
        row = native['rows'][i]; assert row['name'] == name
        oldrate = np.load((ROOT / 'individual_source_spectral_pilot' if i == 0 else folder) / 'generation_0/source_rate_Hz.npy')
        oldfield = np.bincount(display, weights=oldrate[:32000], minlength=400) / np.maximum(counts, 1)
        summary = dict(history=name, spectral_scope='Fresh F(X12), approximate numerical candidate; root rejected' if i == 0 else 'One F(native high source statistics); not a self-generated root',
            native_complete_10s=row, native_tail_relative_window_s=[5, 10],
            spectral_E_field_RMS_from_native_Hz=float(np.sqrt(np.average(delta**2, weights=counts))),
            native_background_change_field_RMS_Hz=float(np.sqrt(np.average((actual-oldfield)**2, weights=counts))),
            native_background_comparison='Variable-drive2s source anchor versus fixed-drive last5s of10s continuation; unequal windows, one history, descriptive difference only.',
            cell_weighted_field_MCSEM_RMS_Hz=float(np.sqrt(np.average(field_sem**2, weights=counts))),
            maximum_abs_field_difference_Hz=float(abs(delta).max()),
            native_tail_causal_R_range_Hz=row['windows'][1]['causal_R_mean_and_range'],
            spectral_G_used=result['G_used'], spectral_G_from_output=result['G_implied_by_output'],
            spectral_rows=result['rows'],
            tail_brief_events=row['brief_events'])
        rows.append(summary)
        arrays[f'native_field_{i}'] = actual; arrays[f'spectral_field_{i}'] = model
        arrays[f'field_difference_{i}'] = delta; arrays[f'field_output_MCSEM_{i}'] = field_sem
        arrays[f'variable_background_anchor_field_{i}'] = oldfield
    np.savez_compressed(OUT / 'fields.npz', **arrays)
    result = dict(status='COMPLETE_MATCHED_BACKGROUND_HISTORY_REVIEW', rows=rows,
        QA=dict(native_external_input_and_final_rng_pairing=True, same_full_Z_K_theta_and_fixed_nu=True),
        unit='Two endogenous histories of one native noise-state pair, not independent seeds; conditional fixedZ/K experiments.',
        uncertainty='Reported MCSEM is only spectral output sampling, excluding candidate-input uncertainty, omitted correlations, finite periodic window bias and native finite-time variability.',
        judgement='Matched-background spatial/state check only. Neither response is a certified fixed point; no physical stability or formal bifurcation is inferred from spectral map residuals.',
        formal_bifurcation_allowed=False, autonomous_loop=False, producer_sha256=sha(__file__))
    write(OUT / 'result.json', result); shutil.copy2(__file__, OUT / 'producer.py')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'svg.fonttype':'none',
                        'axes.spines.top':False, 'axes.spines.right':False})
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 7.4), layout='constrained')
    geo = np.load(NATIVE / 'geometry.npz')
    lim = max(.1, max(float(abs(arrays[f'field_difference_{i}']).max()) for i in range(2)))
    for i in range(2):
        for j, key in enumerate(['native_field', 'spectral_field', 'field_difference']):
            ax = axs[i, j]
            im = ax.imshow(arrays[f'{key}_{i}'].reshape(20,20), origin='lower', extent=(0,20,0,20),
                cmap='magma' if j < 2 else 'RdBu_r', vmin=0 if j < 2 else -lim,
                vmax=500 if j < 2 else lim, interpolation='nearest')
            for label, xy in zip(['A','B'], geo['centers_mm']):
                ax.add_patch(Circle(xy, float(geo['core_radius_mm']), fill=False, edgecolor='#00bfc5', lw=1))
                ax.text(xy[0], xy[1]+1.8, label, ha='center', color='#00bfc5', fontsize=9)
            title = ['Native: fixed background', 'Spectral response', 'Response minus native'][j]
            if j == 2: title += f"\nRMS = {rows[i]['spectral_E_field_RMS_from_native_Hz']:.3f} Hz"
            ax.set(title=title, xlabel='x (mm)', xticks=[0,10,20], yticks=[0,10,20])
            if j == 0: ax.set_ylabel(('Asymmetric history' if i == 0 else 'Double-core history')+'\ny (mm)')
            else: ax.tick_params(labelleft=False)
            if j == 1: field_image=im
            if j == 2: diff_image=im
    fig.colorbar(field_image, ax=axs[:, :2], label='E rate (Hz)', shrink=.83)
    fig.colorbar(diff_image, ax=axs[:, 2], label='Rate difference (Hz)', shrink=.83)
    fig.suptitle('Same held Z/K fields and fixed external mean; finite conditional responses', fontsize=12)
    figures = ROOT / 'figures'
    for ext in ['png','svg']: fig.savefig(figures / f'matched_spectral_histories.{ext}', dpi=190)
    plt.close(fig)
    import xml.etree.ElementTree as ET
    ET.parse(figures / 'matched_spectral_histories.svg')
    write(OUT / 'figure_qa.json', dict(SVG_XML='PASS', agent_visual_review='PENDING', human_visual_review='PENDING'))
    readme=figures / 'README.md'
    if '### matched_spectral_histories.png' not in readme.read_text():
        with readme.open('a') as f:
            f.write('\n### matched_spectral_histories.png / .svg\n两种原生历史在同一固定逐细胞外源期望、同一完整 Z/K 场下继续 10 秒，左列为末 5 秒空间场，中列为对应谱响应，右列为同尺度差值。上排是最终混合候选的一次独立响应，下排是原生双核源统计初始化的一次响应，二者均未认证自洽或稳定性；条件钳制不计入自主闭环。\n**关注点**：背景统一后两种空间活动及 G 分段是否仍被保留；数值复制不是原生独立种子，图待人工目视。\n')
    print(result, flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--wait', action='store_true'); a=p.parse_args()
    main(a.wait)
