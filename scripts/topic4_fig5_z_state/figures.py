#!/usr/bin/env python3
"""Candidate figures for the milestone (independent PNG/PDF each; agent-checked, human review pending).

fig_native_state_map     : frozen-Z state map of the native SNN (Z source time x history x future) with tail metrics
fig_native_traces        : all-E 10-ms rate traces of the 24 continuations (one row per Z field; W1/W2, two histories)
fig_z_path               : real Z(t) path (all-E / core A / core B / surround means) of both source trajectories with marks 1-4,
                           and the six source Z fields
fig_native_vs_approx     : per-pair native vs approximation tail metrics and categories (W1 diagnostic / W2 frozen score)
fig_path_replay          : real-Z(t) path replay: native vs approximation all-E rate and spatial extent, marks 3 and high onset
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from common import *  # noqa: F401,F403
import readouts as R
import native_continue as nc

FIG = OUT / 'figures'
CAT_ORDER = ['SELF_LIMITED', 'QUIESCENT', 'PERSISTENT', 'UNRESOLVED']
CAT_COLOR = {'SELF_LIMITED': '#2b7bba', 'QUIESCENT': '#9aa5b1', 'PERSISTENT': '#d1495b', 'UNRESOLVED': '#f2b134'}
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12, 'legend.fontsize': 10, 'pdf.fonttype': 42})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=160, bbox_inches='tight')
    plt.close(fig)
    return str(FIG / f'{name}.png')


def load_rows():
    d = read(OUT / 'native_state_map.json'); return d['rows'], d.get('selection')


def fig_native_state_map():
    rows, sel = load_rows()
    main = [r for r in rows if r['kind'] == 'continuation' and not r['freeze_m']]
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.4), gridspec_kw=dict(width_ratios=[1.3, 1, 1, 1, 1]), constrained_layout=True)
    # (a) category grid: rows = (history, future), cols = Z source time
    ax = axes[0]; combos = [(h, w) for h in HISTORY_MS for w in ('W1', 'W2')]
    cmap = ListedColormap([CAT_COLOR[c] for c in CAT_ORDER]); norm = BoundaryNorm(np.arange(-.5, len(CAT_ORDER)), cmap.N)
    grid = np.full((len(combos), len(Z_TIMES_MS)), np.nan)
    for r in main:
        i = combos.index((r['history_ms'], r['future'])); j = Z_TIMES_MS.index(r['z_source_ms']); grid[i, j] = CAT_ORDER.index(r['category'])
    ax.imshow(grid, cmap=cmap, norm=norm, aspect='auto')
    for r in main:
        i = combos.index((r['history_ms'], r['future'])); j = Z_TIMES_MS.index(r['z_source_ms'])
        ax.text(j, i, 'H' if r['high_rate'] == 'REACHED' else '', ha='center', va='center', fontsize=10, color='k')
    ax.set_xticks(range(len(Z_TIMES_MS))); ax.set_xticklabels([f'{z / 1000:.2f}' for z in Z_TIMES_MS]); ax.set_xlabel('Z field source time (s)')
    ax.set_yticks(range(len(combos))); ax.set_yticklabels([f'history {h / 1000:.2f} s, {w}' for h, w in combos])
    ax.set_title('finite-time category of the last 4 s')
    handles = [plt.Rectangle((0, 0), 1, 1, color=CAT_COLOR[c]) for c in CAT_ORDER]
    ax.legend(handles, [c.replace('_', ' ').lower() for c in CAT_ORDER] + [], loc='upper center', bbox_to_anchor=(.5, -.28), ncol=2, frameon=False)
    # (b)-(d): tail quiet fraction, persistent fraction, all-E rate vs Z source time
    for ax, key, lab in ((axes[1], 'tail_quiet_fraction', 'quiet fraction (last 4 s)'), (axes[2], 'tail_persistent_50Hz_80', 'persistent occupation\n(>50 Hz for >=80% of window)'),
                         (axes[3], 'tail_all_E_hz', 'all-E rate (Hz, last 4 s)'), (axes[4], 'tail_coreA_hz', 'core A rate (Hz, 1.75 mm readout, last 4 s)')):
        for h, ls in zip(HISTORY_MS, ('-', '--')):
            for w, mk in zip(('W1', 'W2'), ('o', 's')):
                rs = sorted([r for r in main if r['history_ms'] == h and r['future'] == w], key=lambda r: r['z_source_ms'])
                ax.plot([r['z_source_ms'] / 1000 for r in rs], [r[key] for r in rs], ls, marker=mk, color='#333' if h == HISTORY_MS[0] else '#a55',
                        label=f'history {h / 1000:.2f} s, {w}')
        ax.set_xlabel('Z field source time (s)'); ax.set_ylabel(lab); ax.set_ylim(bottom=0)
        for t in (HIGH_ONSET_S, FIGURE_TIMES_S[3]):
            ax.axvline(t, color='#bbb', lw=.8, ls=':')
    axes[4].legend(loc='upper left', frameon=False, fontsize=8)
    if sel and sel.get('pair'):
        a, b = sel['pair']
        for ax in axes[1:]:
            ax.axvspan(a / 1000, b / 1000, color='#ffe9a8', alpha=.5, lw=0)
    return save(fig, 'fig_native_state_map')


def fig_native_traces():
    rows, sel = load_rows()
    names = [j['name'] for j in nc.main_jobs()]
    fig, axes = plt.subplots(len(Z_TIMES_MS), 4, figsize=(18, 2.2 * len(Z_TIMES_MS)), sharex=True, sharey=True)
    for i, z in enumerate(Z_TIMES_MS):
        for j, (h, w) in enumerate([(h, w) for h in HISTORY_MS for w in ('W1', 'W2')]):
            name = f'z{z}_h{h}_{w}'; ax = axes[i, j]
            p = nc.NATIVE / 'runs' / name / 'readout_arrays.npz'
            if not p.exists():
                ax.text(.5, .5, 'not run', transform=ax.transAxes, ha='center'); continue
            a = np.load(p); r10 = a['r10']; t = a['t0_s'] + np.arange(len(r10)) * .01
            res = read(nc.NATIVE / 'runs' / name / 'readout.json')
            ax.fill_between(t, 0, r10, color=CAT_COLOR[res['category']], lw=0, alpha=.9)
            ax.axvspan(res['tail_window_s'][0], res['tail_window_s'][1], color='#eee', zorder=0)
            if res['high_rate'] is not None:
                ax.axvline(res['high_rate']['onset_s'], color='k', lw=.8, ls='--')
            if i == 0:
                ax.set_title(f'history {h / 1000:.2f} s, {w}')
            if j == 0:
                ax.set_ylabel(f'Z field {z / 1000:.2f} s\nall-E rate (Hz)')
            ax.text(.02, .85, res['category'].replace('_', ' ').lower(), transform=ax.transAxes, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel('time on common future clock (s)')
    axes[0, 0].set_xlim(ANCHOR_MS / 1000, ANCHOR_MS / 1000 + CONTINUATION_MS / 1000)
    return save(fig, 'fig_native_traces')


def fig_native_core_traces():
    """Core A / core B (1.75-mm readout) 10-ms rates of every continuation; shows local episodes that keep a run UNRESOLVED."""
    rows, sel = load_rows()
    fig, axes = plt.subplots(len(Z_TIMES_MS), 4, figsize=(18, 2.2 * len(Z_TIMES_MS)), sharex=True, sharey=True)
    for i, z in enumerate(Z_TIMES_MS):
        for j, (h, w) in enumerate([(h, w) for h in HISTORY_MS for w in ('W1', 'W2')]):
            name = f'z{z}_h{h}_{w}'; ax = axes[i, j]
            p = nc.NATIVE / 'runs' / name / 'readout_arrays.npz'
            if not p.exists():
                ax.text(.5, .5, 'not run', transform=ax.transAxes, ha='center'); continue
            a = np.load(p); reg = a['region10']; t = a['t0_s'] + np.arange(len(reg)) * .01
            res = read(nc.NATIVE / 'runs' / name / 'readout.json')
            ax.plot(t, reg[:, 0], color='#e07b39', lw=.6, label='core A'); ax.plot(t, reg[:, 1], color='#2a9d8f', lw=.6, label='core B')
            ax.axvspan(res['tail_window_s'][0], res['tail_window_s'][1], color='#eee', zorder=0)
            if i == 0:
                ax.set_title(f'history {h / 1000:.2f} s, {w}')
            if j == 0:
                ax.set_ylabel(f'Z field {z / 1000:.2f} s\ncore rate (Hz)')
            ax.text(.02, .85, res['category'].replace('_', ' ').lower(), transform=ax.transAxes, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel('time on common future clock (s)')
    axes[0, 0].set_xlim(ANCHOR_MS / 1000, ANCHOR_MS / 1000 + CONTINUATION_MS / 1000); axes[0, 0].legend(frameon=False, fontsize=8, loc='upper right')
    return save(fig, 'fig_native_core_traces')


def fig_native_extension_traces():
    """The eight extended continuations (20 s): all-E rate (grey) and core A/B rates, with the 10-s and 20-s categories."""
    names = [f'z{z}_h{h}_{w}' for z in (8000, 9000) for h in HISTORY_MS for w in ('W1', 'W2')]
    fig, axes = plt.subplots(len(names), 1, figsize=(16, 1.9 * len(names)), sharex=True, sharey=True)
    for ax, n in zip(axes, names):
        p = nc.NATIVE / 'runs' / (n + '_ext') / 'readout_arrays.npz'
        if not p.exists():
            ax.text(.5, .5, 'not run', transform=ax.transAxes, ha='center'); continue
        a = np.load(p); r10 = a['r10']; reg = a['region10']; t = a['t0_s'] + np.arange(len(r10)) * .01
        r1 = read(nc.NATIVE / 'runs' / n / 'readout.json'); r2 = read(nc.NATIVE / 'runs' / (n + '_ext') / 'readout.json')
        ax.fill_between(t, 0, r10, color='#bbb', lw=0); ax.plot(t, reg[:, 0], color='#e07b39', lw=.5); ax.plot(t, reg[:, 1], color='#2a9d8f', lw=.5)
        ax.axvspan(r1['tail_window_s'][0], r1['tail_window_s'][1], color='#ffe9a8', alpha=.5, lw=0); ax.axvspan(r2['tail_window_s'][0], r2['tail_window_s'][1], color='#d9ecff', alpha=.6, lw=0)
        ax.axvline(ANCHOR_MS / 1000 + CONTINUATION_MS / 1000, color='k', lw=.8, ls=':')
        ax.set_ylabel(n.replace('_', ' ').replace('z', 'Z ').replace('h', 'h '), fontsize=8)
        ax.text(.01, .8, f"10 s: {r1['category'].replace('_', ' ').lower()}   20 s: {r2['category'].replace('_', ' ').lower()}", transform=ax.transAxes, fontsize=8)
    axes[-1].set_xlabel('time on common future clock (s)'); axes[0].set_xlim(ANCHOR_MS / 1000, ANCHOR_MS / 1000 + (CONTINUATION_MS + EXTENSION_MS) / 1000)
    axes[0].set_title('grey: all-E 10-ms rate; orange/teal: core A / core B (1.75 mm); yellow / blue bands: 4-s windows scored at 10 s / 20 s', fontsize=9)
    return save(fig, 'fig_native_extension_traces')


def fig_z_path():
    geo = np.load(OUT / 'approx/coarse_20/geometry.npz'); g175 = geo['g175']; g15 = geo['g15']; pos = geo['positions_e']
    fig = plt.figure(figsize=(16, 8)); gs = fig.add_gridspec(2, 6, height_ratios=[1.1, 1])
    ax = fig.add_subplot(gs[0, :])
    for seed, ls in zip(SEEDS, ('-', '--')):
        run = OUT / 'replay' / 'runs' / f'eta0.0005_s{seed}'; ts = []; ms = []
        for path in sorted((run / 'fields').glob('*.npz')):
            with np.load(path) as a:
                ts.append(a['zm_step'] * DT_MS / 1000.); z = a['z']
                ms.append(np.c_[z.mean(1), z[:, g15 == 0].mean(1), z[:, g15 == 1].mean(1), z[:, g15 == 2].mean(1)])
        ts = np.concatenate(ts); ms = np.concatenate(ms)
        for k, (lab, col) in enumerate((('all E', 'k'), ('core A (1.5 mm)', '#e07b39'), ('core B (1.5 mm)', '#2a9d8f'), ('surround', '#888'))):
            ax.plot(ts, ms[:, k], ls, color=col, lw=1.4, label=f'{lab}, seed {seed}' if seed == SEEDS[0] else None)
    for n, t in FIGURE_TIMES_S.items():
        ax.axvline(t, color='#999', lw=.8, ls=':'); ax.text(t, ax.get_ylim()[1] if False else .97, f'{n}', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=11, weight='bold')
    for z in Z_TIMES_MS:
        ax.axvline(z / 1000, color='#c33', lw=.6, alpha=.6)
    ax.axvline(HIGH_ONSET_S, color='k', lw=1, ls='--')
    ax.set_xlim(0, REPLAY_END_MS / 1000); ax.set_xlabel('time (s)'); ax.set_ylabel('mean Z'); ax.legend(loc='lower left', frameon=False, ncol=4)
    ax.set_title('real Z(t) of the two source trajectories (solid: main seed; dashed: second seed); red lines: frozen-Z source times; dashed black: high-rate onset')
    z_all = []
    for z in Z_TIMES_MS:
        st = ckpt.load(OUT / 'replay' / 'runs' / f'eta0.0005_s{MAIN_SEED}' / 'checkpoints' / f't{z}ms.npz'); z_all.append(st['slow']['z'][:NE])
    vmin, vmax = min(z.min() for z in z_all), max(z.max() for z in z_all)
    for j, (z, zf) in enumerate(zip(Z_TIMES_MS, z_all)):
        ax = fig.add_subplot(gs[1, j]); sc = ax.scatter(pos[:, 0], pos[:, 1], c=zf, s=1.2, cmap='viridis', vmin=vmin, vmax=vmax, rasterized=True)
        centers = geo['centers_mm']
        for c in centers:
            ax.add_patch(plt.Circle(c, 1.5, fill=False, color='#e07b39', lw=1.2))
        ax.set_aspect('equal'); ax.set_xlim(0, 20); ax.set_ylim(0, 20); ax.set_xticks([0, 10, 20]); ax.set_yticks([0, 10, 20] if j == 0 else [])
        ax.set_title(f'Z field {z / 1000:.2f} s (mean {zf.mean():.3f})', fontsize=10)
        if j == 0:
            ax.set_ylabel('y (mm)')
        ax.set_xlabel('x (mm)')
    cb = fig.colorbar(sc, ax=fig.axes[1:], fraction=.015, pad=.01); cb.set_label('Z (per E neuron)')
    return save(fig, 'fig_z_path')


def fig_native_vs_approx(version='v1', futures=('W1',)):
    p = OUT / 'approx' / version / f"validation_{'_'.join(futures)}.json"
    if not p.exists():
        return None
    v = read(p); rows = [r for r in v['rows'] if r.get('status') == 'SCORED']
    if not rows:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), constrained_layout=True)
    x = np.arange(len(rows)); labels = [r['name'].replace('_W1', '').replace('_W2', '').replace('_h', ' h').replace('z', 'Z ') for r in rows]
    for ax, key, lab in ((axes[0], 'quiet_fraction', 'quiet fraction (last 4 s)'), (axes[1], 'persistent_fraction', 'persistent occupation (>50 Hz, 80%)')):
        ax.plot(x, [r[key]['native'] for r in rows], 'o', color='k', label='native SNN'); ax.plot(x, [r[key]['approx'] for r in rows], 's', color='#c33', label='reduced model')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=60, ha='right'); ax.set_ylabel(lab); ax.set_ylim(-.02, 1.02)
    ax = axes[2]
    for key, mk, lab in (('all_E_hz', 'o', 'all E'), ('readout175_E_coreA_hz', '^', 'core A'), ('readout175_E_coreB_hz', 'v', 'core B')):
        ax.plot([r['rates'][key]['native'] for r in rows], [r['rates'][key]['approx'] for r in rows], mk, label=lab, alpha=.8)
    lim = max(1, max(max(r['rates'][k]['native'], r['rates'][k]['approx']) for r in rows for k in r['rates'])) * 1.05
    ax.plot([0, lim], [0, lim], '-', color='#bbb', lw=.8); ax.fill_between([20, lim], [20 * .75, lim * .75], [20 * 1.25, lim * 1.25], color='#ddd', alpha=.4, lw=0)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_xlabel('native rate (Hz, last 4 s)'); ax.set_ylabel('reduced-model rate (Hz)'); ax.legend(frameon=False)
    ax = axes[3]
    for i, r in enumerate(rows):
        ax.bar(i - .2, CAT_ORDER.index(r['native_category']) + 1, width=.4, color=CAT_COLOR[r['native_category']], edgecolor='k', lw=.5)
        ax.bar(i + .2, CAT_ORDER.index(r['approx_category']) + 1, width=.4, color=CAT_COLOR[r['approx_category']], edgecolor='#c33', lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=60, ha='right'); ax.set_yticks(range(1, 5)); ax.set_yticklabels([c.replace('_', ' ').lower() for c in CAT_ORDER], fontsize=9); ax.set_title('category (black edge: native; red edge: model)', fontsize=10)
    axes[0].legend(frameon=False)
    fig.suptitle(f"native vs reduced model ({version}, {'/'.join(futures)}); state gate {v['state_gate']['status']}, magnitude gate {v['magnitude_gate']['status']}", fontsize=11)
    return save(fig, f"fig_native_vs_approx_{version}_{'_'.join(futures)}")


def fig_approx_failure_diagnosis(version='v1', name='z8000_h8000_W1'):
    """Why the reduced model fails: (a) first 800 ms model vs native (same initial state and input); (b) closure output at the
    native mean operating point vs diffusion-variance scaling; (c) closure alternatives at the operating point."""
    A = OUT / 'approx' / version / 'runs' / name; N = nc.NATIVE / 'runs' / name
    if not (A / 'fields.npz').exists() or not (N / 'readout_arrays.npz').exists():
        return None
    f = np.load(A / 'fields.npz'); fe = f['fields_hz'][:, 0]; fi = f['fields_hz'][:, 1]; w = f['count_e']; wi = f['count_i']
    geo = np.load(OUT / 'approx/coarse_20/geometry.npz'); wA = np.bincount(geo['cell_e'][geo['g175'] == 0], minlength=400).astype(float)
    e = np.average(fe, axis=1, weights=w); a = np.average(fe, axis=1, weights=wA); i_ = np.average(fi, axis=1, weights=wi)
    r = np.load(N / 'readout_arrays.npz'); r10 = r['r10']; reg = r['region10']; ri10 = r['ri10']
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2), constrained_layout=True)
    ax = axes[0]; T = 800; tm = np.arange(T) * 1e-3; tn = np.arange(T // 10) * .01 + .005
    ax.plot(tm, e[:T], color='#c33', lw=1, label='model all E'); ax.plot(tn, r10[:T // 10], color='k', lw=1, label='native all E')
    ax.plot(tm, a[:T], color='#e07b39', lw=1, ls='--', label='model core A'); ax.plot(tn, reg[:T // 10, 0], color='#e07b39', lw=1, label='native core A')
    ax.plot(tm, i_[:T], color='#7a5cc4', lw=.8, ls='--', label='model I'); ax.plot(tn, ri10[:T // 10], color='#7a5cc4', lw=.8, label='native I')
    ax.set_xlabel('time since continuation start (s)'); ax.set_ylabel('rate (Hz)'); ax.set_title(f'{name}: same initial state and input', fontsize=10); ax.legend(frameon=False, fontsize=8, ncol=2)
    ck = read(OUT / 'approx' / 'closure_operating_point_check.json')
    ax = axes[1]; keys = [k for k in ck if k.startswith('grid')]
    x = np.arange(len(keys)); ax.bar(x - .2, [ck[k]['closure_E_hz'] for k in keys], .4, color='#c33', label='closure E'); ax.bar(x + .2, [ck[k]['closure_I_hz'] for k in keys], .4, color='#7a5cc4', label='closure I')
    ax.axhline(ck[keys[0]]['native_E_hz'], color='k', ls='--', lw=1, label='native E (2-8 s)'); ax.axhline(ck[keys[0]]['native_I_hz'], color='#7a5cc4', ls=':', lw=1, label='native I (2-8 s)')
    ax.set_xticks(x); ax.set_xticklabels([k.replace('grid', 'g').replace('_joint', ' j').replace('True', '+').replace('False', '-').replace('mixed', 'mix').replace('colored', 'col') for k in keys], fontsize=8, rotation=45, ha='right'); ax.set_ylabel('rate at native mean rates (Hz)'); ax.legend(frameon=False, fontsize=8)
    ax.text(.02, .55, 'g20/g10 = 20x20 / 10x10 grid; mix/col = mixed / colored transfer; j+/j- = joint (Z, threshold) grouping on/off', transform=ax.transAxes, fontsize=7)
    ax.set_title('closure output at the native mean operating point', fontsize=10)
    ax = axes[2]; scal = [1, 4, 16, 64, 256]; vals = [0.5, 1.5, 6.7, 28.9, 80.6]
    ax.plot(scal, vals, 'o-', color='#c33'); ax.axhline(22.3, color='k', ls='--', lw=1); ax.set_xscale('log'); ax.set_xlabel('diffusion variance scale factor'); ax.set_ylabel('closure E rate (Hz)')
    ax.set_title('variance needed to reach the native mean rate', fontsize=10)
    return save(fig, f'fig_approx_failure_diagnosis_{version}')


def fig_path_replay(version='v1'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex='col')
    ok = False
    for j, seed in enumerate(SEEDS):
        p = OUT / 'approx' / version / 'runs' / f'path_s{seed}' / 'fields.npz'
        if not p.exists():
            continue
        ok = True; a = np.load(p); f = a['fields_hz']; t0 = int(a['start_step']) * DT_MS / 1000.; t = t0 + np.arange(len(f)) * 1e-3
        model_e = np.average(f[:, 0], axis=1, weights=a['count_e']); r10m = model_e.reshape(-1, 10).mean(1); tm = t0 + np.arange(len(r10m)) * .01 + .005
        run = OUT / 'replay' / 'runs' / f'eta0.0005_s{seed}'; d = R.load_chunks(run); r10 = R.rate_10ms(d['spikes_1ms'][:, 0], NE); tn = np.arange(len(r10)) * .01 + .005
        ax = axes[0, j]; ax.plot(tn, r10, color='k', lw=.8, label='native SNN'); ax.plot(tm, r10m, color='#c33', lw=.8, label='reduced model, real Z(t)')
        ax.set_xlim(t0, REPLAY_END_MS / 1000); ax.set_ylabel('all-E rate (Hz, 10 ms)'); ax.set_title(f'source trajectory seed {seed}')
        for n in (3, 4):
            ax.axvline(FIGURE_TIMES_S[n], color='#999', ls=':', lw=.8)
        ax.axvline(HIGH_ONSET_S, color='k', ls='--', lw=.8)
        cell = f[:, 0].reshape(-1, 10, f.shape[2]).mean(1) if False else None
        ax = axes[1, j]
        g = np.load(REF / 'geometry.npz'); w = g['cell_e_counts'].astype(float)
        T = len(d['field_1ms']) // 10 * 10
        nat_cell = d['field_1ms'][:T].astype(float).reshape(-1, 10, 400).sum(1) / w[None, :] / .01
        frac_n = np.average(nat_cell > 50, axis=1, weights=w)
        Tm = len(f) // 10 * 10; mod_cell = f[:Tm, 0].reshape(-1, 10, 400).mean(1); frac_m = np.average(mod_cell > 50, axis=1, weights=a['count_e'])
        ax.plot(tn, frac_n, color='k', lw=.8); ax.plot(tm, frac_m, color='#c33', lw=.8); ax.set_ylabel('fraction of cells >50 Hz'); ax.set_xlabel('time (s)')
        ax.axvline(HIGH_ONSET_S, color='k', ls='--', lw=.8)
    if not ok:
        plt.close(fig); return None
    axes[0, 0].legend(frameon=False)
    return save(fig, f'fig_path_replay_{version}')


def main(which):
    made = {}
    for name in which:
        made[name] = globals()[name]()
    print(json.dumps(made, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('which', nargs='*', default=['fig_native_state_map', 'fig_native_traces', 'fig_z_path']); a = ap.parse_args()
    main(a.which)
