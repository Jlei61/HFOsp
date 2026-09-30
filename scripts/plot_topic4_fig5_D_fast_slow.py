"""Reference-style D bifurcation figures and native slow-balance analysis.

No inferred nullcline in the rate-D projection; no interpolated missing branches.
Legacy input key `s` is only translated at the input boundary.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/topic4_sef_hfo'
SOURCE = RESULTS / 'fig5_current_network_z_state_v1'
BASE = RESULTS / 'fig5_z_bifurcation_preview_20260915'
OLD = RESULTS / 'fig5_z_branch_extension_20260915'
OUT = RESULTS / 'fig5_D_fast_slow_20260916'
FIG = OUT / 'figures'
PRIOR = BASE / 'frozen_filtered_v1'
BLUE, RED, CYAN, GREEN = '#2759b3', '#d4352b', '#22a3b5', '#44992d'
MARK_TIMES = [1.235, 4.025, 9.420, 10.370]


def read(p):
    return json.loads(p.read_text())


def native():
    cache = OUT / 'native_D_projection.npz'
    if cache.exists():
        return dict(np.load(cache))
    run = SOURCE / 'replay/runs/eta0.0005_s9108401'
    geo = np.load(SOURCE / 'approx/coarse_20/geometry.npz')
    groups = geo['g175']
    nr = np.load(SOURCE / 'replay/geometry.npz')['region_counts']
    assert np.array_equal(nr[:3], np.bincount(groups))
    cfg = read(run / 'applied_configuration.json')
    assert cfg['tau_Z_s'] == 5.
    parts = {k: [] for k in ('field_t', 'D', 'D_A', 'D_B', 'P', 'P_A', 'P_B', 'M', 'path_rms_Z')}
    path = np.load(BASE / 'spatial_z_path.npz')
    closest = np.inf
    for f in sorted((run / 'fields').glob('*.npz')):
        a = np.load(f); t = a['zm_step'] * .0001; z = a['z']; ii = a['ii']
        above = ii >= cfg['threshold']
        closest = min(closest, float(np.min(np.abs(ii.astype(float) - cfg['threshold']))))
        parts['field_t'].append(t); parts['D'].append(1-z.mean(1))
        parts['P'].append(above.mean(1)); parts['M'].append(a['m'].mean(1))
        D = 1-z.mean(1)
        interval = np.clip(np.searchsorted(path['depletion'], D)-1, 0, len(path['depletion'])-2)
        fraction = (D-path['depletion'][interval]) / (path['depletion'][interval+1]-path['depletion'][interval])
        approx = (1-fraction[:, None])*path['z_e'][interval] + fraction[:, None]*path['z_e'][interval+1]
        parts['path_rms_Z'].append(np.sqrt(np.mean((z-approx)**2, axis=1)))
        for k, tag in enumerate(('A', 'B')):
            parts[f'D_{tag}'].append(1-z[:, groups == k].mean(1))
            parts[f'P_{tag}'].append(above[:, groups == k].mean(1))
    out = {k: np.concatenate(v) for k, v in parts.items()}
    counts, regions, times = [], [], []
    for f in sorted((run / 'chunks').glob('*.npz')):
        a = np.load(f); counts.append(a['spikes_1ms']); regions.append(a['regions_1ms']); times.append(a['time_ms'])
    counts, regions = np.concatenate(counts), np.concatenate(regions)
    out['time_s'] = np.concatenate(times).reshape(-1, 10).mean(1) / 1000
    out['r_E'] = counts[:, 0].reshape(-1, 10).sum(1) / 32000 / .01
    rr = regions.reshape(-1, 10, 6).sum(1) / nr / .01
    out['r_A'], out['r_B'], out['r_surround'] = rr[:, 0], rr[:, 1], rr[:, 2]
    out['D_at_rate'] = np.interp(out['time_s'], out['field_t'], out['D'])
    out['Ddot_rhs'] = (out['P'] - out['D']) / 5.
    out['Ddot_fd'] = np.gradient(out['D'], out['field_t'])
    out['threshold'] = cfg['threshold']; out['current_distance_to_threshold'] = closest
    out['region_counts'] = nr
    np.savez_compressed(cache, **out)
    return out


def branch_data():
    m = np.load(SOURCE / 'approx/coarse_20/model.npz'); counts = m['count_e']
    certified = set()
    for p in [PRIOR / 'equilibrium_stability_certificates.json', OLD / 'equilibrium_stability_certificates.json',
              *OLD.glob('*complex_certificates*.json'), OUT / 'target_branch_complex_certificates.json']:
        if p.exists():
            for row in read(p):
                if row['status'].startswith('UNSTABLE'):
                    certified.add((row['branch'], row['index']))
    branches = []
    for folder, names in [(BASE, ['extended_low_equilibria']), (OLD, ['equilibrium_trusted_forward', 'equilibrium_low_trusted_continued', 'equilibrium_high_trusted', 'equilibrium_high_trusted_continued'])]:
        for name in names:
            a = np.load(folder / f'{name}.npz')
            branches.append(dict(name=name, D=a['s'], rate=np.average(a['r_hz'][:, :400], weights=counts, axis=1)))
    for p in OUT.glob('target_equilibria_*.npz'):
        a = np.load(p)
        branches.append(dict(name=p.stem, D=a['s'], rate=np.average(a['r_hz'][:, :400], weights=counts, axis=1)))
    h1, h2, fold = [read(p) for p in [PRIOR/'core_b_crossing.json', PRIOR/'oscillatory_crossing.json', BASE/'low_fold_check.json']]
    a = np.load(BASE / 'extended_resting_equilibria.npz')
    x = np.r_[a['s'], h1['s'], h2['s']]
    y = np.r_[np.average(a['r_hz'][:, :400], weights=counts, axis=1), h1['mean_e_hz'], h2['mean_e_hz']]
    order = np.argsort(x)
    return branches, certified, (x[order], y[order]), (h1, h2, fold), read(OLD / 'figure_data.json')['displayed_cycles']


def draw_branches(ax, data, marks=False):
    branches, certified, rest, events, cycles = data
    x, y = rest
    for sel, ls in [(x <= events[0]['s'], '-'), (x >= events[0]['s'], '--')]:
        ax.plot(x[sel], y[sel], color='k', ls=ls, lw=1.35)
    for b in branches:
        pts = np.c_[b['D'], b['rate']]
        if len(pts) < 2: continue
        seg = np.stack([pts[:-1], pts[1:]], axis=1)
        known = np.array([(b['name'], k) in certified and (b['name'], k+1) in certified for k in range(len(pts)-1)])
        for mask, ls in [(known, '--'), (~known, ':')]:
            if mask.any():
                ax.add_collection(LineCollection(seg[mask], colors='k', linestyles=ls, linewidths=.85, zorder=2))
    for family in ('A', 'B', 'physical'):
        rows = sorted([r for r in cycles if r['family'] == family], key=lambda r: r['order'])
        xx = [r['s'] for r in rows]
        # A single physical waveform is a point, never a fabricated branch.
        for key, color, ls in [('minimum_hz', CYAN, '--'), ('maximum_hz', CYAN, '--'), ('mean_hz', RED, '-')]:
            ax.plot(xx, [r[key] for r in rows], color=color, lw=1.35, ls=ls,
                    marker='o' if len(rows) == 1 else None, ms=4, zorder=4)
    if marks:
        turn = read(OLD / 'physical_secondary_fold_review.json')
        ax.plot(turn['s'], turn['mean_e_hz'], 'o', color='k', mfc='white', ms=4, zorder=7)
        ax.annotate('TP', (turn['s'], turn['mean_e_hz']), xytext=(8, 12), textcoords='offset points', fontsize=10)
        target = OUT / 'target_root_stability.json'
        if target.exists():
            q = read(target)
            ax.plot(q['D'], q['global_E_hz'], 's', color='k', mfc='white', ms=5, zorder=8)
            ax.annotate(r'$Q_3$', (q['D'], q['global_E_hz']), xytext=(8, 12), textcoords='offset points', fontsize=11)


def labels(ax, letter):
    ax.text(-.10, 1.015, letter, transform=ax.transAxes, fontsize=18, fontweight='bold', va='bottom')


def save(fig, name):
    # No titles or prose footnotes: full scientific qualification lives in README.
    assert all(ax.get_title() == '' for ax in fig.axes)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main_figure(n, data):
    fig = plt.figure(figsize=(12.6, 7.9))
    gs = fig.add_gridspec(3, 2, left=.085, right=.975, bottom=.20, top=.97,
                          width_ratios=[2.0, 1], wspace=.30, hspace=.20)
    ax = fig.add_subplot(gs[:, 0]); labels(ax, 'A')
    draw_branches(ax, data, marks=True)
    sel = n['time_s'] <= 10.37
    ax.plot(n['D_at_rate'][sel], n['r_E'][sel], color=BLUE, lw=.75, alpha=.85, zorder=3)
    ax.axvline(0, color='k', ls=':', lw=.7)
    ax.set(xlim=(-.14, .325), ylim=(0, 520), xlabel=r'$D=1-\langle Z_E\rangle$', ylabel=r'Global E rate (Hz / neuron)')
    ax.set_yscale('symlog', linthresh=.1, linscale=.8)
    ticks = [0, .1, 1, 10, 100, 500]; ax.set_yticks(ticks); ax.set_yticklabels([f'{v:g}' for v in ticks]); ax.minorticks_off()
    # Keep the enlargement entirely within the negative-D area; do not cover
    # physical branches, the D=0 waveform maximum, or the native trajectory.
    inset = ax.inset_axes([.060, .65, .225, .29]); draw_branches(inset, data)
    inset.set(xlim=(-.111, -.02), ylim=(.032, .20)); inset.set_xticks([-.1, -.05]); inset.set_yticks([.05, .1, .15])
    inset.tick_params(labelsize=8, length=2)
    for spine in inset.spines.values(): spine.set_visible(True)
    for h, text_, offset in zip(data[3], ['H1', 'H2', 'F0'], [(-6, -24), (18, 7), (-22, 15)]):
        inset.plot(h['s'], h['mean_e_hz'], 'o', ms=3, mfc='white', mec='k', zorder=8)
        inset.annotate(text_, (h['s'], h['mean_e_hz']), xytext=offset, textcoords='offset points', fontsize=8,
                       arrowprops=dict(arrowstyle='-', lw=.6))
    for num, t in enumerate(MARK_TIMES, 1):
        k = int(np.argmin(abs(n['time_s']-t))); D = np.interp(t, n['field_t'], n['D'])
        ax.plot(D, n['r_E'][k], 'o', ms=5, color=BLUE, mfc='white', zorder=6)
        ax.annotate(str(num), (D, n['r_E'][k]), xytext=(5, 8 if num != 2 else -16), textcoords='offset points', fontsize=11, color=BLUE)
    for k, key in enumerate(('r_E', 'r_A', 'r_B')):
        sub = fig.add_subplot(gs[k, 1]); labels(sub, chr(ord('B')+k))
        sub.plot(n['time_s'], n[key], color=BLUE, lw=.75)
        sub.set(xlim=(0, 10.5), ylim=(0, max(n[key][n['time_s'] < 10.5])*1.05),
                ylabel=['Global E (Hz)', 'Core A E (Hz)', 'Core B E (Hz)'][k])
        for t in MARK_TIMES:
            sub.axvline(t, color='k', lw=.5, ls=':')
        sub.set_xticks([0, 3, 6, 9])
        if k == 2: sub.set_xlabel('Time (s)')
        else: sub.tick_params(labelbottom=False)
    handles = [Line2D([], [], color='k', lw=1.3, label='Reduced equilibrium: stable'),
               Line2D([], [], color='k', ls='--', label='Reduced equilibrium: unstable'),
               Line2D([], [], color='k', ls=':', label='Reduced equilibrium: unresolved'),
               Line2D([], [], color=RED, label='Period mean'),
               Line2D([], [], color=CYAN, ls='--', label='Periodic maximum / minimum'),
               Line2D([], [], color=BLUE, label='Native SNN trajectory')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .035), ncol=3, frameon=False,
               columnspacing=1.7, fontsize=10)
    save(fig, 'fig5_D_bifurcation')


def balance_figure(n):
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4))
    fig.subplots_adjust(left=.095, right=.975, bottom=.105, top=.96, wspace=.32, hspace=.35)
    t = n['field_t']; sel = t <= 10.37
    ax = axes[0, 0]; labels(ax, 'A')
    ax.plot(t[sel], n['P'][sel], color=BLUE, lw=.7, label=r'$P(t)$')
    ax.plot(t[sel], n['D'][sel], color=GREEN, lw=1.5, label=r'$D(t)$')
    ax.set(xlim=(0, 10.5), ylim=(0, 1.02), ylabel='E-cell fraction', xlabel='Time (s)')
    ax.legend(frameon=False, loc='upper left')
    ax = axes[0, 1]; labels(ax, 'B')
    ax.plot(n['D'][sel], n['P'][sel], color=BLUE, lw=.65, alpha=.85)
    ax.plot([0, .33], [0, .33], color=GREEN, lw=1.7, label=r'$\dot D=0$')
    ax.set(xlim=(0, .33), ylim=(0, 1.02), xlabel=r'$D=1-\langle Z_E\rangle$', ylabel=r'$P$')
    ax.legend(frameon=False, loc='upper left')
    ax = axes[1, 0]; labels(ax, 'C')
    ax.plot(t[sel], n['Ddot_rhs'][sel], color=BLUE, lw=.7)
    ax.axhline(0, color=GREEN, lw=1.1); ax.set(xlim=(0, 10.5), xlabel='Time (s)', ylabel=r'$\dot D$ (s$^{-1}$)')
    ax = axes[1, 1]; labels(ax, 'D')
    for key, color, name in [('D', 'k', 'Global E'), ('D_A', '#b77a20', 'Core A'), ('D_B', '#238b80', 'Core B')]:
        ax.plot(t[sel], n[key][sel], color=color, lw=1.3, label=name)
    ax.set(xlim=(0, 10.5), xlabel='Time (s)', ylabel=r'$D$'); ax.legend(frameon=False, loc='upper left')
    for ax in axes.ravel():
        if ax is not axes[0, 1]:
            ax.axvline(9.42, color='k', lw=.6, ls=':')
    save(fig, 'fig5_D_slow_balance')


def fixed_figure():
    model = np.load(SOURCE / 'approx/coarse_20/model.npz')
    count, w = model['count_e'], model['threshold_weights_e']
    fig, axes = plt.subplots(3, 1, figsize=(8, 5.4), sharex=True)
    fig.subplots_adjust(left=.14, right=.97, bottom=.14, top=.96, hspace=.20)
    for ax, name in zip(axes, ('s0_extension', 's0.1', 's0.228845')):
        a = np.load(OLD / 'deterministic' / f'{name}.npz'); r = a['r_hz']; D = float(a['s'])
        g = (r[:, :3200].reshape(-1, 400, 8) * w).sum(2) @ count / count.sum()
        ax.plot(np.arange(1, len(g)+1)/1000, g, color=BLUE, lw=.85)
        ax.set(ylabel='E rate (Hz)', xlim=(0, 3), ylim=(0, max(g)*1.12))
        ax.text(.02, .80, rf'$D={D:.5f}$', transform=ax.transAxes, fontsize=11,
                bbox=dict(facecolor='white',edgecolor='none',pad=1.5))
    axes[-1].set_xlabel('Time after field change (s)')
    save(fig, 'fig5_D_frozen_trajectories')


def local_figure():
    if not (OUT/'target_stationary_fold_review.json').exists(): return
    model=np.load(SOURCE/'approx/coarse_20/model.npz'); geo=np.load(SOURCE/'approx/coarse_20/geometry.npz')
    count=model['count_e']; cell=geo['cell_e']; pos=geo['positions_e']
    xy=np.c_[np.bincount(cell,weights=pos[:,0],minlength=400)/count,
             np.bincount(cell,weights=pos[:,1],minlength=400)/count]
    a=np.load(OUT/'target_equilibria_+1.npz'); q=read(OUT/'target_root_stability.json')
    f=read(OUT/'target_stationary_fold_check.json'); D3=q['D']
    fig,axes=plt.subplots(2,2,figsize=(10.2,8.0))
    fig.subplots_adjust(left=.10,right=.93,bottom=.095,top=.965,wspace=.38,hspace=.36)
    ax=axes[0,0]; labels(ax,'A'); xx=(a['s']-D3)*1e4; yy=np.average(a['r_hz'][:,:400],weights=count,axis=1)
    ax.plot(xx,yy,'k--',lw=1.5);ax.plot(0,q['global_E_hz'],'ks',mfc='white',ms=5)
    ax.annotate(r'$Q_3$',(0,q['global_E_hz']),xytext=(7,-8),textcoords='offset points')
    xf=(f['s']-D3)*1e4
    ax.plot(xf,f['mean_e_hz'],'ko',mfc='white',ms=5)
    ax.annotate('TP2',(xf,f['mean_e_hz']),xytext=(-40,-15),textcoords='offset points')
    ax.set(xlabel=r'$10^4(D-D_3)$',ylabel='Global E rate (Hz / neuron)',xlim=(-.2,4))
    ax=axes[0,1];labels(ax,'B');rows=read(OUT/'target_branch_complex_certificates.json')
    ax.plot([(r['D']-D3)*1e4 for r in rows],[r['lambda_per_s'][0] for r in rows],'k-',lw=1.4)
    ax.axhline(0,color='k',lw=.8,ls=':');ax.set(xlabel=r'$10^4(D-D_3)$',ylabel=r'Re $\lambda$ (s$^{-1}$)',xlim=(-.2,4),ylim=(-.4,10))
    spatial=[]
    for ax,name,key,label_,letter in zip(axes[1],['target_root_unstable_mode','target_stationary_fold_state'],
                                        ['mode','right_mode'],[r'$Q_3$','TP2'],['C','D']):
        z=np.load(OUT/f'{name}.npz');v=z[key][:400]
        mag=abs(v);norm=mag/mag.max(); energy=count*mag**2;energy/=energy.sum()
        ix,iy=np.floor(xy).astype(int).T
        assert len(set(zip(ix,iy)))==400
        grid=np.empty((20,20));grid[iy,ix]=norm
        im=ax.imshow(grid,origin='lower',extent=(0,20,0,20),cmap='viridis',vmin=0,vmax=1,interpolation='nearest')
        labels(ax,letter); ax.text(.02,.96,label_,transform=ax.transAxes,va='top',color='white',fontsize=12)
        for txt,(cx,cy) in zip(['A','B'],[(4.19921431597,9.12890135365),(16.47920304044,3.965511533)]):
            ax.add_patch(plt.Circle((cx,cy),1.75,fill=False,color='white',lw=1.1))
            ax.text(cx,cy,txt,ha='center',va='center',color='white',fontsize=10)
        ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)')
        spatial.append(dict(mode=name,effective_spatial_cells=float(1/np.dot(energy,energy)),
                            largest_cell_energy_fraction=float(energy.max()),largest_cell=int(energy.argmax()),
                            largest_cell_xy_mm=xy[energy.argmax()].tolist(),
                            quantity='E macro-rate mode; cell energy = E-cell count times squared mode amplitude'))
    cb=fig.colorbar(im,ax=list(axes[1]),fraction=.03,pad=.035);cb.set_label(r'$|\delta r_E|\,/\,\max|\delta r_E|$')
    (OUT/'spatial_modes.json').write_text(json.dumps(spatial,indent=2)+'\n')
    save(fig,'fig5_D_local_branch_and_modes')


def analysis(n):
    rows = []
    for num, t in enumerate(MARK_TIMES, 1):
        k = int(np.argmin(abs(n['field_t']-t))); D = float(np.interp(t, n['field_t'], n['D']))
        lo, hi = round(t-.05, 6), round(t+.05, 6)
        a = (n['field_t'] > lo) & (n['field_t'] < hi)
        integral_t = np.r_[lo, n['field_t'][a], hi]
        Pmean = np.trapz(np.interp(integral_t, n['field_t'], n['P']), integral_t)/(hi-lo)
        driftmean = np.trapz(np.interp(integral_t, n['field_t'], n['Ddot_rhs']), integral_t)/(hi-lo)
        r = (np.round(n['time_s'], 6) >= lo) & (np.round(n['time_s'], 6) < hi)
        assert r.sum() == 10
        rows.append(dict(state=num, time_s=t, D=D, D_core_A=float(n['D_A'][k]), D_core_B=float(n['D_B'][k]),
                         snapshot_time_s=float(n['field_t'][k]),
                         P_snapshot=float(n['P'][k]), Ddot_snapshot_per_s=float(n['Ddot_rhs'][k]),
                         P_mean_100ms=float(Pmean), Ddot_mean_100ms_per_s=float(driftmean),
                         global_E_rate_mean_100ms_hz=float(n['r_E'][r].mean())))
    # Floating-point checkpoint check uses full precision I_I, independently of float32 sampled current.
    run = SOURCE / 'replay/runs/eta0.0005_s9108401'
    checks = []
    for ms in (8000, 9000, 9300, 9420, 9870, 10370):
        a = np.load(run / 'checkpoints' / f't{ms}ms.npz'); z = a['slow__z'][:32000]
        ii = a['slow__I_I_last'][:32000]; p = np.mean(ii >= n['threshold']); D = 1-z.mean()
        updated = z + .1/5000 * ((ii < n['threshold'])-z)
        empirical = ((1-updated.mean())-D)/.0001; formula=(p-D)/5
        checks.append(dict(time_s=ms/1000, D=float(D), P_last_step=float(p),
                           Ddot_one_step_per_s=float(empirical), Ddot_formula_per_s=float(formula), error=abs(empirical-formula)))
    out = dict(definition='D_i = 1-Z_i; plotted global D is the E-neuron mean',
               equation='dD/dt = (P-D)/tau_Z; P = mean_E(I_I >= I_th); tau_Z=5 s',
               boundary_in_D_P_plane='P=D is the exact instantaneous zero-D-drift locus, not a full-system nullcline or equilibrium',
               boundary_in_D_rate_plane='NOT_CLOSED: P depends on spatial inhibitory currents, not solely global E rate',
               reference='El Houssaini et al. 2020, eNeuro, Figure 13B,D; DOI 10.1523/ENEURO.0485-18.2019',
               reference_URL='https://pmc.ncbi.nlm.nih.gov/articles/PMC7096539/#F13',
               rate_bin_ms=10, field_sampling_ms=[5, 10], unit='one native realization, topology 6101 / noise 9108401',
               mean_window='100 ms, rate: 10 nonoverlapping bins; P and drift: trapezoidal time weighting of sampled fields',
               path_field_rms_Z_max_through_state4=float(n['path_rms_Z'][n['field_t'] <= 10.37].max()),
               state_readouts=rows, checkpoint_euler_checks=checks,
               checkpoint_check_caveat='Algebraic one-step check using saved last input, not an independent replay of the next input',
               target_root_search=read(OUT/'target_root_search.json') if (OUT/'target_root_search.json').exists() else [],
               target_root_stability=read(OUT/'target_root_stability.json') if (OUT/'target_root_stability.json').exists() else {},
               target_stationary_fold_review=read(OUT/'target_stationary_fold_review.json') if (OUT/'target_stationary_fold_review.json').exists() else {},
               native_state3_bifurcation='NOT_ESTABLISHED', user_visual_acceptance='PENDING')
    (OUT / 'analysis.json').write_text(json.dumps(out, indent=2) + '\n')
    with (OUT / 'native_state_D_readouts.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows, indent=2), flush=True)


def main():
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.labelsize':12,
                         'axes.linewidth':.8, 'axes.spines.top':False, 'axes.spines.right':False,
                         'pdf.fonttype':42, 'svg.fonttype':'none'})
    n = native(); data = branch_data()
    main_figure(n, data); balance_figure(n); fixed_figure(); local_figure(); analysis(n)
    print('COMPLETE', FIG, flush=True)


if __name__ == '__main__':
    main()
