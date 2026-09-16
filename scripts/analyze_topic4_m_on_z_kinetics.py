#!/usr/bin/env python3
"""Aggregate only completed native M-on grid cells; retain censoring and pairing."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from run_topic4_m_on_z_kinetics import OUT, read, write, confirmed_entry


def edges(v, log=False):
    a = np.log(v) if log else np.asarray(v)
    b = np.r_[a[0]-(a[1]-a[0])/2, (a[1:]+a[:-1])/2, a[-1]+(a[-1]-a[-2])/2]
    return np.exp(b) if log else b


def main():
    p = read(OUT/'protocol.json')
    tau, thresholds = np.asarray(p['tau_s']), np.asarray(p['thresholds'])
    shape = (7, 7, 3)
    times = np.full(shape, np.nan); observed = np.full(shape, np.nan)
    rows = []; input_reference = {}; pair_checks = []
    for j in p['jobs']:
        folder = OUT/'runs'/j['name']
        if not (folder/'result.json').exists():
            continue
        r = read(folder/'result.json')
        assert r['status'] == 'COMPLETE' and r['job'] == j
        assert r['frozen_identity'] == p['identity']
        assert r['M_enabled'] and r['eta_m'] == .02 and r['tau_M_s'] == 2 and not r['reset_applied']
        x, y, si = j['x'], j['y'], p['seeds'].index(j['seed'])
        with np.load(folder/'observations.npz') as a:
            onset, confirmation = confirmed_entry(a['counts'])
            assert onset == r['onset_s'] and confirmation == r['confirmation_s']
            assert len(a['counts'])/100 == r['duration_s'] or abs(len(a['counts'])/100-r['duration_s']) < 1e-9
            if confirmation is None:
                assert abs(r['duration_s']-p['horizon_s']) < 1e-9
            assert np.array_equal(a['counts'][:, 0], a['regions'][:, :3].sum(1))
            h = a['input_digests']
            if si in input_reference:
                ref = input_reference[si]; n = min(len(ref), len(h))
                assert np.array_equal(ref[:n], h[:n]), 'Paired external input mismatch'
                pair_checks.append(dict(name=j['name'], common_input_prefix_s=n, status='PASS'))
                if len(h) > len(ref): input_reference[si] = h.copy()
            else:
                input_reference[si] = h.copy()
            re = a['counts'][:, 0]/32000/.01
            pre_end = len(re) if onset is None else round(onset*100)
            pre = re[max(0, pre_end-2000):pre_end]
            n = max(1, len(re)-50) if confirmation is not None else len(re)
            cut = a['slow_time_s'] <= (confirmation if confirmation is not None else p['horizon_s'])
            row = dict(name=j['name'], tau_Z_s=tau[x], threshold=thresholds[y], seed=j['seed'],
                 observed=confirmation is not None, onset_s=onset, confirmation_s=confirmation,
                 restricted_time_s=r['restricted_time_s'], horizon_s=p['horizon_s'],
                 pre_observation_s=len(pre)/100,
                 pre_E_mean_Hz=float(pre.mean()) if len(pre) else None,
                 pre_quiet_fraction=float(np.mean(pre<5)) if len(pre) else None,
                 Z_at_endpoint=float(a['Z'][cut][-1, 0]),
                 adaptation_current_at_endpoint=float(.02*a['M'][cut][-1, 0]),
                 reference_prefix_qa=r['reference_prefix_qa'], source=str(folder/'result.json'))
        rows.append(row); times[y, x, si] = row['restricted_time_s']; observed[y, x, si] = row['observed']
    count = np.isfinite(times).sum(-1); complete = count == 3
    mean = np.full((7, 7), np.nan); frac = mean.copy()
    mean[complete] = times[complete].mean(-1); frac[complete] = observed[complete].mean(-1)
    if rows:
        with (OUT/'transition_times.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez_compressed(OUT/'analysis_arrays.npz', tau_s=tau, thresholds=thresholds,
                        run_restricted_time_s=times, observed=observed, completed_count=count,
                        restricted_mean_s=mean, transition_fraction=frac)
    center = [r for r in rows if r['tau_Z_s'] == 5 and r['threshold'] == p['thresholds'][3]]
    summary = dict(status='COMPLETE_PENDING_REVIEW' if len(rows) == 147 else 'PARTIAL',
         completed=len(rows), total=147, complete_cells=int(complete.sum()),
         observed=sum(r['observed'] for r in rows), censored=sum(not r['observed'] for r in rows),
         baseline=center, input_pair_checks=pair_checks,
         latency='Confirmation of >=200 Hz all E for >=200 ms; initialization is time zero.',
         restriction_s=180, statistical_unit='Noise trajectory within fixed topology; three paired seeds, not patients or events.',
         endpoint_qa='Recomputed from saved native spike counts; censored trajectories checked to reach 180 s.',
         acceptance='F candidate only; no claim that complete Fig5 states 1-5 or sustained oscillation are established.',
         human_review='PENDING')
    write(OUT/'analysis_summary.json', summary)
    text = ['# M开启工作点的Z动力学扫描', '',
        f'正式完成 {len(rows)}/147 条，完整网格 {int(complete.sum())}/49 格；已完成中进入 {summary["observed"]} 条、180秒右删失 {summary["censored"]} 条。', '',
        '统计单位为固定拓扑下的一条噪声轨迹。三个配对种子跨参数比较；不能把网格或事件当作独立患者。',
        'M强度0.02、τM=2秒固定开启，Z=1/M=0开始；本扫描没有人工reset。首次进入时间与reset后再进入是不同问题。',
        'τZ控制耗竭和恢复两种速度；I_th是原始GABA电流触发耗竭的阈值，不是突触抑制强度，也不是每个事件耗竭量。',
        'F颜色为三条轨迹min(T确认,180秒)的均值；斜线表示至少一条右删失。灰色是尚未完成三个种子的格点，不属于右删失。',
        '配套进入比例图必须与时间图一起审阅；不能预设所有参数只改变早晚而不改变进入机会。', '',
        '## 当前工作点', '', '| 噪声 | 确认时间(s) | 观察窗内进入 |', '|---|---:|---|']
    for r in center:
        t = f'{r["confirmation_s"]:.2f}' if r['observed'] else '≥180（右删失）'
        text.append(f'| {r["seed"]} | {t} | {r["observed"]} |')
    text += ['', '完整Figure 5继续依照[验收约定](../reset_state_diagnosis_20260911/figure_acceptance.md)。'
             '本参数响应面不单独证明Hopf分岔、持续振荡发作、患者传播恢复或reset后再进入。']
    (OUT/'scientific_review.md').write_text('\n'.join(text)+'\n')
    if not complete.any():
        return
    plt.rcParams.update({'font.size':14, 'axes.labelsize':16, 'axes.titlesize':17,
                         'xtick.labelsize':14, 'ytick.labelsize':14, 'pdf.fonttype':42})
    xe, ye = edges(tau, log=True), edges(thresholds)
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#e5e5e5')
    def plot(ax, values, vmax, label, hatch=False):
        mesh = ax.pcolormesh(xe, ye, np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=vmax,
                             shading='flat', edgecolors=(1, 1, 1, .23), linewidth=.35)
        ax.set_xscale('log'); ax.set_xticks([2.5, 5, 10], ['2.5', '5', '10']); ax.minorticks_off()
        ax.set_yticks([75, p['thresholds'][3], 120], ['75', '95.2', '120'])
        ax.set_xlabel(r'$\tau_Z$ (s)'); ax.set_ylabel('Depletion threshold $I_{th}$\n(mV equiv.)')
        ax.plot(5, p['thresholds'][3], 'o', mfc='none', mec='white', ms=7, mew=1.5)
        if hatch:
            for y, x in zip(*np.where(complete & (frac < 1))):
                ax.add_patch(Rectangle((xe[x], ye[y]), xe[x+1]-xe[x], ye[y+1]-ye[y],
                             facecolor='none', hatch='////', edgecolor='#565656', linewidth=0))
        bar = ax.figure.colorbar(mesh, ax=ax, pad=.04); bar.set_label(label)
    folder = OUT/'figures'; folder.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 5.1))
    plot(ax, mean, 180, 'Restricted mean entry time (s)', True)
    ax.set_title('F  Z kinetics · M on', loc='left', weight='bold')
    fig.tight_layout(); fig.savefig(folder/'F_m_on_z_kinetics.png', dpi=200)
    fig.savefig(folder/'F_m_on_z_kinetics.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))
    plot(axes[0], mean, 180, 'Restricted mean entry time (s)', True)
    plot(axes[1], frac, 1, 'Fraction entering by 180 s')
    axes[0].set_title('Entry time'); axes[1].set_title('Entry fraction')
    fig.tight_layout(); fig.savefig(folder/'entry_time_and_fraction.png', dpi=180)
    fig.savefig(folder/'entry_time_and_fraction.pdf'); plt.close(fig)
    (folder/'README.md').write_text(
        '### F_m_on_z_kinetics.png / .pdf\n'
        '固定手放双核、M强度0.02和τM=2秒，扫描τZ与耗竭电流阈值。'
        '颜色为三个配对噪声种子的180秒限制平均首次进入时间；斜线为至少一个右删失，灰色为未完成网格。\n'
        '**关注点**：这里是首次进入的时间，不是reset后再次进入；待Agent及用户目视验收。\n\n'
        '### entry_time_and_fraction.png / .pdf\n'
        '并列显示限制平均时间与180秒内进入比例，避免把未进入一概解读为延迟。'
        '每格统计单位是三条噪声轨迹，拓扑保持固定。\n'
        '**关注点**：概率估计只有三个种子，须结合逐条记录和删失理解；灰色不代表没有进入。\n')


if __name__ == '__main__':
    main()
