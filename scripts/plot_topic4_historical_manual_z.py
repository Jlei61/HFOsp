#!/usr/bin/env python3
"""Raster, applied neuron-wise Z and population rates from the saved trajectory."""
from validate_topic4_fixed_rate_base import ROOT, read, write
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from topic4_historical_manual_z_common import OUT, TIMES_MS, ARM


def render():
    a = np.load(OUT / 'trajectory.npz'); meta = read(OUT / 'run.json')
    assert all(np.isfinite(a[k]).all() for k in a.files)
    duration = meta['duration_ms'] / 1000.; restore = meta['restore_start_ms']
    t = a['z_time_ms'] / 1000.; z = a['z_stats']; rates = np.c_[a['rate_e_hz'], a['rate_i_hz']].reshape(-1, 50, 2).mean(1)
    tr = (np.arange(len(rates)) + .5) * .005
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(4, 1, figsize=(14, 11), sharex=True, layout='constrained',
                           gridspec_kw={'height_ratios': [1, 1.6, 1.1, .9]})
    ax[0].fill_between(t, z[:, 2], z[:, 4], color='#7b3294', alpha=.15, label='E cells: 10th–90th percentile')
    ax[0].plot(t, z[:, 0], c='#7b3294', label='E mean Z')
    for k, label, c in [(5, 'Near core A', '#ac4086'), (6, 'Near core B', '#3b83b4'), (7, 'Surround', '.45')]:
        ax[0].plot(t, z[:, k], c=c, lw=.8, label=label)
    ax[0].set(ylabel='Applied inhibition\ncoefficient Z', ylim=(-.03, 1.08))
    ax[0].legend(ncol=3, fontsize=9, loc='lower left')
    for lo, hi, color in [(0, 240, '#2876ad'), (240, 300, '#e17c24')]:
        times, ids = np.nonzero(a['sample_spikes'][:, lo:hi])
        ax[1].scatter(times * .0001, ids + lo, s=.3, c=color, linewidths=0, rasterized=True)
    for y in [60, 120, 240]: ax[1].axhline(y-.5, c='.75', lw=.6)
    ax[1].set(ylim=(-1, 300), yticks=[30, 90, 180, 270], yticklabels=['Near core A E', 'Near core B E', 'Other E', 'I'])
    for k, label, c in [(0, 'All E', '#2876ad'), (1, 'All I', '#e17c24')]:
        ax[2].plot(tr, rates[:, k], c=c, lw=.8, label=label)
    ax[2].set(ylabel='Population rate\n(Hz; 5-ms bins)'); ax[2].legend(loc='upper right')
    ax[3].plot(t, z[:, 8], color='#a65f20', label='E cells with GABA above depletion threshold')
    ax[3].plot(t, z[:, 1], color='#7b3294', label='Across-E standard deviation of Z')
    ax[3].set(ylabel='Fraction / Z SD', xlabel='Time (s)', xlim=(0, duration)); ax[3].legend(fontsize=9, loc='upper right')
    if restore is not None:
        r = restore / 1000.
        for axis in ax:
            axis.axvspan(r, r + 1, color='#26845d', alpha=.10)
            axis.axvline(r, c='#26845d', ls='--', lw=.8)
            axis.axvline(r + 1, c='#26845d', ls=':', lw=.8)
        ax[0].set_title(f'Native Z dynamics until {r:.2f} s; external refill to 1 over 1 s, then clamp at 1', fontsize=12)
    else:
        ax[0].set_title('No sustained-high trigger within 20 s: Z remains autonomous; no manual refill was applied', fontsize=12)
    fig.suptitle('Historical manually placed dual-core field: native Z evolution and manual inhibition restoration\nCore centres (4.20, 9.13), (16.48, 3.97) mm; current fast connectivity and OU retained', fontsize=13)
    fig.supxlabel(r'Native E-only Z: $5000\,\mathrm{ms}\,\dot z_i=\mathbf{1}[I_{\mathrm{GABA},i}<95.20]-z_i$; I-cell Z = 1; M off.'
                  '\nManual refill changes only Z; membrane, synapses, delays and random state are continuous. Raster: fixed samples, core neighbourhoods r<1.75 mm; rates: all 40,000. Manual core radius: 1.5 mm.', fontsize=10)
    folder = OUT / 'figures'; folder.mkdir(exist_ok=True)
    fig.savefig(folder / 'autonomous_z_manual_restore.png', dpi=180)
    fig.savefig(folder / 'autonomous_z_manual_restore.pdf'); plt.close(fig)
    def stats(lo, hi):
        sel = (tr >= lo) & (tr < hi); y = rates[sel, 0]
        return {'time_s': [lo, hi], 'mean_E_hz': float(y.mean()), 'peak_E_5ms_hz': float(y.max()),
                'CV_E_5ms': float(y.std() / max(y.mean(), 1e-12)), 'fraction_E_below1Hz': float(np.mean(y < 1))}
    summary = {'status': 'COMPLETE_PENDING_REVIEW', 'baseline': stats(.5, 1.),
               'final_1s': stats(duration-1., duration), 'minimum_mean_Z': float(z[:, 0].min()),
               'max_spatial_Z_std': float(z[:, 1].max()),
               'sustained_high_detected_ms': meta['sustained_high_detected_ms'],
               'restore_start_ms': restore,
               'no_new_biological_variable': True,
               'interpretation_limit': 'One historical parameter transfer; manually restored Z is not autonomous event termination or patient IED validation.'}
    if restore is not None:
        r = restore / 1000.; summary['before_manual_restore'] = stats(r-.5, r)
    write(OUT / 'analysis.json', summary)
    (folder / 'README.md').write_text('### autonomous_z_manual_restore.png\n同一双核 SNN 在原 OU 背景下使用逐神经元 Z 原方程，M 关闭；上排显示 Z 分布及区域均值，中排为真实 raster 和全群放电率，下排区分耗竭驱动与空间异质性。若满足持续高活动触发条件，绿色段只将 Z 手动恢复到 1，随后夹持；若未触发，则全程保持自主演化。PDF 为同名版本。\n**关注点**：旧 Z 能否自行把网络带入高活动，以及补回 Z 后是否退出；不能把手动补回解释为自主终止。\n')





def frozen():
    status=read(OUT/'native_batch_status.json');assert status['status']=='COMPLETE'
    sub=read(OUT/'substrate.json');meta=read(OUT/'run.json');centers=np.array(sub['centers_mm'])
    fig,axs=plt.subplots(3,5,figsize=(16,9.5),layout='constrained',gridspec_kw={'height_ratios':[1,1,1.2]})
    records=[]
    for col,tm in enumerate(TIMES_MS):
        name=f'frozen_t{tm}';a=np.load(OUT/'native'/f'{name}.npz');row=read(OUT/'native'/f'{name}.json')
        assert np.array_equal(a['initial_z_e'],a['final_z_e'])
        e=a['rate_e_hz'].reshape(-1,50).mean(1);t=(np.arange(len(e))+.5)*.005
        restore=meta['restore_start_ms'];phase='native depletion' if restore is None or tm<restore else 'external refill / restored'
        axs[0,col].plot(t,e,c='#262626',lw=.8)
        axs[0,col].set(xlim=(0,2),ylim=(0,max(460,max(r['late_E_peak_5ms_hz'] for r in status['rows'])*1.15)),
            title=f'Freeze at {tm/1000:g} s\nmean Z = {row["initial_Z_mean"]:.3f}')
        axs[0,col].text(.06,.95,f'Late E: {row["late_E_mean_hz"]:.0f} Hz\nQuiet bins: {row["late_E_quiet_fraction"]:.0%}',transform=axs[0,col].transAxes,va='top',fontsize=10)
        ids=np.arange(0,300,4);st,sn=np.where(a['sample_spikes'][-5000:,ids])
        axs[1,col].scatter(st*.0001+1.5,sn,s=.4,c=np.where(ids[sn]<240,'#222222','#477eaa'),rasterized=True)
        axs[1,col].set(xlim=(1.5,2),ylim=(-1,len(ids)),xlabel='Continuation time (s)')
        cells=a['field_e_count_1ms'][-1000:].mean(0)/a['cell_e_counts']*1000
        im=axs[2,col].imshow(cells.reshape(20,20),origin='lower',extent=(0,20,0,20),vmin=0,vmax=500,cmap='magma',interpolation='nearest')
        axs[2,col].scatter(centers[:,0],centers[:,1],s=90,facecolors='none',edgecolors='#65dfdf',lw=1.5)
        axs[2,col].set(xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20])
        if col:
            for rr in range(3):axs[rr,col].tick_params(labelleft=False)
        records.append({**row,'reference_phase_at_checkpoint':phase})
    axs[0,0].set_ylabel('Mean E rate (Hz)');axs[1,0].set_ylabel('Sampled neuron');axs[2,0].set_ylabel('y (mm)')
    fig.suptitle('Historical manually placed dual-core field | actual neuron-wise Z frozen\nNative SNN fast state, synapses, delays and OU history carried forward',fontsize=14)
    fig.colorbar(im,ax=axs[2,:],shrink=.75,label='Late E rate (Hz)')
    folder=OUT/'figures';fig.savefig(folder/'native_frozen_z_raster_and_recruitment.png',dpi=180,bbox_inches='tight');fig.savefig(folder/'native_frozen_z_raster_and_recruitment.pdf',bbox_inches='tight');plt.close(fig)
    with (folder/'README.md').open('a') as stream:
        stream.write('\n### native_frozen_z_raster_and_recruitment.png\n沿本次手放双核原生轨迹的8、9.4、9.8、10.18、10.68秒，各自冻结当时完整逐神经元Z并延续2秒，膜状态、突触、延迟和OU历史均连续。上排为全E率，中排为相同样本最后500毫秒raster，下排为最后1秒空间平均E率，圆圈标出历史双核中心；静默比例为最后1秒5毫秒群体率低于1 Hz的比例。\n**关注点**：时间点与旧图相同，但新基底达到的Z和活动阶段可能不同；这不是共享初态/未来输入的纯Z因果对照，也不能由2秒延续证明渐近稳定性。\n')
    write(OUT/'frozen_figure_metrics.json',records)


def main():
    render();frozen()
    write(OUT/'delivery_status.json',{'status':'FIGURES_COMPLETE_PENDING_VISUAL_REVIEW','figures':['figures/autonomous_z_manual_restore.png','figures/native_frozen_z_raster_and_recruitment.png']})


if __name__=='__main__':main()
