"""Parent waveforms at the six independently checked, closely spaced H2 folds."""
from plot_rate_periodic_completion import *


def main():
    s=RateField();rows=[];colors=plt.get_cmap('viridis')(np.linspace(.08,.92,6))
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.3));fig.subplots_adjust(left=.08,right=.985,bottom=.24,top=.85,wspace=.25)
    for i,color in zip(range(3,9),colors):
        name=f'LPC_B{i}';q=read(PERIODIC_OUT/(name+'_validation.json'));assert q['status']=='VALIDATED_CYCLE_FOLD'
        meta=q['mesh_checks'][-1];z=np.load(meta['orbit']);r=np.array([s.regional_rates(v) for v in z['r']]);N=len(r)
        cf=np.fft.fft(r,axis=0);angle=float(np.angle(cf[1,1]));freq=np.fft.fftfreq(N)*N
        aligned=np.fft.ifft(cf*np.exp(-1j*freq*angle)[:,None],axis=0).real
        smooth=resample(aligned,1024,axis=0);phase=np.arange(1025)/1024
        for k in [0,1]:axes[k].plot(phase,np.r_[smooth[:,k],smooth[0,k]],color=color,lw=1.4,label=CRITICAL_LABELS[name])
        # One common phase shift for both cores; the within-cycle lag is preserved.
        assert np.max(abs(aligned.mean(0)-r.mean(0)))<1e-10
        rows.append(dict(label=CRITICAL_LABELS[name],J_EE_core=q['J_EE_core'],T_ms=q['T_ms'],
            source=str(PERIODIC_OUT/(name+'_validation.json')),orbit=meta['orbit'],
            B_fundamental_phase_rad=angle,core_A_B_mean_Hz=r.mean(0)[:2],core_A_B_max_Hz=r.max(0)[:2]))
    for k,ax in enumerate(axes):
        ax.set(xlim=(0,1),ylim=(0,3.1),xlabel='Phase within the full network period',
            ylabel='Rate (Hz / cell)' if k==0 else '',title=f'{"ab"[k]}   Core {"AB"[k]}')
        style(ax)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=6,frameon=False,bbox_to_anchor=(.52,.035))
    fig.suptitle('Closely spaced cycle folds: full period 303.26–303.39 ms',fontsize=13)
    save(fig,'dense_H2_fold_parent_waveforms')
    write(PERIODIC_OUT/'dense_H2_fold_parent_waveforms.json',dict(rows=rows,
        phase_alignment='One common shift fixing the phase of the Core B first Fourier harmonic; no separate core alignment.',
        scope='Actual parent periodic solutions of the same rate DDE. Low-amplitude oscillations, not large bursts. Fold existence and critical-mode validation come from the linked sources; no adjacent stability claim.'))
    update_readme(dict(dense_H2_fold_parent_waveforms='对比 H2 分支上六个已独立验证的密集折点 LPC14–19 的母周期轨道，两个面板使用同一纵轴范围。每条轨道只作一次共同相位平移，以 Core B 第一谐波定相，保留两核间相对时序；周期范围为 303.26–303.39 ms。**关注点**：Core A 波形变化明显而 Core B 波形近似重合，这些低幅周期解的折返不等于六次大幅 burst 起始，也不证明相邻分支稳定。'))


if __name__=='__main__':main()
