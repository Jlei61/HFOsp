"""Use independently checked fold orbits to display core rhythm relationships."""
from plot_rate_periodic_completion import *


def main():
    names=['LPC_A2','LPC_A3','LPC_A5','LPC_B2','LPC_B3','LPC_B4']
    s=RateField();rows=[];curves=[]
    for name in names:
        source=PERIODIC_OUT/(name+'_validation.json');q=read(source)
        assert q['status']=='VALIDATED_CYCLE_FOLD'
        meta=q['mesh_checks'][-1];z=np.load(meta['orbit']);r=z['r'];N=len(r)
        rates=np.array([s.regional_rates(v) for v in r]);cf=np.fft.rfft(rates,axis=0)/N
        harmonic=np.argmax(abs(cf[1:,:2]),axis=0)+1
        # hB*phiA-hA*phiB is invariant to one common temporal phase.
        phase=float(np.angle(np.exp(1j*(harmonic[1]*np.angle(cf[harmonic[0],0])-
                                       harmonic[0]*np.angle(cf[harmonic[1],1])))))
        shifted_cf=cf*np.exp(2j*np.pi*np.arange(len(cf))[:,None]*.137)
        shifted_phase=float(np.angle(np.exp(1j*(harmonic[1]*np.angle(shifted_cf[harmonic[0],0])-
                                               harmonic[0]*np.angle(shifted_cf[harmonic[1],1])))))
        assert abs(np.angle(np.exp(1j*(shifted_phase-phase))))<1e-12
        shift=int(np.argmax(rates[:,0])-N//4);rates=np.roll(rates,-shift,axis=0)
        curves.append(rates)
        rows.append(dict(label=CRITICAL_LABELS[name],internal_label=name,validation_source=str(source),
            orbit=meta['orbit'],N=N,J_EE_core=float(z['J']),T_ms=float(z['T']),
            mean_A_B_surround_Hz=rates.mean(0),peak_to_peak_A_B_surround_Hz=np.ptp(rates,axis=0),
            dominant_harmonics_A_B=harmonic,phase_invariant_radians=phase,
            common_display_phase_shift_samples=shift,
            critical_type='cycle fold',stability='Not inferred from the critical-mode check'))
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,3,figsize=(13.4,7.2))
    fig.subplots_adjust(left=.065,right=.98,bottom=.10,top=.87,wspace=.26,hspace=.43)
    for index,(q,rates) in enumerate(zip(rows,curves)):
        ax=axes.ravel()[index];t=np.linspace(0,1,len(rates)+1)
        for k,color in enumerate([*COL,'#777777']):
            ax.plot(t,np.r_[rates[:,k],rates[0,k]],color=color,lw=1.3 if k<2 else .8)
        ax.set(xlim=(0,1),ylim=(0,max(rates[:,:2].max()*1.12,1)),xticks=[0,.25,.5,.75,1],
            xlabel='Fraction of full-network period',ylabel='Rate (Hz / E cell)' if index%3==0 else '',
            title=q['label']+rf'  |  $J_{{\mathrm{{EE,core}}}}={q["J_EE_core"]:.6f}$'+
            f'\nT = {q["T_ms"]:.2f} ms; A:B harmonics = {q["dominant_harmonics_A_B"][0]}:{q["dominant_harmonics_A_B"][1]}')
        style(ax)
    fig.legend(handles=[Line2D([0],[0],color=c,label=n) for c,n in
        zip([*COL,'#777777'],['Core A','Core B','Surround'])],loc='upper center',ncol=3,frameon=False)
    save(fig,'validated_cycle_fold_core_interaction')
    write(PERIODIC_OUT/'validated_cycle_fold_core_interaction.json',dict(rows=rows,
        definition='Highest-resolution independently validated critical orbit; one common display phase. Harmonics are relative to the full-network period, and the stored phase combination is invariant under a common shift.',
        scope='The displayed points are cycle folds, not period-doubling points. A 2:1 dominant-harmonic relation is a waveform property; it does not prove a PD origin, a coupling mechanism, stable attraction, or native SNN correspondence.'))
    update_readme(dict(validated_cycle_fold_core_interaction='展示六个已通过独立临界模态检查的周期折叠上的 Core A、Core B 与核外波形，均读取相应精化周期轨道，采用共同的时间平移。上排为 H1 出发分支的三个折点，下排为 H2 出发分支的三个折点，标出完整网络周期及两核主谐波。**关注点**：1:1 或 2:1 是轨道上的波形关系；这些点的类型仍为周期折叠，不能由 2:1 谐波反推倍周期分岔或稳定 burst。'))


if __name__=='__main__':main()
