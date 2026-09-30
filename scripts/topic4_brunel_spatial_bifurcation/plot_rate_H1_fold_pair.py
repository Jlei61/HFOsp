"""Local branch geometry, periods and independent +1-mode checks at LPC61/62."""
from plot_rate_periodic_completion import *


def main():
    s=RateField();segment=read(PERIODIC_OUT/'arcAconnectionStage4_accuracy.json')
    assert segment['status']=='SAMPLED_PASS'
    rows=[read(Path(path).with_suffix('.json')) for path in segment['included_orbits'][30:66]]
    roots=[read(PERIODIC_OUT/f'LPC_A_stage4_turn{i}_validation.json') for i in [1,2]]
    assert all(q['status']=='VALIDATED_CYCLE_FOLD' for q in roots)
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(12,9));fig.subplots_adjust(left=.09,right=.97,bottom=.09,top=.94,wspace=.32,hspace=.42)
    xx=np.array([q['J_EE_core'] for q in rows]);mean=np.array([q['mean_rates_hz'] for q in rows])
    period=np.array([q['T_ms'] for q in rows]);rootrows=[]
    ax=axes[0,0]
    for k in [0,1]:ax.plot(xx,mean[:,k],color=COL[k],lw=1.8,label=f'Core {"AB"[k]}')
    ax.set(xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Period-mean rate (Hz / cell)',title='A  Continuous periodic branch');ax.legend(frameon=False)
    ax=axes[0,1];ax.plot(xx,period,color='#1b9e77',lw=1.8)
    ax.set(xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Period (ms)',title='B  Period along the same branch')
    for i,q in enumerate(roots):
        meta=q['mesh_checks'][-1];z=np.load(meta['orbit']);r=z['r'];T=float(z['T']);J=float(z['J'])
        reg=np.array([s.regional_rates(v) for v in r]);label=f'LPC{61+i}'
        for ax,y in [(axes[0,0],reg.mean(0)[0]),(axes[0,1],T)]:
            ax.plot(J,y,'s',mfc='black',mec='black',ms=6,zorder=5)
            ax.annotate(label,(J,y),xytext=(8,10 if i==0 else -20),textcoords='offset points',fontsize=10)
        errors=q['full_state_fold_mode_checks'];dt=np.array([v['dt_ms'] for v in errors]);defect=np.array([v['generalized_plus_one_relative_defect'] for v in errors])
        axes[1,0].loglog(dt,defect,'o-',label=label,lw=1.4)
        # Use the same phase convention for each orbit; preserve A/B phase differences.
        shift=int(np.argmax(reg[:,0]))-len(r)//2;reg=np.roll(reg,-shift,axis=0)
        phase=np.arange(len(r))/len(r)
        for k in [0,1,2]:axes[1,1].plot(phase,reg[:,k],color=COL[k] if k<2 else '#555555',ls='-' if i==0 else '--',lw=1.3)
        rootrows.append(dict(label=label,J_EE_core=J,T_ms=T,orbit=meta['orbit'],validation=q['label']+'_validation.json',
            mean_rates_hz=reg.mean(0),generalized_plus_one_errors=defect,dt_ms=dt))
    axes[1,0].set(xlabel='Variational time step (ms)',ylabel='Generalized +1 mode relative defect',title='C  Independent full-delay mode checks')
    axes[1,0].legend(frameon=False)
    axes[1,1].set(xlabel='Cycle phase (A peak at 0.5)',ylabel='Rate (Hz / cell)',title='D  Periodic waveforms at both folds',xlim=(0,1))
    handles=[Line2D([0],[0],color=COL[k],label=f'Core {"AB"[k]}') for k in [0,1]]
    handles+=[Line2D([0],[0],color='#555555',label='Surround'),Line2D([0],[0],color='black',ls='-',label='LPC61'),Line2D([0],[0],color='black',ls='--',label='LPC62')]
    axes[1,1].legend(handles=handles,frameon=False,ncol=2,fontsize=9,loc='upper left')
    for ax in axes.ravel():style(ax)
    name='H1_periodic_fold_pair_detail';save(fig,name)
    write(PERIODIC_OUT/(name+'.json'),dict(rows=rootrows,segment_source=str(PERIODIC_OUT/'arcAconnectionStage4_accuracy.json'),
        included_branch_indices=[30,65],
        scope='Local geometry of the same H1 periodic family and independently checked generalized +1 fold directions. Period folds are not period doubling. Smooth plotted geometry does not certify stability between sampled points.'))
    update_readme({name:'放大 H1 周期分支上的 LPC61、LPC62，比较两核周期均值、周期长度，以及完整延迟系统中广义 +1 模态误差的步长收敛。右下波形仅以 A 峰对齐相位，保留各轨道的 A/B 相对时序，均来自相应已验证周期解。**关注点**：曲线沿延拓顺序连续折返；周期折叠不等于倍周期，也不能据分支线型判断稳定性。'})


if __name__=='__main__':main()
