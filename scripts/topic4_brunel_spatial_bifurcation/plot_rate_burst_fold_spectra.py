"""Paired-step, phase-removed spectra at the two checked burst folds."""
from plot_rate_branch_completion import *


def main():
    labels=[('LPC_Bleading_low','LPC7: B-leading burst'),('LPC_burst_low','LPC5: A-leading burst')]
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(9.6,5.4))
    fig.subplots_adjust(left=.08,right=.985,bottom=.23,top=.81,wspace=.28)
    sources=[]
    for ax,(label,title) in zip(axes,labels):
        source=DATA/'primary_folds'/f'{label}_root_spectrum_assessment.json';q=read(source)
        assert q['status']=='FOLD_WITH_REMAINING_SPECTRUM_NUMERICALLY_INSIDE'
        theta=np.linspace(0,2*np.pi,500)
        ax.plot(np.cos(theta),np.sin(theta),'--',color='black',lw=.9)
        ax.axhline(0,color='#dddddd',lw=.6,zorder=0)
        ax.axvline(0,color='#dddddd',lw=.6,zorder=0)
        for k,identity in enumerate(q['critical_mode_identity']):
            spectrum=read(Path(identity['source']));mu=values(spectrum)
            critical=identity['critical_mode_index'];other=np.arange(len(mu))!=critical
            ax.plot(mu[other].real,mu[other].imag,'o',ms=8 if k==0 else 4.5,
                mfc='white' if k==0 else '#2878b5',mec='#2878b5',mew=.85,zorder=3+k)
            ax.plot(mu[critical].real,mu[critical].imag,'s',ms=8 if k==0 else 4.5,
                mfc='white' if k==0 else '#222222',mec='#222222',mew=.85,zorder=5+k)
        other=values(dict(multipliers=q['remaining_multipliers']))
        largest=other[np.argmax(abs(other))]
        ax.annotate(rf'$|\mu|={abs(largest):.3f}$',(largest.real,largest.imag),
            xytext=(-.2,.45),fontsize=11,arrowprops=dict(arrowstyle='-',lw=.8))
        ax.annotate(r'Fold mode: $\mu\to+1$',(1,0),xytext=(-.72,-.52),
            fontsize=11,arrowprops=dict(arrowstyle='-',lw=.8))
        ax.set(xlim=(-1.12,1.12),ylim=(-1.12,1.12),aspect='equal',
            xlabel=r'Re $\mu$',ylabel=r'Im $\mu$',title=title+f'\n'+rf'$J_{{\mathrm{{EE,core}}}}={q["J_EE_core"]:.8f}$')
        ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);style(ax)
        sources.append(str(source))
    fig.suptitle('Burst folds: nontrivial Floquet spectra',fontsize=15,y=.98)
    fig.legend(handles=[
        Line2D([0],[0],marker='o',mfc='white',mec='#2878b5',ls='',label=r'$\Delta t\approx0.05$ ms'),
        Line2D([0],[0],marker='o',color='#2878b5',ls='',label=r'$\Delta t\approx0.025$ ms'),
        Line2D([0],[0],marker='s',color='#222222',ls='',label='Matched fold mode'),
        Line2D([0],[0],color='black',ls='--',label='Unit circle')],
        loc='lower center',bbox_to_anchor=(.5,.04),ncol=2,frameon=False,fontsize=10)
    name='burst_fold_full_spectra';save_new(fig,name)
    write(DATA/(name+'.json'),dict(sources=sources,
        scope='The phase mode is projected out. The fold mode is identified by full-history overlap with the independently computed critical tangent. Both time steps cover the exterior of the unit disk numerically; only the matched +1 mode is excluded when classifying the others. This is a pointwise, non-rigorous numerical spectrum, not interval completeness.'))
    path=OUTPUT/'figures/README.md';text=path.read_text()
    text=re.sub(r'^### '+name+r'\.png\s*\n.*?(?=^### |\Z)','',text,flags=re.M|re.S).rstrip()
    text+='\n\n### '+name+'.png\n显示 LPC7 和 LPC5 在两档时间步长下的非平凡 Floquet 谱；自主相位模态已去除，黑色方块通过完整状态与延迟历史中的特征向量匹配确认是折叠模态。其余已覆盖乘子均在单位圆内，并与已检查的两侧稳定／不稳定周期解相对应。**关注点**：这是折点处的数值谱与局部解释，不是整条分支或全部参数区间的完备性证明。\n'
    path.write_text(text)


if __name__=='__main__':main()
