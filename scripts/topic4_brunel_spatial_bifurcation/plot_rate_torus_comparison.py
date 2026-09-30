"""Two different local torus bifurcations on the folded weak-cycle branch."""
from plot_rate_periodic_completion import *


def main():
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    first=read(PERIODIC_OUT/'TR_A_B_validation.json');second=read(PERIODIC_OUT/'TR_A_return_nonlinear_validation.json')
    fig=plt.figure(figsize=(10,9));gs=fig.add_gridspec(2,2,left=.10,right=.97,bottom=.09,top=.95,hspace=.4,wspace=.4)
    ax=fig.add_subplot(gs[0,:]);rr=[read(f) for f in sorted((PERIODIC_OUT/'orbits').glob('resonanceA_*.json'))]
    ax.plot([q['J_EE_core'] for q in rr],[q['mean_rates_hz'][1] for q in rr],color=FAMILY['A'],lw=1.6)
    names=[('TR_A_B','TR1',(-36,15)),('LPC_resonance_upper','LPC1',(12,8)),('LPC_resonance_lower','LPC2',(-52,-4)),('TR_A_return','TR2',(26,-12))]
    for name,label,off in names:
        q=next(q for q in critical() if q['label']==name);y=read(Path(q['orbit']).with_suffix('.json'))['mean_rates_hz'][1]
        ax.plot(q['J_EE_core'],y,'D' if name.startswith('TR') else 's',mfc='white',mec='black',ms=6)
        ax.annotate(label,(q['J_EE_core'],y),xytext=off,textcoords='offset points',arrowprops=dict(arrowstyle='-',lw=.6),fontsize=11)
    h2=read(RATE_OUT/'hopfs.json')['rows'][1]['J_EE_core'];ax.axvline(h2,color=COL[1],ls=':',lw=.8)
    ax.text(h2,.7481,'H2',color=COL[1],ha='center')
    ax.set(xlim=(.94568,.94588),ylim=(.743,.7485),xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Core B period mean (Hz / cell)',title='A  Two torus bifurcations on the folded H1 branch')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.5f'));style(ax)
    for i,(v,scale,col,ls,kind) in enumerate([(first,1e8,'#c44e52','--','Subcritical'),(second,1e7,'#2166ac','-','Supercritical')]):
        root=v['critical_point'];J0=root['J_EE_core'];rr=sorted([q for q in v['torus_solutions'] if [q['N_theta'],q['N_psi']]==[64,16] and q['amplitude_hz']<=.02],key=lambda q:q['amplitude_hz'])
        x=np.r_[0,[(q['J_EE_core']-J0)*scale for q in rr]];y=np.r_[0,[q['amplitude_hz'] for q in rr]]
        fit=next(q for q in v['fits'] if q['mesh']==[64,16] and q.get('maximum_amplitude_Hz',.02)==.02)
        aa=np.linspace(0,max(y),180);jj=(fit['quadratic_J_per_Hz2']*aa**2+fit['quartic_J_per_Hz4']*aa**4)*scale
        a=fig.add_subplot(gs[1,i]);a.plot(jj,aa,ls,color=col,lw=1.8);a.plot(x[1:],y[1:],'o',color=col,ms=4)
        a.plot(0,0,'D',mfc='white',mec='black',ms=6)
        span=max(abs(x));left=min(min(x)*1.1,-span*.2);right=max(max(x)*1.1,span*.2)
        a.plot([left,0],[0,0],color='black',lw=1.4);a.plot([0,right],[0,0],'--',color='black',lw=1.4)
        a.set(xlim=(left,right),ylim=(-.001,.022),xlabel=rf'$(J-J_{{\mathrm{{TR{i+1}}}}})\times10^{{{int(np.log10(scale))}}}$',
            ylabel='Mode projection amplitude (Hz)',title=f'{"BC"[i]}  TR{i+1}: {kind.lower()}')
        stability='Locally stable' if second['parent_noncritical_modes_numerically_stable'] else 'Radially stable'
        a.text(.04,.96,'Radially unstable' if i==0 else stability,ha='left',va='top',transform=a.transAxes,color=col,fontsize=10)
        style(a)
    save(fig,'two_torus_bifurcations')
    update_readme(dict(two_torus_bifurcations='上图在 H1 出发的同一周期曲线上标明 TR1、两个周期折叠及新确认的 TR2。下图实点为全部空间群体参与的双角度环面边值解，曲线为已验证振幅范围内的二次和四次项局部拟合；TR1 局部亚临界，TR2 局部超临界。TR2 的其他父周期模态经双步长完整状态返回谱及滤波覆盖检查后支持稳定，结合超临界分支曲率支持新生环面的局部稳定。**关注点**：这些是弱活动双频分支，不能直接称为 irregular burst；局部稳定性不等于较大振幅上的全局稳定性延拓。'))


if __name__=='__main__':main()
