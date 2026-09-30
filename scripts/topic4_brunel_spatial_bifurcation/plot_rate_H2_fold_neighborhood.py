"""Resolved fold geometry and spectra at two sides of the same H2 fold."""
from plot_rate_branch_completion import *


def main():
    folder=DATA/'H2_fold_neighborhood'
    paths=[folder/f'offset_{x:+.5f}_refined.json' for x in [-.001,.001]]
    rows=[read(p) for p in paths]
    assert [q['classification']['numerical_unstable_dimension'] for q in rows]==[3,4]
    root=read(PERIODIC_OUT/'LPC_B2_validation.json')['mesh_checks'][-1]
    checked=read(folder/'dense_geometry_checks.json');assert checked['status']=='PASS'
    geometry=[]
    for q in checked['rows']:
        z=np.load(q['orbit']);s=RateField() if not geometry else s
        w=s.geo['group_size']*s.E*(s.geo['group_region']==1);w/=w.sum()
        assert q['filter_state_check']['positive']
        geometry.append((float(z['r'].mean(0)@w*1000),q['J_EE_core']))
    geometry=sorted(geometry+[(root['coordinate_value'],root['J_EE_core'])])
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(12.8,4.7),gridspec_kw={'width_ratios':[1.15,1,1.1]})
    fig.subplots_adjust(left=.07,right=.985,top=.8,bottom=.25,wspace=.36)
    colors=['#2878b5','#c74e39']
    ax=axes[0]
    ax.plot([q[1] for q in geometry],[q[0] for q in geometry],ls=LINESTYLE,color=FAMILY['B'],lw=1.5)
    ax.plot(root['J_EE_core'],root['coordinate_value'],'s',ms=6,color='black')
    ax.annotate('LPC13',(root['J_EE_core'],root['coordinate_value']),xytext=(20,0),
        textcoords='offset points',fontsize=10,arrowprops=dict(arrowstyle='-',lw=.8))
    for q,c in zip(rows,colors):
        ax.plot(q['J_EE_core'],q['coordinate_value'],'x',ms=8,color=c,mew=1.6)
    ax.set(xlim=(.9463098,.9463165),ylim=(1.2088,1.2158),
        xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Core B period mean (Hz / E cell)',
        title='A  Two sides along the cycle branch')
    ax.set_xticks([.946310,.946313,.946316]);ax.xaxis.set_major_formatter(FormatStrFormatter('%.6f'))
    ax.set_yticks([1.209,1.212,1.215]);style(ax)
    ax=axes[1];theta=np.linspace(0,2*np.pi,500)
    ax.plot(np.cos(theta),np.sin(theta),'--',color='black',lw=.9)
    ax.axhline(0,color='#dddddd',lw=.6);ax.axvline(0,color='#dddddd',lw=.6)
    for q,c,marker in zip(rows,colors,['o','^']):
        mu=values(q['classification']);visible=(abs(mu.real)<=1.6)&(abs(mu.imag)<=1.2)
        ax.plot(mu[visible].real,mu[visible].imag,marker,color=c,ms=6,ls='')
    ax.set(xlim=(-1.6,1.6),ylim=(-1.2,1.2),aspect='equal',xlabel=r'Re $\mu$',ylabel=r'Im $\mu$',
        title='B  Near the unit circle')
    ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);style(ax)
    ax=axes[2];ax.axhline(1,color='black',ls='--',lw=.9)
    for k,(q,c,marker) in enumerate(zip(rows,colors,['o','^'])):
        mu=values(q['classification'])
        ax.plot(np.arange(1,len(mu)+1)+(k-.5)*.16,abs(mu),marker,ls='',color=c,ms=6)
    ax.set(yscale='log',ylim=(.002,100),xticks=range(1,7),xlabel='Sorted mode index at each sample',
           ylabel=r'$|\mu|$',title='C  All returned modes')
    ax.set_yticks([.01,.1,1,10,100]);ax.set_yticklabels(['0.01','0.1','1','10','100']);style(ax)
    fig.suptitle('LPC13: both nearby cycle branches are unstable\n'+r'$J_{\mathrm{EE,core}}=0.94631012$',fontsize=14,y=.98)
    fig.legend(handles=[Line2D([0],[0],marker='o',color=colors[0],ls='',
        label='Lower Core B mean: 3 unstable directions'),
        Line2D([0],[0],marker='^',color=colors[1],ls='',label='Higher Core B mean: 4 unstable directions'),
        Line2D([0],[0],marker='s',color='black',ls='',label='Verified cycle fold')],
        loc='lower center',bbox_to_anchor=(.5,.035),ncol=2,frameon=False,fontsize=9)
    name='H2_fold_two_sided_spectra';save_new(fig,name)
    write(DATA/(name+'.json'),dict(sources=[str(p) for p in paths],
        fold_validation=str(PERIODIC_OUT/'LPC_B2_validation.json'),
        geometry_source=str(PERIODIC_OUT/'LPC_B2_neighborhood.json'),
        checked_geometry_source=str(folder/'dense_geometry_checks.json'),
        actual_periodic_BVP_geometry_points=len(geometry),
        numerical_unstable_dimensions=[3,4],
        scope='Two physically checked neighboring cycles, paired valid time steps, numerical exterior-spectrum coverage, and direct eigenpair residual checks. No extrapolation of counts to the intervening interval or additional PD inferred from negative multipliers.'))
    path=OUTPUT/'figures/README.md';body=path.read_text()
    body=re.sub(r'^### '+name+r'\.png\s*\n.*?(?=^### |\Z)','',body,flags=re.M|re.S).rstrip()
    body+='\n\n### '+name+'.png\n显示 LPC13 的局部折返，以及沿周期分支两侧的已检查样本；两者 J 几乎相同，Core B 周期均值不同。配对时间步和独立特征向量传播检查给出两侧分别 3、4 个数值不稳定方向，原始低精度特征向量及后续子空间修正均保留。**关注点**：两侧都不稳定；负实乘子的出现不自动等于新增倍周期，样本间的模态连接仍需加密追踪。\n'
    path.write_text(body)


if __name__=='__main__':main()
