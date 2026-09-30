"""A resolved H2 cycle fold followed by a distinct verified flip."""
from plot_rate_branch_completion import *


def main():
    pd=read(PERIODIC_OUT/'PD_H2_after_LPC13_validation.json')
    assert pd['status']=='VALIDATED_PD' and pd['full_acceptance']
    assert pd['continuous_orbit_check']['filter_state_check']['positive']
    childfile=PERIODIC_OUT/'PD_H2_after_LPC13_child_validation.json'
    child=read(childfile) if childfile.exists() else {}
    child_checked=child.get('status')=='LOCALLY_CHECKED_PD_CHILD' and child.get('full_physical_child_checks',False)
    fold=read(PERIODIC_OUT/'LPC_B2_validation.json')['mesh_checks'][-1]
    profiles=read(DATA/'H2_fold_neighborhood/dense_geometry_checks.json')
    assert profiles['status']=='PASS'
    s=RateField();w=s.geo['group_size']*s.E*(s.geo['group_region']==1);w/=w.sum()
    center=fold['J_EE_core'];geometry=[]
    for row in profiles['rows']:
        z=np.load(row['orbit'])
        geometry.append((float(z['r'].mean(0)@w*1000),row['J_EE_core']))
    geometry+= [(fold['coordinate_value'],center),
                (pd['continuous_orbit_check']['core_B_mean_Hz'],pd['J_EE_core'])]
    geometry=sorted(geometry)
    before=read(DATA/'H2_fold_neighborhood/offset_+0.00100_refined.json')
    after=read(DATA/'H2_fold_neighborhood/offset_+0.00300.json')
    witnesses=[before,after]
    assert [q['classification']['numerical_unstable_dimension'] for q in witnesses]==[4,3]
    mu=[]
    for q in witnesses:
        v=values(q['classification']);negative=np.flatnonzero((abs(v.imag)<1e-10)&(v.real<0))
        i=negative[np.argmin(abs(v[negative]+1))]
        assert q['classification']['reliable_mode_mask'][int(i)]
        mu.append(v[i].real)
    assert mu[0]<-1<mu[1]<0
    checks=sorted(pd['direct_monodromy_checks'],key=lambda q:-q['dt_ms'])
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(13.6,4.8),gridspec_kw={'width_ratios':[1.2,1.1,1]})
    fig.subplots_adjust(left=.06,right=.98,bottom=.25,top=.8,wspace=.5)
    ax=axes[0]
    ax.plot([(q[1]-center)*1e6 for q in geometry],[q[0] for q in geometry],
            color=FAMILY['B'],ls=LINESTYLE,lw=1.7)
    ax.plot(0,fold['coordinate_value'],'s',color='black',ms=6)
    x=(pd['J_EE_core']-center)*1e6;y=pd['continuous_orbit_check']['core_B_mean_Hz']
    ax.plot(x,y,'v',color='black',ms=7)
    ax.annotate('LPC13',(0,fold['coordinate_value']),xytext=(18,-20),textcoords='offset points',
                arrowprops=dict(arrowstyle='-',lw=.7))
    ax.annotate('PD4',(x,y),xytext=(15,20),textcoords='offset points',
                arrowprops=dict(arrowstyle='-',lw=.7))
    colors=['#c74e39','#2878b5']
    for q,c in zip(witnesses,colors):
        ax.plot((q['J_EE_core']-center)*1e6,q['coordinate_value'],'x',ms=7,color=c,mew=1.5)
    ax.set(xlim=(-.35,6.3),ylim=(1.2088,1.216),xticks=[0,2,4,6],
        xlabel=rf'$10^6\,(J_{{\mathrm{{EE,core}}}}-{center:.8f})$',
        ylabel='Core B period mean (Hz / E cell)',title='A  Fold and period doubling are distinct')
    ax=axes[1];ax.axvline(-1,color='black',ls='--',lw=1)
    ax.plot(mu[0],0,'o',color=colors[0],ms=7)
    ax.plot(checks[-1]['recovered_multiplier'],1,'v',color='black',ms=7)
    ax.plot(mu[1],2,'o',color=colors[1],ms=7)
    ax.set(xlim=(-1.5,.1),ylim=(-.45,2.45),xticks=[-1.5,-1,-.5,0],
           yticks=[0,1,2],yticklabels=['Before PD4','PD4','After PD4'],
           xlabel=r'Real multiplier $\mu$',title=r'B  Negative mode crosses $-1$')
    ax=axes[2];dt=np.array([q['dt_ms'] for q in checks]);err=np.array([q['minus_one_relative_defect'] for q in checks])
    ax.loglog(dt,err,'o-',color='#2878b5',ms=5)
    ax.loglog(dt,err[-1]*(dt/dt[-1])**2,'--',color='black',lw=.9,label=r'$\Delta t^2$ reference')
    ax.set(xlabel=r'Integration step $\Delta t$ (ms)',ylabel=r'$\|Mv+v\|/\|v\|$',
           title='C  Independent full-state propagation')
    ax.set_xticks(dt);ax.set_xticklabels([f'{v:.5f}'.rstrip('0') for v in dt]);ax.minorticks_off()
    ax.legend(frameon=False,fontsize=9)
    for ax in axes:style(ax)
    fig.suptitle('H2: LPC13 followed by PD4\n'+rf'PD4 at $J_{{\mathrm{{EE,core}}}}={pd["J_EE_core"]:.11f}$',fontsize=14,y=.98)
    fig.legend(handles=[Line2D([0],[0],marker='s',color='black',ls='',label='Verified cycle fold'),
        Line2D([0],[0],marker='v',color='black',ls='',label='Verified period doubling'),
        Line2D([0],[0],marker='o',color=colors[0],ls='',label='Before: 4 unstable directions'),
        Line2D([0],[0],marker='o',color=colors[1],ls='',label='After: 3 unstable directions')],
        loc='lower center',bbox_to_anchor=(.5,.015),ncol=2,frameon=False,fontsize=9)
    name='H2_fold_to_PD4';save_new(fig,name)
    write(DATA/(name+'.json'),dict(validation=str(PERIODIC_OUT/'PD_H2_after_LPC13_validation.json'),
        fold_J=center,PD_J=pd['J_EE_core'],J_separation=pd['J_EE_core']-center,
        negative_multiplier_witnesses=mu,neighbor_numerical_unstable_dimensions=[4,3],
        scope='Same H2 branch, verified fold and antiperiodic point, paired-step neighboring spectra and independent full-delay critical-mode propagation. Both neighboring cycles are unstable. Child criticality and propagation-template changes are not inferred.'))
    p=OUTPUT/'figures/README.md';body=p.read_text()
    body=re.sub(r'^### '+name+r'\.png\s*\n.*?(?=^### |\Z)','',body,flags=re.M|re.S).rstrip()
    note=('真实子轨道的独立验证见 H2_PD4_nonlinear_child：局部亚临界，子轨道仍不稳定；尚不支持传播模板切换。'
          if child_checked and child.get('criticality')=='SUBCRITICAL_PD' else
          '超／亚临界分类、子分支稳定性和传播模板切换仍需各自证据。')
    body+='\n\n### '+name+'.png\n在同一 H2 周期分支上分开显示 LPC13 和 PD4：左侧为已检查的周期均值几何，中间为 PD4 两侧的相关负实乘子，右侧是独立完整状态及延迟历史传播的步长收敛。两侧周期解分别有 4、3 个数值不稳定方向，均不能作为稳定吸引子。**关注点**：'+note+'\n'
    p.write_text(body)


if __name__=='__main__':main()
