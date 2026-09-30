"""Critical antiperiodic mode and its spatial/contact projection, all rate."""
from plot_rate_periodic_composite import *


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--internal-label',default='PD_double_upper')
    p.add_argument('--output-name',default='upper_period_doubling_spatial_mode')
    p.add_argument('--source-label')
    p.add_argument('--output-dir')
    p.add_argument('--require-accepted-parent',action='store_true')
    p.add_argument('--signed-log-readout',action='store_true',help='Show weak signed contact modes on an explicitly labelled symmetric log scale')
    a=p.parse_args()
    source_label=a.source_label or a.internal_label
    q=read(PERIODIC_OUT/f'{source_label}_mode_readout.json')
    z=np.load(PERIODIC_OUT/f'{source_label}_mode_readout.npz')
    valid=PERIODIC_OUT/f'{a.internal_label}_validation.json'
    validation=read(valid) if valid.exists() else {}
    if a.require_accepted_parent:
        assert validation.get('full_acceptance',False) and q.get('accepted_fine_parent',False)
        assert Path(q['parent_orbit']).resolve()==Path(validation['accepted_parent_orbit']).resolve()
        accepted_mode=validation.get('accepted_mode') or validation['filter_state_followup']['mode']
        assert Path(q['mode_source']).resolve()==Path(accepted_mode).resolve()
        assert abs(q['J_EE_core']-validation['J_EE_core'])<1e-12
    label=validation.get('label','PD candidate') if validation.get('status')=='VALIDATED_PD' else 'PD candidate'
    geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    order=contact_indices(geo['contact_names'].tolist());xy=geo['contact_xy'];s=RateField()
    t=z['time_ms'];T=q['T_ms'];tt=np.r_[t,t+T]
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,4,figsize=(17,4.5),gridspec_kw={'width_ratios':[1,1,1,1.15]})
    fig.subplots_adjust(left=.055,right=.985,bottom=.22,top=.80,wspace=.48)
    for k,c in enumerate([*COL,'#555555']):
        axes[0].plot(t,z['parent_regional_Hz'][:,k],color=c,lw=1.1,label=['Core A','Core B','Surround'][k])
        axes[1].plot(tt,np.r_[z['regional_mode_Hz'][:,k],-z['regional_mode_Hz'][:,k]],color=c,lw=1.1)
    axes[0].set(xlim=(0,T),xlabel='Time (ms)',ylabel='Rate (Hz / E cell)',title='A   Parent periodic orbit')
    axes[0].legend(frameon=False,fontsize=8,loc='upper left')
    axes[1].axhline(0,color='black',lw=.6);axes[1].axvline(T,color='black',lw=.6,ls='--')
    axes[1].set(xlim=(0,2*T),xlabel='Time over two parent periods (ms)',
                ylabel='Rate mode (normalized)',title='B   Opposite changes in consecutive cycles')
    for ax in axes[:2]:style(ax)
    field=z['E_cell_RMS_mode_Hz'];im=axes[2].imshow(field.reshape(20,20),origin='lower',
        extent=(0,20,0,20),cmap='magma',vmin=0)
    for i,center in enumerate(s.geo['centers_mm']):
        axes[2].add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=.8))
        axes[2].text(*center,'AB'[i],color='white',ha='center',va='center',fontsize=8)
    axes[2].scatter(xy[:,0],xy[:,1],s=9,facecolors='none',edgecolors='cyan',linewidths=.5)
    axes[2].set(xlabel='x (mm)',ylabel='y (mm)',title='C   Spatial mode RMS per E cell')
    fig.colorbar(im,ax=axes[2],orientation='horizontal',fraction=.08,pad=.18,label='Normalized mode RMS')
    contact=np.r_[z['contact_mode_Hz'],-z['contact_mode_Hz']][:,order].T;lim=np.max(abs(contact))
    if a.signed_log_readout:
        from matplotlib.colors import SymLogNorm
        lim=float(np.ceil(lim/10**np.floor(np.log10(lim)))*10**np.floor(np.log10(lim)))
        scale=dict(norm=SymLogNorm(linthresh=lim/100,vmin=-lim,vmax=lim,base=10))
    else:scale=dict(vmin=-lim,vmax=lim)
    ic=axes[3].imshow(contact,origin='upper',aspect='auto',extent=(0,2*T,14.5,-.5),cmap='RdBu_r',**scale)
    axes[3].axvline(T,color='black',lw=.6,ls='--');axes[3].axhline(3.5,color='black',lw=.6)
    axes[3].set(yticks=np.arange(15),yticklabels=CONTACT_ORDER,xlabel='Time (ms)',title='D   SEEG-site rate mode')
    axes[3].tick_params(axis='y',labelsize=7,length=2)
    for tick,name in zip(axes[3].get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[name[:3]])
    bar=fig.colorbar(ic,ax=axes[3],orientation='horizontal',fraction=.08,pad=.18,
        label='Signed rate mode (symmetric log)' if a.signed_log_readout else 'Signed normalized rate mode')
    if a.signed_log_readout:
        ticks=[-lim,-lim/10,0,lim/10,lim];bar.set_ticks(ticks)
        bar.set_ticklabels([f'{x:g}' for x in ticks])
    suffix='  |  Already unstable parent' if validation.get('parent_stability')=='ALREADY_UNSTABLE' else ''
    fig.suptitle(label+rf'  |  $J_{{\mathrm{{EE,core}}}}={q["J_EE_core"]:.9f}$'+f'  |  Parent period {T:.2f} ms'+suffix,fontsize=14,y=.96)
    description='展示 '+label+' 的母周期波形、反周期临界模态、按每个兴奋性神经元归一化的空间模态强度和同一模态的触点投影。模态最大 E 群体分量归一为 1，其振幅任意，B/D 中正负变化是线性扰动，不能作为自由运行轨迹或实际两倍周期事件。**关注点**：模态在一个母周期后反号；子分支稳定性与传播顺序变化须另由非线性周期解和 Floquet 计算判断。'
    if a.signed_log_readout:
        description=description.replace('**关注点**：',f'触点颜色使用对称对数尺度，线性区阈值为 {lim/100:g}；母周期和模态采用同一低活动相位起点，反周期跨界部分正确反号。**关注点**：')
    if a.output_dir:
        folder=Path(a.output_dir);folder.mkdir(parents=True,exist_ok=True)
        for ext in ['png','pdf','svg']:fig.savefig(folder/f'{a.output_name}.{ext}',dpi=210,bbox_inches='tight')
        plt.close(fig)
        readme=folder/'README.md';text=readme.read_text() if readme.exists() else ''
        import re
        text=re.sub(r'^### '+re.escape(a.output_name)+r'\.png\s*\n.*?(?=^### |\Z)','',text,flags=re.M|re.S).rstrip()
        readme.write_text(text+f'\n\n### {a.output_name}.png\n{description}\n')
    else:
        save(fig,a.output_name);update_readme({a.output_name:description})


if __name__=='__main__':main()
