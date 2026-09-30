"""Paired full-history spectrum and the growing mode of the physical PD1 child."""
from plot_rate_branch_completion import *
from complete_rate_positive_stability import paired_modes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--with-parent',action='store_true')
    args=parser.parse_args()
    source=DATA/'physical_children/PD_double_low/result.json'
    q=read(source)
    assert q['full_physical_child_checks']
    child=q['rows'][-1]
    assert child['physical_pass'] and child['physical_check']['maximum_group_defect_Hz']<.001
    last=q['attempts'][-1];spectra=last['spectra'];verdict=paired_modes(*spectra)
    assert verdict['status']=='UNSTABLE' and verdict['section_projection_checked']
    assert all(Path(v['orbit']).resolve()==Path(child['orbit']).resolve() for v in spectra)
    fine=values(spectra[-1]);reliable=np.asarray(verdict['reliable_mode_mask'],bool)
    growing=np.flatnonzero(np.asarray(verdict['outside_unit_disk_mask'],bool))
    assert len(growing)==1
    index=int(growing[0]);mu=fine[index]
    assert reliable[index] and abs(mu.imag)<1e-10
    assert mu.real>1+verdict['per_mode_margin'][index]
    s=RateField();mass=s.geo['group_size'];E=s.E;regions=s.geo['group_region']
    modes=[];mode_sources=[];selected=[]
    for spectrum,dt in zip(spectra,last['steps_ms']):
        path=PERIODIC_OUT/'poincare_floquet'/f'PD1_physical_20260920_child_k6_dt{dt:g}.npz'
        z=np.load(path);eigenvalues=values(spectrum)
        assert np.max(abs(z['multipliers']-eigenvalues))<1e-10
        k=int(np.argmin(abs(eigenvalues-mu)))
        vector=z['local_vectors'][:,k]
        assert vector.shape==(9*935,) and np.linalg.norm(vector.imag)<1e-8*np.linalg.norm(vector.real)
        local=vector.real.reshape(9,935)
        dr=s.alpha*local[0]+(1-s.alpha)*local[1]
        sign=np.sign(dr[np.flatnonzero(E)[np.argmax(abs(dr[E]))]])
        dr=dr*sign/np.max(abs(dr[E]))
        modes.append(dr);mode_sources.append(str(path));selected.append(eigenvalues[k])
    weights=mass*E;weights=weights/weights.sum()
    cosine=float(np.dot(weights*modes[0],modes[1])/
        np.sqrt(np.dot(weights,modes[0]**2)*np.dot(weights,modes[1]**2)))
    cell=s.geo['group_cell'];counts=np.bincount(cell[E],weights=mass[E],minlength=400)
    field=np.bincount(cell[E],weights=mass[E]*modes[-1][E],minlength=400)/np.maximum(counts,1)
    energy=np.array([np.sum(mass[E&(regions==k)]*modes[-1][E&(regions==k)]**2) for k in range(3)])
    energy/=energy.sum()
    geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    if args.with_parent:
        fig,grid=plt.subplots(2,2,figsize=(10.8,9.6));axes=grid.ravel()
        fig.subplots_adjust(left=.10,right=.96,bottom=.08,top=.86,wspace=.38,hspace=.59)
    else:
        fig,axes=plt.subplots(1,3,figsize=(13.8,4.8))
        fig.subplots_adjust(left=.06,right=.98,bottom=.19,top=.78,wspace=.36)
    colors=['#d2691e','#2166ac'];theta=np.linspace(0,2*np.pi,600)
    ax=axes[0];ax.plot(np.cos(theta),np.sin(theta),'--',color='black',lw=1)
    for spectrum,color,dt,marker in zip(spectra,colors,last['steps_ms'],['+','o']):
        vv=values(spectrum)
        ax.plot(vv.real,vv.imag,marker,ms=8 if marker=='+' else 6,
            mfc='none',mec=color,label=f'dt = {dt:g} ms')
    ax.set(xlim=(-1.15,1.15),ylim=(-1.15,1.15),aspect='equal',
        xlabel=r'Re $\mu$',ylabel=r'Im $\mu$',title='A   Returned nonphase spectrum')
    ax.legend(frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2);style(ax)
    ax=axes[1]
    values_pair=np.array([v.real for v in selected]);dt=np.array(last['steps_ms'])
    ax.plot(dt,values_pair,'-',color='#555555',lw=1)
    for x,y,color in zip(dt,values_pair,colors):
        ax.plot(x,y,'o',color=color,ms=7)
        ax.annotate(f'{y:.6f}',(x,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
    ax.axhline(1,color='black',ls='--',lw=1)
    ax.set(xlim=(.018,.057),ylim=(.997,1.024),xticks=sorted(dt),
        xlabel='Integration step (ms)',ylabel=r'Growing multiplier $\mu$',
        title='B   Growth persists on the finer step');style(ax)
    ax=axes[2];lim=max(float(abs(field).max()),1e-12)
    im=ax.imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),
        cmap='RdBu_r',vmin=-lim,vmax=lim)
    for k,center in enumerate(s.geo['centers_mm']):
        ax.add_patch(plt.Circle(center,1.5,fill=False,color='black',lw=.9))
        ax.text(*center,'AB'[k],ha='center',va='center',fontsize=9)
    ax.scatter(geo['contact_xy'][:,0],geo['contact_xy'][:,1],s=12,facecolors='none',edgecolors='#00a6b2',linewidths=.8)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',title='C   Rate mode at reference phase')
    fig.colorbar(im,ax=ax,pad=.04,shrink=.85,label='Relative E-rate perturbation')
    parent_rows=[]
    if args.with_parent:
        from check_rate_PD1_parent_display import verified_parent_witnesses
        parent_rows=verified_parent_witnesses();assert len(parent_rows)==2
        root=read(PERIODIC_OUT/'PD_double_low_validation.json');critical_J=root['J_EE_core']
        ax=axes[3];ax.axhline(-1,color='black',ls='--',lw=1)
        ax.axvline(0,color='#555555',ls=':',lw=.8)
        for row in parent_rows:
            vv=values(row['classification']);i=int(np.argmin(abs(vv+1)))
            assert abs(vv[i].imag)<1e-10 and row['classification']['reliable_mode_mask'][i]
            stable=row['status']=='NUMERICALLY_STABLE'
            x=1e6*(row['J_EE_core']-critical_J);y=vv[i].real
            ax.plot(x,y,'o' if stable else 'x',color='#2166ac' if stable else '#d2691e',ms=7,mew=1.5)
            ax.annotate('Stable parent' if stable else 'Unstable parent',(x,y),
                xytext=(-8,-18) if stable else (7,9),textcoords='offset points',
                ha='right' if stable else 'left',fontsize=9)
        ax.plot(0,-1,'v',color='black',ms=6)
        ax.annotate('PD1',(0,-1),xytext=(-9,-18),textcoords='offset points',ha='right',fontsize=9)
        ax.set(xlim=(-16,6),ylim=(-1.066,-.982),xlabel=r'$10^6(J_{\mathrm{EE,core}}-J_{\mathrm{PD1}})$',
            ylabel=r'Parent multiplier $\mu$',title=r'D   Parent crossing at $\mu=-1$')
        style(ax)
    heading=('PD1: parent stability and doubled-child instability' if args.with_parent else
             'PD1 doubled child: verified instability')
    fig.suptitle(heading+'\n'+('Displayed child: ' if args.with_parent else '')+
        rf'$J_{{\mathrm{{EE,core}}}}={child["J_EE_core"]:.9f}$'+
        f' | Full period {child["T_ms"]:.2f} ms',fontsize=14,y=.98)
    name='PD1_parent_and_child_stability' if args.with_parent else 'PD1_child_full_history_instability'
    save_new(fig,name)
    write(DATA/(name+'.json'),dict(status='PHYSICAL_CHILD_UNSTABLE',source=str(source),
        orbit=child['orbit'],J_EE_core=child['J_EE_core'],T_ms=child['T_ms'],
        paired_classification=verdict,growing_multipliers=[[float(v.real),float(v.imag)] for v in selected],
        requested_steps_ms=last['steps_ms'],mode_sources=mode_sources,
        this_mode_growth_exponent_per_s=[float(np.log(abs(v))/(child['T_ms']/1000)) for v in selected],
        this_mode_linear_e_folding_time_s=[float((child['T_ms']/1000)/np.log(abs(v))) for v in selected],
        growth_time_scope='Linear growth along this returned mode near this exact cycle, not an observed transient lifetime or a claim that all unstable directions grow this slowly.',
        growing_mode_full_state_residuals=[v['full_state_generalized_eigen_residuals'][int(np.argmin(abs(values(v)-mu)))] for v in spectra],
        reference_phase_E_rate_mode_cosine_between_steps=cosine,
        exact_parent_witnesses=parent_rows,
        E_mode_energy_fractions_A_B_surround_at_reference_phase=energy,
        mode_normalization='Sign fixed by largest E component; max absolute population E-rate perturbation equals one. Neuron-count-weighted spatial E projection.',
        scope='One physical doubled child. At least one growing Floquet direction; exact unstable dimension remains unresolved. The displayed eigenvector is phase-dependent and is not a spontaneous propagation snapshot or causal contribution. Included parent witnesses retain their own exact parameters. Local PD criticality additionally requires the separately checked radial-mode correspondence.'))
    file=OUTPUT/'figures/README.md';body=file.read_text()
    body=re.sub(r'^### '+name+r'\.png\s*\n.*?(?=^### |\Z)','',body,flags=re.M|re.S).rstrip()
    parent_note=('第四格将两条通过物理与双步长谱检查的父轨道放回各自精确参数，展示非平凡乘子位于 −1 两侧；不在样本间补画稳定性曲线。' if args.with_parent else '')
    body+='\n\n### '+name+'.png\n展示 PD1 真实两倍周期子轨道的两步长完整延迟谱、增长乘子的步长复核，以及该乘子在参考相位的二维 E 率特征向量。'+parent_note+'空间模态按原 935 群体和神经元数投影，未更改方程或空间节点。**关注点**：子轨道已确认至少一个增长方向，不能称为稳定吸引子；完整局部分岔分类另有模态对应检验，模态图不是自发传播快照。\n'
    file.write_text(body)
    print('PD1 GROWING MODE',selected,'spatial cosine',cosine,'regional E energy',energy,flush=True)


if __name__=='__main__':main()
