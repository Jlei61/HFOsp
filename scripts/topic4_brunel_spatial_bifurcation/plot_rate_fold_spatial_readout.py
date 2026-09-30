"""Independently checked fold directions in the spatial/contact projection."""
from plot_rate_periodic_composite import *
from validate_rate_mean_fold import validation_matches_latest_root


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labels',nargs='+',default=['LPC_A1','LPC_B1','LPC_A_connection1','LPC_A_connection2'])
    p.add_argument('--output',default='cycle_fold_spatial_contact_modes')
    p.add_argument('--output-dir',help='Optional figure folder; numerical mode records retain their original result location')
    args=p.parse_args();labels=args.labels
    if args.output_dir:
        import plot_rate_periodic_completion as producer
        producer.F=Path(args.output_dir)
    s=RateField();weights=s.geo['group_size']/s.geo['group_size'].sum()
    eweights=weights*s.E;region=s.geo['group_region'];cell=s.geo['group_cell'];size=s.geo['group_size']
    fractions=np.array([eweights[region==k].sum()/eweights.sum() for k in range(3)])
    counts=np.bincount(cell[s.E],weights=size[s.E],minlength=400)
    geometry=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    names=geometry['contact_names'].tolist();order=contact_indices(names);xy=geometry['contact_xy'];rows=[]
    for name in labels:
        source=PERIODIC_OUT/(name+'_validation.json');check=read(source)
        assert validation_matches_latest_root(name), 'Validation is not for the current root: '+name
        physical=check['continuous_defect']
        assert physical['filter_state_check']['positive'], 'Unresolved filter profile: '+name
        assert physical['maximum_group_defect_Hz']<.001
        meta=check['mesh_checks'][-1];z=np.load(meta['orbit']);r=z['r'];N=len(r)
        assert r.shape[1]==935 and N==meta['N']
        assert Path(physical['orbit']).resolve()==Path(meta['orbit']).resolve()
        v=np.load(PERIODIC_OUT/f'{name}_tangent_N{N}.npz')['tangent'][:-2].reshape(r.shape)
        phase=np.fft.ifft(2j*np.pi*(np.fft.fftfreq(N)*N)[:,None]*np.fft.fft(r*1000,axis=0),axis=0).real
        v=v-np.sum(v*phase*weights)/np.sum(phase**2*weights)*phase
        projection_defect=abs(np.sum(v*phase*weights))/np.sqrt(np.sum(v*v*weights)*np.sum(phase*phase*weights))
        assert projection_defect<1e-10
        energy=np.mean(v**2,axis=0)*eweights
        shares=np.array([energy[region==k].sum()/energy.sum() for k in range(3)])
        assert abs(shares.sum()-1)<1e-12
        scale=np.sqrt(energy.sum()/eweights.sum());v/=scale
        field=np.sqrt(np.bincount(cell[s.E],weights=size[s.E]*np.mean(v[:,s.E]**2,axis=0),minlength=400)/np.maximum(counts,1))
        contacts=np.sqrt(np.mean((v@s.geo['contact_rate_weights'])**2,axis=0))
        rows.append(dict(label=CRITICAL_LABELS[name],internal_label=name,source=str(source),orbit=meta['orbit'],
            J_EE_core=check['J_EE_core'],T_ms=check['T_ms'],E_mode_energy_fraction_A_B_surround=shares,
            E_population_fraction_A_B_surround=fractions,per_E_cell_mode_RMS_over_network_RMS=np.sqrt(shares/fractions),
            normalized_E_field_RMS=field,contact_names=names,normalized_contact_RMS=contacts,
            common_phase_projection_relative_defect=float(projection_defect)))
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,len(labels),figsize=(4.125*len(labels),8.2),squeeze=False,gridspec_kw={'height_ratios':[1,1]})
    fig.subplots_adjust(left=.065,right=.935,top=.87,bottom=.19,wspace=.32,hspace=.43)
    vmax=max(max(q['normalized_E_field_RMS']) for q in rows)
    contactmax=max(max(q['normalized_contact_RMS']) for q in rows)
    for col,q in enumerate(rows):
        ax=axes[0,col];im=ax.imshow(q['normalized_E_field_RMS'].reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0,vmax=vmax)
        for i,center in enumerate(s.geo['centers_mm']):
            ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=1))
            ax.text(*center,'AB'[i],color='white',ha='center',va='center')
        ax.scatter(xy[:,0],xy[:,1],s=12,facecolors='none',edgecolors='cyan',linewidths=.55)
        ax.set(xlabel='x (mm)',ylabel='y (mm)' if col==0 else '',
            title=q['label']+rf'  |  $J_{{\mathrm{{EE,core}}}}={q["J_EE_core"]:.6f}$')
        ax=axes[1,col]
        ax.bar(np.arange(15),q['normalized_contact_RMS'][order],
            color=[SHAFT_COLORS[n[:3]] for n in CONTACT_ORDER],width=.8)
        ax.axvline(3.5,color='black',lw=.6)
        ax.set(xticks=np.arange(15),xticklabels=CONTACT_ORDER,ylim=(0,contactmax*1.08),
            ylabel='Contact-mode RMS / network E RMS' if col==0 else '',title='SEEG-site projection')
        ax.set_yscale('symlog',linthresh=.001)
        ticks=[v for v in [0,.001,.01,.1,1,5] if v<=contactmax*1.08]
        ax.set_yticks(ticks);ax.set_yticklabels([f'{v:g}' for v in ticks])
        ax.set_ylim(0,contactmax*1.08)
        ax.tick_params(axis='x',rotation=90,labelsize=8)
        for tick,n in zip(ax.get_xticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[n[:3]])
        style(ax)
    fig.colorbar(im,cax=fig.add_axes([.95,.555,.012,.285]),label='E-cell mode RMS / network E RMS')
    fig.suptitle('Cycle-fold shape directions: spatial field and contact projection',fontsize=15,y=.97)
    save(fig,args.output)
    write(PERIODIC_OUT/(args.output+'.json'),dict(rows=rows,
        definition='Remove one common temporal phase with E/I neuron weights; normalize each rate-shape tangent by its neuron-weighted network E RMS. Spatial panels and E energy statistics include E groups only. Contact projection uses the unchanged full readout operator.',
        scope='Critical directions of periodic solutions, not spontaneous propagation snapshots, voltage, event participation, or proof of stable-attractor switching.'))
    update_readme({args.output:f'比较 {len(labels)} 个已通过独立临界模态检查的周期折叠，显示去除共同相位后的 E 群体空间形变以及同一模态在接触点上的投影。每个模态均按全网 E 神经元加权 RMS 归一化，所有空间图共享色标，接触投影共享 symlog 纵轴（0.001 以下线性）。**关注点**：这些是周期轨道变化的临界方向，不是自主传播快照或 SEEG 电压；区域能量占比与单神经元强度分别保存在数值结果中。'})
    for q in rows:print(q['label'],'regional per-E-cell RMS',q['per_E_cell_mode_RMS_over_network_RMS'],flush=True)


if __name__=='__main__':main()
