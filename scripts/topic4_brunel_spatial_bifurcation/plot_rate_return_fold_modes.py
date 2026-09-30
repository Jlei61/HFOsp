"""Resolved new H1-family folds and their phase-independent shape changes."""
from plot_rate_periodic_completion import *


def main():
    s=RateField();labels=['LPC_A_return_exchange','LPC_A_return_recruitment'];data=[]
    weights=s.geo['group_size']/s.geo['group_size'].sum()
    regional_neuron_fraction=np.array([weights[s.geo['group_region']==k].sum() for k in range(3)])
    for name in labels:
        q=read(PERIODIC_OUT/(name+'_validation.json'));assert q['status']=='VALIDATED_CYCLE_FOLD'
        m=q['mesh_checks'][-1];z=np.load(m['orbit']);r=z['r'];N=len(r)
        v=np.load(PERIODIC_OUT/f'{name}_tangent_N{N}.npz')['tangent'][:-2].reshape(r.shape)
        phase=np.fft.ifft(2j*np.pi*(np.fft.fftfreq(N)*N)[:,None]*np.fft.fft(r*1000,axis=0),axis=0).real
        projection=np.sum(v*phase*weights)/np.sum(phase*phase*weights);v=v-projection*phase
        energy=np.mean(v*v,axis=0)*weights
        shares=[energy[s.geo['group_region']==k].sum()/energy.sum() for k in range(3)]
        cell=s.geo['group_cell'];size=s.geo['group_size'];count=np.bincount(cell[s.E],weights=size[s.E],minlength=400)
        fld=np.sqrt(np.bincount(cell[s.E],weights=size[s.E]*np.mean(v[:,s.E]**2,axis=0),minlength=400)/np.maximum(count,1));fld/=fld.max()
        data.append(dict(label=CRITICAL_LABELS[name],internal_label=name,J_EE_core=q['J_EE_core'],T_ms=q['T_ms'],
            phase_removed_rate_shape_energy_fraction_A_B_surround=shares,field=fld,
            regional_neuron_fraction_A_B_surround=regional_neuron_fraction,
            per_neuron_shape_RMS_over_network_RMS=np.sqrt(np.array(shares)/regional_neuron_fraction),
            mean_rates_hz=read(Path(m['orbit']).with_suffix('.json'))['mean_rates_hz']))
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(11,8.5),layout='constrained');rr=families()['A']
    cols=[COL[0],COL[1],'#555555']
    local=[q for q in rr if 'arcAreturnStrong_' in q['path']]
    # Insert solved critical/local waveforms in continuation order. No spline
    # is used to invent the missing branch geometry near the folds.
    for name,indices in zip(labels,[(52,55),(75,78)]):
        left,right=[next(i for i,q in enumerate(local) if f'_{k:04d}_N' in q['path']) for k in indices]
        neighborhood=PERIODIC_OUT/(name+'_neighborhood.json')
        extra=([read(Path(q['path']).with_suffix('.json')) for q in read(neighborhood)['rows']]
            if neighborhood.exists() else [read(Path(read(PERIODIC_OUT/(name+'_validation.json'))['mesh_checks'][-1]['orbit']).with_suffix('.json'))])
        lo,hi=sorted([local[left]['mean_rates_hz'][0],local[right]['mean_rates_hz'][0]])
        between=local[left+1:right]+[q for q in extra if lo<q['mean_rates_hz'][0]<hi]
        between.sort(key=lambda q:q['mean_rates_hz'][0],reverse=local[left]['mean_rates_hz'][0]>local[right]['mean_rates_hz'][0])
        local=local[:left+1]+between+local[right:]
    for k,c in enumerate(cols):
        for ax,records in zip(axes[0],[rr,local]):ax.plot([q['J_EE_core'] for q in records],[q['mean_rates_hz'][k] for q in records],color=c,lw=1.2,label=['Core A','Core B','Surround'][k])
    axes[0,0].set(xlim=(.69,1.08),yscale='log',ylim=(.05,90),title='A  Continued H1 family',xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Period mean rate (Hz / cell)')
    axes[0,1].set(xlim=(.9435,.94615),ylim=(.1,5),title='B  Two resolved return folds',xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Period mean rate (Hz / cell)')
    axes[0,1].xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    for q in data:
        axes[0,1].plot(q['J_EE_core'],q['mean_rates_hz'][0],'s',mfc='white',mec='black',ms=6)
        axes[0,1].annotate(q['label'],(q['J_EE_core'],q['mean_rates_hz'][0]),xytext=(8,8),textcoords='offset points')
    axes[0,0].legend(frameon=False,fontsize=9)
    geometry=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz');xy=geometry['contact_xy']
    for i,q in enumerate(data):
        ax=axes[1,i];im=ax.imshow(np.array(q['field']).reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0,vmax=1)
        for k,center in enumerate(s.geo['centers_mm']):
            ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=1));ax.text(*center,'AB'[k],ha='center',va='center',color='white',fontsize=10)
        ax.scatter(xy[:,0],xy[:,1],s=12,facecolors='none',edgecolors='cyan',linewidths=.6)
        ax.set(xlabel='x (mm)',ylabel='y (mm)',title=f'{"CD"[i]}  {q["label"]}: cycle-fold shape mode')
    fig.colorbar(im,ax=list(axes[1]),label='E rate-shape RMS / spatial maximum',shrink=.8)
    for ax in axes.flat:style(ax)
    save(fig,'hopf_return_fold_modes')
    write(PERIODIC_OUT/'hopf_return_fold_modes.json',dict(rows=data,
        definition='Rate-profile component of the phase-fixed cycle-fold tangent; remove one common temporal phase direction with neuron weights, then sum squared RMS over all E/I groups by region. Map shows E groups only, normalized within each fold.',
        scope='Shape change of periodic solutions at a generalized +1 fold mode; not a spontaneous activity snapshot or a complete stability classification.'))
    update_readme(dict(hopf_return_fold_modes='展示从 H1 延续的周期分支及新确认的两个返回折叠，下面给出周期折叠切向量去除共同相位后的空间形状变化。空间图只显示 E 群体并分别归一化，精化点来自实际求解的周期边值方程；区域统计同时保存神经元人数基线与每神经元 RMS。**关注点**：空间图是周期轨道形状的临界变化方向，不是活动快照，也不代表这些周期解已经稳定；外围占神经元总数 96.15%，不能把能量占比直接当作每神经元作用强度。'))


if __name__=='__main__':main()
