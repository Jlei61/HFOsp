"""Matched physical-time spatial snapshots of the accepted PD1 child profile."""
from plot_rate_branch_completion import *
from complete_rate_positive_stability import paired_modes


def main():
    source=DATA/'PD1_physical_child.json';metadata=read(source)
    s=RateField();z=np.load(metadata['orbit']);r=z['r'];T=float(z['T']);N=len(r);half=N//2
    snapshot=read(metadata['source'])
    row=next(q for q in snapshot['rows'] if q['orbit']==metadata['orbit'])
    assert row['physical_check']['filter_state_check']['positive']
    assert row['physical_check']['maximum_group_defect_Hz']<1e-6
    stability_source=Path(metadata['child_stability_source'])
    stability=read(stability_source)
    spectra=stability['attempts'][-1]['spectra']
    assert len(spectra)==2
    assert all(Path(q['orbit']).resolve()==Path(metadata['orbit']).resolve() for q in spectra)
    verdict=paired_modes(*spectra)
    assert verdict['status']=='UNSTABLE' and verdict['reliable_outside_count']>=1
    r=np.roll(r,-metadata['phase_shift_samples'],axis=0)
    regional=np.array([s.regional_rates(v) for v in r]);indices=[]
    for k in range(2):
        peaks=find_peaks(regional[:half,k],height=20,distance=half//4)[0]
        assert len(peaks)==2,(k,peaks)
        indices.extend(peaks.tolist())
    indices=sorted(indices);assert len(set(indices))==4
    cell=s.geo['group_cell'];mass=s.geo['group_size'];e=s.E
    counts=np.bincount(cell[e],weights=mass[e],minlength=400)
    fields=np.array([[np.bincount(cell[e],weights=mass[e]*r[index+offset,e]*1000,
        minlength=400)/np.maximum(counts,1) for index in indices] for offset in [0,half]])
    differences=fields[1]-fields[0];limit=float(np.max(abs(differences)))
    geometry=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    xy=geometry['contact_xy'];times=np.array(indices)*T/N
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(3,4,figsize=(12,8.8))
    fig.subplots_adjust(left=.075,right=.86,bottom=.09,top=.90,wspace=.17,hspace=.25)
    norm=LogNorm(.05,float(np.ceil(fields.max()/50)*50))
    for i in range(3):
        for j,ax in enumerate(axes[i]):
            data=fields[i,j] if i<2 else differences[j]
            opts=dict(cmap='inferno',norm=norm) if i<2 else dict(cmap='RdBu_r',vmin=-limit,vmax=limit)
            im=ax.imshow(data.reshape(20,20),origin='lower',extent=(0,20,0,20),**opts)
            ax.set_facecolor('black' if i<2 else 'white')
            if i<2:rate_image=im
            else:difference_image=im
            for k,center in enumerate(s.geo['centers_mm']):
                color='white' if i<2 else 'black'
                ax.add_patch(plt.Circle(center,1.5,fill=False,color=color,lw=.8))
                if j==0:ax.text(*center,'AB'[k],color=color,ha='center',va='center',fontsize=8)
            ax.scatter(xy[:,0],xy[:,1],s=9,facecolors='none',edgecolors='cyan',linewidths=.55)
            ax.set(xticks=[0,10,20],yticks=[0,10,20])
            if i<2:ax.set_title(f'{times[j]+i*T/2:.1f} ms',fontsize=10)
            else:ax.set_title(f'Matched phase {times[j]:.1f} ms',fontsize=10)
            if j:ax.tick_params(labelleft=False)
            if i==2:ax.set_xlabel('x (mm)')
            if j==0:ax.set_ylabel(['First half\ny (mm)','Second half\ny (mm)','Second − first\ny (mm)'][i])
    fig.colorbar(rate_image,cax=fig.add_axes([.90,.435,.017,.40]),label='E rate (Hz / cell; log scale)')
    fig.colorbar(difference_image,cax=fig.add_axes([.90,.10,.017,.20]),label='Rate difference (Hz / cell)')
    criticality=('local PD criticality pending' if metadata['criticality']=='PENDING'
                 else metadata['criticality'].replace('_',' ').lower())
    fig.suptitle('PD1: spatial propagation in consecutive halves of one doubled cycle\n'
        'Physical unstable child; '+criticality,fontsize=13,y=.985)
    name='PD1_consecutive_half_spatial_snapshots';save_new(fig,name)
    write(DATA/(name+'.json'),dict(source=str(source),orbit=metadata['orbit'],T_ms=T,
        phase_shift_samples=metadata['phase_shift_samples'],first_half_indices=indices,
        first_half_times_ms=times,second_half_times_ms=times+T/2,
        child_stability=verdict['status'],child_stability_source=str(stability_source),
        paired_spectrum_classification=verdict,local_criticality=metadata['criticality'],
        maximum_absolute_field_difference_Hz=limit,shared_rate_limits_Hz=[norm.vmin,norm.vmax],
        frame_selection='The two Core A and two Core B peaks in the first half; second-half frames use exactly the same phase plus half the full child period, with no independent realignment.',
        scope='Actual nonlinear 400-cell E-rate fields and their signed difference in Hz. This does not establish a stable attractor, criticality, or propagation-template switch.'))
    path=OUTPUT/'figures/README.md';text=path.read_text()
    text=re.sub(r'^### '+name+r'\.png\s*\n.*?(?=^### |\Z)','',text,flags=re.M|re.S).rstrip()
    text+='\n\n### '+name+'.png\n并排比较 PD1 同一真实两倍周期轨道的前、后半周期二维传播场，前两行共享放电率色标，第三行是实际 Hz 差值。四个时刻取自前半周期两核的四个峰，后半周期严格加半个完整周期，不重新对齐峰值；同一轨道的双步长完整延迟谱已确认增长方向。**关注点**：这是不稳定周期解的空间读出；场差异不自动等于传播 rank 改变，局部超／亚临界分类仍单独验收。\n'
    path.write_text(text)


if __name__=='__main__':main()
