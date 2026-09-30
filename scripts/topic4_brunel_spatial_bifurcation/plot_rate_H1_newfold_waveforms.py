"""Actual rate fields and contact readouts at three validated H1 cycle folds."""
from plot_rate_periodic_composite import *


def main():
    s=RateField();geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    xy=geo['contact_xy'];order=contact_indices(geo['contact_names'].tolist())
    cell=s.geo['group_cell'];size=s.geo['group_size']
    counts=np.bincount(cell[s.E],weights=size[s.E],minlength=400)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(17,9.3));grid=fig.add_gridspec(3,3,width_ratios=[1,2.1,1.1],
        left=.06,right=.985,top=.9,bottom=.13,wspace=.28,hspace=.65)
    rows=[]
    for i in range(3):
        name=f'LPC_A_stage4_turn{i+1}';q=read(PERIODIC_OUT/(name+'_validation.json'))
        assert q['status']=='VALIDATED_CYCLE_FOLD'
        path=q['mesh_checks'][-1]['orbit'];z=np.load(path);r=z['r'];T=float(z['T']);J=float(z['J']);N=len(r)
        check=filter_state_minima(s,r,T);assert check['positive']
        reg=np.array([s.regional_rates(v) for v in r]);shift=int(np.argmax(reg[:,0]))-N//4
        r=np.roll(r,-shift,axis=0);reg=np.roll(reg,-shift,axis=0);t=np.arange(N)*T/N
        ax=fig.add_subplot(grid[i,0])
        for k in range(3):ax.plot(t,reg[:,k],color=COL[k] if k<2 else '#555555',lw=1.3)
        ax.set(xlim=(0,T),ylim=(0,220),xlabel='Time within one period (ms)',ylabel='Rate (Hz / cell)',
            title=f'LPC{61+i}  |  '+rf'$J_{{\mathrm{{EE,core}}}}={J:.6f}$'+'\n'+f'T = {T:.2f} ms')
        style(ax)
        sub=grid[i,1].subgridspec(1,4,wspace=.12);indices=[0,N//4,N//2,3*N//4]
        for j,ix in enumerate(indices):
            ax=fig.add_subplot(sub[j]);field=np.bincount(cell[s.E],weights=size[s.E]*r[ix,s.E]*1000,minlength=400)/np.maximum(counts,1)
            im=ax.imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='inferno',norm=PowerNorm(.55,0,500))
            for k,center in enumerate(s.geo['centers_mm']):
                ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=.8))
                ax.text(*center,'AB'[k],ha='center',va='center',color='white',fontsize=7)
            ax.scatter(xy[:,0],xy[:,1],s=7,facecolors='none',edgecolors='cyan',linewidths=.5)
            ax.set(xticks=[0,20],yticks=[0,20],title=f'{t[ix]:.1f} ms');ax.tick_params(labelsize=8)
            if j:ax.tick_params(labelleft=False)
            else:ax.set_ylabel('y (mm)')
            if i==2:ax.set_xlabel('x (mm)')
        ax=fig.add_subplot(grid[i,2]);contacts=r@s.geo['contact_rate_weights']*1000
        ic=ax.imshow(contacts[:,order].T,origin='upper',aspect='auto',extent=(0,T,14.5,-.5),
            cmap='magma',norm=PowerNorm(.5,0,200))
        ax.set(yticks=np.arange(15),yticklabels=CONTACT_ORDER,xlabel='Time within one period (ms)')
        ax.axhline(3.5,color='white',lw=.6);ax.tick_params(axis='y',labelsize=7,length=2)
        for tick,label in zip(ax.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[label[:3]])
        rows.append(dict(label=f'LPC{61+i}',orbit=path,validation_source=str(PERIODIC_OUT/(name+'_validation.json')),
            J_EE_core=J,T_ms=T,mean_rates_hz=reg.mean(0),min_rates_hz=reg.min(0),max_rates_hz=reg.max(0),
            common_time_shift_grid=shift,snapshot_times_ms=t[indices],filter_state_check=check))
    fig.text(.06,.965,'A  Core / surround activity',weight='bold',fontsize=12)
    fig.text(.343,.965,'B  Same-orbit spatial rate field',weight='bold',fontsize=12)
    fig.text(.795,.965,'C  SEEG-site rate readout',weight='bold',fontsize=12)
    fig.legend(handles=[Line2D([0],[0],color=COL[k] if k<2 else '#555555',label=['Core A','Core B','Surround'][k]) for k in range(3)],
        loc='lower left',bbox_to_anchor=(.06,.025),frameon=False,ncol=1,fontsize=9)
    fig.colorbar(im,cax=fig.add_axes([.39,.065,.25,.013]),orientation='horizontal',label='E rate (Hz / cell)')
    fig.colorbar(ic,cax=fig.add_axes([.8,.065,.17,.013]),orientation='horizontal',label='Contact-weighted rate (Hz / cell)')
    output='H1_three_new_fold_rate_fields';save(fig,output)
    write(PERIODIC_OUT/(output+'.json'),dict(rows=rows,
        scope='Actual periodic rate solutions at independently checked cycle folds. All columns share the same single common phase shift, with the Core A maximum at T/4. Shared rate scales across rows; readout is contact-weighted rate, not voltage. Existence of these cycles does not imply that an autonomous simulation approaches them or that they are interictal attractors.'))
    update_readme({output:'展示 LPC61–63 的实际周期轨道，左侧为两核及周围放电率，中间为同一周期的二维空间场，右侧为同一轨道在 SEEG 触点上的加权放电率。每行只施加一个共同相位平移，A 峰位于四分之一周期；所有行共享波形纵轴及空间、触点色标。**关注点**：LPC63 的 B 核保持高背景活动；这些是周期解，不是稳定吸引子或自限间期事件的证明。'})


if __name__=='__main__':main()
