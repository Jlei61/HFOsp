"""New equilibrium Hopfs: branch location, crossing, and full spatial modes."""
from plot_rate_periodic_completion import *


def main():
    s=RateField();hh=[h for h in additional_hopfs() if int(h['label'].split('_')[0][1:])<=11];assert hh
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    nrows=1 if len(hh)<=6 else 2;ncols=len(hh) if nrows==1 else int(np.ceil(len(hh)/2))
    fig=plt.figure(figsize=(15,8.5 if nrows==1 else 11.5));grid=fig.add_gridspec(2,2,height_ratios=[1.25,nrows],left=.07,right=.97,bottom=.12,top=.95,hspace=.35,wspace=.3)
    ax=fig.add_subplot(grid[0,0]);z=np.load(RATE_OUT/'equilibrium_branch.npz')
    ids=np.arange(200,292);ax.plot(z['J'][ids],z['regional'][ids,1],'--',color='#555555',lw=1.4,label='Unstable equilibrium')
    offsets={'H3':(-32,-20),'H4':(-43,2),'H5':(-45,-22),'H6':(-10,22),'H7':(15,-10),'H8':(12,20),'H9':(12,-12),'H10':(-43,-20),'H11':(28,12)}
    for h in hh:
        label=h['label'].split('_')[0];c=FAMILY[label];x=h['J_EE_core'];y=h['rates_hz'][1]
        ax.plot(x,y,'o',mfc='white',mec=c,ms=6,zorder=5)
        ax.annotate(label,(x,y),xytext=offsets[label],textcoords='offset points',color=c,fontsize=11,
                    arrowprops=dict(arrowstyle='-',color=c,lw=.6))
    folds=read(OUT/'critical_revision/fold_audit.json')['rows']
    for i,q in enumerate(folds[:6]):
        ax.plot(q['J_EE_core'],q['rates_hz'][1],'s',mfc='white',mec='black',ms=5)
        off=[(12,-4),(12,-19),(12,-3),(12,-36),(12,16),(-18,25)][i]
        ax.annotate(f'LP{i+1}',(q['J_EE_core'],q['rates_hz'][1]),xytext=off,textcoords='offset points',fontsize=9,
            arrowprops=dict(arrowstyle='-',lw=.5,color='black'))
    ax.set(xlim=(1.002,1.034),ylim=(.9,8.2),xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Core B equilibrium rate (Hz / cell)',title='A  Folded equilibria: H3–H11')
    ax.legend(frameon=False,fontsize=9,loc='lower right');style(ax)
    ax=fig.add_subplot(grid[0,1])
    for h in hh:
        label=h['label'].split('_')[0];tr=h['transversality'];J=h['J_EE_core'];r=h.get('coordinate_value',h['core_B_rate_Hz']);step=tr['coordinate_step_Hz']
        records=h['solver_evaluations'];rr=[]
        for c in [r-step,r,r+step]:
            q=min(records,key=lambda x:abs(x.get('coordinate_value',x['core_B_rate_Hz'])-c));rr.append(q)
        xx=np.array([q['J_EE_core']-J for q in rr])*1e6;yy=np.array([q['lambda_per_ms'][0] for q in rr])*1000
        order=np.argsort(xx);ax.plot(xx[order],yy[order],'o-',color=FAMILY[label],ms=4,label=label)
    ax.axhline(0,color='black',lw=.7);ax.axvline(0,color='black',lw=.6,ls=':')
    ax.set(xlabel=r'$(J-J_H)\times10^6$',ylabel=r'Re $\lambda$ (s$^{-1}$)',title='B  Verified local imaginary-axis crossings')
    ax.legend(frameon=False,ncol=3,fontsize=10);style(ax)
    bottom=grid[1,:].subgridspec(nrows,ncols,wspace=.18,hspace=.50)
    geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz');xy=geo['contact_xy']
    sz=s.geo['group_size'];cell=s.geo['group_cell'];counts=np.bincount(cell[s.E],weights=sz[s.E],minlength=400);stats=[]
    for k,h in enumerate(hh):
        label=h['label'].split('_')[0];z=np.load(PERIODIC_OUT/f'stationary_root_counts/{label}_highorder_half.npz');v=z['vector']
        field=np.sqrt(np.bincount(cell[s.E],weights=sz[s.E]*abs(v[s.E])**2,minlength=400)/np.maximum(counts,1));field/=field.max()
        ax=fig.add_subplot(bottom[k]);im=ax.imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),vmin=0,vmax=1,cmap='inferno')
        for center in s.geo['centers_mm']:ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=.8))
        ax.scatter(xy[:,0],xy[:,1],s=8,facecolors='none',edgecolors='cyan',linewidths=.6)
        ax.set(title=f'{label}: {h["frequency_hz"]:.2f} Hz',xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20])
        if k%ncols==0:ax.set_ylabel('y (mm)')
        else:ax.tick_params(labelleft=False)
        mass=sz*abs(v)**2*s.E;fractions=np.array([mass[s.geo['group_region']==i].sum()/mass.sum() for i in range(3)])
        density=np.array([np.average(abs(v[s.E&(s.geo['group_region']==i)])**2,weights=sz[s.E&(s.geo['group_region']==i)]) for i in range(3)])
        stats.append(dict(label=label,J_EE_core=h['J_EE_core'],frequency_hz=h['frequency_hz'],mode_E_energy_fractions=fractions,
            neuron_normalized_mode_power_by_region=density/density.max(),criticality=h['criticality'],validation=h['validation']))
    fig.colorbar(im,cax=fig.add_axes([.38,.055,.26,.018]),orientation='horizontal',label='Normalized E-rate eigenfunction amplitude')
    save(fig,'additional_equilibrium_hopf_audit')
    write(PERIODIC_OUT/'additional_equilibrium_hopf_figure.json',dict(rows=stats,
        scope='Validated local Hopfs on already unstable equilibrium branches. Spatial maps are eigenfunction amplitudes, not time-simulated propagation waves. No claim of exhaustive inventory or stable child cycles.'))
    update_readme(dict(additional_equilibrium_hopf_audit='展示在原两处 Hopf 之外新增、已经通过完整状态扰动与非线性周期子分支检验的 Hopf；左侧保留平衡分支的折返几何，右侧显示各临界点附近复特征值实部的过零。下排为对应 400 网格上的 E 放电率特征模态，逐模态归一化。**关注点**：这些临界点位于已经不稳定的平衡分支；局部超临界不等于全网络产生稳定 burst，模态图也不等于真实传播事件。'))


if __name__=='__main__':main()
