"""First torus branch: precise bifurcation direction and same-DDE spatial mode."""
from plot_rate_periodic_completion import *


def main():
    v=read(PERIODIC_OUT/'TR_A_B_validation.json');root=v['critical_point'];J0=root['J_EE_core']
    s=RateField();rows=sorted([q for q in v['torus_solutions'] if [q['N_theta'],q['N_psi']]==[64,16]],key=lambda q:q['amplitude_hz'])
    fig,axs=plt.subplots(2,3,figsize=(15,8.7),layout='constrained')
    plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none'})
    ax=axs[0,0];x=np.array([q['J_EE_core']-J0 for q in rows])*1e8;y=np.array([q['amplitude_hz'] for q in rows])
    ax.plot(np.r_[0,x],np.r_[0,y],'o--',color='#c44e52',ms=4,label='Locally unstable torus')
    ax.plot([-6,0],[0,0],color='#333333',lw=2,label='Parent: stable in critical mode')
    ax.plot([0,1],[0,0],'--',color='#333333',lw=2)
    ax.plot(0,0,'D',mfc='white',mec='black',ms=6)
    ax.set(xlabel=r'$(J_{\mathrm{EE,core}}-J_{\mathrm{TR1}})\times10^8$',ylabel='Mode projection amplitude (Hz)',title='A  Subcritical torus branch')
    ax.legend(frameon=False,fontsize=8,loc='upper right')
    ax=axs[0,1];delta=root['transversal_slope']['step'];ex=root['transversal_slope']['real_exponents']
    ax.plot([-delta*1e7,0,delta*1e7],[ex[0]*1e8,0,ex[1]*1e8],'o-',color='#333333',ms=4)
    ax.axhline(0,color='#888888',lw=.6)
    ax.set(xlabel=r'$(J_{\mathrm{EE,core}}-J_{\mathrm{TR1}})\times10^7$',ylabel=r'$\mathrm{Re}(\lambda)\times10^8$ (ms$^{-1}$)',title='B  Parent transverse instability')
    ax=axs[0,2]
    for mesh,marker,col in [([64,8],'o','#888888'),([128,8],'x','#2166ac'),([64,16],'+','#c44e52')]:
        rr=sorted([q for q in v['torus_solutions'] if [q['N_theta'],q['N_psi']]==mesh and q['amplitude_hz']<=.04],key=lambda q:q['amplitude_hz'])
        ax.plot([q['amplitude_hz']**2*1e3 for q in rr],[(q['J_EE_core']-J0)*1e8 for q in rr],marker,color=col,ms=7,label=f'{mesh[0]} × {mesh[1]} angles')
    ax.set(xlabel=r'Mode amplitude squared ($10^{-3}$ Hz$^2$)',ylabel=r'$(J-J_{\mathrm{TR1}})\times10^8$',title='C  Independent angle refinements');ax.legend(frameon=False,fontsize=9)
    ax=axs[1,0];ax.plot(y,[q['modulation_period_ms']/1000 for q in rows],'o-',color='#c44e52',ms=4)
    ax.set(xlabel='Mode projection amplitude (Hz)',ylabel='Slow modulation period (s)',title='D  Slow second frequency')
    chosen=next(q for q in rows if q['amplitude_hz']==.04);z=np.load(chosen['source']);r=z['r'];nt,np_,P=r.shape
    regions=[]
    for k in [0,1]:
        mask=s.E&(s.geo['group_region']==k);w=s.geo['group_size'][mask];w=w/w.sum()
        regions.append(np.einsum('tpg,g->tp',r[:,:,mask],w)*1000)
    ax=axs[1,1];time=np.arange(nt)*float(z['T'])/nt
    for k in [0,1]:
        ax.fill_between(time,regions[k].min(1),regions[k].max(1),color=COL[k],alpha=.3)
        ax.plot(time,regions[k].mean(1),color=COL[k],lw=1.1,label=f'Core {"AB"[k]}')
    ax.set(xlabel='Fast phase (ms)',ylabel='Rate (Hz / cell)',title='E  Envelope over the slow angle');ax.legend(frameon=False,fontsize=9)
    ax=axs[1,2];q=z['q'];mode=np.sqrt(np.mean(abs(q)**2,axis=0));cell=s.geo['group_cell'];sz=s.geo['group_size'];mask=s.E
    field=np.bincount(cell[mask],weights=sz[mask]*mode[mask]**2,minlength=400)/np.maximum(np.bincount(cell[mask],weights=sz[mask],minlength=400),1)
    field=np.sqrt(field);im=ax.imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='inferno',vmin=0,vmax=field.max())
    for center in s.geo['centers_mm']:ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=1))
    geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz');xy=geo['contact_xy']
    ax.scatter(xy[:,0],xy[:,1],s=10,facecolors='none',edgecolors='cyan',linewidths=.6)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',title='F  Critical rate mode');fig.colorbar(im,ax=ax,label='Cell RMS (normalized group mode)',shrink=.85)
    for ax in axs.ravel():style(ax)
    save(fig,'subcritical_torus_branch')
    write(PERIODIC_OUT/'torus_figure_metadata.json',dict(validation=str(PERIODIC_OUT/'TR_A_B_validation.json'),example=chosen,
        core_envelope_range_hz=[[float(a.min()),float(a.max())] for a in regions],
        meaning='A local weak-oscillation torus; branch-direction instability is inferred, not full torus Lyapunov certification. No stable irregular-burst claim.'))
    update_readme(dict(subcritical_torus_branch='展示 TR1 的双角度不变环面边值解、母周期轨道的临界指数过零、两个角度分别加密后的分支方向，以及对应双核波形包络和空间特征模态。环面向较小 J 延伸，结合横向特征指数斜率支持局部亚临界分岔；局部径向不稳定性据此推断。**关注点**：此处是弱活动的双频解，未证明稳定 irregular burst；图中慢调制周期与 burst 间隔不是同一个量。'))


if __name__=='__main__':main()
