"""Same parameter: unstable H1-family cycle versus stable burst cycle."""
from plot_rate_periodic_completion import *
from src.snn_contact_display import CONTACT_ORDER,contact_indices,SHAFT_COLORS


def main():
    s=RateField();unstable=read(PERIODIC_OUT/'H1_large_orbit_instability.json');stable=read(PERIODIC_OUT/'strong_sameJ_H1_stability.json')
    assert unstable['status']=='PAIRED_STEP_UNSTABLE' and stable['status']=='NUMERICALLY_STABLE'
    rows=[dict(title='H1-family cycle: unstable',orbit=unstable['orbit'],evidence=unstable),
          dict(title='A-leading burst: numerically stable',orbit=stable['orbit'],evidence=stable)]
    geometry=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz');xy=geometry['contact_xy'];order=contact_indices(geometry['contact_names'].tolist())
    cell=s.geo['group_cell'];size=s.geo['group_size'];count=np.bincount(cell[s.E],weights=size[s.E],minlength=400)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(18,7.2));grid=fig.add_gridspec(2,4,width_ratios=[1.15,1.75,1.25,.65],left=.05,right=.985,bottom=.16,top=.87,wspace=.38,hspace=.55)
    metadata=[]
    for i,row in enumerate(rows):
        z=np.load(row['orbit']);r=z['r'];T=float(z['T']);J=float(z['J']);assert abs(J-unstable['J_EE_core'])<1e-12
        reg=np.array([s.regional_rates(v) for v in r]);N=len(r);shift=int(np.argmax(reg[:,0]))-N//4
        r=np.roll(r,-shift,axis=0);reg=np.roll(reg,-shift,axis=0);t=np.arange(N)*T/N
        ax=fig.add_subplot(grid[i,0])
        for k,c in enumerate([*COL,'#555555']):ax.plot(t,reg[:,k],color=c,lw=1.2)
        ax.set(xlabel='Time within one period (ms)',ylabel='Rate (Hz / cell)',xlim=(0,T),ylim=(0,270),title=f'{"AB"[i]}  {row["title"]}\nT = {T:.2f} ms');style(ax)
        sub=grid[i,1].subgridspec(1,4,wspace=.16)
        for j,dt in enumerate([-20,0,20,40]):
            ix=int(round(N/4+dt/T*N))%N
            field=np.bincount(cell[s.E],weights=size[s.E]*r[ix,s.E]*1000,minlength=400)/np.maximum(count,1)
            ax=fig.add_subplot(sub[j]);im=ax.imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='inferno',norm=PowerNorm(.55,0,500))
            for center in s.geo['centers_mm']:ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=.8))
            ax.scatter(xy[:,0],xy[:,1],s=8,facecolors='none',edgecolors='cyan',linewidths=.6)
            ax.set(xticks=[0,20],yticks=[0,20],title=f'{dt:+d} ms');ax.tick_params(labelsize=8)
            if j:ax.tick_params(labelleft=False)
            else:ax.set_ylabel('y (mm)')
            if i:ax.set_xlabel('x (mm)')
        ax=fig.add_subplot(grid[i,2]);contact=r@s.geo['contact_rate_weights']*1000
        ic=ax.imshow(contact[:,order].T,origin='upper',aspect='auto',extent=(0,T,14.5,-.5),cmap='magma',norm=PowerNorm(.5,0,200))
        ax.set(yticks=np.arange(15),yticklabels=CONTACT_ORDER,xlabel='Time within one period (ms)',title='SEEG-site rate readout')
        ax.tick_params(axis='y',labelsize=7,length=2);ax.axhline(3.5,color='white',lw=.6)
        for tick,n in zip(ax.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[n[:3]])
        ax=fig.add_subplot(grid[i,3])
        if i==0:mu=np.array([complex(*v) for v in unstable['dominant_multipliers']])
        else:
            spectrum=read(Path(stable['floquet_source']));mu=np.array([complex(*v) for v in spectrum['nontrivial_multipliers']]);mu=mu[np.argsort(abs(mu))[::-1]][:2]
        ax.axhline(1,color='black',ls='--',lw=.8);ax.plot([1,2],abs(mu),'o',color='#b2182b' if i==0 else '#2166ac')
        for x,y in zip([1,2],abs(mu)):ax.annotate(f'{y:.4g}',(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
        ax.set(xlim=(.5,2.5),xticks=[1,2],xticklabels=[r'$\mu_1$',r'$\mu_2$'],yscale='log',ylim=(.3,50000),ylabel=r'Largest nontrivial $|\mu|$',title='Floquet');style(ax)
        metadata.append(dict(orbit=row['orbit'],J_EE_core=J,T_ms=T,mean_rates_hz=reg.mean(0),dominant_nontrivial_multipliers=mu,status=row['title']))
    fig.suptitle(r'Same spatial rate model, same $J_{\mathrm{EE,core}}=1.02939676$',fontsize=14)
    fig.legend(handles=[Line2D([0],[0],color=c,label=n) for c,n in zip([*COL,'#555555'],['Core A','Core B','Surround'])],loc='lower left',bbox_to_anchor=(.05,.01),frameon=False,ncol=3)
    fig.colorbar(im,cax=fig.add_axes([.34,.07,.24,.014]),orientation='horizontal',label='E rate (Hz / cell)')
    fig.colorbar(ic,cax=fig.add_axes([.69,.07,.16,.014]),orientation='horizontal',label='Contact-weighted rate (Hz / cell)')
    save(fig,'same_parameter_distinct_cycle_stability')
    write(PERIODIC_OUT/'same_parameter_distinct_cycle_stability.json',dict(rows=metadata,
        scope='Exact same-J periodic BVPs and their own full-state Floquet evidence. An unstable periodic profile is not a stable free-running trajectory. Spatial frames align to the core A peak; electrode weights, channel order and color scales are shared.'))
    update_readme(dict(same_parameter_distinct_cycle_stability='在完全相同的 JEE,core 下比较 H1 延续来的不稳定大幅周期解与另一条数值稳定的 A-leading burst 周期解，给出各自时序、二维空间场、接触点放电率及非平凡 Floquet 乘子。空间快照均相对 Core A 峰对齐，并共用色标与触点顺序。**关注点**：上排是周期边值解，不是可持续的稳定吸引子；两条解不能因参数或波形幅度接近而合并为同一动力学状态。'))


if __name__=='__main__':main()
