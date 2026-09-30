"""Temporal eigenvalues, Floquet crossing and spatial rate eigenfunctions."""
from plot_rate_periodic_completion import *


def main():
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'});s=RateField()
    fig=plt.figure(figsize=(16,9));grid=fig.add_gridspec(2,3,left=.06,right=.97,bottom=.12,top=.94,hspace=.48,wspace=.35)
    ax=fig.add_subplot(grid[0,0]);rows=read(PERIODIC_OUT/'exact_low_branch_eigenvalues.json')['rows']
    for k,core in enumerate('AB'):
        rr=[q for q in rows if q['core']==core];ax.plot([q['J_EE_core'] for q in rr],[q['lambda_per_ms'][0]*1000 for q in rr],'-',color=COL[k],label=f'{core}-localized pair')
    ax.axhline(0,color='black',lw=.7);ax.set(xticks=[.935,.940,.945,.950],xlabel=r'$J_{\mathrm{EE,core}}$',ylabel=r'Re $\lambda$ (s$^{-1}$)',title='Low-rate branch: first Hopf crossings');ax.legend(frameon=False,fontsize=9);style(ax)
    ax=fig.add_subplot(grid[0,1]);q=read(PERIODIC_OUT/'TR_A_B_validation.json')['critical_point'];m=complex(*q['multiplier'])
    theta=np.linspace(-.014,.014,400);ax.plot(np.cos(theta),np.sin(theta),'k--',lw=.8)
    rr=[read(f) for f in sorted((PERIODIC_OUT/'spectral_floquet').glob('resonanceA_*.json'))]
    for row in rr:
        if row['status']!='CONVERGED':continue
        v=complex(*row['multiplier'])
        if abs(v-1)<1e-4:continue
        ax.plot([v.real,v.real],[abs(v.imag),-abs(v.imag)],'o',color='#2171b5' if abs(v)<1 else '#d95f02',ms=4)
    ax.plot([m.real,m.real],[abs(m.imag),-abs(m.imag)],'D',mfc='white',mec='black',ms=6)
    m2=complex(*read(PERIODIC_OUT/'TR_A_return_N128.json')['multiplier'])
    ax.plot([m2.real,m2.real],[abs(m2.imag),-abs(m2.imag)],'s',mfc='white',mec='black',ms=5)
    ax.annotate('TR1',(m.real,abs(m.imag)),xytext=(12,14),textcoords='offset points',fontsize=9)
    ax.annotate('TR2',(m2.real,abs(m2.imag)),xytext=(12,-12),textcoords='offset points',fontsize=9)
    ax.plot(1,0,'+',color='black',ms=8);ax.set(xlim=(.993,1.003),ylim=(-.012,.012),xlabel=r'Re $\mu$',ylabel=r'Im $\mu$',title='Two non-real unit-circle crossings');style(ax)
    ax.legend(handles=[Line2D([0],[0],marker='o',ls='',color=c,label=l) for c,l in [('#2171b5','|μ| < 1'),('#d95f02','|μ| > 1')]],frameon=False,fontsize=8,loc='lower left')
    ax=fig.add_subplot(grid[0,2]);fs=families()
    for family,label in [('A','From H1'),('B','From H2'),('single','A-leading'),('double','Alternating'),('Bleading','B-leading')]:
        rr=fs[family];ax.plot([q['J_EE_core'] for q in rr],[q['T_ms'] for q in rr],color=FAMILY[family],lw=1.4,label=label)
    ax.legend(frameon=False,fontsize=8,ncol=2)
    ax.set(xlim=(.935,1.65),xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Full-network period (ms)',title='Period belongs to a specific branch');style(ax)
    modes=[]
    for core in 'AB':
        z=np.load(RATE_OUT/f'hopf_{core}.npz');modes.append((f'H{1+"AB".index(core)}',z['vector'][None,:],float(z['J'])))
    z=np.load(PERIODIC_OUT/'TR_A_B_highorder_half_mode_N128.npz');modes.append(('TR1',z['u'],float(z['J'])))
    z=np.load(PERIODIC_OUT/'TR_A_return_mode_N128.npz');modes.append(('TR2',z['u'],float(z['J'])))
    geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz');xy=geo['contact_xy'];stats=[]
    sz=s.geo['group_size'];cell=s.geo['group_cell'];reg=s.geo['group_region'];counts=np.bincount(cell[s.E],weights=sz[s.E],minlength=400)
    bottom=grid[1,:].subgridspec(1,4,wspace=.35)
    for k,(name,u,J) in enumerate(modes):
        amp2=np.mean(abs(u)**2,axis=0);mass=sz*amp2*s.E;fractions=np.array([mass[reg==v].sum()/mass.sum() for v in range(3)])
        field=np.sqrt(np.bincount(cell[s.E],weights=mass[s.E],minlength=400)/np.maximum(counts,1));field/=field.max()
        ax=fig.add_subplot(bottom[k]);im=ax.imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='inferno',vmin=0,vmax=1)
        for center in s.geo['centers_mm']:ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=.8))
        ax.scatter(xy[:,0],xy[:,1],s=8,facecolors='none',edgecolors='cyan',linewidths=.6)
        ax.set(xlabel='x (mm)',ylabel='y (mm)',title=f'{name}: rate-mode amplitude\nJ={J:.8f}')
        stats.append(dict(point=name,J_EE_core=J,E_rate_mode_energy_fractions_A_B_surround=fractions))
    fig.colorbar(im,cax=fig.add_axes([.375,.045,.25,.018]),orientation='horizontal',label='Normalized E-rate eigenfunction amplitude')
    save(fig,'critical_eigenvalues_and_spatial_modes');write(PERIODIC_OUT/'critical_mode_energy.json',dict(rows=stats,
        observable='Neuron-count weighted E-rate eigenfunction energy; not mixed physical units of the nine local state variables'))
    update_readme(dict(critical_eigenvalues_and_spatial_modes='上排展示平衡点特征值、两处环面分岔的 Floquet 乘子和各周期分支的周期；下排展示 H1、H2、TR1、TR2 的 E 放电率特征模态。模态振幅单独归一化，不是实际放电率或传播事件。**关注点**：两处 TR 的临界乘子均为非实数，并与自治相位的 +1 乘子区分；空间特征模态不直接代表完整传播事件。'))


if __name__=='__main__':main()
