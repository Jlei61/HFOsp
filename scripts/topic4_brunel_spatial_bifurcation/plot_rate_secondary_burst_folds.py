"""Refined secondary burst folds with their own fields and contact readouts."""
from plot_rate_periodic_composite import *


def main():
    s=RateField();validation=read(PERIODIC_OUT/'secondary_folds_mesh_validation.json')['rows']
    assert len(validation)==4
    geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    order=contact_indices(geo['contact_names'].tolist());xy=geo['contact_xy']
    cell=s.geo['group_cell'];sz=s.geo['group_size'];ct=np.bincount(cell[s.E],weights=sz[s.E],minlength=400)
    fig=plt.figure(figsize=(22,11));grid=fig.add_gridspec(4,4,width_ratios=[1.15,1.15,1.6,1.15],left=.06,right=.985,bottom=.11,top=.91,wspace=.4,hspace=.65)
    ax=fig.add_subplot(grid[:2,0]);rr=[read(f) for f in sorted((PERIODIC_OUT/'orbits').glob('arcDoubleLow_*_N512.json'))]
    ax.plot([q['J_EE_core'] for q in rr],[q['mean_rates_hz'][1] for q in rr],color=FAMILY['double'],lw=1.4,label='Periodic branch geometry')
    ax.set(xlim=(.938,.9408),ylim=(5.2,13.3),xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Core B period mean (Hz)',title='A  Secondary cycle folds')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'));style(ax)
    check=fig.add_subplot(grid[2:,0]);check.semilogy(range(4),[q['J_mesh_difference'] for q in validation],'o',color='#333333',ms=6)
    check.set(xticks=range(4),xticklabels=[f'LPC{i}' for i in range(21,25)],ylabel=r'$|J_{2048}-J_{1024}|$',ylim=(1e-9,1e-7),title='Time-grid refinement')
    for i,q in enumerate(validation):check.annotate(f"{q['J_mesh_difference']:.1e}",(i,q['J_mesh_difference']),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
    style(check);records=[]
    check.plot(3,abs(read(PERIODIC_OUT/'LPC_double_secondary4_N4096.json')['J_EE_core']-validation[3]['J_EE_core']),'s',color='#2166ac',ms=5)
    check.set_ylim(1e-12,1e-7)
    check.annotate('4096 vs 2048', (3,2.274e-12),xytext=(-9,10),textcoords='offset points',ha='right',fontsize=9,color='#2166ac')
    for i in range(4):
        n=4096 if i==3 else 2048
        q=read(PERIODIC_OUT/f'LPC_double_secondary{i+1}_N{n}.json');z=np.load(q['orbit']);r=z['r'];T=float(z['T']);N=len(r);t=np.arange(N)*T/N
        reg=np.array([s.regional_rates(v) for v in r]);label=CRITICAL_LABELS[q['label']]
        ax.plot(q['J_EE_core'],reg[:,1].mean(),'s',mfc='white',mec='black',ms=5)
        ax.annotate(label,(q['J_EE_core'],reg[:,1].mean()),xytext=(7,7),textcoords='offset points',fontsize=9)
        w=fig.add_subplot(grid[i,1])
        for k in [0,1]:w.plot(t,reg[:,k],color=COL[k],lw=1.1)
        w.plot(t,reg[:,2],color='#555555',lw=.7)
        w.set(xlim=(0,T),ylim=(0,270),xlabel='Time (ms)',ylabel='Rate (Hz / cell)')
        w.set_title(f'{label}   J={q["J_EE_core"]:.7f}\nT={T:.2f} ms',loc='left',fontsize=10);style(w)
        peaks=[]
        for k in [0,1]:
            p=find_peaks(np.tile(reg[:,k],3),height=10,prominence=5,distance=int(40/T*N))[0]
            p=p[(p>=N)&(p<2*N)]-N;assert len(p)==2;peaks.append(p)
        a,b=peaks;assert a[0]<b[0] and b[1]<a[1]
        ids=[a[0],b[0],b[1],a[1]];sub=grid[i,2].subgridspec(1,4,wspace=.1)
        for j,index in enumerate(ids):
            f=fig.add_subplot(sub[j]);fld=np.bincount(cell[s.E],weights=sz[s.E]*r[index,s.E]*1000,minlength=400)/np.maximum(ct,1)
            imf=f.imshow(fld.reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='inferno',norm=PowerNorm(.55,0,500))
            for center in s.geo['centers_mm']:f.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=.8))
            f.scatter(xy[:,0],xy[:,1],s=8,facecolors='none',edgecolors='cyan',linewidths=.6)
            f.set(xticks=[0,20],yticks=[0,20],title=f'{t[index]:.0f} ms');f.tick_params(labelsize=8)
            if j:f.tick_params(labelleft=False)
            else:f.set_ylabel('y (mm)')
            if i==3:f.set_xlabel('x (mm)')
        c=fig.add_subplot(grid[i,3]);contact=r@s.geo['contact_rate_weights']*1000
        imc=c.imshow(contact[:,order].T,origin='upper',aspect='auto',extent=(0,T,14.5,-.5),cmap='magma',norm=PowerNorm(.5,0,200))
        c.set(yticks=np.arange(15),yticklabels=CONTACT_ORDER,xlabel='Time (ms)');c.tick_params(axis='y',labelsize=7)
        c.axhline(3.5,color='white',lw=.6)
        for tick,name in zip(c.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[name[:3]])
        records.append(dict(label=label,orbit=q['orbit'],J_EE_core=q['J_EE_core'],T_ms=T,
            peaks_A_B_ms=[t[a],t[b]],peak_rates_A_B_Hz=[reg[a,0],reg[b,1]],snapshot_times_ms=t[ids],
            cycle_stability='Not inferred from the fold marker; pending full Floquet coverage'))
    ax.legend(frameon=False,fontsize=8,loc='upper right')
    for x,text in [(.307,'B  Same-orbit activity'),(.516,'C  Two events in one full period'),(.802,'D  SEEG-site rate readout')]:fig.text(x,.962,text,weight='bold',fontsize=12)
    fig.legend(handles=[Line2D([0],[0],color=COL[k],label=f'Core {"AB"[k]}') for k in [0,1]]+[Line2D([0],[0],color='#555555',label='Surround')],loc='lower left',bbox_to_anchor=(.305,.025),frameon=False,ncol=1)
    fig.colorbar(imf,cax=fig.add_axes([.53,.045,.18,.012]),orientation='horizontal',label='E rate (Hz / cell)')
    fig.colorbar(imc,cax=fig.add_axes([.82,.045,.13,.012]),orientation='horizontal',label='Contact-weighted rate (Hz / cell)')
    save(fig,'secondary_burst_folds_spatial_readout')
    write(PERIODIC_OUT/'secondary_burst_fold_cases.json',dict(rows=records,
        model='Unchanged 935-group rate DDE, N=2048 for LPC21-23 and N=4096 for LPC24',
        interpretation='Each refined fold orbit still has two bursts per core and alternating A-first/B-first order. Amplitudes and phase lags change; these folds are not period-doubling labels.',
        limitation='BVP fold orbits are not presented as stable attractors; full branch stability is still being computed.'))
    update_readme(dict(secondary_burst_folds_spatial_readout='展示交替传播周期分支上 LPC21–24 的精化折点、网格加密误差及各自的双核波形、二维场和接触点放电率读出。LPC21–23 使用 N=2048，LPC24 使用 N=4096；每周期仍各有两次 burst，并保留 A 先、B 先两种顺序。**关注点**：折点不等于稳定状态；本图不将这些折点解释为倍周期或方向切换，稳定性由另行的 Floquet 扫描判定。'))


if __name__=='__main__':main()
