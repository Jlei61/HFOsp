"""Validated PD location, multipliers and full-space antiperiodic mode."""
from plot_rate_periodic_completion import *


def main():
    check=read(PERIODIC_OUT/'PD_double_low_validation.json');assert check['status']=='VALIDATED_PD'
    q=max([read(f) for f in PERIODIC_OUT.glob('PD_double_low_N*.json')],key=lambda x:x['N'])
    s=RateField();z=np.load(PERIODIC_OUT/f'PD_double_low_mode_N{q["N"]}.npz');u=z['u'];T=float(z['T'])
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(2,2,figsize=(13,9.5),layout='constrained');ax=axs[0,0]
    paths=sorted((PERIODIC_OUT/'orbits').glob('arcDoubleLow_*_N512.json'))[::-1]
    paths+=sorted((PERIODIC_OUT/'orbits').glob('arcDoubleDown_*_N512.json'))[::-1]
    rr=[read(f) for f in paths];ax.plot([r['J_EE_core'] for r in rr],[r['mean_rates_hz'][0] for r in rr],color=FAMILY['double'],lw=1.8,label='Two-burst cycle mean')
    fold=next(v for v in critical() if v['label']=='LPC_double_low')
    for v,marker,label,off in [(fold,'s','LPC20',(-8,-23)),(q,'v','PD1',(9,9))]:
        m=read(Path(v['orbit']).with_suffix('.json'));ax.plot(v['J_EE_core'],m['mean_rates_hz'][0],marker,mfc='white',mec='black',ms=8)
        ax.annotate(label,(v['J_EE_core'],m['mean_rates_hz'][0]),xytext=off,textcoords='offset points')
    for filename,marker,label in [('nearBurstFold_J0.938252184_N1024','x','Unstable sampled cycle'),
                                  ('PD_double_low_eval_J0.93900217448_N1024','o','Stable sampled cycle')]:
        m=read(PERIODIC_OUT/f'orbits/{filename}.json');ax.plot(m['J_EE_core'],m['mean_rates_hz'][0],marker,color='black',ms=7,label=label)
    ax.set(xlim=(.93804,.93908),ylim=(8.5,13.5),xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Core A mean rate (Hz / cell)',title='a   Cycle existence and stability are distinct')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'));ax.legend(frameon=False,fontsize=9,loc='upper left');style(ax)
    ax=axs[0,1]
    paths=['nearBurstFold_J0.938252184_N1024_dt0.05','PD_double_low_eval_J0.93900217448_N1024_dt0.1']
    for filename in paths:
        m=read(PERIODIC_OUT/f'floquet/{filename}.json');vals=np.array([complex(*v) for v in m['multipliers']]);v=vals[np.argmin(vals.real)]
        ax.plot(m['J_EE_core'],v.real,'o',color='#bd3754',ms=7);ax.annotate(f'{v.real:.3f}',(m['J_EE_core'],v.real),xytext=(6,5),textcoords='offset points')
    ax.plot(q['J_EE_core'],-1,'v',color='black',ms=8);ax.annotate('PD1',(q['J_EE_core'],-1),xytext=(7,-20),textcoords='offset points')
    ax.axhline(-1,color='black',ls='--',lw=.9);ax.axhline(0,color='#888888',lw=.6)
    ax.set(xlim=(.93814,.93912),ylim=(-2.65,.12),xlabel=r'$J_{\mathrm{EE,core}}$',ylabel=r'Real Floquet multiplier $\mu$',title='b   The negative multiplier crosses −1')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'));style(ax)
    ax=axs[1,0];rr=np.array([s.regional_rates(v) for v in u]);rr/=np.max(abs(rr[:,:2]));t=np.arange(len(u))*T/len(u)
    for k in [0,1]:
        ax.plot(t,rr[:,k],color=COL[k],lw=1.2,label=f'Core {"AB"[k]}')
        ax.plot(t+T,-rr[:,k],color=COL[k],lw=1.2)
    ax.axvline(T,color='black',ls=':',lw=.8);ax.legend(frameon=False,ncol=2)
    ax.set(xlim=(0,2*T),xlabel='Time (ms)',ylabel='Rate perturbation (normalized)',title=r'c   Critical mode: $u(t+T)=-u(t)$');style(ax)
    ax=axs[1,1];sz=s.geo['group_size'];cell=s.geo['group_cell'];e=s.E
    power=np.mean(u*u,axis=0);num=np.bincount(cell[e],weights=sz[e]*power[e],minlength=400);den=np.bincount(cell[e],weights=sz[e],minlength=400)
    field=np.sqrt(num/np.maximum(den,1));field/=field.max()
    im=ax.imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0,vmax=1)
    for k,center in enumerate(s.geo['centers_mm']):
        ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=1.1));ax.text(*center,'AB'[k],color='white',ha='center',va='center',fontsize=11)
    geom=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz');xy=geom['contact_xy']
    ax.scatter(xy[:,0],xy[:,1],s=19,facecolors='none',edgecolors='cyan',linewidths=.8)
    rms=np.sqrt(check['per_E_cell_mean_squared_mode_relative_to_network'])
    ax.set(xlabel='x (mm)',ylabel='y (mm)',title=f'd   Critical mode in the 2D field\nRMS per E cell / network: A {rms[0]:.2f} · B {rms[1]:.2f}')
    fig.colorbar(im,ax=ax,label='Normalized RMS rate perturbation',shrink=.8)
    save(fig,'period_doubling_on_burst_branch')
    with (F/'README.md').open('a') as f:
        f.write('\n### period_doubling_on_burst_branch\n展示双 burst 周期分支上的 PD1、其两侧实际计算的负 Floquet 乘子，以及反周期临界模态的时间和二维空间分布。扰动每经过一个原周期反号，两个周期后才重复；空间图是特征模态，不是实际放电率。**关注点**：PD 点位经过时间网格与全延迟单周期传播交叉验证，子分支临界性和稳定性需单独阅读。\n')


if __name__=='__main__':main()
