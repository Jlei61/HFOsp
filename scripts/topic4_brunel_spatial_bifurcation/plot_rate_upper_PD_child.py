"""Actual nonlinear PD child: temporal, spatial and contact differences."""
from plot_rate_periodic_composite import *
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--branch',choices=['upper','return'],default='upper')
    args=parser.parse_args()
    stem='upper' if args.branch=='upper' else 'return'
    label='PD2' if stem=='upper' else 'PD3'
    output='upper_period_doubling_nonlinear_child' if stem=='upper' else 'H1_return_period_doubling_nonlinear_child'
    q=read(PERIODIC_OUT/f'PD_{stem}_child_classification.json')
    assert q['status'] in ['SUPERCRITICAL_PD','SUBCRITICAL_PD']
    if stem=='return':
        assert q['full_physical_child_checks']
        corrected=read(Path(q['physical_mode_source']))
        assert corrected['status']=='PHYSICAL_CHILD_MODES_CHECKED_REVIEW_PENDING'
        assert Path(corrected['radial_modes'][-1]['orbit']).resolve()==Path(q['child_orbit']).resolve()
    s=RateField();z=np.load(q['child_orbit']);r=z['r'];T=float(z['T']);N=len(r);half=N//2
    assert np.min(r)>0
    regional=np.array([s.regional_rates(v) for v in r])
    shift=int(np.argmin(regional[:half,:2].sum(axis=1)))
    r=np.roll(r,-shift,axis=0);regional=np.roll(regional,-shift,axis=0)
    t=np.arange(N)*T/N;difference=(r[half:]-r[:half])*1000
    regional_difference=regional[half:]-regional[:half]
    geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    order=contact_indices(geo['contact_names'].tolist());xy=geo['contact_xy']
    contact=r@s.geo['contact_rate_weights']*1000
    contact_difference=contact[half:]-contact[:half]
    cell=s.geo['group_cell'];size=s.geo['group_size'];e=s.E
    total=np.bincount(cell[e],weights=size[e],minlength=400)
    projection=sparse.coo_matrix((size[e]/np.maximum(total[cell[e]],1),
        (np.flatnonzero(e),cell[e])),shape=(s.P,400)).tocsr()
    field_difference=np.asarray(difference@projection)
    field_RMS=np.sqrt(np.mean(field_difference**2,axis=0))
    departure=read(PERIODIC_OUT/f'PD_{stem}_child_departure.json')
    finest=max(a['N'] for a in departure['rows'])
    rows=sorted([a for a in departure['rows'] if a['N']==finest],key=lambda a:abs(a['J_shift']))
    exponent=6 if stem=='upper' else 9
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(2,3,figsize=(16.5,9))
    fig.subplots_adjust(left=.065,right=.975,bottom=.12,top=.88,wspace=.5,hspace=.60)
    ax=axs[0,0]
    xx=np.array([a['J_shift'] for a in rows]);yy=np.array([a['odd_rate_RMS_Hz'] for a in rows])
    coefficients=xx/yy**2
    if np.ptp(coefficients)/abs(np.mean(coefficients))<.1:
        grid=np.linspace(0,xx[-1],200)
        ax.plot(grid*10**exponent,np.sqrt(grid/np.mean(coefficients)),
            '--',color='#00897b',lw=1.,label='Local square-root scaling')
    ax.plot(xx*10**exponent,yy,'o',color='#00897b',ms=4,label='Checked child')
    ax.plot(0,0,'v',color='black',ms=5)
    ax.legend(frameon=False,fontsize=8)
    ax.set(xlabel=rf'$(J_{{\mathrm{{EE,core}}}}-J_{{\mathrm{{{label}}}}})\times10^{exponent}$',
        ylabel='Odd-component RMS rate (Hz / cell)',title='A   Local doubled-branch departure')
    style(ax)
    for k,color in enumerate([*COL,'#555555']):
        axs[0,1].plot(t,regional[:,k],color=color,lw=1.,label=['Core A','Core B','Surround'][k])
        axs[0,2].plot(t[:half],regional_difference[:,k],color=color,lw=1.)
    axs[0,1].axvline(T/2,color='black',lw=.65,ls='--')
    axs[0,1].set(xlim=(0,T),xlabel='Time (ms)',ylabel='Rate (Hz / E cell)',title='B   Full nonlinear child period')
    axs[0,1].legend(frameon=False,fontsize=8,loc='upper right')
    axs[0,2].axhline(0,color='black',lw=.65)
    axs[0,2].set(xlim=(0,T/2),xlabel='Time within each half (ms)',
        ylabel='Second − first half (Hz / E cell)',title='C   Consecutive halves differ')
    for ax in axs[0,1:]:style(ax)
    ax=axs[1,0];im=ax.imshow(field_RMS.reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0)
    for k,center in enumerate(s.geo['centers_mm']):
        ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=.8))
        ax.text(*center,'AB'[k],ha='center',va='center',color='white',fontsize=8)
    ax.scatter(xy[:,0],xy[:,1],s=9,facecolors='none',edgecolors='cyan',linewidths=.5)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',title='D   Spatial field difference')
    fig.colorbar(im,ax=ax,pad=.04,label='RMS difference (Hz / E cell)',shrink=.82)
    for ax,data,limit,end,title in [
        (axs[1,1],contact[:,order].T,None,T,'E   SEEG-site rate readout'),
        (axs[1,2],contact_difference[:,order].T,float(abs(contact_difference).max()),T/2,'F   SEEG-site difference')]:
        opts=dict(cmap='magma',vmin=0,vmax=float(contact.max())) if limit is None else dict(cmap='RdBu_r',vmin=-limit,vmax=limit)
        im=ax.imshow(data,origin='upper',aspect='auto',extent=(0,end,14.5,-.5),**opts)
        ax.axhline(3.5,color='black',lw=.65)
        if limit is None:ax.axvline(T/2,color='white',lw=.65,ls='--')
        ax.set(yticks=np.arange(15),yticklabels=CONTACT_ORDER,xlabel='Time (ms)',title=title)
        ax.tick_params(axis='y',labelsize=7,length=2)
        for tick,name in zip(ax.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[name[:3]])
        fig.colorbar(im,ax=ax,pad=.04,label='Rate (Hz / cell)' if limit is None else 'Second − first half (Hz / cell)',shrink=.82)
    title='Supercritical' if q['status']=='SUPERCRITICAL_PD' else 'Subcritical'
    title=q.get('display_title',title+' '+label)
    fig.suptitle(title+rf'  |  $J_{{\mathrm{{EE,core}}}}={float(z["J"]):.9f}$'+
        f'  |  Full period {T:.2f} ms',y=.965,fontsize=14)
    save(fig,output)
    peak_readout=[]
    for k in [0,1]:
        values=regional[:,k];peaks=find_peaks(np.tile(values,3),height=20,prominence=10,
            distance=round(60*N/T))[0];peaks=peaks[(peaks>=N)&(peaks<2*N)]-N
        interp=CubicSpline(np.r_[t,T],np.r_[values,values[0]],bc_type='periodic')
        times=np.sort([minimize_scalar(lambda tt:-float(interp(tt%T)),
            bounds=((i-1)*T/N,(i+1)*T/N),method='bounded').x%T for i in peaks])
        peak_readout.append(dict(core='AB'[k],peak_times_ms=times,
            peak_rates_Hz=interp(times),cyclic_peak_intervals_ms=np.diff(np.r_[times,times[0]+T]) if len(times) else [],
            peak_definition='Regional E-rate peaks above 20 Hz, prominence 10 Hz, distance 60 ms; periodic cubic interpolation.'))
    write(PERIODIC_OUT/f'PD_{stem}_child_readout.json',dict(source=q['child_orbit'],classification=q['status'],
        child_stability=q.get('child_stability','See classification source'),peaks=peak_readout,
        period_ms=T,core_A_B_peaks_per_full_period=q['core_A_B_peaks_per_full_child_period'],
        max_regional_half_difference_Hz=np.max(abs(regional_difference),axis=0),
        contact_names=geo['contact_names'].tolist(),max_contact_half_difference_Hz=np.max(abs(contact_difference),axis=0),
        phase_shift_samples=shift,readout='Same nonlinear 2T orbit; physical rate and field differences, no arbitrary linear mode normalization. Contact-weighted firing rate, not SEEG voltage.'))
    note=f'展示 {label} 分支切换得到的真实两倍周期解，包括局部分支离开、延续后子轨道的全周期波形、前后半周期差值、二维场和接触点读出变化。后五幅读出来自同一条已检查的非线性周期轨道，差值为实际 Hz，没有使用任意归一化的线性特征向量。'
    note+=('PD3 新生方向虽稳定，轨道仍有强不稳定方向；这是不稳定周期解，不能作为自主仿真的稳定吸引子。' if stem=='return' else '')
    note+='**关注点**：完整周期不等于每个 burst 的间隔；空间图是前后半周期的实际差值，接触读出不是 SEEG 电压。'
    update_readme({output:note})


if __name__=='__main__':main()
