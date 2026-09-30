"""Observe the continued two-angle solutions without assigning new bifurcations."""
from plot_rate_periodic_completion import *
from scipy.signal import resample


def main():
    s=RateField();folder=PERIODIC_OUT/'tori';root=read(PERIODIC_OUT/'TR_A_return_N128.json')['J_EE_core']
    paths=[folder/f'strict_TR2_a{a:.6f}Hz_N64x16.npz' for a in [.005,.01,.02,.04]]
    paths +=[folder/'extend_TR2_a0.060000Hz_N64x32.npz']+sorted(folder.glob('arcTR2_*_N64x32.npz'))
    rows=[]
    for path in paths:
        z=np.load(path);meta=read(path.with_suffix('.json'))
        if meta['status']!='CONVERGED':continue
        r=z['r']*1000;rates=[]
        for k in range(3):
            mask=s.E&(s.geo['group_region']==k);w=s.geo['group_size'][mask];w=w/w.sum()
            rates.append(r[:,:,mask]@w)
        rate=np.stack(rates,axis=-1);fast=np.fft.fft(rate,axis=0)[1]/len(rate)
        dense=resample(fast,512,axis=0);relative=dense[:,1]/dense[:,0]
        closed=np.r_[relative,relative[0]];steps=np.angle(closed[1:]/closed[:-1])
        phase=np.unwrap(np.angle(relative));T=float(z['T']);lag=-phase*T/(2*np.pi)
        rr=resample(resample(rate,256,axis=0),512,axis=1)
        rows.append(dict(source=str(path),J_EE_core=float(z['J']),amplitude_hz=float(z['amplitude_hz']),
            fast_period_ms=T,slow_period_s=float(2*np.pi/z['nu']/1000),
            first_harmonic_amplitude_range_Hz=[[float(2*abs(dense[:,k]).min()),float(2*abs(dense[:,k]).max())] for k in range(2)],
            first_harmonic_amplitude_mean_Hz=[float(2*abs(dense[:,k]).mean()) for k in range(2)],
            relative_phase_winding=float(steps.sum()/(2*np.pi)),phase_step_max_rad=float(max(abs(steps))),
            continuous_core_B_phase_lag_range_ms=[float(min(lag)),float(max(lag))],
            continuous_core_B_phase_lag_mean_ms=float(lag.mean()),
            rates_range_Hz=[[float(rr[:,:,k].min()),float(rr[:,:,k].max())] for k in range(2)]))
    x=np.array([(r['J_EE_core']-root)*1e6 for r in rows]);amp=np.array([r['amplitude_hz'] for r in rows])
    turns=np.flatnonzero(np.diff(amp)[:-1]*np.diff(amp)[1:]<0)+1
    out=dict(rows=rows,amplitude_coordinate_turn_indices=turns,
        amplitude_turn_is_not_a_bifurcation=True,
        observable='Neuron-weighted core E rates on the full spatial torus. First fast-angle Fourier harmonic is interpolated at 512 slow angles; peak phase is minus its argument.',
        phase_scope='Winding and phase lag describe the core fundamental oscillation, not event onset, contact propagation rank, or a wavefront direction.',
        stability='Local birth at TR2 supported stable; finite-amplitude torus stability not computed.',
        inventory_complete=False)
    write(PERIODIC_OUT/'TR2_phase_observations.json',out)
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(2,2,figsize=(11,8));fig.subplots_adjust(left=.10,right=.97,bottom=.10,top=.94,hspace=.42,wspace=.37)
    axs[0,0].plot(x,amp,'o-',color='#756bb1',ms=3,lw=1.3)
    axs[0,0].set(ylabel='Mode projection amplitude (Hz)',title='A  Continued two-frequency solutions')
    axs[0,1].plot(x,[r['slow_period_s'] for r in rows],'o-',color='#756bb1',ms=3,lw=1.3)
    axs[0,1].set(ylabel='Slow modulation period (s)',title='B  Modulation slows along the branch')
    for k,c in enumerate(COL[:2]):
        a=np.array([r['first_harmonic_amplitude_range_Hz'][k] for r in rows])
        axs[1,0].fill_between(x,a[:,0],a[:,1],color=c,alpha=.25)
        axs[1,0].plot(x,[r['first_harmonic_amplitude_mean_Hz'][k] for r in rows],color=c,label=f'Core {"AB"[k]}')
    axs[1,0].set(yscale='log',ylabel='Fast-harmonic amplitude (Hz)',title='C  Modulation is concentrated in core B')
    axs[1,0].legend(frameon=False)
    lag=np.array([r['continuous_core_B_phase_lag_range_ms'] for r in rows])
    axs[1,1].fill_between(x,lag[:,0],lag[:,1],color=COL[1],alpha=.25)
    axs[1,1].plot(x,[r['continuous_core_B_phase_lag_mean_ms'] for r in rows],color=COL[1])
    axs[1,1].set(ylabel='Core B phase lag relative to A (ms)',title='D  Relative phase varies within a torus')
    for ax in axs.ravel():ax.set_xlabel(r'$(J-J_{\mathrm{TR2}})\times10^6$');style(ax)
    save(fig,'second_torus_continuation')
    update_readme(dict(second_torus_continuation='从 TR2 延伸的双角度解展示模态投影振幅、慢调制周期及两核快速基频振幅的调制范围；所有点均来自同一个空间 rate DDE。右下角相位差由两核加权平均 E 放电率的快速第一谐波计算，阴影为慢角度变化范围，不是置信区间。**关注点**：振幅投影折返不等于参数分岔；此图没有计算有限振幅环面的稳定性，相位差也不能替代 SEEG 事件传播 rank。'))
    print('TORUS EXTENSION OBSERVATIONS',len(rows),'projection turns',turns,'last J',rows[-1]['J_EE_core'],flush=True)


if __name__=='__main__':main()
