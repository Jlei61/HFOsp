"""Finite nonlinear departure from a Floquet-unstable small periodic orbit."""
from plot_rate_periodic_completion import *
from scipy.ndimage import gaussian_filter1d


def main():
    folder=RATE_OUT/'runs/unstable_cycle_departure/J0.9460000'
    assert read(folder/'result.json')['status']=='COMPLETE'
    z=np.load(folder/'trajectory.npz');t=z['time_ms'];r=z['regional_rates_hz']
    smooth=gaussian_filter1d(r,5,axis=0);rows=[]
    for start in range(0,int(t[-1]),10000):
        end=min(start+10000,len(t));core=[]
        for k in [0,1]:
            peaks=find_peaks(smooth[start:end,k],height=10,prominence=10,distance=60)[0]+start
            intervals=np.diff(t[peaks])
            core.append(dict(core='AB'[k],mean_Hz=float(r[start:end,k].mean()),
                minimum_Hz=float(r[start:end,k].min()),maximum_Hz=float(r[start:end,k].max()),
                burst_peak_count=len(peaks),median_interval_ms=float(np.median(intervals)) if len(intervals) else None,
                interval_CV=float(np.std(intervals)/np.mean(intervals)) if len(intervals)>1 else None))
        rows.append(dict(window_ms=[start,end],core=core))
    final=r[-20000:];first=np.flatnonzero(smooth[:,:2].max(1)>=10)
    late=r[-30000:]
    pa=find_peaks(late[:,0],prominence=.2,distance=80)[0]
    pb=find_peaks(late[:,1],prominence=.05,distance=80)[0]
    phases=[]
    for peak in pb:
        i=np.searchsorted(pa,peak)-1
        if 0<=i<len(pa)-1:phases.append((peak-pa[i])/(pa[i+1]-pa[i]))
    weak_rhythm=dict(window_ms=[max(0,len(t)-30000),len(t)],
        mean_peak_intervals_ms=[float(np.mean(np.diff(peaks))) for peaks in [pa,pb]],
        core_B_peak_phase_in_A_cycle_range=[float(min(phases)),float(max(phases))] if phases else None,
        definition='Small-oscillation timing only: raw 1ms averaged rates, A/B prominences .2/.05Hz, separation 80ms. These peaks are not burst events.',
        interpretation='Different mean frequencies and a drifting relative phase describe finite-time weak modulation; no quasiperiodic-attractor proof.')
    out=dict(source=str(folder/'trajectory.npz'),initial_condition=read(PERIODIC_OUT/'unstable_cycle_departure/J0.9460000_dt0.05_a0.01/contract.json'),
        duration_ms=int(t[-1]),windows=rows,first_smoothed_core_rate_above_10Hz_ms=int(t[first[0]]) if len(first) else None,
        final_20s_max_rates_Hz=final.max(0),
        weak_rhythm=weak_rhythm,
        event_definition='Core E rate averaged per neuron, 5ms Gaussian smoothing; peaks with height and prominence >=10Hz, separation >=60ms. Descriptive assay, not a bifurcation criterion.',
        attractor_class='NOT_ESTABLISHED',
        scope='One 180s deterministic initial-condition experiment at fixed J=.946, no injected noise or native spikes. A finite departure does not prove a global invariant-manifold connection or stable irregular attractor.')
    write(PERIODIC_OUT/'unstable_cycle_departure_diagnostics.json',out)
    fig,axs=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
    for k in [0,1]:
        for ax,mask in [(axs[0],t<=3000),(axs[2],t>t[-1]-3000)]:
            ax.plot(t[mask]/1000,r[mask,k],color=COL[k],lw=.8,label=f'Core {"AB"[k]}')
        centers=np.array([np.mean(q['window_ms']) for q in rows])/1000
        low=[q['core'][k]['minimum_Hz'] for q in rows];high=[q['core'][k]['maximum_Hz'] for q in rows]
        axs[1].fill_between(centers,low,high,color=COL[k],alpha=.22)
        axs[1].plot(centers,[q['core'][k]['mean_Hz'] for q in rows],color=COL[k],lw=1)
    for ax,title in zip(axs,['Perturbed weak cycle: first 3 s','10 s windows: mean and range','Final 3 s']):
        ax.set(xlabel='Time (s)',ylabel='Core E rate (Hz / cell)',title=title);style(ax)
    axs[0].legend(frameon=False)
    save(fig,'weak_unstable_cycle_nonlinear_departure')
    update_readme(dict(weak_unstable_cycle_nonlinear_departure='固定 J=0.946，在精确弱周期轨道上沿已核实的不稳定 Floquet 方向加入 0.01 Hz 初始扰动，保留全部状态及延迟历史，自主运行约 180 秒。图示起始、分窗范围及末段两核活动。**关注点**：这是确定性有限时长的离轨实验，不直接证明全局分支连接或稳定不规则吸引子。'))
    print('DEPARTURE',out['first_smoothed_core_rate_above_10Hz_ms'],out['final_20s_max_rates_Hz'],rows[-3:],flush=True)


if __name__=='__main__':main()
