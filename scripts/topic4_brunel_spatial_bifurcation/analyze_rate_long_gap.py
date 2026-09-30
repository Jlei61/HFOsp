"""Long-window rhythm and full-space recurrence of an autonomous rate run.

Nonzero IEI CV, finite recurrence mismatch and a return-map cloud do not by
themselves distinguish a torus, chaos, long period, or a remaining transient.
"""
from plot_rate_periodic_completion import *
from scipy.ndimage import gaussian_filter1d
from scipy.fft import rfft,irfft,next_fast_len
from scipy.optimize import minimize_scalar


def recurrence(x,maxlag=5000):
    x=np.asarray(x,dtype=float);n,p=x.shape
    maxlag=min(maxlag,n//3);nfft=next_fast_len(2*n)
    power=np.zeros(nfft//2+1)
    for j in range(0,p,64):
        f=rfft(x[:,j:j+64],n=nfft,axis=0)
        power+=(f.real*f.real+f.imag*f.imag).sum(1)
    corr=irfft(power,n=nfft)[:maxlag+1]
    q=np.r_[0.,np.cumsum(np.einsum('ij,ij->i',x,x))]
    lag=np.arange(100,maxlag+1);left=q[n-lag];right=q[n]-q[lag]
    curve=np.sqrt(np.maximum(left+right-2*corr[lag],0)/left)
    mins=find_peaks(-curve,distance=80)[0]
    best=sorted(mins,key=lambda i:curve[i])[:5];checks=[]
    # Refine on one common physical window, using linear interpolation of
    # the recorded 1-ms averages; no separate shifts of individual groups.
    a=x[:n-maxlag-2];norm=np.linalg.norm(a)
    def error(delay):
        k=int(np.floor(delay));frac=delay-k
        b=x[k:k+len(a)]*(1-frac)+x[k+1:k+1+len(a)]*frac
        return np.linalg.norm(b-a)/norm
    for i in best:
        delay=lag[i];fit=minimize_scalar(error,bounds=(delay-1,delay+1),method='bounded',options={'xatol':1e-5})
        checks.append(dict(lag_ms=float(fit.x),relative_full_group_mismatch=float(fit.fun),integer_lag_ms=int(delay)))
    return lag,curve,checks


def main(path):
    assert read(Path(path).parent/'contract.json')['J_EE_core']==.946
    dest=PERIODIC_OUT;z=np.load(path);time=z['time_ms'];reg=z['regional_rates_hz'];groups=z['group_rate_hz']
    duration=int(time[-1]);assert duration>=150000
    rows=[];checks=[];curves=[];smooth=gaussian_filter1d(reg,5,axis=0)
    for start,end in [(20000,30000),(50000,100000),(100000,150000),(150000,min(200000,duration))]:
        for k in [0,1]:
            pk=find_peaks(smooth[start:end,k],height=10,prominence=10,distance=60)[0]+start
            intervals=np.diff(pk);heights=smooth[pk,k]
            rows.append(dict(core='AB'[k],window_ms=[start,end],peaks_ms=pk+1,
                peak_amplitudes_hz=heights,peak_count=len(pk),IEI_ms=intervals,
                IEI_CV=float(np.std(intervals)/np.mean(intervals)) if len(intervals)>1 else None,
                median_IEI_ms=float(np.median(intervals)) if len(intervals) else None))
        stop=end;begin=max(start,stop-20000)
        lag,curve,candidates=recurrence(groups[begin:stop]);curves.append((begin,stop,lag,curve))
        checks.append(dict(window_ms=[begin,stop],candidates=candidates))
    old=RATE_OUT/'runs/periodic_gap/J0.9460000/trajectory.npz';prefix=None
    if old.exists():
        a=np.load(old)['group_rate_hz'];prefix=dict(previous_source=str(old),samples=len(a),
            maximum_group_rate_difference_Hz=float(abs(a-groups[:len(a)]).max()),
            meaning='Same zero-state trajectory prefix, ordinary stepping versus graph execution; not an independent run.')
    from expanded_readouts import OLD,observer,smooth2,describe
    contract=read(OLD/'observer_firing.json');names=contract['contact_names']
    assert z['contact_names'].tolist()==names
    n=duration//2*2;contact=z['contact_rate_hz'][:n].reshape(-1,2,15).sum(1)/1000
    ob=observer.observe(smooth2(contact).T,2.,contract);mu=np.asarray(ob['centroid_ms'],float).reshape(-1,15)
    ids=np.array([i for i in ob['primary_event_indices'] if ob['events'][i]['window_ms'][0]>=100000 and ob['events'][i]['window_ms'][1]<=200000],int)
    obs=dict(window_ms=[100000,200000],metrics=describe(mu[ids],names),qualified_centroids_ms=mu[ids],
        observer_source=str(OLD/'observer_firing.json'),statistical_unit='One deterministic trajectory; detected events are not independent network replicates.')
    write(dest/'gap_J0p946_long_diagnostics.json',dict(source=str(path),duration_ms=duration,rows=rows,
        full_space_recurrence=checks,ordinary_step_prefix_check=prefix,contact_observer=obs,
        attractor_class='NOT_ESTABLISHED',
        interpretation='Finite-time dynamics only. A small candidate mismatch motivates an independent periodic BVP; large mismatch is not proof of chaos or a torus.'))
    fig,axs=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    for k in [0,1]:
        axs[0,0].plot(time[-10000:]/1000,reg[-10000:,k],color=COL[k],lw=.6,label=f'Core {"AB"[k]}')
        for q in [r for r in rows if r['core']=='AB'[k]]:
            axs[0,1].plot(np.array(q['peaks_ms'])[1:]/1000,q['IEI_ms'],'.',color=COL[k],ms=2)
        q=[r for r in rows if r['core']=='AB'[k]][-1];amp=np.array(q['peak_amplitudes_hz'])
        axs[0,2].plot(amp[:-1],amp[1:],'.',color=COL[k],ms=3)
    axs[0,0].set(xlabel='Time (s)',ylabel='Rate (Hz / cell)',title='J=0.946: final 10 s');axs[0,0].legend(frameon=False)
    axs[0,1].set(xlabel='Time (s)',ylabel='Burst interval (ms)',title='Consecutive intervals across windows')
    axs[0,2].set(xlabel='Peak n (Hz / cell)',ylabel='Peak n+1 (Hz / cell)',title='150–200 s return map')
    for ax,(begin,stop,lag,curve) in zip(axs[1],curves[-3:]):
        ax.plot(lag,curve,color='#333333',lw=1)
        ax.set(xlabel='Candidate lag (ms)',ylabel='Relative full-space mismatch',title=f'All 935 groups: {begin/1000:g}–{stop/1000:g} s')
    for ax in axs.ravel():style(ax)
    save(fig,'rate_gap_J0p946_long_dynamics')
    update_readme(dict(rate_gap_J0p946_long_dynamics='展示固定 J=0.946 的约 200 秒自主 rate 仿真，包括末段波形、分窗 burst 间隔、峰值返回图及全部 935 群体的重复误差。使用同一冻结模型与触点观察器，未注入原 SNN 活动。**关注点**：有限时窗统计不直接判定混沌、环面或稳定不规则吸引子；候选重复周期仍须周期边值及 Floquet 验证。'))
    print('LONG GAP',[(r['core'],r['window_ms'],r['peak_count'],r['IEI_CV']) for r in rows],checks,prefix,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trajectory');a=p.parse_args();main(Path(a.trajectory))
