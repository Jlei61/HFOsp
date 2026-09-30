"""Resolve unequal intervals versus irregularity, and halve the time step."""
from rate_field import *
from run_rate_field import dynamics
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    rows=[]
    for J in [.942,1.3]:
        a=np.load(RATE_OUT/f'runs/main/J{J:.7f}/trajectory.npz');b=np.load(RATE_OUT/f'runs/half_step/J{J:.7f}/trajectory.npz')
        ra=gaussian_filter1d(a['regional_rates_hz'],5,axis=0);rb=gaussian_filter1d(b['regional_rates_hz'],5,axis=0)
        ma=dynamics(ra,2000);mb=dynamics(rb,2000)
        rows.append(dict(J_EE_core=J,dt_ms=[.1,.05],
            mean_rate_relative_difference=abs(ra[2000:].mean(0)-rb[2000:].mean(0))/ra[2000:].mean(0),
            waveform_relative_rms=np.linalg.norm(ra[2000:]-rb[2000:],axis=0)/np.linalg.norm(ra[2000:],axis=0),
            burst_count_dt01=[x['self_limited_bursts'] for x in ma],burst_count_dt005=[x['self_limited_bursts'] for x in mb],
            IEI_CV_dt01=[x['IEI_CV'] for x in ma],IEI_CV_dt005=[x['IEI_CV'] for x in mb]))
    z=np.load(RATE_OUT/'runs/long/J0.9420000/trajectory.npz');smooth=gaussian_filter1d(z['regional_rates_hz'],5,axis=0);d=dynamics(smooth,10000)
    recurrence=[]
    for k in [0,1]:
        p=np.array(d[k]['peak_times_ms']);iei=np.diff(p)
        recurrence.append(dict(core='AB'[k],analysis_window_ms=[10000,30000],burst_count=d[k]['self_limited_bursts'],IEI_CV=d[k]['IEI_CV'],
            last_20_intervals_ms=iei[-20:],two_interval_recurrence_RMSE_ms=float(np.sqrt(np.mean((iei[2:]-iei[:-2])**2))),
            interpretation='Long-short interval alternation, not evidence of irregular/chaotic events or a classified period-doubling bifurcation'))
    group=z['group_rate_hz'][20000:];recurrence_field=float(np.linalg.norm(group[785:]-group[:-785])/np.linalg.norm(group[:-785]))
    result=dict(step_halving=rows,long_run=recurrence,full_field_recurrence_785ms_relative_error=recurrence_field,
        interpretation='The middle case approaches a repeating two-burst pattern. CV>0 is insufficient to label it irregular.',
        deterministic=True,native_equivalence='NOT_VALIDATED')
    assert all(max(q['mean_rate_relative_difference'])<.01 for q in rows)
    assert all(q['burst_count_dt01']==q['burst_count_dt005'] for q in rows)
    write(RATE_OUT/'temporal_checks.json',result)
    fig,axes=plt.subplots(3,2,figsize=(13,9),layout='constrained');colors=['#2267ac','#b04da7']
    for k in [0,1]:
        p=np.array(d[k]['peak_times_ms']);iei=np.diff(p)
        axes[0,k].plot(z['time_ms']/1000,smooth[:,k],color=colors[k],lw=.8);axes[0,k].set(xlim=(20,24),xlabel='Time (s)',ylabel='Rate (Hz / cell)',title=f'Core {"AB"[k]}: J=0.942')
        axes[1,k].plot(p[1:]/1000,iei,'-o',ms=3,color=colors[k]);axes[1,k].set(xlabel='Burst time (s)',ylabel='Inter-burst interval (ms)',ylim=(280,480))
        axes[2,k].scatter(iei[:-1],iei[1:],s=22,color=colors[k]);axes[2,k].plot([280,480],[280,480],':',color='black',lw=.7)
        axes[2,k].set(xlim=(280,480),ylim=(280,480),xlabel='Interval n (ms)',ylabel='Interval n+1 (ms)')
    for ax in axes.ravel():ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Autonomous rate model: alternating intervals persist after the transient',fontsize=14)
    f=RATE_OUT/'figures';f.mkdir(exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(f/f'rate_alternating_intervals.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig);print('TEMPORAL',result,flush=True)


if __name__=='__main__':main()
