"""Compare autonomous density trajectories to the finite-g40 controls.

Projected recurrence is a candidate diagnostic only. It cannot certify a
periodic orbit of the complete PDF, delayed currents and dynamic M state.
"""
from analyze_qualification import *
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar


def main():
    selected=OUT/'qualification/selected_g40';rows=[]
    for D,duration in [(.225,4000),(.25,4000)]:
        folder=selected/f'D{D:.6f}_degree6_dv0.125_{duration}ms'
        if not (folder/'status.json').exists():continue
        status=json.load(open(folder/'status.json'))
        if status['status']!='COMPLETE':continue
        config=json.load(open(folder/'config.json'));initial=config.get('initial_ms',0.)
        with np.load(folder/'trajectory.npz') as z:data={k:z[k] for k in z.files}
        rates=data['rate_1ms'];field=data['field_1ms'];count=data['count_e']
        late=rates[-2000:];peaks,_=find_peaks(late[:,0],height=20,prominence=10,distance=50)
        intervals=np.diff(peaks)
        # Resolve the same 1--4 second interval as the matched particle controls.
        summary=summarize(folder,1000,4000)
        stats=dict(D=D,source=str(folder),statistics_1_to_4s=summary,
            final_2s=dict(mean_rate=late.mean(0),minimum_rate=late.min(0),maximum_rate=late.max(0),
            peak_times_ms=(len(rates)-2000+peaks+initial),peak_intervals_ms=intervals,
            peak_interval_cv=intervals.std()/intervals.mean() if len(intervals)>1 else None),
            complete_state_periodic_orbit='NOT_ESTABLISHED')
        if len(peaks)>4 and intervals.mean()>50:
            signals=np.c_[rates,field@count/32000.,data['slow_10ms'][-1,1]*np.ones(len(rates))]
            # Only the first four signals vary; constant columns are not used
            # to reduce the denominator or make an apparent recurrence easier.
            signals=signals[:,:4]
            left=max(0,len(signals)-1500);end=len(signals)-int(np.ceil(intervals.mean()*1.2))-2
            ts=np.arange(left,end)
            def recurrence(period):
                shifted=np.stack([np.interp(ts+period,np.arange(len(signals)),signals[:,j]) for j in range(4)],axis=1)
                base=signals[ts]
                return np.sum((shifted-base)**2)/max(np.sum((base-base.mean(0))**2),1e-20)
            fit=minimize_scalar(recurrence,bounds=(.8*intervals.mean(),1.2*intervals.mean()),method='bounded')
            stats['projected_recurrence']=dict(period_ms=fit.x,relative_RMS_error=np.sqrt(fit.fun),
                interpretation='Four rate observables only; not a full-state periodicity test')
        rows.append(stats)
        with np.load(folder/'checkpoint.npz') as z:
            # Use the complete delayed firing history as an independent root
            # predictor for the original equations, not the plot mean itself.
            seed=z['history'].mean(0)*10000.
        seed_folder=OUT/'equilibrium_predictors'/f'density_terminal_D{D:g}'
        seed_folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(seed_folder/'seed.npz',D=D,rate_hz=seed)
        print(D,summary['category'],summary['mean_E_hz'],summary['finite_events'],stats.get('projected_recurrence'),flush=True)
    comparison=json.load(open(OUT/'grouping_control_comparison.json'))
    result=dict(status='COMPLETE_DIAGNOSTIC' if len(rows)==2 else 'WAITING_FOR_TRAJECTORIES',density=rows,
        finite_particle_controls=comparison['pairs'],
        inference='Same selected-g40 coupling, stationary private Poisson noise and zero common OU. Finite-size fluctuations distinguish particle and deterministic density dynamics.',
        periodic_branch_acceptance='NOT_ESTABLISHED')
    (OUT/'density_attractor_comparison.json').write_text(json.dumps(safe(result),indent=2)+'\n')


if __name__=='__main__':main()
