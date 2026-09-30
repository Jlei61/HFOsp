"""Fine observation bins for the existing supplied-input Gaussian LIF assay.

New recorder resolution only: verify exact aggregation to existing50ms counts.
This is a local validation reference, never a replacement particle network.
"""
from common import OUT,np,read,write,log
from native_input_local_lif import simulate,condition
from datetime import datetime
import argparse,os,time

DEST=OUT/'native_current_memory/millisecond_assay'
SOURCE=OUT/'native_input_bridge'


def register():
    DEST.mkdir(parents=True,exist_ok=True);assert not (DEST/'contract.json').exists()
    c=read(SOURCE/'local_lif/contract.json')
    write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),status='REGISTERED_BEFORE_FINEBIN_REFERENCE',
        question='Are native within-group millisecond coincidences and count fluctuations compatible with the conditional independentGaussianLIF reference, even where50ms mean counts agree?',
        physical='Exactly existing measured_mean_contrast:actualnativeprestepgroupnetmean,privatevarianceforcing,observedZ,groupmeanthreshold. Same groups,replicates,seed andtwo steps; only outputbin changes50msto1ms.',
        groups=c['groups'],seed=c['seed'],dt_ms=c['dt_ms'],replicates='Reuse exact perconditionmultiplesoforiginalgroupN from prior reference.',
        burn_ms=1000.,record_ms=[9000,10350],bin_ms=1,windows_ms=[[9000,9420],[9420,9868.5],[9868.5,10350]],
        controls='Sum adjacent50bins must be bitwise identical to previous50ms reference for everyLIFpath at bothsteps. Native1msaggregation must reproduce priornative50msgroupcounts.',
        statistics='For each window and1/5/50ms resolution, samefixedclock completebins only:totalcount, sum k(k-1) coincidences, count residual squared norm and lag1residualcrossmoment. Firsthalf syntheticgrouptrials define reference mean/variance; otherhalf yieldconditional95percent predictive ranges. Robust flag requires same-side exceedance of each step own reference interval; reportall6groups.',
        meaning='Conditionalmodeldiagnostic for one observednativehistory. MC batches are independent under reference inputlaw but not native networkreplicates. Native-derived input is teacherforced; no causalidentification or globalmodelpromotion.',
        budget='Six existingconditions at two steps only. No fitting, new input perturbation, timealignment, native replay or networklaunch.'))


def acquire(device):
    c=read(DEST/'contract.json');assert not (DEST/'jobs.json').exists()
    z=np.load(SOURCE/'selected_input_history.npz');raw=np.load(SOURCE/'selected_raw_variance_forcing.npz')
    assert np.array_equal(z['groups'],c['groups']) and np.array_equal(z['time_ms'],raw['time_ms'])
    m={k:z['moments'][:,j] for j,k in enumerate(z['moment_names'])};G=len(c['groups'])
    wave=np.stack([m['net'],raw['raw_variances'][:,2],raw['raw_variances'][:,3],m['z']],axis=1).transpose(2,1,0).copy()
    old=np.load(SOURCE/'local_lif/measured_mean_contrast/dt0.1.npz');nrep=old['replicates'];R=int(nrep.max())
    theta=np.broadcast_to(z['theta'][:,None],(G,R)).copy();jobs=dict(status='RUNNING',pid=os.getpid(),expected=2,completed=[]);write(DEST/'jobs.json',jobs)
    for dt in c['dt_ms']:
        pars=np.array([condition(0.,z['theta'][j],1.,1.,'EI'[int(z['population'][j])],dt=dt) for j in range(G)])
        start=time.time();counts=simulate(pars,wave,theta,nrep,dt,1000.,1350,1.,c['seed'],device)
        ref=np.load(SOURCE/f'local_lif/measured_mean_contrast/dt{dt:g}.npz')
        assert np.array_equal(counts.reshape(G,R,27,50).sum(3),ref['counts'])
        assert counts.max()<=1,'E/Irefractory permits atmostone spike perindividual1msbin'
        path=DEST/f'dt{dt:g}.npz';assert not path.exists()
        np.savez_compressed(path,counts=counts.astype('u1'),replicates=nrep,group_sizes=z['group_size'],groups=z['groups'],dt_ms=dt)
        jobs['completed'].append(dt);write(DEST/'jobs.json',jobs);log('NATIVE MILLISECOND LIF',dt,'sec',time.time()-start,'50msparityPASS')
    native=z['spikes'][(z['time_ms']>=9000)&(z['time_ms']<10350)].reshape(1350,10,G).sum(1)
    assert np.array_equal(native.reshape(27,50,G).sum(1),np.load(SOURCE/'fixed_readout.npz')['native_counts'])
    np.savez_compressed(DEST/'native_counts.npz',counts=native,bin_start_ms=np.arange(9000,10350.),groups=z['groups'],group_sizes=z['group_size'])
    jobs.update(status='COMPLETE',perpath_50ms_aggregation_bitwise=True,native_50ms_aggregation_bitwise=True);write(DEST/'jobs.json',jobs)


def metrics(counts,mu,var):
    # Rows are whole synthetic trials, not independent time samples.
    residual=counts-mu;den=max(float(var.sum()),1e-12)
    return dict(total_count=counts.sum(1),coincidence_sum=(counts*(counts-1)).sum(1),
        residual_energy=(residual**2).sum(1)/den,
        residual_lag1_crossmoment=(residual[:,:-1]*residual[:,1:]).sum(1)/den)


def score():
    assert read(DEST/'jobs.json')['status']=='COMPLETE';c=read(DEST/'contract.json');native=np.load(DEST/'native_counts.npz');rows=[]
    for dt in c['dt_ms']:
        z=np.load(DEST/f'dt{dt:g}.npz')
        for j,g in enumerate(c['groups']):
            N=int(z['group_sizes'][j]);nr=int(z['replicates'][j]);assert nr%N==0
            trials=z['counts'][j,:nr].reshape(-1,N,1350).sum(1).astype(float);cut=len(trials)//2
            for bin_ms in [1,5,50]:
                nb=1350//bin_ms;tt=np.arange(nb)*bin_ms+9000
                trial=trials.reshape(-1,nb,bin_ms).sum(2);observed=native['counts'][:,j].reshape(nb,bin_ms).sum(1).astype(float)
                for lo,hi in c['windows_ms']:
                    keep=(tt>=lo)&(tt+bin_ms<=hi);a=trial[:cut,keep];b=trial[cut:,keep];mu=a.mean(0);var=a.var(0,ddof=1)
                    measures=metrics(b,mu,var);actual=metrics(observed[keep][None],mu,var)
                    for key,samples in measures.items():
                        low,median,high=np.quantile(samples,[.025,.5,.975]);value=float(actual[key][0])
                        rows.append(dict(group=g,N=N,dt_ms=dt,bin_ms=bin_ms,window_ms=[lo,hi],complete_bins=int(keep.sum()),training_reference_trials=cut,predictive_trials=len(b),metric=key,
                            observed=value,predictive_low=float(low),predictive_median=float(median),predictive_high=float(high),outside_this_step=bool(value<low or value>high)))
    paired=[]
    for row in rows:
        if row['dt_ms']!=.1:continue
        other=next(x for x in rows if x['dt_ms']==.05 and all(x[k]==row[k] for k in ['group','bin_ms','window_ms','metric']))
        # Residual score normalization depends on its step's reference. Compare
        # each observed score to its OWN reference, do not merge raw score units.
        lower_both=row['observed']<row['predictive_low'] and other['observed']<other['predictive_low']
        upper_both=row['observed']>row['predictive_high'] and other['observed']>other['predictive_high']
        paired.append(dict(group=row['group'],N=row['N'],bin_ms=row['bin_ms'],window_ms=row['window_ms'],metric=row['metric'],
            observed_dt01=row['observed'],observed_dt005=other['observed'],outside_same_side_both_steps=bool(lower_both or upper_both),side='low' if lower_both else 'high' if upper_both else None))
    write(DEST/'result.json',dict(status='CONDITIONAL_FINEBIN_DIAGNOSTIC_COMPLETE',rows=rows,paired=paired,
        scope=c['meaning'],numerical_handling='Nativeclock0.1 and0.05 sensitivities bothreported. A robust flag requires native observedstatistic beyond same side ofeach steps OWN reference interval; no calibratednativeconfidence claim.',
        model_promoted=False,statistics_count=len(paired)))
    log('NATIVE MILLISECOND SCORE',[(r['group'],r['metric'],r['side']) for r in paired if r['bin_ms']==1 and r['window_ms'][0]==9000])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','acquire','score']);p.add_argument('--device',type=int,default=0);a=p.parse_args()
    {'register':register,'acquire':lambda:acquire(a.device),'score':score}[a.command]()
