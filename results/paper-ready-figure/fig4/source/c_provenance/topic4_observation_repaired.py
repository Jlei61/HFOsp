"""Fixed-reference contact observer; independent contacts count once per frame.

Both time descriptors use the same fixed event windows and participation mask.
Recruitment is auxiliary and is NOT compared to patient lagPat centroids as if
they were the same measurement. No model results choose the observer settings.
"""
import numpy as np
from scipy.ndimage import maximum_filter1d


def runs(mask):
    edges=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(int))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def calibrate_reference(envelope,dt_ms,contact_names,*,cal_start_ms=500.,cal_stop_ms=6000.):
    env=np.asarray(envelope,float)
    if env.ndim!=2 or not np.isfinite(env).all() or dt_ms<=0:
        raise ValueError('finite contact x frame reference required')
    cal=env[:,int(cal_start_ms/dt_ms):int(cal_stop_ms/dt_ms)]
    if cal.shape[1]<1000 or len(contact_names)!=len(env):raise ValueError('insufficient reference')
    # Reference-only quiet frames define baseline/noise; robust signal amplitude
    # guards against near-zero simulated shot-noise floors. All constants precede
    # scoring current parameter arms; neither is re-estimated per candidate.
    aggregate=np.mean(cal,axis=0)
    quiet=aggregate<=np.quantile(aggregate,.25)
    baseline=np.median(cal[:,quiet],axis=1)
    mad=np.median(abs(cal[:,quiet]-baseline[:,None]),axis=1)*1.4826
    high=np.quantile(cal,.995,axis=1)
    increment=np.maximum(6*mad,.1*(high-baseline))
    if np.any(increment<=0):raise ValueError('reference cannot calibrate a positive detection scale')
    return {'version':'fixed_reference_unique_contact_observer_v2',
            'contact_names':list(contact_names),'dt_ms':float(dt_ms),
            'baseline':baseline.tolist(),'threshold':(baseline+increment).tolist(),
            'reference_high_q995':high.tolist(),'noise_sigma_MAD':mad.tolist(),
            'calibration_window_ms':[cal_start_ms,cal_stop_ms],
            'threshold_rule':'baseline + max(6 quiet MAD sigma, 0.1 * (reference q99.5 - baseline))',
            'minimum_detection_ms':4.,'extension_ms':30.,'window_ms':250.,
            'channel_fraction':.5,'burnin_ms':500.,'maximum_centroid_mass_outside_local_detection':'reported, not filtered'}


def observe(envelope,dt_ms,contract,*,threshold_scale=1.):
    env=np.asarray(envelope,float);c,n=env.shape
    if not np.isfinite(env).all() or dt_ms!=contract['dt_ms'] or c!=len(contract['contact_names']):
        raise ValueError('observer data shape, units or sampling changed')
    if threshold_scale<=0:raise ValueError('positive threshold scale required')
    base=np.asarray(contract['baseline']);bar=base+threshold_scale*(np.asarray(contract['threshold'])-base)
    detected=env>bar[:,None]
    minimum=max(1,int(np.ceil(contract['minimum_detection_ms']/dt_ms)))
    for ci in range(c):
        for a,b in runs(detected[ci]):
            if b-a<minimum:detected[ci,a:b]=False
    pad=int(round(contract['extension_ms']/dt_ms))
    # Union within channel BEFORE counting across channels, including flutter.
    expanded=maximum_filter1d(detected.astype(np.uint8),2*pad+1,axis=1,mode='constant')>0
    unique=expanded.sum(0)
    required=int(np.ceil(c*contract['channel_fraction']))
    intervals=runs(unique>=required)
    # Join brief dips in the population criterion, without counting a contact twice.
    merged=[]
    for a,b in intervals:
        if merged and a-merged[-1][1]<=pad:merged[-1][1]=b
        else:merged.append([a,b])
    half=contract['window_ms']/2
    events=[];censored=[]
    for a,b in merged:
        anchor=(a+b)*dt_ms/2
        start=int(np.floor((anchor-half)/dt_ms));stop=start+int(round(2*half/dt_ms))
        if start*dt_ms<contract['burnin_ms'] or stop>n:
            censored.append([start*dt_ms,stop*dt_ms]);continue
        mask=detected[:,start:stop].any(1)
        # Padding may recruit a contact outside the final comparison window.
        # Such cases remain audit records, not accepted half-montage events.
        if mask.sum()<required:
            censored.append([start*dt_ms,stop*dt_ms]);continue
        events.append({'window_ms':[start*dt_ms,stop*dt_ms],
            'qualifying_interval_ms':[a*dt_ms,b*dt_ms],
            'prolonged':(b-a)*dt_ms>contract['window_ms'],
            'n_unique_contacts':int(mask.sum()),'start':start,'stop':stop,'mask':mask})
    centroid=np.full((len(events),c),np.nan);recruitment=centroid.copy()
    for k,event in enumerate(events):
        a,b=event.pop('start'),event.pop('stop');mask=event.pop('mask')
        local_baseline=np.maximum(base,np.quantile(env[:,a:b],.1,axis=1))
        segment=np.maximum(env[:,a:b]-local_baseline[:,None],0.)
        event['local_baseline']=local_baseline.tolist()
        times=(np.arange(a,b)+.5)*dt_ms
        first=[];multi=[];below=[]
        for ci in np.flatnonzero(mask):
            weights=segment[ci]
            if weights.sum()>0:centroid[k,ci]=weights@times/weights.sum()
            rr=runs(detected[ci,a:b]);first.append(rr[0][0])
            recruitment[k,ci]=(a+rr[0][0]+.5)*dt_ms
            multi.append(len(rr)>1)
            below.append(float(weights[~detected[ci,a:b]].sum()/weights.sum()) if weights.sum()>0 else 1.)
        event['multiple_local_bursts_contact_fraction']=float(np.mean(multi))
        event['median_mass_outside_detection']=float(np.median(below))
        event['overlap_with_other_windows']=any(j!=k and other['window_ms'][0]<event['window_ms'][1] and other['window_ms'][1]>event['window_ms'][0] for j,other in enumerate(events))
        # The frozen patient table excludes BOTH overlapping windows. Preserve
        # that comparison population; retain every detection above for auditing
        # event yield / crowded activity, never silently hide those trajectories.
        event['primary_exclusion_reasons'] = (
            (['overlapping_window'] if event['overlap_with_other_windows'] else [])
            + (['prolonged_activity'] if event['prolonged'] else [])
            + (['insufficient_estimable_centroids'] if np.isfinite(centroid[k]).sum()<required else []))
        event['primary_eligible'] = not event['primary_exclusion_reasons']
    primary=np.array([e['primary_eligible'] for e in events],dtype=bool)
    return {'centroid_ms':centroid,'recruitment_ms':recruitment,
            'primary_event_indices':np.flatnonzero(primary).tolist(),
            'n_primary_events':int(primary.sum()),
            'primary_selection':'isolated 250 ms windows; both overlapping windows excluded as in frozen patient table; prolonged activity audited separately',
            'events':events,'windows_ms':[e['window_ms'] for e in events],
            'n_groups':len(events),'boundary_or_low_window_support':censored,
            'threshold':bar.tolist(),'required_unique_contacts':required,
            'threshold_scale':threshold_scale,'time_units':'ms, bin centers',
            'recruitment_role':'auxiliary first sustained threshold crossing; not patient centroid ground truth',
            'prolonged_or_multiburst_events_retained':True}


def order_distribution(lags,tolerance_ms=2.):
    lag=np.asarray(lags,float);lag=lag[np.isfinite(lag)]
    if not len(lag):return None
    return np.array([np.mean(lag < -tolerance_ms),np.mean(abs(lag)<=tolerance_ms),np.mean(lag>tolerance_ms)])


def order_error(left,right,tolerance_ms=2.):
    a,b=order_distribution(left,tolerance_ms),order_distribution(right,tolerance_ms)
    return None if a is None or b is None else float(.5*np.sum(abs(a-b)))
