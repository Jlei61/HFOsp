"""Finite-record native burst phenotypes; labels are operational, not bifurcations."""
import numpy as np
from scipy.signal import correlate, welch

LABELS = ['background','sparse_bursts','irregular_bursts','regular_bursts',
          'variable_bursts','sustained_activity','active_without_bursts','insufficient_record']

def intervals(onsets):
    d=np.diff(np.asarray(onsets,float))
    d=d[d>0]
    if len(d)<2:return dict(iei_median_s=None,cv=None,cv2=None,lag1_log_iei_r=None,half_median_ratio=None)
    cv=float(d.std(ddof=1)/d.mean())
    cv2=float(np.mean(2*abs(np.diff(d))/(d[:-1]+d[1:])))
    lag=float(np.corrcoef(np.log(d[:-1]),np.log(d[1:]))[0,1]) if len(d)>=5 and np.std(d[:-1])>1e-12 and np.std(d[1:])>1e-12 else None
    h=len(d)//2;left=np.median(d[:h]);right=np.median(d[h:])
    return dict(iei_median_s=float(np.median(d)),cv=cv,cv2=cv2,lag1_log_iei_r=lag,
        half_median_ratio=float(max(left,right)/min(left,right)))

def segments(fraction, *, onset_threshold=.10, offset_threshold=.03, frame_s=.01, merge_gap_s=.02):
    """Hysteretic episodes; both support thresholds refer to fixed neuron fractions."""
    f=np.asarray(fraction,float)
    above=f>=offset_threshold
    edges=np.diff(np.r_[False,above,False].astype(int))
    starts=np.flatnonzero(edges==1);stops=np.flatnonzero(edges==-1)
    joined=[]
    for start,stop in zip(starts,stops):
        if joined and (start-joined[-1][1])*frame_s<=merge_gap_s+1e-12:joined[-1][1]=int(stop)
        else:joined.append([int(start),int(stop)])
    rows=[]
    for start,stop in joined:
        if np.max(f[start:stop])<onset_threshold:continue
        rows.append(dict(start_s=start*frame_s,stop_s=stop*frame_s,duration_s=(stop-start)*frame_s,
            peak_fraction=float(np.max(f[start:stop])),left_censored=start==0,right_censored=stop==len(f)))
    return rows

def summarize(fraction10, counts2, size, *, burnin_s=2., runaway=False, onset_threshold=.10):
    fraction10=np.asarray(fraction10,float)
    counts2=np.asarray(counts2,float)
    f=fraction10[round(burnin_s/.01):]
    rates=counts2[round(burnin_s/.002):]/max(size,1)/.002
    duration=len(f)*.01
    out=dict(observed_s=duration,n_neurons=int(size),onset_threshold=onset_threshold,
        mean_rate_hz=float(rates.mean()) if len(rates) else None,
        max_fraction_10ms=float(f.max()) if len(f) else None)
    if duration<1.:
        out.update(label='sustained_activity' if runaway else 'insufficient_record',n_bursts=0,events=[],
            **intervals([]),burst_duty=None,max_burst_s=None,acf_peak=None,spectral_peak_hz=None)
        return out
    events=segments(f,onset_threshold=onset_threshold)
    onsets=[e['start_s'] for e in events if not e['left_censored']]
    im=intervals(onsets)
    # ACF on native activity is a complementary description, not a test of an oscillator.
    centered=f-f.mean();ac=correlate(centered,centered,mode='full',method='fft')[len(f)-1:]
    ac=ac/ac[0] if ac[0]>0 else ac*0
    median=im['iei_median_s']
    peak=None
    if median is not None:
        lo=max(5,round(median*.65/.01));hi=min(len(ac)-1,round(median*1.5/.01))
        if hi>=lo:peak=float(np.max(ac[lo:hi+1]))
    freq,power=welch(f,fs=100.,nperseg=min(400,len(f)),noverlap=min(200,len(f)//2))
    sel=(freq>=.25)&(freq<=15.)
    spectral=float(freq[sel][np.argmax(power[sel])]) if sel.any() and power[sel].sum()>0 else None
    duty=sum(e['duration_s'] for e in events)/duration
    longest=max([e['duration_s'] for e in events],default=0.)
    n=len(onsets)
    if runaway or longest>=1. or duty>=.75:
        label='sustained_activity'
    elif not events:
        label='background' if out['mean_rate_hz']<5. else 'active_without_bursts'
    elif n<8:
        label='sparse_bursts'
    elif im['cv']<=.30 and im['cv2']<=.40 and im['half_median_ratio']<=1.5 and peak is not None and peak>=.20:
        label='regular_bursts'
    elif im['cv']>=.40 and im['cv2']>=.50:
        label='irregular_bursts'
    else:
        label='variable_bursts'
    out.update(label=label,n_bursts=n,events=events,**im,burst_duty=float(duty),max_burst_s=float(longest),
        acf_peak=peak,spectral_peak_hz=spectral)
    return out

def run_metrics(path, result):
    out={}
    with np.load(path) as z:
        names=z['group_names'].tolist();sizes=z['group_sizes']
        for group in ['coreAE','coreBE','coreUnionE','allE','surroundE']:
            j=names.index(group);size=int(sizes[j])
            f=z['active_counts_10ms'][:,j]/max(size,1)
            metric=summarize(f,z['spike_counts_2ms'][:,j],size,burnin_s=result['burnin_ms']/1000,
                runaway=result['runaway_early_stop_ms'] is not None)
            alternatives=[]
            for threshold in (.075,.125):
                alternatives.append(summarize(f,z['spike_counts_2ms'][:,j],size,
                    burnin_s=result['burnin_ms']/1000,runaway=result['runaway_early_stop_ms'] is not None,
                    onset_threshold=threshold)['label'])
            metric['threshold_sensitivity_labels']=alternatives
            metric['threshold_stable']=all(x==metric['label'] for x in alternatives)
            fractions=z['active_counts_2ms'][:,j]/max(size,1)
            peaks=[]
            for e in metric['events']:
                a=round((e['start_s']+result['burnin_ms']/1000)/.002)
                b=round((e['stop_s']+result['burnin_ms']/1000)/.002)
                if b>a:peaks.append(float(np.max(fractions[a:b])))
            metric['median_peak_active_fraction_2ms']=float(np.median(peaks)) if peaks else None
            out[group]=metric
    return out
