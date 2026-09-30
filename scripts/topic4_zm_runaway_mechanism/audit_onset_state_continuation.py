"""Independent state readout and recurrence diagnostics near onset.

Rate-field recurrence proposes periods; it never certifies a full periodic
orbit, its stability, or a bifurcation. No thresholds are tuned to results.
"""
from common import OUT, model, np, read, write, log
from onset_state_continuation import DEST, OLD, regional_weights
from refractory_spatial_resolution import mapping, projections
from native_readouts import readouts, window_stats
from scipy.ndimage import uniform_filter1d
from scipy.fft import rfft, irfft, next_fast_len
from scipy.signal import find_peaks
import argparse
from pathlib import Path


def runs(mask):
    d=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))


def recurrence(r,s,max_lag_ms=2000):
    x=r.astype(float);x-=x.mean(0);x*=np.sqrt(s.sizes/s.sizes.sum())
    n=len(x);power=(x*x).sum(1);p=np.r_[0.,np.cumsum(power)]
    if power.mean()<1e-12:return dict(status='NEAR_CONSTANT_RATE_FIELD',variance=float(power.mean()))
    nfft=next_fast_len(2*n);f=rfft(x,n=nfft,axis=0)
    corr=irfft((f.conj()*f).sum(1),n=nfft)[:n]
    lag=np.arange(1,min(max_lag_ms,n//2)+1)
    d=(p[n]-p[lag]+p[n-lag]-2*corr[lag])/(n-lag)/power.mean()
    direct=[]
    for k in [20,33,min(1000,lag[-1])]:
        value=float(((x[k:]-x[:-k])**2).sum()/((n-k)*power.mean()))
        err=abs(value-d[k-1]);assert err<1e-9;direct.append(err)
    peaks,_=find_peaks(-d);peaks=peaks[lag[peaks]>=20]
    best=peaks[np.argsort(d[peaks])[:8]]
    return dict(status='RATE_RECURRENCE_ONLY',best_local_minima=[dict(lag_ms=int(lag[k]),relative_MSE=float(d[k])) for k in best],
        direct_formula_errors=direct,lags_ms=lag,relative_MSE=d,
        definition='E/I original-cell-count weighted squared rate-field recurrence, divided by temporal rate variance;1ms averaged rates. Full delayed-state closure and stability not implied.')


def audit(label,partial=False):
    folder=DEST/label;jobs=read(folder/'jobs.json');c=jobs['condition']
    if not partial:assert jobs['status']=='COMPLETE'
    s=model(40);coarse=model(20);parent,_=mapping(coarse,s);P,count=projections(s,coarse,parent)[20];W=regional_weights(s)
    R=[];F=[];T=[];M=[];tm=[];errors=[]
    prior=[]
    if c.get('previous_elapsed_ms',0):
        if c['previous_elapsed_ms']==5000 and label=='upper_endpoint':
            z=np.load(OLD/'9870/trajectory.npz')
            R.append(z['group_rate_hz']);F.append(z['field_E_hz']);T.append(z['elapsed_time_ms']);M.append(z['M_current']);tm.append(z['state_time_ms'])
        else:
            # A later same-field extension carries its actual source clock.
            # Include audited source records, never fabricate the missing
            # prefix or reset its time axis to the old5s special case.
            source=Path(c['initial']).parent;sj=read(source/'jobs.json')
            assert sj['status']=='COMPLETE'
            assert read(source/'independent_audit.json')['status']=='AUDIT_PASS'
            assert Path(c['initial']).name=='final_state.npz'
            prior=[source/f'block{b:02d}.npz' for b in sj['completed_blocks']]
            first=np.load(prior[0])['elapsed_time_ms'];last=np.load(prior[-1])['elapsed_time_ms']
            assert first[0]==1 and last[-1]==c['previous_elapsed_ms'], 'Source prefix must cover the declared complete prior history'
    Z=np.load(DEST/'fields.npz')[c['field']]
    for path in prior+[folder/f'block{block:02d}.npz' for block in jobs['completed_blocks']]:
        z=np.load(path);r=z['group_rate_hz'];field=z['field_E_hz'];t=z['elapsed_time_ms']
        assert np.array_equal(z['Z'],Z) and np.isfinite(r).all() and r.min()>=0
        err=float(abs((P@r.astype(float).T).T-field).max());assert err<5e-5
        assert np.max(abs(r@W.T-z['regional_rate_hz']))<1e-9
        errors.append(err);R.append(r);F.append(field);T.append(t);M.append(z['M_current']);tm.append(z['state_time_ms'])
    if not jobs['completed_blocks']:return
    r=np.concatenate(R);field=np.concatenate(F);t=np.concatenate(T);m=np.concatenate(M);state_t=np.concatenate(tm)
    assert np.array_equal(np.diff(t),np.ones(len(t)-1))
    events,summary,whole,sm=readouts(t,field.astype(float),count,label)
    complete=[]
    for ev in events:
        a=int(np.searchsorted(t,ev['start_ms']));b=a+int(ev['duration_ms'])
        if a>=20 and b+20<=len(sm) and (sm[a-20:a]<5).all() and (sm[b:b+20]<5).all():complete.append(ev)
    windows=[]
    for lo in range(int(t[0]-1),int(t[-1]),5000):
        mask=(t>lo)&(t<=lo+5000)
        if mask.sum()!=5000:continue
        mm=(state_t>lo)&(state_t<=lo+5000);smm=sm[mask]
        quiet=[b-a for a,b in runs(smm<5) if b-a>=20]
        rr=r[mask].astype(float);ff=field[mask];means=rr@W.T
        windows.append(dict(window_ms=[lo,lo+5000],mean_rates_global_A_B_surround=means.mean(0).tolist(),
            smoothed_global_range_hz=[float(smm.min()),float(smm.max())],quiet_fraction=float((smm<5).mean()),
            longest_quiet_ms=max(quiet,default=0),complete_events=window_stats(complete,lo,lo+5000),
            persistent_spatial_fraction=float((count/count.sum())[(ff>50).mean(0)>=.9].sum()),
            group_rate_relative_variation=float(np.linalg.norm(rr-rr.mean(0))/max(np.linalg.norm(rr),1)),
            M_mean_global_A_B_surround=(m[mm]@W.T).mean(0).tolist()))
    rec=recurrence(r[-5000:],s)
    if rec['status']=='RATE_RECURRENCE_ONLY':
        np.savez_compressed(folder/'rate_recurrence.npz',lag_ms=rec.pop('lags_ms'),relative_MSE=rec.pop('relative_MSE'))
    result=dict(status='PARTIAL_AUDIT_PASS' if partial else 'AUDIT_PASS',label=label,D=float(1-Z[s.E]@s.mean_weights),
        observed_ms=float(t[-1]-t[0]+1),spatial_reconstruction_errors_hz=errors,windows=windows,
        original_high_onset_elapsed_ms=summary['high_onset_ms'],
        complete_events_all=window_stats(complete,t[0],t[-1]+1),recurrence=rec,
        scope='Longer finite-time state/recurrence readout; no certified attractor, periodic stability, separatrix or bifurcation.',model_promoted=False)
    write(folder/('partial_audit.json' if partial else 'independent_audit.json'),result)
    log('ONSET AUDIT',label,result['status'],windows[-1],rec.get('best_local_minima',[])[:3])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--partial',action='store_true');a=p.parse_args();audit(a.label,a.partial)
