"""Join exact same-Z continuations and quantify local stop/re-entry timing."""
from common import OUT,np,read,write,model
from onset_state_continuation import regional_weights
from scipy.ndimage import uniform_filter1d
from audit_onset_state_continuation import recurrence
import argparse

DEST=OUT/'core_a_transition_continuation_20260924'


def runs(mask):
    d=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))


def audit(side):
    labels=[f'mid1_from_{side}',f'mid1_from_{side}_ext',f'mid1_from_{side}_ext_long']
    s=model(40);W=regional_weights(s);regional=[];M=[];tm=[];sources=[];offset=0;old=None
    Z=np.load(DEST/'fields.npz')['mid1']
    for name in labels:
        folder=DEST/name;j=read(folder/'jobs.json');assert j['status']=='COMPLETE'
        assert read(folder/'local_state_audit.json')['status']=='AUDIT_PASS'
        c=j['condition']
        if old is not None:assert c['initial']==str(old/'final_state.npz')
        initial=np.load(c['initial']);final=np.load(folder/'final_state.npz')
        assert int(final['clock'][0]-initial['clock'][0])==round(c['duration_ms']/.05)
        assert np.array_equal(final['syn'][5],Z)
        for block in j['completed_blocks']:
            path=folder/f'block{block:02d}.npz';z=np.load(path)
            assert np.array_equal(z['Z'],Z)
            regional.append(z['group_rate_hz'].astype(float)@W.T)
            M.append(z['M_current']@W.T);tm.append(offset+z['state_time_ms']);sources.append(str(path))
        offset+=c['duration_ms'];old=folder
    r=np.concatenate(regional);m=np.concatenate(M);mt=np.concatenate(tm)
    assert len(r)==offset==30000
    sm=uniform_filter1d(r[:,1],10,mode='nearest')
    quiet=[(a,b) for a,b in runs(sm<5) if b-a>=20]
    # Activities here are bounded by >=20ms local quiet intervals. Short
    # dips do not count as termination; boundary-truncated runs are censored.
    boundaries=[(0,quiet[0][0])] if quiet else [(0,len(sm))]
    boundaries += [(b,quiet[i+1][0]) for i,(_,b) in enumerate(quiet[:-1])]
    if quiet:boundaries += [(quiet[-1][1],len(sm))]
    activities=[]
    for a,b in boundaries:
        if b<=a:continue
        peak=float(sm[a:b].max())
        if peak<20 or b-a<20:continue
        activities.append(dict(start_ms=int(a),end_ms=int(b),duration_ms=int(b-a),peak_hz=peak,
            left_censored=a==0,right_censored=b==len(sm)))
    windows=[]
    for a in range(0,len(sm),5000):
        b=a+5000;x=sm[a:b];q=[(lo,hi) for lo,hi in quiet if lo>=a and hi<=b]
        windows.append(dict(window_ms=[a,b],mean_Core_A_hz=float(r[a:b,1].mean()),
            min_Core_A_10ms_hz=float(x.min()),quiet_fraction=float((x<5).mean()),
            complete_quiet_intervals=len(q),Core_A_above50_fraction=float((x>50).mean())))
    result=dict(status='AUDIT_PASS',side=side,total_same_field_ms=offset,all_Z_fixed=True,all_M_dynamic=True,
        D_A=read(DEST/'contract.json')['coordinates']['mid1']['D_A'],sources=sources,
        continuous_activity_definition='Core A 10ms rate activity with peak>=20Hz/duration>=20ms, separated by >=20ms quiet<5Hz; short dips do not count as termination. Records at0/30000ms are censored.',
        windows=windows,activities=activities,quiet_intervals_ms=quiet,
        inference='Longer same-field state history, not a proof of distinct attractors or any bifurcation type.')
    write(DEST/f'mid1_{side}_30s_history.json',result)
    np.savez_compressed(DEST/f'mid1_{side}_30s_regional.npz',time_ms=np.arange(1,30001),regional_rate_hz=r,
                        Core_A_smoothed_hz=sm,M_current=m,M_time_ms=mt)
    print(side,windows,flush=True)
    # Search longer periods than the original2s screening horizon now that
    # completed local activity can last16s. This remains a rate-field seed
    # diagnostic, not full delayed-state periodicity or a stability test.
    full=np.concatenate([np.load(path)['group_rate_hz'] for path in sources])
    rec=recurrence(full,s,max_lag_ms=15000)
    if rec['status']=='RATE_RECURRENCE_ONLY':
        np.savez_compressed(DEST/f'mid1_{side}_long_recurrence.npz',
            lag_ms=rec.pop('lags_ms'),relative_MSE=rec.pop('relative_MSE'))
    rec.update(observed_ms=30000,maximum_tested_lag_ms=15000,
               scope='Full E/I rate-field recurrence over30s; finite-time nonstationarity can affect this diagnostic. No statement about longer periods, chaos, full-state closure or bifurcation.')
    write(DEST/f'mid1_{side}_long_recurrence.json',rec)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('side',choices=['lower','upper']);audit(p.parse_args().side)
