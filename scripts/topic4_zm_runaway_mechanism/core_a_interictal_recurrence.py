"""Passive full-group recurrence screen at the actual lower entry endpoint.

Event spacing alone cannot identify the full spatial attractor. Use every
group's recent rate history and M, as in the existing reference screen.
This nominates complete-state replay only; it does not certify periodicity.
"""
from common import OUT,np,read,write,model,log
from onset_state_continuation import regional_weights
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import argparse


def main(source=None,destination=None,discard_ms=2000):
    base=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap'
    source=Path(source) if source else base/'actual_D0270/from_interictal_history'
    dest=Path(destination) if destination else base/'actual_D0270/recurrence';dest.mkdir(exist_ok=True)
    assert read(source/'whole_record_audit.json')['status']=='AUDIT_PASS'
    s=model(40);W=regional_weights(s);rr=[];mm=[];tt=[];mt=[];paths=[];Z=None
    for j in read(source/'jobs.json')['completed_blocks']:
        path=source/f'block{j:02d}.npz'
        with np.load(path) as z:
            if Z is None:Z=z['Z'].copy()
            assert np.array_equal(Z,z['Z'])
            rr.append(z['group_rate_hz'].astype(float));mm.append(z['M_current'])
            tt.append(z['elapsed_time_ms']);mt.append(z['state_time_ms']);paths.append(str(path))
    rate=np.concatenate(rr);m=np.concatenate(mm);t=np.concatenate(tt);mt=np.concatenate(mt)
    regional=rate@W.T;sm=uniform_filter1d(regional[:,1],10,mode='nearest')
    ix=np.flatnonzero((sm[:-1]<100)&(sm[1:]>=100))
    crossing=t[ix]+(100-sm[ix])/(sm[ix+1]-sm[ix]);crossing=crossing[crossing>discard_ms]
    def sample(a,grid,times):
        j=np.clip(np.searchsorted(grid,times)-1,0,len(grid)-2)
        f=(times-grid[j])/(grid[j+1]-grid[j])
        return a[j]*(1-f)[...,None]+a[j+1]*f[...,None]
    hist=sample(rate,t,crossing[:,None]-np.arange(36.)[None,:]);mv=sample(m,mt,crossing)
    w=s.sizes/s.sizes.sum();hs=np.mean(np.sum(hist*hist*w,axis=-1));ms=np.mean(np.sum(mv*mv*w,axis=-1))
    rows=[]
    for i in range(len(crossing)):
        for j in range(i+1,min(i+19,len(crossing))):
            hr=float(np.sqrt(np.mean(np.sum((hist[j]-hist[i])**2*w,axis=-1))/hs))
            mr=float(np.sqrt(np.sum((mv[j]-mv[i])**2*w)/ms))
            rows.append(dict(index1=i,index2=j,burst_count=j-i,time1_ms=float(crossing[i]),time2_ms=float(crossing[j]),period_ms=float(crossing[j]-crossing[i]),history_relative_rms=hr,M_relative_rms=mr,score=float(np.hypot(hr,mr)/np.sqrt(2))))
    rows.sort(key=lambda r:r['score']);best=[min((r for r in rows if r['burst_count']==n),key=lambda r:r['score']) for n in range(1,19)]
    np.savez_compressed(dest/'sections.npz',time_ms=crossing,regional_rate_hz=sample(regional,t,crossing),M_regional=mv@W.T,Z=Z)
    write(dest/'result.json',dict(status='PASSIVE_FULL_GROUP_RECURRENCE_SCREEN',source_blocks=paths,D_A=float(1-Z@W[1]),observed_ms=len(rate),discard_ms=discard_ms,section='Positive100Hz crossing of10ms-smoothed originalCoreA rate;1ms linear interpolation.',distance='Equal RMS of all-group cell-weighted35ms history and fullM differences, each normalized by empirical section second moment. Same observable as original reference screen, extended only to18bursts because observedA intervals show a longer pattern.',best_by_burst_count=best,candidates=rows[:24],scope='Stored-observable near-return only. No complete-state cycle, stability, torus, chaos or bifurcation certificate.',model_promoted=False))
    log('LOWER ENTRY RECURRENCE',rows[:5])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source');p.add_argument('--destination');p.add_argument('--discard-ms',type=float,default=2000)
    a=p.parse_args();main(a.source,a.destination,a.discard_ms)
