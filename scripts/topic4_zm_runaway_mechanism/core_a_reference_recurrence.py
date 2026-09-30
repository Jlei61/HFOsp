"""Passive, phase-aligned recurrence of the actual interictal reference.

The one-burst numerical orbit is unstable even at the reference parameter.
Screen the actual multi-burst record before assuming it is that orbit. All
spatial groups, recent rate history and M enter the screen. These are saved
observations, not the complete solver state or a periodicity certificate.
"""
from common import OUT, np, model, read, write, log
from onset_state_continuation import regional_weights
from scipy.ndimage import uniform_filter1d


def main():
    sources = [OUT/'onset_state_continuation_20260923/native9000_from_lower',
               OUT/'core_a_resource_bifurcation_20260923/reference']
    dest = OUT/'core_a_bifurcation_type_20260924/reference_stability_gap/actual_reference_recurrence'
    dest.mkdir(exist_ok=True)
    s = model(40); W = regional_weights(s)
    assert read(sources[0]/'independent_audit.json')['status'] == 'AUDIT_PASS'
    assert read(sources[1]/'local_state_audit.json')['status'] == 'AUDIT_PASS'
    assert (sources[0]/'final_state.npz').resolve() == __import__('pathlib').Path(
        read(sources[1]/'jobs.json')['condition']['initial']).resolve()
    rates=[]; adaptation=[]; clock=[]; stateclock=[]; zref=None; source_index=[]
    for i, folder in enumerate(sources):
        jobs = read(folder/'jobs.json'); assert jobs['status'] == 'COMPLETE'
        for j in jobs['completed_blocks']:
            z = np.load(folder/f'block{j:02d}.npz')
            if zref is None: zref = z['Z'].copy()
            assert np.array_equal(z['Z'], zref)
            q = z['group_rate_hz'].astype(float)
            assert np.max(abs(q@W.T-z['regional_rate_hz'])) < 1e-9
            rates.append(q); adaptation.append(z['M_current'])
            clock.append(i*10000+z['elapsed_time_ms'])
            stateclock.append(i*10000+z['state_time_ms'])
            source_index.append(str(folder/f'block{j:02d}.npz'))
    rate=np.concatenate(rates); m=np.concatenate(adaptation)
    t=np.concatenate(clock); mt=np.concatenate(stateclock)
    assert np.array_equal(t,np.arange(1,20001))
    regional=rate@W.T; sm=uniform_filter1d(regional[:,1],10,mode='nearest')
    indices=np.flatnonzero((sm[:-1]<100)&(sm[1:]>=100))
    crossing=t[indices]+(100-sm[indices])/(sm[indices+1]-sm[indices])
    crossing=crossing[crossing>2000]
    # Interpolate stored1ms samples, retaining every group and35ms history.
    # This phase readout is deliberately independent of the shooting section.
    lag=np.arange(0,36.)
    def sample(a, grid, times):
        j=np.searchsorted(grid,times)-1; j=np.clip(j,0,len(grid)-2)
        f=(times-grid[j])/(grid[j+1]-grid[j])
        return a[j]*(1-f)[...,None]+a[j+1]*f[...,None]
    hist=sample(rate,t,crossing[:,None]-lag[None,:])
    mv=sample(m,mt,crossing)
    w=s.sizes/s.sizes.sum(); mean_hist=np.mean(np.sum(hist*hist*w,axis=-1))
    mean_m=np.mean(np.sum(mv*mv*w,axis=-1))
    rows=[]
    for i in range(len(crossing)):
        for j in range(i+1,min(i+13,len(crossing))):
            dh=np.mean(np.sum((hist[j]-hist[i])**2*w,axis=-1))
            dm=np.sum((mv[j]-mv[i])**2*w)
            hr=float(np.sqrt(dh/mean_hist)); mr=float(np.sqrt(dm/mean_m))
            rows.append(dict(index1=i,index2=j,burst_count=j-i,
                time1_ms=float(crossing[i]),time2_ms=float(crossing[j]),
                period_ms=float(crossing[j]-crossing[i]),history_relative_rms=hr,
                M_relative_rms=mr,score=float(np.sqrt((hr*hr+mr*mr)/2))))
    rows.sort(key=lambda x:x['score'])
    per=[min((r for r in rows if r['burst_count']==n),key=lambda x:x['score']) for n in range(1,13)]
    np.savez_compressed(dest/'sections.npz',time_ms=crossing,
        regional_rate_hz=sample(regional,t,crossing),M_regional=mv@W.T,
        interval_ms=np.diff(crossing),Z=zref)
    write(dest/'result.json',dict(status='PASSIVE_FULL_GROUP_RECURRENCE_SCREEN',
        source_blocks=source_index,D_A=float(1-zref@(W[1])),
        same_field_continuous_ms=20000,discarded_ms=2000,
        section='10ms smoothed Core-A E rate rises through100Hz, linearly interpolated from recorded1ms data.',
        distance='Equal RMS combination of all-group cell-weighted last35ms rate-history difference and entire M-field difference; each normalized by its own empirical second moment across retained sections.',
        best_by_burst_count=per,candidates=rows[:24],section_count=len(crossing),
        scope='A recurrence screen of stored observables, not the full synaptic/covariance/response/delay state. Small distance nominates exact replay and root correction; neither a cycle nor its stability or a bifurcation is inferred.',
        model_promoted=False))
    log('REFERENCE RECURRENCE SCREEN',per)


if __name__=='__main__':main()
