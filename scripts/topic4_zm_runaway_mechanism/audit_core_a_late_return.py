"""Audit the exact late quiet return that invalidates the60s censor label."""
from common import OUT,np,read,write,log,model
from onset_state_continuation import regional_weights
from scipy.ndimage import uniform_filter1d


def main():
    base=OUT/'core_a_bifurcation_type_20260924'
    prior=base/'sustained_A_long_window'
    growth=base/'actual_sustained_growth/coarse'
    assert read(prior/'joined60s_audit.json')['status']=='AUDIT_PASS'
    assert read(growth/'local_state_audit.json')['status']=='AUDIT_PASS'
    initial=np.load(prior/'below_extend_to60s/final_state.npz')
    jobs=read(growth/'jobs.json');assert jobs['status']=='COMPLETE'
    assert jobs['condition']['initial']==str(prior/'below_extend_to60s/final_state.npz')
    old=np.load(prior/'joined60s_regional.npz');rates=[old['regional_rate_hz']]
    W=regional_weights(model(40));sources=[];m=[];tm=[]
    for b in jobs['completed_blocks']:
        path=growth/f'block{b:02d}.npz';z=np.load(path)
        assert np.array_equal(z['Z'],initial['syn'][5])
        rr=z['group_rate_hz'].astype(float)@W.T
        assert np.max(abs(rr-z['regional_rate_hz']))<1e-9
        assert np.array_equal(z['elapsed_time_ms'],b*5000+np.arange(1,5001))
        rates.append(rr);m.append(z['M_current']@W.T);tm.append(z['state_time_ms']+60000);sources.append(str(path))
    rates=np.concatenate(rates);assert len(rates)==70000
    sm=uniform_filter1d(rates[:,1],10,mode='nearest')
    edges=np.diff(np.r_[False,sm<5,False].astype(int))
    quiet=[[int(a),int(b)] for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=20]
    after=[q for q in quiet if q[0]>60000]
    assert after,'This audit requires a real late qualified quiet return'
    first=after[0];last_before=max(b for a,b in quiet if b<first[0])
    out=base/'late_return_counterexample';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'joined70s_regional.npz',time_ms=np.arange(1,70001),regional_rate_hz=rates,
        Core_A_smoothed_hz=sm,M_regional=np.concatenate(m),M_time_ms=np.concatenate(tm))
    write(out/'result.json',dict(status='AUDIT_PASS',D_A=read(prior/'joined60s_audit.json')['D_A'],
        Z_A=1-read(prior/'joined60s_audit.json')['D_A'],observed_ms=70000,
        exact_original_terminal_state_resumed=True,nominal_flow_unaltered_by_passive_tangent=True,
        all_Z_held=True,all_M_dynamic=True,quiet_intervals_ms=quiet,
        first_quiet_after60s_ms=first,ended_activity=dict(start_ms=last_before,end_ms=first[0],duration_ms=first[0]-last_before),
        sources=[str(prior/'joined60s_regional.npz')]+sources,
        conclusion='The activity episode previously right-censored at60s ends at64.830s. D_A=.34315764 cannot serve as a demonstrated asymptotic sustained-side endpoint or certified onset bracket againstD_A=.33564.',
        distinction='Long irregular activity and its later termination are directly observed. This does not identify a crisis, chaos in the infinite-time limit, a unique transition parameter, or the native-SNN onset mechanism.',
        model_promoted=False))
    log('LATE RETURN COUNTEREXAMPLE',first,'episode duration ms',first[0]-last_before)


if __name__=='__main__':main()
