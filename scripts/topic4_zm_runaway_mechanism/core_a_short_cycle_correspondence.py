"""Compare the short-cycle M state with actual rare low-activity returns.

This is a passive same-Z correspondence diagnostic, not a manifold test.
Actual1ms rates and10ms M cannot replace exact Poincare/full-state closure.
"""
from common import OUT,np,read,write,model
from pathlib import Path


def main():
    b=OUT/'core_a_bifurcation_type_20260924'
    root=b/'near_returns/mid_lower/coarse_below_SN_positive_cubic'
    assert read(root/'result.json')['status']=='NUMERICAL_PERIODIC_ROOT'
    reference=dict(np.load(root/'latest_state.npz'));s=model(40)
    mask=s.E&(s.geo['group_region']==0);w=s.sizes[mask]/s.sizes[mask].sum()
    tick=int(reference['clock'][0]);hist=reference['history'];target=float(hist[tick%len(hist),mask]@w*1000)
    assert target>float(hist[(tick-1)%len(hist),mask]@w*1000),'This diagnostic uses the known rising Core-A section'
    audit=read(b/'late_return_counterexample/result.json');assert audit['status']=='AUDIT_PASS'
    prefix=read(b/'sustained_A_long_window/joined60s_audit.json');assert prefix['status']=='AUDIT_PASS'
    paths=[Path(x) for x in prefix['sources']+audit['sources'][1:]]
    M=[]
    for path in paths:
        z=np.load(path);assert np.max(abs(z['Z']-reference['syn'][5]))<2e-12
        M.append(z['M_current'][:,mask])
    M=np.concatenate(M);rr=np.load(b/'late_return_counterexample/joined70s_regional.npz')['regional_rate_hz'][:,1]
    assert len(rr)==10*len(M)==70000
    candidates=np.flatnonzero((rr[:-1]<target)&(rr[1:]>=target))+1
    candidates=candidates[candidates>=2000];times=np.arange(10,70001,10);ref=reference['syn'][4,mask]
    rows=[]
    for j in candidates:
        alpha=(target-rr[j-1])/(rr[j]-rr[j-1]);tm=j+alpha
        k=int(np.searchsorted(times,tm));a=(tm-times[k-1])/(times[k]-times[k-1]);actual=(1-a)*M[k-1]+a*M[k]
        rows.append(dict(crossing_time_ms=float(tm),actual_M_A_mV_equiv=float(actual@w),
                         root_M_A_mV_equiv=float(ref@w),
                         spatial_M_A_relative_rms=float(np.sqrt(((actual-ref)**2)@w)/np.sqrt((ref**2)@w))))
    result=dict(status='PASSIVE_SAME_FIELD_SECTION_M_COMPARISON',D_A=audit['D_A'],
                root=str(root),actual_sources=[str(p) for p in paths],Core_A_rising_section_hz=target,
                root_M_A_mV_equiv=float(ref@w),crossings_after2s=rows,
                definitions='Numerical root uses instantaneous rate; actual crossings use original1ms rate bins, linearly interpolated only for comparison. Actual M is interpolated from original10ms records. Same complete Z field checked in each source. M_A is cell-count-weighted eta_M M in mV equivalent.',
                interpretation='A substantial spatial M mismatch at comparable low-rate phases shows that these actual rare returns are not simply the already-found short cycle. It does not rule out mediation by a distant part of that cycle stable/unstable manifold, prove another attractor, or classify a crisis.',model_promoted=False)
    write(b/'short_cycle_actual_M_correspondence.json',result)
    print('root',target,ref@w,'actual',rows,flush=True)


if __name__=='__main__':main()
