"""Freeze and inspect a continued segment before adding it to figures.

Continuous residuals are checked at selected waveforms, not inferred from
small collocation residuals. Parameter turns remain candidates until separate
fold refinement, temporal mesh and variational checks are available.
"""
from rate_periodic_accuracy import *
import gc


def main():
    p=argparse.ArgumentParser();p.add_argument('label')
    p.add_argument('--device',type=int,default=0);p.add_argument('--stride',type=int,default=12)
    p.add_argument('--check-filter-states',action='store_true')
    p.add_argument('--stream-harmonics',action='store_true')
    p.add_argument('--maximum-group-defect',type=float,default=.1)
    p.add_argument('--prior-segment',help='Actual preceding continuation segment; include its two endpoint profiles when checking joins')
    p.add_argument('--prior-orbits',nargs=2,help='Independently refined versions of the two actual preceding endpoints')
    a=p.parse_args();assert a.stride>=1 and a.maximum_group_defect>0;s=RateField()
    files=sorted((PERIODIC_OUT/'orbits').glob(a.label+'_*_N*.json'))
    rows=[read(f) for f in files if '_accuracy_' not in f.stem]
    assert rows and all(q['status']=='CONVERGED' for q in rows)
    # These four joins are exact continuation restarts from the final two
    # profiles, not overlapping coarse-to-refined restarts.
    predecessors={'arcAconnectionFurther':'arcAglobalConnection',
        'arcBconnectionFurther':'arcBtoBurstFurther',
        'arcAconnectionNext':'arcAconnectionFurther',
        'arcBconnectionNext':'arcBconnectionFurther'}
    prior=a.prior_segment or predecessors.get(a.label);prefix=[]
    if a.prior_orbits:
        prefix=[read(Path(f).with_suffix('.json')) for f in a.prior_orbits]
        assert all(q['status']=='CONVERGED' for q in prefix)
    elif prior:
        parent=read(PERIODIC_OUT/(prior+'_accuracy.json'))
        assert parent['status']=='SAMPLED_PASS'
        prefix=[read(Path(f).with_suffix('.json')) for f in parent['included_orbits'][-2:]]
        assert len(prefix)==2
    combined=prefix+rows;offset=len(prefix)
    J=np.array([q['J_EE_core'] for q in combined])
    turns=np.flatnonzero(np.diff(J)[:-1]*np.diff(J)[1:]<0)+1-offset
    chosen=set(range(0,len(rows),a.stride))|{len(rows)-1}
    for i in turns:chosen.update(range(max(-offset,i-1),min(len(rows),i+2)))
    weights=s.geo['group_size']/s.geo['group_size'].sum();checks=[]
    for i in sorted(chosen):
        meta=combined[i+offset];q=defect(s,meta['path'],a.device,harmonic_chunk_size=64,
            stream_harmonics=a.stream_harmonics)
        z=np.load(meta['path']);r=z['r']*1000
        if a.check_filter_states:q['filter_state_check']=filter_state_minima(s,z['r'],float(z['T']))
        cf=np.fft.fft(r-r.mean(0),axis=0)/len(r);energy=np.sum(abs(cf)**2*weights,axis=1)
        harmonics=np.fft.fftfreq(len(r))*len(r)
        q['proper_subperiod_relative_distances']={str(d):float(np.sqrt(np.sum(energy*abs(1-np.exp(2j*np.pi*harmonics/d))**2)/energy.sum())) for d in [2,3,4]}
        q['index']=i;checks.append(q)
        gc.collect()
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    passed=all(q['maximum_group_defect_Hz']<a.maximum_group_defect and max(q['regional_defect_Hz'])<.001 and q['minimum_rate_Hz']>=-1e-9 for q in checks)
    if a.check_filter_states:passed=passed and all(q['filter_state_check']['positive'] for q in checks)
    result=dict(status='SAMPLED_PASS' if passed else 'RESOLUTION_UNRESOLVED',label=a.label,
        included_orbits=[q['path'] for q in rows],continued_points=len(rows),checks=checks,
        predecessor_segment=prior,predecessor_sources=[q['path'] for q in prefix],
        check_stride=a.stride,maximum_group_defect_tolerance_Hz=a.maximum_group_defect,
        turns=[dict(index=int(i),left=combined[i+offset-1]['path'],center=combined[i+offset]['path'],right=combined[i+offset+1]['path'],
                    J_EE_core=J[i+offset],T_ms=combined[i+offset]['T_ms'],at_segment_join=bool(i<=0)) for i in turns],
        scope='Frozen continued segment; sampled continuous residual and 2/3/4 subperiod checks. Turns are candidates only. No stability or exhaustive interval certificate.')
    write(PERIODIC_OUT/(a.label+'_accuracy.json'),result)
    print('EXTENSION CHECK',a.label,result['status'],len(rows),'points',len(checks),'checks',len(turns),'candidate turns',flush=True)


if __name__=='__main__':main()
