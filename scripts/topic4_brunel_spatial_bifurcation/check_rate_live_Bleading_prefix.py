"""Freeze the finished prefix of the running B-leading continuation.

Every included orbit receives a full off-grid and filter-state check. The
running continuation is neither restarted nor read beyond its saved profiles.
"""
from rate_periodic_accuracy import *
import os


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    p.add_argument('--through-index',type=int)
    p.add_argument('--cpu-checks',type=Path)
    a=p.parse_args();label='arcBleadingConnection_20260920'
    folder=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920/Bleading_extension')
    source=PERIODIC_OUT/f'{label}_accuracy.json'
    previous=read(source);assert previous['status']=='SAMPLED_PASS'
    files=sorted((PERIODIC_OUT/'orbits').glob(label+'_[0-9][0-9][0-9][0-9]_N4096.json'))
    if a.through_index is not None:
        assert a.through_index>=0 and len(files)>a.through_index
        files=files[:a.through_index+1]
    rows=[read(f) for f in files]
    assert rows and all(q['status']=='CONVERGED' for q in rows)
    assert [Path(q['path']).stem.split('_')[-2] for q in rows]==[f'{i:04d}' for i in range(len(rows))]
    cache={str(Path(q['orbit']).resolve()):q for q in previous['checks']}
    cpu={}
    if a.cpu_checks:
        independent=read(a.cpu_checks);assert independent['status']=='SELECTED_POINTS_PASS'
        cpu={str(Path(q['path']).resolve()):q for q in independent['rows']}
        shared=set(cpu)&set(cache);assert shared
        for key in shared:
            left,right=cpu[key],cache[key]
            assert left['check_N']>=4*left['N']
            assert abs(left['between_nodes_integrated_error_Hz']-right['maximum_group_defect_Hz'])<1e-6
    s=RateField();weights=s.geo['group_size']/s.geo['group_size'].sum();checks=[]
    for i,row in enumerate(rows):
        orbit=Path(row['path']);key=str(orbit.resolve())
        if key in cache:q=cache[key]
        else:
            write(folder/'prefix_check_worker.json',dict(status='CHECKING_PREFIX',pid=os.getpid(),index=i,points=len(rows)))
            if a.cpu_checks:
                independent=cpu[key]
                assert independent['status']=='POINT_CHECK_PASS'
                assert independent['check_N']>=4*independent['N']
                assert independent['full_RHS_rate_error_Hz_per_ms']<.001
                assert independent['full_RHS_linear_state_max_abs']<1e-8
                assert independent['refractory_rate_bound_excess_Hz']<1e-8
                assert abs(independent['J']-row['J_EE_core'])<1e-12
                assert abs(independent['T_ms']-row['T_ms'])<1e-8
                q=dict(orbit=str(orbit),N=independent['N'],check_N=independent['check_N'],
                    J_EE_core=independent['J'],T_ms=independent['T_ms'],
                    maximum_group_defect_Hz=independent['between_nodes_integrated_error_Hz'],
                    regional_defect_Hz=independent['between_nodes_regional_error_A_B_surround_Hz'],
                    minimum_rate_Hz=independent['minimum_interpolated_group_rate_Hz'],
                    computation='Independent CPU reconstruction of all nine local states and original physical delays on the fourfold temporal mesh',
                    independent_full_rhs_evidence=str(a.cpu_checks),independent_full_rhs=independent)
            else:q=defect_bounded(s,orbit,a.device)
            z=np.load(orbit);r=z['r']*1000
            q['filter_state_check']=filter_state_minima(s,z['r'],float(z['T']))
            cf=np.fft.fft(r-r.mean(0),axis=0)/len(r)
            energy=np.sum(abs(cf)**2*weights,axis=1);harmonics=np.fft.fftfreq(len(r))*len(r)
            q['proper_subperiod_relative_distances']={str(d):float(np.sqrt(np.sum(energy*abs(1-np.exp(2j*np.pi*harmonics/d))**2)/energy.sum())) for d in [2,3,4]}
        assert q['maximum_group_defect_Hz']<.001 and q['filter_state_check']['positive']
        assert q['minimum_rate_Hz']>=-1e-9
        q['index']=i;checks.append(q)
        gc.collect()
        if not a.cpu_checks:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
    prefix=[read(Path(f).with_suffix('.json')) for f in previous['predecessor_sources']]
    combined=prefix+rows;J=np.array([q['J_EE_core'] for q in combined])
    turns=np.flatnonzero(np.diff(J)[:-1]*np.diff(J)[1:]<0)+1
    result=dict(timestamp=time.time(),status='SAMPLED_PASS',label=label,
        included_orbits=[q['path'] for q in rows],continued_points=len(rows),checks=checks,
        predecessor_segment=previous['predecessor_segment'],predecessor_sources=previous['predecessor_sources'],
        turns=[dict(index=int(i-len(prefix)),left=combined[i-1]['path'],center=combined[i]['path'],right=combined[i+1]['path'],
            J_EE_core=J[i],T_ms=combined[i]['T_ms'],at_segment_join=bool(i<=len(prefix))) for i in turns],
        scope='Every saved profile in this frozen prefix passes off-grid and physical-state checks. The endpoint is computational; no global connection or complete interval spectrum is implied.')
    write(folder/f'accepted_prefix_{len(rows):04d}.json',result)
    # Preserve the earlier frozen prefix; publish only the additive checked
    # prefix after all its checks have passed.
    assert set(previous['included_orbits']).issubset(result['included_orbits'])
    tmp=source.with_suffix('.prefix-tmp.json');write(tmp,result);tmp.replace(source)
    write(folder/'prefix_check_worker.json',dict(status='PREFIX_CHECKED',pid=os.getpid(),points=len(rows),J_range=[min(J),max(J)]))
    print('CHECKED PREFIX',len(rows),flush=True)


if __name__=='__main__':main()
