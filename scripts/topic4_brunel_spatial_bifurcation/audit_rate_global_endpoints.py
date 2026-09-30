"""Full spatial waveform screening of extended Hopf-family endpoints.

One common temporal phase is optimized over all 935 groups. A close match
is only a candidate for common-parameter BVP correction, never a connection.
"""
from plot_rate_periodic_completion import *
from compare_rate_torus_periodic_targets import distances


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='global_endpoints_waveform_audit')
    parser.add_argument('--phase-samples',type=int,default=256)
    args=parser.parse_args()
    s=RateField();fs=families();weights=s.geo['group_size']/s.geo['group_size'].sum()
    result=[];N=args.phase_samples;assert N>=256
    for family,segment in [('A','arcAconnectionFurther'),('B','arcBconnectionFurther')]:
        later=PERIODIC_OUT/f'arc{family}connectionNext_accuracy.json'
        if later.exists() and read(later)['status']=='SAMPLED_PASS':segment=f'arc{family}connectionNext'
        later=PERIODIC_OUT/f'arc{family}connectionStage3_accuracy.json'
        if later.exists() and read(later)['status']=='SAMPLED_PASS':segment=f'arc{family}connectionStage3'
        for stage in [4,5]:
            # H2 uses the dated continuation label; omitting it silently
            # selected an older endpoint despite a checked later segment.
            candidate=f'arc{family}connectionStage{stage}'+('_20260920' if family=='B' else '')
            later=PERIODIC_OUT/(candidate+'_accuracy.json')
            if later.exists() and read(later)['status']=='SAMPLED_PASS':segment=candidate
        accuracy=read(PERIODIC_OUT/(segment+'_accuracy.json'))
        assert accuracy['status']=='SAMPLED_PASS'
        source=Path(accuracy['included_orbits'][-1]);last=read(source.with_suffix('.json'));z=np.load(source)
        endpoint_check=next(q for q in accuracy['checks'] if Path(q['orbit']).resolve()==source.resolve())
        assert endpoint_check['maximum_group_defect_Hz']<.001
        assert endpoint_check['minimum_rate_Hz']>=-1e-9
        assert endpoint_check['filter_state_check']['positive']
        assert z['r'].shape[1]==935
        r=resample(z['r']*1000,N,axis=0)
        scale=float(np.sqrt(np.mean(np.sum((r-r.mean(0))**2*weights,axis=1))))
        period_checks=[]
        for divisor in [2,3,4]:
            shift=1/divisor
            shifted=np.fft.ifft(np.fft.fft(r,axis=0)*np.exp(2j*np.pi*np.fft.fftfreq(N)[:,None]*N*shift),axis=0).real
            period_checks.append(dict(divisor=divisor,relative_difference=float(np.sqrt(np.mean(np.sum((r-shifted)**2*weights,axis=1)))/scale)))
        candidates=[];seen=set()
        recent={str(Path(q['path']).resolve()) for q in fs[family][-12:]}
        for name,rows in fs.items():
            for index,q in enumerate(rows):
                path=Path(q['path']);key=str(path.resolve())
                if key in seen or key in recent:continue
                if abs(q['J_EE_core']-last['J_EE_core'])>.02:continue
                if abs(q['T_ms']/last['T_ms']-1)>.25:continue
                seen.add(key);other=resample(np.load(path)['r']*1000,N,axis=0)
                delta,phase=distances(r[:,None,:],other,weights)
                candidates.append(dict(family=name,index=index,orbit=str(path),
                    J_EE_core=q['J_EE_core'],T_ms=q['T_ms'],
                    full_group_profile_RMS_difference_Hz=float(delta[0]),relative_difference=float(delta[0]/scale),
                    common_phase_shift_cycles=float(phase[0])))
        candidates.sort(key=lambda q:q['relative_difference'])
        row=dict(family=family,source=str(source),J_EE_core=last['J_EE_core'],T_ms=last['T_ms'],
            segment=segment,endpoint_physical_check=endpoint_check,
            mean_rates_hz=last['mean_rates_hz'],min_rates_hz=last['min_rates_hz'],max_rates_hz=last['max_rates_hz'],
            half_and_subperiod_checks=period_checks,screened_candidates=len(candidates),
            nearest_candidates=candidates[:10],
            best_other_family=[min([q for q in candidates if q['family']==name],key=lambda q:q['relative_difference'])
                for name in fs if name!=family and any(q['family']==name for q in candidates)])
        result.append(row);print('ENDPOINT',family,segment,last['J_EE_core'],last['T_ms'],
            'CANDIDATES',len(candidates),'NEAREST_OTHER',row['best_other_family'],flush=True)
    control,_=distances(np.roll(r,37,axis=0)[:,None,:],r,weights)
    assert control[0]<1e-7
    write(PERIODIC_OUT/(args.output+'.json'),dict(status='SCREEN_COMPLETE',rows=result,
        phase_samples=N,identity_phase_control_Hz=float(control[0]),
        target_profile_scope='Candidate targets are cached BVP profiles; this screen does not independently certify their off-grid equations or filter-state positivity. Only the source endpoints are required to pass those checks here.',
        search_window=dict(J_absolute_difference=.02,period_relative_difference=.25),
        observable='Neuron-weighted full 935-group waveform RMS, with one phase common to all groups; denominator is endpoint temporal RMS around its mean.',
        scope='A parameter/period window screen. Neither similarity at different parameters nor a negative screen establishes or excludes a global connection. A period-divisor mismatch only tests those divisors.'))


if __name__=='__main__':main()
