"""Phase-independent waveform-distance bounds for every finite window pair.

In the neuron-weighted, full-population Hilbert norm, any common phase
shift satisfies D^2 >= ||mean(x)-mean(y)||^2 + (sigma_x-sigma_y)^2.
Integer repetition leaves these means and temporal norms unchanged.
The bound excludes close identity of cached waveforms, not global branches.
"""
from plot_rate_periodic_completion import RateField, families, np, Path, read, write
from scipy.signal import resample
from audit_rate_survey_filter_states import fingerprint
import argparse
import os
import time


DEST = Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920')


def statistics(r, weights):
    """Exact period mean/norm of the real trigonometric interpolant.

    An even mesh's single Nyquist coefficient is split between +/- N/2
    during real interpolation; its continuous variance is half its grid
    variance. Account for that instead of treating grid RMS as exact.
    """
    mean = r.mean(0)
    variance = float(np.mean((r-mean)**2, axis=0) @ weights)
    nyquist_correction = 0.
    if len(r) % 2 == 0:
        nyquist = (r[::2].sum(0)-r[1::2].sum(0))/len(r)
        nyquist_correction = float(.5*(nyquist**2 @ weights))
        variance -= nyquist_correction
    assert variance >= -1e-10*max(1., float(mean**2 @ weights))
    return mean, float(np.sqrt(max(0.,variance))), nyquist_correction


def controls():
    rng = np.random.default_rng(1057)
    x = rng.normal(size=(32,7)); weights=np.arange(1,8,dtype=float);weights/=weights.sum()
    mean, sigma, correction = statistics(x,weights)
    finer = resample(x,256,axis=0)
    direct_sigma = float(np.sqrt(np.mean((finer-finer.mean(0))**2,axis=0)@weights))
    assert abs(sigma-direct_sigma)<1e-12
    y = 2.3*x+.7
    other_mean, other_sigma, _ = statistics(y,weights)
    bound = float(np.sqrt((mean-other_mean)**2@weights+(sigma-other_sigma)**2))
    yf = resample(y,256,axis=0)
    distance = float(np.sqrt(np.mean((finer-yf)**2,axis=0)@weights))
    assert abs(bound-distance)<1e-12
    nyquist=(-1.)**np.arange(32)
    _, alternating_sigma, _=statistics(nyquist[:,None],np.ones(1))
    assert abs(alternating_sigma-1/np.sqrt(2))<1e-12
    return dict(continuous_norm_error=abs(sigma-direct_sigma),
                equality_case_error=abs(bound-distance),
                nyquist_only_sigma=alternating_sigma,
                nonzero_Nyquist_correction=correction)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--multiples',type=int,nargs='+',default=[5,7])
    parser.add_argument('--output-name',default='period_multiple_5_7_all_pair_bounds')
    parser.add_argument('--candidate-threshold',type=float,default=.05)
    args=parser.parse_args()
    assert Path(args.output_name).name==args.output_name
    assert all(m>=1 for m in args.multiples) and 0<args.candidate_threshold<1
    multiples=sorted(set(args.multiples));fs=families();catalog={};windows=[];needed={}
    folder=DEST/'period_multiple_bounds';folder.mkdir(exist_ok=True)
    for name in ['A','B','single','Bleading','double']:
        catalog[name]=[dict(orbit=q['path'],canonical_orbit=str(Path(q['path']).resolve()),
            J_EE_core=q['J_EE_core'],T_ms=q['T_ms']) for q in fs[name]]
    for first in ['A','B']:
        a=catalog[first];aj=np.array([q['J_EE_core'] for q in a]);at=np.array([q['T_ms'] for q in a])
        for second in ['single','Bleading','double']:
            b=catalog[second];bj=np.array([q['J_EE_core'] for q in b]);bt=np.array([q['T_ms'] for q in b])
            for multiple in multiples:
                indices=np.argwhere((abs(aj[:,None]-bj[None,:])<=.01)&
                    (abs(np.log(multiple*at[:,None]/bt[None,:]))<=np.log(1.10)))
                windows.append(dict(first=first,second=second,multiple=multiple,indices=indices))
                for i,j in indices:
                    for q in [a[i],b[j]]:needed[q['canonical_orbit']]=q
    catalog_source=folder/(args.output_name+'_catalog.json')
    write(catalog_source,dict(timestamp=time.time(),families=catalog,multiples=multiples,
        parameter_window=.01,relative_period_window=1.10))
    s=RateField();weights=s.geo['group_size']/s.geo['group_size'].sum();assert len(weights)==935
    moments={};saved=[]
    for index,(key,q) in enumerate(needed.items()):
        source=Path(q['orbit']);before=fingerprint(source)
        with np.load(source) as z:
            r=z['r']
            assert r.shape[1]==935
            assert abs(float(z['J'])-q['J_EE_core'])<1e-10
            assert abs(float(z['T'])-q['T_ms'])<1e-6
            mean, sigma, correction=statistics(r*1000,weights)
            mesh=len(r)
        assert fingerprint(source)==before
        moments[key]=(mean,sigma)
        saved.append(dict(orbit=str(source),profile_fingerprint=before,
            J_EE_core=q['J_EE_core'],T_ms=q['T_ms'],temporal_mesh=mesh,
            population_mean_Hz=mean,temporal_RMS_Hz=sigma,
            Nyquist_variance_correction_Hz_squared=correction))
        if (index+1)%25==0:
            write(folder/(args.output_name+'_worker.json'),dict(status='PROFILE_MOMENTS',
                pid=os.getpid(),completed=index+1,expected=len(needed)))
            print('MOMENTS',index+1,'/',len(needed),flush=True)
    moments_source=folder/(args.output_name+'_moments.json')
    write(moments_source,dict(rows=saved))
    rows=[]
    for window in windows:
        a,b=[catalog[window[k]] for k in ['first','second']]
        records=[]
        for i,j in window['indices']:
            x,y=a[i],b[j]
            mx,sx=moments[x['canonical_orbit']]
            my,sy=moments[y['canonical_orbit']]
            mean_difference_squared=float((mx-my)**2@weights)
            lower=float(np.sqrt(mean_difference_squared+(sx-sy)**2))
            scale=max(sx,sy)
            assert scale>0
            records.append(dict(first_index=int(i),second_index=int(j),
                first_orbit=x['orbit'],second_orbit=y['orbit'],
                first_J=x['J_EE_core'],second_J=y['J_EE_core'],
                first_T_ms=x['T_ms'],second_T_ms=y['T_ms'],
                waveform_RMS_lower_bound_Hz=lower,normalization_RMS_Hz=scale,
                normalized_distance_lower_bound=lower/scale))
        records.sort(key=lambda q:q['normalized_distance_lower_bound'])
        remaining=[q for q in records if q['normalized_distance_lower_bound']<=args.candidate_threshold]
        row=dict(families=[window['first'],window['second']],period_multiple=window['multiple'],
            window_pairs=len(records),excluded_pairs=len(records)-len(remaining),
            minimum_normalized_lower_bound=records[0]['normalized_distance_lower_bound'] if records else None,
            weakest_bounds=records[:5],unexcluded_candidates=remaining)
        rows.append(row)
        print('BOUNDS',row['families'],row['period_multiple'],len(records),
              row['minimum_normalized_lower_bound'],'unexcluded',len(remaining),flush=True)
    write(DEST/(args.output_name+'.json'),dict(status='ALL_FINITE_WINDOW_PAIRS_BOUNDED',
        timestamp=time.time(),catalog_source=str(catalog_source),moments_source=str(moments_source),
        checked_profiles=len(moments),rows=rows,multiples=multiples,
        candidate_threshold=args.candidate_threshold,controls=controls(),
        formula='D_phase_min >= sqrt(norm_weighted(mean_x-mean_y)^2 + (sigma_x-sigma_y)^2); divide by max(sigma_x,sigma_y).',
        norm='Full 935 population, neuron-count weighted, continuous trigonometric profile L2 over normalized phase. Means are retained; a common phase shift and integer repetition do not change these moments.',
        target_profile_scope='Moments of the frozen cached BVP profiles. This calculation does not validate their physical equations or positivity, transfer their stability, or correct them to identical J.',
        interval_completeness=False,
        scope='Exclusion of close waveform identity for every catalog pair in the stated parameter/period windows, not exclusion of unsampled branch connections, intervening bifurcations, or other period ratios. Unexcluded bounds are candidates for phase-resolved comparison, not evidence of identity.'))
    write(folder/(args.output_name+'_worker.json'),dict(status='COMPLETE',pid=os.getpid(),
        completed=len(moments),source=str(DEST/(args.output_name+'.json'))))


if __name__=='__main__':
    main()
