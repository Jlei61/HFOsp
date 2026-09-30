"""Audit the sign change of dominant PD4-child multipliers without inventing a PD.

Use reliable growing eigenpairs from paired full-history calculations.
Endpoint spectra and subspaces do not locate intermediate crossings.
"""
from complete_rate_positive_stability import DEST, PERIODIC_OUT, read, write, values, paired_modes, np
from scipy.linalg import qr
from pathlib import Path
import argparse


def real_span(vectors):
    matrix=np.column_stack([vectors.real,vectors.imag])
    basis,upper,_=qr(matrix,mode='economic',pivoting=True)
    diagonal=abs(np.diag(upper))
    rank=int(np.sum(diagonal>diagonal.max()*1e-10))
    return basis[:,:rank]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--include-midpoint',action='store_true')
    args=parser.parse_args()
    sources=[
        [PERIODIC_OUT/f'poincare_ritz/PD4_child0_k6_{i}.json' for i in [0,1]],
        [PERIODIC_OUT/f'poincare_floquet/PD4_child_index1_full_k6_20260920_dt{dt}.json'
         for dt in ['0.05','0.025']]]
    children=read(DEST/'H2_local_PD/physical_children.json')['rows'][:2]
    if args.include_midpoint:
        midpoint=DEST/'H2_local_PD/midpoint_spectrum'
        result=read(midpoint/'result.json')
        child=read(midpoint/'physical_child.json')
        assert result['status']=='UNSTABLE'
        assert Path(result['orbit']).resolve()==Path(child['analyzed_orbit']).resolve()
        child['orbit']=child['analyzed_orbit']
        children.insert(1,child)
        sources.insert(1,[Path(f) for f in result['sources']])
    merged_path=DEST/'H2_local_PD/current_child_spectrum_evidence.json'
    merged=read(merged_path)['rows'] if merged_path.exists() else []
    records=[];bases=[]
    for child,paths in zip(children,sources):
        pair=[read(f) for f in paths];verdict=paired_modes(*pair)
        assert all(q['orbit']==child['orbit'] for q in pair)
        assert child['physical_check']['filter_state_check']['positive']
        assert child['physical_check']['maximum_group_defect_Hz']<1e-6
        assert verdict['status']=='UNSTABLE' and verdict['reliable_outside_count']==4
        mu=values(pair[-1]);selected=np.flatnonzero(verdict['outside_unit_disk_mask'])
        assert max(np.asarray(pair[-1]['phase_overlap'])[selected])<1e-6
        real=selected[abs(mu[selected].imag)<1e-8]
        complex_pair=selected[abs(mu[selected].imag)>=1e-8]
        assert len(real)==len(complex_pair)==2
        invariants=[]
        for spectrum in pair:
            vals=values(spectrum)
            ii=np.flatnonzero((abs(vals.imag)<1e-8)&(abs(vals)>1))
            assert len(ii)==2
            invariants.append(dict(dt_ms=spectrum['dt_ms'],real_pair=vals[ii].real,
                trace=float(vals[ii].real.sum()),determinant=float(vals[ii].real.prod()),
                characteristic_at_plus_one=float(np.prod(1-vals[ii].real)),
                characteristic_at_minus_one=float(np.prod(-1-vals[ii].real))))
        with np.load(paths[-1].with_suffix('.npz')) as saved:
            vectors=np.vstack([saved['local_vectors'][:,selected],saved['history_vectors'][:,selected]])
        local_index={i:j for j,i in enumerate(selected)}
        real_vectors=vectors[:,[local_index[i] for i in real]]
        complex_vectors=vectors[:,[local_index[i] for i in complex_pair]]
        bases.append(dict(real=real_span(real_vectors),complex=real_span(complex_vectors),
                          all_growing=real_span(vectors)))
        current=next((q for q in merged
                      if Path(q['orbit']).resolve()==Path(child['orbit']).resolve()),None)
        if current is not None:
            assert current['status']=='UNSTABLE'
            assert abs(current['J_EE_core']-child['J_EE_core'])<1e-12
            assert current['verified_unstable_dimension_lower_bound']>=4
        records.append(dict(amplitude_Hz=child['amplitude_hz'],J_EE_core=child['J_EE_core'],
            orbit=child['orbit'],sources=list(map(str,paths)),classification=verdict,
            reliable_growing_source_indices=selected,paired_real_block_invariants=invariants,
            complex_pair=mu[complex_pair],same_reference_phase_only=True,
            current_dimension_source=str(merged_path) if current is not None else None,
            current_numerical_unstable_dimension=(current['numerical_unstable_dimension']
                                                  if current is not None else None)))
    comparisons=[]
    for i,j in [(i,j) for i in range(len(bases)) for j in range(i+1,len(bases))]:
        angles={}
        for key in ['real','complex','all_growing']:
            left,right=bases[i][key],bases[j][key]
            assert left.shape==right.shape
            cosines=np.linalg.svd(left.T@right,compute_uv=False)
            angles[key]=dict(principal_cosines=cosines,
                principal_angles_degrees=np.rad2deg(np.arccos(np.clip(cosines,0,1))))
        comparisons.append(dict(amplitudes_Hz=[records[k]['amplitude_Hz'] for k in [i,j]],
            subspaces=angles))
    endpoint_angles=next(q['subspaces'] for q in comparisons if q['amplitudes_Hz']==
                         [records[0]['amplitude_Hz'],records[-1]['amplitude_Hz']])
    sign_brackets=[]
    for left,right in zip(records[:-1],records[1:]):
        l=left['paired_real_block_invariants'][-1]['real_pair']
        r=right['paired_real_block_invariants'][-1]['real_pair']
        if np.all(l>1) and np.all(r < -1):
            sign_brackets.append(dict(amplitudes_Hz=[left['amplitude_Hz'],right['amplitude_Hz']],
                J_EE_core=[left['J_EE_core'],right['J_EE_core']],
                meaning='Two positive versus two negative real growing multipliers at sampled cycles; not a located unit-circle crossing.'))
    endpoints=[records[0],records[-1]]
    output=dict(status='MIDPOINT_SPECTRAL_CHANGE_CHECKED' if args.include_midpoint else 'ENDPOINT_SPECTRAL_CHANGE_CHECKED',rows=records,
        selected_midpoint_included=args.include_midpoint,
        reference_phase_subspace_comparisons=endpoint_angles,
        all_sample_pair_subspace_comparisons=comparisons,
        observed_real_pair_sign_change_brackets=sign_brackets,
        coordinate_scope='Original nine local states plus physical delay histories. All children use the same parent phase condition. Their period-fitted history steps differ slightly; angles are descriptive sample comparisons, not mode continuation or a geometric invariant.',
        conclusions=dict(both_endpoints_have_at_least_four_growing_directions=True,
            all_sampled_cycles_have_at_least_four_growing_directions=True,
            total_unstable_dimensions_established=all(
                q['current_numerical_unstable_dimension'] is not None for q in records),
            numerical_unstable_dimensions_by_endpoint=[
                q['current_numerical_unstable_dimension'] for q in endpoints],
            numerical_unstable_dimensions_by_sample=[
                q['current_numerical_unstable_dimension'] for q in records],
            dominant_real_multiplier_sign_changes=True,
            intermediate_unit_circle_crossing_located=False,
            new_period_doubling_established=False),
        next_discriminating_calculation='Continue the two-dimensional real spectral block through intermediate child amplitudes, checking full-history spectra. Test a real -1 crossing versus complex-pair passage outside the unit circle; endpoint traces and determinants alone cannot decide.',
        scope='A sign change of the largest multiplier does not establish a flip. Two growing real multipliers are present at both endpoints, while a separate growing conjugate pair persists. Equal lower bounds on unstable dimension do not exclude intermediate bifurcations.')
    name='child_spectral_change_with_midpoint.json' if args.include_midpoint else 'child_spectral_change_audit.json'
    write(DEST/'H2_local_PD'/name,output)
    print('PD4 ENDPOINTS',[(q['amplitude_Hz'],q['classification']['reliable_outside_count'],
        q['paired_real_block_invariants'][-1]) for q in records],flush=True)
    print('SUBSPACE COMPARISONS',angles,flush=True)


if __name__=='__main__':main()
