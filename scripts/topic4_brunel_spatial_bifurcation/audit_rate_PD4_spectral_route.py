"""Separate a sampled complex detour from a later possible negative flip.

Only actual saved spectra are used. Missing fine meshes stay pending, and
no interpolation between eigenvalues is promoted to a critical point.
"""
from complete_rate_positive_stability import DEST, PERIODIC_OUT, read, write, values, paired_modes, np
from audit_rate_PD4_child_spectral_change import real_span
from pathlib import Path
import time


def main():
    folder=DEST/'H2_local_PD'
    children=read(folder/'physical_children.json')['rows']
    specs=[
        (read(folder/'midpoint_spectrum/physical_child.json'), 'PD4_child_midpoint_20260920_k6'),
        (read(folder/'trace_center_spectrum/physical_child.json'), 'PD4_child_trace_center_20260920_k6'),
        (children[1], 'PD4_child_index1_full_k6_20260920'),
        (children[2], 'PD4_child_index2_full_k6_20260920')]
    rows=[];bases=[]
    for profile,label in specs:
        files=[PERIODIC_OUT/f'poincare_floquet/{label}_dt{dt}.json' for dt in ['0.05','0.025']]
        coarse=read(files[0]);mu=values(coarse);amp=profile['amplitude_hz']
        orbit=Path(profile.get('analyzed_orbit',profile['orbit'])).resolve()
        assert Path(coarse['orbit']).resolve()==orbit
        assert profile['physical_check']['filter_state_check']['positive']
        assert profile['physical_check']['maximum_group_defect_Hz']<1e-6
        if amp<.034:
            selected=np.flatnonzero((abs(mu.imag)<1e-8)&(mu.real>1))
            selection='The two positive real growing eigenpairs at the lower sample.'
        elif amp<.04:
            selected=np.flatnonzero((abs(mu.imag)>1)&(abs(mu)>10))
            selection='The larger-modulus complex pair at the intervening sample.'
        else:
            selected=np.flatnonzero((abs(mu.imag)<1e-8)&(mu.real<0))
            selection='The two negative real eigenpairs at each later sample.'
        assert len(selected)==2
        other=np.flatnonzero((abs(mu.imag)>1e-6)&(abs(mu)>1)&~np.isin(np.arange(len(mu)),selected))
        assert len(other)==2
        with np.load(files[0].with_suffix('.npz')) as z:
            chosen=np.r_[selected,other]
            vectors=np.r_[z['local_vectors'][:,chosen],z['history_vectors'][:,chosen]]
        bases.append(dict(block=real_span(vectors[:,:2]),four=real_span(vectors)))
        spectral_rows=[]
        pair=None
        for i,file in enumerate(files):
            if not file.exists():continue
            q=read(file);v=values(q)
            assert Path(q['orbit']).resolve()==orbit
            # Matching is within the same orbit and across time meshes only;
            # cross-amplitude mode identity is deliberately not asserted.
            if i==0:ids=selected
            else:
                from scipy.optimize import linear_sum_assignment
                a,b=linear_sum_assignment(abs(mu[:,None]-v[None,:]))
                mapping=dict(zip(a,b));ids=np.array([mapping[j] for j in selected])
                pair=paired_modes(coarse,q)
            block=v[ids];trace=np.sum(block);det=np.prod(block)
            assert abs(trace.imag)<1e-5 and abs(det.imag)<1e-5
            spectral_rows.append(dict(source=str(file),dt_ms=q['dt_ms'],selected_indices=ids,
                multipliers=block,trace=float(trace.real),determinant=float(det.real),
                discriminant=float((trace*trace-4*det).real),
                block_characteristic_at_minus_one=float(np.prod(-1-block).real),
                moduli=abs(block),phase_tangent_relative_defect=q['phase_tangent_relative_defect'],
                normalized_eigen_residuals=np.asarray(q['residuals'])[ids]/np.maximum(1,abs(block)),
                phase_overlaps=np.asarray(q['phase_overlap'])[ids]))
        rows.append(dict(amplitude_Hz=amp,J_EE_core=profile['J_EE_core'],orbit=str(orbit),
            selection_rule=selection,spectra=spectral_rows,paired_classification=pair,
            time_step_pair_available=len(spectral_rows)==2,
            selected_pair_reliable=(bool(all(pair['reliable_mode_mask'][j] for j in ids))
                                    if pair is not None else None),
            selected_pair_phase_projected=bool(all(max(q['phase_overlaps'])<1e-6 for q in spectral_rows))))
    angles=[]
    for i in range(len(rows)-1):
        item=dict(amplitude_interval_Hz=[rows[i]['amplitude_Hz'],rows[i+1]['amplitude_Hz']])
        for key in ['block','four']:
            l,r=bases[i][key],bases[i+1][key];assert l.shape==r.shape
            cs=np.linalg.svd(l.T@r,compute_uv=False)
            item[key+'_principal_angles_degrees']=np.degrees(np.arccos(np.clip(cs,0,1)))
        angles.append(item)
    mid=rows[1]['spectra'][-1]
    assert mid['discriminant']<0 and min(mid['moduli'])>10
    left,right=rows[2]['spectra'][-1],rows[3]['spectra'][-1]
    assert left['block_characteristic_at_minus_one']>0>right['block_characteristic_at_minus_one']
    out=dict(status='SAVED_SPECTRAL_ROUTE_AUDITED',timestamp=time.time(),rows=rows,
        adjacent_reference_phase_subspace_checks=angles,
        complex_pair_at_intermediate_sample=True,
        complex_probe_time_step_pair_available=rows[1]['time_step_pair_available'],
        all_selected_pairs_have_paired_residual_and_phase_checks=bool(all(
            q['time_step_pair_available'] and q['selected_pair_reliable'] and
            q['selected_pair_phase_projected'] for q in rows)),
        later_negative_flip_candidate_time_step_pair_available=rows[3]['time_step_pair_available'],
        sampled_complex_pair_is_outside_unit_circle=True,
        intermediate_eigenvalue_collisions_located=False,
        later_minus_one_crossing_located=False,new_period_doubling_established=False,
        scope='Actual sampled eigenpairs on physical cycles; a complex point distinguishes a possible detour from an inferred direct positive-to-negative jump. Paired-step availability and residual/phase checks are stated separately for the selected pair; total unstable dimensions remain subject to the full spectra. Same-phase subspace angles are descriptive and do not establish cross-amplitude mode continuation, absence of intervening unit-circle crossings, or a new dynamical bifurcation.')
    write(folder/'spectral_route_evidence.json',out)
    print('SPECTRAL ROUTE',[(q['amplitude_Hz'],q['spectra'][-1]['trace'],q['spectra'][-1]['determinant'],
        q['spectra'][-1]['discriminant'],q['time_step_pair_available']) for q in rows],flush=True)
    print('ADJACENT SUBSPACES',angles,flush=True)


if __name__=='__main__':main()
