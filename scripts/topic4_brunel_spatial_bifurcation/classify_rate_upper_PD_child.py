"""Combine nonlinear departure and independent Floquet evidence for PD2."""
from rate_stability_coverage import *
from scipy.optimize import linear_sum_assignment
from audit_rate_filter_states import filter_state_minima


def values(q):
    z=np.asarray(q['multipliers']);return z[:,0]+1j*z[:,1] if z.ndim==2 else z.astype(complex)


def main():
    critical=read(PERIODIC_OUT/'PD_double_upper_validation.json')
    assert critical['status']=='VALIDATED_PD'
    departure=read(PERIODIC_OUT/'PD_upper_child_departure.json')
    fine_orbits=[q for q in departure['rows'] if q['N']==4096]
    branch=max(fine_orbits,key=lambda q:q['switch_coordinate']);stem=Path(branch['orbit']).stem
    spectra=[]
    for dt in [.1,.05]:
        path=PERIODIC_OUT/'floquet'/f'{stem}_dt{dt:g}.json';q=read(path)
        assert Path(q['orbit']).resolve()==Path(branch['orbit']).resolve()
        spectra.append((path,q,assess(q)))
    parent=[]
    for path in (PERIODIC_OUT/'floquet').glob('arcDouble*.json'):
        q=read(path)
        # The directory also contains assessment-only sidecars.
        if not all(k in q for k in ['T_ms','J_EE_core','multipliers','residuals']):continue
        if abs(q['T_ms']/critical['T_ms']-1)>.03 or abs(q['J_EE_core']-critical['J_EE_core'])>5e-5:continue
        verdict=assess(q);mu=values(q)
        rel=np.asarray(q['residuals'])/np.maximum(1,abs(mu))
        negative=mu[(abs(mu.imag)<1e-6)&(mu.real < -1-verdict['margin'])&(rel<1e-6)]
        if verdict['status']=='NUMERICALLY_STABLE' and q['J_EE_core']<critical['J_EE_core']:
            parent.append(dict(side='below',source=str(path),J_EE_core=q['J_EE_core'],status=verdict['status']))
        if len(negative) and q['J_EE_core']>critical['J_EE_core']:
            parent.append(dict(side='above',source=str(path),J_EE_core=q['J_EE_core'],negative_unstable_multipliers=negative,status='UNSTABLE'))
    witnesses=[]
    for side in ['below','above']:
        candidates=[q for q in parent if q['side']==side]
        if candidates:witnesses.append(min(candidates,key=lambda q:abs(q['J_EE_core']-critical['J_EE_core'])))
    v0,v1=[values(q) for _,q,_ in spectra];i,j=linear_sum_assignment(abs(v0[:,None]-v1[None,:]))
    step_change=float(max(abs(v0[i]-v1[j])/np.maximum(1,abs(v1[j]))))
    path,q,verdict=spectra[-1];v=values(q);neutral=q['identified_neutral_index']
    nontrivial=np.delete(v,neutral) if neutral is not None else v
    leading=nontrivial[np.argmax(abs(nontrivial))]
    accuracy=read(PERIODIC_OUT/'PDupperchild_a10_N4096_accuracy.json')
    assert Path(accuracy['orbit']).resolve()==Path(branch['orbit']).resolve()
    s=RateField();z=np.load(branch['orbit']);r=z['r'];regional=np.array([s.regional_rates(x) for x in r])
    counts=[]
    for k in range(2):
        pk=find_peaks(np.tile(regional[:,k],3),height=20,prominence=10,
                      distance=max(1,round(60*len(r)/branch['T_ms'])))[0]
        counts.append(int(np.sum((pk>=len(r))&(pk<2*len(r)))))
    physical_children=[]
    for row in fine_orbits:
        child=np.load(row['orbit'])
        physical_children.append(dict(orbit=row['orbit'],
            **filter_state_minima(s,child['r'],float(child['T']))))
    physical_pass=all(row['positive'] for row in physical_children)
    passed=(critical.get('full_acceptance',False) and physical_pass
        and len(witnesses)==2 and len(fine_orbits)>=3 and departure['all_fine_shifts_positive']
        and departure['mesh_comparison']['coefficient_relative_difference']<.01
        and departure['fine_coefficient_relative_spread']<.1
        and all(r['half_period_relative_mismatch']>1e-5 and r['minimum_collocation_rate_Hz']>0 for r in fine_orbits)
        and accuracy['maximum_group_defect_Hz']<.1 and max(accuracy['regional_defect_Hz'])<.001
        and accuracy['minimum_rate_Hz']>0 and step_change<.01
        and verdict['status']=='NUMERICALLY_STABLE'
        and spectra[-1][1]['phase_tangent_relative_defect']<spectra[0][1]['phase_tangent_relative_defect']/3)
    result=dict(status='SUPERCRITICAL_PD' if passed else 'VALIDATION_INCOMPLETE',
        label='PD2',J_EE_core=critical['J_EE_core'],parent_T_ms=critical['T_ms'],
        child_side='J > J_PD; parent-unstable side',child_mu=leading,
        child_orbit=branch['orbit'],child_T_ms=branch['T_ms'],
        core_A_B_peaks_per_full_child_period=counts,
        child_floquet_source=str(path),paired_spectra=[dict(source=str(p),assessment=a) for p,_,a in spectra],
        matched_multiplier_step_change=step_change,parent_side_witnesses=witnesses,
        departure_source=str(PERIODIC_OUT/'PD_upper_child_departure.json'),continuous_child_check=accuracy,
        constituent_filter_checks=physical_children,
        full_physical_child_checks=physical_pass,
        scope='Local supercritical period doubling when all checks pass: a stable 2T child emerges on the parent-unstable side. This is regular periodic activity, with peak counts measured separately; it does not establish an irregular attractor or a global connection to another burst family.')
    write(PERIODIC_OUT/'PD_upper_child_classification.json',result)
    print('PD2 CHILD CLASSIFICATION',result,flush=True)


if __name__=='__main__':main()
