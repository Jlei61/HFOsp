"""Promote PD3 only from corrected physical children and parent witnesses."""
from rate_periodic import *


def main():
    folder=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920/PD3_child_followup')
    mode_source=folder/'physical_mode_result.json'
    witness_source=folder.parent/'PD3_parent_witnesses/result.json'
    evidence=read(mode_source);witnesses=read(witness_source)
    assert evidence['status']=='PHYSICAL_CHILD_MODES_CHECKED_REVIEW_PENDING'
    assert evidence['full_physical_child_checks']
    assert witnesses['status']=='PHYSICAL_PARENT_CROSSING_CHECKED'
    p=PERIODIC_OUT;root=read(p/'PD_A_return_validation.json')
    parent=root['filter_state_followup']
    assert root['status']=='VALIDATED_PD' and parent['status']=='REFINED_PARENT_AND_MODE_CHECKED'
    assert parent['continuous_orbit_check']['filter_state_check']['positive']
    assert parent['continuous_orbit_check']['maximum_group_defect_Hz']<1e-6
    assert root['mesh_J_difference']<1e-7 and parent['antiperiodic_relative_residual']<2e-9
    parent_checks=sorted(parent['direct_monodromy_checks'],key=lambda q:-q['dt_ms'])
    parent_errors=np.array([q['minus_one_relative_defect'] for q in parent_checks])
    assert len(parent_errors)>=3 and parent_errors[-1]<1e-4
    assert np.all(parent_errors[:-1][-2:]/parent_errors[1:][-2:]>3)
    sides={q['side']:q for q in witnesses['rows']}
    assert sides['below']['J_EE_core']<root['J_EE_core']<sides['above']['J_EE_core']
    assert sides['below']['negative_multiplier']<-1<sides['above']['negative_multiplier']<0
    assert all(q['physical']['filter_state_check']['positive'] for q in sides.values())
    geometry=evidence['physical_geometry'];lo,hi=evidence['radial_modes']
    target=Path(hi['orbit']);mu=hi['multiplier'];gap=1-mu
    assert Path(lo['orbit']).resolve()==target.resolve() and 0<mu<1
    assert max(lo['residual'],hi['residual'])<2e-9
    assert abs(mu-lo['multiplier'])<min(1e-5,gap/100)
    segments=read(evidence['segmented_checks_source'])
    assert segments['status']=='COMPLETE' and Path(segments['orbit']).resolve()==target.resolve()
    checks=sorted(segments['checks'],key=lambda q:-q['dt_ms'])
    for key in ['maximum_mode_relative_defect','maximum_phase_relative_defect']:
        errors=np.array([q[key] for q in checks])
        assert len(errors)>=3 and errors[-1]<min(1e-4,gap/100)
        assert np.all(errors[:-1][-2:]/errors[1:][-2:]>3)
    assert min(q['minimum_negative_control_defect'] for q in checks)>.05
    s=RateField();weights=s.geo['group_size']/s.geo['group_size'].sum()
    z=np.load(target);r=resample(z['r'],hi['N'],axis=0)
    seed=np.load(parent['mode'])['u'].real
    seed=resample(np.r_[seed,-seed],hi['N'],axis=0)
    v=np.load(p/f'PD3child_physical_radial_20260920_mode_N{hi["N"]}.npz')['u']
    phase=np.fft.irfft(np.fft.rfft(r,axis=0)*(2j*np.pi*np.arange(hi['N']//2+1))[:,None],n=hi['N'],axis=0)
    def inner(x,y):return float(np.mean(np.sum(x*y*weights,axis=1)))
    def quotient(x):return x-phase*(inner(x,phase)/inner(phase,phase))
    u,v=quotient(seed),quotient(v)
    overlap=abs(inner(u,v))/np.sqrt(inner(u,u)*inner(v,v));assert overlap>.9
    departure=[]
    for row in geometry:
        assert row['physical']['filter_state_check']['positive'] and row['physical']['maximum_group_defect_Hz']<1e-5
        assert row['J_shift']<0
        zz=np.load(row['orbit']);rr=zz['r'];half=len(rr)//2
        odd=(rr[half:]-rr[:half])*500
        departure.append(dict(N=len(rr),orbit=row['orbit'],J_EE_core=float(zz['J']),
            J_shift=float(zz['J'])-root['J_EE_core'],
            odd_rate_RMS_Hz=float(np.sqrt(np.mean(odd*odd,axis=0)@weights)),
            physical=row['physical']))
    departure.sort(key=lambda q:q['odd_rate_RMS_Hz'])
    coefficients=np.array([q['J_shift']/q['odd_rate_RMS_Hz']**2 for q in departure[:3]])
    assert np.ptp(coefficients)/abs(np.mean(coefficients))<.1
    spectra=evidence['dominant_spectra'];assert len(spectra)>=2
    multipliers=[]
    for q in spectra:
        assert Path(q['orbit']).resolve()==target.resolve()
        raw=np.asarray(q['multipliers']);value=complex(*raw[0]) if raw.ndim==2 else complex(raw[0])
        assert abs(value)>2 and q['residuals'][0]/abs(value)<1e-6
        multipliers.append(value)
    assert abs(multipliers[0]/multipliers[-1]-1)<.01
    regional=np.array([s.regional_rates(x) for x in z['r']]);counts=[]
    for k in [0,1]:
        peaks=find_peaks(np.tile(regional[:,k],3),height=20,prominence=10,
            distance=round(60*len(z['r'])/float(z['T'])))[0]
        counts.append(int(((peaks>=len(z['r']))&(peaks<2*len(z['r']))).sum()))
    previous_departure=p/'PD_return_child_departure.json'
    archived=folder/'PD_return_child_departure_before_physical_review.json'
    if previous_departure.exists() and not archived.exists():write(archived,read(previous_departure))
    write(previous_departure,dict(status='PHYSICAL_DOUBLED_BRANCH_CHECKED',rows=departure[:3],
        physical_profiles_source=evidence['source_profiles'],
        scope='Local departure from the three corrected small children; all-population neuron-weighted odd-component RMS. The larger child is used separately for mode and readout checks.'))
    result=dict(status='SUPERCRITICAL_PD',label='PD3',child_stability='UNSTABLE',
        full_physical_child_checks=True,display_title='Supercritical PD3 | unstable doubled child',
        J_EE_core=root['J_EE_core'],parent_T_ms=root['T_ms'],child_orbit=str(target),child_T_ms=float(z['T']),
        child_side='J < J_PD3; parent flip direction is unstable on this side',child_mu=mu,
        critical_mode_phase_quotient_seed_overlap=overlap,
        inherited_unstable_multiplier=multipliers[-1],
        radial_mode_mesh_difference=abs(mu-lo['multiplier']),
        physical_mode_source=str(mode_source),physical_parent_crossing_source=str(witness_source),
        departure_source=str(previous_departure),segmented_mode_checks_source=evidence['segmented_checks_source'],
        core_A_B_peaks_per_full_child_period=counts,
        criticality_basis='Positive corrected children depart quadratically onto the parent flip-unstable side. The matched child radial multiplier is inside the unit circle. Independent full-delay propagation and finer temporal meshes agree. Other verified growing modes keep the child unstable.',
        scope='Local supercritical flip within an already unstable periodic family. No stable burst onset, global branch connection or exhaustive unstable-dimension count is asserted.')
    write(p/'PD_return_child_classification.json',result)
    root.update(criticality='SUPERCRITICAL_PD',child_stability='UNSTABLE',
        full_acceptance=True,accepted_parent_mesh_N=parent['N'],accepted_mode=parent['mode'],
        child_validation_status='PHYSICAL_CHILD_AND_PARENT_CROSSING_CHECKED',
        child_classification_source=str(p/'PD_return_child_classification.json'),
        scope=result['scope'])
    write(p/'PD_A_return_validation.json',root)
    print('PHYSICAL PD3 CLASSIFICATION',result,flush=True)


if __name__=='__main__':main()
