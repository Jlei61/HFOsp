"""Current physical, spectral and original-readout evidence for cases b/c.

Both cases use exactly J=.942 in the unchanged full spatial rate DDE.
Local Floquet stability does not identify their basin boundary or switching
mechanism. Repeated deterministic cycles are not independent observations.
"""
from complete_rate_positive_stability import DEST, PERIODIC_OUT, read, write, paired_modes, np, RateField
from plot_rate_sameJ_burst_pair import observe_cycle, OLD
from plot_rate_focused_composite import load_cases
from audit_rate_filter_states import filter_state_minima
from audit_rate_survey_filter_states import fingerprint
from pathlib import Path
import time


def main():
    s=RateField();assert s.P==935
    manifest_source=PERIODIC_OUT/'composite_case_resolution.json'
    manifest=read(manifest_source);assert manifest['status']=='COMPLETE'
    cases={q['letter']:q for q in load_cases(s)}
    contract_source=OLD/'observer_firing.json';contract=read(contract_source)
    previous=read(DEST/'SCL_case_and_branch_readout_summary.json')
    previous={q['label']:q for q in previous['rows']}
    rows=[]
    for label in ['b','c']:
        checked=next(q for q in manifest['rows'] if q['case']==label)
        case=cases[label];orbit=Path(checked['orbit'])
        assert Path(case['orbit']).resolve()==orbit.resolve()
        assert Path(previous[label]['orbit']).resolve()==orbit.resolve()
        check=checked['resolution']
        assert check['status']=='RESOLUTION_CHECKED'
        assert Path(check['orbit']).resolve()==orbit.resolve()
        assert check['maximum_group_defect_Hz']<.001
        before=fingerprint(orbit)
        with np.load(orbit) as z:
            r=z['r'];T=float(z['T']);J=float(z['J'])
            assert r.shape[1]==935 and abs(J-.942)<1e-12
            filters=filter_state_minima(s,r,T);assert filters['positive']
            assert check['minimum_rate_Hz']>=-1e-9
        spectra_sources=[PERIODIC_OUT/'poincare_floquet'/f'{orbit.stem}_dt{dt}.json'
                         for dt in ['0.1','0.05']]
        spectra=[read(p) for p in spectra_sources]
        for q in spectra:
            assert Path(q['orbit']).resolve()==orbit.resolve()
            assert abs(q['J_EE_core']-J)<1e-12 and abs(q['T_ms']-T)<1e-7
        verdict=paired_modes(*spectra)
        assert verdict['status']=='NUMERICALLY_STABLE'
        assert verdict['numerical_unstable_dimension']==0
        assert verdict['section_projection_checked'] and verdict['filter_coverage']
        assert all(verdict['reliable_mode_mask'])
        observation=observe_cycle(case['r'],T,s,contract)
        records=[]
        for record,old in zip(observation['records'],previous[label]['records']):
            for key in ['bin_origin_ms','qualified_events','SCL_qualified_events',
                        'sustained_SCL_contact_names']:
                assert record[key]==old[key],(label,key)
            records.append({k:record[k] for k in ['bin_origin_ms','detected_events',
                'qualified_events','SCL_qualified_events','sustained_contact_names',
                'sustained_SCL_contact_names','contact_peak_to_threshold','metrics']})
        assert fingerprint(orbit)==before
        rows.append(dict(case=label,J_EE_core=J,T_ms=T,orbit=str(orbit),
            profile_fingerprint=before,resolution_source=str(manifest_source),
            resolution=check,current_constituent_filter_check=filters,
            spectra_sources=list(map(str,spectra_sources)),classification=verdict,
            maximum_checked_multiplier_modulus=verdict['maximum_returned_modulus'],
            minimum_returned_mode_margin_to_unit_circle=min(1-abs(mu)-margin for mu,margin in
                zip(np.asarray(verdict['multipliers']),verdict['per_mode_margin'])),
            observer_records=records,interior_window_ms=observation['interior_window_ms'],
            interior_cycles=observation['interior_cycles']))
    assert rows[0]['J_EE_core']==rows[1]['J_EE_core']
    assert all(q['qualified_events']==0 for q in rows[0]['observer_records'])
    assert all(q['qualified_events']==16 and q['SCL_qualified_events']==8
               for q in rows[1]['observer_records'])
    result=dict(status='SAME_J_TWO_NUMERICALLY_STABLE_PERIODIC_STATES',timestamp=time.time(),
        model='Frozen 400-cell / 935-population spatial rate DDE',J_EE_core=.942,
        observer_source=str(contract_source),contact_names=contract['contact_names'],rows=rows,
        stability_scope='Local numerical orbital stability: both original full-delay Poincare spectra pass paired-step, phase-removal, residual and coverage checks on the identical physical profiles.',
        SCL_scope='Same original observer and main-figure phase alignment. The burst cycle has two qualified events per full period; one includes SCL. Repeating eight exact cycles does not provide independent trials.',
        basin_boundary_located=False,switching_bifurcation_identified=False,
        global_branch_connection_established=False,
        scope='Numerical coexistence of a small periodic oscillation and an alternating-burst periodic state at the same model parameter. This is not resting-versus-burst coexistence, autonomous switching, a noise-driven irregular state, or native-SNN coexistence verification.')
    write(DEST/'sameJ_small_burst_stability_readout.json',result)
    print('SAME-J CHECK',[(q['case'],q['T_ms'],q['maximum_checked_multiplier_modulus'],
        q['minimum_returned_mode_margin_to_unit_circle']) for q in rows],flush=True)


if __name__=='__main__':
    main()
