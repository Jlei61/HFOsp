"""Supply exact PD3 parent witnesses to plots and continuation-order audits.

Recompute paired spectral classifications from their original files. Preserve
the coarse continuation identity separately from the physical fine orbit.
"""
from complete_rate_positive_stability import DEST, PERIODIC_OUT, read, np
from summarize_rate_PD3_parent_spectra import current_parent_evidence
from pathlib import Path


def verified_parent_witnesses():
    source=DEST/'PD3_parent_witnesses/result.json'
    if not source.exists():
        return []
    witnesses=read(source)
    assert witnesses['status']=='PHYSICAL_PARENT_CROSSING_CHECKED'
    root=read(PERIODIC_OUT/'PD_A_return_validation.json')
    assert root['full_acceptance']
    assert abs(root['J_EE_core']-witnesses['critical_J'])<1e-12
    current=current_parent_evidence()
    rows=[]
    for q in current['rows']:
        witness=next(w for w in witnesses['rows'] if w['side']==q['side'])
        if q['status'] not in ['NUMERICALLY_STABLE','UNSTABLE']:
            continue
        # Metadata live beside the public orbit link, not the relocated array.
        physical=witness['physical'];orbit=Path(witness['orbit'])
        assert Path(q['orbit']).resolve()==orbit.resolve()
        assert Path(physical['orbit']).resolve()==orbit.resolve()
        assert physical['status']=='RESOLUTION_CHECKED'
        assert physical['minimum_rate_Hz']>=-1e-9
        assert witness['relative_waveform_change']<.001
        assert witness['relative_period_change']<1e-6
        assert (q['J_EE_core']<witnesses['critical_J'])==(q['side']=='below')
        accepted=[a['classification'] for a in q['attempts']
                  if a['classification']['status']==q['status']]
        assert accepted and all(a['section_projection_checked'] for a in accepted)
        selected=max(accepted,key=lambda a:(a['numerical_unstable_dimension'] is not None,
                                           a['reliable_outside_count']))
        assert selected['numerical_unstable_dimension']==q['numerical_unstable_dimension']
        with np.load(orbit) as z:
            assert z['r'].shape==(physical['N'],935)
            assert abs(float(z['J'])-q['J_EE_core'])<1e-12
            assert abs(float(z['T'])-physical['T_ms'])<1e-8
        meta=read(orbit.with_suffix('.json'))
        assert abs(meta['J_EE_core']-q['J_EE_core'])<1e-12
        assert abs(meta['T_ms']-physical['T_ms'])<1e-8
        rows.append(dict(family='A',side=q['side'],status=q['status'],
            orbit=str(orbit),original_orbit=witness['source'],source=str(source),
            J_EE_core=q['J_EE_core'],T_ms=physical['T_ms'],
            mean_rates_hz=meta['mean_rates_hz'],classification=selected,
            numerical_unstable_dimension=q['numerical_unstable_dimension'],
            verified_unstable_dimension_lower_bound=q['verified_unstable_dimension_lower_bound'],
            spectral_attempts=q['attempts'],interval_certified=False))
    return rows
