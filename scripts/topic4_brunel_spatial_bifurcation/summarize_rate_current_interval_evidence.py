"""Current paired-step stability evidence in continuation order.

The old survey summary does not consume the repaired 20260920 profiles.
Keep this audit separate and do not reinterpret legacy single-step spectra
as paired-step evidence. Opposite endpoints identify follow-up intervals,
not their number of crossings or an exhaustive bifurcation diagram.
"""
from complete_rate_positive_stability import paired_modes, read, write, DEST, PERIODIC_OUT
from plot_rate_periodic_completion import families, continuation_breaks
from pathlib import Path
from collections import Counter
import numpy as np
import time


def verified_site(path):
    q=read(path);resolution=q.get('resolution',{})
    if (resolution.get('status')!='RESOLUTION_CHECKED' or
        not resolution.get('filter_state_check',{}).get('positive',False) or
        resolution.get('maximum_group_defect_Hz',1)>=.001 or
        resolution.get('minimum_rate_Hz',-1)<-1e-9):
        return None
    if q.get('relative_waveform_refinement_change',1)>=.02 or q.get('relative_period_change',1)>=.01:
        return None
    actual=Path(q['analyzed_orbit']);z=np.load(actual)
    assert z['r'].shape[1]==935
    assert len(z['r'])==resolution['N']
    assert abs(float(z['J'])-q['J_EE_core'])<1e-12
    assert abs(float(z['T'])-q['T_ms'])<1e-8
    attempts=[]
    providers=[dict(attempt,provider=str(path)) for attempt in q.get('attempts',[])]
    # The quotient calculation completes the same physical orbit's spectrum
    # without replacing the earlier direct worker's immutable evidence.
    supplement=DEST/'ritz_checks/LPC6_left_site40_complement.json'
    if q['index']==40 and supplement.exists():
        providers.append(dict(sources=read(supplement)['sources'],provider=str(supplement)))
    for attempt in providers:
        sources=attempt['sources'];assert len(sources)==2
        spectra=[read(source) for source in sources]
        assert all(Path(v['orbit']).resolve()==actual.resolve() for v in spectra)
        for spectrum in spectra:
            if 'locked_subspace_invariance_defect' in spectrum:
                assert spectrum['locked_subspace_invariance_defect']<1e-6
                assert max(spectrum['complement_filter_relative_changes'])<1e-5
                assert max(spectrum['lift_condition_numbers'])<1e10
        result=paired_modes(*spectra)
        attempts.append(dict(sources=sources,provider=attempt['provider'],classification=result))
    verdicts={r['classification']['status'] for r in attempts}-{'UNRESOLVED'}
    status=next(iter(verdicts)) if len(verdicts)==1 else ('CONFLICT_REVIEW' if verdicts else 'UNRESOLVED')
    dimensions=[r['classification']['numerical_unstable_dimension'] for r in attempts
                if r['classification']['numerical_unstable_dimension'] is not None]
    lower_bound=max((r['classification']['reliable_outside_count'] for r in attempts),default=0)
    if len(set(dimensions))>1 or (dimensions and lower_bound>min(dimensions)):
        status='CONFLICT_REVIEW'
    dimension=dimensions[-1] if dimensions and len(set(dimensions))==1 and status!='CONFLICT_REVIEW' else None
    return dict(site_index=q['index'],status=status,numerical_unstable_dimension=dimension,
        dimension_agreement_across_resolved_attempts=len(set(dimensions))<=1,
        verified_unstable_dimension_lower_bound=lower_bound,
        J_EE_core=q['J_EE_core'],T_ms=q['T_ms'],original_orbit=q['original_orbit'],
        analyzed_orbit=str(actual),source=str(path),attempts=attempts,
        interval_certified=False)


def main():
    fs=families();sites=[]
    for path in sorted((DEST/'sites').glob('[0-9][0-9][0-9].json')):
        site=verified_site(path)
        if site is not None:sites.append(site)
    from check_rate_PD3_parent_display import verified_parent_witnesses
    for q in verified_parent_witnesses():
        sites.append(dict(site_index='PD3_'+q['side'],status=q['status'],
            numerical_unstable_dimension=q['numerical_unstable_dimension'],
            dimension_agreement_across_resolved_attempts=True,
            verified_unstable_dimension_lower_bound=q['verified_unstable_dimension_lower_bound'],
            J_EE_core=q['J_EE_core'],T_ms=q['T_ms'],original_orbit=q['original_orbit'],
            analyzed_orbit=q['orbit'],source=q['source'],attempts=q['spectral_attempts'],
            interval_certified=False))
    by_path={str(Path(q['original_orbit']).resolve()):q for q in sites}
    assert len(by_path)==len(sites), 'Duplicate continuation identities need explicit evidence merging'
    rows=[];brackets=[];restarts=[]
    for family in ['A','B','double','Bleading','single']:
        rr=fs[family];bounds=set(continuation_breaks(rr));samples=[]
        for i,r in enumerate(rr):
            key=str(Path(r['path']).resolve())
            if key in by_path:samples.append(dict(index=i,**by_path[key]))
        accepted=[q for q in samples if q['status'] in ['NUMERICALLY_STABLE','UNSTABLE']]
        # An intermediate unstable sample with an unresolved dimension must
        # not erase a dimension change between the surrounding resolved ones.
        pairs={(l['index'],r['index']):(l,r) for l,r in zip(accepted[:-1],accepted[1:])}
        resolved=[q for q in accepted if q['numerical_unstable_dimension'] is not None]
        for left,right in zip(resolved[:-1],resolved[1:]):
            if left['numerical_unstable_dimension']!=right['numerical_unstable_dimension']:
                pairs[(left['index'],right['index'])]=(left,right)
        for left,right in [pairs[key] for key in sorted(pairs)]:
            opposite=left['status']!=right['status']
            dl,dr=left['numerical_unstable_dimension'],right['numerical_unstable_dimension']
            dimension_change=dl is not None and dr is not None and dl!=dr
            if not (opposite or dimension_change):continue
            bracket=dict(family=family,ends=[{k:q[k] for k in ['index','site_index','status',
                'numerical_unstable_dimension','J_EE_core','T_ms','source','original_orbit','analyzed_orbit']}
                for q in [left,right]],opposite_stability=opposite,dimension_change=dimension_change,
                intervening_continuation_samples=right['index']-left['index']-1,
                scope='Sampled change only; known or multiple critical points may lie inside. No interval completeness claim.')
            if any(left['index']<i<=right['index'] for i in bounds):
                bracket['scope']='Across a continuation restart, not a consecutive branch interval.'
                restarts.append(bracket)
            else:brackets.append(bracket)
        rows.append(dict(family=family,total_continuation_samples=len(rr),
            paired_profile_checked_sites=samples,classification_counts=dict(Counter(q['status'] for q in samples)),
            every_interval_certified=False))
    result=dict(status='CURRENT_PAIRED_EVIDENCE_AUDITED',timestamp=time.time(),
        model='Frozen 400-cell / 935-population spatial rate DDE',rows=rows,
        opposite_or_dimension_change_brackets=brackets,restart_comparisons=restarts,
        sites=sites,legacy_single_step_spectra_promoted=False,global_branch_completeness=False,
        scope='Fresh classifications recomputed from paired full-history spectra on physically checked profiles. '
              'An incomplete finer Arnoldi result does not erase a separately reliable growing mode; '
              'opposite accepted verdicts across time meshes trigger review. '
              'Equal endpoint verdicts do not exclude interior crossings.')
    write(DEST/'current_interval_evidence.json',result)
    print('CURRENT PAIRED SITES',dict(Counter(q['status'] for q in sites)),flush=True)
    for q in brackets:
        print('BRACKET',q['family'],[(e['site_index'],e['J_EE_core'],e['status'],e['numerical_unstable_dimension'])
            for e in q['ends']],flush=True)


if __name__=='__main__':main()
