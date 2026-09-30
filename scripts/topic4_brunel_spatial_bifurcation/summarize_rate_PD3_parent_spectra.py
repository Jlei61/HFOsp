"""Merge exact-orbit PD3 parent spectra; keep lower bounds distinct from counts."""
from complete_rate_positive_stability import DEST, read, write, paired_modes
from pathlib import Path
import time


def current_parent_evidence():
    folder=DEST/'PD3_parent_spectra'
    witnesses=read(DEST/'PD3_parent_witnesses/result.json')
    assert witnesses['status']=='PHYSICAL_PARENT_CROSSING_CHECKED'
    rows=[]
    for witness in witnesses['rows']:
        side=witness['side'];physical=witness['physical'];orbit=Path(witness['orbit']).resolve()
        assert Path(physical['orbit']).resolve()==orbit
        assert physical['filter_state_check']['positive'] and physical['maximum_group_defect_Hz']<1e-6
        raw=folder/f'{side}.json'
        sources=[dict(sources=a['sources'],provider=str(raw)) for a in read(raw)['attempts']]
        supplement=DEST/'ritz_checks'/f'PD3_{side}_remaining_k8_20260920.json'
        if supplement.exists():
            sources.append(dict(sources=read(supplement)['sources'],provider=str(supplement)))
        attempts=[]
        for item in sources:
            pair=[read(p) for p in item['sources']]
            assert len(pair)==2 and all(Path(p['orbit']).resolve()==orbit for p in pair)
            assert all(abs(p['J_EE_core']-witness['J_EE_core'])<1e-12 for p in pair)
            for p in pair:
                if 'locked_subspace_invariance_defect' in p:
                    assert p['locked_subspace_invariance_defect']<1e-6
                    assert max(p['complement_filter_relative_changes'])<1e-5
                    assert max(p['lift_condition_numbers'])<1e10
            attempts.append(dict(**item,classification=paired_modes(*pair)))
        accepted=[a['classification'] for a in attempts if a['classification']['status']!='UNRESOLVED']
        verdicts={a['status'] for a in accepted}
        dimensions={a['numerical_unstable_dimension'] for a in accepted if a['numerical_unstable_dimension'] is not None}
        lower=max((a['reliable_outside_count'] for a in accepted),default=0)
        dimension=next(iter(dimensions)) if len(dimensions)==1 else None
        conflict=len(verdicts)>1 or len(dimensions)>1 or (dimension is not None and lower>dimension)
        status='CONFLICT_REVIEW' if conflict else next(iter(verdicts)) if verdicts else 'UNRESOLVED'
        rows.append(dict(side=side,J_EE_core=witness['J_EE_core'],orbit=str(orbit),
            physical_source=str(DEST/'PD3_parent_witnesses/result.json'),status=status,
            numerical_unstable_dimension=None if conflict else dimension,
            verified_unstable_dimension_lower_bound=lower,attempts=attempts))
    counts=[q['numerical_unstable_dimension'] for q in rows]
    return dict(status='CURRENT_PAIRED_PARENT_EVIDENCE_MERGED',
        timestamp=time.time(),rows=rows,both_dimensions_resolved=all(n is not None for n in counts),
        sampled_dimension_change=counts[1]-counts[0] if all(n is not None for n in counts) else None,
        global_branch_completeness=False,live_worker_snapshots_modified=False,
        scope='Known growing modes are lower bounds until coverage and all eigenpair checks pass. '
              'A one-direction change would be compatible with PD3, but does not exclude additional compensating crossings between the sampled parents.')


def main():
    result=current_parent_evidence()
    write(DEST/'PD3_parent_spectra/current_evidence.json',result)
    print('PD3 PARENT EVIDENCE',[(q['side'],q['status'],q['verified_unstable_dimension_lower_bound'],
        q['numerical_unstable_dimension']) for q in result['rows']],flush=True)


if __name__=='__main__':main()
