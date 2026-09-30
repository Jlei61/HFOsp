"""Join SCL observations to stability only for the identical physical cycle.

This is a pointwise evidence join, not a scan of all attractors. Preserve
individual contact recruitment separately from qualified group events.
"""
from pathlib import Path
from collections import Counter
from complete_rate_positive_stability import DEST, read, write
from audit_rate_survey_filter_states import fingerprint


def main():
    folder = DEST/'SCL_branch_scan'
    summary_path = folder/'summary.json'
    manifest_path = folder/'manifest.json'
    summary, manifest = read(summary_path), read(manifest_path)
    points = {(q['family'], q['index']): q for q in manifest['points']}
    stability = {}
    for source in sorted((DEST/'sites').glob('[0-9][0-9][0-9].json')):
        q = read(source)
        if q.get('status') not in ['NUMERICALLY_STABLE', 'UNSTABLE']:
            continue
        physical = q['resolution']
        assert physical['status']=='RESOLUTION_CHECKED'
        assert physical['filter_state_check']['positive']
        assert physical['maximum_group_defect_Hz']<.001
        assert physical['minimum_rate_Hz']>=-1e-9
        assert q['relative_waveform_refinement_change']<.02
        assert q['relative_period_change']<.001
        assert q['classification']['status']==q['status']
        if q['status']=='NUMERICALLY_STABLE':
            assert q['classification']['numerical_unstable_dimension']==0
            assert q['classification']['filter_coverage']
            assert q['classification']['section_projection_checked']
        orbit = str(Path(q['analyzed_orbit']).resolve())
        assert Path(physical['orbit']).resolve()==Path(orbit)
        assert orbit not in stability
        stability[orbit] = dict(source=str(source), evidence=q)
    from check_rate_Bleading_return_witness import verified_return_witness
    witness=verified_return_witness()
    if witness is not None:
        orbit=str(Path(witness['evidence']['orbit']).resolve())
        assert orbit not in stability
        stability[orbit]=witness
    rows, unmatched = [], []
    for observation in summary['rows']:
        key = observation['family'], observation['index']
        point = points[key]
        orbit = str(Path(observation['orbit']).resolve())
        assert orbit==str(Path(point['orbit']).resolve())
        evidence = stability.get(orbit)
        if evidence is None:
            unmatched.append(dict(family=key[0], index=key[1], orbit=orbit,
                J_EE_core=observation['J_EE_core']))
            continue
        assert fingerprint(point['orbit'])==point['profile_fingerprint']
        s = evidence['evidence']
        assert abs(s['J_EE_core']-observation['J_EE_core'])<1e-12
        assert abs(s['T_ms']-observation['T_ms'])<1e-7
        raw = read(observation['source'])
        assert Path(raw['orbit']).resolve()==Path(orbit)
        assert raw['profile_fingerprint']==point['profile_fingerprint']
        records = raw['records']
        assert len(records)==4
        individual = [len(r['sustained_SCL_contact_names']) for r in records]
        groups = [r['qualified_events'] for r in records]
        scl_groups = [r['SCL_qualified_events'] for r in records]
        fractions = [r['SCL_fraction_in_qualified_events'] for r in records]
        assert individual==observation['SCL_count']
        assert groups==observation['qualified_groups']
        for n, n_scl, fraction in zip(groups, scl_groups, fractions):
            assert 0<=n_scl<=n
            if n==0:
                assert fraction is None
            else:
                assert fraction==n_scl/n
        rows.append(dict(family=key[0], family_index=key[1], orbit=orbit,
            J_EE_core=s['J_EE_core'], T_ms=s['T_ms'],
            stability=s['status'], numerical_unstable_dimension=s['classification']['numerical_unstable_dimension'],
            stability_source=evidence['source'], observer_source=observation['source'],
            individual_SCL_contacts_by_bin_origin=[r['sustained_SCL_contact_names'] for r in records],
            individual_SCL_counts_by_bin_origin=individual,
            qualified_group_counts_by_bin_origin=groups,
            SCL_qualified_group_counts_by_bin_origin=scl_groups,
            SCL_group_fractions_by_bin_origin=fractions,
            group_metrics_defined_by_bin_origin=[n>0 for n in groups]))
    output = dict(status='IDENTICAL_ORBIT_STABILITY_READOUT_JOIN_COMPLETE',
        manifest_source=str(manifest_path), observer_summary_source=str(summary_path),
        rows=rows, readout_samples_without_matched_stability=unmatched,
        counts_by_status=dict(Counter(q['stability'] for q in rows)),
        stable_samples_with_individual_SCL_recruitment_at_all_bin_origins=[
            dict(family=q['family'], family_index=q['family_index'], J_EE_core=q['J_EE_core'],
                 qualified_group_counts=q['qualified_group_counts_by_bin_origin'],
                 SCL_qualified_group_counts=q['SCL_qualified_group_counts_by_bin_origin'])
            for q in rows if q['stability']=='NUMERICALLY_STABLE' and
                min(q['individual_SCL_counts_by_bin_origin'])>0],
        stable_samples_with_qualified_SCL_events_at_all_bin_origins=[
            dict(family=q['family'], family_index=q['family_index'], J_EE_core=q['J_EE_core'],
                 qualified_group_counts=q['qualified_group_counts_by_bin_origin'],
                 SCL_qualified_group_counts=q['SCL_qualified_group_counts_by_bin_origin'])
            for q in rows if q['stability']=='NUMERICALLY_STABLE' and
                min(q['SCL_qualified_group_counts_by_bin_origin'])>0],
        interval_completeness=False,
        scope='Exact orbit identity and current profile fingerprint are required. Stable/unstable labels apply only to checked samples; no stability is transferred by proximity in J or along a branch. Four bin origins are deterministic observer checks, not independent trials. No qualified groups means undefined group-level metrics, not zero SCL participation. This join neither locates an SCL window boundary nor proves absence on other attractors.')
    write(folder/'stability_conditioned_samples.json', output)
    print('STABILITY-READOUT JOIN', output['counts_by_status'],
          'unmatched', len(unmatched), flush=True)
    print('STABLE INDIVIDUAL SCL RECRUITMENT',
          output['stable_samples_with_individual_SCL_recruitment_at_all_bin_origins'], flush=True)
    print('STABLE QUALIFIED SCL EVENTS',
          output['stable_samples_with_qualified_SCL_events_at_all_bin_origins'], flush=True)


if __name__=='__main__':
    main()
