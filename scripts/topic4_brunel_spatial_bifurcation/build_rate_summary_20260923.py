"""Freeze a dated all-rate overview from completed physical/spectral evidence."""
import sys
from pathlib import Path
import numpy as np
import plot_rate_branch_completion as plot
from complete_rate_positive_stability import paired_modes, read, write, RateField
from plot_rate_sameJ_burst_pair import observe_cycle, OLD
from audit_rate_survey_filter_states import fingerprint

DEST=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_summary_20260923')
PUBLIC=plot.ROOT/'results/topic4_sef_hfo/interictal_rate_summary_20260923'
DATA=plot.DATA


def new_arm():
    source=DATA/'Bleading_stable_side/checked_segment.json'
    segment=read(source)
    assert segment['status']=='SAMPLED_PASS'
    assert len(segment['included_orbits'])==len(segment['checks'])==80
    rows=[]
    for path,check in zip(segment['included_orbits'],segment['checks']):
        assert Path(path).resolve()==Path(check['orbit']).resolve()
        assert check['maximum_group_defect_Hz']<.001
        assert check['minimum_rate_Hz']>=-1e-9 and check['filter_state_check']['positive']
        row=read(Path(path).with_suffix('.json'))
        assert abs(row['J_EE_core']-check['J_EE_core'])<1e-12
        rows.append(row)
    assert np.all(np.diff([q['J_EE_core'] for q in rows])>0)
    s=RateField();contract=read(OLD/'observer_firing.json');witnesses=[]
    for index in [0,40,79]:
        source=DATA/f'Bleading_stable_side/witness_{index:04d}.json'
        raw=read(source);orbit=Path(raw['orbit'])
        assert orbit.resolve()==Path(rows[index]['path']).resolve()
        verdicts=[]
        for attempt in raw['attempts']:
            pair=[read(p) for p in attempt['sources']]
            assert all(Path(p['orbit']).resolve()==orbit.resolve() for p in pair)
            verdicts.append(paired_modes(*pair))
        accepted=[q for q in verdicts if q['status']!='UNRESOLVED']
        assert accepted and {q['status'] for q in accepted}=={raw['status']}
        with np.load(orbit) as z:
            assert z['r'].shape==(4096,935)
            observed=observe_cycle(z['r'],float(z['T']),s,contract)
        records=[{k:r[k] for k in ['bin_origin_ms','qualified_events','SCL_qualified_events',
            'sustained_SCL_contact_names','contact_peak_to_threshold','metrics']}
            for r in observed['records']]
        witnesses.append(dict(index=index,orbit=str(orbit),source=str(source),
            profile_fingerprint=fingerprint(orbit),J_EE_core=rows[index]['J_EE_core'],
            T_ms=rows[index]['T_ms'],status=raw['status'],classification=accepted[-1],
            records=records))
    write(DEST/'Bleading_increasing_J_readout.json',dict(source=str(source),
        segment_source=str(DATA/'Bleading_stable_side/checked_segment.json'),
        checked_points=80,J_range=[rows[0]['J_EE_core'],rows[-1]['J_EE_core']],
        witnesses=witnesses,observer_source=str(OLD/'observer_firing.json'),
        statistical_unit='One exact periodic orbit. Four bin origins and eight repeated cycles are not independent realizations.',
        scope='All 80 profiles pass physical checks; stability verified at three specific samples only. Individual recruitment differs from qualified-event participation.'))
    return rows,witnesses


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    if not PUBLIC.exists():PUBLIC.symlink_to(DEST,target_is_directory=True)
    assert PUBLIC.resolve()==DEST.resolve()
    rows,witnesses=new_arm()
    old_primary=plot.primary;old_evidence=plot.evidence
    def primary(fs):
        selected=old_primary(fs)
        # A separate path from the same seed. Never join it to the far end
        # of the older returning arm or globally sort the two paths by J.
        selected['BleadingUp']=rows
        return selected
    def evidence(sites):
        result=old_evidence(sites)
        for w in witnesses:
            result[str(Path(w['orbit']).resolve())]=dict(status=w['status'],source=w['source'],orbit=w['orbit'])
        from check_rate_Bleading_return_witness import verified_return_witness
        w=verified_return_witness(56)
        assert w is not None
        q=w['evidence'];result[str(Path(q['orbit']).resolve())]=dict(status=q['status'],source=w['source'],orbit=q['orbit'])
        return result
    plot.primary=primary;plot.evidence=evidence;plot.OUTPUT=PUBLIC
    plot.FAMILY['BleadingUp']='#a05823'
    plot.base.NAMES['BleadingUp']='B-leading: increasing J'
    original_argv=sys.argv;sys.argv=[__file__]
    try:plot.main()
    finally:sys.argv=original_argv
    metadata=read(PUBLIC/'figure_metadata.json')
    metadata.update(summary_producer=str(Path(__file__).resolve()),snapshot_date='2026-09-23',
        increasing_J_Bleading_source=str(DEST/'Bleading_increasing_J_readout.json'),
        increasing_J_Bleading_points=80,
        increasing_J_Bleading_stability='Three individually checked stable cycles; connecting line is unclassified',
        main_figure_scope='Primary bifurcation branches, all 400 spatial cells and 935 populations, with five same-equation representative cases. Dense secondary folds remain in companion figures.',
        global_branch_completeness=False)
    write(PUBLIC/'figure_metadata.json',metadata)
    fold=read(DATA/'current_fold_inventory.json')
    pd={name:read(plot.PERIODIC_OUT/(name+'_validation.json')) for name in
        ['PD_double_low','PD_double_upper','PD_A_return','PD_H2_after_LPC13']}
    summary=dict(model=dict(spatial_cells=400,populations=935,local_states=8415,physical_delays=True),
        parameter='J_EE,core scales both within-core E-to-E weights relative to the frozen reference',
        fold_inventory=dict(located=len(fold['rows']),locally_validated=fold['complete_local_validation_count'],
            pending=[r['label'] for r in fold['rows'] if not r['complete_local_validation']],
            source=str(DATA/'current_fold_inventory.json')),
        period_doubling={name:{k:q.get(k) for k in ['J_EE_core','criticality','full_acceptance','parent_stability']}
            for name,q in pd.items()},
        PD3_parent_spectra=read(DATA/'PD3_parent_spectra/current_evidence.json'),
        PD4_child_spectra=read(DATA/'H2_local_PD/current_child_spectrum_evidence.json'),
        PD2_physical_children_source=str(DATA/'physical_children/PD_double_upper/result.json'),
        same_parameter_stable_coexistence_source=str(DATA/'sameJ_small_burst_stability_readout.json'),
        increasing_J_Bleading_readout=str(DEST/'Bleading_increasing_J_readout.json'),
        pending_global_connections=True,stable_irregular_burst_attractor_established=False,
        all_detected_bifurcations_validated=False,human_visual_acceptance='PENDING')
    write(DEST/'summary_evidence.json',summary)
    readme=PUBLIC/'figures/README.md'
    text=readme.read_text().replace('### spatial_rate_focused_composite.png\n',
        '### spatial_rate_focused_composite.png\n2026-09-23 总图新增 B 领先周期分支向较大 J 延伸的 80 个物理解，三个独立抽查位置的稳定性已核验，支路单独绘制。')
    readme.write_text(text)
    print('SUMMARY',summary['fold_inventory'],flush=True)
    for w in witnesses:
        print('STABLE B ARM',w['J_EE_core'],[r['sustained_SCL_contact_names'] for r in w['records']],
            [r['qualified_events'] for r in w['records']],flush=True)


if __name__=='__main__':main()
