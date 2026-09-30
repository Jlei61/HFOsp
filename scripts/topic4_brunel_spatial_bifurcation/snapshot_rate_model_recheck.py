"""Refresh the evidence index from persisted calculations, without a prose report."""
from plot_rate_periodic_completion import *
from datetime import datetime,timezone
import hashlib


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reviewed-figure',action='append',default=[],
                   help='Names actually inspected as PNG and rendered PDF in this turn')
    a=p.parse_args();now=datetime.now(timezone.utc).isoformat()
    status=read(PERIODIC_OUT/'analysis_status.json')
    coverage=read(PERIODIC_OUT/'stability_coverage/summary.json')
    from audit_rate_fold_evidence import main as audit_fold_evidence
    audit_fold_evidence()
    fold_evidence=read(PERIODIC_OUT/'cycle_fold_evidence_inventory.json')
    points=sum(q['count'] for q in status['periodic_families'].values())
    counts={k:0 for k in ['Hopf','equilibrium_fold','cycle_fold','torus','period_doubling']}
    for q in status['critical_points']:
        key=next(k for prefix,k in [('LPC','cycle_fold'),('LP','equilibrium_fold'),
                 ('H','Hopf'),('TR','torus'),('PD','period_doubling')] if q['label'].startswith(prefix))
        counts[key]+=1
    extensions=[];turns=[]
    for label in ['arcAreturnStrong','arcAglobalConnection','arcSingleUpperConnect','arcBtoBurst','arcBtoBurstFurther','arcAconnectionFurther','arcBconnectionFurther','arcAconnectionNext','arcBconnectionNext','arcAconnectionStage3','arcBconnectionStage3','arcAconnectionStage4','arcAconnectionStage5']:
        f=PERIODIC_OUT/(label+'_accuracy.json')
        if not f.exists():continue
        q=read(f)
        extensions.append(dict(label=label,status=q['status'],points=q['continued_points'],
            checked_profiles=len(q['checks']),maximum_tested_continuous_defect_Hz=
            max(v['maximum_group_defect_Hz'] for v in q['checks']),source=str(f)))
        for n,v in enumerate(q['turns']):
            internal=({54:'LPC_A_return_exchange',76:'LPC_A_return_recruitment',
                       131:'LPC_A_large_return1',145:'LPC_A_large_return2',152:'LPC_A_large_return3'}.get(v['index']) if label=='arcAreturnStrong' else
                      f'LPC_A_global_turn{n+1}' if label=='arcAglobalConnection' else
                      f'LPC_single_upper{n+1}' if label=='arcSingleUpperConnect' else
                      f'LPC_B_burst_turn{n+1}' if label=='arcBtoBurst' else
                      f'LPC_B_further_turn{n+1}' if label=='arcBtoBurstFurther' else None)
            if label=='arcAconnectionFurther':internal=f'LPC_A_connection{n+1}'
            if label=='arcBconnectionFurther':internal=f'LPC_B_connection{n+1}'
            if label=='arcAconnectionNext':internal=f'LPC_A_next{n+1}'
            if label=='arcBconnectionNext':internal=f'LPC_B_next{n+1}'
            if label=='arcAconnectionStage3':internal=f'LPC_A_stage3_turn{n+1}'
            if label=='arcBconnectionStage3':internal=f'LPC_B_stage3_turn{n+1}'
            if label=='arcAconnectionStage4':internal=f'LPC_A_stage4_turn{n+1}'
            if label=='arcAconnectionStage5':internal=f'LPC_A_stage5_turn{n+1}'
            validation=PERIODIC_OUT/(str(internal)+'_validation.json')
            if validation.exists() and read(validation).get('status')=='VALIDATED_CYCLE_FOLD':continue
            turns.append(dict(segment=label,segment_status=q['status'],
                              parent_geometry_accepted=q['status']=='SAMPLED_PASS',**v))
    newfolds=[]
    for name,label in CRITICAL_LABELS.items():
        if not label.startswith('LPC') or int(label[3:])<28:continue
        f=PERIODIC_OUT/f'{name}_validation.json'
        if not f.exists():continue
        q=read(f)
        if q['status']!='VALIDATED_CYCLE_FOLD':continue
        newfolds.append(dict(label=label,J_EE_core=q['J_EE_core'],T_ms=q['T_ms'],source=str(f)))
    pdroots=[]
    for f in sorted(PERIODIC_OUT.glob('PD_double_upper_N*.json')):
        q=read(f);pdroots.append({k:q[k] for k in ['N','J_EE_core','T_ms','antiperiodic_relative_residual']})
    audit=dict(timestamp_utc=now,status='MODEL_INTERNALLY_CHECKED_BIFURCATION_INCOMPLETE',
        model=dict(spatial_cells=400,population_groups=935,local_states=8415,physical_delays=True,
                   Z='fixed 1',M='dynamic, tau 1000 ms',equations_changed=False),
        periodic_points=points,critical_marker_count=sum(counts.values()),counts=counts,
        new_validated_folds=newfolds,new_extension_coverage=extensions,unresolved_new_turns=turns,
        upper_PD=dict(roots=pdroots,validated=((PERIODIC_OUT/'PD_double_upper_validation.json').exists()
                     and read(PERIODIC_OUT/'PD_double_upper_validation.json').get('status')=='VALIDATED_PD'),
                      scope='A null mode alone is not promotion; mesh and independent monodromy required.'),
        same_J_distinct_cycles=dict(J_EE_core=1.029396761345014,
            unstable_source=str(PERIODIC_OUT/'H1_large_orbit_instability.json'),
            stable_source=str(PERIODIC_OUT/'strong_sameJ_H1_stability.json'),
            conclusion='H1 large cycle is unstable; a distinct numerically stable A-leading cycle exists at the same J. No identity or global connecting branch is established.'),
        composite_resolution_source=str(PERIODIC_OUT/'composite_case_resolution.json'),
        stability_survey_counts=coverage['classification_counts'],stability_survey_sites=coverage['planned_sites'],
        full_state_torus_check=str(PERIODIC_OUT/'TR2_full_state_saddle_approach.json'),
        scientific_model_status='Frozen mean-response closure with local spatial correspondence. Native interictal irregular timing and whole-range dynamic equivalence remain unvalidated.',
        interval_completeness=False,human_visual_acceptance='PENDING')
    audit['stability_computed_multiplier_counts']=coverage.get('computed_multiplier_counts',coverage['classification_counts'])
    physical_survey=PERIODIC_OUT/'stability_coverage/constituent_filter_audit.json'
    if physical_survey.exists():
        data=read(physical_survey)
        audit['stability_survey_physical_profiles']=dict(source=str(physical_survey),
            checked_profiles=len(data['rows']),negative_filter_profiles=data['negative_filter_profiles'],
            scope=data['scope'])
    audit['cycle_fold_evidence']=dict(source=str(PERIODIC_OUT/'cycle_fold_evidence_inventory.json'),
        counts=fold_evidence['counts'],all_modes_independently_checked=fold_evidence['all_modes_independently_checked'],
        physical_profile_counts=fold_evidence['physical_profile_counts'],
        complete_local_validation_count=fold_evidence['complete_local_validation_count'])
    mesh_waveforms=PERIODIC_OUT/'cycle_fold_mesh_waveform_audit.json'
    if mesh_waveforms.exists():
        q=read(mesh_waveforms)
        audit['cycle_fold_mesh_waveforms']=dict(source=str(mesh_waveforms),checked_roots=len(q['rows']),
            review_required=[r for r in q['rows'] if r['status']!='MESH_WAVEFORM_AGREEMENT'],
            maximum_neuron_weighted_relative_change=max(r['neuron_weighted_waveform_relative_change'] for r in q['rows']),
            scope=q['scope'])
    for name,key in [('periodic_observer_exclusion_audit','periodic_contact_observer_review'),
                     ('arcAconnectionStage4_encounter_screen_with_higher_hopfs','extended_branch_connection_screen')]:
        source=PERIODIC_OUT/(name+'.json')
        if source.exists():
            data=read(source)
            audit[key]=dict(source=str(source),scope=data['scope'])
            if key=='periodic_contact_observer_review':audit[key]['rows']=data['rows']
            else:
                audit[key].update(status=data['status'],compared_family_pairs=len(data['rows']),
                    same_parameter_correction_candidates=sum(len(row['candidate_same_parameter_corrections']) for row in data['rows']))
    stage4roots=[read(f) for f in PERIODIC_OUT.glob('LPC_A_stage4_turn1_N*.json')]
    if stage4roots:
        validation=PERIODIC_OUT/'LPC_A_stage4_turn1_validation.json'
        audit['H1_stage4_new_fold']=dict(roots=stage4roots,
            status=read(validation)['status'] if validation.exists() else 'MESH_AND_INDEPENDENT_MODE_CHECKS_PENDING',
            scope='A parameter turn and BVP tangent are local candidates until waveform, mesh and full-state +1 checks pass.')
    secondroots=[read(f) for f in PERIODIC_OUT.glob('LPC_A_stage4_turn2_N*.json')]
    if secondroots:
        validation=PERIODIC_OUT/'LPC_A_stage4_turn2_validation.json'
        audit['H1_stage4_second_fold']=dict(roots=secondroots,
            status=read(validation)['status'] if validation.exists() else 'MESH_AND_INDEPENDENT_MODE_CHECKS_PENDING',
            scope='Second parameter turn on the same continued H1 family. No promotion from geometry alone.')
    thirdroots=[read(f) for f in PERIODIC_OUT.glob('LPC_A_stage4_turn3_N*.json')]
    if thirdroots:
        validation=PERIODIC_OUT/'LPC_A_stage4_turn3_validation.json'
        audit['H1_stage4_third_fold']=dict(roots=thirdroots,
            status=read(validation)['status'] if validation.exists() else 'MESH_AND_INDEPENDENT_MODE_CHECKS_PENDING',
            scope='Third parameter turn on the H1 continuation; Core B has elevated background activity. Geometry does not establish stability.')
    for name,key in [('arcAconnectionStage4_independent_CPU_checks','H1_stage4_independent_equation_checks'),
                     ('arcAconnectionStage4_third_turn_CPU_checks','H1_stage4_third_turn_equation_checks'),
                     ('filter_algebraic_precision_check','filter_solver_precision_check'),
                     ('cached_host_fold_solver_check','cached_host_fold_solver_check'),
                     ('warm_tangent_fold_solver_check','warm_tangent_fold_solver_check'),
                     ('monodromy_gain_blocks_regression','monodromy_gain_blocks_regression')]:
        source=PERIODIC_OUT/(name+'.json')
        if source.exists():audit[key]=dict(evidence_source=str(source),**read(source))
    seeds=PERIODIC_OUT/'H1_stage4_seed_refinement.json'
    if seeds.exists():
        q=read(seeds);audit['H1_fine_continuation_seeds']=dict(source=str(seeds),status=q['status'],
            rows=[{k:v[k] for k in ['index','refined_orbit','relative_waveform_change','relative_period_change']} for v in q['rows']],
            scope='Seed refinement only; does not accept the preceding segment or imply a global connection.')
    secondary_roots=list(PERIODIC_OUT.glob('PD_upper_child_next_amplitude_N*.json'))
    if secondary_roots:
        secondary=max(secondary_roots,key=lambda f:read(f)['N'])
        audit['PD2_child_secondary_root_candidate']=dict(source=str(secondary),**read(secondary),
            promotion_status='MESH_AND_INDEPENDENT_MODE_CHECKS_REQUIRED')
        followup=PERIODIC_OUT/'PD2_secondary_validation.json'
        if followup.exists():audit['PD2_child_secondary_root_candidate']['fine_mesh_followup']=read(followup)
        cpu=PERIODIC_OUT/'PD2_secondary_candidate_CPU_check_N2048.json'
        if cpu.exists():audit['PD2_child_secondary_root_candidate']['independent_coarse_profile_check']=read(cpu)
    probe=PERIODIC_OUT/'H2_stage3_resolution_probe.json'
    if probe.exists():
        q=read(probe)
        audit['withheld_H2_segment_resolution_probe']=dict(source=str(probe),status=q['status'],
            checked_endpoints=[dict(index=v['index'],status=v['resolution']['status'],
                N=v['resolution'].get('N'),relative_waveform_change=v['relative_waveform_change'],
                maximum_group_defect_Hz=v['resolution'].get('maximum_group_defect_Hz')) for v in q.get('rows',[])],
            segment_geometry_accepted=False,
            scope='Endpoint repairs do not validate the intervening coarse continuation or its apparent turns.')
    primitive=PERIODIC_OUT/'cycle_primitive_period_audit.json'
    if primitive.exists():
        q=read(primitive)
        audit['cycle_primitive_periods']=dict(source=str(primitive),
            checked_roots=len(q['rows']),
            primitive_period_supported=sum(v['status']=='PRIMITIVE_PERIOD_SUPPORTED' for v in q['rows']),
            minimum_first_harmonic_RMS_fraction=min(v['checks'][-1]['first_harmonic_RMS_fraction'] for v in q['rows']),
            scope=q['scope'])
    positivity=PERIODIC_OUT/'periodic_profile_positivity_audit.json'
    if positivity.exists():
        q=read(positivity);current=[]
        for row in q['rows']:
            evidence=read(Path(row['source']))
            current.append(dict(index=row['index'],previous_status=row['previous_status'],
                status=evidence['status'],repair_source=evidence.get('positive_profile_followup')))
        audit['periodic_profile_positivity_review']=dict(source=str(positivity),
            initially_withdrawn=len(q['rows']),current=current,scope=q['scope'])
    delaycheck=RATE_OUT/'model_audit_20260918/nonconstant_delay_history_checks.json'
    if delaycheck.exists():
        data=read(delaycheck)
        audit['nonconstant_delay_implementation_check']=dict(source=str(delaycheck),
            status=data['status'],checked_parameter_step_pairs=len(data['rows']),
            maximum_arrival_relative_error=max(v['arrivals_relative_error'] for v in data['rows']),
            maximum_step_increment_relative_error=max(v['increment_relative_error'] for v in data['rows']),
            scope=data['scope'])
    spatial_modes=PERIODIC_OUT/'cycle_fold_spatial_contact_modes.json'
    if spatial_modes.exists():
        data=read(spatial_modes)
        audit['critical_spatial_contact_modes']=dict(source=str(spatial_modes),
            checked_labels=[v['label'] for v in data['rows']],scope=data['scope'])
    joined=PERIODIC_OUT/'joined_cycle_fold_spatial_contact_modes.json'
    if joined.exists():
        data=read(joined)
        audit['joined_critical_spatial_contact_modes']=dict(source=str(joined),
            checked_labels=[v['label'] for v in data['rows']],scope=data['scope'])
    for name,key in [('dense_H2_fold_spatial_contact_modes','dense_H2_critical_spatial_modes'),
                     ('dense_H2_fold_parent_waveforms','dense_H2_parent_waveforms'),
                     ('H1_extended_fold_spatial_modes','H1_extended_critical_spatial_modes'),
                     ('H1_fold_pair_spatial_modes','H1_fold_pair_critical_spatial_modes'),
                     ('H1_periodic_fold_pair_detail','H1_periodic_fold_pair_detail'),
                     ('H1_three_new_fold_spatial_modes','H1_three_new_fold_spatial_modes'),
                     ('H1_three_new_fold_rate_fields','H1_three_new_fold_actual_rate_fields'),
                     ('branch_encounter_screen','traced_branch_encounter_screen'),
                     ('PD_child_branch_encounter_screen','PD_child_encounter_screen'),
                     ('arcAconnectionStage4_accepted_encounter_screen_with_higher_hopfs','checked_H1_stage4_connection_screen'),
                     ('global_endpoints_H1_stage4_waveform_audit','checked_H1_stage4_endpoint_screen'),
                     ('stability_bracket_root_associations','stability_bracket_root_associations')]:
        source=PERIODIC_OUT/(name+'.json')
        if source.exists():
            data=read(source)
            audit[key]=dict(source=str(source),scope=data['scope'],rows=len(data['rows']))
            if 'screen' in name:
                audit[key]['same_parameter_correction_candidates']=sum(len(row['candidate_same_parameter_corrections']) for row in data['rows'])
            if data.get('accepted_source_segment'):
                audit[key]['source_points']=len(data['accepted_source_segment']['orbit_snapshot'])
                audit[key]['metadata_window_pairs']=sum(row['metadata_window_pairs'] for row in data['rows'])
                audit[key]['fully_compared_pairs']=sum(row['fully_compared_pairs'] for row in data['rows'])
    asymmetric_probe=PERIODIC_OUT/'H1_highB_equilibrium_trust_region_probe.json'
    if asymmetric_probe.exists():
        data=read(asymmetric_probe)
        audit['H1_high_background_equilibrium_search']=dict(evidence_source=str(asymmetric_probe),**data)
    neighbors=PERIODIC_OUT/'H1_stage4_fold2_neighbor_stability.json'
    if neighbors.exists():
        audit['H1_stage4_second_fold_neighbor_stability']=dict(evidence_source=str(neighbors),**read(neighbors))
    else:
        running=PERIODIC_OUT/'H1_stage4_fold2_neighbors_worker.json'
        if running.exists():
            q=read(running)
            audit['H1_stage4_second_fold_neighbor_stability']=dict(evidence_source=str(running),
                status=q['status'],rows=q['rows'],
                scope='Incomplete two-point batch; report only persisted paired-step results. No full-spectrum or intervening-crossing claim.')
    third_stability=PERIODIC_OUT/'H1_stage4_fold3_neighbor_stability.json'
    if third_stability.exists():
        audit['H1_stage4_third_fold_stability']=dict(evidence_source=str(third_stability),**read(third_stability))
    first_stability=PERIODIC_OUT/'H1_stage4_fold1_neighbor_stability.json'
    if first_stability.exists():
        audit['H1_stage4_first_fold_stability']=dict(evidence_source=str(first_stability),**read(first_stability))
    pd_review=PERIODIC_OUT/'PD_resolution_review.json'
    if pd_review.exists():audit['PD_resolution_review']=read(pd_review)
    filter_review=PERIODIC_OUT/'rate_filter_state_positivity_audit.json'
    if filter_review.exists():
        data=read(filter_review)
        audit['constituent_rate_filter_review']=dict(source=str(filter_review),status=data['status'],
            checked_profiles=len(data['rows']),profiles_requiring_refinement=[q for q in data['rows'] if not q['positive']],
            scope=data['scope'],does_not_revoke_earlier_eigenmode_evidence=True,
            full_physical_state_waveform_acceptance='REVIEW_PENDING')
        refined={}
        for path in sorted(PERIODIC_OUT.glob('filter_state_repair_*.json'),key=lambda f:f.stat().st_mtime):
            for row in read(path).get('rows',[]):
                if row.get('resolution',{}).get('status')=='RESOLUTION_CHECKED':
                    refined[row['label']]=dict(source=str(path),**row)
        audit['constituent_rate_filter_review']['refined_profiles']=list(refined.values())
        audit['constituent_rate_filter_review']['not_yet_refined_labels']=[q['label'] for q in data['rows']
            if not q['positive'] and q['label'] not in refined]
        pdparent=next((q for q in data['rows'] if q['label']=='PD_double_upper'),None)
        if pdparent:
            audit['upper_PD']['eigenmode_validated']=audit['upper_PD']['validated']
            audit['upper_PD']['full_physical_profile_status']='PASS' if pdparent['positive'] else 'TEMPORAL_REFINEMENT_REQUIRED'
            audit['upper_PD']['validated']=audit['upper_PD']['validated'] and pdparent['positive']
            audit['upper_PD']['scope']='Critical-mode evidence and full physical parent-profile acceptance are separate; full validation is withheld while a constituent filter remains underresolved.'
        secondary=next((q for q in data['rows'] if q['label']=='PD2_child_secondary_candidate'),None)
        if secondary and 'PD2_child_secondary_root_candidate' in audit:
            candidate=audit['PD2_child_secondary_root_candidate']
            if Path(candidate['orbit']).resolve()==Path(secondary['orbit']).resolve():
                candidate['filter_state_check']=secondary
                candidate['promotion_status']='PHYSICAL_PROFILE_MESH_AND_INDEPENDENT_MODE_CHECKS_REQUIRED'
            else:candidate['previous_mesh_filter_state_check']=secondary
            followup=candidate.get('fine_mesh_followup',{})
            if followup.get('status')=='VALIDATED_SECONDARY_MINUS_ONE_CROSSING':
                candidate['promotion_status']='MINUS_ONE_CROSSING_VERIFIED_CHILD_CRITICALITY_PENDING'
    local=read(RATE_OUT/'model_audit_20260918/local_assay_recheck.json')
    audit['closure_limitations']=dict(calibration_J=.94,
        filter_constants_frozen_across_J=True,separate_variance_response_fit=False,
        local_assay_source=str(RATE_OUT/'model_audit_20260918/local_assay_recheck.json'),
        local_assay_is_independent_validation=local['independent_validation'],
        heldout_frequency_summaries=[r for r in local['summaries'] if r['frequencies_hz']==[6.]],
        scope='Local mean-response fit is not whole-network dynamical equivalence. Reusing its filters for variance responses is an approximation. Neither that approximation nor omitted finite-population fluctuations has been isolated as the cause of the observed network mismatch.')
    characteristic_check=RATE_OUT/'model_audit_20260918/characteristic_full_rhs_check.json'
    if characteristic_check.exists():
        audit['characteristic_full_rhs_check']=dict(source=str(characteristic_check),**read(characteristic_check))
    for label,name in [('H1_extended_endpoint','H1_global_endpoint_instability'),
                       ('H2_extended_endpoint','H2_further_endpoint_instability')]:
        endpoint=PERIODIC_OUT/(name+'.json')
        if endpoint.exists():
            q=read(endpoint)
            audit[label]=dict(source=str(endpoint),status=q['status'],
                J_EE_core=q['J_EE_core'],T_ms=q['T_ms'],
                full_spectrum_validated=q['full_spectrum_validated'])
    child=PERIODIC_OUT/'PDupperchild_branch_N4096.json'
    if child.exists():
        rows=read(child)
        audit['upper_PD']['nonlinear_child']=dict(source=str(child),
            converged_amplitudes_Hz=[v['amplitude_hz'] for v in rows],
            status='NONZERO_CHILD_EXISTS_CRITICALITY_AND_STABILITY_PENDING')
        classification=PERIODIC_OUT/'PD_upper_child_classification.json'
        if classification.exists():
            audit['upper_PD']['nonlinear_child'].update(
                classification_source=str(classification),status=read(classification)['status'])
    extension=PERIODIC_OUT/'PDupperextension_branch_N4096.json'
    if extension.exists():
        rows=read(extension)
        audit['upper_PD']['child_extension']=dict(source=str(extension),
            converged_amplitudes_Hz=[v['amplitude_hz'] for v in rows],
            scope='Converged child geometry only; local PD2 stability does not automatically extend to these larger amplitudes.')
        extra=PERIODIC_OUT/'PD2_child_extension_stability.json'
        if extra.exists():audit['upper_PD']['child_extension']['stability_followup']=dict(source=str(extra),**read(extra))
    h1pd=PERIODIC_OUT/'PD_A_return_validation.json'
    roots=[read(f) for f in PERIODIC_OUT.glob('PD_A_return_N*.json')]
    if roots:
        audit['H1_return_PD']=dict(roots=[{k:r[k] for k in ['N','J_EE_core','T_ms','antiperiodic_relative_residual']} for r in sorted(roots,key=lambda r:r['N'])],
            status=read(h1pd)['status'] if h1pd.exists() else 'VALIDATION_PENDING',
            validation_source=str(h1pd) if h1pd.exists() else None,
            child_criticality='NOT_COMPUTED',global_connection='NOT_ESTABLISHED')
        h1child=PERIODIC_OUT/'PDreturnchild_branch_N2048.json'
        if h1child.exists():
            children=read(h1child)
            audit['H1_return_PD']['child_branch']=dict(source=str(h1child),
                converged_amplitudes_Hz=[v['amplitude_hz'] for v in children],
                scope='Nonzero doubled BVP solutions; parent instability and branch side alone do not classify the new critical direction or child stability.')
        classification=PERIODIC_OUT/'PD_return_child_classification.json'
        if classification.exists():
            q=read(classification)
            audit['H1_return_PD'].update(child_criticality=q['status'],
                child_stability=q['child_stability'],child_classification_source=str(classification),
                new_direction_multiplier=q['child_mu'],inherited_unstable_multiplier=q['inherited_unstable_multiplier'])
            audit['H1_return_PD']['child_branch']['scope']=q['scope']
    audit['strong_fold_instability_checks']=[]
    for name in ['LPC_single_upper1','LPC_single_upper2']:
        source=PERIODIC_OUT/(name+'_instability.json')
        if source.exists():audit['strong_fold_instability_checks'].append(dict(source=str(source),**read(source)))
    native=read(RATE_OUT/'model_audit_20260918/matched_native_recheck.json')
    # The actual contact order interleaves the two shafts. Never slice the
    # first four values and call them SCL.
    audit['matched_native_key_cases']=[]
    for row in native['rows']:
        if row['J_EE_core'] not in [.942,1.3]:continue
        item=dict(J_EE_core=row['J_EE_core'],window_ms=row['window_ms'],
                  differences=row['metric_differences'])
        for kind in ['native','rate']:
            data=row[kind]
            item[kind]=dict(eligible_events=data['metrics']['N'],
                core_peak_interval_CV=[v['peak_interval_CV'] for v in data['dynamics']],
                SCL_participation_by_name={name:value for name,value in
                    zip(native['contact_names'],data['metrics']['participation']) if name.startswith('SCL')})
        audit['matched_native_key_cases'].append(item)
    audit['matched_native_scope']=native['statistical_unit']
    screen=PERIODIC_OUT/'H1_secondary_mode_screen.json'
    if screen.exists():
        q=read(screen)
        audit['H1_secondary_mode_screen']=dict(source=str(screen),status=q['status'],
            checked_sites=len(q['rows']),classified_new_crossings=int(h1pd.exists() and read(h1pd).get('status')=='VALIDATED_PD'),
            scope='Negative multipliers at isolated sites do not establish a -1 crossing. No new PD is promoted by this screen.')
    audit['unresolved_fold_diagnostics']=[]
    for name in ['LPC_single_upper1','LPC_single_upper2','LPC_single_upper3','LPC_single_upper4']:
        path=PERIODIC_OUT/(name+'_validation.json')
        if not path.exists():continue
        q=read(path)
        if q['status']=='VALIDATED_CYCLE_FOLD':continue
        row=dict(label=name,status=q['status'],source=str(path),J_EE_core=q['J_EE_core'],
            finest_full_period_mode_defect=q['full_state_fold_mode_checks'][-1]['generalized_plus_one_relative_defect'])
        check=PERIODIC_OUT/(name+'_independent_variational_BVP_e0.125.json')
        if check.exists():
            v=read(check)
            row['independent_CPU_BVP']=dict(source=str(check),
                finest_relative_residual=v['checks'][-1]['maximum_relative_residual'],
                negative_control_residual=v['negative_control']['maximum_relative_residual'])
        audit['unresolved_fold_diagnostics'].append(row)
    write(PERIODIC_OUT/'model_and_bifurcation_recheck.json',audit)
    remain=['Finish the frozen Floquet survey and resolve its uncertain/failed sites; sampled stability does not exclude intervening crossings.',
            ('Determine the side and stability of the validated PD2 child branch.' if audit['upper_PD']['validated'] else
             'Validate the upper double-cycle PD and determine the child side and stability.'),
            'Refine the remaining parameter-turn candidates; a plotted turn is not a validated fold.',
            'Continue H2 and resolve global connections of the Hopf-born and stable burst families.',
            'Torus-to-saddle-cycle proximity does not establish the connecting invariant manifolds or stable irregular bursting.',
            'Spatial correspondence at one working point does not establish native SNN temporal equivalence.']
    if not fold_evidence['all_modes_independently_checked']:
        remain.append('Complete independent full-state critical-mode checks for legacy BVP-located LPC markers; marker counts are not uniformly validated counts.')
    if audit['upper_PD'].get('nonlinear_child',{}).get('status') in ['SUPERCRITICAL_PD','SUBCRITICAL_PD']:
        remain[1]='Continue the classified PD2 child beyond its local departure; global connections remain unresolved.'
        if (PERIODIC_OUT/'PD2_child_extension_stability.json').exists():
            remain[1]='Verify the larger PD2 child stability loss and locate any subsequent -1 crossing; a negative outside multiplier alone does not classify it.'
    if audit.get('H1_return_PD',{}).get('status')=='VALIDATED_PD':
        if audit['H1_return_PD']['child_criticality']=='SUPERCRITICAL_PD':
            remain.insert(2,'Continue the classified supercritical PD3 child; its new direction is stable but inherited instability remains. No global connection is established.')
        else:
            remain.insert(2,'Validate and continue the PD3 doubled child on the already unstable H1-return parent. Resolve its criticality and global connection without inferring stable onset from a -1 crossing.')
    if audit.get('constituent_rate_filter_review',{}).get('profiles_requiring_refinement'):
        remain.insert(0,'Refine negative constituent fast-rate filter profiles; positive combined rates alone were insufficient. Existing mode evidence is retained separately from full physical-state waveform acceptance.')
    compact={k:audit[k] for k in ['timestamp_utc','status','periodic_points','critical_marker_count','counts','interval_completeness']}
    compact['source']=str(PERIODIC_OUT/'model_and_bifurcation_recheck.json')
    for filename in ['continuation_checkpoint_20260918.json','delivery_review_20260918.json']:
        q=read(PERIODIC_OUT/filename);q['timestamp_utc']=now;q['latest_model_recheck']=compact
        if filename.startswith('continuation'):
            q.update(periodic_geometry_points=points,stability_survey_counts=coverage['classification_counts'],next_work=remain)
            jobs=q.setdefault('ongoing',[])
            for name,relative in [('Fine fold-mode check','S1_fine_analytic_fold_worker.json'),
                ('Four candidate fold refinements','remaining_turn_refinements/worker_S3_S4_B1_B2.json'),
                ('Four connection-segment fold refinements','remaining_turn_refinements/worker_AC1_AC2_BC1_BC2.json'),
                ('Hopf A segment-join fold refinement','remaining_turn_refinements/worker_AN1.json'),
                ('Primary burst fold resolution','remaining_turn_refinements/worker_BL_AL_AH.json'),
                ('Alternating burst fold resolution','remaining_turn_refinements/worker_DL_DH_DS1_DS2_DS3_DS4.json'),
                ('Two new H2 fold candidates','remaining_turn_refinements/worker_BS1_BS2.json'),
                ('PD1 fine parent and critical mode','PD_double_low_filter_followup_worker.json'),
                ('Near-unit survey refinement','stability_coverage/near_unit_followup_worker.json'),
                ('Survey profile positivity repair','stability_coverage/profile_repair_worker_56.json'),
                ('Remaining survey profile repairs','stability_coverage/profile_repair_worker_58_62_64_63_60_59_53_61_77_81_103_104_105_111_114_118.json'),
                ('Small survey profile recovery','stability_coverage/profile_repair_worker_61_77_81.json'),
                ('Large survey profile recovery','stability_coverage/profile_repair_worker_103_104_105_111_113_114_115_117_118_119.json'),
                ('Stable survey physical-profile rechecks','stability_coverage/profile_repair_worker_41_42_44_45_68_85_86_87_89_91_98_99.json'),
                ('PD1 parent resolution repair','PD1_resolution_repair_worker.json'),
                ('Legacy fold and connection checks','remaining_turn_refinements/worker_LU_LL_BC1_BC2.json'),
                ('Legacy fold tight-coordinate recheck','legacy_fold_recheck_queue.json'),
                ('Legacy fold independent mode batch','legacy_fold_mode_audit/worker.json'),
                ('Strong-fold independent checks','strong_fold_checks_worker.json'),
                ('H1-return PD independent validation','H1_PD_validation_worker.json'),
                ('Hopf-family further continuations','global_connections_worker.json'),
                ('Hopf-family further continuations stage 2','global_connections_stage2_worker.json'),
                ('Hopf-family further continuations stage 3','global_connections_stage3_worker.json'),
                ('PD3 child independent instability','PD3_child60_independent_worker.json'),
                ('PD3 radial mode independent checks','PD3_radial_independent_worker.json'),
                ('PD3 constituent-filter and mode followup','PD3_filter_state_followup_worker.json'),
                ('PD2 extension stability','PD2_extension_stability_worker.json'),
                ('PD2 extended-child antiperiodic seed','PD2_child40_negative_mode_seed_worker.json'),
                ('PD2 child subsequent crossing','PD2_child_next_crossing_worker.json'),
                ('Secondary PD finer-mesh and full-state verification','PD2_secondary_refinement_worker.json'),
                ('H2 withheld-segment endpoint resolution','H2_stage3_resolution_probe.json'),
                ('H2 full-segment physical profile refinement','arcBconnectionStage3_physical_refinement.json'),
                ('PD and global-connection serial followup','pd_connection_followup_queue.json'),
                ('H1 fourth continuation segment','H1_stage4_worker.json'),
                ('H1 fifth continuation segment','arcAconnectionStage5_worker.json'),
                ('H1 fourth-segment fold verification','H1_stage4_fold_worker.json'),
                ('H1 fourth-segment second fold verification','H1_stage4_fold2_worker.json'),
                ('H1 fourth-segment third fold verification','H1_stage4_fold3_worker.json'),
                ('H1 fourth-segment second-fold neighbor Floquet checks','H1_stage4_fold2_neighbors_worker.json'),
                ('H1 fourth-segment first-fold Floquet check','H1_stage4_fold1_neighbors_worker.json'),
                ('H1 fourth-segment third-fold Floquet check','H1_stage4_fold3_neighbors_worker.json'),
                ('H1 fourth-segment independent CPU checks','arcAconnectionStage4_independent_CPU_checks.json'),
                ('H1 secondary crossing followup','H1_secondary_crossing_followup.json')]:
                source=PERIODIC_OUT/relative
                if not source.exists():continue
                data=read(source)
                jobs[:]=[v for v in jobs if v['name']!=name]
                jobs.append(dict(name=name,pid=data.get('pid'),source=str(source)))
            # Record the already-running extension by its exact Python
            # command; do not mistake a shell wrapper or reused PID for it.
            tracked_children={b'PDupperextension':('PD2 child extension a20 a40','/tmp/rate_PD2_extended_child.log'),
                              b'PDreturnchild':('PD3 doubled child','/tmp/rate_H1_PD_child.log')}
            for source in PERIODIC_OUT.glob('filter_state_repair_*.json'):
                data=read(source);name='Constituent filter follow-up: '+source.stem
                jobs[:]=[v for v in jobs if v['name']!=name]
                jobs.append(dict(name=name,pid=data.get('pid'),source=str(source)))
            jobs[:]=[v for v in jobs if v['name'] not in {item[0] for item in tracked_children.values()}]
            for proc in Path('/proc').glob('[0-9]*'):
                try:args=(proc/'cmdline').read_bytes().split(b'\0')
                except (FileNotFoundError,PermissionError,ProcessLookupError):continue
                if (args and b'python' in args[0] and
                    any(v.endswith(b'/rate_period_doubled_branch.py') for v in args)):
                    for tag,(name,log) in tracked_children.items():
                        if tag in args:jobs.append(dict(name=name,pid=int(proc.name),status='RUNNING',log=log))
            for item in q.get('ongoing',[]):
                source=Path(item['source']) if item.get('source') else None
                if source and source.exists():
                    data=read(source)
                    item['status']=data.get('stage',data.get('status','UNKNOWN'))
                pid=item.get('pid');item['alive']=bool(pid and Path(f'/proc/{pid}').exists())
                if item['name']=='PD2 child subsequent crossing':
                    roots=[PERIODIC_OUT/(name+'_N2048.json') for name in ['PD_upper_child_next','PD_upper_child_next_amplitude']]
                    if any(root.exists() for root in roots):item['status']='BVP_ROOT_REQUIRES_MESH_AND_MONODROMY_VALIDATION'
                    elif not item['alive']:item['status']='NO_LIVE_PROCESS_CHECK_LOG_AND_RESULTS'
                if not item['alive'] and item['name'].startswith('PD2 child Floquet'):
                    item['status']=audit['upper_PD'].get('nonlinear_child',{}).get('status',item.get('status'))
                if not item['alive'] and item['name'] in ['PD2 doubled child full-space host Krylov','PD2 child paired temporal mesh']:
                    branch=PERIODIC_OUT/('PDupperchild_branch_N4096.json' if item['name'].startswith('PD2 doubled') else 'PDupperchildMesh_branch_N2048.json')
                    rows=read(branch) if branch.exists() else []
                    expected={2.,5.,10.} if item['name'].startswith('PD2 doubled') else {2.}
                    passed=(expected.issubset({v['amplitude_hz'] for v in rows}) and all(
                        read(Path(v['orbit']).with_suffix('.json'))['status']=='CONVERGED' for v in rows))
                    item['status']='COMPLETED_CONVERGED_CHILDREN' if passed else 'NO_LIVE_PROCESS_CHECK_RESULTS'
        else:
            q.pop('all_confirmed_critical_marker_count',None)
            q.update(periodic_points=points,critical_marker_count=sum(counts.values()),inventory_complete=False,
                cycle_fold_evidence_counts=fold_evidence['counts'])
            reviewed=q.setdefault('current_agent_review',[])
            for name in a.reviewed_figure:
                reviewed[:]=[v for v in reviewed if v.get('name')!=name]
                reviewed.append(dict(name=name,review_timestamp_utc=now,
                    agent_png_and_same_state_pdf_visual_review='PASS',human_review='PENDING',
                    hashes={ext:hashlib.sha256((F/(name+'.'+ext)).read_bytes()).hexdigest() for ext in ['png','pdf']}))
        write(PERIODIC_OUT/filename,q)
    path=RATE_OUT/'model_audit_20260918/audit_status.json';q=read(path)
    q.update(status=audit['status'],followup_timestamp_utc=now,latest_model_recheck=compact,remaining=remain,
        latest_stability_snapshot=dict(classified_sites=sum(coverage['classification_counts'].get(k,0) for k in ['UNSTABLE','NUMERICALLY_STABLE']),
            continued_orbits=points,stability_change_brackets=len(coverage['stability_change_brackets'])))
    q['periodic_stability_coverage']['counts']=coverage['classification_counts']
    write(path,q)
    print('RECHECK',points,'orbits',counts,'unresolved new turns',len(turns),flush=True)


if __name__=='__main__':main()
