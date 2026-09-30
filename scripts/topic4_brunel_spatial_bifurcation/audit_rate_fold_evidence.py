"""Distinguish located periodic folds from independently checked fold modes."""
from plot_rate_periodic_completion import *
from collections import Counter


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=PERIODIC_OUT/'cycle_fold_evidence_inventory.json')
    parser.add_argument('--refresh-profiles',action='store_true',help='Reconstruct both filters on the current root instead of relying on the older survey')
    args=parser.parse_args()
    if args.refresh_profiles:
        from audit_rate_filter_states import filter_state_minima
        s=RateField()
    rows=[]
    profile_audit=PERIODIC_OUT/'rate_filter_state_positivity_audit.json'
    physical={str(Path(q['orbit']).resolve()):q for q in read(profile_audit)['rows']} if profile_audit.exists() else {}
    roots={q['label']:q for q in critical() if q['label'].startswith('LPC')}
    # The plotting catalogue hides unvalidated extensions. An evidence
    # inventory must retain their already located roots as pending work.
    for name in EXTENSION_FOLD_LABELS:
        versions=[read(f) for f in PERIODIC_OUT.glob(name+'_N*.json')]
        if versions:roots[name]=max(versions,key=lambda q:q['N'])
    for root in roots.values():
        name=root['label']
        if not name.startswith('LPC'):continue
        N=root['N'];validation=PERIODIC_OUT/(name+'_validation.json')
        q=read(validation) if validation.exists() else None
        # A validation belongs to a particular refined root and mesh. A
        # concurrently refined orbit must not inherit an older pass label.
        validated_root=q.get('mesh_checks',[{}])[-1] if q else {}
        validation_matches=bool(q and validated_root.get('N')==N and
            Path(validated_root.get('orbit','')).resolve()==Path(root['orbit']).resolve() and
            abs(q.get('J_EE_core',float('inf'))-root['J_EE_core'])<1e-10 and
            abs(q.get('T_ms',float('inf'))-root['T_ms'])<1e-7)
        checks=[read(f) for f in PERIODIC_OUT.glob(f'{name}_monodromy_check_N{N}_*.json')]
        checks=sorted(checks,key=lambda c:c['dt_ms'],reverse=True)
        matched=[c for c in checks if abs(c['J_EE_core']-root['J_EE_core'])<1e-10]
        mode_status='BVP_LOCATED_MODE_CHECK_PENDING'
        profile=q.get('continuous_defect',{}) if q else {}
        positive_profile=(profile.get('minimum_rate_Hz',-float('inf'))>=-1e-9 and
            Path(profile.get('orbit','')).resolve()==Path(root['orbit']).resolve())
        if q and q['status']=='VALIDATED_CYCLE_FOLD' and validation_matches and positive_profile:mode_status='INDEPENDENT_MODE_CHECKED'
        elif matched:
            mode_status='MODE_CHECK_PRESENT_REVIEW_PENDING'
            errors=np.array([c['generalized_plus_one_relative_defect'] for c in matched])
            # A stored incomplete point-specific validation may concern
            # orbit resolution or incompatible curvature coordinates. Raw
            # mode residuals must not silently override that verdict.
            # Convergent variational errors alone do not certify that the
            # base cycle is physically resolved. Require the explicit
            # profile, mesh, curvature and critical-mode validation above.
        versions=sorted([read(f) for f in PERIODIC_OUT.glob(name+'_N*.json')],key=lambda c:c['N'])
        full_profile=physical.get(str(Path(root['orbit']).resolve()))
        profile_source=str(profile_audit) if full_profile else None
        if validation_matches and profile.get('filter_state_check'):
            full_profile=profile['filter_state_check']
            profile_source=str(validation)
        if args.refresh_profiles:
            z=np.load(root['orbit'])
            assert len(z['r'])==N and abs(float(z['J'])-root['J_EE_core'])<1e-10
            full_profile=filter_state_minima(s,z['r'],float(z['T']))
            profile_source='Current root: independent CPU Fourier reconstruction at fourfold temporal sampling'
        physical_status=('PASS' if full_profile['positive'] else 'TEMPORAL_REFINEMENT_REQUIRED') if full_profile else 'NOT_CHECKED'
        rows.append(dict(label=CRITICAL_LABELS[name],internal_label=name,status=mode_status,
            constituent_filter_status=physical_status,
            constituent_filter_check=full_profile,profile_evidence_source=profile_source,
            complete_local_validation=mode_status=='INDEPENDENT_MODE_CHECKED' and physical_status=='PASS',
            J_EE_core=root['J_EE_core'],T_ms=root['T_ms'],N=N,
            geometry_source=root['orbit'],temporal_meshes=[v['N'] for v in versions],
            latest_mesh_J_difference=abs(versions[-1]['J_EE_core']-versions[-2]['J_EE_core']) if len(versions)>1 else None,
            independent_validation_source=str(validation) if q else None,
            independent_validation_status=q['status'] if q else None,
            accepted_validation_method=(q.get('validation_method','FULL_PERIOD_VARIATIONAL_PROPAGATION')
                if q and q['status']=='VALIDATED_CYCLE_FOLD' else None),
            full_period_propagation_status=q.get('full_period_propagation_status',q['status']) if q else None,
            segmented_variational_source=q.get('segmented_variational_source') if q else None,
            segmented_fold_errors=q.get('segmented_fold_errors') if q else None,
            independent_CPU_RHS_source=q.get('independent_CPU_RHS_source') if q else None,
            independent_validation_matches_current_root=validation_matches,
            positive_base_profile_checked=positive_profile,
            matched_mode_check_count=len(matched),
            finest_mode_relative_defect=matched[-1]['generalized_plus_one_relative_defect'] if matched else None))
        print('FOLD EVIDENCE SITE',CRITICAL_LABELS[name],mode_status,physical_status,flush=True)
    result=dict(status='EVIDENCE_INVENTORY_INCOMPLETE',counts=dict(Counter(q['status'] for q in rows)),rows=rows,
        physical_profile_counts=dict(Counter(q['constituent_filter_status'] for q in rows)),
        complete_local_validation_count=sum(q['complete_local_validation'] for q in rows),
        complete_local_validation_methods=dict(Counter(q['accepted_validation_method']
            for q in rows if q['complete_local_validation'])),
        inventory_criteria=dict(direct_check_finest_relative_defect=.001,
            direct_check_minimum_halving_reduction=3,direct_check_minimum_steps=3,
            note='Existing point-specific validated modes are retained. Legacy checks previously accepted at 0.01 are listed for finer follow-up when they do not meet this tighter direct-mode threshold; their original evidence is not erased.'),
        scope='LPC markers with a located BVP null tangent are not all independently checked to the same standard. This inventory records presence and acceptance of full-state mode evidence; it neither rejects an untested fold nor certifies all intervening crossings.',
        all_modes_independently_checked=all(q['status']=='INDEPENDENT_MODE_CHECKED' for q in rows))
    result['profiles_refreshed_from_current_roots']=args.refresh_profiles
    write(args.output,result)
    print('FOLD EVIDENCE',result['counts'],flush=True)


if __name__=='__main__':main()
