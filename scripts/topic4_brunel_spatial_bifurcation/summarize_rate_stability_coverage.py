"""Snapshot persisted numerical evidence without treating jobs as certificates."""
from plot_rate_periodic_completion import *
from collections import Counter
from datetime import datetime, timezone


def main():
    folder=PERIODIC_OUT/'stability_coverage';plan=read(folder/'plan.json')['rows']
    fs=families();rows=[]
    from audit_rate_survey_filter_states import fingerprint
    physical_file=folder/'constituent_filter_audit.json'
    physical={q['index']:q for q in read(physical_file)['rows']} if physical_file.exists() else {}
    for q in plan:
        path=folder/(Path(q['orbit']).stem+'.json')
        result=read(path) if path.exists() else dict(status='PENDING')
        result=dict(result,computed_multiplier_classification=result['status'])
        check=physical.get(len(rows));actual=Path(result.get('analyzed_orbit',q['orbit']))
        if result['status'] in ['UNSTABLE','NUMERICALLY_STABLE']:
            if not check or check['profile_fingerprint']!=fingerprint(actual):
                result['status']='PHYSICAL_PROFILE_CHECK_PENDING'
            elif not check['filter_state_check']['positive']:
                result['status']='PHYSICAL_PROFILE_REFINEMENT_REQUIRED'
            result['constituent_filter_audit']=str(physical_file)
        rows.append(dict(**q,result=result))
    families_out={};changes=[];restart_joins=[]
    for name,rr in fs.items():
        subset=[]
        for q in rows:
            for member in q['memberships']:
                if member['family']==name:
                    subset.append(dict(index=member['index'],status=q['result']['status'],
                        J_EE_core=q['J_EE_core'],T_ms=q['T_ms'],orbit=q['orbit']))
        subset.sort(key=lambda q:q['index'])
        counts=Counter(q['status'] for q in subset)
        classified=[q for q in subset if q['status'] in ['NUMERICALLY_STABLE','UNSTABLE']]
        for a,b in zip(classified[:-1],classified[1:]):
            if {a['status'],b['status']}=={'NUMERICALLY_STABLE','UNSTABLE'}:
                if any(a['index']<i<=b['index'] for i in continuation_breaks(rr)):
                    restart_joins.append(dict(family=name,ends=[a,b],
                        interpretation='Opposite classifications across an overlapping coarse-to-refined restart. This is not a consecutive continuation interval or an additional bifurcation bracket.'))
                    continue
                changes.append(dict(family=name,ends=[a,b],
                    interpretation='Opposite sampled stability along this traced family; intermediate sites may be unresolved. Known critical points may lie inside; crossing count, location and type require follow-up.'))
        families_out[name]=dict(continued_orbits=len(rr),planned_sites=len(subset),
            classification_counts=dict(counts),sites=subset)
    workers=[]
    for f in sorted(folder.glob('worker*.json')):
        q=read(f);q['file']=str(f)
        try:
            os.kill(q['pid'],0);q['process_alive']=True
        except (OSError,KeyError):q['process_alive']=False
        workers.append(q)
    numerical_counts=Counter(q['result']['status'] for q in rows)
    snapshot=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        planned_sites=len(rows),classification_counts=dict(numerical_counts),
        computed_multiplier_counts=dict(Counter(q['result']['computed_multiplier_classification'] for q in rows)),
        physical_profile_audit=str(physical_file),
        continued_orbits=sum(len(rr) for rr in fs.values()),families=families_out,
        stability_change_brackets=changes,overlapping_restart_comparisons=restart_joins,workers=workers,
        critical_orbits=[dict(label=q['label'],J_EE_core=q['J_EE_core'],N=q['N']) for q in critical()],
        complete=False,
        counting_rules=['UNSTABLE requires a residual-checked nontrivial multiplier clearly outside the unit circle.',
            'NUMERICALLY_STABLE is sampled numerical evidence, not a rigorous complete-spectrum theorem.',
            'Outside multiplier counts are lower bounds, never exact unstable dimensions.',
            'Adjacent sample sites are not certified intervals; no unobserved crossing is excluded.',
            'A computation failure or unresolved near-unit mode is not a bifurcation.'])
    write(folder/'summary.json',snapshot)
    audit=RATE_OUT/'model_audit_20260918/audit_status.json'
    q=read(audit)
    q['temporal_discretization']='Four secondary burst folds refined from N=1024 to 2048; J shifts 0.85e-8 to 1.52e-8. LPC24 additionally refined to N=4096: J shift 2.27e-12, continuous full-group defect 3.44e-5 Hz, generalized fold-mode defect decreases by about fourfold with dt halving.'
    q['secondary_fold_validation']=str(PERIODIC_OUT/'secondary_folds_mesh_validation.json')
    q['fold_mode_checks']=[read(f) for f in sorted(PERIODIC_OUT.glob('LPC_double_secondary*_monodromy_check_N*_dt*.json'))]
    q['fold_mode_check_scope']='Parameter-location convergence and full variational-mode convergence are distinct. Review step and orbit-mesh refinements together; no blanket mode-convergence claim.'
    if (PERIODIC_OUT/'LPC_double_secondary4_continuous_defect_N4096.json').exists():q['LPC24_finer_continuous_defect']=read(PERIODIC_OUT/'LPC_double_secondary4_continuous_defect_N4096.json')
    if (PERIODIC_OUT/'TR_A_B_validation.json').exists():q['first_torus_criticality']=dict(source=str(PERIODIC_OUT/'TR_A_B_validation.json'),status=read(PERIODIC_OUT/'TR_A_B_validation.json')['status'])
    if (PERIODIC_OUT/'TR_A_return_nonlinear_validation.json').exists():q['second_torus_criticality']=dict(source=str(PERIODIC_OUT/'TR_A_return_nonlinear_validation.json'),status=read(PERIODIC_OUT/'TR_A_return_nonlinear_validation.json')['status'])
    q['periodic_stability_coverage']=dict(source=str(folder/'summary.json'),counts=dict(numerical_counts),planned_sites=len(rows),complete=False)
    q['remaining']=[x for x in q['remaining'] if x!='Refine/verify four secondary periodic folds']
    if (PERIODIC_OUT/'TR_A_B_validation.json').exists():
        q['remaining']=[('Resolve nonlinear attractors beyond TR1 and the newly detected return-branch crossing' if x=='Resolve torus criticality/nonlinear attractor' else x) for x in q['remaining']]
    foldcheck=PERIODIC_OUT/'secondary_fold_mode_validation.json'
    if foldcheck.exists():
        checks=read(foldcheck)['rows']
        resolved=len(checks)==4 and all(all(3<v<5 for v in row['successive_defect_reduction']) and
            min(c['generalized_plus_one_relative_defect'] for c in row['monodromy_checks'])<.01 for row in checks)
        if resolved:q['remaining']=[x for x in q['remaining'] if x!='Finish time-step and orbit-mesh checks of secondary-fold critical directions']
        q['secondary_fold_direction_checks']=dict(source=str(foldcheck),numerical_refinement_supported=resolved)
    rootcounts=PERIODIC_OUT/'stationary_root_counts/summary.json'
    if rootcounts.exists():q['full_stationary_root_count_coverage']=read(rootcounts)
    hh=additional_hopfs()
    q['additional_equilibrium_hopfs']=[dict(label=h['label'].split('_')[0],J_EE_core=h['J_EE_core'],frequency_hz=h['frequency_hz'],criticality=h['criticality'],validation=h['validation']) for h in hh]
    q['remaining']=[('Complete full-root tracking on the unstable equilibrium branches; multiple previously omitted Hopfs are now confirmed' if x=='Locate additional characteristic-root crossings on unstable stationary branches if needed for a complete inventory' else x) for x in q['remaining']]
    q['remaining']=list(dict.fromkeys(q['remaining']))
    q['followup_timestamp_utc']=snapshot['timestamp_utc']
    q['latest_stability_snapshot']=dict(classified_sites=sum(numerical_counts[k] for k in ['UNSTABLE','NUMERICALLY_STABLE']),
        continued_orbits=snapshot['continued_orbits'],stability_change_brackets=len(changes))
    fullapproach=PERIODIC_OUT/'TR2_full_state_saddle_approach.json'
    if fullapproach.exists():
        full=read(fullapproach)
        q['torus_full_state_approach']=dict(source=str(fullapproach),status=full['status'],
            log_distance_period_slopes_s=full['adjacent_log_distance_period_slopes_s'],scope=full['scope'])
        q['remaining']=[('Solve the connecting invariant manifolds and finite-amplitude torus stability; full nine-state/history proximity has now been checked.'
            if x=='Test the candidate saddle-cycle torus connection in full state/history space and its finite-amplitude stability.' else x) for x in q['remaining']]
    write(audit,q)
    print('STABILITY SNAPSHOT',dict(numerical_counts),'continued',snapshot['continued_orbits'],
        'opposite-stability brackets',len(changes),flush=True)


if __name__=='__main__':main()
