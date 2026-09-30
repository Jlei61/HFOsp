"""Validate a cycle fold by independently matched full-state variational BVP.

Whole-period propagation is retained as evidence, including failures. For a
strongly unstable orbit, use all-section variational matching plus an
independent CPU full-RHS derivative and the generalized periodic boundary
condition. This checks the same fold null mode without propagating numerical
contamination through an entire period. Adjacent stability remains separate.
"""
from rate_periodic import *


def segmented_checks(q):
    rows=sorted(q['checks'],key=lambda r:r['dt_ms'],reverse=True)
    e=np.array([r['maximum_fold_relative_defect'] for r in rows])
    p=np.array([r['maximum_phase_relative_defect'] for r in rows])
    passed=(q['status']=='DIAGNOSTIC_COMPLETE' and len(rows)>=3 and
        e[-1]<1e-4 and p[-1]<1e-4 and np.all(e[:-1][-2:]/e[1:][-2:]>3) and
        np.all(p[:-1][-2:]/p[1:][-2:]>3) and
        min(r['minimum_negative_control_defect'] for r in rows)>.01 and
        max(r['generalized_boundary_relative_error'] for r in rows)<1e-10 and
        max(r['phase_reconstruction_relative_error'] for r in rows)<1e-10 and
        all(len(r['matches'])==r['segments'] and r['segments']>=8 for r in rows))
    return bool(passed),e,p


def validate(label):
    flow=read(PERIODIC_OUT/'segmented_variational_flow_check.json')
    positive=read(PERIODIC_OUT/'LPC_A_global_turn1_validation.json')
    posseg=read(PERIODIC_OUT/'LPC_A_global_turn1_segmented_mode_checks.json')
    assert flow['status']=='PASS' and positive['status']=='VALIDATED_CYCLE_FOLD'
    assert segmented_checks(posseg)[0]
    original=PERIODIC_OUT/(label+'_validation.json');q=read(original)
    segpath=PERIODIC_OUT/(label+'_segmented_mode_checks.json');seg=read(segpath)
    cpu_files=list(PERIODIC_OUT.glob(label+'_independent_variational_BVP*.json'))
    assert cpu_files,'Independent CPU full-state RHS check is required'
    cpu_file=min(cpu_files,key=lambda f:read(f).get('epsilon_scale',1.));cpu=read(cpu_file)
    assert Path(cpu['orbit']).resolve()==Path(seg['orbit']).resolve()
    assert q['mesh_checks'][-1]['N']==seg['N']==cpu['N']
    ok,errors,phaseerrors=segmented_checks(seg)
    h=q['mesh_checks'][-1];continuous=q['continuous_defect']
    derivative_key='dJ_dcoordinate' if 'dJ_dcoordinate' in h else 'dJ_dlogT'
    curvature_key='d2J_dcoordinate2' if 'd2J_dcoordinate2' in h else 'd2J_dlogT2'
    lower=q['mesh_checks'][-2]
    same_coordinate=(curvature_key in lower and
        h.get('coordinate','logT')==lower.get('coordinate','logT'))
    root=(same_coordinate and q['J_mesh_change']<1e-7 and
        q['curvature_relative_change'] is not None and q['curvature_relative_change']<.01 and
        abs(h[derivative_key])<1e-7 and abs(h[curvature_key])>1e-8)
    shape=(continuous['maximum_group_defect_Hz']<.001 and continuous['minimum_rate_Hz']>=-1e-9
        and continuous.get('filter_state_check',{}).get('positive',False))
    nonphase=min(r['tangent_fraction_orthogonal_to_phase'] for r in q['full_state_fold_mode_checks'])>1e-3
    independent=(cpu['checks'][-1]['maximum_relative_residual']<1e-6 and
        cpu['negative_control']['maximum_relative_residual']>.1)
    passed=bool(ok and root and shape and nonphase and independent)
    result=dict(q,status='VALIDATED_CYCLE_FOLD' if passed else 'VALIDATION_INCOMPLETE',
        validation_method='FULL_STATE_SEGMENTED_VARIATIONAL_BVP_AND_INDEPENDENT_CPU_RHS',
        full_period_propagation_status=q.get('full_period_propagation_status',q['status']),
        segmented_variational_source=str(segpath),independent_CPU_RHS_source=str(cpu_file),
        segmented_fold_errors=errors,segmented_phase_errors=phaseerrors,
        independent_CPU_relative_residual=cpu['checks'][-1]['maximum_relative_residual'],
        validation_components=dict(segment_matching=ok,root_and_curvature=root,
            orbit_resolution=shape,nonphase_direction=nonphase,independent_full_RHS=independent),
        segmented_numerical_criteria=dict(finest_fold_defect=1e-4,finest_phase_defect=1e-4,
            two_successive_halving_reductions=3,minimum_negative_control=.01,
            boundary_and_phase_reconstruction=1e-10,CPU_residual=1e-6,CPU_negative_control=.1),
        scope='Numerical cycle-fold evidence from mesh-converged root/curvature, full-state periodic null tangent, independently integrated matches over every segment, and CPU full-RHS derivative. The original whole-period propagation remains recorded. This does not provide a complete Floquet spectrum or classify adjacent stability.')
    # Preserve the original unsuccessful whole-period verdict before adding
    # evidence from the independently verified BVP formulation.
    old=PERIODIC_OUT/(label+'_whole_period_validation.json')
    if not old.exists():write(old,q)
    result['whole_period_validation_source']=str(old)
    write(original,result)
    print('SEGMENTED VALIDATION',label,result['status'],result['validation_components'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('labels',nargs='+');a=p.parse_args()
    for label in a.labels:validate(label)
