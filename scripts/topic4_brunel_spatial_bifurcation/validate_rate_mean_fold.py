"""Combine mesh, continuous residual and full delay fold-mode evidence."""
from verify_rate_fold_monodromy import *
from rate_periodic_accuracy import defect
from audit_rate_filter_states import filter_state_minima


def validation_matches_latest_root(label):
    """A coarse-root pass cannot validate a subsequently refined orbit."""
    destination=PERIODIC_OUT/(label+'_validation.json')
    versions=[read(f) for f in PERIODIC_OUT.glob(label+'_N*.json')]
    if not destination.exists() or not versions:return False
    q=read(destination);root=max(versions,key=lambda v:v['N'])
    checked=q.get('mesh_checks',[{}])[-1]
    return bool(q.get('status')=='VALIDATED_CYCLE_FOLD'
        and checked.get('N')==root['N']
        and Path(checked.get('orbit','')).resolve()==Path(root['orbit']).resolve()
        and abs(q.get('J_EE_core',float('inf'))-root['J_EE_core'])<1e-10
        and abs(q.get('T_ms',float('inf'))-root['T_ms'])<1e-7)


def independent_curvature(label, lo, hi):
    """An independent estimate may resolve subtraction error, never change the root."""
    path=PERIODIC_OUT/'curvature_rechecks'/(label+'.json')
    if not path.exists():return None
    q=read(path)
    if q.get('status')!='CURVATURE_CHECKED':return None
    for root in [lo,hi]:
        rows=[r for r in q['rows'] if r['N']==root['N']]
        if len(rows)<3:return None
        if not all(Path(r['root_orbit']).resolve()==Path(root['orbit']).resolve()
                   and abs(r['J_EE_core']-root['J_EE_core'])<1e-12
                   and r['coordinate']==root.get('coordinate') for r in rows):return None
    if not (q['mesh_relative_change']<.01 and q['step_relative_change']<.01):return None
    return dict(source=str(path),**q)


def main():
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--device',type=int,default=0)
    p.add_argument('--dt',type=float,nargs='+',default=[.05,.025,.0125])
    p.add_argument('--analytic-seed',action='store_true',help='Use separately checked analytic reconstruction; retain the same validation thresholds')
    p.add_argument('--stream-harmonics',action='store_true',help='Force verified exact frequency-block actions in all GPU checks')
    a=p.parse_args();versions=sorted([read(f) for f in PERIODIC_OUT.glob(a.label+'_N*.json')],key=lambda x:x['N'])
    assert len(versions)>=2
    lo,hi=versions[-2:];s=RateField()
    import cupy as cp
    cp.cuda.Device(a.device).use()
    bank=(hi['N']//2+1)*sum(len(v[0]) for v in s.raw)*16
    streamed=a.stream_harmonics or int(cp.cuda.runtime.memGetInfo()[0])<bank+int(4*1024**3)
    if streamed:assert read(PERIODIC_OUT/'streamed_harmonic_operator_check.json')['status']=='PASS'
    continuous=defect(s,hi['orbit'],a.device,harmonic_chunk_size=64,stream_harmonics=streamed)
    profile=np.load(hi['orbit']);continuous['filter_state_check']=filter_state_minima(s,profile['r'],float(profile['T']))
    import cupy as cp
    gc.collect();cp.get_default_memory_pool().free_all_blocks();checks=[]
    for dt in a.dt:
        suffix='_analytic' if a.analytic_seed else ''
        path=PERIODIC_OUT/f'{a.label}_monodromy_check_N{hi["N"]}_dt{dt:g}{suffix}.json'
        q=read(path) if path.exists() else check(PERIODIC_OUT/f'{a.label}_N{hi["N"]}.json',dt,a.device,a.analytic_seed,
            stream_harmonics=True if streamed else None)
        checks.append(q);gc.collect();cp.get_default_memory_pool().free_all_blocks()
    checks.sort(key=lambda x:x['dt_ms'],reverse=True)
    errors=[q['generalized_plus_one_relative_defect'] for q in checks]
    ratios=np.array(errors[:-1])/np.array(errors[1:]);meshshift=abs(hi['J_EE_core']-lo['J_EE_core'])
    curvature_key='d2J_dcoordinate2' if 'd2J_dcoordinate2' in hi else 'd2J_dlogT2'
    derivative_key='dJ_dcoordinate' if 'dJ_dcoordinate' in hi else 'dJ_dlogT'
    same_coordinate=(curvature_key in lo and hi.get('coordinate','logT')==lo.get('coordinate','logT'))
    curvature_change=(abs(hi[curvature_key]-lo[curvature_key])/abs(hi[curvature_key]) if same_coordinate else float('inf'))
    original_curvature_change=curvature_change
    curvature_recheck=independent_curvature(a.label,lo,hi)
    if curvature_recheck is not None:curvature_change=curvature_recheck['mesh_relative_change']
    passed=(meshshift<1e-7 and curvature_change<.01 and abs(hi[derivative_key])<1e-7
        and continuous['maximum_group_defect_Hz']<.001
        and continuous['minimum_rate_Hz']>=-1e-9 and continuous['filter_state_check']['positive'] and errors[-1]<.001
        # A coarse step need not be in the second-order asymptotic regime.
        # Require two successive reductions on the three finest steps; keep
        # all coarser errors visible in the saved evidence.
        # Faster error reduction is acceptable. Physical-delay interpolation
        # fractions change with the non-nested period-fitted time grids; an
        # upper bound on the reduction would reject improved accuracy.
        and len(ratios)>=2 and np.all(ratios[-2:]>3)
        and min(q['tangent_fraction_orthogonal_to_phase'] for q in checks)>1e-3)
    result=dict(status='VALIDATED_CYCLE_FOLD' if passed else 'VALIDATION_INCOMPLETE',label=a.label,
        J_EE_core=hi['J_EE_core'],T_ms=hi['T_ms'],mesh_checks=versions,J_mesh_change=meshshift,
        curvature_relative_change=curvature_change,continuous_defect=continuous,
        original_tangent_curvature_relative_change=original_curvature_change,
        independent_parameter_curvature=curvature_recheck,
        curvature_coordinate=hi.get('coordinate','logT'),matching_mesh_coordinate=same_coordinate,
        full_state_fold_mode_checks=checks,successive_defect_reduction=ratios,
        asymptotic_check_dt_ms=[q['dt_ms'] for q in checks[-3:]],
        seed_method='analytic harmonic derivative' if a.analytic_seed else 'centered finite difference',
        scope='Numerical simple cycle-fold evidence in the frozen full spatial rate DDE; no complete Floquet stability classification of adjacent branches.',
        numerical_criteria=dict(J_mesh_change=1e-7,curvature_relative_change=.01,derivative=1e-7,
            continuous_defect_Hz=.001,finest_mode_relative_defect=.001,minimum_halving_reduction=3,
            convergence_rule='At least threefold reduction twice on the three finest steps, with no upper bound; all coarser data retained.',
            criterion_revision='Remove an unjustified maximum error-reduction ratio; tighten the finest relative defect from .01 to .001. Equations and recorded errors unchanged.'))
    destination=PERIODIC_OUT/(a.label+'_validation.json')
    if a.analytic_seed and destination.exists():
        previous=read(destination)
        if previous.get('seed_method')!='analytic harmonic derivative':
            archive=PERIODIC_OUT/(a.label+'_finite_difference_validation.json')
            if not archive.exists():write(archive,previous)
            result['previous_finite_difference_validation']=str(archive)
    write(destination,result)
    print('MEAN FOLD VALIDATION',a.label,result['status'],'J',hi['J_EE_core'],'meshshift',meshshift,'defects',errors,flush=True)


if __name__=='__main__':main()
