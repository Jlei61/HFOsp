"""Same-graph 0.5-mm checks, separating network resolution and Z-field detail.

Fine groups exactly partition coarse groups. Preserve the full coarse initial
physical state and delay history on its original member cells. Primary lifted-Z
arms keep the identical cell-level piecewise-constant Z field; secondary native-Z
arms resolve the original recorded within-cell variation as a separate factor.
"""
from native_path import *
from scipy import sparse
import argparse

DEST=OUT/'spatial_refinement'
INITIAL=OUT/'initial_states/seed_N1024_dt0.05.npz'


def restriction_and_parent(s20,s40):
    assert s20.prep['graph_identity']==s40.prep['graph_identity']
    assert np.array_equal(s20.geo['original_positions'],s40.geo['original_positions'])
    g20=s20.geo['cell_group'];g40=s40.geo['cell_group'];parent=np.full(s40.P,-1,dtype=int)
    for j in range(s40.P):
        unique=np.unique(g20[g40==j]);assert len(unique)==1
        parent[j]=unique[0]
    assert np.array_equal(np.bincount(parent,weights=s40.sizes,minlength=s20.P),s20.sizes)
    Q=sparse.csr_matrix((s40.sizes/s20.sizes[parent],(parent,np.arange(s40.P))),shape=(s20.P,s40.P))
    assert np.max(abs(Q@s40.theta-s20.theta))<1e-12
    assert np.array_equal(s40.E,s20.E[parent])
    return Q,parent


def prepare():
    contract=read(OUT/'native_spatial_refinement_contract.json')
    DEST.mkdir(exist_ok=True)
    s20=model(20);path=attach_native_path(s20);s40=model(40)
    Q,parent=restriction_and_parent(s20,s40)
    source=np.load(INITIAL)
    assert float(source['dt_ms'])==.05 and int(source['tick'])==0
    assert Path(str(source['source'])).resolve()==(OUT/'periodic/seed_N1024.npz').resolve()
    state=source['state'][:,parent].copy();history=source['history'][:,parent].copy()
    state_error=float(np.max(abs((Q@state.T).T-source['state'])))
    history_error=float(np.max(abs((Q@history.T).T-source['history'])))
    assert state_error<1e-11 and history_error<1e-13
    np.savez_compressed(DEST/'initial_g40.npz',state=state,history=history,tick=0,
        dt_ms=.05,source=str(INITIAL),parent_g20=parent,Z=state[11])
    # Linear graph moments must commute with aggregation for parent-constant
    # rates. This checks physical edge/weight/delay identity independently of
    # the metadata equality; local nonlinear thresholds are allowed to resolve.
    rng=np.random.default_rng(9194020);probes=[source['history'][0],rng.uniform(.001,.1,s20.P)]
    checks=[]
    for lam in [0.,.03j]:
        for kind,(a,b) in enumerate(zip(s20.matrices(lam),s40.matrices(lam))):
            for j,v in enumerate(probes):
                actual=Q@(b@v[parent]);expected=a@v
                error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
                assert error<1e-11,error
                checks.append(dict(lambda_per_ms=[complex(lam).real,complex(lam).imag],moment=kind,probe=j,relative_error=error))
    observed=[]
    checkpoint=ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/runs/eta0.0005_s9108401/checkpoints'
    for t,coarse in zip(path['times_ms'],path['fields']):
        z=np.load(checkpoint/f't{t}ms.npz')['slow__z'][:32000].astype(float)
        fine=np.ones(s40.P);fine[s40.E]=s40.project(z)[s40.E]
        assert np.max(abs(Q@fine-coarse))<2e-14
        observed.append(fine)
    np.savez_compressed(DEST/'native_fields_g40.npz',times_ms=path['times_ms'],
        D=path['D'],fields=np.array(observed),parent_g20=parent)
    rows=[]
    for D in contract['D']:
        s20.set_D(D);lifted=s20.Z[parent]
        fineZ=fine_native_field(s40,D,np.asarray(observed),path['D'])
        assert np.max(abs(Q@fineZ-s20.Z))<2e-14
        rows.append(dict(D=D,global_D_lifted=float(1-lifted[s40.E]@s40.mean_weights),
            global_D_native_fine=float(1-fineZ[s40.E]@s40.mean_weights),
            E_cell_weighted_Z_detail_RMS=float(np.sqrt(np.average((fineZ[s40.E]-lifted[s40.E])**2,weights=s40.sizes[s40.E]))),
            max_Z_detail=float(np.max(abs(fineZ-lifted)))))
    q=dict(status='PREPARATION_PASS',coarse_groups=s20.P,fine_groups=s40.P,
        coarse_grid=20,fine_grid=40,graph_identity=s20.prep['graph_identity'],
        exact_parent_partition=True,state_aggregation_max_error=state_error,
        history_aggregation_max_error=history_error,linear_moment_checks=checks,Z_fields=rows,
        initial_source=str(INITIAL),scope='Projection and application correctness only; fine dynamics not inferred.')
    write(DEST/'preparation.json',q);log('SPATIAL REFINEMENT PREPARATION',q)


def fine_native_field(s,D,fields,knots):
    high=np.ones(s.P);high[s.E]=0
    allD=np.r_[0.,knots,1.];allZ=np.vstack([np.ones(s.P),fields,high])
    j=min(max(np.searchsorted(allD,D,side='right')-1,0),len(allD)-2)
    f=(D-allD[j])/(allD[j+1]-allD[j]);return (1-f)*allZ[j]+f*allZ[j+1]


def run(a):
    import runner
    from endpoint_runs import EndpointIntegrator
    assert read(DEST/'preparation.json')['status']=='PREPARATION_PASS'
    contract=read(OUT/'native_spatial_refinement_contract.json');assert a.D in contract['D']
    s=model(40);fields=np.load(DEST/'native_fields_g40.npz')
    if a.Z_detail=='native':
        Z=fine_native_field(s,a.D,fields['fields'],fields['D'])
    else:
        coarse=model(20);attach_native_path(coarse);coarse.set_D(a.D);Z=coarse.Z[fields['parent_g20']]
    assert abs(1-Z[s.E]@s.mean_weights-a.D)<2e-14
    label=f'native_g40_{a.Z_detail}Z_D{a.D:.7f}_dt0.05'
    runner.Integrator=EndpointIntegrator
    q=runner.run_condition(s,Z,label,contract['duration_ms'],a.device,DEST/'initial_g40.npz',dt=.05)
    log('SPATIAL REFINEMENT ARM COMPLETE',q['label'])
    return q


def coarse_prefix(device):
    import runner
    from endpoint_runs import EndpointIntegrator
    runner.Integrator=EndpointIntegrator;s=model(20);attach_native_path(s);rows=[]
    for D in read(OUT/'native_spatial_refinement_contract.json')['D']:
        s.set_D(D);label=f'native_spatial_coarse_prefix_D{D:.7f}'
        old=np.load(OUT/'runs'/f'endpoint_D{D:.7f}_dt0.05'/'trajectory.npz')
        interpolation_difference=float(np.max(abs(s.Z-old['Z_source'])))
        assert interpolation_difference<2e-14
        if not np.array_equal(s.Z,old['Z_source']):label+='_exactZ'
        # Replay the saved physical field itself. Reconstructing a checkpoint
        # knot through floating-point interpolation can differ by a few ulps.
        runner.run_condition(s,old['Z_source'],label,100,device,INITIAL,dt=.05)
        new=np.load(OUT/'runs'/label/'trajectory.npz')
        checks={}
        for key in ['group_rate_hz','global_E_hz','field_E_hz','Z_every50ms','M_every50ms']:
            x=new[key];y=old[key][:len(x)]
            exact=bool(np.array_equal(x,y))
            # Double-precision reductions may differ by rounding even when the
            # recorded float32 rates/fields and held Z are bitwise identical.
            # Retain that distinction explicitly; do not label this bitwise PASS.
            rounding_allowed=key in ['global_E_hz','M_every50ms']
            scale=np.maximum(np.abs(y),1. if key=='global_E_hz' else 1e-3)
            normalized=float(np.max(np.abs(x-y)/scale))
            eps=np.finfo(float).eps
            # Global means are separately reduced over P groups with different
            # total recording lengths. Two dot products have a conservative
            # 2*gamma_P summation bound; it is not a fixed handful-of-ulps bound.
            allowance=2*s.P*eps/(1-s.P*eps) if key=='global_E_hz' else 32*eps
            passed=exact or (rounding_allowed and normalized<=allowance)
            checks[key]=dict(bitwise_equal=exact,max_absolute_difference=float(np.max(abs(x-y))),
                scaled_difference=normalized,rounding_allowance=allowance if rounding_allowed else 0.,pass_check=passed)
        assert all(c['pass_check'] for c in checks.values()),checks
        rows.append(dict(D=D,checks=checks,source=str(INITIAL),label=label,
            reconstructed_vs_original_Z_max_difference=interpolation_difference))
    write(DEST/'coarse_prefix_replay.json',dict(status='PASS',duration_ms_each=100,rows=rows,
        scope='Verification of current complete initial/history against existing comparison-run prefixes; not new independent dynamics evidence.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    p.add_argument('--batch',action='store_true')
    p.add_argument('--coarse-prefix-check',action='store_true')
    p.add_argument('--D',type=float);p.add_argument('--Z-detail',choices=['lifted','native'],default='lifted')
    p.add_argument('--device',type=int,default=1);a=p.parse_args()
    if a.prepare:prepare()
    elif a.coarse_prefix_check:coarse_prefix(a.device)
    elif a.batch:
        import gc
        rows=[]
        for arm in read(OUT/'native_spatial_refinement_contract.json')['new_arms']:
            a.D=arm['D'];a.Z_detail=arm['Z_detail'];rows.append(run(a))
            write(DEST/'batch.json',dict(status='RUNNING',rows=rows))
            gc.collect()
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        write(DEST/'batch.json',dict(status='COMPLETE',rows=rows))
    else:run(a)
