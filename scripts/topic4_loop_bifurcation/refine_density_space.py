#!/usr/bin/env python3
"""One frozen spatial-grouping diagnostic, unchanged density physics kernel."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from scipy import sparse
from campaign import ROOT,REPO,NATIVE,read,write,sha
import density_spatial as engine

OUT=ROOT/'density_spatial_grouping'
OPERATORS=REPO/'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators'


def prepare():
    coarse=dict(np.load(OPERATORS/'g20/geometry.npz'));fine=dict(np.load(OPERATORS/'g40/geometry.npz'))
    pc=len(coarse['group_size']);pf=len(fine['group_size'])
    parent=np.full(pf,-1,dtype=int)
    for member,g in enumerate(fine['cell_group']):
        old=coarse['cell_group'][member]
        if parent[g]<0:parent[g]=old
        assert parent[g]==old
    assert np.array_equal(fine['original_positions'],coarse['original_positions'])
    assert np.array_equal(np.bincount(parent,weights=fine['group_size'],minlength=pc),coarse['group_size'])
    theta=np.bincount(parent,weights=fine['threshold_mv']*fine['group_size'],minlength=pc)/coarse['group_size']
    assert np.max(abs(theta-coarse['threshold_mv']))<1e-12
    for grid in ['g20','g40']:assert read(OPERATORS/grid/'prepared.json')['graph_identity']==read(NATIVE/'protocol.json')['identity']
    matrix_errors={};rng=np.random.default_rng(927027)
    for kind in ['mean_ampa','variance_ampa','mean_gaba','variance_gaba']:
        a=sparse.load_npz(OPERATORS/f'g20/{kind}.npz').tocsr();b=sparse.load_npz(OPERATORS/f'g40/{kind}.npz').tocsr()
        depth=a.shape[1]//pc;assert b.shape[1]==depth*pf
        rate=rng.uniform(0,.4,(depth,pc));old=a@rate.ravel();new=b@rate[:,parent].ravel()
        reduced=np.bincount(parent,weights=new*fine['group_size'],minlength=pc)/coarse['group_size']
        error=float(abs(reduced-old).max());relative=error/max(1.,float(abs(old).max()))
        assert relative<2e-12,(kind,error,relative)
        matrix_errors[kind]=dict(max_absolute_error=error,relative_to_max=relative)
    # The engine uses group_cell only for saved1mm forcing and display projection.
    # Physics operators remain the original fine-grid operators and delays.
    adapted=OUT/'operators';adapted.mkdir()
    for name in ['prepared.json','mean_ampa.npz','variance_ampa.npz','mean_gaba.npz','variance_gaba.npz']:
        (adapted/name).symlink_to(OPERATORS/'g40'/name)
    gcell=fine['group_cell'];cell20=(gcell%40)//2+20*((gcell//40)//2)
    assert np.array_equal(cell20,coarse['group_cell'][parent])
    fine['group_cell_native_fine']=gcell.copy();fine['group_cell']=cell20
    np.savez_compressed(adapted/'geometry.npz',**fine)
    variance=[]
    for g,R in [(coarse,8192),(fine,2048)]:
        e=g['population']==0;variance.append(float(np.sum(g['group_size'][e]**2)/R/32000**2))
    qa=dict(status='PASS_OPERATOR_REFINEMENT_ALGEBRA',fine_groups=pf,coarse_groups=pc,
        all_fine_groups_nested_in_one_coarse_group=True,original_positions_and_graph_identity=True,
        threshold_average_error=float(abs(theta-coarse['threshold_mv']).max()),
        delayed_first_and_squared_weight_action_errors=matrix_errors,
        original_1mm_external_forcing_held_identical=True,
        homogeneous_Bernoulli_numerical_variance_factors=variance,
        variance_note='Only a homogeneous-independent-sampling size diagnostic; unequal group rates and recurrent feedback prevent exact numerical-noise matching. Fine2048 andcoarse8192 have similar total particles, not paired physical noises.')
    write(OUT/'operator_check.json',qa);return adapted


def main(device):
    evidence=read(ROOT/'density_spatial_resolution/comparison.json');assert evidence['status']=='COMPLETE'
    assert read(ROOT/'density_contact_replay/direction_comparison.json')['status']=='COMPLETE'
    assert not (OUT/'contract.json').exists();OUT.mkdir(exist_ok=True)
    write(OUT/'contract.json',dict(status='REGISTERED_AFTER_BASELINE_RESOLUTION_AND_CONTACT_REVIEW',created_epoch=time.time(),
        question='Does finer spatial grouping improve systematically narrow recruitment and slow resource consumption with the same local distribution equations?',
        evidence='Coarse8192 retains39events,median86ms,area.601 andquiet.612 vsnative40/42events,100/97.5ms,area.746/.676 andquiet.518/.492. D9870=.212 vsnative.256. All3densityruns pass originalbroadA4, but numericalentry shifts.543s with4xparticlecount. Fixedcontactobserver mostlyretains order; directionalnativepositiveclass has only1/3events and cannot certifyequivalence.',
        bounded_design='Exactlyone12.5s run, g40=3479 original spatialgroups,2048numericalparticles/group,seed927611,dt.1ms,G/Koff,same recorded8401one-mmexternalforcing. Unchanged originaldensitykernel. No fit, biologicalparameterchange, newnative seed, or automatic further refinement.',
        comparison='Originalg20 8192-particle run hassimilar totalquadraturecount. This is a spatial-resolution diagnostic, not an exactly noise-paired causal estimate or convergence certificate.',
        implementation_gate='Finegroupsnestedincoarsegroups; originalgraphandpositions; projectedthresholdmeans; all4delayedmean/squaredweightoperators contract exactly on coarseconstant rate fields. Only observation/forcing group_cell remapped to original1mmcells; finephysicsoperatorsunchanged.',
        decision_rule='If eventarea/Z/contacts remain biased, do not certify continuation or fit an onset correction; inspect input/correlation approximation. If they improve, still require numericalconvergence and actualG/Knative correspondence. No new broadergrid follows automatically.',
        engine_sha256=sha(engine.__file__),source_sha256=sha(__file__),device=device,
        formal_bifurcation_allowed=False,human_review='PENDING'))
    adapted=prepare();start=time.time();old=engine.OPERATORS
    try:
        engine.OPERATORS=adapted;e=engine.DensityNetwork(replicas=2048,seed=927611,device=device,duration_ms=12500,gain=0.)
    finally:engine.OPERATORS=old
    e.graph();output=[]
    for tick in range(0,12500,10):
        x=e.chunk();assert np.isfinite(x).all() and x[:,1].min()>=0 and x[:,1].max()<=1;output.append(x)
        if (tick+10)%250==0:write(OUT/'progress.json',dict(status='RUNNING',pid=os.getpid(),time_ms=tick+10,elapsed_s=time.time()-start))
    data=np.concatenate(output);cell=e.geo['group_cell'];field=np.zeros((len(data),400));count=np.zeros(400)
    for g in np.flatnonzero(e.E):field[:,cell[g]]+=data[:,0,g]*e.sizes[g];count[cell[g]]+=e.sizes[g]
    field/=np.maximum(count,1)
    arrays=dict(time_ms=np.arange(12500)+1.,field_E_Hz=field.astype('f4'),cell_counts=count,group_sizes=e.sizes,population_E=e.E)
    for j,k in enumerate(['group_rate_Hz','group_Z','group_M','group_K','group_IE','group_applied_II','group_V','group_abs_current']):arrays[k]=data[:,j].astype('f4')
    np.savez_compressed(OUT/'trajectory.npz',**arrays)
    np.savez_compressed(OUT/'final_state.npz',state=e.state.get(),ref=e.ref.get(),history=e.history.get(),rng=e.rng.get(),clock=e.clock.get(),global_state=e.global_state.get(),accumulator=e.accumulator.get(),particle_count=2048,seed=927611)
    result=dict(status='COMPLETE',duration_ms=12500,elapsed_s=time.time()-start,groups=e.P,particles_per_group=e.R,
        engine_sha256_unchanged=sha(engine.__file__)==read(OUT/'contract.json')['engine_sha256'],native_correspondence_certified=False)
    write(OUT/'result.json',result);write(OUT/'progress.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args().device)
