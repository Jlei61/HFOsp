#!/usr/bin/env python3
"""Test target-cell connection heterogeneity in existing K9 observations.

Only the target-cell projection changes. Native upstream group counts and
R/G are prescribed. No native simulation, fitting, or branch certification.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from scipy import sparse
from campaign import ROOT,read,write,sha
from conditional_density_inputs import ARRIVAL_CODE,OPS
import conditional_exit_branch_density as grouped
from conditional_exit_density import LocalExit
import run_topic4_loop_zk_conditional as native

OUT=ROOT/'conditional_target_heterogeneity'
SEEDS=[928741,928742]


def operators(initial,geo):
    sim,_,_,identity=native.base.old.setup(9108405)
    prep=read(OPS/'prepared.json');assert identity==prep['graph_identity']
    cells=initial['native_cells'];N=len(cells);P=len(geo['group_size']);D=prep['max_delay_steps']
    groups=geo['cell_group'];ids=initial['selected_group_index'];G=len(initial['selected_groups'])
    size=np.bincount(ids,minlength=G)
    aggregation=sparse.coo_matrix((1/size[ids],(ids,np.arange(N))),shape=(G,N)).tocsr()
    result={};checks=[]
    for name in ['ampa','gaba']:
        rise=sim.params.tau_r_AMPA if name=='ampa' else sim.params.tau_r_GABA
        source=groups[:32000] if name=='ampa' else groups[32000:]
        rows=[];cols=[];weights=[]
        for delay,matrix in enumerate(sim.net[name+'_by_delay']):
            if not matrix.nnz:continue
            a=matrix.tocsr()[cells].tocoo()
            if not a.nnz:continue
            assert delay>=1
            physical=a.data/(np.where(cells[a.row]<32000,20.,10.)/rise)
            rows.append(a.row);cols.append(source[a.col]+(delay-1)*P);weights.append(physical)
        rr=np.concatenate(rows);cc=np.concatenate(cols);vv=np.concatenate(weights)
        for label,value in [('mean',vv),('variance',vv**2)]:
            a=sparse.coo_matrix((value,(rr,cc)),shape=(N,P*D)).tocsr()
            reference=sparse.load_npz(OPS/f'{label}_{name}.npz')[initial['selected_groups']]
            delta=aggregation@a-reference;error=float(abs(delta.data).max()) if delta.nnz else 0.
            assert error<1e-10,(name,label,error)
            result[f'{label}_{name}']=a
            sparse.save_npz(OUT/f'{label}_{name}.npz',a)
            checks.append(dict(pathway=name,moment=label,group_projection_error=error,nnz=a.nnz))
    write(OUT/'operator_qa.json',dict(status='PASS',identity=identity,checks=checks,
         interpretation='Exact individual physical weights and delays; squared weights formed before aggregation. Existing grouped operators recovered algebraically.'))
    return result


def main(device):
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(grouped.OUT/'result.json')['status']=='COMPLETE'
    assert read(grouped.SOURCE/'observer_audit.json')['status']=='PASS'
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_TARGET_TEST',created_epoch=time.time(),
        question='Does preserving target-cell fan-in heterogeneity recover the active edge group missed by group-mean input?',
        motivation='Existing geometric group531 has55.54Hz native but zero grouped local density; replacing group current means did not repair it. Most other selectedE groups are near refractory saturation, so aggregateagreement hides this failure.',
        design='Same210originalcells and42-44s observed sourcegroupcounts/R/G. Threearms: group mean/variance; individual target means with group variance; individual target means and variance. Two numerical streams,1024replicas per physical target. Original percell jointstate/pendingZ/K retained. Groupmeanthresholds/externaldrive identical in all arms to isolate target-input projection.',
        evaluation='Fixed42.5-44s tail, all16groups retained including edge531; compare native counts and current group means/spread. The selected failure is a development diagnostic, not independent validation.',
        bounds='Six local conditions only. No new native trajectory, parameter scan, noise fitting, or acceptance threshold.',
        physical_engine_sha256=sha(grouped.physical.__file__),local_kernel_sha256=sha(grouped.__file__),
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    import cupy as cp
    cp.cuda.Device(device).use();started=time.time()
    previous=grouped.OUT;grouped.OUT=OUT/'group_projection';grouped.OUT.mkdir(exist_ok=True)
    try:z,initial,pars,c,_,pending,base=grouped.prepare(cp)
    finally:grouped.OUT=previous
    geo=dict(np.load(OPS/'geometry.npz'));matrices=operators(initial,geo)
    ids=initial['selected_group_index'];N=len(ids);G=len(pars);P=len(geo['group_size']);T=len(z['time_ms'])
    spikes=cp.asarray(z['spikes']);sizes=cp.asarray(geo['group_size'],dtype='f8');output=cp.zeros((T,6,N))
    kernel=cp.RawKernel(ARRIVAL_CODE,'apply',options=('--fmad=false',))
    for j,name in enumerate(['mean_ampa','mean_gaba','variance_ampa','variance_gaba']):
        a=matrices[name];parts=(cp.asarray(a.indptr,dtype='i4'),cp.asarray(a.indices,dtype='i4'),cp.asarray(a.data))
        kernel((N,T),(128,),(*parts,spikes,sizes,output,np.int32(P),np.int32(N),np.int32(j)))
        cp.cuda.get_current_stream().synchronize()
    individual=output.get()[:,:4].copy();homogeneous=base['projected_full'][:,:,ids].copy()
    checks=[]
    for g in range(G):
        error=float(abs(individual[:,:,ids==g].mean(2)-base['projected_full'][:,:,g]).max())
        assert error<1e-8;checks.append(error)
    write(OUT/'input_qa.json',dict(status='PASS',target_to_group_arrival_max_errors=checks,
          initial_pending_same_cell=True,source_counts_and_global_prescribed=True))
    del output,spikes,sizes,base,matrices
    cellpars=pars[ids].copy();cellpars[:,5]=1
    cellz={**z,'external_rate_per_ms':z['external_rate_per_ms'][:,ids]}
    members=np.repeat(np.arange(N,dtype='i4')[:,None],1024,axis=1)
    means_only=homogeneous.copy();means_only[:,:2]=individual[:,:2]
    arms={'homogeneous':homogeneous,'individual_mean':means_only,'individual_full':individual}
    module=cp.RawModule(code=grouped.physical.CODE+grouped.CODE,options=('--fmad=false',),
                        name_expressions=['rng_bytes','init_rng','exit_steps'])
    initial_means=np.array([initial['state'][ids==g].mean(0) for g in range(G)])
    np.savez_compressed(OUT/'reference.npz',selected_groups=z['selected_groups'],native_counts=z['spikes'][:,z['selected_groups']],
         native_moments=z['moments'],moment_names=z['moment_names'],group_sizes=pars[:,5],cell_group_index=ids,
         native_cells=initial['native_cells'],initial_group_means=initial_means)
    completed=[]
    for seed in SEEDS:
        for name,arr in arms.items():
            job=f'{name}_num{seed}';e=LocalExit(cp,module,arr,cellz,initial,cellpars,c,members,pending,seed)
            rates=[];mom=[]
            for tick in range(0,T,100):
                e.counts.fill(0);e.advance(tick,100);rates.append(e.counts.sum(1).get().T/e.R*1000.)
                x=e.state
                mom.append(cp.stack([x[:,:,2].mean(1),x[:,:,4].mean(1),x[:,:,2].var(1),x[:,:,4].var(1)],axis=1).get())
                if (tick+100)%5000==0:write(OUT/'progress.json',dict(status='RUNNING',job=job,pid=os.getpid(),
                      time_ms=42000+(tick+100)*.1,completed=completed,updated_epoch=time.time()))
            assert np.array_equal(e.state.get()[:,:,[6,7]],initial['state'][members][:,:,[6,7]])
            np.savez_compressed(OUT/f'{job}.npz',rate_Hz=np.concatenate(rates),moments=np.array(mom),
                  moment_names=['IE','II','IEvar','IIvar'],final_state=e.state.get(),final_ref=e.ref.get())
            completed.append(job);print('TARGET HETEROGENEITY COMPLETE',job,flush=True)
            del e;cp.get_default_memory_pool().free_all_blocks()
    result=dict(status='COMPLETE',completed=completed,elapsed_s=time.time()-started,
         held_fields_bitwise=True,formal_bifurcation_allowed=False)
    assert sha(grouped.physical.__file__)==read(OUT/'contract.json')['physical_engine_sha256']
    write(OUT/'result.json',result);write(OUT/'progress.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);main(p.parse_args().device)
