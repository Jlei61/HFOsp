#!/usr/bin/env python3
"""Preserve every target's realized fan-in, retaining g40 source populations."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time
import numpy as np
from scipy import sparse
from campaign import ROOT,read,write,sha
from conditional_density_inputs import OPS
import run_topic4_loop_zk_conditional as native

OUT=ROOT/'target_density_exit'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_FREE_NETWORK_RUNS',created_epoch=time.time(),
        question='Does target-cell connection heterogeneity restore the K9 actual-exit-field spatial activity, G, and Z drift under autonomous coupling?',
        basis='Local group531 failure was repaired by individual target input means: native55.54Hz, groupmean0, individual55.36-55.77. This did not certify other groups or autonomous coupling.',
        design='Exactly two10s freely coupled runs: homogeneous target mean/variance control versus individual target mean/variance, both40000physical targets x128particles, pairednumericalseed928751. Actualexitfields Zmean.21/Kmean9; originalhigh12sfullstate and first10s existingpairedexternaldrive. Groupmeanthresholds andmembergroupexternalmean unchanged. Same frozenGaussianlocalcell andsourceg40grouping.',
        limits='Z/K held, M/R/G andrecurrence dynamic. Still a sourcegroup closure; no formalcontinuation or stability, noautomaticgrid/seed/timeextension.',
        evaluation='First10s andfixed5-10s tail: allE/core/surroundrates,400cellspatialfield,G, andevery1ms perparticleZeligibility. Compare native and homogeneouscontrol. Check CPU/GPUoperator/localupdate and capturedstateidentity beforelaunch.',
        replicas=128,numerical_seed=928751,conditions=['homogeneous','individual'],
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    start=time.time();geo=dict(np.load(OPS/'geometry.npz'));prep=read(OPS/'prepared.json')
    sim,_,_,identity=native.base.old.setup(9108405);assert identity==prep['graph_identity']
    assert np.array_equal(sim.net['pos'],geo['original_positions'])
    groups=geo['cell_group'];P=len(geo['group_size']);N=len(groups);D=prep['max_delay_steps']
    aggregation=sparse.coo_matrix((1/geo['group_size'][groups],(groups,np.arange(N))),shape=(P,N)).tocsr()
    checks=[]
    for name in ['ampa','gaba']:
        source=groups[:32000] if name=='ampa' else groups[32000:]
        rise=sim.params.tau_r_AMPA if name=='ampa' else sim.params.tau_r_GABA
        rows=[];cols=[];vals=[]
        for delay,m in enumerate(sim.net[name+'_by_delay']):
            if not m.nnz:continue
            a=m.tocoo();assert delay>=1
            rows.append(a.row);cols.append(source[a.col]+(delay-1)*P)
            vals.append(a.data/(np.where(a.row<32000,20.,10.)/rise))
        rr=np.concatenate(rows);cc=np.concatenate(cols);vv=np.concatenate(vals)
        for label,value in [('mean',vv),('variance',vv**2)]:
            a=sparse.coo_matrix((value,(rr,cc)),shape=(N,P*D)).tocsr()
            reference=sparse.load_npz(OPS/f'{label}_{name}.npz')
            delta=aggregation@a-reference;error=float(abs(delta.data).max()) if delta.nnz else 0.
            assert error<1e-10
            sparse.save_npz(OUT/f'{label}_{name}.npz',a)
            checks.append(dict(pathway=name,moment=label,nnz=a.nnz,group_projection_error=error))
            print('TARGET OPERATOR',name,label,a.nnz,error,flush=True)
        del rows,cols,vals,rr,cc,vv,a,delta,reference
    write(OUT/'operators_qa.json',dict(status='PASS',checks=checks,graph_identity=identity,
        physical_targets=N,source_groups=P,elapsed_s=time.time()-start,
        interpretation='Individual physical weights and delays retained at target; source members still share a group rate. Group aggregation exactly recovers prior first and second moment operators.'))


if __name__=='__main__':main()
