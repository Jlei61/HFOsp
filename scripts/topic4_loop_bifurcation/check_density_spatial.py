#!/usr/bin/env python3
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time,argparse
import numpy as np
from campaign import ROOT,write,read,sha
from density_spatial import DensityNetwork,cpu_cell,OPERATORS,DRIVE

OUT=ROOT/'density_spatial_baseline'


def main(device):
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_SPATIAL_DIAGNOSTIC',created_epoch=time.time(),
        question='Does a full local voltage/current/refractory/Z/M distribution recover the native spatial interictal substrate without a learned hazard?',
        scope='First verify same graph grouping and native local updates. Then only 3s prefixes at512 and2048 particles/group, paired numerical streams, original recorded seed9108401 external forcing. One further2048 numerical seed only if needed for numerical-noise distinction. No automatic12.5s extension or continuation.',
        baseline='Existing g20=935 projected native groups, same physical graph, thresholds, delays, E/I counts, reset and2/1ms refractory, dynamic original Z/M; addedG/K zero for this prerequisite interictal correspondence. Not a new autonomous-loop SNN seed.',
        remaining_approximation='Recurrent arrivals are independent Gaussian increments with exact grouped first/squared-weight second moments; dt0.1ms decay-arrival-current ordering follows native. Group heterogeneity beyond declared thresholds/geometry and cross-cell synaptic noise correlations are omitted. External per1ms group mean drive is recorded; sub-ms and within-group forcing variation are omitted.',
        local_distribution='Each numerical particle retainsV,sE,IE,sI,II,M,Z,K andref. Z eligibility uses its own raw inhibitory current; M uses its own spikes. This preserves their within-cell dependence. Native Euler Z/M and exponential K/R/G update orders are retained. No transfer function, neural weights or refitted network parameters.',
        discretization_note='Earlier colored_population_local checks used the existing continuous-diffusion exact covariance assay. Here Gaussian arrivals approximate the actual native discrete shot-noise update, whose covariance differs by finite-dt terms. This distinct kernel receives its own independent CPU/transport/finite-window checks.',
        implementation_gate='935-group delayed CSR aggregation against independent CPU at three wraparound clocks; complete particle/spike/ref state versus independent CPU with G0 andG30, high/lowglobalR; finite physical bounds; graph-capture prefix equals uncaptured steps; exact discrete Gaussian increment mean/covariance algebra.',
        numerical_gate='512/2048 comparison is numerical resolution, not physiological seed replication. No correspondence acceptance until reduced dynamics remain qualitatively stable with resolution and native events/propagation/contacts match prespecified native-noise tolerance. A3s prefix can only reject obvious mismatch, not certify onset or closure.',
        stop='If no core-led separated brief events, or persistent high firing before native onset, inspect the residual input/correlation/grouping approximation. Do not enlarge response MLP, fit onset or label bifurcation.',
        resources='One extra GPU process, bounded<=2048*935 particles, estimated<1GiB states plus fixed operators. Leaves native12-worker queue unchanged. No broad parameter scan.',
        source_sha256=sha(__file__),engine_sha256=sha(__import__('density_spatial').__file__),
        operators=str(OPERATORS),forcing=str(DRIVE),forcing_sha256=sha(DRIVE),human_review='PENDING'))
    e=DensityNetwork(replicas=64,device=device,duration_ms=3000);cp=e.cp;P,R=e.P,e.R
    rng=np.random.default_rng(927621);errors=[]
    history=rng.uniform(0,.1,(e.depth,P));e.history[:]=cp.asarray(history)
    for tick in [0,e.depth-1,e.depth+17]:
        e.clock.fill(tick)
        e.k['delayed']((P,),(128,),(*e.ops,e.history,e.arr,e.clock,np.int32(e.depth),np.int32(P)))
        expected=[]
        for a,q in e.ops_cpu:
            cols=a.indices;values=history[(tick-(cols//P+1))%e.depth,cols%P]
            row=np.repeat(np.arange(P),np.diff(a.indptr))
            expected.append([np.bincount(row,weights=m.data*values,minlength=P) for m in [a,q]])
        expected=np.array([expected[0][0],expected[1][0],expected[0][1],expected[1][1]])
        error=float(abs(expected-e.arr.get()).max());assert error<1e-11,error;errors.append(dict(tick=tick,max_error=error))
    checks=[]
    for gain,global_state in [(0.,[0.,0.]),(30.,[350.,.6]),(30.,[3.,.3])]:
        initial=np.zeros((P,R,8));initial[:,:,0]=rng.uniform(11,17,(P,R));initial[:,:,1:5]=rng.uniform(0,100,(P,R,4))
        initial[:,:,5]=rng.uniform(0,20,(P,R))*e.E[:,None];initial[:,:,6]=np.where(e.E[:,None],rng.uniform(.2,.99,(P,R)),1.)
        initial[:,:,7]=rng.uniform(0,10,(P,R))*e.E[:,None]
        ref=rng.integers(0,20,(P,R),dtype=np.int32);normal=rng.normal(size=(P,R,2));arr=rng.uniform(0,.6,(4,P));nu=e.drive_cpu[0]
        target=cpu_cell(initial,ref,e.pars_cpu,e.constants_cpu,arr,nu,global_state,gain,normal)
        e.state[:]=cp.asarray(initial);e.ref[:]=cp.asarray(ref);e.arr[:]=cp.asarray(arr);e.global_state[:]=cp.asarray(global_state)
        e.k['supplied'](((P*R+127)//128,),(128,),(e.state,e.ref,cp.asarray(normal),e.pars,e.constants,e.arr,cp.asarray(nu),e.global_state,gain,e.spikes,np.int32(P),np.int32(R)))
        error=float(abs(e.state.get()-target[0]).max());assert error<1e-10,error
        assert np.array_equal(e.ref.get(),target[1]) and np.array_equal(e.spikes.get(),target[2])
        checks.append(dict(gain=gain,global_R=global_state[0],max_state_error=error,spikes_exact=True,ref_exact=True))
    e.reset()
    for _ in range(100):e.step()
    cpu_capture={k:getattr(e,k).get() for k in ['state','ref','spikes','history','clock','global_state','output','rng']}
    e.graph();e.chunk()
    for k,v in cpu_capture.items():assert np.array_equal(v,getattr(e,k).get()),k
    assert np.isfinite(e.state.get()).all()
    result=dict(status='PASS',transport=errors,local_physics=checks,graph_capture_bitwise=True,
        graph_identity_exact=True,source_sha256=sha(__import__('density_spatial').__file__),
        note='Implementation gate only; Gaussian arrival and spatial/noise correspondence unvalidated.')
    write(OUT/'implementation_check.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args().device)
