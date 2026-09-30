#!/usr/bin/env python3
"""Four bounded local tests in the native actual-field K9 high state.

Observed recurrent input and global feedback are prescribed. This diagnoses
local response versus recurrent closure; it is not a bifurcation computation.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import time
import numpy as np
from scipy import sparse
from campaign import ROOT, read, write, sha
import density_spatial as physical
from conditional_density_inputs import ARRIVAL_CODE, OPS
from conditional_exit_density import mean_path, LocalExit
from observe_exit_branch_inputs import OUT as SOURCE

OUT = ROOT / 'conditional_exit_branch_density'
SEEDS = [928731, 928732]
T = 20000
CODE = r'''
extern "C" __global__ void exit_steps(double* state,int* refs,unsigned char* memory,
 const double* pars,const double* constants,const double* arr,const double* drive,
 const double* global,const double* pending,const int* members,int* counts,
 int G,int R,int start,int steps,int depth,int cells){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=G*R)return;
 int g=id/R,ref=refs[id],member=members[id];double x[8];
 for(int j=0;j<8;j++)x[j]=state[id*8+j];
 double heldZ=x[6],heldK=x[7];
 curandStatePhilox4_32_10_t* all=(curandStatePhilox4_32_10_t*)memory;
 curandStatePhilox4_32_10_t rng=all[id];
 for(int k=start;k<start+steps;k++){
  if(k<depth){x[1]+=pending[(long long)k*cells+member]/constants[0];
              x[3]+=pending[((long long)depth+k)*cells+member]/constants[2];}
  float4 n=curand_normal4(&rng);double a[4];
  for(int j=0;j<4;j++)a[j]=arr[((long long)k*4+j)*G+g];
  bool sp=native_cell(x,ref,pars+6*g,constants,a,drive[(long long)k*G+g],global+2*k,30.,n.x,n.z);
  x[6]=heldZ;x[7]=heldK;
  if(sp)counts[(long long)id*10+(k%100)/10]++;
 }
 refs[id]=ref;all[id]=rng;for(int j=0;j<8;j++)state[id*8+j]=x[j];
}
'''


def prepare(cp):
    z = dict(np.load(SOURCE/'inputs.npz'))
    initial = dict(np.load(SOURCE/'initial_local_state.npz'))
    geo = dict(np.load(OPS/'geometry.npz'))
    selected = z['selected_groups']; G = len(selected); P = len(geo['group_size'])
    assert len(z['time_ms']) == T and np.array_equal(selected, initial['selected_groups'])
    matrices = [sparse.load_npz(OPS/f'{name}.npz')[selected].tocsr()
                for name in ['mean_ampa', 'mean_gaba', 'variance_ampa', 'variance_gaba']]
    spikes = cp.asarray(z['spikes']); sizes = cp.asarray(geo['group_size'], dtype='f8')
    output = cp.zeros((T, 6, G)); kernel = cp.RawKernel(ARRIVAL_CODE, 'apply', options=('--fmad=false',))
    for j, a in enumerate(matrices):
        parts = (cp.asarray(a.indptr, dtype='i4'), cp.asarray(a.indices, dtype='i4'), cp.asarray(a.data))
        kernel((G, T), (128,), (*parts, spikes, sizes, output, np.int32(P), np.int32(G), np.int32(j)))
        cp.cuda.get_current_stream().synchronize()
    arr = output.get()[:, :4].copy(); checks = []
    for k in [0, 1, 357, 358, 999, T-1]:
        slots = k - np.arange(1, matrices[0].shape[1]//P+1)
        history = np.zeros((len(slots), P)); ok = slots >= 0
        history[ok] = z['spikes'][slots[ok]] / geo['group_size'] / .1
        oracle = np.array([a @ history.ravel() for a in matrices])
        error = float(abs(oracle-arr[k]).max()); assert error < 1e-8
        checks.append(dict(step=k, error=error))
    del spikes, sizes, output
    p = read(OPS/'prepared.json')['params']; E = geo['population'][selected] == 0
    pars = np.c_[np.where(E, p['tau_m_E'], p['tau_m_I']),
                 np.where(E, p['tau_ref_E'], p['tau_ref_I'])/.1,
                 geo['threshold_mv'][selected], E,
                 np.where(E, p['J_ext_E'], p['J_ext_I']), geo['group_size'][selected]]
    c = np.array([np.exp(-.1/p[n]) for n in ['tau_r_AMPA', 'tau_d_AMPA', 'tau_r_GABA', 'tau_d_GABA']]
                 + [p['tau_r_AMPA'], p['tau_r_GABA'], p['V_reset']])
    ids = initial['selected_group_index']; size = np.bincount(ids, minlength=G)
    depth = len(initial['ring_sE']); order = (int(initial['step'])+np.arange(depth)) % depth
    pending = np.stack([initial['ring_sE'][order], initial['ring_sI'][order]])
    initial_means = np.array([initial['state'][ids == g].mean(0) for g in range(G)])
    pending_means = np.array([[np.bincount(ids, weights=pending[j,k], minlength=G)/size
                               for j in range(2)] for k in range(depth)])
    names = z['moment_names'].tolist(); actual = z['moments'][:, [names.index('IE'), names.index('II')]]
    arr, projected = mean_path(arr, z['external_rate_per_ms'], pars, c, initial_means, pending_means, actual, False)
    matched, measured = mean_path(arr, z['external_rate_per_ms'], pars, c, initial_means, pending_means, actual, True)
    error = float(abs(measured-actual).max()); assert error < 1e-9
    members = np.array([np.flatnonzero(ids == g)[np.arange(8192) % size[g]] for g in range(G)], dtype='i4')
    write(OUT/'input_qa.json', dict(status='PASS', operator_checks=checks,
          measured_mean_inversion_error=error, pending_history_ms=depth*.1,
          initial_cells=len(ids), quadrature='Balanced full joint states, paired original cell pending arrivals; group mean thresholds unchanged.'))
    np.savez_compressed(OUT/'input_summary.npz', selected_groups=selected, pars=pars,
          projected_current_means=projected, native_counts=z['spikes'][:, selected],
          native_moments=z['moments'], moment_names=z['moment_names'], time_ms=z['time_ms'],
          global_R_and_s=z['global_R_and_s'], pending_means=pending_means, initial_means=initial_means)
    return z, initial, pars, c, members, pending, {'projected_full': arr, 'measured_full': matched}


def main(device, wait):
    OUT.mkdir(exist_ok=True); assert not (OUT/'contract.json').exists(), 'No silent restart'
    write(OUT/'contract.json', dict(status='REGISTERED_BEFORE_SOURCE_GATE_AND_SCORING', created_epoch=time.time(),
        question='Does the K9 high-state surrounding-rate deficit persist under native upstream counts and global feedback?',
        source=str(SOURCE), interval_s=[42.,44.], selected='Same 16 geometric targets as preceding diagnostics.',
        replicas=8192, numerical_seeds=SEEDS, conditions=4,
        design='Projected versus measured group-current means, each with two numerical streams; full squared-weight Gaussian variance unchanged. Prescribed native upstream counts, external drive, R/G. Hold each copied cell Z/K after each update; V/current/ref/M evolve.',
        hypothesis='If measured-current forcing repairs a projected deficit, group-current projection matters. If both remain deficient, local input statistics/response approximation matters. If both match, free recurrent coupling or unrepresented regions remain the main gap.',
        evaluation='Fixed 20ms count bins and elapsed0-.5/.5-2s windows, selected surround/core/I separately; current means and spread, held fields and numerical stream differences. No fitted acceptance rule.',
        bounds='Exactly four local conditions, no new network parameters or native seeds. Gate on observer_audit PASS; no automatic extension.',
        engine_sha256=sha(physical.__file__), producer_sha256=sha(__file__), formal_bifurcation_allowed=False))
    while not (SOURCE/'observer_audit.json').exists():
        if not wait:
            raise RuntimeError('Native observer not yet complete')
        write(OUT/'progress.json', dict(status='WAITING_NATIVE_OBSERVER_GATE', pid=os.getpid(), updated_epoch=time.time()))
        time.sleep(20)
    assert read(SOURCE/'observer_audit.json')['status'] == 'PASS'
    import cupy as cp
    cp.cuda.Device(device).use(); started = time.time()
    z, initial, pars, c, members, pending, inputs = prepare(cp)
    module = cp.RawModule(code=physical.CODE+CODE, options=('--fmad=false',),
                          name_expressions=['rng_bytes','init_rng','exit_steps'])
    a = LocalExit(cp,module,inputs['projected_full'],z,initial,pars,c,members[:,:64],pending,928739)
    b = LocalExit(cp,module,inputs['projected_full'],z,initial,pars,c,members[:,:64],pending,928739)
    held = initial['state'][members[:,:64]][:,:,[6,7]].copy()
    a.advance(0,400); b.advance(0,357); b.advance(357,43)
    equal = {key:np.array_equal(getattr(a,key).get(),getattr(b,key).get()) for key in ['state','ref','rng','counts']}
    assert all(equal.values()) and np.array_equal(a.state.get()[:,:,[6,7]],held)
    write(OUT/'chunk_qa.json',dict(status='PASS',equal=equal,held_fields_bitwise=True)); del a,b
    completed = []
    for seed in SEEDS:
        for name, arr in inputs.items():
            job = f'{name}_num{seed}'
            e = LocalExit(cp,module,arr,z,initial,pars,c,members,pending,seed); rates=[]; moments=[]
            for tick in range(0,T,100):
                e.counts.fill(0); e.advance(tick,100); rates.append(e.counts.sum(1).get().T/e.R*1000.)
                x=e.state
                moments.append(cp.stack([x[:,:,6].mean(1),x[:,:,7].mean(1),x[:,:,5].mean(1),
                     x[:,:,2].mean(1),x[:,:,4].mean(1),x[:,:,2].std(1),x[:,:,4].std(1)],axis=1).get())
                if (tick+100)%5000==0:
                    write(OUT/'progress.json',dict(status='RUNNING',job=job,pid=os.getpid(),
                          time_ms=42000+(tick+100)*.1,completed=completed,updated_epoch=time.time()))
            assert np.array_equal(e.state.get()[:,:,[6,7]],initial['state'][members][:,:,[6,7]])
            np.savez_compressed(OUT/f'{job}.npz',rate_Hz=np.concatenate(rates),moments=np.array(moments),
                  moment_names=['Z','K','M','IE','II','IEstd','IIstd'],time_ms=42000+np.arange(10,2001,10.),
                  final_state=e.state.get(),final_ref=e.ref.get(),selected_groups=z['selected_groups'])
            completed.append(job); print('CLAMPED LOCAL COMPLETE',job,flush=True)
            del e; cp.get_default_memory_pool().free_all_blocks()
    assert sha(physical.__file__)==read(OUT/'contract.json')['engine_sha256']
    result=dict(status='COMPLETE',completed=completed,elapsed_s=time.time()-started,
                held_fields_bitwise=True,engine_unchanged=True,formal_bifurcation_allowed=False)
    write(OUT/'result.json',result);write(OUT/'progress.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);p.add_argument('--wait',action='store_true')
    args=p.parse_args();main(args.device,args.wait)
