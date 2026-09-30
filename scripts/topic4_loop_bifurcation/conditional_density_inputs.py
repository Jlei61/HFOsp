#!/usr/bin/env python3
"""Localize the present density closure using already observed native inputs.

This is teacher forcing, never an autonomous network or a bifurcation test.
The frozen density_spatial.native_cell is used without editing its equations.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import time
import numpy as np
from scipy import sparse
from numba import njit
from campaign import ROOT, REPO, NATIVE, read, write, sha
import density_spatial as physical

OLD = REPO / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'
SOURCE = OLD / 'native_early_surround_inputs'
OPS = REPO / 'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators/g40'
OUT = ROOT / 'conditional_density_inputs'

ARRIVAL_CODE = r'''
extern "C" __global__ void apply(const int* ptr,const int* col,const double* val,
 const unsigned short* counts,const double* sizes,double* out,int P,int G,int channel){
 int g=blockIdx.x,k=blockIdx.y,lane=threadIdx.x;double value=0.;
 for(int j=ptr[g]+lane;j<ptr[g+1];j+=blockDim.x){
  int source=col[j]%P,lag=col[j]/P+1,slot=k-lag;
  if(slot>=0)value+=val[j]*counts[(long long)slot*P+source]/(.1*sizes[source]);
 }
 __shared__ double buffer[128];buffer[lane]=value;__syncthreads();
 for(int n=64;n>0;n/=2){if(lane<n)buffer[lane]+=buffer[lane+n];__syncthreads();}
 if(lane==0)out[((long long)k*6+channel)*G+g]=buffer[0];
}'''

LOCAL_CODE = r'''
extern "C" __global__ void local_steps(double* state,int* refs,unsigned char* memory,
 const double* pars,const double* constants,const double* arrivals,const double* drive,
 int* counts,int G,int R,int start,int steps,int stride){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=G*R)return;
 int g=id/R,ref=refs[id];double x[8],global[2]={0.,0.};
 for(int j=0;j<8;j++)x[j]=state[id*8+j];
 curandStatePhilox4_32_10_t* all=(curandStatePhilox4_32_10_t*)memory;
 curandStatePhilox4_32_10_t rng=all[id];
 for(int k=start;k<start+steps;k++){
  float4 n=curand_normal4(&rng);double a[4];
  for(int j=0;j<4;j++)a[j]=arrivals[((long long)k*4+j)*G+g];
  bool sp=native_cell(x,ref,pars+6*g,constants,a,drive[(long long)k*G+g],global,0.,n.x,n.z);
  if(sp)counts[(long long)id*stride+(k%100)/10]++;
 }
 refs[id]=ref;all[id]=rng;for(int j=0;j<8;j++)state[id*8+j]=x[j];
}
'''


@njit
def filtered_means(arr, drive, pars, constants):
    T, _, G = arr.shape
    out = np.zeros((T, 2, G)); syn = np.zeros((2, G)); current = np.zeros((2, G))
    for k in range(T):
        for g in range(G):
            for j in range(2):
                jump = arr[k, j, g] + (pars[g, 4] * drive[k, g] if j == 0 else 0.)
                syn[j, g] = syn[j, g] * constants[2*j] + pars[g, 0]/constants[4+j]*jump*.1
                current[j, g] = syn[j, g] + (current[j, g]-syn[j, g])*constants[2*j+1]
                out[k, j, g] = current[j, g]
    return out


@njit
def substitute_means(arr, drive, observed, pars, constants):
    """Invert only the deterministic discrete filter, preserving covariance.

    The signed resulting drift is a diagnostic input, not a spike intensity.
    """
    out = arr.copy(); previous = np.zeros_like(observed[0]); syn = np.zeros_like(previous)
    for k in range(len(arr)):
        for g in range(len(pars)):
            for j in range(2):
                b = constants[2*j+1]
                new_syn = (observed[k, j, g]-b*previous[j, g])/(1-b)
                total = (new_syn-syn[j, g]*constants[2*j])*constants[4+j]/pars[g, 0]/.1
                out[k, j, g] = total-(pars[g, 4]*drive[k, g] if j == 0 else 0.)
                previous[j, g] = observed[k, j, g]; syn[j, g] = new_syn
    return out


def prepare(cp):
    assert read(SOURCE/'replay_audit.json')['status'] == 'PASS'
    assert read(SOURCE/'independent_response_audit.json')['status'] == 'PASS'
    prep = read(OPS/'prepared.json'); assert prep['graph_identity'] == read(NATIVE/'protocol.json')['identity']
    geo = dict(np.load(SOURCE/'membership.npz')); selected = geo['selected_groups']; G = len(selected)
    actual_geo = dict(np.load(OPS/'geometry.npz'))
    for key in actual_geo: assert np.array_equal(actual_geo[key], geo[key]), key
    pieces = {k: [] for k in ['spikes', 'moments', 'external_rate_per_ms', 'time_ms']}
    for file in sorted((SOURCE/'inputs').glob('*.npz')):
        with np.load(file) as z:
            for key in pieces: pieces[key].append(z[key])
            names = z['moment_names'].tolist()
    counts, moments, drive, times = [np.concatenate(pieces[k]) for k in pieces]
    assert len(times) == 30000 and np.allclose(times, np.arange(30000)*.1, atol=1e-9, rtol=0)
    P = len(geo['group_size']); matrices = []
    for name in ['mean_ampa', 'mean_gaba', 'variance_ampa', 'variance_gaba']:
        matrices.append(sparse.load_npz(OPS/f'{name}.npz')[selected].tocsr())
    for name in ['ampa', 'gaba']:
        matrices.append(sparse.load_npz(OLD/f'physical_delay_variance_split/physical_private_{name}.npz')[selected].tocsr())
    assert all(x.data.min() >= 0 for x in matrices)
    kernel = cp.RawKernel(ARRIVAL_CODE, 'apply', options=('--fmad=false',))
    arrived = cp.empty((len(times), 6, G)); spikes = cp.asarray(counts); sizes = cp.asarray(geo['group_size'], dtype='f8')
    for j, matrix in enumerate(matrices):
        args = (cp.asarray(matrix.indptr, dtype='i4'), cp.asarray(matrix.indices, dtype='i4'), cp.asarray(matrix.data))
        kernel((G, len(times)), (128,), (*args, spikes, sizes, arrived, np.int32(P), np.int32(G), np.int32(j)))
        cp.cuda.get_current_stream().synchronize()
    arr = arrived.get(); checks = []
    for tick in [0, 1, 51, 4999, 17821, 29999]:
        slots = tick-np.arange(1, matrices[0].shape[1]//P+1)
        history = np.zeros((len(slots), P)); ok = slots >= 0
        history[ok] = counts[slots[ok]] / geo['group_size'] / .1
        reference = np.array([a @ history.ravel() for a in matrices])
        error = float(abs(reference-arr[tick]).max()); assert error < 1e-8
        checks.append(dict(step=tick, max_error=error))
    del arrived, spikes, sizes
    E = geo['population'][selected] == 0; p = prep['params']
    pars = np.c_[np.where(E, p['tau_m_E'], p['tau_m_I']), np.where(E, p['tau_ref_E'], p['tau_ref_I'])/.1,
                 geo['threshold_mv'][selected], E, np.where(E, p['J_ext_E'], p['J_ext_I']), geo['group_size'][selected]]
    constants = np.array([np.exp(-.1/p[n]) for n in ['tau_r_AMPA','tau_d_AMPA','tau_r_GABA','tau_d_GABA']]
                         + [p['tau_r_AMPA'], p['tau_r_GABA'], p['V_reset']])
    observed = moments[:, [names.index('ampa'), names.index('gaba')]]
    primary = arr[:, :4].copy(); matched = substitute_means(primary, drive, observed, pars, constants)
    error = float(abs(filtered_means(matched, drive, pars, constants)-observed).max()); assert error < 1e-9
    projected = filtered_means(primary, drive, pars, constants)
    # Independent agreement with the previously reconstructed native-step means.
    with np.load(SOURCE/'projected_inputs.npz') as z:
        old_error = float(abs(projected-z['filtered_moments'][:, 2:4]).max()); assert old_error < 1e-8
    inputs = {}
    for mean, base in [('projected', primary), ('measured', matched)]:
        for variance, indices in [('full', [2, 3]), ('private_sensitivity', [4, 5])]:
            value = base.copy(); value[:, 2:4] = arr[:, indices]
            inputs[f'{mean}_{variance}'] = value
    np.savez_compressed(OUT/'native_inputs.npz', time_ms=times, native_counts=counts[:, selected],
                        native_moments=moments, moment_names=names, drive_per_ms=drive,
                        selected_groups=selected, pars=pars, constants=constants, projected_means=projected,
                        **inputs)
    write(OUT/'input_qa.json', dict(status='PASS', delayed_operator_checks=checks,
        native_discrete_means_match_old_reconstruction=old_error, measured_mean_inversion_max_error=error,
        source_replay='PASS_SIX_NATIVE_CHUNKS_AND_ORIGINAL_CELL_FIELDS',
        private_variance='Only old stationary Poisson split sensitivity; not exact conditional nonstationary covariance.',
        no_autonomous_validation=True))
    return inputs, drive, pars, constants, selected


class Local:
    def __init__(self, cp, module, arr, drive, pars, constants, R, seed):
        self.cp, self.module, self.G, self.R = cp, module, len(pars), R
        self.arr, self.drive, self.pars, self.constants = [cp.asarray(a) for a in [arr, drive, pars, constants]]
        self.state = cp.zeros((self.G, R, 8)); self.state[:, :, 0] = constants[6]; self.state[:, :, 6] = 1.
        self.ref = cp.zeros((self.G, R), dtype='i4'); self.counts = cp.zeros((self.G, R, 10), dtype='i4')
        size = cp.zeros(1, dtype='i4'); module.get_function('rng_bytes')((1,), (1,), (size,))
        self.rng = cp.empty(self.G*R*int(size.get()[0]), dtype='u1')
        module.get_function('init_rng')(((self.G*R+127)//128,), (128,),
            (self.rng, np.int32(self.G*R), np.uint64(seed), np.int32(R)))

    def advance(self, start, steps):
        self.module.get_function('local_steps')(((self.G*self.R+127)//128,), (128,),
            (self.state, self.ref, self.rng, self.pars, self.constants, self.arr, self.drive, self.counts,
             np.int32(self.G), np.int32(self.R), np.int32(start), np.int32(steps), np.int32(10)))


def main(device):
    OUT.mkdir(exist_ok=True); assert not (OUT/'contract.json').exists(), 'No unregistered restart'
    write(OUT/'contract.json', dict(status='REGISTERED_BEFORE_SCORING', created_epoch=time.time(),
        question='Under the same observed native incoming activity, does the current distribution-retaining local closure reproduce early firing and Z depletion, and does physical input projection explain residual error?',
        design='16 previously geometrically selected g40 groups, 0-3s source seed9108401; primary0.5-3s fixed50ms bins. Four arms x two numerical streams,8192particles/group. All use the unchanged density_spatial native_cell, ownV/current/ref/Z/M, G=K=0.',
        arms=['projected_full', 'measured_full', 'projected_private_sensitivity', 'measured_private_sensitivity'],
        mean_contrast='Only the deterministic discrete current-filter mean is changed to actual native IE/II; covariance,localdynamics,numericalstream remainpaired. Inverted signed drift is diagnostic and is not a physically realizable arrival intensity.',
        variance_contrast='Full squared physical weights are the present autonomous closure. The old stationary private split is sensitivity only because supplying native counts adds common fluctuations. Its nonstationary accuracy is unproved; do not silently replace autonomous variance.',
        source='Existing exact native0-3s replay; no newSNN and no networkparameter fitting. Group threshold mean; observed membermean externaldrive at native0.1ms. Numerical replicates are not native seed replication.',
        decision='Local mismatch larger than numerical-stream difference locates closure/input error. Agreement remains conditional, does not certify autonomous propagation, G/Kresponse, derivatives or bifurcation. Do not calibrate onset or relax original spatial gates.',
        bounded_budget='Exactly8localconditions; one GPU process. No automatic newgroups,seeds,grid,networkscan orfitting.',
        producer_sha256=sha(__file__), engine_sha256=sha(physical.__file__),
        selected_groups=read(SOURCE/'contract.json')['selected_groups'], numerical_seeds=[927641,927642],
        formal_bifurcation_allowed=False))
    import cupy as cp
    cp.cuda.Device(device).use(); start = time.time()
    inputs, drive, pars, constants, selected = prepare(cp)
    module = cp.RawModule(code=physical.CODE+LOCAL_CODE, options=('--fmad=false',),
                          name_expressions=['rng_bytes','init_rng','local_steps'])
    # Continuing a chunk must preserve complete particle and RNG state exactly.
    a = Local(cp,module,inputs['projected_full'],drive,pars,constants,64,927649)
    b = Local(cp,module,inputs['projected_full'],drive,pars,constants,64,927649)
    a.advance(0,100); b.advance(0,37); b.advance(37,63)
    identity = {k: np.array_equal(getattr(a,k).get(),getattr(b,k).get()) for k in ['state','ref','rng','counts']}
    assert all(identity.values()); write(OUT/'chunk_qa.json',dict(status='PASS',split37_63_equals100=identity))
    del a,b
    completed = []
    for seed in [927641,927642]:
        for name, arr in inputs.items():
            job = f'{name}_num{seed}'; e = Local(cp,module,arr,drive,pars,constants,8192,seed)
            firing = []; moments = []
            for tick in range(0,30000,100):
                e.counts.fill(0); e.advance(tick,100)
                # 1ms rates and end-of-step distribution moments, never resampled cells.
                firing.append(e.counts.sum(1).get().T/e.R*1000.)
                x = e.state
                moments.append(cp.stack([x[:,:,6].mean(1), x[:,:,5].mean(1), x[:,:,2].mean(1),
                    x[:,:,4].mean(1), x[:,:,2].var(1), x[:,:,4].var(1), x[:,:,0].mean(1)],axis=1).get())
                if (tick+100)%5000 == 0:
                    write(OUT/'progress.json',dict(status='RUNNING',pid=os.getpid(),job=job,time_ms=(tick+100)*.1,
                                                  completed=completed,elapsed_s=time.time()-start))
            np.savez_compressed(OUT/f'{job}.npz',rate_Hz=np.concatenate(firing),moments=np.array(moments),
                moment_names=['Z','M','IE','II','varIE','varII','V'],moments_time_ms=np.arange(10,3001,10.),
                selected_groups=selected,particle_count=e.R,numerical_seed=seed,final_state=e.state.get(),final_ref=e.ref.get())
            completed.append(job); print('LOCAL DENSITY COMPLETE',job,'elapsed',time.time()-start,flush=True)
            del e; cp.get_default_memory_pool().free_all_blocks()
    assert sha(physical.__file__)==read(OUT/'contract.json')['engine_sha256']
    result=dict(status='COMPLETE',completed=completed,elapsed_s=time.time()-start,engine_unchanged=True,
                native_correspondence_certified=False,formal_bifurcation_allowed=False)
    write(OUT/'result.json',result); write(OUT/'progress.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--device',type=int,default=1)
    main(parser.parse_args().device)
