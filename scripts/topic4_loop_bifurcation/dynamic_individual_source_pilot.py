#!/usr/bin/env python3
"""Bounded physical-time test of individual versus grouped recurrent sources.

This is a density approximation with independent Gaussian input increments.
Numerical replicas are not independent native-network seeds. No root/stability
claim follows from this development-anchor comparison.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import pickle
import shutil
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from campaign import ROOT, read, write, sha
from conditional_density_inputs import OPS
import density_spatial as physical
import target_density_exit as old
import observe_high_history_spectra as native
from high_history_spectral_value import OUT as PARAMETERS, MATCHED
from prepare_dynamic_source_operators import OUT as OPERATORS

OUT = ROOT/'dynamic_individual_source_pilot'
MODES = ['group_source', 'individual_source']
DURATION_MS = 1000
REPLICAS = 64
CODE = r'''
extern "C" __global__ void fixed_particles(double* state,int* refs,unsigned char* memory,
 const double* pars,const double* c,const double* arr,const double* nu,
 const int* clock,const double* global,const double* pending,int* spikes,int N,int R,int depth){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=N*R)return;
 int i=id/R,tick=clock[0],ref=refs[id];
 curandStatePhilox4_32_10_t* all=(curandStatePhilox4_32_10_t*)memory;
 curandStatePhilox4_32_10_t rng=all[id];float4 normal=curand_normal4(&rng);all[id]=rng;
 double x[8],a[4];for(int j=0;j<8;j++)x[j]=state[id*8+j];for(int j=0;j<4;j++)a[j]=arr[j*N+i];
 spikes[id]=held_cell(x,ref,pars+6*i,c,a,nu[i],global,30.,normal.x,normal.z,pending,tick,depth,i,N);
 refs[id]=ref;for(int j=0;j<8;j++)state[id*8+j]=x[j];
}
extern "C" __global__ void source_collect(const int* spikes,double* history,double* accumulator,
 double* output,const int* clock,int N,int R,int depth){
 int i=blockIdx.x,lane=threadIdx.x,tick=clock[0];int n=0;
 for(int k=lane;k<R;k+=128)n+=spikes[i*R+k];
 __shared__ int buf[128];buf[lane]=n;__syncthreads();
 for(int s=64;s>0;s/=2){if(lane<s)buf[lane]+=buf[lane+s];__syncthreads();}
 if(lane==0){double count=(double)buf[0]/R;history[(long long)(tick%depth)*N+i]=count/.1;
  accumulator[i]+=count;
  if((tick+1)%10==0){int row=((tick+1)/10-1)%10;output[(long long)row*N+i]=accumulator[i]*1000.;accumulator[i]=0.;}}
}
'''


class DynamicNetwork(old.TargetNetwork):
    def __init__(self, source_mode, replicas, device):
        self.ready = False
        super().__init__('individual', replicas, device)
        cp = self.cp
        self.source_mode = source_mode
        self.raw = dict(np.load(PARAMETERS/'parameters.npz'))
        with native.INITIAL.open('rb') as f:
            saved = pickle.load(f)
        assert saved['identity'] == self.prep['graph_identity']
        s = saved['engine'];assert s['step'] == 720000
        k = np.zeros(self.N);k[:32000] = s['termination_mechanism']['sahp_g']
        self.native_initial = np.stack([s['V'], s['s_E'], s['I_E'], s['s_I'], s['I_I'],
            s['slow']['m'], s['slow']['z'], k], axis=1)
        assert np.array_equal(self.native_initial[:, 6], self.raw['Z'])
        assert np.array_equal(k, self.raw['K'])
        self.native_ref = s['ref']
        self.initial_state = cp.asarray(np.repeat(self.native_initial[:, None, :], replicas, axis=1))
        self.initial_ref = cp.asarray(np.repeat(self.native_ref[:, None], replicas, axis=1), dtype='i4')
        self.initial_global = np.array([s['termination_mechanism']['r_global'], s['global_feedback_response']['global_state']])
        order = (s['step']+np.arange(self.depth)) % self.depth
        self.pending_cpu = np.stack([s['ring_sE'][order], s['ring_sI'][order]])
        self.pending = cp.asarray(self.pending_cpu)
        # Both arms retain exact cell thresholds and external means; only source
        # identity in recurrent feedback changes. Do not project nu to groups.
        self.pars_cpu[:, 2] = self.raw['theta'];self.pars = cp.asarray(self.pars_cpu)
        assert np.array_equal(self.pars_cpu[:, 0], self.raw['tm'])
        assert np.array_equal(self.pars_cpu[:, 1], self.raw['ref_steps'])
        # The spectral sampler stores the actual jump of the synaptic rise
        # state; native_cell stores J and multiplies by tau_m/tau_r itself.
        external_jump = self.pars_cpu[:, 4]*self.pars_cpu[:, 0]/self.constants_cpu[4]
        assert np.max(abs(external_jump-self.raw['jump_external'])) < 1e-12
        self.nu = cp.asarray(self.raw['nu_per_ms'])
        self.drive = None;self.drive_cpu = None
        if source_mode == 'individual_source':
            self.ops = [];self.ops_cpu = []
            for kind in ['ampa', 'gaba']:
                a = sparse.load_npz(OPERATORS/f'mean_{kind}.npz').tocsr()
                q = sparse.load_npz(OPERATORS/f'variance_{kind}.npz').tocsr()
                assert np.array_equal(a.indices, q.indices) and np.array_equal(a.indptr, q.indptr)
                self.ops_cpu.append((a, q))
                self.ops.extend([cp.asarray(a.indptr, dtype='i4'), cp.asarray(a.indices, dtype='i4'), cp.asarray(a.data), cp.asarray(q.data)])
        self.source_history = cp.zeros((self.depth, self.N))
        self.source_accumulator = cp.zeros(self.N);self.source_output = cp.zeros((10, self.N))
        self.extra_module = cp.RawModule(code=physical.CODE+old.EXTRA+old.CLAMP+CODE,
            options=('--fmad=false',), name_expressions=['fixed_particles', 'source_collect'])
        for name in ['fixed_particles', 'source_collect']:
            self.k[name] = self.extra_module.get_function(name)
        self.ready = True;self.reset();cp.get_default_memory_pool().free_all_blocks()

    def reset(self):
        super().reset()
        if self.ready:
            for x in [self.source_history, self.source_accumulator, self.source_output]:x.fill(0)

    def arrivals(self):
        p = self.N if self.source_mode == 'individual_source' else self.P
        history = self.source_history if self.source_mode == 'individual_source' else self.history
        self.k['target_delayed']((self.N,), (128,), (*self.ops, history, self.arr, self.clock,
            np.int32(self.depth), np.int32(p), np.int32(self.N)))

    def step(self):
        self.arrivals()
        self.k['fixed_particles'](((self.N*self.R+127)//128,), (128,),
            (self.state, self.ref, self.rng, self.pars, self.constants, self.arr, self.nu,
             self.clock, self.global_state, self.pending, self.spikes,
             np.int32(self.N), np.int32(self.R), np.int32(self.depth)))
        self.k['source_collect']((self.N,), (128,), (self.spikes, self.source_history,
            self.source_accumulator, self.source_output, self.clock,
            np.int32(self.N), np.int32(self.R), np.int32(self.depth)))
        self.k['target_collect']((self.P,), (128,),
            (self.state, self.spikes, self.ptr, self.order, self.global_state, self.history,
             self.rate, self.accumulator, self.output, self.clock,
             np.int32(self.P), np.int32(self.R), np.int32(self.depth)))
        self.k['global_step']((1,), (128,), (self.rate, self.pars_group, self.global_state, self.clock, np.int32(self.P)))
        self.k['observe_global']((1,), (1,), (self.global_state, self.clock, self.global_output))


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(OPERATORS/'result.json')['status'] == 'COMPLETE_OPERATORS_ONLY'
    assert read(MATCHED/'result.json')['status'] == 'COMPLETE_TWO_NATIVE_CONSTANT_BACKGROUND_CONTROLS'
    dependencies = {str(Path(p)): sha(p) for p in [__file__, physical.__file__, old.__file__,
        Path(old.__file__).with_name('coupled_density_exit.py'), Path(old.__file__).with_name('exit_branch_density.py')]}
    write(OUT/'contract.json', dict(status='REGISTERED_BEFORE_TWO_PHYSICAL_TIME_PILOTS', created_epoch=time.time(),
        question='Can retaining individual source identity and full physical delay preserve the native high-history state without imposing its period?',
        design='Exactly two one-second physical-time density trajectories from the same full72s native high-history state, at held actual exit Z/K fields. Same exact individual thresholds, fixed per-cell external means,64 numerical replicas and RNG streams. Only group-averaged versus individual source feedback changes. M, synapses, membrane, refractory state and R/G evolve. No future neural forcing or phase-period prescription.',
        limitations='Independent Gaussian recurrent/external increments remain an explicit approximation, with finite numerical replica error. Exact source operators alone do not validate this closure. One development high point, not independent native seeds, global correspondence, equilibrium, stability or bifurcation.',
        checks='Complete initial physical state including pending pulses; exact Z/K/threshold/nu; CPU delayed-arrival projection; per-cell-to-group spike conservation; eager/captured bitwise equivalence; fixed-input new kernel versus established kernel on identical uniform drive.',
        decision='Compare matched native first1s and time-resolved spatial/core/G behavior. Native correspondence guard: tail.8-1s fieldRMS<=10Hz, eachcore rate error<=10Hz, causalR remains below200 and negligibleGraw<0.1. These are relevance guards, not final certification. Stop after fixed pair regardless of outcome; no automatic parameter/period/root search.',
        modes=MODES, duration_ms=DURATION_MS, replicas=REPLICAS,
        native_initial=str(native.INITIAL), native_initial_sha256=sha(native.INITIAL),
        matched_native=str(MATCHED/'runs/high_history_constant_background'),
        parameters_sha256=sha(PARAMETERS/'parameters.npz'), dependencies=dependencies,
        formal_bifurcation_allowed=False))
    shutil.copy2(__file__, OUT/'producer.py')


def validate(e):
    cp = e.cp;rng = np.random.default_rng(930181);groups = e.geo['cell_group']
    history = rng.uniform(0, .5, size=(e.depth, e.N if e.source_mode == 'individual_source' else e.P))
    h = e.source_history if e.source_mode == 'individual_source' else e.history
    h[:] = cp.asarray(history);e.clock[0] = e.depth-1;e.arrivals()
    x = history[(e.depth-1-np.arange(1, e.depth)) % e.depth].ravel()
    expected = np.array([e.ops_cpu[0][0]@x, e.ops_cpu[1][0]@x, e.ops_cpu[0][1]@x, e.ops_cpu[1][1]@x])
    error = float(abs(expected-e.arr.get()).max());assert error < 1e-8
    # Test the changed per-cell-drive kernel against the existing physical
    # kernel with an exactly common drive and identical random memory.
    e.reset();rstate = e.rng.copy();e.arr[:] = cp.asarray(rng.uniform(0, 2, (4, e.N)))
    uniform = cp.full(e.N, .2)
    e.k['fixed_particles'](((e.N*e.R+127)//128,), (128,),
        (e.state, e.ref, e.rng, e.pars, e.constants, e.arr, uniform, e.clock, e.global_state,
         e.pending, e.spikes, np.int32(e.N), np.int32(e.R), np.int32(e.depth)))
    fixed = {key: getattr(e, key).get() for key in ['state', 'ref', 'spikes', 'rng']}
    e.state[:] = e.initial_state;e.ref[:] = e.initial_ref;e.rng[:] = rstate
    e.k['target_particles'](((e.N*e.R+127)//128,), (128,),
        (e.state, e.ref, e.rng, e.pars, e.constants, e.arr, cp.full((1, e.P), .2), e.group,
         e.clock, e.global_state, e.pending, e.spikes,
         np.int32(e.N), np.int32(e.R), np.int32(e.P), np.int32(e.depth), np.int32(1)))
    assert all(np.array_equal(a, getattr(e, key).get()) for key, a in fixed.items())
    e.reset()
    assert np.array_equal(e.state.get()[:, 0], e.native_initial)
    assert np.array_equal(e.ref.get()[:, 0], e.native_ref)
    for _ in range(100):e.step()
    keys = ['state', 'ref', 'rng', 'history', 'source_history', 'clock', 'global_state',
        'output', 'global_output', 'accumulator', 'source_accumulator', 'source_output']
    expected = {key: getattr(e, key).get() for key in keys}
    source = e.source_output.get();group = e.output.get()[:, 0]
    projected = np.stack([np.bincount(groups, weights=a, minlength=e.P)/e.sizes for a in source])
    count_error = float(abs(projected-group).max());assert count_error < 1e-10
    e.graph();e.chunk()
    equal = {key: np.array_equal(a, getattr(e, key).get()) for key, a in expected.items()}
    assert all(equal.values())
    e.reset()
    return dict(status='PASS', delayed_arrival_max_error=error, source_group_count_max_error=count_error,
        fixed_drive_kernel_bitwise=True, complete_initial_state_exact=True, captured_bitwise=equal,
        exact_individual_thresholds_and_external_means=True)


def run(device):
    c = read(OUT/'contract.json')
    assert all(sha(path) == digest for path, digest in c['dependencies'].items())
    assert c['native_initial_sha256'] == sha(native.INITIAL)
    assert c['parameters_sha256'] == sha(PARAMETERS/'parameters.npz')
    assert not (OUT/'supervisor.json').exists()
    write(OUT/'supervisor.json', dict(status='RUNNING_FIXED_PAIR', pid=os.getpid(), device=device, updated_epoch=time.time()))
    for mode in MODES:
        folder = OUT/mode;folder.mkdir();started = time.time()
        write(folder/'progress.json', dict(status='INITIALIZING', pid=os.getpid(), updated_epoch=time.time()))
        e = DynamicNetwork(mode, REPLICAS, device)
        qa = validate(e);write(folder/'implementation_qa.json', qa)
        print('PHYSICAL SOURCE QA PASS', mode, flush=True)
        e.graph();groups = [];globals_ = [];cells = []
        for offset in range(0, DURATION_MS, 10):
            groups.append(e.chunk().astype('f4'));globals_.append(e.global_output.get());cells.append(e.source_output.get().astype('f4'))
            if (offset+10) % 100 == 0:
                write(folder/'progress.json', dict(status='RUNNING', pid=os.getpid(), elapsed_simulation_ms=offset+10,
                    elapsed_wall_s=time.time()-started, updated_epoch=time.time()))
                print('PHYSICAL SOURCE', mode, offset+10, flush=True)
        value, global_, source = np.concatenate(groups), np.concatenate(globals_), np.concatenate(cells)
        assert int(e.clock.get()[0]) == DURATION_MS*10
        assert np.isfinite(value).all() and np.isfinite(global_).all() and np.isfinite(source).all()
        assert np.array_equal(e.state.get()[:, :, 6:8], e.initial_state.get()[:, :, 6:8])
        np.savez_compressed(folder/'trajectory.npz', elapsed_time_ms=np.arange(1, DURATION_MS+1),
            group_output=value, global_R_Hz=global_[:, 0], global_s=global_[:, 1], cell_rate_Hz=source)
        np.savez_compressed(folder/'final_state.npz', state=e.state.get(), ref=e.ref.get(), rng=e.rng.get(),
            source_history=e.source_history.get(), history=e.history.get(), clock=e.clock.get(), global_state=e.global_state.get())
        result = dict(status='COMPLETE_ONE_SECOND', mode=mode, duration_ms=DURATION_MS,
            elapsed_wall_s=time.time()-started, held_fields_bitwise=True, replicas=REPLICAS,
            formal_bifurcation_allowed=False)
        write(folder/'result.json', result);write(folder/'progress.json', result)
        cp = e.cp;del e, groups, globals_, cells, value, global_, source;cp.get_default_memory_pool().free_all_blocks()
    write(OUT/'result.json', dict(status='COMPLETE_TWO_PHYSICAL_TIME_PILOTS_ANALYSIS_PENDING', modes=MODES,
        formal_bifurcation_allowed=False))
    write(OUT/'supervisor.json', dict(status='COMPLETE', updated_epoch=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('command', choices=['prepare', 'run'])
    parser.add_argument('--device', type=int, default=1);args = parser.parse_args()
    if args.command == 'prepare':prepare()
    else:
        try:run(args.device)
        except Exception:
            write(OUT/'supervisor.json', dict(status='FAILED', pid=os.getpid(), updated_epoch=time.time()))
            raise
