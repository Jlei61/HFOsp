#!/usr/bin/env python3
"""Correspondence screen at four existing actual-exit-field native conditions.

No continuation or stability claims. Only Z/K are clamped, exactly as in the
four native probes; all fast states, recurrent activity, M, R and G evolve.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import pickle
import time
import numpy as np
from campaign import ROOT, read, write, sha
import density_spatial as physical
import coupled_density_exit as carried
from reconstruct_exit_branch_drive import OUT

NAMES = [f'exit_z0.21_k{k}_fields16p7_{history}' for k in [9, 12] for history in ['high', 'recovery']]
SEED = 928711
CLAMP = r'''
__device__ bool held_cell(double* x,int& ref,const double* pars,const double* constants,
 const double* ar,double drive,const double* global,double gain,double nx,double ni,
 const double* pending,int tick,int depth,int member,int N){
 double z=x[6],k=x[7];bool spike=carried_cell(x,ref,pars,constants,ar,drive,global,gain,nx,ni,pending,tick,depth,member,N);
 x[6]=z;x[7]=k;return spike;
}
extern "C" __global__ void held_particles(double* state,int* refs,unsigned char* memory,
 const double* pars,const double* constants,const double* arr,const double* drive,
 const int* clock,const double* global,double gain,int* spikes,int P,int R,int ndrive,
 const double* pending,const int* members,int depth,int N){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=P*R)return;int g=id/R,tick=clock[0],ref=refs[id];
 curandStatePhilox4_32_10_t* states=(curandStatePhilox4_32_10_t*)memory;
 curandStatePhilox4_32_10_t rng=states[id];float4 n=curand_normal4(&rng);states[id]=rng;
 double ar[4],x[8];for(int j=0;j<4;j++)ar[j]=arr[j*P+g];for(int j=0;j<8;j++)x[j]=state[id*8+j];
 spikes[id]=held_cell(x,ref,pars+6*g,constants,ar,drive[(long long)min(tick,ndrive-1)*P+g],
    global,gain,n.x,n.z,pending,tick,depth,members[id],N);
 refs[id]=ref;for(int j=0;j<8;j++)state[id*8+j]=x[j];
}
extern "C" __global__ void held_supplied(double* state,int* refs,const double* normals,
 const double* pars,const double* constants,const double* arr,const double* drive,
 const int* clock,const double* global,double gain,int* spikes,int P,int R,
 const double* pending,const int* members,int depth,int N){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=P*R)return;int g=id/R,tick=clock[0],ref=refs[id];
 double ar[4],x[8];for(int j=0;j<4;j++)ar[j]=arr[j*P+g];for(int j=0;j<8;j++)x[j]=state[id*8+j];
 spikes[id]=held_cell(x,ref,pars+6*g,constants,ar,drive[g],global,gain,normals[id*2],normals[id*2+1],
    pending,tick,depth,members[id],N);
 refs[id]=ref;for(int j=0;j<8;j++)state[id*8+j]=x[j];
}
'''


class ExitDensity(carried.CarriedDensity):
    def __init__(self, name, replicas, device):
        self.carried_ready = False
        previous = physical.OPERATORS
        try:
            physical.OPERATORS = carried.ADAPTED
            physical.DensityNetwork.__init__(self, replicas=replicas, seed=SEED, device=device, duration_ms=10000, gain=30.)
        finally:
            physical.OPERATORS = previous
        cp = self.cp
        job = read(ROOT/'exit_return_probes/jobs'/f'{name}.json')
        self.job = job
        assert sha(job['held_fields_file']) == job['held_fields_sha256']
        with open(job['source_checkpoint'], 'rb') as handle:
            saved = pickle.load(handle)
        assert saved['identity'] == self.prep['graph_identity']
        state = saved['engine']
        assert state['step'] == round(job['branch_start_s']*10000)
        with np.load(job['held_fields_file']) as z:
            held_z, held_k = z['Z'], z['K']
        ids = self.geo['cell_group']
        assert np.array_equal(self.sizes, self.sizes.astype(int))
        self.members_cpu = np.array([np.flatnonzero(ids == g)[np.arange(replicas) % int(self.sizes[g])] for g in range(self.P)], dtype='i4')
        k = np.zeros(40000); k[:32000] = held_k
        zz = state['slow']['z'].copy(); zz[:32000] = held_z
        self.native_initial = np.stack([state['V'], state['s_E'], state['I_E'], state['s_I'], state['I_I'], state['slow']['m'], zz, k], axis=1)
        self.initial_state_cpu = self.native_initial[self.members_cpu]
        self.initial_ref_cpu = state['ref'][self.members_cpu]
        self.initial_global = np.array([state['termination_mechanism']['r_global'], state['global_feedback_response']['global_state']])
        order = (state['step']+np.arange(self.depth)) % self.depth
        self.pending = cp.asarray(np.stack([state['ring_sE'][order], state['ring_sI'][order]]))
        self.members = cp.asarray(self.members_cpu)
        self.initial_state = cp.asarray(self.initial_state_cpu)
        self.initial_ref = cp.asarray(self.initial_ref_cpu)
        self.drive_cpu = np.load(OUT/'drive_0p1ms.npy', mmap_mode='r')
        assert self.drive_cpu.shape == (100000, self.P)
        self.drive = cp.asarray(self.drive_cpu); self.ndrive = 100000
        self.extra = cp.RawModule(code=physical.CODE+carried.EXTRA+CLAMP, options=('--fmad=false',),
            name_expressions=['held_particles', 'held_supplied', 'observe_global'])
        self.extra_k = {key: self.extra.get_function(key) for key in ['held_particles', 'held_supplied', 'observe_global']}
        self.global_output = cp.zeros((10, 2))
        self.carried_ready = True
        self.reset()

    def particles(self):
        self.extra_k['held_particles'](((self.P*self.R+127)//128,), (128,),
            (self.state, self.ref, self.rng, self.pars, self.constants, self.arr, self.drive, self.clock,
             self.global_state, float(self.gain), self.spikes, np.int32(self.P), np.int32(self.R), np.int32(self.ndrive),
             self.pending, self.members, np.int32(self.depth), np.int32(40000)))


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'contract.json').exists()
    assert read(ROOT/'coupled_density_exit/result.json')['status'] == 'COMPLETE'
    table = read(ROOT/'exit_return_probes/extended_analysis_summary.json')
    assert table['status'] == 'COMPLETE'
    write(OUT/'contract.json', dict(status='REGISTERED_BEFORE_BRANCH_CORRESPONDENCE', created_epoch=time.time(),
        question='At the relevant actual-exitfield family, does the distribution candidate preserve the existing native K9 high-versus-quiet separation and K12 suppression, before any branchcontinuation?',
        design='Exactlyfour10s conditionalruns, Zmean.21,Kmean9or12, originalhigh12s/recovery30s states, originalactual16.7s Z/K fields. Same original50-60s exactexternalmeaninput.3479g40groups,2048particles/group,onepairednumericalseed928711.',
        physics='OnlyZandK fixedperparticle aftereach originallocalupdate, matching nativeConditionalSlow semantics. Jointfast/Mstate andpendingpulses restored fromthe actualhistory. Allrecursive spikes,R/GandM dynamic;nofutureneuralinputprovided. Groupmeanthresholds andGaussianarrivalapproximation unchanged.',
        validation='CPU/GPUlocalstep andclampidentity atinitial/pending/afterpendingtimes;captured100steps/fullstateexact. Exactsource/cutexternalmeanroute requiredbeforelaunch. Groupempiricalinitialmeanerrorreported, nohiddenprojectioncorrection.',
        comparison='Compare first10s nativeprefix atsamebranch-relativeclock, especially5-10s means/core/surround/400cells,G,shorteventsandquiet; also report native20-30s tails separately, not matchedhorizon. No phasealignment.',
        decision='This is a correspondence screen, notstabilityorcontinuation. Requiredqualitativepattern:K9high sustainedglobalactivity,K9recoveryquiet,K12bothquiet. Report allnumericandspatialresiduals; matchingtheseclasses isnecessarybutnotsufficientforacceptance. NoautomaticdenserKgrid,seedcountincrease,parameterfit orhorizonextension.',
        limitation='Onefixednumericalstream perpoint doesnotestablish resolution/noiseconvergence. Conditionalclampsarenotautonomousloops. Natural10-20s countnoiseexperimentremainsseparate andcannotcertifytheseclamps.',
        names=NAMES, numerical_seed=SEED, engine_sha256=sha(physical.__file__), carried_source_sha256=sha(carried.__file__),
        producer_sha256=sha(__file__), native_correspondence_certified=False, formal_bifurcation_allowed=False))


def check(device):
    assert read(OUT/'forcing_qa.json')['status'] == 'PASS'
    assert not (OUT/'implementation_check.json').exists()
    e = ExitDensity(NAMES[0], 64, device)
    cp = e.cp; rng = np.random.default_rng(928710); rows = []
    for tick in [0, 17, e.depth-1, e.depth]:
        e.reset(); e.clock.fill(tick)
        arr = rng.uniform(0, .5, (4, e.P)); normal = rng.normal(size=(e.P, e.R, 2)); nu = np.array(e.drive_cpu[tick])
        adjusted = e.initial_state_cpu.copy()
        if tick < e.depth:
            pulse = e.pending[:, tick].get()[:, e.members_cpu]
            adjusted[:, :, 1] += pulse[0]/e.constants_cpu[0]
            adjusted[:, :, 3] += pulse[1]/e.constants_cpu[2]
        expected, ref, spikes = physical.cpu_cell(adjusted, e.initial_ref_cpu, e.pars_cpu, e.constants_cpu, arr, nu, e.initial_global, 30., normal)
        expected[:, :, 6:8] = e.initial_state_cpu[:, :, 6:8]
        e.extra_k['held_supplied'](((e.P*e.R+127)//128,), (128,),
            (e.state, e.ref, cp.asarray(normal), e.pars, e.constants, cp.asarray(arr), cp.asarray(nu), e.clock,
             e.global_state, 30., e.spikes, np.int32(e.P), np.int32(e.R), e.pending, e.members, np.int32(e.depth), np.int32(40000)))
        actual = e.state.get(); error = float(abs(actual-expected).max())
        assert error < 1e-10
        assert np.array_equal(e.ref.get(), ref) and np.array_equal(e.spikes.get(), spikes)
        assert np.array_equal(actual[:, :, 6:8], e.initial_state_cpu[:, :, 6:8])
        rows.append(dict(tick=tick, max_error=error, ref_spikes_exact=True, Z_K_fixed_bitwise=True))
    e.reset()
    for _ in range(100): e.step()
    expected = {key: getattr(e, key).get() for key in ['state', 'ref', 'rng', 'history', 'clock', 'global_state', 'output', 'global_output']}
    e.graph(); e.chunk()
    captured = {key: np.array_equal(value, getattr(e, key).get()) for key, value in expected.items()}
    assert all(captured.values()) and np.array_equal(e.state.get()[:, :, 6:8], e.initial_state_cpu[:, :, 6:8])
    result = dict(status='PASS', local_checks=rows, captured100steps_bitwise=captured, scope='Implementationonly;no nativecorrespondenceorstability implied.')
    write(OUT/'implementation_check.json', result); print('EXIT BRANCH IMPLEMENTATION PASS', result, flush=True)


def worker(name, device):
    assert name in NAMES
    contract = read(OUT/'contract.json')
    assert sha(__file__) == contract['producer_sha256']
    assert sha(physical.__file__) == contract['engine_sha256'] and sha(carried.__file__) == contract['carried_source_sha256']
    assert read(OUT/'forcing_qa.json')['status'] == read(OUT/'implementation_check.json')['status'] == 'PASS'
    folder = OUT/name; folder.mkdir(); start = time.time()
    e = ExitDensity(name, 2048, device); cp = e.cp
    true = np.array([e.native_initial[e.geo['cell_group'] == g].mean(0) for g in range(e.P)])
    error = abs(e.initial_state_cpu.mean(1)-true).max(0)
    write(folder/'identity.json', dict(native_job=e.job, initial_source_sha256=sha(e.job['source_checkpoint']),
        empirical_initial_max_error=error.tolist(), numerical_seed=SEED, particles=2048, groups=e.P))
    e.graph(); records = []; global_records = []
    for offset in range(0, 10000, 10):
        x = e.chunk(); g = e.global_output.get()
        assert np.isfinite(x).all() and np.isfinite(g).all()
        records.append(x.astype('f4')); global_records.append(g)
        if (offset+10) % 250 == 0:
            write(folder/'progress.json', dict(status='RUNNING', pid=os.getpid(), device=device,
                elapsed_simulation_s=(offset+10)/1000, mean_E_rate_Hz=float(np.average(x[:, 0, e.E].mean(0), weights=e.sizes[e.E])),
                R=float(g[-1, 0]), Graw=float(30*g[-1, 1]), elapsed_s=time.time()-start))
    data = np.concatenate(records); glob = np.concatenate(global_records)
    state = e.state.get()
    assert np.array_equal(state[:, :, 6:8], e.initial_state_cpu[:, :, 6:8])
    count = np.bincount(e.geo['group_cell'][e.E], weights=e.sizes[e.E], minlength=400)
    field = np.zeros((10000, 400))
    for group in np.flatnonzero(e.E): field[:, e.geo['group_cell'][group]] += data[:, 0, group]*e.sizes[group]
    field /= count
    arrays = dict(elapsed_time_ms=np.arange(10000)+1., field_E_Hz=field.astype('f4'), cell_counts=count,
                  global_R_Hz=glob[:, 0], global_s=glob[:, 1])
    for j, key in enumerate(['group_rate_Hz', 'group_Z', 'group_M', 'group_K', 'group_IE', 'group_applied_II', 'group_V', 'group_abs_current']): arrays[key] = data[:, j]
    np.savez_compressed(folder/'trajectory.npz', **arrays)
    np.savez_compressed(folder/'final_state.npz', state=state, ref=e.ref.get(), rng=e.rng.get(), history=e.history.get(),
        clock=e.clock.get(), global_state=e.global_state.get(), accumulator=e.accumulator.get())
    result = dict(status='COMPLETE', name=name, duration_s=10, source_native_prefix_s=[e.job['branch_start_s'], e.job['branch_start_s']+10],
        held_Z_K_bitwise=True, elapsed_s=time.time()-start, native_correspondence_certified=False, formal_bifurcation_allowed=False)
    write(folder/'result.json', result); write(folder/'progress.json', result)
    print('EXIT BRANCH COMPLETE', result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'check', 'worker'])
    parser.add_argument('--device', type=int, default=1); parser.add_argument('--name')
    args = parser.parse_args()
    if args.command == 'prepare': prepare()
    elif args.command == 'check': check(args.device)
    else: worker(args.name, args.device)
