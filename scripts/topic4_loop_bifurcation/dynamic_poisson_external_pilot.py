#!/usr/bin/env python3
"""One matched physical-time assay restoring discrete native external counts."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import shutil
import time
import numpy as np
from campaign import ROOT, read, write, sha
import dynamic_individual_source_pilot as base
import dynamic_bernoulli_source_pilot as bern
import density_spatial as physical
import target_density_exit as target

OUT = ROOT/'dynamic_poisson_external_pilot'
CODE = base.CODE.split('extern "C" __global__ void source_collect')[0]
CODE = CODE.replace('unsigned char* memory,', 'unsigned char* memory,unsigned char* external_memory,double* draws,')
insert = r'''
 curandStatePhilox4_32_10_t* ex=(curandStatePhilox4_32_10_t*)external_memory;
 curandStatePhilox4_32_10_t er=ex[id];double lambda=nu[i]*.1;
 double count=curand_poisson(&er,lambda);ex[id]=er;
 double J=pars[6*i+4],vr=arr[2*N+i]*.1,den=sqrt(fmax(vr+J*J*lambda,0.));
 double nx=den>0.?(sqrt(fmax(vr,0.))*normal.x+J*(count-lambda))/den:0.;
 if(tick==0){draws[3*id]=normal.x;draws[3*id+1]=normal.z;draws[3*id+2]=count;}
'''
assert 'double x[8],a[4];' in CODE and '30.,normal.x,normal.z' in CODE
CODE = CODE.replace('double x[8],a[4];', insert+'\n double x[8],a[4];')
CODE = CODE.replace('30.,normal.x,normal.z', '30.,nx,normal.z')


class PoissonNetwork(bern.BernoulliNetwork):
    def __init__(self, replicas, device):
        self.external_ready = False
        super().__init__(replicas, device);cp = self.cp
        self.external_rng = cp.empty_like(self.rng);self.draws = cp.empty((self.N, self.R, 3))
        self.poisson_module = cp.RawModule(code=physical.CODE+target.EXTRA+target.CLAMP+CODE,
            options=('--fmad=false',), name_expressions=['fixed_particles'])
        kernel = self.poisson_module.get_function('fixed_particles')
        def with_external(grid, block, arguments):
            return kernel(grid, block, (*arguments[:3], self.external_rng, self.draws, *arguments[3:]))
        self.k['fixed_particles'] = with_external;self.external_ready = True;self.reset()

    def reset(self):
        super().reset()
        if self.external_ready:
            self.k['init_rng'](((self.N*self.R+127)//128,), (128,),
                (self.external_rng, np.int32(self.N*self.R), np.uint64(930199), np.int32(self.R)))


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert not read(bern.OUT/'analysis/result.json')['development_relevance_retained']
    write(OUT/'contract.json', dict(status='REGISTERED_ONE_EXTERNAL_INPUT_ASSAY', created_epoch=time.time(),
        question='Does restoring the native discrete Poisson external-input law remove the remaining free-dynamics error after recurrent Bernoulli variance correction?',
        evidence='Recurrent Bernoulli variance delayed the excess recruitment but tail spatialRMS remained18.9Hz and R crossed200; the finite-step variance correction alone failed. The successful local spectral sampler already uses discrete Poisson external counts, whereas the free physical kernel still merges external and recurrent variance into a Gaussian increment.',
        design='One1s physical trajectory at the same complete72s initial state, heldZ/K,64 replicas, exactgraph/thresholds/delays and percellfixednu. Preserve Bernoulli recurrent variance and its Gaussian residual, but replace only the external Gaussian component with independent Poisson(nu*dt) counts times the original jump. Same mainGaussian RNG stream, separate reproducible external numerical RNG. No native biological parameter fitting.',
        implementation='Feed the unchanged membrane kernel an effective Gaussian coordinate whose synaptic increment is exactly recurrent Gaussian plus centered Poisson external count. Existing external mean remains, so its sum is the actual count. Handle zero total variance explicitly. Validate eachcell/replica against CPU using the saved actual normals/counts, including mixed recurrentandexternal variance.',
        limits='External numerical replicas do not share realized counts with the native reference; they implement the same conditional law. Residual source correlations and numerical convergence remain unvalidated. No automatic stability or bifurcation claim.',
        stop='One1s assay only; retain same native relevance guards and stop for analysis.',
        producer_sha256=sha(__file__), bernoulli_sha256=sha(bern.__file__), base_sha256=sha(base.__file__),
        formal_bifurcation_allowed=False))
    shutil.copy2(__file__, OUT/'producer.py')


def qa(e):
    # Reuse the independent mean/variance operator and graph checks, adding a
    # direct mixed-input CPU comparison for the only changed local operation.
    previous = bern.qa(e);cp = e.cp;rng = np.random.default_rng(930211)
    e.reset();arr = rng.uniform(.1, 2, size=(4, e.N));e.arr[:] = cp.asarray(arr)
    initial = e.initial_state.get();refs = e.initial_ref.get()
    initial[:, :, 1] += e.pending_cpu[0, 0, :, None]/e.constants_cpu[0]
    initial[:, :, 3] += e.pending_cpu[1, 0, :, None]/e.constants_cpu[2]
    e.k['fixed_particles'](((e.N*e.R+127)//128,), (128,),
        (e.state, e.ref, e.rng, e.pars, e.constants, e.arr, e.nu, e.clock, e.global_state,
         e.pending, e.spikes, np.int32(e.N), np.int32(e.R), np.int32(e.depth)))
    draws = e.draws.get();J = e.pars_cpu[:, 4, None];lam = e.raw['nu_per_ms'][:, None]*.1
    assert (draws[:, :, 2] >= 0).all() and np.array_equal(draws[:, :, 2], np.floor(draws[:, :, 2]))
    vr = arr[2, :, None]*.1;normal = np.stack([(np.sqrt(vr)*draws[:, :, 0]+J*(draws[:, :, 2]-lam))/np.sqrt(vr+J*J*lam), draws[:, :, 1]], axis=2)
    expected, ref, spikes = physical.cpu_cell(initial, refs, e.pars_cpu, e.constants_cpu,
        arr, e.raw['nu_per_ms'], e.initial_global, 30., normal)
    expected[:, :, 6:8] = e.initial_state.get()[:, :, 6:8]
    error = float(abs(expected-e.state.get()).max());assert error < 2e-11
    assert np.array_equal(ref, e.ref.get()) and np.array_equal(spikes, e.spikes.get())
    e.reset();e.graph();e.chunk();ext = e.external_rng.get();e.reset()
    for _ in range(100):e.step()
    assert np.array_equal(ext, e.external_rng.get());e.reset()
    previous.update(mixed_Poisson_external_CPU_max_error=error, mixed_spikes_and_refs_exact=True,
        external_RNG_captured_bitwise=True, exact_count_synaptic_jump_units=True)
    return previous


def run(device):
    c = read(OUT/'contract.json')
    assert c['producer_sha256'] == sha(__file__) and c['bernoulli_sha256'] == sha(bern.__file__) and c['base_sha256'] == sha(base.__file__)
    assert not (OUT/'supervisor.json').exists()
    started = time.time();write(OUT/'supervisor.json', dict(status='RUNNING_ONE_EXTERNAL_INPUT_ASSAY', pid=os.getpid(), updated_epoch=time.time()))
    e = PoissonNetwork(base.REPLICAS, device);write(OUT/'implementation_qa.json', qa(e));print('POISSON EXTERNAL QA PASS', flush=True)
    e.graph();groups = [];globals_ = [];cells = []
    for offset in range(0, base.DURATION_MS, 10):
        groups.append(e.chunk().astype('f4'));globals_.append(e.global_output.get());cells.append(e.source_output.get().astype('f4'))
        if (offset+10) % 100 == 0:
            write(OUT/'progress.json', dict(status='RUNNING', pid=os.getpid(), elapsed_simulation_ms=offset+10,
                elapsed_wall_s=time.time()-started, updated_epoch=time.time()))
            print('POISSON EXTERNAL', offset+10, flush=True)
    value, global_, source = np.concatenate(groups), np.concatenate(globals_), np.concatenate(cells)
    assert int(e.clock.get()[0]) == 10000 and np.isfinite(value).all() and np.isfinite(global_).all()
    assert np.array_equal(e.state.get()[:, :, 6:8], e.initial_state.get()[:, :, 6:8])
    with np.load(bern.OUT/'final_state.npz') as ref:assert np.array_equal(e.rng.get(), ref['rng'])
    np.savez_compressed(OUT/'trajectory.npz', elapsed_time_ms=np.arange(1, 1001), group_output=value,
        global_R_Hz=global_[:, 0], global_s=global_[:, 1], cell_rate_Hz=source)
    np.savez_compressed(OUT/'final_state.npz', state=e.state.get(), ref=e.ref.get(), rng=e.rng.get(), external_rng=e.external_rng.get(),
        source_history=e.source_history.get(), history=e.history.get(), clock=e.clock.get(), global_state=e.global_state.get())
    write(OUT/'result.json', dict(status='COMPLETE_ONE_EXTERNAL_INPUT_ASSAY_ANALYSIS_PENDING', duration_ms=1000,
        replicas=base.REPLICAS, main_Gaussian_RNG_bitwise_paired=True, held_fields_bitwise=True, elapsed_s=time.time()-started,
        formal_bifurcation_allowed=False))
    write(OUT/'supervisor.json', dict(status='COMPLETE', updated_epoch=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('command', choices=['prepare', 'run']);p.add_argument('--device', type=int, default=1);a = p.parse_args()
    if a.command == 'prepare':prepare()
    else:
        try:run(a.device)
        except Exception:
            write(OUT/'supervisor.json', dict(status='FAILED', pid=os.getpid(), updated_epoch=time.time()));raise
