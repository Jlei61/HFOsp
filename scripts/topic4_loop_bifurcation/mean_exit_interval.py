#!/usr/bin/env python3
"""Four bounded conditional probes inside the validated exit interval."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import shutil
import subprocess
import time
import numpy as np
from campaign import ROOT, PYTHON, read, write, sha
import dynamic_mean_history_pair as history

OUT = ROOT/'mean_exit_interval'
SOURCES = {
    'high': ROOT/'dynamic_mean_history_pair/high/final_state.npz',
    'quiet': ROOT/'mean_boundary_correspondence/model/upper/final_state.npz',
}
JOBS = {
    'high_K9p3875': ('high', 9.3875),
    'high_K9p425': ('high', 9.425),
    'high_K9p4625': ('high', 9.4625),
    'quiet_K9p35': ('quiet', 9.35),
}


class CarriedMean(history.HistoryNetwork):
    def __init__(self, name, device):
        self.resume_ready = False
        super().__init__('high', 64, device)
        source, k = JOBS[name];self.resumed = dict(np.load(SOURCES[source]));assert self.resumed['state'].shape == (40000, 64, 8)
        assert int(self.resumed['clock'][0]) == 100000
        fields = dict(np.load(OUT/'fields'/f'{name}.npz'))
        assert np.array_equal(self.resumed['state'][:32000, 0, 6], fields['Z'])
        before = self.resumed['state'].copy();self.resumed['state'][:32000, :, 7] = fields['K'][:, None]
        assert np.array_equal(before[:, :, :7], self.resumed['state'][:, :, :7])
        self.initial_state = self.cp.asarray(self.resumed['state']);self.initial_ref = self.cp.asarray(self.resumed['ref'])
        self.initial_global = self.resumed['global_state'];self.resume_ready = True;self.reset()
        for key, value in self.resumed.items():assert np.array_equal(getattr(self, key).get(), value), key
        self.cp.get_default_memory_pool().free_all_blocks()

    def reset(self):
        super().reset()
        if self.resume_ready:
            for key, value in self.resumed.items():getattr(self, key)[:] = self.cp.asarray(value)


def prepare():
    assert read(ROOT/'mean_boundary_correspondence/analysis/result.json')['retained']
    assert read(ROOT/'mean_boundary_resolution/analysis/result.json')['retained']
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists();fields = OUT/'fields';fields.mkdir()
    with np.load(history.CASES['high'][0].parent/'held_fields.npz') as f:z, k = f['Z'], f['K']
    with np.load(SOURCES['high']) as a, np.load(SOURCES['quiet']) as b:
        for key in ['rng', 'external_rng', 'clock']:assert np.array_equal(a[key], b[key]), key
        assert np.array_equal(a['state'][:, :, 6], b['state'][:, :, 6])
    for name, (_, target) in JOBS.items():np.savez_compressed(fields/f'{name}.npz', Z=z, K=k*(target/9.35))
    write(OUT/'contract.json', dict(status='REGISTERED_FOUR_RELEVANT_CONDITIONAL_PROBES', created_epoch=time.time(),
        question='Where within9.35-9.5 does the high spatial state cease to persist, and can the quiet history remain quiet when K is returned to9.35?',
        design='Three interiorKvalues9.3875/9.425/9.4625 from the same completedR64highstate, plus quiet-to9.35 from the completedR64quietstate. All four10s. Exact fullparticlejointstates/ref/M/synapses, fullsourcehistory/delayclock andnumericalRNG retained. Change only heldKfield; Zheld/Gfree/originalgraph/nu/Poissonlaw unchanged.',
        pairing='All sources end at modelclock100000 with identical main/externalnumericalRNG. Original pendingprefix is already past; the complete recurrent sourcehistory carries allfuture arrivals. The two histories differ endogenously. Future externalstreams paired; fixedpercellnu has no clockdependence.',
        prerequisite='Both10shistories at9.35, native/modelupperexit9.5, single-replica identity andR64/R256upperexit resolution must all retain their originalguards before dispatch.',
        readouts='Full0-10s and5-10s rates/core asymmetry, spatialfield, counterfactualcoreZdrift, R/G, first100msR<=5 and sustainedactivity/censoring. Plot actual measuredresponses with historylabels; no stability designation or curve between incompatible histories.',
        stop='Exactly four10s trajectories, no automatic extension, seeds, points or formalroots. If highstate transitions to an asymmetricactive pattern, retain that outcome separately fromquiet. If quiet returns, do not claim coexistence. Review before selecting any further parameter or stability test.',
        unit='Four conditional responses with paired numericalexternalstreams, not fourindependent nativeorpatientreplicates. No added autonomousloops.',
        jobs={n: dict(history=h, K=k, source=str(SOURCES[h])) for n, (h, k) in JOBS.items()},
        sources={n: dict(path=str(p), sha256=sha(p)) for n, p in SOURCES.items()},
        dependencies={p: sha(p) for p in [__file__, history.__file__, history.leading.__file__, history.base.__file__, history.leading.previous.__file__]},
        formal_bifurcation_allowed=False))
    shutil.copy2(__file__, OUT/'producer.py')


def run(name, device):
    c = read(OUT/'contract.json');assert all(sha(p) == h for p, h in c['dependencies'].items())
    h, k = JOBS[name];assert sha(SOURCES[h]) == c['sources'][h]['sha256']
    dest = OUT/'runs'/name;dest.mkdir(parents=True, exist_ok=True);assert not (dest/'progress.json').exists()
    started = time.time();write(dest/'progress.json', dict(status='INITIALIZING', pid=os.getpid(), updated_epoch=time.time()))
    e = CarriedMean(name, device);write(dest/'implementation_qa.json', history.leading.qa(e));e.graph()
    chunks = dest/'chunks';chunks.mkdir();groups = [];globals_ = []
    for offset in range(0, 10000, 10):
        groups.append(e.chunk().astype('f4'));globals_.append(e.global_output.get())
        if (offset+10) % 100 == 0:
            value = np.concatenate(groups);glob = np.concatenate(globals_);assert np.isfinite(value).all() and np.isfinite(glob).all()
            np.savez_compressed(chunks/f'{offset-90:05d}_{offset+10:05d}.npz', group_output=value, global_R_Hz=glob[:, 0], global_s=glob[:, 1])
            groups.clear();globals_.clear()
            write(dest/'progress.json', dict(status='RUNNING', pid=os.getpid(), elapsed_simulation_ms=offset+10,
                elapsed_wall_s=time.time()-started, updated_epoch=time.time()))
            if (offset+10) % 1000 == 0:print(name, offset+10, flush=True)
    assert int(e.clock.get()[0]) == 200000 and np.array_equal(e.state.get()[:, :, 6:8], e.initial_state.get()[:, :, 6:8])
    np.savez_compressed(dest/'final_state.npz', **{key: getattr(e, key).get() for key in e.resumed})
    write(dest/'result.json', dict(status='COMPLETE', name=name, history=h, K=k, duration_s=10, full_resume_exceptK_exact=True,
        held_fields_bitwise=True, start_clock=100000, end_clock=200000, formal_bifurcation_allowed=False))
    write(dest/'progress.json', dict(status='COMPLETE', updated_epoch=time.time()));print(name, 'COMPLETE', flush=True)


def lane(device):
    names = list(JOBS)[device::2]
    for name in names:
        log = (OUT/f'{name}.log').open('w')
        p = subprocess.Popen([PYTHON, __file__, 'run', '--name', name, '--device', str(device)], stdout=log, stderr=subprocess.STDOUT)
        write(OUT/f'lane{device}.json', dict(status='RUNNING', pid=os.getpid(), worker_pid=p.pid, name=name, updated_epoch=time.time()))
        code = p.wait();log.close()
        if code:
            write(OUT/f'lane{device}.json', dict(status='FAILED', name=name, exit_code=code));raise RuntimeError((name, code))
    write(OUT/f'lane{device}.json', dict(status='COMPLETE', names=names, updated_epoch=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('command', choices=['prepare', 'run', 'lane']);p.add_argument('--name', choices=list(JOBS));p.add_argument('--device', type=int, default=0);a = p.parse_args()
    if a.command == 'prepare':prepare()
    elif a.command == 'lane':lane(a.device)
    else:
        try:run(a.name, a.device)
        except Exception:
            write(OUT/'runs'/a.name/'progress.json', dict(status='FAILED', pid=os.getpid(), updated_epoch=time.time()));raise
