#!/usr/bin/env python3
"""Bounded, complete-state return tests at one common conditional K."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import shutil
import subprocess
import time
import numpy as np
from campaign import ROOT,PYTHON,read,write,sha
import dynamic_mean_history_pair as history

OUT=ROOT/'physical_history_return'
K=9.425
SOURCES={
    'from_lower_K':'high_K9p3875',
    'same_K_control':'high_K9p425',
    'from_upper_K':'high_K9p4625',
    'from_quiet':'quiet_K9p35',
}


class ReturnNetwork(history.HistoryNetwork):
    def __init__(self,name,device):
        self.resume_ready=False
        super().__init__('high',64,device)
        source=ROOT/'mean_exit_interval/runs'/SOURCES[name]/'final_state.npz'
        self.resumed=dict(np.load(source));assert self.resumed['state'].shape==(40000,64,8)
        assert int(self.resumed['clock'][0])==200000
        fields=dict(np.load(OUT/'held_fields.npz'))
        assert np.array_equal(self.resumed['state'][:32000,0,6],fields['Z'])
        self.resumed['state'][:32000,:,7]=fields['K'][:,None]
        self.initial_state=self.cp.asarray(self.resumed['state']);self.initial_ref=self.cp.asarray(self.resumed['ref'])
        self.initial_global=self.resumed['global_state'];self.resume_ready=True;self.reset()
        for key,value in self.resumed.items():assert np.array_equal(getattr(self,key).get(),value),key
        self.cp.get_default_memory_pool().free_all_blocks()

    def reset(self):
        super().reset()
        if self.resume_ready:
            for key,value in self.resumed.items():getattr(self,key)[:]=self.cp.asarray(value)


def prepare():
    assert read(ROOT/'native_mean_exit_interval_v2/analysis/result.json')['both_conditions_retained']
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    shutil.copy2(ROOT/'mean_exit_interval/fields/high_K9p425.npz',OUT/'held_fields.npz')
    initial=None;source_rows={}
    for name,old in SOURCES.items():
        path=ROOT/'mean_exit_interval/runs'/old/'final_state.npz'
        with np.load(path) as z:
            q={k:z[k] for k in ['rng','external_rng','clock']}
            if initial is None:initial=q
            else:
                for key in q:assert np.array_equal(q[key],initial[key]),(name,key)
        source_rows[name]=dict(path=str(path),sha256=sha(path),preceding_condition=old)
    from analyze_mean_exit_interval import projection
    _,_,counts,_=projection()
    with np.load(ROOT/'mean_exit_interval/analysis/high_K9p425_readouts.npz') as z:reference=z['field'][-1000:].mean(0)
    contrasts={}
    for name,old in SOURCES.items():
        with np.load(ROOT/'mean_exit_interval/analysis'/f'{old}_readouts.npz') as z:field=z['field'][-1000:].mean(0)
        contrasts[name]=float(np.sqrt(np.average((field-reference)**2,weights=counts)))
    assert contrasts['from_lower_K']>1 and contrasts['from_upper_K']>1
    contract=dict(status='REGISTERED_FOUR_COMPLETE_STATE_RETURN_TESTS',created_epoch=time.time(),
        question='At the commonK9.425, do high states arriving from the two neighboringK values return to the same spatial pattern, while a paired quiet history remains distinct?',
        reason='Finite persistence at severalKpoints is not local stability. Complete physically generated neighboringstates avoid constructing an artificial microscopic lift from meanrates. The priorquietK9.35 and highK9.35 references had different future intervals; this test also provides an exactly paired high/quiet comparison at oneK.',
        design='Four5s R64 trajectories at identical actualZfield/heldK9.425, originalgraph andfixedpercellPoissonlaw. Sources are the complete200000clock outputs ofthe precedingfour probes. Retain V/ref/synapses/M/G/fullsource-delayhistory, clock andbothRNGs; changeonlyK. Threehighhistories bracket the target; fourthquiet. No imposedrate, period, spatialpattern, timingorZreset.',
        guards='Report each1s rate/field/coreZdrift and first100ms lowactivity; final4-5s EfieldRMS andIfieldRMS relative to same-Kcontrol each<=1Hz, eachEcoremeanrate difference<=1Hz and allE<=.2Hz, E-MfieldRMS<=1count for eachneighbor highhistory. Report penultimate-to-final1s fieldRMS<=1Hz for allthreehighhistories. Quiet is originaljoint allE/A/B<5Hz in>=95percent10msbins. These are selected-direction macroreturn guards, not fullmicroscopic attraction or stability certification.',
        decision='If either highhistory retains a distinctspatialstate, do not assume one smoothupperbranch. If allthreeconverge, report only tested-direction macroreturn; separately report quiet coexistence under pairedfuturestreams. No universalbasin, deterministiclimit, eigenvalue or formalbifurcation follows.',
        stop='Exactly four5s trajectories andanalysis. No automatic extension, perturbation, parameterpoint, seed, root or iteration.',
        duration_s=5,replicas=64,K=K,sources=source_rows,initial_field_contrasts_Hz=contrasts,
        dependencies={p:sha(p) for p in [__file__,history.__file__,history.leading.__file__,history.base.__file__,history.leading.previous.__file__]},
        formal_bifurcation_allowed=False)
    write(OUT/'contract.json',contract);shutil.copy2(__file__,OUT/'producer.py')


def run(name,device):
    c=read(OUT/'contract.json');assert all(sha(p)==h for p,h in c['dependencies'].items())
    assert sha(c['sources'][name]['path'])==c['sources'][name]['sha256']
    dest=OUT/'runs'/name;dest.mkdir(parents=True,exist_ok=True);assert not (dest/'progress.json').exists()
    start=time.time();write(dest/'progress.json',dict(status='INITIALIZING',pid=os.getpid(),updated_epoch=time.time()))
    e=ReturnNetwork(name,device);write(dest/'implementation_qa.json',history.leading.qa(e));e.graph()
    chunks=dest/'chunks';chunks.mkdir();groups=[];glob=[]
    for offset in range(0,5000,10):
        groups.append(e.chunk().astype('f4'));glob.append(e.global_output.get())
        if (offset+10)%100==0:
            v=np.concatenate(groups);g=np.concatenate(glob);assert np.isfinite(v).all() and np.isfinite(g).all()
            np.savez_compressed(chunks/f'{offset-90:05d}_{offset+10:05d}.npz',group_output=v,global_R_Hz=g[:,0],global_s=g[:,1])
            groups.clear();glob.clear()
            write(dest/'progress.json',dict(status='RUNNING',pid=os.getpid(),elapsed_simulation_ms=offset+10,elapsed_wall_s=time.time()-start,updated_epoch=time.time()))
            if (offset+10)%1000==0:print(name,offset+10,flush=True)
    assert int(e.clock.get()[0])==250000 and np.array_equal(e.state.get()[:,:,6:8],e.initial_state.get()[:,:,6:8])
    np.savez_compressed(dest/'final_state.npz',**{key:getattr(e,key).get() for key in e.resumed})
    write(dest/'result.json',dict(status='COMPLETE',duration_s=5,full_resume_exceptK_exact=True,held_fields_bitwise=True,formal_bifurcation_allowed=False))
    write(dest/'progress.json',dict(status='COMPLETE',updated_epoch=time.time()));print(name,'COMPLETE',flush=True)


def lane(device):
    for name in list(SOURCES)[device::2]:
        with (OUT/f'{name}.log').open('w') as log:
            p=subprocess.Popen([PYTHON,__file__,'run','--name',name,'--device',str(device)],stdout=log,stderr=subprocess.STDOUT)
            write(OUT/f'lane{device}.json',dict(status='RUNNING',pid=os.getpid(),worker_pid=p.pid,name=name,updated_epoch=time.time()))
            code=p.wait()
        if code:
            write(OUT/f'lane{device}.json',dict(status='FAILED',name=name,exit_code=code));raise RuntimeError((name,code))
    write(OUT/f'lane{device}.json',dict(status='COMPLETE',updated_epoch=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','lane']);p.add_argument('--name',choices=list(SOURCES));p.add_argument('--device',type=int,default=0);x=p.parse_args()
    if x.command=='prepare':prepare()
    elif x.command=='lane':lane(x.device)
    else:
        try:run(x.name,x.device)
        except Exception:
            write(OUT/'runs'/x.name/'progress.json',dict(status='FAILED',pid=os.getpid(),updated_epoch=time.time()));raise
