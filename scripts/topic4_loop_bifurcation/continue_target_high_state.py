#!/usr/bin/env python3
"""Carry a validated target-density high state into bounded K ramps.

These are forced finite-time trajectories, not autonomous loops or certified
equilibrium branches. All cell/RNG/delay/global state is carried without lifting.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import subprocess
import time
from pathlib import Path
import numpy as np
from campaign import ROOT, REPO, PYTHON, read, write, sha
import target_density_exit as base
import density_spatial as physical
from coupled_density_exit import EXTRA
from exit_branch_density import CLAMP

OUT = ROOT / 'target_high_state_continuation'
SOURCE = base.OUT / 'individual'
KEYS = ['state', 'ref', 'rng', 'history', 'clock', 'global_state']
JOBS = {'control_K9': (5., 0., 0.), 'ramp2s_K10p5': (7., 2., 1.5),
        'ramp10s_K10p5': (15., 10., 1.5)}


def code():
    text = base.CODE
    replacements = {
        'void target_particles(': 'void continued_particles(',
        'int N,int R,int P,int depth,int individual){':
        'int N,int R,int P,int depth,int individual,const double* baseK,int origin,int ramp_steps,double deltaK){',
        'spikes[id]=held_cell(x,ref,pars+6*i,c,a,drive[(long long)tick*P+g],global,30.,normal.x,normal.z,pending,tick,depth,i,N);':
        '''double fraction=ramp_steps>0?fmin(1.,fmax(0.,(double)(tick-origin)/ramp_steps)):0.;
 if(deltaK!=0.)x[7]=baseK[i]*(1.+deltaK*fraction/9.);
 spikes[id]=held_cell(x,ref,pars+6*i,c,a,drive[g],global,30.,normal.x,normal.z,pending,tick,depth,i,N);'''}
    for old, new in replacements.items():
        assert text.count(old) == 1
        text = text.replace(old, new)
    return physical.CODE + EXTRA + CLAMP + text


class Continued(base.TargetNetwork):
    def __init__(self, source, device, ramp_s=0., delta_K=0.):
        self.continuation_ready = False
        super().__init__('individual', 128, device)
        cp = self.cp
        with np.load(source / 'final_state.npz') as z:
            self.saved = {k: z[k] for k in KEYS}
        assert self.saved['state'].shape == (self.N, self.R, 8)
        self.origin = int(self.saved['clock'][0])
        assert self.origin >= 100000 and self.origin % 100 == 0
        self.ramp_steps = round(ramp_s * 10000)
        self.delta_K = delta_K
        with np.load(ROOT / 'target_stationary_response_audit/response.npz') as z:
            external = z['mean_external_rate_per_ms']
        groups = self.geo['cell_group']
        constant = np.bincount(groups, weights=external, minlength=self.P) / self.sizes
        assert np.allclose(constant[groups], external, rtol=0, atol=2e-14)
        self.constant_cpu = constant
        self.constant_drive = cp.asarray(constant)
        self.base_K = cp.asarray(self.saved['state'][:, 0, 7].copy())
        assert np.array_equal(self.saved['state'][:, :, 7], np.repeat(self.base_K.get()[:, None], self.R, axis=1))
        self.extra_module = cp.RawModule(code=code(), options=('--fmad=false',), name_expressions=['continued_particles'])
        self.continued_particles = self.extra_module.get_function('continued_particles')
        self.continuation_ready = True
        self.reset()

    def reset(self):
        if not self.continuation_ready:
            return super().reset()
        for key in KEYS:
            getattr(self, key)[:] = self.cp.asarray(self.saved[key])
        for key in ['arr', 'spikes', 'output', 'global_output', 'accumulator']:
            getattr(self, key).fill(0)
        # The previous step's rates are derived from its saved delay ring.
        self.rate[:] = self.history[(self.origin - 1) % self.depth]

    def step(self):
        self.arrivals()
        self.continued_particles(((self.N*self.R+127)//128,), (128,),
            (self.state, self.ref, self.rng, self.pars, self.constants, self.arr,
             self.constant_drive, self.group, self.clock, self.global_state,
             self.pending, self.spikes, np.int32(self.N), np.int32(self.R),
             np.int32(self.P), np.int32(self.depth), np.int32(1), self.base_K,
             np.int32(self.origin), np.int32(self.ramp_steps), float(self.delta_K)))
        self.k['target_collect']((self.P,), (128,),
            (self.state, self.spikes, self.ptr, self.order, self.global_state,
             self.history, self.rate, self.accumulator, self.output, self.clock,
             np.int32(self.P), np.int32(self.R), np.int32(self.depth)))
        self.k['global_step']((1,), (128,),
            (self.rate, self.pars_group, self.global_state, self.clock, np.int32(self.P)))
        self.k['observe_global']((1,), (1,), (self.global_state, self.clock, self.global_output))


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'contract.json').exists()
    assert read(SOURCE / 'result.json')['status'] == 'COMPLETE'
    write(OUT / 'contract.json', dict(status='REGISTERED_BEFORE_CONTINUATION', created_epoch=time.time(),
        question='Does the established K9 high state continue toward K10.5 when K changes gradually, or do the cores collapse after a spatial recruitment transition?',
        why='K9.01875 tiny unresolved quiet-group residuals do not locate whole-network termination. Previous K10.5 conditional trajectories begin from a large clamp of a different history and cannot prove absence of a carried high state.',
        design='First carry the exact final40000x128 density state at K9 into5s K9 control. External expected input becomes the SAME stationary mean used by DirectDC, with ongoing local Gaussian streams. If the last2s preserve allE>200Hz and bothcores>300Hz, carry that complete final state into two fixed ramps K9to10.5 lasting2s or10s, each followed by5s heldK10.5. No reset of cells, RNG, delay ring, M, R or G; Z stays at the actual16.7s field mean.21. Same paired numerical RNG future for the ramps.',
        intervention='K is explicitly prescribed. Constant expected external input differs from the earlier time-varying native drive; the control exposes that substitution. Gaussian local sampling and the original cell/recurrent equations remain unchanged. These are reduced-density diagnostic trajectories, not native seeds, equilibrium continuation or autonomous recovery.',
        readouts='Rates allE/coreA/coreB/surround, G/R, counterfactual Zdrift, 400-bin spatialfield, actual imposedK. Compare terminal3s and the two ramp trajectories; record transition brackets only if eachcore sustainedbelow5Hz for1s. Report censoring and speed dependence. No Hopf/fold name and no stability claim.',
        stop='Exactly one control and at most two ramps. If control loses the high-state class, stop before ramps and report baseline mismatch. No automatic horizon, parameter or frequency extension.',
        source=str(SOURCE), source_state_sha256=sha(SOURCE/'final_state.npz'),
        jobs={k: dict(duration_s=v[0], ramp_s=v[1], delta_K=v[2]) for k,v in JOBS.items()},
        producer_sha256=sha(__file__), base_sha256=sha(base.__file__), physical_sha256=sha(physical.__file__),
        formal_bifurcation_allowed=False))


def check(device):
    assert not (OUT / 'implementation_qa.json').exists()
    e = Continued(SOURCE, device)
    cp = e.cp
    checks = {}
    checks['all_saved_states_bitwise'] = all(np.array_equal(getattr(e,k).get(),e.saved[k]) for k in KEYS)
    assert checks['all_saved_states_bitwise']
    # Original kernel at clock0 with zero initial pending has identical delayed
    # arrivals and local physics. Check the local-step substitution directly;
    # absolute clock only controls pending (disabled after the original history).
    e.arrivals()
    e.continued_particles(((e.N*e.R+127)//128,), (128,),
        (e.state,e.ref,e.rng,e.pars,e.constants,e.arr,e.constant_drive,e.group,e.clock,
         e.global_state,e.pending,e.spikes,np.int32(e.N),np.int32(e.R),np.int32(e.P),
         np.int32(e.depth),np.int32(1),e.base_K,np.int32(e.origin),np.int32(0),0.))
    reference={k:getattr(e,k).get() for k in ['state','ref','rng','spikes']}
    e.reset();e.arrivals()
    # Provide the original kernel with a short repeated constant table at a
    # clock beyond pending injection; no out-of-range access is possible.
    local_clock=cp.asarray([e.depth],dtype='i4')
    drive=cp.broadcast_to(e.constant_drive,(e.depth+1,e.P)).copy()
    e.k['target_particles'](((e.N*e.R+127)//128,), (128,),
        (e.state,e.ref,e.rng,e.pars,e.constants,e.arr,drive,e.group,local_clock,e.global_state,
         e.pending,e.spikes,np.int32(e.N),np.int32(e.R),np.int32(e.P),np.int32(e.depth),np.int32(1)))
    checks['original_constant_input_step_bitwise']={k:np.array_equal(v,getattr(e,k).get()) for k,v in reference.items()}
    assert all(checks['original_constant_input_step_bitwise'].values())
    del reference,drive
    e.reset()
    for _ in range(100):e.step()
    names=KEYS+['output','global_output','accumulator','rate']
    reference={k:getattr(e,k).get() for k in names}
    e.graph();e.chunk()
    checks['captured_continuation_bitwise']={k:np.array_equal(v,getattr(e,k).get()) for k,v in reference.items()}
    assert all(checks['captured_continuation_bitwise'].values())
    del reference
    # Verify the ramp acts only on K before the exact held local update, at
    # beginning/midpoint/end, including old delay-state clock semantics.
    rows=[];e.ramp_steps=20000;e.delta_K=1.5
    for offset in [0,10000,20000,30000]:
        e.reset();e.clock[0]=e.origin+offset
        beforeZ=e.state[:,:,6].get();e.step()
        expected=e.base_K.get()*(1+1.5*min(offset/20000,1)/9)
        error=float(abs(e.state[:,:,7].get()-expected[:,None]).max())
        assert error<2e-14 and np.array_equal(beforeZ,e.state[:,:,6].get())
        rows.append(dict(offset_steps=offset,max_K_error=error,Z_bitwise=True))
    checks['ramp_schedule']=rows
    write(OUT/'implementation_qa.json',dict(status='PASS',checks=checks,producer_sha256=sha(__file__)))
    print('CONTINUATION QA PASS',checks,flush=True)


def worker(name,device):
    c=read(OUT/'contract.json')
    assert c['producer_sha256']==sha(__file__)==read(OUT/'implementation_qa.json')['producer_sha256']
    assert c['base_sha256']==sha(base.__file__) and c['physical_sha256']==sha(physical.__file__)
    folder=OUT/name;folder.mkdir(exist_ok=True);assert not (folder/'progress.json').exists()
    start=time.time();write(folder/'progress.json',dict(status='INITIALIZING',pid=os.getpid(),device=device))
    duration,ramp,delta=JOBS[name]
    source=SOURCE if name=='control_K9' else OUT/'control_K9'
    if name!='control_K9':assert read(source/'result.json')['high_state_preserved']
    e=Continued(source,device,ramp,delta)
    assert all(np.array_equal(getattr(e,k).get(),e.saved[k]) for k in KEYS)
    write(folder/'initial_qa.json',dict(status='PASS',full_state_rng_delay_clock_carried_bitwise=True,
        source=str(source),origin_steps=e.origin,initial_R_G=[float(e.saved['global_state'][0]),float(30*e.saved['global_state'][1])]))
    e.graph();outputs=[];globals=[]
    for offset in range(0,round(duration*1000),10):
        outputs.append(e.chunk().astype('f4'));globals.append(e.global_output.get())
        if (offset+10)%500==0:
            write(folder/'progress.json',dict(status='RUNNING',name=name,pid=os.getpid(),device=device,
                simulation_s=(offset+10)/1000,elapsed_wall_s=time.time()-start,updated_epoch=time.time()))
            print(name,(offset+10)/1000,flush=True)
    value=np.concatenate(outputs);glob=np.concatenate(globals)
    assert np.isfinite(value).all() and np.isfinite(glob).all()
    assert int(e.clock.get()[0])==e.origin+round(duration*10000)
    assert np.array_equal(e.state[:,:,6].get(),e.saved['state'][:,:,6])
    np.savez_compressed(folder/'trajectory.npz',elapsed_time_ms=np.arange(1,len(value)+1),group_output=value,
        channels=['rate_Hz','Z','M','K','IE','applied_II','V','abs_current','Z_eligible_fraction'],global_R_Hz=glob[:,0],global_s=glob[:,1])
    np.savez_compressed(folder/'final_state.npz',**{k:getattr(e,k).get() for k in KEYS})
    E=e.geo['population']==0;reg=e.geo['group_region'];masks=[E]+[E&(reg==j) for j in range(3)]
    tail=np.array([np.average(value[-2000:,0,m],weights=e.sizes[m],axis=1).mean() for m in masks])
    result=dict(status='COMPLETE',name=name,duration_s=duration,terminal2s_rates_allE_A_B_surround_Hz=tail.tolist(),
        high_state_preserved=bool(tail[0]>200 and min(tail[1:3])>300),elapsed_s=time.time()-start,
        full_state_continuation=True,forced_K=True,formal_bifurcation_allowed=False)
    write(folder/'result.json',result);write(folder/'progress.json',result);print(result,flush=True)


def supervise():
    assert not (OUT/'status.json').exists()
    def launch(name,device):
        with (OUT/f'{name}.log').open('w') as log:
            return subprocess.Popen([PYTHON,__file__,'worker','--name',name,'--device',str(device)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    active={'control_K9':launch('control_K9',0)};done=[];failed=[];ramps_launched=False
    while active:
        for name,p in list(active.items()):
            status=p.poll()
            if status is None:continue
            path=OUT/name/'result.json'
            if status==0 and path.exists():done.append(name)
            else:failed.append(dict(name=name,exit_code=status))
            del active[name]
        if 'control_K9' in done and not ramps_launched and not failed:
            if read(OUT/'control_K9/result.json')['high_state_preserved']:
                active.update({name:launch(name,d) for d,name in enumerate(['ramp2s_K10p5','ramp10s_K10p5'])})
            else:failed.append(dict(name='control_K9',reason='HIGH_STATE_NOT_PRESERVED_NO_RAMPS_DISPATCHED'))
            ramps_launched=True
        write(OUT/'status.json',dict(status='FAILED' if failed else 'RUNNING' if active else 'COMPLETE',
            supervisor_pid=os.getpid(),completed=done,failed=failed,
            active=[dict(name=n,pid=p.pid) for n,p in active.items()],updated_epoch=time.time()))
        if active:time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','check','worker','supervise'])
    p.add_argument('--name',choices=list(JOBS));p.add_argument('--device',type=int,default=0)
    a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='check':check(a.device)
    elif a.command=='worker':worker(a.name,a.device)
    else:supervise()
