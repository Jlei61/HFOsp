#!/usr/bin/env python3
"""Baseline-gated native tau_G x adaptation-strength experiment for new Fig6.

Keeps the manual-core spatial SNN, local inputs, single current M, native Z,
15-ms causal rate filter and noise seed. Only G's target becomes a smooth Hill
response. Parameter-plane results are finite-horizon native behavior, NEVER
certified Hopf branches. Formal continuation requires dynamic closure evidence.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

MAIN = Path('/home/honglab/leijiaxin/HFOsp')
WORK = MAIN/'.worktrees/topic4-continuous-core-state-r1'
ROOT = Path('/data/hfosp/topic4_sef_hfo/unified_feedback_phase_20261030')
SOURCE = Path('/data/hfosp/topic4_sef_hfo/unified_current_adaptation_20261030')
SOURCE_RUN = 'unified_ref48_s9108405'
SEED = 9108405
MODEL = 'ZGM_UNIFIED_CURRENT_SMOOTH_G_V1'
ETA_REF = .0005
BASELINE = 'smooth_tg0p5_lm1_s9108405'
HILL_N = 14/3
HILL_HALF = 350.
sys.path.insert(0, str(MAIN/'scripts'))
spec = importlib.util.spec_from_file_location('frozen_unified_current',
    SOURCE/'source/run_topic4_unified_current_adaptation.py')
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target_fraction(rate, kind='hill'):
    r = np.maximum(np.asarray(rate, dtype=float), 0.)
    if kind == 'clip':
        return np.clip((r-200.)/300., 0., 1.)
    if kind != 'hill':
        raise ValueError(kind)
    power = (r/HILL_HALF)**HILL_N
    return power/(1.+power)


class SmoothCurrentSlow(u.UnifiedCurrentSlow):
    feedback_kind = 'hill'
    adaptation_scale = 1.

    def step(self, spk, labels, dt):
        # All source terms use the pre-step causal all-E rate, exactly as V1.
        tau = self.current_tau_ms()
        z = self.z[:self.NE]
        z += (dt/self.cfg.tau_z)*((self._I_I_last[:self.NE] < self.cfg.I_th_EI)-z)
        np.clip(z, 0., 1., out=z)
        m = self.m[:self.NE]
        m *= np.exp(-dt/tau)
        jump = .0005*1000./tau + self.high_jump_mV*self.gate
        # eta_ref stays fixed: lambda changes actual spike-triggered current.
        m[np.asarray(spk[:self.NE], bool)] += self.adaptation_scale*jump/ETA_REF
        decay = np.exp(-dt/self.global_tau_ms)
        target = float(target_fraction(self.r_global, self.feedback_kind))
        self.global_state = self.global_state*decay + target*(1.-decay)
        self.r_global = (self.r_global*np.exp(-dt/15.)
            + np.count_nonzero(spk[:self.NE])/self.NE*1000./15.)
        self._step_index += 1


def job_name(tau, strength):
    return f'smooth_tg{tau:g}_lm{strength:g}_s{SEED}'.replace('.', 'p')


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'protocol.json').exists():
        p = read(ROOT/'protocol.json')
        assert p['runner_sha256'] == sha(__file__), 'Campaign source changed after preparation'
        return p
    effective = read(SOURCE/'effective_model.json')
    assert effective['producer_sha256'] == sha(u.__file__)
    baseline_result = read(SOURCE/'analysis'/SOURCE_RUN/'result.json')
    assert baseline_result['run_status'] == 'COMPLETE'
    assert all(baseline_result[k] for k in ['has_high_entry','has_autonomous_exit','has_sustained_return'])
    old = read(SOURCE/'protocol.json')
    for path, digest in old['source_hashes'].items():
        assert sha(path) == digest, path
    code = ROOT/'source'
    code.mkdir(exist_ok=True)
    for path in [Path(__file__),Path(__file__).with_name('local_stability.py'),
                 Path(__file__).with_name('plot_results.py'),
                 WORK/'scripts/analyze_topic4_unified_current_adaptation.py']:
        shutil.copy2(path,code/path.name)
    shutil.copy2(SOURCE/'geometry.npz', ROOT/'geometry.npz')
    shutil.copy2(SOURCE/'effective_model.json', ROOT/'source_effective_model.json')
    axis_tau = [.25,.5,1.]
    axis_m = [.5,1.,2.]
    names = [BASELINE]
    source_job = read(SOURCE/'runs'/SOURCE_RUN/'result.json')['job']
    for tau in axis_tau:
        for strength in axis_m:
            name = job_name(tau,strength)
            j = copy.deepcopy(source_job)
            j.update(name=name,model=MODEL,global_tau_s=tau,
                     adaptation_scale=strength,eta_m=ETA_REF,feedback_kind='hill',
                     hill_n=HILL_N,hill_half_Hz=HILL_HALF,seed=SEED,
                     horizon_s=60.,checkpoint_s=2.,qa=False,stop_after_second_entry=False,
                     stage='smooth_G_native_behavior_plane',K_state=False,sahp_gain=0.,
                     no_intervention=True,
                     mechanism='Single current M; Hill G target; no independent K. See effective_model.json.')
            write(ROOT/'jobs'/f'{name}.json',j)
            if name != BASELINE:
                names.append(name)
    p = copy.deepcopy(old)
    p.update(model=MODEL,runner_sha256=sha(__file__),
        equations=dict(membrane='tau_m Vdot=-V+IE-Z II-Z G(V-EG)-eta_ref M',
            adaptation='M*=exp(-dt/tauA); spike adds lambda_M*[0.0005*1s/tauA+7.68*q]/eta_ref',
            tauA='5s if causal all-E R<=5Hz, otherwise 0.5s',
            global_feedback='tauG dG/dt=30 R^(14/3)/(R^(14/3)+350^(14/3))-G; R filter15ms',
            resource='5s Zdot=1{II+(18-EG)G<95.19851312666987}-Z',
            K_state=False,K_conductance=False),
        source_unified_sha256=sha(u.__file__),
        authorization='2026-10-30 user approved smooth H, Fig6 workpoint validation, and tauG x adaptation strength analysis',
        deadline_epoch=time.time()+24*3600,protected_global_authorized=False,
        baseline=BASELINE,source_baseline=str(SOURCE/'runs'/SOURCE_RUN),
        source_thread_id='01a18ca9-467d-7f03-86d1-2c18c0c08054',
        figure_candidate='unified_current_adaptation_readout14_20261030',
        graph_and_input_identity=old['identity'],noise_seed=SEED,
        statistical_unit='One fixed network; one paired noise realization per setting. No seed-level population inference.',
        axes=dict(tau_G_s=axis_tau,lambda_M=axis_m),queue=names,
        max_new_native_runs=9,max_workers=2,max_external_plus_own_native_workers=4,
        baseline_gate='Full 60s completed; initial brief activity and valid t8s baseline; high entry; autonomous exit; both-core Z recovery and sustained brief return.',
        baseline_gate_failure='Stop parameter-plane dispatch and retain results. Do not tune other parameters to rescue the gate.',
        scan_scope='3x3 paired-noise pilot around the actual Fig6 parameter setting. No extrapolation or interpolation into a formal phase boundary.',
        analysis_levels=['Native finite-horizon behavior','Conditional local equilibrium stability, pending dynamic closure','Full slow recovery mechanism'],
        formal_Hopf_established=False,
        formal_gate='Native-compatible dynamic closure at the SAME equilibrium, isolated complex-pair crossing, remaining modes stable, transversality, nonlinear nondegeneracy and periodic-orbit continuation.',
        Z_policy='Native evolving Z for autonomous runs; frozen spatial Z only in separately identified conditional stability analyses.',
        branch_rules='Local scalar closure excludes R=5Hz tau_A switch and M-drive kinks R=200/500Hz. Smooth H alone does not smooth these mechanisms.',
        pending_dynamic_response=dict(status='NOT_IDENTIFIED',
          required=['Steady response under frozen native spatial Z and fixed G/A',
                    'Small perturbations of rate/current/conductance, matched input, at a stationary operating point',
                    'Retain E/I, spatial and synaptic modes needed to reproduce gain and phase',
                    'Independent transient prediction before using eigenvalues as native evidence']),
        figure6_point='Network, source lineage, units and parameters fixed; layout edits or noise-seed searches in the other conversation do not replace this baseline automatically.')
    write(ROOT/'protocol.json',p)
    write(ROOT/'baseline_source_audit.json',dict(status='PASS',
        source=str(SOURCE),identity=old['identity'],
        route_qa=read(SOURCE/'route_qa.json'),quiet_decay_qa=read(SOURCE/'quiet_decay_qa.json'),
        high_entry_s=[e['onset_s'] for e in baseline_result['temporal']['entries']],
        episodes=[{k:e[k] for k in ['exit_start_s','both_core_Z_reference_s','brief_count','brief_span_s','sustained_return']}
                  for e in baseline_result['episodes']],
        first_exit_is_not_full_return=True,
        independent_K_absent=True,formal_bifurcation=False))
    write(ROOT/'effective_model.json',dict(**{k:v for k,v in effective.items()
        if k not in ['model','producer','producer_sha256','G_equation','statistical_unit']},
        model=MODEL,producer=str(Path(__file__).resolve()),producer_sha256=sha(__file__),
        G_equation='tau_G dG/dt = 30 R^(14/3)/(R^(14/3)+350^(14/3)) - G',
        adaptation_strength='lambda_M multiplies the current jump. eta_M=0.0005 is fixed normalization.',
        adaptation_spike_increment_scaled_mV='lambda_M * [0.0005 * (1 s / tau_A(R)) + 7.68 * q(R)]',
        adaptation_q_unchanged=True,tau_A_rule_unchanged=True,
        statistical_unit=p['statistical_unit'],
        smoothing_reference='Value and slope matched at 350 Hz; this is a function reference, not an identified equilibrium.'))
    return p


def qa():
    from local_stability import jacobian,characteristic,fast_filter_coefficients,fast_filter_hopf_candidates
    p = prepare()
    cfg = u.base.old.MZSlowVarsConfig(use_z=True,use_m=True,tau_z=5000.,I_th_EI=95.19851312666987,tau_adp=1000.,eta_m=ETA_REF)
    rng = np.random.default_rng(20261030)
    old = u.UnifiedCurrentSlow(8,18.,cfg,NE=6,global_gain=30.,mode='native',gamma=0.)
    new = SmoothCurrentSlow(8,18.,cfg,NE=6,global_gain=30.,mode='native',gamma=0.)
    new.feedback_kind = 'clip'
    for obj in [old,new]:obj.global_reversal=-17.662847938268442
    for rate in [0.,5.,5.1,100.,199.,200.,201.,350.,499.,500.,600.]:
        z = rng.uniform(.1,1.,6); m = rng.uniform(0,10000,6)
        ie,ii = rng.uniform(0,300,(2,8)); spikes=rng.random(8)<.5
        for obj in [old,new]:
            obj.r_global=rate;obj.global_state=.37;obj.z[:6]=z;obj.m[:6]=m
        assert np.array_equal(old.apply_currents(ie,ii),new.apply_currents(ie,ii))
        old.step(spikes,None,.1);new.step(spikes,None,.1)
        for key in ['z','m','r_global','global_state']:
            assert np.array_equal(getattr(old,key),getattr(new,key)),key
    # Strength changes actual current, while source state and G response stay fixed.
    measured=[]
    for scale in [.5,1.,2.]:
        obj=SmoothCurrentSlow(8,18.,cfg,NE=6,global_gain=30.,mode='native',gamma=0.)
        obj.global_reversal=-17.662847938268442;obj.r_global=350.;obj.adaptation_scale=scale
        obj.apply_currents(np.zeros(8),np.zeros(8));obj.step(np.ones(8,bool),None,.1)
        measured.append(float(ETA_REF*obj.m[0]))
        assert not hasattr(obj,'g_k')
        assert obj.global_state > 0
    assert np.allclose(measured,np.array([.5,1.,2.])*3.841)
    x=np.linspace(0,2000,10001);q=target_fraction(x)
    assert q[0]==0 and np.all(np.diff(q)>=0) and np.all((q>=0)&(q<1))
    eps=.001
    slope=30*(target_fraction(350+eps)-target_fraction(350-eps))/(2*eps)
    assert abs(30*target_fraction(350)-15)<1e-12 and abs(slope-.1)<1e-9
    # Exact exponential response, persistence after rate falls, and quiet M decay.
    obj=SmoothCurrentSlow(8,18.,cfg,NE=6,global_gain=30.,mode='native',gamma=0.)
    obj.global_reversal=-17.662847938268442; obj.global_state=.4;obj.r_global=0.
    obj.m[:6]=1000.;obj.apply_currents(np.zeros(8),np.zeros(8));obj.step(np.zeros(8,bool),None,.1)
    assert np.isclose(obj.global_state,.4*np.exp(-.1/500.))
    assert np.allclose(obj.m[:6],1000*np.exp(-.1/5000.))
    # Independent determinant identity with the retained 15-ms filter.
    maxerr=0.
    for _ in range(40):
        par=dict(a=rng.uniform(-3,3),b=rng.uniform(.1,2),c=rng.uniform(.1,2),
                 h=rng.uniform(0,.1),d=rng.uniform(.2,3),p=rng.uniform(0,5),
                 e=rng.uniform(0,5),tau_g=rng.uniform(.1,2),tau_r=.015)
        J=jacobian(**par);s=complex(rng.normal(),rng.normal())
        lhs=np.linalg.det(s*np.eye(4)-J)*par['tau_g']*par['tau_r']
        rhs=characteristic(s,**par)
        maxerr=max(maxerr,abs(lhs-rhs)/max(1,abs(rhs)))
    assert maxerr<1e-12
    # Exact analytic case: a=d=2 => tauG=0.5 crossing if d*LG+LM>d^2.
    roots=fast_filter_hopf_candidates(a=2,d=2,L_G=3,L_M=1)
    assert len(roots)==1 and abs(roots[0]['tau_G_s']-.5)<1e-12
    assert roots[0]['transverse_routh_crossing']
    # Source is the actual Fig6 candidate, not a different fitted network.
    assert p['source_unified_sha256']==sha(u.__file__)
    result=dict(status='PASS',legacy_clip_path_bitwise=True,
                strength_current_jumps_mV=measured,smooth_H_reference_slope=float(slope),
                no_K=True,quiet_decay_and_G_tail=True,
                full_filter_polynomial_relative_error=maxerr,
                mathematical_unit_case_only=roots,
                native_Hopf_identified=False)
    write(ROOT/'equation_qa.json',result)
    print(json.dumps(result))


def worker(name,device):
    os.chdir(MAIN)
    p=prepare();assert read(ROOT/'equation_qa.json')['status']=='PASS'
    job=read(ROOT/'jobs'/f'{name}.json')
    folder=ROOT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return
    # Device is execution metadata, not a biological parameter.
    job['device']=device;write(ROOT/'jobs'/f'{name}.json',job)
    SmoothCurrentSlow.global_tau_ms=job['global_tau_s']*1000
    SmoothCurrentSlow.adaptation_scale=job['adaptation_scale']
    SmoothCurrentSlow.feedback_kind=job['feedback_kind']
    capture0,restore0=u.checkpoint.capture,u.checkpoint.restore_slow
    snapshot_steps={round(t*10000) for t in [8,10,12,14,16,24,32,40,60]} if name==BASELINE else set()
    def capture(**kw):
        state=capture0(**kw);o=kw['slow'];step=int(kw['step'])
        state['unified_current_adaptation']=dict(global_state=o.global_state,model=MODEL)
        # The outer fixed.capture wrapper adds these after this callback returns.
        # Populate them now as well so independently retained snapshots are complete.
        state['termination_mechanism']=dict(r_global=float(o.r_global),phi=o.phi.copy())
        if o.dense_times:
            dest=folder/'dense_readout_chunks';dest.mkdir(exist_ok=True)
            np.savez_compressed(dest/f'{round(o.dense_times[0]*10):010d}_{step:010d}.npz',
                time_ms=o.dense_times,contact_current=o.dense_values)
            o.dense_times.clear();o.dense_values.clear()
        if o.unified_records:
            dest=folder/'adaptation_chunks';dest.mkdir(exist_ok=True);r=np.asarray(o.unified_records)
            np.savez_compressed(dest/f'{round(r[0,0]*10):010d}_{step:010d}.npz',
                time_ms=r[:,0],q=r[:,1],global_s=r[:,2],tau_A_s=r[:,3],
                adaptation_mean_mV=r[:,4],adaptation_max_mV=r[:,5])
            o.unified_records.clear()
        if step in snapshot_steps:
            dest=folder/'state_snapshots';dest.mkdir(exist_ok=True)
            u.base.save_pickle(dest/f't{step/10000:g}s.pkl',
                dict(engine=state,job=job,identity=p['identity'],
                     autonomous_trajectory_state=True,equilibrium=False))
        return state
    def restore(state,obj):
        restore0(state,obj)
        obj.global_state=state['unified_current_adaptation']['global_state']
    u.checkpoint.capture,u.checkpoint.restore_slow=capture,restore
    u.fixed.OUT,u.fixed.prepare,u.fixed.TerminationSlow=ROOT,lambda:p,SmoothCurrentSlow
    try:
        u.fixed.worker(name)
        result=read(folder/'result.json')
        result.update(model=MODEL,display_stop_s=result['end_s'],
                      single_intrinsic_adaptation_state=True,independent_K_absent=True,
                      whole_trajectory_from_t0=True,formal_bifurcation=False)
        result['tracker']['stop_reason']='SIMULATION_HORIZON' if result['status']=='COMPLETE' else result['tracker']['stop_reason']
        write(folder/'result.json',result);write(folder/'progress.json',result)
    except Exception as exc:
        write(folder/'failure.json',dict(error=repr(exc),time=time.time()))
        raise


def analyze(name):
    # Reuse the frozen Fig6 observable definitions without changing thresholds.
    path=ROOT/'source/analyze_topic4_unified_current_adaptation.py'
    spec=importlib.util.spec_from_file_location('frozen_phase_assay',path)
    assay=importlib.util.module_from_spec(spec);spec.loader.exec_module(assay)
    assay.ROOT=ROOT
    result=assay.analyze(name)
    result.update(model=MODEL,parameter_setting=read(ROOT/'jobs'/f'{name}.json'),
                  formal_bifurcation=False)
    write(ROOT/'analysis'/name/'result.json',result)
    if name==BASELINE:
        tests=dict(full_horizon=result['run_status']=='COMPLETE' and result['observed_s']>=60.,
                   initial_brief_events=len(result['initial_brief_events'])>=10,
                   valid_8s_baseline=result['reference_time_s']==8.,
                   high_entry=result['has_high_entry'],autonomous_exit=result['has_autonomous_exit'],
                   sustained_return=result['has_sustained_return'],count_conservation=result['count_conservation'])
        write(ROOT/'baseline_gate.json',dict(status='PASS' if all(tests.values()) else 'FAIL',
            tests=tests,source_baseline_unchanged=True,
            interpretation='Same observables preserved in one paired realization; not statistical equivalence or formal Hopf.'))
    return result


def external_workers():
    import psutil
    result=[]
    for proc in psutil.process_iter(['pid','cmdline']):
        argv=proc.info['cmdline'] or []
        if any('run_topic4_unified_adaptation_entry_switch.py' in arg for arg in argv):
            result.append(proc.pid)
    return result


def supervise():
    p=prepare();assert read(ROOT/'equation_qa.json')['status']=='PASS'
    lock=(ROOT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active={};failed=[]
    (ROOT/'logs').mkdir(exist_ok=True)
    while True:
        for name,(proc,handle,device) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            handle.close();del active[name]
            if code or not (ROOT/'runs'/name/'result.json').exists():
                failed.append(name);continue
            analyze(name)
            subprocess.run([sys.executable,str(Path(__file__).with_name('plot_results.py'))],check=True)
        gate=read(ROOT/'baseline_gate.json') if (ROOT/'baseline_gate.json').exists() else None
        if gate and gate['status']!='PASS':
            write(ROOT/'status.json',dict(status='BASELINE_CHANGED_STOPPED_BEFORE_PLANE',
                  baseline_gate=gate,formal_Hopf_established=False,updated_epoch=time.time()))
            return
        completed=[name for name in p['queue'] if (ROOT/'analysis'/name/'result.json').exists()]
        eligible=p['queue'] if gate else [BASELINE]
        pending=[name for name in eligible if name not in completed and name not in active and name not in failed]
        others=external_workers()
        capacity=min(p['max_workers'],max(0,p['max_external_plus_own_native_workers']-len(others)))
        if failed or time.time()>p['deadline_epoch']-300:
            capacity=0
        while pending and len(active)<capacity:
            name=pending.pop(0)
            device=next((d for d in [0,1] if d not in [x[2] for x in active.values()]),0)
            handle=(ROOT/'logs'/f'{name}.log').open('a')
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                'worker','--name',name,'--device',str(device)],cwd=MAIN,
                stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=(proc,handle,device)
        status='RUNNING_NATIVE_PLANE' if gate else 'VALIDATING_SMOOTH_BASELINE'
        if not active and pending and capacity==0 and not failed:
            status='WAITING_FOR_NATIVE_WORKER_CAPACITY'
        if failed:status='FAILED'
        if len(completed)==len(p['queue']):status='NATIVE_PILOT_COMPLETE_FORMAL_HOPF_PENDING'
        write(ROOT/'status.json',dict(status=status,updated_epoch=time.time(),pid=os.getpid(),
            active={n:dict(pid=x[0].pid,device=x[2]) for n,x in active.items()},
            external_native_workers=others,completed=completed,eligible_pending=pending,
            baseline_gate=gate,failed=failed,total_planned=9,
            formal_Hopf_established=False,
            formal_blocker='Dynamic closure at the same native operating point has not yet been identified.'))
        if not active and (failed or len(completed)==len(p['queue']) or time.time()>p['deadline_epoch']-300):return
        time.sleep(10)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','qa','worker','analyze','supervise'])
    ap.add_argument('--name',default=BASELINE);ap.add_argument('--device',type=int,default=0)
    arg=ap.parse_args()
    if arg.action=='prepare':print(json.dumps(prepare(),ensure_ascii=False))
    elif arg.action=='qa':qa()
    elif arg.action=='worker':worker(arg.name,arg.device)
    elif arg.action=='analyze':analyze(arg.name)
    else:supervise()
