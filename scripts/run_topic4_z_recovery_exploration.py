#!/usr/bin/env python3
"""Bounded native-Z recovery exploration. Recording does not alter the equations."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import copy
import fcntl
import json
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import psutil
import run_topic4_k100_recurrence as previous

matched = previous.matched
fixed = previous.fixed
carrier = previous.carrier
base = previous.base
audit = previous.audit
SOURCE = previous.OUT
OUT = Path('/data/hfosp/topic4_sef_hfo/fig5_z_recovery_exploration_20260916')
SEED = 9108401
GAMMAS = [1/6, 1/3, .5]
STRENGTHS = [.1, 1., 5.]
REGIONS = ['all_E', 'core_A', 'core_B', 'other_E']
BUDGET_KEYS = ['Z_start', 'Z_end', 'recovery_per_s', 'consumption_per_s',
               'net_per_s', 'balance_error', 'fraction_recovery_eligible_end', 'J_end_mV']


def write(path, value):
    audit.old.write(path, value)


class RecoveryBudgetSlow(previous.K100Slow):
    """Integrate positive/negative native dZ at every 0.1 ms, pool every 20 ms."""
    instance = None

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        RecoveryBudgetSlow.instance = self
        self.budget_records = []
        self.recovery_sum = np.zeros(self.NE)
        self.consumption_sum = np.zeros(self.NE)
        self.budget_ms = 0.
        self.budget_start = None
        self.max_balance_error = 0.

    def means(self, values):
        return np.r_[values.mean(), [values[ix].mean() for ix in self.region_groups()]]

    def step(self, spk, labels, dt):
        z = self.z[:self.NE]
        if self.budget_ms == 0:
            self.budget_start = self.means(z)
        target = self._I_I_last[:self.NE] < self.cfg.I_th_EI
        dz = (target.astype(float) - z) * (dt / self.cfg.tau_z)
        self.recovery_sum += np.maximum(dz, 0.)
        self.consumption_sum += np.maximum(-dz, 0.)
        self.budget_ms += dt
        super().step(spk, labels, dt)
        if self._step_index % 200 == 0:
            end = self.means(self.z[:self.NE])
            gain, loss = self.means(self.recovery_sum), self.means(self.consumption_sum)
            error = end - self.budget_start - (gain - loss)
            self.max_balance_error = max(self.max_balance_error, float(np.max(np.abs(error))))
            assert self.max_balance_error < 1e-11, 'Native Z accounting failed'
            seconds = self.budget_ms / 1000.
            row = np.stack([self.budget_start, end, gain/seconds, loss/seconds,
                            (gain-loss)/seconds, error, self.means(target),
                            self.means(self._I_I_last[:self.NE])], axis=-1)
            self.budget_records.append((self._step_index*.1, row))
            self.recovery_sum.fill(0.)
            self.consumption_sum.fill(0.)
            self.budget_ms = 0.


def prepare():
    if (OUT/'protocol.json').exists():
        return base.read(OUT/'protocol.json')
    OUT.mkdir(parents=True, exist_ok=True)
    parent = base.read(SOURCE/'protocol.json')
    template = copy.deepcopy(parent['initial_jobs'][0])
    jobs = []
    # First launch includes both weak/strong feedback and a native repeat anchor.
    settings = [(1/3, 1., 5.), (.5, 5., 5.), (1/6, .1, 5.), (1/6, .175, 2.5)]
    settings += [(g, k, 5.) for g in GAMMAS for k in STRENGTHS if (g,k,5.) not in settings]
    settings += [(1/6, .125, 2.5), (.5, 7.5, 5.)]
    assert len(settings) == 12
    for gamma, k100, tau in settings:
        j = copy.deepcopy(template)
        j.update(name=f'g{gamma:.6g}_k{k100:g}_tau{tau:g}_s{SEED}', gamma=gamma,
                 global_gain=gamma*j['C_R']/(18.-j['global_reversal_mV']),
                 seed=SEED, horizon_s=40., checkpoint_s=2., device=len(jobs)%2,
                 stop_after_second_entry=False, qa=False, stage='initial',
                 **previous.derive(k100, tau))
        j.pop('branch', None)
        jobs.append(j)
    p = {key: copy.deepcopy(parent[key]) for key in [
        'identity', 'source_hashes', 'baseline', 'calibration', 'reference_current_scale',
        'k_parameterization', 'native_Z_policy', 'temporal_rule', 'source_reversal']}
    p.update(status='PROSPECTIVE_NATIVE_Z_RECOVERY_SCREEN', created_epoch=time.time(),
             deadline_epoch=time.time()+8*3600, initial_jobs=jobs, branch_jobs=[],
             max_workers=4, min_available_memory_GiB=70., disk_reserve_GiB=50.,
             source_round=str(SOURCE), producer_sha256=base.sha(carrier.__file__),
             wrapper_sha256=base.sha(fixed.__file__), matched_producer_sha256=base.sha(matched.__file__),
             exploration_sha256=base.sha(__file__), protected_global_authorized=False,
             unchanged_M_Z=parent['unchanged_M_Z'], no_new_M_or_Z_equation=True,
             no_added_Z_recovery=True, no_extra_filter=True, no_external_intervention=True,
             question='Can existing global feedback plus local slow K create a sustained native Z net-recovery window after entry and return the same autonomous trajectory to recurrent brief events?',
             matrix=dict(gamma=GAMMAS, k100=STRENGTHS, tau_K_s=5., seed=SEED,
                         additional_conditions='gamma=1/6,k100=0.125/0.175,tauK=2.5s; gamma=1/2,k100=7.5,tauK=5s'),
             stage_policy='Twelve paired-noise 40s cold starts. At most two conditions with preentry brief events, autonomous exit, and native Z recovery are carried to120s, each with two extra noise seeds. Max16 cold starts; max8h wall time; no further automatic parameter search.',
             recovery_observer='For every native0.1ms Euler update integrate positive and negative dZ separately.20ms reports in all E/coreA/coreB/other: restoration, consumption, net and delta-Z identity. Raw J threshold mask is per cell, not threshold of a population mean.',
             recovery_screen='After first entry, both cores must gain at least0.02 in mean Z over the same1s window. This is a prospective candidate screen, not a biological threshold. Full numeric budgets are retained.',
             acceptance='Preentry recurrent brief events -> native high entry -> autonomous low-activity exit with Z recovery -> recurrent brief events. Reentry is a separate stronger recurrence endpoint. No suppression-only or high-low-high without brief events accepted. Fig5 morphology and human visual review remain required.',
             figure_contract='Latest user image: A four E/I groups and core zooms, B aligned Z and etaM*M, C actual Rest/Interictal/Pre-ictal/Onset plus observed recovery/return, D meanZ/IE/rE with time colorbar right, E square measured gamma/k100 grid. F patient early energy remains pending dense validation; do not replace it by another observable.',
             observation_only=True, human_review='PENDING', adaptive_selected=[])
    # Preserve actual executor dependency identities, and freeze this round's code.
    for f in [__file__, previous.__file__, matched.__file__, fixed.__file__, carrier.__file__, audit.__file__]:
        p['source_hashes'][str(Path(f).resolve())] = base.sha(f)
    impl = OUT/'implementation'
    impl.mkdir(exist_ok=True)
    for f in [__file__, previous.__file__, matched.__file__, fixed.__file__, carrier.__file__, audit.__file__]:
        shutil.copy2(f, impl/Path(f).name)
    shutil.copy2(SOURCE/'geometry.npz', OUT/'geometry.npz')
    for j in jobs:
        write(OUT/'jobs'/f"{j['name']}.json", j)
    write(OUT/'protocol.json', p)
    (OUT/'design.md').write_text('''# 原生 Z 净恢复探索（2026-09-16）

目标是让固定手放双核 Z/M SNN 自主经过：短间期事件、进入发作样高活动、自发终止、Z 净恢复和短事件返回。再次进入另作更强的循环证据。统计单位为同一条件下的独立噪声轨迹；首轮为配对噪声探索，不是独立确认。

原生方程为 tauZ*dZ/dt = 1[J<Ith]-Z，J=(1-gamma)*II+gamma*C_R*R_G。Z 已有恢复区；要检查网络能否自主到达并停留。全局反馈仍由原 Z 门控，局部与全局均参与原来的资源负荷。K 仍是前轮归一化的线性慢钾，k100 与 tauK 分开。Z/M、快网络、阈值场、噪声、连接、反转电位保持前轮身份；不加入10s滤波器、Z供给项或状态重置。

12条首轮轨迹各40s：gamma={1/6,1/3,1/2} × k100={0.1,1,5}，tauK=5s；加gamma=1/6、k100=0.125/0.175、tauK=2.5s两条边界，以及gamma=1/2、k100=7.5、tauK=5s历史退出对照。强K对照即使退出，若没有进入前和返回后的短事件，仍不合格。原参数重复条件用于检验新增观察器没有改变轨迹。

逐0.1ms积分恢复量 max(dZ,0) 与消耗量 max(-dZ,0)，20ms输出全E、两核和核外区域，核验二者之差等于Z真实变化。单细胞先判J阈值再汇总，不能用平均电流代替。候选筛查要求进入后两个核在同一个1s窗均净增Z>=0.02；完整数值和短窗仍保留，不把此筛查阈值当作生理界限。

首轮结束后，最多两个同时具备进入前短事件、自主退出、上述Z恢复窗的条件，原状态延长到120s，并各加两个独立噪声种子。优先已有短事件返回或循环的条件；没有合格者即停止参数扩展。最多16条冷启动轨迹、4并发、8小时墙钟；到时已跑时长记为右截尾，未跑记未启动。已完成短窗、延长与独立种子分开列出，不把续跑算独立样本。

验收仍使用最新版Fig5语义。A原生四组raster和两核放大，B对齐Z/M，C仅标注确实观察到的状态，D保持meanZ/IE/rE，E为当前参数实测格。逐区Z收支作为独立机制附图。F患者早期能量仍需密集采样确认；本轮机制筛查不据此声称患者复现。所有图待作者目视，结果不会自动冻结为正式图。
''')
    return p


def qa():
    p = prepare()
    previous.OUT = OUT
    matched.OUT = OUT
    audit.OUT = OUT
    audit.qa()
    matched.qa()
    cfg = base.old.MZSlowVarsConfig(use_z=True, use_m=True, tau_z=5000.,
            I_th_EI=95.19851312666987, tau_adp=1000., eta_m=.0005)
    rng = np.random.default_rng(160916)
    tests = []
    for gamma in GAMMAS:
        pair = []
        for cls in [matched.MatchedSlow, RecoveryBudgetSlow]:
            cls.C_R=p['reference_current_scale']; cls.feedback_form='conductance'
            cls.sahp_gain=.2; cls.sahp_tau_ms=5000.
            obj=cls(12,18,cfg,NE=10,mode='native',gamma=gamma,
                    global_gain=gamma*cls.C_R/(18+17.662847938268442),global_resource='native_z',phi_jump=0.)
            obj.global_reversal=-17.662847938268442
            obj.voltage=np.zeros(12);obj.z[:10]=.4
            obj.record_regions=False;obj._groups=[np.arange(3),np.arange(3,6),np.arange(6,10)]
            pair.append(obj)
        a,b=pair
        for step in range(1000):
            ie=rng.uniform(0,1200,12)
            # Alternate clear recovery and consumption, including equality to Ith.
            ii=np.full(12,0. if step<400 else 1000.)
            if step>=800:ii[:]=cfg.I_th_EI/(1-gamma)
            for obj in pair:obj.r_global=0.
            sp=rng.random(12)<.15
            assert np.array_equal(a.apply_currents(ie,ii), b.apply_currents(ie,ii))
            if step>=800:
                # Exercise exact equality without a multiply/divide rounding offset.
                for obj in pair:obj._I_I_last[:10]=cfg.I_th_EI
            a.step(sp,None,.1);b.step(sp,None,.1)
            for key in ['z','m','g_k','g_global','phi']:
                assert np.array_equal(getattr(a,key),getattr(b,key)),key
        rec=np.stack([v for _,v in b.budget_records])
        assert np.all(rec[:2,:,4]>0) and np.all(rec[2:,:,4]<0)
        assert np.allclose(rec[0,:,0]+(rec[:,:,2]-rec[:,:,3]).sum(0)*.02,rec[-1,:,1],atol=1e-12)
        tests.append(dict(gamma=gamma, steps=1000, bitwise_native_state=True,
                          maximum_budget_error=b.max_balance_error))
    for path,h in p['source_hashes'].items():assert base.sha(path)==h,path
    write(OUT/'recovery_observer_qa.json',dict(status='PASS',tests=tests,
          exact_threshold_equality_consumes=True, original_Z_M_K_preserved=True,
          native_network_prefix_check='PENDING_FIRST_2S_REPEAT_ANCHOR'))
    print(json.dumps(tests))


def flush(folder, step):
    previous.flush_regional(folder, step)
    obj=RecoveryBudgetSlow.instance
    if not obj or not obj.budget_records:return
    assert obj.budget_ms==0., 'Checkpoint must align with a complete observer window'
    times=np.array([t for t,_ in obj.budget_records])
    values=np.stack([v for _,v in obj.budget_records])
    dest=folder/'z_budget_chunks';dest.mkdir(exist_ok=True)
    path=dest/f'{round((times[0]-20)*10):010d}_{step:010d}.npz'
    tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,time_ms=times,values=values,keys=np.array(BUDGET_KEYS),region_names=np.array(REGIONS))
    tmp.replace(path);obj.budget_records.clear()


def analyze(name):
    folder=OUT/'runs'/name
    row=audit.analyze_folder(folder,OUT/'geometry.npz',sensitivities=False)
    if row is None:return None
    budget=audit.old.load(folder,'z_budget_chunks',keys=['time_ms','values'])
    if not budget:return None
    t=budget['time_ms']/1000.;v=budget['values']
    assert np.max(np.abs(v[:,:,5]))<1e-11
    continuity=np.max(np.abs(v[1:,:,0]-v[:-1,:,1])) if len(v)>1 else 0.
    assert continuity<1e-11
    entry=row['primary']['entries'];max_pair=None;best_time=None;windows=[]
    if entry and len(v)>50:
        # Shared1s window: take the smaller of the two core rises.
        delta=v[50:,:,1]-v[:-50,:,1]
        valid=t[:-50]>=entry[0]['confirmation_s']
        if np.any(valid):
            ix=np.flatnonzero(valid)
            best=ix[np.argmax(np.min(delta[ix,1:3],axis=1))]
            max_pair=float(np.min(delta[best,1:3]));best_time=[float(t[best]),float(t[best+50])]
            mask=valid & np.all(delta[:,1:3]>=.02,axis=1)
            windows=[dict(start_s=float(t[lo]),end_s=float(t[hi-1]+1)) for lo,hi in audit.old.spans(mask)]
    row['Z_recovery']=dict(maximum_balance_error=float(np.max(np.abs(v[:,:,5]))),
        continuity_error=float(continuity),core_recovery_screen=bool(windows),windows=windows,
        maximum_shared_core_1s_rise=max_pair,best_window_s=best_time,
        final_Z=v[-1,:,1],regions=REGIONS,
        integrated_recovery=(v[:,:,2]*.02).sum(0),integrated_consumption=(v[:,:,3]*.02).sum(0))
    pp=row['primary']
    row['candidate_for_extension']=bool(pp['preentry_brief_screen'] and pp['low_activity_exits'] and windows)
    write(OUT/'analysis'/f'{name}.json',row)
    write(folder/'live_status.json',dict(time_s=float(t[-1]),classification=pp['classification'],
          preentry_brief=pp['preentry']['brief_count'],entries=pp['entries'],exits=pp['low_activity_exits'],
          Z=v[-1,:,1],core_recovery_screen=bool(windows),updated_epoch=time.time()))
    return row


def prefix_check(name):
    p=prepare();job=base.read(OUT/'jobs'/f'{name}.json')
    if (job['gamma'],job['k100'],job['sahp_tau_s'],job['seed'])!=(1/6,.1,5.,SEED):return
    if (OUT/'native_prefix_qa.json').exists():return
    a=audit.old.load(OUT/'runs'/name,keys=['spikes_1ms','regions_1ms','raster','Z','M','inputs'])
    b=audit.old.load(SOURCE/'runs/k100_0.1_tau5_s9108401',keys=list(a))
    checks={key:np.array_equal(value,b[key][:len(value)]) for key,value in a.items()}
    assert checks and all(checks.values()),checks
    write(OUT/'native_prefix_qa.json',dict(status='PASS',compared_s=len(a['spikes_1ms'])/1000,
          bitwise=checks,reference=str(SOURCE/'runs/k100_0.1_tau5_s9108401')))


def worker(name):
    p=prepare();assert p['exploration_sha256']==base.sha(__file__)
    assert base.read(OUT/'recovery_observer_qa.json')['status']=='PASS'
    folder=OUT/'runs'/name
    previous.OUT=OUT;matched.OUT=OUT;matched.prepare=lambda:p
    RecoveryBudgetSlow.z_gate_off=False;RecoveryBudgetSlow.k_freeze=False;RecoveryBudgetSlow.record_regions=True
    matched.MatchedSlow=RecoveryBudgetSlow
    sink0=fixed.observation_sink
    def sink_factory(sink,job,deadline):
        full=sink0(sink,job,deadline)
        def observe(step,state):
            try:return full(step,state)
            finally:
                flush(folder,step)
                analyze(name)
                prefix_check(name)
        return observe
    fixed.observation_sink=sink_factory
    try:matched.worker(name)
    finally:fixed.observation_sink=sink0
    result=base.read(folder/'result.json');result['display_stop_s']=result['end_s']
    if result['status']!='CENSORED_WALL_DEADLINE':result['tracker']['stop_reason']='SIMULATION_HORIZON'
    write(folder/'result.json',result);write(folder/'progress.json',result)
    analyze(name)


def select_extensions():
    p=prepare();rows=[]
    for j in p['initial_jobs']:
        if j['stage']!='initial':continue
        path=OUT/'analysis'/f"{j['name']}.json"
        if path.exists():
            row=base.read(path)
            if row['candidate_for_extension']:rows.append(row)
    def priority(r):
        pp=r['primary'];post=pp['latest_postexit']
        return (pp['temporal_loop_pass'], bool(post and audit.qualifies(post)),
                r['Z_recovery']['maximum_shared_core_1s_rise'])
    selected=sorted(rows,key=priority,reverse=True)[:2]
    p['adaptive_selection_done']=True
    p['adaptive_selected']=[r['job']['name'] for r in selected]
    p['selection_epoch']=time.time()
    for row in selected:
        name=row['job']['name'];folder=OUT/'runs'/name
        original=base.read(OUT/'jobs'/f'{name}.json')
        if not (folder/'result.json').exists():continue
        archive=folder/'stage1';archive.mkdir(exist_ok=True)
        shutil.copy2(folder/'result.json',archive/'result.json')
        shutil.copy2(OUT/'analysis'/f'{name}.json',archive/'analysis.json')
        shutil.copy2(OUT/'jobs'/f'{name}.json',archive/'job.json')
        changed=copy.deepcopy(original);changed.update(horizon_s=120.,stage='continuation')
        with (folder/'checkpoint.pkl').open('rb') as h:saved=pickle.load(h)
        assert saved['job']==original
        saved['job']=changed
        base.save_pickle(folder/'checkpoint.pkl',saved)
        (folder/'result.json').unlink()
        write(OUT/'jobs'/f'{name}.json',changed)
        p['initial_jobs']=[changed if j['name']==name else j for j in p['initial_jobs']]
        for seed in [SEED+1,SEED+2]:
            j=copy.deepcopy(changed)
            j.update(name=name.rsplit('_s',1)[0]+f'_s{seed}',seed=seed,stage='independent_noise',
                     independent_noise_seed_of=name,device=len(p['initial_jobs'])%2)
            p['initial_jobs'].append(j);write(OUT/'jobs'/f"{j['name']}.json",j)
    assert len(p['initial_jobs'])<=16
    write(OUT/'protocol.json',p)
    write(OUT/'selection.json',dict(selected=p['adaptive_selected'],rows=selected,
          reason='Prespecified preentry events + autonomous exit + native core Z recovery; ranked by return then recovery. No candidate means no expansion.'))


def supervise():
    p=prepare()
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert base.read(OUT/'recovery_observer_qa.json')['status']=='PASS'
    running={};failures=[];logs=OUT/'logs';logs.mkdir(exist_ok=True)
    while True:
        p=prepare()
        for name,(proc,handle) in list(running.items()):
            if proc.poll() is not None:
                handle.close();del running[name]
                if proc.returncode:failures.append(dict(name=name,exit_code=proc.returncode))
        done=[j['name'] for j in p['initial_jobs'] if (OUT/'runs'/j['name']/'result.json').exists()]
        failed={f['name'] for f in failures}
        pending=[j for j in p['initial_jobs'] if j['name'] not in done and j['name'] not in running and j['name'] not in failed]
        may_start=time.time()<p['deadline_epoch']-900 and not failures
        while pending and len(running)<p['max_workers'] and may_start:
            if psutil.virtual_memory().available/2**30<p['min_available_memory_GiB'] or shutil.disk_usage(OUT).free/2**30<p['disk_reserve_GiB']:break
            j=pending.pop(0);handle=(logs/f"{j['name']}.log").open('a')
            proc=subprocess.Popen([sys.executable,'-u',__file__,'worker','--name',j['name']],stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            running[j['name']]=(proc,handle)
            print('START',j['name'],proc.pid,flush=True)
            time.sleep(2)
        status=dict(updated_epoch=time.time(),pid=os.getpid(),running={n:pr.pid for n,(pr,h) in running.items()},
             queued=[j['name'] for j in pending],finished=done,failures=failures,deadline_epoch=p['deadline_epoch'])
        write(OUT/'status.json',status)
        if not running:
            if failures or not may_start:break
            if not pending:
                if not p.get('adaptive_selection_done'):
                    select_extensions();continue
                break
        time.sleep(15)
    p=prepare()
    unstarted=[j['name'] for j in p['initial_jobs'] if not (OUT/'runs'/j['name']/'result.json').exists()]
    write(OUT/'batch_complete.json',dict(finished_epoch=time.time(),failures=failures,
          missing_results=unstarted,status='FAILED' if failures else ('WALL_BUDGET_EXHAUSTED' if unstarted else 'FINISHED'),
          human_review='PENDING',full_Fig5_acceptance='NOT_ESTABLISHED'))
    renderer=Path(__file__).with_name('report_topic4_z_recovery_exploration.py')
    with (logs/'report.log').open('a') as h:
        code=subprocess.call([sys.executable,str(renderer)],stdout=h,stderr=subprocess.STDOUT)
    write(OUT/'report_status.json',dict(exit_code=code,time=time.time(),human_review='PENDING'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','qa','worker','supervise','analyze'])
    parser.add_argument('--name')
    args=parser.parse_args()
    try:
        if args.action in ['worker','analyze']:globals()[args.action](args.name)
        else:globals()[args.action]()
    except Exception as exc:
        target=OUT/'runs'/args.name/'failure.json' if args.name else OUT/f'{args.action}_failure.json'
        write(target,dict(error=repr(exc),time=time.time()))
        raise
