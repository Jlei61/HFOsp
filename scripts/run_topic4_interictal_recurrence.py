#!/usr/bin/env python3
"""Six bounded native-SNN probes for high -> interictal -> high recurrence."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import copy
import fcntl
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import psutil
import analyze_topic4_interictal_recurrence as audit
import run_topic4_zm_matched_spatial_termination as matched

fixed=matched.fixed
OUT=audit.OUT
PARENT=audit.PARENT


def prepare():
    if (OUT/'protocol.json').exists():
        return fixed.carrier.base.read(OUT/'protocol.json')
    parent=fixed.carrier.base.read(PARENT/'sahp_bracket_round5/protocol.json')
    base=next(j for j in parent['initial_jobs'] if j['sahp_gain']==1.5)
    conditions=[(.5,1.5,1.),(.5,1.5,2.),(1/6,.1,1.),(1/6,.1,5.),(0.,.1,1.),(0.,.1,5.)]
    jobs=[]
    for i,(gamma,gain,tau) in enumerate(conditions):
        j=copy.deepcopy(base)
        j.update(name=f'g{gamma:.6g}_k{gain:g}_tau{tau:g}_s9108401',gamma=gamma,
                 global_gain=gamma*base['C_R']/base['reference_gap_mV'],
                 sahp_gain=gain,sahp_tau_s=tau,horizon_s=60.,checkpoint_s=2.,
                 device=i%2,stop_after_second_entry=False)
        jobs.append(j)
    p=copy.deepcopy(parent)
    p.update(initial_jobs=jobs,created_epoch=time.time(),deadline_epoch=time.time()+3*3600,
             status='PROSPECTIVE_INTERICTAL_RECURRENCE_SCREEN',
             producer_sha256=fixed.carrier.base.sha(fixed.carrier.__file__),
             wrapper_sha256=fixed.carrier.base.sha(fixed.__file__),
             matched_producer_sha256=fixed.carrier.base.sha(matched.__file__),
             recurrence_producer_sha256=fixed.carrier.base.sha(__file__),
             audit_producer_sha256=fixed.carrier.base.sha(audit.__file__),
             max_workers=6,min_available_memory_GiB=80.,disk_reserve_GiB=40.,
             question='In one autonomous trajectory, can high activity terminate, recurrent brief self-limited events return, and high activity recur?',
             accepted_user_target='High -> INTERICTAL -> high. Quiet alone is not interictal return.',
             temporal_rule=audit.RULE,
             high_rule='Same allE10ms >=200Hz for200ms. This is an operational high gate; native raster/fields must establish morphology.',
             short_event_reference='Fixed native Z/M baseline before first high:0.04-0.20s, median0.10s. The20-200ms short-event screen is a model reference, not a clinical HFO label.',
             scope='Six paired-noise cold-start conditions, each at most60s; no automatic new rounds. Full morphology/propagation acceptance is separate from temporal screening.',
             changes='Original Z/M and all fast carrier/geometry/noise fixed. Vary only gamma and extra K gain/tau. tauK1/2s are explicit sensitivity variants, not original Liou parameter values.',
             unchanged_M_Z=dict(eta_M=.0005,tau_M_s=1.,tau_Z_s=5.,threshold=base['threshold']),
             no_state_or_parameter_reset=True,no_new_filter=True,no_extra_state_variable=True,
             observation_stop='Default old high-low-high stop suppressed. Stop at checkpoint >=2s after second high of a verified temporal interictal loop, otherwise60s or wall deadline.',
             scientific_acceptance='Temporal pass requires further native spatial/raster review, baseline brief-event preservation and noise confirmation. Never freeze automatically.',
             source_rounds=[str(PARENT/'sahp_bracket_round5'),str(PARENT/'low_fraction_round6')])
    # Keep the exact previous executable source retrievable after adding the
    # opt-in tau parameter. Old protocols/results retain their original hashes.
    versions=OUT/'implementation';versions.mkdir(parents=True,exist_ok=True)
    for path in [Path(__file__),Path(audit.__file__),Path(matched.__file__),Path(fixed.__file__)]:
        shutil.copy2(path,versions/path.name)
    for j in jobs:
        fixed.carrier.base.write(OUT/'jobs'/f"{j['name']}.json",j)
    shutil.copy2(PARENT/'geometry.npz',OUT/'geometry.npz')
    fixed.carrier.base.write(OUT/'protocol.json',p)
    (OUT/'design.md').write_text('''# 发作样高活动—间期—再次高活动：闭环筛查

本版接受用户明确修订：第一次高活动结束后，必须重新出现多次短、自限的事件，再进入高活动。原“低率/静息窗”只标记退出高活动，不能再命名为恢复间期。

固定40k手放双核、阈值场和原输入，Z/M都开启（ηM=0.0005，τM=1s，τZ=5s）。保留已有全局反馈及逐E细胞慢钾的方程形式，改变新增K的衰减时间是明确的敏感性变体；不替换为原文网络，不加新滤波或状态变量。γ=0两组是移除全局反馈的机制对照。

六个条件：γ=0.5、K=1.5配τK=1/2s；γ=1/6、K=0.1配τK=1/5s；γ=0、K=0.1配τK=1/5s。各60s上限，同拓扑6101、同噪声9108401；只作开发诊断。最多6worker，保留80GiB可用内存，输出写数据盘。到期停止，完成后审阅，不自动扩大范围。

高活动入口仍沿用全E10ms率≥200Hz持续200ms。低活动退出仍沿用旧全E和core判据。间期事件沿用已有完整事件检测：峰率≥20Hz，前后至少20ms低于5Hz；短事件20–200ms，参考原基底40–200ms。一次闭环要求退出确认后、下一次高活动开始前至少5次短事件，事件序列跨越至少2s，且完整有限事件中至少80%为短事件。另报告3事件、300ms上限、5ms分箱和1Hz静息阈值敏感性，不能按结果改主判据。

这是模型时序筛查，不是患者HFO诊断。通过还须原生二维传播、局部与整体raster、发作前短事件保留和另一噪声确认。长静默、单次反弹、长波列、初始化暂态或仅通过数值高率门槛均不能代替这些证据。不因为没有在60s内进入而宣称永不发作。

统一交付连续时间轴的E/I及core raster、原Z/M和新增K、原生空间快照与各条件分类。满足时序闭环的候选再接完整Fig5；患者空间一致性沿用独立检验，不回填到动力学目标。
''')
    return p


def qa():
    p=prepare()
    audit.qa()
    matched.OUT=OUT
    matched.MatchedSlow.sahp_tau_ms=5000.
    matched.qa()
    cls=matched.MatchedSlow
    cls.C_R=p['reference_current_scale'];cls.feedback_form='conductance'
    cfg=fixed.carrier.base.old.MZSlowVarsConfig(use_z=True,use_m=True,tau_z=5000,
                                             I_th_EI=95.19851312666987,tau_adp=1000,eta_m=.0005)
    tested=[]
    for tau in [1.,2.,5.]:
        cls.sahp_tau_ms=tau*1000.;cls.sahp_gain=.1
        obj=cls(10,18,cfg,NE=8,mode='native',gamma=0.,global_gain=0.,global_resource='native_z',phi_jump=0.)
        obj.global_reversal=-17.662847938268442
        obj.voltage=np.full(10,18.);obj.g_k[:]=np.linspace(.1,1,8)
        obj.apply_currents(np.zeros(10),np.zeros(10))
        before=obj.g_k.copy();sp=np.arange(10)%2==0;obj.step(sp,None,.1)
        expected=before*np.exp(-.1/(1000*tau))+sp[:8]*.001
        assert np.array_equal(obj.g_k,expected)
        assert np.all(obj.m[8:]==0) and np.all(obj.z[8:]==1)
        tested.append(tau)
    cls.sahp_tau_ms=5000.;cls.sahp_gain=0.
    for path,sha in p['source_hashes'].items():
        assert fixed.carrier.base.sha(path)==sha,path
    fixed.carrier.base.write(OUT/'kinetic_qa.json',dict(status='PASS',tau_K_s=tested,
              default_old_tau_preserved_s=5.,same_per_spike_increment=True,source_physics_unchanged=True))


def worker(name):
    p=prepare();assert p['recurrence_producer_sha256']==fixed.carrier.base.sha(__file__)
    assert p['audit_producer_sha256']==fixed.carrier.base.sha(audit.__file__)
    assert fixed.carrier.base.read(OUT/'kinetic_qa.json')['status']=='PASS'
    job=fixed.carrier.base.read(OUT/'jobs'/f'{name}.json');folder=OUT/'runs'/name
    sink0=fixed.observation_sink
    def sink_factory(sink,job,deadline):
        full=sink0(sink,job,deadline)
        def observe(step,state):
            full(step,state)
            # The checkpoint is already committed. This observer only changes
            # recording duration after an actual brief-event bridge is found.
            row=audit.analyze_folder(folder,OUT/'geometry.npz',sensitivities=False)
            old=audit.old
            old.write(OUT/'analysis'/f'{name}.json',row)
            hits=[part for part in row['primary']['interhigh_intervals'] if part['temporal_pass']]
            if hits and step*.0001>=hits[0]['second_entry_s']+.2+2:
                old.write(folder/'verified_temporal_endpoint.json',dict(time_s=step*.0001,
                          interval=hits[0],full_model_acceptance=False))
                raise fixed.carrier.Stop()
        return observe
    fixed.observation_sink=sink_factory;matched.OUT=OUT
    try:
        matched.worker(name)
    finally:
        fixed.observation_sink=sink0
    result=fixed.carrier.base.read(folder/'result.json')
    result['display_stop_s']=result['end_s']
    result['tracker']['stop_reason']=('VERIFIED_TEMPORAL_LOOP' if (folder/'verified_temporal_endpoint.json').exists()
                                      else 'WALL_DEADLINE' if result['status']=='CENSORED_WALL_DEADLINE' else 'SIMULATION_HORIZON')
    fixed.carrier.base.write(folder/'result.json',result)
    fixed.carrier.base.write(folder/'progress.json',result)
    row=audit.analyze_folder(folder,OUT/'geometry.npz')
    audit.old.write(OUT/'analysis'/f'{name}.json',row)


def summarize():
    rows=[]
    for job in prepare()['initial_jobs']:
        file=OUT/'analysis'/f"{job['name']}.json"
        if not file.exists():continue
        r=fixed.carrier.base.read(file);primary=r['primary']
        rows.append(dict(name=job['name'],status=r['run_status'],observed_s=primary['observed_s'],
                         classification=primary['classification'],preentry_brief=primary['preentry']['brief_count'],
                         temporal_pass=primary['temporal_loop_pass'],gamma=job['gamma'],K=job['sahp_gain'],tau_K_s=job['sahp_tau_s']))
    fixed.carrier.base.write(OUT/'summary.json',dict(updated_epoch=time.time(),rows=rows,
              scope='Six conditions, same one topology and noise; no independent confirmation yet.',full_Fig5_acceptance='NOT_ESTABLISHED'))
    return rows


def supervise():
    p=prepare();qa()
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending=[j for j in p['initial_jobs'] if not (OUT/'runs'/j['name']/'result.json').exists()]
    running={};failures=[];logs=OUT/'logs';logs.mkdir(exist_ok=True)
    while pending or running:
        for name,(proc,h) in list(running.items()):
            if proc.poll() is not None:
                h.close();del running[name]
                if proc.returncode:failures.append(dict(name=name,exit_code=proc.returncode))
        if failures:pending=[]
        while pending and len(running)<p['max_workers'] and time.time()<p['deadline_epoch']-1800:
            if psutil.virtual_memory().available/2**30<80 or shutil.disk_usage(OUT).free/2**30<40:break
            j=pending.pop(0);h=(logs/f"{j['name']}.log").open('a')
            proc=subprocess.Popen([sys.executable,'-u',__file__,'worker','--name',j['name']],stdout=h,stderr=subprocess.STDOUT)
            running[j['name']]=(proc,h);print('START',j['name'],proc.pid,flush=True)
        rows=summarize()
        fixed.carrier.base.write(OUT/'status.json',dict(updated_epoch=time.time(),
                 running={n:proc.pid for n,(proc,h) in running.items()},queued=[j['name'] for j in pending],
                 completed=sum(r['status']=='COMPLETE' for r in rows),failures=failures,
                 no_new_rounds=True,deadline_epoch=p['deadline_epoch']))
        if time.time()>=p['deadline_epoch']-1800 and pending:
            fixed.carrier.base.write(OUT/'undispatched.json',dict(jobs=pending,reason='dispatch deadline'));pending=[]
        if running or pending:time.sleep(20)
    rows=summarize()
    fixed.carrier.base.write(OUT/'batch_complete.json',dict(updated_epoch=time.time(),rows=rows,failures=failures,
               all_six_complete=len(rows)==6 and all(r['status']=='COMPLETE' for r in rows),
               full_Fig5_acceptance='NOT_ESTABLISHED',next='Scientific review; do not automatically expand or freeze.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','qa','worker','supervise','summarize'])
    parser.add_argument('--name');args=parser.parse_args()
    if args.action=='worker':worker(args.name)
    else:globals()[args.action]()
