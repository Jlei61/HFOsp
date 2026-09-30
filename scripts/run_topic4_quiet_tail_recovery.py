#!/usr/bin/env python3
"""Test K retention during near-silence while preserving the pre-exit dynamics."""
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
import run_topic4_recovery_window as window

SOURCE = window.OUT
OUT = Path('/data/hfosp/topic4_sef_hfo/fig5_quiet_tail_recovery_20260918')
parent, base, budget = window.parent, window.base, window.budget
load, write, rate_data = window.load, window.write, window.rate_data
SEED = window.SEED
OldAnalyze = window.analyze


class QuietTailSlow(parent.RhythmSlow):
    off_tau_ms = 5000.
    retention_below_Hz = 5.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_records = []

    def apply_currents(self, *args, **kwargs):
        value = super().apply_currents(*args, **kwargs)
        if self._step_index % 200 == 0:
            groups = [np.arange(self.NE), *self.region_groups()]
            dist = np.array([[self.z[ix].mean(), *np.quantile(self.z[ix], [.1, .5, .9]),
                              np.mean(self.z[ix] < .7), np.mean(self.z[ix] < .8)] for ix in groups])
            tau = self.off_tau_ms if self.r_global <= self.retention_below_Hz else self.sahp_tau_ms
            self.window_records.append((self._step_index * .1, tau, dist))
        return value

    def step(self, spk, labels, dt):
        # This is the pre-step causal rate, before parent.step adds these spikes.
        original_tau = self.sahp_tau_ms
        if self.r_global <= self.retention_below_Hz:
            self.sahp_tau_ms = self.off_tau_ms
        try:
            super().step(spk, labels, dt)
        finally:
            self.sahp_tau_ms = original_tau


def configure():
    window.OUT = OUT
    window.prepare, window.reference, window.configure = prepare, reference, configure
    window.RecoveryWindowSlow, window.analyze = QuietTailSlow, analyze
    parent.OUT = OUT
    parent.prepare, parent.reference = prepare, reference
    parent.previous.OUT = OUT
    budget.OUT = OUT


def reference(seed):
    return OUT / 'references' / f'native_s{seed}.npz'


def make_job(off, threshold, seed=SEED, stage='initial'):
    job = parent.make_job(15., 40., .5, seed, 0, horizon=120.)
    job.update(name=f'quiet{threshold:g}_off{off:g}_s{seed}', stage=stage,
               off_tau_s=off, on_tau_s=.5, retention_below_Hz=threshold, device=0,
               full_state_snapshots=True,
               mechanism=f'K decays with tau={off:g}s only at causal R_G<={threshold:g}Hz; otherwise tau=0.5s. Spike increment stays0.16*q.')
    return job


def reuse_control(job):
    src = SOURCE / 'runs' / f'off0.5_s{job["seed"]}'
    dest = OUT / 'runs' / job['name']
    dest.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_dir() and path.name.endswith('chunks'):
            shutil.copytree(path, dest / path.name, copy_function=os.link)
    for name in ['rhythm_preservation.json', 'feedback_activation.json', 'applied_configuration.json']:
        if (src / name).exists():
            shutil.copy2(src / name, dest / name)
    result = base.read(src / 'result.json')
    assert result['end_s'] == 120 and result['status'] == 'COMPLETE'
    result.update(job=job, reused_from=str(src))
    write(dest / 'result.json', result)
    write(dest / 'reuse_provenance.json', dict(source=str(src),
          new_independent_realization=False, source_result_sha256=base.sha(src / 'result.json')))


def prepare():
    if (OUT / 'protocol.json').exists():
        return base.read(OUT / 'protocol.json')
    OUT.mkdir(parents=True, exist_ok=True)
    old = base.read(SOURCE / 'protocol.json')
    for path, expected in old['source_hashes'].items():
        assert base.sha(path) == expected, path
    jobs = [make_job(2., 5.), make_job(5., 5.), make_job(10., 5.), make_job(5., 200.)]
    control = make_job(.5, 5., stage='reused_control')
    jobs.append(control)
    shutil.copy2(SOURCE / 'geometry.npz', OUT / 'geometry.npz')
    shutil.copytree(SOURCE / 'references', OUT / 'references')
    for job in jobs:
        write(OUT / 'jobs' / f'{job["name"]}.json', job)
    reuse_control(control)
    protocol = {k: copy.deepcopy(old[k]) for k in
                ['identity', 'source_hashes', 'baseline', 'reference_current_scale',
                 'producer_sha256', 'wrapper_sha256']}
    protocol.update(initial_jobs=jobs, initial_screen_names=[j['name'] for j in jobs[:4]],
        branch_jobs=[], created_epoch=time.time(), deadline_epoch=time.time() + 12*3600,
        wall_budget_hours=12, max_workers=2, device=0,
        min_available_memory_GiB=70., disk_reserve_GiB=50.,
        exploration_sha256=base.sha(window.__file__), new_runner_sha256=base.sha(__file__),
        protected_global_authorized=False, source_round=str(SOURCE),
        confirmation_decided=False, confirmation_selected=None,
        question='Can near-silence-specific K retention sustain natural core-Z recovery and restore native interictal events?',
        authorization='User2026-09-18: continue trying after requiring recovery to the pre-entry Z range and the interictal/exit zoom layout.',
        new_fixed_equation=dict(q='clip((R_G-200)/300,0,1); causal15ms global E rate',
            G='15*q*Z_i; same native-Z load and membrane pathway as preceding round',
            K_increment='0.16*q per E spike, unchanged',
            K_decay='tau=off_tau if pre-step R_G<=retention_below_Hz, otherwise0.5s'),
        unchanged='Native Z/M, substrate, local GABA, noise, K increment, high-rate gate, G strength. No artificial Z/M reset or target-dependent release.',
        experiment='4paired single-seed120s cold starts: quiet5Hz x tails2/5/10s, plus broad200Hz x tail5s. One complete0.5s control reused.',
        confirmation_policy='After all4screens complete, at most1candidate with absolute two-core Z recovery AND sustained native event return receives2new paired seeds120s. No further parameter expansion.',
        acceptance='Exact initial8s native rhythm; entry; autonomous exit; both cores regain their own8s native reference Z; >=10native-like brief events spanning>=5s AFTER that recovery. Also report bothZ>=0.8 duration. Human spatial/Fig5 review pending.',
        recovery_reference='8s same-seed native reference (last stored sample<=8s). Values0.75/0.8 are readouts only, not imposed dynamics or proven bifurcation thresholds.',
        figure_contract='Only the accepted5-row2-column interictal preservation and exit zoom for trajectory display; no three-column overview.',
        human_review='PENDING')
    for path in [__file__, window.__file__]:
        protocol['source_hashes'][str(Path(path).resolve())] = base.sha(path)
    impl = OUT / 'implementation'
    impl.mkdir(exist_ok=True)
    for path in [__file__, window.__file__, parent.__file__]:
        shutil.copy2(path, impl / Path(path).name)
    write(OUT / 'protocol.json', protocol)
    (OUT / 'design.md').write_text('''# 接近静默时保留K尾部，检验Z充分恢复

上一轮的R<=200Hz消退分支覆盖高活动振荡低谷；对照种子在12–44.5秒有22.8%的1ms采样落在此区，最低约29.5Hz。新主条件只在既有15ms因果全E率<=5Hz时延长K消退，避免过早改变高活动动力学。这个全局率判据不自动意味着两核都静默，验收另检查全E与两核的共同低活动。新增状态依赖消退是假设检验，不声称是已验证的生理机制。

4条同噪声9108401冷启动，各120秒：5Hz消退分界配2/5/10秒尾部，以及200Hz配5秒的配对对照。复用既有0.5秒完整120秒对照。所有规律从t=0固定，只有消退分界与低率消退时间变化；q、K每spike增量0.16q、原G15、Z/M、局部抑制、连接和噪声均不变。不根据事件检测切换，不直接回补Z，不把0.75或0.8写进更新方程。

前8秒必须逐位保持原间期spikes、两核/核外计数、raster、空间场、Z/M和输入，失败停止后续派发。用逐步Z收支核查恢复大于消耗；分别报告两核均达到同种子8秒间期Z、均达到0.8的时刻和持续时间。充分恢复筛查要求先进入、再自主退出、两核达到原间期参考值，之后至少10个短事件跨5秒，短事件占比>=80%，时长/间隔/全E峰中位数为同种子参考0.5–2倍。单纯延长静默或Z升高都不通过。

四条初筛全部结束后，最多选一个完整返回候选补两个种子。选择先按是否达到0.8，再按返回短事件数、返回跨度和较短尾部排序；没有完整候选就停止。每条120秒、最多6条新轨迹，2个worker使用GPU0，12小时墙钟，到时检查点收尾并明确截尾。原完整状态每10秒及首次进入/退出后保存。图固定“间期保留及退出放大”版式，右窗覆盖实际退出、恢复及返回，缺退出者标明；不画三列总览。最终停在科学和人工图审阅。
''')
    return protocol


def qa():
    prepare()
    cfg = base.old.MZSlowVarsConfig(use_z=True, use_m=True, tau_z=5000.,
           I_th_EI=95.19851312666987, tau_adp=1000., eta_m=.0005)
    def make(cls, threshold=5., off=5.):
        cls.C_R=0.; cls.feedback_form='conductance'; cls.sahp_gain=16.; cls.sahp_tau_ms=500.
        cls.off_tau_ms=off*1000; cls.retention_below_Hz=threshold
        obj=cls(12,18,cfg,NE=10,mode='native',gamma=0.,global_gain=15.,global_resource='native_z',phi_jump=0.)
        obj._groups=[np.arange(3),np.arange(3,6),np.arange(6,10)]
        obj.voltage=np.full(12,5.); obj.global_reversal=parent.EG
        return obj
    rng=np.random.default_rng(18091801)
    a,b=make(parent.RhythmSlow),make(QuietTailSlow)
    for i in range(1200):
        a.r_global=b.r_global=[5.001,20.,199.,200.,350.,500.][i%6]
        ie,ii=rng.uniform(0,1800,(2,12));sp=rng.random(12)<.04
        assert np.array_equal(a.apply_currents(ie,ii),b.apply_currents(ie,ii))
        a.step(sp,None,.1);b.step(sp,None,.1)
        for key in ['z','m','g_k','g_global']:
            assert np.array_equal(getattr(a,key),getattr(b,key)),key
    for off in [2.,5.,10.]:
        b=make(QuietTailSlow,off=off)
        for rate in [0.,4.999,5.,5.001,199.,201.,500.]:
            b.r_global=rate;b.g_k[:]=2.
            b.apply_currents(np.ones(12)*100,np.ones(12)*10);q=b.gate
            b.step(np.ones(12,bool),None,.1)
            tau=off*1000 if rate<=5 else 500
            assert np.array_equal(b.g_k,np.full(10,2*np.exp(-.1/tau)+.16*q))
        b.g_k[:]=0.;b.r_global=4.
        b.apply_currents(np.ones(12),np.ones(12));b.step(np.ones(12,bool),None,.1)
        assert not b.uses_shunt() and np.all(b.g_k==0)
    write(OUT/'mechanism_qa.json',dict(status='PASS',high_phase_bitwise_1200_steps=True,
          low_rate_boundary_gain_and_decay=True,zero_K_low_rate_invariance=True,
          no_added_physical_state=True,
          native_Z_budget_max_error=max(a.max_balance_error,b.max_balance_error)))
    print('PASS: unchanged active-phase updates, quiet-tail boundaries, native Z balance',flush=True)


def analyze(name):
    configure()
    row=OldAnalyze(name)
    if row is None:return None
    folder=OUT/'runs'/name
    d=load(folder,keys=['slow_time_ms','Z']);t=d['slow_time_ms']/1000;cores=d['Z'][:,[5,6]]
    with np.load(reference(row['job']['seed'])) as ref:
        i=np.searchsorted(ref['slow_time_ms'],8000.,side='right')-1
        baseline=ref['Z'][i,[5,6]];reference_time=float(ref['slow_time_ms'][i]/1000)
    rr=rate_data(folder);details=[]
    for ep in row['recovery_window']['episodes']:
        lo=ep['exit']['start_s'];hi=ep['interval']['window_s'][1]
        mask=(t>=lo)&(t<hi);ids=np.flatnonzero(mask)
        if not len(ids):continue
        z=cores[mask];hits=ids[np.all(z>=baseline,axis=1)]
        hit=float(t[hits[0]]) if len(hits) else None
        strong=ids[np.all(z>=.8,axis=1)]
        after=window.audit.interval_events(row['primary']['events'],max(ep['exit']['confirmation_s'],hit),hi) if hit is not None else None
        features=window.event_features(after['brief_events'],rr) if after else {}
        reference_features=row['recovery_window']['reference_features']
        ratios={k:features.get(k)/reference_features[k] if features.get(k) is not None and reference_features[k] else None
                for k in ['duration_ms','interval_ms','peak_Hz']}
        matched=all(v is not None and .5<=v<=2 for v in ratios.values())
        passed=bool(after and window.audit.qualifies(after,minimum_n=10,minimum_span=5.) and matched)
        details.append(dict(exit_start_s=lo,window_end_s=hi,
            peak_shared_core_Z=float(z.min(1).max()),first_both_at_reference_s=hit,
            first_both_above0p8_s=float(t[strong[0]]) if len(strong) else None,
            time_both_above0p8_s=float(len(strong)*.02),
            longest_both_above0p8_s=max([(b-a)*.02 for a,b in window.audit.old.spans(np.all(z>=.8,axis=1))],default=0.),
            post_absolute_recovery_events=after,post_reference_ratios=ratios,
            sustained_return_after_absolute_recovery=passed))
    success=bool(row['native_rhythm']['preservation']['status']=='PASS' and row['native_rhythm']['strict_pre_pass']
                 and any(e['sustained_return_after_absolute_recovery'] for e in details))
    row['absolute_Z_recovery']=dict(reference_time_s=reference_time,reference_core_Z=baseline,
                                  episodes=details,sustained_return_screen=success)
    row['full_sequence_screen']=success
    if success:row['classification']='ABSOLUTE_Z_AND_NATIVE_RETURN_CANDIDATE'
    elif any(e['first_both_at_reference_s'] is not None for e in details):row['classification']='Z_REFERENCE_REGAINED_WITHOUT_NATIVE_RETURN'
    write(OUT/'analysis'/f'{name}.json',row)
    live=base.read(folder/'live_status.json');live.update(absolute_Z_return=success,
         both_reference_reached=any(e['first_both_at_reference_s'] is not None for e in details),classification=row['classification'])
    write(folder/'live_status.json',live)
    return row


def worker(name):
    p=prepare();assert base.sha(__file__)==p['new_runner_sha256']
    job=base.read(OUT/'jobs'/f'{name}.json')
    QuietTailSlow.retention_below_Hz=job['retention_below_Hz']
    configure()
    window.worker(name)


def report():
    log=OUT/'logs';log.mkdir(exist_ok=True)
    with (log/'report.log').open('a') as f:
        code=subprocess.call([sys.executable,str(Path(__file__).with_name('report_topic4_quiet_tail_recovery.py'))],stdout=f,stderr=subprocess.STDOUT)
    write(OUT/'report_status.json',dict(exit_code=code,time=time.time(),human_review='PENDING'))


def supervise():
    p=prepare();lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert base.read(OUT/'mechanism_qa.json')['status']=='PASS'
    if 'launched_epoch' not in p:
        p['launched_epoch']=time.time();p['deadline_epoch']=time.time()+12*3600;write(OUT/'protocol.json',p)
    running={};failures=[];last_report=-1;(OUT/'logs').mkdir(exist_ok=True)
    while True:
        for name,(proc,f) in list(running.items()):
            if proc.poll() is not None:
                f.close();del running[name]
                if proc.returncode:failures.append(dict(name=name,exit_code=proc.returncode))
        done=[j['name'] for j in p['initial_jobs'] if (OUT/'runs'/j['name']/'result.json').exists()]
        if not p['confirmation_decided'] and all(n in done and n not in running for n in p['initial_screen_names']):
            rows=[base.read(OUT/'analysis'/f'{n}.json') for n in p['initial_screen_names']]
            candidates=[r for r in rows if r['run_status']=='COMPLETE' and r['full_sequence_screen']]
            p['confirmation_decided']=True
            if candidates and not failures and time.time()<p['deadline_epoch']-3600:
                def rank(r):
                    eps=[e for e in r['absolute_Z_recovery']['episodes'] if e['sustained_return_after_absolute_recovery']]
                    return max((e['first_both_above0p8_s'] is not None,
                        e['post_absolute_recovery_events']['brief_count'],
                        e['post_absolute_recovery_events']['event_span_s'],-r['job']['off_tau_s']) for e in eps)
                selected=max(candidates,key=rank);p['confirmation_selected']=selected['job']['name']
                for seed in [SEED+1,SEED+2]:
                    j=make_job(selected['job']['off_tau_s'],selected['job']['retention_below_Hz'],seed,'confirmation')
                    p['initial_jobs'].append(j);write(OUT/'jobs'/f'{j["name"]}.json',j)
            write(OUT/'protocol.json',p)
        failed={f['name'] for f in failures}
        pending=[j for j in p['initial_jobs'] if j['name'] not in done and j['name'] not in running and j['name'] not in failed]
        can_start=time.time()<p['deadline_epoch']-600 and not failures
        while pending and len(running)<p['max_workers'] and can_start:
            if psutil.virtual_memory().available/2**30<70 or shutil.disk_usage(OUT).free/2**30<50:break
            j=pending.pop(0);f=(OUT/'logs'/f'{j["name"]}.log').open('a')
            proc=subprocess.Popen([sys.executable,'-u',__file__,'worker','--name',j['name']],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            running[j['name']]=(proc,f);print('START',j['name'],proc.pid,flush=True);time.sleep(2)
        write(OUT/'status.json',dict(updated_epoch=time.time(),pid=os.getpid(),
            running={n:pr.pid for n,(pr,f) in running.items()},queued=[j['name'] for j in pending],
            finished=done,failures=failures,deadline_epoch=p['deadline_epoch'],confirmation_selected=p['confirmation_selected']))
        if len(done)!=last_report:
            for name in done:
                if name not in running and not (OUT/'analysis'/f'{name}.json').exists():analyze(name)
            report();last_report=len(done)
        if not running and (not pending or failures or not can_start):break
        time.sleep(15)
    missing=[j['name'] for j in p['initial_jobs'] if not (OUT/'runs'/j['name']/'result.json').exists()]
    censored=[n for n in done if base.read(OUT/'runs'/n/'result.json')['end_s']<120]
    write(OUT/'batch_complete.json',dict(status='FAILED' if failures else 'CENSORED' if missing or censored else 'FINISHED',
         finished_epoch=time.time(),failures=failures,missing_results=missing,censored=censored,human_review='PENDING'))
    report()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','qa','worker','supervise','analyze'])
    ap.add_argument('--name');args=ap.parse_args()
    try:
        globals()[args.action](args.name) if args.action in ['worker','analyze'] else globals()[args.action]()
    except Exception as exc:
        dest=OUT/'runs'/args.name/'failure.json' if args.name else OUT/f'{args.action}_failure.json'
        write(dest,dict(error=repr(exc),time=time.time()));raise
