#!/usr/bin/env python3
"""Bounded paired eta-M pilot; same manual substrate and early Z-only refill."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import psutil
import run_topic4_m_parameter_modes as core
import run_topic4_early_z_refill_branches as early

ROOT=core.ROOT
PARENT=core.OUT
OUT=ROOT/'results/topic4_sef_hfo/weaker_M_onset_paired_20260914'
WINDOW=ROOT/'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913'
ETAS=[.005,.0025,.001,0.]
SEEDS=[9108401,9108402]
SOURCE={9108401:WINDOW/'early_Z_lookup_dense_figures/runs/early_z_refill_s9108401',
        9108402:WINDOW/'early_z_refill_branches/runs/early_z_refill_s9108402'}
READOUT_KEYS=['time_ms','spikes_1ms','regions_1ms','field_1ms','raster','slow_time_ms',
              'Z','M','currents','lfp_time_ms','lfp_raw','inputs']


def schedule(tr,rate,sec,rescue=True):
    early.early_tracker_step(tr,rate,sec,rescue)
    if len(tr['entries'])>=2:
        tr['stop_s']=min(tr['stop_s'],tr['entries'][1]['confirmation_s']+2.)


def check_sources():
    p=core.read(PARENT/'protocol.json')
    for path,digest in p['source_hashes'].items():assert core.sha(path)==digest,path
    qa_path=WINDOW/'cuda_ordered_scatter_qa/full_network_qa.json'
    qa=core.read(qa_path)
    assert qa['status']=='PASS' and qa['full_engine_state_bitwise_identical']
    assert qa['all_observations_bitwise_identical'] and core.sha(qa['replacement'])==qa['replacement_sha256']
    return p,qa_path


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():return core.read(OUT/'protocol.json')
    parent,qa=check_sources()
    jobs=[]
    for i,eta in enumerate(ETAS[1:]):
        for si,seed in enumerate(SEEDS):
            jobs.append(dict(name=f'eta{eta:g}_s{seed}',eta_m=eta,tau_M_s=1.,seed=seed,
                eta_index=i+1,tau_index=0,tau_z_ms=5000.,threshold=core.old.THRESHOLD,horizon_s=243.5,
                device=si,first_entry_horizon_s=180.,native_return_observation_s=2.,
                recurrence_observation_s=60.,post_second_confirmation_s=2.))
    refs=[]
    for seed,folder in SOURCE.items():
        r=core.read(folder/'result.json')
        assert r['job']['eta_m']==.005 and r['job']['tau_M_s']==1.
        assert r['M_enabled'] and not r['M_reset'] and r['identity']==parent['identity']
        # Replay the exact endpoint observer to establish identical intervention timing.
        tr=core.fresh_tracker();sec=0.;sample=[]
        for path in sorted((folder/'chunks').glob('*.npz')):
            if '.tmp.' in path.name:continue
            with np.load(path) as a:sample.append(a['spikes_1ms'])
        rates=np.concatenate(sample).reshape(-1,10,2).sum(1)[:,0]/320
        for k,rate in enumerate(rates):
            sec=(k+1)*.01;schedule(tr,float(rate),sec)
            if sec>=tr['stop_s']:break
        for key in ['restore_s','release_s']:
            assert abs(tr[key]-r['tracker'][key])<1e-8,key
        for key in ['entries','recoveries']:
            assert len(tr[key])==len(r['tracker'][key]),key
            for v,w in zip(tr[key],r['tracker'][key]):
                for name,value in v.items():
                    assert value==w[name] if isinstance(value,str) else abs(value-w[name])<1e-8,(key,name)
        refs.append(dict(seed=seed,eta_m=.005,source=str(folder),result_sha256=core.sha(folder/'result.json'),
                         original_end_s=r['end_s'],display_end_s=r['tracker']['entries'][1]['confirmation_s']+2,
                         same_endpoint_and_refill_schedule_verified=True,additional_independent_samples=0))
    p=copy.deepcopy(parent)
    p.update(jobs=jobs,total=6,total_conditions_including_reused=8,eta_M=ETAS,tau_M_s=[1.],
        references=refs,source_protocol=str(PARENT/'protocol.json'),cuda_QA=str(qa),
        status='DEFINED_BEFORE_NEW_RUNS',approval='User accepted four eta values, paired seeds, fixed Z and early refill; 2026-09-14.',
        first_entry_horizon_s=180,native_return_observation_s=2,recurrence_observation_s=60,
        maximum_trajectory_s=243.5,post_second_entry_s=2,max_workers=6,
        intervention='Same as accepted early-refill examples: absent established native recovery, begin 1-s Z refill on next 10-ms step at first confirmation+2s; then release Z. M and fast/noise state never reset.',
        effective_M_zero_control='use_m remains true to observe spike-driven M; eta_m=0 removes its current feedback exactly.',
        new_run_initialization='Each new eta starts at t=0 with original seed, not a checkpoint evolved under eta=.005.',
        sampling='0.1-ms native 20x20 E counts and population counts plus existing virtual contacts, raster and Z/M; committed every0.5s.',
        statistical_unit='One condition/noise realization on a single fixed topology; paired differences by noise seed. n=2, descriptive only.',
        endpoints=['First onset and confirmation','Release-Z to second onset','Recovery confirmation delay',
                   'Finite event count and durations before first entry and between recovery and second entry',
                   'Recovery quiet fraction','Actual M feedback and Z at entry'],
        acceptance='Timing improvement must be present in both paired seeds, with >=2 finite events before first entry and after recovery, established recovery, and second entry. No claim of autonomous return or oscillatory seizure from high rate alone.',
        rendering='Continuous ABC overview, synchronized event and transition magnifications; preserve original scaling and all 15 contacts. Original Fig3C remains unchanged.',
        stop='Six new full trajectories plus two existing controls; two short observer QA replays excluded from scientific n. Automatic analysis and candidates, no adaptive eta expansion or model freeze.',
        producer=str(Path(__file__).resolve()),producer_sha256=core.sha(__file__))
    core.write(OUT/'protocol.json',p)
    (OUT/'jobs').mkdir(exist_ok=True)
    for j in jobs:core.write(OUT/'jobs'/(j['name']+'.json'),j)
    for seed in SEEDS:
        j=dict(jobs[0],name=f'qa_s{seed}',seed=seed,device=SEEDS.index(seed),eta_m=.005,
               horizon_s=1.,qa=True)
        core.write(OUT/'jobs'/(j['name']+'.json'),j)
    (OUT/'geometry.npz').symlink_to((PARENT/'geometry.npz').resolve())
    (OUT/'execution_plan.md').write_text('''# 弱M与两次进入时间的配对试验

问题：继续降低适应反馈是否能缩短第一次进入和补Z后的再次进入，同时保留有限间期事件及恢复段？仅改变ηM；固定手放双核、τM=1秒、τZ=5秒、阈值、原生OU及一次早期Z回填。

ηM=0.005、0.0025、0.001、0，各用9108401与9108402。0.005直接复用已完成的两条轨迹；其余六条各从0时刻初始化。0组仍记录M但反馈严格为0。Z回填不清M，不重置神经元、突触、延迟或随机状态。

首次进入沿用全E≥200Hz持续200ms；确认后2秒仍未建立原生返回时，下一10ms时点开始1秒补Z。恢复定义沿用连续两秒每秒平均率<50Hz且静息(<5Hz)比例≥20%；第二次进入确认后2秒停止，checkpoint最多量化0.5秒。未进入观察180秒；恢复/回填后的再入观察最多60秒；总上限243.5秒。

以噪声种子为配对单位分别报告首次时间、Z释放到再入时间、恢复延迟及前后有限事件数/时长。两条种子均提早、前后各至少2次有限事件并恢复后再入，才列为符合本轮目标的候选；不将这两个种子视为跨网络或患者验证。阴性或未恢复也完整记录，不能为了凑五状态换事件、参数或随机种子。

新增原生0.1ms空间/群体计数避免能量观察混叠；0.5秒保存完整状态。先用0.005的两条1秒数值重放逐项核对已存原始观测，再派发六个新条件。重放不增加科学样本。

图保留连续A/B/C总览，加对应间期事件和首次转变的同步放大窗；缩短等待和改善版式分别评价。高率状态标签不证明临床发作振荡，外部补Z后的返回不等于M自主终止。原生能量增加与减少都保留，原Fig3C不替换。完成后自动分析并生成候选，不自动扩展下一轮。
''')
    return p


def worker(name):
    p=core.read(OUT/'protocol.json');check_sources()
    assert core.sha(__file__)==p['producer_sha256'],'Pilot executor changed during running batch'
    job=core.read(OUT/'jobs'/(name+'.json'));folder=OUT/'runs'/name
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return
    core.OUT=OUT;core.tracker_step=schedule
    from src.topic4_cuda_ordered_scatter import wrap_simulator
    gpu=wrap_simulator(core.old.simulate_kick,device_index=job['device'])
    def simulate(params,net,*args,**kwargs):
        ne=net['NE'];assert ne==32000 and params.dt==.1
        slow=kwargs['slow']
        assert slow.cfg.use_m and slow.cfg.use_z and slow.cfg.eta_m==job['eta_m']
        assert slow.cfg.tau_adp==1000 and slow.cfg.tau_z==5000
        core.write(folder/'applied_configuration.json',dict(eta_m=slow.cfg.eta_m,tau_M_s=1.,
            Z_enabled=slow.cfg.use_z,M_observed=slow.cfg.use_m,M_effective_feedback=job['eta_m']>0,
            tau_Z_s=5.,threshold=slow.cfg.I_th_EI,starting_step=slow._step_index,
            starts_from_own_checkpoint=kwargs.get('resume_state') is not None,device=job['device'],
            neurons=40000,step_ms=.1,topology_identity=p['identity'],source_hashes=p['source_hashes']))
        cells=core.old.spatial_cell_index(net['pos'][:ne],n_grid=20,sheet_l_mm=params.L)
        fields=[];populations=[];observe=kwargs['spike_observer'];original_sink=kwargs['checkpoint_sink']
        refs=dict(zip(original_sink.__code__.co_freevars,original_sink.__closure__))
        assert {'block_start','clear','data','identity','prior_wall','started','tracker'}<=refs.keys()
        def observer(tm,spikes):
            counts=np.bincount(cells[spikes[:ne]],minlength=400).astype(np.uint16)
            fields.append(counts);populations.append([int(counts.sum()),int(spikes[ne:].sum())])
            observe(tm,spikes)
        def sink(k,engine):
            c={n:v.cell_contents for n,v in refs.items()}
            start=c['block_start'];assert 0<k-start<=5000
            data={key:np.asarray(val) for key,val in c['data'].items()}
            raw=np.asarray(fields,dtype=np.uint16);pop=np.asarray(populations,dtype=np.uint16)
            assert len(raw)==len(data['raster'])==k-start
            assert np.array_equal(raw.reshape(-1,10,400).sum(1),data['field_1ms'])
            assert np.array_equal(pop.reshape(-1,10,2).sum(1),data['spikes_1ms'])
            assert np.array_equal(raw.sum(1),pop[:,0])
            for key in ['spikes_1ms','regions_1ms','field_1ms']:data[key]=data[key].astype(np.uint16)
            data.update(field_0p1ms=raw,population_0p1ms=pop,start_step=start,end_step=k)
            chunks=folder/'chunks';chunks.mkdir(exist_ok=True)
            dest=chunks/f'{start:010d}_{k:010d}.npz';tmp=dest.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,**data);tmp.replace(dest)
            tr=c['tracker'];tr['wall_s']=c['prior_wall']+time.time()-c['started']
            core.save_pickle(folder/'checkpoint.pkl',dict(job=job,identity=c['identity'],engine=engine,
                             tracker=tr,restore_from=slow.restore_from))
            refs['block_start'].cell_contents=k;c['clear']();fields.clear();populations.clear()
            core.write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),create_time=psutil.Process().create_time(),
                time_s=k*.0001,job=job,entries=tr['entries'],recoveries=tr['recoveries'],
                restore_s=tr['restore_s'],release_s=tr['release_s'],phase=tr['phase'],
                stop_s=min(job['horizon_s'],tr['stop_s']),Z=float(slow.z[:ne].mean()),
                adaptation_current=float(job['eta_m']*slow.m[:ne].mean()),wall_s=tr['wall_s'],
                raw_spatial_conservation=True))
            if k*.0001>=min(job['horizon_s'],tr['stop_s'])-1e-9:raise core.Stop()
        kwargs['spike_observer']=observer;kwargs['checkpoint_sink']=sink
        return gpu(params,net,*args,**kwargs)
    core.old.simulate_kick=simulate
    try:
        result=core.worker(job)
        result.update(post_second_confirmation_s=2.,native_spatial_bin_ms=.1,
                      external_restore_never_clears_M=True,pilot_protocol=str(OUT/'protocol.json'))
        core.write(folder/'result.json',result)
    except Exception as exc:
        core.write(folder/'failure.json',dict(error=repr(exc),time=time.time(),pid=os.getpid()))
        raise


def validate_qa():
    reports=[]
    for seed in SEEDS:
        folder=OUT/'runs'/f'qa_s{seed}'
        r=core.read(folder/'result.json');assert r['end_s']==1.
        values={k:[] for k in READOUT_KEYS+['field_0p1ms','population_0p1ms']}
        for path in sorted((folder/'chunks').glob('*.npz')):
            with np.load(path) as a:
                for k in values:values[k].append(a[k])
        values={k:np.concatenate(v) for k,v in values.items()}
        path=next(iter(sorted((SOURCE[seed]/'chunks').glob('*.npz'))))
        with np.load(path) as ref:
            for k in READOUT_KEYS:assert np.array_equal(values[k],ref[k][:len(values[k])]),(seed,k)
        high=WINDOW/('early_energy_high_resolution_replay' if seed==9108401 else 'early_energy_high_resolution_replay_seed9108402')
        raw=np.load(high/'field_0p1ms.npy',mmap_mode='r')
        assert np.array_equal(values['field_0p1ms'],raw[:10000])
        reports.append(dict(seed=seed,original_observations_bitwise_equal=READOUT_KEYS,
                            native_0p1ms_field_bitwise_equal=True,one_second=True,independent_samples_added=0))
    core.write(OUT/'qa.json',dict(status='PASS',reports=reports,
               same_baseline_physics_and_observers_verified=True,intervention_schedule_parity='Verified from full original baseline trajectories in prepare'))


def launch(name,children):
    folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'result.json').exists()
    log=(folder/'worker.log').open('a')
    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--name',name],
        cwd=ROOT,env=dict(os.environ,LD_LIBRARY_PATH='/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib'),
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close();children[name]=child


def analyze():
    with (OUT/'analysis.log').open('a') as log:
        rc=subprocess.call([sys.executable,str(ROOT/'scripts/analyze_topic4_weaker_M_onset_pilot.py')],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if rc:raise RuntimeError(f'Automatic analysis failed: {rc}')


def supervise():
    p=prepare();lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    children={};failures=[];last_analysis=set()
    for stage,names in [('QA',[f'qa_s{s}' for s in SEEDS]),('SCIENCE',[j['name'] for j in p['jobs']])]:
        pending=[]
        for name in names:
            folder=OUT/'runs'/name
            if (folder/'result.json').exists():continue
            # Adopt a still-live worker only when its complete command identifies this pilot/job.
            found=False
            for process in psutil.process_iter(['cmdline']):
                cmd=process.info['cmdline'] or []
                if str(Path(__file__).resolve()) in cmd and '--name' in cmd and name in cmd:
                    children[name]=process;found=True;break
            if not found:pending.append(name)
        while pending or any(n in names for n in children):
            for name,child in list(children.items()):
                live=child.poll() is None if isinstance(child,subprocess.Popen) else child.is_running() and child.status()!=psutil.STATUS_ZOMBIE
                if live:continue
                del children[name]
                if not (OUT/'runs'/name/'result.json').exists():failures.append(name)
            available=psutil.virtual_memory().available/2**30
            while pending and not failures and len(children)<(2 if stage=='QA' else 6) and available>64:
                launch(pending.pop(0),children);available-=4
            complete={j['name'] for j in p['jobs'] if (OUT/'runs'/j['name']/'result.json').exists()}
            core.write(OUT/'status.json',dict(status='DRAINING_AFTER_FAILURE' if failures else 'RUNNING',
                stage=stage,pid=os.getpid(),create_time=psutil.Process().create_time(),updated_at=time.time(),
                reused_baselines=2,completed_new=len(complete),total_new=6,running={n:c.pid for n,c in children.items()},
                pending=pending,failed=failures,available_memory_GiB=psutil.virtual_memory().available/2**30))
            if stage=='SCIENCE' and complete!=last_analysis:
                analyze();last_analysis=complete
            if failures and not children:raise RuntimeError(failures)
            time.sleep(10)
        if stage=='QA':validate_qa()
    analyze()
    core.write(OUT/'status.json',dict(status='COMPLETE_PENDING_USER_AND_SCIENTIFIC_REVIEW',updated_at=time.time(),
        reused_baselines=2,completed_new=6,total_new=6,running={},pending=[],failed=[]))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','supervise','worker']);ap.add_argument('--name')
    args=ap.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='worker':worker(args.name)
    else:
        try:supervise()
        except Exception as exc:
            core.write(OUT/'supervisor_failure.json',dict(error=repr(exc),time=time.time(),pid=os.getpid()))
            raise
