"""Equal-duration propagation evidence after radius pilot; no optimization gate."""
import argparse,copy,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import pilot_topic4_core_extent as pilot
from scripts import run_topic4_continuous_core_state_r1 as r1
from src.topic4_streaming_spike_readout import simulate_streaming
rt=pilot.rt
SHORT=pilot.OUT
OUT=Path('/data/hfosp/topic4_sef_hfo/core_extent_long_propagation_20260909')
SCRIPT=Path(__file__).resolve()


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():return
    p=copy.deepcopy(rt.read(SHORT/'plan.json'))
    p.update(output_root=str(OUT),max_parallel=8,duration_ms=90000.,
        training_seeds=[847101,847102,847103],confirmation_seed=847107,
        comparison_seeds=[847101,847102,847103,847104,847105,847106],
        source_sha256=rt.sha(pilot.SCRIPT),runner_sha256=rt.sha(SCRIPT),
        stages=[dict(duration_ms=90000,seeds=[847101,847102,847103]),
                dict(duration_ms=90000,seeds=[847101,847102,847103,847104,847105,847106]),
                dict(duration_ms=180000,seeds=[847101,847102,847103,847104,847105,847106])],
        role='all seeds are paired comparison repeats; no selection/confirmation claim',
        precision=dict(bootstrap_replicates=1000,block_ms=15000,min_pooled_mode_events=40,
                       min_runs_with_five_mode_events=3,participation_delta_CI_width=.12,no_SCL_delta_CI_width=.30),
        scientific_scope='one topology, paired noise/state replays; descriptive model propagation, not patient generalization',
        stop='stop extensions when precision targets reached, or report unresolved after 180s x 6; no automatic local optimization or model freeze')
    p['parent_design']['jobs']=[dict(id=f'ou_dyn{s}',kind='ou',z=None,duration_ms=90000.,dynamics_seed=s,
        coupling_seed=s+100000,state_seed=s+10000) for s in p['comparison_seeds']]
    rt.write(OUT/'plan.json',p)
    (OUT/'execution_plan.md').write_text('''# 延长观察：以传播模式为主要问题

用户在 2026-09-09 授权并行延长，直到能给出可靠结论。原范围批次继续收尾；其尚未启动的按 N>=16 排名的微调已暂停。本轮保留基线与全部六个半径条件，不根据事件数或旧排名淘汰条件。

所有条件共享拓扑 2511、相同空间阈值分位数、配对外源噪声和外加局部 I 状态规律；只改变既定 E core 阈值支持范围。第一阶段为 7 条件 × 3 噪声 × 连续 90 秒。证据不足时增加至 6 条噪声；仍不足则同等延长至 180 秒 × 6 噪声。每个阶段固定观察时长，不以获得第 16 个事件为停止条件。最长轨迹覆盖短轨迹，重复前缀不重复计数。这是单拓扑噪声稳健性比较，不声称已跨网络或患者留出验证。

采用在线读出代替全程稠密 spike 矩阵，保留同规格的完整接触包络、未平滑 2 ms 原生空间帧和状态记录。仿真从零重新跑完整长轨迹；当前外加 I 状态接口不支持 checkpoint，不能把多个冷启动短段拼称为连续轨迹。原 30 秒仅作前缀一致性对照与开发证据。

主要报告：两类各自的 SCL/ICL 参与与缺杆、逐接触点参与误差、平均顺序及原生时序，另看局部持续时间、跨接触招募、模式占比与前后时间段。全部合格事件进入展示，保留物理 runaway 与无事件条件。原冻结损失保留诊断；N<16 不阻止传播分析、不作为本轮条件淘汰门槛。

精度指标预先固定为开发层目标，并非临床阈值：每条件每模式累计至少 40 个事件、至少 3 个噪声各观察到 5 个事件；配对参与误差差值的 95% 时间块/噪声层级 bootstrap 区间宽度 <=0.12，缺 SCL 比例差区间宽度 <=0.30。块长 15 秒，1000 次重采样。这些检查衡量比较的观测支持，不构成两模式机制存在性 gate；未满足只能报不确定。正效应、稳定负效应和模式间取舍都可成为结论，不以阳性结果为停止条件。模式时序与原生过程仍需结合全事件图和 GIF 审阅，自动精度达标不等于科学验收。

最多 8 worker；每进程树预留 8 GiB、保留 60 GiB 机器可用内存；单树超过 16 GiB 或可用内存低于 35 GiB 停派并收尾。流式读出先通过稠密对照测试和真实短程核查。第一阶段完成即自动输出全部条件的图、CSV 和 GIF；后续根据精度自动补足，最长阶段后如仍不确定则明确报告，不无界运行或自动进入 Figure 5。
''')


def unit_path(candidate,seed,duration):
    return OUT/f'duration_{round(duration)}'/'units'/candidate['id']/str(seed)/'workers/trajectory.json'


def worker(candidate_path,seed,duration,canary=False):
    plan=rt.read(OUT/'plan.json')
    if rt.sha(SCRIPT)!=plan['runner_sha256']:raise RuntimeError('long runner changed')
    runroot=OUT/('canary' if canary else f'duration_{round(duration)}')
    runroot.mkdir(parents=True,exist_ok=True)
    plan_local=copy.deepcopy(plan);plan_local['duration_ms']=duration
    rt.write(runroot/'plan.json',plan_local)
    pilot.OUT=runroot
    original_simulate=r1.simulate
    def streaming_simulation(sub,transition,design,job,control,observer=None):
        original_engine=r1.simulate_kick
        def engine(p,net,*args,**kwargs):
            return simulate_streaming(original_engine,p,net,*args,positions=sub.positions_e,montage=sub.montage,**kwargs)
        r1.simulate_kick=engine
        try:return original_simulate(sub,transition,design,job,control,observer)
        finally:r1.simulate_kick=original_engine
    r1.simulate=streaming_simulation
    r1.sheet_activity_movie=lambda spikes,*args,**kwargs:spikes.native()
    r1.snn_event_envelope=lambda spikes,*args,**kwargs:spikes.envelope()
    pilot.run_worker(candidate_path,seed,duration)
    root=runroot/'units'/rt.read(candidate_path)['id']/str(seed)
    rec=rt.read(root/'workers/trajectory.json')
    rec['recorder_contract']='streamed linear contact envelope and identical 2-ms active-neuron bins; no dense spike tape'
    rec['long_runner_sha256']=rt.sha(SCRIPT)
    candidate=rt.read(candidate_path)
    reference=(pilot.PARENT/'workers'/f'ou_dyn{seed}.json' if candidate['id']=='baseline'
               else SHORT/'units'/candidate['id']/str(seed)/'workers/trajectory.json')
    if not canary and reference.exists():
        with np.load(root/'workers/trajectory.npz') as x,np.load(reference.with_suffix('.npz')) as y:
            ns=min(len(x['rate_E']),len(y['rate_E']));nf=min(len(x['sheet_activity_counts']),len(y['sheet_activity_counts']))
            # Dense short recording has a convolution end boundary: omit its last 8 bins.
            env_end=max(0,min(x['contact_envelope'].shape[1],y['contact_envelope'].shape[1])-8)
            prefix=dict(rate_E=np.array_equal(x['rate_E'][:ns],y['rate_E'][:ns]),
                state_z=np.array_equal(x['state_z'][:ns],y['state_z'][:ns]),
                native_movie=np.array_equal(x['sheet_activity_counts'][:nf],y['sheet_activity_counts'][:nf]),
                envelope=np.allclose(x['contact_envelope'][:,:env_end],y['contact_envelope'][:,:env_end],atol=1e-7,rtol=1e-6))
        if not all(prefix.values()):raise RuntimeError(f'long trajectory does not preserve short prefix: {prefix}')
        rec['short_prefix_comparison']=dict(checks=prefix,reference=str(reference),n_steps=ns)
    rt.write(root/'workers/trajectory.json',rec)


def queue(candidates,seeds,duration):
    import psutil
    pending=[];active={};failures=[];plan=rt.read(OUT/'plan.json')
    (OUT/'candidates').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    for c in candidates:
        cp=OUT/'candidates'/(c['id']+'.json');rt.write(cp,c)
        for seed in seeds:
            if not unit_path(c,seed,duration).exists():pending.append((c,cp,seed))
    while pending or active:
        rss={}
        for pid,(proc,c,seed,f) in list(active.items()):
            if proc.poll() is not None:
                f.close();del active[pid]
                if proc.returncode:failures.append(dict(candidate=c['id'],seed=seed,exit_code=proc.returncode))
                continue
            try:
                pr=psutil.Process(pid);tree=[pr]+pr.children(recursive=True)
                rss[pid]=sum(x.memory_info().rss for x in tree if x.is_running())/2**30
                if rss[pid]>16 or rt.available_gib()<35:
                    for x in reversed(tree):x.terminate()
                    failures.append(dict(candidate=c['id'],seed=seed,reason='RESOURCE_GUARD'))
            except psutil.NoSuchProcess:pass
        if failures:pending=[]
        allowance=sum(max(0,8-rss.get(pid,0)) for pid in active)
        while pending and len(active)<plan['max_parallel'] and rt.available_gib()>60+allowance+8:
            c,cp,seed=pending.pop(0);f=(OUT/'logs'/f'{c["id"]}_{seed}_{round(duration)}.log').open('a')
            proc=subprocess.Popen([rt.PYTHON,'-u',str(SCRIPT),'worker','--candidate',str(cp),'--seed',str(seed),'--duration',str(duration)],cwd=ROOT,env=rt.ENV,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            active[proc.pid]=(proc,c,seed,f);allowance+=8
        completed=sum(unit_path(c,s,duration).exists() for c in candidates for s in seeds)
        rt.write(OUT/'status.json',dict(status='RUNNING' if not failures else 'STOPPING_AFTER_FAILURE',duration_ms=duration,n_seeds=len(seeds),n_total=len(candidates)*len(seeds),n_complete=completed,
            active=[dict(pid=pid,candidate=c['id'],seed=s,tree_rss_gib=rss.get(pid)) for pid,(pr,c,s,f) in active.items()],queued=len(pending),failures=failures,available_gib=rt.available_gib(),updated_unix=time.time()))
        if pending or active:time.sleep(10)
    if failures:raise RuntimeError(str(failures))


def controller():
    plan=rt.read(OUT/'plan.json')
    if rt.read(OUT/'canary_audit.json')['status']!='PASS':raise RuntimeError('canary not qualified')
    for stage in plan['stages']:
        queue(plan['candidates'],stage['seeds'],stage['duration_ms'])
        rt.write(OUT/'active_analysis_stage.json',stage)
        rt.write(OUT/'status.json',dict(status='ANALYZING',stage=stage,updated_unix=time.time()))
        result=subprocess.run([rt.PYTHON,str(ROOT/'scripts/analyze_topic4_core_extent_long.py')],cwd=ROOT,env=rt.ENV)
        if result.returncode:raise RuntimeError('long analysis failed')
        audit=rt.read(OUT/'precision_audit.json')
        if audit['precision_targets_met']:
            rt.write(OUT/'status.json',dict(status='PRECISION_TARGETS_MET_PENDING_SCIENTIFIC_REVIEW',stage=stage,updated_unix=time.time()));return
    rt.write(OUT/'status.json',dict(status='LONG_COMPARISON_COMPLETE_WITH_UNRESOLVED_PRECISION',stage=stage,updated_unix=time.time()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','worker','controller']);ap.add_argument('--candidate',type=Path);ap.add_argument('--seed',type=int);ap.add_argument('--duration',type=float,default=90000);ap.add_argument('--canary',action='store_true')
    args=ap.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='worker':worker(args.candidate,args.seed,args.duration,args.canary)
    else:
        try:controller()
        except Exception as exc:
            rt.write(OUT/'status.json',dict(status='FAILED',error=repr(exc),updated_unix=time.time()));raise
