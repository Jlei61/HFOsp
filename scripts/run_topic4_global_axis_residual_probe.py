"""Six bounded 20s global-axis probes; existing physical executor stays unchanged."""
from pathlib import Path
import argparse,copy,fcntl,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import run_topic4_shape_output_response as base
from scripts.topic4_phase_memory_reserve import reserve_gib
rt=base.rt;SCRIPT=Path(__file__).resolve()
OUT=Path('/data/hfosp/topic4_sef_hfo/global_axis_residual_probe_20260913')
FOLLOW=Path('/data/hfosp/topic4_sef_hfo/core_recruitment_tradeoff_followup_20260912')
N=Path('/data/hfosp/topic4_sef_hfo/core_multiseed_response_curves_20260913')
WINDOW=Path('/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/window.json')
ANCHOR='up3__circle__EE_core_to_out_scale_1.25__x_minus075'


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():return rt.read(OUT/'plan.json')
    for d in ['candidates','logs','analysis/units','analysis/figures','confirmation','rotation','global_graph_cache']:(OUT/d).mkdir(parents=True,exist_ok=True)
    parent=rt.read(FOLLOW/'candidates'/f'{ANCHOR}.json');inherited=rt.read(N/'plan.json');cs=[]
    for offset in [0.,-15.,15.]:
        c=copy.deepcopy(parent);c.update(id=f'global_axis_{offset:+g}',stage='response',comparison='global_axis_+0',display_name=f'左移圆核｜全局EE轴偏移{offset:+g}°',contrast='global_EE_axis')
        c['parameters']['EE_angle_offset_deg']=offset;cs.append(c);rt.write(OUT/'candidates'/f'{c["id"]}.json',c)
    plan=dict(schema='topic4.global_axis_residual_probe.v1',physics=inherited['physics'],source_snapshot=inherited['source_snapshot'],
        runaway=inherited['runaway'],analysis=dict(burnin_ms=1500.,primary_min_events_per_run=16),candidates=cs,topology_seeds=[2511],seeds=[847101,847102],duration_ms=20000.,
        budget=dict(formal_units=6,canary_units=3,canary_duration_ms=500.),dispatch_deadline_unix=rt.read(WINDOW)['review_due_unix'],
        question='Does global EE connection orientation change fixed-contact TB timing where local-core response has not?',
        baseline_source=str(FOLLOW/'candidates'/f'{ANCHOR}.json'),baseline_source_sha256=rt.sha(FOLLOW/'candidates'/f'{ANCHOR}.json'),
        interpretation='Same neuron positions, lowering-only thresholds, baseline topology seed and noise seed; EE resampling changes actual edges and distance delays. This is not an identical physical graph across angles. Baseline 20s runs replay prefixes of historical 60s runs, not independent replication.',
        unchanged='EE perpendicular multiplier1.5, parallel1, all pathway multipliers, core geometry/thresholds, noise law, Z/M/spatialOU off, GABA18ms, observer/eligibility and frozen loss',
        evidence=[str(Path('/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913')/n) for n in ['parent_confirmation_tb_support.csv','rod_lag_participation_audit_0206/scientific_note.md','historical_global_geometry_response.csv']],
        selector='No score-based selection or early scientific pass gate; complete all dispatched conditions irrespective of propagation quality. Stop new dispatch at8h deadline; let active trajectories finish.',
        resources=dict(max_workers=2,growth_per_tree_GiB=9,host_reserve_GiB=40,tree_hard_limit_GiB=18,host_hard_floor_GiB=30))
    rt.write(OUT/'plan.json',plan);rt.write(OUT/'confirmation/frozen_networks.json',dict(replication_networks={}))
    rt.write(OUT/'status.json',dict(status='PREPARED_APPLICATION_PENDING',formal_budget=6,created_unix=time.time()))
    return plan


def setup():
    base.OUT=OUT;base.SCRIPT=SCRIPT;base.configure()


def shared_growth(own_pids):
    import psutil
    rows=[]
    for name in [FOLLOW,N]:
        if not (name/'status.json').exists():continue
        state=rt.read(name/'status.json')
        if state.get('failures') or 'FAILURE' in state.get('status',''):raise RuntimeError(f'predecessor failure: {name.name}')
        for row in state.get('active',[]):
            try:
                p=psutil.Process(row['pid']);cmd=p.cmdline()
                if p.pid in own_pids or p.status()==psutil.STATUS_ZOMBIE or 'worker' not in cmd:continue
                if not any(t in ' '.join(cmd) for t in ['run_topic4_recruitment_tradeoff_followup.py','run_topic4_multiseed_response_curves.py']):continue
                rss=sum(v.memory_info().rss for v in [p]+p.children(recursive=True))/2**30
                rows.append(dict(pid=p.pid,rss_gib=rss,reserve_gib=reserve_gib(name,state.get('stage','response'),row['unit'])))
            except psutil.NoSuchProcess:pass
    unique={r['pid']:r for r in rows}
    return sum(max(0,r['reserve_gib']-r['rss_gib']) for r in unique.values()),list(unique.values())


class AdoptedWorker:
    """Preserve a running physical worker across a scheduler-only handoff."""
    def __init__(self,row,stage):
        self.pid=int(row['pid']);self.row=row;self.stage=stage;self.returncode=None
    def poll(self):
        import psutil
        try:
            p=psutil.Process(self.pid)
            if abs(p.create_time()-self.row['process_created_unix'])>1e-3:
                raise RuntimeError('adopted PID identity changed')
            if p.status()!=psutil.STATUS_ZOMBIE:return None
        except psutil.NoSuchProcess:pass
        self.returncode=0 if base.base.complete(base.result_path(self.stage,tuple(self.row['unit']))) else -1
        return self.returncode


def queue(stage,units,duration,limit,deadline):
    import psutil
    pending=[];active={};failures=[]
    for u in units:
        p=base.result_path(stage,u)
        if p.exists() and not base.base.complete(p):raise RuntimeError(f'invalid existing result {p}')
        if not p.exists():pending.append(u)
    recovery=OUT/'preserved_active_workers.json'
    if recovery.exists():
        adopted=[]
        for row in rt.read(recovery)['workers']:
            if row['stage']!=stage:continue
            u=tuple(row['unit'])
            if u not in units:raise RuntimeError('preserved unit outside fixed budget')
            if base.base.complete(base.result_path(stage,u)):continue
            p=psutil.Process(row['pid']);cmd=p.cmdline()
            if (str(SCRIPT) not in cmd or 'worker' not in cmd or
                cmd[cmd.index('--candidate')+1]!=u[0] or
                int(cmd[cmd.index('--topology')+1])!=u[1] or
                int(cmd[cmd.index('--seed')+1])!=u[2] or
                abs(p.create_time()-row['process_created_unix'])>1e-3):
                raise RuntimeError('preserved worker identity mismatch')
            if u in pending:pending.remove(u)
            active[p.pid]=(AdoptedWorker(row,stage),u,Path(row['log']).open('a'))
            adopted.append(dict(pid=p.pid,unit=u))
        if adopted:rt.write(OUT/'worker_adoption.json',dict(adopted=adopted,physical_workers_restarted=0,created_unix=time.time(),exit_code_semantics='non-child OS exit code unavailable; complete trajectory required'))
    while pending or active:
        rss={}
        for pid,(proc,u,log) in list(active.items()):
            if proc.poll() is not None:
                log.close();del active[pid]
                if proc.returncode or not base.base.complete(base.result_path(stage,u)):failures.append(dict(unit=u,exit=proc.returncode))
                continue
            try:
                tree=[psutil.Process(pid)]+psutil.Process(pid).children(recursive=True);rss[pid]=sum(v.memory_info().rss for v in tree)/2**30
                if rss[pid]>18 or rt.available_gib()<30:
                    for p in reversed(tree):p.terminate()
                    failures.append(dict(unit=u,error='RESOURCE_GUARD',rss_gib=rss[pid]))
            except psutil.NoSuchProcess:pass
        try:growth,other=shared_growth(set(active))
        except RuntimeError as e:
            growth,other=0.,[]
            if not any(x.get('error')==str(e) for x in failures):failures.append(dict(error=str(e)))
        required=40+growth+sum(max(0,reserve_gib(OUT,stage,u)-rss.get(pid,0)) for pid,(_,u,_) in active.items())+9
        expired=time.time()>=deadline
        if failures:pending=[]
        if pending and not expired and len(active)<limit and rt.available_gib()>required:
            u=pending.pop(0);log=(OUT/'logs'/f'{stage}_{u[0]}_{u[1]}_{u[2]}.log').open('a')
            args=[rt.PYTHON,'-u',str(SCRIPT),'worker','--stage',stage,'--candidate',u[0],'--topology',str(u[1]),'--seed',str(u[2]),'--duration',str(duration)]
            proc=subprocess.Popen(args,cwd=ROOT,env=rt.ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[proc.pid]=(proc,u,log)
            with (OUT/'dispatch.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,unit=u,pid=proc.pid,started_unix=time.time(),required_gib=required,available_gib=rt.available_gib()))+'\n')
        rt.write(OUT/'status.json',dict(status='WINDOW_ELAPSED_DRAINING' if expired and active else 'WINDOW_ELAPSED_NOT_DISPATCHED' if expired and pending else 'ENGINEERING_FAILURE_DRAINING' if failures else stage.upper()+'_RUNNING' if active else 'WAITING_FOR_SHARED_RESOURCES',stage=stage,total=len(units),
            complete=sum(base.result_path(stage,u).exists() for u in units),pending=pending,active=[dict(pid=pid,unit=u,rss_gib=rss.get(pid)) for pid,(_,u,_) in active.items()],shared_active=other,available_gib=rt.available_gib(),required_gib=required,failures=failures,updated_unix=time.time()))
        if expired and not active:return False
        time.sleep(10)
    if failures:raise RuntimeError(json.dumps(failures))
    return True


def validate(plan):
    for c in plan['candidates']:
        path=base.result_path('canary',(c['id'],2511,847101));r=rt.read(path);ap=rt.read(path.parents[1]/'applied_physics.json')
        assert r['actual_duration_ms']==500 and r['maximum_outside_rate_deviation']==0
        assert ap['threshold']['n_raised']==0 and ap['candidate']['parameters']==c['parameters']
        assert abs(ap['graph']['kernel']['theta_deg']-(ap['reference_kernel']['theta_deg']+c['parameters']['EE_angle_offset_deg']))<1e-9
        assert ap['input']['ZM']=='off' and ap['input']['spatial_ou']=='off' and ap['input']['core_ou_correlation']==0
    control=base.result_path('canary',('global_axis_+0',2511,847101));old=FOLLOW/'response/units'/ANCHOR/'2511_847101/workers/trajectory.json'
    assert rt.read(control)['static_array_identity']==rt.read(old)['static_array_identity']
    with np.load(control.with_suffix('.npz')) as a,np.load(old.with_suffix('.npz')) as b:
        check={k:bool(np.array_equal(a[k],b[k][:len(a[k])])) for k in ['sheet_activity_counts','trace_coreAE_spikes','trace_coreBE_spikes','core_ou_mixture_values']}
    assert all(check.values()),check
    rt.write(OUT/'application_validation.json',dict(status='PASS',checks=check,scope='500ms static and raw-dynamic baseline prefix plus actual global-axis application; not a propagation gate'))


def controller():
    plan=prepare();setup()
    with (OUT/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for p,h in plan['source_snapshot'].items():
            if rt.sha(p)!=h:raise RuntimeError(f'frozen physical source changed: {p}')
        canary=[(c['id'],2511,847101) for c in plan['candidates']]
        if not queue('canary',canary,500.,1,plan['dispatch_deadline_unix']):return
        validate(plan)
        import psutil
        processes=rt.read(OUT/'analysis_processes.json') if (OUT/'analysis_processes.json').exists() else []
        def alive(entry):
            try:
                p=psutil.Process(entry['pid']);cmd=p.cmdline()
                return p.status()!=psutil.STATUS_ZOMBIE and str(SCRIPT) in cmd
            except psutil.Error:return False
        for args,role in [(['observer'],'observer'),(['rotation','--gpu','0'],'rotation0'),(['rotation','--gpu','1'],'rotation1')]:
            if any(x['role']==role and alive(x) for x in processes):continue
            env=dict(rt.ENV)
            if role.startswith('rotation'):env['CUDA_VISIBLE_DEVICES']='0,1'
            log=(OUT/'logs'/f'{role}.log').open('a');p=subprocess.Popen([rt.PYTHON,'-u',str(SCRIPT),*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();processes=[x for x in processes if x['role']!=role]+[dict(role=role,pid=p.pid)]
        rt.write(OUT/'analysis_processes.json',processes)
        units=[tuple(u) for u in plan.get('formal_order',[(c['id'],2511,s) for s in plan['seeds'] for c in plan['candidates']])]
        expected=[(c['id'],2511,s) for c in plan['candidates'] for s in plan['seeds']]
        if sorted(units)!=sorted(expected):raise RuntimeError('formal dispatch identities do not match the six-unit budget')
        complete=queue('response',units,20000.,2,plan['dispatch_deadline_unix'])
        rt.write(OUT/'simulation_complete.json',dict(all_six_complete=complete,completed=sum(base.result_path('response',u).exists() for u in units),formal_budget=6,updated_unix=time.time()))


def observer():
    from scripts import analyze_topic4_global_axis_residual_probe as an
    an.observer()


def rotation(gpu):
    import cupy as cp
    from scripts import analyze_topic4_rotation_response as rot
    rot.OUT=OUT;cp.cuda.Device(gpu).use()
    while True:
        paths=sorted((OUT/'response').glob('units/*/*/workers/trajectory.json'));todo=[]
        for p in paths:
            key=hashlib.sha256(str(p).encode()).hexdigest()
            if int(key,16)%2==gpu and rt.read(p)['actual_duration_ms']>=20000 and not (OUT/'rotation'/key[:20]/'result.json').exists():todo.append(p)
        if todo:
            rot.analyze(todo[0],gpu,cp);cp.get_default_memory_pool().free_all_blocks();continue
        done=(OUT/'simulation_complete.json').exists()
        rt.write(OUT/'rotation'/f'gpu{gpu}_status.json',dict(status='COMPLETE' if done else 'WAITING_FOR_TRAJECTORIES',updated_unix=time.time()))
        if done:return
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','controller','worker','observer','rotation']);p.add_argument('--stage');p.add_argument('--candidate');p.add_argument('--topology',type=int);p.add_argument('--seed',type=int);p.add_argument('--duration',type=float);p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.action=='prepare':print(prepare()['budget'])
    elif a.action=='controller':
        try:controller()
        except Exception as e:
            rt.write(OUT/'status.json',dict(status='ENGINEERING_FAILURE',error=repr(e),updated_unix=time.time()));raise
    elif a.action=='worker':setup();base.run.worker(a.stage,a.candidate,a.topology,a.seed,a.duration)
    elif a.action=='observer':observer()
    else:rotation(a.gpu)
