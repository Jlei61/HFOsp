"""Resource-aware overnight orchestration; preserves already-running physical units."""
from pathlib import Path
import argparse,copy,json,math,os,signal,subprocess,sys,time
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
from scripts import run_topic4_geometry_threshold_refinement as refine
rt=refine.rt
NIGHT=refine.OUT/'overnight';SCRIPT=Path(__file__).resolve()
SOURCES={'refinement':refine.OUT,'extent':refine.OLD}
SCRIPTS={'refinement':ROOT/'scripts/run_topic4_geometry_threshold_refinement.py','extent':ROOT/'scripts/long_topic4_core_extent.py'}


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=1)+'\n');tmp.replace(path)


def path_for(job):return SOURCES[job['source']]/f'duration_{int(job["duration_ms"])}'/'units'/job['candidate']['id']/str(job['seed'])/'workers/trajectory.json'

def valid(job):
    p=path_for(job)
    if not p.exists():return False
    try:
        r=rt.read(p)
        return r['status']=='COMPLETE' and p.with_suffix('.npz').exists() and (r['actual_duration_ms']==job['duration_ms'] or r['physical_status']=='RUNAWAY')
    except (KeyError,ValueError):return False


def jobkey(j):return f'{j["source"]}/{int(j["duration_ms"])}/{j["candidate"]["id"]}/{j["seed"]}'


def initial_jobs(plans):
    result=[]
    # Interleave conditions and seeds; exact candidates and seeds are already frozen.
    for source,seeds in [('refinement',[847101,847102]),('extent',[847101,847102,847103,847104,847105,847106])]:
        for c in plans[source]['candidates']:
            for seed in seeds:result.append(dict(source=source,candidate=c,seed=seed,duration_ms=90000))
    return result


def prepare():
    NIGHT.mkdir(exist_ok=True,parents=True)
    if (NIGHT/'plan.json').exists():return
    plans={k:rt.read(v/'plan.json') for k,v in SOURCES.items()};statuses={k:rt.read(v/'status.json') for k,v in SOURCES.items()}
    jobs=initial_jobs(plans);adopt=[]
    for source,st in statuses.items():
        for a in st.get('active',[]):
            c=next(c for c in plans[source]['candidates'] if c['id']==a['candidate']);adopt.append(dict(source=source,candidate=c,seed=a['seed'],duration_ms=st.get('duration_ms',90000),pid=a['pid']))
    now=time.time();write(NIGHT/'plan.json',dict(started_unix=now,review_after_unix=now+8*3600,dispatch_deadline_unix=now+10*3600,
       max_workers=32,cpu_target_fraction=.90,memory_reserve_gib=60,per_worker_reserve_gib=6,hard_tree_rss_gib=16,hard_available_gib=35,
       max_new_refinement_runs=28,extension_scope='only preauthorized extent noise/180s precision stage; no EE/II/input stage',
       gpu_role='monitor existing GPU jobs; current frozen simulator is CPU, no unvalidated GPU engine substitution',
       adopted=adopt,source_plans={k:str(v/'plan.json') for k,v in SOURCES.items()},source_controller_pids={k:rt.read(v/'controller_pid.json')['pid'] for k,v in SOURCES.items()},
       runner_sha256=rt.sha(SCRIPT),jobs=jobs))
    print(json.dumps(dict(adopt=len(adopt),pending=sum(not valid(j) for j in jobs),goal_window_hours=[8,10]),ensure_ascii=False))


def gpu_snapshot():
    try:return subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=4).strip()
    except Exception as e:return repr(e)


def runtime_estimate(active, jobs, historical_90):
    """Estimate a 90 s unit from current throughput, preserving the 15% dispatch margin."""
    live=[];completed=[]
    for j,proc,_ in active.values():
        try:
            d=rt.read(path_for(j).with_suffix('.progress.json'))
            simulated=float(d['simulated_ms'])
            wall=float(d['updated_unix'])-proc.create_time()
            if simulated>=10000 and wall>=300:
                live.append(wall/simulated*90000)
        except (OSError,ValueError,KeyError,psutil.NoSuchProcess):pass
    for j in jobs:
        try:
            d=rt.read(path_for(j))
            if d.get('status')=='COMPLETE' and d.get('actual_duration_ms')==j['duration_ms']:
                completed.append(float(d['wall_seconds'])/j['duration_ms']*90000)
        except (OSError,ValueError,KeyError):pass
    def q75(xs):return sorted(xs)[math.ceil(.75*len(xs))-1] if xs else 0.
    expected=max(historical_90,q75(live),q75(completed))
    return expected,dict(expected_90_wall_seconds=expected,live_estimates_n=len(live),
        live_q75_wall_seconds=q75(live),completed_q75_wall_seconds=q75(completed),
        interpretation='wall-time forecast only; current concurrency; not a scientific result')


def controller():
    plan=rt.read(NIGHT/'plan.json');jobs=plan['jobs'];active={};analyses={};done_analysis={};failures=[];streams={};extended=False;stop_reason=None
    if rt.sha(SCRIPT)!=plan['runner_sha256']:raise RuntimeError('night runner changed after freeze')
    plans={k:rt.read(v/'plan.json') for k,v in SOURCES.items()}
    for j in plan['adopted']:
        pid=j['pid']
        if psutil.pid_exists(pid):
            proc=psutil.Process(pid)
            if proc.status()!=psutil.STATUS_ZOMBIE:
                cmd=proc.cmdline()
                if str(SCRIPTS[j['source']]) not in cmd or 'worker' not in cmd:raise RuntimeError(f'cannot adopt PID {pid}')
                proc.cpu_percent(None);active[pid]=(j,proc,None)
    write(NIGHT/'adoption_ready.json',dict(active=list(active),updated_unix=time.time()))
    # Only controller processes are replaced; physical worker processes retain RNG/state.
    for source,pid in plan['source_controller_pids'].items():
        if psutil.pid_exists(pid):
            proc=psutil.Process(pid);cmd=proc.cmdline()
            if str(SCRIPTS[source]) not in cmd or 'controller' not in cmd:raise RuntimeError('unexpected source controller identity')
            proc.terminate()
            if proc.is_running() and proc.status()==psutil.STATUS_STOPPED:proc.resume()
    psutil.cpu_percent(None)
    completed_paths=list((refine.OLD/'duration_90000/units').glob('*/*/workers/trajectory.json'))
    historical_90=max(7200.,sum(rt.read(p)['wall_seconds'] for p in completed_paths)/max(1,len(completed_paths)))
    while True:
        now=time.time();rss={};own_cpu=0.
        for pid,(j,proc,popen) in list(active.items()):
            alive=proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE
            if popen is not None and popen.poll() is not None:alive=False
            if not alive:
                if not valid(j):failures.append(dict(job=jobkey(j),pid=pid,reason='WORKER_EXIT_WITHOUT_VALID_RESULT'))
                if popen is not None and popen.returncode not in (None,0):failures.append(dict(job=jobkey(j),pid=pid,exit_code=popen.returncode))
                if pid in streams:streams.pop(pid).close()
                del active[pid];continue
            try:
                family=[proc]+proc.children(recursive=True);rss[pid]=sum(x.memory_info().rss for x in family if x.is_running())/2**30;own_cpu+=proc.cpu_percent(None)/100
                if rss[pid]>plan['hard_tree_rss_gib']:
                    for x in reversed(family):x.terminate()
                    failures.append(dict(job=jobkey(j),reason='TREE_MEMORY_LIMIT',rss_gib=rss[pid]))
            except psutil.NoSuchProcess:pass
        for key,(proc,source,stage,log) in list(analyses.items()):
            if proc.poll() is not None:
                log.close();del analyses[key]
                if proc.returncode:failures.append(dict(analysis=key,exit_code=proc.returncode))
                else:done_analysis[key]=True
        avail=rt.available_gib()
        if avail<plan['hard_available_gib']:
            failures.append(dict(reason='LOW_MACHINE_MEMORY',available_gib=avail));stop_reason='RESOURCE_GUARD'
            if active:
                newest=max(active);proc=active[newest][1]
                for x in reversed([proc]+proc.children(recursive=True)):x.terminate()
        if failures:stop_reason='FAILURE_STOPPED_DISPATCH'
        # Each completed physical stage gets its original analysis without changing physics.
        for source,end,key in [('refinement',90000,'refinement90'),('extent',90000,'extent90'),('extent',180000,'extent180')]:
            stagejobs=[j for j in jobs if j['source']==source and j['duration_ms']==end]
            if not stagejobs or key in analyses or key in done_analysis:continue
            if any(j['source']==source and j['duration_ms']==end for j,proc,po in active.values()):continue
            if all(valid(j) for j in stagejobs):
                stage=dict(duration_ms=end,seeds=[847101,847102] if source=='refinement' else [847101,847102,847103,847104,847105,847106])
                if source=='extent':write(SOURCES[source]/'active_analysis_stage.json',stage)
                script=ROOT/('scripts/analyze_topic4_geometry_threshold_refinement.py' if source=='refinement' else 'scripts/analyze_topic4_core_extent_long.py')
                log=(NIGHT/(key+'_analysis.log')).open('a');proc=subprocess.Popen([rt.PYTHON,'-u',str(script)],cwd=ROOT,env=rt.ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);analyses[key]=(proc,source,stage,log)
        if 'extent90' in done_analysis and not extended:
            audit=rt.read(refine.OLD/'precision_audit.json')
            if not audit['precision_targets_met']:
                for seed in [847101,847102,847103,847104,847105,847106]:
                    for c in plans['extent']['candidates']:jobs.append(dict(source='extent',candidate=c,seed=seed,duration_ms=180000))
            extended=True;write(NIGHT/'effective_jobs.json',jobs)
        total_cpu=psutil.cpu_percent(None)/100*psutil.cpu_count();other_cpu=max(0,total_cpu-own_cpu)
        cpu_cap=max(1,min(plan['max_workers'],int(psutil.cpu_count()*plan['cpu_target_fraction']-other_cpu)-len(analyses)))
        activekeys={jobkey(j) for j,proc,po in active.values()};pending=[j for j in jobs if jobkey(j) not in activekeys and not valid(j)]
        # Reserve enough wall-clock time to finish a newly dispatched trajectory by the 10h review boundary.
        expected_90,timing=runtime_estimate(active,jobs,historical_90)
        fit=[j for j in pending if now+expected_90*(j['duration_ms']/90000)*1.15<plan['dispatch_deadline_unix']]
        allowance=sum(max(0,plan['per_worker_reserve_gib']-rss.get(pid,0)) for pid in active)
        while fit and not stop_reason and len(active)<cpu_cap and rt.available_gib()>plan['memory_reserve_gib']+allowance+plan['per_worker_reserve_gib']:
            j=fit.pop(0);source=j['source'];cp=SOURCES[source]/'candidates'/(j['candidate']['id']+'.json')
            if not cp.exists():write(cp,j['candidate'])
            log=(NIGHT/(jobkey(j).replace('/','_')+'.log')).open('a')
            cmd=[rt.PYTHON,'-u',str(SCRIPTS[source]),'worker','--candidate',str(cp),'--seed',str(j['seed']),'--duration',str(j['duration_ms'])]
            po=subprocess.Popen(cmd,cwd=ROOT,env=rt.ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);pr=psutil.Process(po.pid);pr.cpu_percent(None);active[po.pid]=(j,pr,po);streams[po.pid]=log;allowance+=plan['per_worker_reserve_gib']
        stat=dict(status='RUNNING' if not stop_reason else stop_reason,elapsed_hours=(now-plan['started_unix'])/3600,review_window_hours=[8,10],complete=sum(valid(j) for j in jobs),total=len(jobs),
            active=[dict(pid=pid,source=j['source'],candidate=j['candidate']['id'],seed=j['seed'],duration_ms=j['duration_ms'],tree_rss_gib=rss.get(pid)) for pid,(j,pr,po) in active.items()],
            cpu_used_cores=total_cpu,other_cpu_cores=other_cpu,cpu_dispatch_cap=cpu_cap,available_gib=avail,gpus=gpu_snapshot(),timing=timing,analyses=list(analyses),completed_analyses=list(done_analysis),failures=failures,updated_unix=now)
        write(NIGHT/'status.json',stat)
        with (NIGHT/'resource_history.jsonl').open('a') as f:f.write(json.dumps({k:v for k,v in stat.items() if k not in ['active','failures']},ensure_ascii=False)+'\n')
        for source in SOURCES:
            jj=[j for j in jobs if j['source']==source];aa=[x for x in stat['active'] if x['source']==source];end=max(j['duration_ms'] for j in jj);sj=[j for j in jj if j['duration_ms']==end]
            status='RUNNING' if aa else ('ANALYZING' if any(x[1]==source for x in analyses.values()) else 'QUEUED')
            if source=='refinement' and 'refinement90' in done_analysis:status='COMPLETE_PENDING_PATIENT_MODEL_VISUAL_REVIEW'
            if source=='extent' and extended and all(valid(j) for j in jj) and not any(x[1]==source for x in analyses.values()):status='PRECISION_REVIEW_READY'
            write(SOURCES[source]/'status.json',dict(status=status,controller=str(SCRIPT),duration_ms=end,n_complete=sum(valid(j) for j in sj),n_total=len(sj),active=aa,queued=sum(not valid(j) for j in jj)-len(aa),failures=failures,updated_unix=now))
        if not active and not analyses:
            pending=[j for j in jobs if not valid(j)]
            if stop_reason or not pending or not fit:
                final='NIGHT_COMPLETE_PENDING_SCIENTIFIC_REVIEW' if not pending and not failures else ('FAILED' if failures else 'NIGHT_WINDOW_PENDING_UNFINISHED_UNITS')
                write(NIGHT/'status.json',dict(**{k:v for k,v in stat.items() if k!='status'},status=final,unfinished=[jobkey(j) for j in pending]));return
        time.sleep(30)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','controller']);a=ap.parse_args()
    if a.action=='prepare':prepare()
    else:
        try:controller()
        except Exception as e:
            write(NIGHT/'fatal_error.json',dict(error=repr(e),updated_unix=time.time()));raise
