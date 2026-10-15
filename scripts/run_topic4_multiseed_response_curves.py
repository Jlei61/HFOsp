"""Bounded 2026-09-13 multi-seed response curves; unchanged physical executor.

Prepare new graph identities and short application checks while the previous round
finishes. Formal runs then follow it automatically. No score-dependent early stop.
"""
from pathlib import Path
import argparse,copy,dataclasses,fcntl,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
from scripts import run_topic4_shape_output_response as base
from scripts.topic4_phase_memory_reserve import reserve_gib
rt=base.rt
PARENT=Path('/data/hfosp/topic4_sef_hfo/core_shape_output_response_20260911')
FOLLOW=Path('/data/hfosp/topic4_sef_hfo/core_recruitment_tradeoff_followup_20260912')
OUT=Path('/data/hfosp/topic4_sef_hfo/core_multiseed_response_curves_20260913')
SCRIPT=Path(__file__).resolve()


def physical(c):
    return {k:c.get(k,{}) for k in ['centers_mm','radii_mm','parameters','shape','outgoing','core_mean_rate_scale','core_ou_correlation']}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():return rt.read(OUT/'plan.json')
    pp=rt.read(PARENT/'plan.json');candidates={};families=[]
    def add(shape,ee=1.,ei=1.,cid=None,degree=1.,dx=0.,comparison=None):
        cid=cid or f'curve_{shape}_out{ee:g}_EI{ei:g}'
        if cid in candidates:return cid
        c=copy.deepcopy(rt.read(PARENT/'candidates'/f'up3__{shape}.json'))
        c['parameters'].update(EE_core_to_out_scale=ee,EI_same_core_scale=ei,EE_core_to_out_degree_scale=degree)
        c['centers_mm'][0][0]+=dx
        label=('圆核' if shape=='circle' else '椭圆4:1')+f'｜向外EE×{ee:g}，核内EI×{ei:g}'
        if degree!=1:label+=f'，向外边数×{degree:g}'
        if dx:label+=f'，左核x偏移{dx:g}mm'
        c.update(id=cid,stage='response',contrast='response_curve',display_name=label,
            comparison=comparison or f'curve_{shape}_out1_EI1',response_coordinates=dict(shape=shape,EE_out=ee,EI=ei,degree=degree,dx_mm=dx))
        candidates[cid]=c;return cid
    for shape in ['circle','ellipse4']:
        ids=[add(shape,ei=v) for v in [.75,.8125,.875,.9375,1.]]
        families.append(dict(id=shape+'_EI',label=('圆核' if shape=='circle' else '椭圆')+'：核内 E→I',axis='EI',values=[.75,.8125,.875,.9375,1.],candidates=ids,fixed='向外EE=1，其余参数固定'))
    ids=[add('circle',ee=v,ei=.875,comparison='curve_circle_out1_EI0.875') for v in [1.,1.0625,1.125,1.1875,1.25]]
    families.append(dict(id='circle_EE_at_EI0875',label='圆核：向外 E→E',axis='EE_out',values=[1.,1.0625,1.125,1.1875,1.25],candidates=ids,fixed='核内EI=0.875，其余参数固定'))
    normal=add('circle',ee=1.25,cid='bridge_circle_out125')
    shifted=add('circle',ee=1.25,dx=-.75,cid='bridge_circle_out125_xminus075',comparison=normal)
    degree=add('circle',degree=1.5,cid='bridge_circle_degree150')
    dose=add('circle',ee=1.5,cid='bridge_circle_weight150')
    assert len(candidates)==18
    for d in ['candidates','logs','confirmation','analysis','global_graph_cache','rotation']:(OUT/d).mkdir(exist_ok=True)
    for path in (PARENT/'global_graph_cache').glob('*.pkl'):
        target=OUT/'global_graph_cache'/path.name
        if not target.exists():target.symlink_to(path)
    for c in candidates.values():rt.write(OUT/'candidates'/f'{c["id"]}.json',c)
    sources=dict(pp['source_snapshot'])
    for f in [ROOT/'src/topic4_core_field_runner.py',ROOT/'scripts/rebuild_topic4_autapse_corrected_reference.py']:
        sources[str(f)]=rt.sha(f)
    # No old physical code is edited; controller changes do not silently version the engine.
    plan=dict(schema='topic4.multiseed_response_curves.v1',authorization='2026-09-13 user: continue useful parameter experiments, more seeds and parameter points, make response figures solid',
        physics=pp['physics'],source_snapshot=sources,controller_sha256=rt.sha(SCRIPT),
        topology_seed=2511,topology_seeds=[2511,2711,2712],seeds=[847401,847402],duration_ms=60000.,analysis=pp['analysis'],runaway=pp['runaway'],
        resources=dict(max_workers=18,min_available_GiB=50,per_tree_limit_GiB=18,launch_reserve_GiB=9),
        candidates=list(candidates.values()),families=families,
        pairs=[dict(candidate=shifted,reference=normal,question='leftward position shift'),dict(candidate=degree,reference=dose,question='nominal dose-near edge addition versus weight increase')],
        budget=dict(conditions=18,topologies=3,noise_replays=2,new_formal_runs=108,application_runs=3,application_duration_ms=500.),
        confirmation=dict(candidates=['curve_circle_out1_EI1','curve_circle_out1_EI0.75','curve_ellipse4_out1_EI0.75',normal,shifted,degree]),
        start_condition='formal dispatch waits for FOLLOW simulation_complete.json and no engineering failure; readout/rotation of previous round may finish in parallel',
        reuse='old results used only as development references; all 108 formal combinations use new noise seeds; no copied old run is counted as new',
        seed_semantics='same noise IDs fully crossed with 2511/2711/2712; only 2711/2712 are new topologies in this series; shared seed does not assert exact innovation alignment across different physical conditions',
        objective='unchanged L_search and event eligibility; detailed TA/TB routes, width and rotation remain diagnostics; no new loss or route-conditioned drive',
        stop='complete bounded 108 formal units including unfavourable outcomes; engineering failure drains queue; stop for review, no automatic further round/model freeze/Fig5')
    rt.write(OUT/'plan.json',plan)
    rt.write(OUT/'confirmation/frozen_networks.json',dict(replication_networks={}))
    rt.write(OUT/'status.json',dict(status='PREPARED_APPLICATION_PENDING',budget=plan['budget'],updated_unix=time.time()))
    return plan


def setup():
    base.OUT=OUT;base.SCRIPT=SCRIPT;base.configure()


def build_network(seed):
    from params import Params
    from src.topic4_core_field_runner import get_network,connectivity_config,cache_key
    from scripts.rebuild_topic4_autapse_corrected_reference import summarize
    target=OUT/'confirmation'/f'network_{seed}.json'
    if target.exists():
        rec=rt.read(target)
        if rt.sha(rec['path'])!=rec['sha256']:raise RuntimeError('graph cache changed')
        return rec
    design=rt.read(base.base.PARENT);ref=rt.network_record(design,2511)['config']
    fields={f.name for f in dataclasses.fields(Params)};kw={k:v for k,v in ref.items() if k in fields};kw['seed']=int(seed)
    prm=Params(**kw);cache=OUT/'confirmation/network_cache';cache.mkdir(exist_ok=True)
    net,ne,ni,hit=get_network(prm,ref['theta_EE_deg'],ref['AR'],str(cache));stats=summarize(net)
    if any(x['self_edges'] or not x['exact_expected_degree'] for x in stats.values()):raise RuntimeError('new graph structure mismatch')
    cfg=connectivity_config(prm,ref['theta_EE_deg'],ref['AR']);path=cache/(cache_key(cfg)+'.pkl')
    rec=dict(path=str(path),sha256=rt.sha(path),topology_seed=int(seed),config=cfg,pathways=stats,NE=ne,NI=ni,cache_hit=hit,status='CORRECTED_GRAPH_VALIDATED')
    rt.write(target,rec);print(json.dumps(dict(topology=seed,graph='VALIDATED')),flush=True);return rec


def graph_preparation():
    records={}
    for seed in [2711,2712]:
        while rt.available_gib()<65:time.sleep(15)
        log=(OUT/'logs'/f'build_network_{seed}.log').open('a')
        p=subprocess.Popen([rt.PYTHON,'-u',str(SCRIPT),'build-network','--topology',str(seed)],cwd=ROOT,env=rt.ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        rt.write(OUT/'status.json',dict(status='PREPARING_NEW_NETWORK',topology=seed,pid=p.pid,updated_unix=time.time()))
        import psutil
        while p.poll() is None:
            try:
                tree=[psutil.Process(p.pid)]+psutil.Process(p.pid).children(recursive=True)
                rss=sum(x.memory_info().rss for x in tree)/2**30
                if rss>18 or rt.available_gib()<35:
                    for proc in reversed(tree):proc.terminate()
                    raise RuntimeError('network build resource guard')
            except psutil.NoSuchProcess:pass
            time.sleep(5)
        log.close()
        if p.returncode:raise RuntimeError(f'network {seed} failed; see {log.name}')
        records[str(seed)]=rt.read(OUT/'confirmation'/f'network_{seed}.json')
        rt.write(OUT/'confirmation/frozen_networks.json',dict(replication_networks=records))


def launch_analysis():
    path=OUT/'analysis_processes.json';running=[]
    import psutil
    if path.exists():running=rt.read(path)
    def alive(entry):
        try:
            p=psutil.Process(entry['pid'])
            return p.status()!=psutil.STATUS_ZOMBIE and 'analyze_topic4_multiseed_response_curves.py' in ' '.join(p.cmdline())
        except psutil.Error:return False
    for args,role in [(['observer'],'observer'),(['rotation','--gpu','0'],'rotation0'),(['rotation','--gpu','1'],'rotation1')]:
        if any(x['role']==role and alive(x) for x in running):continue
        log=(OUT/'logs'/(role+'.log')).open('a')
        env=dict(rt.ENV)
        if role.startswith('rotation'):env['CUDA_VISIBLE_DEVICES']='0,1'
        p=subprocess.Popen([rt.PYTHON,'-u',str(ROOT/'scripts/analyze_topic4_multiseed_response_curves.py'),*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();running=[x for x in running if x['role']!=role]+[dict(role=role,pid=p.pid)]
    rt.write(path,running)


def validate_application():
    specs=[('curve_circle_out1_EI0.8125',2511,847401),('curve_circle_out1_EI1',2711,847401),('bridge_circle_degree150',2712,847401)]
    for u in specs:
        base.queue('canary',[u],500.,1)
        ap=rt.read(base.result_path('canary',u).parents[1]/'applied_physics.json');c=rt.read(OUT/'candidates'/f'{u[0]}.json')
        assert ap['threshold']['n_raised']==0
        assert ap['input']['ZM']=='off' and ap['input']['spatial_ou']=='off' and ap['input']['slow_I_state']=='off' and ap['input']['kick']=='off'
        assert ap['candidate']['parameters']==c['parameters'] and ap['candidate']['centers_mm']==c['centers_mm']
        assert ap['topology_seed']==u[1] and ap['dynamics_seed']==u[2]
        for k,v in ap['graph']['stage_audits']['weights']['factors'].items():assert v==c['parameters'][k]
        rr=rt.read(base.result_path('canary',u));assert rr['maximum_outside_rate_deviation']==0
        assert rr['actual_duration_ms']==500.
    rt.write(OUT/'application_validation.json',dict(status='PASS',units=specs,scope='applied parameters, graph identity, lowering-only thresholds and core-only random drive; no propagation acceptance gate'))


def formal_order(plan):
    """Freeze complete control/treatment pairs first, without selecting on scores."""
    priority=[]
    contrasts=[['bridge_circle_out125','bridge_circle_out125_xminus075'],
        ['curve_circle_out1_EI1','curve_circle_out1_EI0.75'],
        ['curve_ellipse4_out1_EI1','curve_ellipse4_out1_EI0.75'],
        ['bridge_circle_weight150','bridge_circle_degree150']]
    for pair in contrasts:
        for seed in plan['seeds']:
            for topo in plan['topology_seeds']:
                priority.extend((cid,topo,seed) for cid in pair)
    remaining=[(c['id'],t,s) for c in plan['candidates'] for t in plan['topology_seeds'] for s in plan['seeds']]
    units=list(dict.fromkeys(priority+remaining));assert len(units)==108
    return units


def predecessor_growth():
    """Use live predecessor workers; a stale status row is not a live process."""
    import psutil
    if (FOLLOW/'simulation_complete.json').exists():return 0.,False,0
    state=rt.read(FOLLOW/'status.json')
    if state.get('failures') or 'FAILURE' in state.get('status',''):raise RuntimeError('previous round engineering failure requires review')
    active=[]
    for row in state.get('active',[]):
        try:
            p=psutil.Process(row['pid']);cmd=p.cmdline()
            if p.status()==psutil.STATUS_ZOMBIE or not any('run_topic4_recruitment_tradeoff_followup.py' in x for x in cmd) or 'worker' not in cmd:continue
            tree=[p]+p.children(recursive=True);rss=sum(x.memory_info().rss for x in tree)/2**30
            active.append(max(0.,reserve_gib(FOLLOW,state.get('stage','response'),row['unit'])-rss))
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    # Construction and verified integration have separately measured footprints.
    return sum(active),True,len(active)


class AdoptedWorker:
    """Monitor a preserved worker whose original controller has exited.

    Its OS exit code is unavailable. Success is assessed from the same complete
    trajectory check used for ordinary workers; an absent invalid result fails.
    """
    def __init__(self,row):
        self.pid=int(row['pid']);self.row=row;self.returncode=None
    def poll(self):
        import psutil
        try:
            p=psutil.Process(self.pid)
            if abs(p.create_time()-self.row['process_created_unix'])>1e-3:raise RuntimeError('adopted PID identity changed')
            if p.status()!=psutil.STATUS_ZOMBIE:return None
        except psutil.NoSuchProcess:pass
        self.returncode=0 if base.base.complete(base.result_path('response',tuple(self.row['unit']))) else -1
        return self.returncode


def optional_probe_resources():
    """Count live optional-probe growth; reserve its two slots during the 8h window."""
    import psutil
    out=OUT.parent/'global_axis_residual_probe_20260913'
    if not (out/'plan.json').exists() or not (out/'status.json').exists():return 0.,0,False
    plan=rt.read(out/'plan.json');state=rt.read(out/'status.json');rss=[]
    for row in state.get('active',[]):
        try:
            p=psutil.Process(row['pid']);cmd=p.cmdline()
            if p.status()==psutil.STATUS_ZOMBIE or 'worker' not in cmd or not any('run_topic4_global_axis_residual_probe.py' in x for x in cmd):continue
            actual=sum(q.memory_info().rss for q in [p]+p.children(recursive=True))/2**30
            rss.append(max(0.,reserve_gib(out,state.get('stage','response'),row['unit'])-actual))
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    priority=(time.time()<plan['dispatch_deadline_unix'] and not (out/'simulation_complete.json').exists()
              and not state.get('failures') and 'FAILURE' not in state.get('status',''))
    return sum(rss),len(rss),priority


def response_queue(units):
    """At most eight early runs, accounting for all queues' remaining growth."""
    import psutil
    pending=[]
    for u in units:
        p=base.result_path('response',u)
        if p.exists() and not base.base.complete(p):raise RuntimeError(f'invalid existing result {p}')
        if not p.exists():pending.append(u)
    active={};failures=[];peak=0.
    recovery=OUT/'preserved_active_workers.json'
    if recovery.exists():
        for row in rt.read(recovery)['workers']:
            u=tuple(row['unit'])
            if u not in units:raise RuntimeError('preserved unit is outside frozen 108')
            if base.base.complete(base.result_path('response',u)):continue
            p=psutil.Process(row['pid']);cmd=p.cmdline()
            if str(SCRIPT) not in cmd or 'worker' not in cmd or cmd[cmd.index('--candidate')+1]!=u[0] or int(cmd[cmd.index('--topology')+1])!=u[1] or int(cmd[cmd.index('--seed')+1])!=u[2]:
                raise RuntimeError('preserved worker command does not match its unit')
            if abs(p.create_time()-row['process_created_unix'])>1e-3:raise RuntimeError('preserved process identity changed')
            if u in pending:pending.remove(u)
            active[p.pid]=(AdoptedWorker(row),u,Path(row['log']).open('a'))
        rt.write(OUT/'worker_adoption.json',dict(adopted=[dict(pid=pid,unit=u) for pid,(_,u,_) in active.items()],physical_workers_restarted=0,created_unix=time.time(),exit_code_semantics='Adopted worker OS exit codes unavailable; completion artifacts remain mandatory'))
    while pending or active:
        rss={}
        for pid,(proc,u,log) in list(active.items()):
            if proc.poll() is not None:
                log.close();del active[pid]
                if proc.returncode or not base.base.complete(base.result_path('response',u)):failures.append(dict(unit=u,exit=proc.returncode,log=log.name))
                else:peak=max(peak,rt.read(base.result_path('response',u))['peak_rss_gib'])
                continue
            try:
                tree=[psutil.Process(pid)]+psutil.Process(pid).children(recursive=True);rss[pid]=sum(p.memory_info().rss for p in tree)/2**30
                if rss[pid]>18 or rt.available_gib()<30:
                    for p in reversed(tree):p.terminate()
                    failures.append(dict(unit=u,error='RESOURCE_GUARD',rss_gib=rss[pid]))
            except psutil.NoSuchProcess:pass
        try:pre_growth,previous_running,pre_count=predecessor_growth()
        except RuntimeError as e:
            pre_growth,previous_running,pre_count=0.,True,0
            if not any(x.get('error')==str(e) for x in failures):failures.append(dict(error=str(e)))
        if failures:pending=[]
        probe_growth,probe_active,probe_priority=optional_probe_resources()
        limit=10 if previous_running else (16 if probe_priority else 18)
        own_growth=sum(max(0.,reserve_gib(OUT,'response',u)-rss.get(pid,0.)) for pid,(_,u,_) in active.items())
        required=40.+pre_growth+probe_growth+own_growth+max(9.,peak*1.4)
        available=rt.available_gib()
        if pending and len(active)<limit and available>required:
            u=pending.pop(0);log=(OUT/'logs'/f'response_{u[0]}_{u[1]}_{u[2]}.log').open('a')
            cmd=[rt.PYTHON,'-u',str(SCRIPT),'worker','--stage','response','--candidate',u[0],'--topology',str(u[1]),'--seed',str(u[2]),'--duration','60000.0']
            proc=subprocess.Popen(cmd,cwd=ROOT,env=rt.ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[proc.pid]=(proc,u,log)
            with (OUT/'dispatch.jsonl').open('a') as f:f.write(json.dumps(dict(unit=u,pid=proc.pid,available_gib=available,required_gib=required,predecessor_growth_gib=pre_growth,own_growth_gib=own_growth,previous_running=previous_running,started_unix=time.time()))+'\n')
        rt.write(OUT/'status.json',dict(status='RESPONSE_RUNNING' if active or not pending else 'WAITING_FOR_SHARED_RESOURCES',stage='response',total=len(units),
            complete=sum(base.result_path('response',u).exists() for u in units),queued=len(pending),active=[dict(pid=pid,unit=u,rss_gib=rss.get(pid)) for pid,(p,u,f) in active.items()],
            max_workers=limit,previous_running=previous_running,predecessor_active=pre_count,predecessor_growth_gib=pre_growth,own_growth_gib=own_growth,
            optional_probe_active=probe_active,optional_probe_growth_gib=probe_growth,optional_probe_priority=probe_priority,
            available_gib=available,required_gib=required,peak_rss_gib=peak,failures=failures,updated_unix=time.time()))
        if failures:
            s=rt.read(OUT/'status.json');s['status']='ENGINEERING_FAILURE_DRAINING';rt.write(OUT/'status.json',s)
        time.sleep(5)
    if failures:raise RuntimeError(json.dumps(failures))


def controller():
    plan=prepare();setup()
    with (OUT/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for path,h in plan['source_snapshot'].items():
                if rt.sha(path)!=h:raise RuntimeError(f'frozen physics changed: {path}')
            graph_preparation();validate_application()
            launch_analysis()
            units=formal_order(plan)
            if (OUT/'formal_units.json').exists() and rt.read(OUT/'formal_units.json')!=[list(u) for u in units]:raise RuntimeError('frozen dispatch order changed')
            rt.write(OUT/'formal_units.json',units)
            response_queue(units)
            rt.write(OUT/'simulation_complete.json',dict(formal_runs=len(units),expected_analyzed_runs=108,updated_unix=time.time()))
            rt.write(OUT/'status.json',dict(status='SIMULATIONS_COMPLETE_ANALYSIS_PENDING',formal_runs=len(units),updated_unix=time.time()))
        except Exception as e:
            rt.write(OUT/'status.json',dict(status='ENGINEERING_FAILURE',error=repr(e),updated_unix=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','controller','build-network','worker']);p.add_argument('--topology',type=int);p.add_argument('--seed',type=int);p.add_argument('--candidate');p.add_argument('--stage');p.add_argument('--duration',type=float);a=p.parse_args()
    if a.action=='prepare':print(json.dumps(prepare()['budget']))
    elif a.action=='build-network':build_network(a.topology)
    elif a.action=='worker':setup();base.run.worker(a.stage,a.candidate,a.topology,a.seed,a.duration)
    else:controller()
