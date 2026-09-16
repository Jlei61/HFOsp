"""Bounded overnight continuation using the already frozen core_connectivity_v2 worker.

No simulator or observer changes. The first phase is a frozen residual-guided
combination panel, not DE and not a claim of restored patient propagation.
"""
from pathlib import Path
import argparse,copy,json,os,signal,subprocess,sys,time
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
from scripts import run_topic4_core_connectivity_search as old
rt=old.rt
OUT=Path('/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911')
PHASE='recovery_wave1_20260911'

PROBES=[
 ('up_out_transverse','up4p5__EE_core_to_out_scale_1.25',{'EE_kernel_perp_scale':1.25}),
 ('up_out_extent_depth','up4p5__EE_core_to_out_scale_1.25',{'radius_A_mm':2.5,'depth_A_scale':.5}),
 ('up_out_EI','up4p5__EE_core_to_out_scale_1.25',{'EI_same_core_scale':.85}),
 ('up_out_IE','up4p5__EE_core_to_out_scale_1.25',{'IE_same_core_scale':1.2}),
 ('upper_wide_bias_B','near_upper__EE_kernel_perp_scale_1.5',{'depth_A_scale':.75,'depth_B_scale':1.25}),
 ('upper_wide_bias_A','near_upper__EE_kernel_perp_scale_1.5',{'depth_A_scale':1.25,'depth_B_scale':.75}),
 ('upper_wide_recurrence','near_upper__EE_kernel_perp_scale_1.5',{'EE_same_core_scale':.75}),
 ('upper_output_transverse','near_upper__EE_kernel_perp_scale_1.5',{'EE_core_to_out_scale':1.25,'EE_kernel_perp_scale':1.25}),
]

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    path=OUT/'plan.json'
    if path.exists():return rt.read(path)
    oldplan=rt.read(old.OUT/'plan.json')
    for p,h in oldplan['source_snapshot'].items():
        if rt.sha(p)!=h:raise RuntimeError(f'frozen physics changed: {p}')
    oldcases={c['id']:c for c in oldplan['candidates']};cases=[]
    for slug,parent,changes in PROBES:
        c=copy.deepcopy(oldcases[parent]);c.update(id='recovery_'+slug,parent_id=parent,stage=PHASE,
            changed_parameter='declared_combination',changed_value=None,changed_parameters=changes)
        c['parameters'].update(changes);c['radii_mm']=[c['parameters'][f'radius_{k}_mm'] for k in ['A','B']]
        c['adjacency_changes']=any(c['parameters'][k]!=oldplan['baseline_parameters'][k] for k in ('EE_core_to_out_degree_scale','EE_kernel_perp_scale','EE_kernel_parallel_scale','EE_angle_offset_deg'))
        centers=np.asarray(c['centers_mm']);radii=np.asarray(c['radii_mm'])
        assert np.linalg.norm(centers[1]-centers[0])>radii.sum()
        assert all(min(x[0],x[1],20-x[0],20-x[1])>=r for x,r in zip(centers,radii))
        assert all(oldplan['axes'][k]['adaptive_bounds'][0]<=v<=oldplan['axes'][k]['adaptive_bounds'][1] for k,v in c['parameters'].items())
        dest=old.OUT/'candidates'/f'{c["id"]}.json'
        if dest.exists() and rt.read(dest)!=c:raise RuntimeError('candidate identity collision')
        rt.write(dest,c);cases.append(c)
    # User's autonomous window began at goal creation, 00:12:25 local time.
    start=1789056745.
    plan=dict(version='propagation_recovery_night_v1',authorization='2026-09-11 user approved autonomous 8-10 hour continuation and scientific review first',
        started_unix=start,stop_new_dispatch_unix=start+8*3600,hard_stop_unix=start+10*3600,
        inherited_physics=oldplan['physics'],source_snapshot=oldplan['source_snapshot'],
        original_screen_root=str(old.OUT),screen_completed_runs=120,
        review=dict(primary_events=485,all_detected_events=7735,excluded_only_window_overlap=7250,
            completed_full_trajectories=120,runaway=0,primary_scorable_runs=1,
            decision='patient propagation not accepted; retain isolated-window training contract; all-detection diagnostic cannot replace formal patient-matched target'),
        remaining_budget=dict(new_combination_runs=16,long_support_runs=16,confirmation_runs=16,total_new_max=48,including_old_screen_max=168),
        wave1=dict(stage=PHASE,candidates=cases,seeds=[847101,847102],topology_seed=2511,duration_ms=20000),
        long_support=dict(max_conditions=8,seeds=[847101,847102],topology_seed=2511,duration_ms=60000,selection='after wave1 readout/field review; preserve baseline and direct parents'),
        confirmation=dict(max_conditions=4,new_topologies=2,new_dynamics_per_topology=2,duration_ms=60000,selection='after longer-run review; include reference; explicit unused seed identities'),
        resources=dict(max_workers=8,reserve_per_worker_gib=9.,min_available_gib=40.,kill_tree_rss_gib=18.,
            measured_prior_peak_gib=5.897281646728516,initial_workers=8,reason='prior 120 completed units already supply empirical canary memory measurements'),
        observation='frozen primary selection, centroids, classifier, kernels, masks and positive scales unchanged; all-detection developmental population separately labelled',
        stop='no automatic Fig5/model freeze; no primary relaxation to manufacture scorable output; finite negative scientific results retained')
    rt.write(path,plan);rt.write(OUT/'status.json',dict(status='PREPARED',goal='restore patient-compatible TA/TB propagation',formal_new_completed=0))
    return plan

def stop_tree(pid):
    try:
        process=psutil.Process(pid);children=process.children(recursive=True)
        for p in reversed(children):
            try:p.terminate()
            except psutil.NoSuchProcess:pass
        process.terminate()
    except psutil.NoSuchProcess:pass

def dispatch(stage,units,duration,phase,worker_script=None):
    plan=prepare();res=plan['resources'];active={};failures=[];pending=[]
    budget_key={'wave1':'new_combination_runs','long':'long_support_runs','confirmation':'confirmation_runs','final_A':'confirmation_runs','final_B':'confirmation_runs'}[phase]
    if len(units)>plan['remaining_budget'][budget_key]:raise RuntimeError('phase budget exceeded')
    if phase in ('final_A','final_B') and len(units)>8:raise RuntimeError('final sub-batch exceeds eight units')
    ledger_path=OUT/'dispatch_manifest.json'
    ledger=rt.read(ledger_path) if ledger_path.exists() else []
    for cid,topo,seed in units:
        item=dict(stage=stage,candidate=cid,topology_seed=int(topo),dynamics_seed=int(seed),duration_ms=float(duration),phase=phase)
        if worker_script is not None:item['worker_script']=str(worker_script)
        if item not in ledger:ledger.append(item)
    if len(ledger)>plan['remaining_budget']['total_new_max']:raise RuntimeError('overnight formal budget exceeded')
    rt.write(ledger_path,ledger)
    unit_paths=[]
    for cid,topo,seed in units:
        p=old.result_path(stage,cid,topo,seed);unit_paths.append(str(p))
        if p.exists() and not old.complete(p):raise RuntimeError(f'invalid existing output {p}')
        if not old.complete(p):pending.append((cid,topo,seed))
    rt.write(OUT/f'{phase}_units.json',dict(stage=stage,duration_ms=duration,units=[list(u) for u in units],paths=unit_paths))
    peak=0.;halted=False
    while pending or active:
        for pid,(proc,u,log) in list(active.items()):
            if proc.poll() is not None:
                log.close();del active[pid]
                if proc.returncode or not old.complete(old.result_path(stage,*u)):
                    failures.append(dict(unit=u,exit_code=proc.returncode,log=log.name))
                continue
            try:
                tree=[psutil.Process(pid)]+psutil.Process(pid).children(recursive=True)
                rss=sum(p.memory_info().rss for p in tree)/2**30;peak=max(peak,rss)
                if rss>res['kill_tree_rss_gib'] or rt.available_gib()<res['min_available_gib']:
                    stop_tree(pid);failures.append(dict(unit=u,reason='RESOURCE_GUARD',rss_gib=rss))
            except psutil.NoSuchProcess:pass
        now=time.time()
        if now>=plan['hard_stop_unix']:
            for pid in list(active):stop_tree(pid)
            halted=True
        if failures:halted=True
        if now>=plan['stop_new_dispatch_unix']:halted=True
        while pending and not halted and len(active)<res['max_workers'] and rt.available_gib()>res['min_available_gib']+res['reserve_per_worker_gib']:
            u=pending.pop(0);cid,topo,seed=u
            log=(OUT/'logs'/f'{phase}_{cid}_{topo}_{seed}.log').open('a')
            cmd=[rt.PYTHON,'-u',str(worker_script or old.SCRIPT),'worker','--stage',stage,'--candidate',cid,'--topology',str(topo),'--seed',str(seed),'--duration',str(duration)]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=rt.ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[proc.pid]=(proc,u,log);time.sleep(1)
        complete_count=sum(old.complete(Path(p)) for p in unit_paths)
        rt.write(OUT/'status.json',dict(status=f'{phase.upper()}_RUNNING' if active or (pending and not halted) else f'{phase.upper()}_COMPLETE' if not pending and not failures else 'HALTED_PENDING_REVIEW',
            phase=phase,complete=complete_count,total=len(units),queued=len(pending),
            active=[dict(pid=pid,candidate=u[0],topology=u[1],seed=u[2]) for pid,(pr,u,f) in active.items()],
            failures=failures,peak_tree_rss_gib=peak,available_gib=rt.available_gib(),updated_unix=now,
            stop_new_dispatch_unix=plan['stop_new_dispatch_unix'],hard_stop_unix=plan['hard_stop_unix']))
        if halted and not active:break
        if pending or active:time.sleep(10)
    return failures

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','wave1']);args=ap.parse_args();plan=prepare()
    if args.action=='wave1':
        import fcntl
        with (OUT/'controller.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            w=plan['wave1'];units=[(c['id'],w['topology_seed'],s) for c in w['candidates'] for s in w['seeds']]
            failures=dispatch(w['stage'],units,w['duration_ms'],'wave1')
            if failures:raise RuntimeError(str(failures))
    print(json.dumps(dict(output=str(OUT),action=args.action)),flush=True)

if __name__=='__main__':main()
