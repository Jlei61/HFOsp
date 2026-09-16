"""Authorized bounded three-observable BO; unchanged SNN physical executor."""
from pathlib import Path
import argparse, copy, fcntl, hashlib, json, os, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import run_topic4_shape_output_response as physical
rt=physical.rt
MAIN=Path('/home/honglab/leijiaxin/HFOsp')
BASE=Path('/data/hfosp/topic4_sef_hfo')
OUT=BASE/'geometry_ee_axis_three_observable_optimization_20260914'
SCRIPT=Path(__file__).resolve()
BO_PY='/data/hfosp/envs/topic4_bo_v1/bin/python'
DESIGN=MAIN/'config/topic4_geometry_ee_axis_optimization_v1_design.json'
PHYSICAL_KEYS=['centers_mm','radii_mm','parameters','shape','outgoing','core_mean_rate_scale','core_ou_correlation']

def vector(c):
    return np.r_[np.asarray(c['centers_mm']).ravel(),c['parameters']['EE_core_to_out_scale'],c['parameters']['EE_angle_offset_deg']]

def physics(c):return {k:copy.deepcopy(c[k]) for k in PHYSICAL_KEYS}

def background(c):
    d=physics(c);d.pop('centers_mm')
    for k in ['EE_core_to_out_scale','EE_angle_offset_deg']:d['parameters'].pop(k)
    return d

def bounds(design=None):
    d=design or rt.read(DESIGN);v=[v for g in d['variable_groups'] for v in g['variables']]
    return np.array([x['lower'] for x in v]),np.array([x['upper'] for x in v])

def candidate_at(ref,x,cid,kind,axis=None):
    c=copy.deepcopy(ref);c['centers_mm']=np.asarray(x[:4]).reshape(2,2).tolist()
    c['parameters'].update(EE_core_to_out_scale=float(x[4]),EE_angle_offset_deg=float(x[5]))
    c.update(id=cid,stage='initial',comparison=ref['id'],contrast=kind,display_name=cid,optimization_axis=axis)
    return c

def validate_candidate(c):
    p=rt.read(OUT/'plan.json');ref=rt.read(OUT/'candidates'/f"{p['reference_id']}.json")
    assert background(c)==background(ref),'background parameters changed'
    lo,hi=bounds();x=vector(c);assert np.all(x>=lo-1e-12) and np.all(x<=hi+1e-12)
    xy=np.asarray(c['centers_mm']);r=np.asarray(c['radii_mm'])
    assert np.all(xy>=r[:,None]) and np.all(xy<=20-r[:,None])
    assert np.linalg.norm(xy[0]-xy[1])>r.sum()

def unit_path(stage,cid,topo,noise):return OUT/stage/'units'/cid/f'{topo}_{noise}'/'workers/trajectory.json'

def prepare():
    if (OUT/'plan.json').exists():return rt.read(OUT/'plan.json')
    for d in ['candidates','logs','analysis','scores','confirmation','global_graph_cache','rotation','proposals']:(OUT/d).mkdir(parents=True,exist_ok=True)
    design=rt.read(DESIGN);pointer=rt.read(MAIN/design['reference_pointer']);ref=rt.read(pointer['candidate_path'])
    assert rt.sha(pointer['contract_path'])==design['reference_contract_sha256']
    old=BASE/'core_multiseed_response_curves_20260913';op=rt.read(old/'plan.json')
    for path,h in op['source_snapshot'].items():assert rt.sha(path)==h,path
    for path in (old/'global_graph_cache').glob('*.pkl'):
        target=OUT/'global_graph_cache'/path.name
        if not target.exists():target.symlink_to(path.resolve())
    available=[]
    for path in sorted((old/'candidates').glob('*.json')):
        c=rt.read(path)
        if background(c)!=background(ref):continue
        lo,hi=bounds(design);x=vector(c)
        if not(np.all(x>=lo) and np.all(x<=hi)):continue
        units=[old/'response/units'/c['id']/f'2511_{s}' for s in design['training']['dynamics_seeds']]
        if all((u/'workers/trajectory.json').exists() and rt.read(u/'workers/trajectory.json')['actual_duration_ms']==60000 for u in units):
            for u in units:
                a=rt.read(u/'applied_physics.json');assert physics(a['candidate'])==physics(c)
            available.append((c,units))
    proposed=[(ref,'reference',None)];x0=vector(ref);steps=[.75,.75,.75,.75,.125,10.]
    for j in range(6):
        for sign in [-1,1]:
            x=x0.copy();x[j]+=sign*steps[j]
            proposed.append((candidate_at(ref,x,f'g1_axis{j}_{"minus" if sign<0 else "plus"}','single_axis',j),'single_axis',j))
    candidates=[];reuse={};meta=[]
    def add(c,kind,axis):
        match=next(((a,u) for a,u in available if np.allclose(vector(a),vector(c),rtol=0,atol=1e-12)),None)
        if match is not None:c,units=copy.deepcopy(match[0]),match[1];reuse[c['id']]=units
        if any(np.allclose(vector(a),vector(c),rtol=0,atol=1e-12) for a in candidates):return False
        candidates.append(c);meta.append(dict(candidate=c['id'],kind=kind,axis=axis,vector=vector(c).tolist()));return True
    for c,kind,axis in proposed:add(c,kind,axis)
    for c,_ in available:
        if len(candidates)>=16:break
        add(c,'historical_combination',None)
    from scipy.stats import qmc
    pool=qmc.Sobol(6,scramble=True,seed=2026091402).random_base2(6)
    lo,hi=bounds(design)
    for i,z in enumerate(pool):
        if len(candidates)>=16:break
        x=x0+(z-.5)*(hi-lo)*.5
        add(candidate_at(ref,x,f'g1_combination_{i:03}','space_filling'),'space_filling',None)
    assert len(candidates)==16
    for c in candidates:rt.write(OUT/'candidates'/f"{c['id']}.json",c)
    reused=[]
    for cid,units in reuse.items():
        for u in units:
            target=OUT/'initial/units'/cid/u.name;target.parent.mkdir(parents=True,exist_ok=True)
            target.symlink_to(u.resolve());reused.append(dict(candidate=cid,unit=u.name,source=str(u),trajectory_sha256=rt.sha(u/'workers/trajectory.json')))
    plan=dict(schema='topic4.three_observable_bo.RUN.v1',authorization='2026-09-14 user: 开始，不要让CPU和GPU空置',
        design_path=str(DESIGN),design_sha256=rt.sha(DESIGN),reference_id=ref['id'],reference_pointer=pointer,
        physics=op['physics'],source_snapshot=op['source_snapshot'],analysis=op['analysis'],runaway=op['runaway'],
        topology_seed=2511,topology_seeds=[2511],seeds=design['training']['dynamics_seeds'],duration_ms=60000.,
        confirmation_seeds=dict(topology=[3711,3712],dynamics=[849401,849402]),
        candidates=candidates,initial_ids=[c['id'] for c in candidates],initial_meta=meta,reused_units=reused,
        optimizer=design['optimizer'],budget=design['budget'],resources=dict(max_workers=8,per_tree_limit_GiB=18,min_available_GiB=40,launch_reserve_GiB=18),
        operational_note='Initial proposals are outcome-independent and execute alongside G0. Adaptive proposals require frozen objective and passing diagnostics.',
        controller_source=str(SCRIPT),created_unix=time.time())
    # The selected confirmation namespace has no prior named graph records.
    for s in plan['confirmation_seeds']['topology']:
        assert not list(BASE.glob(f'*/confirmation/network_{s}.json'))
    rt.write(OUT/'plan.json',plan);rt.write(OUT/'confirmation/frozen_networks.json',dict(replication_networks={}))
    rt.write(OUT/'proposals/initial.json',dict(meta=meta,seed=2026091402,reused=reused))
    for c in candidates:validate_candidate(c)
    rt.write(OUT/'status.json',dict(status='AUTHORIZED_INITIAL_AND_CALIBRATION',new_simulations_launched=0,reused_units=len(reused),updated_unix=time.time()))
    return plan

def configure():
    physical.OUT=OUT;physical.configure()

def worker(stage,cid,topo,noise):
    configure();c=rt.read(OUT/'candidates'/f'{cid}.json');validate_candidate(c)
    physical.run.worker(stage,cid,topo,noise,60000.)

def valid_complete(path):
    if not path.exists():return False
    r=rt.read(path)
    return r.get('status')=='COMPLETE' and path.with_suffix('.npz').exists()

class DispatchWindowClosed(RuntimeError):
    """Conditional extension stopped at the authorized window, after draining."""

def response_window_closed(stage):
    window=OUT/'overnight_20260914/window.json'
    return stage in ['response','response_confirmation'] and window.exists() and time.time()>=rt.read(window)['latest_review_unix']

def run_queue(stage,ids,topologies=None,noises=None):
    import psutil
    p=rt.read(OUT/'plan.json');topologies=topologies or [2511];noises=noises or p['seeds']
    units=[(c,t,n) for c in ids for t in topologies for n in noises]
    pending=[u for u in units if not valid_complete(unit_path(stage,*u))];active={};fail=[];deferred=[]
    # Resume only this exact executor/stage/unit; never launch a duplicate while
    # an orphaned physical worker from an interrupted controller is alive.
    class Adopted:
        def __init__(self,p):self.process=p
        def poll(self):
            try:return None if self.process.is_running() and self.process.status()!=psutil.STATUS_ZOMBIE else 0
            except psutil.NoSuchProcess:return 0
    for q in psutil.process_iter(['pid','cmdline']):
        cmd=q.info['cmdline'] or []
        if str(SCRIPT) not in cmd or 'worker' not in cmd:continue
        def arg(key):return cmd[cmd.index(key)+1] if key in cmd else None
        if arg('--stage')!=stage:continue
        u=(arg('--candidate'),int(arg('--topology')),int(arg('--noise')))
        if u not in pending:continue
        log=(OUT/'logs'/f'{stage}_{u[0]}_{u[1]}_{u[2]}.log').open('a')
        active[q.pid]=(Adopted(q),u,log);pending.remove(u)
    while pending or active:
        peak=0.;details=[]
        for pid,(proc,u,log) in list(active.items()):
            code=proc.poll()
            if code is not None:
                log.close();del active[pid]
                if code or not valid_complete(unit_path(stage,*u)):fail.append(dict(unit=u,exit=code,log=log.name))
                continue
            try:
                q=psutil.Process(pid);rss=sum(z.memory_info().rss for z in [q]+q.children(recursive=True))/2**30
                if rss>18 or psutil.virtual_memory().available/2**30<30:
                    for z in reversed([q]+q.children(recursive=True)):z.terminate()
                    fail.append(dict(unit=u,reason='MEMORY_GUARD',rss_GiB=rss))
                peak+=rss;details.append(dict(pid=pid,unit=u,rss_GiB=rss,created_unix=q.create_time()))
            except psutil.NoSuchProcess:pass
        if fail:pending=[]
        if pending and response_window_closed(stage):deferred.extend(pending);pending=[]
        while pending and len(active)<8 and not fail:
            available=psutil.virtual_memory().available/2**30
            reserved_growth=sum(max(0,18-r['rss_GiB']) for r in details)
            if available-reserved_growth<40+18:break
            u=pending.pop(0);log=(OUT/'logs'/f'{stage}_{u[0]}_{u[1]}_{u[2]}.log').open('a')
            proc=subprocess.Popen([rt.PYTHON,'-u',str(SCRIPT),'worker','--stage',stage,'--candidate',u[0],'--topology',str(u[1]),'--noise',str(u[2])],cwd=ROOT,env=rt.ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[proc.pid]=(proc,u,log);details.append(dict(pid=proc.pid,unit=u,rss_GiB=0,created_unix=psutil.Process(proc.pid).create_time()))
            with (OUT/'dispatch.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,unit=u,pid=proc.pid,time=time.time()))+'\n')
        rt.write(OUT/'status.json',dict(status='DRAINING_ENGINEERING_FAILURE' if fail else 'DRAINING_WINDOW_END' if deferred else 'SIMULATING',stage=stage,
            planned_units=len(units),complete=sum(valid_complete(unit_path(stage,*u)) for u in units),active=details,queued=len(pending),failures=fail,updated_unix=time.time()))
        if active:time.sleep(10)
        elif pending:time.sleep(10)
    if fail:raise RuntimeError(json.dumps(fail))
    if deferred:
        rt.write(OUT/f'{stage}_window_paused.json',dict(status='WINDOW_END_NO_NEW_DISPATCH',unstarted=deferred,time=time.time()))
        raise DispatchWindowClosed('Conditional response window ended; active trajectories were preserved to completion')
    rt.write(OUT/f'{stage}_simulation_complete.json',dict(units=units,complete=len(units),time=time.time()))

def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','initial','worker']);a.add_argument('--stage',default='initial');a.add_argument('--candidate');a.add_argument('--topology',type=int);a.add_argument('--noise',type=int);v=a.parse_args()
    if v.action=='prepare':print(json.dumps(dict(initial=len(prepare()['initial_ids']))));return
    if v.action=='worker':worker(v.stage,v.candidate,v.topology,v.noise);return
    p=prepare()
    with (OUT/'initial_controller.lock').open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);run_queue('initial',p['initial_ids'])

if __name__=='__main__':main()
