#!/usr/bin/env python3
"""Resumable paired multidimensional development experiment, one bounded round."""
from pathlib import Path
import argparse
import copy
import fcntl
import json
import os
import pickle
import secrets
import shutil
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_topic4_xy_research as base
from src.topic4_xy_search import field_descriptor, geometry_allowed, audit_geometry
from src.topic4_multidimensional_parameters import DEFAULTS
read,write,sha=base.read,base.write,base.sha
OUT=ROOT/'results/topic4_sef_hfo/multidimensional_interictal_pilot_round1'
WORKER=ROOT/'scripts/run_topic4_multidimensional_worker.py'
SEEDS=[2511,2512,2513,2514]


def status(state, **kw):
    write(OUT/'status.json',{'status':state,'updated_unix':time.time(),**kw})


def design():
    path=OUT/'design.json'
    if path.exists():return read(path)
    pos=base.positions()
    historical=[[4.19921432,9.12890135],[16.47920304,3.96551153]]
    v4=read(ROOT/'results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/execution/race_001/candidate_manifest.json')
    component=next(r for r in v4['candidates'] if r['candidate_id']=='component_r001_015')['node_field']['centers_mm']
    v3=read(ROOT/'results/topic4_sef_hfo/joint_rank_space_dual_core_search_v3/rounds/008/race_nomination.json')
    # Trace the old incumbent by artifact rather than hard-coding its position.
    files=list((ROOT/'results/topic4_sef_hfo/joint_rank_space_dual_core_search_v3/execution').glob('*/candidate_manifest.json'))
    incumbent=None
    for f in files:
        for r in read(f)['candidates']:
            if r['candidate_id']=='replicated_r008_003':incumbent=r['node_field']['centers_mm']
    if incumbent is None:raise RuntimeError('old joint incumbent geometry missing')
    anchors={'historical':historical,'support_rank':component,'old_joint':incumbent}
    arms=[('baseline',{}, {}, {})]
    for name,lo,hi in [('E_to_E_weight_scale',.85,1.15),('E_to_I_weight_scale',.85,1.15),
                       ('I_to_E_weight_scale',.85,1.15),('tau_d_GABA_ms',12.,24.)]:
        for level,val in [('low',lo),('high',hi)]:arms.append((name+'_'+level,{name:val},{},{}))
    for level,val in [('low',.7),('high',1.3)]:arms.append(('vth_'+level,{}, {'node_gain':val},{}))
    for level,val in [('minus20',base.THETA-20),('plus20',base.THETA+20)]:
        arms.append(('EE_axis_'+level,{}, {}, {'ellipse_angle_deg':val}))
    arms.append(('EE_weight_AR1',{}, {}, {'ellipse_aspect_ratio':1.}))
    rows=[]
    def row(cid,centers,arm,parameters=None,node=None,mechanisms=None,anchor=None):
        r=base.decorate({'candidate_id':cid,'node_field':field_descriptor(centers),
             'domain':'whole_sheet','proposal':'prespecified_paired_parameter_test',
             'anchor':anchor,'arm':arm,'geometry':audit_geometry(pos,centers,1499)})
        r['dynamic_parameters']={**DEFAULTS,**(parameters or {})}
        r['node_mapping'].update(node or {});r['mechanisms'].update(mechanisms or {})
        return r
    for anchor,centers in anchors.items():
        for arm,p,n,m in arms:rows.append(row(anchor+'__'+arm,centers,arm,p,n,m,anchor))
    master=secrets.randbits(32);rng=np.random.default_rng(master)
    for i in range(4):
        for _ in range(10000):
            centers=rng.uniform(.75,19.25,(2,2))
            if geometry_allowed(centers,pos,domain='interior'):break
        else:raise RuntimeError('random geometry rejection exhausted')
        anchor=f'random_{i:02d}'
        rows.append(row(anchor+'__baseline',centers,'baseline',anchor=anchor))
        p={'E_to_E_weight_scale':float(rng.uniform(.85,1.15)),
           'E_to_I_weight_scale':float(rng.uniform(.85,1.15)),
           'I_to_E_weight_scale':float(rng.uniform(.85,1.15)),
           'tau_d_GABA_ms':float(rng.uniform(12,24))}
        rows.append(row(anchor+'__joint_random',centers,'joint_random',p,
                        {'node_gain':float(rng.uniform(.7,1.3))},
                        {'ellipse_angle_deg':float(base.THETA+rng.uniform(-20,20)),
                         'ellipse_aspect_ratio':float(rng.uniform(1,2.5))},anchor))
    d={'version':'multidimensional_interictal_pilot_round1','candidates':rows,
       'master_seed':master,'network_and_dynamics_seeds':SEEDS,'duration_ms':12000.,
       'n_jobs':len(rows)*len(SEEDS),'maximum_workers':8,'worker_reservation_gib':18,
       'global_memory_reserve_gib':40,'minimum_free_disk_gib':30,
       'single_factor_arms_per_anchor':len(arms),'historical_mapping':'historical positions with current signed-depth budget; not an exact historical model',
       'axis_intervention':'fixed topology incoming-EE-sum-preserving ellipse reweighting; graph-generation axis remains frozen',
       'dose_levels':'prespecified engineering sensitivity ranges, not patient-derived biological confidence intervals',
       'seed_pairing':'common topology and dynamics seeds across parameter arms; reused seeds for development, not independent confirmation',
       'noise_law':'frozen global and spatial OU; no event-specific stimulation; Z/M off',
       'round_end':'complete paired evaluation and raw diagnostics; no automatic next round, substrate freeze or Fig5 release',
       'evaluator_scope':'new calibrated development diagnostics; full final acceptance evaluator remains unqualified'}
    write(path,d);return d


def prepare_phase(name,rows,seeds,duration,source_hashes,*,legacy=False):
    folder=OUT/'execution'/name;folder.mkdir(parents=True,exist_ok=True)
    cp=folder/'execution_config.json';mp=folder/'candidate_manifest.json';sp=folder/'runtime_snapshot.json'
    if cp.exists():
        base.verify_sources(read(sp)['source_hashes'])
        for p in (cp,mp):
            if read(sp)['input_hashes'].get(str(p.resolve()))!=sha(p):raise RuntimeError('phase input changed')
        return cp,mp,sp
    old=read(ROOT/'results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4/execution/race_001/execution_config.json')
    cfg=copy.deepcopy(old)
    cfg.update(scientific_role='development_only_vth_dual_core_xy_research' if legacy else 'development_only_multidimensional_interictal_pilot',
               output_root=str(folder),candidate_manifest=str(mp),corrected_networks={str(s):base.network_record(s) for s in seeds})
    cfg['search']={'fit_network_seeds':seeds,'simulation':{'duration_ms':duration,'early_stop_runaway':True,'late_runaway_is_invalid':True},
                   'contact_readout':old['search']['contact_readout']}
    write(cp,cfg);write(mp,{'config_sha256':sha(cp),'candidates':rows,'frozen_before_simulation':True})
    write(sp,{'source_hashes':source_hashes,'input_hashes':{str(p.resolve()):sha(p) for p in (cp,mp)},
              'identity_kind':'source_hash_snapshot','not_final_substrate_freeze':True})
    return cp,mp,sp


def worker_complete(path, snapshot):
    if not path.exists():return False
    d=read(path)
    return d['status']=='REV12ND_NODE_WORKER_COMPLETE' and Path(d['arrays']['path']).exists() and sha(d['arrays']['path'])==d['arrays']['sha256'] and d['provenance']['source_hash_snapshot']['sha256']==sha(snapshot)


def outstanding_snn_reserve():
    reserve=0.
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            if not any(Path(os.fsdecode(a)).name in ('run_topic4_rev12_node_worker.py','run_topic4_multidimensional_worker.py') for a in args):continue
            rss=int((p/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')/1024**3
            limit=next(line.split()[3] for line in (p/'limits').read_text().splitlines() if line.startswith('Max address space'))
            budget=float(limit)/1024**3 if limit!='unlimited' else 18.
            reserve+=max(0.,budget-rss)
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return reserve


def run_phase(name,rows,seeds,duration,lock,maximum_workers,*,legacy=False):
    cp,mp,sp=prepare_phase(name,rows,seeds,duration,lock,legacy=legacy)
    folder=cp.parent;workers=folder/'workers';logs=folder/'run_logs'
    workers.mkdir(exist_ok=True);logs.mkdir(exist_ok=True)
    jobs=[(r['candidate_id'],s) for r in rows for s in seeds]
    def output(j):return workers/f'{j[0]}_seed_{j[1]}.json'
    pending=[j for j in jobs if not worker_complete(output(j),sp)];complete=len(jobs)-len(pending)
    active={};failures=[]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    while pending or active:
        for j,(p,f) in list(active.items()):
            code=p.poll()
            if code is None:continue
            f.close();del active[j]
            if code!=0 or not worker_complete(output(j),sp):failures.append({'job':j,'exit':code})
            else:complete+=1
        if failures:
            status('WORKER_FAILURE_DRAINING',phase=name,failures=failures,running=len(active))
            if not active:raise RuntimeError(f'worker failures: {failures}')
            time.sleep(5);continue
        base.verify_sources(lock)
        disk_ok=shutil.disk_usage(OUT).free>30*1024**3
        allowance=max(0,int((base.available_gib()-40-outstanding_snn_reserve())/18))
        slots=min(maximum_workers-len(active),allowance,len(pending)) if disk_ok else 0
        for _ in range(slots):
            j=pending.pop(0);f=open(logs/f'{j[0]}_seed_{j[1]}.log','w')
            script=ROOT/'scripts/run_topic4_rev12_node_worker.py' if legacy else WORKER
            cmd=['/usr/bin/prlimit',f'--as={18*1024**3}','--',base.PYTHON,str(script),'--config',str(cp),'--candidate-id',j[0],
                 '--seed',str(j[1]),'--expected-commit',commit,'--runtime-manifest',str(sp),
                 '--artifact-root',str(base.ART),'--out-json',str(output(j)),
                 '--out-npz',str(output(j).with_suffix('.npz'))]
            env={**base.ENV,'LD_LIBRARY_PATH':'/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib','CUDA_VISIBLE_DEVICES':''}
            active[j]=(subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT),f)
        status('RUNNING_'+name.upper() if active else 'WAITING_RESOURCE_ADMISSION',phase=name,
               total=len(jobs),complete=complete,running=len(active),pending=len(pending),
               memory_available_gib=base.available_gib(),active=[{'candidate_id':j[0],'seed':j[1],'pid':p.pid} for j,(p,f) in active.items()])
        if pending or active:time.sleep(10)
    write(folder/'completion.json',{'status':'COMPLETE','jobs':len(jobs),'runtime_sha256':sha(sp)})


def compare_canary():
    stem='historical__baseline_seed_2511.npz'
    a=OUT/'execution/canary/workers'/stem;b=OUT/'execution/observer_parity/workers'/stem
    with np.load(a) as left,np.load(b) as right:
        keys=['active_fraction','contact_envelope','onsets','event_returned','sheet_activity_counts','h','delta_vtheta']
        checks={k:bool(np.array_equal(left[k],right[k],equal_nan=True)) for k in keys}
    write(OUT/'observer_parity.json',{'checks':checks,'pass':all(checks.values()),'new_arrays_sha256':sha(a),'legacy_arrays_sha256':sha(b)})
    if not all(checks.values()):raise RuntimeError('observer/baseline parity failed')


def verify_parameter_canaries():
    rows=[]
    for p in sorted((OUT/'execution/canary/workers').glob('*.json')):
        d=read(p);a=d['multidimensional_parameter_audit'];effective=a['effective']
        checks={'requested_equals_effective':a['requested']==effective,
                'topology_preserved':a['topology_and_delay_assignments_preserved']}
        for group,keys in [('ampa_by_delay',['E_to_E_weight_scale','E_to_I_weight_scale']),
                           ('gaba_by_delay',['I_to_E_weight_scale','I_to_I_weight_scale'])]:
            dose=a['sparse_pathways'][group]
            checks[group+'_dose']=bool(np.allclose(dose['after_sum'],np.asarray(dose['before_sum'])*[effective[k] for k in keys],rtol=2e-6))
        with np.load(d['arrays']['path']) as z:
            checks['finite_activity']=bool(np.isfinite(z['active_fraction']).all())
            checks['native_movie_present']=z['sheet_activity_counts'].ndim==3
        rows.append({'worker':str(p),'checks':checks,'pass':all(checks.values())})
    result={'canaries':rows,'pass':len(rows)==7 and all(r['pass'] for r in rows)}
    write(OUT/'parameter_canary_audit.json',result)
    if not result['pass']:raise RuntimeError('actual parameter canary audit failed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--maximum-workers',type=int,default=8);args=parser.parse_args()
    if not 1<=args.maximum_workers<=8:raise ValueError('maximum workers 1..8')
    OUT.mkdir(parents=True,exist_ok=True)
    guard=open(OUT/'controller.lock','a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ev=read(OUT/'evaluation_manifest.json')
    if not ev['pilot_dispatch_allowed'] or sha(OUT/'evaluator.pkl')!=ev['evaluator_sha256']:raise RuntimeError('development evaluator sanity checks not passed')
    d=design()
    if args.prepare_only:print(json.dumps({'n_candidates':len(d['candidates']),'n_jobs':d['n_jobs'],'master_seed':d['master_seed']}));return
    lp=OUT/'source_lock.json'
    if lp.exists():lock=read(lp)['source_hashes'];base.verify_sources(lock)
    else:
        lock=base.source_hashes();write(lp,{'source_hashes':lock,'design_sha256':sha(OUT/'design.json'),
            'evaluation_manifest_sha256':sha(OUT/'evaluation_manifest.json')})
    for filename,key in [('design.json','design_sha256'),('evaluation_manifest.json','evaluation_manifest_sha256')]:
        if sha(OUT/filename)!=read(lp)[key]:raise RuntimeError('frozen design/evaluator changed')
    rows=d['candidates']
    canary_arms={'baseline','E_to_E_weight_scale_high','E_to_I_weight_scale_high','I_to_E_weight_scale_high','tau_d_GABA_ms_low','vth_high','EE_axis_plus20'}
    canary=[r for r in rows if r['anchor']=='historical' and r['arm'] in canary_arms]
    run_phase('canary',canary,[2511],2000.,lock,min(4,args.maximum_workers))
    run_phase('observer_parity',[r for r in canary if r['arm']=='baseline'],[2511],2000.,lock,1,legacy=True)
    compare_canary()
    verify_parameter_canaries()
    run_phase('paired_round1',rows,SEEDS,d['duration_ms'],lock,args.maximum_workers)
    status('ANALYZING_COMPLETED_MULTIDIMENSIONAL_ROUND')
    subprocess.run([base.PYTHON,str(ROOT/'scripts/analyze_topic4_multidimensional_pilot.py')],cwd=ROOT,env=base.ENV,check=True)
    status('ROUND1_COMPLETE_PENDING_SCIENTIFIC_REVIEW',n_candidates=len(rows),n_jobs=d['n_jobs'],
           automatic_next_round=False,final_substrate_frozen=False,fig5_released=False)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('ERROR',error=repr(exc));raise
