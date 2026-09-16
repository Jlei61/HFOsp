"""Bounded geometry/threshold response experiments; no propagation acceptance by loss."""
from pathlib import Path
import argparse,copy,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import long_topic4_core_extent as long
from scripts import pilot_topic4_core_extent as pilot
rt=long.rt
OUT=Path('/data/hfosp/topic4_sef_hfo/geometry_threshold_refinement_20260909')
OLD=long.OUT;SCRIPT=Path(__file__).resolve();ORIGINAL_BUILD=pilot.build_modified


def transformed_delta(delta,h,mean_shift=0.,dispersion=1.):
    out=np.asarray(delta).copy();inside=np.asarray(h)>0
    if mean_shift==0 and dispersion==1:return out
    mean=out[inside].mean()
    out[inside]=mean+dispersion*(out[inside]-mean)+mean_shift
    return out


def build_modified(design,seed,c):
    built,control,details=ORIGINAL_BUILD(design,seed,c)
    sub,cand,transition,execution,audit,network=built
    d0=sub.delta_vtheta.copy();sub.delta_vtheta=transformed_delta(d0,sub.h_e,c.get('threshold_mean_shift_mV',0.),c.get('threshold_dispersion_scale',1.))
    sub.vtheta[:sub.n_e]=sub.engine['v_base']+sub.delta_vtheta
    groups,indices,loading,radius=control
    if c.get('state_support')=='follow_E_geometry':
        pos=np.asarray(sub.net['pos']);dd=np.linalg.norm(pos[:,None]-np.asarray(c['centers_mm'])[None],axis=2);near=dd.argmin(1);inside=(dd<=np.asarray(c['radii_mm'])[None]).any(1)
        groups=copy.deepcopy(groups)
        for k,name in enumerate(['coreA','coreB']):groups[name+'I']=np.flatnonzero((np.arange(len(pos))>=sub.n_e)&inside&(near==k))
        aa,bb=groups['coreAI'],groups['coreBI'];nmin=min(len(aa),len(bb))
        if nmin==0:raise ValueError('empty local I projection')
        indices=np.r_[aa,bb];loading=np.r_[np.full(len(aa),nmin/len(aa)),np.full(len(bb),-nmin/len(bb))]
    identity=rt.static_identity(sub,audit)
    assert all(identity[k]==details['baseline_identity'][k] for k in identity if k.startswith(('positions_','ampa_','gaba_')))
    d=sub.delta_vtheta;mask=sub.h_e>0
    details.update(applied_identity=identity,extension_sha256=rt.sha(SCRIPT),threshold_lowering_total_mV=float(-d[d<0].sum()),threshold_raising_total_mV=float(d[d>0].sum()),
       n_threshold_lowered=int((d<0).sum()),n_threshold_raised=int((d>0).sum()),inside_mean_threshold_mV=float(sub.vtheta[:sub.n_e][mask].mean()),inside_threshold_std_mV=float(sub.vtheta[:sub.n_e][mask].std()),
       requested_mean_shift_mV=c.get('threshold_mean_shift_mV',0.),actual_mean_shift_mV=float((d-d0)[mask].mean()),requested_dispersion_scale=c.get('threshold_dispersion_scale',1.),
       state_support=c.get('state_support','historical_fixed'),applied_I_indices_sha256=pilot.array_sha256(indices),applied_I_loading_sha256=pilot.array_sha256(loading),n_state_I=len(indices),
       unchanged=['graph edges and delays','EE/EI/IE/II weights','external noise law','GABA','observer'],qualification='actual executor arrays, after geometry and threshold application')
    if not np.isfinite(sub.vtheta).all() or sub.vtheta.min()<=sub.params.V_reset:raise ValueError('invalid threshold field')
    return built,(groups,indices,loading,radius),details


def prepare():
    OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'plan.json').exists():return
    plan=copy.deepcopy(rt.read(OLD/'plan.json'));base=plan['candidates'][0];ab=next(c for c in plan['candidates'] if c['id']=='expand_AB_2.5');aa=next(c for c in plan['candidates'] if c['id']=='expand_A_4')
    audit=rt.read(OUT/'anchor_audit.json');assert max(r['error'] for r in audit['rows'])<1e-5
    weighted=[r['weighted_center'] for r in audit['rows']];cs=[]
    def make(parent,cid,group,**changes):
        c=copy.deepcopy(parent);c.update(id=cid,parent_id=parent['id'],stage=group,preserve_exact_baseline=False,**changes);cs.append(c);return c
    for par,name in [(base,'base'),(ab,'AB25')]:make(par,'weighted_'+name,'weighted_initialization',centers_mm=weighted)
    for core,axis,label in [(0,1,'A_y'),(1,0,'B_x')]:
        for sign in [-1,1]:
            centers=copy.deepcopy(ab['centers_mm']);centers[core][axis]+=.75*sign
            make(ab,'xy_'+label+('_plus' if sign==1 else '_minus'),'paired_center_shift',centers_mm=centers)
    for par,name in [(ab,'AB25'),(aa,'A4')]:
        make(par,name+'_dose_preserved','threshold_dose_control',dose_matched=True)
        make(par,name+'_mean_raise025','threshold_mean',threshold_mean_shift_mV=.25)
        make(par,name+'_dispersion150','threshold_dispersion',threshold_dispersion_scale=1.5)
        make(par,name+'_state_extent','I_state_geometry_control',state_support='follow_E_geometry')
    plan.update(output_root=str(OUT),candidates=cs,references=rt.read(OLD/'plan.json')['candidates'],max_parallel=6,training_seeds=[847101,847102],comparison_seeds=[847101,847102],confirmation_seed=847199,
      duration_ms=90000.,maximum_formal_runs=28,source_sha256=rt.sha(pilot.SCRIPT),runner_sha256=rt.sha(long.SCRIPT),extension_sha256=rt.sha(SCRIPT),
      stages=[dict(duration_ms=90000,seeds=[847101,847102])],stop='28 formal runs then patient/model visual scientific review; no further EE/II/input interventions until user visual acceptance',
      score='old L_off diagnostic only; no automatic loss-based best-model or success decision',proposal_rule='explicit paired proposals frozen before running, no adaptive loss optimizer',
      effect_scope='geometry prior and threshold/state-support perturbations, same topology, two noise/state replays')
    rt.write(OUT/'plan.json',plan)
    (OUT/'candidates').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    for c in cs:rt.write(OUT/'candidates'/(c['id']+'.json'),c)


def worker(cp,seed,duration,canary=False):
    plan=rt.read(OUT/'plan.json')
    if rt.sha(SCRIPT)!=plan['extension_sha256']:raise RuntimeError('extension differs from frozen plan')
    long.OUT=OUT
    pilot.build_modified=build_modified
    long.worker(cp,seed,duration,canary)


def run_jobs(cs,seeds,duration,canary=False):
    import psutil
    pending=[(c,s) for c in cs for s in seeds];active={};failures=[];maxp=2 if canary else rt.read(OUT/'plan.json')['max_parallel']
    def result(c,s):return OUT/('canary' if canary else f'duration_{round(duration)}')/'units'/c['id']/str(s)/'workers/trajectory.json'
    pending=[(c,s) for c,s in pending if not result(c,s).exists()]
    while pending or active:
        rss={}
        for pid,(pr,c,s,log) in list(active.items()):
            if pr.poll() is not None:
                log.close();del active[pid]
                if pr.returncode or not result(c,s).exists():failures.append(dict(candidate=c['id'],seed=s,exit_code=pr.returncode))
                continue
            try:
                pp=psutil.Process(pid);tree=[pp]+pp.children(recursive=True);rss[pid]=sum(x.memory_info().rss for x in tree if x.is_running())/2**30
                if rss[pid]>16 or rt.available_gib()<35:
                    for x in reversed(tree):x.terminate()
                    failures.append(dict(candidate=c['id'],seed=s,reason='RESOURCE_GUARD'))
            except psutil.NoSuchProcess:pass
        if failures:pending=[]
        reserve=sum(max(0,8-rss.get(pid,0)) for pid in active)
        while pending and len(active)<maxp and rt.available_gib()>60+reserve+8:
            c,s=pending.pop(0);cp=OUT/'candidates'/(c['id']+'.json');log=(OUT/'logs'/f'{c["id"]}_{s}_{round(duration)}.log').open('a')
            cmd=[rt.PYTHON,'-u',str(SCRIPT),'worker','--candidate',str(cp),'--seed',str(s),'--duration',str(duration)]+(['--canary'] if canary else [])
            pr=subprocess.Popen(cmd,cwd=ROOT,env=rt.ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[pr.pid]=(pr,c,s,log);reserve+=8
        rt.write(OUT/'status.json',dict(status='CANARY_RUNNING' if canary else 'RUNNING',complete=sum(result(c,s).exists() for c in cs for s in seeds),total=len(cs)*len(seeds),duration_ms=duration,
           active=[dict(pid=pid,candidate=c['id'],seed=s,tree_rss_gib=rss.get(pid)) for pid,(pr,c,s,log) in active.items()],queued=len(pending),failures=failures,updated_unix=time.time()))
        if pending or active:time.sleep(10)
    if failures:raise RuntimeError(str(failures))


def controller():
    plan=rt.read(OUT/'plan.json')
    if rt.read(OUT/'canary_audit.json')['status']!='PASS':raise RuntimeError('physical canary not passed')
    run_jobs(plan['candidates'],plan['training_seeds'],plan['duration_ms'])
    rt.write(OUT/'status.json',dict(status='ANALYZING',updated_unix=time.time()))
    pr=subprocess.run([rt.PYTHON,str(ROOT/'scripts/analyze_topic4_geometry_threshold_refinement.py')],cwd=ROOT,env=rt.ENV)
    if pr.returncode:raise RuntimeError('analysis failed')
    rt.write(OUT/'status.json',dict(status='COMPLETE_PENDING_PATIENT_MODEL_VISUAL_REVIEW',formal_runs=28,updated_unix=time.time()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','worker','canary','controller']);ap.add_argument('--candidate',type=Path);ap.add_argument('--seed',type=int);ap.add_argument('--duration',type=float,default=90000);ap.add_argument('--canary',action='store_true');args=ap.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='worker':worker(args.candidate,args.seed,args.duration,args.canary)
    elif args.action=='canary':
        pp=rt.read(OUT/'plan.json');run_jobs([next(c for c in pp['candidates'] if c['id']==name) for name in ['A4_dispersion150','AB25_state_extent']],[847101],2000,True)
    else:
        try:controller()
        except Exception as exc:
            rt.write(OUT/'status.json',dict(status='FAILED',error=repr(exc),updated_unix=time.time()));raise
