"""User-authorized bounded continuation AFTER the original 140-run package.

Same physical executor and objective. A 2x3x3 factorial reuses six completed
conditions; geometry probes and new-noise replication bring the NEW budget to
at most 64 sixty-second runs. No route/label/rotation enters physical drive.
"""
from pathlib import Path
import argparse,copy,fcntl,json,os,subprocess,sys,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
from scripts import run_topic4_shape_output_response as base
rt=base.rt
PARENT=Path('/data/hfosp/topic4_sef_hfo/core_shape_output_response_20260911')
OUT=Path('/data/hfosp/topic4_sef_hfo/core_recruitment_tradeoff_followup_20260912')
SCRIPT=Path(__file__).resolve()


def physical(c):
    return {k:c.get(k,{}) for k in ['centers_mm','radii_mm','parameters','shape','outgoing','core_mean_rate_scale','core_ou_correlation']}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():return rt.read(OUT/'plan.json')
    pp=rt.read(PARENT/'plan.json');candidates=[];reuse=[];lookup={};factorial=[]
    for shape in ['circle','ellipse4']:
      baseline=rt.read(PARENT/'candidates'/f'up3__{shape}.json')
      for ee in [1.,1.125,1.25]:
       for ei in [1.,.875,.75]:
        oldid=None
        if ee==1 and ei==1:oldid=f'up3__{shape}'
        elif ee==1.25 and ei==1:oldid=f'up3__{shape}__EE_core_to_out_scale_1.25'
        elif ee==1 and ei==.75:oldid=f'up3__{shape}__EI_same_core_scale_0.75'
        cid=oldid or f'follow_{shape}_out{ee:g}_EI{ei:g}'
        c=copy.deepcopy(baseline);c['parameters'].update(EE_core_to_out_scale=ee,EI_same_core_scale=ei)
        c.update(id=cid,stage='response',contrast='factorial',display_name=f'{"圆核" if shape=="circle" else "椭圆4:1"}｜离核EE×{ee:g}，同核EI×{ei:g}',
                 factorial=dict(shape=shape,EE_out=ee,EI=ei),origin='reused_parent_condition' if oldid else 'new_factorial')
        lookup[(shape,ee,ei)]=cid;candidates.append(c);factorial.append(cid)
        if oldid:reuse.append(dict(candidate=cid,source_stage='shape_range' if '__E' not in oldid else 'response',source_candidate=oldid))
    for c in candidates:
        f=c['factorial'];s,e,i=f['shape'],f['EE_out'],f['EI']
        c['comparison']=lookup[(s,e,1.)] if i!=1 else lookup[(s,1.,1.)]
        if e==i==1 and s=='ellipse4':c['comparison']=lookup[('circle',1.,1.)]
    anchors=[lookup[('circle',1.25,1.)],lookup[('ellipse4',1.,.75)]]
    for anchor in anchors:
      ref=next(c for c in candidates if c['id']==anchor)
      for label,dx,dy,radius in [('x_minus075',-.75,0,None),('x_plus075',.75,0,None),('y_minus10',0,-1.,None),('y_plus10',0,1.,None),('radius205',0,0,2.05),('radius235',0,0,2.35)]:
        c=copy.deepcopy(ref);c.update(id=anchor+'__'+label,comparison=anchor,contrast='geometry',origin='new_geometry',display_name=ref['display_name']+'｜'+label)
        c['centers_mm'][0][0]+=dx;c['centers_mm'][0][1]+=dy
        if radius is not None:
            c['parameters']['radius_A_mm']=radius;c['radii_mm'][0]=radius
            c.setdefault('shape',{}).update(match_dose=True,dose_reference_radius_mm=ref['parameters']['radius_A_mm'])
        c['geometry_probe']=dict(dx_mm=dx,dy_mm=dy,radius_A_mm=radius,threshold_dose_matched=radius is not None)
        c.pop('factorial',None);candidates.append(c)
    assert len(candidates)==30 and len(reuse)==6
    sources=dict(pp['source_snapshot']);sources[str(SCRIPT)]=rt.sha(SCRIPT)
    plan=dict(schema='topic4.recruitment_tradeoff_continuation.v1',physics=pp['physics'],source_snapshot=sources,
        parent=str(PARENT),parent_plan_sha256=rt.sha(PARENT/'plan.json'),authorization='2026-09-12 user: after these 140 finish, continue while I sleep',
        start_condition='parent 140 formal runs complete and parent analysis reaches ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW; engineering failures block dispatch',
        topology_seed=2511,seeds=[847101,847102],duration_ms=60000.,analysis=pp['analysis'],runaway=pp['runaway'],resources=pp['resources'],
        candidates=candidates,reuse=reuse,factorial_candidates=factorial,geometry_anchors=anchors,
        budget=dict(new_factorial_runs=24,new_geometry_runs=24,reused_parent_runs=12,replication_max_new=16,total_max_new=64),
        confirmation=dict(candidates=anchors+[lookup[('circle',1.,1.)],lookup[('ellipse4',1.,1.)]],topology_seeds=[2611,2612],noise_seeds=[847301,847302],
            selection='Two best NEW conditions by fixed network-equal L_search over both training replays (N>=16 each), plus each direct parent; at most four unique conditions. If fewer than two scorable NEW conditions, append predeclared circle/out1.25 and ellipse4/EI.75 and their direct baselines, capped at four.',
            independence='2611/2612 are the two already-used confirmation topologies; 847301/847302 are new noise streams. This is not an unseen-topology claim.'),
        stop='After <=64 NEW runs and analysis, stop for scientific review. Do not freeze model, enter Fig5, alter objective, or add another round automatically.',
        scientific_question='Does core shape modify the participation-versus-timing response to outgoing EE and local EI, and can bounded geometry shifts reduce the residual? Detailed patient routes remain diagnostic only.')
    for d in ['candidates','logs','confirmation','analysis']:(OUT/d).mkdir(exist_ok=True)
    for c in candidates:rt.write(OUT/'candidates'/f'{c["id"]}.json',c)
    rt.write(OUT/'confirmation/frozen_networks.json',rt.read(PARENT/'confirmation/frozen_networks.json'))
    (OUT/'global_graph_cache').symlink_to(PARENT/'global_graph_cache',target_is_directory=True)
    rt.write(OUT/'plan.json',plan);rt.write(OUT/'status.json',dict(status='WAITING_FOR_PARENT_140',updated_unix=time.time(),budget=plan['budget']))
    return plan


def setup():
    base.OUT=OUT;base.SCRIPT=SCRIPT;base.configure()


def parent_ready():
    status=rt.read(PARENT/'status.json')
    if status.get('failures') or 'FAILURE' in status.get('status',''):return 'ENGINEERING_BLOCK',status
    paths=[p for stage in ['shape_range','response','confirmation'] for p in (PARENT/stage).glob('units/*/*/workers/trajectory.json')]
    done=len(paths)==140 and all(rt.read(p).get('status')=='COMPLETE' for p in paths)
    return ('READY' if done and status.get('status')=='ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW' else 'WAIT'),dict(parent_status=status.get('status'),formal_complete=len(paths),analysis=rt.read(PARENT/'analysis/status.json'))


def reuse_and_review(plan):
    for rec in plan['reuse']:
        wanted=rt.read(OUT/'candidates'/f'{rec["candidate"]}.json');actual=rt.read(PARENT/'candidates'/f'{rec["source_candidate"]}.json')
        assert physical(wanted)==physical(actual),rec['candidate']
        for seed in plan['seeds']:
            src=PARENT/rec['source_stage']/'units'/rec['source_candidate']/f'2511_{seed}'
            if not base.base.complete(src/'workers/trajectory.json'):raise RuntimeError('unverified reused unit')
            r=rt.read(src/'workers/trajectory.json');assert r['job']['duration_ms']==60000
            dst=OUT/'response/units'/rec['candidate']/f'2511_{seed}';dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():dst.symlink_to(src,target_is_directory=True)
    # Archive the completed parent evidence used before new proposals are dispatched.
    review=OUT/'parent_review';review.mkdir(exist_ok=True)
    for name in ['counts.csv','observations.csv','response_evidence.csv','paired_run_differences.csv','scientific_note.md','status.json']:
        src=PARENT/'analysis'/name;dst=review/name
        if not dst.exists():dst.write_bytes(src.read_bytes())
    rt.write(review/'manifest.json',dict(parent_status=rt.read(PARENT/'status.json'),sources={str(PARENT/'analysis'/n):rt.sha(PARENT/'analysis'/n) for n in ['counts.csv','observations.csv','scientific_note.md']},
        decision='Proceed with frozen crossed EE/EI-by-shape and bounded geometry probes. Parent confirmation is retained as evidence, not a pass gate for an improved-workpoint claim.',
        already_seen_validation='Subsequent design was informed by this development history; parent confirmation is no longer an untouched evaluation for this continuation.',human_visual_acceptance=False))
    (OUT/'rotation').mkdir(exist_ok=True)
    for stage in ['shape_range','response','confirmation']:
        for path in (PARENT/stage).glob('units/*/*/workers/trajectory.json'):
            key=hashlib.sha256(str(path).encode()).hexdigest()[:20];src=PARENT/'rotation'/key;dst=OUT/'rotation'/key
            if (src/'result.json').exists() and not dst.exists():dst.symlink_to(src,target_is_directory=True)


def launch_analysis():
    children=[]
    for args,label in [(['observer'],'analysis'),(['rotation','--gpu','0'],'rotation_gpu0'),(['rotation','--gpu','1'],'rotation_gpu1')]:
        log=(OUT/'logs'/f'{label}.log').open('a');env=dict(rt.ENV)
        if label.startswith('rotation'):env['CUDA_VISIBLE_DEVICES']='0,1'
        cmd=[rt.PYTHON,'-u',str(ROOT/'scripts/analyze_topic4_recruitment_tradeoff_followup.py'),*args]
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();children.append(dict(pid=proc.pid,role=label))
    rt.write(OUT/'analysis_processes.json',children)


def select_replication(plan):
    import csv
    rows=list(csv.DictReader((OUT/'analysis/counts.csv').open()));groups={}
    for r in rows:
        if int(r['topology'])==2511:groups.setdefault(r['candidate'],[]).append(r)
    fresh={c['id'] for c in plan['candidates'] if c['origin'].startswith('new_')};ranked=[]
    for cid,rr in groups.items():
        if cid not in fresh or len(rr)!=2 or any(int(r['primary'])<16 or not r['L_search'] for r in rr):continue
        ranked.append((sum(float(r['L_search']) for r in rr)/2,cid))
    ranked.sort();chosen=[];pairs=[];lookup={c['id']:c for c in plan['candidates']}
    for score,cid in ranked[:2]:
        parent=lookup[cid]['comparison'];pairs.append(dict(candidate=cid,parent=parent,mean_L_search=score,selection='frozen training score only'))
        chosen.extend(x for x in [cid,parent] if x not in chosen)
    for cid in plan['geometry_anchors']:
        if len(pairs)>=2:break
        if any(p['candidate']==cid for p in pairs):continue
        parent=lookup[cid]['comparison'];pairs.append(dict(candidate=cid,parent=parent,mean_L_search=None,selection='predeclared fallback pair for observation support'))
        chosen.extend(x for x in [cid,parent] if x not in chosen)
    assert len(chosen)<=4
    spec=dict(candidates=chosen[:4],pairs=pairs,ranked_new_conditions=ranked,training_score_csv_sha256=rt.sha(OUT/'analysis/counts.csv'),
        topology_seeds=plan['confirmation']['topology_seeds'],noise_seeds=plan['confirmation']['noise_seeds'],unchanged_loss=True)
    rt.write(OUT/'replication_selection.json',spec);return spec


def controller():
    plan=prepare()
    with (OUT/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            state,info=parent_ready()
            rt.write(OUT/'status.json',dict(status='WAITING_FOR_PARENT_140' if state=='WAIT' else state,parent=info,updated_unix=time.time()))
            if state=='READY':break
            if state=='ENGINEERING_BLOCK':return
            time.sleep(20)
        reuse_and_review(plan);setup();launch_analysis()
        units=[(c['id'],2511,s) for c in plan['candidates'] for s in plan['seeds']]
        base.queue('response',units,60000.,20)
        while True:
            p=OUT/'analysis/status.json'
            if p.exists() and rt.read(p).get('analyzed_runs',0)>=60:break
            time.sleep(20)
        spec=rt.read(OUT/'replication_selection.json') if (OUT/'replication_selection.json').exists() else select_replication(plan)
        units=[(c,t,s) for c in spec['candidates'] for t in spec['topology_seeds'] for s in spec['noise_seeds']]
        base.queue('confirmation',units,60000.,20)
        rt.write(OUT/'simulation_complete.json',dict(status='SIMULATIONS_COMPLETE_ANALYSIS_PENDING',new_runs=48+len(units),reused_runs=12,expected_analyzed_runs=60+len(units),updated_unix=time.time()))
        rt.write(OUT/'status.json',dict(status='SIMULATIONS_COMPLETE_ANALYSIS_PENDING',new_runs=48+len(units),reused_runs=12,updated_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','controller','worker','check']);p.add_argument('--stage');p.add_argument('--candidate');p.add_argument('--topology',type=int);p.add_argument('--seed',type=int);p.add_argument('--duration',type=float)
    a=p.parse_args()
    if a.action=='prepare':print(prepare()['budget'])
    elif a.action=='check':print(parent_ready())
    elif a.action=='worker':setup();base.run.worker(a.stage,a.candidate,a.topology,a.seed,a.duration)
    else:controller()
