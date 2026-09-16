"""Stages 2-3 of the core_connectivity_v2 search (bounded DE and new-identity confirmation).
Kept outside the frozen worker source so the screen's source snapshot stays byte-identical."""
from pathlib import Path
import argparse,os,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from src import topic4_initial_state_runtime as rt
from src import topic4_core_connectivity_v2 as v2
from scripts.run_topic4_core_connectivity_search import OUT,PARENT,prepare,queue,result_path,unit_dir

# ---------------------------------------------------------------- stage 2: bounded DE (C13)
KEY_OBSERVABLES=[('SCL_upper_participation',.05),('both_rods',.05),('ICL_contact_participation',.05),('pair_order_probability_mae',.02),('SCL_minus_ICL_lag_median_ms',2.)]
DESIGN_PRIORITY=[('EE_kernel_perp_scale','EE_core_to_out_degree_scale','EE_core_to_out_scale'),('radius_A_mm','depth_A_scale','radius_B_mm','depth_B_scale'),('EE_core_to_out_scale','IE_same_core_scale','EI_same_core_scale')]


def reflect(v,lo,hi):
    v=float(v)
    if hi<=lo:return float(lo)
    for _ in range(64):
        if v<lo:v=lo+(lo-v)
        elif v>hi:v=hi-(v-hi)
        else:return v
    return float(min(max(v,lo),hi))


def de_offspring(pop,targets,bounds,rng,*,F,CR):
    keys=list(bounds);out=[]
    for t in targets:
        others=[i for i in range(len(pop)) if i!=t];r=[int(i) for i in rng.choice(others,3,replace=False)]
        base,d1,d2=pop[r[0]]['vector'],pop[r[1]]['vector'],pop[r[2]]['vector'];x=pop[t]['vector']
        forced=int(rng.integers(len(keys)));vec={};n=0
        for j,k in enumerate(keys):
            donor=reflect(base[k]+F*(d1[k]-d2[k]),*bounds[k]);take=bool(rng.random()<CR) or j==forced
            vec[k]=donor if take else float(x[k]);n+=int(take and donor!=x[k])
        out.append(dict(target_index=int(t),donor_indices=r,forced_dim=keys[forced],vector=vec,from_donor_dims=n,F=float(F),CR=float(CR)))
    return out


def de_select(parent,child):
    if parent['loss'] is None and child['loss'] is None:return dict(id=parent['id'],reason='both_unscorable_kept_parent')
    if parent['loss'] is None:return dict(id=child['id'],reason='scorable_child_replaces_unscorable_parent')
    if child['loss'] is None:return dict(id=parent['id'],reason='unscorable_child_rejected')
    return dict(id=child['id'],reason='lower_or_equal_loss') if child['loss']<=parent['loss'] else dict(id=parent['id'],reason='parent_better')


def geometry_valid(centers,radii,sheet=20.):
    c=np.asarray(centers,float);r=np.asarray(radii,float)
    return bool(np.linalg.norm(c[1]-c[0])>r.sum() and all(min(x[0],x[1],sheet-x[0],sheet-x[1])>=rad for x,rad in zip(c,r)))


def candidate_from_vector(cid,layout,vector,plan,stage):
    prm=dict(plan['baseline_parameters']);prm.update({k:float(v) for k,v in vector.items()})
    L=next(l for l in plan['layouts'] if l['id']==layout);centers=[list(L['A']),list(L['B'])]
    adjacency=any(prm[k]!=plan['baseline_parameters'][k] for k in ('EE_core_to_out_degree_scale','EE_kernel_perp_scale','EE_kernel_parallel_scale','EE_angle_offset_deg'))
    return dict(id=cid,layout=layout,centers_mm=centers,radii_mm=[prm['radius_A_mm'],prm['radius_B_mm']],parameters=prm,changed_parameter='joint',changed_value=None,adjacency_changes=adjacency,stage=stage)


def load_rows(name):
    """Analysis tables restricted to the frozen primary (isolated-window) layer."""
    import csv
    with (OUT/'analysis'/name).open() as f:rows=list(csv.DictReader(f))
    return [r for r in rows if r.get('layer','primary')=='primary']


def _f(x):
    try:return float(x)
    except (TypeError,ValueError):return None


_CACHE={}


def unit_scores(stage,cid,topo,seed):
    from scripts import analyze_topic4_core_connectivity_search as an
    plan=rt.read(OUT/'plan.json');unit=an.load_unit(result_path(stage,cid,topo,seed),plan['analysis']['burnin_ms'])
    if unit is None:return dict(status='MISSING',loss_off=None,D_mask_off=None,L_search=None,n=0)
    r,a,ids=unit;parent=rt.read(PARENT)
    if 'ref' not in _CACHE:
        objective=rt.load_objective(parent);ev,_,patient,_,maskref,_=an.build_reference(plan);_CACHE['ref']=(objective,maskref)
    objective,maskref=_CACHE['ref']
    x=a['centroid_ms'][ids];sc=objective.score_network(x) if len(ids) else dict(loss_off=None);ms=an.jm.score_times(x,maskref) if len(ids) else dict(D_mask_off=None)
    return dict(status=r['physical_status'],n=len(ids),loss_off=sc.get('loss_off'),D_mask_off=ms.get('D_mask_off'),L_search=an.jm.combined_search_loss(sc.get('loss_off'),ms.get('D_mask_off'),maskref.a_mask),
                n_TA=int((a['event_mode'][ids]==1).sum()),n_TB=int((a['event_mode'][ids]==0).sum()))


def condition_loss(stage,cid,plan,topo=None):
    topo=plan['topology_seed'] if topo is None else topo;units=[unit_scores(stage,cid,topo,s) for s in plan['seeds']]
    loss=float(np.mean([u['L_search'] for u in units])) if all(u['L_search'] is not None for u in units) else None
    return dict(id=cid,loss=loss,units=units)


def select_active_parameters(plan):
    """Paired evidence from both noises (same sign, above threshold) on any key observable, any mode, any layout."""
    rows=load_rows('paired_parameter_effects.csv');evidence={}
    for fam in v2.WEIGHT_FACTORS+v2.GEOMETRY_KEYS+(v2.DEGREE_KEY,)+v2.KERNEL_KEYS:
        hits=[];score=0.
        for layout in [l['id'] for l in plan['layouts']]:
            for value in sorted({r['changed_value'] for r in rows if r['changed_parameter']==fam}):
                for mode in ['ALL','TA','TB']:
                    for key,thr in KEY_OBSERVABLES:
                        ch=[_f(r['change']) for r in rows if r['changed_parameter']==fam and r['changed_value']==value and r['mode']==mode and r['observable']==key and r['layout']==layout]
                        ch=[c for c in ch if c is not None]
                        if len(ch)==2 and np.sign(ch[0])==np.sign(ch[1]) and min(abs(ch[0]),abs(ch[1]))>=thr:
                            hits.append(dict(layout=layout,value=_f(value),mode=mode,observable=key,change_seed1=ch[0],change_seed2=ch[1]));score+=min(abs(ch[0]),abs(ch[1]))/thr
        evidence[fam]=dict(consistent_hits=hits,response_score=score)
    ranked=sorted(evidence,key=lambda k:-evidence[k]['response_score']);responsive=[k for k in ranked if evidence[k]['consistent_hits']]
    chosen=[]
    for group in DESIGN_PRIORITY:
        for k in group:
            if k in responsive and k not in chosen and len(chosen)<plan['adaptive']['maximum_active_dimensions']:chosen.append(k)
    for k in responsive:
        if k not in chosen and len(chosen)<plan['adaptive']['maximum_active_dimensions']:chosen.append(k)
    return chosen,evidence,ranked


def select_populations(plan,scores):
    """Two layouts: most scorable paired runs, then complementary strengths (SCL upper participation vs TA support)."""
    obs=load_rows('run_mode_observations.csv');rows=[]
    for layout in [l['id'] for l in plan['layouts']]:
        sc=[s for s in scores if s['layout']==layout];scorable=sum(1 for s in sc if s['scorable'])
        base=[o for o in obs if o['candidate']==f'{layout}__baseline' and o['source']=='v2']
        rows.append(dict(layout=layout,scorable_runs=scorable,baseline_SCL_upper=[_f(o['SCL_upper_participation']) for o in base if o['mode']=='ALL'],
            baseline_n_TA=[int(_f(o['n']) or 0) for o in base if o['mode']=='TA'],best_L_search=min([s['L_search'] for s in sc if s['scorable']],default=None)))
    order=sorted(rows,key=lambda r:(-r['scorable_runs'],r['best_L_search'] if r['best_L_search'] is not None else np.inf))
    return [r['layout'] for r in order[:2]],rows


def adaptive():
    plan=prepare();A=plan['adaptive'];root=OUT/'adaptive';root.mkdir(exist_ok=True)
    scores=[dict(r,scorable=r['scorable']=='True',L_search=_f(r['L_search'])) for r in load_rows('scores.csv') if r['source']=='v2']
    if not any(s['scorable'] for s in scores):
        rt.write(root/'nomination.json',dict(status='NO_SCORABLE_WORKING_POINT',reason='no screen run reached 16 analysable events in both noises; DE not started',scores=scores));return
    if (root/'population_setup.json').exists():setup=rt.read(root/'population_setup.json')
    else:
        active,evidence,ranked=select_active_parameters(plan);layouts,layout_rows=select_populations(plan,scores)
        if not active:
            rt.write(root/'nomination.json',dict(status='NO_RESPONSIVE_PARAMETER',reason='no parameter changed a key observable in the same direction in both noises',evidence=evidence));return
        bounds={k:plan['axes'][k]['adaptive_bounds'] for k in active};pops={}
        for layout in layouts:
            parents=[]
            for c in plan['candidates']:
                if c['layout']!=layout or (c['changed_parameter']!='baseline' and c['changed_parameter'] not in active):continue
                cl=condition_loss('screen',c['id'],plan)
                parents.append(dict(id=c['id'],vector={k:float(c['parameters'][k]) for k in active},loss=cl['loss'],units=cl['units'],origin='screen'))
            parents.sort(key=lambda p:(p['loss'] is None,p['loss'] if p['loss'] is not None else 0.))
            pops[layout]=parents[:A['parent_size']]
        setup=dict(active_parameters=active,bounds=bounds,evidence=evidence,ranking=ranked,layouts=layouts,layout_evidence=layout_rows,populations=pops,
                   F=A['F'],CR=A['CR'],rng_seed=A['rng_seed'],center_offsets='not used as active dimensions in this round',
                   rule='active = parameters with same-sign paired change above threshold in both noises (design priority pairs first); populations = two layouts with most scorable runs; parents = layout baseline + active-parameter probes ranked by mean L_search, unscorable last')
        rt.write(root/'population_setup.json',setup)
    active=setup['active_parameters'];bounds=setup['bounds'];rng=np.random.default_rng(int(setup['rng_seed']))
    pops={k:list(v) for k,v in setup['populations'].items()}
    for layout,pop in pops.items():
        if len(pop)<4:rt.write(root/'nomination.json',dict(status='INSUFFICIENT_PARENTS',layout=layout,n=len(pop)));return
    history=[]
    for batch in range(1,A['batches']+1):
        bpath=root/f'batch{batch}.json'
        if bpath.exists():record=rt.read(bpath)
        else:
            record=dict(batch=batch,offspring=[],rejections=[],parents_snapshot={k:[dict(id=p['id'],vector=p['vector'],loss=p['loss']) for p in v] for k,v in pops.items()})
            for layout,pop in pops.items():
                targets=[i for i in range(len(pop))][(batch-1)*A['offspring_per_population_per_batch']:batch*A['offspring_per_population_per_batch']]
                for o in de_offspring(pop,targets,bounds,rng,F=setup['F'],CR=setup['CR']):
                    cid=f'de_{layout}_b{batch}_t{o["target_index"]}';cand=candidate_from_vector(cid,layout,o['vector'],plan,'adaptive');tries=0
                    while not geometry_valid(cand['centers_mm'],cand['radii_mm']) and tries<20:
                        record['rejections'].append(dict(candidate=cid,vector=o['vector'],reason='invalid geometry (overlap or sheet clipping)'));tries+=1
                        o=de_offspring(pop,[o['target_index']],bounds,rng,F=setup['F'],CR=setup['CR'])[0];cand=candidate_from_vector(cid,layout,o['vector'],plan,'adaptive')
                    rt.write(OUT/'candidates'/f'{cid}.json',cand);record['offspring'].append(dict(layout=layout,candidate=cid,**o))
            record['rng_state_after']=str(rng.bit_generator.state['state']['state']);rt.write(bpath,record)
        units=[(o['candidate'],plan['topology_seed'],s) for o in record['offspring'] for s in plan['seeds']]
        failures=queue('adaptive',units,plan['duration_ms'],f'adaptive_b{batch}')
        if failures:rt.write(OUT/'status.json',dict(status='ADAPTIVE_ENGINEERING_FAILURE',failures=failures,updated_unix=time.time()));raise RuntimeError(str(failures))
        for o in record['offspring']:
            cl=condition_loss('adaptive',o['candidate'],plan);pop=pops[o['layout']];parent=pop[o['target_index']]
            child=dict(id=o['candidate'],vector=o['vector'],loss=cl['loss'],units=cl['units'],origin=f'de_batch{batch}')
            choice=de_select(parent,child);history.append(dict(batch=batch,layout=o['layout'],target=o['target_index'],parent=parent['id'],parent_loss=parent['loss'],child=child['id'],child_loss=child['loss'],**choice))
            if choice['id']==child['id']:pop[o['target_index']]=child
        rt.write(root/f'batch{batch}_selection.json',history);rt.write(root/'population_state.json',{k:[dict(id=p['id'],vector=p['vector'],loss=p['loss']) for p in v] for k,v in pops.items()})
        if batch<A['batches']:
            # deferred replacement: the second batch proposes from the updated parents only after the first batch completed
            setup['populations']=pops
    rt.write(root/'nomination.json',dict(status='ADAPTIVE_COMPLETE',selection_history=history,final_populations={k:[dict(id=p['id'],vector=p['vector'],loss=p['loss']) for p in v] for k,v in pops.items()},
        note='DE ranked by L_search only; propagation acceptance is a separate scientific review'))
    rt.write(OUT/'status.json',dict(status='ADAPTIVE_COMPLETE_ANALYSIS_PENDING',updated_unix=time.time()))


# ---------------------------------------------------------------- stage 3: confirmation on new identities (C14)
NEW_TOPOLOGIES=[2611,2612];NEW_DYNAMICS=[848101,848102]


def seeds_unused(values):
    """Semantic check: any JSON field whose key contains 'seed' holding one of the proposed values."""
    import json
    roots=[ROOT/'config',Path('/home/honglab/leijiaxin/HFOsp/config'),Path('/data/hfosp/topic4_sef_hfo')]
    values={int(v) for v in values};hits=[]
    def walk(obj,path,key=''):
        if isinstance(obj,dict):
            for k,v in obj.items():walk(v,path,str(k))
        elif isinstance(obj,list):
            for v in obj:walk(v,path,key)
        elif 'seed' in key.lower() and isinstance(obj,(int,float)) and not isinstance(obj,bool) and int(obj)==obj and int(obj) in values:
            hits.append(dict(seed=int(obj),key=key,path=str(path)))
    for root in roots:
        for p in root.rglob('*.json'):
            if p.stat().st_size>5_000_000 or 'core_connectivity_search_20260910' in str(p):continue
            try:walk(json.loads(p.read_text()),p)
            except Exception:continue
    return hits


def build_new_topology(seed,plan):
    from src.topic4_core_field_runner import get_network
    from params import Params
    parent=rt.read(PARENT);execution=rt.execution_config(parent);ref=execution['corrected_networks'][str(plan['topology_seed'])]['config']
    p=Params(g=ref['g'],L=ref['L'],density=ref['density'],dt=ref['dt'],seed=int(seed))
    cache=OUT/'confirmation/network_cache';cache.mkdir(parents=True,exist_ok=True)
    from src.topic4_core_field_runner import connectivity_config,cache_key
    cwd=os.getcwd();os.chdir(ROOT)
    try:
        net,NE,NI,hit=get_network(p,ref['theta_EE_deg'],ref['AR'],str(cache))
        path=cache/(cache_key(connectivity_config(p,ref['theta_EE_deg'],ref['AR']))+'.pkl')
    finally:os.chdir(cwd)
    if not path.exists():raise RuntimeError('network cache path mismatch')
    return dict(path=str(path),sha256=rt.sha(path),topology_seed=int(seed),status='BUILT_FOR_CONFIRMATION',cache_hit=hit,NE=int(NE),NI=int(NI),
                pathways_expected='same sampler/config as topology 2511 except seed')


def nominate_confirmation(plan):
    """Fixed rule (coded before any confirmation result): new in-situ baseline + best L_search + best SCL9/8 participation with both-noise agreement + best TA support."""
    scores=[dict(r,scorable=r['scorable']=='True',L_search=_f(r['L_search'])) for r in load_rows('scores.csv') if r['source']=='v2']
    obs=[r for r in load_rows('run_mode_observations.csv') if r['source']=='v2']
    extra=[]
    nom=rt.read(OUT/'adaptive/nomination.json') if (OUT/'adaptive/nomination.json').exists() else {}
    for layout,pop in nom.get('final_populations',{}).items():
        for p in pop:
            if p['loss'] is not None and p['id'].startswith('de_'):extra.append(dict(candidate=p['id'],layout=layout,L_search=p['loss'],scorable=True))
    by_cond={}
    for s in scores:
        by_cond.setdefault(s['candidate'],[]).append(s)
    conds=[dict(candidate=k,layout=v[0]['layout'],L_search=float(np.mean([x['L_search'] for x in v])) if all(x['scorable'] for x in v) else None) for k,v in by_cond.items()]+extra
    picks=['endpoint__baseline'];reasons={'endpoint__baseline':'new in-situ v2 baseline (design)'}
    best=sorted([c for c in conds if c['L_search'] is not None],key=lambda c:c['L_search'])
    for c in best:
        if c['candidate'] not in picks:picks.append(c['candidate']);reasons[c['candidate']]='lowest mean L_search';break
    def paired(key,mode,sign):
        table={}
        for o in obs:
            if o['mode']!=mode or _f(o[key]) is None:continue
            table.setdefault(o['candidate'],[]).append(sign*_f(o[key]))
        return sorted([(k,min(v)) for k,v in table.items() if len(v)==2],key=lambda kv:-kv[1])
    for (key,mode,sign,why) in [('SCL_upper_participation','ALL',1,'highest SCL9/8 participation (min over both noises)'),('n','TA',1,'largest TA support (min over both noises)')]:
        for cid,val in paired(key,mode,sign):
            if cid not in picks and not cid.endswith('legacy_input'):picks.append(cid);reasons[cid]=f'{why}: {val}';break
    return picks[:plan['confirmation']['conditions']],reasons


def confirmation():
    plan=prepare();root=OUT/'confirmation';root.mkdir(exist_ok=True)
    if (root/'seed_freeze.json').exists():freeze=rt.read(root/'seed_freeze.json')
    else:
        hits=seeds_unused(NEW_TOPOLOGIES+NEW_DYNAMICS)
        if hits:raise RuntimeError(f'proposed confirmation seeds already appear in existing plans: {hits}')
        freeze=dict(topology_seeds=NEW_TOPOLOGIES,dynamics_seeds=NEW_DYNAMICS,checked_roots=['config','/home/honglab/leijiaxin/HFOsp/config','/data/hfosp/topic4_sef_hfo'],status='FROZEN_UNUSED');rt.write(root/'seed_freeze.json',freeze)
    if (root/'frozen_networks.json').exists():nets=rt.read(root/'frozen_networks.json')
    else:
        nets=dict(replication_networks={str(s):build_new_topology(s,plan) for s in freeze['topology_seeds']});rt.write(root/'frozen_networks.json',nets)
    if (root/'nomination.json').exists():nom=rt.read(root/'nomination.json')
    else:
        picks,reasons=nominate_confirmation(plan);nom=dict(conditions=picks,reasons=reasons,duration_ms=plan['confirmation']['duration_ms'],rule='fixed before confirmation results');rt.write(root/'nomination.json',nom)
    units=[(cid,t,s) for cid in nom['conditions'] for t in freeze['topology_seeds'] for s in freeze['dynamics_seeds']]
    failures=queue('confirmation',units,plan['confirmation']['duration_ms'],'confirmation')
    if failures:rt.write(OUT/'status.json',dict(status='CONFIRMATION_ENGINEERING_FAILURE',failures=failures,updated_unix=time.time()));raise RuntimeError(str(failures))
    identity=[]
    for cid in nom['conditions']:
        for t in freeze['topology_seeds']:
            ids=[rt.read(result_path('confirmation',cid,t,s))['static_array_identity'] for s in freeze['dynamics_seeds']]
            identity.append(dict(candidate=cid,topology=t,static_arrays_identical_across_replays=ids[0]==ids[1]))
    rt.write(root/'identity_audit.json',identity)
    if not all(x['static_arrays_identical_across_replays'] for x in identity):raise RuntimeError('static arrays differ across dynamics replays')
    rt.write(OUT/'status.json',dict(status='CONFIRMATION_COMPLETE_ANALYSIS_PENDING',updated_unix=time.time()))



if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['adaptive','confirmation']);a=ap.parse_args()
    try:globals()[a.action]()
    except Exception as exc:rt.write(OUT/'status.json',dict(status='FAILED',action=a.action,error=repr(exc),updated_unix=time.time()));raise
