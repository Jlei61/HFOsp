#!/usr/bin/env python3
"""Prepare and run a bounded, resumable local physical search with a new loss.

Three fixed placements; 2 proposals/anchor in each of two adaptive batches.
Two topology units per condition, then at most two nominees on noise replay:
24 + at most 4 new 24-second simulations. Existing anchors are reused.
"""
from pathlib import Path
import argparse,copy,csv,fcntl,json,os,pickle,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_topic4_xy_research as base
from scripts import run_topic4_multievent_distribution_v2_1 as engine
from scripts.audit_topic4_multievent_execution_parameters_v2_1 import audit as audit_execution
from src.topic4_envelope_joint_pilot import TemporalDistributionObjective,aligned_packet_event,model_events,combined_score
from src.topic4_interictal_repaired_evaluation import rank_features

OLD=ROOT/'results/topic4_sef_hfo/multievent_distribution_search_v2_1'
OUT=ROOT/'results/topic4_sef_hfo/contact_timing_shape_pilot'
TRAIN_PAIRS=[(6101,7101),(6102,7101)]
REPLAY_PAIRS=[(6101,7102),(6102,7102)]
PARAMETERS=['node_gain','E_to_E_weight_scale','E_to_I_weight_scale','I_to_E_weight_scale','tau_d_GABA_ms']
STEP=np.array([.12,.12,.15,.12,6.])
LOW=np.array([.5,.8,.8,.8,12.]);HIGH=np.array([1.4,1.2,1.3,1.2,28.])

def write_csv(path,rows):
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def vector(c):
    return np.array([c['node_mapping']['node_gain']]+[c['dynamic_parameters'][k] for k in PARAMETERS[1:]],float)

def with_vector(c,v,cid,parent,phase):
    c=copy.deepcopy(c);c['candidate_id']=cid;c['node_mapping']['node_gain']=float(v[0])
    for k,x in zip(PARAMETERS[1:],v[1:]):c['dynamic_parameters'][k]=float(x)
    c.update(proposal='paired_random_local_proposal_fixed_centers',arm=phase,parent_candidate_id=parent,
             pilot_parameter_names=PARAMETERS,pilot_parameter_values=v.tolist())
    return c

def get_frozen():
    manifest=base.read(OUT/'design.json')
    for path,digest in manifest['frozen_files'].items():
        if base.sha(path)!=digest:raise RuntimeError(f'pilot frozen input changed: {path}')
    with (OUT/'objective.pkl').open('rb') as f:temporal=pickle.load(f)
    with (OLD/'training_objective_v2_1.pkl').open('rb') as f:old=pickle.load(f)
    return manifest,temporal,old

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'design.json').exists():return get_frozen()
    gs=base.read(OLD/'g3_scores.json');selected=sorted(gs['candidates'],key=lambda c:c['score']['loss_off'])[:3]
    entries={c['candidate_id']:c for c in base.read(OLD/'execution/confirmation_24s/candidate_manifest.json')['candidates']}
    anchors=[entries[c['candidate_id']] for c in selected]
    with (OLD/'training_objective_v2_1.pkl').open('rb') as f:old=pickle.load(f)
    sample=Path(next(iter(selected[0]['units'].values()))['worker_path'])
    with np.load(sample.with_suffix('.npz')) as z:names=z['contact_names'].astype(str).tolist()
    packet=base.read(OLD/'patient_time_packet/packet_manifest.json')['readable_events']
    blocks=sorted({e['block_id'] for e in packet});review_blocks=set(blocks[-8:])
    training=[e for e in packet if e['block_id'] not in review_blocks]
    review=[e for e in packet if e['block_id'] in review_blocks]
    dd=[aligned_packet_event(e['arrays_path'],names) for e in training]
    values=np.array([d['values'] for d in dd]);labels=np.array([e['mode'] for e in training])
    temporal=TemporalDistributionObjective(values,labels,old.proportions,names)
    with (OUT/'objective.pkl').open('wb') as f:pickle.dump(temporal,f)
    np.savez_compressed(OUT/'patient_training_descriptors.npz',values=values,labels=labels,names=names,
                        weights=temporal.patient_weights,event_ids=[e['raw_global_event_index'] for e in training])
    write_csv(OUT/'patient_training_event_observables.csv',[dict(event_id=e['raw_global_event_index'],mode=e['mode'],block_id=e['block_id'],**d['statistics']) for e,d in zip(training,dd)])
    # Verify population objectives at known targets, before any new physics.
    checks=[]
    for q in sorted(set([0.,.1,.25,.5,.75,.9,1.,float(old.proportions[1])])):
        w=np.where(labels==1,q/np.sum(labels==1),(1-q)/np.sum(labels==0))
        checks.append(dict(experiment='mode_frequency',value=q,distance=temporal.population_distance(values,w)))
    for factor in [.2,.5,.75,1.,1.25,1.5,2.]:
        v=values.copy();v[:,:,2:]*=factor
        checks.append(dict(experiment='local_duration_multiplier',value=factor,distance=temporal.population_distance(v,temporal.patient_weights)))
        v=values.copy()
        for lab in [0,1]:
            ii=np.flatnonzero(labels==lab);active=v[ii,:,0].astype(bool)
            mm=np.divide((v[ii,:,1]*active).sum(0),active.sum(0),out=np.zeros(len(names)),where=active.sum(0)>0)
            v[ii,:,1]=np.where(active,mm[None]+factor*(v[ii,:,1]-mm[None]),0)
            for i in ii:v[i,v[i,:,0]>0,1]-=np.median(v[i,v[i,:,0]>0,1])
        checks.append(dict(experiment='within_mode_timing_scatter',value=factor,distance=temporal.population_distance(v,temporal.patient_weights)))
    for name,target in [('mode_frequency',float(old.proportions[1])),('local_duration_multiplier',1.),('within_mode_timing_scatter',1.)]:
        subset=[x for x in checks if x['experiment']==name]
        assert np.isclose(min(subset,key=lambda x:x['distance'])['value'],target)
    rng=np.random.default_rng(820803);u=[]
    for _ in range(256):
        ii=rng.choice(len(values),16,replace=True,p=temporal.patient_weights)
        u.append(temporal.score(values[ii])['D_off'])
    assert abs(np.mean(u))<5*np.std(u)/np.sqrt(len(u))+1e-9
    write_csv(OUT/'objective_population_checks.csv',checks)
    base.write(OUT/'objective_qualification.json',dict(status='PASS',mode_frequency_and_scatter_targets_minimized=True,
        same_distribution_U_mean=float(np.mean(u)),same_distribution_U_standard_error=float(np.std(u)/np.sqrt(len(u))),
        negative_values_allowed=True,normalizer=temporal.normalizer,feature_scales=temporal.scales.tolist(),
        kernel_bandwidth=temporal.bandwidth,interpretation='population identity and IID synthetic check, not an independence claim for continuous SNN events'))
    sources=[Path(__file__),ROOT/'src/topic4_envelope_joint_pilot.py',OLD/'training_objective_v2_1.pkl',
             OLD/'patient_time_packet/packet_manifest.json',OLD/'g3_scores.json',OUT/'objective.pkl',OUT/'patient_training_descriptors.npz']
    sources += [Path(e['arrays_path']) for e in training]
    design=dict(version='contact_timing_shape_pilot_v1',status='FROZEN_BEFORE_NEW_SIMULATION',
        objective='0.5 * original_Loff + 0.5 * contact_timing_shape_Doff / frozen_positive_scale',
        new_features=['participation','contact t50 minus event median t50','t50-t10','t90-t50','envelope centroid-t50'],
        time_features='contact identity retained; 10/50/90 are within-window cumulative mass quantiles, not ignition times',
        kernel='equal mixture of Gaussian kernels at 0.5,1,2 times TRAIN median pair distance; exact kernel, no RFF approximation',
        patient_training=training,patient_review=review,review_blocks=sorted(review_blocks),
        patient_reference_role='previously viewed development packet; held-back blocks are a diagnostic, not pristine validation',
        patient_mode_weights=old.proportions.tolist(),model_mode_reweighting=False,
        original_objective_mode_component_retained=True,manual_origin_route_constraints=False,
        anchors=anchors,anchor_scores=selected,parameter_names=PARAMETERS,local_half_width=STEP.tolist(),
        global_lower_bounds=LOW.tolist(),global_upper_bounds=HIGH.tolist(),centers_and_EE_geometry_fixed=True,
        train_seed_pairs=TRAIN_PAIRS,replay_seed_pairs=REPLAY_PAIRS,
        budget=dict(new_training_conditions=12,new_training_simulations=24,maximum_new_replay_simulations=4,duration_ms=24000,maximum_workers=8),
        adaptation='two antithetic random proposals per anchor in A; two around lowest NEW joint loss parent per anchor in B; delayed batch updates',
        ranking='equal mean of both training units; N>=16 per unit; runaway not ranked; negative Doff retained',
        physical_readout='unchanged 2ms bins / 5ms smoothing / 250ms windows / 500ms burn-in; no offline broadening or time rescaling',
        dose_interpretation='GABA decay at fixed jump; not dose-matched',
        stop='after bounded replay, plots and scientific review; no model freeze, further search, or Fig5',
        frozen_files={str(p):base.sha(p) for p in sources})
    base.write(OUT/'design.json',design)
    base.write(OUT/'status.json',dict(status='PREPARED',updated_unix=time.time(),new_physical_runs_started=0))
    return design,temporal,old

def score_record(path,temporal,old,candidate_id,unit,phase):
    model=model_events(path,engine.repaired_observation)
    if model['names']!=temporal.names:raise RuntimeError('model contact order changed')
    score=combined_score(old,temporal,model)
    labels=old.km.predict(rank_features(model['centroids'])) if len(model['centroids']) else []
    events=[dict(candidate_id=candidate_id,unit=unit,phase=phase,mode=int(lab),**info) for lab,info in zip(labels,model['info'])]
    return dict(worker_path=str(path),score=score),events

def score_candidates(candidates,phase,temporal,old,paths):
    output=[];all_events=[]
    for c in candidates:
        units={}
        for unit,path in paths[c['candidate_id']].items():
            units[unit],events=score_record(path,temporal,old,c['candidate_id'],unit,phase);all_events+=events
        good=all(u['score']['loss'] is not None for u in units.values())
        output.append(dict(candidate_id=c['candidate_id'],candidate=c,units=units,ranking_eligible=good,
                           loss=float(np.mean([u['score']['loss'] for u in units.values()])) if good else None))
    report=dict(phase=phase,candidates=output)
    base.write(OUT/f'{phase}_scores.json',report)
    if all_events:write_csv(OUT/f'{phase}_event_observables.csv',all_events)
    return report

def baseline_scores(design,temporal,old):
    path=OUT/'baseline_train_scores.json'
    if path.exists():return base.read(path)
    paths={c['candidate_id']:{u:d['worker_path'] for u,d in c['units'].items() if u.endswith('dyn_7101')} for c in design['anchor_scores']}
    return score_candidates(design['anchors'],'baseline_train',temporal,old,paths)

def propose(phase,design,parents):
    path=OUT/f'{phase}_proposals.json'
    if path.exists():return base.read(path)['candidates']
    proposals=[];draws=[]
    for ai,(anchor,parent) in enumerate(zip(design['anchors'],parents),1):
        rng=np.random.default_rng((820810 if phase=='A' else 820910)+ai)
        center=vector(parent);origin=vector(anchor)
        low=np.maximum(LOW,origin-STEP);high=np.minimum(HIGH,origin+STEP)
        direction=rng.choice([-1.,1.],5)*rng.uniform(.35,1.,5)
        scale=1. if phase=='A' else .65
        for sign,tag in [(1.,'plus'),(-1.,'minus')]:
            raw=center+sign*scale*STEP*direction
            # Reflection keeps proposals inside the original anchor box.
            width=high-low;v=low+width-np.abs((raw-low)%(2*width)-width)
            cid=f'tshape_anchor{ai}_{phase}_{tag}'
            proposals.append(with_vector(parent,v,cid,parent['candidate_id'],phase))
            draws.append(dict(candidate_id=cid,anchor_id=anchor['candidate_id'],parent_id=parent['candidate_id'],
                              direction=direction.tolist(),sign=sign,raw_values=raw.tolist(),applied_values=v.tolist(),bounds=[low.tolist(),high.tolist()]))
    base.write(path,dict(candidates=proposals,draws=draws,phase=phase,frozen_before_dispatch=True))
    return proposals

def execution(phase,candidates):
    folder=OUT/'execution'/phase;folder.mkdir(parents=True,exist_ok=True)
    cp=folder/'execution_config.json';mp=folder/'candidate_manifest.json';sp=folder/'runtime_snapshot.json'
    if not cp.exists():
        source=OLD/'execution/confirmation_24s'
        cfg=copy.deepcopy(base.read(source/'execution_config.json'))
        cfg.update(output_root=str(folder),candidate_manifest=str(mp))
        base.write(cp,cfg);base.write(mp,dict(config_sha256=base.sha(cp),candidates=candidates,phase=phase,frozen_before_simulation=True))
        hashes=base.read(source/'runtime_snapshot.json')['source_hashes']
        for p,h in hashes.items():
            if base.sha(ROOT/p)!=h:raise RuntimeError(f'physical dependency changed: {p}')
        base.write(sp,dict(source_hashes=hashes,input_hashes={str(cp):base.sha(cp),str(mp):base.sha(mp)},identity_kind='dependency_scoped_source_hash_snapshot',not_final_substrate_freeze=True))
    snap=base.read(sp)
    for p,h in snap['source_hashes'].items():
        if base.sha(ROOT/p)!=h:raise RuntimeError(f'physical dependency changed: {p}')
    for p,h in snap['input_hashes'].items():
        if base.sha(p)!=h:raise RuntimeError(f'phase input changed: {p}')
    if base.read(mp)['candidates']!=candidates:raise RuntimeError('persisted candidate proposals changed')
    return folder,cp,mp,sp

def run_phase(phase,candidates,pairs,temporal,old,workers):
    ex=execution(phase,candidates)
    jobs=[(c['candidate_id'],t,d) for c in candidates for t,d in pairs]
    engine.OUT=OUT
    engine._run_jobs(phase,jobs,maximum_workers=workers,execution=ex)
    get_frozen()
    audit=audit_execution(require_complete=True,execution=ex[0],seed_pairs=pairs,output_path=OUT/f'{phase}_parameter_application_audit.json')
    if audit['status']!='PARAMETER_APPLICATION_AUDIT_PASS':raise RuntimeError('actual applied parameter audit failed')
    paths={c['candidate_id']:{f'topo_{t}_dyn_{d}':str(ex[0]/'workers'/f'{c["candidate_id"]}_topo_{t}_dyn_{d}.json') for t,d in pairs} for c in candidates}
    return score_candidates(candidates,phase,temporal,old,paths)

def run(workers):
    design,temporal,old=prepare();baseline=baseline_scores(design,temporal,old)
    a=propose('A',design,design['anchors']);ar=run_phase('A',a,TRAIN_PAIRS,temporal,old,workers)
    parents=[]
    for ai,anchor in enumerate(design['anchors'],1):
        pool=[r for r in baseline['candidates']+ar['candidates'] if r['candidate_id']==anchor['candidate_id'] or r['candidate_id'].startswith(f'tshape_anchor{ai}_')]
        best=min([r for r in pool if r['ranking_eligible']],key=lambda r:(r['loss'],r['candidate_id']))
        parents.append(best['candidate'])
    b=propose('B',design,parents);br=run_phase('B',b,TRAIN_PAIRS,temporal,old,workers)
    pool=baseline['candidates']+ar['candidates']+br['candidates']
    nominees=sorted([r for r in pool if r['ranking_eligible']],key=lambda r:(r['loss'],r['candidate_id']))[:2]
    base.write(OUT/'nomination.json',dict(selection='new joint loss on both training units only',nominees=nominees,review_opened=False))
    anchor_ids={c['candidate_id'] for c in design['anchors']}
    new=[r['candidate'] for r in nominees if r['candidate_id'] not in anchor_ids]
    if new:run_phase('replay',new,REPLAY_PAIRS,temporal,old,workers)
    # Paired baseline replay is reused; never mix it into training selection.
    paths={c['candidate_id']:{u:d['worker_path'] for u,d in c['units'].items() if u.endswith('dyn_7102')} for c in design['anchor_scores']}
    score_candidates(design['anchors'],'baseline_replay',temporal,old,paths)
    base.write(OUT/'status.json',dict(status='PHYSICAL_PILOT_COMPLETE_ANALYZING',updated_unix=time.time()))
    subprocess.run([engine.PYTHON,str(ROOT/'scripts/analyze_topic4_contact_timing_shape_pilot.py'),'--final'],cwd=ROOT,env=engine.ENV,check=True)
    base.write(OUT/'status.json',dict(status='PILOT_COMPLETE_PENDING_SCIENTIFIC_REVIEW',updated_unix=time.time(),automatic_next_search=False,model_frozen=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    if not 1<=args.workers<=8:p.error('workers must be between 1 and 8')
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.prepare_only:
                design,temporal,old=prepare();baseline_scores(design,temporal,old)
                print('Prepared objective and cached anchor scores',flush=True)
            else:run(args.workers)
        except Exception as exc:
            base.write(OUT/'status.json',dict(status='ERROR_REVIEW_REQUIRED',error=repr(exc),updated_unix=time.time()))
            raise
