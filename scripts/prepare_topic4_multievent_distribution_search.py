#!/usr/bin/env python3
"""Prepare a bounded design and qualify its loss; does not dispatch simulations."""
from pathlib import Path
import copy, itertools, json, pickle, secrets, sys
import numpy as np
from scipy.stats import qmc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.topic4_multievent_distribution_objective import MultieventDistributionObjective, matched_mean_distance
from src.topic4_xy_search import field_descriptor, geometry_allowed, audit_geometry
from scripts import run_topic4_xy_research as base
read,write,sha=base.read,base.write,base.sha
CONFIG=ROOT/'config/topic4_multievent_distribution_search_v2.json'


def qualification(obj,ev):
    rng=np.random.default_rng(20260907031);m=obj.matched_count
    # Freeze scales using CAL alone, before examining controls or any SNN fit.
    samples=[];calibration_blocks=[]
    for _ in range(128):
        pieces=[];chosen=[]
        for block in rng.permutation(ev.partition['CAL']):
            pieces.append(ev.patient[ev.blocks==block]);chosen.append(int(block))
            if sum(len(x) for x in pieces)>=m:break
        p=np.concatenate(pieces)
        if len(p)<m:raise RuntimeError('entire CAL partition too small')
        samples.append(obj.components(p[rng.choice(len(p),m,replace=False)]))
        calibration_blocks.append(chosen)
    obj.normalizers={k:float(np.median([r[k] for r in samples])) for k in ['global','balanced_modes']}
    if any(v<=0 for v in obj.normalizers.values()):raise RuntimeError('degenerate patient normalization')
    # Algebra/invariance checks do not depend on patient routes or a winning model.
    toy=rng.normal(size=(6,4));target=rng.normal(size=4)
    enumerated=np.mean([np.sum((toy[list(ids)].mean(0)-target)**2) for ids in itertools.combinations(range(6),3)])
    checks={'matched_expectation_equals_enumeration':bool(np.isclose(enumerated,matched_mean_distance(toy,target,3),atol=1e-12)),
            'insufficient_events_not_zero':matched_mean_distance(toy,target,7) is None}
    cal=ev.cal;labels=ev.cal_labels;effects=[]
    for _ in range(32):
        # Exactly the same pooled events: one single-mode network for mode 0,
        # two for mode 1, versus random redistribution among the same 3 networks.
        # K=2 here is the inherited patient result, not a new hard-coded discovery.
        if obj.k!=2:raise RuntimeError('redesign segregation control for inherited K')
        a=cal[rng.choice(np.flatnonzero(labels==0),m,replace=False)]
        b=cal[rng.choice(np.flatnonzero(labels==1),2*m,replace=False)]
        pooled=np.concatenate([a,b]);perm=rng.permutation(len(pooled))
        mixed={i:pooled[ix] for i,ix in enumerate(np.array_split(perm,3))}
        segregated={0:a,1:b[:m],2:b[m:]}
        normal=obj.score_candidate(mixed)['loss'];separate=obj.score_candidate(segregated)['loss']
        shuffled=pooled.copy()
        for row in shuffled:
            ii=np.flatnonzero(np.isfinite(row));row[ii]=rng.permutation(row[ii])
        first=np.nanmin(pooled,axis=1)[:,None]
        effects.append({'mixed':normal,'seed_segregated':separate,
            'pooled_original':obj.score_network(pooled)['loss'],
            'pooled_permuted':obj.score_network(pooled[perm])['loss'],
            'order_shuffled':obj.score_candidate({i:shuffled[ix] for i,ix in enumerate(np.array_split(perm,3))})['loss'],
            'time_stretched':obj.score_candidate({i:(first+1.5*(pooled-first))[ix] for i,ix in enumerate(np.array_split(perm,3))})['loss']})
    checks['identical_pooled_distribution_identical_score']=all(np.isclose(r['pooled_original'],r['pooled_permuted'],atol=1e-12) for r in effects)
    checks['same_network_mode_gap_detected']=bool(np.median([r['seed_segregated']-r['mixed'] for r in effects])>0)
    checks['patient_order_shuffle_detected']=bool(np.median([r['order_shuffled']-r['mixed'] for r in effects])>0)
    checks['patient_time_stretch_detected']=bool(np.median([r['time_stretched']-r['mixed'] for r in effects])>0)
    t=cal[:m];checks['absolute_time_shift_invariant']=bool(np.isclose(obj.score_network(t)['loss'],obj.score_network(t+1000)['loss'],atol=1e-9))
    a=obj.score_candidate({0:t,1:cal[:m-1]});checks['insufficient_seed_cannot_be_hidden']=a['loss'] is None
    return {'status':'LOSS_IMPLEMENTATION_AND_PATIENT_CONTROLS_PASS' if all(checks.values()) else 'LOSS_QUALIFICATION_FAILED',
        'checks':checks,'patient_only_normalizers':obj.normalizers,'controls':effects,
        'calibration_block_draws':calibration_blocks,
        'calibration_sampling':'random permutation of CAL blocks; retain whole blocks until >=16 events; draw 16 without replacement; sparse blocks are not dropped',
        'median_control_losses':{k:float(np.median([r[k] for r in effects])) for k in effects[0]},
        'no_SNN_scores_or_route_metrics_used':True,'physical_runner_qualified':False,'clinical_equivalence_qualified':False}


def proposals(cfg,out):
    path=out/'initial_candidate_manifest.json'
    if path.exists():
        d=read(path)
        if d['config_sha256']!=sha(CONFIG):raise RuntimeError('design changed; use another version')
        return d
    master=secrets.randbits(32);pos=base.positions();keys=list(cfg['parameters']);bounds=np.asarray(list(cfg['parameters'].values()))
    old=read(ROOT/'results/topic4_sef_hfo/multidimensional_interictal_pilot_round1/design.json')
    anchors=['historical__baseline','support_rank__baseline','old_joint__baseline','support_rank__vth_low','old_joint__tau_d_GABA_ms_high','historical__tau_d_GABA_ms_high']
    by={r['candidate_id']:r for r in old['candidates']};rows=[]
    for i,cid in enumerate(anchors):
        row=copy.deepcopy(by[cid]);row['candidate_id']='v2_anchor_'+cid;row['population']=i%2;row['reference_candidate_id']=cid;row['proposal']='historical_comparator_not_a_route_constraint';row['domain']='whole_sheet';rows.append(row)
    for population in range(2):
        sampler=qmc.Sobol(d=len(keys),scramble=True,seed=(master+population)%2**32)
        accepted=0
        for index,u in enumerate(sampler.random_base2(12)):
            v=bounds[:,0]+u*(bounds[:,1]-bounds[:,0]);p=dict(zip(keys,v));centers=v[:4].reshape(2,2)
            if not geometry_allowed(centers,pos,domain='whole_sheet'):continue
            r=base.decorate({'candidate_id':f'v2_pop{population}_sobol_{accepted:03d}','population':population,
                'proposal':'fresh_scrambled_sobol_11d','sobol_draw_index':index,'node_field':field_descriptor(centers),
                'geometry':audit_geometry(pos,centers,1499),'domain':'whole_sheet'})
            r['dynamic_parameters']={k:float(p[k]) for k in ['E_to_E_weight_scale','E_to_I_weight_scale','I_to_E_weight_scale','tau_d_GABA_ms']};r['dynamic_parameters']['I_to_I_weight_scale']=1.
            r['node_mapping']['node_gain']=float(p['node_gain']);r['mechanisms']['ellipse_angle_deg']=float(base.THETA+p['EE_angle_offset_deg']);r['mechanisms']['ellipse_aspect_ratio']=float(p['EE_weight_aspect_ratio'])
            rows.append(r);accepted+=1
            if accepted==21:break
        if accepted!=21:raise RuntimeError('geometry proposal pool exhausted')
    d={'config_sha256':sha(CONFIG),'master_seed':master,'n_dimensions':len(keys),'parameter_order':keys,'candidates':rows,
        'n_initial_conditions':len(rows),'frozen_before_SNN_evaluation':True,'route_or_axis_preference':False}
    assert len(rows)==48 and all(sum(r['population']==p for r in rows)==24 for p in range(2))
    write(path,d);return d


def main():
    cfg=read(CONFIG);out=ROOT/cfg['output_root'];out.mkdir(parents=True,exist_ok=True)
    ref=ROOT/cfg['frozen_reference'];record=read(ref/'qualification.json')
    if sha(ref/'evaluator.pkl')!=record['evaluator_sha256']:raise RuntimeError('patient evaluator changed')
    ev=pickle.load(open(ref/'evaluator.pkl','rb'));obj=MultieventDistributionObjective(ev,matched_count=cfg['training']['matched_events_per_network'])
    q=qualification(obj,ev);write(out/'loss_qualification.json',q)
    if not all(q['checks'].values()):raise RuntimeError('patient-only qualification failed; no dispatch')
    with open(out/'training_objective.pkl','wb') as f:pickle.dump(obj,f,protocol=pickle.HIGHEST_PROTOCOL)
    # The optimization payload contains no patient PROBE/CAL event tables or route metrics.
    assert not any(k in vars(obj) for k in ['patient','fit','cal','index','partition','cache_probe'])
    design=proposals(cfg,out)
    write(out/'preparation.json',{'status':'DESIGN_AND_OFFLINE_LOSS_QUALIFIED_PHYSICAL_CANARY_PENDING',
        'config_sha256':sha(CONFIG),'training_objective_sha256':sha(out/'training_objective.pkl'),
        'candidate_manifest_sha256':sha(out/'initial_candidate_manifest.json'),'initial_conditions':design['n_initial_conditions'],
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/topic4_multievent_distribution_objective.py']},
        'simulations_dispatched':0,'full_search_runner_implemented':False,'validation_routes_in_loss':False})
    print(json.dumps({'status':'PREPARED_NO_SIMULATION_DISPATCH','initial_conditions':48,'dimensions':11,'qualification':q['checks'],'control_medians':q['median_control_losses']},indent=2))

if __name__=='__main__':main()
