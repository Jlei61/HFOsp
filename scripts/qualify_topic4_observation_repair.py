#!/usr/bin/env python3
"""Calibrate a fixed observer, test its invariants, then re-observe saved trials."""
from pathlib import Path
import json,pickle,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_topic4_xy_research import read,write,sha,training_contract
from src.topic4_observation_repaired import calibrate_reference,observe
from src.topic4_interictal_repaired_evaluation import RepairedEvaluator
from src.topic4_joint_xy import observable_groups
OLD=ROOT/'results/topic4_sef_hfo/multidimensional_interictal_pilot_round1'
OUT=ROOT/'results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2'


def same_early(left,right,stop_ms=5500.):
    il=[i for i,w in enumerate(left['windows_ms']) if w[1]<stop_ms]
    ir=[i for i,w in enumerate(right['windows_ms']) if w[1]<stop_ms]
    return ([left['windows_ms'][i] for i in il]==[right['windows_ms'][i] for i in ir]
            and [left['events'][i]['primary_eligible'] for i in il]==[right['events'][i]['primary_eligible'] for i in ir]
            and np.array_equal(left['centroid_ms'][il],right['centroid_ms'][ir],equal_nan=True)
            and np.array_equal(left['recruitment_ms'][il],right['recruitment_ms'][ir],equal_nan=True))


def evaluator():
    frozen=read(OLD/'evaluation_manifest.json')
    if sha(OLD/'evaluator.pkl')!=frozen['evaluator_sha256']:raise RuntimeError('patient evaluator changed')
    with open(OLD/'evaluator.pkl','rb') as f:previous=pickle.load(f)
    ev=RepairedEvaluator.__new__(RepairedEvaluator);ev.__dict__.update(previous.__dict__);ev.cache_probe=None
    return ev


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    p=ROOT/'results/topic4_sef_hfo/vth_dual_core_xy_research/selection/workers/control_historical_matched_seed_2515.json'
    d=read(p);arr=Path(d['arrays']['path'])
    if sha(arr)!=d['arrays']['sha256']:raise RuntimeError('reference arrays changed')
    with np.load(arr) as z:
        env=z['contact_envelope'];dt=float(z['contact_envelope_dt_ms']);names=z['contact_names'].astype(str).tolist()
        contract=calibrate_reference(env,dt,names)
        first=observe(env[:,:3000],dt,contract);full=observe(env,dt,contract)
        # Compare exact early windows away from the crop boundary.
        inv=same_early(first,full)
        alternate=env.copy();alternate[0,-10:]=env.max()*10
        suffix=observe(alternate,dt,contract)
        future=same_early(full,suffix)
        validation=[e for e in full['events'] if e['window_ms'][0]>=6250]
        first_counts=[e for e in full['events'] if e['window_ms'][1]<=5750]
        reference_audit={'calibration_events':len(first_counts),'validation_events':len(validation),
            'same_prefix_invariant':inv,'distant_future_peak_invariant':future,
            'minimum_unique_contacts':min([e['n_unique_contacts'] for e in full['events']],default=0)}
    contract['reference']={'worker_path':str(p),'worker_sha256':sha(p),'arrays_path':str(arr),'arrays_sha256':sha(arr),'seed':2515,
        'reference_role':'historical matched baseline on a topology/dynamics seed outside this four-seed parameter round; not untouched patient validation'}
    path=OUT/'observation_contract.json'
    if path.exists() and read(path)!=contract:raise RuntimeError('frozen observation contract changed')
    if not path.exists():write(path,contract)
    ev=evaluator();probe=ev.patient[ev.index['PROBE']];units=ev.blocks[ev.index['PROBE']]
    labels,_,_=ev.classify(probe)
    baseline=ev.metrics(probe,units)
    deleted=ev.metrics(probe[labels==0],units[labels==0])
    first=np.nanmin(probe,axis=1)[:,None]
    stretched=ev.metrics(first+1.5*(probe-first),units)
    shuffled=probe.copy();rng=np.random.default_rng(2026090709)
    for row in shuffled:
        idx=np.flatnonzero(np.isfinite(row));row[idx]=rng.permutation(row[idx])
    reordered=ev.metrics(shuffled,units)
    noisey=ev.metrics(probe+rng.normal(0,.5,probe.shape),units)
    collapsed=probe.copy()
    for m in range(ev.k):
        ix=np.flatnonzero(labels==m);collapsed[ix]=probe[ix[0]]
    collapse=ev.metrics(collapsed,units)
    single=ev.metrics(probe[:50],np.zeros(50))
    empty=ev.metrics(np.empty((0,15)))
    checks={'reference_prefix_invariant':inv,'reference_future_peak_invariant':future,
        'reference_validation_events_present':len(validation)>0,
        'all_reference_events_have_required_contacts':reference_audit['minimum_unique_contacts']>=8,
        'patient_mode_deletion_detected':deleted['mode_presence_fraction']<baseline['mode_presence_fraction'],
        'time_stretch_detected':stretched['kernel_distances']['timing_space']>baseline['kernel_distances']['timing_space'],
        'order_shuffle_detected':reordered['kernel_distances']['rank_space']>baseline['kernel_distances']['rank_space'],
        'reasonable_noise_support_change_under_5pct':abs(noisey['supported_fraction']-baseline['supported_fraction'])<.05,
        'collapsed_rank_variation_detected':all(m.get('rank_variation',1)<1e-12 for m in collapse['modes']),
        'single_network_presence_not_estimable':single['mode_presence_fraction'] is None,
        'empty_not_estimable':empty['joint_distance'] is None}
    with open(OUT/'evaluator.pkl','wb') as f:pickle.dump(ev,f,protocol=pickle.HIGHEST_PROTOCOL)
    result={'status':'IMPLEMENTATION_AND_DEVELOPMENT_CONTROLS_PASS' if all(checks.values()) else 'REPAIR_CONTROLS_FAILED',
        'checks':checks,'reference_audit':reference_audit,'patient_self':baseline,
        'patient_controls':{'deleted':deleted,'stretched':stretched,'reordered':reordered,'noise':noisey,'collapse':collapse},
        'observer_sha256':sha(path),'evaluator_sha256':sha(OUT/'evaluator.pkl'),
        'patient_reference_reused_unchanged':True,'full_model_acceptance_qualified':False,
        'remaining':'population-level patient recruitment timestamps unavailable in the frozen centroid table; no centroid-versus-recruitment fit claim; final scientific acceptance remains separate'}
    write(OUT/'qualification.json',result)
    print('qualification',checks,flush=True)
    return contract,ev,result


def legacy_observation_or_censor(env,dt,settings,runaway_stop_ms):
    """A known early runaway has no interictal observation, not a zero loss."""
    if env[:,int(round(settings['burnin_ms']/dt)):].size==0:
        if runaway_stop_ms is None:
            raise ValueError('unexpected short trajectory without recorded runaway')
        return np.empty((0,len(env))),{'status':'NOT_ESTIMABLE_RUNAWAY_BEFORE_BURNIN','n_groups':0}
    return observable_groups(env,dt,**settings)


def rescore(contract,ev):
    oldobs=read(ROOT/'config/topic4_joint_xy_kernel_v4.json')['observation']
    candidates={};records=[];invariants=[]
    for p in sorted((OLD/'execution/paired_round1/workers').glob('*.json')):
        d=read(p);npz=Path(d['arrays']['path'])
        if sha(npz)!=d['arrays']['sha256']:raise RuntimeError('saved trial changed')
        with np.load(npz) as z:
            if z['contact_names'].astype(str).tolist()!=contract['contact_names']:raise RuntimeError('contact identity changed')
            env=z['contact_envelope'];dt=float(z['contact_envelope_dt_ms'])
            runaway_stop=d['simulation']['runaway_early_stop_ms']
            old,oldmeta=legacy_observation_or_censor(env,dt,oldobs,runaway_stop)
            new=observe(env,dt,contract)
            variants={str(scale):observe(env,dt,contract,threshold_scale=scale) for scale in (.75,1.,1.25)}
            nframe=int(6000/dt)
            same=same_early(observe(env[:,:nframe],dt,contract),new)
            altered=env.copy();altered[0,-10:]=max(float(env.max()),1.)*10
            future=same_early(new,observe(altered,dt,contract))
            short=oldmeta.get('status')=='NOT_ESTIMABLE_RUNAWAY_BEFORE_BURNIN'
            invariants.append({'worker':str(p),'same_prefix':None if short else same,
                'future_peak_invariant':None if short else future,
                'invariant_estimability':'NOT_ESTIMABLE_NO_POST_BURNIN' if short else 'CHECKED',
                'minimum_unique_contacts':min([x['n_unique_contacts'] for x in new['events']],default=8)})
            row={'candidate_id':d['candidate_id'],'seed':d['seed'],'worker_path':str(p),'worker_sha256':sha(p),
                'arrays_sha256':sha(npz),'n_old':len(old),'n_new':new['n_groups'],'n_primary':new['n_primary_events'],
                'observation':new,'centroid_estimable':int(np.isfinite(new['centroid_ms']).any(1).sum()),
                'observation_status':'NOT_ESTIMABLE_RUNAWAY_BEFORE_BURNIN' if short else 'POST_BURNIN_AVAILABLE',
                'observed_duration_ms':env.shape[1]*dt,'runaway_stop_ms':runaway_stop,
                'runaway':d['simulation']['runaway_early_stop_ms'] is not None}
            records.append(row)
            candidates.setdefault(d['candidate_id'],[]).append((d['seed'],old,variants))
    summaries=[]
    for cid,units in candidates.items():
        item={'candidate_id':cid,'seeds':[u[0] for u in units],'complete_four_networks':len(units)==4,'metrics':{}}
        rr=[r for r in records if r['candidate_id']==cid]
        item['simulation_audit']={'runaway_networks':sum(r['runaway'] for r in rr),
            'no_post_burnin_networks':sum(r['observation_status']=='NOT_ESTIMABLE_RUNAWAY_BEFORE_BURNIN' for r in rr),
            'role':'No post-burnin data means interictal fit not estimable; runaway counts remain visible.'}
        for name in ['old','.75','1.0','1.25','recruitment_auxiliary']:
            # str(.75) is '0.75'. Keep display keys explicit below.
            scale='0.75' if name=='.75' else name
            tables=[]
            for u in units:
                if name=='old': tables.append(u[1]);continue
                obs=u[2]['1.0' if name=='recruitment_auxiliary' else scale]
                tables.append(obs['recruitment_ms' if name=='recruitment_auxiliary' else 'centroid_ms'][obs['primary_event_indices']])
            seedids=np.concatenate([np.full(len(t),u[0]) for u,t in zip(units,tables)])
            if name=='recruitment_auxiliary':
                item['recruitment_auxiliary']={'n_events':sum(len(t) for t in tables),'patient_centroid_comparison_performed':False};continue
            metric=ev.metrics(np.concatenate(tables),seedids)
            item['metrics'][name]=metric
        all_tables=[u[2]['1.0']['centroid_ms'] for u in units]
        all_seeds=np.concatenate([np.full(len(t),u[0]) for u,t in zip(units,all_tables)])
        item['all_detected_windows_auxiliary']=ev.metrics(np.concatenate(all_tables),all_seeds)
        ee=[e for u in units for e in u[2]['1.0']['events']]
        item['selection_audit']={'n_detected':len(ee),'n_primary':sum(e['primary_eligible'] for e in ee),
            'n_overlapping':sum(e['overlap_with_other_windows'] for e in ee),
            'n_prolonged':sum(e['prolonged'] for e in ee),
            'interpretation':'primary scores describe isolated events only; all-window scores and excluded counts must accompany model assessment'}
        summaries.append(item);print('rescored',cid,len(units),flush=True)
    write(OUT/'reobserved_workers.json',{'workers':records,'invariants':invariants,
          'n_invariant_not_estimable':sum(x['same_prefix'] is None for x in invariants),
          'all_prefix_checks_pass':all(x['same_prefix'] is not False for x in invariants),
          'all_future_peak_checks_pass':all(x['future_peak_invariant'] is not False for x in invariants)})
    write(OUT/'before_after_parameter_comparison.json',{'candidates':summaries,'new_results_overwrite_old':False,
         'threshold_sensitivity_prespecified':[.75,1.,1.25],'primary_threshold_scale':1.,'model_selected_observer':False})
    return summaries


if __name__=='__main__':
    if '--rescore-only' in sys.argv:
        contract=read(OUT/'observation_contract.json');q=read(OUT/'qualification.json')
        if sha(OUT/'evaluator.pkl')!=q['evaluator_sha256'] or sha(OUT/'observation_contract.json')!=q['observer_sha256']:
            raise RuntimeError('frozen reference changed')
        with open(OUT/'evaluator.pkl','rb') as f:ev=pickle.load(f)
    else:contract,ev,q=prepare()
    if not all(q['checks'].values()):raise RuntimeError('repair checks failed; do not resume')
    rescore(contract,ev)
