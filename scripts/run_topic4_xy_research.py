#!/usr/bin/env python3
"""Persistent XY-only search: global design, local refinement, selection, confirmation.

Selection is development-only on the training patient contract. No patient
held-out or ictal inputs are opened, and no winning substrate is auto-frozen.
"""
from pathlib import Path
import argparse
import copy
import fcntl
import json
import os
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1]
ART=Path('/home/honglab/leijiaxin/HFOsp')
for p in (ROOT,ROOT/'src/snn_engine'):sys.path.insert(0,str(p))
from src.topic4_xy_search import (audit_geometry,canonical_centers,field_descriptor,
    geometry_allowed,pareto_indices,sha,sobol_xy_candidates)
from src.topic4_core_field_runner import atomic_write_json
from scripts import aggregate_topic4_rev22_fit as FIT

OUT=ROOT/'results/topic4_sef_hfo/vth_dual_core_xy_research'
PYTHON='/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python'
ENV={**os.environ,**{k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
THETA=-22.80538396505847
COMPONENTS=('D_support','D_order','D_lag','D_cover')
SEEDS={'canary':[2511],'global':[2511,2512],'refinement':[2511,2512],
       'selection':[2513,2514,2515],'confirmation':[2731,2732,2733,2734,2735,2736]}
DURATIONS={'canary':2000.,'global':8000.,'refinement':8000.,'selection':12000.,'confirmation':20000.}


def write(path,payload):
    atomic_write_json(FIT._json_safe(payload),str(path))


def read(path):return json.loads(Path(path).read_text())


def source_hashes():
    return {str(p.relative_to(ROOT)):sha(p) for folder in ('src','scripts') for p in sorted((ROOT/folder).rglob('*.py'))}


def verify_sources(lock):
    for p,digest in lock.items():
        if sha(ROOT/p)!=digest:raise RuntimeError(f'locked runtime source changed: {p}')


def available_gib():
    return next(float(l.split()[1])/1024**2 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))


def network_record(seed):
    path=OUT/'network_records'/f'{seed}.json'
    if not path.exists():
        subprocess.run([PYTHON,str(ROOT/'scripts/prebuild_topic4_xy_network.py'),'--seed',str(seed)],cwd=ROOT,env=ENV,check=True)
    record=read(path)
    if record['status']!='CORRECTED_GRAPH_VALIDATED' or sha(record['path'])!=record['sha256']:
        raise RuntimeError('network record is stale')
    return record


def positions():
    record=network_record(2511)
    with open(record['path'],'rb') as f:payload=pickle.load(f)
    return np.asarray(payload['net']['pos'][:payload['NE']],float)


def decorate(row):
    return {**row,'node_mapping':{'node_gain':1.,'signed_depth_shrinkage':1.},
            'mechanisms':{'g_EE':0.,'g_EtoI':0.,'Z_M':'off',
                         'ellipse_angle_deg':THETA,'ellipse_aspect_ratio':2.,
                         'ellipse_reference_angle_deg':THETA,'ellipse_reference_aspect_ratio':2.},
            'selection_eligible':row.get('selection_eligible',True)}


def prepare_design():
    path=OUT/'search_design.json'
    if path.exists():return read(path)
    pos=positions()
    rows=sobol_xy_candidates(pos,n_per_domain=64,seed=20260905)
    old=[[1.5377342579886317,1.2264179880730808],[18.606612137053162,2.417713414411992]]
    historical=[[4.19921432,9.12890135],[16.47920304,3.96551153]]
    centered=[[3.8600056398,12.5816949095],[16.1399943602,7.4183050905]]
    for cid,centers,count,eligible in [('control_old_edge',old,1499,True),
        ('control_historical_matched',historical,1499,True),
        ('control_historical_native_budget',historical,1129,False),
        ('control_centered_reference_axis',centered,1499,True)]:
        rows.append({'candidate_id':cid,'domain':'control','proposal':'prespecified_control',
                     'node_field':field_descriptor(centers,count),'selection_eligible':eligible,
                     'geometry':audit_geometry(pos,centers,count)})
    design={'status':'XY_SEARCH_DESIGN_FIXED_SUBSTRATE_REOPENED','candidates':[decorate(r) for r in rows],
            'search_dimensions':['x1','y1','x2','y2'],'target_count':1499,'seed_pools':SEEDS,'duration_ms':DURATIONS,
            'center_bounds_mm':[.75,19.25],'minimum_center_separation_mm':4.,
            'maximum_center_separation':'Only the physical sheet limit; the old 18 mm cap is removed.',
            'boundary_domains':{'whole_sheet':'64 proposals, no disk clearance constraint',
                                'interior':'64 proposals, no alignment constraint; proposal clearance >=1.65 mm, every execution >=1.5 mm'},
            'objective':'Unconditional sliced Wasserstein in the accepted 49-feature patient-training embedding; all returned families, including unreadable families. Four decomposed distances and their Pareto set retained.',
            'shortlist_rule':'Best full-distribution candidates plus component-wise optima in each geometry domain; no KMeans/mode-existence-first ranking',
            'selection_eligibility':'Complete finite non-runaway trajectories, >=20 pooled returned families, all four conditional/unconditional components estimable. Low support is unresolved, not a biological failure.',
            'refinement_rule':'32 local proposals around four best distinct candidate geometries from the global stage; displacements at 0.75 and 2 mm scales; same paired fit seeds',
            'confirmation_rule':'Selected best whole-sheet and best interior development candidates plus old-edge and matched historical controls, on six new topology/dynamics seeds; no reselection or final substrate freeze from confirmation',
            'VTH_contract':'Frozen mean/std, quantile seed, gain and signed-depth mapping. Fixed count; realized selected-neuron depth variation is reported, not silently called identical.',
            'old_s39_status':'HISTORICAL_CONTROL_ONLY_NOT_ACCEPTED_FINAL_SUBSTRATE',
            'patient_holdout_opened':False,'ictal_opened':False,'final_substrate_frozen':False}
    write(path,design);return design


def phase_contract(phase,rows,lock):
    directory=OUT/phase;directory.mkdir(parents=True,exist_ok=True)
    cp=directory/'execution_config.json';mp=directory/'candidate_manifest.json';sp=directory/'runtime_snapshot.json'
    networks={str(seed):network_record(seed) for seed in SEEDS[phase]}
    base=read(ROOT/'config/topic4_rev22_dci_response_execution.json')
    cfg={k:copy.deepcopy(base[k]) for k in ('inputs','event_unit','source_topology','search','reference')}
    cfg.update({'scientific_role':'development_only_vth_dual_core_xy_research',
                'output_root':str(directory),'candidate_manifest':str(mp),'corrected_networks':networks,
                'network_cache':str(ROOT/'results/topic4_sef_hfo/substrate_autapse_correction/network_cache')})
    cfg['reference'].update(g_EE=0.,g_EtoI=0.,base_substrate_candidate_id='joint_04_control')
    cfg['search']={'fit_network_seeds':SEEDS[phase],
                   'simulation':{'duration_ms':DURATIONS[phase],'early_stop_runaway':True,'late_runaway_is_invalid':True},
                   'contact_readout':base['search']['contact_readout']}
    # Fit uses the identical primary event contract; ancillary segmentation
    # sensitivities cannot choose XY and remain available in confirmation.
    if phase!='confirmation':
        for name,primary in [('sensitivity_psp_tail_fractions','psp_tail_fraction'),
           ('sensitivity_minimum_dominances','minimum_dominance'),
           ('sensitivity_minimum_parent_supports','minimum_parent_support'),
           ('sensitivity_edge_delay_roundings','edge_delay_rounding')]:
            cfg['event_unit'][name]=[cfg['event_unit'][primary]]
    write(cp,cfg)
    write(mp,{'config_sha256':sha(cp),'candidates':rows,'phase':phase,'selection_frozen_before_simulation':True})
    write(sp,{'source_hashes':lock,'input_hashes':{str(p.resolve()):sha(p) for p in (cp,mp)},
              'identity_kind':'source_hash_snapshot','not_final_substrate_freeze':True})
    return cp,mp,sp


def run_phase(phase,rows,lock,max_workers):
    verify_sources(lock)
    write(OUT/'status.json',{'status':'PREPARING_PHASE','phase':phase,'updated_unix':time.time()})
    cp,mp,sp=phase_contract(phase,rows,lock)
    directory=OUT/phase;workers=directory/'workers';logs=directory/'run_logs'
    workers.mkdir(exist_ok=True);logs.mkdir(exist_ok=True)
    jobs=[(r['candidate_id'],seed) for r in rows for seed in SEEDS[phase]]
    pending=[];complete=[];active={}
    snapshot_sha=sha(sp)
    def done(cid,seed):
        jp=workers/f'{cid}_seed_{seed}.json';npz=jp.with_suffix('.npz')
        if not jp.exists() or not npz.exists():return False
        d=read(jp)
        return (d.get('status')=='REV12ND_NODE_WORKER_COMPLETE'
                and d['arrays']['sha256']==sha(npz)
                and d['provenance'].get('source_hash_snapshot',{}).get('sha256')==snapshot_sha)
    for job in jobs:
        (complete if done(*job) else pending).append(job)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    while pending or active:
        for job,(proc,stream) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            stream.close();del active[job]
            if code!=0 or not done(*job):
                # Let already started simulations complete naturally; stop dispatch.
                for other,handle in active.values():other.wait();handle.close()
                raise RuntimeError(f'{phase} worker failed: {job}, exit={code}')
            complete.append(job)
        if shutil.disk_usage(OUT).free<30*1024**3:raise RuntimeError('less than 30 GiB free disk')
        # Reserve 32 GiB; admission assumes up to 18 GiB per newly launched worker.
        outstanding_reserve=0.
        for proc,stream in active.values():
            try:
                rss_gib=int(Path(f'/proc/{proc.pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')/1024**3
            except (FileNotFoundError,ProcessLookupError):
                rss_gib=0.
            outstanding_reserve+=max(0.,18-rss_gib)
        slots=max(0,min(max_workers-len(active),int((available_gib()-32-outstanding_reserve)/18)))
        for unused in range(min(slots,len(pending))):
            cid,seed=pending.pop(0);stream=open(logs/f'{cid}_seed_{seed}.log','w')
            command=[PYTHON,str(ROOT/'scripts/run_topic4_rev12_node_worker.py'),
                '--config',str(cp),'--candidate-id',cid,'--seed',str(seed),
                '--expected-commit',commit,'--runtime-manifest',str(sp),'--artifact-root',str(ART),
                '--out-json',str(workers/f'{cid}_seed_{seed}.json'),'--out-npz',str(workers/f'{cid}_seed_{seed}.npz')]
            proc=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT)
            active[(cid,seed)]=(proc,stream)
        write(OUT/'status.json',{'status':'RUNNING','phase':phase,'total':len(jobs),
            'complete':len(complete),'running':len(active),'pending':len(pending),'updated_unix':time.time(),
            'active':[{'candidate_id':j[0],'seed':j[1],'pid':p.pid} for j,(p,f) in active.items()]})
        if active or pending:time.sleep(10)
    write(directory/'completion.json',{'status':'ALL_WORKERS_COMPLETE','jobs':len(jobs),'runtime_snapshot_sha256':snapshot_sha})


def training_contract():
    p=ART/'results/topic4_sef_hfo/data_driven_dual_core_interictal_identifiability/objective_qualification/objective_qualification.json'
    q=read(p);training=FIT._load_training_contract(p,q,q['patient_training_contract_sha256'])
    objective=FIT._load_training_objective()
    training['reference']=objective.patient_reference(training['onsets_ms'],training['groups'],training['pairs'],training['embedding'])
    return training,objective


def aggregate(phase,rows):
    training,objective=training_contract();results=[]
    for row in rows:
        tables=[];units=[]
        for seed in SEEDS[phase]:
            p=OUT/phase/'workers'/f"{row['candidate_id']}_seed_{seed}.json";d=read(p)
            if sha(d['arrays']['path'])!=d['arrays']['sha256']:raise RuntimeError('worker arrays changed')
            with np.load(d['arrays']['path']) as a:
                if list(a['contact_names'].astype(str))!=training['contact_names']:raise RuntimeError('contact order mismatch')
                onsets=np.asarray(a['onsets'],float);returned=np.asarray(a['event_returned'],bool)
                if np.isinf(onsets).any() or not np.isfinite(a['active_fraction']).all() or not np.isfinite(a['contact_envelope']).all():
                    raise RuntimeError('nonfinite worker output')
                if onsets.shape!=(len(returned),len(training['contact_names'])):raise RuntimeError('event shape mismatch')
                table=onsets[returned];tables.append(table)
            units.append({'seed':seed,'n_returned':len(table),
                'n_unreadable_returned':int(np.sum(~np.isfinite(table).any(axis=1))),
                'runaway':d['simulation']['runaway_early_stop_ms'] is not None,
                'geometry':d['xy_geometry_audit'],'node_mapping':d['node_mapping'],
                'worker_sha256':sha(p),'graph_sha256':d['network_cache_source']['sha256']})
        pooled=np.concatenate(tables)
        vector=objective.component_vector(pooled,training['reference'],training['groups'],training['pairs'],training['embedding'],composite=True)
        estimable=all(vector[k]['status']==objective.STATUS_OK for k in COMPONENTS)
        eligible=(row['selection_eligible'] and not any(u['runaway'] for u in units)
                  and len(pooled)>=20 and estimable and vector['D_cloud_composite'] is not None)
        results.append({'candidate_id':row['candidate_id'],'domain':row['domain'],'proposal':row['proposal'],
            'node_field':row['node_field'],'units':units,'n_returned':len(pooled),
            'components':{k:vector[k] for k in COMPONENTS},'D_cloud':vector['D_cloud_composite'],
            'selection_eligible':eligible,'all_components_estimable':estimable,
            'qualification_status':'ELIGIBLE' if eligible else 'UNRESOLVED_OR_CONTROL_ONLY'})
    feasible=[r for r in results if r['selection_eligible']]
    if feasible:
        indices=pareto_indices([[r['components'][k]['value'] for k in COMPONENTS] for r in feasible])
        pareto=[feasible[i]['candidate_id'] for i in indices]
    else:pareto=[]
    ranking=sorted(feasible,key=lambda r:(r['D_cloud'],r['candidate_id']))
    report={'status':'PHASE_AGGREGATED','phase':phase,'candidates':results,
            'ranking':[r['candidate_id'] for r in ranking],'pareto_candidate_ids':pareto,
            'patient_training_contract':{'path':str(training['path']),'sha256':training['sha256']},
            'scientific_status':'DEVELOPMENT_SEARCH_NOT_GLOBAL_OPTIMUM_NOT_FINAL_SUBSTRATE',
            'patient_heldout_opened':False,'final_substrate_frozen':False}
    write(OUT/phase/'aggregate.json',report);return report


def choose_shortlist(reports,candidates):
    scored={r['candidate_id']:r for report in reports for r in report['candidates'] if r['selection_eligible']}
    chosen=[]
    for domain in ('whole_sheet','interior'):
        rows=[r for r in scored.values() if r['domain']==domain]
        if not rows:continue
        choices=sorted(rows,key=lambda r:(r['D_cloud'],r['candidate_id']))[:2]
        choices += [min(rows,key=lambda r:(r['components'][k]['value'],r['candidate_id'])) for k in COMPONENTS]
        for r in choices:
            if r['candidate_id'] not in chosen:chosen.append(r['candidate_id'])
    chosen += [cid for cid in ('control_old_edge','control_historical_matched') if cid not in chosen]
    byid={r['candidate_id']:r for r in candidates}
    return [byid[cid] for cid in chosen]


def refinement(report,candidates):
    byid={r['candidate_id']:r for r in candidates};anchors=[]
    scored={r['candidate_id']:r for r in report['candidates']}
    for domain in ('whole_sheet','interior'):
        ids=[cid for cid in report['ranking'] if scored[cid]['domain']==domain][:2]
        anchors.extend(byid[cid] for cid in ids)
    if not anchors:return []
    pos=positions();seen={r['node_field']['field_sha256'] for r in candidates};rows=[]
    draws=qmc.Sobol(d=4,scramble=True,seed=20260907).random_base2(12)*2-1
    for anchor_index,anchor in enumerate(anchors):
        accepted=0;centers=np.asarray(anchor['node_field']['centers_mm'])
        for i,draw in enumerate(draws):
            scale=.75 if accepted<4 else 2.
            proposed=canonical_centers(centers+scale*draw.reshape(2,2))
            field=field_descriptor(proposed)
            if field['field_sha256'] in seen or not geometry_allowed(proposed,pos,domain=anchor['domain']):continue
            rows.append(decorate({'candidate_id':f'xy_refine_{anchor_index}_{accepted:02d}',
                'domain':anchor['domain'],'proposal':'local_refinement','parent_candidate':anchor['candidate_id'],
                'displacement_scale_mm':scale,'node_field':field,'geometry':audit_geometry(pos,proposed,1499)}))
            seen.add(field['field_sha256']);accepted+=1
            if accepted==8:break
        if accepted!=8:raise RuntimeError('local refinement could not satisfy geometry')
    write(OUT/'refinement_design.json',{'candidates':rows,'parents_selected_from_global_only':True})
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--canary-only',action='store_true');parser.add_argument('--maximum-workers',type=int,default=12)
    args=parser.parse_args()
    if not 1<=args.maximum_workers<=12:raise ValueError('worker limit is 1..12')
    OUT.mkdir(parents=True,exist_ok=True)
    controller_lock=open(OUT/'controller.lock','a')
    fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    design=prepare_design()
    if args.prepare_only:
        print(json.dumps({'status':'PREPARED','candidates':len(design['candidates']),'design':str(OUT/'search_design.json')}));return
    lock_path=OUT/'source_lock.json'
    if lock_path.exists():lock=read(lock_path)['source_hashes'];verify_sources(lock)
    else:lock=source_hashes();write(lock_path,{'source_hashes':lock})
    rows=design['candidates']
    canary_ids=['control_old_edge','control_historical_matched','xy_whole_sheet_000','xy_interior_000']
    canary=[next(r for r in rows if r['candidate_id']==cid) for cid in canary_ids]
    run_phase('canary',canary,lock,args.maximum_workers)
    if args.canary_only:
        write(OUT/'status.json',{'status':'CANARY_COMPLETE_GLOBAL_NOT_STARTED','phase':'canary'});return
    run_phase('global',rows,lock,args.maximum_workers);global_report=aggregate('global',rows)
    refined=refinement(global_report,rows)
    reports=[global_report];all_rows=list(rows)
    if refined:
        run_phase('refinement',refined,lock,args.maximum_workers)
        reports.append(aggregate('refinement',refined));all_rows+=refined
    shortlisted=choose_shortlist(reports,all_rows)
    write(OUT/'development_shortlist.json',{'candidates':shortlisted,'selection_inputs':['global','refinement']})
    run_phase('selection',shortlisted,lock,args.maximum_workers);selected_report=aggregate('selection',shortlisted)
    scored={r['candidate_id']:r for r in selected_report['candidates']}
    selected=[]
    for domain in ('whole_sheet','interior'):
        ids=[cid for cid in selected_report['ranking'] if scored[cid]['domain']==domain]
        if ids:selected.append(ids[0])
    if not selected:
        write(OUT/'status.json',{'status':'SEARCH_COMPLETE_NO_QUALIFIED_GEOMETRY','final_substrate_frozen':False});return
    selected += ['control_old_edge','control_historical_matched']
    byid={r['candidate_id']:r for r in all_rows};confirmation=[byid[cid] for cid in dict.fromkeys(selected)]
    write(OUT/'confirmation_nominees.json',{'candidates':confirmation,'selection_complete_before_confirmation':True,'final_substrate_frozen':False})
    run_phase('confirmation',confirmation,lock,args.maximum_workers);final=aggregate('confirmation',confirmation)
    write(OUT/'final_search_report.json',{'status':'XY_SEARCH_COMPLETE_AWAITING_SCIENTIFIC_REVIEW',
        'confirmation':final,'old_s39_status':'HISTORICAL_CONTROL_ONLY',
        'final_substrate_frozen':False,'global_optimum_established':False,
        'next_scientific_step':'Inspect geometry, event distributions and independent-network uncertainty before any Z/M substrate decision.'})
    subprocess.run([PYTHON,str(ROOT/'scripts/paper_figures/plot_topic4_xy_research.py')],cwd=ROOT,env=ENV,check=True)
    write(OUT/'status.json',{'status':'XY_SEARCH_COMPLETE_AWAITING_SCIENTIFIC_REVIEW','final_substrate_frozen':False})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        write(OUT/'status.json',{'status':'FAILED','reason':str(exc),'updated_unix':time.time()})
        raise
