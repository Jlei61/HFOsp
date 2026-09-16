#!/usr/bin/env python3
"""Traceable pilot summary: paired physical windows, frozen ancestry, no scores."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.topic5_group_event_state.v039.frozen_transfer import equal_anchor_mean
from src.topic5_group_event_state.v035.contracts import atomic_json


def safe_mean(values):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def summarize_values(values):
    values=[float(v) for v in values if v is not None and np.isfinite(v)]
    return dict(n=len(values),median=float(np.median(values)) if values else None,
        positive=sum(v>1e-8 for v in values),negative=sum(v< -1e-8 for v in values),values=values)


def floored_gain(parent,control,state):
    return min(parent,control)-state if all(v is not None for v in [parent,control,state]) else None


def independent_mask(times,valid):
    mask=np.zeros(len(times),bool);last=-np.inf
    for i in np.flatnonzero(valid):
        if times[i]>=last+1800-1e-6:mask[i]=True;last=times[i]
    return mask


def write_csv(path,rows):
    if not rows:return
    names=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=names);w.writeheader();w.writerows(rows)


def run(root,reused,output):
    if output.exists():raise FileExistsError(output)
    queues=json.loads((root/'closure_queue_status.json').read_text())
    if queues['status']!='COMPLETE':raise ValueError('Registered dependency graph is unfinished')
    sha_cache={};integrity=[];code_versions=defaultdict(lambda:defaultdict(set));loaded=[]
    def sha(p):
        p=Path(p);st=p.stat();key=(str(p),st.st_size,st.st_mtime_ns)
        if key not in sha_cache:sha_cache[key]=hashlib.sha256(p.read_bytes()).hexdigest()
        return sha_cache[key]
    def read(p,role):
        p=Path(p);c=json.loads(p.read_text());loaded.append((p,c,role))
        if c.get('status') not in ['COMPLETE','NOT_ESTIMABLE']:raise ValueError('Unfinished card '+str(p))
        if c.get('development_targets_read') is not False or c.get('sealed_partition_opened') is not False:
            raise ValueError('Missing or violated partition contract '+str(p))
        for name,digest in c.get('source_hashes',{}).items():code_versions[role][name].add(digest)
        if c.get('source_sha256'):code_versions[role]['producer_from_card'].add(c['source_sha256'])
        for path_key,hash_key in [('checkpoint','checkpoint_sha256'),('scores','scores_sha256'),('data_path','data_sha256'),
                ('upstream_card','upstream_card_sha256'),('frozen_feature_card','frozen_feature_card_sha256'),('prefix_card','prefix_card_sha256'),
                ('source_card','source_card_sha256'),('export','export_sha256')]:
            if path_key in c and hash_key in c:
                actual=sha(c[path_key]);okay=actual==c[hash_key];integrity.append(dict(card=str(p),field=path_key,pass_=okay))
                if not okay:raise ValueError('Artifact hash mismatch '+str(p)+' '+path_key)
        return c
    queue_attempts=[]
    for name in queues['queues']:
        qp=root/name;manifest=json.loads((qp/'manifest.json').read_text());status=json.loads((qp/'queue_status.json').read_text())
        for file,digest in manifest['source_hashes'].items():
            okay=sha(Path(manifest['source_root'])/file)==digest
            integrity.append(dict(card=str(qp/'manifest.json'),field='source:'+file,pass_=okay))
            if not okay:raise ValueError('Queue source changed '+name+' '+file)
        for job in manifest['jobs']:
            state=status['jobs'][job['id']]
            for file,digest in job['input_hashes'].items():
                okay=sha(file)==digest;integrity.append(dict(card=str(qp/'manifest.json'),field='input:'+file,pass_=okay))
                if not okay:raise ValueError('Queue input changed '+file)
            if state['status'] in ['COMPLETE','NOT_ESTIMABLE']:
                okay=sha(job['output'])==state['output_sha256'];integrity.append(dict(card=str(qp/'queue_status.json'),field='output:'+job['id'],pass_=okay))
                if not okay:raise ValueError('Queue output changed '+job['id'])
        queue_attempts.append(dict(queue=name,status=status['status'],complete=status['complete'],failed=status['failed'],
            manifest_sha256=sha(qp/'manifest.json'),state_sha256=sha(qp/'queue_status.json')))
    if queues.get('runtime_recovery'):
        for rr in queues['runtime_recovery']['recoveries']:
            old=json.loads((root/rr['original_queue']/'manifest.json').read_text());retry=json.loads((root/rr['retry_queue']/'manifest.json').read_text())
            old_jobs={j['id']:j for j in old['jobs']}
            if any(j!=old_jobs[j['id']] for j in retry['jobs']) or old['source_hashes']!=retry['source_hashes']:
                raise ValueError('Runtime retry altered registered scientific recipe')
    data={};availability={};data_cards={}
    for s in ['epilepsiae_1096','epilepsiae_1125','epilepsiae_253']:
        data_cards[s]=read(root/'human_data_v2'/f'{s}.json','human_data')
        data[s]=torch.load(root/'human_data_v2'/f'{s}.pt',map_location='cpu',weights_only=False)
        availability[s]={v['anchor']:bool(np.sum(v['histories']['0.5'][0][:,0])>0) for v in data[s]['samples']}
        tm=read(root/'transfer_data'/f'{s}.json','transfer_data')
        pm=read(root/'frozen_prefix'/f'{s}.json','frozen_prefix')
        if tm['human_data_sha256']!=data_cards[s]['data_sha256']:raise ValueError('Fine-expression pairs reference old human input')
        for p in sorted((root/'decoder_rebuilt'/s/'cards').glob('seed*.json')):
            decoder=read(p,'decoder_reused')
            if sha(Path(decoder['unit_dir'])/'weights.pt')!=decoder['checkpoint_sha256']:raise ValueError('Reused decoder checkpoint changed')
    # All registered fits count in optimization diagnostics, not only LR winners.
    all_fits=[]
    for folder in ['human_main','human_sensitivity','human_views']:
        for p in sorted((root/folder).glob('*/card.json')):
            c=read(p,'human_training');all_fits.append((p,c,folder))
    if len(all_fits)!=270:raise ValueError('Expected 270 finite corrected-input human fit cards')
    for p,c,folder in all_fits:
        if c['data_sha256']!=data_cards[c['subject']]['data_sha256']:raise ValueError('Human fits mix input revisions')
    selected=[];view_selected=[]
    for file,destination in [('main_state_selection.json',selected),('view_state_selection.json',view_selected)]:
        selection=json.loads((root/file).read_text())
        for row in selection['states']:
            if sha(row['source'])!=row['sha256']:raise ValueError('Frozen INNER selection changed')
            c=read(row['source'],'selected_upstream');destination.append((Path(row['source']),c))
    window_rows=[];main_rows=[];lookup={};inventory=[]
    for p,c in selected+view_selected:
        cfg=c['config'];identity=dict(subject=c['subject'],family=c['family'],history_hours=cfg['history_hours'],view=cfg['view'],seed=c['seed'],lr=cfg['lr'],source=str(p))
        for module,params in c['model_inventory'].items():
            for name,v in params.items():inventory.append(dict(**identity,module=module,parameter=name,shape='x'.join(map(str,v['shape'])) or 'scalar',parameters=v['parameters'],requires_grad_at_closeout=v['requires_grad']))
        if cfg['view']!='joint':continue
        with np.load(c['scores']) as z:arrays={k:z[k].copy() for k in z.files}
        lookup[(c['subject'],c['family'],cfg['history_hours'],c['seed'])]=(c,arrays)
        times=arrays['anchor_time'];recent=np.array([availability[c['subject']][t] for t in times])
        for lead in [0,2,6]:
            pre=f'{lead}h_';valid=arrays[pre+'valid'];nonoverlap=independent_mask(times,valid)
            masks={'all_anchors':valid,'nonoverlapping_targets':nonoverlap,'recent_input_available':valid&recent}
            for label,mask in masks.items():
                state=safe_mean(arrays[pre+'state_loss'][mask]);parent=safe_mean(arrays[pre+'baseline_loss'][mask]);constant=safe_mean(arrays[pre+'refitted_constant_loss'][mask])
                shift=mask&np.isfinite(arrays[pre+'shifted_loss'])
                main_rows.append(dict(**identity,lead_hours=lead,target_width_hours=.5,support_subset=label,n_windows=int(mask.sum()),
                    state_loss=state,parent_loss=parent,constant_loss=constant,gain_over_parent=parent-state if state is not None else None,
                    gain_over_constant_floored=floored_gain(parent,constant,state),
                    correct_over_shifted=safe_mean((arrays[pre+'shifted_loss']-arrays[pre+'state_loss'])[shift]),n_shift_pairs=int(shift.sum()),
                    wrongtime_median_offset_hours=c['wrong_time_control']['median_offset_hours'],
                    shifted_scores_identical=int(np.sum(np.abs(arrays[pre+'shifted_loss'][shift]-arrays[pre+'state_loss'][shift])<1e-8))))
            if lead==2:
                for i in np.flatnonzero(valid):window_rows.append(dict(**identity,anchor_time=float(times[i]),target_start=float(times[i]+7200),target_end=float(times[i]+9000),
                    nonoverlapping_target=bool(nonoverlap[i]),recent_input_available=bool(recent[i]),
                    state_loss=float(arrays[pre+'state_loss'][i]),background_loss=float(arrays[pre+'baseline_loss'][i]),constant_loss=float(arrays[pre+'refitted_constant_loss'][i]),
                    shifted_loss=float(arrays[pre+'shifted_loss'][i]) if np.isfinite(arrays[pre+'shifted_loss'][i]) else None))
    paired=[]
    for s in data:
        for seed in [20260905,20260906,20260907]:
            contrasts=[(f'{f}:H8_over_H0.5',(s,f,8.,seed),(s,f,.5,seed)) for f in 'FLN']
            contrasts += [('N_over_L',(s,'N',8.,seed),(s,'L',8.,seed)),('N_over_F',(s,'N',8.,seed),(s,'F',8.,seed)),('L_over_F',(s,'L',8.,seed),(s,'F',8.,seed))]
            for name,left,right in contrasts:
                lc,l=lookup[left];rc,r=lookup[right]
                if not np.array_equal(l['anchor_time'],r['anchor_time']):raise ValueError('Paired state anchors differ')
                recent=np.array([availability[s][t] for t in l['anchor_time']])
                for lead in [0,2,6]:
                    pre=f'{lead}h_';valid=l[pre+'valid']&r[pre+'valid'];delta=r[pre+'state_loss']-l[pre+'state_loss']
                    for label,mask in [('all_anchors',valid),('nonoverlapping_targets',independent_mask(l['anchor_time'],valid)),('recent_input_available',valid&recent)]:
                        paired.append(dict(subject=s,seed=seed,contrast=name,lead_hours=lead,support_subset=label,n_windows=int(mask.sum()),gain=safe_mean(delta[mask])))
    # Frozen transfer tables always preserve the same upstream card identity.
    expression=[];contacts=[];gradient=[];feature_fingerprints=defaultdict(lambda:defaultdict(set))
    for p,c in selected+view_selected:
        cfg=c['config'];view=cfg['view']!='joint';key=p.parent.name
        identity=dict(subject=c['subject'],family=c['family'],history_hours=cfg['history_hours'],view=cfg['view'],seed=c['seed'],source=str(p))
        ec=read(root/('view_expression_transfer' if view else 'expression_transfer')/key/'card.json','expression')
        frozen=read(ec['frozen_feature_card'],'frozen_export')
        if frozen['human_data_sha256']!=c['data_sha256']:raise ValueError('Frozen export uses a different human input')
        with np.load(frozen['export']) as z:
            for arm in ['state','initialized','fixed_history','functional']:
                feature_fingerprints[(c['subject'],c['family'],cfg['history_hours'],cfg['view'])][arm].add(hashlib.sha256(z[arm].tobytes()).hexdigest())
        cc=read(root/('view_contact_transfer' if view else 'contact_transfer')/key/'card.json','contact')
        gc=read(root/('view_gradient_audit' if view else 'human_gradient_audit')/key/'card.json','gradient')
        if any(Path(a['upstream_card'])!=p for a in [ec,cc]) or Path(gc['source_card'])!=p:raise ValueError('Patient/seed/state chain is not identical')
        gradient.append(dict(**identity,selected_step=gc['selected_step'],replay_pass=gc['replay_pass'],finite_difference_pass=gc['finite_difference_pass'],
            max_half_step_loss_difference=max(r['half_step_loss_difference'] for r in gc['records']),
            real_6_to_8h_events=[r['bands']['6_to_8h']['n_real_events'] for r in gc['records']],
            actual_loss_gradient_l1_6_to_8h=[r['bands']['6_to_8h']['gradient_l1'] for r in gc['records']],audit_path=str(root/('view_gradient_audit' if view else 'human_gradient_audit')/key/'card.json')))
        for endpoint,v in ec['results'].items():
            if v['status']!='COMPLETE':expression.append(dict(**identity,endpoint=endpoint,status=v['status']));continue
            arms=v['arms'];parent=arms['prefix_background']['held_out_normalized_mse'];state=arms['state']['held_out_normalized_mse']
            for arm in ['state','functional']:
                loss=arms[arm]['held_out_normalized_mse']
                expression.append(dict(**identity,endpoint=endpoint,arm=arm,status='COMPLETE',n_components=v['n_components'],
                    response_component_indices=json.dumps(v.get('response_component_indices',list(range(v['n_components'])))),
                    n_selection_anchors=v['support']['SELECTION']['anchors'],n_selection_responses=v['support']['SELECTION']['events_or_anchors'],
                    loss=loss,gain_over_parent=parent-loss,gain_over_constant_floored=arms[arm]['gain_over_refitted_constant_floored'],
                    gain_over_fixed_history_floored=floored_gain(parent,arms['fixed_history']['held_out_normalized_mse'],loss),
                    gain_over_initialized_floored=floored_gain(parent,arms['initialized']['held_out_normalized_mse'],loss),
                    selected_parent_fallback=arms[arm]['selected_parent_fallback']))
        with np.load(cc['scores']) as z:cs={k:z[k].copy() for k in z.files}
        with np.load(root/'transfer_data'/f"{c['subject']}.npz") as z:third=np.any(z['ranks']==2,axis=1)
        if len(third)!=len(cs['phase']):raise ValueError('Contact target event mismatch')
        for endpoint,valid in [('exact_next_subset',cs['branch_mask']),('next_subset_all',third),('stop',np.ones(len(third),bool))]:
            mask=(cs['phase']=='SELECTION')&valid;which='_stop' if endpoint=='stop' else '_subset'
            losses={arm:safe_mean(equal_anchor_mean(cs[arm+which][mask],cs['anchor_time'][mask])) for arm in ['state','functional','background','constant','fixed_history','initialized']}
            pair=mask&(cs['shift_donor']>=0)
            recipients=np.flatnonzero(pair);donors=cs['shift_donor'][pair]
            if len(donors) and not np.array_equal(cs['same_prefix_size_key'][recipients],cs['same_prefix_size_key'][donors]):
                raise ValueError('Wrong-time donor breaks exact prefix-and-size contract')
            for arm in ['state','functional']:
                contacts.append(dict(**identity,endpoint=endpoint,arm=arm,n_events=int(mask.sum()),n_anchors=len(np.unique(cs['anchor_time'][mask])),n_raw_blocks=len(np.unique(cs['block'][mask])),
                    loss=losses[arm],gain_over_parent=losses['background']-losses[arm] if losses[arm] is not None else None,
                    gain_over_constant_floored=floored_gain(losses['background'],losses['constant'],losses[arm]),
                    gain_over_fixed_history_floored=floored_gain(losses['background'],losses['fixed_history'],losses[arm]),
                    gain_over_initialized_floored=floored_gain(losses['background'],losses['initialized'],losses[arm]),
                    correct_over_shifted=safe_mean(equal_anchor_mean((cs['shifted_state'+which]-cs['state'+which])[pair],cs['anchor_time'][pair])) if arm=='state' else None,
                    n_shifted_pairs=int(pair.sum()) if arm=='state' else 0,
                    minimum_donor_event_offset_hours=float(np.min(np.abs(cs['event_time'][recipients]-cs['event_time'][donors]))/3600) if len(donors) else None,
                    minimum_donor_state_offset_hours=float(np.min(np.abs(cs['anchor_time'][recipients]-cs['anchor_time'][donors]))/3600) if len(donors) else None,
                    constant_worse_than_parent=losses['constant']>losses['background']+1e-8 if losses['constant'] is not None else None))
    instruments=[]
    for directory,tag in [('instruments','joint'),('capacity_instruments','L32'),('rollout_instruments','rollout'),('frozen_view_instruments','single_view')]:
        groups=defaultdict(list)
        for p in sorted((reused/directory).glob('*/card.json')):
            c=read(p,'synthetic_'+tag);groups[(c['case'],c['family'],c['config']['view'])].append((p,c))
        for key,g in sorted(groups.items()):
            p,c=min(g,key=lambda row:(row[1]['selected_inner'],row[1]['config']['lr']))
            instruments.append(dict(case=key[0],family=key[1],view=key[2],experiment=tag,selected_lr=c['config']['lr'],selected_step=c['selected_step'],
                optimization_limited=c['optimization_limited'],held_out=c['held_out'],truth=c['truth'],source=str(p)))
    synthetic_probes=[]
    for p in sorted((reused/'frozen_view_probes_v2').glob('*/card.json')):
        c=read(p,'synthetic_frozen_probe');synthetic_probes.append({k:c[k] for k in ['case','family','view','results','truth','upstream_optimization_limited','upstream_source_card']})
    optimization={}
    for folder in ['human_main','human_sensitivity','human_views']:
        cards=[c for p,c,f in all_fits if f==folder]
        optimization[folder]={stage:dict(cards=len(cards),initial_selected=sum(c['stages'][stage]['selected_step']==0 for c in cards),
            optimization_limited=sum(c['stages'][stage]['optimization_limited'] for c in cards),
            optimizer_updates=sum(c['stages'][stage]['optimizer_updates'] for c in cards),
            selected_steps=Counter(c['stages'][stage]['selected_step'] for c in cards)) for stage in ['background','event','refitted_constant']}
    sensitivity=[]
    groups=defaultdict(list)
    for p,c,folder in all_fits:
        if folder=='human_sensitivity':
            cfg=c['config'];groups[(c['subject'],c['family'],cfg['width'],cfg['event_only'],cfg['forecast'],c['seed'])].append((p,c))
    for key,g in sorted(groups.items()):
        p,c=min(g,key=lambda row:(row[1]['stages']['event']['selected_inner'],row[1]['config']['lr']))
        direct_card,direct=lookup[(key[0],key[1],8.,key[5])]
        comparisons={}
        with np.load(c['scores']) as z:
            if not np.array_equal(z['anchor_time'],direct['anchor_time']):raise ValueError('Sensitivity query support differs')
            for lead in [0,2,6]:
                pre=f'{lead}h_';mask=z[pre+'valid']&direct[pre+'valid']
                comparisons[str(lead)]=dict(gain_over_direct_same_family=safe_mean((direct[pre+'state_loss']-z[pre+'state_loss'])[mask]),n_anchors=int(mask.sum()))
                if key[2]==32:
                    nc,na=lookup[(key[0],'N',8.,key[5])]
                    comparisons[str(lead)]['gain_over_N16']=safe_mean((na[pre+'state_loss']-z[pre+'state_loss'])[mask&na[pre+'valid']])
        sensitivity.append(dict(subject=key[0],family=key[1],width=key[2],event_only=key[3],forecast=key[4],seed=key[5],source=str(p),
            selected_step=c['stages']['event']['selected_step'],optimization_limited=c['stages']['event']['optimization_limited'],metrics=c['metrics'],paired_comparisons=comparisons))
    group_summary={}
    for label,rows,keys in [('main',main_rows,['subject','family','history_hours','lead_hours','support_subset']),('paired',paired,['subject','contrast','lead_hours','support_subset']),
            ('expression',expression,['subject','family','history_hours','view','endpoint','arm']),('contact',contacts,['subject','family','history_hours','view','endpoint','arm'])]:
        grouped=defaultdict(list)
        for row in rows:
            if row.get('status')=='NOT_ESTIMABLE':continue
            grouped[tuple(row[k] for k in keys)].append(row)
        result=[]
        fields=['gain'] if label=='paired' else ['gain_over_parent','gain_over_constant_floored','correct_over_shifted'] if label=='main' else ['gain_over_parent','gain_over_constant_floored','gain_over_fixed_history_floored','gain_over_initialized_floored','correct_over_shifted']
        for key,g in sorted(grouped.items()):result.append(dict(**dict(zip(keys,key)),effects={f:summarize_values([v.get(f) for v in g]) for f in fields},
            physical_support=[{k:v[k] for k in ['seed','n_windows','n_anchors','n_raw_blocks','n_selection_anchors'] if k in v} for v in g]))
        group_summary[label]=result
    h2b=read(root/'seizure_transfer/support_audit.json','seizure_support')
    nonlinear=read(root/'audits/nonlinear_usage.json','nonlinear_usage')
    calibration=read(root/'calibration/window_support.json','readout_calibration')
    precision=read(root/'audits/stored_score_precision.json','saved_score_precision')
    for role in ['human_training','contact','expression','gradient','frozen_export']:
        if any(len(hashes)>1 for hashes in code_versions[role].values()):
            raise ValueError('Formal human role mixes source versions: '+role)
    output.mkdir(parents=True)
    for name,rows in [('paired_window_scores.csv',window_rows),('model_parameter_inventory.csv',inventory),('main_endpoint_scores.csv',main_rows),
            ('paired_history_transition_comparisons.csv',paired),('frozen_expression_scores.csv',expression),('frozen_contact_scores.csv',contacts)]:write_csv(output/name,rows)
    result=dict(status='COMPLETE',scope='v039 finite design pilot and v038 review-package closeout; original scientific mechanism remains unestablished',
        root=str(root),reused_instrument_root=str(reused),queue_status=queues,queue_attempts=queue_attempts,registered_human_fits=len(all_fits),selected_main_states=len(selected),selected_single_view_states=len(view_selected),
        data_support={s:data_cards[s]['support'] for s in data},input_repair=json.loads((root/'input_boundary/background_repair_parity.json').read_text()),
        single_view_family_selection=json.loads((root/'single_view_family_selection.json').read_text()),
        single_view_training=[dict(source=str(p),subject=c['subject'],family=c['family'],view=c['config']['view'],seed=c['seed'],config=c['config'],
            metrics=c['metrics'],stages=c['stages'],selected_transition_deltas=c['selected_transition_deltas']) for p,c in view_selected],
        optimization=optimization,main_rows=main_rows,paired_comparisons=paired,expression_rows=expression,contact_rows=contacts,grouped=group_summary,
        gradient_audit=gradient,nonlinear_usage=nonlinear,instruments=instruments,synthetic_frozen_probes=synthetic_probes,sensitivity=sensitivity,
        frozen_feature_uniqueness=[dict(subject=k[0],family=k[1],history_hours=k[2],view=k[3],unique_arrays_by_arm={a:len(v) for a,v in arms.items()}) for k,arms in sorted(feature_fingerprints.items())],
        seizure_transfer=h2b,window_support_calibration=calibration,
        saved_score_precision=precision,
        integrity=dict(checks=len(integrity),passed=all(v['pass_'] for v in integrity),source_versions={role:{name:sorted(hashes) for name,hashes in files.items()} for role,files in code_versions.items()}),
        inference_boundaries=['Three already-seen design patients; no independent confirmation or cohort inference.',
            'Seeds quantify optimization variation; nonoverlapping targets still share history and biological context.',
            'Nonzero gradients and nonlinear branch usage do not establish long-history predictive gain or physiological feedback.',
            'Raw latent cross-task information can combine independent causes; functional readout transfer and counterexamples remain separate.',
            'Full training-pipeline and seizure power are not calibrated; oracle-factor support diagnostic is optimistic and conditional.',
            'Block-close publication delay is part of the operator; no immediate event-time or clinical deployment claim.'],
        development_targets_read=False,sealed_partition_opened=False,source_sha256=sha(__file__))
    atomic_json(output/'summary_main.json',result);atomic_json(output/'artifact_integrity.json',dict(status='PASS',checks=integrity,files_hashed=len(sha_cache)))
    print(json.dumps(dict(status='COMPLETE',human_fits=len(all_fits),frozen_states=len(gradient),artifact_checks=len(integrity))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reused',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.reused,a.output)
