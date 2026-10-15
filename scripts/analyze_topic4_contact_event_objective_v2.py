#!/usr/bin/env python3
"""Freeze a separate development loss and audit already generated outputs.

Never changes the active v1 controller, objective, or nomination. Training
waveforms only; no held-out packet events or new physical simulations.
"""
from pathlib import Path
import csv,json,pickle,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.topic4_contact_event_objective_v2 import (
    VERSION,QUANTILES,TIME_GRID_MS,ContactEventObjectiveV2,FrozenCentroidAnchor,
    patient_event,stack_events,worker_events,
)

P=ROOT/'results/topic4_sef_hfo/contact_timing_shape_pilot'
R=ROOT/'results/topic4_sef_hfo/multievent_distribution_search_v2_1'
OUT=ROOT/'results/topic4_sef_hfo/contact_event_objective_revision_v2'
KEYS=['participation','centroid_structure','local_shape','recruitment','joint_envelope']
LABELS=['Which contacts\nparticipate','Centroid order / lag\n(full event table)','Local envelope\nduration / shape','Relative contact\n50% mass times','All-contact\nenvelope structure']
COLORS=['#487dad','#b97a3c','#a85483','#2a9382','#7c65ad']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def writecsv(path,rows):
    with Path(path).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=165,bbox_inches='tight')
    plt.close(fig)
def dist(x):
    x=np.asarray(x,float)
    return dict(mean=float(x.mean()),median=float(np.median(x)),sd=float(x.std()),
                q05=float(np.quantile(x,.05)),q95=float(np.quantile(x,.95)))
def display(cid,anchors):
    if cid in anchors:return ['Starting point 1','Starting point 2','Starting point 3'][anchors.index(cid)]
    s=cid.split('_');return f'Point {s[1][-1]} / {s[2]} {s[3]}'


def main():
    OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    design=json.loads((P/'design.json').read_text())
    v1=pickle.load((P/'objective.pkl').open('rb'))
    old=pickle.load((R/'training_objective_v2_1.pkl').open('rb'))
    ep=ROOT/'results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2/evaluator.pkl'
    ev=pickle.load(ep.open('rb'))
    patient=stack_events([patient_event(e['arrays_path'],v1.names) for e in design['patient_training']])
    labels=np.array([e['mode'] for e in design['patient_training']])
    weights=np.array([old.proportions[k]/np.sum(labels==k) for k in labels])
    objective=ContactEventObjectiveV2(patient,weights,np.isfinite(ev.fit),v1.names,FrozenCentroidAnchor(old,ev.fit))
    # Freeze the reference before reading any output for rescoring. This is a
    # development revision informed by earlier audits, not a preregistered test.
    with (OUT/'objective.pkl').open('wb') as f:pickle.dump(objective,f)
    np.savez_compressed(OUT/'patient_training_representation.npz',**patient,weights=weights,labels=labels,names=v1.names)
    kernel_scales={b:[dict(contact=n,linear_variance=k.linear_variance,linear_scale=k.linear_scale,
                           nonlinear_variance=k.nonlinear_variance,nonlinear_scale=k.nonlinear_scale,bandwidth=k.bandwidth)
                       for n,k in zip(v1.names,ks)] for b,ks in objective.marked.items()}
    manifest=dict(version=VERSION,status='DEVELOPMENT_REFERENCE_FIXED_FOR_OFFLINE_AUDIT',
        full_fit_events=len(ev.fit),waveform_training_events=len(labels),waveform_review_events_read=0,
        full_fit_source=str(ep),full_fit_sha256=sha(ep),patient_training=design['patient_training'],
        names=v1.names,weights=dict(zip(KEYS,[.2]*5)),within_envelope_block_weights={'mean':.5,'nonlinear':.5},
        reference_sampling='Frozen mode-balanced, block-spread packet; mode-frequency weighting does not undo block-spread selection.',
        target_scope='Large FIT event table anchors participation and centroid features; 46 TRAIN waveforms are a development morphology reference.',
        local_and_recruitment_target='Full-FIT contact participation probability times packet contact-conditional shape, with an absent-contact atom.',
        unit='One network and one noise realization; score all its eligible primary events. Equal weight per network; no partial candidate averages.',
        quantiles=QUANTILES.tolist(),joint_time_grid_ms=TIME_GRID_MS.tolist(),time_unit='ms',
        interpolation='Cumulative-bin interpolation; 5 to 95 percent. Descriptor revision; event detector and windows unchanged.',
        absolute_amplitude_used=False,all_contacts_used_in_joint=True,model_classifier_used=False,
        manual_TA_TB_paths_used=False,native_penalty='NOT_INCLUDED_NOT_YET_VALIDATED',
        conditional_support=objective.conditional_reference_support,kernel_scales=kernel_scales,
        centroid_scale=objective.centroid_anchor.scale,
        joint_scales=dict(linear=objective.joint.linear_scale,nonlinear=objective.joint.nonlinear_scale,bandwidth=objective.joint.bandwidth),
        objective_sha256=sha(OUT/'objective.pkl'),source_sha256=sha(ROOT/'src/topic4_contact_event_objective_v2.py'),
        scope='Offline rescoring only. No running-v1 mutation, new simulation, candidate freeze, or mechanism acceptance.')
    dump(OUT/'manifest.json',manifest)

    checks=[]
    for p in sorted(set(np.r_[np.linspace(0,1,21),old.proportions[1]])):
        w=np.where(labels==1,p/np.sum(labels==1),(1-p)/np.sum(labels==0))
        checks.append(dict(experiment='TA_fraction',x=float(p),distance=objective.joint.population(patient['joint_envelope'],w)))
    for f in [.0,.25,.5,.75,1.,1.25,1.5,2.]:
        for kind in ['local_width','shape_scatter']:
            values=[]
            for k in objective.marked['local_shape']:
                x=k.reference.copy()
                if kind=='local_width':x[:,1:]*=f
                else:
                    center=np.average(x[1:,1:],axis=0,weights=k.weights[1:])
                    x[1:,1:]=center+(x[1:,1:]-center)*f
                values.append(k.population(x,k.weights))
            checks.append(dict(experiment=kind,x=float(f),distance=float(np.mean(values))))
    assert abs(min(r['distance'] for r in checks if r['experiment']=='TA_fraction'))<1e-10
    for kind in ['local_width','shape_scatter']:
        assert abs(next(r['distance'] for r in checks if r['experiment']==kind and r['x']==1))<1e-10
        assert all(r['distance']>1e-5 for r in checks if r['experiment']==kind and r['x']!=1)
    writecsv(OUT/'known_reference_checks.csv',checks)
    fig,axs=plt.subplots(1,3,figsize=(12,3.6))
    for ax,kind,title,target in zip(axs,['TA_fraction','local_width','shape_scatter'],
            ['TA frequency in reference mixture','Local duration multiplier','Local shape scatter multiplier'],[old.proportions[1],1,1]):
        rr=[r for r in checks if r['experiment']==kind]
        ax.plot([r['x'] for r in rr],[r['distance'] for r in rr],'o-',c='#487dad',ms=4)
        ax.axvline(target,color='black',ls='--',lw=1);ax.set(xlabel=title,ylabel='Population distance');ax.grid(alpha=.15)
    fig.suptitle('Known-reference checks: correct frequency, duration and scatter are preferred')
    fig.tight_layout(rect=(0,.07,1,.93));fig.text(.5,.01,'Reference-measure interventions; not SNN runs or a patient validation test.',ha='center')
    save(fig,'known_reference_checks')

    # Deliberately use only the completed baseline + A batch, same two TRAIN
    # units per condition. Ongoing B and new-noise replays remain with v1.
    input_paths=[P/'baseline_train_scores.json',P/'A_scores.json']
    records=[r for path in input_paths for r in json.loads(path.read_text())['candidates']]
    anchors=[r['candidate_id'] for r in design['anchors']]
    parameters=[]
    for rec in records:
        c=rec['candidate'];centers=c['node_field']['centers_mm']
        parameters.append(dict(candidate_id=rec['candidate_id'],name=display(rec['candidate_id'],anchors),
            core1_x_mm=centers[0][0],core1_y_mm=centers[0][1],core2_x_mm=centers[1][0],core2_y_mm=centers[1][1],
            threshold_field_gain=c['node_mapping']['node_gain'],
            **{k:c['dynamic_parameters'][k] for k in ['E_to_E_weight_scale','E_to_I_weight_scale','I_to_E_weight_scale','tau_d_GABA_ms']}))
    writecsv(OUT/'candidate_parameters.csv',parameters)
    unitrows=[];candidates=[];detailed={};tables={}
    for rec in records:
        cid=rec['candidate_id'];detailed[cid]={};scores=[]
        for uid,u in rec['units'].items():
            if not uid.endswith('dyn_7101'):raise ValueError('TRAIN units only')
            tab,details,names,worker=worker_events(u['worker_path'])
            if names!=v1.names:raise ValueError('model contact ordering differs')
            sc=objective.score(tab,worker.get('physical_status'));scores.append(sc)
            detailed[cid][uid]=dict(worker_path=u['worker_path'],score=sc,event_observables=details)
            tables[cid,uid]=tab
            row=dict(candidate_id=cid,name=display(cid,anchors),unit=uid,status=sc['status'],N=sc['N'],
                     old_v1_loss=u['score']['loss'],new_loss=sc['loss'],new_A=sc.get('A'),new_B=sc.get('B'),
                     local_width_median_ms=float(np.median([e['local_width_ms'] for e in details])) if details else None)
            for k in KEYS:
                for comp in ['D_off','A','B']:row[f'{k}_{comp}']=sc.get('blocks',{}).get(k,{}).get(comp)
            unitrows.append(row)
        eligible=all(s['loss'] is not None for s in scores)
        row=dict(candidate_id=cid,name=display(cid,anchors),eligible=eligible,old_v1_loss=rec['loss'],
                 new_loss=float(np.mean([s['loss'] for s in scores])) if eligible else None)
        for k in KEYS:row[k]=float(np.mean([s['blocks'][k]['D_off'] for s in scores])) if eligible else None
        candidates.append(row)
    writecsv(OUT/'per_network_scores.csv',unitrows);writecsv(OUT/'candidate_scores.csv',candidates)
    dump(OUT/'per_network_details.json',detailed)
    dump(OUT/'input_snapshot.json',[dict(path=str(path),sha256=sha(path)) for path in input_paths])
    fig,axs=plt.subplots(1,2,figsize=(13,5.1),gridspec_kw={'width_ratios':[1.1,2.0]})
    for i,r in enumerate(candidates):
        for x,prop,color in [(0,'old_v1_loss','#b97a3c'),(1,'new_loss','#487dad')]:
            if r[prop] is not None:axs[0].text(x+.05,i,f'{r[prop]:.3f}',va='center',color=color,fontsize=10)
        if not r['eligible']:axs[0].text(1.05,i,'unranked',va='center',color='gray',fontsize=9)
    axs[0].set(yticks=np.arange(len(candidates)),yticklabels=[r['name'] for r in candidates],
               xticks=[.3,1.3],xticklabels=['Prior pilot score','Revised score'],xlim=(-.05,2),ylim=(len(candidates)-.4,-.6))
    axs[0].tick_params(length=0);[s.set_visible(False) for s in axs[0].spines.values()]
    vals=np.array([[np.nan if r[k] is None else r[k] for k in KEYS] for r in candidates])
    im=axs[1].imshow(np.ma.masked_invalid(vals),aspect='auto',cmap='YlOrRd',vmin=0,vmax=max(1.5,float(np.nanmax(vals))))
    for i in range(len(candidates)):
        for j in range(5):
            if np.isfinite(vals[i,j]):axs[1].text(j,i,f'{vals[i,j]:.2f}',ha='center',va='center',fontsize=9,color='black')
    axs[1].set(xticks=np.arange(5),xticklabels=LABELS,yticks=np.arange(len(candidates)),yticklabels=[])
    axs[1].tick_params(axis='x',labelsize=8);fig.colorbar(im,ax=axs[1],shrink=.8,label='Block distance; lower is better')
    fig.suptitle('Every condition still has a large local-envelope mismatch',fontsize=14)
    fig.tight_layout(rect=(0,.10,1,.92));fig.text(.5,.025,'Two TRAIN networks per condition. Scores across objective versions have different units; compare rankings, not magnitudes.',ha='center',fontsize=9)
    save(fig,'candidate_error_components')

    base=records[0];better=next(r for r in records if r['candidate_id']=='tshape_anchor1_A_plus')
    comparison=[]
    for rec in [base,better]:
        vals=list(rec['units'].values());d={}
        for key in ['A','B']:
            d[key]=float(np.mean([u['score']['temporal'][key]/u['score']['temporal']['normalizer'] for u in vals]))
        d.update(candidate_id=rec['candidate_id'],temporal_loss=d['A']-d['B'],old_joint=rec['loss'],
                 new_score=next(c['new_loss'] for c in candidates if c['candidate_id']==rec['candidate_id']))
        comparison.append(d)
    dump(OUT/'prior_score_decomposition.json',comparison)

    scans=[]
    for uid,u in base['units'].items():
        for factor in [0.,5.,10.,20.,30.,40.]:
            table,detail,names,worker=worker_events(u['worker_path'],factor)
            score=objective.score(table,worker.get('physical_status'))
            scans.append(dict(unit=uid,extra_smoothing_sd_ms=factor,new_loss=score['loss'],
                local_width_median_ms=float(np.median([e['local_width_ms'] for e in detail])),
                retained_mass_median=float(np.median([e['retained_window_mass_fraction'] for e in detail])),
                **{k:score['blocks'][k]['D_off'] for k in KEYS}))
    writecsv(OUT/'offline_local_duration_scan.csv',scans)
    fig,axs=plt.subplots(1,3,figsize=(13,4.1))
    for ui,uid in enumerate(base['units']):
        rr=[r for r in scans if r['unit']==uid];xx=[r['extra_smoothing_sd_ms'] for r in rr];ls=['-','--'][ui]
        for k,c in zip(KEYS,COLORS):axs[0].plot(xx,[r[k] for r in rr],ls,color=c,lw=1.5)
        axs[1].plot(xx,[r['new_loss'] for r in rr],ls,color='#487dad',marker='o',ms=4)
        axs[2].plot(xx,[r['local_width_median_ms'] for r in rr],ls,color='#487dad',marker='o',ms=4)
    pw=[]
    for i in range(len(labels)):
        m=patient['participation'][i]>0;q=patient['local_shape'][i]
        pw.append(float(np.median((q[:,17]-q[:,1])[m])))
    pp=np.asarray(pw);order=np.argsort(pp);cum=np.cumsum(weights[order]);qq=np.interp([.05,.5,.95],cum,pp[order])
    axs[2].axhspan(qq[0],qq[2],alpha=.12,color='black');axs[2].axhline(qq[1],color='black',ls=':',label='TRAIN patient reference')
    for ax,yl in zip(axs,['Separate block distances','Five-block average distance','Event-median local 10-90% width (ms)']):
        ax.set(xlabel='Extra offline smoothing SD (ms)',ylabel=yl);ax.axvline(0,color='gray',ls=':',lw=1);ax.grid(alpha=.12)
    fig.suptitle('Wider local envelopes improve the observable match; this is not a physical simulation')
    fig.legend([plt.Line2D([],[],color=c) for c in COLORS],[x.replace('\n',' ') for x in LABELS],
               loc='lower center',ncol=3,fontsize=8,bbox_to_anchor=(.5,.0))
    fig.tight_layout(rect=(0,.21,1,.92))
    fig.text(.5,.16,'Solid: topology 6101. Dashed: topology 6102. Same dynamics seed 7101. Grey: TRAIN patient 5-95%; dotted: median.',ha='center',fontsize=8)
    save(fig,'offline_duration_response')
    dump(OUT/'summary.json',dict(status='OFFLINE_REVISION_COMPLETE_PENDING_REVIEW',
        n_conditions=len(candidates),n_network_units=len(unitrows),n_rankable_conditions=sum(r['eligible'] for r in candidates),
        n_new_physical_runs=0,patient_reference_event_width_ms=dict(zip(['q05','median','q95'],qq.tolist())),
        candidate_scores=candidates,prior_score_decomposition=comparison,
        offline_duration_scan=scans,minimum_retained_window_mass_median=min(r['retained_mass_median'] for r in scans),
        heldout_waveforms_opened=False,running_v1_modified=False))
    print(json.dumps(dict(output=str(OUT),conditions=len(candidates),units=len(unitrows),patient_width=qq.tolist()),ensure_ascii=False))


if __name__=='__main__':main()
