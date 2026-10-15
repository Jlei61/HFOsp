#!/usr/bin/env python3
"""Prepared-target figures and automatic post-pilot distribution/movie delivery."""
from pathlib import Path
import argparse,copy,csv,json,pickle,sys,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from PIL import Image
from scripts.run_topic4_contact_timing_shape_pilot import OUT,OLD,get_frozen,PARAMETERS,vector,score_record
from scripts import run_topic4_multievent_distribution_v2_1 as engine
from src.topic4_envelope_joint_pilot import aligned_packet_event,model_events
from src.topic4_interictal_repaired_evaluation import rank_features

F=OUT/'figures';F.mkdir(exist_ok=True)
COLORS=['#3379ac','#cc8245','#6a579d']
ARM_COLORS=['#3379ac','#dd922b','#8458a4','#228f7c']
KEYS=['contact_width_ms','centroid_span_ms','overlap','shape_lag_ms']
LABELS=['Median local 10-90% width (ms)','Contact centroid span (ms)','Pairwise temporal overlap','Local-shape contribution to lag (ms)']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
entries=[]

def readable_name(cid,design):
    ids=[c['candidate_id'] for c in design['anchors']]
    if cid in ids:return ['Best joint starting point','Reference placement A','Placement B / smaller threshold shift'][ids.index(cid)]
    parts=cid.split('_')
    return f'Placement {parts[1].replace("anchor","")} / batch {parts[2]} / {parts[3]}'

def readcsv(path):return list(csv.DictReader(Path(path).open()))
def writecsv(path,rows):
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def save(fig,name,description):
    for ext in ['png','pdf']:fig.savefig(F/f'{name}.{ext}',dpi=155,bbox_inches='tight')
    plt.close(fig);entries.append((name+'.png / .pdf',description))

def distribution(values,weights=None):
    a=np.asarray(values,float);ok=np.isfinite(a);a=a[ok]
    if not len(a):return dict(n=0,mean=np.nan,median=np.nan,sd=np.nan,variance=np.nan,q05=np.nan,q25=np.nan,q75=np.nan,q95=np.nan)
    if weights is None:
        q=np.quantile(a,[.05,.25,.5,.75,.95]);mean=a.mean();var=a.var()
    else:
        w=np.asarray(weights,float)[ok];w/=w.sum();ix=np.argsort(a);cum=np.cumsum(w[ix]);q=a[ix[np.minimum(np.searchsorted(cum,[.05,.25,.5,.75,.95]),len(a)-1)]]
        mean=w@a;var=w@((a-mean)**2)
    return dict(n=len(a),mean=float(mean),median=float(q[2]),sd=float(np.sqrt(var)),variance=float(var),q05=float(q[0]),q25=float(q[1]),q75=float(q[3]),q95=float(q[4]))

def initial_figures(design,temporal,old):
    checks=readcsv(OUT/'objective_population_checks.csv');fig,axs=plt.subplots(1,3,figsize=(12,3.8))
    for ax,ex,title,target in zip(axs,['mode_frequency','local_duration_multiplier','within_mode_timing_scatter'],
                                 ['TA proportion','Local duration multiplier','Within-mode timing scatter multiplier'],[old.proportions[1],1.,1.]):
        rr=[r for r in checks if r['experiment']==ex];xx=[float(r['value']) for r in rr];yy=[float(r['distance']) for r in rr]
        ax.plot(xx,yy,'o-',c='#3379ac');ax.axvline(target,c='black',ls='--',lw=1);ax.set(xlabel=title,ylabel='Population kernel distance');ax.grid(alpha=.15)
    fig.suptitle('New objective checks: does the known target minimize the distance?',fontsize=14)
    fig.tight_layout(rect=(0,.05,1,.90));fig.text(.5,.01,'Synthetic changes of TRAIN patient features. These checks do not test SNN physical capability.',ha='center')
    save(fig,'objective_target_checks','已知患者特征分布的人为比例、局部宽度和类内时间散布变化，检查正确目标是否得到最低新距离。黑虚线是已知目标，未使用新模拟或留后波形。**关注点**：这是目标性质检查，不能当作物理拟合成功。')
    # Reuse the prior paired offline diagnostic to ensure that the new loss
    # detects precisely the local-width change the old objective cannot see.
    rr=readcsv(OLD/'local_shape_vs_recruitment_audit/contact_times.csv');group={}
    for row in rr:
        if row['candidate'] not in ['1','2','3'] or not row['unit'].endswith('dyn_7101'):continue
        key=(int(row['candidate']),row['unit'],row['arm'],row['event_id'])
        group.setdefault(key,[]).append(row)
    output=[]
    for ci in [1,2,3]:
        c=design['anchor_scores'][ci-1]
        for unit,u in c['units'].items():
            if not unit.endswith('dyn_7101'):continue
            for arm in ['original','width','timing','both']:
                vv=[];mu=[]
                for key,rs in group.items():
                    if key[:3]!=(ci,unit,arm):continue
                    v=np.zeros((len(temporal.names),5));m=np.full(len(temporal.names),np.nan)
                    med=np.median([float(r['t50_ms']) for r in rs])
                    for r in rs:
                        i=temporal.names.index(r['contact']);q10,q50,q90=[float(r[k]) for k in ['t10_ms','t50_ms','t90_ms']]
                        m[i]=float(r['centroid_ms']);v[i]=[1,q50-med,q50-q10,q90-q50,m[i]-q50]
                    vv.append(v);mu.append(m)
                score=temporal.score(np.array(vv));prior=old.score_network(np.array(mu))['loss_off']
                output.append(dict(anchor=ci,unit=unit,arm=arm,old_loss=prior,temporal_loss=score['loss'],joint_loss=.5*(prior+score['loss'])))
    writecsv(OUT/'offline_objective_sensitivity.csv',output)
    fig,axs=plt.subplots(1,3,figsize=(12,4.3));fig.subplots_adjust(top=.82,bottom=.25,wspace=.3)
    for ax,key,title in zip(axs,['old_loss','temporal_loss','joint_loss'],['Original objective','New timing / shape component','New joint objective']):
        for ci in [1,2,3]:
            for ai,arm in enumerate(['original','width','timing','both']):
                vals=[r[key] for r in output if r['anchor']==ci and r['arm']==arm]
                ax.scatter([ci+(ai-1.5)*.13]*len(vals),vals,c=ARM_COLORS[ai],s=20)
        ax.set(xticks=[1,2,3],xticklabels=['Best joint','Placement A','Placement B'],ylabel='Loss',title=title);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Frozen-output sensitivity: the new objective can see local envelope shape',fontsize=14)
    fig.legend([plt.Line2D([],[],color=c,lw=3) for c in ARM_COLORS],['Original output','Broaden only','Compress timing only','Both'],loc='lower center',ncol=4,bbox_to_anchor=(.5,.045))
    fig.text(.5,.005,'Two training units per anchor; offline transformations are controls, not pilot simulation results.',ha='center')
    save(fig,'offline_objective_sensitivity','对三个固定工作点的旧输出做同一组离线操作，检查旧目标、新时间项和新总目标是否响应局部形状。每条件两个训练单元单独显示。**关注点**：此图只证明新损失能看到变化，不是本轮物理搜索结果。')

def parameter_figures(design,reports):
    records=[r for report in reports for r in report['candidates']]
    anchor_ids=[c['candidate_id'] for c in design['anchors']]
    events=[]
    for phase in ['baseline_train','A','B']:
        path=OUT/f'{phase}_event_observables.csv'
        if path.exists():events+=readcsv(path)
    pointrows=[];rankrows=[]
    for r in records:
        cid=r['candidate_id'];c=r['candidate'];ai=anchor_ids.index(cid)+1 if cid in anchor_ids else int(cid.split('_')[1].replace('anchor',''))
        phase='baseline' if cid in anchor_ids else c['arm'];v=vector(c)
        rankrows.append(dict(candidate_id=cid,anchor=ai,phase=phase,ranking_eligible=r['ranking_eligible'],joint_loss=r['loss'],**dict(zip(PARAMETERS,v))))
        for unit,u in r['units'].items():
            sc=u['score'];er=[e for e in events if e['candidate_id']==cid and e['unit']==unit]
            pointrows.append(dict(candidate_id=cid,anchor=ai,phase=phase,unit=unit,N=sc['N'],physical_status=sc['physical_status'],
                old_loss=sc['old']['loss_off'],temporal_loss=sc['temporal']['loss'],joint_loss=sc['loss'],
                **{k:distribution([float(e[k]) for e in er])['median'] for k in KEYS},**dict(zip(PARAMETERS,v))))
    writecsv(OUT/'training_parameter_observables.csv',pointrows);writecsv(OUT/'candidate_training_ranking.csv',rankrows)
    keys=['old_loss','temporal_loss','joint_loss','contact_width_ms','centroid_span_ms','overlap']
    titles=['Original loss','Timing / shape loss','Joint loss','Local width (ms)','Centroid span (ms)','Temporal overlap']
    fig,axs=plt.subplots(6,5,figsize=(16,15));fig.subplots_adjust(left=.08,right=.985,bottom=.10,top=.91,hspace=.42,wspace=.35)
    for i,(key,title) in enumerate(zip(keys,titles)):
        for j,param in enumerate(PARAMETERS):
            ax=axs[i,j]
            for r in pointrows:
                val=r[key]
                if val is None or not np.isfinite(float(val)):continue
                ax.scatter(r[param],val,c=COLORS[r['anchor']-1],marker={'baseline':'o','A':'^','B':'s'}[r['phase']],s=23,alpha=.7)
            ax.grid(alpha=.15)
            if j==0:ax.set_ylabel(title)
            if i==5:ax.set_xlabel(['Threshold offset scale','E to E strength','E to I strength','I to E strength','GABA decay (ms)'][j])
    fig.suptitle('Physical parameter settings and measured responses\nThree fixed core placements; two training units per condition',fontsize=16)
    fig.text(.5,.025,'Blue: best joint anchor; orange: placement A; purple: placement B. Circle: anchor; triangle: batch A; square: adaptive batch B.\nEach point is one run. Several parameters change together: these scatterplots do not isolate the causal effect of one parameter.\nRunaway and insufficient-event conditions remain in the tables; no numerical joint loss is fabricated for them.',ha='center',fontsize=10)
    save(fig,'physical_parameters_and_observables','五个实际物理参数与原损失、新时间项、总损失、局部宽度、质心跨度和时间重叠的关系。每点是一次运行，颜色区分三个固定位置，形状区分基线及A/B批次。**关注点**：是联合局部扰动而非单参数因果扫描；不能把散点斜率直接解释为该参数机制。')
    return records

def load_review(design,temporal):
    descriptors=[aligned_packet_event(e['arrays_path'],temporal.names) for e in design['patient_review']]
    labels=np.array([e['mode'] for e in design['patient_review']]);values=np.array([d['values'] for d in descriptors])
    obj=copy.deepcopy(temporal)
    obj.reference=obj.features(values)
    obj.patient_weights=np.array([obj.proportions[k]/np.sum(labels==k) for k in labels])
    obj.k_ref=obj.kernel(obj.reference,obj.reference);obj.reference_constant=float(obj.patient_weights@obj.k_ref@obj.patient_weights)
    return obj,descriptors,labels

def replay_figures(design,temporal,old,review_objective,patient,patient_labels):
    base_report=json.loads((OUT/'baseline_replay_scores.json').read_text())
    reports=[base_report]
    if (OUT/'replay_scores.json').exists():reports.append(json.loads((OUT/'replay_scores.json').read_text()))
    records=[r for report in reports for r in report['candidates']]
    summ=[];scores=[];eventrows=[];models={}
    for c in records:
        for unit,u in c['units'].items():
            model=model_events(u['worker_path'],engine.repaired_observation);models[(c['candidate_id'],unit)]=model
            labels=old.km.predict(rank_features(model['centroids']))
            review=review_objective.score(model['descriptors'])
            scores.append(dict(candidate_id=c['candidate_id'],unit=unit,N=len(labels),old_loss=u['score']['old']['loss_off'],
                               training_reference_time_loss=u['score']['temporal']['loss'],review_reference_time_loss=review['loss'],
                               review_A=review.get('A'),review_B=review.get('B')))
            eventrows += [dict(candidate_id=c['candidate_id'],unit=unit,mode=int(lab),**e) for lab,e in zip(labels,model['info'])]
            for mode in ['ALL','TA','TB']:
                ii=np.arange(len(labels)) if mode=='ALL' else np.flatnonzero(labels==(1 if mode=='TA' else 0))
                for key in KEYS:summ.append(dict(candidate_id=c['candidate_id'],unit=unit,mode=mode,observable=key,**distribution([model['info'][i][key] for i in ii])))
    for mode in ['ALL','TA','TB']:
        ii=np.arange(len(patient)) if mode=='ALL' else np.flatnonzero(patient_labels==(1 if mode=='TA' else 0))
        for key in KEYS:
            weights=review_objective.patient_weights if mode=='ALL' else None
            summ.append(dict(candidate_id='patient',unit='review',mode=mode,observable=key,**distribution([patient[i]['statistics'][key] for i in ii],weights)))
    writecsv(OUT/'replay_observable_summary.csv',summ);writecsv(OUT/'replay_review_scores.csv',scores);writecsv(OUT/'replay_event_observables.csv',eventrows)
    for ci,c in enumerate(records,1):
        fig,axs=plt.subplots(3,4,figsize=(13,9));fig.subplots_adjust(top=.87,bottom=.17,left=.14,right=.98,wspace=.35,hspace=.45)
        units=sorted(c['units'])
        for ri,mode in enumerate(['ALL','TA','TB']):
            for j,(key,label) in enumerate(zip(KEYS,LABELS)):
                ax=axs[ri,j]
                for yi,unit in enumerate(['review']+units):
                    ss=next(s for s in summ if s['candidate_id']==('patient' if yi==0 else c['candidate_id']) and s['unit']==unit and s['mode']==mode and s['observable']==key)
                    color='black' if yi==0 else COLORS[yi-1]
                    ax.plot([ss['q05'],ss['q95']],[yi,yi],c=color,lw=1);ax.plot([ss['q25'],ss['q75']],[yi,yi],c=color,lw=5)
                    ax.plot(ss['median'],yi,'o',c=color);ax.plot(ss['mean'],yi,'|',c=color,ms=12)
                ax.set(ylim=(2.6,-.6),xlabel=label,yticks=[0,1,2],yticklabels=['Patient review','Network 1 / new noise','Network 2 / new noise'] if j==0 else [])
                if j==0:ax.set_ylabel(mode,fontweight='bold')
                ax.set_xlim(left=0);ax.grid(axis='x',alpha=.15)
        fig.suptitle(f'{readable_name(c["candidate_id"],design)}\nLocal duration and between-contact timing under noise replay',fontsize=13)
        fig.text(.5,.035,'Point: median; tick: mean; thick: IQR; thin: 5-95% event range, not confidence intervals.\nALL patient reference is weighted to the original FIT mode proportions; generated events retain their natural frequencies.\n18 patient events from held-back blocks are a development diagnostic: these waveforms had been viewed previously.',ha='center',fontsize=9)
        save(fig,f'replay_{ci}_observable_distributions','固定工作点或提名候选在同图新噪声下的时间分布，与留后18个患者事件比较。均值、中位数、SD、方差和分位范围均另存表；ALL患者参照按原FIT自然模式比例加权。**关注点**：重演检验和患者留后诊断不构成全新患者或未见数据验证。')
    return records,scores

def movie_payload(path,mode,old):
    path=Path(path);model=model_events(path,engine.repaired_observation)
    labs=old.km.predict(rank_features(model['centroids']))
    ii=np.flatnonzero(labs==mode)
    if not len(ii):return None
    k=int(ii[0]);idx=model['info'][k]['event_id'];e=model['metadata']['events'][idx]
    with np.load(model['worker']['arrays']['path']) as z:
        env=z['contact_envelope'].astype(float);dt=float(z['contact_envelope_dt_ms']);xy=z['contact_xy_mm'];native=z['sheet_activity_counts'];ndt=float(z['sheet_activity_frame_ms'])
    lo,hi=np.rint(np.asarray(e['window_ms'])/dt).astype(int)
    a=np.maximum(env[:,lo:hi]-np.asarray(e['local_baseline'])[:,None],0);mask=np.isfinite(model['centroids'][k]);a[~mask]=0
    t=(np.arange(lo,hi)+.5)*dt;mu=np.divide(a@t,a.sum(1),out=np.full(len(mask),np.nan),where=mask);zero=np.nanmin(mu)
    return dict(a=a,time=t-zero,zero=zero,mask=mask,xy=xy,native=native,native_dt=ndt,event_id=idx)

def render_movies(design,old):
    nomination=json.loads((OUT/'nomination.json').read_text())['nominees']
    anchor_ids=[c['candidate_id'] for c in design['anchors']];clips=[]
    reports=[json.loads((OUT/f'{p}_scores.json').read_text()) for p in ['baseline_train','A','B','baseline_replay']]
    if (OUT/'replay_scores.json').exists():reports.append(json.loads((OUT/'replay_scores.json').read_text()))
    paths={}
    for report in reports:
        for c in report['candidates']:
            paths.setdefault(c['candidate_id'],{}).update({u:d['worker_path'] for u,d in c['units'].items()})
    for ni,nom in enumerate(nomination,1):
        cid=nom['candidate_id'];ai=anchor_ids.index(cid) if cid in anchor_ids else int(cid.split('_')[1].replace('anchor',''))-1
        baseline_id=anchor_ids[ai]
        for mode in [1,0]:
            patient_entries=[e for e in design['patient_review'] if e['mode']==mode]
            fig,axs=plt.subplots(2,4,figsize=(13,6),gridspec_kw={'height_ratios':[1.4,1]});fig.subplots_adjust(top=.79,bottom=.16,left=.06,right=.96,wspace=.28,hspace=.40)
            allframes=[];shapes=[];heat=[];cursor=[];nat=None;dots=[]
            tgrid=np.arange(-140,261,2.)
            title=fig.suptitle('',fontsize=13)
            prepared=[]
            for ui,unit in enumerate(sorted(paths[cid])):
                if unit not in paths[baseline_id]:continue
                new=movie_payload(paths[cid][unit],mode,old)
                baseline=movie_payload(paths[baseline_id][unit],mode,old)
                if new is None or baseline is None:
                    clips.append(dict(nominee=cid,mode=mode,unit=unit,status='NO_EVENT_OF_THIS_LABEL',not_mechanistic_absence=True));continue
                pe=patient_entries[ui%len(patient_entries)]
                with np.load(Path(paths[cid][unit]).with_suffix('.npz')) as z:names=z['contact_names'].astype(str).tolist()
                with np.load(pe['arrays_path']) as z:
                    raw=z['contact_names'].astype(str).tolist();order=[raw.index(n) for n in names];m=z['packed_window_mask'].astype(bool)
                    a=z['positive_envelope_mass'][order][:,m].astype(float);t=z['time_ms'][m];mask=z['participation_mask'][order].astype(bool)
                a[~mask]=0;mu=np.divide(a@t,a.sum(1),out=np.full(len(mask),np.nan),where=mask);patient=dict(a=a,time=t-np.nanmin(mu),mask=mask)
                prepared.append((ui,unit,new,baseline,pe,patient,names))
            for pi,(ui,unit,new,baseline,pe,patient,names) in enumerate(prepared):
                xy=new['xy'];display=[patient,baseline,new]
                order=sorted(range(len(names)),key=lambda i:(0 if names[i].startswith('SCL') else 1,-int(''.join(filter(str.isdigit,names[i])))))
                rel=[np.divide(d['a'],d['a'].max(1)[:,None],out=np.zeros_like(d['a']),where=d['a'].max(1)[:,None]>0) for d in display]
                if pi==0:
                    for j in range(4):
                        ax=axs[0,j];ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)' if j==0 else '',xticks=[0,10,20],yticks=[0,10,20])
                        ax.set_title(['Patient review','Starting working point','Pilot nominee','Nominee: native neurons'][j],fontsize=10)
                        ax.set_facecolor('black')
                        for shaft in ['SCL','ICL']:
                            ix=[i for i,n in enumerate(names) if n.startswith(shaft)];ix=sorted(ix,key=lambda i:xy[i,0]);ax.plot(xy[ix,0],xy[ix,1],c='gray',lw=.8)
                        if j<3:dots.append(ax.scatter(xy[:,0],xy[:,1],c=np.zeros(15),s=80,cmap='magma',norm=PowerNorm(.5,0,1),edgecolors='white',lw=.5))
                        else:nat=ax.imshow(np.zeros((20,20)),origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0,vmax=1,interpolation='nearest')
                        if j<3:
                            heat.append(axs[1,j].imshow(np.zeros((15,len(tgrid))),origin='upper',aspect='auto',extent=(-141,261,14.5,-.5),cmap='magma',norm=PowerNorm(.5,0,1)))
                            axs[1,j].set(yticks=range(15),yticklabels=[names[i] for i in order] if j==0 else [],xlabel='Time (ms)');axs[1,j].tick_params(axis='y',labelsize=7)
                            cursor.append(axs[1,j].axvline(0,c='cyan',lw=1))
                        else:axs[1,j].axis('off')
                    fig.text(.5,.025,'Actual simulations: no temporal broadening or rescaling. Contact colors: envelope / own whole-window peak (0-1), fixed over time.\nNative field: unsmoothed 1 mm neural bins, own event color scale. Patient/model examples are label-organized, not matched event pairs.',ha='center',fontsize=9)
                for j,d in enumerate(display):heat[j].set_data(np.array([np.interp(tgrid,d['time'],v,left=0,right=0) for v in rel[j]])[order])
                ni0=max(0,int((new['zero']-140)/new['native_dt']));ni1=min(len(new['native']),int((new['zero']+260)/new['native_dt'])+1)
                segment=new['native'][ni0:ni1];cap=max(1.,float(np.quantile(segment[segment>0],.99))) if np.any(segment>0) else 1.
                nat.set_clim(0,cap)
                axs[1,3].clear();axs[1,3].axis('off');axs[1,3].text(0,1,f'Native color range: 0-{cap:.0f}\nspikes / 1 mm bin / 2 ms\n\n{unit}\nPatient event {pe["raw_global_event_index"]}\nAnchor event {baseline["event_id"]}\nNominee event {new["event_id"]}',va='top',fontsize=9)
                for t in range(-140,261,10):
                    title.set_text(f'{readable_name(cid,design)} | {"TA" if mode else "TB"} | example {pi+1}/{len(prepared)} | {t:+d} ms')
                    for j,d in enumerate(display):
                        v=np.array([np.interp(t,d['time'],a,left=0,right=0) for a in rel[j]])
                        dots[j].set_array(np.ma.array(v,mask=~d['mask']));dots[j].cmap.set_bad('#888888');cursor[j].set_xdata([t,t])
                    idx=int((new['zero']+t)/new['native_dt']);nat.set_data(new['native'][idx] if 0<=idx<len(new['native']) else np.zeros((20,20)))
                    fig.canvas.draw();im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()).convert('P',palette=Image.Palette.ADAPTIVE,colors=128);allframes.append(im)
                    if pi==0 and t==0:
                        for ext in ['png','pdf']:fig.savefig(F/f'nominee{ni}_{"ta" if mode else "tb"}_keyframe.{ext}',dpi=155,bbox_inches='tight')
                clips.append(dict(nominee=cid,mode=mode,unit=unit,status='DISPLAYED',patient_id=pe['raw_global_event_index'],anchor_event_id=baseline['event_id'],nominee_event_id=new['event_id']))
            name=f'nominee{ni}_{"ta" if mode else "tb"}_multiple_events'
            if allframes:
                allframes[0].save(F/f'{name}.gif',save_all=True,append_images=allframes[1:],duration=110,loop=0,optimize=False)
                entries.append((name+'.gif','逐单元选该标签的第一个事件，比较患者、原工作点与提名候选；第四列显示候选未经平滑的原生神经场。保留时间热图和不同噪声重演，无人工展宽或时间压缩。**关注点**：患者与模型不是一一配对事件，标签不证明路径恢复；原生场与接触点色标分开解释。'))
                entries.append((f'nominee{ni}_{"ta" if mode else "tb"}_keyframe.png / .pdf','对应多事件GIF的第一事件零时刻索引图，保持相同SEEG布局。接触点显示局部相对包络，原生场显示每2ms的神经元脉冲数。**关注点**：不要混读两种幅值单位。'))
            plt.close(fig)
    (OUT/'movie_event_manifest.json').write_text(json.dumps(clips,indent=2)+'\n')

def main(final=False):
    design,temporal,old=get_frozen();initial_figures(design,temporal,old)
    reports=[json.loads((OUT/'baseline_train_scores.json').read_text())]
    for phase in ['A','B']:
        if (OUT/f'{phase}_scores.json').exists():reports.append(json.loads((OUT/f'{phase}_scores.json').read_text()))
    records=parameter_figures(design,reports)
    if final:
        if not (OUT/'nomination.json').exists():raise RuntimeError('review data cannot open before nomination')
        review,patient,labels=load_review(design,temporal)
        rr,ss=replay_figures(design,temporal,old,review,patient,labels)
        render_movies(design,old)
        nominees=json.loads((OUT/'nomination.json').read_text())['nominees']
        ranked=sorted([r for r in records if r['ranking_eligible']],key=lambda r:r['loss'])
        text='# 新时间分布损失的局部物理pilot\n\n'
        text+=f'本轮已完成固定预算的A/B两批局部搜索与提名重演。训练最低新总损失为{ranked[0]["loss"]:.4f}，对应`{ranked[0]["candidate_id"]}`。该排序只说明开发目标上的结果，不能单独证明患者传播恢复。\n\n'
        text+='## 训练排序与重演\n\n|候选|新总损失|状态|\n|---|---:|---|\n'
        text+=''.join(f'|{r["candidate_id"]}|{r["loss"]:.4f}|可评分|\n' for r in ranked)
        text+='\n不可评分、runaway及其事件数完整保留在阶段表，不制造数值loss。\n\n'
        text+='## 与各自起点比较：留后患者块上的噪声重演\n\n|提名条件|新时间距离改善的重演单元|两单元平均距离变化|\n|---|---:|---:|\n'
        anchor_ids=[a['candidate_id'] for a in design['anchors']]
        changes=[]
        for nom in nominees:
            cid=nom['candidate_id'];ai=anchor_ids.index(cid) if cid in anchor_ids else int(cid.split('_')[1].replace('anchor',''))-1
            paired=[]
            for unit in ['topo_6101_dyn_7102','topo_6102_dyn_7102']:
                start=next((s for s in ss if s['candidate_id']==anchor_ids[ai] and s['unit']==unit),None)
                finish=next((s for s in ss if s['candidate_id']==cid and s['unit']==unit),None)
                if start and finish and start['review_reference_time_loss'] is not None and finish['review_reference_time_loss'] is not None:
                    paired.append(finish['review_reference_time_loss']-start['review_reference_time_loss'])
            delta=float(np.mean(paired)) if paired else None
            changes.append(dict(candidate_id=cid,anchor_id=anchor_ids[ai],estimable_pairs=len(paired),improved_pairs=int(sum(x<0 for x in paired)),mean_time_loss_change=delta))
            text+=f'|{cid}|{sum(x<0 for x in paired)}/{len(paired)}|{delta if delta is not None else "不可估计"}|\n'
        (OUT/'paired_replay_changes.json').write_text(json.dumps(changes,indent=2)+'\n')
        text+='\n负变化表示相对自身原工作点改善；两单元结果只是方向性诊断，不是显著性检验。训练更好而这里不改善，说明选择收益未在该重演/患者块比较中保留；不能继续称为恢复。若提名仍是旧起点，则本预算内未找到更优可评分邻域条件。\n\n'
        text+='## 解释边界\n\n新时间项比较各接触点t50相对位置、上升/下降质量时间和质心相对t50的联合分布，保留参与mask。原有参与、rank、质心lag、空间与模式分量仍在。所有模型读出均来自真实模拟，未使用上一轮离线展宽或缩时差作为输出。\n\n'
        text+='患者46训练事件与18留后诊断事件按记录块分开，仍是此前见过的开发波形包；自然模式比例继承完整FIT事件表，模型事件不重配比。每条件两张拓扑在同一噪声种子下训练，提名后换动力学噪声；两张图不足以证明全局稳健机制。\n\n'
        text+='本轮同时改变阈值和多个连接/抑制参数，散点关系不能唯一归因于某一个参数；GABA改变不是剂量匹配。原生场和多事件GIF只负责检验残差，未回流选择。局部宽度、时间跨度、重叠及形状差异应同时看均值与散布；不能因总loss下降接受机制恢复。\n\n'
        text+='## 必须审阅的产物\n\n`physical_parameters_and_observables`显示实际参数与六项观测；`replay_*_observable_distributions`保留均值、中位数、IQR、5–95%事件范围；`replay_observable_summary.csv`含SD和方差。`nominee*_multiple_events.gif`与原生场列用于核对实际传播，无法显示某标签时在movie_event_manifest中报告支持不足，不自动判为机制缺失。\n\n'
        text+='本轮结束后停在科学审阅点；不自动扩大搜索、冻结网络或进入Fig.5。\n'
        notice=OUT/'native_mechanism_review_notice.md'
        if notice.exists():
            text+='\n## 原生传播机制的补充审阅\n\n'+notice.read_text()+'\n'
        (OUT/'scientific_review.md').write_text(text)
    else:
        (OUT/'prepared_analysis.md').write_text('# Pilot已准备并启动\n\n当前图只含目标性质检查、旧输出离线对照和三个起点的重评分，不能作为新物理结果。新目标和预算见design.json；实际运行状态见status.json。完整参数关系图、留后时间分布和多事件原生场GIF将在两批搜索和提名重演结束后自动生成。\n')
    (F/'README.md').write_text('# 接触点时间与局部形状pilot图件\n\n'+'\n\n'.join(f'### {name}\n\n{description}' for name,description in entries)+'\n')
    validation=[]
    for path in sorted(F.iterdir()):
        if path.suffix in ['.png','.gif']:
            with Image.open(path) as im:
                for i in range(getattr(im,'n_frames',1)):im.seek(i);im.convert('RGB').load()
                validation.append(dict(path=str(path),frames=getattr(im,'n_frames',1),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        elif path.suffix=='.pdf':
            subprocess.run(['pdfinfo',str(path)],check=True,capture_output=True)
            validation.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (OUT/'figure_verification.json').write_text(json.dumps(dict(stage='final' if final else 'prepared',producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),files=validation,human_visual_acceptance='PENDING'),indent=2)+'\n')
    print('Analysis delivery complete:', 'final' if final else 'prepared',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args();main(args.final)
