#!/usr/bin/env python3
"""Fixed-observer, per-network analysis of the isolated local-width pilot."""
from pathlib import Path
import csv,json,pickle,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scripts.run_topic4_local_width_mechanism_pilot import OUT,SOURCE,ARMS,PAIRS,write
from scripts import run_topic4_multievent_distribution_v2_1 as engine
from scripts import run_topic4_xy_research as base
from src.topic4_contact_event_objective_v2 import worker_events,patient_event
from scripts.run_topic4_contact_native_integrated_pilot import distribution
F=OUT/'figures'
LABELS={'baseline':'Original (3.5 ms)','recurrent_7ms':'Internal excitation: 7 ms','recurrent_14ms':'Internal excitation: 14 ms','external_7ms':'External excitation: 7 ms'}
COLORS=dict(zip(ARMS,['#555555','#367aa5','#b74a55','#bc902e']))
METRICS=['local_width_ms','recruitment_span_ms','centroid_span_ms','n_contacts']
MLABELS=['Local width (ms)','Recruitment span (ms)','Centroid span (ms)','Participating contacts']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def csvwrite(path,rows):
    if rows:
        with open(path,'w') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def save(fig,name,description,entries):
    for ext in ['png','pdf']:fig.savefig(F/f'{name}.{ext}',dpi=155,bbox_inches='tight')
    plt.close(fig);entries.append((name,description))

def score(path,obj):
    _,_,op=engine.repaired_observation(path)
    table,details,names,w=worker_events(path)
    if names!=obj.names:raise RuntimeError('contact names/order mismatch')
    events=[]
    if table is not None:
        for i,d in enumerate(details):
            mask=table['participation'][i]>0
            events.append(dict(**d,start_ms=op['events'][d['event_id']]['window_ms'][0],
                 n_contacts=int(mask.sum()),centroid_span_ms=float(np.ptp(table['centroid'][i,mask])),
                 recruitment_span_ms=float(np.ptp(table['recruitment'][i,mask,0]))))
    return dict(worker_path=str(path),physical_status=w['physical_status'],observation=obj.score(table,w['physical_status']),
          events=events,n_detected=op['n_detected_windows'],primary_fraction=len(events)/max(1,op['n_detected_windows']),
          distributions={k:distribution([e[k] for e in events]) for k in METRICS},static=w['static_array_identity']),op

def current_figures(arm,topo,row,op,entries):
    path=Path(row['worker_path']);tracepath=path.with_name(path.stem+'_local_currents.npz')
    with np.load(tracepath) as z:
        t=z['time_ms'];curr=z['currents'];rate=z['spike_rate_hz'];names=z['contact_names'].astype(str);xy=z['contact_xy_mm']
    with np.load(path.with_suffix('.npz')) as z:
        env=z['contact_envelope'];edt=float(z['contact_envelope_dt_ms']);movie=z['sheet_activity_counts'];mdt=float(z['sheet_activity_frame_ms'])
    selected=row['events'][:3]
    if not selected:
        selected=[dict(event_id=i,start_ms=e['window_ms'][0]) for i,e in enumerate(op['events']) if e['window_ms'][0]>=500][:3]
    if not selected:return
    fig,axs=plt.subplots(len(selected),3,figsize=(12,3.1*len(selected)),squeeze=False)
    picks=[]
    for j,e in enumerate(selected):
        lo,hi=op['events'][e['event_id']]['window_ms'];a,b=np.rint(np.array([lo,hi])/edt).astype(int)
        mass=env[:,a:b];ci=int(np.argmax(mass.sum(1)));peak=(a+int(np.argmax(mass[ci])))*edt
        keep=(t>=lo)&(t<hi);x=t[keep]-peak
        for k,label,col in [(2,'Internal E','#ad4e52'),(3,'External E','#b69332'),(1,'GABA','#3c6b9c')]:
            axs[j,0].plot(x,curr[keep,k,ci],color=col,label=label,lw=1.2)
        axs[j,0].set(xlabel='Time from local envelope peak (ms)',ylabel='Current drive (model units)',title=f'Event {e["event_id"]}; contact {names[ci]}')
        for k,label,col in [(0,'Local E spikes','#ad4e52'),(1,'Local I spikes','#3c6b9c')]:
            nr=min(len(t),len(rate));rr=(t[:nr]>=lo)&(t[:nr]<hi)
            axs[j,1].plot(t[:nr][rr]+.5-peak,rate[:nr][rr,k,ci],label=label,color=col,lw=.8)
        axs[j,1].set(xlabel='Time (ms)',ylabel='Weighted local firing rate (Hz)')
        # Native full-sheet snapshot at the chosen contact peak; no lineage mask.
        fr=int(np.clip(round(peak/mdt),0,len(movie)-1));img=movie[fr]
        axs[j,2].imshow(img,origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0,vmax=max(1,float(np.max(img))));axs[j,2].scatter(xy[:,0],xy[:,1],s=14,facecolors='none',edgecolors='cyan')
        axs[j,2].scatter(*xy[ci],s=50,facecolors='none',edgecolors='white');axs[j,2].set(xlabel='x (mm)',ylabel='y (mm)',title='All native activity at local peak',aspect='equal')
        picks.append(dict(event_id=e['event_id'],contact=str(names[ci]),peak_ms=peak,primary=e in row['events']))
    axs[0,0].legend(fontsize=8);axs[0,1].legend(fontsize=8)
    fig.suptitle(f'{LABELS[arm]} | network {topo} | first events, highest-mass contact',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95));name=f'local_currents_{arm}_topo_{topo}'
    save(fig,name,'按时间取最早三个 primary 事件；若无 primary，则展示已检测窗口并在索引标明。每事件固定选包络总质量最大的接触点，展示局部兴奋/抑制电流、原始 1 ms 发放率，以及该接触点峰时的全场活动。电流和发放是模型量，不是患者 HFO。**关注点**：先后关系是诊断证据，不能仅凭时序宣称因果；并行热点仍需看全场。',entries)
    write(OUT/f'{name}_selection.json',picks)

def main():
    F.mkdir(parents=True,exist_ok=True)
    obj=pickle.load(open(SOURCE/'objective.pkl','rb'));rows=[];csvrows=[];all_events=[];entries=[];audits=[]
    reference=next(r for r in base.read(SOURCE/'confirmation_scores.json')['candidates'] if r['candidate_id']=='tshape_anchor3_B_minus')
    for arm in ARMS:
        for topo,dyn in PAIRS:
            path=OUT/'execution/workers'/f'{arm}_topo_{topo}_dyn_{dyn}.json'
            if not path.exists():continue
            row,op=score(path,obj);row.update(arm=arm,topology_seed=topo,dynamics_seed=dyn);rows.append(row)
            audit=base.read(path.with_name(path.stem+'_kinetics.json'))
            old=reference['units'][f'topo_{topo}_dyn_{dyn}'];same_static=row['static']==old['static_array_identity']
            rec=dict(arm=arm,topology_seed=topo,same_static_as_historical_anchor=same_static,
                 poisson_prefix=audit['poisson_prefix_sha256'],actual_base_AMPA_ms=audit['base_AMPA_decay_ms'])
            if arm=='baseline':
                with np.load(path.with_suffix('.npz')) as z,np.load(Path(old['worker_path']).with_suffix('.npz')) as o:
                    rec['baseline_full_contact_byte_equal']=np.array_equal(z['contact_envelope'],o['contact_envelope'])
                    rec['baseline_native_movie_byte_equal']=np.array_equal(z['sheet_activity_counts'],o['sheet_activity_counts'])
                if not rec['baseline_full_contact_byte_equal'] or not rec['baseline_native_movie_byte_equal']:raise RuntimeError('full-size baseline observer replay differs from historical output')
            if not same_static:raise RuntimeError('static identity changed across kinetics intervention')
            audits.append(rec)
            for k,v in row['distributions'].items():csvrows.append(dict(arm=arm,topology_seed=topo,metric=k,**v))
            for e in row['events']:all_events.append(dict(arm=arm,topology_seed=topo,**e))
            current_figures(arm,topo,row,op,entries)
    if len(rows)!=8:raise RuntimeError(f'expected eight complete units, found {len(rows)}')
    for topo,_ in PAIRS:
        if len({r['poisson_prefix'] for r in audits if r['topology_seed']==topo})!=1:raise RuntimeError('initial Poisson innovations do not match across interventions')
    write(OUT/'scores.json',dict(units=rows));write(OUT/'execution_audit.json',audits)
    csvwrite(OUT/'observable_distributions.csv',csvrows);csvwrite(OUT/'events.csv',all_events)
    fig,axs=plt.subplots(2,4,figsize=(14,6),squeeze=False)
    for i,(topo,_) in enumerate(PAIRS):
        for j,k in enumerate(METRICS):
            ax=axs[i,j]
            for r in rows:
                if r['topology_seed']!=topo:continue
                vals=np.sort([e[k] for e in r['events']])
                if len(vals):ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',color=COLORS[r['arm']],label=f'{LABELS[r["arm"]]} (n={len(vals)})')
            ax.set(xlabel=MLABELS[j],ylabel=f'Network {topo}\nFraction of events',ylim=(0,1.02));ax.grid(alpha=.15)
            if j==0:
                ax.axvspan(52.0872,119.3506,color='#aaaaaa',alpha=.16,label='Patient TRAIN: 5-95%')
                ax.axvline(85.0219,color='black',ls=':',lw=1,label='Patient TRAIN median')
        axs[i,0].legend(fontsize=6.5,loc='lower right')
    fig.suptitle('Internal versus external excitation duration | fixed contact readout',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.95))
    save(fig,'duration_parameter_response','每张网络分别比较四组干预的事件分布，不混合网络。灰带和虚线仅为既有患者 46 个 TRAIN 波形包的加权局部宽度参考，不代表全部患者事件；事件数写在图例。**关注点**：宽度增加是否伴随招募跨度、参与或事件量的异常变化；无事件条件不画成零误差。',entries)
    fig,axs=plt.subplots(1,6,figsize=(15,4));keys=['participation','centroid_structure','local_shape','recruitment','joint_envelope']
    # Weights comprise the five frozen terms; preserve their scale and negative values.
    for j,key in enumerate(keys):
        for i,arm in enumerate(ARMS):
            for topo,marker,off in [(6101,'o',-.08),(6102,'^',.08)]:
                r=next(v for v in rows if v['arm']==arm and v['topology_seed']==topo)
                val=r['observation'].get('blocks',{}).get(key,{}).get('D_off')
                if val is not None:axs[j].scatter(i+off,val,color=COLORS[arm],marker=marker)
        axs[j].set(title=key.replace('_',' '),xticks=range(4),xticklabels=['Base','Internal 7','Internal 14','External 7']);axs[j].tick_params(axis='x',labelrotation=45)
    for i,arm in enumerate(ARMS):
        for topo,marker,off in [(6101,'o',-.08),(6102,'^',.08)]:
            r=next(v for v in rows if v['arm']==arm and v['topology_seed']==topo)
            axs[5].scatter(i+off,len(r['events']),marker=marker,color=COLORS[arm])
    axs[5].set(title='Primary event count',xticks=range(4),xticklabels=['Base','Internal 7','Internal 14','External 7']);axs[5].tick_params(axis='x',labelrotation=45)
    fig.tight_layout();save(fig,'separate_errors_and_event_count','冻结的五项观测误差与事件量分别展示。圆点和三角分别是两张拓扑；事件不足的误差保持缺失。**关注点**：局部宽度不能通过丢掉事件或破坏其他观测而单独宣布成功。',entries)
    md='# 局部宽度物理实验：完成后审阅\n\n每条件两张拓扑、相同噪声种子；每单元 24 秒或明确物理 runaway。新干预只改变递归或外部 AMPA 电流的衰减，原参数对象保持 3.5 ms，使基准输入率不随干预重算。内部干预同时影响 E→E 与 E→I，不能归因于单独 EE 通路。\n\n|条件|网络|事件数|局部宽度中位数 ms|5–95% ms|招募跨度中位数 ms|物理状态|\n|---|---|---:|---:|---|---:|---|\n'
    for r in rows:
        q=r['distributions']['local_width_ms'];s=r['distributions']['recruitment_span_ms']
        md+=f'|{LABELS[r["arm"]]}|{r["topology_seed"]}|{q["n"]}|{q["median"]}|{q["q05"]}–{q["q95"]}|{s["median"]}|{r["physical_status"]}|\n'
    md+='\n判断原则：内部干预延长宽度而外部对照不延长，更支持局部递归时间尺度的限制；若只有外部干预有效，更支持输入/观测表征问题；若均无明显延长或导致 runaway，则不支持直接通过本次时间尺度调整解决缺口。少数事件、窗口筛选和并行热点仍是替代解释。\n\n已检验同拓扑静态身份、前 100 ms Poisson 输入创新，以及基线完整包络和全场活动对历史输出的逐元素一致性。新数据尚需结合逐事件电流图进行科学和人工目视审阅；没有自动扩搜或冻结模型。\n'
    (OUT/'scientific_review.md').write_text(md)
    (F/'README.md').write_text('\n\n'.join('### '+n+'.png / .pdf\n\n'+desc for n,desc in entries)+'\n')
    validation=[]
    for p in F.glob('*.png'):
        with Image.open(p) as im:im.load();validation.append(dict(file=str(p),size=list(im.size),sha256=base.sha(p)))
    write(OUT/'figure_validation.json',dict(png_decoded=validation,human_review='PENDING'))

if __name__=='__main__':main()
