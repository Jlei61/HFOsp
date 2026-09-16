"""Compare within-contact envelope widths, retaining events as the summary unit."""
from pathlib import Path
import sys,json,pickle,csv
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.topic4_interictal_repaired_evaluation import rank_features
R=ROOT/'results/topic4_sef_hfo/multievent_distribution_search_v2_1';OUT=R/'timing_capacity_diagnostics';F=OUT/'figures'
with (ROOT/'results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2/evaluator.pkl').open('rb') as f:ev=pickle.load(f)
cs=sorted(json.loads((R/'g3_scores.json').read_text())['candidates'],key=lambda c:c['score']['loss_off'])[:2]
packet=json.loads((R/'patient_time_packet/packet_manifest.json').read_text());rows=[];byevent=[];data={}
def widths(mass,time,mask):
    ww=np.full(len(mask),np.nan)
    for c in np.flatnonzero(mask):
        a=np.maximum(mass[c],0);total=a.sum()
        if total<=0:continue
        q=np.searchsorted(np.cumsum(a)/total,[.1,.9]);q=np.minimum(q,len(time)-1);ww[c]=time[q[1]]-time[q[0]]
    return ww
def record(source,event_id,mode,names,w):
    for name,v in zip(names,w):
        if np.isfinite(v):rows.append(dict(source=source,event_id=event_id,mode=mode,contact=name,width_ms=float(v)))
    median=float(np.nanmedian(w));byevent.append(dict(source=source,event_id=event_id,mode=mode,n_contacts=int(np.isfinite(w).sum()),median_contact_width_ms=median))
    data.setdefault((source,mode),[]).append(median)
for e in packet['readable_events']:
    with np.load(e['arrays_path']) as z:
        m=z['packed_window_mask'].astype(bool);w=widths(z['positive_envelope_mass'][:,m],z['time_ms'][m],z['participation_mask'].astype(bool));names=z['contact_names'].astype(str).tolist()
    record('patient_packet',e['raw_global_event_index'],'TA' if e['mode']==1 else 'TB',names,w)
for c in cs:
    for unit,u in sorted(c['units'].items()):
        wp=Path(u['worker_path']);op=wp.parent.parent/'repaired_observation'/wp.with_suffix('.npz').name;meta=json.loads(op.with_suffix('.json').read_text())
        with np.load(op) as z:primary=z['primary_event_indices'];times=np.asarray(z['centroid_ms'][primary],float)
        ll=ev.km.predict(rank_features(times))
        with np.load(wp.with_suffix('.npz')) as z:env=z['contact_envelope'];dt=float(z['contact_envelope_dt_ms']);names=z['contact_names'].astype(str).tolist()
        for local,idx in enumerate(primary):
            e=meta['events'][int(idx)];lo,hi=np.rint(np.array(e['window_ms'])/dt).astype(int);a=np.maximum(env[:,lo:hi]-np.array(e['local_baseline'])[:,None],0);w=widths(a,(np.arange(lo,hi)+.5)*dt,np.isfinite(times[local]))
            record(c['candidate_id']+'/'+unit,int(idx),'TA' if ll[local]==1 else 'TB',names,w)
stats=[]
for (source,mode),a in data.items():
    a=np.asarray(a);q=np.quantile(a,[.05,.25,.5,.75,.95]);stats.append(dict(source=source,mode=mode,n_events=len(a),mean=float(a.mean()),variance=float(a.var()),median=float(q[2]),q05=float(q[0]),q25=float(q[1]),q75=float(q[3]),q95=float(q[4])))
colors=['#256a9c','#76b7dd','#975426','#d5a173']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(14,7));fig.subplots_adjust(left=.19,right=.985,top=.82,bottom=.18,wspace=.5)
for ax,mode in zip(axs,['TA','TB']):
    sources=['patient_packet']+[c['candidate_id']+'/'+u for c in cs for u in sorted(c['units'])]
    for i,source in enumerate(sources):
        s=next(r for r in stats if r['source']==source and r['mode']==mode);col='black' if i==0 else colors[(i-1)%4]
        ax.plot([s['q05'],s['q95']],[i,i],color=col,lw=1);ax.plot([s['q25'],s['q75']],[i,i],color=col,lw=5);ax.plot(s['median'],i,'o',color=col,ms=5)
    ax.set(title=mode,xlabel='Median within-contact 10-90% width per event (ms)',ylim=(8.6,-.6),yticks=range(9),yticklabels=['Patient n=32']+[f'C{ci} network{net}/noise{noise}' for ci in [1,2] for net in [1,2] for noise in [1,2]]);ax.grid(axis='x',alpha=.15);ax.axhline(4.5,color='gray',ls=':',lw=.7)
fig.suptitle('Individual contact envelopes: separate local duration from between-contact timing\nC1: best joint candidate; C2: reference placement A',fontsize=14)
fig.text(.5,.025,'Each point in the distribution is one event: median width across its participating contacts. Dot: median; thick: IQR; thin: 5-95%.\nPatient: positive robust-z envelope; model: positive baseline-subtracted firing density. Same saved windows; these are different signal observers.',ha='center',fontsize=9)
for ext in ['png','pdf']:fig.savefig(F/('within_contact_envelope_widths.'+ext),dpi=160,bbox_inches='tight')
plt.close(fig)
for filename,r in [('within_contact_widths.csv',rows),('event_median_contact_widths.csv',byevent),('within_contact_width_summary.csv',stats)]:
    with (OUT/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
readme=F/'README.md';entry='### within_contact_envelope_widths.png / .pdf\n\n先计算每个参与接触点包络累计质量10–90%宽度，再取该事件各参与点的中位数，以事件为分布单位。比较冻结患者64事件与最佳/参考候选四次运行，接触点明细和事件中位数单独保存。\n\n**关注点**：没有把所有接触点冒充独立事件；患者为正robust-z包络，模型为局部基线扣除的正放电密度代理，宽度差异仍不能单独归因于生物机制。\n'
text=readme.read_text();text=text.split('### within_contact_envelope_widths.png')[0].rstrip();readme.write_text(text+'\n\n'+entry)
print('DONE',json.dumps(stats),flush=True)
