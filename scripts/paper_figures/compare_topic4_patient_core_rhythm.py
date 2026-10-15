#!/usr/bin/env python3
"""Frozen-output E10 event timing / native core rhythm comparison and paired sweep."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
MAIN=Path('/home/honglab/leijiaxin/HFOsp')
from src.epilepsiae_dataset import _parse_sql_subject
from scripts.paper_figures.audit_topic4_core_burst_rhythm import metrics,regions,LABELS
OUT=ROOT/'results/topic4_sef_hfo/patient_model_rhythm_comparison'
F=OUT/'figures';P=ROOT/'results/topic4_sef_hfo/contact_native_integrated_pilot'
FS=100;T=23.5;ORDER=list(LABELS)
NAMES=['Original working point','New batch: placement 1','Lower hotspot penalty: placement 2','Wider activity: placement 3']
COLORS=['#6b4276','#bf616a','#d99540']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvwrite(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def spectrum(times,duration=T):
    a=np.zeros(round(duration*FS));ix=np.floor(np.asarray(times)*FS).astype(int)
    ix=ix[(ix>=0)&(ix<len(a))];np.add.at(a,ix,1)
    ff,pp=welch(a,fs=FS,nperseg=400,noverlap=200)
    return ff,pp
def norm(p,f):
    mask=(f>=.5)&(f<=10);return p/np.trapz(p[mask],f[mask]) if p[mask].sum()>0 else p
def interval_metrics(times):
    d=np.diff(times);d=d[d>0]
    return dict(n_events=len(times),interval_median_s=float(np.median(d)) if len(d) else None,
                interval_cv=float(d.std()/d.mean()) if len(d)>1 else None,
                lag1_log_interval_r=float(np.corrcoef(np.log(d[:-1]),np.log(d[1:]))[0,1]) if len(d)>=8 else None)
def savefig(fig,name):
    for ext in ['png','pdf']:fig.savefig(F/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

def patient():
    raw=Path('/mnt/epilepsia_data/interilca_inter_results/all_data_lns/1146/all_recs')
    sql=Path('/mnt/epilepsia_data/all_data_sqls/pat_114602_2012-12-20.sql')
    md=_parse_sql_subject(sql)
    blocks={(b['recording_id'],b['block_no']):b for b in md['blocks']}
    rows=[];ps=[];ds=[];sources=[];traces=[];block_stats=[];masked_ps=[];matched_intervals=[]
    for path in sorted(raw.glob('*_lagPat.npz')):
        rid,bn=path.name.removesuffix('_lagPat.npz').split('_');b=blocks[(rid,int(bn))]
        duration=b['samples']/md['recordings'][rid]['sample_rate_sql']
        with np.load(path,allow_pickle=True) as z:
            lag=z['lagPatRaw'];valid=z['eventsBool'].astype(bool)
        packed=np.load(path.with_name(path.name.replace('_lagPat.npz','_packedTimes.npy')))
        if not lag.size or not packed.size:continue
        assert lag.shape[1]==len(packed)
        width=np.mean(packed[:,1]-packed[:,0])
        times=np.sort(packed[:,0]+np.min(lag,axis=0)%width)
        # Exact archived calibration, retained only for reproducing this figure.
        masked=np.sort(packed[:,0]+np.nanmin(np.where(valid,lag,np.nan),axis=0)%width)
        times=times[(times>=0)&(times<duration)];masked=masked[np.isfinite(masked)&(masked>=0)&(masked<duration)]
        ds.extend(np.diff(times).tolist())
        block_stats.append(dict(block=path.stem,n_events=len(times),duration_s=duration,**{k:v for k,v in interval_metrics(times).items() if k!='n_events'}))
        sources.append(dict(path=str(path),packed_path=str(path.with_name(path.name.replace('_lagPat.npz','_packedTimes.npy'))),sql_block_id=b['block_id'],duration_s=duration))
        for j in range(int(duration//T)):
            lo=j*T;tt=times[(times>=lo)&(times<lo+T)]-lo
            mt=masked[(masked>=lo)&(masked<lo+T)]-lo
            ff,p=spectrum(tt);ps.append(p);masked_ps.append(spectrum(mt)[1]);traces.append(tt)
            matched_intervals.extend(np.diff(tt).tolist())
            rows.append(dict(block=path.stem,window=j,start_s=lo,rate_hz=len(tt)/T,**interval_metrics(tt)))
    csvwrite(OUT/'patient_windows.csv',rows);csvwrite(OUT/'patient_blocks.csv',block_stats)
    return dict(f=ff,psd=np.mean(ps,0),masked_psd=np.mean(masked_ps,0),intervals=np.array(ds),matched_intervals=np.array(matched_intervals),rows=rows,traces=traces,sources=sources,blocks=block_stats)

def current():
    data={};rows=[]
    scores=json.loads((P/'confirmation_scores.json').read_text())['candidates']
    for c in scores:
        cid=c['candidate_id'];units=[]
        for uid,u in c['units'].items():
            path=Path(u['worker_path']);w=json.loads(path.read_text())
            op=path.parent.parent/'repaired_observation'/path.name;meta=json.loads(op.read_text())
            with np.load(op.with_suffix('.npz')) as z:idx=z['primary_event_indices']
            times=np.array([meta['events'][int(i)]['qualifying_interval_ms'][0]/1000-.5 for i in idx])
            times=times[(times>=0)&(times<T)];times.sort()
            ff,ps=spectrum(times);core_ps=[];core_times=[];native_ps=[]
            with np.load(w['arrays']['path']) as z:
                _,_,den,pure,_=regions(z,np.asarray(c['candidate']['node_field']['centers_mm']),w['xy_geometry_audit']['distance_cutoff_mm'])
                raw=z['sheet_activity_counts'].astype(float)[250:]
            for k in range(2):
                x=raw[:,pure[k]].sum(1)/den[pure[k]].sum();mm,peaks,f,p,_=metrics(x)
                tt=peaks*.002;core_times.append(tt);core_ps.append(spectrum(tt)[1]);native_ps.append((f,p))
                rows.append(dict(candidate_id=cid,unit=uid,layer=f'core{k+1}',worker_path=str(path),rate_hz=len(tt)/T,**interval_metrics(tt),native_peak_hz=mm['frequency_peak_hz']))
            rows.append(dict(candidate_id=cid,unit=uid,layer='contact_events',worker_path=str(path),rate_hz=len(times)/T,**interval_metrics(times)))
            units.append(dict(times=times,psd=ps,core_times=core_times,core_psd=np.mean(core_ps,0),native_psd=native_ps,unit=uid))
        data[cid]=units
    csvwrite(OUT/'model_confirmation_metrics.csv',rows)
    return data,rows

def paired_sweep():
    dest=OUT/'paired_parameter_core_metrics.csv'
    if dest.exists():
        with dest.open() as f:return list(csv.DictReader(f))
    base=ROOT/'results/topic4_sef_hfo/multidimensional_interictal_pilot_round1/execution/paired_round1'
    cs=json.loads((base/'candidate_manifest.json').read_text())['candidates'];cs={c['candidate_id']:c for c in cs if c['anchor'] in ['historical','old_joint','support_rank']}
    rows=[]
    for i,path in enumerate(sorted((base/'workers').glob('*.json'))):
        w=json.loads(path.read_text());cid=w['candidate_id']
        if cid not in cs:continue
        c=cs[cid];stop=w['simulation']['runaway_early_stop_ms']
        if stop is not None:
            rows.append(dict(candidate_id=cid,anchor=c['anchor'],arm=c['arm'],seed=w['seed'],core='',physical_state='runaway',worker_path=str(path)));continue
        with np.load(w['arrays']['path']) as z:
            _,_,den,pure,_=regions(z,np.asarray(c['node_field']['centers_mm']),w['xy_geometry_audit']['distance_cutoff_mm'])
            raw=z['sheet_activity_counts'].astype(float)[250:]
        for k in range(2):
            x=raw[:,pure[k]].sum(1)/den[pure[k]].sum();m,*_=metrics(x)
            rows.append(dict(candidate_id=cid,anchor=c['anchor'],arm=c['arm'],seed=w['seed'],core=k+1,physical_state='complete',worker_path=str(path),**m))
        if i%20==0:print('paired sweep',i,flush=True)
    csvwrite(dest,rows);return rows

def compare_figure(patient,data,modelrows):
    from matplotlib.lines import Line2D
    from matplotlib.font_manager import FontProperties
    legend_font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size=11)
    legend_handles=[Line2D([],[],c=COLORS[0],lw=2,ls='-'),Line2D([],[],c=COLORS[1],lw=2,ls='--'),Line2D([],[],c=COLORS[2],lw=2,ls='-.')]
    legend_labels=['患者：群体事件','模型：筛选后接触事件','模型：core 原生爆发']
    f=patient['f'];mask=(f>=.5)&(f<=10)
    fig,axs=plt.subplots(4,3,figsize=(15,13),gridspec_kw={'width_ratios':[1.25,1.15,1]})
    patcv=[r['interval_cv'] for r in patient['rows'] if r['n_events']>=8 and r['interval_cv'] is not None]
    for i,cid in enumerate(ORDER):
        u=data[cid];a=axs[i,0]
        a.plot(f[mask],norm(patient['psd'],f)[mask],c=COLORS[0],lw=2,label='Patient group events')
        for key,color,label,style in [('psd',COLORS[1],'Model accepted contact events','--'),('core_psd',COLORS[2],'Model core bursts','-.')]:
            pp=np.array([norm(z[key],f) for z in u]);a.plot(f[mask],pp.mean(0)[mask],c=color,lw=1.8,label=label,ls=style)
            a.fill_between(f[mask],pp.min(0)[mask],pp.max(0)[mask],color=color,alpha=.13)
        a.set(xlabel='Frequency (Hz)',ylabel='Normalized onset PSD',xlim=(.5,10),ylim=(0,.8));a.set_title(NAMES[i],loc='left',fontsize=12)
        a=axs[i,1]
        for tt,color,style in [(patient['matched_intervals'],COLORS[0],'-'),(np.concatenate([np.diff(z['times']) for z in u]),COLORS[1],'--'),(np.concatenate([np.diff(t) for z in u for t in z['core_times']]),COLORS[2],'-.')]:
            tt=np.sort(tt[tt>0]);a.plot(tt,np.arange(1,len(tt)+1)/len(tt),c=color,lw=1.8,ls=style)
        a.set(xscale='log',xlim=(.02,25),ylim=(0,1.02),xlabel='Between-event interval (s)',ylabel='Cumulative fraction');a.axvline(.5,c='gray',lw=.6,ls=':')
        a=axs[i,2];a.boxplot([patcv],positions=[0],widths=.4,showfliers=False,patch_artist=True,boxprops={'facecolor':COLORS[0],'alpha':.22},medianprops={'color':COLORS[0]})
        for x,layer,color in [(1,'contact_events',COLORS[1]),(2,'core',COLORS[2])]:
            rr=[r for r in modelrows if r['candidate_id']==cid and (r['layer']==layer if x==1 else r['layer'].startswith('core'))]
            vals=[r['interval_cv'] for r in rr];a.scatter(x+np.linspace(-.10,.10,len(vals)),vals,c=color,s=38,zorder=3)
        a.set(xticks=[0,1,2],xticklabels=['Patient','Model\ncontact','Model\ncore'],ylabel='Interval CV (SD / mean)',ylim=(0,2.5))
        a.legend(legend_handles,legend_labels,loc='upper right',prop=legend_font,frameon=True,facecolor='white',edgecolor='#dddddd',framealpha=1,handlelength=2.6)
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.963),ncol=3,frameon=False)
    fig.suptitle('E10 patient versus current models: event timing and regularity',fontsize=17,y=.994)
    fig.tight_layout(rect=(0,.05,1,.934))
    fig.text(.5,.013,'Both sides: 23.5 s windows; within-window intervals only. PSD: 10 ms onset bins, 4 s Welch segments; 0.5–10 Hz area = 1.\nPatient: all available archived E10 blocks; CV uses windows with ≥8 events. Model: two new-noise confirmation networks per candidate.\nShading is the range of two model networks. Event detectors differ; this compares timing patterns, not equivalent tissue activity.',ha='center',fontsize=10)
    savefig(fig,'patient_model_event_rhythm')

def sweep_figure(rows):
    families=[('tau_d_GABA_ms_low','tau_d_GABA_ms_high','GABA decay\n12 / 18 / 24 ms'),('E_to_I_weight_scale_low','E_to_I_weight_scale_high','E → I strength\n0.85 / 1 / 1.15'),('I_to_E_weight_scale_low','I_to_E_weight_scale_high','I → E strength\n0.85 / 1 / 1.15'),('E_to_E_weight_scale_low','E_to_E_weight_scale_high','E → E strength\n0.85 / 1 / 1.15'),('vth_low','vth_high','Threshold-field amplitude\n0.7 / 1 / 1.3')]
    cols=['interval_median_ms','interval_cv','peak_active_fraction_median']
    groups={}
    for r in rows:
        if r['physical_state']!='complete':continue
        key=(r['anchor'],int(r['seed']),r['arm']);groups.setdefault(key,[]).append(r)
    means={k:{c:np.mean([float(r[c]) for r in rs]) for c in cols} for k,rs in groups.items()}
    effects=[];fig,axs=plt.subplots(3,5,figsize=(17,9),sharex='col')
    ac={'historical':'#7a8c67','old_joint':'#5285a0','support_rank':'#b7815d'}
    for j,(low,high,title) in enumerate(families):
        for (anchor,seed,arm),baseline in means.items():
            if arm!='baseline':continue
            for ii,col in enumerate(cols):
                yy=[means.get((anchor,seed,a),{}).get(col,np.nan) for a in [low,'baseline',high]]
                axs[ii,j].plot([0,1,2],yy,color=ac[anchor],lw=.8,alpha=.45,marker='.',ms=5)
                for level,y in zip(['low','baseline','high'],yy):
                    if np.isfinite(y):effects.append(dict(parameter=title.replace('\n',' '),anchor=anchor,seed=seed,metric=col,level=level,value=float(y),paired_delta=float(y-baseline[col])))
        axs[0,j].set_title(title,fontsize=11)
        for ii in range(3):axs[ii,j].set(xticks=[0,1,2],xticklabels=['Lower','Baseline','Higher'])
    for i,label in enumerate(['Burst interval\n(median, ms)','Interval variability\n(CV)','Peak active fraction\n(per 2 ms)']):axs[i,0].set_ylabel(label)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([],[],color=c,lw=2) for c in ac.values()],['Manual placement','Reference placement A','Reference placement B'],loc='upper center',bbox_to_anchor=(.5,.96),ncol=3,frameon=False)
    fig.suptitle('Which existing parameter interventions change core rhythm?',fontsize=17,y=.996)
    fig.tight_layout(rect=(0,.055,1,.92))
    fig.text(.5,.013,'One parameter changed at a time; each line is a paired seed at a fixed core placement (two cores averaged).\nHistorical 12 s experiment, 0.5 s burn-in; four paired seeds per placement. Runaway conditions are retained in the status CSV, not plotted as rhythmic states.\nGABA decay changes both temporal shape and integrated inhibitory dose. OU laws and membrane time constants were not varied.',ha='center',fontsize=10)
    savefig(fig,'parameter_rhythm_response');csvwrite(OUT/'paired_parameter_effects.csv',effects)
    return effects

def main():
    F.mkdir(parents=True,exist_ok=True)
    pat=patient();print('patient',len(pat['rows']),'windows',sum(r['n_events'] for r in pat['blocks']),'events',flush=True)
    data,mr=current();compare_figure(pat,data,mr);print('comparison figures complete',flush=True)
    sr=paired_sweep();effects=sweep_figure(sr)
    cache=json.loads((MAIN/'results/event_periodicity/epilepsiae/1146_periodicity.json').read_text())['group']
    summary=dict(status='FROZEN_OUTPUT_COMPARISON_COMPLETE',new_physical_runs=0,patient_event_table='Archived E10 1146 lagPat, for the user-provided historical rhythm figure; not the 30049-event modern qualification table',
      patient_sources=pat['sources'],patient_archived_peak_hz=cache['specparam']['peaks'][0][0],patient_isi_shuffle_p=cache['surrogate_isi']['p_value'],patient_gamma_renewal_p=cache['surrogate_gamma']['p_value'],
      patient_total_events=sum(r['n_events'] for r in pat['blocks']),patient_window_count=len(pat['rows']),patient_empty_window_fraction=np.mean([r['n_events']==0 for r in pat['rows']]),patient_cv_eligible_windows=sum(r['n_events']>=8 for r in pat['rows']),
      patient_interval_quantiles_s=np.quantile(pat['intervals'],[.05,.5,.95]).tolist(),patient_window_cv_quantiles=np.quantile([r['interval_cv'] for r in pat['rows'] if r['n_events']>=8],[.05,.5,.95]).tolist(),
      patient_matched_window_interval_quantiles_s=np.quantile(pat['matched_intervals'],[.05,.5,.95]).tolist(),
      model_confirmation_units=8,core_sampling='Strict interior bins only',spectral_contract='Onset counts at 100 Hz; Welch 4 s Hann segments, 50% overlap, 23.5 s windows; normalize integral 0.5–10 Hz after averaging patient raw PSD',
      historical_event_calibration='packed start + min(lagPatRaw) modulo mean packed width, as original producer; participant-masked onset sensitivity computed separately',
      masked_onset_psd_max_absolute_difference=float(np.max(np.abs(norm(pat['psd'],pat['f'])-norm(pat['masked_psd'],pat['f'])))),
      limitations=['Patient group events, model contact events, and core bursts have different detection operators','Patient available archived blocks are not a complete continuous all-state patient record','23.5 s model cannot validate multi-minute or multi-hour rate modulation','Historical paired sweep is not a one-parameter causal dissection of the latest jointly optimized candidate','No new noise/feedback intervention; cannot name a unique physiological oscillator'],human_visual_acceptance='PENDING')
    write(OUT/'manifest.json',summary)
    np.savez_compressed(OUT/'patient_common_spectrum.npz',frequency_hz=pat['f'],onset_psd=pat['psd'],participant_masked_onset_psd=pat['masked_psd'])
    for path in F.glob('*.png'):
        with Image.open(path) as im:im.verify()
    print('SUMMARY',{k:v for k,v in summary.items() if k.startswith('patient_') and k!='patient_sources'},flush=True)
    for cid in ORDER:
        print(cid,[{k:r[k] for k in ['unit','layer','rate_hz','interval_median_s','interval_cv']} for r in mr if r['candidate_id']==cid],flush=True)

if __name__=='__main__':main()
