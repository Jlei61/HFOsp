#!/usr/bin/env python3
"""Launch frozen-template cohort preparation with broadband qualification first.

No association uses a seizure label before broadband qualification is resolved.
Phase one saves frequency-resolved diagnostics and provisional source scores.
"""
from pathlib import Path
import sys,json,os,time,traceback,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.epilepsiae_dataset import _parse_sql_subject,_parse_ts,_subject_from_sql_name
from src.interictal_propagation import load_subject_propagation_events
from src.topic5_template_axis_field import scorers_from_interictal_record,interictal_field_quality_tier
from src.topic5_tspectral_field_concordance import distal_baseline_robust_z,aggregate_complete_windows
from scripts.analyze_e1146_source_signed_correspondence import common_scores,classify_source
from scripts.analyze_e1146_preseizure_template_share import write_json,sha,coverage
from scripts.analyze_e1146_event_time_correspondence import contained,outside_seizures
FIELD_ROOT=ROOT/'results/interictal_propagation_masked/template_gradient_fields_all_events_timing_plus_space/per_subject'
OUT=ROOT/'results/topic5_preseizure_template_association_broadband/cohort_20260909'
SQLROOT=Path('/mnt/epilepsia_data/all_data_sqls')
VERSION='clinical_0_10_broadband_5of6_v2'
BANDS={'delta':(1,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,80),'high_frequency':(80,150.0001)}


def get_inventory(sid):
    candidates=[p for p in SQLROOT.glob('pat_*_*.sql') if _subject_from_sql_name(p)==sid]
    if len(candidates)!=1:raise ValueError(f'sql_sources:{len(candidates)}')
    sql=_parse_sql_subject(candidates[0]);truth={str(x['seizure_id']):x for x in sql['seizures']}
    df=pd.read_csv(ROOT/'results/epilepsiae_seizure_inventory.csv',dtype=str)
    df=df[df.subject==sid].copy();df['sortkey']=df.clin_onset_epoch.astype(float);df=df.sort_values('sortkey')
    inv=[]
    for i,(_,r) in enumerate(df.iterrows()):
        s=truth[r.seizure_id];d=dict(sz=i+1,seizure_idx=i,seizure_id=r.seizure_id)
        for key in ('eeg_onset','eeg_offset','clin_onset','clin_offset'):
            value=_parse_ts(s[key]);stored=float(r[key+'_epoch']) if pd.notna(r[key+'_epoch']) else None
            assert (value is None and stored is None) or (value is not None and stored is not None and abs(value-stored)<1e-5), (sid,i,key,value,stored)
            d[key+'_epoch']=value
        d.update(onset=min(v for v in [d['eeg_onset_epoch'],d['clin_onset_epoch']] if v is not None),offset=max(v for v in [d['eeg_offset_epoch'],d['clin_offset_epoch']] if v is not None))
        inv.append(d)
    assert len(inv)==len(truth)
    return sql,inv,candidates[0]


def prepare_manifest():
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for p in sorted(FIELD_ROOT.glob('epilepsiae_*.json')):
        sid=p.stem.split('_')[1];d=json.loads(p.read_text());reason='';available=False
        try:
            sc=scorers_from_interictal_record(d);available='shared_a' in sc and 'shared_b' in sc
            if not available:reason='no_frozen_shared_field_no_own_field_fallback'
        except Exception as e:reason=str(e)
        sql,inv,sp=get_inventory(sid)
        quality=interictal_field_quality_tier(d)
        rows.append(dict(subject=sid,n_seizures=len(inv),n_events=len(d.get('template_discovery',{}).get('event_labels',[])),
            geometry_tier=quality,selected=available,primary_geometry=available and quality in ('strict_2d','non_strict_2d'),
            reason=reason,field=str(p),field_sha256=sha(p),sql=str(sp),sql_sha256=sha(sp)))
    pd.DataFrame(rows).to_csv(OUT/'cohort_manifest.csv',index=False)
    write_json(OUT/'contract.json',dict(version=VERSION,field_root=str(FIELD_ROOT),
        selected_subjects=[r['subject'] for r in rows if r['selected']],n_selected_seizures=sum(r['n_seizures'] for r in rows if r['selected']),
        primary_subjects=[r['subject'] for r in rows if r['primary_geometry']],
        source_label='fixed original shared geometry; common minimum participation support and Gaussian operator; r(energy, -rank)',
        template_naming='TA = more events, TB = fewer events; shared statistical role, not shared anatomy',
        broadband_qualification='Matched clinical 0-10s adaptation of existing Fig3 v3: >=5/6 bands, >=2 low and >=2 fast, q75 spatial trace, 2s smoothing, > baseline q99 sustained >=2s; independent of source correlation',
        baseline='EEG onset [-120,-90] s; report baseline overlap with any clinical/electrographic seizure',
        early_window='clinical onset [0,10] s, complete 1s spectral frames',
        spectral='CAR and existing Epilepsiae notch; 1s PSD /0.5s hop; source activation exact 1-150Hz primary, lower-Nyquist adaptive band diagnostic only',
        qualification_origin='scripts/plot_topic5_early_spectral_phenotypes.py v3 logic; new matched window and raw distal robust-z, not identical frozen phenotype labels',diagnostic_bands=BANDS,line_exclusion_for_band_diagnostics='exclude within +/-2 Hz of 50/100/150Hz; source activation keeps original Fig3 integral',
        diagnostic_selection='display contact with most positive non-delta bands; this is display only, not a qualification or hypothesis test',
        geometry='2D cohort primary; shared 1D E139 diagnostic/sensitivity only; no own-field fallback',
        temporal='actual event start/end containment; no parent-hour exclusion; no 50pct coverage filter',
        next_analysis='reuse bounded 22 readouts after qualification; individual patient statistics, no pooled-event pseudoreplication',
        user_visual_acceptance='pending'))
    return rows


def prepare_interictal(sid,record,sql,inv,out):
    src=Path('/mnt/epilepsia_data/interilca_inter_results/all_data_lns')/sid/'all_recs'
    loaded=load_subject_propagation_events(src)
    old=json.loads((ROOT/f'results/interictal_propagation_masked/per_subject/epilepsiae_{sid}.json').read_text())
    assert loaded['record_names']==old['event_metadata']['record_names']
    assert loaded['block_boundaries']==old['event_metadata']['block_boundaries']
    # The all-event producer selects from raw artifact columns BEFORE fitting
    # spatial directions. Events with <3 participating contacts retain timing
    # labels; applying the older propagation validity mask loses these events.
    valid=frozen_event_indices(record,len(loaded['event_abs_times']))
    labels=np.asarray(record['template_discovery']['event_labels'])
    assert len(labels)==len(valid)
    starts=loaded['event_abs_times'][valid];ids=loaded['block_ids'][valid];ends=np.full(len(valid),np.nan)
    bm={f'{b["recording_id"]}_{b["block_no"]:04d}':b for b in sql['blocks']};ranges=[]
    full_ends=np.full(len(loaded['event_abs_times']),np.nan)
    for bid,stem in enumerate(loaded['record_names']):
        p=src/f'{stem}_packedTimes_withFreqCent.npy'
        if not p.exists():raise FileNotFoundError(p)
        packed=np.load(p);m=loaded['block_ids']==bid;b=bm[stem]
        np.testing.assert_allclose(loaded['event_abs_times'][m],b['begin_epoch']+packed[:,0],rtol=0,atol=1e-5)
        full_ends[m]=b['begin_epoch']+packed[:,1]
        assert contained(loaded['event_abs_times'][m],full_ends[m],b['begin_epoch'],b['end_epoch']).all()
        ranges.append((b['begin_epoch'],b['end_epoch']))
    ends=full_ends[valid];good=outside_seizures(starts,ends,inv)
    order=np.argsort(starts,kind='stable')
    np.savez_compressed(out/'event_index.npz',event_abs_time=starts[order],event_end_time=ends[order],template_label=labels[order],
        nonseizure_eligible=good[order],source_event_index=valid[order],source_block_id=ids[order],source_record_names=loaded['record_names'],coverage_ranges=np.asarray(ranges))
    rows=[]
    for i,r in enumerate(inv):
        base=inv[i-1]['offset'] if i else min(b['begin_epoch'] for b in sql['blocks'])
        for win,sec in [('whole',None),('pre60',3600),('pre15',900)]:
            lo=max(base,r['onset']-sec) if sec else base;hi=r['onset'];duration=max(0,hi-lo);seconds=coverage(ranges,lo,hi) if duration else 0
            m=good&contained(starts,ends,lo,hi);na=int(np.sum(m&(labels==0)));nb=int(np.sum(m&(labels==1)))
            rows.append(dict(subject=sid,sz=r['sz'],window=win,n_ta=na,n_tb=nb,n_events=na+nb,
                observed_hours=seconds/3600,interval_hours=duration/3600,coverage_fraction=seconds/duration if duration else 0,
                start_epoch=lo,end_epoch=hi,has_previous=bool(i),complete_window=sec is None or duration>=sec,
                qualified_source_label='',qualification_status='PENDING_BROADBAND_GATE'))
    pd.DataFrame(rows).to_csv(out/'interval_observations_unlabelled.csv',index=False)
    return dict(n_events=len(valid),n_nonseizure=int(good.sum()),n_source_blocks=len(ranges))


def frozen_event_indices(record,n_raw):
    discovery=record['template_discovery']
    indices=np.asarray(discovery['sampled_event_indices'],dtype=int)
    assert discovery['method']=='timing_plus_space_all_events_missing_view_v1'
    assert np.array_equal(indices,np.arange(n_raw)), 'Frozen all-event index differs from raw artifact order'
    assert len(discovery['event_labels'])==n_raw
    return indices


def spectral_diagnostics(signal,fs,times_offset,baseline_clinical):
    fs=float(fs);freq,t,psd=spectrogram(np.asarray(signal,float),fs=fs,nperseg=int(round(fs)),noverlap=int(round(fs))-int(round(.5*fs)),scaling='density',mode='psd',axis=-1)
    t=t-times_offset;valid=(freq>=1)&(freq<=150)&(freq<fs/2);freq=freq[valid];psd=psd[:,valid,:]
    if len(freq)<2:raise ValueError('insufficient_frequency_support')
    logpsd=np.log(np.maximum(psd,1e-30));base=(t>=baseline_clinical[0]-1e-9)&(t<=baseline_clinical[1]+1e-9)
    early=(t-.5>=-1e-9)&(t+.5<=10+1e-9)
    if base.sum()<50 or early.sum()<18:raise ValueError(f'incomplete_spectral_support:{base.sum()}/{early.sum()}')
    logbb=np.log(np.maximum(psd.sum(axis=1),1e-30));robust=distal_baseline_robust_z(logbb,t,baseline_clinical,min_frames=50)
    av,complete=aggregate_complete_windows(robust['delta'],t,np.array([[0,10,5]]),spectral_window_sec=1.)
    assert complete[0]
    band_db=[];band_z=[];keys=[];bounds=[];hits=[];hit_diagnostics=[]
    from scripts.plot_topic5_early_spectral_phenotypes import _finite_spatial_trace
    from src.topic5_energy_timing import detect_sustained_enhancement
    eegrel=baseline_clinical[0]+120.;teeg=t-eegrel
    for key,(lo,hi) in BANDS.items():
        m=(freq>=lo)&(freq<hi)
        for line in (50,100,150):m &= np.abs(freq-line)>2
        if m.sum()<2:continue
        logpow=np.log(np.maximum(psd[:,m,:].sum(axis=1),1e-30));b=distal_baseline_robust_z(logpow,t,baseline_clinical,min_frames=50)
        db=(logpow[:,early].mean(axis=1)-np.median(logpow[:,base],axis=1))*10/np.log(10)
        band_db.append(db);band_z.append(b['delta'][:,early].mean(axis=1));keys.append(key);bounds.append([float(freq[m][0]),float(freq[m][-1])])
        trace=_finite_spatial_trace(b['delta'],teeg)
        hit=detect_sustained_enhancement(trace,teeg,baseline=(-120.,-90.),search=(-eegrel,10.-eegrel),baseline_quantile=.99,sustain_sec=2.)
        hits.append(bool(hit.detected));hit_diagnostics.append(dict(detected=bool(hit.detected),peak_minus_q99=float(hit.peak_value-hit.threshold),longest_above_sec=float(hit.longest_above_sec)))
    baseline_spectrum=np.median(logpsd[:,:,base],axis=-1);early_spectrum=logpsd[:,:,early].mean(axis=-1)
    return dict(frequencies_hz=freq,times_clinical=t,log_psd_baseline=logpsd[:,:,base].astype(np.float32),log_psd_early=logpsd[:,:,early].astype(np.float32),
        baseline_spectrum_log=baseline_spectrum,early_spectrum_log=early_spectrum,
        db_spectrogram=((logpsd-baseline_spectrum[:,:,None])*10/np.log(10)).astype(np.float32),
        band_db=np.array(band_db).T,band_z=np.array(band_z).T,band_names=keys,band_bounds=bounds,
        band_hits=hits,hit_diagnostics=hit_diagnostics,activation=av[0],upper_hz=float(freq[-1]),n_baseline_frames=int(base.sum()),n_early_frames=int(early.sum()))


def extract_one(sid,idx,record,inv,out):
    from src.ictal_onset_extraction import extract_seizure_window
    from src.topic5_ictal_recruitment import bipolar_alias_label
    r=inv[idx]
    if r['eeg_onset_epoch'] is None:raise ValueError('missing_EEG_onset_for_baseline')
    eegrel=r['eeg_onset_epoch']-r['clin_onset_epoch'];pre=max(121.,121.-eegrel)
    sw=extract_seizure_window('epilepsiae/'+sid,idx,pre_sec=pre,post_sec=11.,reference='car')
    assert str(sw.seizure_id)==str(r['seizure_id'])
    assert abs(sw.clin_onset_epoch-r['clin_onset_epoch'])<1e-6 and abs(sw.eeg_onset_epoch-r['eeg_onset_epoch'])<1e-6
    names=[bipolar_alias_label(n) for n in sw.ch_names];target=record['interictal_field']['contact_order']
    assert len(set(names))==len(names)
    missing=[n for n in target if n not in names]
    if missing:raise ValueError(f'missing_frozen_target_contacts:{missing}')
    x=sw.signal[[names.index(n) for n in target]]
    d=spectral_diagnostics(x,sw.fs,sw.pre_sec,(eegrel-120,eegrel-90))
    q=dict(analysis_version=VERSION,subject=sid,seizure_idx=idx,sz=idx+1,seizure_id=r['seizure_id'],status='BROADBAND_ASSESSED',
           field_sha256=sha(FIELD_ROOT/f'epilepsiae_{sid}.json'),sample_rate_hz=float(sw.fs),contact_order=target,
           baseline_reference='EEG onset',baseline_eeg_sec=[-120,-90],baseline_clinical_sec=[eegrel-120,eegrel-90],
           early_clinical_sec=[0,10],is_exact_1_150=bool(np.isclose(d['upper_hz'],150)),band_hz=[1,d['upper_hz']],
           diagnostic_band_names=d['band_names'],diagnostic_band_bounds=d['band_bounds'],activation=d['activation'],
           n_baseline_frames=d['n_baseline_frames'],n_early_frames=d['n_early_frames'],
           broadband_qualified=False,qualified_source_label='',qualification_status='ASSESSED',band_hits=d['band_hits'],band_hit_diagnostics=d['hit_diagnostics'],
           early_window_within_seizure=bool(r['clin_onset_epoch']+10<=r['offset']),
           baseline_seizure_overlap_ids=[z['seizure_id'] for z in inv if r['eeg_onset_epoch']-120<z['offset'] and r['eeg_onset_epoch']-90>z['onset']])
    sc=scorers_from_interictal_record(record);a,b=sc['shared_a'],sc['shared_b'];assert np.array_equal(a['points'],b['points']) and a['sigma']==b['sigma']
    support=np.minimum(a['support'],b['support']);assert np.all(support>0)
    rs,_,_=common_scores(d['activation'],record['rank_a'],record['rank_b'],a['points'],support,a['sigma'])
    label,provisional=classify_source(*rs)
    q.update(provisional_r_a=rs[0],provisional_r_b=rs[1],provisional_source_label=label,
             geometry_tier=interictal_field_quality_tier(record),band_db=d['band_db'],band_robust_z=d['band_z'])
    nlow=sum(d['band_hits'][:3]);nfast=sum(d['band_hits'][3:]);pass_bands=len(d['band_hits'])==6 and nlow>=2 and nfast>=2 and nlow+nfast>=5
    q.update(n_low_band_hits=nlow,n_fast_band_hits=nfast,n_total_band_hits=nlow+nfast,passes_spectral_5of6=bool(pass_bands))
    reasons=[]
    if not q['is_exact_1_150']:reasons.append('incomplete_1_150Hz_support')
    if q['baseline_seizure_overlap_ids']:reasons.append('baseline_overlaps_seizure')
    if not q['early_window_within_seizure']:reasons.append('clinical_0_10_exceeds_annotated_seizure')
    if not pass_bands:reasons.append('does_not_meet_matched_window_broadband_5of6')
    q.update(broadband_qualified=not reasons,qualification_exclusions=reasons)
    if not reasons and label in ('TA','TB'):q.update(qualified_source_label=label,status='QUALIFIED_SOURCE_READY')
    elif not reasons:q['status']='BROADBAND_PASS_SOURCE_UNCLEAR'
    else:q['status']='EXCLUDED_BY_QUALIFICATION'
    # No field-matching result is allowed to establish enhancement eligibility.
    if sid=='1146':
        previous=ROOT/f'results/topic5_preseizure_template_share/epilepsiae_1146/per_seizure/seizure_{idx:03d}.json'
        if previous.exists():
            p=json.loads(previous.read_text())
            if p.get('status')=='ok':
                np.testing.assert_allclose(d['activation'],p['activation'],rtol=1e-9,atol=1e-9)
                q['e1146_activation_parity_max_error']=float(np.max(np.abs(d['activation']-p['activation'])))
    folder=out/'per_seizure';folder.mkdir(parents=True,exist_ok=True)
    arrays={k:v for k,v in d.items() if isinstance(v,np.ndarray)}
    np.savez_compressed(folder/f'seizure_{idx:03d}_spectral.npz',**arrays,contact_order=target,band_names=d['band_names'])
    plot_diagnostic(q,d,out)
    write_json(folder/f'seizure_{idx:03d}.json',q)
    return q


def plot_diagnostic(q,d,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nondelta=[i for i,k in enumerate(d['band_names']) if k!='delta']
    positive=np.sum(d['band_db'][:,nondelta]>0,axis=1);contact=int(np.argmax(positive))
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    im=axes[0].imshow(d['band_db'],aspect='auto',cmap='RdBu_r',vmin=-12,vmax=12)
    axes[0].set(yticks=np.arange(len(q['contact_order'])),yticklabels=q['contact_order'],xticks=np.arange(len(d['band_names'])),xticklabels=[f'{v[0]:g}–{v[1]:g}' for v in d['band_bounds']],xlabel='Diagnostic band (Hz; notch bins omitted)',title='Early − baseline band power')
    axes[0].tick_params(axis='y',labelsize=8);axes[0].tick_params(axis='x',labelsize=8);fig.colorbar(im,ax=axes[0],label='dB',fraction=.035)
    f=d['frequencies_hz'];axes[1].plot(f,d['baseline_spectrum_log'][contact]*10/np.log(10),c='black',label='EEG −120 to −90 s')
    axes[1].plot(f,d['early_spectrum_log'][contact]*10/np.log(10),c='#2166AC',label='Clinical 0–10 s')
    axes[1].set(xlim=(1,d['upper_hz']),xlabel='Frequency (Hz)',ylabel='PSD (dB, signal units²/Hz)',title=f'Diagnostic contact: {q["contact_order"][contact]}');axes[1].legend(fontsize=8,frameon=False)
    t=d['times_clinical'];m=(t>=-20)&(t<=10)
    im=axes[2].pcolormesh(t[m],f,d['db_spectrogram'][contact][:,m],cmap='RdBu_r',vmin=-12,vmax=12,shading='nearest')
    axes[2].axvline(0,c='black',lw=.7);axes[2].set(xlabel='Seconds from clinical onset',ylabel='Frequency (Hz)',title='Frequency-resolved enhancement')
    fig.colorbar(im,ax=axes[2],label='dB vs baseline',fraction=.035)
    fig.suptitle(f'E{q["subject"]} SZ{q["sz"]}: broadband {q["n_total_band_hits"]}/6 bands | source label: {q["qualified_source_label"] or "not eligible"}',fontsize=13)
    fp=out/'figures';fp.mkdir(parents=True,exist_ok=True);name=f'seizure_{q["seizure_idx"]:03d}_broadband_diagnostic.png'
    fig.savefig(fp/name,dpi=140,bbox_inches='tight');plt.close(fig)
    with (fp/'README.md').open('a') as h:h.write(f'\n### {name}\n展示E{q["subject"]}第{q["sz"]}次发作的逐触点分频能量变化、基线与早期频谱、以及临床起点附近频谱变化。示例触点按非delta频段中正变化的数量选择，仅供诊断展示，不能充当增强检验；实际band边界和采样率见JSON。\n**关注点**：broadband采用同一临床0–10秒窗的5/6频段持续增强判据；source标签还要求基线未混入发作且早期窗未超过发作结束，图待人工检查。\n')


def process_subject(row,limit=None):
    sid=row['subject'];out=OUT/'per_subject'/f'epilepsiae_{sid}';out.mkdir(parents=True,exist_ok=True)
    status=dict(subject=sid,pid=os.getpid(),status='PREPARING',started_at=time.time(),n_total=row['n_seizures'],n_done=0,n_unavailable=0)
    write_json(out/'status.json',status)
    try:
        record=json.loads(Path(row['field']).read_text());assert sha(row['field'])==row['field_sha256']
        sql,inv,sp=get_inventory(sid);assert sha(sp)==row['sql_sha256']
        write_json(out/'seizure_inventory.json',inv)
        status['interictal']=prepare_interictal(sid,record,sql,inv,out);status['status']='EXTRACTING_SPECTRAL_DIAGNOSTICS';write_json(out/'status.json',status)
        results=[]
        for idx in range(min(len(inv),limit) if limit else len(inv)):
            target=out/'per_seizure'/f'seizure_{idx:03d}.json'
            cached=json.loads(target.read_text()) if target.exists() else None
            if cached is not None and cached.get('analysis_version')==VERSION:
                q=cached;assert q.get('field_sha256')==row['field_sha256']
            else:
                try:q=extract_one(sid,idx,record,inv,out)
                except Exception as e:
                    q=dict(analysis_version=VERSION,subject=sid,seizure_idx=idx,sz=idx+1,seizure_id=inv[idx]['seizure_id'],status='UNAVAILABLE',
                        reason=f'{type(e).__name__}: {e}',field_sha256=row['field_sha256'],broadband_qualified=None,qualified_source_label='')
                    write_json(target,q)
                    with (out/'errors.log').open('a') as h:h.write(f'\nSZ{idx+1}\n'+traceback.format_exc())
            results.append(q);status['n_done']=len(results);status['n_unavailable']=sum(x['status']=='UNAVAILABLE' for x in results)
            write_json(out/'status.json',status)
            print(f'E{sid} SZ{idx+1}/{len(inv)} {q["status"]}',flush=True)
        columns=['subject','sz','seizure_id','status','reason','is_exact_1_150','geometry_tier','provisional_r_a','provisional_r_b','provisional_source_label','broadband_qualified','qualified_source_label']
        pd.DataFrame(results).reindex(columns=columns).to_csv(out/'seizure_qualification.csv',index=False)
        if not limit:status['association']=run_associations(record,inv,results,out)
        status['n_broadband_pass']=sum(bool(x.get('broadband_qualified')) for x in results)
        status['n_qualified_source']=sum(x.get('qualified_source_label') in ('TA','TB') for x in results)
        status['status']='SMOKE_COMPLETE' if limit else 'QUALIFICATION_COMPLETE';status['finished_at']=time.time();write_json(out/'status.json',status)
    except Exception as e:
        status.update(status='SUBJECT_PREPARATION_FAILED',reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc());write_json(out/'status.json',status)
        print(f'E{sid} preparation failed: {e}',flush=True)
    return status


def run_associations(record,inv,results,out):
    from scripts.explore_e1146_seizure_interictal_association import build_features,screen
    labels=pd.DataFrame([dict(sz=q['sz'],label=q.get('qualified_source_label',''),r_a=q.get('provisional_r_a',np.nan),r_b=q.get('provisional_r_b',np.nan)) for q in results])
    labels.to_csv(out/'qualified_seizure_source_labels.csv',index=False)
    counts=labels.iloc[1:].label.value_counts().to_dict()
    if not counts.get('TA',0) and not counts.get('TB',0):return dict(status='NO_QUALIFIED_SOURCE_INTERVALS',n_ta=0,n_tb=0)
    z=np.load(out/'event_index.npz');starts=z['event_abs_time'];ends=z['event_end_time'];labs=z['template_label'];ranges=z['coverage_ranges'].tolist()
    merged=[]
    for lo,hi in sorted(ranges):
        if merged and lo<=merged[-1][1]+1e-6:merged[-1][1]=max(hi,merged[-1][1])
        else:merged.append([lo,hi])
    segments=np.full(len(starts),-1,int)
    for i,(lo,hi) in enumerate(merged):segments[contained(starts,ends,lo,hi)]=i
    assert (segments>=0).all()
    dest=out/'association';dest.mkdir(parents=True,exist_ok=True)
    f,n,w=build_features(events=(inv,starts,ends,labs,segments,ranges),labels=labels,out_dir=dest)
    if min(counts.get('TA',0),counts.get('TB',0))<2:
        result=dict(status='NOT_ESTIMABLE_FEWER_THAN_TWO_SEIZURES_PER_LABEL',n_ta=counts.get('TA',0),n_tb=counts.get('TB',0))
        write_json(dest/'summary.json',result);return result
    stats=screen(f,n,out_dir=dest)
    result=dict(status='EXPLORATORY_SCREEN_COMPLETE',n_ta=counts.get('TA',0),n_tb=counts.get('TB',0),
                n_tests=len(stats),min_within_patient_q=float(stats.q_bh22.min()),geometry_tier=interictal_field_quality_tier(record))
    write_json(dest/'input_contract.json',dict(qualification_version=VERSION,field_sha256=sha(FIELD_ROOT/f'epilepsiae_{record["subject"]}.json'),
                statistical_unit='seizure within patient',source_labels='broadband-qualified only',permutation='exact if <=200000 assignments, otherwise 19999 Monte Carlo draws with +1 correction',
                primary_geometry=interictal_field_quality_tier(record) in ('strict_2d','non_strict_2d')))
    return result


def summarize_cohort(results):
    from scripts.explore_e1146_seizure_interictal_association import bh_adjust
    rows=[];tables=[]
    manifest=pd.read_csv(OUT/'cohort_manifest.csv',dtype={'subject':str}).set_index('subject')
    for s in results:
        sid=s['subject'];a=s.get('association',{})
        rows.append(dict(subject=sid,geometry_tier=manifest.loc[sid,'geometry_tier'],primary_geometry=bool(manifest.loc[sid,'primary_geometry']),
            status=s['status'],n_total=s['n_total'],n_processed=s['n_done'],n_unavailable=s['n_unavailable'],
            n_broadband_pass=s.get('n_broadband_pass',0),n_qualified_source=s.get('n_qualified_source',0),
            n_ta_intervals=a.get('n_ta',0),n_tb_intervals=a.get('n_tb',0),association_status=a.get('status','NOT_RUN')))
        p=OUT/'per_subject'/f'epilepsiae_{sid}'/'association/association_screen.csv'
        if p.exists():
            t=pd.read_csv(p);t.insert(0,'subject',sid);t['primary_geometry']=bool(manifest.loc[sid,'primary_geometry']);tables.append(t)
    pd.DataFrame(rows).to_csv(OUT/'cohort_summary.csv',index=False)
    if tables:
        joined=pd.concat(tables,ignore_index=True);m=joined.primary_geometry
        joined['q_across_primary_patients_and_features']=np.nan
        joined.loc[m,'q_across_primary_patients_and_features']=bh_adjust(joined.loc[m,'p'])
        joined.to_csv(OUT/'all_patient_association_tests.csv',index=False)
    lines=['# 多患者发作前间期统计：先确认broadband增强','','本次只使用共同空间场可用的12名Epilepsiae患者；其中11名二维几何作为主队列，E139为一维几何敏感性病例。TA/TB为患者内多数/少数事件模板，不是跨患者相同解剖位置。', '',
        '新资格规则借用Fig3 strict broadband的六频段5/6、至少2低频+2快频、q75空间读出、2秒平滑和超过基线q99持续2秒逻辑，但改在source标签同一临床0–10秒内判定，并使用本次原始频谱的EEG远端robust-z。因此这是matched-window操作性资格，不冒称旧自适应T_spectral表型的原样复现。', '',
        '基线为EEG起点前[-120,-90]秒；基线混入其他发作、0–10秒超过发作结束、无法覆盖完整1–150Hz、谱段不足等不进入标签关联。通过broadband后仍要求明确的正signed source相关，否则不强迫分类。通过算法判据不等于人工验收，逐发作诊断图待目视检查。', '',
        '统计沿用前轮22项候选，逐事件时间归属，不用整块排除或50%覆盖率门槛；每患者以发作为样本，不把所有患者事件拼成独立重复。每种source不足两次只给描述，不运行两组推断。组合数过大时用19,999次Monte Carlo置换而非枚举。汇总表另对所有主队列患者×候选特征的p统一BH校正，不替代患者内表或时间限制敏感性。', '',
        '|患者|几何|总发作|broadband通过|明确source|TA/TB可用间隔|统计状态|','|---|---|---:|---:|---:|---:|---|']
    for r in sorted(rows,key=lambda x:int(x['subject'])):lines.append(f"|E{r['subject']}|{r['geometry_tier']}|{r['n_total']}|{r['n_broadband_pass']}|{r['n_qualified_source']}|{r['n_ta_intervals']}/{r['n_tb_intervals']}|{r['association_status']}|")
    lines += ['', '文件：`cohort_manifest.csv`为全候选及排除来源，`cohort_summary.csv`为资格与样本流，`all_patient_association_tests.csv`在有可比病例时保存全部检验。每患者`per_seizure/`保留频谱数组/资格JSON，`figures/`保留原始频谱诊断，`association/`保存通过资格后的特征与统计。', '',
        '当前合同与原始频谱提取、source评分、间期事件顺序均可追溯；E1146用于提取一致性回归。后续解释须查看时间层内p和跨患者×特征校正，不能仅选择最小的患者内p。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--subject');parser.add_argument('--limit',type=int);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();rows=prepare_manifest();chosen=[r for r in rows if r['selected'] and (not args.subject or r['subject']==args.subject)]
    if args.prepare_only:print(pd.DataFrame(rows).to_string(index=False));return
    write_json(OUT/'run_status.json',dict(status='RUNNING_BROADBAND_DIAGNOSTICS',pid=os.getpid(),workers=args.workers,started_at=time.time(),subjects=[r['subject'] for r in chosen],qualification=VERSION))
    if args.subject:results=[process_subject(chosen[0],args.limit)]
    else:
        results=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(process_subject,row,args.limit):row['subject'] for row in chosen}
            for future in as_completed(futures):
                results.append(future.result());write_json(OUT/'completed_subjects.json',results);summarize_cohort(results)
    summarize_cohort(results)
    write_json(OUT/'run_status.json',dict(status='QUALIFICATION_AND_ASSOCIATION_BATCH_FINISHED',pid=os.getpid(),finished_at=time.time(),subjects=results,
        note='Only same-window broadband-qualified source labels may enter association statistics.'))

if __name__=='__main__':main()
