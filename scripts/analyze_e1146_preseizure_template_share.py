#!/usr/bin/env python3
"""Exploratory E1146 seizure labels versus preceding interictal TA/TB shares.

Consumes frozen labels; never fits templates using seizure outcomes. The actual
Fig3C metadata, not the staging directory name, selects the primary field.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.epilepsiae_dataset import _parse_sql_subject, _parse_ts
from src.interictal_propagation import load_subject_propagation_events, _valid_event_indices
from src.topic5_template_axis_field import scorers_from_interictal_record
from src.topic5_tspectral_field_concordance import score_observed_bundle

OUT = ROOT / 'results/topic5_preseizure_template_share/epilepsiae_1146'
ARTIFACT = Path('/mnt/epilepsia_data/interilca_inter_results/all_data_lns/1146/all_recs')
SQL = Path('/mnt/epilepsia_data/all_data_sqls/pat_114602_2012-12-20.sql')
META = ROOT / 'results/paper-ready-figure/fig3/fig3_panelc_metadata.json'
LABELS = ROOT / 'results/interictal_propagation_masked/per_subject/epilepsiae_1146.json'
MIN_EVENTS = 20
MIN_COVERAGE = .5
SEED = 20260909
WINDOWS = {'whole': None, 'pre60min': 3600., 'pre30min': 1800., 'pre120min': 7200.}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def inventory():
    sql = _parse_sql_subject(SQL)
    rows = list(csv.DictReader((ROOT/'results/epilepsiae_seizure_inventory.csv').open()))
    rows = sorted([r for r in rows if r['subject']=='1146'], key=lambda r: float(r['clin_onset_epoch']))
    by_id = {r['seizure_id']: r for r in sql['seizures']}
    assert len(rows) == len(by_id) == 26
    for i, r in enumerate(rows):
        for key in ('eeg_onset','eeg_offset','clin_onset','clin_offset'):
            value = _parse_ts(by_id[r['seizure_id']][key])
            assert abs(value-float(r[key+'_epoch'])) < 1e-5, (i,key)
            r[key+'_epoch'] = value
        r['seizure_idx'] = i
        r['sz'] = i+1
        r['onset'] = min(r['eeg_onset_epoch'],r['clin_onset_epoch'])
        r['offset'] = max(r['eeg_offset_epoch'],r['clin_offset_epoch'])
    assert np.all(np.diff([r['onset'] for r in rows]) > 0)
    return sql, rows


def extract(out):
    from scripts.paper_figures.plot_fig3b_interictal_ictal_shared_field import _extract_clinical_activation
    meta = json.loads(META.read_text())
    fields = {'fig3c': ROOT/meta['frozen_record'],
              'timing_space': ROOT/'results/interictal_propagation_masked/template_gradient_fields_timing_plus_space/per_subject/epilepsiae_1146.json'}
    records = {k: json.loads(p.read_text()) for k,p in fields.items()}
    assert records['fig3c']['interictal_field']['fingerprint_sha256'] == meta['frozen_fingerprint']
    scorers = {k: {n:s for n,s in scorers_from_interictal_record(v).items() if n in ('shared_a','shared_b')}
               for k,v in records.items()}
    assert all(set(s)=={'shared_a','shared_b'} for s in scorers.values())
    assert records['fig3c']['interictal_field']['contact_order'] == records['timing_space']['interictal_field']['contact_order']
    _, inv = inventory()
    all_rows = []
    for row in inv:
        idx = row['seizure_idx']
        target = out/'per_seizure'/f'seizure_{idx:03d}.json'
        hashes = {k:sha(p) for k,p in fields.items()}
        if target.exists():
            cached = json.loads(target.read_text())
            assert cached['field_sha256'] == hashes
            all_rows.append(cached)
            continue
        result = {'seizure_idx':idx,'sz':idx+1,'seizure_id':row['seizure_id'], 'field_sha256':hashes}
        try:
            act, extraction = _extract_clinical_activation('epilepsiae_1146',idx,records['fig3c'])
            assert extraction['seizure_id']==row['seizure_id']
            assert extraction['is_exact_1_150'] and extraction['n_finite_contacts']==15
            result.update(status='ok',activation=act,extraction=extraction)
            result['scores'] = {k:score_observed_bundle(s,act) for k,s in scorers.items()}
            checkpoint = ROOT/f'results/topic5_ictal_recruitment/tspectral_field_concordance/per_subject/clinical_onset_shared_field/epilepsiae_1146/seizure_{idx:03d}.json'
            if checkpoint.exists():
                prior = json.loads(checkpoint.read_text())
                assert prior['field_sha256']==hashes['fig3c']
                err=max(abs(prior['event'][k]-result['scores']['fig3c'][k]) for k in ('shared_a_signed','shared_b_signed'))
                assert err < 1e-10, (idx,err)
                result['checkpoint_max_error']=err
        except (ValueError,FileNotFoundError,RuntimeError) as exc:
            result.update(status='not_estimable',reason=f'{type(exc).__name__}: {exc}')
        write_json(target,result)
        all_rows.append(result)
        print(f'extract SZ{idx+1:02d}: {result["status"]}',flush=True)
    write_json(out/'extraction_summary.json',all_rows)


def daynight(t):
    h=datetime.fromtimestamp(t,ZoneInfo('Europe/Berlin')).hour
    return 'day' if 8<=h<20 else 'night'


def overlap(a,b,c,d):
    return a<d and b>c


def coverage(ranges,lo,hi):
    # SQL blocks are disjoint; missing intervals never count as observations.
    return sum(max(0.,min(hi,b)-max(lo,a)) for a,b in ranges)


def event_data(sql,inv,out):
    d=json.loads(LABELS.read_text())
    loaded=load_subject_propagation_events(ARTIFACT)
    valid=_valid_event_indices(loaded['bools'],min_participating=3)
    labels=np.asarray(d['adaptive_cluster']['labels'])
    assert len(valid)==len(labels)==46683
    assert loaded['channel_names']==d['channel_names']
    assert loaded['block_boundaries']==d['event_metadata']['block_boundaries']
    pair=json.loads((ROOT/'results/interictal_propagation_masked/rank_displacement/per_subject/epilepsiae_1146.json').read_text())['pairs'][0]
    field=json.loads((ROOT/json.loads(META.read_text())['frozen_record']).read_text())
    assert (pair['cluster_id_a'],pair['cluster_id_b'])==(0,1)
    for c,k in zip(d['adaptive_cluster']['clusters'],('a','b')):
        assert np.array_equal(c['template_rank'],pair[f'rank_{k}_full'])
        assert np.array_equal(field[f'rank_{k}'],pair[f'rank_{k}_full'])
    blocks={f'{b["recording_id"]}_{b["block_no"]:04d}':b for b in sql['blocks']}
    times=loaded['event_abs_times'][valid]
    ids=loaded['block_ids'][valid]
    good=np.ones(len(times),bool)
    block_rows=[]
    ranges=[]
    # Whole interval deliberately retains homogeneous postictal blocks; a
    # separate no-post60 analysis tests whether the previous seizure drives it.
    for bid,stem in enumerate(loaded['record_names']):
        b=blocks[stem]; lo,hi=b['begin_epoch'],b['end_epoch']
        idx=(ids==bid)
        assert abs(loaded['block_start_times'][bid]-lo)<1e-5
        packed=np.load(ARTIFACT/f'{stem}_packedTimes_withFreqCent.npy')
        assert len(packed)==idx.sum()
        assert np.all(times[idx]>=lo) and np.all(packed[:,1]+lo<=hi+1e-5)
        reasons=[]
        if any(overlap(lo,hi,r['onset'],r['offset']) for r in inv):reasons.append('seizure_overlap')
        if any(lo<r['offset']+3600.<hi for r in inv):reasons.append('post60_boundary')
        if daynight(lo)!=daynight(hi-1e-4):reasons.append('day_night_boundary')
        good[idx]=not reasons
        if not reasons:ranges.append((lo,hi))
        block_rows.append({'block':stem,'start':lo,'end':hi,'n_events':int(idx.sum()),
                           'strict_eligible':not reasons,'exclusions':'|'.join(reasons),
                           'gap_before_sec':b['gap_sec'],'daynight':daynight(lo)})
    pd.DataFrame(block_rows).to_csv(out/'block_coverage_audit.csv',index=False)
    rawranges=[(b['begin_epoch'],b['end_epoch']) for b in sql['blocks']]
    art_ranges=[(blocks[s]['begin_epoch'],blocks[s]['end_epoch']) for s in loaded['record_names']]
    # Save event indices and labels so count tables can be independently audited.
    np.savez_compressed(out/'event_index.npz',event_abs_time=times,template_label=labels,
                        strict_eligible=good,source_event_index=valid,source_block_id=ids,
                        source_record_names=loaded['record_names'])
    return times,labels,good,ranges,rawranges,art_ranges,block_rows


def labels_table(out,inv):
    extracted=json.loads((out/'extraction_summary.json').read_text())
    result=[]
    for row,e in zip(inv,extracted):
        assert row['seizure_id']==e['seizure_id']
        r=dict(row,status=e['status'],reason=e.get('reason',''))
        for field in ('fig3c','timing_space'):
            if e['status']!='ok':continue
            s=e['scores'][field]
            a,b=s['shared_a_signed'],s['shared_b_signed']
            r.update({f'{field}_r_a':a,f'{field}_r_b':b,
                      f'{field}_abs_label':'A' if abs(a)>abs(b) else 'B',
                      f'{field}_signed_label':'A' if a>b else 'B',
                      f'{field}_abs_margin':abs(abs(a)-abs(b)),
                      f'{field}_signed_margin':abs(a-b),
                      f'{field}_winner_sign':float(np.sign(a if abs(a)>abs(b) else b))})
        result.append(r)
    pd.DataFrame(result).to_csv(out/'seizure_labels.csv',index=False)
    return result


def counts(out,inv,events):
    times,labels,good,ranges,rawranges,art_ranges,_=events
    rows=[]
    for i,r in enumerate(inv):
        previous=inv[i-1] if i else None
        base=previous['offset'] if previous else min(a for a,b in rawranges)
        for post in (0,60):
            for name,secs in WINDOWS.items():
                lo=max(base+post*60.,r['onset']-secs) if secs else base+post*60.
                hi=r['onset'];duration=max(0.,hi-lo)
                ix=good&(times>=lo)&(times<hi)
                a=int(np.sum(ix&(labels==0)));b=int(np.sum(ix&(labels==1)));n=a+b
                cov=coverage(ranges,lo,hi) if duration else 0.
                f=cov/duration if duration else 0.
                rt=coverage(rawranges,lo,hi) if duration else 0.
                ac=coverage(art_ranges,lo,hi) if duration else 0.
                # The nominal pre-window must fit between seizures, even when
                # its available part contains many events.
                complete_request=(secs is None or duration>=secs-1e-5)
                eligible=bool(i and n>=MIN_EVENTS and f>=MIN_COVERAGE and complete_request and r['status']=='ok')
                why=[]
                if not i:why.append('no_previous_seizure')
                if not duration:why.append('no_nonoverlapping_interval')
                if not complete_request:why.append('previous_seizure_truncates_window')
                if n<MIN_EVENTS:why.append('fewer_than_20_events')
                if f<MIN_COVERAGE:why.append('strict_coverage_below_50pct')
                if r['status']!='ok':why.append('no_ictal_label')
                q=dict(seizure_idx=i,sz=i+1,seizure_id=r['seizure_id'],window=name,
                       exclude_post_minutes=post,previous_sz=i if i else None,start_epoch=lo,end_epoch=hi,
                       interval_hours=duration/3600,n_ta=a,n_tb=b,n_events=n,ta_share=a/n if n else np.nan,
                       strict_coverage_hours=cov/3600,strict_coverage_fraction=f,
                       raw_coverage_fraction=rt/duration if duration else 0.,
                       artifact_coverage_fraction=ac/duration if duration else 0.,
                       complete_requested_window=complete_request,inference_eligible=eligible,
                       exclusions='|'.join(why),onset_daynight=daynight(hi))
                for field in ('fig3c','timing_space'):
                    for rule in ('abs','signed'):
                        label=r.get(f'{field}_{rule}_label','')
                        q[f'{field}_{rule}_label']=label
                        q[f'{field}_{rule}_matched_share']=(a if label=='A' else b)/n if n and label else np.nan
                rows.append(q)
    df=pd.DataFrame(rows);df.to_csv(out/'interval_template_shares.csv',index=False)
    return df


def association(x,y,strata=None):
    """Positive mean TA-share contrast for TA-labelled versus TB-labelled SZs.

    All permutations operate on seizures, leaving every event burst and
    interval denominator intact. Rotations are a serial-dependence sensitivity,
    not a distribution-free test under arbitrary temporal nonstationarity.
    """
    x=np.asarray(x,float);y=np.asarray(y)=='A';n=len(x);na=int(y.sum())
    if min(na,n-na)<2:return {'status':'not_estimable','n':n,'n_ta_sz':na,'n_tb_sz':n-na}
    obs=float(x[y].mean()-x[~y].mean())
    permutations=math.comb(n,na)
    rng=np.random.default_rng(SEED)
    if permutations<=200000:
        null=np.array([x[list(ii)].mean()-(x.sum()-x[list(ii)].sum())/(n-na)
                       for ii in itertools.combinations(range(n),na)])
        p=float(np.mean(null>=obs-1e-12));method='exhaustive_label_allocation'
    else:
        null=np.array([x[rng.permutation(y)].mean() for _ in range(49999)])
        null=(n*null-x.sum())/(n-na)
        p=float((1+np.sum(null>=obs-1e-12))/(1+len(null)));method='monte_carlo_label_allocation'
    shifts=[]
    for k in range(n):
        yy=np.roll(y,k);shifts.append(x[yy].mean()-x[~yy].mean())
    matched=np.where(y,x,1-x)
    result={'status':'ok','n':n,'n_ta_sz':na,'n_tb_sz':n-na,
            'ta_share_mean_before_ta':x[y].mean(),'ta_share_mean_before_tb':x[~y].mean(),
            'delta_ta_share':obs,'label_permutation_p_greater':p,'permutation_method':method,
            'n_permutations':len(null),'circular_rotation_p_greater':float(np.mean(np.asarray(shifts)>=obs-1e-12)),
            'mean_matched_share':matched.mean(),'median_matched_share':np.median(matched),
            'n_matched_majority':int(np.sum(matched>.5)),
            'matched_majority_sign_p_greater_descriptive':binomtest(int(np.sum(matched>.5)),int(np.sum(matched!=.5)),.5,alternative='greater').pvalue,
            'matched_share_by_ta_seizures':x[y].mean(),'matched_share_by_tb_seizures':(1-x[~y]).mean()}
    if strata is not None:
        ss=np.asarray(strata);groups=[np.flatnonzero(ss==s) for s in np.unique(ss)]
        nn=[]
        for _ in range(9999):
            yy=y.copy()
            for g in groups:yy[g]=rng.permutation(y[g])
            nn.append(x[yy].mean()-x[~yy].mean())
        result['daynight_stratified_permutation_p_greater']=(1+np.sum(np.asarray(nn)>=obs-1e-12))/10000
    return result


def statistics(df,out):
    result=[]
    for (win,post),sub in df.groupby(['window','exclude_post_minutes'],sort=False):
        for field in ('fig3c','timing_space'):
            for rule in ('abs','signed'):
                for pool in ('coverage50','observed_only'):
                    keep=sub.inference_eligible if pool=='coverage50' else ((sub.seizure_idx>0)&(sub.n_events>=MIN_EVENTS)&sub[f'{field}_{rule}_label'].isin(['A','B']))
                    s=sub[keep].sort_values('seizure_idx')
                    st=association(s.ta_share,s[f'{field}_{rule}_label'],s.onset_daynight)
                    result.append(dict(window=win,exclude_post_minutes=int(post),field=field,rule=rule,pool=pool,
                                       seizure_indices=s.seizure_idx.tolist(),**st))
    # Two requested scales are co-primary; all alternate labels/windows are
    # sensitivities and cannot rescue the primary interpretation.
    primary=[s for s in result if s['window'] in ('whole','pre60min') and s['exclude_post_minutes']==0 and s['field']=='fig3c' and s['rule']=='abs' and s['pool']=='coverage50']
    ordered=sorted(primary,key=lambda s:s.get('label_permutation_p_greater',1.))
    last=0.
    for j,s in enumerate(ordered):
        if s['status']!='ok':
            s['holm_p_two_primary']=None
            continue
        last=max(last,min(1.,(len(ordered)-j)*s.get('label_permutation_p_greater',1.)))
        s['holm_p_two_primary']=last
    write_json(out/'statistics.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--extract-only',action='store_true')
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    if not (out/'analysis_contract.json').exists():
        write_json(out/'analysis_contract.json',{
            'question':'Do TA/TB-labelled seizures follow intervals enriched in the matching interictal template?',
            'tier':'single_patient_retrospective_exploratory',
            'primary_field':'actual Fig3C metadata frozen_record and fingerprint',
            'primary_label':'argmax(abs(r_TA),abs(r_TB)); polarity retained; not directional similarity',
            'sensitivity_labels':['argmax(signed_r_TA,signed_r_TB)','Timing+Space frozen field'],
            'energy_window':'clinical onset [0,10] seconds, 1-150 Hz CAR log PSD, EEG baseline [-120,-90] seconds',
            'interval':'previous max(EEG,clinical) offset to next min(EEG,clinical) onset; first seizure descriptive only',
            'postictal':'whole includes homogeneous postictal blocks; separate exclusion of first 60 minutes',
            'parent_exclusion':['any seizure overlap','post-offset+60min boundary','day/night boundary'],
            'gap_policy':'SQL block containment; never fill recording or artifact gaps',
            'primary_windows':['whole','pre60min'],'sensitivity_windows':['pre30min','pre120min'],
            'statistical_unit':'seizure and its preceding interval; equal seizure weights',
            'minimum_events':MIN_EVENTS,'minimum_strict_coverage_fraction':MIN_COVERAGE,
            'primary_test':'one-sided TA-share difference between TA- and TB-labelled seizures, seizure-label permutation, two-window Holm',
            'dependencies':'circular label rotations and day/night-stratified label permutation sensitivities; no iid event binomial test',
            'seed':SEED,'source_sha256':{str(p):sha(p) for p in (SQL,LABELS,META,Path(__file__))}})
    extract(out)
    if args.extract_only:return
    sql,inv=inventory();inv=labels_table(out,inv);events=event_data(sql,inv,out)
    df=counts(out,inv,events);st=statistics(df,out)
    print(json.dumps(clean([s for s in st if s['field']=='fig3c' and s['pool']=='coverage50' and s['exclude_post_minutes']==0]),indent=2))


if __name__=='__main__':main()
