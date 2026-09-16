#!/usr/bin/env python
"""Machine summary, common-support pairing table and training-sufficiency table."""
import json,sys,glob
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

ROOT=Path('/data/hfosp_group_event_state_rich_event_identification_v0311')
VIEWS=('count','spatial','morphology');SEC=('load',)
HOR=('1','5','30','120')


DIRS=('runs_rescored','runs_reference_variants_rescored')


def load_cards(dirs=DIRS):
    out={}
    for d in dirs:
        for f in sorted(glob.glob(str(ROOT/d/'*.card.json'))):
            c=json.load(open(f));c['_dir']=d
            out[Path(f).name.replace('.card.json','')]=c
    return out


def key(c):
    k=c['config']
    tag=f"{k['inputs']}/{k['family']}/{k['arm']}"
    if k.get('target_ablation'):tag+='+oldtargets'
    if k.get('crossview_ablation'):tag+='+nodelay'
    if k.get('shuffle_marks'):tag+='+shuffledmarks'
    if c.get('variant'):tag+='+'+c['variant']
    return k['subject'],k['split'],tag


def robust_per_packet(card_key,results_dir):
    """Median per-packet loss beside the mean: a few bursts can own the mean."""
    f=ROOT/f'{results_dir}'/f'{card_key}.outer_detail.npz'
    if not f.exists():return None
    try:z=np.load(f)
    except Exception:return None
    out={}
    for h in HOR:
        row={}
        for view in VIEWS+SEC:
            ks=[k for k in z.files if k.startswith(h+'_') and k.endswith(f'logp_{view}')]
            if not ks:continue
            lp=np.concatenate([z[k] for k in ks])
            un=np.concatenate([z[k.replace('logp','units')] for k in ks])
            m=un>0
            if m.sum()<3:continue
            v=-lp[m]/np.maximum(un[m],1e-9)
            srt=np.sort(-lp[m])
            row[view]=dict(mean=float(v.mean()),median=float(np.median(v)),n=int(m.sum()),
                           worst3_share=float(srt[-3:].sum()/max(srt.sum(),1e-9)))
        if row:out[h]=row
    return out


def sign_test(v):
    """Two-sided exact sign test; the unit is the seizure cluster, not the event."""
    v=[x for x in v if x is not None and np.isfinite(x) and x!=0]
    n=len(v)
    if n<3:return dict(n=n,positive=None,p_two_sided=None,note='fewer than three usable clusters')
    k=sum(1 for x in v if x>0)
    from math import comb
    tail=sum(comb(n,i) for i in range(max(k,n-k),n+1))
    return dict(n=n,positive=k,median=float(np.median(v)),p_two_sided=min(1.,2*tail/2**n))


def seizure_signs(d):
    out={}
    cut=d.get('windows',[{}])[0].get('after_producer_cutoff')
    for window in ('primary_-2h_-30min','secondary_-30min_-5min'):
        rows=[c for c in d.get('per_cluster',[]) if c['window']==window]
        if not rows:continue
        stats={}
        for k in ('recent_30min_count','prefix_exposure_fraction','prefix_readable_fraction',
                  'log_rate','state_norm','posterior_trace',
                  'composition_entropy','morphology_scale','delay_zero_logit'):
            for lab in ('clock_matched','clock_and_recent_rate_matched'):
                v=[c[k][lab]['difference'] for c in rows if isinstance(c.get(k),dict) and c[k].get(lab)]
                stats[f'{k}|{lab}']=sign_test(v)
        out[window]=dict(n_clusters=len(rows),statistics=stats)
    pro=[w for w in d.get('windows',[]) if w.get('after_producer_cutoff') and w['n_legal']>0]
    out['n_prospective_windows']=len(pro)
    out['n_windows_with_queries']=sum(1 for w in d.get('windows',[]) if w['n_legal']>0)
    out['caveat']=('single patient, seizure clusters inside one recording are not independent, '
                   'several statistics are tested without correction, and most clusters precede the '
                   'producer cutoff so they are retrospective')
    return out


def main(dirs=DIRS,out_name='machine_summary.json'):
    cards=load_cards(dirs)
    plan=json.load(open(ROOT/'execution_plan.json')) if (ROOT/'execution_plan.json').exists() else dict(tasks=[])
    done={Path(t['tag']).name for t in plan['tasks'] if t['tag'] in cards}
    rows=defaultdict(dict)
    for card_name,c in cards.items():
        s,sp,tag=key(c)
        c['_card_key']=card_name
        rows[(s,sp)][tag]=c
    table={}
    for (s,sp),arms in sorted(rows.items()):
        # common support check
        units={a:{h:{v:c['outer_units'][h][v] for v in VIEWS+SEC} for h in c['outer_units']} for a,c in arms.items()}
        ref_units=next(iter(units.values()))
        mismatched=[a for a,u in units.items() if u!=ref_units]
        refs={a:c for a,c in arms.items() if '/state' not in a}
        states={a:c for a,c in arms.items() if '/state' in a}
        # strongest reference on the aggregate loss over common support
        def agg(c):
            vals=[c['outer'][h][v] for h in c['outer'] for v in VIEWS if c['outer'][h][v] is not None]
            return float(np.mean(vals)) if vals else None
        best_ref=None
        for a,c in refs.items():
            g=agg(c)
            if g is not None and (best_ref is None or g<best_ref[1]):best_ref=(a,g)
        entry=dict(n_outer_eligible=next(iter(arms.values()))['outer_eligible'],
                   n_outer_requested=next(iter(arms.values()))['outer_requested'],
                   not_estimable=next(iter(arms.values()))['outer_not_estimable'],
                   common_support_ok=not mismatched,mismatched_arms=mismatched,
                   strongest_reference=best_ref[0] if best_ref else None,
                   strongest_reference_aggregate=best_ref[1] if best_ref else None,
                   arms={})
        for a,c in arms.items():
            e=dict(card_key=c['_card_key'],results_dir=c.get('_dir'),
                   updates=c['updates'],stop_reason=c['stop_reason'],selected_updates=c['selected_updates'],
                   inner_selection=c['inner_selection'],aggregate=agg(c),
                   per_horizon={h:{v:c['outer'][h][v] for v in VIEWS+SEC} for h in c['outer']},
                   gradient_state_fraction=c.get('gradient_state_fraction'),
                   peak_gpu_gib=c.get('peak_gpu_gib'),seconds=c['seconds'])
            rb=robust_per_packet(c['_card_key'],c.get('_dir','runs_rescored'))
            if rb:e['per_packet_robust']=rb
            if best_ref:
                e['delta_vs_strongest_reference']={h:{v:(None if (c['outer'][h][v] is None
                        or arms[best_ref[0]]['outer'][h][v] is None)
                        else arms[best_ref[0]]['outer'][h][v]-c['outer'][h][v]) for v in VIEWS+SEC}
                        for h in c['outer']}
            for name,ref in (('intercept','P_marks/I-L-G1/intercept'),
                             ('marked_history','P_marks/I-L-G1/marked_history'),
                             ('constant_state','P_marks/I-L-G1/constant_state')):
                if ref in arms:
                    e[f'delta_vs_{name}']={h:{v:(None if (c['outer'][h][v] is None or arms[ref]['outer'][h][v] is None)
                                                 else arms[ref]['outer'][h][v]-c['outer'][h][v]) for v in VIEWS+SEC}
                                           for h in c['outer']}
            entry['arms'][a]=e
        table[f'{s}|{sp}']=entry
    status=[]
    for t in plan['tasks']:
        c=cards.get(t['tag'])
        st=('NOT_RUN' if c is None else
            ('BUDGET_STOPPED' if c['stop_reason']=='budget' else
             ('PLATEAU_STOPPED' if c['stop_reason']=='plateau' else c['stop_reason'].upper())))
        status.append(dict(tag=t['tag'],priority=t['priority'],status=st,
                           inner=None if c is None else c['inner_selection'],
                           updates=None if c is None else c['updates']))
    sufficiency={}
    for card_name,c in cards.items():
        mv={k:v for k,v in c['parameter_update'].items()}
        moved=[k for k,v in mv.items() if v['absolute_l2_change']>0]
        sufficiency[card_name]=dict(n_parameters=sum(v['numel'] for v in c['param_groups'].values()),
                               n_tensors=len(mv),n_moved=len(moved),
                               largest_relative=max([v['relative_l2_change'] for v in mv.values()
                                                     if v['relative_l2_change'] is not None]+[0.]),
                               plateau_levels=[dict(lr=l['lr'],start=l['start_update'],end=l['end_update'],
                                                    best=l['best_at_level']) for l in c['plateau']['levels']],
                               stop_reason=c['stop_reason'],updates=c['updates'],
                               selected_updates=c['selected_updates'],
                               inner_curve=[(r['updates'],r['inner']) for r in c['curve']])
    frozen={Path(f).name:json.load(open(f)) for f in glob.glob(str(ROOT/'frozen'/'*.frozen.json'))}
    seiz={Path(f).name:json.load(open(f)) for f in glob.glob(str(ROOT/'seizure'/'*.seizure.json'))}
    calib={Path(f).name:json.load(open(f)) for f in glob.glob(str(ROOT/'calibration'/'*.calibration.json'))}
    seizure_summary={k:seizure_signs(v) for k,v in seiz.items() if v.get('status')=='COMPLETE'}
    ledger=[
      dict(item='strict 0.5-hour history arm',status='NOT_ESTIMABLE',
           reason='marks publish with the closed one-hour block, so a 30-minute prefix contains no '
                  'published packet at all; there is no legal query point, not a missing run'),
      dict(item='S-ID 120-minute horizon',status='NOT_ESTIMABLE',
           reason='the held-out core is 30 minutes, so a 120-minute target always falls outside it; '
                  'only 1/5/30-minute horizons are scored inside the core'),
      dict(item='epilepsiae_1096 count view',status='INSTRUMENT_FAILURE',
           reason='per-minute counts are extremely over-dispersed (mean 23.9, median 5, max 141, 28% '
                  'zeros); a single negative binomial cannot fit them and every arm including the '
                  'intercept scores 7.6-16.3 nats, so no arm comparison on this view is meaningful'),
      dict(item='epilepsiae_253 fast-ripple morphology components',status='NOT_ESTIMABLE',
           reason='512 Hz sampling leaves the fast-ripple band unavailable; the fast-ripple energy '
                  'ratio and the ripple->fast-ripple lag are missing by measurement, not by result'),
      dict(item='identity adapter with recent marked history',status='INSTRUMENT_UNINFORMATIVE',
           reason='at this number of fitting queries the history adapter scores worse than the fixed '
                  'per-contact trait even after capacity matching, so it bounds nothing'),
      dict(item='coupled-nonlinear versus linear-uncoupled family',status='NOT_ATTRIBUTABLE',
           reason='both switches move together by design in the first round, so their difference '
                  'cannot be assigned to coupling or to nonlinearity'),
    ]
    out=dict(n_cards=len(cards),table=table,task_status=status,training_sufficiency=sufficiency,
             not_estimable_ledger=ledger,
             frozen_exports=frozen,seizure=seiz,seizure_summary=seizure_summary,calibration=calib,
             delta_convention='delta = loss(reference) - loss(state); positive means the state arm is better',
             reference_choice='strongest reference chosen on the aggregate loss over common support, '
                              'never per-point oracle minimum')
    (ROOT/out_name).write_text(json.dumps(out,indent=1,default=str))
    print(json.dumps(dict(n_cards=len(cards),
        by_status={s:sum(1 for r in status if r['status']==s) for s in {r['status'] for r in status}},
        pairs=list(table.keys())),indent=1))

if __name__=='__main__':
    main()
    if (ROOT/'runs_extended_rescored').exists() and list((ROOT/'runs_extended_rescored').glob('*.card.json')):
        main(dirs=('runs_extended_rescored',),out_name='machine_summary_extended_budget.json')
