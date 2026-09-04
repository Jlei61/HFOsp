#!/usr/bin/env python3
"""Choose one full-event recipe using FIT/INNER evidence only."""

from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.topic5_group_event_state.v035.contracts import OUTPUT_ROOT,atomic_json  # noqa: E402

EXPECTED_SUBJECTS=('epilepsiae_253','epilepsiae_548','epilepsiae_583','epilepsiae_1096')
EXPECTED_SEEDS=(20260903,20260904,20260905)

def main():
    root=OUTPUT_ROOT/'full_mark_search'; rows=[]
    for p in sorted(root.glob('*/*/decoder_seed*_state_seed*/card.json')):
        d=json.loads(p.read_text(encoding='utf-8')); recipe=p.parents[2].name
        if d.get('selection',{}).get('status')!='HELD_UNREAD_DURING_HYPERPARAMETER_SEARCH':
            raise ValueError(f'search card opened selection: {p}')
        rows.append({'recipe':recipe,'subject':d['subject'],'seed':d['seed'],'inner_loss':d['best_inner_loss'],'selected_epoch':d['selected_epoch'],'card':str(p)})
    recipes=sorted({r['recipe'] for r in rows}); subjects=list(EXPECTED_SUBJECTS); seeds=list(EXPECTED_SEEDS)
    by={(r['recipe'],r['subject'],r['seed']):r['inner_loss'] for r in rows}
    unexpected=sorted({r['subject'] for r in rows}-set(subjects))
    if unexpected: raise RuntimeError(f'unexpected search subjects: {unexpected}')
    if not all(('base',s,z) in by for s in subjects for z in seeds): raise RuntimeError('base recipe incomplete')
    summary={}
    for recipe in recipes:
        delta=[]; ranks=[]
        for s in subjects:
            for z in seeds:
                if (recipe,s,z) in by and np.isfinite(by[(recipe,s,z)]): delta.append(by[(recipe,s,z)]-by[('base',s,z)])
        for s in subjects:
            values=[(by[(r,s,z)],r) for r in recipes for z in seeds if (r,s,z) in by and np.isfinite(by[(r,s,z)])]
            if values:
                med={r:np.median([v for v,rr in values if rr==r]) for r in recipes if any(rr==r for _,rr in values)}
                order={r:i for i,(_,r) in enumerate(sorted((v,r) for r,v in med.items()))}
                ranks.extend(order.get(recipe,np.nan) for _ in [0] if recipe in order)
        summary[recipe]={'n_units':len(delta),'median_inner_delta_vs_base':float(np.median(delta)) if delta else None,'median_subject_rank':float(np.median(ranks)) if ranks else None}
    eligible=[r for r in recipes if summary[r]['n_units']>=len(subjects)*len(seeds)]
    chosen=min(eligible,key=lambda r:(summary[r]['median_subject_rank'],summary[r]['median_inner_delta_vs_base'],r))
    payload={'format':'group_event_state_v0_3_5_full_mark_recipe_selection_v1','selected_recipe':chosen,'selection_basis':'median per-subject FIT/INNER rank; no SELECTION target read','subjects':subjects,'seeds':seeds,'recipes':summary,'units':rows,'selection_targets_read':False,'development_targets_read':False,'sealed_partition_opened':False}
    atomic_json(OUTPUT_ROOT/'full_mark_search'/'selected_recipe.json',payload); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
