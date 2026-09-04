#!/usr/bin/env python3
"""Open SELECTION once for an INNER-selected full-event checkpoint."""

from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import torch  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.we_decoder import decoder_tensors,load_frozen_decoder  # noqa: E402
from src.topic5_group_event_state.v035.contracts import DECODER_ROOT,OUTPUT_ROOT,atomic_json  # noqa: E402
from src.topic5_group_event_state.v035.full_mark_state import evaluate_physical_selection,evaluate_selection,load_full_mark_data,restore_full_mark_model  # noqa: E402
FITS={'epilepsiae_253':'epilepsiae_253__own_a','epilepsiae_548':'epilepsiae_548__shared','epilepsiae_583':'epilepsiae_583__shared','epilepsiae_1096':'epilepsiae_1096__own_a','epilepsiae_384':'epilepsiae_384__shared','epilepsiae_1125':'epilepsiae_1125__own_a'}
ARM='L3_LOCAL_PLUS_LEARNED_LR'
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--subject',choices=tuple(FITS),required=True); ap.add_argument('--decoder-seed',type=int,required=True); ap.add_argument('--state-seed',type=int,required=True); ap.add_argument('--source-unit',type=Path,required=True); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--out-root',type=Path,default=OUTPUT_ROOT/'full_mark_final'); a=ap.parse_args()
    device=torch.device(a.device); fit=FITS[a.subject]
    bundle=load_frozen_decoder(DECODER_ROOT/'formal_units'/fit/ARM/f'seed{a.decoder_seed}',DECODER_ROOT/'cache'/fit,device=device)
    rate=OUTPUT_ROOT/'dynamic_rate'/a.subject/f'seed{a.state_seed}'/'trajectory_and_scores.npz'; adapter=OUTPUT_ROOT/'stepwise_decoder'/a.subject/f'decoder_seed{a.decoder_seed}_state_seed{a.state_seed}'/'adapter.pt'
    data=load_full_mark_data(a.subject,bundle,rate); source=json.loads((a.source_unit/'card.json').read_text(encoding='utf-8'))
    if source.get('selection',{}).get('status')!='HELD_UNREAD_DURING_HYPERPARAMETER_SEARCH': raise PermissionError('source was not a held-selection search checkpoint')
    model,cfg=restore_full_mark_model(data,bundle,adapter,a.source_unit/'checkpoint.pt',device)
    selection=evaluate_selection(model,data,decoder_tensors(bundle,device),cfg,device)
    with np.load(a.source_unit/'state_trajectory.npz',allow_pickle=False) as trajectory:
        post_state=np.asarray(trajectory['state_post'],dtype=np.float32)
    physical_selection=evaluate_physical_selection(model,data,post_state,device)
    out=a.out_root/a.subject/f'decoder_seed{a.decoder_seed}_state_seed{a.state_seed}'; out.mkdir(parents=True,exist_ok=True)
    card={**source,'format':'group_event_state_v0_3_5_full_mark_final_card_v1','selection':selection,'physical_selection':physical_selection,'source_search_unit':str(a.source_unit),'checkpoint':str(a.source_unit/'checkpoint.pt'),'state_trajectory':str(a.source_unit/'state_trajectory.npz'),'selection_opened_once_after_recipe_lock':True}
    atomic_json(out/'card.json',card); print(json.dumps({'subject':a.subject,'seed':a.state_seed,'output':str(out),'selection':selection},indent=2))
if __name__=='__main__': main()
