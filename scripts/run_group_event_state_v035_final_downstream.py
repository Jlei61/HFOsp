#!/usr/bin/env python3
"""Run all W4--W6 analyses for one locked final full-event state unit."""

from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import torch  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.we_decoder import load_frozen_decoder  # noqa: E402
from src.topic5_group_event_state.v035.contracts import (  # noqa: E402
    DECODER_ROOT, OUTPUT_ROOT, V035_DECODER_FITS, atomic_json,
)
from src.topic5_group_event_state.v035.feedback_models import run_feedback_models  # noqa: E402
from src.topic5_group_event_state.v035.full_mark_state import load_full_mark_data  # noqa: E402
from src.topic5_group_event_state.v035.functional_readouts import run_functional_readouts  # noqa: E402
from src.topic5_group_event_state.v035.seizure_transfer import run_seizure_transfer  # noqa: E402
from src.topic5_group_event_state.v035.stepwise_auxiliary import AuxiliaryConfig,run_auxiliary_heads  # noqa: E402
FITS=V035_DECODER_FITS; ARM='L3_LOCAL_PLUS_LEARNED_LR'
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--subject',choices=tuple(FITS),required=True); ap.add_argument('--decoder-seed',type=int,required=True); ap.add_argument('--state-seed',type=int,required=True); ap.add_argument('--batch-events',type=int,default=96); ap.add_argument('--device',default='cuda:0'); a=ap.parse_args()
    device=torch.device(a.device); unit=OUTPUT_ROOT/'full_mark_final'/a.subject/f'decoder_seed{a.decoder_seed}_state_seed{a.state_seed}'; source=json.loads((unit/'card.json').read_text(encoding='utf-8'))
    trajectory=Path(source['state_trajectory']); rate=OUTPUT_ROOT/'dynamic_rate'/a.subject/f'seed{a.state_seed}'/'trajectory_and_scores.npz'; fit=FITS[a.subject]
    bundle=load_frozen_decoder(DECODER_ROOT/'formal_units'/fit/ARM/f'seed{a.decoder_seed}',DECODER_ROOT/'cache'/fit,device=device); data=load_full_mark_data(a.subject,bundle,rate)
    tag=f'decoder_seed{a.decoder_seed}_state_seed{a.state_seed}'
    stages={}
    stages['functional']=run_functional_readouts(data,trajectory,rate,out_dir=OUTPUT_ROOT/'functional_readouts_final'/a.subject/tag)
    stages['auxiliary']=run_auxiliary_heads(data,bundle,trajectory,AuxiliaryConfig(batch_events=a.batch_events,seed=a.state_seed),device=device,out_dir=OUTPUT_ROOT/'stepwise_auxiliary_final'/a.subject/tag)
    stages['seizure']=run_seizure_transfer(a.subject,trajectory,rate,out_dir=OUTPUT_ROOT/'seizure_transfer_final'/a.subject/tag)
    stages['feedback']=run_feedback_models(a.subject,trajectory,rate,out_dir=OUTPUT_ROOT/'feedback_models_final'/a.subject/tag)
    card={'format':'group_event_state_v0_3_5_final_downstream_card_v1','subject':a.subject,'state_seed':a.state_seed,'state_unit':str(unit),'trajectory':str(trajectory),'outputs':{k:str(OUTPUT_ROOT/({'functional':'functional_readouts_final','auxiliary':'stepwise_auxiliary_final','seizure':'seizure_transfer_final','feedback':'feedback_models_final'}[k])/a.subject/tag/'card.json') for k in stages},'development_targets_read':False,'sealed_partition_opened':False}
    atomic_json(OUTPUT_ROOT/'final_downstream'/a.subject/tag/'card.json',card); print(json.dumps(card,indent=2))
if __name__=='__main__': main()
