#!/usr/bin/env python3
"""Run the intentional future-oracle sensitivity assay on the per-step decoder."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import torch  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.contracts import TrainConfig  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.data import load_human_spatial_data  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.we_decoder import load_frozen_decoder  # noqa: E402
from src.topic5_group_event_state.v035.contracts import DECODER_ROOT,OUTPUT_ROOT  # noqa: E402
from src.topic5_group_event_state.v035.stepwise_train import StepwiseTrainConfig,run_stepwise_future_oracle  # noqa: E402
FITS={'epilepsiae_253':'epilepsiae_253__own_a','epilepsiae_548':'epilepsiae_548__shared','epilepsiae_583':'epilepsiae_583__shared'}; ARM='L3_LOCAL_PLUS_LEARNED_LR'
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--subject',choices=tuple(FITS),required=True); ap.add_argument('--decoder-seed',type=int,required=True); ap.add_argument('--state-seed',type=int,required=True); ap.add_argument('--device',default='cuda:0'); a=ap.parse_args(); device=torch.device(a.device); fit=FITS[a.subject]
    bundle=load_frozen_decoder(DECODER_ROOT/'formal_units'/fit/ARM/f'seed{a.decoder_seed}',DECODER_ROOT/'cache'/fit,device=device); data=load_human_spatial_data(a.subject,train_config=TrainConfig(max_steps=900,seed=a.state_seed)); rate=OUTPUT_ROOT/'dynamic_rate'/a.subject/f'seed{a.state_seed}'/'trajectory_and_scores.npz'; base=OUTPUT_ROOT/'stepwise_decoder'/a.subject/f'decoder_seed{a.decoder_seed}_state_seed{a.state_seed}'/'adapter.pt'; out=OUTPUT_ROOT/'stepwise_oracle'/a.subject/f'decoder_seed{a.decoder_seed}_state_seed{a.state_seed}'
    card=run_stepwise_future_oracle(data,bundle,rate,base,StepwiseTrainConfig(seed=a.state_seed),device=device,out_dir=out); print(json.dumps({'subject':a.subject,'seed':a.state_seed,'selection':card['selection_means']},indent=2))
if __name__=='__main__': main()
