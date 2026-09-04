#!/usr/bin/env python3
"""Run one event-time-only versus causal-background rate comparison."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from src.topic5_group_event_state.v035.background_rate import run_background_rate  # noqa: E402
from src.topic5_group_event_state.v035.contracts import (  # noqa: E402
    OUTPUT_ROOT, RateTrainConfig, V035_ALL_DEVELOPMENT_SUBJECTS,
)
from src.topic5_group_event_state.v035.dynamic_rate import load_rate_data  # noqa: E402

def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--subject',choices=V035_ALL_DEVELOPMENT_SUBJECTS,required=True)
    ap.add_argument('--seed',type=int,required=True); ap.add_argument('--device',default='cpu'); ap.add_argument('--overwrite',action='store_true')
    ap.add_argument('--rate-root',type=Path,default=OUTPUT_ROOT/'dynamic_rate')
    ap.add_argument('--out-root',type=Path,default=OUTPUT_ROOT/'background_rate')
    a=ap.parse_args(); data=load_rate_data(a.subject,RateTrainConfig(seed=a.seed))
    base=a.rate_root/a.subject/f'seed{a.seed}'; out=a.out_root/a.subject/f'seed{a.seed}'
    card=run_background_rate(data,base,device=torch.device(a.device),out_dir=out,seed=a.seed,overwrite=a.overwrite)
    print(json.dumps({'subject':a.subject,'seed':a.seed,'selection':card['selection']},indent=2))
if __name__=='__main__': main()
