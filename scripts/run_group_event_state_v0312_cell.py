#!/usr/bin/env python3
"""Run or exactly resume one v0312 fit from a reviewed JSON configuration."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('NVIDIA_TF32_OVERRIDE','0')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.topic5_group_event_state.v0312.train import RunConfig,run_cell

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True)
    ap.add_argument('--device');ap.add_argument('--stop-after',type=int);ap.add_argument('--deadline-epoch',type=float)
    a=ap.parse_args();cfg=json.loads(Path(a.config).read_text())
    if a.device:cfg['device']=a.device
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    def progress(row,action):print(json.dumps(dict(**row,action=action)),flush=True)
    card=run_cell(RunConfig(**cfg),stop_after=a.stop_after,deadline=a.deadline_epoch,progress=progress)
    print(json.dumps({k:card.get(k) for k in ('status','updates','selected_updates','stop_reason','seconds','peak_allocated_gib')}),flush=True)
