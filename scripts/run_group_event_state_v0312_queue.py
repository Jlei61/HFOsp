#!/usr/bin/env python3
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('NVIDIA_TF32_OVERRIDE','0')
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.topic5_group_event_state.v0312.queue import build_plan,run_worker
from src.topic5_group_event_state.v0312.train import ROOT
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--create',action='store_true');p.add_argument('--root',default=ROOT)
 p.add_argument('--device',default='cuda:0');p.add_argument('--slot',type=int,default=0);p.add_argument('--hours',type=float,default=8);p.add_argument('--quick',action='store_true');p.add_argument('--readiness-only-bypass',action='store_true');a=p.parse_args()
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
 if a.create:
  result=build_plan(a.plan,a.root,a.quick);print(json.dumps(dict(status='PLAN_WRITTEN',tasks=len(result['tasks']))))
 else:
  if a.readiness_only_bypass and ('/readiness/' not in a.plan or not a.quick):raise ValueError('bypass only allowed for explicit quick readiness plan')
  print(json.dumps(run_worker(a.plan,a.device,time.time()+a.hours*3600,a.readiness_only_bypass,a.slot)))
