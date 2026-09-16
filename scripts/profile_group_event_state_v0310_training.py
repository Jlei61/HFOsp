#!/usr/bin/env python3
"""FIT-only engineering profiles; no scientific comparison or held-out scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.topic5_group_event_state.v039.transition import EventTransition, FutureReadout, endpoint_loss


def run(args):
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type != 'cuda': raise ValueError('This is a CUDA memory/throughput profile')
    torch.cuda.set_device(device)
    data = torch.load(args.data, map_location='cpu', weights_only=False)
    fit = [s for s in data['samples'] if s['phase'] == 'FIT' and s['targets'][1][2]]
    fit = sorted(fit, key=lambda s: len(s['histories']['8.0'][0]), reverse=True)[:32]
    length = max(len(s['histories']['8.0'][0]) for s in fit)
    x = np.zeros((len(fit), length, data['input_dim']), np.float32)
    dt = np.zeros(x.shape[:2], np.float32)
    for i,s in enumerate(fit):
        values, elapsed = s['histories']['8.0']
        x[i,:len(values)] = values
        dt[i,:len(elapsed)] = elapsed
    x, dt = torch.from_numpy(x).to(device), torch.from_numpy(dt).to(device)
    counts = torch.tensor(np.array([s['targets'][1][0] for s in fit]), device=device, dtype=torch.float32)
    recruits = torch.tensor(np.array([s['targets'][1][1] for s in fit]), device=device, dtype=torch.float32)
    output = dict(status='RUNNING', purpose='FIT_ONLY_ENGINEERING_NO_SCIENTIFIC_SCORING',
                  data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(), device=str(device),
                  gpu_name=torch.cuda.get_device_name(device), torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, subject=data['subject'], history_hours=8,
                  fit_rows=len(fit), padded_steps=length, input_dim=data['input_dim'],
                  development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False,
                  selection_targets_read=False, profiles=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for family in args.families:
        sizes = [(16,32),(32,64),(64,64),(64,128)] if family != 'F' else [(16,32),(16,64),(16,128)]
        for width,hidden in sizes:
            torch.manual_seed(20260905)
            observer = EventTransition(data['input_dim'],family,width=width,rank=min(width//2,16),seed=20260905).to(device)
            readout = FutureReadout(observer.width,data['n_recruitment'],hidden=hidden).to(device)
            optimizer = torch.optim.AdamW(list(observer.parameters())+list(readout.parameters()),lr=.001)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            start=time.monotonic(); values=[]
            for step in range(8):
                optimizer.zero_grad(set_to_none=True)
                state=observer.scan(x,dt,checkpoint_chunk=32)
                mu,logits=readout(state,state.new_empty((len(state),0)),2.)
                loss=endpoint_loss(mu,logits,counts,recruits,readout.log_dispersion)[0].mean()
                if not torch.isfinite(loss): raise FloatingPointError('Profile objective is nonfinite')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(list(observer.parameters())+list(readout.parameters()),2.,error_if_nonfinite=True)
                optimizer.step();values.append(float(loss.detach()))
            torch.cuda.synchronize(device)
            output['profiles'].append(dict(family=family,width=observer.width,hidden=hidden,
                parameter_count=sum(p.numel() for p in observer.parameters())+sum(p.numel() for p in readout.parameters()),
                inventory={name:{k:list(v.shape) for k,v in module.named_parameters()} for name,module in [('observer',observer),('readout',readout)]},
                microbatch=len(fit),iterations=8,seconds_per_step=(time.monotonic()-start)/8,
                max_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
                max_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20,
                initial_loss=values[0],last_loss=values[-1],last_gradient_norm=float(norm)))
            args.output.write_text(json.dumps(output,indent=2)+'\n')
            print(json.dumps(output['profiles'][-1]),flush=True)
            del observer,readout,optimizer,state,mu,logits,loss
            torch.cuda.empty_cache()
    output['status']='COMPLETE'
    args.output.write_text(json.dumps(output,indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--device',required=True)
    p.add_argument('--families',nargs='+',choices=['F','L','N'],required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
