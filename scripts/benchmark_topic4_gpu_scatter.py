"""Isolated actual-graph recurrent scatter benchmark; never a simulator replacement."""
import sys,time,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
import torch
from scripts import run_topic4_geometry_threshold_refinement as r
ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);args=ap.parse_args();dev=f'cuda:{args.gpu}';torch.cuda.set_device(args.gpu)
p=r.rt.read(r.OUT/'plan.json');sub,*_=r.rt.build_frozen_substrate(p['parent_design'],2511,847101)
from kick_probe import _flatten_by_source
kind='ampa' if args.gpu==0 else 'gaba';by_delay=sub.net[kind+'_by_delay'];ptr,dst,dly,w=_flatten_by_source(by_delay,[i for i,a in enumerate(by_delay) if a.nnz>0],sub.n_e if kind=='ampa' else sub.n_i);N=sub.n_e+sub.n_i;M=sub.net['max_delay_steps']+1;rows=[];rng=np.random.default_rng(9094100+args.gpu)
for density in [.001,.01,.05]:
 n=max(1,round((len(ptr)-1)*density));sp=np.sort(rng.choice(len(ptr)-1,n,replace=False))
 def gather():
  st=ptr[sp];cnt=ptr[sp+1]-st;total=int(cnt.sum());ix=np.arange(total)-np.repeat(np.cumsum(cnt)-cnt,cnt)+np.repeat(st,cnt);return (dly[ix]%M).astype(np.int64)*N+dst[ix],w[ix]
 idx,weights=gather();cpu=np.zeros(M*N);np.add.at(cpu,idx,weights);ti=torch.as_tensor(idx,device=dev);tw=torch.as_tensor(weights,dtype=torch.float64,device=dev);gpu=torch.zeros(M*N,dtype=torch.float64,device=dev);gpu.index_add_(0,ti,tw);torch.cuda.synchronize();snapshot=gpu.cpu().numpy();error=float(np.max(abs(cpu-snapshot)));exact=bool(np.array_equal(cpu,snapshot))
 reps=80
 cpu.fill(0);t=time.perf_counter()
 for _ in range(reps):ii,ww=gather();np.add.at(cpu,ii,ww)
 cpu_ms=(time.perf_counter()-t)*1000/reps
 gpu.zero_()
 for _ in range(10):gpu.index_add_(0,ti,tw)
 torch.cuda.synchronize();t=time.perf_counter()
 for _ in range(reps):gpu.index_add_(0,ti,tw)
 torch.cuda.synchronize();resident_ms=(time.perf_counter()-t)*1000/reps
 t=time.perf_counter()
 for _ in range(reps):
  ii,ww=gather();gpu.index_add_(0,torch.as_tensor(ii,device=dev),torch.as_tensor(ww,dtype=torch.float64,device=dev));gpu[:N].cpu()
 torch.cuda.synchronize();hybrid_ms=(time.perf_counter()-t)*1000/reps
 rows.append(dict(pathway=kind,n_firing=n,density=density,n_edges=len(idx),cpu_gather_scatter_ms=cpu_ms,gpu_resident_scatter_ms=resident_ms,gpu_host_gather_upload_scatter_slot_download_ms=hybrid_ms,max_abs_error=error,bitwise_equal=exact))
 del gpu,cpu,ti,tw
out=r.OUT/'gpu_scatter_feasibility';out.mkdir(exist_ok=True)
r.rt.write(out/f'{kind}.json',dict(gpu=args.gpu,device=torch.cuda.get_device_name(),torch_version=torch.__version__,rows=rows,scope='Actual frozen graph, isolated recurrent-ring scatter. No voltage, OU, threshold, event, or full trajectory parity tested; no production engine change.',source_hash=r.rt.sha(__file__)))
print(json.dumps(rows),flush=True)
