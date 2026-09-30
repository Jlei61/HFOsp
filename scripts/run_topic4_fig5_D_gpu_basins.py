"""Run the predeclared six-condition exploration with the validated GPU map."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import pickle
import time
from pathlib import Path
import numpy as np
import torch
from explore_topic4_fig5_D_basins import OUT,OLD,D3,make,write,state_error
from topic4_fig5_D_gpu_map import GPUMap


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--duration-s',type=float,default=6.)
    p.add_argument('--D',type=float,nargs='+',default=[.1,.16,D3]);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    assert json.loads((OUT/'gpu_map_qa.json').read_text())['max_error']<1e-8
    eq=make(False);m=eq.m
    specs=[dict(name=f'D{D:.6f}_{h}',D=D,history=h,source=str(OLD/f))
           for D in a.D for h,f in [('burst','s0_extension_end.pkl'),('tonic','s0.228845_end.pkl')]]
    states=[];prior={}
    for spec in specs:
        name=spec['name'];file=OUT/(name+'_end.pkl') if a.resume else Path(spec['source'])
        with file.open('rb') as f:state=pickle.load(f)
        state['z_u'],state['z2_u']=eq.z(spec['D']);states.append(state)
        if a.resume:prior[name]=dict(np.load(OUT/(name+'.npz')))
    b=GPUMap(eq,states,device=a.gpu)
    count=torch.as_tensor(m.unit_count/m.unit_count.sum(),device=b.device)
    rows=[[] for _ in specs];cells=[[] for _ in specs]
    names=['global_E','core_A','core_B','surround','global_I','mean_M','active_E_fraction','spatial_rate_sd']
    readout_weights=np.stack([m.count_e,*[m.region_w[f'175_{j}'] for j in range(3)]])
    readout_weights/=readout_weights.sum(1,keepdims=True)
    def save(elapsed):
        allstates=b.state_dicts()
        for spec,state,rr,cc in zip(specs,allstates,rows,cells):
            name=spec['name'];r=np.asarray(rr);c=np.asarray(cc)
            if name in prior:
                r=np.concatenate([prior[name]['readouts'],r]);c=np.concatenate([prior[name]['cell_E_hz'],c])
            temp=OUT/(name+'.tmp.npz')
            np.savez_compressed(temp,D=spec['D'],readouts=r,readout_names=names,time_s=np.arange(1,len(r)+1)*.001,
                                cell_E_hz=c,cell_time_s=np.arange(1,len(c)+1)*.01)
            temp.replace(OUT/(name+'.npz'))
            with (OUT/(name+'_end.pkl')).open('wb') as f:pickle.dump(state,f)
        write('progress.json',dict(simulated_s=elapsed,conditions=specs,wall_s=time.time()-start))
    write('status.json',dict(status='RUNNING',pid=os.getpid(),conditions=specs,duration_added_s=a.duration_s,
                             resume=a.resume,backend='validated float64 GPU map'))
    start=time.time()
    for k in range(round(a.duration_s*1000/m.dt)):
        b.step()
        if k%10==9:
            re=((b.r*b.w).reshape(b.B,m.n,m.K).sum(2)*1000).cpu().numpy()
            ri=b.ri.cpu().numpy()*1000;M=(b.M*count).sum(1).cpu().numpy()
            reg=re@readout_weights.T
            for i,rr in enumerate(rows):
                rr.append([*reg[i],np.average(ri[i],weights=m.count_i),M[i],
                           np.average(re[i]>5,weights=m.count_e),
                           np.sqrt(np.average((re[i]-reg[i,0])**2,weights=m.count_e))])
            if k%100==99:
                for i,cc in enumerate(cells):cc.append(re[i].astype(np.float32))
        if k%5000==4999:
            elapsed=(k+1)*m.dt/1000
            save(elapsed)
            print('PROGRESS',elapsed,'wall_s',round(time.time()-start,1),'last_500ms_mean_Hz',
                  np.round([np.asarray(r)[-500:,0].mean() for r in rows],3).tolist(),flush=True)
    save(a.duration_s)
    comparisons=[]
    for spec in specs:
        ref=OUT/'cpu_reference'/f"{spec['name']}.npz"
        if ref.exists() and not a.resume:
            cpu=np.load(ref);gpu=np.load(OUT/(spec['name']+'.npz'));n=min(len(cpu['readouts']),len(gpu['readouts']))
            comparisons.append(dict(name=spec['name'],duration_s=n/1000,
                                    global_rate_max_error_Hz=float(abs(cpu['readouts'][:n,0]-gpu['readouts'][:n,0]).max()),
                                    core_rate_max_error_Hz=float(abs(cpu['readouts'][:n,1:3]-gpu['readouts'][:n,1:3]).max()),
                                    mean_M_max_error=float(abs(cpu['readouts'][:n,5]-gpu['readouts'][:n,5]).max())))
    if comparisons:write('gpu_long_trajectory_qa.json',comparisons)
    write('status.json',dict(status='EXECUTION_COMPLETE',active_processes=[],conditions=specs,duration_added_s=a.duration_s,
                             resume=a.resume,wall_s=time.time()-start,backend='validated float64 GPU map',scientific_status='ANALYSIS_PENDING'))
    print('COMPLETE',time.time()-start,flush=True)


if __name__=='__main__':main()
