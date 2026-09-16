"""GPU sparse products for the unchanged CPU frozen-v1 time step."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,pickle,argparse
import numpy as np
import torch
from scipy import sparse
from topic4_fig5_z_filtered_guide import make
from continue_topic4_fig5_z_periodic_arc import OUT


class Product:
    def __init__(self,matrix,device):
        self.device=device
        self.matrix=torch.sparse_csr_tensor(torch.tensor(matrix.indptr,device=device),torch.tensor(matrix.indices,device=device),torch.tensor(matrix.data,device=device),size=matrix.shape)
    def apply(self,x):
        return torch.mv(self.matrix,torch.as_tensor(x,device=self.device)).cpu().numpy()


class Part:
    def __init__(self,pair,index):self.pair=pair;self.index=index
    def __matmul__(self,x):
        if self.index==0:self.pair.result=self.pair.apply(x).reshape(2,-1)
        return self.pair.result[self.index]


def gpu_make(s,device=1):
    eq=make(False,s);m=eq.m;torch.set_num_threads(1)
    for k in m.ops:
        pair=Product(sparse.vstack([m.ops[k],m.vops[k]],format='csr'),f'cuda:{device}')
        m.ops[k]=Part(pair,0);m.vops[k]=Part(pair,1)
    return eq


def main():
    p=argparse.ArgumentParser();p.add_argument('--qa',action='store_true');p.add_argument('--s',type=float,default=0.);p.add_argument('--duration',type=float,default=3000.);p.add_argument('--gpu',type=int,default=1);p.add_argument('--initial');p.add_argument('--name');a=p.parse_args()
    eq=gpu_make(a.s,a.gpu) if a.gpu>=0 else make(True,a.s);m=eq.m;nu=np.full(m.n,m.nu_sig)
    if a.initial:
        with open(a.initial,'rb') as f:m.load_state_dict(pickle.load(f))
        m.z_u,m.z2_u=eq.z(a.s)
    if a.qa:
        other=make(True,a.s).m;other.load_state_dict(m.state_dict());t0=time.time()
        for _ in range(50):m.step(nu,m.nu_sig);other.step(nu,other.nu_sig)
        errors={k:float(abs(getattr(m,k)-getattr(other,k)).max()) for k in ['r_u','r_i','m_u','gAE','cAE','gGE','cGE','hE','hI']}
        t0=time.time()
        for _ in range(200):m.step(nu,m.nu_sig)
        row=dict(max_errors=errors,seconds_per_step=(time.time()-t0)/200)
        (OUT/'gpu_guide_qa.json').write_text(json.dumps(row,indent=2)+'\n');print(row,flush=True);return
    folder=OUT/'deterministic';folder.mkdir(exist_ok=True);name=a.name or f's{a.s:g}';rows=[];t0=time.time()
    for k in range(round(a.duration/m.dt)):
        m.step(nu,m.nu_sig)
        if k%10==9:rows.append(np.r_[m.r_u,m.r_i].astype(np.float32)*1000)
        if k%5000==4999:print('GPU GUIDE',a.s,(k+1)*m.dt,time.time()-t0,flush=True)
    np.savez_compressed(folder/f'{name}.npz',r_hz=rows,dt_ms=1.,s=a.s)
    with (folder/f'{name}_end.pkl').open('wb') as f:pickle.dump(m.state_dict(),f)
    print('GPU GUIDE COMPLETE',a.s,time.time()-t0,flush=True)


if __name__=='__main__':main()
