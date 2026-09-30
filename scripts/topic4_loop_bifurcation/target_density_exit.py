#!/usr/bin/env python3
"""Free target-resolved density at the actual-field K9 conditional high state."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,pickle,time
import numpy as np
from scipy import sparse
from campaign import ROOT,read,write,sha
from conditional_density_inputs import OPS
from prepare_target_density import OUT
import density_spatial as physical
from coupled_density_exit import EXTRA
from exit_branch_density import CLAMP

NAME='exit_z0.21_k9_fields16p7_high'
CODE=r'''
extern "C" __global__ void target_delayed(const int* pa,const int* ca,const double* wa,const double* va,
 const int* pb,const int* cb,const double* wb,const double* vb,const double* hist,double* out,
 const int* clock,int depth,int P,int N){
 int g=blockIdx.x,lane=threadIdx.x,tick=clock[0];double a=0.,b=0.,q=0.,v=0.;
 for(int i=pa[g]+lane;i<pa[g+1];i+=128){int d=ca[i]/P+1,slot=(tick-d)%depth;if(slot<0)slot+=depth;
  double r=hist[(long long)slot*P+ca[i]%P];a+=wa[i]*r;q+=va[i]*r;}
 for(int i=pb[g]+lane;i<pb[g+1];i+=128){int d=cb[i]/P+1,slot=(tick-d)%depth;if(slot<0)slot+=depth;
  double r=hist[(long long)slot*P+cb[i]%P];b+=wb[i]*r;v+=vb[i]*r;}
 __shared__ double buf[4][128];buf[0][lane]=a;buf[1][lane]=b;buf[2][lane]=q;buf[3][lane]=v;__syncthreads();
 for(int s=64;s>0;s/=2){if(lane<s)for(int j=0;j<4;j++)buf[j][lane]+=buf[j][lane+s];__syncthreads();}
 if(lane==0)for(int j=0;j<4;j++)out[(long long)j*N+g]=buf[j][0];
}
extern "C" __global__ void target_particles(double* state,int* refs,unsigned char* memory,
 const double* pars,const double* c,const double* arr,const double* drive,const int* group,
 const int* clock,const double* global,const double* pending,int* spikes,int N,int R,int P,int depth,int individual){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=N*R)return;
 int i=id/R,g=group[i],tick=clock[0],ref=refs[id],A=individual?N:P,ag=individual?i:g;
 curandStatePhilox4_32_10_t* all=(curandStatePhilox4_32_10_t*)memory;
 curandStatePhilox4_32_10_t rng=all[id];float4 normal=curand_normal4(&rng);all[id]=rng;
 double x[8],a[4];for(int j=0;j<8;j++)x[j]=state[id*8+j];for(int j=0;j<4;j++)a[j]=arr[j*A+ag];
 spikes[id]=held_cell(x,ref,pars+6*i,c,a,drive[(long long)tick*P+g],global,30.,normal.x,normal.z,pending,tick,depth,i,N);
 refs[id]=ref;for(int j=0;j<8;j++)state[id*8+j]=x[j];
}
extern "C" __global__ void target_collect(const double* state,const int* spikes,const int* ptr,const int* order,
 const double* global,double* hist,double* rates,double* accumulator,double* out,const int* clock,int P,int R,int depth){
 int g=blockIdx.x,lane=threadIdx.x,tick=clock[0],number=(ptr[g+1]-ptr[g])*R;double sum[9]={0,0,0,0,0,0,0,0,0};
 for(int k=lane;k<number;k+=128){int cell=order[ptr[g]+k/R],id=cell*R+k%R;const double* x=state+id*8;
  sum[0]+=spikes[id];sum[1]+=x[6];sum[2]+=x[5];sum[3]+=x[7];sum[4]+=x[2];sum[5]+=x[6]*x[4];
  sum[6]+=x[0];sum[7]+=fabs(x[2])+fabs(x[6]*x[4]);
  sum[8]+=cell<32000 && x[4]+35.662847938268442*30.*global[1]<95.19851312666987;}
 __shared__ double buf[9][128];for(int j=0;j<9;j++)buf[j][lane]=sum[j];__syncthreads();
 for(int s=64;s>0;s/=2){if(lane<s)for(int j=0;j<9;j++)buf[j][lane]+=buf[j][lane+s];__syncthreads();}
 if(lane==0){double r=buf[0][0]/number/.1;hist[(long long)(tick%depth)*P+g]=r;rates[g]=r;accumulator[g]+=r*.1;
  if((tick+1)%10==0){int row=((tick+1)/10-1)%10;out[(row*9)*P+g]=accumulator[g]*1000.;accumulator[g]=0.;
   for(int j=1;j<9;j++)out[(row*9+j)*P+g]=buf[j][0]/number;}}
}
'''


class TargetNetwork:
    def __init__(self,mode,replicas,device):
        import cupy as cp
        cp.cuda.Device(device).use();self.cp=cp;self.mode=mode;self.R=replicas;self.N=40000
        assert read(OUT/'operators_qa.json')['status']=='PASS'
        assert read(ROOT/'exit_branch_density/forcing_qa.json')['status']=='PASS'
        self.geo=dict(np.load(OPS/'geometry.npz'));self.prep=read(OPS/'prepared.json');p=self.prep['params']
        groups=self.geo['cell_group'];self.P=len(self.geo['group_size']);self.sizes=self.geo['group_size']
        self.depth=self.prep['max_delay_steps']+1;self.E=self.geo['population']==0
        self.pars_group_cpu=np.c_[np.where(self.E,p['tau_m_E'],p['tau_m_I']),np.where(self.E,p['tau_ref_E'],p['tau_ref_I'])/.1,
            self.geo['threshold_mv'],self.E,np.where(self.E,p['J_ext_E'],p['J_ext_I']),self.sizes]
        self.pars_cpu=self.pars_group_cpu[groups].copy();self.pars_cpu[:,5]=1
        self.constants_cpu=np.array([np.exp(-.1/p[n]) for n in ['tau_r_AMPA','tau_d_AMPA','tau_r_GABA','tau_d_GABA']]
                       +[p['tau_r_AMPA'],p['tau_r_GABA'],p['V_reset']])
        self.pars=cp.asarray(self.pars_cpu);self.pars_group=cp.asarray(self.pars_group_cpu);self.constants=cp.asarray(self.constants_cpu)
        self.group=cp.asarray(groups,dtype='i4');self.ptr=cp.asarray(np.r_[0,np.cumsum(self.sizes)].astype('i4'))
        self.order=cp.asarray(np.argsort(groups,kind='stable'),dtype='i4')
        self.ops=[];self.ops_cpu=[];folder=OUT if mode=='individual' else OPS
        for kind in ['ampa','gaba']:
            a=sparse.load_npz(folder/f'mean_{kind}.npz').tocsr();q=sparse.load_npz(folder/f'variance_{kind}.npz').tocsr()
            assert np.array_equal(a.indptr,q.indptr) and np.array_equal(a.indices,q.indices)
            self.ops_cpu.append((a,q));self.ops.extend([cp.asarray(a.indptr,dtype='i4'),cp.asarray(a.indices,dtype='i4'),cp.asarray(a.data),cp.asarray(q.data)])
        job=read(ROOT/'exit_return_probes/jobs'/f'{NAME}.json')
        assert sha(job['held_fields_file'])==job['held_fields_sha256']
        with open(job['source_checkpoint'],'rb') as f:saved=pickle.load(f)
        assert saved['identity']==self.prep['graph_identity'];s=saved['engine'];assert s['step']==120000
        with np.load(job['held_fields_file']) as z:zz,kk=z['Z'],z['K']
        heldz=s['slow']['z'].copy();heldz[:32000]=zz;k=np.zeros(self.N);k[:32000]=kk
        self.native_initial=np.stack([s['V'],s['s_E'],s['I_E'],s['s_I'],s['I_I'],s['slow']['m'],heldz,k],axis=1)
        self.native_ref=s['ref'];self.initial_state=cp.asarray(np.repeat(self.native_initial[:,None,:],replicas,axis=1))
        self.initial_ref=cp.asarray(np.repeat(s['ref'][:,None],replicas,axis=1),dtype='i4')
        self.initial_global=np.array([s['termination_mechanism']['r_global'],s['global_feedback_response']['global_state']])
        order=(s['step']+np.arange(self.depth))%self.depth
        self.pending_cpu=np.stack([s['ring_sE'][order],s['ring_sI'][order]]);self.pending=cp.asarray(self.pending_cpu)
        self.drive_cpu=np.load(ROOT/'exit_branch_density/drive_0p1ms.npy',mmap_mode='r');self.drive=cp.asarray(self.drive_cpu)
        assert self.drive.shape==(100000,self.P)
        names=['rng_bytes','init_rng','delayed','global_step','held_supplied','target_delayed','target_particles','target_collect','observe_global']
        self.module=cp.RawModule(code=physical.CODE+EXTRA+CLAMP+CODE,options=('--fmad=false',),name_expressions=names)
        self.k={name:self.module.get_function(name) for name in names}
        size=cp.zeros(1,dtype='i4');self.k['rng_bytes']((1,),(1,),(size,))
        self.rng=cp.empty(self.N*replicas*int(size.get()[0]),dtype='u1')
        self.state=cp.empty_like(self.initial_state);self.ref=cp.empty_like(self.initial_ref);self.spikes=cp.zeros((self.N,replicas),dtype='i4')
        self.history=cp.zeros((self.depth,self.P));self.rate=cp.zeros(self.P);self.clock=cp.zeros(1,dtype='i4')
        self.arr=cp.zeros((4,self.N if mode=='individual' else self.P));self.global_state=cp.zeros(2)
        self.output=cp.zeros((10,9,self.P));self.global_output=cp.zeros((10,2));self.accumulator=cp.zeros(self.P);self.reset()

    def reset(self):
        self.state[:]=self.initial_state;self.ref[:]=self.initial_ref;self.global_state[:]=self.cp.asarray(self.initial_global)
        for x in [self.history,self.rate,self.clock,self.arr,self.spikes,self.output,self.global_output,self.accumulator]:x.fill(0)
        self.k['init_rng'](((self.N*self.R+127)//128,),(128,),(self.rng,np.int32(self.N*self.R),np.uint64(928751),np.int32(self.R)))

    def arrivals(self):
        if self.mode=='individual':
            self.k['target_delayed']((self.N,),(128,),(*self.ops,self.history,self.arr,self.clock,np.int32(self.depth),np.int32(self.P),np.int32(self.N)))
        else:self.k['delayed']((self.P,),(128,),(*self.ops,self.history,self.arr,self.clock,np.int32(self.depth),np.int32(self.P)))

    def step(self):
        self.arrivals()
        self.k['target_particles'](((self.N*self.R+127)//128,),(128,),
            (self.state,self.ref,self.rng,self.pars,self.constants,self.arr,self.drive,self.group,self.clock,self.global_state,
             self.pending,self.spikes,np.int32(self.N),np.int32(self.R),np.int32(self.P),np.int32(self.depth),np.int32(self.mode=='individual')))
        self.k['target_collect']((self.P,),(128,),
            (self.state,self.spikes,self.ptr,self.order,self.global_state,self.history,self.rate,self.accumulator,self.output,
             self.clock,np.int32(self.P),np.int32(self.R),np.int32(self.depth)))
        self.k['global_step']((1,),(128,),(self.rate,self.pars_group,self.global_state,self.clock,np.int32(self.P)))
        self.k['observe_global']((1,),(1,),(self.global_state,self.clock,self.global_output))

    graph=physical.DensityNetwork.graph
    chunk=physical.DensityNetwork.chunk


def check(device):
    assert not (OUT/'implementation_qa.json').exists();checks=[]
    for mode in ['homogeneous','individual']:
        e=TargetNetwork(mode,2,device);cp=e.cp;rng=np.random.default_rng(928759)
        history=rng.uniform(0,.5,size=e.history.shape);tick=358;e.history[:]=cp.asarray(history);e.clock[0]=tick;e.arrivals()
        ids=(tick-np.arange(1,e.depth))%e.depth;hist=history[ids].ravel()
        expected=np.array([e.ops_cpu[0][0]@hist,e.ops_cpu[1][0]@hist,e.ops_cpu[0][1]@hist,e.ops_cpu[1][1]@hist])
        error=float(abs(expected-e.arr.get()).max());assert error<1e-8
        e.reset();local=[]
        for tick in [0,358,359]:
            e.reset();e.clock[0]=tick
            normal=rng.standard_normal((e.N,e.R,2));arr=rng.uniform(0,2,size=(4,e.N));drive=np.asarray(e.drive_cpu[tick,e.geo['cell_group']])
            initial=np.repeat(e.native_initial[:,None,:],e.R,axis=1)
            if tick<e.depth:
                initial[:,:,1]+=e.pending_cpu[0,tick,:,None]/e.constants_cpu[0]
                initial[:,:,3]+=e.pending_cpu[1,tick,:,None]/e.constants_cpu[2]
            expected,ref,spikes=physical.cpu_cell(initial,e.initial_ref.get(),e.pars_cpu,e.constants_cpu,arr,drive,e.initial_global,30.,normal)
            expected[:,:,6:8]=e.initial_state.get()[:,:,6:8]
            members=cp.asarray(np.repeat(np.arange(e.N,dtype='i4')[:,None],e.R,axis=1))
            e.k['held_supplied'](((e.N*e.R+127)//128,),(128,),
                (e.state,e.ref,cp.asarray(normal),e.pars,e.constants,cp.asarray(arr),cp.asarray(drive),e.clock,e.global_state,30.,e.spikes,
                 np.int32(e.N),np.int32(e.R),e.pending,members,np.int32(e.depth),np.int32(e.N)))
            err=float(abs(e.state.get()-expected).max());assert err<2e-11
            assert np.array_equal(e.ref.get(),ref) and np.array_equal(e.spikes.get(),spikes)
            local.append(dict(tick=tick,max_state_error=err))
        e.reset()
        for _ in range(100):e.step()
        keys=['state','ref','rng','history','clock','global_state','output','global_output','accumulator']
        reference={key:getattr(e,key).get() for key in keys}
        e.graph();e.chunk()
        equal={key:np.array_equal(reference[key],getattr(e,key).get()) for key in keys};assert all(equal.values())
        assert np.array_equal(e.state.get()[:,:,6:8],e.initial_state.get()[:,:,6:8])
        out=e.output.get();assert np.all((out[:,8]>=0)&(out[:,8]<=1))
        checks.append(dict(mode=mode,operator_max_error=error,local=local,captured_bitwise=equal,held_fields_bitwise=True))
        print('TARGET IMPLEMENTATION PASS',mode,flush=True);del e;cp.get_default_memory_pool().free_all_blocks()
    write(OUT/'implementation_qa.json',dict(status='PASS',checks=checks,physical_engine_sha256=sha(physical.__file__),producer_sha256=sha(__file__)))


def worker(mode,device):
    assert read(OUT/'implementation_qa.json')['status']=='PASS'
    assert sha(__file__)==read(OUT/'implementation_qa.json')['producer_sha256']
    folder=OUT/mode;folder.mkdir(exist_ok=True);assert not (folder/'progress.json').exists(),'No silent restart'
    write(folder/'progress.json',dict(status='INITIALIZING',pid=os.getpid(),device=device,updated_epoch=time.time()))
    started=time.time();e=TargetNetwork(mode,128,device);e.graph();allout=[];globals=[]
    for offset in range(0,10000,10):
        allout.append(e.chunk().astype('f4'));globals.append(e.global_output.get())
        if (offset+10)%500==0:
            write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),device=device,elapsed_simulation_s=(offset+10)/1000.,
                elapsed_wall_s=time.time()-started,updated_epoch=time.time()))
            print('TARGET FREE',mode,(offset+10)/1000.,flush=True)
    values=np.concatenate(allout);glob=np.concatenate(globals)
    assert int(e.clock.get()[0])==100000 and np.array_equal(e.state.get()[:,:,6:8],e.initial_state.get()[:,:,6:8])
    assert np.isfinite(values).all() and np.isfinite(glob).all()
    np.savez_compressed(folder/'trajectory.npz',elapsed_time_ms=np.arange(1,10001),group_output=values,
         channels=['rate_Hz','Z','M','K','IE','applied_II','V','abs_current','Z_eligible_fraction'],global_R_Hz=glob[:,0],global_s=glob[:,1])
    np.savez_compressed(folder/'final_state.npz',state=e.state.get(),ref=e.ref.get(),rng=e.rng.get(),
         history=e.history.get(),clock=e.clock.get(),global_state=e.global_state.get())
    result=dict(status='COMPLETE',mode=mode,physical_targets=e.N,replicas=e.R,duration_s=10.,elapsed_s=time.time()-started,
         held_fields_bitwise=True,formal_bifurcation_allowed=False)
    write(folder/'result.json',result);write(folder/'progress.json',result);print('TARGET FREE COMPLETE',result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['check','worker']);p.add_argument('--mode',choices=['homogeneous','individual']);p.add_argument('--device',type=int,default=0)
    args=p.parse_args();check(args.device) if args.command=='check' else worker(args.mode,args.device)
