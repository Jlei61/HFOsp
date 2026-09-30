"""RK4 variational DDE flow with cubic interpolation of delayed rates.

Same frozen model, local gradients and physical delays as CachedMonodromy.
The higher-order method is independently checked against refined Heun flows.
Minimum delay must exceed one integration step, so all stage arrivals are known.
"""
from cached_monodromy import *


DELAY=r'''
extern "C" __global__ void delayed_cubic(const int* pa,const int* ca,const double* wa,const double* va,
 const int* pb,const int* cb,const double* wb,const double* vb,const double* hist,double* out,
 const int* offset,int local,double stage,int depth,double dt){
 int g=blockIdx.x,lane=threadIdx.x,tick=*offset+local;double a=0.,b=0.,q=0.,v=0.;
 for(int i=pa[g]+lane;i<pa[g+1];i+=blockDim.x){
  double d=(ca[i]/P+1)*.1/dt-stage;int lag=(int)floor(d);double f=d-lag;
  double c0=-f*(1-f)*(2-f)/6.,c1=(1+f)*(1-f)*(2-f)/2.;
  double c2=(1+f)*f*(2-f)/2.,c3=-(1+f)*f*(1-f)/6.;
  int s0=(tick-lag+1)%depth;if(s0<0)s0+=depth;int s1=(s0+depth-1)%depth;
  int s2=(s1+depth-1)%depth,s3=(s2+depth-1)%depth,h=ca[i]%P;
  double r=c0*hist[s0*P+h]+c1*hist[s1*P+h]+c2*hist[s2*P+h]+c3*hist[s3*P+h];a+=wa[i]*r;q+=va[i]*r;
 }
 for(int i=pb[g]+lane;i<pb[g+1];i+=blockDim.x){
  double d=(cb[i]/P+1)*.1/dt-stage;int lag=(int)floor(d);double f=d-lag;
  double c0=-f*(1-f)*(2-f)/6.,c1=(1+f)*(1-f)*(2-f)/2.;
  double c2=(1+f)*f*(2-f)/2.,c3=-(1+f)*f*(1-f)/6.;
  int s0=(tick-lag+1)%depth;if(s0<0)s0+=depth;int s1=(s0+depth-1)%depth;
  int s2=(s1+depth-1)%depth,s3=(s2+depth-1)%depth,h=cb[i]%P;
  double r=c0*hist[s0*P+h]+c1*hist[s1*P+h]+c2*hist[s2*P+h]+c3*hist[s3*P+h];b+=wb[i]*r;v+=vb[i]*r;
 }
 __shared__ double buf[4][128];buf[0][lane]=a;buf[1][lane]=b;buf[2][lane]=q;buf[3][lane]=v;__syncthreads();
 for(int k=64;k>0;k/=2){if(lane<k)for(int j=0;j<4;j++)buf[j][lane]+=buf[j][lane+k];__syncthreads();}
 if(lane==0)for(int j=0;j<4;j++)out[j*P+g]=buf[j][0];
}
extern "C" __global__ void rkfinish(double* y,const double* f1,const double* f2,
 const double* f3,const double* f4,double dt){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<14*P)y[i]+=dt*(f1[i]+2*f2[i]+2*f3[i]+f4[i])/6.;
}
'''


class RK4Monodromy(ChunkEndpointMonodromy):
    def __init__(self,s,o,sol,dtmax=.025,dynamic_z=False,device=0,use_gradients=True,block=128,host_gain_cache=False,host_gain_storage='auto'):
        import cupy as cp
        assert not dynamic_z and use_gradients and dtmax<.1
        self.cp=cp;self.s=s;self.dynamic_z=False;cp.cuda.Device(device).use()
        self.T=sol['T'];self.n=int(np.ceil(self.T/dtmax));self.dt=self.T/self.n
        self.Dd=int(np.ceil(s.delays[-1]/self.dt))+2;self.depth=self.Dd+1;self.dim=(NS+self.Dd)*s.P
        sampler=getattr(o,'sample_state',None)
        if sampler is None:
            full,rr=orbit_states(o,sol,2*self.n);allorbit=cp.asarray(np.ascontiguousarray(full))
            self.orbit=allorbit[::2].copy();self.rr=rr[::2];del full,rr
        else:
            self.orbit=cp.asarray(sampler(np.array([0.])))
            self.rr=None
        self.ops=[]
        for kind in ['ampa','gaba']:
            a=sparse.load_npz(s.folder/f'mean_{kind}.npz').tocsr();q=sparse.load_npz(s.folder/f'variance_{kind}.npz').tocsr()
            self.ops.extend([cp.asarray(a.indptr,dtype=cp.int32),cp.asarray(a.indices,dtype=cp.int32),cp.asarray(a.data),cp.asarray(q.data)])
        self.pars,self.consts,self.SE,self.SI,self.WE,self.WI=model_device_arrays(s,cp,dynamic_z=False)
        self.consts=cp.concatenate([self.consts,cp.asarray([1.])]);self.Z=cp.asarray(s.Z)
        self.host_gain_cache=host_gain_cache
        code=CACHED.replace('(*offset+local)*8*P',
                           'local*8*P' if host_gain_cache else '(2*(*offset)+local)*8*P')
        code+=TANGENT[TANGENT.index('extern "C" __global__ void tangent_rhs'):TANGENT.index('extern "C" __global__ void tfinish')]
        code+=DELAY+'\nextern "C" __global__ void advance(int* offset,int n){*offset+=n;}\n'
        names=['cache_gains','cached_rhs','tangent_rhs','tpredictor','delayed_cubic','rkfinish','advance']
        mod=cp.RawModule(code=f'#define P {s.P}\n'+CUDA_DEVICE+CUDA_RESP+code,options=('--fmad=false',),name_expressions=names)
        self.k={name:mod.get_function(name) for name in names}
        self.gains=cp.empty((2*min(block,self.n)+1 if host_gain_cache else 2*self.n+1,8,s.P))
        if host_gain_cache:
            assert sampler is not None,'Host gain cache requires bounded state sampling'
            from host_array_storage import allocate_host_array
            self.host_gains,self.host_gain_storage=allocate_host_array((2*self.n+1,8,s.P),host_gain_storage)
            log('HOST GAIN STORAGE',self.host_gain_storage)
            work_gains=cp.empty((512,8,s.P))
        if sampler is None:
            self.k['cache_gains']((((2*self.n+1)*s.P+127)//128,),(128,),
                (allorbit,self.pars,self.consts,self.SE,self.SI,self.WE,self.WI,self.gains,np.int32(2*self.n+1)))
            del allorbit
        else:
            for lo in range(0,2*self.n+1,512):
                hi=min(lo+512,2*self.n+1)
                states=cp.asarray(sampler(np.arange(lo,hi)*self.dt/2))
                target=work_gains[:hi-lo] if host_gain_cache else self.gains[lo:hi]
                self.k['cache_gains']((((hi-lo)*s.P+127)//128,),(128,),
                    (states,self.pars,self.consts,self.SE,self.SI,self.WE,self.WI,target,np.int32(hi-lo)))
                if host_gain_cache:self.host_gains[lo:hi]=target.get()
            del states
        if host_gain_cache:
            if isinstance(self.host_gains,np.memmap):self.host_gains.flush()
            self.gains.set(self.host_gains[:len(self.gains)])
            del work_gains,target
        cp.cuda.get_current_stream().synchronize();cp.get_default_memory_pool().free_all_blocks()
        self.y=cp.zeros((NS,s.P));self.hist=cp.zeros((self.depth,s.P));self.arr=cp.zeros((4,s.P))
        self.f=cp.zeros_like(self.y);self.f2=cp.zeros_like(self.y);self.f3=cp.zeros_like(self.y);self.f4=cp.zeros_like(self.y)
        self.pred=cp.zeros_like(self.y);self.dr=cp.zeros(s.P);self.dr2=cp.zeros(s.P);self.offset=cp.zeros(1,dtype=cp.int32)
        self.order=cp.asarray((-np.arange(1,self.Dd+1))%self.depth);self.finalorder=cp.asarray((self.n-np.arange(1,self.Dd+1))%self.depth)
        self.stream=cp.cuda.Stream(non_blocking=True);cp.cuda.get_current_stream().synchronize()
        with self.stream:self.step(0)
        self.stream.synchronize();self.block=min(block,self.n);self.graphs={}
        for length in sorted(set([self.block,self.n%self.block])-{0}):
            with self.stream:
                self.stream.begin_capture()
                for i in range(length):self.step(i)
                self.k['advance']((1,),(1,),(self.offset,np.int32(length)))
                self.graphs[length]=self.stream.end_capture()
        self.calls=0;self.start=time.time();cp.get_default_memory_pool().free_all_blocks()

    def step(self,i):
        p=self.s.P;npop=(p+127)//128;nstate=(NS*p+127)//128
        def arrivals(stage):
            self.k['delayed_cubic']((p,),(128,),(*self.ops,self.hist,self.arr,self.offset,np.int32(i),float(stage),np.int32(self.depth),self.dt))
        def rhs(stage,state,f):
            self.k['cached_rhs']((npop,),(128,),(self.gains,self.Z,self.offset,np.int32(2*i+stage),state,self.arr,self.pars,self.consts,f,self.dr2))
        def pred(f,dt):self.k['tpredictor']((nstate,),(128,),(self.y,f,self.pred,dt))
        arrivals(0.);rhs(0,self.y,self.f);pred(self.f,.5*self.dt)
        arrivals(.5);rhs(1,self.pred,self.f2);pred(self.f2,.5*self.dt)
        rhs(1,self.pred,self.f3);pred(self.f3,self.dt)
        arrivals(1.);rhs(2,self.pred,self.f4)
        self.k['rkfinish']((nstate,),(128,),(self.y,self.f,self.f2,self.f3,self.f4,self.dt))
        rhs(2,self.y,self.f4);self.store(self.dr2,self.hist,self.offset,np.int32(i+1),np.int32(self.depth),size=p)

    def release_full_orbit(self):
        self.orbit=self.orbit[:1].copy();self.cp.get_default_memory_pool().free_all_blocks()

    def matvec(self,x):
        if not self.host_gain_cache:return super().matvec(x)
        cp=self.cp;p=self.s.P
        with self.stream:
            self.offset.fill(0);x=cp.asarray(x)
            self.y[:]=x[:NS*p].reshape(NS,p)
            self.hist[self.order]=x[NS*p:].reshape(self.Dd,p)
            self.y[11]=0.
            self.k['tangent_rhs'](((p+127)//128,),(128,),
                (self.orbit[0],self.y,self.arr,self.pars,self.consts,
                 self.SE,self.SI,self.WE,self.WI,self.f,self.dr))
            cp.copyto(self.hist[0],self.dr)
            for lo in range(0,self.n,self.block):
                length=min(self.block,self.n-lo)
                self.gains[:2*length+1].set(self.host_gains[2*lo:2*(lo+length)+1],stream=self.stream)
                self.graphs[length].launch(stream=self.stream)
                if getattr(self,'chunk_progress',False) and self.calls==0 and ((lo+length)//50000>lo//50000 or lo+length==self.n):
                    self.stream.synchronize()
                    log('VARIATIONAL FIRST MAP STEPS',lo+length,'/',self.n,
                        'seconds',round(time.time()-self.start,1))
            result=cp.concatenate([self.y.ravel(),self.hist[self.finalorder].ravel()])
        self.stream.synchronize();self.calls+=1
        return result.get()
