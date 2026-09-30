"""Same endpoint variational integrator, with a reusable small CUDA graph.

Only graph scheduling differs from EndpointMonodromy. A device tick supplies
the absolute orbit/history indices so a long period need not be captured as
hundreds of thousands of distinct graph nodes.
"""
from phase_audit import EndpointMonodromy
from floquet_v3 import *


class ChunkEndpointMonodromy(EndpointMonodromy):
    def __init__(self,s,o,sol,dtmax=.1,dynamic_z=False,device=0,use_gradients=True,block=128):
        import cupy as cp
        self.cp=cp;self.s=s;self.dynamic_z=dynamic_z;cp.cuda.Device(device).use()
        self.T=sol['T'];self.n=int(np.ceil(self.T/dtmax));self.dt=self.T/self.n
        self.Dd=int(np.ceil(s.delays[-1]/self.dt))+1;self.depth=self.Dd+1;self.dim=(NS+self.Dd)*s.P
        full,self.rr=orbit_states(o,sol,self.n);self.orbit=cp.asarray(np.ascontiguousarray(full))
        self.ops=[]
        for kind in ['ampa','gaba']:
            a=sparse.load_npz(s.folder/f'mean_{kind}.npz').tocsr();q=sparse.load_npz(s.folder/f'variance_{kind}.npz').tocsr()
            self.ops.extend([cp.asarray(a.indptr,dtype=cp.int32),cp.asarray(a.indices,dtype=cp.int32),cp.asarray(a.data),cp.asarray(q.data)])
        self.pars,self.consts,self.SE,self.SI,self.WE,self.WI=model_device_arrays(s,cp,dynamic_z=dynamic_z)
        self.consts=cp.concatenate([self.consts,cp.asarray([1. if use_gradients else 0.])])
        code=TANGENT.replace('int tick,int depth,double dt){','const int* offset,int tick,int depth,double dt){\n tick+=*offset;')
        code=code.replace('double dt,int tick,int depth){','double dt,const int* offset,int tick,int depth){\n tick+=*offset;')
        start=code.index('extern "C" __global__ void tangent_rhs')
        end=code.index('extern "C" __global__ void tpredictor')
        indexed=code[start:end].replace('void tangent_rhs(const double* orbit,','void tangent_indexed(const double* allorbit,const int* offset,int local,')
        indexed=indexed.replace(' int g=blockIdx.x*blockDim.x+threadIdx.x;',' const double* orbit=allorbit+((*offset+local)*14*P);\n int g=blockIdx.x*blockDim.x+threadIdx.x;',1)
        code+=indexed+'\nextern "C" __global__ void advance(int* offset,int n){*offset+=n;}\n'
        names=['delayed_linear','tangent_rhs','tangent_indexed','tpredictor','tfinish','advance']
        mod=cp.RawModule(code=f'#define P {s.P}\n'+CUDA_DEVICE+CUDA_RESP+code,options=('--fmad=false',),name_expressions=names)
        self.k={n:mod.get_function(n) for n in names};self.y=cp.zeros((NS,s.P));self.hist=cp.zeros((self.depth,s.P))
        self.arr=cp.zeros((4,s.P));self.f=cp.zeros_like(self.y);self.f2=cp.zeros_like(self.y);self.pred=cp.zeros_like(self.y);self.dr=cp.zeros(s.P);self.dr2=cp.zeros(s.P)
        self.offset=cp.zeros(1,dtype=cp.int32)
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
        p=self.s.P;n=(p+127)//128
        def rhs(stage,state,f,dr):
            self.k['tangent_indexed']((n,),(128,),(self.orbit,self.offset,np.int32(stage),state,self.arr,self.pars,self.consts,self.SE,self.SI,self.WE,self.WI,f,dr))
        self.k['delayed_linear']((p,),(128,),(*self.ops,self.hist,self.arr,self.offset,np.int32(i),np.int32(self.depth),self.dt))
        rhs(i,self.y,self.f,self.dr)
        self.k['tpredictor'](((NS*p+127)//128,),(128,),(self.y,self.f,self.pred,self.dt))
        self.k['delayed_linear']((p,),(128,),(*self.ops,self.hist,self.arr,self.offset,np.int32(i+1),np.int32(self.depth),self.dt))
        rhs(i+1,self.pred,self.f2,self.dr2)
        self.k['tfinish']((n,),(128,),(self.y,self.f,self.f2,self.dr,self.dr2,self.hist,self.dt,self.offset,np.int32(i+1),np.int32(self.depth)))
        rhs(i+1,self.y,self.f2,self.dr2)
        # Slot depends on absolute time, so use a tiny indexed copy kernel.
        self.store(self.dr2,self.hist,self.offset,np.int32(i+1),np.int32(self.depth),size=p)

    @property
    def store(self):
        if not hasattr(self,'_store'):
            self._store=self.cp.ElementwiseKernel('raw float64 rate, raw int32 offset, int32 local, int32 depth','raw float64 hist',
                'hist[((offset[0]+local)%depth)*P+i]=rate[i];','endpoint_history',preamble=f'#define P {self.s.P}')
            # reorder wrapper to keep step's arguments easy to audit
            self._store_call=lambda rate,hist,offset,local,depth,size:self._store(rate,offset,local,depth,hist,size=size)
        return self._store_call

    def matvec(self,x):
        cp=self.cp;p=self.s.P
        with self.stream:
            self.offset.fill(0);x=cp.asarray(x);self.y[:]=x[:NS*p].reshape(NS,p);self.hist[self.order]=x[NS*p:].reshape(self.Dd,p)
            if not self.dynamic_z:self.y[11]=0.
            self.k['tangent_rhs'](((p+127)//128,),(128,),(self.orbit[0],self.y,self.arr,self.pars,self.consts,self.SE,self.SI,self.WE,self.WI,self.f,self.dr))
            cp.copyto(self.hist[0],self.dr)
            for _ in range(self.n//self.block):self.graphs[self.block].launch(stream=self.stream)
            if self.n%self.block:self.graphs[self.n%self.block].launch(stream=self.stream)
            result=cp.concatenate([self.y.ravel(),self.hist[self.finalorder].ravel()])
        self.stream.synchronize();self.calls+=1
        if self.calls%10==0:log('CHUNK MONODROMY',self.calls,'sec',round(time.time()-self.start,1))
        return result.get()


if __name__=='__main__':
    from native_path import *
    from streaming_periodic import StreamPeriodic
    import argparse,gc
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);a=p.parse_args()
    s=model();attach_native_path(s);z=np.load(OUT/'periodic/seed_N512.npz');sol=dict(r=z['r'],T=float(z['T']),D=float(z['D']))
    o=StreamPeriodic(s,512,a.device);o.cache_mean_operators=False
    old=EndpointMonodromy(s,o,sol,dtmax=.1,device=a.device)
    x=np.random.default_rng(845).normal(size=old.dim);x[11*s.P:12*s.P]=0.
    expected=old.matvec(x);del old;gc.collect();o.cp.get_default_memory_pool().free_all_blocks()
    new=ChunkEndpointMonodromy(s,o,sol,dtmax=.1,device=a.device)
    actual=new.matvec(x);err=float(abs(actual-expected).max());rel=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    assert rel<1e-12,(err,rel)
    write(OUT/'chunk_monodromy_check.json',dict(status='PASS',max_error=err,relative_error=rel,bitwise_equal=bool(np.array_equal(actual,expected)),dt=new.dt,n=new.n,block=new.block))
    log('CHUNK PARITY',err,rel)
