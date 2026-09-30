"""Reuse identical delay/source interpolation across all incoming edges.

No approximation: interpolation coefficients, history endpoints, CSR reduction
order and RK4 stages match RK4Monodromy. The delay/source table is only a cache.
"""
from rk4_monodromy import RK4Monodromy, np, NS


CODE=r'''
extern "C" __global__ void interpolate_history(const double* hist,double* table,
 const int* offset,int local,double stage,int depth,double dt,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 int tick=*offset+local,h=i%P;
 double d=(i/P+1)*.1/dt-stage;int lag=(int)floor(d);double f=d-lag;
 double c0=-f*(1-f)*(2-f)/6.,c1=(1+f)*(1-f)*(2-f)/2.;
 double c2=(1+f)*f*(2-f)/2.,c3=-(1+f)*f*(1-f)/6.;
 int s0=(tick-lag+1)%depth;if(s0<0)s0+=depth;int s1=(s0+depth-1)%depth;
 int s2=(s1+depth-1)%depth,s3=(s2+depth-1)%depth;
 table[i]=c0*hist[s0*P+h]+c1*hist[s1*P+h]+c2*hist[s2*P+h]+c3*hist[s3*P+h];
}
extern "C" __global__ void arrivals_from_table(const int* pa,const int* ca,const double* wa,const double* va,
 const int* pb,const int* cb,const double* wb,const double* vb,const double* table,double* out){
 int g=blockIdx.x,lane=threadIdx.x;double a=0.,b=0.,q=0.,v=0.;
 for(int i=pa[g]+lane;i<pa[g+1];i+=blockDim.x){
  double r=table[ca[i]];a+=wa[i]*r;q+=va[i]*r;
 }
 for(int i=pb[g]+lane;i<pb[g+1];i+=blockDim.x){
  double r=table[cb[i]];b+=wb[i]*r;v+=vb[i]*r;
 }
 __shared__ double buf[4][128];buf[0][lane]=a;buf[1][lane]=b;buf[2][lane]=q;buf[3][lane]=v;__syncthreads();
 for(int k=64;k>0;k/=2){if(lane<k)for(int j=0;j<4;j++)buf[j][lane]+=buf[j][lane+k];__syncthreads();}
 if(lane==0)for(int j=0;j<4;j++)out[j*P+g]=buf[j][0];
}
'''


class PreinterpolatedRK4(RK4Monodromy):
    def __init__(self,s,*args,device=0,**kwargs):
        import cupy as cp
        cp.cuda.Device(device).use()
        self.delay_table=cp.empty((int(np.ceil(float(s.delays[-1])/.1))+1,s.P))
        module=cp.RawModule(code=f'#define P {s.P}\n'+CODE,options=('--fmad=false',),
            name_expressions=['interpolate_history','arrivals_from_table'])
        self.interpolate=module.get_function('interpolate_history')
        self.table_arrivals=module.get_function('arrivals_from_table')
        super().__init__(s,*args,device=device,**kwargs)
        assert max(int(self.ops[j].max()) for j in [1,5])<self.delay_table.size

    def arrivals(self,i,stage):
        p=self.s.P
        self.interpolate(((self.delay_table.size+127)//128,),(128,),
            (self.hist,self.delay_table,self.offset,np.int32(i),float(stage),
             np.int32(self.depth),self.dt,np.int32(self.delay_table.size)))
        self.table_arrivals((p,),(128,),(*self.ops,self.delay_table,self.arr))

    def step(self,i):
        p=self.s.P;npop=(p+127)//128;nstate=(NS*p+127)//128
        def rhs(stage,state,f):
            self.k['cached_rhs']((npop,),(128,),
                (self.gains,self.Z,self.offset,np.int32(2*i+stage),state,self.arr,self.pars,self.consts,f,self.dr2))
        def pred(f,dt):self.k['tpredictor']((nstate,),(128,),(self.y,f,self.pred,dt))
        self.arrivals(i,0.);rhs(0,self.y,self.f);pred(self.f,.5*self.dt)
        self.arrivals(i,.5);rhs(1,self.pred,self.f2);pred(self.f2,.5*self.dt)
        rhs(1,self.pred,self.f3);pred(self.f3,self.dt)
        self.arrivals(i,1.);rhs(2,self.pred,self.f4)
        self.k['rkfinish']((nstate,),(128,),(self.y,self.f,self.f2,self.f3,self.f4,self.dt))
        rhs(2,self.y,self.f4);self.store(self.dr2,self.hist,self.offset,np.int32(i+1),np.int32(self.depth),size=p)
