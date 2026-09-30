"""GPU target ownership preserves the CPU recurrent addition order."""
from shared_source import *
import cupy as cp

CODE=r'''
extern "C" __global__ void update_cells(
 const long long* ptr,const int* reg,const double* th,const int* exidx,
 const int* fieldidx,const double* weights,const int* rasteridx,const unsigned char* ext,
 const double* p,double* V,int* ref,double* gE,double* cE,double* gI,double* cI,
 double* ringE,double* ringI,int* spike,unsigned int* counts,double* raw,unsigned int* field,unsigned char* raster,
 int N,int P,int M,int Q,int R,int t,int k) {
 int a=blockIdx.x;if(a>=P)return;
 if(threadIdx.x==0)spike[a]=0;__syncthreads();
 int r=reg[a];bool excit=r<3;double decay=excit?p[5]:p[6];int refractory=(int)(excit?p[7]:p[8]);
 double increment=excit?p[9]:p[10];int slot=t%M;int frame=k/20;
 for(long long i=ptr[a]+threadIdx.x;i<ptr[a+1];i+=blockDim.x){
  long long ri=(long long)slot*N+i;
  gE[i]=gE[i]*p[1]+ringE[ri];gI[i]=gI[i]*p[3]+ringI[ri];ringE[ri]=0.;ringI[ri]=0.;
  double drive=exidx[i]>=0?(double)ext[(long long)k*Q+exidx[i]]:p[11]*p[0];gE[i]+=drive*increment;
  cE[i]=gE[i]+(cE[i]-gE[i])*p[2];cI[i]=gI[i]+(cI[i]-gI[i])*p[4];ref[i]=max(0,ref[i]-1);
  if(ref[i]==0){double net=cE[i]-cI[i];V[i]=net+(V[i]-net)*decay;
   if(V[i]>=th[i]){V[i]=p[12];ref[i]=refractory;atomicAdd(spike+a,1);atomicAdd(counts+(long long)frame*P+a,1u);
    if(excit){atomicAdd(field+(long long)frame*400+fieldidx[i],1u);for(int j=0;j<15;j++)atomicAdd(raw+(long long)frame*15+j,weights[i*15+j]);}
    if(rasteridx[i]>=0)raster[(long long)k*R+rasteridx[i]]=1;
   }
  }else V[i]=p[12];
 }
}
extern "C" __global__ void incoming(
 const long long* ptr,const int* src,const short* delay,const double* weight,const unsigned char* kind,
 const int* spike,double* ringE,double* ringI,int N,int M,int t){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
 for(long long j=ptr[i];j<ptr[i+1];j++){
  int count=spike[src[j]];if(!count)continue;long long index=(long long)((t+delay[j])%M)*N+i;
  double value=__dmul_rn(weight[j],(double)count);
  if(kind[j])ringI[index]=__dadd_rn(ringI[index],value);else ringE[index]=__dadd_rn(ringE[index],value);
 }
}
extern "C" __global__ void moments(
 const long long* ptr,const double* V,const int* ref,const double* cE,const double* cI,double* out,int P,int k){
 int a=blockIdx.x*blockDim.x+threadIdx.x;if(a>=P)return;
 double v=0.,v2=0.,re=0.,e=0.,e2=0.,inh=0.,ei=0.;
 for(long long i=ptr[a];i<ptr[a+1];i++){v+=V[i];v2+=V[i]*V[i];re+=(ref[i]>0);e+=cE[i];e2+=cE[i]*cE[i];inh+=cI[i];ei+=cE[i]*cI[i];}
 double n=(double)(ptr[a+1]-ptr[a]);long long base=((long long)(k/100)*P+a)*7;
 out[base]=v/n;out[base+1]=re/n;out[base+2]=e/n;out[base+3]=fmax(0.,e2/n-(e/n)*(e/n));
 out[base+4]=inh/n;out[base+5]=ei/n-(e/n)*(inh/n);out[base+6]=fmax(0.,v2/n-(v/n)*(v/n));
}
'''

class GPU:
    def __init__(self,ptr,reg,threshold,ext_index,field_index,weights,raster_index,ampa,gaba,p,state,device=0):
        cp.cuda.Device(device).use();self.N=len(state[0]);self.P=len(ptr)-1;self.M=state[-1].shape[0];self.R=int(np.max(raster_index))+1
        self.ptr=cp.asarray(ptr,dtype=np.int64);self.reg=cp.asarray(reg,dtype=np.int32);self.threshold=cp.asarray(threshold,dtype=np.float64)
        self.ext_index=cp.asarray(ext_index,dtype=np.int32);self.field_index=cp.asarray(field_index,dtype=np.int32)
        self.weights=cp.asarray(weights,dtype=np.float64);self.raster_index=cp.asarray(raster_index,dtype=np.int32);self.p=cp.asarray(p)
        self.state=tuple(cp.asarray(a) for a in state);self.spike=cp.zeros(self.P,np.int32)
        source=np.concatenate([np.repeat(np.arange(self.P,dtype=np.int32),np.diff(op[0])) for op in (ampa,gaba)])
        target=np.concatenate([op[1] for op in (ampa,gaba)]);delay=np.concatenate([op[2] for op in (ampa,gaba)])
        weight=np.concatenate([op[3] for op in (ampa,gaba)])
        kind=np.r_[np.zeros(len(ampa[1]),np.uint8),np.ones(len(gaba[1]),np.uint8)]
        order=np.argsort(target,kind='stable');incptr=np.r_[0,np.cumsum(np.bincount(target,minlength=self.N))].astype(np.int64)
        self.incoming=[cp.asarray(x) for x in [incptr,source[order],delay[order],weight[order],kind[order]]]
        module=cp.RawModule(code=CODE,options=('--fmad=false',),name_expressions=['update_cells','incoming','moments'])
        self.update=module.get_function('update_cells');self.scatter=module.get_function('incoming');self.moments=module.get_function('moments')

    def evolve(self,ext,start):
        ext=cp.asarray(ext);steps=len(ext);assert steps%100==0
        counts=cp.zeros((steps//20,self.P),np.uint32);raw=cp.zeros((steps//20,15),np.float64);field=cp.zeros((steps//20,400),np.uint32)
        raster=cp.zeros((steps,self.R),np.uint8);moments=cp.zeros((steps//100,self.P,7),np.float64)
        V,ref,gE,cE,gI,cI,ringE,ringI=self.state
        args=(self.ptr,self.reg,self.threshold,self.ext_index,self.field_index,self.weights,self.raster_index,ext,self.p,
            *self.state,self.spike,counts,raw,field,raster,np.int32(self.N),np.int32(self.P),np.int32(self.M),np.int32(ext.shape[1]),np.int32(self.R))
        for k in range(steps):
            t=np.int32(start+k);kk=np.int32(k)
            self.update((self.P,),(128,),args+(t,kk))
            self.scatter(((self.N+127)//128,),(128,),(*self.incoming,self.spike,ringE,ringI,np.int32(self.N),np.int32(self.M),t))
            if (k+1)%100==0:self.moments(((self.P+127)//128,),(128,),(self.ptr,V,ref,cE,cI,moments,np.int32(self.P),kk))
        return counts.get(),raw.get(),field.get(),raster.get().astype(bool),moments.get()

    def host_state(self):return tuple(a.get() for a in self.state)
