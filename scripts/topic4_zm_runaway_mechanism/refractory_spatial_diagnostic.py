"""Bounded whole-network diagnostic of the fixed conditioned rate candidate.

Local validation is still failed. These runs test the original spatial/native
question directly; they neither waive local gates nor launch bifurcation.
"""
from common import OUT,BASE,np,read,write,log,model
from conditioned_refractory_rate import DEST as LOCAL,load_models
from refractory_rate_cuda import LocalGPUResponse
from dynamics_v3 import Integrator
from run_network import group_drive
from shared_variance_network_sensitivity import split
from runner import summarize
from native_readouts import readouts,window_stats
from datetime import datetime
from pathlib import Path
import argparse,time,hashlib,os

DEST=OUT/'conditioned_refractory_spatial_diagnostic'

def code(P):
    return f'#define P {P}\n'+r'''
#include <curand_kernel.h>
extern "C" __global__ void delayed(const int* pa,const int* ca,const double* wa,const double* va,
 const int* pb,const int* cb,const double* wb,const double* vb,const double* hist,double* out,const int* clock,int depth,int factor){
 int g=blockIdx.x,lane=threadIdx.x,tick=clock[0]+1;double a=0.,b=0.,q=0.,v=0.;
 for(int i=pa[g]+lane;i<pa[g+1];i+=blockDim.x){int d=(ca[i]/P+1)*factor,slot=(tick-d)%depth;if(slot<0)slot+=depth;double r=hist[slot*P+ca[i]%P];a+=wa[i]*r;q+=va[i]*r;}
 for(int i=pb[g]+lane;i<pb[g+1];i+=blockDim.x){int d=(cb[i]/P+1)*factor,slot=(tick-d)%depth;if(slot<0)slot+=depth;double r=hist[slot*P+cb[i]%P];b+=wb[i]*r;v+=vb[i]*r;}
 __shared__ double buf[4][128];buf[0][lane]=a;buf[1][lane]=b;buf[2][lane]=q;buf[3][lane]=v;__syncthreads();
 for(int k=64;k>0;k/=2){if(lane<k)for(int j=0;j<4;j++)buf[j][lane]+=buf[j][lane+k];__syncthreads();}
 if(lane==0)for(int j=0;j<4;j++)out[j*P+g]=buf[j][0];
}
// syn=[qa,ia,qg,ig,m,Z], local=[6rawcovariance,36history].
extern "C" __global__ void physical_step(double* syn,const double* arr,const double* pars,const double* coefficient,
 const double* drive,int drive_on,int ndrive,const int* clock,double dt,double* physical){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 double tm=pars[g],areaA=pars[4*P+g],areaG=pars[5*P+g],pm=pars[6*P+g],pv=pars[7*P+g];
 if(drive_on){int index=min((int)(clock[0]*dt),ndrive-1);double nu=drive[(long long)index*P+g],j=pars[22*P+g];pm=tm*areaA*j*nu;pv=tm*areaA*areaA*j*j*nu;}
 for(int c=0;c<2;c++){
  double a=coefficient[3*c],b=coefficient[3*c+1],d=coefficient[3*c+2],force=tm*(c==0?areaA:areaG)*arr[c*P+g];
  double q=syn[(2*c)*P+g],I=syn[(2*c+1)*P+g];
  syn[(2*c)*P+g]=a*q+(1.-a)*force;syn[(2*c+1)*P+g]=b*q+d*I+(1.-d-b)*force;
 }
 physical[g]=syn[P+g]-syn[5*P+g]*syn[3*P+g]-syn[4*P+g]+pm;
 physical[P+g]=tm*areaA*areaA*arr[2*P+g]+pv;
 physical[2*P+g]=tm*areaG*areaG*arr[3*P+g];
}
extern "C" __global__ void finish(double* syn,const double* local,const double* rate,const double* pars,const double* constants,
 double* history,double* emitted,const int* clock,int depth,double dt,int noise,unsigned long long seed){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;int tick=clock[0]+1;double r=rate[g],E=pars[3*P+g];
 double em=exp(-dt/constants[6]),ez=exp(-dt/constants[7]);
 if(pars[20*P+g]>.5)syn[4*P+g]=em*syn[4*P+g]+(1.-em)*.5*E*r;
 double sd=sqrt(fmax(local[5*P+g],1e-20)),target=.5*erfc((syn[3*P+g]-constants[8])/(sqrt(2.)*sd));
 if(pars[19*P+g]>.5 && E>.5)syn[5*P+g]=ez*syn[5*P+g]+(1.-ez)*target;
 if(noise){curandStatePhilox4_32_10_t rng;curand_init(seed,g,(unsigned long long)tick,&rng);double n=pars[21*P+g];unsigned int count=curand_poisson(&rng,fmax(r,0.)*n*dt);r=(double)count/(n*dt);}
 emitted[g]=r;history[(long long)(tick%depth)*P+g]=r;
}
extern "C" __global__ void record(const double* emitted,const double* expected,double* accumulator,double* output,
 const int* clock,double dt,int per_ms,int chunk_ms){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;int tick=clock[0];
 accumulator[g]+=emitted[g]*dt;accumulator[P+g]+=expected[g]*dt;
 if(tick%per_ms==0){int row=(tick/per_ms-1)%chunk_ms;for(int c=0;c<2;c++){output[(row*2+c)*P+g]=accumulator[c*P+g]*1000.;accumulator[c*P+g]=0.;}}
}
'''

class SpatialEngine:
    def __init__(self,dt=.05,drive='seed9108401',noise=False,seed=1,device=0):
        self.s=s=model();s.set_Z(np.ones(s.P),source='physical zero state Z1');self.dt=dt;self.noise=noise;self.seed=seed
        import cupy as cp
        cp.cuda.Device(device).use();self.cp=cp
        shared=None;self.split_qa=[]
        if noise:
            operators,self.split_qa=split(s,dt);shared={k:(q,None,None) for k,q in operators.items()}
        depth=s.prep['max_delay_steps']*round(.1/dt)+1;assert depth>round(2/dt)
        self.transport=t=Integrator(s,dt=dt,history=np.zeros((depth,s.P)),dynamic_z=True,dynamic_m=True,
            drive=group_drive(s,drive) if drive!='mean' else None,noise=False,device=device,shared_split=shared)
        self.syn=cp.zeros((6,s.P));self.syn[5]=1.;physical=cp.zeros((3,s.P))
        self.local=LocalGPUResponse((~s.E).astype(int),s.theta,dt,depth,physical,Z=self.syn[5],device=device)
        assert np.array_equal(self.local.refractory.get(),s.ref)
        self.emitted=cp.zeros(s.P);self.accumulator=cp.zeros((2,s.P));self.output=cp.zeros((10,2,s.P))
        coeff=[]
        for tr,td in zip(s.rise,s.decay):
            a=np.exp(-dt/tr);d=np.exp(-dt/td);coeff.extend([a,tr/(tr-td)*(a-d),d])
        self.coefficients=cp.asarray(coeff)
        self.module=cp.RawModule(code=code(s.P),options=('--fmad=false',),name_expressions=['delayed','physical_step','finish','record'])
        self.k={name:self.module.get_function(name) for name in ['delayed','physical_step','finish','record']}

    def arrivals(self):
        t=self.transport;l=self.local
        self.k['delayed']((self.s.P,),(128,),(*t.ops,t.history,t.arr,l.clock,np.int32(t.depth),np.int32(t.factor)))

    def step(self):
        t=self.transport;l=self.local;n=(self.s.P+127)//128
        self.arrivals()
        self.k['physical_step']((n,),(128,),(self.syn,t.arr,t.pars,self.coefficients,t.drive,np.int32(t.drive_on),np.int32(t.n_drive),l.clock,self.dt,l.physical))
        l.update()
        self.k['finish']((n,),(128,),(self.syn,l.state,l.rate,t.pars,t.consts,t.history,self.emitted,l.clock,np.int32(t.depth),self.dt,np.int32(self.noise),np.uint64(self.seed)))
        l.advance((1,),(1,),(l.clock,))
        self.k['record']((n,),(128,),(self.emitted,l.rate,self.accumulator,self.output,l.clock,self.dt,np.int32(round(1/self.dt)),np.int32(10)))

    def reset(self):
        self.syn.fill(0);self.syn[5]=1.;self.local.state.fill(0);self.local.history.fill(0);self.local.rate.fill(0);self.local.clock.fill(0)
        self.transport.history.fill(0);self.transport.arr.fill(0);self.local.physical.fill(0);self.emitted.fill(0);self.accumulator.fill(0);self.output.fill(0)

    def graph(self):
        self.step();self.cp.cuda.get_current_stream().synchronize();self.reset();self.cp.cuda.get_current_stream().synchronize()
        self.stream=self.cp.cuda.Stream(non_blocking=True)
        with self.stream:
            self.stream.begin_capture()
            for _ in range(round(10/self.dt)):self.step()
            self.graph_object=self.stream.end_capture()

    def chunk(self):
        self.graph_object.launch(self.stream);self.stream.synchronize();return self.output.get()

def register():
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    assert read(LOCAL/'independent_audit.json')['status']=='PASS'
    assert read(LOCAL/'spatial_implementation/local_cuda_parity.json')['status']=='PASS'
    write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        status='REGISTERED_BEFORE_NETWORK_DIAGNOSTIC',question='How do the remaining measured local-response errors affect autonomous interictal events,Z/M entry and2D recruitment in the fixed network?',
        reason='Original goal concerns native spatial dynamics. Local correction improved23/24oldwaveforms but stillfails; bounded end-to-end diagnosis informs the next repair without promoting this model or certifying bifurcation.',
        fixed='common.model(grid20),same graph,group sizes,thresholds,delays and external input definitions. No connectivity,resource,adaptation or fitted response change.',
        rate='Frozen conditioned_refractory_rate weights.6raw currentcovariance+36inputhistory states and own-rate refractory integral. Synaptic means retain2rise-decay pairs. Z multiplies GABA mean and postfilter covariance; rawGABA variance drives Z target.',
        slow='BothZandMdynamic inallruns; m=eta_M*M,dm/dt=(.5*E*r-m)/1000ms; Z follows unchanged Gaussian threshold target/5000ms; I Z=1.',
        clocks='dt.05ms; delayed arrival uses newstep endpoint, recorded externaldrive is held on each1ms input interval. Exact linear synaptic,covariance,history updates and slow-target exponential steps; refractory implicit Euler.',
        noise='Two contrasts inherit the old Poisson finite-group emission approximation and delay-aware stationary shared/private variance subtraction. Own refractory history and M use expected rate; synaptic arrivals use sampled counts. This is not an exact renewal-noise reduction and its limitations remain explicit.',
        initialization='All synaptic/covariance/history states and past firing zero; Z1,M0. No SNN future activity or state transplant.',
        runs=[dict(label='mean_drive_expected',drive='mean',noise=False,seed=1),dict(label='recorded_drive_expected',drive='seed9108401',noise=False,seed=1),
              dict(label='recorded_drive_poisson_seed1',drive='seed9108401',noise=True,seed=1),dict(label='recorded_drive_poisson_seed2',drive='seed9108401',noise=True,seed=2)],
        duration_ms=12500.,dt_ms=.05,implementation_gate='Local CUDA parity plus same delayed operators and independent one-step physical/slow-state checks before full runs.',
        readouts='Original native event,quiet occupancy,duration,bothcore participation,forward/reverse propagation,spatial area,entry andsame-clockD definitions. Expected versus emitted rates stored separately; compare0.5-3,4-8,8-9.42,andfull12.5s.',
        boundaries='Diagnostic only: local acceptance stillFAIL; no newbifurcation label,no formalmodelpromotion,no onsettimefitting or adaptiveparametersearch. If unstable/nonfinite, stop thatrun and report it; no hiddenclamp or parameterrepair.'))

def check(device):
    from conditioned_refractory_rate import load_models
    e=SpatialEngine(device=device);s=e.s;cp=e.cp;t=e.transport;l=e.local
    rng=np.random.default_rng(920080);t.history[:]=cp.asarray(rng.uniform(0,.002,t.history.shape));errors=[]
    for tick in [0,t.depth-2,t.depth+7]:
        l.clock.fill(tick);t.arrivals(tick+1);reference=t.arr.get();e.arrivals();observed=t.arr.get()
        assert np.array_equal(reference,observed);errors.append(dict(tick=tick,delayed_bitwise=True))
    e.reset();e.graph()
    prefix=e.chunk();assert np.isfinite(prefix).all()
    # Independent CPU one-step update at the actual active prefix state.
    syn=e.syn.get();oldlocal=l.state.get();history=l.history.get();tick=int(l.clock.get()[0]);e.arrivals();arr=t.arr.get();pars=t.pars.get();con=t.consts.get();coeff=e.coefficients.get()
    expected_syn=syn.copy();pm=pars[6].copy();pv=pars[7].copy()
    if t.drive_on:
        nu=t.drive[e.transport.drive_index(tick)].get();pm=s.tm*s.area[0]*s.jext*nu;pv=s.tm*s.area[0]**2*s.jext**2*nu
    for c in range(2):
        a,b,d=coeff[3*c:3*c+3];F=s.tm*s.area[c]*arr[c];q,I=syn[2*c:2*c+2]
        expected_syn[2*c]=a*q+(1-a)*F;expected_syn[2*c+1]=b*q+d*I+(1-d-b)*F
    physical=np.array([expected_syn[1]-syn[5]*expected_syn[3]-syn[4]+pm,s.tm*s.area[0]**2*arr[2]+pv,s.tm*s.area[1]**2*arr[3]])
    local=oldlocal.copy();u=np.zeros((3,s.P));f=np.zeros((s.P,39));net,bases,_=load_models();expected_rate=np.empty(s.P)
    from refractory_rate_response import covariance_matrices
    for pop,mask in [('E',s.E),('I',~s.E)]:
        A,B,C,_,_=covariance_matrices(pop,e.dt)
        for c in range(2):local[3*c:3*c+3,mask]=A[c]@oldlocal[3*c:3*c+3,mask]+B[c,:,None]*physical[c+1,mask]
        scale=s.theta[mask]-11;vA=C[0]*local[2,mask];vG=C[1]*local[5,mask]*syn[5,mask]**2
        u[:,mask]=np.array([np.arcsinh((physical[0,mask]-11)/scale)/3,np.log1p(vA/scale**2)/2,np.log1p(vG/scale**2)/2])
    bank=l.bank.get()
    for c in range(3):
        for j in range(4):
            k=6+12*c+3*j;a,b,b2,f1,f2,f3=bank[j];h1,h2,h3=oldlocal[k:k+3]
            local[k]=a*h1+f1*u[c];local[k+1]=a*(h2+b*h1)+f2*u[c];local[k+2]=a*(h3+b*h2+b2*h1)+f3*u[c]
            f[:,3+12*c+3*j:3+12*c+3*j+3]=(local[k:k+3]-u[c]).T
    f[:,:3]=u.T
    import torch
    from nonlinear_rate_response import physical_from_features
    for pop,mask in [('E',s.E),('I',~s.E)]:
        base=bases[pop].evaluate(physical_from_features(f[mask],s.theta[mask]),s.theta[mask])
        with torch.no_grad():ell=net[pop].logits(torch.tensor(f[mask]),torch.tensor(base)).numpy()
        pr=1/(1+np.exp(-(ell+np.log(e.dt/.1))));nref=round(float(s.ref[mask][0])/e.dt)
        available=1-e.dt*history[(tick+1-np.arange(1,nref))%t.depth][:,mask].sum(0);expected_rate[mask]=available*pr/e.dt
    from scipy.special import ndtr
    target=ndtr((con[8]-expected_syn[3])/np.sqrt(np.maximum(local[5],1e-20)))
    em=np.exp(-e.dt/con[6]);ez=np.exp(-e.dt/con[7]);expected_syn[4]=em*syn[4]+(1-em)*.5*s.E*expected_rate;expected_syn[5,s.E]=ez*syn[5,s.E]+(1-ez)*target[s.E]
    e.step();cp.cuda.get_current_stream().synchronize()
    pe=float(np.max(abs(l.physical.get()-physical)));le=float(np.max(abs(l.state.get()-local)));re=float(np.max(abs(l.rate.get()-expected_rate))*1000);se=float(np.max(abs(e.syn.get()-expected_syn)))
    assert pe<1e-9 and le<1e-8 and re<1e-5 and se<1e-8,(pe,le,re,se)
    result=dict(status='PASS',P=s.P,delays=errors,physical_moment_error=pe,local_state_error=le,rate_error_hz=re,slow_synaptic_error=se,
        initial_prefix_ms=10,Z_and_M_dynamic=True,scope='Implementation check only; no native correspondence or new bifurcation acceptance.')
    write(DEST/'implementation_check.json',result);log('SPATIAL REFRACTORY IMPLEMENTATION PASS',result)

def run(device):
    c=read(DEST/'contract.json');assert read(DEST/'implementation_check.json')['status']=='PASS'
    progress=dict(status='RUNNING',expected=4,completed=[],pid=os.getpid());assert not (DEST/'jobs.json').exists();write(DEST/'jobs.json',progress)
    for row in c['runs']:
        folder=DEST/row['label'];folder.mkdir();e=SpatialEngine(dt=c['dt_ms'],drive=row['drive'],noise=row['noise'],seed=row['seed'],device=device);s=e.s;e.graph()
        R=[];Z=[];M=[];start=time.time()
        write(folder/'identity.json',dict(graph_identity=s.prep['graph_identity'],response=str(LOCAL/'fit/locked_weights.json'),groups=s.P,spatial_grid=s.grid,shared_split_qa=e.split_qa))
        for k in range(round(c['duration_ms']/10)):
            output=e.chunk();assert np.isfinite(output).all(),(row['label'],k,'NONFINITE_RATE')
            R.append(output);Z.append(e.syn[5].get());M.append(e.syn[4].get())
            assert Z[-1].min()>=-1e-10 and Z[-1].max()<=1+1e-10
            if (k+1)%100==0:
                log('REFRACTORY SPATIAL',row['label'],(k+1)*10,'ms','seconds',round(time.time()-start,1));write(folder/'progress.json',dict(status='RUNNING',time_ms=(k+1)*10))
        rates=np.concatenate(R);z=np.array(Z);m=np.array(M);_,field,whole,count=summarize(rates[:,0],s);tms=np.arange(len(rates))+1.;ts=(np.arange(len(z))+1)*10.;D=1-z[:,s.E]@s.mean_weights
        events,summary,_,_=readouts(tms,field,count,row['label']);summary['windows']={f'{a}-{b}':window_stats(events,a,b) for a,b in [(500,3000),(4000,8000),(8000,9420),(1000,12500)]}
        np.savez_compressed(folder/'trajectory.npz',time_ms=tms,group_rate_hz=rates[:,0].astype('f4'),group_expected_rate_hz=rates[:,1].astype('f4'),
            field_E_hz=field.astype('f4'),global_E_hz=whole,cell_counts=count,Z=z.astype('f4'),M_current=m.astype('f4'),D=D,state_time_ms=ts,
            final_synaptic_slow_state=e.syn.get(),final_local_state=e.local.state.get(),final_own_history=e.local.history.get(),final_emitted_history=e.transport.history.get(),final_tick=e.local.clock.get(),dt_ms=e.dt)
        summary.update(status='COMPLETE',label=row['label'],duration_ms=c['duration_ms'],Z_and_M_dynamic=True,D8000=float(D[799]),D9870=float(D[986]),D_final=float(D[-1]),seconds=time.time()-start,model_promoted=False)
        summary['events']=[{k:v for k,v in ev.items() if k!='onset'} for ev in events];write(folder/'result.json',summary)
        progress['completed'].append(row['label']);write(DEST/'jobs.json',progress);log('REFRACTORY SPATIAL COMPLETE',row['label'],summary['high_onset_ms'],summary['D9870'])
        del e
    progress['status']='COMPLETE';write(DEST/'jobs.json',progress)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run']);p.add_argument('--device',type=int,default=0);a=p.parse_args();{'register':register,'check':lambda:check(a.device),'run':lambda:run(a.device)}[a.command]()
