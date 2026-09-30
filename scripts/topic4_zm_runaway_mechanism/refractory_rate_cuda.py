"""GPU realization of the fixed local rate equations, for spatial integration.

State layout: six raw current covariances, then36input-history states.
Own expected spike history determines refractory occupancy. No particles.
"""
from conditioned_refractory_rate import load_models,DEST,np,write,read,log,covariance_matrices,PARAMS
from common import OUT
from transfer_spline import device_block,CUDA_DEVICE
from nonlinear_rate_response import Baseline
import argparse

def packed_network(net):
    layers=net.network.layers;A=net.network.transform.detach().numpy();center=net.network.center.detach().numpy()
    W=layers[0].weight.detach().numpy()@A;b=layers[0].bias.detach().numpy()-W@center
    return np.concatenate([W.ravel(),b,layers[2].weight.detach().numpy().ravel(),layers[2].bias.detach().numpy(),layers[4].weight.detach().numpy().ravel(),layers[4].bias.detach().numpy()])

def source(P):
    return f'#define P {P}\n'+CUDA_DEVICE+r'''
extern "C" __global__ void local_rate(double* state,const double* physical,const double* Z,
 const double* theta,const int* pop,const double* refractory,const double* coefficients,const double* bank,
 const double* weights,const double* SE,const double* SI,double* own_history,double* rate,
 const int* clock,int depth,double dt,int streamed_input){
 int g=blockIdx.x,lane=threadIdx.x,p=pop[g],tick=clock[0]+1;
 const double* input=physical+(streamed_input?(long long)clock[0]*3*P:0);
 const double* w=weights+p*6785;
 __shared__ double f[39],hidden[64],hidden2[64],baseline;
 if(lane==0){
  double mu=input[g],raw[2]={input[P+g],input[2*P+g]},v[2];
  for(int c=0;c<2;c++){
   const double* a=coefficients+(p*2+c)*13;
   double x[3]={state[(3*c)*P+g],state[(3*c+1)*P+g],state[(3*c+2)*P+g]};
   for(int j=0;j<3;j++)state[(3*c+j)*P+g]=a[j*3]*x[0]+a[j*3+1]*x[1]+a[j*3+2]*x[2]+a[9+j]*raw[c];
   v[c]=a[12]*state[(3*c+2)*P+g];
  }
  v[1]*=Z[g]*Z[g];double scale=theta[g]-11.;
  f[0]=asinh((mu-11.)/scale)/3.;f[1]=log1p(v[0]/(scale*scale))/2.;f[2]=log1p(v[1]/(scale*scale))/2.;
  for(int c=0;c<3;c++)for(int j=0;j<4;j++){
   const double* a=bank+j*6;int k=6+c*12+j*3;
   double h1=state[k*P+g],h2=state[(k+1)*P+g],h3=state[(k+2)*P+g],u=f[c];
   state[k*P+g]=a[0]*h1+a[3]*u;
   state[(k+1)*P+g]=a[0]*(h2+a[1]*h1)+a[4]*u;
   state[(k+2)*P+g]=a[0]*(h3+a[1]*h2+a[2]*h1)+a[5]*u;
   for(int l=0;l<3;l++)f[3+c*12+j*3+l]=state[(k+l)*P+g]-u;
  }
  double r,d1,d2,d3;phi_spline(p==0?SE:SI,mu,v[0],v[1],theta[g],&r,&d1,&d2,&d3);
  double maximum=1./refractory[g],a=r/maximum;
  double logden=log1p(pow(a,16.));double capped=a*exp(-logden/16.);
  double probability=1e-8+(1.-2e-8)*capped;
  double q=maximum*probability*.1,available=1.-(refractory[g]/.1-1.)*q,pr=q/available;
  baseline=log(pr)-log1p(-pr);
 }
 __syncthreads();
 double x=w[2496+lane];for(int j=0;j<39;j++)x+=w[lane*39+j]*f[j];hidden[lane]=tanh(x);
 __syncthreads();
 x=w[6656+lane];for(int j=0;j<64;j++)x+=w[2560+lane*64+j]*hidden[j];hidden2[lane]=tanh(x);
 __syncthreads();
 if(lane==0){
  double ell=baseline+w[6784];for(int j=0;j<64;j++)ell+=w[6720+j]*hidden2[j];
  double occupied=0.;int nref=(int)llround(refractory[g]/dt);
  for(int j=1;j<nref;j++){int slot=(tick-j)%depth;if(slot<0)slot+=depth;occupied+=own_history[(long long)slot*P+g]*dt;}
  double available=1.-occupied,l=ell+log(dt/.1),pr=l>=0?1./(1.+exp(-l)):exp(l)/(1.+exp(l));
  double r=available*pr/dt;
  if(available < -1e-10 || available > 1.+1e-10)r=nan("");
  rate[g]=r;own_history[(long long)(tick%depth)*P+g]=r;
 }
}
extern "C" __global__ void advance_clock(int* clock){if(threadIdx.x==0)clock[0]++;}
'''

class LocalGPUResponse:
    def __init__(self,pop,theta,dt,depth,physical,Z=None,device=0):
        import cupy as cp
        cp.cuda.Device(device).use();self.cp=cp;self.P=len(pop);self.dt=dt;self.depth=depth
        nets,_,_=load_models();self.network=cp.asarray(np.array([packed_network(nets[p]) for p in 'EI']))
        assert self.network.shape==(2,6785)
        coef=[]
        for p in 'EI':
            A,b,c,_,_=covariance_matrices(p,dt)
            coef.extend([np.r_[A[k].ravel(),b[k],c[k]] for k in range(2)])
        self.coefficients=cp.asarray(np.array(coef));bank=[]
        for tau in [1.,4.,16.,64.]:
            a=dt/tau;e=np.exp(-a);bank.append([e,a,.5*a*a,1-e,1-e*(1+a),1-e*(1+a+.5*a*a)])
        self.bank=cp.asarray(bank);self.pop=cp.asarray(pop,dtype=cp.int32);self.theta=cp.asarray(theta)
        self.refractory=cp.asarray([PARAMS['tau_ref_E'] if p==0 else PARAMS['tau_ref_I'] for p in pop],dtype=cp.float64)
        self.Z=cp.ones(self.P) if Z is None else Z;self.physical=cp.asarray(physical);self.streamed=int(self.physical.ndim==3)
        self.state=cp.zeros((42,self.P));self.history=cp.zeros((depth,self.P));self.rate=cp.zeros(self.P);self.clock=cp.zeros(1,dtype=cp.int32)
        self.SE=cp.asarray(device_block(Baseline('E').spline));self.SI=cp.asarray(device_block(Baseline('I').spline))
        self.module=cp.RawModule(code=source(self.P),options=('--fmad=false',),name_expressions=['local_rate','advance_clock'])
        self.kernel=self.module.get_function('local_rate');self.advance=self.module.get_function('advance_clock')

    def update(self):
        self.kernel((self.P,),(64,),(self.state,self.physical,self.Z,self.theta,self.pop,self.refractory,self.coefficients,self.bank,
            self.network,self.SE,self.SI,self.history,self.rate,self.clock,np.int32(self.depth),self.dt,np.int32(self.streamed)))

    def step(self):
        self.update();self.advance((1,),(1,),(self.clock,))

def check(device):
    from validate_refractory_rate_response import evaluate
    from conditioned_refractory_rate import load_models
    dest=DEST/'spatial_implementation';dest.mkdir(exist_ok=True)
    z=np.load(OUT/'factorial_waveform/prepared.npz');indices=[0,3,6,9];T=float(z['T_ms']);dt=.05
    # Same full-cycle inputs, actual start-from-zero burn and all subsequent steps.
    burn=round(5*T/dt);steps=round(2*T/dt);time=(np.arange(-burn,steps)+1)*dt;wave=z['wave'][indices];W=wave.shape[-1]
    phase=((time/T)%1)*W;lo=np.floor(phase).astype(int)%W;hi=(lo+1)%W;a=phase-np.floor(phase)
    physical=np.transpose((1-a)*wave[:,:,lo]+a*wave[:,:,hi],(2,1,0)).copy();theta=z['pars'][indices,1];pop=np.array([0,0,0,1])
    e=LocalGPUResponse(pop,theta,dt,len(time)+1,physical,device=device);cp=e.cp
    # Compile and warm, then reset before graph capture.
    e.step();cp.cuda.get_current_stream().synchronize();e.state.fill(0);e.history.fill(0);e.rate.fill(0);e.clock.fill(0)
    stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for k in range(100):e.step()
        graph=stream.end_capture()
    for k in range(len(time)//100):graph.launch(stream)
    stream.synchronize()
    for k in range(len(time)%100):e.step()
    cp.cuda.get_current_stream().synchronize();assert int(e.clock.get()[0])==len(time)
    gpu=e.history.get()[1:]*1000;nets,bases,_=load_models();rows=[]
    for j,index in enumerate(indices):
        p='E' if pop[j]==0 else 'I';cpu,_=evaluate(nets[p],bases[p],wave[j],T,dt,burn,steps,theta[j]);observed=gpu[burn:,j]
        error=float(np.max(abs(cpu-observed)));relative=float(np.linalg.norm(cpu-observed)/max(np.linalg.norm(cpu),1.))
        assert error<1e-5 and relative<1e-8,(index,error,relative)
        rows.append(dict(source_index=index,maximum_rate_error_hz=error,relative_L2=relative))
    write(dest/'local_cuda_parity.json',dict(status='PASS',dt_ms=dt,burn_steps=burn,record_steps=steps,rows=rows,
        scope='Same fixed candidate response and local prescribed inputs. Not spatial equivalence, acceptance or a bifurcation result.'))
    log('LOCAL CUDA PARITY PASS',rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);check(p.parse_args().device)
