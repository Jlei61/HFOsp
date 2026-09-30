"""Independent local stationary response under the native SNN update.

The statistical unit is an independent private-Poisson neuron. Conditions use
common random numbers for paired finite-difference precision; neurons remain
independent across replicate index. This is a local response validation, not
a spatial simulation, a replacement rate model or a bifurcation analysis.
"""
from equilibrium_predictor import *


CODE=r'''
#include <curand_kernel.h>
extern "C" __global__ void state_size(int* n){n[0]=sizeof(curandStatePhilox4_32_10_t);}
extern "C" __global__ void initialize(curandStatePhilox4_32_10_t* states,int N,unsigned long long seed){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N)curand_init(seed,i,0,&states[i]);}
extern "C" __global__ void local_native(
 curandStatePhilox4_32_10_t* states,double* q,double* syn,double* voltage,int* refractory,
 unsigned int* counts,unsigned int* sumK,unsigned int* sumK2,const double* theta,const double* input,
 const double* ratio,const double* decay,const int* nr,
 int N,int C,int steps,int measure,double ar,double ad,double jump,double lam,
 unsigned char* recorded,int record){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
 auto rng=states[i];double qq=q[i],ii=syn[i];double v[16];int ref[16];unsigned int out[16];
 #pragma unroll
 for(int c=0;c<16;c++)if(c<C){v[c]=voltage[c*N+i];ref[c]=refractory[c*N+i];out[c]=counts[c*N+i];}
 unsigned int sk=sumK[i],sk2=sumK2[i];
 for(int t=0;t<steps;t++){
  unsigned int K=curand_poisson(&rng,lam);if(record)recorded[t*N+i]=K;
  qq=ar*qq+jump*K;ii=ad*ii+(1-ad)*qq;
  if(measure){sk+=K;sk2+=K*K;}
  #pragma unroll
  for(int c=0;c<16;c++)if(c<C){
   ref[c]=max(0,ref[c]-1);
   if(ref[c]==0){double u=ii*ratio[c]+input[c];v[c]=u+(v[c]-u)*decay[c];
    if(v[c]>=theta[c]){v[c]=11.;ref[c]=nr[c];if(measure)out[c]++;}}
  }
 }
 q[i]=qq;syn[i]=ii;states[i]=rng;sumK[i]=sk;sumK2[i]=sk2;
 #pragma unroll
 for(int c=0;c<16;c++)if(c<C){voltage[c*N+i]=v[c];refractory[c*N+i]=ref[c];counts[c*N+i]=out[c];}
}
'''


class NativeLocal:
    def __init__(self,theta,current,N,seed,device,population=None):
        cp.cuda.Device(device).use();self.N=N;self.C=len(theta);assert self.C<=16
        p=read(OPERATORS/'prepared.json');nu=p['nu_ext_per_ms'];p=p['params']
        self.pars=(float(np.exp(-DT/p['tau_r_AMPA'])),float(np.exp(-DT/p['tau_d_AMPA'])),
                   float(p['tau_m_E']/p['tau_r_AMPA']*p['J_ext_E']),float(nu*DT))
        pop=np.zeros(self.C,dtype=int) if population is None else np.asarray(population)
        self.ratio=cp.asarray(np.where(pop==0,1.,p['tau_m_I']*p['J_ext_I']/(p['tau_m_E']*p['J_ext_E'])))
        self.decay=cp.asarray(np.exp(-DT/np.where(pop==0,p['tau_m_E'],p['tau_m_I'])))
        self.nr=cp.asarray(np.where(pop==0,round(p['tau_ref_E']/DT),round(p['tau_ref_I']/DT)),dtype=cp.int32)
        module=cp.RawModule(code=CODE,options=('--fmad=false','-I/usr/local/cuda/include'),
                           name_expressions=['state_size','initialize','local_native'])
        size=cp.zeros(1,dtype=cp.int32);module.get_function('state_size')((1,),(1,),(size,))
        self.rng=cp.empty(int(size.get()[0])*N,dtype=cp.uint8)
        module.get_function('initialize')(((N+127)//128,),(128,),(self.rng,np.int32(N),np.uint64(seed)))
        self.step=module.get_function('local_native');self.theta=cp.asarray(theta);self.input=cp.asarray(current)
        self.q=cp.zeros(N);self.syn=cp.zeros(N);self.v=cp.full((self.C,N),11.)
        self.ref=cp.zeros((self.C,N),dtype=cp.int32);self.count=cp.zeros((self.C,N),dtype=cp.uint32)
        self.sumK=cp.zeros(N,dtype=cp.uint32);self.sumK2=cp.zeros(N,dtype=cp.uint32)

    def advance(self,steps,measure=True,record=False):
        saved=cp.empty((steps,self.N) if record else (1,),dtype=cp.uint8)
        ar,ad,jump,lam=self.pars
        self.step(((self.N+127)//128,),(128,),(self.rng,self.q,self.syn,self.v,self.ref,self.count,
            self.sumK,self.sumK2,self.theta,self.input,self.ratio,self.decay,self.nr,np.int32(self.N),np.int32(self.C),np.int32(steps),
            np.int32(measure),ar,ad,jump,lam,saved,np.int32(record)))
        return saved


def conditions(mode='critical_response'):
    rows=[];g=395
    if mode=='critical_mode':
        q=OUT/'corrected_rate_sections/rE0.14950000_degree8_dv0.125_matching_predictor'
        a=np.load(q/'stationary_local_state.npz');m=np.load(q/'static_response_eps0.0005/stationary_tangent.npz')
        participation=abs(m['left_rate_mode']*m['right_rate_mode']);pop=a['population']
        E=np.flatnonzero(pop==0);I=np.flatnonzero(pop==1)
        E=E[np.argsort(participation[E])[::-1][:3]];I=I[np.argsort(participation[I])[::-1][:10]]
        groups=list(E)+[int(I[0]),int(I[np.argmax(a['current_mv'][I])])]
        for g in groups:
            for delta in [-.02,0.,.02]:
                rows.append(dict(label=f'critical_mode_g{g}_{delta:+g}',source_degree=8,group=int(g),population=int(pop[g]),
                    threshold_mv=float(a['theta'][g]),current_mv=float(a['current_mv'][g]+delta),delta_mv=delta,
                    left_right_mode_participation=float(participation[g])))
        rows.append(dict(label='background_theta18_u0',source_degree=None,group=None,population=0,threshold_mv=18.,current_mv=0.,delta_mv=0.))
        return rows
    source=OUT/'corrected_rate_sections/rE0.15256800_degree6_dv0.125_secant/stationary_local_state.npz'
    a=np.load(source);bases=[(6,float(a['theta'][g]),float(a['current_mv'][g]))]
    for degree,name in [(8,'low_degree8_dense'),(12,'low_degree12_high_precision'),(16,'low_degree16_high_precision_rate')]:
        q=OUT/'equilibrium_predictors'/name;z=np.load(q/'branch.npz');i=int(z['D'].argmax())
        p=EquilibriumProblem(OUT/'stationary_response/degree6_dv0.125','pchip')
        u=p.equations(z['rate_hz'][i],float(z['D'][i]))[-1]
        bases.append((degree,float(p.theta[g]),float(u[g])))
    for degree,theta,u in bases:
        for delta in [-.02,0.,.02]:
            rows.append(dict(label=f'degree{degree}_candidate_g{g}_{delta:+g}',source_degree=degree,
                             group=g,threshold_mv=theta,current_mv=u+delta,delta_mv=delta))
    rows.append(dict(label='background_theta18_u0',source_degree=None,group=None,threshold_mv=18.,current_mv=0.,delta_mv=0.))
    return rows


def audit(theta,current,device,population):
    ids=np.unique(np.r_[0,np.flatnonzero(population==1)[:1],len(theta)-1]).astype(int)
    theta=theta[ids];current=current[ids];population=population[ids];C=len(ids)
    m=NativeLocal(theta,current,16,20191,device,population);K=cp.asnumpy(m.advance(1000,record=True))
    q=np.zeros(16);syn=q.copy();v=np.full((C,16),11.);ref=np.zeros((C,16),int);counts=np.zeros((C,16),int)
    ar,ad,jump,lam=m.pars;ratio=cp.asnumpy(m.ratio)[:,None];dec=cp.asnumpy(m.decay)[:,None];nr=cp.asnumpy(m.nr)[:,None]
    for k in K:
        q=ar*q+jump*k;syn=ad*syn+(1-ad)*q;ref=np.maximum(0,ref-1)
        u=syn[None,:]*ratio+current[:,None];v=np.where(ref==0,u+(v-u)*dec,v)
        hit=(ref==0)&(v>=theta[:,None]);counts+=hit;ref=np.where(hit,nr,ref);v[hit]=11.
    result=dict(counts_equal=bool(np.array_equal(counts,cp.asnumpy(m.count))),
                voltage_max_error=float(np.max(abs(v-cp.asnumpy(m.v)))),
                synaptic_current_max_error=float(np.max(abs(syn-cp.asnumpy(m.syn)))))
    result['pass']=result['counts_equal'] and max(result['voltage_max_error'],result['synaptic_current_max_error'])<1e-10
    assert result['pass'],result
    return result


def run(args):
    folder=OUT/'stationary_native_mc'/args.label;folder.mkdir(parents=True,exist_ok=False)
    rows=conditions(args.conditions);theta=np.array([r['threshold_mv'] for r in rows]);current=np.array([r['current_mv'] for r in rows]);pop=np.array([r.get('population',0) for r in rows])
    cfg=dict(conditions=rows,neurons_per_condition=args.neurons,burn_ms=args.burn,duration_ms=args.duration,seed=args.seed,
        observable='Stationary spikes per second per neuron at fixed local net recurrent current',
        statistical_unit='Independent private-Poisson neuron; common innovations paired across input conditions',
        baseline='Native discrete voltage/refractory and two-stage colored-Poisson updates at0.1ms',
        scope='Local response/noise-truncation audit only; neither spatial correspondence nor a bifurcation claim')
    write(folder/'config.json',cfg);write(folder/'operator_audit.json',audit(theta,current,args.device,pop))
    m=NativeLocal(theta,current,args.neurons,args.seed,args.device,pop);started=time.time();last=started
    for phase,duration,measure in [('burn',args.burn,False),('sample',args.duration,True)]:
        steps=round(duration/DT)
        for at in range(0,steps,100):
            m.advance(min(100,steps-at),measure);cp.cuda.get_current_stream().synchronize()
            if time.time()-last>20:
                write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),phase=phase,completed_ms=(at+100)*DT,wall_s=time.time()-started))
                print('native local MC',phase,(at+100)*DT,flush=True);last=time.time()
    rates=cp.asnumpy(m.count)/(args.duration/1000.);mean=rates.mean(1);se=rates.std(1,ddof=1)/np.sqrt(args.neurons)
    result=[]
    for i,row in enumerate(rows):result.append(dict(row,rate_hz=float(mean[i]),standard_error_hz=float(se[i]),ci95_hz=[float(mean[i]-1.96*se[i]),float(mean[i]+1.96*se[i])]))
    paired=[]
    for at in range(0,len(rows)-1,3):
        sample=(rates[at+2]-rates[at])/.04
        paired.append(dict(source_degree=rows[at]['source_degree'],group=rows[at]['group'],population=rows[at].get('population',0),central_difference_hz_per_mv=float(sample.mean()),
                           standard_error=float(sample.std(ddof=1)/np.sqrt(args.neurons)),step_mv=.02))
    sk=cp.asnumpy(m.sumK);sk2=cp.asnumpy(m.sumK2);n=round(args.duration/DT)*args.neurons
    noise_mean=float(sk.sum(dtype=np.uint64)/n);noise_variance=float(sk2.sum(dtype=np.uint64)/n-noise_mean**2)
    np.savez_compressed(folder/'neuron_counts.npz',counts=cp.asnumpy(m.count),theta_mv=theta,current_mv=current)
    write(folder/'result.json',dict(status='COMPLETE',conditions=result,paired_slopes=paired,
        empirical_Poisson_mean=noise_mean,empirical_Poisson_variance=noise_variance,expected_Poisson_mean=m.pars[3],wall_s=time.time()-started))
    print(result,paired,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--label',default='critical_response_v1');ap.add_argument('--neurons',type=int,default=100000)
    ap.add_argument('--burn',type=float,default=1000.);ap.add_argument('--duration',type=float,default=10000.)
    ap.add_argument('--conditions',choices=['critical_response','critical_mode'],default='critical_response')
    ap.add_argument('--seed',type=int,default=21917);ap.add_argument('--device',type=int,default=0);run(ap.parse_args())
