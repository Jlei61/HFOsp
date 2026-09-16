"""Independent bootstrap-particle likelihood for exact-time OU marked observations.

Each CUDA block is one independent parameter/replicate; systematic resampling at ESS<P/2.
This is a stochastic likelihood validation, not an optimizer or a seizure generator.
"""
import sys,math,json,time,argparse,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from numba import cuda,float64
from numba.cuda.random import create_xoroshiro128p_states,xoroshiro128p_normal_float64,xoroshiro128p_uniform_float64
from scripts.patient_state_v1.common import RUN,write_json

@cuda.jit(max_registers=60)
def likelihood_kernel(dt,reset,y,n,x,params,rng,out,trace):
    tid=cuda.threadIdx.x;bid=cuda.blockIdx.x;P=cuda.blockDim.x
    states=cuda.shared.array(1024,float64);weights=cuda.shared.array(1024,float64)
    scratch=cuda.shared.array(1024,float64);scan=cuda.shared.array(1024,float64)
    aux=cuda.shared.array(4,float64)
    ri=bid*P+tid
    b=params[bid,0];bs=params[bid,1];bc=params[bid,2]
    tau=math.exp(params[bid,3]);sd=math.exp(params[bid,4]);ll=0.
    states[tid]=0.;weights[tid]=1./P
    for i in range(len(dt)):
        z=xoroshiro128p_normal_float64(rng,ri)
        if reset[i]:
            s=sd*z;weights[tid]=1./P
        else:
            a=math.exp(-dt[i]/tau)
            s=a*states[tid]+sd*math.sqrt(-math.expm1(-2*dt[i]/tau))*z
        eta=b+bs*x[i,1]+bc*x[i,2]+s
        logp=y[i]*eta-n[i]*(max(eta,0.)+math.log1p(math.exp(-abs(eta))))
        states[tid]=s;weights[tid]*=math.exp(logp);scratch[tid]=weights[tid]
        cuda.syncthreads()
        stride=P//2
        while stride>0:
            if tid<stride:scratch[tid]+=scratch[tid+stride]
            cuda.syncthreads();stride//=2
        if tid==0:aux[0]=max(scratch[0],1e-300)
        cuda.syncthreads();ll+=math.log(aux[0]);weights[tid]/=aux[0]
        if trace.shape[0]>0:
            scratch[tid]=states[tid]*weights[tid]
            cuda.syncthreads();stride=P//2
            while stride>0:
                if tid<stride:scratch[tid]+=scratch[tid+stride]
                cuda.syncthreads();stride//=2
            if tid==0:
                trace[bid,i,0]=math.log(aux[0]);trace[bid,i,1]=scratch[0]
                trace[bid,i,3]=aux[0] if y[i]==1 else 1-aux[0]
            scratch[tid]=states[tid]*states[tid]*weights[tid]
            cuda.syncthreads();stride=P//2
            while stride>0:
                if tid<stride:scratch[tid]+=scratch[tid+stride]
                cuda.syncthreads();stride//=2
            if tid==0:trace[bid,i,2]=max(scratch[0]-trace[bid,i,1]**2,0.)
            cuda.syncthreads()
        scratch[tid]=weights[tid]*weights[tid]
        cuda.syncthreads();stride=P//2
        while stride>0:
            if tid<stride:scratch[tid]+=scratch[tid+stride]
            cuda.syncthreads();stride//=2
        if tid==0:aux[1]=scratch[0]
        cuda.syncthreads()
        if aux[1]>2./P:
            scan[tid]=weights[tid]
            cuda.syncthreads();offset=1
            while offset<P:
                value=scan[tid]
                if tid>=offset:value+=scan[tid-offset]
                cuda.syncthreads();scan[tid]=value;cuda.syncthreads();offset*=2
            if tid==0:aux[2]=xoroshiro128p_uniform_float64(rng,ri)
            cuda.syncthreads();u=(tid+aux[2])/P
            lo=0;hi=P-1
            while lo<hi:
                mid=(lo+hi)//2
                if scan[mid]<u:lo=mid+1
                else:hi=mid
            value=states[lo]
            cuda.syncthreads();states[tid]=value;weights[tid]=1./P
        cuda.syncthreads()
    if tid==0:out[bid]=ll

def evaluate(data,params,particles=512,seed=260909,batch=64):
    arrays=[cuda.to_device(np.ascontiguousarray(data[k])) for k in ('dt','reset','y','n','x')]
    answer=[]
    for start in range(0,len(params),batch):
        par=np.ascontiguousarray(params[start:start+batch],dtype=np.float64)
        rng=create_xoroshiro128p_states(len(par)*particles,seed=seed+start)
        out=cuda.device_array(len(par),dtype=np.float64)
        likelihood_kernel[len(par),particles](*arrays,cuda.to_device(par),rng,out,cuda.device_array((0,0,0),dtype=np.float64))
        answer.extend(out.copy_to_host().tolist())
    return np.array(answer)

def evaluate_trace(data,params,particles=1024,seed=260911):
    assert np.all(data['n']==1), 'Trace probability readout currently requires Bernoulli marks'
    arrays=[cuda.to_device(np.ascontiguousarray(data[k])) for k in ('dt','reset','y','n','x')]
    par=np.ascontiguousarray(params,dtype=np.float64);rng=create_xoroshiro128p_states(len(par)*particles,seed=seed)
    out=cuda.device_array(len(par),dtype=np.float64);trace=cuda.device_array((len(par),len(data['y']),4),dtype=np.float64)
    likelihood_kernel[len(par),particles](*arrays,cuda.to_device(par),rng,out,trace)
    return out.copy_to_host(),trace.copy_to_host()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--particles',type=int,default=512);ap.add_argument('--replicates',type=int,default=32)
    ap.add_argument('--mode',choices=['canary','validate'],default='canary');args=ap.parse_args()
    cuda.select_device(args.gpu)
    data=dict(np.load(RUN/'observations.npz'));jobs=[]
    if args.mode=='canary':
        data={k:v[:4000].copy() for k,v in data.items() if v.ndim>0}
        for tau,sd in ((1.,.6),(.0166667,.47),(6.,1.5)):
            for rep in range(args.replicates):jobs.append(dict(theta=[-.9,0,0,np.log(tau),np.log(sd)],name=f't{tau}_s{sd}',rep=rep))
    else:
        files=list((RUN/'cpu_fits').glob('full_ou*.json'));best={}
        for f in files:
            r=json.loads(f.read_text())
            if r.get('status')!='COMPLETE':continue
            model=r['model']
            if model not in best or r['loglik']>best[model]['loglik']:best[model]=r
        if not best:raise RuntimeError('No completed full fits')
        for model,r in best.items():
            t=r['theta'];par=t if model=='ou_cycle' else [t[0],0,0,t[1],t[2]]
            for rep in range(args.replicates):jobs.append(dict(theta=par,name=model,rep=rep,laplace_loglik=r['loglik'],adf_loglik=r['adf_loglik']))
    start=time.time();params=np.array([j['theta'] for j in jobs]);ll=evaluate(data,params,args.particles,seed=260909+args.gpu*10000)
    for j,l in zip(jobs,ll):j['loglik']=l
    name=f'particles_{args.mode}_gpu{args.gpu}_p{args.particles}_r{args.replicates}'
    result=dict(status='COMPLETE',device=args.gpu,particles=args.particles,rows=jobs,n_observations=len(data['y']),elapsed_seconds=time.time()-start)
    write_json(RUN/'gpu_checks'/f'{name}.json',result)
    print(json.dumps(dict(name=name,elapsed=result['elapsed_seconds'],groups={name:dict(mean=np.mean([j['loglik'] for j in jobs if j['name']==name]),sd=np.std([j['loglik'] for j in jobs if j['name']==name])) for name in set(j['name'] for j in jobs)})),flush=True)

if __name__=='__main__':main()
