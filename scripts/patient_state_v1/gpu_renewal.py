"""Stable log-weight particle check of the latent activity inference, independent of Laplace."""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from numba import cuda,float64
from numba.cuda.random import create_xoroshiro128p_states,xoroshiro128p_normal_float64,xoroshiro128p_uniform_float64
from scipy.special import logsumexp
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.renewal import prepare
import math

@cuda.jit(max_registers=60)
def kernel(dt,reset,n,risk,params,rng,out,guided):
    tid=cuda.threadIdx.x;bid=cuda.blockIdx.x;P=cuda.blockDim.x;ri=bid*P+tid
    state=cuda.shared.array(1024,float64);weight=cuda.shared.array(1024,float64);scratch=cuda.shared.array(1024,float64);scan=cuda.shared.array(1024,float64);aux=cuda.shared.array(4,float64)
    a=params[bid,0];tau=math.exp(params[bid,1]);sd=math.exp(params[bid,2]);state[tid]=0.;weight[tid]=1./P;ll=0.
    for i in range(len(dt)):
        z=xoroshiro128p_normal_float64(rng,ri)
        if reset[i]:prior_mean=0.;prior_var=sd*sd;weight[tid]=1./P
        else:
            rho=math.exp(-dt[i]/tau);prior_mean=rho*state[tid];prior_var=sd*sd*(-math.expm1(-2*dt[i]/tau))
        correction=0.
        if guided:
            # Gaussian conditional mode: m+q*n-W(q*risk*exp(a+m+q*n)).
            qq=prior_var;mm=prior_mean+qq*n[i];lambert=0.
            if risk[i]>0:
                zz=math.log(qq*risk[i])+a+mm
                if zz<-35:lambert=math.exp(zz)
                else:
                    lambert=math.exp(zz) if zz<=0 else max(.5,zz-math.log(max(zz,1.)))
                    for j in range(10):lambert=max(1e-300,lambert-(lambert+math.log(lambert)-zz)*lambert/(lambert+1.))
            mode=mm-lambert;variance=qq/(1+lambert)
            # Defensive 10% prior component protects broad posterior tails.
            if xoroshiro128p_uniform_float64(rng,ri)<.1:s=prior_mean+math.sqrt(prior_var)*z
            else:s=mode+math.sqrt(variance)*z
            logprior=-.5*(math.log(prior_var)+(s-prior_mean)**2/prior_var)
            logpost=-.5*(math.log(variance)+(s-mode)**2/variance)
            aa=math.log(.1)+logprior;bb=math.log(.9)+logpost;mx=max(aa,bb);logproposal=mx+math.log(math.exp(aa-mx)+math.exp(bb-mx));correction=logprior-logproposal
        else:s=prior_mean+math.sqrt(prior_var)*z
        eta=a+s
        logw=math.log(weight[tid])+n[i]*eta-risk[i]*math.exp(min(eta,50.))+correction if weight[tid]>0 else -math.inf
        state[tid]=s;scratch[tid]=logw
        cuda.syncthreads();stride=P//2
        while stride>0:
            if tid<stride:scratch[tid]=max(scratch[tid],scratch[tid+stride])
            cuda.syncthreads();stride//=2
        if tid==0:aux[0]=scratch[0]
        cuda.syncthreads();weight[tid]=math.exp(logw-aux[0]);scratch[tid]=weight[tid]
        cuda.syncthreads();stride=P//2
        while stride>0:
            if tid<stride:scratch[tid]+=scratch[tid+stride]
            cuda.syncthreads();stride//=2
        if tid==0:aux[1]=max(scratch[0],1e-300)
        cuda.syncthreads();ll+=aux[0]+math.log(aux[1]);weight[tid]/=aux[1];scratch[tid]=weight[tid]*weight[tid]
        cuda.syncthreads();stride=P//2
        while stride>0:
            if tid<stride:scratch[tid]+=scratch[tid+stride]
            cuda.syncthreads();stride//=2
        if tid==0:aux[2]=scratch[0]
        cuda.syncthreads()
        if aux[2]>2./P:
            scan[tid]=weight[tid];cuda.syncthreads();offset=1
            while offset<P:
                value=scan[tid]
                if tid>=offset:value+=scan[tid-offset]
                cuda.syncthreads();scan[tid]=value;cuda.syncthreads();offset*=2
            if tid==0:aux[3]=xoroshiro128p_uniform_float64(rng,ri)
            cuda.syncthreads();u=(tid+aux[3])/P;lo=0;hi=P-1
            while lo<hi:
                mid=(lo+hi)//2
                if scan[mid]<u:lo=mid+1
                else:hi=mid
            value=state[lo];cuda.syncthreads();state[tid]=value;weight[tid]=1./P
        cuda.syncthreads()
    if tid==0:out[bid]=ll

def evaluate(d,params,particles=1024,seed=310000,batch=64,guided=False):
    arrays=[cuda.to_device(np.ascontiguousarray(d[k])) for k in ('dt','reset','n','exposure')];answer=[]
    for start in range(0,len(params),batch):
        p=np.ascontiguousarray(params[start:start+batch],dtype=np.float64);rng=create_xoroshiro128p_states(len(p)*particles,seed=seed+start);out=cuda.device_array(len(p),dtype=np.float64)
        kernel[len(p),particles](*arrays,cuda.to_device(p),rng,out,guided);answer.extend(out.copy_to_host().tolist())
    return np.array(answer)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--seconds',type=int,default=5);ap.add_argument('--particles',type=int,default=1024);ap.add_argument('--replicates',type=int,default=64);ap.add_argument('--guided',action='store_true');args=ap.parse_args();cuda.select_device(args.gpu)
    out=RUN/'renewal_particle_audit_v1_6'
    if args.guided:out=out/'guided'
    out.mkdir(parents=True,exist_ok=True);d=prepare(args.seconds,.25);fits=[]
    for path in (RUN/'renewal_observation_v1_3/fits').glob('*.json'):
        r=json.loads(path.read_text());j=r['job']
        if r['status']=='COMPLETE' and j['seconds']==args.seconds and j['deadtime']==.25 and j['scope']=='full' and j['model']=='rate':fits.append(r)
    best=max(fits,key=lambda r:r['loglik']);base=np.array(best['theta']);rows=[]
    for i,shift in enumerate((0.,-.5,.5,-1.,1.)):
        for j,scale in enumerate((1.,.8,.6,1.2)):
            dest=out/f'g{args.seconds}_p{args.particles}_a{i}_s{j}.json'
            if dest.exists():continue
            theta=base.copy();theta[0]+=shift;theta[2]+=np.log(scale);start=time.time();ll=evaluate(d,np.tile(theta,(args.replicates,1)),args.particles,seed=310000+100*i+10*j,guided=args.guided)
            result=dict(status='COMPLETE',seconds=args.seconds,particles=args.particles,replicates=args.replicates,theta=theta,
                        likelihood_log_mean=logsumexp(ll)-np.log(len(ll)),loglik_sd=np.std(ll),loglik_replicates=ll,
                        base_laplace_loglik=best['loglik'],shift=shift,sd_scale=scale,elapsed=time.time()-start,guided=args.guided)
            write_json(dest,result);print(json.dumps(dict(a_shift=shift,sd_scale=scale,particle_ll=float(result['likelihood_log_mean']),loglik_sd=float(result['loglik_sd']),elapsed=result['elapsed'])),flush=True)
    write_json(out/f'g{args.seconds}_p{args.particles}_status.json',dict(status='COMPLETE',n_candidates=20))

if __name__=='__main__':main()
