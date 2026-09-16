"""Independent two-state bootstrap-particle likelihood for the scalar mark readout."""
import sys,json,time,math,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from numba import cuda,float64
from numba.cuda.random import create_xoroshiro128p_states,xoroshiro128p_normal_float64,xoroshiro128p_uniform_float64
from scipy.special import logsumexp
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.advanced_controls import history_data
from scripts.patient_state_v1.two_timescale import filter2

@cuda.jit(max_registers=60)
def kernel(dt,reset,slow_dt,slow_reset,y,h,params,slow_tau,rng,out,trace):
    tid=cuda.threadIdx.x;bid=cuda.blockIdx.x;P=cuda.blockDim.x;ri=bid*P+tid
    sf=cuda.shared.array(1024,float64);sb=cuda.shared.array(1024,float64);weight=cuda.shared.array(1024,float64);scratch=cuda.shared.array(1024,float64);scan=cuda.shared.array(1024,float64);aux=cuda.shared.array(4,float64)
    b=params[bid,0];gamma=params[bid,1];tau=math.exp(params[bid,2]);sd=math.exp(params[bid,3]);ss=math.exp(params[bid,4]);sf[tid]=0.;sb[tid]=0.;weight[tid]=1./P;ll=0.
    if params.shape[1]>5:slow_tau=math.exp(params[bid,5])
    for i in range(len(dt)):
        z=xoroshiro128p_normal_float64(rng,ri);zz=xoroshiro128p_normal_float64(rng,ri)
        if reset[i]:f=sd*z
        else:
            a=math.exp(-dt[i]/tau);f=a*sf[tid]+sd*math.sqrt(-math.expm1(-2*dt[i]/tau))*z
        if slow_reset[i]:s=ss*zz
        else:
            a=math.exp(-slow_dt[i]/slow_tau);s=a*sb[tid]+ss*math.sqrt(-math.expm1(-2*slow_dt[i]/slow_tau))*zz
        if reset[i] and slow_reset[i]:weight[tid]=1./P
        eta=b+gamma*h[i]+f+s;lp=y[i]*eta-(max(eta,0.)+math.log1p(math.exp(-abs(eta))));sf[tid]=f;sb[tid]=s;weight[tid]*=math.exp(lp);scratch[tid]=weight[tid]
        cuda.syncthreads();stride=P//2
        while stride:
            if tid<stride:scratch[tid]+=scratch[tid+stride]
            cuda.syncthreads();stride//=2
        if tid==0:aux[0]=max(scratch[0],1e-300)
        cuda.syncthreads();ll+=math.log(aux[0]);weight[tid]/=aux[0]
        if trace.shape[0]>0:
            if tid==0:trace[bid,i,0]=math.log(aux[0]);trace[bid,i,1]=aux[0] if y[i]==1 else 1-aux[0]
            for column in range(3):
                value=f if column==0 else s if column==1 else (f+s)*(f+s)
                scratch[tid]=weight[tid]*value;cuda.syncthreads();stride=P//2
                while stride:
                    if tid<stride:scratch[tid]+=scratch[tid+stride]
                    cuda.syncthreads();stride//=2
                if tid==0:trace[bid,i,column+2]=scratch[0]
                cuda.syncthreads()
        scratch[tid]=weight[tid]*weight[tid];cuda.syncthreads();stride=P//2
        while stride:
            if tid<stride:scratch[tid]+=scratch[tid+stride]
            cuda.syncthreads();stride//=2
        if tid==0:aux[1]=scratch[0]
        cuda.syncthreads()
        if aux[1]>2./P:
            scan[tid]=weight[tid];cuda.syncthreads();offset=1
            while offset<P:
                value=scan[tid]
                if tid>=offset:value+=scan[tid-offset]
                cuda.syncthreads();scan[tid]=value;cuda.syncthreads();offset*=2
            if tid==0:aux[2]=xoroshiro128p_uniform_float64(rng,ri)
            cuda.syncthreads();u=(tid+aux[2])/P;lo=0;hi=P-1
            while lo<hi:
                mid=(lo+hi)//2
                if scan[mid]<u:lo=mid+1
                else:hi=mid
            newf=sf[lo];news=sb[lo];cuda.syncthreads();sf[tid]=newf;sb[tid]=news;weight[tid]=1./P
        cuda.syncthreads()
    if tid==0:out[bid]=ll

def dataset(carry):
    d=history_data(dict(np.load(RUN/'observations.npz')))
    if carry:d['slow_dt']=np.r_[0,np.diff(d['t'])];d['slow_reset']=np.r_[True,np.zeros(len(d['t'])-1,bool)]
    else:d['slow_dt']=d['dt'].copy();d['slow_reset']=d['reset'].copy()
    return d

def expand(theta,history):
    theta=np.asarray(theta)
    return theta if history else np.column_stack([theta[:,0],np.zeros(len(theta)),theta[:,1:]])

def evaluate(d,parameters,slow_tau,particles=1024,seed=360000,batch=64,trace=False):
    assert np.all(d['n']==1),'The current observation likelihood is Bernoulli'
    arrays=[cuda.to_device(np.ascontiguousarray(d[k])) for k in ('dt','reset','slow_dt','slow_reset','y')];h=cuda.to_device(np.ascontiguousarray(d['x'][:,1]));answer=[];traces=[]
    for start in range(0,len(parameters),batch):
        theta=np.ascontiguousarray(parameters[start:start+batch],dtype=np.float64);rng=create_xoroshiro128p_states(len(theta)*particles,seed=seed+start);out=cuda.device_array(len(theta),dtype=np.float64);tr=cuda.device_array((len(theta),len(d['y']),5) if trace else (0,0,0),dtype=np.float64)
        kernel[len(theta),particles](*arrays,h,cuda.to_device(theta),slow_tau,rng,out,tr);answer.extend(out.copy_to_host().tolist())
        if trace:traces.append(tr.copy_to_host())
    return (np.array(answer),np.concatenate(traces)) if trace else np.array(answer)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--replicates',type=int,default=128);ap.add_argument('--particles',type=int,default=1024);args=ap.parse_args();cuda.select_device(args.gpu);out=RUN/'two_timescale_particle_audit_v1_11';out.mkdir(exist_ok=True);best={}
    for path in (RUN/'two_timescale_marks_v1_11/fits').glob('full_*.json'):
        r=json.loads(path.read_text());j=r['job'];key=(j['slow_tau'],j['history'],j['carry_slow'])
        if r['status']=='COMPLETE' and (key not in best or r['loglik']>best[key][0]['loglik']):best[key]=(r,path)
    for index,((ts,history,carry),(r,path)) in enumerate(sorted(best.items())):
        name=f'slow{int(ts)}_hist{int(history)}_carry{int(carry)}';dest=out/(name+'.json')
        if dest.exists():continue
        d=dataset(carry);theta=np.array(r['theta']);start=time.time();ll=evaluate(d,expand(np.tile(theta,(args.replicates,1)),history),ts,args.particles,seed=360000+index*10000);f=filter2(theta,d,ts,history);v=logsumexp(ll)-np.log(len(ll));w=np.exp(ll-logsumexp(ll));result=dict(status='COMPLETE',fit_source=str(path),theta=theta,slow_tau=ts,history=history,carry_slow=carry,particles=args.particles,replicates=args.replicates,loglik_replicates=ll,particle_logmean=v,loglik_sd=np.std(ll),likelihood_ess=1/np.sum(w*w),laplace_loglik=r['loglik'],adf_loglik=f['loglik'],elapsed=time.time()-start)
        write_json(dest,result);print(json.dumps({k:result[k] for k in ['slow_tau','history','carry_slow','particle_logmean','loglik_sd','likelihood_ess','elapsed']}),flush=True)
    write_json(out/'status.json',dict(status='COMPLETE',n_models=len(best)))

if __name__=='__main__':main()
