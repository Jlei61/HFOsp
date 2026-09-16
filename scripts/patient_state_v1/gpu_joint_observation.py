"""Independent importance-particle likelihood for joint rate/mode OU states.

A defensive Gaussian proposal changes sampling efficiency, not the target:
each weight includes the exact prior/proposal density ratio. The proposal need
not be the exact conditional mode. Rejection/resampling retains paired states.
"""
import sys,json,time,math,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from numba import cuda,float64
from numba.cuda.random import create_xoroshiro128p_states,xoroshiro128p_normal_float64,xoroshiro128p_uniform_float64
from scipy.special import logsumexp
from numpy.polynomial.hermite import hermgauss
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.joint_two_state import unpack
from scripts.patient_state_v1.renewal import prepare
OUT=RUN/'joint_particle_audit_v1_24'

@cuda.jit(max_registers=60)
def kernel(dt,reset,n,y,risk,params,rng,out,guided):
    tid=cuda.threadIdx.x;bid=cuda.blockIdx.x;P=cuda.blockDim.x;ri=bid*P+tid
    state_s=cuda.shared.array(1024,float64);state_r=cuda.shared.array(1024,float64);weight=cuda.shared.array(1024,float64);scratch=cuda.shared.array(1024,float64);scan=cuda.shared.array(1024,float64);aux=cuda.shared.array(4,float64)
    b=params[bid,0];a=params[bid,1];c=params[bid,2];ts=math.exp(params[bid,3]);ss=math.exp(params[bid,4]);tr=math.exp(params[bid,5]);sr=math.exp(params[bid,6]);state_s[tid]=0.;state_r[tid]=0.;weight[tid]=1./P;ll=0.
    for i in range(len(dt)):
        if reset[i]:ms=0.;mr=0.;qs=ss*ss;qr=sr*sr;weight[tid]=1./P
        else:
            rs=math.exp(-dt[i]/ts);rr=math.exp(-dt[i]/tr);ms=rs*state_s[tid];mr=rr*state_r[tid];qs=ss*ss*(-math.expm1(-2*dt[i]/ts));qr=sr*sr*(-math.expm1(-2*dt[i]/tr))
        z1=xoroshiro128p_normal_float64(rng,ri);z2=xoroshiro128p_normal_float64(rng,ri);correction=0.
        if guided:
            mode_s=ms;mode_r=mr;is_=1./qs;ir=1./qr
            for _ in range(12):
                eta=b+mode_s;p=1./(1.+math.exp(-max(-700.,eta)));mu=risk[i]*math.exp(min(a+c*mode_s+mode_r,50.));g0=(mode_s-ms)*is_+n[i]*p-y[i]+c*(mu-n[i]);g1=(mode_r-mr)*ir+mu-n[i];h11=ir+mu;ratio=c*mu/h11;schur=is_+n[i]*p*(1-p)+c*c*mu*ir/h11;ds=(g0-ratio*g1)/schur;dr=(g1-c*mu*ds)/h11;scale=1./max(1.,max(abs(ds),abs(dr))/3.);mode_s-=scale*ds;mode_r-=scale*dr
            eta=b+mode_s;p=1./(1.+math.exp(-max(-700.,eta)));mu=risk[i]*math.exp(min(a+c*mode_s+mode_r,50.));h11=ir+mu;ratio=c*mu/h11;schur=is_+n[i]*p*(1-p)+c*c*mu*ir/h11
            if xoroshiro128p_uniform_float64(rng,ri)<.1:s=ms+math.sqrt(qs)*z1;r=mr+math.sqrt(qr)*z2
            else:s=mode_s+z1/math.sqrt(schur);r=mode_r-ratio*(s-mode_s)+z2/math.sqrt(h11)
            logprior=-.5*(math.log(qs)+math.log(qr)+(s-ms)**2/qs+(r-mr)**2/qr);lognormal=.5*(math.log(schur)+math.log(h11))-.5*(schur*(s-mode_s)**2+h11*(r-mode_r+ratio*(s-mode_s))**2);aa=math.log(.1)+logprior;bb=math.log(.9)+lognormal;mx=max(aa,bb);correction=logprior-mx-math.log(math.exp(aa-mx)+math.exp(bb-mx))
        else:s=ms+math.sqrt(qs)*z1;r=mr+math.sqrt(qr)*z2
        rate_eta=a+c*s+r;mark_eta=b+s;obs=n[i]*rate_eta-risk[i]*math.exp(min(rate_eta,50.))+y[i]*mark_eta-n[i]*(max(mark_eta,0.)+math.log1p(math.exp(-abs(mark_eta))));logw=math.log(weight[tid])+obs+correction if weight[tid]>0 else -math.inf;state_s[tid]=s;state_r[tid]=r;scratch[tid]=logw
        cuda.syncthreads();stride=P//2
        while stride>0:
            if tid<stride:scratch[tid]=max(scratch[tid],scratch[tid+stride])
            cuda.syncthreads();stride//=2
        if tid==0:aux[0]=scratch[0]
        cuda.syncthreads();weight[tid]=math.exp(logw-aux[0]);scratch[tid]=weight[tid];cuda.syncthreads();stride=P//2
        while stride>0:
            if tid<stride:scratch[tid]+=scratch[tid+stride]
            cuda.syncthreads();stride//=2
        if tid==0:aux[1]=max(scratch[0],1e-300)
        cuda.syncthreads();ll+=aux[0]+math.log(aux[1]);weight[tid]/=aux[1];scratch[tid]=weight[tid]*weight[tid];cuda.syncthreads();stride=P//2
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
            sv=state_s[lo];rv=state_r[lo];cuda.syncthreads();state_s[tid]=sv;state_r[tid]=rv;weight[tid]=1./P
        cuda.syncthreads()
    if tid==0:out[bid]=ll

def evaluate(d,params,particles=1024,seed=920000,batch=64,guided=True):
    assert particles in [128,256,512,1024];arrays=[cuda.to_device(np.ascontiguousarray(d[k])) for k in ['dt','reset','n','y','exposure']];answer=[]
    for start in range(0,len(params),batch):
        p=np.ascontiguousarray(params[start:start+batch],dtype=np.float64);rng=create_xoroshiro128p_states(len(p)*particles,seed=seed+start);out=cuda.device_array(len(p),dtype=np.float64);kernel[len(p),particles](*arrays,cuda.to_device(p),rng,out,guided);answer.extend(out.copy_to_host().tolist())
    return np.array(answer)

def canary():
    nodes,w=hermgauss(100);w/=np.sqrt(np.pi);rows=[]
    for k,(c,n,y,risk) in enumerate([(0.,0.,0.,0.),(0.,2.,1.,.002),(.7,3.,2.,.004),(-.6,0.,0.,.002)]):
        pars=np.array([-.8,5.,c,np.log(.3),np.log(.6),np.log(.1),np.log(.8)]);s=np.sqrt(2)*.6*nodes[:,None];r=np.sqrt(2)*.8*nodes[None,:];eta=5+c*s+r;obs=n*eta-risk*np.exp(eta)+y*(-.8+s)-n*np.logaddexp(0,-.8+s);truth=logsumexp(np.log(w[:,None])+np.log(w[None,:])+obs);d=dict(dt=np.array([0.]),reset=np.array([True]),n=np.array([n]),y=np.array([y]),exposure=np.array([risk]))
        for guided in [False,True]:
            ll=evaluate(d,np.tile(pars,(256,1)),1024,930000+k*1000+int(guided),64,guided);estimate=logsumexp(ll)-np.log(len(ll));error=float(estimate-truth);assert abs(error)<.02;rows.append(dict(c=c,n=n,y=y,risk=risk,guided=guided,quadrature_loglik=float(truth),particle_logmean=float(estimate),error=error,loglik_sd=float(np.std(ll))))
    write_json(OUT/'one_step_canary.json',dict(status='PASS',checks=rows,scope='Independent bivariate Gaussian quadrature versus bootstrap and defensive guided importance likelihood; empty observation and positive/negative couplings'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=1);ap.add_argument('--canary-only',action='store_true');ap.add_argument('--seconds',type=int,default=5);ap.add_argument('--replicates',type=int,default=128);args=ap.parse_args();cuda.select_device(args.gpu);canary()
    if args.canary_only:return
    jobs=[]
    for method in ['laplace','adf']:
        for coupled in [False,True]:
            pattern=f'b5_full_c{int(coupled)}_*.json' if method=='laplace' else f'full_c{int(coupled)}_*.json';folder='joint_two_state_v1_20' if method=='laplace' else 'joint_adf_refit_v1_23';choices=[json.loads(p.read_text()) for p in (RUN/folder/'fits').glob(pattern)];fit=max(choices,key=lambda r:r['loglik']);b,a,c,ts,ss,tr,sr=unpack(fit['theta'],coupled);jobs.append(dict(id=f'{method}_c{int(coupled)}_g{args.seconds}',method=method,coupled=coupled,theta=[b,a,c,np.log(ts),np.log(ss),np.log(tr),np.log(sr)],fit=fit))
    write_json(OUT/'contract.json',dict(question='Does the full coupled likelihood support the apparent ADF or Laplace parameter improvement?',likelihood='Two-state importance particle filter with exact OU transitions, rate plus mark observation likelihood, defensive 10% prior / 90% Gaussian proposal, exact proposal correction and paired-state resampling',n_replicates=args.replicates,particles=1024,seconds=args.seconds,limits='Monte Carlo log-mean uncertainty and likelihood-weight ESS must be checked; agreement of point estimates alone is insufficient'))
    d=prepare(args.seconds,.25)
    for k,j in enumerate(jobs):
        path=OUT/(j['id']+'.json')
        if path.exists():continue
        start=time.time();ll=evaluate(d,np.tile(j['theta'],(args.replicates,1)),seed=940000+k*10000);weights=np.exp(ll-logsumexp(ll));r=dict(status='COMPLETE',job=j,elapsed=time.time()-start,loglik_replicates=ll,particle_logmean=float(logsumexp(ll)-np.log(len(ll))),loglik_sd=float(np.std(ll)),likelihood_ess=float(1/np.sum(weights**2)));write_json(path,r);print(json.dumps({k:v for k,v in r.items() if k not in ['job','loglik_replicates']}),flush=True)
    write_json(OUT/f'status_g{args.seconds}.json',dict(status='COMPLETE',n_models=len(jobs)))
if __name__=='__main__':main()
