"""v1.1: binned count/mark observation models, separate from exact-time mark v1.

NB counts account phenomenologically for dispersion in the selected event process.
This model does not claim to reconstruct sub-bin intervals or pre-packing detections.
"""
import sys,json,time,argparse,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.special import expit,gammaln,logsumexp
from scipy.optimize import minimize
from scipy.linalg import solveh_banded,cholesky_banded
from numpy.polynomial.hermite import hermgauss
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import prior,qmul

OUT=RUN/'joint_counts_v1_1'

def prepare(seconds):
    path=OUT/f'bins_{seconds}s.npz'
    if path.exists():return dict(np.load(path))
    OUT.mkdir(parents=True,exist_ok=True)
    events=pd.read_csv(RUN/'events.csv');exposure=pd.read_csv(RUN/'exposure.csv');inv=json.loads((RUN/'seizures.json').read_text());origin=float(np.load(RUN/'observations.npz')['origin_epoch']);rows=[]
    for rr in exposure.itertuples():
        a,b=rr.start_epoch,rr.end_epoch;internal=np.arange(np.floor((a-origin)/seconds)+1,np.ceil((b-origin)/seconds))*seconds+origin
        edges=np.r_[a,internal[(internal>a)&(internal<b)],b];ev=events[(events.start_epoch>=a)&(events.start_epoch<b)]
        n=np.histogram(ev.start_epoch,edges)[0];y=np.histogram(ev.start_epoch,edges,weights=ev.label_tb)[0]
        for i in range(len(n)):rows.append((edges[i],edges[i+1],n[i],y[i]))
    arr=np.array(rows);t=((arr[:,0]+arr[:,1])/2-origin)/3600;epoch=np.searchsorted([r['offset'] for r in inv],(arr[:,0]+arr[:,1])/2,side='right');reset=np.r_[True,np.diff(epoch)!=0];dt=np.r_[0,np.diff(t)];dt[reset]=0
    d=dict(t=t,dt=dt,reset=reset,epoch=epoch,exposure=(arr[:,1]-arr[:,0])/3600,n=arr[:,2],y=arr[:,3],lo=arr[:,0],hi=arr[:,1],origin_epoch=np.array(origin))
    assert d['n'].sum()==len(events) and d['y'].sum()==events.label_tb.sum() and np.all(dt[~reset]>0)
    np.savez_compressed(path,**d);return d

def unpack(theta,model):
    if model=='mark':b,lt,ls=theta;return b,0.,0.,np.exp(lt),np.exp(ls),1.
    if model=='rate':a,lt,ls,lk=theta;return 0.,a,1.,np.exp(lt),np.exp(ls),np.exp(lk)
    b,a,c,lt,ls,lk=theta;return b,a,c,np.exp(lt),np.exp(ls),np.exp(lk)

def observation(s,theta,d,model):
    b,a,c,tau,sd,k=unpack(theta,model);n,y,ex=d['n'],d['y'],d['exposure']
    ll=np.zeros_like(s);g=np.zeros_like(s);h=np.zeros_like(s)
    if model!='rate':
        eta=b+s;p=expit(eta)
        ll+=gammaln(n+1)-gammaln(y+1)-gammaln(n-y+1)+y*eta-n*np.logaddexp(0,eta)
        g+=n*p-y;h+=n*p*(1-p)
    if model!='mark':
        eta=np.log(ex)+a+c*s;mu=np.exp(np.clip(eta,-700,60));logsum=np.logaddexp(np.log(k),eta)
        ll+=gammaln(n+k)-gammaln(k)-gammaln(n+1)+k*(np.log(k)-logsum)+n*(eta-logsum)
        g+=c*k*(mu-n)/(k+mu);h+=c*c*k*mu*(k+n)/(k+mu)**2
    return ll,g,h

def marginal(theta,d,model,state=False,observation_fn=observation):
    b,a,c,tau,sd,k=unpack(theta,model);_,_,diag,off,ldq=prior(d['dt'],d['reset'],tau,sd);s=np.zeros(len(d['n']))
    def objective(s):return -observation_fn(s,theta,d,model)[0].sum()+.5*s@qmul(s,diag,off)
    value=objective(s);converged=False
    for it in range(80):
        ll,g,h=observation_fn(s,theta,d,model);band=np.zeros((2,len(s)));band[0]=diag+h;band[1,:-1]=off;grad=qmul(s,diag,off)+g
        delta=solveh_banded(band,grad,lower=True,check_finite=False);step=1.
        for _ in range(25):
            trial=s-step*delta;v=objective(trial)
            if v<=value+1e-8:break
            step*=.5
        s=trial;value=v
        if max(abs(step*delta))<1e-6:converged=True;break
    ll,g,h=observation_fn(s,theta,d,model);band[0]=diag+h;chol=cholesky_banded(band,lower=True,check_finite=False);result=-value+.5*(ldq-2*np.log(chol[0]).sum())
    if state:return dict(loglik=result,mode=s,converged=converged)
    return result if converged else result-1e3

def fit_model(d,model,init,maxiter=150):
    timebound=(np.log(1/60),np.log(24));sdbound=(np.log(.01),np.log(5));kb=(np.log(.02),np.log(1e4))
    bounds={'mark':[(-8,8),timebound,sdbound],'rate':[(np.log(.1),np.log(1e5)),timebound,sdbound,kb],
            'shared':[(-8,8),(np.log(.1),np.log(1e5)),(-8,8),timebound,sdbound,kb]}[model]
    def f(t):
        try:return -marginal(t,d,model)
        except (ValueError,np.linalg.LinAlgError,FloatingPointError):return 1e20
    r=minimize(f,np.array(init),method='L-BFGS-B',bounds=bounds,options={'maxiter':maxiter,'ftol':1e-10,'eps':1e-5,'maxls':25})
    return dict(theta=r.x,loglik=-r.fun,success=r.success,nfev=r.nfev,message=r.message,model=model)

def filter_model(theta,d,model,order=64):
    b,a,c,tau,sd,k=unpack(theta,model);nodes,w=hermgauss(order);w/=np.sqrt(np.pi);lw=np.log(w);m=0.;v=sd*sd
    means=[];vars=[];lls=[];rates=[];markprobs=[];tbrates=[]
    for i in range(len(d['n'])):
        if d['reset'][i]:m=0.;v=sd*sd
        else:
            rho=np.exp(-d['dt'][i]/tau);m*=rho;v=rho*rho*v+sd*sd*(-np.expm1(-2*d['dt'][i]/tau))
        s=m+np.sqrt(2*v)*nodes;one={key:np.full(len(s),d[key][i]) for key in ('n','y','exposure')}
        lp=observation(s,theta,one,model)[0];norm=logsumexp(lw+lp);post=np.exp(lw+lp-norm)
        markprobs.append(np.dot(w,expit(b+s)) if model!='rate' else np.nan)
        rates.append(np.dot(w,np.exp(np.clip(a+c*s,-700,60))) if model!='mark' else np.nan)
        tbrates.append(np.dot(w,np.exp(np.clip(a+c*s,-700,60))*expit(b+s)) if model=='shared' else np.nan)
        m=np.dot(post,s);v=max(np.dot(post,s*s)-m*m,1e-10);means.append(m);vars.append(v);lls.append(norm)
    return dict(mean=np.array(means),variance=np.array(vars),loglik_terms=np.array(lls),predicted_rate=np.array(rates),prior_tb_probability=np.array(markprobs),predicted_tb_rate=np.array(tbrates))

def worker(job):
    path=OUT/'fits'/f"{job['id']}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time()
    try:
        full=prepare(job['seconds']);d={k:np.array(v[:job['end']],copy=True) for k,v in full.items() if v.ndim>0};p=np.clip(d['y'].sum()/d['n'].sum(),.001,.999);b=np.log(p/(1-p));a=np.log(d['n'].sum()/d['exposure'].sum())
        tau,sd,c=job['tau'],job['sd'],job.get('c',1.)
        init={'mark':[b,np.log(tau),np.log(sd)],'rate':[a,np.log(tau),np.log(sd),np.log(2)],'shared':[b,a,c,np.log(tau),np.log(sd),np.log(2)]}[job['model']]
        result=fit_model(d,job['model'],init,maxiter=160);result.update(status='COMPLETE',job=job,elapsed=time.time()-start)
    except Exception:result=dict(status='FAILED',job=job,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(path,result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=24);args=ap.parse_args();jobs=[];folds=json.loads((RUN/'splits.json').read_text());events=np.load(RUN/'observations.npz')
    for seconds in (60,300,15):
        d=prepare(seconds);scopes=[('full',len(d['n']))]
        if seconds!=15:
            for f in folds:
                cutoff=float(events['origin_epoch'])+events['t'][f['train_end']]*3600;end=int(np.searchsorted(d['hi'],cutoff,side='right'));scopes.append((f"fold{f['fold']}",end))
        for scope,end in scopes:
            for model in ('mark','rate','shared'):
                initials=[(.1,.5,1.),(1.,1.2,2.),(6.,.5,-2.)] if model=='shared' else [(.1,.5,1.),(1.,1.2,1.)]
                for i,(tau,sd,c) in enumerate(initials):jobs.append(dict(id=f'b{seconds}_{scope}_{model}_i{i}',seconds=seconds,scope=scope,end=end,model=model,tau=tau,sd=sd,c=c))
    write_json(OUT/'queue.json',jobs)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(complete=i+1,total=len(jobs),id=r['job']['id'],status=r['status'],elapsed=r['elapsed'],ll=r.get('loglik'))),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs),finished_unix=time.time()))

if __name__=='__main__':main()
