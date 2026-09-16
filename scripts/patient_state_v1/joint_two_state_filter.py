"""Independent Gaussian quadrature audit of joint-state Laplace solutions.

Sequential scalar moment projections are approximate and order dependent.
Orders and quadrature resolutions are explicitly compared, not hidden.
"""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numba import njit
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.renewal import prepare
from scripts.patient_state_v1.joint_two_state import unpack
OUT=RUN/'joint_two_state_filter_v1_20'

@njit(cache=True)
def kernel(dt,reset,n,y,risk,b,a,c,ts,ss,tr,sr,nodes,weights,rate_first):
    L=len(n);M=len(nodes);m=np.zeros(2);P=np.diag(np.array([ss*ss,sr*sr]));means=np.empty((L,2));variances=np.empty((L,3));terms=np.empty(L);rate_terms=np.empty(L);mark_terms=np.empty(L);mark_predict=np.empty(L);logweights=np.log(weights);logw=np.empty(M);zs=np.empty(M)
    for i in range(L):
        if reset[i]:m=np.zeros(2);P=np.diag(np.array([ss*ss,sr*sr]))
        else:
            rhos=np.array([np.exp(-dt[i]/ts),np.exp(-dt[i]/tr)]);m*=rhos
            for j in range(2):
                for k in range(2):P[j,k]*=rhos[j]*rhos[k]
            P[0,0]+=ss*ss*(-np.expm1(-2*dt[i]/ts));P[1,1]+=sr*sr*(-np.expm1(-2*dt[i]/tr))
        total=0.
        for stage in range(2):
            israte=(stage==0)==rate_first;v=np.array([c,1.]) if israte else np.array([1.,0.]);offset=a if israte else b;mean=np.dot(v,m);Pv=P@v;var=max(np.dot(v,Pv),1e-14);prediction=0.
            for j in range(M):
                z=mean+np.sqrt(2*var)*nodes[j];zs[j]=z;eta=offset+z
                if israte:lp=n[i]*eta-risk[i]*np.exp(min(eta,60.))
                else:
                    lp=y[i]*eta-n[i]*(max(eta,0.)+np.log1p(np.exp(-abs(eta))));prediction+=weights[j]/(1+np.exp(-eta))
                logw[j]=logweights[j]+lp
            peak=np.max(logw);w=np.exp(logw-peak);norm=w.sum();w/=norm;first=np.dot(w,zs);second=np.dot(w,zs*zs);pv=max(second-first*first,1e-12);K=Pv/var;m+=K*(first-mean)
            for j in range(2):
                for k in range(2):P[j,k]+=K[j]*K[k]*(pv-var)
            ll=peak+np.log(norm);total+=ll
            if israte:rate_terms[i]=ll
            else:mark_terms[i]=ll;mark_predict[i]=prediction
        means[i]=m;variances[i]=np.array([P[0,0],P[1,1],P[0,1]]);terms[i]=total
    return terms,means,rate_terms,mark_terms,mark_predict,variances

def evaluate(t,d,coupled,order=40,rate_first=True):
    params=unpack(t,coupled);nodes,w=hermgauss(order);ll,m,lr,lm,p,v=kernel(d['dt'],d['reset'],d['n'],d['y'],d['exposure'],*params,nodes,w/np.sqrt(np.pi),rate_first);return dict(loglik=float(ll.sum()),loglik_terms=ll,mean=m,rate_terms=lr,mark_terms=lm,predict_tb=p,variance=v)

def worker(j):
    path=OUT/'evaluations'/f"{j['id']}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time()
    try:
        r=json.loads(Path(j['source']).read_text());full=prepare(r['job']['seconds'],.25);f=evaluate(r['theta'],full,r['job']['coupled'],j['order'],j['rate_first']);end=r['job']['end'];ans=dict(status='COMPLETE',job=j,elapsed=time.time()-start,train_adf_loglik=float(f['loglik_terms'][:end].sum()),train_laplace_loglik=r['loglik']);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),**f)
    except Exception:ans=dict(status='FAILED',job=j,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(path,ans);return ans

def main():
    best={}
    for p in (RUN/'joint_two_state_v1_20/fits').glob('*.json'):
        r=json.loads(p.read_text());j=r['job'];key=(j['seconds'],j['scope'],j['coupled'])
        if r['status']=='COMPLETE' and (key not in best or r['loglik']>best[key][1]['loglik']):best[key]=(p,r)
    jobs=[]
    for (sec,scope,coupled),(p,r) in best.items():
        if sec not in [5,15]:continue
        for order in [40,80]:
            for rate_first in [False,True]:jobs.append(dict(id=f"b{sec}_{scope}_c{int(coupled)}_q{order}_rf{int(rate_first)}",source=str(p),seconds=sec,scope=scope,coupled=coupled,order=order,rate_first=rate_first))
    assert len(jobs)==64
    write_json(OUT/'contract.json',dict(question='Are coupling gains robust to a different likelihood approximation and observation-projection order?',n_jobs=len(jobs),orders=[40,80],projection_orders=['rate then mark','mark then rate'],limits='ADF remains approximate; joint bin log density is the valid combined score. Rate-then-mark probabilities condition on the observed bin count and are not next-event forecasts.'))
    with ProcessPoolExecutor(max_workers=16) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(done=i+1,total=len(jobs),id=r['job']['id'],status=r['status'],elapsed=r['elapsed'])),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs)))
if __name__=='__main__':main()
