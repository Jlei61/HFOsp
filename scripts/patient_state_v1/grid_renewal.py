"""Deterministic finite-volume state-grid check of renewal OU likelihood.

Integrate each Gaussian transition over destination cells, truncate at eight
innovation SD, and do not renormalize lost boundary mass. Grid refinement, rather
than agreement with Laplace or noisy particles, determines numerical accuracy.
"""
import sys,json,time,math,argparse,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from numba import njit
from numba.typed import List
from scipy.special import ndtr
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.renewal import prepare

@njit(cache=True)
def transition(grid,dx,rho,innovation_sd):
    size=len(grid);ptr=np.zeros(size+1,np.int64)
    for i in range(size):
        lo=max(0,int(math.floor((rho*grid[i]-8*innovation_sd-grid[0])/dx)))
        hi=min(size,int(math.ceil((rho*grid[i]+8*innovation_sd-grid[0])/dx))+1)
        ptr[i+1]=ptr[i]+max(hi-lo,0)
    dest=np.empty(ptr[-1],np.int32);value=np.empty(ptr[-1]);root2=math.sqrt(2.)
    for i in range(size):
        lo=max(0,int(math.floor((rho*grid[i]-8*innovation_sd-grid[0])/dx)))
        for k in range(ptr[i],ptr[i+1]):
            j=lo+k-ptr[i];mean=rho*grid[i]
            value[k]=.5*(math.erf((grid[j]+dx/2-mean)/(innovation_sd*root2))-math.erf((grid[j]-dx/2-mean)/(innovation_sd*root2)))
            dest[k]=j
    return ptr,dest,value

@njit(cache=True)
def filtering(grid,initial,indices,reset,n,risk,ptrs,dests,values,a):
    size=len(grid);posterior=initial.copy();ll=0.;maxlost=0.;total_lost=0.;edge_max=0.
    means=np.empty(len(n));variances=np.empty(len(n));terms=np.empty(len(n))
    for i in range(len(n)):
        if reset[i]:prior=initial.copy()
        else:
            prior=np.zeros(size);k=indices[i];ptr=ptrs[k];dest=dests[k];v=values[k]
            for origin in range(size):
                mass=posterior[origin]
                if mass>1e-300:
                    for j in range(ptr[origin],ptr[origin+1]):prior[dest[j]]+=mass*v[j]
            loss=max(0.,1-prior.sum());maxlost=max(maxlost,loss);total_lost+=loss
        obs=n[i]*(a+grid)-risk[i]*np.exp(np.minimum(a+grid,50.));mx=np.max(obs)
        posterior=prior*np.exp(obs-mx);norm=posterior.sum()
        if norm<=0:return -np.inf,maxlost,total_lost,edge_max,means,variances,terms
        terms[i]=mx+math.log(norm);ll+=terms[i];posterior/=norm;means[i]=np.sum(posterior*grid);variances[i]=np.sum(posterior*(grid-means[i])**2)
        edge_max=max(edge_max,posterior[0]+posterior[-1])
    return ll,maxlost,total_lost,edge_max,means,variances,terms

def evaluate(d,theta,points=512,width_sd=8.,return_terms=False):
    a,lt,ls=np.asarray(theta);tau=np.exp(lt);sd=np.exp(ls);dx=2*width_sd*sd/points;grid=(np.arange(points)+.5)*dx-width_sd*sd
    edges=np.r_[grid-dx/2,grid[-1]+dx/2];initial=np.diff(ndtr(edges/sd))
    rounded=np.round(d['dt'],10);unique,indices=np.unique(rounded,return_inverse=True);ptrs=List();dests=List();values=List()
    for dt in unique:
        if dt==0:dt=1e-12
        rho=np.exp(-dt/tau);innov=sd*np.sqrt(-np.expm1(-2*dt/tau));p,j,v=transition(grid,dx,rho,innov);ptrs.append(p);dests.append(j);values.append(v)
    ll,lost,total,edge,mean,var,terms=filtering(grid,initial,indices,d['reset'],d['n'],d['exposure'],ptrs,dests,values,a)
    result=dict(loglik=ll,max_transition_mass_loss=lost,sum_transition_mass_loss=total,max_posterior_edge_mass=edge,mean=mean,variance=var,points=points,width_sd=width_sd,grid_spacing=dx)
    if return_terms:result['loglik_terms']=terms
    return result

def worker(job):
    out=RUN/'renewal_grid_audit_v1_6';path=out/f"{job['id']}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time()
    try:
        d=prepare(job['seconds'],.25);result=evaluate(d,job['theta'],job['points']);mean=result.pop('mean');var=result.pop('variance')
        result.update(status='COMPLETE',job=job,elapsed=time.time()-start)
        if job['shift']==0 and job['scale']==1:np.savez_compressed(path.with_suffix('.npz'),mean=mean,variance=var)
    except Exception:result=dict(status='FAILED',job=job,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(path,result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=12);ap.add_argument('--points',type=int,nargs='+',default=[256,512]);ap.add_argument('--seconds',type=int,default=5);args=ap.parse_args();out=RUN/'renewal_grid_audit_v1_6';out.mkdir(parents=True,exist_ok=True)
    fits=[]
    for p in (RUN/'renewal_observation_v1_3/fits').glob('*.json'):
        r=json.loads(p.read_text());j=r['job']
        if r['status']=='COMPLETE' and j['seconds']==args.seconds and j['deadtime']==.25 and j['scope']=='full' and j['model']=='rate':fits.append(r)
    best=max(fits,key=lambda r:r['loglik']);jobs=[]
    for points in args.points:
        for shift,scale in [(0,1),(-.5,1),(.5,1),(1,1),(0,.8),(0,1.2)]:
            t=np.array(best['theta']);t[0]+=shift;t[2]+=np.log(scale)
            jobs.append(dict(id=f'g{args.seconds}_m{points}_a{shift}_s{scale}',seconds=args.seconds,points=points,shift=shift,scale=scale,theta=t,base_laplace_loglik=best['loglik']))
    write_json(out/f'queue_g{args.seconds}_m{max(args.points)}.json',jobs)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(worker,j) for j in jobs]):
            r=f.result();print(json.dumps({k:r.get(k) for k in ('status','job','elapsed','loglik','max_transition_mass_loss','max_posterior_edge_mass')},default=lambda x:x.tolist()),flush=True)
    write_json(out/f'status_g{args.seconds}_m{max(args.points)}.json',dict(status='COMPLETE',n_jobs=len(jobs)))

if __name__=='__main__':main()
