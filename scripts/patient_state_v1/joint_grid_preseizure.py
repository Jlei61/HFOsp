"""Causal pre-seizure queries from a full joint state grid, no Gaussian projection.

Fit parameters stay fixed at existing preinterval fits. Only current-epoch
completed observations enter the query; the full posterior is propagated from
the observation midpoint to query time, including silent observed exposure.
"""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd,cupy as cp
import cupyx.scipy.sparse as csp
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.joint_state_grid import grid,matrix,WEIGHT
from scripts.patient_state_v1.joint_two_state import unpack
from scripts.patient_state_v1.joint_two_state_filter import evaluate as adf_evaluate
from scripts.patient_state_v1.review_joint_preseizure import query as adf_query
from scripts.patient_state_v1.renewal import prepare
from scripts.patient_state_v1.preseizure import covered
OUT=RUN/'joint_grid_preseizure_v1_25'

def filter_queries(d,theta,coupled,times,size):
    b,a,c,ts,ss,tr,sr=unpack(theta,coupled);xs,ps,dxs=grid(ss,size);xr,pr,dxr=grid(sr,size);s=cp.asarray(xs[:,None]);r=cp.asarray(xr[None,:]);init=cp.asarray(np.outer(ps,pr));p=init.copy();operators={};idx=np.searchsorted(d['hi'],times,side='right')-1;out=np.empty((len(times),7));last=-1;maxloss=0.;peakedge=0.
    def propagate(mass,dt):
        dt=round(float(dt),10)
        if dt<=0:return mass
        if dt not in operators:operators[dt]=(csp.csr_matrix(matrix(xs,dxs,ts,ss,dt)),csp.csr_matrix(matrix(xr,dxr,tr,sr,dt)))
        T,U=operators[dt];return (U@(T@mass).T.copy()).T.copy()
    for qi,target in enumerate(idx):
        for i in range(last+1,target+1):
            p=init.copy() if d['reset'][i] else propagate(p,d['dt'][i]);mass=float(p.sum().get());maxloss=max(maxloss,1-mass);n=float(d['n'][i]);y=float(d['y'][i]);risk=float(d['exposure'][i]);shift=n*np.log(n/risk)-n if n>0 and risk>0 else (n*(a+abs(c)*max(abs(xs))+max(abs(xr))) if n>0 else 0.)
            WEIGHT(p,s,r,n,y,risk,shift,b,a,c,p);p/=p.sum()
        last=int(target)
        if target<0:q=init.copy()
        else:
            assert d['hi'][target]<=times[qi]
            q=propagate(p,(times[qi]-(d['lo'][target]+d['hi'][target])/2)/3600)
        mass=float(q.sum().get());maxloss=max(maxloss,1-mass);q=q/mass;sm=q.sum(axis=1);mean=float((sm*cp.asarray(xs)).sum().get());mean2=float((sm*cp.asarray(xs*xs)).sum().get());prob=1/(1+cp.exp(-(b+s)));hazard=cp.exp(cp.clip(a+c*s+r,-700,60));rate=hazard/(1+hazard*.25/3600);pp=float((q*prob).sum().get());total=float((q*rate).sum().get());tb=float((q*rate*prob).sum().get());cdf=cp.asnumpy(cp.cumsum(sm));quant=np.interp([.025,.975],cdf,xs)+b;out[qi]=[b+mean,np.sqrt(max(mean2-mean*mean,0)),pp,total,tb,*quant];peakedge=max(peakedge,float((q[0,:].sum()+q[-1,:].sum()+q[:,0].sum()+q[:,-1].sum()).get()))
    assert np.isfinite(out).all()
    return dict(mode_state=out[:,0],mode_state_sd=out[:,1],uniform_tb_probability=out[:,2],expected_total_rate=out[:,3],expected_tb_rate=out[:,4],mode_state_lower=out[:,5],mode_state_upper=out[:,6],max_mass_loss=maxloss,max_edge_mass=peakedge)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=1);args=ap.parse_args();cp.cuda.Device(args.gpu).use();full=prepare(5,.25);inv=json.loads((RUN/'seizures.json').read_text());wins=pd.read_csv(RUN/'frozen_seizure_windows.csv');wins=wins[wins.window=='pre15'].sort_values('sz');ex=pd.read_csv(RUN/'exposure.csv');best={}
    for p in (RUN/'joint_preseizure_v1_22/fits').glob('*.json'):
        r=json.loads(p.read_text());j=r['job'];key=(j['sz'],j['coupled'])
        if key not in best or r['loglik']>best[key][1]['loglik']:best[key]=(p,r)
    assert len(best)==24;write_json(OUT/'contract.json',dict(question='Are pre-seizure state levels and opposite TB-case changes robust to a full joint posterior, rather than a Gaussian approximation?',sizes=[256,512],n_jobs=48,parameters='Existing strict preinterval Laplace fits unchanged; no seizure labels fitted',query='Full current-epoch posterior uses only completed bins and exact OU finite-volume propagation to query time',limits='5-second observation model and fitted parameter approximation remain; fixed-parameter state intervals exclude parameter uncertainty'))
    done=0
    for size in [256,512]:
        for row in wins.itertuples():
            onset=inv[row.sz-1]['onset'];previous=inv[row.sz-2]['offset'];ix=(full['lo']>=previous)&(full['hi']<=onset);d={k:v[ix].copy() for k,v in full.items() if np.ndim(v)>0};assert len(d['n']) and d['reset'][0];edges=np.arange(max(previous,onset-3600),onset,5.);duration=np.minimum(edges+5,onset)-edges;times=edges+duration/2;mask=covered(times,ex)
            for coupled in [False,True]:
                done+=1;path=OUT/'evaluations'/f'sz{row.sz}_c{int(coupled)}_g{size}.json'
                if path.exists():continue
                start=time.time();source,fit=best[(row.sz,coupled)];assert fit['job']['cutoff_epoch']<=previous;r=filter_queries(d,fit['theta'],coupled,times,size);adf=adf_evaluate(fit['theta'],d,coupled,80,True);aq=adf_query(times,fit['theta'],coupled,d,adf,inv);maxloss=r.pop('max_mass_loss');maxedge=r.pop('max_edge_mass');r.update(times=times,minutes=(times-onset)/60,covered=mask,duration=duration,adf_mode_state=aq['mode_state'],adf_uniform_tb_probability=aq['uniform_tb_probability']);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),**r)
                summary={}
                for key in ['mode_state','uniform_tb_probability']:
                    vals=[]
                    for low,high in [(-30,-15),(-15,0)]:
                        ok=mask&(r['minutes']>=low)&(r['minutes']<high);vals.append(float(np.average(r[key][ok],weights=duration[ok])) if ok.any() else None)
                    summary[key]=dict(prior15=vals[0],pre15=vals[1],delta=vals[1]-vals[0] if all(v is not None for v in vals) else None)
                record=dict(status='COMPLETE',sz=int(row.sz),label=row.label,coupled=coupled,grid=size,fit_source=str(source),elapsed=time.time()-start,max_mass_loss=maxloss,max_edge_mass=maxedge,summary=summary,max_adf_state_difference=float(np.max(abs(r['mode_state']-r['adf_mode_state']))),max_adf_probability_difference=float(np.max(abs(r['uniform_tb_probability']-r['adf_uniform_tb_probability']))));write_json(path,record);print(json.dumps(dict(done=done,total=48,sz=row.sz,coupled=coupled,grid=size,elapsed=record['elapsed'],summary=summary)),flush=True);cp.get_default_memory_pool().free_all_blocks()
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=48))

if __name__=='__main__':main()
