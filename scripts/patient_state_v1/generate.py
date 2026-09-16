"""Autonomous 24-hour effective retained-event sequences, no patient event-time replay."""
import sys,json,time,argparse,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from numba import njit
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.analyze_first import best_fits

@njit(cache=True)
def simulate(hours,step,seed,a,b,tau_r,sd_r,tau_s,sd_s,c,kind,gamma,deadtime):
    np.random.seed(seed);duration=hours*3600.;capacity=int(duration/max(deadtime,.01))+2
    ts=np.empty(capacity);ys=np.empty(capacity,np.int8);tr=np.empty(int(np.ceil(duration/step)));ss=np.empty(len(tr));rr=np.empty(len(tr));nn=0
    r=np.random.normal()*sd_r;s=np.random.normal()*sd_s;last=-1e12;lastmark=.5
    for i in range(len(tr)):
        t=i*step;hi=min(t+step,duration)
        if i:
            ar=np.exp(-step/(tau_r*3600));as_=np.exp(-step/(tau_s*3600))
            r=ar*r+sd_r*np.sqrt(-np.expm1(-2*step/(tau_r*3600)))*np.random.normal()
            s=as_*s+sd_s*np.sqrt(-np.expm1(-2*step/(tau_s*3600)))*np.random.normal()
        if kind==1:eta=a+c*s
        elif kind==2:eta=a+r+c*s
        else:eta=a+r
        hazard=np.exp(min(eta,50.))/3600.;now=max(t,last+deadtime)
        while now<hi:
            event=now+np.random.exponential(1/hazard)
            if event>=hi:break
            logit=b+s+gamma*(lastmark-.5)*np.exp(-(event-last))
            probability=1/(1+np.exp(-logit));mark=np.random.uniform()<probability
            ts[nn]=event;ys[nn]=mark;nn+=1;last=event;lastmark=1.0 if mark else 0.0;now=event+deadtime
        tr[i]=t;ss[i]=s;rr[i]=r+c*s if kind==2 else (r if kind!=1 else c*s)
    return ts[:nn],ys[:nn],tr,ss,rr

def summaries(times,y,hours,segments=None,exposure=None,seizures=None):
    times=np.asarray(times);y=np.asarray(y,dtype=np.int64);seg=np.zeros(len(y),int) if segments is None else np.asarray(segments)
    valid=np.diff(seg)==0;dt=np.diff(times)[valid];same=(np.diff(y)==0)[valid];p=float(y.mean()) if len(y) else np.nan
    result=dict(n_events=len(y),rate_per_hour=len(y)/hours,tb_fraction=p,interval_seconds_quantiles=np.quantile(dt,[.01,.1,.5,.9,.99]) if len(dt) else [],
                adjacent_same=float(same.mean()) if len(same) else np.nan,adjacent_excess=float(same.mean()-(p*p+(1-p)**2)) if len(same) else np.nan,
                n_shorter_than_250ms=int(np.sum(dt<.25-1e-7)),windows={})
    for minutes in (1,5,15,60):
        width=minutes*60;extent=float(np.max(exposure[:,1])) if exposure is not None else hours*3600
        edges=np.arange(0,np.ceil(extent/width)*width+width,width);n=np.histogram(times,edges)[0];tb=np.histogram(times,edges,weights=y)[0];duration=np.full(len(n),width,dtype=float)
        assert np.all((tb>=0)&(tb<=n)), 'Mode counts must lie between zero and total counts'
        assert tb.sum()==y[(times>=edges[0])&(times<=edges[-1])].sum(), 'Histogram must conserve TB labels'
        if exposure is not None:
            duration[:]=0
            for lo,hi in exposure:duration+=np.maximum(0,np.minimum(edges[1:],hi)-np.maximum(edges[:-1],lo))
        else:duration[-1]=max(0,min(hours*3600,edges[-1])-edges[-2])
        eligible=duration>=.95*width
        if seizures is not None:
            for lo,hi in seizures:eligible&=~((edges[:-1]<hi)&(edges[1:]>lo))
        normalized=n[eligible]*width/duration[eligible];share=np.full(len(n),np.nan);ok=eligible&(n>=10);share[ok]=tb[ok]/n[ok]
        correlation=[]
        for lag in (1,2,4):
            good=np.isfinite(share[:-lag])&np.isfinite(share[lag:])
            correlation.append(float(np.corrcoef(share[:-lag][good],share[lag:][good])[0,1]) if good.sum()>=10 and np.std(share[:-lag][good])>0 and np.std(share[lag:][good])>0 else np.nan)
        result['windows'][str(minutes)]=dict(n_eligible=int(eligible.sum()),count_mean=float(np.mean(normalized)) if len(normalized) else np.nan,
            count_fano=float(np.var(normalized,ddof=1)/np.mean(normalized)) if len(normalized)>1 and np.mean(normalized)>0 else np.nan,
            tb_share_quantiles=np.quantile(share[ok],[.1,.25,.5,.75,.9]) if ok.any() else [],n_share_windows=int(ok.sum()),tb_share_autocorrelation_lags124=correlation)
    return result

def build_configs(seconds):
    rows=[]
    for p in (RUN/'renewal_observation_v1_3/fits').glob('*.json'):
        r=json.loads(p.read_text());j=r['job']
        if r['status']=='COMPLETE' and j['seconds']==seconds and j['deadtime']==.25 and j['scope']=='full':r['source']=str(p);rows.append(r)
    best={m:max([r for r in rows if r['model']==m],key=lambda r:r['loglik']) for m in ('mark','rate','shared')}
    a,lr,lsr=best['rate']['theta'];b,lts,lss=best['mark']['theta'];shared=best['shared']['theta'];data=np.load(RUN/'observations.npz');obs=json.loads((RUN/'data_audit.json').read_text());p=data['y'].sum()/data['n'].sum();rate=data['n'].sum()/obs['observed_interictal_hours'];constant_hazard=rate/(1-rate*.25/3600)
    common=dict(hours=24.,step=.5,deadtime=.25,a=a,b=b,tau_r=np.exp(lr),sd_r=np.exp(lsr),tau_s=np.exp(lts),sd_s=np.exp(lss),c=0.,kind=0,gamma=0.)
    cfg={
        'constant':dict(common,a=np.log(constant_hazard),b=np.log(p/(1-p)),sd_r=0.,sd_s=0.),
        'mode_only':dict(common,a=np.log(constant_hazard),sd_r=0.),
        'activity_only':dict(common,b=np.log(p/(1-p)),sd_s=0.),
        'independent':dict(common),
        'shared':dict(common,b=shared[0],a=shared[1],c=shared[2],tau_s=np.exp(shared[3]),sd_s=np.exp(shared[4]),sd_r=0.,kind=1)}
    h=[]
    for pp in (RUN/'advanced_controls_v1_2/fits').glob('full_ou_history*.json'):h.append(json.loads(pp.read_text()))
    hist=max(h,key=lambda r:r['loglik'])['theta'];cfg['independent_history']=dict(common,b=hist[0],gamma=hist[1],tau_s=np.exp(hist[2]),sd_s=np.exp(hist[3]))
    return cfg,best

def worker(job):
    out=Path(job['out']);path=out/f"{job['model']}_{job['rep']:03d}.json"
    if path.exists():
        previous=json.loads(path.read_text())
        if previous.get('status')=='COMPLETE':return previous
        failed=out/'failed_compile_attempt';failed.mkdir(exist_ok=True);path.replace(failed/path.name)
    start=time.time()
    try:
        cfg=job['config'];ts,y,t,s,r=simulate(seed=job['seed'],**cfg);result=summaries(ts,y,cfg['hours']);result.update(status='COMPLETE',job=job,elapsed=time.time()-start)
        if job['rep']<3:np.savez_compressed(path.with_suffix('.npz'),event_seconds=ts,label_tb=y,state_time_seconds=t[::20],s=s[::20],r=r[::20])
    except Exception:result=dict(status='FAILED',job=job,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(path,result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=int,default=5);ap.add_argument('--replicates',type=int,default=64);ap.add_argument('--workers',type=int,default=24);args=ap.parse_args();out=RUN/'autonomous_generator_v1_4'/f'fit_grid_{args.seconds}s';out.mkdir(parents=True,exist_ok=True)
    cfg,best=build_configs(args.seconds);write_json(out/'contract.json',dict(configs=cfg,fit_sources=best,scope='24-hour autonomous effective observed-event sequences; no patient event-time or seizure-type inputs',
        observation_constraint='250ms minimum spacing models retained-event support; not biological refractoriness or full packing reconstruction',
        comparison='same summary definitions; real windows require >=95% coverage and no seizure overlap; simulation has complete coverage',
        claim_limit='generation assesses distribution, not reconstruction of a particular patient path or seizure timing'))
    ev=pd.read_csv(RUN/'events.csv');ex=pd.read_csv(RUN/'exposure.csv');inv=json.loads((RUN/'seizures.json').read_text());z=np.load(RUN/'observations.npz');origin=float(z['origin_epoch']);audit=json.loads((RUN/'data_audit.json').read_text())
    real=summaries(ev.start_epoch.to_numpy()-origin,ev.label_tb.to_numpy(),audit['observed_interictal_hours'],segments=ev.coverage_segment.to_numpy()*1000+ev.interictal_epoch.to_numpy(),
        exposure=ex[['start_epoch','end_epoch']].to_numpy()-origin,seizures=np.array([[s['onset'],s['offset']] for s in inv])-origin)
    write_json(out/'real_summary.json',real)
    jobs=[dict(model=m,rep=i,seed=290000+j*1000+i,config=c,out=str(out)) for j,(m,c) in enumerate(cfg.items()) for i in range(args.replicates)];write_json(out/'queue.json',jobs)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(complete=i+1,total=len(jobs),model=r['job']['model'],rep=r['job']['rep'],status=r['status'],rate=r.get('rate_per_hour'),tb=r.get('tb_fraction'))),flush=True)
    write_json(out/'status.json',dict(status='COMPLETE',n_jobs=len(jobs),finished_unix=time.time()))

if __name__=='__main__':main()
