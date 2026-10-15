"""Label-free, shaft-balanced extension of the established three-component loss.

Geometry initialization is separate. This artifact contains no mode classifier,
mode target, or held-out event. Missing contacts never receive invented ranks.
"""
from itertools import combinations
import warnings
import numpy as np
from scipy.stats import rankdata
from modeling.cohort_helpers import GROUPS, DISPLAY, shafts

def ranks(t):
    t=np.asarray(t,float);m=np.isfinite(t)
    if t.ndim!=2 or np.isinf(t).any() or np.any(m.sum(1)<2):raise ValueError('Invalid event times')
    r=np.where(m,rankdata(t,axis=1,method='average',nan_policy='omit')-1,0.)
    return r/np.maximum(m.sum(1)-1,1)[:,None],m

def statistic(total,norm,n,target):
    mean=total/n;v=float(norm/n-mean@mean)
    if v < -1e-8:raise ArithmeticError('Negative feature variance')
    v=max(v,0.)  # roundoff only; NEVER truncate D_off.
    a=float(np.sum((mean-target)**2));b=v/(n-1)
    return dict(N=n,A=a,B=b,D_off=a-b,D16=a+(n-16)*v/(16*(n-1)))

class CohortObjective:
    version='cohort_multishaft_label_free_v1'
    def __init__(self,names,seed,n_fourier=1024):
        self.names=list(names);self.shafts=shafts(names);self.seed=seed;self.n_fourier=n_fourier
        if len(self.shafts)<2:raise ValueError('Primary cohort needs >=2 shafts')
        self.pairs={k:np.array(list(combinations(ix,2)),int) for k,ix in self.shafts.items() if len(ix)>1}
        if not self.pairs:raise ValueError('No within-shaft pair')
        self.shaft_pairs=np.array(list(combinations(range(len(self.shafts)),2)),int)
        self.time_scales={};self.maps={};self.targets={};self.scales=None

    def raw_features(self,t,fit=False):
        t=np.asarray(t,float);r,m=ranks(t);nshaft=len(self.shafts)
        out=dict(rank=np.column_stack([np.column_stack([m[:,ix],r[:,ix]])/np.sqrt(2*len(ix)) for ix in self.shafts.values()])/np.sqrt(nshaft),
                 mask=np.column_stack([m[:,ix]/np.sqrt(len(ix)) for ix in self.shafts.values()])/np.sqrt(nshaft))
        def add(name,d):
            valid=np.isfinite(d)
            if fit:self.time_scales[name]=max(2.,float(np.median(abs(d[valid]))) if valid.any() else 2.)
            d=np.where(valid,d,0.)
            out[name]=np.column_stack([valid,np.sign(d),np.arcsinh(d/self.time_scales[name])])/np.sqrt(3*d.shape[1])
        for k,p in self.pairs.items():add('within_'+k,t[:,p[:,1]]-t[:,p[:,0]])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            med=np.column_stack([np.nanmedian(t[:,ix],axis=1) for ix in self.shafts.values()])
        p=self.shaft_pairs
        # Sign agrees with historical median(SCL)-median(ICL) for its two-shaft order.
        add('between_shafts',med[:,p[:,0]]-med[:,p[:,1]])
        return out

    def fit_maps(self,t):
        rng=np.random.default_rng(self.seed)
        # Kernel estimation subset is fixed without looking at model outcomes.
        ids=np.sort(rng.choice(len(t),min(8192,len(t)),replace=False))
        for key,x in self.raw_features(t[ids],fit=True).items():
            a=rng.integers(len(x),size=4096);b=rng.integers(len(x),size=4096)
            d=np.linalg.norm(x[a]-x[b],axis=1);good=d[d>1e-12];bw=float(np.median(good)) if len(good) else 1.
            self.maps[key]=dict(bandwidth=bw,linear_scale=max(float(np.sqrt(np.mean(np.sum(x*x,axis=1)))),1e-12),
                weights=np.column_stack([rng.normal(size=(x.shape[1],self.n_fourier))/(bw*s) for s in [.5,1,2]]),phases=rng.uniform(0,2*np.pi,3*self.n_fourier))

    def features(self,t,device='cpu'):
        mapped={}
        for k,x in self.raw_features(t).items():
            s=self.maps[k]
            if device=='cpu':y=np.cos(x@s['weights']+s['phases'])*np.sqrt(2/(3*self.n_fourier))
            else:
                import torch
                with torch.no_grad():
                    y=(torch.cos(torch.as_tensor(x,device=device)@torch.as_tensor(s['weights'],device=device)+torch.as_tensor(s['phases'],device=device))*np.sqrt(2/(3*self.n_fourier))).cpu().numpy()
            mapped[k]=np.column_stack([x/s['linear_scale'],y])/np.sqrt(2) if k in ('rank','mask') else y
        time=[x for k,x in mapped.items() if k not in ('rank','mask')]
        return dict(rank_pattern=mapped['rank'],participation=mapped['mask'],local_and_interrod_timing=np.column_stack(time)/np.sqrt(len(time)))

    def moments(self,t,device='cpu'):
        sums={}
        for start in range(0,len(t),128):
            for g,x in self.features(t[start:start+128],device).items():
                if g not in sums:sums[g]=dict(n=0,total=np.zeros(x.shape[1]),norm=0.)
                s=sums[g];s['n']+=len(x);s['total']+=x.sum(0);s['norm']+=float(np.sum(x*x))
        return sums

    def fit_targets(self,t,device='cpu'):
        self.targets={g:s['total']/s['n'] for g,s in self.moments(t,device).items()}

    def calibrate(self,t,blocks,device='cpu'):
        rng=np.random.default_rng(self.seed+1)
        pools=[np.flatnonzero(blocks==b) for b in np.unique(blocks)]
        pools=[p for p in pools if len(p)>=16]
        if not pools:raise ValueError('No CAL block with >=16 qualifying events')
        draws=[]
        for _ in range(256):
            pool=pools[rng.integers(len(pools))];start=rng.integers(len(pool)-15);draws.append(pool[start:start+16])
        used,inverse=np.unique(np.asarray(draws),return_inverse=True);local=inverse.reshape(256,16)
        phi={g:[] for g in GROUPS}
        for start in range(0,len(used),128):
            for g,x in self.features(t[used[start:start+128]],device).items():phi[g].append(x)
        values={}
        for g,parts in phi.items():
            x=np.concatenate(parts)
            values[g]=[float(np.sum((x[ix].mean(0)-self.targets[g])**2)) for ix in local]
        self.scales={g:max(float(np.median(v)),1e-8) for g,v in values.items()}
        return dict(scales=self.scales,draw_indices=np.asarray(draws),A_values=values)

    def score(self,t,device='cpu'):
        n=len(t)
        if n<16:return dict(status='INSUFFICIENT_EVENTS',N=n,J=None,groups={})
        groups={}
        for g,s in self.moments(t,device).items():
            v=statistic(s['total'],s['norm'],n,self.targets[g]);v['scaled']=v['D_off']/self.scales[g];groups[g]=v
        return dict(status='SCORABLE',N=n,J=float(np.mean([groups[g]['scaled'] for g in GROUPS])),groups=groups)

def summaries(t,names):
    t=np.asarray(t,float);r,m=ranks(t)
    counts=m.sum(0);mean=np.divide((r*m).sum(0),counts,out=np.full(t.shape[1],np.nan),where=counts>0)
    pair={}
    for k,ix in shafts(names).items():
        p=np.array(list(combinations(ix,2)),int)
        if not len(p):continue
        d=t[:,p[:,1]]-t[:,p[:,0]];v=np.isfinite(d);n=v.sum(0)
        # Ties split the mass equally; no absent pair is assigned probability zero.
        prob=np.divide(((d>0)+.5*(d==0)).sum(0),n,out=np.full(len(p),np.nan),where=n>0)
        pair[k]=dict(probability=prob,count=n,pairs=p)
    return dict(mean_rank=mean,rank_count=counts,participation=m.mean(0),pair=pair)

def descriptive(model,target,names):
    if not len(model):return {k:None for k in DISPLAY}
    a=summaries(model,names);b=target;rank=[];part=[];orders=[];coverage={}
    rank_support=True
    for k,ix in shafts(names).items():
        valid=np.isfinite(a['mean_rank'][ix])&np.isfinite(b['mean_rank'][ix])
        rank_support=rank_support and bool(np.all(np.isfinite(a['mean_rank'][ix][np.isfinite(b['mean_rank'][ix])])) )
        rank.append(float(np.mean(abs(a['mean_rank'][ix][valid]-b['mean_rank'][ix][valid]))) if valid.any() else np.nan)
        part.append(float(np.mean(abs(a['participation'][ix]-b['participation'][ix]))))
        if k in a['pair']:
            ap=a['pair'][k]['probability'];bp=b['pair'][k]['probability'];v=np.isfinite(ap)&np.isfinite(bp)
            orders.append(float(np.mean(abs(ap[v]-bp[v]))) if v.any() else np.nan)
            coverage[k]=dict(estimated_pairs=int(v.sum()),total_pairs=int(np.isfinite(bp).sum()),all_anatomical_pairs=len(v))
    # Complete per-shaft support is required for a primary aggregate. Partial
    # values are shown separately; omitting unavailable shafts must not look good.
    full=all(q['estimated_pairs']==q['total_pairs'] for q in coverage.values()) and rank_support and np.isfinite(rank).all() and np.isfinite(orders).all()
    out=dict(zip(DISPLAY,[float(np.mean(rank)),float(np.mean(orders)),float(np.mean(part))]))
    out.update(full_contact_pair_support=full,pair_coverage=coverage)
    return out
