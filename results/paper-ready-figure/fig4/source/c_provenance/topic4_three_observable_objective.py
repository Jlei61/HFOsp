"""Three frozen event-distribution components, with missingness and signed D_off.

No simulation or event-specific drive is defined here. GPU use only accelerates
the same float64 Fourier feature multiplication used on CPU.
"""
from itertools import combinations
import warnings
import numpy as np
from scipy.stats import rankdata
from src.topic4_interictal_repaired_evaluation import rank_features

GROUPS=('rank_pattern','local_and_interrod_timing','participation')

def validate(t):
    t=np.asarray(t,dtype=np.float64)
    if t.ndim!=2 or np.isinf(t).any() or np.any(np.isfinite(t).sum(1)<2):
        raise ValueError('events require >=2 participating contacts; missing values must be NaN')
    return t

def normalized_ranks(t):
    t=validate(t);m=np.isfinite(t);r=np.zeros_like(t)
    for i,row in enumerate(t):r[i,m[i]]=(rankdata(row[m[i]],method='average')-1)/(m[i].sum()-1)
    return r,m

def signed_statistic(sum_x,sum_norm,n,target,matched_count=16):
    if n<2:return dict(N=int(n),A=None,B=None,D_off=None,D16=None)
    mu=np.asarray(sum_x,dtype=float)/n
    v=float(sum_norm/n-np.dot(mu,mu))
    if v < -1e-8:raise ArithmeticError('negative feature variance beyond roundoff')
    a=float(np.sum((mu-target)**2));b=v/(n-1)
    return dict(N=int(n),A=a,B=b,D_off=a-b,
        D16=a+(n-matched_count)*v/(matched_count*(n-1)) if n>=matched_count else None)

class ThreeObservableObjective:
    version='three_observable_off_diagonal_v1'
    def __init__(self,names,km,proportions,*,seed=2026091401,n_fourier=1024):
        self.names=list(names);self.km=km;self.proportions=np.asarray(proportions,float)
        assert len(self.proportions)==2 and np.all(self.proportions>0)
        self.scl=np.array([i for i,n in enumerate(names) if n.startswith('SCL')]);self.icl=np.array([i for i,n in enumerate(names) if n.startswith('ICL')])
        assert len(self.scl)==4 and len(self.icl)==11
        self.pairs=[np.array(list(combinations(ix,2)),int) for ix in [self.scl,self.icl]]
        self.seed=seed;self.n_fourier=n_fourier;self.time_scales={};self.maps={};self.targets={};self.scales=None

    def labels(self,t):
        t=validate(t)
        return self.km.predict(rank_features(t)) if len(t) else np.array([],dtype=int)

    def raw_features(self,t,*,fit=False):
        t=validate(t);r,m=normalized_ranks(t);m=m.astype(float)
        parts=[np.column_stack([m[:,ix],r[:,ix]])/np.sqrt(len(ix)*2) for ix in [self.scl,self.icl]]
        result={'rank':np.column_stack(parts)/np.sqrt(2),
            'mask':np.column_stack([m[:,ix]/np.sqrt(len(ix)) for ix in [self.scl,self.icl]])/np.sqrt(2)}
        for name,pair in zip(['scl_pairs','icl_pairs'],self.pairs):
            delta=t[:,pair[:,1]]-t[:,pair[:,0]];valid=np.isfinite(delta)
            if fit:self.time_scales[name]=max(2.,float(np.median(abs(delta[valid]))) if valid.any() else 2.)
            delta=np.where(valid,delta,0.)
            result[name]=np.column_stack([valid,np.sign(delta),np.arcsinh(delta/self.time_scales[name])])/np.sqrt(3*len(pair))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            lag=np.nanmedian(t[:,self.scl],axis=1)-np.nanmedian(t[:,self.icl],axis=1)
        valid=np.isfinite(lag)
        if fit:self.time_scales['rod_lag']=max(2.,float(np.median(abs(lag[valid]))) if valid.any() else 2.)
        lag=np.where(valid,lag,0.)
        result['rod_lag']=np.column_stack([valid,np.sign(lag),np.arcsinh(lag/self.time_scales['rod_lag'])])/np.sqrt(3)
        return result

    def fit_maps(self,fit):
        x=self.raw_features(fit,fit=True);rng=np.random.default_rng(self.seed)
        for key,v in x.items():
            a=rng.integers(len(v),size=4096);b=rng.integers(len(v),size=4096);dist=np.linalg.norm(v[a]-v[b],axis=1);good=dist[dist>1e-12]
            bw=float(np.median(good)) if len(good) else 1.
            self.maps[key]=dict(bandwidth=bw,degenerate=not len(good),linear_scale=max(float(np.sqrt(np.mean(np.sum(v*v,axis=1)))),1e-12),
                weights=np.column_stack([rng.normal(size=(v.shape[1],self.n_fourier))/(bw*s) for s in [.5,1,2]]),
                phases=rng.uniform(0,2*np.pi,size=3*self.n_fourier))

    def mapped(self,raw,device='cpu'):
        result={}
        if device!='cpu':
            import torch
            torch.set_num_threads(2)
        for key,x in raw.items():
            spec=self.maps[key]
            if device=='cpu':y=np.cos(x@spec['weights']+spec['phases'])*np.sqrt(2/(3*self.n_fourier))
            else:
                import torch
                with torch.no_grad():
                    y=(torch.cos(torch.as_tensor(x,device=device)@torch.as_tensor(spec['weights'],device=device)+torch.as_tensor(spec['phases'],device=device))*np.sqrt(2/(3*self.n_fourier))).cpu().numpy()
            result[key]=np.column_stack([x/spec['linear_scale'],y])/np.sqrt(2) if key in ['rank','mask'] else y
        return dict(rank_pattern=result['rank'],participation=result['mask'],
            local_and_interrod_timing=np.column_stack([result[k] for k in ['scl_pairs','icl_pairs','rod_lag']])/np.sqrt(3))

    def features(self,t,device='cpu'):return self.mapped(self.raw_features(t),device)

    def moments(self,t,labels=None,device='cpu'):
        t=validate(t);labels=self.labels(t) if labels is None else np.asarray(labels,int)
        assert labels.shape==(len(t),) and np.all(np.isin(labels,[0,1]))
        stats={}
        for start in range(0,len(t),256):
            lab=labels[start:start+256];features=self.features(t[start:start+256],device)
            for group,phi in features.items():
                if group not in stats:stats[group]=dict(n=0,sum=np.zeros(phi.shape[1]),norm=0.,counts=np.zeros(2,int),mode_sum=np.zeros((2,phi.shape[1])),mode_norm=np.zeros(2))
                s=stats[group];norm=np.sum(phi*phi,axis=1);s['n']+=len(phi);s['sum']+=phi.sum(0);s['norm']+=norm.sum()
                for k in range(2):
                    ix=lab==k;s['counts'][k]+=int(ix.sum());s['mode_sum'][k]+=phi[ix].sum(0);s['mode_norm'][k]+=norm[ix].sum()
        return stats

    def psi_moments(self,s):
        terms=[s['sum']/np.sqrt(2)];norm=.5*s['norm']
        for k,p in enumerate(self.proportions):
            terms.append(.5*np.r_[s['counts'][k],s['mode_sum'][k]]/p)
            norm+=.25*(s['counts'][k]+s['mode_norm'][k])/(p*p)
        return np.concatenate(terms),norm

    def fit_targets(self,fit,labels,device='cpu'):
        for group,s in self.moments(fit,labels,device).items():
            total,_=self.psi_moments(s)
            self.targets[group]=dict(psi=total/s['n'],global_mean=s['sum']/s['n'],mode_means=s['mode_sum']/s['counts'][:,None])

    def score(self,t,labels=None,device='cpu'):
        t=validate(t);labels=self.labels(t) if labels is None else np.asarray(labels,int)
        assert labels.shape==(len(t),) and np.all(np.isin(labels,[0,1]))
        n=len(t);out=dict(version=self.version,status='SCORABLE' if n>=16 else 'INSUFFICIENT_EVENTS',N=n,
            mode_counts={'TA':int((labels==1).sum()),'TB':int((labels==0).sum())},groups={})
        if n<16:out.update(J=None,components=None);return out
        for group,s in self.moments(t,labels,device).items():
            target=self.targets[group];total,norm=self.psi_moments(s);v=signed_statistic(total,norm,n,target['psi'])
            v['scaled']=None if self.scales is None else v['D_off']/self.scales[group]
            v['conditional']={}
            for k,name in [(1,'TA'),(0,'TB')]:
                v['conditional'][name]=signed_statistic(s['mode_sum'][k],s['mode_norm'][k],int(s['counts'][k]),target['mode_means'][k])
            out['groups'][group]=v
        out['components']=None if self.scales is None else [out['groups'][g]['scaled'] for g in GROUPS]
        out['J']=None if self.scales is None else float(np.mean(out['components']))
        return out

    def calibrate(self,cal,blocks,device='cpu'):
        blocks=np.asarray(blocks);unique=np.unique(blocks);rng=np.random.default_rng(self.seed+10)
        byblock={str(b):np.flatnonzero(blocks==b) for b in unique};eligible=[b for b,ix in byblock.items() if len(ix)>=16]
        draws=[];values={g:[] for g in GROUPS}
        for j in range(256):
            if eligible:
                b=eligible[int(rng.integers(len(eligible)))];ix=byblock[b];start=int(rng.integers(len(ix)-15));ids=ix[start:start+16];used=[b]
            else:
                order=rng.permutation(len(unique));seq=[byblock[str(unique[i])] for i in order];ids=np.concatenate(seq)[:16];used=[str(blocks[i]) for i in ids]
            assert len(ids)==16
            sc=self.score(cal[ids],device=device)
            for g in GROUPS:values[g].append(sc['groups'][g]['A'])
            draws.append(dict(draw=j,cal_indices=ids.tolist(),blocks=used))
        self.scales={g:max(float(np.median(v)),1e-8) for g,v in values.items()}
        return dict(scales=self.scales,draws=draws,statistic='nonnegative squared embedding mean distance A',sample_count=16,
            row_order='within stored CAL block index order; chronological meaning must be independently checked',seed=self.seed+10)
