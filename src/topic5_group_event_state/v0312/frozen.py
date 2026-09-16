"""Frozen consumers share infer_asof; adapter outcomes never select the producer."""
from dataclasses import replace
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from scipy.stats import nbinom
from . import data as D
from .engine import infer_asof,predict,build_model,apply_ablations
from .prepare import Prepared
from .objective import interval_log_mean_rate,view_log_probs
from .train import RunConfig,evaluate,tensor_hash,atomic_json,atomic_torch,source_digest,file_hash


def load_selected(path,device='cpu'):
    if device.startswith('cuda'):torch.cuda.set_device(device)
    record=torch.load(path,weights_only=False,map_location='cpu')
    if record['source_digest']!=source_digest()[0]:raise ValueError('frozen source mismatch')
    cfg=RunConfig(**(record['config']|{'device':device}))
    packet_path=Path(cfg.packets_root)/f'{cfg.subject}.pt'
    if record.get('packets_sha256')!=file_hash(packet_path):raise ValueError('frozen measurement packets changed')
    payload=torch.load(packet_path,weights_only=False,map_location='cpu')
    prep=Prepared(payload,record['split'],record['scaling'],torch.device(device));apply_ablations(prep,cfg.crossview)
    model=build_model(prep,cfg.inputs,cfg.family,cfg.arm,cfg.seed);model.load_state_dict(record['state_dict']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,prep,cfg,record


@torch.no_grad()
def query_rows(model,prep,cfg,queries,role):
    for a in range(0,len(queries),cfg.eval_chunk):
        yield infer_asof(model,prep,np.asarray(queries[a:a+cfg.eval_chunk]),role,cfg.history_hours,
                         producer_hash=tensor_hash(model.state_dict()))


@torch.no_grad()
def fit_reference(model,prep,cfg,tau_hours=2.):
    queries=np.flatnonzero(prep.split['train_packet'])[::60]
    states=list(query_rows(model,prep,cfg,queries,'fit'))
    if not states or states[0].m is None:raise ValueError('state distribution requires a state producer')
    m=torch.cat([s.m for s in states]);P=torch.cat([s.P for s in states]);m0=m.mean(0)
    cov=(m-m0).T@(m-m0)/max(1,len(m)-1)+P.mean(0)
    return dict(m0=m0,P0=cov,tau_hours=float(tau_hours),fit_query_count=len(m),
                fit_query_digest=D.digest(queries),selection='tau fixed or selected in nested INNER',
                donor_m=m,donor_P=P,donor_time=np.concatenate([np.where(np.isfinite(s.release_time),s.query_time,np.nan) for s in states]),
                donor_age=np.concatenate([s.information_age_minutes for s in states]))


@torch.no_grad()
def calibration(model,prep,cfg,role):
    """Exact NB-mixture CDF (untruncated tail), from the same interval and paths as g."""
    table=D.target_table(prep.payload,prep.split,role,cfg.eval_stride);rows=[];subviews=[]
    targets=set(map(tuple,table[:,:2]));rng=np.random.default_rng(cfg.eval_seed+31)
    for st in query_rows(model,prep,cfg,np.unique(table[:,2]),role):
        predictions=predict(model,prep,st,paths=cfg.eval_paths,seed=cfg.eval_seed)
        for h,p in predictions.items():
            keep=[j for j,q in enumerate(st.query_packet) if (int(q+h),h) in targets]
            if not keep:continue
            ix=st.query_packet[keep]+h;ti=torch.as_tensor(ix,device=prep.device)
            grid=p.grid[:,:,keep];lr=interval_log_mean_rate(model.readout,grid,prep,ti)
            mu=(lr.exp()*prep.exposure_hours[ti]).cpu().numpy();y=prep.count[ti].cpu().numpy()
            r=float((-model.readout.log_nb_dispersion).clamp(math.log(1e-3),math.log(1e6)).exp())
            prob=r/(r+mu);lower=nbinom.cdf(y[None,:]-1,r,prob).mean(0);upper=nbinom.cdf(y[None,:],r,prob).mean(0)
            pit=lower+rng.uniform(size=len(ix))*(upper-lower)
            lp,un=view_log_probs(model.readout,p.z_start[:,keep],p.z_end[:,keep],prep,ti,state_grid=grid,diagnostics=True)
            for j,i in enumerate(ix):
                rows.append(dict(packet=int(i),query=int(st.query_packet[keep[j]]),horizon=h,count=float(y[j]),
                    predicted_mean=float(mu[:,j].mean()),cdf_lower=float(lower[j]),cdf_upper=float(upper[j]),pit=float(pit[j]),
                    logp=float(lp['count'][j]),scipy_logp=float(np.log(np.maximum(nbinom.pmf(y[j],r,prob[:,j]).mean(),1e-300)))))
            for v in ('band_ratio','signed_xlag','delay_iqr'):
                if v in lp:subviews.append(dict(horizon=h,view=v,logp_sum=float((lp[v]*(un[v]>0)).sum()),units=float(un[v].sum())))
    return dict(status='COMPLETE' if rows else 'NOT_ESTIMABLE',rows=rows,subviews=subviews,
                pit_histogram=np.histogram([r['pit'] for r in rows],np.linspace(0,1,11))[0],
                interpretation='PIT distribution and interval tails required; mean PIT is not calibration')


def log_elementary_symmetric(w,k):
    e=[torch.full(w.shape[:-1],-math.inf,device=w.device,dtype=w.dtype) for _ in range(k+1)]
    e[0]=torch.zeros_like(e[0])
    for i in range(w.shape[-1]):
        for j in range(min(k,i+1),0,-1):e[j]=torch.logaddexp(e[j],e[j-1]+w[...,i])
    return e[k]


def conditional_set_lp(logits,members,community):
    out=logits.new_zeros(len(logits))
    for c in torch.unique(community):
        cols=community==c;xx=logits[:,cols];yy=members[:,cols];ks=yy.sum(-1).long()
        # Exclude K=0 and K=C: these are deterministic sets, not informative units.
        for k in torch.unique(ks):
            if int(k) in (0,int(cols.sum())):continue
            mask=ks==k;out[mask]+=(xx[mask]*yy[mask]).sum(-1)-log_elementary_symmetric(xx[mask],int(k))
    return out


def identity_units(members,community):
    use=torch.zeros(len(members),dtype=torch.bool,device=members.device)
    for c in torch.unique(community):
        k=members[:,community==c].sum(-1);use|=(k>0)&(k<int((community==c).sum()))
    return use


@torch.no_grad()
def adapter_data(model,prep,cfg,role,horizon=1):
    table=D.target_table(prep.payload,prep.split,role,cfg.eval_stride,(horizon,));out=[]
    pk=prep.payload['packets'];allowed=D.input_mask(prep.split,role)
    for st in query_rows(model,prep,cfg,table[:,2],role):
        if st.m is None:raise ValueError('adapter expects a state checkpoint')
        for j,q in enumerate(st.query_packet):
            t=int(q+horizon);a,b=pk['event_lo'][t],pk['event_hi'][t]
            if a==b:continue
            hix=np.flatnonzero(allowed&(pk['end']<=st.query_time[j])&(pk['end']>st.query_time[j]-7200)&
                (pk['release']<=st.query_time[j])&(np.arange(prep.n_packets)>=st.prefix_start[j]))
            ei=np.concatenate([np.arange(pk['event_lo'][i],pk['event_hi'][i]) for i in hix]) if len(hix) else np.empty(0,int)
            count=prep.part[ei].sum(0) if len(ei) else prep.part.new_zeros(prep.part.shape[1])
            freq=(count+.5)/(len(ei)+1.);hist=torch.logit(freq.clamp(1e-4,1-1e-4))
            ids=torch.arange(a,b,device=prep.device)
            out.append(dict(packet=t,query=int(q),state=st.m[j].expand(b-a,-1),history=hist.expand(b-a,-1),
                clock=prep.clock[t].expand(b-a,-1),
                identity=prep.identity_target[ids],delay=prep.xlag[ids],delay_valid=torch.isfinite(torch.as_tensor(prep.payload['targets']['signed_xlag'][a:b],device=prep.device)),
                event_ids=ids,query_metadata=st.subset([j]).metadata()))
    if not out:return None
    result={k:torch.cat([r[k] for r in out]) for k in ('state','history','clock','identity','delay','delay_valid','event_ids')}
    result['packet']=np.concatenate([np.full(len(r['identity']),r['packet']) for r in out]);result['query_metadata']=[r['query_metadata'] for r in out]
    return result


def fit_adapter(train,validation,test,community,kind='state_history',steps=400,seed=20260906):
    """FIT-only scaling and validation, frozen producer, no OUTER checkpoint choice."""
    if any(d is None for d in (train,validation,test)):return dict(status='NOT_ESTIMABLE',reason='missing adapter split')
    features={'trait':(),'history':('history',),'state':('state',),'state_history':('state','history'),
              'delay_crossview':('state',)}[kind]
    # Every reference gets the identical known target clock, including trait.
    def get(d):return torch.cat([d[k] for k in features]+[d['clock']],-1)
    x,xv,xt=map(get,(train,validation,test));center=x.mean(0);scale=x.std(0,unbiased=False).clamp(min=.1)
    x=(x-center)/scale;xv=(xv-center)/scale;xt=(xt-center)/scale
    is_delay=kind=='delay_crossview';Y=train['delay'] if is_delay else train['identity']
    counts=[int(d['delay_valid'].sum()) if is_delay else int(identity_units(d['identity'],community).sum()) for d in (train,validation,test)]
    if min(counts)<=0:return dict(status='NOT_ESTIMABLE',reason='no informative conditional target in an adapter split',split_units=counts)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);net=nn.Linear(x.shape[1],Y.shape[1]).to(x.device)
    with torch.no_grad():
        net.weight.zero_()
        if is_delay:
            vv=train['delay_valid'];net.bias.copy_((Y*vv).sum(0)/vv.sum(0).clamp(min=1))
        else:
            pr=(Y.sum(0)+.5)/(len(Y)+1);net.bias.copy_(torch.logit(pr))
    def loss(d,xx):
        if is_delay:
            valid=d['delay_valid'];sq=(net(xx)-d['delay']).square()
            return (sq*valid).sum()/valid.sum().clamp(min=1)
        u=identity_units(d['identity'],community)
        return -conditional_set_lp(net(xx),d['identity'],community)[u].sum()/u.sum().clamp(min=1)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.01,weight_decay=1e-3)
    with torch.no_grad():best=float(loss(validation,xv));best_state={k:v.clone() for k,v in net.state_dict().items()}
    best_step=0;bad=0
    for step in range(1,steps+1):
        optimizer.zero_grad();v=loss(train,x);v.backward();optimizer.step()
        if step%20==0:
            with torch.no_grad():score=float(loss(validation,xv))
            if score<best-1e-5:best=score;best_step=step;bad=0;best_state={k:v.clone() for k,v in net.state_dict().items()}
            else:bad+=1
            if bad>=6:break
    net.load_state_dict(best_state)
    with torch.no_grad():
        score=float(loss(test,xt))
        if not np.isfinite(score):raise FloatingPointError('nonfinite frozen adapter score')
        values=((net(xt)-test['delay']).square()*test['delay_valid']).sum(-1) if is_delay else -conditional_set_lp(net(xt),test['identity'],community)
        units=test['delay_valid'].sum(-1) if is_delay else identity_units(test['identity'],community).float()
    exact=None if is_delay else float(-conditional_set_lp(net(xt),test['identity'],torch.zeros_like(community)).mean().detach())
    return dict(exact_K_set_score=exact,exact_K_note='secondary score of the conditional-identity-fitted head; between-community offsets were not separately optimized',status='COMPLETE',kind=kind,score=score,selected_step=best_step,executed_steps=step,
        metric='standardized_squared_error' if is_delay else 'conditional_set_nll',units=int(units.sum()),
        rows=dict(event_ids=test['event_ids'].cpu(),packet=test['packet'],loss=values.cpu(),units=units.cpu()),
        fitted=dict(state_dict=net.state_dict(),center=center,scale=scale,feature_names=[*features,'clock']),test_event_digest=D.digest(test['event_ids'].cpu().numpy()))


def subset_adapter(data,mask):
    if data is None or not np.any(mask):return None
    return {k:(v[mask] if isinstance(v,(np.ndarray,torch.Tensor)) else v) for k,v in data.items()}


def run_bundle(selected,out_dir,device='cpu',quick=False,tau_hours=2.):
    model,prep,cfg,record=load_selected(selected,device)
    if quick:cfg=replace(cfg,eval_stride=180,eval_paths=4,eval_chunk=8)
    role='outer' if cfg.stage in ('outer','sid') else 'inner';out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    prep.frozen_query_cache={}
    before=tensor_hash(model.state_dict());primary=evaluate(model,prep,cfg,role,collect=True)
    result=dict(status='COMPLETE',source_digest=record['source_digest'],producer_hash=before,config=vars(cfg),
                scope='engineering_smoke' if quick else 'development',score_role=role,
                primary={k:v for k,v in primary.items() if k not in ('rows','query_metadata')})
    atomic_torch(primary,out/'primary.pt')
    if cfg.arm=='state':
        if cfg.recipe_path:
            recipe=__import__('json').loads(Path(cfg.recipe_path).read_text());tau_hours=recipe.get('relax_tau_hours',tau_hours)
        reference=fit_reference(model,prep,cfg,tau_hours)
        result['rules']={r:evaluate(model,prep,cfg,role,rule=r,reference=reference) for r in ('HOLD','EVOLVE','RELAX','RESET','WRONGTIME')}
        atomic_torch(reference,out/'fit_reference.pt')
        train=adapter_data(model,prep,cfg,'fit',1)
        if train is not None:
            # The adapter has its own chronological FIT validation tail.
            unique=np.unique(train['packet']);cut=unique[max(0,int(.8*len(unique))-1)]
            tr=subset_adapter(train,train['packet']<=cut);va=subset_adapter(train,train['packet']>cut)
            community=torch.as_tensor(prep.payload['shaft_index'],device=prep.device)
            result['adapters']={}
            for h in ((1,) if quick else (1,5,30,120)):
                te=adapter_data(model,prep,cfg,role,h)
                for kind in ('trait','history','state','state_history')+ (('delay_crossview',) if cfg.crossview else ()):
                    res=fit_adapter(tr,va,te,community,kind,steps=20 if quick else 400,seed=cfg.seed)
                    atomic_torch(res,out/f'adapter_{kind}_{h}min.pt')
                    result['adapters'][f'{kind}_{h}min']={k:v for k,v in res.items() if k not in ('rows','fitted')}
        from .seizure import association
        result['seizure']=association(model,prep,cfg,quick=quick,reference=reference)
        from .spatial_transfer import clinical_spatial_association
        result['seizure']['S_B']=clinical_spatial_association(model,prep,cfg,out)
    precision=[]
    for r in range(2 if quick else 4):
        precision.append(evaluate(model,prep,cfg,role,seed=cfg.eval_seed+1009*r)['scores'])
    result['mc_precision']={}
    for h in (1,5,30,120):
        result['mc_precision'][h]={}
        for v in ('count','spatial','morphology','load'):
            values=[s[h][v] for s in precision if s[h][v] is not None]
            se=float(np.std(values,ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
            result['mc_precision'][h][v]=dict(values=values,se_of_replicate_mean=se,numerically_resolved=se is not None and se<=1e-3)
    cal=calibration(model,prep,cfg,role);atomic_torch(cal,out/'calibration.pt')
    result['calibration']={k:v for k,v in cal.items() if k!='rows'}
    result['producer_unchanged']=before==tensor_hash(model.state_dict())
    if not result['producer_unchanged']:raise RuntimeError('frozen producer changed')
    atomic_json(result,out/'summary.json');return result
