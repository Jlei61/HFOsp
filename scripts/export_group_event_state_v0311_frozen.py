#!/usr/bin/env python
"""Frozen-checkpoint export: HOLD/EVOLVE/RELAX and fine contact identity.

The producer is never retrained here. Identity adapters see only producer
FIT/INNER support; the outer segment is scored once, at the end.
"""
import argparse,json,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from torch import nn

from src.topic5_group_event_state.v0311 import data as D
from src.topic5_group_event_state.v0311.prepare import Prepared,HISTORY_TAU_HOURS
from src.topic5_group_event_state.v0311.train import RunConfig,build_run,HORIZON_PACKETS
from src.topic5_group_event_state.v0311 import frozen as FZ
from src.topic5_group_event_state.v0311.objective import VIEWS

ROOT=Path('/data/hfosp_group_event_state_rich_event_identification_v0311')


def event_rows(prep,packets):
    lo=prep.event_lo[packets];hi=prep.event_hi[packets]
    counts=hi-lo;total=int(counts.sum())
    if total==0:return None,None
    offs=torch.cumsum(counts,0)-counts
    pos=torch.arange(total,device=prep.device)
    row=torch.searchsorted(offs+counts,pos,right=True)
    return lo[row]+(pos-offs[row]),row


def history_features(prep,packets,n_columns=None):
    """Causal exponential history of the packet stream at each query.

    Only the coarse count / load / composition columns are kept, so the reference
    carries a capacity comparable to the 48-dim frozen state instead of losing to
    its own dimensionality.
    """
    x=prep.stats if n_columns is None else prep.stats[:,:n_columns]
    n=prep.n_packets
    ema=torch.zeros(len(HISTORY_TAU_HOURS),x.shape[-1],device=prep.device,dtype=x.dtype)
    decay=torch.exp(-torch.as_tensor(1/60./np.asarray(HISTORY_TAU_HOURS),device=prep.device,dtype=x.dtype))
    # Normalised weighted average so every timescale enters on the same scale.
    want=set(int(p) for p in packets);out={}
    for k in range(n):
        if prep.seizure_masked_np[k]:ema=ema*0.;continue
        ema=ema*decay.reshape(-1,1)+float(prep.valid[k])*(1-decay).reshape(-1,1)*x[k]
        if k in want:out[k]=ema.reshape(-1).clone()
    return torch.stack([out[int(p)] for p in packets])


def identity_transfer(model,prep,cfg,fit_packets,inner_packets,outer_packets,paths=32,updates=600,lr=3e-3):
    """Fine identity from the frozen state, against trait / history / constant baselines."""
    dev=prep.device;C=prep.part.shape[-1]
    comm=torch.as_tensor(prep.payload['shaft_index'],device=dev)
    def features(packets):
        m,P,kept=FZ.states_at_queries(model,prep,cfg,packets)
        if m is None:return None
        hist=history_features(prep,kept,n_columns=2+prep.n_shaft)
        return dict(m=m,P=P,kept=kept,hist=hist)
    tr=features(fit_packets);iv=features(inner_packets);ou=features(outer_packets)
    if tr is None or ou is None or iv is None:
        return dict(status='NOT_ESTIMABLE',reason='no eligible query prefix')
    # Standardise the history block on fitting support; 55 raw kernel features against
    # a few dozen fitting queries otherwise overfits the reference into uselessness.
    hm=tr['hist'].mean(0,keepdim=True);hs=tr['hist'].std(0,keepdim=True).clamp(min=0.05)
    for d in (tr,iv,ou):d['hist']=((d['hist']-hm)/hs).clamp(-5,5)
    arms={'fixed_trait':(0,True),'state_only':(2*24,False),
          'trait_plus_recent_history':(tr['hist'].shape[-1],True),'trait_plus_state':(2*24,True)}
    results={}
    for name,(dim,use_trait) in arms.items():
        torch.manual_seed(int(cfg.seed))
        ad=FZ.IdentityAdapter(dim,C,use_trait=use_trait).to(dev)
        decay=1e-2 if name=='trait_plus_recent_history' else 1e-4
        opt=torch.optim.AdamW(ad.parameters(),lr=lr,weight_decay=decay)
        def feat(pack,d,name=name):
            if name=='fixed_trait':return torch.zeros(len(d['kept']),0,device=dev)
            if name=='trait_plus_recent_history':return d['hist']
            return torch.cat((d['m'],torch.diagonal(d['P'],dim1=-2,dim2=-1)),dim=-1)
        best=(float('inf'),None,0)
        for u in range(1,updates+1):
            opt.zero_grad(set_to_none=True)
            loss=identity_loss(ad,feat(fit_packets,tr),tr,prep,comm)
            if loss is None:break
            loss.backward();opt.step()
            if u%25==0:
                with torch.no_grad():
                    v=identity_loss(ad,feat(inner_packets,iv),iv,prep,comm)
                if v is not None and float(v)<best[0]:
                    best=(float(v),{k:t.detach().clone() for k,t in ad.state_dict().items()},u)
        if best[1] is not None:ad.load_state_dict(best[1])
        with torch.no_grad():
            out_all=identity_score(ad,feat(outer_packets,ou),ou,prep,comm)
        results[name]=dict(outer=out_all,selected_update=best[2],inner_identity_nats=best[0])
    return dict(status='COMPLETE',arms=results,n_fit_queries=int(len(tr['kept'])),
                n_inner_queries=int(len(iv['kept'])),n_outer_queries=int(len(ou['kept'])),
                contract=('adapters fitted on producer FIT/INNER support only; the outer segment is '
                          'scored once with the frozen producer state; each adapter keeps the update '
                          'with the best held-in identity loss. History features are standardised on '
                          'fitting support and carry a heavier weight decay, so the reference is not '
                          'handicapped by its own dimensionality.'))


def _targets(prep,d,horizon):
    base=torch.as_tensor(d['kept'],device=prep.device)+horizon
    keep=base<prep.n_packets
    idx,row=event_rows(prep,base[keep])
    return base,keep,idx,row


def identity_loss(ad,x,d,prep,comm,within=False,horizon=1):
    base,keep,idx,row=_targets(prep,d,horizon)
    if idx is None:return None
    logits=ad(x[keep])[row]
    members=prep.identity_target[idx]
    lp=FZ.set_log_prob_batch(logits,members)
    n=members.sum(-1)
    return -(lp[n>0].mean())


@torch.no_grad()
def identity_score(ad,x,d,prep,comm):
    out={}
    for h in HORIZON_PACKETS:
        base,keep,idx,row=_targets(prep,d,h)
        if idx is None:continue
        allow=(prep.valid[base[keep]]*(1.-prep.seizure_masked[base[keep]]))[row]>0
        logits=ad(x[keep])[row]
        members=prep.identity_target[idx]
        n=members.sum(-1)
        m=(n>0)&allow
        if not bool(m.any()):continue
        out[h]=dict(exact_set_nats=float(-FZ.set_log_prob_batch(logits[m],members[m]).mean()),
                    within_community_nats=float(-FZ.within_community_set_log_prob(logits[m],members[m],comm).mean()),
                    n_events=int(m.sum()))
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--card',required=True);ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--paths',type=int,default=32)
    a=ap.parse_args()
    card=json.loads(Path(a.card).read_text())
    cfg=RunConfig(**{**card['config'],'device':a.device})
    if cfg.arm!='state':print(json.dumps(dict(status='SKIPPED',reason='reference arm')));return
    payload=torch.load(f'{cfg.packets_root}/{cfg.subject}.pt',weights_only=False)
    split=D.build_split(payload,cfg.subject,cfg.seed) if cfg.split=='S-E' else D.build_split_id(payload,cfg.subject,cfg.seed)
    px,pt,_=D.packet_tables(payload,split);scaling=D.fit_scaling(payload,split,px,pt)
    dev=torch.device(a.device);prep=Prepared(payload,split,scaling,dev)
    prep.payload=payload
    from src.topic5_group_event_state.v0311.train import apply_ablations
    apply_ablations(prep,cfg)
    model,_=build_run(cfg,payload,split,scaling,prep)
    ck=torch.load(Path(a.card).with_suffix('').with_suffix('.ckpt.pt'),weights_only=False,map_location=dev)
    model.load_state_dict(ck['state_dict']);model.eval()
    pe=prep.packet_end
    L=cfg.warm_packets+cfg.grad_packets
    def to_packets(times):
        idx=np.searchsorted(pe,np.asarray(times,float)-1e-6)
        return np.array([int(i) for i in idx if i-L+1>=0 and i+max(HORIZON_PACKETS)<prep.n_packets
                         and not prep.seizure_masked_np[i]],np.int64)
    inner=to_packets(split['inner_starts']);outer=to_packets(split['forward_starts'])
    fit_grid=np.arange(split['support_start']+L*60,split['fit_end'],600.)
    fitp=to_packets(fit_grid)
    t0=time.time()
    relax=FZ.fit_relax(model,prep,cfg,fitp[:80],inner)
    rules,kept=FZ.score_rules(model,prep,cfg,outer,relax,paths=a.paths)
    ident=identity_transfer(model,prep,cfg,fitp,inner,outer,paths=a.paths)
    out=dict(card=str(a.card),subject=cfg.subject,split=cfg.split,inputs=cfg.inputs,family=cfg.family,
             seconds=round(time.time()-t0,1),
             relax=dict(tau_hours=relax['tau_hours'],grid=relax['grid'],
                        inner_selected_score=relax['inner_selected_score']) if relax else None,
             rules=rules,n_outer_queries=int(len(kept)),identity=ident,
             note=('HOLD keeps the current state distribution, EVOLVE runs the learned f, RELAX returns '
                   'to the FIT stationary distribution; all three share observation head, clock, exposure '
                   'and mask. This compares predictive value, not a physiological intervention.'))
    p=ROOT/'frozen'/(Path(a.card).name.replace('.card.json','.frozen.json'))
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,indent=1,default=str))
    print(json.dumps({k:out[k] for k in ('subject','split','inputs','family','seconds','n_outer_queries')},default=str))

if __name__=='__main__':main()
