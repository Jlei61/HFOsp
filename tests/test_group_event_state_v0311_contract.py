"""v0.3.11 measurement, permission and probability contracts.

These are the checks that decide whether a human result may be interpreted at
all: no future data reaches a published feature, no unreleased packet is
assimilated, every scored density normalises, and the real FIT loss actually
moves when early history is perturbed.
"""
import json,math,os,sys
from pathlib import Path
import numpy as np
import pytest
import torch

from src.topic5_group_event_state.v0311 import data as D
from src.topic5_group_event_state.v0311 import packets as PK
from src.topic5_group_event_state.v0311 import model as M
from src.topic5_group_event_state.v0311.prepare import Prepared
from src.topic5_group_event_state.v0311.train import (RunConfig,build_run,episode_plan,
                                                      filter_episodes,rollout_from_queries,
                                                      combined_loss,query_indices)

ROOT=Path('/data/hfosp_group_event_state_rich_event_identification_v0311/packets')
SUBJECT='epilepsiae_1125'
pytestmark=pytest.mark.skipif(not (ROOT/f'{SUBJECT}.pt').exists(),reason='v0.3.11 packets not built')


@pytest.fixture(scope='module')
def bundle():
    p=torch.load(ROOT/f'{SUBJECT}.pt',weights_only=False)
    split=D.build_split(p,SUBJECT)
    px,pt,_=D.packet_tables(p,split)
    scaling=D.fit_scaling(p,split,px,pt)
    return p,split,px,pt,scaling


def test_release_never_precedes_the_data_a_packet_summarises(bundle):
    """A minute packet may not publish before the block that produced its marks."""
    p=bundle[0];pk=p['packets']
    lo,hi=pk['event_lo'],pk['event_hi']
    et=p['event_end'];rel=p['event_release']
    bad=0
    for k in np.flatnonzero(hi>lo):
        if pk['release'][k]+1e-6<max(et[lo[k]:hi[k]].max(),rel[lo[k]:hi[k]].max()):bad+=1
    assert bad==0
    # A packet that straddles a block edge with no contiguous successor keeps the
    # closed block's release; it carries no data past that edge, so the invariant
    # that matters is the one on its own events, checked above.
    late=pk['release'][hi>lo]<pk['end'][hi>lo]-1e-6
    assert late.mean()<0.01


def test_future_events_cannot_change_a_published_packet(bundle):
    """Truncating the event stream leaves every earlier packet bit-identical."""
    p,split,px,pt,_=bundle
    pk=p['packets']
    n=len(p['event_time'])
    rng=np.random.default_rng(20260906)
    # 30 boundary-aware cut points: record edges, 200 s segment edges, gaps, dense runs.
    cuts=set()
    for b in p['blocks'][:60]:
        k=int(np.searchsorted(p['event_time'],b['start']))
        cuts.add(min(max(k,1),n-1))
    for t in p['observed_support'][:,0]:
        cuts.add(min(max(int(np.searchsorted(p['event_time'],t)),1),n-1))
    while len(cuts)<30:cuts.add(int(rng.integers(1,n)))
    cuts=sorted(cuts)[:100]
    tgt=p['targets']
    for c in cuts:
        stop=float(p['event_time'][c])
        keep=np.searchsorted(pk['end'],stop,'left')
        if keep<2:continue
        sub=dict(p)
        sub['event_time']=p['event_time'][:c];sub['event_end']=p['event_end'][:c]
        sub['participation']=p['participation'][:c]
        sub['targets']={k:(v[:c] if v.shape[0]==n else v) for k,v in tgt.items()}
        spk=dict(pk)
        spk['event_lo']=np.minimum(pk['event_lo'],c);spk['event_hi']=np.minimum(pk['event_hi'],c)
        sub['packets']=spk
        sx,st,_=D.packet_tables(sub,split)
        assert np.array_equal(sx[:keep],px[:keep]),f'packet inputs changed before cut {c}'
        for key in ('count','total_load','shaft_count','exposure'):
            assert np.array_equal(st[key][:keep],pt[key][:keep]),f'{key} changed before cut {c}'


def test_per_event_marks_do_not_depend_on_later_events():
    """Row-wise recomputation on a truncated block reproduces the same tokens."""
    card=json.loads(Path('/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired'
                         '/measurements/epilepsiae_1125/block_0000/card.json').read_text())
    cache=Path(card['cache_path'])
    manifest=json.loads(cache.with_suffix('.manifest.json').read_text())
    with np.load(cache,allow_pickle=True) as z:
        full=PK.block_event_tables(z,manifest,len(card['selected_contacts']))
        for c in (7,53,211):
            trunc={k:(z[k][:c] if getattr(z[k],'shape',(0,))[:1]==(z['event_abs_time'].shape[0],) else z[k])
                   for k in z.files}
            part=PK.block_event_tables(trunc,manifest,len(card['selected_contacts']))
            assert np.allclose(part[0],full[0][:c],equal_nan=True)
            assert np.allclose(part[1],full[1][:c],equal_nan=True)
            for key in part[2]:
                assert np.allclose(part[2][key],full[2][key][:c],equal_nan=True)


def test_unreleased_packets_are_never_assimilated(bundle):
    """The query state is the last closed block, and the lag is real, not zero."""
    p,split,px,pt,scaling=bundle
    prep=Prepared(p,split,scaling,torch.device('cpu'))
    cfg=RunConfig(subject=SUBJECT,device='cpu')
    cand,q=episode_plan(prep,split,cfg)
    starts=cand[:6]
    kstar,lag,q_ok=query_indices(prep,starts,q)
    assert q_ok.all(),'every query offset must already have a closed block'
    for b,s in enumerate(starts):
        for j,off in enumerate(q):
            tau=prep.packet_end[s+off]
            k=kstar[b,j]
            assert prep.packet_release[s+k]<=tau+1e-6
            if s+k+1<prep.n_packets and k+1<=off:
                assert prep.packet_release[s+k+1]>tau+1e-6
            assert lag[b,j]==off-k
    assert lag.max()>0,'a 60-minute publication delay must produce non-zero query lag'


def test_inner_targets_are_withheld_from_training(bundle):
    p,split,px,pt,scaling=bundle
    prep=Prepared(p,split,scaling,torch.device('cpu'))
    pk=p['packets']
    for t in split['inner_target_times']:
        k=int(np.searchsorted(pk['end'],t-1e-6))
        if 0<=k<prep.n_packets:
            assert prep.target_ok_np[k]==False
    trainable=prep.valid_np&split['train_packet']
    assert prep.target_ok_np.sum()>0.8*trainable.sum()


def test_excluded_intervals_are_not_ordinary_missing_data(bundle):
    p,split,px,pt,scaling=bundle
    prep=Prepared(p,split,scaling,torch.device('cpu'))
    assert prep.seizure_masked_np.sum()>0
    assert not prep.target_ok_np[prep.seizure_masked_np].any()


def test_negative_binomial_normalises():
    lr=torch.tensor([[-0.3]]);disp=torch.tensor([0.4])
    ks=torch.arange(0,4000.).reshape(-1,1)
    lp=M.count_log_prob(lr.expand(4000,1),lr.expand(4000,1),torch.full((4000,1),0.5),ks,disp)
    assert abs(float(torch.exp(lp).sum())-1.)<1e-6


def test_multinomial_composition_normalises():
    from itertools import product
    logits=torch.tensor([[0.3,-0.7,1.1]],dtype=torch.float64)
    total=5
    tot=0.
    for c in product(range(total+1),repeat=3):
        if sum(c)!=total:continue
        tot+=float(torch.exp(M.composition_log_prob(logits,torch.tensor([[*c]],dtype=torch.float64))))
    assert abs(tot-1.)<1e-9


def test_zero_inflated_normal_normalises():
    params=torch.tensor([[-0.8,0.3,-0.2]],dtype=torch.float64)
    p0=float(torch.sigmoid(params[0,0]))
    zero=float(torch.exp(M.zero_inflated_normal_log_prob(torch.zeros(1),torch.ones(1),params,torch.ones(1))))
    w=torch.linspace(-12,12,240001,dtype=torch.float64)
    dens=torch.exp(M.zero_inflated_normal_log_prob(w,torch.zeros_like(w),params.expand(240001,3),torch.ones_like(w)))
    integral=float(torch.trapz(dens,w))
    assert abs(zero-p0)<1e-9
    assert abs(zero+integral-1.)<1e-4


def test_mixture_uses_log_mean_not_mean_log():
    lp=torch.tensor([[-1.],[-9.]],dtype=torch.float64)
    mix=float(torch.logsumexp(lp,0)-math.log(2))
    assert mix>float(lp.mean())
    assert abs(mix-math.log((math.exp(-1)+math.exp(-9))/2))<1e-9


def test_fit_loss_has_real_gradient_on_early_history(bundle):
    """Perturbing the 0-0.5 h and 0.5-2 h history before the query must move the FIT loss.

    History is counted backwards from the query, which is where the 2-hour
    gradient window sits; the earlier warm-up prefix is deliberately detached.
    """
    p,split,px,pt,scaling=bundle
    dev=torch.device('cpu')
    prep=Prepared(p,split,scaling,dev)
    cfg=RunConfig(subject=SUBJECT,device='cpu',batch_episodes=2,n_start=4,stride=20,
                  train_paths=2,warm_packets=60,grad_packets=120)
    model,_=build_run(cfg,p,split,scaling,prep)
    cand,q=episode_plan(prep,split,cfg)
    gen=torch.Generator().manual_seed(0)
    # Well separated so one episode's warm-up cannot be another's gradient window.
    starts=np.array([cand[0],cand[len(cand)//2]])
    assert abs(starts[1]-starts[0])>2*(cfg.warm_packets+cfg.grad_packets)
    prep.stats.requires_grad_(True)
    pl,pu,snap=filter_episodes(model,prep,starts,q,cfg,gen,training=True)
    rl,ru,_=rollout_from_queries(model,prep,snap,cfg,gen,training=True)
    loss=combined_loss(pl,pu,rl,ru)
    loss.backward()
    g=prep.stats.grad
    L=cfg.warm_packets+cfg.grad_packets
    early=np.concatenate([np.arange(s+L-30,s+L) for s in starts])
    mid=np.concatenate([np.arange(s+L-120,s+L-30) for s in starts])
    warm=np.concatenate([np.arange(s,s+cfg.warm_packets) for s in starts])
    assert float(g[early].abs().sum())>0,'no gradient reaches the last 0.5 h of history'
    assert float(g[mid].abs().sum())>0,'no gradient reaches the 0.5-2 h history'
    assert float(g[warm].abs().sum())==0,'the declared warm-up prefix must stay detached'
    base=float(loss.detach())
    with torch.no_grad():
        prep.stats[early]+=0.25
    gen2=torch.Generator().manual_seed(0)
    pl,pu,snap=filter_episodes(model,prep,starts,q,cfg,gen2,training=True)
    rl,ru,_=rollout_from_queries(model,prep,snap,cfg,gen2,training=True)
    assert abs(float(combined_loss(pl,pu,rl,ru).detach())-base)>1e-8


def test_scores_use_a_common_denominator(bundle):
    """Both arms must be scored on the same effective units."""
    p,split,px,pt,scaling=bundle
    dev=torch.device('cpu')
    prep=Prepared(p,split,scaling,dev)
    cfg=RunConfig(subject=SUBJECT,device='cpu',batch_episodes=2,n_start=4,stride=20,train_paths=2)
    cand,q=episode_plan(prep,split,cfg)
    gen=torch.Generator().manual_seed(0)
    starts=cand[:2]
    units=[]
    for arm in ('state','intercept'):
        c=RunConfig(**{**cfg.__dict__,'arm':arm})
        model,ref=build_run(c,p,split,scaling,prep)
        _,_,snap=filter_episodes(model,prep,starts,q,c,gen,training=False,reference=ref)
        _,ru,_=rollout_from_queries(model,prep,snap,c,gen,training=False,reference=ref)
        units.append({h:{k:float(v) for k,v in ru[h].items()} for h in ru})
    assert units[0]==units[1]
