"""The held-out scoring pass must observe the forward segment, not only propagate.

Masking the forward segment out of the model's inputs would silently turn every
state arm into a constant-state model at scoring time.
"""
from pathlib import Path
import numpy as np
import pytest
import torch

from src.topic5_group_event_state.v0311 import data as D
from src.topic5_group_event_state.v0311.prepare import Prepared
from src.topic5_group_event_state.v0311.train import (RunConfig,build_run,filter_episodes,
                                                      HORIZON_PACKETS)

ROOT=Path('/data/hfosp_group_event_state_rich_event_identification_v0311/packets')
SUBJECT='epilepsiae_1125'
pytestmark=pytest.mark.skipif(not (ROOT/f'{SUBJECT}.pt').exists(),reason='v0.3.11 packets not built')


def _setup():
    p=torch.load(ROOT/f'{SUBJECT}.pt',weights_only=False)
    split=D.build_split(p,SUBJECT)
    px,pt,_=D.packet_tables(p,split)
    scaling=D.fit_scaling(p,split,px,pt)
    prep=Prepared(p,split,scaling,torch.device('cpu'))
    cfg=RunConfig(subject=SUBJECT,device='cpu',batch_episodes=2,n_start=1,stride=1,
                  q_first=179,train_paths=2,eval_paths=2)
    model,_=build_run(cfg,p,split,scaling,prep)
    return p,split,prep,cfg,model


def test_forward_segment_states_depend_on_observed_data():
    p,split,prep,cfg,model=_setup()
    L=cfg.warm_packets+cfg.grad_packets
    pe=prep.packet_end
    idx=np.searchsorted(pe,np.asarray(split['forward_starts'],float)-1e-6)
    starts=np.array([int(i)-L+1 for i in idx if i-L+1>=0 and i+max(HORIZON_PACKETS)<prep.n_packets][:6])
    assert len(starts)>=4
    q=np.array([L-1])
    gen=torch.Generator().manual_seed(0)
    _,_,snap=filter_episodes(model,prep,starts,q,cfg,gen,training=False)
    m=snap['m'][0]
    spread=float(m.std(0).max())
    assert spread>1e-6,'forward-segment states are identical, so nothing was assimilated'
    gen2=torch.Generator().manual_seed(0)
    _,_,tr=filter_episodes(model,prep,starts,q,cfg,gen2,training=True)
    assert float((tr['m'][0]-m).abs().max())>1e-6, \
        'training and scoring must differ on the forward segment: training may not read it'


def test_training_region_states_are_identical_under_both_flags():
    """Inside the fitting region the two masks agree, so nothing else changed."""
    p,split,prep,cfg,model=_setup()
    L=cfg.warm_packets+cfg.grad_packets
    pe=prep.packet_end
    idx=np.searchsorted(pe,np.asarray(split['inner_starts'],float)-1e-6)
    starts=np.array([int(i)-L+1 for i in idx if i-L+1>=0][:4])
    q=np.array([L-1])
    a=filter_episodes(model,prep,starts,q,cfg,torch.Generator().manual_seed(0),training=False)[2]['m'][0]
    b=filter_episodes(model,prep,starts,q,cfg,torch.Generator().manual_seed(0),training=True)[2]['m'][0]
    assert float((a-b).abs().max())<1e-8
