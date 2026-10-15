"""Bounded DE/rand/1/bin proposal mechanics (design §5 stage 2; checklist C13)."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
from scripts import run_topic4_core_connectivity_stages as run


BOUNDS={'EE_core_to_out_scale':[.6,1.6],'radius_A_mm':[1.4,3.2],'depth_A_scale':[.5,2.]}


def parents():
    rng=np.random.default_rng(0);keys=list(BOUNDS)
    return [dict(id=f'p{i}',vector={k:float(rng.uniform(*BOUNDS[k])) for k in keys},loss=float(rng.random())) for i in range(8)]


def test_rand1bin_offspring_uses_three_distinct_others_and_reflects_bounds():
    pop=parents();rng=np.random.default_rng(934711)
    out=run.de_offspring(pop,[0,1,2,3],BOUNDS,rng,F=.6,CR=.7)
    assert [o['target_index'] for o in out]==[0,1,2,3]
    for o in out:
        r=o['donor_indices'];assert len(set(r))==3 and o['target_index'] not in r
        base=pop[r[0]]['vector'];d1=pop[r[1]]['vector'];d2=pop[r[2]]['vector']
        crossed=0
        for k in BOUNDS:
            lo,hi=BOUNDS[k];assert lo<=o['vector'][k]<=hi
            donor=base[k]+.6*(d1[k]-d2[k]);donor=run.reflect(donor,lo,hi)
            if o['vector'][k]==donor and donor!=pop[o['target_index']]['vector'][k]:crossed+=1
            assert o['vector'][k] in (donor,pop[o['target_index']]['vector'][k])
        assert o['from_donor_dims']>=1 and o['F']==.6 and o['CR']==.7
    again=run.de_offspring(pop,[0,1,2,3],BOUNDS,np.random.default_rng(934711),F=.6,CR=.7)
    assert [a['vector'] for a in again]==[o['vector'] for o in out]


def test_reflection_keeps_values_inside_and_is_identity_inside():
    assert run.reflect(1.,0.,2.)==1. and run.reflect(2.3,0.,2.)==pytest.approx(1.7) and run.reflect(-.4,0.,2.)==pytest.approx(.4) and 0.<=run.reflect(7.9,0.,2.)<=2.


def test_selection_never_compares_two_unscorable_and_prefers_scorable_child():
    parent=dict(id='p',loss=None);child=dict(id='c',loss=None)
    assert run.de_select(parent,child)['id']=='p' and run.de_select(parent,child)['reason']=='both_unscorable_kept_parent'
    assert run.de_select(dict(id='p',loss=None),dict(id='c',loss=.3))['id']=='c'
    assert run.de_select(dict(id='p',loss=.2),dict(id='c',loss=None))['id']=='p'
    assert run.de_select(dict(id='p',loss=.2),dict(id='c',loss=.1))['id']=='c' and run.de_select(dict(id='p',loss=.1),dict(id='c',loss=.2))['id']=='p'
