import copy
from src.topic4_xy_replicated_anchors import anchor_eligible,proposal_pool,incumbent

PLAN={'search':{'fit_seeds':[1,2],'race_seeds':[3,4,5,6,7,8],'minimum_pool_events':64}}


def row(name,seeds,score,n=80):
    return {'candidate_id':name,'explorable':True,'n_events':n,'exploration_score':score,
            'candidate':{'node_field':{'field_sha256':name}},
            'units':[{'seed':s,'runaway':False,'geometry':{'minimum_clearance_mm':1,'full_disks_disjoint':True}} for s in seeds]}


def test_lucky_short_run_cannot_be_local_anchor_but_hash_stays_seen():
    low=row('lucky',[1,2],-100);full=row('replicated',list(range(1,9)),.2)
    a=proposal_pool([low,full],PLAN)
    assert [r['candidate_id'] for r in a if r['explorable']]==['replicated']
    assert {r['candidate']['node_field']['field_sha256'] for r in a}=={'lucky','replicated'}
    assert incumbent([low,full],PLAN)['candidate_id']=='replicated'
    assert low['explorable'] # no mutation of saved scientific rows


def test_incomplete_events_wrong_or_duplicate_seeds_and_clipped_cores_rejected():
    valid=row('valid',list(range(1,9)),.2)
    cases=[row('low_n',list(range(1,9)),.01,n=63),row('other',[1,2,3,4,5,6,7,9],.01),
           row('dup',[1,2,3,4,5,6,7,7],.01)]
    clipped=copy.deepcopy(valid);clipped['units'][0]['geometry']['minimum_clearance_mm']=-.01;cases.append(clipped)
    runaway=copy.deepcopy(valid);runaway['units'][0]['runaway']=True;cases.append(runaway)
    overlap=copy.deepcopy(valid);overlap['units'][0]['geometry']['full_disks_disjoint']=False;cases.append(overlap)
    assert anchor_eligible(valid,PLAN)
    assert not any(anchor_eligible(x,PLAN) for x in cases)
    assert incumbent(cases,PLAN) is None
