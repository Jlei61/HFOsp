import copy
import json
import numpy as np
from scripts import run_topic4_contact_native_integrated_pilot as run


def anchors():
    return json.loads((run.prior.OUT/'design.json').read_text())['anchors']


def unit(loss, r, n=30):
    return dict(observation=dict(loss=loss,N=n),regularizer=dict(mean_unsupported_fraction=r))


def test_negative_score_survives_and_networks_have_equal_weight():
    c=anchors()[0]
    row=run.aggregate(c,{'t1':unit(-.5,.2,16),'t2':unit(.1,.6,100)},.2)
    assert np.isclose(row['observation_loss'],-.2)
    assert np.isclose(row['loss'],-.12)
    assert row['ranking_eligible'] and not row['model_labels_used_for_ranking']


def test_insufficient_one_network_cannot_be_hidden_by_other_network():
    r=run.aggregate(anchors()[0],{'t1':unit(None,.2,15),'t2':unit(.1,.6,100)},.2)
    assert r['observation_loss'] is None and r['loss'] is None
    assert not r['ranking_eligible']
    assert not run.aggregate(anchors()[0],{'t1':unit(.1,.2)},.2)['ranking_eligible']


def test_lambda_is_rescaled_to_new_observation_units():
    rows=[dict(ranking_eligible=True,observation_loss=a,R_unsupported=b) for a,b in zip([.1,.3,.5,.7],[.1,.2,.3,.4])]
    one=run.freeze_lambda(rows)
    doubled=run.freeze_lambda([{**r,'observation_loss':2*r['observation_loss']} for r in rows])
    assert np.isclose(doubled['primary'],2*one['primary'])
    assert not one['new_native_wave_outputs_used_to_choose_lambda']


def test_updated_parent_depends_on_combined_loss_and_has_no_mode_gate():
    aa=anchors();rows=[]
    for i,c in enumerate(aa):
        for label,score in [('pure',.5),('integrated',.2)]:
            cc=copy.deepcopy(c);cc['candidate_id']=str(i)+label
            rows.append(dict(candidate=cc,candidate_id=cc['candidate_id'],loss=score,ranking_eligible=True))
    assert [c['candidate_id'] for c in run.parents(rows,aa)]==[str(i)+'integrated' for i in range(3)]


def test_proposal_restart_preserves_draws_and_core_geometry(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'OUT',tmp_path)
    aa=anchors();cc=run.proposals({'anchors':aa},aa)
    assert len(cc)==6 and len({run.condition_key(c) for c in cc})==6
    assert run.proposals({'anchors':aa},list(reversed(aa)))==cc
    for c in cc:
        i=run.execution.anchor_index(c,aa)
        assert c['node_field']==aa[i]['node_field']
        assert np.all(np.abs(run.prior.vector(c)-run.prior.vector(aa[i]))<=run.prior.STEP+1e-10)
        assert np.all(run.prior.vector(c)>=run.prior.LOW)
        assert np.all(run.prior.vector(c)<=run.prior.HIGH)


def test_nominees_compare_same_pool_and_budget_is_bounded():
    aa=anchors();rows=[]
    for i,c in enumerate(aa):
        rows.append(dict(candidate=c,candidate_id=c['candidate_id'],loss=float(i),observation_loss=float(2-i),ranking_eligible=True))
    result=run.nominate(rows,aa[0]['candidate_id'])
    assert len(result['nominees'])==3 and result['max_conditions']==5
    assert all(r['ranking_eligible'] for r in result['nominees'])
    assert 'starting_reference' in result['roles'][aa[0]['candidate_id']]


def test_missing_native_support_does_not_become_zero_penalty():
    row=run.aggregate(anchors()[0],{'t1':unit(.1,None),'t2':unit(.1,.6)},.2)
    assert row['observation_loss']==.1
    assert row['loss'] is None and not row['ranking_eligible']
