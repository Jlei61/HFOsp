import copy,json
from pathlib import Path
import numpy as np
from scripts import run_topic4_observable_loss_physical_pilot as run
from scripts.analyze_topic4_observable_loss_physical_pilot import disconnected_fraction


def anchors():return json.loads((run.prior.OUT/'design.json').read_text())['anchors']


def test_proposals_are_persisted_bounded_and_keep_geometry(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'OUT',tmp_path);a=anchors();d={'anchors':a}
    cc=run.proposals('C',d,a)
    assert len(cc)==6 and len({run.condition_key(c) for c in cc})==6
    assert cc==run.proposals('C',d,list(reversed(a)))
    for c in cc:
        i=run.anchor_index(c,a);v=run.prior.vector(c)
        assert c['node_field']==a[i]['node_field']
        assert np.all(v>=run.prior.LOW) and np.all(v<=run.prior.HIGH)
        assert np.all(np.abs(v-run.prior.vector(a[i]))<=run.prior.STEP+1e-12)


def test_parent_selection_uses_new_loss_only():
    a=anchors();pool=[]
    for i,c in enumerate(a):
        for tag,score in [('x',.5),('y',.2)]:
            cc=copy.deepcopy(c);cc['candidate_id']=str(i)+tag
            pool.append(dict(candidate=cc,candidate_id=cc['candidate_id'],loss=score,old_loss=1-score,ranking_eligible=True))
    assert [c['candidate_id'] for c in run.parents(pool,a)]==['0y','1y','2y']


def test_same_pool_nominees_are_bounded_and_ignore_unrankable():
    a=anchors();pool=[]
    for i,c in enumerate(a):
        pool.append(dict(candidate=c,candidate_id=c['candidate_id'],loss=float(i),old_loss=float(2-i),ranking_eligible=True))
    invalid=copy.deepcopy(pool[0]);invalid['candidate_id']='invalid';invalid['candidate']['candidate_id']='invalid';invalid['ranking_eligible']=False;invalid['loss']=-100
    # Preserve original candidate object after intentionally making an invalid copy.
    result=run.nominate(pool+[invalid],pool[0]['candidate_id'])
    assert len(result['nominees'])<=5
    assert all(r['candidate_id']!='invalid' for r in result['nominees'])
    assert not result['native_diagnostics_used'] and not result['confirmation_data_used']


def test_fragmentation_diagnostic_distinguishes_disconnected_mass():
    den=np.ones((20,20))*80;one=np.zeros((5,20,20));one[:,2:5,2:5]=10
    two=one.copy();two[:,14:17,14:17]=10
    assert disconnected_fraction(one,den,.2)==0
    assert np.isclose(disconnected_fraction(two,den,.2),.5)
    assert disconnected_fraction(np.zeros_like(one),den,.2) is None
