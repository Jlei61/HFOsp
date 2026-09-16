import copy
import numpy as np
import pytest
from src.topic4_xy_relative_position_round import (relative_position_ranking,
                                                   primary_and_unconstrained_nominees)


def example():
    return {'candidates':[
        {'candidate_id':'off_axis','domain':'whole_sheet','J_direction':1.,'core_alignment_penalty':1.,'selection_eligible':True},
        {'candidate_id':'aligned','domain':'whole_sheet','J_direction':1.04,'core_alignment_penalty':0.,'selection_eligible':True},
        {'candidate_id':'invalid','domain':'interior','J_direction':0.1,'core_alignment_penalty':0.,'selection_eligible':False},
        {'candidate_id':'few_events','domain':'interior','J_direction':None,'core_alignment_penalty':0.,'selection_eligible':False},
        {'candidate_id':'inside','domain':'interior','J_direction':1.2,'core_alignment_penalty':0.2,'selection_eligible':True},
    ]}


def test_prior_can_change_primary_without_overwriting_measured_fit():
    original=example();saved=copy.deepcopy(original)
    result=relative_position_ranking(original)
    assert original==saved
    assert result['ranking'][:2]==['aligned','off_axis']
    assert result['ranking_without_position_prior'][:2]==['off_axis','aligned']
    assert result['candidates'][0]['J_direction']==1.
    assert result['candidates'][0]['J_round1']==pytest.approx(1.1)


def test_unqualified_rows_do_not_enter_any_nomination():
    result=relative_position_ranking(example())
    assert primary_and_unconstrained_nominees(result)==['aligned','off_axis','inside']


def test_zero_weight_is_exact_no_prior_comparator():
    result=relative_position_ranking(example(),weight=0)
    assert result['ranking']==result['ranking_without_position_prior']


@pytest.mark.parametrize('weight',[-1,np.nan,np.inf])
def test_bad_weight_fails_closed(weight):
    with pytest.raises(ValueError):relative_position_ranking(example(),weight=weight)


def test_score_or_penalty_corruption_fails_closed():
    result=example();result['candidates'][0]['core_alignment_penalty']=1.3
    with pytest.raises(ValueError):relative_position_ranking(result)
    result=example();result['candidates'][0]['J_direction']=np.nan
    with pytest.raises(ValueError):relative_position_ranking(result)
