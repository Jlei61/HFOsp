import numpy as np
from scripts.analyze_e1146_relative_enrichment import matched_excess, balanced_enrichment


def test_minority_can_be_enriched_and_majority_depleted():
    # TB=37% over TB baseline=32%; TA=57% under TA baseline=68%.
    np.testing.assert_allclose(matched_excess([.63,.57],.68,[False,True]),[.05,-.11])


def test_equal_class_weighting_cancels_common_patient_baseline():
    p=np.array([.8,.75,.7,.6,.65]);a=np.array([True,True,True,False,False])
    for q in (.4,.68,.85):
        assert np.isclose(balanced_enrichment(p,q,a),(p[a].mean()-p[~a].mean())/2)


def test_leave_window_out_reference_keeps_direction():
    n=100;total=1000;total_a=680
    for count_a in (40,68,90):
        q_other=(total_a-count_a)/(total-n)
        for label in (True,False):
            assert np.sign(matched_excess(count_a/n,total_a/total,label))==np.sign(matched_excess(count_a/n,q_other,label))
