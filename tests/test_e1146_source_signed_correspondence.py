import numpy as np
from scripts.analyze_e1146_source_signed_correspondence import common_scores, classify_source


def test_energy_at_source_has_positive_not_absolute_alignment():
    points=np.array([[0,0],[1,0],[2,0],[0,1],[1,1],[2,1]],float)
    ranks=np.array([0,1,2,3,4,5.],float)
    r,_,_=common_scores(-ranks,ranks,5-ranks,points,np.ones(6),.4)
    np.testing.assert_allclose(r,[1,-1],atol=1e-12)
    assert classify_source(*r)==('TA','TA')
    r,_,_=common_scores(ranks,ranks,5-ranks,points,np.ones(6),.4)
    assert classify_source(*r)==('TB','TB')


def test_uniform_power_and_positive_scaling_do_not_change_morphology():
    points=np.array([[0,0],[1,0],[2,0],[0,1],[1,1],[2,1]],float)
    ranks=np.array([0,1,2,3,4,5.]);energy=np.array([3,2,4,1,0,2.])
    a,*_=common_scores(energy,ranks,5-ranks,points,np.ones(6),.4)
    b,*_=common_scores(7*energy+40,ranks,5-ranks,points,np.ones(6),.4)
    np.testing.assert_allclose(a,b,atol=1e-12)


def test_nonpositive_and_close_scores_are_not_forced_classes():
    assert classify_source(-.1,-.3)==('neither','')
    assert classify_source(.02,.04)==('ambiguous','TB')
    assert classify_source(np.nan,.5)==('unavailable','')
