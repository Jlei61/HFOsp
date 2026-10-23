import numpy as np
from src.interictal_two_target_holdout import rank_error_suite,direction_error_suite,block_direction_null,holm_adjust


def test_pair_order_and_rmse_hand_calculation():
    truth=np.array([[0,1/3,2/3,1.]])
    pred=np.array([[0,2/3,1/3,1.]])
    predictions=np.tile(pred[None,:,None,:],(2,1,2,1))
    result=rank_error_suite(truth,np.ones_like(truth,bool),predictions)
    np.testing.assert_allclose(result['rank_mae'],1/6)
    np.testing.assert_allclose(result['rank_rmse'],np.sqrt(1/18))
    np.testing.assert_allclose(result['pair_order_error'],1/6)


def test_template_ties_half_penalty_and_unobserved_contacts_ignored():
    truth=np.array([[0,.5,1,0.]])
    mask=np.array([[1,1,1,0]],bool)
    pred=np.array([[0,0,1,900.]])
    predictions=np.tile(pred[None,:,None,:],(2,1,2,1))
    result=rank_error_suite(truth,mask,predictions)
    np.testing.assert_allclose(result['pair_order_error'],1/6)


def test_direction_does_not_choose_its_own_axis():
    directions=np.array([[1.,0,0],[-1.,0,0]])
    axes=np.tile(directions[None,:,:],(2,1,1))
    assigned=np.ones((2,2),int)
    result=direction_error_suite(directions,axes,assigned)
    np.testing.assert_allclose(result['angular_mae_deg'],[[180,0],[180,0]])
    np.testing.assert_allclose(result['cosine_loss'],[[2,0],[2,0]])


def test_null_preserves_blocks_and_is_shared_between_methods():
    angles=np.tile(np.array([[0.,180],[180,0],[45,135]])[None,:,:],(2,1,1))
    assignments=np.array([[0,1,0],[0,1,0]])
    null,info=block_direction_null(angles,assignments,np.array([0,0,1]),100,4)
    np.testing.assert_array_equal(null[0],null[1])
    assert set(null[0])=={15.,135.}
    assert info['singleton_events']==1


def test_holm_correction():
    np.testing.assert_allclose(holm_adjust([.04,.001,.03]),[.06,.003,.06])
