import numpy as np


def test_response_support_keeps_available_bands_without_using_heldout_labels():
    from src.topic5_group_event_state.v039.ridge_transfer import fit_supported_response_columns
    y=np.array([[1,np.nan,np.nan],[2,np.nan,np.nan],[3,np.nan,np.nan],[4,9,8],[5,8,9]],float)
    phases=np.array(['FIT','FIT','FIT','INNER','SELECTION']);anchors=np.arange(5)
    assert fit_supported_response_columns(y,phases,anchors).tolist()==[True,False,False]
    # Repeated events in one anchor cannot manufacture component support.
    assert not fit_supported_response_columns(y,phases,np.array([0,0,0,1,2])).any()
from src.topic5_group_event_state.v039.ridge_transfer import fit_candidates,select_probe


def test_pooling_repeated_state_features_preserves_weighted_ridge_solution():
    rng=np.random.default_rng(5);state=rng.normal(size=(8,30));anchor=np.repeat(np.arange(8),[3,7,2,6,5,4,2,3]);x=state[anchor]
    y=x[:,:2]+rng.normal(size=(len(anchor),2))*.1;fit=np.flatnonzero(anchor<5)
    full,_=fit_candidates(x,y,fit,anchor);pooled,_=fit_candidates(x,y,fit,anchor,pool_fit_by_anchor=True)
    for a,b in zip(full,pooled):np.testing.assert_allclose(a['prediction'],b['prediction'],rtol=1e-9,atol=1e-9)


def test_zero_residual_remains_available_when_a_probe_harms_its_parent():
    target=np.zeros((6,2));parent=target.copy();anchor=np.arange(6)
    selected=select_probe([dict(alpha=1.,prediction=np.ones_like(target))],target,np.arange(3,6),anchor,parent)
    assert selected['model']['zero_residual'] and selected['inner_loss']==0
