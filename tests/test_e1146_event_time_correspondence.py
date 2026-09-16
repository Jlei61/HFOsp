import numpy as np
from scripts.analyze_e1146_event_time_correspondence import contained, outside_seizures, label_permutation


def test_actual_event_boundary_instead_of_parent_hour():
    # One hour contains a seizure at 10--20 s, but the other events remain usable.
    starts=np.array([5.,9.9,10.,19.9,20.,3599.9])
    ends=starts+.25
    inv=[{'onset':10.,'offset':20.}]
    np.testing.assert_array_equal(outside_seizures(starts,ends,inv),[True,False,False,False,True,True])
    np.testing.assert_array_equal(contained(starts,ends,20.,3600.),[False,False,False,False,True,False])
    assert not contained(starts,ends,30.,20.).any()


def test_postictal_sensitivity_uses_event_time_and_exact_edges():
    inv=[{'onset':10.,'offset':20.}]
    starts=np.array([5.,20.,3619.9,3620.]);ends=starts+.25
    np.testing.assert_array_equal(outside_seizures(starts,ends,inv,60),[True,False,False,True])


def test_exact_permutation_balanced_known_case_and_missing_class():
    r=label_permutation([0.,0.,1.,1.],[False,False,True,True])
    assert r['delta_ta_share']==1
    assert r['n_permutations']==6
    assert np.isclose(r['exact_two_sided_p'],2/6)
    assert np.isclose(r['exact_one_sided_positive_p'],1/6)
    assert label_permutation([.2,.3],[True,True])['status']=='NOT_ESTIMABLE_MISSING_LABEL_GROUP'
