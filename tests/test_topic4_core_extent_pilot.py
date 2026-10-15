"""Scientific invariants of geometry-only interventions and frozen observations."""
import numpy as np
from scripts.pilot_topic4_core_extent import geometry_field
from scripts.paper_figures.plot_topic4_core_extent_pilot import ranks


def test_expand_a_preserves_b_and_adds_only_a_neighborhood():
    pos=np.array([[0.,0.],[1.5,0.],[5.,0.],[6.5,0.],[3.,3.]])
    centers=[[0.,0.],[5.,0.]]
    base,_=geometry_field(pos,centers,[1.,1.])
    expand,_=geometry_field(pos,centers,[2.,1.])
    np.testing.assert_array_equal(base,[1,0,1,0,0])
    np.testing.assert_array_equal(expand,[1,1,1,0,0])


def test_overlapping_cores_do_not_double_threshold_amplitude():
    field,_=geometry_field([[1.,0.]],[[0.,0.],[2.,0.]],[2.,2.])
    np.testing.assert_array_equal(field,[1.])


def test_permutation_of_equal_cores_keeps_physical_field():
    pos=np.array([[0.,0.],[1.,0.],[2.,0.]])
    a,_=geometry_field(pos,[[0.,0.],[2.,0.]],[.5,.5])
    b,_=geometry_field(pos,[[2.,0.],[0.,0.]],[.5,.5])
    np.testing.assert_array_equal(a,b)


def test_missing_contact_stays_missing_and_rank_does_not_encode_absolute_time():
    table=np.array([[10.,30.,np.nan,20.],[110.,130.,np.nan,120.]])
    result=ranks(table)
    np.testing.assert_allclose(result[0],[0,1,np.nan,.5],equal_nan=True)
    np.testing.assert_allclose(result[0],result[1],equal_nan=True)
