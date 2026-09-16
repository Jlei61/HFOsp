import numpy as np
import pytest
from src.topic4_axis_aligned_core import axis_aligned_core_field


def positions():
    return np.random.default_rng(2511).uniform(0,20,(32000,2))


@pytest.mark.parametrize('angle',[-22.805,-31.375,40.])
def test_geometry_preserves_budget_axis_and_interior_margin(angle):
    h,a=axis_aligned_core_field(positions(),midpoint_mm=[10,10],separation_mm=12,
                                axis_deg=angle,target_count=1499)
    assert h.sum()==1499
    cs=np.array(a['centers_mm']);v=cs[1]-cs[0]
    np.testing.assert_allclose(cs.mean(0),[10,10])
    np.testing.assert_allclose(v/np.linalg.norm(v),[np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
    assert min(a['boundary_clearance_mm'])>=1.5
    assert min(a['selected_per_core'])>0


def test_edge_geometry_is_rejected_instead_of_clipped():
    with pytest.raises(ValueError,match='boundary clearance'):
        axis_aligned_core_field(positions(),midpoint_mm=[10,1.8],separation_mm=17,
                                axis_deg=4,target_count=1499)


def test_overlapping_cores_are_rejected():
    with pytest.raises(ValueError,match='distinct'):
        axis_aligned_core_field(positions(),midpoint_mm=[10,10],separation_mm=.1,
                                axis_deg=0,target_count=1499)
