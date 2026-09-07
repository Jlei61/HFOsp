"""Low-dimensional dual-core geometry with explicit axis and boundary constraints."""
from __future__ import annotations
import numpy as np
from src.topic4_manual_dual_core import budget_matched_dual_core_h


def axis_aligned_core_field(positions, *, midpoint_mm, separation_mm,
                            axis_deg, target_count, sheet_size_mm=20.,
                            boundary_clearance_mm=1.5):
    """Construct equal-radius, budget-matched cores; reject boundary violations.

    The declared axis is a structural kernel parameter. A downstream weighted
    graph must separately report its achieved axis; this function cannot ensure
    that a sampled/reweighted graph has precisely the declared direction.
    """
    pos=np.asarray(positions,float);mid=np.asarray(midpoint_mm,float)
    values=np.asarray([separation_mm,axis_deg,sheet_size_mm,boundary_clearance_mm],float)
    if pos.ndim!=2 or pos.shape[1]!=2 or not np.isfinite(pos).all():
        raise ValueError('positions must be finite (N,2) coordinates')
    if mid.shape!=(2,) or not np.isfinite(mid).all() or not np.isfinite(values).all():
        raise ValueError('geometry parameters must be finite')
    if separation_mm<=0 or sheet_size_mm<=0 or boundary_clearance_mm<0:
        raise ValueError('invalid separation, sheet size or boundary clearance')
    if np.any((pos<0)|(pos>sheet_size_mm)):
        raise ValueError('neuron positions lie outside the sheet')
    angle=np.deg2rad(axis_deg);direction=np.array([np.cos(angle),np.sin(angle)])
    centers=mid[None,:]+np.array([-.5,.5])[:,None]*separation_mm*direction
    h,audit=budget_matched_dual_core_h(pos,centers,target_count=target_count)
    radius=audit['distance_cutoff_mm']
    clearance=np.minimum(centers,sheet_size_mm-centers).min(axis=1)-radius
    if radius*2>=separation_mm or min(audit['selected_per_core'])==0:
        raise ValueError('two distinct populated cores are required')
    if np.any(clearance<boundary_clearance_mm-1e-10):
        raise ValueError(f'core boundary clearance {clearance.tolist()} is below requirement')
    return h,{**audit,'centers_mm':centers.tolist(),'midpoint_mm':mid.tolist(),
              'axis_deg':float(axis_deg),'separation_mm':float(separation_mm),
              'boundary_clearance_mm':clearance.tolist(),
              'required_boundary_clearance_mm':float(boundary_clearance_mm),
              'geometry_family':'axis_aligned_interior_budget_matched_v1'}
