# src/sef_hfo_pulse.py
"""SEF-HFO Step-0b finite-pulse response: wavefront-aware classifier, adaptive thresholds."""
import numpy as np

def _centroid_x(frame, rest, detect):
    above = frame > (rest + detect)
    if not above.any(): return np.nan
    xs = np.where(above)[1]; return float(xs.mean())

def classify_response(activity, stim_mask, rest_level, runaway_frac=0.5, return_frac=0.2,
                      detect=0.05, travel_cells=1.0):
    above = activity > (rest_level + detect)
    extent = above.reshape(above.shape[0], -1).mean(axis=1)
    stim_extent = float(stim_mask.mean()); max_extent = float(extent.max()); final_extent = float(extent[-1])
    cx = np.array([_centroid_x(f, rest_level, detect) for f in activity])
    cx0 = next((c for c in cx if not np.isnan(c)), np.nan)
    moved = np.nanmax(np.abs(cx - cx0)) if not np.isnan(cx0) else 0.0
    grew = max_extent > 1.5 * stim_extent
    returned = final_extent <= return_frac * max(max_extent, 1e-12)
    if max_extent >= runaway_frac:
        return "runaway"
    if grew and moved < travel_cells:
        return "global_synchronous"          # extent grew but front did NOT travel -> not propagation
    if grew and moved >= travel_cells and not returned:
        return "runaway"
    if grew and moved >= travel_cells and returned:
        return "self_limited_propagation"
    if returned:
        return "extinction"
    return "local_bump"
