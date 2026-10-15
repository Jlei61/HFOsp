"""Spatial bins used by the new-spike activity movie."""
import numpy as np

def sheet_bin_indices(positions: np.ndarray, *, bin_mm: float, sheet_mm: float) -> tuple[np.ndarray, int]:
    """Map E-neuron positions to a fixed square sheet grid."""
    positions = np.asarray(positions, float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError('positions must have shape (neuron, 2)')
    size = int(round(float(sheet_mm) / float(bin_mm)))
    if size <= 0 or not np.isclose(size * float(bin_mm), float(sheet_mm)):
        raise ValueError('bin size must tile the square sheet')
    xy = np.floor(positions / float(bin_mm)).astype(int)
    xy = np.clip(xy, 0, size - 1)
    return (xy[:, 1] * size + xy[:, 0], size)
