"""Physical-mm display smoother copied from plot_contact_plane_static.py."""
import numpy as np
VIS_GRID_N=140
VIS_MASK_REL=.02

def _smooth_rank_field_mm(x, y, rank, support, xlim, ylim, sigma_mm):
    gx = np.linspace(xlim[0], xlim[1], VIS_GRID_N)
    gy = np.linspace(ylim[0], ylim[1], VIS_GRID_N)
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    use = np.isfinite(x) & np.isfinite(y) & np.isfinite(rank) & (support > 0)
    S = np.zeros(X.size, float)
    WT = np.zeros(X.size, float)
    xx = X.ravel()
    yy = Y.ravel()
    sig2 = 2.0 * max(float(sigma_mm), 1e-6) ** 2
    for xi, yi, ri, si in zip(x[use], y[use], rank[use], support[use]):
        w = si * np.exp(-((xx - xi) ** 2 + (yy - yi) ** 2) / sig2)
        S += w
        WT += w * ri
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.where(S > 1e-12, WT / S, np.nan).reshape(X.shape)
    S = S.reshape(X.shape)
    mask = S >= (VIS_MASK_REL * float(np.nanmax(S))) if np.nanmax(S) > 0 else np.zeros_like(S, bool)
    return X, Y, np.where(mask, T, np.nan), S, mask
