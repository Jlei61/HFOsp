"""Per-SNN-run event-diagnostic figure (Topic 4 M3+).

User request (2026-06-23): every SNN simulation should emit a figure in this style.
One representative (kick, seed), three panels:
  1. RASTER          — E-cell spikes; cells sorted by distance from the kick so the
                       outward spatial spread reads top-to-bottom. Subsampled if large.
  2. EARLY HEATMAP   — per-bin differenced (kick - sham) early-window response on the
                       spatial grid; marks the source bin + kick center.
  3. RETURN-TO-QUIET — population active fraction over time (kick solid, sham dashed);
                       marks t_kick, the kick drive window, and the event-aligned window.

Pure data helpers (tested) + a thin matplotlib layer. The runner supplies in-run spikes
(E_spk_bool), which are not persisted anywhere else, so the figure is generated in-run.
"""
import numpy as np

__all__ = ["distance_sort_index", "reshape_bins_to_grid", "active_fraction_trace",
           "median_representative", "plot_event_diagnostic"]


def median_representative(seeds, values):
    """The seed whose value is closest to the median of ``values`` — used to pick a
    REPRESENTATIVE (typical) event for the figure rather than cherry-picking seed 0.
    Returns None if empty."""
    if len(seeds) == 0:
        return None
    vals = np.asarray(values, dtype=float)
    med = float(np.median(vals))
    return seeds[int(np.argmin(np.abs(vals - med)))]


def distance_sort_index(posE, kick_center):
    """Cell indices ordered by distance from the kick center (nearest first)."""
    posE = np.asarray(posE, dtype=float)
    kc = np.asarray(kick_center, dtype=float)
    return np.argsort(np.linalg.norm(posE - kc[None, :], axis=1))


def reshape_bins_to_grid(values, n_bins_per_axis):
    """Per-bin vector (spatial_bins row-major: bin = iy*nb+ix, ix fast) -> (nb, nb) grid
    indexed [iy, ix], ready for imshow(origin='lower')."""
    nb = int(n_bins_per_axis)
    return np.asarray(values, dtype=float).reshape(nb, nb)


def active_fraction_trace(E_spk_bool, dt, bin_ms):
    """Fraction of E cells with >=1 spike per ``bin_ms`` time bin (1-D over time)."""
    E = np.asarray(E_spk_bool, dtype=bool)
    bs = max(1, int(round(bin_ms / dt)))
    nb = E.shape[0] // bs
    if nb == 0:
        return np.zeros(0, dtype=float)
    binned = E[:nb * bs].reshape(nb, bs, E.shape[1]).any(axis=1)   # (nb, NE)
    return binned.mean(axis=1).astype(float)


def plot_event_diagnostic(E_spk_kick, E_spk_only, posE, ea_net_bins, src_bin_idx,
                          bin_centers, n_bins_per_axis, dt, t_kick, dur_kick,
                          t0_ms, ea_lo, ea_hi, kick_center, out_path, title,
                          trace_bin_ms=1.0, max_raster_cells=1500):
    """Render the 3-panel event-diagnostic figure for one (kick, seed) to ``out_path``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    E_kick = np.asarray(E_spk_kick, dtype=bool)
    n_steps, NE = E_kick.shape
    t_ms = np.arange(n_steps) * dt
    order = distance_sort_index(posE, kick_center)
    # subsample cells (distance-sorted) for a readable raster
    if NE > max_raster_cells:
        sel = order[np.linspace(0, NE - 1, max_raster_cells).astype(int)]
    else:
        sel = order
    rank_of = {int(c): r for r, c in enumerate(sel)}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    # --- Panel 1: raster (distance-sorted) -------------------------------------
    ax = axes[0]
    xs, ys = [], []
    for r, c in enumerate(sel):
        ts = np.nonzero(E_kick[:, c])[0]
        if ts.size:
            xs.append(ts * dt)
            ys.append(np.full(ts.size, r))
    if xs:
        ax.scatter(np.concatenate(xs), np.concatenate(ys), s=0.5, c="k", marker=".",
                   linewidths=0)
    ax.axvline(t_kick, color="tab:blue", lw=1, label="kick on")
    ax.axvline(t_kick + dur_kick, color="tab:blue", ls=":", lw=1, label="kick off")
    if np.isfinite(t0_ms):
        ax.axvspan(t0_ms + ea_lo, t0_ms + ea_hi, color="tab:orange", alpha=0.2,
                   label="EA window")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("E cell (sorted by dist. from kick)")
    ax.set_title("raster — outward spread")
    ax.legend(loc="upper right", fontsize=7)

    # --- Panel 2: early per-bin differenced response heatmap -------------------
    # SOURCE-EXCLUDED color scale (P1-3): the source bin's huge response would otherwise
    # compress the neighbour differences (which ARE the W_shape). Mask the source bin and
    # scale the colormap on the non-source bins only. This mirrors the source-excluded
    # W_shape used for the numbers; the source is shown as a marker, not a color.
    ax = axes[1]
    grid = reshape_bins_to_grid(ea_net_bins, n_bins_per_axis).copy()
    nb = int(n_bins_per_axis)
    src_iy, src_ix = int(src_bin_idx) // nb, int(src_bin_idx) % nb
    grid[src_iy, src_ix] = np.nan                       # exclude source from the color scale
    bc = np.asarray(bin_centers, dtype=float)
    extent = [bc[:, 0].min(), bc[:, 0].max(), bc[:, 1].min(), bc[:, 1].max()]
    import matplotlib.cm as _cm
    cmap = _cm.get_cmap("viridis").copy()
    cmap.set_bad("lightgrey")
    vmax = np.nanmax(grid) if np.any(np.isfinite(grid)) else 1.0
    im = ax.imshow(grid, origin="lower", extent=extent, aspect="equal", cmap=cmap,
                   vmin=0.0, vmax=(vmax if vmax > 0 else 1.0))
    fig.colorbar(im, ax=ax, fraction=0.046, label="early Δ spikes (kick−sham), src excl.")
    sb = bc[int(src_bin_idx)]
    ax.plot(sb[0], sb[1], "rs", ms=9, mfc="none", mew=1.5, label="source bin (excl.)")
    ax.plot(kick_center[0], kick_center[1], "rx", ms=8, label="kick center")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title("early recruitment by bin (source-excluded scale)")
    ax.legend(loc="upper right", fontsize=7)

    # --- Panel 3: return-to-quiet trace ---------------------------------------
    ax = axes[2]
    tr_k = active_fraction_trace(E_kick, dt, trace_bin_ms)
    tr_o = active_fraction_trace(np.asarray(E_spk_only, dtype=bool), dt, trace_bin_ms)
    tt = (np.arange(tr_k.size) + 0.5) * trace_bin_ms
    ax.plot(tt, tr_k, "k-", lw=1.2, label="kick")
    ax.plot((np.arange(tr_o.size) + 0.5) * trace_bin_ms, tr_o, color="0.6", ls="--",
            lw=1, label="sham (no kick)")
    ax.axvline(t_kick, color="tab:blue", lw=1)
    ax.axvspan(t_kick, t_kick + dur_kick, color="tab:blue", alpha=0.12, label="kick drive")
    if np.isfinite(t0_ms):
        ax.axvspan(t0_ms + ea_lo, t0_ms + ea_hi, color="tab:orange", alpha=0.2,
                   label="EA window")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("active fraction of E cells")
    ax.set_title("return-to-quiet"); ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
