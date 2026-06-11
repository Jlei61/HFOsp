"""Topic 5 Stage-2 recruitment instrument — figures.

Phase 2: sentinel overlay only (manual gate). Cohort figures are added in Phase 4.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def feature_center_time(n_frames, *, win_sec, hop_sec, pre_sec):
    """Window-CENTER time (sec, rel clinical onset) for a feature's frames. Reviewer P1:
    do NOT mix per-trace `t` (line-length returns start time, spectrogram returns center)."""
    return np.arange(n_frames) * hop_sec + win_sec / 2.0 - pre_sec


def plot_sentinel_overlay(sw, out, lambdas, pre_sec, path, *, k=8):
    """Two panels for the human gate:
      A: raw EEG of the earliest-K recruited contacts, vlines at each contact's fused
         onset (black) + the data-driven global onset (red), zoomed to the recruitment band.
      B: per-feature onset dots for those contacts (do the 5 detectors agree on the
         earliest channels and their order?). x = onset seconds rel clinical onset.
    """
    fused = np.asarray(out["fused_onset"], float)
    ch_names = list(out["channels"])
    t_global = float(out["t_global_sec"])
    recruited = np.where(np.isfinite(fused))[0]
    if recruited.size == 0:
        return
    order = recruited[np.argsort(fused[recruited])][:k]      # earliest-K recruited
    t_axis = sw.t_axis
    lo, hi = t_global - 3.0, t_global + 15.0
    sel = (t_axis >= lo) & (t_axis <= hi)

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(11, 9))
    # --- Panel A: raw EEG, offset-stacked, earliest contacts at top ---
    sig = np.asarray(sw.signal, float)
    step = 6.0 * (np.nanstd(sig[order][:, sel]) + 1e-9)
    for row, ci in enumerate(order):
        y = sig[ci, sel]
        y = (y - np.nanmedian(y))
        axA.plot(t_axis[sel], y + row * step, lw=0.5, color="0.25")
        axA.axvline(fused[ci], color="k", lw=0.8, alpha=0.6)
        axA.text(lo, row * step, f" {ch_names[ci]}", va="center", ha="left", fontsize=8)
    axA.axvline(t_global, color="crimson", lw=1.6, ls="--", label="data-driven global onset")
    axA.set_xlim(lo, hi)
    axA.set_yticks([])
    axA.set_xlabel("time relative to seizure onset (s)")
    axA.set_title(f"{sw.subject} seizure {sw.seizure_id}: raw EEG of earliest-recruited contacts")
    axA.legend(loc="upper right", fontsize=8)

    # --- Panel B: per-feature onset dots, contacts sorted by fused onset ---
    pf = out["per_feature_onset"]
    feats = [f for f in ("line_length", "broadband", "hfa", "spectral_edge", "er") if f in pf]
    colors = {"line_length": "#1f77b4", "broadband": "#2ca02c", "hfa": "#ff7f0e",
              "spectral_edge": "#9467bd", "er": "#7f7f7f"}
    labels = {"line_length": "line length", "broadband": "broadband power",
              "hfa": "high-freq activity", "spectral_edge": "spectral edge",
              "er": "energy ratio (held-out)"}
    for row, ci in enumerate(order):
        for f in feats:
            o = np.asarray(pf[f], float)[ci]
            if np.isfinite(o):
                axB.scatter(o, row, s=42, color=colors[f], edgecolor="none",
                            label=labels[f] if row == 0 else None, zorder=3)
        axB.scatter(fused[ci], row, s=90, marker="x", color="k",
                    label="fused onset" if row == 0 else None, zorder=4)
    axB.axvline(t_global, color="crimson", lw=1.4, ls="--")
    axB.set_yticks(range(len(order)))
    axB.set_yticklabels([ch_names[ci] for ci in order], fontsize=8)
    axB.invert_yaxis()
    axB.set_xlabel("per-detector onset time relative to seizure onset (s)")
    axB.set_title("Do the detectors agree on who lights up first?  "
                  f"(amp-family agreement={out['agreement'].get('amplitude_family_agreement')}, "
                  f"spectral support={out['agreement'].get('spectral_support')})")
    axB.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("Phase 2 sentinel plotter. Driven by scripts/run_topic5_ictal_recruitment.py sentinel.")
