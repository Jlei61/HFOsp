#!/usr/bin/env python3
"""2D 触点平面读出静态图（per subject/template）。
口径锁：早端画成带不确定度的小入口群；禁 "固定稳定早端 / 弹性入口区"。
SOZ：仅描述性叠加，图注固定 "SOZ overlay only, not metric input"。
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src import propagation_contact_plane_readout as R

BASE = _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout"


def plot_record(rec, out_png):
    X, Y = R.make_plane_grid()
    fldT = R.smooth_field(rec, X, Y, sigma_xy=None, scalar="rank")
    chans = rec["channels"]
    xs = np.array([c["x_norm"] for c in chans]); ys = np.array([c["y_norm"] for c in chans])
    rk = np.array([c["typical_rank"] for c in chans]); sp = np.array([c["support"] for c in chans])
    soz = np.array([bool(c.get("is_soz")) for c in chans])
    ingrid = (xs >= R.X_LO) & (xs <= R.X_HI) & (ys >= -R.Y_EXT) & (ys <= R.Y_EXT)
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    ax = ax.ravel()
    sc = ax[0].scatter(xs[ingrid], ys[ingrid], c=rk[ingrid],
                       s=40 + 200*sp[ingrid], cmap="viridis",
                       vmin=0, vmax=1,
                       edgecolors=["k" if z else "none" for z in soz[ingrid]],
                       linewidths=1.5)
    if (~ingrid).any():
        xc = np.clip(xs[~ingrid], R.X_LO, R.X_HI); yc = np.clip(ys[~ingrid], -R.Y_EXT, R.Y_EXT)
        ax[0].scatter(xc, yc, marker="^", facecolors="none", edgecolors="red",
                      s=70, linewidths=1.5, zorder=5)
        ax[0].text(0.02, 0.02, f"{int((~ingrid).sum())} contacts outside comparison field",
                   transform=ax[0].transAxes, fontsize=8, color="red", va="bottom")
    ax[0].set_xlim(R.X_LO, R.X_HI); ax[0].set_ylim(-R.Y_EXT, R.Y_EXT)
    ax[0].set_aspect("equal", adjustable="box")
    ax[0].set_title("contacts: color=order, size=support\n(black ring = SOZ overlay)")
    ax[0].set_xlabel("along-axis (norm; 0 = early-core centroid)")
    ax[0].set_ylabel("transverse (norm, signed; display only)")
    # subject-specific mm reference layer — does NOT alter the normalized comparison
    # plane (field correlation stays on shared norm coords). x mm = projection along
    # the early→late axis from the early-core centroid; y mm = signed PCA-residual
    # projection (sign is display convention, NOT anatomical left/right).
    nsm = rec.get("norm_scale_mm"); axlen = rec.get("axis_length_mm", float("nan"))
    if nsm and np.isfinite(nsm) and nsm > 0:
        ax[0].text(0.02, 0.98, f"1 norm = {nsm:.0f} mm | core dist = {axlen:.0f} mm",
                   transform=ax[0].transAxes, fontsize=8, va="top", color="0.25")
        bar = 20.0 / nsm  # 20 mm expressed in normalized units
        if bar < 0.9 * (R.X_HI - R.X_LO):
            x0 = R.X_HI - 0.06 * (R.X_HI - R.X_LO) - bar
            y0 = -R.Y_EXT + 0.08 * (2 * R.Y_EXT)
            ax[0].plot([x0, x0 + bar], [y0, y0], color="k", lw=3, solid_capstyle="butt")
            ax[0].text(x0 + bar / 2, y0 + 0.025 * (2 * R.Y_EXT), "20 mm",
                       ha="center", va="bottom", fontsize=8)
    plt.colorbar(sc, ax=ax[0], label="typical order (0=early,1=late)")
    for a, key, ttl, cm in [(ax[1], "T", "smoothed order field", "viridis"),
                            (ax[2], "S", "support field", "magma"),
                            (ax[3], "U", "uncertainty field\n(early = small taking-turns group)",
                             "cividis")]:
        F = fldT[key].copy()
        if key != "S":
            F = np.where(fldT["mask"], F, np.nan)
        im = a.imshow(F, origin="lower", extent=[R.X_LO, R.X_HI, -R.Y_EXT, R.Y_EXT],
                      aspect="equal", cmap=cm)
        a.set_title(ttl); plt.colorbar(im, ax=a)
    flags = rec.get("flags", {})
    scalars = rec.get("scalars", {})
    flag_txt = "1D" if flags.get("one_dimensional_sampling") else "2D"
    rho = scalars.get("rank_vs_xnorm_spearman", float("nan"))
    oof = rec.get("out_of_field", {}).get("count", 0)
    weak = " | WEAK rank-axis" if (np.isfinite(rho) and abs(rho) < 0.3) else ""
    amb = rec.get("soz_ambiguous", [])
    fig.suptitle(
        f"{rec['dataset']}:{rec['subject']} {rec['template_id']} | {flag_txt} | "
        f"rho_x_rank={rho:.2f} | out_of_field={oof}{weak} | "
        "SOZ overlay only, not metric input"
        + (f" | SOZ ambiguous: {amb}" if amb else ""))
    fig.tight_layout()
    fig.savefig(out_png, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", default=str(BASE / "real_subjects"))
    ap.add_argument("--out", default=str(BASE / "figures/static_maps"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(args.real_dir).glob("*.json")):
        rec = json.loads(f.read_text())
        if not rec.get("channels"):
            continue
        plot_record(rec, out / f"{f.stem}.png")
        print(f"  {f.stem}.png")


if __name__ == "__main__":
    main()
