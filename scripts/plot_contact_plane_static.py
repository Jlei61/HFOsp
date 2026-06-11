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
    xs = [c["x_norm"] for c in chans]; ys = [c["y_norm"] for c in chans]
    rk = [c["typical_rank"] for c in chans]; sp = [c["support"] for c in chans]
    soz = [c.get("is_soz", False) for c in chans]
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    sc = ax[0].scatter(xs, ys, c=rk, s=[40 + 200*s for s in sp], cmap="viridis",
                       vmin=0, vmax=1, edgecolors=["k" if z else "none" for z in soz],
                       linewidths=1.5)
    ax[0].set_title("contacts: color=order, size=support\n(black ring = SOZ overlay)")
    ax[0].set_xlabel("along-axis (norm)"); ax[0].set_ylabel("transverse (norm, display only)")
    plt.colorbar(sc, ax=ax[0], label="typical order (0=early,1=late)")
    for a, key, ttl, cm in [(ax[1], "T", "smoothed order field", "viridis"),
                            (ax[2], "S", "support field", "magma"),
                            (ax[3], "U", "uncertainty field\n(early = small taking-turns group)",
                             "cividis")]:
        F = fldT[key].copy()
        if key != "S":
            F = np.where(fldT["mask"], F, np.nan)
        im = a.imshow(F, origin="lower", extent=[R.X_LO, R.X_HI, -R.Y_EXT, R.Y_EXT],
                      aspect="auto", cmap=cm)
        a.set_title(ttl); plt.colorbar(im, ax=a)
    amb = rec.get("soz_ambiguous", [])
    fig.suptitle(f"{rec['dataset']}:{rec['subject']} {rec['template_id']} | "
                 f"SOZ overlay only, not metric input"
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
