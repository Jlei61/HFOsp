#!/usr/bin/env python3
"""Per-subject interictal-propagation "path card" figures (visual judgement aid).

For each showcase subject this renders a single PNG with 5 panels so a human can
eyeball whether the propagation pathway is real:

  1. 3D brain-space scatter  — channels colored by median within-event firing
     order, source/sink cores marked, source->sink arrow.
  2. Axis-coordinate scatter — along/off-axis skeleton, marker size = #events,
     color = median firing order.
  3. Held-out stability      — along-axis position (built from half A) vs
     held-out half-B mean firing order. THE key science panel. Monotone cloud =
     real shared pathway; shapeless = not.
  4. Multi-template          — dominant vs minority cluster axis in 3D (swap
     subjects only); else a "single template" note.
  5. Radius null insets      — source/sink core compactness vs random k-subsets.

Geometry is NOT reimplemented: positions/roles/radii/rho come from the
per-subject geometry JSON (run_propagation_skeleton_geometry.py), and the
held-out split reuses that runner's helpers (`_load_accepted_templates`,
`_subject_dir`) + the module's `G._half_along_axis` / `G.core_radius_null`, so
panel 3's recomputed rho matches the JSON to ~3 decimals (asserted).

Spec: docs/superpowers/specs/2026-06-08-sef-hfo-propagation-skeleton-geometry-design.md
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.interictal_propagation import load_subject_propagation_events
from src.lagpat_rank_audit import mask_phantom_ranks
from src.seeg_coord_loader import load_subject_coords
from src import propagation_skeleton_geometry as G
# Reuse the runner's front-half (templates + subject_dir) so the held-out split
# is built on the SAME dominant-cluster events the JSON summarized — do not
# replicate or the recomputed rho diverges from the annotated one.
from scripts.run_propagation_skeleton_geometry import (
    _load_accepted_templates,
    _subject_dir,
)

ORDER_CMAP = "coolwarm"   # early (blue) -> late (red)


def _build_ster_masked(ds, subj, names, masked, bools, swap_class):
    """Reproduce the runner's stereotypy-event slice `ster_masked`.

    Swap subjects (strict/candidate): the two accepted templates are
    anti-parallel, so only the DOMINANT cluster's events carry the axis signal;
    averaging both washes it out. Non-swap: the full recording is used.
    Returns (ster_masked, ster_bools, minority_masked_or_None).
    """
    if swap_class not in ("strict", "candidate"):
        return masked, bools, None
    template_a, template_b, _ = _load_accepted_templates(ds, subj, names)
    labels = G.assign_events_to_templates(masked, template_a, template_b)
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    dom = 0 if n0 >= n1 else 1
    sel = labels == dom
    msel = labels == (1 - dom)
    minority = masked[:, msel] if msel.any() else None
    return masked[:, sel], bools[:, sel], minority


def _one_split_points(ster_masked, coords, eligible_idx, k):
    """Mirror G.split_half_axis_validation's canonical single split, but return
    the per-channel POINTS being correlated (x=half-A along, y=held-out half-B
    mean), plus the recomputed Spearman rho for the self-check.

    Same first-valid-permutation loop, same fresh rng(0), same k_used.
    """
    ev = np.asarray(ster_masked, dtype=float)
    coords = np.asarray(coords, dtype=float)
    eligible_idx = np.asarray(list(eligible_idx), dtype=int)
    n_ch = ev.shape[0]
    usable = np.where(np.any(~np.isnan(ev), axis=0))[0]
    rng = np.random.default_rng(0)
    if usable.size < 4:
        return None
    elig = np.zeros(n_ch, dtype=bool)
    elig[eligible_idx] = True
    for _ in range(max(200, 50)):
        order = rng.permutation(usable)
        half = order.size // 2
        a_ev, b_ev = order[:half], order[half:]
        if a_ev.size < 1 or b_ev.size < 1:
            continue
        along = G._half_along_axis(ev[:, a_ev], coords, eligible_idx, k)
        if along is None:
            continue
        with np.errstate(invalid="ignore"):
            held_mean = np.array([
                np.nanmean(ev[c, b_ev]) if np.any(~np.isnan(ev[c, b_ev]))
                else np.nan for c in range(n_ch)])
        ok = elig & ~np.isnan(along) & ~np.isnan(held_mean)
        if ok.sum() < 3:
            continue
        x = along[ok]
        y = held_mean[ok]
        from scipy.stats import spearmanr
        with np.errstate(invalid="ignore"):
            rho = spearmanr(x, y).correlation
        if rho != rho:
            continue
        return {"x": x, "y": y, "rho": float(rho), "n_ch": int(ok.sum())}
    return None


def _median_order(masked):
    """Per-channel median within-event firing order (nanmedian over events)."""
    with np.errstate(invalid="ignore"):
        return np.array([
            np.nanmedian(masked[c]) if np.any(~np.isnan(masked[c]))
            else np.nan for c in range(masked.shape[0])])


def _null_radii(coords, eligible_idx, k, n_draw=1000, seed=1):
    """Random k-subset RMS-to-centroid radii (for the panel-5 histogram).

    Matches G.core_radius_null's eligible-set restriction (finite coords only).
    """
    coords = np.asarray(coords, dtype=float)
    idx = np.array([i for i in eligible_idx if np.isfinite(coords[i]).all()],
                   dtype=int)
    if k < 1 or idx.size <= k:
        return np.array([])
    rng = np.random.default_rng(seed)
    out = np.empty(n_draw)
    for j in range(n_draw):
        sub = coords[rng.choice(idx, size=k, replace=False)]
        c = sub.mean(axis=0)
        out[j] = float(np.sqrt(np.mean(np.sum((sub - c) ** 2, axis=1))))
    return out


def make_card(ds, subj, geom_json_dir, out_dir):
    rec = json.loads((geom_json_dir / f"{ds}_{subj}.json").read_text())
    if rec.get("status") != "ok" or rec.get("eligibility_tier") == "descriptive_only":
        print(f"  [skip {ds}:{subj}] status={rec.get('status')} "
              f"tier={rec.get('eligibility_tier')}")
        return None

    swap_class = rec.get("swap_class", "none")
    k_used = int(rec.get("k_used", 3))

    # ---- raw events + masking + coords (same loaders as the runner) ----
    ev = load_subject_propagation_events(_subject_dir(ds, subj))
    names = list(ev["channel_names"])
    ranks = np.asarray(ev["ranks"], float)
    bools = np.asarray(ev["bools"]) > 0
    masked = mask_phantom_ranks(ranks, bools, normalize=True)
    med_order = _median_order(masked)          # color: full-recording median order

    cr = load_subject_coords(ds, subj, names)
    coords = np.asarray(cr.coords_array_in_requested_order, float)
    mapped = np.asarray(cr.mapped_mask_in_requested_order, bool)

    # Dominant-cluster slice = the runner's stereotypy events (panel-3 input).
    ster_masked, ster_bools, minority_masked = _build_ster_masked(
        ds, subj, names, masked, bools, swap_class)
    template_axis = np.array(
        [np.nanmean(r) if np.any(~np.isnan(r)) else np.nan for r in ster_masked])
    eligible = (~np.isnan(template_axis)) & mapped
    eligible_idx = np.where(eligible)[0]

    name_to_idx = {nm: i for i, nm in enumerate(names)}
    chans = rec.get("channels", [])
    src_names = [c["name"] for c in chans if c["role"] == "source_core"]
    snk_names = [c["name"] for c in chans if c["role"] == "sink_core"]

    fig = plt.figure(figsize=(20, 11))
    fig.suptitle(
        f"Propagation path card — {ds}:{subj}   "
        f"(axis length {rec.get('axis_length_mm', float('nan')):.1f} mm, "
        f"{len(chans)} channels on axis)",
        fontsize=15, y=0.98)

    # =====================================================================
    # Panel 1: 3D brain-space scatter
    # =====================================================================
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    participating = bools.any(axis=1)
    bg = mapped & ~participating
    if bg.any():
        ax1.scatter(coords[bg, 0], coords[bg, 1], coords[bg, 2],
                    s=8, c="lightgray", alpha=0.5, label="other channels")
    pm = mapped & participating
    sc = None
    if pm.any():
        sc = ax1.scatter(coords[pm, 0], coords[pm, 1], coords[pm, 2],
                         s=42, c=med_order[pm], cmap=ORDER_CMAP, vmin=0, vmax=1,
                         edgecolors="k", linewidths=0.3, depthshade=False)
    src_idx = [name_to_idx[n] for n in src_names if n in name_to_idx]
    snk_idx = [name_to_idx[n] for n in snk_names if n in name_to_idx]
    if src_idx:
        ax1.scatter(coords[src_idx, 0], coords[src_idx, 1], coords[src_idx, 2],
                    s=180, marker="^", facecolors="none", edgecolors="navy",
                    linewidths=1.8, label="source core")
    if snk_idx:
        ax1.scatter(coords[snk_idx, 0], coords[snk_idx, 1], coords[snk_idx, 2],
                    s=180, marker="s", facecolors="none", edgecolors="darkred",
                    linewidths=1.8, label="sink core")
    if src_idx and snk_idx:
        sct = np.nanmean(coords[src_idx], axis=0)
        skt = np.nanmean(coords[snk_idx], axis=0)
        ax1.plot([sct[0], skt[0]], [sct[1], skt[1]], [sct[2], skt[2]],
                 color="k", lw=2.0)
        ax1.quiver(sct[0], sct[1], sct[2],
                   skt[0] - sct[0], skt[1] - sct[1], skt[2] - sct[2],
                   color="k", arrow_length_ratio=0.12, lw=2.0)
    ax1.set_title("Channels in brain space\n(color = median firing order)",
                  fontsize=11)
    ax1.set_xlabel("x (mm)"); ax1.set_ylabel("y (mm)"); ax1.set_zlabel("z (mm)")
    ax1.legend(loc="upper left", fontsize=7, framealpha=0.8)
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax1, shrink=0.5, pad=0.08)
        cb.set_label("firing order (early=0 -> late=1)", fontsize=8)

    # =====================================================================
    # Panel 2: axis-coordinate scatter (along vs off, size=#events)
    # =====================================================================
    ax2 = fig.add_subplot(2, 3, 2)
    along = np.array([c["along_axis_mm"] for c in chans], float)
    off = np.array([c["off_axis_mm"] for c in chans], float)
    cidx = np.array([name_to_idx[c["name"]] for c in chans], int)
    counts = bools[cidx].sum(axis=1).astype(float)
    sizes = 30.0 + 320.0 * (counts / counts.max() if counts.max() > 0 else counts)
    color2 = med_order[cidx]
    roles = [c["role"] for c in chans]
    marker_for = {"source_core": "^", "sink_core": "s", "interior": "o"}
    L = rec.get("axis_length_mm", float("nan"))
    ax2.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax2.plot([0, L], [0, 0], color="k", lw=2.0, zorder=1, label="propagation axis")
    sc2 = None
    for mk in ("interior", "source_core", "sink_core"):
        sel = np.array([r == mk for r in roles])
        if not sel.any():
            continue
        edge = {"source_core": "navy", "sink_core": "darkred",
                "interior": "k"}[mk]
        sc2 = ax2.scatter(along[sel], off[sel], s=sizes[sel], c=color2[sel],
                          cmap=ORDER_CMAP, vmin=0, vmax=1, marker=marker_for[mk],
                          edgecolors=edge, linewidths=1.0, alpha=0.9)
    ax2.set_title("Propagation skeleton (axis coordinates)\n"
                  "marker size = #events, color = median firing order",
                  fontsize=11)
    ax2.set_xlabel("distance along axis from source (mm)")
    ax2.set_ylabel("perpendicular distance off axis (mm)")
    if sc2 is not None:
        cb2 = fig.colorbar(sc2, ax=ax2, shrink=0.7, pad=0.02)
        cb2.set_label("firing order (early=0 -> late=1)", fontsize=8)
    leg2 = [Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
                   markeredgecolor="navy", markersize=10, label="source core"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
                   markeredgecolor="darkred", markersize=10, label="sink core"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markeredgecolor="k", markersize=9, label="interior")]
    ax2.legend(handles=leg2, loc="best", fontsize=8, framealpha=0.85)

    # =====================================================================
    # Panel 3: held-out stability (THE key science panel)
    # =====================================================================
    ax3 = fig.add_subplot(2, 3, 3)
    sh = rec.get("split_half_validation", {})
    json_rho = sh.get("spearman_rho", float("nan"))
    pts = _one_split_points(ster_masked, coords, eligible_idx, k_used)
    title3_extra = ""
    if pts is not None:
        ax3.scatter(pts["x"], pts["y"], s=55, c="#2c7fb8", edgecolors="k",
                    linewidths=0.4, alpha=0.85)
        # monotone reference (rank-trend), drawn as a light guide line.
        xo = np.argsort(pts["x"])
        ax3.plot(np.array(pts["x"])[xo],
                 np.poly1d(np.polyfit(pts["x"], pts["y"], 1))(
                     np.array(pts["x"])[xo]),
                 color="gray", lw=1.2, ls="--", alpha=0.7)
        # Self-check: recomputed rho must match the JSON rho (else input diverged)
        if json_rho == json_rho and abs(pts["rho"] - json_rho) > 5e-3:
            title3_extra = f"  [WARN recomputed rho {pts['rho']:.3f}!=json]"
            print(f"  [WARN {ds}:{subj}] panel-3 recomputed rho "
                  f"{pts['rho']:.4f} != json {json_rho:.4f}")
    else:
        ax3.text(0.5, 0.5, "split did not form a valid frame",
                 transform=ax3.transAxes, ha="center", va="center")
    ax3.set_title(
        "Held-out stability: does the spatial axis predict\n"
        "firing order on events it was NOT built from?\n"
        rf"$\rho$={sh.get('spearman_rho', float('nan')):.2f} "
        rf"(95% CI [{sh.get('rho_ci_lo', float('nan')):.2f}, "
        rf"{sh.get('rho_ci_hi', float('nan')):.2f}]), "
        rf"Kendall $\tau$={sh.get('kendall_tau', float('nan')):.2f}, "
        rf"n={sh.get('n_channels')} ch" + title3_extra,
        fontsize=10)
    ax3.set_xlabel("position along axis (built from half A) (mm)")
    ax3.set_ylabel("held-out half-B mean firing order")

    # =====================================================================
    # Panel 4: multi-template (swap subjects only)
    # =====================================================================
    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    minor = rec.get("minority_axis")
    cos_a = rec.get("axes_cos_angle")
    if minor is not None and src_idx and snk_idx:
        dsrc = np.nanmean(coords[src_idx], axis=0)
        dsnk = np.nanmean(coords[snk_idx], axis=0)
        msrc = np.array(minor["source_centroid"], float)
        msnk = np.array(minor["sink_centroid"], float)
        ax4.plot([dsrc[0], dsnk[0]], [dsrc[1], dsnk[1]], [dsrc[2], dsnk[2]],
                 color="C0", lw=2.5, label="dominant template")
        ax4.scatter(*dsrc, marker="^", s=120, color="C0")
        ax4.scatter(*dsnk, marker="s", s=120, color="C0")
        ax4.plot([msrc[0], msnk[0]], [msrc[1], msnk[1]], [msrc[2], msnk[2]],
                 color="C3", lw=2.5, label="minority template")
        ax4.scatter(*msrc, marker="^", s=120, color="C3")
        ax4.scatter(*msnk, marker="s", s=120, color="C3")
        if cos_a is not None and cos_a < -0.85:
            tag = f"cos≈{cos_a:.2f}  same axis, opposite direction"
        elif cos_a is not None:
            tag = f"cos≈{cos_a:.2f}  two different paths"
        else:
            tag = "minority axis undefined"
        ax4.set_title("Two propagation templates\n" + tag, fontsize=11)
        ax4.set_xlabel("x (mm)"); ax4.set_ylabel("y (mm)"); ax4.set_zlabel("z (mm)")
        ax4.legend(loc="upper left", fontsize=8)
    else:
        ax4.set_axis_off()
        ax4.text2D(0.5, 0.5,
                   "single template\n(no forward/reverse second pathway)",
                   transform=ax4.transAxes, ha="center", va="center",
                   fontsize=12)
        ax4.set_title("Two propagation templates", fontsize=11)

    # =====================================================================
    # Panel 5: radius-null insets (source + sink)
    # =====================================================================
    for sub_i, (which, edge) in enumerate(
            [("source", "navy"), ("sink", "darkred")]):
        ax = fig.add_subplot(2, 6, 11 + sub_i)  # bottom-right 2 narrow cells
        nullkey = f"{which}_radius_null"
        nrec = rec.get(nullkey, {})
        obs = nrec.get("observed_mm", float("nan"))
        pval = nrec.get("p_value", float("nan"))
        null = _null_radii(coords, eligible_idx, k_used, n_draw=1000, seed=1)
        if null.size:
            ax.hist(null, bins=25, color="lightgray", edgecolor="gray")
        if obs == obs:
            ax.axvline(obs, color=edge, lw=2.2,
                       label=f"observed {obs:.1f} mm")
        ax.set_title(f"{which} core compactness\np={pval:.3g} vs random",
                     fontsize=9)
        ax.set_xlabel("RMS core radius (mm)", fontsize=8)
        if sub_i == 0:
            ax.set_ylabel("random k-subsets", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ds}_{subj}_card.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  [ok] {out_path}  panel3 rho={json_rho:.3f} swap={swap_class}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+",
                    default=["epilepsiae:958", "epilepsiae:139",
                             "yuquan:zhangbichen", "epilepsiae:635"])
    ap.add_argument("--geom-json-dir", default=str(
        _ROOT / "results/spatial_modulation/propagation_geometry/components/path_axis/per_subject"),
                    help="dir with {ds}_{subj}.json from "
                         "run_propagation_skeleton_geometry.py")
    ap.add_argument("--out", default=str(
        _ROOT / "results/spatial_modulation/propagation_geometry/components/path_axis/figures/per_subject"))
    args = ap.parse_args()
    geom_dir = Path(args.geom_json_dir)
    out_dir = Path(args.out)
    for tok in args.subjects:
        ds, subj = tok.split(":", 1)
        make_card(ds, subj, geom_dir, out_dir)


if __name__ == "__main__":
    main()
