#!/usr/bin/env python3
"""2D 触点平面读出主展示图（per subject）。

展示版只画两件事：触点散点 + 平滑顺序场。support / uncertainty 仍在
record 和 comparison 里使用，但不放进主图，避免读者把辅助诊断图当结果。
同一 subject 的 t_a/t_b 放在同一张 2x2 图里，并使用同一个 3D->2D
display frame，便于横向比较。
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src import propagation_contact_plane_readout as R
from src.seeg_coord_loader import load_subject_coords

BASE = _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout"
VIS_GRID_N = 140
VIS_SIGMA_MULT = 2.5
VIS_SIGMA_MIN_MM = 6.0
VIS_MASK_REL = 0.02


def _coord_array(rec):
    rows = []
    ok = []
    for c in rec.get("channels", []):
        xyz = c.get("coord_mm")
        good = xyz is not None and len(xyz) == 3 and np.isfinite(xyz).all()
        rows.append(xyz if good else [np.nan, np.nan, np.nan])
        ok.append(good)
    return np.asarray(rows, float), np.asarray(ok, bool)


def _estimate_axis_from_record(rec):
    coords, ok = _coord_array(rec)
    if coords.size == 0:
        return None
    along = np.array([c.get("along_axis_mm", np.nan) for c in rec["channels"]], float)
    use = ok & np.isfinite(along)
    if use.sum() < 3:
        return None
    C = coords[use]
    a = along[use]
    Cc = C - C.mean(axis=0)
    ac = a - a.mean()
    beta, *_ = np.linalg.lstsq(Cc, ac, rcond=None)
    n = float(np.linalg.norm(beta))
    if n < 1e-9:
        return None
    u = beta / n
    origin_proj = float(np.median(C @ u - a))
    return u, origin_proj


def _record_sort_key(rec):
    # Prefer t_a as the subject display reference; fallback keeps deterministic order.
    tid = str(rec.get("template_id", ""))
    return (0 if tid == "t_a" else 1, tid)


def _subject_display_frame(records):
    """Build a subject-fixed mm display frame shared by all templates.

    x-axis = reference template early->late axis, in true mm.
    y-axis = first transverse PCA direction of the subject contact cloud, in true mm.
    This is a plotting frame only; comparison metrics stay in normalized readout coords.
    """
    recs = sorted([r for r in records if r.get("channels")], key=_record_sort_key)
    ref = None
    axis = None
    for rec in recs:
        axis = _estimate_axis_from_record(rec)
        if axis is not None:
            ref = rec
            break
    if axis is None:
        return None
    u, origin_proj = axis
    all_coords = []
    for rec in recs:
        coords, ok = _coord_array(rec)
        if coords.size:
            all_coords.append(coords[ok])
    C = np.vstack(all_coords)
    xproj = C @ u
    residual = C - np.outer(xproj, u)
    Rc = residual - residual.mean(axis=0)
    if Rc.shape[0] >= 3 and np.linalg.norm(Rc) > 1e-9:
        _, _, vt = np.linalg.svd(Rc, full_matrices=False)
        v = vt[0]
        v = v - np.dot(v, u) * u
        v = v / max(np.linalg.norm(v), 1e-12)
    else:
        # Deterministic fallback: any unit vector orthogonal to u.
        base = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(base, u))) > 0.8:
            base = np.array([0.0, 1.0, 0.0])
        v = base - np.dot(base, u) * u
        v = v / max(np.linalg.norm(v), 1e-12)

    # Sign y to agree with the reference template's signed transverse coordinate.
    coords_ref, ok_ref = _coord_array(ref)
    ref_y = np.array([c.get("signed_transverse_mm", np.nan) for c in ref["channels"]], float)
    y_tmp = coords_ref @ v
    use = ok_ref & np.isfinite(ref_y) & np.isfinite(y_tmp)
    if use.sum() >= 3 and np.corrcoef(y_tmp[use], ref_y[use])[0, 1] < 0:
        v = -v

    y_center = float(np.median(C @ v))
    xs, ys = [], []
    for rec in recs:
        x, y = _display_points(rec, {"u": u, "origin_proj": origin_proj,
                                     "v": v, "y_center": y_center})
        xs.extend(x[np.isfinite(x)])
        ys.extend(y[np.isfinite(y)])
    xlim = _limits_with_padding(np.asarray(xs), include_zero=True, min_span=35.0)
    ylim = _limits_with_padding(np.asarray(ys), include_zero=True, min_span=35.0)
    pts = np.column_stack([xs, ys]) if xs and ys else np.zeros((0, 2))
    sigma = _median_nn_spacing(pts) * VIS_SIGMA_MULT if pts.shape[0] >= 2 else 10.0
    return {"u": u, "origin_proj": origin_proj, "v": v, "y_center": y_center,
            "xlim": xlim, "ylim": ylim, "sigma_mm": float(max(sigma, VIS_SIGMA_MIN_MM))}


def _limits_with_padding(vals, include_zero=False, min_span=30.0):
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        lo, hi = -min_span / 2.0, min_span / 2.0
    else:
        lo, hi = float(vals.min()), float(vals.max())
        if include_zero:
            lo, hi = min(lo, 0.0), max(hi, 0.0)
        span = max(hi - lo, min_span)
        mid = (lo + hi) / 2.0
        lo, hi = mid - span / 2.0, mid + span / 2.0
    pad = 0.12 * max(hi - lo, 1.0)
    return lo - pad, hi + pad


def _display_points(rec, frame):
    chans = rec.get("channels", [])
    coords, ok = _coord_array(rec)
    if frame is not None and coords.size and ok.any():
        x = coords @ frame["u"] - frame["origin_proj"]
        y = coords @ frame["v"] - frame["y_center"]
        x[~ok] = np.nan
        y[~ok] = np.nan
        return x, y
    scale = float(rec.get("norm_scale_mm") or rec.get("axis_length_mm") or 1.0)
    x = np.array([c.get("x_norm", np.nan) * scale for c in chans], float)
    y = np.array([c.get("y_norm", np.nan) * scale for c in chans], float)
    return x, y


def _median_nn_spacing(pts):
    if pts.shape[0] < 2:
        return 10.0
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.nanmedian(d.min(axis=1)))


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


def _add_electrode_labels(ax, x, y, names):
    xr = ax.get_xlim()[1] - ax.get_xlim()[0]
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    dx = 0.010 * xr
    dy = 0.010 * yr
    for xi, yi, nm in zip(x, y, names):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.text(xi + dx, yi + dy, str(nm), fontsize=6.5, color="0.15",
                    ha="left", va="bottom")


def _record_display_payload(rec, display_frame=None, sigma_mult=VIS_SIGMA_MULT):
    chans = rec["channels"]
    xs, ys = _display_points(rec, display_frame)
    rk = np.array([c["typical_rank"] for c in chans], float)
    sp = np.array([c["support"] for c in chans], float)
    soz = np.array([bool(c.get("is_soz")) for c in chans])
    if display_frame is not None:
        xlim, ylim = display_frame["xlim"], display_frame["ylim"]
        sigma_mm = display_frame["sigma_mm"] * (sigma_mult / VIS_SIGMA_MULT)
    else:
        xlim = _limits_with_padding(xs, include_zero=True, min_span=35.0)
        ylim = _limits_with_padding(ys, include_zero=True, min_span=35.0)
        sigma_mm = max(_median_nn_spacing(np.column_stack([xs, ys])) * sigma_mult,
                       VIS_SIGMA_MIN_MM)
    X, Y, T, _, _ = _smooth_rank_field_mm(xs, ys, rk, sp, xlim, ylim, sigma_mm)
    return {
        "names": [c["name"] for c in chans],
        "xs": xs,
        "ys": ys,
        "rank": rk,
        "support": sp,
        "soz": soz,
        "xlim": xlim,
        "ylim": ylim,
        "sigma_mm": sigma_mm,
        "field": T,
    }


def _record_title_bits(rec):
    flags = rec.get("flags", {})
    scalars = rec.get("scalars", {})
    flag_txt = "1D" if flags.get("one_dimensional_sampling") else "2D"
    rho = scalars.get("rank_vs_xnorm_spearman", float("nan"))
    oof = rec.get("out_of_field", {}).get("count", 0)
    weak = " | WEAK rank-axis" if (np.isfinite(rho) and abs(rho) < 0.3) else ""
    return flag_txt, rho, oof, weak


def _draw_record_panels(ax_contact, ax_field, rec, display_frame=None,
                        sigma_mult=VIS_SIGMA_MULT, show_xlabel=True):
    payload = _record_display_payload(rec, display_frame, sigma_mult)
    xs = payload["xs"]
    ys = payload["ys"]
    rk = payload["rank"]
    sp = payload["support"]
    soz = payload["soz"]
    xlim = payload["xlim"]
    ylim = payload["ylim"]
    tid = rec.get("template_id", "")
    flag_txt, rho, oof, weak = _record_title_bits(rec)

    sc = ax_contact.scatter(xs, ys, c=rk, s=45 + 220 * sp, cmap="viridis",
                            vmin=0, vmax=1,
                            edgecolors=["k" if z else "white" for z in soz],
                            linewidths=[1.8 if z else 0.4 for z in soz], zorder=3)
    _add_electrode_labels(ax_contact, xs, ys, payload["names"])
    ax_contact.set_title(
        f"{tid} contacts | {flag_txt} | rho_x_rank={rho:.2f} | "
        f"comparison-grid out_of_field={oof}{weak}",
        fontsize=10,
    )
    ax_contact.set_ylabel("subject-fixed display y (mm)")
    if show_xlabel:
        ax_contact.set_xlabel("subject-fixed display x (mm)")
    else:
        ax_contact.set_xlabel("")
        ax_contact.tick_params(labelbottom=False)
    ax_contact.set_xlim(*xlim)
    ax_contact.set_ylim(*ylim)
    ax_contact.set_aspect("equal", adjustable="box")
    ax_contact.axvline(0, color="0.82", lw=0.8, zorder=0)
    ax_contact.axhline(0, color="0.90", lw=0.8, zorder=0)

    im = ax_field.imshow(payload["field"], origin="lower",
                         extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                         aspect="equal", cmap="viridis", vmin=0, vmax=1)
    ax_field.scatter(xs, ys, c="none", edgecolors="0.35", s=16, linewidths=0.5)
    ax_field.set_title(
        f"{tid} smoothed order field (Gaussian sigma={payload['sigma_mm']:.1f} mm)",
        fontsize=10,
    )
    ax_field.set_ylabel("subject-fixed display y (mm)")
    if show_xlabel:
        ax_field.set_xlabel("subject-fixed display x (mm)")
    else:
        ax_field.set_xlabel("")
        ax_field.tick_params(labelbottom=False)
    ax_field.set_xlim(*xlim)
    ax_field.set_ylim(*ylim)
    ax_field.set_aspect("equal", adjustable="box")
    return sc, im


def plot_record(rec, out_png, display_frame=None, sigma_mult=VIS_SIGMA_MULT):
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.8), constrained_layout=True)
    sc, im = _draw_record_panels(ax[0], ax[1], rec, display_frame=display_frame,
                                 sigma_mult=sigma_mult)
    plt.colorbar(sc, ax=ax[0], label="typical order (0=early, 1=late)")
    plt.colorbar(im, ax=ax[1], label="smoothed typical order")
    flag_txt, rho, oof, weak = _record_title_bits(rec)
    amb = rec.get("soz_ambiguous", [])
    fig.suptitle(
        f"{rec['dataset']}:{rec['subject']} {rec['template_id']} | {flag_txt} | "
        f"rho_x_rank={rho:.2f} | comparison-grid out_of_field={oof}{weak} | "
        "SOZ overlay only, not metric input"
        + (f" | SOZ ambiguous: {amb}" if amb else ""))
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_subject_records(records, out_png, display_frame=None, sigma_mult=VIS_SIGMA_MULT):
    recs = sorted([r for r in records if r.get("channels")], key=_record_sort_key)
    if not recs:
        return
    n_rows = len(recs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14.5, 5.2 * n_rows),
                             constrained_layout=True, squeeze=False)
    sc = im = None
    for row, rec in enumerate(recs):
        sc, im = _draw_record_panels(
            axes[row, 0], axes[row, 1], rec,
            display_frame=display_frame,
            sigma_mult=sigma_mult,
            show_xlabel=(row == n_rows - 1),
        )
    if sc is not None:
        fig.colorbar(sc, ax=axes[:, 0], label="typical order (0=early, 1=late)",
                     fraction=0.030, pad=0.018)
    if im is not None:
        fig.colorbar(im, ax=axes[:, 1], label="smoothed typical order",
                     fraction=0.030, pad=0.018)
    amb = sorted({str(a) for rec in recs for a in rec.get("soz_ambiguous", [])})
    fig.suptitle(
        f"{recs[0]['dataset']}:{recs[0]['subject']} | t_a top, t_b bottom | "
        "SOZ overlay only, not metric input"
        + (f" | SOZ ambiguous: {amb}" if amb else ""),
        fontsize=12,
    )
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def _load_records(record_dir):
    recs = []
    for f in sorted(Path(record_dir).glob("*.json")):
        rec = json.loads(f.read_text())
        if rec.get("channels"):
            recs.append((f, rec))
    return recs


def _attach_real_coords(records):
    """Attach coord_mm to real records at plot time when old records lack it."""
    if not records:
        return records
    ds = records[0].get("dataset")
    subj = records[0].get("subject")
    if ds not in {"yuquan", "epilepsiae"}:
        return records
    names = []
    for rec in records:
        for c in rec.get("channels", []):
            if c.get("name") not in names:
                names.append(c.get("name"))
    try:
        cr = load_subject_coords(ds, subj, names)
    except Exception:
        return records
    coord_map = {}
    for nm, xyz, mapped in zip(names, cr.coords_array_in_requested_order,
                               cr.mapped_mask_in_requested_order):
        if mapped and np.isfinite(xyz).all():
            coord_map[str(nm)] = [float(v) for v in xyz]
    for rec in records:
        for c in rec.get("channels", []):
            if c.get("coord_mm") is None and c.get("name") in coord_map:
                c["coord_mm"] = coord_map[c["name"]]
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", default=str(BASE / "real_subjects"))
    ap.add_argument("--out", default=str(BASE / "figures/static_maps"))
    ap.add_argument("--sigma-mult", type=float, default=VIS_SIGMA_MULT,
                    help="visual Gaussian sigma multiplier over median nearest-neighbor spacing")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = _load_records(args.real_dir)
    groups = defaultdict(list)
    for _, rec in recs:
        groups[(rec.get("dataset"), rec.get("subject"))].append(rec)
    for key, group in list(groups.items()):
        groups[key] = _attach_real_coords(group)
    frames = {k: _subject_display_frame(v) for k, v in groups.items()}
    for stale in out.glob("*_t_*.png"):
        stale.unlink()
    for key, group in sorted(groups.items()):
        ds, subj = key
        frame = frames.get(key)
        out_png = out / f"{ds}_{subj}.png"
        plot_subject_records(group, out_png, display_frame=frame,
                             sigma_mult=args.sigma_mult)
        print(f"  {out_png.name}")


if __name__ == "__main__":
    main()
