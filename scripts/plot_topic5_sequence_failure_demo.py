"""方法学示意图（didactic，非结果图）：真实电极布局下，逐电极序列推方向失败。

这张图只回答一个展示问题：同一批真实触点、同一个 2D display frame，
如果把一次 noisy contact-level readout 压缩成"最早触点 -> 最晚触点"，
少数触点 outlier 就能把方向箭头带反；而 field / full-pattern readout
使用同一批触点的空间汇集，更适合承载粗空间形状。

默认使用 epilepsiae_1146 narrow t_a 的真实布局。观测值是 didactic
构造，不是该 subject 的真实发作统计。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.plot_contact_plane_static import (  # noqa: E402
    _attach_real_coords,
    _display_points,
    _smooth_rank_field_mm,
    _subject_display_frame,
)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.style"] = "normal"

OUT_DIR = _ROOT / "results/topic5_ictal_recruitment/field_dynamics/figures/methods"
GEO_DIR = {
    "narrow": _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects",
    "broad": _ROOT / "results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects",
}


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-12)


def load_real_layout(subject: str, substrate: str, template: str) -> dict:
    """Load real contact layout and template values in the canonical display frame."""
    root = GEO_DIR[substrate]
    ta = json.loads((root / f"{subject}_t_a.json").read_text())
    tb = json.loads((root / f"{subject}_t_b.json").read_text())
    records = [ta, tb]
    _attach_real_coords(records)
    frame = _subject_display_frame(records)
    if frame is None:
        raise RuntimeError(f"could not build display frame for {subject} {substrate}")
    rec = ta if template == "t_a" else tb
    x, y = _display_points(rec, frame)
    values = np.asarray([c.get("typical_rank", np.nan) for c in rec["channels"]], dtype=float)
    support = np.asarray([c.get("support", 1.0) for c in rec["channels"]], dtype=float)
    names = [str(c.get("name", i)) for i, c in enumerate(rec["channels"])]
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    return {
        "record": rec,
        "frame": frame,
        "x": np.asarray(x[ok], dtype=float),
        "y": np.asarray(y[ok], dtype=float),
        "values": _normalize(values[ok]),
        "support": np.clip(np.asarray(support[ok], dtype=float), 0.05, None),
        "names": [name for name, keep in zip(names, ok) if keep],
    }


def fit_gradient(P: np.ndarray, values: np.ndarray) -> np.ndarray:
    A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
    coeff, *_ = np.linalg.lstsq(A, values, rcond=None)
    return np.asarray(coeff[:2], dtype=float)


def angle_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def field_grid(x: np.ndarray, y: np.ndarray, values: np.ndarray, support: np.ndarray, frame: dict) -> np.ndarray:
    _, _, field, _, _ = _smooth_rank_field_mm(
        x,
        y,
        values,
        support,
        frame["xlim"],
        frame["ylim"],
        float(frame["sigma_mm"]),
    )
    return field


def smooth_at_contacts(P: np.ndarray, values: np.ndarray, support: np.ndarray, sigma_mm: float) -> np.ndarray:
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
    W = np.exp(-d2 / (2.0 * sigma_mm**2)) * support[None, :]
    W /= W.sum(axis=1, keepdims=True)
    return W @ values


def draw_axis_arrow(
    ax: plt.Axes,
    vec: np.ndarray,
    label: str,
    color: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    y_frac: float = 0.15,
    scale_frac: float = 0.26,
) -> None:
    span_x = xlim[1] - xlim[0]
    span_y = ylim[1] - ylim[0]
    start = np.array([xlim[0] + 0.10 * span_x, ylim[0] + y_frac * span_y])
    v = np.asarray(vec, dtype=float)
    v = v / (np.linalg.norm(v) + 1e-12) * (scale_frac * span_x)
    ax.annotate(
        "",
        xy=start + v,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=2.6, color=color),
        zorder=12,
    )
    ax.text(
        *(start + v * 1.08),
        label,
        fontsize=9.8,
        color=color,
        fontweight="bold",
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
        zorder=13,
    )


def draw_sequence_arrow(ax: plt.Axes, start: np.ndarray, end: np.ndarray) -> None:
    arrow = FancyArrowPatch(
        posA=start,
        posB=end,
        arrowstyle="-|>",
        mutation_scale=18,
        lw=3.0,
        color="crimson",
        connectionstyle="arc3,rad=-0.18",
        zorder=14,
    )
    ax.add_patch(arrow)
    mid = (start + end) / 2.0
    ax.text(
        mid[0],
        mid[1],
        "序列箭头",
        color="crimson",
        fontsize=9.8,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78),
        zorder=15,
    )


def annotate_outliers(ax: plt.Axes, P: np.ndarray, names: list[str], early_idx: int, late_idx: int) -> None:
    ax.scatter(
        P[[early_idx, late_idx], 0],
        P[[early_idx, late_idx], 1],
        s=180,
        facecolors="none",
        edgecolors="crimson",
        linewidths=2.0,
        zorder=13,
    )
    labels = [(early_idx, f"{names[early_idx]}\n误早"), (late_idx, f"{names[late_idx]}\n误晚")]
    for idx, label in labels:
        ax.text(
            P[idx, 0],
            P[idx, 1] + 2.6,
            label,
            fontsize=8.4,
            color="crimson",
            fontweight="bold",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.78),
            zorder=15,
        )


def style_plane(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.28")
        spine.set_linewidth(1.0)


def draw_contacts(
    ax: plt.Axes,
    P: np.ndarray,
    values: np.ndarray,
    sizes: np.ndarray,
    edge: str = "white",
    alpha: float = 1.0,
) -> None:
    ax.scatter(
        P[:, 0],
        P[:, 1],
        c=values,
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=sizes,
        edgecolors=edge,
        linewidths=0.7,
        alpha=alpha,
        zorder=7,
    )


def make_demo_values(P: np.ndarray, true_values: np.ndarray, true_vec: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Make a deterministic noisy contact readout that reverses the min->max sequence."""
    rng = np.random.default_rng(5)
    observed = np.clip(true_values + rng.normal(0.0, 0.055, len(true_values)), 0.0, 1.0)
    proj = P @ (true_vec / (np.linalg.norm(true_vec) + 1e-12))
    spuriously_early = int(np.nanargmax(proj))
    spuriously_late = int(np.nanargmin(proj))
    observed[spuriously_early] = 0.0
    observed[spuriously_late] = 1.0
    return observed, spuriously_early, spuriously_late


def plot_demo(args: argparse.Namespace) -> Path:
    layout = load_real_layout(args.subject, args.substrate, args.template)
    x = layout["x"]
    y = layout["y"]
    P = np.column_stack([x, y])
    true_values = layout["values"]
    support = layout["support"]
    names = layout["names"]
    frame = layout["frame"]
    xlim = tuple(frame["xlim"])
    ylim = tuple(frame["ylim"])

    true_vec = fit_gradient(P, true_values)
    observed, early_idx, late_idx = make_demo_values(P, true_values, true_vec)
    smoothed_contacts = _normalize(smooth_at_contacts(P, observed, support, float(frame["sigma_mm"])))
    field_vec = fit_gradient(P, smoothed_contacts)

    rank_order = np.argsort(observed)
    seq_start = P[rank_order[0]]
    seq_end = P[rank_order[-1]]
    seq_vec = seq_end - seq_start

    sequence_error = angle_error_deg(true_vec, seq_vec)
    field_error = angle_error_deg(true_vec, field_vec)
    field_corr = float(np.corrcoef(smoothed_contacts, true_values)[0, 1])
    print(
        f"{args.subject} {args.substrate} {args.template}: "
        f"sequence_error={sequence_error:.1f} deg  "
        f"field_error={field_error:.1f} deg  field_corr={field_corr:.2f}  "
        f"outliers={names[early_idx]},{names[late_idx]}"
    )

    sizes = 42 + 74 * _normalize(support)
    true_field = field_grid(x, y, true_values, support, frame)
    observed_field = field_grid(x, y, observed, support, frame)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), constrained_layout=True)

    im = axes[0].imshow(
        true_field,
        origin="lower",
        extent=[*xlim, *ylim],
        aspect="equal",
        cmap="viridis",
        vmin=0,
        vmax=1,
        zorder=0,
    )
    draw_contacts(axes[0], P, true_values, sizes)
    draw_axis_arrow(axes[0], true_vec, "TA 粗轴", "black", xlim, ylim, y_frac=0.13, scale_frac=0.22)
    axes[0].set_title("① 真实电极布局上的 TA field", fontsize=12.5)
    style_plane(axes[0], xlim, ylim)

    draw_contacts(axes[1], P, observed, sizes)
    annotate_outliers(axes[1], P, names, early_idx, late_idx)
    draw_sequence_arrow(axes[1], seq_start, seq_end)
    draw_axis_arrow(axes[1], true_vec, "TA 粗轴", "black", xlim, ylim, y_frac=0.13, scale_frac=0.18)
    axes[1].text(
        0.50,
        0.045,
        f"最早→最晚被带反   误差 {sequence_error:.0f}°",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=9.8,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="crimson", lw=1.0),
        zorder=16,
    )
    axes[1].set_title("② 逐电极序列方向：失败", fontsize=12.5)
    style_plane(axes[1], xlim, ylim)

    axes[2].imshow(
        observed_field,
        origin="lower",
        extent=[*xlim, *ylim],
        aspect="equal",
        cmap="viridis",
        vmin=0,
        vmax=1,
        zorder=0,
    )
    axes[2].scatter(P[:, 0], P[:, 1], c="none", edgecolors="0.34", s=26, linewidths=0.7, zorder=7)
    annotate_outliers(axes[2], P, names, early_idx, late_idx)
    draw_axis_arrow(axes[2], field_vec, "field 粗轴", "#1f77b4", xlim, ylim, y_frac=0.13, scale_frac=0.22)
    axes[2].text(
        0.50,
        0.045,
        f"同一触点空间汇集   r={field_corr:.2f}, 误差 {field_error:.0f}°",
        transform=axes[2].transAxes,
        ha="center",
        va="bottom",
        fontsize=9.8,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#1f77b4", lw=1.0),
        zorder=16,
    )
    axes[2].set_title("③ field / full-pattern：粗结构更稳", fontsize=12.5)
    style_plane(axes[2], xlim, ylim)

    cbar = fig.colorbar(im, ax=axes, fraction=0.024, pad=0.012)
    cbar.set_label("order / activation  (0=早, 1=晚)")
    fig.suptitle(f"{args.subject} 真实布局示意：逐电极序列方向失败，field 保留粗空间形状（非发作统计）", fontsize=12.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "sequence_direction_failure_field_motivation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="epilepsiae_1146")
    parser.add_argument("--substrate", choices=sorted(GEO_DIR), default="narrow")
    parser.add_argument("--template", choices=("t_a", "t_b"), default="t_a")
    args = parser.parse_args()
    out = plot_demo(args)
    print("saved", out)


if __name__ == "__main__":
    main()
