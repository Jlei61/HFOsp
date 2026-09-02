"""The one load-bearing figure of Agent A, and its companions.

SP 7 asks for a single figure: held-out future-block score relative to the
multiscale baseline, against real physical horizon, split into count and
conditional mark, showing the local model, the multi-horizon model, the correct
time and the time-shifted state, with the same-prefix continuation alongside.

Panel discipline (CLAUDE.md 7): the conditional mark is *two* independent
questions -- which contacts take part, and what the events look like -- carried in
different units, so they get one panel each rather than being averaged into a
single "mark" number that means nothing.

Style (figure_style_guide 0): no internal identifiers on the canvas, one shared
legend, tight axes, a signed quantity gets a zero reference line.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# Plain-language names.  Nothing on the canvas may say ``P_slow`` or ``B_multiscale``.
ARM_LABELS = {
    "P_local": "next-event model",
    "P_slow": "multi-horizon model",
    "shift": "same model, time-shifted state",
    "memoryless": "carries only the gap since the last event",
}
ARM_COLORS = {
    "P_local": "#1b7f79",     # teal
    "P_slow": "#c1571c",      # burnt orange
    "shift": "#8a8a8a",       # grey
    "memoryless": "#5b5ea6",  # muted indigo
}

# A handful of patients sit orders of magnitude away from the rest, and letting
# them set the y range hides the very comparison the figure exists for.  The axis
# is clipped to this central span of the pooled per-patient values and the number
# of points left outside is printed on the panel.
SCATTER_CLIP_PERCENTILES = (2.0, 98.0)

PANEL_TITLES = {
    "count": "How many events arrive",
    "participation": "Which contacts take part",
    "continuous": "What the events look like",
}
PANEL_YLABELS = {
    "count": "gain over baseline\n(nats per window)",
    "participation": "gain over baseline\n(nats per event x contact)",
    "continuous": "gain over baseline\n(nats per event x dimension)",
}

HORIZON_LABELS = {300.0: "5 min", 1800.0: "30 min", 7200.0: "2 h"}

PREFIX_LABELS = {
    "continues": "did it keep\nspreading",
    "later_participation": "which contacts\nit reached",
    "extent": "how far it got\nbefore stopping",
    "later_multiband": "later spectral\nexpression",
}


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "savefig.bbox": "tight",
    })


def _robust_limits(values: np.ndarray, medians, los, his) -> tuple[float, float, int]:
    """Axis span that shows the medians and the bulk of the scatter, not the tails."""

    pool = np.asarray(values, dtype=float)
    pool = pool[np.isfinite(pool)]
    anchors = [v for v in list(medians) + list(los) + list(his) if np.isfinite(v)]
    if pool.size == 0:
        return (-1.0, 1.0, 0)
    lo = float(np.percentile(pool, SCATTER_CLIP_PERCENTILES[0]))
    hi = float(np.percentile(pool, SCATTER_CLIP_PERCENTILES[1]))
    if anchors:
        lo = min(lo, min(anchors))
        hi = max(hi, max(anchors))
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)
    pad = 0.08 * max(hi - lo, 1e-9)
    lo, hi = lo - pad, hi + pad
    return lo, hi, int(((pool < lo) | (pool > hi)).sum())


def _bootstrap_ci(values: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True)
    med = np.median(draws, axis=1)
    return float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


@dataclass(frozen=True)
class CurvePoint:
    horizon: float
    values: np.ndarray
    n_positive: int
    p_sign: float
    n_independent_windows_median: float


def collect_curve(
    results: Sequence[Mapping[str, Any]],
    arm_pattern: str,
    endpoint: str,
    *,
    reference_arm: str = "B_multiscale",
) -> list[CurvePoint]:
    """Per-horizon per-patient gain of every arm whose name matches a pattern.

    Several seeds of the same producer are collapsed to that patient's median
    first, so a patient with three fits still counts once (seeds are repeated
    fits, not samples).
    """

    from .aggregate import sign_test_p

    horizons = sorted({float(h[:-1]) for r in results for h in r["horizons"]})
    out: list[CurvePoint] = []
    for horizon in horizons:
        key = f"{int(horizon)}s"
        values: list[float] = []
        indep: list[float] = []
        for r in results:
            entry = r["horizons"].get(key)
            if entry is None or entry.get("status") != "ok":
                continue
            ref = entry["arms"].get(reference_arm)
            if ref is None:
                continue
            if ref.get("estimability", {}).get(endpoint, "ok") != "ok":
                continue
            base = ref["scores"].get(endpoint)
            per_seed = []
            for name, payload in entry["arms"].items():
                if arm_pattern not in name:
                    continue
                score = payload["scores"].get(endpoint)
                flag = payload.get("estimability", {}).get(endpoint, "ok")
                if score is None or base is None or flag != "ok":
                    continue
                local_ref = entry["arms"].get(
                    f"{reference_arm}|subset({name.split('(')[-1].rstrip(')')})"
                )
                b = (local_ref or ref)["scores"].get(endpoint, base)
                per_seed.append(b["nll_per_unit"] - score["nll_per_unit"])
            if per_seed:
                values.append(float(np.median(per_seed)))
                indep.append(entry["denominators"]["test"]["n_independent_windows"])
        v = np.array(values, dtype=float)
        out.append(CurvePoint(
            horizon=horizon, values=v,
            n_positive=int((v > 0).sum()),
            p_sign=sign_test_p(int((v > 0).sum()), int(v.size)),
            n_independent_windows_median=float(np.median(indep)) if indep else float("nan"),
        ))
    return out


def plot_future_block_figure(
    results: Sequence[Mapping[str, Any]],
    prefix_results: Sequence[Mapping[str, Any]] | None,
    out_png: Path,
    out_pdf: Path,
    *,
    arms: Mapping[str, str],
    seed: int = 0,
) -> dict[str, Any]:
    """Render the load-bearing figure and return its machine-readable payload."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    endpoints = ("count", "participation", "continuous")
    n_prefix = 1 if prefix_results else 0
    fig, axes = plt.subplots(
        1, 3 + n_prefix, figsize=(3.3 * (3 + n_prefix), 3.3),
        gridspec_kw={"width_ratios": [1, 1, 1] + ([1.25] * n_prefix)},
    )
    payload: dict[str, Any] = {"panels": {}}

    rng = np.random.default_rng(seed)
    for ax, endpoint in zip(axes[:3], endpoints):
        curves: dict[str, list[CurvePoint]] = {
            key: collect_curve(results, pattern, endpoint)
            for key, pattern in arms.items()
        }
        horizons = [c.horizon for c in next(iter(curves.values()))]
        xs = np.arange(len(horizons), dtype=float)
        ax.axhline(0.0, color="#333333", lw=0.9, zorder=1)
        for offset, (key, points) in enumerate(curves.items()):
            colour = ARM_COLORS.get(key, "#444444")
            med = [float(np.median(p.values)) if p.values.size else np.nan for p in points]
            lo, hi = zip(*[_bootstrap_ci(p.values, seed=seed) for p in points])
            dx = (offset - (len(curves) - 1) / 2) * 0.12
            for x, p in zip(xs + dx, points):
                if p.values.size:
                    jitter = rng.uniform(-0.035, 0.035, p.values.size)
                    ax.scatter(x + jitter, p.values, s=4, color=colour, alpha=0.25,
                               linewidths=0, zorder=2)
            ax.errorbar(xs + dx, med, yerr=[np.array(med) - np.array(lo),
                                            np.array(hi) - np.array(med)],
                        color=colour, marker="o", ms=4, capsize=2, lw=1.6, zorder=3,
                        label=ARM_LABELS.get(key, key))
            payload["panels"].setdefault(endpoint, {})[key] = [
                {"horizon_seconds": p.horizon, "median_gain": m,
                 "ci95": [l, h], "n_subjects": int(p.values.size),
                 "n_positive": p.n_positive, "p_sign": p.p_sign,
                 "median_independent_windows": p.n_independent_windows_median}
                for p, m, l, h in zip(points, med, lo, hi)
            ]
        pooled = np.concatenate(
            [p.values for pts in curves.values() for p in pts if p.values.size]
            or [np.zeros(1)]
        )
        all_med = [m for k in curves for m in
                   [float(np.median(p.values)) if p.values.size else np.nan
                    for p in curves[k]]]
        all_ci = [b for k in curves for p in curves[k] for b in _bootstrap_ci(p.values)]
        ylo, yhi, n_out = _robust_limits(pooled, all_med, all_ci, all_ci)
        ax.set_ylim(ylo, yhi)
        if n_out:
            ax.text(0.98, 0.02, f"{n_out} patient points outside", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=6, color="#666666")
        ticks = []
        for i, h in enumerate(horizons):
            ref = next(iter(curves.values()))[i]
            # The independent-window count belongs on the canvas: a 2 h window
            # stepped every 5 min gives 24 overlapping anchors per genuinely
            # independent one, and a reader who sees only "n patients" will
            # silently assume the anchors were the sample.
            ticks.append(
                f"{HORIZON_LABELS.get(h, f'{h / 60:.0f} min')}\n{ref.values.size} patients"
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks, fontsize=7.5, linespacing=1.3)
        ax.set_xlim(xs[0] - 0.45, xs[-1] + 0.45)
        ax.set_title(PANEL_TITLES[endpoint])
        ax.set_ylabel(PANEL_YLABELS[endpoint])
        # The independent-window count has to be visible: a 2 h window stepped
        # every 5 min gives 24 overlapping anchors for each genuinely independent
        # one, and a reader who sees only the patient count will assume the
        # anchors were the sample.
        indep = " / ".join(
            f"{next(iter(curves.values()))[i].n_independent_windows_median:.0f}"
            for i in range(len(horizons))
        )
        ax.set_xlabel(
            f"how far ahead the block starts\nnon-overlapping windows: {indep}",
            fontsize=7.5,
        )

    if prefix_results:
        ax = axes[3]
        payload["panels"]["same_prefix"] = _plot_prefix_panel(
            ax, prefix_results, arms, rng
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        "Does interictal history predict the next stretch of recording, "
        "beyond a multiscale baseline?", y=1.04, fontsize=10,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    return payload


def _plot_prefix_panel(ax, prefix_results, arms, rng) -> dict[str, Any]:
    """Same-prefix continuation: gain over the prefix-only model, per outcome."""

    from .aggregate import sign_test_p

    outcomes = list(PREFIX_LABELS)
    xs = np.arange(len(outcomes), dtype=float)
    ax.axhline(0.0, color="#333333", lw=0.9, zorder=1)
    out: dict[str, Any] = {}
    for offset, (key, pattern) in enumerate(arms.items()):
        colour = ARM_COLORS.get(key, "#444444")
        med, lo, hi, sizes = [], [], [], []
        for outcome in outcomes:
            values = []
            for r in prefix_results:
                base = r["arms"].get("prefix", {}).get(outcome)
                per_seed = [
                    base["nll_per_unit"] - payload[outcome]["nll_per_unit"]
                    for name, payload in r["arms"].items()
                    if pattern in name and outcome in payload and base is not None
                ]
                if per_seed:
                    values.append(float(np.median(per_seed)))
            v = np.array(values, dtype=float)
            med.append(float(np.median(v)) if v.size else np.nan)
            l, h = _bootstrap_ci(v)
            lo.append(l)
            hi.append(h)
            sizes.append(v)
            out.setdefault(key, {})[outcome] = {
                "median_gain": med[-1], "ci95": [l, h], "n_subjects": int(v.size),
                "n_positive": int((v > 0).sum()),
                "p_sign": sign_test_p(int((v > 0).sum()), int(v.size)),
            }
        dx = (offset - (len(arms) - 1) / 2) * 0.14
        for x, v in zip(xs + dx, sizes):
            if v.size:
                ax.scatter(x + rng.uniform(-0.04, 0.04, v.size), v, s=4, color=colour,
                           alpha=0.25, linewidths=0, zorder=2)
        ax.errorbar(xs + dx, med, yerr=[np.array(med) - np.array(lo),
                                        np.array(hi) - np.array(med)],
                    color=colour, marker="o", ms=4, capsize=2, lw=1.6, zorder=3)
    pooled = np.concatenate([v for v in sizes if v.size] or [np.zeros(1)])
    ylo, yhi, n_out = _robust_limits(pooled, med, lo, hi)
    ax.set_ylim(ylo, yhi)
    if n_out:
        ax.text(0.98, 0.02, f"{n_out} patient points outside", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6, color="#666666")
    ax.set_xticks(xs)
    ax.set_xticklabels([PREFIX_LABELS[o] for o in outcomes], fontsize=6.5,
                       linespacing=1.35)
    ax.set_xlim(xs[0] - 0.45, xs[-1] + 0.45)
    ax.set_title("Events that started the same way")
    ax.set_ylabel("gain over the start-only model\n(nats per unit)")
    ax.set_xlabel("what happened next")
    return out


# Only the conditional-mark blocks: every one of these is a per-observation
# log-loss, so they share a y axis.  The event count is nats *per window* and
# belongs on its own panel of the load-bearing figure -- putting it here would
# plot two different units against one scale.
BLOCK_LABELS = {
    "participation": "which contacts",
    "continuous:size": "how many contacts",
    "continuous:span": "recruitment duration",
    "continuous:band_energy": "band energy",
    "continuous:band_peak": "band timing",
    "continuous:embedding": "overall event shape",
}
BLOCK_ORDER = (
    "participation", "continuous:size", "continuous:span",
    "continuous:band_energy", "continuous:band_peak", "continuous:embedding",
)


def plot_mark_block_figure(
    results: Sequence[Mapping[str, Any]],
    out_png: Path,
    out_pdf: Path,
    *,
    arms: Mapping[str, str],
    horizons: Sequence[float] = (300.0, 1800.0, 7200.0),
) -> dict[str, Any]:
    """Auxiliary: *which kind* of state, if any (SP 8's rate / extent / repertoire).

    The load-bearing figure pools the continuous mark into one number.  The
    allowed conclusions distinguish a rate state from an extent state from a
    repertoire state, and that distinction lives exactly in these blocks, so it
    gets its own figure rather than a sentence asserting it.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    fig, axes = plt.subplots(1, len(horizons), figsize=(3.5 * len(horizons), 3.4),
                             sharey=False)
    axes = np.atleast_1d(axes)
    payload: dict[str, Any] = {}
    xs = np.arange(len(BLOCK_ORDER), dtype=float)
    for ax, horizon in zip(axes, horizons):
        ax.axhline(0.0, color="#333333", lw=0.9, zorder=1)
        for offset, (key, pattern) in enumerate(arms.items()):
            colour = ARM_COLORS.get(key, "#444444")
            med, lo, hi = [], [], []
            for block in BLOCK_ORDER:
                points = [p for p in collect_curve(results, pattern, block)
                          if p.horizon == horizon]
                v = points[0].values if points else np.zeros(0)
                med.append(float(np.median(v)) if v.size else np.nan)
                l, h = _bootstrap_ci(v)
                lo.append(l)
                hi.append(h)
                payload.setdefault(f"{int(horizon)}s", {}).setdefault(key, {})[block] = {
                    "median_gain": med[-1], "ci95": [l, h],
                    "n_subjects": int(v.size),
                    "n_positive": int((v > 0).sum()) if v.size else 0,
                    "p_sign": points[0].p_sign if points else float("nan"),
                }
            dx = (offset - (len(arms) - 1) / 2) * 0.16
            ax.errorbar(xs + dx, med, yerr=[np.array(med) - np.array(lo),
                                            np.array(hi) - np.array(med)],
                        color=colour, marker="o", ms=4, capsize=2, lw=0, elinewidth=1.4,
                        label=ARM_LABELS.get(key, key))
        ax.set_xticks(xs)
        ax.set_xticklabels([BLOCK_LABELS[b] for b in BLOCK_ORDER], fontsize=7,
                           rotation=30, ha="right")
        ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
        ax.set_title(f"{HORIZON_LABELS.get(horizon, f'{horizon / 60:.0f} min')} ahead")
        if ax is axes[0]:
            ax.set_ylabel("gain over baseline\n(nats per scored observation)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(
        "Given that events arrive, which part of what they look like does the "
        "state help with?", y=1.02, fontsize=10,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    return payload


RESET_ORDER = (
    ("reset_e1", "1 event"),
    ("reset_e100", "100 events"),
    ("reset_e1000", "1000 events"),
    ("reset_t300", "5 min"),
    ("reset_t1800", "30 min"),
    ("reset_t7200", "2 h"),
    ("", "whole segment"),
)


def plot_memory_truncation_figure(
    diagnostic_results: Sequence[Mapping[str, Any]],
    full_results: Sequence[Mapping[str, Any]],
    out_png: Path,
    out_pdf: Path,
    *,
    producer: str = "P_slow_seed1",
    endpoints: Sequence[str] = ("count", "participation", "continuous"),
) -> dict[str, Any]:
    """Auxiliary diagnostic: how much of the past the frozen state actually uses.

    CC 6 is explicit that the useful history scale is read off the future-block
    curve against real horizons, **not** from "which reset first stops being
    significant".  This figure is therefore labelled as a diagnostic and carries
    no verdict of its own.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    fig, axes = plt.subplots(1, len(endpoints), figsize=(3.5 * len(endpoints), 3.3))
    axes = np.atleast_1d(axes)
    payload: dict[str, Any] = {}
    xs = np.arange(len(RESET_ORDER), dtype=float)
    horizons = (300.0, 1800.0, 7200.0)
    shades = {300.0: "#c1571c", 1800.0: "#8a5a2b", 7200.0: "#4a4a4a"}
    for ax, endpoint in zip(axes, endpoints):
        ax.axhline(0.0, color="#333333", lw=0.9, zorder=1)
        for horizon in horizons:
            med = []
            for label, _pretty in RESET_ORDER:
                source = full_results if label == "" else diagnostic_results
                pattern = (f"B+S({producer})" if label == ""
                           else f"B+S({producer}_{label})")
                points = [p for p in collect_curve(source, pattern, endpoint)
                          if p.horizon == horizon]
                v = points[0].values if points else np.zeros(0)
                med.append(float(np.median(v)) if v.size else np.nan)
                payload.setdefault(endpoint, {}).setdefault(f"{int(horizon)}s", {})[
                    label or "full"
                ] = {"median_gain": med[-1], "n_subjects": int(v.size)}
            ax.plot(xs, med, marker="o", ms=4, color=shades[horizon],
                    label=HORIZON_LABELS.get(horizon, f"{horizon / 60:.0f} min"))
        ax.set_xticks(xs)
        ax.set_xticklabels([p for _l, p in RESET_ORDER], fontsize=6.5, rotation=35,
                           ha="right")
        ax.set_xlim(xs[0] - 0.4, xs[-1] + 0.4)
        ax.set_title(PANEL_TITLES.get(endpoint, endpoint))
        ax.set_ylabel("gain over baseline (nats per unit)")
        ax.set_xlabel("how much past the state is allowed to keep")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, -0.10), title="block starts")
    fig.suptitle("Diagnostic: how much of the past the state actually uses", y=1.02,
                 fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    return payload
