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
    "memoryless": "no carried state",
}
ARM_COLORS = {
    "P_local": "#1b7f79",     # teal
    "P_slow": "#c1571c",      # burnt orange
    "shift": "#8a8a8a",       # grey
    "memoryless": "#5b5ea6",  # muted indigo
}

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
    ax.set_xticks(xs)
    ax.set_xticklabels([PREFIX_LABELS[o] for o in outcomes], fontsize=6.5,
                       linespacing=1.35)
    ax.set_xlim(xs[0] - 0.45, xs[-1] + 0.45)
    ax.set_title("Events that started the same way")
    ax.set_ylabel("gain over the start-only model\n(nats per unit)")
    ax.set_xlabel("what happened next")
    return out
